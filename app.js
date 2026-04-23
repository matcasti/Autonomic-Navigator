'use strict';

// ============================================================
//  GLOBAL STATE
// ============================================================
const state = {
  recordings: [],
  activeIdx: -1,
  analysisCache: {},
  playback: { active: false, frame: 0, speed: 1, raf: null },
  visMode: 'ans',
  threeScene: null,
  currentAnalysis: null
};

// ============================================================
//  UTILITIES
// ============================================================
function mean(arr) {
  return arr.reduce((a, b) => a + b, 0) / arr.length;
}

function std(arr) {
  const m = mean(arr);
  return Math.sqrt(arr.reduce((s, x) => s + (x - m) ** 2, 0) / (arr.length - 1));
}

function nextPow2(n) {
  let p = 1;
  while (p < n) p <<= 1;
  return p;
}

function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

function fmt2(v) { return isNaN(v) ? '—' : v.toFixed(2); }
function fmt0(v) { return isNaN(v) ? '—' : Math.round(v).toString(); }
function fmtMs(v_s) { return isNaN(v_s) ? '—' : (v_s * 1000).toFixed(1) + ' ms'; }

// ============================================================
//  FFT (Cooley-Tukey, power of 2)
// ============================================================
function fft(re, im) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      [re[i], re[j]] = [re[j], re[i]];
      [im[i], im[j]] = [im[j], im[i]];
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = -2 * Math.PI / len;
    const wRe = Math.cos(ang), wIm = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let curRe = 1, curIm = 0;
      for (let j = 0; j < len >> 1; j++) {
        const uRe = re[i + j], uIm = im[i + j];
        const vRe = re[i + j + (len>>1)] * curRe - im[i + j + (len>>1)] * curIm;
        const vIm = re[i + j + (len>>1)] * curIm + im[i + j + (len>>1)] * curRe;
        re[i + j] = uRe + vRe; im[i + j] = uIm + vIm;
        re[i + j + (len>>1)] = uRe - vRe; im[i + j + (len>>1)] = uIm - vIm;
        const nr = curRe * wRe - curIm * wIm;
        const ni = curRe * wIm + curIm * wRe;
        curRe = nr; curIm = ni;
      }
    }
  }
}

// ============================================================
//  FILE PARSING
// ============================================================
function parseFile(text, filename) {
  const ext = filename.split('.').pop().toLowerCase();
  let rr = [];
  const lines = text.trim().split(/\r?\n/).filter(l => l.trim());

  if (ext === 'csv' || filename.includes(',')) {
    const header = lines[0].toLowerCase();
    const seps = /[,;\t ]+/;
    const cols = header.split(seps);
    const rrIdx = cols.findIndex(c => /rr|interval|ibi|nn|r[-_]r/i.test(c));
    const start = rrIdx >= 0 ? 1 : 0;
    const idx = rrIdx >= 0 ? rrIdx : -1;

    for (let i = start; i < lines.length; i++) {
      const parts = lines[i].trim().split(seps);
      let val;
      if (idx >= 0 && idx < parts.length) {
        val = parseFloat(parts[idx]);
      } else {
        // try last numeric column
        for (let k = parts.length - 1; k >= 0; k--) {
          const n = parseFloat(parts[k]);
          if (!isNaN(n) && n > 0) { val = n; break; }
        }
      }
      if (!isNaN(val) && val > 0) rr.push(val);
    }
  } else {
    // TXT: try one per line or space-separated
    for (const line of lines) {
      const parts = line.trim().split(/[\s,;]+/);
      for (const p of parts) {
        const n = parseFloat(p);
        if (!isNaN(n) && n > 0) rr.push(n);
      }
    }
  }

  if (rr.length === 0) return null;

  // Sort to find median for unit detection
  const sorted = [...rr].sort((a, b) => a - b);
  const median = sorted[Math.floor(sorted.length / 2)];

  // Auto-detect: if median > 5, assume ms
  if (median > 5) {
    rr = rr.map(x => x / 1000);
  }

  // Filter physiologically valid RR intervals (200ms–3000ms)
  rr = rr.filter(v => v >= 0.2 && v <= 3.0);

  return rr.length >= 10 ? rr : null;
}

// ============================================================
//  HRV TIME DOMAIN ANALYSIS
// ============================================================
function computeTimeDomain(rr) {
  const n = rr.length;
  const m = mean(rr);
  const sdnn = std(rr);
  let sumSq = 0, nn50 = 0;
  for (let i = 1; i < n; i++) {
    const d = rr[i] - rr[i - 1];
    sumSq += d * d;
    if (Math.abs(d) > 0.05) nn50++;
  }
  const rmssd = Math.sqrt(sumSq / (n - 1));
  const pnn50 = nn50 / (n - 1) * 100;

  // Poincaré
  const sd1 = rmssd / Math.sqrt(2);
  const sd2 = Math.sqrt(2 * sdnn * sdnn - sd1 * sd1);

  return {
    n, meanRR: m, sdnn, rmssd, nn50, pnn50,
    meanHR: 60 / m, sd1, sd2, sd1sd2: sd1 / sd2
  };
}

// ============================================================
//  SPECTRAL ANALYSIS (Lomb-Scargle proxy via resampled FFT)
// ============================================================
function computeSpectral(rr) {
  const n = rr.length;
  // Build cumulative time vector
  const t = new Float64Array(n + 1);
  for (let i = 0; i < n; i++) t[i + 1] = t[i] + rr[i];
  const T = t[n];

  const fs = 4.0;
  const nSamples = Math.max(64, Math.floor(T * fs));

  // Linear interpolation onto uniform grid
  const rrU = new Float64Array(nSamples);
  let j = 0;
  for (let i = 0; i < nSamples; i++) {
    const ti = i / fs;
    while (j < n - 1 && t[j + 1] <= ti) j++;
    const frac = (ti - t[j]) / Math.max(1e-9, t[j + 1] - t[j]);
    const jj = Math.min(j, n - 1);
    const jj2 = Math.min(j + 1, n - 1);
    rrU[i] = rr[jj] * (1 - frac) + rr[jj2] * frac;
  }

  // Remove mean, Hanning window
  const mu = mean(Array.from(rrU));
  const N = nextPow2(nSamples);
  const re = new Float64Array(N);
  const im = new Float64Array(N);
  for (let i = 0; i < nSamples; i++) {
    const w = 0.5 * (1 - Math.cos(2 * Math.PI * i / (nSamples - 1)));
    re[i] = (rrU[i] - mu) * w;
  }

  fft(re, im);

  // One-sided PSD in ms²/Hz (convert s to ms)
  const df = fs / N;
  const freqs = [], psd = [];
  const half = Math.floor(N / 2);
  for (let i = 0; i <= half; i++) {
    freqs.push(i * df);
    const mag2 = re[i] * re[i] + im[i] * im[i];
    const p = mag2 / (nSamples * fs) * 1e6; // s² → ms²
    psd.push(i === 0 || i === half ? p : 2 * p);
  }

  function bp(f1, f2) {
    let pow = 0;
    for (let i = 0; i < freqs.length; i++) {
      if (freqs[i] >= f1 && freqs[i] <= f2 && i > 0) {
        pow += psd[i] * df;
      }
    }
    return pow;
  }

  function peakFreq(f1, f2) {
    let maxP = -1, mf = (f1 + f2) / 2;
    for (let i = 0; i < freqs.length; i++) {
      if (freqs[i] >= f1 && freqs[i] <= f2 && psd[i] > maxP) {
        maxP = psd[i]; mf = freqs[i];
      }
    }
    return mf;
  }

  const vlf = bp(0.003, 0.04);
  const lf  = bp(0.04,  0.15);
  const hf  = bp(0.15,  0.40);
  const tp  = bp(0.003, 0.40);
  const lf_nu = lf / (lf + hf + 1e-10) * 100;
  const hf_nu = hf / (lf + hf + 1e-10) * 100;

  return {
    freqs, psd, vlf, lf, hf, tp,
    lfhf: lf / (hf + 1e-10),
    lf_nu, hf_nu,
    lfPeak: peakFreq(0.04, 0.15),
    hfPeak: peakFreq(0.15, 0.40)
  };
}

// ============================================================
//  PARAMETER ESTIMATION
// ============================================================
function estimateParams(rr, td, spec) {
  const mu0 = td.meanRR;
  const rho = clamp(td.sdnn / mu0, 0.05, 0.5);

  // Power-weighted spectral centroid within a band. More robust than
  // peak-picking because it uses the full band shape rather than a single bin.
  function bandCentroid(f1, f2, fallback) {
    let num = 0, den = 0;
    for (let i = 1; i < spec.freqs.length; i++) {
      const f = spec.freqs[i];
      if (f >= f1 && f <= f2) {
        num += f * spec.psd[i];
        den += spec.psd[i];
      }
    }
    return den > 0 ? num / den : fallback;
  }

  // Decay rates estimated from the data. For an OU process the autocorrelation
  // rolls off at a characteristic frequency f_c = a / (2π), so the band
  // centroid of each autonomic band gives a = 2π · f_centroid. Clamped to
  // physiologically plausible ranges to keep the EKF numerically well-posed.
  const fHF = bandCentroid(0.15, 0.40, 0.25);
  const fLF = bandCentroid(0.04, 0.15, 0.10);
  const ap = clamp(2 * Math.PI * fHF, 0.5, 5.0);
  const as = clamp(2 * Math.PI * fLF, 0.1, 1.5);

  // Spectral decomposition of variance
  const varDelta = rho * rho;
  const totalSpecPow = spec.lf + spec.hf + 1e-10;
  const hfFrac = spec.hf / totalSpecPow;
  const lfFrac = spec.lf / totalSpecPow;

  // sigma²_p = hfFrac * varDelta * 2 * ap
  const sigma_p = Math.sqrt(Math.max(1e-6, hfFrac * varDelta * 2 * ap));
  const sigma_s = Math.sqrt(Math.max(1e-6, lfFrac * varDelta * 2 * as));
  const kappa   = mu0 / (rho * rho);

  return { ap, as, sigma_p, sigma_s, mu0, rho, kappa };
}

// ============================================================
//  EKF (2D: [p, s] state)
// ============================================================
class ANS_EKF {
  constructor(params) {
    this.ap = params.ap;
    this.as = params.as;
    this.sp = params.sigma_p;
    this.ss = params.sigma_s;
    this.mu0 = params.mu0;
    this.kap = params.kappa;

    // State [p, s]
    this.m = [0, 0];
    // Covariance (diagonal)
    const vp = this.sp * this.sp / (2 * this.ap);
    const vs = this.ss * this.ss / (2 * this.as);
    this.P = [[vp, 0], [0, vs]];
  }

  step(tau) {
    // Prediction: exact OU transition
    const ep = Math.exp(-this.ap * tau);
    const es = Math.exp(-this.as * tau);
    const mp = [this.m[0] * ep, this.m[1] * es];

    const vp = this.sp * this.sp / (2 * this.ap) * (1 - Math.exp(-2 * this.ap * tau));
    const vs = this.ss * this.ss / (2 * this.as) * (1 - Math.exp(-2 * this.as * tau));

    const Pp = [
      [this.P[0][0] * ep * ep + vp, this.P[0][1] * ep * es],
      [this.P[1][0] * ep * es,      this.P[1][1] * es * es + vs]
    ];

    // Predicted observation: h(x) = mu0 * exp(p - s)
    const delta_p = mp[1] - mp[0];
    const mu_p    = this.mu0 * Math.exp(-delta_p);
    const muCl    = clamp(mu_p, 0.2, 3.0);

    // IG observation variance: R = mu³/kappa
    const R = muCl * muCl * muCl / this.kap;

    // Jacobian: H = [mu_p, -mu_p] (dh/dp, dh/ds)
    const Hp = muCl, Hs = -muCl;

    // Innovation variance S = H P H' + R
    const S = Hp * (Hp * Pp[0][0] + Hs * Pp[1][0]) +
              Hs * (Hp * Pp[0][1] + Hs * Pp[1][1]) + R;

    // Kalman gain K = P H' / S
    const Kp = (Hp * Pp[0][0] + Hs * Pp[0][1]) / S;
    const Ks = (Hp * Pp[1][0] + Hs * Pp[1][1]) / S;

    // Innovation
    const innov = clamp(tau - muCl, -1.5, 1.5);

    // Update
    this.m = [mp[0] + Kp * innov, mp[1] + Ks * innov];

    // Joseph-form covariance update
    const KSKp = Kp * S * Kp, KSKs = Ks * S * Ks, KSKps = Kp * S * Ks;
    this.P = [
      [Pp[0][0] - KSKp,  Pp[0][1] - KSKps],
      [Pp[1][0] - KSKps, Pp[1][1] - KSKs]
    ];

    const delta = this.m[1] - this.m[0];
    return {
      p: this.m[0], s: this.m[1], delta,
      mu: this.mu0 * Math.exp(-delta),
      p_std: Math.sqrt(Math.abs(this.P[0][0])),
      s_std: Math.sqrt(Math.abs(this.P[1][1]))
    };
  }
}

// ============================================================
//  FULL ANALYSIS PIPELINE
// ============================================================
function analyzeRR(rr) {
  const td     = computeTimeDomain(rr);
  const spec   = computeSpectral(rr);
  const params = estimateParams(rr, td, spec);

  // Run EKF
  const ekf = new ANS_EKF(params);
  const filter = [];
  let cumTime = 0;
  for (const tau of rr) {
    const res = ekf.step(tau);
    cumTime += tau;
    filter.push({ t: cumTime, rr: tau, ...res });
  }

  // Build cumulative time for tachogram
  const times = [];
  let ct = 0;
  for (const v of rr) { ct += v; times.push(ct); }

  return { rr, times, td, spec, params, filter };
}

// ============================================================
//  THREE.JS ANS SCENE
// ============================================================
let renderer, scene, camera, animId;
let heartMesh, paraNode, sympNode, neuralMesh, particles, particlePos;
let connLines = [];
let animTime = 0;
let lastBeatTime = 0;

function initThreeScene() {
  const container = document.getElementById('vis-panel');
  const canvas = document.getElementById('three-canvas');

  renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(container.clientWidth, container.clientHeight);
  renderer.setClearColor(0x020912, 1);
  renderer.shadowMap.enabled = false;

  scene = new THREE.Scene();
  scene.fog = new THREE.FogExp2(0x020912, 0.025);

  camera = new THREE.PerspectiveCamera(55, container.clientWidth / container.clientHeight, 0.1, 100);
  camera.position.set(0, 2, 14);
  camera.lookAt(0, 0, 0);

  // Lights
  const ambient = new THREE.AmbientLight(0x0a1020, 2);
  scene.add(ambient);
  const paraLight = new THREE.PointLight(0x00e5ff, 3, 15);
  paraLight.position.set(-5, 2, 0);
  scene.add(paraLight);
  const sympLight = new THREE.PointLight(0xff6d00, 3, 15);
  sympLight.position.set(5, 2, 0);
  scene.add(sympLight);
  const centerLight = new THREE.PointLight(0xff2244, 4, 8);
  centerLight.position.set(0, 0, 0);
  scene.add(centerLight);

  // Star field
  const starGeo = new THREE.BufferGeometry();
  const starPos = new Float32Array(1500 * 3);
  for (let i = 0; i < 1500 * 3; i++) starPos[i] = (Math.random() - 0.5) * 80;
  starGeo.setAttribute('position', new THREE.BufferAttribute(starPos, 3));
  const starMat = new THREE.PointsMaterial({ color: 0x4488aa, size: 0.08, transparent: true, opacity: 0.6 });
  scene.add(new THREE.Points(starGeo, starMat));

  // Neural mesh (icosahedron wireframe)
  const meshGeo = new THREE.IcosahedronGeometry(4.5, 2);
  const meshMat = new THREE.MeshBasicMaterial({ color: 0x002244, wireframe: true, transparent: true, opacity: 0.18 });
  neuralMesh = new THREE.Mesh(meshGeo, meshMat);
  scene.add(neuralMesh);

  // Inner neural mesh
  const meshGeo2 = new THREE.IcosahedronGeometry(3.0, 1);
  const meshMat2 = new THREE.MeshBasicMaterial({ color: 0x001133, wireframe: true, transparent: true, opacity: 0.25 });
  const innerMesh = new THREE.Mesh(meshGeo2, meshMat2);
  scene.add(innerMesh);

  // Heart node (SA node)
  function makeGlowSphere(radius, color, emissiveColor, emissiveIntensity) {
    const geo = new THREE.SphereGeometry(radius, 32, 32);
    const mat = new THREE.MeshStandardMaterial({
      color, emissive: emissiveColor, emissiveIntensity,
      metalness: 0.3, roughness: 0.4, transparent: true, opacity: 0.95
    });
    return new THREE.Mesh(geo, mat);
  }

  heartMesh = makeGlowSphere(0.7, 0xffffff, 0xff2244, 3);
  scene.add(heartMesh);

  // Heart glow layers
  for (let i = 1; i <= 3; i++) {
    const g = new THREE.SphereGeometry(0.7 + i * 0.25, 16, 16);
    const m = new THREE.MeshBasicMaterial({
      color: 0xff2244, transparent: true, opacity: 0.06 / i,
      blending: THREE.AdditiveBlending, side: THREE.BackSide
    });
    scene.add(new THREE.Mesh(g, m));
  }

  // Parasympathetic node
  paraNode = makeGlowSphere(0.55, 0x00e5ff, 0x00e5ff, 2.5);
  paraNode.position.set(-5, 1.5, 0);
  scene.add(paraNode);

  // Para glow layers
  for (let i = 1; i <= 3; i++) {
    const g = new THREE.SphereGeometry(0.55 + i * 0.22, 16, 16);
    const m = new THREE.MeshBasicMaterial({
      color: 0x00e5ff, transparent: true, opacity: 0.07 / i,
      blending: THREE.AdditiveBlending, side: THREE.BackSide
    });
    const s = new THREE.Mesh(g, m);
    s.position.copy(paraNode.position);
    scene.add(s);
    paraNode.userData['glow' + i] = s;
  }

  // Sympathetic node
  sympNode = makeGlowSphere(0.55, 0xff6d00, 0xff6d00, 2.5);
  sympNode.position.set(5, 1.5, 0);
  scene.add(sympNode);

  // Symp glow layers
  for (let i = 1; i <= 3; i++) {
    const g = new THREE.SphereGeometry(0.55 + i * 0.22, 16, 16);
    const m = new THREE.MeshBasicMaterial({
      color: 0xff6d00, transparent: true, opacity: 0.07 / i,
      blending: THREE.AdditiveBlending, side: THREE.BackSide
    });
    const s = new THREE.Mesh(g, m);
    s.position.copy(sympNode.position);
    scene.add(s);
    sympNode.userData['glow' + i] = s;
  }

  // Connection tubes
  function makeTube(from, to, color) {
    const points = [from.clone(), new THREE.Vector3(0, 0, 0)];
    const geo = new THREE.BufferGeometry().setFromPoints(points);
    const mat = new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.3 });
    const line = new THREE.Line(geo, mat);
    scene.add(line);
    return { line, from, to, mat };
  }

  connLines.push(makeTube(new THREE.Vector3(-5, 1.5, 0), new THREE.Vector3(0, 0, 0), 0x00e5ff));
  connLines.push(makeTube(new THREE.Vector3(5, 1.5, 0), new THREE.Vector3(0, 0, 0), 0xff6d00));

  // Particle system
  const N = 600;
  particlePos = new Float32Array(N * 3);
  const colors = new Float32Array(N * 3);
  const pData = [];

  for (let i = 0; i < N; i++) {
    const isPara = i < N / 2;
    const src = isPara ? new THREE.Vector3(-5, 1.5, 0) : new THREE.Vector3(5, 1.5, 0);
    const t = Math.random();
    const offset = new THREE.Vector3((Math.random() - 0.5) * 0.4, (Math.random() - 0.5) * 0.4, (Math.random() - 0.5) * 0.4);
    pData.push({ isPara, t, speed: 0.002 + Math.random() * 0.003, offset });

    const cx = src.x * (1 - t);
    particlePos[i * 3]     = cx + offset.x;
    particlePos[i * 3 + 1] = src.y * (1 - t) + offset.y;
    particlePos[i * 3 + 2] = offset.z;

    if (isPara) { colors[i*3]=0; colors[i*3+1]=0.9; colors[i*3+2]=1; }
    else        { colors[i*3]=1; colors[i*3+1]=0.43; colors[i*3+2]=0; }
  }

  const pGeo = new THREE.BufferGeometry();
  pGeo.setAttribute('position', new THREE.BufferAttribute(particlePos, 3));
  pGeo.setAttribute('color',    new THREE.BufferAttribute(colors, 3));
  const pMat = new THREE.PointsMaterial({
    vertexColors: true, size: 0.06, transparent: true, opacity: 0.75,
    blending: THREE.AdditiveBlending, sizeAttenuation: true
  });
  particles = new THREE.Points(pGeo, pMat);
  particles.userData.pData = pData;
  scene.add(particles);

  // Orbit ring for para node
  const ringGeo = new THREE.TorusGeometry(5.5, 0.02, 8, 80);
  const ringMat = new THREE.MeshBasicMaterial({ color: 0x001e2e, transparent: true, opacity: 0.5 });
  const ring = new THREE.Mesh(ringGeo, ringMat);
  ring.rotation.x = Math.PI / 2;
  ring.rotation.z = 0.3;
  scene.add(ring);

  // Camera controls
  let dragging = false, prevX = 0, prevY = 0;
  let theta = 0.3, phi = 1.1, radius = 14;

  renderer.domElement.addEventListener('mousedown', e => { dragging = true; prevX = e.clientX; prevY = e.clientY; });
  renderer.domElement.addEventListener('touchstart', e => { dragging = true; prevX = e.touches[0].clientX; prevY = e.touches[0].clientY; });
  window.addEventListener('mouseup', () => dragging = false);
  window.addEventListener('touchend', () => dragging = false);

  renderer.domElement.addEventListener('mousemove', e => {
    if (!dragging) return;
    theta += (e.clientX - prevX) * 0.008;
    phi    = clamp(phi - (e.clientY - prevY) * 0.008, 0.3, 2.0);
    prevX = e.clientX; prevY = e.clientY;
  });
  renderer.domElement.addEventListener('touchmove', e => {
    e.preventDefault();
    if (!dragging) return;
    theta += (e.touches[0].clientX - prevX) * 0.008;
    phi    = clamp(phi - (e.touches[0].clientY - prevY) * 0.008, 0.3, 2.0);
    prevX = e.touches[0].clientX; prevY = e.touches[0].clientY;
  }, { passive: false });
  renderer.domElement.addEventListener('wheel', e => {
    radius = clamp(radius + e.deltaY * 0.015, 6, 22);
  });

  scene.userData.cameraState = { theta, phi, radius };
  scene.userData.getCam = () => ({ theta, phi, radius });
  scene.userData.setCam = (t, p, r) => { theta = t; phi = p; radius = r; };

  function animateScene() {
    animId = requestAnimationFrame(animateScene);
    animTime += 0.016;

    const cs = scene.userData.getCam();
    const autoTheta = cs.theta + (dragging ? 0 : 0.003);
    scene.userData.setCam(autoTheta, cs.phi, cs.radius);
    const t = autoTheta, p = cs.phi, r = cs.radius;

    camera.position.set(r * Math.sin(p) * Math.cos(t), r * Math.cos(p), r * Math.sin(p) * Math.sin(t));
    camera.lookAt(0, 0.5, 0);

    // Get current ANS state
    const an = state.currentAnalysis;
    let curP = 0, curS = 0, curDelta = 0;
    if (an && state.playback.frame < an.filter.length) {
      const f = an.filter[state.playback.frame] || an.filter[an.filter.length - 1];
      curP = f.p; curS = f.s; curDelta = f.delta;
    }

    // Animate heart pulse
    const rrInterval = an ? (an.td.meanRR || 0.9) : 0.9;
    const beatPhase = (animTime % rrInterval) / rrInterval;
    const heartScale = 1 + 0.22 * Math.exp(-8 * beatPhase) * Math.sin(10 * beatPhase);
    heartMesh.scale.setScalar(heartScale);
    heartMesh.material.emissiveIntensity = 2.5 + 1.5 * heartScale;

    // Flash beat
    if (beatPhase < 0.05) {
      const flash = document.getElementById('beat-flash');
      if (flash) { flash.style.opacity = '1'; setTimeout(() => { if(flash) flash.style.opacity = '0'; }, 60); }
    }

    // Animate parasympathetic node
    const paraSize = 0.55 + clamp(Math.abs(curP) * 0.4, 0, 0.5);
    const paraAngle = animTime * (0.3 + Math.abs(curP) * 0.1);
    paraNode.position.set(-5 * Math.cos(paraAngle * 0.3), 1.5 + Math.sin(animTime * 0.8) * 0.3, -5 * Math.sin(paraAngle * 0.3));
    paraNode.scale.setScalar(paraSize);
    paraNode.material.emissiveIntensity = 1.5 + 2 * Math.max(0, curP);

    // Update para glow positions
    for (let i = 1; i <= 3; i++) {
      const g = paraNode.userData['glow' + i];
      if (g) g.position.copy(paraNode.position);
    }

    // Animate sympathetic node
    const sympSize = 0.55 + clamp(Math.abs(curS) * 0.4, 0, 0.5);
    const sympAngle = -animTime * (0.2 + Math.abs(curS) * 0.08);
    sympNode.position.set(5 * Math.cos(sympAngle * 0.25), 1.5 + Math.sin(animTime * 0.5 + 1) * 0.3, 5 * Math.sin(sympAngle * 0.25));
    sympNode.scale.setScalar(sympSize);
    sympNode.material.emissiveIntensity = 1.5 + 2 * Math.max(0, curS);

    for (let i = 1; i <= 3; i++) {
      const g = sympNode.userData['glow' + i];
      if (g) g.position.copy(sympNode.position);
    }

    // Update connection lines
    if (connLines[0]) {
      const pts0 = [paraNode.position.clone(), new THREE.Vector3(0, 0, 0)];
      connLines[0].line.geometry.setFromPoints(pts0);
      connLines[0].mat.opacity = 0.15 + 0.25 * Math.max(0, curP);
    }
    if (connLines[1]) {
      const pts1 = [sympNode.position.clone(), new THREE.Vector3(0, 0, 0)];
      connLines[1].line.geometry.setFromPoints(pts1);
      connLines[1].mat.opacity = 0.15 + 0.25 * Math.max(0, curS);
    }

    // Animate particles
    const pData = particles.userData.pData;
    const paraSpeed = 1 + clamp(Math.abs(curP) * 2, 0, 3);
    const sympSpeed = 1 + clamp(Math.abs(curS) * 2, 0, 3);
    for (let i = 0; i < pData.length; i++) {
      const pd = pData[i];
      pd.t += pd.speed * (pd.isPara ? paraSpeed : sympSpeed);
      if (pd.t > 1) pd.t = 0;

      const src = pd.isPara ? paraNode.position : sympNode.position;
      const tt = pd.t;
      particlePos[i * 3]     = src.x * (1 - tt) + pd.offset.x;
      particlePos[i * 3 + 1] = src.y * (1 - tt) + pd.offset.y;
      particlePos[i * 3 + 2] = src.z * (1 - tt) + pd.offset.z;
    }
    particles.geometry.attributes.position.needsUpdate = true;

    // Neural mesh rotation
    neuralMesh.rotation.y = animTime * 0.06;
    neuralMesh.rotation.x = animTime * 0.02;

    // Update vis labels
    document.getElementById('vis-para-label').textContent = `VAGAL p̂ = ${curP.toFixed(3)}`;
    document.getElementById('vis-symp-label').textContent = `ADREN ŝ = ${curS.toFixed(3)}`;

    renderer.render(scene, camera);
  }

  animateScene();

  // Resize observer
  const ro = new ResizeObserver(() => {
    const w = container.clientWidth, h = container.clientHeight;
    renderer.setSize(w, h);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  });
  ro.observe(container);
}

// ============================================================
//  PHASE SPACE CANVAS
// ============================================================
function drawPhaseSpace(analysis) {
  const canvas = document.getElementById('phase-canvas');
  const panel  = document.getElementById('vis-panel');
  canvas.width  = panel.clientWidth  * window.devicePixelRatio;
  canvas.height = panel.clientHeight * window.devicePixelRatio;
  canvas.style.width  = panel.clientWidth  + 'px';
  canvas.style.height = panel.clientHeight + 'px';

  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  const pad = 60 * window.devicePixelRatio;

  ctx.fillStyle = '#020912';
  ctx.fillRect(0, 0, W, H);

  const filt = analysis.filter;
  if (!filt || filt.length < 2) return;

  const ps = filt.map(f => f.p), ss = filt.map(f => f.s);
  const minP = Math.min(...ps), maxP = Math.max(...ps);
  const minS = Math.min(...ss), maxS = Math.max(...ss);
  const rangeP = Math.max(0.2, maxP - minP), rangeS = Math.max(0.2, maxS - minS);

  const sx = x => pad + (x - minP) / rangeP * (W - 2 * pad);
  const sy = y => H - pad - (y - minS) / rangeS * (H - 2 * pad);

  // Grid
  ctx.strokeStyle = 'rgba(0,100,150,0.2)';
  ctx.lineWidth = 1;
  for (let i = 0; i <= 5; i++) {
    const xg = pad + i / 5 * (W - 2 * pad);
    const yg = pad + i / 5 * (H - 2 * pad);
    ctx.beginPath(); ctx.moveTo(xg, pad); ctx.lineTo(xg, H - pad); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(pad, yg); ctx.lineTo(W - pad, yg); ctx.stroke();
  }

  // Axes labels
  ctx.font = `${10 * window.devicePixelRatio}px JetBrains Mono, monospace`;
  ctx.fillStyle = '#4a6080';
  ctx.textAlign = 'center';
  ctx.fillText('p̂(t) VAGAL →', W / 2, H - 10 * window.devicePixelRatio);
  ctx.save(); ctx.translate(15 * window.devicePixelRatio, H / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText('ŝ(t) ADRENERGIC →', 0, 0);
  ctx.restore();

  // Diagonal (Δ=0 line)
  ctx.strokeStyle = 'rgba(0,230,118,0.2)';
  ctx.setLineDash([5, 5]);
  const pMid = (minP + maxP) / 2, sMid = (minS + maxS) / 2;
  ctx.beginPath();
  ctx.moveTo(sx(minP), sy(minP + sMid - pMid));
  ctx.lineTo(sx(maxP), sy(maxP + sMid - pMid));
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = 'rgba(0,230,118,0.5)';
  ctx.font = `${8 * window.devicePixelRatio}px JetBrains Mono, monospace`;
  ctx.fillText('Δ=0', sx(maxP) - 20, sy(maxP + sMid - pMid) - 8);

  // Trajectory with color gradient
  const N = filt.length;
  for (let i = 1; i < N; i++) {
    const t = i / N;
    const f0 = filt[i - 1], f1 = filt[i];
    const x0 = sx(f0.p), y0 = sy(f0.s);
    const x1 = sx(f1.p), y1 = sy(f1.s);

    // Color by delta
    const d = f1.delta;
    const r = Math.round(clamp(d > 0 ? 200 + d * 100 : 0, 0, 255));
    const g = Math.round(clamp(d < 0 ? 180 + Math.abs(d) * 100 : 50, 0, 255));
    const b = Math.round(clamp(d < 0 ? 255 : 50, 0, 255));
    ctx.strokeStyle = `rgba(${r},${g},${b},${0.3 + 0.5 * t})`;
    ctx.lineWidth = 1.5 * window.devicePixelRatio;
    ctx.beginPath();
    ctx.moveTo(x0, y0);
    ctx.lineTo(x1, y1);
    ctx.stroke();
  }

  // Current position
  const last = filt[filt.length - 1];
  ctx.beginPath();
  ctx.arc(sx(last.p), sy(last.s), 5 * window.devicePixelRatio, 0, Math.PI * 2);
  ctx.fillStyle = '#00e676';
  ctx.fill();
  ctx.strokeStyle = '#00e676';
  ctx.lineWidth = 2;
  ctx.stroke();

  // Title
  ctx.fillStyle = 'rgba(0,229,255,0.6)';
  ctx.font = `${11 * window.devicePixelRatio}px Orbitron, monospace`;
  ctx.textAlign = 'left';
  ctx.fillText('AUTONOMIC PHASE SPACE', pad, 25 * window.devicePixelRatio);
  ctx.font = `${9 * window.devicePixelRatio}px JetBrains Mono, monospace`;
  ctx.fillStyle = '#4a6080';
  ctx.fillText(`N=${N} beats · ${(filt[N-1].t/60).toFixed(1)} min`, pad, 42 * window.devicePixelRatio);
}

// ============================================================
//  CANVAS CHARTS
// ============================================================
function dpr() { return Math.min(window.devicePixelRatio || 1, 2); }

function sizeCanvas(canvas) {
  const D = dpr();
  const w = canvas.parentElement.clientWidth;
  const h = parseInt(canvas.getAttribute('height')) || 160;
  if (!w || w < 20) return; // guard against mid-reflow zero reads
  canvas.width  = Math.round(w * D);
  canvas.height = Math.round(h * D);
  canvas.style.width  = w + 'px';
  canvas.style.height = h + 'px';
}

function drawBranchChart(analysis) {
  const canvas = document.getElementById('branches-canvas');
  sizeCanvas(canvas);
  const ctx = canvas.getContext('2d');
  const D = dpr();
  const W = canvas.width, H = canvas.height;
  const padL = 50 * D, padR = 20 * D, padT = 16 * D, padB = 36 * D;
  const IW = W - padL - padR, IH = H - padT - padB;

  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = 'rgba(6,15,30,0.5)';
  ctx.fillRect(0, 0, W, H);

  const filt = analysis.filter;
  if (!filt || filt.length < 2) return;

  const N = filt.length;
  const tEnd = filt[N - 1].t;
  const ps = filt.map(f => f.p), ss = filt.map(f => f.s), ds = filt.map(f => f.delta);
  const allVals = [...ps, ...ss, ...ds];
  const minV = Math.min(...allVals), maxV = Math.max(...allVals);
  const rng = Math.max(0.5, maxV - minV);
  const pad_v = rng * 0.1;

  const sx = t => padL + t / tEnd * IW;
  const sy = v => padT + IH - (v - minV + pad_v) / (rng + 2 * pad_v) * IH;

  // Zero line
  const y0 = sy(0);
  ctx.strokeStyle = 'rgba(255,255,255,0.06)';
  ctx.lineWidth = 1;
  ctx.setLineDash([4, 4]);
  ctx.beginPath(); ctx.moveTo(padL, y0); ctx.lineTo(W - padR, y0); ctx.stroke();
  ctx.setLineDash([]);

  // Grid
  ctx.strokeStyle = 'rgba(0,100,150,0.1)';
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = padT + i / 4 * IH;
    ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(W - padR, y); ctx.stroke();
  }

  // Draw area under p(t)
  ctx.beginPath();
  ctx.moveTo(sx(filt[0].t), sy(0));
  for (const f of filt) ctx.lineTo(sx(f.t), sy(f.p));
  ctx.lineTo(sx(filt[N - 1].t), sy(0));
  ctx.closePath();
  ctx.fillStyle = 'rgba(0,229,255,0.06)';
  ctx.fill();

  // Draw p(t) line
  ctx.beginPath();
  for (let i = 0; i < N; i++) {
    const x = sx(filt[i].t), y = sy(filt[i].p);
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  }
  ctx.strokeStyle = '#00e5ff';
  ctx.lineWidth = 1.5 * D;
  ctx.stroke();

  // Draw area under s(t)
  ctx.beginPath();
  ctx.moveTo(sx(filt[0].t), sy(0));
  for (const f of filt) ctx.lineTo(sx(f.t), sy(f.s));
  ctx.lineTo(sx(filt[N - 1].t), sy(0));
  ctx.closePath();
  ctx.fillStyle = 'rgba(255,109,0,0.06)';
  ctx.fill();

  // Draw s(t) line
  ctx.beginPath();
  for (let i = 0; i < N; i++) {
    const x = sx(filt[i].t), y = sy(filt[i].s);
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  }
  ctx.strokeStyle = '#ff6d00';
  ctx.lineWidth = 1.5 * D;
  ctx.stroke();

  // Draw Δ(t) dashed
  ctx.beginPath();
  for (let i = 0; i < N; i++) {
    const x = sx(filt[i].t), y = sy(filt[i].delta);
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  }
  ctx.strokeStyle = '#00e676';
  ctx.lineWidth = 1.5 * D;
  ctx.setLineDash([4 * D, 3 * D]);
  ctx.stroke();
  ctx.setLineDash([]);

  // Playback cursor
  if (state.playback.frame < filt.length) {
    const f = filt[state.playback.frame];
    const xc = sx(f.t);
    ctx.strokeStyle = 'rgba(255,255,255,0.3)';
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(xc, padT); ctx.lineTo(xc, H - padB); ctx.stroke();
  }

  // Y-axis labels
  ctx.font = `${9 * D}px JetBrains Mono, monospace`;
  ctx.fillStyle = '#4a6080';
  ctx.textAlign = 'right';
  for (let i = 0; i <= 4; i++) {
    const v = minV - pad_v + (i / 4) * (rng + 2 * pad_v);
    ctx.fillText(v.toFixed(2), padL - 4 * D, padT + IH - i / 4 * IH + 3 * D);
  }

  // X-axis labels
  ctx.textAlign = 'center';
  const ticks = 6;
  for (let i = 0; i <= ticks; i++) {
    const t_val = i / ticks * tEnd;
    const x = padL + i / ticks * IW;
    ctx.fillText((t_val / 60).toFixed(1) + 'm', x, H - 8 * D);
  }

  ctx.font = `${8 * D}px Exo 2, sans-serif`;
  ctx.fillStyle = '#2a3848';
  ctx.textAlign = 'left';
  ctx.fillText('Time (min)', padL, H - 8 * D);
}

function drawTachogram(analysis) {
  const canvas = document.getElementById('tachogram-canvas');
  sizeCanvas(canvas);
  const ctx = canvas.getContext('2d');
  const D = dpr();
  const W = canvas.width, H = canvas.height;
  const padL = 50 * D, padR = 16 * D, padT = 16 * D, padB = 30 * D;
  const IW = W - padL - padR, IH = H - padT - padB;

  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = 'rgba(6,15,30,0.5)';
  ctx.fillRect(0, 0, W, H);

  const { rr, times } = analysis;
  const N = rr.length;
  if (N < 2) return;

  document.getElementById('tachogram-info').textContent = `${N} beats · ${(times[N-1]/60).toFixed(1)} min`;

  const minRR = Math.min(...rr) * 1000, maxRR = Math.max(...rr) * 1000;
  const rng = Math.max(50, maxRR - minRR);
  const tEnd = times[N - 1];

  const sx = t => padL + t / tEnd * IW;
  const sy = v => padT + IH - (v * 1000 - minRR + 10) / (rng + 20) * IH;

  // Grid
  ctx.strokeStyle = 'rgba(0,100,150,0.1)';
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = padT + i / 4 * IH;
    ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(W - padR, y); ctx.stroke();
  }

  // Area fill
  ctx.beginPath();
  ctx.moveTo(sx(times[0]), H - padB);
  for (let i = 0; i < N; i++) ctx.lineTo(sx(times[i]), sy(rr[i]));
  ctx.lineTo(sx(times[N-1]), H - padB);
  ctx.closePath();
  const grad = ctx.createLinearGradient(0, padT, 0, H - padB);
  grad.addColorStop(0, 'rgba(124,77,255,0.25)');
  grad.addColorStop(1, 'rgba(124,77,255,0.02)');
  ctx.fillStyle = grad;
  ctx.fill();

  // Line
  ctx.beginPath();
  for (let i = 0; i < N; i++) {
    i === 0 ? ctx.moveTo(sx(times[i]), sy(rr[i])) : ctx.lineTo(sx(times[i]), sy(rr[i]));
  }
  ctx.strokeStyle = '#b388ff';
  ctx.lineWidth = 1.2 * D;
  ctx.stroke();

  // Labels
  ctx.font = `${9 * D}px JetBrains Mono, monospace`;
  ctx.fillStyle = '#4a6080';
  ctx.textAlign = 'right';
  ctx.fillText(Math.round(maxRR) + 'ms', padL - 4, padT + 9 * D);
  ctx.fillText(Math.round(minRR) + 'ms', padL - 4, H - padB - 2 * D);

  ctx.textAlign = 'center';
  ctx.fillText('0m', padL, H - 6 * D);
  ctx.fillText(`${(tEnd/60).toFixed(1)}m`, W - padR, H - 6 * D);
}

function drawPoincare(analysis) {
  const canvas = document.getElementById('poincare-canvas');
  sizeCanvas(canvas);
  const ctx = canvas.getContext('2d');
  const D = dpr();
  const W = canvas.width, H = canvas.height;
  const pad = 40 * D, sz = Math.min(W, H) - 2 * pad;
  const ox = (W - sz) / 2, oy = (H - sz) / 2;

  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = 'rgba(6,15,30,0.5)';
  ctx.fillRect(0, 0, W, H);

  const { rr } = analysis;
  const N = rr.length;
  if (N < 3) return;

  const rms = rr.map(x => x * 1000);
  const minV = Math.min(...rms), maxV = Math.max(...rms);
  const rng = Math.max(100, maxV - minV);
  const ctr = (minV + maxV) / 2;

  const sc = v => ox + (v - (ctr - rng * 0.55)) / (rng * 1.1) * sz;

  // Identity line
  ctx.strokeStyle = 'rgba(100,120,140,0.2)';
  ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(ox, oy + sz); ctx.lineTo(ox + sz, oy); ctx.stroke();

  // SD1/SD2 ellipse
  const td = analysis.td;
  const sd1px = td.sd1 * 1000 / (rng * 1.1) * sz * 1.414;
  const sd2px = td.sd2 * 1000 / (rng * 1.1) * sz * 1.414;
  const cx = sc(ctr), cy = sc(ctr);

  ctx.save();
  ctx.translate(cx, H - (cy - oy) - oy);
  ctx.rotate(-Math.PI / 4);
  ctx.beginPath();
  ctx.ellipse(0, 0, sd2px, sd1px, 0, 0, Math.PI * 2);
  ctx.strokeStyle = 'rgba(255,214,0,0.25)';
  ctx.lineWidth = 1.5 * D;
  ctx.stroke();
  ctx.fillStyle = 'rgba(255,214,0,0.04)';
  ctx.fill();
  ctx.restore();

  // Points
  for (let i = 0; i < N - 1; i++) {
    const x = sc(rms[i]);
    const y = oy + sz - (sc(rms[i + 1]) - ox);
    ctx.beginPath();
    ctx.arc(x, y, 1.8 * D, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(124,77,255,0.55)';
    ctx.fill();
  }

  // SD labels
  ctx.font = `${8 * D}px JetBrains Mono, monospace`;
  ctx.fillStyle = '#ffd600';
  ctx.textAlign = 'center';
  ctx.fillText(`SD1: ${(td.sd1*1000).toFixed(1)}ms`, W / 2, H - 6 * D);

  document.getElementById('poincare-info').textContent = `SD1 ${(td.sd1*1000).toFixed(1)} | SD2 ${(td.sd2*1000).toFixed(1)} ms`;
}

function drawPSD(analysis) {
  const canvas = document.getElementById('psd-canvas');
  sizeCanvas(canvas);
  const ctx = canvas.getContext('2d');
  const D = dpr();
  const W = canvas.width, H = canvas.height;
  const padL = 55 * D, padR = 20 * D, padT = 16 * D, padB = 30 * D;
  const IW = W - padL - padR, IH = H - padT - padB;

  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = 'rgba(6,15,30,0.5)';
  ctx.fillRect(0, 0, W, H);

  const { freqs, psd } = analysis.spec;
  if (!freqs || freqs.length < 2) return;

  // Only show 0–0.5 Hz
  const maxFreq = 0.5;
  const visIdx = freqs.reduce((last, f, i) => f <= maxFreq ? i : last, 0);

  const fSlice = freqs.slice(0, visIdx + 1);
  const pSlice = psd.slice(0, visIdx + 1);
  const maxP = Math.max(...pSlice, 1);

  const sx = f => padL + (f / maxFreq) * IW;
  const sy = v => padT + IH - (v / maxP) * IH;

  // Band regions
  ctx.fillStyle = 'rgba(255,109,0,0.06)';
  ctx.fillRect(sx(0.04), padT, sx(0.15) - sx(0.04), IH);
  ctx.fillStyle = 'rgba(0,229,255,0.06)';
  ctx.fillRect(sx(0.15), padT, sx(0.40) - sx(0.15), IH);

  // Band labels
  ctx.font = `${8 * D}px Orbitron, monospace`;
  ctx.textAlign = 'center';
  ctx.fillStyle = 'rgba(255,109,0,0.5)';
  ctx.fillText('LF', sx(0.095), padT + 14 * D);
  ctx.fillStyle = 'rgba(0,229,255,0.5)';
  ctx.fillText('HF', sx(0.275), padT + 14 * D);

  // Band boundaries
  ctx.strokeStyle = 'rgba(100,120,140,0.2)';
  ctx.lineWidth = 1;
  ctx.setLineDash([3, 3]);
  [0.04, 0.15, 0.40].forEach(f => {
    ctx.beginPath(); ctx.moveTo(sx(f), padT); ctx.lineTo(sx(f), H - padB); ctx.stroke();
  });
  ctx.setLineDash([]);

  // LF area
  ctx.beginPath();
  ctx.moveTo(sx(0.04), H - padB);
  for (let i = 0; i < fSlice.length; i++) {
    if (fSlice[i] >= 0.04 && fSlice[i] <= 0.15) {
      ctx.lineTo(sx(fSlice[i]), sy(pSlice[i]));
    }
  }
  ctx.lineTo(sx(0.15), H - padB); ctx.closePath();
  ctx.fillStyle = 'rgba(255,109,0,0.25)';
  ctx.fill();

  // HF area
  ctx.beginPath();
  ctx.moveTo(sx(0.15), H - padB);
  for (let i = 0; i < fSlice.length; i++) {
    if (fSlice[i] >= 0.15 && fSlice[i] <= 0.40) {
      ctx.lineTo(sx(fSlice[i]), sy(pSlice[i]));
    }
  }
  ctx.lineTo(sx(0.40), H - padB); ctx.closePath();
  ctx.fillStyle = 'rgba(0,229,255,0.25)';
  ctx.fill();

  // Full line
  ctx.beginPath();
  for (let i = 0; i < fSlice.length; i++) {
    i === 0 ? ctx.moveTo(sx(fSlice[i]), sy(pSlice[i])) : ctx.lineTo(sx(fSlice[i]), sy(pSlice[i]));
  }
  ctx.strokeStyle = 'rgba(179,136,255,0.8)';
  ctx.lineWidth = 1.5 * D;
  ctx.stroke();

  // Y labels
  ctx.font = `${8 * D}px JetBrains Mono, monospace`;
  ctx.fillStyle = '#4a6080';
  ctx.textAlign = 'right';
  ctx.fillText(maxP.toFixed(0), padL - 4, padT + 9 * D);
  ctx.fillText('0', padL - 4, H - padB);

  // X labels
  ctx.textAlign = 'center';
  [0, 0.1, 0.2, 0.3, 0.4, 0.5].forEach(f => {
    ctx.fillText(f.toFixed(1), sx(f), H - 6 * D);
  });

  ctx.fillStyle = '#2a3848';
  ctx.textAlign = 'left';
  ctx.fillText('Hz', padL, H - 6 * D);

  // Update band power labels
  document.getElementById('lf-power').textContent  = analysis.spec.lf.toFixed(1);
  document.getElementById('hf-power').textContent  = analysis.spec.hf.toFixed(1);
  document.getElementById('lfhf-ratio').textContent = analysis.spec.lfhf.toFixed(2);
}

// ============================================================
//  UI UPDATES
// ============================================================
function updateStats(analysis) {
  const td = analysis.td, spec = analysis.spec, p = analysis.params;

  // Header metrics — show last filter value
  const last = analysis.filter[analysis.filter.length - 1];
  document.getElementById('hm-hr').textContent = Math.round(60 / last.mu);
  document.getElementById('hm-p').textContent  = last.p.toFixed(3);
  document.getElementById('hm-s').textContent  = last.s.toFixed(3);
  document.getElementById('hm-d').textContent  = last.delta.toFixed(3);
  document.getElementById('header-metrics').style.display = 'flex';

  // Metric cards (live state)
  updateLiveState(last.p, last.s, last.delta, 60 / last.mu);

  // HRV table
  document.getElementById('st-meanrr').textContent = fmtMs(td.meanRR);
  document.getElementById('st-sdnn').textContent   = fmtMs(td.sdnn);
  document.getElementById('st-rmssd').textContent  = fmtMs(td.rmssd);
  document.getElementById('st-pnn50').textContent  = td.pnn50.toFixed(1) + '%';
  document.getElementById('st-hr').textContent     = td.meanHR.toFixed(1) + ' bpm';
  document.getElementById('st-lf').textContent     = spec.lf.toFixed(1) + ' ms²';
  document.getElementById('st-hf').textContent     = spec.hf.toFixed(1) + ' ms²';
  document.getElementById('st-lfhf').textContent   = spec.lfhf.toFixed(2);
  document.getElementById('st-lfnu').textContent   = spec.lf_nu.toFixed(1) + ' n.u.';
  document.getElementById('st-hfnu').textContent   = spec.hf_nu.toFixed(1) + ' n.u.';
  document.getElementById('st-sd1').textContent    = fmtMs(td.sd1);
  document.getElementById('st-sd2').textContent    = fmtMs(td.sd2);
  document.getElementById('st-sd1sd2').textContent = td.sd1sd2.toFixed(3);

  // Model params
  document.getElementById('par-ap').textContent    = p.ap.toFixed(2) + ' Hz';
  document.getElementById('par-as').textContent    = p.as.toFixed(3) + ' Hz';
  document.getElementById('par-sp').textContent    = p.sigma_p.toFixed(4);
  document.getElementById('par-ss').textContent    = p.sigma_s.toFixed(4);
  document.getElementById('par-mu0').textContent   = (p.mu0 * 1000).toFixed(1) + ' ms';
  document.getElementById('par-rho').textContent   = p.rho.toFixed(4);
  document.getElementById('par-kappa').textContent = p.kappa.toFixed(1);
  document.getElementById('par-taup').textContent  = (1 / p.ap).toFixed(3) + ' s';
}

function updateLiveState(p, s, delta, hr) {
  document.getElementById('m-p').textContent  = p.toFixed(4);
  document.getElementById('m-s').textContent  = s.toFixed(4);
  document.getElementById('m-d').textContent  = delta.toFixed(4);
  document.getElementById('m-hr').textContent = isNaN(hr) ? '—' : Math.round(hr);

  // Bar widths (normalize to [-1, 1] range)
  const pPct = clamp(50 + p * 50, 5, 95);
  const sPct = clamp(50 + s * 50, 5, 95);
  document.getElementById('bar-p').style.width = pPct + '%';
  document.getElementById('bar-s').style.width = sPct + '%';

  // Balance indicator
  const totalTone = Math.abs(p) + Math.abs(s) + 0.01;
  const paraFrac = (Math.max(0, p) + 0.001) / (Math.max(0, p) + Math.max(0, s) + 0.002) * 100;
  const sympFrac = 100 - paraFrac;
  document.getElementById('bal-para').style.width = Math.min(paraFrac, 48) + '%';
  document.getElementById('bal-symp').style.width = Math.min(sympFrac, 48) + '%';

  const state_str = delta < -0.1 ? 'VAGAL DOMINANT' : delta > 0.1 ? 'SYMPATHETIC DOMINANT' : 'BALANCED';
  document.getElementById('bal-center').textContent = state_str;
}

function updatePlaybackDisplay(analysis) {
  if (!analysis || !analysis.filter.length) return;
  const N = analysis.filter.length;
  const f = state.playback.frame;
  const fdata = analysis.filter[Math.min(f, N - 1)];

  const pct = f / (N - 1) * 100;
  document.getElementById('timeline-fill').style.width  = pct + '%';
  document.getElementById('timeline-thumb').style.left  = pct + '%';

  const tCur = fdata.t;
  const tTot = analysis.filter[N - 1].t;
  const fmt = s => `${Math.floor(s/60)}:${('0'+Math.floor(s%60)).slice(-2)}`;
  document.getElementById('playback-time').textContent = `${fmt(tCur)} / ${fmt(tTot)}`;

  updateLiveState(fdata.p, fdata.s, fdata.delta, 60 / fdata.mu);
  document.getElementById('hm-hr').textContent = Math.round(60 / fdata.mu);
  document.getElementById('hm-p').textContent  = fdata.p.toFixed(3);
  document.getElementById('hm-s').textContent  = fdata.s.toFixed(3);
  document.getElementById('hm-d').textContent  = fdata.delta.toFixed(3);
}

// ============================================================
//  PLAYBACK
// ============================================================
function togglePlayback() {
  const pb = state.playback;
  pb.active = !pb.active;
  document.getElementById('play-btn').textContent = pb.active ? '⏸' : '▶';

  if (pb.active) {
    function tick() {
      if (!pb.active) return;
      const an = state.currentAnalysis;
      if (!an) return;
      pb.frame = Math.min(pb.frame + 1, an.filter.length - 1);
      if (pb.frame >= an.filter.length - 1) {
        pb.active = false;
        document.getElementById('play-btn').textContent = '▶';
        return;
      }
      updatePlaybackDisplay(an);
      pb.raf = setTimeout(tick, (an.rr[pb.frame] || 0.8) * 1000 / pb.speed);
    }
    pb.raf = setTimeout(tick, 50);
  }
}

function seekTo(frac) {
  const an = state.currentAnalysis;
  if (!an) return;
  state.playback.frame = Math.round(frac * (an.filter.length - 1));
  updatePlaybackDisplay(an);
}

// Timeline click
document.getElementById('timeline-bar').addEventListener('click', function(e) {
  const rect = this.getBoundingClientRect();
  const frac = (e.clientX - rect.left) / rect.width;
  seekTo(clamp(frac, 0, 1));
});

// ============================================================
//  RECORDINGS MANAGEMENT
// ============================================================
function renderRecordings() {
  const list = document.getElementById('recordings-list');
  if (state.recordings.length === 0) {
    list.innerHTML = '<div style="font-size:12px;color:var(--text-dim);text-align:center;padding:20px">No recordings loaded</div>';
    return;
  }
  list.innerHTML = state.recordings.map((r, i) => `
    <div class="rec-item${i === state.activeIdx ? ' active' : ''}" onclick="selectRecording(${i})">
      <div class="rec-icon">📈</div>
      <div class="rec-info">
        <div class="rec-name">${r.name}</div>
        <div class="rec-meta">${r.n} beats · ${r.duration}</div>
      </div>
      <button class="rec-del" onclick="event.stopPropagation();deleteRecording(${i})">✕</button>
    </div>
  `).join('');
}

function selectRecording(idx) {
  state.activeIdx = idx;
  state.playback.frame = 0;
  state.playback.active = false;
  document.getElementById('play-btn').textContent = '▶';
  const analysis = state.analysisCache[idx];
  state.currentAnalysis = analysis;
  renderRecordings();
  renderDashboard(analysis);
}

function deleteRecording(idx) {
  state.recordings.splice(idx, 1);
  delete state.analysisCache[idx];
  // Re-index
  const newCache = {};
  Object.keys(state.analysisCache).forEach(k => {
    const ki = parseInt(k);
    if (ki > idx) newCache[ki - 1] = state.analysisCache[k];
    else if (ki < idx) newCache[ki] = state.analysisCache[k];
  });
  state.analysisCache = newCache;
  if (state.activeIdx >= state.recordings.length) state.activeIdx = state.recordings.length - 1;
  if (state.recordings.length === 0) {
    document.getElementById('dashboard').style.display = 'none';
    document.getElementById('upload-screen').style.display = 'flex';
    state.activeIdx = -1;
    state.currentAnalysis = null;
  } else if (state.activeIdx >= 0) {
    selectRecording(state.activeIdx);
  }
  renderRecordings();
}

// ============================================================
//  RENDER FULL DASHBOARD
// ============================================================
function renderDashboard(analysis) {
  updateStats(analysis);
  drawBranchChart(analysis);
  drawTachogram(analysis);
  drawPoincare(analysis);
  drawPSD(analysis);
  if (state.visMode === 'phase') drawPhaseSpace(analysis);
}

// ============================================================
//  VIS MODE SWITCH
// ============================================================
function switchVisMode(mode, btn) {
  state.visMode = mode;
  document.querySelectorAll('.vis-tab').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  const panel = document.getElementById('vis-panel');
  panel.className = 'vis-mode-' + mode;
  document.getElementById('three-canvas').style.display = mode === 'ans' ? 'block' : 'none';
  document.getElementById('phase-canvas').style.display = mode === 'phase' ? 'block' : 'none';
  if (mode === 'phase' && state.currentAnalysis) drawPhaseSpace(state.currentAnalysis);
}

// ============================================================
//  FILE UPLOAD
// ============================================================
function triggerUpload() {
  document.getElementById('file-input').click();
}

document.getElementById('file-input').addEventListener('change', function(e) {
  const files = Array.from(e.target.files);
  if (files.length === 0) return;
  processFiles(files);
  this.value = '';
});

// Drag & drop
const uploadZone = document.getElementById('upload-zone');
uploadZone.addEventListener('dragover', e => { e.preventDefault(); uploadZone.classList.add('dragging'); });
uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('dragging'));
uploadZone.addEventListener('drop', e => {
  e.preventDefault();
  uploadZone.classList.remove('dragging');
  const files = Array.from(e.dataTransfer.files).filter(f => /\.(txt|csv)$/i.test(f.name));
  if (files.length) processFiles(files);
});

// Global drag drop
document.addEventListener('dragover', e => e.preventDefault());
document.addEventListener('drop', e => {
  e.preventDefault();
  const files = Array.from(e.dataTransfer.files).filter(f => /\.(txt|csv)$/i.test(f.name));
  if (files.length) processFiles(files);
});

async function processFiles(files) {
  showLoading(true, 'Reading files...');

  for (const file of files) {
    const text = await readFile(file);
    const rr = parseFile(text, file.name);
    if (!rr) {
      alert(`Could not parse file: ${file.name}\nMake sure it contains valid RR interval data.`);
      continue;
    }

    updateLoadingText('Running SDE-IG analysis...');
    await sleep(10); // Allow UI to update

    const analysis = analyzeRR(rr);
    const tEnd = analysis.times[analysis.times.length - 1];
    const mins = Math.floor(tEnd / 60), secs = Math.floor(tEnd % 60);

    const idx = state.recordings.length;
    state.recordings.push({
      name: file.name.replace(/\.[^.]+$/, ''),
      n: rr.length,
      duration: `${mins}:${('0'+secs).slice(-2)} min`
    });
    state.analysisCache[idx] = analysis;

    updateLoadingText('Rendering visualizations...');
    await sleep(10);

    // Show dashboard
    document.getElementById('upload-screen').style.display = 'none';
    document.getElementById('dashboard').style.display = 'block';

    if (!renderer) initThreeScene();

    state.activeIdx = idx;
    state.currentAnalysis = analysis;
    state.playback.frame = analysis.filter.length - 1; // Show last state
    document.getElementById('playback-section').style.display = 'block';

    renderRecordings();
    renderDashboard(analysis);
    updatePlaybackDisplay(analysis);
  }

  showLoading(false);

  // Status
  document.getElementById('status-dot').className = 'status-dot active';
  document.getElementById('status-text').textContent = 'ANALYSIS COMPLETE';
  document.getElementById('rec-info-pill').style.display = 'flex';
  const an = state.currentAnalysis;
  if (an) document.getElementById('rec-info-text').textContent = `${an.rr.length} beats · ${an.td.meanHR.toFixed(0)} bpm`;
}

function readFile(file) {
  return new Promise((res, rej) => {
    const r = new FileReader();
    r.onload = e => res(e.target.result);
    r.onerror = rej;
    r.readAsText(file);
  });
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

function showLoading(show, msg) {
  const el = document.getElementById('loading-overlay');
  el.style.display = show ? 'flex' : 'none';
  if (msg) document.getElementById('loading-sub').textContent = msg;
}

function updateLoadingText(msg) {
  document.getElementById('loading-sub').textContent = msg;
}

// ============================================================
//  EXPORT REPORT
// ============================================================
function exportReport() {
  const an = state.currentAnalysis;
  if (!an) { alert('No analysis loaded.'); return; }

  const td = an.td, sp = an.spec, p = an.params;
  const last = an.filter[an.filter.length - 1];
  const recName = state.recordings[state.activeIdx]?.name || 'recording';
  const now = new Date().toLocaleString();

  const html = `<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<title>ANS Report — ${recName}</title>
<style>
  body{font-family:'Segoe UI',sans-serif;margin:0;background:#f8fafc;color:#1e293b}
  .header{background:linear-gradient(135deg,#020912,#0a1628);color:#fff;padding:32px 40px}
  h1{font-size:24px;margin:0;letter-spacing:0.05em}
  .sub{color:#00e5ff;font-size:12px;letter-spacing:0.1em;margin-top:4px}
  .meta{font-size:11px;color:#64748b;margin-top:8px}
  .body{padding:32px 40px}
  .section{margin-bottom:28px}
  h2{font-size:14px;letter-spacing:0.08em;text-transform:uppercase;color:#475569;border-bottom:2px solid #e2e8f0;padding-bottom:8px;margin-bottom:16px}
  .grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}
  .card{background:#fff;border:1px solid #e2e8f0;border-radius:8px;padding:16px}
  .card-label{font-size:11px;color:#94a3b8;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:4px}
  .card-val{font-size:20px;font-weight:600;color:#0f172a;font-family:monospace}
  .card-val.para{color:#0891b2}
  .card-val.symp{color:#ea580c}
  .card-val.net{color:#059669}
  table{width:100%;border-collapse:collapse;font-size:13px}
  th{text-align:left;padding:8px 12px;background:#f1f5f9;font-size:11px;text-transform:uppercase;letter-spacing:0.06em;color:#64748b}
  td{padding:8px 12px;border-bottom:1px solid #f1f5f9}
  .footer{margin-top:40px;padding-top:16px;border-top:1px solid #e2e8f0;font-size:11px;color:#94a3b8}
</style>
</head><body>
<div class="header">
  <h1>AUTONOMIC NAVIGATOR</h1>
  <div class="sub">SDE-INVERSE GAUSSIAN AUTONOMIC BRANCH ANALYSIS</div>
  <div class="meta">Recording: ${recName} · Generated: ${now}</div>
</div>
<div class="body">
  <div class="section">
    <h2>Current Autonomic State (Final Estimate)</h2>
    <div class="grid">
      <div class="card"><div class="card-label">Vagal Tone p̂(t)</div><div class="card-val para">${last.p.toFixed(4)}</div></div>
      <div class="card"><div class="card-label">Adrenergic Tone ŝ(t)</div><div class="card-val symp">${last.s.toFixed(4)}</div></div>
      <div class="card"><div class="card-label">Net Autonomic Drive Δ̂(t)</div><div class="card-val net">${last.delta.toFixed(4)}</div></div>
      <div class="card"><div class="card-label">Instantaneous HR</div><div class="card-val">${Math.round(60/last.mu)} bpm</div></div>
    </div>
  </div>
  <div class="section">
    <h2>Time Domain HRV Statistics</h2>
    <table>
      <tr><th>Parameter</th><th>Value</th><th>Unit</th></tr>
      <tr><td>Number of RR intervals</td><td>${td.n}</td><td>beats</td></tr>
      <tr><td>Mean RR</td><td>${(td.meanRR*1000).toFixed(2)}</td><td>ms</td></tr>
      <tr><td>SDNN</td><td>${(td.sdnn*1000).toFixed(2)}</td><td>ms</td></tr>
      <tr><td>RMSSD</td><td>${(td.rmssd*1000).toFixed(2)}</td><td>ms</td></tr>
      <tr><td>pNN50</td><td>${td.pnn50.toFixed(2)}</td><td>%</td></tr>
      <tr><td>Mean Heart Rate</td><td>${td.meanHR.toFixed(2)}</td><td>bpm</td></tr>
      <tr><td>SD1 (Poincaré)</td><td>${(td.sd1*1000).toFixed(2)}</td><td>ms</td></tr>
      <tr><td>SD2 (Poincaré)</td><td>${(td.sd2*1000).toFixed(2)}</td><td>ms</td></tr>
      <tr><td>SD1/SD2</td><td>${td.sd1sd2.toFixed(4)}</td><td>—</td></tr>
    </table>
  </div>
  <div class="section">
    <h2>Frequency Domain HRV Statistics</h2>
    <table>
      <tr><th>Parameter</th><th>Value</th><th>Unit</th></tr>
      <tr><td>VLF Power (0.003–0.04 Hz)</td><td>${sp.vlf.toFixed(2)}</td><td>ms²</td></tr>
      <tr><td>LF Power (0.04–0.15 Hz)</td><td>${sp.lf.toFixed(2)}</td><td>ms²</td></tr>
      <tr><td>HF Power (0.15–0.40 Hz)</td><td>${sp.hf.toFixed(2)}</td><td>ms²</td></tr>
      <tr><td>Total Power (0.003–0.40 Hz)</td><td>${sp.tp.toFixed(2)}</td><td>ms²</td></tr>
      <tr><td>LF/HF Ratio</td><td>${sp.lfhf.toFixed(4)}</td><td>—</td></tr>
      <tr><td>LF Normalized Units</td><td>${sp.lf_nu.toFixed(2)}</td><td>n.u.</td></tr>
      <tr><td>HF Normalized Units</td><td>${sp.hf_nu.toFixed(2)}</td><td>n.u.</td></tr>
      <tr><td>LF Peak Frequency</td><td>${sp.lfPeak.toFixed(4)}</td><td>Hz</td></tr>
      <tr><td>HF Peak Frequency</td><td>${sp.hfPeak.toFixed(4)}</td><td>Hz</td></tr>
    </table>
  </div>
  <div class="section">
    <h2>SDE-IG Model Parameters (Estimated)</h2>
    <table>
      <tr><th>Parameter</th><th>Symbol</th><th>Value</th><th>Interpretation</th></tr>
      <tr><td>Vagal self-decay rate</td><td>aₚ</td><td>${p.ap.toFixed(4)} Hz</td><td>τₚ = ${(1/p.ap).toFixed(3)} s</td></tr>
      <tr><td>Sympathetic self-decay rate</td><td>aₛ</td><td>${p.as.toFixed(4)} Hz</td><td>τₛ = ${(1/p.as).toFixed(2)} s</td></tr>
      <tr><td>Vagal noise amplitude</td><td>σₚ</td><td>${p.sigma_p.toFixed(6)}</td><td>HF spectral power</td></tr>
      <tr><td>Sympathetic noise amplitude</td><td>σₛ</td><td>${p.sigma_s.toFixed(6)}</td><td>LF spectral power</td></tr>
      <tr><td>Baseline mean RR interval</td><td>μ₀</td><td>${(p.mu0*1000).toFixed(2)} ms</td><td>${(60/p.mu0).toFixed(1)} bpm at neutrality</td></tr>
      <tr><td>Baseline coefficient of variation</td><td>ρ</td><td>${p.rho.toFixed(6)}</td><td>Interval regularity</td></tr>
      <tr><td>Shape parameter</td><td>κ</td><td>${p.kappa.toFixed(4)}</td><td>κ = μ₀/ρ²</td></tr>
    </table>
  </div>
  <div class="footer">
    Report generated by AUTONOMIC NAVIGATOR · SDE-IG framework (Castillo-Aguilar, 2026)<br>
    Based on: Continuous-Time Stochastic State-Space Framework for Heart Rate Variability<br>
    Data processed locally — no information transmitted externally.
  </div>
</div></body></html>`;

  const blob = new Blob([html], { type: 'text/html' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `ANS_Report_${recName}_${Date.now()}.html`;
  a.click();
  URL.revokeObjectURL(url);
}

// ============================================================
//  LOGO ANIMATION
// ============================================================
function initLogo() {
  const canvas = document.getElementById('logo-canvas');
  const ctx = canvas.getContext('2d');
  let t = 0;

  function drawLogo() {
    ctx.clearRect(0, 0, 36, 36);
    ctx.fillStyle = '#020912';
    ctx.fillRect(0, 0, 36, 36);

    const cx = 18, cy = 18, r = 12;
    // Outer ring
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.strokeStyle = 'rgba(0,229,255,0.3)';
    ctx.lineWidth = 1;
    ctx.stroke();

    // Rotating para arc
    ctx.beginPath();
    ctx.arc(cx, cy, r, t, t + Math.PI * 1.2);
    ctx.strokeStyle = '#00e5ff';
    ctx.lineWidth = 2;
    ctx.stroke();

    // Rotating symp arc
    ctx.beginPath();
    ctx.arc(cx, cy, r - 4, -t * 0.7, -t * 0.7 + Math.PI * 0.9);
    ctx.strokeStyle = '#ff6d00';
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // Center dot
    ctx.beginPath();
    ctx.arc(cx, cy, 2.5, 0, Math.PI * 2);
    ctx.fillStyle = 0.5 + 0.5 * Math.sin(t * 3) > 0.7 ? '#ff2244' : '#ff4466';
    ctx.fill();

    t += 0.04;
    requestAnimationFrame(drawLogo);
  }
  drawLogo();
}

// ============================================================
//  WINDOW RESIZE
// ============================================================
let _resizeTimer = null;
window.addEventListener('resize', () => {
  clearTimeout(_resizeTimer);
  _resizeTimer = setTimeout(() => {
    if (state.currentAnalysis) {
      drawBranchChart(state.currentAnalysis);
      drawTachogram(state.currentAnalysis);
      drawPoincare(state.currentAnalysis);
      drawPSD(state.currentAnalysis);
      if (state.visMode === 'phase') drawPhaseSpace(state.currentAnalysis);
    }
  }, 150);
});

// ============================================================
//  BACKGROUND ANIMATION (upload screen)
// ============================================================
function initBgAnimation() {
  const canvas = document.getElementById('bg-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  const nodes = Array.from({ length: 40 }, () => ({
    x: Math.random() * window.innerWidth,
    y: Math.random() * window.innerHeight,
    vx: (Math.random() - 0.5) * 0.4,
    vy: (Math.random() - 0.5) * 0.4,
    r: 1 + Math.random() * 2,
    type: Math.random() > 0.5 ? 'para' : 'symp',
    pulse: Math.random() * Math.PI * 2
  }));

  function resize() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
  }
  resize();
  window.addEventListener('resize', resize);

  let frame = 0;
  function draw() {
    if (document.getElementById('upload-screen').style.display === 'none') return;
    requestAnimationFrame(draw);
    frame++;

    ctx.fillStyle = 'rgba(2,9,18,0.25)';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    nodes.forEach(n => {
      n.x += n.vx; n.y += n.vy; n.pulse += 0.03;
      if (n.x < 0 || n.x > canvas.width)  n.vx *= -1;
      if (n.y < 0 || n.y > canvas.height) n.vy *= -1;

      const glow = 0.5 + 0.5 * Math.sin(n.pulse);
      const color = n.type === 'para' ? `rgba(0,229,255,${0.3 + 0.4 * glow})` : `rgba(255,109,0,${0.3 + 0.4 * glow})`;
      ctx.beginPath();
      ctx.arc(n.x, n.y, n.r * (1 + 0.3 * glow), 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();
    });

    // Connections
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const dx = nodes[i].x - nodes[j].x;
        const dy = nodes[i].y - nodes[j].y;
        const d = Math.sqrt(dx * dx + dy * dy);
        if (d < 140) {
          const alpha = (1 - d / 140) * 0.15;
          const iColor = nodes[i].type === 'para' ? '0,229,255' : '255,109,0';
          ctx.strokeStyle = `rgba(${iColor},${alpha})`;
          ctx.lineWidth = 0.5;
          ctx.beginPath();
          ctx.moveTo(nodes[i].x, nodes[i].y);
          ctx.lineTo(nodes[j].x, nodes[j].y);
          ctx.stroke();
        }
      }
    }
  }
  draw();
}

// ============================================================
//  INIT
// ============================================================
initLogo();
initBgAnimation();

// Handle window resize for vis panel after dashboard shown
const visResizeObs = new ResizeObserver(() => {
  if (state.visMode === 'phase' && state.currentAnalysis) {
    drawPhaseSpace(state.currentAnalysis);
  }
});
visResizeObs.observe(document.getElementById('vis-panel'));

// Demo data generator (for testing without file)
window.loadDemoData = function() {
  const n = 600;
  const rr = [];
  let mu = 0.85;
  for (let i = 0; i < n; i++) {
    const t = i / n * 300;
    // Simulate stress: sympathetic increases after t=100s
    const stress = t > 100 && t < 200 ? (t - 100) / 100 : t >= 200 ? 0 : 0;
    const hr_target = 60 + stress * 25;
    mu = 60 / hr_target;
    // Add HRV
    const hf = 0.04 * Math.sin(2 * Math.PI * 0.25 * t);
    const lf = 0.06 * Math.sin(2 * Math.PI * 0.1 * t);
    const noise = (Math.random() - 0.5) * 0.02;
    rr.push(clamp(mu + hf + lf + noise, 0.35, 1.5));
  }
  const blob = new Blob([rr.map(v => (v * 1000).toFixed(1)).join('\n')], { type: 'text/plain' });
  processFiles([new File([blob], 'demo_recording.txt')]);
};
