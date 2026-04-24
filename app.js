'use strict';

// ============================================================
//  GLOBAL STATE
// ============================================================
const state = {
  recordings: [],
  activeIdx: -1,
  analysisCache: {},
  playback: { active: false, frame: 0, speed: 5, virtualTime: 0, lastTs: null },
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

// ── Inverse FFT (conjugate trick) ────────────────────────────
function ifft(re, im) {
  const n = re.length;
  for (let i = 0; i < n; i++) im[i] = -im[i];
  fft(re, im);
  for (let i = 0; i < n; i++) { re[i] /= n; im[i] = -im[i] / n; }
}

// ── FFT-based bandpass filter ────────────────────────────────
function bandpassFFT(x, fs, f1, f2) {
  const N = nextPow2(x.length * 2);
  const re = new Float64Array(N), im = new Float64Array(N);
  for (let i = 0; i < x.length; i++) re[i] = x[i];
  fft(re, im);
  const df = fs / N;
  for (let i = 0; i < N; i++) {
    const f = Math.abs(i <= N / 2 ? i * df : (i - N) * df);
    if (f < f1 || f > f2) { re[i] = 0; im[i] = 0; }
  }
  ifft(re, im);
  return Array.from(re).slice(0, x.length);
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
    hfPeak: peakFreq(0.15, 0.40),
    rrU: Array.from(rrU),   // uniform-grid RR for coupling & wavelet
    fs                       // sample rate used
  };
}

// ============================================================
//  COUPLING ESTIMATION  (bivariate AR-1 on band-filtered proxies)
// ============================================================
function estimateCoupling(rrU, fs, ap, as_) {
  const n = rrU.length;
  if (n < 40) return { a_ps: 0, a_sp: 0 };
  const dt = 1 / fs;
  const mu = mean(rrU);
  const x  = rrU.map(v => v - mu);

  // Band-filtered proxies  (HF ↔ p,  LF ↔ s)
  const pP = bandpassFFT(x, fs, 0.15, 0.40);
  const sP = bandpassFFT(x, fs, 0.04, 0.15);

  // Bivariate AR(1) normal equations
  let sp2=0, ss2=0, sps=0, spnp=0, spns=0, ssnp=0, ssns=0;
  for (let i = 1; i < n; i++) {
    const pp = pP[i-1], sp = sP[i-1], pn = pP[i], sn = sP[i];
    sp2  += pp*pp;  ss2  += sp*sp;  sps  += pp*sp;
    spnp += pp*pn;  spns += sp*pn;
    ssnp += pp*sn;  ssns += sp*sn;
  }
  const det = sp2*ss2 - sps*sps + 1e-20;

  // Row-wise OLS
  const alpha_ps = (sp2*spns - sps*spnp) / det;  // coupling: s[n-1] → p[n]
  const alpha_sp = (ss2*ssnp - sps*ssns) / det;  // coupling: p[n-1] → s[n]

  // First-order continuous-time conversion: A_offdiag ≈ alpha/dt
  let a_ps = alpha_ps / dt;
  let a_sp = alpha_sp / dt;

  // Stability guard: |a_ps * a_sp| < 0.8 * ap * as
  const maxCpl = 0.8 * ap * as_;
  const mag    = Math.sqrt(Math.abs(a_ps * a_sp)) + 1e-10;
  if (mag > Math.sqrt(maxCpl)) {
    const scale = Math.sqrt(maxCpl) / mag;
    a_ps *= scale; a_sp *= scale;
  }
  a_ps = clamp(a_ps, -ap * 0.85, ap * 0.85);
  a_sp = clamp(a_sp, -as_ * 0.85, as_ * 0.85);

  return { a_ps, a_sp };
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

  const { a_ps, a_sp } = estimateCoupling(spec.rrU || [], spec.fs || 4, ap, as);
  return { ap, as, sigma_p, sigma_s, mu0, rho, kappa, a_ps, a_sp };
}

// ── 2×2 matrix exponential via Cayley-Hamilton theorem ───────
function matexp2x2(a00, a01, a10, a11, tau) {
  const b00=a00*tau, b01=a01*tau, b10=a10*tau, b11=a11*tau;
  const tr=b00+b11, det=b00*b11-b01*b10, disc=tr*tr/4-det;

  if (Math.abs(disc) < 1e-12) {
    const lam=tr/2, el=Math.exp(lam);
    return [el*(1+b00-lam), el*b01, el*b10, el*(1+b11-lam)];
  }
  if (disc > 0) {
    const sq=Math.sqrt(disc), lam1=tr/2+sq, lam2=tr/2-sq;
    const e1=Math.exp(lam1), e2=Math.exp(lam2), c=1/(lam1-lam2);
    return [
      c*(e1*(b00-lam2)-e2*(b00-lam1)), c*b01*(e1-e2),
      c*b10*(e1-e2),                   c*(e1*(b11-lam2)-e2*(b11-lam1))
    ];
  }
  // Complex eigenvalues → damped oscillation
  const sq=Math.sqrt(-disc), eat=Math.exp(tr/2);
  const cb=Math.cos(sq), sb=Math.sin(sq)/sq;
  return [
    eat*(cb+(b00-tr/2)*sb), eat*b01*sb,
    eat*b10*sb,             eat*(cb+(b11-tr/2)*sb)
  ];
}

// ============================================================
//  EKF (2D: [p, s] state)
// ============================================================
// ============================================================
//  EKF — coupled 2-D OU model  [p, s]
//  dp = (−ap·p + a_ps·s) dt + σp dWp
//  ds = (a_sp·p − as·s) dt + σs dWs
// ============================================================
class ANS_EKF {
  constructor(params) {
    this.ap  = params.ap;     this.as  = params.as;
    this.sp  = params.sigma_p; this.ss  = params.sigma_s;
    this.aps = params.a_ps || 0;   // symp → para coupling
    this.asp = params.a_sp || 0;   // para → symp coupling
    this.mu0 = params.mu0;
    this.kap = params.kappa;

    this.m = [0, 0];
    const vp = this.sp*this.sp / (2*this.ap);
    const vs = this.ss*this.ss / (2*this.as);
    this.P = [[vp, 0], [0, vs]];
  }

  step(tau) {
    // ── Transition: Φ = expm(A·τ), A = [[-ap, aps],[asp, -as]]
    const [p00,p01,p10,p11] = matexp2x2(-this.ap, this.aps, this.asp, -this.as, tau);

    const mp = [
      p00*this.m[0] + p01*this.m[1],
      p10*this.m[0] + p11*this.m[1]
    ];

    // ── Predicted covariance: Pp = Φ·P·Φ' + Q  (Q ≈ diag(σ²·τ))
    const P = this.P;
    const Pp = [
      [
        p00*(p00*P[0][0]+p01*P[1][0]) + p01*(p00*P[0][1]+p01*P[1][1]) + this.sp*this.sp*tau,
        p00*(p10*P[0][0]+p11*P[1][0]) + p01*(p10*P[0][1]+p11*P[1][1])
      ],[
        p10*(p00*P[0][0]+p01*P[1][0]) + p11*(p00*P[0][1]+p01*P[1][1]),
        p10*(p10*P[0][0]+p11*P[1][0]) + p11*(p10*P[0][1]+p11*P[1][1]) + this.ss*this.ss*tau
      ]
    ];

    // ── Observation: h(x) = μ₀ · exp(p − s)
    const delta_p = mp[0] - mp[1];
    const mu_p    = clamp(this.mu0 * Math.exp(delta_p), 0.2, 3.0);
    const R       = mu_p*mu_p*mu_p / this.kap;

    // Jacobian H = [∂h/∂p, ∂h/∂s] = [μ, −μ]
    const Hp = mu_p, Hs = -mu_p;

    const S  = Hp*(Hp*Pp[0][0]+Hs*Pp[1][0]) + Hs*(Hp*Pp[0][1]+Hs*Pp[1][1]) + R;
    const Kp = (Hp*Pp[0][0]+Hs*Pp[0][1]) / S;
    const Ks = (Hp*Pp[1][0]+Hs*Pp[1][1]) / S;

    const innov = clamp(tau - mu_p, -1.5, 1.5);
    this.m = [mp[0]+Kp*innov, mp[1]+Ks*innov];

    // Joseph-form update
    const KSKp=Kp*S*Kp, KSKs=Ks*S*Ks, KSKps=Kp*S*Ks;
    this.P = [
      [Pp[0][0]-KSKp,  Pp[0][1]-KSKps],
      [Pp[1][0]-KSKps, Pp[1][1]-KSKs]
    ];

    const delta = this.m[1] - this.m[0];   // s − p  (net adrenergic drive)
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

// ── Phase-space animated flow particles ──────────────────────
const phaseAnim = {
  active: false, rafId: null, particles: [],
  N: 72, analysis: null,
  gammaPS: 0, gammaSP: 0, ap: 1, as_val: 1, AX: 3.5,
};

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

  // Expose lights so animateScene can modulate them from EKF output
  scene.userData.paraLight   = paraLight;
  scene.userData.sympLight   = sympLight;
  scene.userData.centerLight = centerLight;

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
    const beatPhase  = (animTime % rrInterval) / rrInterval;
    const heartScale = 1 + 0.22 * Math.exp(-8 * beatPhase) * Math.sin(10 * beatPhase);
    heartMesh.scale.setScalar(heartScale);
    heartMesh.material.emissiveIntensity = 2.5 + 1.5 * heartScale;

    // Flash beat — ANS globe mode only
    if (beatPhase < 0.05 && state.visMode === 'ans') {
      const flash = document.getElementById('beat-flash');
      if (flash) { flash.style.opacity = '1'; setTimeout(() => { if (flash) flash.style.opacity = '0'; }, 60); }
    }

    // ── σ-normalised EKF values ──────────────────────────────────
    // Raw p/s values are zero-mean by construction; normalise by the
    // stationary σ so the visual magnitudes are meaningful.
    const pr_g  = an ? an.params : null;
    const sigPg = pr_g ? pr_g.sigma_p / Math.sqrt(2 * pr_g.ap) : 0.05;
    const sigSg = pr_g ? pr_g.sigma_s / Math.sqrt(2 * pr_g.as) : 0.05;
    const pSig  = clamp(curP / (sigPg * 2.5), -1, 1);   // ±1 ≈ ±2.5σ
    const sSig  = clamp(curS / (sigSg * 2.5), -1, 1);

    // Heart colour tracks net balance: cyan (vagal) → red (neutral) → orange (sympathetic)
    // curDelta = s − p, so positive = sympathetic dominant
    const balT = clamp(curDelta * 2.5 + 0.5, 0, 1);
    heartMesh.material.emissive.setRGB(
      0.50 + balT  * 0.50,
      0.04 + (1 - Math.abs(balT * 2 - 1)) * 0.10,
      0.20 * (1 - balT)
    );

    // Dynamic point lights track actual branch estimates
    const pL = scene.userData.paraLight;
    const sL = scene.userData.sympLight;
    if (pL) pL.intensity = 1.0 + clamp((pSig + 1) * 2.5, 0, 6);
    if (sL) sL.intensity = 1.0 + clamp((sSig + 1) * 2.5, 0, 6);

    // ANS state badge (only when globe is visible)
    const badge = document.getElementById('ans-state-badge');
    if (badge && state.visMode === 'ans') {
      const diff = pSig - sSig;
      if (diff > 0.25) {
        badge.textContent = 'VAGAL DOMINANT';
        badge.className   = 'ans-state-badge vagal';
      } else if (diff < -0.25) {
        badge.textContent = 'SYMPATHETIC DOMINANT';
        badge.className   = 'ans-state-badge sympathetic';
      } else {
        badge.textContent = 'BALANCED';
        badge.className   = 'ans-state-badge balanced';
      }
    }

    // ── Parasympathetic node — size/brightness from σ-normalised p ──
    const paraAct   = Math.max(0, pSig);    // 0 → 1
    const paraSize  = 0.45 + paraAct * 0.55;
    const paraAngle = animTime * (0.3 + paraAct * 0.12);
    paraNode.position.set(
      -5 * Math.cos(paraAngle * 0.3),
       1.5 + Math.sin(animTime * 0.8) * 0.3,
      -5 * Math.sin(paraAngle * 0.3)
    );
    paraNode.scale.setScalar(paraSize);
    paraNode.material.emissiveIntensity = 0.8 + paraAct * 3.5;

    for (let i = 1; i <= 3; i++) {
      const g = paraNode.userData['glow' + i];
      if (g) g.position.copy(paraNode.position);
    }

    // ── Sympathetic node — size/brightness from σ-normalised s ──
    const sympAct   = Math.max(0, sSig);
    const sympSize  = 0.45 + sympAct * 0.55;
    const sympAngle = -animTime * (0.2 + sympAct * 0.09);
    sympNode.position.set(
       5 * Math.cos(sympAngle * 0.25),
       1.5 + Math.sin(animTime * 0.5 + 1) * 0.3,
       5 * Math.sin(sympAngle * 0.25)
    );
    sympNode.scale.setScalar(sympSize);
    sympNode.material.emissiveIntensity = 0.8 + sympAct * 3.5;

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
//  PHASE SPACE ANIMATION (flow particles)
// ============================================================
function _initPhaseParticles(analysis) {
  const pr   = analysis.params;
  const sigP = pr.sigma_p / Math.sqrt(2 * pr.ap);
  const sigS = pr.sigma_s / Math.sqrt(2 * pr.as);
  phaseAnim.gammaPS  = (pr.a_ps || 0) * sigS / sigP;
  phaseAnim.gammaSP  = (pr.a_sp || 0) * sigP / sigS;
  phaseAnim.ap       = pr.ap;
  phaseAnim.as_val   = pr.as;
  phaseAnim.analysis = analysis;
  const AX = phaseAnim.AX;
  phaseAnim.particles = Array.from({ length: phaseAnim.N }, () => ({
    p: (Math.random() * 2 - 1) * AX * 0.88,
    s: (Math.random() * 2 - 1) * AX * 0.88,
    age: Math.floor(Math.random() * 70),
    maxAge: 55 + Math.floor(Math.random() * 85),
    trail: [],
  }));
}

function startPhaseAnim(analysis) {
  stopPhaseAnim();
  _initPhaseParticles(analysis);
  phaseAnim.active = true;
  phaseAnim.rafId  = requestAnimationFrame(_phaseTick);
}

function stopPhaseAnim() {
  phaseAnim.active = false;
  if (phaseAnim.rafId) { cancelAnimationFrame(phaseAnim.rafId); phaseAnim.rafId = null; }
}

function _phaseTick() {
  if (!phaseAnim.active) return;
  const an = phaseAnim.analysis;
  if (!an) return;
  const { ap, as_val, gammaPS, gammaSP, AX } = phaseAnim;
  const DT = 0.09;   // virtual seconds per frame  (~real-time × 5.5 at 60 fps)

  for (const pt of phaseAnim.particles) {
    pt.trail.push({ p: pt.p, s: pt.s });
    if (pt.trail.length > 16) pt.trail.shift();

    const dvp  = (-ap * pt.p + gammaPS * pt.s) * DT;
    const dvs  = (gammaSP * pt.p - as_val * pt.s) * DT;
    pt.p += dvp; pt.s += dvs; pt.age++;

    const dist = Math.hypot(pt.p, pt.s);
    if (dist < 0.07 || Math.abs(pt.p) > AX * 1.06 || Math.abs(pt.s) > AX * 1.06 || pt.age > pt.maxAge) {
      pt.p      = (Math.random() * 2 - 1) * AX * 0.87;
      pt.s      = (Math.random() * 2 - 1) * AX * 0.87;
      pt.age    = 0;
      pt.maxAge = 55 + Math.floor(Math.random() * 85);
      pt.trail  = [];
    }
  }
  drawPhaseSpace(an);
  phaseAnim.rafId = requestAnimationFrame(_phaseTick);
}

// ============================================================
//  PHASE SPACE CANVAS
// ============================================================
function drawPhaseSpace(analysis) {
  const canvas = document.getElementById('phase-canvas');
  const panel  = document.getElementById('vis-panel');
  const D      = window.devicePixelRatio || 1;

  canvas.width  = panel.clientWidth  * D;
  canvas.height = panel.clientHeight * D;
  canvas.style.width  = panel.clientWidth  + 'px';
  canvas.style.height = panel.clientHeight + 'px';

  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  const pad = { l: 62*D, r: 34*D, t: 52*D, b: 58*D };
  const IW  = W - pad.l - pad.r;
  const IH  = H - pad.t - pad.b;

  ctx.fillStyle = '#020912';
  ctx.fillRect(0, 0, W, H);

  const filt = analysis.filter;
  if (!filt || filt.length < 2) return;

  const pr = analysis.params;

  // ── Stationary σ for each OU branch ─────────────────────────
  // σ_stat = σ / √(2a)  →  the phase-space axes are in these units
  const sigP = pr.sigma_p / Math.sqrt(2 * pr.ap);
  const sigS = pr.sigma_s / Math.sqrt(2 * pr.as);

  const normP = filt.map(f => f.p / sigP);
  const normS = filt.map(f => f.s / sigS);

  // Axis range: ±3.5 σ covers >99.9 % of the stationary distribution
  const AX = 3.5;
  const toX = v => pad.l + (v + AX) / (2 * AX) * IW;
  const toY = v => pad.t + IH - (v + AX) / (2 * AX) * IH;   // canvas-y inverted

  const ox = toX(0), oy = toY(0);   // equilibrium pixel

  // ── Quadrant fills ───────────────────────────────────────────
  // Convention: right = more vagal (p↑ → longer RR), up = more adrenergic
  // Q-TR (high p, high s) → co-activation
  ctx.fillStyle = 'rgba(255,214,0,0.025)';
  ctx.fillRect(ox, pad.t, W - pad.r - ox, oy - pad.t);
  // Q-TL (low p, high s) → stress / SNS dominant
  ctx.fillStyle = 'rgba(255,50,50,0.035)';
  ctx.fillRect(pad.l, pad.t, ox - pad.l, oy - pad.t);
  // Q-BL (low p, low s) → withdrawal
  ctx.fillStyle = 'rgba(80,80,130,0.030)';
  ctx.fillRect(pad.l, oy, ox - pad.l, H - pad.b - oy);
  // Q-BR (high p, low s) → rest / PNS dominant
  ctx.fillStyle = 'rgba(0,229,255,0.035)';
  ctx.fillRect(ox, oy, W - pad.r - ox, H - pad.b - oy);

  // ── Sigma grid ───────────────────────────────────────────────
  for (let s = -3; s <= 3; s++) {
    const xg = toX(s), yg = toY(s);
    const isOrigin = s === 0;
    ctx.strokeStyle = isOrigin ? 'rgba(255,255,255,0.15)' : 'rgba(0,120,180,0.11)';
    ctx.lineWidth   = isOrigin ? 1.5 : 0.7;
    ctx.setLineDash(isOrigin ? [] : [3, 6]);
    ctx.beginPath(); ctx.moveTo(xg, pad.t); ctx.lineTo(xg, H - pad.b); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(pad.l, yg); ctx.lineTo(W - pad.r, yg); ctx.stroke();
  }
  ctx.setLineDash([]);

  // ── Stationary-distribution contours (circles in normalised space) ─
  const rings = [
    { r: 1, alpha: 0.22, dash: [] },
    { r: 2, alpha: 0.11, dash: [4, 4] },
    { r: 3, alpha: 0.05, dash: [2, 7] },
  ];
  for (const { r, alpha, dash } of rings) {
    const rx = toX(r)  - ox;
    const ry = oy      - toY(r);      // ry is always positive (canvas flipped)
    ctx.beginPath();
    ctx.ellipse(ox, oy, rx, ry, 0, 0, Math.PI * 2);
    ctx.strokeStyle = `rgba(255,255,255,${alpha})`;
    ctx.lineWidth   = 1;
    ctx.setLineDash(dash);
    ctx.stroke();
    ctx.setLineDash([]);

    ctx.font      = `${6.5 * D}px JetBrains Mono, monospace`;
    ctx.fillStyle = `rgba(120,160,185,${alpha * 1.6})`;
    ctx.textAlign = 'left';
    ctx.fillText(`${r}σ`, ox + rx + 3 * D, oy - 2 * D);
  }

  // ── Asymmetric OU vector field ────────────────────────────────
  // In σ-normalised coordinates the drift speed ratio is ap/as
  // (vagal is typically faster → arrows tilt strongly left/right near x-axis)
  // ── Vector field: colour + length encode speed ────────────────
  const pr_n    = analysis.params;
  const gammaPS = (pr_n.a_ps || 0) * sigS / sigP;
  const gammaSP = (pr_n.a_sp || 0) * sigP / sigS;

  // Pass 1 — find max speed for normalisation
  const GN = 14;
  let maxSpd = 1e-9;
  for (let i = 0; i <= GN; i++) {
    for (let j = 0; j <= GN; j++) {
      const pv = -AX + i * 2 * AX / GN, sv = -AX + j * 2 * AX / GN;
      const sp = Math.hypot(-pr.ap * pv + gammaPS * sv, gammaSP * pv - pr.as * sv);
      if (sp > maxSpd) maxSpd = sp;
    }
  }

  // Pass 2 — draw colour + length encoded arrows
  const maxArrow = IW / GN * 0.46;
  for (let i = 0; i <= GN; i++) {
    for (let j = 0; j <= GN; j++) {
      const pv  = -AX + i * 2 * AX / GN;
      const sv  = -AX + j * 2 * AX / GN;
      const dvx = -pr.ap * pv + gammaPS * sv;
      const dvy =  gammaSP * pv - pr.as * sv;
      const spd = Math.hypot(dvx, dvy);
      if (spd < 1e-9) continue;

      // t ∈ [0,1]: perceptually-uniform speed mapping
      const t   = Math.sqrt(spd / maxSpd);
      // Saturating arrow length (very slow = tiny, fast = capped)
      const len = maxArrow * (1 - Math.exp(-spd / (maxSpd * 0.32)));

      // Colour ramp: dark navy → cyan → warm white
      const r = Math.round(t < 0.5 ? 20  + t * 2 * 100 : 120 + (t - 0.5) * 2 * 128);
      const g = Math.round(t < 0.5 ? 70  + t * 2 * 165 : 235 + (t - 0.5) * 2 * 18 );
      const b = Math.round(t < 0.5 ? 210 - t * 2 * 55  : 155 - (t - 0.5) * 2 * 130);
      const a = 0.10 + t * 0.5;

      // Unit vector (canvas y-axis flipped)
      const ux = dvx / spd, uy = -dvy / spd;
      const ax = toX(pv), ay = toY(sv);
      const ex = ax + ux * len, ey = ay + uy * len;

      // Shaft
      ctx.strokeStyle = `rgba(${r},${g},${b},${a})`;
      ctx.lineWidth   = 1.0;
      ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(ex, ey); ctx.stroke();

      // Solid triangle arrowhead
      const hs = Math.max(D, len * 0.25), hw = hs * 0.25;
      ctx.fillStyle = `rgba(${r},${g},${b},${a})`;
      ctx.beginPath();
      ctx.moveTo(ex + ux * hs * 0.55,  ey + uy * hs * 0.55 );
      ctx.lineTo(ex - ux * hs - uy * hw, ey - uy * hs + ux * hw);
      ctx.lineTo(ex - ux * hs + uy * hw, ey - uy * hs - ux * hw);
      ctx.closePath(); ctx.fill();
    }
  }

  // ── Nullclines ───────────────────────────────────────────────
  // dp̃/dt = 0  →  s̃ = (ap / γps) · p̃   (p-nullcline, cyan)
  // ds̃/dt = 0  →  s̃ = (γsp / as) · p̃   (s-nullcline, orange)
  function drawNullcline(slopeS_over_P, color) {
    // s̃ = slope · p̃
    ctx.setLineDash([5, 4]);
    ctx.lineWidth = 1.3*D;
    ctx.strokeStyle = color;
    ctx.beginPath();
    // Clamp endpoints to the visible AX range
    const p1 = -AX, s1 = clamp(slopeS_over_P * p1, -AX, AX);
    const p2 =  AX, s2 = clamp(slopeS_over_P * p2, -AX, AX);
    ctx.moveTo(toX(p1), toY(s1));
    ctx.lineTo(toX(p2), toY(s2));
    ctx.stroke();
    ctx.setLineDash([]);
  }

  if (Math.abs(gammaPS) > 0.01) {
    drawNullcline(pr.ap / gammaPS, 'rgba(0,229,255,0.55)');
    // Label
    ctx.font = `${6.5*D}px JetBrains Mono, monospace`;
    ctx.fillStyle = 'rgba(0,229,255,0.55)';
    ctx.textAlign = 'left';
    ctx.fillText('dp/dt=0', toX(AX*0.55), toY(pr.ap/gammaPS * AX*0.55) - 5*D);
  }
  if (Math.abs(gammaSP) > 0.01 || true) {
    const slopeDS = Math.abs(pr.as) > 1e-6 ? gammaSP / pr.as : 0;
    drawNullcline(slopeDS, 'rgba(255,109,0,0.55)');
    ctx.font = `${6.5*D}px JetBrains Mono, monospace`;
    ctx.fillStyle = 'rgba(255,109,0,0.55)';
    ctx.textAlign = 'right';
    ctx.fillText('ds/dt=0', toX(-AX*0.45), toY(slopeDS * -AX*0.45) - 5*D);
  }

  // ── Equilibrium & attractor type ─────────────────────────────
  const tr_sys   = -(pr.ap + pr.as);
  const det_sys  = pr.ap*pr.as - gammaPS*gammaSP;
  const disc_sys = tr_sys*tr_sys/4 - det_sys;
  const attrType = disc_sys > 0 ? 'STABLE NODE' : disc_sys < -0.01 ? 'STABLE SPIRAL' : 'STABLE FOCUS';

  ctx.font      = `${7.5*D}px JetBrains Mono, monospace`;
  ctx.fillStyle = 'rgba(0,230,118,0.65)';
  ctx.textAlign = 'center';
  ctx.fillText(`⊕ ${attrType}`, ox, oy - 14*D);
  ctx.fillText(
    `coupling: aₚₛ=${(pr_n.a_ps||0).toFixed(2)} aₛₚ=${(pr_n.a_sp||0).toFixed(2)} Hz`,
    (pad.l + W - pad.r) / 2, 46*D
  );

  // ── Trajectory – Kirsanov-style glowing tail ─────────────────
  const N         = filt.length;
  const TAIL      = Math.min(N, 90);          // recent window rendered as bright trail
  const tailStart = Math.max(0, N - TAIL);

  // Full-history ghost — very faint, drawn only when there's history beyond the tail
  if (tailStart > 2) {
    ctx.save();
    ctx.strokeStyle = 'rgba(30,80,140,1)';
    ctx.globalAlpha = 0.06;
    ctx.lineWidth   = 0.8 * D;
    ctx.beginPath();
    ctx.moveTo(toX(normP[0]), toY(normS[0]));
    for (let i = 1; i <= tailStart; i++) ctx.lineTo(toX(normP[i]), toY(normS[i]));
    ctx.stroke();
    ctx.restore();
  }

  // Outer glow pass — wide, faint cyan-green haze
  ctx.save();
  ctx.lineWidth   = 10 * D;
  ctx.lineJoin    = 'round';
  ctx.lineCap     = 'round';
  ctx.strokeStyle = 'rgba(0,215,140,0.055)';
  ctx.beginPath();
  ctx.moveTo(toX(normP[tailStart]), toY(normS[tailStart]));
  for (let i = tailStart + 1; i < N; i++) ctx.lineTo(toX(normP[i]), toY(normS[i]));
  ctx.stroke();

  // Mid glow pass
  ctx.lineWidth   = 5 * D;
  ctx.strokeStyle = 'rgba(0,235,155,0.10)';
  ctx.beginPath();
  ctx.moveTo(toX(normP[tailStart]), toY(normS[tailStart]));
  for (let i = tailStart + 1; i < N; i++) ctx.lineTo(toX(normP[i]), toY(normS[i]));
  ctx.stroke();
  ctx.restore();

  // Core line — per-segment colour fading dim-blue → bright cyan-green
  for (let i = tailStart + 1; i < N; i++) {
    const tNorm = (i - tailStart) / TAIL;          // 0 = oldest in tail, 1 = newest
    const alpha = 0.15 + 0.78 * tNorm;
    ctx.strokeStyle = `rgba(${Math.round(10 + tNorm * 30)},${Math.round(110 + tNorm * 120)},${Math.round(225 - tNorm * 130)},${alpha})`;
    ctx.lineWidth   = (0.8 + tNorm * 1.3) * D;
    ctx.beginPath();
    ctx.moveTo(toX(normP[i - 1]), toY(normS[i - 1]));
    ctx.lineTo(toX(normP[i]),     toY(normS[i]));
    ctx.stroke();
  }

  // ── Directional chevrons — tail only, sparser ────────────────
  const chevEvery = Math.max(1, Math.floor(TAIL / 7));
  for (let i = tailStart + chevEvery; i < N; i += chevEvery) {
    const dx = normP[i] - normP[i - 1], dy = normS[i] - normS[i - 1];
    const dm = Math.hypot(dx, dy);
    if (dm < 1e-5) continue;
    const ux = dx / dm, uy = -dy / dm;
    const cx2 = toX(normP[i]), cy2 = toY(normS[i]);
    const cs   = 5 * D;
    const tNorm = (i - tailStart) / TAIL;
    ctx.fillStyle = `rgba(${Math.round(10 + tNorm * 30)},${Math.round(110 + tNorm * 120)},${Math.round(225 - tNorm * 130)},${0.50 + tNorm * 0.45})`;
    ctx.beginPath();
    ctx.moveTo(cx2 + ux * cs,                      cy2 + uy * cs);
    ctx.lineTo(cx2 - ux * cs - uy * cs * 0.55,     cy2 - uy * cs + ux * cs * 0.55);
    ctx.lineTo(cx2 - ux * cs + uy * cs * 0.55,     cy2 - uy * cs - ux * cs * 0.55);
    ctx.closePath(); ctx.fill();
  }

  // ── Current frame marker — multi-ring glow ───────────────────
  const fi  = clamp(state.playback.frame, 0, N - 1);
  const cpx = toX(normP[fi]), cpy = toY(normS[fi]);

  // Outer halo
  const halo1 = ctx.createRadialGradient(cpx, cpy, 0, cpx, cpy, 30 * D);
  halo1.addColorStop(0, 'rgba(0,255,160,0.20)');
  halo1.addColorStop(1, 'rgba(0,255,160,0)');
  ctx.beginPath(); ctx.arc(cpx, cpy, 30 * D, 0, Math.PI * 2);
  ctx.fillStyle = halo1; ctx.fill();

  // Inner halo
  const halo2 = ctx.createRadialGradient(cpx, cpy, 0, cpx, cpy, 13 * D);
  halo2.addColorStop(0, 'rgba(120,255,210,0.50)');
  halo2.addColorStop(1, 'rgba(0,230,130,0)');
  ctx.beginPath(); ctx.arc(cpx, cpy, 13 * D, 0, Math.PI * 2);
  ctx.fillStyle = halo2; ctx.fill();

  // Bright core
  ctx.beginPath(); ctx.arc(cpx, cpy, 5 * D, 0, Math.PI * 2);
  ctx.fillStyle = '#ffffff'; ctx.fill();
  ctx.beginPath(); ctx.arc(cpx, cpy, 3 * D, 0, Math.PI * 2);
  ctx.fillStyle = '#00ffaa'; ctx.fill();

  // ── Equilibrium — glowing ring ───────────────────────────────
  const eqGrd = ctx.createRadialGradient(ox, oy, 0, ox, oy, 16 * D);
  eqGrd.addColorStop(0, 'rgba(255,255,255,0.30)');
  eqGrd.addColorStop(1, 'rgba(255,255,255,0)');
  ctx.beginPath(); ctx.arc(ox, oy, 16 * D, 0, Math.PI * 2);
  ctx.fillStyle = eqGrd; ctx.fill();
  ctx.beginPath(); ctx.arc(ox, oy, 6 * D, 0, Math.PI * 2);
  ctx.strokeStyle = 'rgba(255,255,255,0.55)'; ctx.lineWidth = 1.5 * D; ctx.stroke();
  ctx.beginPath(); ctx.arc(ox, oy, 2.5 * D, 0, Math.PI * 2);
  ctx.fillStyle = 'rgba(255,255,255,0.85)'; ctx.fill();

  // ── Instantaneous velocity arrow at current state ─────────────
  const cvx   = -pr.ap * normP[fi] + gammaPS * normS[fi];
  const cvy   =  gammaSP * normP[fi] - pr.as * normS[fi];
  const cvSpd = Math.hypot(cvx, cvy) + 1e-9;
  const vaLen = Math.min(IW * 0.23, (cvSpd / maxSpd) * IW * 0.32);
  const vux   = cvx / cvSpd, vuy = -cvy / cvSpd;   // canvas y-flip
  const vex   = cpx + vux * vaLen, vey = cpy + vuy * vaLen;

  // Glow halo
  const haloR = Math.max(vaLen * 1.0, 20 * D);
  const vHalo = ctx.createRadialGradient(cpx, cpy, 0, cpx, cpy, haloR);
  vHalo.addColorStop(0, 'rgba(0,255,140,0.20)');
  vHalo.addColorStop(1, 'rgba(0,255,140,0)');
  ctx.beginPath(); ctx.arc(cpx, cpy, haloR, 0, Math.PI * 2);
  ctx.fillStyle = vHalo; ctx.fill();

  // Shaft
  ctx.strokeStyle = '#00ff88';
  ctx.lineWidth   = 2.8 * D;
  ctx.lineJoin    = 'round';
  ctx.beginPath(); ctx.moveTo(cpx, cpy); ctx.lineTo(vex, vey); ctx.stroke();

  // Solid arrowhead
  const vaHs = 9 * D, vaHw = 5 * D;
  ctx.fillStyle = '#00ff88';
  ctx.beginPath();
  ctx.moveTo(vex + vux * vaHs * 0.55,        vey + vuy * vaHs * 0.55);
  ctx.lineTo(vex - vux * vaHs - vuy * vaHw,  vey - vuy * vaHs + vux * vaHw);
  ctx.lineTo(vex - vux * vaHs + vuy * vaHw,  vey - vuy * vaHs - vux * vaHw);
  ctx.closePath(); ctx.fill();

  // Speed annotation
  ctx.font      = `${7 * D}px JetBrains Mono, monospace`;
  ctx.fillStyle = 'rgba(0,255,140,0.62)';
  ctx.textAlign = 'left';
  ctx.fillText(`v = ${cvSpd.toFixed(2)}`, cpx + 11 * D, cpy - 11 * D);

  // ── Animated flow particles ───────────────────────────────────
  for (const pt of phaseAnim.particles) {
    if (pt.trail.length < 2) continue;
    const tLen = pt.trail.length;
    for (let ti = 1; ti < tLen; ti++) {
      const alpha = (ti / tLen) * 0.30;
      ctx.strokeStyle = `rgba(155,210,255,${alpha})`;
      ctx.lineWidth   = 1;
      ctx.beginPath();
      ctx.moveTo(toX(pt.trail[ti - 1].p), toY(pt.trail[ti - 1].s));
      ctx.lineTo(toX(pt.trail[ti].p),     toY(pt.trail[ti].s));
      ctx.stroke();
    }
  }

  // ── Quadrant labels ──────────────────────────────────────────
  const qLabel = (x, y, title, sub, color) => {
    ctx.textAlign   = 'center';
    ctx.font        = `bold ${8 * D}px Orbitron, monospace`;
    ctx.fillStyle   = color;
    ctx.fillText(title, x, y);
    ctx.font        = `${7 * D}px JetBrains Mono, monospace`;
    ctx.fillStyle   = color.replace(/[\d.]+\)$/, '0.18)');
    ctx.fillText(sub, x, y + 10 * D);
  };

  const qx = (side) => side === 'r' ? ox + (W - pad.r - ox) * 0.52 : pad.l + (ox - pad.l) * 0.48;
  const qy = (side) => side === 't' ? pad.t + (oy - pad.t) * 0.30  : oy + (H - pad.b - oy) * 0.30;

  qLabel(qx('r'), qy('t'), 'CO-ACTIVATION', 'high p & s',     'rgba(255,214,0,0.38)');
  qLabel(qx('l'), qy('t'), 'STRESS',        'SNS dominant',   'rgba(255,70,70,0.38)');
  qLabel(qx('l'), qy('b'), 'WITHDRAWAL',    'both suppressed','rgba(140,140,185,0.32)');
  qLabel(qx('r'), qy('b'), 'REST',          'PNS dominant',   'rgba(0,229,255,0.38)');

  // ── Axis labels & ticks ──────────────────────────────────────
  ctx.font      = `${9 * D}px Orbitron, monospace`;
  ctx.textAlign = 'center';
  ctx.fillStyle = 'rgba(0,229,255,0.70)';
  ctx.fillText('VAGAL p̂(t)  [σ-units]', W / 2, H - 10 * D);

  ctx.save();
  ctx.translate(13 * D, H / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillStyle = 'rgba(255,109,0,0.70)';
  ctx.fillText('ADRENERGIC ŝ(t)  [σ-units]', 0, 0);
  ctx.restore();

  ctx.font      = `${7.5 * D}px JetBrains Mono, monospace`;
  ctx.fillStyle = '#2a4460';
  for (let s = -3; s <= 3; s++) {
    if (s === 0) continue;
    ctx.textAlign = 'center';
    ctx.fillText(s, toX(s), H - pad.b + 13 * D);
    ctx.textAlign = 'right';
    ctx.fillText(s, pad.l - 5 * D, toY(s) + 3 * D);
  }

  // ── Title & legend ───────────────────────────────────────────
  ctx.font      = `${10 * D}px Orbitron, monospace`;
  ctx.fillStyle = 'rgba(0,229,255,0.72)';
  ctx.textAlign = 'left';
  ctx.fillText('ANS PHASE SPACE', pad.l, 20 * D);

  ctx.font      = `${7.5 * D}px JetBrains Mono, monospace`;
  ctx.fillStyle = '#2a4460';
  ctx.fillText(
    `N=${N} beats  ·  τ_p=${(1/pr.ap).toFixed(2)}s  τ_s=${(1/pr.as).toFixed(1)}s  ` +
    `σ_p=${sigP.toFixed(3)}  σ_s=${sigS.toFixed(3)}`,
    pad.l, 34 * D
  );

  // Time colour bar
  const lgX = W - pad.r - 72 * D, lgY = 16 * D;
  const lg  = ctx.createLinearGradient(lgX, 0, lgX + 72 * D, 0);
  lg.addColorStop(0,   'rgba(20,110,235,0.9)');
  lg.addColorStop(0.5, 'rgba(70,200,90,0.9)');
  lg.addColorStop(1,   'rgba(240,90,20,0.9)');
  ctx.fillStyle = lg;
  ctx.fillRect(lgX, lgY, 72 * D, 5 * D);
  ctx.font      = `${6.5 * D}px JetBrains Mono, monospace`;
  ctx.fillStyle = '#2a4460';
  ctx.textAlign = 'left';  ctx.fillText('start', lgX, lgY + 14 * D);
  ctx.textAlign = 'right'; ctx.fillText('end',   lgX + 72 * D, lgY + 14 * D);
}

// ============================================================
//  CANVAS CHARTS
// ============================================================
function dpr() { return Math.min(window.devicePixelRatio || 1, 2); }

function sizeCanvas(canvas) {
  const D = dpr();
  const parent = canvas.parentElement;
  const cs = getComputedStyle(parent);
  const w = parent.clientWidth
    - parseFloat(cs.paddingLeft  || '0')
    - parseFloat(cs.paddingRight || '0');
  // Cache the original logical height ONCE — setting canvas.height rewrites
  // the HTML attribute, so reading it again on next resize would compound.
  if (!canvas.dataset.logicalHeight) {
    canvas.dataset.logicalHeight = canvas.getAttribute('height') || '160';
  }
  const h = parseInt(canvas.dataset.logicalHeight);
  if (!w || w < 20) return;
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
  const padL = 50 * D, padR = 20 * D, padT = 16 * D, padB = 36 * D;
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

  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = 'rgba(6,15,30,0.5)';
  ctx.fillRect(0, 0, W, H);

  const { rr, td } = analysis;
  const N = rr.length;
  if (N < 3) return;

  const rms   = rr.map(x => x * 1000);
  const xPts  = rms.slice(0, N - 1);   // RR_n
  const yPts  = rms.slice(1);           // RR_{n+1}

  // True centroid – both axes centre on mean RR
  const cx = mean(xPts), cy = mean(yPts);

  // Symmetric display window around the centroid
  const allVals = xPts.concat(yPts);
  const spread  = Math.max(...allVals) - Math.min(...allVals);
  const half    = Math.max(80, spread * 0.58);

  const pad   = 44 * D;
  const plotW = W - 2 * pad;
  const plotH = H - 2 * pad;

  // Coordinate helpers – both axes use the SAME mapping centred on (cx, cy)
  const toX = v => pad + (v - (cx - half)) / (2 * half) * plotW;
  const toY = v => H - pad - (v - (cy - half)) / (2 * half) * plotH;

  const pcx = toX(cx), pcy = toY(cy);   // centroid in pixel space

  // ── Grid ──────────────────────────────────────────────────────
  ctx.strokeStyle = 'rgba(0,100,150,0.12)';
  ctx.lineWidth = 0.7;
  for (let i = 0; i <= 4; i++) {
    const x = pad + i / 4 * plotW, y = pad + i / 4 * plotH;
    ctx.beginPath(); ctx.moveTo(x, pad); ctx.lineTo(x, H - pad); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(pad, y); ctx.lineTo(W - pad, y); ctx.stroke();
  }

  // ── Line of Identity (RRn = RRn+1) centred on cloud ──────────
  ctx.strokeStyle = 'rgba(130,160,180,0.22)';
  ctx.lineWidth   = 1;
  ctx.setLineDash([5, 5]);
  ctx.beginPath();
  ctx.moveTo(toX(cx - half), toY(cy - half));
  ctx.lineTo(toX(cx + half), toY(cy + half));
  ctx.stroke();
  ctx.setLineDash([]);

  // ── SD1 / SD2 ellipse – properly centred on (pcx, pcy) ───────
  const scl   = plotW / (2 * half);          // pixels per ms
  const sd1px = td.sd1 * 1000 * scl * Math.SQRT2;
  const sd2px = td.sd2 * 1000 * scl * Math.SQRT2;

  ctx.save();
  ctx.translate(pcx, pcy);
  ctx.rotate(-Math.PI / 4);

  ctx.beginPath();
  ctx.ellipse(0, 0, sd2px, sd1px, 0, 0, Math.PI * 2);
  ctx.strokeStyle = 'rgba(255,214,0,0.50)';
  ctx.lineWidth   = 1.5 * D;
  ctx.stroke();
  ctx.fillStyle   = 'rgba(255,214,0,0.04)';
  ctx.fill();

  // SD-axis dashes
  ctx.lineWidth = 0.8;
  ctx.setLineDash([2, 4]);
  ctx.strokeStyle = 'rgba(0,229,255,0.30)';
  ctx.beginPath(); ctx.moveTo(0, -sd1px); ctx.lineTo(0, sd1px); ctx.stroke();
  ctx.strokeStyle = 'rgba(255,109,0,0.30)';
  ctx.beginPath(); ctx.moveTo(-sd2px, 0); ctx.lineTo(sd2px, 0); ctx.stroke();
  ctx.setLineDash([]);
  ctx.restore();

  // ── Point cloud – colour encodes time (blue → orange) ────────
  const nPts = xPts.length;
  for (let i = 0; i < nPts; i++) {
    const t = i / nPts;
    const r = Math.round(60  + t * 190);
    const g = Math.round(80  + t * 20);
    const b = Math.round(210 - t * 140);
    ctx.beginPath();
    ctx.arc(toX(xPts[i]), toY(yPts[i]), 1.8 * D, 0, Math.PI * 2);
    ctx.fillStyle = `rgba(${r},${g},${b},0.55)`;
    ctx.fill();
  }

  // ── Centroid marker ───────────────────────────────────────────
  ctx.beginPath();
  ctx.arc(pcx, pcy, 4 * D, 0, Math.PI * 2);
  ctx.fillStyle = 'rgba(255,214,0,0.90)';
  ctx.fill();

  // ── Axis labels ───────────────────────────────────────────────
  ctx.font      = `${8 * D}px JetBrains Mono, monospace`;
  ctx.fillStyle = '#4a6080';
  ctx.textAlign = 'center';
  ctx.fillText('RRₙ (ms)', W / 2, H - 4 * D);
  ctx.save();
  ctx.translate(11 * D, H / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText('RRₙ₊₁ (ms)', 0, 0);
  ctx.restore();

  // Tick values at ± half * 0.7
  ctx.font      = `${7 * D}px JetBrains Mono, monospace`;
  ctx.fillStyle = '#2e4a62';
  [cx - half * 0.7, cx, cx + half * 0.7].forEach(v => {
    ctx.textAlign = 'center';
    ctx.fillText(Math.round(v), toX(v), H - pad + 12 * D);
    ctx.textAlign = 'right';
    ctx.fillText(Math.round(v), pad - 3 * D, toY(v) + 3 * D);
  });

  // ── SD legend ─────────────────────────────────────────────────
  ctx.textAlign = 'left';
  ctx.font      = `${8 * D}px JetBrains Mono, monospace`;
  ctx.fillStyle = 'rgba(0,229,255,0.65)';
  ctx.fillText(`SD1: ${(td.sd1 * 1000).toFixed(1)} ms`, pad, H - pad + 14 * D);
  ctx.fillStyle = 'rgba(255,109,0,0.65)';
  ctx.fillText(`SD2: ${(td.sd2 * 1000).toFixed(1)} ms`, pad + 84 * D, H - pad + 14 * D);

  document.getElementById('poincare-info').textContent =
    `SD1 ${(td.sd1 * 1000).toFixed(1)} | SD2 ${(td.sd2 * 1000).toFixed(1)} ms`;
}

// ── Plasma-like colormap ──────────────────────────────────────
function heatmapColor(t) {
  t = clamp(t, 0, 1);
  if (t < 0.25) {
    const f = t / 0.25;
    return [Math.round(13+f*106), Math.round(8), Math.round(135-f*10)];
  } else if (t < 0.50) {
    const f = (t-0.25)/0.25;
    return [Math.round(119+f*102), Math.round(8+f*40), Math.round(125-f*45)];
  } else if (t < 0.75) {
    const f = (t-0.50)/0.25;
    return [Math.round(221+f*32), Math.round(48+f*111), Math.round(80-f*70)];
  } else {
    const f = (t-0.75)/0.25;
    return [Math.round(253), Math.round(159+f*72), Math.round(10+f*21)];
  }
}

// ── Analytic Morlet wavelet transform ─────────────────────────
// Returns: Array[nFreqs] of Float32Array[nSamples]
function computeWavelet(rrU, fs, freqs) {
  const n = rrU.length;
  const N = nextPow2(n * 2);
  const f0 = 6; // Morlet cycles per wavelet
  const mu = mean(Array.from(rrU));

  // FFT of mean-removed signal (converted to ms)
  const re = new Float64Array(N), im = new Float64Array(N);
  for (let i = 0; i < n; i++) re[i] = (rrU[i] - mu) * 1000;
  fft(re, im);

  return freqs.map(freq => {
    const s    = f0 / (2 * Math.PI * freq);   // scale (seconds)
    const norm = Math.pow(Math.PI, -0.25) * Math.sqrt(2 * Math.PI * s / N * fs);
    const wRe  = new Float64Array(N), wIm = new Float64Array(N);

    for (let i = 1; i <= N / 2; i++) {
      const omega = 2 * Math.PI * i * fs / N;
      const arg   = s * omega - f0;
      const psi   = norm * Math.exp(-0.5 * arg * arg);
      wRe[i] = re[i] * psi;
      wIm[i] = im[i] * psi;
    }
    ifft(wRe, wIm);

    const pow = new Float32Array(n);
    for (let i = 0; i < n; i++) pow[i] = wRe[i]*wRe[i] + wIm[i]*wIm[i];
    return pow;
  });
}

// ── Build cached ImageData for wavelet heatmap ────────────────
function buildWaveletImage(analysis, IW, IH) {
  const { rr, times } = analysis;
  const N  = rr.length;
  const tEnd = times[N-1];
  const fs = 4.0;
  const nS = Math.max(64, Math.floor(tEnd * fs));

  // Resample to uniform grid
  const rrU  = new Float64Array(nS);
  const cumT = [0];
  for (const v of rr) cumT.push(cumT[cumT.length-1] + v);
  let j = 0;
  for (let i = 0; i < nS; i++) {
    const ti = i / fs;
    while (j < N-1 && cumT[j+1] <= ti) j++;
    const fr = clamp((ti - cumT[j]) / Math.max(1e-9, cumT[j+1]-cumT[j]), 0, 1);
    rrU[i] = rr[clamp(j,0,N-1)] * (1-fr) + rr[clamp(j+1,0,N-1)] * fr;
  }

  // 40 log-spaced frequencies from 0.04 to 0.40 Hz
  const nF = 40;
  const freqs = Array.from({length: nF}, (_, i) =>
    0.04 * Math.pow(10, i/(nF-1) * Math.log10(0.40/0.04)));

  const power = computeWavelet(rrU, fs, freqs);

  let maxP = 1e-20;
  for (const row of power) for (const v of row) if (isFinite(v) && v > maxP) maxP = v;

  const img = new ImageData(IW, IH);
  const d = img.data;

  for (let iy = 0; iy < IH; iy++) {
    const frac_f = 1 - iy / (IH - 1);
    const logF   = Math.log10(0.04) + frac_f * Math.log10(0.40/0.04);
    const fTgt   = Math.pow(10, logF);
    // Nearest freq bin
    let fi = 0, minD = Infinity;
    for (let k = 0; k < nF; k++) {
      const dd = Math.abs(Math.log(freqs[k]) - Math.log(fTgt));
      if (dd < minD) { minD = dd; fi = k; }
    }
    const row = power[fi];
    for (let ix = 0; ix < IW; ix++) {
      const ti  = Math.round(ix / (IW-1) * (row.length-1));
      const val = Math.pow(clamp(row[ti] / maxP, 0, 1), 0.42); // gamma for visual balance
      const [r,g,b] = heatmapColor(val);
      const idx = (iy*IW + ix)*4;
      d[idx]=r; d[idx+1]=g; d[idx+2]=b; d[idx+3]=225;
    }
  }
  return img;
}

function drawWavelet(analysis) {
  const canvas = document.getElementById('psd-canvas');
  sizeCanvas(canvas);
  const ctx = canvas.getContext('2d');
  const D   = dpr();
  const W = canvas.width, H = canvas.height;
  const padL = 46*D, padR = 16*D, padT = 16*D, padB = 32*D;
  const IW = Math.max(4, Math.round(W - padL - padR));
  const IH = Math.max(4, Math.round(H - padT - padB));

  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = 'rgba(6,15,30,0.6)';
  ctx.fillRect(0, 0, W, H);

  // ── Build or retrieve cached heatmap ────────────────────────
  if (!analysis._wvCache || analysis._wvCache.w !== W || analysis._wvCache.h !== H) {
    analysis._wvCache = { img: buildWaveletImage(analysis, IW, IH), w: W, h: H };
  }
  ctx.putImageData(analysis._wvCache.img, Math.round(padL), Math.round(padT));

  // ── Frequency → pixel helper (log scale, top = high freq) ───
  const fMin = Math.log10(0.04), fMax = Math.log10(0.40);
  const freqToY = f => padT + IH * (1 - (Math.log10(f) - fMin) / (fMax - fMin));

  // ── Band boundary lines ──────────────────────────────────────
  ctx.setLineDash([3, 4]);
  ctx.lineWidth = 1;
  [[0.04,'rgba(255,109,0,0.35)'], [0.15,'rgba(0,229,255,0.35)'], [0.40,'rgba(0,229,255,0.20)']].forEach(([f, col]) => {
    const y = freqToY(f);
    ctx.strokeStyle = col;
    ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(W-padR, y); ctx.stroke();
  });
  ctx.setLineDash([]);

  // ── Cone of Influence shading (edge-effect region) ──────────
  const tEnd  = analysis.times[analysis.times.length-1];
  const toCOI = (f) => 6 / (2*Math.PI*f) * Math.sqrt(2);
  const freqBins = [0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.40];
  ctx.fillStyle = 'rgba(2,9,18,0.38)';
  for (const f of freqBins) {
    const coi  = toCOI(f);
    const y1   = freqToY(f * 1.18), y0 = freqToY(f * 0.84);
    const xCoi = padL + (coi / tEnd) * IW;
    const xRev = (W - padR) - (coi / tEnd) * IW;
    if (xCoi > padL)   ctx.fillRect(padL,  Math.min(y1,y0), xCoi - padL, Math.abs(y1-y0));
    if (xRev < W-padR) ctx.fillRect(xRev, Math.min(y1,y0), (W-padR)-xRev, Math.abs(y1-y0));
  }

  // ── Playback cursor ──────────────────────────────────────────
  if (analysis.filter.length > 0) {
    const fi = clamp(state.playback.frame, 0, analysis.filter.length-1);
    const xc = padL + (analysis.filter[fi].t / tEnd) * IW;
    ctx.strokeStyle = 'rgba(255,255,255,0.45)';
    ctx.lineWidth   = 1.5;
    ctx.beginPath(); ctx.moveTo(xc, padT); ctx.lineTo(xc, padT+IH); ctx.stroke();
  }

  // ── Frequency axis labels ────────────────────────────────────
  ctx.font      = `${8*D}px JetBrains Mono, monospace`;
  ctx.fillStyle = '#4a6080';
  ctx.textAlign = 'right';
  [0.04, 0.08, 0.15, 0.25, 0.40].forEach(f => {
    const y = freqToY(f);
    if (y >= padT && y <= padT+IH) ctx.fillText(f.toFixed(2), padL-3*D, y+3*D);
  });

  // ── Band name overlays ───────────────────────────────────────
  ctx.textAlign = 'left';
  ctx.font = `${8*D}px Orbitron, monospace`;
  ctx.fillStyle = 'rgba(255,109,0,0.60)';
  ctx.fillText('LF', padL+4*D, freqToY(0.095)-2*D);
  ctx.fillStyle = 'rgba(0,229,255,0.60)';
  ctx.fillText('HF', padL+4*D, freqToY(0.27)-2*D);

  // ── Time axis ────────────────────────────────────────────────
  ctx.font      = `${8*D}px JetBrains Mono, monospace`;
  ctx.fillStyle = '#4a6080';
  ctx.textAlign = 'center';
  for (let i = 0; i <= 5; i++) {
    const xT = padL + i/5 * IW;
    ctx.fillText((i/5 * tEnd / 60).toFixed(1)+'m', xT, H-6*D);
  }

  // ── Title ────────────────────────────────────────────────────
  ctx.textAlign = 'left';
  ctx.font = `${7.5*D}px JetBrains Mono, monospace`;
  ctx.fillStyle = '#2a4460';
  ctx.fillText('Hz', padL-3*D, H-6*D);

  // Keep header band power labels from spectral analysis
  document.getElementById('lf-power').textContent   = analysis.spec.lf.toFixed(1);
  document.getElementById('hf-power').textContent   = analysis.spec.hf.toFixed(1);
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
  const apsEl = document.getElementById('par-aps');
  const aspEl = document.getElementById('par-asp');
  if (apsEl) apsEl.textContent = (p.a_ps || 0).toFixed(3) + ' Hz';
  if (aspEl) aspEl.textContent = (p.a_sp || 0).toFixed(3) + ' Hz';
}

function updateLiveState(p, s, delta, hr) {
  document.getElementById('m-p').textContent  = p.toFixed(4);
  document.getElementById('m-s').textContent  = s.toFixed(4);
  document.getElementById('m-d').textContent  = delta.toFixed(4);
  document.getElementById('m-hr').textContent = isNaN(hr) ? '—' : Math.round(hr);

  // Normalise by stationary σ so the bars reflect σ-units, not raw OU values
  // (raw p, s hover near 0 by construction; σ-normalised values are meaningful)
  let pNorm = p, sNorm = s;
  const an = state.currentAnalysis;
  if (an) {
    const pr   = an.params;
    const sigP = pr.sigma_p / Math.sqrt(2 * pr.ap);
    const sigS = pr.sigma_s / Math.sqrt(2 * pr.as);
    pNorm = p / (sigP * 3);   // ±3σ maps to ±100 %
    sNorm = s / (sigS * 3);
  }
  const pPct = clamp(50 + pNorm * 50, 5, 95);
  const sPct = clamp(50 + sNorm * 50, 5, 95);
  document.getElementById('bar-p').style.width = pPct + '%';
  document.getElementById('bar-s').style.width = sPct + '%';

  // Balance: compare normalised activities so it reflects relative dominance
  const relBalance = clamp(50 + (pNorm - sNorm) * 30, 8, 92);
  document.getElementById('bal-para').style.width = Math.min(relBalance,     48) + '%';
  document.getElementById('bal-symp').style.width = Math.min(100-relBalance, 48) + '%';

  const thresh = an ? (an.params.sigma_p / Math.sqrt(2 * an.params.ap)) * 0.5 : 0.05;
  const state_str = (pNorm - sNorm) >  0.4 ? 'VAGAL DOMINANT'
                  : (sNorm - pNorm) >  0.4 ? 'SYMPATHETIC DOMINANT'
                  :                           'BALANCED';
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
  drawOUSim();
  document.getElementById('hm-hr').textContent = Math.round(60 / fdata.mu);
  document.getElementById('hm-p').textContent  = fdata.p.toFixed(3);
  document.getElementById('hm-s').textContent  = fdata.s.toFixed(3);
  document.getElementById('hm-d').textContent  = fdata.delta.toFixed(3);
}

// ============================================================
//  OU IMPULSE-RESPONSE SIMULATOR
// ============================================================
const ouSim = { perturbP: 0, perturbS: 0 };

function perturbANS(branch, sigmaSteps) {
  const an = state.currentAnalysis;
  if (!an) return;
  const pr   = an.params;
  const sigP = pr.sigma_p / Math.sqrt(2 * pr.ap);
  const sigS = pr.sigma_s / Math.sqrt(2 * pr.as);
  if (branch === 'p') ouSim.perturbP += sigmaSteps * sigP;
  else                ouSim.perturbS += sigmaSteps * sigS;
  drawOUSim();
}

function resetOUSim() {
  ouSim.perturbP = 0;
  ouSim.perturbS = 0;
  drawOUSim();
}

function drawOUSim() {
  const canvas = document.getElementById('ou-sim-canvas');
  if (!canvas || !state.currentAnalysis) return;
  sizeCanvas(canvas);
  const ctx = canvas.getContext('2d');
  const D   = dpr();
  const W   = canvas.width, H = canvas.height;

  const an  = state.currentAnalysis;
  const pr  = an.params;
  const fi  = Math.min(state.playback.frame, an.filter.length - 1);
  const f   = an.filter[fi];

  const sigP = pr.sigma_p / Math.sqrt(2 * pr.ap);
  const sigS = pr.sigma_s / Math.sqrt(2 * pr.as);
  const p0   = f.p + ouSim.perturbP;
  const s0   = f.s + ouSim.perturbS;

  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = 'rgba(6,15,30,0.5)';
  ctx.fillRect(0, 0, W, H);

  const elTauP = document.getElementById('ou-tau-p');
  const elTauS = document.getElementById('ou-tau-s');
  if (elTauP) elTauP.textContent = `τ_p = ${(1 / pr.ap).toFixed(2)} s`;
  if (elTauS) elTauS.textContent = `τ_s = ${(1 / pr.as).toFixed(1)} s`;

  // ── Euler-Maruyama simulation setup ──────────────────────────
  const N_PATHS = 22;
  const N_STEPS = 380;
  const horizon = Math.max(20, 5.5 / Math.min(pr.ap, pr.as));
  const dt      = horizon / N_STEPS;
  const sqDt    = Math.sqrt(dt);
  const aps     = pr.a_ps || 0;
  const asp     = pr.a_sp || 0;

  // Box-Muller normal sample
  function randn() {
    return Math.sqrt(-2 * Math.log(Math.random() + 1e-14)) *
           Math.cos(2 * Math.PI * Math.random());
  }

  // Generate coupled Euler-Maruyama paths  [p, s] jointly
  const pathsP = Array.from({ length: N_PATHS }, () => new Float32Array(N_STEPS + 1));
  const pathsS = Array.from({ length: N_PATHS }, () => new Float32Array(N_STEPS + 1));

  for (let k = 0; k < N_PATHS; k++) {
    let p = p0, s = s0;
    pathsP[k][0] = p; pathsS[k][0] = s;
    for (let i = 0; i < N_STEPS; i++) {
      const dp = (-pr.ap * p + aps * s) * dt + pr.sigma_p * sqDt * randn();
      const ds = ( asp * p - pr.as * s) * dt + pr.sigma_s * sqDt * randn();
      p += dp; s += ds;
      pathsP[k][i + 1] = p; pathsS[k][i + 1] = s;
    }
  }

  // Pointwise mean & std across paths
  const meanP = new Float32Array(N_STEPS + 1);
  const meanS = new Float32Array(N_STEPS + 1);
  const stdP  = new Float32Array(N_STEPS + 1);
  const stdS  = new Float32Array(N_STEPS + 1);
  for (let i = 0; i <= N_STEPS; i++) {
    let mp = 0, ms = 0;
    for (let k = 0; k < N_PATHS; k++) { mp += pathsP[k][i]; ms += pathsS[k][i]; }
    meanP[i] = mp / N_PATHS; meanS[i] = ms / N_PATHS;
    let vp = 0, vs = 0;
    for (let k = 0; k < N_PATHS; k++) {
      vp += (pathsP[k][i] - meanP[i]) ** 2;
      vs += (pathsS[k][i] - meanS[i]) ** 2;
    }
    stdP[i] = Math.sqrt(vp / (N_PATHS - 1) + 1e-20);
    stdS[i] = Math.sqrt(vs / (N_PATHS - 1) + 1e-20);
  }

  const halfW = Math.floor(W / 2);

  // ── Inner panel renderer ──────────────────────────────────────
  function renderPanel(x0, pw, paths, mArr, sArr, sigma, initVal, decayRate, noiseAmp, color, rgb, title) {
    const padL = 46 * D, padR = 12 * D, padT = 24 * D, padB = 34 * D;
    const iw = pw - padL - padR, ih = H - padT - padB;
    const ox = x0 + padL, oy = padT;

    // Visible range: ±AY_σ centred on 0, always includes the IC
    const AY = Math.max(3.0, Math.abs(initVal / sigma) + 1.3);
    const toXp = t => ox + (t / horizon) * iw;
    const toYp = v => oy + ih * (0.5 - (v / sigma) / (2 * AY));
    const inY  = y => y >= oy - 1 && y <= oy + ih + 1;
    const cy   = v => clamp(toYp(v), oy, oy + ih);

    // Grid
    ctx.strokeStyle = 'rgba(0,70,110,0.14)';
    ctx.lineWidth = 0.5;
    for (let i = 0; i <= 4; i++) {
      const y = oy + (i / 4) * ih;
      ctx.beginPath(); ctx.moveTo(ox, y); ctx.lineTo(ox + iw, y); ctx.stroke();
    }

    // σ reference lines
    for (const s of [-2, -1, 0, 1, 2]) {
      const y = toYp(s * sigma);
      if (!inY(y)) continue;
      ctx.strokeStyle = s === 0 ? 'rgba(255,255,255,0.14)' : 'rgba(0,100,145,0.11)';
      ctx.lineWidth   = s === 0 ? 1.0 : 0.5;
      ctx.setLineDash(s === 0 ? [] : [2, 7]);
      ctx.beginPath(); ctx.moveTo(ox, y); ctx.lineTo(ox + iw, y); ctx.stroke();
      ctx.setLineDash([]);
    }

    // Stationary ±σ noise-floor band
    const yP1 = toYp(sigma), yN1 = toYp(-sigma);
    ctx.fillStyle = `rgba(${rgb},0.04)`;
    ctx.fillRect(ox, Math.min(yP1, yN1), iw, Math.abs(yP1 - yN1));

    // Individual EM paths (clipped to visible range)
    for (let k = 0; k < N_PATHS; k++) {
      ctx.beginPath();
      let penDown = false;
      for (let i = 0; i <= N_STEPS; i++) {
        const x = toXp(i * dt), y = toYp(paths[k][i]);
        if (inY(y)) {
          penDown ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
          penDown = true;
        } else { penDown = false; }
      }
      ctx.strokeStyle = `rgba(${rgb},0.13)`;
      ctx.lineWidth   = 0.75;
      ctx.stroke();
    }

    // 2σ envelope (shaded)
    ctx.beginPath();
    for (let i = 0; i <= N_STEPS; i++) ctx[i ? 'lineTo' : 'moveTo'](toXp(i * dt), cy(mArr[i] + 2 * sArr[i]));
    for (let i = N_STEPS; i >= 0;   i--) ctx.lineTo(toXp(i * dt), cy(mArr[i] - 2 * sArr[i]));
    ctx.closePath();
    ctx.fillStyle = `rgba(${rgb},0.055)`;
    ctx.fill();

    // 1σ envelope (shaded)
    ctx.beginPath();
    for (let i = 0; i <= N_STEPS; i++) ctx[i ? 'lineTo' : 'moveTo'](toXp(i * dt), cy(mArr[i] + sArr[i]));
    for (let i = N_STEPS; i >= 0;   i--) ctx.lineTo(toXp(i * dt), cy(mArr[i] - sArr[i]));
    ctx.closePath();
    ctx.fillStyle = `rgba(${rgb},0.11)`;
    ctx.fill();

    // Analytical deterministic mean: initVal · exp(−decayRate · t)  [dashed, for reference]
    ctx.beginPath();
    for (let i = 0; i <= N_STEPS; i++) {
      const t = i * dt;
      ctx[i ? 'lineTo' : 'moveTo'](toXp(t), cy(initVal * Math.exp(-decayRate * t)));
    }
    ctx.strokeStyle = `rgba(${rgb},0.32)`;
    ctx.lineWidth   = 1.0 * D;
    ctx.setLineDash([4 * D, 3 * D]);
    ctx.stroke();
    ctx.setLineDash([]);

    // Empirical mean path (bold)
    ctx.beginPath();
    for (let i = 0; i <= N_STEPS; i++) ctx[i ? 'lineTo' : 'moveTo'](toXp(i * dt), cy(mArr[i]));
    ctx.strokeStyle = color;
    ctx.lineWidth   = 2.4 * D;
    ctx.stroke();

    // Time-constant marker τ = 1/decayRate
    const tau  = 1 / decayRate;
    const xTau = toXp(tau);
    if (tau < horizon) {
      ctx.strokeStyle = 'rgba(255,255,255,0.18)';
      ctx.lineWidth   = 1;
      ctx.setLineDash([2, 5]);
      ctx.beginPath(); ctx.moveTo(xTau, oy); ctx.lineTo(xTau, oy + ih); ctx.stroke();
      ctx.setLineDash([]);

      ctx.font      = `${6.5 * D}px JetBrains Mono, monospace`;
      ctx.fillStyle = 'rgba(255,255,255,0.32)';
      ctx.textAlign = 'center';
      ctx.fillText(`τ=${tau.toFixed(1)}s`, xTau, oy + 8 * D);

      // e⁻¹ point on analytical curve
      if (Math.abs(initVal / sigma) > 0.08) {
        const yTau = cy(initVal * Math.exp(-1));
        ctx.beginPath(); ctx.arc(xTau, yTau, 3 * D, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(${rgb},0.55)`; ctx.fill();
      }
    }

    // Initial-state dot
    const yInit = cy(initVal);
    ctx.beginPath(); ctx.arc(toXp(0), yInit, 5.5 * D, 0, Math.PI * 2);
    ctx.fillStyle   = color; ctx.fill();
    ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.2; ctx.stroke();

    // Perturbation label
    const initNorm = initVal / sigma;
    if (Math.abs(initNorm) > 0.04) {
      const sign = initNorm > 0 ? '+' : '';
      ctx.font      = `bold ${8 * D}px JetBrains Mono, monospace`;
      ctx.fillStyle = color;
      ctx.textAlign = 'left';
      ctx.fillText(`${sign}${initNorm.toFixed(2)}σ`, toXp(0) + 8 * D, yInit - 10 * D);
    }

    // Equilibrium dashed line
    const y0 = toYp(0);
    if (inY(y0)) {
      ctx.strokeStyle = `rgba(${rgb},0.22)`;
      ctx.lineWidth   = 1;
      ctx.setLineDash([3, 7]);
      ctx.beginPath(); ctx.moveTo(ox, y0); ctx.lineTo(ox + iw, y0); ctx.stroke();
      ctx.setLineDash([]);
    }

    // Y-axis σ ticks
    ctx.font      = `${7 * D}px JetBrains Mono, monospace`;
    ctx.fillStyle = '#2e4a62';
    ctx.textAlign = 'right';
    for (const s of [-2, -1, 0, 1, 2]) {
      const y = toYp(s * sigma);
      if (inY(y)) ctx.fillText(`${s}σ`, ox - 3 * D, y + 3 * D);
    }

    // X-axis ticks
    ctx.textAlign = 'center';
    const nTick = Math.min(6, Math.ceil(horizon / 5));
    for (let k = 0; k <= nTick; k++) {
      const t = (k / nTick) * horizon;
      ctx.fillText(`${t.toFixed(0)}s`, toXp(t), oy + ih + 14 * D);
    }

    // Title
    ctx.font      = `${8.5 * D}px Orbitron, monospace`;
    ctx.fillStyle = color;
    ctx.textAlign = 'left';
    ctx.fillText(title, ox, oy - 7 * D);

    // Path count + method label
    ctx.font      = `${7 * D}px JetBrains Mono, monospace`;
    ctx.fillStyle = '#2a4060';
    ctx.textAlign = 'right';
    ctx.fillText(`n=${N_PATHS} · Euler-Maruyama`, ox + iw, oy - 7 * D);
  }

  renderPanel(0,      halfW, pathsP, meanP, stdP, sigP, p0, pr.ap, pr.sigma_p, '#00e5ff', '0,229,255',  'VAGAL p(t)');
  renderPanel(halfW,  halfW, pathsS, meanS, stdS, sigS, s0, pr.as, pr.sigma_s, '#ff6d00', '255,109,0',  'ADRENERGIC s(t)');

  // Divider + coupling annotation
  ctx.strokeStyle = 'rgba(0,180,255,0.09)';
  ctx.lineWidth   = 1;
  ctx.beginPath(); ctx.moveTo(halfW, 0); ctx.lineTo(halfW, H); ctx.stroke();

  if (Math.abs(aps) + Math.abs(asp) > 1e-4) {
    ctx.font      = `${7 * D}px JetBrains Mono, monospace`;
    ctx.fillStyle = 'rgba(100,160,200,0.45)';
    ctx.textAlign = 'center';
    ctx.fillText(`← aₛₚ=${asp.toFixed(3)}  aₚₛ=${aps.toFixed(3)} →`, halfW, H - 6 * D);
  }
}


// ============================================================
//  ANS TRAJECTORY PLAYGROUND
// ============================================================

// ── Scenario library ─────────────────────────────────────────
// Each scenario returns { up, us } in σ-units given (t, T, sigP, sigS).
// Slider values are the parameter scale multipliers applied at run-time.
const PG_SCENARIOS = {
  baseline: {
    label: 'BASELINE',
    desc:  'Unperturbed OU dynamics with estimated parameters. Useful as a null-hypothesis reference.',
    sliders: { ap:1, as:1, sp:1, ss:1, aps:1, asp:1, p0:0, s0:0 },
    input: ()  => ({ up: 0, us: 0 }),
  },
  stress: {
    label: 'ACUTE STRESS',
    desc:  'Sympathetic ramp-up t=15→45 s followed by gradual recovery. Models psychosocial or physical stressor.',
    sliders: { ap:1, as:0.65, sp:0.8, ss:1.6, aps:1, asp:1.4, p0:-0.3, s0:0.5 },
    input: (t, T, sigP, sigS) => {
      const ramp = t < 15 ? 0 : t < 45 ? (t-15)/30 : Math.max(0, 1-(t-45)/Math.max(1,T-45));
      return { up: -ramp*sigP*1.8, us: ramp*sigS*2.2 };
    },
  },
  relax: {
    label: 'RELAXATION',
    desc:  'Progressive vagal augmentation; sympathetic withdrawal. Models meditation, biofeedback, or rest.',
    sliders: { ap:0.75, as:1.3, sp:1.3, ss:0.65, aps:1, asp:0.7, p0:-0.5, s0:0.4 },
    input: (t, T, sigP, sigS) => {
      const ramp = Math.min(1, t / Math.max(1, T * 0.55));
      return { up: ramp*sigP*1.4, us: -ramp*sigS*0.9 };
    },
  },
  exercise: {
    label: 'EXERCISE',
    desc:  'Sustained sympathetic activation with vagal withdrawal. Onset at t=20 s; high SNS drive throughout.',
    sliders: { ap:1.6, as:0.45, sp:0.55, ss:2.0, aps:0.5, asp:2.0, p0:-1.2, s0:1.8 },
    input: (t, T, sigP, sigS) => {
      const onset = Math.min(1, Math.max(0, (t-20)/25));
      return { up: -onset*sigP*2.2, us: onset*sigS*2.8 };
    },
  },
  breathing: {
    label: 'RSA BREATHING',
    desc:  'Resonant slow breathing at 0.1 Hz (6 breaths/min) driving oscillatory vagal input. Increases HF power.',
    sliders: { ap:0.85, as:1.1, sp:1.8, ss:0.75, aps:1, asp:1, p0:0, s0:0 },
    input: (t, T, sigP, sigS) => {
      const f_rsb = 0.1; // Hz
      const amp   = sigP * 2.0;
      return { up: amp*Math.sin(2*Math.PI*f_rsb*t), us: -amp*0.35*Math.sin(2*Math.PI*f_rsb*t) };
    },
  },
  orthostatic: {
    label: 'ORTHOSTATIC',
    desc:  'Step sympathetic surge at t=20 s (passive standing). Vagal withdrawal with partial spontaneous recovery.',
    sliders: { ap:1, as:0.8, sp:1, ss:1.4, aps:1, asp:1.2, p0:0, s0:0 },
    input: (t, T, sigP, sigS) => {
      if (t < 20) return { up: 0, us: 0 };
      const decay = Math.exp(-(t-20) / 35);
      const net   = 1 - decay * 0.55;
      return { up: -net*sigP*1.1, us: net*sigS*2.0 };
    },
  },
  custom: {
    label: 'CUSTOM',
    desc:  'Use sliders to freely specify parameter scales and sustained external inputs.',
    sliders: { ap:1, as:1, sp:1, ss:1, aps:1, asp:1, p0:0, s0:0 },
    input: (t, T, sigP, sigS) => ({
      up: playground.params.up * sigP,
      us: playground.params.us * sigS,
    }),
  },
};

// ── Playground state ──────────────────────────────────────────
const playground = {
  scenario : 'baseline',
  duration : 60,
  view     : 'trajectory',
  result   : null,
  isRunning: false,
  params: {
    ap: 1, as: 1, sp: 1, ss: 1, aps: 1, asp: 1,
    p0: 0, s0: 0, up: 0, us: 0,
  },
};

// ── Slider sync helper ────────────────────────────────────────
function _pgSetSlider(id, val, labelId, fmt) {
  const el = document.getElementById(id);
  if (el) el.value = val;
  const lb = document.getElementById(labelId);
  if (lb) lb.textContent = fmt(parseFloat(val));
}

// ── Param update (called by oninput) ─────────────────────────
function pg_updateParam(key, rawVal) {
  const v = parseFloat(rawVal);
  const fmtScale = x => x.toFixed(2) + '×';
  const fmtSigma = x => (x >= 0 ? '+' : '') + x.toFixed(1) + ' σ';
  switch (key) {
    case 'ap':  playground.params.ap  = v; document.getElementById('pg-ap-val').textContent  = fmtScale(v); break;
    case 'as':  playground.params.as  = v; document.getElementById('pg-as-val').textContent  = fmtScale(v); break;
    case 'sp':  playground.params.sp  = v; document.getElementById('pg-sp-val').textContent  = fmtScale(v); break;
    case 'ss':  playground.params.ss  = v; document.getElementById('pg-ss-val').textContent  = fmtScale(v); break;
    case 'aps': playground.params.aps = v; document.getElementById('pg-aps-val').textContent = fmtScale(v); break;
    case 'asp': playground.params.asp = v; document.getElementById('pg-asp-val').textContent = fmtScale(v); break;
    case 'p0':  playground.params.p0  = v; document.getElementById('pg-p0-val').textContent  = fmtSigma(v); break;
    case 's0':  playground.params.s0  = v; document.getElementById('pg-s0-val').textContent  = fmtSigma(v); break;
    case 'up':  playground.params.up  = v; document.getElementById('pg-up-val').textContent  = fmtSigma(v); break;
    case 'us':  playground.params.us  = v; document.getElementById('pg-us-val').textContent  = fmtSigma(v); break;
  }
}

// ── Scenario selection ────────────────────────────────────────
function pg_setScenario(name, btn) {
  playground.scenario = name;
  document.querySelectorAll('.btn-scenario').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');

  const sc = PG_SCENARIOS[name];
  if (!sc) return;

  // Update all sliders to match preset (custom keeps user values)
  if (name !== 'custom') {
    const { ap, as, sp, ss, aps, asp, p0, s0 } = sc.sliders;
    _pgSetSlider('pg-ap',  ap,  'pg-ap-val',  x => x.toFixed(2) + '×');
    _pgSetSlider('pg-as',  as,  'pg-as-val',  x => x.toFixed(2) + '×');
    _pgSetSlider('pg-sp',  sp,  'pg-sp-val',  x => x.toFixed(2) + '×');
    _pgSetSlider('pg-ss',  ss,  'pg-ss-val',  x => x.toFixed(2) + '×');
    _pgSetSlider('pg-aps', aps, 'pg-aps-val', x => x.toFixed(2) + '×');
    _pgSetSlider('pg-asp', asp, 'pg-asp-val', x => x.toFixed(2) + '×');
    _pgSetSlider('pg-p0',  p0,  'pg-p0-val',  x => (x>=0?'+':'')+x.toFixed(1)+' σ');
    _pgSetSlider('pg-s0',  s0,  'pg-s0-val',  x => (x>=0?'+':'')+x.toFixed(1)+' σ');
    playground.params = { ...playground.params, ap, as, sp, ss, aps, asp, p0, s0 };
  }

  const descEl = document.getElementById('pg-scenario-desc');
  if (descEl) descEl.textContent = sc.desc;
}

// ── Duration selector ─────────────────────────────────────────
function pg_setDuration(dur, btn) {
  playground.duration = dur;
  document.querySelectorAll('.btn-pg-dur').forEach(b => b.classList.remove('active'));
  if (btn) btn.classList.add('active');
}

// ── View selector ─────────────────────────────────────────────
function pg_setView(view, btn) {
  playground.view = view;
  document.querySelectorAll('.btn-pg-view').forEach(b => b.classList.remove('active'));
  if (btn) btn.classList.add('active');
  pg_draw();
}

// ── Reset ─────────────────────────────────────────────────────
function pg_reset() {
  playground.result = null;
  pg_setScenario('baseline', document.getElementById('scen-baseline'));
  // Reset custom inputs
  _pgSetSlider('pg-up', 0, 'pg-up-val', x => '+0.0 σ');
  _pgSetSlider('pg-us', 0, 'pg-us-val', x => '+0.0 σ');
  playground.params.up = 0;
  playground.params.us = 0;
  // Clear stats
  ['hr','rmssd','p','s','d','balance'].forEach(k => {
    const el = document.getElementById('pg-stat-' + k);
    if (el) el.textContent = '—';
  });
  _pgClearCanvas();
}

function _pgClearCanvas() {
  const canvas = document.getElementById('playground-canvas');
  if (!canvas) return;
  sizeCanvas(canvas);
  const ctx = canvas.getContext('2d');
  const D = dpr();
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = 'rgba(6,15,30,0.55)';
  ctx.fillRect(0, 0, W, H);
  ctx.font = `${11 * D}px Orbitron, monospace`;
  ctx.fillStyle = 'rgba(74,96,128,0.45)';
  ctx.textAlign = 'center';
  ctx.fillText('Select a scenario and press  ▶ RUN  to simulate autonomic trajectories', W / 2, H / 2);
  ctx.font = `${9 * D}px JetBrains Mono, monospace`;
  ctx.fillStyle = 'rgba(74,96,128,0.28)';
  ctx.fillText('Parameters are scaled from the EKF estimates of the loaded recording', W / 2, H / 2 + 20 * D);
}

// ── Main run function ─────────────────────────────────────────
async function pg_run() {
  if (!state.currentAnalysis) {
    alert('Load a recording first — the playground uses its estimated SDE-IG parameters as baseline.');
    return;
  }
  if (playground.isRunning) return;

  playground.isRunning = true;
  const btn = document.getElementById('pg-run-btn');
  if (btn) { btn.textContent = '⏳  SIMULATING…'; btn.classList.add('running'); }

  await sleep(12); // yield to allow UI repaint

  try {
    playground.result = _pgSimulate();
    _pgUpdateStats(playground.result);
    pg_draw();
  } finally {
    playground.isRunning = false;
    if (btn) { btn.textContent = '▶  RUN'; btn.classList.remove('running'); }
  }
}

// ── Euler-Maruyama ensemble simulation ────────────────────────
function _pgSimulate() {
  const an  = state.currentAnalysis;
  const pr  = an.params;
  const pg  = playground.params;
  const sc  = PG_SCENARIOS[playground.scenario];
  const T   = playground.duration;
  const nP  = parseInt(document.getElementById('pg-npaths')?.value || 20);

  // How many time steps: 80 Hz resolution (fine enough for HF band)
  const N     = Math.min(48000, Math.round(T * 80));
  const dt    = T / N;
  const sqDt  = Math.sqrt(dt);

  // ── Scaled parameters ────────────────────────────────────────
  const ap  = pr.ap         * pg.ap;
  const as_ = pr.as         * pg.as;
  const sp  = pr.sigma_p    * pg.sp;
  const ss  = pr.sigma_s    * pg.ss;
  const aps = (pr.a_ps||0)  * pg.aps;
  const asp = (pr.a_sp||0)  * pg.asp;
  const mu0 = pr.mu0;

  // Stationary σ of the unscaled process (for σ-unit conversions)
  const sigP_stat = pr.sigma_p / Math.sqrt(2 * pr.ap + 1e-10);
  const sigS_stat = pr.sigma_s / Math.sqrt(2 * pr.as + 1e-10);

  // Initial conditions [σ-units → raw]
  const p0 = pg.p0 * sigP_stat;
  const s0 = pg.s0 * sigS_stat;

  function randn() {
    return Math.sqrt(-2 * Math.log(Math.random() + 1e-14)) * Math.cos(2 * Math.PI * Math.random());
  }

  const inputFn = sc ? sc.input : (t, T, sP, sS) => ({
    up: pg.up * sP,
    us: pg.us * sS,
  });

  // ── Allocate path arrays (store every 4th step to save memory) ─
  const stride = 4;
  const nOut   = Math.floor(N / stride) + 1;

  const times  = new Float32Array(nOut);
  const pathsP = Array.from({ length: nP }, () => new Float32Array(nOut));
  const pathsS = Array.from({ length: nP }, () => new Float32Array(nOut));
  const pathsRR= Array.from({ length: nP }, () => new Float32Array(nOut));

  for (let s = 0; s < nOut; s++) times[s] = Math.min(s * stride * dt, T);

  for (let k = 0; k < nP; k++) {
    let p = p0, s = s0;
    pathsP[k][0] = p; pathsS[k][0] = s;
    pathsRR[k][0] = clamp(mu0 * Math.exp(-(p - s)), 0.20, 3.0);
    let outIdx = 1;

    for (let i = 0; i < N; i++) {
      const t_i = i * dt;
      const { up, us } = inputFn(t_i, T, sigP_stat, sigS_stat);
      const dp = (-ap * p + aps * s + up) * dt + sp * sqDt * randn();
      const ds = ( asp * p - as_ * s + us) * dt + ss * sqDt * randn();
      p += dp; s += ds;

      if ((i + 1) % stride === 0 && outIdx < nOut) {
        pathsP[k][outIdx]  = p;
        pathsS[k][outIdx]  = s;
        pathsRR[k][outIdx] = clamp(mu0 * Math.exp(-(p - s)), 0.20, 3.0);
        outIdx++;
      }
    }
  }

  // ── Pointwise ensemble statistics ────────────────────────────
  const meanP  = new Float32Array(nOut);
  const meanS  = new Float32Array(nOut);
  const meanRR = new Float32Array(nOut);
  const stdP   = new Float32Array(nOut);
  const stdS   = new Float32Array(nOut);
  const stdRR  = new Float32Array(nOut);

  for (let i = 0; i < nOut; i++) {
    let mp = 0, ms = 0, mr = 0;
    for (let k = 0; k < nP; k++) { mp += pathsP[k][i]; ms += pathsS[k][i]; mr += pathsRR[k][i]; }
    meanP[i] = mp/nP; meanS[i] = ms/nP; meanRR[i] = mr/nP;
    let vp = 0, vs = 0, vr = 0;
    for (let k = 0; k < nP; k++) {
      vp += (pathsP[k][i] - meanP[i])**2;
      vs += (pathsS[k][i] - meanS[i])**2;
      vr += (pathsRR[k][i] - meanRR[i])**2;
    }
    const inv = 1 / Math.max(1, nP - 1);
    stdP[i]  = Math.sqrt(vp * inv + 1e-20);
    stdS[i]  = Math.sqrt(vs * inv + 1e-20);
    stdRR[i] = Math.sqrt(vr * inv + 1e-20);
  }

  // RMSSD from mean RR path
  let sumSqD = 0;
  for (let i = 1; i < nOut; i++) {
    const d = meanRR[i] - meanRR[i-1];
    sumSqD += d * d;
  }
  const simRMSSD_ms = Math.sqrt(sumSqD / Math.max(1, nOut - 1)) * 1000;
  const termHR      = 60 / Math.max(0.2, meanRR[nOut - 1]);
  const termP       = meanP[nOut - 1];
  const termS       = meanS[nOut - 1];
  const termDelta   = termS - termP;

  return {
    times, nOut, nP,
    pathsP, pathsS, pathsRR,
    meanP, meanS, meanRR,
    stdP, stdS, stdRR,
    stats: {
      rmssd: simRMSSD_ms, hr: termHR,
      termP, termS, termDelta,
      sigP_stat, sigS_stat,
      ap, as: as_, aps, asp, mu0,
    },
  };
}

// ── Update summary stat cells ─────────────────────────────────
function _pgUpdateStats(res) {
  const s = res.stats;
  const sigP = s.sigP_stat, sigS = s.sigS_stat;
  document.getElementById('pg-stat-hr').textContent    = s.hr.toFixed(1) + ' bpm';
  document.getElementById('pg-stat-rmssd').textContent = s.rmssd.toFixed(2) + ' ms';
  document.getElementById('pg-stat-p').textContent     = s.termP.toFixed(4);
  document.getElementById('pg-stat-s').textContent     = s.termS.toFixed(4);
  document.getElementById('pg-stat-d').textContent     = s.termDelta.toFixed(4);
  const pN = s.termP / (sigP * 3), sN = s.termS / (sigS * 3);
  const bal = pN - sN;
  document.getElementById('pg-stat-balance').textContent =
    bal > 0.18 ? 'VAGAL DOM.' : bal < -0.18 ? 'SYMP. DOM.' : 'BALANCED';
}

// ── Drawing dispatcher ────────────────────────────────────────
function pg_draw() {
  if (!playground.result) { _pgClearCanvas(); return; }
  switch (playground.view) {
    case 'trajectory': _pgDrawTrajectory(playground.result); break;
    case 'phase':      _pgDrawPhase(playground.result);      break;
    case 'rr':         _pgDrawRR(playground.result);         break;
    case 'hr':         _pgDrawHR(playground.result);         break;
  }
}

// ── Shared canvas setup ───────────────────────────────────────
function _pgCanvas() {
  const canvas = document.getElementById('playground-canvas');
  sizeCanvas(canvas);
  const ctx = canvas.getContext('2d');
  const D = dpr();
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = 'rgba(6,15,30,0.55)';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  return { canvas, ctx, D, W: canvas.width, H: canvas.height };
}

// Shared: draw stochastic envelope + individual paths + mean line
function _pgBand(ctx, times, nOut, paths, meanArr, stdArr, nP, sx, cy, alpha_band1, alpha_band2, pathAlpha, rgb, lineW, D) {
  // 2-σ fill
  ctx.beginPath();
  for (let i=0;i<nOut;i++) ctx[i?'lineTo':'moveTo'](sx(times[i]), cy(meanArr[i]+2*stdArr[i]));
  for (let i=nOut-1;i>=0;i--) ctx.lineTo(sx(times[i]), cy(meanArr[i]-2*stdArr[i]));
  ctx.closePath(); ctx.fillStyle=`rgba(${rgb},${alpha_band2})`; ctx.fill();
  // 1-σ fill
  ctx.beginPath();
  for (let i=0;i<nOut;i++) ctx[i?'lineTo':'moveTo'](sx(times[i]), cy(meanArr[i]+stdArr[i]));
  for (let i=nOut-1;i>=0;i--) ctx.lineTo(sx(times[i]), cy(meanArr[i]-stdArr[i]));
  ctx.closePath(); ctx.fillStyle=`rgba(${rgb},${alpha_band1})`; ctx.fill();
  // Individual paths (skip if too many to keep it readable)
  const drawEvery = Math.max(1, Math.round(nP / 12));
  for (let k=0;k<nP;k+=drawEvery) {
    ctx.beginPath(); let penDown=false;
    for (let i=0;i<nOut;i++) {
      const y=cy(paths[k][i]);
      penDown?ctx.lineTo(sx(times[i]),y):ctx.moveTo(sx(times[i]),y); penDown=true;
    }
    ctx.strokeStyle=`rgba(${rgb},${pathAlpha})`; ctx.lineWidth=0.6; ctx.stroke();
  }
  // Mean
  ctx.beginPath();
  for (let i=0;i<nOut;i++) ctx[i?'lineTo':'moveTo'](sx(times[i]), cy(meanArr[i]));
  ctx.strokeStyle=`rgba(${rgb},0.95)`; ctx.lineWidth=lineW*D; ctx.stroke();
}

// ── View 1: p(t), s(t), Δ(t) ─────────────────────────────────
function _pgDrawTrajectory(res) {
  const { canvas, ctx, D, W, H } = _pgCanvas();
  const { times, nOut, nP, pathsP, pathsS, meanP, meanS, stdP, stdS, stats } = res;
  const T = times[nOut-1];

  const padL=54*D, padR=22*D, padT=24*D, padB=38*D;
  const IW=W-padL-padR, IH=H-padT-padB;

  // Value range: include ±2σ envelopes
  let minV =  Infinity, maxV = -Infinity;
  for (let i=0;i<nOut;i++) {
    minV=Math.min(minV, meanP[i]-2*stdP[i], meanS[i]-2*stdS[i]);
    maxV=Math.max(maxV, meanP[i]+2*stdP[i], meanS[i]+2*stdS[i]);
  }
  const rng  = Math.max(0.3, maxV-minV);
  const padV = rng*0.12;

  const sx = t => padL + (t/T)*IW;
  const sy = v => padT + IH - (v-minV+padV)/(rng+2*padV)*IH;
  const cy = v => clamp(sy(v), padT, padT+IH);

  // Grid
  ctx.strokeStyle='rgba(0,100,150,0.10)'; ctx.lineWidth=0.7;
  for (let i=0;i<=4;i++) { const y=padT+i/4*IH; ctx.beginPath();ctx.moveTo(padL,y);ctx.lineTo(W-padR,y);ctx.stroke(); }
  // Zero line
  const y0=sy(0);
  if (y0>=padT&&y0<=padT+IH) {
    ctx.strokeStyle='rgba(255,255,255,0.10)'; ctx.lineWidth=1; ctx.setLineDash([3,6]);
    ctx.beginPath();ctx.moveTo(padL,y0);ctx.lineTo(W-padR,y0);ctx.stroke(); ctx.setLineDash([]);
  }

  // p(t) band + paths + mean
  _pgBand(ctx,times,nOut,pathsP,meanP,stdP,nP,sx,cy, 0.12,0.055,0.09,'0,229,255',1.9,D);
  // s(t) band + paths + mean
  _pgBand(ctx,times,nOut,pathsS,meanS,stdS,nP,sx,cy, 0.12,0.055,0.09,'255,109,0',1.9,D);

  // Net Δ(t) = s−p  [dashed green]
  ctx.beginPath();
  for (let i=0;i<nOut;i++) ctx[i?'lineTo':'moveTo'](sx(times[i]), cy(meanS[i]-meanP[i]));
  ctx.strokeStyle='#00e676'; ctx.lineWidth=1.4*D; ctx.setLineDash([4*D,3*D]); ctx.stroke(); ctx.setLineDash([]);

  // Scenario-specific input overlay (shaded region)
  const sc = PG_SCENARIOS[playground.scenario];
  if (sc && playground.scenario !== 'baseline') {
    // Annotate input regions via faint vertical bands
    // Sample the input function to find non-zero regions
    const step = Math.max(1, Math.round(T / 200));
    const sigP = stats.sigP_stat, sigS = stats.sigS_stat;
    let inRegion = false, regionStart = 0;
    for (let ti = 0; ti <= T; ti += step) {
      const { up, us } = sc.input(ti, T, sigP, sigS);
      const active = Math.hypot(up / (sigP+1e-10), us / (sigS+1e-10)) > 0.05;
      if (active && !inRegion) { regionStart = ti; inRegion = true; }
      if (!active && inRegion) {
        ctx.fillStyle = 'rgba(255,214,0,0.04)';
        ctx.fillRect(sx(regionStart), padT, sx(ti)-sx(regionStart), IH);
        inRegion = false;
      }
    }
    if (inRegion) {
      ctx.fillStyle = 'rgba(255,214,0,0.04)';
      ctx.fillRect(sx(regionStart), padT, sx(T)-sx(regionStart), IH);
    }
  }

  // ── Overlay measured trajectory ───────────────────────────────
  const showOverlay = document.getElementById('pg-overlay')?.checked && state.currentAnalysis;
  if (showOverlay) {
    const filt = state.currentAnalysis.filter;
    const tMax = Math.min(T, filt[filt.length-1].t);
    ctx.save();
    ctx.strokeStyle='rgba(0,229,255,0.35)'; ctx.lineWidth=1.2*D; ctx.setLineDash([2,5]);
    ctx.beginPath();
    for (let i=0;i<filt.length;i++) {
      if (filt[i].t>tMax) break;
      i===0?ctx.moveTo(sx(filt[i].t),cy(filt[i].p)):ctx.lineTo(sx(filt[i].t),cy(filt[i].p));
    }
    ctx.stroke();
    ctx.strokeStyle='rgba(255,109,0,0.35)';
    ctx.beginPath();
    for (let i=0;i<filt.length;i++) {
      if (filt[i].t>tMax) break;
      i===0?ctx.moveTo(sx(filt[i].t),cy(filt[i].s)):ctx.lineTo(sx(filt[i].t),cy(filt[i].s));
    }
    ctx.stroke(); ctx.setLineDash([]); ctx.restore();

    ctx.font=`${7.5*D}px JetBrains Mono,monospace`; ctx.fillStyle='rgba(255,255,255,0.30)';
    ctx.textAlign='right'; ctx.fillText('- - MEASURED',W-padR,padT+13*D);
  }

  // Axis labels
  ctx.font=`${8*D}px JetBrains Mono,monospace`; ctx.fillStyle='#2e4a62';
  ctx.textAlign='right';
  for (let i=0;i<=4;i++) {
    const v=minV-padV+i/4*(rng+2*padV);
    ctx.fillText(v.toFixed(2),padL-4*D,padT+IH-i/4*IH+3*D);
  }
  ctx.textAlign='center';
  const nTicks=Math.min(6,Math.ceil(T/10));
  for (let i=0;i<=nTicks;i++) ctx.fillText((i/nTicks*T).toFixed(0)+'s', sx(i/nTicks*T), H-9*D);

  // Legend
  ctx.textAlign='left'; ctx.font=`${8*D}px JetBrains Mono,monospace`;
  ctx.fillStyle='#00e5ff'; ctx.fillText('━  p̂(t) vagal',     padL+6*D,  padT+14*D);
  ctx.fillStyle='#ff6d00'; ctx.fillText('━  ŝ(t) adren.',    padL+100*D, padT+14*D);
  ctx.fillStyle='#00e676'; ctx.fillText('╌  Δ̂(t) drive',    padL+196*D, padT+14*D);

  // Title
  ctx.font=`${9*D}px Orbitron,monospace`; ctx.fillStyle='rgba(0,229,255,0.62)';
  ctx.textAlign='right';
  ctx.fillText(
    `${PG_SCENARIOS[playground.scenario]?.label}  ·  n=${nP} paths  ·  T=${T}s`,
    W-padR, padT-6*D
  );
}

// ── View 2: Phase portrait ────────────────────────────────────
function _pgDrawPhase(res) {
  const { canvas, ctx, D, W, H } = _pgCanvas();
  const { nOut, nP, pathsP, pathsS, meanP, meanS, stats } = res;
  const sigP = stats.sigP_stat, sigS = stats.sigS_stat;
  const AX = 3.5;

  const pad=48*D, plotW=W-2*pad, plotH=H-2*pad;
  const toX = v => pad + (v/sigP + AX)/(2*AX)*plotW;
  const toY = v => H-pad - (v/sigS + AX)/(2*AX)*plotH;
  const ox=toX(0), oy=toY(0);

  // Quadrant fills
  ctx.fillStyle='rgba(0,229,255,0.03)'; ctx.fillRect(ox,oy,W-pad-ox,H-pad-oy);     // REST
  ctx.fillStyle='rgba(255,109,0,0.03)'; ctx.fillRect(pad,pad,ox-pad,oy-pad);         // STRESS

  // Grid σ lines
  ctx.lineWidth=0.6;
  for (let s=-3;s<=3;s++) {
    const isO=s===0;
    ctx.strokeStyle=isO?'rgba(255,255,255,0.12)':'rgba(0,100,160,0.10)';
    ctx.setLineDash(isO?[]:[3,7]);
    ctx.beginPath(); ctx.moveTo(toX(s*sigP),pad); ctx.lineTo(toX(s*sigP),H-pad); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(pad,toY(s*sigS)); ctx.lineTo(W-pad,toY(s*sigS)); ctx.stroke();
  }
  ctx.setLineDash([]);

  // σ contour ellipses
  for (const [r,a] of [[1,0.18],[2,0.08]]) {
    ctx.beginPath();
    ctx.ellipse(ox, oy, toX(r*sigP)-ox, oy-toY(r*sigS), 0, 0, Math.PI*2);
    ctx.strokeStyle=`rgba(255,255,255,${a})`; ctx.lineWidth=1; ctx.stroke();
    ctx.font=`${6.5*D}px JetBrains Mono,monospace`; ctx.fillStyle=`rgba(120,160,185,${a*2})`;
    ctx.textAlign='left'; ctx.fillText(`${r}σ`, toX(r*sigP)+3*D, oy-2*D);
  }

  // Individual paths (colour = k/nP gradient)
  const drawEvery = Math.max(1, Math.round(nP / 14));
  for (let k=0;k<nP;k+=drawEvery) {
    ctx.beginPath();
    for (let i=0;i<nOut;i++) {
      const t=i/nOut;
      i===0?ctx.moveTo(toX(pathsP[k][i]),toY(pathsS[k][i])):ctx.lineTo(toX(pathsP[k][i]),toY(pathsS[k][i]));
    }
    const t=k/nP;
    ctx.strokeStyle=`rgba(${Math.round(60+t*170)},${Math.round(90+t*30)},${Math.round(210-t*130)},0.18)`;
    ctx.lineWidth=0.8; ctx.stroke();
  }

  // Mean path — glowing
  ctx.save();
  ctx.lineWidth=8*D; ctx.strokeStyle='rgba(0,255,150,0.06)';
  ctx.beginPath(); for (let i=0;i<nOut;i++) ctx[i?'lineTo':'moveTo'](toX(meanP[i]),toY(meanS[i])); ctx.stroke();
  ctx.lineWidth=3*D; ctx.strokeStyle='rgba(0,255,150,0.15)';
  ctx.beginPath(); for (let i=0;i<nOut;i++) ctx[i?'lineTo':'moveTo'](toX(meanP[i]),toY(meanS[i])); ctx.stroke();
  ctx.lineWidth=2*D; ctx.strokeStyle='#00ffaa';
  ctx.beginPath(); for (let i=0;i<nOut;i++) ctx[i?'lineTo':'moveTo'](toX(meanP[i]),toY(meanS[i])); ctx.stroke();
  ctx.restore();

  // Direction chevrons along mean path
  const chevEvery = Math.max(1, Math.floor(nOut/9));
  for (let i=chevEvery;i<nOut;i+=chevEvery) {
    const dx=meanP[i]-meanP[i-1], dy=meanS[i]-meanS[i-1];
    const dm=Math.hypot(dx,dy); if (dm<1e-5) continue;
    const ux=dx/dm*(toX(sigP)-toX(0))/(sigP||1);
    const uy=-dy/dm*(oy-toY(sigS))/(sigS||1);
    const um=Math.hypot(ux,uy)+1e-10;
    const cx2=toX(meanP[i]),cy2=toY(meanS[i]);
    const cs=6*D;
    ctx.fillStyle='rgba(0,255,170,0.55)';
    ctx.beginPath();
    ctx.moveTo(cx2+ux/um*cs,          cy2+uy/um*cs);
    ctx.lineTo(cx2-ux/um*cs-uy/um*cs*0.5, cy2-uy/um*cs+ux/um*cs*0.5);
    ctx.lineTo(cx2-ux/um*cs+uy/um*cs*0.5, cy2-uy/um*cs-ux/um*cs*0.5);
    ctx.closePath(); ctx.fill();
  }

  // Start marker (cyan) / End marker (orange)
  [[meanP[0],      meanS[0],      '#00e5ff','START'  ],
   [meanP[nOut-1], meanS[nOut-1], '#ff6d00','END'    ]].forEach(([p,s,col,lbl]) => {
    const px=toX(p), py=toY(s);
    ctx.beginPath(); ctx.arc(px, py, 6*D, 0, Math.PI*2);
    ctx.fillStyle=col; ctx.fill();
    ctx.strokeStyle='rgba(255,255,255,0.6)'; ctx.lineWidth=1.5; ctx.stroke();
    ctx.font=`${7.5*D}px JetBrains Mono,monospace`; ctx.fillStyle=col;
    ctx.textAlign='center'; ctx.fillText(lbl, px, py-9*D);
  });

  // Overlay measured
  if (document.getElementById('pg-overlay')?.checked && state.currentAnalysis) {
    const filt=state.currentAnalysis.filter;
    ctx.strokeStyle='rgba(255,255,255,0.20)'; ctx.lineWidth=1.2; ctx.setLineDash([2,5]);
    ctx.beginPath();
    for (let i=0;i<filt.length;i++) {
      i===0?ctx.moveTo(toX(filt[i].p),toY(filt[i].s)):ctx.lineTo(toX(filt[i].p),toY(filt[i].s));
    }
    ctx.stroke(); ctx.setLineDash([]);
    ctx.font=`${7*D}px JetBrains Mono,monospace`; ctx.fillStyle='rgba(255,255,255,0.28)';
    ctx.textAlign='right'; ctx.fillText('- - measured',W-pad,pad+10*D);
  }

  // Quadrant labels
  const ql=(x,y,txt,col)=>{
    ctx.font=`bold ${7.5*D}px Orbitron,monospace`; ctx.fillStyle=col;
    ctx.textAlign='center'; ctx.fillText(txt,x,y);
  };
  const qxR=ox+(W-pad-ox)*0.52, qxL=pad+(ox-pad)*0.48;
  const qyT=pad+(oy-pad)*0.30,  qyB=oy+(H-pad-oy)*0.30;
  ql(qxR,qyT,'CO-ACT.',  'rgba(255,214,0,0.35)');
  ql(qxL,qyT,'STRESS',   'rgba(255,70,70,0.35)');
  ql(qxL,qyB,'WITHDRAW.','rgba(140,140,185,0.30)');
  ql(qxR,qyB,'REST',     'rgba(0,229,255,0.35)');

  // Axis labels
  ctx.font=`${9*D}px Orbitron,monospace`; ctx.textAlign='center';
  ctx.fillStyle='rgba(0,229,255,0.65)'; ctx.fillText('VAGAL p̂(t)  [σ-units]',W/2,H-8*D);
  ctx.save(); ctx.translate(14*D,H/2); ctx.rotate(-Math.PI/2);
  ctx.fillStyle='rgba(255,109,0,0.65)'; ctx.fillText('ADRENERGIC ŝ(t)  [σ-units]',0,0);
  ctx.restore();

  // Title
  ctx.font=`${9*D}px Orbitron,monospace`; ctx.fillStyle='rgba(0,229,255,0.60)';
  ctx.textAlign='left'; ctx.fillText('PHASE PORTRAIT · SIMULATED ENSEMBLE',pad,22*D);
}

// ── View 3: RR tachogram ─────────────────────────────────────
function _pgDrawRR(res) {
  const { canvas, ctx, D, W, H } = _pgCanvas();
  const { times, nOut, nP, pathsRR, meanRR, stdRR, stats } = res;
  const T = times[nOut-1];

  const padL=56*D, padR=52*D, padT=24*D, padB=38*D;
  const IW=W-padL-padR, IH=H-padT-padB;

  let minV=Infinity, maxV=-Infinity;
  for (let i=0;i<nOut;i++) {
    minV=Math.min(minV,(meanRR[i]-2*stdRR[i])*1000);
    maxV=Math.max(maxV,(meanRR[i]+2*stdRR[i])*1000);
  }
  const rng=Math.max(50,maxV-minV), padV=rng*0.1;
  const sx = t => padL + t/T*IW;
  const sy = v => padT + IH - (v*1000-minV+padV)/(rng+2*padV)*IH;
  const cy = v => clamp(sy(v), padT, padT+IH);

  // Grid
  ctx.strokeStyle='rgba(0,100,150,0.10)'; ctx.lineWidth=0.7;
  for (let i=0;i<=4;i++) { const y=padT+i/4*IH; ctx.beginPath();ctx.moveTo(padL,y);ctx.lineTo(W-padR,y);ctx.stroke(); }

  // Bands + paths + mean
  _pgBand(ctx,times,nOut,pathsRR,meanRR,stdRR,nP,sx,cy,0.14,0.06,0.10,'124,77,255',1.9,D);

  // Overlay measured RR
  if (document.getElementById('pg-overlay')?.checked && state.currentAnalysis) {
    const an=state.currentAnalysis;
    ctx.strokeStyle='rgba(180,136,255,0.30)'; ctx.lineWidth=1.2*D; ctx.setLineDash([2,5]);
    ctx.beginPath();
    for (let i=0;i<an.times.length;i++) {
      if (an.times[i]>T) break;
      i===0?ctx.moveTo(sx(an.times[i]),cy(an.rr[i])):ctx.lineTo(sx(an.times[i]),cy(an.rr[i]));
    }
    ctx.stroke(); ctx.setLineDash([]);
  }

  // Y-axis (ms) + HR right axis
  ctx.font=`${8*D}px JetBrains Mono,monospace`; ctx.fillStyle='#2e4a62';
  for (let i=0;i<=4;i++) {
    const v=minV-padV+i/4*(rng+2*padV);
    ctx.textAlign='right'; ctx.fillText(Math.round(v)+'ms',padL-4*D,padT+IH-i/4*IH+3*D);
    const hr=60000/Math.max(1,v);
    ctx.textAlign='left'; ctx.fillStyle='rgba(179,136,255,0.40)';
    ctx.fillText(Math.round(hr)+'bpm',W-padR+4*D,padT+IH-i/4*IH+3*D);
    ctx.fillStyle='#2e4a62';
  }
  ctx.textAlign='center';
  const nTicks=Math.min(6,Math.ceil(T/10));
  for (let i=0;i<=nTicks;i++) ctx.fillText((i/nTicks*T).toFixed(0)+'s',sx(i/nTicks*T),H-9*D);

  ctx.font=`${9*D}px Orbitron,monospace`; ctx.fillStyle='rgba(179,136,255,0.65)';
  ctx.textAlign='left'; ctx.fillText('SIMULATED RR TACHOGRAM',padL,padT-7*D);
}

// ── View 4: Instantaneous HR ──────────────────────────────────
function _pgDrawHR(res) {
  const { canvas, ctx, D, W, H } = _pgCanvas();
  const { times, nOut, nP, pathsRR, meanRR, stdRR } = res;
  const T = times[nOut-1];

  const padL=56*D, padR=22*D, padT=24*D, padB=38*D;
  const IW=W-padL-padR, IH=H-padT-padB;

  // Convert RR→HR
  const meanHR = meanRR.map(rr => 60/Math.max(0.2,rr));
  const stdHR  = stdRR.map((s,i)  => (60/(Math.max(0.2,meanRR[i])**2))*s);
  const pathsHR = pathsRR.map(path => path.map(rr => 60/Math.max(0.2,rr)));

  let minV=Infinity, maxV=-Infinity;
  for (let i=0;i<nOut;i++) {
    minV=Math.min(minV,meanHR[i]-2*stdHR[i]);
    maxV=Math.max(maxV,meanHR[i]+2*stdHR[i]);
  }
  const rng=Math.max(10,maxV-minV), padV=rng*0.1;
  const sx = t => padL + t/T*IW;
  const sy = v => padT + IH - (v-minV+padV)/(rng+2*padV)*IH;
  const cy = v => clamp(sy(v), padT, padT+IH);

  ctx.strokeStyle='rgba(0,100,150,0.10)'; ctx.lineWidth=0.7;
  for (let i=0;i<=4;i++) { const y=padT+i/4*IH; ctx.beginPath();ctx.moveTo(padL,y);ctx.lineTo(W-padR,y);ctx.stroke(); }

  _pgBand(ctx,times,nOut,pathsHR,meanHR,stdHR,nP,sx,cy,0.12,0.05,0.09,'255,70,120',2.0,D);

  // Overlay measured HR (from filter)
  if (document.getElementById('pg-overlay')?.checked && state.currentAnalysis) {
    const filt=state.currentAnalysis.filter;
    ctx.strokeStyle='rgba(255,80,130,0.30)'; ctx.lineWidth=1.2*D; ctx.setLineDash([2,5]);
    ctx.beginPath();
    for (let i=0;i<filt.length;i++) {
      if (filt[i].t>T) break;
      const hr=60/Math.max(0.2,filt[i].mu);
      i===0?ctx.moveTo(sx(filt[i].t),cy(hr)):ctx.lineTo(sx(filt[i].t),cy(hr));
    }
    ctx.stroke(); ctx.setLineDash([]);
  }

  ctx.font=`${8*D}px JetBrains Mono,monospace`; ctx.fillStyle='#2e4a62';
  ctx.textAlign='right';
  for (let i=0;i<=4;i++) {
    const v=minV-padV+i/4*(rng+2*padV);
    ctx.fillText(Math.round(v)+' bpm',padL-4*D,padT+IH-i/4*IH+3*D);
  }
  ctx.textAlign='center';
  const nTicks=Math.min(6,Math.ceil(T/10));
  for (let i=0;i<=nTicks;i++) ctx.fillText((i/nTicks*T).toFixed(0)+'s',sx(i/nTicks*T),H-9*D);

  ctx.font=`${9*D}px Orbitron,monospace`; ctx.fillStyle='rgba(255,70,120,0.65)';
  ctx.textAlign='left'; ctx.fillText('SIMULATED INSTANTANEOUS HEART RATE',padL,padT-7*D);
}

// ============================================================
//  PLAYBACK  (requestAnimationFrame, variable speed)
// ============================================================
function togglePlayback() {
  const pb = state.playback;
  pb.active = !pb.active;
  document.getElementById('play-btn').textContent = pb.active ? '⏸' : '▶';
  if (pb.active) {
    const an = state.currentAnalysis;
    if (!an || !an.filter.length) { pb.active = false; return; }
    pb.virtualTime = an.filter[clamp(pb.frame, 0, an.filter.length-1)].t;
    pb.lastTs = null;
    requestAnimationFrame(_pbTick);
  }
}

function _pbTick(ts) {
  const pb = state.playback;
  if (!pb.active) return;
  const an = state.currentAnalysis;
  if (!an) return;

  if (pb.lastTs === null) pb.lastTs = ts;
  const wall = (ts - pb.lastTs) / 1000;
  pb.lastTs = ts;
  pb.virtualTime += wall * pb.speed;

  const tEnd = an.filter[an.filter.length-1].t;
  if (pb.virtualTime >= tEnd) {
    pb.virtualTime = tEnd;
    pb.active = false;
    document.getElementById('play-btn').textContent = '▶';
  }

  // Binary search for frame at virtualTime
  let lo = 0, hi = an.filter.length - 1;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    an.filter[mid].t <= pb.virtualTime ? (lo = mid) : (hi = mid - 1);
  }
  pb.frame = lo;

  updatePlaybackDisplay(an);
  drawBranchChart(an);       // fast — just line drawing
  if (state.visMode === 'phase') drawPhaseSpace(an);

  // Redraw wavelet cursor without recomputing heatmap
  _redrawWaveletCursor(an);

  if (pb.active) requestAnimationFrame(_pbTick);
}

function _redrawWaveletCursor(an) {
  const canvas = document.getElementById('psd-canvas');
  if (!an._wvCache) return;
  const ctx = canvas.getContext('2d');
  const D   = dpr();
  const padL = 46*D, padR = 16*D, padT = 16*D, padB = 32*D;
  const IW  = Math.round(canvas.width - padL - padR);
  const IH  = Math.round(canvas.height - padT - padB);
  // Restore cached image
  ctx.putImageData(an._wvCache.img, Math.round(padL), Math.round(padT));
  // Draw cursor
  const tEnd = an.times[an.times.length-1];
  const fi   = clamp(state.playback.frame, 0, an.filter.length-1);
  const xc   = padL + (an.filter[fi].t / tEnd) * IW;
  ctx.strokeStyle = 'rgba(255,255,255,0.50)';
  ctx.lineWidth   = 1.5;
  ctx.beginPath(); ctx.moveTo(xc, padT); ctx.lineTo(xc, padT+IH); ctx.stroke();
}

function seekTo(frac) {
  const an = state.currentAnalysis;
  if (!an) return;
  state.playback.frame = Math.round(frac * (an.filter.length - 1));
  state.playback.virtualTime = an.filter[state.playback.frame].t;
  updatePlaybackDisplay(an);
  drawBranchChart(an);
  if (state.visMode === 'phase') drawPhaseSpace(an);
  _redrawWaveletCursor(an);
}

document.getElementById('timeline-bar').addEventListener('click', function(e) {
  const rect = this.getBoundingClientRect();
  seekTo(clamp((e.clientX - rect.left) / rect.width, 0, 1));
});

// Speed selector
document.addEventListener('DOMContentLoaded', () => {
  const sel = document.getElementById('pb-speed');
  if (sel) sel.addEventListener('change', () => {
    state.playback.speed = parseFloat(sel.value);
  });
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
  updatePlaybackDisplay(analysis);
  if (state.visMode === 'phase') startPhaseAnim(analysis);  // re-seed particles for new recording
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
  drawWavelet(analysis);
  drawOUSim();
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
  if (mode === 'phase' && state.currentAnalysis) {
    startPhaseAnim(state.currentAnalysis);   // starts RAF loop + particles
  } else {
    stopPhaseAnim();
  }
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
      drawWavelet(state.currentAnalysis);
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

// ── Playground: clear canvas once DOM is ready ────────────────
document.addEventListener('DOMContentLoaded', () => { _pgClearCanvas(); });

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
