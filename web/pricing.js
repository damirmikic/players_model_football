// pricing.js - browser port of the Python engine (score matrix, Dixon-Coles, margins, settlement).
// Must give the same probabilities as pricing.py for the same lambdas (checked by test_parity).
const PE = (() => {
  const MAXG = 12;
  function pois(k, l) { let f = 1; for (let i = 2; i <= k; i++) f *= i; return Math.exp(-l) * Math.pow(l, k) / f; }
  function scoreMatrix(lh, la, rho = 0, n = MAXG) {
    const ph = [], pa = [];
    for (let i = 0; i <= n; i++) { ph.push(pois(i, lh)); pa.push(pois(i, la)); }
    const M = ph.map(h => pa.map(a => h * a));
    if (rho) {
      M[0][0] *= 1 - lh * la * rho; M[0][1] *= 1 + lh * rho; M[1][0] *= 1 + la * rho; M[1][1] *= 1 - rho;
    }
    let s = 0; M.forEach(r => r.forEach(v => s += v));
    return M.map(r => r.map(v => v / s));
  }
  function oneXTwo(M) {
    let h = 0, d = 0, a = 0;
    M.forEach((r, i) => r.forEach((v, j) => { if (i > j) h += v; else if (i === j) d += v; else a += v; }));
    return [h, d, a];
  }
  function dist(M, f) { const d = new Map(); M.forEach((r, i) => r.forEach((v, j) => { const k = f(i, j); d.set(k, (d.get(k) || 0) + v); })); return d; }
  function splitLine(line) { const r = Math.round(line * 4); return (r % 2 === 0) ? [line] : [line - 0.25, line + 0.25]; }
  function settle(value, line) {
    const parts = splitLine(line); let w = 0, l = 0;
    for (const ln of parts) { const r = value + ln; if (r > 1e-9) w += 1 / parts.length; else if (r < -1e-9) l += 1 / parts.length; }
    return [w, l];
  }
  // bet wins when sign*value + line > 0. Returns {odds, W, push, L}
  function fairFromDist(d, line, sign = 1) {
    let W = 0, L = 0;
    for (const [v, p] of d) { const [w, l] = settle(sign * v, line); W += p * w; L += p * l; }
    return { odds: W > 0 ? 1 + L / W : Infinity, W, push: 1 - W - L, L };
  }
  function bisect(f, lo, hi, it = 100) {
    let flo = f(lo);
    for (let i = 0; i < it; i++) { const m = (lo + hi) / 2, fm = f(m); if ((fm > 0) === (flo > 0)) { lo = m; flo = fm; } else hi = m; }
    return (lo + hi) / 2;
  }
  // fair probabilities -> priced odds with overround 1+margin (power method, as pricing.apply_margin)
  function applyMargin(probs, margin) {
    const p = probs.map(x => Math.max(x, 1e-12)), t = 1 + margin;
    const k = bisect(k => p.reduce((s, x) => s + Math.pow(x, k), 0) - t, 0.2, 1.0);
    return p.map(x => Math.max(1.01, 1 / Math.pow(x, k)));
  }
  // two-way market with push: margin applied on the win/lose stake fractions
  function twoWay(oddsA, oddsB, margin) {
    const pa = 1 / oddsA, pb = 1 / oddsB, s = pa + pb;   // fair effective probabilities (sum 1)
    return applyMargin([pa / s, pb / s], margin);
  }
  function ladder(o) {           // round DOWN to the house ladder
    const step = o < 2 ? 0.01 : o < 3 ? 0.02 : o < 5 ? 0.05 : o < 10 ? 0.1 : o < 20 ? 0.5 : 1;
    return Math.max(1.01, Math.floor(o / step + 1e-9) * step);
  }
  function htft(lh, la, rho, share = 0.45) {
    const H1 = scoreMatrix(lh * share, la * share, rho * 0.5), H2 = scoreMatrix(lh * (1 - share), la * (1 - share), 0);
    const lab = (x, y) => x > y ? "1" : x === y ? "X" : "2", out = {};
    const n = H1.length;
    for (let a1 = 0; a1 < n; a1++) for (let b1 = 0; b1 < n; b1++) {
      const p1 = H1[a1][b1]; if (p1 < 1e-10) continue;
      for (let a2 = 0; a2 < n; a2++) for (let b2 = 0; b2 < n; b2++) {
        const p = p1 * H2[a2][b2]; if (p < 1e-12) continue;
        const k = lab(a1, b1) + "/" + lab(a1 + a2, b1 + b2); out[k] = (out[k] || 0) + p;
      }
    }
    return { htft: out, ht1x2: oneXTwo(H1) };
  }
  function statCovariates(lh, la, t0 = Math.log(2.7)) {
    const z = Math.log(lh / la), t = Math.log(lh + la) - t0;
    return [[z, t], [-z, t]];
  }
  function statMeans(r, home, away, lh, la) {
    const [xh, xa] = statCovariates(lh, la);
    const bh = (r.beta || []).reduce((s, b, i) => s + b * xh[i], 0);
    const ba = (r.beta || []).reduce((s, b, i) => s + b * xa[i], 0);
    return [
      Math.exp(r.mu + r.H + (r.att[home] || 0) + (r.dfn[away] || 0) + bh),
      Math.exp(r.mu + (r.att[away] || 0) + (r.dfn[home] || 0) + ba)
    ];
  }
  function nbPmfVec(mu, k, n) {
    const out = new Array(n + 1).fill(0);
    if (!k || k >= 1e6) {
      let lp = -mu;
      for (let y = 0; y <= n; y++) { out[y] = Math.exp(lp); lp += Math.log(mu) - Math.log(y + 1); }
      return out;
    }
    const q = mu / (k + mu);
    let lp = k * Math.log(k / (k + mu));
    for (let y = 0; y <= n; y++) { out[y] = Math.exp(lp); lp += Math.log((y + k) / (y + 1) * q); }
    return out;
  }
  function betabinomRow(t, p, c) {
    const out = new Array(t + 1).fill(0);
    if (c <= 1e-9) {
      const lq = Math.log(1 - p), lr = Math.log(p) - Math.log(1 - p);
      let lp = t * lq;
      for (let y = 0; y <= t; y++) { out[y] = Math.exp(lp); if (y < t) lp += Math.log((t - y) / (y + 1)) + lr; }
      return out;
    }
    const a = p / c, b = (1 - p) / c, ab = a + b;
    let lp = 0;
    for (let j = 0; j < t; j++) lp += Math.log((b + j) / (ab + j));
    for (let y = 0; y <= t; y++) {
      out[y] = Math.exp(lp);
      if (y < t) lp += Math.log((t - y) / (y + 1) * (y + a) / (t - y - 1 + b));
    }
    return out;
  }
  function statNmax(mh, ma, k) {
    const mt = mh + ma, sd = Math.sqrt(mt + (k && k < 1e6 ? mt * mt / k : 0));
    return Math.ceil(mt + 8 * sd + 5);
  }
  function statJoint(mh, ma, k, c, nmax) {
    nmax = nmax || statNmax(mh, ma, k);
    const pt = nbPmfVec(mh + ma, k, nmax), s = pt.reduce((a, b) => a + b, 0);
    for (let i = 0; i <= nmax; i++) pt[i] /= s;
    const p = mh / (mh + ma);
    const J = Array.from({ length: nmax + 1 }, () => new Array(nmax + 1).fill(0));
    for (let t = 0; t <= nmax; t++) {
      if (pt[t] < 1e-15) continue;
      const row = betabinomRow(t, p, c);
      for (let y = 0; y <= t; y++) J[y][t - y] += pt[t] * row[y];
    }
    return J;
  }
  return { pois, scoreMatrix, oneXTwo, dist, settle, fairFromDist, applyMargin, twoWay, ladder, htft,
           statCovariates, statMeans, nbPmfVec, betabinomRow, statNmax, statJoint };
})();
if (typeof module !== "undefined") module.exports = PE;
