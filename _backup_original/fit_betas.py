"""Fit lineup elasticities beta_a, beta_d on top of walk-forward team predictions."""
import json, math, itertools
import numpy as np
import load, team_model as tm, pricing as P

CFG = json.load(open("../model_config.json"))


def base_preds(M):
    kw = CFG["fit"]
    out = {}
    for s in ("2025-26", "2026-27"):
        for p in tm.walk_forward(M, s, CFG["xg_w"], rho=CFG["rho"], fit_kw=kw, promo_shift=CFG["promo_shift"]):
            out[(s, p["home"], p["away"])] = p
    return out


def evaluate(pairs, ba, bd, rho):
    rps = ll = gnll = xnll = 0.0
    for p, f, m in pairs:
        lh = p["lh"] * f["h_ratio"] ** ba * math.exp(bd * f["a_dd"])
        la = p["la"] * f["a_ratio"] ** ba * math.exp(bd * f["h_dd"])
        pr = P.one_x_two(P.score_matrix(lh, la, rho))
        rps += P.rps(pr, p["o"]); ll -= math.log(pr[p["o"]])
        gnll += lh - p["hg"] * math.log(lh) + la - p["ag"] * math.log(la)
        xnll += lh - m["hxg"] * math.log(lh) + la - m["axg"] * math.log(la)   # quasi-Poisson on xG
    n = len(pairs)
    return {"ba": ba, "bd": bd, "rps": rps / n, "logloss": ll / n, "goal_nll": gnll / n, "xg_qnll": xnll / n, "n": n}


if __name__ == "__main__":
    T, PL, M, PM, cur = load.build()
    bp = base_preds(M)
    LF = json.load(open("../out_lineup_factors.json"))
    mi = {(m["season"], m["home"], m["away"]): m for m in M}
    pairs = [(bp[(f["season"], f["home"], f["away"])], f, mi[(f["season"], f["home"], f["away"])]) for f in LF]
    res = [evaluate(pairs, ba, bd, CFG["rho"]) for ba, bd in
           itertools.product([0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5], [0, 1, 2, 3, 4])]
    for key in ("rps", "goal_nll", "xg_qnll"):
        b = min(res, key=lambda r: r[key]); print("best by", key, {k: round(v, 5) for k, v in b.items()})
    z = [r for r in res if r["ba"] == 0 and r["bd"] == 0][0]
    print("no lineup adj ", {k: round(v, 5) for k, v in z.items()})
    for r in res:
        if r["bd"] == 0: print("ba", r["ba"], round(r["rps"], 5), round(r["goal_nll"], 4), round(r["xg_qnll"], 4))
    for r in res:
        if r["ba"] == 0.75: print("bd", r["bd"], round(r["rps"], 5), round(r["goal_nll"], 4), round(r["xg_qnll"], 4))
    # bootstrap the rps gain of the chosen setting vs none
    rng = np.random.default_rng(7)
    def per(ba, bd):
        v = []
        for p, f, m in pairs:
            lh = p["lh"] * f["h_ratio"] ** ba * math.exp(bd * f["a_dd"]); la = p["la"] * f["a_ratio"] ** ba * math.exp(bd * f["h_dd"])
            v.append(P.rps(P.one_x_two(P.score_matrix(lh, la, CFG["rho"])), p["o"]))
        return np.array(v)
    best = min(res, key=lambda r: r["xg_qnll"])
    g = per(0, 0) - per(best["ba"], best["bd"])
    bs = [g[rng.integers(0, len(g), len(g))].mean() for _ in range(2000)]
    print("RPS gain %.5f, 90%% CI [%.5f, %.5f]" % (g.mean(), np.percentile(bs, 5), np.percentile(bs, 95)))
