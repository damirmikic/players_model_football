"""Fit the Dixon-Coles rho per league on walk-forward predictions (λ adjusted for the actual lineup with the
league's beta, as in the calculator). Criterion: exact-score log-likelihood — rho only moves 0-0/1-0/0-1/1-1.

    python fit_rho.py                  # all leagues, print grid
    python fit_rho.py --save           # also write per-league rho into ../model_config.json"""
import io, json, math, sys, contextlib
import numpy as np
import load, leagues, pricing as P
from fit_betas import base_preds

GRID = [round(x, 2) for x in np.arange(-0.30, 0.1001, 0.02)]


def league_pairs(key):
    load.set_league(key)
    with contextlib.redirect_stdout(io.StringIO()):
        T, PL, M, PM, cur = load.build()
        bp = base_preds(M)
    cfg = load.config()
    LF = {(f["season"], f["home"], f["away"]): f for f in json.load(open(load.out("lineup_factors.json")))}
    out = []
    for k, p in bp.items():
        f = LF.get(k)
        lh, la = p["lh"], p["la"]
        if f:
            lh *= f["h_ratio"] ** cfg["beta_a"] * math.exp(cfg["beta_d"] * f["a_dd"])
            la *= f["a_ratio"] ** cfg["beta_a"] * math.exp(cfg["beta_d"] * f["h_dd"])
        out.append((lh, la, p["hg"], p["ag"], p["o"]))
    return out, cfg["rho"]


def evaluate(rows, rho):
    nll = rps = dp = 0.0
    for lh, la, hg, ag, o in rows:
        M = P.score_matrix(lh, la, rho)
        nll -= math.log(max(M[min(hg, 12), min(ag, 12)], 1e-12))
        pr = P.one_x_two(M)
        rps += P.rps(pr, o); dp += pr[1]
    n = len(rows)
    return {"nll": nll / n, "rps": rps / n, "draw_pred": dp / n, "draw_obs": sum(r[4] == 1 for r in rows) / n, "n": n}


if __name__ == "__main__":
    save = "--save" in sys.argv
    best, pooled = {}, {}
    for key in leagues.LEAGUES:
        rows, cur_rho = league_pairs(key)
        res = {r: evaluate(rows, r) for r in GRID}
        b = min(res, key=lambda r: res[r]["nll"])
        # 2-unit likelihood interval: rho values within 1 log-lik point (total) of the best
        tot = {r: res[r]["nll"] * res[r]["n"] for r in GRID}
        ci = [r for r in GRID if tot[r] - tot[b] <= 1.0]
        best[key] = b = float(b)
        print(f"== {key} n={res[b]['n']}  draw obs {res[b]['draw_obs']:.3f}  current rho {cur_rho}")
        for r in GRID:
            x = res[r]
            print(f"  rho {r:+.2f}  score-NLL {x['nll']:.4f}  RPS {x['rps']:.5f}  draw pred {x['draw_pred']:.3f}"
                  + ("  <- best" if r == b else ""))
        print(f"  best {b:+.2f}, ~95% interval [{min(ci):+.2f}, {max(ci):+.2f}]")
        for r in GRID:
            pooled[r] = pooled.get(r, 0.0) + tot[r]
    pb = float(min(pooled, key=pooled.get))
    print(f"== pooled best rho {pb:+.2f}; per league {best}")
    if save:
        path = "../model_config.json"
        c = json.load(open(path, encoding="utf-8"))
        c["rho"] = pb
        for key, r in best.items():
            c.setdefault("leagues", {}).setdefault(key, {})["rho"] = r
        c["_rho_note"] = ("rho per league = argmin exact-score NLL (fit_rho.py), walk-forward 2025/26-2026/27, "
                          "lineup-adjusted λ; default = pooled optimum.")
        json.dump(c, open(path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
        print("saved to", path)
