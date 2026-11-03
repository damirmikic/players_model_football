"""Tune the team-rating model per league: xg_w, ridge, half_life_days, prev_season_weight, then promo_shift.
Walk-forward over 2025-26 + 2026-27; criterion = exact-score NLL (league rho), RPS reported alongside.

    python tune_team.py [LEAGUE ...]        # print results, write ../out/<league>/team_grid.json
    python tune_team.py --save [LEAGUE ...] # also write the chosen values into ../model_config.json

Note: in-sample optimum overfits with ~2 seasons of data; check holdout before using --save (see _team_note in config)."""
import io, itertools, json, math, sys, contextlib
from multiprocessing import Pool
import numpy as np
import load, leagues, team_model as tm, pricing as P

XG_W = [0.0, 0.25, 0.5, 0.7, 0.85, 1.0]
RIDGE = [1.0, 2.5, 5.0, 10.0]
HALF_LIFE = [120, 240, 365, 730]
PREV_W = [0.5, 0.65, 0.8, 1.0]
PROMO = [0.0, 0.04, 0.08, 0.12, 0.18, 0.25, 0.35]

_M = _RHO = None


def _init(key):
    global _M, _RHO
    load.set_league(key)
    with contextlib.redirect_stdout(io.StringIO()):
        _M = load.build()[2]
    _RHO = load.config()["rho"]


def _eval(args):
    xg_w, ridge, hl, pw, promo = args
    kw = {"ridge": ridge, "half_life_days": hl, "prev_season_weight": pw}
    pr = []
    for s in ("2025-26", "2026-27"):
        pr += tm.walk_forward(_M, s, xg_w, rho=_RHO, fit_kw=kw, promo_shift=promo)
    nll = rps = 0.0
    for p in pr:
        M = P.score_matrix(p["lh"], p["la"], _RHO)
        nll -= math.log(max(M[min(p["hg"], 12), min(p["ag"], 12)], 1e-12))
        rps += P.rps(P.one_x_two(M), p["o"])
    return {"xg_w": xg_w, "ridge": ridge, "half_life_days": hl, "prev_season_weight": pw, "promo_shift": promo,
            "nll": nll / len(pr), "rps": rps / len(pr), "n": len(pr)}


def tune(key):
    _init(key)
    cfg = load.config()
    cur = (cfg["xg_w"], cfg["fit"]["ridge"], cfg["fit"]["half_life_days"], cfg["fit"]["prev_season_weight"], cfg["promo_shift"])
    with Pool(8, initializer=_init, initargs=(key,)) as pool:
        base = pool.map(_eval, [cur])[0]
        grid = pool.map(_eval, [(*c, cfg["promo_shift"]) for c in itertools.product(XG_W, RIDGE, HALF_LIFE, PREV_W)])
        b = min(grid, key=lambda r: r["nll"])
        promo = pool.map(_eval, [(b["xg_w"], b["ridge"], b["half_life_days"], b["prev_season_weight"], p) for p in PROMO])
    best = min(promo, key=lambda r: r["nll"])
    json.dump({"current": base, "grid": grid, "promo": promo, "best": best}, open(load.out("team_grid.json"), "w"), indent=1)
    return base, best, grid


if __name__ == "__main__":
    save = "--save" in sys.argv
    keys = [a for a in sys.argv[1:] if not a.startswith("--")] or list(leagues.LEAGUES)
    fmt = lambda r: (f"xg_w {r['xg_w']:.2f} ridge {r['ridge']:4.1f} hl {r['half_life_days']:3d} prev_w {r['prev_season_weight']:.2f} "
                     f"promo {r['promo_shift']:.2f} | NLL {r['nll']:.4f} RPS {r['rps']:.5f}")
    chosen = {}
    for key in keys:
        base, best, grid = tune(key)
        n = best["n"]
        print(f"== {key} (n={n})\n  current: {fmt(base)}\n  best:    {fmt(best)}\n"
              f"  gain: NLL {base['nll'] - best['nll']:+.4f} (total log-lik {n * (base['nll'] - best['nll']):+.1f}), "
              f"RPS {base['rps'] - best['rps']:+.5f}", flush=True)
        for r in sorted(grid, key=lambda r: r["nll"])[:5]:
            print("   top5", fmt(r))
        chosen[key] = best
    if save:
        path = "../model_config.json"
        c = json.load(open(path, encoding="utf-8"))
        for key, b in chosen.items():
            o = c.setdefault("leagues", {}).setdefault(key, {})
            o.update(xg_w=b["xg_w"], promo_shift=b["promo_shift"],
                     fit={"ridge": b["ridge"], "half_life_days": b["half_life_days"], "prev_season_weight": b["prev_season_weight"]})
        c["_team_note"] = ("xg_w / fit / promo_shift per league = argmin exact-score NLL (tune_team.py), walk-forward "
                           "2025/26-2026/27; grid then promo_shift line search.")
        json.dump(c, open(path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
        print("saved to", path)
