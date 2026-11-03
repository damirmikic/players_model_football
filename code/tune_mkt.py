"""Shared (all-league) tuning of ridge / half-life / previous-season weight for market-λ targets (mkt_w = 1).

    python tune_mkt.py            # print grid, write ../out/mkt_grid.json
    python tune_mkt.py --save     # also write the chosen values into model_config.json "fit"

Closing-market λ is far less noisy than goals, so less shrinkage and faster decay may pay off.
Criterion: 1X2 log-loss pooled over 5 leagues on 2025-26; 2026-27 is the holdout (reported, not used to choose)."""
import io, itertools, json, math, os, sys, contextlib
from multiprocessing import Pool
import load, leagues, team_model as tm, pricing as P

RIDGE = [0.5, 1.0, 2.5, 5.0]
HALF_LIFE = [90, 180, 365, 730]
PREV_W = [0.5, 0.8, 1.0]
SEASONS = ("2025-26", "2026-27")

_M = {}


def _eval(args):
    key, ridge, hl, pw = args
    load.set_league(key)
    if key not in _M:
        with contextlib.redirect_stdout(io.StringIO()):
            _M[key] = load.build()[2]
    cfg = load.config()
    kw = {"ridge": ridge, "half_life_days": hl, "prev_season_weight": pw}
    out = {}
    for s in SEASONS:
        pr = tm.walk_forward(_M[key], s, cfg["xg_w"], rho=cfg["rho"], fit_kw=kw, promo_shift=cfg["promo_shift"],
                             mkt_w=cfg.get("mkt_w", 0.0))
        out[s] = (sum(-math.log(p["p"][p["o"]]) for p in pr), sum(P.rps(p["p"], p["o"]) for p in pr), len(pr))
    return (ridge, hl, pw), out


if __name__ == "__main__":
    keys = list(leagues.LEAGUES)
    combos = list(itertools.product(RIDGE, HALF_LIFE, PREV_W))
    with Pool(8) as pool:
        res = pool.map(_eval, [(k, *c) for k in keys for c in combos], chunksize=len(combos))
    agg = {}
    for c, o in res:
        a = agg.setdefault(c, {s: [0.0, 0.0, 0] for s in SEASONS})
        for s in SEASONS:
            for i in range(3):
                a[s][i] += o[s][i]
    rows = [{"ridge": c[0], "half_life_days": c[1], "prev_season_weight": c[2],
             **{f"{s}_{m}": a[s][i] / a[s][2] for s in SEASONS for i, m in ((0, "ll"), (1, "rps"))}, "n": a["2025-26"][2]}
            for c, a in agg.items()]
    cfg = load.config()["fit"]
    cur = next(r for r in rows if (r["ridge"], r["half_life_days"], r["prev_season_weight"]) ==
               (cfg["ridge"], cfg["half_life_days"], cfg["prev_season_weight"]))
    best = min(rows, key=lambda r: r["2025-26_ll"])
    fmt = lambda r: (f"ridge {r['ridge']:3.1f} hl {r['half_life_days']:3d} prev_w {r['prev_season_weight']:.1f} | "
                     f"25/26 LL {r['2025-26_ll']:.4f} RPS {r['2025-26_rps']:.5f} | 26/27 LL {r['2026-27_ll']:.4f} RPS {r['2026-27_rps']:.5f}")
    for r in sorted(rows, key=lambda r: r["2025-26_ll"])[:8]:
        print("  top", fmt(r))
    print("current:", fmt(cur)); print("chosen: ", fmt(best))
    json.dump({"rows": rows, "current": cur, "best": best}, open(os.path.join(load.RAW, "..", "out", "mkt_grid.json"), "w"), indent=1)
    if "--save" in sys.argv:
        path = os.path.join(load.RAW, "..", "model_config.json")
        c = json.load(open(path, encoding="utf-8"))
        c["fit"] = {k: best[k] for k in ("ridge", "half_life_days", "prev_season_weight")}
        json.dump(c, open(path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
        print("saved to model_config.json")
