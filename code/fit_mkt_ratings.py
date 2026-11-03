"""Team ratings learned from the market's view of PAST matches, for pricing a match before its own odds exist.

    python fit_mkt_ratings.py        # all leagues, ~minutes (market λ cached in ../out/<league>/mkt_lambda.json)

For every finished match with closing 1X2 + O/U 2.5 (football-data.co.uk) solve (λh, λa) reproducing the fair
prices; the team model then fits on target (1-m)·(goals/xG blend) + m·market λ. Walk-forward with a date cut,
so a round is priced only from matches (and their closing odds) before it — the market for the round itself is
never used. m chosen on 2025-26, scored on 2026-27, against results and against the pre-match market."""
import io, json, math, os, contextlib, sys
from multiprocessing import Pool
import numpy as np
import load, leagues, team_model as tm, pricing as P

GRID = (0.0, 0.25, 0.5, 0.75, 1.0)


def fair(odds):
    inv = [1 / o for o in odds]
    return [x / sum(inv) for x in inv]



def run(args):
    key, mw = args
    load.set_league(key)
    cfg = load.config()
    with contextlib.redirect_stdout(io.StringIO()):
        T, PL, M, PM, cur = load.build()
    mi = {(m["season"], m["home"], m["away"]): m for m in M}
    out = []
    for s in ("2025-26", "2026-27"):
        for p in tm.walk_forward(M, s, cfg["xg_w"], rho=cfg["rho"], fit_kw=cfg["fit"], promo_shift=cfg["promo_shift"], mkt_w=mw):
            m = mi[(s, p["home"], p["away"])]
            out.append({"league": key, "season": s, "m": mw, "p": p["p"], "o": p["o"], "lh": p["lh"], "la": p["la"],
                        "po": m.get("odds_open") and fair(m["odds_open"]), "pc": m.get("odds_close") and fair(m["odds_close"])})
    return out


def summary(rows):
    rps = np.mean([P.rps(r["p"], r["o"]) for r in rows])
    ll = np.mean([-math.log(r["p"][r["o"]]) for r in rows])
    wm = [r for r in rows if r["po"]]
    # distance to the pre-match market: KL(market || model) and mean |ΔP(home)|
    kl = np.mean([sum(q * math.log(q / p) for q, p in zip(r["po"], r["p"])) for r in wm])
    dh = np.mean([abs(r["po"][0] - r["p"][0]) for r in wm])
    return rps, ll, kl, dh


if __name__ == "__main__":
    keys = sys.argv[1:] or list(leagues.LEAGUES)
    for k in keys:                                  # fill the λ cache once per league before the parallel runs
        load.set_league(k)
        with contextlib.redirect_stdout(io.StringIO()):
            load.build()                            # computes and caches market λ
    with Pool(8) as pool:
        rows = [r for rs in pool.map(run, [(k, m) for k in keys for m in GRID]) for r in rs]
    tr = {m: [r for r in rows if r["m"] == m and r["season"] == "2025-26"] for m in GRID}
    best = min(GRID, key=lambda m: summary(tr[m])[1])
    for s in ("2025-26", "2026-27"):
        print(f"== {s}" + ("  (choice)" if s == "2025-26" else "  (holdout)"))
        for m in GRID:
            rs = [r for r in rows if r["m"] == m and r["season"] == s]
            rps, ll, kl, dh = summary(rs)
            print(f"   m={m:<4}  n={len(rs)}  RPS {rps:.5f}  logloss {ll:.4f}  | vs pre-match market: KL {kl:.4f}  |ΔP1| {dh:.4f}"
                  + ("   <- chosen" if m == best else ""))
        wm = [r for r in rows if r["m"] == 0.0 and r["season"] == s and r["po"]]
        print(f"   pre-match market n={len(wm)}  RPS {np.mean([P.rps(r['po'], r['o']) for r in wm]):.5f}"
              f"  logloss {np.mean([-math.log(r['po'][r['o']]) for r in wm]):.4f}")
    print(f"chosen m={best}")
    for lg in keys:
        rs = {m: [r for r in rows if r["m"] == m and r["season"] == "2026-27" and r["league"] == lg] for m in (0.0, best)}
        a, b = summary(rs[0.0]), summary(rs[best])
        print(f"   holdout {lg:11s} RPS {a[0]:.5f} -> {b[0]:.5f}   |ΔP1| {a[3]:.4f} -> {b[3]:.4f}")
