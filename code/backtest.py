"""Walk-forward backtest summary for the calculator (../out/<league>/backtest.json).

    python backtest.py [LEAGUE]

Benchmarks: previous-season base rates, and the closing market (football-data.co.uk, Betfair exchange
closing when available, else market average closing; margin removed proportionally) on the same matches."""
import json
import numpy as np
import load, team_model as tm, pricing as P
from fit_betas import base_preds, evaluate

SEASONS = ("2025-26", "2026-27")


def market_probs(odds):
    inv = [1 / o for o in odds]
    s = sum(inv)
    return [x / s for x in inv]


if __name__ == "__main__":
    load.league_from_argv()
    cfg = load.config()
    T, PL, M, PM, cur = load.build()
    bp = base_preds(M)
    preds = list(bp.values())
    sc = tm.score(preds)
    # baseline: previous-season H/D/A rates, weighted by matches per season
    fin = {s: sum(1 for m in M if m["season"] == s and m["finished"]) for s in SEASONS}
    base = sum(tm.baseline(M, s) * fin[s] for s in SEASONS) / sum(fin.values())
    # lineup adjustment on matches with player data
    LF = json.load(open(load.out("lineup_factors.json")))
    mi = {(m["season"], m["home"], m["away"]): m for m in M}
    pairs = [(bp[k], f, mi[k]) for f in LF for k in [(f["season"], f["home"], f["away"])] if k in bp]
    # lineup effects as the calculator applies them (injuries + lineup surprises, centred; fit_lineup.py)
    import fit_lineup
    lrows = fit_lineup.rows_for(load.LEAGUE)
    with_lu = fit_lineup.score(lrows, cfg.get("beta_inj", 0.0), cfg.get("beta_rot", 0.0))
    no_lu = fit_lineup.score(lrows, 0.0, 0.0)
    # closing market on the same matches
    mk = [(p, mi[(p["season"], p["home"], p["away"])]) for p in preds]
    mk = [(p, m) for p, m in mk if m.get("odds_close")]
    rps_mkt = float(np.mean([P.rps(market_probs(m["odds_close"]), p["o"]) for p, m in mk])) if mk else None
    rps_mod_mkt = float(np.mean([P.rps(p["p"], p["o"]) for p, m in mk])) if mk else None
    src = sorted({m["odds_src"] for _, m in mk})
    out = {"n_matches": sc["n"], "period": "2025/26 – 2026/27 (walk-forward)",
           "rps_team": sc["rps"], "rps_baseline": round(base, 5),
           "skill_vs_baseline": round(1 - sc["rps"] / base, 4),
           "rps_with_lineup": round(with_lu["rps"], 5), "rps_team_same_matches": round(no_lu["rps"], 5),
           "lineup_n": with_lu["n"], "beta_inj": cfg.get("beta_inj", 0.0), "beta_rot": cfg.get("beta_rot", 0.0), "draw_pred": sc["draw_pred"], "draw_obs": sc["draw_obs"],
           "market_n": len(mk), "rps_market": rps_mkt and round(rps_mkt, 5),
           "rps_model_market_matches": rps_mod_mkt and round(rps_mod_mkt, 5),
           "market_src": "+".join(src)}
    json.dump(out, open(load.out("backtest.json"), "w"), indent=1)
    print(out)
