"""Match-statistics model: shots, shots on target, fouls, corners, yellow cards (team + match markets).

    python stat_model.py [LEAGUE ...]            # walk-forward backtest with model_config "stats" params
    python stat_model.py [LEAGUE ...] --save     # + refit dispersion (k, c) pooled over leagues -> model_config.json
    python stat_model.py [LEAGUE ...] --tune     # grid ridge x half-life (select 2025-26, holdout 2026-27)

Per stat and league (pricing.fit_count_ratings), penalized time-decayed Poisson GLM:
    log mu_home = mu + H + att[home] + dfn[away] + b_z*z + b_t*t,   log mu_away = mu + att[away] + dfn[home] - b_z*z + b_t*t
    z = log(lh/la), t = log(lh+la) - log 2.7 from the goals model. Training rows use each past match's
    closing-market lambdas (m["mkt_lam"], as the goals model does); predictions use the goals model's lambdas,
    so the market of the match being priced is never used.
Joint distribution (pricing.stat_joint): total ~ NB(mu_h+mu_a, k), home | total ~ BetaBinomial(share, c).
k, c per stat are pooled over the five leagues and fitted by ML on walk-forward (out-of-sample) means.
Data: football-data.co.uk HS/AS, HST/AST, HF/AF, HC/AC, HY/AY (load.attach_odds -> m["stats"]).
Outputs ../out/<league>/stats_backtest.json (and ../out/stats_tuning.json with --tune)."""
import json, math, os, sys, datetime as dt
from collections import defaultdict
import numpy as np
import load, leagues, team_model as tm, pricing as P

STATS = ["shots", "sot", "corners", "fouls", "yellow"]
SEASONS = ("2025-26", "2026-27")
DEFAULT_FIT = {"ridge": 5.0, "half_life_days": 240, "prev_season_weight": 0.7}
# usual book lines, for the calibration check of P(over)
MAIN_LINE = {"shots": 24.5, "sot": 8.5, "corners": 9.5, "fouls": 23.5, "yellow": 3.5}


def stat_cfg(cfg, stat):
    """Fit params + dispersion for one stat: cfg["stats"]["fit"], overridden by cfg["stats"][stat]."""
    sc = cfg.get("stats", {})
    out = {**DEFAULT_FIT, **sc.get("fit", {}), "k": 50.0, "c": 0.0, "covariates": sc.get("covariates", True)}
    out.update(sc.get(stat, {}))
    return out


def fit_kw(sc):
    return {k: sc[k] for k in ("ridge", "half_life_days", "prev_season_weight")}


def rows_for(matches, stat, cur_season, covariates=True):
    """Finished matches up to cur_season with this stat and market lambdas -> fit_count_ratings rows."""
    si = tm.SEASON_IDX
    out = []
    for m in matches:
        if not m["finished"] or si[m["season"]] > si[cur_season] or stat not in m.get("stats", {}) or not m.get("mkt_lam"):
            continue
        xh, xa = P.stat_covariates(*m["mkt_lam"]) if covariates else ([], [])
        yh, ya = m["stats"][stat]
        out.append({"home": m["home"], "away": m["away"], "yh": yh, "ya": ya, "xh": xh, "xa": xa,
                    "date": dt.date.fromisoformat(m["date"]), "round": m["round"],
                    "season": 1 if m["season"] == cur_season else 0})
    return out


def fit(matches, stat, asof, teams, sc, cur_season=load.CURRENT, variant="full"):
    """Ratings for one stat from matches strictly before asof. variant: full | teams (no covariates) | league."""
    rows = [r for r in rows_for(matches, stat, cur_season, variant == "full") if r["date"] < asof]
    return P.fit_count_ratings(rows, asof=asof, current_teams=teams, use_teams=variant != "league", **fit_kw(sc))


def means(r, home, away, lh, la, variant="full"):
    xh, xa = P.stat_covariates(lh, la) if variant == "full" else ([], [])
    return P.count_means(r, home, away, xh, xa)


def goal_preds(M):
    """Walk-forward goals-model lambdas per (season, home, away), exactly as in backtest.py."""
    cfg = load.config()
    out = {}
    for s in SEASONS:
        for p in tm.walk_forward(M, s, cfg["xg_w"], rho=cfg["rho"], fit_kw=lambda k: tm.fit_kw_for(cfg, k),
                                 promo_shift=cfg["promo_shift"], mkt_w=cfg.get("mkt_w", 0.0)):
            out[(s, p["home"], p["away"])] = p
    return out


def walk_forward(M, GP, stats=STATS, fits=None, variants=("full",)):
    """Out-of-sample means: for each round of each backtest season refit on matches before the round.
    fits: {stat: fit kwargs}. Returns {(stat, variant): [obs]}, obs = dict(season, round, yh, ya, mh, ma)."""
    fits = fits or {s: fit_kw(stat_cfg(load.config(), s)) for s in stats}
    out = defaultdict(list)
    for season in SEASONS:
        cur = [m for m in M if m["season"] == season and m["finished"]]
        teams = sorted({m["home"] for m in cur} | {m["away"] for m in cur})
        for k in range(1, max(m["round"] for m in cur) + 1):
            test = [m for m in cur if m["round"] == k and (season, m["home"], m["away"]) in GP]
            if not test:
                continue
            asof = min(dt.date.fromisoformat(m["date"]) for m in test)
            for stat in stats:
                for v in variants:
                    r = fit(M, stat, asof, teams, fits[stat], season, v)
                    for m in test:
                        if stat not in m.get("stats", {}):
                            continue
                        g = GP[(season, m["home"], m["away"])]
                        mh, ma = means(r, m["home"], m["away"], g["lh"], g["la"], v)
                        yh, ya = m["stats"][stat]
                        out[(stat, v)].append({"season": season, "round": k, "yh": yh, "ya": ya, "mh": mh, "ma": ma})
    return out


def score(obs, k, c, stat):
    """Mean joint NLL, total NLL, total RPS-style (CRPS over counts), mean bias, calibration at MAIN_LINE."""
    if not obs:
        return {}
    nll = tot = crps = 0.0
    ln = MAIN_LINE[stat]
    p_over, o_over = [], []
    for o in obs:
        lt, ls = P.stat_loglik(o["yh"], o["ya"], o["mh"], o["ma"], k, c)
        nll -= lt + ls; tot -= lt
        mt = o["mh"] + o["ma"]
        n = P.stat_nmax(o["mh"], o["ma"], k)
        pt = P.nb_pmf_vec(mt, k, n); pt /= pt.sum()
        cdf = np.cumsum(pt); y = o["yh"] + o["ya"]
        crps += float(np.sum((cdf - (np.arange(n + 1) >= y)) ** 2))
        p_over.append(float(1 - cdf[int(math.floor(ln))])); o_over.append(y > ln)
    n = len(obs)
    return {"n": n, "nll": round(nll / n, 4), "total_nll": round(tot / n, 4), "crps_total": round(crps / n, 4),
            "mean_pred": round(float(np.mean([o["mh"] + o["ma"] for o in obs])), 2),
            "mean_obs": round(float(np.mean([o["yh"] + o["ya"] for o in obs])), 2),
            "line": ln, "p_over_pred": round(float(np.mean(p_over)), 3), "p_over_obs": round(float(np.mean(o_over)), 3)}


def collect(keys, fits=None, variants=("full", "teams", "league")):
    """Walk-forward obs for several leagues: {(league, stat, variant): [obs]}."""
    allobs = {}
    for key in keys:
        load.set_league(key)
        T, PL, M, PM, cur = load.build()
        GP = goal_preds(M)
        for (stat, v), obs in walk_forward(M, GP, fits=fits, variants=variants).items():
            allobs[(key, stat, v)] = obs
        print(f"[stats {key}] walk-forward done", flush=True)
    return allobs


def pooled(allobs, stat, v, season=None):
    return [o for (lg, s, vv), obs in allobs.items() if s == stat and vv == v for o in obs
            if season is None or o["season"] == season]


def report(allobs, keys, disp):
    """Per-league and pooled scores for all variants; dispersion of each variant fitted on 2025-26 (pooled)."""
    res = {}
    for stat in STATS:
        for v in ("full", "teams", "league"):
            d = disp.get((stat, v)) or P.fit_stat_dispersion(
                [(o["yh"], o["ya"], o["mh"], o["ma"]) for o in pooled(allobs, stat, v, SEASONS[0])])
            disp[(stat, v)] = d
            for lg in list(keys) + ["ALL"]:
                obs = pooled(allobs, stat, v) if lg == "ALL" else allobs.get((lg, stat, v), [])
                for s in SEASONS:
                    res.setdefault(lg, {}).setdefault(stat, {}).setdefault(v, {})[s] = \
                        score([o for o in obs if o["season"] == s], d["k"], d["c"], stat)
                res[lg][stat][v]["dispersion"] = d
    return res


def save_config(disp, fits=None):
    path = os.path.join(load.RAW, "..", "model_config.json")
    c = json.load(open(path, encoding="utf-8"))
    sc = c.setdefault("stats", {})
    sc.setdefault("fit", dict(DEFAULT_FIT))
    sc.setdefault("covariates", True)
    for stat in STATS:
        sc[stat] = {**sc.get(stat, {}), **disp[(stat, "full")], **((fits or {}).get(stat, {}))}
    json.dump(c, open(path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("saved stats block ->", path)


def print_table(res):
    print(f"\n{'stat':8s} {'variant':7s} {'k':>8s} {'c':>7s} | {'NLL 25/26':>9s} {'NLL 26/27':>9s} {'CRPS 26/27':>10s}"
          f" | mean pred/obs | P(over) pred/obs @line")
    for stat in STATS:
        for v in ("league", "teams", "full"):
            r = res["ALL"][stat][v]; a, b = r[SEASONS[0]], r[SEASONS[1]]
            print(f"{stat:8s} {v:7s} {r['dispersion']['k']:8.1f} {r['dispersion']['c']:7.4f} | {a['nll']:9.4f} {b['nll']:9.4f} "
                  f"{b['crps_total']:10.4f} | {b['mean_pred']:5.2f}/{b['mean_obs']:5.2f} | {b['p_over_pred']:.3f}/{b['p_over_obs']:.3f} @{b['line']}")


def tune(keys):
    grid = [{"ridge": r, "half_life_days": h, "prev_season_weight": w}
            for r in (2.0, 5.0, 15.0) for h in (120, 240, 480) for w in (0.7,)]
    out, best = [], {}
    for g in grid:
        allobs = collect(keys, fits={s: g for s in STATS}, variants=("full",))
        for stat in STATS:
            sel = pooled(allobs, stat, "full", SEASONS[0])
            d = P.fit_stat_dispersion([(o["yh"], o["ya"], o["mh"], o["ma"]) for o in sel])
            a = score(sel, d["k"], d["c"], stat); b = score(pooled(allobs, stat, "full", SEASONS[1]), d["k"], d["c"], stat)
            row = {"stat": stat, **g, **d, "nll_sel": a["nll"], "nll_hold": b["nll"]}
            out.append(row); print(row, flush=True)
            if stat not in best or row["nll_sel"] < best[stat]["nll_sel"]:
                best[stat] = row
    json.dump(out, open(os.path.join(load.RAW, "..", "out", "stats_tuning.json"), "w"), indent=1)
    print("BEST (selected on 2025-26):")
    for s, r in best.items():
        print(" ", r)
    return best


if __name__ == "__main__":
    args = sys.argv[1:]
    keys = [a for a in args if not a.startswith("--")] or list(leagues.LEAGUES)
    fits = None
    if "--tune" in args:
        best = tune(keys)
        fits = {s: {k: best[s][k] for k in ("ridge", "half_life_days", "prev_season_weight")} for s in STATS}
    allobs = collect(keys, fits=fits)
    disp = {}
    res = report(allobs, keys, disp)
    for lg in keys:
        load.set_league(lg)
        json.dump(res[lg], open(load.out("stats_backtest.json"), "w"), indent=1)
    json.dump(res["ALL"], open(os.path.join(load.RAW, "..", "out", "stats_backtest_all.json"), "w"), indent=1)
    print_table(res)
    if "--save" in args or "--tune" in args and "--save" in args:
        save_config(disp, fits)

