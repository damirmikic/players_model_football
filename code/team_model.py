"""Team ratings per league: penalized Poisson GLM (pricing.fit_ratings) on a goals/xG blend,
validated walk-forward over 2025-26 (full season) and 2026-27 (rounds so far)."""
import math, datetime as dt
import numpy as np
import pricing as P
import load

SEASON_IDX = {s: i for i, s in enumerate(load.SEASONS)}


def as_rows(matches, cur_season, xg_w, mkt_w=0.0):
    """Matches up to and including cur_season, flagged season 1 / 0, target = blend of goals and xG;
    with mkt_w > 0 also blended with the closing-market λ of that (past) match where available (m["mkt_lam"])."""
    out = []
    for m in matches:
        if not m["finished"] or SEASON_IDX[m["season"]] > SEASON_IDX[cur_season]:
            continue
        hx, ax = m.get("hxg"), m.get("axg")
        if hx is None or xg_w == 0:
            th, ta = m["hg"], m["ag"]
        else:
            th = (1 - xg_w) * min(m["hg"], 6) + xg_w * hx
            ta = (1 - xg_w) * min(m["ag"], 6) + xg_w * ax
        if mkt_w and m.get("mkt_lam"):
            th = (1 - mkt_w) * th + mkt_w * m["mkt_lam"][0]
            ta = (1 - mkt_w) * ta + mkt_w * m["mkt_lam"][1]
        out.append({"home": m["home"], "away": m["away"], "hg": th, "ag": ta, "rg": (m["hg"], m["ag"]),
                    "date": m["date"], "round": m["round"], "season": 1 if m["season"] == cur_season else 0})
    return out


def priors_for(rows, teams, shift=0.15):
    prev = {r["home"] for r in rows if r["season"] == 0} | {r["away"] for r in rows if r["season"] == 0}
    last_prev = {}
    for r in rows:   # did the team play in the immediately previous top-flight season?
        pass
    return {t: (P.new_team_prior("promoted", shift) if t not in prev else (0.0, 0.0)) for t in teams}


def walk_forward(matches, cur_season, xg_w=0.0, rho=-0.05, start_round=1, fit_kw=None, promoted=None,
                 promo_shift=0.25, mkt_w=0.0):
    fit_kw = fit_kw or {}
    rows = as_rows(matches, cur_season, xg_w, mkt_w)
    cur = [r for r in rows if r["season"] == 1]
    teams = sorted({r["home"] for r in cur} | {r["away"] for r in cur})
    promoted = promoted if promoted is not None else promoted_teams(matches, cur_season)
    prior = {t: P.new_team_prior("promoted", promo_shift) for t in promoted}
    prev_rows = [r for r in rows if r["season"] == 0]
    out = []
    for k in range(start_round, max(r["round"] for r in cur) + 1):
        test = [r for r in cur if r["round"] == k]
        if not test:
            continue
        asof = min(dt.date.fromisoformat(r["date"]) for r in test)
        train = prev_rows + [r for r in cur if dt.date.fromisoformat(r["date"]) < asof]
        kw = fit_kw(k) if callable(fit_kw) else fit_kw
        rt = P.fit_ratings(train, asof=asof, current_teams=teams, prior=prior, goal_cap=99, **kw)
        for r in test:
            lh, la = P.ratings_lambdas(rt, r["home"], r["away"])
            p = P.one_x_two(P.score_matrix(lh, la, rho))
            hg, ag = r["rg"]
            o = 0 if hg > ag else (1 if hg == ag else 2)
            out.append({"round": k, "home": r["home"], "away": r["away"], "lh": lh, "la": la, "p": p, "o": o,
                        "hg": hg, "ag": ag, "season": cur_season})
    return out


def fit_kw_for(cfg, rnd):
    """Rating-fit params for a match of round rnd: cfg["fit"], or cfg["fit_late"] from round cfg["late_round"] on
    (with market targets, later rounds prefer a shorter memory; tune_mkt.py)."""
    late = cfg.get("fit_late")
    return late if late and rnd >= cfg.get("late_round", 11) else cfg["fit"]


def promoted_teams(matches, season):
    i = SEASON_IDX[season]
    if i == 0:
        return []
    prev = {m["home"] for m in matches if m["season"] == load.SEASONS[i - 1]}
    cur = {m["home"] for m in matches if m["season"] == season}
    return sorted(cur - prev)


def score(preds, rho=None):
    """RPS, log-loss, goals Poisson log-lik (needs lh/la), draw calibration."""
    rps = np.mean([P.rps(p["p"], p["o"]) for p in preds])
    ll = np.mean([-math.log(max(p["p"][p["o"]], 1e-12)) for p in preds])
    gll = np.mean([-(P.math.log(P.poisson_pmf(p["hg"], p["lh"])) + math.log(P.poisson_pmf(p["ag"], p["la"]))) for p in preds])
    return {"n": len(preds), "rps": round(float(rps), 5), "logloss": round(float(ll), 5), "goal_nll": round(float(gll), 4),
            "draw_pred": round(float(np.mean([p["p"][1] for p in preds])), 3),
            "draw_obs": round(float(np.mean([p["o"] == 1 for p in preds])), 3)}


def baseline(matches, season):
    prev = [m for m in matches if m["season"] == load.SEASONS[SEASON_IDX[season] - 1] and m["finished"]]
    bh = np.mean([m["hg"] > m["ag"] for m in prev]); bd = np.mean([m["hg"] == m["ag"] for m in prev])
    cur = [m for m in matches if m["season"] == season and m["finished"]]
    return float(np.mean([P.rps([bh, bd, 1 - bh - bd], 0 if m["hg"] > m["ag"] else (1 if m["hg"] == m["ag"] else 2)) for m in cur]))


if __name__ == "__main__":
    import itertools, json, sys
    load.league_from_argv()
    T, PL, M, PM, cur = load.build()
    for s in ["2025-26", "2026-27"]:
        print(s, "promoted:", [T[c]["name"] for c in promoted_teams(M, s)], "baseline RPS %.4f" % baseline(M, s))
    res = []
    grid = itertools.product([0.0, 0.5, 0.7, 1.0], [3, 6, 10], [120, 240], [0.5, 0.8])
    for xg_w, ridge, hl, psw in grid:
        kw = {"ridge": ridge, "half_life_days": hl, "prev_season_weight": psw}
        pr = walk_forward(M, "2025-26", xg_w, fit_kw=kw) + walk_forward(M, "2026-27", xg_w, fit_kw=kw)
        sc = score(pr)
        res.append({"xg_w": xg_w, **kw, **sc})
        print(res[-1], flush=True)
    res.sort(key=lambda d: d["rps"])
    json.dump(res, open(load.out("team_grid.json"), "w"), indent=1)
    print("BEST", res[:5])
