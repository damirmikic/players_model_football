"""Does soccer-rating.com's Elo (and its fixture calendar) add information to our team model?

    python fit_elo.py            # needs soccer_rating.py run first; pooled over all leagues

1. leakage check: an Elo-only Poisson model must NOT beat the closing market (else the shown ratings are post-match)
2. blend: e = Elo diff minus the part explained by our model (OLS of d on log(λh/λa), fitted on the train season);
   λh *= exp(b·e/400), λa *= exp(-b·e/400)
3. rest: λ *= exp(c · [≤3 days since the team's previous match in any competition])
Choose on 2025-26, score on 2026-27. Criterion: goal NLL + xG quasi-NLL; RPS and closing-market RPS reported."""
import io, json, math, os, contextlib, datetime as dt
from collections import Counter, defaultdict
import numpy as np
import load, leagues, pricing as P
from fit_betas import base_preds
from soccer_rating import CODES


def team_map(sr, M, code):
    """soccer-rating team id -> Understat id, voting on league matches with the same date (±1) and score."""
    by = defaultdict(list)
    for m in M:
        if m["finished"]:
            by[(m["date"], m["hg"], m["ag"])].append(m)
    votes = defaultdict(Counter)
    for r in sr:
        if r["comp"] != code or r["hg"] is None:
            continue
        d = dt.date.fromisoformat(r["date"])
        for off in (0, -1, 1):
            for m in by[(str(d + dt.timedelta(days=off)), r["hg"], r["ag"])]:
                votes[r["home_id"]][m["home"]] += 1
                votes[r["away_id"]][m["away"]] += 1
    return {k: c.most_common(1)[0][0] for k, c in votes.items() if c.most_common(1)[0][1] >= 3}


def rest_days(sr, idmap):
    """(understat team, date) -> days since that team's previous match in any competition."""
    games = defaultdict(set)
    for r in sr:
        for side in ("home_id", "away_id"):
            if r[side] in idmap:
                games[idmap[r[side]]].add(dt.date.fromisoformat(r["date"]))
    out = {}
    for t, ds in games.items():
        ds = sorted(ds)
        for a, b in zip(ds, ds[1:]):
            out[(t, str(b))] = (b - a).days
    return out


def rows_for(key):
    load.set_league(key)
    with contextlib.redirect_stdout(io.StringIO()):
        T, PL, M, PM, cur = load.build()
        bp = base_preds(M)
    path = os.path.join(load.RAW, key, "soccer_rating.json")
    sr = json.load(open(path, encoding="utf-8"))
    code = CODES[key][1]
    idmap = team_map(sr, M, code)
    rest = rest_days(sr, idmap)
    srm = {}
    for r in sr:
        if r["comp"] == code and r["home_id"] in idmap and r["away_id"] in idmap:
            srm[(idmap[r["home_id"]], idmap[r["away_id"]], r["date"])] = r
    mi = {(m["season"], m["home"], m["away"]): m for m in M}
    out, miss = [], 0
    for k, p in bp.items():
        m = mi[k]
        d0 = dt.date.fromisoformat(m["date"])
        r = next((srm.get((m["home"], m["away"], str(d0 + dt.timedelta(days=o)))) for o in (0, -1, 1)
                  if srm.get((m["home"], m["away"], str(d0 + dt.timedelta(days=o))))), None)
        if not r:
            miss += 1
            continue
        out.append({"league": key, "season": k[0], "lh": p["lh"], "la": p["la"], "hg": p["hg"], "ag": p["ag"], "o": p["o"],
                    "hxg": m["hxg"], "axg": m["axg"], "d": r["r_home"] - r["r_away"], "odds": m.get("odds_close"),
                    "rest_h": rest.get((m["home"], r["date"]), 7), "rest_a": rest.get((m["away"], r["date"]), 7),
                    "rho": load.config()["rho"]})
    print(f"  {key}: {len(idmap)} teams mapped, {len(out)} matches with Elo, {miss} without", flush=True)
    return out


def crit(rows, lam):
    gn = xn = rps = 0.0
    for r in rows:
        lh, la = lam(r)
        gn += lh - r["hg"] * math.log(lh) + la - r["ag"] * math.log(la)
        xn += lh - r["hxg"] * math.log(lh) + la - r["axg"] * math.log(la)
        rps += P.rps(P.one_x_two(P.score_matrix(lh, la, r["rho"])), r["o"])
    n = len(rows)
    return (gn + xn) / n, rps / n


def mkt_rps(rows):
    v = [P.rps([1 / o / sum(1 / x for x in r["odds"]) for o in r["odds"]], r["o"]) for r in rows if r["odds"]]
    return float(np.mean(v)), len(v)


if __name__ == "__main__":
    rows = [r for k in CODES for r in rows_for(k)]
    tr = [r for r in rows if r["season"] == "2025-26"]
    te = [r for r in rows if r["season"] == "2026-27"]
    print(f"train {len(tr)}, test {len(te)}")

    # 1. Elo-only model (fit a, H, b by grid on train goals)
    def elo_lam(a, H, b):
        return lambda r: (math.exp(a + H + b * r["d"] / 400), math.exp(a - b * r["d"] / 400))
    best = min(((a, H, b) for a in np.arange(0.1, 0.45, 0.05) for H in np.arange(0.0, 0.35, 0.05)
                for b in np.arange(0.2, 1.6, 0.1)), key=lambda x: crit(tr, elo_lam(*x))[0])
    for name, rs in (("train 25/26", tr), ("test 26/27", te)):
        ce, re_ = crit(rs, elo_lam(*best)); cm, rm = crit(rs, lambda r: (r["lh"], r["la"])); mk, n = mkt_rps(rs)
        print(f"== {name}: RPS Elo-only {re_:.5f} | our model {rm:.5f} | closing market {mk:.5f} (n={n})")

    # 2. blend: residual Elo information
    X = np.array([math.log(r["lh"] / r["la"]) for r in tr]); Y = np.array([r["d"] for r in tr])
    gam, alp = np.polyfit(X, Y, 1)
    print(f"Elo diff ≈ {alp:.1f} + {gam:.1f}·log(λh/λa) on train (corr {np.corrcoef(X, Y)[0, 1]:.3f})")
    def blend(b, c=0.0):
        def f(r):
            e = r["d"] - (alp + gam * math.log(r["lh"] / r["la"]))
            sh, sa = (r["rest_h"] <= 3), (r["rest_a"] <= 3)
            return (r["lh"] * math.exp(b * e / 400 + c * sh), r["la"] * math.exp(-b * e / 400 + c * sa))
        return f
    grid_b = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]
    grid_c = [-0.1, -0.05, -0.025, 0.0, 0.025, 0.05]
    res = {(b, c): crit(tr, blend(b, c)) for b in grid_b for c in grid_c}
    bb = min(res, key=lambda k: res[k][0])
    print("train crit by b (c=0):", " ".join(f"{b}:{res[(b, 0.0)][0]:.4f}" for b in grid_b))
    print("train crit by c (b=0):", " ".join(f"{c}:{res[(0.0, c)][0]:.4f}" for c in grid_c))
    print(f"chosen on 2025-26: b={bb[0]} c={bb[1]}")
    share = np.mean([(r["rest_h"] <= 3) + (r["rest_a"] <= 3) for r in rows]) / 2
    print(f"share of team-matches on ≤3 days rest: {share:.3f}")
    for name, rs in (("test 26/27", te), ("all", rows)):
        c0, r0 = crit(rs, blend(0, 0)); c1, r1 = crit(rs, blend(*bb)); cb, rb = crit(rs, blend(bb[0], 0)); cc, rc = crit(rs, blend(0, bb[1]))
        mk, n = mkt_rps(rs)
        print(f"== {name} n={len(rs)}: crit none {c0:.4f} | elo {cb:.4f} | rest {cc:.4f} | both {c1:.4f}  ||  "
              f"RPS none {r0:.5f} | elo {rb:.5f} | rest {rc:.5f} | both {r1:.5f} | market {mk:.5f}")
    json.dump({"alpha": alp, "gamma": gam, "b": bb[0], "c": bb[1]}, open(os.path.join(load.RAW, "..", "out", "elo_fit.json"), "w"), indent=1)
