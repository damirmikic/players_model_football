"""Player ratings and lineup adjustment.

Attack  c_i = shrunk per-90 (xG + xA), pulled toward the position average with Kc 90s of pseudo-data.
Defence d_i = shrunk log(on-pitch xGA rate / off-pitch xGA rate) for the player's own team
              (<0 = team concedes less with him on the pitch). Players who never sit out get d ~ 0.
Lineup adjustment for a match:
    lam_team *= (sum_XI c / sum_expXI c) ** beta_a          (own attack)
    lam_team *= exp(beta_d * (mean_XI_opp d - mean_expXI_opp d))   (opponent's defence)
Expected XI = 1 GK + 10 outfield with most recency-weighted starts in their last 5 matches.
beta_a, beta_d are fitted walk-forward on 2025-26 + 2026-27 (no data after the match date)."""
import math, datetime as dt
import numpy as np, pandas as pd
import load

HALF_LIFE = 365.0
KC = 6.0          # attack shrinkage, in 90s
KD = 10.0         # defence shrinkage, in 90s (on and off each)


def prep(M, PM):
    d = pd.DataFrame(PM).rename(columns={"home": "is_home"})
    mt = pd.DataFrame([{k: m[k] for k in ("season", "fid", "round", "date", "hxg", "axg")} for m in M])
    d = d.merge(mt, on=["season", "fid"], how="left")
    d["date"] = pd.to_datetime(d["date"])
    d["team_xga"] = np.where(d["is_home"], d["axg"], d["hxg"]).astype(float)
    d["n90"] = d["min"] / 90.0
    d["ax"] = d["xg"] + d["xa"]
    d["off_xga"] = d["team_xga"] * (90 - d["min"]).clip(lower=0) / 90.0
    d["off90"] = (90 - d["min"]).clip(lower=0) / 90.0
    return d


def pos_prior(d):
    s = d[d["min"] > 0].groupby("pos").agg(ax=("ax", "sum"), n90=("n90", "sum"))
    return (s["ax"] / s["n90"]).to_dict()


def player_ratings(d, asof, prior):
    """Ratings from all rows strictly before `asof` (Timestamp), time-decayed."""
    h = d[d["date"] < asof]
    w = 0.5 ** ((asof - h["date"]).dt.days / HALF_LIFE)
    g = pd.DataFrame({"pcode": h["pcode"], "pos": h["pos"], "team": h["team"],
                      "w90": w * h["n90"], "wax": w * h["ax"], "won": w * h["xgc"], "woff": w * h["off_xga"],
                      "woff90": w * h["off90"], "wplayed": w * (h["min"] > 0)}).groupby("pcode")
    a = g.agg(pos=("pos", "last"), w90=("w90", "sum"), wax=("wax", "sum"), won=("won", "sum"),
              woff=("woff", "sum"), woff90=("woff90", "sum"))
    pr = a["pos"].map(prior).fillna(np.mean(list(prior.values())))
    a["c"] = (a["wax"] + KC * pr) / (a["w90"] + KC)
    rt = (a["won"] + a["woff"]) / (a["w90"] + a["woff90"]).clip(lower=1e-9)
    on = (a["won"] + KD * rt) / (a["w90"] + KD)
    off = (a["woff"] + KD * rt) / (a["woff90"] + KD)
    a["d"] = np.log(on.clip(lower=1e-6) / off.clip(lower=1e-6))
    return a


def recent_starts(d, asof, pcodes, n=5):
    """Recency-weighted starts over each player's last n match rows before asof."""
    h = d[(d["date"] < asof) & (d["pcode"].isin(pcodes))].sort_values("date")
    out = {}
    for p, grp in h.groupby("pcode"):
        st = grp["start"].values[-n:][::-1]
        out[p] = float(sum(s * (0.85 ** i) for i, s in enumerate(st))) + 1e-3 * float(grp["min"].values[-n:].sum())
    return out


def expected_xi(squad, rs):
    """squad: [(pcode, pos)]. 1 GK + 10 outfield by recent-start score."""
    gks = sorted([p for p, pos in squad if pos == "GK"], key=lambda p: -rs.get(p, 0))
    of = sorted([p for p, pos in squad if pos != "GK"], key=lambda p: -rs.get(p, 0))
    return gks[:1] + of[:10]


def lineup_factors(d, M, seasons=("2025-26", "2026-27")):
    """For each match with player data: attack ratio and defence delta for both teams."""
    prior = pos_prior(d[d["season"] == load.SEASONS[0]])
    out = []
    for s in seasons:
        ms = [m for m in M if m["season"] == s and m["finished"] and m.get("hxg") is not None]
        for k in sorted({m["round"] for m in ms}):
            rm = [m for m in ms if m["round"] == k]
            asof = pd.Timestamp(min(m["date"] for m in rm))
            pr = player_ratings(d, asof, prior)
            for m in rm:
                rows = d[(d["season"] == s) & (d["fid"] == m["fid"])]
                rec = {"season": s, "round": k, "home": m["home"], "away": m["away"]}
                for side, t in (("h", m["home"]), ("a", m["away"])):
                    sq = rows[rows["team"] == t]
                    squad = list(zip(sq["pcode"], sq["pos"]))
                    rs = recent_starts(d, asof, [p for p, _ in squad])
                    exi = expected_xi(squad, rs)
                    axi = list(sq.loc[sq["start"] == 1, "pcode"])
                    cget = lambda p: float(pr["c"].get(p, prior.get(dict(squad).get(p), 0.3)))
                    dget = lambda p: float(pr["d"].get(p, 0.0))
                    rec[side + "_ratio"] = sum(map(cget, axi)) / max(sum(map(cget, exi)), 1e-9)
                    rec[side + "_dd"] = float(np.mean(list(map(dget, axi))) - np.mean(list(map(dget, exi))))
                    rec[side + "_changes"] = len(set(exi) - set(axi))
                out.append(rec)
    return out


if __name__ == "__main__":
    import json, team_model as tm, pricing as P
    T, PL, M, PM, cur = load.build()
    d = prep(M, PM)
    LF = lineup_factors(d, M)
    json.dump(LF, open("../out_lineup_factors.json", "w"))
    lf = pd.DataFrame(LF)
    print(lf[["h_ratio", "a_ratio", "h_dd", "a_dd", "h_changes"]].describe())
