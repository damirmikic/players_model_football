"""Player ratings and lineup adjustment.

Attack  c_i = shrunk per-90 (npxG + xA), pulled toward the position average with Kc 90s of pseudo-data.
Defence d_i = shrunk log(on-pitch npxGA rate / off-pitch npxGA rate) for the player's own team
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
    mt = pd.DataFrame([{k: m.get(k) for k in ("season", "fid", "round", "date", "hnpxg", "anpxg")} for m in M])
    d = d.merge(mt, on=["season", "fid"], how="left")
    d["date"] = pd.to_datetime(d["date"])
    d["team_xga"] = np.where(d["is_home"], d["anpxg"], d["hnpxg"]).astype(float)
    d["n90"] = d["min"] / 90.0
    d["ax"] = d["npxg"] + d["xa"]
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


def formation(d, asof, team, n=5):
    """Most common (DEF, MID, FWD) count of the team's starting XIs in its last n matches before asof."""
    h = d[(d["team"] == team) & (d["start"] == 1) & (d["date"] < asof)]
    fids = h.drop_duplicates("fid").sort_values("date")["fid"].values[-n:]
    shapes = [tuple(int((g["pos"] == p).sum()) for p in ("DEF", "MID", "FWD"))
              for _, g in h[h["fid"].isin(fids)].groupby("fid")]
    shapes = [s for s in shapes if sum(s) == 10]
    return max(set(shapes), key=shapes.count) if shapes else None


def expected_xi(squad, rs, shape=None):
    """squad: [(pcode, pos)]. 1 GK + 10 outfield by recent-start score, filling the team's usual shape
    (DEF, MID, FWD) first so a rotating striker pair doesn't leave the XI without a forward."""
    by = lambda pos: sorted([p for p, q in squad if q == pos], key=lambda p: -rs.get(p, 0))
    xi = by("GK")[:1]
    if shape:
        for pos, k in zip(("DEF", "MID", "FWD"), shape):
            xi += by(pos)[:k]
    rest = sorted([p for p, q in squad if q != "GK" and p not in xi], key=lambda p: -rs.get(p, 0))
    return xi + rest[:11 - len(xi)]


def team_squad(d, asof, team, season):
    """The team's current-season squad (previous season before its first match) with recent-start scores."""
    h = d[(d["team"] == team) & (d["date"] < asof)]
    cur = h[h["season"] == season]
    pool = cur if len(cur) else h[h["season"] == h["season"].max()] if len(h) else h
    squad = list(pool.drop_duplicates("pcode")[["pcode", "pos"]].itertuples(index=False, name=None))
    return squad, recent_starts(d, asof, [p for p, _ in squad])


def habitual_xi(d, asof, team, season):
    """Baseline XI as the calculator builds it: squad, recency-weighted starts, usual shape.
    Injured players are NOT removed — that is the point."""
    squad, rs = team_squad(d, asof, team, season)
    return expected_xi(squad, rs, formation(d, asof, team))


def available_xi(exi, squad, rs, out):
    """Habitual XI minus `out`, each vacancy filled by the best available player of the same position
    (else any outfielder) — the calculator's computeXI for a team with injured/suspended players."""
    pos = dict(squad)
    xi = [p for p in exi if p not in out]
    for v in [pos.get(p) for p in exi if p in out]:
        pool = sorted([p for p, q in squad if p not in out and p not in xi], key=lambda p: -rs.get(p, 0))
        pick = next((p for p in pool if pos[p] == v), None) or (v != "GK" and next((p for p in pool if pos[p] != "GK"), None))
        if pick:
            xi.append(pick)
    return xi


def lineup_factors(d, M, seasons=("2025-26", "2026-27"), unavail=None):
    """For each match with player data: attack ratios and defence delta for both teams.
    ratio = actual XI / habitual XI;  inj = available XI / habitual (known days before: injuries, suspensions);
    rot = actual XI / available XI (known ~1h before: the confirmed lineup). inj * rot == ratio when the
    available XI is well defined. `unavail`: lineups.history_unavailable() output; without it inj = 1."""
    unavail = unavail or {}
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
                    axi = list(sq.loc[sq["start"] == 1, "pcode"])
                    tsq, trs = team_squad(d, asof, t, s)
                    exi = expected_xi(tsq, trs, formation(d, asof, t)) or axi   # no history: neutral
                    out_ids = unavail.get((s, m["home"], m["away"]), {}).get(side, set())
                    vxi = available_xi(exi, tsq, trs, out_ids) if exi is not axi else axi
                    pos_of = dict(zip(d["pcode"], d["pos"]))
                    squad += [(p, pos_of[p]) for p in exi + vxi if p not in dict(squad)]
                    cget = lambda p: float(pr["c"].get(p, prior.get(dict(squad).get(p), 0.3)))
                    dget = lambda p: float(pr["d"].get(p, 0.0))
                    rec[side + "_ratio"] = sum(map(cget, axi)) / max(sum(map(cget, exi)), 1e-9)
                    rec[side + "_dd"] = float(np.mean(list(map(dget, axi))) - np.mean(list(map(dget, exi))))
                    rec[side + "_changes"] = len(set(exi) - set(axi))
                    sc = lambda xs: max(sum(map(cget, xs)), 1e-9)
                    rec[side + "_inj"] = sc(vxi) / sc(exi)
                    rec[side + "_rot"] = sc(axi) / sc(vxi)
                    rec[side + "_nout"] = len(set(exi) & out_ids)
                    rec[side + "_fm"] = (s, m["home"], m["away"]) in unavail
                out.append(rec)
    return out


if __name__ == "__main__":
    import json, team_model as tm, pricing as P
    load.league_from_argv()
    T, PL, M, PM, cur = load.build()
    import lineups
    d = prep(M, PM)
    UA, ust = lineups.history_unavailable(M, PM)
    print("[fotmob history]", len(UA), "matches", ust)
    LF = lineup_factors(d, M, unavail=UA)
    json.dump(LF, open(load.out("lineup_factors.json"), "w"))
    lf = pd.DataFrame(LF)
    print(lf[["h_ratio", "h_inj", "h_rot", "h_nout", "h_changes"]].describe())
