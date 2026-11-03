"""Fit everything as of today and write the calculator's data file (../build/data_<league>.json).

    python export.py [LEAGUE]        # TODAY=YYYY-MM-DD env var overrides the as-of date"""
from collections import Counter
import json, datetime as dt, os
import numpy as np, pandas as pd
import load, leagues, lineups, team_model as tm, player_model as pm, stat_model as sm, pricing as P

TODAY = dt.date.fromisoformat(os.environ["TODAY"]) if os.environ.get("TODAY") else dt.date.today()
HORIZON_DAYS = 10     # fixtures shown in the calculator


def main():
    load.league_from_argv()
    T, PL, M, PM_, cur = load.build()
    for m in M:      # as-of in the past: later results are unknown
        if m["finished"] and m["date"] >= str(TODAY):
            m.update(finished=False, hg=None, ag=None, hxg=None, axg=None)
    # team ratings: all finished matches before TODAY
    rows = tm.as_rows(M, load.CURRENT, load.config()["xg_w"], load.config().get("mkt_w", 0.0))
    promoted = tm.promoted_teams(M, load.CURRENT)
    prior = {t: P.new_team_prior("promoted", load.config()["promo_shift"]) for t in promoted}
    played = Counter(m[k] for m in M if m["season"] == load.CURRENT and m["finished"] for k in ("home", "away"))
    next_round = max(played.values(), default=0) + 1
    r = P.fit_ratings(rows, asof=TODAY, current_teams=sorted(cur), prior=prior, goal_cap=99,
                      **tm.fit_kw_for(load.config(), next_round))
    st = P.standings([m for m in M if m["season"] == load.CURRENT and m["finished"]])

    # lineup source (FotMob): unavailable players -> status, confirmed/predicted XIs per fixture
    lu_fx, lu_status, lu_meta = lineups.attach(T, PL, M)
    for pid, (s, news) in lu_status.items():
        p = PL.get(pid)
        rank = {"a": 0, "u": 0, "d": 1, "i": 2, "s": 2}
        if p and rank[s] > rank.get(p["status"], 0):     # only upgrades: never weakens an official FPL flag
            p.update(status=s, news=(p["news"] + " · " if p["news"] else "") + news)
    lu_ids = {pid for r in lu_fx.values() for side in ("h", "a") for pid in r[side]["xi"]}

    d = pm.prep(M, PM_)
    prior_pos = pm.pos_prior(d[d["season"] == load.SEASONS[0]])
    pr = pm.player_ratings(d, pd.Timestamp(TODAY), prior_pos)
    teams = {}
    for c in sorted(cur, key=lambda c: T[c]["name"]):
        # current squad; a lineup starter is kept even without a current-season appearance (e.g. back from injury)
        squad = [p for p in PL.values() if p["team"] == c and (p["status"] != "u" or p["code"] in lu_ids)]
        rs = pm.recent_starts(d, pd.Timestamp(TODAY), [p["code"] for p in squad])
        avail = [(p["code"], p["pos"]) for p in squad
                 if not (p["status"] in ("i", "s") or (p["chance"] not in ("", "None") and float(p["chance"]) <= 25))]
        exi_all = pm.expected_xi([(p["code"], p["pos"]) for p in squad], rs,      # habitual XI (baseline)
                                 pm.formation(d, pd.Timestamp(TODAY), c))
        cur_rows = d[(d["season"] == load.CURRENT) & (d["team"] == c)]
        min_season = cur_rows.groupby("pcode")["min"].sum().to_dict()
        players = []
        for p in squad:
            code = p["code"]
            players.append({"id": code, "n": p["web"], "pos": p["pos"],
                            "c": round(float(pr["c"].get(code, prior_pos.get(p["pos"], 0.2))), 4),
                            "d": round(float(pr["d"].get(code, 0.0)), 4),
                            "n90": round(float(pr["w90"].get(code, 0.0)), 1),
                            "rs": round(rs.get(code, 0.0), 3), "min": int(min_season.get(code, 0)),
                            "st": p["status"], "news": p["news"], "xi": code in exi_all})
        # keep the relevant part of the squad: XI + anyone with minutes / recent starts, max 26
        players.sort(key=lambda x: (-x["xi"], -(x["id"] in lu_ids), -x["rs"], -x["min"]))
        players = [p for p in players if p["xi"] or p["min"] > 0 or p["rs"] > 0 or p["id"] in lu_ids][:26]
        for p in players:
            if p["st"] == "u":
                p["st"] = "a"
        s = st.get(c, {})
        teams[c] = {"name": T[c]["name"], "short": T[c]["short"], "att": round(r["att"][c], 4), "dfn": round(r["dfn"][c], 4),
                    "promoted": c in promoted, "pos": s.get("pos"), "pts": s.get("pts"), "p": s.get("p"),
                    "gf": s.get("gf"), "ga": s.get("ga"), "players": players}
    end = str(TODAY + dt.timedelta(days=HORIZON_DAYS))
    # kicked off more than 2h ago = played (Understat marks results with a delay)
    cutoff = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M") if not os.environ.get("TODAY") else ""
    fixtures = [{"r": m["round"], "ko": m["kickoff"], "h": m["home"], "a": m["away"]}
                for m in M if m["season"] == load.CURRENT and not m["finished"] and str(TODAY) <= m["date"] <= end
                and m["kickoff"][:16] >= cutoff]
    for f in fixtures:
        lu = lu_fx.get((f["h"], f["a"]))
        if lu and lu["type"] in ("confirmed", "predicted"):
            f["lu"] = lu
    fixtures.sort(key=lambda f: f["ko"])
    bt = json.load(open(load.out("backtest.json")))
    # centring of lineup factors (see fit_lineup.py): geometric mean over the backtest matches of this league
    LF = json.load(open(load.out("lineup_factors.json")))
    gm = lambda xs: float(np.exp(np.mean(np.log(xs)))) if xs else 1.0
    inj_c = gm([f[s + "_inj"] for f in LF if f.get("h_fm") for s in "ha"])
    rot_c = gm([f[s + "_rot"] for f in LF if f.get("h_fm") for s in "ha"] or [f[s + "_ratio"] for f in LF for s in "ha"])
    lc = leagues.LEAGUES[load.LEAGUE]
    cs = [m for m in M if m["season"] == load.CURRENT and m["finished"]]
    last = max(m["date"] for m in cs) if cs else None
    no_pl = sum(1 for m in cs if m.get("no_players"))
    fmt = lambda d: ".".join(reversed(d.split("-")))
    note = (f"Rezultati, xG i postave (Understat) do {fmt(last)}, {len(cs)} odigranih mečeva"
            + (f" ({no_pl} bez postava)" if no_pl else "") + ". "
            + ("Povrede: FPL API" + (" + FotMob" if lu_meta else "") + ". " if lc["fpl"] else
               "Povrede i postave: FotMob. " if lu_meta else "Bez izvora za povrede – igrače van stroja isključi ručno. ")
            + (f"Postave osvežene {lu_meta['fetched_at'][8:10]}.{lu_meta['fetched_at'][5:7]}. u {lu_meta['fetched_at'][11:16]} UTC." if lu_meta else ""))
    yr = load.CURRENT.split("-")
    cfg = load.config()
    stats_out = {}
    for s in sm.STATS:
        sc = sm.stat_cfg(cfg, s)
        sr = sm.fit(M, s, TODAY, sorted(cur), sc, variant="full" if sc.get("covariates", True) else "teams")
        stats_out[s] = {
            "mu": round(sr["mu"], 5), "H": round(sr["H"], 5),
            "beta": [round(x, 5) for x in sr["beta"]],
            "k": sc["k"], "c": sc["c"],
            "att": {c: round(v, 4) for c, v in sr["att"].items()},
            "dfn": {c: round(v, 4) for c, v in sr["dfn"].items()},
        }
    out = {"asof": str(TODAY), "league": f"{lc['name']} {yr[0]}/{yr[1]}", "league_key": load.LEAGUE,
           "leagues": [{"key": k, "name": v["name"]} for k, v in leagues.LEAGUES.items()], "mu": round(r["mu"], 5), "H": round(r["H"], 5),
           "rho": cfg["rho"], "beta_a": cfg["beta_a"], "beta_inj": cfg.get("beta_inj", cfg["beta_a"]),
           "beta_rot": cfg.get("beta_rot", cfg["beta_a"]),
           "inj_center": round(inj_c, 4), "rot_center": round(rot_c, 4), "beta_d": cfg["beta_d"], "teams": teams,
           "stats": stats_out,
           "fixtures": fixtures, "backtest": bt, "config": cfg,
           "data_note": note}
    os.makedirs("../build", exist_ok=True)
    json.dump(out, open(f"../build/data_{load.LEAGUE}.json", "w"), ensure_ascii=False, separators=(",", ":"))
    print("mu %.3f H %.3f -> base home %.2f away %.2f" % (r["mu"], r["H"], np.exp(r["mu"] + r["H"]), np.exp(r["mu"])))
    for c in sorted(teams, key=lambda c: -(teams[c]["att"] - teams[c]["dfn"])):
        t = teams[c]
        top = sorted(t["players"], key=lambda p: -p["c"] * (p["n90"] > 3))[:3]
        print("%-24s att %+.3f dfn %+.3f  pos %2s  top: %s" % (t["name"], t["att"], t["dfn"], t["pos"],
              ", ".join("%s %.2f" % (p["n"], p["c"]) for p in top)))


if __name__ == "__main__":
    main()
