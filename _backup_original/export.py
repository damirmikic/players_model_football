"""Fit everything as of today and write the calculator's data file (../build/epl_data.json)."""
import json, datetime as dt, os
import numpy as np, pandas as pd
import load, team_model as tm, player_model as pm, pricing as P

CFG = json.load(open("../model_config.json"))
TODAY = dt.date(2026, 10, 30)


def main():
    T, PL, M, PM_, cur = load.build()
    # team ratings: all finished matches (web-patched results count as goals only)
    rows = tm.as_rows(M, load.CURRENT, CFG["xg_w"])
    promoted = tm.promoted_teams(M, load.CURRENT)
    prior = {t: P.new_team_prior("promoted", CFG["promo_shift"]) for t in promoted}
    r = P.fit_ratings(rows, asof=TODAY, current_teams=sorted(cur), prior=prior, goal_cap=99, **CFG["fit"])
    st = P.standings([m for m in M if m["season"] == load.CURRENT and m["finished"]])

    d = pm.prep(M, PM_)
    prior_pos = pm.pos_prior(d[d["season"] == load.SEASONS[0]])
    pr = pm.player_ratings(d, pd.Timestamp(TODAY), prior_pos)
    teams = {}
    for c in sorted(cur, key=lambda c: T[c]["name"]):
        squad = [p for p in PL.values() if p["team"] == c and p["status"] != "u"]
        rs = pm.recent_starts(d, pd.Timestamp(TODAY), [p["code"] for p in squad])
        avail = [(p["code"], p["pos"]) for p in squad
                 if not (p["status"] in ("i", "s") or (p["chance"] not in ("", "None") and float(p["chance"]) <= 25))]
        exi_all = pm.expected_xi([(p["code"], p["pos"]) for p in squad], rs)     # habitual XI (baseline)
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
        players.sort(key=lambda x: (-x["xi"], -x["rs"], -x["min"]))
        players = [p for p in players if p["xi"] or p["min"] > 0 or p["rs"] > 0][:26]
        s = st.get(c, {})
        teams[c] = {"name": T[c]["name"], "short": T[c]["short"], "att": round(r["att"][c], 4), "dfn": round(r["dfn"][c], 4),
                    "promoted": c in promoted, "pos": s.get("pos"), "pts": s.get("pts"), "p": s.get("p"),
                    "gf": s.get("gf"), "ga": s.get("ga"), "players": players}
    fixtures = [{"r": m["round"], "ko": m["kickoff"], "h": m["home"], "a": m["away"]}
                for m in M if m["season"] == load.CURRENT and not m["finished"] and m["round"] in (9, 10)]
    fixtures.sort(key=lambda f: f["ko"])
    bt = json.load(open("../out_backtest.json"))
    out = {"asof": str(TODAY), "league": "Premier League 2026/27", "mu": round(r["mu"], 5), "H": round(r["H"], 5),
           "rho": CFG["rho"], "beta_a": CFG["beta_a"], "beta_d": CFG["beta_d"], "teams": teams,
           "fixtures": fixtures, "backtest": bt, "config": CFG,
           "data_note": "Rezultati do 8. kola (25.10.2026). xG i postave igrača do 6. kola (12.10.2026); "
                        "status povreda iz FPL snimka posle 6. kola."}
    os.makedirs("../build", exist_ok=True)
    json.dump(out, open("../build/epl_data.json", "w"), ensure_ascii=False, separators=(",", ":"))
    print("mu %.3f H %.3f -> base home %.2f away %.2f" % (r["mu"], r["H"], np.exp(r["mu"] + r["H"]), np.exp(r["mu"])))
    for c in sorted(teams, key=lambda c: -(teams[c]["att"] - teams[c]["dfn"])):
        t = teams[c]
        top = sorted(t["players"], key=lambda p: -p["c"] * (p["n90"] > 3))[:3]
        print("%-15s att %+.3f dfn %+.3f  pos %2s  top: %s" % (t["name"], t["att"], t["dfn"], t["pos"],
              ", ".join("%s %.2f" % (p["n"], p["c"]) for p in top)))


if __name__ == "__main__":
    main()
