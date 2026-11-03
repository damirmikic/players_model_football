"""Load EPL data (FPL dataset, vaastav/Fantasy-Premier-League) into clean tables.

Stable team id = FPL team `code` (same across seasons). Player id = `code` from players_raw
(the per-season `element` id changes every season)."""
import csv, json, os, datetime as dt
from collections import defaultdict

RAW = os.path.join(os.path.dirname(__file__), "..", "raw")
SEASONS = ["2024-25", "2025-26", "2026-27"]
CURRENT = SEASONS[-1]


def _rows(season, name):
    with open(os.path.join(RAW, season, name), encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fnum(x, d=0.0):
    try:
        return float(x)
    except (TypeError, ValueError):
        return d


def load_season(season):
    teams = {int(r["id"]): {"code": int(r["code"]), "name": r["name"], "short": r.get("short_name", r["name"][:3])}
             for r in _rows(season, "teams.csv")}
    praw = _rows(season, "players_raw.csv")
    pl = {}
    for r in praw:
        pl[int(r["id"])] = {"code": int(r["code"]), "name": (r["first_name"] + " " + r["second_name"]).strip(),
                            "web": r["web_name"], "team": teams[int(r["team"])]["code"],
                            "pos": {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}.get(int(r["element_type"]), "?"),
                            "status": r.get("status", "a"), "news": r.get("news", ""),
                            "chance": r.get("chance_of_playing_next_round", "")}
    fixtures = []
    for r in _rows(season, "fixtures.csv"):
        fixtures.append({"season": season, "fid": int(r["id"]),
                         "round": int(fnum(r["event"], 0)) or None,
                         "date": r["kickoff_time"][:10] if r["kickoff_time"] else None,
                         "kickoff": r["kickoff_time"],
                         "home": teams[int(r["team_h"])]["code"], "away": teams[int(r["team_a"])]["code"],
                         "finished": r["finished"] == "True",
                         "hg": int(fnum(r["team_h_score"])) if r["finished"] == "True" else None,
                         "ag": int(fnum(r["team_a_score"])) if r["finished"] == "True" else None})
    # player-match rows; team code from players_raw (works for transfers within season too? use 'team' name)
    name2code = {t["name"]: t["code"] for t in teams.values()}
    pm = []
    for r in _rows(season, "merged_gw.csv"):
        el = int(r["element"])
        tcode = name2code.get(r.get("team"), pl.get(el, {}).get("team"))
        pm.append({"season": season, "fid": int(r["fixture"]), "pcode": pl[el]["code"] if el in pl else -el,
                   "team": tcode, "home": r["was_home"] == "True", "min": int(fnum(r["minutes"])),
                   "start": int(fnum(r.get("starts"))), "xg": fnum(r.get("expected_goals")),
                   "xa": fnum(r.get("expected_assists")), "xgc": fnum(r.get("expected_goals_conceded")),
                   "gc": int(fnum(r.get("goals_conceded"))), "g": int(fnum(r.get("goals_scored"))),
                   "pos": r.get("position")})
    return teams, pl, fixtures, pm


def build():
    allteams, players, matches, pmatch = {}, {}, [], []
    for s in SEASONS:
        teams, pl, fx, pm = load_season(s)
        for t in teams.values():
            allteams[t["code"]] = t
        for p in pl.values():
            players[p["code"]] = {**players.get(p["code"], {}), **p}   # latest season wins
        matches += fx
        pmatch += pm
    # team xG per fixture = sum of its players' xG (FPL xG excludes own goals; includes penalties)
    txg = defaultdict(float)
    for r in pmatch:
        txg[(r["season"], r["fid"], r["team"])] += r["xg"]
    for m in matches:
        m["hxg"] = round(txg.get((m["season"], m["fid"], m["home"]), 0.0), 3)
        m["axg"] = round(txg.get((m["season"], m["fid"], m["away"]), 0.0), 3)
    cur_teams = {t["code"] for t in load_season(CURRENT)[0].values()}
    patch_web_results(matches, allteams)
    return allteams, players, matches, pmatch, cur_teams


WEB_ALIAS = {"Leeds United": "Leeds", "Newcastle United": "Newcastle", "Brighton & Hove Albion": "Brighton",
             "Manchester United": "Man Utd", "Manchester City": "Man City", "AFC Bournemouth": "Bournemouth",
             "Tottenham Hotspur": "Spurs", "Nottingham Forest": "Nott'm Forest"}


def patch_web_results(matches, teams):
    """The FPL dataset lags; fill finished results from results_web_<season>.csv.
    Fails loudly if a web result contradicts a result already in the dataset."""
    path = os.path.join(RAW, f"results_web_{CURRENT}.csv")
    if not os.path.exists(path):
        return 0
    n2c = {t["name"]: c for c, t in teams.items()}
    idx = {(m["home"], m["away"]): m for m in matches if m["season"] == CURRENT}
    added = checked = 0
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            h, a = n2c[WEB_ALIAS.get(r["home"], r["home"])], n2c[WEB_ALIAS.get(r["away"], r["away"])]
            m = idx[(h, a)]
            hg, ag = int(r["hg"]), int(r["ag"])
            if m["finished"]:
                if (m["hg"], m["ag"]) != (hg, ag):
                    raise ValueError(f"web result mismatch {r}")
                checked += 1
            else:
                m.update(finished=True, hg=hg, ag=ag, hxg=None, axg=None, src="web")
                added += 1
    print(f"[patch] web results: {checked} confirmed, {added} added (no xG/player data for added)")
    return added


if __name__ == "__main__":
    import pricing
    T, P, M, PM, cur = build()
    for s in SEASONS:
        fm = [m for m in M if m["season"] == s and m["finished"]]
        g = sum(m["hg"] + m["ag"] for m in fm) / max(len(fm), 1)
        xg = sum(m["hxg"] + m["axg"] for m in fm if m["hxg"] is not None) / max(len(fm), 1)
        hw = sum(m["hg"] > m["ag"] for m in fm) / max(len(fm), 1)
        dr = sum(m["hg"] == m["ag"] for m in fm) / max(len(fm), 1)
        print(s, "played", len(fm), "goals/g %.2f xG/g %.2f home %.3f draw %.3f" % (g, xg, hw, dr),
              "rounds", max([m["round"] or 0 for m in fm] or [0]))
    st = pricing.standings([m for m in M if m["season"] == CURRENT and m["finished"]])
    for c in sorted(st, key=lambda c: st[c]["pos"]):
        s = st[c]
        print("%2d %-15s %2d %2d-%2d-%2d %2d:%2d %3d" % (s["pos"], T[c]["name"], s["p"], s["w"], s["d"], s["l"], s["gf"], s["ga"], s["pts"]))
    nxt = [m for m in M if m["season"] == CURRENT and not m["finished"]]
    print("unplayed", len(nxt), "next round", min(m["round"] or 99 for m in nxt))
