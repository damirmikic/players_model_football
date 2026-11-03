"""Load one league (Understat + football-data.co.uk odds + FPL status for EPL) into clean tables.

Same contract as the old FPL-based loader: build() -> (teams, players, matches, player_match_rows, current_teams).
Stable team id = Understat team id, player id = Understat player id (both stable across seasons).
Select the league with set_league(key) or the LEAGUE env var (default EPL)."""
import csv, json, os, re, unicodedata, datetime as dt
from collections import defaultdict, Counter
import leagues

RAW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "raw")
SEASONS = leagues.SEASONS
CURRENT = leagues.CURRENT
LEAGUE = os.environ.get("LEAGUE", "EPL")
# football-data.co.uk match statistics: key -> column suffix (H<col> home, A<col> away)
STAT_COLS = {"shots": "S", "sot": "ST", "fouls": "F", "corners": "C", "yellow": "Y"}


def config():
    """model_config.json defaults, overridden by its "leagues": {<key>: {...}} entry for the active league."""
    c = json.load(open(os.path.join(RAW, "..", "model_config.json"), encoding="utf-8"))
    over = c.pop("leagues", {}).get(LEAGUE, {})
    for k, v in over.items():
        c[k] = {**c[k], **v} if isinstance(v, dict) and isinstance(c.get(k), dict) else v
    c["league"] = LEAGUE
    return c


def out(name):
    """Per-league output path for model artefacts (../out/<league>/<name>)."""
    d = os.path.join(RAW, "..", "out", LEAGUE)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, name)


def league_from_argv():
    import sys
    if len(sys.argv) > 1:
        set_league(sys.argv[1])
    return LEAGUE


def set_league(key):
    global LEAGUE
    if key not in leagues.LEAGUES:
        raise KeyError(f"unknown league {key}; known: {', '.join(leagues.LEAGUES)}")
    LEAGUE = key


def _p(season, *parts):
    return os.path.join(RAW, LEAGUE, season, *parts)


def fnum(x, d=0.0):
    try:
        return float(x)
    except (TypeError, ValueError):
        return d


def pos_group(p):
    """Understat roster position (GK, DR, DMC, AML, FW, Sub, ...) -> GK/DEF/MID/FWD (None for Sub)."""
    if p in ("Sub", "", None):
        return None
    if p == "GK":
        return "GK"
    if p == "FW":
        return "FWD"
    if p in ("DR", "DC", "DL"):
        return "DEF"
    return "MID"


def _norm(s):
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z ]", " ", s).split()


def short_name(full):
    t = full.split()
    return full if len(t) == 1 else " ".join(t[1:]) if len(t) == 2 else t[-1]


def on_pitch(r, end):
    """(start, stop) minute interval a roster row was on the pitch."""
    t = int(fnum(r["time"]))
    if t <= 0:
        return None
    if r["position"] == "Sub":
        return (max(end - t, 0), end)
    # replaced (roster_in set) or sent off (short time, nobody in): off at t
    return (0, t if (r["roster_in"] != "0" or t < end - 5) else end)


def load_season(season):
    lg = json.load(open(_p(season, "understat_league.json"), encoding="utf-8"))
    teams = {int(t["id"]): {"code": int(t["id"]), "name": t["title"]} for t in lg["teams"].values()}
    fixtures, pm = [], []
    ngame = Counter()
    for m in sorted(lg["dates"], key=lambda m: (m["datetime"], int(m["id"]))):
        h, a = int(m["h"]["id"]), int(m["a"]["id"])
        for side, t in (("h", h), ("a", a)):
            teams.setdefault(t, {"code": t, "name": m[side]["title"]})
            teams[t]["short"] = m[side]["short_title"]
        ngame[h] += 1; ngame[a] += 1
        fin = bool(m["isResult"])
        fixtures.append({"season": season, "fid": int(m["id"]), "round": max(ngame[h], ngame[a]),
                         "date": m["datetime"][:10], "kickoff": m["datetime"].replace(" ", "T") + "Z",
                         "home": h, "away": a, "finished": fin,
                         "hg": int(m["goals"]["h"]) if fin else None, "ag": int(m["goals"]["a"]) if fin else None,
                         "hxg": round(fnum(m["xG"]["h"]), 3) if fin else None,
                         "axg": round(fnum(m["xG"]["a"]), 3) if fin else None})
    for t in teams.values():
        t.setdefault("short", t["name"][:3].upper())
    for f in fixtures:
        if not f["finished"]:
            continue
        path = _p(season, "matches", f"{f['fid']}.json")
        if not os.path.exists(path):
            f["no_players"] = True
            continue
        md = json.load(open(path, encoding="utf-8"))
        if not all(isinstance(md["rosters"][s], dict) and md["rosters"][s] for s in ("h", "a")):
            f["no_players"] = True        # Understat has the result but no roster for a few matches
            continue
        # non-penalty shots drive player ratings: penalties are rare, high-xG and go to one taker
        allshots = {s: md["shots"][s] or [] for s in ("h", "a")}
        shots = {s: [(int(fnum(x["minute"])), fnum(x["xG"]), x["result"]) for x in allshots[s]
                     if x["situation"] != "Penalty"] for s in ("h", "a")}
        pen_xg = Counter()
        for s in ("h", "a"):
            for x in allshots[s]:
                if x["situation"] == "Penalty":
                    pen_xg[int(x["player_id"])] += fnum(x["xG"])
        f["hnpxg"], f["anpxg"] = (round(sum(x for _, x, _ in shots[s]), 3) for s in ("h", "a"))
        end = max([90] + [int(fnum(x["minute"])) for s in allshots.values() for x in s])
        for side, opp in (("h", "a"), ("a", "h")):
            tcode = f["home"] if side == "h" else f["away"]
            for r in md["rosters"][side].values():
                iv = on_pitch(r, end)
                xgc = gc = 0.0
                if iv:
                    for mi, x, res in shots[opp]:      # non-penalty xG conceded while on the pitch
                        if iv[0] <= mi <= iv[1]:
                            xgc += x; gc += res == "Goal"
                pid = int(r["player_id"])
                pm.append({"season": season, "fid": f["fid"], "pcode": pid, "name": r["player"],
                           "team": tcode, "home": side == "h", "min": int(fnum(r["time"])),
                           "start": int(r["position"] != "Sub"), "xg": fnum(r["xG"]), "npxg": max(fnum(r["xG"]) - pen_xg[pid], 0.0), "xa": fnum(r["xA"]),
                           "xgc": round(xgc, 4), "gc": int(gc), "g": int(fnum(r["goals"])),
                           "rpos": pos_group(r["position"])})
    return teams, fixtures, pm


def attach_odds(matches, season):
    """Closing 1X2 odds from football-data.co.uk. Team names are mapped by voting on (date, score) pairs,
    so no alias table is needed. Prefers Betfair exchange closing, then market average closing; also stores the
    pre-match odds (football-data collects them on Friday / Tuesday before the round) as odds_open.
    Also attaches match statistics m["stats"] = {stat: (home, away)} (STAT_COLS) and m["referee"] (England only)."""
    path = _p(season, "fd_odds.csv")
    if not os.path.exists(path):
        return 0
    rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))
    fin = [m for m in matches if m["season"] == season and m["finished"]]
    by_key = defaultdict(list)
    for m in fin:
        by_key[(m["date"], m["hg"], m["ag"])].append(m)
    votes = defaultdict(Counter)
    parsed = []
    for r in rows:
        if not r.get("HomeTeam") or not r.get("FTHG"):
            continue
        try:
            d = dt.datetime.strptime(r["Date"], "%d/%m/%Y" if len(r["Date"]) == 10 else "%d/%m/%y").date()
        except ValueError:
            continue
        hg, ag = int(r["FTHG"]), int(r["FTAG"])
        parsed.append((d, r))
        for off in (0, -1, 1):
            for m in by_key[(str(d + dt.timedelta(days=off)), hg, ag)]:
                votes[r["HomeTeam"]][m["home"]] += 1; votes[r["AwayTeam"]][m["away"]] += 1
    name2id = {n: c.most_common(1)[0][0] for n, c in votes.items()}
    idx = {(m["home"], m["away"]): m for m in matches if m["season"] == season}
    n = 0
    for d, r in parsed:
        m = idx.get((name2id.get(r["HomeTeam"]), name2id.get(r["AwayTeam"])))
        if not m:
            continue
        st = {}
        for key, col in STAT_COLS.items():                          # match statistics (stat_model.py)
            h, a = r.get("H" + col, ""), r.get("A" + col, "")
            if h.strip() != "" and a.strip() != "":
                st[key] = (int(fnum(h)), int(fnum(a)))
        if st:
            m["stats"] = st
        if r.get("Referee"):
            m["referee"] = r["Referee"].strip()
        for pre in ("BFEC", "AvgC", "PSC", "B365C"):
            o = [fnum(r.get(pre + x)) for x in "HDA"]
            if all(v > 1 for v in o):
                m["odds_close"], m["odds_src"] = o, pre
                ou = [fnum(r.get(pre + x + "2.5")) for x in "><"]
                if all(v > 1 for v in ou):
                    m["ou_close"] = ou                                # closing over / under 2.5, same source
                n += 1
                break
        for pre in ("BFE", "Avg", "PS", "B365"):                    # pre-match (collected days before kick-off)
            o = [fnum(r.get(pre + x)) for x in "HDA"]
            if all(v > 1 for v in o):
                m["odds_open"], m["odds_open_src"] = o, pre
                break
    return n


def market_lambdas(matches):
    """(λh, λa) reproducing each finished match's fair closing 1X2 + O/U 2.5 (m["mkt_lam"]), used as a rating
    target for later matches (team_model.as_rows, mkt_w). Cached in ../out/<league>/mkt_lambda.json."""
    import pricing as P
    path = out("mkt_lambda.json")
    cache = json.load(open(path)) if os.path.exists(path) else {}
    n0 = len(cache)
    fair = lambda o: [1 / x / sum(1 / y for y in o) for x in o]
    for m in matches:
        if not (m["finished"] and m.get("odds_close") and m.get("ou_close")):
            continue
        k = f'{m["season"]}|{m["home"]}|{m["away"]}'
        if k not in cache:
            ph, _, pa = fair(m["odds_close"])
            cache[k] = P.fit_lambdas(ph, pa, fair(m["ou_close"])[0], rho=config()["rho"], iters=8)
        m["mkt_lam"] = cache[k]
    if len(cache) > n0:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        json.dump(cache, open(path, "w"))
    return sum(1 for m in matches if m.get("mkt_lam"))


def fpl_status(teams, players):
    """EPL only: injury/suspension status from the FPL API, matched by team + name."""
    path = _p(CURRENT, "fpl_bootstrap.json")
    if not leagues.LEAGUES[LEAGUE]["fpl"] or not os.path.exists(path):
        return 0
    b = json.load(open(path, encoding="utf-8"))
    ft = {t["id"]: t for t in b["teams"]}
    sh2us = {t["short"]: c for c, t in teams.items()}
    name2us = {" ".join(_norm(t["name"])): c for c, t in teams.items()}
    team_map = {}
    for i, t in ft.items():
        team_map[i] = sh2us.get(t["short_name"]) or name2us.get(" ".join(_norm(t["name"])))
    by_team = defaultdict(list)
    for p in players.values():
        by_team[p["team"]].append(p)
    n = 0
    for e in b["elements"]:
        cand = by_team.get(team_map.get(e["team"]), [])
        full = _norm(e["first_name"] + " " + e["second_name"]); web = _norm(e["web_name"])
        best = None
        for p in cand:
            un = _norm(p["name"])
            if un == full or un == web or (len(un) > 1 and un[-1] in full and un[0] in full) \
                    or (len(un) == 1 and un[0] in web):
                best = p
                break
        if not best:
            continue
        ch = e.get("chance_of_playing_next_round")
        best.update(status=e["status"], news=e.get("news") or "", chance="" if ch is None else str(ch),
                    web=e["web_name"], pos={1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}[e["element_type"]])
        n += 1
    return n


def build():
    allteams, matches, pmatch = {}, [], []
    for s in SEASONS:
        teams, fx, pm = load_season(s)
        allteams.update(teams)
        matches += fx
        pmatch += pm
    # player table: most common on-pitch position; team = team of the latest appearance
    pos_c, last = defaultdict(Counter), {}
    for r in sorted(pmatch, key=lambda r: (r["season"], r["fid"])):
        if r["rpos"]:
            pos_c[r["pcode"]][r["rpos"]] += 1
        last[r["pcode"]] = r
    players = {}
    for code, r in last.items():
        pos = pos_c[code].most_common(1)[0][0] if pos_c[code] else "MID"
        players[code] = {"code": code, "name": r["name"], "web": short_name(r["name"]), "team": r["team"], "pos": pos,
                         # left the league / no current-season appearance -> 'u' (unavailable, dropped from squads)
                         "status": "a" if r["season"] == CURRENT else "u", "news": "", "chance": ""}
    nf = fpl_status(allteams, players)
    for r in pmatch:
        r["pos"] = players[r["pcode"]]["pos"]
    no = {s: attach_odds(matches, s) for s in SEASONS}
    nl = market_lambdas(matches)
    cur_teams = {m[k] for m in matches if m["season"] == CURRENT for k in ("home", "away")}
    miss = sum(1 for m in matches if m.get("no_players"))
    print(f"[load {LEAGUE}] {sum(m['finished'] for m in matches)} finished matches, {len(pmatch)} player rows, "
          f"odds {no}, market λ {nl}, FPL status matched {nf}" + (f", {miss} matches without roster file" if miss else ""))
    return allteams, players, matches, pmatch, cur_teams


if __name__ == "__main__":
    import sys, pricing
    if len(sys.argv) > 1:
        set_league(sys.argv[1])
    T, P, M, PM, cur = build()
    for s in SEASONS:
        fm = [m for m in M if m["season"] == s and m["finished"]]
        g = sum(m["hg"] + m["ag"] for m in fm) / max(len(fm), 1)
        xg = sum(m["hxg"] + m["axg"] for m in fm) / max(len(fm), 1)
        hw = sum(m["hg"] > m["ag"] for m in fm) / max(len(fm), 1)
        dr = sum(m["hg"] == m["ag"] for m in fm) / max(len(fm), 1)
        print(s, "played", len(fm), "goals/g %.2f xG/g %.2f home %.3f draw %.3f" % (g, xg, hw, dr))
    st = pricing.standings([m for m in M if m["season"] == CURRENT and m["finished"]])
    for c in sorted(st, key=lambda c: st[c]["pos"]):
        s = st[c]
        print("%2d %-24s %2d %2d-%2d-%2d %2d:%2d %3d" % (s["pos"], T[c]["name"], s["p"], s["w"], s["d"], s["l"], s["gf"], s["ga"], s["pts"]))
