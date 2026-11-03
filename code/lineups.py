"""Lineup source: predicted / confirmed XIs and unavailable players for upcoming matches.

    python lineups.py                 # all leagues
    python lineups.py EPL Serie_A     # selected leagues

Writes ../raw/<league>/<season>/lineups.json in a source-neutral format (see `normalize`), so another
provider (e.g. API-Football) only needs its own fetch function producing the same shape:

    {"source", "fetched_at", "season_matches": [{"id", "kickoff", "home", "away"}],   # for team mapping
     "matches": [{"id", "kickoff", "home": {"id", "name"}, "away": {...},
                  "type": "confirmed" | "predicted" | "none",
                  "h": {"starters": [name, ...], "unavailable": [{"name", "type", "ret"}]}, "a": {...}}]}

Current source: FotMob's web-app JSON (api/data/leagues, api/data/matchDetails). Unofficial, not licensed for
commercial use — fine for prototyping; swap for a licensed feed in production.
`attach()` maps everything onto Understat team/player ids for export.py."""
import datetime as dt, difflib, json, os, sys, time, urllib.request
from collections import Counter, defaultdict
import leagues, load

UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36"
HORIZON_DAYS = 10
DELAY = 1.0


def _get(path):
    req = urllib.request.Request("https://www.fotmob.com/api/data/" + path,
                                 headers={"User-Agent": UA, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def _side(team):
    return {"starters": [p["name"] for p in team.get("starters") or []],
            "unavailable": [{"name": p["name"], "type": (p.get("unavailability") or {}).get("type", "unknown"),
                             "ret": (p.get("unavailability") or {}).get("expectedReturn", "")}
                            for p in team.get("unavailable") or []]}


def fetch_fotmob(key, now=None):
    now = now or dt.datetime.now(dt.timezone.utc)
    lg = _get(f"leagues?id={leagues.LEAGUES[key]['fotmob']}")
    allm = lg["fixtures"]["allMatches"]
    season = [{"id": m["id"], "kickoff": m["status"]["utcTime"][:16] + "Z",
               "home": m["home"]["name"], "away": m["away"]["name"]} for m in allm]
    out = []
    for m in allm:
        s = m["status"]
        ko = dt.datetime.fromisoformat(s["utcTime"].replace("Z", "+00:00"))
        if s.get("finished") or s.get("cancelled") or not (now - dt.timedelta(hours=3) <= ko <= now + dt.timedelta(days=HORIZON_DAYS)):
            continue
        rec = {"id": m["id"], "kickoff": s["utcTime"][:16] + "Z",
               "home": {"id": m["home"]["id"], "name": m["home"]["name"]},
               "away": {"id": m["away"]["id"], "name": m["away"]["name"]}, "type": "none",
               "h": {"starters": [], "unavailable": []}, "a": {"starters": [], "unavailable": []}}
        try:
            lu = (_get(f"matchDetails?matchId={m['id']}").get("content") or {}).get("lineup") or {}
            if lu.get("homeTeam"):
                rec["h"], rec["a"] = _side(lu["homeTeam"]), _side(lu["awayTeam"])
                # lineupType: "predicted" | "standard" (= official, but an empty shell until announced) | "unavailable"
                full = len(rec["h"]["starters"]) == 11 and len(rec["a"]["starters"]) == 11
                rec["type"] = {"predicted": "predicted", "standard": "confirmed"}.get(lu.get("lineupType"), "none") if full else "none"
                if rec["type"] == "none":
                    rec["h"]["starters"], rec["a"]["starters"] = [], []
        except Exception as e:
            print(f"  {m['id']}: {e}")
        out.append(rec)
        time.sleep(DELAY)
    return {"source": "FotMob", "fetched_at": now.strftime("%Y-%m-%dT%H:%MZ"), "season_matches": season, "matches": out}


def update(key):
    try:
        data = fetch_fotmob(key)
    except Exception as e:
        print(f"[lineups {key}] [warn] FotMob unreachable ({e}); keeping cached lineups.json", flush=True)
        return
    path = os.path.join(load.RAW, key, leagues.CURRENT, "lineups.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(data, open(path, "w", encoding="utf-8"), ensure_ascii=False)
    c = Counter(m["type"] for m in data["matches"])
    print(f"[lineups {key}] {len(data['matches'])} upcoming: {dict(c)}", flush=True)


# ---------------------------------------------------------------- mapping onto Understat ids

def _ko(s):
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00"))


def team_map(season_matches, M, season=None):
    """Source team name -> Understat team id, by voting on matches with the same kickoff (±2h) over the season."""
    us = defaultdict(list)
    for m in M:
        if m["season"] == (season or leagues.CURRENT):
            us[m["date"]].append(m)
    votes = defaultdict(Counter)
    for s in season_matches:
        k = _ko(s["kickoff"])
        for d in {str(k.date()), str((k - dt.timedelta(hours=2)).date()), str((k + dt.timedelta(hours=2)).date())}:
            for m in us.get(d, []):
                if abs((_ko(m["kickoff"]) - k).total_seconds()) <= 7200:
                    votes[s["home"]][m["home"]] += 1
                    votes[s["away"]][m["away"]] += 1
    return {n: c.most_common(1)[0][0] for n, c in votes.items()}


def player_matcher(players):
    """name -> Understat player id within one team; None when no confident match."""
    norm = {p["code"]: load._norm(p["name"]) for p in players}
    last = Counter(t[-1] for t in norm.values() if t)

    def match(name):
        n = load._norm(name)
        if not n:
            return None
        full = " ".join(n)
        for code, t in norm.items():                       # exact
            if " ".join(t) == full:
                return code
        for code, t in norm.items():                       # same token set (order / middle names)
            if set(t) == set(n) or (len(t) > 1 and set(t) <= set(n)) or (len(n) > 1 and set(n) <= set(t)):
                return code
        for code, t in norm.items():                       # mononym on one side ("Gabriel" vs "Gabriel Magalhaes")
            if (len(t) == 1 and t[0] in n) or (len(n) == 1 and n[0] in t):
                if sum(1 for u in norm.values() if n[0] in u or (len(t) == 1 and t[0] in u)) == 1:
                    return code
        for code, t in norm.items():                       # same surname, unique in squad or same initial
            if t and t[-1] == n[-1] and (last[t[-1]] == 1 or t[0][0] == n[0][0]):
                return code
        best = max(norm, key=lambda c: difflib.SequenceMatcher(None, " ".join(norm[c]), full).ratio(), default=None)
        if best and difflib.SequenceMatcher(None, " ".join(norm[best]), full).ratio() >= 0.85:
            return best
        return None
    return match


OUT_TYPES = ("injury", "suspension", "internationalDuty")


def history_unavailable(M, PM, seasons=("2025-26", "2026-27")):
    """{(season, home_id, away_id): {"h": {pid,...}, "a": {...}}} — players FotMob listed as unavailable before
    each finished match (injury / suspension / international duty; "Doubtful" excluded). Names are matched against
    everyone who has ever played for that team in the data."""
    names = defaultdict(dict)
    for r in PM:
        names[r["team"]][r["pcode"]] = r["name"]
    matchers = {t: player_matcher([{"code": c, "name": n} for c, n in ps.items()]) for t, ps in names.items()}
    out, stats = {}, Counter()
    for s in seasons:
        root = os.path.join(load.RAW, load.LEAGUE, s, "fotmob")
        if not os.path.isdir(root):
            continue
        recs = [json.load(open(os.path.join(root, f), encoding="utf-8")) for f in os.listdir(root) if f.endswith(".json")]
        tm = team_map(recs, M, s)
        for r in recs:
            h, a = tm.get(r["home"]), tm.get(r["away"])
            if h is None or a is None:
                stats["match_unmapped"] += 1
                continue
            rec = {}
            for side, t in (("h", h), ("a", a)):
                ids = set()
                for u in r[side]["unavailable"]:
                    if u["type"] not in OUT_TYPES or "doubt" in (u["ret"] or "").lower():
                        stats["skipped_doubtful_or_other"] += 1
                        continue
                    pid = matchers.get(t, lambda n: None)(u["name"])
                    stats["matched" if pid else "unmatched"] += 1
                    if pid:
                        ids.add(pid)
                rec[side] = ids
            out[(s, h, a)] = rec
    return out, dict(stats)


def attach(T, PL, M):
    """Returns ({(home_id, away_id): lineup info for export}, {player_id: (status, news)}, meta) or empties."""
    path = os.path.join(load.RAW, load.LEAGUE, leagues.CURRENT, "lineups.json")
    if not os.path.exists(path):
        return {}, {}, None
    data = json.load(open(path, encoding="utf-8"))
    tm = team_map(data["season_matches"], M)
    squads = defaultdict(list)
    for p in PL.values():
        squads[p["team"]].append(p)
    matchers = {c: player_matcher(ps) for c, ps in squads.items()}
    fx, status, miss_t, n_names, n_miss = {}, {}, set(), 0, 0
    for m in sorted(data["matches"], key=lambda m: m["kickoff"]):
        h, a = tm.get(m["home"]["name"]), tm.get(m["away"]["name"])
        if h is None or a is None:
            miss_t |= {m["home"]["name"], m["away"]["name"]} - set(tm)
            continue
        rec = {"type": m["type"], "src": data["source"], "at": data["fetched_at"]}
        for side, c in (("h", h), ("a", a)):
            mt = matchers.get(c, lambda n: None)
            xi, missing = [], []
            for name in m[side]["starters"]:
                pid = mt(name); n_names += 1
                (xi.append(pid) if pid else missing.append(name))
            n_miss += len(missing)
            rec[side] = {"xi": xi, "miss": missing}
            for u in m[side]["unavailable"]:
                pid = mt(u["name"])
                if pid and pid not in status:                  # first (= nearest) match wins
                    doubtful = "doubt" in (u["ret"] or "").lower()
                    st = "d" if doubtful else "s" if u["type"] == "suspension" else "i" if u["type"] == "injury" else "d"
                    lab = {"s": "suspenzija", "i": "povreda", "d": "upitan"}[st]
                    status[pid] = (st, f"{data['source']}: {lab}" + (f", povratak {u['ret']}" if u["ret"] and not doubtful else ""))
        fx[(h, a)] = rec
    meta = {"source": data["source"], "fetched_at": data["fetched_at"], "unmapped_teams": sorted(miss_t),
            "starters_unmatched": n_miss, "starters_total": n_names}
    return fx, status, meta


if __name__ == "__main__":
    for k in (sys.argv[1:] or list(leagues.LEAGUES)):
        update(k)
