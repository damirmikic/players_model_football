"""soccer-rating.com: pre-match Elo-type ratings for both teams of every match on a team's page (all competitions,
~3 seasons back). Used as an external rating signal and for the fixture calendar (rest days, European games).

    python soccer_rating.py                 # all leagues
    python soccer_rating.py EPL Serie_A

Fetches the league page (current teams), each team page, then the pages of league opponents not yet fetched
(relegated teams), so every league match of the backtest seasons is covered. Raw HTML is cached in
../raw/<league>/soccer_rating/<team_id>.html (re-fetched on every run for current teams); parsed matches go to
../raw/<league>/soccer_rating.json:
    [{"date", "comp", "home", "away", "home_id", "away_id", "r_home", "r_away", "odds": [1, X, 2] | None, "hg", "ag"}]
robots.txt allows this (only /chart.pl is disallowed); no explicit terms found — ask the site before production use."""
import datetime as dt, html, json, os, re, sys, time, urllib.request
import leagues, load

BASE = "https://www.soccer-rating.com"
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36"
DELAY = 2.0
CODES = {"EPL": ("England", "UK1"), "La_liga": ("Spain", "ES1"), "Bundesliga": ("Germany", "DE1"),
         "Serie_A": ("Italy", "IT1"), "Ligue_1": ("France", "FR1")}
SINCE = dt.date(2024, 7, 1)

ROW = re.compile(r'<tr bgcolor="#[0-9a-f]+"><td class="nomobil">\d*</td><td>(\d\d\.\d\d\.\d\d)<div class="ismobil">([A-Z0-9]+)</div></td>'
                 r'<td>(.*?)</td>(.*?)</tr>', re.S)
LINK = re.compile(r'<a href="/([^"/]+)/(w?\d+)/" class="list">(.*?)</a>')


def _get(path):
    req = urllib.request.Request(BASE + path, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode("utf-8", errors="ignore")


def league_teams(key):
    country, code = CODES[key]
    s = _get(f"/{country}/{code}/")
    return {tid: html.unescape(name) for _, tid, name in re.findall(r'href="/([^"/]+)/(\d+)/"[^>]*>([^<]+)</a>', s)}


def parse_team(page, tid, tname):
    out = []
    for date, comp, teams, rest in ROW.findall(page):
        d = dt.datetime.strptime(date, "%d.%m.%y").date()
        if d < SINCE:
            continue
        cells = [html.unescape(re.sub(r"<[^>]+>", "", c)).strip() for c in re.findall(r"<td[^>]*>(.*?)</td>", rest, re.S)]
        # cells: odds1, oddsX, odds2, mobile-odds, comp, r_home, r_away, result
        try:
            r_h, r_a = float(cells[5]), float(cells[6])
        except (IndexError, ValueError):
            continue
        res = re.match(r"^(\d+):(\d+)$", cells[7] if len(cells) > 7 else "")
        try:
            odds = [float(cells[0]), float(cells[1]), float(cells[2])]
        except ValueError:
            odds = None
        link = LINK.search(teams)
        if not link:
            continue
        opp_id, opp = link.group(2), html.unescape(link.group(3))
        home_is_opp = teams.strip().startswith("<a ")
        h, hid, a, aid = (opp, opp_id, tname, tid) if home_is_opp else (tname, tid, opp, opp_id)
        out.append({"date": str(d), "comp": comp, "home": h, "away": a, "home_id": hid, "away_id": aid,
                    "r_home": r_h, "r_away": r_a, "odds": odds,
                    "hg": int(res.group(1)) if res else None, "ag": int(res.group(2)) if res else None})
    return out


def update(key):
    root = os.path.join(load.RAW, key, "soccer_rating")
    os.makedirs(root, exist_ok=True)
    code = CODES[key][1]
    teams = league_teams(key)
    names = dict(teams)
    matches, done = {}, set()
    queue = list(teams)
    fetched = 0
    while queue:
        tid = queue.pop(0)
        if tid in done:
            continue
        done.add(tid)
        path = os.path.join(root, f"{tid}.html")
        if tid in teams or not os.path.exists(path):          # current teams always refreshed
            slug = re.sub(r"[^A-Za-z0-9]+", "-", names.get(tid, "x")).strip("-")
            try:
                page = _get(f"/{slug}/{tid}/")
            except Exception as e:
                print(f"  {tid} {names.get(tid)}: {e}", flush=True)
                continue
            open(path, "w", encoding="utf-8").write(page)
            fetched += 1
            time.sleep(DELAY)
        page = open(path, encoding="utf-8").read()
        for m in parse_team(page, tid, names.get(tid, tid)):
            matches[(m["date"], m["home_id"], m["away_id"])] = m
            for side in ("home", "away"):
                oid = m[side + "_id"]
                names.setdefault(oid, m[side])
                if m["comp"] == code and oid not in done and oid not in queue:
                    queue.append(oid)                          # league opponent (e.g. relegated team)
    rows = sorted(matches.values(), key=lambda m: m["date"])
    json.dump(rows, open(os.path.join(load.RAW, key, "soccer_rating.json"), "w", encoding="utf-8"), ensure_ascii=False)
    lg = [m for m in rows if m["comp"] == code]
    print(f"[soccer-rating {key}] {len(done)} teams ({fetched} fetched), {len(rows)} matches, {len(lg)} league matches", flush=True)


if __name__ == "__main__":
    for k in (sys.argv[1:] or list(CODES)):
        update(k)
