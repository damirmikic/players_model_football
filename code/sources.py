"""Download raw data into ../raw/<league>/. Finished-match files are immutable and cached forever;
league tables, odds and FPL status for the current season are re-fetched on every run.

    python sources.py                 # all leagues
    python sources.py EPL Serie_A     # selected leagues

Understat:  getLeagueData/<league>/<year>  (fixtures, results, match xG)
            getMatchData/<match_id>        (rosters: minutes, positions, subs, xG/xA; shots with minute)
            Endpoints need a session cookie + Referer; responses are gzip.
football-data.co.uk: mmz4281/<yyyy>/<code>.csv (results + closing odds, incl. Betfair exchange)
FPL API:    bootstrap-static (EPL only: injury status, news, chance of playing)"""
import gzip, json, os, sys, time, urllib.request, http.cookiejar
import leagues as L

RAW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "raw")
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36"
DELAY = 0.35          # seconds between Understat match requests


class Understat:
    BASE = "https://understat.com/"

    def __init__(self):
        self.op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
        self._get("league/EPL", ajax=False)          # session cookie

    def _get(self, path, referer=None, ajax=True, tries=4):
        h = {"User-Agent": UA, "Accept-Encoding": "gzip"}
        if ajax:
            h.update({"X-Requested-With": "XMLHttpRequest", "Referer": referer or self.BASE})
        for i in range(tries):
            try:
                with self.op.open(urllib.request.Request(self.BASE + path, headers=h), timeout=30) as r:
                    b = r.read()
                    if r.headers.get("Content-Encoding") == "gzip":
                        b = gzip.decompress(b)
                    return b
            except Exception as e:
                if i == tries - 1:
                    raise
                print(f"  retry {path}: {e}", flush=True)
                time.sleep(3 * (i + 1))

    def league(self, code, year):
        return json.loads(self._get(f"getLeagueData/{code}/{year}", f"{self.BASE}league/{code}/{year}"))

    def match(self, mid):
        return json.loads(self._get(f"getMatchData/{mid}", f"{self.BASE}match/{mid}"))


def _fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read()


def _write(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data if isinstance(data, bytes) else json.dumps(data, ensure_ascii=False).encode("utf-8"))
    os.replace(tmp, path)


def update(key, us=None):
    cfg = L.LEAGUES[key]
    root = os.path.join(RAW, key)
    us = us or Understat()
    for s in L.SEASONS:
        cur = s == L.CURRENT
        lp = os.path.join(root, s, "understat_league.json")
        if cur or not os.path.exists(lp):
            _write(lp, us.league(cfg["understat"], L.understat_year(s)))
        lg = json.load(open(lp, encoding="utf-8"))
        done = [m["id"] for m in lg["dates"] if m["isResult"]]
        todo = [mid for mid in done if not os.path.exists(os.path.join(root, s, "matches", mid + ".json"))]
        print(f"[{key} {s}] {len(done)} finished, {len(todo)} match files to fetch", flush=True)
        for i, mid in enumerate(todo):
            _write(os.path.join(root, s, "matches", mid + ".json"), us.match(mid))
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(todo)}", flush=True)
            time.sleep(DELAY)
        op = os.path.join(root, s, "fd_odds.csv")
        if cur or not os.path.exists(op):
            try:
                _write(op, _fetch(f"https://football-data.co.uk/mmz4281/{L.fd_code(s)}/{cfg['fd']}.csv"))
            except Exception as e:
                print(f"  football-data {s}: {e}")
    if cfg["fpl"]:
        _write(os.path.join(root, L.CURRENT, "fpl_bootstrap.json"),
               _fetch("https://fantasy.premierleague.com/api/bootstrap-static/"))
    print(f"[{key}] done", flush=True)


if __name__ == "__main__":
    keys = sys.argv[1:] or list(L.LEAGUES)
    us = Understat()
    for k in keys:
        update(k, us)
