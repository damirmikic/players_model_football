"""Historical FotMob lineups + pre-match unavailable lists for finished matches (backtest of injury effects).

    python fotmob_history.py                    # all leagues, backtest seasons
    python fotmob_history.py EPL Serie_A

One compact file per match in ../raw/<league>/<season>/fotmob/<id>.json (cached; finished matches never change):
    {"id", "kickoff", "home", "away", "h": {"starters": [...], "unavailable": [{"name", "type", "ret"}]}, "a": {...}}
`expectedReturn` texts are as FotMob stores them today and may contain hindsight; only list membership is used."""
import json, os, sys, time, urllib.parse
import leagues, load
from lineups import _get, _side, DELAY

SEASONS = ("2025-26", "2026-27")


def fm_season(s):                     # "2025-26" -> "2025/2026"
    return f"{s[:4]}/{int(s[:4]) + 1}"


def update(key):
    for s in SEASONS:
        root = os.path.join(load.RAW, key, s, "fotmob")
        os.makedirs(root, exist_ok=True)
        lg = _get(f"leagues?id={leagues.LEAGUES[key]['fotmob']}&season={urllib.parse.quote(fm_season(s))}")
        fin = [m for m in lg["fixtures"]["allMatches"] if m["status"].get("finished")]
        todo = [m for m in fin if not os.path.exists(os.path.join(root, f"{m['id']}.json"))]
        print(f"[fotmob {key} {s}] {len(fin)} finished, {len(todo)} to fetch", flush=True)
        for i, m in enumerate(todo):
            try:
                lu = (_get(f"matchDetails?matchId={m['id']}").get("content") or {}).get("lineup") or {}
            except Exception as e:
                print(f"  {m['id']}: {e}", flush=True)
                time.sleep(5)
                continue
            rec = {"id": m["id"], "kickoff": m["status"]["utcTime"][:16] + "Z",
                   "home": m["home"]["name"], "away": m["away"]["name"],
                   "h": _side(lu.get("homeTeam") or {}), "a": _side(lu.get("awayTeam") or {})}
            json.dump(rec, open(os.path.join(root, f"{m['id']}.json"), "w", encoding="utf-8"), ensure_ascii=False)
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(todo)}", flush=True)
            time.sleep(DELAY)
    print(f"[fotmob {key}] done", flush=True)


if __name__ == "__main__":
    for k in (sys.argv[1:] or list(leagues.LEAGUES)):
        update(k)
