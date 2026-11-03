"""Timing check for soccer-rating.com: are the ratings shown next to a match frozen before kick-off, or rewritten
afterwards? Compares saved snapshots of upcoming matches (../raw/snapshots/) with the current soccer_rating.json.

    python soccer_rating.py && python check_sr_snapshot.py

If ratings of matches played since the snapshot are unchanged, the backtest in fit_elo.py used pre-match values."""
import glob, json, os, re
import numpy as np
import load

for path in sorted(glob.glob(os.path.join(load.RAW, "snapshots", "soccer_rating_*.json"))):
    key, ts = re.search(r"soccer_rating_(.+)_(\d{8}T\d{4}Z)\.json$", path).groups()
    snap = {(m["date"], m["home_id"], m["away_id"]): m for m in json.load(open(path, encoding="utf-8"))}
    now = {(m["date"], m["home_id"], m["away_id"]): m for m in json.load(open(os.path.join(load.RAW, key, "soccer_rating.json"), encoding="utf-8"))}
    played = [k for k in snap if k in now and now[k]["hg"] is not None]
    if not played:
        print(f"{key} {ts}: none of the snapshot matches played yet")
        continue
    diff = np.array([abs(now[k]["r_home"] - snap[k]["r_home"]) + abs(now[k]["r_away"] - snap[k]["r_away"]) for k in played])
    print(f"{key} {ts}: {len(played)} played; rating changed in {(diff > 0.01).sum()} "
          f"(mean |Δ| {diff.mean():.2f}, max {diff.max():.2f}) -> {'REWRITTEN after the fact' if (diff > 0.01).mean() > 0.2 else 'frozen pre-match'}")
