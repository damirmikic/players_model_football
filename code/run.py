"""Full pipeline: download -> lineups -> lineup factors -> backtest -> export -> HTML.

    python run.py                    # all leagues
    python run.py EPL Serie_A        # selected leagues
    python run.py --no-fetch EPL     # reuse raw data already on disk (no downloads at all)
    python run.py --lineups          # only refresh lineups/injuries and rebuild pages (~1 min per league)"""
import subprocess, sys
import leagues

args = sys.argv[1:]
fetch = "--no-fetch" not in args
only_lineups = "--lineups" in args
keys = [a for a in args if not a.startswith("--")] or list(leagues.LEAGUES)
py = sys.executable


def step(*cmd):
    print(">>", " ".join(cmd), flush=True)
    subprocess.run([py, *cmd], check=True)


if only_lineups:
    step("lineups.py", *keys)
    for k in keys:
        step("export.py", k)
    step("build.py")
    sys.exit()
if fetch:
    step("sources.py", *keys)
    step("lineups.py", *keys)
for k in keys:
    for script in ("player_model.py", "backtest.py", "export.py"):
        step(script, k)
step("build.py")
