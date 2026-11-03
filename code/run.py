"""Full pipeline: download -> lineups -> lineup factors -> backtest -> export -> HTML.

    python run.py                    # all leagues
    python run.py EPL Serie_A        # selected leagues
    python run.py --no-fetch EPL     # reuse raw data already on disk (no downloads at all)
    python run.py --lineups          # only refresh lineups/injuries and rebuild pages (~1 min per league)
    python run.py --serve            # start local web server after running pipeline"""
import os, subprocess, sys
import leagues

args = sys.argv[1:]
fetch = "--no-fetch" not in args
only_lineups = "--lineups" in args
serve = "--serve" in args
keys = [a for a in args if not a.startswith("--")] or list(leagues.LEAGUES)
venv_py = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".venv", "bin", "python"))
py = venv_py if os.path.exists(venv_py) else sys.executable


def step(*cmd):
    print(">>", " ".join(cmd), flush=True)
    subprocess.run([py, *cmd], check=True)


if only_lineups:
    step("lineups.py", *keys)
    for k in keys:
        step("export.py", k)
    step("build.py")
    if serve:
        step("serve.py")
    sys.exit()
if fetch:
    step("sources.py", *keys)
    step("lineups.py", *keys)
for k in keys:
    for script in ("player_model.py", "backtest.py", "export.py"):
        step(script, k)
step("build.py")
if serve:
    step("serve.py")
