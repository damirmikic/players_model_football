"""Blend our model with the pre-match market (football-data.co.uk odds collected days before kick-off).

    python fit_market.py         # pooled over all leagues

p ∝ p_model^(1-w) · p_open^w on 1X2 (logarithmic pooling, margin removed proportionally).
w chosen by log-loss on 2025-26, scored on 2026-27; closing market reported as the target."""
import io, json, math, os, contextlib
import numpy as np
import load, leagues, pricing as P
from fit_betas import base_preds


def fair(odds):
    inv = [1 / o for o in odds]
    return [x / sum(inv) for x in inv]


def rows_for(key):
    load.set_league(key)
    with contextlib.redirect_stdout(io.StringIO()):
        T, PL, M, PM, cur = load.build()
        bp = base_preds(M)
    mi = {(m["season"], m["home"], m["away"]): m for m in M}
    out = []
    for k, p in bp.items():
        m = mi[k]
        if m.get("odds_open") and m.get("odds_close"):
            out.append({"league": key, "season": k[0], "o": p["o"], "pm": p["p"],
                        "po": fair(m["odds_open"]), "pc": fair(m["odds_close"])})
    print(f"  {key}: {len(out)} matches with pre-match + closing odds", flush=True)
    return out


def pool(pm, po, w):
    q = [a ** (1 - w) * b ** w for a, b in zip(pm, po)]
    return [x / sum(q) for x in q]


def score(rows, f):
    rps = ll = 0.0
    for r in rows:
        p = f(r)
        rps += P.rps(p, r["o"]); ll -= math.log(p[r["o"]])
    return rps / len(rows), ll / len(rows)


if __name__ == "__main__":
    rows = [r for k in leagues.LEAGUES for r in rows_for(k)]
    tr = [r for r in rows if r["season"] == "2025-26"]
    te = [r for r in rows if r["season"] == "2026-27"]
    grid = [round(x, 2) for x in np.arange(0, 1.01, 0.05)]
    res = {w: score(tr, lambda r, w=w: pool(r["pm"], r["po"], w)) for w in grid}
    wb = min(res, key=lambda w: res[w][1])
    print("train log-loss by w:", " ".join(f"{w}:{res[w][1]:.4f}" for w in grid[::2]))
    print(f"chosen on 2025-26: w={wb}")
    out = {"w": wb}
    for name, rs in (("train 25/26", tr), ("test 26/27", te)):
        sm, so, sb, sc = (score(rs, f) for f in (lambda r: r["pm"], lambda r: r["po"],
                                                lambda r: pool(r["pm"], r["po"], wb), lambda r: r["pc"]))
        print(f"== {name} n={len(rs)}  RPS / logloss: model {sm[0]:.5f}/{sm[1]:.4f} | pre-match {so[0]:.5f}/{so[1]:.4f} | "
              f"blend {sb[0]:.5f}/{sb[1]:.4f} | closing {sc[0]:.5f}/{sc[1]:.4f}")
        for lg in leagues.LEAGUES:
            r2 = [r for r in rs if r["league"] == lg]
            if r2:
                a, b, c = (score(r2, f)[0] for f in (lambda r: r["pm"], lambda r: pool(r["pm"], r["po"], wb), lambda r: r["pc"]))
                print(f"     {lg:11s} n={len(r2):4d}  model {a:.5f}  blend {b:.5f}  closing {c:.5f}")
    json.dump(out, open(os.path.join(load.RAW, "..", "out", "market_blend.json"), "w"), indent=1)
