"""Fit the two lineup elasticities on walk-forward team predictions, pooled over leagues:
    λ_team *= inj ** beta_inj * rot ** beta_rot
inj = available XI / habitual XI (FotMob pre-match unavailable list), rot = actual XI / available XI.
Both are centred per league (divided by their geometric mean on the fitting data): the habitual XI is a slightly
weaker attack than what managers actually field (+1-3%), and that drift must not move λ.
Criterion: goal NLL + xG quasi-NLL (both λ-level, like fit_betas); RPS reported. Holdout: choose on 2025-26,
score on 2026-27.

    python fit_lineup.py            # print grids (needs player_model.py run per league first)
    python fit_lineup.py --save     # write pooled beta_inj / beta_rot into ../model_config.json"""
import io, json, math, sys, contextlib
import numpy as np
import load, leagues, pricing as P
from fit_betas import base_preds

GRID = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5]


def rows_for(key):
    load.set_league(key)
    with contextlib.redirect_stdout(io.StringIO()):
        T, PL, M, PM, cur = load.build()
        bp = base_preds(M)
    rho = load.config()["rho"]
    mi = {(m["season"], m["home"], m["away"]): m for m in M}
    out = []
    for f in json.load(open(load.out("lineup_factors.json"))):
        k = (f["season"], f["home"], f["away"])
        if k in bp and f.get("h_fm") and f.get("a_fm"):
            out.append((bp[k], {**f, "league": key}, mi[k], rho))
    return out


def centres(rows):
    """Per-league geometric mean of inj and rot over `rows`."""
    acc = {}
    for p, f, m, rho in rows:
        a = acc.setdefault(f["league"], [0.0, 0.0, 0])
        a[0] += math.log(f["h_inj"]) + math.log(f["a_inj"]); a[1] += math.log(f["h_rot"]) + math.log(f["a_rot"]); a[2] += 2
    return {k: (math.exp(i / n), math.exp(r / n)) for k, (i, r, n) in acc.items()}


def score(rows, bi, br, cen=None):
    cen = cen or centres(rows)
    gn = xn = rps = 0.0
    for p, f, m, rho in rows:
        ci, cr = cen[f["league"]]
        lh = p["lh"] * (f["h_inj"] / ci) ** bi * (f["h_rot"] / cr) ** br
        la = p["la"] * (f["a_inj"] / ci) ** bi * (f["a_rot"] / cr) ** br
        gn += lh - p["hg"] * math.log(lh) + la - p["ag"] * math.log(la)
        xn += lh - m["hxg"] * math.log(lh) + la - m["axg"] * math.log(la)
        rps += P.rps(P.one_x_two(P.score_matrix(lh, la, rho)), p["o"])
    n = len(rows)
    return {"crit": (gn + xn) / n, "goal": gn / n, "xg": xn / n, "rps": rps / n, "n": n}


if __name__ == "__main__":
    allrows = {k: rows_for(k) for k in leagues.LEAGUES}
    pooled = [r for rs in allrows.values() for r in rs]
    print("matches with FotMob lists:", {k: len(v) for k, v in allrows.items()})
    for name, rs in [("POOLED", pooled)] + list(allrows.items()):
        res = {(bi, br): score(rs, bi, br) for bi in GRID for br in GRID}
        b = min(res, key=lambda k: res[k]["crit"])
        z = res[(0.0, 0.0)]
        print(f"== {name} n={z['n']}: best beta_inj {b[0]} beta_rot {b[1]} | crit {res[b]['crit']:.4f} vs none {z['crit']:.4f} "
              f"| RPS {res[b]['rps']:.5f} vs none {z['rps']:.5f}")
        if name == "POOLED":
            print("   crit grid (rows beta_inj, cols beta_rot):")
            for bi in GRID:
                print(f"   {bi:4}: " + " ".join(f"{res[(bi, br)]['crit']:.4f}" for br in GRID))
            print("   RPS grid:")
            for bi in GRID:
                print(f"   {bi:4}: " + " ".join(f"{res[(bi, br)]['rps']:.5f}" for br in GRID))
    # holdout: choose on 2025-26, score on 2026-27
    tr = [r for r in pooled if r[1]["season"] == "2025-26"]
    te = [r for r in pooled if r[1]["season"] == "2026-27"]
    ctr = centres(tr)
    b = min(((bi, br) for bi in GRID for br in GRID), key=lambda k: score(tr, *k, ctr)["crit"])
    s_b, s_0 = score(te, *b, ctr), score(te, 0, 0, ctr)
    print(f"== HOLDOUT chosen on 2025-26: {b}; on 2026-27 (n={s_b['n']}): crit {s_b['crit']:.4f} vs none {s_0['crit']:.4f}, "
          f"RPS {s_b['rps']:.5f} vs none {s_0['rps']:.5f}")
    if "--save" in sys.argv:
        best = min(((bi, br) for bi in GRID for br in GRID), key=lambda k: score(pooled, *k)["crit"])
        c = json.load(open("../model_config.json", encoding="utf-8"))
        c["beta_inj"], c["beta_rot"] = best
        json.dump(c, open("../model_config.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
        print("saved", best)
