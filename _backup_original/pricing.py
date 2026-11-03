"""
pricing.py - sportsbook pricing & risk engine (football, tennis, basketball).
Pure Python + numpy; scipy only for section 7 (fit_ratings).
"""
import math
import numpy as np

# 1. MARGIN ---------------------------------------------------------------

def implied(odds):
    return [1.0 / o for o in odds]


def overround(odds):
    """Book total of implied probabilities, e.g. 1.06 = 6% margin."""
    return sum(implied(odds))


def _bisect(f, lo, hi, tol=1e-12, it=200):
    flo = f(lo)
    for _ in range(it):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if (fm > 0) == (flo > 0):
            lo, flo = mid, fm
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def _shin_true(q, z):
    Q = sum(q)
    return [(math.sqrt(z * z + 4 * (1 - z) * qi * qi / Q) - z) / (2 * (1 - z)) for qi in q]


def remove_margin(odds, method="power"):
    """Bookmaker odds -> fair probabilities (sum to 1).
    proportional: p = q / sum(q); power: p = q**k; shin: Shin insider model;
    additive: p = q - (sum(q)-1)/n (equal absolute margin per outcome; can go <0)."""
    q = implied(odds)
    n = len(q)
    if method == "proportional":
        s = sum(q)
        return [x / s for x in q]
    if method == "additive":
        m = (sum(q) - 1) / n
        return [x - m for x in q]
    if method == "power":
        k = _bisect(lambda k: sum(x ** k for x in q) - 1, 0.5, 5.0)
        return [x ** k for x in q]
    if method == "shin":
        if sum(q) <= 1:
            return [x / sum(q) for x in q]
        z = _bisect(lambda z: sum(_shin_true(q, z)) - 1, 1e-9, 0.5)
        return _shin_true(q, z)
    raise ValueError(method)


def apply_margin(probs, margin, method="power", min_odds=1.01):
    """Fair probabilities -> priced odds with total overround 1+margin.
    Relative cut on longshots, heaviest first: additive > power > shin > proportional."""
    p = [max(x, 1e-12) for x in probs]
    target = 1 + margin
    n = len(p)
    if method == "proportional":
        q = [x * target for x in p]
    elif method == "additive":
        q = [x + margin / n for x in p]
    elif method == "power":
        k = _bisect(lambda k: sum(x ** k for x in p) - target, 0.2, 1.0)
        q = [x ** k for x in p]
    elif method == "shin":
        def book(z):
            r = [math.sqrt(z * x + (1 - z) * x * x) for x in p]
            s = sum(r)
            return [ri * s for ri in r]
        z = _bisect(lambda z: sum(book(z)) - target, 0.0, 0.99)
        q = book(z)
    else:
        raise ValueError(method)
    return [max(min_odds, 1.0 / x) for x in q]


def two_way_line_margin(p, margin, method="power"):
    return apply_margin([p, 1 - p], margin, method)


# 2. SETTLEMENT (Asian / whole / quarter lines) ---------------------------

def _split_line(line):
    """Quarter lines split into two half-stakes: -0.25 -> (-0.5, 0)."""
    r = round(line * 4)
    if r % 2 == 0:
        return [line]
    return [line - 0.25, line + 0.25]


def settle_fracs(value, line):
    """Bet wins if value + line > 0. Returns (win_fraction, lose_fraction)."""
    parts = _split_line(line)
    w = l = 0.0
    for ln in parts:
        r = value + ln
        if r > 1e-9:
            w += 1.0 / len(parts)
        elif r < -1e-9:
            l += 1.0 / len(parts)
    return w, l


def fair_odds_from_dist(dist, line, sign=+1):
    """dist: value -> prob. Bet wins when sign*value + line > 0.
    Returns (fair_odds, P_win, P_push, P_lose) in stake fractions."""
    W = L = 0.0
    for v, pr in dist.items():
        w, l = settle_fracs(sign * v, line)
        W += pr * w
        L += pr * l
    odds = 1 + L / W if W > 0 else float("inf")
    return float(odds), float(W), float(1 - W - L), float(L)


# 3. FOOTBALL ------------------------------------------------------------

def poisson_pmf(k, lam):
    return math.exp(-lam) * lam ** k / math.factorial(k)


def score_matrix(lh, la, rho=0.0, max_goals=12):
    """P[h, a]. rho = Dixon-Coles correction (typ. -0.05..-0.15)."""
    ph = np.array([poisson_pmf(i, lh) for i in range(max_goals + 1)])
    pa = np.array([poisson_pmf(i, la) for i in range(max_goals + 1)])
    M = np.outer(ph, pa)
    if rho:
        M[0, 0] *= 1 - lh * la * rho
        M[0, 1] *= 1 + lh * rho
        M[1, 0] *= 1 + la * rho
        M[1, 1] *= 1 - rho
    return M / M.sum()


def gd_dist(M):
    d = {}
    n = M.shape[0]
    for h in range(n):
        for a in range(n):
            d[h - a] = d.get(h - a, 0) + M[h, a]
    return d


def total_dist(M):
    d = {}
    n = M.shape[0]
    for h in range(n):
        for a in range(n):
            d[h + a] = d.get(h + a, 0) + M[h, a]
    return d


def one_x_two(M):
    return float(np.tril(M, -1).sum()), float(np.trace(M)), float(np.triu(M, 1).sum())


def football_markets(lh, la, rho=-0.08, lines_total=(1.5, 2.5, 3.5),
                     lines_ah=(-1.5, -1, -0.75, -0.5, -0.25, 0, 0.25, 0.5),
                     first_half_share=0.45):
    M = score_matrix(lh, la, rho)
    h, d, a = one_x_two(M)
    out = {"lambda": (lh, la), "1X2": (h, d, a),
           "double_chance": {"1X": h + d, "12": h + a, "X2": d + a}}
    out["draw_no_bet_home"] = h / (h + a)
    tdist, gdist = total_dist(M), gd_dist(M)
    out["totals"] = {ln: {"over_fair": fair_odds_from_dist(tdist, -ln)[0],
                          "under_fair": fair_odds_from_dist(tdist, ln, -1)[0],
                          "P_over": fair_odds_from_dist(tdist, -ln)[1]}
                     for ln in lines_total}
    out["asian_handicap_home"] = {ln: {"home_fair": fair_odds_from_dist(gdist, ln)[0],
                                       "away_fair": fair_odds_from_dist(gdist, -ln, -1)[0]}
                                  for ln in lines_ah}
    out["btts_yes"] = float(1 - M[0, :].sum() - M[:, 0].sum() + M[0, 0])
    out["home_clean_sheet"] = float(M[:, 0].sum())
    out["away_clean_sheet"] = float(M[0, :].sum())
    cs = {f"{i}-{j}": float(M[i, j]) for i in range(6) for j in range(6)}
    out["correct_score"] = dict(sorted(cs.items(), key=lambda kv: -kv[1])[:15])
    out["correct_score_any_other"] = float(1 - sum(M[i, j] for i in range(6) for j in range(6)))
    s = first_half_share
    H1 = score_matrix(lh * s, la * s, rho * 0.5)
    H2 = score_matrix(lh * (1 - s), la * (1 - s), 0)
    n = H1.shape[0]
    htft = {}
    lab = lambda x, y: "1" if x > y else ("X" if x == y else "2")
    for a1 in range(n):
        for b1 in range(n):
            p1 = H1[a1, b1]
            if p1 < 1e-10:
                continue
            for a2 in range(n):
                for b2 in range(n):
                    p = p1 * H2[a2, b2]
                    if p < 1e-12:
                        continue
                    k = lab(a1, b1) + "/" + lab(a1 + a2, b1 + b2)
                    htft[k] = htft.get(k, 0) + p
    out["ht_ft"] = htft
    out["half_time_1X2"] = one_x_two(H1)
    return out


def fit_lambdas(p_home, p_away, p_over25, rho=-0.08, iters=60):
    """Solve (lh, la) reproducing fair 1X2 and O2.5."""
    T, S = 2.6, 0.3
    for _ in range(iters):
        def f_s(s):
            lh, la = max((T + s) / 2, 0.02), max((T - s) / 2, 0.02)
            h, _, a = one_x_two(score_matrix(lh, la, rho))
            return (h - a) - (p_home - p_away)
        S = _bisect(f_s, -T + 0.04, T - 0.04, tol=1e-7, it=60)

        def f_t(t):
            lh, la = max((t + S) / 2, 0.02), max((t - S) / 2, 0.02)
            M = score_matrix(lh, la, rho)
            return (1 - sum(M[i, j] for i in range(3) for j in range(3 - i))) - p_over25
        T = _bisect(f_t, abs(S) + 0.05, 8.0, tol=1e-7, it=60)
    return (T + S) / 2, (T - S) / 2


def fit_maher(matches, xi=0.0, iters=100):
    """Weighted Poisson attack/defence ratings.
    matches: [{home, away, hg, ag, days_ago}]. xi: decay per day.
    lh = base*home_adv*att[home]*def[away]; la = base*att[away]*def[home]."""
    teams = sorted({m["home"] for m in matches} | {m["away"] for m in matches})
    att = {t: 1.0 for t in teams}
    dfn = {t: 1.0 for t in teams}
    w = [math.exp(-xi * m.get("days_ago", 0)) for m in matches]
    tot_g = sum(wi * (m["hg"] + m["ag"]) for wi, m in zip(w, matches))
    base = tot_g / (2 * sum(w))
    hadv = 1.3
    for _ in range(iters):
        num = sum(wi * m["hg"] for wi, m in zip(w, matches))
        den = sum(wi * base * att[m["home"]] * dfn[m["away"]] for wi, m in zip(w, matches))
        hadv = num / den
        for t in teams:
            gs = ex = 0.0
            for wi, m in zip(w, matches):
                if m["home"] == t:
                    gs += wi * m["hg"]; ex += wi * base * hadv * dfn[m["away"]]
                elif m["away"] == t:
                    gs += wi * m["ag"]; ex += wi * base * dfn[m["home"]]
            att[t] = gs / ex if ex else 1.0
        for t in teams:
            gc = ex = 0.0
            for wi, m in zip(w, matches):
                if m["home"] == t:
                    gc += wi * m["ag"]; ex += wi * base * att[m["away"]]
                elif m["away"] == t:
                    gc += wi * m["hg"]; ex += wi * base * hadv * att[m["home"]]
            dfn[t] = gc / ex if ex else 1.0
        ma = sum(att.values()) / len(att)
        att = {t: v / ma for t, v in att.items()}
        dfn = {t: v * ma for t, v in dfn.items()}
    return {"attack": att, "defence": dfn, "home_adv": hadv, "base": base}


def maher_lambdas(r, home, away):
    return (r["base"] * r["home_adv"] * r["attack"][home] * r["defence"][away],
            r["base"] * r["attack"][away] * r["defence"][home])


def elo_update(r_home, r_away, result, k=20, home_adv=60):
    """result: 1 home win, 0.5 draw, 0 away."""
    e = 1 / (1 + 10 ** (-(r_home + home_adv - r_away) / 400))
    return r_home + k * (result - e), r_away - k * (result - e), e


# 4. TENNIS (Barnett & Clarke Markov chain) ------------------------------

def p_game(p):
    q = 1 - p
    deuce = p * p / (p * p + q * q)
    return p ** 4 * (1 + 4 * q + 10 * q * q) + 20 * p ** 3 * q ** 3 * deuce


def _tb_server(n, a_first):
    block = (n + 1) // 2
    first = (block % 2 == 0)
    return first if a_first else not first


def p_tiebreak(pa, pb, a_serves_first=True, to=7):
    from functools import lru_cache

    @lru_cache(None)
    def P(i, j):
        if i >= to and i - j >= 2:
            return 1.0
        if j >= to and j - i >= 2:
            return 0.0
        if i >= to - 1 and j >= to - 1 and i == j:
            n = i + j
            w1 = pa if _tb_server(n, a_serves_first) else 1 - pb
            w2 = pa if _tb_server(n + 1, a_serves_first) else 1 - pb
            ww, ll = w1 * w2, (1 - w1) * (1 - w2)
            return ww / (ww + ll)
        n = i + j
        w = pa if _tb_server(n, a_serves_first) else 1 - pb
        return w * P(i + 1, j) + (1 - w) * P(i, j + 1)
    return P(0, 0)


def set_dist(pa, pb, a_serves_first=True, tiebreak=True, tb_to=7):
    """Final set score distribution {(ga, gb): prob}."""
    ha, hb = p_game(pa), p_game(pb)
    states = {(0, 0): 1.0}
    final = {}
    while states:
        new = {}
        for (i, j), pr in states.items():
            n = i + j
            a_srv = (n % 2 == 0) == a_serves_first
            if tiebreak and i == 6 and j == 6:
                t = p_tiebreak(pa, pb, a_srv, tb_to)
                final[(7, 6)] = final.get((7, 6), 0) + pr * t
                final[(6, 7)] = final.get((6, 7), 0) + pr * (1 - t)
                continue
            w = ha if a_srv else 1 - hb
            for (ni, nj), p2 in (((i + 1, j), w), ((i, j + 1), 1 - w)):
                done = (ni >= 6 and ni - nj >= 2) or (nj >= 6 and nj - ni >= 2)
                if not tiebreak and ni >= 6 and nj >= 6 and ni == nj:
                    ni, nj = ni - 2, nj - 2
                if done:
                    final[(ni, nj)] = final.get((ni, nj), 0) + pr * p2
                else:
                    new[(ni, nj)] = new.get((ni, nj), 0) + pr * p2
        states = {k: v for k, v in new.items() if v > 1e-15}
    return final


def tennis_match(pa, pb, best_of=3, a_serves_first=True, tiebreak_final_set=True):
    """pa = A serve-point win prob, pb = B serve-point win prob."""
    target = best_of // 2 + 1
    cache = {}

    def sd(a_first, final):
        key = (a_first, final)
        if key not in cache:
            cache[key] = set_dist(pa, pb, a_first, tiebreak=(tiebreak_final_set or not final))
        return cache[key]

    states = {(0, 0, a_serves_first, 0, 0): 1.0}
    finals = {}
    while states:
        new = {}
        for (sa, sb, af, ga, gb), pr in states.items():
            final_set = (sa == target - 1 and sb == target - 1)
            for (x, y), p2 in sd(af, final_set).items():
                nsa, nsb = sa + (x > y), sb + (y > x)
                naf = af if (x + y) % 2 == 0 else not af
                key = (nsa, nsb, naf, ga + x, gb + y)
                if nsa == target or nsb == target:
                    finals[key] = finals.get(key, 0) + pr * p2
                else:
                    new[key] = new.get(key, 0) + pr * p2
        states = {k: v for k, v in new.items() if v > 1e-14}
    win = sum(p for (sa, sb, _, _, _), p in finals.items() if sa == target)
    set_bet, tot, hcap = {}, {}, {}
    for (sa, sb, _, ga, gb), p in finals.items():
        set_bet[f"{sa}-{sb}"] = set_bet.get(f"{sa}-{sb}", 0) + p
        tot[ga + gb] = tot.get(ga + gb, 0) + p
        hcap[ga - gb] = hcap.get(ga - gb, 0) + p
    s1 = sd(a_serves_first, best_of == 1)
    return {"p_match_A": win, "set_betting": set_bet, "total_games": tot,
            "games_diff_A": hcap,
            "p_first_set_A": sum(p for (x, y), p in s1.items() if x > y),
            "p_hold_A": p_game(pa), "p_hold_B": p_game(pb),
            "expected_games": sum(k * v for k, v in tot.items())}


def fit_tennis(p_match_A, total_games_line, best_of=3, iters=40):
    """Solve (pa, pb) from a fair match price and the fair total-games line."""
    def solve_d(avg):
        return _bisect(lambda x: tennis_match(min(avg + x, .97), max(min(avg - x, .97), .03),
                                              best_of)["p_match_A"] - p_match_A,
                       -0.3, 0.3, tol=1e-5, it=30)

    def p_over(avg):
        d = solve_d(avg)
        r = tennis_match(min(avg + d, .97), max(min(avg - d, .97), .03), best_of)
        return fair_odds_from_dist(r["total_games"], -total_games_line)[1] - 0.5

    lo, hi = 0.50, 0.90
    if p_over(lo) > 0:
        raise ValueError("total_games_line is below the lowest median this match price allows "
                         "(serve-point avg 0.50). Check the line / margin removal.")
    if p_over(hi) < 0:
        raise ValueError("total_games_line is above what serve-point avg 0.90 produces.")
    avg = _bisect(p_over, lo, hi, tol=1e-5, it=iters)
    d = solve_d(avg)
    return avg + d, avg - d


# 5. BASKETBALL ------------------------------------------------------------

def norm_cdf(x, mu=0.0, sd=1.0):
    return 0.5 * (1 + math.erf((x - mu) / (sd * math.sqrt(2))))


def discrete_normal(mu, sd, lo=None, hi=None):
    lo = int(math.floor(mu - 6 * sd)) if lo is None else lo
    hi = int(math.ceil(mu + 6 * sd)) if hi is None else hi
    d = {k: norm_cdf(k + 0.5, mu, sd) - norm_cdf(k - 0.5, mu, sd) for k in range(lo, hi + 1)}
    s = sum(d.values())
    return {k: v / s for k, v in d.items()}


# Rough starting SDs - recalibrate from residuals (result - closing line).
BASKETBALL_SD = {
    "NBA": {"margin": 12.0, "total": 18.0},
    "EUROLEAGUE": {"margin": 11.0, "total": 15.0},
    "NCAAB": {"margin": 10.5, "total": 16.0},
    "WNBA": {"margin": 11.0, "total": 15.0},
}


def basketball_markets(spread_home, total, league="NBA", sd_margin=None, sd_total=None,
                       spread_lines=None, total_lines=None, period_share=1.0):
    """spread_home: expected home margin (home -5.5 -> 5.5)."""
    sdm = (sd_margin or BASKETBALL_SD[league]["margin"]) * math.sqrt(period_share)
    sdt = (sd_total or BASKETBALL_SD[league]["total"]) * math.sqrt(period_share)
    mu_m, mu_t = spread_home * period_share, total * period_share
    md, td = discrete_normal(mu_m, sdm), discrete_normal(mu_t, sdt)
    p0 = md.get(0, 0)
    p_home = sum(v for k, v in md.items() if k > 0)
    out = {"mean_margin": mu_m, "mean_total": mu_t,
           "moneyline_home_incl_OT": p_home + 0.5 * p0 if period_share == 1.0 else None,
           "three_way_home_draw_away": (p_home, p0, 1 - p_home - p0)}
    spread_lines = spread_lines or [round(-mu_m * 2) / 2 + x for x in (-2, -1, -0.5, 0, 0.5, 1, 2)]
    total_lines = total_lines or [round(mu_t * 2) / 2 + x for x in (-3, -1.5, -0.5, 0, 0.5, 1.5, 3)]
    out["spread_home"] = {ln: fair_odds_from_dist(md, ln)[0] for ln in spread_lines}
    out["totals_over"] = {ln: fair_odds_from_dist(td, -ln)[0] for ln in total_lines}
    out["team_total_sd"] = math.sqrt((sdm ** 2 + sdt ** 2) / 4)
    out["home_team_total_mean"] = (mu_t + mu_m) / 2
    out["away_team_total_mean"] = (mu_t - mu_m) / 2
    return out


def team_total_over(mean, sd, line):
    return fair_odds_from_dist(discrete_normal(mean, sd), -line)[0]


# 6. RISK ------------------------------------------------------------------

def kelly(p, odds, fraction=1.0):
    f = (p * odds - 1) / (odds - 1)
    return max(0.0, f * fraction)


def accumulator_overround(overrounds):
    return float(np.prod(overrounds))


def expected_hold(bets):
    """bets: list of (stake, odds, true_prob)."""
    return sum(s * (1 - p * o) for s, o, p in bets)


def _bet_pnl_customer(b, h, a):
    s, o = b["stake"], b["odds"]
    if b["market"] == "1X2":
        w = {"1": h > a, "X": h == a, "2": h < a}[b["sel"]]
        return s * (o - 1) if w else -s
    if b["market"] == "btts":
        y = h > 0 and a > 0
        return s * (o - 1) if (y == (b["sel"] == "yes")) else -s
    if b["market"] == "cs":
        hh, aa = map(int, b["sel"].split("-"))
        return s * (o - 1) if (h, a) == (hh, aa) else -s
    if b["market"] == "total":
        v = (h + a) * (1 if b["sel"] == "over" else -1)
        ln = -b["line"] if b["sel"] == "over" else b["line"]
    else:
        v = (h - a) * (1 if b["sel"] == "home" else -1)
        ln = b["line"]
    w, l = settle_fracs(v, ln)
    return s * w * (o - 1) - s * l


def worst_scores(lh, la, bets, rho=-0.08, top=8, max_goals=8):
    """Exact BOOK P&L by scoreline, worst first: [(score, prob, pnl)]."""
    M = score_matrix(lh, la, rho, max_goals=max_goals)
    res = []
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            pnl = -sum(_bet_pnl_customer(b, h, a) for b in bets)
            res.append((f"{h}-{a}", float(M[h, a]), pnl))
    res.sort(key=lambda x: x[2])
    return res[:top]


def simulate_football_book(lh, la, bets, rho=-0.08, n=200_000, seed=1):
    """Monte Carlo BOOK P&L on one match. bets: [{market, sel, line, stake, odds}],
    market in {'1X2','total','ah','btts','cs'}; AH line from the selection's view."""
    rng = np.random.default_rng(seed)
    M = score_matrix(lh, la, rho)
    k = M.shape[0]
    # exact P&L per scoreline, then sample scorelines
    grid = np.array([[-sum(_bet_pnl_customer(b, h, a) for b in bets) for a in range(k)]
                     for h in range(k)])
    idx = rng.choice(k * k, size=n, p=M.ravel())
    pnl = grid.ravel()[idx]
    return {"expected_pnl": float((M * grid).sum()), "sd": float(pnl.std()),
            "p_loss": float(M[grid < 0].sum()),
            "VaR95": float(np.percentile(pnl, 5)), "VaR99": float(np.percentile(pnl, 1)),
            "worst": float(grid[M > 1e-6].min()), "best": float(grid[M > 1e-6].max()),
            "turnover": float(sum(b["stake"] for b in bets))}


# 7. RATINGS FROM RESULTS (penalized Poisson ML; leagues with no reference market)
# Needs scipy for fit_ratings (pip install scipy --break-system-packages).

import datetime as _dt


def fit_ratings(matches, asof=None, current_teams=None, prior=None, ridge=3.0,
                half_life_days=180, prev_season_weight=0.7, goal_cap=6):
    """Penalized weighted Poisson regression (a regularized GLM) on match results.
    matches: [{home, away, hg, ag, date (datetime.date or 'YYYY-MM-DD'), season: 1 current / 0 previous,
               forfeit: bool (optional)}]; home/away should be STABLE ids (club id), not display names.
    log lam_home = mu + H + att[home] + dfn[away];  log lam_away = mu + att[away] + dfn[home]
    (dfn > 0 = concedes more). Weights: exp(-ln2*age/half_life) * (prev_season_weight if season 0).
    L2 penalty ridge*((att-prior_att)^2 + (dfn-prior_dfn)^2) shrinks thin data toward the prior.
    prior: {team: (att0, dfn0)}; missing teams -> (0, 0). Forfeits are excluded; goals capped.
    Returns {mu, H, att, dfn} centred on current_teams so mu is this season's league baseline."""
    from scipy.optimize import minimize
    rows = []
    for m in matches:
        if m.get("forfeit") or m.get("hg") is None:
            continue
        d = m["date"] if isinstance(m["date"], _dt.date) else _dt.date.fromisoformat(m["date"])
        rows.append((m["home"], m["away"], min(m["hg"], goal_cap), min(m["ag"], goal_cap), d, m.get("season", 1)))
    asof = asof or max(r[4] for r in rows)
    current_teams = list(current_teams or sorted({r[0] for r in rows if r[5] == 1} | {r[1] for r in rows if r[5] == 1}, key=str))
    prior = prior or {}
    teams = sorted({r[0] for r in rows} | {r[1] for r in rows} | set(current_teams), key=str)
    ix = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    hi = np.array([ix[r[0]] for r in rows]); ai = np.array([ix[r[1]] for r in rows])
    hg = np.array([r[2] for r in rows], float); ag = np.array([r[3] for r in rows], float)
    xi = math.log(2) / half_life_days
    w = np.array([math.exp(-xi * max((asof - r[4]).days, 0)) * (1.0 if r[5] == 1 else prev_season_weight)
                  for r in rows])
    pa = np.array([prior.get(t, (0.0, 0.0))[0] for t in teams])
    pd = np.array([prior.get(t, (0.0, 0.0))[1] for t in teams])

    def nll(x):
        mu, H, att, dfn = x[0], x[1], x[2:2 + n], x[2 + n:]
        lh = np.exp(mu + H + att[hi] + dfn[ai]); la = np.exp(mu + att[ai] + dfn[hi])
        ll = np.sum(w * (hg * np.log(lh) - lh + ag * np.log(la) - la))
        pen = ridge * (np.sum((att - pa) ** 2) + np.sum((dfn - pd) ** 2))
        rh, ra = w * (hg - lh), w * (ag - la)
        g = np.zeros_like(x)
        g[0] = rh.sum() + ra.sum(); g[1] = rh.sum()
        g[2:2 + n] = np.bincount(hi, rh, n) + np.bincount(ai, ra, n) - 2 * ridge * (att - pa)
        g[2 + n:] = np.bincount(ai, rh, n) + np.bincount(hi, ra, n) - 2 * ridge * (dfn - pd)
        return -(ll - pen), -g

    x0 = np.zeros(2 + 2 * n); x0[0] = math.log(1.3); x0[1] = 0.2
    res = minimize(nll, x0, jac=True, method="L-BFGS-B")
    x = res.x
    att = dict(zip(teams, x[2:2 + n])); dfn = dict(zip(teams, x[2 + n:]))
    ca = float(np.mean([att[t] for t in current_teams])); cd = float(np.mean([dfn[t] for t in current_teams]))
    return {"mu": float(x[0] + ca + cd), "H": float(x[1]),
            "att": {t: float(att[t] - ca) for t in current_teams},
            "dfn": {t: float(dfn[t] - cd) for t in current_teams}, "converged": bool(res.success)}


def ratings_lambdas(r, home, away):
    return (math.exp(r["mu"] + r["H"] + r["att"][home] + r["dfn"][away]),
            math.exp(r["mu"] + r["att"][away] + r["dfn"][home]))


def new_team_prior(kind, shift=0.15):
    """Starting prior for a team with no history in this league.
    'promoted' (from the tier below): weaker attack, weaker defence; 'relegated' (from above): stronger.
    Replace the fixed shift with the measured tier offset once enough promoted/relegated history exists."""
    return {"promoted": (-shift, shift), "relegated": (shift, -shift)}.get(kind, (0.0, 0.0))


def rps(probs, outcome_index):
    """Ranked probability score for ordered outcomes (1, X, 2). Lower is better."""
    o = np.zeros(len(probs)); o[outcome_index] = 1
    c, e = np.cumsum(probs), np.cumsum(o)
    return float(np.sum((c[:-1] - e[:-1]) ** 2) / (len(probs) - 1))


def walk_forward_backtest(matches, fit_kwargs=None, rho=0.0, start_round=3, prior=None):
    """Time-series cross-validation: for each round k >= start_round of the current season, fit only on
    matches before round k (plus previous season) and score round k. matches need 'round' for season 1.
    Baseline = previous-season 1/X/2 frequencies. Returns mean RPS, baseline RPS, log-loss, n, draw calibration."""
    fit_kwargs = fit_kwargs or {}
    cur = [m for m in matches if m.get("season", 1) == 1 and m.get("hg") is not None and not m.get("forfeit")]
    prev = [m for m in matches if m.get("season", 1) == 0 and m.get("hg") is not None and not m.get("forfeit")]
    base_src = prev or cur
    bh = np.mean([m["hg"] > m["ag"] for m in base_src]); bd = np.mean([m["hg"] == m["ag"] for m in base_src])
    base = [bh, bd, 1 - bh - bd]
    teams = sorted({m["home"] for m in cur} | {m["away"] for m in cur}, key=str)
    s_model, s_base, ll, pdraw, odraw = [], [], [], [], []
    for k in range(start_round, max(m["round"] for m in cur) + 1):
        test = [m for m in cur if m["round"] == k]
        train = prev + [m for m in cur if m["round"] < k]
        if not test or not any(m.get("season", 1) == 1 for m in train):
            continue
        asof = min(m["date"] if isinstance(m["date"], _dt.date) else _dt.date.fromisoformat(m["date"]) for m in test)
        r = fit_ratings(train, asof=asof, current_teams=teams, prior=prior, **fit_kwargs)
        for m in test:
            p = one_x_two(score_matrix(*ratings_lambdas(r, m["home"], m["away"]), rho))
            o = 0 if m["hg"] > m["ag"] else (1 if m["hg"] == m["ag"] else 2)
            s_model.append(rps(p, o)); s_base.append(rps(base, o)); ll.append(-math.log(max(p[o], 1e-12)))
            pdraw.append(p[1]); odraw.append(o == 1)
    return {"rps": float(np.mean(s_model)), "rps_baseline": float(np.mean(s_base)),
            "skill_vs_baseline": float(1 - np.mean(s_model) / np.mean(s_base)),
            "logloss": float(np.mean(ll)), "n": len(s_model),
            "draw_pred": float(np.mean(pdraw)), "draw_obs": float(np.mean(odraw))}


def tune_ratings(matches, grid=None, **kw):
    """Grid search of the ML hyperparameters by walk-forward RPS. Returns results sorted best first."""
    import itertools
    grid = grid or {"ridge": [1, 3, 6], "half_life_days": [90, 180, 365], "prev_season_weight": [0.4, 0.7, 1.0]}
    names = list(grid)
    out = []
    for vals in itertools.product(*[grid[n] for n in names]):
        fk = dict(zip(names, vals))
        out.append({**fk, **walk_forward_backtest(matches, fit_kwargs=fk, **kw)})
    return sorted(out, key=lambda d: d["rps"])


def rho_profile(matches, ratings, grid=np.arange(-0.2, 0.11, 0.02)):
    """Dixon-Coles rho by profile likelihood on (in-sample) fitted lambdas. Returns (best_rho, {rho: loglik})."""
    ll = {}
    for rho in grid:
        s = 0.0
        for m in matches:
            if m.get("hg") is None or m["home"] not in ratings["att"] or m["away"] not in ratings["att"]:
                continue
            M = score_matrix(*ratings_lambdas(ratings, m["home"], m["away"]), rho)
            s += math.log(max(M[min(m["hg"], 12), min(m["ag"], 12)], 1e-15))
        ll[round(float(rho), 3)] = s
    return max(ll, key=ll.get), ll


def standings(matches, deductions=None):
    """League table from played matches: points, then goal difference, then goals scored."""
    t = {}
    for m in matches:
        if m.get("hg") is None:
            continue
        for team, f, a in ((m["home"], m["hg"], m["ag"]), (m["away"], m["ag"], m["hg"])):
            s = t.setdefault(team, {"p": 0, "w": 0, "d": 0, "l": 0, "gf": 0, "ga": 0, "pts": 0})
            s["p"] += 1; s["gf"] += f; s["ga"] += a
            if f > a: s["w"] += 1; s["pts"] += 3
            elif f == a: s["d"] += 1; s["pts"] += 1
            else: s["l"] += 1
    for team, d in (deductions or {}).items():
        if team in t:
            t[team]["pts"] -= d
    order = sorted(t, key=lambda x: (-t[x]["pts"], -(t[x]["gf"] - t[x]["ga"]), -t[x]["gf"], str(x)))
    for i, team in enumerate(order):
        t[team]["pos"] = i + 1
    return t
