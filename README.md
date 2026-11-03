# Football Match Pricing & Lineup Model (`epl_model_kod`)

End-to-end football match pricing engine, walk-forward backtesting suite, and standalone HTML sportsbook calculator covering Europe's top five leagues:

- **Premier League** (`EPL`)
- **La Liga** (`La_liga`)
- **Bundesliga** (`Bundesliga`)
- **Serie A** (`Serie_A`)
- **Ligue 1** (`Ligue_1`)

---

## Overview

The pipeline ingests match results, shot/player xG data, closing market odds, and pre-match lineup/injury feeds, fits team and player ratings as of any target date, evaluates walk-forward out-of-sample accuracy, and compiles self-contained interactive HTML calculators (`build/kalkulator_<league>.html`) for pricing upcoming fixtures.

### Core Modeling Pipeline

1. **Team Ratings (`code/team_model.py`, `code/pricing.py`)**
   - Penalized time-decayed Poisson GLM (`pricing.fit_ratings`) estimating team attack ($\alpha_i$), defence ($\delta_j$), league base rate ($\mu$), and home advantage ($H$).
   - **Target blend (`mkt_w`, `xg_w`)**: Supports fitting on a blend of actual goals, Understat xG, and implied closing-market goal expectations $(\lambda_h, \lambda_a)$ inverted from historical closing 1X2 + O/U 2.5 odds (`football-data.co.uk`). Crucially, only **past** matches' closing odds are used to estimate team ratings; the market of the match being priced is never used.
   - **Two-regime time decay (`fit` / `fit_late`)**: Uses stronger regularization and longer memory (`ridge=1.0`, `half_life_days=365`) in early rounds (`1–10`), switching to shorter memory (`ridge=0.5`, `half_life_days=90`, `prev_season_weight=0.5`) from round 11 onward (`late_round`).
   - **Promoted team priors (`promo_shift`)**: Applies an initial penalty prior to newly promoted teams until current-season results accumulate.

2. **Score Matrix & Dixon–Coles Adjustment (`code/pricing.py`, `web/pricing.js`)**
   - Bivariate Poisson score grid with per-league Dixon–Coles low-score correlation parameter $\rho$ (`fit_rho.py`) adjusting `0-0`, `1-0`, `0-1`, and `1-1` cells.
   - Derives fair probabilities and bookmaker odds (with power/Shin/proportional/additive overround) across **1X2**, **Double Chance**, **Draw No Bet**, **Asian Handicap** (whole, half, and quarter lines), **Totals (Over/Under)**, **BTTS**, **Exact Score**, **Team Totals**, and **HT/FT**.

3. **Player Ratings & Lineup Adjustments (`code/player_model.py`, `code/lineups.py`)**
   - **Attack rating ($c_i$)**: Empirical-Bayes shrunk per-90 $\text{npxG} + \text{xA}$ pulled toward positional priors (`GK`, `DEF`, `MID`, `FWD`).
   - **Defence rating ($d_i$)**: Shrunk log ratio of on-pitch vs. off-pitch team $\text{npxGA}$ rate.
   - **Habitual / Available / Confirmed XI**: Infers each team's habitual formation and starting XI from recency-weighted starts, flags injured/suspended players (via FotMob and FPL API), and separates lineup effects into pre-match **injury impact** ($\beta_{\text{inj}}$) and **lineup rotation surprise** ($\beta_{\text{rot}}$), centered per league.

---

## Directory Structure

```text
.
├── model_config.json        # Global & per-league hyperparameters and calibration notes
├── code/                    # Python pipeline, models, and calibration scripts
│   ├── run.py               # Master CLI entrypoint for full or incremental runs
│   ├── leagues.py           # League registry (Understat, football-data.co.uk, FotMob IDs)
│   ├── sources.py           # Downloader for Understat, football-data.co.uk CSVs, and FPL API
│   ├── lineups.py           # Upcoming match lineups & injuries fetcher (FotMob) + ID matcher
│   ├── fotmob_history.py    # Historical FotMob lineup/unavailable fetcher for backtesting
│   ├── soccer_rating.py     # External Elo & all-competition fixture calendar scraper
│   ├── load.py              # Normalizes raw files into clean tables & computes market λ cache
│   ├── pricing.py           # Core sportsbook math, Poisson GLM solver, and market pricing engine
│   ├── team_model.py        # Walk-forward team model evaluation & helper routines
│   ├── player_model.py      # Player attack/defence ratings & expected XI builder
│   ├── backtest.py          # Walk-forward backtest vs. base-rate and closing-market benchmarks
│   ├── export.py            # Fits models as of TODAY and exports build/data_<league>.json
│   ├── build.py             # Bundles web/app.html + web/pricing.js + data_<league>.json into HTML
│   ├── fit_mkt_ratings.py   # Calibrates closing-market λ weight (mkt_w)
│   ├── tune_mkt.py          # Grid search for ridge / half-life / prev_season_weight
│   ├── tune_team.py         # Per-league tuning of team model hyperparameters
│   ├── fit_rho.py           # Per-league Dixon-Coles rho calibration
│   ├── fit_lineup.py        # Fits centered injury (beta_inj) and rotation (beta_rot) elasticities
│   ├── fit_betas.py         # Fits raw lineup elasticities (beta_a, beta_d)
│   ├── fit_market.py        # Log-pooling blend experiment with opening market odds
│   ├── fit_elo.py           # Tests orthogonalized Elo & rest-day adjustments
│   └── serve.py             # Local web server with live FotMob lineup refresh endpoint
├── web/                     # Frontend calculator source
│   ├── app.html             # Single-page calculator UI template
│   ├── pricing.js           # Browser port of the pricing engine
│   └── test_parity.js       # Node check verifying numerical parity between pricing.py & pricing.js
├── raw/                     # Cached raw data per league & season (Understat, odds CSVs, FotMob)
├── out/                     # Per-league backtest summaries, lineup factors, and tuning grids
└── build/                   # Generated index.html (all leagues), data_<league>.json, and kalkulator_<league>.html
```

---

## Quick Start

### 1. Environment Setup

Use the local virtual environment (requires `numpy`, `scipy`, `pandas`):

```bash
source .venv/bin/activate
```

### 2. Run the Full Pipeline

All commands in `code/` are designed to be run from inside the `code/` directory:

```bash
cd code

# Full run across all 5 leagues (fetch -> lineups -> player model -> backtest -> export -> build HTML)
python run.py

# Run for specific league(s) only
python run.py EPL Serie_A

# Recompute models and rebuild HTML using existing cached data on disk (no network requests)
python run.py --no-fetch

# Fast lineup/injury refresh (~1 min per league): fetches latest FotMob lineups and rebuilds calculators
python run.py --lineups
python run.py --lineups EPL
```

### 3. Open or Serve the Web App

After running the pipeline, you have three ways to use the web app:

1. **Unified Multi-League Web App (Offline / Static):**
   Open `build/index.html` directly in your browser (all 5 leagues bundled into one app with instant league switching via tabs or URL hashes like `#EPL`, `#La_liga`), or open individual league files (`build/kalkulator_<league>.html`).
2. **Local Web Server with Live Refresh (`serve.py`):**
   ```bash
   cd code
   python serve.py 8000
   ```
   Opens `http://127.0.0.1:8000` where clicking **↻ Osveži postave** runs `lineups.py` + `export.py` on demand and updates lineups in the browser without reloading.
3. **Static Hosting (Netlify / Cloudflare Pages / Firebase):**
   Deploy the pre-built `build/` folder directly (`netlify.toml` is preconfigured to publish `build/` with zero cloud build minutes):
   ```bash
   npx netlify-cli deploy --prod --dir=build
   ```

Each calculator allows interactive inspection and override of:
- Upcoming fixtures (with predicted or confirmed lineups from FotMob) or custom head-to-head matchups
- Starting XIs and player availability/injury toggles
- Team expectations ($\lambda_h, \lambda_a$), Dixon–Coles $\rho$, lineup elasticities ($\beta_{\text{inj}}, \beta_{\text{rot}}, \beta_d$), and bookmaker margin per market
- Walk-forward backtest metrics (RPS, log-loss, draw calibration, and comparison against the closing market)

---

## Step-by-Step CLI Reference

If you want to run individual stages manually from `code/`:

| Command | Description |
| :--- | :--- |
| `python sources.py [LEAGUE ...]` | Downloads Understat league/match JSONs, `football-data.co.uk` odds CSVs, and FPL bootstrap JSON (EPL). |
| `python lineups.py [LEAGUE ...]` | Fetches upcoming match lineups and unavailability lists from FotMob (next 10 days). |
| `python player_model.py [LEAGUE]` | Computes historical player ratings and writes `../out/<league>/lineup_factors.json`. |
| `python backtest.py [LEAGUE]` | Runs walk-forward evaluation (`2025-26` and `2026-27`) and writes `../out/<league>/backtest.json`. |
| `TODAY=YYYY-MM-DD python export.py [LEAGUE]` | Fits ratings strictly before `TODAY` and writes `../build/data_<league>.json`. |
| `python build.py` | Bundles `web/pricing.js` and `build/data_<league>.json` into `build/index.html` and `build/kalkulator_<league>.html`. |
| `python serve.py [PORT]` | Serves `build/` locally and enables the in-browser **↻ Osveži postave** button. |

---

## Model Calibration & Research Scripts

All hyperparameters in `model_config.json` are backed by walk-forward out-of-sample experiments (`2025-26` selection $\rightarrow$ `2026-27` holdout):

- `python fit_mkt_ratings.py`: Evaluates blending historical closing-market implied $\lambda$ into team rating targets (`mkt_w`).
- `python tune_mkt.py [--save]`: Tunes GLM ridge penalty, half-life decay, and previous-season weight across early (`fit`) and late (`fit_late`) season rounds.
- `python fit_rho.py [--save]`: Fits per-league Dixon–Coles $\rho$ by minimizing exact-score negative log-likelihood.
- `python fit_lineup.py [--save]`: Fits centered pre-match injury ($\beta_{\text{inj}}$) and lineup rotation ($\beta_{\text{rot}}$) elasticities using historical FotMob unavailability lists (`fotmob_history.py`).
- `python tune_team.py [LEAGUE ...]`: Grid search over `xg_w`, regularization, decay half-life, and `promo_shift`.
- `python fit_market.py`: Evaluates logarithmic pooling between model probabilities and opening market odds.
- `python fit_elo.py`: Tests whether external Elo ratings (`soccer_rating.py`) and short rest days ($\le 3$ days across all competitions) add orthogonal signal.

