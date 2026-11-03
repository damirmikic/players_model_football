"""League registry. Adding a league = one entry here (Understat code + football-data.co.uk code + FotMob id).

Understat covers only these five (plus RFPL); football-data.co.uk covers many more leagues but without
player data, so a league outside Understat would need a goals-only path (no lineup model)."""

LEAGUES = {
    "EPL":        {"name": "Premier League", "understat": "EPL",        "fd": "E0",  "fpl": True, "fotmob": 47},
    "La_liga":    {"name": "La Liga",        "understat": "La_liga",    "fd": "SP1", "fpl": False, "fotmob": 87},
    "Bundesliga": {"name": "Bundesliga",     "understat": "Bundesliga", "fd": "D1",  "fpl": False, "fotmob": 54},
    "Serie_A":    {"name": "Serie A",        "understat": "Serie_A",    "fd": "I1",  "fpl": False, "fotmob": 55},
    "Ligue_1":    {"name": "Ligue 1",        "understat": "Ligue_1",    "fd": "F1",  "fpl": False, "fotmob": 53},
}

SEASONS = ["2024-25", "2025-26", "2026-27"]
CURRENT = SEASONS[-1]


def understat_year(season):      # "2026-27" -> 2026
    return int(season[:4])


def fd_code(season):             # "2026-27" -> "2627"
    return season[2:4] + season[5:7]
