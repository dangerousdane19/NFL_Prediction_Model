"""
ELO rating calculator for NFL teams.

Initialises all 32 teams at 1500 ELO at the start of the earliest season,
then walks completed games chronologically to produce pre-game ELO ratings
consistent with the 538 methodology (K=20, home-field advantage=65 pts).

QB ELO is set equal to team ELO — we don't have QB-level data from this API.
neutral site games get no home-field adjustment.

Results are stored in the elo_ratings table so predictions can look them up.
"""
import logging
from datetime import date

import pandas as pd

log = logging.getLogger(__name__)

# 538 NFL ELO parameters
K = 20
HOME_ADVANTAGE = 65   # ELO points added to home team
STARTING_ELO = 1500
# Mean-reversion between seasons (538 pulls each team 1/3 toward 1505 at season start)
MEAN_REVERSION = 1505
REVERSION_FACTOR = 1 / 3


def _expected(elo_a: float, elo_b: float) -> float:
    return 1 / (1 + 10 ** ((elo_b - elo_a) / 400))


def _new_elo(elo: float, result: float, expected: float) -> float:
    return elo + K * (result - expected)


def calculate_elo_ratings(conn) -> pd.DataFrame:
    """
    Calculate pre-game ELO ratings for all games in betting_odds table.
    Returns a DataFrame ready to upsert into elo_ratings.
    """
    try:
        games = pd.read_sql(
            """
            SELECT HomeTeamName, AwayTeamName, year, month, dayofyear,
                   weekofyear, Season_x as Season, HomeTeamScore, AwayTeamScore
            FROM betting_odds
            WHERE HomeTeamScore IS NOT NULL AND AwayTeamScore IS NOT NULL
            ORDER BY year, dayofyear
            """,
            conn,
        )
    except Exception:
        # Try without Season_x alias
        games = pd.read_sql(
            """
            SELECT HomeTeamName, AwayTeamName, year, month, dayofyear,
                   weekofyear, Season, HomeTeamScore, AwayTeamScore
            FROM betting_odds
            WHERE HomeTeamScore IS NOT NULL AND AwayTeamScore IS NOT NULL
            ORDER BY year, dayofyear
            """,
            conn,
        )

    if games.empty:
        log.warning("No completed games found in betting_odds — skipping ELO calculation")
        return pd.DataFrame()

    # Initialise ELO for all teams
    all_teams = set(games["HomeTeamName"].tolist() + games["AwayTeamName"].tolist())
    elo = {team: float(STARTING_ELO) for team in all_teams}

    rows = []
    current_season = None

    for _, game in games.iterrows():
        home = game["HomeTeamName"]
        away = game["AwayTeamName"]
        season = int(game.get("Season") or game.get("Season_x") or 0)

        # Apply mean reversion at the start of each new season
        if season != current_season:
            if current_season is not None:
                for team in elo:
                    elo[team] = elo[team] + REVERSION_FACTOR * (MEAN_REVERSION - elo[team])
            current_season = season

        home_elo = elo.get(home, STARTING_ELO)
        away_elo = elo.get(away, STARTING_ELO)

        # Record PRE-game ELO for this matchup
        rows.append({
            "HomeTeamName": home,
            "month": int(game["month"]),
            "year": int(game["year"]),
            "dayofyear": int(game["dayofyear"]),
            "elo1_pre": round(home_elo, 1),
            "elo2_pre": round(away_elo, 1),
            "qbelo1_pre": round(home_elo, 1),   # mirrors team ELO
            "qbelo2_pre": round(away_elo, 1),
            "neutral": 0,
        })

        # Update ELO from result
        home_score = float(game["HomeTeamScore"])
        away_score = float(game["AwayTeamScore"])

        if home_score > away_score:
            home_result, away_result = 1.0, 0.0
        elif away_score > home_score:
            home_result, away_result = 0.0, 1.0
        else:
            home_result, away_result = 0.5, 0.5

        # Home team gets +HOME_ADVANTAGE in expected calculation
        home_expected = _expected(home_elo + HOME_ADVANTAGE, away_elo)
        away_expected = _expected(away_elo, home_elo + HOME_ADVANTAGE)

        elo[home] = _new_elo(home_elo, home_result, home_expected)
        elo[away] = _new_elo(away_elo, away_result, away_expected)

    df = pd.DataFrame(rows)
    log.info(f"Calculated ELO for {len(df)} games across {len(all_teams)} teams")
    return df


def get_current_elo(conn) -> dict:
    """
    Returns the most recent ELO for each team — useful for predicting future games.
    Returns dict: {team_name: {"elo": float, "qbelo": float}}
    """
    try:
        games = pd.read_sql(
            """
            SELECT HomeTeamName, AwayTeamName, year, dayofyear,
                   HomeTeamScore, AwayTeamScore,
                   Season_x as Season
            FROM betting_odds
            WHERE HomeTeamScore IS NOT NULL AND AwayTeamScore IS NOT NULL
            ORDER BY year, dayofyear
            """,
            conn,
        )
    except Exception:
        games = pd.read_sql(
            """
            SELECT HomeTeamName, AwayTeamName, year, dayofyear,
                   HomeTeamScore, AwayTeamScore, Season
            FROM betting_odds
            WHERE HomeTeamScore IS NOT NULL AND AwayTeamScore IS NOT NULL
            ORDER BY year, dayofyear
            """,
            conn,
        )

    all_teams = set(games["HomeTeamName"].tolist() + games["AwayTeamName"].tolist())
    elo = {team: float(STARTING_ELO) for team in all_teams}
    current_season = None

    for _, game in games.iterrows():
        home = game["HomeTeamName"]
        away = game["AwayTeamName"]
        season = int(game.get("Season") or game.get("Season_x") or 0)

        if season != current_season:
            if current_season is not None:
                for team in elo:
                    elo[team] = elo[team] + REVERSION_FACTOR * (MEAN_REVERSION - elo[team])
            current_season = season

        home_elo = elo.get(home, STARTING_ELO)
        away_elo = elo.get(away, STARTING_ELO)

        home_score = float(game["HomeTeamScore"])
        away_score = float(game["AwayTeamScore"])

        if home_score > away_score:
            home_result, away_result = 1.0, 0.0
        elif away_score > home_score:
            home_result, away_result = 0.0, 1.0
        else:
            home_result, away_result = 0.5, 0.5

        home_expected = _expected(home_elo + HOME_ADVANTAGE, away_elo)
        away_expected = _expected(away_elo, home_elo + HOME_ADVANTAGE)

        elo[home] = _new_elo(home_elo, home_result, home_expected)
        elo[away] = _new_elo(away_elo, away_result, away_expected)

    return {team: {"elo": round(v, 1), "qbelo": round(v, 1)} for team, v in elo.items()}
