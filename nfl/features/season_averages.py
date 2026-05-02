"""
Tiered stat vector assembly for predicting future games.

Strategy (in priority order):
  Tier 1 — rolling last-5-game average for current season
  Tier 2 — full current-season average (if <5 games played)
  Tier 3 — prior-season average (Week 1, no current-season data)
"""
import logging

import pandas as pd

log = logging.getLogger(__name__)

STAT_COLS = [
    "PassingYards", "RushingYards", "Penalties", "PenaltyYards",
    "Turnovers", "ThirdDownPercentage", "RedZonePercentage",
    "FirstDowns", "FourthDownPercentage",
]


def _load_team_stats(conn) -> pd.DataFrame:
    try:
        return pd.read_sql("SELECT * FROM team_game_stats", conn)
    except Exception:
        return pd.DataFrame()


def compute_rolling_averages(team: str, season: int, n: int = 5, conn=None) -> dict:
    """Returns mean of last-n-games stats for a team in a season."""
    if conn is None:
        return {}
    try:
        df = pd.read_sql(
            "SELECT * FROM team_game_stats WHERE Team=? AND Season=? ORDER BY dayofyear DESC LIMIT ?",
            conn,
            params=(team, season, n),
        )
        if df.empty:
            return {}
        result = {}
        for col in STAT_COLS:
            if col in df.columns:
                result[col] = df[col].mean()
        result["_strategy"] = f"rolling_{min(len(df), n)}_games"
        result["_game_count"] = len(df)
        return result
    except Exception as e:
        log.warning(f"Rolling averages failed for {team} {season}: {e}")
        return {}


def compute_season_averages(team: str, season: int, conn=None) -> dict:
    """Returns full season mean stats for a team."""
    if conn is None:
        return {}
    try:
        df = pd.read_sql(
            "SELECT * FROM team_game_stats WHERE Team=? AND Season=?",
            conn,
            params=(team, season),
        )
        if df.empty:
            return {}
        result = {}
        for col in STAT_COLS:
            if col in df.columns:
                result[col] = df[col].mean()
        result["_strategy"] = "season_avg"
        result["_game_count"] = len(df)
        return result
    except Exception as e:
        log.warning(f"Season averages failed for {team} {season}: {e}")
        return {}


def get_team_stat_vector(
    home_team: str,
    season: int,
    conn=None,
    strategy: str = "rolling_5",
) -> tuple:
    """
    Returns (stat_dict, strategy_used) for the home team.

    stat_dict keys match TeamGameStats column names.
    strategy_used is one of: 'rolling_5_games', 'season_avg', 'prior_season_avg', 'empty'
    """
    stat_dict = {}
    strategy_used = "empty"

    if strategy == "rolling_5" or strategy == "auto":
        # Tier 1: rolling 5-game average
        stats = compute_rolling_averages(home_team, season, n=5, conn=conn)
        if stats:
            stat_dict = stats
            strategy_used = stats.get("_strategy", "rolling_5_games")
        else:
            # Tier 2: full current-season average
            stats = compute_season_averages(home_team, season, conn=conn)
            if stats:
                stat_dict = stats
                strategy_used = "season_avg"
            else:
                # Tier 3: prior season average
                stats = compute_season_averages(home_team, season - 1, conn=conn)
                if stats:
                    stat_dict = stats
                    strategy_used = "prior_season_avg"

    # Strip internal metadata keys
    stat_dict = {k: v for k, v in stat_dict.items() if not k.startswith("_")}
    return stat_dict, strategy_used


def get_away_team_stat_vector(
    away_team: str,
    season: int,
    conn=None,
) -> dict:
    """
    Returns stat dict for the away team with away_ prefix, using the same
    tiered strategy as get_team_stat_vector.
    Keys: away_PassingYards, away_RushingYards, etc.
    """
    stat_dict, _ = get_team_stat_vector(away_team, season, conn=conn)
    return {f"away_{k}": v for k, v in stat_dict.items()}
