"""
Faithful replication of notebook Cell 42 merge chain.
Joins all raw DataFrames into a single NFLdataset.
"""
import logging

import pandas as pd

log = logging.getLogger(__name__)


def build_nfl_dataset(
    teamstats: pd.DataFrame,
    stadiums: pd.DataFrame,
    betting_odds: pd.DataFrame,
    referee_assignments: pd.DataFrame,
    elo_ratings: pd.DataFrame,
    google_trends: pd.DataFrame,
    weather_df: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Replicates Cell 42 merge chain:
      1. teamstats + stadiums → teamstatswstadium
      2. betting_odds + referees → df
      3. df + elo → df
      4. df + teamstatswstadium → df  (home team stats)
      5. df + google_trends (home) → df
      6. df + google_trends (away) → df
      7. df + weather → NFLdataset
    """
    if betting_odds.empty:
        log.warning("betting_odds is empty — cannot build dataset")
        return pd.DataFrame()

    # Step 1: Attach stadium metadata to team stats
    if not stadiums.empty and not teamstats.empty:
        teamstatswstadium = pd.merge(
            teamstats,
            stadiums[["StadiumID", "Stadium", "PlayingSurface", "Type"]],
            on="Stadium",
            how="left",
        )
    else:
        teamstatswstadium = teamstats.copy()

    # Step 2: Attach referee data to betting odds
    df = betting_odds.copy()
    if not referee_assignments.empty:
        df = pd.merge(
            df,
            referee_assignments[["HomeTeamName", "month", "year", "dayofyear", "Referee"]],
            on=["HomeTeamName", "month", "year", "dayofyear"],
            how="left",
        )

    # Step 3: Attach ELO ratings
    if not elo_ratings.empty:
        df = pd.merge(
            df,
            elo_ratings[["HomeTeamName", "month", "year", "dayofyear",
                          "elo1_pre", "elo2_pre", "qbelo1_pre", "qbelo2_pre", "neutral"]],
            on=["HomeTeamName", "month", "year", "dayofyear"],
            how="left",
        )

    # Step 4a: Attach home team game stats + stadium
    if not teamstatswstadium.empty:
        df = pd.merge(
            df,
            teamstatswstadium,
            left_on=["HomeTeamName", "year", "dayofyear"],
            right_on=["Team", "year", "dayofyear"],
            how="left",
        )

    # Step 4b: Attach away team game stats (prefixed away_)
    _AWAY_STAT_COLS = [
        "PassingYards", "RushingYards", "Penalties", "PenaltyYards",
        "Turnovers", "ThirdDownPercentage", "RedZonePercentage",
        "FirstDowns", "FourthDownPercentage",
    ]
    if not teamstats.empty:
        available_away = ["Team", "year", "dayofyear"] + [
            c for c in _AWAY_STAT_COLS if c in teamstats.columns
        ]
        away_stats = teamstats[available_away].copy()
        away_stats.rename(
            columns={c: f"away_{c}" for c in _AWAY_STAT_COLS if c in away_stats.columns},
            inplace=True,
        )
        df = pd.merge(
            df,
            away_stats,
            left_on=["AwayTeamName", "year", "dayofyear"],
            right_on=["Team", "year", "dayofyear"],
            how="left",
            suffixes=("", "_away_drop"),
        )
        # Drop the redundant Team key column from the away join
        df.drop(columns=[c for c in df.columns if c.endswith("_away_drop")], inplace=True)
        if "Team_y" in df.columns:
            df.drop(columns=["Team_y"], inplace=True)

    # Steps 5 & 6: Attach Google Trends (home + away)
    trends_available = not google_trends.empty
    if trends_available:
        df = pd.merge(
            df,
            google_trends[["Team", "year", "weekofyear", "HomeTeamGoogleTrend"]],
            left_on=["HomeTeamId", "year", "weekofyear"],
            right_on=["Team", "year", "weekofyear"],
            how="left",
        )
        NFLdataset = pd.merge(
            df,
            google_trends[["Team", "year", "weekofyear", "AwayTeamGoogleTrend"]],
            left_on=["AwayTeamId", "year", "weekofyear"],
            right_on=["Team", "year", "weekofyear"],
            how="left",
        )
    else:
        NFLdataset = df.copy()

    # Fill missing trend columns with 0
    for col in ["HomeTeamGoogleTrend", "AwayTeamGoogleTrend"]:
        if col not in NFLdataset.columns:
            NFLdataset[col] = 0
        else:
            NFLdataset[col].fillna(0, inplace=True)

    # Step 7: Attach weather data
    if weather_df is not None and not weather_df.empty:
        weather_cols = [
            "HomeTeamName", "year", "dayofyear",
            "temperature", "wind_speed", "wind_direction",
            "precipitation", "snowfall", "weather_code", "is_dome",
        ]
        available = [c for c in weather_cols if c in weather_df.columns]
        NFLdataset = pd.merge(
            NFLdataset,
            weather_df[available],
            on=["HomeTeamName", "year", "dayofyear"],
            how="left",
        )

    log.info(f"Built NFLdataset: {len(NFLdataset)} rows, {len(NFLdataset.columns)} columns")
    return NFLdataset
