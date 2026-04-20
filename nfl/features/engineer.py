"""
Feature engineering:
  - Drop unneeded columns (from notebook Cells 44 and 128)
  - Add classification targets: HomeTeamCover, AwayTeamCover, BetOutcome (Cells 125-127)
  - fillna with column means
"""
import logging

import pandas as pd

log = logging.getLogger(__name__)

# Columns dropped from NFLdataset before modelling (Cell 44)
DROP_COLS_REGRESSION = [
    "Season_y", "ScoreId", "Day", "HomeRotationNumber", "AwayRotationNumber",
    "PregameOdds", "GameOddId", "Sportsbook", "Created", "Updated",
    "DrawMoneyLine", "SportsbookId", "Team", "Date",
    "DayOfWeek", "GameKey", "HomeOrAway", "month_y", "Opponent",
    "OpponentScoreOvertime", "OpponentScoreQuarter1", "OpponentScoreQuarter2",
    "OpponentScoreQuarter3", "OpponentScoreQuarter4", "Score", "ScoreID",
    "ScoreOvertime", "ScoreQuarter1", "ScoreQuarter2", "ScoreQuarter3",
    "ScoreQuarter4", "SeasonType_y", "Stadium", "Status", "TeamGameID",
    "TotalScore_y", "Week_y", "OpponentTimeOfPossession", "TimeOfPossession",
    "OpponentTouchdowns", "OpponentFieldGoalsMade", "FieldGoalsMade", "Touchdowns",
    "ExtraPointKickingAttempts", "ExtraPointKickingConversions",
    "ExtraPointsHadBlocked", "ExtraPointPassingAttempts",
    "ExtraPointPassingConversions", "ExtraPointRushingAttempts",
    "ExtraPointRushingConversions", "OpponentExtraPointKickingAttempts",
    "OpponentExtraPointKickingConversions", "OpponentExtraPointsHadBlocked",
    "OpponentExtraPointPassingAttempts", "OpponentExtraPointPassingConversions",
    "OpponentExtraPointRushingAttempts", "OpponentExtraPointRushingConversions",
    "RushingTouchdowns", "PassingTouchdowns", "OpponentRushingTouchdowns",
    "OpponentPassingTouchdowns", "RedZoneConversions", "OpponentRedZoneConversions",
    "Team_y", "Team_x",
]

# Additional columns dropped for classification dataset (Cell 128 mirrors Cell 44)
DROP_COLS_CLASSIFICATION = DROP_COLS_REGRESSION.copy()


def _home_cover(row) -> str:
    try:
        spread = float(row["HomePointSpread"])
        diff = float(row["HomeTeamScore"]) - float(row["AwayTeamScore"])
        if spread < 0 and diff > abs(spread):
            return "Cover"
        if spread > 0 and float(row["AwayTeamScore"]) - float(row["HomeTeamScore"]) < spread:
            return "Cover"
        if abs(diff) == abs(spread):
            return "Push"
        return "Lose"
    except (TypeError, ValueError):
        return "Lose"


def _away_cover(row) -> str:
    try:
        spread = float(row["AwayPointSpread"])
        diff = float(row["AwayTeamScore"]) - float(row["HomeTeamScore"])
        if spread < 0 and diff > abs(spread):
            return "Cover"
        if spread > 0 and float(row["HomeTeamScore"]) - float(row["AwayTeamScore"]) < spread:
            return "Cover"
        if abs(diff) == abs(spread):
            return "Push"
        return "Lose"
    except (TypeError, ValueError):
        return "Lose"


def _bet_outcome(row) -> str:
    try:
        total = float(row.get("TotalScore_x") or row.get("TotalScore") or 0)
        ou = float(row["OverUnder"])
        if total > ou:
            return "Cover"
        if total == ou:
            return "Push"
        return "Lose"
    except (TypeError, ValueError):
        return "Lose"


def _add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer weather flags. Dome games get neutral indoor values before flagging."""
    weather_present = "temperature" in df.columns or "wind_speed" in df.columns

    if not weather_present:
        return df

    # Fill dome games with neutral indoor conditions before computing flags
    if "is_dome" in df.columns:
        dome_mask = df["is_dome"] == 1
        for col, val in [("temperature", 72.0), ("wind_speed", 0.0),
                         ("wind_direction", 0.0), ("precipitation", 0.0),
                         ("snowfall", 0.0), ("weather_code", 0)]:
            if col in df.columns:
                df.loc[dome_mask, col] = df.loc[dome_mask, col].fillna(val)

    # Fill remaining NaNs (outdoor games with missing data) with column means
    for col in ["temperature", "wind_speed", "precipitation", "snowfall"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mean())

    # Engineered flags
    if "wind_speed" in df.columns:
        df["high_wind"] = (df["wind_speed"] > 15).astype(int)
    if "precipitation" in df.columns:
        df["is_precipitation"] = (df["precipitation"] > 0).astype(int)
    if "temperature" in df.columns:
        df["freezing"] = (df["temperature"] < 32).astype(int)

    # Combined bad-weather index (0–3)
    wind_flag = df.get("high_wind", pd.Series(0, index=df.index))
    precip_flag = df.get("is_precipitation", pd.Series(0, index=df.index))
    freeze_flag = df.get("freezing", pd.Series(0, index=df.index))
    df["bad_weather_index"] = wind_flag + precip_flag + freeze_flag

    if "is_dome" not in df.columns:
        df["is_dome"] = 0

    return df


def prepare_model_data(NFLdataset: pd.DataFrame) -> tuple:
    """
    Returns (NFLmodeldata, NFLmodeldata1).
    NFLmodeldata  — used for regression models (scores)
    NFLmodeldata1 — used for classification models (cover/outcome)
    """
    # --- Regression dataset ---
    drop_reg = [c for c in DROP_COLS_REGRESSION if c in NFLdataset.columns]
    NFLmodeldata = NFLdataset.drop(columns=drop_reg).copy()
    NFLmodeldata = _add_weather_features(NFLmodeldata)
    # Keep only numeric columns for fillna
    num_cols = NFLmodeldata.select_dtypes(include="number").columns
    NFLmodeldata[num_cols] = NFLmodeldata[num_cols].fillna(NFLmodeldata[num_cols].mean()).fillna(0)

    # --- Classification dataset ---
    NFLmodeldata1 = NFLdataset.copy()
    NFLmodeldata1 = _add_weather_features(NFLmodeldata1)

    # Add cover targets (Cells 125-127)
    NFLmodeldata1["HomeTeamCover"] = NFLmodeldata1.apply(_home_cover, axis=1)
    NFLmodeldata1["AwayTeamCover"] = NFLmodeldata1.apply(_away_cover, axis=1)
    NFLmodeldata1["BetOutcome"] = NFLmodeldata1.apply(_bet_outcome, axis=1)

    # Encode: Cover=0, Push=1, Lose=1
    for col in ["HomeTeamCover", "AwayTeamCover", "BetOutcome"]:
        NFLmodeldata1[col] = NFLmodeldata1[col].replace({"Cover": 0, "Push": 1, "Lose": 1})

    drop_cls = [c for c in DROP_COLS_CLASSIFICATION if c in NFLmodeldata1.columns]
    NFLmodeldata1.drop(columns=drop_cls, inplace=True)
    num_cols1 = NFLmodeldata1.select_dtypes(include="number").columns
    NFLmodeldata1[num_cols1] = NFLmodeldata1[num_cols1].fillna(NFLmodeldata1[num_cols1].mean()).fillna(0)

    log.info(f"NFLmodeldata: {len(NFLmodeldata)} rows × {len(NFLmodeldata.columns)} cols")
    log.info(f"NFLmodeldata1: {len(NFLmodeldata1)} rows × {len(NFLmodeldata1.columns)} cols")
    return NFLmodeldata, NFLmodeldata1
