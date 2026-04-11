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
    "DrawMoneyLine", "SportsbookId", "Team", "AwayTeamScore", "Date",
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


def prepare_model_data(NFLdataset: pd.DataFrame) -> tuple:
    """
    Returns (NFLmodeldata, NFLmodeldata1).
    NFLmodeldata  — used for regression models (scores)
    NFLmodeldata1 — used for classification models (cover/outcome)
    """
    # --- Regression dataset ---
    drop_reg = [c for c in DROP_COLS_REGRESSION if c in NFLdataset.columns]
    NFLmodeldata = NFLdataset.drop(columns=drop_reg).copy()
    # Keep only numeric columns for fillna
    num_cols = NFLmodeldata.select_dtypes(include="number").columns
    NFLmodeldata[num_cols] = NFLmodeldata[num_cols].fillna(NFLmodeldata[num_cols].mean())

    # --- Classification dataset ---
    NFLmodeldata1 = NFLdataset.copy()

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
    NFLmodeldata1[num_cols1] = NFLmodeldata1[num_cols1].fillna(NFLmodeldata1[num_cols1].mean())

    log.info(f"NFLmodeldata: {len(NFLmodeldata)} rows × {len(NFLmodeldata.columns)} cols")
    log.info(f"NFLmodeldata1: {len(NFLmodeldata1)} rows × {len(NFLmodeldata1.columns)} cols")
    return NFLmodeldata, NFLmodeldata1
