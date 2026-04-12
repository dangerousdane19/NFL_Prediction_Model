"""
FiveThirtyEight ELO ratings loader.
Tries the live URL first; falls back to a committed CSV snapshot.
"""
import logging
import os

import pandas as pd

log = logging.getLogger(__name__)

FTE_URL = "https://projects.fivethirtyeight.com/nfl-api/nfl_elo.csv"
FTE_CACHE = os.path.join(os.path.dirname(__file__), "..", "..", "data", "nfl_elo_cache.csv")

KEEP_COLS = ["date", "season", "team1", "elo1_pre", "elo2_pre", "qbelo1_pre", "qbelo2_pre", "neutral"]

TEAM_RENAMES = {"OAK": "LV", "WSH": "WAS", "STL": "LAR", "SD": "LAC"}


def fetch_elo_ratings() -> pd.DataFrame:
    df = None
    # Try live URL first (sniff columns before committing to usecols)
    try:
        raw = pd.read_csv(FTE_URL, nrows=1, low_memory=False)
        if all(c in raw.columns for c in KEEP_COLS):
            df = pd.read_csv(FTE_URL, usecols=KEEP_COLS, low_memory=False)
            log.info("Loaded ELO from live FiveThirtyEight URL")
        else:
            log.warning(f"FTE URL returned unexpected columns: {list(raw.columns)[:5]} — trying cache")
    except Exception as e:
        log.warning(f"FTE live URL failed ({e}), trying cache...")

    if df is None:
        cache_path = os.path.abspath(FTE_CACHE)
        if os.path.exists(cache_path):
            try:
                df = pd.read_csv(cache_path, usecols=KEEP_COLS, low_memory=False)
                log.info("Loaded ELO from local cache")
            except Exception as e2:
                log.warning(f"Cache also failed: {e2}")

    if df is None or df.empty:
        log.warning("ELO data unavailable — returning empty DataFrame")
        return pd.DataFrame(columns=["HomeTeamName", "month", "year", "dayofyear",
                                     "elo1_pre", "elo2_pre", "qbelo1_pre", "qbelo2_pre", "neutral"])

    df["team1"] = df["team1"].replace(TEAM_RENAMES)
    df.rename(columns={"date": "Day", "team1": "HomeTeamName"}, inplace=True)
    df["Day"] = pd.to_datetime(df["Day"], errors="coerce")
    df.dropna(subset=["Day"], inplace=True)
    df["month"] = df["Day"].dt.month
    df["year"] = df["Day"].dt.year
    df["dayofyear"] = df["Day"].dt.dayofyear

    # Keep only NFL season games (seasons 2000+)
    df = df[df["season"] >= 2000]
    log.info(f"Loaded {len(df)} ELO rows")
    return df[["HomeTeamName", "month", "year", "dayofyear",
               "elo1_pre", "elo2_pre", "qbelo1_pre", "qbelo2_pre", "neutral"]]
