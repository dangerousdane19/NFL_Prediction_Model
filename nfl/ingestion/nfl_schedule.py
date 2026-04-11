"""
NFL schedule import via nfl_data_py.
"""
import logging

import pandas as pd

log = logging.getLogger(__name__)


def fetch_schedule(years: list = None) -> pd.DataFrame:
    if years is None:
        from datetime import date
        current_year = date.today().year
        years = list(range(2020, current_year + 1))
    try:
        import nfl_data_py as nfl
        df = nfl.import_schedules(years)
        keep = ["gameday", "season", "home_team", "away_team"]
        keep = [c for c in keep if c in df.columns]
        df = df[keep].copy()
        df.rename(columns={"gameday": "Day", "home_team": "HomeTeamName", "away_team": "AwayTeamName"}, inplace=True)
        df["Day"] = pd.to_datetime(df["Day"], errors="coerce")
        df.dropna(subset=["Day"], inplace=True)
        df["month"] = df["Day"].dt.month
        df["year"] = df["Day"].dt.year
        df["dayofyear"] = df["Day"].dt.dayofyear
        df["weekofyear"] = df["Day"].dt.isocalendar().week.astype(int)
        log.info(f"Loaded {len(df)} schedule rows from nfl_data_py")
        return df
    except Exception as e:
        log.warning(f"nfl_data_py schedule fetch failed: {e}")
        return pd.DataFrame()
