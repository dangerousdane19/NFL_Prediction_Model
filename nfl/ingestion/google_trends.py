"""
Google Trends ingestion via pytrends.
Disabled by default (GOOGLE_TRENDS_ENABLED=false).
Returns an empty DataFrame on failure — callers fill trend features with 0.
"""
import logging
import time

import pandas as pd

from nfl import config

log = logging.getLogger(__name__)

NFL_TEAMS = [
    "Arizona Cardinals", "Atlanta Falcons", "Baltimore Ravens", "Buffalo Bills",
    "Carolina Panthers", "Chicago Bears", "Cincinnati Bengals", "Cleveland Browns",
    "Dallas Cowboys", "Denver Broncos", "Detroit Lions", "Green Bay Packers",
    "Houston Texans", "Indianapolis Colts", "Jacksonville Jaguars",
    "Kansas City Chiefs", "Las Vegas Raiders", "Los Angeles Chargers",
    "Los Angeles Rams", "Miami Dolphins", "Minnesota Vikings",
    "New England Patriots", "New Orleans Saints", "New York Giants",
    "New York Jets", "Philadelphia Eagles", "Pittsburgh Steelers",
    "San Francisco 49ers", "Seattle Seahawks", "Tampa Bay Buccaneers",
    "Tennessee Titans", "Washington Football Team",
]

TEAM_ID_MAP = {
    "Arizona Cardinals": 1, "Atlanta Falcons": 2, "Baltimore Ravens": 3,
    "Buffalo Bills": 4, "Carolina Panthers": 5, "Chicago Bears": 6,
    "Cincinnati Bengals": 7, "Cleveland Browns": 8, "Dallas Cowboys": 9,
    "Denver Broncos": 10, "Detroit Lions": 11, "Green Bay Packers": 12,
    "Houston Texans": 13, "Indianapolis Colts": 14, "Jacksonville Jaguars": 15,
    "Kansas City Chiefs": 16, "Miami Dolphins": 19, "Minnesota Vikings": 20,
    "New England Patriots": 21, "New Orleans Saints": 22, "New York Giants": 23,
    "New York Jets": 24, "Las Vegas Raiders": 25, "Philadelphia Eagles": 26,
    "Pittsburgh Steelers": 28, "Los Angeles Chargers": 29, "Seattle Seahawks": 30,
    "San Francisco 49ers": 31, "Los Angeles Rams": 32, "Tampa Bay Buccaneers": 33,
    "Tennessee Titans": 34, "Washington Football Team": 35,
}

_EMPTY = pd.DataFrame(columns=["Team", "year", "weekofyear", "HomeTeamGoogleTrend", "AwayTeamGoogleTrend"])


def fetch_google_trends() -> pd.DataFrame:
    if not config.GOOGLE_TRENDS_ENABLED:
        log.info("Google Trends disabled — skipping")
        return _EMPTY.copy()
    try:
        from pytrends.request import TrendReq
        pytrend = TrendReq(hl="en-US", tz=360)
        mean_value = None

        for team in NFL_TEAMS:
            try:
                pytrend.build_payload(kw_list=[team], timeframe="today 5-y", geo="US")
                dfz = pytrend.interest_over_time()
                dfz = dfz.resample("W").mean()
                dfz.index = pd.to_datetime(dfz.index)
                if mean_value is None:
                    mean_value = pd.DataFrame(index=dfz.index)
                mean_value[team] = dfz.iloc[:, 0]
                time.sleep(1)
            except Exception as e:
                log.warning(f"Trends failed for {team}: {e}")
                if mean_value is not None:
                    mean_value[team] = 0

        if mean_value is None or mean_value.empty:
            return _EMPTY.copy()

        mean_value["dayofyear"] = mean_value.index.dayofyear
        mean_value["weekofyear"] = mean_value.index.isocalendar().week.astype(int)
        mean_value["year"] = mean_value.index.year
        mean_value.reset_index(drop=True, inplace=True)

        mean_value = pd.melt(
            mean_value,
            id_vars=["dayofyear", "weekofyear", "year"],
            var_name="TeamName",
            value_name="GoogleMean_Value",
        )
        mean_value["Team"] = mean_value["TeamName"].map(TEAM_ID_MAP)
        mean_value = mean_value.rename(columns={"GoogleMean_Value": "AwayTeamGoogleTrend"})
        mean_value["HomeTeamGoogleTrend"] = mean_value["AwayTeamGoogleTrend"]
        return mean_value[["Team", "year", "weekofyear", "HomeTeamGoogleTrend", "AwayTeamGoogleTrend"]]

    except Exception as e:
        log.warning(f"Google Trends fetch failed entirely: {e}")
        return _EMPTY.copy()
