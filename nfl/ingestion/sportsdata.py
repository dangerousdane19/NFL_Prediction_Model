"""
SportsData.io API ingestion:
  - fetch_stadiums()
  - fetch_team_game_stats(seasons)
  - fetch_game_odds(seasons)
"""
import logging
import time

import pandas as pd
import requests

from nfl import config

log = logging.getLogger(__name__)

_BASE = "https://api.sportsdata.io/api/nfl/odds/json"

STADIUM_RENAMES = {
    "Paul Brown Stadium": "Paycor Stadium",
    "Mercedes-Benz Superdome": "Caesars Superdome",
    "Heinz Field": "Acrisure Stadium",
    "CenturyLink Field": "Lumen Field",
    "Bills Stadium": "Highmark Stadium",
}


def _headers() -> dict:
    return {"Ocp-Apim-Subscription-Key": config.SPORTSDATA_API_KEY}


def _get(url: str) -> list:
    try:
        r = requests.get(url, headers=_headers(), timeout=15)
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        log.warning(f"Request failed: {url} — {e}")
    return []


def fetch_stadiums() -> pd.DataFrame:
    data = _get(f"{_BASE}/Stadiums")
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    df.drop(
        columns=[c for c in ["City", "State", "Country", "Capacity", "GeoLat", "GeoLong"] if c in df.columns],
        inplace=True,
    )
    df["PlayingSurface"] = df["PlayingSurface"].replace({"Artificial": 1, "Dome": 2, "Grass": 3})
    df["Type"] = df["Type"].replace({"Dome": 1, "Outdoor": 2, "RetractableDome": 3})
    df.rename(columns={"Name": "Stadium"}, inplace=True)
    log.info(f"Fetched {len(df)} stadiums")
    return df


def fetch_team_game_stats(seasons: list = None) -> pd.DataFrame:
    if seasons is None:
        seasons = config.TRAINING_SEASONS
    frames = []

    for season in seasons:
        # Regular season
        weeks = 17 if season == 2020 else config.REG_WEEKS
        for week in range(1, weeks + 1):
            data = _get(f"{_BASE}/TeamGameStats/{season}REG/{week}")
            if data:
                frames.append(pd.DataFrame(data))
            time.sleep(0.1)

        # Postseason
        for week in range(1, config.POST_WEEKS + 1):
            data = _get(f"{_BASE}/TeamGameStats/{season}POST/{week}")
            if data:
                frames.append(pd.DataFrame(data))
            time.sleep(0.1)

    if not frames:
        return pd.DataFrame()

    teamstats = pd.concat(frames, ignore_index=True)
    teamstats.dropna(subset=["OpponentScore", "TotalScore"], inplace=True)
    teamstats["Stadium"] = teamstats["Stadium"].replace(STADIUM_RENAMES)
    teamstats["Date"] = pd.to_datetime(teamstats["Date"])
    teamstats["month"] = teamstats["Date"].dt.month
    teamstats["year"] = teamstats["Date"].dt.year
    teamstats["dayofyear"] = teamstats["Date"].dt.dayofyear
    teamstats["weekofyear"] = teamstats["Date"].dt.isocalendar().week.astype(int)
    log.info(f"Fetched {len(teamstats)} team game stat rows")
    return teamstats


def _parse_pregame_odds(row_str: str) -> dict:
    """Parse a PregameOdds string representation into a dict of betting fields."""
    parts = row_str.split(",")

    def _val(idx: int) -> str:
        try:
            return parts[idx].split(":", 1)[1].strip().strip("'\"")
        except (IndexError, ValueError):
            return None

    return {
        "GameOddId": _val(0),
        "Sportsbook": _val(1),
        "ScoreId": _val(2),
        "Created": _val(3),
        "Updated": _val(4),
        "HomeMoneyLine": _val(5),
        "AwayMoneyLine": _val(6),
        "DrawMoneyLine": _val(7),
        "HomePointSpread": _val(8),
        "AwayPointSpread": _val(9),
        "HomePointSpreadPayout": _val(10),
        "AwayPointSpreadPayout": _val(11),
        "OverUnder": _val(12),
        "OverPayout": _val(13),
        "UnderPayout": _val(14),
        "SportsbookId": _val(15),
    }


def _extract_first_odds(pregame_odds) -> dict:
    """Extract betting fields from the first valid PregameOdds entry."""
    if not pregame_odds or not isinstance(pregame_odds, list):
        return {}
    for entry in pregame_odds:
        if isinstance(entry, dict) and entry.get("HomeMoneyLine") is not None:
            return {
                "GameOddId": entry.get("GameOddId"),
                "Sportsbook": entry.get("Sportsbook"),
                "SportsbookId": entry.get("SportsbookId"),
                "HomeMoneyLine": entry.get("HomeMoneyLine"),
                "AwayMoneyLine": entry.get("AwayMoneyLine"),
                "DrawMoneyLine": entry.get("DrawMoneyLine"),
                "HomePointSpread": entry.get("HomePointSpread"),
                "AwayPointSpread": entry.get("AwayPointSpread"),
                "HomePointSpreadPayout": entry.get("HomePointSpreadPayout"),
                "AwayPointSpreadPayout": entry.get("AwayPointSpreadPayout"),
                "OverUnder": entry.get("OverUnder"),
                "OverPayout": entry.get("OverPayout"),
                "UnderPayout": entry.get("UnderPayout"),
            }
    return {}


def fetch_game_odds(seasons: list = None) -> pd.DataFrame:
    if seasons is None:
        seasons = config.TRAINING_SEASONS
    frames = []

    for season in seasons:
        weeks = 17 if season == 2020 else config.REG_WEEKS
        for week in range(1, weeks + 1):
            data = _get(f"{_BASE}/GameOddsByWeek/{season}/{week}")
            if data:
                frames.append(pd.DataFrame(data))
            time.sleep(0.1)

        for week in range(1, config.POST_WEEKS + 1):
            data = _get(f"{_BASE}/GameOddsByWeek/{season}POST/{week}")
            if data:
                frames.append(pd.DataFrame(data))
            time.sleep(0.1)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)

    # Extract betting fields from PregameOdds (list of dicts from API)
    odds_expanded = df["PregameOdds"].apply(_extract_first_odds).apply(pd.Series)
    df = pd.concat([df.drop(columns=["PregameOdds"], errors="ignore"), odds_expanded], axis=1)

    # Parse datetime features
    df["DateTime"] = pd.to_datetime(df["DateTime"])
    df["hour"] = df["DateTime"].dt.hour
    df["dayofweek"] = df["DateTime"].dt.dayofweek
    df["quarter"] = df["DateTime"].dt.quarter
    df["month"] = df["DateTime"].dt.month
    df["year"] = df["DateTime"].dt.year
    df["dayofyear"] = df["DateTime"].dt.dayofyear
    df["dayofmonth"] = df["DateTime"].dt.day
    df["weekofyear"] = df["DateTime"].dt.isocalendar().week.astype(int)

    # Drop future/incomplete games
    df.dropna(subset=["HomeTeamScore", "AwayTeamScore"], inplace=True)

    # Cast betting columns to float
    float_cols = [
        "HomeMoneyLine", "AwayMoneyLine", "HomePointSpread", "AwayPointSpread",
        "HomePointSpreadPayout", "AwayPointSpreadPayout", "OverUnder",
        "OverPayout", "UnderPayout",
    ]
    for col in float_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["TotalScore"] = df["HomeTeamScore"] + df["AwayTeamScore"]
    log.info(f"Fetched {len(df)} betting odds rows")
    return df
