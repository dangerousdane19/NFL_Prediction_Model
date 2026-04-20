"""
Weather ingestion using Open-Meteo (free, no API key required).
Fetches historical archive and forecast weather for NFL game locations.
"""
import logging
import time
from datetime import datetime, timezone

import pandas as pd
import requests

log = logging.getLogger(__name__)

OPEN_METEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_FORECAST = "https://api.open-meteo.com/v1/forecast"

WEATHER_VARS = [
    "temperature_2m",
    "wind_speed_10m",
    "wind_direction_10m",
    "precipitation",
    "snowfall",
    "weather_code",
]

# Team abbreviation → (latitude, longitude, is_dome)
# is_dome=True means retractable or fixed roof — weather irrelevant
TEAM_COORDS = {
    "ARI": (33.5276, -112.2626, True),   # State Farm Stadium
    "ATL": (33.7554, -84.4010, True),    # Mercedes-Benz Stadium
    "BAL": (39.2780, -76.6227, False),   # M&T Bank Stadium
    "BUF": (42.7738, -78.7870, False),   # Highmark Stadium
    "CAR": (35.2258, -80.8528, False),   # Bank of America Stadium
    "CHI": (41.8623, -87.6167, False),   # Soldier Field
    "CIN": (39.0955, -84.5160, False),   # Paycor Stadium
    "CLE": (41.5061, -81.6995, False),   # Cleveland Browns Stadium
    "DAL": (32.7473, -97.0945, True),    # AT&T Stadium
    "DEN": (39.7439, -105.0200, False),  # Empower Field
    "DET": (42.3400, -83.0456, True),    # Ford Field
    "GB":  (44.5013, -88.0622, False),   # Lambeau Field
    "HOU": (29.6847, -95.4107, True),    # NRG Stadium
    "IND": (39.7601, -86.1639, True),    # Lucas Oil Stadium
    "JAX": (30.3239, -81.6373, False),   # EverBank Stadium
    "KC":  (39.0489, -94.4839, False),   # Arrowhead Stadium
    "LAC": (33.9535, -118.3392, True),   # SoFi Stadium (covered)
    "LAR": (33.9535, -118.3392, True),   # SoFi Stadium (covered)
    "LV":  (36.0909, -115.1833, True),   # Allegiant Stadium
    "MIA": (25.9580, -80.2389, False),   # Hard Rock Stadium
    "MIN": (44.9740, -93.2581, True),    # U.S. Bank Stadium
    "NE":  (42.0909, -71.2643, False),   # Gillette Stadium
    "NO":  (29.9511, -90.0812, True),    # Caesars Superdome
    "NYG": (40.8135, -74.0744, False),   # MetLife Stadium
    "NYJ": (40.8135, -74.0744, False),   # MetLife Stadium
    "PHI": (39.9007, -75.1675, False),   # Lincoln Financial Field
    "PIT": (40.4468, -80.0158, False),   # Acrisure Stadium
    "SEA": (47.5952, -122.3316, False),  # Lumen Field
    "SF":  (37.4033, -121.9694, False),  # Levi's Stadium
    "TB":  (27.9759, -82.5033, False),   # Raymond James Stadium
    "TEN": (36.1665, -86.7713, False),   # Nissan Stadium
    "WAS": (38.9078, -76.8645, False),   # Northwest Stadium
}


def get_team_coords(team_abbr: str) -> tuple:
    """Returns (lat, lon, is_dome) for a home team abbreviation."""
    return TEAM_COORDS.get(team_abbr, (None, None, False))


def fetch_game_weather(lat: float, lon: float, game_date: str, hour: int, is_dome: bool) -> dict:
    """
    Fetch weather for a single game from Open-Meteo.
    game_date: 'YYYY-MM-DD'
    hour: kickoff hour in local time (0-23)
    Returns dict with weather variables. All None for dome games.
    """
    null_weather = {v: None for v in WEATHER_VARS}
    null_weather["is_dome"] = 1

    if is_dome or lat is None or lon is None:
        return null_weather

    try:
        game_dt = datetime.strptime(game_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        is_past = game_dt < datetime.now(timezone.utc)
        base_url = OPEN_METEO_ARCHIVE if is_past else OPEN_METEO_FORECAST

        r = requests.get(
            base_url,
            params={
                "latitude": lat,
                "longitude": lon,
                "start_date": game_date,
                "end_date": game_date,
                "hourly": ",".join(WEATHER_VARS),
                "wind_speed_unit": "mph",
                "temperature_unit": "fahrenheit",
                "timezone": "America/New_York",
            },
            timeout=15,
        )
        r.raise_for_status()
        hourly = r.json().get("hourly", {})
        idx = min(max(int(hour), 0), 23)
        result = {v: hourly.get(v, [None] * 24)[idx] for v in WEATHER_VARS}
        result["is_dome"] = 0
        return result

    except Exception as e:
        log.warning(f"Weather fetch failed ({lat},{lon}) {game_date} h={hour}: {e}")
        return null_weather


def backfill_weather(conn) -> pd.DataFrame:
    """
    Fetch and return weather rows for all games in betting_odds that are
    not yet present in the weather table. Caller is responsible for upserting.
    """
    try:
        existing = pd.read_sql(
            "SELECT HomeTeamName, year, dayofyear FROM weather", conn
        )
    except Exception:
        existing = pd.DataFrame(columns=["HomeTeamName", "year", "dayofyear"])

    all_games = pd.read_sql(
        "SELECT HomeTeamName, year, dayofyear, hour FROM betting_odds "
        "WHERE year IS NOT NULL AND dayofyear IS NOT NULL",
        conn,
    )

    if not existing.empty:
        merged = all_games.merge(
            existing.assign(_exists=1),
            on=["HomeTeamName", "year", "dayofyear"],
            how="left",
        )
        games = merged[merged["_exists"].isna()].drop(columns=["_exists"]).reset_index(drop=True)
    else:
        games = all_games

    if games.empty:
        log.info("No games need weather backfill")
        return pd.DataFrame()

    log.info(f"Fetching weather for {len(games)} games via Open-Meteo...")
    rows = []

    for i, row in games.iterrows():
        team = row["HomeTeamName"]
        lat, lon, is_dome = get_team_coords(team)

        try:
            game_date = (
                pd.Timestamp(year=int(row["year"]), month=1, day=1)
                + pd.Timedelta(days=int(row["dayofyear"]) - 1)
            ).strftime("%Y-%m-%d")
        except Exception:
            continue

        hour = int(row["hour"]) if pd.notna(row.get("hour")) else 13
        weather = fetch_game_weather(lat, lon, game_date, hour, is_dome)
        weather["HomeTeamName"] = team
        weather["year"] = int(row["year"])
        weather["dayofyear"] = int(row["dayofyear"])
        rows.append(weather)

        # Polite rate-limiting — Open-Meteo has no hard limit but we're nice
        if (i + 1) % 100 == 0:
            log.info(f"  {i + 1}/{len(games)} processed")
            time.sleep(1)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df.rename(
        columns={
            "temperature_2m": "temperature",
            "wind_speed_10m": "wind_speed",
            "wind_direction_10m": "wind_direction",
        },
        inplace=True,
    )
    keep = [
        "HomeTeamName", "year", "dayofyear",
        "temperature", "wind_speed", "wind_direction",
        "precipitation", "snowfall", "weather_code", "is_dome",
    ]
    df = df[[c for c in keep if c in df.columns]]
    log.info(f"Weather backfill complete: {len(df)} rows")
    return df
