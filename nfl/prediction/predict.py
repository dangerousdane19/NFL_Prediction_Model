"""
Inference pipeline.
predict_game(game_inputs, conn) → full result dict with all 6 model outputs.
"""
import logging
from datetime import date

import numpy as np
import pandas as pd

from nfl import database
from nfl.features.season_averages import get_team_stat_vector
from nfl.ingestion.elo_calculator import get_current_elo
from nfl.ingestion.weather import fetch_game_weather, get_team_coords
from nfl.training.train import load_models

log = logging.getLogger(__name__)


def build_feature_vector(
    game_inputs: dict,
    stat_vector: dict,
    feature_columns: list,
    weather: dict = None,
) -> pd.DataFrame:
    """
    Construct a single-row DataFrame matching the training feature set.

    game_inputs keys (user-provided):
      home_team, away_team, game_date, season, week,
      over_under, home_point_spread, away_point_spread,
      home_money_line, away_money_line,
      stadium_id, playing_surface, stadium_type,
      neutral, referee, elo1_pre, elo2_pre, qbelo1_pre, qbelo2_pre,
      home_google_trend, away_google_trend

    stat_vector keys: TeamGameStats column averages for the home team
    weather: dict from fetch_game_weather (optional)
    feature_columns: ordered list saved at training time
    """
    gd = game_inputs.get("game_date") or date.today()
    if isinstance(gd, str):
        gd = pd.to_datetime(gd)

    row = {
        # Time features (from betting_odds DateTime)
        "hour": gd.hour if hasattr(gd, "hour") else 13,
        "dayofweek": gd.weekday() if hasattr(gd, "weekday") else 6,
        "quarter": (gd.month - 1) // 3 + 1,
        "month": gd.month,
        "year": gd.year,
        "dayofyear": gd.timetuple().tm_yday,
        "dayofmonth": gd.day,
        "weekofyear": int(gd.isocalendar()[1]),
        # Vegas lines
        "OverUnder": game_inputs.get("over_under", 45.0),
        "HomePointSpread": game_inputs.get("home_point_spread", -3.0),
        "AwayPointSpread": game_inputs.get("away_point_spread", 3.0),
        "HomeMoneyLine": game_inputs.get("home_money_line", -150.0),
        "AwayMoneyLine": game_inputs.get("away_money_line", 130.0),
        "HomePointSpreadPayout": game_inputs.get("home_spread_payout", -110.0),
        "AwayPointSpreadPayout": game_inputs.get("away_spread_payout", -110.0),
        "OverPayout": game_inputs.get("over_payout", -110.0),
        "UnderPayout": game_inputs.get("under_payout", -110.0),
        # Stadium
        "PlayingSurface": game_inputs.get("playing_surface", 3),
        "Type": game_inputs.get("stadium_type", 2),
        # ELO
        "elo1_pre": game_inputs.get("elo1_pre", 1505.0),
        "elo2_pre": game_inputs.get("elo2_pre", 1495.0),
        "qbelo1_pre": game_inputs.get("qbelo1_pre", 1505.0),
        "qbelo2_pre": game_inputs.get("qbelo2_pre", 1495.0),
        "neutral": int(game_inputs.get("neutral", 0)),
        # Referee
        "Referee": game_inputs.get("referee", 0),
        # Google Trends
        "HomeTeamGoogleTrend": game_inputs.get("home_google_trend", 0),
        "AwayTeamGoogleTrend": game_inputs.get("away_google_trend", 0),
        # Season info
        "Season_x": game_inputs.get("season", gd.year),
        "Week_x": game_inputs.get("week", 1),
    }

    # Weather features
    if weather:
        is_dome = int(weather.get("is_dome", 0))
        temp = weather.get("temperature_2m") or weather.get("temperature")
        wind = weather.get("wind_speed_10m") or weather.get("wind_speed")
        precip = weather.get("precipitation")

        # Use neutral indoor values for dome games
        if is_dome:
            temp = temp if temp is not None else 72.0
            wind = 0.0
            precip = 0.0

        row["temperature"] = temp if temp is not None else 55.0
        row["wind_speed"] = wind if wind is not None else 8.0
        row["wind_direction"] = (
            weather.get("wind_direction_10m") or weather.get("wind_direction") or 0.0
        )
        row["precipitation"] = precip if precip is not None else 0.0
        row["snowfall"] = weather.get("snowfall") or 0.0
        row["weather_code"] = weather.get("weather_code") or 0
        row["is_dome"] = is_dome

        # Engineered flags (must match what engineer.py produces at training time)
        row["high_wind"] = int(row["wind_speed"] > 15)
        row["is_precipitation"] = int(row["precipitation"] > 0)
        row["freezing"] = int(row["temperature"] < 32)
        row["bad_weather_index"] = row["high_wind"] + row["is_precipitation"] + row["freezing"]

    # Overlay home-team stat averages
    row.update(stat_vector)

    # Build DataFrame aligned to training feature columns
    df = pd.DataFrame([row])

    # Add any missing columns with 0
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0.0

    # Reorder and keep only training columns
    df = df[feature_columns]
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return df


def predict_game(game_inputs: dict, conn=None) -> dict:
    """
    Main prediction entry point.

    Returns a dict with:
      home_score, away_score, total_score, predicted_winner,
      home_cover (0=Cover 1=Push/Lose), home_cover_proba,
      away_cover, away_cover_proba,
      bet_outcome (0=Over 1=Push/Under), bet_outcome_proba,
      stat_strategy_used, google_trends_used
    """
    models, feature_columns = load_models()

    home_team = game_inputs.get("home_team", "")
    season = game_inputs.get("season", date.today().year)

    # Get home team stat vector
    if conn is None:
        conn = database.get_connection()
        _close_conn = True
    else:
        _close_conn = False

    stat_vector, strategy_used = get_team_stat_vector(home_team, season, conn=conn)

    # Auto-fill ELO from calculator if not provided by caller
    if not game_inputs.get("elo1_pre") or not game_inputs.get("elo2_pre"):
        try:
            current_elo = get_current_elo(conn)
            away_team = game_inputs.get("away_team", "")
            home_elo_data = current_elo.get(home_team, {})
            away_elo_data = current_elo.get(away_team, {})
            game_inputs.setdefault("elo1_pre", home_elo_data.get("elo", 1500.0))
            game_inputs.setdefault("elo2_pre", away_elo_data.get("elo", 1500.0))
            game_inputs.setdefault("qbelo1_pre", home_elo_data.get("qbelo", 1500.0))
            game_inputs.setdefault("qbelo2_pre", away_elo_data.get("qbelo", 1500.0))
        except Exception as e:
            log.warning(f"ELO auto-fill failed: {e}")

    if _close_conn:
        conn.close()

    # Auto-fetch weather from Open-Meteo unless caller already supplied it
    weather = game_inputs.get("weather")
    if weather is None:
        try:
            gd = game_inputs.get("game_date") or date.today()
            if isinstance(gd, str):
                gd = pd.to_datetime(gd).date()
            lat, lon, is_dome = get_team_coords(home_team)
            weather = fetch_game_weather(
                lat, lon,
                gd.strftime("%Y-%m-%d"),
                gd.hour if hasattr(gd, "hour") else 13,
                is_dome,
            )
        except Exception as e:
            log.warning(f"Weather auto-fetch failed: {e}")
            weather = None

    feature_vector = build_feature_vector(game_inputs, stat_vector, feature_columns, weather=weather)

    # Regression predictions
    home_score = float(models["modelhs"].predict(feature_vector)[0])
    away_score = float(models["modelas"].predict(feature_vector)[0])
    total_score = float(models["modelts"].predict(feature_vector)[0])
    predicted_winner = "HOME" if home_score > away_score else "AWAY"

    # Classification predictions
    home_cover = int(models["logreg_homecover"].predict(feature_vector)[0])
    home_cover_proba = float(models["logreg_homecover"].predict_proba(feature_vector)[0][0])

    away_cover = int(models["logreg_awaycover"].predict(feature_vector)[0])
    away_cover_proba = float(models["logreg_awaycover"].predict_proba(feature_vector)[0][0])

    bet_outcome = int(models["logreg_betoutcome"].predict(feature_vector)[0])
    bet_outcome_proba = float(models["logreg_betoutcome"].predict_proba(feature_vector)[0][0])

    google_trends_used = bool(
        game_inputs.get("home_google_trend", 0) != 0
        or game_inputs.get("away_google_trend", 0) != 0
    )

    result = {
        "home_score": round(home_score, 1),
        "away_score": round(away_score, 1),
        "total_score": round(total_score, 1),
        "predicted_winner": predicted_winner,
        "home_cover": home_cover,
        "home_cover_proba": round(home_cover_proba, 3),
        "away_cover": away_cover,
        "away_cover_proba": round(away_cover_proba, 3),
        "bet_outcome": bet_outcome,
        "bet_outcome_proba": round(bet_outcome_proba, 3),
        "stat_strategy_used": strategy_used,
        "google_trends_used": google_trends_used,
        "weather": weather,
    }

    # Log prediction to DB
    try:
        record = {
            "home_team": game_inputs.get("home_team", ""),
            "away_team": game_inputs.get("away_team", ""),
            "season": game_inputs.get("season"),
            "week": game_inputs.get("week"),
            "game_date": str(game_inputs.get("game_date", "")),
            "over_under_input": game_inputs.get("over_under"),
            "home_point_spread_input": game_inputs.get("home_point_spread"),
            "away_point_spread_input": game_inputs.get("away_point_spread"),
            "home_money_line": game_inputs.get("home_money_line"),
            "away_money_line": game_inputs.get("away_money_line"),
            "pred_home_score": result["home_score"],
            "pred_away_score": result["away_score"],
            "pred_total_score": result["total_score"],
            "pred_winner": result["predicted_winner"],
            "pred_home_cover": result["home_cover"],
            "pred_home_cover_proba": result["home_cover_proba"],
            "pred_away_cover": result["away_cover"],
            "pred_away_cover_proba": result["away_cover_proba"],
            "pred_bet_outcome": result["bet_outcome"],
            "pred_bet_outcome_proba": result["bet_outcome_proba"],
            "stat_strategy_used": result["stat_strategy_used"],
            "google_trends_used": int(result["google_trends_used"]),
        }
        database.insert_prediction(record)
    except Exception as e:
        log.warning(f"Failed to log prediction to DB: {e}")

    return result
