#!/usr/bin/env python3
"""
CLI script: pull all data from APIs and populate the SQLite database.
Run: python scripts/run_ingestion.py
"""
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

from nfl import config, database
from nfl.ingestion import fivethirtyeight, google_trends, referee_scraper, sportsdata
from nfl.ingestion.elo_calculator import calculate_elo_ratings
from nfl.ingestion.weather import backfill_weather


def run():
    log.info("=== NFL Data Ingestion ===")
    database.create_all_tables()

    with database.managed_conn() as conn:

        # 1. Stadiums
        log.info("Fetching stadiums...")
        df_stadiums = sportsdata.fetch_stadiums()
        if not df_stadiums.empty:
            database.upsert_df(df_stadiums[["StadiumID", "Stadium", "PlayingSurface", "Type"]], "stadiums", conn)
            log.info(f"  Stored {len(df_stadiums)} stadiums")

        # 2. Team game stats
        log.info("Fetching team game stats (this takes a few minutes)...")
        df_stats = sportsdata.fetch_team_game_stats(config.TRAINING_SEASONS)
        if not df_stats.empty:
            keep = [
                "TeamGameID", "Team", "Season", "Week", "SeasonType",
                "year", "month", "dayofyear", "weekofyear", "Stadium",
                "PassingYards", "RushingYards", "Penalties", "PenaltyYards",
                "Turnovers", "ThirdDownPercentage", "RedZonePercentage",
                "FirstDowns", "FourthDownPercentage",
            ]
            keep = [c for c in keep if c in df_stats.columns]
            # Add GameDate from Date column
            if "Date" in df_stats.columns:
                df_stats["GameDate"] = df_stats["Date"].astype(str)
                keep.append("GameDate")
            database.upsert_df(df_stats[keep].drop_duplicates(subset=["TeamGameID"]), "team_game_stats", conn)
            log.info(f"  Stored {len(df_stats)} team game stat rows")

            # Precompute season averages
            log.info("Computing season averages...")
            avg_cols = [
                "PassingYards", "RushingYards", "Penalties", "PenaltyYards",
                "Turnovers", "ThirdDownPercentage", "RedZonePercentage",
                "FirstDowns", "FourthDownPercentage",
            ]
            avg_cols = [c for c in avg_cols if c in df_stats.columns]
            agg = {c: "mean" for c in avg_cols}
            season_avgs = df_stats.groupby(["Team", "Season"]).agg(agg).reset_index()
            season_avgs.columns = ["Team", "Season"] + [f"avg_{c}" for c in avg_cols]
            database.upsert_df(season_avgs, "team_season_averages", conn)
            log.info(f"  Stored {len(season_avgs)} season average rows")

        # 3. Betting odds
        log.info("Fetching betting odds (this takes a few minutes)...")
        df_odds = sportsdata.fetch_game_odds(config.TRAINING_SEASONS)
        if not df_odds.empty:
            keep_odds = [
                "GameID" if "GameID" in df_odds.columns else "ScoreID",
                "Season_x" if "Season_x" in df_odds.columns else "Season",
                "Week", "HomeTeamName", "AwayTeamName", "HomeTeamId", "AwayTeamId",
                "year", "month", "dayofyear", "weekofyear", "hour", "dayofweek",
                "HomePointSpread", "AwayPointSpread", "OverUnder",
                "HomeMoneyLine", "AwayMoneyLine", "HomePointSpreadPayout",
                "AwayPointSpreadPayout", "OverPayout", "UnderPayout",
                "HomeTeamScore", "AwayTeamScore", "TotalScore",
            ]
            keep_odds = [c for c in keep_odds if c in df_odds.columns]
            df_save = df_odds[keep_odds].copy()
            if "GameID" not in df_save.columns and "ScoreID" in df_save.columns:
                df_save.rename(columns={"ScoreID": "GameID"}, inplace=True)
            if "Season_x" in df_save.columns:
                df_save.rename(columns={"Season_x": "Season"}, inplace=True)
            df_save["GameDate"] = df_odds.get("DateTime", "").astype(str) if "DateTime" in df_odds.columns else ""
            database.upsert_df(
                df_save.drop_duplicates(subset=["HomeTeamName", "year", "dayofyear"]),
                "betting_odds", conn,
            )
            log.info(f"  Stored {len(df_save)} betting odds rows")

        # 4. Referee assignments
        log.info("Scraping referee assignments (this takes several minutes)...")
        df_refs = referee_scraper.fetch_referee_assignments(years=config.TRAINING_SEASONS)
        if not df_refs.empty:
            database.upsert_df(df_refs, "referee_assignments", conn)
            log.info(f"  Stored {len(df_refs)} referee assignment rows")

        # 5. ELO ratings — calculated from completed game results
        log.info("Calculating ELO ratings from game results...")
        df_elo = calculate_elo_ratings(conn)
        if not df_elo.empty:
            database.upsert_df(df_elo, "elo_ratings", conn)
            log.info(f"  Stored {len(df_elo)} ELO rows")

        # 6. Google Trends (optional)
        if config.GOOGLE_TRENDS_ENABLED:
            log.info("Fetching Google Trends...")
            df_trends = google_trends.fetch_google_trends()
            if not df_trends.empty:
                database.upsert_df(df_trends, "google_trends", conn)
                log.info(f"  Stored {len(df_trends)} Google Trends rows")
        else:
            log.info("Google Trends disabled — skipping")

        # 7. Weather data (Open-Meteo — free, no key required)
        log.info("Fetching weather data from Open-Meteo...")
        df_weather = backfill_weather(conn)
        if not df_weather.empty:
            database.upsert_df(df_weather, "weather", conn)
            log.info(f"  Stored {len(df_weather)} weather rows")
        else:
            log.info("  No new weather data to store")

    log.info("=== Ingestion complete ===")


if __name__ == "__main__":
    run()
