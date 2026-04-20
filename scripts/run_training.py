#!/usr/bin/env python3
"""
CLI script: build the merged dataset from the DB and train all 6 models.
Run: python scripts/run_training.py
"""
import json
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

import pandas as pd

from nfl import config, database
from nfl.features.engineer import prepare_model_data
from nfl.features.merge import build_nfl_dataset
from nfl.training.train import train_all_models


def run():
    log.info("=== NFL Model Training ===")
    database.create_all_tables()

    with database.managed_conn() as conn:
        log.info("Loading data from database...")
        stadiums = pd.read_sql("SELECT * FROM stadiums", conn)
        teamstats = pd.read_sql("SELECT * FROM team_game_stats", conn)
        betting_odds = pd.read_sql("SELECT * FROM betting_odds", conn)
        referee_assignments = pd.read_sql("SELECT * FROM referee_assignments", conn)
        elo_ratings = pd.read_sql("SELECT * FROM elo_ratings", conn)

        try:
            google_trends = pd.read_sql("SELECT * FROM google_trends", conn)
        except Exception:
            google_trends = pd.DataFrame()

        try:
            weather_df = pd.read_sql("SELECT * FROM weather", conn)
        except Exception:
            weather_df = pd.DataFrame()

    # Rename betting_odds columns to match notebook expectations
    col_renames = {
        "Season": "Season_x",
        "TotalScore": "TotalScore_x",
    }
    betting_odds.rename(columns={k: v for k, v in col_renames.items() if k in betting_odds.columns}, inplace=True)

    # Alias AwayTeamScore → OpponentScore so training targets are consistent
    if "AwayTeamScore" in betting_odds.columns and "OpponentScore" not in betting_odds.columns:
        betting_odds["OpponentScore"] = betting_odds["AwayTeamScore"]

    # teamstats needs TotalScore column
    if "TotalScore" not in teamstats.columns and "TotalScore_x" not in teamstats.columns:
        log.warning("TotalScore not found in team_game_stats — check ingestion")

    log.info("Building NFLdataset (merging all sources)...")
    NFLdataset = build_nfl_dataset(
        teamstats=teamstats,
        stadiums=stadiums,
        betting_odds=betting_odds,
        referee_assignments=referee_assignments,
        elo_ratings=elo_ratings,
        google_trends=google_trends,
        weather_df=weather_df,
    )

    if NFLdataset.empty:
        log.error("NFLdataset is empty — cannot train. Check that ingestion ran successfully.")
        sys.exit(1)

    log.info("Engineering features...")
    NFLmodeldata, NFLmodeldata1 = prepare_model_data(NFLdataset)

    # Validate targets exist
    for target in ["TotalScore_x", "OpponentScore", "HomeTeamScore"]:
        if target not in NFLmodeldata.columns:
            log.error(f"Target column '{target}' not found in NFLmodeldata. Available: {list(NFLmodeldata.columns)}")
            sys.exit(1)

    log.info(f"Training on {len(NFLmodeldata)} records...")
    metrics = train_all_models(NFLmodeldata, NFLmodeldata1)

    # Log training run to DB (only columns defined in training_runs schema)
    run_record = {
        "seasons_trained": json.dumps(config.TRAINING_SEASONS),
        "num_records": len(NFLmodeldata),
        "modelts_rmse": metrics.get("modelts_rmse"),
        "modelas_rmse": metrics.get("modelas_rmse"),
        "modelhs_rmse": metrics.get("modelhs_rmse"),
        "logreg_home_accuracy": metrics.get("logreg_home_accuracy"),
        "logreg_away_accuracy": metrics.get("logreg_away_accuracy"),
        "logreg_bet_accuracy": metrics.get("logreg_bet_accuracy"),
    }
    database.insert_training_run(run_record)

    log.info("=== Training complete ===")
    log.info(f"  Total Score RMSE:        {metrics.get('modelts_rmse', 'N/A'):.2f}")
    log.info(f"  Away Score RMSE:         {metrics.get('modelas_rmse', 'N/A'):.2f}")
    log.info(f"  Home Score RMSE:         {metrics.get('modelhs_rmse', 'N/A'):.2f}")
    log.info(f"  Home Cover Accuracy:     {metrics.get('logreg_home_accuracy', 'N/A'):.3f}")
    log.info(f"  Away Cover Accuracy:     {metrics.get('logreg_away_accuracy', 'N/A'):.3f}")
    log.info(f"  Over/Under Accuracy:     {metrics.get('logreg_bet_accuracy', 'N/A'):.3f}")
    log.info(f"  Models saved to:         {config.MODEL_DIR}/")


if __name__ == "__main__":
    run()
