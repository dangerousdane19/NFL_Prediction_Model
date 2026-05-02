"""
Train all 6 models and save as joblib files.
Exact hyperparameters from the Kaggle notebook.
"""
import logging
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from nfl import config, database

log = logging.getLogger(__name__)

# Feature columns to drop when building X
REGRESSION_DROP = ["HomeTeamScore", "AwayTeamScore", "TotalScore_x", "OpponentScore", "DateTime", "AwayTeamName", "HomeTeamName"]
CLASSIFICATION_DROP = REGRESSION_DROP + ["BetOutcome", "HomeTeamCover", "AwayTeamCover"]


def _split(df: pd.DataFrame, target: str, drop_cols: list):
    cols_to_drop = [c for c in drop_cols if c in df.columns]
    X = df.drop(columns=cols_to_drop)
    # Keep only numeric columns
    X = X.select_dtypes(include="number")
    y = df[target]
    return train_test_split(X, y, test_size=0.2, random_state=1234)


def train_all_models(NFLmodeldata: pd.DataFrame, NFLmodeldata1: pd.DataFrame) -> dict:
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    metrics = {}

    # ── Regression models ──────────────────────────────────────────────────────

    # Total Score (Cell 100)
    X_train, X_test, y_train, y_test = _split(NFLmodeldata, "TotalScore_x", REGRESSION_DROP)
    feature_columns = list(X_train.columns)
    joblib.dump(feature_columns, os.path.join(config.MODEL_DIR, "feature_columns.joblib"))

    modelts = XGBRegressor(
        n_estimators=400, min_child_weight=5, max_depth=3,
        learning_rate=0.1, gamma=0.5, random_state=42,
    )
    modelts.fit(X_train, y_train)
    y_pred = modelts.predict(X_test)
    metrics["modelts_rmse"] = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    metrics["modelts_mae"] = float(mean_absolute_error(y_test, y_pred))
    metrics["modelts_r2"] = float(r2_score(y_test, y_pred))
    joblib.dump(modelts, os.path.join(config.MODEL_DIR, "modelts.joblib"))
    log.info(f"modelts RMSE={metrics['modelts_rmse']:.2f}")

    # Away Score (Cell 115)
    X_train, X_test, y_train, y_test = _split(NFLmodeldata, "OpponentScore", REGRESSION_DROP)
    modelas = XGBRegressor(
        n_estimators=300, min_child_weight=7, max_depth=3,
        learning_rate=0.1, gamma=1, random_state=42,
    )
    modelas.fit(X_train, y_train)
    y_pred = modelas.predict(X_test)
    metrics["modelas_rmse"] = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    joblib.dump(modelas, os.path.join(config.MODEL_DIR, "modelas.joblib"))
    log.info(f"modelas RMSE={metrics['modelas_rmse']:.2f}")

    # Home Score (Cell 122)
    X_train, X_test, y_train, y_test = _split(NFLmodeldata, "HomeTeamScore", REGRESSION_DROP)
    modelhs = XGBRegressor(
        n_estimators=300, min_child_weight=1, max_depth=3,
        learning_rate=0.1, gamma=0, random_state=42,
    )
    modelhs.fit(X_train, y_train)
    y_pred = modelhs.predict(X_test)
    metrics["modelhs_rmse"] = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    joblib.dump(modelhs, os.path.join(config.MODEL_DIR, "modelhs.joblib"))
    log.info(f"modelhs RMSE={metrics['modelhs_rmse']:.2f}")

    # ── Classification models ───────────────────────────────────────────────────

    # Home Team Cover (Cell 136)
    X_train, X_test, y_train, y_test = _split(NFLmodeldata1, "HomeTeamCover", CLASSIFICATION_DROP)
    logreg_homecover = LogisticRegression(max_iter=10000)
    logreg_homecover.fit(X_train, y_train)
    metrics["logreg_home_accuracy"] = float(accuracy_score(y_test, logreg_homecover.predict(X_test)))
    joblib.dump(logreg_homecover, os.path.join(config.MODEL_DIR, "logreg_homecover.joblib"))
    log.info(f"logreg_homecover accuracy={metrics['logreg_home_accuracy']:.3f}")

    # Away Team Cover (Cell 142)
    X_train, X_test, y_train, y_test = _split(NFLmodeldata1, "AwayTeamCover", CLASSIFICATION_DROP)
    logreg_awaycover = LogisticRegression(max_iter=10000)
    logreg_awaycover.fit(X_train, y_train)
    metrics["logreg_away_accuracy"] = float(accuracy_score(y_test, logreg_awaycover.predict(X_test)))
    joblib.dump(logreg_awaycover, os.path.join(config.MODEL_DIR, "logreg_awaycover.joblib"))
    log.info(f"logreg_awaycover accuracy={metrics['logreg_away_accuracy']:.3f}")

    # Bet Outcome / Over-Under (Cell 148)
    X_train, X_test, y_train, y_test = _split(NFLmodeldata1, "BetOutcome", CLASSIFICATION_DROP)
    logreg_betoutcome = LogisticRegression(max_iter=10000)
    logreg_betoutcome.fit(X_train, y_train)
    metrics["logreg_bet_accuracy"] = float(accuracy_score(y_test, logreg_betoutcome.predict(X_test)))
    joblib.dump(logreg_betoutcome, os.path.join(config.MODEL_DIR, "logreg_betoutcome.joblib"))
    log.info(f"logreg_betoutcome accuracy={metrics['logreg_bet_accuracy']:.3f}")

    return metrics


def load_models() -> tuple:
    """Returns (models_dict, feature_columns)."""
    model_files = {
        "modelts": "modelts.joblib",
        "modelas": "modelas.joblib",
        "modelhs": "modelhs.joblib",
        "logreg_homecover": "logreg_homecover.joblib",
        "logreg_awaycover": "logreg_awaycover.joblib",
        "logreg_betoutcome": "logreg_betoutcome.joblib",
    }
    models = {}
    for name, fname in model_files.items():
        path = os.path.join(config.MODEL_DIR, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}. Run `python scripts/run_training.py` first.")
        models[name] = joblib.load(path)

    fc_path = os.path.join(config.MODEL_DIR, "feature_columns.joblib")
    if not os.path.exists(fc_path):
        raise FileNotFoundError(f"Feature columns file not found: {fc_path}.")
    feature_columns = joblib.load(fc_path)
    return models, feature_columns
