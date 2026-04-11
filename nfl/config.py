import os
from dotenv import load_dotenv

load_dotenv()


def _get(key: str, default: str = "") -> str:
    val = os.getenv(key)
    if val:
        return val
    try:
        import streamlit as st
        return st.secrets[key]
    except Exception:
        return default


SPORTSDATA_API_KEY: str = _get("SPORTSDATA_API_KEY")
DB_PATH: str = _get("DB_PATH", "data/nfl_predictions.db")
DATABASE_URL: str = _get("DATABASE_URL", "")  # PostgreSQL for Streamlit Cloud
MODEL_DIR: str = _get("MODEL_DIR", "models")
GOOGLE_TRENDS_ENABLED: bool = _get("GOOGLE_TRENDS_ENABLED", "false").lower() == "true"

# Seasons to ingest (REG + POST)
TRAINING_SEASONS: list = [2020, 2021, 2022, 2023, 2024]
REG_WEEKS: int = 18
POST_WEEKS: int = 4
