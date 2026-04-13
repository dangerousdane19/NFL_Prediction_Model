"""
Database helpers — supports SQLite (local) and PostgreSQL (Streamlit Cloud via DATABASE_URL).
All schema creation and CRUD lives here.
"""
import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime

import pandas as pd

from nfl import config


def _is_postgres() -> bool:
    return bool(config.DATABASE_URL)


def get_connection():
    """Return a raw DB connection (sqlite3 or psycopg2)."""
    if _is_postgres():
        import psycopg2
        return psycopg2.connect(config.DATABASE_URL)
    os.makedirs(os.path.dirname(config.DB_PATH) or ".", exist_ok=True)
    return sqlite3.connect(config.DB_PATH, check_same_thread=False)


@contextmanager
def managed_conn():
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def create_all_tables() -> None:
    with managed_conn() as conn:
        cur = conn.cursor()
        cur.executescript("""
            CREATE TABLE IF NOT EXISTS stadiums (
                StadiumID       TEXT PRIMARY KEY,
                Stadium         TEXT NOT NULL,
                PlayingSurface  INTEGER,
                Type            INTEGER,
                updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS team_game_stats (
                TeamGameID              TEXT PRIMARY KEY,
                Team                    TEXT,
                Season                  INTEGER,
                Week                    INTEGER,
                SeasonType              TEXT,
                GameDate                TEXT,
                year                    INTEGER,
                month                   INTEGER,
                dayofyear               INTEGER,
                weekofyear              INTEGER,
                Stadium                 TEXT,
                PassingYards            REAL,
                RushingYards            REAL,
                Penalties               REAL,
                PenaltyYards            REAL,
                Turnovers               REAL,
                ThirdDownPercentage     REAL,
                RedZonePercentage       REAL,
                FirstDowns              REAL,
                FourthDownPercentage    REAL,
                updated_at              TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS team_season_averages (
                Team                        TEXT NOT NULL,
                Season                      INTEGER NOT NULL,
                avg_PassingYards            REAL,
                avg_RushingYards            REAL,
                avg_Penalties               REAL,
                avg_PenaltyYards            REAL,
                avg_Turnovers               REAL,
                avg_ThirdDownPercentage     REAL,
                avg_RedZonePercentage       REAL,
                avg_FirstDowns              REAL,
                avg_FourthDownPercentage    REAL,
                PRIMARY KEY (Team, Season)
            );

            CREATE TABLE IF NOT EXISTS betting_odds (
                GameID              TEXT,
                Season              INTEGER,
                Week                INTEGER,
                HomeTeamName        TEXT NOT NULL,
                AwayTeamName        TEXT,
                HomeTeamId          TEXT,
                AwayTeamId          TEXT,
                GameDate            TEXT,
                year                INTEGER,
                month               INTEGER,
                dayofyear           INTEGER,
                weekofyear          INTEGER,
                hour                INTEGER,
                dayofweek           INTEGER,
                HomePointSpread     REAL,
                AwayPointSpread     REAL,
                OverUnder           REAL,
                HomeMoneyLine       REAL,
                AwayMoneyLine       REAL,
                HomePointSpreadPayout REAL,
                AwayPointSpreadPayout REAL,
                OverPayout          REAL,
                UnderPayout         REAL,
                HomeTeamScore       REAL,
                AwayTeamScore       REAL,
                TotalScore          REAL,
                PRIMARY KEY (HomeTeamName, year, dayofyear)
            );

            CREATE TABLE IF NOT EXISTS referee_assignments (
                HomeTeamName    TEXT NOT NULL,
                month           INTEGER NOT NULL,
                year            INTEGER NOT NULL,
                dayofyear       INTEGER NOT NULL,
                Referee         INTEGER,
                PRIMARY KEY (HomeTeamName, year, dayofyear)
            );

            CREATE TABLE IF NOT EXISTS elo_ratings (
                HomeTeamName    TEXT NOT NULL,
                month           INTEGER NOT NULL,
                year            INTEGER NOT NULL,
                dayofyear       INTEGER NOT NULL,
                elo1_pre        REAL,
                elo2_pre        REAL,
                qbelo1_pre      REAL,
                qbelo2_pre      REAL,
                neutral         INTEGER,
                PRIMARY KEY (HomeTeamName, year, dayofyear)
            );

            CREATE TABLE IF NOT EXISTS google_trends (
                Team            TEXT NOT NULL,
                year            INTEGER NOT NULL,
                weekofyear      INTEGER NOT NULL,
                trend_value     REAL,
                HomeTeamGoogleTrend REAL,
                AwayTeamGoogleTrend REAL,
                PRIMARY KEY (Team, year, weekofyear)
            );

            CREATE TABLE IF NOT EXISTS predictions (
                id                      INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at              TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                home_team               TEXT NOT NULL,
                away_team               TEXT NOT NULL,
                season                  INTEGER,
                week                    INTEGER,
                game_date               TEXT,
                over_under_input        REAL,
                home_point_spread_input REAL,
                away_point_spread_input REAL,
                home_money_line         REAL,
                away_money_line         REAL,
                pred_home_score         REAL,
                pred_away_score         REAL,
                pred_total_score        REAL,
                pred_winner             TEXT,
                pred_home_cover         INTEGER,
                pred_home_cover_proba   REAL,
                pred_away_cover         INTEGER,
                pred_away_cover_proba   REAL,
                pred_bet_outcome        INTEGER,
                pred_bet_outcome_proba  REAL,
                stat_strategy_used      TEXT,
                google_trends_used      INTEGER,
                actual_home_score       REAL,
                actual_away_score       REAL,
                actual_total_score      REAL,
                actual_winner           TEXT,
                notes                   TEXT
            );

            CREATE TABLE IF NOT EXISTS training_runs (
                id                      INTEGER PRIMARY KEY AUTOINCREMENT,
                run_at                  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                seasons_trained         TEXT,
                num_records             INTEGER,
                modelts_rmse            REAL,
                modelas_rmse            REAL,
                modelhs_rmse            REAL,
                logreg_home_accuracy    REAL,
                logreg_away_accuracy    REAL,
                logreg_bet_accuracy     REAL
            );
        """) if not _is_postgres() else _create_postgres_tables(cur)


def _create_postgres_tables(cur) -> None:
    statements = [
        """CREATE TABLE IF NOT EXISTS stadiums (
            "StadiumID" TEXT PRIMARY KEY, "Stadium" TEXT NOT NULL,
            "PlayingSurface" INTEGER, "Type" INTEGER,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""",
        """CREATE TABLE IF NOT EXISTS betting_odds (
            "GameID" TEXT, "Season" INTEGER, "Week" INTEGER,
            "HomeTeamName" TEXT NOT NULL, "AwayTeamName" TEXT,
            "HomeTeamId" TEXT, "AwayTeamId" TEXT,
            "GameDate" TEXT, year INTEGER, month INTEGER,
            dayofyear INTEGER, weekofyear INTEGER, hour INTEGER, dayofweek INTEGER,
            "HomePointSpread" REAL, "AwayPointSpread" REAL, "OverUnder" REAL,
            "HomeMoneyLine" REAL, "AwayMoneyLine" REAL,
            "HomePointSpreadPayout" REAL, "AwayPointSpreadPayout" REAL,
            "OverPayout" REAL, "UnderPayout" REAL,
            "HomeTeamScore" REAL, "AwayTeamScore" REAL, "TotalScore" REAL,
            PRIMARY KEY ("HomeTeamName", year, dayofyear))""",
        """CREATE TABLE IF NOT EXISTS referee_assignments (
            "HomeTeamName" TEXT NOT NULL, month INTEGER NOT NULL,
            year INTEGER NOT NULL, dayofyear INTEGER NOT NULL,
            "Referee" INTEGER, PRIMARY KEY ("HomeTeamName", year, dayofyear))""",
        """CREATE TABLE IF NOT EXISTS elo_ratings (
            "HomeTeamName" TEXT NOT NULL, month INTEGER NOT NULL,
            year INTEGER NOT NULL, dayofyear INTEGER NOT NULL,
            elo1_pre REAL, elo2_pre REAL, qbelo1_pre REAL, qbelo2_pre REAL,
            neutral INTEGER, PRIMARY KEY ("HomeTeamName", year, dayofyear))""",
        """CREATE TABLE IF NOT EXISTS google_trends (
            "Team" TEXT NOT NULL, year INTEGER NOT NULL, weekofyear INTEGER NOT NULL,
            trend_value REAL, "HomeTeamGoogleTrend" REAL, "AwayTeamGoogleTrend" REAL,
            PRIMARY KEY ("Team", year, weekofyear))""",
        """CREATE TABLE IF NOT EXISTS team_game_stats (
            "TeamGameID" TEXT PRIMARY KEY, "Team" TEXT, "Season" INTEGER, "Week" INTEGER,
            "SeasonType" TEXT, "GameDate" TEXT, year INTEGER, month INTEGER,
            dayofyear INTEGER, weekofyear INTEGER, "Stadium" TEXT,
            "PassingYards" REAL, "RushingYards" REAL, "Penalties" REAL,
            "PenaltyYards" REAL, "Turnovers" REAL, "ThirdDownPercentage" REAL,
            "RedZonePercentage" REAL, "FirstDowns" REAL, "FourthDownPercentage" REAL,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""",
        """CREATE TABLE IF NOT EXISTS team_season_averages (
            "Team" TEXT NOT NULL, "Season" INTEGER NOT NULL,
            avg_PassingYards REAL, avg_RushingYards REAL, avg_Penalties REAL,
            avg_PenaltyYards REAL, avg_Turnovers REAL, avg_ThirdDownPercentage REAL,
            avg_RedZonePercentage REAL, avg_FirstDowns REAL, avg_FourthDownPercentage REAL,
            PRIMARY KEY ("Team", "Season"))""",
        """CREATE TABLE IF NOT EXISTS predictions (
            id SERIAL PRIMARY KEY, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            home_team TEXT NOT NULL, away_team TEXT NOT NULL,
            season INTEGER, week INTEGER, game_date TEXT,
            over_under_input REAL, home_point_spread_input REAL, away_point_spread_input REAL,
            home_money_line REAL, away_money_line REAL,
            pred_home_score REAL, pred_away_score REAL, pred_total_score REAL,
            pred_winner TEXT, pred_home_cover INTEGER, pred_home_cover_proba REAL,
            pred_away_cover INTEGER, pred_away_cover_proba REAL,
            pred_bet_outcome INTEGER, pred_bet_outcome_proba REAL,
            stat_strategy_used TEXT, google_trends_used INTEGER,
            actual_home_score REAL, actual_away_score REAL,
            actual_total_score REAL, actual_winner TEXT, notes TEXT)""",
        """CREATE TABLE IF NOT EXISTS training_runs (
            id SERIAL PRIMARY KEY, run_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            seasons_trained TEXT, num_records INTEGER,
            modelts_rmse REAL, modelas_rmse REAL, modelhs_rmse REAL,
            logreg_home_accuracy REAL, logreg_away_accuracy REAL, logreg_bet_accuracy REAL)""",
    ]
    for s in statements:
        cur.execute(s)


def upsert_df(df: pd.DataFrame, table: str, conn) -> None:
    """Insert-or-replace rows from a DataFrame into a table."""
    if df.empty:
        return
    ph = "?" if not _is_postgres() else "%s"
    cols = list(df.columns)
    col_str = ", ".join(f'"{c}"' for c in cols)
    val_str = ", ".join([ph] * len(cols))
    if _is_postgres():
        sql = f"INSERT INTO {table} ({col_str}) VALUES ({val_str}) ON CONFLICT DO NOTHING"
    else:
        sql = f"INSERT OR REPLACE INTO {table} ({col_str}) VALUES ({val_str})"
    cur = conn.cursor()
    cur.executemany(sql, df.values.tolist())


def insert_prediction(record: dict) -> int:
    with managed_conn() as conn:
        cols = list(record.keys())
        ph = "?" if not _is_postgres() else "%s"
        col_str = ", ".join(cols)
        val_str = ", ".join([ph] * len(cols))
        sql = f"INSERT INTO predictions ({col_str}) VALUES ({val_str})"
        cur = conn.cursor()
        cur.execute(sql, list(record.values()))
        if _is_postgres():
            cur.execute("SELECT lastval()")
        return cur.lastrowid or 0


def fetch_prediction_history(limit: int = 200) -> pd.DataFrame:
    with managed_conn() as conn:
        return pd.read_sql(
            f"SELECT * FROM predictions ORDER BY created_at DESC LIMIT {limit}",
            conn,
        )


def delete_prediction(prediction_id: int) -> None:
    ph = "?" if not _is_postgres() else "%s"
    with managed_conn() as conn:
        conn.cursor().execute(f"DELETE FROM predictions WHERE id = {ph}", (prediction_id,))


def fetch_last_training_run() -> dict:
    with managed_conn() as conn:
        df = pd.read_sql(
            "SELECT * FROM training_runs ORDER BY run_at DESC LIMIT 1", conn
        )
    return df.iloc[0].to_dict() if not df.empty else {}


def insert_training_run(record: dict) -> None:
    with managed_conn() as conn:
        cols = list(record.keys())
        ph = "?" if not _is_postgres() else "%s"
        col_str = ", ".join(cols)
        val_str = ", ".join([ph] * len(cols))
        sql = f"INSERT INTO training_runs ({col_str}) VALUES ({val_str})"
        conn.cursor().execute(sql, list(record.values()))
