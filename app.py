"""
NFL Game Predictor — Streamlit App
4 pages: Predict, History, Retrain, Model Info
"""
import logging
import os
import sys
from datetime import date

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO)

from nfl import config, database
from nfl.prediction.predict import predict_game
from nfl.team_logos import get_logo_path

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NFL Predictor",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ──────────────────────────────────────────────────────────────────
NFL_TEAMS = [
    "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE",
    "DAL", "DEN", "DET", "GB", "HOU", "IND", "JAX", "KC",
    "LAC", "LAR", "LV", "MIA", "MIN", "NE", "NO", "NYG",
    "NYJ", "PHI", "PIT", "SEA", "SF", "TB", "TEN", "WAS",
]

TEAM_FULL = {
    "ARI": "Arizona Cardinals", "ATL": "Atlanta Falcons", "BAL": "Baltimore Ravens",
    "BUF": "Buffalo Bills", "CAR": "Carolina Panthers", "CHI": "Chicago Bears",
    "CIN": "Cincinnati Bengals", "CLE": "Cleveland Browns", "DAL": "Dallas Cowboys",
    "DEN": "Denver Broncos", "DET": "Detroit Lions", "GB": "Green Bay Packers",
    "HOU": "Houston Texans", "IND": "Indianapolis Colts", "JAX": "Jacksonville Jaguars",
    "KC": "Kansas City Chiefs", "LAC": "Los Angeles Chargers", "LAR": "Los Angeles Rams",
    "LV": "Las Vegas Raiders", "MIA": "Miami Dolphins", "MIN": "Minnesota Vikings",
    "NE": "New England Patriots", "NO": "New Orleans Saints", "NYG": "New York Giants",
    "NYJ": "New York Jets", "PHI": "Philadelphia Eagles", "PIT": "Pittsburgh Steelers",
    "SEA": "Seattle Seahawks", "SF": "San Francisco 49ers", "TB": "Tampa Bay Buccaneers",
    "TEN": "Tennessee Titans", "WAS": "Washington Commanders",
}

PLAYING_SURFACES = {"Grass": 3, "Artificial": 1, "Dome": 2}
STADIUM_TYPES = {"Outdoor": 2, "Dome": 1, "Retractable Dome": 3}

REFEREE_IDS = {
    "Unknown": 0, "Adrian Hill": 1, "Alex Kemp": 2, "Bill Vinovich": 3,
    "Brad Allen": 4, "Brad Rogers": 5, "Carl Cheffers": 6, "Clay Martin": 7,
    "Clete Blakeman": 8, "Craig Wrolstad": 9, "Jerome Boger": 10,
    "John Hussey": 11, "Ron Torbert": 12, "Scott Novak": 13,
    "Shawn Hochuli": 15, "Shawn Smith": 16, "Tra Blake": 17,
    "Land Clark": 18, "Tony Corrente": 19,
}


@st.cache_resource
def init_db():
    database.create_all_tables()
    return database.get_connection()


@st.cache_data(ttl=300)
def load_prediction_history():
    return database.fetch_prediction_history(limit=200)


@st.cache_data(ttl=3600)
def load_schedule(season: int) -> pd.DataFrame:
    try:
        import nfl_data_py as nfl
        df = nfl.import_schedules([season])
        if df.empty:
            return pd.DataFrame()
        # Normalise team abbreviations (nfl_data_py uses 'LA' for Rams)
        team_map = {"LA": "LAR"}
        df["home_team"] = df["home_team"].replace(team_map)
        df["away_team"] = df["away_team"].replace(team_map)
        df["gameday"] = pd.to_datetime(df["gameday"], errors="coerce")
        return df[df["gameday"].notna()].copy()
    except Exception:
        return pd.DataFrame()


SURFACE_MAP = {
    "grass": "Grass",
    "dessograss": "Grass",
}
ROOF_MAP = {
    "outdoors": "Outdoor",
    "open": "Outdoor",
    "dome": "Dome",
    "closed": "Retractable Dome",
}


def _init_predict_state():
    defaults = {
        "pred_home_team": "KC",
        "pred_away_team": "SF",
        "pred_game_date": date.today(),
        "pred_season": date.today().year,
        "pred_week": 1,
        "pred_over_under": 45.0,
        "pred_home_spread": -3.0,
        "pred_away_spread": 3.0,
        "pred_home_ml": -150,
        "pred_away_ml": 130,
        "pred_surface": "Grass",
        "pred_stadium_type": "Outdoor",
        "pred_neutral_site": False,
        "pred_referee": "Unknown",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


@st.cache_data(ttl=600)
def load_last_training_run():
    return database.fetch_last_training_run()


def models_exist() -> bool:
    return all(
        os.path.exists(os.path.join(config.MODEL_DIR, f))
        for f in ["modelts.joblib", "modelas.joblib", "modelhs.joblib",
                  "logreg_homecover.joblib", "logreg_awaycover.joblib",
                  "logreg_betoutcome.joblib", "feature_columns.joblib"]
    )


# ── Sidebar navigation ─────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🏈 NFL Predictor")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["Predict a Game", "Prediction History", "Retrain / Refresh Data", "Model Info"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    if models_exist():
        st.success("Models loaded")
    else:
        st.warning("Models not trained yet\nGo to **Retrain** to train.")


# ── Page: Predict a Game ───────────────────────────────────────────────────────
if page == "Predict a Game":
    st.title("Predict a Game")

    if not models_exist():
        st.error("No trained models found. Go to **Retrain / Refresh Data** to train the models first.")
        st.stop()

    _init_predict_state()

    # ── Schedule picker ────────────────────────────────────────────────────────
    st.subheader("Select from Schedule")
    sched_col1, sched_col2 = st.columns([1, 3])
    with sched_col1:
        sched_season = st.number_input(
            "Season", value=date.today().year, min_value=2020, max_value=2030, step=1, key="sched_season_input"
        )

    df_sched = load_schedule(int(sched_season))

    if df_sched.empty:
        st.info(f"No schedule data available for {int(sched_season)}. The schedule may not have been released yet — try {int(sched_season) - 1}.")
    else:
        reg_types = ["REG", "WC", "DIV", "CON", "SB"]
        type_labels = {"REG": "Regular Season", "WC": "Wild Card", "DIV": "Divisional", "CON": "Conference", "SB": "Super Bowl"}
        available_types = [t for t in reg_types if t in df_sched["game_type"].unique()]

        with sched_col2:
            game_type_sel = st.selectbox(
                "Game Type",
                available_types,
                format_func=lambda t: type_labels.get(t, t),
                key="sched_game_type",
            )

        type_df = df_sched[df_sched["game_type"] == game_type_sel]
        weeks = sorted(type_df["week"].dropna().unique().astype(int).tolist())

        sc1, sc2, sc3 = st.columns([1, 3, 1])
        with sc1:
            sel_week = st.selectbox("Week", weeks, key="sched_week_sel")

        week_df = type_df[type_df["week"] == sel_week].sort_values("gameday")

        game_labels = [
            f"{row['away_team']} @ {row['home_team']}  —  {pd.Timestamp(row['gameday']).strftime('%a %b %-d')}"
            for _, row in week_df.iterrows()
        ]

        with sc2:
            sel_game_label = st.selectbox("Matchup", game_labels, key="sched_game_sel")

        sel_idx = game_labels.index(sel_game_label)
        sel_row = week_df.iloc[sel_idx]

        with sc3:
            st.write("")  # vertical align
            st.write("")
            if st.button("Load Game", type="primary"):
                home = sel_row["home_team"]
                away = sel_row["away_team"]
                if home in NFL_TEAMS and away in NFL_TEAMS:
                    st.session_state.pred_home_team = home
                    st.session_state.pred_away_team = away
                    st.session_state.pred_game_date = sel_row["gameday"].date()
                    st.session_state.pred_season = int(sel_row["season"])
                    st.session_state.pred_week = int(sel_row["week"])
                    # Vegas lines (may be NaN for future games)
                    if pd.notna(sel_row.get("total_line")):
                        st.session_state.pred_over_under = float(sel_row["total_line"])
                    if pd.notna(sel_row.get("spread_line")):
                        st.session_state.pred_away_spread = float(sel_row["spread_line"])
                        st.session_state.pred_home_spread = float(-sel_row["spread_line"])
                    if pd.notna(sel_row.get("home_moneyline")):
                        st.session_state.pred_home_ml = int(sel_row["home_moneyline"])
                    if pd.notna(sel_row.get("away_moneyline")):
                        st.session_state.pred_away_ml = int(sel_row["away_moneyline"])
                    # Stadium info
                    roof_raw = sel_row.get("roof", "")
                    surf_raw = sel_row.get("surface", "")
                    if isinstance(roof_raw, str) and roof_raw in ROOF_MAP:
                        st.session_state.pred_stadium_type = ROOF_MAP[roof_raw]
                    if isinstance(surf_raw, str) and pd.notna(surf_raw):
                        st.session_state.pred_surface = SURFACE_MAP.get(surf_raw, "Artificial")
                    # Neutral site
                    location = sel_row.get("location", "Home")
                    st.session_state.pred_neutral_site = (location == "Neutral")
                    # Referee
                    ref_name = sel_row.get("referee", "")
                    if isinstance(ref_name, str) and ref_name in REFEREE_IDS:
                        st.session_state.pred_referee = ref_name
                    else:
                        st.session_state.pred_referee = "Unknown"
                    st.rerun()

    st.markdown("---")

    # ── Manual form ───────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        title_col, logo_col = st.columns([3, 1])
        with title_col:
            st.subheader("Home Team")
        with logo_col:
            home_logo = get_logo_path(st.session_state.get("pred_home_team", "KC"))
            if home_logo:
                st.image(home_logo, width=50)
        home_team = st.selectbox("Home Team", NFL_TEAMS, key="pred_home_team")
        st.caption(TEAM_FULL.get(home_team, ""))
    with col2:
        title_col2, logo_col2 = st.columns([3, 1])
        with title_col2:
            st.subheader("Away Team")
        with logo_col2:
            away_logo = get_logo_path(st.session_state.get("pred_away_team", "SF"))
            if away_logo:
                st.image(away_logo, width=50)
        away_team = st.selectbox("Away Team", NFL_TEAMS, key="pred_away_team")
        st.caption(TEAM_FULL.get(away_team, ""))

    if home_team == away_team:
        st.warning("Home and away teams must be different.")

    col3, col4, col5 = st.columns(3)
    with col3:
        game_date = st.date_input("Game Date", key="pred_game_date")
    with col4:
        season = st.number_input("Season (Year)", min_value=2020, max_value=2030, step=1, key="pred_season")
    with col5:
        week = st.number_input("Week", min_value=1, max_value=22, step=1, key="pred_week")

    st.markdown("---")
    st.subheader("Vegas Lines")
    col6, col7, col8 = st.columns(3)
    with col6:
        over_under = st.number_input("Over/Under", min_value=20.0, max_value=80.0, step=0.5, key="pred_over_under")
    with col7:
        home_spread = st.number_input("Home Point Spread", min_value=-30.0, max_value=30.0, step=0.5,
                                      help="Negative = home team favored", key="pred_home_spread")
    with col8:
        away_spread = st.number_input("Away Point Spread", min_value=-30.0, max_value=30.0, step=0.5, key="pred_away_spread")

    col9, col10 = st.columns(2)
    with col9:
        home_ml = st.number_input("Home Money Line", min_value=-2000, max_value=2000, step=5, key="pred_home_ml")
    with col10:
        away_ml = st.number_input("Away Money Line", min_value=-2000, max_value=2000, step=5, key="pred_away_ml")

    with st.expander("Stadium & Advanced Options"):
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            surface = st.selectbox("Playing Surface", list(PLAYING_SURFACES.keys()),
                                   index=list(PLAYING_SURFACES.keys()).index(st.session_state.pred_surface),
                                   key="pred_surface")
            stadium_type = st.selectbox("Stadium Type", list(STADIUM_TYPES.keys()),
                                        index=list(STADIUM_TYPES.keys()).index(st.session_state.pred_stadium_type),
                                        key="pred_stadium_type")
        with col_s2:
            neutral_site = st.checkbox("Neutral Site", value=st.session_state.pred_neutral_site, key="pred_neutral_site")
            referee = st.selectbox(
                "Referee", list(REFEREE_IDS.keys()),
                index=list(REFEREE_IDS.keys()).index(st.session_state.pred_referee),
                key="pred_referee",
                help="Assignments are announced Wednesday of game week. Set manually once known.",
            )

        col_e1, col_e2 = st.columns(2)
        with col_e1:
            elo1 = st.number_input("Home ELO (pre-game)", value=1505.0, step=1.0)
            qbelo1 = st.number_input("Home QB ELO (pre-game)", value=1505.0, step=1.0)
        with col_e2:
            elo2 = st.number_input("Away ELO (pre-game)", value=1495.0, step=1.0)
            qbelo2 = st.number_input("Away QB ELO (pre-game)", value=1495.0, step=1.0)

    st.markdown("---")
    predict_btn = st.button("Predict", type="primary", disabled=(home_team == away_team))

    if predict_btn:
        game_inputs = {
            "home_team": home_team,
            "away_team": away_team,
            "game_date": game_date,
            "season": int(season),
            "week": int(week),
            "over_under": over_under,
            "home_point_spread": home_spread,
            "away_point_spread": away_spread,
            "home_money_line": float(home_ml),
            "away_money_line": float(away_ml),
            "playing_surface": PLAYING_SURFACES[surface],
            "stadium_type": STADIUM_TYPES[stadium_type],
            "neutral": int(neutral_site),
            "referee": REFEREE_IDS[referee],
            "elo1_pre": elo1,
            "elo2_pre": elo2,
            "qbelo1_pre": qbelo1,
            "qbelo2_pre": qbelo2,
        }

        with st.spinner("Running models..."):
            try:
                result = predict_game(game_inputs)
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.stop()

        # Strategy warnings
        if result["stat_strategy_used"] == "prior_season_avg":
            st.warning("Using prior-season averages — no current season data available yet. Accuracy may be lower.")
        elif result["stat_strategy_used"] == "empty":
            st.warning("No historical stats found for this team. Predictions are based on Vegas lines and context only.")

        st.markdown("---")
        st.subheader("Results")

        # Scores
        rc1, rc2, rc3 = st.columns(3)
        with rc1:
            st.metric(f"Predicted {home_team} Score", f"{result['home_score']:.1f}")
        with rc2:
            st.metric(f"Predicted {away_team} Score", f"{result['away_score']:.1f}")
        with rc3:
            st.metric("Predicted Total", f"{result['total_score']:.1f}",
                      delta=f"Vegas O/U: {over_under}", delta_color="off")

        # Winner
        winner_team = home_team if result["predicted_winner"] == "HOME" else away_team
        margin = abs(result["home_score"] - result["away_score"])
        st.markdown(f"### Predicted Winner: **{winner_team}** by {margin:.1f} pts")

        st.markdown("---")
        st.subheader("Betting Predictions")
        bc1, bc2, bc3 = st.columns(3)
        with bc1:
            cover_label = "COVER" if result["home_cover"] == 0 else "PUSH / NO COVER"
            st.metric(
                f"{home_team} Spread ({home_spread:+.1f})",
                cover_label,
                f"{result['home_cover_proba']:.0%} confidence",
            )
        with bc2:
            acover_label = "COVER" if result["away_cover"] == 0 else "PUSH / NO COVER"
            st.metric(
                f"{away_team} Spread ({away_spread:+.1f})",
                acover_label,
                f"{result['away_cover_proba']:.0%} confidence",
            )
        with bc3:
            ou_label = "OVER" if result["bet_outcome"] == 0 else "PUSH / UNDER"
            st.metric(
                f"Over/Under ({over_under})",
                ou_label,
                f"{result['bet_outcome_proba']:.0%} confidence",
            )

        # Invalidate history cache
        load_prediction_history.clear()


# ── Page: Prediction History ───────────────────────────────────────────────────
elif page == "Prediction History":
    st.title("Prediction History")

    df = load_prediction_history()

    if df.empty:
        st.info("No predictions yet. Make a prediction on the **Predict a Game** page.")
    else:
        # Filters
        col1, col2 = st.columns(2)
        with col1:
            team_filter = st.multiselect("Filter by Team", NFL_TEAMS)
        with col2:
            seasons = sorted(df["season"].dropna().unique().astype(int).tolist(), reverse=True)
            season_filter = st.selectbox("Season", ["All"] + seasons)

        if team_filter:
            df = df[df["home_team"].isin(team_filter) | df["away_team"].isin(team_filter)]
        if season_filter != "All":
            df = df[df["season"] == int(season_filter)]

        st.caption(f"Showing {len(df)} predictions — select rows to delete")

        display_cols = [
            "created_at", "home_team", "away_team", "season", "week", "game_date",
            "over_under_input", "home_point_spread_input",
            "pred_home_score", "pred_away_score", "pred_total_score",
            "pred_winner", "pred_home_cover_proba", "pred_away_cover_proba",
            "pred_bet_outcome_proba", "stat_strategy_used",
            "actual_home_score", "actual_away_score",
        ]
        event = st.dataframe(
            df[display_cols],
            column_config={
                "created_at": st.column_config.TextColumn("Predicted At"),
                "home_team": "Home",
                "away_team": "Away",
                "pred_home_score": st.column_config.NumberColumn("Pred Home", format="%.1f"),
                "pred_away_score": st.column_config.NumberColumn("Pred Away", format="%.1f"),
                "pred_total_score": st.column_config.NumberColumn("Pred Total", format="%.1f"),
                "pred_home_cover_proba": st.column_config.ProgressColumn("Home Cover %", format="%.0f%%", min_value=0, max_value=1),
                "pred_away_cover_proba": st.column_config.ProgressColumn("Away Cover %", format="%.0f%%", min_value=0, max_value=1),
                "pred_bet_outcome_proba": st.column_config.ProgressColumn("Over %", format="%.0f%%", min_value=0, max_value=1),
                "actual_home_score": st.column_config.NumberColumn("Actual Home", format="%.0f"),
                "actual_away_score": st.column_config.NumberColumn("Actual Away", format="%.0f"),
            },
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="multi-row",
        )

        selected_rows = event.selection.rows
        if selected_rows and "id" in df.columns:
            selected_ids = df.iloc[selected_rows]["id"].astype(int).tolist()
            label = f"Delete {len(selected_ids)} record{'s' if len(selected_ids) > 1 else ''}"
            if st.button(label, type="primary"):
                for pid in selected_ids:
                    database.delete_prediction(pid)
                load_prediction_history.clear()
                st.rerun()


# ── Page: Retrain / Refresh Data ──────────────────────────────────────────────
elif page == "Retrain / Refresh Data":
    st.title("Retrain / Refresh Data")

    last_run = load_last_training_run()
    if last_run:
        st.info(f"Last training run: {last_run.get('run_at', 'Unknown')} — {last_run.get('num_records', '?')} records")
    else:
        st.info("No training runs recorded yet.")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Refresh Data")
        st.caption("Pulls latest stats, odds, and referee data from all APIs.")
        if st.button("Refresh Data", type="secondary"):
            with st.spinner("Fetching data from APIs (this may take 10-30 minutes)..."):
                try:
                    from scripts.run_ingestion import run as run_ingestion
                    run_ingestion()
                    load_last_training_run.clear()
                    st.success("Data refreshed successfully.")
                except Exception as e:
                    st.error(f"Ingestion failed: {e}")

    with col2:
        st.subheader("Retrain Models")
        st.caption("Rebuilds the dataset from DB and retrains all 6 models.")
        if st.button("Retrain Models", type="primary"):
            with st.spinner("Training models (2-5 minutes)..."):
                try:
                    from scripts.run_training import run as run_training
                    run_training()
                    load_last_training_run.clear()
                    st.success("Models retrained successfully.")
                    st.balloons()
                except Exception as e:
                    st.error(f"Training failed: {e}")

    st.markdown("---")
    st.subheader("Run Both (Full Pipeline)")
    st.caption("Refresh data then immediately retrain. Use this for a full update.")
    if st.button("Refresh + Retrain", type="primary"):
        with st.spinner("Running full pipeline..."):
            try:
                from scripts.run_ingestion import run as run_ingestion
                from scripts.run_training import run as run_training
                run_ingestion()
                run_training()
                load_last_training_run.clear()
                st.success("Full pipeline complete.")
                st.balloons()
            except Exception as e:
                st.error(f"Pipeline failed: {e}")


# ── Page: Model Info ───────────────────────────────────────────────────────────
elif page == "Model Info":
    st.title("Model Info")

    last_run = load_last_training_run()

    if not last_run:
        st.info("No training runs found. Train the models first.")
    else:
        st.subheader("Last Training Run")
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Total Score RMSE", f"{last_run.get('modelts_rmse', 0):.2f} pts")
            st.metric("Away Score RMSE", f"{last_run.get('modelas_rmse', 0):.2f} pts")
            st.metric("Home Score RMSE", f"{last_run.get('modelhs_rmse', 0):.2f} pts")
        with m2:
            st.metric("Home Cover Accuracy", f"{last_run.get('logreg_home_accuracy', 0):.1%}")
            st.metric("Away Cover Accuracy", f"{last_run.get('logreg_away_accuracy', 0):.1%}")
            st.metric("Over/Under Accuracy", f"{last_run.get('logreg_bet_accuracy', 0):.1%}")
        with m3:
            st.metric("Training Records", f"{last_run.get('num_records', 0):,}")
            st.metric("Trained At", str(last_run.get("run_at", ""))[:16])

    st.markdown("---")
    st.subheader("Model Descriptions")
    st.markdown("""
| Model | Type | Target | Algorithm |
|---|---|---|---|
| `modelts` | Regression | Total combined score | XGBoost (n=400, depth=3, lr=0.1) |
| `modelas` | Regression | Away team score | XGBoost (n=300, depth=3, lr=0.1) |
| `modelhs` | Regression | Home team score | XGBoost (n=300, depth=3, lr=0.1) |
| `logreg_homecover` | Classification | Home team covers spread | Logistic Regression |
| `logreg_awaycover` | Classification | Away team covers spread | Logistic Regression |
| `logreg_betoutcome` | Classification | Over/Under outcome | Logistic Regression |
""")

    if models_exist():
        st.markdown("---")
        st.subheader("Feature Importance (XGBoost — Total Score)")
        try:
            import joblib
            import matplotlib.pyplot as plt
            from xgboost import plot_importance

            modelts = joblib.load(os.path.join(config.MODEL_DIR, "modelts.joblib"))
            fig, ax = plt.subplots(figsize=(8, 6))
            plot_importance(modelts, max_num_features=15, ax=ax)
            ax.set_title("Top 15 Features — Total Score Model")
            st.pyplot(fig)
        except Exception as e:
            st.caption(f"Feature importance chart unavailable: {e}")

    st.markdown("---")
    st.subheader("Data Sources")
    st.markdown("""
- **SportsData.io** — Stadium metadata, team game statistics (2020-2024), betting odds
- **nflpenalties.com** — Referee assignments (scraped)
- **FiveThirtyEight** — Team ELO and QB ELO ratings (with cached fallback)
- **Google Trends** — Team search interest (optional, disabled by default)
""")
