# NFL Game Prediction Model

A machine learning system that predicts NFL game outcomes — scores, spread coverage, and over/under — deployed as a Streamlit web application.

---

## Overview

This project pulls data from four live sources, merges them into a single dataset, trains six machine learning models, and exposes predictions through an interactive Streamlit app. You can predict any upcoming game by entering the matchup details and Vegas lines, then see predicted scores, the likely winner, spread coverage probabilities, and over/under direction.

### What It Predicts

| Output | Model | Type |
|---|---|---|
| Total combined score | XGBoost Regressor (`modelts`) | Regression |
| Away team score | XGBoost Regressor (`modelas`) | Regression |
| Home team score | XGBoost Regressor (`modelhs`) | Regression |
| Predicted winner | Derived from home vs. away score | — |
| Home team covers spread | Logistic Regression (`logreg_homecover`) | Classification |
| Away team covers spread | Logistic Regression (`logreg_awaycover`) | Classification |
| Over/Under result | Logistic Regression (`logreg_betoutcome`) | Classification |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Data Sources                           │
│  SportsData.io API  │  nflpenalties.com  │  538 ELO CSV    │
│  (Stadium, Stats,   │  (Referee          │  (Team ELO,     │
│   Betting Odds)     │   assignments)     │   QB ELO)       │
│                     │                    │                  │
│          Google Trends (optional)        │  nfl_data_py    │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
              ┌──────────────────┐
              │  Ingestion Layer │   nfl/ingestion/
              │  (Python modules │
              │   → SQLite DB)   │
              └────────┬─────────┘
                       │
                       ▼
              ┌──────────────────┐
              │  Feature Layer   │   nfl/features/
              │  Merge → Engineer│
              │  → Season Avgs   │
              └────────┬─────────┘
                       │
                       ▼
              ┌──────────────────┐
              │  Training Layer  │   nfl/training/train.py
              │  6 Models        │
              │  → .joblib files │
              └────────┬─────────┘
                       │
                       ▼
              ┌──────────────────┐
              │ Prediction Layer │   nfl/prediction/predict.py
              │  Input dict →    │
              │  Feature vector  │
              │  → 7 outputs     │
              └────────┬─────────┘
                       │
                       ▼
              ┌──────────────────┐
              │  Streamlit App   │   app.py
              │  4-page UI       │
              │  Streamlit Cloud │
              └──────────────────┘
```

---

## Directory Structure

```
NFL_Prediction_Model/
├── app.py                          # Streamlit entry point
├── requirements.txt
├── .env                            # gitignored — local dev secrets
├── .env.example                    # committed env template
├── .gitignore
│
├── .streamlit/
│   └── secrets.toml                # gitignored — Streamlit Cloud secrets
│
├── data/
│   ├── nfl_predictions.db          # SQLite database (gitignored, regenerated)
│   └── nfl_elo_cache.csv           # Committed fallback snapshot for 538 ELO data
│
├── models/
│   ├── modelts.joblib              # XGBRegressor — TotalScore
│   ├── modelas.joblib              # XGBRegressor — AwayScore
│   ├── modelhs.joblib              # XGBRegressor — HomeScore
│   ├── logreg_homecover.joblib     # LogisticRegression — HomeTeamCover
│   ├── logreg_awaycover.joblib     # LogisticRegression — AwayTeamCover
│   ├── logreg_betoutcome.joblib    # LogisticRegression — BetOutcome
│   └── feature_columns.joblib      # Ordered training feature column list
│
├── nfl/                            # Core Python package
│   ├── __init__.py
│   ├── config.py                   # Secrets/config loading
│   ├── database.py                 # SQLite schema + CRUD helpers
│   ├── ingestion/
│   │   ├── sportsdata.py           # SportsData.io API calls
│   │   ├── referee_scraper.py      # nflpenalties.com scraping
│   │   ├── fivethirtyeight.py      # 538 ELO CSV with fallback
│   │   ├── google_trends.py        # pytrends (optional, default off)
│   │   └── nfl_schedule.py         # nfl_data_py schedule import
│   ├── features/
│   │   ├── merge.py                # 6-step merge chain
│   │   ├── engineer.py             # Drop columns, cover targets, fillna
│   │   └── season_averages.py      # Tiered stat vectors for new games
│   ├── training/
│   │   └── train.py                # Train + save all 6 models
│   └── prediction/
│       └── predict.py              # predict_game() → full result dict
│
└── scripts/
    ├── run_ingestion.py            # CLI: pull all data → SQLite
    └── run_training.py             # CLI: train from DB → models/
```

---

## Data Sources

### 1. SportsData.io API
**Endpoint base:** `https://api.sportsdata.io/api/nfl/odds/json/`

| Data | Endpoint | Seasons |
|---|---|---|
| Stadium metadata | `/Stadiums` | Static |
| Team game stats | `/TeamGameStats/{YEAR}{TYPE}/{WEEK}` | 2021–2024 REG + POST |
| Betting odds | `/GameOddsByWeek/{YEAR}/{WEEK}` | 2021–2024 |

Extracted betting fields: `HomeMoneyLine`, `AwayMoneyLine`, `HomePointSpread`, `AwayPointSpread`, `OverUnder` and associated payouts.

### 2. nflpenalties.com (Web Scraping)
Referee assignments scraped from `https://www.nflpenalties.com/referee/<name>` for 18 active referees across seasons 2020–2024:

Brad-Allen, Tra-Blake, Clete-Blakeman, Alan-Eck, Jerome-Boger, Carl-Cheffers, Land-Clark, Adrian-Hill, Shawn-Hochuli, John-Hussey, Alex-Kemp, Clay-Martin, Scott-Novak, Brad-Rogers, Shawn-Smith, Ron-Torbert, Bill-Vinovich, Craig-Wrolstad

### 3. FiveThirtyEight ELO Ratings
**URL:** `https://projects.fivethirtyeight.com/nfl-api/nfl_elo.csv`

Features used: `elo1_pre`, `elo2_pre`, `qbelo1_pre`, `qbelo2_pre`, `neutral`

A committed snapshot (`data/nfl_elo_cache.csv`) is used as a fallback if the URL is unavailable.

### 4. Google Trends (Optional)
Team search interest via `pytrends`. **Disabled by default** (`GOOGLE_TRENDS_ENABLED=false`) due to rate limiting. When unavailable, trend features default to 0.

---

## Models

### Regression Models (XGBoost)

| Model | Target | n_estimators | max_depth | learning_rate | gamma | min_child_weight |
|---|---|---|---|---|---|---|
| `modelts` | Total Score | 400 | 3 | 0.1 | 0.5 | 5 |
| `modelas` | Away Score | 300 | 3 | 0.1 | 1.0 | 7 |
| `modelhs` | Home Score | 300 | 3 | 0.1 | 0.0 | 1 |

### Classification Models (Logistic Regression)

| Model | Target | Encoding | max_iter |
|---|---|---|---|
| `logreg_homecover` | Home team covers spread | Cover=0, Push/Lose=1 | 10000 |
| `logreg_awaycover` | Away team covers spread | Cover=0, Push/Lose=1 | 10000 |
| `logreg_betoutcome` | Over/Under outcome | Over=0, Push/Under=1 | 10000 |

### Cover Definitions
- **HomeTeamCover**: Home spread < 0 and home margin > |spread|, OR home spread > 0 and loss margin < spread
- **AwayTeamCover**: Mirror of above for away team
- **BetOutcome**: Total score > OverUnder = Over (Cover); equal = Push; less = Under (Lose)

### Feature Set
Training drops these columns from the merged dataset:
`HomeTeamScore`, `TotalScore_x`, `OpponentScore`, `DateTime`, `AwayTeamName`, `HomeTeamName`
(Classification models also drop: `BetOutcome`, `HomeTeamCover`, `AwayTeamCover`)

Train/test split: 80/20, `random_state=1234`

---

## Predicting Future Games

The model was trained on per-game rows that include the home team's actual game statistics. For future games, those stats don't exist yet. The system uses a **tiered stat vector strategy**:

| Tier | Condition | Strategy |
|---|---|---|
| 1 | ≥5 games played this season | Rolling last-5-game average |
| 2 | 1–4 games played this season | Full current-season average |
| 3 | Week 1 (no current-season data) | Prior season average |

The app displays a warning when Tier 3 is used, since accuracy is lower.

---

## Database Schema (SQLite)

| Table | Purpose |
|---|---|
| `stadiums` | Stadium reference data with numeric surface/type encodings |
| `team_game_stats` | Raw per-team per-game stats from API |
| `team_season_averages` | Precomputed season averages (rebuilt on ingestion) |
| `betting_odds` | Game odds with home/away scores |
| `referee_assignments` | Referee-to-game mapping |
| `elo_ratings` | 538 ELO ratings per game |
| `google_trends` | Team weekly search interest |
| `predictions` | Full prediction history with nullable actual results |
| `training_runs` | Model training timestamps and performance metrics |

---

## Streamlit App

Four pages accessed via sidebar navigation:

### 1. Predict a Game
Enter home/away teams, game date, stadium, and Vegas lines. Advanced expander provides referee and ELO inputs (auto-populated from DB when available). Outputs:
- Predicted home score, away score, total score
- Predicted winner
- Home cover probability, away cover probability, over/under direction with confidence

### 2. Prediction History
Filterable table of all predictions logged to the database. Nullable columns for actual results allow post-game accuracy tracking.

### 3. Retrain / Refresh Data
- **Refresh Data**: Re-runs the ingestion pipeline against live APIs
- **Retrain Models**: Retrains all 6 models on the latest database contents
- Shows timestamps from the last ingestion and training runs

### 4. Model Info
Training metrics (MSE, RMSE, accuracy) from the last training run, XGBoost feature importance charts, and model descriptions.

---

## Setup

### Prerequisites
- Python 3.10+
- SportsData.io API key (free tier covers the required endpoints)

### Local Installation

```bash
git clone <repo-url>
cd NFL_Prediction_Model
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Environment Variables

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

```
# .env
SPORTSDATA_API_KEY=your_key_here
DB_PATH=data/nfl_predictions.db
MODEL_DIR=models
GOOGLE_TRENDS_ENABLED=false
```

### Initialize Data and Train Models

```bash
# Step 1: Pull all data from APIs and populate SQLite
python scripts/run_ingestion.py

# Step 2: Train 6 models and save to models/
python scripts/run_training.py
```

The ingestion step takes 10–30 minutes (rate-limited API calls and web scraping). Subsequent runs only fetch new data.

### Run Locally

```bash
streamlit run app.py
```

Navigate to `http://localhost:8501`.

---

## Deployment to Streamlit Cloud

### 1. Prepare the Repository

Ensure `.gitignore` excludes:
```
.env
.streamlit/secrets.toml
data/nfl_predictions.db
**/__pycache__
*.pickle
```

Commit the initial trained models and ELO cache:
```bash
git add models/*.joblib data/nfl_elo_cache.csv
git commit -m "Add initial trained models and ELO cache"
git push
```

> For large model files (>100 MB), use Git LFS: `git lfs track "models/*.joblib"`

### 2. Create Streamlit Cloud App

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click **New app** → select your GitHub repo
3. Set **Main file path** to `app.py`
4. Under **Advanced settings → Secrets**, add:

```toml
SPORTSDATA_API_KEY = "your_actual_key"
GOOGLE_TRENDS_ENABLED = "false"

# For persistent database, add a Supabase connection string:
DATABASE_URL = "postgresql://user:password@host:5432/dbname"
```

5. Click **Deploy**

### 3. Database Persistence on Streamlit Cloud

Streamlit Cloud has an ephemeral filesystem — SQLite is wiped on restart. Options:

- **Supabase (recommended)**: Free PostgreSQL tier. `database.py` detects `DATABASE_URL` and switches from SQLite to PostgreSQL automatically.
- **Git LFS**: Commit the pre-built SQLite file. The app reads it as read-only for historical data.
- **Session storage**: Predictions are stored in-memory per session (simplest, but no history persistence).

---

## Requirements

```
streamlit>=1.32.0
pandas>=2.0.0
numpy>=1.24.0
xgboost>=2.0.0
scikit-learn>=1.3.0
requests>=2.31.0
beautifulsoup4>=4.12.0
nfl_data_py>=0.3.0
pytrends>=4.9.0
python-dotenv>=1.0.0
joblib>=1.3.0
psycopg2-binary>=2.9.0
```

---

## Known Limitations

- **Week 1 predictions** use prior-season team averages, which are less accurate for teams with significant off-season changes (coaching staff, key players).
- **Google Trends** is disabled by default due to rate limiting; predictions default trend features to 0.
- **FiveThirtyEight ELO**: The live URL may be down; a cached CSV snapshot from 2024 is used as a fallback, meaning ELO values won't reflect the most recent seasons until the source is restored.
- **Referee data**: The 18 referees listed cover the primary crews but new or substitute referees will map to a default value.

---

## Security Note

Never commit `.env` or `.streamlit/secrets.toml`. The SportsData.io API key is loaded exclusively from environment variables or Streamlit secrets at runtime.
