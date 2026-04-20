#!/usr/bin/env python3
"""
Statistical significance analysis: correlation + p-values for every feature
against each dependent variable.

Outputs:
  - Console: ranked significance tables per target
  - tests/significance_report.txt: full text report
  - tests/correlation_heatmap.png: heatmap of feature–target correlations

Targets tested:
  Regression  : TotalScore_x, HomeTeamScore, OpponentScore (AwayTeamScore)
  Classification: HomeTeamCover, AwayTeamCover, BetOutcome  (0=Cover 1=Push/Lose)

Method:
  Pearson r + two-sided p-value via scipy.stats.pearsonr for all targets.
  (Point-biserial for binary targets is mathematically equivalent to Pearson r.)
  Significance thresholds:  *** p<0.001  ** p<0.01  * p<0.05  . p<0.1
"""
import logging
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.WARNING)

import numpy as np
import pandas as pd
from scipy import stats

from nfl import config, database
from nfl.features.engineer import prepare_model_data
from nfl.features.merge import build_nfl_dataset

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

REGRESSION_TARGETS = ["TotalScore_x", "HomeTeamScore", "OpponentScore"]
CLASSIFICATION_TARGETS = ["HomeTeamCover", "AwayTeamCover", "BetOutcome"]
ALL_TARGETS = REGRESSION_TARGETS + CLASSIFICATION_TARGETS

# Columns that are targets or IDs — exclude from feature list
NON_FEATURES = set(ALL_TARGETS + [
    "HomeTeamName", "AwayTeamName", "GameID", "GameDate", "DateTime",
    "Season_x", "Season_y", "Week_x", "Week_y", "Team_x", "Team_y",
])

SIGNIFICANCE_STARS = [
    (0.001, "***"),
    (0.01,  "**"),
    (0.05,  "*"),
    (0.1,   "."),
    (1.0,   " "),
]


def stars(p: float) -> str:
    for threshold, label in SIGNIFICANCE_STARS:
        if p < threshold:
            return label
    return " "


def compute_significance(df: pd.DataFrame, features: list, target: str) -> pd.DataFrame:
    """Return DataFrame of (feature, r, p_value, stars) sorted by |r| desc."""
    y = df[target].dropna()
    rows = []
    for feat in features:
        if feat not in df.columns or feat == target:
            continue
        x = df[feat]
        combined = pd.concat([x, y], axis=1).dropna()
        if len(combined) < 10 or combined.iloc[:, 0].std() == 0:
            continue
        r, p = stats.pearsonr(combined.iloc[:, 0], combined.iloc[:, 1])
        rows.append({"feature": feat, "r": r, "p_value": p, "sig": stars(p)})

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.reindex(result["r"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def print_table(df: pd.DataFrame, target: str, n: int = 30):
    print(f"\n{'=' * 70}")
    print(f"  TARGET: {target}  (top {n} by |r|)")
    print(f"{'=' * 70}")
    if df.empty:
        print("  No results.")
        return
    print(f"  {'Feature':<32} {'r':>8}  {'p-value':>10}  Sig")
    print(f"  {'-'*32}  {'-'*8}  {'-'*10}  ---")
    for _, row in df.head(n).iterrows():
        print(f"  {row['feature']:<32} {row['r']:>8.4f}  {row['p_value']:>10.4e}  {row['sig']}")


def build_dataset() -> pd.DataFrame:
    """Load from DB, build merged + engineered dataset with all targets."""
    with database.managed_conn() as conn:
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

    # Align column names (same as run_training.py)
    col_renames = {"Season": "Season_x", "TotalScore": "TotalScore_x"}
    betting_odds.rename(columns={k: v for k, v in col_renames.items() if k in betting_odds.columns}, inplace=True)
    if "AwayTeamScore" in betting_odds.columns and "OpponentScore" not in betting_odds.columns:
        betting_odds["OpponentScore"] = betting_odds["AwayTeamScore"]

    dataset = build_nfl_dataset(
        teamstats=teamstats,
        stadiums=stadiums,
        betting_odds=betting_odds,
        referee_assignments=referee_assignments,
        elo_ratings=elo_ratings,
        google_trends=google_trends,
        weather_df=weather_df,
    )

    # Add classification targets (needed before significance tests)
    from nfl.features.engineer import _add_weather_features, _home_cover, _away_cover, _bet_outcome
    dataset = _add_weather_features(dataset.copy())
    dataset["HomeTeamCover"] = dataset.apply(_home_cover, axis=1).replace({"Cover": 0, "Push": 1, "Lose": 1})
    dataset["AwayTeamCover"] = dataset.apply(_away_cover, axis=1).replace({"Cover": 0, "Push": 1, "Lose": 1})
    dataset["BetOutcome"]    = dataset.apply(_bet_outcome, axis=1).replace({"Cover": 0, "Push": 1, "Lose": 1})

    # Numeric only
    dataset = dataset.select_dtypes(include="number")
    return dataset


def run():
    print("Loading dataset from database...")
    df = build_dataset()
    print(f"Dataset: {len(df)} rows × {len(df.columns)} columns")
    print(f"Targets available: {[t for t in ALL_TARGETS if t in df.columns]}")

    features = [c for c in df.columns if c not in NON_FEATURES]

    results = {}
    for target in ALL_TARGETS:
        if target not in df.columns:
            print(f"  Skipping {target} — not in dataset")
            continue
        results[target] = compute_significance(df, features, target)

    # ── Print to console ──────────────────────────────────────────────────────
    for target, res in results.items():
        print_table(res, target)

    # ── Summary: count significant features per target ──────────────────────
    print(f"\n{'=' * 70}")
    print("  SIGNIFICANCE SUMMARY")
    print(f"{'=' * 70}")
    print(f"  {'Target':<22} {'n_features':>10} {'p<0.05':>8} {'p<0.01':>8} {'p<0.001':>8}")
    print(f"  {'-'*22}  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*8}")
    for target, res in results.items():
        if res.empty:
            continue
        total = len(res)
        sig05 = (res["p_value"] < 0.05).sum()
        sig01 = (res["p_value"] < 0.01).sum()
        sig001 = (res["p_value"] < 0.001).sum()
        print(f"  {target:<22} {total:>10}  {sig05:>8}  {sig01:>8}  {sig001:>8}")

    # ── Weather feature spotlight ────────────────────────────────────────────
    weather_features = [
        "temperature", "wind_speed", "wind_direction", "precipitation",
        "snowfall", "weather_code", "is_dome",
        "high_wind", "is_precipitation", "freezing", "bad_weather_index",
    ]
    weather_present = [f for f in weather_features if f in df.columns]
    if weather_present:
        print(f"\n{'=' * 70}")
        print("  WEATHER FEATURE SIGNIFICANCE (all targets)")
        print(f"{'=' * 70}")
        print(f"  {'Feature':<22} {'Target':<22} {'r':>8}  {'p-value':>10}  Sig")
        print(f"  {'-'*22}  {'-'*22}  {'-'*8}  {'-'*10}  ---")
        for feat in weather_present:
            for target in ALL_TARGETS:
                if target not in results:
                    continue
                row = results[target][results[target]["feature"] == feat]
                if row.empty:
                    continue
                r = row.iloc[0]["r"]
                p = row.iloc[0]["p_value"]
                print(f"  {feat:<22}  {target:<22} {r:>8.4f}  {p:>10.4e}  {row.iloc[0]['sig']}")

    # ── Save text report ─────────────────────────────────────────────────────
    report_path = os.path.join(OUTPUT_DIR, "significance_report.txt")
    lines = []
    for target, res in results.items():
        lines.append(f"\n{'=' * 70}")
        lines.append(f"TARGET: {target}")
        lines.append(f"{'=' * 70}")
        if res.empty:
            lines.append("  No results.")
            continue
        lines.append(f"  {'Feature':<32} {'r':>8}  {'p-value':>10}  Sig")
        lines.append(f"  {'-'*32}  {'-'*8}  {'-'*10}  ---")
        for _, row in res.iterrows():
            lines.append(f"  {row['feature']:<32} {row['r']:>8.4f}  {row['p_value']:>10.4e}  {row['sig']}")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nFull report saved to: {report_path}")

    # ── Heatmap ──────────────────────────────────────────────────────────────
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        targets_present = [t for t in ALL_TARGETS if t in df.columns]
        # Pick top 25 features by max |r| across all targets
        max_r = pd.DataFrame({
            t: results[t].set_index("feature")["r"].abs()
            for t in targets_present if not results[t].empty
        }).max(axis=1).nlargest(25)
        top_features = max_r.index.tolist()

        heatmap_data = pd.DataFrame(index=top_features, columns=targets_present, dtype=float)
        pval_data    = pd.DataFrame(index=top_features, columns=targets_present, dtype=float)
        for target in targets_present:
            res = results[target].set_index("feature")
            for feat in top_features:
                if feat in res.index:
                    heatmap_data.loc[feat, target] = res.loc[feat, "r"]
                    pval_data.loc[feat, target]    = res.loc[feat, "p_value"]

        heatmap_data = heatmap_data.astype(float)
        pval_data    = pval_data.astype(float)

        # Annotation: r value + stars
        annot = heatmap_data.copy().astype(str)
        for feat in top_features:
            for target in targets_present:
                r = heatmap_data.loc[feat, target]
                p = pval_data.loc[feat, target]
                if pd.notna(r):
                    annot.loc[feat, target] = f"{r:.2f}{stars(p)}"
                else:
                    annot.loc[feat, target] = ""

        fig, ax = plt.subplots(figsize=(12, 14))
        sns.heatmap(
            heatmap_data,
            annot=annot,
            fmt="",
            cmap="RdYlGn",
            center=0,
            vmin=-1,
            vmax=1,
            linewidths=0.5,
            ax=ax,
            cbar_kws={"label": "Pearson r"},
        )
        ax.set_title(
            "Feature–Target Correlation (Pearson r)\n"
            "Stars: *** p<0.001  ** p<0.01  * p<0.05  . p<0.1",
            fontsize=13,
            pad=12,
        )
        ax.set_xlabel("Dependent Variable", fontsize=11)
        ax.set_ylabel("Feature", fontsize=11)
        plt.xticks(rotation=30, ha="right")
        plt.yticks(fontsize=9)
        plt.tight_layout()

        heatmap_path = os.path.join(OUTPUT_DIR, "correlation_heatmap.png")
        fig.savefig(heatmap_path, dpi=150)
        plt.close(fig)
        print(f"Correlation heatmap saved to: {heatmap_path}")
    except ImportError:
        print("seaborn not installed — skipping heatmap (pip install seaborn)")
    except Exception as e:
        print(f"Heatmap error: {e}")


if __name__ == "__main__":
    run()
