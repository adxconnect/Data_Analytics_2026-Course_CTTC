"""
Robust ML Training Pipeline — Global AI Jobs Salary Prediction
==============================================================
Trains multiple models, evaluates with cross-validation,
and exports artifacts for dashboard consumption.

Usage:
    python ml_pipeline.py
"""

import os
import json
import warnings
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
)

warnings.filterwarnings("ignore")

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "global_ai_jobs.csv")
ARTIFACT_DIR = os.path.join(BASE_DIR, "ml_artifacts")
os.makedirs(ARTIFACT_DIR, exist_ok=True)

TARGET = "salary_usd"
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Features that leak target information (derived from salary_usd)
LEAKY_FEATURES = {
    "salary_percentile",   # directly computed from salary
    "total_compensation",  # salary_usd + bonus_usd
    "bonus_ratio",         # bonus_usd / salary_usd
    "salary_to_col",       # salary_usd / cost_of_living_index
    "hourly_rate",         # salary_usd / annual_hours
}


# ═══════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING & VALIDATION
# ═══════════════════════════════════════════════════════════════════════════
def load_data(path: str) -> pd.DataFrame:
    """Load CSV and run basic validation."""
    log.info("Loading data from %s", path)
    df = pd.read_csv(path)
    log.info("Raw shape: %s rows × %s cols", *df.shape)

    # Drop unneeded ID column
    if "id" in df.columns:
        df.drop(columns=["id"], inplace=True)

    # Validate target exists
    assert TARGET in df.columns, f"Target column '{TARGET}' missing!"

    # Drop rows where target is null
    before = len(df)
    df.dropna(subset=[TARGET], inplace=True)
    if len(df) < before:
        log.warning("Dropped %d rows with null target", before - len(df))

    log.info("Clean shape: %s rows × %s cols", *df.shape)
    return df


# ═══════════════════════════════════════════════════════════════════════════
# 2. FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create derived features for better predictive power.
    NOTE: Features derived from salary_usd are intentionally excluded
    to prevent data leakage.
    """
    log.info("Engineering features …")
    df = df.copy()

    # Experience buckets
    if "experience_years" in df.columns:
        df["experience_bucket"] = pd.cut(
            df["experience_years"],
            bins=[-1, 2, 5, 10, 100],
            labels=["Junior", "Mid", "Senior", "Lead"],
        )

    # Weekly hours to annual hours (independent of salary)
    if "weekly_hours" in df.columns:
        df["annual_hours"] = df["weekly_hours"] * 52

    # Interaction: experience × company rating
    if "experience_years" in df.columns and "company_rating" in df.columns:
        df["exp_x_rating"] = df["experience_years"] * df["company_rating"]

    # Interaction: skill demand × ai adoption
    if "skill_demand_score" in df.columns and "ai_adoption_score" in df.columns:
        df["skill_x_ai"] = df["skill_demand_score"] * df["ai_adoption_score"]

    log.info("Feature engineering complete — %d cols", df.shape[1])
    return df


# ═══════════════════════════════════════════════════════════════════════════
# 3. OUTLIER HANDLING
# ═══════════════════════════════════════════════════════════════════════════
def cap_outliers(df: pd.DataFrame, col: str, factor: float = 1.5) -> pd.DataFrame:
    """Cap outliers using IQR method."""
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - factor * IQR, Q3 + factor * IQR
    before = ((df[col] < lower) | (df[col] > upper)).sum()
    df[col] = df[col].clip(lower, upper)
    log.info("Capped %d outliers in '%s' [%.0f – %.0f]", before, col, lower, upper)
    return df


# ═══════════════════════════════════════════════════════════════════════════
# 4. PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════
def preprocess(df: pd.DataFrame):
    """
    Encode categoricals, scale numericals, split into X / y.
    Returns: X_train, X_test, y_train, y_test, scaler, label_encoders, feature_names
    """
    log.info("Preprocessing …")
    df = df.copy()

    # Identify columns (exclude leaky features)
    cat_cols = [
        c for c in df.select_dtypes(include=["object", "category"]).columns
        if c not in LEAKY_FEATURES
    ]
    num_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns.tolist()
        if c != TARGET and c not in LEAKY_FEATURES
    ]
    log.info("Excluded leaky features: %s", LEAKY_FEATURES & set(df.columns))

    # Label-encode categoricals
    label_encoders: dict[str, LabelEncoder] = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    feature_cols = [c for c in cat_cols + num_cols if c != TARGET]
    X = df[feature_cols]
    y = df[TARGET]

    # Fill any remaining NaNs with median
    X = X.fillna(X.median())

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # Scale numerical features
    scaler = StandardScaler()
    num_in_features = [c for c in num_cols if c in feature_cols]
    X_train[num_in_features] = scaler.fit_transform(X_train[num_in_features])
    X_test[num_in_features] = scaler.transform(X_test[num_in_features])

    log.info(
        "Train: %d samples, Test: %d samples, Features: %d",
        len(X_train), len(X_test), len(feature_cols),
    )
    return X_train, X_test, y_train, y_test, scaler, label_encoders, feature_cols


# ═══════════════════════════════════════════════════════════════════════════
# 5. MODEL TRAINING & EVALUATION
# ═══════════════════════════════════════════════════════════════════════════
def evaluate_model(name: str, model, X_test, y_test) -> dict:
    """Compute regression metrics."""
    preds = model.predict(X_test)
    metrics = {
        "r2": round(r2_score(y_test, preds), 4),
        "mae": round(mean_absolute_error(y_test, preds), 2),
        "rmse": round(np.sqrt(mean_squared_error(y_test, preds)), 2),
        "mape": round(mean_absolute_percentage_error(y_test, preds) * 100, 2),
    }
    log.info(
        "  %-28s │ R²: %.4f │ MAE: %10.2f │ RMSE: %10.2f │ MAPE: %.2f%%",
        name, metrics["r2"], metrics["mae"], metrics["rmse"], metrics["mape"],
    )
    return metrics


def train_models(X_train, X_test, y_train, y_test):
    """Train multiple models, cross-validate, and pick the best one."""
    models = {
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            random_state=RANDOM_STATE,
        ),
        "Ridge Regression": Ridge(alpha=1.0),
    }

    results: dict[str, dict] = {}
    log.info("Training models …")
    print("\n" + "=" * 90)
    print(f"  {'Model':<28} │ {'R²':>7} │ {'MAE':>10} │ {'RMSE':>10} │ {'MAPE':>7}")
    print("─" * 90)

    for name, model in models.items():
        model.fit(X_train, y_train)

        test_metrics = evaluate_model(name, model, X_test, y_test)

        # Cross-validation
        cv_scores = cross_val_score(
            model, X_train, y_train,
            cv=CV_FOLDS, scoring="r2", n_jobs=-1,
        )
        test_metrics["cv_r2_mean"] = round(cv_scores.mean(), 4)
        test_metrics["cv_r2_std"] = round(cv_scores.std(), 4)

        results[name] = {"model": model, "metrics": test_metrics}

    print("=" * 90 + "\n")

    # Cross-validation summary
    log.info("Cross-Validation R² (5-fold):")
    for name, info in results.items():
        m = info["metrics"]
        log.info("  %-28s │ %.4f ± %.4f", name, m["cv_r2_mean"], m["cv_r2_std"])

    # Pick best model by test R²
    best_name = max(results, key=lambda k: results[k]["metrics"]["r2"])
    log.info("Best model: %s (R² = %.4f)", best_name, results[best_name]["metrics"]["r2"])

    return results, best_name


# ═══════════════════════════════════════════════════════════════════════════
# 6. FEATURE IMPORTANCE
# ═══════════════════════════════════════════════════════════════════════════
def get_feature_importance(model, feature_names: list[str]) -> pd.DataFrame:
    """Extract feature importances (works for tree-based models)."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_)
    else:
        return pd.DataFrame()

    fi = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    fi["importance_pct"] = (fi["importance"] / fi["importance"].sum() * 100).round(2)
    return fi


# ═══════════════════════════════════════════════════════════════════════════
# 7. EXPORT ARTIFACTS
# ═══════════════════════════════════════════════════════════════════════════
def export_artifacts(
    best_model,
    best_name: str,
    results: dict,
    scaler,
    label_encoders,
    feature_names: list[str],
    feature_importance: pd.DataFrame,
):
    """Save all training artifacts for dashboard consumption."""
    log.info("Exporting artifacts to %s", ARTIFACT_DIR)

    # 1. Best model
    joblib.dump(best_model, os.path.join(ARTIFACT_DIR, "best_model.pkl"))

    # 2. Scaler
    joblib.dump(scaler, os.path.join(ARTIFACT_DIR, "scaler.pkl"))

    # 3. Label encoders
    joblib.dump(label_encoders, os.path.join(ARTIFACT_DIR, "label_encoders.pkl"))

    # 4. Feature importance
    feature_importance.to_csv(
        os.path.join(ARTIFACT_DIR, "feature_importance.csv"), index=False
    )

    # 5. Metrics (all models)
    metrics_out = {
        "trained_at": datetime.now().isoformat(),
        "best_model": best_name,
        "test_size": TEST_SIZE,
        "cv_folds": CV_FOLDS,
        "models": {},
    }
    for name, info in results.items():
        metrics_out["models"][name] = info["metrics"]
    metrics_out["best_model_metrics"] = results[best_name]["metrics"]

    with open(os.path.join(ARTIFACT_DIR, "metrics.json"), "w") as f:
        json.dump(metrics_out, f, indent=2)

    # 6. Column list
    with open(os.path.join(ARTIFACT_DIR, "columns.json"), "w") as f:
        json.dump({"features": feature_names, "target": TARGET}, f, indent=2)

    log.info("✓ All artifacts saved successfully")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════
def main():
    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║    Global AI Jobs — Robust ML Training Pipeline            ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()

    # 1. Load
    df = load_data(DATA_PATH)

    # 2. Feature engineering
    df = engineer_features(df)

    # 3. Outlier handling
    df = cap_outliers(df, TARGET)

    # 4. Preprocess
    X_train, X_test, y_train, y_test, scaler, label_encoders, feature_names = preprocess(df)

    # 5. Train & evaluate
    results, best_name = train_models(X_train, X_test, y_train, y_test)
    best_model = results[best_name]["model"]

    # 6. Feature importance
    fi = get_feature_importance(best_model, feature_names)
    if not fi.empty:
        print("\n🏆 Top 15 Feature Importances:")
        print(fi.head(15).to_string(index=False))
        print()

    # 7. Export
    export_artifacts(
        best_model, best_name, results,
        scaler, label_encoders, feature_names, fi,
    )

    print()
    print("─" * 60)
    print(f"  ✅ Pipeline complete! Best model: {best_name}")
    print(f"     R² = {results[best_name]['metrics']['r2']}")
    print(f"     Artifacts saved to:  {ARTIFACT_DIR}")
    print("─" * 60)
    print()


if __name__ == "__main__":
    main()
