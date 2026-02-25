"""
AnomalyWatchers-DonutPuff — Unified ML Pipeline
=================================================
COS30049 Assignment 2: Tri-Model Fraud Detection Architecture

Pipeline:
    Phase 1  ─  Data Engineering  (Load, dtype-optimize, stratified sample)
    Phase 2  ─  Feature Engineering  (error-balance features, label encoding)
    Phase 3  ─  Imbalance Handling  (SMOTE with documented justification)
    Phase 4  ─  Tri-Model Training
                  ├── Model 1: Logistic Regression  (Baseline)
                  ├── Model 2: XGBoost Classifier   (Champion, GridSearchCV)
                  └── Model 3: Isolation Forest      (Unsupervised anomaly)
    Phase 5  ─  Evaluation  (AUPRC, Confusion Matrix, Classification Report)
    Phase 6  ─  Serialization  (joblib → backend/models/)

Datasets:
    Primary  (Trainer)  : Rupak Roy  — Online Payments Fraud Detection (Paysim)
    Secondary (Validator): Kartik2112 — Fraud Detection (credit-card domain)

Usage:
    python scripts/train_models.py [--sample-frac 0.1] [--full]

Authors: AnomalyWatchers (DonutPuff)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder

try:
    import xgboost as xgb
except ImportError:
    raise SystemExit("[FATAL] XGBoost not installed. Run:  pip install xgboost")

try:
    from imblearn.over_sampling import SMOTE
except ImportError:
    raise SystemExit(
        "[FATAL] imbalanced-learn not installed. Run:  pip install imbalanced-learn"
    )

# ---------------------------------------------------------------------------
# Logging Configuration (Rule 8: No print() — structured logging)
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("anomaly_watchers.pipeline")

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Constants (Rule 5: No magic numbers)
# ---------------------------------------------------------------------------
RANDOM_STATE: int = 42
TEST_SIZE: float = 0.2

FEATURE_COLUMNS: List[str] = [
    "type",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "errorBalanceOrg",
    "errorBalanceDest",
]
TARGET_COLUMN: str = "isFraud"

# Fraud-capable transaction types in the Paysim dataset
FRAUD_CAPABLE_TYPES: List[str] = ["CASH_OUT", "TRANSFER"]

# Paths
DATA_DIR = Path("data")
MODEL_DIR = Path("backend") / "models"
PRIMARY_DATASET = DATA_DIR / "onlinefraud.csv"

# dtype map for memory-efficient loading
DTYPE_MAP = {
    "step": "int32",
    "type": "category",
    "amount": "float32",
    "nameOrig": "object",
    "oldbalanceOrg": "float32",
    "newbalanceOrig": "float32",
    "nameDest": "object",
    "oldbalanceDest": "float32",
    "newbalanceDest": "float32",
    "isFraud": "int8",
    "isFlaggedFraud": "int8",
}


# ---------------------------------------------------------------------------
# Data Transfer Objects (Rule 2: Immutable DTOs)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ModelResult:
    """Immutable container for a single model's evaluation metrics."""

    name: str
    model: object
    predictions: np.ndarray
    probabilities: Optional[np.ndarray]
    auprc: Optional[float]
    f1: float
    best_params: Optional[Dict] = field(default=None)


# ---------------------------------------------------------------------------
# Phase 1: Data Engineering
# ---------------------------------------------------------------------------
def load_primary_dataset(filepath: Path, sample_frac: float = 0.1) -> pd.DataFrame:
    """
    Load the Rupak Roy (Paysim) dataset with memory-optimized dtypes.

    Uses stratified sampling to preserve fraud/legit ratio when
    ``sample_frac < 1.0`` (development mode).

    Args:
        filepath: Path to the ``onlinefraud.csv`` file.
        sample_frac: Fraction of *legitimate* transactions to retain.
                     All fraudulent rows are always kept.

    Returns:
        A pandas DataFrame ready for feature engineering.

    Raises:
        FileNotFoundError: If the CSV is missing from ``data/``.
    """
    logger.info("=" * 60)
    logger.info("PHASE 1: DATA ENGINEERING")
    logger.info("=" * 60)

    if not filepath.exists():
        raise FileNotFoundError(
            f"Dataset not found: {filepath}. "
            "Run `python scripts/download_data.py` to fetch the CSV."
        )

    logger.info("[1.1] Loading dataset: %s", filepath)
    df = pd.read_csv(filepath, dtype=DTYPE_MAP)
    logger.info("   Total records: %s", f"{len(df):,}")
    logger.info("   Memory usage: %.1f MB", df.memory_usage().sum() / 1e6)

    # Stratified sampling — keep ALL fraud rows, sample legit rows
    if sample_frac < 1.0:
        logger.info(
            "[1.2] Stratified sampling %.0f%% of legit data...",
            sample_frac * 100,
        )
        df_fraud = df[df[TARGET_COLUMN] == 1]
        df_legit = df[df[TARGET_COLUMN] == 0].sample(
            frac=sample_frac, random_state=RANDOM_STATE
        )
        df = (
            pd.concat([df_fraud, df_legit])
            .sample(frac=1, random_state=RANDOM_STATE)
            .reset_index(drop=True)
        )
        logger.info("   Sampled records: %s", f"{len(df):,}")

    fraud_rate = df[TARGET_COLUMN].mean()
    logger.info("[1.3] Class Distribution:")
    logger.info("   Legitimate: %.2f%%", (1 - fraud_rate) * 100)
    logger.info("   Fraudulent: %.4f%%", fraud_rate * 100)

    return df


# ---------------------------------------------------------------------------
# Phase 2: Feature Engineering
# ---------------------------------------------------------------------------
def engineer_features(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, LabelEncoder]:
    """
    Construct model-ready features from the raw Paysim data.

    Key transforms:
        * Filter to CASH_OUT / TRANSFER (only types with fraud).
        * ``errorBalanceOrg``  = newbalanceOrig + amount − oldbalanceOrg
        * ``errorBalanceDest`` = oldbalanceDest + amount − newbalanceDest
        * Label-encode the ``type`` column.

    Args:
        df: Raw DataFrame from :func:`load_primary_dataset`.

    Returns:
        Tuple of (features DataFrame, fitted LabelEncoder).
    """
    logger.info("=" * 60)
    logger.info("PHASE 2: FEATURE ENGINEERING")
    logger.info("=" * 60)

    df = df.copy()

    # Filter to fraud-capable transaction types
    logger.info("[2.1] Filtering to %s...", FRAUD_CAPABLE_TYPES)
    df = df[df["type"].isin(FRAUD_CAPABLE_TYPES)]
    logger.info("   Records after filter: %s", f"{len(df):,}")

    # Error-balance features — key discriminators for fraud
    # Why: In legitimate transactions these features ≈ 0; in fraud the
    # balance math is inconsistent, producing non-zero error signals.
    logger.info("[2.2] Creating error-balance features...")
    df["errorBalanceOrg"] = df["newbalanceOrig"] + df["amount"] - df["oldbalanceOrg"]
    df["errorBalanceDest"] = df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]

    # Label encode transaction type
    logger.info("[2.3] Label-encoding 'type'...")
    le_type = LabelEncoder()
    df["type"] = le_type.fit_transform(df["type"])
    logger.info("   Classes: %s", list(le_type.classes_))

    # Final feature selection
    features = df[FEATURE_COLUMNS + [TARGET_COLUMN]].copy()
    missing = features.isnull().sum().sum()
    logger.info(
        "[2.4] Final feature set — %d columns, %d missing",
        len(FEATURE_COLUMNS),
        missing,
    )

    return features, le_type


# ---------------------------------------------------------------------------
# Phase 3: Imbalance Handling (SMOTE)
# ---------------------------------------------------------------------------
def apply_smote(
    X_train: np.ndarray, y_train: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply SMOTE (Synthetic Minority Over-sampling Technique) to balance
    the training set.

    Justification (Assignment 2 requirement):
        The Rupak Roy dataset exhibits extreme class imbalance (~0.13% fraud).
        SMOTE generates synthetic fraud examples by interpolating between
        existing minority-class neighbours, preventing the model from
        learning a trivial "always predict legit" strategy.  We apply SMOTE
        *only* to training data to avoid data leakage into validation.

    Args:
        X_train: Feature matrix (train split only).
        y_train: Target vector (train split only).

    Returns:
        Tuple of (X_resampled, y_resampled).
    """
    logger.info("=" * 60)
    logger.info("PHASE 3: IMBALANCE HANDLING (SMOTE)")
    logger.info("=" * 60)

    logger.info("[3.0] Pre-SMOTE fraud ratio: %.4f%%", y_train.mean() * 100)

    smote = SMOTE(random_state=RANDOM_STATE)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    logger.info("[3.1] Post-SMOTE train size: %s", f"{len(X_resampled):,}")
    logger.info(
        "[3.2] Post-SMOTE fraud ratio: %.2f%%",
        y_resampled.mean() * 100,
    )

    return X_resampled, y_resampled


# ---------------------------------------------------------------------------
# Phase 4: Tri-Model Architecture
# ---------------------------------------------------------------------------
def train_tri_models(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> List[ModelResult]:
    """
    Train the three required models and return their results.

    Models:
        1. **Logistic Regression** (Baseline) — interpretable linear model.
        2. **XGBoost** (Champion) — gradient-boosted trees with GridSearchCV.
        3. **Isolation Forest** (Unsupervised) — anomaly detector.

    Args:
        X_train: Balanced training features.
        X_test:  Hold-out test features.
        y_train: Balanced training labels.
        y_test:  Hold-out test labels.

    Returns:
        List of :class:`ModelResult` objects.
    """
    logger.info("=" * 60)
    logger.info("PHASE 4: TRI-MODEL ARCHITECTURE")
    logger.info("=" * 60)

    results: List[ModelResult] = []

    # ------------------------------------------------------------------
    # Model 1: Logistic Regression (Baseline)
    # ------------------------------------------------------------------
    logger.info("[4.1] Model 1 — Logistic Regression (Baseline)")
    lr = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=RANDOM_STATE,
        solver="lbfgs",
    )
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    y_prob_lr = lr.predict_proba(X_test)[:, 1]

    lr_result = ModelResult(
        name="logistic_regression",
        model=lr,
        predictions=y_pred_lr,
        probabilities=y_prob_lr,
        auprc=average_precision_score(y_test, y_prob_lr),
        f1=f1_score(y_test, y_pred_lr),
    )
    results.append(lr_result)
    logger.info("   AUPRC: %.4f  |  F1: %.4f", lr_result.auprc, lr_result.f1)

    # ------------------------------------------------------------------
    # Model 2: XGBoost (Champion) — GridSearchCV
    # ------------------------------------------------------------------
    logger.info("[4.2] Model 2 — XGBoost (Champion)")

    scale_pos_weight = float((y_train == 0).sum()) / max(float((y_train == 1).sum()), 1)
    logger.info("   scale_pos_weight: %.2f", scale_pos_weight)

    param_grid = {
        "max_depth": [4, 6],
        "learning_rate": [0.1, 0.3],
        "n_estimators": [100, 200],
        "scale_pos_weight": [scale_pos_weight],
    }

    xgb_base = xgb.XGBClassifier(
        random_state=RANDOM_STATE,
        verbosity=0,
        eval_metric="aucpr",
        use_label_encoder=False,
    )

    logger.info("   Running GridSearchCV (scoring=average_precision)...")
    grid_search = GridSearchCV(
        xgb_base,
        param_grid,
        cv=3,
        scoring="average_precision",
        n_jobs=-1,
        verbose=0,
    )
    grid_search.fit(X_train, y_train)

    best_xgb = grid_search.best_estimator_
    y_pred_xgb = best_xgb.predict(X_test)
    y_prob_xgb = best_xgb.predict_proba(X_test)[:, 1]

    xgb_result = ModelResult(
        name="xgboost",
        model=best_xgb,
        predictions=y_pred_xgb,
        probabilities=y_prob_xgb,
        auprc=average_precision_score(y_test, y_prob_xgb),
        f1=f1_score(y_test, y_pred_xgb),
        best_params=grid_search.best_params_,
    )
    results.append(xgb_result)
    logger.info("   Best params: %s", grid_search.best_params_)
    logger.info("   AUPRC: %.4f  |  F1: %.4f", xgb_result.auprc, xgb_result.f1)

    # ------------------------------------------------------------------
    # Model 3: Isolation Forest (Unsupervised Anomaly Detector)
    # ------------------------------------------------------------------
    logger.info("[4.3] Model 3 — Isolation Forest (Anomaly Detector)")

    # Train only on legitimate transactions (unsupervised paradigm)
    X_legit = X_train[y_train == 0]
    logger.info("   Training on %s legitimate samples", f"{len(X_legit):,}")

    iso_forest = IsolationForest(
        n_estimators=100,
        contamination=0.01,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    iso_forest.fit(X_legit)

    # Isolation Forest returns -1 for anomalies, 1 for inliers
    y_pred_iso_raw = iso_forest.predict(X_test)
    y_pred_iso = np.where(y_pred_iso_raw == -1, 1, 0)

    iso_result = ModelResult(
        name="isolation_forest",
        model=iso_forest,
        predictions=y_pred_iso,
        probabilities=None,
        auprc=None,
        f1=f1_score(y_test, y_pred_iso),
    )
    results.append(iso_result)
    logger.info("   F1: %.4f", iso_result.f1)

    return results


# ---------------------------------------------------------------------------
# Phase 5: Evaluation
# ---------------------------------------------------------------------------
def evaluate_models(results: List[ModelResult], y_test: np.ndarray) -> None:
    """
    Print AUPRC comparison table and XGBoost confusion matrix.

    Primary metric: **AUPRC** (Area Under Precision-Recall Curve).
    Rationale: Standard accuracy is misleading under severe imbalance
    (~99.87% legit). AUPRC focuses on minority-class performance.

    Args:
        results: List of :class:`ModelResult` from training.
        y_test:  Ground-truth labels for the test split.
    """
    logger.info("=" * 60)
    logger.info("PHASE 5: EVALUATION")
    logger.info("=" * 60)

    logger.info("")
    logger.info("%-25s %-10s %-10s", "Model", "AUPRC", "F1-Score")
    logger.info("-" * 50)

    for res in results:
        auprc_str = f"{res.auprc:.4f}" if res.auprc is not None else "N/A"
        logger.info("%-25s %-10s %-10.4f", res.name, auprc_str, res.f1)

    # Champion model (XGBoost) confusion matrix
    xgb_result = next(r for r in results if r.name == "xgboost")
    cm = confusion_matrix(y_test, xgb_result.predictions)
    logger.info("")
    logger.info("Champion (XGBoost) Confusion Matrix:")
    logger.info("   True Negatives:  %s", f"{cm[0, 0]:,}")
    logger.info("   False Positives: %s  (unnecessary blocks)", f"{cm[0, 1]:,}")
    logger.info("   False Negatives: %s  (missed fraud — CRITICAL)", f"{cm[1, 0]:,}")
    logger.info("   True Positives:  %s", f"{cm[1, 1]:,}")
    logger.info("")
    logger.info("Classification Report (XGBoost):")
    report = classification_report(
        y_test,
        xgb_result.predictions,
        target_names=["Legitimate", "Fraud"],
    )
    for line in report.splitlines():
        logger.info("   %s", line)


# ---------------------------------------------------------------------------
# Phase 6: Serialization
# ---------------------------------------------------------------------------
def save_models(
    results: List[ModelResult],
    le_type: LabelEncoder,
    model_dir: Path,
) -> None:
    """
    Serialize models to disk for FastAPI backend consumption.

    Artifacts:
        - ``model_primary.pkl``        — XGBoost champion
        - ``model_logistic.pkl``       — Logistic Regression baseline
        - ``model_isolation_forest.pkl``— Isolation Forest
        - ``label_encoder_type.pkl``   — Transaction type encoder

    Args:
        results:   List of :class:`ModelResult` from training.
        le_type:   Fitted :class:`LabelEncoder` for the ``type`` column.
        model_dir: Output directory path.
    """
    logger.info("=" * 60)
    logger.info("PHASE 6: SERIALIZATION")
    logger.info("=" * 60)

    model_dir.mkdir(parents=True, exist_ok=True)

    artifact_map = {
        "xgboost": "model_primary.pkl",
        "logistic_regression": "model_logistic.pkl",
        "isolation_forest": "model_isolation_forest.pkl",
    }

    for res in results:
        filename = artifact_map.get(res.name)
        if filename:
            path = model_dir / filename
            joblib.dump(res.model, path)
            logger.info("   Saved %s → %s", res.name, path)

    encoder_path = model_dir / "label_encoder_type.pkl"
    joblib.dump(le_type, encoder_path)
    logger.info("   Saved LabelEncoder → %s", encoder_path)

    logger.info("")
    logger.info("✅ All artifacts saved to %s/", model_dir)


# ---------------------------------------------------------------------------
# CLI & Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="AnomalyWatchers — Tri-Model Fraud Detection Pipeline"
    )
    parser.add_argument(
        "--sample-frac",
        type=float,
        default=0.1,
        help="Fraction of legitimate data to sample (default: 0.1 for dev)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Use full dataset (overrides --sample-frac to 1.0)",
    )
    return parser.parse_args()


def main() -> None:
    """
    Execute the complete ML pipeline end-to-end.

    Steps:
        1. Parse CLI args
        2. Load and sample data  (Phase 1)
        3. Engineer features     (Phase 2)
        4. Apply SMOTE           (Phase 3)
        5. Train tri-models      (Phase 4)
        6. Evaluate              (Phase 5)
        7. Serialize             (Phase 6)
    """
    args = parse_args()
    sample_frac = 1.0 if args.full else args.sample_frac

    logger.info("")
    logger.info("=" * 60)
    logger.info("ANOMALYWATCHERS — UNIFIED ML PIPELINE")
    logger.info("Timestamp: %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logger.info(
        "Mode: %s",
        "FULL" if sample_frac >= 1.0 else f"SAMPLE ({sample_frac * 100:.0f}%)",
    )
    logger.info("=" * 60)

    # Phase 1 — Data
    df = load_primary_dataset(PRIMARY_DATASET, sample_frac=sample_frac)

    # Phase 2 — Features
    features_df, le_type = engineer_features(df)

    X = features_df[FEATURE_COLUMNS].values
    y = features_df[TARGET_COLUMN].values

    logger.info(
        "[2.5] Train/test split (%.0f/%.0f)...", (1 - TEST_SIZE) * 100, TEST_SIZE * 100
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    logger.info("   Train: %s  |  Test: %s", f"{len(X_train):,}", f"{len(X_test):,}")

    # Phase 3 — SMOTE
    X_train_balanced, y_train_balanced = apply_smote(X_train, y_train)

    # Phase 4 — Train
    results = train_tri_models(X_train_balanced, X_test, y_train_balanced, y_test)

    # Phase 5 — Evaluate
    evaluate_models(results, y_test)

    # Phase 6 — Save
    save_models(results, le_type, MODEL_DIR)

    logger.info("")
    logger.info("=" * 60)
    logger.info("✅ ML PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info("Next steps:")
    logger.info("  1. Restart backend:  uvicorn backend.app.main:app --reload")
    logger.info("  2. Start frontend:   npm run dev")
    logger.info("  3. Test with the High Risk preset in the Simulator")


if __name__ == "__main__":
    main()
