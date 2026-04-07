"""
Model training module — unsupervised anomaly detection.

Supports:
  - IsolationForest
  - LocalOutlierFactor (novelty=True)
  - OneClassSVM
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

logger = logging.getLogger(__name__)

# Registry of supported model classes
MODEL_REGISTRY: dict[str, type] = {
    "IsolationForest": IsolationForest,
    "LocalOutlierFactor": LocalOutlierFactor,
    "OneClassSVM": OneClassSVM,
}


def train_model(
    feature_df: pd.DataFrame,
    model_type: str = "IsolationForest",
    model_params: dict[str, Any] | None = None,
    model_path: str = "artifacts/model.joblib",
    feature_columns: list[str] | None = None,
) -> Any:
    """
    Train an unsupervised anomaly detection model.

    Args:
        feature_df:       DataFrame with account_number + feature columns.
        model_type:       One of IsolationForest, LocalOutlierFactor, OneClassSVM.
        model_params:     Keyword arguments forwarded to the model constructor.
        model_path:       Where to persist the trained model.
        feature_columns:  Ordered feature column names (excludes account_number).

    Returns:
        The fitted sklearn estimator.
    """
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unsupported model type '{model_type}'. Choose from {list(MODEL_REGISTRY)}")

    params = model_params or {}
    logger.info("Training %s with params %s", model_type, params)

    # Prepare feature matrix
    if feature_columns:
        X = feature_df[feature_columns].values
    else:
        X = feature_df.drop(columns=["account_number"], errors="ignore").values

    # Fill any remaining NaNs
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    logger.info("Feature matrix shape: %s", X.shape)

    # Instantiate and fit
    model_cls = MODEL_REGISTRY[model_type]
    model = model_cls(**params)
    model.fit(X)

    # Compute training anomaly stats
    scores = model.decision_function(X)
    labels = model.predict(X)
    n_anomalies = int((labels == -1).sum())
    logger.info(
        "Training complete — anomalies: %d / %d (%.2f%%)",
        n_anomalies,
        len(labels),
        100.0 * n_anomalies / len(labels),
    )
    logger.info(
        "Score stats — mean=%.4f  std=%.4f  min=%.4f  max=%.4f",
        scores.mean(),
        scores.std(),
        scores.min(),
        scores.max(),
    )

    # Persist model
    save_path = Path(model_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, save_path)
    logger.info("Model saved to %s", save_path)

    return model
