"""
Prediction module — single-transaction and batch scoring.

Uses SHAP TreeExplainer for per-transaction feature importance.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import shap

logger = logging.getLogger(__name__)


def create_shap_explainer(model: Any) -> shap.TreeExplainer:
    """
    Create a SHAP TreeExplainer for a tree-based model (IsolationForest).

    The explainer is created once and reused for all predictions.
    """
    explainer = shap.TreeExplainer(model)
    logger.info("SHAP TreeExplainer created for %s", type(model).__name__)
    return explainer


def compute_top_features(
    feature_vector: list[float],
    feature_columns: list[str],
    explainer: shap.TreeExplainer,
    top_n: int = 5,
) -> dict[str, float]:
    """
    Compute top contributing features using SHAP TreeExplainer.

    Returns a dict of {feature_name: shap_value} for the top-N features
    ranked by absolute SHAP value.
    """
    X = np.array(feature_vector, dtype=np.float64).reshape(1, -1)
    X = np.nan_to_num(X, nan=0.0)

    shap_values = explainer.shap_values(X)  # shape: (1, n_features)
    sv = shap_values[0]  # single row

    # Pair feature names with their SHAP values
    pairs = list(zip(feature_columns, sv.tolist()))

    # Sort by absolute SHAP value descending, take top N
    ranked = sorted(pairs, key=lambda kv: abs(kv[1]), reverse=True)
    return {name: round(val, 4) for name, val in ranked[:top_n]}


def compute_top_features_batch(
    feature_matrix: np.ndarray,
    feature_columns: list[str],
    explainer: shap.TreeExplainer,
    top_n: int = 5,
) -> list[list[str]]:
    """
    Compute top contributing feature *names* for every row in a batch.

    Returns a list (one entry per sample) of lists containing the top-N
    feature names ranked by absolute SHAP value.
    """
    X = np.nan_to_num(feature_matrix.astype(np.float64), nan=0.0)
    shap_values = explainer.shap_values(X)  # shape: (n_samples, n_features)

    result: list[list[str]] = []
    for sv in shap_values:
        indices = np.argsort(np.abs(sv))[::-1][:top_n]
        result.append([feature_columns[i] for i in indices])

    logger.info("SHAP top-%d features computed for %d samples", top_n, len(X))
    return result


def predict_single(model: Any, feature_vector: list[float]) -> tuple[float, int]:
    """
    Score a single feature vector.

    Args:
        model:          Fitted sklearn estimator with decision_function and predict.
        feature_vector: Ordered list of floats matching feature_columns.

    Returns:
        (anomaly_score, anomaly_label)
        anomaly_label:  -1 = anomaly,  1 = normal
    """
    X = np.array(feature_vector, dtype=np.float64).reshape(1, -1)
    X = np.nan_to_num(X, nan=0.0)

    score = float(model.decision_function(X)[0])
    label = int(model.predict(X)[0])
    return score, label


def predict_batch(
    model: Any,
    feature_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Score a batch of feature vectors.

    Args:
        model:          Fitted sklearn estimator.
        feature_matrix: 2-D numpy array (n_samples, n_features).

    Returns:
        (scores, labels) — both 1-D arrays of length n_samples.
    """
    X = np.nan_to_num(feature_matrix.astype(np.float64), nan=0.0)
    scores = model.decision_function(X)
    labels = model.predict(X)

    n_anomalies = int((labels == -1).sum())
    logger.info(
        "Batch prediction — %d samples, %d anomalies (%.2f%%)",
        len(labels),
        n_anomalies,
        100.0 * n_anomalies / max(len(labels), 1),
    )
    return scores, labels
