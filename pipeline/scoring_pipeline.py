"""
Scoring Pipeline — batch scoring with Ray for distributed processing.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np
import pandas as pd

from data.schema import BatchScoringResult
from features.feature_pipeline import FeaturePipeline
from model.model_loader import load_model
from model.predict import predict_batch, create_shap_explainer, compute_top_features_batch

logger = logging.getLogger(__name__)


def run_scoring_pipeline(
    config: dict[str, Any],
    transactions_df: pd.DataFrame,
    use_ray: bool = True,
) -> tuple[BatchScoringResult, pd.DataFrame]:
    """
    Score a batch of transactions.

    When use_ray=True, uses Ray Dataset.map_batches for parallel feature
    engineering and prediction. Falls back to sequential processing otherwise.

    Args:
        config:           Parsed config.yaml.
        transactions_df:  Raw transaction DataFrame.
        use_ray:          Whether to use Ray for distributed processing.

    Returns:
        (BatchScoringResult, scored_df)
        scored_df contains all feature columns, top_5_features, anomaly_score,
        and is_anomaly.
    """
    logger.info("Scoring pipeline — %d transactions, ray=%s", len(transactions_df), use_ray)

    model_path = config.get("model", {}).get("path", "artifacts/model.joblib")
    feature_columns = config["feature_columns"]
    model = load_model(model_path)

    if use_ray:
        return _score_with_ray(config, transactions_df, model, feature_columns)
    else:
        return _score_sequential(config, transactions_df, model, feature_columns)


def _score_sequential(
    config: dict[str, Any],
    df: pd.DataFrame,
    model: Any,
    feature_columns: list[str],
) -> tuple[BatchScoringResult, pd.DataFrame]:
    """Sequential (non-Ray) scoring path."""
    pipeline = FeaturePipeline(config)
    feature_df = pipeline.transform(df)
    feature_df = feature_df.fillna(0)

    X = feature_df[feature_columns].values
    scores, labels = predict_batch(model, X)

    # Build output DataFrame: all features + score + label
    scored_df = feature_df[feature_columns].copy()
    scored_df.insert(0, "account_number", feature_df["account_number"].astype(str).values)
    scored_df["anomaly_score"] = scores
    scored_df["is_anomaly"] = labels == -1

    # SHAP top-5 feature names — only for anomaly rows
    scored_df["top_5_features"] = None
    anomaly_mask = scored_df["is_anomaly"]
    if anomaly_mask.any():
        explainer = create_shap_explainer(model)
        top_features = compute_top_features_batch(
            X[anomaly_mask.values], feature_columns, explainer,
        )
        scored_df.loc[anomaly_mask, "top_5_features"] = pd.Series(
            top_features, index=scored_df.index[anomaly_mask],
        )

    n_anomalies = int((labels == -1).sum())
    summary = BatchScoringResult(
        total_transactions=len(df),
        anomalies_detected=n_anomalies,
        anomaly_rate=n_anomalies / max(len(df), 1),
        results=[],
    )
    return summary, scored_df


def _score_with_ray(
    config: dict[str, Any],
    df: pd.DataFrame,
    model: Any,
    feature_columns: list[str],
) -> tuple[BatchScoringResult, pd.DataFrame]:
    """Ray-parallelised scoring path using Dataset.map_batches."""
    try:
        import ray
        from ray_cluster.ray_config import ensure_ray_initialized
    except ImportError:
        logger.warning("Ray not available — falling back to sequential scoring")
        return _score_sequential(config, df, model, feature_columns)

    ensure_ray_initialized(config)

    ray_cfg = config.get("ray", {})
    batch_size = ray_cfg.get("batch_size", 4096)

    # Put model in Ray object store for zero-copy sharing
    model_ref = ray.put(model)
    config_ref = ray.put(config)
    feature_cols_ref = ray.put(feature_columns)

    # Create Ray Dataset from pandas
    dataset = ray.data.from_pandas(df)

    def score_batch(batch: pd.DataFrame) -> pd.DataFrame:
        """Map function applied to each batch by Ray."""
        _config = ray.get(config_ref)
        _model = ray.get(model_ref)
        _feature_cols = ray.get(feature_cols_ref)

        _pipeline = FeaturePipeline(_config)
        _feature_df = _pipeline.transform(batch)
        _feature_df = _feature_df.fillna(0)

        X = _feature_df[_feature_cols].values
        X = np.nan_to_num(X.astype(np.float64), nan=0.0)

        out = _feature_df[_feature_cols].copy()
        out.insert(0, "account_number", _feature_df["account_number"].astype(str).values)
        out["anomaly_score"] = _model.decision_function(X)
        labels = _model.predict(X)
        out["is_anomaly"] = labels == -1

        out["top_5_features"] = None
        anomaly_mask = out["is_anomaly"]
        if anomaly_mask.any():
            _explainer = create_shap_explainer(_model)
            _top_features = compute_top_features_batch(
                X[anomaly_mask.values], _feature_cols, _explainer,
            )
            out.loc[anomaly_mask, "top_5_features"] = pd.Series(
                _top_features, index=out.index[anomaly_mask],
            )
        return out

    scored_ds = dataset.map_batches(score_batch, batch_size=batch_size, batch_format="pandas")
    scored_df = scored_ds.to_pandas()

    n_anomalies = int(scored_df["is_anomaly"].sum())

    summary = BatchScoringResult(
        total_transactions=len(scored_df),
        anomalies_detected=n_anomalies,
        anomaly_rate=n_anomalies / max(len(scored_df), 1),
        results=[],
    )
    return summary, scored_df
