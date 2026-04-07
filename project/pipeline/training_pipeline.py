"""
Training Pipeline — end-to-end: load data → feature engineering → train → persist.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from features.feature_pipeline import FeaturePipeline
from model.train import train_model

logger = logging.getLogger(__name__)


def run_training_pipeline(
    config: dict[str, Any],
    raw_df: pd.DataFrame,
) -> Any:
    """
    Execute the full training pipeline.

    Steps:
      1. Instantiate FeaturePipeline
      2. Compute features for entire dataset (pipeline.fit)
      3. Train unsupervised model
      4. Persist model artifact

    Args:
        config: Parsed config.yaml as dict.
        raw_df: Raw transaction DataFrame with at minimum:
                account_number, posting_timestamp, transaction_ac_amount,
                descr, debit_credit_code, input_source

    Returns:
        Fitted sklearn model.
    """
    logger.info("=" * 60)
    logger.info("TRAINING PIPELINE START")
    logger.info("=" * 60)

    # Ensure required columns
    required = [
        "account_number",
        "posting_timestamp",
        "transaction_ac_amount",
        "descr",
        "debit_credit_code",
        "input_source",
    ]
    missing = [c for c in required if c not in raw_df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Deduplicate columns (e.g. SQL CAST produces two 'account_number')
    raw_df = raw_df.loc[:, ~raw_df.columns.duplicated(keep="last")]

    n_accounts = int(raw_df["account_number"].nunique())
    logger.info("Input rows: %d  |  Accounts: %d", len(raw_df), n_accounts)

    # --- Step 1: Feature engineering ---
    pipeline = FeaturePipeline(config)
    feature_df = pipeline.fit(raw_df)

    # Fill NaN (first txn per account has NaN time gaps)
    feature_df = feature_df.fillna(0)

    logger.info("Feature matrix: %s  columns: %s", feature_df.shape, list(feature_df.columns))

    # --- Step 2: Train model ---
    model_cfg = config.get("model", {})
    model_type = model_cfg.get("type", "IsolationForest")
    model_params = model_cfg.get("params", {}).get(model_type, {})
    model_path = model_cfg.get("path", "artifacts/model.joblib")
    feature_columns = config["feature_columns"]

    model = train_model(
        feature_df=feature_df,
        model_type=model_type,
        model_params=model_params,
        model_path=model_path,
        feature_columns=feature_columns,
    )

    logger.info("=" * 60)
    logger.info("TRAINING PIPELINE COMPLETE")
    logger.info("=" * 60)

    return model
