"""
Feature Pipeline — unified interface for training and real-time inference.

Combines the stateful feature engine (velocity, duplicates, amount) with
static categorical/temporal features into a single ordered feature vector.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from data.schema import FeatureVector, Transaction
from features.amount_features import compute_amount_features, compute_categorical_features
from features.duplicate_features import compute_duplicate_features
from features.velocity_features import compute_time_gap_features, compute_velocity_features
from state.transaction_state_store import AccountState, TransactionStateStore, TxnRecord

logger = logging.getLogger(__name__)


class FeaturePipeline:
    """
    Reusable feature pipeline for both batch training and single-transaction scoring.

    Usage (training):
        pipeline = FeaturePipeline(config)
        feature_df = pipeline.fit(training_df)

    Usage (real-time):
        features = pipeline.transform_single(transaction)
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config
        self._feature_columns: list[str] = config["feature_columns"]
        self._velocity_windows: dict[str, int] = config.get(
            "velocity_windows",
            {"txn_count_1s": 1, "txn_count_10s": 10, "txn_count_30s": 30},
        )
        self._encoding = config.get("encoding", {})
        self._state_cfg = config.get("state", {})

        # State store — created fresh per pipeline instance
        self._state_store = TransactionStateStore(
            retention_seconds=self._state_cfg.get("retention_seconds", 600),
            max_history_per_account=self._state_cfg.get("max_history_per_account", 1000),
            prune_interval_seconds=self._state_cfg.get("prune_interval_seconds", 60),
        )

    @property
    def state_store(self) -> TransactionStateStore:
        return self._state_store

    @property
    def feature_columns(self) -> list[str]:
        return list(self._feature_columns)

    def _parse_posting_timestamps(self, ts_series: pd.Series) -> pd.Series:
        """Parse mixed timestamp formats and fail with a helpful error if invalid values exist."""
        # Some batch files mix second-precision and microsecond-precision values.
        # `format="mixed"` handles both reliably in recent pandas versions.
        parsed = pd.to_datetime(ts_series, format="mixed", errors="coerce")

        invalid_mask = parsed.isna() & ts_series.notna()
        if invalid_mask.any():
            invalid_values = ts_series[invalid_mask].astype(str).head(5).tolist()
            raise ValueError(
                "Invalid posting_timestamp values encountered; "
                f"sample={invalid_values}. Expected parseable datetime strings."
            )

        return parsed

    # ------------------------------------------------------------------
    # Training: batch transform
    # ------------------------------------------------------------------
    def fit(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute features for an entire training DataFrame.

        The DataFrame must contain:
          account_number, posting_timestamp, transaction_ac_amount,
          descr, debit_credit_code, input_source

        Returns a DataFrame with one row per transaction and all feature columns.
        """
        logger.info("FeaturePipeline.fit — processing %d rows", len(df))

        df = df.copy()
        df["posting_timestamp"] = self._parse_posting_timestamps(df["posting_timestamp"])
        df = df.sort_values(["account_number", "posting_timestamp"]).reset_index(drop=True)

        # Pre-compute dc_sign column
        dc_map = {"D": 1, "C": -1}
        df["_dc_sign"] = df["debit_credit_code"].map(dc_map).fillna(0).astype(int)

        # Initialise result arrays
        n = len(df)
        result: dict[str, np.ndarray] = {col: np.zeros(n, dtype=np.float64) for col in self._feature_columns}

        # Clear state store for fresh training
        self._state_store.clear()

        for idx in range(n):
            row = df.iloc[idx]
            acct = str(row["account_number"])
            ts = row["posting_timestamp"].timestamp()
            amount = float(row["transaction_ac_amount"])
            descr = str(row["descr"])
            dc_sign = int(row["_dc_sign"])
            input_src = str(row["input_source"])
            posting_dt = row["posting_timestamp"].to_pydatetime()

            rec = TxnRecord(ts=ts, amount=amount, descr=descr, dc_sign=dc_sign, input_source=input_src)
            self._state_store.record_transaction(acct, rec)

            features = self._compute_all_features(
                acct=acct,
                current_ts=ts,
                current_amount=amount,
                current_descr=descr,
                current_dc_sign=dc_sign,
                input_source=input_src,
                posting_dt=posting_dt,
            )

            for col in self._feature_columns:
                result[col][idx] = features.get(col, 0.0)

            if (idx + 1) % 10000 == 0:
                logger.info("  processed %d / %d rows", idx + 1, n)

        feature_df = pd.DataFrame(result, columns=self._feature_columns)
        feature_df.insert(0, "account_number", df["account_number"].values)

        logger.info("FeaturePipeline.fit complete — shape %s", feature_df.shape)
        return feature_df

    # ------------------------------------------------------------------
    # Batch transform (for already-fitted state — e.g. scoring pipeline)
    # ------------------------------------------------------------------
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform a batch of transactions using the current state store.

        Same logic as fit() but does not clear state first, allowing
        incremental state accumulation across batches.
        """
        logger.info("FeaturePipeline.transform — processing %d rows", len(df))

        df = df.copy()
        df["posting_timestamp"] = self._parse_posting_timestamps(df["posting_timestamp"])
        df = df.sort_values(["account_number", "posting_timestamp"]).reset_index(drop=True)

        dc_map = {"D": 1, "C": -1}
        df["_dc_sign"] = df["debit_credit_code"].map(dc_map).fillna(0).astype(int)

        n = len(df)
        result: dict[str, np.ndarray] = {col: np.zeros(n, dtype=np.float64) for col in self._feature_columns}

        for idx in range(n):
            row = df.iloc[idx]
            acct = str(row["account_number"])
            ts = row["posting_timestamp"].timestamp()
            amount = float(row["transaction_ac_amount"])
            descr = str(row["descr"])
            dc_sign = int(row["_dc_sign"])
            input_src = str(row["input_source"])
            posting_dt = row["posting_timestamp"].to_pydatetime()

            rec = TxnRecord(ts=ts, amount=amount, descr=descr, dc_sign=dc_sign, input_source=input_src)
            self._state_store.record_transaction(acct, rec)

            features = self._compute_all_features(
                acct=acct,
                current_ts=ts,
                current_amount=amount,
                current_descr=descr,
                current_dc_sign=dc_sign,
                input_source=input_src,
                posting_dt=posting_dt,
            )

            for col in self._feature_columns:
                result[col][idx] = features.get(col, 0.0)

        feature_df = pd.DataFrame(result, columns=self._feature_columns)
        feature_df.insert(0, "account_number", df["account_number"].values)
        return feature_df

    # ------------------------------------------------------------------
    # Real-time: single transaction
    # ------------------------------------------------------------------
    def transform_single(self, txn: Transaction) -> FeatureVector:
        """
        Compute features for a single incoming transaction.

        Mutates the state store (appends the transaction) and returns
        the full FeatureVector.
        """
        acct = txn.account_number
        ts = txn.posting_timestamp.timestamp()
        amount = txn.transaction_ac_amount
        descr = txn.descr
        dc_sign = 1 if txn.debit_credit_code == "D" else -1
        input_src = txn.input_source
        posting_dt = txn.posting_timestamp

        # Record into state
        rec = TxnRecord(ts=ts, amount=amount, descr=descr, dc_sign=dc_sign, input_source=input_src)
        self._state_store.record_transaction(acct, rec)

        features = self._compute_all_features(
            acct=acct,
            current_ts=ts,
            current_amount=amount,
            current_descr=descr,
            current_dc_sign=dc_sign,
            input_source=input_src,
            posting_dt=posting_dt,
        )

        return FeatureVector(**features)

    # ------------------------------------------------------------------
    # Internal: compute all feature groups
    # ------------------------------------------------------------------
    def _compute_all_features(
        self,
        acct: str,
        current_ts: float,
        current_amount: float,
        current_descr: str,
        current_dc_sign: int,
        input_source: str,
        posting_dt: datetime,
    ) -> dict[str, float]:
        """Combine all feature groups into a single dict."""
        account_state = self._state_store.get_account_state(acct)
        if account_state is None:
            # Should not happen (just recorded), but defensive
            return {col: 0.0 for col in self._feature_columns}

        features: dict[str, float] = {}

        # 1. Velocity
        features.update(
            compute_velocity_features(current_ts, account_state, self._velocity_windows)
        )

        # 2. Time-gap
        features.update(
            compute_time_gap_features(current_ts, current_amount, current_descr, account_state)
        )

        # 3. Duplicates + D/C cycling
        features.update(
            compute_duplicate_features(
                current_ts, current_amount, current_descr, current_dc_sign, account_state
            )
        )

        # 4. Amount context
        features.update(
            compute_amount_features(current_ts, current_amount, account_state)
        )

        # 5. Categorical / temporal
        features.update(
            compute_categorical_features(
                posting_dt,
                current_descr,
                "D" if current_dc_sign == 1 else "C",
                input_source,
                descr_map=self._encoding.get("descr"),
                dc_map=self._encoding.get("debit_credit_code"),
                input_source_map=self._encoding.get("input_source"),
            )
        )

        return features
