"""
Real-time Scoring Service — single-transaction scoring with low latency.

Designed as the primary integration point for a banking transaction
monitoring system. Call `score_transaction()` for each incoming transaction.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Any

from data.schema import FeatureVector, ScoringResult, Transaction
from features.feature_pipeline import FeaturePipeline
from model.model_loader import load_model
from model.predict import compute_top_features, create_shap_explainer, predict_single

logger = logging.getLogger(__name__)


class ScoringService:
    """
    Stateful scoring service for near real-time transaction anomaly detection.

    Holds:
      - FeaturePipeline (with embedded TransactionStateStore)
      - Loaded model (cached in memory)

    Thread-safety: the underlying state store uses a lock; multiple threads
    can call score_transaction concurrently.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config
        self._pipeline = FeaturePipeline(config)
        self._model = load_model(config["model"]["path"])
        self._feature_columns = config["feature_columns"]
        self._explainer = create_shap_explainer(self._model)

        # Start background pruning for state store
        self._pipeline.state_store.start_background_pruning()

        logger.info(
            "ScoringService initialised — model=%s, features=%d",
            type(self._model).__name__,
            len(self._feature_columns),
        )

    def score_transaction(self, txn_dict: dict[str, Any]) -> ScoringResult:
        """
        Score a single transaction end-to-end.

        Steps:
          1. Validate input → Transaction schema
          2. Update transaction state store
          3. Compute stateful features
          4. Run model prediction
          5. Return ScoringResult

        Args:
            txn_dict: Raw transaction as a dictionary.

        Returns:
            ScoringResult with anomaly_score, anomaly_label, is_anomaly.
        """
        t0 = time.perf_counter()

        # 1. Validate
        txn = Transaction(**txn_dict)

        # 2 + 3. Update state and compute features (done inside transform_single)
        feature_vec: FeatureVector = self._pipeline.transform_single(txn)

        # 4. Predict
        features_list = feature_vec.to_list(self._feature_columns)
        score, label = predict_single(self._model, features_list)

        # 4b. Compute top contributing features (SHAP TreeExplainer)
        top_features = compute_top_features(
            features_list, self._feature_columns, self._explainer, top_n=5,
        )

        # 5. Build result
        result = ScoringResult(
            account_number=txn.account_number,
            posting_timestamp=txn.posting_timestamp,
            transaction_ac_amount=txn.transaction_ac_amount,
            anomaly_score=score,
            anomaly_label=label,
            is_anomaly=(label == -1),
            top_features=top_features,
        )

        elapsed_ms = (time.perf_counter() - t0) * 1000
        log_fn = logger.warning if result.is_anomaly else logger.debug
        log_fn(
            "Scored acct=%s amount=%.2f → score=%.4f label=%d  (%.1fms)",
            txn.account_number,
            txn.transaction_ac_amount,
            score,
            label,
            elapsed_ms,
        )

        return result

    def score_batch(self, txn_dicts: list[dict[str, Any]]) -> list[ScoringResult]:
        """Score multiple transactions sequentially (preserves ordering for state)."""
        return [self.score_transaction(t) for t in txn_dicts]

    def shutdown(self) -> None:
        """Clean up resources."""
        self._pipeline.state_store.stop_background_pruning()
        logger.info("ScoringService shut down")

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    @property
    def state_stats(self) -> dict[str, int]:
        store = self._pipeline.state_store
        return {
            "accounts_tracked": store.num_accounts,
            "total_records": store.total_records,
        }
