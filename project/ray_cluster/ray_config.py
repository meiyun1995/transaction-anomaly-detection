"""
Ray cluster configuration and initialisation.

Provides:
  - ensure_ray_initialized():  idempotent Ray init
  - ModelActor:                Ray actor holding a loaded model for zero-copy scoring
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


def ensure_ray_initialized(config: dict[str, Any]) -> None:
    """
    Initialise Ray if not already running.

    Reads `ray.num_cpus` from config; defaults to os.cpu_count().
    """
    import ray

    if ray.is_initialized():
        logger.debug("Ray already initialised")
        return

    ray_cfg = config.get("ray", {})
    num_cpus = ray_cfg.get("num_cpus") or os.cpu_count()

    logger.info("Initialising Ray with %d CPUs", num_cpus)
    ray.init(num_cpus=num_cpus, ignore_reinit_error=True)
    logger.info("Ray cluster: %s", ray.cluster_resources())


def shutdown_ray() -> None:
    """Gracefully shut down Ray."""
    import ray

    if ray.is_initialized():
        ray.shutdown()
        logger.info("Ray shut down")


# ---------------------------------------------------------------------------
# Ray Actor for model serving (optional — used in high-throughput mode)
# ---------------------------------------------------------------------------
def create_model_actor(config: dict[str, Any]) -> Any:
    """
    Create a Ray actor that holds the model in shared memory.

    Usage:
        actor = create_model_actor(config)
        score, label = ray.get(actor.predict.remote(feature_vector))
    """
    import ray

    ensure_ray_initialized(config)

    @ray.remote
    class ModelActor:
        """Stateful Ray actor holding a trained model for remote scoring."""

        def __init__(self, model_path: str) -> None:
            import joblib

            self.model = joblib.load(model_path)
            logger.info("ModelActor loaded model from %s", model_path)

        def predict(self, feature_vector: list[float]) -> tuple[float, int]:
            """Score a single transaction."""
            import numpy as np

            X = np.array(feature_vector, dtype=np.float64).reshape(1, -1)
            X = np.nan_to_num(X, nan=0.0)
            score = float(self.model.decision_function(X)[0])
            label = int(self.model.predict(X)[0])
            return score, label

        def predict_batch(self, feature_matrix: list[list[float]]) -> tuple[list[float], list[int]]:
            """Score a batch of transactions."""
            import numpy as np

            X = np.array(feature_matrix, dtype=np.float64)
            X = np.nan_to_num(X, nan=0.0)
            scores = self.model.decision_function(X).tolist()
            labels = self.model.predict(X).tolist()
            return scores, labels

    model_path = config.get("model", {}).get("path", "artifacts/model.joblib")
    actor = ModelActor.remote(model_path)
    logger.info("ModelActor created for %s", model_path)
    return actor
