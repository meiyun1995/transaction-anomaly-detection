"""
Model loader — load and cache a trained model from disk.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib

logger = logging.getLogger(__name__)

# Module-level cache
_cached_model: Any | None = None
_cached_path: str | None = None


def load_model(model_path: str, force_reload: bool = False) -> Any:
    """
    Load a joblib-persisted sklearn model.

    Caches the model in memory so subsequent calls with the same path
    return the already-loaded object (zero latency).

    Args:
        model_path:    Path to the .joblib file.
        force_reload:  If True, bypass cache and reload from disk.

    Returns:
        The fitted sklearn estimator.

    Raises:
        FileNotFoundError: If model_path does not exist.
    """
    global _cached_model, _cached_path

    if not force_reload and _cached_model is not None and _cached_path == model_path:
        return _cached_model

    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    logger.info("Loading model from %s", path)
    model = joblib.load(path)
    _cached_model = model
    _cached_path = model_path
    logger.info("Model loaded: %s", type(model).__name__)
    return model


def clear_cache() -> None:
    """Clear the in-memory model cache."""
    global _cached_model, _cached_path
    _cached_model = None
    _cached_path = None
    logger.info("Model cache cleared")
