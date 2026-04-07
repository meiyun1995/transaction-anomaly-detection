"""
Data schemas for the transaction anomaly detection system.

Pydantic models for input validation, feature vectors, and scoring results.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class Transaction(BaseModel):
    """Raw incoming transaction from the core banking system."""

    account_number: str = Field(..., description="Account identifier (string to preserve precision)")
    posting_timestamp: datetime = Field(..., description="Exact time the transaction was posted")
    transaction_ac_amount: float = Field(..., ge=0, description="Transaction amount in account currency")
    descr: str = Field(..., description="Transaction description / type")
    debit_credit_code: str = Field(..., pattern="^[DC]$", description="D=Debit, C=Credit")
    input_source: str = Field(..., description="Channel code: T/Z/M/A")

    # Optional fields carried through but not used for features
    business_date: Optional[str] = None
    account_type: Optional[str] = None
    status: Optional[str] = None
    bank_code: Optional[str] = None
    posting_mode: Optional[str] = None
    auxiliary_transaction_code: Optional[str] = None
    transaction_ac_currency: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "account_number": "1000000000000000042",
                "posting_timestamp": "2026-01-15T10:23:45",
                "transaction_ac_amount": 1500.00,
                "descr": "Fund Transfer",
                "debit_credit_code": "D",
                "input_source": "T",
            }
        }


class FeatureVector(BaseModel):
    """Complete feature vector produced by the feature pipeline."""

    # Velocity
    txn_count_1s: int = 0
    txn_count_10s: int = 0
    txn_count_30s: int = 0

    # Time-gap
    time_since_last_txn: float = 0.0
    time_since_last_txn_same_amount: float = 0.0
    time_since_last_txn_same_descr: float = 0.0
    avg_time_gap_last_5txn: float = 0.0
    min_time_gap_last_10txn: float = 0.0

    # Duplicate detection
    same_acct_amt_descr_count_10min: int = 0
    same_acct_amt_count_10min: int = 0
    is_exact_duplicate_10min: int = 0
    duplicate_streak: int = 0

    # Debit/Credit cycling
    dc_sign: int = 0
    dc_alternation_count_10txn: int = 0
    reversal_count_10min: int = 0

    # Amount context
    amount_zscore_account: float = 0.0
    same_amount_frequency: int = 0
    cumulative_amount_10min: float = 0.0

    # Categorical / temporal
    descr_encoded: int = 0
    debit_credit_code_encoded: int = 0
    input_source_encoded: int = 0
    hour_of_day: int = 0
    is_business_hours: int = 0

    def to_list(self, feature_columns: list[str]) -> list[float]:
        """Return feature values in the order specified by feature_columns."""
        return [float(getattr(self, col)) for col in feature_columns]

    def to_dict_ordered(self, feature_columns: list[str]) -> dict[str, float]:
        """Return feature values as an ordered dict."""
        return {col: float(getattr(self, col)) for col in feature_columns}


class ScoringResult(BaseModel):
    """Output of the scoring service for a single transaction."""

    account_number: str
    posting_timestamp: datetime
    transaction_ac_amount: float
    anomaly_score: float = Field(..., description="Raw decision function score (lower = more anomalous)")
    anomaly_label: int = Field(..., description="-1 = anomaly, 1 = normal")
    is_anomaly: bool = Field(..., description="True if anomaly_label == -1")
    top_features: Optional[dict[str, float]] = Field(
        default=None,
        description="Top contributing features (from SHAP or feature importance)",
    )


class BatchScoringResult(BaseModel):
    """Aggregated result for a batch scoring run."""

    total_transactions: int
    anomalies_detected: int
    anomaly_rate: float
    results: list[ScoringResult]
