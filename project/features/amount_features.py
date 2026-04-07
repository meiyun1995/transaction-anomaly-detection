"""
Amount-context and categorical / temporal feature computation.

Features produced:
  - amount_zscore_account
  - same_amount_frequency
  - cumulative_amount_10min
  - descr_encoded
  - debit_credit_code_encoded
  - input_source_encoded
  - hour_of_day
  - is_business_hours
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from state.transaction_state_store import AccountState, TxnRecord

logger = logging.getLogger(__name__)

WINDOW_10MIN = 600  # seconds

# Default encoding maps (overridden by config at runtime)
DEFAULT_DESCR_MAP: dict[str, int] = {
    "ATM": 0,
    "Charges": 1,
    "Fund Transfer": 2,
    "GIRO": 3,
    "PayNow": 4,
    "Payment to IBT": 5,
}

DEFAULT_DC_MAP: dict[str, int] = {"C": 0, "D": 1}

DEFAULT_INPUT_SOURCE_MAP: dict[str, int] = {"A": 0, "M": 1, "T": 2, "Z": 3}


def compute_amount_features(
    current_ts: float,
    current_amount: float,
    account_state: AccountState,
) -> dict[str, float]:
    """
    Compute per-account amount context features.

    Returns:
      - amount_zscore_account:   z-score of current amount vs account history
      - same_amount_frequency:   cumulative count of this exact amount for account
      - cumulative_amount_10min: sum of all amounts in last 10 min window
    """
    # --- z-score ---
    std = account_state.amount_std
    mean = account_state.amount_mean
    if std > 0:
        zscore = (current_amount - mean) / std
    else:
        zscore = 0.0

    # --- same-amount frequency (from running counter) ---
    same_freq = account_state.amount_freq.get(current_amount, 0)

    # --- cumulative amount in 10 min ---
    cutoff = current_ts - WINDOW_10MIN
    cum_amount = 0.0
    for rec in reversed(account_state.history):
        if rec.ts < cutoff:
            break
        cum_amount += rec.amount

    return {
        "amount_zscore_account": round(zscore, 6),
        "same_amount_frequency": same_freq,
        "cumulative_amount_10min": round(cum_amount, 2),
    }


def compute_categorical_features(
    posting_timestamp: datetime,
    descr: str,
    debit_credit_code: str,
    input_source: str,
    descr_map: dict[str, int] | None = None,
    dc_map: dict[str, int] | None = None,
    input_source_map: dict[str, int] | None = None,
) -> dict[str, int]:
    """
    Encode categorical fields and extract temporal features.

    Returns:
      - descr_encoded
      - debit_credit_code_encoded
      - input_source_encoded
      - hour_of_day
      - is_business_hours
    """
    _descr_map = descr_map or DEFAULT_DESCR_MAP
    _dc_map = dc_map or DEFAULT_DC_MAP
    _input_map = input_source_map or DEFAULT_INPUT_SOURCE_MAP

    hour = posting_timestamp.hour
    weekday = posting_timestamp.weekday()  # 0=Mon … 6=Sun
    is_biz = 1 if (weekday < 5 and 8 <= hour < 18) else 0

    return {
        "descr_encoded": _descr_map.get(descr, -1),
        "debit_credit_code_encoded": _dc_map.get(debit_credit_code, -1),
        "input_source_encoded": _input_map.get(input_source, -1),
        "hour_of_day": hour,
        "is_business_hours": is_biz,
    }
