"""
Duplicate / deduplication feature computation.

Features produced:
  - same_acct_amt_descr_count_10min
  - same_acct_amt_count_10min
  - is_exact_duplicate_10min
  - duplicate_streak
  - dc_sign
  - dc_alternation_count_10txn
  - reversal_count_10min
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from state.transaction_state_store import AccountState, TxnRecord

logger = logging.getLogger(__name__)

WINDOW_10MIN = 600  # seconds


def compute_duplicate_features(
    current_ts: float,
    current_amount: float,
    current_descr: str,
    current_dc_sign: int,
    account_state: AccountState,
) -> dict[str, int | float]:
    """
    Compute duplicate-detection and debit/credit cycling features.

    Scans the 10-minute window backwards from *current_ts*.

    Returns dict with:
      - same_acct_amt_descr_count_10min
      - same_acct_amt_count_10min
      - is_exact_duplicate_10min
      - duplicate_streak        (from AccountState running tracker)
      - dc_sign
      - dc_alternation_count_10txn
      - reversal_count_10min
    """
    history = account_state.history
    cutoff = current_ts - WINDOW_10MIN

    same_amt_descr = 0
    same_amt = 0
    reversal_count = 0

    for rec in reversed(history):
        if rec.ts < cutoff:
            break
        # Same amount + descr
        if rec.amount == current_amount and rec.descr == current_descr:
            same_amt_descr += 1
        # Same amount (any descr)
        if rec.amount == current_amount:
            same_amt += 1
        # Reversal: same amount, opposite sign, both non-zero
        if (
            rec.amount == current_amount
            and rec.dc_sign != current_dc_sign
            and rec.dc_sign != 0
            and current_dc_sign != 0
            and rec is not history[-1]  # skip the current txn itself
        ):
            reversal_count += 1

    is_exact_dup = 1 if same_amt_descr > 1 else 0

    # --- D/C alternation over last 10 txns ---
    last_10 = list(history)[-10:]
    flips = 0
    for i in range(1, len(last_10)):
        if last_10[i].dc_sign != last_10[i - 1].dc_sign:
            flips += 1

    return {
        "same_acct_amt_descr_count_10min": same_amt_descr,
        "same_acct_amt_count_10min": same_amt,
        "is_exact_duplicate_10min": is_exact_dup,
        "duplicate_streak": account_state.current_streak,
        "dc_sign": current_dc_sign,
        "dc_alternation_count_10txn": flips,
        "reversal_count_10min": reversal_count,
    }
