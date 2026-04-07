"""
Velocity & time-gap feature computation.

Features produced:
  - txn_count_1s, txn_count_10s, txn_count_30s
  - time_since_last_txn
  - time_since_last_txn_same_amount
  - time_since_last_txn_same_descr
  - avg_time_gap_last_5txn
  - min_time_gap_last_10txn
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from state.transaction_state_store import AccountState, TxnRecord

logger = logging.getLogger(__name__)

# Default window sizes in seconds
DEFAULT_VELOCITY_WINDOWS: dict[str, int] = {
    "txn_count_1s": 1,
    "txn_count_10s": 10,
    "txn_count_30s": 30,
}


def compute_velocity_features(
    current_ts: float,
    account_state: AccountState,
    velocity_windows: dict[str, int] | None = None,
) -> dict[str, int]:
    """
    Count transactions in each rolling time window for the same account.

    Scans the account history backwards from *current_ts* and counts records
    falling within each window boundary.

    Returns:
        dict mapping feature name → transaction count.
    """
    windows = velocity_windows or DEFAULT_VELOCITY_WINDOWS
    counts: dict[str, int] = {name: 0 for name in windows}

    # Largest window determines how far back we scan
    max_window = max(windows.values())

    for rec in reversed(account_state.history):
        delta = current_ts - rec.ts
        if delta < 0:
            continue  # future record (shouldn't happen, safety guard)
        if delta > max_window:
            break  # no more records can fall in any window
        for name, win_sec in windows.items():
            if delta <= win_sec:
                counts[name] += 1

    return counts


def compute_time_gap_features(
    current_ts: float,
    current_amount: float,
    current_descr: str,
    account_state: AccountState,
) -> dict[str, float]:
    """
    Compute inter-transaction time-gap features.

    Features:
      - time_since_last_txn:              seconds since the previous txn (any)
      - time_since_last_txn_same_amount:  seconds since last txn with same amount
      - time_since_last_txn_same_descr:   seconds since last txn with same descr
      - avg_time_gap_last_5txn:           mean gap over last 5 inter-txn intervals
      - min_time_gap_last_10txn:          min gap over last 10 inter-txn intervals
    """
    history = account_state.history
    n = len(history)

    # --- time_since_last_txn ---
    # The current txn is already appended, so the "previous" is at index -2
    time_since_last: float = 0.0
    if n >= 2:
        time_since_last = current_ts - history[-2].ts

    # --- time_since_last_txn_same_amount ---
    time_since_same_amount: float = 0.0
    for i in range(n - 2, -1, -1):
        if history[i].amount == current_amount:
            time_since_same_amount = current_ts - history[i].ts
            break

    # --- time_since_last_txn_same_descr ---
    time_since_same_descr: float = 0.0
    for i in range(n - 2, -1, -1):
        if history[i].descr == current_descr:
            time_since_same_descr = current_ts - history[i].ts
            break

    # --- avg_time_gap_last_5txn ---
    gaps: list[float] = []
    scan_depth = min(n, 11)  # need up to 10 gaps (11 records) for min_gap_10
    for i in range(n - 1, max(n - scan_depth, 0), -1):
        gaps.append(history[i].ts - history[i - 1].ts)

    avg_gap_5 = sum(gaps[:5]) / len(gaps[:5]) if gaps else 0.0
    min_gap_10 = min(gaps[:10]) if gaps else 0.0

    return {
        "time_since_last_txn": time_since_last,
        "time_since_last_txn_same_amount": time_since_same_amount,
        "time_since_last_txn_same_descr": time_since_same_descr,
        "avg_time_gap_last_5txn": avg_gap_5,
        "min_time_gap_last_10txn": min_gap_10,
    }
