"""
Transaction State Store — high-performance in-memory sliding-window state.

Maintains per-account transaction history for real-time feature computation.
Uses collections.deque with timestamp-based pruning for O(1) amortised appends
and efficient window scans.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lightweight record stored per transaction
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class TxnRecord:
    """Minimal transaction record kept in state for feature computation."""

    ts: float                     # Unix epoch seconds
    amount: float
    descr: str
    dc_sign: int                  # +1 debit, -1 credit
    input_source: str


# ---------------------------------------------------------------------------
# Per-account state bucket
# ---------------------------------------------------------------------------
@dataclass
class AccountState:
    """Sliding-window buffer for one account."""

    history: deque[TxnRecord] = field(default_factory=lambda: deque(maxlen=1000))

    # Running statistics for z-score
    amount_sum: float = 0.0
    amount_sq_sum: float = 0.0
    txn_count: int = 0

    # Cumulative same-amount counter  {amount -> count}
    amount_freq: dict[float, int] = field(default_factory=dict)

    # Last descr for duplicate-streak tracking
    last_amount: Optional[float] = None
    last_descr: Optional[str] = None
    current_streak: int = 0

    def push(self, rec: TxnRecord) -> None:
        """Append a new transaction and update running stats."""
        self.history.append(rec)

        # Running amount statistics (Welford-lite)
        self.txn_count += 1
        self.amount_sum += rec.amount
        self.amount_sq_sum += rec.amount * rec.amount

        # Same-amount frequency
        self.amount_freq[rec.amount] = self.amount_freq.get(rec.amount, 0) + 1

        # Duplicate streak
        if rec.amount == self.last_amount and rec.descr == self.last_descr:
            self.current_streak += 1
        else:
            self.current_streak = 1
        self.last_amount = rec.amount
        self.last_descr = rec.descr

    @property
    def amount_mean(self) -> float:
        return self.amount_sum / self.txn_count if self.txn_count else 0.0

    @property
    def amount_std(self) -> float:
        if self.txn_count < 2:
            return 0.0
        variance = (self.amount_sq_sum / self.txn_count) - (self.amount_mean ** 2)
        return max(variance, 0.0) ** 0.5


# ---------------------------------------------------------------------------
# Main state store
# ---------------------------------------------------------------------------
class TransactionStateStore:
    """
    Thread-safe in-memory state store for transaction history.

    Supports:
      • O(1) append per transaction
      • Sliding-window queries (1s, 10s, 30s, 10min)
      • Background pruning of expired records
      • Thousands of transactions per second throughput
    """

    def __init__(
        self,
        retention_seconds: int = 600,
        max_history_per_account: int = 1000,
        prune_interval_seconds: int = 60,
    ) -> None:
        self._retention = retention_seconds
        self._max_history = max_history_per_account
        self._prune_interval = prune_interval_seconds

        self._accounts: dict[str, AccountState] = {}
        self._lock = threading.Lock()

        self._pruner_running = False
        self._pruner_thread: Optional[threading.Thread] = None

        logger.info(
            "StateStore initialised  retention=%ds  max_history=%d  prune_interval=%ds",
            retention_seconds,
            max_history_per_account,
            prune_interval_seconds,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def record_transaction(self, account_number: str, rec: TxnRecord) -> None:
        """Insert a new transaction into the state store."""
        with self._lock:
            state = self._accounts.get(account_number)
            if state is None:
                state = AccountState(history=deque(maxlen=self._max_history))
                self._accounts[account_number] = state
            state.push(rec)

    def get_account_state(self, account_number: str) -> Optional[AccountState]:
        """Retrieve the full account state (caller must not mutate)."""
        return self._accounts.get(account_number)

    def get_window(
        self, account_number: str, window_seconds: int, current_ts: float
    ) -> list[TxnRecord]:
        """Return transactions within *window_seconds* of *current_ts*."""
        state = self._accounts.get(account_number)
        if state is None:
            return []
        cutoff = current_ts - window_seconds
        # Iterate from newest (right) — short-circuit once outside window
        result: list[TxnRecord] = []
        for rec in reversed(state.history):
            if rec.ts < cutoff:
                break
            result.append(rec)
        result.reverse()
        return result

    def get_last_n(self, account_number: str, n: int) -> list[TxnRecord]:
        """Return the last *n* transactions for an account."""
        state = self._accounts.get(account_number)
        if state is None:
            return []
        history = state.history
        start = max(0, len(history) - n)
        return list(history)[start:]

    # ------------------------------------------------------------------
    # Pruning
    # ------------------------------------------------------------------
    def prune(self, now: Optional[float] = None) -> int:
        """Remove expired records across all accounts. Returns count pruned."""
        now = now or time.time()
        cutoff = now - self._retention
        pruned = 0
        with self._lock:
            empty_accounts: list[str] = []
            for acct, state in self._accounts.items():
                before = len(state.history)
                while state.history and state.history[0].ts < cutoff:
                    state.history.popleft()
                pruned += before - len(state.history)
                if not state.history:
                    empty_accounts.append(acct)
            for acct in empty_accounts:
                del self._accounts[acct]
        if pruned:
            logger.debug("Pruned %d expired records from state store", pruned)
        return pruned

    def start_background_pruning(self) -> None:
        """Launch a daemon thread that periodically prunes old records."""
        if self._pruner_running:
            return
        self._pruner_running = True
        self._pruner_thread = threading.Thread(target=self._prune_loop, daemon=True)
        self._pruner_thread.start()
        logger.info("Background pruning thread started (interval=%ds)", self._prune_interval)

    def stop_background_pruning(self) -> None:
        """Stop the background pruning thread."""
        self._pruner_running = False
        if self._pruner_thread:
            self._pruner_thread.join(timeout=5)
            logger.info("Background pruning thread stopped")

    def _prune_loop(self) -> None:
        while self._pruner_running:
            time.sleep(self._prune_interval)
            try:
                self.prune()
            except Exception:
                logger.exception("Error during state pruning")

    # ------------------------------------------------------------------
    # Bulk loading (training mode)
    # ------------------------------------------------------------------
    def bulk_load(self, account_number: str, records: list[TxnRecord]) -> None:
        """Load a sorted list of records for one account (training bootstrap)."""
        with self._lock:
            state = AccountState(history=deque(maxlen=self._max_history))
            for rec in records:
                state.push(rec)
            self._accounts[account_number] = state

    def clear(self) -> None:
        """Wipe all state (used between training runs)."""
        with self._lock:
            self._accounts.clear()
        logger.info("State store cleared")

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    @property
    def num_accounts(self) -> int:
        return len(self._accounts)

    @property
    def total_records(self) -> int:
        return sum(len(s.history) for s in self._accounts.values())

    def __repr__(self) -> str:
        return (
            f"TransactionStateStore(accounts={self.num_accounts}, "
            f"records={self.total_records}, retention={self._retention}s)"
        )
