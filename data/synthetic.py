"""
Synthetic burst-anomaly data generator.

Generates realistic-looking transactions with a configurable percentage
of "burst anomaly" events — multiple transactions from the same account
within a short time window — matching the pattern used in the exploration
notebook.
"""

from __future__ import annotations

import logging
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Lookup arrays (match notebook) ──────────────────────
_ACCOUNT_TYPES = ["S", "D"]
_ACU_DBU_FLAGS = ["A", "D"]
_DEBIT_CREDIT = ["D", "C"]
_INPUT_SOURCES = ["T", "Z", "M", "A"]
_CURRENCIES = ["USD", "SGD", "MYR"]
_CHANNEL_CODES = ["BR", "ATM", "MOB", "ONL", "API"]
_PRODUCT_CODES = ["CA001", "SA001", "SA002", "FD001", "FD002"]
_CUSTOMER_CATEGORIES = ["R1", "R2", "C1", "C2", "P1"]
_AUX_TXN_CODES = ["7100", "7110", "7120", "7130", "7140", "7150"]
_POSTING_MODES = ["O", "R"]
_BANK_CODES = ["7339", "7340", "7341"]
_DESCR_CHOICES = [
    "Fund Transfer",
    "GIRO",
    "Payment to IBT",
    "Charges",
    "ATM",
    "PayNow",
]


def _rand_timestamp(d: datetime) -> datetime:
    """Random timestamp on date *d* during business hours (08–17)."""
    return d.replace(
        hour=random.randint(8, 17),
        minute=random.randint(0, 59),
        second=random.randint(0, 59),
        microsecond=random.randint(0, 999_999),
    )


def generate_synthetic_data(
    n_rows: int = 50_000,
    n_accounts: int = 250,
    anomaly_rate: float = 0.025,
    date_start: datetime = datetime(2025, 12, 1),
    date_end: datetime = datetime(2026, 3, 4),
    n_bursts: int = 50,
    burst_min_size: int = 20,
    burst_max_size: int = 30,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a synthetic transaction DataFrame.

    Parameters
    ----------
    n_rows : int
        Total number of rows to generate.
    n_accounts : int
        Number of distinct accounts.
    anomaly_rate : float
        Fraction of rows that are burst anomalies (0.0 – 1.0).
    date_start, date_end : datetime
        Date range for transactions.
    n_bursts : int
        Number of burst events to create.
    burst_min_size, burst_max_size : int
        Min/max transactions per burst.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns matching the production schema plus
        an ``is_anomaly`` column (0 = normal, 1 = burst anomaly).
    """
    np.random.seed(seed)
    random.seed(seed)

    days_span = (date_end - date_start).days
    accounts = [int(1e18) + i for i in range(n_accounts)]

    n_anomalies = int(n_rows * anomaly_rate)
    n_normal = n_rows - n_anomalies

    # ── Normal transactions ─────────────────────────────
    normal_accounts = np.random.choice(accounts, n_normal)
    normal_dates = [
        date_start + timedelta(days=random.randint(0, days_span))
        for _ in range(n_normal)
    ]
    normal_ts = [_rand_timestamp(d) for d in normal_dates]
    normal_amounts = np.round(
        np.random.lognormal(mean=7.5, sigma=0.8, size=n_normal), 2
    )

    # ── Burst anomalies ─────────────────────────────────
    burst_accounts: list[int] = []
    burst_dates: list[datetime] = []
    burst_ts: list[datetime] = []
    burst_amounts_list: list[float] = []

    remaining = n_anomalies
    for _ in range(n_bursts + 100):  # generous upper bound
        if remaining <= 0:
            break
        burst_size = min(random.randint(burst_min_size, burst_max_size), remaining)
        remaining -= burst_size

        acct = random.choice(accounts)
        day = date_start + timedelta(days=random.randint(0, days_span))
        base_hour = random.randint(8, 17)
        base_minute = random.randint(0, 55)
        window_secs = random.randint(60, 300)  # 1–5 minutes

        for j in range(burst_size):
            offset = random.randint(0, window_secs)
            ts = day.replace(
                hour=base_hour, minute=base_minute, second=0, microsecond=0
            ) + timedelta(seconds=offset)
            burst_accounts.append(acct)
            burst_dates.append(day)
            burst_ts.append(ts)
            burst_amounts_list.append(
                round(np.random.lognormal(mean=7.5, sigma=0.8), 2)
            )

    burst_amounts = np.array(burst_amounts_list)
    actual_anomalies = len(burst_accounts)

    # ── Combine ─────────────────────────────────────────
    all_accounts = np.concatenate([normal_accounts, burst_accounts])
    all_dates = normal_dates + burst_dates
    all_ts = normal_ts + burst_ts
    all_amounts = np.concatenate([normal_amounts, burst_amounts])
    all_anomaly = np.array([0] * n_normal + [1] * actual_anomalies)

    n_total = len(all_accounts)

    rows = {
        "account_number": all_accounts,
        "account_type": np.random.choice(_ACCOUNT_TYPES, n_total),
        "business_date": [d.strftime("%Y-%m-%d") for d in all_dates],
        "generated_txn_seq": np.arange(1, n_total + 1),
        "posting_timestamp": all_ts,
        "descr": np.random.choice(_DESCR_CHOICES, n_total),
        "status": np.where(all_anomaly, "AB", "AA"),
        "created_datetime": all_ts,
        "created_workstation": np.random.choice(
            ["WS001", "WS002", "WS003", "WS004", "WS005"], n_total
        ),
        "version": np.zeros(n_total, dtype=int),
        "account_acu_dbu_flag": np.random.choice(_ACU_DBU_FLAGS, n_total),
        "affects_code": np.full(n_total, "B"),
        "auxiliary_transaction_code": np.random.choice(_AUX_TXN_CODES, n_total),
        "bank_code": np.random.choice(_BANK_CODES, n_total),
        "debit_credit_code": np.random.choice(_DEBIT_CREDIT, n_total),
        "input_source": np.random.choice(_INPUT_SOURCES, n_total),
        "posting_mode": np.random.choice(_POSTING_MODES, n_total, p=[0.95, 0.05]),
        "teller_journal_sequence": np.random.randint(1000, 10000, n_total),
        "transaction_ac_amount": all_amounts,
        "transaction_ac_currency": np.random.choice(_CURRENCIES, n_total),
        "transaction_curr_decimal": np.full(n_total, "2"),
        "is_anomaly": all_anomaly,
    }

    df_synthetic = pd.DataFrame(rows)

    logger.info(
        "Synthetic data generated: %d rows  |  anomalies: %d (%.2f%%)",
        len(df_synthetic),
        actual_anomalies,
        df_synthetic["is_anomaly"].mean() * 100,
    )

    return df_synthetic
