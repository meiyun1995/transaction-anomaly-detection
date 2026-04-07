"""
Transaction Anomaly Detection System — Main Entry Point.

Modes:
  python main.py train   --data <csv_path>
  python main.py score   --data <csv_path>  [--ray]
  python main.py serve   (starts real-time scoring service demo)

All configuration is read from config/config.yaml.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_config(config_path: str = "config/config.yaml") -> dict[str, Any]:
    """Load and return the YAML configuration."""
    path = PROJECT_ROOT / config_path
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def setup_logging(config: dict[str, Any]) -> None:
    """Configure root logger from config."""
    log_cfg = config.get("logging", {})
    logging.basicConfig(
        level=getattr(logging, log_cfg.get("level", "INFO")),
        format=log_cfg.get("format", "%(asctime)s | %(name)-28s | %(levelname)-5s | %(message)s"),
    )


# ---------------------------------------------------------------------------
# TRAIN mode
# ---------------------------------------------------------------------------
def cmd_train(args: argparse.Namespace, config: dict[str, Any]) -> None:
    """Run the training pipeline."""
    from pipeline.training_pipeline import run_training_pipeline
    from data.synthetic import generate_synthetic_data

    logger = logging.getLogger("main.train")

    # Load data
    data_path = Path(args.data)
    if data_path.suffix == ".csv":
        raw_df = pd.read_csv(data_path)
    elif data_path.suffix in (".parquet", ".pq"):
        raw_df = pd.read_parquet(data_path)
    else:
        # Try loading from MySQL using config
        logger.info("Loading data from MySQL...")
        import mysql.connector

        db_cfg = config["database"]
        conn = mysql.connector.connect(
            host=db_cfg["host"],
            user=db_cfg["user"],
            password=db_cfg["password"],
            database=db_cfg["database"],
        )
        raw_df = pd.read_sql(db_cfg["query"], conn)
        conn.close()
        # SELECT t.*, CAST(account_number AS CHAR) produces two columns
        # named 'account_number' — drop the first (numeric) one, keep the CHAR version
        raw_df = raw_df.iloc[:, 1:]

        # --- Generate synthetic burst anomaly data and combine ---
        logger.info("Generating synthetic burst anomaly data...")
        synth_cfg = config.get("synthetic_data", {})
        df_synthetic = generate_synthetic_data(
            n_rows=synth_cfg.get("n_rows", 50_000),
            n_accounts=synth_cfg.get("n_accounts", 250),
            anomaly_rate=synth_cfg.get("anomaly_rate", 0.025),
            n_bursts=synth_cfg.get("n_bursts", 50),
            burst_min_size=synth_cfg.get("burst_min_size", 20),
            burst_max_size=synth_cfg.get("burst_max_size", 30),
            seed=synth_cfg.get("seed", 42),
        )

        # Mark original data as normal
        raw_df["is_anomaly"] = 0

        # Combine: union of columns, missing become NaN
        raw_df = pd.concat([raw_df, df_synthetic], ignore_index=True, sort=False)
        logger.info(
            "Combined dataset: %d rows  |  Anomalies: %d (%.2f%%)",
            len(raw_df),
            int(raw_df["is_anomaly"].sum()),
            raw_df["is_anomaly"].mean() * 100,
        )

    logger.info("Loaded %d rows from %s", len(raw_df), args.data)

    # Safety: if any duplicate column names remain, keep the last
    if raw_df.columns.duplicated().any():
        raw_df = raw_df.loc[:, ~raw_df.columns.duplicated(keep="last")]
    raw_df.to_csv("full_transaction_history.csv", index=False)

    run_training_pipeline(config, raw_df)


# ---------------------------------------------------------------------------
# SCORE mode (batch)
# ---------------------------------------------------------------------------
def cmd_score(args: argparse.Namespace, config: dict[str, Any]) -> None:
    """Run the batch scoring pipeline."""
    from pipeline.scoring_pipeline import run_scoring_pipeline

    logger = logging.getLogger("main.score")

    data_path = Path(args.data)
    if data_path.suffix == ".csv":
        df = pd.read_csv(data_path)
    elif data_path.suffix in (".parquet", ".pq"):
        df = pd.read_parquet(data_path)
    else:
        raise ValueError(f"Unsupported file format: {data_path.suffix}")

    logger.info("Loaded %d transactions for scoring", len(df))

    result, scored_df = run_scoring_pipeline(config, df, use_ray=args.ray)

    logger.info("=" * 60)
    logger.info("SCORING RESULTS")
    logger.info("=" * 60)
    logger.info("Total transactions : %d", result.total_transactions)
    logger.info("Anomalies detected : %d", result.anomalies_detected)
    logger.info("Anomaly rate       : %.2f%%", result.anomaly_rate * 100)

    # Save results
    out_path = Path(args.output) if args.output else data_path.with_name("scored_" + data_path.name)
    scored_df.to_csv(out_path, index=False)

    # Post-process: wrap account_number (first column) with ="..." so Excel
    # keeps it as text instead of converting to scientific notation.
    with open(out_path, "r", encoding="utf-8") as f:
        content = f.read()
    lines = content.split("\n")
    fixed = [lines[0]]  # header unchanged
    for line in lines[1:]:
        if line:
            sep = line.index(",")
            fixed.append(f'="{line[:sep]}"' + line[sep:])
        else:
            fixed.append(line)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        f.write("\n".join(fixed))

    logger.info("Results saved to %s", out_path)


# ---------------------------------------------------------------------------
# SERVE mode (real-time demo)
# ---------------------------------------------------------------------------
def cmd_serve(args: argparse.Namespace, config: dict[str, Any]) -> None:
    """Run real-time scoring service demo."""
    from service.scoring_service import ScoringService

    logger = logging.getLogger("main.serve")
    service = ScoringService(config)

    logger.info("ScoringService ready. Sending demo transactions...")

    import random
    import numpy as np
    from datetime import timedelta

    random.seed(123)
    np.random.seed(123)

    descr_choices = ["Fund Transfer", "GIRO", "Payment to IBT", "Charges", "ATM", "PayNow"]
    dc_choices = ["D", "C"]
    input_choices = ["T", "Z", "M", "A"]
    acct_burst = "1000000000000000042"
    acct_normal = "1000000000000000099"

    demo_transactions: list[dict] = []

    # ── Phase 1: seed normal transactions over the past ~3 months ─────
    # Provides per-account baseline history so amount z-scores and
    # time-gap features behave realistically.
    seed_start = datetime.now() - timedelta(days=90)
    for acct, count in [(acct_burst, 80), (acct_normal, 80)]:
        for i in range(count):
            ts = seed_start + timedelta(
                days=random.randint(0, 89),
                hours=random.randint(8, 17),
                minutes=random.randint(0, 59),
                seconds=random.randint(0, 59),
            )
            demo_transactions.append({
                "account_number": acct,
                "posting_timestamp": ts.isoformat(),
                "transaction_ac_amount": round(np.random.lognormal(7.5, 0.8), 2),
                "descr": random.choice(descr_choices),
                "debit_credit_code": random.choice(dc_choices),
                "input_source": random.choice(input_choices),
                "_phase": "seed",
            })

    # ── Phase 2: 5 normal transactions today (spread out) ────────────
    base_time = datetime.now().replace(hour=10, minute=0, second=0, microsecond=0)
    for i in range(5):
        demo_transactions.append({
            "account_number": acct_burst,
            "posting_timestamp": (base_time + timedelta(hours=i * 1)).isoformat(),
            "transaction_ac_amount": round(np.random.lognormal(7.5, 0.8), 2),
            "descr": random.choice(descr_choices),
            "debit_credit_code": random.choice(dc_choices),
            "input_source": random.choice(input_choices),
            "_phase": "normal",
        })

    # ── Phase 3: BURST — 30 rapid-fire duplicate txns ────────────────
    # Designed to trigger the model's anomaly boundary based on analysis
    # of the training data.  Key drivers:
    #   • high txn_count_30s (≥20)       — many txns in short window
    #   • high cumulative_amount_10min    — large total value
    #   • high same_acct_amt_count_10min  — repeated identical amounts
    #   • is_exact_duplicate_10min = 1    — same acct+amt+descr
    #   • high reversal_count_10min       — D/C alternation
    #   • same_amount_frequency high      — many repeats of one amount
    burst_start = datetime.now().replace(hour=8, minute=30, second=0, microsecond=0)
    burst_amount = 50000.00  # fixed amount → triggers duplicate/same-amount features
    burst_size = 30
    for i in range(burst_size):
        offset_secs = random.randint(0, 120)  # tight 2-minute window
        demo_transactions.append({
            "account_number": acct_burst,
            "posting_timestamp": (burst_start + timedelta(seconds=offset_secs)).isoformat(),
            "transaction_ac_amount": burst_amount,
            "descr": "Charges",
            "debit_credit_code": random.choice(dc_choices),  # D/C mix → reversals
            "input_source": random.choice(["T", "Z"]),
            "_phase": "burst",
        })

    # ── Phase 4: 3 normal transactions for contrast account ──────────
    for i in range(3):
        demo_transactions.append({
            "account_number": acct_normal,
            "posting_timestamp": (base_time + timedelta(hours=i * 3)).isoformat(),
            "transaction_ac_amount": round(np.random.lognormal(7.5, 0.8), 2),
            "descr": random.choice(descr_choices),
            "debit_credit_code": random.choice(dc_choices),
            "input_source": random.choice(input_choices),
            "_phase": "normal",
        })

    # Sort by timestamp to simulate real arrival order
    demo_transactions.sort(key=lambda t: t["posting_timestamp"])

    n_seed = sum(1 for t in demo_transactions if t["_phase"] == "seed")
    n_normal = sum(1 for t in demo_transactions if t["_phase"] == "normal")
    n_burst = sum(1 for t in demo_transactions if t["_phase"] == "burst")
    logger.info(
        "Sending %d transactions: %d seed (silent) + %d normal + %d burst",
        len(demo_transactions), n_seed, n_normal, n_burst,
    )

    anomaly_count = 0
    for i, txn in enumerate(demo_transactions):
        phase = txn.pop("_phase")
        result = service.score_transaction(txn)

        # Only log the interesting transactions (not the 160 seed rows)
        if phase == "seed":
            continue

        flag = " 🚨 ANOMALY" if result.is_anomaly else ""
        if result.is_anomaly:
            anomaly_count += 1
        logger.info(
            "[Txn %2d] acct=...%s  time=%s  amt=%8.2f  score=%+.4f  label=%2d  phase=%-6s%s",
            i + 1,
            result.account_number[-3:],
            txn["posting_timestamp"][11:19],
            txn["transaction_ac_amount"],
            result.anomaly_score,
            result.anomaly_label,
            phase,
            flag,
        )
        # Show top-5 SHAP feature importance for every visible transaction
        if result.top_features:
            feat_str = "  |  ".join(
                f"{name}={shap_val:+.4f}" for name, shap_val in result.top_features.items()
            )
            logger.info("         top features (SHAP): %s", feat_str)

    # Summary
    n_visible = n_normal + n_burst
    logger.info("=" * 60)
    logger.info("Anomalies flagged: %d / %d visible transactions", anomaly_count, n_visible)
    logger.info("State stats: %s", service.state_stats)
    service.shutdown()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transaction Anomaly Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to config file (default: config/config.yaml)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # train
    train_parser = subparsers.add_parser("train", help="Train the anomaly detection model")
    train_parser.add_argument("--data", required=True, help="Path to training data (csv/parquet/db)")

    # score
    score_parser = subparsers.add_parser("score", help="Batch-score transactions")
    score_parser.add_argument("--data", required=True, help="Path to transaction data (csv/parquet)")
    score_parser.add_argument("--ray", action="store_true", help="Use Ray for distributed scoring")
    score_parser.add_argument("--output", default=None, help="Output path for scored results")

    # serve
    subparsers.add_parser("serve", help="Run real-time scoring service demo")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    config = load_config(args.config)
    setup_logging(config)

    if args.command == "train":
        cmd_train(args, config)
    elif args.command == "score":
        cmd_score(args, config)
    elif args.command == "serve":
        cmd_serve(args, config)


if __name__ == "__main__":
    main()
