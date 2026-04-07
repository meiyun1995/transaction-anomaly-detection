"""
Generate a transactions.csv with normal + burst anomaly data for testing.
Uses the same synthetic generator as the production pipeline.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data.synthetic import generate_synthetic_data

df = generate_synthetic_data(
    n_rows=5_000,
    n_accounts=50,
    anomaly_rate=0.05,       # 5% anomalies — easier to spot in a small file
    n_bursts=10,
    burst_min_size=15,
    burst_max_size=30,
    seed=99,
)

# Keep only the columns the scoring/training pipeline needs
keep_cols = [
    "account_number",
    "posting_timestamp",
    "transaction_ac_amount",
    "descr",
    "debit_credit_code",
    "input_source",
    "is_anomaly",
]
df = df[keep_cols].sort_values(["account_number", "posting_timestamp"]).reset_index(drop=True)

out = Path(__file__).resolve().parent / "transactions.csv"
df.to_csv(out, index=False)

n_anom = int(df["is_anomaly"].sum())
print(f"✅ Wrote {len(df):,} rows to {out}")
print(f"   Normal   : {len(df) - n_anom:,}")
print(f"   Anomalies: {n_anom:,}  ({n_anom/len(df):.1%})")
print(f"   Accounts : {df['account_number'].nunique()}")
