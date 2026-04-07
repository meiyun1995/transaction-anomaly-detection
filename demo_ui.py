"""
Transaction Anomaly Detection — Bank Demo Dashboard
====================================================
Launch:  streamlit run demo_ui.py
"""

from __future__ import annotations

import sys
import time
import random
from pathlib import Path
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yaml

# ── Ensure project root on path ──────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from service.scoring_service import ScoringService
from data.schema import ScoringResult

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Page config & theme
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.set_page_config(
    page_title="Anomaly Detection — Live Demo",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS for banking-grade look ────────────────────────────────────
st.markdown("""
<style>
/* ── Global ─────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Header banner ──────────────────────────────────────── */
.main-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #0f766e 100%);
    padding: 1.8rem 2.2rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 24px rgba(0,0,0,0.18);
}
.main-header h1 {
    color: #ffffff;
    font-size: 1.75rem;
    font-weight: 700;
    margin: 0 0 0.25rem 0;
    letter-spacing: -0.02em;
}
.main-header p {
    color: #94a3b8;
    font-size: 0.95rem;
    margin: 0;
}

/* ── Metric cards ───────────────────────────────────────── */
.metric-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    text-align: center;
}
.metric-card .label {
    color: #64748b;
    font-size: 0.78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 0.35rem;
}
.metric-card .value {
    font-size: 1.85rem;
    font-weight: 700;
    margin: 0;
    line-height: 1.1;
}
.metric-card .value.green  { color: #059669; }
.metric-card .value.red    { color: #dc2626; }
.metric-card .value.blue   { color: #2563eb; }
.metric-card .value.amber  { color: #d97706; }

/* ── Transaction feed ───────────────────────────────────── */
.txn-row {
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: 0.75rem 1rem;
    border-radius: 8px;
    margin-bottom: 0.5rem;
    font-size: 0.88rem;
    transition: all 0.2s;
}
.txn-row:hover { transform: translateX(3px); }
.txn-normal {
    background: #f0fdf4;
    border-left: 4px solid #22c55e;
}
.txn-anomaly {
    background: #fef2f2;
    border-left: 4px solid #ef4444;
    animation: pulse-red 1.5s ease-in-out;
}
@keyframes pulse-red {
    0%   { box-shadow: 0 0 0 0 rgba(239,68,68,0.4); }
    70%  { box-shadow: 0 0 0 8px rgba(239,68,68,0); }
    100% { box-shadow: 0 0 0 0 rgba(239,68,68,0); }
}
.txn-badge {
    display: inline-block;
    padding: 0.15rem 0.6rem;
    border-radius: 9999px;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.04em;
}
.badge-normal  { background: #dcfce7; color: #166534; }
.badge-anomaly { background: #fee2e2; color: #991b1b; }

/* ── Sidebar ────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: #f8fafc;
}
section[data-testid="stSidebar"] h1 {
    font-size: 1.1rem;
    color: #1e293b;
}

/* ── Pretty scrollbar ──────────────────────────────────── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 3px; }

/* ── Hide Streamlit branding ────────────────────────────── */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }

/* ── Feature importance chips ───────────────────────────── */
.feat-chip {
    display: inline-block;
    background: #eff6ff;
    color: #1e40af;
    padding: 0.2rem 0.55rem;
    border-radius: 6px;
    font-size: 0.75rem;
    font-weight: 500;
    margin: 0.15rem 0.15rem;
    border: 1px solid #bfdbfe;
}
.feat-chip.neg {
    background: #fef2f2;
    color: #991b1b;
    border-color: #fecaca;
}
</style>
""", unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Session state helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _get_service() -> ScoringService:
    """Singleton scoring service in session state."""
    if "service" not in st.session_state:
        cfg_path = PROJECT_ROOT / "config" / "config.yaml"
        with open(cfg_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        st.session_state["service"] = ScoringService(config)
        st.session_state["config"] = config
    return st.session_state["service"]


def _init_state():
    """Initialise session state for the demo."""
    defaults = {
        "results": [],          # list[dict]  – scored results + metadata
        "demo_running": False,
        "seed_done": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Helper: score & record
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _score_and_record(txn_dict: dict[str, Any], phase: str = "manual") -> ScoringResult:
    service = _get_service()
    t0 = time.perf_counter()
    result = service.score_transaction(txn_dict)
    latency_ms = (time.perf_counter() - t0) * 1000

    st.session_state["results"].append({
        "account": txn_dict["account_number"],
        "timestamp": txn_dict["posting_timestamp"],
        "amount": txn_dict["transaction_ac_amount"],
        "descr": txn_dict["descr"],
        "dc": txn_dict["debit_credit_code"],
        "channel": txn_dict["input_source"],
        "score": result.anomaly_score,
        "label": result.anomaly_label,
        "is_anomaly": result.is_anomaly,
        "top_features": result.top_features or {},
        "phase": phase,
        "latency_ms": latency_ms,
    })
    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Demo transaction generators
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DESCR_CHOICES = ["Fund Transfer", "GIRO", "Payment to IBT", "Charges", "ATM", "PayNow"]
DC_CHOICES = ["D", "C"]
CHANNEL_CHOICES = ["T", "Z"]
CHANNEL_LABELS = {"T": "Teller", "Z": "Online"}

ACCT_DEMO = "1000000000000000042"
ACCT_NORMAL = "1000000000000000099"


def _seed_history():
    """Quietly seed ~80 historical txns per account for realistic features."""
    if st.session_state["seed_done"]:
        return
    service = _get_service()
    rng = random.Random(42)
    np_rng = np.random.RandomState(42)
    seed_start = datetime.now() - timedelta(days=90)

    for acct in [ACCT_DEMO, ACCT_NORMAL]:
        for _ in range(80):
            ts = seed_start + timedelta(
                days=rng.randint(0, 89),
                hours=rng.randint(8, 17),
                minutes=rng.randint(0, 59),
                seconds=rng.randint(0, 59),
            )
            service.score_transaction({
                "account_number": acct,
                "posting_timestamp": ts.isoformat(),
                "transaction_ac_amount": round(float(np_rng.lognormal(7.5, 0.8)), 2),
                "descr": rng.choice(DESCR_CHOICES),
                "debit_credit_code": rng.choice(DC_CHOICES),
                "input_source": rng.choice(CHANNEL_CHOICES),
            })
    st.session_state["seed_done"] = True


def _generate_normal_txn(acct: str, ts: datetime) -> dict:
    return {
        "account_number": acct,
        "posting_timestamp": ts.isoformat(),
        "transaction_ac_amount": round(float(np.random.lognormal(7.5, 0.8)), 2),
        "descr": random.choice(DESCR_CHOICES),
        "debit_credit_code": random.choice(DC_CHOICES),
        "input_source": random.choice(CHANNEL_CHOICES),
    }


def _generate_burst_txn(acct: str, ts: datetime, amount: float = 50_000.0) -> dict:
    return {
        "account_number": acct,
        "posting_timestamp": ts.isoformat(),
        "transaction_ac_amount": amount,
        "descr": "Charges",
        "debit_credit_code": random.choice(DC_CHOICES),
        "input_source": random.choice(["T", "Z"]),
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# UI Components
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def render_header():
    st.markdown("""
    <div class="main-header">
        <h1>🛡️ Real-Time Transaction Anomaly Detection</h1>
        <p>Isolation Forest ML model · 23 engineered features · Model explainability · Sub-10 ms latency</p>
    </div>
    """, unsafe_allow_html=True)


def render_metrics():
    results = st.session_state["results"]
    total = len(results)
    anomalies = sum(1 for r in results if r["is_anomaly"])
    normal = total - anomalies
    avg_latency = np.mean([r["latency_ms"] for r in results]) if results else 0
    rate = (anomalies / total * 100) if total else 0

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">Total Transactions</div>
            <div class="value blue">{total:,}</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">Normal Transactions</div>
            <div class="value green">{normal:,}</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">Anomalous Transactions</div>
            <div class="value red">{anomalies:,}</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">Anomaly Rate</div>
            <div class="value amber">{rate:.1f}%</div>
        </div>""", unsafe_allow_html=True)


def render_score_chart():
    """Anomaly score timeline chart."""
    results = st.session_state["results"]
    if not results:
        return

    df = pd.DataFrame(results)
    df["index"] = range(1, len(df) + 1)
    df["status"] = df["is_anomaly"].map({True: "Anomaly", False: "Normal"})

    fig = go.Figure()

    # Normal points
    normal_df = df[~df["is_anomaly"]]
    fig.add_trace(go.Scatter(
        x=normal_df["index"], y=normal_df["score"],
        mode="markers",
        marker=dict(color="#22c55e", size=7, opacity=0.7,
                    line=dict(width=1, color="#166534")),
        name="Normal",
        hovertemplate=(
            "<b>Transaction #%{x}</b><br>"
            "Score: %{y:.4f}<br>"
            "Amount: $%{customdata[0]:,.2f}<br>"
            "Account: ...%{customdata[1]}<extra></extra>"
        ),
        customdata=list(zip(normal_df["amount"], normal_df["account"].str[-4:])),
    ))

    # Anomaly points
    anom_df = df[df["is_anomaly"]]
    fig.add_trace(go.Scatter(
        x=anom_df["index"], y=anom_df["score"],
        mode="markers",
        marker=dict(color="#ef4444", size=11, symbol="x",
                    line=dict(width=2, color="#991b1b")),
        name="Anomaly",
        hovertemplate=(
            "<b>🚨 Transaction #%{x}</b><br>"
            "Score: %{y:.4f}<br>"
            "Amount: $%{customdata[0]:,.2f}<br>"
            "Account: ...%{customdata[1]}<extra></extra>"
        ),
        customdata=list(zip(anom_df["amount"], anom_df["account"].str[-4:])),
    ))

    # Decision boundary line at 0
    fig.add_hline(y=0, line_dash="dash", line_color="#94a3b8",
                  annotation_text="Decision Boundary",
                  annotation_position="top left",
                  annotation_font_color="#64748b")

    fig.update_layout(
        title=dict(text="Anomaly Scores Over Time", font=dict(size=16)),
        xaxis_title="Transaction #",
        yaxis_title="Anomaly Score (lower = more anomalous)",
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        height=370,
        margin=dict(l=60, r=20, t=50, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(gridcolor="#f1f5f9"),
        yaxis=dict(gridcolor="#f1f5f9", zeroline=True, zerolinecolor="#e2e8f0"),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_feature_importance(result_dict: dict):
    """Horizontal bar chart for SHAP top features."""
    top = result_dict.get("top_features", {})
    if not top:
        st.caption("No SHAP features available.")
        return

    feature_descriptions = {
        "amount_zscore_account": "How far this amount is from the account's usual amount pattern (z-score).",
        "same_amount_frequency": "How often this exact amount has appeared before for this account.",
        "cumulative_amount_10min": "Total transacted amount within the last 10 minutes.",
        "descr_encoded": "Encoded transaction description/type used by the model.",
        "debit_credit_code_encoded": "Encoded debit/credit indicator used by the model.",
        "input_source_encoded": "Encoded transaction channel (for example teller vs online).",
        "hour_of_day": "Hour when the transaction occurred (0-23).",
        "is_business_hours": "Whether the transaction was during business hours on weekdays.",
        "txn_count_1s": "Number of account transactions in the past 1 second.",
        "txn_count_10s": "Number of account transactions in the past 10 seconds.",
        "txn_count_30s": "Number of account transactions in the past 30 seconds.",
        "time_since_last_txn": "Seconds since the previous transaction for this account.",
        "time_since_last_txn_same_amount": "Seconds since the last transaction with the same amount.",
        "time_since_last_txn_same_descr": "Seconds since the last transaction with the same description.",
        "avg_time_gap_last_5txn": "Average time gap between the last 5 transaction intervals.",
        "min_time_gap_last_10txn": "Smallest time gap among the last 10 transaction intervals.",
        "same_acct_amt_descr_count_10min": "Count of transactions with same account, amount, and description in 10 minutes.",
        "same_acct_amt_count_10min": "Count of transactions with same account and amount in 10 minutes.",
        "is_exact_duplicate_10min": "Flag indicating repeated identical amount+description in 10 minutes.",
        "duplicate_streak": "Current streak length of repeated duplicate-like transactions.",
        "dc_sign": "Transaction direction represented as numeric sign (debit/credit).",
        "dc_alternation_count_10txn": "How many debit/credit flips occurred in the last 10 transactions.",
        "reversal_count_10min": "Count of potential reversal-like transactions in the last 10 minutes.",
    }

    feat_df = pd.DataFrame({
        "Feature": list(top.keys()),
        "SHAP Value": list(top.values()),
    }).sort_values("SHAP Value", ascending=False)
    feat_df["Description"] = feat_df["Feature"].map(feature_descriptions).fillna(
        feat_df["Feature"].str.replace("_", " ").str.title()
    )

    colors = ["#ef4444" if v < 0 else "#22c55e" for v in feat_df["SHAP Value"]]

    fig = go.Figure(go.Bar(
        x=feat_df["SHAP Value"],
        y=feat_df["Feature"],
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.4f}" for v in feat_df["SHAP Value"]],
        textposition="outside",
        customdata=feat_df[["Description"]],
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Contributing Factor Value: %{x:+.4f}<br>"
            "Meaning: %{customdata[0]}<extra></extra>"
        ),
    ))
    fig.update_layout(
        title=dict(text="Top Contributing Factors", font=dict(size=14)),
        xaxis_title="Contributing Factor Value (← anomalous | normal →)",
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        height=250,
        margin=dict(l=180, r=60, t=40, b=40),
        xaxis=dict(gridcolor="#f1f5f9", zeroline=True, zerolinecolor="#cbd5e1"),
        yaxis=dict(gridcolor="#f1f5f9"),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_transaction_feed():
    """Live scrolling feed of scored transactions."""
    results = st.session_state["results"]
    if not results:
        st.info("No transactions scored yet. Use the sidebar to submit or run the demo.")
        return

    for r in reversed(results[-50:]):
        cls = "txn-anomaly" if r["is_anomaly"] else "txn-normal"
        badge_cls = "badge-anomaly" if r["is_anomaly"] else "badge-normal"
        badge_text = "ANOMALY" if r["is_anomaly"] else "NORMAL"
        acct_short = "..." + r["account"][-4:]
        channel = CHANNEL_LABELS.get(r["channel"], r["channel"])

        feat_chips = ""
        if r["is_anomaly"] and r.get("top_features"):
            for fname, fval in list(r["top_features"].items())[:3]:
                chip_cls = "feat-chip neg" if fval < 0 else "feat-chip"
                feat_chips += f'<span class="{chip_cls}">{fname}: {fval:+.3f}</span>'

        st.markdown(f"""
        <div class="{cls}">
            <span class="txn-badge {badge_cls}">{badge_text}</span>
            <span style="width:80px; font-weight:600;">{acct_short}</span>
            <span style="width:90px;">{r["timestamp"][11:19] if len(r["timestamp"]) > 11 else r["timestamp"]}</span>
            <span style="width:110px; font-weight:600;">${r["amount"]:,.2f}</span>
            <span style="width:110px; color:#64748b;">{r["descr"]}</span>
            <span style="width:55px; color:#64748b;">{channel}</span>
            <span style="width:60px; color:#64748b;">score: {r["score"]:+.3f}</span>
            <span style="width:50px; color:#94a3b8;">{r["latency_ms"]:.0f}ms</span>
            {feat_chips}
        </div>
        """, unsafe_allow_html=True)


def render_distribution_chart():
    """Score distribution histogram."""
    results = st.session_state["results"]
    if len(results) < 3:
        return

    df = pd.DataFrame(results)
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=df[~df["is_anomaly"]]["score"],
        nbinsx=30, name="Normal",
        marker_color="#22c55e", opacity=0.7,
    ))
    fig.add_trace(go.Histogram(
        x=df[df["is_anomaly"]]["score"],
        nbinsx=30, name="Anomaly",
        marker_color="#ef4444", opacity=0.7,
    ))
    fig.update_layout(
        title=dict(text="Anomaly Score Distribution", font=dict(size=14)),
        barmode="overlay",
        xaxis_title="Anomaly Score",
        yaxis_title="Count",
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        height=280,
        margin=dict(l=50, r=20, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_account_breakdown():
    """Per-account anomaly summary."""
    results = st.session_state["results"]
    if not results:
        return

    df = pd.DataFrame(results)
    summary = df.groupby("account").agg(
        total=("score", "count"),
        anomalies=("is_anomaly", "sum"),
        avg_score=("score", "mean"),
        total_amount=("amount", "sum"),
    ).reset_index()
    summary["account_display"] = "..." + summary["account"].str[-4:]
    summary["anomaly_rate"] = (summary["anomalies"] / summary["total"] * 100).round(1)
    summary = summary.sort_values("anomalies", ascending=False)

    fig = go.Figure(go.Bar(
        x=summary["account_display"],
        y=summary["total"],
        name="Normal",
        marker_color="#22c55e",
    ))
    fig.add_trace(go.Bar(
        x=summary["account_display"],
        y=summary["anomalies"],
        name="Anomalies",
        marker_color="#ef4444",
    ))
    fig.update_layout(
        title=dict(text="Anomalies by Account", font=dict(size=14)),
        barmode="group",
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        height=280,
        margin=dict(l=50, r=20, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Sidebar
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def render_sidebar():
    with st.sidebar:
        st.markdown("## ⚡ Controls")

        # ── Automated Demo ───────────────────────────────────────────
        st.markdown("### 🎬 Automated Demo")
        st.caption("Simulates normal transactions followed by a series of rapid transactions to demonstrate anomaly detection in action.")

        col_a, col_b = st.columns(2)
        with col_a:
            n_normal = st.number_input("Normal txns", min_value=1, max_value=20, value=5, key="n_normal")
        with col_b:
            n_burst = st.number_input("Anomalous txns", min_value=5, max_value=50, value=20, key="n_burst")

        burst_amount = st.number_input(
            "Transaction amount ($)", min_value=1000.0, max_value=500000.0,
            value=50000.0, step=5000.0, key="burst_amt",
        )

        demo_active = st.session_state.get("demo_running", False)
        if st.button("▶ Run Demo", type="primary", use_container_width=True, disabled=demo_active):
            _start_demo(int(n_normal), int(n_burst), burst_amount)

        # Live progress while demo is running
        if demo_active:
            cfg = st.session_state["demo_config"]
            step = st.session_state.get("demo_step", 0)
            n_norm = cfg["n_normal"]
            n_brst = cfg["n_burst"]
            total = n_norm + 3 + n_brst
            pct = min(step / total, 1.0) if total else 1.0
            if step < n_norm:
                phase_text = f"Phase 1: Normal txn {step + 1}/{n_norm}"
            elif step < n_norm + 3:
                phase_text = f"Phase 1: Normal txn (account 2)"
            else:
                burst_i = step - n_norm - 3 + 1
                phase_text = f"Phase 2: 🚨 Rapid txn {burst_i}/{n_brst}"
            st.progress(pct, text=phase_text)

        st.divider()

        # ── Manual Transaction ───────────────────────────────────────
        st.markdown("### 📝 Simulate Streaming Transaction")

        acct = st.text_input("Account Number", value=ACCT_DEMO, key="m_acct")
        amount = st.number_input("Amount ($)", min_value=0.01, value=1500.00, step=100.0, key="m_amt")
        descr = st.selectbox("Type", DESCR_CHOICES, key="m_descr")
        dc = st.radio("Debit / Credit", DC_CHOICES, horizontal=True, key="m_dc")
        channel = st.selectbox("Channel", list(CHANNEL_LABELS.keys()),
                               format_func=lambda x: CHANNEL_LABELS[x], key="m_chan")

        if st.button("🔍 Score Transaction", use_container_width=True):
            txn = {
                "account_number": acct,
                "posting_timestamp": datetime.now().isoformat(),
                "transaction_ac_amount": amount,
                "descr": descr,
                "debit_credit_code": dc,
                "input_source": channel,
            }
            _seed_history()
            result = _score_and_record(txn, phase="manual")
            if result.is_anomaly:
                st.error(f"🚨 ANOMALY detected! Score: {result.anomaly_score:+.4f}")
            else:
                st.success(f"✅ Normal. Score: {result.anomaly_score:+.4f}")

        st.divider()

        # ── Reset ────────────────────────────────────────────────────
        if st.button("🗑️ Clear All Results", use_container_width=True):
            st.session_state["results"] = []
            st.session_state["seed_done"] = False
            if "service" in st.session_state:
                st.session_state["service"].shutdown()
                del st.session_state["service"]
            st.rerun()

        st.divider()
        st.markdown("### ℹ️ Model Info")
        service = _get_service()
        st.markdown(f"""
        - **Model**: Isolation Forest
        - **Features**: 23 engineered
        - **Contamination**: 1%
        - **Explainability**: SHAP TreeExplainer
        - **Accounts tracked**: {service.state_stats['accounts_tracked']}
        - **Records in state**: {service.state_stats['total_records']:,}
        """)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Automated demo flow  (state-machine — one txn per rerun for live updates)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _start_demo(n_normal: int, n_burst: int, burst_amount: float):
    """Seed history, store demo parameters, and kick off the first rerun."""
    _seed_history()
    st.session_state["demo_running"] = True
    st.session_state["demo_step"] = 0
    st.session_state["demo_config"] = {
        "n_normal": n_normal,
        "n_burst": n_burst,
        "burst_amount": burst_amount,
        "base_time": datetime.now().replace(hour=10, minute=0, second=0, microsecond=0).isoformat(),
        "burst_start": datetime.now().replace(hour=8, minute=30, second=0, microsecond=0).isoformat(),
    }
    st.rerun()


def _demo_tick():
    """Process exactly one demo transaction, then rerun so the UI updates live."""
    if not st.session_state.get("demo_running"):
        return

    cfg = st.session_state["demo_config"]
    step = st.session_state["demo_step"]
    n_normal = cfg["n_normal"]
    n_burst = cfg["n_burst"]
    burst_amount = cfg["burst_amount"]
    base_time = datetime.fromisoformat(cfg["base_time"])
    burst_start = datetime.fromisoformat(cfg["burst_start"])
    total_steps = n_normal + 3 + n_burst

    # ── Generate & score one transaction ──────────────────────────
    if step < n_normal:
        ts = base_time + timedelta(hours=step)
        txn = _generate_normal_txn(ACCT_DEMO, ts)
        _score_and_record(txn, phase="normal")
    elif step < n_normal + 3:
        i = step - n_normal
        ts = base_time + timedelta(hours=i * 3)
        txn = _generate_normal_txn(ACCT_NORMAL, ts)
        _score_and_record(txn, phase="normal")
    else:
        offset = random.randint(0, 120)
        ts = burst_start + timedelta(seconds=offset)
        txn = _generate_burst_txn(ACCT_DEMO, ts, burst_amount)
        _score_and_record(txn, phase="burst")

    st.session_state["demo_step"] = step + 1

    if step + 1 >= total_steps:
        st.session_state["demo_running"] = False
    else:
        time.sleep(0.15)
        st.rerun()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main layout
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main():
    render_sidebar()
    render_header()
    render_metrics()

    st.markdown("")  # spacer

    # Charts row
    results = st.session_state["results"]

    if results:
        tab_timeline, tab_dist, tab_acct, tab_details = st.tabs([
            "📈 Transaction Timeline",
            "📊 Anomaly Score Distribution",
            "👥 Account Breakdown",
            "🔬 Transaction Details",
        ])

        with tab_timeline:
            render_score_chart()

        with tab_dist:
            render_distribution_chart()

        with tab_acct:
            render_account_breakdown()

        with tab_details:
            # Detailed table + feature drill-down
            df = pd.DataFrame(results).sort_values("score", ascending=True)
            df_display = df[["account", "timestamp", "amount", "descr", "dc",
                             "channel", "score", "is_anomaly", "phase", "latency_ms"]].copy()
            df_display.columns = ["Account", "Timestamp", "Amount", "Type", "D/C",
                                  "Channel", "Score", "Anomaly?", "Phase", "Latency (ms)"]
            df_display["Amount"] = df_display["Amount"].map("${:,.2f}".format)
            df_display["Score"] = df_display["Score"].map("{:+.4f}".format)
            df_display["Latency (ms)"] = df_display["Latency (ms)"].map("{:.1f}".format)
            df_display["Account"] = "..." + df_display["Account"].str[-4:]

            st.dataframe(
                df_display.style.map(
                    lambda v: "background-color: #fee2e2; color: #991b1b; font-weight: 600"
                    if v is True else "",
                    subset=["Anomaly?"],
                ),
                use_container_width=True,
                height=400,
            )

        # Feature importance for latest anomaly
        st.markdown("---")
        st.markdown("### 🔍 Top Contributing Factors - Information on Latest Anomalous Transaction")
        anomalies_list = [r for r in results if r["is_anomaly"]]
        if anomalies_list:
            latest_anom = anomalies_list[-1]
            col_info, col_chart = st.columns([1, 2])
            with col_info:
                st.markdown(f"""
                **Account**: ...{latest_anom['account'][-4:]}  
                **Amount**: ${latest_anom['amount']:,.2f}  
                **Type**: {latest_anom['descr']}  
                **Channel**: {CHANNEL_LABELS.get(latest_anom['channel'], latest_anom['channel'])}  
                **Anomaly Score**: {latest_anom['score']:+.4f} ℹ️
                """)
                st.caption("ℹ️ **Anomaly Score** — A negative score means the model considers the transaction anomalous (the more negative, the more unusual). A positive score means normal behaviour.")
            with col_chart:
                render_feature_importance(latest_anom)
        else:
            st.caption("No anomalies detected yet. Run the demo or submit transactions to see model explanations.")

    # Transaction feed
    st.markdown("---")
    st.markdown("### 📋 Live Transaction Feed")
    render_transaction_feed()

    # ── Drive the demo state-machine (one txn per rerun) ─────────
    _demo_tick()


if __name__ == "__main__":
    main()
