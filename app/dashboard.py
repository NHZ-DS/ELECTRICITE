"""Streamlit dashboard.

    streamlit run app/dashboard.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from gridcast.config import REPORTS_DIR  # noqa: E402
from gridcast.data.store import Store  # noqa: E402
from gridcast.pipeline import MAIN_MODEL  # noqa: E402

st.set_page_config(page_title="GridCast FR", page_icon="⚡", layout="wide")

MODEL_LABELS = {
    "hybrid_physics_lgbm": "Hybrid (physics + LightGBM)",
    "lightgbm": "LightGBM (data-driven)",
    "thermal_physics": "Thermal physics model",
    "seasonal_naive": "Seasonal naive (last week)",
}


@st.cache_data(ttl=600)
def load_data():
    store = Store(read_only=True)
    try:
        return store.load_dataset(), store.load_forecasts()
    finally:
        store.close()


def read_json(name: str) -> dict | None:
    path = REPORTS_DIR / name
    return json.loads(path.read_text()) if path.exists() else None


data, forecasts = load_data()
report, monitoring = read_json("backtest_report.json"), read_json("monitoring.json")

st.title("⚡ GridCast FR")
st.caption("Day-ahead forecast of French electricity consumption, with calibrated uncertainty and drift monitoring.")

if forecasts.empty:
    st.warning("No forecast yet. Run `python -m gridcast.cli replay` then `python -m gridcast.cli forecast`.")
    st.stop()

model = st.sidebar.selectbox("Model", list(MODEL_LABELS), format_func=MODEL_LABELS.get,
                             index=list(MODEL_LABELS).index(MAIN_MODEL))
days = st.sidebar.slider("Days shown", 3, 30, 10)

fc = (forecasts[forecasts["model"] == model]
      .sort_values("issued_at").drop_duplicates("ts", keep="last")  # latest forecast for each hour
      .sort_values("ts"))
latest_issue = fc["issued_at"].max()
next_day = fc[fc["issued_at"] == latest_issue]

# ----------------------------------------------------------------- headline numbers
c1, c2, c3, c4 = st.columns(4)
peak = next_day.loc[next_day["yhat"].idxmax()]
c1.metric("Next-day peak", f"{peak.yhat / 1000:.1f} GW", help=f"at {peak.ts:%H:%M}")
c2.metric("Peak range (90 %)", f"{peak.lower / 1000:.1f} – {peak.upper / 1000:.1f} GW")
if report:
    row = next((r for r in report["summary"] if r["model"] == model), None)
    if row:
        c3.metric("Backtest MAPE", f"{row['mape_pct']:.2f} %", help=f"over {report['test_weeks']} weeks")
    c4.metric("Heating sensitivity", f"{report['heating_sensitivity_mw_per_degC'] / 1000:.2f} GW/°C",
              help="Extra demand for each degree of cold below 15 °C, learned by the physics model")

# ----------------------------------------------------------------- main chart
cutoff = latest_issue - pd.Timedelta(days=days)
hist = data[(data["ts"] >= cutoff)]
shown = fc[fc["ts"] >= cutoff]
fig = go.Figure()
fig.add_trace(go.Scatter(x=shown["ts"], y=shown["upper"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
fig.add_trace(go.Scatter(x=shown["ts"], y=shown["lower"], fill="tonexty", line=dict(width=0),
                         fillcolor="rgba(31, 119, 180, 0.18)", name="90 % interval"))
fig.add_trace(go.Scatter(x=shown["ts"], y=shown["yhat"], name="Forecast", line=dict(color="#1f77b4", width=2)))
fig.add_trace(go.Scatter(x=hist["ts"], y=hist["load_mw"], name="Actual (RTE)", line=dict(color="#222", width=1.5)))
fig.update_layout(height=430, margin=dict(l=10, r=10, t=30, b=10), yaxis_title="MW",
                  legend=dict(orientation="h", y=1.08), hovermode="x unified")
st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------------- model comparison + monitoring
left, right = st.columns([3, 2])
with left:
    st.subheader("Model comparison (rolling backtest)")
    if report:
        table = pd.DataFrame(report["summary"]).assign(model=lambda d: d["model"].map(MODEL_LABELS))
        table = table.rename(columns={"model": "Model", "mae_mw": "MAE (MW)", "rmse_mw": "RMSE (MW)",
                                      "mape_pct": "MAPE (%)", "coverage_pct": "Coverage (%)",
                                      "mean_width_mw": "Interval width (MW)", "mape_cold_pct": "MAPE < 5 °C (%)"})
        st.dataframe(table.round(2), hide_index=True, use_container_width=True)
    else:
        st.info("Run `python -m gridcast.cli backtest`.")
with right:
    st.subheader("Monitoring")
    if monitoring:
        badge = "🟢 OK" if monitoring["status"] == "OK" else "🔴 ALERT"
        st.markdown(f"**Status: {badge}** — last {monitoring['recent_days']} days")
        for c in monitoring["checks"]:
            st.markdown(f"{'🔴' if c['alert'] else '🟢'} {c['message']}")
    else:
        st.info("Run `python -m gridcast.cli monitor`.")
