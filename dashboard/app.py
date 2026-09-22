from pathlib import Path
import json

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "transactions.parquet"
CSV_DATA = ROOT / "data" / "transactions.csv"
METRICS = ROOT / "reports" / "metrics.json"

st.set_page_config(page_title="SentinelPay Control Room", page_icon="🛡️", layout="wide")
st.markdown("""
<style>
.block-container {max-width: 1400px; padding-top: 2rem;}
[data-testid="stMetricValue"] {font-size: 2rem; color: #0f766e;}
div[data-testid="stMetric"] {background: #f0fdfa; border: 1px solid #99f6e4; padding: 1rem; border-radius: 14px;}
</style>
""", unsafe_allow_html=True)
st.title("🛡️ SentinelPay Control Room")
st.caption("支付风控模型运营看板 · 时间切分 · 成本敏感阈值 · 可解释决策")

if (not DATA.exists() and not CSV_DATA.exists()) or not METRICS.exists():
    st.warning("请先运行 `python -m sentinelpay.pipeline` 生成数据和模型产物。")
    st.stop()

df = pd.read_parquet(DATA) if DATA.exists() else pd.read_csv(CSV_DATA)
metrics = json.loads(METRICS.read_text(encoding="utf-8"))
left, mid, right, last = st.columns(4)
left.metric("测试集 PR-AUC", metrics["pr_auc"])
mid.metric("欺诈召回率", f'{metrics["recall"]:.1%}')
right.metric("误报率", f'{metrics["false_positive_rate"]:.1%}')
last.metric("业务阈值", metrics["threshold"])

st.divider()
col1, col2 = st.columns([1.35, 1])
with col1:
    st.subheader("风险趋势")
    trend = df.assign(timestamp=pd.to_datetime(df["timestamp"])).set_index("timestamp").resample("D").agg(
        fraud_rate=("is_fraud", "mean"), volume=("is_fraud", "size")
    )
    st.line_chart(trend[["fraud_rate"]], height=300)
with col2:
    st.subheader("渠道风险画像")
    channel = df.groupby("channel", as_index=False).agg(
        transactions=("is_fraud", "size"), fraud_rate=("is_fraud", "mean"), avg_amount=("amount", "mean")
    ).sort_values("fraud_rate", ascending=False)
    st.dataframe(channel.style.format({"fraud_rate": "{:.2%}", "avg_amount": "¥{:,.0f}"}), use_container_width=True, hide_index=True)

st.subheader("重点风险分群")
segment = df.groupby(["merchant_category", "country"], as_index=False).agg(
    transactions=("is_fraud", "size"), fraud_rate=("is_fraud", "mean"), avg_amount=("amount", "mean")
).query("transactions >= 50").sort_values("fraud_rate", ascending=False).head(12)
st.dataframe(segment.style.format({"fraud_rate": "{:.2%}", "avg_amount": "¥{:,.2f}"}), use_container_width=True, hide_index=True)
