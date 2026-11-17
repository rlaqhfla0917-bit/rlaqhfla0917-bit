"""FabPulse Streamlit dashboard: real held-out scores or an unmistakable synthetic preview."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"

st.set_page_config(page_title="FabPulse | 반도체 품질 점검", page_icon="📊", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
    .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1450px; }
    h1,h2,h3 { letter-spacing: -.03em; }
    [data-testid="stMetric"] {border:1px solid #dbe5ea; border-radius:14px; padding: 13px 16px; background:#f7fafc;}
    [data-testid="stMetricLabel"] {color:#475569;}
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_results() -> tuple[dict, pd.DataFrame, pd.DataFrame] | None:
    metric_path = ARTIFACTS / "metrics.json"
    if not metric_path.exists() or not (ARTIFACTS / "test_predictions.csv").exists():
        return None
    doc = json.loads(metric_path.read_text(encoding="utf-8"))
    pred = pd.read_csv(ARTIFACTS / "test_predictions.csv", parse_dates=["timestamp"])
    imp = pd.read_csv(ARTIFACTS / "feature_importance.csv")
    return doc, pred, imp


def demo_results() -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Purely synthetic mock-up; never mix this with SECOM experimental evidence."""
    rng = np.random.default_rng(2030)
    n = 210
    y = (rng.random(n) < 0.10).astype(int)
    scores = rng.uniform(0, .55, n) + y * rng.uniform(.15, .6, n)
    scores = np.minimum(scores, 1.0)
    dates = pd.date_range("2026-01-01", periods=n, freq="h")
    pred = pd.DataFrame({
        "sample_id": [f"MOCK-{i:04d}" for i in range(1, n + 1)],
        "timestamp": dates,
        "true_fail": y, "risk_score": scores,
    }).sort_values("risk_score", ascending=False).reset_index(drop=True)
    from src.modeling import screening_metrics
    metrics = screening_metrics(pred["true_fail"], pred["risk_score"])
    importance = pd.DataFrame({
        "variable": [f"sensor_{i:03d}" for i in range(1, 16)],
        "importance": np.sort(rng.uniform(.03, .22, 15))[::-1],
        "method": "synthetic_mock_values",
    })
    return {"model_name": "가상 예시 (학습된 AI 없음)", "test_metrics": metrics,
            "total_samples": n, "features": 590, "total_failures": int(sum(y)),
            "split": "NOT APPLICABLE - synthetic preview only",
            "source": "SYNTHETIC MOCK DATA - NOT SECOM"}, pred, importance


def plot_capture(pred: pd.DataFrame, budget_pct: int) -> go.Figure:
    ordered = pred.sort_values("risk_score", ascending=False)
    n, total = len(ordered), int(ordered["true_fail"].sum())
    x = np.arange(1, n + 1) / n * 100
    captured = np.cumsum(ordered["true_fail"].to_numpy()) / max(total, 1) * 100
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.r_[0, x], y=np.r_[0, captured], mode="lines", name="모델 위험도 순위", line=dict(color="#0f766e", width=3)))
    fig.add_trace(go.Scatter(x=[0, 100], y=[0, 100], mode="lines", name="무작위 점검 기대값", line=dict(color="#94a3b8", dash="dash")))
    fig.add_vline(x=budget_pct, line_width=1.5, line_dash="dot", line_color="#b45309")
    fig.update_layout(title="점검 비율에 따른 누적 불량 포착률", xaxis_title="우선 점검할 샘플 비율 (%)", yaxis_title="실제 불량 포착률 (%)", xaxis_range=[0, 100], yaxis_range=[0, 105], height=410, margin=dict(l=25, r=25, t=50, b=20), legend=dict(orientation="h", y=-.25))
    return fig


def plot_risk(pred: pd.DataFrame) -> go.Figure:
    colors = np.where(pred["true_fail"].to_numpy() == 1, "#dc6a4d", "#0f766e")
    fig = go.Figure(go.Histogram(x=pred["risk_score"], nbinsx=25, marker_color="#0f766e", opacity=.8))
    fig.update_layout(title="보류 평가 데이터의 상대 위험 점수 분포", xaxis_title="모델 점수 (확률 보정되지 않음)", yaxis_title="샘플 수", height=360, margin=dict(l=25, r=25, t=50, b=20))
    return fig


st.title("FabPulse")
st.caption("Semiconductor Quality Screening | 제조 데이터 기반 불량 위험 분류·점검 우선순위 지원")
real = load_results()
with st.sidebar:
    st.markdown("### 프로젝트 안내")
    st.caption("UCI SECOM · 공개 반도체 제조 데이터 · 2008년")
    st.markdown("**분석 목적**")
    st.write("불량 위험도가 높은 샘플을 우선 선별하고, 제한된 점검 비율에서 실제 불량이 얼마나 포착되는지 검증합니다.")
    st.markdown("**데이터 관련 한계**")
    st.caption("센서의 실명·단위, 설비 ID 및 불량 원인은 공개되지 않았습니다. 실시간 FAB 연결이나 인과 추정은 수행하지 않습니다.")
    st.caption("원본: UCI Machine Learning Repository · SECOM (McCann & Johnston, 2008), CC BY 4.0")
    st.link_button("공식 데이터 출처", "https://archive.ics.uci.edu/dataset/179/secom")

if real is None:
    st.warning("현재 공식 데이터로 학습한 결과 파일이 없습니다. 아래의 모든 수치와 차트는 **화면 동작 확인을 위한 합성 시연 데이터**입니다. 실제 SECOM 모델 성능이 아닙니다.", icon="⚠️")
    st.code("python scripts/download_data.py\npython train.py\nstreamlit run app.py", language="powershell")
    info, pred, importance = demo_results()
    is_demo = True
else:
    info, pred, importance = real
    st.success("실제 UCI SECOM 데이터로 학습하고 별도 보류 테스트 구간에서 평가한 결과입니다.")
    is_demo = False

from src.modeling import screening_metrics

st.markdown("### 01 · 제조 품질 요약")
metrics = info["test_metrics"]
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("평가 샘플", f"{len(pred):,}개")
with c2:
    st.metric("실제 불량", f"{int(pred['true_fail'].sum()):,}건", f"{pred['true_fail'].mean():.1%} 비율", delta_color="off")
with c3:
    st.metric("평균 정밀도 (AP)", f"{metrics['average_precision']:.3f}" if metrics["average_precision"] is not None else "측정 불가")
with c4:
    st.metric("선택 모델", info["model_name"])

st.divider()
st.markdown("### 02 · 품질 점검 우선순위 시뮬레이션")
budget = st.slider("점검할 샘플의 비율 (%)", min_value=5, max_value=50, step=5, value=10,
                   help="점수 상위 N%를 점검한다는 가정입니다. 실제 기업의 검사 정책이 아닙니다.")
m = screening_metrics(pred["true_fail"], pred["risk_score"], budget / 100)
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.metric("우선 점검", f"{m['review_count']:,}개")
with k2:
    st.metric("포착된 실제 불량", f"{m['caught_failures']:,}건")
with k3:
    st.metric("불량 포착률", f"{m['recall_at_budget']:.1%}" if m['recall_at_budget'] is not None else "해당 없음")
with k4:
    st.metric("무작위 점검 대비 Lift", f"{m['lift_at_budget']:.2f}배" if m['lift_at_budget'] is not None else "해당 없음")

p1, p2 = st.columns([1.35, 1])
with p1:
    st.plotly_chart(plot_capture(pred, budget), use_container_width=True)
with p2:
    st.plotly_chart(plot_risk(pred), use_container_width=True)

st.caption("AP는 테스트 양성 비율(무작위 기준)과 비교해 해석해야 합니다. 점검 후 재작업·비용절감 효과는 실제 데이터로 검증되지 않았습니다.")

st.markdown("### 03 · 점검 대상 상세")
ranked = pred.sort_values("risk_score", ascending=False).reset_index(drop=True).copy()
ranked.insert(0, "순위", range(1, len(ranked) + 1))
ranked["우선 점검"] = np.where(ranked["순위"] <= m["review_count"], "대상", "비대상")
ranked = ranked.rename(columns={"sample_id": "샘플 ID", "timestamp": "검사시각", "true_fail": "실제 불량(1=불량)", "risk_score": "상대 위험 점수"})
st.dataframe(ranked.head(100), use_container_width=True, hide_index=True)
st.download_button("전체 평가 결과 CSV", ranked.to_csv(index=False).encode("utf-8-sig"), file_name="fabpulse_screening_results.csv", mime="text/csv")
st.caption("실제 불량 판정은 과거 데이터 검증 목적이며, 현실의 미검사 샘플에서는 정답을 미리 알 수 없습니다.")

st.divider()
st.markdown("### 04 · 분류에 중요했던 측정 변수")
imp = importance.head(15).sort_values("importance")
fig = go.Figure(go.Bar(y=imp["variable"], x=imp["importance"], orientation="h", marker_color="#0f766e"))
fig.update_layout(xaxis_title="모델 내 상대 중요도 (모델별 척도 상이)", yaxis_title=None, height=430, margin=dict(l=30, r=20, t=10, b=30))
st.plotly_chart(fig, use_container_width=True)
st.info("변수 번호는 익명화된 식별자입니다. 중요도가 높다고 실제 불량 원인이라고 확정할 수 없으며, 특정 장비나 공정 조건으로 해석할 수 없습니다.")

st.divider()
st.markdown("### 05 · 분석 절차 및 신뢰성")
st.write("**시간순 학습 60% → 검증 20% → 최종 테스트 20%**. 결측값 대체·상수 변수 제거·변수 선택은 학습 파이프라인 내부에서 학습 구간만 이용해 수행합니다.")
st.write("검증 구간에서 점검 비율 10%의 불량 포착률과 AP를 비교해 모델을 선택한 다음, 선택된 모델을 학습+검증 구간에서 다시 학습하고 테스트를 1회 평가합니다.")
if not is_demo:
    st.json({"데이터 출처": info["source"], "데이터 건수": info["total_samples"], "전체 불량": info["total_failures"], "선택 모델": info["model_name"], "테스트 기간": info["test_period"], "평가 지표": info["test_metrics"]}, expanded=False)
    st.markdown("#### 사용자 측정 데이터 평가 (선택 기능)")
    st.caption("같은 590개 측정 변수(sensor_001~sensor_590)를 가진 CSV를 업로드할 때만 모델 점수를 계산합니다. 다른 장비·공정의 데이터와 섞지 마세요.")
    uploaded = st.file_uploader("입력 CSV 업로드", type="csv")
    if uploaded is not None:
        expected = [f"sensor_{i:03d}" for i in range(1, info["features"] + 1)]
        df = pd.read_csv(uploaded)
        if set(df.columns) != set(expected):
            st.error(f"CSV에 지정된 측정 변수 {len(expected)}개가 필요합니다. 현재 {len(df.columns)}개 열이 있습니다.")
        else:
            import joblib
            model = joblib.load(ARTIFACTS / "model.joblib")  # local trusted model created by train.py
            scores = model.predict_proba(df[expected].apply(pd.to_numeric, errors="coerce"))[:, 1]
            result = pd.DataFrame({"row": range(1, len(scores) + 1), "risk_score": scores}).sort_values("risk_score", ascending=False)
            st.dataframe(result, use_container_width=True, hide_index=True)
            st.caption("위 점수는 보정된 불량 확률이 아닌 상대 위험 점수입니다. 원 데이터 분포와 다른 측정값의 신뢰성은 검증하지 않았습니다.")
else:
    st.caption("원본 데이터를 확보하고 train.py를 실행하면 이 화면은 합성 시연에서 실제 학습 결과로 자동 전환됩니다.")
