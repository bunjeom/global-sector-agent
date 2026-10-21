import json
import os
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Global AI Quant Agent v2.0", page_icon="🎯", layout="wide")

st.title("🎯 [v2.0] Top-Down 인과 구조 & 시계열 정렬 글로벌 AI 섹터·종목 에이전트")
st.caption("Architecture: Stage 1 (Macro Regime) ➔ Stage 2 (Sector Rotation) ➔ Stage 3 (Causal Stock Filtering) ➔ Stage 4 (Agentic Loop & Thesis)")

report_path = "latest_global_report.json"
if not os.path.exists(report_path):
    st.warning("latest_global_report.json 파일이 없습니다.")
    st.stop()

with open(report_path, "r", encoding="utf-8") as f:
    report = json.load(f)

th = report.get("time_horizon", {})
regime = report.get("macro_regime", {})
exec_conc = report.get("executive_conclusion", {})

# 1. 두괄식 최종 투자 결론 (Executive Conclusion)
st.subheader("1️⃣ 최종 투자 결론 (Executive Conclusion - 두괄식 요약)")
st.success(
    f"**💡 {exec_conc.get('headline', '')}**\n\n"
    f"- **Stage 1 거시 국면 진단 (`{regime.get('regime_name', '')}`):** {regime.get('rationale', '')}\n"
    f"- **🇰🇷 KOSPI 포트폴리오 배분:** {exec_conc.get('kospi_action', '')}\n"
    f"- **🇺🇸 NASDAQ 포트폴리오 배분:** {exec_conc.get('nasdaq_action', '')}\n"
    f"- **🛡️ 청산 및 리스크 관리 규칙:** {exec_conc.get('risk_management', '')}"
)

# 2. 시계열 프레임 정의 & 데이터 품질 게이트
st.subheader("2️⃣ 분석 시계열 프레임(Time Horizon) 및 Point-in-Time 거시 스냅샷")
c_th, c_qa = st.columns([3, 2])
with c_th:
    df_th = pd.DataFrame([
        {"시간축 구분": "분석 기준일 (T0)", "적용 기간 및 정의": th.get("anchor_date_t0", "-")},
        {"시간축 구분": "단기 관측 (T-20영업일)", "적용 기간 및 정의": th.get("lookback_short_20d", "-")},
        {"시간축 구분": "중기 관측 (T-60영업일)", "적용 기간 및 정의": th.get("lookback_mid_60d", "-")},
        {"시간축 구분": "수급 관측 (5D vs 20D)", "적용 기간 및 정의": th.get("lookback_flow", "-")},
        {"시간축 구분": "목표 보유기간 (+20영업일)", "적용 기간 및 정의": th.get("target_holding_period", "-")}
    ])
    st.dataframe(df_th, use_container_width=True, hide_index=True)

with c_qa:
    for qa in report.get("qa_gates", []):
        st.metric(f"🛡️ {qa['market']} 지수 QA ({qa['date']})", f"{qa['status']} (오차 {qa['diff_pct']}%)", f"종가: {qa['close']:,}")
    macro_line = " | ".join([f"**{m['지표']}** `{m['표기']}`" for m in report.get("macros", [])])
    st.info(f"🌐 **거시 지표:** {macro_line}")

st.divider()

# 3. 시장별 Top-Down 분석 결과 탭
st.subheader("3️⃣ 시장별 섹터 스코어링 및 최종 12종목 실행 포트폴리오")
df_all = pd.DataFrame(report.get("top12_stocks", []))
tab_kr, tab_us, tab_all = st.tabs(["🇰🇷 KOSPI 포트폴리오 (6종목)", "🇺🇸 NASDAQ 포트폴리오 (6종목)", "📊 글로벌 12종목 통합 뷰"])

def render_market_tab(market_name: str, summary_dict: dict, sec_records: list):
    c1, c2 = st.columns(2)
    c1.error(f"#### 🔥 {market_name} 주목 섹터 1위: **{summary_dict.get('hot_sector')}** (비중 {regime.get('hot_allocation_pct', 60)}%)")
    c2.info(f"#### ❄️ {market_name} 소외 섹터 1위: **{summary_dict.get('cold_sector')}** (비중 {regime.get('cold_allocation_pct', 40)}%, 흑자필터 통과)")

    if sec_records:
        with st.expander(f"🔎 {market_name} 전체 섹터 정량 백분위 스코어링 상세 보기"):
            st.dataframe(pd.DataFrame(sec_records), use_container_width=True, hide_index=True)

    df_m = df_all[df_all["market"] == market_name].copy()
    if df_m.empty:
        return

    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        fig_score = px.bar(
            df_m, x="name", y="final_score", color="role", text="final_score",
            title=f"{market_name} 선별 6종목 정량 백분위 점수"
        )
        st.plotly_chart(fig_score, use_container_width=True)
    with col_chart2:
        fig_weight = px.pie(
            df_m, names="name", values="weight_pct", color="role",
            title=f"{market_name} 거시 국면 기반 종목별 추천 포트폴리오 비중 (%)"
        )
        st.plotly_chart(fig_weight, use_container_width=True)

    cols = ["role", "ticker", "name", "weight_pct", "ret_1m", "ret_3m", "roe", "pbr", "final_score", "exit_rule", "catalyst", "quote"]
    existing_cols = [c for c in cols if c in df_m.columns]
    st.dataframe(df_m[existing_cols], use_container_width=True, hide_index=True)

with tab_kr:
    render_market_tab("KOSPI", report.get("kospi_summary", {}), report.get("kospi_sectors_table", []))

with tab_us:
    render_market_tab("NASDAQ", report.get("nasdaq_summary", {}), report.get("nasdaq_sectors_table", []))

with tab_all:
    st.dataframe(df_all, use_container_width=True, hide_index=True)
