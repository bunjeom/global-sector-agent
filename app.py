import json
import os
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Global AI Sector & Stock Agent", page_icon="🌍", layout="wide")

st.title("🌍 AI 기반 글로벌(KOSPI · NASDAQ) 섹터·투자 후보 선별 대시보드")
st.caption("핵심 철학: 'LLM은 판단하지 않는다 — 구조가 판단한다.' | Point-in-Time QA & 20d→90d Agentic Loop")

report_path = "latest_global_report.json"
if not os.path.exists(report_path):
    st.warning("아직 생성된 리포트 파일(latest_global_report.json)이 없습니다. 파이프라인을 먼저 실행해 주세요.")
    st.stop()

with open(report_path, "r", encoding="utf-8") as f:
    report = json.load(f)

# 상단 메타 정보 및 QA 배지
col_meta1, col_meta2, col_meta3 = st.columns(3)
col_meta1.metric("🕒 최근 리포트 생성 시각 (KST)", report.get("generated_at", "-"))
for idx, qa in enumerate(report.get("qa_gates", [])):
    col = col_meta2 if idx == 0 else col_meta3
    col.metric(
        f"🛡️ {qa['market']} 지수 교차검증 (QA)",
        f"{qa['status']} (오차 {qa['diff_pct']}%)",
        f"종가: {qa['close']:,} ({qa['date']})"
    )

st.divider()

# 거시경제 스냅샷 (Point-in-Time)
st.subheader("🌐 글로벌 거시경제 스냅샷 (Point-in-Time 검증)")
macro_cols = st.columns(len(report.get("macros", [])) or 1)
for idx, m in enumerate(report.get("macros", [])):
    macro_cols[idx].info(f"**{m['지표']}**\n\n`{m['표기']}`")

st.divider()

# 시장별 탭 구성 (KOSPI / NASDAQ)
df_all = pd.DataFrame(report.get("top12_stocks", []))
tab_kr, tab_us, tab_all = st.tabs(["🇰🇷 KOSPI 선별 리포트", "🇺🇸 NASDAQ 선별 리포트", "📊 글로벌 12종목 통합 비교"])

def render_market_tab(market_name: str, summary_dict: dict):
    c1, c2 = st.columns(2)
    c1.error(f"### 🔥 {market_name} 주목 섹터 1위: **{summary_dict.get('hot_sector')}**\n* 모멘텀(35%) + 거래대금(25%) + 회전율(20%) + 상승확산도(20%)")
    c2.info(f"### ❄️ {market_name} 소외 섹터 1위: **{summary_dict.get('cold_sector')}**\n* 구조적 흑자 필터 통과 + 상대부진(35%) + 저PBR(30%)")

    df_m = df_all[df_all["market"] == market_name].copy()
    if df_m.empty:
        st.write("데이터가 없습니다.")
        return

    fig = px.bar(
        df_m, x="name", y="final_score", color="role", text="final_score",
        title=f"{market_name} 상위 6종목 정량 백분위 종합 점수",
        labels={"name": "종목명", "final_score": "종합 점수(100점 만점)", "role": "선정 구분"}
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### 📋 종목별 정량 지표 및 자율 루프(Agentic Loop) 촉매 검증")
    display_cols = ["role", "ticker", "name", "ret_1m", "roe", "pbr", "debt_ratio", "final_score", "loop_info", "catalyst", "quote"]
    st.dataframe(df_m[display_cols], use_container_width=True, hide_index=True)

with tab_kr:
    render_market_tab("KOSPI", report.get("kospi_summary", {}))

with tab_us:
    render_market_tab("NASDAQ", report.get("nasdaq_summary", {}))

with tab_all:
    st.dataframe(df_all, use_container_width=True, hide_index=True)
