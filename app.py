import streamlit as st
from utils import load_report, init_session

st.set_page_config(page_title="글로벌 퀀트 리서치 터미널", page_icon="🏠", layout="wide")
init_session()
report, df_all = load_report()

th = report.get("time_horizon", {})
regime = report.get("macro_regime", {})
exec_conc = report.get("executive_conclusion", {})

st.title("🏠 글로벌 마켓 & Top-Down 포트폴리오 홈")
st.info(
    f"**💡 오늘의 핵심 결론:** {exec_conc.get('headline', '')}\n\n"
    f"- **거시 국면 진단:** `{regime.get('regime_name', '')}` — {regime.get('rationale', '')}\n"
    f"- **분석 시계열:** 기준일 `{th.get('anchor_date_t0', '-')}` | 목표 보유 `{th.get('target_holding_period', '-')}`"
)

# 거시 지표 & 지수 QA 배지
items_count = len(report.get("macros", [])) + len(report.get("qa_gates", []))
m_cols = st.columns(items_count if items_count > 0 else 1)
idx_c = 0
for m in report.get("macros", []):
    m_cols[idx_c].metric(m["지표"], m["표기"].split("|")[0].replace("[", "").strip(), m["표기"])
    idx_c += 1
for qa in report.get("qa_gates", []):
    m_cols[idx_c].metric(f"{qa['market']} 지수", f"{qa['close']:,}", f"QA {qa['status']} ({qa['date']})")
    idx_c += 1

st.divider()
st.subheader("🎯 오늘의 선별 12종목 (💡 표의 행을 마우스로 클릭하면 상세 차트 페이지로 이동합니다!)")

market_filter = st.radio("시장 필터 선택", ["전체 (All)", "KOSPI", "NASDAQ"], horizontal=True)
df_view = df_all.copy()
if market_filter in ["KOSPI", "NASDAQ"]:
    df_view = df_view[df_view["market"] == market_filter].reset_index(drop=True)

cols_show = ["market", "role", "ticker", "name", "weight_pct", "ret_1m", "ret_3m", "roe", "pbr", "final_score", "catalyst"]
cols_show = [c for c in cols_show if c in df_view.columns]

event = st.dataframe(
    df_view[cols_show],
    use_container_width=True,
    hide_index=True,
    on_select="rerun",
    selection_mode="single-row"
)

if event.selection.rows:
    sel_idx = event.selection.rows[0]
    sel_row = df_view.iloc[sel_idx]
    st.success(f"📌 선택된 종목: **{sel_row['name']} ({sel_row['ticker']})** | 투자 논리: {sel_row.get('catalyst', '')}")
    if st.button(f"🔎 [{sel_row['name']}] 딥다이브 차트 페이지로 바로 이동하기 🚀", type="primary"):
        st.session_state["selected_ticker"] = sel_row["ticker"]
        st.session_state["selected_name"] = sel_row["name"]
        st.switch_page("pages/1_🔎_종목_딥다이브.py")
