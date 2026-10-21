import datetime
import streamlit as st
import plotly.graph_objects as go
import FinanceDataReader as fdr
from utils import load_report, init_session

st.set_page_config(page_title="종목 딥다이브 탐색기", page_icon="🔎", layout="wide")
init_session()
report, df_all = load_report()

st.title("🔎 개별 종목 딥다이브 (Deep-Dive) 탐색기")

col1, col2, col3 = st.columns([2, 2, 1])
with col1:
    custom_tkr = st.text_input("티커 직접 입력 (예: 005930, NVDA, LITE, SNPS, JEPQ)", value=st.session_state["selected_ticker"])
with col2:
    period_days = st.selectbox("차트 조회 기간", [60, 120, 250, 500], index=1, format_func=lambda x: f"최근 {x}일")
with col3:
    st.write("")
    st.write("")
    if st.button("🏠 홈으로 돌아가기", use_container_width=True):
        st.switch_page("app.py")

matched = df_all[df_all["ticker"] == custom_tkr]
if not matched.empty:
    r_info = matched.iloc[0]
    st.success(
        f"**🤖 AI 에이전트 선별 종목: {r_info['name']} ({r_info['ticker']}) — {r_info['role']}**\n\n"
        f"- **종합점수:** `{r_info['final_score']}점` | **추천 비중:** `{r_info.get('weight_pct', '-')}%` | **청산 규칙:** `{r_info.get('exit_rule', '-')}`\n"
        f"- **투자 결론:** {r_info.get('catalyst', '')}\n"
        f"- **검증 원문(Quote):** `{r_info.get('quote', '')}`"
    )

start_dt = (datetime.date.today() - datetime.timedelta(days=int(period_days))).strftime("%Y-%m-%d")
try:
    df_price = fdr.DataReader(custom_tkr, start_dt)
    if not df_price.empty:
        df_price["MA20"] = df_price["Close"].rolling(20).mean()
        df_price["MA60"] = df_price["Close"].rolling(60).mean()

        last_close = float(df_price["Close"].iloc[-1])
        prev_close = float(df_price["Close"].iloc[-2]) if len(df_price) > 1 else last_close
        pct_chg = (last_close / prev_close - 1.0) * 100.0
        high_p = float(df_price["High"].max())
        low_p = float(df_price["Low"].min())

        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("현재 종가", f"{last_close:,.2f}", f"{pct_chg:+.2f}%")
        mc2.metric("기간 최고가", f"{high_p:,.2f}", f"고점대비 {(last_close/high_p - 1)*100:.1f}%")
        mc3.metric("기간 최저가", f"{low_p:,.2f}", f"저점대비 {(last_close/low_p - 1)*100:+.1f}%")
        mc4.metric("20일 이평선 괴리율", f"{(last_close / df_price['MA20'].iloc[-1] - 1)*100:+.2f}%")

        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df_price.index, open=df_price["Open"], high=df_price["High"],
            low=df_price["Low"], close=df_price["Close"], name="캔들차트"
        ))
        fig.add_trace(go.Scatter(x=df_price.index, y=df_price["MA20"], mode="lines", name="20일선(단기)", line=dict(color="orange", width=1.5)))
        fig.add_trace(go.Scatter(x=df_price.index, y=df_price["MA60"], mode="lines", name="60일선(중기)", line=dict(color="royalblue", width=1.5)))
        fig.update_layout(title=f"[{custom_tkr}] 일봉 캔들스틱 및 이동평균선(20D/60D)", xaxis_rangeslider_visible=False, height=520)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("시세 데이터를 찾을 수 없습니다. 티커명을 확인해 주세요.")
except Exception as e:
    st.error(f"데이터 조회 오류: {e}")
