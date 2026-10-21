import datetime
import pandas as pd
import streamlit as st
import plotly.express as px
import FinanceDataReader as fdr
from utils import load_report, init_session

st.set_page_config(page_title="포트폴리오 백테스터", page_icon="📈", layout="wide")
init_session()
report, df_all = load_report()

st.title("📈 나만의 멀티 에셋 포트폴리오 시뮬레이터")
st.caption("관심 있는 한국/미국 주식이나 ETF 티커들을 쉼표(,)로 넣어 누적 수익률을 비교해 보세요.")

default_tkrs = df_all["ticker"].head(4).tolist() if not df_all.empty else ["005930", "NVDA"]
user_tkrs = st.text_input("비교할 티커 입력 (예: 005930, NVDA, LITE, JEPQ)", value=", ".join(default_tkrs))
lookback = st.slider("백테스트 조회 기간 (일)", 30, 365, 120)

if st.button("시뮬레이션 실행 🚀", type="primary"):
    tkr_list = [t.strip() for t in user_tkrs.split(",") if t.strip()]
    start_dt = (datetime.date.today() - datetime.timedelta(days=lookback)).strftime("%Y-%m-%d")
    norm_df = pd.DataFrame()
    for t in tkr_list:
        try:
            p = fdr.DataReader(t, start_dt)["Close"]
            if not p.empty:
                norm_df[t] = (p / p.iloc[0] - 1.0) * 100.0
        except Exception:
            pass
    if not norm_df.empty:
        norm_df["동일비중_포트폴리오_합산"] = norm_df.mean(axis=1)
        fig = px.line(norm_df, title=f"최근 {lookback}일 누적 수익률 비교 (%)", labels={"value": "수익률 (%)", "index": "날짜"})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("유효한 주가 데이터를 불러오지 못했습니다.")
