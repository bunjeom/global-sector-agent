import streamlit as st
import plotly.express as px
from utils import load_report, init_session

st.set_page_config(page_title="커스텀 팩터 실험실", page_icon="🎛️", layout="wide")
init_session()
report, df_all = load_report()

st.title("🎛️ 내 맘대로 퀀트 가중치 실험실 (Interactive Factor Lab)")
st.caption("슬라이더를 움직여 내가 원하는 팩터(모멘텀 vs ROE 퀄리티 vs 저PBR 밸류) 비중을 조절하면 실시간으로 순위가 재계산됩니다.")

c1, c2, c3, c4 = st.columns(4)
w_mom = c1.slider("🚀 단기 모멘텀(20D) 비중", 0, 100, 35, 5)
w_mid = c2.slider("🌊 중기 추세(60D) 비중", 0, 100, 20, 5)
w_roe = c3.slider("💎 수익성(ROE) 비중", 0, 100, 25, 5)
w_pbr = c4.slider("🏷️ 저평가(1/PBR) 비중", 0, 100, 20, 5)

total_w = w_mom + w_mid + w_roe + w_pbr or 1
df_lab = df_all.copy()
df_lab["mom_rank"] = df_lab["ret_1m"].rank(pct=True) * 100
df_lab["mid_rank"] = df_lab["ret_3m"].rank(pct=True) * 100
df_lab["roe_rank"] = df_lab["roe"].rank(pct=True) * 100
df_lab["pbr_rank"] = df_lab["pbr"].rank(pct=True, ascending=False) * 100

df_lab["내_커스텀_점수"] = (
    (w_mom * df_lab["mom_rank"] + w_mid * df_lab["mid_rank"] + w_roe * df_lab["roe_rank"] + w_pbr * df_lab["pbr_rank"]) / total_w
).round(2)

df_lab = df_lab.sort_values("내_커스텀_점수", ascending=False).reset_index(drop=True)
fig = px.bar(df_lab, x="name", y=["내_커스텀_점수", "final_score"], barmode="group", title="기존 에이전트 점수 vs 내 커스텀 점수 비교")
st.plotly_chart(fig, use_container_width=True)
st.dataframe(df_lab[["market", "ticker", "name", "내_커스텀_점수", "final_score", "ret_1m", "ret_3m", "roe", "pbr"]], use_container_width=True, hide_index=True)
