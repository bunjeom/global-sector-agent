import datetime
import pandas as pd
import streamlit as st
import plotly.express as px
import FinanceDataReader as fdr
from utils import load_report, init_session
import global_pipeline

st.set_page_config(page_title="글로벌 섹터 히트맵", page_icon="🗺️", layout="wide")
init_session()
report, df_all = load_report()

st.title("🗺️ 글로벌 섹터 & 종목 머니플로우 히트맵 (Finviz Style)")
st.caption("💡 팁: 차트 안의 섹터 이름을 마우스로 클릭하면 해당 섹터만 크게 확대(Zoom-in)되고, 상단 바를 누르면 다시 전체 화면으로 돌아옵니다!")

# 1. 상단 컨트롤 패널
c1, c2, c3 = st.columns([2, 2, 2])
with c1:
    map_mode = st.radio(
        "📊 히트맵 대상 선택",
        ["🎯 오늘의 선별 12종목 (즉시 표시)", "🌐 전체 유니버스 45종목 (실시간 시세 조회)"],
        horizontal=False
    )
with c2:
    color_metric = st.selectbox(
        "🎨 색상 기준 (무엇을 색깔로 볼까요?)",
        ["ret_1m", "ret_3m", "roe"],
        format_func=lambda x: {
            "ret_1m": "단기 모멘텀 (20영업일 수익률 %)",
            "ret_3m": "중기 추세 (60영업일 수익률 %)",
            "roe": "수익성 퀄리티 (ROE %)"
        }[x]
    )
with c3:
    size_metric = st.selectbox(
        "📐 사각형 크기 기준 (무엇을 면적으로 볼까요?)",
        ["weight_or_score", "equal", "roe_size"],
        format_func=lambda x: {
            "weight_or_score": "포트폴리오 비중 / 정량 점수 비례",
            "equal": "모든 종목 동일 크기 (1:1)",
            "roe_size": "ROE(수익성) 크기 비례"
        }[x]
    )

# 2. 데이터 준비 (선별 12종목 vs 전체 45종목 실시간 캐싱)
@st.cache_data(ttl=1800, show_spinner="🌐 KOSPI 25개 + NASDAQ 20개 전체 종목의 실시간 시세를 수집하여 히트맵을 그리는 중입니다 (최초 1회 약 15초 소요)...")
def fetch_full_universe_df():
    start_dt = (datetime.date.today() - datetime.timedelta(days=115)).strftime("%Y-%m-%d")
    rows = []
    all_uni = (
        [("KOSPI", t, n, s, r, p) for t, n, _, s, r, p, _ in global_pipeline.KOSPI_UNIVERSE] +
        [("NASDAQ", t, n, s, r, p) for t, n, s, r, p, _ in global_pipeline.NASDAQ_UNIVERSE]
    )
    for mkt, tkr, name, sec, roe, pbr in all_uni:
        ret_20d, ret_60d = 0.0, 0.0
        try:
            df_p = fdr.DataReader(tkr, start_dt)
            if len(df_p) >= 61:
                c = df_p["Close"]
                ret_20d = float((c.iloc[-1] / c.iloc[-21] - 1.0) * 100.0)
                ret_60d = float((c.iloc[-1] / c.iloc[-61] - 1.0) * 100.0)
            elif len(df_p) >= 22:
                c = df_p["Close"]
                ret_20d = float((c.iloc[-1] / c.iloc[-21] - 1.0) * 100.0)
                ret_60d = float((c.iloc[-1] / c.iloc[0] - 1.0) * 100.0)
        except Exception:
            pass
        rows.append({
            "market": mkt, "role": sec, "ticker": tkr, "name": name,
            "ret_1m": round(ret_20d, 2), "ret_3m": round(ret_60d, 2),
            "roe": roe, "pbr": pbr, "weight_pct": 10.0, "final_score": round(max(roe, 5.0), 1)
        })
    return pd.DataFrame(rows)

if "12종목" in map_mode:
    df_map = df_all.copy()
else:
    df_map = fetch_full_universe_df()

# 사각형 크기 컬럼 계산 (0 이하 값 방지)
if size_metric == "weight_or_score":
    df_map["box_size"] = df_map.get("weight_pct", df_map.get("final_score", 10)).clip(lower=2.0)
elif size_metric == "roe_size":
    df_map["box_size"] = df_map["roe"].clip(lower=2.0)
else:
    df_map["box_size"] = 10.0

# 사각형 안에 표시할 텍스트 라벨 만들기
df_map["display_label"] = df_map.apply(
    lambda r: f"<b>{r['name']}</b> ({r['ticker']})<br>20D: {r['ret_1m']:+.1f}% | 60D: {r['ret_3m']:+.1f}%<br>ROE: {r['roe']}% | PBR: {r['pbr']}",
    axis=1
)

# 3. Plotly 인터랙티브 Treemap 그리기
color_scale = "RdYlGn" if color_metric in ["ret_1m", "ret_3m"] else "Viridis"
midpoint = 0.0 if color_metric in ["ret_1m", "ret_3m"] else None

fig = px.treemap(
    df_map,
    path=[px.Constant("🌍 글로벌 마켓 (클릭 시 전체 보기)"), "market", "role", "display_label"],
    values="box_size",
    color=color_metric,
    color_continuous_scale=color_scale,
    color_continuous_midpoint=midpoint,
    hover_data={"ticker": True, "ret_1m": True, "ret_3m": True, "roe": True, "pbr": True, "box_size": False}
)

fig.update_traces(
    textinfo="label",
    textfont=dict(size=14),
    marker=dict(cornerradius=5)
)
fig.update_layout(
    margin=dict(t=30, l=10, r=10, b=10),
    height=620
)

st.plotly_chart(fig, use_container_width=True)

st.divider()

# 4. 히트맵 하단: 원하는 종목 선택해서 바로 차트 딥다이브로 점프하기!
st.subheader("🔎 히트맵에서 눈에 띄는 종목이 있나요? 바로 캔들차트로 이동해 보세요!")
col_jump1, col_jump2 = st.columns([3, 1])
with col_jump1:
    pick_stock = st.selectbox(
        "차트를 확인하고 싶은 종목을 선택하세요",
        options=df_map.apply(lambda r: f"[{r['market']} | {r['role']}] {r['name']} ({r['ticker']}) — 20일 수익률: {r['ret_1m']:+.2f}%", axis=1).tolist()
    )
with col_jump2:
    st.write("")
    st.write("")
    if st.button("📈 이 종목 캔들차트 열기 👉", type="primary", use_container_width=True):
        tkr = pick_stock.split("(")[-1].split(")")[0].strip()
        nm = pick_stock.split("]")[1].split("(")[0].strip()
        st.session_state["selected_ticker"] = tkr
        st.session_state["selected_name"] = nm
        st.switch_page("pages/1_🔎_종목_딥다이브.py")
