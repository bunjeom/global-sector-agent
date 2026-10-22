import datetime
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import FinanceDataReader as fdr
import yfinance as yf
from utils import load_report, init_session
import global_pipeline

st.set_page_config(page_title="3차원 종목 딥다이브", page_icon="🔎", layout="wide")
init_session()
report, df_all = load_report()

# 가독성 향상을 위한 커스텀 CSS (지표 글자 잘림 방지 및 카드 여백 정돈)
st.markdown("""
<style>
div[data-testid="stMetricValue"] {
    font-size: 1.55rem !important;
    font-weight: 700 !important;
}
div[data-testid="stMetricLabel"] {
    font-size: 0.9rem !important;
}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# 1. 상단 컨트롤 바 & 종목 선택
# ------------------------------------------------------------------------------
st.title("🔎 3차원 종목 딥다이브 리포트")
st.caption("📈 기술적 분석(차트 강의 5단계 원리) · 💎 가치 분석(EPS×PER 기대감 합당성) · 🌐 거시 민감도 통합 판독기")

with st.container(border=True):
    col1, col2, col3 = st.columns([2.2, 1.8, 1])
    with col1:
        custom_tkr = st.text_input("종목 티커 입력 (예: 005930, 000660, NVDA, LITE, SNPS)", value=st.session_state["selected_ticker"]).strip().upper()
    with col2:
        period_days = st.selectbox("차트 및 분석 조회 기간", [120, 180, 250, 365], index=1, format_func=lambda x: f"최근 {x}일 (달력 기준)")
    with col3:
        st.write("")
        st.write("")
        if st.button("🏠 홈으로 이동", use_container_width=True):
            st.switch_page("app.py")

is_kr = custom_tkr.isdigit() and len(custom_tkr) == 6
market_type = "KOSPI" if is_kr else "NASDAQ"
bench_sym = "KS11" if is_kr else "IXIC"

# 한국 주식은 소수점 없이 정수 포맷, 미국 주식은 소수점 2자리 포맷 헬퍼 함수
def fmt_p(val: float) -> str:
    return f"{val:,.0f}원" if is_kr else f"${val:,.2f}"

# 유니버스 기본 정보 매칭
uni_map = {}
for t, n, _, s, r, p, d in global_pipeline.KOSPI_UNIVERSE:
    uni_map[t] = {"name": n, "sector": s, "roe": r, "pbr": p, "debt": d, "market": "KOSPI"}
for t, n, s, r, p, d in global_pipeline.NASDAQ_UNIVERSE:
    uni_map[t] = {"name": n, "sector": s, "roe": r, "pbr": p, "debt": d, "market": "NASDAQ"}

stock_meta = uni_map.get(custom_tkr, {"name": custom_tkr, "sector": "개별종목", "roe": 12.0, "pbr": 1.8, "debt": 80.0, "market": market_type})

# ------------------------------------------------------------------------------
# 2. 데이터 수집 및 기술적/거시적 지표 계산 함수
# ------------------------------------------------------------------------------
@st.cache_data(ttl=1800, show_spinner="📊 주가·지수·거시 데이터 및 재무 지표를 정밀 연산 중입니다...")
def load_deep_dive_data(tkr: str, mkt: str, days: int):
    fetch_start = (datetime.date.today() - datetime.timedelta(days=max(days + 180, 320))).strftime("%Y-%m-%d")
    df_p = fdr.DataReader(tkr, fetch_start)
    if df_p.empty or len(df_p) < 30:
        return None, None, None

    b_sym = "KS11" if mkt == "KOSPI" else "IXIC"
    df_b = fdr.DataReader(b_sym, fetch_start)
    df_fx = fdr.DataReader("USD/KRW", fetch_start)

    df_p["MA5"] = df_p["Close"].rolling(5).mean()
    df_p["MA20"] = df_p["Close"].rolling(20).mean()
    df_p["MA60"] = df_p["Close"].rolling(60).mean()
    df_p["MA120"] = df_p["Close"].rolling(120).mean()

    std20 = df_p["Close"].rolling(20).std()
    df_p["BB_Upper"] = df_p["MA20"] + 2 * std20
    df_p["BB_Lower"] = df_p["MA20"] - 2 * std20
    df_p["BB_Width"] = (df_p["BB_Upper"] - df_p["BB_Lower"]) / (df_p["MA20"] + 1e-9) * 100.0

    delta = df_p["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df_p["RSI"] = 100.0 - (100.0 / (1.0 + rs))

    ema12 = df_p["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df_p["Close"].ewm(span=26, adjust=False).mean()
    df_p["MACD"] = ema12 - ema26
    df_p["MACD_Sig"] = df_p["MACD"].ewm(span=9, adjust=False).mean()
    df_p["MACD_Hist"] = df_p["MACD"] - df_p["MACD_Sig"]

    high, low, close = df_p["High"], df_p["Low"], df_p["Close"]
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    atr14 = tr.rolling(14).mean()
    plus_di = 100.0 * (pd.Series(plus_dm, index=df_p.index).rolling(14).mean() / (atr14 + 1e-9))
    minus_di = 100.0 * (pd.Series(minus_dm, index=df_p.index).rolling(14).mean() / (atr14 + 1e-9))
    dx = 100.0 * ((plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9))
    df_p["ADX"] = dx.rolling(14).mean()
    df_p["Plus_DI"] = plus_di
    df_p["Minus_DI"] = minus_di

    if not df_b.empty:
        df_b["MA20"] = df_b["Close"].rolling(20).mean()
        df_b["MA60"] = df_b["Close"].rolling(60).mean()

    return df_p, df_b, df_fx

df_full, df_bench, df_fx = load_deep_dive_data(custom_tkr, market_type, int(period_days))
if df_full is None:
    st.error("해당 티커의 시세 데이터를 불러올 수 없습니다. 티커명을 다시 확인해 주세요.")
    st.stop()

df_p = df_full.iloc[-int(period_days):].copy()
close_now = float(df_p["Close"].iloc[-1])

# ------------------------------------------------------------------------------
# 3. [차트 강의 알고리즘] 판독 로직
# ------------------------------------------------------------------------------
mkt_regime = "🟡 횡보/전환장 (돌파 대기)"
if df_bench is not None and len(df_bench) >= 60:
    b_close = float(df_bench["Close"].iloc[-1])
    b_ma20 = float(df_bench["MA20"].iloc[-1])
    b_ma60 = float(df_bench["MA60"].iloc[-1])
    b_ma20_slope = b_ma20 - float(df_bench["MA20"].iloc[-5])
    if b_close > b_ma20 and b_ma20 > b_ma60 and b_ma20_slope > 0:
        mkt_regime = "🟢 상승장 (상승주 적극 탐색)"
    elif b_close < b_ma20 and b_ma20 < b_ma60 and b_ma20_slope < 0:
        mkt_regime = "🔴 하락장 (리스크 축소·방어 우선)"

stock_ret_20d = (close_now / float(df_p["Close"].iloc[-21]) - 1.0) * 100.0 if len(df_p) >= 21 else 0.0
bench_ret_20d = (float(df_bench["Close"].iloc[-1]) / float(df_bench["Close"].iloc[-21]) - 1.0) * 100.0 if (df_bench is not None and len(df_bench) >= 21) else 0.0
sec_table = pd.DataFrame(report.get("kospi_sectors_table" if market_type == "KOSPI" else "nasdaq_sectors_table", []))
sec_ret_20d = bench_ret_20d
if not sec_table.empty and stock_meta["sector"] in sec_table["sector"].values:
    sec_ret_20d = float(sec_table[sec_table["sector"] == stock_meta["sector"]]["ret20_mean"].iloc[0])

rs_sec_vs_mkt = sec_ret_20d - bench_ret_20d
rs_stk_vs_sec = stock_ret_20d - sec_ret_20d

recent_60 = df_p.iloc[-60:] if len(df_p) >= 60 else df_p
box_high = float(recent_60["High"].iloc[:-5].max()) if len(recent_60) > 10 else float(recent_60["High"].max())
box_low = float(recent_60["Low"].min())
box_height = max(box_high - box_low, close_now * 0.05)

target_025 = box_high + 0.25 * box_height
target_050 = box_high + 0.50 * box_height
target_100 = box_high + 1.00 * box_height

recent_20 = df_p.iloc[-20:]
up_days_vol = recent_20[recent_20["Close"] >= recent_20["Open"]]["Volume"].mean()
down_days_vol = recent_20[recent_20["Close"] < recent_20["Open"]]["Volume"].mean()
vol_power_ratio = (up_days_vol / (down_days_vol + 1e-9)) if not np.isnan(up_days_vol) else 1.0

cond1_vol = vol_power_ratio >= 1.15 or (float(df_p["Volume"].iloc[-5:].mean()) > float(df_p["Volume"].iloc[-25:-5].mean()) * 1.2)
cond2_hold = close_now >= float(df_p["MA20"].iloc[-1])
cond3_ma_up = float(df_p["MA20"].iloc[-1]) > float(df_p["MA20"].iloc[-5])
cond4_hl = float(df_p["Low"].iloc[-10:].min()) >= float(df_p["Low"].iloc[-30:-10].min())
breakout_score = sum([cond1_vol, cond2_hold, cond3_ma_up, cond4_hl])

ma5_v, ma20_v, ma60_v = float(df_p["MA5"].iloc[-1]), float(df_p["MA20"].iloc[-1]), float(df_p["MA60"].iloc[-1])
ma120_v = float(df_p["MA120"].iloc[-1]) if not np.isnan(df_p["MA120"].iloc[-1]) else ma60_v
ma120_up = ma120_v >= (float(df_p["MA120"].iloc[-10]) if len(df_p) >= 10 and not np.isnan(df_p["MA120"].iloc[-10]) else ma120_v)
is_aligned = (ma5_v > ma20_v > ma60_v)

lows_30 = df_p["MA5"].iloc[-30:].values
first_half_min = np.nanmin(lows_30[:15])
second_half_min = np.nanmin(lows_30[15:])
double_bottom_signal = (second_half_min >= first_half_min * 0.99) and (ma5_v > second_half_min * 1.01)

rsi_now = float(df_p["RSI"].iloc[-1])
adx_now = float(df_p["ADX"].iloc[-1]) if not np.isnan(df_p["ADX"].iloc[-1]) else 20.0
plus_di_now = float(df_p["Plus_DI"].iloc[-1]) if not np.isnan(df_p["Plus_DI"].iloc[-1]) else 20.0
minus_di_now = float(df_p["Minus_DI"].iloc[-1]) if not np.isnan(df_p["Minus_DI"].iloc[-1]) else 20.0
macd_hist_now = float(df_p["MACD_Hist"].iloc[-1])
macd_hist_prev = float(df_p["MACD_Hist"].iloc[-3])

if is_aligned and breakout_score >= 3:
    lifecycle = "🚀 추세 유지·강화 (정배열 + 거래량 수반)"
elif close_now > ma20_v and double_bottom_signal:
    lifecycle = "🌱 추세 시작 시도 (쌍바닥/이평선 상향 돌파)"
elif close_now >= ma60_v and (close_now < ma20_v or macd_hist_now < macd_hist_prev):
    lifecycle = "⏸️ 추세 약화·정상 조정 (60일선 지지 테스트 구간)"
else:
    lifecycle = "⚠️ 추세 붕괴·하락 전환 주의 (지지 이탈 후 회복 실패)"

if "상승" in lifecycle or is_aligned:
    rsi_comment = f"RSI {rsi_now:.1f} — 상승 추세에서는 RSI 70 이상도 강한 매수세 유지(볼린저 상단 타고 상승 가능)로 해석합니다."
elif "조정" in lifecycle or "횡보" in mkt_regime:
    rsi_comment = f"RSI {rsi_now:.1f} — 횡보/박스권에서는 RSI 70 부근 저항 반락 주의, 볼린저 수축(폭 {df_p['BB_Width'].iloc[-1]:.1f}%) 후 확장 방향을 확인하세요."
else:
    ev1 = cond4_hl
    ev2 = close_now > float(df_p["High"].iloc[-10:-1].max())
    ev3 = (macd_hist_now < 0 and macd_hist_now > macd_hist_prev) or macd_hist_now > 0
    ev4 = vol_power_ratio >= 1.1
    ev_cnt = sum([ev1, ev2, ev3, ev4])
    rsi_comment = f"RSI {rsi_now:.1f} — 하락 추세에서는 RSI 30 아래서도 추가 하락 가능합니다. [하락 종료 4대 증거 충족: {ev_cnt}/4개]"

# ------------------------------------------------------------------------------
# 4. [가치 분석 알고리즘] 주가(P) = EPS(이익) × PER(기대감) 분해
# ------------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def fetch_valuation_metrics(tkr: str, is_kr_mkt: bool, close_p: float, meta: dict):
    roe = float(meta["roe"])
    pbr = float(meta["pbr"])
    est_per = round(pbr / max(roe / 100.0, 0.01), 2)
    est_eps = round(close_p / max(est_per, 1.0), 2)
    fwd_eps_growth = round(max(roe * 0.85, 5.0), 1)

    yf_sym = f"{tkr}.KS" if is_kr_mkt else tkr
    try:
        info = yf.Ticker(yf_sym).info
        if info.get("trailingEps") and info.get("trailingPE"):
            est_eps = float(info["trailingEps"])
            est_per = float(info["trailingPE"])
        if info.get("forwardEps") and est_eps > 0:
            fwd_g = ((float(info["forwardEps"]) / est_eps) - 1.0) * 100.0
            if -50 < fwd_g < 300:
                fwd_eps_growth = round(fwd_g, 1)
    except Exception:
        pass

    peg = round(est_per / max(fwd_eps_growth, 1.0), 2)
    return {
        "eps": est_eps,
        "per": round(est_per, 2),
        "eps_growth": fwd_eps_growth,
        "peg": peg,
        "roe": roe,
        "pbr": pbr
    }

val_data = fetch_valuation_metrics(custom_tkr, is_kr, close_now, stock_meta)

sec_peers = [v for k, v in uni_map.items() if v["market"] == market_type and v["sector"] == stock_meta["sector"]]
sec_avg_per = round(np.mean([p["pbr"] / max(p["roe"] / 100.0, 0.02) for p in sec_peers]), 2) if sec_peers else 15.0
per_premium_pct = round((val_data["per"] / max(sec_avg_per, 1.0) - 1.0) * 100.0, 1)

if val_data["eps_growth"] >= 12.0 and val_data["peg"] <= 1.2:
    val_verdict = "🟢 [실적 주도 합당한 기대] 높은 이익 성장(EPS↑)이 멀티플(PER)을 정당화하는 구간 (돌파 신뢰도 높음)"
elif val_data["eps_growth"] >= 8.0 and val_data["per"] < sec_avg_per:
    val_verdict = "🔵 [저평가 턴어라운드] 이익 체력 대비 시장 기대감(PER)이 섹터 평균보다 할인된 안전마진 구간"
elif val_data["peg"] > 1.8 or per_premium_pct > 40.0:
    val_verdict = "🟡 [기대감 선반영 구간] 현재 이익 대비 높은 멀티플 부여 — 추세선/20일선 이탈 시 리스크 관리 필수"
else:
    val_verdict = "⚪ [적정 밸류에이션 균형] 업종 평균 수준의 이익 반영도와 기대감을 유지 중"

# ------------------------------------------------------------------------------
# 5. [거시 분석 알고리즘] 상관계수 및 Beta 계산
# ------------------------------------------------------------------------------
ret_df = pd.DataFrame({"Stock": df_p["Close"].pct_change()})
if df_bench is not None and not df_bench.empty:
    ret_df["Benchmark"] = df_bench["Close"].pct_change()
if df_fx is not None and not df_fx.empty:
    ret_df["USD_KRW"] = df_fx["Close"].pct_change()
ret_df = ret_df.dropna()

corr_bench = float(ret_df["Stock"].corr(ret_df["Benchmark"])) if "Benchmark" in ret_df else 0.5
corr_fx = float(ret_df["Stock"].corr(ret_df["USD_KRW"])) if "USD_KRW" in ret_df else 0.0
cov_bm = float(ret_df["Stock"].cov(ret_df["Benchmark"])) if "Benchmark" in ret_df else 0.0
var_bm = float(ret_df["Benchmark"].var()) if "Benchmark" in ret_df else 1.0
beta_val = round(cov_bm / (var_bm + 1e-9), 2)

# ------------------------------------------------------------------------------
# 6. 상단 3차원 종합 밸런스 스코어카드 & 레이더 차트 출력 (가독성 개선 레이아웃)
# ------------------------------------------------------------------------------
tech_score = min(max(int(50 + rs_stk_vs_sec * 1.5 + breakout_score * 8 + (10 if is_aligned else -5)), 20), 98)
fund_score = min(max(int(85 - val_data["peg"] * 18 + val_data["roe"] * 0.8), 25), 98)
macro_score = min(max(int(65 + rs_sec_vs_mkt * 2.0 + (10 if "상승" in mkt_regime else 0)), 30), 95)

st.markdown(f"### 📌 {stock_meta['name']} (`{custom_tkr}`) — {stock_meta['market']} · {stock_meta['sector']}")

# 4대 핵심 지표 카드 (잘림 방지 및 정확한 등락 색상 적용)
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("💰 현재 주가 (20일 등락)", fmt_p(close_now), f"{stock_ret_20d:+.2f}%")
with m2:
    st.metric("📈 기술적 추세 점수", f"{tech_score}점 / 100", f"돌파조건 {breakout_score}/4 충족", delta_color="off")
with m3:
    st.metric("💎 가치 합당성 점수", f"{fund_score}점 / 100", f"PER {val_data['per']}배 · PEG {val_data['peg']}", delta_color="off")
with m4:
    st.metric("🌐 상대강도·거시 점수", f"{macro_score}점 / 100", f"{rs_stk_vs_sec:+.1f}%p (업종대비)")

# 레이더 차트 + 깔끔한 3줄 브리핑 카드 나란히 배치
top_c1, top_c2 = st.columns([1.15, 2.85])
with top_c1:
    with st.container(border=True):
        fig_radar = go.Figure(go.Scatterpolar(
            r=[tech_score, fund_score, macro_score, tech_score],
            theta=["기술적 추세", "가치 합당성", "거시·상대강도", "기술적 추세"],
            fill="toself",
            line=dict(color="#FF4B4B", width=2.5),
            fillcolor="rgba(255, 75, 75, 0.2)"
        ))
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(size=10)),
                angularaxis=dict(tickfont=dict(size=12, color="#333"))
            ),
            showlegend=False, height=240, margin=dict(t=35, b=25, l=55, r=55),
            title=dict(text="🕸️ 3차원 밸런스 요약", font=dict(size=14), x=0.5, xanchor="center")
        )
        st.plotly_chart(fig_radar, use_container_width=True)

with top_c2:
    with st.container(border=True):
        st.markdown("#### 🧭 핵심 투자 진단 브리핑")
        st.markdown(
            f"**1️⃣ [차트 구조 판독]** 현재 시장(`{bench_sym}`)은 **{mkt_regime}** 상태이며, 이 종목은 **{lifecycle}** 국면에 위치해 있습니다.\n\n"
            f"**2️⃣ [EPS×PER 가치 판독]** {val_verdict}\n\n"
            f"**3️⃣ [박스권 목표가 눈금]** 지지 박스 하단 `{fmt_p(box_low)}` | 돌파 기준선(0) `{fmt_p(box_high)}` "
            f"➔ **0.25 확인선 `{fmt_p(target_025)}`** ➔ **0.5 중립선 `{fmt_p(target_050)}`** ➔ **1:1 최종 목표가 `{fmt_p(target_100)}`**"
        )

# ------------------------------------------------------------------------------
# 7. 3대 영역 상세 분석 탭 (Technical / Fundamental / Macro)
# ------------------------------------------------------------------------------
tab_tech, tab_val, tab_macro = st.tabs([
    "📈 1. 기술적 분석 (강의 5단계 차트 판독)",
    "💎 2. 가치적 분석 (EPS × PER 기대감 정당성)",
    "🌐 3. 거시적 분석 (상관계수 · 베타 · 공시/재료)"
])

with tab_tech:
    st.subheader("📐 Step-by-Step 차트 강의 원리 판독표")
    chk1, chk2, chk3, chk4 = st.columns(4)
    with chk1:
        with st.container(border=True):
            st.markdown(
                f"**① 시장 & 상대강도 (1·2주차)**\n"
                f"- 시장: {mkt_regime}\n"
                f"- 업종 vs 지수: **{rs_sec_vs_mkt:+.2f}%p**\n"
                f"- 종목 vs 업종: **{rs_stk_vs_sec:+.2f}%p**"
            )
    with chk2:
        with st.container(border=True):
            st.markdown(
                f"**② 돌파 4조건 체크 (4주차)**\n"
                f"- 거래량 우위(`{vol_power_ratio:.2f}배`): {'✅ 통과' if cond1_vol else '❌ 미달'}\n"
                f"- 20일선 위 안착: {'✅ 통과' if cond2_hold else '❌ 이탈'}\n"
                f"- 20일선 상승 전환: {'✅ 통과' if cond3_ma_up else '❌ 하락'}\n"
                f"- 저점 상승(HL): {'✅ 통과' if cond4_hl else '❌ 이탈'}"
            )
    with chk3:
        with st.container(border=True):
            st.markdown(
                f"**③ 배열 & 선행신호 (MC ①)**\n"
                f"- 이평선 배열: **{'✅ 정배열' if is_aligned else '⏳ 비정배열'}**\n"
                f"- 120일선 필터: **{'🟢 우상향' if ma120_up else '🔻 하향/횡보'}**\n"
                f"- 5이평 쌍바닥: **{'🟢 감지됨' if double_bottom_signal else '⚪ 대기'}**"
            )
    with chk4:
        with st.container(border=True):
            st.markdown(
                f"**④ 박스 복사 목표가 (MC ④)**\n"
                f"- 지지 박스(하단): **{fmt_p(box_low)}**\n"
                f"- 돌파 기준선(0): **{fmt_p(box_high)}**\n"
                f"- 0.25 진짜돌파선: **{fmt_p(target_025)}**\n"
                f"- 1.0 목표가(1:1): **{fmt_p(target_100)}**"
            )

    st.info(f"💡 **[3주차 시장 상태별 지표 해석]** {rsi_comment}  \n📊 **DMI/ADX 추세 강도:** ADX `{adx_now:.1f}` (+DI `{plus_di_now:.1f}` / -DI `{minus_di_now:.1f}`) — *ADX는 추세의 강도만 알려주며, 꺾이는 것은 추세 반전이 아니라 속도 둔화입니다.*")

    fig_t = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04,
        row_heights=[0.58, 0.20, 0.22],
        subplot_titles=(
            f"[{stock_meta['name']}] 가격 구조 · 이평선(5/20/60/120) · 매물대 박스 & 1:1 목표가 눈금",
            "거래량 비교 (빨강=양봉 매수 힘 / 파랑=음봉 매도 힘)",
            "RSI(14) 상대강도지수"
        )
    )

    fig_t.add_trace(go.Candlestick(
        x=df_p.index, open=df_p["Open"], high=df_p["High"], low=df_p["Low"], close=df_p["Close"], name="캔들"
    ), row=1, col=1)
    fig_t.add_trace(go.Scatter(x=df_p.index, y=df_p["MA5"], mode="lines", name="5일선", line=dict(color="#00C853", width=1.2)), row=1, col=1)
    fig_t.add_trace(go.Scatter(x=df_p.index, y=df_p["MA20"], mode="lines", name="20일선(중심)", line=dict(color="#FF9800", width=1.8)), row=1, col=1)
    fig_t.add_trace(go.Scatter(x=df_p.index, y=df_p["MA60"], mode="lines", name="60일선(수급)", line=dict(color="#2979FF", width=1.8)), row=1, col=1)
    fig_t.add_trace(go.Scatter(x=df_p.index, y=df_p["MA120"], mode="lines", name="120일선(상위필터)", line=dict(color="#9C27B0", width=2.0, dash="dot")), row=1, col=1)
    fig_t.add_trace(go.Scatter(x=df_p.index, y=df_p["BB_Upper"], mode="lines", name="볼린저상단", line=dict(color="gray", width=0.8, dash="dash")), row=1, col=1)
    fig_t.add_trace(go.Scatter(x=df_p.index, y=df_p["BB_Lower"], mode="lines", name="볼린저하단", line=dict(color="gray", width=0.8, dash="dash")), row=1, col=1)

    fig_t.add_hrect(y0=box_high * 0.985, y1=box_high, fillcolor="red", opacity=0.12, line_width=0, annotation_text="저항/돌파 박스(0)", row=1, col=1)
    fig_t.add_hrect(y0=box_low, y1=box_low * 1.015, fillcolor="green", opacity=0.12, line_width=0, annotation_text="지지 박스", row=1, col=1)
    fig_t.add_hline(y=target_025, line_dash="dot", line_color="orange", annotation_text=f"0.25 진짜돌파선 ({fmt_p(target_025)})", row=1, col=1)
    fig_t.add_hline(y=target_100, line_dash="dash", line_color="crimson", annotation_text=f"1:1 목표가 ({fmt_p(target_100)})", row=1, col=1)

    vol_colors = ["#EF5350" if c >= o else "#42A5F5" for c, o in zip(df_p["Close"], df_p["Open"])]
    fig_t.add_trace(go.Bar(x=df_p.index, y=df_p["Volume"], marker_color=vol_colors, name="거래량"), row=2, col=1)

    fig_t.add_trace(go.Scatter(x=df_p.index, y=df_p["RSI"], mode="lines", name="RSI(14)", line=dict(color="#AB47BC", width=1.8)), row=3, col=1)
    fig_t.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
    fig_t.add_hline(y=30, line_dash="dash", line_color="blue", row=3, col=1)

    fig_t.update_layout(height=780, xaxis_rangeslider_visible=False, margin=dict(t=40, b=20, l=20, r=20))
    st.plotly_chart(fig_t, use_container_width=True)

with tab_val:
    st.subheader("💎 주가(Price) = 이익(EPS) × 기대감(PER) 완전 분해")
    st.markdown(
        "주가가 움직일 때 **① 실제로 기업이 돈을 잘 벌어서(EPS 상승)**인지, **② 시장 참여자들의 기대감(PER 멀티플 확장)** 때문인지 분해하고, "
        "**그 기대감이 이익 성장 속도(PEG)에 비추어 합당한지** 검증합니다."
    )

    vc1, vc2, vc3, vc4, vc5 = st.columns(5)
    vc1.metric("① 주당순이익 (EPS)", fmt_p(val_data['eps']), f"예상 성장 +{val_data['eps_growth']}%", delta_color="off")
    vc2.metric("② 시장 기대치 (PER)", f"{val_data['per']}배", f"업종대비 {per_premium_pct:+.1f}%", delta_color="inverse")
    vc3.metric("③ 기대 합당성 (PEG)", f"{val_data['peg']}배", "1.0 이하 저평가", delta_color="off")
    vc4.metric("④ 자본수익성 (ROE)", f"{val_data['roe']}%", "이익 창출력", delta_color="off")
    vc5.metric("⑤ 순자산배수 (PBR)", f"{val_data['pbr']}배", f"부채비율 {stock_meta['debt']}%", delta_color="off")

    st.success(f"**⚖️ 밸류에이션 정당성 최종 판정:** {val_verdict}")

    col_v1, col_v2 = st.columns(2)
    with col_v1:
        peer_rows = []
        for k, v in uni_map.items():
            if v["market"] == market_type:
                p_per = round(v["pbr"] / max(v["roe"] / 100.0, 0.02), 1)
                peer_rows.append({
                    "종목명": v["name"], "섹터": v["sector"],
                    "이익체력(ROE_%)": v["roe"], "기대감(PER_배)": min(p_per, 80.0),
                    "구분": "🎯 현재 분석 종목" if k == custom_tkr else ("동일 섹터 경쟁사" if v["sector"] == stock_meta["sector"] else "타 섹터 종목")
                })
        df_peers = pd.DataFrame(peer_rows)
        fig_scatter = px.scatter(
            df_peers, x="이익체력(ROE_%)", y="기대감(PER_배)", color="구분", hover_name="종목명",
            size="이익체력(ROE_%)",
            title=f"[{market_type}] 이익 체력(X축) 대비 시장 기대감 PER(Y축) 합당성 매트릭스"
        )
        fig_scatter.add_shape(type="line", x0=0, y0=0, x1=60, y1=60, line=dict(color="gray", dash="dot"))
        st.plotly_chart(fig_scatter, use_container_width=True)
        st.caption("💡 점선(PEG≈1 균형선) **오른쪽 아래**에 위치할수록 '실적(이익 체력) 대비 기대감(PER)이 덜 반영된 합당한 저평가' 종목입니다.")

    with col_v2:
        fair_price_by_sec = round(val_data["eps"] * sec_avg_per, 2)
        fair_price_by_peg1 = round(val_data["eps"] * val_data["eps_growth"], 2)
        df_bar_val = pd.DataFrame([
            {"비교 기준": "① 현재 시장 주가", "가격 수준": close_now, "표시": fmt_p(close_now)},
            {"비교 기준": "② 업종 평균 PER 적정가", "가격 수준": fair_price_by_sec, "표시": fmt_p(fair_price_by_sec)},
            {"비교 기준": "③ PEG 1.0(이익성장) 합당가", "가격 수준": fair_price_by_peg1, "표시": fmt_p(fair_price_by_peg1)}
        ])
        fig_val_bar = px.bar(
            df_bar_val, x="비교 기준", y="가격 수준", text="표시", color="비교 기준",
            title="현재 주가 vs 이익(EPS) 기반 합당 주가 시뮬레이션"
        )
        st.plotly_chart(fig_val_bar, use_container_width=True)

with tab_macro:
    st.subheader("🌐 거시환경 민감도(Beta · Correlation) 및 재료(Catalyst) 검증")
    mc_a, mc_b, mc_c = st.columns(3)
    mc_a.metric("시장 지수 상관계수", f"{corr_bench:+.2f}", "1에 가까울수록 지수와 동행", delta_color="off")
    mc_b.metric("시장 대비 베타 (Beta)", f"{beta_val:.2f}", "1 이상이면 지수보다 변동성 큼", delta_color="off")
    mc_c.metric("원/달러 환율 상관계수", f"{corr_fx:+.2f}", "+면 강달러 수혜 / -면 원화강세 수혜", delta_color="off")

    norm_comp = pd.DataFrame(index=df_p.index)
    norm_comp[f"{stock_meta['name']} ({custom_tkr})"] = (df_p["Close"] / df_p["Close"].iloc[0] - 1.0) * 100.0
    if df_bench is not None and not df_bench.empty:
        b_slice = df_bench.reindex(df_p.index).ffill()
        norm_comp[f"벤치마크 지수 ({bench_sym})"] = (b_slice["Close"] / b_slice["Close"].iloc[0] - 1.0) * 100.0

    fig_rs = px.line(
        norm_comp,
        title=f"[2주차 상대강도 실증] 조회 기간 내 벤치마크 지수({bench_sym}) 대비 종목 누적 수익률 비교 (%)",
        labels={"value": "누적 수익률 (%)", "index": "날짜"}
    )
    st.plotly_chart(fig_rs, use_container_width=True)

    st.markdown("#### 📰 거래량 뒤의 이유 확인 (DART 공시 & 뉴스 촉매)")
    matched_row = df_all[df_all["ticker"] == custom_tkr]
    if not matched_row.empty:
        r_cat = matched_row.iloc[0]
        st.success(
            f"- **에이전트 포착 핵심 재료:** {r_cat.get('catalyst', '-')}\n"
            f"- **실제 검증 원문(Quote):** `{r_cat.get('quote', '-')}` ({r_cat.get('loop_info', '')})"
        )
    else:
        st.info("이 종목은 오늘 선별된 Top 12 리스트 외부의 개별 조회 종목입니다. 거래량 급증 캔들 발생 시 공시 및 실적 발표 일정을 함께 점검하세요.")