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

# ------------------------------------------------------------------------------
# 1. 상단 컨트롤 바 & 종목 선택
# ------------------------------------------------------------------------------
st.title("🔎 3차원 종목 딥다이브 (Technical · Fundamental · Macro)")
st.caption("차트 강의 원리(시장상태·상대강도·구조·매물대·박스목표가) + 가치 분해(EPS×PER 합당성) + 거시 민감도 통합 판독기")

col1, col2, col3 = st.columns([2, 2, 1])
with col1:
    custom_tkr = st.text_input("티커 입력 (예: 005930, 000660, NVDA, LITE, SNPS)", value=st.session_state["selected_ticker"]).strip().upper()
with col2:
    period_days = st.selectbox("분석 및 차트 조회 기간", [120, 180, 250, 365], index=1, format_func=lambda x: f"최근 {x}일 (달력 기준)")
with col3:
    st.write("")
    st.write("")
    if st.button("🏠 홈으로 돌아가기", use_container_width=True):
        st.switch_page("app.py")

is_kr = custom_tkr.isdigit() and len(custom_tkr) == 6
market_type = "KOSPI" if is_kr else "NASDAQ"
bench_sym = "KS11" if is_kr else "IXIC"

# 유니버스 기본 정보 매칭
uni_map = {}
for t, n, _, s, r, p, d in global_pipeline.KOSPI_UNIVERSE:
    uni_map[t] = {"name": n, "sector": s, "roe": r, "pbr": p, "debt": d, "market": "KOSPI"}
for t, n, s, r, p, d in global_pipeline.NASDAQ_UNIVERSE:
    uni_map[t] = {"name": n, "sector": s, "roe": r, "pbr": p, "debt": d, "market": "NASDAQ"}

stock_meta = uni_map.get(custom_tkr, {"name": custom_tkr, "sector": "기타/개별종목", "roe": 12.0, "pbr": 1.8, "debt": 80.0, "market": market_type})

# ------------------------------------------------------------------------------
# 2. 데이터 수집 및 기술적/거시적 지표 계산 함수 (캐싱 적용)
# ------------------------------------------------------------------------------
@st.cache_data(ttl=1800, show_spinner="📊 주가·지수·거시 데이터 및 재무 지표를 정밀 연산 중입니다...")
def load_deep_dive_data(tkr: str, mkt: str, days: int):
    # 120일 이평선 계산을 위해 충분한 과거 데이터 확보
    fetch_start = (datetime.date.today() - datetime.timedelta(days=max(days + 180, 320))).strftime("%Y-%m-%d")
    df_p = fdr.DataReader(tkr, fetch_start)
    if df_p.empty or len(df_p) < 30:
        return None, None, None

    # 벤치마크 지수 및 환율/금리 로드
    b_sym = "KS11" if mkt == "KOSPI" else "IXIC"
    df_b = fdr.DataReader(b_sym, fetch_start)
    df_fx = fdr.DataReader("USD/KRW", fetch_start)

    # 이평선 (5, 20, 60, 120)
    df_p["MA5"] = df_p["Close"].rolling(5).mean()
    df_p["MA20"] = df_p["Close"].rolling(20).mean()
    df_p["MA60"] = df_p["Close"].rolling(60).mean()
    df_p["MA120"] = df_p["Close"].rolling(120).mean()

    # 볼린저 밴드 (20, 2)
    std20 = df_p["Close"].rolling(20).std()
    df_p["BB_Upper"] = df_p["MA20"] + 2 * std20
    df_p["BB_Lower"] = df_p["MA20"] - 2 * std20
    df_p["BB_Width"] = (df_p["BB_Upper"] - df_p["BB_Lower"]) / (df_p["MA20"] + 1e-9) * 100.0

    # RSI (14)
    delta = df_p["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df_p["RSI"] = 100.0 - (100.0 / (1.0 + rs))

    # MACD (12, 26, 9)
    ema12 = df_p["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df_p["Close"].ewm(span=26, adjust=False).mean()
    df_p["MACD"] = ema12 - ema26
    df_p["MACD_Sig"] = df_p["MACD"].ewm(span=9, adjust=False).mean()
    df_p["MACD_Hist"] = df_p["MACD"] - df_p["MACD_Sig"]

    # DMI / ADX (14)
    high, low, close = df_p["High"], df_p["Low"], df_p["Close"]
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
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

    # 벤치마크 지수 이평선
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
# 3. [차트 강의 알고리즘] 1~4주차 & 마스터클래스 판독 로직
# ------------------------------------------------------------------------------
# (1) 1주차: 시장 상태 읽기 (지수 20/60일선 + ADX 강도)
mkt_regime = "횡보장 (추세 대기)"
if df_bench is not None and len(df_bench) >= 60:
    b_close = float(df_bench["Close"].iloc[-1])
    b_ma20 = float(df_bench["MA20"].iloc[-1])
    b_ma60 = float(df_bench["MA60"].iloc[-1])
    b_ma20_slope = b_ma20 - float(df_bench["MA20"].iloc[-5])
    if b_close > b_ma20 and b_ma20 > b_ma60 and b_ma20_slope > 0:
        mkt_regime = "🟢 상승장 (상승 종목 적극 탐색)"
    elif b_close < b_ma20 and b_ma20 < b_ma60 and b_ma20_slope < 0:
        mkt_regime = "🔴 하락장 (리스크 축소 · 방어 우선)"
    else:
        mkt_regime = "🟡 횡보/전환장 (박스권 지지·저항 및 돌파 대기)"

# (2) 2주차: 2단계 상대강도 (지수 대비 업종 RS, 업종 대비 종목 RS)
stock_ret_20d = (close_now / float(df_p["Close"].iloc[-21]) - 1.0) * 100.0 if len(df_p) >= 21 else 0.0
bench_ret_20d = (float(df_bench["Close"].iloc[-1]) / float(df_bench["Close"].iloc[-21]) - 1.0) * 100.0 if (df_bench is not None and len(df_bench) >= 21) else 0.0
sec_table = pd.DataFrame(report.get("kospi_sectors_table" if market_type == "KOSPI" else "nasdaq_sectors_table", []))
sec_ret_20d = bench_ret_20d
if not sec_table.empty and stock_meta["sector"] in sec_table["sector"].values:
    sec_ret_20d = float(sec_table[sec_table["sector"] == stock_meta["sector"]]["ret20_mean"].iloc[0])

rs_sec_vs_mkt = sec_ret_20d - bench_ret_20d
rs_stk_vs_sec = stock_ret_20d - sec_ret_20d

# (3) 4주차 & 마스터클래스: 가격 구조 / 돌파 4조건 / 쌍바닥 / 박스 1:1 목표가
recent_60 = df_p.iloc[-60:] if len(df_p) >= 60 else df_p
box_high = float(recent_60["High"].iloc[:-5].max()) if len(recent_60) > 10 else float(recent_60["High"].max())
box_low = float(recent_60["Low"].min())
box_height = max(box_high - box_low, close_now * 0.05)

# 박스 눈금 (이탈선=box_high 기준 0.25, 0.5, 1.0 목표가)
target_025 = box_high + 0.25 * box_height
target_050 = box_high + 0.50 * box_height
target_100 = box_high + 1.00 * box_height

# 양봉 거래량 vs 음봉 거래량 비교 (마스터클래스 거래량 규칙 1·2)
recent_20 = df_p.iloc[-20:]
up_days_vol = recent_20[recent_20["Close"] >= recent_20["Open"]]["Volume"].mean()
down_days_vol = recent_20[recent_20["Close"] < recent_20["Open"]]["Volume"].mean()
vol_power_ratio = (up_days_vol / (down_days_vol + 1e-9)) if not np.isnan(up_days_vol) else 1.0

# 돌파 4조건 판정
cond1_vol = vol_power_ratio >= 1.15 or (float(df_p["Volume"].iloc[-5:].mean()) > float(df_p["Volume"].iloc[-25:-5].mean()) * 1.2)
cond2_hold = close_now >= float(df_p["MA20"].iloc[-1])
cond3_ma_up = float(df_p["MA20"].iloc[-1]) > float(df_p["MA20"].iloc[-5])
cond4_hl = float(df_p["Low"].iloc[-10:].min()) >= float(df_p["Low"].iloc[-30:-10].min())
breakout_score = sum([cond1_vol, cond2_hold, cond3_ma_up, cond4_hl])

# 이평선 정배열 & 120일 상위 추세 필터
ma5_v, ma20_v, ma60_v = float(df_p["MA5"].iloc[-1]), float(df_p["MA20"].iloc[-1]), float(df_p["MA60"].iloc[-1])
ma120_v = float(df_p["MA120"].iloc[-1]) if not np.isnan(df_p["MA120"].iloc[-1]) else ma60_v
ma120_up = ma120_v >= (float(df_p["MA120"].iloc[-10]) if len(df_p) >= 10 and not np.isnan(df_p["MA120"].iloc[-10]) else ma120_v)
is_aligned = (ma5_v > ma20_v > ma60_v)

# 5이평 쌍바닥(Double Bottom) 간이 스캔 (최근 30일 내 저점 2개 중 우측 저점이 같거나 높은 구조)
lows_30 = df_p["MA5"].iloc[-30:].values
first_half_min = np.nanmin(lows_30[:15])
second_half_min = np.nanmin(lows_30[15:])
double_bottom_signal = (second_half_min >= first_half_min * 0.99) and (ma5_v > second_half_min * 1.01)

# 추세 생애주기 판정 (시작 / 유지 / 약화 / 붕괴)
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
    lifecycle = "⏸️ 추세 약화·정상 조정 (60일선 지지 테스트 구간 — 약화≠종료)"
else:
    lifecycle = "⚠️ 추세 붕괴·하락 전환 주의 (주요 지지 이탈 후 회복 실패)"

# (4) 3주차: 시장 상태별 보조지표 동적 해석
if "상승" in lifecycle or is_aligned:
    rsi_comment = f"RSI {rsi_now:.1f} — 상승 추세에서는 RSI 70 이상도 강한 매수세 유지(볼린저 상단 타고 상승 가능)로 해석합니다."
elif "조정" in lifecycle or "횡보" in mkt_regime:
    rsi_comment = f"RSI {rsi_now:.1f} — 횡보/박스권에서는 RSI 70 부근 저항 반락 주의, 볼린저 수축(폭 {df_p['BB_Width'].iloc[-1]:.1f}%) 후 확장 방향을 확인하세요."
else:
    # 하락 종료 4대 증거 체크
    ev1 = cond4_hl  # 이전 저점 방어
    ev2 = close_now > float(df_p["High"].iloc[-10:-1].max())  # 새로운 단기 고점
    ev3 = (macd_hist_now < 0 and macd_hist_now > macd_hist_prev) or macd_hist_now > 0  # MACD 음수 폭 축소
    ev4 = vol_power_ratio >= 1.1  # 거래량 붙은 반등
    ev_cnt = sum([ev1, ev2, ev3, ev4])
    rsi_comment = f"RSI {rsi_now:.1f} — 하락 추세에서는 RSI 30 아래서도 추가 하락 가능합니다. [하락 종료 4대 증거 충족: {ev_cnt}/4개]"

# ------------------------------------------------------------------------------
# 4. [가치 분석 알고리즘] 주가(P) = EPS(이익) × PER(기대감) 분해 및 합당성 검증
# ------------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def fetch_valuation_metrics(tkr: str, is_kr_mkt: bool, close_p: float, meta: dict):
    roe = float(meta["roe"])
    pbr = float(meta["pbr"])
    # 기본 추정치: PER = PBR / (ROE/100), EPS = Price / PER
    est_per = round(pbr / max(roe / 100.0, 0.01), 2)
    est_eps = round(close_p / max(est_per, 1.0), 2)
    fwd_eps_growth = round(max(roe * 0.85, 5.0), 1)  # 지속가능 성장률 기반 추정

    # yfinance에서 실제 Trailing/Forward EPS & PER 확보 시도
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

# 섹터 평균 PER 산출
sec_peers = [v for k, v in uni_map.items() if v["market"] == market_type and v["sector"] == stock_meta["sector"]]
sec_avg_per = round(np.mean([p["pbr"] / max(p["roe"] / 100.0, 0.02) for p in sec_peers]), 2) if sec_peers else 15.0
per_premium_pct = round((val_data["per"] / max(sec_avg_per, 1.0) - 1.0) * 100.0, 1)

# 기대감 합당성 판정 (4사분면)
if val_data["eps_growth"] >= 12.0 and val_data["peg"] <= 1.2:
    val_verdict = "🟢 [실적 주도 합당한 기대] 높은 이익 성장(EPS↑)이 멀티플(PER)을 정당화하는 구간 (돌파 신뢰도 높음)"
elif val_data["eps_growth"] >= 8.0 and val_data["per"] < sec_avg_per:
    val_verdict = "🔵 [저평가 턴어라운드] 이익 체력 대비 시장 기대감(PER)이 섹터 평균보다 할인된 안전마진 구간"
elif val_data["peg"] > 1.8 or per_premium_pct > 40.0:
    val_verdict = "🟡 [기대감 선반영 구간] 현재 이익 대비 높은 멀티플(기대치)이 부여됨 — 차트상 추세선/20일선 이탈 시 철저한 리스크 관리 필수"
else:
    val_verdict = "⚪ [적정 밸류에이션 균형] 업종 평균 수준의 이익 반영도와 기대감을 유지 중"

# ------------------------------------------------------------------------------
# 5. [거시 분석 알고리즘] 지수·환율 상관계수 및 Beta 계산
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
# 6. 상단 3차원 종합 밸런스 스코어카드 & 레이더 차트 출력
# ------------------------------------------------------------------------------
tech_score = min(max(int(50 + rs_stk_vs_sec * 1.5 + breakout_score * 8 + (10 if is_aligned else -5)), 20), 98)
fund_score = min(max(int(85 - val_data["peg"] * 18 + val_data["roe"] * 0.8), 25), 98)
macro_score = min(max(int(65 + rs_sec_vs_mkt * 2.0 + (10 if "상승" in mkt_regime else 0)), 30), 95)

top_c1, top_c2 = st.columns([1.2, 2.8])
with top_c1:
    fig_radar = go.Figure(go.Scatterpolar(
        r=[tech_score, fund_score, macro_score, tech_score],
        theta=["📈 기술적 구조", "💎 가치·기대 합당성", "🌐 거시·상대강도", "📈 기술적 구조"],
        fill="toself",
        line=dict(color="#FF4B4B", width=2)
    ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=False, height=250, margin=dict(t=25, b=20, l=35, r=35),
        title=f"[{stock_meta['name']}] 3차원 밸런스"
    )
    st.plotly_chart(fig_radar, use_container_width=True)

with top_c2:
    st.markdown(f"### 📌 {stock_meta['name']} (`{custom_tkr}`) — {stock_meta['market']} · {stock_meta['sector']}")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("현재 주가", f"{close_now:,.2f}", f"20D: {stock_ret_20d:+.2f}%")
    m2.metric("📈 기술적 추세 판정", f"{tech_score}점", f"돌파조건 {breakout_score}/4 충족")
    m3.metric("💎 가치 합당성 (PEG)", f"{fund_score}점 (PEG {val_data['peg']})", f"PER {val_data['per']}배")
    m4.metric("🌐 업종·종목 상대강도", f"{macro_score}점", f"업종대비 {rs_stk_vs_sec:+.1f}%p")

    st.info(
        f"**🧭 종합 진단 요약**\n"
        f"- **[1~4주차 차트 판독]** 시장 상태는 `{mkt_regime}`이며, 현재 종목은 **`{lifecycle}`** 구간입니다.\n"
        f"- **[EPS×PER 가치 판독]** {val_verdict}\n"
        f"- **[박스권 1:1 목표가 가이드]** 박스 상단 돌파선(`{box_high:,.1f}`) 기준 **0.25 확인선 `{target_025:,.1f}`** ➔ **0.5 중립선 `{target_050:,.1f}`** ➔ **1:1 최종 목표가 `{target_100:,.1f}`**"
    )

st.divider()

# ------------------------------------------------------------------------------
# 7. 3대 영역 상세 분석 탭 (Technical / Fundamental / Macro)
# ------------------------------------------------------------------------------
tab_tech, tab_val, tab_macro = st.tabs([
    "📈 1. 기술적 분석 (강의 5단계 차트 판독)",
    "💎 2. 가치적 분석 (EPS × PER 기대감 정당성)",
    "🌐 3. 거시적 분석 (상관계수 · 베타 · 공시/재료)"
])

# ==============================================================================
# [Tab 1] 기술적 분석 (차트 강의 총정리 반영)
# ==============================================================================
with tab_tech:
    st.subheader("📐 Step-by-Step 차트 강의 원리 판독표")
    chk1, chk2, chk3, chk4 = st.columns(4)
    chk1.markdown(
        f"**① 시장 & 상대강도 (1·2주차)**\n"
        f"- 시장 상태: `{mkt_regime}`\n"
        f"- 업종 vs 지수(20D): `{rs_sec_vs_mkt:+.2f}%p`\n"
        f"- 종목 vs 업종(20D): `{rs_stk_vs_sec:+.2f}%p`"
    )
    chk2.markdown(
        f"**② 돌파 4조건 체크 (4주차)**\n"
        f"- 거래량 우위(양봉/음봉 `{vol_power_ratio:.2f}배`): {'✅' if cond1_vol else '❌'}\n"
        f"- 기준선(20일선) 위 안착: {'✅' if cond2_hold else '❌'}\n"
        f"- 20일선 기울기 상승 전환: {'✅' if cond3_ma_up else '❌'}\n"
        f"- 저점 상승(Higher Low): {'✅' if cond4_hl else '❌'}"
    )
    chk3.markdown(
        f"**③ 이평선 배열 & 선행신호 (MC ①)**\n"
        f"- 정배열(5>20>60): `{'✅ 정배열' if is_aligned else '⏳ 비정배열'}`\n"
        f"- 120일 상위추세 필터: `{'🟢 우상향' if ma120_up else '🔻 하향/횡보'}`\n"
        f"- 5이평 쌍바닥 신호: `{'🟢 감지됨' if double_bottom_signal else '⚪ 대기'}`"
    )
    chk4.markdown(
        f"**④ 박스 복사 눈금 목표가 (MC ④)**\n"
        f"- 박스 하단(지지): `{box_low:,.1f}`\n"
        f"- 돌파 기준선(0): `{box_high:,.1f}`\n"
        f"- **0.25 진짜돌파선:** `{target_025:,.1f}`\n"
        f"- **1.0 목표가(1:1):** `{target_100:,.1f}`"
    )

    st.caption(f"💡 **[3주차 시장 상태별 지표 해석]** {rsi_comment} | **DMI/ADX 강도:** ADX `{adx_now:.1f}` (+DI `{plus_di_now:.1f}` / -DI `{minus_di_now:.1f}` — ADX는 추세의 강도만 나타내며 꺾이는 것은 속도 둔화입니다.)")

    # 3단 서브플롯 차트: (1) 캔들+이평+볼린저+매물대박스+목표가눈금 (2) 거래량(양봉/음봉 색상 구분) (3) RSI & MACD
    fig_t = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04,
        row_heights=[0.58, 0.20, 0.22],
        subplot_titles=(
            f"[{stock_meta['name']}] 가격 구조 · 이평선(5/20/60/120) · 매물대 박스 & 1:1 목표가 눈금",
            "거래량 비교 (빨강=양봉 매수세 / 파랑=음봉 매도세)",
            "RSI(14) 및 추세 강도"
        )
    )

    # Row 1: 캔들 + 이평선 + 볼린저밴드
    fig_t.add_trace(go.Candlestick(
        x=df_p.index, open=df_p["Open"], high=df_p["High"], low=df_p["Low"], close=df_p["Close"], name="캔들"
    ), row=1, col=1)
    fig_t.add_trace(go.Scatter(x=df_p.index, y=df_p["MA5"], mode="lines", name="5일선", line=dict(color="#00C853", width=1.2)), row=1, col=1)
    fig_t.add_trace(go.Scatter(x=df_p.index, y=df_p["MA20"], mode="lines", name="20일선(중심)", line=dict(color="#FF9800", width=1.8)), row=1, col=1)
    fig_t.add_trace(go.Scatter(x=df_p.index, y=df_p["MA60"], mode="lines", name="60일선(수급)", line=dict(color="#2979FF", width=1.8)), row=1, col=1)
    fig_t.add_trace(go.Scatter(x=df_p.index, y=df_p["MA120"], mode="lines", name="120일선(상위필터)", line=dict(color="#9C27B0", width=2.0, dash="dot")), row=1, col=1)
    fig_t.add_trace(go.Scatter(x=df_p.index, y=df_p["BB_Upper"], mode="lines", name="볼린저상단", line=dict(color="gray", width=0.8, dash="dash")), row=1, col=1)
    fig_t.add_trace(go.Scatter(x=df_p.index, y=df_p["BB_Lower"], mode="lines", name="볼린저하단", line=dict(color="gray", width=0.8, dash="dash")), row=1, col=1)

    # 마스터클래스 ②·④: 지지/저항은 선이 아니라 '박스(Zone)'로 작도 + 0.25 / 1.0 목표가 눈금선
    fig_t.add_hrect(y0=box_high * 0.985, y1=box_high, fillcolor="red", opacity=0.12, line_width=0, annotation_text="저항/돌파 박스(0)", row=1, col=1)
    fig_t.add_hrect(y0=box_low, y1=box_low * 1.015, fillcolor="green", opacity=0.12, line_width=0, annotation_text="지지 박스", row=1, col=1)
    fig_t.add_hline(y=target_025, line_dash="dot", line_color="orange", annotation_text=f"0.25 진짜돌파 확인선 ({target_025:,.1f})", row=1, col=1)
    fig_t.add_hline(y=target_100, line_dash="dash", line_color="crimson", annotation_text=f"1:1 박스복사 목표가 ({target_100:,.1f})", row=1, col=1)

    # Row 2: 거래량 (양봉/음봉 색상 분리)
    vol_colors = ["#EF5350" if c >= o else "#42A5F5" for c, o in zip(df_p["Close"], df_p["Open"])]
    fig_t.add_trace(go.Bar(x=df_p.index, y=df_p["Volume"], marker_color=vol_colors, name="거래량"), row=2, col=1)

    # Row 3: RSI (14)
    fig_t.add_trace(go.Scatter(x=df_p.index, y=df_p["RSI"], mode="lines", name="RSI(14)", line=dict(color="#AB47BC", width=1.8)), row=3, col=1)
    fig_t.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
    fig_t.add_hline(y=30, line_dash="dash", line_color="blue", row=3, col=1)

    fig_t.update_layout(height=780, xaxis_rangeslider_visible=False, margin=dict(t=40, b=20, l=20, r=20))
    st.plotly_chart(fig_t, use_container_width=True)

# ==============================================================================
# [Tab 2] 가치적 분석 (주가 = EPS × PER 분해 및 기대감 합당성 검증)
# ==============================================================================
with tab_val:
    st.subheader("💎 주가(Price) = 이익(EPS) × 기대감(PER) 완전 분해")
    st.markdown(
        "주가가 움직일 때 **① 실제로 기업이 돈을 잘 벌어서(EPS 상승)**인지, **② 시장 참여자들의 기대감(PER 멀티플 확장)** 때문인지 분해하고, "
        "**그 기대감이 이익 성장 속도(PEG)에 비추어 합당한지** 검증합니다."
    )

    vc1, vc2, vc3, vc4, vc5 = st.columns(5)
    vc1.metric("① 주당순이익 (EPS)", f"{val_data['eps']:,}", f"예상 성장률 +{val_data['eps_growth']}%")
    vc2.metric("② 시장 기대치 (PER)", f"{val_data['per']}배", f"업종평균({sec_avg_per}배) 대비 {per_premium_pct:+.1f}%")
    vc3.metric("③ 기대 합당성 (PEG)", f"{val_data['peg']}", "1.0 이하 저평가 / 1.5 이상 과열")
    vc4.metric("④ 자본수익성 (ROE)", f"{val_data['roe']}%", "자기자본 이익창출력")
    vc5.metric("⑤ 순자산배수 (PBR)", f"{val_data['pbr']}배", f"부채비율 {stock_meta['debt']}%")

    st.success(f"**⚖️ 밸류에이션 정당성 최종 판정:** {val_verdict}")

    col_v1, col_v2 = st.columns(2)
    with col_v1:
        # 섹터 내 이익성장률(ROE/EPS Growth) vs 기대감(PER) 정당성 포지셔닝 산점도
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
        # 합당성 대각선 가이드라인 (ROE=PER 균형선)
        fig_scatter.add_shape(type="line", x0=0, y0=0, x1=60, y1=60, line=dict(color="gray", dash="dot"))
        st.plotly_chart(fig_scatter, use_container_width=True)
        st.caption("💡 점선(PEG≈1 균형선) **오른쪽 아래**에 위치할수록 '실적(이익 체력) 대비 기대감(PER)이 덜 반영된 합당한 저평가' 종목입니다.")

    with col_v2:
        # 현재 주가의 이익 반영분 vs 기대감 할증/할인 분해 바 차트
        fair_price_by_sec = round(val_data["eps"] * sec_avg_per, 2)
        fair_price_by_peg1 = round(val_data["eps"] * val_data["eps_growth"], 2)
        df_bar_val = pd.DataFrame([
            {"비교 기준": "① 현재 시장 주가", "가격 수준": close_now, "설명": f"현재 PER {val_data['per']}배 적용"},
            {"비교 기준": "② 업종 평균 PER 적용 시 적정가", "가격 수준": fair_price_by_sec, "설명": f"섹터 평균 PER {sec_avg_per}배 기준"},
            {"비교 기준": "③ PEG 1.0(이익성장률) 기준 합당가", "가격 수준": fair_price_by_peg1, "설명": f"EPS 성장률 {val_data['eps_growth']}% 기준"}
        ])
        fig_val_bar = px.bar(
            df_bar_val, x="비교 기준", y="가격 수준", text="가격 수준", color="비교 기준",
            title="현재 주가 vs 이익(EPS) 기반 합당 주가 시뮬레이션"
        )
        st.plotly_chart(fig_val_bar, use_container_width=True)

# ==============================================================================
# [Tab 3] 거시적 분석 (상관계수 · 베타 · 거래량 뒤의 이유/촉매 검증)
# ==============================================================================
with tab_macro:
    st.subheader("🌐 거시환경 민감도(Beta · Correlation) 및 재료(Catalyst) 검증")
    mc_a, mc_b, mc_c = st.columns(3)
    mc_a.metric("시장 지수 상관계수", f"{corr_bench:+.2f}", "1에 가까울수록 지수와 동행")
    mc_b.metric("시장 대비 베타 (Beta)", f"{beta_val:.2f}", "1 이상이면 지수보다 변동성 큼")
    mc_c.metric("원/달러 환율 상관계수", f"{corr_fx:+.2f}", "+면 강달러 수혜(수출주) / -면 원화강세 수혜")

    # 지수 대비 상대강도 누적 추이 차트 (2주차 강의: 지수보다 먼저 전고점을 돌파하는가?)
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

    # 마스터클래스 거래량 규칙 4: 거래량 뒤의 이유(공시/뉴스 재료) 확인
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