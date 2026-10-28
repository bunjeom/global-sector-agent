import datetime
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import FinanceDataReader as fdr
import yfinance as yf
from utils import load_report, init_session, line_icon
import global_pipeline

st.set_page_config(page_title="3차원 종목 딥다이브", page_icon="⌕", layout="wide")
init_session()
report, df_all = load_report()

# 공통 디자인 토큰(CSS 변수) 및 [data-testid] 기반 가독성·반응형 스타일
st.markdown("""
<style>
:root {
    --text-primary: #0F172A;
    --text-secondary: #334155;
    --text-muted: #475569;
    --color-up: #15803D;
    --bg-up: #DCFCE7;
    --color-down: #B91C1C;
    --bg-down: #FEE2E2;
    --color-neutral: #334155;
    --bg-neutral: #F1F5F9;
    --border-soft: #E2E8F0;
    --space-xs: 0.25rem;
    --space-sm: 0.5rem;
    --space-md: 1rem;
}

/* 본문 16px 이상, 줄간격 1.65, 왼쪽 정렬 */
div[data-testid="stMarkdownContainer"] p,
div[data-testid="stMarkdownContainer"] li {
    font-size: 1rem !important;
    line-height: 1.65 !important;
    color: var(--text-primary);
    text-align: left;
    word-break: keep-all;
}

div[data-testid="stCaptionContainer"] p {
    font-size: 0.92rem !important;
    line-height: 1.6 !important;
    color: var(--text-muted) !important;
}

/* 블랙 라인아웃 섹션 헤더 타이포그래피 */
.main-header {
    font-size: 1.85rem;
    font-weight: 800;
    color: var(--text-primary);
    margin: 0 0 0.25rem 0;
    display: flex;
    align-items: center;
}
.sec-header {
    font-size: 1.3rem;
    font-weight: 700;
    color: var(--text-primary);
    margin: 0.35rem 0 0.5rem 0;
    display: flex;
    align-items: center;
}

/* 메트릭 위계: 값(크고 굵게) vs 라벨(작고 보통 굵기) 및 말줄임 해제 */
div[data-testid="stMetricLabel"],
div[data-testid="stMetricLabel"] > div,
div[data-testid="stMetricLabel"] p {
    white-space: normal !important;
    overflow: visible !important;
    text-overflow: unset !important;
    line-height: 1.35 !important;
    font-size: 0.88rem !important;
    font-weight: 400 !important;
    color: var(--text-muted) !important;
}

div[data-testid="stMetricValue"],
div[data-testid="stMetricValue"] > div {
    white-space: normal !important;
    overflow: visible !important;
    text-overflow: unset !important;
    font-size: 1.55rem !important;
    font-weight: 700 !important;
    line-height: 1.25 !important;
    color: var(--text-primary) !important;
}

div[data-testid="stMetricDelta"],
div[data-testid="stMetricDelta"] > div {
    white-space: normal !important;
    overflow: visible !important;
    text-overflow: unset !important;
    line-height: 1.35 !important;
    font-size: 0.84rem !important;
}

/* 상단/탭 요약 카드 컴포넌트 */
.kpi-box {
    padding: 0.2rem 0;
}
.kpi-label {
    font-size: 0.88rem;
    font-weight: 400;
    color: var(--text-muted);
    margin-bottom: 0.2rem;
    line-height: 1.35;
}
.kpi-value {
    font-size: 1.55rem;
    font-weight: 700;
    color: var(--text-primary);
    line-height: 1.25;
    margin-bottom: 0.35rem;
}

/* 카드 제목과 부제 분리 스타일 */
.horizon-title {
    font-size: 1.15rem;
    font-weight: 700;
    color: var(--text-primary);
    margin: 0;
    line-height: 1.3;
}

/* 확률 카드 내부 2단 지표 블록 */
.prob-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.75rem;
    margin: 0.65rem 0 0.9rem 0;
}
.prob-item {
    background: #F8FAFC;
    border: 1px solid var(--border-soft);
    border-radius: 8px;
    padding: 0.65rem 0.75rem;
}
.prob-label {
    font-size: 0.86rem;
    font-weight: 400;
    color: var(--text-secondary);
    line-height: 1.35;
    margin-bottom: 0.25rem;
    word-break: keep-all;
}
.prob-value {
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--text-primary);
    line-height: 1.2;
    margin-bottom: 0.35rem;
}

/* 의미론적 색상 배지 (상승=초록, 하락=빨강, 중립=회색) */
.semantic-badge {
    display: inline-block;
    font-size: 0.82rem;
    font-weight: 600;
    padding: 0.15rem 0.5rem;
    border-radius: 6px;
    line-height: 1.35;
    word-break: keep-all;
}
.badge-up {
    color: var(--color-up);
    background-color: var(--bg-up);
}
.badge-down {
    color: var(--color-down);
    background-color: var(--bg-down);
}
.badge-neutral {
    color: var(--color-neutral);
    background-color: var(--bg-neutral);
}

/* 초보자용 한 줄 요약 박스 */
.easy-summary {
    margin-top: 0.75rem;
    padding: 0.6rem 0.75rem;
    background-color: var(--bg-neutral);
    border-left: 3px solid var(--text-secondary);
    border-radius: 4px;
    font-size: 0.92rem;
    line-height: 1.5;
    color: var(--text-primary);
    word-break: keep-all;
}

/* 모바일 폭(약 400px) 대응 */
@media (max-width: 480px) {
    .prob-grid {
        grid-template-columns: 1fr;
    }
    div[data-testid="stMetricValue"],
    div[data-testid="stMetricValue"] > div,
    .kpi-value {
        font-size: 1.3rem !important;
    }
    div[data-testid="stHorizontalBlock"] {
        gap: var(--space-sm) !important;
    }
}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# 1. 상단 컨트롤 바 & 종목 선택 (블랙 라인 검색 아이콘 적용)
# ------------------------------------------------------------------------------
st.markdown(f"<div class='main-header'>{line_icon('search', 26)}3차원 종목 딥다이브 & 10년 조건부 확률 터미널</div>", unsafe_allow_html=True)
st.caption("기술적 구조(차트 강의 5단계) · 당시 밸류에이션(EPS×PER) · 당시 금리 기대(FedWatch·선물) 10년(2016–2026) 통합 백테스트")

uni_map = {}
name_to_tkr = {}
for t, n, _, s, r, p, d in global_pipeline.KOSPI_UNIVERSE:
    uni_map[t] = {"name": n, "sector": s, "roe": r, "pbr": p, "debt": d, "market": "KOSPI"}
    name_to_tkr[n.replace(" ", "").upper()] = t
for t, n, s, r, p, d in global_pipeline.NASDAQ_UNIVERSE:
    uni_map[t] = {"name": n, "sector": s, "roe": r, "pbr": p, "debt": d, "market": "NASDAQ"}
    name_to_tkr[n.replace(" ", "").upper()] = t

US_KR_ALIAS = {
    "엔비디아": ("NVDA", "NVIDIA"), "테슬라": ("TSLA", "Tesla"), "애플": ("AAPL", "Apple"),
    "마이크로소프트": ("MSFT", "Microsoft"), "아마존": ("AMZN", "Amazon"), "구글": ("GOOGL", "Alphabet"),
    "알파벳": ("GOOGL", "Alphabet"), "메타": ("META", "Meta"), "브로드컴": ("AVGO", "Broadcom"),
    "팔란티어": ("PLTR", "Palantir"), "아이온큐": ("IONQ", "IonQ"), "코인베이스": ("COIN", "Coinbase"),
    "마이크로스트래티지": ("MSTR", "MicroStrategy"), "스트래티지": ("MSTR", "MicroStrategy"),
    "일라이릴리": ("LLY", "Eli Lilly"), "노보노디스크": ("NVO", "Novo Nordisk"),
    "쿠팡": ("CPNG", "Coupang"), "오라클": ("ORCL", "Oracle"), "넷플릭스": ("NFLX", "Netflix"),
    "에이엠디": ("AMD", "AMD"), "인텔": ("INTC", "Intel"), "마이크론": ("MU", "Micron"),
    "퀄컴": ("QCOM", "Qualcomm"), "암홀딩스": ("ARM", "Arm Holdings"), "슈퍼마이크로": ("SMCI", "Super Micro"),
    "버티브": ("VRT", "Vertiv"), "뉴스케일파워": ("SMR", "NuScale Power"), "오클로": ("OKLO", "Oklo"),
    "로켓랩": ("RKLB", "Rocket Lab"), "인튜이티브서지컬": ("ISRG", "Intuitive Surgical"),
    "시놉시스": ("SNPS", "Synopsys"), "루멘텀": ("LITE", "Lumentum"), "크라우드스트라이크": ("CRWD", "CrowdStrike"),
    "세일즈포스": ("CRM", "Salesforce"), "서비스나우": ("NOW", "ServiceNow"), "어도비": ("ADBE", "Adobe"),
    "아이에스엠엘": ("ASML", "ASML"), "티에스엠씨": ("TSM", "TSMC"), "암젠": ("AMGN", "Amgen"),
    "코스트코": ("COST", "Costco"), "스타벅스": ("SBUX", "Starbucks"), "보잉": ("BA", "Boeing"),
    "블룸에너지": ("BE", "Bloom Energy")
}

@st.cache_data(ttl=86400, show_spinner=False)
def get_global_market_dicts():
    """무한 로딩 방지를 위해 KRX 상장 사전만 경량 캐싱하고 미국 티커는 즉시 통과"""
    krx_dict, us_dict, us_ticker_map = {}, {}, {}
    try:
        df_krx = fdr.StockListing("KRX")
        if df_krx is not None and not df_krx.empty:
            for code, name in zip(df_krx["Code"], df_krx["Name"]):
                c_name = str(name).replace(" ", "").upper()
                krx_dict[c_name] = (str(code).strip(), str(name).strip())
    except Exception:
        pass
    return krx_dict, us_dict, us_ticker_map

def resolve_search_query(query: str):
    q_raw = query.strip()
    q_clean = q_raw.replace(" ", "").upper()
    if not q_clean:
        return "005930", "삼성전자"

    krx_dict, us_dict, us_ticker_map = get_global_market_dicts()

    # 1순위: 정확한 티커 일치
    if q_clean in uni_map:
        return q_clean, uni_map[q_clean]["name"]
    if q_clean in us_ticker_map:
        return q_clean, us_ticker_map[q_clean]

    # 2순위: 정확한 종목명/별칭 1:1 완전 일치
    if q_clean in name_to_tkr:
        tkr = name_to_tkr[q_clean]
        return tkr, uni_map[tkr]["name"]
    if q_clean in krx_dict:
        return krx_dict[q_clean]
    if q_clean in us_dict:
        return us_dict[q_clean]
    for kr_alias, (us_tkr, us_name) in US_KR_ALIAS.items():
        if q_clean == kr_alias.replace(" ", ""):
            return us_tkr, f"{kr_alias} ({us_name})"

    # 3순위: 영문 1~5글자 입력(예: BE, ON, ARM, PLTR)은 부분 검색을 타지 않고 티커 그대로 인정
    if q_clean.isascii() and q_clean.isalpha() and len(q_clean) <= 5:
        return q_clean, q_clean

    # 4순위: 한글 종목명 또는 긴 회사명 부분 검색
    for kr_alias, (us_tkr, us_name) in US_KR_ALIAS.items():
        if len(q_clean) >= 2 and q_clean in kr_alias.replace(" ", ""):
            return us_tkr, f"{kr_alias} ({us_name})"
    for u_name, u_tkr in name_to_tkr.items():
        if len(q_clean) >= 2 and q_clean in u_name:
            return u_tkr, uni_map[u_tkr]["name"]
    for k_name, (k_code, orig_name) in krx_dict.items():
        if len(q_clean) >= 2 and q_clean in k_name:
            return k_code, orig_name

    return q_clean, q_raw.upper()

default_tkr = st.session_state.get("selected_ticker", "005930")
default_display = uni_map[default_tkr]["name"] if default_tkr in uni_map else default_tkr

with st.container(border=True):
    col1, col2, col3 = st.columns([2.2, 1.8, 1])
    with col1:
        user_query = st.text_input(
            "종목명 또는 티커 검색 (예: 삼성전자, SK하이닉스, 팔란티어, NVDA, BE)",
            value=default_display
        )
    with col2:
        period_days = st.selectbox("차트 작도 표시 기간 (백테스트는 10년 전체 자동 수행)", [120, 180, 250, 365], index=1, format_func=lambda x: f"최근 {x}일 차트 확대")
    with col3:
        st.write("")
        st.write("")
        if st.button("⌂ 홈으로 이동", use_container_width=True):
            st.switch_page("app.py")

    with st.expander("시계열별 목표 수익률 기준(Threshold) 직접 설정하기 (기본값: 단기 +5% / 중기 +15% / 장기 +30%)"):
        t_c1, t_c2, t_c3 = st.columns(3)
        target_ret_short = t_c1.number_input("단기(2주 / 10영업일) 목표 수익률 (%)", value=5.0, step=1.0)
        target_ret_mid = t_c2.number_input("중기(3개월 / 60영업일) 목표 수익률 (%)", value=15.0, step=2.5)
        target_ret_long = t_c3.number_input("장기(1년 / 250영업일) 목표 수익률 (%)", value=30.0, step=5.0)

custom_tkr, resolved_name = resolve_search_query(user_query)
st.session_state["selected_ticker"] = custom_tkr
st.session_state["selected_name"] = resolved_name

is_kr = custom_tkr.isdigit() and len(custom_tkr) == 6
market_type = "KOSPI" if is_kr else "NASDAQ"
bench_sym = "KS11" if is_kr else "IXIC"

def fmt_p(val: float) -> str:
    return f"{val:,.0f}원" if is_kr else f"${val:,.2f}"

stock_meta = uni_map.get(
    custom_tkr,
    {"name": resolved_name, "sector": "개별종목", "roe": 12.0, "pbr": 1.8, "debt": 80.0, "market": market_type}
)

# ------------------------------------------------------------------------------
# 2. 10년(2016~2026) 주가·거시·금리선물(FedWatch Proxy) 통합 로드 및 3차원 상태 복원
# ------------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner="2016년–2026년(10개년) 주가·밸류에이션·금리선물(FedWatch) 데이터를 통합 스캔 중입니다...")
def load_10yr_3d_dataset(tkr: str, mkt: str):
    start_10y = "2016-01-01"
    df_p = fdr.DataReader(tkr, start_10y)
    if df_p is None or df_p.empty or len(df_p) < 60:
        return None, None, None, {}

    b_sym = "KS11" if mkt == "KOSPI" else "IXIC"
    df_b = fdr.DataReader(b_sym, start_10y)
    df_fx = fdr.DataReader("USD/KRW", start_10y)

    rate_df = pd.DataFrame(index=df_p.index)
    try:
        tnx = yf.download("^TNX", start=start_10y, progress=False)
        irx = yf.download("^IRX", start=start_10y, progress=False)
        zqf = yf.download("ZQ=F", start=start_10y, progress=False)

        def extract_close(df_raw):
            if df_raw is None or df_raw.empty:
                return pd.Series(dtype=float)
            c = df_raw["Close"]
            return c.iloc[:, 0] if isinstance(c, pd.DataFrame) else c

        s_tnx = extract_close(tnx)
        s_irx = extract_close(irx)
        s_zqf = extract_close(zqf)

        rate_df["US10Y"] = s_tnx.reindex(rate_df.index).ffill().bfill()
        rate_df["US3M_Policy"] = s_irx.reindex(rate_df.index).ffill().bfill()
        if not s_zqf.empty:
            implied_ffr = 100.0 - s_zqf.reindex(rate_df.index).ffill().bfill()
            rate_df["Fed_Implied"] = implied_ffr
        else:
            rate_df["Fed_Implied"] = rate_df["US3M_Policy"]
    except Exception:
        rate_df["US10Y"] = 3.5
        rate_df["US3M_Policy"] = 3.5
        rate_df["Fed_Implied"] = 3.5

    rate_df = rate_df.ffill().bfill().fillna(3.5)

    rate_df["Cut_Expect_Spread"] = (rate_df["US10Y"] - rate_df["US3M_Policy"])
    rate_df["Rate_Mom_60d"] = rate_df["US10Y"] - rate_df["US10Y"].rolling(60).mean()
    rate_df["Fed_Future_Mom_20d"] = rate_df["Fed_Implied"].diff(20).fillna(0.0)

    df_p["MA5"] = df_p["Close"].rolling(5).mean()
    df_p["MA20"] = df_p["Close"].rolling(20).mean()
    df_p["MA60"] = df_p["Close"].rolling(60).mean()
    df_p["MA120"] = df_p["Close"].rolling(120).mean()
    df_p["MA240"] = df_p["Close"].rolling(240).mean()

    std20 = df_p["Close"].rolling(20).std()
    df_p["BB_Upper"] = df_p["MA20"] + 2 * std20
    df_p["BB_Lower"] = df_p["MA20"] - 2 * std20
    df_p["BB_Width"] = (df_p["BB_Upper"] - df_p["BB_Lower"]) / (df_p["MA20"] + 1e-9) * 100.0

    delta = df_p["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    df_p["RSI"] = 100.0 - (100.0 / (1.0 + gain / (loss + 1e-9)))

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

    df_p["Ret_20d"] = df_p["Close"].pct_change(20) * 100.0
    df_p["MA20_Gap"] = (df_p["Close"] / (df_p["MA20"] + 1e-9) - 1.0) * 100.0
    df_p["Vol_Ratio"] = df_p["Volume"].rolling(5).mean() / (df_p["Volume"].rolling(20).mean() + 1e-9)

    val_base = df_p["MA240"].fillna(df_p["MA120"]).fillna(df_p["MA60"])
    val_ratio = df_p["Close"] / (val_base + 1e-9)
    df_p["Hist_Val_Z"] = (val_ratio - val_ratio.rolling(250, min_periods=60).mean()) / (val_ratio.rolling(250, min_periods=60).std() + 1e-9)

    if not df_b.empty:
        df_b["MA20"] = df_b["Close"].rolling(20).mean()
        df_b["MA60"] = df_b["Close"].rolling(60).mean()
        b_ret20 = df_b["Close"].pct_change(20).reindex(df_p.index).ffill() * 100.0
        df_p["RS_vs_Bench"] = df_p["Ret_20d"] - b_ret20.fillna(0.0)
    else:
        df_p["RS_vs_Bench"] = 0.0

    df_p["Rate_Mom_60d"] = rate_df["Rate_Mom_60d"]
    df_p["Cut_Expect_Spread"] = rate_df["Cut_Expect_Spread"]
    df_p["US10Y"] = rate_df["US10Y"]
    df_p["US3M_Policy"] = rate_df["US3M_Policy"]
    df_p["Fed_Implied"] = rate_df["Fed_Implied"]
    df_p["Fed_Future_Mom_20d"] = rate_df["Fed_Future_Mom_20d"]

    last_r = df_p.iloc[-1]
    ffr_now = float(last_r["US3M_Policy"])
    fed_imp = float(last_r["Fed_Implied"])
    us10y_now = float(last_r["US10Y"])
    rate_mom = float(last_r["Rate_Mom_60d"])
    fed_mom = float(last_r["Fed_Future_Mom_20d"])

    # 신호등 아이콘(🟢, 🔴, 🟡)은 요청대로 유지
    if (fed_imp < ffr_now - 0.10) or (rate_mom < -0.15) or (fed_mom < -0.05):
        fed_short_tag = "금리 인하 기대 국면"
        fed_regime_str = "🟢 금리 인하 기대 국면 (선물·국채시장이 기준금리 인하 경로를 선반영 중)"
    elif (fed_imp > ffr_now + 0.15) or (rate_mom > 0.20) or (fed_mom > 0.08):
        fed_short_tag = "금리 인상·긴축 경계"
        fed_regime_str = "🔴 금리 인상·긴축 경계 국면 (시장이 고금리 장기화 또는 긴축을 반영 중)"
    else:
        fed_short_tag = "금리 동결·중립 국면"
        fed_regime_str = "🟡 금리 동결·중립 국면 (급격한 금리 변동 없이 현 수준 유지 전망)"

    macro_summary = {
        "ffr_proxy": round(ffr_now, 2),
        "fed_implied": round(fed_imp, 2),
        "us10y": round(us10y_now, 2),
        "rate_mom_60d": round(rate_mom, 2),
        "fed_regime_str": fed_regime_str,
        "fed_short_tag": fed_short_tag,
        "start_date": df_p.index[0].strftime("%Y-%m-%d"),
        "total_days": len(df_p)
    }

    return df_p, df_b, df_fx, macro_summary

df_full, df_bench, df_fx, macro_info = load_10yr_3d_dataset(custom_tkr, market_type)
if df_full is None:
    st.error(f"'{user_query}' (변환 티커: {custom_tkr})의 시세 데이터를 찾을 수 없습니다. 정확한 종목명이나 티커를 입력해 주세요.")
    st.stop()

df_p = df_full.iloc[-int(period_days):].copy()
close_now = float(df_p["Close"].iloc[-1])

# ------------------------------------------------------------------------------
# 3. [10년 3차원 조건부 확률 백테스트 엔진]
# ------------------------------------------------------------------------------
def run_3d_conditional_backtest(df_10y: pd.DataFrame, horizon: int, target_pct: float, weights: tuple):
    w_tech, w_val, w_macro = weights
    cols_needed = ["MA20_Gap", "RSI", "Vol_Ratio", "Hist_Val_Z", "RS_vs_Bench", "Rate_Mom_60d"]
    sub = df_10y[cols_needed + ["Close", "High", "Low"]].dropna().copy()

    if len(sub) <= horizon + 60:
        return {
            "touch_prob": 50.0, "close_prob": 40.0, "mode_low": 0.0, "mode_high": 5.0,
            "median_ret": 2.0, "mfe_median": 5.0, "mae_median": -3.0,
            "sample_cnt": 0, "similar_dates": [], "close_rets": [], "max_rets": []
        }

    norm_df = (sub[cols_needed] - sub[cols_needed].mean()) / (sub[cols_needed].std() + 1e-9)
    now_vec = norm_df.iloc[-1]
    valid_norm = norm_df.iloc[:-horizon]

    dist_tech = np.sqrt(
        (valid_norm["MA20_Gap"] - now_vec["MA20_Gap"]) ** 2 +
        (valid_norm["RSI"] - now_vec["RSI"]) ** 2 +
        (valid_norm["Vol_Ratio"] - now_vec["Vol_Ratio"]) ** 2
    ) / np.sqrt(3)
    dist_val = (valid_norm["Hist_Val_Z"] - now_vec["Hist_Val_Z"]).abs()
    dist_macro = np.sqrt(
        (valid_norm["RS_vs_Bench"] - now_vec["RS_vs_Bench"]) ** 2 +
        (valid_norm["Rate_Mom_60d"] - now_vec["Rate_Mom_60d"]) ** 2
    ) / np.sqrt(2)

    total_dist = w_tech * dist_tech + w_val * dist_val + w_macro * dist_macro

    sorted_indices = np.argsort(total_dist.values)
    picked_pos = []
    min_gap = 5 if horizon <= 20 else 10
    for idx_pos in sorted_indices:
        if all(abs(idx_pos - p) >= min_gap for p in picked_pos):
            picked_pos.append(int(idx_pos))
        if len(picked_pos) >= 45:
            break

    close_arr = sub["Close"].values
    high_arr = sub["High"].values
    low_arr = sub["Low"].values
    dates_arr = sub.index

    max_rets, min_rets, end_rets, sim_dates = [], [], [], []
    for p in picked_pos:
        entry_p = close_arr[p]
        future_high = np.max(high_arr[p + 1 : p + 1 + horizon])
        future_low = np.min(low_arr[p + 1 : p + 1 + horizon])
        future_close = close_arr[p + horizon]

        max_rets.append((future_high / entry_p - 1.0) * 100.0)
        min_rets.append((future_low / entry_p - 1.0) * 100.0)
        end_rets.append((future_close / entry_p - 1.0) * 100.0)
        sim_dates.append(dates_arr[p].strftime("%Y-%m-%d"))

    max_rets = np.array(max_rets)
    min_rets = np.array(min_rets)
    end_rets = np.array(end_rets)

    touch_prob = float(np.mean(max_rets >= target_pct) * 100.0)
    close_prob = float(np.mean(end_rets >= target_pct) * 100.0)

    counts, bin_edges = np.histogram(end_rets, bins=6)
    best_bin = int(np.argmax(counts))
    mode_low = float(bin_edges[best_bin])
    mode_high = float(bin_edges[best_bin + 1])

    return {
        "touch_prob": round(touch_prob, 1),
        "close_prob": round(close_prob, 1),
        "mode_low": round(mode_low, 1),
        "mode_high": round(mode_high, 1),
        "median_ret": round(float(np.median(end_rets)), 1),
        "mfe_median": round(float(np.median(max_rets)), 1),
        "mae_median": round(float(np.median(min_rets)), 1),
        "sample_cnt": len(picked_pos),
        "similar_dates": sim_dates[:4],
        "close_rets": end_rets.tolist(),
        "max_rets": max_rets.tolist()
    }

bt_short = run_3d_conditional_backtest(df_full, horizon=10, target_pct=target_ret_short, weights=(0.60, 0.15, 0.25))
bt_mid = run_3d_conditional_backtest(df_full, horizon=60, target_pct=target_ret_mid, weights=(0.40, 0.30, 0.30))
bt_long = run_3d_conditional_backtest(df_full, horizon=250, target_pct=target_ret_long, weights=(0.20, 0.50, 0.30))

# ------------------------------------------------------------------------------
# 4. [차트 강의 알고리즘] 1~4주차 & 마스터클래스 판독 로직 (신호등 아이콘만 유지)
# ------------------------------------------------------------------------------
mkt_regime = "🟡 횡보·전환장 (돌파 대기)"
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
    lifecycle = "🟢 추세 유지·강화 (정배열 + 거래량 수반)"
elif close_now > ma20_v and double_bottom_signal:
    lifecycle = "🟢 추세 시작 시도 (쌍바닥·이평선 상향 돌파)"
elif close_now >= ma60_v and (close_now < ma20_v or macd_hist_now < macd_hist_prev):
    lifecycle = "🟡 추세 약화·정상 조정 (60일선 지지 테스트 구간)"
else:
    lifecycle = "🔴 추세 붕괴·하락 전환 주의 (지지 이탈 후 회복 실패)"

if "추세 유지" in lifecycle or is_aligned:
    rsi_comment = f"RSI {rsi_now:.1f} — 상승 추세에서는 RSI 70 이상도 강한 매수세 유지(볼린저 상단 타고 상승 가능)로 해석합니다."
elif "조정" in lifecycle or "횡보" in mkt_regime:
    rsi_comment = f"RSI {rsi_now:.1f} — 횡보·박스권에서는 RSI 70 부근 저항 반락 주의, 볼린저 수축(폭 {df_p['BB_Width'].iloc[-1]:.1f}%) 후 확장 방향을 확인하세요."
else:
    ev1 = cond4_hl
    ev2 = close_now > float(df_p["High"].iloc[-10:-1].max())
    ev3 = (macd_hist_now < 0 and macd_hist_now > macd_hist_prev) or macd_hist_now > 0
    ev4 = vol_power_ratio >= 1.1
    ev_cnt = sum([ev1, ev2, ev3, ev4])
    rsi_comment = f"RSI {rsi_now:.1f} — 하락 추세에서는 RSI 30 아래서도 추가 하락 가능합니다. [하락 종료 4대 증거 충족: {ev_cnt}/4개]"

# ------------------------------------------------------------------------------
# 5. [가치 분석 알고리즘] 주가(P) = EPS(이익) × PER(기대감) 분해
# ------------------------------------------------------------------------------
YF_SECTOR_MAP = {
    "Technology": "AI·반도체·SW",
    "Communication Services": "플랫폼·미디어",
    "Consumer Cyclical": "소비재·모빌리티",
    "Consumer Defensive": "필수소비재",
    "Healthcare": "바이오·헬스케어",
    "Financial Services": "금융·핀테크",
    "Industrials": "산업재·우주방산",
    "Energy": "에너지·전력",
    "Utilities": "전력·유틸리티",
    "Real Estate": "리츠·부동산",
    "Basic Materials": "소재·화학"
}

@st.cache_data(ttl=3600)
def fetch_valuation_metrics(tkr: str, is_kr_mkt: bool, close_p: float, meta: dict):
    roe = float(meta["roe"])
    pbr = float(meta["pbr"])
    debt = float(meta["debt"])
    sector_str = str(meta["sector"])
    comp_name = str(meta["name"])

    est_per = round(pbr / max(abs(roe) / 100.0, 0.01), 2)
    est_eps = round(close_p / max(est_per, 1.0), 2)
    fwd_eps_growth = round(max(roe * 0.85, 5.0), 1)

    yf_sym = f"{tkr}.KS" if is_kr_mkt else tkr
    try:
        info = yf.Ticker(yf_sym).info
        if is_kr_mkt and (not info or info.get("regularMarketPrice") is None):
            info = yf.Ticker(f"{tkr}.KQ").info

        if info:
            if sector_str == "개별종목" and info.get("sector"):
                raw_sec = str(info.get("sector"))
                sector_str = YF_SECTOR_MAP.get(raw_sec, raw_sec)
            if comp_name == tkr and (info.get("shortName") or info.get("longName")):
                comp_name = str(info.get("shortName") or info.get("longName"))

            if info.get("returnOnEquity") is not None:
                roe = round(float(info["returnOnEquity"]) * 100.0, 1)
            if info.get("priceToBook") is not None and float(info["priceToBook"]) > 0:
                pbr = round(float(info["priceToBook"]), 2)
            if info.get("debtToEquity") is not None and float(info["debtToEquity"]) >= 0:
                debt = round(float(info["debtToEquity"]), 1)

            if info.get("trailingEps") and info.get("trailingPE"):
                est_eps = round(float(info["trailingEps"]), 2)
                est_per = round(float(info["trailingPE"]), 2)
            elif info.get("forwardPE") and float(info["forwardPE"]) > 0:
                est_per = round(float(info["forwardPE"]), 2)
                est_eps = round(close_p / est_per, 2)
            else:
                est_per = round(pbr / max(abs(roe) / 100.0, 0.02), 2)
                est_eps = round(close_p / max(est_per, 1.0), 2)

            if info.get("earningsGrowth") is not None:
                eg = float(info["earningsGrowth"]) * 100.0
                if -50 < eg < 300:
                    fwd_eps_growth = round(eg, 1)
            elif info.get("forwardEps") and est_eps > 0:
                fwd_g = ((float(info["forwardEps"]) / est_eps) - 1.0) * 100.0
                if -50 < fwd_g < 300:
                    fwd_eps_growth = round(fwd_g, 1)
            elif info.get("revenueGrowth") is not None:
                rg = float(info["revenueGrowth"]) * 100.0
                if -30 < rg < 250:
                    fwd_eps_growth = round(max(rg, 5.0), 1)
    except Exception:
        pass

    peg = round(est_per / max(fwd_eps_growth, 1.0), 2)
    return {
        "name": comp_name,
        "sector": sector_str,
        "eps": est_eps,
        "per": round(est_per, 2),
        "eps_growth": fwd_eps_growth,
        "peg": peg,
        "roe": roe,
        "pbr": pbr,
        "debt": debt
    }

val_data = fetch_valuation_metrics(custom_tkr, is_kr, close_now, stock_meta)
stock_meta["name"] = val_data["name"]
stock_meta["sector"] = val_data["sector"]
stock_meta["roe"] = val_data["roe"]
stock_meta["pbr"] = val_data["pbr"]
stock_meta["debt"] = val_data["debt"]
uni_map[custom_tkr] = stock_meta

sec_peers = [v for k, v in uni_map.items() if v["market"] == market_type and v["sector"] == stock_meta["sector"]]
if not sec_peers:
    sec_peers = [v for k, v in uni_map.items() if v["market"] == market_type]
sec_avg_per = round(np.mean([p["pbr"] / max(abs(p["roe"]) / 100.0, 0.02) for p in sec_peers]), 2) if sec_peers else 20.0
per_premium_pct = round((val_data["per"] / max(sec_avg_per, 1.0) - 1.0) * 100.0, 1)

if val_data["eps_growth"] >= 12.0 and val_data["peg"] <= 1.2:
    val_verdict = "🟢 [실적 주도 합당한 기대] 높은 이익 성장(EPS 상승)이 멀티플(PER)을 정당화하는 구간"
elif val_data["eps_growth"] >= 8.0 and val_data["per"] < sec_avg_per:
    val_verdict = "🟢 [저평가 턴어라운드] 이익 체력 대비 시장 기대감(PER)이 업종 평균보다 할인된 안전마진 구간"
elif val_data["peg"] > 1.8 or per_premium_pct > 40.0:
    val_verdict = "🟡 [기대감 선반영 구간] 현재 이익 대비 높은 멀티플 부여 — 20일선 이탈 시 리스크 관리 필수"
else:
    val_verdict = "⚪ [적정 밸류에이션 균형] 업종 평균 수준의 이익 반영도와 기대감을 유지 중"

# ------------------------------------------------------------------------------
# 6. 상단: 핵심 지표 요약 & 단기·중기·장기 10년 조건부 확률 패널 (블랙 라인 아이콘)
# ------------------------------------------------------------------------------
st.markdown(
    f"<div class='sec-header'>{line_icon('bookmark', 22)}{stock_meta['name']} ({custom_tkr}) — {stock_meta['market']} · {stock_meta['sector']}</div>",
    unsafe_allow_html=True
)

def render_top_kpi(col_obj, label_str: str, val_str: str, badge_html: str):
    with col_obj:
        st.markdown(
            f"<div class='kpi-box'>"
            f"  <div class='kpi-label'>{label_str}</div>"
            f"  <div class='kpi-value'>{val_str}</div>"
            f"  {badge_html}"
            f"</div>",
            unsafe_allow_html=True
        )

h1, h2, h3, h4 = st.columns(4)
ret_cls = "badge-up" if stock_ret_20d > 0 else ("badge-down" if stock_ret_20d < 0 else "badge-neutral")
rs_cls = "badge-up" if rs_stk_vs_sec > 0 else ("badge-down" if rs_stk_vs_sec < 0 else "badge-neutral")

render_top_kpi(
    h1, "현재 주가 (20일 등락률)", fmt_p(close_now),
    f"<span class='semantic-badge {ret_cls}'>{stock_ret_20d:+.2f}%</span>"
)
render_top_kpi(
    h2, "시장 금리 기대 (FedWatch·선물)", macro_info["fed_short_tag"],
    f"<span class='semantic-badge badge-neutral'>선물내재 {macro_info['fed_implied']}% · 미10년 {macro_info['us10y']}%</span>"
)
render_top_kpi(
    h3, "현재 밸류에이션 (PER · PEG)", f"PER {val_data['per']}배",
    f"<span class='semantic-badge badge-neutral'>PEG {val_data['peg']}배 (업종대비 {per_premium_pct:+.1f}%)</span>"
)
render_top_kpi(
    h4, "차트 생애주기 & 돌파 조건", f"돌파 {breakout_score}/4 충족",
    f"<span class='semantic-badge {rs_cls}'>업종대비 RS {rs_stk_vs_sec:+.1f}%p</span>"
)

st.caption(
    f"백테스트 검증 모수: {macro_info['start_date']} – 오늘 "
    f"(총 {macro_info['total_days']:,}영업일, 2019–2020 금리 인하기 및 2022–2023 금리 인상기 모두 포함) 중 "
    f"현재의 <strong>[기술적 구조 + 당시 밸류에이션 괴리도 + 당시 금리 기대({macro_info['fed_short_tag']})]</strong>와 "
    f"가장 일치했던 과거 유사 국면 표본 실증 결과입니다.",
    unsafe_allow_html=True
)

p_col1, p_col2, p_col3 = st.columns(3)

def make_beginner_summary(period_name: str, target_pct: float, bt_res: dict) -> str:
    t_prob = bt_res["touch_prob"]
    c_prob = bt_res["close_prob"]
    med = bt_res["median_ret"]

    # 신호등 아이콘(🟢, 🟡, ⚪, 🔴)은 유지
    if t_prob >= 60.0 and c_prob >= 45.0:
        return f"🟢 <strong>긍정적:</strong> {period_name} 내 목표(+{target_pct:.0f}%) 달성 및 추세 유지 확률이 모두 높습니다."
    elif t_prob >= 50.0 and c_prob < 40.0:
        return f"🟡 <strong>단타·분할익절 유리:</strong> 장중 +{target_pct:.0f}% 터치 기회는 잦으나 끝까지 보유 시 수익이 반납될 수 있습니다."
    elif med > 0:
        return f"⚪ <strong>완만한 흐름:</strong> +{target_pct:.0f}% 급등보다는 {med:+.1f}% 수준의 완만한 등락이 예상됩니다."
    else:
        return f"🔴 <strong>보수적 접근:</strong> 과거 유사 국면에서 상승 탄력이 약했습니다. 섣부른 진입보다 관망이 유리합니다."

def render_prob_card(col_obj, main_title, sub_period, target_pct, bt_res):
    mfe_cls = "badge-up" if bt_res["mfe_median"] > 0 else ("badge-down" if bt_res["mfe_median"] < 0 else "badge-neutral")
    med_cls = "badge-up" if bt_res["median_ret"] > 0 else ("badge-down" if bt_res["median_ret"] < 0 else "badge-neutral")
    mae_cls = "badge-down" if bt_res["mae_median"] < 0 else "badge-neutral"

    mfe_badge = f"<span class='semantic-badge {mfe_cls}'>최고상승 중앙값 {bt_res['mfe_median']:+.1f}%</span>"
    med_badge = f"<span class='semantic-badge {med_cls}'>종가수익 중앙값 {bt_res['median_ret']:+.1f}%</span>"
    mae_badge = f"<span class='semantic-badge {mae_cls}'>하방 {bt_res['mae_median']:+.1f}% (손실주의)</span>"

    mode_mid = (bt_res["mode_low"] + bt_res["mode_high"]) / 2.0
    mode_cls = "badge-up" if mode_mid > 0 else ("badge-down" if mode_mid < 0 else "badge-neutral")
    mode_badge = f"<span class='semantic-badge {mode_cls}'>{bt_res['mode_low']:+.1f}% – {bt_res['mode_high']:+.1f}%</span>"
    easy_comment = make_beginner_summary(sub_period, target_pct, bt_res)

    with col_obj:
        with st.container(border=True):
            st.markdown(
                f"<div class='horizon-title'>{main_title} <span style='font-size:0.9rem; font-weight:500; color:var(--text-muted);'>({sub_period})</span></div>"
                f"<div class='prob-grid'>"
                f"  <div class='prob-item'>"
                f"    <div class='prob-label'>기간 내 +{target_pct:.0f}% 터치 확률</div>"
                f"    <div class='prob-value'>{bt_res['touch_prob']:.1f}%</div>"
                f"    {mfe_badge}"
                f"  </div>"
                f"  <div class='prob-item'>"
                f"    <div class='prob-label'>만기 종가 +{target_pct:.0f}% 유지 확률</div>"
                f"    <div class='prob-value'>{bt_res['close_prob']:.1f}%</div>"
                f"    {med_badge}"
                f"  </div>"
                f"</div>"
                f"<ul style='margin:0; padding-left:1.2rem; line-height:1.75;'>"
                f"  <li><strong>최빈 수익률 구간:</strong> {mode_badge}</li>"
                f"  <li><strong>평균 하방 위험(MAE):</strong> {mae_badge}</li>"
                f"</ul>"
                f"<div class='easy-summary'>{easy_comment}</div>",
                unsafe_allow_html=True
            )

render_prob_card(p_col1, "단기 기대 확률", "2주 이내", target_ret_short, bt_short)
render_prob_card(p_col2, "중기 기대 확률", "3개월 이내", target_ret_mid, bt_mid)
render_prob_card(p_col3, "장기 기대 확률", "1년 이상", target_ret_long, bt_long)

with st.container(border=True):
    st.markdown(
        f"<div style='font-size:1.05rem; font-weight:700; margin-bottom:0.5rem; display:flex; align-items:center;'>"
        f"{line_icon('compass', 18)}3차원 통합 판독 요약</div>"
        f"<ul style='margin:0; padding-left:1.2rem; line-height:1.8;'>"
        f"  <li><strong>거시·금리 환경:</strong> {macro_info['fed_regime_str']}</li>"
        f"  <li><strong>시장 지수 상태:</strong> 벤치마크({bench_sym}) 기준 <strong>{mkt_regime}</strong> 구간입니다.</li>"
        f"  <li><strong>차트 생애주기:</strong> 현재 <strong>{lifecycle}</strong> 상태입니다.</li>"
        f"  <li><strong>주요 목표 가격:</strong> 돌파 기준선 <strong>{fmt_p(box_high)}</strong> → 0.25 확인선 <strong>{fmt_p(target_025)}</strong> → 1:1 목표가 <strong>{fmt_p(target_100)}</strong></li>"
        f"  <li><strong>가치·기대 합당성:</strong> {val_verdict}</li>"
        f"</ul>",
        unsafe_allow_html=True
    )
    )

st.divider()

# ------------------------------------------------------------------------------
# 7. 3대 영역 상세 분석 탭 + 백테스트 분포 상세 탭 (차트 탭 📈은 유지, 나머지는 블랙 라인 심볼)
# ------------------------------------------------------------------------------
tab_tech, tab_val, tab_macro, tab_bt = st.tabs([
    "📈 1. 기술적 분석 (차트 5단계 판독)",
    "◇ 2. 가치적 분석 (EPS × PER 정당성)",
    "⊕ 3. 거시·FedWatch 금리 기대",
    "📊 4. 10년 조건부 확률 분포"
])

with tab_tech:
    st.markdown(f"<div class='sec-header'>{line_icon('layers', 22)}Step-by-Step 차트 강의 원리 판독표</div>", unsafe_allow_html=True)
    chk1, chk2, chk3, chk4 = st.columns(4)
    with chk1:
        with st.container(border=True, height=215):
            st.markdown(
                f"**① 시장 & 상대강도 (1·2주차)**\n"
                f"- 시장: {mkt_regime}\n"
                f"- 업종 vs 지수: **{rs_sec_vs_mkt:+.2f}%p**\n"
                f"- 종목 vs 업종: **{rs_stk_vs_sec:+.2f}%p**"
            )
    with chk2:
        with st.container(border=True, height=215):
            st.markdown(
                f"**② 돌파 4조건 체크 (4주차)**\n"
                f"- 거래량 우위({vol_power_ratio:.2f}배): {'🟢 통과' if cond1_vol else '🔴 미달'}\n"
                f"- 20일선 안착: {'🟢 통과' if cond2_hold else '🔴 이탈'}\n"
                f"- 20일선 방향: {'🟢 상승' if cond3_ma_up else '🔴 하락'}\n"
                f"- 저점 상승(HL): {'🟢 통과' if cond4_hl else '🔴 이탈'}"
            )
    with chk3:
        with st.container(border=True, height=215):
            st.markdown(
                f"**③ 배열 & 선행신호 (MC ①)**\n"
                f"- 이평선 배열: **{'🟢 정배열' if is_aligned else '🟡 비정배열'}**\n"
                f"- 120일선 필터: **{'🟢 우상향' if ma120_up else '🔴 하향·횡보'}**\n"
                f"- 5이평 쌍바닥: **{'🟢 감지됨' if double_bottom_signal else '⚪ 대기'}**"
            )
    with chk4:
        with st.container(border=True, height=215):
            st.markdown(
                f"**④ 박스 복사 목표가 (MC ④)**\n"
                f"- 지지 하단: **{fmt_p(box_low)}**\n"
                f"- 돌파 기준(0): **{fmt_p(box_high)}**\n"
                f"- 0.25 확인선: **{fmt_p(target_025)}**\n"
                f"- 1.0 목표가: **{fmt_p(target_100)}**"
            )

    st.info(
        f"**[3주차 시장 상태별 지표 해석]** {rsi_comment}  \n"
        f"**[DMI/ADX 추세 강도]** ADX {adx_now:.1f} (+DI {plus_di_now:.1f} / -DI {minus_di_now:.1f}) — "
        f"ADX는 추세의 강도만 알려주며, 꺾이는 것은 추세 반전이 아니라 속도 둔화입니다."
    )

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
    st.markdown(f"<div class='sec-header'>{line_icon('diamond', 22)}주가(Price) = 이익(EPS) × 기대감(PER) 완전 분해</div>", unsafe_allow_html=True)
    vc1, vc2, vc3, vc4, vc5 = st.columns(5)
    render_top_kpi(vc1, "① 주당순이익 (EPS)", fmt_p(val_data["eps"]), f"<span class='semantic-badge badge-neutral'>예상 성장 {val_data['eps_growth']:+.1f}%</span>")
    render_top_kpi(vc2, "② 시장 기대치 (PER)", f"{val_data['per']}배", f"<span class='semantic-badge badge-neutral'>업종대비 {per_premium_pct:+.1f}%</span>")
    render_top_kpi(vc3, "③ 기대 합당성 (PEG)", f"{val_data['peg']}배", "<span class='semantic-badge badge-neutral'>1.0 이하 저평가</span>")
    render_top_kpi(vc4, "④ 자본수익성 (ROE)", f"{val_data['roe']}%", "<span class='semantic-badge badge-neutral'>이익 창출력</span>")
    render_top_kpi(vc5, "⑤ 순자산배수 (PBR)", f"{val_data['pbr']}배", f"<span class='semantic-badge badge-neutral'>부채비율 {stock_meta['debt']}%</span>")

    st.success(f"**밸류에이션 정당성 최종 판정:** {val_verdict}")

    col_v1, col_v2 = st.columns(2)
    with col_v1:
        peer_rows = []
        for k, v in uni_map.items():
            if v["market"] == market_type:
                roe_v = float(v["roe"]) if pd.notna(v["roe"]) else 0.0
                pbr_v = float(v["pbr"]) if pd.notna(v["pbr"]) else 1.0
                p_per = round(pbr_v / max(abs(roe_v) / 100.0, 0.02), 1)
                peer_rows.append({
                    "종목명": v["name"],
                    "섹터": v["sector"],
                    "이익체력(ROE_%)": roe_v,
                    "기대감(PER_배)": min(max(p_per, 0.0), 80.0),
                    "점크기": max(roe_v, 5.0),
                    "구분": "현재 분석 종목" if k == custom_tkr else ("동일 섹터 경쟁사" if v["sector"] == stock_meta["sector"] else "타 섹터 종목")
                })
        df_peers = pd.DataFrame(peer_rows)
        fig_scatter = px.scatter(
            df_peers, x="이익체력(ROE_%)", y="기대감(PER_배)", color="구분", hover_name="종목명",
            size="점크기",
            title=f"[{market_type}] 이익 체력(X축) 대비 시장 기대감 PER(Y축) 합당성 매트릭스"
        )
        fig_scatter.add_shape(type="line", x0=0, y0=0, x1=60, y1=60, line=dict(color="gray", dash="dot"))
        st.plotly_chart(fig_scatter, use_container_width=True)

    with col_v2:
        fair_price_by_sec = round(val_data["eps"] * sec_avg_per, 2)
        fair_price_by_peg1 = round(val_data["eps"] * max(val_data["eps_growth"], 5.0), 2)
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
    st.markdown(f"<div class='sec-header'>{line_icon('globe', 22)}FedWatch 금리 선물 기대치 & 거시 민감도(Beta · Correlation)</div>", unsafe_allow_html=True)
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

    mc1, mc2, mc3, mc4 = st.columns(4)
    render_top_kpi(mc1, "현재 단기 기준금리 대용(^IRX)", f"{macro_info['ffr_proxy']}%", "<span class='semantic-badge badge-neutral'>연준 실효금리 동행</span>")
    render_top_kpi(mc2, "CME 선물 내재 금리(ZQ=F)", f"{macro_info['fed_implied']}%", f"<span class='semantic-badge badge-neutral'>기준금리 대비 {macro_info['fed_implied'] - macro_info['ffr_proxy']:+.2f}%p</span>")
    render_top_kpi(mc3, "시장 지수 베타 (Beta)", f"{beta_val:.2f}", f"<span class='semantic-badge badge-neutral'>지수 상관계수 {corr_bench:+.2f}</span>")
    render_top_kpi(mc4, "원/달러 환율 상관계수", f"{corr_fx:+.2f}", "<span class='semantic-badge badge-neutral'>+강달러 수혜 / -원화강세 수혜</span>")

    st.info(f"**현재 금리 사이클 판독:** {macro_info['fed_regime_str']}")

    fig_rate = go.Figure()
    fig_rate.add_trace(go.Scatter(x=df_full.index, y=df_full["US3M_Policy"], name="단기 기준금리 (^IRX)", line=dict(color="#1E88E5", width=1.8)))
    fig_rate.add_trace(go.Scatter(x=df_full.index, y=df_full["Fed_Implied"], name="CME 연방기금 선물 내재금리 (FedWatch Proxy)", line=dict(color="#00C853", width=1.8, dash="dot")))
    fig_rate.add_trace(go.Scatter(x=df_full.index, y=df_full["US10Y"], name="미 국채 10년물 금리 (^TNX)", line=dict(color="#FB8C00", width=1.5)))
    fig_rate.update_layout(
        title="2016–2026년(10개년) 기준금리 · 선물 기대금리(FedWatch) · 미 10년물 추이 (%)",
        height=380, margin=dict(t=40, b=20, l=20, r=20)
    )
    st.plotly_chart(fig_rate, use_container_width=True)

    st.markdown(f"<div class='sec-header' style='font-size:1.1rem;'>{line_icon('news', 18)}거래량 뒤의 이유 확인 (DART 공시 & 뉴스 촉매)</div>", unsafe_allow_html=True)
    matched_row = df_all[df_all["ticker"] == custom_tkr] if not df_all.empty and "ticker" in df_all.columns else pd.DataFrame()
    if not matched_row.empty:
        r_cat = matched_row.iloc[0]
        st.success(
            f"- **에이전트 포착 핵심 재료:** {r_cat.get('catalyst', '-')}\n"
            f"- **실제 검증 원문(Quote):** {r_cat.get('quote', '-')} ({r_cat.get('loop_info', '')})"
        )
    else:
        st.info("이 종목은 오늘 선별된 Top 12 리스트 외부의 개별 조회 종목입니다. 거래량 급증 캔들 발생 시 공시 및 실적 발표 일정을 함께 점검하세요.")

with tab_bt:
    # 차트 아이콘(📊)은 요청대로 유지
    st.subheader("📊 2016–2026년 3차원 유사 국면 백테스트 수익률 분포")
    st.caption("현재의 [기술적 구조 + 밸류에이션 위치 + 금리 기대]와 가장 닮았던 과거 표본들의 실제 수익률 히스토그램입니다.")
    b_c1, b_c2, b_c3 = st.columns(3)
    with b_c1:
        fig_h1 = px.histogram(
            x=bt_short["max_rets"], nbins=12,
            labels={"x": "10영업일 내 최고 수익률 (%)"},
            title=f"단기(2주) 최고 수익률 분포 (목표 +{target_ret_short}%)"
        )
        fig_h1.add_vline(x=target_ret_short, line_dash="dash", line_color="red", annotation_text=f"목표 +{target_ret_short}%")
        st.plotly_chart(fig_h1, use_container_width=True)
    with b_c2:
        fig_h2 = px.histogram(
            x=bt_mid["max_rets"], nbins=12,
            labels={"x": "60영업일 내 최고 수익률 (%)"},
            title=f"중기(3개월) 최고 수익률 분포 (목표 +{target_ret_mid}%)"
        )
        fig_h2.add_vline(x=target_ret_mid, line_dash="dash", line_color="red", annotation_text=f"목표 +{target_ret_mid}%")
        st.plotly_chart(fig_h2, use_container_width=True)
    with b_c3:
        fig_h3 = px.histogram(
            x=bt_long["max_rets"], nbins=12,
            labels={"x": "250영업일 내 최고 수익률 (%)"},
            title=f"장기(1년) 최고 수익률 분포 (목표 +{target_ret_long}%)"
        )
        fig_h3.add_vline(x=target_ret_long, line_dash="dash", line_color="red", annotation_text=f"목표 +{target_ret_long}%")
        st.plotly_chart(fig_h3, use_container_width=True)