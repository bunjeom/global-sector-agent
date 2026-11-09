# C:\Users\hanju\global-sector-agent\pages\1_🔎_종목_딥다이브.py (상단 자동 분류 업그레이드 버전)
from datetime import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf

import engine
import utils

st.set_page_config(page_title="종목 딥다이브 | 투자터미널", layout="wide")
utils.init_session()

# =====================================================================
# 1. 한글 종목명 / 6자리 숫자 -> yfinance 표준 티커 자동 변환기
# =====================================================================
KR_US_NAME_TO_TICKER = {
    "삼성전자": "005930.KS", "삼전": "005930.KS",
    "SK하이닉스": "000660.KS", "하이닉스": "000660.KS",
    "한미반도체": "042700.KS", "리노공업": "058470.KQ",
    "현대차": "005380.KS", "현대자동차": "005380.KS", "기아": "000270.KS",
    "현대건설": "000720.KS", "삼성물산": "028260.KS",
    "LG에너지솔루션": "373220.KS", "에코프로비엠": "247540.KQ", "에코프로": "086520.KQ",
    "POSCO홀딩스": "005490.KS", "포스코홀딩스": "005490.KS",
    "삼성바이오로직스": "207940.KS", "셀트리온": "068270.KS", "알테오젠": "196170.KQ",
    "KB금융": "105560.KS", "신한지주": "055550.KS", "하나금융지주": "086790.KS",
    "NAVER": "035420.KS", "네이버": "035420.KS", "카카오": "035720.KS",
    "한화에어로스페이스": "012450.KS", "현대로템": "064350.KS", "LIG넥스원": "079550.KS",
    "HD현대일렉트릭": "267260.KS", "효성중공업": "298040.KS", "두산에너빌리티": "034020.KS",
    "KT": "030200.KS", "SK텔레콤": "017670.KS", "한국전력": "015760.KS",
    "엔비디아": "NVDA", "테슬라": "TSLA", "애플": "AAPL",
    "마이크로소프트": "MSFT", "알파벳": "GOOGL", "구글": "GOOGL",
    "아마존": "AMZN", "메타": "META", "브로드컴": "AVGO", "TSMC": "TSM",
    "마이크론": "MU", "어플라이드머티어리얼즈": "AMAT", "ASML": "ASML",
}

def resolve_ticker(user_input: str) -> str:
    """한글 종목명, 6자리 숫자, 또는 영문 티커를 yfinance 표준 티커로 자동 변환"""
    raw = user_input.strip()
    if not raw:
        return "005930.KS"
    clean_key = raw.replace(" ", "").upper()
    for k, v in KR_US_NAME_TO_TICKER.items():
        if k.replace(" ", "").upper() == clean_key:
            return v
    report_data = utils.load_latest_report()
    for item in report_data.get("stocks", []):
        if item.get("name", "").replace(" ", "") == raw.replace(" ", ""):
            return item.get("ticker", raw)
    if raw.isdigit() and len(raw) == 6:
        return f"{raw}.KS"
    return raw.upper()

# =====================================================================
# 2. 종목별 [섹터 유형] & [거시 산업 분류] 100% 자동 판별기
# =====================================================================
# 주요 종목 정밀 매핑 사전: (섹터 유형[Track 2·3], 거시 산업 분류[Track 4])[cite: 2, 5]
PREDEFINED_SECTOR_MACRO_MAP = {
    "005930.KS": ("사이클형", "메모리 (한국 상장)"),
    "000660.KS": ("사이클형", "메모리 (한국 상장)"),
    "042700.KS": ("사이클형", "반도체 장비"),
    "058470.KQ": ("사이클형", "반도체 장비"),
    "005380.KS": ("사이클형", "메모리 (한국 상장)"),
    "000270.KS": ("사이클형", "메모리 (한국 상장)"),
    "000720.KS": ("사이클형", "DC 전력·인프라"),          # 현대건설
    "005490.KS": ("사이클형", "에너지·정유"),             # POSCO홀딩스
    "105560.KS": ("사이클형", "은행·금융"),               # KB금융
    "055550.KS": ("사이클형", "은행·금융"),               # 신한지주
    "086790.KS": ("사이클형", "은행·금융"),               # 하나금융지주
    "035420.KS": ("구조성장형", "고멀티플 SW·플랫폼"),    # NAVER
    "035720.KS": ("구조성장형", "고멀티플 SW·플랫폼"),    # 카카오
    "207940.KS": ("구조성장형", "관리의료·헬스케어"),      # 삼성바이오로직스
    "068270.KS": ("구조성장형", "관리의료·헬스케어"),      # 셀트리온
    "196170.KQ": ("구조성장형", "관리의료·헬스케어"),      # 알테오젠
    "012450.KS": ("정책의존형", "방산 (한국)"),           # 한화에어로스페이스
    "064350.KS": ("정책의존형", "방산 (한국)"),           # 현대로템
    "079550.KS": ("정책의존형", "방산 (한국)"),           # LIG넥스원
    "373220.KS": ("정책의존형", "DC 전력·인프라"),        # LG에너지솔루션
    "247540.KQ": ("정책의존형", "DC 전력·인프라"),        # 에코프로비엠
    "086520.KQ": ("정책의존형", "DC 전력·인프라"),        # 에코프로
    "267260.KS": ("정책의존형", "DC 전력·인프라"),        # HD현대일렉트릭
    "298040.KS": ("정책의존형", "DC 전력·인프라"),        # 효성중공업
    "034020.KS": ("정책의존형", "DC 전력·인프라"),        # 두산에너빌리티
    "030200.KS": ("방어형", "관리의료·헬스케어"),         # KT
    "017670.KS": ("방어형", "관리의료·헬스케어"),         # SK텔레콤
    "015760.KS": ("방어형", "DC 전력·인프라"),            # 한국전력
    "NVDA": ("구조성장형", "AI 반도체·네트워킹(팹리스)"),
    "AVGO": ("구조성장형", "AI 반도체·네트워킹(팹리스)"),
    "TSM": ("구조성장형", "AI 반도체·네트워킹(팹리스)"),
    "MU": ("사이클형", "AI 반도체·네트워킹(팹리스)"),
    "AMAT": ("사이클형", "반도체 장비"),
    "ASML": ("사이클형", "반도체 장비"),
    "MSFT": ("구조성장형", "하이퍼스케일러"),
    "GOOGL": ("구조성장형", "하이퍼스케일러"),
    "AMZN": ("구조성장형", "하이퍼스케일러"),
    "META": ("구조성장형", "고멀티플 SW·플랫폼"),
    "AAPL": ("구조성장형", "고멀티플 SW·플랫폼"),
    "TSLA": ("구조성장형", "고멀티플 SW·플랫폼"),
}

@st.cache_data(ttl=86400, show_spinner=False)
def auto_detect_sector_and_macro(ticker: str) -> tuple[str, str]:
    """
    종목 티커를 입력받아 (섹터 유형, 거시 산업 분류)를 자동 판별하여 반환[cite: 2, 5].
    1순위: PREDEFINED_SECTOR_MACRO_MAP 사전 매칭
    2순위: yfinance Ticker.info의 sector / industry 키워드 실시간 분석
    """
    t_up = ticker.upper()
    if t_up in PREDEFINED_SECTOR_MACRO_MAP:
        return PREDEFINED_SECTOR_MACRO_MAP[t_up]

    is_krx = t_up.endswith((".KS", ".KQ"))
    try:
        info = yf.Ticker(t_up).info or {}
        sec = str(info.get("sector", "")).lower()
        ind = str(info.get("industry", "")).lower()
        combined = f"{sec} {ind}"
    except Exception:
        combined = ""

    # 키워드 기반 자동 추론[cite: 2, 5]
    if any(k in combined for k in ["aerospace", "defense"]):
        return "정책의존형", ("방산 (한국)" if is_krx else "DC 전력·인프라")
    if any(k in combined for k in ["solar", "renewable", "nuclear", "electrical equipment", "battery"]):
        return "정책의존형", "DC 전력·인프라"
    if any(k in combined for k in ["bank", "financial", "insurance", "capital markets", "credit"]):
        return "사이클형", "은행·금융"
    if any(k in combined for k in ["oil", "gas", "energy", "refining", "chemical", "steel", "metal", "mining"]):
        return "사이클형", "에너지·정유"
    if any(k in combined for k in ["semiconductor equipment", "materials"]):
        return "사이클형", "반도체 장비"
    if "semiconductor" in combined:
        return ("사이클형", "메모리 (한국 상장)") if is_krx else ("구조성장형", "AI 반도체·네트워킹(팹리스)")
    if any(k in combined for k in ["construction", "engineering", "building", "machinery", "industrial", "shipping", "marine", "auto"]):
        return "사이클형", ("DC 전력·인프라" if "construction" in combined or "engineering" in combined else ("메모리 (한국 상장)" if is_krx else "에너지·정유"))
    if any(k in combined for k in ["healthcare", "biotechnology", "drug", "medical", "pharma"]):
        return "구조성장형", "관리의료·헬스케어"
    if any(k in combined for k in ["utilities", "telecom", "consumer defensive", "grocery", "beverage", "tobacco", "food"]):
        return "방어형", "관리의료·헬스케어"
    if any(k in combined for k in ["software", "internet", "cloud", "information technology", "communication"]):
        return "구조성장형", "고멀티플 SW·플랫폼"

    # 기본 Fallback
    return ("사이클형", "메모리 (한국 상장)") if is_krx else ("구조성장형", "AI 반도체·네트워킹(팹리스)")

# =====================================================================
# 데이터 로드 및 캐싱 (@st.cache_data)[cite: 1, 5, 6]
# =====================================================================
@st.cache_data(ttl=1800, show_spinner=False)
def fetch_ohlcv_and_benchmark(ticker: str, period: str = "4y"):
    """종목 및 기준지수(KOSPI/KOSDAQ/NASDAQ), VKOSPI/VIX 시계열 로드"""
    stock_df = yf.Ticker(ticker).history(period=period, auto_adjust=False)
    if (stock_df is None or len(stock_df) == 0) and ticker.endswith(".KS"):
        alt_ticker = ticker.replace(".KS", ".KQ")
        stock_df = yf.Ticker(alt_ticker).history(period=period, auto_adjust=False)
        if stock_df is not None and len(stock_df) > 0:
            ticker = alt_ticker

    if stock_df is None or len(stock_df) == 0:
        return pd.DataFrame(), pd.DataFrame(), pd.Series(dtype=float)

    if ticker.endswith(".KS"):
        bench_sym, vix_sym = "^KS11", "^VIX"
    elif ticker.endswith(".KQ"):
        bench_sym, vix_sym = "^KQ11", "^VIX"
    else:
        bench_sym, vix_sym = "^IXIC", "^VIX"

    bench_df = yf.Ticker(bench_sym).history(period=period, auto_adjust=False)
    vix_df = yf.Ticker(vix_sym).history(period=period, auto_adjust=False)
    vix_s = vix_df["Close"] if (vix_df is not None and "Close" in vix_df) else pd.Series(dtype=float)
    return stock_df, bench_df, vix_s

@st.cache_data(ttl=1800, show_spinner=False)
def run_cached_earnings_track(ticker: str, sector_type: str, fin_override: dict) -> dict:
    return engine.analyze_earnings_track(ticker=ticker, sector_type=sector_type, fin_override=fin_override)

@st.cache_data(ttl=1800, show_spinner=False)
def run_cached_macro_track(ticker: str, industry_category: str) -> dict:
    stock_df, bench_df, _ = fetch_ohlcv_and_benchmark(ticker)
    return engine.analyze_macro_track(
        ticker=ticker, stock_df=stock_df, bench_df=bench_df, industry_category=industry_category
    )

@st.cache_data(ttl=1800, show_spinner=False)
def run_cached_fundamental_track(
    ticker: str, current_price: float, earn_out: dict, macro_input: dict, fund_override: dict
) -> dict:
    return engine.analyze_fundamental_track(
        ticker=ticker, current_price=current_price, earnings_output=earn_out,
        macro_input=macro_input, fund_override=fund_override
    )

@st.cache_data(ttl=1800, show_spinner=False)
def run_cached_technical_track(ticker: str, value_macro_input: dict, account_cfg: dict) -> dict:
    stock_df, bench_df, vix_s = fetch_ohlcv_and_benchmark(ticker)
    if len(stock_df) < 30:
        return {}
    return engine.analyze_technical_track(
        df=stock_df, bench_df=bench_df, vix_series=vix_s,
        value_macro_input=value_macro_input, account_cfg=account_cfg
    )

# =====================================================================
# 상단 종목 선택 및 컨트롤 바 (티커 변경 시 섹터·거시분류 자동 스위칭)
# =====================================================================
st.markdown(f'## {utils.line_icon("target", 22)} 종목 딥다이브 — 4대 트랙 통합 분석 터미널', unsafe_allow_html=True)

col_t1, col_t2, col_t3, col_t4, col_t5 = st.columns([1.7, 1.2, 1.5, 1.3, 1.1])
with col_t1:
    raw_query = st.text_input(
    "종목명 또는 티커 (예: 삼성전자, 000720, NVDA)",
    value=st.session_state.get("selected_ticker", "삼성전자")
)
ticker_input = resolve_ticker(raw_query)
# 티커가 바뀌면 자동으로 해당 종목의 (섹터 유형, 거시 산업 분류)를 감지하여 드롭다운 초기값 갱신
sector_options = ["구조성장형", "사이클형", "방어형", "정책의존형"]
macro_ind_list = list(engine.INDUSTRY_MACRO_HYPOTHESIS_MAP.keys())

auto_sec, auto_mac = auto_detect_sector_and_macro(ticker_input)
if st.session_state.get("_last_detected_ticker") != ticker_input:
    st.session_state["_last_detected_ticker"] = ticker_input
    st.session_state["auto_sector_select"] = auto_sec if auto_sec in sector_options else "구조성장형"
    st.session_state["auto_macro_select"] = auto_mac if auto_mac in macro_ind_list else macro_ind_list[0]

with col_t2:
    sector_type_opt = st.selectbox(
        "섹터 유형 (자동 감지·변경 가능)",
        sector_options,
        key="auto_sector_select"
    )
with col_t3:
    macro_ind_opt = st.selectbox(
        "거시 산업 분류 (자동 감지·변경 가능)",
        macro_ind_list,
        key="auto_macro_select"
    )
with col_t4:
    sim_mode_opt = st.selectbox(
        "특수 상황 시뮬레이션 (Step 1·3 점검)",
        ["정상 가동", "단발 어닝쇼크 (PEAD 냉각기 D-3)", "실적 발표 임박 (블랙아웃 D-2)", "하드 킬 (악성 이벤트/드라이버 붕괴)", "거래정지 동결 (FROZEN_HALT)"],
        index=0
    )
with col_t5:
    log_scale_opt = st.checkbox("로그 스케일 차트", value=True)

is_krx_ticker = ticker_input.upper().endswith((".KS", ".KQ"))
stock_df_raw, _, _ = fetch_ohlcv_and_benchmark(ticker_input)

if len(stock_df_raw) < 30:
    st.error(f"'{raw_query}' (변환 티커: {ticker_input})의 일봉 데이터를 불러올 수 없습니다. 티커(예: 005930.KS, NVDA)를 확인해 주세요.")
    st.stop()

last_close = float(stock_df_raw["Close"].iloc[-1])

evt_block_opt = sim_mode_opt == "하드 킬 (악성 이벤트/드라이버 붕괴)"
is_halted_opt = sim_mode_opt == "거래정지 동결 (FROZEN_HALT)"
pead_days_opt = 3 if sim_mode_opt == "단발 어닝쇼크 (PEAD 냉각기 D-3)" else 0
days_to_earn_opt = 2 if sim_mode_opt == "실적 발표 임박 (블랙아웃 D-2)" else 45

# ---------------------------------------------------------------------
# 4대 트랙 + 종합 결론(Synthesis v1.1) 단방향 파이프라인 실행[cite: 1, 2, 4, 5, 6]
# ---------------------------------------------------------------------
# 1) Track 4 (거시적 분석 v1.0)
macro_out = run_cached_macro_track(ticker_input, macro_ind_opt)

# 2) Track 2 (실적 분석 v1.0)
fin_override_cfg = {"thesis": {"hard_block_triggered": evt_block_opt}} if evt_block_opt else {}
earn_out = run_cached_earnings_track(ticker_input, sector_type_opt, fin_override_cfg)

# 3) Track 3 (펀더멘탈 분석 v1.0)
macro_to_fund_input = {
    "rf_10y": macro_out["macro_regime"]["rf_10y_current"],
    "market_erp": 0.055,
}
fund_out = run_cached_fundamental_track(ticker_input, last_close, earn_out, macro_to_fund_input, {})

# 4) Track 1 (기술적 분석 v1.2)
value_macro_input = {
    "thesis_status": fund_out["thesis_status"],
    "kill_flags": ["DRIVER_BROKEN"] if evt_block_opt else [],
    "buy_tranches": fund_out["buy_tranches"],
    "sell_target": fund_out["sell_target"],
    "position_cap": fund_out["position_cap"],
    "real_signal": earn_out["verdict"]["real_signal"],
    "analyzed_at": datetime.now().strftime("%Y-%m-%d"),
    "macro_regime_override": None,
}
account_cfg = {
    "total_equity": st.session_state.get("total_equity", 100_000_000),
    "t_sleeve_ratio": st.session_state.get("t_sleeve_ratio", 0.30),
    "is_krx": is_krx_ticker,
    "hard_block_evt": evt_block_opt or is_halted_opt,
    "earnings_blackout": days_to_earn_opt <= 3,
    "mktcap_krw": 1_000_000_000_000,
    "loss_streak": 0,
    "existing_v_weight": 0.0,
}

tech_out = run_cached_technical_track(ticker_input, value_macro_input, account_cfg)
if not tech_out:
    st.warning("기술적 분석 연산 결과가 없습니다.")
    st.stop()

# 5) 종합 결론 (Synthesis v1.1) 실행[cite: 6]
synth_out = engine.synthesize_4tracks(
    ticker=ticker_input,
    current_price=last_close,
    tech_out=tech_out,
    earn_out=earn_out,
    fund_out=fund_out,
    macro_out=macro_out,
    synth_cfg={
        "stock_name": ticker_input.upper(),
        "is_halted": is_halted_opt,
        "bad_evt_flag": evt_block_opt,
        "pead_lockout_days_left": pead_days_opt,
        "days_to_earnings": days_to_earn_opt,
    }
)

pq_info = fund_out["price_qualification"]
m_cls = macro_out["classification"]
arch_info = synth_out["archetype"]

# =====================================================================
# 종합 결론(첫 번째 탭) + 4대 트랙 상세 탭 구성[cite: 6]
# =====================================================================
tab_syn, tab_t1, tab_t2, tab_t3, tab_t4 = st.tabs([
    f"종합 결론 · Synthesis v1.1 ({arch_info['code']})",
    f"Track 1 · 기술적 분석 v1.2 ({tech_out['track_signal']})",
    f"Track 2 · 실적 분석 v1.0 ({earn_out['verdict']['grade']} {earn_out['regime']['earnings_regime']})",
    f"Track 3 · 펀더멘탈 분석 v1.0 ({pq_info['grade_icon']} {synth_out['v_action_card']['price_qual_code']})",
    f"Track 4 · 거시적 분석 v1.0 ({m_cls['grade_icon']} 고유 R² {int(m_cls['r2_macro_own']*100)}%)"
])


# =====================================================================
# [탭 0] 4대 트랙 종합 결론 (Synthesis v1.1 — Waterfall Gate & 3줄 브리핑)[cite: 6]
# =====================================================================
with tab_syn:
    hk = synth_out["hard_kill_gate"]
    tl_brief = synth_out["three_line_briefing"]
    lights = synth_out["track_lights"]
    v_card = synth_out["v_action_card"]
    t_card = synth_out["t_action_card"]
    m_ov = synth_out["macro_risk_overlay"]
    m_mat = synth_out["mispricing_matrix"]
    u_watch = synth_out["unified_watchlist"]
    r_flags = synth_out["radar_flags"]

    # 1. 하드 킬 / 동결 / PEAD 냉각기 경보 바 (발동 시 최상단 강조)[cite: 6]
    if hk["is_killed"]:
        st.markdown(f"""
        <div class="term-banner" style="border:2px solid #dc2626;background:#fef2f2;">
            <div class="term-banner-line1" style="color:#dc2626;">
                {utils.line_icon("alert", 20)}
                <span class="term-badge-inv" style="background:#dc2626;border-color:#dc2626;">Step 1 하드 킬 / 거래 동결 발동</span>
                {" · ".join(hk["kill_codes"])}
            </div>
            <div style="font-size:0.95rem;font-weight:600;color:#991b1b;">
                사유: {hk["kill_reason"]}
            </div>
        </div>
        """, unsafe_allow_html=True)
    elif hk["pead_lockout_days_left"] > 0:
        st.markdown(f"""
        <div class="term-card" style="border-left:4px solid #f97316;background:#fff7ed;">
            <b>{utils.line_icon("alert", 16)}일회성 어닝 쇼크 분리 처리 (PEAD 냉각기 작동 중 — 잔여 {hk["pead_lockout_days_left"]}거래일):</b>
            드라이버 붕괴 없는 단발성 실적 하회이므로 영구 하드 킬(G5)로 차단하지 않고, 발표 후 5거래일 냉각기 경과 뒤 바닥 확인 시 매수를 재개방합니다.
        </div>
        """, unsafe_allow_html=True)

    # 2. 아키타입 헤더 & 직관적 3줄 결론 템플릿 (3장 명세)[cite: 6]
    hc_badge = (
        f'<span class="term-badge-inv" style="background:#16a34a;border-color:#16a34a;">🔥 최고 확신 기회 (ARCH_1 · 거시 조건 무관)</span>'
        if arch_info["high_conviction_badge"] else ""
    )
    radar_badges = "".join([
        f'<span class="term-badge">{k}</span>' for k, val in r_flags.items() if val
    ])

    st.markdown(f"""
    <div class="term-banner">
        <div style="margin-bottom:8px;">
            {utils.line_icon("target", 18)}
            <span class="term-badge-inv">{arch_info["code"]}</span>
            <span style="font-weight:700;font-size:1.05rem;">{arch_info["title"]}</span>
            &nbsp;{hc_badge} {radar_badges}
        </div>
        <div class="term-banner-line1">
            ① [오늘의 행동 — Action] {tl_brief["line_1_action"]}
        </div>
        <div class="term-banner-line2">
            <b>② [핵심 근거 — Why]</b> {tl_brief["line_2_why"]}
        </div>
        <div class="term-banner-line3">
            {utils.line_icon("check", 14)}<b>③ [다음 방아쇠 & 종료 조건 — Trigger]</b> {tl_brief["line_3_trigger"]}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 3. 4대 트랙 권한 분리 요약 스트립 (엔진 · 가격표 · 방아쇠 · 계기판)[cite: 6]
    c_l1, c_l2, c_l3, c_l4 = st.columns(4)
    strip_items = [
        (c_l1, "트랙 2 · 실적 (엔진 Engine)", lights["earnings"]),
        (c_l2, "트랙 3 · 펀더멘탈 (가격표 Price Tag)", lights["fundamental"]),
        (c_l3, "트랙 1 · 기술 (방아쇠 Trigger)", lights["technical"]),
        (c_l4, "트랙 4 · 거시 (계기판 Dashboard)", lights["macro"]),
    ]
    for col_obj, role_title, l_data in strip_items:
        with col_obj:
            st.markdown(f"""
            <div class="term-card">
                <div style="font-size:0.78rem;color:#64748b;font-weight:600;">{role_title}</div>
                <div style="font-size:0.98rem;font-weight:700;margin:4px 0;color:#111;">
                    {l_data["icon"]} {l_data["label"]}
                </div>
                <div style="font-size:0.82rem;color:#475569;">{l_data["state"]}</div>
            </div>
            """, unsafe_allow_html=True)

    # 4. V 트랙 (가치 저가매수) & T 트랙 (독립 트레이딩) 듀얼 실행 티켓 (Step 2 & Step 3)[cite: 6]
    col_vt1, col_vt2 = st.columns(2)

    with col_vt1:
        ov_tag = "🟢 고신뢰 수평 지지(⭐⭐+) 겹침" if v_card["high_confidence_overlap"] else "일반 구간"
        st.markdown(f"""
        <div class="term-card">
            <div style="font-weight:700;font-size:1.02rem;margin-bottom:8px;">
                {utils.line_icon("layers", 18)}Step 2. V 트랙 (가치 연동 저가매수) 실행 카드
            </div>
            <div style="margin-bottom:10px;">
                <span class="term-badge-inv">{v_card["badge_text"]}</span>
                <span class="term-badge">가격 자격: {v_card["price_qual_code"]}</span>
                <span class="term-badge">상태 코드: {v_card["status_code"]}</span>
            </div>
            <ul class="term-list">
                <li><b>현재가 vs 1차 매수기준선 괴리율:</b> 현재 <b>{int(last_close):,}원</b> (1차선 대비 <b>{v_card["distance_to_1st_tranche_pct"]*100:+.1f}%</b>)</li>
                <li><b>기준 차수 및 매수기준선 가격:</b> <b>{v_card["active_tranche"]}차 매수선 ({v_card["tranche_price"]:,}원)</b> · {ov_tag}</li>
                <li><b>당일 집행 비율 (단일 감속 하한 50% 규칙):</b> <b>{int(v_card["execution_ratio"]*100)}%</b> → 권장 매수 비중 <b>총자산의 {v_card["recommended_weight"]*100:.2f}%</b></li>
                <li><b>1Y 채택 매도목표가 / 기대수익률:</b> <b>{v_card["sell_target_1y"]:,}원</b> / 확률가중 <b>{v_card["expected_return_weighted"]*100:+.1f}%</b> (MDD Ratio <b>{v_card["mdd_ratio"]:.2f}</b>)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col_vt2:
        st.markdown(f"""
        <div class="term-card">
            <div style="font-weight:700;font-size:1.02rem;margin-bottom:8px;">
                {utils.line_icon("activity", 18)}Step 3. T 트랙 (독립 트레이딩) 실행 카드
            </div>
            <div style="margin-bottom:10px;">
                <span class="term-badge-inv">{t_card["badge_text"]}</span>
                <span class="term-badge">상태 코드: {t_card["status_code"]}</span>
                <span class="term-badge">고평가 위험계수 조정: ×{t_card["risk_modifier_applied"]:.1f}</span>
            </div>
            <ul class="term-list">
                <li><b>진입가 / 손절가 (종가 기준):</b> 진입 <b>{t_card["entry_price"]:,}원</b> / 손절 <b>{t_card["stop_price"]:,}원</b></li>
                <li><b>산출 주문 수량 / 1차 목표가:</b> <b>{t_card["shares"]:,}주</b> / 목표 <b>{t_card["target_price"]:,}원</b> (예상 <b>{t_card["expected_r"]}R</b>)</li>
                <li><b>시간 손절 및 무효화 규칙:</b> {t_card["time_stop_rule"]} ｜ {t_card["invalidation_rule"]}</li>
                <li><b>실적 발표 보유분 지침 (T_EARNINGS_RULE):</b> {t_card["earnings_hold_guide"]}</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # 5. 시장 오판 포인트 매트릭스 + 거시 위험 속성 오버레이 & 통합 D-day 감시 캘린더[cite: 6]
    col_b1, col_b2 = st.columns([1.15, 1.0])

    with col_b1:
        sens_badges = " · ".join([
            f"{s['var_name']} 1σ({s['shock_1sd']}) → <b>{s['impact']}</b>"
            for s in m_ov["top_sensitivities"]
        ])
        st.markdown(f"""
        <div class="term-card">
            <div style="font-weight:700;margin-bottom:8px;">
                {utils.line_icon("compass", 16)}Step 4. 시장 오판 2대 축 & 거시 수익률 분해·위험 속성 오버레이 (찬반 투표 미합산)
            </div>
            <ul class="term-list">
                <li>{m_mat["earnings_edge"]}</li>
                <li>{m_mat["multiple_edge"]}</li>
                <li>{m_mat["macro_attribution"]}</li>
                <li><b>⚠️ 거시 1σ 조건부 민감도 (위험 속성):</b> {sens_badges}</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col_b2:
        st.markdown(f'**{utils.line_icon("shield", 16)} 통합 D-day 감시 캘린더 (우선순위 순 정렬)**', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(u_watch), width="stretch", hide_index=True)


# =====================================================================
# [탭 1] 기술적 분석 트랙 v1.2[cite: 1]
# =====================================================================
with tab_t1:
    banner = tech_out["summary_banner"]
    t_ticket = tech_out["t_ticket"]
    lc = tech_out["lifecycle_state"]
    ctx = tech_out["context"]
    mods = tech_out["modules"]
    lvls = tech_out["levels"]
    chart_df = tech_out["chart_df"]

    verdict_badge = f'<span class="term-badge-inv">T {banner["t_verdict"]}</span>'
    setup_badge = f'<span class="term-badge">{banner["setup_id"] or "셋업 없음"} · 등급 {banner["grade"] or "-"}</span>'

    if banner["t_verdict"] == "진입" and t_ticket.get("is_executable"):
        line2_html = (
            f'진입가 <b>{t_ticket["entry_price"]:,}원</b> · '
            f'손절가(종가) <b>{t_ticket["stop_price"]:,}원</b> · '
            f'수량 <b>{t_ticket["shares"]:,}주</b> ({t_ticket["position_value"]:,}원) · '
            f'1차목표 <b>{t_ticket["target_1"]:,}원</b> · '
            f'예상 <b>{t_ticket["expected_r"]}R</b> &nbsp;|&nbsp; {banner["next_or_end_condition"]}'
        )
    else:
        line2_html = f'{banner["next_or_end_condition"]}'

    high_conf_badge = (
        f' <span class="term-badge">{utils.line_icon("check", 12)}고신뢰 지지 겹침</span>'
        if tech_out["overlap_flags"]["high_conf_buy_tranches"] else ""
    )

    st.markdown(f"""
    <div class="term-banner">
        <div class="term-banner-line1">
            {utils.line_icon("activity", 18)}{verdict_badge}{setup_badge} {banner["headline_reason"]}
        </div>
        <div class="term-banner-line2">
            {line2_html}
        </div>
        <div class="term-banner-line3">
            {utils.line_icon("layers", 14)}<b>{banner["v_summary_line"]}</b>{high_conf_badge}
        </div>
    </div>
    """, unsafe_allow_html=True)

    plot_df = chart_df.iloc[-180:].copy()
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03,
        row_heights=[0.62, 0.18, 0.20],
        subplot_titles=("📈 일봉 가격 구조 · 수평 ZONE · 추세 채널 · 측정 목표(S-TGT)", "📊 거래량 & RVOL", "📊 RSI(14) & 모드 구간")
    )

    fig.add_trace(go.Candlestick(
        x=plot_df.index, open=plot_df["Open"], high=plot_df["High"],
        low=plot_df["Low"], close=plot_df["Close"], name="OHLC",
        increasing_line_color="#16a34a", decreasing_line_color="#dc2626"
    ), row=1, col=1)

    ma_styles = [
        ("SMA5", "#64748b", "dot", 1.0),
        ("SMA20", "#111111", "solid", 1.6),
        ("SMA25", "#2563eb", "dash", 1.3),
        ("SMA60", "#059669", "solid", 1.4),
        ("SMA120", "#d97706", "dot", 1.4),
        ("SMA200", "#7c3aed", "solid", 1.5),
    ]
    for ma_col, color, dash, width in ma_styles:
        if ma_col in plot_df:
            fig.add_trace(go.Scatter(
                x=plot_df.index, y=plot_df[ma_col], mode="lines", name=ma_col,
                line=dict(color=color, width=width, dash=dash)
            ), row=1, col=1)

    for hz in lvls["horizontal_zones"][:6]:
        opacity = 0.06 * hz["stars"]
        fill_col = "rgba(22, 163, 74, " + str(opacity) + ")" if hz["type"] == "지지" else "rgba(220, 38, 38, " + str(opacity) + ")"
        fig.add_hrect(
            y0=hz["low"], y1=hz["high"], fillcolor=fill_col, line_width=0,
            annotation_text=f"{'⭐'*hz['stars']} {hz['type']} {int(hz['price']):,}",
            annotation_position="right", row=1, col=1
        )

    tgts = lvls.get("targets", {})
    for t_key, t_val in tgts.items():
        if t_key in ["MB0.25", "MB0.5", "T1"]:
            fig.add_hline(
                y=t_val, line_dash="dashdot", line_color="#111111", line_width=1,
                annotation_text=f"S-TGT {t_key}: {int(t_val):,}", annotation_position="top left", row=1, col=1
            )

    vol_colors = np.where(plot_df["Close"] >= plot_df["Open"], "#16a34a", "#dc2626")
    fig.add_trace(go.Bar(
        x=plot_df.index, y=plot_df["Volume"], name="거래량", marker_color=vol_colors
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=plot_df.index, y=plot_df["RSI14"], mode="lines", name="RSI(14)",
        line=dict(color="#111111", width=1.5)
    ), row=3, col=1)
    fig.add_hrect(y0=40, y1=50, fillcolor="rgba(17,17,17,0.08)", line_width=0, row=3, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="#dc2626", row=3, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="#2563eb", row=3, col=1)

    fig.update_yaxes(type="log" if log_scale_opt else "linear", row=1, col=1)
    fig.update_layout(
        height=720, margin=dict(l=20, r=20, t=35, b=20),
        xaxis_rangeslider_visible=False, template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
    )
    st.plotly_chart(fig, width="stretch")

    with st.expander("상세 기술 진단 펼치기 — A. 7상태 기계 & 환경 · B. 6대 판정 모듈 · C. 수직 레벨 맵 & T 티켓", expanded=True):
        st.markdown(f'#### {utils.line_icon("compass", 18)} 섹션 A — 생애주기 상태 기계 (3장) & 환경 배지 (2-2~2-4)', unsafe_allow_html=True)
        all_states = [("D", "D 하락진행"), ("B", "B 바닥형성"), ("R", "R 횡보"), ("S", "S 추세시작"), ("H", "H 추세유지"), ("W", "W 약화"), ("K", "K 붕괴경계")]
        nodes_html = "".join([
            f'<span class="{"state-node-active" if lc["state"] == code else "state-node"}">'
            f'{lc["state_label"] if lc["state"] == code else lbl}</span>'
            for code, lbl in all_states
        ])
        st.markdown(f"""
        <div class="term-card">
            <div style="margin-bottom:10px;">{nodes_html}</div>
            <div style="font-size:0.88rem;color:#333;">
                <b>직전 상태:</b> {lc["prev_state"]} &nbsp;→&nbsp; <b>현재 상태:</b> {lc["state_label"]} &nbsp;|&nbsp;
                <b>최근 전이 사유:</b> {lc["transition_reason"]}
            </div>
            <hr style="margin:10px 0;border:none;border-top:1px solid #e2e8f0;">
            <div style="font-size:0.88rem;">
                <span class="term-badge">S-REG 지수 레짐: {ctx["s_reg_label"]}</span>
                <span class="term-badge">S-VIX 공포 국면: {ctx["s_vix"]}</span>
                <span class="term-badge">S-MODE 타이밍 모드: {ctx["s_mode"]}</span>
                <span class="term-badge">S-RS 상대강도: 주봉 [{ctx["s_rs_weekly"]}] · 일봉 [{ctx["s_rs_daily"]}]</span>
                <span class="term-badge">S-MAT 배열: {ctx["s_mat"]} (SMA120 {ctx["sma120_slope"]})</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f'#### {utils.line_icon("layers", 18)} 섹션 B — 6대 판정 모듈 단계 (2-8)', unsafe_allow_html=True)
        m_cols = st.columns(3)
        mod_titles = [
            ("breakout", "추세돌파 (Breakout)"),
            ("disparity", "이격도 과이탈 (BNF)"),
            ("bottom", "바닥 (Bottom)"),
            ("pullback", "눌림 (Pullback)"),
            ("range", "레인지 (Range)"),
            ("weakness", "약화·붕괴 (Weakness)"),
        ]
        for idx_m, (m_key, m_title) in enumerate(mod_titles):
            m_data = mods[m_key]
            with m_cols[idx_m % 3]:
                st.markdown(f"""
                <div class="term-card">
                    <div style="font-size:0.82rem;color:#64748b;font-weight:600;">{m_title}</div>
                    <div style="font-size:1.05rem;font-weight:700;margin:4px 0;color:#111;">{m_data["label"]}</div>
                    <div style="font-size:0.82rem;color:#475569;">{m_data["detail"]}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown(f'#### {utils.line_icon("shield", 18)} 섹션 C — 단일 수직 가격축 레벨 맵 & T 거래 티켓 / 리스크 명세', unsafe_allow_html=True)
        col_c1, col_c2 = st.columns([1.15, 1.0])

        with col_c1:
            ladder_items = [{"price": int(last_close), "tag": "현재 종가 (Current Close)", "kind": "NOW"}]
            for hz in lvls["horizontal_zones"]:
                ladder_items.append({
                    "price": int(hz["price"]),
                    "tag": f"{'⭐'*hz['stars']} 수평 {hz['type']} ZONE [{int(hz['low']):,} ~ {int(hz['high']):,}]",
                    "kind": hz["type"]
                })
            for tv in tech_out["tranche_verdicts"]:
                ladder_items.append({
                    "price": int(tv["price"]),
                    "tag": f"V {tv['tranche']}차 매수기준선 → 판정: {tv['verdict']} ({int(tv['ratio']*100)}%)" + (" [고신뢰]" if tv["high_confidence"] else ""),
                    "kind": "V_BUY"
                })
            if value_macro_input.get("sell_target"):
                ladder_items.append({
                    "price": int(value_macro_input["sell_target"]),
                    "tag": "V 펀더멘탈 매도목표가 (앵커2·3 평균)",
                    "kind": "V_SELL"
                })
            ladder_items.sort(key=lambda x: x["price"], reverse=True)

            li_html = "".join([
                f'<li style="{"background:#f1f5f9;font-weight:700;padding:8px 6px;" if it["kind"]=="NOW" else ""}">'
                f'<span style="font-family:monospace;display:inline-block;width:110px;">{it["price"]:,}원</span> '
                f'<span class="term-badge">{it["kind"]}</span> {it["tag"]}</li>'
                for it in ladder_items
            ])
            st.markdown(f"""
            <div class="term-card">
                <div style="font-weight:700;margin-bottom:8px;">{utils.line_icon("tag", 15)}단일 수직 가격축 레벨 맵 (Track 3 매수·매도선 실시간 연동)</div>
                <ul class="term-list">{li_html}</ul>
            </div>
            """, unsafe_allow_html=True)

        with col_c2:
            if t_ticket.get("setup_id"):
                st.markdown(f"""
                <div class="term-card">
                    <div style="font-weight:700;margin-bottom:8px;">{utils.line_icon("check", 15)}T 거래 티켓 & 5-1 리스크·사이징 명세</div>
                    <ul class="term-list">
                        <li><b>셋업 ID / 등급:</b> {t_ticket["setup_id"]} ({t_ticket.get("setup_name", "")}) · 등급 {t_ticket.get("grade", "-")}</li>
                        <li><b>목적:</b> {t_ticket.get("purpose", "-")}</li>
                        <li><b>진입 기준가 (호가 보정):</b> {t_ticket["entry_price"]:,}원 (추격 금지 상한: {t_ticket.get("chase_ban_price", 0):,}원)</li>
                        <li><b>손절 기준가 (종가 기준):</b> {t_ticket["stop_price"]:,}원 (주당 1R 위험: {t_ticket["unit_r_krw"]:,}원)</li>
                        <li><b>할당 위험 금액 (R$):</b> {t_ticket.get("r_budget_krw", 0):,}원</li>
                        <li><b>산출 수량 / 포지션 금액:</b> {t_ticket["shares"]:,}주 ({t_ticket["position_value"]:,}원)</li>
                        <li><b>1차 목표 / 잔량 규칙:</b> {t_ticket["target_1"]:,}원 (예상 {t_ticket["expected_r"]}R) / {t_ticket["target_2"]}</li>
                        <li><b>시간 손절 / 최대 보유:</b> {t_ticket["time_stop_days"]}거래일 내 {t_ticket["time_stop_min_price"]:,}원 미달 시 청산 / 최대 {t_ticket["max_hold_days"]}일</li>
                        <li><b>무효화 조건:</b> {t_ticket["invalidation"]}</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="term-card">
                    <div style="font-weight:700;margin-bottom:6px;">{utils.line_icon("alert", 15)}T 거래 티켓 미생성</div>
                    <div style="font-size:0.9rem;color:#555;">{t_ticket.get("reason", "성립된 셋업 없음")}</div>
                </div>
                """, unsafe_allow_html=True)

            bt = tech_out["backtest_validation"]
            st.markdown(f"""
            <div class="term-card">
                <div style="font-weight:700;margin-bottom:6px;">{utils.line_icon("activity", 15)}10장. 조건부 백테스트 기대값 검증 (왕복비용 0.30% 차감)</div>
                <div style="font-size:0.88rem;">
                    표본 수(N): <b>{bt["sample_n"]}회</b> · 승률(p): <b>{bt["win_rate"]}%</b> ·
                    평균 이익: <b>+{bt["avg_win_r"]}R</b> · 평균 손실: <b>-{bt["avg_loss_r"]}R</b> ·
                    기대값(E): <b>{bt["expectancy_r"]}R</b> · 최대 연속 손실: <b>{bt["max_loss_streak"]}회</b>
                </div>
            </div>
            """, unsafe_allow_html=True)


# =====================================================================
# [탭 2] 실적 분석 트랙 v1.0[cite: 2]
# =====================================================================
with tab_t2:
    reg = earn_out["regime"]
    verd = earn_out["verdict"]
    proj = earn_out["projections"]
    sus = earn_out["sustainability_metrics"]
    rev_d = earn_out["revision_dynamics"]
    qual = earn_out["quality_checks"]
    th_drv = earn_out["thesis_driver"]

    cycle_badge = f'<span class="term-badge">{reg["cycle_stage"]}</span>' if reg.get("cycle_stage") else ""
    peak_badge = f'<span class="term-badge-inv">피크아웃 경고</span>' if verd["peak_out_warning"] else ""
    inv_fcf_badge = (
        f'<span class="term-badge">{utils.line_icon("check", 12)}G1 정상 투자기 FCF 감소 면제</span>'
        if qual["is_investment_phase_fcf"] else ""
    )

    st.markdown(f"""
    <div class="term-banner">
        <div class="term-banner-line1">
            {utils.line_icon("activity", 18)}
            <span class="term-badge-inv">{reg["earnings_regime"]} · {reg["regime_label"]}</span>
            <span class="term-badge">{verd["grade"]} 종합 {verd["score"]:.2f}점 / 2.00</span>
            <span class="term-badge">섹터: {reg["sector_type"]}</span>
            <span class="term-badge">2차 미분: 성장 {reg["growth_acceleration"]}</span>
            {cycle_badge}{peak_badge}
        </div>
        <div class="term-banner-line2">
            {verd["headline_reason"]}
        </div>
        <div class="term-banner-line3">
            {utils.line_icon("shield", 14)}<b>Thesis 상태: [{th_drv["status"]}]</b> &nbsp;|&nbsp;
            <b>무효화 조건:</b> {th_drv["invalidation_condition"]} ({th_drv["current_status"]}) {inv_fcf_badge}
        </div>
    </div>
    """, unsafe_allow_html=True)

    c_m1, c_m2, c_m3 = st.columns(3)
    with c_m1:
        lev_tag = "🟢 고정비 레버리지 작동" if sus["operating_leverage_active"] else "🔴 한계비용 상승(마진 둔화)"
        st.markdown(f"""
        <div class="term-card">
            <div style="font-size:0.82rem;color:#64748b;font-weight:600;">① 증분 영업이익률 (Incremental OPM · E7)</div>
            <div style="font-size:1.15rem;font-weight:700;margin:6px 0;font-family:monospace;">
                증분 {sus["incremental_opm"]*100:.1f}% &nbsp;vs&nbsp; 평균 {sus["average_opm"]*100:.1f}%
            </div>
            <div style="font-size:0.84rem;color:#333;">판정: <b>{lev_tag}</b> (ΔOP / ΔSales)</div>
        </div>
        """, unsafe_allow_html=True)

    with c_m2:
        rpo_ok = sus["rpo_yoy_growth"] > sus["sales_yoy_growth"]
        rpo_tag = "🟢 미래 매출 예약 누적 (최우수)" if rpo_ok else "🔴 수주 소진 (성장 저뭄 경고)"
        st.markdown(f"""
        <div class="term-card">
            <div style="font-size:0.82rem;color:#64748b;font-weight:600;">② 수주잔고(RPO) vs 매출 성장률 (E7)</div>
            <div style="font-size:1.15rem;font-weight:700;margin:6px 0;font-family:monospace;">
                RPO +{sus["rpo_yoy_growth"]*100:.1f}% &nbsp;vs&nbsp; 매출 +{sus["sales_yoy_growth"]*100:.1f}%
            </div>
            <div style="font-size:0.84rem;color:#333;">판정: <b>{rpo_tag}</b></div>
        </div>
        """, unsafe_allow_html=True)

    with c_m3:
        st.markdown(f"""
        <div class="term-card">
            <div style="font-size:0.82rem;color:#64748b;font-weight:600;">③ NTM EPS 리비전 변곡점 추적 (E3)</div>
            <div style="font-size:1.15rem;font-weight:700;margin:6px 0;font-family:monospace;">
                90일 {rev_d["change_90d"]*100:+.1f}% &nbsp;/&nbsp; 30일 {rev_d["change_30d"]*100:+.1f}%
            </div>
            <div style="font-size:0.84rem;color:#333;">판정: <b>{rev_d["status"]}</b> {"(변곡점 포착)" if rev_d["inflection_flag"] else ""}</div>
        </div>
        """, unsafe_allow_html=True)

    col_p1, col_p2 = st.columns([1.15, 1.0])
    with col_p1:
        e1y = proj["eps_1y"]
        e2y = proj["eps_2y"]
        c_bear_str = f"{e1y['cycle_bear']:,}원" if e1y.get("cycle_bear") is not None else "해당 없음 (비사이클)"
        st.markdown(f"""
        <div class="term-card">
            <div style="font-weight:700;margin-bottom:8px;">
                {utils.line_icon("target", 16)}E4·E5. 평가 시점별 미래 시나리오 NTM EPS (펀더멘탈 트랙 분자 전달값)
            </div>
            <div style="margin-bottom:8px;font-size:0.85rem;">
                <span class="term-badge-inv">E2 가이던스 확실성 조정: {proj["certainty_adjustment"]*100:+.0f}%p</span>
                <span style="color:#475569;">({proj["certainty_label"]})</span>
            </div>
            <ul class="term-list">
                <li><b>1Y NTM EPS 혼합식 ({e1y["formula"]}):</b> Base(50%) <b>{e1y["base"]:,}원</b> · Bull(25%) <b>{e1y["bull"]:,}원</b> · Bear(25%) <b>{e1y["bear"]:,}원</b></li>
                <li><b>1Y CycleBear (사이클형 정점→저점 낙폭 기저율):</b> <b>{c_bear_str}</b></li>
                <li><b>2Y NTM EPS 혼합식 ({e2y["formula"]}):</b> Base(50%) <b>{e2y["base"]:,}원</b> · Bull(25%) <b>{e2y["bull"]:,}원</b> · Bear(25%) <b>{e2y["bear"]:,}원</b></li>
                <li><b>1-2. 자본구조(분모) 점검:</b> {th_drv["capital_structure_note"]}</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col_p2:
        rev_rows_html = "".join([
            f'<li><span style="display:inline-block;width:115px;font-weight:600;">{r["period"]}</span> '
            f'NTM <b>{r["ntm_eps"]}</b> · FY+1 <b>{r["fy1_eps"]}</b> · FY+2 <b>{r["fy2_eps"]}</b> ({r["analysts"]})</li>'
            for r in rev_d["table"]
        ])
        st.markdown(f"""
        <div class="term-card">
            <div style="font-weight:700;margin-bottom:8px;">
                {utils.line_icon("activity", 16)}표 2 — E3. NTM EPS 리비전 다이나믹스 (창 이동 착시 보정)
            </div>
            <ul class="term-list">{rev_rows_html}</ul>
            <div style="font-size:0.82rem;color:#555;margin-top:6px;">
                * 분기 롤오버 착시를 배제하기 위해 FY+1과 FY+2 추정치가 동반 상승할 때만 진성 상향으로 인정합니다.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with st.expander("상세 실적 진단 펼치기 — 5대 미래 지속성 평가 축 · 표 1 서프라이즈 · 표 3 이익의 질 · 시장 오판 포인트", expanded=True):
        st.markdown(f'#### {utils.line_icon("compass", 18)} 3장. 실적 생애주기 5단계 (Earnings Regime) 위치', unsafe_allow_html=True)
        g_regimes = [
            ("G1", "G1 초기 가속/턴어라운드"),
            ("G2", "G2 우량 지속 성장"),
            ("G3", "G3 성숙/현금 회수"),
            ("G4", "G4 피크아웃/성장 둔화"),
            ("G5", "G5 구조적 쇠퇴/쇼크"),
        ]
        g_nodes_html = "".join([
            f'<span class="{"state-node-active" if reg["earnings_regime"] == g_code else "state-node"}">{g_lbl}</span>'
            for g_code, g_lbl in g_regimes
        ])
        st.markdown(f'<div class="term-card">{g_nodes_html}</div>', unsafe_allow_html=True)

        col_d1, col_d2 = st.columns(2)
        with col_d1:
            axes_html = "".join([
                f'<li><span class="term-badge">{"🟢 2점" if ax["score"]==2 else ("🟡 1점" if ax["score"]==1 else "🔴 0점")}</span> '
                f'<b>{ax["axis"]}</b> ({ax["weight"]}): {ax["status"]}</li>'
                for ax in earn_out["evaluation_axes"]
            ])
            st.markdown(f"""
            <div class="term-card">
                <div style="font-weight:700;margin-bottom:8px;">{utils.line_icon("layers", 16)}4장. 미래 지속성 5대 평가 축 점수표</div>
                <ul class="term-list">{axes_html}</ul>
            </div>
            """, unsafe_allow_html=True)

        with col_d2:
            qual_html = "".join([
                f'<li><span class="term-badge">{"🔴 경고" if q["warn"] else "🟢 정상"}</span> '
                f'<b>{q["name"]}:</b> {q["value"]} — <span style="color:#475569;">{q["desc"]}</span></li>'
                for q in qual["items"]
            ])
            st.markdown(f"""
            <div class="term-card">
                <div style="font-weight:700;margin-bottom:8px;">{utils.line_icon("shield", 16)}표 3 — E6. 이익의 질 및 건전성 5대 체크</div>
                <ul class="term-list">{qual_html}</ul>
            </div>
            """, unsafe_allow_html=True)

        col_e1, col_e2 = st.columns([1.2, 1.0])
        with col_e1:
            st.markdown(f'**{utils.line_icon("activity", 16)} 표 1 — E1. 실적 서프라이즈 및 성장률 궤적**', unsafe_allow_html=True)
            st.dataframe(pd.DataFrame(earn_out["e1_surprise_table"]), width="stretch", hide_index=True)

        with col_e2:
            squeeze_msg = "🔴 매출 상회 + 마진 하회(마진 압착) 감지됨" if qual["margin_squeeze_detected"] else "🟢 매출 상회 + 마진 하회(마진 압착) 없음"
            st.markdown(f"""
            <div class="term-card">
                <div style="font-weight:700;margin-bottom:6px;">{utils.line_icon("tag", 16)}E7·E8. 선행 파이프라인 & 시장의 이익 오판 포인트</div>
                <ul class="term-list">
                    <li><b>마진 압착 여부 (E1):</b> {squeeze_msg}</li>
                    <li><b>디자인 윈 / 파이프라인 (E7):</b> {sus["design_wins_note"]}</li>
                    <li><b>{th_drv["mispricing_type"]} (E8-2):</b> {th_drv["mispricing_detail"]}</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)


# =====================================================================
# [탭 3] 펀더멘탈 분석 트랙 v1.0[cite: 4]
# =====================================================================
with tab_t3:
    pq = fund_out["price_qualification"]
    msel = fund_out["multiple_selection"]
    tz = fund_out["triple_z_score"]
    ind_p = fund_out["industry_and_peer"]
    mdec = fund_out["multiple_decomposition"]
    rr = fund_out["rerating_assessment"]
    vt = fund_out["valuation_targets"]
    fsurv = fund_out["financial_survival"]
    mpts = fund_out["mispricing_points"]

    trap_badge = f'<span class="term-badge-inv">G4/G5 디레이팅 Cap 작동</span>' if pq["is_value_trap_blocked"] else ""
    cycle_trap_badge = f'<span class="term-badge-inv">사이클 정점 착시 방어 작동</span>' if pq["is_cycle_peak_trap"] else ""
    hold_cap_badge = f'<span class="term-badge-inv">보유기한 상한 1Y 경고</span>' if vt["holding_period_cap_warning"] else ""

    st.markdown(f"""
    <div class="term-banner">
        <div class="term-banner-line1">
            {utils.line_icon("layers", 18)}
            <span class="term-badge-inv">{pq["grade_icon"]} 가격 자격: {synth_out["v_action_card"]["price_qual_code"]}</span>
            <span class="term-badge">주 멀티플: {msel["primary_metric"]} ({msel["current_multiple"]:.1f}배)</span>
            <span class="term-badge">재평가 방향: {rr["direction"]}</span>
            {trap_badge}{cycle_trap_badge}{hold_cap_badge}
        </div>
        <div class="term-banner-line2">
            {pq["headline_reason"]} &nbsp;|&nbsp;
            확률가중 기대수익률 <b>{vt["expected_return_weighted"]*100:+.1f}%</b> (요구수익률 r={vt["required_return_r"]*100:.1f}%) ·
            <b>MDD Ratio {vt["mdd_ratio"]:.2f}</b> · Downside Convexity <b>{vt["downside_convexity"]:.2f}</b>
        </div>
        <div class="term-banner-line3">
            {utils.line_icon("target", 14)}<b>채택 매도목표가(1Y): {vt["sell_target_formula"]}</b>
            &nbsp;|&nbsp; <b>F3 3중 위치:</b> {tz["summary_text"]}
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_ch1, col_ch2 = st.columns(2)
    with col_ch1:
        p_dates = stock_df_raw.index[-250:]
        eff_eps_now = max(float(vt["effective_eps_1y"]), 1.0)
        hist_per_series = stock_df_raw["Close"].iloc[-250:] / eff_eps_now
        mu_s = tz["self_5y_mean"]
        sd_s = tz["self_5y_std"]

        fig_band = go.Figure()
        fig_band.add_trace(go.Scatter(
            x=p_dates, y=hist_per_series, mode="lines", name="내재 NTM PER 추이",
            line=dict(color="#111111", width=2.0)
        ))
        fig_band.add_hline(y=mu_s + 2*sd_s, line_dash="dot", line_color="#dc2626", annotation_text=f"+2σ ({mu_s+2*sd_s:.1f}x)")
        fig_band.add_hline(y=mu_s + 1*sd_s, line_dash="dash", line_color="#f97316", annotation_text=f"+1σ ({mu_s+1*sd_s:.1f}x)")
        fig_band.add_hline(y=mu_s, line_dash="solid", line_color="#475569", annotation_text=f"5Y 평균 μ ({mu_s:.1f}x)")
        fig_band.add_hline(y=mu_s - 1*sd_s, line_dash="dash", line_color="#16a34a", annotation_text=f"-1σ ({mu_s-1*sd_s:.1f}x)")
        fig_band.add_hline(
            y=vt["base_multiple_target"], line_dash="dashdot", line_color="#2563eb",
            annotation_text=f"F6 Base 목표배수 ({vt['base_multiple_target']:.1f}x)", annotation_position="bottom left"
        )
        fig_band.update_layout(
            title="📈 F3·F6. 역사적 멀티플 σ 밴드 및 Base 목표 배수 위치",
            height=380, margin=dict(l=20, r=20, t=40, b=20), template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
        )
        st.plotly_chart(fig_band, width="stretch")

    with col_ch2:
        eps_c = mdec["eps_contribution_12m"] * 100.0
        mkt_c = mdec["layer_market"] * 100.0
        ind_c = mdec["layer_industry"] * 100.0
        cmp_c = mdec["layer_company"] * 100.0
        tot_c = mdec["price_return_12m"] * 100.0

        fig_wf = go.Figure(go.Waterfall(
            name="3층 분해",
            orientation="v",
            measure=["relative", "relative", "relative", "relative", "total"],
            x=["① 이익(EPS) 몫", "② 시장층(금리·ERP)", "③ 산업층(업종배수)", "④ 기업층(개별프리미엄)", "12M 로그 주가수익률"],
            textposition="outside",
            text=[f"{eps_c:+.1f}%", f"{mkt_c:+.1f}%", f"{ind_c:+.1f}%", f"{cmp_c:+.1f}%", f"{tot_c:+.1f}%"],
            y=[eps_c, mkt_c, ind_c, cmp_c, tot_c],
            connector={"line": {"color": "#64748b"}},
            increasing={"marker": {"color": "#16a34a"}},
            decreasing={"marker": {"color": "#dc2626"}},
            totals={"marker": {"color": "#111111"}}
        ))
        fig_wf.update_layout(
            title=f"📊 F4. 최근 12M 주가 수익률 3층 로그 분해 ({mdec['dominant_driver'][:22]}...)",
            height=380, margin=dict(l=20, r=20, t=40, b=20), template="plotly_white"
        )
        st.plotly_chart(fig_wf, width="stretch")

    col_tr1, col_tr2 = st.columns([1.2, 1.0])
    with col_tr1:
        tr_rows_html = "".join([
            f'<li>'
            f'<span style="font-family:monospace;font-weight:700;display:inline-block;width:105px;">{tr["price"]:,}원</span> '
            f'<span class="term-badge">{"도달(체결권)" if last_close <= tr["price"] else "미도달(대기)"}</span> '
            f'<b>{tr["label"]}</b> — 목표비중 <b>{tr["weight"]*100:.2f}%</b>'
            f'</li>'
            for tr in vt["buy_tranches"]
        ])
        st.markdown(f"""
        <div class="term-card">
            <div style="font-weight:700;margin-bottom:8px;">
                {utils.line_icon("check", 16)}F7-3. 기술 트랙(V 라우터) 직결 0~3차 분할 매수기준선 (유효 1Y EPS: {vt["effective_eps_1y"]:,}원)
            </div>
            <ul class="term-list">{tr_rows_html}</ul>
            <div style="font-size:0.82rem;color:#555;margin-top:6px;">
                * 적용 규칙: {msel["rule_description"]}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_tr2:
        fa = rr["four_axes"]
        st.markdown(f"""
        <div class="term-card">
            <div style="font-weight:700;margin-bottom:8px;">
                {utils.line_icon("activity", 16)}F5. 5년 성숙기 수렴 내재 성장률(g_내재) 역산 & 재평가 4축 진단
            </div>
            <div style="font-size:1.05rem;font-weight:700;font-family:monospace;margin:6px 0;">
                현재가 내재 5Y CAGR: {rr["implied_5y_cagr"]*100:.1f}% &nbsp;vs&nbsp; 컨센서스 2Y CAGR: {rr["consensus_2y_cagr"]*100:.1f}%
            </div>
            <div style="font-size:0.84rem;color:#333;margin-bottom:8px;">
                종착점 성숙 배수(PER_종착): <b>{rr["terminal_per_used"]:.1f}배</b> (이론 성숙 PER {ind_p["theoretical_mature_per"]:.1f}배, 시장 k_e={ind_p["market_ke"]*100:.1f}%)
            </div>
            <div style="font-size:0.85rem;">
                <span class="term-badge">축1 지속기간: {fa["duration"]}</span>
                <span class="term-badge">축2 증분수익성: {fa["incremental_margin"]}</span>
                <span class="term-badge">축3 확실성: {fa["certainty"]}</span>
                <span class="term-badge">축4 선반영도: {fa["priced_in"]}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with st.expander("상세 펀더멘탈 밸류에이션 펼치기 — F8 매도 4대 앵커 · F2 피어 8대 요인 · F9 재무 생존성 · F10 멀티플 오판 포인트", expanded=True):
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            st.markdown(f'**{utils.line_icon("target", 16)} F8-1. 매도목표가 4대 독립 앵커 검증표**', unsafe_allow_html=True)
            st.dataframe(pd.DataFrame(vt["anchors_table"]), width="stretch", hide_index=True)

            sr = vt["scenario_returns"]
            cb_str = f" · CycleBear {sr['cycle_bear']*100:+.1f}%" if sr.get("cycle_bear") is not None else ""
            st.markdown(f"""
            <div class="term-card">
                <div style="font-weight:700;margin-bottom:6px;">{utils.line_icon("layers", 15)}F7-4 & F9. 시나리오 기대수익률 및 재무 생존성 3지표</div>
                <ul class="term-list">
                    <li><b>1Y 시나리오 수익률 (배당 포함):</b> Bull <b>{sr["bull"]*100:+.1f}%</b> · Base <b>{sr["base"]*100:+.1f}%</b> · Bear <b>{sr["bear"]*100:+.1f}%</b>{cb_str}</li>
                    <li><b>① FCF 소진 런웨이:</b> {fsurv["fcf_runway_years"]}</li>
                    <li><b>② 1년 내 차입 만기 부담 (현금 대비):</b> {fsurv["debt_maturity_1y_ratio"]*100:.0f}% {"(🔴 차환 경고)" if fsurv["debt_maturity_1y_ratio"]>1.0 else "(🟢 안정)"}</li>
                    <li><b>③ 연간 주식 희석률 (YoY):</b> {fsurv["share_dilution_yoy"]*100:.1f}% {"(🔴 희석 경고)" if fsurv["share_dilution_yoy"]>0.05 else "(🟢 적정)"}</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        with col_f2:
            st.markdown(
                f'**{utils.line_icon("shield", 16)} F2-2. 피어 대비 프리미엄·디스카운트 8대 요인 (순우위 점수: {ind_p["peer_net_advantage_score"]:+d}점 — {ind_p["peer_justification"]})**',
                unsafe_allow_html=True
            )
            st.dataframe(pd.DataFrame(ind_p["peer_factors_table"]), width="stretch", hide_index=True)

            st.markdown(f"""
            <div class="term-card">
                <div style="font-weight:700;margin-bottom:6px;">{utils.line_icon("tag", 15)}F10. 멀티플 측면 시장 오판 포인트 & F4 원인 분류</div>
                <ul class="term-list">
                    <li><b>F4 배수 변동 주원인:</b> {mdec["dominant_driver"]}</li>
                    <li><b>📈 멀티플 과소평가 포인트:</b> {mpts["undervalued_point"]}</li>
                    <li><b>📉 멀티플 과대평가 포인트:</b> {mpts["overvalued_point"]}</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)


# =====================================================================
# [탭 4] 거시적 분석 트랙 v1.0 (HTML 노출 및 차트 글자 겹침 완벽 해결 버전)[cite: 5]
# =====================================================================
with tab_t4:
    m_meta = macro_out["meta"]
    m_cls = macro_out["classification"]
    m_reg = macro_out["macro_regime"]
    m_sens = macro_out["sensitivities"]
    m_attr = macro_out["attribution_13w"]
    m_res = macro_out["residual_check"]
    m_state = macro_out["macro_state_13w"]
    m_cal = macro_out["watch_calendar"]

    warn_badges = "".join([f'<span class="term-badge-inv">{w}</span>' for w in m_cls.get("warnings", [])])

    # HTML 태그 노출 방지를 위해 st.markdown에 unsafe_allow_html=True 명시
    st.markdown(f"""
    <div class="term-banner">
        <div class="term-banner-line1">
            {utils.line_icon("compass", 18)}
            <span class="term-badge-inv">{m_cls["grade_icon"]} {m_cls["label"]}</span>
            <span class="term-badge">모형 B 고유 R² {m_cls["r2_macro_own"]*100:.1f}% (원시 R² {m_cls["r2_raw"]*100:.1f}%)</span>
            <span class="term-badge">OOS ΔR² {m_cls["oos_delta_r2_mean"]*100:+.1f}%p</span>
            <span class="term-badge">시장 β_m {m_cls["market_beta"]:.2f}</span>
            <span class="term-badge">국면: {m_reg["current_regime"]} (인플레 주 {int(m_reg["inflation_week_ratio_13w"]*100)}%)</span>
            {warn_badges}
        </div>
        <div class="term-banner-line2" style="margin-top:8px;">
            <b>거시 코멘트:</b> {macro_out["comment"]}
        </div>
        <div class="term-banner-line3" style="margin-top:6px;">
            {utils.line_icon("shield", 14)}<b>데이터 품질: {m_meta["data_quality"]} ({m_meta["sample_weeks"]}주 W-FRI)</b> &nbsp;|&nbsp;
            <b>사전 등록 가설 ID:</b> {m_meta["hypothesis_id"]} ({m_meta.get("industry_category", "")}) &nbsp;|&nbsp;
            <b>OOS 잔차 점검:</b> {m_res.get("status", "-")} (z_oos = {m_res.get("z_oos", 0.0):+.2f}) &nbsp;|&nbsp;
            <b>Track 3 전달 10Y 금리(R_f):</b> {m_reg["rf_10y_current"]*100:.2f}%
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_mc1, col_mc2 = st.columns(2)
    with col_mc1:
        if m_attr:
            w4 = m_attr["4w"]
            w13 = m_attr["13w"]
            fig_attr = go.Figure()
            fig_attr.add_trace(go.Bar(
                name="① 시장 몫", x=["최근 4주", "최근 13주"], y=[w4["market"]*100, w13["market"]*100], marker_color="#64748b",
                text=[f"{w4['market']*100:+.1f}%p", f"{w13['market']*100:+.1f}%p"], textposition="auto"
            ))
            fig_attr.add_trace(go.Bar(
                name="② 거시 추가 몫", x=["최근 4주", "최근 13주"], y=[w4["macro_extra"]*100, w13["macro_extra"]*100], marker_color="#2563eb",
                text=[f"{w4['macro_extra']*100:+.1f}%p", f"{w13['macro_extra']*100:+.1f}%p"], textposition="auto"
            ))
            fig_attr.add_trace(go.Bar(
                name="③ 기업 고유 몫", x=["최근 4주", "최근 13주"], y=[w4["idiosyncratic"]*100, w13["idiosyncratic"]*100], marker_color="#111111",
                text=[f"{w4['idiosyncratic']*100:+.1f}%p", f"{w13['idiosyncratic']*100:+.1f}%p"], textposition="auto"
            ))
            fig_attr.add_trace(go.Scatter(
                name="실제 합계", x=["최근 4주", "최근 13주"], y=[w4["actual"]*100, w13["actual"]*100],
                mode="markers+text", marker=dict(color="#dc2626", size=10, symbol="diamond"),
                text=[f"실제 {w4['actual']*100:+.1f}%", f"실제 {w13['actual']*100:+.1f}%"], textposition="top center"
            ))
            # 차트 타이틀과 범례 겹침 방지를 위해 상단 여백(t=65) 및 범례 위치 조정
            fig_attr.update_layout(
                barmode="relative", title="<b>6-1. 최근 4주·13주 수익률 기여 분해</b>",
                height=420, margin=dict(l=20, r=20, t=65, b=20), template="plotly_white",
                legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="left", x=0, font=dict(size=11))
            )
            st.plotly_chart(fig_attr, width="stretch")

    with col_mc2:
        if m_sens:
            v_names = [s["var_name"] for s in m_sens]
            v_resps = [s["price_response_1sd_num"] for s in m_sens]
            v_colors = ["#16a34a" if val >= 0 else "#dc2626" for val in v_resps]
            v_texts = [f"{s['price_response_1sd']} (t={s['t_stat_hac']:+.2f})" for s in m_sens]

            fig_sens = go.Figure(go.Bar(
                orientation="h", y=v_names, x=v_resps, marker_color=v_colors, text=v_texts, textposition="outside"
            ))
            fig_sens.add_vline(x=0, line_color="#111111", line_width=1.2)
            # 차트 타이틀 글자 겹침 방지 여백(t=65) 확보
            fig_sens.update_layout(
                title="<b>4-2·6-1. 가설 변수 1σ 변동 시 조건부 주가 반응</b>",
                xaxis_title="예상 주가 반응 (%)", height=420, margin=dict(l=20, r=60, t=65, b=20), template="plotly_white"
            )
            st.plotly_chart(fig_sens, width="stretch")

    col_ms1, col_ms2 = st.columns([1.25, 1.0])
    with col_ms1:
        sens_li = "".join([
            f'<li>'
            f'<span class="term-badge">{"🟢 확인" if s["status"]=="확인" else ("🟡 경고" if s["status"]=="경고" else "⚪ 미확인")}</span> '
            f'<b>{s["var_name"]}</b> (부호 {s["expected_sign"]}, 강도 {s["strength"]}) — '
            f'1σ({s["shock_1sd_str"]}) 변동 시 <b>{s["price_response_1sd"]}</b> (t={s["t_stat_hac"]:+.2f})<br>'
            f'<span style="font-size:0.82rem;color:#555;">국면별 β: {s["regime_beta_str"]} '
            f'{" | 중복: " + s["duplicate_of"] if s["duplicate_of"] else ""}</span>'
            f'</li>'
            for s in m_sens
        ])
        st.markdown(f"""
        <div class="term-card">
            <div style="font-weight:700;margin-bottom:8px;">
                {utils.line_icon("activity", 16)}4-2 & 6-0. 사전 등록 가설 변수 조건부 민감도 (위험 속성)
            </div>
            <ul class="term-list">{sens_li}</ul>
        </div>
        """, unsafe_allow_html=True)

    with col_ms2:
        cal_li = "".join([
            f'<li>'
            f'<b>{c["var_name"]}</b> — 차기 발표: <b>{c["next_release"]}</b><br>'
            f'<span style="font-size:0.83rem;color:#333;">경계 규칙: {c["threshold_rule"]}</span> '
            f'<span class="term-badge">{c["duplicate_tag"]}</span>'
            f'</li>'
            for c in m_cal
        ])
        st.markdown(f"""
        <div class="term-card">
            <div style="font-weight:700;margin-bottom:8px;">
                {utils.line_icon("compass", 16)}6-4. 거시 감시 캘린더 (중복 계상 방지 태그 병기)
            </div>
            <ul class="term-list">{cal_li}</ul>
        </div>
        """, unsafe_allow_html=True)

    with st.expander("상세 거시 진단 펼치기 — 5-1 가설 변수 13주 경로 · 6-2 표본 외(OOS) 잔차 점검", expanded=True):
        col_md1, col_md2 = st.columns([1.2, 1.0])
        with col_md1:
            st.markdown(f'**{utils.line_icon("layers", 16)} 5-1. 가설 변수 최근 13주 동일 구간 변화 및 4주 쏠림 몫**', unsafe_allow_html=True)
            st.dataframe(pd.DataFrame(m_state), width="stretch", hide_index=True)

        with col_md2:
            ev_weeks = m_res.get("event_weeks", [])
            ev_html = "".join([
                f'<li><b>{ev["date"]}</b> ({ev["type"]}): 표본 외 잔차 <b>{ev["residual_sigma"]:+.2f}σ</b> → 기업 고유 충격 분리</li>'
                for ev in ev_weeks
            ]) or "<li>최근 13주 내 분리된 기업 고유 이벤트 충격 주간 없음</li>"

            st.markdown(f"""
            <div class="term-card">
                <div style="font-weight:700;margin-bottom:6px;">
                    {utils.line_icon("shield", 16)}6-2. 최근 13주 표본 외(OOS) 잔차 점검 (Look-Ahead 차단)
                </div>
                <ul class="term-list">
                    <li><b>판정 결과:</b> <b>{m_res.get("status", "-")}</b> (누적 표준화 점수 z_oos = <b>{m_res.get("z_oos", 0.0):+.2f}</b>)</li>
                    <li><b>점검에 사용된 평범한 주:</b> {m_res.get("normal_weeks_count", 0)}주</li>
                    <li><b>분리된 기업 고유 이벤트 주간:</b></li>
                    {ev_html}
                </ul>
            </div>
            """, unsafe_allow_html=True)
