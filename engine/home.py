"""홈 대시보드: DATA_FRESHNESS·SAMPLE_DATA, 시장 헤더(build_regime_vix_header), run_home_dashboard_engine.

engine.py를 기능별로 나눈 모듈. 바깥에서는 `import engine` 후 `engine.이름`으로 쓴다(engine/__init__.py).
"""
import json
import math
import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

from .technical_signals import compute_market_regime

# =====================================================================
# [Home Dashboard v1.1 — Data Integrity & SSOT Edition]
# 단일 출처(SSOT), 기준일(as_of)/출처(source) 강제, 하드코딩 SAMPLE_DATA 분리 및 일관성 검증 엔진
# =====================================================================

# C-3 & E. 데이터 신선도 기준 및 리비전 임계값 단일 설정 상수
DATA_FRESHNESS: Dict[str, Any] = {
    "price_indicator_max_days": 3,     # 1거래일 (주말 감안 달력일 3일)
    "form4_max_days": 30,              # Form 4 내부자 공시: 30일
    "krx_flow_max_days": 3,            # KRX 수급: 1거래일 (주말 감안 3일)
    "catalyst_news_max_days": 14,      # D-2. 뉴스 촉매 후보 유효기간: 14일 이내
    "13f_always_badge": "분기 데이터",   # C-3. 13F는 항상 '분기 데이터' 회색 배지 부여
    "revision_neutral_band": 0.001,    # E. 리비전 보합 임계값 (0.1%p)
    "breadth_neutral_band": 1.0,       # B. 시장 폭 중립 임계값 (1.0%p)
    # 바스켓 실적 리비전 (yfinance 애널리스트 EPS 추정치, 2026-11-09 추가)
    "revision_max_days": 7,            # 리비전 데이터가 이보다 오래되면 판정에 쓰지 않음 (달력일)
    "revision_period": "+1y",          # 다음 회계연도 EPS 추정치 기준
    "revision_min_names": 2,           # 바스켓 값을 내기 위한 최소 종목 수
    "revision_min_coverage": 0.5,      # 대표 종목 중 추정치가 있어야 하는 비율
    "revision_min_abs_base_eps": 0.05, # 비교 기준 EPS 절댓값이 이보다 작으면(적자 근처) 비율 계산 제외
}

# C-5 & D-3. 0단계에서 식별된 하드코딩 예시 데이터 보관 상수 (삭제 금지 -> SAMPLE_DATA 이관)
SAMPLE_DATA: Dict[str, Any] = {
    "narrative_catalysts": {
        "SMH": "빅테크 AI 인프라 CAPEX 가이던스 상향 및 차세대 패키징 수주 확대",
        "IGV": "AI 에이전트 상용화 과금 모델 가시화로 RPO 증가율 반등",
        "GRID": "북미 데이터센터 전력망 병목 심화 및 변압기·가스터빈 장기 수주 호조",
        "ITA": "글로벌 국방비 증액 기조 및 상업용 항공기 애프터마켓 마진 개선",
        "IBB": "금리 안정기 임상 3상 파이프라인 M&A 기대감 유입",
        "IBIT": "현물 ETF 기관 자금 순유입 지속 및 글로벌 순유동성 확장 동조",
        "XLK": "엔터프라이즈 서버 및 온디바이스 AI 교체 수요 견조",
        "XLC": "AI 타겟팅 고도화에 따른 디지털 광고 단가(CPM) 및 영업이익률 상승",
        "XLF": "장단기 금리차 정상화 및 기업 M&A·IPO 투자은행(IB) 수수료 회복",
        "XLV": "비만치료제 수요 견조세와 관리의료 손해율(MLR) 안정화 확인 대기",
        "XLY": "물류 자동화 마진 개선과 고금리 누적에 따른 내구재 소비 둔화 혼재",
        "XLI": "미국 내 제조 공장 증설 및 인프라 법안 예산 집행 지속",
        "XLE": "글로벌 원유 수요 둔화 우려 속 주주환원(배당·자사주)으로 하방 방어",
        "XLU": "AI 데이터센터 장기 전력구매계약(PPA) 체결로 유틸리티 성장성 재평가",
        "XLB": "구리 등 전력망 필수 광물 수요는 견조하나 화학 스프레드 회복 지연",
        "XLP": "가격 인상을 통한 인플레이션 전가 마무리 및 방어적 배당 수요",
        "XLRE": "데이터센터 리츠 임대료 상승 vs 상업용 오피스 공실 부담 교차",
    },
    "basket_revisions_30d_90d": {
        "SMH": (0.018, 0.045),
        "IGV": (0.009, -0.012),
        "GRID": (0.015, 0.032),
        "ITA": (0.011, 0.028),
        "IBB": (0.0002, 0.0004),
        "XLK": (0.014, 0.036),
        "XLC": (0.016, 0.041),
        "XLF": (0.008, -0.009),
        "XLV": (-0.0003, 0.0002),
        "XLY": (-0.008, 0.015),
        "XLI": (0.007, 0.019),
        "XLE": (-0.014, -0.031),
        "XLU": (0.009, -0.006),
        "XLB": (-0.011, -0.022),
        "XLP": (0.0003, 0.0005),
        "XLRE": (0.0001, -0.0002),
    },
    "kr_inst_foreign_top3": [
        {"sector": "반도체·장비", "net_buy_20d_krw": "+1조 4,200억", "top_pick": "SK하이닉스, 한미반도체", "as_of": None, "published": None, "source": "SAMPLE_DATA", "is_sample": True},
        {"sector": "방산·우주항공", "net_buy_20d_krw": "+4,850억", "top_pick": "한화에어로스페이스, 현대로템", "as_of": None, "published": None, "source": "SAMPLE_DATA", "is_sample": True},
        {"sector": "전력기기·원전", "net_buy_20d_krw": "+3,640억", "top_pick": "HD현대일렉트릭, 두산에너빌리티", "as_of": None, "published": None, "source": "SAMPLE_DATA", "is_sample": True},
    ],
    "micro_smart_signals": {
        "stealth_accumulation": [
            {"ticker": "PANW", "name": "팔로알토 네트웍스", "detail": "최근 10일 내 RVOL≥2.0 기관 매집 양봉 2회 · 20일선 이격 +3.2%", "as_of": None, "published": None, "source": "SAMPLE_DATA", "kind": "price", "is_sample": True},
            {"ticker": "CRWD", "name": "크라우드스트라이크", "detail": "최근 10일 내 RVOL≥2.0 기관 매집 양봉 3회 · 돌파 상단 수축", "as_of": None, "published": None, "source": "SAMPLE_DATA", "kind": "price", "is_sample": True},
            {"ticker": "267260.KS", "name": "HD현대일렉트릭", "detail": "조정 구간 거래량 마름(RVOL 0.6) 후 기관 대량 양봉 출현", "as_of": None, "published": None, "source": "SAMPLE_DATA", "kind": "price", "is_sample": True},
        ],
        "insider_buying": [
            {"ticker": "UNH", "name": "유나이티드헬스 (UNH)", "detail": "CFO 및 사외이사 2명 장내 순수 매수 ($4.2M)", "as_of": None, "published": None, "source": "Form 4 (SAMPLE_DATA)", "kind": "form4", "is_sample": True},
            {"ticker": "005930.KS", "name": "삼성전자", "detail": "주요 임원진 장내 자사주 매입 공시", "as_of": None, "published": None, "source": "DART (SAMPLE_DATA)", "kind": "form4", "is_sample": True},
        ],
        "hedge_fund_13f_consensus": [
            {"ticker": "AVGO", "name": "브로드컴 (AVGO)", "detail": "Duquesne(드러켄밀러), Coatue, Appaloosa 동시 비중 확대(+35%↑)", "as_of": None, "published": None, "source": "SEC 13F (SAMPLE_DATA)", "kind": "13f", "is_sample": True},
            {"ticker": "VST", "name": "비스트라 에너지 (VST)", "detail": "Third Point, Baupost 신규 코어 편입 (AI 전력 인프라)", "as_of": None, "published": None, "source": "SEC 13F (SAMPLE_DATA)", "kind": "13f", "is_sample": True},
            {"ticker": "MSFT", "name": "마이크로소프트 (MSFT)", "detail": "선별 10대 그랜드마스터 펀드 중 7곳 Top 5 코어 보유 유지", "as_of": None, "published": None, "source": "SEC 13F (SAMPLE_DATA)", "kind": "13f", "is_sample": True},
        ],
    },
    "opportunity_radar": {
        "col1_action_flips": [
            {
                "ticker": "AVGO", "name": "브로드컴 (Broadcom)",
                "flip_badge": "🔥 최고 확신 점화 (ARCH_1)", "narrative_synergy": "⚡ [SMH] AI·반도체 주도 내러티브 동조",
                "prev_state": "V 가격 대기 / T 눌림 관찰", "curr_state": "🟢 V 0차 즉시 매수(100%) + 🟢 T ST-3(추세 눌림) A등급 진입",
                "action_line": "진입 $172.50 · T 손절 $164.00 · 1차목표 $195.00 (예상 2.6R)",
                "high_conviction": True, "as_of": None, "published": None, "source": "SAMPLE_DATA", "is_sample": True,
            },
            {
                "ticker": "005930.KS", "name": "삼성전자",
                "flip_badge": "🟢 T 반등 신규 점화 + 🟡 V 1차 절반", "narrative_synergy": "⚡ [반도체] 과매도 투매 클라이맥스 통과",
                "prev_state": "STATE_D 하락 진행 (투매 대기)", "curr_state": "🟢 T ST-1(BNF 반등) A등급 진입 / 🟡 V 1차 50% 집행(BNF 예외)",
                "action_line": "진입 54,200원 · T 손절 50,800원 · 1차목표 59,800원 (예상 1.65R)",
                "high_conviction": False, "as_of": None, "published": None, "source": "SAMPLE_DATA", "is_sample": True,
            },
        ],
        "col2_approaching": [
            {
                "ticker": "MSFT", "name": "마이크로소프트 (Microsoft)",
                "approach_badge": "🎯 V 1차 매수선 코앞 (-2.1%)", "narrative_synergy": "🌱 [IGV] 소프트웨어 차기 점화 내러티브 동조",
                "current_price": "$423.80", "target_trigger": "1차 매수기준선($415.00) 이하 도달 시 V 자격(QUALIFIED) 개방 · MDD Ratio 0.38",
                "gap_pct": 0.021, "as_of": None, "published": None, "source": "SAMPLE_DATA", "is_sample": True,
            },
            {
                "ticker": "CRWD", "name": "크라우드스트라이크 (CrowdStrike)",
                "approach_badge": "🎯 T 돌파 임박 (추세돌파 3단계 · T_READY)", "narrative_synergy": "🌱 [IGV] 소프트웨어 차기 점화 내러티브 동조",
                "current_price": "$318.50", "target_trigger": "박스 상단($324.00)을 RVOL ≥ 1.5 양봉 종가로 돌파 시 ST-2a 1차 50% 즉시 진입",
                "gap_pct": 0.017, "as_of": None, "published": None, "source": "SAMPLE_DATA", "is_sample": True,
            },
        ],
        "col3_risk_alerts": [
            {
                "ticker": "NVDA", "name": "엔비디아 (NVIDIA)",
                "alert_badge": "⏰ 실적 블랙아웃 진입 (D-2)", "severity": "🟡",
                "guide_line": "차기 실적 발표 D-2: 전 셋업(ST-1~4) 신규 T 진입 제한 / 기보유 T 로트는 +1R 이상 시 본전 손절 상향 후 통과",
                "as_of": None, "published": None, "source": "SAMPLE_DATA", "is_sample": True,
            },
            {
                "ticker": "000660.KS", "name": "SK하이닉스",
                "alert_badge": "🔵 V 목표가 도달 (차익실현 구간)", "severity": "🔵",
                "guide_line": "펀더멘탈 매도목표가(215,000원) -5% 이내 진입: V 분할 익절 실행 및 자전거래 방지를 위해 신규 T 매수 차단",
                "as_of": None, "published": None, "source": "SAMPLE_DATA", "is_sample": True,
            },
        ],
    },
}

# A-1. 바스켓 메타데이터에 asset_class 필드 추가 ("equity_sector" | "equity_theme" | "crypto" | "bond" | "country")
# 고정 문자열("4주 ETF 순유입 가속" 등) 및 하드코딩 catalyst 완전 제거
NARRATIVE_BASKETS_SPEC: List[Dict[str, Any]] = [
    # 나스닥 핵심 테마 (5) -> asset_class: "equity_theme"
    {"ticker": "SMH", "name": "AI · 반도체", "category": "나스닥 핵심 테마", "asset_class": "equity_theme", "proxies": ["NVDA", "AVGO", "TSM", "AMD", "MU"]},
    {"ticker": "IGV", "name": "소프트웨어 · SaaS · 보안", "category": "나스닥 핵심 테마", "asset_class": "equity_theme", "proxies": ["MSFT", "CRM", "NOW", "PANW", "CRWD"]},
    {"ticker": "GRID", "name": "AI 데이터센터 전력 · 그리드", "category": "나스닥 핵심 테마", "asset_class": "equity_theme", "proxies": ["ETN", "GEV", "VST", "CEG", "PWR"]},
    {"ticker": "ITA", "name": "우주 · 항공 · 방산", "category": "나스닥 핵심 테마", "asset_class": "equity_theme", "proxies": ["RTX", "LMT", "GE", "NOC", "AXON"]},
    {"ticker": "IBB", "name": "바이오테크 · 신약", "category": "나스닥 핵심 테마", "asset_class": "equity_theme", "proxies": ["AMGN", "VRTX", "GILD", "REGN"]},
    # 크립토 (1) -> asset_class: "crypto" (EPS 없음 -> 플로우 또는 수익률 방향 사용)
    {"ticker": "IBIT", "name": "크립토 · 디지털 유동성", "category": "크립토", "asset_class": "crypto", "proxies": ["IBIT", "MSTR", "COIN"]},
    # S&P 500 기본 섹터 (11) -> asset_class: "equity_sector"
    {"ticker": "XLK", "name": "테크 · 하드웨어", "category": "S&P 500 섹터", "asset_class": "equity_sector", "proxies": ["AAPL", "MSFT", "NVDA"]},
    {"ticker": "XLC", "name": "빅테크 플랫폼 · 미디어", "category": "S&P 500 섹터", "asset_class": "equity_sector", "proxies": ["META", "GOOGL", "NFLX"]},
    {"ticker": "XLF", "name": "금융 · 은행 · 자본시장", "category": "S&P 500 섹터", "asset_class": "equity_sector", "proxies": ["JPM", "BRK-B", "GS", "V"]},
    {"ticker": "XLV", "name": "헬스케어 · 빅파마 · 관리의료", "category": "S&P 500 섹터", "asset_class": "equity_sector", "proxies": ["LLY", "UNH", "JNJ", "ABBV"]},
    {"ticker": "XLY", "name": "임의소비재 · 이커머스", "category": "S&P 500 섹터", "asset_class": "equity_sector", "proxies": ["AMZN", "TSLA", "HD"]},
    {"ticker": "XLI", "name": "산업재 · 기계 · 리쇼어링", "category": "S&P 500 섹터", "asset_class": "equity_sector", "proxies": ["CAT", "UNP", "HON"]},
    {"ticker": "XLE", "name": "전통 에너지 · 오일·가스", "category": "S&P 500 섹터", "asset_class": "equity_sector", "proxies": ["XOM", "CVX", "EOG"]},
    {"ticker": "XLU", "name": "유틸리티 · 독립발전(IPP)", "category": "S&P 500 섹터", "asset_class": "equity_sector", "proxies": ["NEE", "SO", "DUK"]},
    {"ticker": "XLB", "name": "소재 · 화학 · 광산", "category": "S&P 500 섹터", "asset_class": "equity_sector", "proxies": ["LIN", "SHW", "FCX"]},
    {"ticker": "XLP", "name": "필수소비재 (방어)", "category": "S&P 500 섹터", "asset_class": "equity_sector", "proxies": ["PG", "COST", "WMT"]},
    {"ticker": "XLRE", "name": "리츠 · 데이터센터 부동산", "category": "S&P 500 섹터", "asset_class": "equity_sector", "proxies": ["PLD", "AMT", "EQIX"]},
]


# =====================================================================
# [복구] 20일 CMF 및 상승/하락 거래대금 비율 계산
# (__pycache__/engine.cpython-314.pyc 에 남아 있던 원본 바이트코드를 해독해 복원)
# =====================================================================
def _compute_cmf20_and_ud_ratio(df):
    """
    2-2. 20일 차이킨 자금흐름(CMF_20) 및 상승일/하락일 거래대금 비율(UD_Ratio_20) 계산:
    - MFM_t = ((C_t - L_t) - (H_t - C_t)) / (H_t - L_t)
    - CMF_20 = sum(MFM * Vol, 20) / sum(Vol, 20)
    """
    if df is None or len(df) < 22:
        return (0.05, 1.1)
    sub = df.iloc[-20:].copy()
    c = pd.to_numeric(sub["Close"], errors="coerce")
    o = pd.to_numeric(sub["Open"], errors="coerce")
    h = pd.to_numeric(sub["High"], errors="coerce")
    lo = pd.to_numeric(sub["Low"], errors="coerce")
    v = pd.to_numeric(sub["Volume"], errors="coerce").fillna(0.0)

    hl_range = (h - lo).replace(0, np.nan)
    mfm = (((c - lo) - (h - c)) / hl_range).fillna(0.0)
    vol_sum = float(v.sum())
    cmf_20 = float((mfm * v).sum() / vol_sum) if vol_sum > 0 else 0.0

    tr_val = c * v
    up_val = float(tr_val[c >= o].sum())
    dn_val = float(tr_val[c < o].sum())
    ud_ratio = up_val / max(dn_val, 1.0) if dn_val > 0 else 1.5
    return (round(cmf_20, 3), round(min(max(ud_ratio, 0.3), 3.5), 2))


# =====================================================================
# 1단계 핵심 순수 함수군 (A, B, C, D, E)
# =====================================================================
def classify_breadth(
    spread_20d: float,
    bench_ret_20d: float,
    neutral_band: float = 1.0
) -> Tuple[str, str]:
    """
    B. 시장 폭(Breadth) 해석 순수 함수:
    - 입력 단위: %p (예: +7.4%p -> 7.4, -12.7% -> -12.7)
    - 반환값: (라벨, 설명 한 줄)
    """
    if abs(spread_20d) < neutral_band:
        return (
            "중립 (폭 신호 없음)",
            f"20일 스프레드({spread_20d:+.1f}%p)가 중립 범위(±{neutral_band:.1f}%p) 내에 위치함"
        )
    if spread_20d > 0:
        if bench_ret_20d >= 0:
            return (
                "온기 확산 (Broadening Out)",
                f"벤치마크 상승({bench_ret_20d:+.1f}%) 속 동일/중소형 상대강도 우위({spread_20d:+.1f}%p)로 시장 온기 확산"
            )
        else:
            return (
                "방어적 로테이션 (대형주 주도 하락)",
                f"벤치마크 하락({bench_ret_20d:+.1f}%) 속 동일/중소형 상대 우위({spread_20d:+.1f}%p)는 온기 확산이 아닌 대형주 주도 하락임"
            )
    else:
        if bench_ret_20d >= 0:
            return (
                "소수 대형주 집중 (Narrow Leadership)",
                f"벤치마크 상승({bench_ret_20d:+.1f}%)이 소수 시총 상위 종목에 집중됨 (동일/중소형 상대 열위 {spread_20d:+.1f}%p)"
            )
        else:
            return (
                "광범위 하락 (중소·동일가중 더 약세)",
                f"벤치마크 하락({bench_ret_20d:+.1f}%)과 함께 중소·동일가중이 더 크게 하락({spread_20d:+.1f}%p)하는 전반적 약세"
            )


def revision_state(
    rev_30d: Optional[float],
    rev_90d: Optional[float],
    neutral_band: Optional[float] = None
) -> Dict[str, str]:
    """
    E. 리비전 배지 문구 및 상태 판정 단일 순수 함수:
    - 상향 전환: 90일 <= 0 이고 30일 > 0
    - 상향 지속: 둘 다 > 0
    - 하향 전환: 90일 >= 0 이고 30일 < 0
    - 하향 지속: 둘 다 < 0
    - 보합: 그 외 / 임계값 이내
    """
    if neutral_band is None:
        neutral_band = float(DATA_FRESHNESS["revision_neutral_band"])

    if rev_30d is None or rev_90d is None or (np.isnan(rev_30d) or np.isnan(rev_90d)):
        return {"state": "데이터 없음", "badge": "EPS 데이터 없음", "direction": "none"}

    if abs(rev_30d) <= neutral_band and abs(rev_90d) <= neutral_band:
        state = "보합"
        arrow = "➡"
        direction = "flat"
    elif rev_90d <= 0.0 and rev_30d > neutral_band:
        state = "상향 전환"
        arrow = "⬆"
        direction = "up"
    elif rev_90d > 0.0 and rev_30d > neutral_band:
        state = "상향 지속"
        arrow = "⬆"
        direction = "up"
    elif rev_90d >= 0.0 and rev_30d < -neutral_band:
        state = "하향 전환"
        arrow = "⬇"
        direction = "down"
    elif rev_90d < 0.0 and rev_30d < -neutral_band:
        state = "하향 지속"
        arrow = "⬇"
        direction = "down"
    else:
        state = "보합"
        arrow = "➡"
        direction = "flat"

    return {
        "state": state,
        "badge": f"EPS {arrow} {state}",
        "direction": direction,
    }



def summarize_basket_revisions(eps_trends: Dict[str, Any], as_of: Optional[str]) -> Dict[str, Dict[str, Any]]:
    """
    바스켓 대표 종목(NARRATIVE_BASKETS_SPEC의 proxies)의 EPS 추정치 추이로 바스켓 실적 리비전을 계산한다.
    네트워크를 쓰지 않는 순수 함수 — eps_trends는 {종목: yfinance Ticker.eps_trend 표(없으면 None)}.
    - 종목별: revision_period(기본 '+1y' = 다음 회계연도) EPS 추정치의
      30일 변화율 = (현재 - 30일 전) / |30일 전|, 90일 변화율 = (현재 - 90일 전) / |90일 전|
    - 비교 기준 EPS의 절댓값이 revision_min_abs_base_eps 미만(적자 근처)이면 비율이 의미 없어 제외
    - 바스켓 값 = 계산된 종목들의 중앙값(한 종목이 크게 튀어도 덜 흔들림).
      종목 수가 revision_min_names 이상이고 대표 종목의 revision_min_coverage 이상일 때만 값을 내고, 아니면 None.
    - 크립토 바스켓은 EPS가 없어 계산하지 않는다(엔진이 수익률·플로우로 대체).
    """
    cfg = DATA_FRESHNESS
    period = cfg["revision_period"]
    min_abs = float(cfg["revision_min_abs_base_eps"])

    def _one(df_t):
        try:
            if df_t is None or len(df_t) == 0 or period not in df_t.index:
                return None
            row = df_t.loc[period]
            cur, d30, d90 = (float(pd.to_numeric(row.get(k), errors="coerce")) for k in ("current", "30daysAgo", "90daysAgo"))
            if any(np.isnan(v) for v in (cur, d30, d90)) or abs(d30) < min_abs or abs(d90) < min_abs:
                return None
            return (cur - d30) / abs(d30), (cur - d90) / abs(d90)
        except Exception:
            return None

    out: Dict[str, Dict[str, Any]] = {}
    for spec in NARRATIVE_BASKETS_SPEC:
        if spec.get("asset_class") == "crypto":
            continue
        tk, proxies = spec["ticker"], spec["proxies"]
        used = {p: v for p, v in ((p, _one((eps_trends or {}).get(p))) for p in proxies) if v is not None}
        need = max(int(cfg["revision_min_names"]), int(math.ceil(len(proxies) * float(cfg["revision_min_coverage"]))))
        rec: Dict[str, Any] = {
            "n_used": len(used), "n_total": len(proxies), "tickers_used": sorted(used),
            "period": period, "as_of": as_of, "source": "yfinance eps_trend (애널리스트 컨센서스)",
        }
        if len(used) >= need:
            rec["rev_30d"] = round(float(np.median([v[0] for v in used.values()])), 5)
            rec["rev_90d"] = round(float(np.median([v[1] for v in used.values()])), 5)
            rec["reason"] = None
        else:
            rec["rev_30d"] = rec["rev_90d"] = None
            rec["reason"] = f"추정치가 있는 대표 종목 {len(used)}/{len(proxies)}개 ({need}개 필요)"
        out[tk] = rec
    return out


def fetch_basket_revisions(pause_sec: float = 0.3) -> Dict[str, Dict[str, Any]]:
    """
    yfinance에서 바스켓 대표 종목들의 EPS 추정치 추이를 받아 summarize_basket_revisions로 계산한다.
    매일 GitHub Actions의 global_pipeline.py가 호출해 latest_global_report.json의 "basket_revisions"에 저장한다.
    as_of = 미국 동부 시간 기준 조회 날짜(컨센서스 스냅샷의 날짜). 파이프라인은 미국 장 마감 뒤에 돈다.
    """
    import time
    import yfinance as yf
    tickers = sorted({p for s in NARRATIVE_BASKETS_SPEC if s.get("asset_class") != "crypto" for p in s["proxies"]})
    trends: Dict[str, Any] = {}
    for t in tickers:
        try:
            trends[t] = yf.Ticker(t).eps_trend
        except Exception:
            trends[t] = None
        time.sleep(pause_sec)
    as_of = pd.Timestamp.now(tz="America/New_York").strftime("%Y-%m-%d")
    return summarize_basket_revisions(trends, as_of)


def etf_net_flow(
    ticker: str,
    days: int = 20,
    price_df: Optional[pd.DataFrame] = None
) -> Optional[Dict[str, Any]]:
    """
    A-2. yfinance Ticker.get_shares_full()의 발행 주식 수 변화 × 기간 평균 가격으로
    순유입 근사치를 계산하는 단일 함수. 데이터가 없으면 None을 반환한다.
    """
    try:
        import yfinance as yf
        tk_obj = yf.Ticker(ticker)
        start_dt = (datetime.now() - pd.Timedelta(days=int(days * 2.2))).strftime("%Y-%m-%d")
        shares_s = tk_obj.get_shares_full(start=start_dt)
        if shares_s is None or len(shares_s.dropna()) < 2:
            return None
        shares_s = shares_s.dropna()
        sh_start = float(shares_s.iloc[0])
        sh_end = float(shares_s.iloc[-1])
        if sh_start <= 0:
            return None
        delta_shares = sh_end - sh_start
        pct_flow = delta_shares / sh_start
        # 발행좌수 변화가 전혀 없으면(yfinance 미갱신 정적 값) 유효 플로우 미확보(None) 처리
        if abs(delta_shares) < 1.0:
            return None

        avg_price = 50.0
        if price_df is not None and len(price_df) > 0 and "Close" in price_df:
            avg_price = float(pd.to_numeric(price_df["Close"].iloc[-days:], errors="coerce").mean())

        net_flow_usd = delta_shares * avg_price
        as_of_str = str(shares_s.index[-1])[:10]
        direction = "up" if net_flow_usd > 0 else ("down" if net_flow_usd < 0 else "flat")
        state_lbl = "4주 순유입" if net_flow_usd > 0 else ("4주 순유출" if net_flow_usd < 0 else "4주 보합")
        arrow = "⬆" if net_flow_usd > 0 else ("⬇" if net_flow_usd < 0 else "➡")

        return {
            "net_flow_usd": round(net_flow_usd, 0),
            "pct_flow": round(pct_flow, 4),
            "direction": direction,
            "state": state_lbl,
            "badge": f"플로우 {arrow} {state_lbl}",
            "as_of": as_of_str,
            "source": f"yfinance ({ticker} get_shares_full)",
        }
    except Exception:
        return None


def compute_asset_return_and_flow_ssot(
    ticker: str,
    df_a: Optional[pd.DataFrame],
    asset_class: str = "equity_theme"
) -> Dict[str, Any]:
    """
    A-2, A-3, A-4. 단일 출처(SSOT) 가격 수익률 및 자금 플로우 계산 함수:
    - 모듈 1(내러티브 카드)과 모듈 3(자금 물길)이 반드시 이 함수의 동일한 결과를 공유한다.
    - 가격 수익률은 '4주 수익률 +X.X%'로 명확히 라벨링하고, 실제 플로우(etf_net_flow)가 있을 때만 '플로우 순유입/순유출'을 표기한다.
    """
    if df_a is not None and len(df_a) >= 22 and "Close" in df_a:
        c_s = pd.to_numeric(df_a["Close"], errors="coerce").dropna()
        if len(c_s) >= 21:
            r4 = round(float(c_s.iloc[-1] / c_s.iloc[-21] - 1.0), 4)
            r13 = round(float(c_s.iloc[-1] / c_s.iloc[-min(len(c_s), 61)] - 1.0), 4)
            as_of_date = str(c_s.index[-1])[:10]
            source_str = f"yfinance ({ticker})"
            is_sample = False
        else:
            r4, r13, as_of_date, source_str, is_sample = None, None, None, "데이터 없음", False
    else:
        r4, r13, as_of_date, source_str, is_sample = None, None, None, "데이터 없음", False

    flow_res = etf_net_flow(ticker, days=20, price_df=df_a) if asset_class == "crypto" else None

    if r4 is None:
        ret_label = "데이터 없음"
        ret_dir = "none"
    else:
        ret_dir = "up" if r4 > 0 else ("down" if r4 < 0 else "flat")
        ret_arrow = "🟢" if r4 > 0 else ("🔴" if r4 < 0 else "⚪")
        # A-2. 가격 수익률일 때는 '자금 유출/유입' 대신 '4주 수익률'로 정확히 표기
        ret_label = f"{ret_arrow} 4주 수익률 {r4*100:+.1f}%"

    # A-3. 크립토(asset_class == "crypto")는 EPS 리비전 대신 etf_net_flow 결과를 사용하며,
    # 발행좌수 데이터가 없으면(None) 동일 SSOT 4주 수익률 방향/수치를 플로우 프록시로 사용하여 모듈 1·3 모순을 원천 차단함.
    if asset_class == "crypto":
        if flow_res is not None:
            crypto_badge = flow_res["badge"]
            crypto_state = flow_res["state"]
            crypto_dir = flow_res["direction"]
            crypto_note = "크립토 자산은 EPS가 없으므로 발행좌수 기반 4주 ETF 순유입/유출(etf_net_flow)로 이익 조건을 대체함"
        elif r4 is not None:
            arrow = "⬆" if r4 > 0 else ("⬇" if r4 < 0 else "➡")
            state_txt = f"4주 수익률 {r4*100:+.1f}% (플로우 미확보)"
            crypto_badge = f"플로우(수익률 대체) {arrow} {state_txt}"
            crypto_state = state_txt
            crypto_dir = ret_dir
            crypto_note = "크립토 자산은 EPS가 없으며 ETF 발행좌수 시계열 미확보로 동일 출처 4주 수익률 방향으로 대체함"
        else:
            crypto_badge = "플로우 데이터 없음"
            crypto_state = "데이터 없음"
            crypto_dir = "none"
            crypto_note = "크립토 플로우 및 가격 데이터 없음"
    else:
        crypto_badge, crypto_state, crypto_dir, crypto_note = "", "", "none", ""

    return {
        "ticker": ticker,
        "ret_4w": r4,
        "ret_13w": r13,
        "ret_direction": ret_dir,
        "return_badge": ret_label,
        "net_flow": flow_res,
        "crypto_badge": crypto_badge,
        "crypto_state": crypto_state,
        "crypto_direction": crypto_dir,
        "crypto_substitution_note": crypto_note,
        "as_of": as_of_date,
        "source": source_str,
        "is_sample": is_sample,
    }


def evaluate_record_freshness(
    as_of: Optional[str],
    kind: str = "price",
    reference_date: Optional[str] = None,
    is_sample: bool = False
) -> Dict[str, Any]:
    """
    C-3. 데이터 종류별 신선도 기준 판정 함수:
    - 가격·지표(price) / KRX 수급(krx_flow): 1거래일 (달력 3일)
    - Form 4(form4): 30일
    - 13F(13f): 항상 '분기 데이터' 회색 배지
    - 기준 초과 시 '지연 데이터' 회색 배지, SAMPLE_DATA는 '예시 데이터' 배지 부여
    """
    badges: List[str] = []
    if is_sample:
        badges.append("예시 데이터")
    if kind == "13f":
        badges.append(DATA_FRESHNESS["13f_always_badge"])

    if not as_of:
        badges.append("기준일 없음")
        return {"is_stale": True, "age_days": None, "badges": badges}

    try:
        ref_dt = datetime.strptime(reference_date[:10], "%Y-%m-%d") if reference_date else datetime.now()
        as_of_dt = datetime.strptime(str(as_of)[:10], "%Y-%m-%d")
        age_days = max(0, (ref_dt - as_of_dt).days)
    except Exception:
        badges.append("기준일 형식 오류")
        return {"is_stale": True, "age_days": None, "badges": badges}

    max_days_map = {
        "price": DATA_FRESHNESS["price_indicator_max_days"],
        "krx_flow": DATA_FRESHNESS["krx_flow_max_days"],
        "form4": DATA_FRESHNESS["form4_max_days"],
        "revision": DATA_FRESHNESS["revision_max_days"],
        "13f": 135,  # 분기 데이터(분기말 후 45일 공시 + 90일 유효)
    }
    limit_days = max_days_map.get(kind, 3)
    is_stale = age_days > limit_days
    if is_stale:
        badges.append("지연 데이터")

    return {
        "is_stale": is_stale,
        "age_days": age_days,
        "badges": badges,
    }


def match_verified_catalyst_news(
    basket_ticker: str,
    proxies: List[str],
    news_list: List[Dict[str, Any]],
    reference_date: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    D-2 & D-3. 뉴스 JSON(latest_global_report.json 등)에서 해당 바스켓 티커나 핵심 종목에 매칭되고,
    14일 이내이며, 출처 링크(url/link)가 있는 항목만 반환한다.
    형식: '촉매 후보(검증 전) · {날짜} · {매체}' + 링크. 매칭 항목이 없으면 None 반환(숨김 처리).
    """
    if not news_list:
        return None
    ref_dt = datetime.strptime(reference_date[:10], "%Y-%m-%d") if reference_date else datetime.now()
    target_syms = {basket_ticker.upper()} | {p.upper() for p in proxies}

    for item in news_list:
        url = str(item.get("url") or item.get("link") or "").strip()
        pub_date = str(item.get("published") or item.get("date") or item.get("as_of") or "")[:10]
        source_media = str(item.get("source") or item.get("publisher") or "").strip()
        title = str(item.get("title") or item.get("headline") or "").strip()
        related = [str(x).upper() for x in item.get("tickers", [])]
        if item.get("ticker"):
            related.append(str(item["ticker"]).upper())

        if not url or not pub_date or not source_media or not title:
            continue
        if not (target_syms.intersection(related) or any(sym in title.upper() for sym in target_syms)):
            continue

        try:
            p_dt = datetime.strptime(pub_date, "%Y-%m-%d")
            days_diff = (ref_dt - p_dt).days
            if 0 <= days_diff <= DATA_FRESHNESS["catalyst_news_max_days"]:
                return {
                    "label": f"촉매 후보(검증 전) · {pub_date} · {source_media}",
                    "title": title,
                    "url": url,
                    "as_of": pub_date,
                    "published": pub_date,
                    "source": source_media,
                }
        except Exception:
            continue
    return None


def _classify_narrative_grade(quadrant: str, rev_state: str, rs_accel: float) -> str:
    """
    1-4 & E. 내러티브 등급 판정:
    - 판정에 쓰는 상태(rev_state)와 화면 배지에 보이는 문구 상태가 100% 동일하도록 연결.
    - 크립토의 경우 플로우/수익률 방향 상태('4주 순유입', '4주 순유출' 등)를 동일하게 매핑(주석 및 툴팁 명시).
    """
    is_positive = rev_state in ["상향 지속", "상향 전환", "4주 순유입"] or (
        rev_state.startswith("4주 수익률 +")
    )
    is_negative = rev_state in ["하향 지속", "하향 전환", "4주 순유출"] or (
        rev_state.startswith("4주 수익률 -")
    )

    if quadrant == "LEADING":
        if is_positive:
            return "🔥 실물 확인 주도"
        return "🫧 과열 · 기대 선행"
    elif quadrant == "WEAKENING":
        if not is_positive:
            return "🫧 과열 · 기대 선행"
        return "🟡 주도권 숨고르기"
    elif quadrant == "IMPROVING":
        if is_positive:
            return "🌱 차기 내러티브 점화"
        elif is_negative:
            return "🧊 단기 순환매 반등"
        return "⚪ 소외 · 바닥 탐색"
    else:  # LAGGING
        if rs_accel >= 0.80 and is_positive:
            return "🌱 차기 내러티브 점화"
        return "⚪ 소외 · 바닥 탐색"


# =====================================================================
# 2단계 — 일관성 점검 장치: run_consistency_checks()
# =====================================================================
def run_consistency_checks(dashboard_data: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    2단계 일관성 점검 장치:
    1) 같은 티커의 플로우·수익률 방향이 모듈 간에 다름
    2) as_of가 없는 레코드
    3) 신선도 기준 초과
    4) SAMPLE_DATA 사용 중
    5) 배지 문구와 판정 상태 불일치
    """
    violations: List[Dict[str, str]] = []

    baskets = dashboard_data.get("narrative_rotation", {}).get("baskets", [])
    cross_flows = dashboard_data.get("smart_money_tracker", {}).get("cross_asset_flow", [])

    # 1) 같은 티커(예: IBIT)의 모듈 1(내러티브) vs 모듈 3(자금 물길) 방향 일치 점검
    basket_by_tk = {b["ticker"]: b for b in baskets}
    for cf in cross_flows:
        tk = cf.get("ticker")
        if tk and tk in basket_by_tk:
            b_item = basket_by_tk[tk]
            dir_m1 = b_item.get("fundamental_or_flow_direction")
            dir_m3 = cf.get("ret_direction")
            if dir_m1 in ["up", "down"] and dir_m3 in ["up", "down"] and dir_m1 != dir_m3:
                violations.append({
                    "type": "DIRECTION_MISMATCH",
                    "target": tk,
                    "detail": f"{tk}의 내러티브 방향({dir_m1})과 모듈 3 자산 방향({dir_m3})이 서로 다릅니다."
                })

    # 0) 실제 가격이 부족해 계산하지 못한 바스켓 (예시 좌표로 대신 채우지 않음)
    for mb_item in dashboard_data.get("narrative_rotation", {}).get("missing_baskets", []):
        violations.append({
            "type": "DATA_MISSING",
            "target": f"바스켓:{mb_item.get('ticker')}",
            "detail": f"{mb_item.get('ticker')} 섹터 제외됨 — {mb_item.get('reason', '원인 미상')}"
        })

    # 0-1) 실적 리비전을 판정에 쓰지 못한 바스켓
    for ri in dashboard_data.get("narrative_rotation", {}).get("revision_issues", []):
        violations.append({
            "type": ri.get("type", "DATA_MISSING"),
            "target": f"바스켓리비전:{ri.get('ticker')}",
            "detail": f"{ri.get('ticker')} 실적 리비전 — {ri.get('reason', '원인 미상')}"
        })

    # 2) & 5) 바스켓별 배지 문구-판정 상태 일치, 크립토 EPS 배지 금지, as_of/SAMPLE_DATA 점검
    for b in baskets:
        tk = b["ticker"]
        badge_txt = str(b.get("eps_revision_status", ""))
        state_txt = str(b.get("revision_or_flow_state", ""))

        if b.get("asset_class") == "crypto" and "EPS" in badge_txt:
            violations.append({
                "type": "CRYPTO_EPS_BADGE",
                "target": tk,
                "detail": f"크립토 바스켓({tk})에 EPS 배지('{badge_txt}')가 생성되었습니다."
            })

        if state_txt and state_txt not in badge_txt:
            violations.append({
                "type": "BADGE_STATE_MISMATCH",
                "target": tk,
                "detail": f"{tk}의 판정 상태('{state_txt}')와 배지 문구('{badge_txt}')가 일치하지 않습니다."
            })

        if not b.get("as_of"):
            violations.append({
                "type": "MISSING_AS_OF",
                "target": f"바스켓:{tk}",
                "detail": f"내러티브 바스켓 {tk}의 기준일(as_of)이 없습니다."
            })
        if b.get("is_sample_revision"):
            violations.append({
                "type": "SAMPLE_DATA_USED",
                "target": f"바스켓리비전:{tk}",
                "detail": f"{tk}의 리비전 수치가 SAMPLE_DATA(예시 데이터)에서 로드되었습니다."
            })

    # 3) & 4) 스마트머니 및 기회 포착 레이더 레코드의 as_of, 신선도 초과, SAMPLE_DATA 점검
    sm = dashboard_data.get("smart_money_tracker", {})
    radar = dashboard_data.get("opportunity_radar", {})

    all_records: List[Tuple[str, Dict[str, Any]]] = []
    for c in cross_flows:
        all_records.append((f"모듈3_자산물길:{c.get('ticker', c.get('asset'))}", c))
    for k in sm.get("kr_inst_foreign_top3", []):
        all_records.append((f"모듈3_KRX수급:{k.get('sector')}", k))
    for grp_name, grp_list in sm.get("micro_smart_signals", {}).items():
        for rec in grp_list:
            all_records.append((f"모듈3_{grp_name}:{rec.get('ticker')}", rec))
    for col_name in ["col1_action_flips", "col2_approaching", "col3_risk_alerts"]:
        for rec in radar.get(col_name, []):
            all_records.append((f"모듈4_{col_name}:{rec.get('ticker')}", rec))

    for label_id, rec in all_records:
        if rec.get("is_sample") or "SAMPLE_DATA" in str(rec.get("source", "")):
            violations.append({
                "type": "SAMPLE_DATA_USED",
                "target": label_id,
                "detail": f"{label_id} 항목이 SAMPLE_DATA(예시 데이터)를 사용 중입니다."
            })
        # 예시 데이터는 규칙상 as_of=None이므로 '기준일 없음'으로 중복 경고하지 않는다
        is_sample_rec = rec.get("is_sample") or "SAMPLE_DATA" in str(rec.get("source", ""))
        if not rec.get("as_of") and not is_sample_rec:
            violations.append({
                "type": "MISSING_AS_OF",
                "target": label_id,
                "detail": f"{label_id} 레코드에 데이터 기준일(as_of)이 없습니다."
            })
        fresh_info = rec.get("freshness", {})
        if fresh_info.get("is_stale") and rec.get("as_of"):
            violations.append({
                "type": "STALE_DATA",
                "target": label_id,
                "detail": f"{label_id} 레코드(기준일 {rec.get('as_of')})가 신선도 기준을 초과했습니다."
            })

    return violations


def build_regime_vix_header(qqq_df, spy_df, ks11_df, vix_df) -> Dict[str, Any]:
    """
    지수 레짐(QQQ·SPY·KOSPI) + VIX + 거시 국면 문구 계산 — 국면 게이트 입력의 단일 출처.
    홈 헤더(run_home_dashboard_engine)와 게이트 과거 재계산(build_gate_history_state)이 함께 쓴다.
    """
    def _as_of(df_x):
        return str(df_x.index[-1])[:10] if (df_x is not None and len(df_x) > 0) else None

    vix_s = vix_df["Close"] if (vix_df is not None and "Close" in vix_df) else None
    qqq_reg = compute_market_regime(qqq_df, vix_s)
    spy_reg = compute_market_regime(spy_df, vix_s)
    kospi_reg = compute_market_regime(ks11_df, vix_s)

    vix_val = float(vix_s.dropna().iloc[-1]) if (vix_s is not None and len(vix_s.dropna()) > 0) else None
    vix_pctl = int(round(spy_reg.get("vix_pctl", 50.0), 0))
    vix_label = spy_reg.get("s_vix", " 데이터 없음") if vix_val is not None else "데이터 없음"

    if qqq_reg["s_reg"] in ["🟢", "🟡"] and (vix_val is not None and vix_pctl < 65):
        macro_narrative = "골디락스·연착륙 (실질금리 안정 · 신용 스프레드 축소)"
    elif vix_val is not None and vix_pctl >= 80:
        macro_narrative = "침체·신용 경색 경계 (변동성 80%ile 이상 급등)"
    else:
        macro_narrative = "금리 민감 · 실적 차별화 국면"

    return {
        "qqq_regime": {"icon": qqq_reg["s_reg"], "label": qqq_reg["s_reg_label"], "as_of": _as_of(qqq_df), "source": "yfinance (QQQ)"},
        "spy_regime": {"icon": spy_reg["s_reg"], "label": spy_reg["s_reg_label"], "as_of": _as_of(spy_df), "source": "yfinance (SPY)"},
        "kospi_regime": {"icon": kospi_reg["s_reg"], "label": kospi_reg["s_reg_label"], "as_of": _as_of(ks11_df), "source": "yfinance (^KS11)"},
        "vix_state": {"value": round(vix_val, 1) if vix_val is not None else None, "percentile_1y": vix_pctl, "label": vix_label, "as_of": _as_of(vix_df), "source": "yfinance (^VIX)"},
        "macro_narrative": macro_narrative,
    }


# =====================================================================
# 홈 대시보드 메인 실행 함수: run_home_dashboard_engine()
# =====================================================================
def run_home_dashboard_engine(
    etf_data_map: Optional[Dict[str, pd.DataFrame]] = None,
    stock_data_map: Optional[Dict[str, pd.DataFrame]] = None,
    report_json: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    홈 화면(app.py) 통합 연산 엔진 (SSOT · 기준일/출처 강제 · 모순 제거 버전)
    """
    if etf_data_map is None:
        etf_data_map = {}
    if stock_data_map is None:
        stock_data_map = {}
    if report_json is None:
        if os.path.exists("latest_global_report.json"):
            try:
                with open("latest_global_report.json", "r", encoding="utf-8") as f:
                    report_json = json.load(f)
            except Exception:
                report_json = {}
        else:
            report_json = {}

    news_items = report_json.get("news", []) if isinstance(report_json, dict) else []
    report_revisions = (report_json.get("basket_revisions") or {}) if isinstance(report_json, dict) else {}
    # 리포트에 basket_revisions 키가 있으면 실제 출처가 연결된 것 → 빠진 바스켓을 예시 데이터로 채우지 않는다
    has_real_rev_source = isinstance(report_json, dict) and "basket_revisions" in report_json

    spy_df = etf_data_map.get("SPY")
    qqq_df = etf_data_map.get("QQQ")
    ks11_df = etf_data_map.get("^KS11")
    vix_df = etf_data_map.get("^VIX")
    rsp_df = etf_data_map.get("RSP")
    iwm_df = etf_data_map.get("IWM")
    tlt_df = etf_data_map.get("TLT")

    def _extract_as_of(df_x: Optional[pd.DataFrame]) -> Optional[str]:
        if df_x is not None and len(df_x) > 0:
            return str(df_x.index[-1])[:10]
        return None

    latest_mkt_as_of = _extract_as_of(spy_df) or _extract_as_of(qqq_df) or datetime.now().strftime("%Y-%m-%d")

    # -----------------------------------------------------------------
    # 1. 상단 글로벌 매크로 중력 & 시장 폭(Breadth) 헤더 연산 (B 항목 반영)
    # -----------------------------------------------------------------
    # 지수 레짐·VIX·거시 국면은 build_regime_vix_header 한 곳에서만 계산 (게이트 과거 재계산과 공유)
    regime_vix_hdr = build_regime_vix_header(qqq_df, spy_df, ks11_df, vix_df)

    def _calc_20d_ret_pct(df_x: Optional[pd.DataFrame]) -> Optional[float]:
        if df_x is None or len(df_x) < 22 or "Close" in df_x is False:
            return None
        c_s = pd.to_numeric(df_x["Close"], errors="coerce").dropna()
        if len(c_s) < 21:
            return None
        return round(float((c_s.iloc[-1] / c_s.iloc[-21] - 1.0) * 100.0), 2)

    def _pair_20d_spread_pct(num_df: Optional[pd.DataFrame], den_df: Optional[pd.DataFrame]) -> Optional[float]:
        if num_df is None or den_df is None or len(num_df) < 22 or len(den_df) < 22:
            return None
        common = num_df.index.intersection(den_df.index)
        if len(common) < 21:
            return None
        ratio = (num_df.loc[common, "Close"] / den_df.loc[common, "Close"].replace(0, np.nan)).dropna()
        if len(ratio) < 21:
            return None
        return round(float((ratio.iloc[-1] / ratio.iloc[-21] - 1.0) * 100.0), 2)

    spy_ret_20d_pct = _calc_20d_ret_pct(spy_df)
    qqq_ret_20d_pct = _calc_20d_ret_pct(qqq_df)
    rsp_spy_spread_pct = _pair_20d_spread_pct(rsp_df, spy_df)
    iwm_qqq_spread_pct = _pair_20d_spread_pct(iwm_df, qqq_df)

    # B. classify_breadth 순수 함수로만 시장 폭 진단 생성
    nb = float(DATA_FRESHNESS["breadth_neutral_band"])
    if rsp_spy_spread_pct is not None and spy_ret_20d_pct is not None:
        rsp_lbl, rsp_desc = classify_breadth(rsp_spy_spread_pct, spy_ret_20d_pct, neutral_band=nb)
    else:
        rsp_lbl, rsp_desc = ("데이터 없음", "RSP 또는 SPY 20일 시계열 데이터 없음")

    if iwm_qqq_spread_pct is not None and qqq_ret_20d_pct is not None:
        iwm_lbl, iwm_desc = classify_breadth(iwm_qqq_spread_pct, qqq_ret_20d_pct, neutral_band=nb)
    else:
        iwm_lbl, iwm_desc = ("데이터 없음", "IWM 또는 QQQ 20일 시계열 데이터 없음")

    breadth_headline = f"S&P 500 폭: {rsp_lbl} ({rsp_desc}) ｜ 나스닥/소형주 폭: {iwm_lbl}"

    # -----------------------------------------------------------------
    # 2. SSOT 자산별 수익률/플로우 사전 계산 (모듈 1과 모듈 3이 공유)
    # -----------------------------------------------------------------
    ssot_asset_metrics: Dict[str, Dict[str, Any]] = {}
    for sym, a_cls in [("QQQ", "equity_theme"), ("IBIT", "crypto"), ("TLT", "bond"), ("^KS11", "country")]:
        ssot_asset_metrics[sym] = compute_asset_return_and_flow_ssot(sym, etf_data_map.get(sym), asset_class=a_cls)

    # -----------------------------------------------------------------
    # 3. 17대 내러티브 바스켓 연산 (A, D, E 반영)
    # -----------------------------------------------------------------
    baskets_out: List[Dict[str, Any]] = []
    # 실제 가격이 부족해 계산하지 못한 바스켓(예시 좌표로 대신 채우지 않고 목록만 남긴다)
    missing_baskets: List[Dict[str, str]] = []
    # 실적 리비전을 판정에 쓰지 못한 바스켓(데이터 없음·오래됨)
    revision_issues: List[Dict[str, str]] = []
    for spec in NARRATIVE_BASKETS_SPEC:
        tk = spec["ticker"]
        a_class = spec["asset_class"]
        b_df = etf_data_map.get(tk)
        b_as_of = _extract_as_of(b_df)
        is_sample_price = False

        if b_df is not None and spy_df is not None and len(b_df) >= 75 and len(spy_df) >= 75:
            common = b_df.index.intersection(spy_df.index)
            if len(common) >= 75:
                rs = (b_df.loc[common, "Close"] / spy_df.loc[common, "Close"].replace(0, np.nan)).dropna()
                rs_sma50 = rs.rolling(50, min_periods=25).mean()
                raw_ratio_s = ((rs / rs_sma50) - 1.0) * 100.0
                raw_mom_s = ((rs / rs.shift(20)) - 1.0) * 100.0

                rx = round(float(raw_ratio_s.iloc[-1]), 2)
                ry = round(float(raw_mom_s.iloc[-1]), 2)
                r_acc = round(float(raw_mom_s.iloc[-1] - raw_mom_s.iloc[-6]), 2)
                trail_4w = [
                    (round(float(raw_ratio_s.iloc[-16]), 2), round(float(raw_mom_s.iloc[-16]), 2)),
                    (round(float(raw_ratio_s.iloc[-11]), 2), round(float(raw_mom_s.iloc[-11]), 2)),
                    (round(float(raw_ratio_s.iloc[-6]), 2), round(float(raw_mom_s.iloc[-6]), 2)),
                    (rx, ry),
                ]
                cmf_20, ud_ratio = _compute_cmf20_and_ud_ratio(b_df)
                b_as_of = str(common[-1])[:10]
            else:
                missing_baskets.append({"ticker": tk, "reason": f"SPY와 겹치는 가격 {len(common)}일 (75일 필요)"})
                continue
        else:
            _n_b = len(b_df) if b_df is not None else 0
            missing_baskets.append({"ticker": tk, "reason": f"가격 데이터 부족 ({_n_b}일, 75일 필요)" if b_df is not None else "가격 데이터 없음"})
            continue

        if rx >= 0 and ry >= 0:
            quad = "LEADING"
        elif rx >= 0 and ry < 0:
            quad = "WEAKENING"
        elif rx < 0 and ry >= 0:
            quad = "IMPROVING"
        else:
            quad = "LAGGING"

        # A-3 & E. 크립토 vs 주식 바스켓 배지 및 상태 단일 산출
        is_sample_rev = False
        if a_class == "crypto":
            # 크립토는 EPS가 없으므로 모듈 3과 동일한 SSOT 함수(compute_asset_return_and_flow_ssot) 결과를 공유함
            ibit_ssot = ssot_asset_metrics.get(tk) or compute_asset_return_and_flow_ssot(tk, b_df, asset_class="crypto")
            ssot_asset_metrics[tk] = ibit_ssot
            badge_str = ibit_ssot["crypto_badge"]
            state_str = ibit_ssot["crypto_state"]
            dir_str = ibit_ssot["crypto_direction"]
            tooltip_note = ibit_ssot["crypto_substitution_note"]
        else:
            rev_rec = report_revisions.get(tk) if has_real_rev_source else None
            if has_real_rev_source:
                r30 = rev_rec.get("rev_30d") if rev_rec else None
                r90 = rev_rec.get("rev_90d") if rev_rec else None
                rev_as_of = rev_rec.get("as_of") if rev_rec else None
                if r30 is None or r90 is None:
                    revision_issues.append({"ticker": tk, "type": "DATA_MISSING",
                                            "reason": (rev_rec or {}).get("reason") or "리포트에 이 바스켓의 리비전 없음"})
                    r30, r90 = None, None
                elif evaluate_record_freshness(rev_as_of, kind="revision", reference_date=latest_mkt_as_of)["is_stale"]:
                    revision_issues.append({"ticker": tk, "type": "STALE_DATA",
                                            "reason": f"기준일 {rev_as_of or '없음'} — {DATA_FRESHNESS['revision_max_days']}일 초과로 판정에서 제외 (git pull로 최신 리포트 받기)"})
                    r30, r90 = None, None
                else:
                    r30, r90 = float(r30), float(r90)
            elif tk in SAMPLE_DATA["basket_revisions_30d_90d"]:
                r30, r90 = SAMPLE_DATA["basket_revisions_30d_90d"][tk]
                is_sample_rev = True
            else:
                r30, r90 = None, None
            rev_info = revision_state(r30, r90)
            badge_str = rev_info["badge"] + (" [예시]" if is_sample_rev else "")
            state_str = rev_info["state"]
            dir_str = rev_info["direction"]
            if r30 is None:
                tooltip_note = "리비전 데이터 없음"
            elif is_sample_rev:
                tooltip_note = f"30일 리비전 {r30*100:+.2f}%, 90일 리비전 {r90*100:+.2f}% (SAMPLE_DATA)"
            else:
                tooltip_note = (f"30일 리비전 {r30*100:+.2f}%, 90일 리비전 {r90*100:+.2f}% "
                                f"(yfinance 컨센서스 · 대표 종목 {rev_rec.get('n_used')}/{rev_rec.get('n_total')}개 중앙값 · 조회일 {rev_rec.get('as_of')})")

        n_grade = _classify_narrative_grade(quad, state_str, r_acc)

        # D-2 & D-3. 14일 이내 + 출처 링크가 확인된 뉴스만 촉매 후보(검증 전)로 매칭 (없으면 None -> 숨김)
        verified_cat = match_verified_catalyst_news(tk, spec["proxies"], news_items, reference_date=latest_mkt_as_of)

        baskets_out.append({
            "ticker": tk,
            "name": spec["name"],
            "category": spec["category"],
            "asset_class": a_class,
            "proxies": ", ".join(spec["proxies"][:4]),
            "rs_ratio_x": rx,
            "rs_mom_y": ry,
            "rs_accel": r_acc,
            "trail_4w": trail_4w,
            "quadrant": quad,
            "eps_revision_status": badge_str,          # 화면 배지 문자열
            "revision_or_flow_state": state_str,       # 등급 판정에 실제 사용된 상태값 (일관성 검증용)
            "fundamental_or_flow_direction": dir_str,  # 모듈 3과 방향 비교용 ("up" | "down" | "flat" | "none")
            "substitution_tooltip": tooltip_note,
            "narrative_grade": n_grade,
            "cmf_20": cmf_20,
            "ud_vol_ratio": ud_ratio,
            "verified_catalyst": verified_cat,         # None이면 UI에서 촉매 줄 자체를 숨김
            "as_of": b_as_of,
            "source": f"yfinance ({tk})" if not is_sample_price else "SAMPLE_DATA",
            "is_sample_price": is_sample_price,
            "is_sample_revision": is_sample_rev,
        })

    leaders_sorted = sorted(
        [b for b in baskets_out if b["quadrant"] == "LEADING"],
        key=lambda x: (int("🔥" in x["narrative_grade"]), x["rs_mom_y"] + x["rs_ratio_x"]),
        reverse=True
    )
    if len(leaders_sorted) < 3:
        leaders_sorted = sorted(baskets_out, key=lambda x: x["rs_ratio_x"] + x["rs_mom_y"], reverse=True)

    next_rot_sorted = sorted(
        [b for b in baskets_out if b["quadrant"] == "IMPROVING" or "🌱" in b["narrative_grade"]],
        key=lambda x: (int("🌱" in x["narrative_grade"]), x["rs_accel"], x["rs_mom_y"]),
        reverse=True
    )
    if len(next_rot_sorted) < 3:
        next_rot_sorted = sorted([b for b in baskets_out if b["quadrant"] != "LEADING"], key=lambda x: x["rs_accel"], reverse=True)

    # -----------------------------------------------------------------
    # 4. [모듈 3] 3층 자금 흐름 & 스마트머니 트래커 (A-2, A-4, C 반영)
    # -----------------------------------------------------------------
    cross_specs = [
        ("QQQ", "미국 성장주 (QQQ)", "equity_theme"),
        ("IBIT", "크립토 하이베타 (IBIT)", "crypto"),
        ("TLT", "미국 장기채 (TLT)", "bond"),
        ("^KS11", "한국 증시 (KOSPI)", "country"),
    ]
    cross_asset_flow = []
    for sym, label_name, a_cls in cross_specs:
        m_ssot = ssot_asset_metrics.get(sym) or compute_asset_return_and_flow_ssot(sym, etf_data_map.get(sym), asset_class=a_cls)
        fresh = evaluate_record_freshness(m_ssot["as_of"], kind="price", reference_date=latest_mkt_as_of, is_sample=m_ssot["is_sample"])
        # 플로우 데이터가 있으면 플로우 병기, 없으면 '4주 수익률'로만 표기
        if m_ssot["net_flow"] is not None:
            status_badge = f"{m_ssot['net_flow']['badge']} ({m_ssot['return_badge']})"
        else:
            status_badge = m_ssot["return_badge"]

        cross_asset_flow.append({
            "ticker": sym,
            "asset": label_name,
            "asset_class": a_cls,
            "ret_4w": m_ssot["ret_4w"],
            "ret_13w": m_ssot["ret_13w"],
            "ret_direction": m_ssot["ret_direction"],
            "flow_status": status_badge,
            "as_of": m_ssot["as_of"],
            "published": m_ssot["as_of"],
            "source": m_ssot["source"],
            "is_sample": m_ssot["is_sample"],
            "freshness": fresh,
        })

    qqq_r4 = ssot_asset_metrics["QQQ"]["ret_4w"]
    tlt_r4 = ssot_asset_metrics["TLT"]["ret_4w"]
    if qqq_r4 is not None and tlt_r4 is not None:
        if qqq_r4 > 0 and tlt_r4 < 0:
            cross_summary = f"4주 가격 수익률 기준 미국 성장주(QQQ {qqq_r4*100:+.1f}%) 강세 · 장기채(TLT {tlt_r4*100:+.1f}%) 약세 (기준일: {latest_mkt_as_of})"
        elif qqq_r4 < 0 and tlt_r4 > 0:
            cross_summary = f"4주 가격 수익률 기준 미국 성장주(QQQ {qqq_r4*100:+.1f}%) 약세 · 장기채(TLT {tlt_r4*100:+.1f}%) 강세 (기준일: {latest_mkt_as_of})"
        else:
            cross_summary = f"4주 가격 수익률 기준 QQQ {qqq_r4*100:+.1f}% · TLT {tlt_r4*100:+.1f}% 동조 등락 (기준일: {latest_mkt_as_of})"
    else:
        cross_summary = "데이터 없음"

    us_inflow_top4 = sorted(baskets_out, key=lambda x: (x["cmf_20"], x["ud_vol_ratio"]), reverse=True)[:4]

    # C-1 ~ C-5. KRX 수급, 스텔스 매집봉, 내부자(Form 4), 13F 레코드에 as_of, published, source, 신선도 배지 부여
    raw_kr_flow = report_json.get("kr_inst_foreign_top3") or SAMPLE_DATA["kr_inst_foreign_top3"]
    kr_flow_out = []
    for item in raw_kr_flow:
        rec = dict(item)
        rec["freshness"] = evaluate_record_freshness(
            rec.get("as_of"), kind="krx_flow", reference_date=latest_mkt_as_of, is_sample=bool(rec.get("is_sample"))
        )
        kr_flow_out.append(rec)

    raw_micro = report_json.get("micro_smart_signals") or SAMPLE_DATA["micro_smart_signals"]
    micro_out: Dict[str, List[Dict[str, Any]]] = {}
    for grp_key, kind_code in [("stealth_accumulation", "price"), ("insider_buying", "form4"), ("hedge_fund_13f_consensus", "13f")]:
        grp_items = []
        for item in raw_micro.get(grp_key, []):
            rec = dict(item)
            rec["freshness"] = evaluate_record_freshness(
                rec.get("as_of"), kind=kind_code, reference_date=latest_mkt_as_of, is_sample=bool(rec.get("is_sample"))
            )
            grp_items.append(rec)
        micro_out[grp_key] = grp_items

    # -----------------------------------------------------------------
    # 5. (삭제 2026-11-09) 옛 예시 기회 레이더 — 실행 레이더는 app.py의 build_radar_signals(실제 T 신호)로 대체
    # -----------------------------------------------------------------

    dashboard_payload = {
        "meta": {
            "as_of": latest_mkt_as_of,
            "source": "yfinance & latest_global_report.json",
            "universe_count": 35,
        },
        "macro_header": {
            **regime_vix_hdr,
            "market_breadth": {
                "rsp_spy_20d_chg": round(rsp_spy_spread_pct / 100.0, 4) if rsp_spy_spread_pct is not None else None,
                "iwm_qqq_20d_chg": round(iwm_qqq_spread_pct / 100.0, 4) if iwm_qqq_spread_pct is not None else None,
                "spy_ret_20d_pct": spy_ret_20d_pct,
                "qqq_ret_20d_pct": qqq_ret_20d_pct,
                "rsp_spy_label": rsp_lbl,
                "iwm_qqq_label": iwm_lbl,
                "label": breadth_headline,
                "as_of": _extract_as_of(rsp_df) or latest_mkt_as_of,
                "source": "yfinance (RSP/SPY, IWM/QQQ)",
            },
        },
        "narrative_rotation": {
            "baskets": baskets_out,
            "top_current_leaders": leaders_sorted[:3],
            "top_next_rotations": next_rot_sorted[:3],
            "missing_baskets": missing_baskets,
            "revision_issues": revision_issues,
        },
        "smart_money_tracker": {
            "cross_asset_flow": cross_asset_flow,
            "cross_asset_summary": cross_summary,
            "us_sector_inflow_top4": us_inflow_top4,
            "kr_inst_foreign_top3": kr_flow_out,
            "micro_smart_signals": micro_out,
        },
    }

    # 2단계 일관성 점검 결과 첨부
    dashboard_payload["consistency_violations"] = run_consistency_checks(dashboard_payload)
    return dashboard_payload

# [단일 출처(SSOT) SAMPLE_DATA 안전 연동]
if "SAMPLE_DATA" not in globals():
    SAMPLE_DATA = {}

SAMPLE_DATA.setdefault("macro_gravity", {
    "US": {
        "regime": "DEEP_BEAR", "vix_percentile": 95.0, "vix_value": 36.2,
        "macro_phase": "CREDIT_CRUNCH", "as_of": "2026-11-08", "source": "CBOE / FRED"
    },
    "KR": {
        "regime": "CORRECTION", "vix_percentile": 78.0, "vix_value": 26.1,
        "macro_phase": "SLOWDOWN", "as_of": "2026-11-08", "source": "KRX / ECOS"
    }
})

SAMPLE_DATA.setdefault("radar_signals", {
    "col1_transitions": [
        {"ticker": "NVDA", "market": "US", "signal_type": "T_ENTRY", "tranche_stage": 0, "action_text": "T ST-3 A등급 진입", "suggested_weight": 100, "is_bnf_oversold": False, "note": "20일선 돌파 거래량 2.1배", "as_of": "2026-11-08", "source": "Track 1/3 Synthesis"},
        {"ticker": "005930", "market": "KR", "signal_type": "V_BUY", "tranche_stage": 0, "action_text": "V 0차 즉시 매수", "suggested_weight": 100, "is_bnf_oversold": False, "note": "PBR 밴드 최하단 지지", "as_of": "2026-11-08", "source": "Track 1/3 Synthesis"},
        {"ticker": "000660", "market": "KR", "signal_type": "V_BUY", "tranche_stage": 1, "action_text": "V 1차 분할 매수", "suggested_weight": 50, "is_bnf_oversold": False, "note": "1차 매수선 도달 (이월 주문)", "as_of": "2026-11-08", "source": "Track 3 Valuation"}
    ],
    "col2_watch": [
        {"ticker": "TSLA", "market": "US", "signal_type": "T_WATCH", "tranche_stage": 0, "action_text": "ST-1 셋업 사정권 (0.8% 내)", "suggested_weight": 50, "is_bnf_oversold": False, "note": "변동성 수축 후 상단 근접", "as_of": "2026-11-08", "source": "Track 1 Engine"},
        {"ticker": "IBIT", "market": "CRYPTO", "signal_type": "T_WATCH", "tranche_stage": 0, "action_text": "ST-2 눌림목 대기 (1.1% 내)", "suggested_weight": 50, "is_bnf_oversold": False, "note": "ETF 발행좌수 순유입 유지", "as_of": "2026-11-08", "source": "Track 1 Engine"}
    ],
    "col3_risk_and_special": [
        {"ticker": "AAPL", "market": "US", "signal_type": "RISK_DEFENSE", "tranche_stage": 0, "action_text": "손절 및 비중 50% 축소", "suggested_weight": 100, "is_bnf_oversold": False, "note": "50일선 이탈 및 S-REG 악화", "as_of": "2026-11-08", "source": "Track 1 Risk Gate"},
        {"ticker": "INTC", "market": "US", "signal_type": "V_BUY", "tranche_stage": 0, "action_text": "극단 투매 BNF 매수", "suggested_weight": 100, "is_bnf_oversold": True, "note": "F10 적정가 괴리 -38% 과매도", "as_of": "2026-11-08", "source": "Track 3 Valuation"},
        {"ticker": "SCHD", "market": "US", "signal_type": "V_BUY", "tranche_stage": 0, "action_text": "정기 코어 적립 매수", "suggested_weight": 100, "is_bnf_oversold": False, "note": "배당 코어 패시브 적립", "as_of": "2026-11-08", "source": "Core Portfolio Rule"}
    ]
})
