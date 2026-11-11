"""공통 파라미터(TECH_PARAMS·레짐 허용 매트릭스)와 가격 표기·호가·실적 발표일 유틸, NO_DATA.

engine.py를 기능별로 나눈 모듈. 바깥에서는 `import engine` 후 `engine.이름`으로 쓴다(engine/__init__.py).
"""
import math
import os
from typing import Dict, Optional, Tuple, Any
import numpy as np
import pandas as pd

# 프로젝트 폴더(app.py·.env가 있는 곳). engine/ 폴더의 한 단계 위.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# =====================================================================
# 11. 파라미터 표 (TP-01 ~ TP-37 확정값)
# =====================================================================
TECH_PARAMS: Dict[str, Any] = {
    "SWING_LR_BARS": 5,                  # TP-01: 스윙 판정 좌우 봉 수[cite: 1]
    "BIG_BODY_MULT": 2.0,                # TP-02: 장대봉 몸통 배수[cite: 1]
    "BIG_CLOSE_POS": 0.25,               # TP-03: 장대봉 종가 상·하위 비율[cite: 1]
    "ZONE_PCT_MIN": 0.01,                # TP-04a: ZONE 최소 비율 폭 (1%)[cite: 1]
    "ZONE_ATR_MULT": 0.5,                # TP-04b: ZONE ATR 배수 (0.5 * ATR14)[cite: 1]
    "MA_SLOPE_DAYS": 5,                  # TP-05: 이평 기울기 비교 기간[cite: 1]
    "VIX_FEAR_PCTL": 80.0,               # TP-06: S-VIX 공포 기준 백분위[cite: 1]
    "BB_SQZ_PCTL": 20.0,                 # TP-07: 볼린저 수축 기준 120일 백분위[cite: 1]
    "EXP_CONFIRM_RVOL": 1.5,             # TP-08: 확장/돌파 확인 RVOL[cite: 1]
    "DISP_EXTREME_PCTL": 2.5,            # TP-09: 이격도 극단 (750일 하위 2.5%)[cite: 1]
    "CLIMAX_RVOL": 2.0,                  # TP-10: 투매 클라이맥스 RVOL[cite: 1]
    "VOLA_SPIKE_PCTL": 90.0,             # TP-11: 변동성 급등 1년 백분위[cite: 1]
    "LEVEL_3STAR_HITS": 3,               # TP-12: 3성 레벨 반응 횟수[cite: 1]
    "TGT_MIN_WIDTH": 0.05,               # TP-13: 측정 목표 최소 폭 (5%)[cite: 1]
    "MIN_TRADING_VAL_KRW": 5_000_000_000,# TP-14: 최소 20일 평균 거래대금 (50억)[cite: 1]
    "MIN_MKTCAP_KRW": 300_000_000_000,   # TP-15: 최소 시가총액 (3,000억)[cite: 1]
    "ST1_MAX_HOLD_DAYS": 30,             # TP-16: ST-1 잔량 최대 보유 기간[cite: 1]
    "ST1_LOOKBACK_HIGH": 40,             # TP-17: 급락 시작일 탐색 범위[cite: 1]
    "EARNINGS_BLACKOUT_DAYS": 3,         # 실적 발표 D-3 ~ 당일: 전 셋업(ST-1~4) 신규 T 진입 금지
    "MAX_CHASE_GAP": 0.03,               # TP-18: 추격 금지 갭상승 (3%)[cite: 1]
    "ST1_MAX_STOP_PCT": 0.12,            # TP-19: ST-1 최대 손절폭 (12%)[cite: 1]
    "ST1_TIME_STOP_DAYS": 5,             # TP-20: ST-1 초기 반응 시간 손절 기한[cite: 1]
    "ST1_TIME_STOP_MIN_R": 0.5,          # TP-21: ST-1 5일 시간 손절 최소 반등 (+0.5R)[cite: 1]
    "MIN_EXPECTED_R": 1.5,               # TP-22: 최소 예상 R[cite: 1]
    "ST2A_MIN_BASE_DAYS": 15,            # TP-23: ST-2a 최소 베이스 기간 & B->R 전이 기간[cite: 1]
    "ST2_TIME_STOP_DAYS": 15,            # TP-24: ST-2 시간 손절 (MB0.25 도달 기한)[cite: 1]
    "ST4_MIN_BOX_WIDTH": 0.08,           # TP-25: ST-4 최소 박스 폭 (8%)[cite: 1]
    "T_SLEEVE_RATIO": 0.30,              # TP-26: T 슬리브 비중 기본값[cite: 1]
    "RISK_PER_TRADE": 0.01,              # TP-27: 거래당 위험 (슬리브 대비 1%)[cite: 1]
    "MAX_POS_RATIO_TOTAL": 0.05,         # TP-28: 종목당 최대 비중 (총자산 대비 5%)[cite: 1]
    "GAP_SIGMA_MULT": 2.0,               # TP-29: 갭 가정 시그마 배수[cite: 1]
    "MAX_CONCURRENT_RISK": 0.04,         # TP-30: 동시 위험 한도 (슬리브 대비 4%)[cite: 1]
    "MIN_POS_RATIO_SLEEVE": 0.03,        # TP-31: 최소 포지션 (슬리브 대비 3%)[cite: 1]
    "ROUND_TRIP_COST": 0.003,            # TP-32: 왕복 비용 (0.30%)[cite: 1]
    "SLEEVE_DD_BRAKE": -0.08,            # TP-33: 슬리브 낙폭 브레이크 (-8%)[cite: 1]
    "LOSS_STREAK_BRAKE": 5,              # TP-34: 연속 손실 감속 기준 (5회)[cite: 1]
    "VALUE_FRESH_DAYS": 90,              # TP-35: 가치 기록 신선도 (90일)[cite: 1]
    "TRANCHE_OVERLAP_TOL": 0.03,         # TP-36: 매수기준선-지지 겹침 폭 (±3%)[cite: 1]
    "ST4_EXTREME_SQZ_PCTL": 10.0,        # TP-37: ST-4 금지용 극단 수축 백분위[cite: 1]
}

# 8-4. 레짐별 허용 매트릭스[cite: 1]
REGIME_RISK_MATRIX: Dict[str, Dict[str, float]] = {
    "ST-1":  {"🟢": 1.0, "🟡": 1.0, "🔴": 1.0, "⚫": 0.5},
    "ST-2a": {"🟢": 1.0, "🟡": 0.5, "🔴": 0.0, "⚫": 0.0},
    "ST-2b": {"🟢": 1.0, "🟡": 1.0, "🔴": 0.5, "⚫": 0.0},
    "ST-3":  {"🟢": 1.0, "🟡": 0.5, "🔴": 0.0, "⚫": 0.0},
    "ST-4":  {"🟢": 1.0, "🟡": 1.0, "🔴": 0.5, "⚫": 0.0},
}


# =====================================================================
# 유틸리티: KRX 호가 단위 보정 및 ZONE 계산 (9-1)[cite: 1]
# =====================================================================
def get_krx_tick_size(price: float) -> int:
    """KRX 가격대별 호가 단위 반환[cite: 1]"""
    if price < 2000:
        return 1
    elif price < 5000:
        return 5
    elif price < 20000:
        return 10
    elif price < 50000:
        return 50
    elif price < 200000:
        return 100
    elif price < 500000:
        return 500
    return 1000

def adjust_krx_price(price: float, direction: str = "nearest", is_krx: bool = True) -> float:
    """주문가(up: 불리한 올림)와 손절가(down: 불리한 내림) KRX 호가 보정[cite: 1]"""
    if not is_krx or price <= 0 or math.isnan(price):
        return round(float(price), 2)
    tick = get_krx_tick_size(price)
    if direction == "up":
        return float(math.ceil(price / tick) * tick)
    elif direction == "down":
        return float(math.floor(price / tick) * tick)
    return float(round(price / tick) * tick)

def with_paren(main: Any, extra: Any) -> str:
    """'본문 (보충)' 표기. 보충이 없거나 '데이터 없음'이면 괄호를 붙이지 않는다 ('데이터 없음 (데이터 없음)' 중복 방지)."""
    m = NO_DATA if main in (None, "") else str(main)
    e = None if extra in (None, "", NO_DATA) else str(extra)
    return m if e is None or e == m else f"{m} ({e})"


def fmt_price(v: Any, is_krx: bool = True) -> str:
    """가격·금액 문구: 한국 종목 '232,500원', 미국 종목 '$172.50' (T 엔진·딥다이브 공통)"""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    return f"{int(f):,}원" if is_krx else f"${f:,.2f}"


def trading_days_until(as_of: Any, event_date: Any) -> Optional[int]:
    """기준일(데이터 날짜)부터 행사일까지 남은 거래일 수 (당일=0, 주말 제외·공휴일 미반영). 지났거나 없으면 None."""
    if not as_of or not event_date:
        return None
    try:
        a = pd.Timestamp(str(as_of)[:10]).date()
        e = pd.Timestamp(str(event_date)[:10]).date()
    except Exception:
        return None
    if e < a:
        return None
    return int(np.busday_count(a, e))


def is_earnings_blackout(days_to_earnings: Optional[int]) -> bool:
    """실적 발표 블랙아웃 여부 (D-EARNINGS_BLACKOUT_DAYS ~ 당일). 일정을 모르면 False."""
    return days_to_earnings is not None and int(days_to_earnings) <= TECH_PARAMS["EARNINGS_BLACKOUT_DAYS"]


def fetch_next_earnings_date(ticker: str) -> Optional[str]:
    """yfinance 일정(calendar)의 다음 실적 발표일. 범위로 주어지면 가장 이른 날(보수적). 없거나 실패하면 None."""
    try:
        import yfinance as yf
        cal = yf.Ticker(ticker).calendar or {}
        ds = [d for d in (cal.get("Earnings Date") or []) if d is not None]
        return str(min(ds))[:10] if ds else None
    except Exception:
        return None


def calc_zone_bounds(level_price: float, atr14: float) -> Tuple[float, float]:
    """2-1. ZONE: P ± max(0.01 * P, 0.5 * ATR14)[cite: 1]"""
    half_width = max(
        TECH_PARAMS["ZONE_PCT_MIN"] * level_price,
        TECH_PARAMS["ZONE_ATR_MULT"] * (atr14 if not np.isnan(atr14) and atr14 > 0 else level_price * 0.02)
    )
    return float(level_price - half_width), float(level_price + half_width)



# (원래 engine.py 1984~1984줄에서 옮김: 순환 참조 방지)
NO_DATA = "데이터 없음"
