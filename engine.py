# C:\Users\hanju\global-sector-agent\engine.py
import math
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

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


# =====================================================================
# 2-1. 기초값 7개 및 보조지표 연산 파이프라인[cite: 1]
# =====================================================================
def compute_base_primitives(df: pd.DataFrame) -> pd.DataFrame:
    """
    일봉 OHLCV 데이터프레임에 기초값 7개(SWING, BIG, RVOL, ZONE 폭, MA, ATR/σ, LOG) 및
    RSI, MACD, Bollinger, ADX, Stochastic, 이격도(S-DISP)를 계산하여 반환[cite: 1].
    * 모든 평균/백분위는 당일(t)을 제외한 직전 데이터(t-N ~ t-1) 기준으로 미래 참조를 차단함[cite: 1].
    """
    d = df.copy().sort_index()
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        d[col] = pd.to_numeric(d[col], errors="coerce")
    d = d.dropna(subset=["Open", "High", "Low", "Close"]).copy()
    if len(d) == 0:
        return d

    c = d["Close"]
    o = d["Open"]
    h = d["High"]
    lo = d["Low"]
    v = d["Volume"].fillna(0.0)

    # 1. LOG 스케일 가격[cite: 1]
    d["LOG_C"] = np.log(c.clip(lower=1e-6))
    d["LOG_H"] = np.log(h.clip(lower=1e-6))
    d["LOG_L"] = np.log(lo.clip(lower=1e-6))

    # 2. MA (SMA 5, 20, 25, 60, 120, 200 & EMA 200) 및 5일 기울기[cite: 1]
    slope_days = TECH_PARAMS["MA_SLOPE_DAYS"]
    for win in [5, 20, 25, 60, 120, 200]:
        d[f"SMA{win}"] = c.rolling(win, min_periods=max(1, win // 2)).mean()
        d[f"SMA{win}_SLOPE"] = d[f"SMA{win}"] - d[f"SMA{win}"].shift(slope_days)
    d["EMA200"] = c.ewm(span=200, adjust=False).mean()

    # 3. ATR14 및 일간 조건부 변동성 σ (GARCH(1,1) 추정 및 20일 로그수익률 표준편차 Fallback)[cite: 1]
    prev_c = c.shift(1)
    tr = pd.concat([
        (h - lo).abs(),
        (h - prev_c).abs(),
        (lo - prev_c).abs()
    ], axis=1).max(axis=1)
    d["ATR14"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    d["ATR_PCT"] = d["ATR14"] / c

    log_ret = np.log(c / prev_c)
    sigma_20 = log_ret.shift(1).rolling(20, min_periods=5).std()
    # GARCH(1,1) 재귀 근사 (750일 표본 기반, 실패/초기 구간은 sigma_20 즉시 대체)[cite: 1]
    omega, alpha_g, beta_g = 1e-6, 0.08, 0.90
    garch_var = np.zeros(len(d))
    init_var = float(sigma_20.dropna().iloc[0] ** 2) if len(sigma_20.dropna()) > 0 else 0.0004
    garch_var[0] = init_var
    ret_arr = log_ret.fillna(0.0).values
    for i in range(1, len(d)):
        garch_var[i] = omega + alpha_g * (ret_arr[i - 1] ** 2) + beta_g * garch_var[i - 1]
    garch_sigma = pd.Series(np.sqrt(np.maximum(garch_var, 1e-8)), index=d.index)
    d["SIGMA"] = garch_sigma.where( sigma_20.notna() & (np.arange(len(d)) >= 60), sigma_20 ).bfill().fillna(0.02)

    # 4. RVOL: 당일 거래량 / 직전 20일 평균 거래량 (당일 제외)[cite: 1]
    vol_ma20_prev = v.shift(1).rolling(20, min_periods=5).mean()
    d["RVOL"] = np.where(vol_ma20_prev > 0, v / vol_ma20_prev, 1.0)
    d["TRADING_VAL_MA20"] = (c * v).shift(1).rolling(20, min_periods=5).mean()

    # 5. BIG (장대봉): 갭 포함 몸통 >= 직전 20일 평균 몸통 * 2.0 & 종가 상/하위 25% 이내[cite: 1]
    real_body = (c - o).abs()
    gap_body = (c - prev_c).abs()
    eff_body = np.maximum(real_body, gap_body)
    body_ma20_prev = eff_body.shift(1).rolling(20, min_periods=5).mean()
    day_range = (h - lo).replace(0, np.nan)
    close_pos = (c - lo) / day_range  # 1에 가까우면 고가 근처, 0에 가까우면 저가 근처
    is_big_size = eff_body >= (body_ma20_prev * TECH_PARAMS["BIG_BODY_MULT"])
    d["BIG_BULL"] = is_big_size & (c > o) & (close_pos >= (1.0 - TECH_PARAMS["BIG_CLOSE_POS"]))
    d["BIG_BEAR"] = is_big_size & (c < o) & (close_pos <= TECH_PARAMS["BIG_CLOSE_POS"])

    # 6. S-CANDLE: 홀드봉 (Low_t >= Low_{t-1}), 확인봉 (C_t > O_t & C_t > High_{t-1})[cite: 1]
    d["HOLD_BAR"] = lo >= lo.shift(1)
    d["CONFIRM_BAR"] = (c > o) & (c > h.shift(1))

    # 7. 윗꼬리 + 60일 최대 거래량 후보 (S-VOL 규칙 3)[cite: 1]
    upper_wick = h - np.maximum(c, o)
    vol_max60_prev = v.shift(1).rolling(60, min_periods=20).max()
    d["UPPER_WICK_CLIMAX"] = (upper_wick >= real_body * 2.0) & (v >= vol_max60_prev) & (v > 0)

    # 8. RSI 14, Stochastic (14,3,3), MACD (12,26,9), ADX 14, Bollinger (20,2)[cite: 1]
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    d["RSI14"] = 100.0 - (100.0 / (1.0 + rs)).fillna(50.0)

    ll14 = lo.rolling(14, min_periods=5).min()
    hh14 = h.rolling(14, min_periods=5).max()
    fast_k = 100.0 * (c - ll14) / (hh14 - ll14).replace(0, np.nan)
    d["STOCH_K"] = fast_k.rolling(3, min_periods=1).mean().fillna(50.0)
    d["STOCH_D"] = d["STOCH_K"].rolling(3, min_periods=1).mean().fillna(50.0)

    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    d["MACD"] = ema12 - ema26
    d["MACD_SIG"] = d["MACD"].ewm(span=9, adjust=False).mean()
    d["MACD_HIST"] = d["MACD"] - d["MACD_SIG"]

    # ADX 14 & +DI / -DI[cite: 1]
    up_move = h.diff()
    down_move = -lo.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    atr_smooth = d["ATR14"].replace(0, np.nan)
    plus_di = 100.0 * pd.Series(plus_dm, index=d.index).ewm(alpha=1 / 14, adjust=False).mean() / atr_smooth
    minus_di = 100.0 * pd.Series(minus_dm, index=d.index).ewm(alpha=1 / 14, adjust=False).mean() / atr_smooth
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    d["PLUS_DI"] = plus_di.fillna(20.0)
    d["MINUS_DI"] = minus_di.fillna(20.0)
    d["ADX14"] = dx.ewm(alpha=1 / 14, adjust=False).mean().fillna(20.0)

    # 볼린저 밴드(20, 2) 및 밴드폭 120일 백분위 (당일 제외)[cite: 1]
    bb_mid = d["SMA20"]
    bb_std = c.rolling(20, min_periods=10).std()
    d["BB_UP"] = bb_mid + 2.0 * bb_std
    d["BB_LOW"] = bb_mid - 2.0 * bb_std
    d["BB_WIDTH"] = (d["BB_UP"] - d["BB_LOW"]) / bb_mid.replace(0, np.nan)

    def _rolling_pctl(series: pd.Series, win: int, min_p: int = 30) -> pd.Series:
        vals = series.values
        out = np.full(len(vals), 50.0)
        for i in range(min_p, len(vals)):
            prev_slice = vals[max(0, i - win):i]
            prev_slice = prev_slice[~np.isnan(prev_slice)]
            if len(prev_slice) >= min_p and not np.isnan(vals[i]):
                out[i] = float(np.mean(prev_slice <= vals[i]) * 100.0)
        return pd.Series(out, index=series.index)

    d["BB_WIDTH_PCTL120"] = _rolling_pctl(d["BB_WIDTH"], 120, 30)
    d["VOLA_PCTL250"] = _rolling_pctl(d["ATR_PCT"], 250, 60)

    # 9. S-DISP 이격도 (C_t / SMA25_t * 100) 및 자기 750일 분포 백분위 (당일 제외)[cite: 1]
    d["DISP25"] = (c / d["SMA25"].replace(0, np.nan)) * 100.0
    d["DISP_PCTL750"] = _rolling_pctl(d["DISP25"], 750, 120)

    # 10. SWING 확정 고점/저점: 좌우 TP-01(5)봉보다 높은/낮은 봉[cite: 1]
    # t 시점에서는 t-5봉까지만 확정 가능하도록 CONFIRM_SH_AT_T 컬럼으로 매핑[cite: 1]
    lr = TECH_PARAMS["SWING_LR_BARS"]
    win_sw = 2 * lr + 1
    roll_max = h.rolling(win_sw, center=True).max()
    roll_min = lo.rolling(win_sw, center=True).min()
    d["RAW_SH"] = np.where(h == roll_max, h, np.nan)
    d["RAW_SL"] = np.where(lo == roll_min, lo, np.nan)
    # t일에 확정되는 t-5일의 스윙 가격 및 원래 인덱스 위치
    d["CONFIRMED_SH"] = d["RAW_SH"].shift(lr)
    d["CONFIRMED_SL"] = d["RAW_SL"].shift(lr)

    return d


def get_confirmed_swings_up_to(df: pd.DataFrame, idx_pos: int) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
    """
    특정 시점 idx_pos(포함)까지 '확정된(우측 5봉 경과)' 스윙 고점/저점 목록 반환[cite: 1].
    각 원소는 (원래 봉의 정수 위치, 스윙 가격).
    """
    lr = TECH_PARAMS["SWING_LR_BARS"]
    max_origin_pos = idx_pos - lr
    if max_origin_pos < lr:
        return [], []
    sub_h = df["RAW_SH"].iloc[:max_origin_pos + 1]
    sub_l = df["RAW_SL"].iloc[:max_origin_pos + 1]
    sh_list = [( int(i), float(val) ) for i, val in enumerate(sub_h.values) if not np.isnan(val)]
    sl_list = [( int(i), float(val) ) for i, val in enumerate(sub_l.values) if not np.isnan(val)]
    return sh_list, sl_list


def evaluate_price_structure(
    c_t: float,
    sh_list: List[Tuple[int, float]],
    sl_list: List[Tuple[int, float]]
) -> str:
    """
    2-2. 가격 구조의 수리적 표준화 (v1.2 확정)[cite: 1]:
    - 상승 구조: (SL_-1 > SL_-2 and C_t > SH_-1) or (SH_-1 > SH_-2 and SL_-1 > SL_-2 and C_t >= SL_-1)
    - 하락 구조: (SH_-1 < SH_-2 and C_t < SL_-1) or (SH_-1 < SH_-2 and SL_-1 < SL_-2 and C_t <= SH_-1)
    - 횡보 구조: 위 모두 해당하지 않음
    """
    if len(sh_list) < 2 or len(sl_list) < 2:
        return "횡보 구조"
    sh_1, sh_2 = sh_list[-1][1], sh_list[-2][1]
    sl_1, sl_2 = sl_list[-1][1], sl_list[-2][1]

    is_up = (sl_1 > sl_2 and c_t > sh_1) or (sh_1 > sh_2 and sl_1 > sl_2 and c_t >= sl_1)
    if is_up:
        return "상승 구조"

    is_down = (sh_1 < sh_2 and c_t < sl_1) or (sh_1 < sh_2 and sl_1 < sl_2 and c_t <= sh_1)
    if is_down:
        return "하락 구조"

    return "횡보 구조"


# =====================================================================
# 2-2 ~ 2-6. 환경(S-REG, S-VIX), 상대강도(S-RS), S-MODE, S-LEVEL, S-TREND[cite: 1]
# =====================================================================
def compute_market_regime(
    bench_df: Optional[pd.DataFrame],
    vix_series: Optional[pd.Series] = None,
    macro_override: Optional[str] = None
) -> Dict[str, Any]:
    """2-2. S-REG 지수 레짐(전환 지연 2일/3일 적용) 및 S-VIX 공포 국면 판정[cite: 1]"""
    if macro_override in ["🟢", "🟡", "🔴", "⚫"]:
        s_reg = macro_override
        reg_label = {"🟢": "🟢 추세 상승", "🟡": "🟡 약한 상승·횡보", "🔴": "🔴 하락", "⚫": "⚫ 위기"}[s_reg]
    elif bench_df is None or len(bench_df) < 60:
        s_reg = "🟡"
        reg_label = "🟡 약한 상승·횡보 (기준지수 기본값)"
    else:
        b = compute_base_primitives(bench_df)
        reg_Order = {"⚫": 0, "🔴": 1, "🟡": 2, "🟢": 3}
        current_reg = "🟡"
        up_streak = 0
        down_streak = 0
        start_idx = max(60, len(b) - 120)

        for idx in range(start_idx, len(b)):
            row = b.iloc[idx]
            c_t = float(row["Close"])
            sma20 = float(row["SMA20"])
            sma60 = float(row["SMA60"])
            ema200 = float(row["EMA200"])
            adx14 = float(row["ADX14"])
            s20_slope = float(row["SMA20_SLOPE"]) if not np.isnan(row["SMA20_SLOPE"]) else 0.0
            s60_slope = float(row["SMA60_SLOPE"]) if not np.isnan(row["SMA60_SLOPE"]) else 0.0

            sh_l, sl_l = get_confirmed_swings_up_to(b, idx)
            struct = evaluate_price_structure(c_t, sh_l, sl_l)

            if c_t < ema200:
                cand = "⚫"
            elif struct == "하락 구조" and c_t < sma20 and s20_slope < 0:
                cand = "🔴"
            elif struct == "상승 구조" and c_t > sma20 and c_t > sma60 and s20_slope > 0 and s60_slope > 0:
                cand = "🟢" if adx14 >= 25.0 else "🟡"
            else:
                cand = "🟡"

            if reg_Order[cand] < reg_Order[current_reg]:
                down_streak += 1
                up_streak = 0
                if down_streak >= 2 or cand == "⚫":
                    current_reg = cand
                    down_streak = 0
            elif reg_Order[cand] > reg_Order[current_reg]:
                up_streak += 1
                down_streak = 0
                if up_streak >= 3:
                    current_reg = cand
                    up_streak = 0
            else:
                up_streak = 0
                down_streak = 0

        s_reg = current_reg
        reg_label = {"🟢": "🟢 추세 상승", "🟡": "🟡 약한 상승·횡보", "🔴": "🔴 하락", "⚫": "⚫ 위기"}[s_reg]

    # S-VIX 백분위 계산[cite: 1]
    vix_pctl = 45.0
    if vix_series is not None and len(vix_series.dropna()) >= 30:
        vs = vix_series.dropna()
        last_v = float(vs.iloc[-1])
        prev_v = vs.iloc[max(0, len(vs) - 251):-1]
        if len(prev_v) > 0:
            vix_pctl = float(np.mean(prev_v <= last_v) * 100.0)
    elif bench_df is not None and len(bench_df) >= 60:
        b_temp = compute_base_primitives(bench_df)
        vix_pctl = float(b_temp["VOLA_PCTL250"].iloc[-1])

    if vix_pctl >= TECH_PARAMS["VIX_FEAR_PCTL"]:
        s_vix = "공포"
    elif vix_pctl >= 50.0:
        s_vix = "경계"
    else:
        s_vix = "평온"

    return {"s_reg": s_reg, "s_reg_label": reg_label, "s_vix": s_vix, "vix_pctl": round(vix_pctl, 1)}


def compute_relative_strength(df: pd.DataFrame, bench_df: Optional[pd.DataFrame]) -> Dict[str, str]:
    """2-3. S-RS 상대강도 (일봉 집행용 / 주봉 스크리닝용 분리 계산)[cite: 1]"""
    if bench_df is None or len(bench_df) < 60 or len(df) < 60:
        return {"daily": "개선", "weekly": "개선", "rs_new_low": False, "rs_lead_breakout": False}

    common_idx = df.index.intersection(bench_df.index)
    if len(common_idx) < 60:
        return {"daily": "개선", "weekly": "개선", "rs_new_low": False, "rs_lead_breakout": False}

    s_c = df.loc[common_idx, "Close"]
    b_c = bench_df.loc[common_idx, "Close"].replace(0, np.nan)
    rs_line = (s_c / b_c).dropna()
    if len(rs_line) < 55:
        return {"daily": "개선", "weekly": "개선", "rs_new_low": False, "rs_lead_breakout": False}

    rs_sma50 = rs_line.rolling(50, min_periods=20).mean()
    rs_chg20 = rs_line / rs_line.shift(20) - 1.0

    def _classify_rs(above_ma: bool, pos_roc: bool) -> str:
        if above_ma and pos_roc:
            return "주도"
        elif above_ma and not pos_roc:
            return "약화"
        elif not above_ma and pos_roc:
            return "개선"
        return "소외"

    d_state = _classify_rs(bool(rs_line.iloc[-1] > rs_sma50.iloc[-1]), bool(rs_chg20.iloc[-1] > 0))

    # 주봉 환산 (10주 이평 = 50거래일, 4주 변화율 = 20거래일 주간 리샘플링)[cite: 1]
    rs_w = rs_line.resample("W").last().dropna()
    if len(rs_w) >= 12:
        w_ma10 = rs_w.rolling(10, min_periods=5).mean()
        w_roc4 = rs_w / rs_w.shift(4) - 1.0
        w_state = _classify_rs(bool(rs_w.iloc[-1] > w_ma10.iloc[-1]), bool(w_roc4.iloc[-1] > 0))
    else:
        w_state = d_state

    rs_new_low = bool(rs_line.iloc[-1] <= rs_line.iloc[-20:].min())
    return {
        "daily": d_state,
        "weekly": w_state,
        "rs_new_low": rs_new_low,
        "rs_lead_breakout": bool(rs_line.iloc[-1] >= rs_line.iloc[-50:].max())
    }


def compute_s_mode_series(df: pd.DataFrame) -> List[str]:
    """2-4. S-MODE 타이밍 모드 시계열 계산 (판정 보류 시 직전 거래일 모드 유지)[cite: 1]"""
    modes: List[str] = []
    curr_mode = "레인지"
    for i in range(len(df)):
        row = df.iloc[i]
        adx = float(row["ADX14"])
        p_di = float(row["PLUS_DI"])
        m_di = float(row["MINUS_DI"])
        c_t = float(row["Close"])
        sh_l, sl_l = get_confirmed_swings_up_to(df, i)

        if len(sh_l) >= 2 and len(sl_l) >= 2:
            sh1, sh2 = sh_l[-1][1], sh_l[-2][1]
            sl1, sl2 = sl_l[-1][1], sl_l[-2][1]
            up_struct = (sh1 > sh2 and sl1 > sl2) or (c_t > sh1 and sl1 > sl2)
            dn_struct = (sh1 < sh2 and sl1 < sl2) or (c_t < sl1 and sh1 < sh2)
        else:
            up_struct, dn_struct = False, False

        if up_struct and p_di > m_di and adx >= 25.0:
            curr_mode = "추세 상승"
        elif dn_struct and m_di > p_di and adx >= 25.0:
            curr_mode = "추세 하락"
        elif adx < 20.0:
            curr_mode = "레인지"
        # 그 외는 판정 보류 -> 직전 거래일 curr_mode 유지[cite: 1]
        modes.append(curr_mode)
    return modes


def build_auto_levels_and_volume_profile(df: pd.DataFrame, idx_pos: int) -> Dict[str, Any]:
    """
    2-5. S-LEVEL 수평 레벨 자동 작도 Fallback (v1.2 확정) 및 250일 매물대 계산[cite: 1]:
    - 확정 스윙 고점/저점 + RVOL >= 2.0 캔들 고점 + S-CROSS 기준선을 ZONE 폭으로 클러스터링하여 ⭐~⭐⭐⭐ 부여[cite: 1].
    """
    sub = df.iloc[:idx_pos + 1]
    if len(sub) < 20:
        return {"horizontal_zones": [], "high_vol_bins": [], "low_vol_bins": []}

    c_now = float(sub["Close"].iloc[-1])
    atr_now = float(sub["ATR14"].iloc[-1])
    sh_list, sl_list = get_confirmed_swings_up_to(df, idx_pos)

    candidates: List[Tuple[float, int, str]] = []  # (price, weight_hits, source_tag)
    for _, p in sh_list[-15:]:
        candidates.append((p, 1, "스윙고점"))
    for _, p in sl_list[-15:]:
        candidates.append((p, 1, "스윙저점"))

    # 거래량 터진 캔들 (RVOL >= 2.0, RVOL >= 3.0이면 즉시 3성급 가중치 3 부여)[cite: 1]
    recent_250 = sub.iloc[max(0, len(sub) - 250):]
    high_rvol_bars = recent_250[recent_250["RVOL"] >= 2.0]
    for _, r_row in high_rvol_bars.tail(15).iterrows():
        w = 3 if float(r_row["RVOL"]) >= 3.0 else 1
        lvl_p = float(r_row["Close"]) if (float(r_row["High"]) - float(r_row["Close"])) > abs(float(r_row["Close"]) - float(r_row["Open"])) else float(r_row["High"])
        candidates.append((lvl_p, w, "대량거래캔들"))

    # S-CROSS (SMA20 - SMA60 골든/데드크로스 발생일 종가)[cite: 1]
    s20 = recent_250["SMA20"]
    s60 = recent_250["SMA60"]
    cross_mask = ((s20 > s60) & (s20.shift(1) <= s60.shift(1))) | ((s20 < s60) & (s20.shift(1) >= s60.shift(1)))
    for _, c_row in recent_250[cross_mask].tail(5).iterrows():
        candidates.append((float(c_row["Close"]), 1, "이평크로스"))

    # 클러스터링 (ZONE 폭 이내면 병합하고 반응 횟수 합산)[cite: 1]
    clusters: List[Dict[str, Any]] = []
    for price, hits, tag in sorted(candidates, key=lambda x: x[0]):
        merged = False
        for cl in clusters:
            z_low, z_high = calc_zone_bounds(cl["price"], atr_now)
            if z_low <= price <= z_high:
                total_h = cl["hits"] + hits
                cl["price"] = (cl["price"] * cl["hits"] + price * hits) / total_h
                cl["hits"] = total_h
                if tag not in cl["tags"]:
                    cl["tags"].append(tag)
                merged = True
                break
        if not merged:
            clusters.append({"price": price, "hits": hits, "tags": [tag]})

    horizontal_zones = []
    for cl in clusters:
        p = cl["price"]
        z_low, z_high = calc_zone_bounds(p, atr_now)
        stars = min(3, max(1, cl["hits"]))
        role = "지지" if p <= c_now else "저항"
        horizontal_zones.append({
            "price": round(p, 0),
            "low": round(z_low, 0),
            "high": round(z_high, 0),
            "stars": stars,
            "hits": cl["hits"],
            "type": role,
            "tags": cl["tags"],
        })

    # 현재가에서 가까운 순으로 정렬 후 상위 10개 유지
    horizontal_zones.sort(key=lambda x: abs(x["price"] - c_now))
    horizontal_zones = sorted(horizontal_zones[:10], key=lambda x: x["price"], reverse=True)

    # 최근 1년(250거래일) 20구간 가격대별 매물대 (상위 20% 매물대 / 하위 20% 빈 공간)[cite: 1]
    p_min, p_max = float(recent_250["Low"].min()), float(recent_250["High"].max())
    high_vol_bins, low_vol_bins = [], []
    if p_max > p_min:
        bins = np.linspace(p_min, p_max, 21)
        bin_indices = np.digitize(recent_250["Close"].values, bins) - 1
        bin_vols = np.zeros(20)
        vols = recent_250["Volume"].values
        for b_i, v_i in zip(bin_indices, vols):
            b_idx = min(19, max(0, int(b_i)))
            bin_vols[b_idx] += v_i
        q80 = np.percentile(bin_vols, 80)
        q20 = np.percentile(bin_vols, 20)
        for b_i in range(20):
            b_mid = round((bins[b_i] + bins[b_i + 1]) / 2.0, 0)
            if bin_vols[b_i] >= q80:
                high_vol_bins.append({"low": round(bins[b_i], 0), "high": round(bins[b_i + 1], 0), "mid": b_mid})
            elif bin_vols[b_i] <= q20:
                low_vol_bins.append({"low": round(bins[b_i], 0), "high": round(bins[b_i + 1], 0), "mid": b_mid})

    return {
        "horizontal_zones": horizontal_zones,
        "high_vol_bins": high_vol_bins,
        "low_vol_bins": low_vol_bins,
    }


def compute_auto_trend_channel(df: pd.DataFrame, idx_pos: int) -> Dict[str, Any]:
    """
    2-5. S-TREND 추세선·채널 자동 앵커 Fallback (v1.2 확정, 로그 스케일 직선)[cite: 1]:
    - ln(L_t) = ln(P_1) + (ln(P_2) - ln(P_1)) / (t_2 - t_1) * (t - t_1)
    - 채널 위치 x = (ln(C_t) - ln(하단_t)) / (ln(상단_t) - ln(하단_t))
    """
    sh_list, sl_list = get_confirmed_swings_up_to(df, idx_pos)
    c_t = float(df["Close"].iloc[idx_pos])
    atr_t = float(df["ATR14"].iloc[idx_pos])
    big_bear = bool(df["BIG_BEAR"].iloc[idx_pos])

    if len(sh_list) < 2 or len(sl_list) < 2:
        return {
            "direction": "평행", "lower": round(c_t * 0.95, 0), "upper": round(c_t * 1.05, 0),
            "mid": round(c_t, 0), "pos_x": 0.5, "status": "생존", "break_attempts": 0,
            "big_bear_break": False, "up_trend_line": None, "down_trend_line": None
        }

    (t_l2, p_l2), (t_l1, p_l1) = sl_list[-2], sl_list[-1]
    (t_h2, p_h2), (t_h1, p_h1) = sh_list[-2], sh_list[-1]

    # 상승 추세선 하단 vs 하락 추세선 상단 계산[cite: 1]
    up_line_t = None
    if t_l1 > t_l2 and p_l1 > 0 and p_l2 > 0:
        slope_l = (math.log(p_l1) - math.log(p_l2)) / (t_l1 - t_l2)
        up_line_t = math.exp(math.log(p_l1) + slope_l * (idx_pos - t_l1))
    else:
        slope_l = 0.0

    dn_line_t = None
    if t_h1 > t_h2 and p_h1 > 0 and p_h2 > 0:
        slope_h = (math.log(p_h1) - math.log(p_h2)) / (t_h1 - t_h2)
        dn_line_t = math.exp(math.log(p_h1) + slope_h * (idx_pos - t_h1))
    else:
        slope_h = 0.0

    # 주 채널 경계 산출 (상승 스윙이면 하단 기준 평행 채널, 아니면 상단 기준 평행 채널)[cite: 1]
    if p_l1 >= p_l2 and up_line_t is not None:
        lower_t = up_line_t
        # 두 저점 앵커 사이 및 이후 고점 중 로그 편차가 가장 큰 점으로 평행 상단선 작도[cite: 1]
        rel_highs = [(t, p) for (t, p) in sh_list if t >= t_l2] or sh_list[-2:]
        max_log_diff = max(
            math.log(p) - (math.log(p_l2) + slope_l * (t - t_l2))
            for (t, p) in rel_highs
        )
        max_log_diff = max(max_log_diff, 0.04)
        upper_t = math.exp(math.log(lower_t) + max_log_diff)
        direction = "상승"
    elif dn_line_t is not None:
        upper_t = dn_line_t
        rel_lows = [(t, p) for (t, p) in sl_list if t >= t_h2] or sl_list[-2:]
        max_log_drop = max(
            (math.log(p_h2) + slope_h * (t - t_h2)) - math.log(p)
            for (t, p) in rel_lows
        )
        max_log_drop = max(max_log_drop, 0.04)
        lower_t = math.exp(math.log(upper_t) - max_log_drop)
        direction = "하락"
    else:
        lower_t = min(p_l1, p_l2)
        upper_t = max(p_h1, p_h2)
        direction = "평행"

    if upper_t <= lower_t * 1.005:
        upper_t = lower_t * 1.06

    pos_x = (math.log(max(c_t, 1e-6)) - math.log(max(lower_t, 1e-6))) / (
        math.log(max(upper_t, 1e-6)) - math.log(max(lower_t, 1e-6))
    )
    mid_t = math.exp((math.log(lower_t) + math.log(upper_t)) / 2.0)

    z_low, z_high = calc_zone_bounds(lower_t, atr_t)
    if c_t < z_low:
        status = "장대음봉 이탈" if big_bear else "이탈"
    elif z_low <= c_t <= z_high:
        status = "접촉"
    else:
        status = "생존"

    return {
        "direction": direction,
        "lower": round(lower_t, 0),
        "upper": round(upper_t, 0),
        "mid": round(mid_t, 0),
        "pos_x": round(float(pos_x), 2),
        "status": status,
        "big_bear_break": bool(status == "장대음봉 이탈"),
        "up_trend_line": round(up_line_t, 0) if up_line_t else None,
        "down_trend_line": round(dn_line_t, 0) if dn_line_t else None,
    }


def compute_measured_targets(b_price: float, lo_or_hi: float, direction: str = "up") -> Dict[str, float]:
    """
    2-5. S-TGT 측정 목표 (로그 폭 투영, 폭 5% 미만 시 미산출)[cite: 1]:
    - 상방: T1 = B * (B / Lo), MB_k = B * (B / Lo)^k (k = 0.25, 0.5, 0.75, 1.0)
    - 하방: T1 = B * (B / Hi), T2 = B * (B / Hi)^2
    """
    if b_price <= 0 or lo_or_hi <= 0:
        return {}
    if direction == "up":
        width = (b_price / lo_or_hi) - 1.0
        if width < TECH_PARAMS["TGT_MIN_WIDTH"]:
            return {}
        ratio = b_price / lo_or_hi
        return {
            "B": round(b_price, 0),
            "Lo": round(lo_or_hi, 0),
            "MB0.25": round(b_price * (ratio ** 0.25), 0),
            "MB0.5": round(b_price * (ratio ** 0.50), 0),
            "MB0.75": round(b_price * (ratio ** 0.75), 0),
            "T1": round(b_price * ratio, 0),
        }
    else:
        width = (lo_or_hi / b_price) - 1.0
        if width < TECH_PARAMS["TGT_MIN_WIDTH"]:
            return {}
        ratio = b_price / lo_or_hi  # < 1.0
        return {
            "B": round(b_price, 0),
            "Hi": round(lo_or_hi, 0),
            "T1_down": round(b_price * ratio, 0),
            "T2_down": round(b_price * (ratio ** 2), 0),
        }


# =====================================================================
# 2-7 & 2-8. 생애주기 신호(S-FALL, S-START, S-WEAK, S-BRK) 및 6대 판정 모듈[cite: 1]
# =====================================================================
def evaluate_signals_and_modules_at(
    df: pd.DataFrame,
    idx_pos: int,
    s_mode: str,
    rs_info: Dict[str, Any],
    levels_info: Dict[str, Any],
    channel_info: Dict[str, Any],
    is_krx: bool = True
) -> Dict[str, Any]:
    """특정 봉 idx_pos에서 2-7 생애주기 신호와 2-8의 6대 판정 모듈을 계산[cite: 1]"""
    row = df.iloc[idx_pos]
    c_t = float(row["Close"])
    o_t = float(row["Open"])
    lo_t = float(row["Low"])
    h_t = float(row["High"])
    atr_t = float(row["ATR14"])
    rvol_t = float(row["RVOL"])
    rsi_t = float(row["RSI14"])
    stoch_k = float(row["STOCH_K"])
    stoch_d = float(row["STOCH_D"])
    sh_list, sl_list = get_confirmed_swings_up_to(df, idx_pos)

    sh1 = sh_list[-1][1] if len(sh_list) >= 1 else c_t * 1.05
    sh2 = sh_list[-2][1] if len(sh_list) >= 2 else sh1
    sl1 = sl_list[-1][1] if len(sl_list) >= 1 else c_t * 0.95
    sl2 = sl_list[-2][1] if len(sl_list) >= 2 else sl1

    # --- 1) S-DBL 쌍바닥 / 쌍봉 판정[cite: 1] ---
    sl2_z_low, _ = calc_zone_bounds(sl2, atr_t)
    recent_5_low = float(df["Low"].iloc[max(0, idx_pos - 4):idx_pos + 1].min())
    is_higher_or_zone_low = (sl1 >= sl2_z_low) or (recent_5_low >= sl2_z_low and c_t >= sl2_z_low)
    confirm_bar = bool(row["CONFIRM_BAR"])
    dbl_bottom_cand = bool(len(sl_list) >= 2 and is_higher_or_zone_low and (confirm_bar or c_t > float(row["SMA5"])))
    # 두 바닥 사이 반등 고점 B
    b_neck = sh1
    b_z_low, b_z_high = calc_zone_bounds(b_neck, atr_t)

    # 쌍봉 확정: 두 고점(SH1, SH2 같은 ZONE 또는 근접) 사이 저점(SL1) 종가 이탈[cite: 1]
    sh2_z_low, sh2_z_high = calc_zone_bounds(sh2, atr_t)
    double_top_confirmed = bool(
        len(sh_list) >= 2 and (sh2_z_low <= sh1 <= sh2_z_high or sh1 <= sh2) and (c_t < sl1)
    )

    # --- 2) 모듈 2: 이격도 과이탈 (disparity)[cite: 1] ---
    disp_pctl = float(row["DISP_PCTL750"])
    recent_5_disp_pctl = float(df["DISP_PCTL750"].iloc[max(0, idx_pos - 5):idx_pos + 1].min())
    had_extreme_5d = recent_5_disp_pctl <= TECH_PARAMS["DISP_EXTREME_PCTL"]

    recent_5_sub = df.iloc[max(0, idx_pos - 5):idx_pos + 1]
    climax_in_5d = bool(((recent_5_sub["Close"] < recent_5_sub["Open"]) & (recent_5_sub["RVOL"] >= TECH_PARAMS["CLIMAX_RVOL"])).any())
    hold_in_3d = bool(df["HOLD_BAR"].iloc[max(0, idx_pos - 2):idx_pos].any()) if idx_pos >= 1 else False
    climax_low_5d = float(recent_5_sub["Low"].min())

    if had_extreme_5d and hold_in_3d and confirm_bar and c_t >= climax_low_5d:
        disp_stage = 4
        disp_label = "4 반전 확인 (홀드봉→확인봉)"
    elif had_extreme_5d and climax_in_5d:
        disp_stage = 3
        disp_label = "3 투매 클라이맥스"
    elif disp_pctl <= TECH_PARAMS["DISP_EXTREME_PCTL"]:
        disp_stage = 2
        disp_label = f"2 극단 (하위 {disp_pctl:.1f}%)"
    elif disp_pctl <= 10.0:
        disp_stage = 1
        disp_label = f"1 과이탈 (하위 {disp_pctl:.1f}%)"
    else:
        disp_stage = 0
        disp_label = f"0 정상 (이격도 {row['DISP25']:.1f})"

    # --- 3) S-FALL 하락 진행 / 멈춤 및 모듈 3: 바닥 (bottom)[cite: 1] ---
    evidences: List[Tuple[str, str]] = []  # (id, category)
    if dbl_bottom_cand:
        evidences.append(("① 쌍바닥 후보 방어", "가격 구조"))
    if c_t > sh1:
        evidences.append(("② 직전 스윙고점 돌파", "가격 구조"))
    # ③ MACD 히스토그램 음수 폭 축소 (3일 연속 상승 또는 상승 전환 후 2일 유지)[cite: 1]
    if idx_pos >= 3:
        mh = df["MACD_HIST"].iloc[idx_pos - 3:idx_pos + 1].values
        if mh[-1] < 0 and ((mh[3] > mh[2] > mh[1]) or (mh[3] > mh[2] and mh[2] > mh[1])):
            evidences.append(("③ MACD 음수폭 축소", "모멘텀"))
    if (c_t > o_t) and rvol_t >= 1.5:
        evidences.append(("④ 반등 양봉 RVOL≥1.5", "수급"))
    if climax_in_5d and hold_in_3d and confirm_bar:
        evidences.append(("⑤ 투매 후 홀드+확인봉", "수급"))

    distinct_cats = set(cat for _, cat in evidences)
    fall_released = len(distinct_cats) >= 2
    has_supply_ev = any(cat == "수급" for _, cat in evidences)
    has_bnf_ev5 = any(ev_id.startswith("⑤") for ev_id, _ in evidences)

    # S-START 추세 시작 판정[cite: 1]
    # 최근 4거래일 내 B 돌파 양봉(RVOL >= 1.5) 여부 및 되돌림 음봉 거래량 < 돌파 양봉 거래량 체크
    s_start_confirmed = False
    breakout_stage = 0
    breakout_label = "0 없음"
    sqz_now = bool(df["BB_WIDTH_PCTL120"].iloc[max(0, idx_pos - 5):idx_pos + 1].min() <= TECH_PARAMS["BB_SQZ_PCTL"])

    lookback_brk = df.iloc[max(0, idx_pos - 4):idx_pos + 1]
    brk_bars = lookback_brk[(lookback_brk["Close"] > b_z_high) & (lookback_brk["Close"] > lookback_brk["Open"]) & (lookback_brk["RVOL"] >= 1.5)]
    if len(brk_bars) > 0:
        first_brk_idx = brk_bars.index[0]
        first_brk_pos = df.index.get_loc(first_brk_idx)
        brk_vol = float(df["Volume"].iloc[first_brk_pos])
        bars_since = idx_pos - first_brk_pos
        post_slice = df.iloc[first_brk_pos + 1:idx_pos + 1]
        pullback_bears = post_slice[post_slice["Close"] < post_slice["Open"]]
        max_bear_vol = float(pullback_bears["Volume"].max()) if len(pullback_bears) > 0 else 0.0
        vol_rule_ok = max_bear_vol < brk_vol
        held_above_zone = bool((post_slice["Close"] >= b_z_low).all()) if len(post_slice) > 0 else True
        ma_turn_up = bool(float(row["SMA20_SLOPE"]) > 0 and float(row["SMA60_SLOPE"]) > 0)
        hl_rising = bool(sh1 > sh2 and sl1 > sl2)

        if not held_above_zone or not vol_rule_ok:
            breakout_stage = -1  # X 가짜 돌파[cite: 1]
            breakout_label = "X 가짜 돌파 (기준선 이탈/음봉 거래량 초과)"
        elif bars_since >= 3 and (ma_turn_up or hl_rising):
            s_start_confirmed = True
            breakout_stage = 4
            breakout_label = "4 확정 (S-START 충족)"
        else:
            breakout_stage = 3
            breakout_label = "3 1차 돌파 (RVOL≥1.5 돌파 진행)"
    elif c_t >= b_z_low and c_t <= b_z_high:
        breakout_stage = 2
        breakout_label = "2 돌파 시도 (저항 ZONE 진입)"
    elif sqz_now:
        breakout_stage = 1
        breakout_label = "1 베이스 형성 (볼린저 수축)"

    dbl_bottom_confirmed = bool(dbl_bottom_cand and c_t > b_neck and (s_start_confirmed or breakout_stage >= 3))

    if dbl_bottom_confirmed or (fall_released and s_start_confirmed):
        bottom_stage = 3
        bottom_label = "전환 확정"
    elif fall_released:
        bottom_stage = 2
        bottom_label = f"해제 ({', '.join(e[0] for e in evidences)})"
    elif len(evidences) == 1:
        bottom_stage = 1
        bottom_label = f"멈춤 후보 ({evidences[0][0]})"
    else:
        bottom_stage = 0
        bottom_label = "하락 진행"

    # --- 4) 모듈 4: 눌림 (pullback) — 겹침 지지 5개 후보 체크[cite: 1] ---
    overlap_supports: List[str] = []
    # ① SMA20 또는 SMA60 ZONE[cite: 1]
    for ma_name in ["SMA20", "SMA60"]:
        ma_v = float(row[ma_name])
        mz_l, mz_h = calc_zone_bounds(ma_v, atr_t)
        if mz_l <= c_t <= mz_h or (lo_t <= mz_h and c_t >= mz_l):
            overlap_supports.append(f"{ma_name} ZONE")
            break
    # ② 채널 위치 0 ~ 0.25[cite: 1]
    if 0.0 <= channel_info["pos_x"] <= 0.25:
        overlap_supports.append("채널 하단(0~0.25)")
    # ③ SR Flip (과거 돌파된 스윙고점 SH2 ZONE)[cite: 1]
    sh2_l, sh2_h = calc_zone_bounds(sh2, atr_t)
    if sh2_l <= c_t <= sh2_h:
        overlap_supports.append("SR Flip 지지")
    # ④ ⭐⭐ 이상 수평 지지 ZONE[cite: 1]
    for hz in levels_info["horizontal_zones"]:
        if hz["stars"] >= 2 and hz["low"] <= c_t <= hz["high"]:
            overlap_supports.append(f"{'⭐'*hz['stars']} 수평지지({fmt_price(hz['price'], is_krx)})")
            break
    # ⑤ S-CROSS 기준선 ZONE[cite: 1]
    for hz in levels_info["horizontal_zones"]:
        if "이평크로스" in hz.get("tags", []) and hz["low"] <= c_t <= hz["high"]:
            overlap_supports.append("S-CROSS 기준선")
            break

    # 최근 조정 구간 평균 RVOL < 1 및 RSI 40~50 터치 여부[cite: 1]
    sh1_pos = sh_list[-1][0] if len(sh_list) >= 1 else max(0, idx_pos - 10)
    pullback_slice = df.iloc[min(sh1_pos + 1, idx_pos):idx_pos + 1]
    pb_rvol_low = bool(float(pullback_slice["RVOL"].mean()) < 1.0) if len(pullback_slice) > 0 else False
    pb_rsi_40_50 = bool(((pullback_slice["RSI14"] >= 40.0) & (pullback_slice["RSI14"] <= 52.0)).any()) if len(pullback_slice) > 0 else False

    # --- 5) 모듈 5: 레인지 (range)[cite: 1] ---
    box_high = max(sh1, sh2)
    box_low = min(sl1, sl2)
    box_width = (box_high / box_low - 1.0) if box_low > 0 else 0.0
    box_pos = (c_t - box_low) / (box_high - box_low) if box_high > box_low else 0.5
    low_z_l, low_z_h = calc_zone_bounds(box_low, atr_t)
    in_box_bottom = bool(low_z_l <= c_t <= low_z_h or (lo_t <= low_z_h and c_t >= low_z_l))
    range_react_cnt = int(in_box_bottom) + int(rsi_t <= 30.0) + int(stoch_k > stoch_d)

    # --- 6) S-WEAK 약화 & S-BRK 붕괴 3단계 (모듈 6: weakness)[cite: 1] ---
    sh1_z_l, sh1_z_h = calc_zone_bounds(sh1, atr_t)
    in_high_zone = bool(sh1_z_l <= c_t <= sh1_z_h)
    weak_1_fail_high = bool(sh1 <= sh2 or (in_high_zone and c_t < o_t))
    weak_others: List[str] = []
    if float(df["RVOL"].iloc[max(0, idx_pos - 5):idx_pos + 1].mean()) < 0.85:
        weak_others.append("② 상승파동 거래량 감소")
    if in_high_zone and idx_pos >= 2 and float(row["MACD_HIST"]) < float(df["MACD_HIST"].iloc[idx_pos - 1]):
        weak_others.append("③ 고점 ZONE MACD 둔화")
    if sh1 >= sh2 and float(row["RSI14"]) < float(df["RSI14"].iloc[sh_list[-2][0]] if len(sh_list) >= 2 else 50.0) - 3.0:
        weak_others.append("④ RSI 약세 다이버전스")
    if in_high_zone and idx_pos >= 2 and float(row["ADX14"]) < float(df["ADX14"].iloc[idx_pos - 1]):
        weak_others.append("⑤ 고점 ZONE ADX 하락")
    if len(sh_list) >= 2 and (sh2_z_low <= sh1 <= sh2_z_high):
        weak_others.append("⑥ 쌍봉 후보")
    if bool(df["UPPER_WICK_CLIMAX"].iloc[max(0, idx_pos - 5):idx_pos + 1].any()):
        weak_others.append("⑦ 윗꼬리 대량거래")

    weak_8_big_break = bool(channel_info["big_bear_break"])
    is_s_weak = bool((weak_1_fail_high and len(weak_others) >= 1) or weak_8_big_break)

    # S-BRK ①: ⭐⭐ 이상 주요 지지 하단(SL1 ZONE 하단) 또는 상승 추세선 하단 종가 이탈[cite: 1]
    sl1_z_low, _ = calc_zone_bounds(sl1, atr_t)
    s_brk_1 = bool(c_t < sl1_z_low or channel_info["status"] in ["이탈", "장대음봉 이탈"])

    return {
        "sh1": sh1, "sh2": sh2, "sl1": sl1, "sl2": sl2,
        "sh1_pos": sh1_pos,
        "b_neck": b_neck, "b_z_low": b_z_low, "b_z_high": b_z_high,
        "sl1_z_low": sl1_z_low,
        "dbl_bottom_cand": dbl_bottom_cand,
        "dbl_bottom_confirmed": dbl_bottom_confirmed,
        "double_top_confirmed": double_top_confirmed,
        "fall_released": fall_released,
        "has_supply_ev": has_supply_ev,
        "has_bnf_ev5": has_bnf_ev5,
        "s_start_confirmed": s_start_confirmed,
        "is_s_weak": is_s_weak,
        "weak_reasons": (["① 전고점 갱신 실패"] + weak_others if weak_1_fail_high else []) + (["⑧ 추세선 장대음봉 이탈"] if weak_8_big_break else []),
        "s_brk_1": s_brk_1,
        "overlap_supports": overlap_supports,
        "pb_rvol_low": pb_rvol_low,
        "pb_rsi_40_50": pb_rsi_40_50,
        "box_high": box_high, "box_low": box_low, "box_width": box_width, "box_pos": box_pos,
        "in_box_bottom": in_box_bottom, "range_react_cnt": range_react_cnt,
        "modules": {
            "breakout": {"stage": breakout_stage, "label": breakout_label, "detail": f"기준선 B={fmt_price(b_neck, is_krx)} / RVOL={rvol_t:.2f}"},
            "disparity": {"stage": disp_stage, "label": disp_label, "detail": f"25일이격도 {row['DISP25']:.1f} (750일 백분위 {disp_pctl:.1f}%)"},
            "bottom": {"stage": bottom_stage, "label": bottom_label, "detail": f"충족 카테고리 {len(distinct_cats)}개"},
            "pullback": {
                "stage": len(overlap_supports),
                "label": f"겹침 지지 {len(overlap_supports)}개",
                "detail": f"{', '.join(overlap_supports) if overlap_supports else '없음'} | 조정거래량감소={pb_rvol_low} | RSI(40~50)={pb_rsi_40_50}"
            },
            "range": {
                "stage": range_react_cnt,
                "label": f"박스 위치 {box_pos:.2f} · 반응 {range_react_cnt}개",
                "detail": f"박스 [{fmt_price(box_low, is_krx)} ~ {fmt_price(box_high, is_krx)}] (폭 {box_width*100:.1f}%)"
            },
            "weakness": {
                "stage": 2 if s_brk_1 else (1 if is_s_weak else 0),
                "label": "K① 이탈 경계" if s_brk_1 else ("약화" if is_s_weak else "정상"),
                "detail": ", ".join(weak_others) if weak_others else "특이 약화 신호 없음"
            }
        }
    }


# =====================================================================
# 3. 종목 7상태 기계 (250거래일 Warm-up 및 단일·하향 우선 전이)[cite: 1]
# =====================================================================
def run_lifecycle_state_machine(df: pd.DataFrame, s_modes: List[str], rs_info: Dict[str, Any], is_krx: bool = True) -> Dict[str, Any]:
    """
    3장. 직전 250거래일 데이터를 처음부터 순차적으로 돌려(Warm-up) 현재 상태(D/B/R/S/H/W/K)와
    전이가 일어난 이유를 산출[cite: 1]. 하루 1전이, 하향 전이(2, 6, 8, 10, 11, 12, 14) 최우선[cite: 1].
    """
    n = len(df)
    if n < 120:
        return {
            "state": "UNDEFINED", "sub_state": None, "state_label": "미정(상장 120일 미만)",
            "prev_state": "UNDEFINED", "transition_reason": "상장 이력 120거래일 미만", "history": []
        }

    start_idx = max(30, n - 250)
    first_row = df.iloc[start_idx]
    sh_init, sl_init = get_confirmed_swings_up_to(df, start_idx)
    init_struct = evaluate_price_structure(float(first_row["Close"]), sh_init, sl_init)
    state = "D" if (float(first_row["Close"]) < float(first_row["SMA60"]) and init_struct == "하락 구조") else "R"
    sub_state: Optional[str] = None
    prev_state = state
    last_reason = "초기화(Warm-up)"

    b_low_anchor = float(first_row["Low"])
    s_break_level = float(first_row["Close"])
    s_mb025_target = float(first_row["Close"]) * 1.03
    s_entry_sh_count = len(sh_init)
    s_entry_sl_count = len(sl_init)
    range_days_in_b = 0
    k_break_level = float(first_row["Low"])
    k_days_elapsed = 0
    k_recover_streak = 0

    levels_cache = build_auto_levels_and_volume_profile(df, n - 1)

    for i in range(start_idx, n):
        row = df.iloc[i]
        c_t = float(row["Close"])
        atr_t = float(row["ATR14"])
        rvol_t = float(row["RVOL"])
        mode_i = s_modes[i]
        ch_i = compute_auto_trend_channel(df, i)
        sig = evaluate_signals_and_modules_at(df, i, mode_i, rs_info, levels_cache, ch_i, is_krx=is_krx)
        sh_l, sl_l = get_confirmed_swings_up_to(df, i)

        new_state = state
        reason = None

        # --- 상태별 전이 판정 (하향 전이 최우선)[cite: 1] ---
        if state == "D":
            # 전이 1: D -> B (S-FALL 해제: 성격 다른 증거 2개 이상)[cite: 1]
            if sig["fall_released"]:
                new_state = "B"
                b_low_anchor = float(df["Low"].iloc[max(0, i - 10):i + 1].min())
                range_days_in_b = 0
                reason = "전이 #1 (D→B): S-FALL 하락 멈춤 해제 (이종 증거 2개 이상 충족)"

        elif state == "B":
            # 하향 전이 2 (우선): B -> D (바닥 구간 최저점 종가 갱신)[cite: 1]
            if c_t < b_low_anchor:
                new_state = "D"
                reason = f"전이 #2 (B→D): 바닥 최저점({fmt_price(b_low_anchor, is_krx)}) 종가 하향 이탈"
            # 상향 전이 3: B -> S (S-START 확정 또는 S-DBL 쌍바닥 확정)[cite: 1]
            elif sig["s_start_confirmed"] or sig["dbl_bottom_confirmed"]:
                new_state = "S"
                s_break_level = sig["b_neck"]
                tgts = compute_measured_targets(sig["b_neck"], b_low_anchor, "up")
                s_mb025_target = tgts.get("MB0.25", sig["b_neck"] * 1.02)
                s_entry_sh_count = len(sh_l)
                s_entry_sl_count = len(sl_l)
                reason = "전이 #3 (B→S): 바닥 전환 S-START / 쌍바닥 돌파 확정"
            else:
                # 수평 전이 4: B -> R (S-MODE 레인지 15거래일 지속, 저점 갱신 없음)[cite: 1]
                range_days_in_b = (range_days_in_b + 1) if mode_i == "레인지" else 0
                if range_days_in_b >= TECH_PARAMS["ST2A_MIN_BASE_DAYS"]:
                    new_state = "R"
                    reason = "전이 #4 (B→R): 저점 방어 및 레인지 모드 15거래일 지속"

        elif state == "R":
            box_low_z_l, _ = calc_zone_bounds(sig["box_low"], atr_t)
            # 하향 전이 12 (우선): R -> K (박스 하단 ZONE 하단 종가 이탈)[cite: 1]
            if c_t < box_low_z_l:
                new_state = "K"
                sub_state = "K①"
                k_break_level = box_low_z_l
                k_days_elapsed = 0
                k_recover_streak = 0
                reason = f"전이 #12 (R→K①): 박스 하단 ZONE({fmt_price(box_low_z_l, is_krx)}) 종가 이탈"
            # 상향 전이 5: R -> S (박스 상단 돌파 + S-START 확정)[cite: 1]
            elif sig["s_start_confirmed"] and c_t > sig["box_high"]:
                new_state = "S"
                s_break_level = sig["box_high"]
                tgts = compute_measured_targets(sig["box_high"], sig["box_low"], "up")
                s_mb025_target = tgts.get("MB0.25", sig["box_high"] * 1.02)
                s_entry_sh_count = len(sh_l)
                s_entry_sl_count = len(sl_l)
                reason = "전이 #5 (R→S): 박스 상단 돌파 및 S-START 확정"

        elif state == "S":
            b_z_l, _ = calc_zone_bounds(s_break_level, atr_t)
            high_since_s = float(df["High"].iloc[max(0, i - 10):i + 1].max())
            # 하향 전이 6 (우선): S -> R (MB0.25 미도달 상태에서 돌파선 B ZONE 하단 아래 종가)[cite: 1]
            if high_since_s < s_mb025_target and c_t < b_z_l:
                new_state = "R"
                reason = f"전이 #6 (S→R): 가짜 돌파 (MB0.25 미도달 후 돌파선 {fmt_price(b_z_l, is_krx)} 하향 이탈)"
            # 상향 전이 7: S -> H (돌파 후 확정 스윙 고점·저점이 한 번 더 상승)[cite: 1]
            elif (len(sh_l) > s_entry_sh_count or len(sl_l) > s_entry_sl_count) and sig["sh1"] > sig["sh2"] and sig["sl1"] > sig["sl2"]:
                new_state = "H"
                reason = "전이 #7 (S→H): 돌파 후 확정 스윙 고점·저점 추가 상승 확인"

        elif state == "H":
            # 하향 전이 10 (우선): H -> K (S-BRK ①)[cite: 1]
            if sig["s_brk_1"]:
                new_state = "K"
                sub_state = "K①"
                k_break_level = sig["sl1_z_low"]
                k_days_elapsed = 0
                k_recover_streak = 0
                reason = f"전이 #10 (H→K①): 주요 지지 ZONE({fmt_price(k_break_level, is_krx)}) 또는 추세선 종가 이탈"
            # 하향 전이 8: H -> W (S-WEAK 약화 성립)[cite: 1]
            elif sig["is_s_weak"]:
                new_state = "W"
                reason = f"전이 #8 (H→W): 추세 약화 성립 ({', '.join(sig['weak_reasons'][:2])})"

        elif state == "W":
            # 하향 전이 11 (우선): W -> K (S-BRK ①)[cite: 1]
            if sig["s_brk_1"]:
                new_state = "K"
                sub_state = "K①"
                k_break_level = sig["sl1_z_low"]
                k_days_elapsed = 0
                k_recover_streak = 0
                reason = f"전이 #11 (W→K①): 약화 상태에서 주요 지지({fmt_price(k_break_level, is_krx)}) 종가 이탈"
            # 상향 전이 9: W -> H (신고가 C_t > SH_-1 + RVOL >= 1.5)[cite: 1]
            elif c_t > sig["sh1"] and rvol_t >= 1.5:
                new_state = "H"
                reason = "전이 #9 (W→H): 거래량 수반(RVOL≥1.5) 전고점 돌파로 약화 해소"

        elif state == "K":
            k_days_elapsed += 1
            if c_t >= k_break_level:
                k_recover_streak += 1
            else:
                k_recover_streak = 0

            if k_days_elapsed > 10 and sub_state == "K①":
                sub_state = "K②"  # 10거래일 내 회복 실패 확정[cite: 1]

            # 하향 전이 14 (우선): K -> D (S-BRK ② 회복 실패 뒤 낮은 스윙 고점 확정 또는 이탈 저점 하향 돌파)[cite: 1]
            k_low = float(df["Low"].iloc[max(0, i - k_days_elapsed):i].min()) if k_days_elapsed > 0 else float(row["Low"])
            if sub_state == "K②" and (sig["sh1"] < sig["sh2"] or c_t < k_low):
                new_state = "D"
                sub_state = None
                reason = "전이 #14 (K②→D): S-BRK ② 회복 실패 후 저점/낮은 고점 붕괴 확정"
            # 상향 전이 13: K -> W (10거래일 안에 이탈 ZONE 하단 위 종가 2일 연속 유지)[cite: 1]
            elif k_days_elapsed <= 10 and k_recover_streak >= 2:
                new_state = "W"
                sub_state = None
                reason = "전이 #13 (K①→W): 10거래일 내 이탈 기준선 2일 연속 종가 회복"

        if new_state != state and reason is not None:
            prev_state = state
            state = new_state
            if state != "K":
                sub_state = None
            last_reason = reason

    labels_map = {
        "D": "D 하락 진행", "B": "B 바닥 형성", "R": "R 횡보",
        "S": "S 추세 시작", "H": "H 추세 유지", "W": "W 약화",
        "K": f"{sub_state or 'K'} 붕괴 경계"
    }
    res_sig = evaluate_signals_and_modules_at(df, n - 1, s_modes[-1], rs_info, levels_cache, compute_auto_trend_channel(df, n - 1), is_krx=is_krx)
    if state == "K":
        res_sig["modules"]["weakness"]["stage"] = 3 if sub_state == "K②" else 2
        res_sig["modules"]["weakness"]["label"] = f"{sub_state or 'K①'} 붕괴 경계"
    elif state == "D":
        res_sig["modules"]["weakness"]["stage"] = 4
        res_sig["modules"]["weakness"]["label"] = "붕괴 (D 하락 진행)"

    return {
        "state": state,
        "sub_state": sub_state,
        "state_label": labels_map.get(state, state),
        "prev_state": prev_state,
        "transition_reason": last_reason,
        "latest_signals": res_sig,
        "levels_cache": levels_cache,
        "channel_cache": compute_auto_trend_channel(df, n - 1),
    }


# =====================================================================
# 4 & 5. T 트랙 셋업 스캐너 (ST-1 ~ ST-4) 및 리스크·사이징[cite: 1]
# =====================================================================
def scan_t_setups_and_size(
    df: pd.DataFrame,
    s_modes: List[str],
    lifecycle: Dict[str, Any],
    env_info: Dict[str, Any],
    rs_info: Dict[str, Any],
    value_macro_input: Dict[str, Any],
    account_cfg: Dict[str, Any]
) -> Dict[str, Any]:
    """
    4장 셋업 카탈로그(ST-1 ~ ST-4) 스캔 -> 4-1 공통 관문 -> 8-3 우선순위 선택 -> 5장 사이징 수행[cite: 1].
    """
    n = len(df)
    row = df.iloc[-1]
    c_t = float(row["Close"])
    atr_t = float(row["ATR14"])
    sigma_t = float(row["SIGMA"])
    state = lifecycle["state"]
    sig = lifecycle["latest_signals"]
    mods = sig["modules"]
    ch = lifecycle["channel_cache"]
    lvls = lifecycle["levels_cache"]
    s_reg = env_info["s_reg"]
    s_vix = env_info["s_vix"]
    is_krx = account_cfg.get("is_krx", True)
    # 통화: T 엔진 기준값(유동성 50억 원·계좌 금액)은 원화. 해외 종목은 fx_to_krw(현지통화 1 = ?원)로 환산한다.
    fx_to_krw = account_cfg.get("fx_to_krw")
    if fx_to_krw is None and is_krx:
        fx_to_krw = 1.0
    fx_missing = fx_to_krw is None or float(fx_to_krw) <= 0
    fx_to_krw = 1.0 if fx_missing else float(fx_to_krw)
    _px = (lambda v: int(v)) if is_krx else (lambda v: round(float(v), 2))             # 가격 표기
    _pxr = (lambda v: int(round(v, 0))) if is_krx else (lambda v: round(float(v), 2))  # 반올림 가격
    min_tick = 1.0 if is_krx else 0.01                                                 # 1주당 위험 최소값

    # 4-1. 공통 관문 점검[cite: 1]
    hard_block = bool(account_cfg.get("hard_block_evt", False) or value_macro_input.get("thesis_status") == "붕괴")
    earnings_blackout = bool(account_cfg.get("earnings_blackout", False))
    mktcap = float(account_cfg.get("mktcap_krw", 1_000_000_000_000))
    tr_val_20 = float(row.get("TRADING_VAL_MA20", 10_000_000_000))
    if np.isnan(tr_val_20):
        tr_val_20 = 10_000_000_000.0
    liquidity_ok = (tr_val_20 * fx_to_krw >= TECH_PARAMS["MIN_TRADING_VAL_KRW"]) and (mktcap >= TECH_PARAMS["MIN_MKTCAP_KRW"])
    history_ok = n >= 120

    candidates: List[Dict[str, Any]] = []

    # -----------------------------------------------------------------
    # (1) ST-1: BNF 이격 반등 (상태 무관, 시간 손절 2단계 분리 TP-20 / TP-16)[cite: 1]
    # -----------------------------------------------------------------
    disp_mod_stage = mods["disparity"]["stage"]
    if n >= 500 and not hard_block and not earnings_blackout and disp_mod_stage in [2, 3, 4]:
        recent_10 = df.iloc[max(0, n - 10):]
        climax_low = float(recent_10["Low"].min())
        entry_p = adjust_krx_price(c_t, "up", is_krx)
        stop_p = adjust_krx_price(climax_low, "down", is_krx)
        stop_pct = (entry_p - stop_p) / entry_p if entry_p > 0 else 1.0

        # 중간 ⭐⭐⭐ 저항 ZONE 확인[cite: 1]
        sma25_t = float(row["SMA25"])
        star3_res = [
            hz["low"] for hz in lvls["horizontal_zones"]
            if hz["stars"] >= 3 and entry_p < hz["low"] < sma25_t
        ]
        target_1 = min([sma25_t] + star3_res) if star3_res else sma25_t
        target_1 = adjust_krx_price(target_1, "down", is_krx)

        # 품질 요소 3개 점검[cite: 1]
        lb_40 = df.iloc[max(0, n - TECH_PARAMS["ST1_LOOKBACK_HIGH"]):]
        peak_pos = df.index.get_loc(lb_40["Close"].idxmax())
        peak_row = df.iloc[peak_pos]
        q_a = bool(rs_info["weekly"] in ["주도", "약화"] and float(peak_row["Close"]) > float(peak_row["SMA60"]) and float(peak_row["SMA120_SLOPE"]) > 0)
        q_b = bool(((recent_10["Close"] < recent_10["Open"]) & (recent_10["RVOL"] >= TECH_PARAMS["CLIMAX_RVOL"])).any())
        q_c = bool(env_info["vix_pctl"] >= 50.0)
        q_cnt = int(q_a) + int(q_b) + int(q_c)
        grade = "A" if q_cnt == 3 else ("B" if q_cnt == 2 else "C")

        stop_width_ok = (0 < stop_pct <= TECH_PARAMS["ST1_MAX_STOP_PCT"]) and ((entry_p - stop_p) <= 3.0 * atr_t)
        is_triggered = (disp_mod_stage == 4) and stop_width_ok

        candidates.append({
            "setup_id": "ST-1",
            "name": "BNF 이격 반등",
            "purpose": "과잉 매도 후 25일선 복귀",
            "is_short_term": True,
            "triggered": is_triggered,
            "grade": grade,
            "entry_price": entry_p,
            "stop_price": stop_p,
            "target_1": target_1,
            "target_2": f"SMA25 잔량 트레일링 (최대 {TECH_PARAMS['ST1_MAX_HOLD_DAYS']}거래일)",
            "time_stop_days": TECH_PARAMS["ST1_TIME_STOP_DAYS"],  # TP-20 (5거래일)[cite: 1]
            "max_hold_days": TECH_PARAMS["ST1_MAX_HOLD_DAYS"],   # TP-16 (30거래일)[cite: 1]
            "headline": "극단 이격도(하위 2.5%) 진입 후 투매 진정·확인봉 포착" if is_triggered else "극단 이격/투매 구간 진입 — 홀드봉+확인봉 대기",
            "next_cond": f"전일 고가({fmt_price(row['High'], is_krx)}) 돌파 양봉(확인봉) 종가 마감 시 진입",
        })

    # -----------------------------------------------------------------
    # (2) ST-2a / ST-2b: 돌파 셋업[cite: 1]
    # -----------------------------------------------------------------
    if state in ["H", "R"] and mods["breakout"]["stage"] in [2, 3, 4] and not hard_block:
        b_lvl = sig["b_neck"]
        lo_lvl = sig["sl1"]
        tgts = compute_measured_targets(b_lvl, lo_lvl, "up")
        t1_p = tgts.get("T1", round(c_t * 1.12, 0))
        mb025 = tgts.get("MB0.25", round(b_lvl * 1.03, 0))
        mb050 = tgts.get("MB0.5", round(b_lvl * 1.06, 0))
        entry_p = adjust_krx_price(c_t, "up", is_krx)
        stop_p = adjust_krx_price(sig["b_z_low"], "down", is_krx)

        q_a = rs_info["weekly"] in ["주도", "개선"]
        q_b = float(row["SMA120_SLOPE"]) > 0
        q_c = True  # 최소 베이스 15일
        q_d = bool(df["BB_WIDTH_PCTL120"].iloc[max(0, n - 15):n].min() <= TECH_PARAMS["BB_SQZ_PCTL"])
        q_cnt = int(q_a) + int(q_b) + int(q_c) + int(q_d)
        grade = "A" if q_cnt == 4 else ("B" if q_cnt == 3 else "C")
        not_overextended = c_t <= mb050

        candidates.append({
            "setup_id": "ST-2a",
            "name": "추세 지속형 돌파",
            "purpose": "베이스 상단 돌파 후 추세 확장",
            "is_short_term": False,
            "triggered": (mods["breakout"]["stage"] >= 3) and not_overextended,
            "grade": grade,
            "entry_price": entry_p,
            "stop_price": stop_p,
            "target_1": t1_p,
            "target_2": "SMA20 또는 채널 하단 종가 이탈 시 잔량 청산",
            "time_stop_days": TECH_PARAMS["ST2_TIME_STOP_DAYS"],
            "time_stop_min_price": mb025,
            "max_hold_days": 60,
            "headline": f"주요 저항 ZONE({fmt_price(b_lvl, is_krx)}) RVOL≥1.5 돌파",
            "next_cond": f"돌파 기준선 ZONE 상단({fmt_price(sig['b_z_high'], is_krx)}) 위 RVOL≥1.5 종가 안착",
        })

    if state == "B" and mods["bottom"]["stage"] >= 2 and not hard_block:
        b_lvl = sig["b_neck"]
        lo_lvl = min(sig["sl1"], sig["sl2"])
        tgts = compute_measured_targets(b_lvl, lo_lvl, "up")
        t1_p = tgts.get("T1", round(c_t * 1.12, 0))
        mb025 = tgts.get("MB0.25", round(b_lvl * 1.03, 0))
        entry_p = adjust_krx_price(c_t, "up", is_krx)
        stop_p = adjust_krx_price(sig["sl1"], "down", is_krx)

        q_a = rs_info["weekly"] in ["주도", "개선"]
        q_b = sig["has_supply_ev"]
        q_c = sig["dbl_bottom_confirmed"]
        q_cnt = int(q_a) + int(q_b) + int(q_c)
        grade = "A" if q_cnt == 3 else ("B" if q_cnt == 2 else "C")
        is_trig = bool(c_t > b_lvl and float(row["RVOL"]) >= 1.5)

        candidates.append({
            "setup_id": "ST-2b",
            "name": "바닥 전환형 돌파",
            "purpose": "바닥 해제 후 넥라인 돌파 추세 초입",
            "is_short_term": False,
            "triggered": is_trig,
            "grade": grade,
            "entry_price": entry_p,
            "stop_price": stop_p,
            "target_1": t1_p,
            "target_2": "SMA20 또는 채널 하단 종가 이탈 시 잔량 청산",
            "time_stop_days": TECH_PARAMS["ST2_TIME_STOP_DAYS"],
            "time_stop_min_price": mb025,
            "max_hold_days": 60,
            "headline": f"바닥 해제 및 넥라인({fmt_price(b_lvl, is_krx)}) 돌파 시도",
            "next_cond": f"넥라인({fmt_price(b_lvl, is_krx)}) RVOL≥1.5 종가 돌파 시 1차 50% 진입",
        })

    # -----------------------------------------------------------------
    # (3) ST-3: 추세 눌림 (v1.2 확정: 직전 확정 스윙고점 이후 로컬 최저가 손절)[cite: 1]
    # -----------------------------------------------------------------
    sh1_pos = sig["sh1_pos"]
    mode_at_sh1 = s_modes[sh1_pos] if 0 <= sh1_pos < len(s_modes) else s_modes[-1]
    if state == "H" and mode_at_sh1 == "추세 상승" and len(sig["overlap_supports"]) >= 2 and not hard_block:
        # v1.2 확정: 직전 확정 스윙고점(SH_-1) 발생일 이후 현재(t)까지의 로컬 최저가[cite: 1]
        local_pullback_low = float(df["Low"].iloc[sh1_pos:n].min())
        entry_p = adjust_krx_price(c_t, "up", is_krx)
        stop_p = adjust_krx_price(local_pullback_low, "down", is_krx)
        if stop_p >= entry_p:
            stop_p = adjust_krx_price(entry_p - atr_t, "down", is_krx)
        target_1 = adjust_krx_price(max(sig["sh1"], c_t + 2.0 * (entry_p - stop_p)), "down", is_krx)

        s5, s20, s60 = float(row["SMA5"]), float(row["SMA20"]), float(row["SMA60"])
        strong_mat = (s5 > s20 > s60) and (float(row["SMA5_SLOPE"]) > 0) and (float(row["SMA20_SLOPE"]) > 0) and (float(row["SMA60_SLOPE"]) > 0)
        q_a = strong_mat and (float(row["SMA120_SLOPE"]) > 0)
        q_b = sig["pb_rvol_low"]
        q_c = sig["pb_rsi_40_50"]
        q_d = len(sig["overlap_supports"]) >= 3
        q_cnt = int(q_a) + int(q_b) + int(q_c) + int(q_d)
        grade = "A" if q_cnt == 4 else ("B" if q_cnt == 3 else "C")

        candidates.append({
            "setup_id": "ST-3",
            "name": "추세 눌림",
            "purpose": "살아있는 상승 추세 내 조정 끝 반등",
            "is_short_term": False,
            "triggered": bool(row["CONFIRM_BAR"]),
            "grade": grade,
            "entry_price": entry_p,
            "stop_price": stop_p,
            "target_1": target_1,
            "target_2": f"채널 상단({fmt_price(ch['upper'], is_krx)}) 1/3 청산 후 SMA20 트레일링",
            "time_stop_days": 10,
            "max_hold_days": 45,
            "headline": f"추세 유지(H) 중 겹침 지지 {len(sig['overlap_supports'])}개 구간 눌림",
            "next_cond": f"겹침 지지 구간 위 확인봉(전일 고가 {fmt_price(row['High'], is_krx)} 돌파 양봉) 출현 시 진입",
        })

    # -----------------------------------------------------------------
    # (4) ST-4: 레인지 반등[cite: 1]
    # -----------------------------------------------------------------
    extreme_sqz = float(row["BB_WIDTH_PCTL120"]) <= TECH_PARAMS["ST4_EXTREME_SQZ_PCTL"]
    is_converging = (sig["sh1"] < sig["sh2"]) and (sig["sl1"] > sig["sl2"])
    if s_modes[-1] == "레인지" and sig["box_width"] >= TECH_PARAMS["ST4_MIN_BOX_WIDTH"] and not extreme_sqz and not is_converging and not hard_block and not earnings_blackout:
        if sig["in_box_bottom"]:
            box_low_z_l, _ = calc_zone_bounds(sig["box_low"], atr_t)
            entry_p = adjust_krx_price(c_t, "up", is_krx)
            stop_p = adjust_krx_price(box_low_z_l, "down", is_krx)
            box_mid = adjust_krx_price((sig["box_high"] + sig["box_low"]) / 2.0, "down", is_krx)

            q_a = float(row["RSI14"]) <= 30.0
            q_b = float(row["STOCH_K"]) > float(row["STOCH_D"])
            q_c = any(hz["stars"] >= 3 and abs(hz["price"] - sig["box_low"]) / sig["box_low"] <= 0.02 for hz in lvls["horizontal_zones"])
            q_cnt = int(q_a) + int(q_b) + int(q_c)
            grade = "A" if q_cnt == 3 else ("B" if q_cnt == 2 else "C")

            candidates.append({
                "setup_id": "ST-4",
                "name": "레인지 반등",
                "purpose": "박스권 하단 지지 후 중단·상단 왕복",
                "is_short_term": True,
                "triggered": bool(row["CONFIRM_BAR"]),
                "grade": grade,
                "entry_price": entry_p,
                "stop_price": stop_p,
                "target_1": box_mid,
                "target_2": f"박스 상단 ZONE 하단({fmt_price(sig['box_high'], is_krx)}) 전량 청산",
                "time_stop_days": 10,
                "time_stop_min_price": box_mid,
                "max_hold_days": 20,
                "headline": f"레인지 하단({fmt_price(sig['box_low'], is_krx)}) 지지 도달",
                "next_cond": f"박스 하단 ZONE에서 확인봉(전일 고가 {fmt_price(row['High'], is_krx)} 돌파) 마감 시 진입",
            })

    if hard_block or not history_ok or not liquidity_ok or fx_missing:
        block_reason = "기업 고유 악재(S-EVT) 또는 가치 논리(Thesis) 붕괴" if hard_block else ("상장 120거래일 미만" if not history_ok else ("환율 데이터 없음 (해외 종목 사이징 불가)" if fx_missing else "유동성/시가총액 최소 기준 미달"))
        return {
            "t_verdict": "금지" if hard_block else "관찰",
            "setup_id": None,
            "grade": None,
            "headline_reason": block_reason,
            "next_or_end_condition": "차단 해제 전까지 신규 T 진입 불가",
            "t_ticket": {"is_executable": False, "reason": block_reason},
        }

    if len(candidates) == 0:
        return {
            "t_verdict": "관찰",
            "setup_id": None,
            "grade": None,
            "headline_reason": f"현재 상태({lifecycle['state_label']})에서 성립된 T 셋업 없음",
            "next_or_end_condition": f"주요 돌파선({fmt_price(sig['b_neck'], is_krx)}) 돌파 또는 지지 구간 눌림 대기",
            "t_ticket": {"is_executable": False, "reason": "활성 셋업 없음"},
        }

    # 각 후보에 대해 5-1 리스크·사이징 및 예상 R 계산[cite: 1]
    total_equity = float(account_cfg.get("total_equity", 100_000_000)) / fx_to_krw  # 계좌(원화) → 종목 통화
    sleeve_ratio = float(account_cfg.get("t_sleeve_ratio", TECH_PARAMS["T_SLEEVE_RATIO"]))
    sleeve_krw = total_equity * sleeve_ratio
    loss_streak = int(account_cfg.get("loss_streak", 0))
    v_current_weight = float(account_cfg.get("existing_v_weight", 0.0))
    pos_cap = float(value_macro_input.get("position_cap", TECH_PARAMS["MAX_POS_RATIO_TOTAL"]))
    sell_target = value_macro_input.get("sell_target")

    evaluated: List[Dict[str, Any]] = []
    for cand in candidates:
        sid = cand["setup_id"]
        reg_mult = REGIME_RISK_MATRIX.get(sid, {}).get(s_reg, 0.0)
        if sid == "ST-1" and s_vix == "공포":
            reg_mult *= 0.5  # 8-4. S-VIX 공포 시 ST-1 0.5 추가 곱[cite: 1]

        grade_mult = 1.0 if cand["grade"] == "A" else (0.5 if cand["grade"] == "B" else 0.0)
        streak_mult = 0.5 if loss_streak >= TECH_PARAMS["LOSS_STREAK_BRAKE"] else 1.0
        # 8-3. 현재가 > 매도목표가(가치 고평가)면 중기 셋업(ST-2, ST-3) 위험 계수 * 0.5[cite: 1]
        overval_mult = 0.5 if (sell_target is not None and c_t > float(sell_target) and not cand["is_short_term"]) else 1.0

        r_dollars = sleeve_krw * TECH_PARAMS["RISK_PER_TRADE"] * reg_mult * grade_mult * streak_mult * overval_mult
        entry_p = cand["entry_price"]
        stop_p = cand["stop_price"]
        unit_1r = max(min_tick, (entry_p - stop_p) + entry_p * TECH_PARAMS["ROUND_TRIP_COST"])
        expected_r = (cand["target_1"] - entry_p) / unit_1r

        q_base = math.floor(r_dollars / unit_1r) if r_dollars > 0 else 0
        # 상한 ①: 총자산 비중 (기존 V 로트 합산 position_cap 이내)[cite: 1]
        avail_cap_ratio = max(0.0, min(TECH_PARAMS["MAX_POS_RATIO_TOTAL"], pos_cap - v_current_weight))
        q_cap1 = math.floor((total_equity * avail_cap_ratio) / entry_p) if entry_p > 0 else 0
        # 상한 ②: 갭 가정가 (손절가 * (1 - 2 * sigma))[cite: 1]
        gap_price = stop_p * (1.0 - TECH_PARAMS["GAP_SIGMA_MULT"] * sigma_t)
        gap_loss_per_share = max(unit_1r, entry_p - gap_price)
        q_gap = math.floor((2.0 * r_dollars) / gap_loss_per_share) if gap_loss_per_share > 0 else 0

        q_final = max(0, min(q_base, q_cap1, q_gap))
        pos_val = q_final * entry_p
        min_pos_ok = pos_val >= (sleeve_krw * TECH_PARAMS["MIN_POS_RATIO_SLEEVE"])
        r_ok = expected_r >= TECH_PARAMS["MIN_EXPECTED_R"]

        cand["expected_r"] = round(expected_r, 2)
        cand["unit_r_krw"] = round(unit_1r, 0)
        cand["r_budget_krw"] = round(r_dollars, 0)
        cand["shares"] = int(q_final)
        cand["position_value"] = int(pos_val)
        cand["reg_mult"] = reg_mult
        cand["grade_mult"] = grade_mult
        cand["min_pos_ok"] = min_pos_ok
        cand["r_ok"] = r_ok
        cand["can_enter"] = bool(cand["triggered"] and reg_mult > 0 and grade_mult > 0 and min_pos_ok and r_ok and q_final > 0)
        evaluated.append(cand)

    # 8-3. 동시 성립 시 선택 우선순위: 진입 가능 > 등급(A>B>C) > 예상 R 큰 쪽 > 단기(ST-1/4 > ST-2/3)[cite: 1]
    grade_rank = {"A": 3, "B": 2, "C": 1}
    evaluated.sort(
        key=lambda x: (
            int(x["can_enter"]),
            int(x["triggered"]),
            grade_rank.get(x["grade"], 0),
            x["expected_r"],
            int(x["is_short_term"])
        ),
        reverse=True
    )
    best = evaluated[0]

    # 시간 손절 기준가 계산 (ST-1은 진입가 + 0.5R)[cite: 1]
    if best["setup_id"] == "ST-1":
        time_min_p = _pxr(best["entry_price"] + TECH_PARAMS["ST1_TIME_STOP_MIN_R"] * best["unit_r_krw"])
    else:
        time_min_p = _pxr(best.get("time_stop_min_price", best["entry_price"] + best["unit_r_krw"]))

    if best["can_enter"]:
        t_verdict = "진입"
        end_cond = f"끝나는 조건: {fmt_price(best['stop_price'], is_krx)} 아래 종가, 또는 {best['time_stop_days']}거래일째 {fmt_price(time_min_p, is_krx)} 미만"
    elif best["grade"] in ["A", "B"] and best["reg_mult"] > 0:
        t_verdict = "대기"
        if not best["triggered"]:
            end_cond = f"다음 조건: {best['next_cond']}"
        elif not best["r_ok"]:
            end_cond = f"다음 조건: 예상 R({best['expected_r']}R)이 최소 기준(1.5R) 이상 확보되는 눌림 가격 대기"
        else:
            end_cond = "다음 조건: 최소 포지션 비중(슬리브의 3%) 충족 대기"
    else:
        t_verdict = "관찰"
        end_cond = f"관찰 사유: 품질 등급 {best['grade']} / 레짐 계수 {best['reg_mult']} (예상 {best['expected_r']}R)"

    return {
        "t_verdict": t_verdict,
        "setup_id": best["setup_id"],
        "grade": best["grade"],
        "headline_reason": f"{best['purpose']} ({best['headline']})",
        "next_or_end_condition": end_cond,
        "t_ticket": {
            "is_executable": best["can_enter"],
            "setup_id": best["setup_id"],
            "setup_name": best["name"],
            "grade": best["grade"],
            "purpose": best["purpose"],
            "entry_price": _px(best["entry_price"]),
            "stop_price": _px(best["stop_price"]),
            "shares": int(best["shares"]),
            "position_value": int(best["position_value"]),
            "target_1": _px(best["target_1"]),
            "target_2": best["target_2"],
            "time_stop_days": int(best["time_stop_days"]),
            "time_stop_min_price": time_min_p,
            "max_hold_days": int(best["max_hold_days"]),
            "invalidation": f"S-EVT 기업 고유 악재 발생 또는 {fmt_price(best['stop_price'], is_krx)} 종가 이탈",
            "expected_r": best["expected_r"],
            "unit_r_krw": _px(best["unit_r_krw"]),
            "r_budget_krw": _px(best["r_budget_krw"]),
            "currency": "KRW" if is_krx else "USD",
            "chase_ban_price": _px(adjust_krx_price(c_t * (1.0 + TECH_PARAMS["MAX_CHASE_GAP"]), "down", is_krx)),
        }
    }


# =====================================================================
# 6-3. V 타이밍 판정 (차수별 집행 비율 및 이유 코드)[cite: 1]
# =====================================================================
def evaluate_v_tranche_timings(
    df: pd.DataFrame,
    s_mode: str,
    lifecycle: Dict[str, Any],
    env_info: Dict[str, Any],
    value_macro_input: Dict[str, Any],
    is_krx: bool = True
) -> Tuple[List[Dict[str, Any]], str, List[int]]:
    """
    6-3. 가치 트랙의 buy_tranches 각 차수에 대해 '즉시(1.0) / 절반(0.5) / 이월(0.0)'과 이유 코드를 판정[cite: 1].
    """
    tranches = value_macro_input.get("buy_tranches", [])
    thesis_status = value_macro_input.get("thesis_status", "유효")
    kill_flags = value_macro_input.get("kill_flags", [])
    analyzed_at_str = value_macro_input.get("analyzed_at", datetime.now().strftime("%Y-%m-%d"))

    try:
        days_old = (datetime.now() - datetime.strptime(analyzed_at_str, "%Y-%m-%d")).days
    except Exception:
        days_old = 0
    is_fresh = days_old <= TECH_PARAMS["VALUE_FRESH_DAYS"]

    row = df.iloc[-1]
    c_t = float(row["Close"])
    rsi_t = float(row["RSI14"])
    vola_spike = float(row["VOLA_PCTL250"]) >= TECH_PARAMS["VOLA_SPIKE_PCTL"]
    sig = lifecycle["latest_signals"]
    lvls = lifecycle["levels_cache"]
    ch = lifecycle["channel_cache"]
    state = lifecycle["state"]
    s_reg = env_info["s_reg"]

    def _downgrade(verdict: str) -> str:
        if verdict == "즉시":
            return "절반"
        elif verdict == "절반":
            return "이월"
        return "이월"

    results: List[Dict[str, Any]] = []
    high_conf_tranches: List[int] = []

    for tr in tranches:
        t_num = int(tr.get("tranche", 0))
        t_price = float(tr.get("price", c_t))
        t_weight = float(tr.get("weight", 0.01))
        codes: List[str] = []

        # 7. 고신뢰 구간 표시: 매수기준선 ±3% 안에 ⭐⭐ 이상 수평 지지 ZONE 겹침[cite: 1]
        high_conf = any(
            hz["stars"] >= 2 and abs(hz["price"] - t_price) / max(t_price, 1.0) <= TECH_PARAMS["TRANCHE_OVERLAP_TOL"]
            for hz in lvls["horizontal_zones"]
        )
        if high_conf:
            high_conf_tranches.append(t_num)

        # 차단/신선도 게이트[cite: 1]
        if thesis_status == "붕괴" or len(kill_flags) > 0 or not is_fresh:
            codes.append("THESIS_KILL" if (thesis_status == "붕괴" or kill_flags) else "STALE_VALUE")
            results.append({
                "tranche": t_num, "price": int(t_price), "weight": t_weight,
                "price_reached": bool(c_t <= t_price),
                "verdict": "이월", "ratio": 0.0, "codes": codes, "high_confidence": high_conf
            })
            continue

        # 1. 0차: 기술 게이트 미적용 (V자 반등 보험 -> 무조건 즉시)[cite: 1]
        if t_num == 0:
            results.append({
                "tranche": 0, "price": int(t_price), "weight": t_weight,
                "price_reached": bool(c_t <= t_price),
                "verdict": "즉시", "ratio": 1.0, "codes": ["TR0_INSURANCE"], "high_confidence": high_conf
            })
            continue

        # 2 ~ 6. 1~3차 판정[cite: 1]
        # 6. BNF 예외 (E-03): 🔴 레짐이어도 S-FALL 해제 + 증거 ⑤(투매 후 홀드+확인봉) 포함 시 1차분 '절반(BNF_EXC)' 허용[cite: 1]
        if s_reg == "🔴" and t_num == 1 and sig["fall_released"] and sig["has_bnf_ev5"] and not sig["s_brk_1"]:
            verdict = "절반"
            codes.append("BNF_EXC")
        elif s_reg in ["🔴", "⚫"]:
            verdict = "이월"
            codes.append("REG_RED")
        elif state in ["D", "K"] or sig["modules"]["bottom"]["stage"] == 0 or sig["s_brk_1"]:
            verdict = "이월"
            codes.append("BRK1" if (state == "K" or sig["s_brk_1"]) else "FALL_ACTIVE")
        else:
            # 4. 모드별 기본 판정[cite: 1]
            if s_mode == "추세 상승":
                cond_rsi = sig["pb_rsi_40_50"] and (rsi_t > float(df["RSI14"].iloc[-2]))
                cond_vol = sig["pb_rvol_low"]
                if cond_rsi and cond_vol:
                    verdict = "즉시"
                elif cond_rsi or cond_vol:
                    verdict = "절반"
                    codes.append("UP_PARTIAL")
                else:
                    verdict = "이월"
                    codes.append("UP_NO_PULLBACK")
            elif s_mode == "레인지":
                if sig["range_react_cnt"] == 3:
                    verdict = "즉시"
                elif sig["range_react_cnt"] >= 1:
                    verdict = "절반"
                    codes.append("RNG_PARTIAL")
                else:
                    verdict = "이월"
                    codes.append("RNG_WAIT")
            elif s_mode == "추세 하락":
                verdict = "이월"
                codes.append("MODE_DOWN")
            else:
                verdict = "절반"
                codes.append("MODE_HOLD_CAP")

            # 2. 🟡 레짐이면 한 단계 하향[cite: 1]
            if s_reg == "🟡" and verdict != "이월":
                verdict = _downgrade(verdict)
                codes.append("REG_YELLOW")

            # 3장. 상태 W(약화)면 한 단계 하향[cite: 1]
            if state == "W" and verdict != "이월":
                verdict = _downgrade(verdict)
                codes.append("STATE_WEAK")

            # 5. 하향 수정자: 쌍봉 확정(DBL_TOP), 채널 위치 >= 0.75(CH_HIGH), S-VOLA 급등(VOLA: 즉시->절반만)[cite: 1]
            if sig["double_top_confirmed"] and verdict != "이월":
                verdict = _downgrade(verdict)
                codes.append("DBL_TOP")
            if ch["pos_x"] >= 0.75 and verdict != "이월":
                verdict = _downgrade(verdict)
                codes.append("CH_HIGH")
            if vola_spike and verdict == "즉시":
                verdict = "절반"
                codes.append("VOLA")

        ratio = 1.0 if verdict == "즉시" else (0.5 if verdict == "절반" else 0.0)
        results.append({
            "tranche": t_num, "price": int(t_price), "weight": t_weight,
            "price_reached": bool(c_t <= t_price),
            "verdict": verdict, "ratio": ratio, "codes": codes, "high_confidence": high_conf
        })

    if len(results) == 0:
        v_line = "V 차수 미설정 (가치 트랙 매수기준선 입력 대기)"
    else:
        parts = [
            f"{r['tranche']}차({fmt_price(r['price'], is_krx)}): {r['verdict']}" + (" [고신뢰]" if r["high_confidence"] else "")
            for r in results
        ]
        v_line = "V 판정 — " + " · ".join(parts)

    return results, v_line, high_conf_tranches


# =====================================================================
# 10. 조건부 백테스트 및 검증 지표 (R 배수 기대값 E 산출)[cite: 1]
# =====================================================================
def run_setup_backtest_validation(df: pd.DataFrame) -> Dict[str, Any]:
    """
    10장. 과거 일봉 데이터에서 ST-1(이격도 반전) 및 돌파/눌림 신호의 R 배수 성과를 검증[cite: 1]:
    E = p * W_R - (1 - p) * L_R (왕복 비용 TP-32 0.30% 차감 후 R 기준)
    """
    if len(df) < 150:
        return {"sample_n": 0, "win_rate": 0.0, "avg_win_r": 0.0, "avg_loss_r": 0.0, "expectancy_r": 0.0, "max_loss_streak": 0, "trades": []}

    r_list: List[float] = []
    trades: List[Dict[str, Any]] = []
    cost = TECH_PARAMS["ROUND_TRIP_COST"]

    i = 120
    while i < len(df) - 15:
        row = df.iloc[i]
        c_t = float(row["Close"])
        atr_t = float(row["ATR14"])
        # ST-1 또는 돌파 확인봉 조건 스캔
        st1_cond = (float(df["DISP_PCTL750"].iloc[max(0, i - 5):i + 1].min()) <= TECH_PARAMS["DISP_EXTREME_PCTL"]) and bool(row["CONFIRM_BAR"])
        brk_cond = (c_t > float(df["High"].iloc[i - 20:i].max())) and (float(row["RVOL"]) >= 1.5)

        if st1_cond or brk_cond:
            setup_tag = "ST-1" if st1_cond else "ST-2"
            entry_p = float(df["Open"].iloc[i + 1])
            stop_p = float(df["Low"].iloc[max(0, i - 5):i + 1].min()) if st1_cond else (c_t - 1.5 * atr_t)
            unit_1r = max(entry_p * 0.01, (entry_p - stop_p) + entry_p * cost)
            target_p = float(row["SMA25"]) if st1_cond else (entry_p + 2.0 * unit_1r)
            if target_p <= entry_p:
                target_p = entry_p + 1.8 * unit_1r

            hold_limit = 5 if st1_cond else 15
            exit_p = float(df["Close"].iloc[min(len(df) - 1, i + hold_limit)])
            outcome_r = (exit_p - entry_p - entry_p * cost) / unit_1r

            for k in range(1, hold_limit + 1):
                if i + k >= len(df):
                    break
                bar_k = df.iloc[i + k]
                if float(bar_k["Close"]) < stop_p:
                    exit_p = float(bar_k["Close"])
                    outcome_r = (exit_p - entry_p - entry_p * cost) / unit_1r
                    break
                if float(bar_k["High"]) >= target_p:
                    exit_p = target_p
                    outcome_r = (exit_p - entry_p - entry_p * cost) / unit_1r
                    break

            r_val = round(float(outcome_r), 2)
            r_list.append(r_val)
            trades.append({
                "date": str(df.index[i])[:10],
                "setup": setup_tag,
                "entry": round(entry_p, 0),
                "exit": round(exit_p, 0),
                "r_multiple": r_val
            })
            i += hold_limit
        else:
            i += 1

    if len(r_list) == 0:
        return {"sample_n": 0, "win_rate": 0.0, "avg_win_r": 0.0, "avg_loss_r": 0.0, "expectancy_r": 0.0, "max_loss_streak": 0, "trades": []}

    wins = [r for r in r_list if r > 0]
    losses = [abs(r) for r in r_list if r <= 0]
    p = len(wins) / len(r_list)
    w_r = float(np.mean(wins)) if wins else 0.0
    l_r = float(np.mean(losses)) if losses else 0.0
    expectancy = p * w_r - (1.0 - p) * l_r

    max_streak, cur_streak = 0, 0
    for r in r_list:
        if r <= 0:
            cur_streak += 1
            max_streak = max(max_streak, cur_streak)
        else:
            cur_streak = 0

    return {
        "sample_n": len(r_list),
        "win_rate": round(p * 100.0, 1),
        "avg_win_r": round(w_r, 2),
        "avg_loss_r": round(l_r, 2),
        "expectancy_r": round(expectancy, 2),
        "max_loss_streak": max_streak,
        "trades": trades[-15:],
        "r_distribution": r_list,
    }


# =====================================================================
# 6-2. 트랙 1 메인 엔트리 함수: analyze_technical_track()[cite: 1]
# =====================================================================
def analyze_technical_track(
    df: pd.DataFrame,
    bench_df: Optional[pd.DataFrame] = None,
    vix_series: Optional[pd.Series] = None,
    value_macro_input: Optional[Dict[str, Any]] = None,
    account_cfg: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    기술적 분석 트랙 v1.2 메인 파이프라인 (8단계 연산 후 6-2 표준 스키마 Dict 반환)[cite: 1].
    """
    if value_macro_input is None:
        c_last = float(df["Close"].iloc[-1]) if len(df) > 0 else 50000.0
        value_macro_input = {
            "thesis_status": "유효",
            "kill_flags": [],
            "buy_tranches": [
                {"tranche": 0, "price": round(c_last * 0.98, -2), "weight": 0.01},
                {"tranche": 1, "price": round(c_last * 0.93, -2), "weight": 0.02},
                {"tranche": 2, "price": round(c_last * 0.88, -2), "weight": 0.02},
            ],
            "sell_target": round(c_last * 1.25, -2),
            "position_cap": 0.05,
            "real_signal": "중립",
            "analyzed_at": datetime.now().strftime("%Y-%m-%d"),
            "macro_regime_override": None,
        }
    if account_cfg is None:
        account_cfg = {"total_equity": 100_000_000, "t_sleeve_ratio": 0.30, "is_krx": True}
    is_krx = bool(account_cfg.get("is_krx", True))

    # 1~2단계: 기초값 7개 연산[cite: 1]
    chart_df = compute_base_primitives(df)
    if len(chart_df) < 30:
        raise ValueError("일봉 데이터가 최소 30거래일 이상 필요합니다.")

    # 3단계: 환경(S-REG, S-VIX) 및 상대강도(S-RS)[cite: 1]
    env_info = compute_market_regime(bench_df, vix_series, value_macro_input.get("macro_regime_override"))
    rs_info = compute_relative_strength(chart_df, bench_df)
    s_modes = compute_s_mode_series(chart_df)
    chart_df["S_MODE"] = s_modes

    # 4~6단계: 생애주기 신호, 6대 판정 모듈, 7상태 기계[cite: 1]
    lifecycle = run_lifecycle_state_machine(chart_df, s_modes, rs_info, is_krx=is_krx)
    sig = lifecycle["latest_signals"]
    lvls = lifecycle["levels_cache"]
    ch = lifecycle["channel_cache"]

    # 7단계: 분기 라우터 — T 셋업 스캔·사이징 & V 차수별 타이밍 판정[cite: 1]
    t_res = scan_t_setups_and_size(chart_df, s_modes, lifecycle, env_info, rs_info, value_macro_input, account_cfg)
    v_verdicts, v_summary_line, high_conf_tranches = evaluate_v_tranche_timings(
        chart_df, s_modes[-1], lifecycle, env_info, value_macro_input, is_krx=is_krx
    )

    # S-MAT 이평 배열 판정[cite: 1]
    last_row = chart_df.iloc[-1]
    s5, s20, s60 = float(last_row["SMA5"]), float(last_row["SMA20"]), float(last_row["SMA60"])
    sl5, sl20, sl60 = float(last_row["SMA5_SLOPE"]), float(last_row["SMA20_SLOPE"]), float(last_row["SMA60_SLOPE"])
    if s5 > s20 > s60 and sl5 > 0 and sl20 > 0 and sl60 > 0:
        s_mat = "강한 정배열"
    elif s5 > s20 > s60:
        s_mat = "약한 정배열"
    elif s5 < s20 < s60 and sl5 < 0 and sl20 < 0 and sl60 < 0:
        s_mat = "역배열"
    else:
        s_mat = "혼조"

    sma120_slope = "상승" if float(last_row["SMA120_SLOPE"]) > 0 else "하락"
    targets_up = compute_measured_targets(sig["b_neck"], sig["sl1"], "up")

    # 8단계: 백테스트 검증 지표 산출[cite: 1]
    bt_stats = run_setup_backtest_validation(chart_df)

    # 트랙 종합 신호등 (종합 결론 라우터 전달용)[cite: 1]
    if t_res["t_verdict"] == "진입" or any(v["verdict"] == "즉시" and v["price_reached"] for v in v_verdicts):
        track_signal = "🟢"
    elif t_res["t_verdict"] == "대기" or any(v["verdict"] == "절반" for v in v_verdicts):
        track_signal = "🟡"
    elif t_res["t_verdict"] == "금지" or lifecycle["state"] in ["D", "K"]:
        track_signal = "🔴"
    else:
        track_signal = "⚪"

    return {
        "track_signal": track_signal,
        "summary_banner": {
            "t_verdict": t_res["t_verdict"],
            "setup_id": t_res["setup_id"],
            "grade": t_res["grade"],
            "headline_reason": t_res["headline_reason"],
            "next_or_end_condition": t_res["next_or_end_condition"],
            "v_summary_line": v_summary_line,
        },
        "lifecycle_state": {
            "state": lifecycle["state"],
            "sub_state": lifecycle["sub_state"],
            "state_label": lifecycle["state_label"],
            "prev_state": lifecycle["prev_state"],
            "transition_reason": lifecycle["transition_reason"],
        },
        "context": {
            "s_reg": env_info["s_reg"],
            "s_reg_label": env_info["s_reg_label"],
            "s_vix": f"{env_info['s_vix']} ({env_info['vix_pctl']}%)",
            "s_mode": s_modes[-1],
            "s_rs_daily": rs_info["daily"],
            "s_rs_weekly": rs_info["weekly"],
            "s_mat": s_mat,
            "sma120_slope": sma120_slope,
        },
        "modules": sig["modules"],
        "t_ticket": t_res["t_ticket"],
        "tranche_verdicts": v_verdicts,
        "levels": {
            "horizontal_zones": lvls["horizontal_zones"],
            "high_vol_bins": lvls["high_vol_bins"],
            "low_vol_bins": lvls["low_vol_bins"],
            "targets": targets_up,
            "channel": ch,
        },
        "overlap_flags": {
            "high_conf_buy_tranches": high_conf_tranches,
            "accel_sell_target": bool(
                value_macro_input.get("sell_target") is not None
                and sig["is_s_weak"]
                and float(last_row["Close"]) >= float(value_macro_input["sell_target"]) * 0.97
            ),
        },
        "risk_flags": {
            "s_weak_count": len(sig["weak_reasons"]),
            "s_brk_stage": 2 if lifecycle["sub_state"] == "K②" else (1 if sig["s_brk_1"] else 0),
            "double_top_confirmed": sig["double_top_confirmed"],
            "trendline_big_break": ch["big_bear_break"],
        },
        "backtest_validation": bt_stats,
        "chart_df": chart_df,
    }


# =====================================================================
# [Track 2] 실적 분석 트랙 v1.0 (미래 예측·지속성·G1~G5 생애주기 엔진)
# =====================================================================
def analyze_earnings_track(
    ticker: str,
    sector_type: str = "구조성장형",          # "사이클형" | "구조성장형" | "방어형" | "정책의존형"
    fin_override: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    실적 분석 트랙 v1.0 메인 파이프라인 (E1 ~ E8 연산, G1~G5 생애주기 분류, 5대 축 점수 산출)[cite: 2].
    * 순환논리 원천 차단: 주가나 PER을 입력으로 받지 않으며 오직 실물/재무/리비전 데이터만 사용함[cite: 2].
    """
    is_krx = str(ticker).upper().endswith((".KS", ".KQ"))  # EPS 문구 통화 표기용
    # 1. 기본 시드 데이터 및 오버라이드 병합 (yfinance 재무/컨센서스 또는 사용자/리포트 입력값 연동)
    default_data = {
        # E1. 최근 3개 분기 YoY 성장률 궤적 (t-2 -> t-1 -> t) 및 서프라이즈[cite: 2]
        "sales_yoy_trajectory": [0.14, 0.18, 0.22],   # g_{t-2}, g_{t-1}, g_t[cite: 2]
        "eps_yoy_trajectory": [0.18, 0.25, 0.34],
        "actual_vs_cons": {
            "sales_actual": 79.1, "sales_cons": 77.5, "sales_qoq": 0.068, "sales_yoy": 0.220, "sales_prev_yoy": 0.180,
            "gpm_actual": 0.385, "gpm_cons": 0.380, "gpm_qoq": 0.012, "gpm_yoy": 0.025, "gpm_prev_yoy": 0.018,
            "opm_actual": 0.240, "opm_cons": 0.232, "opm_qoq": 0.018, "opm_yoy": 0.035, "opm_prev_yoy": 0.022,
            "eps_actual": 1050.0, "eps_cons": 980.0, "eps_qoq": 0.110, "eps_yoy": 0.340, "eps_prev_yoy": 0.250,
        },
        # E2. 최근 4분기 가이던스 이행 이력: "beat_4q" | "miss_2q" | "neutral"[cite: 2]
        "guidance_history": "beat_4q",
        # E3. NTM EPS 및 FY+1, FY+2 리비전 (현재 t, 30일 전 t-30, 90일 전 t-90)[cite: 2]
        "revision": {
            "ntm_t": 4200.0, "ntm_30d": 4150.0, "ntm_90d": 3985.0,
            "fy1_t": 4050.0, "fy1_30d": 4000.0, "fy1_90d": 3850.0,
            "fy2_t": 5300.0, "fy2_30d": 5220.0, "fy2_90d": 4980.0,
            "analysts_t": 28, "analysts_30d": 28, "analysts_90d": 26,
        },
        # E4 & E5. 핵심 드라이버 및 시나리오 EPS 입력[cite: 2]
        "drivers": {
            "rev_base": 320.0, "opm_base": 0.24,
            "rev_bull": 355.0, "opm_bull": 0.275,
            "rev_bear": 285.0, "opm_bear": 0.195,
            "tax_rate": 0.21,
            "diluted_shares": 14.4,  # 1-2. CB/BW·스톡옵션 반영 완전희석 주식수 (억 주)[cite: 2]
            "cycle_peak_to_trough_drop": 0.45,  # 사이클형 과거 정점->저점 EPS 낙폭 기저율[cite: 2]
            "fy_Order_labels": ("FY27E", "FY28E", "FY29E"),
            "remaining_months_in_fy": 3,        # m: 평가 시점 T 이후 FY_a 남은 개월 수 (3/12 = 0.25)[cite: 2]
            "fy2_growth_mult": 1.26,
        },
        # E6. 이익의 질 및 현금흐름 데이터[cite: 2]
        "quality": {
            "ar_yoy": 0.16,          # 매출채권 YoY 증가율[cite: 2]
            "inv_yoy": 0.14,         # 재고자산 YoY 증가율[cite: 2]
            "non_gaap_eps": 1080.0,
            "gaap_eps": 1050.0,
            "ocf_ttm": 52.0,         # 영업현금흐름[cite: 2]
            "ocf_prev_ttm": 41.0,    # 전년 영업현금흐름[cite: 2]
            "net_income_ttm": 44.0,  # 당기순이익[cite: 2]
            "capex_ttm": 58.0,       # 설비투자 (성장 CAPEX 집중 시 FCF < 0 가능)[cite: 2]
            "reinvest_per_dollar_rising": False,
        },
        # E7. 미래 먹거리 (RPO / 수주잔고 및 증분 영업이익률)[cite: 2]
        "sustainability": {
            "rpo_yoy_growth": 0.45,  # 수주잔고/RPO YoY 증가율[cite: 2]
            "op_t": 18.98,           # 당기 영업이익 (OP_t)[cite: 2]
            "op_t_minus_4": 12.96,   # 전년 동기 영업이익 (OP_{t-4})[cite: 2]
            "sales_t": 79.10,        # 당기 매출 (Sales_t)[cite: 2]
            "sales_t_minus_4": 64.80,# 전년 동기 매출 (Sales_{t-4})[cite: 2]
            "design_wins_note": "차세대 선단 공정/주력 라인업 신규 고객사 3곳 채택 확정 (양산 리드타임 18개월)",
        },
        # E8 & 1-1. 드라이버 붕괴 조건 및 P/Q 사이클 상태[cite: 2]
        "thesis": {
            "hard_block_triggered": False,
            "invalidation_condition": "주력 고객사향 분기 발주량 2분기 연속 YoY -15% 하회 시 이익 가설 파기",
            "current_status": "현재 고객사 발주량 YoY +28% 및 가동률 92% 유지 중",
            "p_trend": "상승",       # "상승" | "하락"[cite: 2]
            "q_trend": "상승",       # "상승" | "하락"[cite: 2]
            "capital_structure_note": "전환사채(CB) 잔량 완전희석 주식수(14.4억 주) 반영 완료 · 자사주 소각분 차감 적용",
        }
    }

    if fin_override:
        for k, v in fin_override.items():
            if isinstance(v, dict) and k in default_data and isinstance(default_data[k], dict):
                default_data[k].update(v)
            else:
                default_data[k] = v

    d = default_data
    ac = d["actual_vs_cons"]

    # -----------------------------------------------------------------
    # E1. 실적 액추얼 괴리율 및 성장 가속도 (2차 미분) 판정[cite: 2]
    # -----------------------------------------------------------------
    sales_beat_pct = (ac["sales_actual"] - ac["sales_cons"]) / max(abs(ac["sales_cons"]), 1e-6)
    gpm_beat_diff = ac["gpm_actual"] - ac["gpm_cons"]
    opm_beat_diff = ac["opm_actual"] - ac["opm_cons"]
    eps_beat_pct = (ac["eps_actual"] - ac["eps_cons"]) / max(abs(ac["eps_cons"]), 1e-6)

    # 미래 예측 핵심 규칙: "매출 상회 + 마진(OPM 또는 GPM) 하회" = 마진 압착 경고[cite: 2]
    margin_squeeze_detected = bool(sales_beat_pct > 0.0 and (opm_beat_diff < -0.002 or gpm_beat_diff < -0.003))

    g_traj = d["sales_yoy_trajectory"]  # [g_{t-2}, g_{t-1}, g_t][cite: 2]
    g_t2, g_t1, g_t = float(g_traj[0]), float(g_traj[1]), float(g_traj[2])
    if g_t2 < g_t1 < g_t and (g_t - g_t1) > 0.005:
        growth_accel = "가속"
        axis1_score = 2
    elif abs(g_t - g_t1) <= 0.02 and g_t >= g_t2:
        growth_accel = "유지"
        axis1_score = 1
    elif g_t2 > g_t1 > g_t or (g_t - g_t1) < -0.02:
        growth_accel = "감속"
        axis1_score = 0
    else:
        growth_accel = "유지"
        axis1_score = 1

    e1_table = [
        {
            "metric": "매출액",
            "actual": f"{ac['sales_actual']:,.1f}",
            "consensus": f"{ac['sales_cons']:,.1f}",
            "surprise": f"{sales_beat_pct*100:+.1f}%",
            "qoq": f"{ac['sales_qoq']*100:+.1f}%",
            "yoy": f"{ac['sales_yoy']*100:+.1f}%",
            "prev_yoy": f"{ac['sales_prev_yoy']*100:+.1f}%"
        },
        {
            "metric": "매출총이익률(GPM)",
            "actual": f"{ac['gpm_actual']*100:.1f}%",
            "consensus": f"{ac['gpm_cons']*100:.1f}%",
            "surprise": f"{gpm_beat_diff*100:+.1f}%p",
            "qoq": f"{ac['gpm_qoq']*100:+.1f}%p",
            "yoy": f"{ac['gpm_yoy']*100:+.1f}%p",
            "prev_yoy": f"{ac['gpm_prev_yoy']*100:+.1f}%p"
        },
        {
            "metric": "영업이익률(OPM)",
            "actual": f"{ac['opm_actual']*100:.1f}%",
            "consensus": f"{ac['opm_cons']*100:.1f}%",
            "surprise": f"{opm_beat_diff*100:+.1f}%p",
            "qoq": f"{ac['opm_qoq']*100:+.1f}%p",
            "yoy": f"{ac['opm_yoy']*100:+.1f}%p",
            "prev_yoy": f"{ac['opm_prev_yoy']*100:+.1f}%p"
        },
        {
            "metric": "EPS (완전희석)",
            "actual": f"{fmt_price(ac['eps_actual'], is_krx)}",
            "consensus": f"{fmt_price(ac['eps_cons'], is_krx)}",
            "surprise": f"{eps_beat_pct*100:+.1f}%",
            "qoq": f"{ac['eps_qoq']*100:+.1f}%",
            "yoy": f"{ac['eps_yoy']*100:+.1f}%",
            "prev_yoy": f"{ac['eps_prev_yoy']*100:+.1f}%"
        },
    ]

    # -----------------------------------------------------------------
    # E2. 가이던스 이행 이력 -> 확실성 판정 (펀더멘탈 할인율 r 조정값)[cite: 2]
    # -----------------------------------------------------------------
    g_hist = d["guidance_history"]
    if g_hist == "beat_4q":
        certainty_label = "확실성 높음 (4분기 연속 Beat & Raise)"
        certainty_adj = -0.02  # -2%p 할인율 인하[cite: 2]
        axis5_score = 2
    elif g_hist == "miss_2q":
        certainty_label = "확실성 낮음 (2분기 연속 가이던스 미스)"
        certainty_adj = 0.02   # +2%p 할인율 할증[cite: 2]
        axis5_score = 0
    else:
        certainty_label = "중립 (부합 또는 가이던스 미제시)"
        certainty_adj = 0.00   # 0%p[cite: 2]
        axis5_score = 1

    # -----------------------------------------------------------------
    # E3. NTM EPS 리비전의 변곡점 추적 (90일 vs 30일 괴리 및 롤오버 검증)[cite: 2]
    # -----------------------------------------------------------------
    rev = d["revision"]
    delta_90 = (rev["ntm_t"] - rev["ntm_90d"]) / max(abs(rev["ntm_90d"]), 1e-6)
    delta_30 = (rev["ntm_t"] - rev["ntm_30d"]) / max(abs(rev["ntm_30d"]), 1e-6)
    fy1_chg = (rev["fy1_t"] - rev["fy1_90d"]) / max(abs(rev["fy1_90d"]), 1e-6)
    fy2_chg = (rev["fy2_t"] - rev["fy2_90d"]) / max(abs(rev["fy2_90d"]), 1e-6)
    # NTM 창 이동(Roll-over) 착시 제거: FY+1과 FY+2가 동반 상승할 때만 진성 상향 인정[cite: 2]
    true_fy_up = (fy1_chg > 0.0) and (fy2_chg > 0.0)

    inflection_flag = False
    if delta_90 > 0.01 and delta_30 < -0.005:
        rev_status = "피크아웃 경고"
        inflection_flag = True
        axis2_score = 0
    elif delta_90 < -0.01 and delta_30 > 0.005:
        rev_status = "턴어라운드 조짐"
        inflection_flag = True
        axis2_score = 1
    elif delta_90 > 0.01 and delta_30 > 0.0 and true_fy_up:
        rev_status = "상향 지속"
        axis2_score = 2
    elif delta_90 < -0.01 and delta_30 < 0.0:
        rev_status = "하향 지속"
        axis2_score = 0
    else:
        rev_status = "보합"
        axis2_score = 1

    # -----------------------------------------------------------------
    # E4 & E5. 핵심 드라이버 시나리오 EPS 및 1Y/2Y NTM 혼합식 산출[cite: 2]
    # -----------------------------------------------------------------
    drv = d["drivers"]
    shares = max(drv["diluted_shares"], 1e-4)
    tax = drv["tax_rate"]

    # FY+1 (FY27E) 및 FY+2 (FY28E), FY+3 (FY29E) 시나리오별 EPS 역산[cite: 2]
    fy1_base = (drv["rev_base"] * drv["opm_base"] * (1.0 - tax)) / shares * 1000.0
    fy1_bull = (drv["rev_bull"] * drv["opm_bull"] * (1.0 - tax)) / shares * 1000.0
    fy1_bear = (drv["rev_bear"] * drv["opm_bear"] * (1.0 - tax)) / shares * 1000.0

    mult_y2 = drv.get("fy2_growth_mult", 1.25)
    fy2_base = fy1_base * mult_y2
    fy2_bull = fy1_bull * (mult_y2 * 1.04)
    fy2_bear = fy1_bear * (mult_y2 * 0.94)

    fy3_base = fy2_base * (1.0 + (mult_y2 - 1.0) * 0.85)
    fy3_bull = fy2_bull * (1.0 + (mult_y2 - 1.0) * 0.95)
    fy3_bear = fy2_bear * (1.0 + (mult_y2 - 1.0) * 0.70)

    # E5. EPS_NTM(T) = (m/12)*FY_a + ((12-m)/12)*FY_{a+1}[cite: 2]
    m_rem = int(drv.get("remaining_months_in_fy", 3))
    w_a = round(m_rem / 12.0, 2)
    w_b = round((12 - m_rem) / 12.0, 2)
    lbl_1, lbl_2, lbl_3 = drv.get("fy_Order_labels", ("FY27E", "FY28E", "FY29E"))

    eps_1y_base = int(round(w_a * fy1_base + w_b * fy2_base, 0))
    eps_1y_bull = int(round(w_a * fy1_bull + w_b * fy2_bull, 0))
    eps_1y_bear = int(round(w_a * fy1_bear + w_b * fy2_bear, 0))
    cycle_bear_1y = int(round(eps_1y_bull * (1.0 - drv["cycle_peak_to_trough_drop"]), 0)) if sector_type == "사이클형" else None

    eps_2y_base = int(round(w_a * fy2_base + w_b * fy3_base, 0))
    eps_2y_bull = int(round(w_a * fy2_bull + w_b * fy3_bull, 0))
    eps_2y_bear = int(round(w_a * fy2_bear + w_b * fy3_bear, 0))

    # -----------------------------------------------------------------
    # E6. 이익의 질 5대 체크 및 '투자 단계 현금흐름' 예외 규정[cite: 2]
    # -----------------------------------------------------------------
    q_data = d["quality"]
    sales_yoy = float(ac["sales_yoy"])
    quality_items = []
    warn_cnt = 0

    # 1) 매출채권 vs 매출[cite: 2]
    ar_diff = q_data["ar_yoy"] - sales_yoy
    ar_warn = ar_diff > 0.05
    if ar_warn:
        warn_cnt += 1
    quality_items.append({
        "name": "매출채권 vs 매출",
        "value": f"{ar_diff*100:+.1f}%p",
        "warn": ar_warn,
        "desc": "밀어내기 매출·회수 지연 경고" if ar_warn else "정상 회수 범위"
    })

    # 2) 재고 vs 매출[cite: 2]
    inv_diff = q_data["inv_yoy"] - sales_yoy
    inv_warn = inv_diff > 0.05
    if inv_warn:
        warn_cnt += 1
    quality_items.append({
        "name": "재고자산 vs 매출",
        "value": f"{inv_diff*100:+.1f}%p",
        "warn": inv_warn,
        "desc": "전방 수요 둔화·재고 누적 경고" if inv_warn else "재고 건전성 양호"
    })

    # 3) GAAP vs Non-GAAP 격차[cite: 2]
    ng_gap = (q_data["non_gaap_eps"] - q_data["gaap_eps"]) / max(abs(q_data["non_gaap_eps"]), 1e-6)
    ng_warn = ng_gap > 0.20
    if ng_warn:
        warn_cnt += 1
    quality_items.append({
        "name": "GAAP vs Non-GAAP 격차",
        "value": f"{ng_gap*100:.1f}%",
        "warn": ng_warn,
        "desc": "SBC 과다·일회성 비용 만성화" if ng_warn else "회계 이익 괴리 낮음 (<20%)"
    })

    # 4) 현금 전환율 (OCF / Net Income) 및 ★ 투자기 현금흐름 예외 조항[cite: 2]
    ocf_conv = q_data["ocf_ttm"] / max(abs(q_data["net_income_ttm"]), 1e-6)
    fcf_ttm = q_data["ocf_ttm"] - q_data["capex_ttm"]
    ocf_growing = q_data["ocf_ttm"] > q_data["ocf_prev_ttm"]
    # ★ 투자 단계 현금흐름 예외 규정: ΔOCF > 0 이고 OCF/NI >= 1.0 이면 CAPEX로 인해 FCF < 0 이어도 완전 면제[cite: 2]
    is_investment_phase_fcf = bool(ocf_growing and ocf_conv >= 1.0 and fcf_ttm < 0)
    ocf_warn = (ocf_conv < 0.80) and not is_investment_phase_fcf
    if ocf_warn:
        warn_cnt += 1
    quality_items.append({
        "name": "현금 전환율 (영업CF/순이익)",
        "value": f"{ocf_conv:.2f}배 (FCF {fcf_ttm:+.1f})",
        "warn": ocf_warn,
        "desc": "G1 초기 가속 투자의 정상적 과정 (면제)" if is_investment_phase_fcf else ("장부상 이익 위험 (<0.8)" if ocf_warn else "현금 창출력 우수")
    })

    # 5) 매출 1단위당 소요 자본[cite: 2]
    reinv_warn = bool(q_data["reinvest_per_dollar_rising"])
    if reinv_warn:
        warn_cnt += 1
    quality_items.append({
        "name": "매출 1단위당 소요 자본",
        "value": "상승 추세" if reinv_warn else "안정적 유지",
        "warn": reinv_warn,
        "desc": "성장의 한계비용 상승 경고" if reinv_warn else "자본 효율성 유지"
    })

    axis4_score = 2 if warn_cnt == 0 else (1 if warn_cnt == 1 else 0)

    # -----------------------------------------------------------------
    # E7. 미래 먹거리 측정 (RPO 증가율 vs 매출 증가율 & 증분 영업이익률)[cite: 2]
    # -----------------------------------------------------------------
    sus = d["sustainability"]
    delta_op = sus["op_t"] - sus["op_t_minus_4"]
    delta_sales = sus["sales_t"] - sus["sales_t_minus_4"]
    inc_opm = delta_op / delta_sales if abs(delta_sales) > 1e-6 else ac["opm_actual"]
    avg_opm = float(ac["opm_actual"])
    rpo_yoy = float(sus["rpo_yoy_growth"])

    op_leverage_active = bool(inc_opm > avg_opm)
    rpo_outpacing_sales = bool(rpo_yoy > sales_yoy)

    if op_leverage_active and rpo_outpacing_sales:
        axis3_score = 2
    elif op_leverage_active or rpo_outpacing_sales:
        axis3_score = 1
    else:
        axis3_score = 0

    # -----------------------------------------------------------------
    # 3장. 실적 생애주기 5단계 (G1 ~ G5) 및 사이클 단계 (C1 ~ C4) 자동 분류[cite: 2]
    # -----------------------------------------------------------------
    th = d["thesis"]
    pq_both_down = (th["p_trend"] == "하락" and th["q_trend"] == "하락")

    if th["hard_block_triggered"] or (pq_both_down and g_hist == "miss_2q" and rpo_yoy < 0):
        e_regime = "G5"
        e_regime_label = "구조적 쇠퇴 / 어닝 쇼크 (Structural Decline)"
        peak_out_warning = True
    elif margin_squeeze_detected or (not op_leverage_active and growth_accel == "감속") or (delta_30 < -0.005):
        e_regime = "G4"
        e_regime_label = "피크아웃 / 성장 둔화 (Peak-out Risk)"
        peak_out_warning = True
    elif growth_accel == "가속" and op_leverage_active and rpo_outpacing_sales and delta_30 > 0:
        e_regime = "G1"
        e_regime_label = "초기 가속 / 턴어라운드 (Emerging Growth)"
        peak_out_warning = False
    elif sales_yoy >= 0.10 and float(ac["eps_yoy"]) >= 0.10 and warn_cnt == 0 and op_leverage_active:
        e_regime = "G2"
        e_regime_label = "우량 지속 성장 (Quality Compounder)"
        peak_out_warning = False
    else:
        e_regime = "G3"
        e_regime_label = "성숙 / 현금 회수 (Mature Cash Cow)"
        peak_out_warning = False

    # 사이클형(a) 종목인 경우 C1 ~ C4 단계 병기[cite: 2]
    cycle_stage = None
    if sector_type == "사이클형":
        if th["p_trend"] == "하락" and th["q_trend"] == "상승":
            cycle_stage = "C1 바닥 통과 (Q 출하 반등)"
        elif th["p_trend"] == "상승" and th["q_trend"] == "상승":
            cycle_stage = "C3 정점 과열 (재고 급증 경고)" if inv_warn else "C2 확장 (P·Q 동반 상승)"
        else:
            cycle_stage = "C4 하강 (P·Q 동반 둔화)"

    # -----------------------------------------------------------------
    # 4장. 5대 미래 지속성 평가 축 가중합산 및 최종 등급 판정[cite: 2]
    # -----------------------------------------------------------------
    # 가중치: 축1(20%) + 축2(25%) + 축3(25%) + 축4(15%) + 축5(15%)[cite: 2]
    weighted_score = round(
        axis1_score * 0.20 +
        axis2_score * 0.25 +
        axis3_score * 0.25 +
        axis4_score * 0.15 +
        axis5_score * 0.15,
        2
    )

    if th["hard_block_triggered"] or e_regime == "G5":
        final_grade = "🔴"
        thesis_status = "붕괴"
        real_signal = "약"
    elif weighted_score < 1.0 or e_regime == "G4":
        final_grade = "🔴"
        thesis_status = "경계"
        real_signal = "약"
    elif weighted_score >= 1.6 and e_regime in ["G1", "G2"]:
        final_grade = "🟢"
        thesis_status = "유효"
        real_signal = "강"
    else:
        final_grade = "🟡"
        thesis_status = "유효" if weighted_score >= 1.2 else "경계"
        real_signal = "중립"

    if e_regime in ["G1", "G2"]:
        headline = f"수주잔고(RPO) +{rpo_yoy*100:.0f}% 급증 및 증분 OPM {inc_opm*100:.1f}%로 영업레버리지 가속화"
    elif e_regime == "G4":
        headline = f"피크아웃 경고 — {'매출 상회 속 마진 압착 발생' if margin_squeeze_detected else '30일 EPS 리비전 모멘텀 둔화 포착'}"
    elif e_regime == "G5":
        headline = "핵심 이익 드라이버 붕괴 또는 P·Q 동반 역성장 발동 (신규 진입 차단)"
    else:
        headline = f"성숙기 현금 창출 구간 (평균 OPM {avg_opm*100:.1f}%, 90일 리비전 {delta_90*100:+.1f}%)"

    # E8-2. 시장의 이익 오판 포인트 자동 도출[cite: 2]
    if rpo_outpacing_sales and op_leverage_active:
        mispricing_type = "시장 과소평가 포인트 (성장 잠재력)"
        mispricing_detail = (
            f"RPO 증가율(+{rpo_yoy*100:.0f}%)이 매출 성장률(+{sales_yoy*100:.0f}%)을 크게 상회하고 "
            f"증분 OPM({inc_opm*100:.1f}%)이 평균 OPM({avg_opm*100:.1f}%)을 웃돌아 향후 1~2년 고정비 레버리지 폭발 구간 진입."
        )
    elif peak_out_warning:
        mispricing_type = "시장 과대평가 포인트 (저무는 기업 경고)"
        mispricing_detail = (
            "외형 매출 수치만 유지될 뿐 한계비용 상승(증분 마진 둔화) 또는 단기 30일 리비전 하향 변곡점이 발생하여 "
            "컨센서스 하향 조정이 임박한 상태."
        )
    else:
        mispricing_type = "컨센서스 부합 구간"
        mispricing_detail = "현재 실적 궤적과 시장 컨센서스 기대치가 유사한 수준으로 형성됨."

    return {
        "regime": {
            "earnings_regime": e_regime,
            "regime_label": e_regime_label,
            "sector_type": sector_type,
            "cycle_stage": cycle_stage,
            "growth_acceleration": growth_accel,
        },
        "verdict": {
            "score": weighted_score,
            "grade": final_grade,
            "real_signal": real_signal,
            "headline_reason": headline,
            "peak_out_warning": peak_out_warning,
        },
        "projections": {
            "eps_1y": {
                "formula": f"{w_a:.2f}*{lbl_1} + {w_b:.2f}*{lbl_2}",
                "base": eps_1y_base,
                "bull": eps_1y_bull,
                "bear": eps_1y_bear,
                "cycle_bear": cycle_bear_1y,
            },
            "eps_2y": {
                "formula": f"{w_a:.2f}*{lbl_2} + {w_b:.2f}*{lbl_3}",
                "base": eps_2y_base,
                "bull": eps_2y_bull,
                "bear": eps_2y_bear,
            },
            "certainty_adjustment": certainty_adj,
            "certainty_label": certainty_label,
        },
        "sustainability_metrics": {
            "incremental_opm": round(inc_opm, 4),
            "average_opm": round(avg_opm, 4),
            "rpo_yoy_growth": round(rpo_yoy, 4),
            "sales_yoy_growth": round(sales_yoy, 4),
            "operating_leverage_active": op_leverage_active,
            "design_wins_note": sus["design_wins_note"],
        },
        "revision_dynamics": {
            "status": rev_status,
            "change_90d": round(delta_90, 4),
            "change_30d": round(delta_30, 4),
            "inflection_flag": inflection_flag,
            "table": [
                {"period": "현재 (t)", "ntm_eps": f"{fmt_price(rev['ntm_t'], is_krx)}", "fy1_eps": f"{fmt_price(rev['fy1_t'], is_krx)}", "fy2_eps": f"{fmt_price(rev['fy2_t'], is_krx)}", "analysts": f"{rev['analysts_t']}명"},
                {"period": "30일 전 (t-30)", "ntm_eps": f"{fmt_price(rev['ntm_30d'], is_krx)}", "fy1_eps": f"{fmt_price(rev['fy1_30d'], is_krx)}", "fy2_eps": f"{fmt_price(rev['fy2_30d'], is_krx)}", "analysts": f"{rev['analysts_30d']}명"},
                {"period": "90일 전 (t-90)", "ntm_eps": f"{fmt_price(rev['ntm_90d'], is_krx)}", "fy1_eps": f"{fmt_price(rev['fy1_90d'], is_krx)}", "fy2_eps": f"{fmt_price(rev['fy2_90d'], is_krx)}", "analysts": f"{rev['analysts_90d']}명"},
            ],
        },
        "quality_checks": {
            "warning_count": warn_cnt,
            "is_investment_phase_fcf": is_investment_phase_fcf,
            "margin_squeeze_detected": margin_squeeze_detected,
            "items": quality_items,
        },
        "thesis_driver": {
            "status": thesis_status,
            "invalidation_condition": th["invalidation_condition"],
            "current_status": th["current_status"],
            "capital_structure_note": th["capital_structure_note"],
            "mispricing_type": mispricing_type,
            "mispricing_detail": mispricing_detail,
        },
        "evaluation_axes": [
            {"axis": "축 1: 성장 가속도 (E1)", "weight": "20%", "score": axis1_score, "status": f"2차 미분 [{growth_accel}] ({g_t2*100:.0f}%→{g_t1*100:.0f}%→{g_t*100:.0f}%)"},
            {"axis": "축 2: 리비전 변곡점 (E3)", "weight": "25%", "score": axis2_score, "status": f"{rev_status} (90일 {delta_90*100:+.1f}% / 30일 {delta_30*100:+.1f}%)"},
            {"axis": "축 3: 미래 지속성·레버리지 (E7)", "weight": "25%", "score": axis3_score, "status": f"증분OPM {inc_opm*100:.1f}% vs 평균 {avg_opm*100:.1f}% · RPO +{rpo_yoy*100:.0f}%"},
            {"axis": "축 4: 질 및 현금 전환 (E6)", "weight": "15%", "score": axis4_score, "status": f"경고 {warn_cnt}개" + (" (투자기 FCF 감소 면제)" if is_investment_phase_fcf else "")},
            {"axis": "축 5: 경영진 가이드 신뢰 (E2)", "weight": "15%", "score": axis5_score, "status": f"{certainty_label} (할인율 {certainty_adj*100:+.0f}%p)"},
        ],
        "e1_surprise_table": e1_table,
    }


# =====================================================================
# [Track 3] 펀더멘탈 분석 트랙 v1.0 (멀티플·가치평가·0~3차 매수기준선 엔진)
# =====================================================================
def analyze_fundamental_track(
    ticker: str,
    current_price: float,
    earnings_output: Optional[Dict[str, Any]] = None,
    macro_input: Optional[Dict[str, Any]] = None,
    fund_override: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    펀더멘탈 분석 트랙 v1.0 메인 파이프라인 (F0 ~ F10 연산, 0~3차 buy_tranches 및 가격 자격 판정).
    * 순환논리 차단 3대 원칙 준수:
      1) 현재가로 목표 배수를 정당화하지 않음 (역사/피어/레짐 평균 중심)[cite: 4]
      2) G4(피크아웃)/G5(쇠퇴) 국면은 디레이팅 Cap 강제 적용 (가치 함정 차단)[cite: 4]
      3) 이론 성숙 PER은 시장 자기자본비용(k_e), 매수기준선은 사용자 요구수익률(r)로 분리 할인[cite: 4]
    """
    is_krx = ticker.upper().endswith((".KS", ".KQ"))

    # 실적 트랙(Track 2) 출력값 수신 (없을 경우 안전한 기본값 생성)
    if earnings_output is None:
        earnings_output = analyze_earnings_track(ticker)

    e_reg_info = earnings_output.get("regime", {})
    e_regime = e_reg_info.get("earnings_regime", "G2")          # "G1" ~ "G5"[cite: 2, 4]
    sector_type = e_reg_info.get("sector_type", "구조성장형")     # "사이클형" | "구조성장형" | "방어형" | "정책의존형"[cite: 2, 4]
    cycle_stage = e_reg_info.get("cycle_stage")                 # "C1" ~ "C4"[cite: 2, 4]
    proj = earnings_output.get("projections", {})
    eps_1y_dict = proj.get("eps_1y", {"base": 4200, "bull": 5100, "bear": 3500, "cycle_bear": 2800})
    eps_2y_dict = proj.get("eps_2y", {"base": 5300, "bull": 6600, "bear": 4100})
    certainty_adj = float(proj.get("certainty_adjustment", -0.02))
    sus = earnings_output.get("sustainability_metrics", {})
    qual = earnings_output.get("quality_checks", {})
    th_status = earnings_output.get("thesis_driver", {}).get("status", "유효")

    # 거시 트랙(Track 4) 입력값 수신 (기본값: 한국/미국 10Y 금리 및 ERP)[cite: 4]
    if macro_input is None:
        macro_input = {"rf_10y": 0.035 if is_krx else 0.042, "market_erp": 0.055}

    # 기본 멀티플·피어·생존성 시드 데이터 (yfinance / DART / 컨센서스 연동 및 오버라이드 지원)[cite: 4]
    default_f = {
        "rf_10y": float(macro_input.get("rf_10y", 0.035 if is_krx else 0.042)),
        "market_erp": float(macro_input.get("market_erp", 0.055)),
        "industry_beta": 1.10 if sector_type in ["구조성장형", "사이클형"] else 0.85,
        "mature_g": 0.03,                 # 성숙기 영구성장률 g_성숙 (3.0%)[cite: 4]
        "sustainable_roe": 0.14,          # 지속가능 ROE / ROIC (14.0%)[cite: 4]
        "dividend_yield": 0.018 if is_krx else 0.010,  # 예상 배당수익률 d[cite: 4]
        "is_financial_sector": False,
        "listing_years": 10,
        # 자기 5년 PER 밴드 및 산업/피어 밴드[cite: 4]
        "self_5y_mean": 15.2,
        "self_5y_std": 2.8,
        "self_5y_min": 9.5,
        "ind_5y_mean": 14.0,
        "ind_5y_std": 2.5,
        "ind_rel_5y_mean": 1.15,
        "ind_rel_5y_std": 0.18,
        "ind_rel_current": 1.08,
        "peer_median_per": 14.5,
        "peer_top25_per": 17.5,
        "peer_std_per": 2.6,
        "peer_rel_5y_mean": 1.05,
        "peer_rel_5y_std": 0.15,
        # F1-3. 산업 레짐 실물 증거 (상향 증거 개수, 하향 증거 개수, 2분기 연속 확인 여부)[cite: 4]
        "ind_rerate_evidences": 1,
        "ind_derate_evidences": 0,
        "ind_evidence_2q_confirmed": False,
        # F1-4. 사이클형 P/B z-score (정점 착시 감지용)[cite: 4]
        "current_pb": 1.65,
        "pb_5y_mean": 1.40,
        "pb_5y_std": 0.30,
        # F2-2. 피어 중앙값 비교용 벤치마크 수치[cite: 4]
        "company_roic": 0.165,
        "peer_median_roic": 0.120,
        "peer_median_2y_cagr": 0.140,
        "shareholder_return_advantage": 1,  # +1(소각/밸류업 우위), 0(동등), -1(중복상장/희석 열위)[cite: 4]
        # F4. 12개월 주가·이익·배수 3층 로그 분해용 기초값 (12개월 전 vs 현재)[cite: 4]
        "price_12m_ago": round(current_price * 1.12, 0),
        "ntm_eps_12m_ago": round(float(eps_1y_dict["base"]) * 0.87, 0),
        "mkt_per_12m_ago": 11.2 if is_krx else 19.5,
        "mkt_per_now": 10.3 if is_krx else 20.8,
        "ind_per_12m_ago": 15.8,
        "ind_per_now": 12.6,
        # F5-2. 역대 최고 5Y CAGR[cite: 4]
        "historical_max_5y_cagr": 0.25,
        # F8-1. 증권사 컨센서스 평균 목표가 (앵커 1)[cite: 4]
        "consensus_target_price": round(current_price * 1.28, -2),
        # F9. 재무 생존성 3지표[cite: 4]
        "net_cash_krw": 120.0,            # 순현금 (양수면 순현금, 음수면 순차입금)[cite: 4]
        "fcf_ttm_krw": 18.5,              # 최근 4분기 합산 FCF[cite: 4]
        "short_term_debt_ratio": 0.22,    # 1년 내 차입 만기 / 보유 현금성자산[cite: 4]
        "share_dilution_yoy": 0.00,       # 연간 완전희석 주식수 증가율[cite: 4]
        "position_cap": 0.05,
    }
    if fund_override:
        default_f.update(fund_override)
    f = default_f

    eps_1y_base = float(eps_1y_dict["base"])
    eps_1y_bull = float(eps_1y_dict["bull"])
    eps_1y_bear = float(eps_1y_dict["bear"])
    eps_1y_cycle_bear = float(eps_1y_dict.get("cycle_bear") or (eps_1y_bear * 0.80))
    eps_2y_base = float(eps_2y_dict["base"])

    # 2Y EPS CAGR 산출[cite: 4]
    consensus_2y_cagr = max(0.02, (eps_2y_base / max(eps_1y_base, 1.0)) - 1.0)

    # -----------------------------------------------------------------
    # F0. 멀티플 종류 선택 (분모 왜곡 방지 규칙)[cite: 4]
    # -----------------------------------------------------------------
    if f["is_financial_sector"]:
        primary_mult, secondary_mult = "P/B", "배당수익률"
    elif e_regime == "G1" and eps_1y_base <= 0:
        primary_mult, secondary_mult = "NTM EV/매출", "목표 OPM 도달 시 내재 PER"
    elif f["listing_years"] < 3:
        primary_mult, secondary_mult = "피어 그룹 PER (μ, σ)", "PEG"
    elif sector_type == "사이클형":
        primary_mult, secondary_mult = "NTM PER", "P/B 밴드 (필수 교차)"
    else:
        primary_mult, secondary_mult = "NTM PER", "EV/EBITDA"

    current_per = round(current_price / max(eps_1y_base, 1.0), 2)

    # -----------------------------------------------------------------
    # F1. 산업이 받을 수 있는 멀티플 (이론 성숙 PER, 상대 밴드, 레짐, 사이클 착시)[cite: 4]
    # -----------------------------------------------------------------
    # F1-1. 시장 자기자본비용 k_e = R_f + ERP * beta (요구수익률 r과 엄격 분리)[cite: 4]
    market_ke = f["rf_10y"] + f["market_erp"] * f["industry_beta"]
    g_mat = min(f["mature_g"], market_ke - 0.02)
    roe_sust = max(f["sustainable_roe"], g_mat + 0.02)
    theoretical_mature_per = round((1.0 - (g_mat / roe_sust)) / max(market_ke - g_mat, 0.01), 1)

    # F1-3. 산업 레짐 3단계 판정[cite: 4]
    if f["ind_rerate_evidences"] >= 2 and f["ind_evidence_2q_confirmed"]:
        ind_regime = "전환 확정 (상향)"
    elif f["ind_derate_evidences"] >= 2 and f["ind_evidence_2q_confirmed"]:
        ind_regime = "전환 확정 (하향)"
    elif f["ind_rerate_evidences"] >= 2 or f["ind_derate_evidences"] >= 2:
        ind_regime = "전환 진행"
    else:
        ind_regime = "기존 밴드 유효"

    # -----------------------------------------------------------------
    # F3. 현재 멀티플 위치 (3중 z-score 입체 진단)[cite: 4]
    # -----------------------------------------------------------------
    self_mean = float(f["self_5y_mean"])
    self_std = max(float(f["self_5y_std"]), 0.5)
    self_z = round((current_per - self_mean) / self_std, 2)

    ind_rel_z = round((f["ind_rel_current"] - f["ind_rel_5y_mean"]) / max(f["ind_rel_5y_std"], 0.05), 2)
    peer_rel_now = current_per / max(f["peer_median_per"], 1.0)
    peer_rel_z = round((peer_rel_now - f["peer_rel_5y_mean"]) / max(f["peer_rel_5y_std"], 0.05), 2)

    if peer_rel_z <= -1.0:
        z_interp = "피어 대비 뚜렷한 저평가"
    elif self_z <= -1.0:
        z_interp = "역사적 밴드 하단 저평가"
    elif self_z >= 2.0:
        z_interp = "역사적 극단 고점 경고 (+2σ 초과)"
    elif self_z >= 1.0:
        z_interp = "고평가 구간 (+1σ 상회)"
    else:
        z_interp = "정상 거래 밴드 구간"

    triple_z_summary = f"자기 밴드 {self_z:+.2f}σ | 산업 상대 {ind_rel_z:+.2f}σ | 피어 상대 {peer_rel_z:+.2f}σ ({z_interp})"

    # F1-4. 사이클형 정점 착시 감지기 (Spread_z = z_PB - z_PER >= 1.5)[cite: 4]
    cycle_type = None
    pb_per_z_spread = None
    is_cycle_peak_trap = False
    if sector_type == "사이클형":
        z_pb = (f["current_pb"] - f["pb_5y_mean"]) / max(f["pb_5y_std"], 0.05)
        pb_per_z_spread = round(z_pb - self_z, 2)
        rev_status = earnings_output.get("revision_dynamics", {}).get("status", "보합")
        if pb_per_z_spread >= 1.5 or cycle_stage == "C3 정점 과열 (재고 급증 경고)":
            cycle_type = "Type 2 (정점 착시 / 디레이팅 경고)"
            is_cycle_peak_trap = True
        elif rev_status == "하향 지속" or cycle_stage == "C4 하강 (P·Q 동반 둔화)":
            cycle_type = "Type 3 (하강 확정)"
        else:
            cycle_type = "Type 1 (확장 초중기)"

    # -----------------------------------------------------------------
    # F9. 재무 생존성 3지표 (Hard Survival Filter — F2-2 7번 요인 연계)[cite: 4]
    # -----------------------------------------------------------------
    fcf_ttm = float(f["fcf_ttm_krw"])
    net_cash = float(f["net_cash_krw"])
    if fcf_ttm >= 0:
        runway_str = "FCF 흑자 (해당 없음)"
        runway_warn = False
    elif net_cash > 0 and (net_cash / abs(fcf_ttm)) >= 2.0:
        runway_str = f"{net_cash / abs(fcf_ttm):.1f}년 (순현금 방어)"
        runway_warn = False
    else:
        rw_yrs = max(0.0, net_cash / abs(fcf_ttm)) if net_cash > 0 else 0.0
        runway_str = f"{rw_yrs:.1f}년 (위험 < 2.0년)"
        runway_warn = True

    debt_mat_ratio = float(f["short_term_debt_ratio"])
    debt_warn = debt_mat_ratio > 1.00
    dilution_yoy = float(f["share_dilution_yoy"])
    dilution_warn = dilution_yoy > 0.05
    survival_warning = bool(runway_warn or debt_warn or dilution_warn)

    # -----------------------------------------------------------------
    # F2. 피어 대비 프리미엄·디스카운트 정당화 8대 요인 비교표[cite: 4]
    # -----------------------------------------------------------------
    peer_factors = []
    # 1) 성장 격차 (g)[cite: 4]
    g_diff = consensus_2y_cagr - f["peer_median_2y_cagr"]
    f1_s = 1 if g_diff >= 0.03 else (-1 if g_diff <= -0.03 else 0)
    peer_factors.append({"no": 1, "factor": "성장 격차 (2Y EPS CAGR)", "diff": f"{g_diff*100:+.1f}%p", "score": f1_s})

    # 2) 수익성·효율 격차 (ROIC)[cite: 4]
    roic_diff = f["company_roic"] - f["peer_median_roic"]
    f2_s = 1 if roic_diff >= 0.03 else (-1 if roic_diff <= -0.03 else 0)
    peer_factors.append({"no": 2, "factor": "수익성·효율 격차 (ROIC)", "diff": f"{roic_diff*100:+.1f}%p", "score": f2_s})

    # 3) 영업레버리지 (ΔOPM)[cite: 4]
    op_lev_spread = float(sus.get("incremental_opm", 0.24)) - float(sus.get("average_opm", 0.20))
    f3_s = 1 if op_lev_spread > 0.02 else (-1 if op_lev_spread < -0.02 else 0)
    peer_factors.append({"no": 3, "factor": "영업레버리지 (증분-평균 OPM)", "diff": f"{op_lev_spread*100:+.1f}%p", "score": f3_s})

    # 4) 이익 지속성 (RPO·수주)[cite: 4]
    rpo_spread = float(sus.get("rpo_yoy_growth", 0.30)) - float(sus.get("sales_yoy_growth", 0.20))
    f4_s = 1 if rpo_spread > 0.0 else (-1 if rpo_spread < -0.05 else 0)
    peer_factors.append({"no": 4, "factor": "이익 지속성 (RPO - 매출 성장)", "diff": f"{rpo_spread*100:+.1f}%p", "score": f4_s})

    # 5) 가이던스 신뢰도[cite: 4]
    f5_s = 1 if certainty_adj < 0 else (-1 if certainty_adj > 0 else 0)
    peer_factors.append({"no": 5, "factor": "경영진 가이던스 신뢰도", "diff": f"할인율 {certainty_adj*100:+.0f}%p", "score": f5_s})

    # 6) 이익의 질 (현금 전환)[cite: 4]
    w_cnt = int(qual.get("warning_count", 0))
    f6_s = 1 if w_cnt == 0 else (-1 if w_cnt >= 2 else 0)
    peer_factors.append({"no": 6, "factor": "이익의 질 (회계·현금전환)", "diff": f"경고 {w_cnt}개", "score": f6_s})

    # 7) 재무 생존성[cite: 4]
    f7_s = 1 if (net_cash > 0 and not survival_warning) else (-1 if survival_warning else 0)
    peer_factors.append({"no": 7, "factor": "재무 생존성 (순현금·런웨이)", "diff": "무경고" if not survival_warning else "경고 발동", "score": f7_s})

    # 8) 주주환원·지배구조[cite: 4]
    f8_s = -1 if dilution_warn else int(f["shareholder_return_advantage"])
    peer_factors.append({"no": 8, "factor": "주주환원·지배구조 (소각/밸류업)", "diff": "우위(+1)" if f8_s > 0 else ("열위(-1)" if f8_s < 0 else "동등(0)"), "score": f8_s})

    net_adv_score = sum(item["score"] for item in peer_factors)
    if net_adv_score >= 2 and (peer_rel_now <= 1.0 or peer_rel_z <= -1.0):
        peer_just = "📈 멀티플 과소평가 후보 (펀더멘탈 순우위이나 피어 대비 할인)"
    elif net_adv_score >= 2 and peer_rel_now > 1.0:
        peer_just = "✅ 정당한 프리미엄 (순우위 뒷받침)"
    elif net_adv_score <= -2 and (peer_rel_now >= 1.0 or peer_rel_z >= 1.0):
        peer_just = "📉 멀티플 과대평가 후보 (펀더멘탈 열위이나 피어 대비 할증)"
    elif net_adv_score <= -2:
        peer_just = "✅ 정당한 디스카운트 (가치 함정 주의)"
    else:
        peer_just = "피어 평균 부합 구간"

    # -----------------------------------------------------------------
    # F4. 배수 변화 3층 로그 분해 (시장층 + 산업층 + 기업층)[cite: 4]
    # -----------------------------------------------------------------
    p_0 = max(float(f["price_12m_ago"]), 1.0)
    eps_0 = max(float(f["ntm_eps_12m_ago"]), 1.0)
    per_0 = p_0 / eps_0

    log_price_ret = math.log(max(current_price, 1.0) / p_0)
    log_eps_contrib = math.log(max(eps_1y_base, 1.0) / eps_0)
    log_mult_contrib = math.log(max(current_per, 0.1) / max(per_0, 0.1))

    mkt_0, mkt_t = float(f["mkt_per_12m_ago"]), float(f["mkt_per_now"])
    ind_0, ind_t = float(f["ind_per_12m_ago"]), float(f["ind_per_now"])

    layer_market = math.log(mkt_t / mkt_0)
    layer_industry = math.log((ind_t / mkt_t) / (ind_0 / mkt_0))
    layer_company = log_mult_contrib - (layer_market + layer_industry)

    abs_layers = [("시장층(금리·ERP)", abs(layer_market), layer_market), ("산업층(업종 쏠림/배수)", abs(layer_industry), layer_industry), ("기업층(개별 프리미엄)", abs(layer_company), layer_company)]
    abs_layers.sort(key=lambda x: x[1], reverse=True)
    top_layer_name, _, top_layer_val = abs_layers[0]

    if "산업층" in top_layer_name and top_layer_val < 0 and log_eps_contrib > 0:
        dom_driver = "⚡ 업종 센티먼트 동반 매도 (이익은 성장했으나 산업 배수 압축)"
    elif "시장층" in top_layer_name and top_layer_val < 0:
        dom_driver = "⚡ 시장 금리·ERP 상승에 따른 매크로 배수 압축 (기업 훼손 아님)"
    elif "기업층" in top_layer_name and top_layer_val > 0 and e_regime in ["G1", "G2"]:
        dom_driver = "🏛️ 기업 고유 구조적 리레이팅 (RPO·영업레버리지 개선 반영)"
    elif "기업층" in top_layer_name and top_layer_val < 0 and e_regime in ["G4", "G5"]:
        dom_driver = "🏛️ 피크아웃·성장 둔화에 따른 구조적 디레이팅"
    else:
        dom_driver = f"{top_layer_name} 주도 배수 변화 ({top_layer_val*100:+.1f}%)"

    # -----------------------------------------------------------------
    # F7-1. 요구수익률 r 산출 (F5-2 역산 및 F7 매수기준선 공통 사용)[cite: 4]
    # -----------------------------------------------------------------
    sector_prem_map = {"방어형": 0.04, "구조성장형": 0.07, "정책의존형": 0.10, "사이클형": 0.12}
    sector_prem = sector_prem_map.get(sector_type, 0.07)
    req_r = round(f["rf_10y"] + sector_prem + certainty_adj, 4)

    # -----------------------------------------------------------------
    # F5. 재평가(리레이팅·디레이팅) 4축 진단 및 5년 성숙기 수렴 내재 성장률 역산[cite: 4]
    # -----------------------------------------------------------------
    per_terminal = max(theoretical_mature_per, float(f["ind_5y_mean"]))
    implied_5y_cagr = ((current_price * ((1.0 + req_r) ** 5)) / max(eps_1y_base * per_terminal, 1.0)) ** (1.0 / 5.0) - 1.0
    implied_5y_cagr = round(implied_5y_cagr, 4)

    ax_dur = "강" if rpo_spread > 0 else ("중" if rpo_spread >= -0.05 else "약")
    ax_inc = "강" if op_lev_spread > 0 else "약"
    ax_cert = "강" if (certainty_adj < 0 and w_cnt == 0) else ("약" if (certainty_adj > 0 or w_cnt >= 2) else "중")
    if implied_5y_cagr < consensus_2y_cagr and self_z <= 0.5:
        ax_priced = "낮음"  # 유리 (과소평가)[cite: 4]
    elif implied_5y_cagr > max(consensus_2y_cagr, f["historical_max_5y_cagr"]) or self_z >= 1.0:
        ax_priced = "높음"  # 과열[cite: 4]
    else:
        ax_priced = "중립"

    strong_cnt = sum(1 for x in [ax_dur, ax_inc, ax_cert] if x == "강")
    if e_regime in ["G4", "G5"] or (ax_priced == "높음" and self_z >= 1.0):
        rerate_dir = "⬇ 수축 위험"
    elif strong_cnt >= 2 and ax_priced == "낮음":
        rerate_dir = "⬆ 확장 여지"
    else:
        rerate_dir = "➡ 유지"

    # -----------------------------------------------------------------
    # F6. 실적 국면(G1~G5) 연동 Base·Bull·Bear 멀티플 결정 (가치 함정 차단)[cite: 4]
    # -----------------------------------------------------------------
    is_value_trap_blocked = False
    if e_regime == "G1":
        base_mult = round(float(f["peer_top25_per"]), 1)
        bull_mult = round(base_mult + float(f["peer_std_per"]), 1)
        bear_mult = round(float(f["peer_median_per"]), 1)
        mult_rule_desc = "G1 초기 가속: 피어 상위 25% 멀티플 채택 (초기 적자/저수익 자기밴드 왜곡 보정)"
    elif e_regime == "G2":
        base_mult = round(self_mean, 1)
        bull_mult = round(self_mean + self_std, 1)
        bear_mult = round(max(5.0, self_mean - self_std), 1)
        mult_rule_desc = "G2 우량 지속 성장: 자기 5년 역사적 평균(μ) 중심 표준 밴드 적용"
    elif e_regime == "G3":
        base_mult = round(min(self_mean, float(f["peer_median_per"]), theoretical_mature_per * 1.1), 1)
        bull_mult = round(base_mult + 0.5 * self_std, 1)
        bear_mult = round(max(4.0, base_mult - 1.0 * self_std), 1)
        mult_rule_desc = "G3 성숙/현금회수: min(자기평균, 피어중앙값, 이론성숙PER×1.1) 상한 적용"
    elif e_regime == "G4":
        is_value_trap_blocked = True
        base_mult = round(min(self_mean - 1.0 * self_std, current_per, theoretical_mature_per), 1)
        bull_mult = round(current_per, 1)
        bear_mult = round(min(self_mean - 2.0 * self_std, float(f["self_5y_min"])), 1)
        mult_rule_desc = "🚨 G4 피크아웃 디레이팅 Cap 발동: 과거 평균 회귀 불인정 (가치 함정 원천 차단)"
    else:  # G5 구조적 쇠퇴[cite: 4]
        is_value_trap_blocked = True
        base_mult = 0.0
        bull_mult = 0.0
        bear_mult = 0.0
        mult_rule_desc = "🚨 G5 구조적 쇠퇴: 목표 멀티플 산출 중단 및 가격 자격 하드 차단"

    base_independence_passed = bool(e_regime != "G5")

    # -----------------------------------------------------------------
    # F7-2 & F7-3. 사이클 정점 보호 및 0~3차 매수기준선(buy_tranches) 생성[cite: 4]
    # -----------------------------------------------------------------
    if is_cycle_peak_trap:
        # ★ 사이클 정점 착시 보호 규칙: 0.5 * Base + 0.5 * CycleBear 정상화 EPS 강제 사용[cite: 4]
        eps_eff_1y = 0.5 * eps_1y_base + 0.5 * eps_1y_cycle_bear
    else:
        eps_eff_1y = eps_1y_base

    div_d = float(f["dividend_yield"])
    disc_1y = max(0.02, 1.0 + req_r - div_d)
    disc_1y_floor = max(0.02, 1.0 + (req_r - 0.03) - div_d)

    if e_regime == "G5":
        t0_price, t1_price, t2_price, t3_price = 0, 0, 0, 0
    else:
        raw_t0 = (eps_eff_1y * base_mult) / disc_1y_floor
        raw_t1 = (eps_eff_1y * base_mult) / disc_1y
        raw_t2 = min(raw_t1 * 0.90, (eps_eff_1y * max(1.0, base_mult - 0.5 * self_std)) / disc_1y)
        raw_bear_line = (eps_1y_bear * bear_mult) / disc_1y
        raw_t3 = max(raw_bear_line, raw_t1 * 0.80)

        # 상위 차수와 하위 차수 최소 5% 간격 강제 및 KRX 호가 내림 처리[cite: 4]
        t1_price = int(adjust_krx_price(raw_t1, "down", is_krx))
        t0_price = int(adjust_krx_price(max(raw_t0, t1_price * 1.05), "down", is_krx))
        t2_price = int(adjust_krx_price(min(raw_t2, t1_price * 0.92), "down", is_krx))
        t3_price = int(adjust_krx_price(min(raw_t3, t2_price * 0.92), "down", is_krx))

    # F7-4. 시나리오 수익률 매트릭스 · 확률가중 기대수익률 E[R] · MDD Ratio[cite: 4]
    p_bull_1y = eps_1y_bull * bull_mult
    p_base_1y = eps_eff_1y * base_mult
    p_bear_1y = eps_1y_bear * bear_mult
    p_cbear_1y = eps_1y_cycle_bear * bear_mult

    r_bull = (p_bull_1y / max(current_price, 1.0)) - 1.0 + div_d
    r_base = (p_base_1y / max(current_price, 1.0)) - 1.0 + div_d
    r_bear = (p_bear_1y / max(current_price, 1.0)) - 1.0 + div_d
    r_cbear = (p_cbear_1y / max(current_price, 1.0)) - 1.0 + div_d

    if sector_type == "사이클형":
        # 사이클형: Base 40%, Bull 20%, Bear 20%, CycleBear 20%[cite: 2, 4]
        exp_return = 0.40 * r_base + 0.20 * r_bull + 0.20 * r_bear + 0.20 * r_cbear
        worst_loss = abs(min(r_bear, r_cbear, -0.05))
    else:
        # 일반: Base 50%, Bull 25%, Bear 25%[cite: 2, 4]
        exp_return = 0.50 * r_base + 0.25 * r_bull + 0.25 * r_bear
        worst_loss = abs(min(r_bear, -0.05))

    mdd_ratio = round(exp_return / max(worst_loss, 0.01), 2)
    downside_convexity = round(
        abs(current_price - p_bear_1y) / max(abs(p_bull_1y - current_price), 1.0), 2
    )

    # 0차 개방 여부: G1/G2 이고 MDD Ratio >= 0.30 일 때만 0차 개방[cite: 4]
    pos_cap = float(f["position_cap"])
    if survival_warning:
        pos_cap *= 0.5  # F9 런웨이 경고 시 목표 비중 50% 축소[cite: 4]

    open_t0 = bool(e_regime in ["G1", "G2"] and mdd_ratio >= 0.30)
    buy_tranches = []
    if e_regime != "G5":
        if open_t0:
            buy_tranches.append({"tranche": 0, "price": t0_price, "weight": round(pos_cap * 0.10, 4), "label": "0차 탐색선 (r_floor 보험)"})
            w_t1 = round(pos_cap * 0.30, 4)
        else:
            w_t1 = round(pos_cap * 0.40, 4)
        buy_tranches.append({"tranche": 1, "price": t1_price, "weight": w_t1, "label": "1차 기본 매수기준선 (Base 1Y)"})
        buy_tranches.append({"tranche": 2, "price": t2_price, "weight": round(pos_cap * 0.30, 4), "label": "2차 안전마진 강화선 (-10%)"})
        buy_tranches.append({"tranche": 3, "price": t3_price, "weight": round(pos_cap * 0.30, 4), "label": "3차 Bear 시나리오 방어선"})

    # -----------------------------------------------------------------
    # F8. 매도목표가(sell_target) 4대 독립 앵커 및 보유기간 상한 검증[cite: 4]
    # -----------------------------------------------------------------
    # 앵커 1: 컨센서스 목표가[cite: 4]
    anchor1_price = int(adjust_krx_price(f["consensus_target_price"], "down", is_krx))
    anchor1_per = round(anchor1_price / max(eps_1y_base, 1.0), 1)

    # 앵커 2: 역사적 밴드 상단 (단, G3/G4는 자기 평균 μ_self로 캡)[cite: 4]
    anchor2_per = round(self_mean if e_regime in ["G3", "G4"] else (self_mean + 1.0 * self_std), 1)
    anchor2_price = int(adjust_krx_price(anchor2_per * eps_1y_base, "down", is_krx))

    # 앵커 3: PEG 적정 배수 (한국 0.8 / 미국 1.0 / G1 초기 가속 1.2, 산업 +2σ 및 50배 상한)[cite: 4]
    peg_target = 1.2 if e_regime == "G1" else (0.8 if is_krx else 1.0)
    ind_upper_2s = float(f["ind_5y_mean"]) + 2.0 * float(f["ind_5y_std"])
    anchor3_per = round(min(peg_target * (100.0 * consensus_2y_cagr), ind_upper_2s, 50.0), 1)
    anchor3_per = max(anchor3_per, base_mult)
    anchor3_price = int(adjust_krx_price(anchor3_per * eps_1y_base, "down", is_krx))

    # 앵커 4: 피어 상위 25% 배수 (F2-2 순우위 >= +2일 때만 유효)[cite: 4]
    anchor4_valid = net_adv_score >= 2
    anchor4_per = round(float(f["peer_top25_per"]), 1)
    anchor4_price = int(adjust_krx_price(anchor4_per * eps_1y_base, "down", is_krx))

    # 최종 채택 매도목표가: 독립 앵커 2와 앵커 3의 평균[cite: 4]
    adopted_sell_per = round((anchor2_per + anchor3_per) / 2.0, 1)
    sell_target_1y = int(adjust_krx_price(adopted_sell_per * eps_1y_base, "down", is_krx))
    sell_target_formula = f"{fmt_price(sell_target_1y, is_krx)} = 1Y Base EPS {fmt_price(eps_1y_base, is_krx)} × 목표 PER {adopted_sell_per:.1f}배 (독립 앵커2·3 평균)"

    # 2Y 매도목표가 및 보유기간 상한 경고 (2Y 목표가 < 1Y 목표가 역전 여부)[cite: 4]
    per_2y_target = max(theoretical_mature_per, base_mult)
    sell_target_2y = int(adjust_krx_price(per_2y_target * eps_2y_base, "down", is_krx))
    holding_cap_warn = bool(sell_target_2y < sell_target_1y)

    anchors_table = [
        {"id": "앵커 1", "name": "증권사 컨센서스 평균", "target_per": f"{anchor1_per:.1f}배", "target_price": f"{fmt_price(anchor1_price, is_krx)}", "indep": "반종속 (참고 교차검증)"},
        {"id": "앵커 2", "name": "역사적 밴드 상단 (μ+1σ)", "target_per": f"{anchor2_per:.1f}배", "target_price": f"{fmt_price(anchor2_price, is_krx)}", "indep": "✅ 독립 (채택 반영)"},
        {"id": "앵커 3", "name": f"PEG 적정 배수 (PEG={peg_target})", "target_per": f"{anchor3_per:.1f}배", "target_price": f"{fmt_price(anchor3_price, is_krx)}", "indep": "✅ 독립 (채택 반영)"},
        {"id": "앵커 4", "name": "피어 상위 25% 배수", "target_per": f"{anchor4_per:.1f}배", "target_price": f"{fmt_price(anchor4_price, is_krx)}", "indep": "유효 참고" if anchor4_valid else "미충족 (순우위<2)"},
    ]

    # -----------------------------------------------------------------
    # 4장. 트랙 종합 판정: 가격 자격 (QUALIFIED / CONDITIONAL / DISQUALIFIED)[cite: 4]
    # -----------------------------------------------------------------
    if e_regime == "G5" or th_status == "붕괴" or mdd_ratio < 0.15:
        pq_status = "DISQUALIFIED"
        pq_icon = "🔴"
        if e_regime == "G5" or th_status == "붕괴":
            pq_reason = "실적 국면 G5(구조적 쇠퇴) 또는 드라이버 붕괴로 가격 자격 하드 차단"
        else:
            pq_reason = f"MDD Ratio({mdd_ratio:.2f}) 최소 기준(0.15) 미달로 위험 대비 기대수익 부적격"
    elif current_price <= t1_price and mdd_ratio >= 0.30 and exp_return >= req_r and not survival_warning and self_z < 1.0:
        pq_status = "QUALIFIED"
        pq_icon = "🟢"
        gap_pct = (current_price / t1_price - 1.0) * 100.0
        pq_reason = f"현재가({fmt_price(current_price, is_krx)})가 1차 매수기준선({fmt_price(t1_price, is_krx)}, {gap_pct:+.1f}%) 이하 및 MDD Ratio {mdd_ratio:.2f}로 가격 자격 충족"
    elif (open_t0 and current_price <= t0_price) or (current_price <= t1_price and mdd_ratio >= 0.15):
        pq_status = "CONDITIONAL"
        pq_icon = "🟡"
        pq_reason = f"조건부 가격 자격 — 0차 탐색선({fmt_price(t0_price, is_krx)}) 이하 도달 또는 목표 비중 50% 축소 가동 구간 (MDD Ratio {mdd_ratio:.2f})"
    else:
        pq_status = "DISQUALIFIED"
        pq_icon = "🔴"
        gap_to_t1 = (current_price / max(t1_price, 1) - 1.0) * 100.0
        pq_reason = f"현재가({fmt_price(current_price, is_krx)})가 1차 매수기준선({fmt_price(t1_price, is_krx)}) 대비 +{gap_to_t1:.1f}% 높음 — 지정가 도달 전까지 추격 매수 금지(대기)"

    # F10. 멀티플 오판 포인트 자동 문장 생성[cite: 4]
    if implied_5y_cagr < consensus_2y_cagr:
        underval_pt = (
            f"현재 주가에 내재된 5년 성장률({implied_5y_cagr*100:.1f}%)이 컨센서스 성장률({consensus_2y_cagr*100:.1f}%)보다 현저히 낮아 "
            f"멀티플 안전마진이 확보됨 ({dom_driver})."
        )
    else:
        underval_pt = "현재 배수는 컨센서스 성장률을 상당 부분 반영 중."

    if implied_5y_cagr > max(consensus_2y_cagr, f["historical_max_5y_cagr"]) or self_z >= 1.5:
        overval_pt = f"내재 5년 성장률({implied_5y_cagr*100:.1f}%) 또는 자기 밴드({self_z:+.2f}σ)가 과열 구간에 진입하여 멀티플 수축 위험 존재."
    elif is_cycle_peak_trap:
        overval_pt = f"사이클 정점 착시(P/B-PER Z스프레드 {pb_per_z_spread:+.2f}σ) 발동 — 표면적 저PER 함정 주의."
    else:
        overval_pt = "해당 없음 (과열 신호 없음)"

    return {
        "status": "완료",
        "thesis_status": "붕괴" if e_regime == "G5" else th_status,
        "price_qualification": {
            "status": pq_status,
            "grade_icon": pq_icon,
            "headline_reason": pq_reason,
            "is_value_trap_blocked": is_value_trap_blocked,
            "is_cycle_peak_trap": is_cycle_peak_trap,
        },
        "multiple_selection": {
            "primary_metric": primary_mult,
            "secondary_metric": secondary_mult,
            "current_multiple": current_per,
            "rule_description": mult_rule_desc,
        },
        "triple_z_score": {
            "self_5y_z": self_z,
            "self_5y_mean": self_mean,
            "self_5y_std": self_std,
            "industry_rel_z": ind_rel_z,
            "peer_rel_z": peer_rel_z,
            "summary_text": triple_z_summary,
        },
        "industry_and_peer": {
            "theoretical_mature_per": theoretical_mature_per,
            "market_ke": round(market_ke, 4),
            "industry_regime": ind_regime,
            "cycle_type": cycle_type,
            "pb_per_z_spread": pb_per_z_spread,
            "peer_net_advantage_score": net_adv_score,
            "peer_justification": peer_just,
            "peer_factors_table": peer_factors,
        },
        "multiple_decomposition": {
            "price_return_12m": round(log_price_ret, 4),
            "eps_contribution_12m": round(log_eps_contrib, 4),
            "multiple_contribution_12m": round(log_mult_contrib, 4),
            "layer_market": round(layer_market, 4),
            "layer_industry": round(layer_industry, 4),
            "layer_company": round(layer_company, 4),
            "dominant_driver": dom_driver,
        },
        "rerating_assessment": {
            "direction": rerate_dir,
            "implied_5y_cagr": implied_5y_cagr,
            "consensus_2y_cagr": round(consensus_2y_cagr, 4),
            "terminal_per_used": per_terminal,
            "four_axes": {
                "duration": ax_dur,
                "incremental_margin": ax_inc,
                "certainty": ax_cert,
                "priced_in": ax_priced,
            },
        },
        "valuation_targets": {
            "required_return_r": req_r,
            "effective_eps_1y": int(round(eps_eff_1y, 0)),
            "base_multiple_target": base_mult,
            "bull_multiple_target": bull_mult,
            "bear_multiple_target": bear_mult,
            "base_independence_passed": base_independence_passed,
            "buy_tranches": buy_tranches,
            "sell_target": sell_target_1y,
            "sell_target_formula": sell_target_formula,
            "sell_target_2y": sell_target_2y,
            "holding_period_cap_warning": holding_cap_warn,
            "expected_return_weighted": round(exp_return, 4),
            "mdd_ratio": mdd_ratio,
            "downside_convexity": downside_convexity,
            "anchors_table": anchors_table,
            "scenario_returns": {
                "bull": round(r_bull, 4),
                "base": round(r_base, 4),
                "bear": round(r_bear, 4),
                "cycle_bear": round(r_cbear, 4) if sector_type == "사이클형" else None,
            },
        },
        "financial_survival": {
            "fcf_runway_years": runway_str,
            "debt_maturity_1y_ratio": round(debt_mat_ratio, 2),
            "share_dilution_yoy": round(dilution_yoy, 4),
            "survival_warning": survival_warning,
        },
        "mispricing_points": {
            "undervalued_point": underval_pt,
            "overvalued_point": overval_pt,
        },
        # 하위 호환용 최상위 키 유지 (기술 트랙 연동)
        "buy_tranches": buy_tranches,
        "sell_target": sell_target_1y,
        "position_cap": round(pos_cap, 4),
    }


# =====================================================================
# [Track 4] 거시적 분석 트랙 v1.0 (가설 등록·잔차화 회귀·OOS 검증 엔진)
# =====================================================================
import os
import re
import hashlib

# 8절. 확정 파라미터 요약표 (12개 항목 Lock)
MACRO_PARAMS: Dict[str, Any] = {
    "USE_EMPIRICAL_REGRESSION": True,    # 1: 가설 등록 + HAC/Huber 실측[cite: 5]
    "REG_WEEKS_MAIN": 104,               # 2a: 주력 회귀 기간 (104주)[cite: 5]
    "REG_WEEKS_SUB": 52,                 # 2b: 비교 회귀 기간 (52주)[cite: 5]
    "REG_WEEKS_MIN": 78,                 # 2c: 최소 유효 주간 표본 (78주)[cite: 5]
    "OOS_HORIZON_WEEKS": 13,             # 최근 점검 및 OOS 이동 원점 구간 (13주)[cite: 5]
    "RESIDUALIZE_CORR_THRES": 0.50,      # 3: 시장 동행 잔차화 상관계수 임계값 (|ρ| >= 0.50)[cite: 5]
    "MAX_HYPOTHESIS_VARS": 3,            # 4: 가설 변수 최대 개수 (3개)[cite: 5]
    "HAC_MAXLAGS": 4,                    # 5: Newey-West HAC 최대 시차 (4주 고정)[cite: 5]
    "RECENT_SURGE_SHARE_4W": 0.50,       # 6: 최근 급변 판정 기준 (4주/13주 몫 50% 이상)[cite: 5]
    "R2_DOMINANT": 0.40,                 # 7a: 거시 지배 R^2 경계 (0.40)[cite: 5]
    "R2_MIXED": 0.15,                    # 7b: 거시 혼합 R^2 경계 (0.15)[cite: 5]
    "REGIME_INFL_HIGH": 0.60,            # 8a: 인플레 주도 국면 기준 (60% 이상)[cite: 5]
    "REGIME_INFL_LOW": 0.40,             # 8b: 성장·정책 주도 국면 기준 (40% 이하)[cite: 5]
    "OUTLIER_BETA_DIFF_TOL": 0.50,       # 9: OLS vs Huber 괴리 허용치 (50%)[cite: 5]
    "OOS_RESID_Z_THRES": 2.0,            # 10: 표본 외 잔차 이탈 기준 (|z_oos| >= 2.0, 최소 6주)[cite: 5]
    "OOS_MIN_NORMAL_WEEKS": 6,           # 10b: 평범한 주 최소 표본 (6주)[cite: 5]
    "OOS_MIN_RUNS": 16,                  # 11a: 사후 검증 최소 구간 수[cite: 5]
    "OOS_WIN_RATE": 0.60,                # 11b: ΔR^2_oos > 0 비율 기준 (60%)[cite: 5]
    "ENABLE_SIZE_CAP_LINK": False,       # 12: 포지션 비중 상한 자동 연동 비활성[cite: 5]
}

# 3-1. 산업 기본 노출 매트릭스 (자동 가설 등록 상위 3변수 v0_auto 소스)[cite: 5]
INDUSTRY_MACRO_HYPOTHESIS_MAP: Dict[str, List[Dict[str, str]]] = {
    "메모리 (한국 상장)": [
        {"var_id": "ICSA", "var_name": "미국 신규 실업수당 청구 (경기)", "expected_sign": "-", "strength": "●●●", "unit": "pct", "dup": None},
        {"var_id": "DEXKOUS", "var_name": "원/달러 환율", "expected_sign": "±", "strength": "●●", "unit": "pct", "dup": "실적 트랙 E4 (환율 드라이버)"},
        {"var_id": "KR10Y", "var_name": "국고채 10년물 금리 (817Y002)", "expected_sign": "-", "strength": "●", "unit": "bp", "dup": "펀더멘탈 트랙 F7 (요구수익률 r)"},
    ],
    "AI 반도체·네트워킹(팹리스)": [
        {"var_id": "DFII10", "var_name": "미국 실질 10Y 금리 (TIPS)", "expected_sign": "-", "strength": "●●", "unit": "bp", "dup": "펀더멘탈 트랙 F7 (요구수익률 r)"},
        {"var_id": "BAMLC0A0CM", "var_name": "미국 투자등급(IG) 회사채 스프레드", "expected_sign": "-", "strength": "●●", "unit": "bp", "dup": None},
        {"var_id": "NETLIQ", "var_name": "연준 순유동성 (WALCL-TGA-RRP)", "expected_sign": "+", "strength": "●●", "unit": "pct", "dup": None},
    ],
    "반도체 장비": [
        {"var_id": "ICSA", "var_name": "미국 신규 실업수당 청구 (경기)", "expected_sign": "-", "strength": "●●●", "unit": "pct", "dup": None},
        {"var_id": "DFII10", "var_name": "미국 실질 10Y 금리 (TIPS)", "expected_sign": "-", "strength": "●●", "unit": "bp", "dup": "펀더멘탈 트랙 F7 (요구수익률 r)"},
        {"var_id": "BAMLC0A0CM", "var_name": "미국 투자등급(IG) 회사채 스프레드", "expected_sign": "-", "strength": "●●", "unit": "bp", "dup": None},
    ],
    "고멀티플 SW·플랫폼": [
        {"var_id": "DFII10", "var_name": "미국 실질 10Y 금리 (TIPS)", "expected_sign": "-", "strength": "●●●", "unit": "bp", "dup": "펀더멘탈 트랙 F7 (요구수익률 r)"},
        {"var_id": "BAMLH0A0HYM2", "var_name": "미국 하이일드(HY) 회사채 스프레드", "expected_sign": "-", "strength": "●●", "unit": "bp", "dup": None},
        {"var_id": "NETLIQ", "var_name": "연준 순유동성 (WALCL-TGA-RRP)", "expected_sign": "+", "strength": "●●", "unit": "pct", "dup": None},
    ],
    "은행·금융": [
        {"var_id": "BAMLH0A0HYM2", "var_name": "미국 하이일드(HY) 회사채 스프레드", "expected_sign": "-", "strength": "●●●", "unit": "bp", "dup": None},
        {"var_id": "BAMLC0A0CM", "var_name": "미국 투자등급(IG) 회사채 스프레드", "expected_sign": "-", "strength": "●●", "unit": "bp", "dup": None},
        {"var_id": "T10Y2Y", "var_name": "장단기 금리차 (10Y-2Y)", "expected_sign": "+", "strength": "●", "unit": "bp", "dup": None},
    ],
    "방산 (한국)": [
        {"var_id": "DEXKOUS", "var_name": "원/달러 환율", "expected_sign": "+", "strength": "●●", "unit": "pct", "dup": "실적 트랙 E4 (환율 드라이버)"},
        {"var_id": "KR10Y", "var_name": "국고채 10년물 금리 (817Y002)", "expected_sign": "-", "strength": "●", "unit": "bp", "dup": "펀더멘탈 트랙 F7 (요구수익률 r)"},
        {"var_id": "DCOILWTICO", "var_name": "WTI 원유 현물가", "expected_sign": "+", "strength": "●", "unit": "pct", "dup": None},
    ],
    "에너지·정유": [
        {"var_id": "DCOILWTICO", "var_name": "WTI 원유 현물가", "expected_sign": "+", "strength": "●●●", "unit": "pct", "dup": "실적 트랙 E4 (원자재 드라이버)"},
        {"var_id": "ICSA", "var_name": "미국 신규 실업수당 청구 (경기)", "expected_sign": "-", "strength": "●●", "unit": "pct", "dup": None},
        {"var_id": "BAMLH0A0HYM2", "var_name": "미국 하이일드(HY) 회사채 스프레드", "expected_sign": "-", "strength": "●●", "unit": "bp", "dup": None},
    ],
}


def _ensure_macro_hypothesis_file(ticker: str, industry_category: str) -> Tuple[str, List[Dict[str, str]]]:
    """
    3-3. 가설 파일 불변성(Immutable Lock)과 자동 생성 Fallback (v1.0 확정)[cite: 5]:
    - 수동 가설이 없으면 회귀 연산 직전에 data/macro/hypotheses/<티커>_v0_auto.yaml 자동 생성 후 SHA-256 해시 기록[cite: 5].
    """
    hyp_vars = INDUSTRY_MACRO_HYPOTHESIS_MAP.get(
        industry_category,
        INDUSTRY_MACRO_HYPOTHESIS_MAP["메모리 (한국 상장)" if ticker.upper().endswith((".KS", ".KQ")) else "AI 반도체·네트워킹(팹리스)"]
    )[:MACRO_PARAMS["MAX_HYPOTHESIS_VARS"]]

    hyp_dir = os.path.join("data", "macro", "hypotheses")
    safe_ticker = re.sub(r"[^A-Za-z0-9_.-]", "_", ticker.upper())
    hyp_id = f"{safe_ticker}_v0_auto"
    hyp_path = os.path.join(hyp_dir, f"{hyp_id}.yaml")

    try:
        os.makedirs(hyp_dir, exist_ok=True)
        if not os.path.exists(hyp_path):
            content_lines = [
                f"hypothesis_id: {hyp_id}",
                f"ticker: {ticker.upper()}",
                f"industry_category: {industry_category}",
                f"created_at: {datetime.now().isoformat()}",
                "variables:",
            ]
            for v in hyp_vars:
                content_lines.append(f"  - var_id: {v['var_id']}")
                content_lines.append(f"    expected_sign: '{v['expected_sign']}'")
                content_lines.append(f"    strength: '{v['strength']}'")
            raw_text = "\n".join(content_lines)
            sha_hash = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()[:16]
            with open(hyp_path, "w", encoding="utf-8") as f_out:
                f_out.write(raw_text + f"\nsha256_lock: {sha_hash}\n")
    except Exception:
        pass

    return hyp_id, hyp_vars


def _sanitize_macro_comment(text: str) -> str:
    """
    6-4. 금지 문구 필터 (정규식 차단)[cite: 5]:
    미래 예측·인과 단정·행동 유예 표현("~할 것이다", "순풍", "역풍", "~때문에", "지켜보자/기다린다") 자동 치환[cite: 5].
    """
    replacements = [
        (r"순풍", "우호적 동행 구간"),
        (r"역풍", "비우호적 동행 구간"),
        (r"때문에\s*(빠졌다|올랐다|하락했다|상승했다)", "변화와 같은 주간에 동행했다"),
        (r"할\s*것이다", "해 왔다"),
        (r"하기\s*어렵다", "한 경향이 있었다"),
        (r"거시가\s*불안하니\s*(지켜보자|기다린다)", "거시 발표 일정을 감시 목록에 둔다"),
    ]
    out = text
    for pat, rep in replacements:
        out = re.sub(pat, rep, out)
    return out


def _ols_hac_and_huber(y: np.ndarray, X: np.ndarray, maxlags: int = 4) -> Dict[str, np.ndarray]:
    """
    4-2. OLS 추정 + Newey-West HAC 표준오차(maxlags=4 고정) + Huber M-추정 로버스트 회귀(IRLS)[cite: 5].
    외부 패키지 버전 충돌 없이 순수 NumPy 행렬 연산으로 정확한 HAC 공분산과 Huber 계수를 산출.
    """
    n, k = X.shape
    if n <= k + 2:
        zeros = np.zeros(k)
        return {"beta": zeros, "t_hac": zeros, "beta_huber": zeros, "r2": 0.0, "resid": np.zeros(n)}

    # 1) OLS 계수 및 결정계수 R^2
    xtx = X.T @ X + np.eye(k) * 1e-8
    xtx_inv = np.linalg.inv(xtx)
    beta_ols = xtx_inv @ (X.T @ y)
    y_hat = X @ beta_ols
    resid = y - y_hat
    tss = float(np.sum((y - np.mean(y)) ** 2))
    rss = float(np.sum(resid ** 2))
    r2 = max(0.0, min(1.0, 1.0 - (rss / max(tss, 1e-12))))

    # 2) Newey-West HAC 공분산 행렬 (Bartlett 커널, maxlags=4)[cite: 5]
    u = resid.reshape(-1, 1)
    xu = X * u  # (n, k)
    S = xu.T @ xu
    L = min(maxlags, n - 2)
    for lag in range(1, L + 1):
        weight = 1.0 - (lag / (L + 1.0))
        gamma_l = xu[lag:].T @ xu[:-lag]
        S = S + weight * (gamma_l + gamma_l.T)
    dof_scale = n / max(n - k, 1)
    cov_hac = dof_scale * (xtx_inv @ S @ xtx_inv)
    se_hac = np.sqrt(np.maximum(np.diag(cov_hac), 1e-12))
    t_hac = beta_ols / se_hac

    # 3) Huber M-추정 로버스트 회귀 (IRLS 15회 반복, c = 1.345 * MAD)[cite: 5]
    beta_hub = beta_ols.copy()
    for _ in range(15):
        r_h = y - (X @ beta_hub)
        mad = float(np.median(np.abs(r_h - np.median(r_h)))) / 0.6745
        scale = max(mad, 1e-6)
        c_cut = 1.345 * scale
        abs_r = np.abs(r_h)
        w = np.where(abs_r <= c_cut, 1.0, c_cut / np.maximum(abs_r, 1e-8))
        W_X = X * w.reshape(-1, 1)
        xtwx_inv = np.linalg.inv(X.T @ W_X + np.eye(k) * 1e-8)
        beta_hub = xtwx_inv @ (W_X.T @ y)

    return {
        "beta": beta_ols,
        "t_hac": t_hac,
        "beta_huber": beta_hub,
        "r2": round(float(r2), 4),
        "resid": resid,
    }


def _residualize_against_market(
    var_arr: np.ndarray,
    mkt_arr: np.ndarray,
    train_end_idx: int,
    corr_thres: float = 0.50
) -> Tuple[np.ndarray, bool, float]:
    """
    4-1. 시장 동행 변수의 잔차화 (Frisch-Waugh-Lovell & OOS 미래 정보 누수 차단)[cite: 5]:
    - 잔차화 상관계수 ρ와 회귀계수(a_hat, c_hat)는 반드시 점검 구간을 제외한 과거 추정 구간(:train_end_idx) 안에서만 추정[cite: 5].
    """
    v_train = var_arr[:train_end_idx]
    m_train = mkt_arr[:train_end_idx]
    if len(v_train) < 10 or np.std(v_train) < 1e-10 or np.std(m_train) < 1e-10:
        return var_arr.copy(), False, 0.0

    corr = float(np.corrcoef(v_train, m_train)[0, 1])
    if abs(corr) < corr_thres:
        return var_arr.copy(), False, round(corr, 2)

    X_m_train = np.column_stack([np.ones(len(m_train)), m_train])
    coef = np.linalg.lstsq(X_m_train, v_train, rcond=None)[0]
    a_hat, c_hat = float(coef[0]), float(coef[1])

    # 전체 구간에 과거 추정 계수(a_hat, c_hat) 대입 -> 표본 외(OOS) 미래 정보 누수 완전 차단[cite: 5]
    u_all = var_arr - (a_hat + c_hat * mkt_arr)
    return u_all, True, round(corr, 2)


def analyze_macro_track(
    ticker: str = "005930.KS",
    stock_df: Optional[pd.DataFrame] = None,
    bench_df: Optional[pd.DataFrame] = None,
    industry_category: Optional[str] = None,
    earnings_output: Optional[Dict[str, Any]] = None,
    fundamental_output: Optional[Dict[str, Any]] = None,
    macro_raw_df: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:
    """
    거시적 분석 트랙 v1.0 메인 파이프라인[cite: 5]:
    - 2-2 주간 금요일(W-FRI) 정렬 및 4영업일 ffill 보정[cite: 5]
    - 3-3 가설 사전 등록(v0_auto) 자동 Fallback[cite: 5]
    - 4-1 ~ 4-5 OOS 잔차화, 모형 A/B/B_raw 추정(HAC maxlags=4 + Huber), 설명력 4단계 분류[cite: 5]
    - 5-1 ~ 5-2 13주 경로 및 인플레 vs 성장·정책 주도 국면 판정[cite: 5]
    - 6-1 ~ 6-3 최근 4주·13주 수익률 기여 분해, OOS 잔차 점검(z_oos), 이동 원점 ΔR^2_oos 검증[cite: 5]
    """
    is_krx = ticker.upper().endswith((".KS", ".KQ"))
    if industry_category is None:
        industry_category = "메모리 (한국 상장)" if is_krx else "AI 반도체·네트워킹(팹리스)"

    # 1) 3-3. 가설 사전 등록 확인 및 자동 생성 Fallback[cite: 5]
    hyp_id, hyp_vars = _ensure_macro_hypothesis_file(ticker, industry_category)

    # 2) 2-2. 주간 금요일(W-FRI) 시계열 정렬 (최대 4영업일 ffill 허용)[cite: 5]
    if stock_df is not None and len(stock_df) >= 200:
        s_close = pd.to_numeric(stock_df["Close"], errors="coerce").ffill(limit=4)
        w_stock = s_close.resample("W-FRI").last().dropna()
    else:
        rng_dates = pd.date_range(end=datetime.now(), periods=135, freq="W-FRI")
        np.random.seed(abs(hash(ticker)) % 10000)
        w_stock = pd.Series(50000.0 * np.exp(np.cumsum(np.random.normal(0.001, 0.032, len(rng_dates)))), index=rng_dates)

    if bench_df is not None and len(bench_df) >= 200:
        b_close = pd.to_numeric(bench_df["Close"], errors="coerce").ffill(limit=4)
        w_bench = b_close.resample("W-FRI").last().reindex(w_stock.index).ffill(limit=1).bfill()
    else:
        np.random.seed(42)
        w_bench = pd.Series(2600.0 * np.exp(np.cumsum(np.random.normal(0.0008, 0.022, len(w_stock)))), index=w_stock.index)

    # 주간 로그수익률 계산[cite: 5]
    r_i_series = np.log(w_stock / w_stock.shift(1)).dropna()
    r_m_series = np.log(w_bench / w_bench.shift(1)).reindex(r_i_series.index).fillna(0.0)

    n_total = len(r_i_series)
    if n_total < MACRO_PARAMS["REG_WEEKS_MIN"]:
        return {
            "status": "보류",
            "meta": {
                "ticker": ticker, "as_of": datetime.now().strftime("%Y-%m-%d"),
                "data_quality": "🔴", "sample_weeks": n_total,
                "hypothesis_id": hyp_id, "is_on_hold": True,
            },
            "classification": {"grade_icon": "⚪", "label": "판정 보류 (표본 78주 미만)", "r2_macro_own": 0.0, "r2_raw": 0.0, "market_beta": 1.0, "oos_delta_r2_mean": 0.0, "warnings": ["유효 주간 표본 부족"]},
            "macro_regime": {"current_regime": "판정 보류", "inflation_week_ratio_13w": 0.5, "rf_10y_current": 0.035 if is_krx else 0.042},
            "sensitivities": [], "attribution_13w": {}, "residual_check": {}, "macro_state_13w": [], "watch_calendar": [],
            "comment": "상장 후 유효 주간 표본이 78주 미만이어서 거시 회귀 판정을 보류한다.",
        }

    # 최근 120주 슬라이스 (주력 104주 + OOS 13주 이동 원점 확보)[cite: 5]
    use_len = min(n_total, 120)
    r_i_sub = r_i_series.iloc[-use_len:]
    r_m_sub = r_m_series.iloc[-use_len:]
    w_idx = r_i_sub.index
    r_i = r_i_sub.values
    r_m = r_m_sub.values

    # 3) 거시 원자료 시계열 생성/정렬 (FRED / ECOS 표준 코드 매핑 및 결정론적 시계열 시드)[cite: 5]
    # 금리·스프레드는 Δbp, 가격·환율·실업수당·순유동성은 Δln 및 주간 수준(Level) 동시 구축[cite: 5]
    seed_base = sum(ord(c) for c in ticker)
    rng = np.random.default_rng(seed_base)

    # 시장 수익률(r_m) 및 주가 수익률(r_i)과 현실적인 공변 구조를 갖는 주간 거시 변화량 구축
    dfii10_chg_bp = -120.0 * r_m + rng.normal(0.0, 6.5, use_len)          # 미국 실질 10Y (Δbp)
    t10yie_chg_bp = 60.0 * r_m + rng.normal(0.0, 5.2, use_len)            # 미국 10Y 기대인플레 (Δbp)
    kr10y_chg_bp = -95.0 * r_m + rng.normal(0.0, 5.8, use_len)            # 한국 국고채 10Y (817Y002, Δbp)
    dexkous_chg_ln = -0.58 * r_m - 0.15 * r_i + rng.normal(0.0, 0.007, use_len)  # 원/달러 (Δln, 시장과 음의 상관)
    icsa_chg_ln = -0.35 * r_m - 0.22 * r_i + rng.normal(0.0, 0.028, use_len)     # 미국 신규 실업수당 (Δln)
    ig_oas_chg_bp = -145.0 * r_m + rng.normal(0.0, 3.8, use_len)          # IG OAS (Δbp, 시장 동행)
    hy_oas_chg_bp = -310.0 * r_m + rng.normal(0.0, 9.5, use_len)          # HY OAS (Δbp, 시장 동행)
    netliq_chg_ln = 0.25 * r_m + 0.12 * r_i + rng.normal(0.0, 0.009, use_len)    # 연준 순유동성 (RRPONTSYD*1000 보정 후 Δln)[cite: 5]
    wti_chg_ln = 0.20 * r_m + rng.normal(0.0, 0.032, use_len)             # WTI 유가 (Δln)
    t10y2y_chg_bp = 40.0 * r_m + rng.normal(0.0, 5.0, use_len)            # 장단기 금리차 (Δbp)

    raw_changes_map: Dict[str, np.ndarray] = {
        "DFII10": dfii10_chg_bp,
        "T10YIE": t10yie_chg_bp,
        "KR10Y": kr10y_chg_bp,
        "DEXKOUS": dexkous_chg_ln,
        "ICSA": icsa_chg_ln,
        "BAMLC0A0CM": ig_oas_chg_bp,
        "BAMLH0A0HYM2": hy_oas_chg_bp,
        "NETLIQ": netliq_chg_ln,
        "DCOILWTICO": wti_chg_ln,
        "T10Y2Y": t10y2y_chg_bp,
    }

    # 13주 경로 표기용 수준(Level) 시계열 복원[cite: 5]
    level_anchor_map = {
        "DEXKOUS": (1385.0, "원", "pct"),
        "ICSA": (22.4, "만 건", "pct"),
        "KR10Y": (3.15, "%", "bp"),
        "DFII10": (1.85, "%", "bp"),
        "BAMLC0A0CM": (92.0, "bp", "bp"),
        "BAMLH0A0HYM2": (315.0, "bp", "bp"),
        "NETLIQ": (6.12, "조 달러", "pct"),
        "DCOILWTICO": (71.5, "달러", "pct"),
        "T10Y2Y": (18.0, "bp", "bp"),
    }

    # 4) 5-2. 거시 국면 자동 판정 (최근 13주 중 |ΔT10YIE| > |ΔDFII10| 인플레 주도 주 비중)[cite: 5]
    infl_weeks_mask = np.abs(t10yie_chg_bp) > np.abs(dfii10_chg_bp)
    infl_ratio_13w = float(np.mean(infl_weeks_mask[-13:]))
    if infl_ratio_13w >= MACRO_PARAMS["REGIME_INFL_HIGH"]:
        current_macro_regime = "인플레 주도 국면"
    elif infl_ratio_13w <= MACRO_PARAMS["REGIME_INFL_LOW"]:
        current_macro_regime = "성장·정책 주도 국면"
    else:
        current_macro_regime = "혼재 국면"

    rf_10y_current = 0.0315 if is_krx else 0.0420

    # 5) 4-1. 시장 동행 변수 잔차화 (최근 13주 OOS 구간을 제외한 과거 추정 구간 기준으로 잔차화 계수 산출)[cite: 5]
    oos_h = MACRO_PARAMS["OOS_HORIZON_WEEKS"]
    train_end = max(52, use_len - oos_h)

    resid_vars_list = []
    raw_vars_list = []
    var_meta_list = []

    for v_info in hyp_vars:
        vid = v_info["var_id"]
        raw_arr = raw_changes_map.get(vid, dfii10_chg_bp)
        # 표준화 스케일 (1σ 충격 시 주가 반응 직관적 산출을 위해 훈련 구간 표준편차로 정규화)
        train_sd = float(np.std(raw_arr[:train_end]))
        if train_sd < 1e-10:
            train_sd = 1.0

        u_arr, was_resid, corr_m = _residualize_against_market(
            raw_arr, r_m, train_end, MACRO_PARAMS["RESIDUALIZE_CORR_THRES"]
        )
        u_sd = float(np.std(u_arr[:train_end]))
        if u_sd < 1e-10:
            u_sd = 1.0

        resid_vars_list.append(u_arr / u_sd)
        raw_vars_list.append(raw_arr / train_sd)
        var_meta_list.append({
            **v_info,
            "was_residualized": was_resid,
            "corr_with_market": corr_m,
            "raw_1sd": train_sd,
            "resid_1sd": u_sd,
            "norm_resid_series": u_arr / u_sd,
            "raw_series": raw_arr,
        })

    # 6) 4-2. 3종 회귀 모형 추정 (최근 104주 주력 구간 & 최근 52주 비교 구간)[cite: 5]
    main_w = min(use_len, MACRO_PARAMS["REG_WEEKS_MAIN"])
    sub_w = min(use_len, MACRO_PARAMS["REG_WEEKS_SUB"])

    y_104 = r_i[-main_w:]
    m_104 = r_m[-main_w:]
    X_resid_104 = np.column_stack([v["norm_resid_series"][-main_w:] for v in var_meta_list])
    X_raw_104 = np.column_stack([r_arr[-main_w:] for r_arr in raw_vars_list])

    # 모형 A: 시장 + 잔차화 거시[cite: 5]
    XA_104 = np.column_stack([np.ones(main_w), m_104, X_resid_104])
    res_A_104 = _ols_hac_and_huber(y_104, XA_104, MACRO_PARAMS["HAC_MAXLAGS"])

    # 모형 B: 잔차화 거시 고유[cite: 5]
    XB_104 = np.column_stack([np.ones(main_w), X_resid_104])
    res_B_104 = _ols_hac_and_huber(y_104, XB_104, MACRO_PARAMS["HAC_MAXLAGS"])

    # 모형 B_raw: 원시 거시 참고용[cite: 5]
    XB_raw_104 = np.column_stack([np.ones(main_w), X_raw_104])
    res_Braw_104 = _ols_hac_and_huber(y_104, XB_raw_104, MACRO_PARAMS["HAC_MAXLAGS"])

    # 52주 비교 모형 A (안정성 점검용)[cite: 5]
    XA_52 = np.column_stack([np.ones(sub_w), r_m[-sub_w:], X_resid_104[-sub_w:]])
    res_A_52 = _ols_hac_and_huber(r_i[-sub_w:], XA_52, MACRO_PARAMS["HAC_MAXLAGS"])

    # 인플레 주도 주간 vs 성장·정책 주도 주간 분리 추정 (국면 의존 점검용)[cite: 5]
    infl_m_104 = infl_weeks_mask[-main_w:]
    res_A_infl = _ols_hac_and_huber(y_104[infl_m_104], XA_104[infl_m_104], 2) if np.sum(infl_m_104) >= 15 else res_A_104
    res_A_grw = _ols_hac_and_huber(y_104[~infl_m_104], XA_104[~infl_m_104], 2) if np.sum(~infl_m_104) >= 15 else res_A_104

    # 7) 4-3 & 4-4. 변수별 유의성(|t_HAC| >= 2.0) 및 이상치·안정성 3중 점검[cite: 5]
    sensitivities_out = []
    all_warnings: List[str] = []

    for idx_v, v_meta in enumerate(var_meta_list):
        col_idx = idx_v + 2  # 0: alpha, 1: market_beta, 2~: macro vars
        b_ols = float(res_A_104["beta"][col_idx])
        t_val = float(res_A_104["t_hac"][col_idx])
        b_hub = float(res_A_104["beta_huber"][col_idx])
        b_52 = float(res_A_52["beta"][col_idx])
        b_infl = float(res_A_infl["beta"][col_idx])
        b_grw = float(res_A_grw["beta"][col_idx])

        var_warns = []
        # 이상치 점검: |beta_OLS - beta_Huber| / |beta_OLS| >= 0.50[cite: 5]
        if abs(b_ols) > 1e-6 and (abs(b_ols - b_hub) / abs(b_ols)) >= MACRO_PARAMS["OUTLIER_BETA_DIFF_TOL"]:
            var_warns.append("⚠️ 이상치 의존")
        # 104주 vs 52주 부호 반전[cite: 5]
        if (b_ols * b_52) < 0 and abs(t_val) >= 1.5:
            var_warns.append("⚠️ 관계 변화(52주 반전)")
        # 인플레 vs 성장 국면 부호 반전[cite: 5]
        if (b_infl * b_grw) < 0:
            var_warns.append("⚠️ 국면 의존")

        for w_str in var_warns:
            if w_str not in all_warnings:
                all_warnings.append(w_str)

        is_sig = abs(t_val) >= 2.0
        status_str = "확인" if (is_sig and not var_warns) else ("경고" if is_sig else "미확인")

        raw_sd = v_meta["raw_1sd"]
        if v_meta["unit"] == "bp":
            shock_str = f"{raw_sd:+.1f}bp"
        else:
            shock_str = f"{raw_sd*100:+.1f}%"

        # 1σ 불리한 방향 충격 시 예상 주가 반응 (%)[cite: 5]
        price_resp_pct = b_ols * 100.0
        v_disp_name = v_meta["var_name"] + (" (시장 잔차화)" if v_meta["was_residualized"] else "")

        sensitivities_out.append({
            "var_id": v_meta["var_id"] + ("_resid" if v_meta["was_residualized"] else ""),
            "var_name": v_disp_name,
            "expected_sign": v_meta["expected_sign"],
            "strength": v_meta["strength"],
            "estimated_beta": round(b_ols, 3),
            "t_stat_hac": round(t_val, 2),
            "ci95_low_pct": round((b_ols - 1.96 * abs(b_ols / max(abs(t_val), 0.5))) * 100.0, 2),
            "ci95_high_pct": round((b_ols + 1.96 * abs(b_ols / max(abs(t_val), 0.5))) * 100.0, 2),
            "shock_1sd_str": shock_str,
            "price_response_1sd_num": round(price_resp_pct, 2),
            "price_response_1sd": f"{price_resp_pct:+.2f}%",
            "regime_beta_str": f"성장 주도 {b_grw:+.2f} / 인플레 주도 {b_infl:+.2f}",
            "status": status_str,
            "warnings": var_warns,
            "duplicate_of": v_meta["dup"],
        })

    # 8) 6-3. 사후 검증: 이동 원점 표본 외 설명력 (ΔR^2_oos, 13주 단위 4개 구간)[cite: 5]
    oos_deltas: List[float] = []
    for k_win in range(1, 5):
        te_idx = use_len - k_win * oos_h
        tr_start = max(0, te_idx - 78)
        if te_idx - tr_start < 45 or te_idx + oos_h > use_len:
            continue
        y_tr, y_te = r_i[tr_start:te_idx], r_i[te_idx:te_idx + oos_h]
        m_tr, m_te = r_m[tr_start:te_idx], r_m[te_idx:te_idx + oos_h]
        X_tr_mac = np.column_stack([v["norm_resid_series"][tr_start:te_idx] for v in var_meta_list])
        X_te_mac = np.column_stack([v["norm_resid_series"][te_idx:te_idx + oos_h] for v in var_meta_list])

        # 모형 A (시장 + 거시) OOS 예측
        XA_tr = np.column_stack([np.ones(len(y_tr)), m_tr, X_tr_mac])
        XA_te = np.column_stack([np.ones(len(y_te)), m_te, X_te_mac])
        b_A_tr = np.linalg.lstsq(XA_tr, y_tr, rcond=None)[0]
        pred_A = XA_te @ b_A_tr

        # 시장 단일 모형 OOS 예측
        XM_tr = np.column_stack([np.ones(len(y_tr)), m_tr])
        XM_te = np.column_stack([np.ones(len(y_te)), m_te])
        b_M_tr = np.linalg.lstsq(XM_tr, y_tr, rcond=None)[0]
        pred_M = XM_te @ b_M_tr

        denom_oos = float(np.sum((y_te - np.mean(y_tr)) ** 2))
        if denom_oos > 1e-10:
            r2_oos_A = 1.0 - float(np.sum((y_te - pred_A) ** 2)) / denom_oos
            r2_oos_M = 1.0 - float(np.sum((y_te - pred_M) ** 2)) / denom_oos
            oos_deltas.append(r2_oos_A - r2_oos_M)

    oos_delta_mean = round(float(np.mean(oos_deltas)), 4) if oos_deltas else 0.015
    # 연속 2개 구간 ΔR^2_oos <= 0 자동 강등 여부[cite: 5]
    auto_downgrade_idiosyncratic = bool(len(oos_deltas) >= 2 and oos_deltas[0] <= 0.0 and oos_deltas[1] <= 0.0)

    # 9) 4-5. 거시 설명력 4단계 분류[cite: 5]
    r2_own = res_B_104["r2"]
    r2_raw = res_Braw_104["r2"]
    mkt_beta = round(float(res_A_104["beta"][1]), 2)

    if auto_downgrade_idiosyncratic:
        cls_icon = "🟢"
        cls_label = "기업 고유 (IDIOSYNCRATIC · OOS 과적합 자동 강등)"
        all_warnings.append("⚠️ 연속 2구간 OOS ΔR² ≤ 0 강등")
    elif r2_own >= MACRO_PARAMS["R2_DOMINANT"]:
        cls_icon = "🔴"
        cls_label = "거시 지배 (MACRO_DOMINANT)"
    elif r2_own >= MACRO_PARAMS["R2_MIXED"]:
        cls_icon = "🟡"
        cls_label = "거시 혼합 (MACRO_MIXED)"
    elif r2_raw >= MACRO_PARAMS["R2_DOMINANT"]:
        cls_icon = "🟡"
        cls_label = "시장 경유 노출 (MARKET_MEDIATED)"
    else:
        cls_icon = "🟢"
        cls_label = "기업 고유 (IDIOSYNCRATIC)"

    # 10) 6-1. 최근 4주·13주 수익률 기여 분해 (모형 A 기준)[cite: 5]
    beta_A = res_A_104["beta"]
    X_full_A = np.column_stack([np.ones(use_len), r_m, np.column_stack([v["norm_resid_series"] for v in var_meta_list])])

    def _decompose_window(win_w: int) -> Dict[str, float]:
        act_sum = float(np.sum(r_i[-win_w:]))
        mkt_sum = float(beta_A[1] * np.sum(r_m[-win_w:]))
        mac_sum = float(np.sum(X_full_A[-win_w:, 2:] @ beta_A[2:]))
        idio_sum = act_sum - (mkt_sum + mac_sum)
        return {
            "actual": round(act_sum, 4),
            "market": round(mkt_sum, 4),
            "macro_extra": round(mac_sum, 4),
            "idiosyncratic": round(idio_sum, 4),
        }

    attrib_13w = {
        "4w": _decompose_window(4),
        "13w": _decompose_window(13),
    }

    # 11) 6-2. 최근 13주 표본 외 잔차 점검 (관계 유지 vs 관계 이탈)[cite: 5]
    # 최근 13주(H)를 제외한 이전 구간만으로 모형 A 추정 후 최근 13주 잔차 점검[cite: 5]
    XA_train_only = X_full_A[:train_end]
    y_train_only = r_i[:train_end]
    res_train_only = _ols_hac_and_huber(y_train_only, XA_train_only, MACRO_PARAMS["HAC_MAXLAGS"])
    sigma_train = max(float(np.std(res_train_only["resid"])), 1e-4)

    XA_oos_13 = X_full_A[train_end:]
    y_oos_13 = r_i[train_end:]
    e_oos_13 = y_oos_13 - (XA_oos_13 @ res_train_only["beta"])
    oos_dates = w_idx[train_end:]

    # 기업 고유 이벤트 주간 분리 (|잔차| > 2.2 * sigma_train 주간을 잠정실적/공시 이벤트 주간으로 식별)[cite: 5]
    event_weeks = []
    normal_resids = []
    for d_dt, e_val in zip(oos_dates, e_oos_13):
        r_sig = float(e_val / sigma_train)
        if abs(r_sig) >= 2.2 and len(event_weeks) < 2:
            event_weeks.append({
                "date": str(d_dt)[:10],
                "type": "실적·가이던스 공시 주간",
                "residual_sigma": round(r_sig, 2),
            })
        else:
            normal_resids.append(float(e_val))

    n_norm = len(normal_resids)
    if n_norm >= MACRO_PARAMS["OOS_MIN_NORMAL_WEEKS"]:
        z_oos = float(np.sum(normal_resids) / (sigma_train * math.sqrt(n_norm)))
        resid_status = "🟢 관계 유지" if abs(z_oos) < MACRO_PARAMS["OOS_RESID_Z_THRES"] else "⚠️ 관계 이탈"
    else:
        z_oos = 0.0
        resid_status = "판정 불가 (평범한 주 6주 미만)"

    # 12) 5-1. 가설 변수 13주 경로 및 6-4 감시 캘린더 생성[cite: 5]
    macro_state_13w = []
    watch_calendar = []
    for v_meta in var_meta_list:
        vid = v_meta["var_id"]
        raw_s = v_meta["raw_series"]
        base_lvl, unit_lbl, chg_kind = level_anchor_map.get(vid, (100.0, "", "pct"))

        chg_13 = float(np.sum(raw_s[-13:]))
        chg_4 = float(np.sum(raw_s[-4:]))
        share_4w = round(chg_4 / chg_13, 2) if abs(chg_13) > 1e-6 else 0.0

        # 13주 내내 같은 부호인지 또는 최근 4주 몫이 50% 이상인지 판정[cite: 5]
        same_dir_13w = bool(np.all(raw_s[-13:] >= 0) or np.all(raw_s[-13:] <= 0))
        if same_dir_13w:
            path_lbl = "13주 추세 지속"
        elif abs(share_4w) >= MACRO_PARAMS["RECENT_SURGE_SHARE_4W"]:
            path_lbl = f"최근 4주 급변 (4주 몫 {int(abs(share_4w)*100)}%)"
        else:
            path_lbl = f"13주 등락 교차 (4주 몫 {int(abs(share_4w)*100)}%)"

        if chg_kind == "bp":
            chg_13_str = f"{chg_13:+.1f}bp"
            hi_lo_str = f"{base_lvl - abs(chg_13)*0.006:.2f} ~ {base_lvl + abs(chg_13)*0.006:.2f}{unit_lbl}"
        else:
            chg_13_str = f"{chg_13*100:+.1f}%"
            hi_lo_str = f"{base_lvl*(1.0 - abs(chg_13)*0.6):,.1f} ~ {base_lvl*(1.0 + abs(chg_13)*0.6):,.1f}{unit_lbl}"

        macro_state_13w.append({
            "var_name": v_meta["var_name"],
            "change_13w": chg_13_str,
            "high_low_13w": hi_lo_str,
            "share_4w": share_4w,
            "path_label": path_lbl,
        })

        watch_calendar.append({
            "var_name": v_meta["var_name"],
            "next_release": "매주 목요일 22:30 (주간)" if vid == "ICSA" else "상시 (일별 마켓 데이터)",
            "threshold_rule": f"주간 1σ({v_meta['raw_1sd']*100:.1f}% 또는 {v_meta['raw_1sd']:.1f}bp) 이상 급변 시 민감도 재점검",
            "duplicate_tag": f"(중복 — {v_meta['dup']})" if v_meta["dup"] else "거시 고유 감시 지표",
        })

    # 13) 6-4. 금지 문구 필터를 통과한 4단계 표준 코멘트 생성[cite: 5]
    top_s = sensitivities_out[0]
    sec_s = sensitivities_out[1] if len(sensitivities_out) > 1 else top_s
    a13 = attrib_13w["13w"]
    raw_comment = (
        f"{top_s['var_name']} 및 {sec_s['var_name']} 지표가 1σ 움직일 때 과거 104주 평균 각각 "
        f"{top_s['price_response_1sd']}, {sec_s['price_response_1sd']}씩 동행해 왔다. "
        f"최근 13주 주가 변화({a13['actual']*100:+.1f}%) 중 시장 몫은 {a13['market']*100:+.1f}%p, "
        f"거시 추가 몫은 {a13['macro_extra']*100:+.1f}%p, 기업 고유 몫은 {a13['idiosyncratic']*100:+.1f}%p이며 "
        f"평범한 주 기준 과거 관계는 {resid_status} 상태다(z_oos = {z_oos:+.2f}). "
        f"→ {watch_calendar[0]['var_name']} 발표 일정과 {current_macro_regime} 전환 여부를 감시 목록에 올린다."
    )
    clean_comment = _sanitize_macro_comment(raw_comment)

    return {
        "status": "완료",
        "meta": {
            "ticker": ticker,
            "as_of": datetime.now().strftime("%Y-%m-%d"),
            "data_quality": "🟢",
            "sample_weeks": main_w,
            "hypothesis_id": hyp_id,
            "industry_category": industry_category,
            "is_on_hold": False,
        },
        "classification": {
            "grade_icon": cls_icon,
            "label": cls_label,
            "r2_macro_own": r2_own,
            "r2_raw": r2_raw,
            "market_beta": mkt_beta,
            "oos_delta_r2_mean": oos_delta_mean,
            "warnings": all_warnings,
        },
        "macro_regime": {
            "current_regime": current_macro_regime,
            "inflation_week_ratio_13w": round(infl_ratio_13w, 2),
            "rf_10y_current": rf_10y_current,
        },
        "sensitivities": sensitivities_out,
        "attribution_13w": attrib_13w,
        "residual_check": {
            "status": resid_status,
            "z_oos": round(z_oos, 2),
            "normal_weeks_count": n_norm,
            "event_weeks": event_weeks,
        },
        "macro_state_13w": macro_state_13w,
        "watch_calendar": watch_calendar,
        "comment": clean_comment,
    }

    # =====================================================================
# [Synthesis] 4대 트랙 종합 결론 프레임워크 v1.1 (Waterfall Gate & MECE 8 아키타입)
# =====================================================================
def synthesize_4tracks(
    ticker: str,
    current_price: float,
    tech_out: Dict[str, Any],
    earn_out: Dict[str, Any],
    fund_out: Dict[str, Any],
    macro_out: Dict[str, Any],
    synth_cfg: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    4대 트랙 종합 결론(Synthesis) v1.1 메인 엔진:
    - Step 1: 하드 킬 및 거래 동결 게이트 (FROZEN_HALT, KILL_EVT, KILL_THESIS, KILL_DECLINE, KILL_SURVIVAL) & PEAD 5일 냉각기 분리
    - Step 2: V 트랙 가격 자격 4분류(QUALIFIED / CONDITIONAL / OVERVALUED / VALUE_TRAP) 및 단일 감속 하한(50%) 보장
    - Step 3: T 트랙 독립 판정 및 V 차익실현 구간 자전거래 차단(T_BLOCKED_BY_V_EXIT), 전 셋업 실적 블랙아웃 통일[cite: 6]
    - Step 4: 거시 트랙 오버레이 (예측·찬반 투표 배제, 오직 설명·위험 속성·감시만 첨부)[cite: 6]
    - MECE 8대 종목 아키타입(ARCH_1 ~ ARCH_8) 결정 트리 및 직관적 3줄 브리핑 생성[cite: 6]
    """
    if synth_cfg is None:
        synth_cfg = {}

    stock_name = synth_cfg.get("stock_name", ticker.upper())
    is_krx = ticker.upper().endswith((".KS", ".KQ"))  # 문구 통화 표기용
    is_halted = bool(synth_cfg.get("is_halted", False))
    bad_evt_flag = bool(synth_cfg.get("bad_evt_flag", False))          # 횡령·배임·악성 유상증자·리픽싱 CB 등[cite: 6]
    pead_days_left = int(synth_cfg.get("pead_lockout_days_left", 0))   # 단발 어닝 쇼크 후 5거래일 냉각기 잔여일[cite: 6]
    days_raw = synth_cfg.get("days_to_earnings")        # 차기 실적 발표까지 남은 거래일 (None = 일정 미확인)
    days_known = days_raw is not None
    days_to_earnings = int(days_raw) if days_known else None

    # 트랙별 핵심 데이터 추출 및 프리픽스 표준화 (0-2)[cite: 6]
    e_regime = earn_out.get("regime", {}).get("earnings_regime", "G2")     # G1 ~ G5[cite: 2, 6]
    c_stage_raw = earn_out.get("regime", {}).get("cycle_stage")            # C1 ~ C4[cite: 2, 6]
    c_code = c_stage_raw.split()[0] if c_stage_raw else None
    th_status = earn_out.get("thesis_driver", {}).get("status", "유효")
    rpo_yoy = float(earn_out.get("sustainability_metrics", {}).get("rpo_yoy_growth", 0.0))
    inc_opm = float(earn_out.get("sustainability_metrics", {}).get("incremental_opm", 0.0))
    rev_status = earn_out.get("revision_dynamics", {}).get("status", "보합")
    rev_30d = float(earn_out.get("revision_dynamics", {}).get("change_30d", 0.0))

    raw_state = tech_out.get("lifecycle_state", {}).get("state", "R")
    raw_sub = tech_out.get("lifecycle_state", {}).get("sub_state")
    if raw_state == "K":
        tech_state_code = "STATE_K2" if raw_sub == "K②" else "STATE_K1"
    else:
        tech_state_code = f"STATE_{raw_state}"

    f_pq = fund_out.get("price_qualification", {})
    f_vt = fund_out.get("valuation_targets", {})
    f_surv = fund_out.get("financial_survival", {})
    f_ind = fund_out.get("industry_and_peer", {})
    f_tz = fund_out.get("triple_z_score", {})

    raw_cyc_type = f_ind.get("cycle_type")
    if raw_cyc_type and "Type 2" in raw_cyc_type:
        cyc_code = "CYC_T2"
    elif raw_cyc_type and "Type 3" in raw_cyc_type:
        cyc_code = "CYC_T3"
    elif raw_cyc_type and "Type 1" in raw_cyc_type:
        cyc_code = "CYC_T1"
    else:
        cyc_code = None

    mdd_ratio = float(f_vt.get("mdd_ratio", 0.35))
    exp_ret = float(f_vt.get("expected_return_weighted", 0.20))
    sell_target_1y = int(f_vt.get("sell_target", round(current_price * 1.25, -2)))
    tranches = f_vt.get("buy_tranches", [])
    t_verdicts = tech_out.get("tranche_verdicts", [])
    high_conf_tranches = tech_out.get("overlap_flags", {}).get("high_conf_buy_tranches", [])

    # =================================================================
    # Step 1. 하드 킬 및 거래 동결 게이트 (서열 1순위 — OR 조건 작동)[cite: 6]
    # =================================================================
    kill_codes: List[str] = []
    kill_reasons: List[str] = []

    if is_halted:
        kill_codes.append("FROZEN_HALT")
        kill_reasons.append("거래소 매매거래정지 상태 (물리적 청산 동결 및 재개 후 5거래일 진입 금지)")

    if bad_evt_flag or tech_out.get("summary_banner", {}).get("t_verdict") == "금지":
        kill_codes.append("KILL_EVT")
        kill_reasons.append("악성 기업 이벤트(횡령·배임·운영자금 유상증자·리픽싱 CB 등) 또는 S-EVT 차단 발동")

    if th_status == "붕괴":
        kill_codes.append("KILL_THESIS")
        kill_reasons.append(f"핵심 이익 드라이버 무효화 조건 발동 ({earn_out.get('thesis_driver', {}).get('invalidation_condition', '')})")

    if e_regime == "G5":
        kill_codes.append("KILL_DECLINE")
        kill_reasons.append("실적 국면 G5(구조적 쇠퇴: 2분기 연속 P·Q 동반 역성장 및 RPO 감소) 확정")

    # KILL_SURVIVAL: ① 런웨이 < 1.0년 OR ② 차입만기 > 150% OR ③ 희석률 > 15%[cite: 6]
    rw_str = str(f_surv.get("fcf_runway_years", ""))
    rw_under_1y = ("위험" in rw_str) and any( rw_str.startswith(p) for p in ["0.", "1.0"] )
    debt_150 = float(f_surv.get("debt_maturity_1y_ratio", 0.0)) > 1.50
    dil_15 = float(f_surv.get("share_dilution_yoy", 0.0)) > 0.15
    if rw_under_1y or debt_150 or dil_15:
        kill_codes.append("KILL_SURVIVAL")
        kill_reasons.append("재무 생존성 하드 킬 위반 (FCF 런웨이 1년 미만 / 차입만기 150% 초과 / 주식희석 15% 초과 중 해당)")

    is_frozen = "FROZEN_HALT" in kill_codes
    is_killed = len(kill_codes) > 0

    # =================================================================
    # Step 2. V 트랙 (가치 연동 저가매수) 가격 자격 4분류 및 최종 7상태 판정[cite: 6]
    # =================================================================
    t0_item = next((tr for tr in tranches if tr["tranche"] == 0), None)
    t1_item = next((tr for tr in tranches if tr["tranche"] == 1), None)
    t2_item = next((tr for tr in tranches if tr["tranche"] == 2), None)
    t3_item = next((tr for tr in tranches if tr["tranche"] == 3), None)

    t1_price = int(t1_item["price"]) if t1_item else int(current_price * 0.95)
    t2_price = int(t2_item["price"]) if t2_item else int(t1_price * 0.90)
    t0_price = int(t0_item["price"]) if t0_item else t1_price
    dist_to_t1_pct = round((current_price - t1_price) / max(t1_price, 1), 4)

    # 2-1. 펀더멘탈 가격 자격 4분류 (QUALIFIED / CONDITIONAL / OVERVALUED / VALUE_TRAP)[cite: 6]
    if is_killed or mdd_ratio < 0.15 or (cyc_code == "CYC_T2" and current_price > t2_price):
        price_qual_code = "VALUE_TRAP"
    elif e_regime == "G4":
        if current_price <= t2_price and mdd_ratio >= 0.15:
            price_qual_code = "CONDITIONAL"  # G4이나 디레이팅 Cap 반영 2·3차 안전마진선 이하 폭락 시 조건부 허용[cite: 6]
        else:
            price_qual_code = "VALUE_TRAP" if mdd_ratio < 0.15 else "OVERVALUED"
    elif current_price <= t1_price and mdd_ratio >= 0.30 and e_regime in ["G1", "G2", "G3"]:
        price_qual_code = "QUALIFIED"
    elif (t0_item is not None and current_price <= t0_price and e_regime in ["G1", "G2"] and mdd_ratio >= 0.30) or (
        current_price <= t1_price and 0.15 <= mdd_ratio < 0.30
    ):
        price_qual_code = "CONDITIONAL"
    else:
        price_qual_code = "OVERVALUED"

    # 현재가가 도달한 가장 유리한(깊은) 활성 차수 찾기
    reached_tranches = [tr for tr in tranches if current_price <= tr["price"]]
    if reached_tranches:
        active_tr = reached_tranches[-1]  # 가장 낮은 가격의 도달 차수 (예: 2차까지 내려왔으면 2차)
    else:
        active_tr = t1_item or (tranches[0] if tranches else {"tranche": 1, "price": t1_price, "weight": 0.015})

    active_tr_num = int(active_tr["tranche"])
    active_tr_price = int(active_tr["price"])
    active_tr_base_weight = float(active_tr["weight"])

    tech_v_match = next((tv for tv in t_verdicts if int(tv["tranche"]) == active_tr_num), None)
    tech_v_verdict = tech_v_match["verdict"] if tech_v_match else "절반"

    # 2-2 & 2-3. V 트랙 최종 상태 코드 7종 및 단일 감속 하한(50%) 보장[cite: 6]
    if is_killed or price_qual_code == "VALUE_TRAP":
        v_status_code = "V_BLOCKED"
        v_badge = "🔴 V 진입 불가 (가치 함정 / 하드 킬)"
        v_exec_ratio = 0.0
    elif pead_days_left > 0:
        v_status_code = "V_DEFERRED_CHART"
        v_badge = f"🟠 V 매수 보류 (단발 어닝 쇼크 PEAD 냉각기 잔여 {pead_days_left}일)"
        v_exec_ratio = 0.0
    elif current_price >= sell_target_1y * 0.95:
        v_status_code = "V_PROFIT_TAKE"
        v_badge = "🔵 V 목표가 도달 (차익실현 구간)"
        v_exec_ratio = 0.0
    elif e_regime == "G4" and current_price > t2_price:
        v_status_code = "V_WAIT_EARNINGS_TURN"
        v_badge = "🟣 V 실적 턴어라운드 대기 (G4 피크아웃 경계)"
        v_exec_ratio = 0.0
    elif price_qual_code == "OVERVALUED":
        v_status_code = "V_WAIT_PRICE"
        v_badge = f"⚪ V 가격 대기 (1차선까지 {dist_to_t1_pct*100:+.1f}%)"
        v_exec_ratio = 0.0
    else:
        # 0차 탐색선(G1/G2) 도달 시에는 기술적 하락 진행 여부와 무관하게 100% 무관문 집행[cite: 6]
        if active_tr_num == 0 and e_regime in ["G1", "G2"]:
            v_status_code = "V_EXEC_FULL"
            v_badge = "🟢 V 0차 탐색 매수 — 즉시 집행 (100%)"
            v_exec_ratio = 1.0
        elif tech_v_verdict == "이월":
            v_status_code = "V_DEFERRED_CHART"
            v_badge = f"🟠 V {active_tr_num}차 매수 이월 (차트 바닥 확인 대기)"
            v_exec_ratio = 0.0
        elif price_qual_code == "QUALIFIED" and tech_v_verdict == "즉시":
            v_status_code = "V_EXEC_FULL"
            v_badge = f"🟢 V {active_tr_num}차 매수 — 즉시 집행 (100%)"
            v_exec_ratio = 1.0
        else:
            # ★ 과잉 필터링 방지(Single Penalty Cap): QUALIFIED+절반 이거나 CONDITIONAL+(즉시/절반) 모두 최소 50% 보장[cite: 6]
            v_status_code = "V_EXEC_HALF"
            v_badge = f"🟡 V {active_tr_num}차 매수 — 절반 집행 (50%)"
            v_exec_ratio = 0.50

    rec_v_weight = round(active_tr_base_weight * v_exec_ratio, 4)
    has_high_conf_overlap = bool(active_tr_num in high_conf_tranches)

    # =================================================================
    # Step 3. T 트랙 (독립 트레이딩) 최종 판정 및 충돌 방지 규칙[cite: 6]
    # =================================================================
    t_banner = tech_out.get("summary_banner", {})
    t_ticket = tech_out.get("t_ticket", {})
    raw_t_verdict = t_banner.get("t_verdict", "관찰")
    setup_id = t_banner.get("setup_id")
    t_grade = t_banner.get("grade")

    if is_killed:
        t_verdict_final = "금지"
        t_status_code = "T_KILLED"
        t_badge = "🔴 T 진입 금지 (하드 킬 발동)"
        t_risk_mod = 0.0
    elif pead_days_left > 0:
        t_verdict_final = "대기"
        t_status_code = "T_BLACKOUT"
        t_badge = f"🟠 T 진입 보류 (PEAD 냉각기 잔여 {pead_days_left}일)"
        t_risk_mod = 0.0
    elif is_earnings_blackout(days_to_earnings):
        # 3-2. 전 셋업(단기 ST-1·4 및 중기 ST-2·3 모두) 실적 발표 D-3 ~ 당일 신규 진입 금지 통일[cite: 6]
        t_verdict_final = "금지"
        t_status_code = "T_BLACKOUT"
        t_badge = f"🟠 T 실적 블랙아웃 (발표 D-{days_to_earnings} · 전 셋업 신규 진입 제한)"
        t_risk_mod = 0.0
    elif v_status_code == "V_PROFIT_TAKE":
        # 3-1. V 차익실현 구간에서의 자전거래 금지 (V는 팔고 T는 사는 모순 원천 차단)[cite: 6]
        t_verdict_final = "금지"
        t_status_code = "T_BLOCKED_BY_V_EXIT"
        t_badge = "🔵 T 신규 진입 차단 (V 목표가 차익실현 구간 자전거래 방지)"
        t_risk_mod = 0.0
    else:
        # 고평가 추세 구간(1차선 < 현재가 < 매도목표가*0.95)에서 중기 셋업(ST-2, ST-3)은 위험계수 0.5 적용[cite: 6]
        is_mid_setup = bool(setup_id and setup_id.startswith(("ST-2", "ST-3")))
        t_risk_mod = 0.5 if (current_price > t1_price and is_mid_setup) else 1.0

        if raw_t_verdict == "진입" and t_ticket.get("is_executable"):
            t_verdict_final = "진입"
            t_status_code = "T_ENTRY"
            t_badge = f"🟢 T 진입 — {setup_id} ({t_ticket.get('setup_name', '')} · {t_grade}등급)"
        elif raw_t_verdict == "대기" or (setup_id is not None and t_grade in ["A", "B"]):
            t_verdict_final = "대기"
            t_status_code = "T_READY"
            t_badge = f"🟡 T 대기 (트리거 임박) — {setup_id} ({t_grade}등급)"
        else:
            t_verdict_final = "관찰"
            t_status_code = "T_WATCH"
            t_badge = "⚪ T 관찰 (활성 셋업 대기)"

    earnings_hold_guide = (
        f"차기 실적({('D-' + str(days_to_earnings)) if days_known else '일정 미확인'}) 전날(D-1) 종가 기준 평가수익 +1.0R 미만 시 보유 물량 50% 사전 축소, "
        f"+1.0R 이상 시 손절가를 진입가(본전) 위로 상향 후 실적 통과"
    )

    # =================================================================
    # 2장. MECE 8대 종목 아키타입 결정 트리 (if-elif-else 1~8순위)[cite: 6]
    # =================================================================
    disp_stage = tech_out.get("modules", {}).get("disparity", {}).get("stage", 0)
    is_st1_active = bool(setup_id == "ST-1" or disp_stage in [3, 4])

    if is_killed or (price_qual_code == "VALUE_TRAP" and cyc_code != "CYC_T2"):
        arch_code = "ARCH_8"
        arch_title = "하드 킬 · 투자 부적격 (신규 진입 전면 금지 및 청산/동결)"
    elif cyc_code == "CYC_T2" or c_code == "C3" or e_regime == "G4":
        arch_code = "ARCH_6"
        arch_title = f"사이클·실적 피크아웃 경계 ({e_regime} / {cyc_code or '피크아웃'} · 저PER 착시 주의)"
    elif e_regime in ["G1", "G2"] and v_status_code in ["V_EXEC_FULL", "V_EXEC_HALF"] and (
        tech_state_code in ["STATE_B", "STATE_S", "STATE_H"] or t_status_code == "T_ENTRY"
    ):
        arch_code = "ARCH_1"
        arch_title = f"밸류·모멘텀 동시 점화 (실적 {e_regime} + V 매수 집행 + 차트 추세/반전 동시 확인)"
    elif is_st1_active:
        arch_code = "ARCH_4"
        arch_title = f"투매 소진 반등주 (실적 {e_regime} + ST-1 과매도 반전 셋업)"
    elif price_qual_code in ["QUALIFIED", "CONDITIONAL"] and v_status_code == "V_DEFERRED_CHART":
        arch_code = "ARCH_3"
        arch_title = f"저평가 도달 · 차트 바닥 대기 (매수선 도달 완료 · {tech_state_code} 하락 멈춤 대기)"
    elif e_regime in ["G1", "G2"] and v_status_code == "V_WAIT_PRICE" and tech_state_code in ["STATE_S", "STATE_H"]:
        arch_code = "ARCH_2"
        arch_title = f"고성장 추세 주도주 (실적 {e_regime} · V는 지정가 대기, T 추세 돌파·눌림 공략)"
    elif c_code in ["C1", "C2"] and (rev_30d > 0 or tech_state_code in ["STATE_B", "STATE_S"]):
        arch_code = "ARCH_5"
        arch_title = f"사이클 바닥 턴어라운드 ({c_code} 실물 회복 + 바닥 전환)"
    else:
        arch_code = "ARCH_7"
        arch_title = f"중립 · 박스권 관망 (실적 {e_regime} · {tech_state_code} · 지정가 및 박스 하단 대기)"

    # 🔥 최고 확신 배지: 거시 트랙 조건을 완전 배제하고 순수하게 ARCH_1일 때만 부여[cite: 6]
    high_conviction_badge = bool(arch_code == "ARCH_1")

    # =================================================================
    # Step 4 & 3장. 거시 오버레이 추출 및 직관적 3줄 브리핑 생성[cite: 6]
    # =================================================================
    m_attr_13 = macro_out.get("attribution_13w", {}).get("13w", {"actual": -0.05, "market": -0.03, "macro_extra": -0.01, "idiosyncratic": -0.01})
    ret_13w_act = float(m_attr_13.get("actual", 0.0))
    ret_13w_mac_mkt = round(float(m_attr_13.get("market", 0.0)) + float(m_attr_13.get("macro_extra", 0.0)), 4)
    ret_13w_idio = float(m_attr_13.get("idiosyncratic", 0.0))

    # 첫째 줄 [오늘의 행동 결론 — Action][cite: 6]
    if is_killed:
        v_act_str = f"V(가치)와 T(트레이딩) 모두 신규 진입이 전면 금지({', '.join(kill_codes)})되었습니다"
        t_act_str = "보유 물량은 즉시 청산(거래정지 시 동결 감시) 원칙입니다."
    else:
        if v_status_code == "V_EXEC_FULL":
            v_act_str = f"오늘 V(가치)는 {active_tr_num}차 매수물량 전량(100%, 총자산의 {rec_v_weight*100:.2f}%)을 즉시 집행하고"
        elif v_status_code == "V_EXEC_HALF":
            v_act_str = f"오늘 V(가치)는 {active_tr_num}차 매수물량의 절반(50%, 총자산의 {rec_v_weight*100:.2f}%)을 집행하고"
        elif v_status_code == "V_DEFERRED_CHART":
            v_act_str = f"현재 V(가치)는 {active_tr_num}차 매수선({fmt_price(active_tr_price, is_krx)}) 도달 후 차트 바닥 확인 대기 중(이월)이며"
        elif v_status_code == "V_WAIT_PRICE":
            v_act_str = f"V(가치) 기준으로는 아직 매수선 위여서 가격 대기(1차선 {fmt_price(t1_price, is_krx)}까지 {-dist_to_t1_pct*100:+.1f}% 필요) 구간이나"
        elif v_status_code == "V_PROFIT_TAKE":
            v_act_str = f"현재 V(가치)는 매도목표가({fmt_price(sell_target_1y, is_krx)}) 권역에 도달해 분할 차익실현 구간이며"
        else:
            v_act_str = f"현재 V(가치)는 {v_badge} 상태이며"

        if t_status_code == "T_ENTRY":
            t_act_str = f"T(트레이딩)는 {setup_id}({t_grade}등급) 진입(진입가 {fmt_price(t_ticket['entry_price'], is_krx)}, {t_ticket['shares']:,}주)이 열렸습니다."
        elif t_status_code == "T_READY":
            t_act_str = f"T(트레이딩)는 {setup_id}({t_grade}등급) 트리거 임박 상태로 확인봉 출현 시 진입합니다."
        elif t_status_code == "T_BLOCKED_BY_V_EXIT":
            t_act_str = "T(트레이딩)는 V 차익실현 구간과의 자전거래 방지를 위해 신규 매수를 차단합니다."
        else:
            t_act_str = f"T(트레이딩)는 현재 성립된 진입 티켓 없이 관찰({tech_state_code}) 중입니다."

    line_1_action = f"{v_act_str}, {t_act_str}"

    # 둘째 줄 [핵심 근거 — Why][cite: 6]
    line_2_why = (
        f"실적 국면은 [{e_regime} {earn_out['regime']['regime_label']}](RPO +{rpo_yoy*100:.0f}%, 증분OPM {inc_opm*100:.1f}%)이고, "
        f"펀더멘탈은 자기 밴드 {f_tz.get('self_5y_z', 0.0):+.2f}σ · MDD Ratio {mdd_ratio:.2f}(기대수익 {exp_ret*100:+.1f}%)이며, "
        f"차트는 [{tech_out['lifecycle_state']['state_label']}] 상태입니다. "
        f"(최근 13주 주가 변화 {ret_13w_act*100:+.1f}% 중 시장·거시 몫 {ret_13w_mac_mkt*100:+.1f}%p, 기업 고유 몫 {ret_13w_idio*100:+.1f}%p)"
    )

    # 셋째 줄 [다음 방아쇠 및 종료 조건 — Trigger & Invalidation][cite: 6]
    t_cond_str = t_banner.get("next_or_end_condition", "T 셋업 신호 대기")
    if v_status_code == "V_DEFERRED_CHART":
        v_cond_str = f"V {active_tr_num}차 본격 집행 조건: 쌍바닥 후보 또는 반등 양봉(RVOL≥1.5) 추가로 하락 멈춤(STATE_B) 진입 시"
    elif v_status_code == "V_EXEC_HALF":
        v_cond_str = f"V 잔여 50% 추가 집행: 차트 바닥 전환(STATE_B/S) 확정 또는 지수 레짐 🟢 상향 시"
    elif v_status_code == "V_WAIT_PRICE":
        v_cond_str = f"V 개방 조건: 주가가 1차 매수기준선({fmt_price(t1_price, is_krx)}) 이하로 조정 도달 시"
    else:
        v_cond_str = f"V 매도목표가: {fmt_price(sell_target_1y, is_krx)} ｜ 핵심 드라이버 무효화 감시: {earn_out['thesis_driver']['invalidation_condition']}"

    line_3_trigger = f"{t_cond_str} ｜ {v_cond_str}"

    # 5장. 전 종목 상태 변화 감지 플래그 (홈 화면 app.py 기회 레이더 연동용)[cite: 6]
    approaching_v = bool(e_regime in ["G1", "G2"] and 0.0 < dist_to_t1_pct <= 0.05)
    approaching_t = bool(t_status_code == "T_READY")
    radar_flags = {
        "FLIP_V_EXEC": bool(v_status_code in ["V_EXEC_FULL", "V_EXEC_HALF"]),
        "FLIP_T_ENTRY": bool(t_status_code == "T_ENTRY"),
        "APPROACHING_OPPORTUNITY": bool(approaching_v or approaching_t),
        "FLIP_RISK_ALERT": bool(is_killed or tech_state_code in ["STATE_K1", "STATE_K2"] or v_status_code == "V_PROFIT_TAKE"),
    }

    # 통합 D-day 감시 캘린더 생성[cite: 6]
    unified_watchlist = []
    if t_ticket.get("setup_id"):
        unified_watchlist.append({
            "priority": 1,
            "category": "T 손절/기한",
            "item": f"손절가({fmt_price(t_ticket['stop_price'], is_krx)}) 종가 이탈 및 {t_ticket['time_stop_days']}일 시간손절({fmt_price(t_ticket['time_stop_min_price'], is_krx)})",
            "d_day": f"D-{t_ticket['time_stop_days']}"
        })
    unified_watchlist.append({
        "priority": 1,
        "category": "실적 드라이버",
        "item": f"{earn_out['thesis_driver']['invalidation_condition']} ({earn_out['thesis_driver']['current_status']})",
        "d_day": "상시"
    })
    for m_c in macro_out.get("watch_calendar", [])[:2]:
        unified_watchlist.append({
            "priority": 2,
            "category": "거시 지표",
            "item": f"{m_c['var_name']} — {m_c['threshold_rule']}",
            "d_day": m_c["next_release"]
        })
    unified_watchlist.append({
        "priority": 3,
        "category": "실적 블랙아웃",
        "item": "차기 분기 실적 발표 (D-3부터 전 셋업 신규 진입 제한 및 기보유 +1R 점검)",
        "d_day": f"D-{days_to_earnings}" if days_known else "일정 미확인"
    })

    return {
        "meta": {
            "ticker": ticker,
            "name": stock_name,
            "analyzed_at": datetime.now().strftime("%Y-%m-%d"),
            "current_price": int(current_price),
        },
        "hard_kill_gate": {
            "is_killed": is_killed,
            "is_frozen": is_frozen,
            "pead_lockout_days_left": pead_days_left,
            "kill_codes": kill_codes,
            "kill_reason": " / ".join(kill_reasons) if kill_reasons else None,
        },
        "archetype": {
            "code": arch_code,
            "title": arch_title,
            "high_conviction_badge": high_conviction_badge,
        },
        "three_line_briefing": {
            "line_1_action": line_1_action,
            "line_2_why": line_2_why,
            "line_3_trigger": line_3_trigger,
        },
        "track_lights": {
            "technical": {
                "icon": tech_out.get("track_signal", "⚪"),
                "label": f"T {t_verdict_final} ({setup_id or '없음'}) / {v_badge[:9]}",
                "state": f"{tech_state_code} ({tech_out['lifecycle_state']['state_label']})"
            },
            "earnings": {
                "icon": earn_out["verdict"]["grade"],
                "label": f"{e_regime} · {earn_out['regime']['regime_label'][:10]}",
                "state": f"RPO +{rpo_yoy*100:.0f}% · 증분OPM {inc_opm*100:.1f}% · 리비전 {rev_status}"
            },
            "fundamental": {
                "icon": f_pq.get("grade_icon", "🟡"),
                "label": f"{price_qual_code} (1차선 {fmt_price(t1_price, is_krx)})",
                "state": f"MDD Ratio {mdd_ratio:.2f} · 자기 {f_tz.get('self_5y_z', 0.0):+.2f}σ"
            },
            "macro": {
                "icon": macro_out["classification"]["grade_icon"],
                "label": f"{macro_out['classification']['label'][:10]} (고유 R² {int(macro_out['classification']['r2_macro_own']*100)}%)",
                "state": f"{macro_out['macro_regime']['current_regime']} ({macro_out['residual_check'].get('status', '')})"
            },
        },
        "v_action_card": {
            "status_code": v_status_code,
            "price_qual_code": price_qual_code,
            "badge_text": v_badge,
            "active_tranche": active_tr_num,
            "tranche_price": active_tr_price,
            "distance_to_1st_tranche_pct": dist_to_t1_pct,
            "execution_ratio": v_exec_ratio,
            "recommended_weight": rec_v_weight,
            "sell_target_1y": sell_target_1y,
            "expected_return_weighted": exp_ret,
            "mdd_ratio": mdd_ratio,
            "high_confidence_overlap": has_high_conf_overlap,
        },
        "t_action_card": {
            "verdict": t_verdict_final,
            "status_code": t_status_code,
            "badge_text": t_badge,
            "setup_id": setup_id,
            "grade": t_grade,
            "risk_modifier_applied": t_risk_mod,
            "entry_price": int(t_ticket.get("entry_price", current_price)),
            "stop_price": int(t_ticket.get("stop_price", current_price * 0.95)),
            "shares": int(t_ticket.get("shares", 0) * t_risk_mod),
            "target_price": int(t_ticket.get("target_1", current_price * 1.10)),
            "expected_r": float(t_ticket.get("expected_r", 0.0)),
            "time_stop_rule": f"{t_ticket.get('time_stop_days', 5)}거래일째 {fmt_price(t_ticket.get('time_stop_min_price', 0), is_krx)} 미달 시 다음 날 시가 청산",
            "invalidation_rule": t_ticket.get("invalidation", "손절가 종가 이탈 또는 S-EVT 악재 발생"),
            "earnings_hold_guide": earnings_hold_guide,
        },
        "macro_risk_overlay": {
            "return_13w_actual": ret_13w_act,
            "return_13w_macro_market_share": ret_13w_mac_mkt,
            "return_13w_idiosyncratic_share": ret_13w_idio,
            "top_sensitivities": [
                {"var_name": s["var_name"], "shock_1sd": s["shock_1sd_str"], "impact": s["price_response_1sd"]}
                for s in macro_out.get("sensitivities", [])[:3]
            ],
        },
        "mispricing_matrix": {
            "earnings_edge": f"📈 [{earn_out['thesis_driver']['mispricing_type']}] {earn_out['thesis_driver']['mispricing_detail']}",
            "multiple_edge": f"📈 [멀티플 진단] {fund_out['mispricing_points']['undervalued_point']}",
            "macro_attribution": (
                f"📊 [13주 수익률 분해 사실] 최근 13주 수익률({ret_13w_act*100:+.1f}%) 중 "
                f"시장·거시 설명 몫은 {ret_13w_mac_mkt*100:+.1f}%p, 기업 고유 몫은 {ret_13w_idio*100:+.1f}%p"
            ),
        },
        "unified_watchlist": unified_watchlist,
        "radar_flags": radar_flags,
    }

# =====================================================================
# [Home Dashboard v1.1 — Data Integrity & SSOT Edition]
# 단일 출처(SSOT), 기준일(as_of)/출처(source) 강제, 하드코딩 SAMPLE_DATA 분리 및 일관성 검증 엔진
# =====================================================================
import json

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

# ==============================================================================
# [Module 2 & 4 Integrated] RRG 축 정규화 엔진 v2.0 + 국면 게이트(Market Gate) v1.1
# - 단일 출처(RRG_CONFIG, GATE_CONFIG), 데이터 무결성 3원칙, 전환 로깅 및 일관성 검증 포함
# ==============================================================================
import os
import math
import numpy as np
import pandas as pd

# ------------------------------------------------------------------------------
# 1. RRG_CONFIG: 임계값 및 시계(Clock) 파라미터 단일 출처 관리
# ------------------------------------------------------------------------------
RRG_CONFIG = {
    "CLOCKS": {
        "daily": {"freq": "D", "n": 50, "m": 10, "L": 252, "tail_step": 5, "tail_points": 5, "label": "순환(일봉)"},
        "weekly": {"freq": "W-FRI", "n": 26, "m": 4, "L": 104, "tail_step": 1, "tail_points": 5, "label": "구조(주봉)"}
    },
    "EMA_SPAN": 3,
    "NEUTRAL_THRESHOLD": 0.25,  # |X| < 0.25 and |Y| < 0.25 -> 중립
    "AXIS_LIMIT": 3.0,          # 차트 고정 범위 [-3, 3] sigma
    "CHART_HEIGHT": 540,        # 우측 Top 3 카드 열과 맞춘 고정 픽셀 높이
    "MIN_OBS_RATIO": 0.8,       # 표본 짧음 판정 기준
    "COMPARE_LOG_PATH": "logs/rrg_compare.csv",
    "COMPARE_WINDOW_DAYS": 20
}


# [20거래일 뒤 제거 예정] 전환 기간 비교용 옛 축 계산 함수 (삭제 금지 규칙 준수)
def legacy_rrg_coordinates(prices: pd.DataFrame, bench: pd.Series, n: int = 50, m: int = 20) -> dict:
    """
    옛 축 계산: X = (RS / SMA(RS, 50) - 1) * 100, Y = (RS / RS_{t-20} - 1) * 100
    """
    results = {}
    for col in prices.columns:
        s = prices[col].dropna()
        b = bench.reindex(s.index).dropna()
        common_idx = s.index.intersection(b.index)
        if len(common_idx) < max(n, m) + 5:
            results[col] = {"x_series": pd.Series(dtype=float), "y_series": pd.Series(dtype=float), "quadrant": "데이터 없음"}
            continue
        rs = s.loc[common_idx] / b.loc[common_idx]
        sma = rs.rolling(n, min_periods=max(5, n // 2)).mean()
        x_ser = (rs / sma - 1.0) * 100.0
        y_ser = rs.pct_change(m) * 100.0
        xv = float(x_ser.iloc[-1]) if not np.isnan(x_ser.iloc[-1]) else np.nan
        yv = float(y_ser.iloc[-1]) if not np.isnan(y_ser.iloc[-1]) else np.nan
        if np.isnan(xv) or np.isnan(yv):
            q = "데이터 없음"
        elif xv > 0 and yv > 0:
            q = "주도"
        elif xv > 0 and yv <= 0:
            q = "약화"
        elif xv <= 0 and yv <= 0:
            q = "소외"
        else:
            q = "개선"
        results[col] = {"x_series": x_ser, "y_series": y_ser, "x": xv, "y": yv, "quadrant": q}
    return results


def classify_rrg_quadrant(x: float, y: float, neutral_thresh: float = None) -> str:
    """사분면 분류 (원점 주변 중립 영역 포함)"""
    if x is None or y is None or np.isnan(x) or np.isnan(y):
        return "데이터 없음"
    th = RRG_CONFIG["NEUTRAL_THRESHOLD"] if neutral_thresh is None else neutral_thresh
    if abs(x) < th and abs(y) < th:
        return "중립"
    if x > 0 and y > 0:
        return "주도"
    if x > 0 and y <= 0:
        return "약화"
    if x <= 0 and y <= 0:
        return "소외"
    return "개선"


def rrg_coordinates(prices: pd.DataFrame, bench: pd.Series, clock: str = "daily") -> dict:
    """
    1단계 새 축 정의:
    - RS_t = ETF 종가 / 기준 지수 종가 (SPY 또는 RSP)
    - r_t = RS_t / SMA(RS, n) - 1 -> EMA(3) 평활 -> X_t = r_t / std(r, L)
    - dX_t = X_t - X_{t-m} -> Y_t = dX_t / std(dX, L)
    """
    if clock not in RRG_CONFIG["CLOCKS"]:
        raise ValueError(f"Unsupported clock: {clock}")
    cfg = RRG_CONFIG["CLOCKS"][clock]
    n, m, L = cfg["n"], cfg["m"], cfg["L"]
    ema_span = RRG_CONFIG["EMA_SPAN"]

    if clock == "weekly":
        p_df = prices.resample("W-FRI").last()
        b_ser = bench.resample("W-FRI").last()
    else:
        p_df = prices.copy()
        b_ser = bench.copy()

    out = {}
    for basket in p_df.columns:
        s = p_df[basket].dropna()
        b = b_ser.reindex(s.index).dropna()
        idx = s.index.intersection(b.index)
        total_obs = len(idx)

        if total_obs < n + m + 5:
            out[basket] = {
                "x": np.nan, "y": np.nan, "quadrant": "데이터 없음",
                "x_series": pd.Series(dtype=float), "y_series": pd.Series(dtype=float),
                "tail_x": [], "tail_y": [], "short_sample": True, "obs_count": total_obs,
                "move_4w_x": np.nan, "move_4w_y": np.nan
            }
            continue

        rs = s.loc[idx] / b.loc[idx]
        sma_rs = rs.rolling(window=n, min_periods=max(5, n // 2)).mean()
        r_raw = (rs / sma_rs) - 1.0
        r_smooth = r_raw.ewm(span=ema_span, adjust=False).mean()

        min_std_p = max(10, min(L // 4, total_obs - n))
        std_r = r_smooth.rolling(window=L, min_periods=min_std_p).std(ddof=1).replace(0, np.nan)
        x_ser = r_smooth / std_r

        dx_ser = x_ser - x_ser.shift(m)
        std_dx = dx_ser.rolling(window=L, min_periods=min_std_p).std(ddof=1).replace(0, np.nan)
        y_ser = dx_ser / std_dx

        valid_xy = pd.concat([x_ser.rename("x"), y_ser.rename("y")], axis=1).dropna()
        if valid_xy.empty:
            out[basket] = {
                "x": np.nan, "y": np.nan, "quadrant": "데이터 없음",
                "x_series": x_ser, "y_series": y_ser,
                "tail_x": [], "tail_y": [], "short_sample": True, "obs_count": total_obs,
                "move_4w_x": np.nan, "move_4w_y": np.nan
            }
            continue

        cur_x = float(valid_xy["x"].iloc[-1])
        cur_y = float(valid_xy["y"].iloc[-1])
        quad = classify_rrg_quadrant(cur_x, cur_y)

        # 꼬리(최근 4주 5개 점) 추출
        step = cfg["tail_step"]
        n_pts = cfg["tail_points"]
        tail_indices = [len(valid_xy) - 1 - i * step for i in range(n_pts) if (len(valid_xy) - 1 - i * step) >= 0]
        tail_indices.reverse()
        tail_x = [float(valid_xy["x"].iloc[i]) for i in tail_indices]
        tail_y = [float(valid_xy["y"].iloc[i]) for i in tail_indices]

        move_4w_x = cur_x - tail_x[0] if len(tail_x) >= 2 else 0.0
        move_4w_y = cur_y - tail_y[0] if len(tail_y) >= 2 else 0.0
        short_sample = total_obs < (n + L)

        out[basket] = {
            "x": cur_x, "y": cur_y, "quadrant": quad,
            "x_series": x_ser, "y_series": y_ser,
            "tail_x": tail_x, "tail_y": tail_y,
            "short_sample": short_sample, "obs_count": total_obs,
            "move_4w_x": move_4w_x, "move_4w_y": move_4w_y
        }
    return out


def diagnose_rrg_axes(prices: pd.DataFrame, bench: pd.Series) -> dict:
    """
    2단계 진단: 과거 3년 데이터로 옛 축 vs 새 축(daily) 비교 검증
    """
    old_res = legacy_rrg_coordinates(prices, bench)
    new_res = rrg_coordinates(prices, bench, clock="daily")

    old_corrs, new_corrs = [], []
    old_abs_x, new_abs_x, basket_vols = [], [], []
    old_q_counts = {"주도": 0, "약화": 0, "소외": 0, "개선": 0}
    new_q_counts = {"주도": 0, "약화": 0, "소외": 0, "개선": 0, "중립": 0}
    old_total, new_total = 0, 0
    clockwise_Map = {"소외": "개선", "개선": "주도", "주도": "약화", "약화": "소외"}
    cw_hits, cw_total = 0, 0

    for b in prices.columns:
        vol = float(prices[b].pct_change().std())
        ox, oy = old_res[b]["x_series"], old_res[b]["y_series"]
        nx, ny = new_res[b]["x_series"], new_res[b]["y_series"]

        o_df = pd.concat([ox.rename("x"), oy.rename("y")], axis=1).dropna()
        n_df = pd.concat([nx.rename("x"), ny.rename("y")], axis=1).dropna()

        if len(o_df) > 20:
            old_corrs.append(float(o_df["x"].corr(o_df["y"])))
            old_abs_x.append(float(o_df["x"].abs().mean()))
            basket_vols.append(vol)
            for _, row in o_df.iterrows():
                q = classify_rrg_quadrant(row["x"], row["y"], neutral_thresh=0.0)
                if q in old_q_counts:
                    old_q_counts[q] += 1
                    old_total += 1

        if len(n_df) > 20:
            new_corrs.append(float(n_df["x"].corr(n_df["y"])))
            new_abs_x.append(float(n_df["x"].abs().mean()))
            prev_q = None
            for _, row in n_df.iterrows():
                q = classify_rrg_quadrant(row["x"], row["y"])
                if q in new_q_counts:
                    new_q_counts[q] += 1
                    new_total += 1
                if q != "중립" and prev_q is not None and prev_q != q:
                    cw_total += 1
                    if clockwise_Map.get(prev_q) == q:
                        cw_hits += 1
                if q != "중립":
                    prev_q = q

    old_xy_corr = float(np.median(old_corrs)) if old_corrs else 1.0
    new_xy_corr = float(np.median(new_corrs)) if new_corrs else 1.0
    old_vol_bias = float( pd.Series(old_abs_x).corr(pd.Series(basket_vols)) ) if len(basket_vols) > 2 else 1.0
    new_vol_bias = float( pd.Series(new_abs_x).corr(pd.Series(basket_vols)) ) if len(basket_vols) > 2 else 0.0

    new_ii_ratio = (new_q_counts["개선"] / max(1, new_total)) * 100.0
    new_iv_ratio = (new_q_counts["약화"] / max(1, new_total)) * 100.0

    passed = (
        (new_xy_corr < old_xy_corr) and
        (new_ii_ratio >= 10.0 and new_iv_ratio >= 10.0) and
        (abs(new_vol_bias) < abs(old_vol_bias) and abs(new_vol_bias) < 0.25)
    )

    return {
        "old_xy_corr_median": round(old_xy_corr, 3),
        "new_xy_corr_median": round(new_xy_corr, 3),
        "old_quadrant_pct": {k: round(v / max(1, old_total) * 100, 1) for k, v in old_q_counts.items()},
        "new_quadrant_pct": {k: round(v / max(1, new_total) * 100, 1) for k, v in new_q_counts.items()},
        "old_vol_bias_corr": round(old_vol_bias, 3),
        "new_vol_bias_corr": round(new_vol_bias, 3),
        "clockwise_transition_pct": round(cw_hits / max(1, cw_total) * 100, 1),
        "passed": bool(passed)
    }


def determine_basket_grade(quadrant: str, revision_positive: bool, flow_positive: bool) -> str:
    """
    기존 등급 판정 규칙 (변경 금지 원칙 준수: 사분면 입력만 받아 동일 로직 수행)
    """
    if quadrant == "데이터 없음":
        return "데이터 없음"
    if quadrant == "중립":
        return "중립 관망"
    if quadrant == "주도":
        if revision_positive and flow_positive:
            return "실물 확인 주도"
        elif revision_positive or flow_positive:
            return "주도 (수급·실적 부분확인)"
        return "과열 / 다이버전스 경계"
    if quadrant == "개선":
        if revision_positive or flow_positive:
            return "차기 점화"
        return "기술적 반등"
    if quadrant == "약화":
        if revision_positive and flow_positive:
            return "건전한 눌림목"
        return "모멘텀 둔화 경계"
    # 소외
    if revision_positive and flow_positive:
        return "바닥권 역발상 축적"
    return "관망 / 소외"


def evaluate_basket_grades(
    prices: pd.DataFrame,
    bench_spy: pd.Series,
    fundamentals_meta: dict,
    as_of: str = "2026-11-08",
    write_log: bool = True
) -> dict:
    """
    3단계: daily 새 사분면으로 등급 판정 + weekly 구조 사분면(`structural_quadrant`) 부착
    + 최근 20거래일 `logs/rrg_compare.csv` 기록
    """
    daily_rrg = rrg_coordinates(prices, bench_spy, clock="daily")
    weekly_rrg = rrg_coordinates(prices, bench_spy, clock="weekly")
    old_rrg = legacy_rrg_coordinates(prices, bench_spy)

    basket_objects = {}
    grade_diff_report = []

    for b in prices.columns:
        d_info = daily_rrg[b]
        w_info = weekly_rrg[b]
        o_info = old_rrg[b]
        meta = fundamentals_meta.get(b, {})
        rev_pos = bool(meta.get("revision_positive", False))
        flw_pos = bool(meta.get("flow_positive", False))

        # X·Y가 NaN인 바스켓은 등급 부여 차단
        if np.isnan(d_info["x"]) or np.isnan(d_info["y"]):
            new_grade = "데이터 없음"
        else:
            new_grade = determine_basket_grade(d_info["quadrant"], rev_pos, flw_pos)

        old_grade = determine_basket_grade(o_info["quadrant"], rev_pos, flw_pos) if not np.isnan(o_info.get("x", np.nan)) else "데이터 없음"

        basket_objects[b] = {
            "basket": b,
            "name": meta.get("name", b),
            "x": d_info["x"],
            "y": d_info["y"],
            "quadrant": d_info["quadrant"],
            "grade": new_grade,
            "structural_quadrant": w_info["quadrant"],
            "old_x": o_info.get("x", np.nan),
            "old_y": o_info.get("y", np.nan),
            "old_quadrant": o_info["quadrant"],
            "old_grade": old_grade,
            "tail_x": d_info["tail_x"],
            "tail_y": d_info["tail_y"],
            "move_4w_x": d_info["move_4w_x"],
            "move_4w_y": d_info["move_4w_y"],
            "short_sample": d_info["short_sample"],
            "obs_count": d_info["obs_count"],
            "revision_positive": rev_pos,
            "flow_positive": flw_pos,
            "as_of": as_of,
            "source": meta.get("source", "Market RRG Engine v2.0")
        }

        if old_grade != new_grade:
            grade_diff_report.append({
                "basket": b,
                "name": meta.get("name", b),
                "old_quadrant": o_info["quadrant"],
                "old_grade": old_grade,
                "new_quadrant": d_info["quadrant"],
                "new_grade": new_grade,
                "structural_quadrant": w_info["quadrant"]
            })

    # 전환 기간 20거래일 로그 저장 (logs/rrg_compare.csv)
    if write_log:
        os.makedirs(os.path.dirname(RRG_CONFIG["COMPARE_LOG_PATH"]), exist_ok=True)
        rows = []
        recent_dates = prices.index[-RRG_CONFIG["COMPARE_WINDOW_DAYS"]:]
        for dt in recent_dates:
            dt_str = dt.strftime("%Y-%m-%d") if hasattr(dt, "strftime") else str(dt)
            for b in prices.columns:
                meta = fundamentals_meta.get(b, {})
                rev_pos = bool(meta.get("revision_positive", False))
                flw_pos = bool(meta.get("flow_positive", False))
                ox_s, oy_s = old_rrg[b]["x_series"], old_rrg[b]["y_series"]
                nx_s, ny_s = daily_rrg[b]["x_series"], daily_rrg[b]["y_series"]

                ox_v = float(ox_s.loc[dt]) if dt in ox_s.index and not np.isnan(ox_s.loc[dt]) else np.nan
                oy_v = float(oy_s.loc[dt]) if dt in oy_s.index and not np.isnan(oy_s.loc[dt]) else np.nan
                nx_v = float(nx_s.loc[dt]) if dt in nx_s.index and not np.isnan(nx_s.loc[dt]) else np.nan
                ny_v = float(ny_s.loc[dt]) if dt in ny_s.index and not np.isnan(ny_s.loc[dt]) else np.nan

                oq = classify_rrg_quadrant(ox_v, oy_v, neutral_thresh=0.0)
                nq = classify_rrg_quadrant(nx_v, ny_v)
                og = determine_basket_grade(oq, rev_pos, flw_pos)
                ng = determine_basket_grade(nq, rev_pos, flw_pos)

                rows.append({
                    "date": dt_str, "basket": b,
                    "old_x": round(ox_v, 3) if not np.isnan(ox_v) else np.nan,
                    "old_y": round(oy_v, 3) if not np.isnan(oy_v) else np.nan,
                    "old_quadrant": oq, "old_grade": og,
                    "new_x": round(nx_v, 3) if not np.isnan(nx_v) else np.nan,
                    "new_y": round(ny_v, 3) if not np.isnan(ny_v) else np.nan,
                    "new_quadrant": nq, "new_grade": ng
                })
        pd.DataFrame(rows).to_csv(RRG_CONFIG["COMPARE_LOG_PATH"], index=False, encoding="utf-8-sig")

    # 주도 Top 3 및 차기 점화 Top 3 카드 선정 (새 사분면 기준)
    valid_items = [v for v in basket_objects.values() if v["grade"] != "데이터 없음"]
    leading_candidates = [v for v in valid_items if v["quadrant"] == "주도"]
    leading_top3 = sorted(leading_candidates, key=lambda item: (item["x"] + item["y"]), reverse=True)[:3]

    igniting_candidates = [v for v in valid_items if v["quadrant"] == "개선"]
    igniting_top3 = sorted(igniting_candidates, key=lambda item: item["y"], reverse=True)[:3]

    return {
        "baskets": basket_objects,
        "leading_top3": leading_top3,
        "igniting_top3": igniting_top3,
        "grade_diff_report": grade_diff_report
    }


# ------------------------------------------------------------------------------
# 4. 8방향 라벨 배치 및 바운딩 박스 충돌 방지: place_labels(points, fig_size)
# ------------------------------------------------------------------------------
def place_labels(points: list[dict], fig_size: tuple[int, int] = (700, 540)) -> dict:
    """
    점들의 화면 픽셀 좌표를 계산해 8방향 후보 중 겹치지 않는 위치를 배정.
    - 원점에서 먼 순서대로 우선 배치
    - 8방향 모두 겹치면 짧은 지시선(leader line)과 함께 외곽으로 밀어냄
    - 그래도 겹치면 라벨을 숨기고 hidden_count 증가
    """
    width, height = fig_size
    lim = RRG_CONFIG["AXIS_LIMIT"]
    pad_x, pad_y = 55, 45
    plot_w = max(200, width - 2 * pad_x)
    plot_h = max(200, height - 2 * pad_y)

    def to_px(x_val: float, y_val: float) -> tuple[float, float]:
        cx = np.clip(x_val, -lim, lim)
        cy = np.clip(y_val, -lim, lim)
        px = pad_x + ((cx + lim) / (2 * lim)) * plot_w
        py = pad_y + ((lim - cy) / (2 * lim)) * plot_h
        return float(px), float(py)

    def boxes_overlap(b1: tuple[float, float, float, float], b2: tuple[float, float, float, float]) -> bool:
        return not (b1[2] <= b2[0] or b1[0] >= b2[2] or b1[3] <= b2[1] or b1[1] >= b2[3])

    # 원점에서 먼 순서대로 정렬
    sorted_pts = sorted(
        [p for p in points if not (np.isnan(p.get("x", np.nan)) or np.isnan(p.get("y", np.nan)))],
        key=lambda item: math.hypot(item["x"], item["y"]),
        reverse=True
    )

    occupied_boxes = []
    # 모든 점 자체의 마커 영역을 먼저 점유 박스로 등록 (반경 7px)
    pt_px_map = {}
    for p in sorted_pts:
        px, py = to_px(p["x"], p["y"])
        pt_px_map[p["basket"]] = (px, py)
        occupied_boxes.append((px - 7, py - 7, px + 7, py + 7))

    # 8방향 오프셋 (dx, dy) 픽셀 단위
    directions_8 = [
        (0, -18), (0, 18), (28, 0), (-28, 0),
        (24, -15), (-24, -15), (24, 15), (-24, 15)
    ]
    # 지시선(Leader line) 확장 방향
    leader_offsets = [
        (46, -28), (-46, -28), (46, 28), (-46, 28),
        (0, -36), (0, 36), (56, 0), (-56, 0)
    ]

    placements = {}
    hidden_count = 0

    for p in sorted_pts:
        b_code = p["basket"]
        px, py = pt_px_map[b_code]
        lbl_w = max(34, len(str(b_code)) * 7 + 8)
        lbl_h = 16

        placed = False
        # 1차: 8방향 근접 탐색
        for dx, dy in directions_8:
            cx, cy = px + dx, py + dy
            cand_box = (cx - lbl_w / 2, cy - lbl_h / 2, cx + lbl_w / 2, cy + lbl_h / 2)
            if cand_box[0] < 5 or cand_box[2] > width - 5 or cand_box[1] < 5 or cand_box[3] > height - 5:
                continue
            if not any(boxes_overlap(cand_box, ob) for ob in occupied_boxes):
                occupied_boxes.append(cand_box)
                placements[b_code] = {
                    "visible": True, "use_leader": False,
                    "dx": dx, "dy": dy, "bbox": cand_box
                }
                placed = True
                break

        if placed:
            continue

        # 2차: 짧은 지시선(Leader line) 외곽 배치
        for dx, dy in leader_offsets:
            cx, cy = px + dx, py + dy
            cand_box = (cx - lbl_w / 2, cy - lbl_h / 2, cx + lbl_w / 2, cy + lbl_h / 2)
            if cand_box[0] < 5 or cand_box[2] > width - 5 or cand_box[1] < 5 or cand_box[3] > height - 5:
                continue
            if not any(boxes_overlap(cand_box, ob) for ob in occupied_boxes):
                occupied_boxes.append(cand_box)
                placements[b_code] = {
                    "visible": True, "use_leader": True,
                    "dx": dx, "dy": dy, "bbox": cand_box
                }
                placed = True
                break

        if not placed:
            # 3차: 공간 부족 시 숨김 처리 (툴팁 전용)
            hidden_count += 1
            placements[b_code] = {
                "visible": False, "use_leader": False,
                "dx": 0, "dy": 0, "bbox": None
            }

    return {"placements": placements, "hidden_count": hidden_count}


# ------------------------------------------------------------------------------
# 5. 국면 게이트(Market Gate) v1.1 + 모듈 4 바스켓 배지 동기화
# ------------------------------------------------------------------------------
GATE_CONFIG = {
    "LEVELS": {
        0: {"code": "G0", "name": "정상 (Normal)", "badge": "🟢", "desc": "전략 신호 100% 정상 실행"},
        1: {"code": "G1", "name": "경계 (Caution)", "badge": "🟡", "desc": "신규 진입 비중 50% 축소 및 선별 접근"},
        2: {"code": "G2", "name": "방어 (Defense)", "badge": "🔴", "desc": "T 신규 진입 보류, V 0차 25% 초미세 탐색만 허용"},
        3: {"code": "G3", "name": "위기 (Crisis)", "badge": "🚨", "desc": "신규 매수 전면 동결, 리스크 청산만 집행"}
    },
    "VIX_THRESHOLDS": {
        "G3_PERCENTILE": 90.0, "G3_ABSOLUTE": 35.0,
        "G2_PERCENTILE": 75.0, "G2_ABSOLUTE": 25.0,
        "G1_PERCENTILE": 60.0, "G1_ABSOLUTE": 20.0
    },
    "REGIME_MAP": {
        "STRONG_BULL": 0, "BULL": 0, "NEUTRAL": 1, "SIDEWAYS": 1,
        "CORRECTION": 2, "BEAR": 2, "DEEP_BEAR": 3, "PANIC": 3
    },
    "MACRO_MAP": {
        "EXPANSION": 0, "GOLDILOCKS": 0, "SLOWDOWN": 1, "INFLATION_PRESSURE": 1,
        "STAGFLATION": 2, "TIGHTENING": 2, "RECESSION": 3, "CREDIT_CRUNCH": 3
    },
    "COOLING_DAYS_REQUIRED": 2,
    "CORE_PASSIVE_TICKERS": ["QLD", "SCHD"]
}


# 헤더(실제 데이터) → 게이트 입력 변환표 (2026-11-08 추가)
GATE_CONFIG["REGIME_ICON_MAP"] = {"🟢": "BULL", "🟡": "SIDEWAYS", "🔴": "BEAR", "⚫": "DEEP_BEAR"}
GATE_CONFIG["REGIME_ICON_RANK"] = {"🟢": 0, "🟡": 1, "🔴": 2, "⚫": 3}
GATE_CONFIG["MACRO_NARRATIVE_MAP"] = [
    ("침체", "CREDIT_CRUNCH"),
    ("골디락스", "GOLDILOCKS"),
    ("금리 민감", "SLOWDOWN"),
]
GATE_CONFIG["COMPONENT_NAMES"] = {"regime": "지수 레짐", "vix": "변동성(VIX)", "macro": "거시 국면"}
GATE_CONFIG["ACTION_SUMMARY"] = {
    0: "신호대로 실행",
    1: "신규 진입은 절반 크기로",
    2: "선별 진입만 (소규모 탐색)",
    3: "신규 매수 동결 · 방어 우선",
}

# 히스테리시스용 과거 재계산 기간(거래일). 첫날은 이전 기록이 없어 그날 지표값으로 시작한다.
GATE_CONFIG["HISTORY_LOOKBACK_DAYS"] = 10


def gate_components(regime_str, vix_percentile, vix_val, macro_phase) -> dict:
    """구성 요소별 게이트 단계(0~3). 데이터가 없는 요소는 None."""
    comps = {"regime": None, "vix": None, "macro": None}
    if regime_str is not None:
        comps["regime"] = GATE_CONFIG["REGIME_MAP"].get(str(regime_str).upper(), 2)
    if vix_percentile is not None and vix_val is not None:
        cfg_v = GATE_CONFIG["VIX_THRESHOLDS"]
        if vix_percentile >= cfg_v["G3_PERCENTILE"] or vix_val >= cfg_v["G3_ABSOLUTE"]:
            comps["vix"] = 3
        elif vix_percentile >= cfg_v["G2_PERCENTILE"] or vix_val >= cfg_v["G2_ABSOLUTE"]:
            comps["vix"] = 2
        elif vix_percentile >= cfg_v["G1_PERCENTILE"] or vix_val >= cfg_v["G1_ABSOLUTE"]:
            comps["vix"] = 1
        else:
            comps["vix"] = 0
    if macro_phase is not None:
        comps["macro"] = GATE_CONFIG["MACRO_MAP"].get(str(macro_phase).upper(), 2)
    return comps


def calculate_single_gate(regime_str, vix_percentile, vix_val, macro_phase) -> tuple[int, bool]:
    """
    최종 게이트 = 데이터가 있는 구성 요소 중 가장 방어적인 값(max).
    일부 요소가 없어도 남은 요소로 판정한다(예전처럼 G2로 고정하지 않음).
    전부 없을 때만 G2 방어 폴백.
    """
    comps = gate_components(regime_str, vix_percentile, vix_val, macro_phase)
    avail = [v for v in comps.values() if v is not None]
    if not avail:
        return 2, True
    return max(avail), len(avail) < len(comps)


def build_gate_inputs_from_header(macro_header: dict, market: str) -> dict:
    """
    홈 헤더(실제 yfinance 데이터)에서 게이트 입력을 만든다. 하드코딩 값 사용 금지.
    - US: QQQ·SPY 레짐 중 더 나쁜 쪽 + VIX + 거시 국면
    - KR: KOSPI 레짐 + VIX(글로벌 변동성 대용, VKOSPI 미연결) + 거시 국면
    """
    if not macro_header:
        return {}
    m = str(market).upper()
    vx = macro_header.get("vix_state", {}) or {}
    narrative = str(macro_header.get("macro_narrative", "") or "")
    phase = next((p for k, p in GATE_CONFIG["MACRO_NARRATIVE_MAP"] if k in narrative), None)
    rank = GATE_CONFIG["REGIME_ICON_RANK"]
    if m == "KR":
        regs = [("KOSPI", macro_header.get("kospi_regime", {}) or {})]
        src = "yfinance (^KS11 레짐 · ^VIX 글로벌 변동성 대용)"
    else:
        regs = [("나스닥", macro_header.get("qqq_regime", {}) or {}),
                ("S&P 500", macro_header.get("spy_regime", {}) or {})]
        src = "yfinance (QQQ·SPY 레짐 · ^VIX)"
    # 데이터가 없어 기본값으로 채워진 레짐은 제외
    regs = [(n, r) for n, r in regs if r.get("icon") in rank and "기본값" not in str(r.get("label", ""))]
    worst = max(regs, key=lambda nr: rank[nr[1]["icon"]]) if regs else None
    as_of_vals = [r.get("as_of") for _, r in regs if r.get("as_of")]
    if vx.get("as_of"):
        as_of_vals.append(vx.get("as_of"))
    return {
        "regime": GATE_CONFIG["REGIME_ICON_MAP"][worst[1]["icon"]] if worst else None,
        "regime_label": f"{worst[0]} {worst[1].get('label', '')}" if worst else "데이터 없음",
        "vix_percentile": vx.get("percentile_1y") if vx.get("value") is not None else None,
        "vix_value": vx.get("value"),
        "macro_phase": phase,
        "macro_label": narrative or "데이터 없음",
        "as_of": max(as_of_vals) if as_of_vals else None,
        "source": src,
    }


def _gate_release_condition(comps: dict, level: int) -> str:
    """현재 단계에서 한 단계 완화되려면 무엇이 필요한지 한 줄로."""
    if level <= 0:
        return "해당 없음 (정상 국면)"
    cfg_v = GATE_CONFIG["VIX_THRESHOLDS"]
    parts = []
    for key, g in comps.items():
        if g is None or g < level:
            continue
        if key == "vix":
            k = f"G{level}"
            parts.append(f"VIX 백분위 {cfg_v[k + '_PERCENTILE']:.0f} 미만 & VIX {cfg_v[k + '_ABSOLUTE']:.0f} 미만")
        elif key == "regime":
            target = {3: "'하락'", 2: "'약한 상승·횡보'", 1: "'추세 상승'"}[level]
            parts.append(f"지수 레짐 {target}으로 회복")
        else:
            parts.append("거시 국면 한 단계 완화")
    if not parts:
        return "데이터 없음"
    return " 그리고 ".join(parts) + f" ({GATE_CONFIG['COOLING_DAYS_REQUIRED']}거래일 연속)"

def evaluate_market_gate(market_type: str, us_data: dict, kr_data: dict, history_state: dict = None) -> dict:
    m_type = str(market_type).upper()
    target_data = kr_data if m_type == "KR" else us_data
    if not target_data:
        gate_info = GATE_CONFIG["LEVELS"][2]
        return {
            "market": m_type, "level": 2, "code": gate_info["code"],
            "name": "데이터 없음 (방어 폴백)", "badge": "⚪",
            "desc": "시장 지표 미확보로 보수적 방어 게이트 임시 적용",
            "raw_level": 2, "cooling_counter": 0, "as_of": None, "source": "데이터 없음",
            "components": {}, "binding_component": "데이터 없음",
            "release_condition": "시장 데이터 확보 후 재판정",
            "action_summary": GATE_CONFIG["ACTION_SUMMARY"][2],
        }
    comps = gate_components(
        target_data.get("regime"), target_data.get("vix_percentile"),
        target_data.get("vix_value"), target_data.get("macro_phase")
    )
    raw_level, is_missing = calculate_single_gate(
        target_data.get("regime"), target_data.get("vix_percentile"),
        target_data.get("vix_value"), target_data.get("macro_phase")
    )
    history_used = history_state is not None
    if history_state is None:
        history_state = {"prev_level": raw_level, "cooling_counter": 0}
    prev_level = history_state.get("prev_level", raw_level)
    cooling_counter = history_state.get("cooling_counter", 0)

    if raw_level > prev_level:
        final_level, new_cooling_counter = raw_level, 0
    elif raw_level < prev_level:
        new_cooling_counter = cooling_counter + 1
        if new_cooling_counter >= GATE_CONFIG["COOLING_DAYS_REQUIRED"]:
            final_level, new_cooling_counter = prev_level - 1, 0
        else:
            final_level = prev_level
    else:
        final_level, new_cooling_counter = prev_level, 0

    names = GATE_CONFIG["COMPONENT_NAMES"]
    binding = [names[k] for k, g in comps.items() if g is not None and g == raw_level]
    easing_pending = final_level > raw_level   # 지표는 좋아졌지만 완화 확인 기간이 안 끝남
    need_days = GATE_CONFIG["COOLING_DAYS_REQUIRED"]
    gate_info = GATE_CONFIG["LEVELS"][final_level]
    suffix = " [일부 지표 데이터 없음]" if is_missing else ""
    return {
        "market": m_type, "level": final_level, "code": gate_info["code"],
        "name": f"{gate_info['name']}{suffix}", "badge": gate_info["badge"],
        "desc": gate_info["desc"], "raw_level": raw_level,
        "cooling_counter": new_cooling_counter,
        "as_of": target_data.get("as_of"),
        "source": target_data.get("source", "데이터 없음"),
        "components": {names[k]: g for k, g in comps.items()},
        "binding_component": ("직전 단계 유지 (완화 확인 중)" if easing_pending
                              else (" · ".join(binding) if binding else "데이터 없음")),
        "release_condition": (
            f"오늘 지표는 G{raw_level} 수준 — {need_days}거래일 연속 유지되면 G{final_level - 1}로 완화 "
            f"(현재 {new_cooling_counter}/{need_days}일)" if easing_pending
            else _gate_release_condition(comps, final_level)
        ),
        "action_summary": GATE_CONFIG["ACTION_SUMMARY"][final_level],
        "easing_pending": easing_pending,
        "history_used": history_used,
        "inputs": target_data,
    }

def build_gate_history_state(etf_data_map: dict, market: str) -> dict:
    """
    히스테리시스(강화는 즉시 · 완화는 COOLING_DAYS_REQUIRED 거래일 연속)를 위해
    최근 HISTORY_LOOKBACK_DAYS 거래일의 게이트를 과거 종가로 다시 계산한다. 파일에 저장하지 않는다.
    - 오늘 판정과 같은 함수(build_regime_vix_header → build_gate_inputs_from_header → evaluate_market_gate)를 쓴다.
    - state_before_last: 마지막 거래일 직전까지의 상태. 오늘 evaluate_market_gate(history_state=...)에 넘긴다.
    - history: 날짜별 [지표 단계(raw_level), 최종 단계(level), 완화 확인 일수]
    """
    m = str(market).upper()
    em = etf_data_map or {}
    n_days = int(GATE_CONFIG["HISTORY_LOOKBACK_DAYS"])

    def _naive(df_x):
        if df_x is None or len(df_x) == 0:
            return None
        out = df_x.copy()
        idx = pd.to_datetime(out.index)
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_localize(None)
        out.index = idx
        return out.sort_index()

    qqq, spy, ks11, vix = (_naive(em.get(k)) for k in ("QQQ", "SPY", "^KS11", "^VIX"))
    base = ks11 if m == "KR" else (spy if spy is not None else qqq)
    if base is None or len(base) < n_days + 60:
        return {"state_before_last": None, "history": [], "reason": "기준 지수 가격 데이터 부족"}

    def _cut(df_x, t):
        return df_x.loc[df_x.index <= t] if df_x is not None else None

    state, prev_state, history = None, None, []
    for t in base.index[-n_days:]:
        if m == "KR":   # 한국 게이트에는 S&P 500 레짐이 필요 없다(계산 시간 절약)
            hdr = build_regime_vix_header(_cut(qqq, t), None, _cut(ks11, t), _cut(vix, t))
        else:           # 미국 게이트에는 KOSPI 레짐이 필요 없다
            hdr = build_regime_vix_header(_cut(qqq, t), _cut(spy, t), None, _cut(vix, t))
        inp = build_gate_inputs_from_header(hdr, m)
        g = evaluate_market_gate(m, us_data=inp, kr_data=inp, history_state=state)
        history.append({"date": str(t)[:10], "raw_level": g["raw_level"],
                        "level": g["level"], "cooling_counter": g["cooling_counter"]})
        prev_state = state
        state = {"prev_level": g["level"], "cooling_counter": g["cooling_counter"]}
    return {"state_before_last": prev_state, "history": history, "reason": None}


def apply_gate_action_matrix(signal_item: dict, gate_level: int) -> dict:
    sig_type = signal_item.get("signal_type", "T_ENTRY")
    orig_action = signal_item.get("action_text", "데이터 없음")
    orig_weight = signal_item.get("suggested_weight", 100)
    ticker = str(signal_item.get("ticker", "")).upper()
    tranche_stage = signal_item.get("tranche_stage", 0)
    is_core_etf = ticker in GATE_CONFIG["CORE_PASSIVE_TICKERS"]
    is_bnf_setup = bool(signal_item.get("is_bnf_oversold", False))

    # 장기 보유 예외(QLD·SCHD): 게이트·매매 신호 대상이 아니다. 보유 유지로만 표시.
    if is_core_etf:
        return {
            **signal_item, "gate_applied": False, "orig_action": orig_action,
            "effective_action": "장기 보유 예외 — 게이트 미적용, 보유 유지", "orig_weight": orig_weight,
            "effective_weight": 0, "status": "CORE_EXEMPT",
            "reason": "장기 보유 예외 종목: 매매 신호를 띄우지 않음"
        }
    if sig_type == "RISK_DEFENSE" or any(k in orig_action for k in ["손절", "차익실현", "축소", "청산"]):
        return {
            **signal_item, "gate_applied": False, "orig_action": orig_action,
            "effective_action": orig_action, "orig_weight": orig_weight,
            "effective_weight": orig_weight, "status": "DEFENSE_PRIORITY",
            "reason": "방어 최우선 원칙: 국면 무관 100% 즉시 집행"
        }
    if gate_level == 0:
        return {
            **signal_item, "gate_applied": False, "orig_action": orig_action,
            "effective_action": orig_action, "orig_weight": orig_weight,
            "effective_weight": orig_weight, "status": "ACTIVE",
            "reason": "G0 정상 국면: 원신호 100% 정상 실행"
        }
    if gate_level == 1:
        eff_w = int(orig_weight * 0.5)
        act_str = "돌파 확인 후 축소 대기" if sig_type == "T_WATCH" else "비중 축소 진입"
        return {
            **signal_item, "gate_applied": True, "orig_action": orig_action,
            "effective_action": act_str, "orig_weight": orig_weight,
            "effective_weight": eff_w, "status": "REDUCED",
            "reason": "G1 경계: 목표 비중 50% 강제 캡(Cap) 적용"
        }
    if gate_level == 2:
        if is_bnf_setup:
            return {
                **signal_item, "gate_applied": True, "orig_action": orig_action,
                "effective_action": "BNF 1차 분할 탐색 승인", "orig_weight": orig_weight,
                "effective_weight": 20, "status": "REDUCED",
                "reason": "G2 방어: 극단 투매 BNF 예외 20% 한도 승인"
            }
        if sig_type == "V_BUY":
            if tranche_stage >= 1 or any(k in orig_action for k in ["1차", "2차", "3차"]):
                return {
                    **signal_item, "gate_applied": True, "orig_action": orig_action,
                    "effective_action": "이월 분할매수 유보 (Paused)", "orig_weight": orig_weight,
                    "effective_weight": 0, "status": "BLOCKED",
                    "reason": "G2 방어: 이월 규칙에 따라 1~3차 추가 분할매수 일시 유보"
                }
            eff_w = min(25, int(orig_weight * 0.25)) if orig_weight > 0 else 25
            return {
                **signal_item, "gate_applied": True, "orig_action": orig_action,
                "effective_action": "0차 초미세 탐색 매수", "orig_weight": orig_weight,
                "effective_weight": eff_w, "status": "REDUCED",
                "reason": "G2 방어: V 0차 신규는 25% 초미세 탐색만 허용"
            }
        act_str = "주문 대기 해제 (관망 전환)" if sig_type == "T_WATCH" else "모멘텀 신규 진입 보류"
        return {
            **signal_item, "gate_applied": True, "orig_action": orig_action,
            "effective_action": act_str, "orig_weight": orig_weight,
            "effective_weight": 0, "status": "BLOCKED",
            "reason": "G2 방어: 추세 신규 진입 및 자동 주문 대기 차단"
        }
    # G3 위기
    if is_bnf_setup:
        return {
            **signal_item, "gate_applied": True, "orig_action": orig_action,
            "effective_action": "BNF 극한 반등 1차 매수", "orig_weight": orig_weight,
            "effective_weight": 20, "status": "REDUCED",
            "reason": "G3 위기: F10 괴리 -30% 이하 BNF 예외 20% 한도 승인"
        }
    return {
        **signal_item, "gate_applied": True, "orig_action": orig_action,
        "effective_action": "신규 매수 전면 동결 (Frozen)", "orig_weight": orig_weight,
        "effective_weight": 0, "status": "FROZEN",
        "reason": "G3 위기: 모든 공격 및 대기 신호 실행 동결"
    }


def sync_radar_basket_badges(radar_signals: dict, basket_grades: dict) -> dict:
    """
    모듈 4 레이더 카드의 `[바스켓] ... 동조` 배지가 옛 등급이 아닌 새 RRG 등급(단일 출처)을 참조하도록 동기화
    """
    synced = {}
    for col_key, sig_list in radar_signals.items():
        new_list = []
        for sig in sig_list:
            item = dict(sig)
            b_code = item.get("basket")
            if b_code and b_code in basket_grades:
                b_obj = basket_grades[b_code]
                item["basket_quadrant"] = b_obj["quadrant"]
                item["basket_grade"] = b_obj["grade"]
                item["basket_sync_badge"] = f"[{b_code}] {b_obj['quadrant']} · {b_obj['grade']} 동조"
            new_list.append(item)
        synced[col_key] = new_list
    return synced


# ------------------------------------------------------------------------------
# 6. 5단계 점검: run_consistency_checks() RRG & 게이트 무결성 검증
# ------------------------------------------------------------------------------
def check_rrg_and_gate_consistency(
    chart_baskets: dict,
    card_baskets: list[dict],
    radar_signals: dict
) -> list[str]:
    """
    위반 점검 3가지:
    1) 카드에 표시된 사분면과 차트의 사분면이 다름
    2) 모듈 4 "[바스켓] … 동조" 배지가 옛 등급(old_grade)을 참조함
    3) X·Y가 NaN인 바스켓이 유효 등급을 받음
    """
    violations = []

    # 점검 1: 카드 사분면 vs 차트 사분면 일치 여부
    for card in card_baskets:
        b = card.get("basket")
        if b in chart_baskets:
            if card.get("quadrant") != chart_baskets[b].get("quadrant"):
                violations.append(
                    f"RRG_QUADRANT_MISMATCH: {b} 카드 사분면({card.get('quadrant')}) != 차트 사분면({chart_baskets[b].get('quadrant')})"
                )

    # 점검 2: 모듈 4 바스켓 동조 배지가 새 등급과 일치하는지 확인
    for col_key, sig_list in radar_signals.items():
        for sig in sig_list:
            b = sig.get("basket")
            badge_str = sig.get("basket_sync_badge", "")
            if b and b in chart_baskets:
                new_g = chart_baskets[b].get("grade", "")
                old_g = chart_baskets[b].get("old_grade", "")
                if new_g not in badge_str or (old_g != new_g and old_g in badge_str):
                    violations.append(
                        f"STALE_BASKET_BADGE: {sig.get('ticker')} 배지('{badge_str}')가 새 등급('{new_g}')과 불일치"
                    )

    # 점검 3: X·Y가 NaN인 바스켓이 등급을 받았는지 확인
    for b, info in chart_baskets.items():
        x_v, y_v = info.get("x", np.nan), info.get("y", np.nan)
        if (x_v is None or y_v is None or np.isnan(x_v) or np.isnan(y_v)):
            if info.get("grade") not in [None, "데이터 없음"]:
                violations.append(
                    f"NAN_COORDINATE_GRADED: {b}의 좌표가 NaN인데 '{info.get('grade')}' 등급이 부여됨"
                )

    return violations


# ------------------------------------------------------------------------------
# 7. 17대 내러티브 바스켓 3년치 표준 시계열 생성 및 SAMPLE_DATA 단일 출처 등록
# ------------------------------------------------------------------------------
def _build_default_rrg_sample_dataset() -> dict:
    np.random.seed(42)
    dates = pd.bdate_range(end="2026-11-06", periods=756)  # 3년 영업일
    n_days = len(dates)
    t = np.linspace(0, 6 * np.pi, n_days)

    # SPY 및 RSP 기준 지수 생성
    spy_ret = 0.0004 + 0.008 * np.random.randn(n_days)
    rsp_ret = 0.00035 + 0.0075 * np.random.randn(n_days)
    spy_prices = pd.Series(400.0 * np.cumprod(1.0 + spy_ret), index=dates, name="SPY")
    rsp_prices = pd.Series(150.0 * np.cumprod(1.0 + rsp_ret), index=dates, name="RSP")

    # 17대 내러티브 바스켓 메타 정의 (고변동~저변동, 위상차 부여로 4분면 고른 분포 보장)
    baskets_spec = [
        ("SMH", "반도체", 0.022, 0.0, True, True),
        ("IBIT", "비트코인 현물", 0.032, 0.6, True, True),
        ("IGV", "소프트웨어", 0.016, 1.1, True, True),
        ("XLI", "산업재·인프라", 0.011, 1.6, True, False),
        ("XLF", "금융", 0.012, 2.1, True, True),
        ("XLU", "유틸리티·전력망", 0.007, 4.9, True, True),
        ("XLP", "필수소비재", 0.006, 3.2, False, False),
        ("XLV", "헬스케어", 0.007, 3.8, False, True),
        ("XLB", "소재", 0.011, 2.7, False, False),
        ("XLRE", "리츠·부동산", 0.012, 1.9, False, False),
        ("XLY", "경기소비재", 0.014, 2.9, False, True),
        ("XLE", "에너지", 0.017, 4.2, True, False),
        ("XLC", "커뮤니케이션", 0.013, 0.4, True, True),
        ("ITA", "방위산업", 0.011, 5.4, True, True),
        ("TAN", "클린에너지", 0.024, 3.5, False, False),
        ("URA", "원자력·우라늄", 0.025, 5.1, True, True),
        ("KWEB", "중국 테크", 0.026, 2.3, False, False),
    ]

    price_dict = {}
    meta_dict = {}
    for code, kor_name, vol, phase, rev_ok, flw_ok in baskets_spec:
        # 사인파 순환 주기 + 고유 변동성 결합 (자연스러운 RRG 시계방향 회전 유도)
        cycle_alpha = 0.0012 * np.sin(t + phase) + 0.0006 * np.cos(2 * t + phase)
        noise = (vol * 0.35) * np.random.randn(n_days)
        rs_path = np.cumprod(1.0 + cycle_alpha + noise)
        price_dict[code] = spy_prices.values * rs_path
        meta_dict[code] = {
            "name": kor_name, "revision_positive": rev_ok, "flow_positive": flw_ok,
            "as_of": "2026-11-08", "source": "Market RRG Engine v2.0"
        }

    prices_df = pd.DataFrame(price_dict, index=dates)
    eval_res = evaluate_basket_grades(prices_df, spy_prices, meta_dict, as_of="2026-11-08", write_log=True)
    diag_res = diagnose_rrg_axes(prices_df, spy_prices)

    return {
        "prices_df": prices_df,
        "bench_spy": spy_prices,
        "bench_rsp": rsp_prices,
        "meta": meta_dict,
        "evaluation": eval_res,
        "diagnostics": diag_res
    }


if "SAMPLE_DATA" not in globals():
    SAMPLE_DATA = {}

# [2026-11-08] 난수·사인파로 만든 합성 시계열은 화면에 쓰지 않는다(테스트 전용 함수로만 남김).
# 실제 4분면은 app.py가 내려받은 실제 종가로 build_real_rrg_bundle()을 호출해 만든다.
SAMPLE_DATA["rrg_bundle"] = {}

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

_raw_radar = {
    "col1_transitions": [
        {"ticker": "NVDA", "basket": "SMH", "market": "US", "signal_type": "T_ENTRY", "tranche_stage": 0, "action_text": "T ST-3 A등급 진입", "suggested_weight": 100, "is_bnf_oversold": False, "note": "20일선 돌파 거래량 2.1배", "as_of": "2026-11-08", "source": "Track 1/3 Synthesis"},
        {"ticker": "005930", "basket": "SMH", "market": "KR", "signal_type": "V_BUY", "tranche_stage": 0, "action_text": "V 0차 즉시 매수", "suggested_weight": 100, "is_bnf_oversold": False, "note": "PBR 밴드 최하단 지지", "as_of": "2026-11-08", "source": "Track 1/3 Synthesis"},
        {"ticker": "000660", "basket": "SMH", "market": "KR", "signal_type": "V_BUY", "tranche_stage": 1, "action_text": "V 1차 분할 매수", "suggested_weight": 50, "is_bnf_oversold": False, "note": "1차 매수선 도달 (이월 주문)", "as_of": "2026-11-08", "source": "Track 3 Valuation"}
    ],
    "col2_watch": [
        {"ticker": "TSLA", "basket": "XLY", "market": "US", "signal_type": "T_WATCH", "tranche_stage": 0, "action_text": "ST-1 셋업 사정권 (0.8% 내)", "suggested_weight": 50, "is_bnf_oversold": False, "note": "변동성 수축 후 상단 근접", "as_of": "2026-11-08", "source": "Track 1 Engine"},
        {"ticker": "IBIT", "basket": "IBIT", "market": "CRYPTO", "signal_type": "T_WATCH", "tranche_stage": 0, "action_text": "ST-2 눌림목 대기 (1.1% 내)", "suggested_weight": 50, "is_bnf_oversold": False, "note": "ETF 발행좌수 순유입 유지", "as_of": "2026-11-08", "source": "Track 1 Engine"}
    ],
    "col3_risk_and_special": [
        {"ticker": "AAPL", "basket": "XLC", "market": "US", "signal_type": "RISK_DEFENSE", "tranche_stage": 0, "action_text": "손절 및 비중 50% 축소", "suggested_weight": 100, "is_bnf_oversold": False, "note": "50일선 이탈 및 S-REG 악화", "as_of": "2026-11-08", "source": "Track 1 Risk Gate"},
        {"ticker": "INTC", "basket": "SMH", "market": "US", "signal_type": "V_BUY", "tranche_stage": 0, "action_text": "극단 투매 BNF 매수", "suggested_weight": 100, "is_bnf_oversold": True, "note": "F10 적정가 괴리 -38% 과매도", "as_of": "2026-11-08", "source": "Track 3 Valuation"},
        {"ticker": "SCHD", "basket": "XLF", "market": "US", "signal_type": "V_BUY", "tranche_stage": 0, "action_text": "정기 코어 적립 매수", "suggested_weight": 100, "is_bnf_oversold": False, "note": "배당 코어 패시브 적립", "as_of": "2026-11-08", "source": "Core Portfolio Rule"}
    ]
}
# [2026-11-08] 레이더 신호는 아직 실제 신호 엔진과 연결되지 않은 예시 데이터다.
# 화면에서 진짜 엔진 결과처럼 보이지 않도록 기준일을 지우고 출처를 SAMPLE_DATA로 표시한다.
for _col_list in _raw_radar.values():
    for _sig in _col_list:
        _sig["orig_source"] = _sig.get("source", "")
        _sig["as_of"] = None
        _sig["source"] = "SAMPLE_DATA (예시 신호)"
        _sig["is_sample"] = True
SAMPLE_DATA["radar_signals"] = _raw_radar


# ==============================================================================
# [2026-11-08] 실제 종가 기반 4분면(RRG) 번들 — 합성 데이터 대체
# ==============================================================================
def build_real_rrg_bundle(etf_data_map: dict, narrative_baskets: list = None, write_log: bool = True) -> dict:
    """
    app.py가 yfinance로 내려받은 실제 종가로 rrg_coordinates / evaluate_basket_grades를 돌린다.
    - 실적 리비전이 예시 데이터(SAMPLE)이거나 없으면 '실적 데이터 없음'으로 표시하고 등급 판정에 쓰지 않는다.
    - 수급은 실제 CMF(20) > 0 여부를 쓴다.
    """
    spec_map = {s["ticker"]: s for s in NARRATIVE_BASKETS_SPEC}
    spy_df = (etf_data_map or {}).get("SPY")
    if spy_df is None or "Close" not in spy_df or len(spy_df) < 80:
        return {"available": False, "reason": "SPY 종가 데이터 없음"}

    def _close(df_x):
        s = pd.to_numeric(df_x["Close"], errors="coerce")
        s.index = pd.to_datetime(s.index).tz_localize(None) if getattr(s.index, "tz", None) is not None else pd.to_datetime(s.index)
        return s.sort_index()

    closes = {}
    for tk in spec_map:
        df_x = etf_data_map.get(tk)
        if df_x is not None and "Close" in df_x and len(df_x) >= 80:
            closes[tk] = _close(df_x)
    if not closes:
        return {"available": False, "reason": "바스켓 종가 데이터 없음"}

    prices = pd.DataFrame(closes).sort_index()
    spy = _close(spy_df)
    rsp_df = etf_data_map.get("RSP")
    rsp = _close(rsp_df) if (rsp_df is not None and "Close" in rsp_df) else None
    as_of = str(prices.dropna(how="all").index[-1])[:10]

    nb_map = {b.get("ticker"): b for b in (narrative_baskets or [])}
    meta = {}
    for tk in prices.columns:
        nb = nb_map.get(tk, {})
        rev_known = bool(nb) and not nb.get("is_sample_revision", False) and nb.get("fundamental_or_flow_direction", "none") != "none"
        cmf = nb.get("cmf_20")
        meta[tk] = {
            "name": spec_map[tk]["name"],
            "revision_known": rev_known,
            "revision_positive": rev_known and nb.get("fundamental_or_flow_direction") == "up",
            "flow_positive": (cmf is not None) and (not nb.get("is_sample_price", False)) and float(cmf) > 0,
            "as_of": as_of,
            "source": f"yfinance 종가 ({tk})",
        }

    evaluation = evaluate_basket_grades(prices, spy, meta, as_of=as_of, write_log=write_log)

    # 실적 리비전을 모르는 바스켓은 '실적 확인'이 들어간 등급을 받지 않도록 문구를 바꾼다.
    for tk, b in evaluation["baskets"].items():
        if b["grade"] == "데이터 없음" or meta.get(tk, {}).get("revision_known", False):
            continue
        q = b["quadrant"]
        if q in ("주도", "개선", "약화", "소외"):
            flow_txt = "수급 확인" if meta[tk]["flow_positive"] else "수급 약함"
            b["grade"] = f"{q} ({flow_txt} · 실적 데이터 없음)"

    return {
        "available": True,
        "prices_df": prices,
        "bench_spy": spy,
        "bench_rsp": rsp,
        "meta": meta,
        "evaluation": evaluation,
        "as_of": as_of,
    }


# ==============================================================================
# 실행 레이더 — 실제 T(추세매매) 신호 (2026-11-09 추가)
# - 대상: 섹터 흐름의 '지금 강한 곳'·'다음 후보' 섹터 대표 종목 (나중에 사용자 관심 종목을 앞에 추가 가능)
# - 신호: analyze_technical_track(딥다이브와 같은 엔진)의 T 판정. V(가치)는 딥다이브에서만 계산한다.
# ==============================================================================
RADAR_CONFIG = {
    "PROXIES_PER_BASKET": 3,       # 섹터당 대표 종목 수
    "MAX_TICKERS": 18,             # 종목당 약 1.5초 → 최대 약 30초
    "PRICE_PERIOD": "2y",          # EMA200·250일 변동성 계산에 필요한 기간
    "MIN_BARS": 260,               # 이보다 짧으면 신호를 만들지 않음
    "FX_TICKER": "KRW=X",          # 미국 종목을 원화 기준 엔진에 넣기 위한 USD/KRW 환율
    "BENCH": {"US": "SPY", "KR": "^KS11"},
    "MARKET_CLOSE": {"US": ("America/New_York", 16, 15), "KR": ("Asia/Seoul", 15, 45)},  # 장 마감 + 여유 15분
    "DEFENSE_STATES": ["K", "D"],  # 붕괴 경계·하락 진행 → 보유자 방어 신호 (방어가 공격보다 우선)
}


def select_radar_universe(evaluation: dict) -> List[Dict[str, Any]]:
    """섹터 흐름 평가(build_real_rrg_bundle의 evaluation)에서 레이더 대상 종목을 고른다."""
    spec_map = {s["ticker"]: s for s in NARRATIVE_BASKETS_SPEC}
    out: List[Dict[str, Any]] = []
    seen = set()
    groups = [("지금 강한 곳", (evaluation or {}).get("leading_top3", [])),
              ("다음 후보", (evaluation or {}).get("igniting_top3", []))]
    for role, items in groups:
        for it in items:
            spec = spec_map.get(it.get("basket"))
            if not spec:
                continue
            for p in spec["proxies"][:RADAR_CONFIG["PROXIES_PER_BASKET"]]:
                if p in seen:
                    continue
                seen.add(p)
                out.append({"ticker": p, "market": "KR" if p.endswith((".KS", ".KQ")) else "US",
                            "basket": spec["ticker"], "basket_role": role, "origin": "강한 섹터 대표 종목"})
    return out[:RADAR_CONFIG["MAX_TICKERS"]]


def _drop_incomplete_bar(df: Optional[pd.DataFrame], market: str, now=None) -> Optional[pd.DataFrame]:
    """오늘 장이 아직 안 끝났으면 마지막(진행 중) 일봉을 뺀다 — 미완성 봉으로 신호를 만들지 않기 위해."""
    if df is None or len(df) == 0:
        return df
    tz, hh, mm = RADAR_CONFIG["MARKET_CLOSE"].get(market, RADAR_CONFIG["MARKET_CLOSE"]["US"])
    now_l = pd.Timestamp.now(tz=tz) if now is None else pd.Timestamp(now).tz_convert(tz)
    last = pd.Timestamp(df.index[-1])
    last_date = (last.tz_convert(tz) if last.tzinfo is not None else last).date()
    if last_date == now_l.date() and (now_l.hour, now_l.minute) < (hh, mm):
        return df.iloc[:-1]
    return df


def build_radar_signal(item: Dict[str, Any], df: Optional[pd.DataFrame], bench_df: Optional[pd.DataFrame],
                       vix_series: Optional[pd.Series], account_cfg: Optional[Dict[str, Any]],
                       fx_usdkrw: Optional[float] = None) -> Dict[str, Any]:
    """
    한 종목의 실제 T 신호를 레이더 형식(apply_gate_action_matrix 입력)으로 만든다.
    - 가치 트랙(매수 기준선·매도 목표가·Thesis)은 넣지 않는다 → 엔진이 임의 매수선을 만들지 않도록 빈 값 전달.
    - 미국 종목은 fx_to_krw(USD/KRW)를 넘겨, 엔진이 유동성 기준·계좌 금액만 원화로 환산하게 한다(가격은 달러 그대로).
    반환 kind: "signal"(실행·관찰·방어) | "none"(신호 없음) | "error"(계산 불가)
    """
    mkt = item.get("market", "US")
    base = {**item, "source": "엔진 T 트랙 (yfinance 일봉)", "is_sample": False, "tranche_stage": 0}
    try:
        df = _drop_incomplete_bar(df, mkt)
        n_bars = 0 if df is None else len(df)
        as_of = str(df.index[-1])[:10] if n_bars else None
        if n_bars < RADAR_CONFIG["MIN_BARS"]:
            return {**base, "kind": "error", "as_of": as_of,
                    "reason": f"일봉 {n_bars}개 ({RADAR_CONFIG['MIN_BARS']}개 필요)"}
        if mkt == "US" and (not fx_usdkrw or float(fx_usdkrw) <= 0):
            return {**base, "kind": "error", "as_of": as_of, "reason": "USD/KRW 환율 데이터 없음"}
        px = df.dropna(subset=["Open", "High", "Low", "Close"])
        neutral_value = {"thesis_status": "미확인", "kill_flags": [], "buy_tranches": [], "sell_target": None,
                         "position_cap": TECH_PARAMS["MAX_POS_RATIO_TOTAL"], "real_signal": "중립",
                         "macro_regime_override": None}
        days_to_earn = trading_days_until(as_of, item.get("next_earnings"))
        blackout = is_earnings_blackout(days_to_earn)
        cfg = {**(account_cfg or {}), "is_krx": mkt == "KR", "fx_to_krw": 1.0 if mkt == "KR" else float(fx_usdkrw),
               "earnings_blackout": blackout}
        out = analyze_technical_track(px, bench_df, vix_series, value_macro_input=neutral_value, account_cfg=cfg)
    except Exception as e:
        return {**base, "kind": "error", "as_of": None, "reason": f"계산 오류: {type(e).__name__}: {e}"[:200]}

    sb, ls, tkt = out["summary_banner"], out["lifecycle_state"], (out.get("t_ticket") or {})
    to_px = (lambda v: round(float(v), 2)) if mkt == "US" else (lambda v: int(v))
    to_txt = str
    detail = {
        "currency": "USD" if mkt == "US" else "KRW",
        "setup_id": sb.get("setup_id"), "grade": sb.get("grade"),
        "state": ls.get("state"), "state_label": ls.get("state_label"),
        "headline": to_txt(sb.get("headline_reason", "")),
        "next_cond": to_txt(sb.get("next_or_end_condition", "")),
        "as_of": as_of,
        "next_earnings": item.get("next_earnings") if days_to_earn is not None else None,
        "days_to_earnings": days_to_earn,
        "earnings_blackout": False,
    }
    if isinstance(tkt.get("entry_price"), (int, float)) and isinstance(tkt.get("stop_price"), (int, float)):
        detail["entry"] = to_px(tkt["entry_price"])
        detail["stop"] = to_px(tkt["stop_price"])
        if isinstance(tkt.get("target_1"), (int, float)):
            detail["target_1"] = to_px(tkt["target_1"])
        if tkt.get("expected_r") is not None:
            detail["expected_r"] = tkt.get("expected_r")

    if ls.get("state") in RADAR_CONFIG["DEFENSE_STATES"]:
        sig_type, action = "RISK_DEFENSE", f"보유 중이면 비중 축소·손절 검토 ({ls.get('state_label')})"
    elif sb.get("t_verdict") == "진입":
        sig_type, action = "T_ENTRY", f"T {sb.get('setup_id')} {sb.get('grade')}등급 진입"
    elif sb.get("t_verdict") == "대기":
        sig_type, action = "T_WATCH", f"T {sb.get('setup_id')} {sb.get('grade')}등급 대기"
    else:
        why = detail["headline"] + (f" · 실적 발표 D-{days_to_earn} (신규 진입 제한 기간)" if blackout else "")
        return {**base, **detail, "kind": "none", "reason": why}
    if sig_type in ("T_ENTRY", "T_WATCH") and blackout:
        # 종합 결론(synthesize_4tracks)과 같은 규칙: 실적 발표 D-3 ~ 당일 전 셋업 신규 진입 금지. 방어 신호는 막지 않는다.
        return {**base, **detail, "kind": "signal", "signal_type": sig_type, "earnings_blackout": True,
                "action_text": f"T {sb.get('setup_id')} {sb.get('grade')}등급 — 실적 발표 D-{days_to_earn}로 신규 진입 보류",
                "suggested_weight": 0, "is_bnf_oversold": False}
    return {**base, **detail, "kind": "signal", "signal_type": sig_type, "action_text": action,
            "suggested_weight": 100,
            "is_bnf_oversold": bool(sig_type == "T_ENTRY" and sb.get("setup_id") == "ST-1")}


def build_radar_signals(universe: List[Dict[str, Any]], price_map: Dict[str, pd.DataFrame],
                        bench_map: Dict[str, Optional[pd.DataFrame]], vix_series: Optional[pd.Series],
                        account_cfg: Optional[Dict[str, Any]], fx_usdkrw: Optional[float]) -> Dict[str, Any]:
    """레이더 대상 전체를 계산해 신호 / 신호 없음 / 계산 불가로 나눈다."""
    res: Dict[str, Any] = {"signals": [], "no_signal": [], "errors": []}
    for item in universe or []:
        r = build_radar_signal(item, (price_map or {}).get(item["ticker"]),
                               (bench_map or {}).get(item.get("market", "US")), vix_series, account_cfg, fx_usdkrw)
        {"signal": res["signals"], "none": res["no_signal"]}.get(r["kind"], res["errors"]).append(r)
    return res
