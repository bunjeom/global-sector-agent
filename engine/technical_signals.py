"""T 트랙 기초값·보조지표, 환경·상대강도·추세 신호, 생애주기 신호와 6대 판정 모듈.

engine.py를 기능별로 나눈 모듈. 바깥에서는 `import engine` 후 `engine.이름`으로 쓴다(engine/__init__.py).
"""
import math
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

from .common import TECH_PARAMS, calc_zone_bounds, fmt_price

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
