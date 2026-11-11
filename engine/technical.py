"""T 트랙 7상태 기계, 셋업 스캐너·리스크·사이징, V 타이밍 판정, 메인 진입점 analyze_technical_track.

engine.py를 기능별로 나눈 모듈. 바깥에서는 `import engine` 후 `engine.이름`으로 쓴다(engine/__init__.py).
"""
import math
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

from .common import REGIME_RISK_MATRIX, TECH_PARAMS, adjust_krx_price, calc_zone_bounds, fmt_price
from .technical_signals import (
    build_auto_levels_and_volume_profile,
    compute_auto_trend_channel,
    compute_base_primitives,
    compute_market_regime,
    compute_measured_targets,
    compute_relative_strength,
    compute_s_mode_series,
    evaluate_price_structure,
    evaluate_signals_and_modules_at,
    get_confirmed_swings_up_to,
)
from .backtest import run_setup_backtest_validation

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
                "tranche": t_num, "price": (int(t_price) if is_krx else round(float(t_price), 2)), "weight": t_weight,
                "price_reached": bool(c_t <= t_price),
                "verdict": "이월", "ratio": 0.0, "codes": codes, "high_confidence": high_conf
            })
            continue

        # 1. 0차: 기술 게이트 미적용 (V자 반등 보험 -> 무조건 즉시)[cite: 1]
        if t_num == 0:
            results.append({
                "tranche": 0, "price": (int(t_price) if is_krx else round(float(t_price), 2)), "weight": t_weight,
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
            "tranche": t_num, "price": (int(t_price) if is_krx else round(float(t_price), 2)), "weight": t_weight,
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
