"""T 셋업 조건부 백테스트·검증 지표(R 배수 기대값).

engine.py를 기능별로 나눈 모듈. 바깥에서는 `import engine` 후 `engine.이름`으로 쓴다(engine/__init__.py).
"""
from typing import Dict, List, Any
import numpy as np
import pandas as pd

from .common import TECH_PARAMS

# =====================================================================
# 10. 조건부 백테스트 및 검증 지표 (R 배수 기대값 E 산출)[cite: 1]
# =====================================================================
def _bt_summary(rs: List[float]) -> Dict[str, Any]:
    """R 배수 목록 → 표본 수·승률·기대값과 기대값의 표준오차(표본이 작을수록 커짐)."""
    n = len(rs)
    if n == 0:
        return {"sample_n": 0, "win_rate": None, "expectancy_r": None, "se_r": None}
    arr = np.asarray(rs, dtype=float)
    se = float(arr.std(ddof=1) / np.sqrt(n)) if n >= 2 else None
    return {"sample_n": n, "win_rate": round(float((arr > 0).mean() * 100.0), 1),
            "expectancy_r": round(float(arr.mean()), 2), "se_r": None if se is None else round(se, 2)}


def run_setup_backtest_validation(df: pd.DataFrame) -> Dict[str, Any]:
    """
    10장. 과거 일봉 데이터에서 ST-1(이격도 반전) 및 돌파/눌림 신호의 R 배수 성과를 검증[cite: 1]:
    E = p * W_R - (1 - p) * L_R (왕복 비용 TP-32 0.30% 차감 후 R 기준)
    """
    if len(df) < 150:
        return {"sample_n": 0, "win_rate": 0.0, "avg_win_r": 0.0, "avg_loss_r": 0.0, "expectancy_r": 0.0, "max_loss_streak": 0, "trades": []}

    r_list: List[float] = []
    tag_list: List[str] = []
    trades: List[Dict[str, Any]] = []
    all_trades: List[Dict[str, Any]] = []   # 날짜가 붙은 전체 거래 (rule_backtest의 게이트별 집계용)
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
            tag_list.append(setup_tag)
            trades.append({
                "date": str(df.index[i])[:10],
                "setup": setup_tag,
                "entry": round(entry_p, 0),
                "exit": round(exit_p, 0),
                "r_multiple": r_val
            })
            all_trades.append({"date": str(df.index[i])[:10], "setup": setup_tag, "r_multiple": r_val})
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
        "by_setup": {tag: _bt_summary([r for r, t in zip(r_list, tag_list) if t == tag]) for tag in ("ST-1", "ST-2")},
        "r_by_setup": {tag: [r for r, t in zip(r_list, tag_list) if t == tag] for tag in ("ST-1", "ST-2")},
        "all_trades": all_trades,
    }
