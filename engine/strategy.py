"""전략 선택 (딥다이브 개편 4단계, 2026-11-16) — 단기 규칙 판정과 '어느 전략으로 들어갈지' 우선순위.

규칙은 AGENTS.md 4번 '투자 방향과 딥다이브 개편 기획'에 2026-11-15 사전 등록했다. 결과를 보고 숫자를 바꾸지 않는다.

단기 규칙 (관행값 고정)
    진입  틀 A 상태가 상승 또는 횡보 + RSI(2, 와일더) < 10 눌림(T-B와 같은 눌림 정의) → 다음 날 시가.
          단 다음 날 시가에서 손절선까지 거리 ≤ 3% (목표 +3%와 손익비 ≥ 1). 시가가 손절선 이하이면 진입하지 않음.
    지지선  신호일까지 확정된 가장 최근 스윙 저점(틀 A 지그재그 3×ATR)
    손절선  지지선 − 0.5×ATR(20, 신호일 값)
    청산(먼저 오는 것)  ① 장중 +3% 도달 → +3%(시가가 이미 위면 시가) ② 종가 < 손절선 → 다음 날 시가
                       ③ 틀 A 상태가 '하락' → 다음 날 시가 ④ 진입일 포함 10거래일 안에 +3% 미도달 → 다음 날 시가
    왕복 비용 0.30%.

전략 선택 우선순위
    1 공통 차단: 생존성 하드 킬 → 신규 진입 없음. 국면 게이트 크기 0(G2·G3) → 의견만.
    2 가치: 틀 B '진입 가능'·'조건부 진입' 그리고 가치 관점 순위 피어 상위 1/3.
    3 추세: 틀 A 진입 의견 그리고 추세 관점 순위 피어 상위 1/3.
    4 단기: 가치·추세가 아니고 단기 규칙 진입 조건 충족.
    5 여럿 충족: 보유 기간이 긴 쪽(가치 > 추세 > 단기)이 진입 전략, 나머지는 보조 근거.
    (구현 때 정함, 결과 보기 전) 조건 자료가 없으면(피어 순위 없음 등) '확인 불가' = 충족 아님.
"""
import math
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .common import TECH_PARAMS
from .playbook import PLAYBOOK_CONFIG, STATE_DOWN, STATE_SIDE, STATE_UP, _isna, swing_points, trend_indicators
from .rule_backtest import _bt_naive

SHORT_TERM_CONFIG: Dict[str, Any] = {
    "VERSION": "단기 v1 (2026-11-15 등록)",
    "RSI_MAX": 10.0,          # RSI(2) < 10 = 눌림 (T-B와 같음)
    "STOP_ATR": 0.5,          # 손절선 = 지지선 − 0.5×ATR(20)
    "MAX_STOP_DIST": 0.03,    # 다음 날 시가에서 손절선까지 ≤ 3%
    "TARGET": 0.03,           # +3% 익절
    "MAX_DAYS": 10,           # 진입일 포함 10거래일
    "STATES": (STATE_UP, STATE_SIDE),
}
STRATEGY_VERIFICATION: Dict[str, str] = {
    "short": ("검증 미통과 — 참고만. 사전 등록 시험(SHORT_TERM.md, 2026-11-16, 새 종목 82개·2,543건): 비용 0.30% 뒤 거래당 −0.25%로 "
              "같은 보유 기간 무작위 진입(−0.20%)보다 낫지 않음(z −0.5), RSI(2) 눌림의 기여도 없음"),
    "trend": "미검증 · 의견 — 틀 A는 과거 검증에서 우위 미확인(FRAME_A.md), 신호 기록 장부로 채점 중",
    "value": "미검증 — 틀 B v1은 우위 미확인(FRAME_B.md), v2 추정치 기준은 2026-11-14부터 기록",
    "peer_rank": "피어 순위가 수익을 예측한다는 검증은 없음(사실 비교) — 매월 상위·하위 1/3 채점 예정",
}
VALUE_ENTRY_STAGES = ("진입 가능", "조건부 진입")


# ---------------------------------------------------------------------
# 단기 규칙 — 지표 (그날 종가까지의 자료만)
# ---------------------------------------------------------------------
def short_rsi2(close: pd.Series) -> pd.Series:
    """RSI(2, 와일더). engine/frame_backtest.py·t_backtest.py의 T-B 눌림 정의와 같은 식."""
    delta = close.astype(float).diff()
    up, dn = delta.clip(lower=0.0), (-delta).clip(lower=0.0)
    au = up.ewm(alpha=1 / 2, adjust=False, min_periods=2).mean()
    ad = dn.ewm(alpha=1 / 2, adjust=False, min_periods=2).mean()
    return (100.0 - 100.0 / (1.0 + au / ad.replace(0.0, np.nan))).where(ad > 0, 100.0)


def short_indicators(df: pd.DataFrame, ind: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """틀 A 지표(추세 상태·ATR) + RSI(2) + 그날까지 확정된 가장 최근 스윙 저점(지지선)과 손절선."""
    c = SHORT_TERM_CONFIG
    base = ind if ind is not None else trend_indicators(df)
    out = base.copy()
    out["RSI2"] = short_rsi2(out["Close"])
    hi = out["High"].to_numpy(dtype=float)
    lo = out["Low"].to_numpy(dtype=float)
    atr = out["ATR20"].to_numpy(dtype=float)
    n = len(out)
    sup_px = np.full(n, np.nan)
    sup_idx = np.full(n, -1, dtype=int)
    lows = [p for p in swing_points(hi, lo, atr) if p["kind"] == "L"]
    k, cur_px, cur_idx = 0, np.nan, -1
    for t in range(n):
        while k < len(lows) and lows[k]["confirm"] <= t:
            cur_px, cur_idx = lows[k]["price"], lows[k]["idx"]
            k += 1
        sup_px[t], sup_idx[t] = cur_px, cur_idx
    out["support"] = sup_px
    out["support_idx"] = sup_idx
    out["stop"] = out["support"] - c["STOP_ATR"] * out["ATR20"]
    st = out["state"].to_numpy(dtype=object)
    state_ok = np.array([(not _isna(s)) and s in c["STATES"] for s in st], dtype=bool)
    out["short_signal"] = (state_ok & (out["RSI2"] < c["RSI_MAX"]).to_numpy(dtype=bool)
                           & np.isfinite(out["stop"].to_numpy(dtype=float)))
    return out


def _px(v: Optional[float]) -> str:
    if v is None or not math.isfinite(float(v)):
        return "—"
    v = float(v)
    return f"{v:,.0f}" if abs(v) >= 1000 else f"{v:,.2f}"


def short_term_evaluate(df: pd.DataFrame, ind: Optional[pd.DataFrame] = None, at: Any = None) -> Dict[str, Any]:
    """오늘(또는 at일) 종가 기준 단기 규칙 판정. 진입은 다음 날 시가라, 시가가 맞아야 할 범위(손절선 < 시가 ≤ 손절선 ÷ 0.97)를 함께 준다."""
    c = SHORT_TERM_CONFIG
    base = {"strategy": "short", "version": c["VERSION"], "verification": STRATEGY_VERIFICATION["short"]}
    d = _bt_naive(df)
    if d is None or not len(d):
        return {**base, "available": False, "reason": "가격 데이터 없음"}
    si = short_indicators(d, ind)
    if at is not None:
        si = si.loc[:pd.Timestamp(at)]
    if not len(si):
        return {**base, "available": False, "reason": "가격 데이터 없음"}
    row = si.iloc[-1]
    state = None if _isna(row.get("state")) else row.get("state")
    if state is None:
        return {**base, "available": False, "as_of": str(si.index[-1].date()), "reason": "가격 이력 부족(추세 상태를 정할 수 없음)"}
    close = float(row["Close"])
    rsi = None if pd.isna(row["RSI2"]) else float(row["RSI2"])
    atr = None if pd.isna(row["ATR20"]) else float(row["ATR20"])
    stop = None if pd.isna(row["stop"]) else float(row["stop"])
    sup = None if pd.isna(row["support"]) else float(row["support"])
    sup_date = str(si.index[int(row["support_idx"])].date()) if int(row["support_idx"]) >= 0 else None
    state_ok = state in c["STATES"]
    rsi_ok = rsi is not None and rsi < c["RSI_MAX"]
    open_max = None if stop is None else stop / (1.0 - c["MAX_STOP_DIST"])
    dist_close = None if stop is None or close <= 0 else (close - stop) / close
    near_ok = None if dist_close is None else bool(0 < dist_close <= c["MAX_STOP_DIST"])
    conds = [
        {"key": "state", "label": "틀 A 추세 상태가 상승 또는 횡보", "ok": state_ok, "text": f"지금 {state}"},
        {"key": "rsi", "label": f"RSI(2) < {c['RSI_MAX']:.0f} (눌림)", "ok": rsi_ok,
         "text": "데이터 없음" if rsi is None else f"RSI(2) {rsi:.1f}"},
        {"key": "support", "label": "지지선(확정된 최근 스윙 저점) 있음", "ok": stop is not None,
         "text": "확정된 스윙 저점 없음" if stop is None else
         f"지지선 {_px(sup)}({sup_date}) − 0.5×ATR {_px(atr)} = 손절선 {_px(stop)}"},
        {"key": "dist", "label": f"다음 날 시가에서 손절선까지 ≤ {c['MAX_STOP_DIST'] * 100:.0f}%", "ok": None,
         "text": "손절선 없음" if stop is None else
         (f"내일 시가가 {_px(stop)} 초과 {_px(open_max)} 이하일 때만 · 오늘 종가 {_px(close)} 기준 거리 "
          + ("—" if dist_close is None else f"{dist_close * 100:.1f}%") + (" (충족)" if near_ok else " (미충족)"))},
    ]
    signal = bool(state_ok and rsi_ok and stop is not None)
    if signal:
        label = "단기 진입 신호 — 내일 시가 조건 확인"
    elif state == STATE_DOWN:
        label = "해당 없음 (하락 추세)"
    elif not rsi_ok:
        label = "대기 (눌림 아님)"
    else:
        label = "대기 (지지선 없음)"
    plan = None
    if signal:
        plan = {"open_min_excl": stop, "open_max": open_max, "stop": stop,
                "target_text": f"진입가 × {1 + c['TARGET']:.2f} (장중 도달 시 익절, 시가가 이미 위면 시가)",
                "exits": [f"장중 +{c['TARGET'] * 100:.0f}% 도달 → 익절",
                          f"종가 < 손절선 {_px(stop)} → 다음 날 시가",
                          "틀 A 추세 상태가 '하락' → 다음 날 시가",
                          f"진입일 포함 {c['MAX_DAYS']}거래일 안에 +{c['TARGET'] * 100:.0f}% 미도달 → 다음 날 시가"],
                "cost_text": f"왕복 비용 {TECH_PARAMS['ROUND_TRIP_COST'] * 100:.2f}% 가정"}
    return {**base, "available": True, "as_of": str(si.index[-1].date()), "state": state, "close": close, "rsi2": rsi,
            "atr": atr, "support": sup, "support_date": sup_date, "stop": stop, "open_max": open_max,
            "dist_at_close": dist_close, "dist_at_close_ok": near_ok, "signal": signal, "label": label,
            "conditions": conds, "plan": plan}


# ---------------------------------------------------------------------
# 전략 선택 우선순위
# ---------------------------------------------------------------------
def _cond(label: str, ok: Optional[bool], text: str) -> Dict[str, Any]:
    return {"label": label, "ok": ok, "text": text}


def choose_strategy(survival_kill: Optional[bool] = None, gate_level: Optional[int] = None,
                    frame_b_stage: Optional[str] = None, value_top: Optional[bool] = None, value_rank_text: Optional[str] = None,
                    frame_a_entry: Optional[bool] = None, frame_a_label: Optional[str] = None,
                    trend_top: Optional[bool] = None, trend_rank_text: Optional[str] = None,
                    short_signal: Optional[bool] = None, short_label: Optional[str] = None) -> Dict[str, Any]:
    """전략 카드 3장과 진입 전략. ok=None(자료 없음)은 충족 아님으로 본다."""
    def rank_cond(label: str, top: Optional[bool], text: Optional[str]) -> Dict[str, Any]:
        if top is None:
            return _cond(label, None, text or "피어 순위 없음 — 확인 불가")
        return _cond(label, bool(top), text or ("상위 1/3" if top else "상위 1/3 아님"))

    value_conds = [_cond("틀 B 단계 '진입 가능' 또는 '조건부 진입'", None if frame_b_stage is None else frame_b_stage in VALUE_ENTRY_STAGES,
                         frame_b_stage or "판정 없음"),
                   rank_cond("가치 관점 순위 피어 상위 1/3", value_top, value_rank_text)]
    trend_conds = [_cond("틀 A 진입 의견(상승 전환 의견)", frame_a_entry, frame_a_label or ("판정 없음" if frame_a_entry is None else "")),
                   rank_cond("추세 관점 순위 피어 상위 1/3", trend_top, trend_rank_text)]
    short_conds = [_cond("단기 규칙 진입 신호(추세 상승·횡보 + RSI(2) < 10 + 지지선)", short_signal, short_label or "판정 없음")]
    cards = [
        {"key": "value", "label": "가치 스윙", "frame": "틀 B v2", "conditions": value_conds,
         "entry": "틀 B 단계에 따라 분할 매수(조건부는 축소)",
         "exit": "고평가 도달(자기 밴드 z ≥ +1)이고 상승이 배수 주도 → 분할 매도 · 손절 조건(킬·발표 실적 감소·추정치 하향으로 가격 자격 미충족)",
         "hold": "수개월~", "verification": STRATEGY_VERIFICATION["value"]},
        {"key": "trend", "label": "추세", "frame": "틀 A v2", "conditions": trend_conds,
         "entry": "상승 전환 의견 — 다음 날 시가, 크기는 근거 차이(+3 이상 1배, +2 절반) × 국면 크기",
         "exit": "추세 종료·전환 실패(하락 쪽 근거 합 > 상승 쪽 근거 합) → 다음 날 시가",
         "hold": "수주~수개월", "verification": STRATEGY_VERIFICATION["trend"]},
        {"key": "short", "label": "단기", "frame": SHORT_TERM_CONFIG["VERSION"], "conditions": short_conds,
         "entry": "다음 날 시가(손절선까지 3% 이내일 때만)",
         "exit": "+3% 익절 · 종가 < 손절선 · 추세 '하락' · 10거래일 — 먼저 오는 것, 다음 날 시가",
         "hold": "며칠(최대 10거래일)", "verification": STRATEGY_VERIFICATION["short"]},
    ]
    for cd in cards:
        cd["met"] = all(x["ok"] is True for x in cd["conditions"])
        cd["unknown"] = [x["label"] for x in cd["conditions"] if x["ok"] is None]
    met = [cd["key"] for cd in cards if cd["met"]]   # 카드 순서 = 보유 기간이 긴 순서(가치 > 추세 > 단기)
    chosen = met[0] if met else None
    gate_mult = None if gate_level is None else float(PLAYBOOK_CONFIG["GATE_SIZE"].get(int(gate_level), 0.0))
    block = None
    if survival_kill:
        block = {"kind": "survival", "text": "재무 생존성 하드 킬 — 신규 진입 없음(보유 중이면 틀 B 손절 조건)"}
    elif gate_mult is not None and gate_mult == 0:
        block = {"kind": "gate", "text": f"국면 G{int(gate_level)}: 신규 진입 크기 0 — 의견만 표시"}
    for cd in cards:
        cd["role"] = ("진입 전략" if cd["key"] == chosen else ("보조 근거" if cd["met"] else "해당 없음"))
    labels = {cd["key"]: cd["label"] for cd in cards}
    if chosen is None:
        headline = "맞는 진입 전략 없음 — 대기"
    elif block and block["kind"] == "survival":
        headline = f"{labels[chosen]} 조건은 충족하지만 생존성 하드 킬 — 신규 진입 없음"
    elif block:
        headline = f"{labels[chosen]} (의견만 — 국면 크기 0)"
    else:
        headline = f"진입 전략: {labels[chosen]}" + (f" · 보조 근거 {', '.join(labels[k] for k in met[1:])}" if len(met) > 1 else "")
        if chosen == "short":   # 사전 등록 시험 미통과(SHORT_TERM.md) — 규칙은 그대로 두고 표시만 '참고'
            headline += " — 검증 미통과, 참고만"
    return {"chosen": chosen, "chosen_label": labels.get(chosen) if chosen else None, "met": met, "block": block,
            "gate_mult": gate_mult, "headline": headline, "cards": cards,
            "rule": "가치 > 추세 > 단기(보유 기간이 긴 쪽 우선), 조건 자료가 없으면 충족 아님",
            "peer_note": STRATEGY_VERIFICATION["peer_rank"]}
