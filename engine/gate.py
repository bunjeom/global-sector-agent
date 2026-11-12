"""국면 게이트(Market Gate) v1.1: GATE_CONFIG, 히스테리시스, 바스켓 배지 동기화.

engine.py를 기능별로 나눈 모듈. 바깥에서는 `import engine` 후 `engine.이름`으로 쓴다(engine/__init__.py).
"""
import pandas as pd

from .home import build_regime_vix_header

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

def apply_gate_hysteresis(raw_level: int, history_state: dict = None) -> tuple[int, int]:
    """히스테리시스: 강화는 즉시, 완화는 COOLING_DAYS_REQUIRED 거래일 연속 확인 후 한 단계씩.
    반환 (최종 단계, 새 완화 확인 일수). history_state가 없으면 그날 지표 단계로 시작한다.
    evaluate_market_gate와 백테스트 후보 게이트(rule_backtest)가 함께 쓰는 단일 출처."""
    if history_state is None:
        history_state = {"prev_level": raw_level, "cooling_counter": 0}
    prev_level = history_state.get("prev_level", raw_level)
    cooling_counter = history_state.get("cooling_counter", 0)
    if raw_level > prev_level:
        return raw_level, 0
    if raw_level < prev_level:
        new_cooling_counter = cooling_counter + 1
        if new_cooling_counter >= GATE_CONFIG["COOLING_DAYS_REQUIRED"]:
            return prev_level - 1, 0
        return prev_level, new_cooling_counter
    return prev_level, 0


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
    final_level, new_cooling_counter = apply_gate_hysteresis(raw_level, history_state)

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
