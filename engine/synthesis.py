"""4트랙 종합 판정 synthesize_4tracks와 가격 자격 라벨(PRICE_QUAL_LABELS).

engine.py를 기능별로 나눈 모듈. 바깥에서는 `import engine` 후 `engine.이름`으로 쓴다(engine/__init__.py).
"""
import math
from datetime import datetime
from typing import Dict, List, Optional, Any

from .common import NO_DATA, fmt_price, is_earnings_blackout, with_paren

# =====================================================================
# 종합 V 가격 자격 코드 → 화면 표시명 (코드는 내부용, 사용자에게는 이 이름을 보여준다)
PRICE_QUAL_LABELS = {
    "QUALIFIED": "매수 자격 충족",
    "CONDITIONAL": "조건부 매수 자격",
    "OVERVALUED": "고평가 — 매수선 위 (가격 대기)",
    "POOR_RISK_REWARD": "고평가 — 위험 대비 기대수익 부족",
    "VALUE_TRAP": "가치 함정 (저PER 착시)",
    "HARD_KILL": "하드 킬 (진입 금지)",
    "NA": "판정 불가 (데이터 없음)",
}


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
    if current_price is None or not math.isfinite(float(current_price)) or float(current_price) <= 0:
        raise ValueError("현재가가 비어 있어(NaN·0) 종합 판정을 계산할 수 없습니다. 가격 데이터의 마지막 행을 확인하세요.")
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
    e_regime = earn_out.get("regime", {}).get("earnings_regime", "G3")     # G1 ~ G5 또는 "NA"(판정 불가)
    e_regime_display = e_regime
    if e_regime not in ("G1", "G2", "G3", "G4", "G5"):
        e_regime = "G3"   # 판정 불가는 문구에는 그대로 표시하고, 분기 판정에서만 중립(G3) 취급
    c_stage_raw = earn_out.get("regime", {}).get("cycle_stage")            # C1 ~ C4[cite: 2, 6]
    c_code = c_stage_raw.split()[0] if c_stage_raw else None
    th_status = earn_out.get("thesis_driver", {}).get("status", "유효")
    _sm = earn_out.get("sustainability_metrics", {})
    _rd = earn_out.get("revision_dynamics", {})
    rpo_yoy = float(_sm["rpo_yoy_growth"]) if _sm.get("rpo_yoy_growth") is not None else 0.0   # 판정용(없으면 0)
    inc_opm = float(_sm["incremental_opm"]) if _sm.get("incremental_opm") is not None else 0.0
    rev_status = _rd.get("status", "보합")
    rev_30d = float(_rd["change_30d"]) if _rd.get("change_30d") is not None else 0.0
    # 문구용: 값이 없으면 '데이터 없음'
    rpo_txt = f"RPO +{rpo_yoy*100:.0f}%" if _sm.get("rpo_yoy_growth") is not None else f"RPO {NO_DATA}"
    inc_opm_txt = f"증분OPM {inc_opm*100:.1f}%" if _sm.get("incremental_opm") is not None else f"증분OPM {NO_DATA}"

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

    # V 트랙 값은 못 받은 경우 None('데이터 없음')일 수 있다. 가짜 기본값(MDD 0.35 등)으로 채우지 않는다.
    tranches = f_vt.get("buy_tranches", [])
    v_na = bool(f_pq.get("status") == "NA" and not tranches)   # 목표 배수를 만들 근거가 없어 가격선 자체가 없음
    _mdd, _exp = f_vt.get("mdd_ratio"), f_vt.get("expected_return_weighted")
    mdd_ratio = float(_mdd) if _mdd is not None else 0.0
    exp_ret = float(_exp) if _exp is not None else 0.0
    mdd_txt = f"{mdd_ratio:.2f}" if _mdd is not None else NO_DATA
    exp_txt = f"{exp_ret*100:+.1f}%" if _exp is not None else NO_DATA
    _st = f_vt.get("sell_target")
    sell_target_1y = int(_st) if _st is not None else None
    sell_txt = fmt_price(sell_target_1y, is_krx) if sell_target_1y is not None else NO_DATA
    _sz = f_tz.get("self_5y_z")
    self_z_txt = f"{_sz:+.2f}σ" if _sz is not None else NO_DATA
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
    debt_150 = float(f_surv.get("debt_maturity_1y_ratio") or 0.0) > 1.50    # 값이 없으면(None) 킬 조건으로 보지 않는다
    dil_15 = float(f_surv.get("share_dilution_yoy") or 0.0) > 0.15
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

    if v_na:
        # 가격선이 없다: 아래 계산이 깨지지 않도록 중립 자리값만 두고, 화면 문구에서는 쓰지 않는다 (V_NA 분기)
        t1_price = t2_price = t0_price = int(current_price)
        dist_to_t1_pct = None
    else:
        t1_price = int(t1_item["price"]) if t1_item else int(current_price * 0.95)
        t2_price = int(t2_item["price"]) if t2_item else int(t1_price * 0.90)
        t0_price = int(t0_item["price"]) if t0_item else t1_price
        dist_to_t1_pct = round((current_price - t1_price) / max(t1_price, 1), 4)

    # 2-1. 펀더멘탈 가격 자격 4분류 (QUALIFIED / CONDITIONAL / OVERVALUED / VALUE_TRAP)[cite: 6]
    # MDD Ratio를 계산하지 못했으면(None) 0.0으로 보고 탈락시키지 않는다 (데이터 없음 ≠ 위험)
    _mdd_known = _mdd is not None
    _rr_poor = bool(_mdd_known and mdd_ratio < 0.15)
    if is_killed:
        price_qual_code = "HARD_KILL"          # Step 1 하드 킬 (악재·Thesis 붕괴·G5·재무 생존성·거래정지)
    elif not v_na and cyc_code == "CYC_T2" and current_price > t2_price:
        price_qual_code = "VALUE_TRAP"         # 사이클 고점: 이익이 정점이라 PER이 낮아 보이는 착시
    elif v_na:
        price_qual_code = "NA"   # 가격 자격 판정 불가 (목표 배수 근거 데이터 없음)
    elif e_regime == "G4":
        if current_price <= t2_price and mdd_ratio >= 0.15:
            price_qual_code = "CONDITIONAL"  # G4이나 디레이팅 Cap 반영 2·3차 안전마진선 이하 폭락 시 조건부 허용[cite: 6]
        else:
            price_qual_code = "VALUE_TRAP" if _rr_poor else "OVERVALUED"   # 피크아웃 국면의 저PER 착시
    elif _rr_poor:
        price_qual_code = "POOR_RISK_REWARD"   # 실적 훼손 아님: 현재가 대비 시나리오 기대수익이 하락 위험보다 작음(비쌈)
    elif current_price <= t1_price and mdd_ratio >= 0.30 and e_regime in ["G1", "G2", "G3"]:
        price_qual_code = "QUALIFIED"
    elif (t0_item is not None and current_price <= t0_price and e_regime in ["G1", "G2"] and mdd_ratio >= 0.30) or (
        current_price <= t1_price and 0.15 <= mdd_ratio < 0.30
    ):
        price_qual_code = "CONDITIONAL"
    else:
        price_qual_code = "OVERVALUED"
    # 가치 트랙이 '조건부'(예: 자기 밴드가 없어 정상 밴드 확인 불가)로 낮춘 경우 종합에서 '충족'으로 올리지 않는다 (단일 출처)
    if price_qual_code == "QUALIFIED" and f_pq.get("status") == "CONDITIONAL":
        price_qual_code = "CONDITIONAL"
    price_qual_label = PRICE_QUAL_LABELS.get(price_qual_code, price_qual_code)

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
    if is_killed:
        v_status_code = "V_BLOCKED"
        v_badge = "🔴 V 진입 불가 (하드 킬 발동)"
        v_exec_ratio = 0.0
    elif price_qual_code == "VALUE_TRAP":
        v_status_code = "V_BLOCKED"
        v_badge = "🔴 V 진입 불가 (가치 함정 — 이익 정점·피크아웃의 저PER 착시)"
        v_exec_ratio = 0.0
    elif pead_days_left > 0:
        v_status_code = "V_DEFERRED_CHART"
        v_badge = f"🟠 V 매수 보류 (단발 어닝 쇼크 PEAD 냉각기 잔여 {pead_days_left}일)"
        v_exec_ratio = 0.0
    elif sell_target_1y is not None and current_price >= sell_target_1y * 0.95:
        v_status_code = "V_PROFIT_TAKE"
        _over = current_price / max(sell_target_1y, 1e-9) - 1.0
        v_badge = (f"🔵 V 매도목표가 초과 (현재가가 목표가보다 {_over*100:+.1f}% · 보유분 분할 차익실현, 신규 매수 없음)"
                   if _over > 0 else "🔵 V 목표가 도달 (차익실현 구간 · 신규 매수 없음)")
        v_exec_ratio = 0.0
    elif v_na:
        v_status_code = "V_NA"
        v_badge = "⚪ V 가격 판정 불가 (데이터 없음)"
        v_exec_ratio = 0.0
    elif e_regime == "G4" and current_price > t2_price:
        v_status_code = "V_WAIT_EARNINGS_TURN"
        v_badge = "🟣 V 실적 턴어라운드 대기 (G4 피크아웃 경계)"
        v_exec_ratio = 0.0
    elif price_qual_code == "POOR_RISK_REWARD":
        v_status_code = "V_WAIT_PRICE"
        v_badge = f"⚪ V 가격 대기 (위험 대비 기대수익 부족 · MDD Ratio {mdd_ratio:.2f} < 0.15)"
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
    _t_has = bool(t_ticket.get("setup_id"))                       # 실제 계산된 T 티켓 존재 여부
    _t_px = ((lambda v: int(v)) if is_krx else (lambda v: round(float(v), 2)))   # USD 센트 절사 방지
    _t_px = (lambda f: (lambda v: None if v is None else f(v)))(_t_px)

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

    if is_killed:
        arch_code = "ARCH_8"
        arch_title = "하드 킬 · 투자 부적격 (신규 진입 전면 금지 및 보유분 청산/동결)"
    elif price_qual_code == "VALUE_TRAP" and cyc_code != "CYC_T2":
        arch_code = "ARCH_8"
        arch_title = f"가치 함정 경계 · 신규 진입 금지 (실적 {e_regime_display} 피크아웃 — 낮은 PER은 착시)"
    elif cyc_code == "CYC_T2" or c_code == "C3" or e_regime == "G4":
        arch_code = "ARCH_6"
        arch_title = f"사이클·실적 피크아웃 경계 ({e_regime_display} / {cyc_code or '피크아웃'} · 저PER 착시 주의)"
    elif e_regime in ["G1", "G2"] and v_status_code in ["V_EXEC_FULL", "V_EXEC_HALF"] and (
        tech_state_code in ["STATE_B", "STATE_S", "STATE_H"] or t_status_code == "T_ENTRY"
    ):
        arch_code = "ARCH_1"
        arch_title = f"밸류·모멘텀 동시 점화 (실적 {e_regime_display} + V 매수 집행 + 차트 추세/반전 동시 확인)"
    elif is_st1_active:
        arch_code = "ARCH_4"
        arch_title = f"투매 소진 반등주 (실적 {e_regime_display} + ST-1 과매도 반전 셋업)"
    elif price_qual_code in ["QUALIFIED", "CONDITIONAL"] and v_status_code == "V_DEFERRED_CHART":
        arch_code = "ARCH_3"
        arch_title = f"저평가 도달 · 차트 바닥 대기 (매수선 도달 완료 · {tech_state_code} 하락 멈춤 대기)"
    elif e_regime in ["G1", "G2"] and v_status_code == "V_WAIT_PRICE" and tech_state_code in ["STATE_S", "STATE_H"]:
        arch_code = "ARCH_2"
        arch_title = f"고성장 추세 주도주 (실적 {e_regime_display} · V는 지정가 대기, T 추세 돌파·눌림 공략)"
    elif c_code in ["C1", "C2"] and (rev_30d > 0 or tech_state_code in ["STATE_B", "STATE_S"]):
        arch_code = "ARCH_5"
        arch_title = f"사이클 바닥 턴어라운드 ({c_code} 실물 회복 + 바닥 전환)"
    else:
        arch_code = "ARCH_7"
        if price_qual_code in ("POOR_RISK_REWARD", "OVERVALUED"):
            arch_title = f"중립 · 가치 기준 고평가 관망 (실적 {e_regime_display} · {tech_state_code} · V는 매수선까지 가격 대기)"
        else:
            arch_title = f"중립 · 박스권 관망 (실적 {e_regime_display} · {tech_state_code} · 지정가 및 박스 하단 대기)"

    # 🔥 최고 확신 배지: 거시 트랙 조건을 완전 배제하고 순수하게 ARCH_1일 때만 부여[cite: 6]
    high_conviction_badge = bool(arch_code == "ARCH_1")

    # =================================================================
    # Step 4 & 3장. 거시 오버레이 추출 및 직관적 3줄 브리핑 생성[cite: 6]
    # =================================================================
    # 거시 트랙이 보류되면 13주 분해가 없다 → None('데이터 없음'). 임의 기본값(-5% 등)을 넣지 않는다.
    m_attr_13 = (macro_out.get("attribution_13w") or {}).get("13w")
    if m_attr_13:
        ret_13w_act = float(m_attr_13["actual"])
        ret_13w_mac_mkt = round(float(m_attr_13["market"]) + float(m_attr_13["macro_extra"]), 4)
        ret_13w_idio = float(m_attr_13["idiosyncratic"])
        attr13_txt = (f"(최근 13주 주가 변화 {ret_13w_act*100:+.1f}% 중 시장·거시 몫 {ret_13w_mac_mkt*100:+.1f}%p, "
                      f"기업 고유 몫 {ret_13w_idio*100:+.1f}%p)")
    else:
        ret_13w_act = ret_13w_mac_mkt = ret_13w_idio = None
        attr13_txt = f"(13주 수익률 분해: 거시 트랙 보류로 {NO_DATA})"

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
        elif v_status_code == "V_WAIT_PRICE" and price_qual_code == "POOR_RISK_REWARD":
            v_act_str = f"V(가치)는 현재가 기준 시나리오 기대수익({exp_txt})이 하락 위험 대비 부족(MDD Ratio {mdd_txt})해 신규 매수 없이 가격 대기이며"
        elif v_status_code == "V_WAIT_PRICE":
            v_act_str = f"V(가치) 기준으로는 아직 매수선 위여서 가격 대기(1차선 {fmt_price(t1_price, is_krx)}까지 {-dist_to_t1_pct*100:+.1f}% 필요) 구간이나"
        elif v_status_code == "V_NA":
            v_act_str = f"V(가치)는 목표 배수를 만들 데이터가 없어 가격 판정이 불가능({NO_DATA})하며"
        elif v_status_code == "V_PROFIT_TAKE":
            if current_price > sell_target_1y:
                v_act_str = (f"V(가치) 기준으로는 현재가가 이미 매도목표가({sell_txt})보다 {(current_price / max(sell_target_1y, 1e-9) - 1) * 100:+.1f}% 높아 "
                             f"신규 매수는 없고 보유분만 분할 차익실현 구간이며")
            else:
                v_act_str = f"현재 V(가치)는 매도목표가({sell_txt}) 권역에 도달해 분할 차익실현 구간이며"
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
        f"실적 국면은 [{e_regime_display} {earn_out['regime']['regime_label']}]({rpo_txt}, {inc_opm_txt})이고, "
        f"펀더멘탈은 자기 밴드 {self_z_txt} · MDD Ratio {mdd_txt}(기대수익 {exp_txt})이며, "
        f"차트는 [{tech_out['lifecycle_state']['state_label']}] 상태입니다. "
        f"{attr13_txt}"
    )

    # 셋째 줄 [다음 방아쇠 및 종료 조건 — Trigger & Invalidation][cite: 6]
    t_cond_str = t_banner.get("next_or_end_condition", "T 셋업 신호 대기")
    if v_status_code == "V_DEFERRED_CHART":
        v_cond_str = f"V {active_tr_num}차 본격 집행 조건: 쌍바닥 후보 또는 반등 양봉(RVOL≥1.5) 추가로 하락 멈춤(STATE_B) 진입 시"
    elif v_status_code == "V_EXEC_HALF":
        v_cond_str = f"V 잔여 50% 추가 집행: 차트 바닥 전환(STATE_B/S) 확정 또는 지수 레짐 🟢 상향 시"
    elif v_status_code == "V_WAIT_PRICE" and price_qual_code == "POOR_RISK_REWARD":
        v_cond_str = f"V 개방 조건: MDD Ratio 0.15 이상 회복(주가 조정 또는 이익 추정 상향) 및 1차 매수기준선({fmt_price(t1_price, is_krx)}) 도달 시"
    elif v_status_code == "V_WAIT_PRICE":
        v_cond_str = f"V 개방 조건: 주가가 1차 매수기준선({fmt_price(t1_price, is_krx)}) 이하로 조정 도달 시"
    elif v_status_code == "V_NA":
        v_cond_str = f"V 개방 조건: 목표 배수 근거 데이터({NO_DATA}) 확보 전까지 판정 보류 ｜ 핵심 드라이버 무효화 감시: {earn_out['thesis_driver']['invalidation_condition']}"
    else:
        v_cond_str = f"V 매도목표가: {sell_txt} ｜ 핵심 드라이버 무효화 감시: {earn_out['thesis_driver']['invalidation_condition']}"

    line_3_trigger = f"{t_cond_str} ｜ {v_cond_str}"

    # 5장. 전 종목 상태 변화 감지 플래그 (홈 화면 app.py 기회 레이더 연동용)[cite: 6]
    approaching_v = bool(e_regime in ["G1", "G2"] and dist_to_t1_pct is not None and 0.0 < dist_to_t1_pct <= 0.05)
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
        "item": with_paren(earn_out['thesis_driver']['invalidation_condition'], earn_out['thesis_driver']['current_status']),
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
            "current_price": (int(current_price) if is_krx else round(float(current_price), 2)),   # 미국은 센트 유지
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
                "label": f"{e_regime_display} · {earn_out['regime']['regime_label'][:10]}",
                "state": f"{rpo_txt} · {inc_opm_txt} · 리비전 {rev_status}"
            },
            "fundamental": {
                "icon": f_pq.get("grade_icon", "🟡"),
                "label": price_qual_label if v_na else f"{price_qual_label} (1차선 {fmt_price(t1_price, is_krx)})",
                "state": f"MDD Ratio {mdd_txt} · 자기 밴드 {self_z_txt}"
            },
            "macro": {
                "icon": macro_out["classification"]["grade_icon"],
                "label": (f"{macro_out['classification']['label'][:10]} (고유 R² {int(macro_out['classification']['r2_macro_own']*100)}%)"
                          if macro_out['classification'].get('r2_macro_own') is not None else macro_out['classification']['label']),
                "state": f"{macro_out['macro_regime']['current_regime']} ({macro_out['residual_check'].get('status', '')})"
            },
        },
        "v_action_card": {
            "status_code": v_status_code,
            "price_qual_code": price_qual_code,
            "price_qual_label": price_qual_label,
            "badge_text": v_badge,
            "active_tranche": active_tr_num,
            "tranche_price": active_tr_price,
            "distance_to_1st_tranche_pct": dist_to_t1_pct,
            "execution_ratio": v_exec_ratio,
            "recommended_weight": rec_v_weight,
            "sell_target_1y": sell_target_1y,
            "expected_return_weighted": exp_ret if _exp is not None else None,
            "mdd_ratio": mdd_ratio if _mdd is not None else None,
            "high_confidence_overlap": has_high_conf_overlap,
        },
        "t_action_card": {
            "verdict": t_verdict_final,
            "status_code": t_status_code,
            "badge_text": t_badge,
            "setup_id": setup_id,
            "grade": t_grade,
            "risk_modifier_applied": t_risk_mod,
            # 티켓(셋업)이 실제로 계산된 경우에만 가격·수량을 싣는다. 셋업이 없으면 현재가로 만든
            # 임의 기본값을 넣지 않고 None(화면 '데이터 없음')과 사유만 전달한다 (가짜 숫자 금지).
            "has_ticket": _t_has,
            "no_ticket_reason": None if _t_has else (t_ticket.get("reason") or t_banner.get("headline_reason") or "활성 셋업 없음"),
            "next_condition": t_banner.get("next_or_end_condition"),
            "entry_price": _t_px(t_ticket.get("entry_price")) if _t_has else None,
            "stop_price": _t_px(t_ticket.get("stop_price")) if _t_has else None,
            "shares": int(int(t_ticket.get("shares") or 0) * t_risk_mod) if _t_has else None,
            "target_price": _t_px(t_ticket.get("target_1")) if _t_has else None,
            "expected_r": (float(t_ticket["expected_r"]) if t_ticket.get("expected_r") is not None else None) if _t_has else None,
            "time_stop_rule": (
                f"{t_ticket['time_stop_days']}거래일째 {fmt_price(t_ticket['time_stop_min_price'], is_krx)} 미달 시 다음 날 시가 청산"
                if _t_has and t_ticket.get("time_stop_days") is not None and t_ticket.get("time_stop_min_price") is not None
                else None),
            "invalidation_rule": t_ticket.get("invalidation") if _t_has else None,
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
                (f"📊 [13주 수익률 분해 사실] 최근 13주 수익률({ret_13w_act*100:+.1f}%) 중 "
                 f"시장·거시 설명 몫은 {ret_13w_mac_mkt*100:+.1f}%p, 기업 고유 몫은 {ret_13w_idio*100:+.1f}%p")
                if ret_13w_act is not None else f"📊 [13주 수익률 분해] 거시 트랙 보류로 {NO_DATA}"
            ),
        },
        "unified_watchlist": unified_watchlist,
        "radar_flags": radar_flags,
    }
