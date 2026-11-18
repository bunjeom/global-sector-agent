"""강의 조건 시험 — 2단계: L2 매수 사건 · 환경 · 확인 정의 (2026-11-18 사전 등록, AGENTS.md 4장 '강의 조건 시험').

정의의 근거: 사전등록안 3장(사건 E01~E15, 환경 F01~F09, 확인 C01~C07, 짝짓기 표)과 질문 결정 Q13~Q60.
정리본·Q 결정이 정하지 않은 세부(창 길이, 사건 위치 등)는 '구현에서 정한 것'으로 AGENTS.md에 적었고, **결과를 보기 전에 정했다.**
모든 사건·조건은 그날 종가까지의 자료만 쓴다. 스윙은 오른쪽 5봉이 지난 날부터 '확정'.

종목 하나의 재료 p(lec_prepare 결과) → lec_l2_events(p, ...) → 사건 표(사건 하나에 한 줄, 환경·확인 열은 1/0/NaN).
NaN = 그 조건에 필요한 자료가 없음(섹터 ETF·지수·실적일). 그 조합에서 빠진다(통과·실패로 세지 않음).
"""
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from .lecture_signals import LECTURE_CONFIG, lec_rs_daily, lec_state, lec_zone

# ---------------------------------------------------------------------
# 사건 · 환경 · 확인 · 짝짓기 표 (사전등록안 3장)
# ---------------------------------------------------------------------
L2_EVENTS: Dict[str, Tuple[str, str]] = {
    "E01": ("기준보다 먼저 전고점 돌파", "돌파"),
    "E02": ("돌파 확정(4조건)", "돌파"),
    "E03": ("볼린저 수축 뒤 상향 확장", "돌파"),
    "E04": ("SR Flip(확정 돌파 뒤 돌파선 리테스트)", "눌림"),
    "E05": ("쌍바닥 확정(넥 돌파)", "바닥"),
    "E06": ("쌍바닥 후보(L1 확정일, 넥 돌파 전)", "단독"),
    "E07": ("이평 쌍바닥 공격(캔들 쌍바닥 + 5일선 상승)", "바닥"),
    "E08": ("이평 쌍바닥 보수(5일선 쌍바닥 + 20일선 상승)", "바닥"),
    "E09": ("하락 멈춤(전환 증거 성격 다른 2개)", "바닥"),
    "E10": ("하락 추세선 4번째 이상 돌파 시도", "단독"),
    "E11": ("하락 추세선 3번째 돌파 시도", "단독"),
    "E12": ("0.25 확장 진짜 판정(추세선·쌍바닥 박스)", "단독"),
    "E13": ("분출(상승 쐐기 상단 장대양봉 돌파)", "단독"),
    "E14": ("채널 복사 리테스트 성공", "단독"),
    "E15": ("삼각수렴 상향 이탈", "단독"),
}
L2_ENVS: Dict[str, str] = {
    "F01": "시장 🟢", "F02": "시장 🟢·🟡", "F03": "시장 🔴·⚫", "F04": "종목 상승 상태", "F05": "종목 횡보 상태",
    "F06": "RS 주도(종목 vs 섹터 ETF, 일봉)", "F07": "RS 주도·개선", "F08": "120일선 상승", "F09": "정배열 + 세 선 상승",
}
L2_CONFS: Dict[str, str] = {
    "C01": "사건 날 양봉 + RVOL ≥ 1.5", "C02": "돌파봉 거래량 > 되돌림 음봉 거래량 최대", "C03": "사건 자리 겹침 지지 3종류 이상",
    "C04": "물량 소화(저항 아래 저점 2개 연속 상승)", "C05": "조정 구간 평균 RVOL < 1", "C06": "전환 증거 3개 이상",
    "C07": "사건 날 거래량 이유 = 실적",
}
L2_PAIRING: Dict[str, Dict[str, Sequence[str]]] = {
    "돌파": {"events": ("E01", "E02", "E03"), "envs": ("F01", "F02", "F03", "F04", "F05", "F06", "F07", "F08"),
            "confs": ("C01", "C02", "C04", "C07")},
    "바닥": {"events": ("E05", "E07", "E08", "E09"), "envs": ("F01", "F02", "F03", "F07", "F08"), "confs": ("C01", "C03", "C06")},
    "눌림": {"events": ("E04",), "envs": ("F01", "F02", "F03", "F04", "F06", "F07", "F08", "F09"), "confs": ("C01", "C03", "C05")},
}
L2_ONLY_FOR: Dict[str, Tuple[str, ...]] = {"C02": ("E02",)}     # C02는 E02에만 (짝짓기 표)
L2_REF_PAIRS: Tuple[Tuple[str, str, str], ...] = (
    ("E05", "E06", "쌍바닥 확정 − 후보"), ("E10", "E11", "추세선 4번째 이상 − 3번째 시도"), ("E08", "E07", "이평 쌍바닥 보수 − 공격"))


def lec_l2_combos() -> List[Dict[str, Optional[str]]]:
    """짝짓기 표의 조합 전부: 사건 1 + 환경 0~1 + 확인 0~1. 단독 사건은 사건만."""
    out = []
    for grp in L2_PAIRING.values():
        for e in grp["events"]:
            for f in (None,) + tuple(grp["envs"]):
                for c in (None,) + tuple(grp["confs"]):
                    if c is not None and c in L2_ONLY_FOR and e not in L2_ONLY_FOR[c]:
                        continue
                    out.append({"event": e, "env": f, "conf": c})
    for e, (_, typ) in L2_EVENTS.items():
        if typ == "단독":
            out.append({"event": e, "env": None, "conf": None})
    for x in out:
        x["id"] = "+".join(k for k in (x["event"], x["env"], x["conf"]) if k)
    return out


# ---------------------------------------------------------------------
# 작은 도우미
# ---------------------------------------------------------------------
_LEC_LR = LECTURE_CONFIG["SWING_LR"]


def _lec_nconf(origins: np.ndarray, t: int) -> int:
    """t일까지 확정된 스윙 개수(origin ≤ t − 5)."""
    return int(np.searchsorted(origins, t - _LEC_LR, "right"))


def _lec_hw(level: float, atr: float) -> float:
    lo, hi = lec_zone(level, atr)
    return (hi - lo) / 2.0


def _lec_logline(o1: int, p1: float, o2: int, p2: float):
    """로그 가격 축의 직선(Q46): 두 점 (o1, p1), (o2, p2)를 지나는 선의 t일 값."""
    s = (np.log(p2) - np.log(p1)) / float(o2 - o1)
    return lambda t: float(np.exp(np.log(p2) + s * (t - o2))), s


def _lec_arr(b: pd.DataFrame, k: str) -> np.ndarray:
    return b[k].to_numpy(dtype=float)


# ---------------------------------------------------------------------
# 돌파 (E01·E02·E04, L4의 W14 가짜 돌파에도 씀)
# ---------------------------------------------------------------------
def lec_breakouts(b: pd.DataFrame, sw: Dict[str, np.ndarray]) -> List[Dict[str, Any]]:
    """돌파 = 종가가 '그날까지의 최근 확정 스윙 고점' 위로 처음 마감한 날 t0(Q7). 그 고점이 최근 고점이 된 뒤 처음 넘은 날만.

    돌파 확정(E02, Q19~Q22): t0 ~ t0+20 안에 4조건이 모두 처음 갖춰진 날
        ① 돌파봉(t0) 양봉 + RVOL ≥ 1.5  ② t0부터 그날까지 종가가 돌파선 구간 하단 아래로 마감한 적 없음
        ③ 20·60일선 5일 기울기 모두 양수  ④ t0 뒤에 생긴 확정 저점이 t0 때의 최근 확정 저점보다 높음(돌파 이후 구조)
    가짜(W14): 확정 전에 종가 < 돌파선 구간 하단인 날(20거래일 안).
    SR Flip(E04, Q23): 확정일 뒤 20거래일 안에 저가 ≤ 구간 상단 이고 종가 ≥ 구간 하단인 첫날(그 전에 종가 < 구간 하단이면 없음).
    구간 = 돌파선 ± max(1%, 0.5×ATR14), ATR은 돌파일 값으로 고정."""
    c = LECTURE_CONFIG
    o, cl, lo = _lec_arr(b, "Open"), _lec_arr(b, "Close"), _lec_arr(b, "Low")
    atr, rvol = _lec_arr(b, "ATR14"), _lec_arr(b, "RVOL")
    g20, g60 = _lec_arr(b, "SMA20_SLOPE"), _lec_arr(b, "SMA60_SLOPE")
    ho, hp, lo_o, lo_p = sw["h_o"], sw["h_p"], sw["l_o"], sw["l_p"]
    n = len(cl)
    W, R = c["BREAKOUT_CONFIRM_DAYS"], c["RETEST_DAYS"]
    out = []
    cur_k, crossed = -1, False
    for t in range(n):
        nh = _lec_nconf(ho, t)
        if nh == 0:
            continue
        k = nh - 1
        if k != cur_k:
            cur_k, crossed = k, False
        if crossed or not (cl[t] > hp[k]):
            continue
        crossed = True
        H = float(hp[k])
        zl, zu = lec_zone(H, atr[t])
        c1 = bool(cl[t] > o[t] and rvol[t] >= c["RVOL_UP"])
        nl0 = _lec_nconf(lo_o, t)
        lprev = float(lo_p[nl0 - 1]) if nl0 >= 1 else np.nan
        conf = fake = flip = None
        for d in range(t, min(t + W, n - 1) + 1):
            if cl[d] < zl:
                fake = d
                break
            if not c1:
                continue
            nl = _lec_nconf(lo_o, d)
            new_lows = lo_p[nl0:nl][lo_o[nl0:nl] > t]
            c4 = bool(np.isfinite(lprev) and len(new_lows) and np.any(new_lows > lprev))
            if g20[d] > 0 and g60[d] > 0 and c4:
                conf = d
                break
        if conf is not None:
            for d in range(conf + 1, min(conf + R, n - 1) + 1):
                if cl[d] < zl:
                    break
                if lo[d] <= zu and cl[d] >= zl:
                    flip = d
                    break
        out.append({"t0": t, "level": H, "h_origin": int(ho[k]), "zl": zl, "zu": zu, "c1": c1,
                    "confirm": conf, "fake": fake, "flip": flip})
    return out


def lec_etf_below_high(etf_b: Optional[pd.DataFrame], etf_sw: Optional[Dict[str, np.ndarray]]) -> pd.Series:
    """섹터 ETF 종가 < 그날까지의 ETF 최근 확정 스윙 고점 (E01의 '기준은 아직 전고점 아래', Q7). 날짜별 1/0."""
    if etf_b is None or not len(etf_b):
        return pd.Series(dtype=float)
    cl = _lec_arr(etf_b, "Close")
    ho, hp = etf_sw["h_o"], etf_sw["h_p"]
    out = np.full(len(cl), np.nan)
    for t in range(len(cl)):
        nh = _lec_nconf(ho, t)
        if nh:
            out[t] = float(cl[t] < hp[nh - 1])
    return pd.Series(out, index=etf_b.index)


# ---------------------------------------------------------------------
# 볼린저 (E03, Q17)
# ---------------------------------------------------------------------
def lec_bb_up(b: pd.DataFrame) -> List[int]:
    """수축(밴드 폭 120일 백분위 ≤ 20) 뒤 처음으로 종가가 밴드 밖에 마감한 날 중 위쪽(상단 위)인 날. 밖으로 나가면 수축 해제."""
    c = LECTURE_CONFIG
    cl, up, dn, pc = _lec_arr(b, "Close"), _lec_arr(b, "BB_UP"), _lec_arr(b, "BB_LOW"), _lec_arr(b, "BB_WIDTH_PCTL120")
    out, sq = [], False
    for t in range(len(cl)):
        if sq and np.isfinite(up[t]) and cl[t] > up[t]:
            out.append(t)
            sq = False
        elif sq and np.isfinite(dn[t]) and cl[t] < dn[t]:
            sq = False
        if pc[t] <= c["BB_SQZ_PCTL"]:
            sq = True
    return out


# ---------------------------------------------------------------------
# 쌍바닥 (E05·E06·E07·E12 박스, Q55·Q57~Q59)
# ---------------------------------------------------------------------
def lec_double_bottoms(b: pd.DataFrame, sw: Dict[str, np.ndarray], st: np.ndarray) -> List[Dict[str, Any]]:
    """새 확정 저점 L1이 생길 때마다: 바로 앞 확정 저점 L2, 두 저점 사이 가장 높은 확정 고점 N(넥).
    조건: L2 날 종목 상태 = 하락(Q55 '하락 구조 → L2'), L1 ≥ L2 구간 하단, 간격 ≤ 250거래일, 폭(N − 낮은 바닥) ÷ N ≥ 5%(Q59).
    E06 후보 = L1 확정일(그날 종가 ≤ 넥 구간 상단일 때만). E05 확정 = 그날부터 종가 > 넥 구간 상단인 첫날
        (그 전에 종가 < L1 구간 하단이면 무효, 확정일 뒤 250거래일까지).
    E07 = E05 날부터 5일선 기울기 > 0인 첫날(무효 전). E12 박스 = E05 날부터 10거래일 안 장대양봉 종가 > 넥 + 0.25×폭
        (그 전에 종가 < 넥 구간 하단이면 가짜). 구간의 ATR은 L1 확정일 값."""
    c = LECTURE_CONFIG
    cl, atr, big = _lec_arr(b, "Close"), _lec_arr(b, "ATR14"), b["BIG_BULL"].to_numpy(dtype=bool)
    g5 = _lec_arr(b, "SMA5_SLOPE")
    ho, hp, lo_o, lo_p = sw["h_o"], sw["h_p"], sw["l_o"], sw["l_p"]
    n = len(cl)
    out = []
    for j in range(1, len(lo_o)):
        o1, o2 = int(lo_o[j]), int(lo_o[j - 1])
        tc = o1 + _LEC_LR
        if tc >= n:
            break
        L1, L2 = float(lo_p[j]), float(lo_p[j - 1])
        if o1 - o2 > c["DBL_MAX_GAP"] or st[o2] != -1:
            continue
        a_ = np.searchsorted(ho, o2, "right")
        z_ = np.searchsorted(ho, o1, "left")
        if z_ <= a_:
            continue
        kk = a_ + int(np.argmax(hp[a_:z_]))
        N = float(hp[kk])
        a_tc = atr[tc]
        if L1 < lec_zone(L2, a_tc)[0]:
            continue
        width = N - min(L1, L2)
        if N <= max(L1, L2) or width / N < c["TGT_MIN_WIDTH"]:
            continue
        nzl, nzu = lec_zone(N, a_tc)
        l1zl = lec_zone(L1, a_tc)[0]
        cand = bool(cl[tc] <= nzu)
        conf = e07 = e12 = None
        for d in range(tc, min(tc + c["DBL_MAX_GAP"], n - 1) + 1):
            if cl[d] < l1zl:
                break
            if cl[d] > nzu:
                conf = d
                break
        if conf is not None:
            for d in range(conf, min(tc + c["DBL_MAX_GAP"], n - 1) + 1):
                if cl[d] < l1zl:
                    break
                if g5[d] > 0:
                    e07 = d
                    break
            lvl = N + 0.25 * width
            for d in range(conf, min(conf + c["EXT_DAYS"], n - 1) + 1):
                if cl[d] < nzl:
                    break
                if big[d] and cl[d] > lvl:
                    e12 = d
                    break
        out.append({"tc": tc, "l1": L1, "l1_origin": o1, "l2": L2, "l2_origin": o2, "neck": N, "neck_origin": int(ho[kk]),
                    "cand": cand, "confirm": conf, "e07": e07, "e12": e12})
    return out


def lec_ma_double_bottoms(b: pd.DataFrame, st: np.ndarray) -> List[Dict[str, Any]]:
    """5일선 쌍바닥(Q27·Q29·Q34): 5일선 값의 좌우 5봉 스윙 저점 두 개 m2(앞)·m1(뒤), m1 ≥ m2 구간 하단, 간격 ≤ 250, m2 날 상태 = 하락.
    '이어짐' = 5일선이 두 저점 사이 5일선 최고값을 다시 넘은 날, 그날 5일선 기울기 > 0('방향 바뀜') → 5일선 쌍바닥 완성.
    E08 = 완성일부터 20일선 기울기 > 0인 첫날. 5일선이 m1 아래로 가면 무효(Q34). m1 확정일 뒤 250거래일까지."""
    c = LECTURE_CONFIG
    m = _lec_arr(b, "SMA5")
    g5, g20, atr, low = _lec_arr(b, "SMA5_SLOPE"), _lec_arr(b, "SMA20_SLOPE"), _lec_arr(b, "ATR14"), _lec_arr(b, "Low")
    n = len(m)
    rm = pd.Series(m).rolling(2 * _LEC_LR + 1, center=True).min().to_numpy()
    lows = np.flatnonzero(np.isfinite(m) & (m == rm))
    out = []
    for j in range(1, len(lows)):
        i1, i2 = int(lows[j]), int(lows[j - 1])
        tc = i1 + _LEC_LR
        if tc >= n:
            break
        if i1 - i2 > c["DBL_MAX_GAP"] or st[i2] != -1:
            continue
        m1, m2 = m[i1], m[i2]
        if m1 < lec_zone(m2, atr[i1])[0]:
            continue
        neck = float(np.nanmax(m[i2:i1 + 1]))
        done = e08 = None
        for d in range(tc, min(tc + c["DBL_MAX_GAP"], n - 1) + 1):
            if m[d] < m1:
                break
            if done is None and m[d] > neck and g5[d] > 0:
                done = d
            if done is not None and g20[d] > 0:
                e08 = d
                break
        if e08 is not None:
            out.append({"t": e08, "loc": float(low[i1]), "loc_origin": i1})
    return out


# ---------------------------------------------------------------------
# 전환 증거 (E09·C06, 정리본 3-5, Q13~Q16)
# ---------------------------------------------------------------------
def lec_transition(b: pd.DataFrame, sw: Dict[str, np.ndarray], upf: np.ndarray) -> Dict[str, np.ndarray]:
    """그날의 전환 증거 4개(1/0):
    ev1 이전 저점 지킴 = 최근 확정 저점 ≥ 그 앞 확정 저점 구간 하단(Q14)   ev2 새 고점 = 최근 확정 저점 뒤 종가가 그 앞 확정 고점을 넘은 적 있음(Q2 정의)
    ev3 MACD 하락 모멘텀 감소 = 히스토그램 < 0 + 3일 연속 상승(Q16)          ev4 거래량 붙은 반등 = 양봉 + RVOL ≥ 1.5(Q15)
    성격: ev1·ev2 = 가격 구조, ev3 = 모멘텀, ev4 = 수급(앱 S-FALL과 같은 묶음). cats = 성격 수, cnt = 증거 수."""
    c = LECTURE_CONFIG
    n = len(b)
    cl, o, atr, rv, mh = _lec_arr(b, "Close"), _lec_arr(b, "Open"), _lec_arr(b, "ATR14"), _lec_arr(b, "RVOL"), _lec_arr(b, "MACD_HIST")
    lo_o, lo_p = sw["l_o"], sw["l_p"]
    ev1 = np.zeros(n, dtype=bool)
    for t in range(n):
        nl = _lec_nconf(lo_o, t)
        if nl >= 2:
            ev1[t] = lo_p[nl - 1] >= lec_zone(lo_p[nl - 2], atr[lo_o[nl - 1]])[0]
    ev2 = np.asarray(upf, dtype=bool)
    ev3 = np.zeros(n, dtype=bool)
    if n > 3:
        ev3[3:] = (mh[3:] < 0) & (mh[3:] > mh[2:-1]) & (mh[2:-1] > mh[1:-2]) & (mh[1:-2] > mh[:-3])
    ev4 = (cl > o) & (rv >= c["RVOL_UP"])
    cats = (ev1 | ev2).astype(int) + ev3.astype(int) + ev4.astype(int)
    cnt = ev1.astype(int) + ev2.astype(int) + ev3.astype(int) + ev4.astype(int)
    return {"ev1": ev1, "ev2": ev2, "ev3": ev3, "ev4": ev4, "cats": cats, "cnt": cnt}


def lec_fall_release(st: np.ndarray, tr: Dict[str, np.ndarray]) -> List[int]:
    """E09: 전날 상태가 하락이고, 성격이 다른 증거 2개 이상이 처음 갖춰진 날(하락 상태 한 번에 한 번)."""
    out, fired = [], False
    for t in range(1, len(st)):
        if st[t - 1] != -1:
            fired = False
            continue
        if not fired and tr["cats"][t] >= 2:
            out.append(t)
            fired = True
    return out


# ---------------------------------------------------------------------
# 추세선 (E10·E11·E12, Q45·Q48~Q50)
# ---------------------------------------------------------------------
def lec_downline_attempts(b: pd.DataFrame, sw: Dict[str, np.ndarray]) -> Dict[str, List[Tuple[int, int]]]:
    """하락 추세선: 낮아진 확정 고점 두 개(가장 최근 두 고점)로 로그 축 직선. 선이 없을 때 새 쌍이 생기면 긋는다.
    돌파 시도(Q49) = 종가 > 선 구간 상단(구간은 그날 선 값 ± max(1%, 0.5×ATR)). 다음 시도는 종가가 구간 상단 이하로 돌아온 뒤.
    시도마다 0.25 판정(Q50): 폭 = 첫 고점 − 첫 고점 뒤 시도일까지 최저 저가(산술), 시도일부터 10거래일 안
        장대양봉 종가 > 그날 선 값 + 0.25×폭 = 진짜(그날 = E12, 선 끝) / 종가 < 선 구간 하단 = 가짜(선 유지).
    새 확정 고점이 선 구간 위에 생기면 선은 깨진 것으로 보고 지운다(그 아래면 '닿음', 선 유지).
    결과: attempts = [(시도일, 몇 번째)], real = [진짜 판정일]."""
    c = LECTURE_CONFIG
    cl, lo, atr = _lec_arr(b, "Close"), _lec_arr(b, "Low"), _lec_arr(b, "ATR14")
    big = b["BIG_BULL"].to_numpy(dtype=bool)
    ho, hp = sw["h_o"], sw["h_p"]
    n = len(cl)
    attempts: List[Tuple[int, int]] = []
    real: List[int] = []
    line = None
    used = 0          # 이 개수까지의 고점은 이미 선에 썼음
    for t in range(n):
        nh = _lec_nconf(ho, t)
        if line is not None and nh > line["nh"]:
            for k in range(line["nh"], nh):
                if hp[k] > lec_zone(line["f"](int(ho[k])), atr[int(ho[k])])[1]:
                    line = None
                    break
            if line is not None:
                line["nh"] = nh
        if line is None:
            if nh >= 2 and nh - 1 > used and hp[nh - 1] < hp[nh - 2]:
                oa, ob = int(ho[nh - 2]), int(ho[nh - 1])
                f, _ = _lec_logline(oa, hp[nh - 2], ob, hp[nh - 1])
                zu0 = lec_zone(f(t), atr[t])[1]
                line = {"f": f, "oa": oa, "ha": float(hp[nh - 2]), "nh": nh, "k": 0, "armed": bool(cl[t] <= zu0), "pend": None}
                used = nh - 1
            else:
                continue
        v = line["f"](t)
        zl, zu = lec_zone(v, atr[t])
        if line["armed"] and cl[t] > zu:
            line["k"] += 1
            attempts.append((t, line["k"]))
            line["armed"] = False
            width = line["ha"] - float(np.min(lo[line["oa"]:t + 1]))
            line["pend"] = {"until": t + c["EXT_DAYS"], "width": width}
        elif (not line["armed"]) and cl[t] <= zu:
            line["armed"] = True
        pd_ = line["pend"]
        if pd_ is not None:
            if cl[t] < zl:
                line["pend"] = None
            elif big[t] and cl[t] > v + 0.25 * pd_["width"]:
                real.append(t)
                line = None
                continue
            elif t >= pd_["until"]:
                line["pend"] = None
    return {"attempts": attempts, "real": real}


# ---------------------------------------------------------------------
# 쐐기·삼각·채널 (E13·E15·E14, Q51~Q54·Q60)
# ---------------------------------------------------------------------
def _lec_last_two(origins: np.ndarray, prices: np.ndarray, t: int):
    k = _lec_nconf(origins, t)
    if k < 2:
        return None
    return int(origins[k - 2]), float(prices[k - 2]), int(origins[k - 1]), float(prices[k - 1])


def lec_wedge_triangle(b: pd.DataFrame, sw: Dict[str, np.ndarray]) -> Dict[str, List[int]]:
    """가장 최근 확정 고점 두 개(위 선)·확정 저점 두 개(아래 선), 네 점 모두 250거래일 안. 선은 로그 축.
    E13 분출(Q54): 고점·저점 모두 높아짐 + 위 선 기울기 < 아래 선 기울기(좁아지는 상승 쐐기) + 장대양봉 종가가 위 선 구간 상단을 처음 넘음.
    E15 삼각수렴 상향 이탈(Q60): 고점 낮아짐 + 저점 높아지거나 같음(뒤 저점 ≥ 앞 저점 구간 하단) + 종가가 위 선 구간 상단을 처음 넘음."""
    c = LECTURE_CONFIG
    cl, atr = _lec_arr(b, "Close"), _lec_arr(b, "ATR14")
    big = b["BIG_BULL"].to_numpy(dtype=bool)
    n = len(cl)
    e13, e15 = [], []
    for t in range(1, n):
        hh = _lec_last_two(sw["h_o"], sw["h_p"], t)
        ll = _lec_last_two(sw["l_o"], sw["l_p"], t)
        if hh is None or ll is None:
            continue
        if min(hh[0], ll[0]) < t - c["DBL_MAX_GAP"]:
            continue
        fu, su = _lec_logline(*hh)
        zu_t, zu_p = lec_zone(fu(t), atr[t])[1], lec_zone(fu(t - 1), atr[t - 1])[1]
        cross = cl[t] > zu_t and cl[t - 1] <= zu_p
        if not cross:
            continue
        _, sl = _lec_logline(*ll)
        if hh[3] > hh[1] and ll[3] > ll[1] and 0 < su < sl and big[t]:
            e13.append(t)
        if hh[3] < hh[1] and ll[3] >= lec_zone(ll[1], atr[ll[2]])[0]:
            e15.append(t)
    return {"E13": e13, "E15": e15}


def lec_channel_retests(b: pd.DataFrame, sw: Dict[str, np.ndarray]) -> List[int]:
    """상승 채널(Q45·Q51): 높아진 확정 저점 두 개로 아래 선(로그), 위 선 = 그 평행선 중 아래 선에서 가장 먼 확정 고점(첫 저점 이후)을 지나는 선.
    폭(위 ÷ 아래 − 1) ≥ 4%. 채널 돌파 = 종가가 위 선 구간 상단을 처음 넘은 날(그날 아래 선 구간 하단 위).
    E14 리테스트 성공(Q53 = Q23과 같은 정의): 돌파 뒤 20거래일 안 저가 ≤ 위 선 구간 상단 이고 종가 ≥ 구간 하단인 첫날
    (그 전에 종가 < 위 선 구간 하단이면 없음). 선은 돌파일에 정한 그대로 연장."""
    c = LECTURE_CONFIG
    cl, lo, atr = _lec_arr(b, "Close"), _lec_arr(b, "Low"), _lec_arr(b, "ATR14")
    ho, hp = sw["h_o"], sw["h_p"]
    n = len(cl)
    out = []
    t = 1
    while t < n:
        ll = _lec_last_two(sw["l_o"], sw["l_p"], t)
        if ll is None or not (ll[3] > ll[1]):
            t += 1
            continue
        fl, s = _lec_logline(*ll)
        k = _lec_nconf(ho, t)
        idx = np.flatnonzero(ho[:k] >= ll[0])
        if not len(idx):
            t += 1
            continue
        dist = np.log(hp[idx]) - np.array([np.log(fl(int(ho[i]))) for i in idx])
        w = float(np.max(dist))
        if np.exp(w) - 1.0 < c["CHANNEL_MIN_WIDTH"]:
            t += 1
            continue
        fu = lambda x, fl=fl, w=w: fl(x) * float(np.exp(w))
        zu_t, zu_p = lec_zone(fu(t), atr[t])[1], lec_zone(fu(t - 1), atr[t - 1])[1]
        if not (cl[t] > zu_t and cl[t - 1] <= zu_p and cl[t] >= lec_zone(fl(t), atr[t])[0]):
            t += 1
            continue
        got = None
        for d in range(t + 1, min(t + c["RETEST_DAYS"], n - 1) + 1):
            zl_d, zu_d = lec_zone(fu(d), atr[d])
            if cl[d] < zl_d:
                break
            if lo[d] <= zu_d and cl[d] >= zl_d:
                got = d
                break
        if got is not None:
            out.append(got)
            t = got + 1
        else:
            t += 1
    return out


# ---------------------------------------------------------------------
# 겹침 지지 (C03, 정리본 6-1, Q35~Q39) — L5 V01·V02에서도 쓴다
# ---------------------------------------------------------------------
def _lec_line_alive(lv: float, start: int, t: int, hi: np.ndarray, lo: np.ndarray, cl: np.ndarray, atr: np.ndarray) -> bool:
    """선(Q35): 생긴 뒤 250거래일 안에 반응(Q36)이 한 번도 없으면 삭제, 반응이 한 번이라도 있었으면 계속 쓴다.
    반응 = 구간에 닿은 날 d(저가 ≤ 상단, 고가 ≥ 하단) 뒤 10거래일 안에 다가온 반대 방향으로 종가가 구간 밖 1×ATR 넘게 움직임.
    t일까지 끝난 반응만 센다."""
    c = LECTURE_CONFIG
    if t - start <= c["LINE_IDLE_DAYS"]:
        return True
    end = min(start + c["LINE_IDLE_DAYS"], t)
    for d in range(start + 1, end + 1):
        hw = _lec_hw(lv, atr[d])
        if not (lo[d] <= lv + hw and hi[d] >= lv - hw):
            continue
        prev = cl[d - 1]
        lim = min(d + c["REACT_DAYS"], t)
        seg = cl[d + 1:lim + 1]
        if not len(seg):
            continue
        if prev > lv + hw and np.any(seg > lv + hw + c["REACT_ATR"] * atr[d]):
            return True
        if prev < lv - hw and np.any(seg < lv - hw - c["REACT_ATR"] * atr[d]):
            return True
    return False


def lec_support_types(b: pd.DataFrame, sw: Dict[str, np.ndarray], t: int, loc: float,
                      exclude_origins: Sequence[int] = (), cache: Optional[Dict[str, Any]] = None) -> Set[str]:
    """t일에 가격 loc가 놓인 자리의 지지 기준 종류(Q39 겹침 수 = 종류 개수). loc가 그 선의 구간(± max(1%, 0.5×ATR_t)) 안이면 겹침.
    swing  확정 스윙 고점·저점 전부(Q35, 6-1 기준 1) — 사건이 기댄 그 저점 자체는 뺀다(exclude_origins)
    vol    RVOL ≥ 2 캔들의 고가(6-1 기준 2, Q37)        wick  꼬리 ≥ 몸통 2배 + RVOL ≥ 2 캔들의 종가(기준 3, Q37)
    dwell  지난 250거래일 종가를 20칸으로 나눈 체류 일수 상위 20% 칸(기준 4, Q38)
    swing·vol·wick 선은 Q35 삭제 규칙을 따른다. vol·wick은 t일 전 캔들만."""
    c = LECTURE_CONFIG
    hi, lo, cl, o, atr, rv = (_lec_arr(b, k) for k in ("High", "Low", "Close", "Open", "ATR14", "RVOL"))
    if cache is None:
        cache = {}
    if "lines" not in cache:
        body = np.abs(cl - o)
        wick = np.maximum(hi - np.maximum(cl, o), np.minimum(cl, o) - lo)
        vd = np.flatnonzero(rv >= c["RVOL_CLIMAX"])
        wd = np.flatnonzero((rv >= c["RVOL_CLIMAX"]) & (wick >= c["WICK_BODY"] * body))
        cache["lines"] = {
            "swing": (np.r_[sw["h_o"], sw["l_o"]].astype(int), np.r_[sw["h_p"], sw["l_p"]], _LEC_LR),
            "vol": (vd, hi[vd], 0), "wick": (wd, cl[wd], 0)}
    hw = _lec_hw(loc, atr[t])
    got: Set[str] = set()
    excl = set(int(x) for x in exclude_origins)
    for typ, (orig, lv, lag) in cache["lines"].items():
        ok = (orig + lag <= t) if lag else (orig < t)
        sel = np.flatnonzero(ok & (np.abs(lv - loc) <= hw))
        for i in sel:
            if int(orig[i]) in excl:
                continue
            if _lec_line_alive(float(lv[i]), int(orig[i]) + lag, t, hi, lo, cl, atr):
                got.add(typ)
                break
    W = c["DWELL_DAYS"]
    if t + 1 >= W // 2:
        seg = cl[max(0, t - W + 1):t + 1]
        a, z = float(np.min(seg)), float(np.max(seg))
        if z > a:
            cnt, edges = np.histogram(seg, bins=c["VP_BINS"], range=(a, z))
            top = np.argsort(-cnt, kind="stable")[:max(1, int(round(c["VP_BINS"] * c["DWELL_TOP"])))]
            if any(edges[i] <= loc <= edges[i + 1] for i in top):
                got.add("dwell")
    return got


# ---------------------------------------------------------------------
# 종목 하나의 L2 사건 표
# ---------------------------------------------------------------------
def lec_l2_events(p: Dict[str, Any], etf_below: Optional[pd.Series], grade: Optional[pd.Series],
                  earn_dates: Optional[np.ndarray] = None, earn_cover: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """사건 하나에 한 줄: t, event, 환경 F01~F09·확인 C01~C07 열(1.0/0.0/NaN). 짝짓기 표에 없는 조건도 계산은 해 두지만 쓰지 않는다."""
    c = LECTURE_CONFIG
    b, sw = p["b"], p["sw"]
    st, upf, _ = lec_state(b, sw, return_flags=True)
    n = len(b)
    dates = b.index
    cl, o, vol, rv = _lec_arr(b, "Close"), _lec_arr(b, "Open"), _lec_arr(b, "Volume"), _lec_arr(b, "RVOL")
    rows: List[Dict[str, Any]] = []

    def add(t, ev, **kw):
        if 0 <= t < n:
            rows.append({"t": int(t), "event": ev, **kw})

    # 돌파 계열
    eb = etf_below.reindex(dates).to_numpy(dtype=float) if (etf_below is not None and len(etf_below)) else np.full(n, np.nan)
    for x in lec_breakouts(b, sw):
        t0 = x["t0"]
        base = {"t0": t0, "res_level": x["level"], "res_origin": x["h_origin"]}
        if np.isfinite(eb[t0]):
            if eb[t0] == 1.0:
                add(t0, "E01", **base)
        else:
            add(t0, "E01", missing=1.0, **base)       # 섹터 ETF 자료 없음 → 사건에서 빠짐(개수만 센다)
        if x["confirm"] is not None:
            add(x["confirm"], "E02", **base)
        if x["flip"] is not None:
            add(x["flip"], "E04", loc=float(_lec_arr(b, "Low")[x["flip"]]), loc_origin=-1, **base)
    for t in lec_bb_up(b):
        nh = _lec_nconf(sw["h_o"], t)
        add(t, "E03", t0=t, res_level=float(sw["h_p"][nh - 1]) if nh else np.nan, res_origin=int(sw["h_o"][nh - 1]) if nh else -1)
    # 바닥 계열
    for x in lec_double_bottoms(b, sw, st):
        kw = {"loc": x["l1"], "loc_origin": x["l1_origin"]}
        if x["cand"]:
            add(x["tc"], "E06", **kw)
        if x["confirm"] is not None:
            add(x["confirm"], "E05", **kw)
        if x["e07"] is not None:
            add(x["e07"], "E07", **kw)
        if x["e12"] is not None:
            add(x["e12"], "E12")
    for x in lec_ma_double_bottoms(b, st):
        add(x["t"], "E08", loc=x["loc"], loc_origin=x["loc_origin"])
    tr = lec_transition(b, sw, upf)
    for t in lec_fall_release(st, tr):
        nl = _lec_nconf(sw["l_o"], t)
        add(t, "E09", loc=float(sw["l_p"][nl - 1]) if nl else np.nan, loc_origin=int(sw["l_o"][nl - 1]) if nl else -1)
    # 추세선·패턴
    dl = lec_downline_attempts(b, sw)
    for t, k in dl["attempts"]:
        if k == 3:
            add(t, "E11")
        elif k >= 4:
            add(t, "E10")
    for t in dl["real"]:
        add(t, "E12")
    wt = lec_wedge_triangle(b, sw)
    for t in wt["E13"]:
        add(t, "E13")
    for t in wt["E15"]:
        add(t, "E15")
    for t in lec_channel_retests(b, sw):
        add(t, "E14")
    if not rows:
        return pd.DataFrame(columns=["t", "event"])
    ev = pd.DataFrame(rows).sort_values(["event", "t"]).drop_duplicates(["event", "t"]).reset_index(drop=True)
    if "missing" not in ev:
        ev["missing"] = np.nan
    T = ev["t"].to_numpy(dtype=int)
    ev["date"] = dates[T]

    # 환경
    g = grade.reindex(dates).to_numpy(dtype=object) if (grade is not None and len(grade)) else np.full(n, None, dtype=object)
    gt = g[T]
    has_g = np.array([x is not None and x == x for x in gt])
    ev["F01"] = np.where(has_g, [x == "🟢" for x in gt], np.nan)
    ev["F02"] = np.where(has_g, [x in ("🟢", "🟡") for x in gt], np.nan)
    ev["F03"] = np.where(has_g, [x in ("🔴", "⚫") for x in gt], np.nan)
    ev["F04"] = (st[T] == 1).astype(float)
    ev["F05"] = (st[T] == 0).astype(float)
    rsd = lec_rs_daily(p["rs"]).reindex(dates).to_numpy(dtype=float) if (p.get("rs") is not None and len(p["rs"])) else np.full(n, np.nan)
    r = rsd[T]
    ev["F06"] = np.where(np.isfinite(r), r == 2, np.nan)
    ev["F07"] = np.where(np.isfinite(r), np.isin(r, (2, -1)), np.nan)
    s5, s20, s60 = _lec_arr(b, "SMA5"), _lec_arr(b, "SMA20"), _lec_arr(b, "SMA60")
    g5, g20, g60, g120 = (_lec_arr(b, k) for k in ("SMA5_SLOPE", "SMA20_SLOPE", "SMA60_SLOPE", "SMA120_SLOPE"))
    ev["F08"] = (g120[T] > 0).astype(float)
    ev["F09"] = ((s5[T] > s20[T]) & (s20[T] > s60[T]) & (g5[T] > 0) & (g20[T] > 0) & (g60[T] > 0)).astype(float)

    # 확인
    ev["C01"] = ((cl[T] > o[T]) & (rv[T] >= c["RVOL_UP"])).astype(float)
    c02, c03, c04, c05 = (np.full(len(ev), np.nan) for _ in range(4))
    cache: Dict[str, Any] = {}
    lo_o, lo_p = sw["l_o"], sw["l_p"]
    atr = _lec_arr(b, "ATR14")
    hi = _lec_arr(b, "High")
    for i, row in ev.iterrows():
        t, e = int(row["t"]), row["event"]
        if e == "E02":
            t0 = int(row["t0"])
            seg = slice(t0 + 1, t + 1)
            bears = vol[seg][cl[seg] < o[seg]]
            c02[i] = float(vol[t0] > (bears.max() if len(bears) else 0.0))
        if e in ("E01", "E02", "E03") and np.isfinite(row.get("res_level", np.nan)):
            t0, oh = int(row["t0"]), int(row["res_origin"])
            zl = lec_zone(float(row["res_level"]), atr[t0])[0]
            nl = _lec_nconf(lo_o, t0)
            k = np.flatnonzero(lo_o[:nl] > oh)
            c04[i] = float(len(k) >= 2 and lo_p[k[-1]] > lo_p[k[-2]] and lo_p[k[-1]] < zl and lo_p[k[-2]] < zl)
        if e in ("E04", "E05", "E07", "E08", "E09") and np.isfinite(row.get("loc", np.nan)):
            c03[i] = float(len(lec_support_types(b, sw, t, float(row["loc"]), (int(row.get("loc_origin", -1)),), cache)) >= 3)
        if e == "E04":
            t0 = int(row["t0"])
            pk = t0 + int(np.argmax(hi[t0:t])) if t > t0 else t
            c05[i] = float(np.mean(rv[pk + 1:t + 1]) < 1.0) if t > pk else float(rv[t] < 1.0)
    ev["C02"], ev["C03"], ev["C04"], ev["C05"] = c02, c03, c04, c05
    ev["C06"] = (tr["cnt"][T] >= 3).astype(float)
    c07 = np.full(len(ev), np.nan)
    if earn_cover is not None and earn_dates is not None:
        ed = np.asarray(earn_dates, dtype="datetime64[ns]")
        for i, t in enumerate(T):
            if dates[t] < earn_cover or t < 1:
                continue
            a_ = np.searchsorted(ed, np.datetime64(dates[t - 1]), "left")
            z_ = np.searchsorted(ed, np.datetime64(dates[t]), "right")
            c07[i] = float(z_ > a_)
    ev["C07"] = c07
    return ev
