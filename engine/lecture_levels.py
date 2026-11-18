"""강의 조건 시험 — 2단계: L4 매도·약화 · L5 레벨 (2026-11-18 사전 등록, AGENTS.md 4장 '강의 조건 시험').

L4 매도·약화 (사전등록안 6장, 정리본 4·5·6·7·8번)
    사건 W01~W17. 정리본의 목적은 '팔아라'가 아니라 '주의·비중 축소'라서 수익이 아니라 위험으로 잰다(N5).
    결과   이후 20거래일 최대 낙폭 = (t+1 ~ t+20 최저가) ÷ t일 종가 − 1
    비교   같은 종목·같은 상태(그날 상태)의 탐색 기간 모든 날 평균 낙폭(무작위 날 20곳의 기대값과 같고 잡음만 적음).
           '추가 낙폭' = 비교 평균 − 사건 낙폭 (클수록 사건 뒤에 더 깊게 빠짐). 기대 방향 > 0.
    섞기   사건을 같은 종목·같은 상태의 무작위 날로 바꿔 가족(17개) 최고 t의 95% 분위 = 통과선. 비용 없음(위험 지표).
    참고   10거래일 초과수익, 60거래일 안 상태 '하락' 도달 비율(약화 ≠ 종료), 변동성 단위(÷ ATR%) 추가 낙폭, 짝 비교.
L5 레벨 (사전등록안 7장, 정리본 5·6·7·8번)
    LEVELS.md와 같은 방식: 사건마다 같은 종목에서 최근 5거래일 변화(ATR 단위)가 ±0.5 안인 다른 날 20곳에 '같은 거리'
    가짜 선을 긋고 같은 방식으로 깨짐(또는 반응·도달)을 센다. 값 = 정리본이 말하는 쪽으로 (실제 − 가짜 평균).
    섞기   사건마다 실제 1개와 가짜 20개 중 하나를 무작위로 '실제'로 바꿔(바꿔도 같은 분포라면) 가족 최고 t 분포를 만든다.
           V10(빈 공간 vs 매물대)은 같은 종목 안에서 두 묶음 이름표를 섞는다.
모든 사건은 그날 종가까지의 자료로 정한다. 결과(깨짐·낙폭)만 미래를 본다.
"""
import math
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .frame_backtest import compare_clustered
from .lecture_backtest import _lec_pct, lec_forward_returns, lec_market_of, lec_month_drop
from .lecture_events import (_lec_arr, _lec_hw, _lec_logline, _lec_nconf, lec_breakouts, lec_double_bottoms, lec_support_types)
from .lecture_family import (_lec_f, lec_cluster_t, lec_combo_stats, lec_judge_one, lec_market_means, lec_month_code,
                             lec_null_pool, lec_pbo, lec_reality_check, lec_select)
from .lecture_signals import LECTURE_CONFIG, lec_zone

_LEC_LR5 = LECTURE_CONFIG["SWING_LR"]

L4_EVENTS: Dict[str, str] = {
    "W01": "약화 ① 전고점 갱신 실패", "W02": "약화 ② 상승 구간 거래량 감소", "W03": "약화 ③ MACD 둔화(고점 높이는데)",
    "W04": "약화 ④ RSI 다이버전스", "W05": "약화 ⑤ ADX 하락", "W06": "약화 2개 이상(③·④는 하나로)",
    "W07": "쌍고점 확정(전저점 이탈)", "W08": "이평 쌍봉 공격(쌍고점 + 5일선 하락)", "W09": "이평 쌍봉 보수(5일선 쌍봉 + 20일선 하락)",
    "W10": "붕괴 3단계 완성(이탈 → 회복 실패 → 낮은 고점)", "W11": "지지 이탈만(1단계)",
    "W12": "상승 추세선 장대음봉 이탈", "W13": "상승 추세선 일반 이탈(장대음봉 아님)", "W14": "가짜 돌파 판정일",
    "W15": "윗꼬리 대량 확정(저가 이탈)", "W16": "윗꼬리 대량 취소(고가 돌파)", "W17": "음봉 + RVOL ≥ 2",
}
L4_REF_PAIRS = (("W10", "W11", "붕괴 3단계 − 지지 이탈만"), ("W12", "W13", "장대음봉 이탈 − 일반 이탈"),
                ("W15", "W16", "윗꼬리 대량 확정 − 취소"))

# L5 질문: (이름, 잣대, 기대 부호). 잣대 'below' = 지지 아래로 깨짐, 'above' = 저항 위로 뚫림, 'react' = 반응, 'target' = 목표 먼저 도달
L5_QUESTIONS: Dict[str, Tuple[str, str, int]] = {
    "V01": ("겹침 3종류 이상 수평 지지 — 덜 깨지나", "below", -1),
    "V02": ("반응 3회 이상 수평 지지 — 덜 깨지나", "below", -1),
    "V03": ("상승 채널 하단 — 덜 깨지나", "below", -1),
    "V04": ("채널 중단선 — 더 반응하나", "react", +1),
    "V05": ("쌍바닥 박스 0.5 자리 — 상승 중 더 막히나", "above", -1),
    "V06": ("쌍바닥 박스 0.75 자리 — 상승 중 더 막히나", "above", -1),
    "V07": ("골든크로스 기준선(120일선 상승) — 덜 깨지나", "below", -1),
    "V08": ("골든크로스 기준선(120일선 하락) — 덜 깨지나", "below", -1),
    "V09": ("데드크로스 기준선 — 저항으로 더 막히나", "above", -1),
    "V10": ("빈 공간 진입 vs 매물대 진입 — 10거래일 더 크게 움직이나", "move", +1),
    "V11": ("쌍바닥 1대1 목표 — 무효선보다 먼저 닿나", "target", +1),
}
L5_REF = {"SUPALL": "참고 — 모든 확정 스윙 지지 접근(겹침·반응 조건 없음)"}


def _lec_fwd_min(x: np.ndarray, h: int) -> np.ndarray:
    """t+1 ~ t+h 최솟값(모자라면 NaN)."""
    s = pd.Series(x)
    return s[::-1].rolling(h, min_periods=h).min()[::-1].shift(-1).to_numpy()


def _lec_period_mask(index: pd.DatetimeIndex) -> Dict[str, np.ndarray]:
    c = LECTURE_CONFIG
    d = index.values
    return {"explore": (d >= np.datetime64(pd.Timestamp(c["EXPLORE_START"]))) & (d <= np.datetime64(pd.Timestamp(c["EXPLORE_END"]))),
            "verify": d >= np.datetime64(pd.Timestamp(c["VERIFY_START"]))}


# =====================================================================
# L4 사건
# =====================================================================
def lec_weakness(b: pd.DataFrame, sw: Dict[str, np.ndarray]) -> List[Dict[str, Any]]:
    """약화 5개(Q24): 새 확정 고점 H1이 확정되는 날(origin + 5)마다 H1과 '그 앞 고점' H2를 비교.
    H2 = H1 바로 앞 확정 저점보다 앞선 가장 최근 확정 고점(사이에 저점이 있어야 고점 둘 — 같은 높이 이웃 봉은 한 고점).
    ① H1 ≤ H2 구간 상단  ② (H1 앞 저점 → H1) 평균 RVOL < (H2 앞 저점 → H2)  ③ H1 > H2 이고 MACD 히스토그램 H1 < H2
    ④ H1 > H2 이고 RSI14 H1 < H2  ⑤ ADX14 H1 < H2.  개수 = ① + ② + (③ 또는 ④) + ⑤."""
    ho, hp, lo_o = sw["h_o"], sw["h_p"], sw["l_o"]
    atr, rv, mh, rsi, adx = (_lec_arr(b, k) for k in ("ATR14", "RVOL", "MACD_HIST", "RSI14", "ADX14"))
    n = len(b)
    out = []
    for k in range(len(ho)):
        o1 = int(ho[k])
        tc = o1 + _LEC_LR5
        if tc >= n:
            break
        j1 = int(np.searchsorted(lo_o, o1, "left")) - 1
        if j1 < 0:
            continue
        k2 = int(np.searchsorted(ho, lo_o[j1], "left")) - 1
        if k2 < 0:
            continue
        o2 = int(ho[k2])
        H1, H2 = float(hp[k]), float(hp[k2])
        j2 = int(np.searchsorted(lo_o, o2, "left")) - 1
        w = {"W01": H1 <= lec_zone(H2, atr[o1])[1]}
        w["W02"] = bool(j2 >= 0 and np.nanmean(rv[int(lo_o[j1]) + 1:o1 + 1]) < np.nanmean(rv[int(lo_o[j2]) + 1:o2 + 1]))
        w["W03"] = bool(H1 > H2 and mh[o1] < mh[o2])
        w["W04"] = bool(H1 > H2 and rsi[o1] < rsi[o2])
        w["W05"] = bool(adx[o1] < adx[o2])
        cnt = int(w["W01"]) + int(w["W02"]) + int(w["W03"] or w["W04"]) + int(w["W05"])
        w["W06"] = cnt >= 2
        out.append({"t": tc, "setup": o2 + _LEC_LR5, **w})
    return out


def lec_double_tops(b: pd.DataFrame, sw: Dict[str, np.ndarray], st: np.ndarray) -> List[Dict[str, Any]]:
    """쌍고점(Q56, 쌍바닥의 대칭): 새 확정 고점 H1, 바로 앞 확정 고점 H2, 사이 가장 낮은 확정 저점 V.
    조건: H2 날 상태 상승, H1 ≤ H2 구간 상단, 간격 ≤ 250, 폭(높은 고점 − V) ÷ 높은 고점 ≥ 5%.
    W07 확정 = H1 확정일부터 종가 < V 구간 하단인 첫날(그 전에 종가 > H1 구간 상단이면 무효, 250거래일까지).
    W08 공격 = 확정일부터 5일선 기울기 < 0인 첫날(무효 전). ATR은 H1 확정일 값."""
    c = LECTURE_CONFIG
    cl, atr, g5 = _lec_arr(b, "Close"), _lec_arr(b, "ATR14"), _lec_arr(b, "SMA5_SLOPE")
    ho, hp, lo_o, lo_p = sw["h_o"], sw["h_p"], sw["l_o"], sw["l_p"]
    n = len(cl)
    out = []
    for k in range(1, len(ho)):
        o1, o2 = int(ho[k]), int(ho[k - 1])
        tc = o1 + _LEC_LR5
        if tc >= n:
            break
        if o1 - o2 > c["DBL_MAX_GAP"] or st[o2] != 1:
            continue
        a_ = np.searchsorted(lo_o, o2, "right")
        z_ = np.searchsorted(lo_o, o1, "left")
        if z_ <= a_:
            continue
        V = float(np.min(lo_p[a_:z_]))
        H1, H2 = float(hp[k]), float(hp[k - 1])
        if H1 > lec_zone(H2, atr[tc])[1]:
            continue
        top = max(H1, H2)
        if V >= min(H1, H2) or (top - V) / top < c["TGT_MIN_WIDTH"]:
            continue
        vzl = lec_zone(V, atr[tc])[0]
        h1zu = lec_zone(H1, atr[tc])[1]
        conf = w08 = None
        for d in range(tc, min(tc + c["DBL_MAX_GAP"], n - 1) + 1):
            if cl[d] > h1zu:
                break
            if cl[d] < vzl:
                conf = d
                break
        if conf is not None:
            for d in range(conf, min(tc + c["DBL_MAX_GAP"], n - 1) + 1):
                if cl[d] > h1zu:
                    break
                if g5[d] < 0:
                    w08 = d
                    break
        out.append({"confirm": conf, "w08": w08})
    return out


def lec_ma_double_tops(b: pd.DataFrame, st: np.ndarray) -> List[int]:
    """5일선 쌍봉(Q27·Q29 대칭): 5일선 스윙 고점 m2(앞)·m1(뒤), m1 ≤ m2 구간 상단, 간격 ≤ 250, m2 날 상태 상승.
    5일선이 두 고점 사이 최저값 아래로 내려가고 그날 5일선 기울기 < 0 → 완성. W09 = 완성일부터 20일선 기울기 < 0인 첫날.
    5일선이 m1 위로 가면 무효. m1 확정일 뒤 250거래일까지."""
    c = LECTURE_CONFIG
    m = _lec_arr(b, "SMA5")
    g5, g20, atr = _lec_arr(b, "SMA5_SLOPE"), _lec_arr(b, "SMA20_SLOPE"), _lec_arr(b, "ATR14")
    n = len(m)
    rm = pd.Series(m).rolling(2 * _LEC_LR5 + 1, center=True).max().to_numpy()
    highs = np.flatnonzero(np.isfinite(m) & (m == rm))
    out = []
    for j in range(1, len(highs)):
        i1, i2 = int(highs[j]), int(highs[j - 1])
        tc = i1 + _LEC_LR5
        if tc >= n:
            break
        if i1 - i2 > c["DBL_MAX_GAP"] or st[i2] != 1:
            continue
        m1, m2 = m[i1], m[i2]
        if m1 > lec_zone(m2, atr[i1])[1]:
            continue
        neck = float(np.nanmin(m[i2:i1 + 1]))
        done = None
        for d in range(tc, min(tc + c["DBL_MAX_GAP"], n - 1) + 1):
            if m[d] > m1:
                break
            if done is None and m[d] < neck and g5[d] < 0:
                done = d
            if done is not None and g20[d] < 0:
                out.append(d)
                break
    return out


def lec_collapse(b: pd.DataFrame, sw: Dict[str, np.ndarray]) -> List[Dict[str, Any]]:
    """붕괴(Q26): 지지 = 그날까지의 최근 확정 스윙 저점. 이탈 = 그 저점이 최근 저점이 된 뒤 종가가 구간 하단 아래로 처음 마감한 날 tb.
    회복 = tb 뒤 10거래일 안에 종가가 구간 하단 이상으로 2일 연속. 회복이 없으면 '회복 실패'(tb + 10에 확정).
    전환 = tb 뒤 처음 생긴 확정 고점(250거래일 안)이 tb 때의 최근 확정 고점보다 낮음. W10 날 = 그 고점 확정일과 tb + 10 중 늦은 날.
    '새 저가' 경로는 쓰지 않는다. 구간 ATR은 이탈일 값."""
    c = LECTURE_CONFIG
    cl, atr = _lec_arr(b, "Close"), _lec_arr(b, "ATR14")
    ho, hp, lo_o, lo_p = sw["h_o"], sw["h_p"], sw["l_o"], sw["l_p"]
    n = len(cl)
    R, S = c["BREAK_RECOVER_DAYS"], c["BREAK_RECOVER_STREAK"]
    out = []
    cur, crossed = -1, False
    for t in range(n):
        nl = _lec_nconf(lo_o, t)
        if nl == 0:
            continue
        if nl - 1 != cur:
            cur, crossed = nl - 1, False
        zl = lec_zone(float(lo_p[cur]), atr[t])[0]
        if crossed or not (cl[t] < zl):
            continue
        crossed = True
        rec, streak = False, 0
        for d in range(t + 1, min(t + R, n - 1) + 1):
            streak = streak + 1 if cl[d] >= zl else 0
            if streak >= S:
                rec = True
                break
        w10 = None
        if not rec and t + R < n:
            nh = _lec_nconf(ho, t)
            prev = float(hp[nh - 1]) if nh else np.nan
            k = int(np.searchsorted(ho, t, "right"))
            if k < len(ho) and np.isfinite(prev) and ho[k] - t <= c["DBL_MAX_GAP"] and hp[k] < prev:
                w10 = max(int(ho[k]) + _LEC_LR5, t + R)
                if w10 >= n:
                    w10 = None
        out.append({"tb": t, "w10": w10})
    return out


def lec_upline(b: pd.DataFrame, sw: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """상승 추세선(Q45·Q48): 가장 최근 확정 저점 두 개가 높아지면 로그 축 직선. 선 구간 = 선 값 ± max(1%, 0.5×ATR).
    이탈 = 종가 < 선 구간 하단(그 전날은 하단 이상) → 선 끝. 새 확정 저점이 선 구간 아래에 생겨도 선 끝(이탈 사건 아님).
    선이 없을 때 새 쌍이 생기면 다시 긋는다. 날마다 그날의 선 값·기울기·채널 폭(선에서 가장 먼 확정 고점, Q51)도 돌려준다."""
    c = LECTURE_CONFIG
    cl, atr = _lec_arr(b, "Close"), _lec_arr(b, "ATR14")
    ho, hp, lo_o, lo_p = sw["h_o"], sw["h_p"], sw["l_o"], sw["l_p"]
    n = len(cl)
    val = np.full(n, np.nan)
    slope = np.full(n, np.nan)
    width = np.full(n, np.nan)
    breaks: List[int] = []
    line = None
    used = 0
    for t in range(n):
        nl = _lec_nconf(lo_o, t)
        if line is not None and nl > line["nl"]:
            for k in range(line["nl"], nl):
                if lo_p[k] < lec_zone(line["f"](int(lo_o[k])), atr[int(lo_o[k])])[0]:
                    line = None
                    break
            if line is not None:
                line["nl"] = nl
        if line is None:
            if nl >= 2 and nl - 1 > used and lo_p[nl - 1] > lo_p[nl - 2]:
                oa, ob = int(lo_o[nl - 2]), int(lo_o[nl - 1])
                f, s = _lec_logline(oa, lo_p[nl - 2], ob, lo_p[nl - 1])
                used = nl - 1
                if cl[t] < lec_zone(f(t), atr[t])[0]:
                    continue
                line = {"f": f, "s": s, "oa": oa, "nl": nl}
            else:
                continue
        v = line["f"](t)
        if cl[t] < lec_zone(v, atr[t])[0]:
            breaks.append(t)
            line = None
            continue
        val[t], slope[t] = v, line["s"]
        nh = _lec_nconf(ho, t)
        idx = np.flatnonzero(ho[:nh] >= line["oa"])
        if len(idx):
            width[t] = float(np.max(np.log(hp[idx]) - np.log([line["f"](int(ho[i])) for i in idx])))
    return {"val": val, "slope": slope, "width": width, "breaks": breaks}


def lec_l4_events(b: pd.DataFrame, sw: Dict[str, np.ndarray], st: np.ndarray, upline: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """종목 하나의 L4 사건(t, event). '상승 상태에서 일어난' 조건 = 시작점의 상태가 상승:
    W01~W06 = H2(앞 고점) 확정일, W07·W08 = H2 날(Q56), W09 = 5일선 m2 날, W10·W11 = 지지 이탈일, W15·W16 = 윗꼬리 캔들 날,
    W12~W14·W17 = 사건 날. (약화를 H1 확정일 상태로 보면 ① '갱신 실패'는 Q2 정의상 거의 상승 상태일 수 없다 — 새 저점 뒤
    앞 고점을 못 넘었으니 — 그래서 고점끼리 비교하는 약화는 앞 고점 때의 상태를 본다.)"""
    c = LECTURE_CONFIG
    n = len(b)
    cl, o, hi, lo, rv = (_lec_arr(b, k) for k in ("Close", "Open", "High", "Low", "RVOL"))
    big_bear = b["BIG_BEAR"].to_numpy(dtype=bool)
    wick = b["UPPER_WICK_CLIMAX"].to_numpy(dtype=bool)
    rows: List[Tuple[int, str]] = []
    for x in lec_weakness(b, sw):
        if st[x["setup"]] == 1:
            rows += [(x["t"], k) for k in ("W01", "W02", "W03", "W04", "W05", "W06") if x[k]]
    for x in lec_double_tops(b, sw, st):
        if x["confirm"] is not None:
            rows.append((x["confirm"], "W07"))
        if x["w08"] is not None:
            rows.append((x["w08"], "W08"))
    rows += [(t, "W09") for t in lec_ma_double_tops(b, st)]
    for x in lec_collapse(b, sw):
        if st[x["tb"]] != 1:
            continue
        rows.append((x["tb"], "W11"))
        if x["w10"] is not None:
            rows.append((x["w10"], "W10"))
    ul = upline or lec_upline(b, sw)
    for t in ul["breaks"]:
        if st[t] == 1:
            rows.append((t, "W12" if big_bear[t] else "W13"))
    for x in lec_breakouts(b, sw):
        if x["fake"] is not None and st[x["fake"]] == 1:
            rows.append((x["fake"], "W14"))
    for w in np.flatnonzero(wick & (st == 1)):
        for d in range(w + 1, min(w + c["WICK_CONFIRM_DAYS"], n - 1) + 1):
            if cl[d] > hi[w]:
                rows.append((d, "W16"))
                break
            if cl[d] < lo[w]:
                rows.append((d, "W15"))
                break
    rows += [(int(t), "W17") for t in np.flatnonzero((cl < o) & (rv >= c["RVOL_CLIMAX"]) & (st == 1))]
    if not rows:
        return pd.DataFrame(columns=["t", "event"])
    return pd.DataFrame(rows, columns=["t", "event"]).drop_duplicates().sort_values(["event", "t"]).reset_index(drop=True)


def lec_l4_table(prepared: Dict[str, Dict[str, Any]], progress: Optional[Callable[[int, int], None]] = None) -> Dict[str, Any]:
    """L4 사건 표 + 결과. xdd = 같은 종목·같은 상태·같은 기간 평균 20일 낙폭 − 사건 낙폭(> 0이면 더 깊게 빠짐)."""
    c = LECTURE_CONFIG
    D = c["SELL_DD_DAYS"]
    H = c["HORIZON"]
    m10 = lec_market_means(prepared, H)
    evs, pools = [], {"explore": {}, "verify": {}}
    for i, (sym, p) in enumerate(prepared.items()):
        b, sw, st = p["b"], p["sw"], p["st"]
        n = len(b)
        mk = lec_market_of(sym)
        cl, lo, atrp = _lec_arr(b, "Close"), _lec_arr(b, "Low"), _lec_arr(b, "ATR_PCT")
        dd = _lec_fwd_min(lo, D) / cl - 1.0
        dda = dd / np.where(atrp > 0, atrp, np.nan)
        pm = _lec_period_mask(b.index)
        base, basea = {}, {}
        for per, msk in pm.items():
            for s in (1, 0, -1):
                sel = msk & (st == s) & np.isfinite(dd)
                if sel.sum() >= 20:
                    base[(per, s)] = float(np.mean(dd[sel]))
                    basea[(per, s)] = float(np.nanmean(dda[sel]))
                    pools[per][f"{sym}|{s}"] = pd.Series(base[(per, s)] - dd[sel], index=b.index[sel])
        ev = lec_l4_events(b, sw, st)
        if len(ev):
            T = ev["t"].to_numpy(dtype=int)
            per = np.where(pm["explore"][T], "explore", np.where(pm["verify"][T], "verify", ""))
            sv = st[T]
            bb = np.array([base.get((pp, s), np.nan) for pp, s in zip(per, sv)])
            ba = np.array([basea.get((pp, s), np.nan) for pp, s in zip(per, sv)])
            ev["date"] = b.index[T]
            ev["state"] = sv
            ev["pkey"] = [f"{sym}|{s}" for s in sv]
            ev["dd20"] = dd[T]
            ev["xdd"] = bb - dd[T]
            ev["xdda"] = ba - dda[T]
            fwd = lec_forward_returns(b, H).to_numpy()
            ev["x10"] = fwd[T] - m10[mk].reindex(b.index).to_numpy()[T]
            r60 = np.full(len(T), np.nan)
            for j, t in enumerate(T):
                if t + c["SELL_STATE_DAYS"] < n:
                    r60[j] = float(np.any(st[t + 1:t + c["SELL_STATE_DAYS"] + 1] == -1))
            ev["down60"] = r60
            ev["missing"] = np.nan
            evs.append(ev.assign(ticker=sym, market=mk))
        if progress:
            progress(i + 1, len(prepared))
    allev = pd.concat(evs, ignore_index=True) if evs else pd.DataFrame(columns=["t", "event", "date", "ticker", "market", "missing", "pkey", "xdd"])
    return {"events": allev, "pools": pools, "n_tickers": len(prepared)}


def run_lecture_l4(table: Dict[str, Any], progress: Optional[Callable[[str, int, int], None]] = None) -> Dict[str, Any]:
    c = LECTURE_CONFIG
    col = "xdd"
    ev_all = table["events"]
    dts = pd.to_datetime(ev_all["date"]) if len(ev_all) else pd.Series(dtype="datetime64[ns]")
    s0, s1, v0 = pd.Timestamp(c["EXPLORE_START"]), pd.Timestamp(c["EXPLORE_END"]), pd.Timestamp(c["VERIFY_START"])
    ex = ev_all[(dts >= s0) & (dts <= s1)].reset_index(drop=True)
    ve = ev_all[dts >= v0].reset_index(drop=True)
    combos = [{"event": e, "env": None, "conf": None, "id": e} for e in L4_EVENTS]
    stats, sel = {}, {}
    for cb in combos:
        r = lec_select(ex, cb, col)
        s = lec_combo_stats(ex, r["idx"], col, 0.0)
        sub = ex.iloc[r["idx"]]
        s.update({"event": cb["event"], "dd20": float(sub["dd20"].mean()) if len(sub) else None,
                  "x10": float(sub["x10"].mean()) if len(sub) else None,
                  "down60": float(sub["down60"].mean()) if len(sub) else None,
                  "xdda": lec_cluster_t(sub["xdda"].to_numpy(dtype=float), lec_month_code(sub["date"])) if len(sub) else {}})
        stats[cb["id"]] = s
        if s["n"] >= c["MIN_EVENTS"] and s["n_tickers"] >= c["MIN_TICKERS"]:
            sel[cb["id"]] = r["idx"]
    pool = lec_null_pool(table["pools"]["explore"], (s0, s1))
    rc = lec_reality_check(ex, sel, pool, c["RC_REPS"], c["SEED"] + 4, key_col="pkey",
                           progress=(lambda i, n: progress("섞기", i, n)) if progress else None)
    thr = rc["threshold"]
    pbo = lec_pbo(ex, sel, col, (s0, s1), c["PBO_SLICES"])
    cands = []
    for k, s in stats.items():
        s["null_mean"] = rc.get("null_mean", {}).get(k)
        s["judge"] = lec_judge_one(s, thr, "정리본이 말한 방향 아님") if (k in sel and thr is not None) else {
            "candidate": False, "reasons": ["가족에 못 들어감(사건 < 300 또는 종목 < 50)" if k not in sel else "통과선 없음"]}
        if s["judge"]["candidate"]:
            cands.append(k)
    cands.sort(key=lambda k: -stats[k]["t"])
    verify = {}
    for k in cands:
        r = lec_select(ve, {"event": k, "env": None, "conf": None}, col)
        s = lec_combo_stats(ve, r["idx"], col, 0.0)
        verify[k] = {"pass": bool(s["mean"] is not None and s["mean"] > 0 and s["t"] is not None and s["t"] >= c["VERIFY_Z"]),
                     "n": s["n"], "mean": s["mean"], "t": s["t"]}
    refs = {}
    for a, bq, lab in L4_REF_PAIRS:
        A = ex.iloc[lec_select(ex, {"event": a, "env": None, "conf": None}, col)["idx"]]
        B = ex.iloc[lec_select(ex, {"event": bq, "env": None, "conf": None}, col)["idx"]]
        refs[f"{a}-{bq}"] = {"label": lab, "cmp": compare_clustered(A, B, col)}
    return {"family": "L4", "n_questions": len(combos), "n_family": len(sel), "threshold": thr, "rc_max_t": rc["max_t"], "pbo": pbo,
            "stats": stats, "candidates": cands, "verify": verify, "refs": refs, "n_tickers_used": table["n_tickers"],
            "n_event_rows": int(len(ex)), "config_version": c["VERSION"]}


def render_lecture_l4_report(r: Dict[str, Any], generated_at: Optional[str] = None) -> str:
    c = LECTURE_CONFIG
    st = r["stats"]
    L = ["# 강의 조건 시험 L4 — 매도·약화 신호 (4·5·6·7·8번)", "",
         "**탐색 단계 결과 — 이것만으로는 증거가 아니다.** 검증(2014~)과 최종(새 종목, z ≥ 3)까지 통과해야 '통과'.", "",
         f"- 생성: {generated_at or '—'} · 설정 {r['config_version']} (사전 등록 {c['REGISTERED']}) · 종목 {r['n_tickers_used']}개 · 탐색 기간 사건 줄 {r['n_event_rows']:,}개",
         f"- 사건 {r['n_questions']}개 중 가족 경쟁(사건 ≥ {c['MIN_EVENTS']}, 종목 ≥ {c['MIN_TICKERS']}) **{r['n_family']}개**. 상승 상태에서 시작한 사건만.",
         f"- 결과 = 이후 {c['SELL_DD_DAYS']}거래일 최대 낙폭(최저가 ÷ 그날 종가 − 1). **추가 낙폭** = 같은 종목·같은 상태 평균 낙폭 − 사건 낙폭 (+이면 사건 뒤 더 깊게 빠짐). 비용 없음",
         f"- **통과선 t = {_lec_f(r['threshold'], 2, False)}** (같은 종목·같은 상태 무작위 날로 바꿔 {c['RC_REPS']}번 다시 잰 '가족 안 최고 t'의 {int(c['RC_Q'] * 100)}% 분위)",
         f"- 과적합 확률 PBO(참고) = {_lec_f(None if r['pbo']['pbo'] is None else r['pbo']['pbo'] * 100, 0, False)}% (사건 {r['pbo']['n_combos']}개)", "",
         "## 판정", ""]
    if not r["candidates"]:
        L += ["**탐색 후보 없음.** 검증 기간(2014~)은 열지 않았다.", ""]
    else:
        L += ["| 사건 | 사건 수 | 추가 낙폭 | t | 검증(2014~) |", "|---|---|---|---|---|"]
        for k in r["candidates"]:
            s, v = st[k], r["verify"].get(k, {})
            vt = "—" if not v else f"{'통과' if v['pass'] else '미통과'} ({_lec_pct(v['mean'])}, t {_lec_f(v['t'])}, {v['n']:,}건)" + (" → 최종 확인 대기" if v.get("pass") else "")
            L.append(f"| {k} {L4_EVENTS[k]} | {s['n']:,} | {_lec_pct(s['mean'])} | {_lec_f(s['t'])} | {vt} |")
        L.append("")
    L += ["## 사건별", "",
          "'변동성 단위'는 낙폭을 그날 ATR%로 나눠 같은 계산을 한 참고값이다(사건 날 변동성이 커서 생긴 낙폭인지 보려고). 판정은 위 '추가 낙폭'만.", "",
          "| 사건 | 사건 수 | 종목 | 20일 낙폭 | 추가 낙폭 | t | 변동성 단위 추가 낙폭(t) | 60일 안 하락 상태 | 10일 초과수익 | 미국 / 한국 | 판정 |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for k in L4_EVENTS:
        s = st[k]
        xa = s.get("xdda") or {}
        bm = s["by_market"]
        why = "후보" if s["judge"]["candidate"] else ", ".join(s["judge"]["reasons"])
        L.append(f"| {k} {L4_EVENTS[k]} | {s['n']:,} | {s['n_tickers']} | {_lec_pct(s['dd20'])} | {_lec_pct(s['mean'])} | {_lec_f(s['t'])} | "
                 f"{_lec_f(xa.get('mean'), 2)} ({_lec_f(xa.get('t'))}) | {'—' if s['down60'] is None else format(s['down60'] * 100, '.0f') + '%'} | "
                 f"{_lec_pct(s['x10'])} | {_lec_pct(bm['US']['net'])} / {_lec_pct(bm['KR']['net'])} | {why} |")
    L += ["", "## 참고 비교(판정 아님, 추가 낙폭 차이)", ""]
    for v in r["refs"].values():
        x = v["cmp"]
        a, b_ = x.get("actual", {}), x.get("baseline", {})
        L.append(f"- {v['label']}: {_lec_pct(a.get('mean'))} ({a.get('n', 0):,}건) vs {_lec_pct(b_.get('mean'))} ({b_.get('n', 0):,}건) · "
                 f"차이 {_lec_pct(x.get('diff'))} · z {_lec_f(x.get('z'))}")
    L += ["", "## 읽는 법", "",
          "- 후보가 아니면 여기서 끝낸다. 결과를 보고 정의·숫자를 바꿔 다시 보지 않는다.",
          "- 통과해도 '팔아라'가 아니라 '이후 낙폭이 평소보다 컸다'는 뜻이다(정리본도 '주의'로 읽는다).", ""]
    return "\n".join(L)


# =====================================================================
# L5 레벨
# =====================================================================
def lec_hit(cl: np.ndarray, atr: np.ndarray, t: int, a: float, s: float, direction: int, h: int) -> Optional[float]:
    """t+1 ~ t+h 종가가 선을 넘었는가(1/0). 선 = (t일 종가 + 방향 × a × ATR_t) × exp(s × 지난 날 수).
    direction −1 = 아래로 깨짐, +1 = 위로 넘음. 실제 선과 가짜 선을 같은 식으로 잰다. 자료가 모자라면 None."""
    n = len(cl)
    if t + h >= n or not (np.isfinite(atr[t]) and atr[t] > 0):
        return None
    base = cl[t] + direction * a * atr[t]
    k = np.arange(1, h + 1)
    line = base * np.exp(s * k)
    seg = cl[t + 1:t + h + 1]
    return float(np.any(direction * (seg - line) > 0))


def lec_target_first(hi: np.ndarray, cl: np.ndarray, t: int, up: float, dn: float, days: int) -> Optional[float]:
    """t+1부터 days거래일 안에 고가 ≥ 위 선이 종가 < 아래 선보다 먼저면 1, 아니면 0(둘 다 없으면 0). 같은 날 둘 다면 0(보수적)."""
    n = len(cl)
    if t + days >= n:
        return None
    for d in range(t + 1, t + days + 1):
        if cl[d] < dn:
            return 0.0
        if hi[d] >= up:
            return 1.0
    return 0.0


def _lec_line_reactions(lv: float, start: int, t: int, hi, lo, cl, atr) -> int:
    """선의 반응 횟수(Q36): 닿은 날 d 뒤 10거래일 안에 다가온 반대 방향으로 구간 밖 1×ATR 넘게. t일까지 끝난 반응만, 반응 뒤 다음 닿음부터 다시 셈."""
    c = LECTURE_CONFIG
    if t <= start + 1:
        return 0
    d_ = np.arange(start + 1, t + 1)
    hw = np.maximum(c["ZONE_PCT_MIN"] * lv, c["ZONE_ATR_MULT"] * np.where(np.isfinite(atr[d_]) & (atr[d_] > 0), atr[d_], lv * 0.02))
    touch = d_[(lo[d_] <= lv + hw) & (hi[d_] >= lv - hw)]
    cnt, free = 0, -1
    for d in touch:
        if d < free or d < 1:
            continue
        h = _lec_hw(lv, atr[d])
        prev = cl[d - 1]
        lim = min(d + c["REACT_DAYS"], t)
        seg = cl[d + 1:lim + 1]
        if not len(seg):
            continue
        if prev > lv + h:
            hit = np.flatnonzero(seg > lv + h + c["REACT_ATR"] * atr[d])
        elif prev < lv - h:
            hit = np.flatnonzero(seg < lv - h - c["REACT_ATR"] * atr[d])
        else:
            continue
        if len(hit):
            cnt += 1
            free = d + 1 + int(hit[0]) + 1
    return cnt


def lec_swing_lines(b: pd.DataFrame, sw: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """확정 스윙 고점·저점 전부를 수평선으로(Q35). start = 확정일. fr = 생긴 뒤 250거래일 안에 닿은 반응이 끝난 첫날(없으면 큰 수).
    t일에 살아 있음 = t − start ≤ 250 또는 fr ≤ t (lecture_events._lec_line_alive와 같은 뜻)."""
    c = LECTURE_CONFIG
    hi, lo, cl, atr = (_lec_arr(b, k) for k in ("High", "Low", "Close", "ATR14"))
    n = len(cl)
    orig = np.r_[sw["h_o"], sw["l_o"]].astype(int)
    lv = np.r_[sw["h_p"], sw["l_p"]].astype(float)
    start = orig + _LEC_LR5
    keep = start < n
    orig, lv, start = orig[keep], lv[keep], start[keep]
    fr = np.full(len(lv), 10 ** 9, dtype=np.int64)
    for i in range(len(lv)):
        s0 = int(start[i])
        end = min(s0 + c["LINE_IDLE_DAYS"], n - 1)
        if end <= s0:
            continue
        d_ = np.arange(s0 + 1, end + 1)
        hw = np.maximum(c["ZONE_PCT_MIN"] * lv[i], c["ZONE_ATR_MULT"] * atr[d_])
        touch = d_[(lo[d_] <= lv[i] + hw) & (hi[d_] >= lv[i] - hw)]
        for d in touch:
            h = _lec_hw(lv[i], atr[d])
            seg = cl[d + 1:min(d + c["REACT_DAYS"], n - 1) + 1]
            if not len(seg):
                continue
            if cl[d - 1] > lv[i] + h:
                hit = np.flatnonzero(seg > lv[i] + h + c["REACT_ATR"] * atr[d])
            elif cl[d - 1] < lv[i] - h:
                hit = np.flatnonzero(seg < lv[i] - h - c["REACT_ATR"] * atr[d])
            else:
                continue
            if len(hit):
                fr[i] = d + 1 + int(hit[0])
                break
    o = np.argsort(start, kind="stable")
    return {"orig": orig[o], "lv": lv[o], "start": start[o], "fr": fr[o]}


def lec_l5_events(b: pd.DataFrame, sw: Dict[str, np.ndarray], st: np.ndarray, upline: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """종목 하나의 L5 사건. 사건마다 잣대와 실제 선의 거리 a(ATR 단위)·기울기 s를 정해 둔다(가짜 선이 같은 값을 쓴다).
    지지 접근(V01·V02·V03·V07·V08·SUPALL) = 전날 종가 > 구간 상단, 그날 저가 ≤ 구간 상단, 종가 ≥ 구간 하단 → 깨짐 = 10거래일 안 종가 < 구간 하단.
    저항 접근(V05·V06·V09) = 전날 종가 < 구간 하단, 그날 고가 ≥ 구간 하단, 종가 ≤ 구간 상단 → 뚫림 = 10거래일 안 종가 > 구간 상단.
    중단선(V04) = 전날 종가가 구간 밖, 그날 고가·저가가 구간에 닿음 → 반응 = 다가온 반대 방향으로 구간 밖 1×ATR 넘게.
    구간 = 선 값 ± max(1%, 0.5×ATR_t)."""
    c = LECTURE_CONFIG
    H = c["REACT_DAYS"]
    cl, hi, lo, atr = (_lec_arr(b, k) for k in ("Close", "High", "Low", "ATR14"))
    s20, s60, g120 = _lec_arr(b, "SMA20"), _lec_arr(b, "SMA60"), _lec_arr(b, "SMA120_SLOPE")
    n = len(cl)
    out: List[Dict[str, Any]] = []

    def sup_touch(t, v):
        zl, zu = lec_zone(v, atr[t])
        zl0, zu0 = lec_zone(v, atr[t - 1])
        return cl[t - 1] > zu0 and lo[t] <= zu and cl[t] >= zl, zl, zu

    def res_touch(t, v):
        zl, zu = lec_zone(v, atr[t])
        zl0, zu0 = lec_zone(v, atr[t - 1])
        return cl[t - 1] < zl0 and hi[t] >= zl and cl[t] <= zu, zl, zu

    # --- 확정 스윙 수평 지지(V01·V02·SUPALL) ---
    L = lec_swing_lines(b, sw)
    cache: Dict[str, Any] = {}
    for t in range(1, n - 1):
        k = int(np.searchsorted(L["start"], t, "right"))
        if k == 0:
            continue
        lv, s0, fr = L["lv"][:k], L["start"][:k], L["fr"][:k]
        alive = (t - s0 <= c["LINE_IDLE_DAYS"]) | (fr <= t)
        hwp = np.maximum(c["ZONE_PCT_MIN"] * lv, c["ZONE_ATR_MULT"] * atr[t - 1])
        below = alive & (lv + hwp < cl[t - 1])
        if not below.any():
            continue
        i = int(np.flatnonzero(below)[np.argmax(lv[below])])
        v = float(lv[i])
        ok, zl, zu = sup_touch(t, v)
        if not ok:
            continue
        a = (cl[t] - zl) / atr[t]
        out.append({"t": t, "q": "SUPALL", "kind": "below", "a": a, "s": 0.0})
        if len(lec_support_types(b, sw, t, v, (), cache)) >= 3:
            out.append({"t": t, "q": "V01", "kind": "below", "a": a, "s": 0.0})
        if _lec_line_reactions(v, int(L["start"][i]), t, hi, lo, cl, atr) >= 3:
            out.append({"t": t, "q": "V02", "kind": "below", "a": a, "s": 0.0})
    # --- 상승 채널 하단·중단선(V03·V04) ---
    ul = upline or lec_upline(b, sw)
    for t in range(1, n - 1):
        v, s, w = ul["val"][t], ul["slope"][t], ul["width"][t]
        if not (np.isfinite(v) and np.isfinite(ul["val"][t - 1]) and np.isfinite(w)) or np.exp(w) - 1.0 < c["CHANNEL_MIN_WIDTH"]:
            continue
        ok, zl, zu = sup_touch(t, v)
        if ok:
            out.append({"t": t, "q": "V03", "kind": "below", "a": (cl[t] - zl) / atr[t], "s": float(s)})
        mid, mid0 = v * np.exp(w / 2.0), ul["val"][t - 1] * np.exp(w / 2.0)
        zl, zu = lec_zone(mid, atr[t])
        zl0, zu0 = lec_zone(mid0, atr[t - 1])
        if lo[t] <= zu and hi[t] >= zl:
            if cl[t - 1] > zu0:
                out.append({"t": t, "q": "V04", "kind": "react_up", "a": (zu + atr[t] - cl[t]) / atr[t], "s": float(s)})
            elif cl[t - 1] < zl0:
                out.append({"t": t, "q": "V04", "kind": "react_dn", "a": (cl[t] - zl + atr[t]) / atr[t], "s": float(s)})
    # --- 쌍바닥 박스 0.5·0.75(V05·V06), 1대1 목표(V11) ---
    for x in lec_double_bottoms(b, sw, st):
        cf = x["confirm"]
        if cf is None:
            continue
        N, w_ = x["neck"], x["neck"] - min(x["l1"], x["l2"])
        l1zl = lec_zone(x["l1"], atr[x["tc"]])[0]
        up = N + w_
        out.append({"t": cf, "q": "V11", "kind": "target", "u": up / cl[cf] - 1.0, "d": 1.0 - l1zl / cl[cf]})
        for q, frac in (("V05", 0.5), ("V06", 0.75)):
            lvl = N + frac * w_
            for d in range(cf + 1, min(x["tc"] + c["DBL_MAX_GAP"], n - 2) + 1):
                if cl[d] < l1zl:
                    break
                ok, zl, zu = res_touch(d, lvl)
                if ok:
                    out.append({"t": d, "q": q, "kind": "above", "a": (zu - cl[d]) / atr[d], "s": 0.0})
                    break
                if cl[d] > zu:
                    break
    # --- 크로스 기준선(V07·V08 골든 = 지지, V09 데드 = 저항, Q30·Q31) ---
    diff = s20 - s60
    crosses = [t for t in range(1, n) if np.isfinite(diff[t]) and np.isfinite(diff[t - 1]) and np.sign(diff[t]) != np.sign(diff[t - 1]) and diff[t] != 0]
    for j, cdy in enumerate(crosses):
        nxt = crosses[j + 1] if j + 1 < len(crosses) else n
        lvl = (s20[cdy] + s60[cdy]) / 2.0
        golden = diff[cdy] > 0
        if golden and g120[cdy] == 0:
            continue
        q = ("V07" if g120[cdy] > 0 else "V08") if golden else "V09"
        free = -1
        for t in range(cdy + 1, min(nxt, n - 1)):
            if t < free:
                continue
            ok, zl, zu = (sup_touch if golden else res_touch)(t, lvl)
            if ok:
                if golden:
                    out.append({"t": t, "q": q, "kind": "below", "a": (cl[t] - zl) / atr[t], "s": 0.0})
                else:
                    out.append({"t": t, "q": q, "kind": "above", "a": (zu - cl[t]) / atr[t], "s": 0.0})
                free = t + c["GAP"] + 1
    return [x for x in out if 1 <= x["t"] < n and (x["kind"] == "target" or (np.isfinite(x["a"]) and x["a"] >= 0))]


def lec_volume_profile_events(b: pd.DataFrame) -> List[Dict[str, Any]]:
    """V10(Q43): 지난 250거래일(t−250 ~ t−1) 종가를 20칸으로 나눠 칸별 거래량 합. 위 20%(4칸) = 매물대, 아래 20% = 빈 공간.
    사건 = 종가가 전날과 다른 칸으로 들어간 날(그 칸이 매물대 또는 빈 공간, 250일 범위 안). 결과 = |10거래일 뒤 종가 ÷ 그날 종가 − 1| ÷ 그날 ATR%."""
    c = LECTURE_CONFIG
    cl, vol, atrp = _lec_arr(b, "Close"), _lec_arr(b, "Volume"), _lec_arr(b, "ATR_PCT")
    n = len(cl)
    W, B = c["VP_DAYS"], c["VP_BINS"]
    top = max(1, int(round(B * c["VP_Q"])))
    out = []
    for t in range(W + 1, n):
        seg, vs = cl[t - W:t], vol[t - W:t]
        a, z = float(np.min(seg)), float(np.max(seg))
        if not (z > a) or not (a <= cl[t] <= z) or not (a <= cl[t - 1] <= z):
            continue
        edges = np.linspace(a, z, B + 1)
        prof = np.bincount(np.clip(np.searchsorted(edges, seg, "right") - 1, 0, B - 1), weights=vs, minlength=B)
        bt = min(int(np.searchsorted(edges, cl[t], "right")) - 1, B - 1)
        by = min(int(np.searchsorted(edges, cl[t - 1], "right")) - 1, B - 1)
        if bt == by:
            continue
        order = np.argsort(prof, kind="stable")
        grp = 1 if bt in order[:top] else (0 if bt in order[-top:] else None)
        if grp is None:
            continue
        mv = abs(cl[t + c["HORIZON"]] / cl[t] - 1.0) / atrp[t] if (t + c["HORIZON"] < n and atrp[t] > 0) else np.nan
        out.append({"t": t, "grp": grp, "move": mv})
    return out


def _lec_dedupe_rows(rows: List[Dict[str, Any]], key) -> List[Dict[str, Any]]:
    rows = sorted(rows, key=lambda r: (key(r), r["t"]))
    out, last = [], {}
    for r in rows:
        k = key(r)
        if r["t"] >= last.get(k, -10 ** 9):
            out.append(r)
            last[k] = r["t"] + LECTURE_CONFIG["GAP"] + 1
    return out


def lec_l5_table(prepared: Dict[str, Dict[str, Any]], progress: Optional[Callable[[int, int], None]] = None) -> Dict[str, Any]:
    """사건마다 실제 결과 1개 + 가짜 선 20개(같은 종목·같은 기간, 최근 5거래일 변화 ±0.5 ATR, 사건과 10거래일 넘게 떨어진 날).
    V11은 같은 종목 같은 기간의 무작위 날(하락 폭 맞춤 없음, Q61), 같은 % 거리. V10은 두 묶음 표."""
    c = LECTURE_CONFIG
    K = c["BASELINE_SAMPLES"]
    rows, vrows = [], []
    for i, (sym, p) in enumerate(prepared.items()):
        b, sw, st = p["b"], p["sw"], p["st"]
        n = len(b)
        mk = lec_market_of(sym)
        cl, hi, atr, drop = (_lec_arr(b, k) for k in ("Close", "High", "ATR14", "DROP5"))
        pm = _lec_period_mask(b.index)
        rng = np.random.default_rng(c["SEED"] + 13 * i)
        ul = lec_upline(b, sw)
        evs = _lec_dedupe_rows(lec_l5_events(b, sw, st, ul), key=lambda r: r["q"])
        for e in evs:
            t = e["t"]
            per = "explore" if pm["explore"][t] else ("verify" if pm["verify"][t] else None)
            if per is None:
                continue
            if e["kind"] == "target":
                days = c["TGT_MAX_DAYS"]
                real = lec_target_first(hi, cl, t, cl[t] * (1 + e["u"]), cl[t] * (1 - e["d"]), days)
                if real is None:
                    continue
                cand = np.flatnonzero(pm[per] & (np.arange(n) + days < n))
                cand = cand[np.abs(cand - t) > c["HORIZON"]]
                if not len(cand):
                    continue
                js = rng.choice(cand, size=K, replace=True)
                fakes = [lec_target_first(hi, cl, int(j), cl[j] * (1 + e["u"]), cl[j] * (1 - e["d"]), days) for j in js]
            else:
                direction = -1 if e["kind"] in ("below", "react_dn") else +1
                real = lec_hit(cl, atr, t, e["a"], e["s"], direction, c["REACT_DAYS"])
                if real is None or not np.isfinite(drop[t]):
                    continue
                cand = np.flatnonzero(pm[per] & np.isfinite(drop) & (np.abs(drop - drop[t]) <= c["DROP_TOL"]) &
                                      (np.arange(n) + c["REACT_DAYS"] < n))
                cand = cand[np.abs(cand - t) > c["HORIZON"]]
                if not len(cand):
                    continue
                js = rng.choice(cand, size=K, replace=True)
                fakes = [lec_hit(cl, atr, int(j), e["a"], e["s"], direction, c["REACT_DAYS"]) for j in js]
            fakes = [f for f in fakes if f is not None]
            if len(fakes) < K:
                continue
            rows.append({"ticker": sym, "market": mk, "t": t, "date": b.index[t], "period": per, "q": e["q"], "kind": e["kind"],
                         "real": real, **{f"f{k}": fakes[k] for k in range(K)}})
        for e in _lec_dedupe_rows(lec_volume_profile_events(b), key=lambda r: r["grp"]):
            t = e["t"]
            per = "explore" if pm["explore"][t] else ("verify" if pm["verify"][t] else None)
            if per is None or not np.isfinite(e["move"]):
                continue
            vrows.append({"ticker": sym, "market": mk, "t": t, "date": b.index[t], "period": per, "grp": e["grp"], "move": e["move"]})
        if progress:
            progress(i + 1, len(prepared))
    return {"paired": pd.DataFrame(rows), "vp": pd.DataFrame(vrows), "n_tickers": len(prepared)}


def _lec_two_t(x: np.ndarray, g: np.ndarray, grp: np.ndarray) -> float:
    """두 묶음(grp 1·0) 평균 차이 ÷ √(월 묶음 SE² 합)."""
    a = lec_cluster_t(x[grp == 1], g[grp == 1])
    b_ = lec_cluster_t(x[grp == 0], g[grp == 0])
    if a["se"] is None or b_["se"] is None:
        return np.nan
    se = math.sqrt(a["se"] ** 2 + b_["se"] ** 2)
    return (a["mean"] - b_["mean"]) / se if se > 0 else np.nan


def _lec_l5_question(df: pd.DataFrame, q: str) -> Dict[str, Any]:
    """짝 질문 하나의 값: x = 기대 부호 × (실제 − 가짜 평균). 가짜·실제 비율도 함께."""
    K = LECTURE_CONFIG["BASELINE_SAMPLES"]
    sub = df[df["q"] == q] if len(df) else df
    if not len(sub):
        return {"n": 0}
    sign = L5_QUESTIONS[q][2] if q in L5_QUESTIONS else -1
    real = sub["real"].to_numpy(dtype=float)
    F = sub[[f"f{k}" for k in range(K)]].to_numpy(dtype=float)
    x = sign * (real - F.mean(axis=1))
    g = lec_month_code(sub["date"])
    r = lec_cluster_t(x, g)
    out = {"n": r["n"], "mean": r["mean"], "t": r["t"], "real_rate": float(real.mean()), "fake_rate": float(F.mean()),
           "n_tickers": int(sub["ticker"].nunique()), "by_market": {}, "sign": sign}
    for mk in ("US", "KR"):
        mm = sub["market"].to_numpy() == mk
        rr = lec_cluster_t(x[mm], g[mm])
        out["by_market"][mk] = {"n": rr["n"], "net": rr["mean"], "t": rr["t"]}
    s = pd.Series(x).groupby(g).sum()
    if len(s) >= 3:
        worst = s.idxmax()
        keep = g != worst
        rr = lec_cluster_t(x[keep], g[keep])
        yy, mm_ = divmod(int(worst), 12)
        out["month_drop"] = {"month": f"{yy}-{mm_ + 1:02d}", "net": rr["mean"], "t": rr["t"]}
    else:
        out["month_drop"] = {"month": None, "net": None, "t": None}
    out["net"] = out["mean"]
    out["_arr"] = {"real": real, "F": F, "g": g, "sign": sign}
    return out


def _lec_vp_question(vp: pd.DataFrame) -> Dict[str, Any]:
    if not len(vp):
        return {"n": 0}
    x, grp, g = vp["move"].to_numpy(dtype=float), vp["grp"].to_numpy(dtype=int), lec_month_code(vp["date"])
    a = vp[vp["grp"] == 1]
    b_ = vp[vp["grp"] == 0]
    cmp_ = compare_clustered(a.assign(ret=a["move"]), b_.assign(ret=b_["move"]), "ret")
    out = {"n": int(len(vp)), "n_empty": int(len(a)), "n_heavy": int(len(b_)), "mean": cmp_.get("diff"), "t": cmp_.get("z"),
           "empty_move": cmp_.get("actual", {}).get("mean"), "heavy_move": cmp_.get("baseline", {}).get("mean"),
           "n_tickers": int(vp["ticker"].nunique()), "by_market": {}}
    for mk in ("US", "KR"):
        aa, bb = a[a["market"] == mk], b_[b_["market"] == mk]
        cm = compare_clustered(aa.assign(ret=aa["move"]), bb.assign(ret=bb["move"]), "ret")
        out["by_market"][mk] = {"n": int(len(aa) + len(bb)), "net": cm.get("diff"), "t": cm.get("z")}
    md = lec_month_drop(a.assign(ret=a["move"]), b_.assign(ret=b_["move"]), "ret", +1)
    out["month_drop"] = {"month": md["month"], "net": md["diff"], "t": md["z"]}
    out["net"] = out["mean"]
    tick = pd.factorize(vp["ticker"])[0]
    out["_arr"] = {"x": x, "grp": grp, "g": g, "tick": tick}
    return out


def lec_l5_null(arrs: Dict[str, Dict[str, Any]], reps: int, seed: int, progress=None) -> Dict[str, Any]:
    """L5 섞기: 짝 질문은 사건마다 (실제 + 가짜 20) 중 하나를 무작위로 '실제'로, V10은 같은 종목 안에서 묶음 이름표를 섞어 t → 가족 최고 t."""
    rng = np.random.default_rng(seed)
    mx = []
    for rep in range(reps):
        best = -np.inf
        for q, a in arrs.items():
            if q == "V10":
                order_t = np.argsort(a["tick"], kind="stable")
                perm = np.lexsort((rng.random(len(a["tick"])), a["tick"]))
                newg = np.empty_like(a["grp"])
                newg[perm] = a["grp"][order_t]
                tt = _lec_two_t(a["x"], a["g"], newg)
            else:
                allv = np.column_stack([a["real"], a["F"]])
                kk = rng.integers(0, allv.shape[1], size=allv.shape[0])
                r = allv[np.arange(allv.shape[0]), kk]
                fm = (allv.sum(axis=1) - r) / (allv.shape[1] - 1)
                tt = lec_cluster_t(a["sign"] * (r - fm), a["g"])["t"]
            if tt is not None and np.isfinite(tt) and tt > best:
                best = tt
        mx.append(float(best) if np.isfinite(best) else np.nan)
        if progress:
            progress(rep + 1, reps)
    arr = np.array([v for v in mx if np.isfinite(v)])
    return {"max_t": mx, "threshold": float(np.quantile(arr, LECTURE_CONFIG["RC_Q"])) if len(arr) else None}


def run_lecture_l5(table: Dict[str, Any], progress: Optional[Callable[[str, int, int], None]] = None) -> Dict[str, Any]:
    c = LECTURE_CONFIG
    pr, vp = table["paired"], table["vp"]
    exp_p = pr[pr["period"] == "explore"] if len(pr) else pr
    ver_p = pr[pr["period"] == "verify"] if len(pr) else pr
    exp_v = vp[vp["period"] == "explore"] if len(vp) else vp
    ver_v = vp[vp["period"] == "verify"] if len(vp) else vp
    stats, arrs = {}, {}
    for q in L5_QUESTIONS:
        s = _lec_vp_question(exp_v) if q == "V10" else _lec_l5_question(exp_p, q)
        if not s.get("n"):
            s = {"n": 0, "mean": None, "t": None, "net": None, "n_tickers": 0, "by_market": {"US": {"n": 0, "net": None}, "KR": {"n": 0, "net": None}},
                 "month_drop": {"month": None, "net": None, "t": None}}
        arr = s.pop("_arr", None)
        big = (min(s.get("n_empty", 0), s.get("n_heavy", 0)) if q == "V10" else s["n"]) >= c["MIN_EVENTS"]
        if big and s["n_tickers"] >= c["MIN_TICKERS"] and arr is not None:
            arrs[q] = arr
        stats[q] = s
    ref = _lec_l5_question(exp_p, "SUPALL")
    ref.pop("_arr", None)
    nl = lec_l5_null(arrs, c["RC_REPS"], c["SEED"] + 5, progress=(lambda i, n: progress("섞기", i, n)) if progress else None)
    thr = nl["threshold"]
    cands = []
    for q, s in stats.items():
        if q in arrs and thr is not None:
            s["judge"] = lec_judge_one({**s, "n": (min(s["n_empty"], s["n_heavy"]) if q == "V10" else s["n"])}, thr, "정리본이 말한 방향 아님")
        else:
            s["judge"] = {"candidate": False, "reasons": ["가족에 못 들어감(사건 < 300 또는 종목 < 50)" if q not in arrs else "통과선 없음"]}
        if s["judge"]["candidate"]:
            cands.append(q)
    cands.sort(key=lambda q: -stats[q]["t"])
    verify = {}
    for q in cands:
        s = _lec_vp_question(ver_v) if q == "V10" else _lec_l5_question(ver_p, q)
        s.pop("_arr", None)
        verify[q] = {"pass": bool(s.get("mean") is not None and s["mean"] > 0 and s.get("t") is not None and s["t"] >= c["VERIFY_Z"]),
                     "n": s.get("n", 0), "mean": s.get("mean"), "t": s.get("t")}
    return {"family": "L5", "n_questions": len(L5_QUESTIONS), "n_family": len(arrs), "threshold": thr, "rc_max_t": nl["max_t"],
            "stats": stats, "candidates": cands, "verify": verify, "ref_all": ref, "n_tickers_used": table["n_tickers"],
            "config_version": c["VERSION"]}


def render_lecture_l5_report(r: Dict[str, Any], generated_at: Optional[str] = None) -> str:
    c = LECTURE_CONFIG
    st = r["stats"]
    word = {"below": "깨짐", "above": "위로 뚫림", "react": "반응", "target": "목표 먼저", "move": "움직임"}
    L = ["# 강의 조건 시험 L5 — 레벨(선이 버티나) (5·6·7·8번)", "",
         "**탐색 단계 결과 — 이것만으로는 증거가 아니다.** 검증(2014~)과 최종(새 종목, z ≥ 3)까지 통과해야 '통과'.", "",
         f"- 생성: {generated_at or '—'} · 설정 {r['config_version']} (사전 등록 {c['REGISTERED']}) · 종목 {r['n_tickers_used']}개",
         f"- 질문 {r['n_questions']}개 중 가족 경쟁(사건 ≥ {c['MIN_EVENTS']}, 종목 ≥ {c['MIN_TICKERS']}) **{r['n_family']}개**",
         "- 방식: LEVELS.md와 같음. 사건마다 같은 종목에서 최근 5거래일 변화가 비슷한(±0.5 ATR) 다른 날 20곳에 **같은 거리** 가짜 선을 긋고 10거래일 안 깨짐·뚫림·반응을 같은 식으로 센다. "
         "값 = 정리본이 말하는 쪽으로 (실제 − 가짜). V11은 같은 % 거리 목표·무효선(250거래일), V10은 빈 공간 − 매물대(움직임 ÷ ATR%)",
         f"- **통과선 t = {_lec_f(r['threshold'], 2, False)}** (사건마다 실제·가짜 중 하나를 무작위로 '실제'로 바꿔 {c['RC_REPS']}번 다시 잰 '가족 안 최고 t'의 {int(c['RC_Q'] * 100)}% 분위)",
         "- PBO는 내지 않는다: 질문마다 잣대(깨짐·뚫림·반응·도달·움직임)가 달라 '1등'을 같은 줄에 세울 수 없다.", "",
         "## 판정", ""]
    if not r["candidates"]:
        L += ["**탐색 후보 없음.** 검증 기간(2014~)은 열지 않았다.", ""]
    else:
        L += ["| 질문 | 사건 | 값 | t | 검증(2014~) |", "|---|---|---|---|---|"]
        for q in r["candidates"]:
            s, v = st[q], r["verify"].get(q, {})
            vt = "—" if not v else f"{'통과' if v['pass'] else '미통과'} (값 {_lec_f(None if v['mean'] is None else v['mean'] * 100, 1)}%p, t {_lec_f(v['t'])}, {v['n']:,}건)" + (" → 최종 확인 대기" if v.get("pass") else "")
            L.append(f"| {q} {L5_QUESTIONS[q][0]} | {s['n']:,} | {_lec_f(None if s['mean'] is None else s['mean'] * 100, 1)}%p | {_lec_f(s['t'])} | {vt} |")
        L.append("")
    L += ["## 질문별", "", "| 질문 | 사건 | 종목 | 실제 | 가짜 선 | 값(정리본 쪽 +) | t | 미국 / 한국 값 | 판정 |", "|---|---|---|---|---|---|---|---|---|"]
    for q, (lab, kind, _) in L5_QUESTIONS.items():
        s = st[q]
        bm = s["by_market"]
        why = "후보" if s.get("judge", {}).get("candidate") else ", ".join(s.get("judge", {}).get("reasons", []))
        if q == "V10":
            real = f"빈 공간 {_lec_f(s.get('empty_move'), 2, False)} ATR ({s.get('n_empty', 0):,})"
            fake = f"매물대 {_lec_f(s.get('heavy_move'), 2, False)} ATR ({s.get('n_heavy', 0):,})"
            val = _lec_f(s.get("mean"), 2)
            mk = f"{_lec_f(bm['US']['net'], 2)} / {_lec_f(bm['KR']['net'], 2)}"
        else:
            real = "—" if s.get("real_rate") is None else f"{word[kind]} {s['real_rate'] * 100:.1f}%"
            fake = "—" if s.get("fake_rate") is None else f"{s['fake_rate'] * 100:.1f}%"
            val = _lec_f(None if s.get("mean") is None else s["mean"] * 100, 1) + "%p"
            mk = " / ".join(_lec_f(None if bm[m]["net"] is None else bm[m]["net"] * 100, 1) + "%p" for m in ("US", "KR"))
        L.append(f"| {q} {lab} | {s['n']:,} | {s.get('n_tickers', 0)} | {real} | {fake} | {val} | {_lec_f(s.get('t'))} | {mk} | {why} |")
    ra = r["ref_all"]
    if ra.get("n"):
        L += ["", f"참고 — 모든 확정 스윙 지지 접근(겹침·반응 조건 없음): 깨짐 {ra['real_rate'] * 100:.1f}% vs 가짜 선 {ra['fake_rate'] * 100:.1f}% "
                  f"({ra['n']:,}건, 덜 깨진 정도 {ra['mean'] * 100:+.1f}%p, t {_lec_f(ra['t'])})"]
    L += ["", "## 읽는 법", "",
          "- 후보가 아니면 여기서 끝낸다. 결과를 보고 선의 정의·거리·기간을 바꿔 다시 보지 않는다.",
          "- '실제'가 높고 낮음보다 '가짜 선'과의 차이가 중요하다 — 같은 거리의 아무 가격도 그만큼 깨지거나 버틴다.", ""]
    return "\n".join(L)


def lec_l45_log_lines(r: Dict[str, Any], generated_at: str) -> List[str]:
    st = r["stats"]
    fam = [k for k in st if st[k].get("judge", {}).get("reasons", [""])[0:1] != ["가족에 못 들어감(사건 < 300 또는 종목 < 50)"]]
    best = max(fam, key=lambda k: st[k]["t"] if st[k].get("t") is not None else -9) if fam else None
    lab = L4_EVENTS if r["family"] == "L4" else {k: v[0] for k, v in L5_QUESTIONS.items()}
    what = "매도·약화 사건 17개" if r["family"] == "L4" else "레벨 질문 11개"
    rows = []
    if best:
        s = st[best]
        val = (f"{s['mean'] * 100:+.2f}%p(추가 낙폭)" if r["family"] == "L4" else
               (f"{s['mean']:+.2f} ATR" if best == "V10" else f"{s['mean'] * 100:+.1f}%p(정리본 쪽)")) if s.get("mean") is not None else "—"
        rows.append(f"| {generated_at} | `lec-{r['family']}` | {what}(가족 {r['n_family']}개) 중 최고 t `{best}` {lab[best]} · 통과선 {_lec_f(r['threshold'], 2, False)} "
                    f"(탐색 2001~2013) | {s['n']:,} | {val} | {_lec_f(s['t'])} | {'후보 ' + str(len(r['candidates'])) + '개' if r['candidates'] else '후보 없음'} |")
    for k in r["candidates"]:
        s = st[k]
        rows.append(f"| {generated_at} | `lec-{r['family']}-{k}` | {lab[k]} | {s['n']:,} | — | {_lec_f(s['t'])} | 후보 |")
    return rows
