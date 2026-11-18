"""강의 조건 시험 — 1단계: L1 BNF 전체 규칙 · L3 상태별 RSI (2026-11-18 사전 등록, AGENTS.md 4장 '강의 조건 시험').

실행: run_backtest.py --lecture data → --lecture L1 / --lecture L3 (결과 data/backtest/LECTURE_L1.md·LECTURE_L3.md).
L2(매수 사건 조합)·L4(매도·약화)·L5(레벨)는 다음 단계에서 만든다. L2는 사건이 모두 만들어진 뒤 가족 전체를 한 번에
계산해야 '섞기 200회 통과선'을 정할 수 있으므로 일부만 먼저 돌리지 않는다.

세 겹 분리 (사전등록안 8장)
    탐색  EXPLORE_CONFIG 종목 전부, 신호일 2001~2013. 후보 조건: 기대한 방향 + z ≥ 2(질문 1~2개인 가족) + 사건 ≥ 300 +
          종목 ≥ 50 + 미국·한국 같은 방향(둘 다 30건 이상일 때) + 기여가 가장 큰 달을 빼도 같은 방향·z ≥ 2
    검증  같은 종목, 신호일 2014~. 탐색 후보일 때만 계산해 보여 준다(후보가 아닌 질문의 뒤 기간은 열지 않는다). 같은 방향 + z ≥ 2
    최종  새 종목(다음 단계에서 규칙으로 고정한 목록), z ≥ 3. 검증까지 통과한 질문만, 전체 최대 5개
L1 BNF — 두 질문 (사전등록안 N6, 사용자 지적 반영)
    질문 1 반등   진입 뒤 5거래일 안에 종가 ≥ 진입가 + 0.5R 이 나왔는가. 비교 = 같은 종목에서 최근 5거래일 변화(ATR 단위)가
                  ±0.5 안인 다른 날 20곳에 다음 날 시가로 들어갔다고 보고 같은 % 거리를 같은 5거래일 안에 넘는 비율
    질문 2 수익   거래당 순수익(비용 0.30% 뒤) vs 같은 종목 무작위 날 20곳에서 같은 날수만큼 들고 판 수익(보유 기간 맞춤)
    앱 '매매 규칙 통과'는 두 질문 모두 최종 통과일 때만.
L3 상태별 RSI — RSI14가 30 아래(70 위)로 처음 들어간 날, 그날의 종목 상태로 나눠 10거래일 초과수익(같은 날 같은 시장 종목 평균을 뺌)을
    비교한다. L3-a: RSI ≤ 30에서 상승 > 하락, L3-b: RSI ≥ 70에서 상승 > 횡보. 두 집단 각각 사건 ≥ 300.
"""
import math
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .explore import EXPLORE_CONFIG
from .frame_backtest import compare_clustered
from .lecture_signals import (LECTURE_CONFIG, LEC_STATE_NAMES, lec_base, lec_bnf_base_series, lec_market_grade, lec_rs_line,
                              lec_rs_weekly_series, lec_state, lec_swings)
from .rule_backtest import _bt_naive
from .sec_facts import SEC_CONFIG, _sec_cached, _sec_http_json

LEC_BNF_VARIANTS: Dict[str, str] = {
    "full": "정리본 규칙 그대로(판정)",
    "no1": "참고 — 조건 1(급락 전 주도주) 뺌",
    "no2": "참고 — 조건 2(기준 이격도) 뺌",
    "no3": "참고 — 조건 3(거래량 폭증) 뺌",
    "no4": "참고 — 조건 4(홀드봉) 뺌",
    "no5": "참고 — 조건 5(확인봉) 뺌: 홀드봉 날 진입",
    "immediate": "참고 — 기준 도달 즉시 매수(조건 4·5 없이, 정리본 9-3 비교)",
    "code_base": "참고 — 기준 이격도를 코드식(750일 분포 하위 2.5%)으로",
}


def lec_market_of(sym: str) -> str:
    return "KR" if str(sym).upper().endswith((".KS", ".KQ")) else "US"


def lec_sector_etf(sym: str, sector: Optional[str]) -> Optional[str]:
    """종목 → 섹터 ETF(Q6). 섹터를 모르거나 짝이 없으면 None = '데이터 없음'."""
    if not sector:
        return None
    tab = LECTURE_CONFIG["SECTOR_ETF_KR"] if lec_market_of(sym) == "KR" else LECTURE_CONFIG["SECTOR_ETF_US"]
    return tab.get(str(sector))


def lec_index_of(sym: str, sector: Optional[str]) -> str:
    """시장 등급에 쓸 기준 지수(Q4): 미국 Technology → 나스닥, 나머지 → S&P 500, .KS → KOSPI, .KQ → KOSDAQ."""
    ix = LECTURE_CONFIG["INDEX"]
    s = str(sym).upper()
    if s.endswith(".KQ"):
        return ix["KQ"]
    if s.endswith(".KS"):
        return ix["KS"]
    return ix["US_TECH"] if str(sector) == "Technology" else ix["US"]


def lec_clean(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    d = _bt_naive(df)
    if d is None or not len(d):
        return None
    cols = [k for k in ("Open", "High", "Low", "Close") if k in d]
    d = d[(d[cols].astype(float) > 0).all(axis=1)]
    if "Volume" not in d:
        d = d.assign(Volume=0.0)
    return d if len(d) else None


def _lec_period_bounds(index: pd.DatetimeIndex, period: str) -> Tuple[int, int]:
    """신호일 위치 범위 [first, last] (포함). period = 'explore' | 'verify'."""
    c = LECTURE_CONFIG
    if period == "explore":
        s, e = pd.Timestamp(c["EXPLORE_START"]), pd.Timestamp(c["EXPLORE_END"])
    else:
        s, e = pd.Timestamp(c["VERIFY_START"]), pd.Timestamp("2262-01-01")
    first = int(np.searchsorted(index.values, np.datetime64(s), "left"))
    last = int(np.searchsorted(index.values, np.datetime64(e), "right")) - 1
    return first, last


# ---------------------------------------------------------------------
# L1 BNF — 거래 만들기
# ---------------------------------------------------------------------
def _lec_bnf_exits(o, h, l, cl, ma25, e: int, entry: float, stop: float) -> Optional[Dict[str, Any]]:
    """진입일 e(시가 entry)부터 청산. 같은 날 종가 조건은 손절 → 25일선 → 시간 순. 모든 청산은 다음 날 시가.
    손절: 종가 < 투매 저점 → 전부. 익절: 종가 ≥ 25일선 → 절반, 나머지는 이후 종가 < 25일선에서.
    시간 손절: 진입 뒤 5거래일(진입일 포함) 안에 종가 ≥ 진입가 + 0.5R 이 없고 익절도 없으면 6번째 날 시가에 전부.
    데이터가 끝날 때까지 안 끝나면 None(열린 거래 — 판정에서 뺀다)."""
    bn = LECTURE_CONFIG["BNF"]
    n = len(o)
    thr = entry + bn["TIME_STOP_R"] * (entry - stop)
    frac = bn["TAKE_FRAC"]
    half = None          # (실행 위치, 가격)
    rebound = False
    for d in range(e, n):
        if d + 1 >= n:
            return None
        if cl[d] < stop:
            px = o[d + 1]
            return _lec_bnf_result(entry, e, half, (d + 1, px), "손절" if half is None else "익절 뒤 손절", frac)
        if half is None and np.isfinite(ma25[d]) and cl[d] >= ma25[d]:
            half = (d + 1, o[d + 1])
        elif half is not None and np.isfinite(ma25[d]) and cl[d] < ma25[d]:
            return _lec_bnf_result(entry, e, half, (d + 1, o[d + 1]), "익절 + 25일선 이탈", frac)
        if cl[d] >= thr:
            rebound = True
        if (not rebound) and half is None and d == e + bn["TIME_STOP_DAYS"] - 1:
            return _lec_bnf_result(entry, e, None, (d + 1, o[d + 1]), "시간 손절", frac)
    return None


def _lec_bnf_result(entry, e, half, last, kind, frac) -> Optional[Dict[str, Any]]:
    j2, x2 = last
    if not (np.isfinite(x2) and x2 > 0):
        return None
    if half is None:
        gross = x2 / entry - 1.0
        return {"gross": gross, "h1": j2 - e, "h2": j2 - e, "frac1": 0.0, "exit_idx": j2, "exit_kind": kind}
    j1, x1 = half
    gross = frac * (x1 / entry) + (1 - frac) * (x2 / entry) - 1.0
    return {"gross": gross, "h1": j1 - e, "h2": j2 - e, "frac1": frac, "exit_idx": j2, "exit_kind": kind}


def lec_bnf_trades(b: pd.DataFrame, sw: Dict[str, np.ndarray], rsw: Optional[pd.Series],
                   earn: Optional[Dict[str, Any]] = None, variant: str = "full",
                   period: Optional[Tuple[int, int]] = None, base: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """한 종목의 BNF 거래(정리본 9번, Q62~Q69). 신호일 t = 조건 5(확인봉) 날, 진입 = t+1 시가. 종목당 1포지션.

    조건 1  급락 시작일 p = [t−40, t]에서 종가가 가장 높은 날. 그날의 주봉 RS(종목 ÷ 섹터 ETF, rsw = lec_rs_weekly_series)가 주도·약화
    조건 2  [t−5, t−1] 안의 어느 날 d2에 이격도(종가 ÷ 25일선 × 100) ≤ 그날의 기준 이격도(Q62)
    조건 3  [d2−5, min(d2+5, t−1)] 안에 음봉 + RVOL ≥ 2
    조건 4  m = [t−3, t−1]에서 저가가 가장 낮은 날(같으면 늦은 날), m 뒤 t−1까지 1~2개 봉이 모두 홀드봉(저가 ≥ 전날 저가)
    조건 5  t가 확인봉(양봉 + 종가 > 전날 고가)
    악재    p ~ t 사이 실적 발표일이 있으면 제외. 그 기간에 실적일 자료가 없으면 제외하지 못함('확인 불가'로 셈)
    손절    투매 저점 = [p, t] 최저가. 진입가가 그 아래면 패스, 손절폭 > 12%면 패스
    """
    c = LECTURE_CONFIG
    bn = c["BNF"]
    n = len(b)
    o, h, l, cl = (b[k].to_numpy(dtype=float) for k in ("Open", "High", "Low", "Close"))
    ma25 = b["SMA25"].to_numpy(dtype=float)
    disp = b["DISP25"].to_numpy(dtype=float)
    pctl = b["DISP_PCTL750"].to_numpy(dtype=float)
    rvol = b["RVOL"].to_numpy(dtype=float)
    hold = b["HOLD_BAR"].to_numpy(dtype=bool)
    conf = b["CONFIRM_BAR"].to_numpy(dtype=bool)
    climax = (cl < o) & (rvol >= c["RVOL_CLIMAX"])
    first, last = period if period is not None else (0, n - 1)
    first = max(first, bn["PEAK_LOOKBACK"] + 5)
    dates = b.index
    base_arr = base if base is not None else lec_bnf_base_series(b, sw)

    def base_ok(d: int) -> bool:
        if variant == "code_base":
            return bool(np.isfinite(pctl[d]) and pctl[d] <= bn["CODE_BASE_PCTL"])
        v = base_arr[d]
        return bool(np.isfinite(v) and np.isfinite(disp[d]) and disp[d] <= v)

    skips = {"rs_missing": 0, "rs_fail": 0, "earn_excluded": 0, "earn_unknown": 0, "gap_below_stop": 0, "stop_too_wide": 0,
             "open_at_end": 0}
    out: List[Dict[str, Any]] = []
    free = first
    earn_dates = None if not earn else earn.get("dates")
    cover_from = None if not earn else earn.get("cover_from")
    for t in range(first, min(last, n - 2) + 1):
        if t < free:
            continue
        # --- 조건 4·5 (변형마다 다름) ---
        if variant == "immediate":
            if not base_ok(t):
                continue
            d2, m = t, t
        else:
            if variant == "no5":
                if not hold[t]:
                    continue
                w0, w1, end_hold = t - 2, t - 1, t
            else:
                if not conf[t]:
                    continue
                w0, w1, end_hold = t - 3, t - 1, t - 1
            seg = l[w0:w1 + 1]
            m = w0 + int(len(seg) - 1 - np.argmin(seg[::-1]))     # 가장 낮은 날(같으면 늦은 날)
            k = end_hold - m
            if variant == "no4":
                if k < 0:
                    continue
            else:
                if not (bn["HOLD_MIN"] <= k <= bn["HOLD_MAX"]) or not bool(np.all(hold[m + 1:end_hold + 1])):
                    continue
            # --- 조건 2 ---
            if variant == "no2":
                d2 = m
            else:
                d2 = next((d for d in range(t - bn["COND2_WINDOW"], t) if d >= 0 and base_ok(d)), None)
                if d2 is None:
                    continue
        # --- 조건 3 ---
        if variant != "no3":
            hi = t if variant == "immediate" else min(d2 + bn["CLIMAX_WINDOW"], t - 1)
            if not bool(np.any(climax[max(0, d2 - bn["CLIMAX_WINDOW"]):hi + 1])):
                continue
        # --- 조건 1 ---
        p0 = max(0, t - bn["PEAK_LOOKBACK"])
        p = p0 + int(np.argmax(cl[p0:t + 1]))
        if variant != "no1":
            st = rsw.get(dates[p]) if (rsw is not None and len(rsw)) else None
            if st is None or not np.isfinite(st):
                skips["rs_missing"] += 1
                continue
            if st not in (2, 1):
                skips["rs_fail"] += 1
                continue
        # --- 악재 제외 ---
        earn_known = bool(cover_from is not None and dates[p] >= cover_from)
        if earn_known and earn_dates is not None and len(earn_dates):
            lo_i = np.searchsorted(earn_dates, np.datetime64(dates[p]), "left")
            hi_i = np.searchsorted(earn_dates, np.datetime64(dates[t]), "right")
            if hi_i > lo_i:
                skips["earn_excluded"] += 1
                continue
        if not earn_known:
            skips["earn_unknown"] += 1
        # --- 손절·진입 ---
        stop = float(np.min(l[p:t + 1]))
        e = t + 1
        entry = o[e]
        if not (np.isfinite(entry) and entry > stop):
            skips["gap_below_stop"] += 1
            continue
        stop_pct = (entry - stop) / entry
        if stop_pct > bn["MAX_STOP_PCT"]:
            skips["stop_too_wide"] += 1
            continue
        res = _lec_bnf_exits(o, h, l, cl, ma25, e, entry, stop)
        if res is None:
            skips["open_at_end"] += 1
            break
        thr = entry * (1.0 + bn["TIME_STOP_R"] * stop_pct)
        w = cl[e:min(n, e + bn["TIME_STOP_DAYS"])]
        reached25 = next((d - e for d in range(e, res["exit_idx"]) if np.isfinite(ma25[d]) and cl[d] >= ma25[d]), None)
        out.append({"date": dates[t], "t": t, "e": e, "peak": dates[p], "entry": entry, "stop": stop, "stop_pct": stop_pct,
                    "drop": float(b["DROP5"].iloc[t]), "reb": float(bool(np.any(w >= thr))),
                    "reb_size": float(np.nanmax(w) / entry - 1.0) if len(w) else np.nan,
                    "gross": res["gross"], "ret": res["gross"] - c["COST"], "h1": res["h1"], "h2": res["h2"],
                    "frac1": res["frac1"], "exit_kind": res["exit_kind"], "reach25_days": reached25,
                    "earn_known": earn_known})
        free = res["exit_idx"]
    return {"trades": pd.DataFrame(out), "skips": skips}


def lec_bnf_hold_baseline(b: pd.DataFrame, trades: pd.DataFrame, period: Tuple[int, int], seed: int) -> pd.DataFrame:
    """질문 2 비교: 거래마다 같은 종목·같은 기간의 무작위 날 j 20곳, j+1 시가 매수 → 같은 날수 뒤 시가에 같은 비율로 매도."""
    c = LECTURE_CONFIG
    if trades is None or not len(trades):
        return pd.DataFrame()
    o = b["Open"].to_numpy(dtype=float)
    n = len(o)
    rng = np.random.default_rng(seed)
    out = []
    for h1, h2, f1 in zip(trades["h1"].astype(int), trades["h2"].astype(int), trades["frac1"].astype(float)):
        lo_j, hi_j = period[0], min(period[1], n - 2 - h2)
        if hi_j < lo_j:
            continue
        for j in rng.integers(lo_j, hi_j + 1, size=c["BASELINE_SAMPLES"]):
            fill, x1, x2 = o[j + 1], o[j + 1 + h1], o[j + 1 + h2]
            if not (np.isfinite(fill) and fill > 0 and np.isfinite(x1) and np.isfinite(x2)):
                continue
            g = f1 * (x1 / fill) + (1 - f1) * (x2 / fill) - 1.0
            out.append({"date": b.index[j], "ret": g - c["COST"]})
    return pd.DataFrame(out)


def lec_bnf_rebound_baseline(b: pd.DataFrame, trades: pd.DataFrame, period: Tuple[int, int], seed: int) -> pd.DataFrame:
    """질문 1 비교: 거래마다 같은 종목·같은 기간에서 최근 5거래일 변화(ATR 단위)가 ±0.5 안인 다른 날 j 20곳.
    j+1 시가에 들어갔다고 보고, 그 거래와 같은 % 거리(0.5 × 손절폭%)를 j+1~j+5 종가가 넘는지."""
    c = LECTURE_CONFIG
    bn = c["BNF"]
    if trades is None or not len(trades):
        return pd.DataFrame()
    o, cl = b["Open"].to_numpy(dtype=float), b["Close"].to_numpy(dtype=float)
    drop = b["DROP5"].to_numpy(dtype=float)
    n = len(o)
    D = bn["TIME_STOP_DAYS"]
    hi_j = min(period[1], n - 1 - D)
    cand = np.arange(period[0], hi_j + 1)
    cand = cand[np.isfinite(drop[cand])] if len(cand) else cand
    if not len(cand):
        return pd.DataFrame()
    order = np.argsort(drop[cand])
    cs, ds = cand[order], drop[cand][order]
    rng = np.random.default_rng(seed + 1)
    out = []
    for t, dr, sp in zip(trades["t"].astype(int), trades["drop"].astype(float), trades["stop_pct"].astype(float)):
        if not np.isfinite(dr):
            continue
        pool = cs[np.searchsorted(ds, dr - c["DROP_TOL"], "left"):np.searchsorted(ds, dr + c["DROP_TOL"], "right")]
        pool = pool[np.abs(pool - t) > c["HORIZON"]]
        if not len(pool):
            continue
        for j in rng.choice(pool, size=c["BASELINE_SAMPLES"], replace=True):
            fill = o[j + 1]
            if not (np.isfinite(fill) and fill > 0):
                continue
            w = cl[j + 1:j + 1 + D]
            out.append({"date": b.index[j], "reb": float(bool(np.any(w >= fill * (1.0 + bn["TIME_STOP_R"] * sp))))})
    return pd.DataFrame(out)


# ---------------------------------------------------------------------
# 판정 도우미 (탐색·검증 공통)
# ---------------------------------------------------------------------
def lec_month_drop(a: pd.DataFrame, bb: pd.DataFrame, col: str, expect: int) -> Dict[str, Any]:
    """달 쏠림 점검(N3): 차이(a 평균 − b 평균)에 기대 방향으로 가장 크게 기여한 달 하나를 a·b 모두에서 빼고 다시 잰다."""
    if not len(a) or not len(bb):
        return {"month": None, "diff": None, "z": None}
    ma = pd.to_datetime(a["date"]).dt.to_period("M")
    mb = pd.to_datetime(bb["date"]).dt.to_period("M")
    xa, xb = a[col].astype(float), bb[col].astype(float)
    na, nb = int(xa.notna().sum()), int(xb.notna().sum())
    if na == 0 or nb == 0:
        return {"month": None, "diff": None, "z": None}
    # 차이 = Σ_달 (a의 그달 합 ÷ a 전체 수 − b의 그달 합 ÷ b 전체 수) — 그달 기여
    contrib = xa.groupby(ma).sum().div(na).sub(xb.groupby(mb).sum().div(nb), fill_value=0.0)
    if not len(contrib):
        return {"month": None, "diff": None, "z": None}
    worst = (contrib * (expect if expect else 1)).idxmax()
    r = compare_clustered(a[ma != worst], bb[mb != worst], col)
    return {"month": str(worst), "diff": r.get("diff"), "z": r.get("z")}


def lec_judge(cmp_all: Dict[str, Any], a: pd.DataFrame, bb: pd.DataFrame, col: str, expect: int,
              n_tickers: int, by_market: Dict[str, Dict[str, Any]], z_need: float) -> Dict[str, Any]:
    """탐색 후보 판정: 기대 방향 + z ≥ z_need + 사건 ≥ 300 + 종목 ≥ 50 + 미국·한국 같은 방향(둘 다 ≥ 30) + 달 쏠림."""
    c = LECTURE_CONFIG
    reasons = []
    n = int(cmp_all.get("actual", {}).get("n", 0))
    d, z = cmp_all.get("diff"), cmp_all.get("z")
    if n < c["MIN_EVENTS"]:
        reasons.append(f"사건 {n:,}건 < {c['MIN_EVENTS']}")
    if n_tickers < c["MIN_TICKERS"]:
        reasons.append(f"종목 {n_tickers}개 < {c['MIN_TICKERS']}")
    if d is None or z is None or np.sign(d) != expect or abs(z) < z_need:
        reasons.append(f"기대 방향 z ≥ {z_need:.0f} 아님")
    mk = {k: v for k, v in by_market.items() if v.get("actual", {}).get("n", 0) >= c["MIN_MARKET_EVENTS"] and v.get("diff") is not None}
    if len(mk) == 2 and len({np.sign(v["diff"]) for v in mk.values()}) > 1:
        reasons.append("미국·한국 방향이 다름")
    md = lec_month_drop(a, bb, col, expect)
    if md["diff"] is None or np.sign(md["diff"]) != expect or md["z"] is None or abs(md["z"]) < c["MONTH_DROP_Z"]:
        reasons.append(f"기여가 가장 큰 달({md['month']})을 빼면 기준 미달")
    return {"candidate": not reasons, "reasons": reasons, "month_drop": md,
            "market_check": "둘 다 30건 이상" if len(mk) == 2 else "한쪽이 30건 미만 — 비교 안 함"}


def lec_verify(cmp_v: Dict[str, Any], expect: int) -> Dict[str, Any]:
    d, z = cmp_v.get("diff"), cmp_v.get("z")
    ok = bool(d is not None and z is not None and np.sign(d) == expect and abs(z) >= LECTURE_CONFIG["VERIFY_Z"])
    return {"pass": ok, "diff": d, "z": z, "n": cmp_v.get("actual", {}).get("n", 0)}


# ---------------------------------------------------------------------
# L1 실행
# ---------------------------------------------------------------------
def _lec_bnf_universe(prepared: Dict[str, Dict[str, Any]], variant: str, period_name: str) -> Dict[str, pd.DataFrame]:
    parts: Dict[str, List[pd.DataFrame]] = {"tr": [], "hold": [], "reb": []}
    skips: Dict[str, int] = {}
    for i, (sym, p) in enumerate(prepared.items()):
        b = p["b"]
        per = _lec_period_bounds(b.index, period_name)
        if per[1] < per[0]:
            continue
        if "base" not in p:
            p["base"] = lec_bnf_base_series(b, p["sw"])
        if "rsw" not in p:
            p["rsw"] = lec_rs_weekly_series(p["rs"]) if p.get("rs") is not None else None
        r = lec_bnf_trades(b, p["sw"], p["rsw"], p["earn"], variant=variant, period=per, base=p["base"])
        for k, v in r["skips"].items():
            skips[k] = skips.get(k, 0) + v
        tr = r["trades"]
        if not len(tr):
            continue
        mk = lec_market_of(sym)
        seed = LECTURE_CONFIG["SEED"] + 7 * i
        parts["tr"].append(tr.assign(ticker=sym, market=mk))
        hb = lec_bnf_hold_baseline(b, tr, per, seed)
        rb = lec_bnf_rebound_baseline(b, tr, per, seed)
        if len(hb):
            parts["hold"].append(hb.assign(ticker=sym, market=mk))
        if len(rb):
            parts["reb"].append(rb.assign(ticker=sym, market=mk))
    cat = lambda lst: pd.concat(lst, ignore_index=True) if lst else pd.DataFrame(columns=["date", "ret", "reb", "ticker", "market"])
    return {"tr": cat(parts["tr"]), "hold": cat(parts["hold"]), "reb": cat(parts["reb"]), "skips": skips}


def _lec_q_block(u: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    tr = u["tr"]
    out = {"n_trades": int(len(tr)), "n_tickers": int(tr["ticker"].nunique()) if len(tr) else 0, "skips": u["skips"]}
    for q, base, col in (("q1", u["reb"], "reb"), ("q2", u["hold"], "ret")):
        out[q] = compare_clustered(tr, base, col)
        out[q + "_mk"] = {mk: compare_clustered(tr[tr["market"] == mk], base[base["market"] == mk], col) for mk in ("US", "KR")}
    if len(tr):
        out["exit_kinds"] = tr["exit_kind"].value_counts().to_dict()
        out["win_rate"] = float((tr["ret"] > 0).mean())
        out["mean_hold"] = float(tr["h2"].mean())
        r25 = tr["reach25_days"].dropna()
        out["reach25_share"] = float(len(r25) / len(tr))
        out["reach25_median_days"] = float(r25.median()) if len(r25) else None
        out["reb_size_q"] = [float(x) for x in tr["reb_size"].quantile([0.25, 0.5, 0.75]).to_numpy()]
        out["earn_known_share"] = float(tr["earn_known"].mean())
    return out


def run_lecture_l1(prepared: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """L1 BNF: 탐색 판정 → (후보일 때만) 검증. 참고 변형은 탐색 기간에서만 본다(판정 아님)."""
    c = LECTURE_CONFIG
    ex = _lec_bnf_universe(prepared, "full", "explore")
    blk = _lec_q_block(ex)
    judge = {}
    for q, base, col in (("q1", ex["reb"], "reb"), ("q2", ex["hold"], "ret")):
        judge[q] = lec_judge(blk[q], ex["tr"], base, col, +1, blk["n_tickers"], blk[q + "_mk"], c["SMALL_FAMILY_Z"])
    verify, vblk = {}, None
    if any(j["candidate"] for j in judge.values()):
        vu = _lec_bnf_universe(prepared, "full", "verify")
        vblk = _lec_q_block(vu)
        for q in ("q1", "q2"):
            verify[q] = lec_verify(vblk[q], +1) if judge[q]["candidate"] else None
    refs = {}
    for v in LEC_BNF_VARIANTS:
        if v == "full":
            continue
        u = _lec_bnf_universe(prepared, v, "explore")
        b2 = _lec_q_block(u)
        refs[v] = {"label": LEC_BNF_VARIANTS[v], "n_trades": b2["n_trades"], "q1": b2["q1"], "q2": b2["q2"],
                   "win_rate": b2.get("win_rate"), "mean_hold": b2.get("mean_hold")}
    return {"family": "L1", "explore": blk, "judge": judge, "verify": verify, "verify_block": vblk, "refs": refs,
            "n_tickers_used": len(prepared), "config_version": c["VERSION"]}


# ---------------------------------------------------------------------
# L3 상태별 RSI
# ---------------------------------------------------------------------
def lec_rsi_events(b: pd.DataFrame, st: np.ndarray) -> pd.DataFrame:
    """RSI14가 30 이하로 처음 들어간 날(kind='low', 전날 > 30)·70 이상으로 처음 들어간 날(kind='high', 전날 < 70).
    종류별로 10거래일 안의 새 사건은 세지 않는다. 결과 = 다음 날 시가 → h거래일 뒤 종가."""
    c = LECTURE_CONFIG
    rsi = b["RSI14"].to_numpy(dtype=float)
    o, cl = b["Open"].to_numpy(dtype=float), b["Close"].to_numpy(dtype=float)
    n = len(b)
    rows = []
    for kind, hit in (("low", (rsi <= c["RSI_LOW"]) & (np.roll(rsi, 1) > c["RSI_LOW"])),
                      ("high", (rsi >= c["RSI_HIGH"]) & (np.roll(rsi, 1) < c["RSI_HIGH"]))):
        hit[0] = False
        free = -1
        for t in np.flatnonzero(hit):
            if t < free:
                continue
            free = t + c["GAP"] + 1
            if t + 1 >= n:
                continue
            row = {"date": b.index[t], "t": int(t), "kind": kind, "state": int(st[t])}
            for hz in (c["HORIZON"],) + tuple(c["HORIZON_REF"]):
                row[f"r{hz}"] = float(cl[t + hz] / o[t + 1] - 1.0) if t + hz < n and o[t + 1] > 0 else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def lec_forward_returns(b: pd.DataFrame, h: int) -> pd.Series:
    """모든 날의 '다음 날 시가 → h거래일 뒤 종가' 수익률 (같은 날 시장 평균을 만들 때 쓴다)."""
    o, cl = b["Open"].to_numpy(dtype=float), b["Close"].to_numpy(dtype=float)
    n = len(b)
    r = np.full(n, np.nan)
    if n > h:
        with np.errstate(divide="ignore", invalid="ignore"):
            r[:n - h] = cl[h:] / o[1:n - h + 1] - 1.0
    return pd.Series(r, index=b.index)


def lec_rsi_event_table(prepared: Dict[str, Dict[str, Any]], grades: Optional[Dict[str, pd.Series]] = None) -> pd.DataFrame:
    """모든 종목의 RSI 사건 + 초과수익 x{h} = 결과 − 같은 날 같은 시장(미국·한국) 종목 전체의 같은 결과 평균."""
    c = LECTURE_CONFIG
    hz_all = (c["HORIZON"],) + tuple(c["HORIZON_REF"])
    evs, fwd = [], {mk: {hz: [] for hz in hz_all} for mk in ("US", "KR")}
    for sym, p in prepared.items():
        b = p["b"]
        mk = lec_market_of(sym)
        e = lec_rsi_events(b, p["st"])
        for hz in hz_all:
            fwd[mk][hz].append(lec_forward_returns(b, hz).rename(sym))
        if len(e):
            g = None
            if grades:
                gs = grades.get(lec_index_of(sym, p.get("sector")))
                if gs is not None and len(gs):
                    g = gs.reindex(e["date"]).to_numpy()
            evs.append(e.assign(ticker=sym, market=mk, grade=g if g is not None else None))
    ev = pd.concat(evs, ignore_index=True) if evs else pd.DataFrame(columns=["date", "kind", "state", "ticker", "market"])
    for mk in ("US", "KR"):
        for hz in hz_all:
            lst = fwd[mk][hz]
            if not lst or not len(ev):
                continue
            mean = pd.concat(lst, axis=1).mean(axis=1, skipna=True)
            sel = ev["market"] == mk
            ev.loc[sel, f"x{hz}"] = ev.loc[sel, f"r{hz}"].to_numpy() - mean.reindex(ev.loc[sel, "date"]).to_numpy()
    return ev


def run_lecture_l3(prepared: Dict[str, Dict[str, Any]], grades: Optional[Dict[str, pd.Series]] = None) -> Dict[str, Any]:
    c = LECTURE_CONFIG
    hz_all = (c["HORIZON"],) + tuple(c["HORIZON_REF"])
    ev = lec_rsi_event_table(prepared, grades)
    H = c["HORIZON"]
    col = f"x{H}"
    s0, s1, v0 = pd.Timestamp(c["EXPLORE_START"]), pd.Timestamp(c["EXPLORE_END"]), pd.Timestamp(c["VERIFY_START"])
    dts = pd.to_datetime(ev["date"]) if len(ev) else pd.Series(dtype="datetime64[ns]")
    exp_ev = ev[(dts >= s0) & (dts <= s1)] if len(ev) else ev
    ver_ev = ev[dts >= v0] if len(ev) else ev
    qs = {"L3-a": ("low", 1, -1, "RSI ≤ 30 사건: 상승 상태 − 하락 상태"),
          "L3-b": ("high", 1, 0, "RSI ≥ 70 사건: 상승 상태 − 횡보 상태")}
    out = {"family": "L3", "questions": {}, "config_version": c["VERSION"], "n_tickers_used": len(prepared)}
    for q, (kind, sa, sb, label) in qs.items():
        def grp(df, s):
            return df[(df["kind"] == kind) & (df["state"] == s)].dropna(subset=[col]) if len(df) else df
        A, B = grp(exp_ev, sa), grp(exp_ev, sb)
        cmp_all = compare_clustered(A, B, col)
        by_mk = {mk: compare_clustered(A[A["market"] == mk], B[B["market"] == mk], col) for mk in ("US", "KR")}
        n_t = int(pd.concat([A["ticker"], B["ticker"]]).nunique()) if len(A) or len(B) else 0
        jd = lec_judge(cmp_all, A, B, col, +1, n_t, by_mk, c["SMALL_FAMILY_Z"])
        nb = int(cmp_all.get("baseline", {}).get("n", 0))
        if nb < c["MIN_EVENTS"]:
            jd["candidate"] = False
            jd["reasons"].append(f"비교 집단 사건 {nb:,}건 < {c['MIN_EVENTS']}")
        ver = None
        if jd["candidate"]:
            ver = lec_verify(compare_clustered(grp(ver_ev, sa), grp(ver_ev, sb), col), +1)
        ref = {}
        for s in (1, 0, -1):
            g = grp(exp_ev, s)
            ref[LEC_STATE_NAMES[s]] = {f"x{hz}": (float(g[f"x{hz}"].mean()) if len(g) and f"x{hz}" in g else None) for hz in hz_all}
            ref[LEC_STATE_NAMES[s]]["n"] = int(len(g))
        by_grade = {}
        if len(exp_ev) and "grade" in exp_ev and exp_ev["grade"].notna().any():
            for gname in ("🟢", "🟡", "🔴", "⚫"):
                gg = exp_ev[(exp_ev["kind"] == kind) & (exp_ev["grade"] == gname)].dropna(subset=[col])
                by_grade[gname] = {"n": int(len(gg)), "mean": float(gg[col].mean()) if len(gg) else None}
        out["questions"][q] = {"label": label, "cmp": cmp_all, "by_market": by_mk, "n_tickers": n_t, "judge": jd,
                               "verify": ver, "by_state": ref, "by_grade": by_grade}
    return out


# ---------------------------------------------------------------------
# 준비 (가격 → 기초 지표·스윙·상태·RS·실적일)
# ---------------------------------------------------------------------
def lec_prepare(price_map: Dict[str, pd.DataFrame], sectors: Dict[str, Optional[str]],
                bench_map: Dict[str, pd.DataFrame], earn_table: Optional[pd.DataFrame],
                earn_cover: Optional[Dict[str, Optional[str]]] = None, need_rs: bool = True,
                progress: Optional[Callable[[int, int], None]] = None) -> Dict[str, Dict[str, Any]]:
    """종목마다 시험에 쓸 재료를 한 번 계산한다. 가격이 260일 미만이면 뺀다."""
    out = {}
    syms = list(price_map)
    ed = {}
    if earn_table is not None and len(earn_table):
        for tk, g in earn_table.groupby("ticker"):
            ed[tk] = np.sort(pd.to_datetime(g["date"]).values.astype("datetime64[ns]"))
    for i, sym in enumerate(syms):
        d = lec_clean(price_map[sym])
        if d is None or len(d) < EXPLORE_CONFIG["MIN_BARS"]:
            continue
        b = lec_base(d)
        sw = lec_swings(b)
        sec = sectors.get(sym) if sectors else None
        rs = None
        if need_rs:
            etf = lec_sector_etf(sym, sec)
            bdf = lec_clean(bench_map.get(etf)) if etf else None
            rs = lec_rs_line(b["Close"], bdf["Close"]) if bdf is not None else pd.Series(dtype=float)
        cov = (earn_cover or {}).get(sym)
        out[sym] = {"b": b, "sw": sw, "st": lec_state(b, sw), "rs": rs, "sector": sec,
                    "earn": {"dates": ed.get(sym, np.array([], dtype="datetime64[ns]")),
                             "cover_from": pd.Timestamp(cov) if cov else None}}
        if progress:
            progress(i + 1, len(syms))
    return out


# ---------------------------------------------------------------------
# 자료 받기 (run_backtest.py --lecture data). 받는 함수는 바꿔 끼울 수 있게(테스트는 가짜 함수)
# ---------------------------------------------------------------------
def lec_cache_path(name: str) -> str:
    return os.path.join(LECTURE_CONFIG["CACHE_DIR"], name)


def lec_load_csv_map(name: str, key: str, val: str) -> Dict[str, Any]:
    p = lec_cache_path(name)
    if not os.path.exists(p):
        return {}
    df = pd.read_csv(p, dtype=str, keep_default_na=False)
    return {r[key]: (r[val] if r[val] != "" else None) for _, r in df.iterrows()}


def lec_save_csv_map(name: str, key: str, val: str, data: Dict[str, Any]) -> str:
    p = lec_cache_path(name)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    pd.DataFrame({key: list(data), val: ["" if v is None else v for v in data.values()]}).to_csv(p, index=False, encoding="utf-8")
    return p


def lec_sec_earnings_dates(cik: int, ua: str, fetch: Optional[Callable[[str], Any]] = None) -> List[str]:
    """SEC 제출 목록에서 8-K Item 2.02(실적 발표) 제출일. 오래된 목록(추가 파일)까지 읽는다. 원자료는 data/sec_cache/."""
    get = fetch or (lambda u: _sec_http_json(u, ua))
    base = "https://data.sec.gov/submissions/"
    main = _sec_cached(os.path.join(SEC_CONFIG["CACHE_DIR"], f"submissions_{cik:010d}.json"),
                       lambda: get(f"{base}CIK{cik:010d}.json"))
    tables = [main.get("filings", {}).get("recent", {})]
    for f in main.get("filings", {}).get("files", []) or []:
        nm = f.get("name")
        if nm:
            tables.append(_sec_cached(os.path.join(SEC_CONFIG["CACHE_DIR"], nm), lambda nm=nm: get(base + nm), max_age_days=3650))
    dates = set()
    for tb in tables:
        forms, fdates, items = tb.get("form", []), tb.get("filingDate", []), tb.get("items", [])
        for k in range(min(len(forms), len(fdates))):
            it = items[k] if k < len(items) else ""
            if forms[k] == "8-K" and "2.02" in str(it):
                dates.add(str(fdates[k])[:10])
    return sorted(dates)


def lec_dart_earnings_dates(code6: str, dart: Any, end: Optional[str] = None) -> List[str]:
    """DART 공시 목록(거래소공시 I)에서 '잠정' + '실적'이 들어간 보고서 접수일."""
    try:
        df = dart.list(code6, start="1999-01-01", end=end or pd.Timestamp.today().strftime("%Y-%m-%d"), kind="I")
    except Exception:
        return []
    if df is None or not len(df) or "report_nm" not in df:
        return []
    m = df["report_nm"].astype(str).str.contains("잠정") & df["report_nm"].astype(str).str.contains("실적")
    col = "rcept_dt" if "rcept_dt" in df else None
    if col is None:
        return []
    return sorted({pd.Timestamp(str(x)).strftime("%Y-%m-%d") for x in df.loc[m, col]})


# ---------------------------------------------------------------------
# 보고서·기록
# ---------------------------------------------------------------------
def _lec_pct(v: Optional[float], d: int = 2) -> str:
    return "—" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v * 100:+.{d}f}%"


def _lec_cmp(x: Dict[str, Any], rate: bool = False) -> str:
    a, b_ = x.get("actual", {}), x.get("baseline", {})
    if not a.get("n"):
        return "표본 없음"
    f = (lambda v: f"{v * 100:.1f}%") if rate else (lambda v: f"{v * 100:+.2f}%")
    z = "—" if x.get("z") is None else f"{x['z']:+.1f}"
    d = "—" if x.get("diff") is None else (f"{x['diff'] * 100:+.1f}%p" if rate else f"{x['diff'] * 100:+.2f}%p")
    return f"{f(a['mean'])} ({a['n']:,}건) vs {f(b_['mean']) if b_.get('n') else '—'} · 차이 {d} · z {z}"


def _lec_verdict(j: Dict[str, Any], v: Optional[Dict[str, Any]]) -> str:
    if not j["candidate"]:
        return "탐색 후보 아님 (" + ", ".join(j["reasons"]) + ")"
    if v is None:
        return "탐색 후보"
    return f"탐색 후보 → 검증 {'통과' if v['pass'] else '미통과'} (2014~: {_lec_pct(v['diff'])}, z {('—' if v['z'] is None else format(v['z'], '+.1f'))}, {v['n']:,}건)" + \
        (" → 최종 확인(새 종목) 대기" if v["pass"] else "")


def render_lecture_l1_report(r: Dict[str, Any], generated_at: Optional[str] = None) -> str:
    c = LECTURE_CONFIG
    ex = r["explore"]
    L = ["# 강의 조건 시험 L1 — BNF 이격도 매매 (9번)", "",
         "**탐색 단계 결과 — 이것만으로는 증거가 아니다.** 이미 다른 시험에 쓴 종목의 2001~2013년이다. "
         "검증(2014~)과 최종(새 종목, z ≥ 3)까지 통과해야 '통과'로 쓴다.", "",
         f"- 생성: {generated_at or '—'} · 설정 {r['config_version']} (사전 등록 {c['REGISTERED']}) · 종목 {r['n_tickers_used']}개 · "
         f"거래 {ex['n_trades']:,}건 · 거래가 나온 종목 {ex['n_tickers']}개",
         "- 규칙: 조건 1~5 모두 + 실적일 제외 + 손절(투매 저점 아래 종가)·시간 손절(5거래일 안 +0.5R 없음)·25일선 절반 익절. 진입 = 확인봉 다음 날 시가, 비용 0.30%.",
         "", "## 판정", "",
         "| 질문 | 탐색(2001~2013) | 판정 |", "|---|---|---|",
         f"| 1. 5거래일 안 반등(종가 ≥ 진입가 + 0.5R) 비율 vs 하락 폭 맞춘 날의 같은 거리 | {_lec_cmp(ex['q1'], rate=True)} | {_lec_verdict(r['judge']['q1'], r['verify'].get('q1'))} |",
         f"| 2. 거래당 순수익 vs 보유 기간 맞춘 무작위 진입 | {_lec_cmp(ex['q2'])} | {_lec_verdict(r['judge']['q2'], r['verify'].get('q2'))} |",
         "", "앱에서 '매매 규칙 통과'로 쓰려면 두 질문 모두 최종 통과여야 한다. 하나만 통과하면 그대로 적는다(예: '반등은 잦지만 수익은 없음').", "",
         "## 시장별·점검", ""]
    for q in ("q1", "q2"):
        mk = ex[q + "_mk"]
        md = r["judge"][q]["month_drop"]
        L.append(f"- 질문 {q[1]}: 미국 {_lec_cmp(mk['US'], rate=q == 'q1')} / 한국 {_lec_cmp(mk['KR'], rate=q == 'q1')} · "
                 f"{r['judge'][q]['market_check']} · 기여가 가장 큰 달 {md['month']}을 빼면 차이 "
                 f"{'—' if md['diff'] is None else (format(md['diff'] * 100, '+.2f') + '%p')}, z {'—' if md['z'] is None else format(md['z'], '+.1f')}")
    sk = ex["skips"]
    L += ["", "## 거래 모습(참고)", "",
          f"- 청산 종류: {', '.join(f'{k} {v}' for k, v in (ex.get('exit_kinds') or {}).items()) or '—'} · 이긴 비율 "
          f"{'—' if ex.get('win_rate') is None else format(ex['win_rate'] * 100, '.0f') + '%'} · 평균 보유 {ex.get('mean_hold') or 0:.1f}일",
          f"- 25일선까지 돌아간 비율 {'—' if ex.get('reach25_share') is None else format(ex['reach25_share'] * 100, '.0f') + '%'}"
          f"(걸린 날 중앙 {ex.get('reach25_median_days') or '—'}일) · 5거래일 안 최고 종가 / 진입가 − 1 사분위 "
          f"{', '.join(_lec_pct(x, 1) for x in ex.get('reb_size_q', [])) or '—'}",
          f"- 빠진 신호: 주봉 RS 자료 없음 {sk.get('rs_missing', 0)} · 급락 전 주도·약화 아님 {sk.get('rs_fail', 0)} · 실적일 제외 {sk.get('earn_excluded', 0)} · "
          f"진입가가 손절선 아래 {sk.get('gap_below_stop', 0)} · 손절폭 > 12% {sk.get('stop_too_wide', 0)} · 열린 채 끝남 {sk.get('open_at_end', 0)}",
          f"- 실적일 자료로 악재 제외를 확인한 거래 비율 {'—' if ex.get('earn_known_share') is None else format(ex['earn_known_share'] * 100, '.0f') + '%'}"
          " (미국 2004-08 전·한국 2002-11 전·자료 없는 종목은 확인 불가)",
          "", "## 참고 변형(판정 아님, 탐색 기간)", "", "| 변형 | 거래 | 질문 1 반등 | 질문 2 수익 |", "|---|---|---|---|"]
    for k, v in r["refs"].items():
        L.append(f"| {v['label']} | {v['n_trades']:,} | {_lec_cmp(v['q1'], rate=True)} | {_lec_cmp(v['q2'])} |")
    L += ["", "## 읽는 법", "",
          "- 후보가 아니면 여기서 끝낸다. 결과를 보고 숫자(기준 이격도·5일·0.5R·12%)를 바꿔 다시 보지 않는다.",
          "- 참고 변형은 어느 조건이 효과를 내는지 보는 진단이다. 더 좋아 보이는 변형을 골라 판정으로 올리지 않는다.", ""]
    return "\n".join(L)


def render_lecture_l3_report(r: Dict[str, Any], generated_at: Optional[str] = None) -> str:
    c = LECTURE_CONFIG
    L = ["# 강의 조건 시험 L3 — 상태별 RSI 해석 (3번)", "",
         "**탐색 단계 결과 — 이것만으로는 증거가 아니다.** 검증(2014~)과 최종(새 종목)까지 통과해야 '통과'.", "",
         f"- 생성: {generated_at or '—'} · 설정 {r['config_version']} · 종목 {r['n_tickers_used']}개",
         f"- 사건: RSI14가 30 이하(70 이상)로 처음 들어간 날, 그날의 종목 상태(상승·횡보·하락)로 나눔. 결과 = 다음 날 시가 → "
         f"{c['HORIZON']}거래일 뒤 종가 − 같은 날 같은 시장 종목 평균(초과수익).", "",
         "| 질문 | 탐색(2001~2013) | 판정 |", "|---|---|---|"]
    for q, v in r["questions"].items():
        L.append(f"| {q} {v['label']} | {_lec_cmp(v['cmp'])} | {_lec_verdict(v['judge'], v['verify'])} |")
    L += ["", "## 상태별 평균 초과수익(참고)", ""]
    for q, v in r["questions"].items():
        L += [f"**{q}**", "", "| 상태 | 사건 | 5일 | 10일 | 20일 |", "|---|---|---|---|---|"]
        for s, x in v["by_state"].items():
            L.append(f"| {s} | {x['n']:,} | {_lec_pct(x.get('x5'))} | {_lec_pct(x.get('x10'))} | {_lec_pct(x.get('x20'))} |")
        if v["by_grade"]:
            L.append("")
            L.append("시장 등급별 10일 초과수익(참고): " + " · ".join(f"{g} {_lec_pct(x['mean'])} ({x['n']:,}건)" for g, x in v["by_grade"].items()))
        md = v["judge"]["month_drop"]
        L += ["", f"- 미국 {_lec_cmp(v['by_market']['US'])} / 한국 {_lec_cmp(v['by_market']['KR'])} · {v['judge']['market_check']} · "
                  f"기여가 가장 큰 달 {md['month']}을 빼면 z {'—' if md['z'] is None else format(md['z'], '+.1f')}", ""]
    L += ["## 읽는 법", "", "- 후보가 아니면 여기서 끝낸다. RSI 30·70이나 상태 정의를 바꿔 다시 보지 않는다.", ""]
    return "\n".join(L)


def lec_log_lines(r: Dict[str, Any], generated_at: str) -> List[str]:
    """EXPLORE_LOG.md 형식의 줄(본 횟수를 남긴다)."""
    rows = []
    if r["family"] == "L1":
        for q, lab, col in (("q1", "BNF 5거래일 안 반등 비율 vs 하락 폭 맞춘 날", "q1"), ("q2", "BNF 거래 수익 vs 보유 기간 맞춘 무작위", "q2")):
            x = r["explore"][q]
            d = "—" if x.get("diff") is None else f"{x['diff'] * 100:+.2f}%p"
            z = "—" if x.get("z") is None else f"{x['z']:+.1f}"
            rows.append(f"| {generated_at} | `lec-L1-{q}` | {lab} (강의 9번, 탐색 2001~2013) | {r['explore']['n_trades']:,} | {d} | {z} | "
                        f"{'후보' if r['judge'][q]['candidate'] else '후보 아님'} |")
    else:
        for q, v in r["questions"].items():
            x = v["cmp"]
            d = "—" if x.get("diff") is None else f"{x['diff'] * 100:+.2f}%p"
            z = "—" if x.get("z") is None else f"{x['z']:+.1f}"
            rows.append(f"| {generated_at} | `lec-{q}` | {v['label']} (강의 3번, 탐색 2001~2013) | {x.get('actual', {}).get('n', 0):,} | {d} | {z} | "
                        f"{'후보' if v['judge']['candidate'] else '후보 아님'} |")
    return rows


def lec_append_log(r: Dict[str, Any], generated_at: str, out_dir: str) -> str:
    p = os.path.join(out_dir, EXPLORE_CONFIG["LOG_FILE"])
    os.makedirs(out_dir, exist_ok=True)
    if not os.path.exists(p):
        with open(p, "w", encoding="utf-8") as f:
            f.write("# 탐색 기록\n\n| 본 때 | 이름 | 가설 | 사건 수 | 무작위 날 대비 차이 | z | 판정 |\n|---|---|---|---|---|---|---|\n")
    with open(p, "a", encoding="utf-8") as f:
        for line in lec_log_lines(r, generated_at):
            f.write(line + "\n")
    return p
