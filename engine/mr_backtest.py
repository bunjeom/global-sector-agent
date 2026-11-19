"""H-MR1 기술적 단기 반등 단타(평균 회귀) 시험 (2026-11-19 사전 등록, data/backtest/MR1_PREREG.md · AGENTS.md 4장).

사용자 안(강의 9번 BNF 중심 + 6번 지지·거래량 + 3번 과매도 + 1번 시장 필터)을 규칙 셋으로 비교한다.
    A  사용자 규칙 전체: BNF 조건 2~5 + 볼린저 하단 아래 종가 + 바닥이 지지 구간 안 + 지수 강한 −DI 날 제외
    B  BNF 핵심(조건 1 상대강도 뺌): BNF 조건 2~5
    C  단순 반전(대조군): 최근 5거래일 변화 ≤ −3 ATR이 처음 된 날
대상(판정): 신호일마다 같은 시장(미국·한국) 시험 종목 중 지난 60거래일 평균 거래대금 상위 절반 → 그중 지난 250거래일
변동성(일간 로그 수익률 표준편차) 상위 절반에 든 종목의 신호만(2026-11-19 사용자 결정, 결과 전). 전체 종목 결과는 참고.
공통: 진입 = 신호 다음 날 시가, 손절 = 급락 시작일~신호일 최저가(손절폭 ≤ 12%), 실적일 제외, 청산 = L1 BNF와 같은 코드
(25일선 절반 익절 · 5일 +0.5R 시간 손절), 비용 0.30%, 종목당 1포지션.
판정: 거래당 순수익 − 보유 기간 맞춘 같은 종목 무작위 진입(같은 대상 조건인 날에서) > 0, 월 묶음 z ≥ 2(탐색, 기존 545종목 2001~2019) →
후보만 새 151종목(2001~2026)에서 z ≥ 3 한 번. 숫자는 MR1_CONFIG 한 곳 — 결과를 보고 바꾸지 않는다.
"""
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .frame_backtest import compare_clustered
from .lecture_backtest import _lec_bnf_exits, _lec_pct, lec_clean, lec_index_of, lec_market_of, lec_month_drop
from .lecture_levels import lec_swing_lines
from .lecture_signals import LECTURE_CONFIG, lec_base, lec_bnf_base_series, lec_market_grade, lec_zone

MR1_CONFIG: Dict[str, Any] = {
    "VERSION": "mr1-2",                # mr1-1 → mr1-2(2026-11-19, 결과 전): 대상 = 거래대금 상위 절반 중 변동성 상위 절반
    "REGISTERED": "2026-11-19",
    "EXPLORE_START": "2001-01-01",     # 탐색: EXPLORE_CONFIG 545종목(이미 쓴 종목), 신호일 2001~2019
    "EXPLORE_END": "2019-12-31",
    "CONFIRM_START": "2001-01-01",     # 확인: LECTURE_CONFIG FINAL_UNIVERSE 151종목, 2001~끝, 후보만 한 번
    "EXPLORE_Z": 2.0,
    "CONFIRM_Z": 3.0,
    "MIN_TRADES": 300,
    "MIN_TICKERS": 50,
    "MIN_MARKET": 30,                  # 미국·한국 방향은 둘 다 이 이상일 때만 본다
    "MONTH_DROP_Z": 2.0,
    "COST": 0.003,
    "BB_WINDOW": 5,                    # A ⑤: [t−5, t−1] 안에 종가 < 볼린저 하단
    "MKT_ADX": 25.0,                   # A ⑦: 지수 ADX14 ≥ 25 이고 −DI > +DI인 날은 진입 안 함
    "C_DROP_ATR": -3.0,                # C: 최근 5거래일 변화 ÷ ATR14 ≤ −3.0 이 처음 된 날
    "PERIOD_BLOCKS": (("2001-01-01", "2007-12-31"), ("2008-01-01", "2013-12-31"), ("2014-01-01", "2019-12-31")),
    "CONFIRM_BLOCKS": (("2020-01-01", "2023-12-31"), ("2024-01-01", "2026-12-31")),
    "SEED": 1119,
    # --- 대상(판정) — 신호일 기준, 같은 시장·시험 종목 안 순위 ---
    "LIQ_DAYS": 60,                    # 평균 거래대금(종가 × 거래량) 기간, 그날 제외(급락일 대량 거래가 순위를 끌어올리지 않게)
    "LIQ_Q": 0.5,                      # 거래대금 상위 절반
    "VOL_DAYS": 250,                   # 변동성 = 지난 250거래일 일간 로그 수익률 표준편차(그날 포함, 250일이 다 차야 함)
    "VOL_Q": 0.5,                      # 거래대금 상위 절반 안에서 변동성 상위 절반
    "RANK_MIN_STOCKS": 10,             # 그날 그 시장에 순위를 낼 종목이 10개 미만이면 대상 없음
}
MR1_RULES: Dict[str, str] = {
    "A": "A 사용자 규칙 전체(BNF 2~5 + 볼린저 하단 + 지지 구간 + 시장 필터)",
    "B": "B BNF 핵심(조건 1 상대강도 뺌)",
    "C": "C 단순 반전 대조군(5일 −3 ATR)",
}


# ---------------------------------------------------------------------
# 지수 맥락 (A ⑦ 시장 필터, 참고용 시장 등급)
# ---------------------------------------------------------------------
def mr_index_context(bench_map: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, pd.Series]]:
    """지수마다: block = ADX14 ≥ 25 이고 −DI > +DI(그날 종가까지), grade = 시장 등급(1번 표, 참고)."""
    out = {}
    for sym in set(LECTURE_CONFIG["INDEX"].values()):
        d = lec_clean(bench_map.get(sym)) if bench_map else None
        if d is None or len(d) < 30:
            continue
        b = lec_base(d)
        blk = (b["ADX14"] >= MR1_CONFIG["MKT_ADX"]) & (b["MINUS_DI"] > b["PLUS_DI"])
        out[sym] = {"block": blk.astype(float), "grade": lec_market_grade(d)}
    return out


# ---------------------------------------------------------------------
# 대상(판정) — 거래대금 상위 절반 중 변동성 상위 절반 (신호일 기준, 같은 시장 안 순위)
# ---------------------------------------------------------------------
def mr_liq_vol(b: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """종목 하나의 날짜별 평균 거래대금(지난 60거래일, 그날 제외)과 변동성(지난 250거래일 일간 로그 수익률 표준편차, 그날 포함)."""
    c = MR1_CONFIG
    cl = b["Close"].astype(float)
    val = (cl * b["Volume"].astype(float)).shift(1).rolling(c["LIQ_DAYS"], min_periods=c["LIQ_DAYS"]).mean()
    vol = np.log(cl).diff().rolling(c["VOL_DAYS"], min_periods=c["VOL_DAYS"]).std()
    return val, vol


def mr_eligibility(prepared: Dict[str, Dict[str, Any]]) -> Dict[str, np.ndarray]:
    """날짜마다 같은 시장(미국·한국) 종목들 안에서: 평균 거래대금 순위 ≥ 상위 절반 → 그 안에서 변동성 순위 ≥ 상위 절반 = 대상.
    순위는 그날 두 값이 모두 있는 종목끼리(백분위, 같은 값은 평균 순위). 종목이 10개 미만인 날은 대상 없음.
    결과: 종목마다 그 종목 날짜에 맞춘 True/False 배열. 그날까지의 자료만 쓴다(종목 목록 자체는 시험 종목 그대로)."""
    c = MR1_CONFIG
    out: Dict[str, np.ndarray] = {}
    for mk in ("US", "KR"):
        syms = [s for s in prepared if lec_market_of(s) == mk]
        if not syms:
            continue
        lv = {s: mr_liq_vol(prepared[s]["b"]) for s in syms}
        L = pd.DataFrame({s: v[0] for s, v in lv.items()})
        V = pd.DataFrame({s: v[1] for s, v in lv.items()})
        ok = L.notna() & V.notna()
        L, V = L.where(ok), V.where(ok)
        cnt = ok.sum(axis=1)
        liq = L.rank(axis=1, pct=True) > (1.0 - c["LIQ_Q"])          # 상위 절반(백분위 > 0.5)
        Vl = V.where(liq)
        hv = Vl.rank(axis=1, pct=True) > (1.0 - c["VOL_Q"])
        E = (liq & hv) & (cnt >= c["RANK_MIN_STOCKS"]).to_numpy()[:, None]
        for s in syms:
            out[s] = E[s].reindex(prepared[s]["b"].index).fillna(False).to_numpy(dtype=bool)
    return out


def mr_hold_baseline(b: pd.DataFrame, trades: pd.DataFrame, period: Tuple[int, int], seed: int,
                     eligible: Optional[np.ndarray] = None) -> pd.DataFrame:
    """보유 기간 맞춘 무작위 진입(거래마다 20곳). eligible이 있으면 같은 종목이 대상 조건이었던 날에서만 뽑는다."""
    lc = LECTURE_CONFIG
    if trades is None or not len(trades):
        return pd.DataFrame()
    o = b["Open"].to_numpy(dtype=float)
    n = len(o)
    rng = np.random.default_rng(seed)
    pos = np.arange(period[0], period[1] + 1)
    if eligible is not None:
        pos = pos[eligible[pos]]
    out = []
    for h1, h2, f1 in zip(trades["h1"].astype(int), trades["h2"].astype(int), trades["frac1"].astype(float)):
        cand = pos[pos <= n - 2 - h2]
        if not len(cand):
            continue
        for j in rng.choice(cand, size=lc["BASELINE_SAMPLES"], replace=True):
            fill, x1, x2 = o[j + 1], o[j + 1 + h1], o[j + 1 + h2]
            if not (np.isfinite(fill) and fill > 0 and np.isfinite(x1) and np.isfinite(x2)):
                continue
            out.append({"date": b.index[j], "ret": f1 * (x1 / fill) + (1 - f1) * (x2 / fill) - 1.0 - MR1_CONFIG["COST"]})
    return pd.DataFrame(out)


def mr_rebound_baseline(b: pd.DataFrame, trades: pd.DataFrame, period: Tuple[int, int], seed: int,
                        eligible: Optional[np.ndarray] = None) -> pd.DataFrame:
    """참고(반등): 최근 5거래일 변화 ±0.5 ATR인 같은 종목 다른 날 20곳의 같은 % 거리(L1 질문 1과 같은 방식). eligible이면 대상 날에서만."""
    lc = LECTURE_CONFIG
    bn = lc["BNF"]
    if trades is None or not len(trades):
        return pd.DataFrame()
    o, cl, drop = (b[k].to_numpy(dtype=float) for k in ("Open", "Close", "DROP5"))
    n = len(o)
    D = bn["TIME_STOP_DAYS"]
    cand = np.arange(period[0], min(period[1], n - 1 - D) + 1)
    if eligible is not None and len(cand):
        cand = cand[eligible[cand]]
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
        pool = cs[np.searchsorted(ds, dr - lc["DROP_TOL"], "left"):np.searchsorted(ds, dr + lc["DROP_TOL"], "right")]
        pool = pool[np.abs(pool - t) > lc["HORIZON"]]
        if not len(pool):
            continue
        for j in rng.choice(pool, size=lc["BASELINE_SAMPLES"], replace=True):
            fill = o[j + 1]
            if np.isfinite(fill) and fill > 0:
                out.append({"date": b.index[j], "reb": float(bool(np.any(cl[j + 1:j + 1 + D] >= fill * (1.0 + bn["TIME_STOP_R"] * sp))))})
    return pd.DataFrame(out)


# ---------------------------------------------------------------------
# 신호 (포지션 규칙 없이 — 그날 종가까지의 자료만)
# ---------------------------------------------------------------------
def mr_signals(b: pd.DataFrame, sw: Dict[str, np.ndarray], rule: str, base: Optional[np.ndarray] = None,
               block: Optional[np.ndarray] = None, lines: Optional[Dict[str, np.ndarray]] = None,
               first: int = 0, last: Optional[int] = None, eligible: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """규칙 A·B·C의 신호일 목록과 빠진 사유 수. block = 그날 지수 '강한 −DI'(1/0/NaN, 종목 날짜에 맞춤).
    eligible이 있으면 그날 대상(거래대금·변동성 순위)이 아닌 신호는 빼고 개수만 센다(포지션 규칙 전에 — 빠진 신호가 다음 신호를 막지 않음)."""
    c, lc = MR1_CONFIG, LECTURE_CONFIG
    bn = lc["BNF"]
    n = len(b)
    last = n - 1 if last is None else min(last, n - 1)
    o, h, l, cl = (b[k].to_numpy(dtype=float) for k in ("Open", "High", "Low", "Close"))
    disp, rvol, bbl, atr, drop = (b[k].to_numpy(dtype=float) for k in ("DISP25", "RVOL", "BB_LOW", "ATR14", "DROP5"))
    hold = b["HOLD_BAR"].to_numpy(dtype=bool)
    conf = b["CONFIRM_BAR"].to_numpy(dtype=bool)
    climax = (cl < o) & (rvol >= lc["RVOL_CLIMAX"])
    below_bb = np.isfinite(bbl) & (cl < bbl)
    sigs: List[int] = []
    skips = {"bb": 0, "support": 0, "market": 0, "market_nodata": 0, "universe": 0}
    start = max(first, bn["PEAK_LOOKBACK"] + 5)
    if rule in ("A", "B") and base is None:
        base = lec_bnf_base_series(b, sw)
    if rule == "A" and lines is None:
        lines = lec_swing_lines(b, sw)
    for t in range(start, last + 1):
        if rule == "C":
            if np.isfinite(drop[t]) and drop[t] <= c["C_DROP_ATR"] and np.isfinite(drop[t - 1]) and drop[t - 1] > c["C_DROP_ATR"]:
                if eligible is not None and not eligible[t]:
                    skips["universe"] += 1
                else:
                    sigs.append(t)
            continue
        if not conf[t]:
            continue
        w0, w1 = t - 3, t - 1
        seg = l[w0:w1 + 1]
        m = w0 + int(len(seg) - 1 - np.argmin(seg[::-1]))
        k = (t - 1) - m
        if not (bn["HOLD_MIN"] <= k <= bn["HOLD_MAX"]) or not bool(np.all(hold[m + 1:t])):
            continue
        d2 = next((d for d in range(t - bn["COND2_WINDOW"], t)
                   if np.isfinite(base[d]) and np.isfinite(disp[d]) and disp[d] <= base[d]), None)
        if d2 is None:
            continue
        hi = min(d2 + bn["CLIMAX_WINDOW"], t - 1)
        if not bool(np.any(climax[max(0, d2 - bn["CLIMAX_WINDOW"]):hi + 1])):
            continue
        if rule == "A":
            if not bool(np.any(below_bb[t - c["BB_WINDOW"]:t])):
                skips["bb"] += 1
                continue
            alive = (lines["start"] <= m) & ((m - lines["start"] <= lc["LINE_IDLE_DAYS"]) | (lines["fr"] <= m))
            hw = np.maximum(lc["ZONE_PCT_MIN"] * lines["lv"], lc["ZONE_ATR_MULT"] * atr[m])
            if not bool(np.any(alive & (np.abs(lines["lv"] - l[m]) <= hw))):
                skips["support"] += 1
                continue
            bk = block[t] if block is not None else np.nan
            if not np.isfinite(bk):
                skips["market_nodata"] += 1
                continue
            if bk == 1.0:
                skips["market"] += 1
                continue
        if eligible is not None and not eligible[t]:
            skips["universe"] += 1
            continue
        sigs.append(t)
    return {"t": sigs, "skips": skips}


def mr_trades(b: pd.DataFrame, sig_t: List[int], earn: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """신호 → 거래. 종목당 1포지션, 급락 시작일 p = [t−40, t] 종가 최고일, 손절 = [p, t] 최저가, 실적일 제외, L1과 같은 청산."""
    c, bn = MR1_CONFIG, LECTURE_CONFIG["BNF"]
    n = len(b)
    o, h, l, cl = (b[k].to_numpy(dtype=float) for k in ("Open", "High", "Low", "Close"))
    ma25 = b["SMA25"].to_numpy(dtype=float)
    drop = b["DROP5"].to_numpy(dtype=float)
    dates = b.index
    skips = {"earn_excluded": 0, "earn_unknown": 0, "gap_below_stop": 0, "stop_too_wide": 0, "open_at_end": 0}
    earn_dates = None if not earn else earn.get("dates")
    cover_from = None if not earn else earn.get("cover_from")
    out, free = [], 0
    for t in sig_t:
        if t < free or t + 2 >= n:
            continue
        p0 = max(0, t - bn["PEAK_LOOKBACK"])
        p = p0 + int(np.argmax(cl[p0:t + 1]))
        earn_known = bool(cover_from is not None and dates[p] >= cover_from)
        if earn_known and earn_dates is not None and len(earn_dates):
            a_ = np.searchsorted(earn_dates, np.datetime64(dates[p]), "left")
            z_ = np.searchsorted(earn_dates, np.datetime64(dates[t]), "right")
            if z_ > a_:
                skips["earn_excluded"] += 1
                continue
        if not earn_known:
            skips["earn_unknown"] += 1
        stop = float(np.min(l[p:t + 1]))
        e = t + 1
        entry = o[e]
        if not (np.isfinite(entry) and entry > stop):
            skips["gap_below_stop"] += 1
            continue
        sp = (entry - stop) / entry
        if sp > bn["MAX_STOP_PCT"]:
            skips["stop_too_wide"] += 1
            continue
        res = _lec_bnf_exits(o, h, l, cl, ma25, e, entry, stop)
        if res is None:
            skips["open_at_end"] += 1
            break
        w = cl[e:min(n, e + bn["TIME_STOP_DAYS"])]
        out.append({"date": dates[t], "t": t, "e": e, "entry": entry, "stop": stop, "stop_pct": sp, "drop": float(drop[t]),
                    "reb": float(bool(np.any(w >= entry * (1.0 + bn["TIME_STOP_R"] * sp)))),
                    "gross": res["gross"], "ret": res["gross"] - c["COST"], "h1": res["h1"], "h2": res["h2"], "frac1": res["frac1"],
                    "exit_kind": res["exit_kind"], "earn_known": earn_known})
        free = res["exit_idx"]
    return {"trades": pd.DataFrame(out), "skips": skips}


# ---------------------------------------------------------------------
# 실행
# ---------------------------------------------------------------------
def _mr_bounds(index: pd.DatetimeIndex, start: str, end: Optional[str]) -> Tuple[int, int]:
    s = np.datetime64(pd.Timestamp(start))
    e = np.datetime64(pd.Timestamp(end)) if end else np.datetime64(pd.Timestamp("2262-01-01"))
    return int(np.searchsorted(index.values, s, "left")), int(np.searchsorted(index.values, e, "right")) - 1


def mr_universe(prepared: Dict[str, Dict[str, Any]], idx_ctx: Dict[str, Dict[str, pd.Series]], rule: str,
                start: str, end: Optional[str], progress: Optional[Callable[[int, int], None]] = None,
                eligible: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, Any]:
    """종목 전부의 거래 + 비교(보유 기간 맞춘 무작위 20곳, 최근 하락 폭 맞춘 날의 같은 거리 20곳).
    eligible(종목 → 날짜별 대상 여부)이 있으면 신호도 무작위 비교 날도 대상인 날에서만."""
    parts = {"tr": [], "hold": [], "reb": []}
    skips: Dict[str, int] = {}
    for i, (sym, p) in enumerate(prepared.items()):
        b = p["b"]
        lo_i, hi_i = _mr_bounds(b.index, start, end)
        if hi_i < lo_i:
            continue
        if rule in ("A", "B") and "base" not in p:
            p["base"] = lec_bnf_base_series(b, p["sw"])
        if rule == "A" and "lines" not in p:
            p["lines"] = lec_swing_lines(b, p["sw"])
        ix = idx_ctx.get(lec_index_of(sym, p.get("sector")), {})
        blk = ix.get("block").reindex(b.index).to_numpy(dtype=float) if ix.get("block") is not None else None
        el = eligible.get(sym, np.zeros(len(b), dtype=bool)) if eligible is not None else None
        sg = mr_signals(b, p["sw"], rule, p.get("base"), blk, p.get("lines"), lo_i, hi_i, el)
        r = mr_trades(b, sg["t"], p.get("earn"))
        for k, v in list(sg["skips"].items()) + list(r["skips"].items()):
            skips[k] = skips.get(k, 0) + v
        tr = r["trades"]
        if progress:
            progress(i + 1, len(prepared))
        if not len(tr):
            continue
        mk = lec_market_of(sym)
        g = ix.get("grade")
        tr = tr.assign(ticker=sym, market=mk, grade=(g.reindex(tr["date"]).to_numpy() if g is not None and len(g) else None))
        seed = MR1_CONFIG["SEED"] + 7 * i + {"A": 0, "B": 1, "C": 2}[rule]
        parts["tr"].append(tr)
        hb = mr_hold_baseline(b, tr, (lo_i, hi_i), seed, el)
        rb = mr_rebound_baseline(b, tr, (lo_i, hi_i), seed, el)
        if len(hb):
            parts["hold"].append(hb.assign(ticker=sym, market=mk))
        if len(rb):
            parts["reb"].append(rb.assign(ticker=sym, market=mk))
    cat = lambda x: pd.concat(x, ignore_index=True) if x else pd.DataFrame(columns=["date", "ret", "reb", "ticker", "market"])
    return {"tr": cat(parts["tr"]), "hold": cat(parts["hold"]), "reb": cat(parts["reb"]), "skips": skips}


def mr_judge(cmp_: Dict[str, Any], tr: pd.DataFrame, hold: pd.DataFrame, z_need: float) -> Dict[str, Any]:
    c = MR1_CONFIG
    rs = []
    n = int(cmp_.get("actual", {}).get("n", 0))
    nt = int(tr["ticker"].nunique()) if len(tr) else 0
    d, z = cmp_.get("diff"), cmp_.get("z")
    if n < c["MIN_TRADES"]:
        rs.append(f"거래 {n:,} < {c['MIN_TRADES']}")
    if nt < c["MIN_TICKERS"]:
        rs.append(f"종목 {nt} < {c['MIN_TICKERS']}")
    if d is None or z is None or d <= 0 or z < z_need:
        rs.append(f"무작위보다 나음 z ≥ {z_need:.0f} 아님")
    mk = {}
    for m in ("US", "KR"):
        cm = compare_clustered(tr[tr["market"] == m], hold[hold["market"] == m], "ret") if len(tr) else {}
        mk[m] = cm
    both = [v for v in mk.values() if v.get("actual", {}).get("n", 0) >= c["MIN_MARKET"] and v.get("diff") is not None]
    if len(both) == 2 and np.sign(both[0]["diff"]) != np.sign(both[1]["diff"]):
        rs.append("미국·한국 방향이 다름")
    md = lec_month_drop(tr, hold, "ret", +1) if len(tr) and len(hold) else {"month": None, "diff": None, "z": None}
    if md["diff"] is None or md["diff"] <= 0 or md["z"] is None or md["z"] < c["MONTH_DROP_Z"]:
        rs.append(f"기여가 가장 큰 달({md['month']})을 빼면 기준 미달")
    return {"candidate": not rs, "reasons": rs, "by_market": mk, "month_drop": md, "n_tickers": nt}


def _mr_block(u: Dict[str, Any]) -> Dict[str, Any]:
    tr, hold, reb = u["tr"], u["hold"], u["reb"]
    out = {"n_trades": int(len(tr)), "cmp": compare_clustered(tr, hold, "ret"), "reb": compare_clustered(tr, reb, "reb"), "skips": u["skips"]}
    if len(tr):
        out["exit_kinds"] = tr["exit_kind"].value_counts().to_dict()
        out["win_rate"] = float((tr["ret"] > 0).mean())
        out["mean_hold"] = float(tr["h2"].mean())
        out["earn_known_share"] = float(tr["earn_known"].mean())
    return out


def run_mr1(prepared: Dict[str, Dict[str, Any]], idx_ctx: Dict[str, Dict[str, pd.Series]],
            progress: Optional[Callable[[str, int, int], None]] = None) -> Dict[str, Any]:
    """탐색(기존 종목, 2001~2019): 대상(거래대금 상위 절반 중 변동성 상위 절반)으로 규칙 A·B·C 판정 + 참고(전체 종목).
    확인은 후보가 생겼을 때 새 151종목에서 따로(이 함수는 열지 않음)."""
    c = MR1_CONFIG
    elig = mr_eligibility(prepared)
    s0, s1 = pd.Timestamp(c["EXPLORE_START"]), pd.Timestamp(c["EXPLORE_END"])
    days = sum(int(((p["b"].index >= s0) & (p["b"].index <= s1)).sum()) for p in prepared.values())
    edays = sum(int(np.sum(elig.get(sym, np.zeros(0, dtype=bool))[(p["b"].index >= s0) & (p["b"].index <= s1)]))
                for sym, p in prepared.items())
    res: Dict[str, Any] = {"rules": {}, "all_stocks": {}, "config_version": c["VERSION"], "n_tickers_used": len(prepared),
                           "eligible_share": (edays / days) if days else None,
                           "eligible_tickers": int(sum(1 for sym, p in prepared.items()
                                                       if np.any(elig.get(sym, np.zeros(0, dtype=bool))[(p["b"].index >= s0) & (p["b"].index <= s1)])))}
    U = {}
    for r in ("A", "B", "C"):
        ua = mr_universe(prepared, idx_ctx, r, c["EXPLORE_START"], c["EXPLORE_END"],
                         progress=(lambda i, n, r=r: progress(f"규칙 {r} 전체(참고)", i, n)) if progress else None)
        res["all_stocks"][r] = {"cmp": compare_clustered(ua["tr"], ua["hold"], "ret"), "n_trades": int(len(ua["tr"]))}
        u = mr_universe(prepared, idx_ctx, r, c["EXPLORE_START"], c["EXPLORE_END"],
                        progress=(lambda i, n, r=r: progress(f"규칙 {r} 대상", i, n)) if progress else None, eligible=elig)
        U[r] = u
        blk = _mr_block(u)
        blk["judge"] = mr_judge(blk["cmp"], u["tr"], u["hold"], c["EXPLORE_Z"])
        blocks = {}
        for s0, s1 in c["PERIOD_BLOCKS"]:
            ta = u["tr"][(pd.to_datetime(u["tr"]["date"]) >= s0) & (pd.to_datetime(u["tr"]["date"]) <= s1)] if len(u["tr"]) else u["tr"]
            ha = u["hold"][(pd.to_datetime(u["hold"]["date"]) >= s0) & (pd.to_datetime(u["hold"]["date"]) <= s1)] if len(u["hold"]) else u["hold"]
            blocks[f"{s0[:4]}~{s1[:4]}"] = compare_clustered(ta, ha, "ret")
        blk["periods"] = blocks
        gr = {}
        if len(u["tr"]) and "grade" in u["tr"]:
            for gname in ("🟢", "🟡", "🔴", "⚫"):
                gg = u["tr"][u["tr"]["grade"] == gname]
                gr[gname] = {"n": int(len(gg)), "ret": float(gg["ret"].mean()) if len(gg) else None}
        blk["by_grade"] = gr
        res["rules"][r] = blk
    res["vs_C"] = {r: compare_clustered(U[r]["tr"], U["C"]["tr"], "ret") for r in ("A", "B")}
    res["candidates"] = [r for r in ("A", "B", "C") if res["rules"][r]["judge"]["candidate"]]
    return res


# ---------------------------------------------------------------------
# 보고서 · 기록
# ---------------------------------------------------------------------
def _mr_cmp(x: Dict[str, Any], rate: bool = False) -> str:
    a, b_ = x.get("actual", {}), x.get("baseline", {})
    if not a.get("n"):
        return "표본 없음"
    f = (lambda v: f"{v * 100:.1f}%") if rate else (lambda v: f"{v * 100:+.2f}%")
    z = "—" if x.get("z") is None else f"{x['z']:+.1f}"
    d = "—" if x.get("diff") is None else (f"{x['diff'] * 100:+.1f}%p" if rate else f"{x['diff'] * 100:+.2f}%p")
    return f"{f(a['mean'])} ({a['n']:,}건) vs {f(b_['mean']) if b_.get('n') else '—'} · 차이 {d} · z {z}"


def render_mr1_report(r: Dict[str, Any], generated_at: Optional[str] = None) -> str:
    c = MR1_CONFIG
    L = ["# H-MR1 기술적 단기 반등 단타(평균 회귀) — 탐색", "",
         "**탐색 단계 결과 — 이것만으로는 증거가 아니다.** 이미 다른 시험에 쓴 545종목의 2001~2019년이다. "
         "후보가 나오면 새 151종목(2001~2026)에서 z ≥ 3으로 한 번 확인해야 '통과'(MR1_PREREG.md).", "",
         f"- 생성: {generated_at or '—'} · 설정 {r['config_version']} (사전 등록 {c['REGISTERED']}) · 종목 {r['n_tickers_used']}개",
         "- 공통: 신호 다음 날 시가 진입, 손절 = 급락 시작일~신호일 최저가(손절폭 ≤ 12%), 실적일 제외, 25일선 절반 익절·5일 +0.5R 시간 손절, 비용 0.30%, 종목당 1포지션",
         "- **대상(판정)**: 신호일마다 같은 시장 시험 종목 중 지난 60거래일 평균 거래대금 상위 절반 → 그중 지난 250거래일 변동성 상위 절반. "
         f"탐색 기간 종목·날의 {'—' if r.get('eligible_share') is None else format(r['eligible_share'] * 100, '.0f') + '%'}가 대상, 한 번이라도 대상이 된 종목 {r.get('eligible_tickers', 0)}개. "
         "무작위 비교 날도 같은 종목이 대상이었던 날에서만 뽑는다",
         "- 판정: 거래당 순수익 − 보유 기간 맞춘 같은 종목 무작위 진입(비용 뒤) > 0, 월 묶음 z ≥ 2, 거래 ≥ 300, 종목 ≥ 50, 미국·한국 같은 방향, 달 쏠림 점검", "",
         "## 판정 (대상: 거래대금 상위 절반 중 변동성 상위 절반)", "", "| 규칙 | 거래당 순수익 vs 무작위 | 판정 |", "|---|---|---|"]
    for k in ("A", "B", "C"):
        x = r["rules"][k]
        j = x["judge"]
        v = "**탐색 후보** → 새 151종목 확인 대기" if j["candidate"] else "탐색 후보 아님 (" + ", ".join(j["reasons"]) + ")"
        L.append(f"| {MR1_RULES[k]} | {_mr_cmp(x['cmp'])} | {v} |")
    L += ["", "참고 — 대상 조건 없이 전체 종목(판정 아님): " + " · ".join(
        f"{k} {_mr_cmp(r['all_stocks'][k]['cmp'])}" for k in ("A", "B", "C") if k in r.get("all_stocks", {}))]
    L += ["", "## 참고 — 강의 부품이 단순 반전보다 나은가(판정 아님, 대상 안에서)", ""]
    for k in ("A", "B"):
        L.append(f"- {k} − C 거래당 순수익: {_mr_cmp(r['vs_C'][k])}")
    L += ["", "## 참고 — 규칙별 자세히", ""]
    for k in ("A", "B", "C"):
        x = r["rules"][k]
        j = x["judge"]
        sk = x["skips"]
        md = j["month_drop"]
        L += [f"**{MR1_RULES[k]}**", "",
              f"- 미국 {_mr_cmp(j['by_market']['US'])} / 한국 {_mr_cmp(j['by_market']['KR'])}",
              f"- 기여가 가장 큰 달 {md['month']}을 빼면 차이 {'—' if md['diff'] is None else format(md['diff'] * 100, '+.2f') + '%p'}, z {'—' if md['z'] is None else format(md['z'], '+.1f')}",
              f"- 5거래일 안 +0.5R 반등: {_mr_cmp(x['reb'], rate=True)} (최근 하락 폭 맞춘 날의 같은 거리)",
              "- 기간별: " + " · ".join(f"{p} {_mr_cmp(v)}" for p, v in x["periods"].items()),
              "- 시장 등급별 거래당 순수익: " + (" · ".join(f"{g} {_lec_pct(v['ret'])} ({v['n']:,})" for g, v in x["by_grade"].items()) or "—"),
              f"- 청산: {', '.join(f'{a} {b_}' for a, b_ in (x.get('exit_kinds') or {}).items()) or '—'} · 이긴 비율 "
              f"{'—' if x.get('win_rate') is None else format(x['win_rate'] * 100, '.0f') + '%'} · 평균 보유 {x.get('mean_hold') or 0:.1f}일 · "
              f"실적일 확인 비율 {'—' if x.get('earn_known_share') is None else format(x['earn_known_share'] * 100, '.0f') + '%'}",
              f"- 빠진 신호: 대상 아님(거래대금·변동성) {sk.get('universe', 0)} · 볼린저 하단 없음 {sk.get('bb', 0)} · 지지 구간 밖 {sk.get('support', 0)} · 시장 강한 −DI {sk.get('market', 0)} · 지수 자료 없음 {sk.get('market_nodata', 0)} · "
              f"실적일 제외 {sk.get('earn_excluded', 0)} · 진입가 ≤ 손절가 {sk.get('gap_below_stop', 0)} · 손절폭 > 12% {sk.get('stop_too_wide', 0)} · 열린 채 끝남 {sk.get('open_at_end', 0)}", ""]
    L += ["## 읽는 법", "",
          "- 후보가 아니면 여기서 끝낸다. 결과를 보고 숫자(−3 ATR·볼린저·지지 구간·시장 필터·청산)를 바꿔 다시 보지 않는다.",
          "- 기존 545종목의 2020년 이후와 새 151종목은 이 단계에서 열지 않았다.", ""]
    return "\n".join(L)


def mr1_log_lines(r: Dict[str, Any], generated_at: str) -> List[str]:
    rows = []
    for k in ("A", "B", "C"):
        x = r["rules"][k]["cmp"]
        d = "—" if x.get("diff") is None else f"{x['diff'] * 100:+.2f}%p"
        z = "—" if x.get("z") is None else f"{x['z']:+.1f}"
        rows.append(f"| {generated_at} | `mr1-{k}` | {MR1_RULES[k]} — 거래대금 상위 절반 중 변동성 상위 절반, 거래당 순수익 vs 보유 기간 맞춘 무작위 (탐색 545종목 2001~2019) | "
                    f"{x.get('actual', {}).get('n', 0):,} | {d} | {z} | {'후보' if r['rules'][k]['judge']['candidate'] else '후보 아님'} |")
    return rows
