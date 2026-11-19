"""H-TR 추세 투자 흐름 검증 (2026-11-20 사전 등록 — 결과 보기 전 고정, data/backtest/TR_PREREG.md, tr-1).

사용자 흐름: 1번 시장(🟢·🟡) → 2번 RS 주도 → 5번 120일선 상승 → 3·4·6번 진입 신호(A 돌파 확정 / B SR Flip, 물량 소화는 건수로 결정)
→ 9번 실적 쇼크 제외 + 손절폭 ≤ 12% → 7·8번 박스 1대1 목표 · 추세선 장대음봉 이탈 절반 · 붕괴·무효 전부.
판정 비교 = 같은 종목 무작위 날 20곳에서 같은 날 간격·같은 비율로 판 결과(보유 기간 맞춤). 판정 규칙은 A·B 두 개뿐.
모든 신호·조건은 신호일 종가까지의 자료만 쓰고, 진입은 다음 날 시가. 탐색은 2013-12-31 뒤 가격을 보지 않는다.
숫자는 TR_CONFIG 한 곳. 결과를 보고 바꾸지 않는다.
"""
import math
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .lecture_backtest import lec_index_of, lec_market_of
from .lecture_events import _lec_arr, _lec_nconf, lec_breakouts, lec_double_bottoms, lec_ma_double_bottoms
from .lecture_family import lec_cluster_t, lec_month_code
from .lecture_levels import lec_collapse, lec_upline
from .lecture_signals import lec_rs_daily, lec_state

TR_CONFIG: Dict[str, Any] = {
    "VERSION": "tr-1",
    "REGISTERED": "2026-11-20",
    "EXPLORE_START": "2001-01-01", "EXPLORE_END": "2013-12-31",   # 탐색: 545종목, 이 기간 뒤 가격은 보지 않음
    "VERIFY_START": "2014-01-01",                                  # 확인: 같은 종목 2014~자료 끝 (탐색 후보만)
    "FINAL_START": "2001-01-01",                                   # 최종: 새 151종목 2001~자료 끝 (확인 통과만)
    # --- 진입 조건 ---
    "MAX_STOP_PCT": 0.12,          # 손절폭 > 12%면 패스 (Q67)
    "TGT_MIN_WIDTH": 0.05,         # 박스 폭 ÷ 돌파선 < 5%면 목표 없음 → 거래 안 함 (Q59)
    "BOX_MAX_DAYS": 250,           # 박스 시작을 찾는 최대 거리(Q55와 같은 250거래일)
    "SHOCK_ATR": 2.0,              # 실적 쇼크 = 발표 반응 ≤ −2 × ATR14(발표 전 거래일) (T3)
    "ABSORB_MIN_SIGNALS": 300,     # T1: A + 물량 소화 신호 수 ≥ 300이면 판정 사슬에 넣음
    "MADB_DAYS": 60,               # 참고: 신호 전 60거래일 안 이평 쌍바닥
    # --- 청산 ---
    "MAX_HOLD": 250,               # 진입일 포함 250거래일째 종가 (Q61)
    "HALF": 0.5,                   # 추세선 장대음봉 이탈 = 절반 (7-7)
    "COST": 0.003,
    # --- 판정 ---
    "BASELINE_SAMPLES": 20,
    "MIN_TRADES": 300, "MIN_TICKERS": 50, "MIN_MARKET": 30,
    "EXPLORE_Z": 2.0, "YEAR_DROP_Z": 2.0, "YEAR_MIN_TRADES": 30, "YEAR_AGREE": 2.0 / 3.0,
    "VERIFY_Z": 2.0, "VERIFY_MIN_TRADES": 100,
    "FINAL_Z": 3.0, "FINAL_MIN_TRADES": 100,
    "POWER_Z": 0.84,
    "MDE_HOLDS": (20, 40, 60), "MDE_DRAWS": 4000, "MDE_CLUSTER": 1.3,
    "RISK_PCT": 0.01,              # 참고: 1회 위험 1% (9-6)
    "SEED": 1122,
}
TR_RULES: Tuple[str, ...] = ("A", "B")
TR_LABELS: Dict[str, str] = {"A": "A 돌파 확정 즉시 진입(E02 다음 날 시가)", "B": "B SR Flip 진입(E04 다음 날 시가)"}
TR_FUNNEL: Tuple[Tuple[str, str], ...] = (
    ("signal", "신호(규칙 A 돌파 확정 · 규칙 B SR Flip)"), ("market", "+ ① 시장 🟢·🟡"), ("rs", "+ ② RS 주도"),
    ("ma120", "+ ③ 120일선 상승"), ("absorb", "+ ⑤ 물량 소화(T1)"), ("earn", "+ ⑥ 실적 쇼크 제외"),
    ("stop", "+ ⑦ 진입가 > 무효선 · 손절폭 ≤ 12%"), ("box", "+ ⑧ 박스 폭 ≥ 5% · 진입가 < 목표"))
TR_ABLATE: Dict[str, str] = {"market": "① 시장 빼기", "rs": "② RS 빼기", "ma120": "③ 120일선 빼기", "earn": "⑥ 실적 쇼크 제외 빼기",
                             "absorb": "⑤ 물량 소화 빼기(넣었을 때만)"}


# ---------------------------------------------------------------------
# 박스·실적 쇼크 (순수 함수 — 테스트에서 손으로 확인)
# ---------------------------------------------------------------------
def tr_box(ho: np.ndarray, hp: np.ndarray, lo_o: np.ndarray, lo_p: np.ndarray, k: int, t0: int, zl: float, zu: float,
           max_days: Optional[int] = None) -> Dict[str, Any]:
    """박스 시작·박스 고점 수·물량 소화(T1). k = 돌파선 고점의 스윙 번호.
    과거로 확정 고점을 거슬러 가며 구간 [zl, zu] 안의 가장 이른 고점이 박스 시작(구간 위 고점이나 250거래일 넘으면 멈춤, 낮은 고점은 지나감).
    물량 소화 = 박스 고점 ≥ 2 그리고 박스 시작 뒤 t0까지 확정된 저점의 마지막 두 개가 높아짐 + 둘 다 zl 아래."""
    md = TR_CONFIG["BOX_MAX_DAYS"] if max_days is None else max_days
    start, tops = int(ho[k]), 1
    for j in range(k - 1, -1, -1):
        if ho[k] - ho[j] > md or hp[j] > zu:
            break
        if hp[j] >= zl:
            start, tops = int(ho[j]), tops + 1
    nl = _lec_nconf(lo_o, t0)
    idx = np.flatnonzero(lo_o[:nl] > start)
    absorb = bool(tops >= 2 and len(idx) >= 2 and lo_p[idx[-1]] > lo_p[idx[-2]] and lo_p[idx[-1]] < zl and lo_p[idx[-2]] < zl)
    return {"start": start, "tops": tops, "absorb": absorb}


def tr_earn_status(dates: np.ndarray, cl: np.ndarray, atr: np.ndarray, ed: Optional[np.ndarray], start: int, s: int,
                   cover_from: Optional[pd.Timestamp]) -> str:
    """박스 시작 ~ 신호일 실적 발표의 반응(발표 다음 거래일 종가 − 발표 전 거래일 종가, s를 넘으면 s 종가) ≤ −2 ATR → 'shock'.
    실적일 자료가 박스 시작을 덮지 못하면 'unknown'(제외하지 않음), 아니면 'ok'."""
    if cover_from is None or ed is None or pd.Timestamp(dates[start]) < pd.Timestamp(cover_from):
        return "unknown"
    ed = np.asarray(ed, dtype="datetime64[ns]")
    dv = np.asarray(dates, dtype="datetime64[ns]")
    a_ = int(np.searchsorted(ed, dv[start], "left"))
    z_ = int(np.searchsorted(ed, dv[s], "right"))
    k = TR_CONFIG["SHOCK_ATR"]
    for e in ed[a_:z_]:
        a = int(np.searchsorted(dv, e, "left"))
        if a < 1 or a > s:
            continue
        j = min(a + 1, s)
        if np.isfinite(atr[a - 1]) and atr[a - 1] > 0 and cl[j] - cl[a - 1] <= -k * atr[a - 1]:
            return "shock"
    return "ok"


# ---------------------------------------------------------------------
# 종목 하나: 신호 표 + 청산에 쓸 배열 (수익은 아직 보지 않음)
# ---------------------------------------------------------------------
def _tr_bool(x) -> np.ndarray:
    return np.asarray(x, dtype=bool)


def tr_context(sym: str, p: Dict[str, Any], grade: Optional[pd.Series]) -> Dict[str, Any]:
    cf = TR_CONFIG
    b, sw = p["b"], p["sw"]
    n = len(b)
    dates = b.index
    o, lo, cl, atr = _lec_arr(b, "Open"), _lec_arr(b, "Low"), _lec_arr(b, "Close"), _lec_arr(b, "ATR14")
    g = grade.reindex(dates).to_numpy(dtype=object) if (grade is not None and len(grade)) else np.full(n, None, dtype=object)
    mk_ok = np.array([x in ("🟢", "🟡") for x in g], dtype=bool)
    rs = p.get("rs")
    rsd = lec_rs_daily(rs).reindex(dates).to_numpy(dtype=float) if (rs is not None and len(rs)) else np.full(n, np.nan)
    rs_ok = rsd == 2
    ma_ok = _lec_arr(b, "SMA120_SLOPE") > 0
    earn = p.get("earn") or {}
    ed, cov = earn.get("dates"), earn.get("cover_from")
    dv = dates.values
    rows = []
    for x in lec_breakouts(b, sw):
        if x["confirm"] is None:
            continue
        t0, H, zl, zu = x["t0"], float(x["level"]), float(x["zl"]), float(x["zu"])
        k = int(np.searchsorted(sw["h_o"], x["h_origin"]))
        bx = tr_box(sw["h_o"], sw["h_p"], sw["l_o"], sw["l_p"], k, t0, zl, zu)
        width = H - float(np.min(lo[bx["start"]:t0 + 1]))
        target = H + width
        for rule, s in (("A", x["confirm"]), ("B", x["flip"])):
            if s is None or s + 1 >= n:
                continue
            E = float(o[s + 1])
            sp = (E - zl) / E if E > 0 else np.nan
            rows.append({"t": int(s), "t0": int(t0), "rule": rule, "entry": E, "zl": zl, "zu": zu, "level": H, "target": target,
                         "width": width / H, "box_start": bx["start"], "tops": bx["tops"], "absorb": bx["absorb"],
                         "market": bool(mk_ok[s]), "grade": g[s], "rs": bool(rs_ok[s]), "ma120": bool(ma_ok[s]),
                         "earn": tr_earn_status(dv, cl, atr, ed, bx["start"], s, cov),
                         "stop_ok": bool(E > zl and sp <= cf["MAX_STOP_PCT"]), "stop_pct": sp,
                         "box_ok": bool(width / H >= cf["TGT_MIN_WIDTH"] and E < target)})
    st = p.get("st")
    st = st if st is not None else lec_state(b, sw)
    madb = sorted({int(x["e07"]) for x in lec_double_bottoms(b, sw, st) if x.get("e07") is not None}
                  | {int(x["t"]) for x in lec_ma_double_bottoms(b, st)})
    ul = lec_upline(b, sw)
    brk = np.zeros(n, dtype=bool)
    if ul["breaks"]:
        brk[np.asarray(ul["breaks"], dtype=int)] = True
    coll = np.array([(x["tb"], x["w10"]) for x in lec_collapse(b, sw) if x["w10"] is not None], dtype=int).reshape(-1, 2)
    sig = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["t", "rule"])
    return {"sig": sig, "setup": mk_ok & rs_ok & ma_ok, "bb_break": brk & _tr_bool(b["BIG_BEAR"].to_numpy()),
            "coll": coll, "madb": np.asarray(madb, dtype=int), "market": lec_market_of(sym)}


def tr_contexts(prepared: Dict[str, Dict[str, Any]], grades: Dict[str, pd.Series],
                progress: Optional[Callable[[int, int], None]] = None) -> Dict[str, Dict[str, Any]]:
    out = {}
    syms = sorted(prepared)
    for i, sym in enumerate(syms):
        p = prepared[sym]
        out[sym] = tr_context(sym, p, grades.get(lec_index_of(sym, p.get("sector"))))
        if progress:
            progress(i + 1, len(syms))
    return out


def tr_bounds(dates: pd.DatetimeIndex, start, end) -> Tuple[int, int]:
    """[start, end] 날짜의 위치 범위(포함). end None = 자료 끝."""
    lo = int(np.searchsorted(dates.values, np.datetime64(pd.Timestamp(start)), "left"))
    hi = len(dates) - 1 if end is None else int(np.searchsorted(dates.values, np.datetime64(pd.Timestamp(end)), "right")) - 1
    return lo, hi


def tr_pass(sg: pd.DataFrame, use_absorb: bool, drop: Optional[str] = None) -> np.ndarray:
    """판정 사슬 통과(drop = 참고용으로 뺄 단계)."""
    if not len(sg):
        return np.zeros(0, dtype=bool)
    m = np.ones(len(sg), dtype=bool)
    for k in ("market", "rs", "ma120"):
        if k != drop:
            m &= sg[k].to_numpy(dtype=bool)
    if use_absorb and drop != "absorb":
        m &= sg["absorb"].to_numpy(dtype=bool)
    if drop != "earn":
        m &= (sg["earn"] != "shock").to_numpy()
    m &= sg["stop_ok"].to_numpy(dtype=bool) & sg["box_ok"].to_numpy(dtype=bool)
    return m


def tr_funnel(ctx: Dict[str, Dict[str, Any]], prepared: Dict[str, Dict[str, Any]], rule: str, start, end) -> Dict[str, Any]:
    """깔때기(신호 수, 사용자 순서). 'with' = 물량 소화를 넣은 사슬, 'without' = 뺀 사슬(⑤ 단계 없음)."""
    keys = [k for k, _ in TR_FUNNEL]
    w = {k: 0 for k in keys}
    wo = {k: 0 for k in keys if k != "absorb"}
    unknown, reach = 0, 0
    for sym, cx in ctx.items():
        sg = cx["sig"]
        if not len(sg):
            continue
        lo, hi = tr_bounds(prepared[sym]["b"].index, start, end)
        sg = sg[(sg["rule"] == rule) & (sg["t"] >= lo) & (sg["t"] + 1 <= hi)]
        if not len(sg):
            continue
        m = np.ones(len(sg), dtype=bool)
        w["signal"] += len(sg)
        wo["signal"] += len(sg)
        for k in ("market", "rs", "ma120"):
            m &= sg[k].to_numpy(dtype=bool)
            w[k] += int(m.sum())
            wo[k] += int(m.sum())
        ok_e = (sg["earn"] != "shock").to_numpy()
        unk = (sg["earn"] == "unknown").to_numpy()
        so, bo = sg["stop_ok"].to_numpy(dtype=bool), sg["box_ok"].to_numpy(dtype=bool)
        for chain, mm in ((w, m & sg["absorb"].to_numpy(dtype=bool)), (wo, m.copy())):
            if chain is w:
                w["absorb"] += int(mm.sum())
            else:
                reach += int(mm.sum())
                unknown += int((mm & unk).sum())
            mm = mm & ok_e
            chain["earn"] += int(mm.sum())
            mm = mm & so
            chain["stop"] += int(mm.sum())
            mm = mm & bo
            chain["box"] += int(mm.sum())
    return {"with": w, "without": wo, "with_absorb": w["box"], "no_absorb": wo["box"], "earn_unknown": unknown, "earn_reach": reach}


# ---------------------------------------------------------------------
# 청산 · 보유 기간 맞춘 비교 기준
# ---------------------------------------------------------------------
def tr_exit(o: np.ndarray, h: np.ndarray, c: np.ndarray, e: int, hi: int, zl: float, target: float, w10: set,
            bb_break: np.ndarray, simple: bool = False) -> List[Tuple[int, float, float, str, str]]:
    """매도 묶음 [(진입일부터 날 수, 비율, 가격, 'open'|'close', 이유)]. 사전 등록 6장 순서 그대로."""
    cf = TR_CONFIG
    legs: List[Tuple[int, float, float, str, str]] = []
    rem, halved, pend = 1.0, False, None
    for d in range(e, hi + 1):
        if pend is not None:
            frac = rem if pend[0] == "all" else min(cf["HALF"], rem)
            legs.append((d - e, frac, float(o[d]), "open", pend[1]))
            rem -= frac
            halved = halved or pend[0] == "half"
            pend = None
            if rem <= 1e-12:
                return legs
        if h[d] >= target:
            px = float(o[d]) if (d > e and o[d] >= target) else float(target)
            legs.append((d - e, rem, px, "close", "목표"))
            return legs
        if d - e + 1 >= cf["MAX_HOLD"]:
            legs.append((d - e, rem, float(c[d]), "close", "최대 보유"))
            return legs
        if d == hi:
            legs.append((d - e, rem, float(c[d]), "close", "기간 끝"))
            return legs
        if c[d] < zl:
            pend = ("all", "무효(돌파선 아래)")
        elif d in w10:
            pend = ("all", "붕괴")
        elif bb_break[d] and not halved:
            pend = ("all" if simple else "half", "추세선 장대음봉 이탈")
    return legs


def tr_leg_return(legs, entry: float) -> float:
    return float(sum(f * px for _, f, px, _, _ in legs) / entry - 1.0)


def tr_base(o: np.ndarray, c: np.ndarray, legs, lo: int, hi: int, rng: np.random.Generator, pool_mask: Optional[np.ndarray] = None,
            k: Optional[int] = None) -> float:
    """같은 종목 무작위 진입일 r(시험 기간 안, r + 마지막 날 수 ≤ hi)에서 시가에 사서 같은 날 간격·비율로 판 순수익의 평균.
    pool_mask가 있으면 전날 pool_mask가 참인 날만(셋업만). 후보가 없으면 NaN."""
    cf = TR_CONFIG
    kmax = max(x[0] for x in legs)
    r0, r1 = max(lo + 1, 1), hi - kmax
    if r1 < r0:
        return np.nan
    pool = np.arange(r0, r1 + 1)
    if pool_mask is not None:
        pool = pool[pool_mask[pool - 1]]
    if not len(pool):
        return np.nan
    r = rng.choice(pool, size=k or cf["BASELINE_SAMPLES"], replace=True)
    val = np.zeros(len(r))
    for kk, f, _, kind, _ in legs:
        val += f * (o[r + kk] if kind == "open" else c[r + kk])
    return float(np.mean(val / o[r] - 1.0) - cf["COST"])


def tr_simulate(ctx: Dict[str, Dict[str, Any]], prepared: Dict[str, Dict[str, Any]], rule: str, start, end, use_absorb: bool,
                drop: Optional[str] = None, simple: bool = False, seed: Optional[int] = None) -> pd.DataFrame:
    """규칙 하나를 시험 기간에서 거래로 만든다(종목당 1포지션, 판 날 다음 날부터 새 신호). 수익·기준·셋업만 기준을 함께."""
    cf = TR_CONFIG
    rng = np.random.default_rng(cf["SEED"] if seed is None else seed)
    rng_s = np.random.default_rng((cf["SEED"] if seed is None else seed) + 7)
    rows = []
    for sym in sorted(ctx):
        cx = ctx[sym]
        sg = cx["sig"]
        if not len(sg):
            continue
        b = prepared[sym]["b"]
        lo, hi = tr_bounds(b.index, start, end)
        sg = sg[(sg["rule"] == rule) & (sg["t"] >= lo) & (sg["t"] + 1 <= hi)]
        if not len(sg):
            continue
        sg = sg[tr_pass(sg, use_absorb, drop)].sort_values("t")
        if not len(sg):
            continue
        o, h, c = _lec_arr(b, "Open"), _lec_arr(b, "High"), _lec_arr(b, "Close")
        coll = cx["coll"]
        free = -1
        for r in sg.itertuples(index=False):
            s = int(r.t)
            if s <= free:
                continue
            e = s + 1
            w10 = set(coll[coll[:, 0] >= e, 1].tolist()) if len(coll) else set()
            legs = tr_exit(o, h, c, e, hi, r.zl, r.target, w10, cx["bb_break"], simple=simple)
            net = tr_leg_return(legs, o[e]) - cf["COST"]
            free = e + legs[-1][0]
            madb = cx["madb"]
            rows.append({"ticker": sym, "market": cx["market"], "date": b.index[s], "t": s, "exit_t": free, "entry": float(o[e]), "net": net,
                         "base": tr_base(o, c, legs, lo, hi, rng),
                         "base_setup": tr_base(o, c, legs, lo, hi, rng_s, pool_mask=cx["setup"]),
                         "hold": legs[-1][0] + 1, "reason": legs[-1][4], "half": any(x[4] == "추세선 장대음봉 이탈" and x[1] < 1 for x in legs),
                         "stop_pct": float(r.stop_pct), "width": float(r.width), "grade": r.grade, "absorb": bool(r.absorb),
                         "madb": bool(len(madb) and np.any((madb >= s - cf["MADB_DAYS"]) & (madb <= s)))})
    tr = pd.DataFrame(rows)
    if len(tr):
        tr["x"] = tr["net"] - tr["base"]
        tr["x_setup"] = tr["net"] - tr["base_setup"]
    return tr


# ---------------------------------------------------------------------
# 통계 · 판정
# ---------------------------------------------------------------------
def tr_year_checks(x: np.ndarray, dates) -> Dict[str, Any]:
    """기여가 가장 큰 해 빼기 + 해별 방향(거래 ≥ 30인 해 중 차이 > 0 비율)."""
    cf = TR_CONFIG
    x = np.asarray(x, dtype=float)
    d = pd.DatetimeIndex(pd.to_datetime(dates))
    yr = d.year.to_numpy()
    g = lec_month_code(d)
    out = {"drop_year": None, "drop_diff": None, "drop_t": None, "years_ok": 0, "years_n": 0, "years_share": None, "by_year": {}}
    m = np.isfinite(x)
    x, yr, g = x[m], yr[m], g[m]
    if len(x) < 3:
        return out
    s = pd.Series(x).groupby(yr).sum()
    worst = int(s.idxmax())
    keep = yr != worst
    r = lec_cluster_t(x[keep], g[keep])
    out.update({"drop_year": worst, "drop_diff": r["mean"], "drop_t": r["t"]})
    for y in np.unique(yr):
        mm = yr == y
        out["by_year"][int(y)] = {"n": int(mm.sum()), "diff": float(x[mm].mean())}
        if mm.sum() >= cf["YEAR_MIN_TRADES"]:
            out["years_n"] += 1
            out["years_ok"] += int(x[mm].mean() > 0)
    out["years_share"] = (out["years_ok"] / out["years_n"]) if out["years_n"] else None
    return out


def _tr_ct(x, dates) -> Dict[str, Any]:
    x = np.asarray(x, dtype=float)
    return lec_cluster_t(x, lec_month_code(dates)) if len(x) else {"n": 0, "mean": None, "se": None, "t": None, "k": 0}


def tr_stats(tr: pd.DataFrame) -> Dict[str, Any]:
    cf = TR_CONFIG
    if tr is None or not len(tr):
        return {"n": 0, "n_tickers": 0, "diff": None, "t": None, "se": None, "net": None, "base": None,
                "by_market": {m: {"n": 0, "diff": None, "t": None} for m in ("US", "KR")}, "years": tr_year_checks(np.array([]), [])}
    tr = tr[np.isfinite(tr["x"].to_numpy(dtype=float))]
    r = _tr_ct(tr["x"], tr["date"])
    rn = _tr_ct(tr["net"], tr["date"])
    rs = _tr_ct(tr["x_setup"].dropna(), tr.loc[tr["x_setup"].notna(), "date"])
    w = np.minimum(1.0, cf["RISK_PCT"] / np.maximum(tr["stop_pct"].to_numpy(dtype=float), 1e-9))
    acct = w * tr["net"].to_numpy(dtype=float)
    d = pd.to_datetime(tr["date"])
    years = max((d.max() - d.min()).days / 365.25, 1.0)
    out = {"n": r["n"], "n_tickers": int(tr["ticker"].nunique()), "diff": r["mean"], "t": r["t"], "se": r["se"], "n_months": r["k"],
           "net": float(tr["net"].mean()), "net_t": rn["t"], "base": float(tr["base"].mean()), "win": float((tr["net"] > 0).mean()),
           "hold": float(tr["hold"].mean()), "hold_med": float(tr["hold"].median()),
           "reasons": {k: float(v) for k, v in tr["reason"].value_counts(normalize=True).items()},
           "reason_net": {k: float(v) for k, v in tr.groupby("reason")["net"].mean().items()},
           "half_share": float(tr["half"].mean()), "stop_med": float(tr["stop_pct"].median()), "width_med": float(tr["width"].median()),
           "setup": {"n": rs["n"], "diff": rs["mean"], "t": rs["t"]},
           "acct": {"weight_med": float(np.median(w)), "per_trade": float(acct.mean()), "per_year": float(acct.sum() / years)},
           "by_market": {}}
    for m in ("US", "KR"):
        mm = (tr["market"] == m).to_numpy()
        rr = _tr_ct(tr["x"].to_numpy()[mm], tr["date"].to_numpy()[mm])
        out["by_market"][m] = {"n": rr["n"], "diff": rr["mean"], "t": rr["t"]}
    out["years"] = tr_year_checks(tr["x"].to_numpy(), tr["date"])
    return out


def tr_mde(s: Dict[str, Any], z_need: float) -> Optional[float]:
    se = s.get("se")
    return None if se is None or not np.isfinite(se) else float((z_need + TR_CONFIG["POWER_Z"]) * se)


def _tr_market_conflict(s: Dict[str, Any]) -> bool:
    mk = [v for v in s["by_market"].values() if v["n"] >= TR_CONFIG["MIN_MARKET"] and v["diff"] is not None]
    return len(mk) == 2 and np.sign(mk[0]["diff"]) != np.sign(mk[1]["diff"])


def tr_judge_explore(s: Dict[str, Any]) -> Dict[str, Any]:
    cf = TR_CONFIG
    rs = []
    if s["n"] < cf["MIN_TRADES"]:
        rs.append(f"거래 {s['n']:,} < {cf['MIN_TRADES']}")
    if s["n_tickers"] < cf["MIN_TICKERS"]:
        rs.append(f"종목 {s['n_tickers']} < {cf['MIN_TICKERS']}")
    if s.get("net") is None or s["net"] <= 0:
        rs.append("비용 뒤 거래당 순수익 ≤ 0")
    if s["diff"] is None or s["diff"] <= 0:
        rs.append("보유 기간 맞춘 무작위보다 나쁨(차이 ≤ 0)")
    if s["t"] is None or not np.isfinite(s["t"]) or s["t"] < cf["EXPLORE_Z"]:
        rs.append(f"t < {cf['EXPLORE_Z']:.0f}")
    if _tr_market_conflict(s):
        rs.append("미국·한국 방향이 다름")
    y = s["years"]
    if y["drop_diff"] is None or y["drop_diff"] <= 0 or y["drop_t"] is None or y["drop_t"] < cf["YEAR_DROP_Z"]:
        rs.append(f"기여가 가장 큰 해({y['drop_year']})를 빼면 기준 미달" if y["drop_year"] is not None else "해 빼기 점검 불가(거래 부족)")
    if y["years_share"] is None or y["years_share"] < cf["YEAR_AGREE"]:
        rs.append(f"해별 방향 일치 {y['years_ok']}/{y['years_n']} < 3분의 2")
    return {"candidate": not rs, "reasons": rs}


def tr_judge_verify(s: Dict[str, Any]) -> Dict[str, Any]:
    cf = TR_CONFIG
    rs = []
    if s["n"] < cf["VERIFY_MIN_TRADES"]:
        rs.append(f"거래 {s['n']:,} < {cf['VERIFY_MIN_TRADES']}")
    if s["diff"] is None or s["diff"] <= 0:
        rs.append("차이 ≤ 0")
    if s["t"] is None or not np.isfinite(s["t"]) or s["t"] < cf["VERIFY_Z"]:
        rs.append(f"t < {cf['VERIFY_Z']:.0f}")
    if s.get("net") is None or s["net"] <= 0:
        rs.append("비용 뒤 거래당 순수익 ≤ 0")
    return {"passed": not rs, "reasons": rs}


def tr_judge_final(s: Dict[str, Any]) -> Dict[str, Any]:
    cf = TR_CONFIG
    rs = []
    if s["n"] < cf["FINAL_MIN_TRADES"]:
        rs.append(f"거래 {s['n']:,} < {cf['FINAL_MIN_TRADES']}")
    if s["diff"] is None or s["diff"] <= 0:
        rs.append("차이 ≤ 0")
    if s["t"] is None or not np.isfinite(s["t"]) or s["t"] < cf["FINAL_Z"]:
        rs.append(f"t < {cf['FINAL_Z']:.0f}")
    if s.get("net") is None or s["net"] <= 0:
        rs.append("비용 뒤 거래당 순수익 ≤ 0")
    if _tr_market_conflict(s):
        rs.append("미국·한국 방향이 다름")
    return {"passed": not rs, "reasons": rs}


# ---------------------------------------------------------------------
# 실행 전: 깔때기 · T1 결정 · 가려낼 수 있는 최소 차이 (수익 안 봄)
# ---------------------------------------------------------------------
def tr_noise(prepared: Dict[str, Dict[str, Any]], start, end, seed: Optional[int] = None) -> Dict[int, Optional[float]]:
    """같은 종목 무작위 h일 보유(시가 → h번째 날 종가) − 같은 종목 무작위 20곳 평균의 표준편차. 신호 결과는 보지 않는다."""
    cf = TR_CONFIG
    rng = np.random.default_rng((cf["SEED"] if seed is None else seed) + 99)
    arrs = []
    for sym in sorted(prepared):
        b = prepared[sym]["b"]
        lo, hi = tr_bounds(b.index, start, end)
        if hi - lo > 300:
            arrs.append((_lec_arr(b, "Open"), _lec_arr(b, "Close"), lo, hi))
    out: Dict[int, Optional[float]] = {}
    if not arrs:
        return {h: None for h in cf["MDE_HOLDS"]}
    for h in cf["MDE_HOLDS"]:
        xs = []
        for _ in range(cf["MDE_DRAWS"]):
            o, c, lo, hi = arrs[int(rng.integers(len(arrs)))]
            r = rng.integers(lo + 1, hi - h + 2, size=cf["BASELINE_SAMPLES"] + 1)
            v = c[r + h - 1] / o[r] - 1.0
            xs.append(v[0] - v[1:].mean())
        out[h] = float(np.std(xs, ddof=1))
    return out


def tr_mde_table(noise: Dict[int, Optional[float]], n: int, z: float) -> Dict[int, Dict[str, Optional[float]]]:
    cf = TR_CONFIG
    out = {}
    for h, sd in noise.items():
        if sd is None or n <= 0:
            out[h] = {"indep": None, "cluster": None}
            continue
        v = (z + cf["POWER_Z"]) * sd / math.sqrt(n)
        out[h] = {"indep": v, "cluster": v * cf["MDE_CLUSTER"]}
    return out


def tr_absorb_decision(funnel_a: Dict[str, Any]) -> Dict[str, Any]:
    n = int(funnel_a["with_absorb"])
    use = n >= TR_CONFIG["ABSORB_MIN_SIGNALS"]
    return {"use": use, "n": n, "text": (f"넣음 — 규칙 A + 물량 소화 신호 {n:,}건 ≥ {TR_CONFIG['ABSORB_MIN_SIGNALS']}" if use else
                                         f"뺌(참고 줄) — 규칙 A + 물량 소화 신호 {n:,}건 < {TR_CONFIG['ABSORB_MIN_SIGNALS']}")}


def run_tr_count(prepared: Dict[str, Dict[str, Any]], grades: Dict[str, pd.Series],
                 progress: Optional[Callable[[str, int, int], None]] = None, ctx: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cf = TR_CONFIG
    if ctx is None:
        ctx = tr_contexts(prepared, grades, progress=(lambda i, n: progress("신호", i, n)) if progress else None)
    fun = {r: tr_funnel(ctx, prepared, r, cf["EXPLORE_START"], cf["EXPLORE_END"]) for r in TR_RULES}
    dec = tr_absorb_decision(fun["A"])
    noise = tr_noise(prepared, cf["EXPLORE_START"], cf["EXPLORE_END"])
    mde = {}
    for r in TR_RULES:
        n = fun[r]["with_absorb"] if dec["use"] else fun[r]["no_absorb"]
        mde[r] = {"n_signals": int(n), "table": tr_mde_table(noise, int(n), cf["EXPLORE_Z"])}
    return {"config_version": cf["VERSION"], "funnel": fun, "absorb": dec, "noise": noise, "mde": mde, "n_tickers": len(prepared)}


# ---------------------------------------------------------------------
# 실행: 탐색 → (후보만) 확인
# ---------------------------------------------------------------------
def _tr_sub(tr: pd.DataFrame, mask) -> Dict[str, Any]:
    s = tr_stats(tr[np.asarray(mask, dtype=bool)]) if len(tr) else tr_stats(tr)
    return {"n": s["n"], "diff": s["diff"], "t": s["t"], "net": s["net"]}


def _tr_slim(s: Dict[str, Any]) -> Dict[str, Any]:
    return {"n": s["n"], "diff": s["diff"], "t": s["t"], "net": s["net"]}


def run_tr(prepared: Dict[str, Dict[str, Any]], grades: Dict[str, pd.Series],
           progress: Optional[Callable[[str, int, int], None]] = None) -> Dict[str, Any]:
    cf = TR_CONFIG
    ctx = tr_contexts(prepared, grades, progress=(lambda i, n: progress("신호", i, n)) if progress else None)
    cnt = run_tr_count(prepared, grades, ctx=ctx)
    use_abs = cnt["absorb"]["use"]
    es, ee = cf["EXPLORE_START"], cf["EXPLORE_END"]
    res: Dict[str, Any] = {"config_version": cf["VERSION"], "count": cnt, "use_absorb": use_abs, "explore": {}, "verify": {},
                           "candidates": [], "verified": [], "n_tickers": len(prepared)}
    for i, rule in enumerate(TR_RULES):
        tr = tr_simulate(ctx, prepared, rule, es, ee, use_abs)
        s = tr_stats(tr)
        j = tr_judge_explore(s)
        refs = {"green": _tr_sub(tr, tr["grade"] == "🟢") if len(tr) else _tr_slim(tr_stats(tr)),
                "madb": _tr_sub(tr, tr["madb"]) if len(tr) else _tr_slim(tr_stats(tr)),
                "simple": _tr_slim(tr_stats(tr_simulate(ctx, prepared, rule, es, ee, use_abs, simple=True)))}
        if not use_abs:
            refs["absorb_only"] = _tr_sub(tr, tr["absorb"]) if len(tr) else _tr_slim(tr_stats(tr))
        abl = {}
        for k in TR_ABLATE:
            if k == "absorb" and not use_abs:
                continue
            abl[k] = _tr_slim(tr_stats(tr_simulate(ctx, prepared, rule, es, ee, use_abs, drop=k)))
        res["explore"][rule] = {"stats": s, "judge": j, "mde": tr_mde(s, cf["EXPLORE_Z"]), "refs": refs, "ablate": abl}
        if j["candidate"]:
            res["candidates"].append(rule)
        if progress:
            progress("탐색", i + 1, len(TR_RULES))
    for rule in res["candidates"]:
        tr = tr_simulate(ctx, prepared, rule, cf["VERIFY_START"], None, use_abs, seed=cf["SEED"] + 1)
        s = tr_stats(tr)
        j = tr_judge_verify(s)
        res["verify"][rule] = {"stats": s, "judge": j, "mde": tr_mde(s, cf["VERIFY_Z"])}
        if j["passed"]:
            res["verified"].append(rule)
    res["verify_status"] = ("확인 단계 안 엶(탐색 후보 없음)" if not res["candidates"] else
                            f"확인: 후보 {len(res['candidates'])}개 중 통과 {len(res['verified'])}개")
    hold = []
    for sym, p in prepared.items():
        b = p["b"]
        lo, hi = tr_bounds(b.index, es, ee)
        if hi - lo > 250:
            c = _lec_arr(b, "Close")
            hold.append((c[hi] / c[lo]) ** (365.25 / max((b.index[hi] - b.index[lo]).days, 1)) - 1.0)
    res["buy_hold_cagr_mean"] = float(np.mean(hold)) if hold else None
    return res


def run_tr_final(prepared: Dict[str, Dict[str, Any]], grades: Dict[str, pd.Series], rules: List[str], use_absorb: bool,
                 progress: Optional[Callable[[str, int, int], None]] = None) -> Dict[str, Any]:
    cf = TR_CONFIG
    ctx = tr_contexts(prepared, grades, progress=(lambda i, n: progress("신호", i, n)) if progress else None)
    out: Dict[str, Any] = {"config_version": cf["VERSION"], "use_absorb": use_absorb, "final": {}, "passed": [], "n_tickers": len(prepared)}
    for rule in rules:
        tr = tr_simulate(ctx, prepared, rule, cf["FINAL_START"], None, use_absorb, seed=cf["SEED"] + 2)
        s = tr_stats(tr)
        j = tr_judge_final(s)
        out["final"][rule] = {"stats": s, "judge": j, "mde": tr_mde(s, cf["FINAL_Z"])}
        if j["passed"]:
            out["passed"].append(rule)
    return out


# ---------------------------------------------------------------------
# 보고서
# ---------------------------------------------------------------------
def _tr_pp(v: Optional[float], d: int = 2) -> str:
    return "-" if v is None or not np.isfinite(v) else f"{v * 100:+.{d}f}%"


def _tr_t(v: Optional[float]) -> str:
    return "-" if v is None or not np.isfinite(v) else f"{v:+.2f}"


def _tr_funnel_lines(cnt: Dict[str, Any]) -> List[str]:
    fun, dec = cnt["funnel"], cnt["absorb"]
    key = "with" if dec["use"] else "without"
    lines = ["| 단계(사용자 순서) | " + " | ".join(TR_LABELS[r] for r in TR_RULES) + " |", "|---|" + "---|" * len(TR_RULES)]
    for k, lab in TR_FUNNEL:
        if k in fun[TR_RULES[0]][key]:
            lines.append(f"| {lab} | " + " | ".join(f"{fun[r][key][k]:,}" for r in TR_RULES) + " |")
    other = "without" if dec["use"] else "with"
    lines += ["", f"- T1 물량 소화: **{dec['text']}**(사전 등록 4장 규칙, 코드가 자동으로 정함). 판정 사슬은 위 표.",
              f"- 참고 · 물량 소화를 {'빼면' if dec['use'] else '넣으면'} ①~⑧ 통과: " + " · ".join(f"{r} {fun[r][other]['box']:,}" for r in TR_RULES),
              "- 실적일 확인 불가(박스 시작이 실적일 자료보다 앞섬 — 제외하지 않음, ③까지 통과한 신호 중): " +
              " · ".join(f"{r} {fun[r]['earn_unknown']:,}/{fun[r]['earn_reach']:,}" for r in TR_RULES),
              "- 신호 수다. 실제 거래는 보유 중에 나온 신호를 건너뛰어 더 적다."]
    return lines


def render_tr_count_report(r: Dict[str, Any], generated_at: Optional[str] = None) -> str:
    cf = TR_CONFIG
    lines = [f"# H-TR 추세 투자 흐름 — 실행 전 건수 (수익은 보지 않음)", "",
             f"- 생성: {generated_at or '-'} · 설정 {r['config_version']} · 545종목 중 재료 {r['n_tickers']}개 · 신호 {cf['EXPLORE_START'][:4]}~{cf['EXPLORE_END'][:4]}",
             "- 정의는 data/backtest/TR_PREREG.md. 이 숫자를 보고 규칙을 바꾸지 않는다.", "", "## 깔때기(신호 수)", ""]
    lines += _tr_funnel_lines(r)
    lines += ["", "## 가려낼 수 있는 최소 차이(검정력 80%, t ≥ 2)", "",
              "같은 종목 무작위 보유(시가 → h번째 날 종가)와 같은 종목 무작위 20곳 평균의 차이 흩어짐으로 계산. 앞 숫자 = 거래가 서로 독립일 때, 괄호 = 같은 달 몰림을 감안한 대략(×1.3).", "",
              "| 규칙 | 신호 수 | " + " | ".join(f"{h}일 보유 가정" for h in cf["MDE_HOLDS"]) + " |", "|---|---|" + "---|" * len(cf["MDE_HOLDS"])]
    for rule in TR_RULES:
        m = r["mde"][rule]
        cells = []
        for h in cf["MDE_HOLDS"]:
            x = m["table"][h]
            cells.append("-" if x["indep"] is None else f"{x['indep'] * 100:.1f}%p ({x['cluster'] * 100:.1f}%p)")
        lines.append(f"| {TR_LABELS[rule]} | {m['n_signals']:,} | " + " | ".join(cells) + " |")
    nz = r["noise"]
    lines += ["", "- 흩어짐(표준편차): " + " · ".join(f"{h}일 {('-' if v is None else f'{v * 100:.1f}%')}" for h, v in nz.items()),
              "- 지금까지 보인 차이(L2 +0.2%p 안팎, 틀 A +0.3%p)가 이 숫자보다 작으면, 미통과는 '효과 없음'이 아니라 '이만큼 큰 효과는 없음'으로 읽는다.",
              "", "다음: 숫자를 확인한 뒤 `python run_backtest.py --tr` (탐색 → 후보만 2014~ 확인)."]
    return "\n".join(lines) + "\n"


def _tr_stat_row(name: str, s: Dict[str, Any], mde: Optional[float], verdict: str) -> str:
    return (f"| {name} | {s['n']:,} · {s['n_tickers']} | {_tr_pp(s.get('net'))} | {_tr_pp(s.get('base'))} | {_tr_pp(s.get('diff'))} | "
            f"{_tr_t(s.get('t'))} | {'-' if mde is None else f'{mde * 100:.1f}%p'} | {verdict} |")


def render_tr_report(r: Dict[str, Any], generated_at: Optional[str] = None) -> str:
    cf = TR_CONFIG
    cand, ver = r["candidates"], r["verified"]
    head = ("통과(확인까지) — 최종 단계 대기: " + ", ".join(ver)) if ver else (("확인 미통과" if cand else "탐색 후보 없음") + " — 미통과")
    lines = [f"# H-TR 추세 투자 흐름 검증 — 결과", "",
             f"- 생성: {generated_at or '-'} · 설정 {r['config_version']} · 종목 {r['n_tickers']}개 · 정의 data/backtest/TR_PREREG.md",
             f"- 흐름: ① 시장 🟢·🟡 → ② RS 주도 → ③ 120일선 상승 → ④ 진입(A 돌파 확정 / B SR Flip) → ⑤ 물량 소화({'넣음' if r['use_absorb'] else '참고'}) "
             "→ ⑥ 실적 쇼크 제외 → ⑦ 손절폭 ≤ 12% → ⑧ 박스 1대1 목표 · 추세선 장대음봉 이탈 절반 · 붕괴·무효 전부 · 최대 250거래일. 비용 0.30%.",
             "- 판정 비교: 같은 종목 무작위 날 20곳에서 같은 날 간격·비율로 판 순수익(보유 기간 맞춤). t = 월 묶음.", "",
             f"## 판정: {head}", ""]
    lines += ["| 규칙 | 거래 · 종목 | 거래당 순수익 | 무작위(같은 보유) | 차이 | t | 가려낼 수 있는 최소 차이 | 판정 |", "|---|---|---|---|---|---|---|---|"]
    for rule in TR_RULES:
        e = r["explore"][rule]
        lines.append(_tr_stat_row(f"탐색 {TR_LABELS[rule]}", e["stats"], e["mde"], "후보" if e["judge"]["candidate"] else "후보 아님"))
    for rule, v in r["verify"].items():
        lines.append(_tr_stat_row(f"확인 {TR_LABELS[rule]}", v["stats"], v["mde"], "통과" if v["judge"]["passed"] else "미통과"))
    lines.append("")
    for rule in TR_RULES:
        e = r["explore"][rule]
        s = e["stats"]
        lines += [f"### {TR_LABELS[rule]} (탐색 {cf['EXPLORE_START'][:4]}~{cf['EXPLORE_END'][:4]})", ""]
        if e["judge"]["reasons"]:
            lines.append("- 후보 아닌 이유: " + " · ".join(e["judge"]["reasons"]))
        if not s["n"]:
            lines += ["- 거래 없음", ""]
            continue
        y = s["years"]
        ytxt = (f"기여가 가장 큰 해 {y['drop_year']} 빼면 차이 {_tr_pp(y['drop_diff'])} (t {_tr_t(y['drop_t'])})" if y["drop_year"] is not None
                else "해 빼기 점검 불가(거래 부족)")
        lines += [f"- 승률 {s['win'] * 100:.0f}% · 평균 보유 {s['hold']:.0f}일(중앙 {s['hold_med']:.0f}) · 손절폭 중앙 {s['stop_med'] * 100:.1f}% · 박스 폭 중앙 {s['width_med'] * 100:.1f}% · 절반 매도가 있었던 거래 {s['half_share'] * 100:.0f}%",
                  "- 청산 이유(비중 · 거래당 순수익): " + " · ".join(f"{k} {v * 100:.0f}% ({_tr_pp(s['reason_net'].get(k))})" for k, v in sorted(s["reasons"].items(), key=lambda z: -z[1])),
                  "- 시장별: " + " · ".join(f"{m} {v['n']:,}건 차이 {_tr_pp(v['diff'])} (t {_tr_t(v['t'])})" for m, v in s["by_market"].items()),
                  f"- 해: {ytxt} · 거래 30건 이상인 해 중 차이 > 0: {y['years_ok']}/{y['years_n']}",
                  f"- 참고 · 셋업만(①②③이 맞는 아무 날, 같은 보유) 대비 차이 {_tr_pp(s['setup']['diff'])} (t {_tr_t(s['setup']['t'])}) — 진입 신호가 보태는 몫",
                  f"- 참고 · 1% 위험 수량(9-6): 비중 중앙 {s['acct']['weight_med'] * 100:.0f}% · 거래당 계좌 손익 {_tr_pp(s['acct']['per_trade'], 3)} · 연 합계 {_tr_pp(s['acct']['per_year'])}(겹침·복리 무시)"]
        rf = e["refs"]
        ref_txt = [f"🟢만 {rf['green']['n']:,}건 {_tr_pp(rf['green']['diff'])} (t {_tr_t(rf['green']['t'])})",
                   f"이평 쌍바닥이 앞에 있던 거래 {rf['madb']['n']:,}건 {_tr_pp(rf['madb']['diff'])} (t {_tr_t(rf['madb']['t'])})",
                   f"청산 단순형 {rf['simple']['n']:,}건 {_tr_pp(rf['simple']['diff'])} (t {_tr_t(rf['simple']['t'])})"]
        if "absorb_only" in rf:
            ref_txt.append(f"물량 소화 있던 거래 {rf['absorb_only']['n']:,}건 {_tr_pp(rf['absorb_only']['diff'])} (t {_tr_t(rf['absorb_only']['t'])})")
        lines.append("- 참고: " + " · ".join(ref_txt))
        lines.append("- 참고 · 단계 하나씩 빼기: " + " · ".join(f"{TR_ABLATE[k]} {v['n']:,}건 {_tr_pp(v['diff'])} (t {_tr_t(v['t'])})" for k, v in e["ablate"].items()))
        lines.append("")
    for rule, v in r["verify"].items():
        if v["judge"]["reasons"]:
            lines.append(f"- 확인 {rule} 미통과 이유: " + " · ".join(v["judge"]["reasons"]))
    lines += ["", "## 실행 전 깔때기(신호 수)", ""] + _tr_funnel_lines(r["count"])
    if r.get("buy_hold_cagr_mean") is not None:
        lines += ["", f"- 참고: 시험 종목을 탐색 기간 내내 들고 있었을 때 연 수익(종목 평균) {_tr_pp(r['buy_hold_cagr_mean'])}"]
    lines += ["", f"- {r['verify_status']}. 결과를 보고 규칙·숫자를 바꾸지 않는다(TR_PREREG.md)."]
    return "\n".join(lines) + "\n"


def render_tr_final_report(r: Dict[str, Any], dropped: List[str], generated_at: Optional[str] = None) -> str:
    lines = [f"# H-TR 최종 (새 151종목)", "", f"- 생성: {generated_at or '-'} · 설정 {r['config_version']} · 종목 {r['n_tickers']}개 · 못 받은 종목 {len(dropped)}개",
             "", "| 규칙 | 거래 · 종목 | 거래당 순수익 | 무작위(같은 보유) | 차이 | t | 가려낼 수 있는 최소 차이 | 판정 |", "|---|---|---|---|---|---|---|---|"]
    for rule, v in r["final"].items():
        lines.append(_tr_stat_row(TR_LABELS[rule], v["stats"], v["mde"], "통과" if v["judge"]["passed"] else "미통과"))
        if v["judge"]["reasons"]:
            lines.append(f"| ↳ 이유 | {' · '.join(v['judge']['reasons'])} | | | | | | |")
    return "\n".join(lines) + "\n"


def tr_log_lines(r: Dict[str, Any], generated_at: str) -> List[str]:
    out = []
    for rule in TR_RULES:
        e = r["explore"][rule]
        s = e["stats"]
        out.append(f"- {generated_at} H-TR {rule} 탐색: {s['n']:,}건 차이 {_tr_pp(s.get('diff'))} t {_tr_t(s.get('t'))} → "
                   f"{'후보' if e['judge']['candidate'] else '후보 아님'}")
    for rule, v in r["verify"].items():
        out.append(f"- {generated_at} H-TR {rule} 확인: {v['stats']['n']:,}건 차이 {_tr_pp(v['stats'].get('diff'))} t {_tr_t(v['stats'].get('t'))} → "
                   f"{'통과' if v['judge']['passed'] else '미통과'}")
    return out


def tr_final_log_lines(r: Dict[str, Any], generated_at: str) -> List[str]:
    return [f"- {generated_at} H-TR {rule} 최종: {v['stats']['n']:,}건 차이 {_tr_pp(v['stats'].get('diff'))} t {_tr_t(v['stats'].get('t'))} → "
            f"{'통과' if v['judge']['passed'] else '미통과'}" for rule, v in r["final"].items()]
