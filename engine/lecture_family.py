"""강의 조건 시험 — 2단계: 큰 가족 판정(섞기 200회 통과선 · PBO · 검증)과 L2 실행 (2026-11-18 사전 등록, AGENTS.md 4장).

큰 가족(L2 지금, L4·L5 다음)은 조합이 많아서 '가장 좋아 보이는 하나'가 우연히 좋을 수 있다. 그래서 (사전등록안 8장)
    통과선  같은 종목 안에서 사건 날짜를 무작위 날로 바꿔 가족 전체를 200번 다시 계산하고, 매번 '가족 안 최고 t'를 모아
            그 95% 분위를 통과선으로 쓴다(White Reality Check 방식). 한 번 바꿀 때 같은 사건(종목·날짜)은 모든 조합에서 같은 날로
            바뀐다 — 조합끼리 겹치는 정도를 그대로 둬야 통과선이 공정하다.
    t       10거래일 초과수익(같은 날 같은 시장 종목 평균을 뺌)의 평균 ÷ 월 묶음 표준오차. 비용은 t에 넣지 않고 '방향' 조건에 넣는다:
            후보는 평균 − 왕복 비용 0.30% > 0 이어야 한다(돈이 되는 방향). 무작위 날과 실제 사건을 같은 식으로 재야 비교가 되기 때문.
    가족    사건 ≥ 300 · 종목 ≥ 50인 조합만 최고 t 경쟁에 넣는다(사건 수는 결과와 무관하게 정해지므로 고르는 데 써도 된다).
    PBO     탐색 기간을 16조각으로 나눠 8조각에서 가장 좋았던 조합이 나머지 8조각에서 중앙보다 아래인 비율(참고, Bailey 등).
    검증    후보인 조합만 2014년 이후를 계산해 보여 준다. 같은 방향(평균 − 비용 > 0) + t ≥ 2.
"""
import math
import os
from itertools import combinations
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .explore import EXPLORE_CONFIG
from .frame_backtest import compare_clustered
from .lecture_backtest import (_lec_pct, lec_clean, lec_forward_returns, lec_index_of, lec_market_of, lec_sector_etf)
from .lecture_events import (L2_CONFS, L2_ENVS, L2_EVENTS, L2_REF_PAIRS, lec_etf_below_high, lec_l2_combos, lec_l2_events)
from .lecture_signals import LECTURE_CONFIG, lec_base, lec_swings


# ---------------------------------------------------------------------
# 통계 도우미
# ---------------------------------------------------------------------
def lec_month_code(dates) -> np.ndarray:
    d = pd.DatetimeIndex(pd.to_datetime(dates))
    return (d.year * 12 + d.month - 1).to_numpy(dtype=np.int64)


def lec_cluster_t(x: np.ndarray, g: np.ndarray) -> Dict[str, Any]:
    """한 표본 평균과 월 묶음 표준오차(frame_backtest.cluster_stats와 같은 식). t = 평균 ÷ 표준오차."""
    x = np.asarray(x, dtype=float)
    g = np.asarray(g)
    m = np.isfinite(x)
    x, g = x[m], g[m]
    n = len(x)
    if n < 2:
        return {"n": int(n), "mean": float(x.mean()) if n else None, "se": None, "t": None, "k": 0}
    mean = float(x.mean())
    _, inv = np.unique(g, return_inverse=True)
    sums = np.bincount(inv, weights=x - mean)
    k = len(sums)
    if k < 2:
        return {"n": n, "mean": mean, "se": None, "t": None, "k": k}
    se = math.sqrt(k / (k - 1) * float(np.sum(sums ** 2))) / n
    return {"n": n, "mean": mean, "se": se, "t": (mean / se if se > 0 else None), "k": k}


def _lec_fast_t(x: np.ndarray, inv: np.ndarray, k: int) -> float:
    n = len(x)
    if n < 2 or k < 2:
        return np.nan
    mean = x.mean()
    sums = np.bincount(inv, weights=x - mean, minlength=k)
    se = math.sqrt(k / (k - 1) * float(np.dot(sums, sums))) / n
    return mean / se if se > 0 else np.nan


def lec_dedupe(t: np.ndarray, gap: int) -> np.ndarray:
    """정렬된 사건 위치에서 앞 사건 뒤 gap 거래일 안의 사건을 뺀 자리(불리언)."""
    keep = np.zeros(len(t), dtype=bool)
    free = -10 ** 9
    for i, x in enumerate(t):
        if x >= free:
            keep[i] = True
            free = x + gap + 1
    return keep


def lec_select(ev: pd.DataFrame, combo: Dict[str, Optional[str]], value: str) -> Dict[str, Any]:
    """조합의 사건 줄 번호. 조건 열이 NaN(자료 없음)이면 빠지고 개수만 센다. 같은 종목·같은 조합은 10거래일 안의 새 사건을 세지 않는다."""
    m = (ev["event"].to_numpy() == combo["event"]) & (ev["missing"].to_numpy() != 1.0)
    n_ev = int(m.sum())
    miss = np.zeros(len(ev), dtype=bool)
    for k in ("env", "conf"):
        if combo.get(k):
            col = ev[combo[k]].to_numpy(dtype=float)
            miss |= m & ~np.isfinite(col)
            m &= col == 1.0
    idx = np.flatnonzero(m)
    if len(idx):
        sub = ev.iloc[idx]
        keep = np.zeros(len(idx), dtype=bool)
        for _, pos in sub.groupby("ticker", sort=False).indices.items():
            keep[pos] = lec_dedupe(sub["t"].to_numpy()[pos], LECTURE_CONFIG["GAP"])
        idx = idx[keep]
        idx = idx[np.isfinite(ev[value].to_numpy(dtype=float)[idx])]
    return {"idx": idx, "n_event_rows": n_ev, "n_missing": int(miss.sum())}


def _lec_month_drop_one(x: np.ndarray, g: np.ndarray, cost: float) -> Dict[str, Any]:
    """달 쏠림 점검(N3, 한 표본): '평균 − 비용'에 가장 크게 기여한 달을 빼고 다시 잰다."""
    if len(x) < 3:
        return {"month": None, "net": None, "t": None}
    s = pd.Series(x - cost).groupby(g).sum()
    worst = s.idxmax()
    keep = g != worst
    r = lec_cluster_t(x[keep], g[keep])
    yy, mm = divmod(int(worst), 12)
    return {"month": f"{yy}-{mm + 1:02d}", "net": (r["mean"] - cost) if r["mean"] is not None else None, "t": r["t"]}


def lec_combo_stats(ev: pd.DataFrame, idx: np.ndarray, value: str, cost: float) -> Dict[str, Any]:
    sub = ev.iloc[idx]
    x = sub[value].to_numpy(dtype=float)
    g = lec_month_code(sub["date"])
    r = lec_cluster_t(x, g)
    out = {"n": r["n"], "mean": r["mean"], "net": (r["mean"] - cost) if r["mean"] is not None else None, "t": r["t"],
           "n_months": r["k"], "n_tickers": int(sub["ticker"].nunique()) if len(sub) else 0, "by_market": {}}
    for mk in ("US", "KR"):
        mm = sub["market"].to_numpy() == mk
        rr = lec_cluster_t(x[mm], g[mm])
        out["by_market"][mk] = {"n": rr["n"], "net": (rr["mean"] - cost) if rr["mean"] is not None else None, "t": rr["t"]}
    out["month_drop"] = _lec_month_drop_one(x, g, cost)
    return out


def lec_judge_one(s: Dict[str, Any], threshold: float) -> Dict[str, Any]:
    """탐색 후보(사전등록안 8장): t ≥ 통과선, 평균 − 비용 > 0, 사건 ≥ 300, 종목 ≥ 50, 미국·한국 같은 방향(둘 다 ≥ 30),
    기여가 가장 큰 달을 빼도 평균 − 비용 > 0 이고 t ≥ 2."""
    c = LECTURE_CONFIG
    rs = []
    if s["n"] < c["MIN_EVENTS"]:
        rs.append(f"사건 {s['n']:,} < {c['MIN_EVENTS']}")
    if s["n_tickers"] < c["MIN_TICKERS"]:
        rs.append(f"종목 {s['n_tickers']} < {c['MIN_TICKERS']}")
    if s["t"] is None or not np.isfinite(s["t"]) or s["t"] < threshold:
        rs.append(f"t < 통과선 {threshold:.2f}")
    if s["net"] is None or s["net"] <= 0:
        rs.append("비용 뒤 평균 ≤ 0")
    mk = [v for v in s["by_market"].values() if v["n"] >= c["MIN_MARKET_EVENTS"] and v["net"] is not None]
    if len(mk) == 2 and np.sign(mk[0]["net"]) != np.sign(mk[1]["net"]):
        rs.append("미국·한국 방향이 다름")
    md = s["month_drop"]
    if md["net"] is None or md["net"] <= 0 or md["t"] is None or md["t"] < c["MONTH_DROP_Z"]:
        rs.append(f"기여가 가장 큰 달({md['month']})을 빼면 기준 미달")
    return {"candidate": not rs, "reasons": rs}


# ---------------------------------------------------------------------
# 섞기 통과선 · PBO
# ---------------------------------------------------------------------
def lec_null_pool(prepared_values: Dict[str, pd.Series], period: Sequence[pd.Timestamp]) -> Dict[str, Any]:
    """무작위 날 후보: 종목마다 기간 안에서 결과값이 있는 날 전부."""
    s0, s1 = pd.Timestamp(period[0]), pd.Timestamp(period[1])
    tick, vals, mons, offs, sizes = {}, [], [], [], []
    off = 0
    for i, (sym, ser) in enumerate(prepared_values.items()):
        s = ser[(ser.index >= s0) & (ser.index <= s1)].dropna()
        tick[sym] = i
        offs.append(off)
        sizes.append(len(s))
        vals.append(s.to_numpy(dtype=float))
        mons.append(lec_month_code(s.index))
        off += len(s)
    return {"tick": tick, "vals": np.concatenate(vals) if vals else np.array([]),
            "mons": np.concatenate(mons) if mons else np.array([], dtype=np.int64),
            "offs": np.array(offs, dtype=np.int64), "sizes": np.array(sizes, dtype=np.int64)}


def lec_reality_check(ev: pd.DataFrame, sel: Dict[str, np.ndarray], pool: Dict[str, Any], reps: int, seed: int,
                      progress: Optional[Callable[[int, int], None]] = None) -> Dict[str, Any]:
    """sel = {조합: 줄 번호}. 매번 사건(줄)마다 같은 종목의 무작위 날 하나를 뽑아(모든 조합에 같은 날) 조합별 t → 최고 t."""
    if not sel:
        return {"max_t": [], "threshold": None}
    rows = np.unique(np.concatenate(list(sel.values())))
    pos = {r: i for i, r in enumerate(rows)}
    tid = np.array([pool["tick"].get(s, -1) for s in ev["ticker"].to_numpy()[rows]], dtype=np.int64)
    ok = (tid >= 0)
    sizes = np.where(ok, pool["sizes"][np.maximum(tid, 0)], 0)
    ok &= sizes > 0
    maps = {k: np.array([pos[r] for r in v], dtype=np.int64) for k, v in sel.items()}
    rng = np.random.default_rng(seed)
    mx = []
    msum = {k: 0.0 for k in maps}
    mcnt = {k: 0 for k in maps}
    for rep in range(reps):
        u = rng.random(len(rows))
        j = pool["offs"][np.maximum(tid, 0)] + np.floor(u * np.maximum(sizes, 1)).astype(np.int64)
        xv = np.where(ok, pool["vals"][np.minimum(j, len(pool["vals"]) - 1)] if len(pool["vals"]) else np.nan, np.nan)
        gv = np.where(ok, pool["mons"][np.minimum(j, len(pool["mons"]) - 1)] if len(pool["mons"]) else 0, 0)
        best = -np.inf
        for k, mp in maps.items():
            x = xv[mp]
            g = gv[mp]
            f = np.isfinite(x)
            x, g = x[f], g[f]
            if len(x) < 2:
                continue
            msum[k] += float(x.mean())
            mcnt[k] += 1
            _, inv = np.unique(g, return_inverse=True)
            tt = _lec_fast_t(x, inv, int(inv.max()) + 1)
            if np.isfinite(tt) and tt > best:
                best = tt
        mx.append(float(best) if np.isfinite(best) else np.nan)
        if progress:
            progress(rep + 1, reps)
    arr = np.array([v for v in mx if np.isfinite(v)])
    thr = float(np.quantile(arr, LECTURE_CONFIG["RC_Q"])) if len(arr) else None
    return {"max_t": mx, "threshold": thr, "null_mean": {k: (msum[k] / mcnt[k] if mcnt[k] else None) for k in maps}}


def lec_pbo(ev: pd.DataFrame, sel: Dict[str, np.ndarray], value: str, period: Sequence[pd.Timestamp], slices: int) -> Dict[str, Any]:
    """CSCV 과적합 확률(참고): 기간을 같은 길이 16조각 → 8조각 고르는 모든 경우(12,870)에서 고른 쪽 평균 1등 조합의
    나머지 쪽 상대 순위 ω, λ = log(ω/(1−ω)). PBO = λ ≤ 0 비율."""
    keys = list(sel)
    if len(keys) < 2:
        return {"pbo": None, "n_splits": 0, "n_combos": len(keys)}
    s0, s1 = pd.Timestamp(period[0]), pd.Timestamp(period[1])
    edges = pd.date_range(s0, s1 + pd.Timedelta(days=1), periods=slices + 1)
    S = np.zeros((slices, len(keys)))
    N = np.zeros((slices, len(keys)))
    for j, k in enumerate(keys):
        sub = ev.iloc[sel[k]]
        sl = np.clip(np.searchsorted(edges.values, pd.to_datetime(sub["date"]).values, "right") - 1, 0, slices - 1)
        x = sub[value].to_numpy(dtype=float)
        S[:, j] = np.bincount(sl, weights=x, minlength=slices)
        N[:, j] = np.bincount(sl, minlength=slices)
    lam = []
    for comb in combinations(range(slices), slices // 2):
        isx = np.zeros(slices, dtype=bool)
        isx[list(comb)] = True
        with np.errstate(invalid="ignore", divide="ignore"):
            mi = S[isx].sum(0) / N[isx].sum(0)
            mo = S[~isx].sum(0) / N[~isx].sum(0)
        if not np.any(np.isfinite(mi)):
            continue
        best = int(np.nanargmax(np.where(np.isfinite(mi), mi, -np.inf)))
        if not np.isfinite(mo[best]):
            continue
        valid = np.isfinite(mo)
        rank = float(np.sum(mo[valid] < mo[best]) + 0.5 * (np.sum(mo[valid] == mo[best]) - 1) + 1)
        w = rank / (valid.sum() + 1)
        lam.append(math.log(w / (1 - w)))
    lam = np.array(lam)
    return {"pbo": float(np.mean(lam <= 0)) if len(lam) else None, "n_splits": int(len(lam)), "n_combos": len(keys)}


# ---------------------------------------------------------------------
# L2 표 만들기
# ---------------------------------------------------------------------
def lec_market_means(prepared: Dict[str, Dict[str, Any]], h: int) -> Dict[str, pd.Series]:
    """같은 날 같은 시장(미국·한국) 종목 전체의 '다음 날 시가 → h거래일 뒤 종가' 평균."""
    fw: Dict[str, List[pd.Series]] = {"US": [], "KR": []}
    for sym, p in prepared.items():
        fw[lec_market_of(sym)].append(lec_forward_returns(p["b"], h).rename(sym))
    return {mk: (pd.concat(v, axis=1).mean(axis=1, skipna=True) if v else pd.Series(dtype=float)) for mk, v in fw.items()}


def lec_etf_context(bench_map: Dict[str, pd.DataFrame]) -> Dict[str, pd.Series]:
    out = {}
    for sym, df in bench_map.items():
        d = lec_clean(df)
        if d is None or len(d) < 30:
            continue
        b = lec_base(d)
        out[sym] = lec_etf_below_high(b, lec_swings(b))
    return out


def lec_l2_table(prepared: Dict[str, Dict[str, Any]], bench_map: Dict[str, pd.DataFrame], grades: Dict[str, pd.Series],
                 progress: Optional[Callable[[int, int], None]] = None) -> Dict[str, Any]:
    """모든 종목의 L2 사건 표 + 결과(x10 초과수익, 5·20일 참고, E13용 10일 최대 상승·하락) + 무작위 날 후보용 종목별 x10."""
    c = LECTURE_CONFIG
    H = c["HORIZON"]
    hz_all = (H,) + tuple(c["HORIZON_REF"])
    means = {h: lec_market_means(prepared, h) for h in hz_all}
    etfb = lec_etf_context(bench_map)
    evs, xser = [], {}
    for i, (sym, p) in enumerate(prepared.items()):
        b = p["b"]
        mk = lec_market_of(sym)
        etf = lec_sector_etf(sym, p.get("sector"))
        ev = lec_l2_events(p, etfb.get(etf) if etf else None, grades.get(lec_index_of(sym, p.get("sector"))),
                           p["earn"]["dates"], p["earn"]["cover_from"])
        fwd = {h: lec_forward_returns(b, h) for h in hz_all}
        xser[sym] = fwd[H] - means[H][mk].reindex(b.index)
        if len(ev):
            T = ev["t"].to_numpy(dtype=int)
            for h in hz_all:
                ev[f"x{h}"] = fwd[h].to_numpy()[T] - means[h][mk].reindex(b.index).to_numpy()[T]
            o, hi, lo = (b[k].to_numpy(dtype=float) for k in ("Open", "High", "Low"))
            n = len(b)
            mfe, mae = np.full(len(T), np.nan), np.full(len(T), np.nan)
            for j, t in enumerate(T):
                if t + H < n and o[t + 1] > 0:
                    mfe[j] = hi[t + 1:t + H + 1].max() / o[t + 1] - 1.0
                    mae[j] = lo[t + 1:t + H + 1].min() / o[t + 1] - 1.0
            ev["mfe10"], ev["mae10"] = mfe, mae
            evs.append(ev.assign(ticker=sym, market=mk))
        if progress:
            progress(i + 1, len(prepared))
    allev = pd.concat(evs, ignore_index=True) if evs else pd.DataFrame(columns=["t", "event", "date", "ticker", "market", "missing"])
    return {"events": allev, "xser": xser}


# ---------------------------------------------------------------------
# L2 실행
# ---------------------------------------------------------------------
def run_lecture_l2(table: Dict[str, Any], progress: Optional[Callable[[str, int, int], None]] = None) -> Dict[str, Any]:
    c = LECTURE_CONFIG
    H = c["HORIZON"]
    col = f"x{H}"
    cost = c["COST"]
    ev_all = table["events"]
    dts = pd.to_datetime(ev_all["date"]) if len(ev_all) else pd.Series(dtype="datetime64[ns]")
    s0, s1, v0 = pd.Timestamp(c["EXPLORE_START"]), pd.Timestamp(c["EXPLORE_END"]), pd.Timestamp(c["VERIFY_START"])
    ex = ev_all[(dts >= s0) & (dts <= s1)].reset_index(drop=True)
    ve = ev_all[dts >= v0].reset_index(drop=True)
    combos = lec_l2_combos()
    stats, sel = {}, {}
    for cb in combos:
        r = lec_select(ex, cb, col)
        s = lec_combo_stats(ex, r["idx"], col, cost)
        s.update({"event": cb["event"], "env": cb["env"], "conf": cb["conf"], "n_missing": r["n_missing"]})
        stats[cb["id"]] = s
        if s["n"] >= c["MIN_EVENTS"] and s["n_tickers"] >= c["MIN_TICKERS"]:
            sel[cb["id"]] = r["idx"]
    pool = lec_null_pool(table["xser"], (s0, s1))
    rc = lec_reality_check(ex, sel, pool, c["RC_REPS"], c["SEED"],
                           progress=(lambda i, n: progress("섞기", i, n)) if progress else None)
    thr = rc["threshold"]
    pbo = lec_pbo(ex, sel, col, (s0, s1), c["PBO_SLICES"])
    cands = []
    for k, s in stats.items():
        s["null_mean"] = rc.get("null_mean", {}).get(k)
        if k in sel and thr is not None:
            s["judge"] = lec_judge_one(s, thr)
        else:
            s["judge"] = {"candidate": False, "reasons": ["가족에 못 들어감(사건 < 300 또는 종목 < 50)"] if k not in sel else ["통과선 없음"]}
        if s["judge"]["candidate"]:
            cands.append(k)
    cands = sorted(cands, key=lambda k: -stats[k]["t"])
    verify = {}
    for k in cands:
        cb = next(x for x in combos if x["id"] == k)
        r = lec_select(ve, cb, col)
        s = lec_combo_stats(ve, r["idx"], col, cost)
        ok = bool(s["net"] is not None and s["net"] > 0 and s["t"] is not None and s["t"] >= c["VERIFY_Z"])
        verify[k] = {"pass": ok, "n": s["n"], "net": s["net"], "t": s["t"]}
    refs = {}
    for a, bq, lab in L2_REF_PAIRS:
        A = ex.iloc[lec_select(ex, {"event": a, "env": None, "conf": None}, col)["idx"]]
        B = ex.iloc[lec_select(ex, {"event": bq, "env": None, "conf": None}, col)["idx"]]
        refs[f"{a}-{bq}"] = {"label": lab, "cmp": compare_clustered(A, B, col)}
    e13 = ex.iloc[lec_select(ex, {"event": "E13", "env": None, "conf": None}, col)["idx"]]
    e13_ref = {"n": int(len(e13)), "mfe10": float(e13["mfe10"].mean()) if len(e13) else None,
               "mae10": float(e13["mae10"].mean()) if len(e13) else None}
    e01_missing = int(((ex["event"] == "E01") & (ex["missing"] == 1.0)).sum()) if len(ex) else 0
    return {"family": "L2", "n_combos": len(combos), "n_family": len(sel), "threshold": thr, "rc_max_t": rc["max_t"],
            "pbo": pbo, "stats": stats, "candidates": cands, "verify": verify, "refs": refs, "e13_ref": e13_ref,
            "e01_missing": e01_missing, "n_tickers_used": len(table["xser"]), "config_version": c["VERSION"],
            "n_event_rows": int(len(ex))}


# ---------------------------------------------------------------------
# 보고서 · 기록
# ---------------------------------------------------------------------
def _lec_f(v, d=1, sign=True):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    return format(v, f"+.{d}f" if sign else f".{d}f")


def _lec_combo_label(s: Dict[str, Any]) -> str:
    parts = [f"{s['event']} {L2_EVENTS[s['event']][0]}"]
    if s.get("env"):
        parts.append(f"{s['env']} {L2_ENVS[s['env']]}")
    if s.get("conf"):
        parts.append(f"{s['conf']} {L2_CONFS[s['conf']]}")
    return " + ".join(parts)


def render_lecture_l2_report(r: Dict[str, Any], generated_at: Optional[str] = None) -> str:
    c = LECTURE_CONFIG
    st = r["stats"]
    thr = r["threshold"]
    fam = [k for k, s in st.items() if s["n"] >= c["MIN_EVENTS"] and s["n_tickers"] >= c["MIN_TICKERS"]]
    L = ["# 강의 조건 시험 L2 — 매수 사건 조합 (2~8번)", "",
         "**탐색 단계 결과 — 이것만으로는 증거가 아니다.** 이미 다른 시험에 쓴 종목의 2001~2013년이다. 검증(2014~)과 최종(새 종목, z ≥ 3)까지 통과해야 '통과'.", "",
         f"- 생성: {generated_at or '—'} · 설정 {r['config_version']} (사전 등록 {c['REGISTERED']}) · 종목 {r['n_tickers_used']}개 · 탐색 기간 사건 줄 {r['n_event_rows']:,}개",
         f"- 조합 **{r['n_combos']}개**(짝짓기 표: 사건 1 + 환경 0~1 + 확인 0~1) 중 가족 경쟁에 든 조합(사건 ≥ {c['MIN_EVENTS']}, 종목 ≥ {c['MIN_TICKERS']}) **{r['n_family']}개**",
         f"- 결과 = 다음 날 시가 → {c['HORIZON']}거래일 뒤 종가 − 같은 날 같은 시장 종목 평균(초과수익). 후보는 '평균 − 비용 {c['COST'] * 100:.2f}%' > 0 이어야 함",
         f"- **통과선 t = {_lec_f(thr, 2, False)}** (같은 종목 무작위 날로 바꿔 가족 전체를 {c['RC_REPS']}번 다시 잰 '가족 안 최고 t'의 {int(c['RC_Q'] * 100)}% 분위)",
         f"- 과적합 확률 PBO(참고) = {_lec_f(None if r['pbo']['pbo'] is None else r['pbo']['pbo'] * 100, 0, False)}% (조합 {r['pbo']['n_combos']}개, 나눈 경우 {r['pbo']['n_splits']:,}) — 50% 근처면 '탐색 1등'이 다른 기간에선 평범하다는 뜻",
         f"- 섹터 ETF 자료가 없어 빠진 E01 사건: {r['e01_missing']:,}건", "",
         "## 판정", ""]
    if not r["candidates"]:
        L += ["**탐색 후보 없음.** 통과선을 넘으면서 다른 조건(비용 뒤 평균 > 0, 시장 방향, 달 쏠림)까지 갖춘 조합이 없다. 검증 기간(2014~)은 열지 않았다.", ""]
    else:
        L += ["| 조합 | 사건 | 비용 뒤 평균 | t | 검증(2014~) |", "|---|---|---|---|---|"]
        for k in r["candidates"]:
            s, v = st[k], r["verify"].get(k, {})
            vtxt = "—" if not v else f"{'통과' if v['pass'] else '미통과'} ({_lec_pct(v['net'])}, t {_lec_f(v['t'])}, {v['n']:,}건)" + (" → 최종 확인 대기" if v.get("pass") else "")
            L.append(f"| {_lec_combo_label(s)} | {s['n']:,} | {_lec_pct(s['net'])} | {_lec_f(s['t'])} | {vtxt} |")
        L.append("")
    L += ["## 가족 안 t 상위 25 (참고 — 순위만으로는 아무것도 아님)", "",
          "'같은 종목 무작위 날'은 섞기에서 같은 종목의 아무 날이나 골랐을 때의 평균 초과수익이다. 사건 쪽이 이보다 높아야 '그날'에 뜻이 있다.", "",
          "| 조합 | 사건 | 종목 | 초과수익 | 같은 종목 무작위 날 | 비용 뒤 | t | 미국 / 한국 비용 뒤 | 판정 |", "|---|---|---|---|---|---|---|---|---|"]
    for k in sorted(fam, key=lambda k: -(st[k]["t"] if st[k]["t"] is not None else -9))[:25]:
        s = st[k]
        bm = s["by_market"]
        why = "후보" if s["judge"]["candidate"] else ", ".join(s["judge"]["reasons"])
        L.append(f"| {_lec_combo_label(s)} | {s['n']:,} | {s['n_tickers']} | {_lec_pct(s['mean'])} | {_lec_pct(s.get('null_mean'))} | {_lec_pct(s['net'])} | {_lec_f(s['t'])} | "
                 f"{_lec_pct(bm['US']['net'])} ({bm['US']['n']:,}) / {_lec_pct(bm['KR']['net'])} ({bm['KR']['n']:,}) | {why} |")
    L += ["", "## 사건 단독(환경·확인 없음)", "", "| 사건 | 사건 수 | 종목 | 초과수익 | 비용 뒤 | t |", "|---|---|---|---|---|---|"]
    for e in L2_EVENTS:
        s = st.get(e)
        if s:
            L.append(f"| {e} {L2_EVENTS[e][0]} | {s['n']:,} | {s['n_tickers']} | {_lec_pct(s['mean'])} | {_lec_pct(s['net'])} | {_lec_f(s['t'])} |")
    L += ["", "## 참고 비교(판정 아님)", ""]
    for k, v in r["refs"].items():
        x = v["cmp"]
        a, b_ = x.get("actual", {}), x.get("baseline", {})
        L.append(f"- {v['label']}: {_lec_pct(a.get('mean'))} ({a.get('n', 0):,}건) vs {_lec_pct(b_.get('mean'))} ({b_.get('n', 0):,}건) · "
                 f"차이 {_lec_pct(x.get('diff'))} · z {_lec_f(x.get('z'))}")
    e = r["e13_ref"]
    L.append(f"- E13 분출의 10거래일 최대 상승 평균 {_lec_pct(e['mfe10'])} · 최대 하락 평균 {_lec_pct(e['mae10'])} ({e['n']:,}건, 초과수익 아님)")
    L += ["", "## 읽는 법", "",
          "- 후보가 아니면 여기서 끝낸다. 결과를 보고 사건 정의·환경·확인·짝짓기·숫자를 바꿔 다시 보지 않는다.",
          "- 상위 25 표는 '가장 좋아 보이는 것'을 보여 줄 뿐이다. 통과선은 조합이 이만큼 많을 때 우연히 나오는 최고 t의 크기다.", ""]
    return "\n".join(L)


def lec_l2_log_lines(r: Dict[str, Any], generated_at: str) -> List[str]:
    st = r["stats"]
    fam = [k for k in st if st[k]["n"] >= LECTURE_CONFIG["MIN_EVENTS"] and st[k]["n_tickers"] >= LECTURE_CONFIG["MIN_TICKERS"]]
    best = max(fam, key=lambda k: st[k]["t"] if st[k]["t"] is not None else -9) if fam else None
    rows = []
    if best:
        s = st[best]
        rows.append(f"| {generated_at} | `lec-L2` | 매수 사건 조합 {r['n_combos']}개(가족 {r['n_family']}개) 중 최고 t `{best}` · 통과선 {_lec_f(r['threshold'], 2, False)} "
                    f"(강의 2~8번, 탐색 2001~2013) | {s['n']:,} | {s['net'] * 100:+.2f}%p(비용 뒤) | {_lec_f(s['t'])} | "
                    f"{'후보 ' + str(len(r['candidates'])) + '개' if r['candidates'] else '후보 없음'} |")
    for k in r["candidates"]:
        s = st[k]
        rows.append(f"| {generated_at} | `lec-L2-{k}` | {_lec_combo_label(s)} | {s['n']:,} | {s['net'] * 100:+.2f}%p(비용 뒤) | {_lec_f(s['t'])} | 후보 |")
    return rows


def lec_append_lines(lines: List[str], out_dir: str) -> str:
    p = os.path.join(out_dir, EXPLORE_CONFIG["LOG_FILE"])
    os.makedirs(out_dir, exist_ok=True)
    if not os.path.exists(p):
        with open(p, "w", encoding="utf-8") as f:
            f.write("# 탐색 기록\n\n| 본 때 | 이름 | 가설 | 사건 수 | 무작위 날 대비 차이 | z | 판정 |\n|---|---|---|---|---|---|---|\n")
    with open(p, "a", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")
    return p
