"""H-MR3 변동성 국면 이격도·지지 반등 (2026-11-19 사전 등록, data/backtest/MR3_PREREG.md · AGENTS.md 4장).

평상시에는 단타를 하지 않고, 앞선 미국 거래일 VIX 종가 ≥ 20인 날(변동성 국면)에만 이격도·지지 규칙 5개를 켠다.
    V1 이격도 기준(MR2 S10) 종가 · V2 같은 기준 지정가 · V3 지지 구간(MR2 S12) 종가 · V4 지지 구간 지정가
    V5 이격도 ≤ 기준 + V3 같은 날 종가
청산: 손절 = 진입가 − 2×ATR_{t−1}(장중, 시가가 아래면 시가), 종가 ≥ 25일선이면 그날 종가, 최대 10거래일(같은 날 둘 다면 손절).
비교 기준: 같은 종목·같은 단계 기간의 대상이면서 변동성 국면인 날에 종가로 사서 같은 보유일 뒤 종가에 판 평균(보유 기간 맞춤).
단계: 찾기(545종목 2001~2019) → 후보만 검증(미국 1990~2000 + 545종목 2020~) → 검증 통과만 최종(새 151종목, --mr3-final).
숫자는 MR3_CONFIG 한 곳 — 결과를 보고 바꾸지 않는다.
"""
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .explore import EXPLORE_CONFIG
from .frame_backtest import compare_clustered
from .lecture_backtest import _lec_pct, lec_clean, lec_market_of
from .lecture_family import _lec_fast_t, lec_cluster_t, lec_month_code, lec_pbo
from .lecture_signals import lec_base, lec_swings
from .mr_backtest import _mr_bounds, mr_eligibility
from .mr2_backtest import _mr2_shift, mr2_signal_arrays

MR3_CONFIG: Dict[str, Any] = {
    "VERSION": "mr3-1",
    "REGISTERED": "2026-11-19",
    "VIX_SYMBOL": "^VIX",
    "VIX_START": "1990-01-01",
    "VIX_MIN": 20.0,                   # 변동성 국면 = 앞선 미국 거래일 VIX 종가 ≥ 20 (사용자 결정)
    "VIX_STALE_DAYS": 10,              # 가장 최근 VIX가 이보다 오래되면(달력일) 자료 없음
    "VIX_REF_BANDS": ((20.0, 30.0), (30.0, float("inf"))),    # 참고: VIX 구간별
    "EXPLORE_START": "2001-01-01", "EXPLORE_END": "2019-12-31",
    "VERIFY_OLD": ("1990-01-01", "2000-12-31"),   # 검증 ①: 미국 종목, 어떤 시험에도 안 쓴 기간
    "VERIFY_NEW_START": "2020-01-01",             # 검증 ②: 545종목 2020~끝
    "OLD_DATA_START": "1989-01-01",               # 1990년대 자료 받기 시작(대상 순위 250일 준비)
    "OLD_DATA_END": "2001-06-30",                 # 1990년대 자료는 여기서 자름(청산 10일 여유 + 메모리)
    "FINAL_START": "2001-01-01",
    # --- 청산 ---
    "STOP_ATR": 2.0,
    "MAX_DAYS": 10,
    "COST": 0.003,
    "STRICT_FILL": 0.999,
    "STRICT_Z": 2.0,
    # --- 판정 ---
    "MIN_TRADES": 300, "MIN_TICKERS": 50, "MIN_MARKET": 30, "Z_FLOOR": 2.0,
    "YEAR_DROP_Z": 2.0, "YEAR_MIN_TRADES": 30, "YEAR_AGREE": 2.0 / 3.0,
    "RC_REPS": 200, "RC_Q": 0.95, "PBO_SLICES": 16,
    "VERIFY_Z": 2.0, "VERIFY_MIN_TRADES": 100, "VERIFY_PART_MIN": 30,
    "FINAL_Z": 3.0, "FINAL_MIN_TRADES": 100, "FINAL_MAX": 5,
    "POWER_Z": 0.84,                   # 검출 가능한 최소 차이 = (기준 z + 0.84) × 표준오차 (검정력 80%)
    "TIMING_DAYS": (5, 10),            # 참고: 시점 효과(종가 보유일)
    "PERIOD_BLOCKS": (("2001-01-01", "2007-12-31"), ("2008-01-01", "2013-12-31"), ("2014-01-01", "2019-12-31")),
    "SEED": 1121,
}
MR3_RULES: List[str] = ["V1-C", "V2-L", "V3-C", "V4-L", "V5-C"]
MR3_LABELS: Dict[str, str] = {
    "V1-C": "V1 이격도 기준 도달(처음) · 종가",
    "V2-L": "V2 이격도 기준 가격 · 지정가",
    "V3-C": "V3 지지 구간 도달 · 종가",
    "V4-L": "V4 지지 구간 상단 · 지정가",
    "V5-C": "V5 이격도 ≤ 기준 + 지지 구간 · 종가",
}
MR3_KIND = {1: "25일선 회복", 2: "손절", 3: "10일 만기", 5: "시가 손절", 6: "체결일 손절", 7: "체결일 회복"}


# ---------------------------------------------------------------------
# VIX 국면 (앞선 미국 거래일 값만)
# ---------------------------------------------------------------------
def mr3_vix_prev(dates: pd.DatetimeIndex, vix: Optional[pd.Series]) -> np.ndarray:
    """날짜마다 그 날짜보다 앞선 가장 최근 VIX 종가(같은 날은 쓰지 않음). 10일(달력)보다 오래되면 NaN."""
    out = np.full(len(dates), np.nan)
    if vix is None or not len(vix):
        return out
    v = vix.dropna().sort_index()
    vd = pd.DatetimeIndex(v.index).values.astype("datetime64[ns]")
    dd = pd.DatetimeIndex(dates).values.astype("datetime64[ns]")
    k = np.searchsorted(vd, dd, "left") - 1
    ok = k >= 0
    kk = np.maximum(k, 0)
    age = (dd - vd[kk]) / np.timedelta64(1, "D")
    ok &= age <= MR3_CONFIG["VIX_STALE_DAYS"]
    out[ok] = v.to_numpy(dtype=float)[kk[ok]]
    return out


# ---------------------------------------------------------------------
# 청산 (2 ATR 손절 · 25일선 회복 · 10거래일)
# ---------------------------------------------------------------------
def mr3_exit(o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray, ma: np.ndarray,
             t: int, E: float, stop: float, fill_day: bool = False) -> Tuple[float, int, int]:
    """t일 진입(E), 손절가 stop. (비용 전 수익, kind, 보유일). fill_day = 지정가 체결일(그날 손절·회복도 봄)."""
    D = MR3_CONFIG["MAX_DAYS"]
    if fill_day:
        if l[t] <= stop:
            return stop / E - 1.0, 6, 0
        if np.isfinite(ma[t]) and c[t] >= ma[t]:
            return c[t] / E - 1.0, 7, 0
    for k in range(1, D + 1):
        d = t + k
        if o[d] <= stop:
            return o[d] / E - 1.0, 5, k
        if l[d] <= stop:
            return stop / E - 1.0, 2, k
        if np.isfinite(ma[d]) and c[d] >= ma[d]:
            return c[d] / E - 1.0, 1, k
    return c[t + D] / E - 1.0, 3, D


# ---------------------------------------------------------------------
# 종목 하나
# ---------------------------------------------------------------------
def mr3_ticker(b: pd.DataFrame, sw: Dict[str, np.ndarray], el: np.ndarray, vixv: np.ndarray, lo_i: int, hi_i: int,
               rules: List[str], strict: bool = True) -> Dict[str, Any]:
    """종목 하나, 신호일 [lo_i, hi_i]. 비교 기준 = 대상·변동성 국면인 날 j의 종가 → h거래일 뒤 종가(h = 1~10)."""
    cf = MR3_CONFIG
    D = cf["MAX_DAYS"]
    n = len(b)
    o, h, l, c = (b[k].to_numpy(dtype=float) for k in ("Open", "High", "Low", "Close"))
    ma = b["SMA25"].to_numpy(dtype=float)
    patr = _mr2_shift(b["ATR14"].to_numpy(dtype=float), 1)
    el = np.asarray(el, dtype=bool)
    pel = np.r_[False, el[:-1]]
    with np.errstate(invalid="ignore"):
        cr = np.isfinite(vixv) & (vixv >= cf["VIX_MIN"])
    inper = np.zeros(n, dtype=bool)
    inper[lo_i:hi_i + 1] = True
    room = np.arange(n) + D <= n - 1
    need = inper & (el | pel) & cr
    cache: Dict[str, Any] = {}
    sig, px = mr2_signal_arrays(b, sw, {"S10", "S12"}, need=need, cache=cache)
    with np.errstate(invalid="ignore"):
        pb = _mr2_shift(cache["base"], 1)
        v5 = (b["DISP25"].to_numpy(dtype=float) <= pb) & sig["S12"]
    sigV = {"V1": sig["S10"], "V3": sig["S12"], "V5": v5}
    pxV = {"V2": px["S10"], "V4": px["S12"]}
    J = np.flatnonzero(inper & el & cr & room)
    R = (np.stack([c[J + k] for k in range(1, D + 1)], axis=1) / c[J][:, None] - 1.0 - cf["COST"]) if len(J) else np.zeros((0, D))
    tj = np.flatnonzero(inper & el & np.isfinite(vixv) & room)
    timing = {"t": tj, "crisis": cr[tj]}
    for k in cf["TIMING_DAYS"]:
        timing[f"r{k}"] = c[tj + k] / c[tj] - 1.0 - cf["COST"] if len(tj) else np.array([])
    out: Dict[str, Any] = {"pool_t": J, "pool_R": R, "timing": timing, "rules": {}, "strict": {}, "limit": {}, "open_end": {}}

    def rec(ts, es, gs, ks, hs, of):
        return {"t": np.array(ts, dtype=np.int64), "entry": np.array(es, dtype=float), "gross": np.array(gs, dtype=float),
                "kind": np.array(ks, dtype=np.int8), "hold": np.array(hs, dtype=np.int64), "open_fill": np.array(of, dtype=bool)}
    for r in rules:
        vid, mode = r.split("-")
        if mode == "C":
            cand = np.flatnonzero(sigV[vid] & el & cr & inper)
            ts, es, gs, ks, hs, of = [], [], [], [], [], []
            free, oe = -1, 0
            for t in cand:
                if t <= free:
                    continue
                if t + D > n - 1:
                    oe += 1
                    break
                E = c[t]
                stop = E - cf["STOP_ATR"] * patr[t]
                if not (np.isfinite(stop) and stop > 0):
                    continue
                g, kd, hd = mr3_exit(o, h, l, c, ma, t, E, stop)
                ts.append(t); es.append(E); gs.append(g); ks.append(kd); hs.append(hd); of.append(False)
                free = t + hd
            out["rules"][r] = rec(ts, es, gs, ks, hs, of)
            out["open_end"][r] = oe
            continue
        P = pxV[vid]
        cand = np.flatnonzero(np.isfinite(P) & (P > 0) & pel & cr & inper)
        for variant in ((False, True) if strict else (False,)):
            ts, es, gs, ks, hs, of = [], [], [], [], [], []
            free, oe = -1, 0
            st = {"orders": 0, "fills": 0, "fill_day_exit": 0, "open_fills": 0}
            for t in cand:
                if t <= free:
                    continue
                if t + D > n - 1:
                    oe += 1
                    break
                st["orders"] += 1
                p = P[t]
                hit = ((o[t] <= p) or (l[t] <= p * cf["STRICT_FILL"])) if variant else (l[t] <= p)
                if not hit:
                    continue
                F = min(o[t], p)
                stop = F - cf["STOP_ATR"] * patr[t]
                if not (np.isfinite(stop) and stop > 0):
                    continue
                st["fills"] += 1
                st["open_fills"] += int(o[t] <= p)
                g, kd, hd = mr3_exit(o, h, l, c, ma, t, F, stop, fill_day=True)
                st["fill_day_exit"] += int(hd == 0)
                ts.append(t); es.append(F); gs.append(g); ks.append(kd); hs.append(hd); of.append(o[t] <= p)
                free = t + hd
            if variant:
                out["strict"][r] = rec(ts, es, gs, ks, hs, of)
            else:
                out["rules"][r] = rec(ts, es, gs, ks, hs, of)
                out["limit"][r] = st
                out["open_end"][r] = oe
    return out


# ---------------------------------------------------------------------
# 한 단계(기간)
# ---------------------------------------------------------------------
def mr3_stage(prepared: Dict[str, Dict[str, Any]], eligible: Dict[str, np.ndarray], vix: Optional[pd.Series],
              rules: List[str], start: str, end: Optional[str], progress: Optional[Callable[[int, int], None]] = None,
              strict: bool = True) -> Dict[str, Any]:
    """거래 전부 + 비교 기준. x = 순수익 − 그 종목의 같은 보유일 평균(대상·변동성 국면인 날)."""
    cf = MR3_CONFIG
    D = cf["MAX_DAYS"]
    cols = ["rule", "ticker", "market", "t", "date", "entry", "gross", "kind", "hold", "open_fill", "vix"]
    parts, sparts, tparts = [], [], []
    pools: Dict[str, Dict[str, Any]] = {}
    lim = {r: {"orders": 0, "fills": 0, "fill_day_exit": 0, "open_fills": 0} for r in rules if r.endswith("-L")}
    open_end = {r: 0 for r in rules}
    days = cdays = 0
    for i, (sym, p) in enumerate(prepared.items()):
        b = p["b"]
        if progress:
            progress(i + 1, len(prepared))
        lo_i, hi_i = _mr_bounds(b.index, start, end)
        el = eligible.get(sym)
        if hi_i < lo_i or el is None or not np.any(el[max(0, lo_i - 1):hi_i + 1]):
            continue
        vixv = mr3_vix_prev(b.index, vix)
        e_in = el[lo_i:hi_i + 1]
        days += int(e_in.sum())
        with np.errstate(invalid="ignore"):
            cdays += int((e_in & (vixv[lo_i:hi_i + 1] >= cf["VIX_MIN"])).sum())
        res = mr3_ticker(b, p["sw"], el, vixv, lo_i, hi_i, rules, strict=strict)
        mk = lec_market_of(sym)
        if len(res["pool_t"]):
            pools[sym] = {"R": res["pool_R"], "mons": lec_month_code(b.index[res["pool_t"]])}
        tm = res["timing"]
        if len(tm["t"]):
            tparts.append(pd.DataFrame({"ticker": sym, "date": b.index[tm["t"]], "crisis": tm["crisis"],
                                        **{f"r{k}": tm[f"r{k}"] for k in cf["TIMING_DAYS"]}}))
        for key, store in (("rules", parts), ("strict", sparts)):
            for r, d in res[key].items():
                if len(d["t"]):
                    store.append(pd.DataFrame({"rule": r, "ticker": sym, "market": mk, "t": d["t"], "date": b.index[d["t"]],
                                               "entry": d["entry"], "gross": d["gross"], "kind": d["kind"], "hold": d["hold"],
                                               "open_fill": d["open_fill"], "vix": vixv[d["t"]]}))
        for r, st in res["limit"].items():
            for k, v in st.items():
                lim[r][k] += v
        for r, v in res["open_end"].items():
            open_end[r] += v
    mu = {s: v["R"].mean(axis=0) for s, v in pools.items()}

    def finish(store: List[pd.DataFrame]) -> pd.DataFrame:
        tr = pd.concat(store, ignore_index=True) if store else pd.DataFrame(columns=cols)
        if not len(tr):
            return tr.assign(net=[], hb=[], base=[], x=[])
        tr["net"] = tr["gross"].astype(float) - cf["COST"]
        tr["hb"] = np.maximum(tr["hold"].astype(int).to_numpy(), 1)
        base = np.full(len(tr), np.nan)
        tk = tr["ticker"].to_numpy()
        hb = tr["hb"].to_numpy()
        for s in np.unique(tk):
            if s in mu:
                m = tk == s
                base[m] = mu[s][hb[m] - 1]
        tr["base"] = base
        tr["x"] = tr["net"] - tr["base"]
        return tr[np.isfinite(tr["x"].to_numpy(dtype=float))].reset_index(drop=True)
    pool_dm = {s: {"R": v["R"] - mu[s][None, :], "mons": v["mons"]} for s, v in pools.items()}
    timing = pd.concat(tparts, ignore_index=True) if tparts else pd.DataFrame(columns=["ticker", "date", "crisis"])
    npool = int(sum(len(v["R"]) for v in pools.values()))
    return {"tr": finish(parts), "strict": finish(sparts), "pool": pool_dm, "mu": mu, "pool_n": npool,
            "pool_mean5": (float(np.mean(np.concatenate([v["R"][:, 4] for v in pools.values()]))) if npool else None),
            "timing": timing, "limit": lim, "open_end": open_end, "period": (start, end),
            "elig_days": days, "crisis_days": cdays, "crisis_share": (cdays / days) if days else None}


# ---------------------------------------------------------------------
# 통계 · 판정
# ---------------------------------------------------------------------
def mr3_year_checks(x: np.ndarray, dates) -> Dict[str, Any]:
    """해 하나 빼기(차이에 가장 크게 기여한 해) + 해별 방향(거래 ≥ 30인 해 중 차이 > 0 비율)."""
    cf = MR3_CONFIG
    x = np.asarray(x, dtype=float)
    d = pd.DatetimeIndex(pd.to_datetime(dates))
    yr = d.year.to_numpy()
    g = lec_month_code(d)
    out = {"drop_year": None, "drop_diff": None, "drop_t": None, "years_ok": 0, "years_n": 0, "years_share": None, "by_year": {}}
    if len(x) < 3:
        return out
    s = pd.Series(x).groupby(yr).sum()
    worst = int(s.idxmax())
    keep = yr != worst
    r = lec_cluster_t(x[keep], g[keep])
    out.update({"drop_year": worst, "drop_diff": r["mean"], "drop_t": r["t"]})
    for y in np.unique(yr):
        m = yr == y
        out["by_year"][int(y)] = {"n": int(m.sum()), "diff": float(x[m].mean())}
        if m.sum() >= cf["YEAR_MIN_TRADES"]:
            out["years_n"] += 1
            out["years_ok"] += int(x[m].mean() > 0)
    out["years_share"] = (out["years_ok"] / out["years_n"]) if out["years_n"] else None
    return out


def mr3_stats(tr: pd.DataFrame, col: str = "x") -> Dict[str, Any]:
    empty_m = {m: {"n": 0, "diff": None, "t": None} for m in ("US", "KR")}
    if not len(tr):
        return {"n": 0, "n_tickers": 0, "diff": None, "t": None, "se": None, "net": None, "base": None, "by_market": empty_m,
                "years": mr3_year_checks(np.array([]), []), "bands": {}, "periods": {}}
    x = tr[col].to_numpy(dtype=float)
    g = lec_month_code(tr["date"])
    r = lec_cluster_t(x, g)
    out = {"n": r["n"], "diff": r["mean"], "t": r["t"], "se": r["se"], "n_months": r["k"], "n_tickers": int(tr["ticker"].nunique()),
           "net": float(tr["net"].mean()), "base": float(tr["base"].mean()), "win": float((tr["net"] > 0).mean()),
           "hold": float(tr["hold"].mean()),
           "kinds": {MR3_KIND[int(k)]: int(v) for k, v in tr["kind"].value_counts().sort_index().items()}, "by_market": {}}
    rn = lec_cluster_t(tr["net"].to_numpy(dtype=float), g)
    out["net_t"] = rn["t"]
    for m in ("US", "KR"):
        mm = tr["market"].to_numpy() == m
        rr = lec_cluster_t(x[mm], g[mm])
        out["by_market"][m] = {"n": rr["n"], "diff": rr["mean"], "t": rr["t"]}
    out["years"] = mr3_year_checks(x, tr["date"])
    vx = tr["vix"].to_numpy(dtype=float)
    bands = {}
    for lo, hi in MR3_CONFIG["VIX_REF_BANDS"]:
        mb = (vx >= lo) & (vx < hi)
        rb = lec_cluster_t(x[mb], g[mb])
        bands[f"{lo:.0f}~{'' if not np.isfinite(hi) else f'{hi:.0f}'}"] = {"n": rb["n"], "diff": rb["mean"], "t": rb["t"]}
    out["bands"] = bands
    d = pd.to_datetime(tr["date"])
    blocks = {}
    for s0, s1 in MR3_CONFIG["PERIOD_BLOCKS"]:
        mb = ((d >= s0) & (d <= s1)).to_numpy()
        if mb.any():
            rb = lec_cluster_t(x[mb], g[mb])
            blocks[f"{s0[:4]}~{s1[:4]}"] = {"n": rb["n"], "diff": rb["mean"], "t": rb["t"]}
    out["periods"] = blocks
    return out


def mr3_mde(s: Dict[str, Any], z_need: float) -> Optional[float]:
    """검출 가능한 최소 차이(검정력 80%) = (기준 z + 0.84) × 표준오차."""
    se = s.get("se")
    return None if se is None or not np.isfinite(se) else float((z_need + MR3_CONFIG["POWER_Z"]) * se)


def _mr3_market_conflict(s: Dict[str, Any]) -> bool:
    mk = [v for v in s["by_market"].values() if v["n"] >= MR3_CONFIG["MIN_MARKET"] and v["diff"] is not None]
    return len(mk) == 2 and np.sign(mk[0]["diff"]) != np.sign(mk[1]["diff"])


def _mr3_strict_reason(strict: Optional[Dict[str, Any]]) -> List[str]:
    if strict is None:
        return []
    z = MR3_CONFIG["STRICT_Z"]
    if not strict.get("n") or strict.get("diff") is None or strict["diff"] <= 0 or strict.get("t") is None or strict["t"] < z:
        return [f"엄격 체결로는 기준 미달(차이 > 0, t ≥ {z:.0f} 아님)"]
    return []


def mr3_judge_explore(s: Dict[str, Any], threshold: float, strict: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cf = MR3_CONFIG
    rs = []
    if s["n"] < cf["MIN_TRADES"]:
        rs.append(f"거래 {s['n']:,} < {cf['MIN_TRADES']}")
    if s["n_tickers"] < cf["MIN_TICKERS"]:
        rs.append(f"종목 {s['n_tickers']} < {cf['MIN_TICKERS']}")
    if s["diff"] is None or s["diff"] <= 0:
        rs.append("기준보다 나쁨(차이 ≤ 0)")
    if s["t"] is None or not np.isfinite(s["t"]) or s["t"] < threshold:
        rs.append(f"t < 통과선 {threshold:.2f}")
    if s.get("net") is None or s["net"] <= 0:
        rs.append("비용 뒤 거래당 순수익 ≤ 0")
    if _mr3_market_conflict(s):
        rs.append("미국·한국 방향이 다름")
    y = s["years"]
    if y["drop_diff"] is None or y["drop_diff"] <= 0 or y["drop_t"] is None or y["drop_t"] < cf["YEAR_DROP_Z"]:
        rs.append(f"기여가 가장 큰 해({y['drop_year']})를 빼면 기준 미달")
    if y["years_share"] is None or y["years_share"] < cf["YEAR_AGREE"]:
        rs.append(f"해별 방향 일치 {y['years_ok']}/{y['years_n']} < 3분의 2")
    rs += _mr3_strict_reason(strict)
    return {"candidate": not rs, "reasons": rs}


def mr3_judge_verify(s: Dict[str, Any], parts: Dict[str, Dict[str, Any]], strict: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cf = MR3_CONFIG
    rs = []
    if s["n"] < cf["VERIFY_MIN_TRADES"]:
        rs.append(f"거래 {s['n']:,} < {cf['VERIFY_MIN_TRADES']}")
    if s["diff"] is None or s["diff"] <= 0:
        rs.append("기준보다 나쁨(차이 ≤ 0)")
    if s["t"] is None or not np.isfinite(s["t"]) or s["t"] < cf["VERIFY_Z"]:
        rs.append(f"t < {cf['VERIFY_Z']:.0f}")
    if s.get("net") is None or s["net"] <= 0:
        rs.append("비용 뒤 거래당 순수익 ≤ 0")
    for name, ps in parts.items():
        if ps.get("n", 0) >= cf["VERIFY_PART_MIN"] and (ps.get("diff") is None or ps["diff"] <= 0):
            rs.append(f"{name} 구간 방향이 반대")
    rs += _mr3_strict_reason(strict)
    return {"passed": not rs, "reasons": rs}


def mr3_judge_final(s: Dict[str, Any], strict: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cf = MR3_CONFIG
    rs = []
    if s["n"] < cf["FINAL_MIN_TRADES"]:
        rs.append(f"거래 {s['n']:,} < {cf['FINAL_MIN_TRADES']}")
    if s["diff"] is None or s["diff"] <= 0:
        rs.append("기준보다 나쁨(차이 ≤ 0)")
    if s["t"] is None or not np.isfinite(s["t"]) or s["t"] < cf["FINAL_Z"]:
        rs.append(f"t < {cf['FINAL_Z']:.0f}")
    if s.get("net") is None or s["net"] <= 0:
        rs.append("비용 뒤 거래당 순수익 ≤ 0")
    if _mr3_market_conflict(s):
        rs.append("미국·한국 방향이 다름")
    rs += _mr3_strict_reason(strict)
    return {"passed": not rs, "reasons": rs}


def mr3_reality(stage: Dict[str, Any], rules: List[str], seed: int,
                progress: Optional[Callable[[int, int], None]] = None) -> Dict[str, Any]:
    """섞기 통과선: 거래(종목·신호일)마다 같은 종목의 대상·변동성 국면인 무작위 날 하나(같은 종목·신호일은 모든 규칙에서 같은 날)에서
    그 거래의 보유일만큼 종가 보유한 값(종목 평균을 뺌) → 규칙별 t → 최고 t."""
    cf = MR3_CONFIG
    tr = stage["tr"]
    if not rules or not len(tr):
        return {"max_t": [], "threshold": None, "null_mean": {}}
    sub = tr[tr["rule"].isin(rules)].reset_index(drop=True)
    pool = stage["pool"]
    syms = [s for s in pool if len(pool[s]["R"])]
    tid = {s: i for i, s in enumerate(syms)}
    sizes = np.array([len(pool[s]["R"]) for s in syms], dtype=np.int64)
    offs = np.r_[0, np.cumsum(sizes)[:-1]].astype(np.int64) if len(syms) else np.array([], dtype=np.int64)
    Rall = np.concatenate([pool[s]["R"] for s in syms]) if syms else np.zeros((0, cf["MAX_DAYS"]))
    Mall = np.concatenate([pool[s]["mons"] for s in syms]) if syms else np.array([], dtype=np.int64)
    keys = sub[["ticker", "t"]].drop_duplicates().reset_index(drop=True)
    kid = pd.MultiIndex.from_frame(keys).get_indexer(pd.MultiIndex.from_frame(sub[["ticker", "t"]]))
    ktk = np.array([tid.get(s, -1) for s in keys["ticker"]], dtype=np.int64)
    kok = ktk >= 0
    hb = sub["hb"].to_numpy(dtype=np.int64) - 1
    rows_by_rule = {r: np.flatnonzero((sub["rule"] == r).to_numpy() & kok[kid]) for r in rules}
    rng = np.random.default_rng(seed)
    mx, msum, mcnt = [], {r: 0.0 for r in rules}, {r: 0 for r in rules}
    for rep in range(cf["RC_REPS"]):
        u = rng.random(len(keys))
        row = offs[np.maximum(ktk, 0)] + np.floor(u * sizes[np.maximum(ktk, 0)]).astype(np.int64) if len(syms) else np.zeros(len(keys), dtype=np.int64)
        best = -np.inf
        for r, ix in rows_by_rule.items():
            if len(ix) < 2:
                continue
            rr = row[kid[ix]]
            xv = Rall[rr, hb[ix]]
            _, inv = np.unique(Mall[rr], return_inverse=True)
            tt = _lec_fast_t(xv, inv, int(inv.max()) + 1)
            msum[r] += float(xv.mean())
            mcnt[r] += 1
            if np.isfinite(tt) and tt > best:
                best = tt
        mx.append(float(best) if np.isfinite(best) else np.nan)
        if progress:
            progress(rep + 1, cf["RC_REPS"])
    arr = np.array([v for v in mx if np.isfinite(v)])
    return {"max_t": mx, "threshold": float(np.quantile(arr, cf["RC_Q"])) if len(arr) else None,
            "null_mean": {r: (msum[r] / mcnt[r] if mcnt[r] else None) for r in rules}}


def mr3_timing(stage: Dict[str, Any]) -> Dict[str, Any]:
    """참고(시점 효과): 대상인 날 중 변동성 국면 vs 평상시, k거래일 종가 보유(비용 뒤)."""
    tm = stage["timing"]
    out = {}
    if not len(tm):
        return out
    cr = tm["crisis"].to_numpy(dtype=bool)
    for k in MR3_CONFIG["TIMING_DAYS"]:
        out[f"{k}일"] = compare_clustered(tm[cr].rename(columns={f"r{k}": "ret"}), tm[~cr].rename(columns={f"r{k}": "ret"}), "ret")
    return out


def _mr3_limit_view(stage: Dict[str, Any], r: str) -> Dict[str, Any]:
    st = stage["limit"].get(r)
    if not st:
        return {}
    return {**st, "fill_rate": (st["fills"] / st["orders"]) if st["orders"] else None}


def _mr3_sub(stage: Dict[str, Any], key: str, r: str) -> pd.DataFrame:
    t = stage[key]
    return t[t["rule"] == r] if len(t) else t


def run_mr3(prepared: Dict[str, Dict[str, Any]], vix: pd.Series,
            old_loader: Optional[Callable[[], Optional[Dict[str, Dict[str, Any]]]]] = None,
            progress: Optional[Callable[[str, int, int], None]] = None) -> Dict[str, Any]:
    """찾기(545종목 2001~2019, VIX ≥ 20 국면) 5개 규칙 + 섞기 통과선 → 후보만 검증(미국 1990~2000 + 545종목 2020~).
    old_loader = 1990년대 재료를 돌려주는 함수(후보가 있을 때만 부름 — 없으면 그 자료는 받지도 열지도 않음)."""
    cf = MR3_CONFIG
    pg = (lambda s: (lambda i, n: progress(s, i, n))) if progress else (lambda s: None)
    elig = mr_eligibility(prepared)
    ex = mr3_stage(prepared, elig, vix, MR3_RULES, cf["EXPLORE_START"], cf["EXPLORE_END"], progress=pg("찾기 종목"))
    stats = {r: mr3_stats(_mr3_sub(ex, "tr", r)) for r in MR3_RULES}
    strict = {r: mr3_stats(_mr3_sub(ex, "strict", r)) for r in MR3_RULES if r.endswith("-L")}
    comp = [r for r in MR3_RULES if stats[r]["n"] >= cf["MIN_TRADES"] and stats[r]["n_tickers"] >= cf["MIN_TICKERS"]]
    rc = mr3_reality(ex, comp, cf["SEED"], progress=pg(f"섞기 {cf['RC_REPS']}회"))
    thr = max(rc["threshold"], cf["Z_FLOOR"]) if rc["threshold"] is not None else cf["Z_FLOOR"]
    judge = {r: mr3_judge_explore(stats[r], thr, strict.get(r)) for r in MR3_RULES}
    cands = [r for r in MR3_RULES if judge[r]["candidate"]]
    if len(comp) >= 2:
        sub = ex["tr"][ex["tr"]["rule"].isin(comp)].reset_index(drop=True)
        pbo = lec_pbo(sub, {r: np.flatnonzero((sub["rule"] == r).to_numpy()) for r in comp}, "x",
                      (pd.Timestamp(cf["EXPLORE_START"]), pd.Timestamp(cf["EXPLORE_END"])), cf["PBO_SLICES"])
    else:
        pbo = {"pbo": None, "n_splits": 0, "n_combos": len(comp)}
    res: Dict[str, Any] = {
        "config_version": cf["VERSION"], "n_tickers_used": len(prepared), "crisis_share": ex["crisis_share"],
        "elig_days": ex["elig_days"], "crisis_days": ex["crisis_days"], "pool_n": ex["pool_n"], "pool_mean5": ex["pool_mean5"],
        "competing": comp, "threshold_raw": rc["threshold"], "threshold": thr, "max_t": rc["max_t"], "pbo": pbo,
        "timing": mr3_timing(ex), "rules": {}, "candidates": cands, "verify": {}, "verified": [], "verify_status": None,
    }
    for r in MR3_RULES:
        res["rules"][r] = {"stats": stats[r], "judge": judge[r], "limit": _mr3_limit_view(ex, r), "open_end": ex["open_end"].get(r, 0),
                           "mde": mr3_mde(stats[r], thr)}
        if r in strict:
            res["rules"][r]["strict"] = strict[r]
    if not cands:
        res["verify_status"] = "후보 없음 — 1990년대·2020년 이후 자료는 열지 않음"
        return res
    old = old_loader() if old_loader else None
    if not old:
        res["verify_status"] = "검증 보류 — 1990년대 자료를 받지 못함(다시 실행)"
        return res
    vs_new = mr3_stage(prepared, elig, vix, cands, cf["VERIFY_NEW_START"], None, progress=pg("검증 2020~"))
    elig_old = mr_eligibility(old)
    vs_old = mr3_stage(old, elig_old, vix, cands, cf["VERIFY_OLD"][0], cf["VERIFY_OLD"][1], progress=pg("검증 1990~2000"))
    res["verify_meta"] = {"1990~2000": {"tickers": len(old), "crisis_share": vs_old["crisis_share"], "pool_n": vs_old["pool_n"]},
                          "2020~": {"tickers": len(prepared), "crisis_share": vs_new["crisis_share"], "pool_n": vs_new["pool_n"]},
                          "timing_1990": mr3_timing(vs_old), "timing_2020": mr3_timing(vs_new)}
    res["verify_status"] = "검증함"
    for r in cands:
        po, pn = _mr3_sub(vs_old, "tr", r), _mr3_sub(vs_new, "tr", r)
        both = pd.concat([po.assign(part="1990~2000"), pn.assign(part="2020~")], ignore_index=True)
        s = mr3_stats(both)
        parts = {"1990~2000": mr3_stats(po), "2020~": mr3_stats(pn)}
        sx = None
        if r.endswith("-L"):
            sx = mr3_stats(pd.concat([_mr3_sub(vs_old, "strict", r), _mr3_sub(vs_new, "strict", r)], ignore_index=True))
        j = mr3_judge_verify(s, parts, sx)
        res["verify"][r] = {"stats": s, "parts": parts, "judge": j, "strict": sx, "mde": mr3_mde(s, cf["VERIFY_Z"]),
                            "limit_old": _mr3_limit_view(vs_old, r), "limit_new": _mr3_limit_view(vs_new, r)}
    ok = [r for r in cands if res["verify"][r]["judge"]["passed"]]
    res["verified"] = sorted(ok, key=lambda r: -(res["verify"][r]["stats"]["t"] or 0))[:cf["FINAL_MAX"]]
    return res


# ---------------------------------------------------------------------
# 재료 · 최종
# ---------------------------------------------------------------------
MR3_OLD_COLS = ["Open", "High", "Low", "Close", "Volume", "ATR14", "SMA25", "DISP25", "RAW_SH", "RAW_SL"]


def mr3_prepare_old(price_map: Dict[str, pd.DataFrame]) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    """1990년대 검증 재료: 미국 종목만, OLD_DATA_END에서 자르고 필요한 열만 남긴다(메모리). 260봉 미만은 뺀다."""
    cf = MR3_CONFIG
    out, dropped = {}, []
    end = pd.Timestamp(cf["OLD_DATA_END"])
    for sym, df in price_map.items():
        if lec_market_of(sym) != "US":
            continue
        d = lec_clean(df)
        d = d[d.index <= end] if d is not None else None
        if d is None or len(d) < EXPLORE_CONFIG["MIN_BARS"] or d.index[0] > pd.Timestamp(cf["VERIFY_OLD"][1]):
            dropped.append(sym)
            continue
        b = lec_base(d)
        sw = lec_swings(b)
        out[sym] = {"b": b[[k for k in MR3_OLD_COLS if k in b]].copy(), "sw": sw}
    return out, dropped


def run_mr3_final(prepared: Dict[str, Dict[str, Any]], vix: pd.Series, rules: List[str],
                  progress: Optional[Callable[[str, int, int], None]] = None) -> Dict[str, Any]:
    cf = MR3_CONFIG
    elig = mr_eligibility(prepared)
    st = mr3_stage(prepared, elig, vix, rules, cf["FINAL_START"], None,
                   progress=(lambda i, n: progress("최종 종목", i, n)) if progress else None)
    res: Dict[str, Any] = {"config_version": cf["VERSION"], "n_tickers_used": len(prepared), "crisis_share": st["crisis_share"],
                           "pool_n": st["pool_n"], "timing": mr3_timing(st), "rules": {}, "passed": []}
    for r in rules:
        s = mr3_stats(_mr3_sub(st, "tr", r))
        sx = mr3_stats(_mr3_sub(st, "strict", r)) if r.endswith("-L") else None
        j = mr3_judge_final(s, sx)
        res["rules"][r] = {"stats": s, "judge": j, "strict": sx, "limit": _mr3_limit_view(st, r), "mde": mr3_mde(s, cf["FINAL_Z"])}
        if j["passed"]:
            res["passed"].append(r)
    return res


# ---------------------------------------------------------------------
# 보고서 · 기록
# ---------------------------------------------------------------------
def _mr3_t(v: Optional[float]) -> str:
    return "—" if v is None or not np.isfinite(v) else f"{v:+.1f}"


def _mr3_pp(v: Optional[float]) -> str:
    return "—" if v is None or not np.isfinite(v) else f"{v * 100:+.2f}%p"


def _mr3_short(s: Dict[str, Any]) -> str:
    return f"{_mr3_pp(s.get('diff'))} · t {_mr3_t(s.get('t'))} ({s.get('n', 0):,}건)" if s.get("n") else "표본 없음"


def _mr3_mde_txt(v: Optional[float]) -> str:
    return "—" if v is None else f"{v * 100:.2f}%p"


def _mr3_detail(s: Dict[str, Any], lim: Dict[str, Any]) -> List[str]:
    if not s.get("n"):
        return ["- 거래 없음"]
    tot = max(1, s["n"])
    y = s["years"]
    L = [f"- 승률 {s['win'] * 100:.0f}% · 청산 " + " · ".join(f"{a} {v / tot * 100:.0f}%" for a, v in s.get("kinds", {}).items())
         + f" · 평균 보유 {s['hold']:.1f}일 · 종목 {s['n_tickers']}개 · 비용 뒤 순수익 t {_mr3_t(s.get('net_t'))}",
         f"- 미국 {_mr3_short(s['by_market']['US'])} / 한국 {_mr3_short(s['by_market']['KR'])}",
         f"- 기여가 가장 큰 해 {y['drop_year']}를 빼면 {_mr3_pp(y['drop_diff'])} · t {_mr3_t(y['drop_t'])} · 해별 방향 {y['years_ok']}/{y['years_n']}",
         "- 해별 차이(거래 ≥ 30): " + (" · ".join(f"{k} {_mr3_pp(v['diff'])}({v['n']})" for k, v in y["by_year"].items() if v["n"] >= MR3_CONFIG["YEAR_MIN_TRADES"]) or "—"),
         "- VIX 구간(참고): " + (" · ".join(f"{k} {_mr3_short(v)}" for k, v in s.get("bands", {}).items()) or "—")]
    if s.get("periods"):
        L.append("- 기간별: " + " · ".join(f"{p} {_mr3_short(v)}" for p, v in s["periods"].items()))
    if lim:
        L.append(f"- 지정가: 주문 {lim['orders']:,}일 · 체결 {lim['fills']:,}건"
                 f"({'—' if lim.get('fill_rate') is None else format(lim['fill_rate'] * 100, '.0f') + '%'}) · 시가 체결 {lim['open_fills']:,} · 체결일 청산 {lim['fill_day_exit']:,}")
    return L


def _mr3_timing_lines(tm: Dict[str, Any]) -> List[str]:
    out = []
    for k, cm in tm.items():
        a, b_ = cm.get("actual", {}), cm.get("baseline", {})
        if a.get("n") and b_.get("n"):
            out.append(f"- {k} 종가 보유(비용 뒤): 변동성 국면 {_lec_pct(a['mean'])} ({a['n']:,}일) vs 평상시 {_lec_pct(b_['mean'])} ({b_['n']:,}일) · "
                       f"차이 {_mr3_pp(cm.get('diff'))} · z {_mr3_t(cm.get('z'))}")
    return out or ["- 자료 없음"]


def render_mr3_report(r: Dict[str, Any], generated_at: Optional[str] = None) -> str:
    cf = MR3_CONFIG
    L = ["# H-MR3 변동성 국면(VIX ≥ 20) 이격도·지지 반등", "",
         "**찾기 단계 결과 — 이것만으로는 증거가 아니다.** 이미 다른 시험에 쓴 545종목의 2001~2019년이다. 후보는 VIX 조건으로 본 적 없는 "
         "미국 1990~2000 + 545종목 2020년 이후로 검증하고, 통과해야 새 151종목 최종(`--mr3-final`, z ≥ 3)으로 간다(MR3_PREREG.md).", "",
         f"- 생성: {generated_at or '—'} · 설정 {r['config_version']} (사전 등록 {cf['REGISTERED']}) · 종목 {r['n_tickers_used']}개",
         f"- 변동성 국면 = 앞선 미국 거래일 VIX 종가 ≥ {cf['VIX_MIN']:.0f} · 찾기 기간 대상 종목·날 {r.get('elig_days', 0):,}일 중 "
         f"{'—' if r.get('crisis_share') is None else format(r['crisis_share'] * 100, '.0f') + '%'}가 변동성 국면",
         "- 청산: 손절 = 진입가 − 2×ATR(전날), 종가 ≥ 25일선이면 그날 종가, 최대 10거래일(같은 날 둘 다면 손절) · 비용 0.30% · 규칙마다 종목당 1포지션",
         "- 비교 기준: 같은 종목이 대상·변동성 국면이던 날에 종가로 사서 같은 보유일 뒤 종가에 판 평균(비용 뒤, 보유 기간 맞춤)",
         f"- **통과선 t = {r['threshold']:.2f}** (경쟁 {len(r['competing'])}개, 섞기 {cf['RC_REPS']}회 '최고 t' {int(cf['RC_Q'] * 100)}% 분위 "
         f"{'—' if r['threshold_raw'] is None else format(r['threshold_raw'], '.2f')}, 최소 {cf['Z_FLOOR']:.0f}) · PBO(참고) "
         f"{'—' if r['pbo'].get('pbo') is None else format(r['pbo']['pbo'] * 100, '.0f') + '%'}",
         "- 후보 조건: 차이 > 0 · t ≥ 통과선 · 비용 뒤 순수익 > 0 · 거래 ≥ 300 · 종목 ≥ 50 · 미국·한국 같은 방향 · 기여가 가장 큰 해를 빼도 t ≥ 2 · "
         "해별 방향 3분의 2 이상 · 지정가는 엄격 체결로도 t ≥ 2", "",
         "## 판정 (찾기)", "",
         "| 규칙 | 거래(종목) | 거래당 순수익 | 같은 종목 같은 보유일 | 차이 · t | 검출 가능한 최소 차이 | 판정 |", "|---|---|---|---|---|---|---|"]
    for k in MR3_RULES:
        x = r["rules"][k]
        s, j = x["stats"], x["judge"]
        if s["n"]:
            v = "**찾기 후보** → 검증" if j["candidate"] else "후보 아님 (" + ", ".join(j["reasons"]) + ")"
            L.append(f"| {MR3_LABELS[k]} | {s['n']:,} ({s['n_tickers']}) | {_lec_pct(s['net'])} | {_lec_pct(s['base'])} | "
                     f"{_mr3_pp(s['diff'])} · {_mr3_t(s['t'])} | {_mr3_mde_txt(x.get('mde'))} | {v} |")
        else:
            L.append(f"| {MR3_LABELS[k]} | 0 | — | — | — | — | 후보 아님 (거래 없음) |")
    L += ["", "## 검증 (후보만: 미국 1990~2000 + 545종목 2020년~)", "", f"- {r.get('verify_status') or '—'}"]
    if r["verify"]:
        L += [f"- 기준: 합쳐서 차이 > 0, t ≥ {cf['VERIFY_Z']:.0f}, 거래 ≥ {cf['VERIFY_MIN_TRADES']}, 비용 뒤 순수익 > 0, 30건 이상인 구간은 각각 차이 > 0", "",
              "| 규칙 | 합친 차이 · t | 1990~2000 | 2020~ | 거래당 순수익 | 최소 차이 | 판정 |", "|---|---|---|---|---|---|---|"]
        for k, v in r["verify"].items():
            s = v["stats"]
            L.append(f"| {MR3_LABELS[k]} | {_mr3_short(s)} | {_mr3_short(v['parts']['1990~2000'])} | {_mr3_short(v['parts']['2020~'])} | "
                     f"{_lec_pct(s.get('net'))} | {_mr3_mde_txt(v.get('mde'))} | "
                     + ("**검증 통과** → 최종 대기" if v["judge"]["passed"] else "미통과 (" + ", ".join(v["judge"]["reasons"]) + ")") + " |")
        if r["verified"]:
            L.append(f"\n- 최종으로 갈 규칙: " + ", ".join(MR3_LABELS[k] for k in r["verified"]))
    L += ["", "## 참고 — 시점 효과(대상인 날, 변동성 국면 vs 평상시, 판정 아님)", ""] + _mr3_timing_lines(r.get("timing", {}))
    if r.get("verify_meta"):
        L += ["", "검증 구간:"] + ["  " + t for t in _mr3_timing_lines(r["verify_meta"].get("timing_1990", {}))] \
             + ["  " + t for t in _mr3_timing_lines(r["verify_meta"].get("timing_2020", {}))]
    L += ["", "## 참고 — 규칙별 자세히(찾기)", ""]
    for k in MR3_RULES:
        x = r["rules"][k]
        L += [f"**{MR3_LABELS[k]}**", ""] + _mr3_detail(x["stats"], x["limit"])
        if x.get("strict"):
            L.append(f"- 엄격 체결(판정 조건): {_mr3_short(x['strict'])} · 거래당 순수익 {_lec_pct(x['strict'].get('net'))}")
        if k in r["verify"]:
            v = r["verify"][k]
            L += ["- 검증 자세히:"] + ["  " + t for t in _mr3_detail(v["stats"], {})]
            if v.get("strict"):
                L.append(f"  - 검증 엄격 체결: {_mr3_short(v['strict'])}")
        L.append("")
    L += ["## 읽는 법", "",
          "- '차이'는 거래당 순수익(비용 뒤)에서 같은 종목을 변동성 국면의 아무 날에 사서 같은 날 수만큼 들고 판 평균을 뺀 값(선별 효과)이다.",
          "- '시점 효과'는 변동성 국면에 사는 것 자체가 평상시보다 나은지(참고)다. 둘 다 좋아야 '변동성 국면 단타'가 말이 된다.",
          "- '검출 가능한 최소 차이'보다 작은 우위는 있어도 이 표본으로는 잡지 못한다. 미통과가 곧 '효과 없음'은 아니다.",
          "- 후보가 아니면 여기서 끝낸다. 결과를 보고 VIX 기준·규칙·청산을 바꿔 다시 보지 않는다.", ""]
    return "\n".join(L)


def render_mr3_final_report(r: Dict[str, Any], dropped: List[str], generated_at: Optional[str] = None) -> str:
    cf = MR3_CONFIG
    L = ["# H-MR3 최종 — 새 151종목(2001~)", "",
         f"- 생성: {generated_at or '—'} · 설정 {r['config_version']} · 종목 {r['n_tickers_used']}개(빠진 종목 {len(dropped)}개: {', '.join(dropped) or '없음'})",
         f"- 변동성 국면 비율 {'—' if r.get('crisis_share') is None else format(r['crisis_share'] * 100, '.0f') + '%'} · 기준: 차이 > 0, t ≥ {cf['FINAL_Z']:.0f}, "
         f"거래 ≥ {cf['FINAL_MIN_TRADES']}, 비용 뒤 순수익 > 0, 미국·한국 같은 방향", "",
         "| 규칙 | 거래당 순수익 | 같은 종목 같은 보유일 | 차이 · t | 최소 차이 | 판정 |", "|---|---|---|---|---|---|"]
    for k, x in r["rules"].items():
        s, j = x["stats"], x["judge"]
        L.append(f"| {MR3_LABELS[k]} | {_lec_pct(s.get('net'))} ({s['n']:,}건) | {_lec_pct(s.get('base'))} | {_mr3_pp(s['diff'])} · {_mr3_t(s['t'])} | "
                 f"{_mr3_mde_txt(x.get('mde'))} | " + ("**통과**" if j["passed"] else "미통과 (" + ", ".join(j["reasons"]) + ")") + " |")
    L += ["", "시점 효과(참고):"] + _mr3_timing_lines(r.get("timing", {})) + [""]
    for k, x in r["rules"].items():
        L += [f"**{MR3_LABELS[k]}**", ""] + _mr3_detail(x["stats"], x["limit"])
        if x.get("strict"):
            L.append(f"- 엄격 체결: {_mr3_short(x['strict'])}")
        L.append("")
    L += ["- 이 단계는 한 번만 한다. 통과 = 앱 '통과'. 그래도 앞으로의 실제 결과(장부)로 다시 확인한다.", ""]
    return "\n".join(L)


def mr3_log_lines(r: Dict[str, Any], generated_at: str) -> List[str]:
    rows = []
    for k in MR3_RULES:
        s = r["rules"][k]["stats"]
        verdict = "후보" if r["rules"][k]["judge"]["candidate"] else "후보 아님"
        if k in r.get("verify", {}):
            verdict += " → 검증 " + ("통과" if r["verify"][k]["judge"]["passed"] else "미통과")
        rows.append(f"| {generated_at} | `mr3-{k}` | {MR3_LABELS[k]} — VIX ≥ 20 국면, 2 ATR 손절·25일선 회복·10일, 차이 = 순수익 − 같은 종목 같은 보유일 "
                    f"(찾기 545종목 2001~2019, 통과선 t {r['threshold']:.2f}) | {s.get('n', 0):,} | {_mr3_pp(s.get('diff'))} | {_mr3_t(s.get('t'))} | {verdict} |")
    return rows


def mr3_final_log_lines(r: Dict[str, Any], generated_at: str) -> List[str]:
    return [f"| {generated_at} | `mr3-final-{k}` | {MR3_LABELS[k]} — 최종(새 151종목 2001~, z ≥ 3) | {x['stats'].get('n', 0):,} | "
            f"{_mr3_pp(x['stats'].get('diff'))} | {_mr3_t(x['stats'].get('t'))} | {'통과' if x['judge']['passed'] else '미통과'} |"
            for k, x in r["rules"].items()]
