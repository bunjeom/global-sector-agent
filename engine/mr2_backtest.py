"""H-MR2 단기 신호 찾기 — 신호 즉시 진입(종가·지정가) · +5% 익절 · −5% 손절 · 최대 3거래일
(2026-11-19 사전 등록, data/backtest/MR2_PREREG.md · AGENTS.md 4장).

신호 15개를 규칙 26개(종가 진입 18 + 지정가 진입 8)로 한 번에 계산하고, 섞기 200회 통과선(White Reality Check,
강의 L2와 같은 방식)으로 '26개 중 가장 좋아 보이는 것'이 우연인지 가린다.
    종가 진입   신호일 t 종가에 산다. t+1~t+3일: 시가가 −5% 아래면 시가 손절 · +5% 위면 시가 익절 · 저가 ≤ −5% 손절
                (같은 날 +5%도 닿았어도 손절) · 고가 ≥ +5% 익절 · t+3일 종가.
    지정가 진입 전날까지의 자료로 정한 가격 P. 저가 ≤ P면 min(시가, P)에 체결. 체결일 저가 ≤ 체결가 × 0.95면 그날 손절,
                체결일 +5%는 세지 않음. 그 뒤는 종가 진입과 같다. 지정가 규칙은 '엄격 체결'(시가 ≤ P 또는 저가 ≤ P × 0.999)로도
                차이 > 0 · t ≥ 2여야 후보(닿기만 하면 체결이라는 가정은 딱 닿고 튄 가장 좋은 날을 넣어 낙관적이라).
    대상        mr1-2와 같음(거래대금 상위 절반 중 변동성 상위 절반, mr_eligibility). 종가 = t일 대상, 지정가 = t−1일 대상.
    비교 기준   같은 종목이 대상이었던 같은 기간 모든 날의 종가 진입(같은 청산) 평균. 값 = 거래 순수익 − 그 종목 평균.
    단계        찾기(기존 545종목 2001~2019) → 후보만 검증(같은 종목 2020~) → 검증 통과만 최종(새 151종목, --mr2-final).
숫자는 MR2_CONFIG 한 곳 — 결과를 보고 바꾸지 않는다.
"""
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .explore import EXPLORE_CONFIG
from .frame_backtest import compare_clustered
from .lecture_backtest import _lec_pct, lec_clean, lec_market_of
from .lecture_family import _lec_month_drop_one, lec_cluster_t, lec_month_code, lec_null_pool, lec_pbo, lec_reality_check
from .lecture_levels import lec_swing_lines
from .lecture_signals import LECTURE_CONFIG, lec_base, lec_bnf_base_series, lec_swings
from .mr_backtest import MR1_CONFIG, _mr_bounds, mr_eligibility, mr_signals
from .strategy import short_rsi2

MR2_CONFIG: Dict[str, Any] = {
    "VERSION": "mr2-1",
    "REGISTERED": "2026-11-19",
    "EXPLORE_START": "2001-01-01",     # 찾기: EXPLORE_CONFIG 545종목(이미 쓴 종목), 신호일 2001~2019
    "EXPLORE_END": "2019-12-31",
    "VERIFY_START": "2020-01-01",      # 검증: 같은 종목 2020~끝(아직 안 연 기간), 찾기 후보만
    "FINAL_START": "2001-01-01",       # 최종: LECTURE_CONFIG FINAL_UNIVERSE 151종목, 2001~끝, 검증 통과만
    # --- 매매 ---
    "TP": 0.05,                        # +5% 익절
    "SL": 0.05,                        # −5% 손절 (같은 날 둘 다 닿으면 손절)
    "MAX_DAYS": 3,                     # 진입일 다음 날부터 3거래일, 마지막 날 종가
    "COST": 0.003,                     # 왕복 비용
    "STRICT_FILL": 0.999,              # 엄격 체결 = 시가 ≤ P 이거나 저가 ≤ P × 0.999 (지정가 규칙은 이것으로도 기준을 넘어야 함)
    "STRICT_Z": 2.0,                   # 지정가 규칙: 엄격 체결로도 차이 > 0 이고 t ≥ 2 (찾기·검증·최종 모두, 2026-11-19 결과 전 사용자 결정)
    # --- 신호 숫자 ---
    "ATR_DROP": {"S01a": 2.0, "S01b": 3.0},
    "PCT_DROP": {"S02a": 0.05, "S02b": 0.08},
    "GAP_DOWN": 0.03,
    "DOWN_DAYS": {"S04a": 3, "S04b": 4},
    "DROP5_ATR": -3.0,                 # S05 = MR1 규칙 C와 같은 값
    "RSI2_LOW": 10.0,
    "SMA_LONG": 200,
    "RSI14_LOW": 30.0,
    "NEW_LOW_DAYS": 20,
    "RVOL_MIN": 2.0,
    # --- 판정 ---
    "MIN_TRADES": 300,                 # 찾기: 경쟁·후보 조건
    "MIN_TICKERS": 50,
    "MIN_MARKET": 30,                  # 미국·한국 방향은 둘 다 이 이상일 때만
    "Z_FLOOR": 2.0,                    # 통과선이 이보다 낮으면 이 값
    "MONTH_DROP_Z": 2.0,
    "RC_REPS": 200, "RC_Q": 0.95, "PBO_SLICES": 16,
    "VERIFY_Z": 2.0, "VERIFY_MIN_TRADES": 100,
    "FINAL_Z": 3.0, "FINAL_MIN_TRADES": 100, "FINAL_MAX": 5,
    "PERIOD_BLOCKS": (("2001-01-01", "2007-12-31"), ("2008-01-01", "2013-12-31"), ("2014-01-01", "2019-12-31")),
    "SEED": 1120,
}

# 신호 번호 → (이름, 지정가 규칙이 있나)
MR2_SIGNALS: Dict[str, Tuple[str, bool]] = {
    "S01a": ("하루 급락 2 ATR", True),
    "S01b": ("하루 급락 3 ATR", True),
    "S02a": ("하루 −5%", True),
    "S02b": ("하루 −8%", True),
    "S03": ("갭 하락 3%", False),
    "S04a": ("3일 연속 하락", False),
    "S04b": ("4일 연속 하락", False),
    "S05": ("5일 누적 −3 ATR(처음)", True),
    "S06": ("RSI(2) < 10", False),
    "S07": ("RSI(2) < 10 + 200일선 위", False),
    "S08": ("RSI(14) ≤ 30(처음)", False),
    "S09": ("볼린저 하단 아래 종가", False),
    "S10": ("BNF 기준 이격도(처음)", True),
    "S11": ("20일 신저가", True),
    "S12": ("지지 구간 도달", True),
    "S13": ("대량 거래 음봉", False),
    "S14": ("BNF 확인봉(MR1 B)", False),
    "S15": ("장대음봉", False),
}
MR2_RULES: List[str] = [f"{s}-C" for s in MR2_SIGNALS] + [f"{s}-L" for s, (_, lim) in MR2_SIGNALS.items() if lim]
MR2_KIND = {1: "익절", 2: "손절", 3: "시간", 4: "시가 익절", 5: "시가 손절", 6: "체결일 손절"}


def mr2_label(rule: str) -> str:
    sid, mode = rule.split("-")
    return f"{sid} {MR2_SIGNALS[sid][0]} · {'종가' if mode == 'C' else '지정가'}"


def _mr2_shift(a: np.ndarray, k: int) -> np.ndarray:
    """위치 t에 a[t−k] (앞은 NaN)."""
    out = np.full(len(a), np.nan)
    if 0 < k < len(a):
        out[k:] = a[:-k]
    return out


# ---------------------------------------------------------------------
# 청산 (+5% · −5% · 3거래일)
# ---------------------------------------------------------------------
def mr2_bracket_all(o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray) -> Dict[str, np.ndarray]:
    """모든 날 t에 종가로 샀을 때의 결과(비용 전). t + 3 > 끝이면 NaN. kind: 1 익절 · 2 손절 · 3 시간 · 4 시가 익절 · 5 시가 손절."""
    cf = MR2_CONFIG
    D = cf["MAX_DAYS"]
    n = len(c)
    gross = np.full(n, np.nan)
    kind = np.zeros(n, dtype=np.int8)
    hold = np.zeros(n, dtype=np.int64)
    m = n - D
    if m <= 0:
        return {"gross": gross, "kind": kind, "hold": hold}
    E = c[:m].astype(float)
    sl, tp = E * (1.0 - cf["SL"]), E * (1.0 + cf["TP"])
    px = np.full(m, np.nan)
    kd = np.zeros(m, dtype=np.int8)
    hd = np.zeros(m, dtype=np.int64)
    done = np.zeros(m, dtype=bool)
    base = np.arange(m)
    for k in range(1, D + 1):
        d = base + k
        od, hh, ld, cd = o[d], h[d], l[d], c[d]
        a = ~done
        r5 = a & (od <= sl)
        r4 = a & ~r5 & (od >= tp)
        r2 = a & ~r5 & ~r4 & (ld <= sl)
        r1 = a & ~r5 & ~r4 & ~r2 & (hh >= tp)
        r3 = a & ~(r5 | r4 | r2 | r1) if k == D else np.zeros(m, dtype=bool)
        for msk, val, code in ((r5, od, 5), (r4, od, 4), (r2, sl, 2), (r1, tp, 1), (r3, cd, 3)):
            px[msk] = val[msk]
            kd[msk] = code
            hd[msk] = k
            done |= msk
    gross[:m] = px / E - 1.0
    kind[:m] = kd
    hold[:m] = hd
    return {"gross": gross, "kind": kind, "hold": hold}


def mr2_after(o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray, t: int, E: float) -> Tuple[float, int, int]:
    """진입가 E로 t일에 산 뒤 t+1~t+3일 청산(mr2_bracket_all과 같은 규칙). (비용 전 수익, kind, 보유일)."""
    cf = MR2_CONFIG
    D = cf["MAX_DAYS"]
    sl, tp = E * (1.0 - cf["SL"]), E * (1.0 + cf["TP"])
    for k in range(1, D + 1):
        d = t + k
        if o[d] <= sl:
            return o[d] / E - 1.0, 5, k
        if o[d] >= tp:
            return o[d] / E - 1.0, 4, k
        if l[d] <= sl:
            return sl / E - 1.0, 2, k
        if h[d] >= tp:
            return tp / E - 1.0, 1, k
    return c[t + D] / E - 1.0, 3, D


# ---------------------------------------------------------------------
# 신호 (종가 진입: 그날 종가까지 · 지정가: 전날까지의 자료로 가격)
# ---------------------------------------------------------------------
def mr2_support_zone(b: pd.DataFrame, sw: Dict[str, np.ndarray], need: np.ndarray,
                     lines: Optional[Dict[str, np.ndarray]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """S12: t일마다 '전날까지 살아 있는 확정 스윙 수평선 중 구간 상단(ATR_{t−1}) < 전날 종가인 가장 높은 선'의 구간 [하단, 상단].
    need가 True인 날만 계산(나머지 NaN)."""
    lc = LECTURE_CONFIG
    n = len(b)
    zl = np.full(n, np.nan)
    zu = np.full(n, np.nan)
    lines = lines if lines is not None else lec_swing_lines(b, sw)
    lv, st, fr = lines["lv"], lines["start"], lines["fr"]
    if not len(lv):
        return zl, zu
    cl = b["Close"].to_numpy(dtype=float)
    atr = b["ATR14"].to_numpy(dtype=float)
    for t in np.flatnonzero(need):
        y = t - 1
        if y < 0:
            continue
        k = int(np.searchsorted(st, y, "right"))
        if k == 0:
            continue
        L_, S_, F_ = lv[:k], st[:k], fr[:k]
        alive = ((y - S_) <= lc["LINE_IDLE_DAYS"]) | (F_ <= y)
        a = atr[y]
        if np.isfinite(a) and a > 0:
            hw = np.maximum(lc["ZONE_PCT_MIN"] * L_, lc["ZONE_ATR_MULT"] * a)
        else:
            hw = np.maximum(lc["ZONE_PCT_MIN"] * L_, lc["ZONE_ATR_MULT"] * L_ * 0.02)     # lec_zone과 같은 대체값
        up = L_ + hw
        ok = alive & (up < cl[y])
        if not ok.any():
            continue
        j = int(np.argmax(np.where(ok, L_, -np.inf)))
        zl[t], zu[t] = L_[j] - hw[j], up[j]
    return zl, zu


def mr2_signal_arrays(b: pd.DataFrame, sw: Dict[str, np.ndarray], sids: set, need: Optional[np.ndarray] = None,
                      cache: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """신호 번호마다 종가 진입 신호(True/False)와 지정가 P(없으면 NaN). need = S12 지지선을 계산할 날(None이면 전부)."""
    cf, lc = MR2_CONFIG, LECTURE_CONFIG
    cache = cache if cache is not None else {}
    n = len(b)
    o, h, l, c = (b[k].to_numpy(dtype=float) for k in ("Open", "High", "Low", "Close"))
    atr = b["ATR14"].to_numpy(dtype=float)
    pc, patr = _mr2_shift(c, 1), _mr2_shift(atr, 1)
    sig: Dict[str, np.ndarray] = {}
    px: Dict[str, np.ndarray] = {}
    with np.errstate(invalid="ignore"):
        for s, k in cf["ATR_DROP"].items():
            if s in sids:
                thr = pc - k * patr
                sig[s] = c <= thr
                px[s] = np.where(thr > 0, thr, np.nan)
        for s, k in cf["PCT_DROP"].items():
            if s in sids:
                thr = pc * (1.0 - k)
                sig[s] = c <= thr
                px[s] = thr
        if "S03" in sids:
            sig["S03"] = o <= pc * (1.0 - cf["GAP_DOWN"])
        for s, k in cf["DOWN_DAYS"].items():
            if s in sids:
                m = np.ones(n, dtype=bool)
                for i in range(k):
                    m &= _mr2_shift(c, i) < _mr2_shift(c, i + 1) if i else c < pc
                sig[s] = m
        if "S05" in sids:
            dr = b["DROP5"].to_numpy(dtype=float)
            pdr = _mr2_shift(dr, 1)
            sig["S05"] = (dr <= cf["DROP5_ATR"]) & (pdr > cf["DROP5_ATR"])
            p5 = _mr2_shift(c, lc["SLOPE_DAYS"]) + cf["DROP5_ATR"] * patr
            px["S05"] = np.where((pdr > cf["DROP5_ATR"]) & (p5 > 0), p5, np.nan)
        if "S06" in sids or "S07" in sids:
            r2 = short_rsi2(b["Close"]).to_numpy(dtype=float)
            if "S06" in sids:
                sig["S06"] = r2 < cf["RSI2_LOW"]
            if "S07" in sids:
                sma = b["Close"].astype(float).rolling(cf["SMA_LONG"], min_periods=cf["SMA_LONG"]).mean().to_numpy()
                sig["S07"] = (r2 < cf["RSI2_LOW"]) & (c > sma)
        if "S08" in sids:
            r14 = b["RSI14"].to_numpy(dtype=float)
            sig["S08"] = (r14 <= cf["RSI14_LOW"]) & (_mr2_shift(r14, 1) > cf["RSI14_LOW"])
        if "S09" in sids:
            sig["S09"] = c < b["BB_LOW"].to_numpy(dtype=float)
        if "S10" in sids or "S14" in sids:
            if "base" not in cache:
                cache["base"] = lec_bnf_base_series(b, sw)
        if "S10" in sids:
            disp = b["DISP25"].to_numpy(dtype=float)
            pb, pdisp = _mr2_shift(cache["base"], 1), _mr2_shift(disp, 1)
            first = pdisp > pb
            sig["S10"] = first & (disp <= pb)
            s24 = _mr2_shift(pd.Series(c).rolling(24, min_periods=24).sum().to_numpy(), 1)
            p10 = pb * s24 / (2500.0 - pb)
            px["S10"] = np.where(first & (p10 > 0), p10, np.nan)
        if "S11" in sids:
            m20 = _mr2_shift(pd.Series(l).rolling(cf["NEW_LOW_DAYS"], min_periods=cf["NEW_LOW_DAYS"]).min().to_numpy(), 1)
            sig["S11"] = l < m20
            px["S11"] = m20
        if "S12" in sids:
            if "lines" not in cache:
                cache["lines"] = lec_swing_lines(b, sw)
            zl, zu = mr2_support_zone(b, sw, np.ones(n, dtype=bool) if need is None else need, cache["lines"])
            sig["S12"] = (l <= zu) & (c >= zl)
            px["S12"] = zu
        if "S13" in sids:
            sig["S13"] = (c < o) & (b["RVOL"].to_numpy(dtype=float) >= cf["RVOL_MIN"])
        if "S14" in sids:
            m = np.zeros(n, dtype=bool)
            ts = mr_signals(b, sw, "B", base=cache["base"])["t"]
            m[np.asarray(ts, dtype=np.int64)] = True
            sig["S14"] = m
        if "S15" in sids:
            sig["S15"] = b["BIG_BEAR"].to_numpy(dtype=bool)
    sig = {k: np.asarray(v, dtype=bool) for k, v in sig.items()}
    return sig, px


# ---------------------------------------------------------------------
# 종목 하나
# ---------------------------------------------------------------------
def mr2_ticker(b: pd.DataFrame, sw: Dict[str, np.ndarray], el: np.ndarray, lo_i: int, hi_i: int,
               rules: List[str], strict: bool = True) -> Dict[str, Any]:
    """종목 하나, 신호일 [lo_i, hi_i]: 비교 기준(대상인 모든 날 종가 진입)과 규칙별 거래. 종목당 1포지션(판 날 다음 날부터 새 신호)."""
    cf = MR2_CONFIG
    D = cf["MAX_DAYS"]
    n = len(b)
    o, h, l, c = (b[k].to_numpy(dtype=float) for k in ("Open", "High", "Low", "Close"))
    el = np.asarray(el, dtype=bool)
    pel = np.r_[False, el[:-1]]
    inper = np.zeros(n, dtype=bool)
    inper[lo_i:hi_i + 1] = True
    sids = {r.split("-")[0] for r in rules}
    need = inper & (el | pel)
    sig, px = mr2_signal_arrays(b, sw, sids, need=need)
    bk = mr2_bracket_all(o, h, l, c)
    pool = np.flatnonzero(inper & el & np.isfinite(bk["gross"]))
    out: Dict[str, Any] = {"pool_t": pool, "pool_net": bk["gross"][pool] - cf["COST"], "rules": {}, "strict": {}, "limit": {}, "open_end": {}}
    for r in rules:
        sid, mode = r.split("-")
        if mode == "C":
            cand = np.flatnonzero(sig[sid] & el & inper)
            ts, gs, ks, hs = [], [], [], []
            free, oe = -1, 0
            for t in cand:
                if t <= free:
                    continue
                if not np.isfinite(bk["gross"][t]):
                    oe += 1
                    break
                ts.append(t)
                gs.append(bk["gross"][t])
                ks.append(int(bk["kind"][t]))
                hs.append(int(bk["hold"][t]))
                free = t + int(bk["hold"][t])
            out["rules"][r] = {"t": np.array(ts, dtype=np.int64), "entry": c[np.array(ts, dtype=np.int64)] if ts else np.array([]),
                               "gross": np.array(gs, dtype=float), "kind": np.array(ks, dtype=np.int8),
                               "hold": np.array(hs, dtype=np.int64), "open_fill": np.zeros(len(ts), dtype=bool)}
            out["open_end"][r] = oe
            continue
        P = px[sid]
        cand = np.flatnonzero(np.isfinite(P) & (P > 0) & pel & inper)
        for variant in ((False, True) if strict else (False,)):
            ts, es, gs, ks, hs, of = [], [], [], [], [], []
            free, oe = -1, 0
            st = {"orders": 0, "fills": 0, "fill_day_stop": 0, "open_fills": 0}
            for t in cand:
                if t <= free:
                    continue
                if t + D > n - 1:
                    oe += 1
                    break
                st["orders"] += 1
                p = P[t]
                hit = (o[t] <= p) or (l[t] <= p * cf["STRICT_FILL"]) if variant else (l[t] <= p)
                if not hit:
                    continue
                F = min(o[t], p)
                st["fills"] += 1
                st["open_fills"] += int(o[t] <= p)
                if l[t] <= F * (1.0 - cf["SL"]):
                    g, kd, hd = -cf["SL"], 6, 0
                    st["fill_day_stop"] += 1
                else:
                    g, kd, hd = mr2_after(o, h, l, c, t, F)
                ts.append(t)
                es.append(F)
                gs.append(g)
                ks.append(kd)
                hs.append(hd)
                of.append(o[t] <= p)
                free = t + hd
            rec = {"t": np.array(ts, dtype=np.int64), "entry": np.array(es, dtype=float), "gross": np.array(gs, dtype=float),
                   "kind": np.array(ks, dtype=np.int8), "hold": np.array(hs, dtype=np.int64), "open_fill": np.array(of, dtype=bool)}
            if variant:
                out["strict"][r] = rec
            else:
                out["rules"][r] = rec
                out["limit"][r] = st
                out["open_end"][r] = oe
    return out


# ---------------------------------------------------------------------
# 종목 전부 (한 단계)
# ---------------------------------------------------------------------
def _mr2_frame(parts: List[pd.DataFrame]) -> pd.DataFrame:
    cols = ["rule", "ticker", "market", "t", "date", "entry", "gross", "kind", "hold", "open_fill"]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=cols)


def mr2_stage(prepared: Dict[str, Dict[str, Any]], eligible: Dict[str, np.ndarray], rules: List[str],
              start: str, end: Optional[str], progress: Optional[Callable[[int, int], None]] = None,
              strict: bool = True) -> Dict[str, Any]:
    """한 단계(기간)의 거래 전부 + 비교 기준. 값 x = 순수익 − 그 종목 기준 평균, mref = 같은 날 같은 시장 대상 종목 평균(참고)."""
    cf = MR2_CONFIG
    parts, sparts = [], []
    pools: Dict[str, pd.Series] = {}
    lim = {r: {"orders": 0, "fills": 0, "fill_day_stop": 0, "open_fills": 0} for r in rules if r.endswith("-L")}
    open_end = {r: 0 for r in rules}
    for i, (sym, p) in enumerate(prepared.items()):
        b = p["b"]
        lo_i, hi_i = _mr_bounds(b.index, start, end)
        el = eligible.get(sym)
        if progress:
            progress(i + 1, len(prepared))
        if hi_i < lo_i or el is None or not np.any(el[max(0, lo_i - 1):hi_i + 1]):
            continue
        res = mr2_ticker(b, p["sw"], el, lo_i, hi_i, rules, strict=strict)
        mk = lec_market_of(sym)
        pools[sym] = pd.Series(res["pool_net"], index=b.index[res["pool_t"]])
        for key, store in (("rules", parts), ("strict", sparts)):
            for r, d in res[key].items():
                if len(d["t"]):
                    store.append(pd.DataFrame({"rule": r, "ticker": sym, "market": mk, "t": d["t"], "date": b.index[d["t"]],
                                               "entry": d["entry"], "gross": d["gross"], "kind": d["kind"], "hold": d["hold"],
                                               "open_fill": d["open_fill"]}))
        for r, st in res["limit"].items():
            for k, v in st.items():
                lim[r][k] += v
        for r, v in res["open_end"].items():
            open_end[r] += v
    mu = {s: float(v.mean()) for s, v in pools.items() if len(v)}
    mref = {}
    for m in ("US", "KR"):
        ss = {s: v for s, v in pools.items() if lec_market_of(s) == m and len(v)}
        mref[m] = pd.concat(ss, axis=1).mean(axis=1) if ss else pd.Series(dtype=float)

    def finish(tr: pd.DataFrame) -> pd.DataFrame:
        if not len(tr):
            return tr.assign(net=[], base=[], x=[], mref=[], xm=[])
        tr = tr.copy()
        tr["net"] = tr["gross"].astype(float) - cf["COST"]
        tr["base"] = tr["ticker"].map(mu).astype(float)
        tr["x"] = tr["net"] - tr["base"]
        mr = np.full(len(tr), np.nan)
        for m in ("US", "KR"):
            mm = (tr["market"] == m).to_numpy()
            if mm.any() and len(mref[m]):
                mr[mm] = mref[m].reindex(pd.DatetimeIndex(tr.loc[mm, "date"])).to_numpy(dtype=float)
        tr["mref"] = mr
        tr["xm"] = tr["net"] - tr["mref"]
        return tr[np.isfinite(tr["x"].to_numpy(dtype=float))].reset_index(drop=True)
    pool_dm = {s: v - mu[s] for s, v in pools.items() if s in mu}
    return {"tr": finish(_mr2_frame(parts)), "strict": finish(_mr2_frame(sparts)), "pool": pool_dm, "mu": mu,
            "pool_n": int(sum(len(v) for v in pools.values())), "pool_mean": (float(np.mean(np.concatenate([v.to_numpy() for v in pools.values()])))
                                                                             if pools and sum(len(v) for v in pools.values()) else None),
            "limit": lim, "open_end": open_end, "period": (start, end)}


# ---------------------------------------------------------------------
# 통계 · 판정
# ---------------------------------------------------------------------
def mr2_stats(tr: pd.DataFrame, col: str = "x") -> Dict[str, Any]:
    """규칙 하나: 차이(= 평균 x) · 월 묶음 t · 종목 수 · 미국/한국 · 달 쏠림 · 참고 숫자."""
    if not len(tr):
        return {"n": 0, "n_tickers": 0, "diff": None, "t": None, "by_market": {m: {"n": 0, "diff": None, "t": None} for m in ("US", "KR")},
                "month_drop": {"month": None, "net": None, "t": None}}
    x = tr[col].to_numpy(dtype=float)
    g = lec_month_code(tr["date"])
    r = lec_cluster_t(x, g)
    out = {"n": r["n"], "diff": r["mean"], "t": r["t"], "n_months": r["k"], "n_tickers": int(tr["ticker"].nunique()),
           "net": float(tr["net"].mean()), "base": float(tr["base"].mean()),
           "win": float((tr["net"] > 0).mean()), "hold": float(tr["hold"].mean()),
           "kinds": {MR2_KIND[int(k)]: int(v) for k, v in tr["kind"].value_counts().sort_index().items()}, "by_market": {}}
    for m in ("US", "KR"):
        mm = tr["market"].to_numpy() == m
        rr = lec_cluster_t(x[mm], g[mm])
        out["by_market"][m] = {"n": rr["n"], "diff": rr["mean"], "t": rr["t"]}
    out["month_drop"] = _lec_month_drop_one(x, g, 0.0)
    xm = tr["xm"].to_numpy(dtype=float) if "xm" in tr else np.array([])
    fm = np.isfinite(xm)
    rm = lec_cluster_t(xm[fm], g[fm]) if fm.any() else {"n": 0, "mean": None, "t": None}
    out["vs_market"] = {"n": rm["n"], "diff": rm["mean"], "t": rm["t"]}
    blocks = {}
    d = pd.to_datetime(tr["date"])
    for s0, s1 in MR2_CONFIG["PERIOD_BLOCKS"]:
        mb = ((d >= s0) & (d <= s1)).to_numpy()
        if mb.any():
            rb = lec_cluster_t(x[mb], g[mb])
            blocks[f"{s0[:4]}~{s1[:4]}"] = {"n": rb["n"], "diff": rb["mean"], "t": rb["t"]}
    out["periods"] = blocks
    return out


def mr2_market_conflict(s: Dict[str, Any]) -> bool:
    mk = [v for v in s["by_market"].values() if v["n"] >= MR2_CONFIG["MIN_MARKET"] and v["diff"] is not None]
    return len(mk) == 2 and np.sign(mk[0]["diff"]) != np.sign(mk[1]["diff"])


def _mr2_strict_reason(strict: Optional[Dict[str, Any]]) -> List[str]:
    """지정가 규칙만: 엄격 체결(시가 ≤ P 또는 저가 ≤ P × 0.999)로도 차이 > 0 이고 t ≥ 2."""
    if strict is None:
        return []
    z = MR2_CONFIG["STRICT_Z"]
    if not strict.get("n") or strict.get("diff") is None or strict["diff"] <= 0 or strict.get("t") is None or strict["t"] < z:
        return [f"엄격 체결로는 기준 미달(차이 > 0, t ≥ {z:.0f} 아님)"]
    return []


def mr2_judge_explore(s: Dict[str, Any], threshold: float, strict: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cf = MR2_CONFIG
    rs = []
    if s["n"] < cf["MIN_TRADES"]:
        rs.append(f"거래 {s['n']:,} < {cf['MIN_TRADES']}")
    if s["n_tickers"] < cf["MIN_TICKERS"]:
        rs.append(f"종목 {s['n_tickers']} < {cf['MIN_TICKERS']}")
    if s["diff"] is None or s["diff"] <= 0:
        rs.append("기준보다 나쁨(차이 ≤ 0)")
    if s["t"] is None or not np.isfinite(s["t"]) or s["t"] < threshold:
        rs.append(f"t < 통과선 {threshold:.2f}")
    if mr2_market_conflict(s):
        rs.append("미국·한국 방향이 다름")
    md = s["month_drop"]
    if md["net"] is None or md["net"] <= 0 or md["t"] is None or md["t"] < cf["MONTH_DROP_Z"]:
        rs.append(f"기여가 가장 큰 달({md['month']})을 빼면 기준 미달")
    rs += _mr2_strict_reason(strict)
    return {"candidate": not rs, "reasons": rs}


def mr2_judge_simple(s: Dict[str, Any], z_need: float, n_need: int, check_market: bool,
                     strict: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    rs = []
    if s["n"] < n_need:
        rs.append(f"거래 {s['n']:,} < {n_need}")
    if s["diff"] is None or s["diff"] <= 0:
        rs.append("기준보다 나쁨(차이 ≤ 0)")
    if s["t"] is None or not np.isfinite(s["t"]) or s["t"] < z_need:
        rs.append(f"t < {z_need:.0f}")
    if check_market and mr2_market_conflict(s):
        rs.append("미국·한국 방향이 다름")
    rs += _mr2_strict_reason(strict)
    return {"passed": not rs, "reasons": rs}


def mr2_reality(stage: Dict[str, Any], rules: List[str], seed: int,
                progress: Optional[Callable[[int, int], None]] = None) -> Dict[str, Any]:
    """섞기 통과선: 거래(종목·신호일)마다 같은 종목의 무작위 대상 날 하나(같은 종목·신호일은 모든 규칙에서 같은 날) → 규칙별 t → 최고 t."""
    cf = MR2_CONFIG
    tr = stage["tr"]
    if not rules or not len(tr):
        return {"max_t": [], "threshold": None}
    sub = tr[tr["rule"].isin(rules)]
    keys = sub[["ticker", "t"]].drop_duplicates().reset_index(drop=True)
    kid = pd.MultiIndex.from_frame(keys).get_indexer(pd.MultiIndex.from_frame(sub[["ticker", "t"]]))
    sel = {r: kid[(sub["rule"] == r).to_numpy()] for r in rules}
    s0, s1 = stage["period"]
    pool = lec_null_pool(stage["pool"], (pd.Timestamp(s0), pd.Timestamp(s1 or "2262-01-01")))
    rc = lec_reality_check(keys, sel, pool, cf["RC_REPS"], seed, progress=progress, key_col="ticker")
    arr = np.array([v for v in rc["max_t"] if np.isfinite(v)])
    thr = float(np.quantile(arr, cf["RC_Q"])) if len(arr) else None
    return {"max_t": rc["max_t"], "threshold": thr, "null_mean": rc.get("null_mean")}


def mr2_pbo(stage: Dict[str, Any], rules: List[str]) -> Dict[str, Any]:
    tr = stage["tr"]
    sub = tr[tr["rule"].isin(rules)].reset_index(drop=True)
    sel = {r: np.flatnonzero((sub["rule"] == r).to_numpy()) for r in rules}
    s0, s1 = stage["period"]
    return lec_pbo(sub, sel, "x", (pd.Timestamp(s0), pd.Timestamp(s1)), MR2_CONFIG["PBO_SLICES"])


def _mr2_elig_share(prepared: Dict[str, Dict[str, Any]], elig: Dict[str, np.ndarray], start: str, end: Optional[str]) -> Tuple[Optional[float], int]:
    days = edays = nt = 0
    for sym, p in prepared.items():
        lo_i, hi_i = _mr_bounds(p["b"].index, start, end)
        if hi_i < lo_i:
            continue
        e = elig.get(sym, np.zeros(len(p["b"]), dtype=bool))[lo_i:hi_i + 1]
        days += hi_i - lo_i + 1
        edays += int(e.sum())
        nt += int(e.any())
    return (edays / days if days else None), nt


def _mr2_limit_view(stage: Dict[str, Any], r: str) -> Dict[str, Any]:
    st = stage["limit"].get(r)
    if not st:
        return {}
    return {**st, "fill_rate": (st["fills"] / st["orders"]) if st["orders"] else None}


def run_mr2(prepared: Dict[str, Dict[str, Any]], progress: Optional[Callable[[str, int, int], None]] = None) -> Dict[str, Any]:
    """찾기(기존 545종목, 2001~2019) 26개 규칙 + 섞기 통과선 + PBO(참고) → 후보만 검증(같은 종목 2020~)."""
    cf = MR2_CONFIG
    elig = mr_eligibility(prepared)
    share, ntk = _mr2_elig_share(prepared, elig, cf["EXPLORE_START"], cf["EXPLORE_END"])
    pg = (lambda s: (lambda i, n: progress(s, i, n))) if progress else (lambda s: None)
    ex = mr2_stage(prepared, elig, MR2_RULES, cf["EXPLORE_START"], cf["EXPLORE_END"], progress=pg("찾기 종목"))
    stats = {r: mr2_stats(ex["tr"][ex["tr"]["rule"] == r]) for r in MR2_RULES}
    comp = [r for r in MR2_RULES if stats[r]["n"] >= cf["MIN_TRADES"] and stats[r]["n_tickers"] >= cf["MIN_TICKERS"]]
    rc = mr2_reality(ex, comp, cf["SEED"], progress=pg(f"섞기 {cf['RC_REPS']}회"))
    thr = max(rc["threshold"], cf["Z_FLOOR"]) if rc["threshold"] is not None else cf["Z_FLOOR"]
    strict = {r: mr2_stats(ex["strict"][ex["strict"]["rule"] == r]) for r in MR2_RULES if r.endswith("-L")}
    judge = {r: mr2_judge_explore(stats[r], thr, strict.get(r)) for r in MR2_RULES}
    cands = [r for r in MR2_RULES if judge[r]["candidate"]]
    pbo = mr2_pbo(ex, comp) if len(comp) >= 2 else {"pbo": None, "n_splits": 0, "n_combos": len(comp)}
    pairs = {}
    for s, (_, lim) in MR2_SIGNALS.items():
        if lim:
            a = ex["tr"][ex["tr"]["rule"] == f"{s}-L"]
            bb = ex["tr"][ex["tr"]["rule"] == f"{s}-C"]
            pairs[s] = compare_clustered(a, bb, "x")
    res: Dict[str, Any] = {
        "config_version": cf["VERSION"], "n_tickers_used": len(prepared), "eligible_share": share, "eligible_tickers": ntk,
        "pool_n": ex["pool_n"], "pool_mean": ex["pool_mean"], "rules": {}, "competing": comp,
        "threshold_raw": rc["threshold"], "threshold": thr, "max_t": rc["max_t"], "pbo": pbo, "pairs": pairs, "candidates": cands,
    }
    for r in MR2_RULES:
        res["rules"][r] = {"stats": stats[r], "judge": judge[r], "limit": _mr2_limit_view(ex, r), "open_end": ex["open_end"].get(r, 0)}
        if r in strict:
            res["rules"][r]["strict"] = strict[r]
    res["verify"] = {}
    res["verified"] = []
    if cands:
        vs = mr2_stage(prepared, elig, cands, cf["VERIFY_START"], None, progress=pg("검증 종목"))
        vshare, vnt = _mr2_elig_share(prepared, elig, cf["VERIFY_START"], None)
        res["verify_meta"] = {"eligible_share": vshare, "eligible_tickers": vnt, "pool_n": vs["pool_n"], "pool_mean": vs["pool_mean"]}
        for r in cands:
            s = mr2_stats(vs["tr"][vs["tr"]["rule"] == r])
            sx = mr2_stats(vs["strict"][vs["strict"]["rule"] == r]) if r.endswith("-L") else None
            j = mr2_judge_simple(s, cf["VERIFY_Z"], cf["VERIFY_MIN_TRADES"], check_market=False, strict=sx)
            res["verify"][r] = {"stats": s, "judge": j, "limit": _mr2_limit_view(vs, r)}
            if sx is not None:
                res["verify"][r]["strict"] = sx
        ok = [r for r in cands if res["verify"][r]["judge"]["passed"]]
        res["verified"] = sorted(ok, key=lambda r: -(res["verify"][r]["stats"]["t"] or 0))[:cf["FINAL_MAX"]]
    return res


# ---------------------------------------------------------------------
# 최종 (새 151종목, 검증 통과 규칙만)
# ---------------------------------------------------------------------
def mr2_prepare_plain(price_map: Dict[str, pd.DataFrame]) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    """최종 단계 재료: 가격만(섹터·실적일 필요 없음). 260봉 미만·못 받은 종목은 빼고 대체하지 않는다."""
    out, dropped = {}, []
    for sym, df in price_map.items():
        d = lec_clean(df)
        if d is None or len(d) < EXPLORE_CONFIG["MIN_BARS"]:
            dropped.append(sym)
            continue
        b = lec_base(d)
        out[sym] = {"b": b, "sw": lec_swings(b)}
    return out, dropped


def run_mr2_final(prepared: Dict[str, Dict[str, Any]], rules: List[str],
                  progress: Optional[Callable[[str, int, int], None]] = None) -> Dict[str, Any]:
    cf = MR2_CONFIG
    elig = mr_eligibility(prepared)
    share, ntk = _mr2_elig_share(prepared, elig, cf["FINAL_START"], None)
    st = mr2_stage(prepared, elig, rules, cf["FINAL_START"], None,
                   progress=(lambda i, n: progress("최종 종목", i, n)) if progress else None)
    res: Dict[str, Any] = {"config_version": cf["VERSION"], "n_tickers_used": len(prepared), "eligible_share": share,
                           "eligible_tickers": ntk, "pool_n": st["pool_n"], "pool_mean": st["pool_mean"], "rules": {}, "passed": []}
    for r in rules:
        s = mr2_stats(st["tr"][st["tr"]["rule"] == r])
        sx = mr2_stats(st["strict"][st["strict"]["rule"] == r]) if r.endswith("-L") else None
        j = mr2_judge_simple(s, cf["FINAL_Z"], cf["FINAL_MIN_TRADES"], check_market=True, strict=sx)
        res["rules"][r] = {"stats": s, "judge": j, "limit": _mr2_limit_view(st, r)}
        if sx is not None:
            res["rules"][r]["strict"] = sx
        if j["passed"]:
            res["passed"].append(r)
    return res


# ---------------------------------------------------------------------
# 보고서 · 기록
# ---------------------------------------------------------------------
def _mr2_t(v: Optional[float]) -> str:
    return "—" if v is None or not np.isfinite(v) else f"{v:+.1f}"


def _mr2_pp(v: Optional[float]) -> str:
    return "—" if v is None or not np.isfinite(v) else f"{v * 100:+.2f}%p"


def _mr2_short(s: Dict[str, Any]) -> str:
    return f"{_mr2_pp(s.get('diff'))} · t {_mr2_t(s.get('t'))} ({s.get('n', 0):,}건)" if s.get("n") else "표본 없음"


def _mr2_detail(s: Dict[str, Any], lim: Dict[str, Any], close_mode: bool) -> List[str]:
    if not s.get("n"):
        return ["- 거래 없음"]
    k = s.get("kinds", {})
    tot = max(1, s["n"])
    L = [f"- 승률 {s['win'] * 100:.0f}% · 청산 " + " · ".join(f"{a} {v / tot * 100:.0f}%" for a, v in k.items())
         + f" · 평균 보유 {s['hold']:.1f}일 · 종목 {s['n_tickers']}개",
         f"- 미국 {_mr2_short(s['by_market']['US'])} / 한국 {_mr2_short(s['by_market']['KR'])}",
         f"- 기여가 가장 큰 달 {s['month_drop']['month']}을 빼면 {_mr2_pp(s['month_drop']['net'])} · t {_mr2_t(s['month_drop']['t'])}",
         "- 기간별: " + (" · ".join(f"{p} {_mr2_short(v)}" for p, v in s.get("periods", {}).items()) or "—")]
    if close_mode:
        L.append(f"- 같은 날 같은 시장 대상 종목 평균 대비(시장 전체 반등 몫을 뺀 값): {_mr2_short(s.get('vs_market', {}))}")
    if lim:
        L.append(f"- 지정가: 주문 {lim['orders']:,}일 · 체결 {lim['fills']:,}건"
                 f"({'—' if lim.get('fill_rate') is None else format(lim['fill_rate'] * 100, '.0f') + '%'}) · "
                 f"시가 체결 {lim['open_fills']:,} · 체결일 손절 {lim['fill_day_stop']:,}")
    return L


def render_mr2_report(r: Dict[str, Any], generated_at: Optional[str] = None) -> str:
    cf = MR2_CONFIG
    L = ["# H-MR2 단기 신호 찾기 — 신호 즉시 진입 · +5% 익절 · −5% 손절 · 최대 3거래일", "",
         "**찾기 단계 결과 — 이것만으로는 증거가 아니다.** 이미 다른 시험에 쓴 545종목의 2001~2019년이다. 후보는 같은 종목의 "
         "2020년 이후(이번에 처음 엶)로 검증하고, 검증을 통과해야 새 151종목 최종(`--mr2-final`, z ≥ 3)으로 간다(MR2_PREREG.md).", "",
         f"- 생성: {generated_at or '—'} · 설정 {r['config_version']} (사전 등록 {cf['REGISTERED']}) · 종목 {r['n_tickers_used']}개",
         "- 종가 진입 = 신호일 종가 / 지정가 = 전날까지 자료로 정한 가격, 저가가 닿으면 min(시가, 가격)에 체결(체결일 −5%는 손절, +5%는 안 셈)",
         "- 청산: 다음 날부터 3거래일, +5% 익절 · −5% 손절(같은 날 둘 다면 손절, 시가가 넘으면 시가) · 3거래일째 종가 · 비용 0.30% · 규칙마다 종목당 1포지션",
         "- 지정가 규칙은 엄격 체결(시가 ≤ P 또는 저가 ≤ P × 0.999)로 다시 계산해도 차이 > 0 · t ≥ 2여야 후보(찾기·검증·최종 모두)",
         f"- 대상: 거래대금 상위 절반 중 변동성 상위 절반(mr1-2) · 찾기 기간 종목·날의 "
         f"{'—' if r.get('eligible_share') is None else format(r['eligible_share'] * 100, '.0f') + '%'}, 한 번이라도 대상인 종목 {r.get('eligible_tickers', 0)}개",
         f"- 비교 기준: 같은 종목이 대상이었던 모든 날 종가 진입(같은 청산) — {r.get('pool_n', 0):,}일, 평균 {_lec_pct(r.get('pool_mean'))}(비용 뒤)",
         f"- **통과선 t = {r['threshold']:.2f}** (거래 ≥ 300 · 종목 ≥ 50인 규칙 {len(r['competing'])}개 경쟁, 같은 종목 무작위 대상 날로 바꿔 "
         f"{cf['RC_REPS']}번 다시 잰 '최고 t'의 {int(cf['RC_Q'] * 100)}% 분위 {'—' if r['threshold_raw'] is None else format(r['threshold_raw'], '.2f')}, 최소 {cf['Z_FLOOR']:.0f})",
         f"- 과적합 확률 PBO(참고): {'—' if r['pbo'].get('pbo') is None else format(r['pbo']['pbo'] * 100, '.0f') + '%'} (16조각)", "",
         "## 판정 (찾기)", "",
         "| 규칙 | 거래(종목) | 거래당 순수익 | 같은 종목 아무 날 | 차이 · t | 판정 |", "|---|---|---|---|---|---|"]
    for k in MR2_RULES:
        x = r["rules"][k]
        s, j = x["stats"], x["judge"]
        if s["n"]:
            v = "**찾기 후보** → 검증" if j["candidate"] else "후보 아님 (" + ", ".join(j["reasons"]) + ")"
            L.append(f"| {mr2_label(k)} | {s['n']:,} ({s['n_tickers']}) | {_lec_pct(s['net'])} | {_lec_pct(s['base'])} | "
                     f"{_mr2_pp(s['diff'])} · {_mr2_t(s['t'])} | {v} |")
        else:
            L.append(f"| {mr2_label(k)} | 0 | — | — | — | 후보 아님 (거래 없음) |")
    L += ["", "## 검증 (후보만, 같은 종목 2020년~)", ""]
    if not r["candidates"]:
        L.append("- 찾기 후보가 없어 2020년 이후 자료는 열지 않았다.")
    else:
        vm = r.get("verify_meta", {})
        L.append(f"- 대상 종목·날 {'—' if vm.get('eligible_share') is None else format(vm['eligible_share'] * 100, '.0f') + '%'} · "
                 f"비교 기준 {vm.get('pool_n', 0):,}일 평균 {_lec_pct(vm.get('pool_mean'))} · 기준: 차이 > 0, t ≥ {cf['VERIFY_Z']:.0f}, 거래 ≥ {cf['VERIFY_MIN_TRADES']}")
        L += ["", "| 규칙 | 거래당 순수익 | 같은 종목 아무 날 | 차이 · t | 판정 |", "|---|---|---|---|---|"]
        for k in r["candidates"]:
            v = r["verify"][k]
            s, j = v["stats"], v["judge"]
            L.append(f"| {mr2_label(k)} | {_lec_pct(s.get('net'))} ({s['n']:,}건) | {_lec_pct(s.get('base'))} | {_mr2_pp(s['diff'])} · {_mr2_t(s['t'])} | "
                     + ("**검증 통과** → 최종 대기" if j["passed"] else "미통과 (" + ", ".join(j["reasons"]) + ")") + " |")
        if r["verified"]:
            L.append(f"\n- 최종으로 갈 규칙(검증 t 큰 순, 최대 {cf['FINAL_MAX']}개): " + ", ".join(mr2_label(k) for k in r["verified"]))
    L += ["", "## 참고 — 진입 방식 비교(같은 신호, 판정 아님)", "", "| 신호 | 종가 진입 차이 · t | 지정가 차이 · t | 지정가 − 종가 |", "|---|---|---|---|"]
    for s, cm in r["pairs"].items():
        L.append(f"| {s} {MR2_SIGNALS[s][0]} | {_mr2_short(r['rules'][s + '-C']['stats'])} | {_mr2_short(r['rules'][s + '-L']['stats'])} | "
                 f"{_mr2_pp(cm.get('diff'))} · z {_mr2_t(cm.get('z'))} |")
    L += ["", "## 참고 — 규칙별 자세히(찾기)", ""]
    for k in MR2_RULES:
        x = r["rules"][k]
        L += [f"**{mr2_label(k)}**", ""] + _mr2_detail(x["stats"], x["limit"], k.endswith("-C"))
        if x.get("strict"):
            L.append(f"- 엄격 체결(시가 ≤ P 또는 저가 ≤ P × 0.999, 판정 조건): {_mr2_short(x['strict'])} · 거래당 순수익 {_lec_pct(x['strict'].get('net'))}")
        if r["verify"].get(k):
            v = r["verify"][k]
            L += [f"- 검증 자세히:"] + ["  " + t for t in _mr2_detail(v["stats"], v["limit"], k.endswith("-C"))]
            if v.get("strict"):
                L.append(f"  - 검증 엄격 체결: {_mr2_short(v['strict'])}")
        L.append("")
    L += ["## 읽는 법", "",
          "- '차이'는 거래당 순수익(비용 뒤)에서 같은 종목이 대상이던 아무 날에 종가로 사서 같은 방식으로 판 평균을 뺀 값이다. 승률은 참고만 — 익절·손절 폭이 같아서 아무 날에 사도 승률이 거의 일정하다.",
          "- 통과선은 26개 규칙 중 가장 좋아 보이는 것이 우연히 나올 만큼을 뺀 값이다. t가 2를 넘어도 통과선 아래면 후보가 아니다.",
          "- 후보가 아니면 여기서 끝낸다. 결과를 보고 신호·숫자(ATR 배수·%·RSI 기준)·청산(±5%·3일)을 바꿔 다시 보지 않는다.",
          "- 새 151종목은 이 단계에서 열지 않았다.", ""]
    return "\n".join(L)


def render_mr2_final_report(r: Dict[str, Any], dropped: List[str], generated_at: Optional[str] = None) -> str:
    cf = MR2_CONFIG
    L = ["# H-MR2 최종 — 새 151종목(2001~)", "",
         f"- 생성: {generated_at or '—'} · 설정 {r['config_version']} · 종목 {r['n_tickers_used']}개(빠진 종목 {len(dropped)}개: {', '.join(dropped) or '없음'})",
         f"- 대상 종목·날 {'—' if r.get('eligible_share') is None else format(r['eligible_share'] * 100, '.0f') + '%'} (새 151종목 안에서 순위) · "
         f"비교 기준 {r.get('pool_n', 0):,}일 평균 {_lec_pct(r.get('pool_mean'))}",
         f"- 기준: 차이 > 0, t ≥ {cf['FINAL_Z']:.0f}, 거래 ≥ {cf['FINAL_MIN_TRADES']}, 미국·한국 같은 방향", "",
         "| 규칙 | 거래당 순수익 | 같은 종목 아무 날 | 차이 · t | 판정 |", "|---|---|---|---|---|"]
    for k, x in r["rules"].items():
        s, j = x["stats"], x["judge"]
        L.append(f"| {mr2_label(k)} | {_lec_pct(s.get('net'))} ({s['n']:,}건) | {_lec_pct(s.get('base'))} | {_mr2_pp(s['diff'])} · {_mr2_t(s['t'])} | "
                 + ("**통과**" if j["passed"] else "미통과 (" + ", ".join(j["reasons"]) + ")") + " |")
    L += [""]
    for k, x in r["rules"].items():
        L += [f"**{mr2_label(k)}**", ""] + _mr2_detail(x["stats"], x["limit"], k.endswith("-C"))
        if x.get("strict"):
            L.append(f"- 엄격 체결: {_mr2_short(x['strict'])}")
        L.append("")
    L += ["- 이 단계는 한 번만 한다. 통과 = 앱 '통과'. 그래도 앞으로의 실제 결과(장부)로 다시 확인한다.", ""]
    return "\n".join(L)


def mr2_log_lines(r: Dict[str, Any], generated_at: str) -> List[str]:
    rows = []
    for k in MR2_RULES:
        s = r["rules"][k]["stats"]
        verdict = "후보" if r["rules"][k]["judge"]["candidate"] else "후보 아님"
        if k in r.get("verify", {}):
            verdict += " → 검증 " + ("통과" if r["verify"][k]["judge"]["passed"] else "미통과")
        rows.append(f"| {generated_at} | `mr2-{k}` | {mr2_label(k)} — +5%/−5%/3일, 거래대금·변동성 상위 대상, 차이 = 순수익 − 같은 종목 아무 날 "
                    f"(찾기 545종목 2001~2019, 통과선 t {r['threshold']:.2f}) | {s.get('n', 0):,} | {_mr2_pp(s.get('diff'))} | {_mr2_t(s.get('t'))} | {verdict} |")
    return rows


def mr2_final_log_lines(r: Dict[str, Any], generated_at: str) -> List[str]:
    rows = []
    for k, x in r["rules"].items():
        s = x["stats"]
        rows.append(f"| {generated_at} | `mr2-final-{k}` | {mr2_label(k)} — 최종(새 151종목 2001~, z ≥ 3) | {s.get('n', 0):,} | "
                    f"{_mr2_pp(s.get('diff'))} | {_mr2_t(s.get('t'))} | {'통과' if x['judge']['passed'] else '미통과'} |")
    return rows
