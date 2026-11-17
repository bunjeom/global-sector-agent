"""지지선·손절선 검증 (2026-11-17) — run_backtest.py --levels → data/backtest/LEVELS.md.

단기 규칙이 사전 등록 시험에서 미통과(SHORT_TERM.md)라 화면은 진입 신호 대신 지지선·손절선만 보여 준다(engine/strategy.py
support_stop_levels). 이 파일은 '그 선이 의미 있는 선인가'를 과거 가격으로 본다. 수익 규칙 시험이 아니라 선의 성질 시험이다.
선의 정의는 단기 규칙 그대로다(바꾸지 않음): 지지선 = 그날까지 확정된 가장 최근 스윙 저점(틀 A 지그재그 3×ATR),
손절선 = 지지선 − 0.5×ATR(20).

질문 1 — 지지선: 같은 거리의 임의의 선보다 덜 깨지는가
    사건  저가가 지지선 + 0.5×ATR 이하로 내려왔고 종가는 손절선 이상인 날 t (사건 뒤 10거래일은 같은 종목의 새 사건을 세지 않음)
    결과  t+1 ~ t+10 종가 중 하나라도 그날의 손절선 아래 = '깨짐'
    가짜 선(판정 기준)  같은 종목의 다른 날 j 가운데 '최근 5거래일 하락 폭(ATR 단위)이 사건일과 비슷한(±0.5)' 날을 뽑아,
        종가에서 사건일과 같은 거리(ATR 단위)만큼 아래에 선을 긋고 같은 방식으로 깨짐을 센다.
        → 최근에 똑같이 떨어졌고 선까지 거리도 같은데, 그 선이 '스윙 저점'이냐 '아무 가격'이냐만 다르다.
    참고  하락 폭을 맞추지 않은 무작위 날의 가짜 선.
질문 2 — 손절선: 깨진 뒤 정말 더 나빠지는가
    사건  종가가 손절선 아래로 처음 내려온 날 t(전날 종가는 손절선 이상. 사건 뒤 10거래일은 새 사건을 세지 않음)
    결과  다음 날 시가에 팔지 않고 들고 있었다면: t+1 시가 → t+10 종가 수익률(판정), t+20 종가 수익률·−10% 이하 비율(참고)
    비교(판정 기준)  같은 종목 무작위 날 j의 j+1 시가 → j+10 종가 수익률. 이탈 뒤가 더 낮아야 '손절이 추가 손실을 피했다'.
    참고  최근 5거래일 하락 폭이 비슷한 날(이탈이 아닌 그냥 하락과 비교).

가격 기반 시험의 여덟·아홉 번째 질문이다. 판정 기준(LEVEL_CRITERIA)은 결과를 보기 전에 z 3으로 고정했고,
판정 묶음(HOLDOUT)은 어떤 백테스트에도 쓰지 않은 종목이다. 표준오차는 같은 달 사건을 한 덩어리로 묶는다.
"""
import math
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .frame_backtest import FRAME_A_CONFIG, compare_clustered
from .rule_backtest import _bt_naive
from .strategy import LEVEL_CONFIG, short_indicators

LEVEL_BT_CONFIG: Dict[str, Any] = {
    "START": "2001-01-01",
    "DATA_START": "1999-01-01",
    "SPLIT_DATE": FRAME_A_CONFIG["SPLIT_DATE"],
    "HORIZON": 10,                 # 깨짐·수익률을 보는 거래일 수 (단기 규칙의 10거래일과 같음)
    "HORIZON_LONG": 20,            # 참고
    "TAIL": -0.10,                 # 참고: 20거래일 −10% 이하
    "DROP_DAYS": 5,                # 가짜 선을 그을 날 고르기: 최근 5거래일 변화(ATR 단위)
    "DROP_TOL": 0.5,               # 사건일과 ±0.5 ATR 이내
    "BASELINE_SAMPLES": 20,
    "SEED": 31,
    "MIN_BARS": 260,
    "DEV_UNIVERSE": list(FRAME_A_CONFIG["DEV_UNIVERSE"]),
    # 판정 묶음(HOLDOUT, 2026-11-17 결과 보기 전 고정): 지금까지 어떤 백테스트(게이트·T 셋업·스캐너·재설계·틀 A·틀 B·단기 규칙)에도
    # 쓰지 않은 종목. 미국 11개 섹터 85 · 한국 코스피 25.
    "HOLDOUT_UNIVERSE": [
        "AKAM", "CTSH", "FFIV", "NTAP", "STX", "WDC", "VRSN", "FIS", "SWKS", "TER",          # 기술
        "TTWO", "LYV",                                                                          # 통신·미디어
        "DRI", "RCL", "MGM", "TPR", "DHI", "KMX", "ULTA", "CMG", "TSCO", "DLTR",              # 경기소비재
        "MNST", "HRL", "SJM", "MKC", "CHD", "TAP",                                             # 필수소비재
        "VLO", "MPC", "KMI", "OKE", "BKR", "APA", "EQT",                                       # 에너지
        "STT", "FITB", "RF", "MTB", "HIG", "AMP", "RJF", "MCO", "AJG", "CINF",                 # 금융
        "A", "ALGN", "DVA", "HOLX", "ILMN", "LH", "RMD", "UHS", "WAT", "CNC",                  # 헬스케어
        "CMI", "JCI", "AME", "FAST", "RSG", "EFX", "TXT", "URI", "DAL", "JBHT", "EXPD", "MAS",  # 산업재
        "CF", "ALB", "EMN", "MLM", "PKG", "AVY",                                               # 소재
        "EXC", "WEC", "DTE", "ETR", "PPL", "CMS",                                              # 유틸리티
        "WELL", "EQR", "VTR", "HST", "KIM", "WY",                                              # 리츠
        "000080.KS", "000120.KS", "000150.KS", "000210.KS", "000240.KS", "000880.KS", "001450.KS", "002380.KS",
        "003490.KS", "004370.KS", "005940.KS", "006260.KS", "006360.KS", "007310.KS", "009830.KS", "010060.KS",
        "011790.KS", "012450.KS", "012750.KS", "018880.KS", "023530.KS", "028050.KS", "030000.KS", "034020.KS",
        "035250.KS",
    ],
}
LEVEL_CRITERIA: Dict[str, Any] = {
    # 지지선 통과: HOLDOUT 전체 기간 사건 ≥ MIN_EVENTS, (실제 깨짐 비율 − 가짜 선 깨짐 비율) < 0 이고 z ≤ −MIN_Z,
    #             그리고 개발 묶음 앞·뒤 기간 모두 차이 < 0
    # 손절선 통과: HOLDOUT 전체 기간 사건 ≥ MIN_EVENTS, (이탈 뒤 10거래일 수익률 − 무작위 날 수익률) < 0 이고 z ≤ −MIN_Z,
    #             그리고 개발 묶음 앞·뒤 기간 모두 차이 < 0
    "MIN_Z": 3.0,
    "MIN_EVENTS": 100,
}
# 결과에 따라 화면에 붙일 문구(결과 보기 전 고정) — engine/strategy.py LEVEL_VERIFICATION에 옮겨 적는다
LEVEL_VERDICT_TEXT: Dict[str, Dict[bool, str]] = {
    "support": {True: "검증됨 — 과거에 같은 거리의 임의의 선보다 덜 깨졌음(LEVELS.md). 앞으로도 그렇다는 보장은 아님",
                False: "검증 안 됨 — 과거에 같은 거리의 임의의 선보다 덜 깨지지 않았음(LEVELS.md). 위험 계산용 기준선으로만"},
    "stop": {True: "검증됨 — 과거에 이탈 뒤 10거래일 수익이 평소보다 나빴음(LEVELS.md). 이탈 시 정리가 추가 손실을 줄였음",
             False: "검증 안 됨 — 과거에 이탈 뒤 수익이 평소보다 나쁘지 않았음(LEVELS.md). 추가 하락 예고가 아니라 손실 한도로만"},
}


def _lv_market(sym: str) -> str:
    return "KR" if str(sym).upper().endswith((".KS", ".KQ")) else "US"


def level_arrays(si: pd.DataFrame) -> Dict[str, np.ndarray]:
    """사건·가짜 선 계산에 쓰는 배열. fmin[t] = t+1 ~ t+H 종가 최솟값(자료가 모자라면 NaN)."""
    c = LEVEL_BT_CONFIG
    H, HL, D = c["HORIZON"], c["HORIZON_LONG"], c["DROP_DAYS"]
    a = {k: si[k].to_numpy(dtype=float) for k in ("Open", "Low", "Close", "ATR20", "support", "stop")}
    cl, op, atr = a["Close"], a["Open"], a["ATR20"]
    n = len(cl)
    s = pd.Series(cl)
    a["fmin"] = s[::-1].rolling(H, min_periods=H).min()[::-1].shift(-1).to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        drop = np.full(n, np.nan)
        drop[D:] = (cl[D:] - cl[:-D]) / atr[D:]
        a["drop"] = drop
        nxt_open = np.append(op[1:], np.nan)
        for h, key in ((H, "r_h"), (HL, "r_hl")):
            fut = np.full(n, np.nan)
            if n > h:
                fut[:n - h] = cl[h:]
            a[key] = fut / nxt_open - 1.0
        a["fwd_close"] = np.full(n, np.nan)
        if n > H:
            a["fwd_close"][:n - H] = cl[H:] / cl[:n - H] - 1.0
    return a


def support_events(si: pd.DataFrame, a: Optional[Dict[str, np.ndarray]] = None, start: Optional[str] = None) -> pd.DataFrame:
    """질문 1 사건: 저가 ≤ 지지선 + 0.5×ATR 이고 종가 ≥ 손절선. 결과 broke = 이후 10거래일 종가가 손절선 아래로."""
    c = LEVEL_BT_CONFIG
    a = a or level_arrays(si)
    lo, cl, atr, sup, stop, fmin = a["Low"], a["Close"], a["ATR20"], a["support"], a["stop"], a["fmin"]
    s0 = pd.Timestamp(start or c["START"])
    with np.errstate(invalid="ignore"):
        ok = (np.isfinite(stop) & np.isfinite(atr) & (atr > 0) & np.isfinite(fmin) & np.isfinite(a["drop"])
              & (lo <= sup + LEVEL_CONFIG["NEAR_ATR"] * atr) & (cl >= stop) & (si.index >= s0))
    out, free = [], -1
    for t in np.flatnonzero(ok):
        if t < free:
            continue
        out.append({"date": si.index[t], "t": int(t), "d_atr": float((cl[t] - stop[t]) / atr[t]), "drop": float(a["drop"][t]),
                    "broke": float(fmin[t] < stop[t]), "fwd": float(a["fwd_close"][t])})
        free = t + c["HORIZON"] + 1
    return pd.DataFrame(out)


def support_placebo(si: pd.DataFrame, events: pd.DataFrame, a: Optional[Dict[str, np.ndarray]] = None, matched: bool = True,
                    start: Optional[str] = None, samples: Optional[int] = None, seed: Optional[int] = None) -> pd.DataFrame:
    """가짜 선: 다른 날 j의 종가에서 사건과 같은 ATR 거리만큼 아래. matched=True면 최근 5거래일 변화(ATR 단위)가 ±DROP_TOL 이내인 날만."""
    c = LEVEL_BT_CONFIG
    a = a or level_arrays(si)
    if events is None or not len(events):
        return pd.DataFrame()
    cl, atr, fmin, drop = a["Close"], a["ATR20"], a["fmin"], a["drop"]
    s0 = pd.Timestamp(start or c["START"])
    with np.errstate(invalid="ignore"):
        valid = np.isfinite(atr) & (atr > 0) & np.isfinite(fmin) & np.isfinite(drop) & (si.index >= s0)
    cand_all = np.flatnonzero(valid)
    if not len(cand_all):
        return pd.DataFrame()
    order = np.argsort(drop[cand_all])
    cand_sorted, drop_sorted = cand_all[order], drop[cand_all][order]
    rng = np.random.default_rng((c["SEED"] if seed is None else seed) + (0 if matched else 1))
    k = int(samples or c["BASELINE_SAMPLES"])
    out = []
    for t, d_atr, dr in zip(events["t"].astype(int).to_numpy(), events["d_atr"].to_numpy(), events["drop"].to_numpy()):
        if matched:
            lo_i = np.searchsorted(drop_sorted, dr - c["DROP_TOL"], "left")
            hi_i = np.searchsorted(drop_sorted, dr + c["DROP_TOL"], "right")
            pool = cand_sorted[lo_i:hi_i]
        else:
            pool = cand_all
        pool = pool[np.abs(pool - t) > c["HORIZON"]]      # 사건 자신과 겹치는 날은 빼고
        if not len(pool):
            continue
        js = rng.choice(pool, size=k, replace=True)
        broke = fmin[js] < (cl[js] - d_atr * atr[js])
        for j, b in zip(js, broke):
            out.append({"date": si.index[j], "broke": float(b), "fwd": float(a["fwd_close"][j])})
    return pd.DataFrame(out)


def stop_events(si: pd.DataFrame, a: Optional[Dict[str, np.ndarray]] = None, start: Optional[str] = None) -> pd.DataFrame:
    """질문 2 사건: 종가가 손절선 아래로 처음 내려온 날(전날 종가는 손절선 이상). 결과 = 다음 날 시가 → 10·20거래일 뒤 종가."""
    c = LEVEL_BT_CONFIG
    a = a or level_arrays(si)
    cl, stop = a["Close"], a["stop"]
    s0 = pd.Timestamp(start or c["START"])
    n = len(cl)
    with np.errstate(invalid="ignore"):
        below = np.isfinite(stop) & (cl < stop)
        prev_ok = np.zeros(n, dtype=bool)
        prev_ok[1:] = np.isfinite(stop[:-1]) & (cl[:-1] >= stop[:-1])
        ok = below & prev_ok & np.isfinite(a["r_h"]) & np.isfinite(a["drop"]) & (si.index >= s0)
    out, free = [], -1
    for t in np.flatnonzero(ok):
        if t < free:
            continue
        r20 = a["r_hl"][t]
        out.append({"date": si.index[t], "t": int(t), "drop": float(a["drop"][t]), "r10": float(a["r_h"][t]),
                    "r20": float(r20) if np.isfinite(r20) else np.nan,
                    "tail": float(r20 <= c["TAIL"]) if np.isfinite(r20) else np.nan})
        free = t + c["HORIZON"] + 1
    return pd.DataFrame(out)


def stop_baseline(si: pd.DataFrame, events: pd.DataFrame, a: Optional[Dict[str, np.ndarray]] = None, matched: bool = False,
                  start: Optional[str] = None, samples: Optional[int] = None, seed: Optional[int] = None) -> pd.DataFrame:
    """비교 표본: 같은 종목 무작위 날(matched=False, 판정 기준) 또는 최근 5거래일 변화가 비슷한 날(matched=True, 참고)."""
    c = LEVEL_BT_CONFIG
    a = a or level_arrays(si)
    if events is None or not len(events):
        return pd.DataFrame()
    drop = a["drop"]
    s0 = pd.Timestamp(start or c["START"])
    valid = np.isfinite(a["r_h"]) & np.isfinite(drop) & (si.index >= s0)
    cand_all = np.flatnonzero(valid)
    if not len(cand_all):
        return pd.DataFrame()
    order = np.argsort(drop[cand_all])
    cand_sorted, drop_sorted = cand_all[order], drop[cand_all][order]
    rng = np.random.default_rng((c["SEED"] if seed is None else seed) + (3 if matched else 2))
    k = int(samples or c["BASELINE_SAMPLES"])
    out = []
    for t, dr in zip(events["t"].astype(int).to_numpy(), events["drop"].to_numpy()):
        if matched:
            pool = cand_sorted[np.searchsorted(drop_sorted, dr - c["DROP_TOL"], "left"):np.searchsorted(drop_sorted, dr + c["DROP_TOL"], "right")]
        else:
            pool = cand_all
        pool = pool[np.abs(pool - t) > c["HORIZON"]]
        if not len(pool):
            continue
        for j in rng.choice(pool, size=k, replace=True):
            r20 = a["r_hl"][j]
            out.append({"date": si.index[j], "r10": float(a["r_h"][j]), "r20": float(r20) if np.isfinite(r20) else np.nan,
                        "tail": float(r20 <= c["TAIL"]) if np.isfinite(r20) else np.nan})
    return pd.DataFrame(out)


def run_level_universe(price_map: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    c = LEVEL_BT_CONFIG
    parts: Dict[str, List[pd.DataFrame]] = {k: [] for k in ("sup", "sup_m", "sup_r", "stp", "stp_r", "stp_m")}
    used = []
    for sym, df in (price_map or {}).items():
        d = _bt_naive(df)
        if d is not None and len(d):   # 거래 정지일 등 0으로 온 줄은 뺀다
            cols = [k for k in ("Open", "High", "Low", "Close") if k in d]
            d = d[(d[cols].astype(float) > 0).all(axis=1)]
        if d is None or len(d) < c["MIN_BARS"]:
            continue
        si = short_indicators(d)
        a = level_arrays(si)
        used.append(sym)
        se, te = support_events(si, a), stop_events(si, a)
        got = {"sup": se, "sup_m": support_placebo(si, se, a, matched=True), "sup_r": support_placebo(si, se, a, matched=False),
               "stp": te, "stp_r": stop_baseline(si, te, a, matched=False), "stp_m": stop_baseline(si, te, a, matched=True)}
        mk = _lv_market(sym)
        for k, x in got.items():
            if len(x):
                parts[k].append(x.assign(ticker=sym, market=mk))
    cat = lambda lst, cols: pd.concat(lst, ignore_index=True) if lst else pd.DataFrame(columns=cols)
    return {"sup": cat(parts["sup"], ["date", "broke", "fwd", "market", "ticker"]),
            "sup_m": cat(parts["sup_m"], ["date", "broke", "fwd", "market", "ticker"]),
            "sup_r": cat(parts["sup_r"], ["date", "broke", "fwd", "market", "ticker"]),
            "stp": cat(parts["stp"], ["date", "r10", "r20", "tail", "market", "ticker"]),
            "stp_r": cat(parts["stp_r"], ["date", "r10", "r20", "tail", "market", "ticker"]),
            "stp_m": cat(parts["stp_m"], ["date", "r10", "r20", "tail", "market", "ticker"]), "tickers": used}


def _lv_block(u: Dict[str, Any], f) -> Dict[str, Any]:
    sup, sm, sr, stp, tr, tm = (f(u[k]) for k in ("sup", "sup_m", "sup_r", "stp", "stp_r", "stp_m"))
    return {"support": {"vs_matched": compare_clustered(sup, sm, "broke"), "vs_random": compare_clustered(sup, sr, "broke"),
                        "fwd_vs_matched": compare_clustered(sup, sm, "fwd")},
            "stop": {"vs_random": compare_clustered(stp, tr, "r10"), "vs_matched": compare_clustered(stp, tm, "r10"),
                     "r20_vs_random": compare_clustered(stp, tr, "r20"), "tail_vs_random": compare_clustered(stp, tr, "tail")}}


def evaluate_levels(dev_map: Dict[str, pd.DataFrame], holdout_map: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """개발 묶음(앞·뒤 기간)과 HOLDOUT에서 지지선·손절선을 평가하고 사전 기준으로 판정."""
    c, crit = LEVEL_BT_CONFIG, LEVEL_CRITERIA
    split_t = pd.Timestamp(c["SPLIT_DATE"])
    res: Dict[str, Any] = {"version": LEVEL_CONFIG["VERSION"], "criteria": crit,
                           "config": {k: c[k] for k in ("START", "SPLIT_DATE", "HORIZON", "HORIZON_LONG", "TAIL", "DROP_DAYS",
                                                        "DROP_TOL", "BASELINE_SAMPLES")},
                           "universes": {}, "verdict": {}}
    raw = {"개발(대표 종목)": run_level_universe(dev_map), "HOLDOUT(새 종목)": run_level_universe(holdout_map)}
    per = {"전체 기간": lambda x: x, "앞 기간": lambda x: x[x["date"] < split_t], "뒤 기간": lambda x: x[x["date"] >= split_t]}
    for uname, u in raw.items():
        blocks = {p: _lv_block(u, f) for p, f in per.items()}
        by_mk = {mk: _lv_block(u, lambda x, mk=mk: x[x["market"] == mk]) for mk in ("US", "KR")
                 if len(u["sup"]) and (u["sup"]["market"] == mk).any()}
        res["universes"][uname] = {"n_tickers": len(u["tickers"]), "blocks": blocks, "by_market": by_mk}

    def judge(kind: str, key: str) -> Dict[str, Any]:
        ho = res["universes"]["HOLDOUT(새 종목)"]["blocks"]["전체 기간"][kind][key]
        dv = res["universes"]["개발(대표 종목)"]["blocks"]
        n = ho["actual"].get("n", 0)
        ok_ho = bool(n >= crit["MIN_EVENTS"] and ho["diff"] is not None and ho["diff"] < 0 and ho["z"] is not None
                     and ho["z"] <= -crit["MIN_Z"])
        ok_dev = all(dv[p][kind][key]["diff"] is not None and dv[p][kind][key]["diff"] < 0 for p in ("앞 기간", "뒤 기간"))
        return {"passed": ok_ho and ok_dev, "holdout_pass": ok_ho, "dev_both_negative": ok_dev, "enough_events": n >= crit["MIN_EVENTS"],
                "z2_reference": bool(n >= crit["MIN_EVENTS"] and ho["z"] is not None and ho["z"] <= -2.0), "n": n,
                "diff": ho["diff"], "z": ho["z"]}

    vs, vt = judge("support", "vs_matched"), judge("stop", "vs_random")
    res["verdict"] = {
        "support": {**vs, "text": LEVEL_VERDICT_TEXT["support"][vs["passed"]],
                    "criteria": (f"HOLDOUT 전체 기간 사건 {crit['MIN_EVENTS']}건 이상, (지지선 깨짐 비율 − 하락 폭 맞춘 가짜 선 깨짐 비율) < 0 이고 "
                                 f"z ≤ −{crit['MIN_Z']:.0f}, 그리고 개발 묶음 앞·뒤 기간 모두 차이 < 0")},
        "stop": {**vt, "text": LEVEL_VERDICT_TEXT["stop"][vt["passed"]],
                 "criteria": (f"HOLDOUT 전체 기간 사건 {crit['MIN_EVENTS']}건 이상, (이탈 뒤 {c['HORIZON']}거래일 수익률 − 무작위 날 수익률) < 0 이고 "
                              f"z ≤ −{crit['MIN_Z']:.0f}, 그리고 개발 묶음 앞·뒤 기간 모두 차이 < 0")},
    }
    return {"summary": res}


# ---------------------------------------------------------------------
# 보고서
# ---------------------------------------------------------------------
def _lv_rate(s: Dict[str, Any], pct_d: int = 1) -> str:
    if not s.get("n"):
        return "표본 없음"
    se = f" ±{s['se'] * 100:.{pct_d}f}" if s.get("se") is not None else ""
    return f"{s['mean'] * 100:.{pct_d}f}%{se} ({s['n']:,}건)"


def _lv_ret(s: Dict[str, Any]) -> str:
    if not s.get("n"):
        return "표본 없음"
    se = f" ±{s['se'] * 100:.2f}" if s.get("se") is not None else ""
    return f"{s['mean'] * 100:+.2f}%{se} ({s['n']:,}건)"


def _lv_cmp(x: Dict[str, Any], fmt, unit: str = "%p") -> str:
    d = "—" if x.get("diff") is None else f"{x['diff'] * 100:+.2f}{unit}"
    z = "—" if x.get("z") is None else f"{x['z']:+.1f}"
    return f"{fmt(x['actual'])} | {fmt(x['baseline'])} | {d} | {z}"


def render_levels_report(r: Dict[str, Any], generated_at: Optional[str] = None) -> str:
    v, cfg = r["verdict"], r["config"]
    H = cfg["HORIZON"]
    L = ["# 지지선·손절선 검증", "",
         f"- 생성: {generated_at or '—'} · {r['version']} · 사건 {cfg['START'][:4]}년~",
         "- 선의 정의(단기 규칙 그대로, 바꾸지 않음): 지지선 = 그날까지 확정된 가장 최근 스윙 저점(지그재그 3×ATR), 손절선 = 지지선 − 0.5×ATR(20).",
         "- 수익 규칙 시험이 아니라 '선이 의미 있는 선인가'를 보는 시험이다. 개발 묶음 = 앱 대표 종목, HOLDOUT = 어떤 백테스트에도 쓰지 않은 "
         "미국 85·한국 25 종목(2026-11-17 결과 보기 전 고정).",
         f"- 질문 1(지지선): 저가가 지지선 + 0.5×ATR 이하로 내려오고 종가는 손절선 이상인 날 뒤 {H}거래일 안에 종가가 손절선 아래로 깨지는 비율. "
         f"가짜 선 = 같은 종목에서 최근 {cfg['DROP_DAYS']}거래일 하락 폭이 비슷한(±{cfg['DROP_TOL']} ATR) 다른 날의 종가에서 같은 ATR 거리만큼 아래 그은 선.",
         f"- 질문 2(손절선): 종가가 손절선 아래로 처음 내려온 날, 다음 날 시가에 팔지 않고 들고 있었을 때 {H}거래일 뒤 수익률. 비교 = 같은 종목 무작위 날.",
         "- 가격 기반 시험의 여덟·아홉 번째 질문이라 판정 z 기준 3(z ≤ −3)을 미리 고정했다. 표준오차는 같은 달 사건을 한 덩어리로 묶어 계산.", "",
         f"## 판정 1 — 지지선: {'통과' if v['support']['passed'] else '미통과'}", "",
         f"- 기준: {v['support']['criteria']}.",
         f"- HOLDOUT {'통과' if v['support']['holdout_pass'] else '미통과'}(참고 z ≤ −2 {'충족' if v['support']['z2_reference'] else '미충족'}) · "
         f"개발 묶음 앞·뒤 기간 {'모두 < 0' if v['support']['dev_both_negative'] else '하나 이상 ≥ 0'} · 사건 수 {'충분' if v['support']['enough_events'] else '부족'}",
         f"- 화면 문구: {v['support']['text']}", "",
         f"## 판정 2 — 손절선: {'통과' if v['stop']['passed'] else '미통과'}", "",
         f"- 기준: {v['stop']['criteria']}.",
         f"- HOLDOUT {'통과' if v['stop']['holdout_pass'] else '미통과'}(참고 z ≤ −2 {'충족' if v['stop']['z2_reference'] else '미충족'}) · "
         f"개발 묶음 앞·뒤 기간 {'모두 < 0' if v['stop']['dev_both_negative'] else '하나 이상 ≥ 0'} · 사건 수 {'충분' if v['stop']['enough_events'] else '부족'}",
         f"- 화면 문구: {v['stop']['text']}", ""]
    for uname, u in r["universes"].items():
        L += [f"## {uname} — 종목 {u['n_tickers']}개", "",
              f"**지지선: {H}거래일 안에 깨진 비율** (낮을수록 잘 버팀)", "",
              "| 기간 | 지지선 | 가짜 선(하락 폭 맞춤, 판정) | 차이 | z | 가짜 선(무작위 날, 참고) 차이 · z |", "|---|---|---|---|---|---|"]
        rows = list(u["blocks"].items()) + [(f"전체 기간 · {'미국' if mk == 'US' else '한국'}", b) for mk, b in u["by_market"].items()]
        for per, b in rows:
            s = b["support"]
            rr = s["vs_random"]
            ref = "—" if rr.get("diff") is None else f"{rr['diff'] * 100:+.2f}%p · {'—' if rr.get('z') is None else format(rr['z'], '+.1f')}"
            L.append(f"| {per} | {_lv_cmp(s['vs_matched'], _lv_rate)} | {ref} |")
        full = u["blocks"]["전체 기간"]
        fw = full["support"]["fwd_vs_matched"]
        L += ["", f"참고 — 지지선 접근 뒤 {H}거래일 종가 수익률: {_lv_ret(fw['actual'])} vs 가짜 선 날 {_lv_ret(fw['baseline'])} "
                  f"(차이 {'—' if fw.get('diff') is None else format(fw['diff'] * 100, '+.2f') + '%p'}, z {'—' if fw.get('z') is None else format(fw['z'], '+.1f')})", "",
              f"**손절선: 이탈 다음 날 시가 → {H}거래일 뒤 종가 수익률** (낮을수록 '팔았으면 손실을 피했다')", "",
              "| 기간 | 이탈 뒤 | 무작위 날(판정) | 차이 | z | 하락 폭 맞춘 날(참고) 차이 · z |", "|---|---|---|---|---|---|"]
        for per, b in rows:
            s = b["stop"]
            rm = s["vs_matched"]
            ref = "—" if rm.get("diff") is None else f"{rm['diff'] * 100:+.2f}%p · {'—' if rm.get('z') is None else format(rm['z'], '+.1f')}"
            L.append(f"| {per} | {_lv_cmp(s['vs_random'], _lv_ret)} | {ref} |")
        r20, tl = full["stop"]["r20_vs_random"], full["stop"]["tail_vs_random"]
        L += ["", f"참고 — {cfg['HORIZON_LONG']}거래일 뒤 수익률: {_lv_ret(r20['actual'])} vs 무작위 날 {_lv_ret(r20['baseline'])} "
                  f"(z {'—' if r20.get('z') is None else format(r20['z'], '+.1f')}) · {cfg['TAIL'] * 100:.0f}% 이하 비율: {_lv_rate(tl['actual'])} vs "
                  f"{_lv_rate(tl['baseline'])} (z {'—' if tl.get('z') is None else format(tl['z'], '+.1f')})", ""]
    L += ["## 읽는 법", "",
          "- 통과는 '과거에 그랬다'는 뜻이지 보장이 아니다. 미통과여도 손절선까지 거리로 매수 비중을 정하는 위험 계산(산수)은 그대로 쓸 수 있다.",
          "- 결과를 보고 선의 정의(3×ATR 스윙·0.5×ATR)나 기준을 바꿔 다시 시험하지 않는다. 바꾸려면 새 질문으로 등록하고 새 HOLDOUT이 필요하다.", ""]
    return "\n".join(L)
