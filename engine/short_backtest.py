"""단기 규칙 검증 (딥다이브 개편 4단계, 2026-11-16) — run_backtest.py --short-term.

규칙은 engine/strategy.py 맨 위(2026-11-15 사전 등록, 관행값)와 같다. 이 파일은 그 규칙을 과거 가격에 그대로 적용한다.
가격 기반 타이밍 규칙의 일곱 번째 시험이다(앞의 여섯 번은 우위가 없거나 미미 — AGENTS.md 4번 '검증 상태').
그래서 판정 기준(SHORT_TERM_CRITERIA)은 결과를 보기 전에 z ≥ 3으로 높여 고정했다(여러 번 시험하면 우연히 z ≥ 2가 나올
확률이 커진다 — Harvey·Liu·Zhu 2016은 새 요인에 t > 3을 권고). 판정 묶음(HOLDOUT)은 어떤 백테스트에도 쓰지 않은 종목이다.

비교 대상
    보유 기간 맞춘 무작위(판정 기준)  실제 거래마다 같은 종목의 무작위 날짜 j 다음 날 시가에 사서, 그 거래와 같은 날 수만큼 지나
        판다. 실제 거래가 종가 조건으로 다음 날 시가에 나갔으면 무작위도 시가에, +3% 익절(장중)이면 그날 종가에 판다.
    눌림 없이(참고)  틀 A 상태가 상승·횡보이고 지지선이 있는 날 중 무작위 — RSI(2) 눌림 조건만 빼고 진입 거리·청산 규칙은 같음.
        RSI(2) 눌림을 기다리는 것이 보태는 것이 있는지 본다.
표준오차는 같은 달 신호를 한 덩어리로 묶어 계산한다(engine/frame_backtest.py cluster_stats와 같음).
"""
import math
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .common import TECH_PARAMS
from .frame_backtest import FRAME_A_CONFIG, cluster_stats, compare_clustered
from .playbook import STATE_DOWN, _isna
from .rule_backtest import _bt_naive
from .strategy import SHORT_TERM_CONFIG, short_indicators

SHORT_BT_CONFIG: Dict[str, Any] = {
    "START": "2001-01-01",
    "DATA_START": "1999-01-01",
    "SPLIT_DATE": FRAME_A_CONFIG["SPLIT_DATE"],   # 앞 기간 / 뒤 기간 (2014-01-01)
    "BASELINE_SAMPLES": 20,
    "SEED": 23,
    "COST_SCENARIOS": [0.0, 0.003, 0.006],
    "MIN_BARS": 260,
    # 개발 묶음: 앱 대표 종목(틀 A 검증과 같은 묶음 — 이 규칙으로는 처음)
    "DEV_UNIVERSE": list(FRAME_A_CONFIG["DEV_UNIVERSE"]),
    # 판정 묶음(HOLDOUT, 2026-11-16 결과 보기 전 고정): 지금까지 어떤 백테스트(게이트·T 셋업·스캐너·재설계·틀 A·틀 B)에도
    # 쓰지 않은 종목. 미국은 11개 섹터에서 고르게(오래 부진한 종목 포함), 한국은 코스피 대형·중형주.
    "HOLDOUT_UNIVERSE": [
        "ADSK", "CDNS", "GLW", "MCHP", "XRX",                 # 기술
        "OMC", "EA", "CHTR", "LUMN",                          # 통신·미디어
        "BBY", "GPC", "M", "LEN", "LVS", "CCL",               # 경기소비재
        "CLX", "HSY", "CPB", "TSN", "STZ",                    # 필수소비재
        "SLB", "HAL", "OXY", "DVN", "WMB",                    # 에너지
        "AON", "PRU", "NTRS", "TROW", "BEN", "KEY",           # 금융
        "CAH", "BAX", "ZBH", "DGX", "BIIB", "IDXX",           # 헬스케어
        "ROK", "DOV", "PCAR", "GWW", "SWK", "LUV", "CHRW",    # 산업재
        "IP", "BALL", "NEM", "VMC", "IFF",                    # 소재
        "ED", "XEL", "PEG", "EIX", "PCG",                     # 유틸리티
        "O", "PSA", "AVB", "BXP", "VNO",                      # 리츠
        "000100.KS", "000720.KS", "004020.KS", "004170.KS", "005830.KS", "006800.KS", "008770.KS", "009540.KS",
        "010140.KS", "011070.KS", "011780.KS", "016360.KS", "021240.KS", "024110.KS", "029780.KS", "034220.KS",
        "042660.KS", "047050.KS", "051900.KS", "071050.KS", "078930.KS", "097950.KS", "001040.KS", "007070.KS",
        "090430.KS",
    ],
}
SHORT_TERM_CRITERIA: Dict[str, Any] = {
    # HOLDOUT 전체 기간: 거래 ≥ MIN_TRADES, 거래당 순수익률(비용 0.30% 뒤) 평균 > 0,
    #                   (실제 − 보유 기간 맞춘 무작위) ≥ 월 묶음 표준오차 × MIN_Z
    # 개발 묶음: 앞 기간·뒤 기간 모두 거래당 순수익률 평균 > 0
    "MIN_Z": 3.0,
    "MIN_TRADES": 30,
}
SHORT_EXIT_LABELS = {"target": "+3% 익절", "stop": "손절선 이탈(종가)", "trend_down": "추세 '하락' 전환",
                     "time": "10거래일 시간 청산", "open": "미청산(마지막 종가로 평가)"}


def _sb_cost(cost: Optional[float]) -> float:
    return float(TECH_PARAMS["ROUND_TRIP_COST"] if cost is None else cost)


def _sb_market(sym: str) -> str:
    return "KR" if str(sym).upper().endswith((".KS", ".KQ")) else "US"


def _sb_arrays(si: pd.DataFrame) -> Dict[str, np.ndarray]:
    a = {k: si[k].to_numpy(dtype=float) for k in ("Open", "High", "Low", "Close", "stop")}
    a["down"] = np.array([(not _isna(s)) and s == STATE_DOWN for s in si["state"].to_numpy(dtype=object)], dtype=bool)
    a["signal"] = si["short_signal"].to_numpy(dtype=bool)
    st = si["state"].to_numpy(dtype=object)
    a["pool_nopull"] = np.array([(not _isna(s)) and s in SHORT_TERM_CONFIG["STATES"] for s in st], dtype=bool) & np.isfinite(a["stop"])
    return a


def simulate_short_trade(a: Dict[str, np.ndarray], i: int, cost: Optional[float] = None) -> Optional[Dict[str, Any]]:
    """i일 종가 신호 → i+1일 시가 진입(손절선 < 시가, 거리 ≤ 3%일 때만) → 청산 규칙. 진입 안 하면 {'skipped': 이유}."""
    c = SHORT_TERM_CONFIG
    o, h, cl, stop_arr, down = a["Open"], a["High"], a["Close"], a["stop"], a["down"]
    n = len(cl)
    if i + 1 >= n:
        return None
    fill, stop = float(o[i + 1]), float(stop_arr[i])
    if not (np.isfinite(fill) and np.isfinite(stop) and fill > 0):
        return None
    if fill <= stop:
        return {"skipped": "below_stop"}
    dist = (fill - stop) / fill
    if dist > c["MAX_STOP_DIST"]:
        return {"skipped": "far"}
    s = i + 1
    target = fill * (1.0 + c["TARGET"])
    kind, exit_idx, exit_px, exit_at = "open", n - 1, float(cl[n - 1]), "close"
    for k in range(s, n):
        if k > s and o[k] >= target:
            kind, exit_idx, exit_px, exit_at = "target", k, float(o[k]), "open"
            break
        if h[k] >= target:
            kind, exit_idx, exit_px, exit_at = "target", k, float(target), "close"
            break
        if k == n - 1:
            break
        why = ("stop" if cl[k] < stop else ("trend_down" if down[k] else ("time" if k - s + 1 >= c["MAX_DAYS"] else None)))
        if why:
            kind, exit_idx, exit_px, exit_at = why, k + 1, float(o[k + 1]), "open"
            break
    gross = exit_px / fill - 1.0
    cst = _sb_cost(cost)
    return {"fill": fill, "stop": stop, "stop_dist": float(dist), "exit_px": exit_px, "exit_kind": kind,
            "exit_idx": int(exit_idx), "exit_at": exit_at, "hold_days": int(exit_idx - s), "gross": float(gross),
            "ret": float(gross - cst), "r": float((exit_px - fill - fill * cst) / (fill - stop))}


def short_trades(si: pd.DataFrame, start: Optional[str] = None, cost: Optional[float] = None,
                 arrs: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, Any]:
    """한 종목의 단기 거래 (종목당 1포지션: 청산 실행일 종가부터 새 신호를 받음)."""
    a = arrs or _sb_arrays(si)
    s0 = pd.Timestamp(start or SHORT_BT_CONFIG["START"])
    out, skipped, free_from = [], {"far": 0, "below_stop": 0}, -1
    for i in np.flatnonzero(a["signal"]):
        if si.index[i] < s0 or i < free_from:
            continue
        res = simulate_short_trade(a, int(i), cost)
        if res is None:
            continue
        if res.get("skipped"):
            skipped[res["skipped"]] += 1
            continue
        out.append({"date": si.index[i], **res})
        if res["exit_kind"] == "open":
            break
        free_from = res["exit_idx"]
    return {"trades": pd.DataFrame(out), "skipped": skipped}


def short_matched_baseline(si: pd.DataFrame, trades: pd.DataFrame, start: Optional[str] = None,
                           samples: Optional[int] = None, seed: Optional[int] = None, cost: Optional[float] = None,
                           arrs: Optional[Dict[str, np.ndarray]] = None) -> pd.DataFrame:
    """보유 기간 맞춘 무작위: 거래마다 무작위 날짜 j → j+1 시가 매수 → j+1+h일 시가(실제가 시가 청산) 또는 종가(실제가 장중 익절)."""
    a = arrs or _sb_arrays(si)
    if trades is None or not len(trades):
        return pd.DataFrame()
    o, cl = a["Open"], a["Close"]
    n = len(o)
    first = int(np.searchsorted(si.index.values, np.datetime64(pd.Timestamp(start or SHORT_BT_CONFIG["START"]))))
    rng = np.random.default_rng(SHORT_BT_CONFIG["SEED"] if seed is None else seed)
    k = int(samples or SHORT_BT_CONFIG["BASELINE_SAMPLES"])
    cst = _sb_cost(cost)
    out = []
    for h, at in zip(trades["hold_days"].astype(int).to_numpy(), trades["exit_at"].astype(str).to_numpy()):
        last = n - 2 - h
        if last < first:
            continue
        for j in rng.integers(first, last + 1, size=k):
            fill = o[j + 1]
            ex = o[j + 1 + h] if at == "open" else cl[j + 1 + h]
            if not (np.isfinite(fill) and np.isfinite(ex) and fill > 0):
                continue
            g = ex / fill - 1.0
            out.append({"date": si.index[j], "fill": float(fill), "exit_px": float(ex), "gross": float(g),
                        "ret": float(g - cst), "hold_days": int(h), "exit_kind": "hold", "exit_at": at})
    return pd.DataFrame(out)


def short_nopull_baseline(si: pd.DataFrame, n_trades: int, start: Optional[str] = None, samples: Optional[int] = None,
                          seed: Optional[int] = None, cost: Optional[float] = None,
                          arrs: Optional[Dict[str, np.ndarray]] = None) -> pd.DataFrame:
    """눌림 없이(참고): 추세 상승·횡보 + 지지선 있는 날 중 무작위, 같은 진입 거리·청산 규칙."""
    a = arrs or _sb_arrays(si)
    if n_trades == 0:
        return pd.DataFrame()
    s0 = pd.Timestamp(start or SHORT_BT_CONFIG["START"])
    ok = a["pool_nopull"] & (si.index >= s0)
    valid = np.flatnonzero(ok)
    valid = valid[valid < len(si) - 2]
    if not len(valid):
        return pd.DataFrame()
    rng = np.random.default_rng((SHORT_BT_CONFIG["SEED"] if seed is None else seed) + 1)
    out = []
    for j in rng.choice(valid, size=n_trades * int(samples or SHORT_BT_CONFIG["BASELINE_SAMPLES"]), replace=True):
        res = simulate_short_trade(a, int(j), cost)
        if res and not res.get("skipped"):
            out.append({"date": si.index[j], **res})
    return pd.DataFrame(out)


def run_short_universe(price_map: Dict[str, pd.DataFrame], cost: Optional[float] = None) -> Dict[str, Any]:
    cfg = SHORT_BT_CONFIG
    trs, rnds, nps, used = [], [], [], []
    skipped = {"far": 0, "below_stop": 0}
    years = 0.0
    s0 = pd.Timestamp(cfg["START"])
    for sym, df in (price_map or {}).items():
        d = _bt_naive(df)
        if d is not None and len(d):   # 거래 정지일 등 시가·고가·저가가 0으로 온 줄은 뺀다(0원 저점이 지지선이 되지 않게)
            px_cols = [k for k in ("Open", "High", "Low", "Close") if k in d]
            d = d[(d[px_cols].astype(float) > 0).all(axis=1)]
        if d is None or len(d) < cfg["MIN_BARS"]:
            continue
        si = short_indicators(d)
        a = _sb_arrays(si)
        r = short_trades(si, cost=cost, arrs=a)
        tr = r["trades"]
        for k2 in skipped:
            skipped[k2] += r["skipped"][k2]
        used.append(sym)
        mk = _sb_market(sym)
        rnd = short_matched_baseline(si, tr, cost=cost, arrs=a)
        npb = short_nopull_baseline(si, len(tr), cost=cost, arrs=a)
        for lst, x in ((trs, tr), (rnds, rnd), (nps, npb)):
            if len(x):
                lst.append(x.assign(ticker=sym, market=mk))
        dd = si.index[si.index >= s0]
        if len(dd) > 1:
            years += (dd[-1] - dd[0]).days / 365.25
    cols = ["date", "ticker", "market", "fill", "stop", "exit_px", "ret", "gross", "r", "exit_kind", "exit_at", "hold_days"]
    cat = lambda lst: pd.concat(lst, ignore_index=True) if lst else pd.DataFrame(columns=cols)
    return {"trades": cat(trs), "random": cat(rnds), "nopull": cat(nps), "skipped": skipped, "tickers": used, "years": years}


def _sb_block(tr: pd.DataFrame, rnd: pd.DataFrame, npb: pd.DataFrame) -> Dict[str, Any]:
    out = {"vs_random": compare_clustered(tr, rnd), "vs_nopull": compare_clustered(tr, npb),
           "r": cluster_stats(tr["r"], tr["date"]) if len(tr) else {"n": 0},
           "hold_days": float(tr["hold_days"].mean()) if len(tr) else None,
           "target_rate": float((tr["exit_kind"] == "target").mean() * 100.0) if len(tr) else None,
           "worst_ret": float(tr["ret"].min()) if len(tr) else None,
           "p5_ret": float(tr["ret"].quantile(0.05)) if len(tr) else None,
           "exits": {}, "cost": {}}
    if len(tr):
        for k, g in tr.groupby("exit_kind"):
            out["exits"][SHORT_EXIT_LABELS.get(k, k)] = {"share": float(len(g) / len(tr) * 100.0), "mean_ret": float(g["ret"].mean()),
                                                         "hold_days": float(g["hold_days"].mean())}
        for cst in SHORT_BT_CONFIG["COST_SCENARIOS"]:
            a2 = tr.assign(ret=tr["gross"] - cst)
            b2 = rnd.assign(ret=rnd["gross"] - cst) if len(rnd) else rnd
            cm = compare_clustered(a2, b2)
            out["cost"][f"{cst * 100:.2f}%"] = {"mean": cm["actual"].get("mean"), "diff": cm["diff"], "z": cm["z"]}
    return out


def evaluate_short_term(dev_map: Dict[str, pd.DataFrame], holdout_map: Dict[str, pd.DataFrame],
                        cost: Optional[float] = None) -> Dict[str, Any]:
    """개발 묶음(앞·뒤 기간)과 HOLDOUT에서 단기 규칙을 평가하고 사전 기준으로 판정."""
    cfg, crit = SHORT_BT_CONFIG, SHORT_TERM_CRITERIA
    split_t = pd.Timestamp(cfg["SPLIT_DATE"])
    c = SHORT_TERM_CONFIG
    rules = (f"진입: 틀 A 상태 상승·횡보 + RSI(2) < {c['RSI_MAX']:.0f} → 다음 날 시가(손절선까지 ≤ {c['MAX_STOP_DIST'] * 100:.0f}%, "
             f"손절선 = 최근 확정 스윙 저점 − {c['STOP_ATR']}×ATR(20)) · 청산: 장중 +{c['TARGET'] * 100:.0f}% 익절 / 종가 < 손절선 / "
             f"추세 '하락' / {c['MAX_DAYS']}거래일 — 종가 조건은 다음 날 시가")
    res: Dict[str, Any] = {"rules": rules, "version": c["VERSION"], "criteria": crit,
                           "config": {k: cfg[k] for k in ("START", "SPLIT_DATE", "BASELINE_SAMPLES", "COST_SCENARIOS")},
                           "cost": _sb_cost(cost), "universes": {}, "verdict": {}}
    raw = {"개발(대표 종목)": run_short_universe(dev_map, cost), "HOLDOUT(새 종목)": run_short_universe(holdout_map, cost)}
    for uname, u in raw.items():
        tr, rnd, npb = u["trades"], u["random"], u["nopull"]
        per = {"전체 기간": lambda x: x, "앞 기간": lambda x: x[x["date"] < split_t], "뒤 기간": lambda x: x[x["date"] >= split_t]}
        blocks = {p: _sb_block(f(tr), f(rnd), f(npb)) for p, f in per.items()}
        by_mk = {mk: _sb_block(tr[tr["market"] == mk], rnd[rnd["market"] == mk], npb[npb["market"] == mk])
                 for mk in ("US", "KR") if len(tr) and (tr["market"] == mk).any()}
        res["universes"][uname] = {"n_tickers": len(u["tickers"]), "skipped": u["skipped"],
                                   "trades_per_ticker_year": float(len(tr) / u["years"]) if u["years"] else None,
                                   "blocks": blocks, "by_market": by_mk}
    ho = res["universes"]["HOLDOUT(새 종목)"]["blocks"]["전체 기간"]["vs_random"]
    dv = res["universes"]["개발(대표 종목)"]["blocks"]
    n_ho = ho["actual"].get("n", 0)
    ok_ho = bool(n_ho >= crit["MIN_TRADES"] and ho["actual"]["mean"] > 0 and ho["z"] is not None and ho["z"] >= crit["MIN_Z"])
    ok_dev = all(dv[p]["vs_random"]["actual"].get("n", 0) >= 2 and dv[p]["vs_random"]["actual"]["mean"] > 0 for p in ("앞 기간", "뒤 기간"))
    npz = res["universes"]["HOLDOUT(새 종목)"]["blocks"]["전체 기간"]["vs_nopull"].get("z")
    res["verdict"] = {"passed": ok_ho and ok_dev, "holdout_pass": ok_ho, "dev_both_positive": ok_dev,
                      "enough_trades": n_ho >= crit["MIN_TRADES"],
                      "z2_reference": bool(n_ho >= crit["MIN_TRADES"] and ho["actual"]["mean"] > 0 and ho["z"] is not None and ho["z"] >= 2.0),
                      "pullback_adds": bool(npz is not None and npz >= 2.0),
                      "criteria": (f"HOLDOUT 전체 기간 거래 {crit['MIN_TRADES']}건 이상, 거래당 순수익률(비용 뒤) > 0 이고 "
                                   f"(실제 − 보유 기간 맞춘 무작위) ≥ 월 묶음 표준오차 × {crit['MIN_Z']:.0f}, "
                                   "그리고 개발 묶음 앞·뒤 기간 모두 거래당 순수익률 > 0")}
    trades = (pd.concat([u["trades"].assign(universe=k) for k, u in raw.items() if len(u["trades"])], ignore_index=True)
              if any(len(u["trades"]) for u in raw.values()) else pd.DataFrame())
    return {"summary": res, "trades": trades}


# ---------------------------------------------------------------------
# 보고서
# ---------------------------------------------------------------------
def _sb_pct(v: Optional[float], d: int = 2) -> str:
    return "—" if v is None or (isinstance(v, float) and not math.isfinite(v)) else f"{v * 100:+.{d}f}%"


def _sb_one(s: Dict[str, Any]) -> str:
    if not s.get("n"):
        return "표본 없음"
    se = f" ±{s['se'] * 100:.2f}" if s.get("se") is not None else ""
    return f"{s['mean'] * 100:+.2f}%{se} ({s['win_rate']:.0f}% · {s['n']:,}건)"


def _sb_row(x: Dict[str, Any]) -> str:
    z = "—" if x.get("z") is None else f"{x['z']:+.1f}"
    zi = "—" if x.get("z_iid") is None else f"{x['z_iid']:+.1f}"
    return f"{_sb_one(x['actual'])} | {_sb_one(x['baseline'])} | {_sb_pct(x.get('diff'))} | {z} ({zi})"


def render_short_term_report(r: Dict[str, Any], generated_at: Optional[str] = None) -> str:
    v, cfg = r["verdict"], r["config"]
    L = ["# 단기 규칙 검증 (딥다이브 4단계)", "",
         f"- 생성: {generated_at or '—'} · 규칙 {r['version']} · 신호 {cfg['START'][:4]}년~ · 왕복 비용 {r['cost'] * 100:.2f}% 차감",
         f"- 규칙(2026-11-15 사전 등록, 관행값): {r['rules']}",
         "- 개발 묶음 = 앱 대표 종목, HOLDOUT = 어떤 백테스트에도 쓰지 않은 미국 59·한국 25 종목(2026-11-16 결과 보기 전 고정).",
         f"- 무작위 비교(판정 기준): 실제 거래마다 같은 종목의 무작위 날짜 {cfg['BASELINE_SAMPLES']}곳에서 사서 같은 날 수만큼 들고 판 결과"
         "(실제가 장중 익절이면 그날 종가, 시가 청산이면 시가).",
         "- '눌림 없이'(참고): 추세 상승·횡보 + 지지선이 있는 날 중 무작위, 같은 진입 거리·청산 규칙 — RSI(2) 눌림을 기다리는 것이 보태는지.",
         "- 가격 기반 타이밍의 일곱 번째 시험이라 판정 z 기준을 3으로 높여 미리 고정했다(z ≥ 2는 참고로만 적음). "
         "표준오차는 같은 달 신호를 한 덩어리로 묶어 계산, 괄호 안 z는 거래를 서로 독립으로 본 값(참고).",
         f"- 판정 기준: {v['criteria']}.", "",
         f"## 판정: {'통과' if v['passed'] else '미통과'}", "",
         f"- HOLDOUT 기준(z ≥ 3) {'통과' if v['holdout_pass'] else '미통과'} · 참고 z ≥ 2 {'충족' if v['z2_reference'] else '미충족'} · "
         f"개발 묶음 앞·뒤 기간 {'모두 > 0' if v['dev_both_positive'] else '하나 이상 ≤ 0'} · 거래 수 {'충분' if v['enough_trades'] else '부족'} · "
         f"눌림의 기여('눌림 없이' 대비 z ≥ 2) {'확인' if v['pullback_adds'] else '확인 안 됨'}", ""]
    for uname, u in r["universes"].items():
        sk = u["skipped"]
        L += [f"## {uname}", "",
              f"종목 {u['n_tickers']}개 · 종목당 연 {u['trades_per_ticker_year'] or 0:.1f}건 · 진입 안 함: 손절선까지 3% 초과 {sk['far']:,}건, "
              f"시가가 손절선 이하 {sk['below_stop']:,}건", "",
              "| 기간 | 단기 거래당 순수익률 ±표준오차 (승률·건수) | 보유 기간 맞춘 무작위 | 차이 | z (독립 가정) | 평균 보유일 | +3% 도달 |",
              "|---|---|---|---|---|---|---|"]
        for per, b in u["blocks"].items():
            hd = "—" if b["hold_days"] is None else f"{b['hold_days']:.1f}"
            tg = "—" if b["target_rate"] is None else f"{b['target_rate']:.0f}%"
            L.append(f"| {per} | {_sb_row(b['vs_random'])} | {hd} | {tg} |")
        for mk, b in u["by_market"].items():
            L.append(f"| 전체 기간 · {'미국' if mk == 'US' else '한국'} | {_sb_row(b['vs_random'])} | "
                     f"{'—' if b['hold_days'] is None else format(b['hold_days'], '.1f')} | "
                     f"{'—' if b['target_rate'] is None else format(b['target_rate'], '.0f') + '%'} |")
        full = u["blocks"]["전체 기간"]
        L += ["", "| 비교 | 단기 | 눌림 없이(같은 규칙) | 차이 | z (독립 가정) |", "|---|---|---|---|---|",
              f"| 전체 기간 | {_sb_row(full['vs_nopull'])} |", ""]
        if full["exits"]:
            L += ["| 청산 이유 | 비중 | 거래당 순수익률 | 평균 보유일 |", "|---|---|---|---|"]
            for k, e in sorted(full["exits"].items(), key=lambda kv: -kv[1]["share"]):
                L.append(f"| {k} | {e['share']:.0f}% | {_sb_pct(e['mean_ret'])} | {e['hold_days']:.1f} |")
            L.append("")
        if full["cost"]:
            L += ["| 왕복 비용 | 거래당 순수익률 | 무작위 대비 차이 | z |", "|---|---|---|---|"]
            for k, cs in full["cost"].items():
                L.append(f"| {k} | {_sb_pct(cs['mean'])} | {_sb_pct(cs['diff'])} | {'—' if cs['z'] is None else format(cs['z'], '+.1f')} |")
            L.append("")
        if full["worst_ret"] is not None:
            L += [f"최악 거래 {_sb_pct(full['worst_ret'])} · 하위 5% {_sb_pct(full['p5_ret'])} · R 단위(손절선까지 거리 = 1R) "
                  f"평균 {(full['r'].get('mean') or 0):+.2f}R", ""]
    L += ["## 읽는 법", "",
          "- 통과해도 '과거 이 규칙이 무작위보다 나았다'는 뜻이지 앞으로의 수익 보장이 아니다. 미통과면 화면의 단기 카드는 '검증 미통과 — 참고만'으로 표시한다.",
          "- 결과를 보고 숫자(RSI 10·3%·0.5×ATR·10거래일)를 바꿔 다시 시험하지 않는다. 바꾸려면 새 질문으로 등록하고 새 HOLDOUT이 필요하다.", ""]
    return "\n".join(L)
