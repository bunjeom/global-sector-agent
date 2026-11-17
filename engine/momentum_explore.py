"""탐색: 모멘텀(상대강도) — 최근 몇 달 남보다 많이 오른 종목이 다음 몇 달에도 더 오르는가 (2026-11-17).
run_backtest.py --explore mom12 / --explore ret63. 탐색용 종목으로 보는 것이라 결과는 증거가 아니다(engine/explore.py 규칙).

연구 배경: 과거 3~12개월 수익률이 높은 종목이 이후 3~12개월 더 오른다(Jegadeesh·Titman 1993). 직전 1개월은 반대로 되돌아가는
경향이 있어 12개월 모멘텀은 최근 1개월을 뺀다. 이번 시험은 같은 현상이 우리 종목·기간에서 보이는지를 본다.

정의(결과 보기 전 고정)
    줄 세우는 날  2001년부터 매달 마지막 거래일(종목마다 그달 마지막 거래일, 그날이 그달 말에서 5일 넘게 떨어지면 제외)
    신호  mom12 = 252거래일 전 종가 → 21거래일 전 종가 수익률(최근 1개월 제외)
          ret63 = 63거래일 전 종가 → 그날 종가 수익률(앱 rs63과 같은 기간. 같은 시장 안 순위라 기준 지수를 빼도 순서가 같음)
    결과  다음 날 시가 → 21거래일 뒤 종가(1개월, 판정) · 63거래일 뒤 종가(3개월, 참고)
    묶음  같은 달·같은 시장(미국·한국 따로)에서 신호와 결과가 있는 종목이 20개 이상일 때만. 상위 20% vs 하위 20%(순위 백분위)
    달마다  상위 − 하위(판정), 상위 − 전체 평균(사기만 하는 경우, 참고). 전체 = 그달 미국·한국 값의 평균(있는 시장만)
    표준오차  달마다 한 값이라 1개월은 그대로, 3개월은 겹치는 기간 때문에 Newey-West(시차 2)
    후보  1개월 상위 − 하위 > 0, t ≥ 2, 앞·뒤 기간(2014 분할) 모두 > 0, 달 수 ≥ 100
"""
import math
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .explore import EXPLORE_CONFIG, EXPLORE_HYPOTHESES
from .rule_backtest import _bt_naive

MOM_CONFIG: Dict[str, Any] = {
    "SIGNALS": {"mom12": (252, 21), "ret63": (63, 0)},   # (몇 거래일 전부터, 몇 거래일 전까지)
    "H1": 21, "H3": 63,
    "TOP": 0.8, "BOTTOM": 0.2,
    "MIN_STOCKS": 20,
    "MONTH_GAP_DAYS": 5,
    "MIN_MONTHS": 100,
    "NW_LAG_3M": 2,
}


def nw_se(x: np.ndarray, lag: int = 0) -> Optional[float]:
    """평균의 표준오차(Newey-West). lag 0이면 보통 표준오차."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 3:
        return None
    e = x - x.mean()
    s = float((e * e).sum()) / n
    for k in range(1, lag + 1):
        w = 1.0 - k / (lag + 1.0)
        s += 2.0 * w * float((e[k:] * e[:-k]).sum()) / n
    return math.sqrt(max(s, 0.0) / n)


def stock_month_rows(df: pd.DataFrame, signal: str) -> pd.DataFrame:
    """한 종목의 달마다 신호·1·3개월 결과. 열: month, sig, f1, f3."""
    c = MOM_CONFIG
    d = _bt_naive(df)
    if d is None or not len(d):
        return pd.DataFrame(columns=["month", "sig", "f1", "f3"])
    cols = [k for k in ("Open", "High", "Low", "Close") if k in d]
    d = d[(d[cols].astype(float) > 0).all(axis=1)]
    cl = d["Close"].to_numpy(dtype=float)
    op = d["Open"].to_numpy(dtype=float) if "Open" in d else cl
    idx = d.index
    n = len(cl)
    lb, skip = c["SIGNALS"][signal]
    per = idx.to_period("M")
    last_pos = pd.Series(np.arange(n), index=idx).groupby(per).max()
    out = []
    for m, i in last_pos.items():
        i = int(i)
        if (m.to_timestamp(how="end") - idx[i]).days > c["MONTH_GAP_DAYS"] or i - lb < 0:
            continue
        sig = cl[i - skip] / cl[i - lb] - 1.0
        f1 = cl[i + c["H1"]] / op[i + 1] - 1.0 if i + c["H1"] <= n - 1 else np.nan
        f3 = cl[i + c["H3"]] / op[i + 1] - 1.0 if i + c["H3"] <= n - 1 else np.nan
        out.append({"month": m, "sig": sig, "f1": f1, "f3": f3})
    return pd.DataFrame(out, columns=["month", "sig", "f1", "f3"])


def monthly_spreads(panel: pd.DataFrame) -> pd.DataFrame:
    """panel(열: month, market, sig, f1, f3) → 달·시장마다 상위·하위·전체 평균과 차이."""
    c = MOM_CONFIG
    rows = []
    for (m, mk), g in panel.groupby(["month", "market"], sort=True):
        g = g[np.isfinite(g["sig"])]
        r = {"month": m, "market": mk}
        for h in ("f1", "f3"):
            gg = g[np.isfinite(g[h])]
            if len(gg) < c["MIN_STOCKS"]:
                r.update({f"top_{h}": np.nan, f"bot_{h}": np.nan, f"all_{h}": np.nan, f"n_{h}": len(gg)})
                continue
            pct = gg["sig"].rank(pct=True, method="average")
            r.update({f"top_{h}": float(gg.loc[pct > c["TOP"], h].mean()), f"bot_{h}": float(gg.loc[pct <= c["BOTTOM"], h].mean()),
                      f"all_{h}": float(gg[h].mean()), f"n_{h}": len(gg)})
        rows.append(r)
    ms = pd.DataFrame(rows)
    for h in ("f1", "f3"):
        ms[f"ls_{h}"] = ms[f"top_{h}"] - ms[f"bot_{h}"]
        ms[f"tm_{h}"] = ms[f"top_{h}"] - ms[f"all_{h}"]
    return ms


def _stat(x: pd.Series, lag: int) -> Dict[str, Any]:
    v = x.dropna().to_numpy(dtype=float)
    if len(v) < 3:
        return {"n": int(len(v)), "mean": None, "se": None, "t": None, "hit": None, "worst": None}
    se = nw_se(v, lag)
    return {"n": int(len(v)), "mean": float(v.mean()), "se": se, "t": (float(v.mean()) / se if se else None),
            "hit": float((v > 0).mean()), "worst": float(v.min()), "best": float(v.max())}


def _mom_block(ms: pd.DataFrame) -> Dict[str, Any]:
    lag3 = MOM_CONFIG["NW_LAG_3M"]
    return {"ls_f1": _stat(ms["ls_f1"], 0), "tm_f1": _stat(ms["tm_f1"], 0), "ls_f3": _stat(ms["ls_f3"], lag3), "tm_f3": _stat(ms["tm_f3"], lag3),
            "top_f1": _stat(ms["top_f1"], 0), "bot_f1": _stat(ms["bot_f1"], 0), "all_f1": _stat(ms["all_f1"], 0)}


def run_momentum_explore(price_map: Dict[str, pd.DataFrame], signal: str = "mom12") -> Dict[str, Any]:
    ec, c = EXPLORE_CONFIG, MOM_CONFIG
    parts, used = [], 0
    s0 = pd.Timestamp(ec["START"]).to_period("M")
    for sym, df in (price_map or {}).items():
        r = stock_month_rows(df, signal)
        if not len(r):
            continue
        used += 1
        r = r[r["month"] >= s0]
        if len(r):
            parts.append(r.assign(ticker=sym, market="KR" if str(sym).upper().endswith((".KS", ".KQ")) else "US"))
    panel = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["month", "sig", "f1", "f3", "ticker", "market"])
    ms = monthly_spreads(panel) if len(panel) else pd.DataFrame(columns=["month", "market"])
    cols = [k for k in ms.columns if k not in ("month", "market") and not k.startswith("n_")]
    overall = ms.groupby("month")[cols].mean().reset_index() if len(ms) else ms
    split = pd.Timestamp(ec["SPLIT_DATE"]).to_period("M")
    sel = {"전체 기간": overall, "앞 기간": overall[overall["month"] < split], "뒤 기간": overall[overall["month"] >= split],
           "전체 기간 · 미국": ms[ms["market"] == "US"], "전체 기간 · 한국": ms[ms["market"] == "KR"]}
    blocks = {p: _mom_block(x) for p, x in sel.items()}
    f, fr, bk = (blocks[p]["ls_f1"] for p in ("전체 기간", "앞 기간", "뒤 기간"))
    same = bool(f["mean"] is not None and fr["mean"] is not None and bk["mean"] is not None and f["mean"] > 0 and fr["mean"] > 0 and bk["mean"] > 0)
    cand = bool(f["n"] >= c["MIN_MONTHS"] and f["t"] is not None and f["t"] >= ec["CANDIDATE_Z"] and same)
    reasons = []
    if f["n"] < c["MIN_MONTHS"]:
        reasons.append(f"달 수 {f['n']} < {c['MIN_MONTHS']}")
    if f["mean"] is not None and f["mean"] <= 0:
        reasons.append("상위 20%가 하위 20%보다 낫지 않음")
    elif f["t"] is None or f["t"] < ec["CANDIDATE_Z"]:
        reasons.append(f"t < {ec['CANDIDATE_Z']:.0f}")
    if not same and (f["mean"] is None or f["mean"] > 0):
        reasons.append("앞·뒤 기간 방향이 다름")
    h = EXPLORE_HYPOTHESES[signal]
    log_row = {"actual": {"n": f["n"]}, "diff": f["mean"], "z": f["t"]}
    lb, skip = c["SIGNALS"][signal]
    return {"name": signal, "label": h["label"], "signal": signal, "n_tickers": used, "n_events": f["n"], "n_stock_months": int(len(panel)),
            "blocks_mom": blocks, "candidate": cand, "not_candidate_reasons": reasons, "blocks": {"전체 기간": {"vs_random": log_row}},
            "config": {"START": ec["START"], "SPLIT_DATE": ec["SPLIT_DATE"], "LOOKBACK": lb, "SKIP": skip, **{k: c[k] for k in ("H1", "H3", "TOP", "BOTTOM", "MIN_STOCKS", "MIN_MONTHS")}},
            "criteria": (f"후보 = 1개월 뒤 (상위 20% − 하위 20%) > 0, t ≥ {ec['CANDIDATE_Z']:.0f}, 앞·뒤 기간 모두 > 0, 달 수 {c['MIN_MONTHS']} 이상")}


def render_momentum_report(r: Dict[str, Any], generated_at: Optional[str] = None) -> str:
    c = r["config"]
    p = lambda v, d=2: "—" if v is None else f"{v * 100:+.{d}f}%"
    pp = lambda v, d=2: "—" if v is None else f"{v * 100:+.{d}f}%p"
    t = lambda v: "—" if v is None else f"{v:+.1f}"
    sig_txt = ("12개월 모멘텀 = 252거래일 전 → 21거래일 전 수익률(최근 1개월 제외)" if r["signal"] == "mom12"
               else "63거래일 수익률(앱 rs63과 같은 기간, 같은 시장 안 순위라 기준 지수를 빼도 순서가 같음)")
    L = [f"# 탐색: {r['label']}", "",
         "**탐색 결과 — 증거 아님.** 이미 판정에 쓴 종목으로 본 것이라, 좋게 나와도 '확인해 볼 만하다'는 뜻일 뿐이다.", "",
         f"- 생성: {generated_at or '—'} · 가설 이름 `{r['name']}` · 종목 {r['n_tickers']}개 · {c['START'][:4]}년~ · 종목·달 {r['n_stock_months']:,}개",
         f"- 줄 세우기: 매달 마지막 거래일, 시장별(미국·한국 따로), {sig_txt}. 상위 20% vs 하위 20%, 그달·그 시장 종목 {c['MIN_STOCKS']}개 이상일 때만.",
         f"- 결과: 다음 날 시가 → {c['H1']}거래일(1개월, 판정)·{c['H3']}거래일(3개월, 참고) 뒤 종가. 같은 비중 평균, 비용 빼기 전.",
         "- 전체 = 그달 미국·한국 값의 평균. 3개월은 기간이 겹쳐 Newey-West 표준오차(시차 2).",
         f"- {r['criteria']}.",
         "- 주의: 지금 살아 있는 회사만 들어 있어(상장폐지 종목 없음) 하위 20%가 실제보다 좋게 보일 수 있다 — 상위 − 하위 차이가 실제보다 작게 나오는 쪽.", "",
         f"## 후보 여부: {'후보 — 확인 시험으로 올릴 만함' if r['candidate'] else '후보 아님'}"
         + ("" if r["candidate"] else f" ({', '.join(r['not_candidate_reasons'])})"), "",
         "**1개월 뒤 (판정)**", "",
         "| 구분 | 달 수 | 상위 20% | 하위 20% | 상위 − 하위 | t | 상위가 이긴 달 | 최악의 달 | 상위 − 전체 평균 | t |", "|---|---|---|---|---|---|---|---|---|---|"]
    for per, b in r["blocks_mom"].items():
        ls, tm = b["ls_f1"], b["tm_f1"]
        L.append(f"| {per} | {ls['n']} | {p(b['top_f1']['mean'])} | {p(b['bot_f1']['mean'])} | {pp(ls['mean'])} | {t(ls['t'])} | "
                 f"{'—' if ls['hit'] is None else format(ls['hit'] * 100, '.0f') + '%'} | {pp(ls['worst'], 1)} | {pp(tm['mean'])} | {t(tm['t'])} |")
    L += ["", "**3개월 뒤 (참고)**", "", "| 구분 | 상위 − 하위 | t | 상위 − 전체 평균 | t |", "|---|---|---|---|---|"]
    for per, b in r["blocks_mom"].items():
        L.append(f"| {per} | {pp(b['ls_f3']['mean'])} | {t(b['ls_f3']['t'])} | {pp(b['tm_f3']['mean'])} | {t(b['tm_f3']['t'])} |")
    L += ["", "## 읽는 법", "",
          "- '상위 − 하위'는 강한 종목을 사고 약한 종목을 팔았을 때, '상위 − 전체 평균'은 강한 종목만 샀을 때 평소보다 얼마나 나았는지다.",
          "- 달마다 한 번 바꾸는 전략이라 실제로는 거래 비용이 든다. 최악의 달은 모멘텀이 한 번에 크게 무너진 때다.",
          "- 후보가 아니면 여기서 끝낸다. 기간(252·63·21일)이나 20% 기준을 바꿔 가며 다시 보지 않는다.", ""]
    return "\n".join(L)


EXPLORE_HYPOTHESES["mom12"] = {
    "label": "12개월 모멘텀(최근 1개월 제외) 상위 20%가 하위 20%보다 다음 1개월 더 오르는가", "expect": 1, "custom": "mom",
    "note": "연구로 알려진 현상(Jegadeesh·Titman 1993)을 우리 데이터로 확인. 기록의 '차이'는 1개월 상위 − 하위(%p), z 칸은 t",
}
EXPLORE_HYPOTHESES["ret63"] = {
    "label": "63거래일 수익률(앱 rs63과 같은 기간) 상위 20%가 하위 20%보다 다음 1개월 더 오르는가", "expect": 1, "custom": "mom",
    "note": "앱 피어 비교 추세 관점의 63일 상대강도가 의미 있는지 확인. 기록의 '차이'는 1개월 상위 − 하위(%p), z 칸은 t",
}
