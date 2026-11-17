"""지지선 이탈 뒤 회복 검증 (2026-11-17) — run_backtest.py --reclaim → data/backtest/RECLAIM.md.

사용자 가설: "지지선에서 바로 지지받기보다, 한 번 깼다가 다시 올라오는 경우가 많다."
지지·손절선 시험(LEVELS.md)이 끝난 뒤 나온 새 질문이라, 새 종목 묶음과 미리 고정한 기준으로 따로 시험한다(가격 기반 열 번째 질문).
선의 정의는 그대로다: 지지선 = 그날까지 확정된 가장 최근 스윙 저점(지그재그 3×ATR).

A. 얼마나 자주 깨고, 깬 뒤 얼마나 자주 회복하나 (판정 없는 집계)
    접근 사건(LEVELS와 같음: 저가 ≤ 지지선 + 0.5×ATR, 종가 ≥ 손절선) 뒤 10거래일:
        버팀(종가가 지지선 아래로 안 감) / 깼다가 회복(종가가 지지선 아래로 간 뒤 그 창 안에서 다시 지지선 위 종가) / 깨고 못 돌아옴
    이탈 사건(종가가 지지선 아래로 처음 내려온 날, 전날 종가는 지지선 이상) 뒤 10거래일 안에 종가가 다시 지지선 위로 온 비율,
        걸린 날 수(중앙값), 가장 깊이 내려간 폭(ATR 단위, 중앙값)
B. 깨고 올라온 뒤가 정말 좋은가 (판정)
    사건  이탈 사건 뒤 10거래일 안에 종가가 처음으로 다시 지지선(이탈일의 지지선) 위로 올라온 날 u
    결과  u+1 시가 → 10거래일 뒤 종가 수익률(판정), 20거래일(참고), 10거래일 안에 종가가 다시 지지선 아래로 간 비율(참고)
    비교(판정 기준)  같은 종목 무작위 날 j의 j+1 시가 → 10거래일 뒤 종가. 참고: 최근 5거래일 변화(ATR 단위)가 비슷한 날.
    통과  HOLDOUT 사건 ≥ 100, (회복 뒤 − 무작위) > 0, z ≥ 3, 개발 묶음 앞·뒤 기간 모두 차이 > 0.
이탈 사건 뒤 10거래일은 같은 종목의 새 이탈 사건을 세지 않는다. 표준오차는 같은 달 사건을 한 덩어리로 묶는다.
"""
import math
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .frame_backtest import FRAME_A_CONFIG, compare_clustered
from .level_backtest import LEVEL_BT_CONFIG, level_arrays, stop_baseline, support_events
from .rule_backtest import _bt_naive
from .strategy import LEVEL_CONFIG, short_indicators

RECLAIM_BT_CONFIG: Dict[str, Any] = {
    "START": "2001-01-01",
    "DATA_START": "1999-01-01",
    "SPLIT_DATE": FRAME_A_CONFIG["SPLIT_DATE"],
    "WINDOW": 10,                  # 이탈 뒤 회복을 기다리는 거래일 수 (LEVELS·단기 규칙의 10거래일과 같음)
    "MIN_BARS": 260,
    "DEV_UNIVERSE": list(FRAME_A_CONFIG["DEV_UNIVERSE"]),
    # 판정 묶음(HOLDOUT, 2026-11-17 결과 보기 전 고정): 지금까지 어떤 백테스트(단기 규칙·지지·손절선 시험 포함)에도 쓰지 않은 종목.
    # 미국 11개 섹터 123 · 한국 코스피 25.
    "HOLDOUT_UNIVERSE": [
        "ANET", "CDW", "HPE", "KEYS", "MPWR", "ON", "PTC", "TRMB", "TYL", "ZBRA", "IT", "JBL", "FSLR", "FICO", "BR", "JKHY",
        "NXPI", "MRVL", "SNPS", "FTNT",                                                                    # 기술
        "WBD", "NWSA", "MTCH",                                                                             # 통신·미디어
        "EXPE", "WYNN", "RL", "PHM", "NVR", "GRMN", "MHK", "LKQ", "BWA", "POOL", "DPZ", "WSM", "HLT", "LULU",  # 경기소비재
        "KDP", "BF-B", "BG", "LW",                                                                         # 필수소비재
        "PSX", "FANG", "CTRA", "TRGP",                                                                     # 에너지
        "HBAN", "CFG", "L", "LNC", "PFG", "NDAQ", "MSCI", "WTW", "SYF", "GL", "AIZ", "CBOE", "BRO", "WRB", "FDS", "IVZ",  # 금융
        "COO", "MTD", "STE", "XRAY", "COR", "ELV", "MOH", "INCY", "IQV", "WST", "TFX", "BIO", "CRL", "PODD", "DXCM", "VTRS",  # 헬스케어
        "IR", "TT", "ROL", "VRSK", "XYL", "PNR", "SNA", "WAB", "UAL", "ALK", "ODFL", "HII", "LDOS", "J", "IEX", "NDSN",
        "HWM", "CPRT",                                                                                     # 산업재
        "DD", "LYB", "STLD", "CTVA",                                                                       # 소재
        "ES", "FE", "AEE", "CNP", "AWK", "NI", "NRG", "AES", "ATO", "LNT", "EVRG", "PNW",                  # 유틸리티
        "DLR", "ESS", "MAA", "REG", "UDR", "ARE", "IRM", "EXR", "CCI", "SBAC", "CPT", "FRT",               # 리츠
        "000670.KS", "001740.KS", "002790.KS", "004990.KS", "008930.KS", "014680.KS", "036460.KS", "069960.KS",
        "139480.KS", "001120.KS", "001430.KS", "002350.KS", "004000.KS", "004800.KS", "005300.KS", "005850.KS",
        "006280.KS", "009240.KS", "010120.KS", "011210.KS", "012630.KS", "017800.KS", "026960.KS", "069620.KS",
        "073240.KS",
    ],
}
RECLAIM_CRITERIA: Dict[str, Any] = {"MIN_Z": 3.0, "MIN_EVENTS": 100}
# 결과에 따라 화면(지지·손절선 카드)에 붙일 문구 — 결과 보기 전 고정
RECLAIM_VERDICT_TEXT: Dict[bool, str] = {
    True: "검증됨 — 과거에 지지선을 깼다가 10거래일 안에 다시 올라온 뒤 10거래일 수익이 평소보다 좋았음(RECLAIM.md). 보장은 아님",
    False: "검증 안 됨 — 과거에 지지선을 깼다가 다시 올라온 뒤 수익이 평소보다 좋지 않았음(RECLAIM.md). 회복 여부는 참고 표시만",
}


def _rc_market(sym: str) -> str:
    return "KR" if str(sym).upper().endswith((".KS", ".KQ")) else "US"


def approach_outcomes(si: pd.DataFrame, a: Optional[Dict[str, np.ndarray]] = None, start: Optional[str] = None) -> pd.DataFrame:
    """접근 사건 뒤 WINDOW일: held(지지선 아래 종가 없음) / reclaimed(아래로 갔다가 창 안에서 다시 위) / lost(아래로 가서 창 끝까지 못 돌아옴)."""
    W = RECLAIM_BT_CONFIG["WINDOW"]
    a = a or level_arrays(si)
    ev = support_events(si, a, start or RECLAIM_BT_CONFIG["START"])
    cl, sup = a["Close"], a["support"]
    out = []
    for t in ev["t"].astype(int).to_numpy() if len(ev) else []:
        w = cl[t + 1:t + 1 + W]
        below = w < sup[t]
        if not below.any():
            kind = "held"
        else:
            first = int(np.argmax(below))
            kind = "reclaimed" if (w[first + 1:] > sup[t]).any() else "lost"
        out.append({"date": si.index[t], "kind": kind})
    return pd.DataFrame(out)


def break_events(si: pd.DataFrame, a: Optional[Dict[str, np.ndarray]] = None, start: Optional[str] = None) -> pd.DataFrame:
    """이탈 사건: 종가가 지지선 아래로 처음 내려온 날 t. reclaimed = WINDOW일 안에 종가가 다시 이탈일의 지지선 위로.
    u = 회복일(없으면 −1), days = 걸린 거래일, depth = (지지선 − 회복 전 최저 종가) ÷ ATR(이탈일)."""
    W = RECLAIM_BT_CONFIG["WINDOW"]
    a = a or level_arrays(si)
    cl, sup, atr = a["Close"], a["support"], a["ATR20"]
    n = len(cl)
    s0 = pd.Timestamp(start or RECLAIM_BT_CONFIG["START"])
    with np.errstate(invalid="ignore"):
        below = np.isfinite(sup) & (cl < sup)
        prev_ok = np.zeros(n, dtype=bool)
        prev_ok[1:] = np.isfinite(sup[:-1]) & (cl[:-1] >= sup[:-1])
        ok = below & prev_ok & np.isfinite(atr) & (atr > 0) & (si.index >= s0)
    ok[max(0, n - W):] = False        # 회복 창이 다 보이는 날만
    out, free = [], -1
    for t in np.flatnonzero(ok):
        if t < free:
            continue
        w = cl[t + 1:t + 1 + W]
        back = w > sup[t]
        if back.any():
            k = int(np.argmax(back)) + 1
            low = float(min(cl[t], w[:k - 1].min())) if k > 1 else float(cl[t])
            out.append({"date": si.index[t + k], "break_date": si.index[t], "t": int(t), "u": int(t + k), "reclaimed": 1.0, "days": k,
                        "depth": float((sup[t] - low) / atr[t]), "level": float(sup[t])})
        else:
            low = float(min(cl[t], w.min()))
            out.append({"date": si.index[t], "break_date": si.index[t], "t": int(t), "u": -1, "reclaimed": 0.0, "days": np.nan,
                        "depth": float((sup[t] - low) / atr[t]), "level": float(sup[t])})
        free = t + W + 1
    return pd.DataFrame(out)


def reclaim_events(si: pd.DataFrame, breaks: pd.DataFrame, a: Optional[Dict[str, np.ndarray]] = None) -> pd.DataFrame:
    """판정 사건: 회복일 u. 결과 r10·r20 = u+1 시가 → 10·20거래일 뒤 종가, rebreak = 이후 10거래일 종가가 다시 지지선 아래."""
    a = a or level_arrays(si)
    if breaks is None or not len(breaks):
        return pd.DataFrame()
    b = breaks[breaks["reclaimed"] == 1.0]
    out = []
    for u, lvl in zip(b["u"].astype(int).to_numpy(), b["level"].to_numpy()):
        r10, r20 = a["r_h"][u], a["r_hl"][u]
        if not (np.isfinite(r10) and np.isfinite(a["drop"][u])):
            continue
        out.append({"date": si.index[u], "t": int(u), "drop": float(a["drop"][u]), "r10": float(r10),
                    "r20": float(r20) if np.isfinite(r20) else np.nan,
                    "tail": float(r20 <= LEVEL_BT_CONFIG["TAIL"]) if np.isfinite(r20) else np.nan,
                    "rebreak": float(a["fmin"][u] < lvl) if np.isfinite(a["fmin"][u]) else np.nan})
    return pd.DataFrame(out)


def run_reclaim_universe(price_map: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    c = RECLAIM_BT_CONFIG
    parts: Dict[str, List[pd.DataFrame]] = {k: [] for k in ("app", "brk", "rec", "rnd", "mat")}
    used = []
    for sym, df in (price_map or {}).items():
        d = _bt_naive(df)
        if d is not None and len(d):
            cols = [k for k in ("Open", "High", "Low", "Close") if k in d]
            d = d[(d[cols].astype(float) > 0).all(axis=1)]
        if d is None or len(d) < c["MIN_BARS"]:
            continue
        si = short_indicators(d)
        a = level_arrays(si)
        used.append(sym)
        brk = break_events(si, a)
        rec = reclaim_events(si, brk, a)
        got = {"app": approach_outcomes(si, a), "brk": brk, "rec": rec,
               "rnd": stop_baseline(si, rec, a, matched=False, start=c["START"]), "mat": stop_baseline(si, rec, a, matched=True, start=c["START"])}
        mk = _rc_market(sym)
        for k, x in got.items():
            if len(x):
                parts[k].append(x.assign(ticker=sym, market=mk))
    cat = lambda lst, cols: pd.concat(lst, ignore_index=True) if lst else pd.DataFrame(columns=cols + ["market", "ticker"])
    return {"app": cat(parts["app"], ["date", "kind"]), "brk": cat(parts["brk"], ["date", "break_date", "reclaimed", "days", "depth"]),
            "rec": cat(parts["rec"], ["date", "r10", "r20", "tail", "rebreak"]), "rnd": cat(parts["rnd"], ["date", "r10", "r20", "tail"]),
            "mat": cat(parts["mat"], ["date", "r10", "r20", "tail"]), "tickers": used}


def _rc_block(u: Dict[str, Any], f) -> Dict[str, Any]:
    app, brk, rec, rnd, mat = (f(u[k]) for k in ("app", "brk", "rec", "rnd", "mat"))
    n_app = len(app)
    shares = {k: (float((app["kind"] == k).mean()) if n_app else None) for k in ("held", "reclaimed", "lost")}
    rc = brk[brk["reclaimed"] == 1.0] if len(brk) else brk
    desc = {"n_approach": n_app, "approach": shares, "n_break": int(len(brk)),
            "reclaim_rate": float(brk["reclaimed"].mean()) if len(brk) else None,
            "days_median": float(rc["days"].median()) if len(rc) else None,
            "depth_median_reclaimed": float(rc["depth"].median()) if len(rc) else None,
            "depth_median_lost": float(brk.loc[brk["reclaimed"] == 0.0, "depth"].median()) if len(brk) and (brk["reclaimed"] == 0.0).any() else None,
            "rebreak_rate": float(rec["rebreak"].mean()) if len(rec) else None}
    return {"desc": desc, "vs_random": compare_clustered(rec, rnd, "r10"), "vs_matched": compare_clustered(rec, mat, "r10"),
            "r20_vs_random": compare_clustered(rec, rnd, "r20")}


def evaluate_reclaim(dev_map: Dict[str, pd.DataFrame], holdout_map: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    c, crit = RECLAIM_BT_CONFIG, RECLAIM_CRITERIA
    split_t = pd.Timestamp(c["SPLIT_DATE"])
    res: Dict[str, Any] = {"version": LEVEL_CONFIG["VERSION"], "criteria": crit, "config": {k: c[k] for k in ("START", "SPLIT_DATE", "WINDOW")},
                           "universes": {}, "verdict": {}}
    raw = {"개발(대표 종목)": run_reclaim_universe(dev_map), "HOLDOUT(새 종목)": run_reclaim_universe(holdout_map)}
    per = {"전체 기간": lambda x: x, "앞 기간": lambda x: x[x["date"] < split_t], "뒤 기간": lambda x: x[x["date"] >= split_t]}
    for uname, u in raw.items():
        blocks = {p: _rc_block(u, f) for p, f in per.items()}
        by_mk = {mk: _rc_block(u, lambda x, mk=mk: x[x["market"] == mk]) for mk in ("US", "KR")
                 if len(u["brk"]) and (u["brk"]["market"] == mk).any()}
        res["universes"][uname] = {"n_tickers": len(u["tickers"]), "blocks": blocks, "by_market": by_mk}
    ho = res["universes"]["HOLDOUT(새 종목)"]["blocks"]["전체 기간"]["vs_random"]
    dv = res["universes"]["개발(대표 종목)"]["blocks"]
    n = ho["actual"].get("n", 0)
    ok_ho = bool(n >= crit["MIN_EVENTS"] and ho["diff"] is not None and ho["diff"] > 0 and ho["z"] is not None and ho["z"] >= crit["MIN_Z"])
    ok_dev = all(dv[p]["vs_random"]["diff"] is not None and dv[p]["vs_random"]["diff"] > 0 for p in ("앞 기간", "뒤 기간"))
    res["verdict"] = {"passed": ok_ho and ok_dev, "holdout_pass": ok_ho, "dev_both_positive": ok_dev, "enough_events": n >= crit["MIN_EVENTS"],
                      "z2_reference": bool(n >= crit["MIN_EVENTS"] and ho["z"] is not None and ho["z"] >= 2.0),
                      "text": RECLAIM_VERDICT_TEXT[ok_ho and ok_dev],
                      "criteria": (f"HOLDOUT 전체 기간 사건 {crit['MIN_EVENTS']}건 이상, (회복 뒤 {c['WINDOW']}거래일 수익률 − 무작위 날 수익률) > 0 이고 "
                                   f"z ≥ {crit['MIN_Z']:.0f}, 그리고 개발 묶음 앞·뒤 기간 모두 차이 > 0")}
    return {"summary": res}


def _rc_pct(v: Optional[float], d: int = 1) -> str:
    return "—" if v is None or (isinstance(v, float) and not math.isfinite(v)) else f"{v * 100:.{d}f}%"


def _rc_ret(s: Dict[str, Any]) -> str:
    if not s.get("n"):
        return "표본 없음"
    se = f" ±{s['se'] * 100:.2f}" if s.get("se") is not None else ""
    return f"{s['mean'] * 100:+.2f}%{se} ({s['n']:,}건)"


def _rc_num(v: Optional[float], d: int = 1) -> str:
    return "—" if v is None or (isinstance(v, float) and not math.isfinite(v)) else f"{v:.{d}f}"


def render_reclaim_report(r: Dict[str, Any], generated_at: Optional[str] = None) -> str:
    v, W = r["verdict"], r["config"]["WINDOW"]
    L = ["# 지지선 이탈 뒤 회복 검증", "",
         f"- 생성: {generated_at or '—'} · {r['version']} · 사건 {r['config']['START'][:4]}년~",
         "- 가설(사용자, 2026-11-17): 지지선에서 바로 지지받기보다 한 번 깼다가 다시 올라오는 경우가 많다. LEVELS.md 결과 뒤에 나온 새 질문이라 "
         "새 종목 묶음(미국 123·한국 25, 결과 보기 전 고정)과 미리 고정한 기준으로 시험했다. 가격 기반 열 번째 질문, z 기준 3.",
         "- 지지선 = 그날까지 확정된 가장 최근 스윙 저점(바꾸지 않음). 이탈 = 종가가 지지선 아래로 처음 내려옴, 회복 = 그 뒤 "
         f"{W}거래일 안에 종가가 다시 그 지지선 위로.",
         f"- 판정: 회복한 날 다음 날 시가에 사서 {W}거래일 뒤 종가까지의 수익률이 같은 종목 무작위 날보다 좋은가. 표준오차는 월 묶음.",
         f"- 판정 기준: {v['criteria']}.", "",
         f"## 판정: {'통과' if v['passed'] else '미통과'}", "",
         f"- HOLDOUT {'통과' if v['holdout_pass'] else '미통과'}(참고 z ≥ 2 {'충족' if v['z2_reference'] else '미충족'}) · 개발 묶음 앞·뒤 기간 "
         f"{'모두 > 0' if v['dev_both_positive'] else '하나 이상 ≤ 0'} · 사건 수 {'충분' if v['enough_events'] else '부족'}",
         f"- 화면 문구: {v['text']}", ""]
    for uname, u in r["universes"].items():
        rows = list(u["blocks"].items()) + [(f"전체 기간 · {'미국' if mk == 'US' else '한국'}", b) for mk, b in u["by_market"].items()]
        L += [f"## {uname} — 종목 {u['n_tickers']}개", "",
              f"**A. 얼마나 자주 깨고 회복하나** (판정 없는 집계, {W}거래일 기준)", "",
              "| 기간 | 지지선 접근 뒤: 버팀 | 깼다가 회복 | 깨고 못 돌아옴 | (접근 사건) | 이탈 뒤 회복 비율 | (이탈 사건) | 회복까지 중앙 일수 | "
              "가장 깊이 내려간 폭 중앙(회복 · 미회복, ATR) | 회복 뒤 다시 이탈 |", "|---|---|---|---|---|---|---|---|---|---|"]
        for per, b in rows:
            d = b["desc"]
            ap = d["approach"]
            L.append(f"| {per} | {_rc_pct(ap['held'])} | {_rc_pct(ap['reclaimed'])} | {_rc_pct(ap['lost'])} | {d['n_approach']:,}건 | "
                     f"{_rc_pct(d['reclaim_rate'])} | {d['n_break']:,}건 | {_rc_num(d['days_median'], 0)}일 | "
                     f"{_rc_num(d['depth_median_reclaimed'])} · {_rc_num(d['depth_median_lost'])} | {_rc_pct(d['rebreak_rate'])} |")
        L += ["", f"**B. 회복 다음 날 시가 → {W}거래일 뒤 종가 수익률**", "",
              "| 기간 | 회복 뒤 | 무작위 날(판정) | 차이 | z | 하락 폭 맞춘 날(참고) 차이 · z |", "|---|---|---|---|---|---|"]
        for per, b in rows:
            x, m = b["vs_random"], b["vs_matched"]
            dd = "—" if x.get("diff") is None else f"{x['diff'] * 100:+.2f}%p"
            zz = "—" if x.get("z") is None else f"{x['z']:+.1f}"
            ref = "—" if m.get("diff") is None else f"{m['diff'] * 100:+.2f}%p · {'—' if m.get('z') is None else format(m['z'], '+.1f')}"
            L.append(f"| {per} | {_rc_ret(x['actual'])} | {_rc_ret(x['baseline'])} | {dd} | {zz} | {ref} |")
        r20 = u["blocks"]["전체 기간"]["r20_vs_random"]
        L += ["", f"참고 — 20거래일 뒤: {_rc_ret(r20['actual'])} vs 무작위 날 {_rc_ret(r20['baseline'])} "
                  f"(z {'—' if r20.get('z') is None else format(r20['z'], '+.1f')})", ""]
    L += ["## 읽는 법", "",
          "- A는 '이런 일이 얼마나 흔한가'이고 B는 '그때 사면 평소보다 나은가'이다. 흔하다고 해서 사서 이득이라는 뜻은 아니다 — B가 통과해야 신호로 쓸 수 있다.",
          "- 결과를 보고 정의(10거래일·종가 기준)나 기준을 바꿔 다시 시험하지 않는다.", ""]
    return "\n".join(L)
