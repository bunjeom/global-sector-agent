"""피어 비교 v1 (2026-11-15) — 딥다이브 개편 1단계: 피어 묶음·요인 지표·전략별 순위(가치 관점·추세 관점).

규칙은 AGENTS.md 4번 '투자 방향과 딥다이브 개편 기획'에 사전 등록했다. 결과를 보고 가중치·기준을 바꾸지 않는다.
- 지표마다 묶음 안 백분위(0~100, 좋은 쪽이 높게), 요인 점수 = 지표 백분위 평균.
- 전략 점수 = 핵심 요인 50 + 보조 요인 25 + 25 (보조가 하나 없으면 있는 것끼리 비율 유지 재정규화).
- 생존성 하드 킬은 순위에서 제외(걸러내기), 경고는 표시만.
- 전략 점수가 있는 종목이 5개 미만이면 '피어 부족' — 순위를 매기지 않는다.

데이터 갱신: 재무·추정치·밴드·생존성은 주 1회(FUND_REFRESH_DAYS), 가격 흐름(상대강도·추세 상태)은 매일.
기록: data/peers/peer_fund_YYYY.csv(주간 재무), data/peers/peer_rank_YYYY.csv(매일 순위). 손으로 고치지 않는다.
피어 묶음 수정: data/peers/peer_map.csv (ticker,group,group_name,note) — group이 '-'이면 자동 피어에서 뺀다.

engine.py를 기능별로 나눈 모듈. 바깥에서는 `import engine` 후 `engine.이름`으로 쓴다(engine/__init__.py).
"""
import math
import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .common import NO_DATA
from .earnings import _stmt_series
from .fundamental import SURVIVAL_CONFIG, _us_liquidity_and_survival, evaluate_survival
from .home import NARRATIVE_BASKETS_SPEC
from .playbook import STATE_DOWN, STATE_SIDE, STATE_UP, _pb_market, light_band_z, trend_indicators
from .snapshot import _cell, _f, snapshot_universe

PEER_CONFIG: Dict[str, Any] = {
    "VERSION": "피어 비교 v1.2 (2026-11-16)",
    "DIR": os.path.join("data", "peers"),
    "MAP_FILE": "peer_map.csv",
    "TIMEZONE": "Asia/Seoul",
    "MIN_MEMBERS": 5,              # 전략 점수가 있는 종목이 이보다 적으면 순위 없음
    "AUTO_PEERS_CAP": 8,           # 야후 산업 상위 기업에서 가져올 자동 피어 수(커버리지 종목은 별도로 모두 포함)
    "ETF_HOLDINGS_TOP": 10,        # 커버리지: 섹터 ETF 상위 구성 종목 수
    "FUND_REFRESH_DAYS": 7,        # 재무·추정치 갱신 주기(일)
    "PRICE_PERIOD": "2y",          # 추세 상태(200일선·스윙)에 2년 필요
    "PAUSE_SEC": 0.3,
    "CORE_W": 0.5,
    "AUX_W": 0.25,
    "TOP_FRACTION": 1.0 / 3.0,     # 전략 선택에 쓰는 '상위 1/3'
    "STRATEGIES": {
        "value": {"label": "가치 관점", "core": "valuation", "aux": ["profitability", "revisions"]},
        "trend": {"label": "추세 관점", "core": "price", "aux": ["revisions", "growth"]},
    },
    "SECTOR_ETF": {
        "Technology": "XLK", "Communication Services": "XLC", "Financial Services": "XLF", "Healthcare": "XLV",
        "Consumer Cyclical": "XLY", "Consumer Defensive": "XLP", "Industrials": "XLI", "Energy": "XLE",
        "Utilities": "XLU", "Basic Materials": "XLB", "Real Estate": "XLRE",
    },
    "BENCH_FALLBACK": {"US": "SPY", "KR": "^KS11"},
}

FACTOR_LABELS: Dict[str, str] = {"growth": "성장", "profitability": "수익성", "revisions": "실적 모멘텀",
                                 "valuation": "밸류에이션", "price": "가격 흐름"}
# (지표 키, 요인, 방향 +1 = 클수록 좋음 / -1 = 작을수록 좋음, 화면 이름)
PEER_METRICS: List[Tuple[str, str, int, str]] = [
    ("rev_growth_ttm", "growth", 1, "매출 성장률(최근 분기, 전년 동기 대비)"),
    ("eps_growth_fwd", "growth", 1, "선행 EPS 성장(FY1→FY2)"),
    ("op_margin_ttm", "profitability", 1, "영업이익률(최근 4분기, 없으면 야후 값)"),
    ("roe", "profitability", 1, "ROE"),
    ("rev_net_30d", "revisions", 1, "30일 추정치 순상향 비율((상향−하향)÷FY1 애널리스트 수)"),
    ("fy1_change_30d", "revisions", 1, "FY1 추정치 30일 변화"),
    ("fwd_pe", "valuation", -1, "선행 PER"),
    ("peg", "valuation", -1, "PEG(선행 PER ÷ 선행 EPS 성장%)"),
    ("band_z", "valuation", -1, "자기 PER 밴드 z"),
    ("rs63", "price", 1, "섹터 ETF 대비 63일 상대강도"),
    ("state_pt", "price", 1, "틀 A 추세 상태(상승 2·횡보 1·하락 0)"),
]
STATE_POINTS: Dict[str, int] = {STATE_UP: 2, STATE_SIDE: 1, STATE_DOWN: 0}

FUND_COLUMNS: List[str] = [
    "fund_date", "ticker", "market", "group", "group_name", "group_source", "is_coverage", "industry_key", "industry",
    "sector", "currency", "price", "fwd_pe_info", "eps_fy1", "eps_fy2", "n_fy1", "up_30d", "down_30d", "fy1_30d_ago",
    "rev_ttm", "rev_ttm_prev", "op_income_ttm", "roe", "band_z", "band_reason",
    "surv_kill", "surv_warning", "surv_applicable", "surv_text", "next_earnings_date", "notes", "version",
    # v1.1 (새 열은 맨 뒤): 야후 분기 손익은 보통 4~5분기라 최근 4분기 합 대 그 전 4분기 합을 못 만든다 → 최근 분기 대 전년 동기
    "rev_q0", "rev_q4", "rev_growth_info", "op_margin_info",
    "short_name",   # 화면 표시용 회사 이름(야후 shortName)
]
RANK_COLUMNS: List[str] = (
    ["rank_date", "group", "group_name", "group_source", "ticker", "is_coverage", "n_group", "fund_date", "price_as_of",
     "trend_state"]
    + [m[0] for m in PEER_METRICS]
    + [f"f_{k}" for k in FACTOR_LABELS]
    + ["value_score", "value_rank", "value_n", "value_top", "value_note",
       "trend_score", "trend_rank", "trend_n", "trend_top", "trend_note",
       "surv_kill", "surv_warning", "notes", "version"]
)


# ------------------------------------------------------------------ 순위 계산 (순수 함수, 테스트 대상)
def derive_metrics(fund: Dict[str, Any], price: Optional[Dict[str, Any]] = None) -> Dict[str, Optional[float]]:
    """주간 재무 기록 한 줄 + 오늘 가격 지표 → 비교 지표 11개. 정의가 안 되는 값(적자 PER, 성장 ≤ 0인 PEG 등)은 None."""
    g = lambda k: _f(fund.get(k))
    out: Dict[str, Optional[float]] = {m[0]: None for m in PEER_METRICS}
    rev, opi = g("rev_ttm"), g("op_income_ttm")
    q0, q4 = g("rev_q0"), g("rev_q4")
    # 매출 성장률 = 최근 분기 ÷ 전년 동기 − 1 (분기 자료가 5개 미만이면 야후 revenueGrowth — 같은 정의)
    if q0 is not None and q4 is not None and q4 > 0:
        out["rev_growth_ttm"] = q0 / q4 - 1.0
    else:
        out["rev_growth_ttm"] = g("rev_growth_info")
    if rev is not None and rev > 0 and opi is not None:
        out["op_margin_ttm"] = opi / rev
    else:
        out["op_margin_ttm"] = g("op_margin_info")
    out["roe"] = g("roe")
    e1, e2 = g("eps_fy1"), g("eps_fy2")
    if e1 is not None and e2 is not None and e1 > 0 and e2 > 0:
        out["eps_growth_fwd"] = e2 / e1 - 1.0
    up, dn, n = g("up_30d"), g("down_30d"), g("n_fy1")
    if up is not None and dn is not None and n is not None and n > 0:
        out["rev_net_30d"] = (up - dn) / n
    e30 = g("fy1_30d_ago")
    if e1 is not None and e30 is not None and e30 > 0:
        out["fy1_change_30d"] = e1 / e30 - 1.0
    px = _f((price or {}).get("price")) or g("price")
    # v1.2: 선행 PER = 오늘 가격 ÷ FY1(이번 회계연도) 컨센서스 EPS — 모든 종목을 같은 기준으로.
    # 야후 forwardPE는 EPS 기준 연도가 회사·시기마다 달라 FY1 EPS가 없을 때만 쓴다. FY1 EPS가 적자면 PER 없음.
    if e1 is not None:
        pe = px / e1 if (px is not None and e1 > 0) else None
    else:
        pe = g("fwd_pe_info")
    out["fwd_pe"] = pe if (pe is not None and pe > 0) else None
    eg = out["eps_growth_fwd"]
    if out["fwd_pe"] is not None and eg is not None and eg > 0:
        out["peg"] = out["fwd_pe"] / (eg * 100.0)
    out["band_z"] = g("band_z")
    out["rs63"] = _f((price or {}).get("rs63"))
    st = (price or {}).get("trend_state")
    out["state_pt"] = float(STATE_POINTS[st]) if st in STATE_POINTS else None
    return out


def percentile_ranks(values: pd.Series, direction: int = 1) -> pd.Series:
    """묶음 안 백분위(0~100). 좋은 쪽(direction 방향)이 100. 같은 값은 평균 순위. 값이 2개 미만이면 모두 NaN."""
    v = pd.to_numeric(values, errors="coerce").astype(float)
    m = v.notna()
    out = pd.Series(np.nan, index=v.index, dtype=float)
    n = int(m.sum())
    if n < 2:
        return out
    r = (v[m] * float(direction)).rank(method="average")
    out[m] = (r - 1.0) / (n - 1.0) * 100.0
    return out


def strategy_score(factors: Dict[str, Optional[float]], strategy: str) -> Tuple[Optional[float], str]:
    """요인 점수 → 전략 점수(0~100). 핵심 요인이 없거나 요인이 2개 미만이면 None. 반환 (점수, 메모)."""
    cfg = PEER_CONFIG["STRATEGIES"][strategy]
    core, aux = cfg["core"], cfg["aux"]
    def fv(k: str) -> Optional[float]:
        v = factors.get(k)
        return None if v is None or (isinstance(v, float) and math.isnan(v)) else float(v)
    if fv(core) is None:
        return None, f"핵심 요인({FACTOR_LABELS[core]}) 자료 없음"
    parts = [(core, PEER_CONFIG["CORE_W"])] + [(a, PEER_CONFIG["AUX_W"]) for a in aux if fv(a) is not None]
    if len(parts) < 2:
        return None, "요인이 2개 미만"
    wsum = sum(w for _, w in parts)
    score = sum(fv(k) * w for k, w in parts) / wsum
    missing = [FACTOR_LABELS[a] for a in aux if fv(a) is None]
    return float(score), ("자료 부족: " + ", ".join(missing)) if missing else ""


def score_group(df: pd.DataFrame) -> pd.DataFrame:
    """한 피어 묶음(행 = 종목, 열 = derive_metrics 지표 + surv_kill)의 요인 점수·전략 점수·순위.
    생존성 하드 킬 종목은 백분위 계산에서 빼고 '순위 제외'로 남긴다."""
    out = df.copy()
    sk = out["surv_kill"] if "surv_kill" in out else pd.Series(False, index=out.index)
    killed = pd.Series([v is True or str(v).strip().lower() == "true" for v in sk], index=out.index, dtype=bool)
    base = out[~killed]
    pct = pd.DataFrame(index=out.index)
    for key, factor, direction, _ in PEER_METRICS:
        col = pd.Series(np.nan, index=out.index, dtype=float)
        if key in base:
            col.loc[base.index] = percentile_ranks(base[key], direction)
        pct[key] = col
    for fk in FACTOR_LABELS:
        keys = [m[0] for m in PEER_METRICS if m[1] == fk]
        out[f"f_{fk}"] = pct[keys].mean(axis=1, skipna=True)
    for sname in PEER_CONFIG["STRATEGIES"]:
        scores, notes = [], []
        for idx, r in out.iterrows():
            if killed.loc[idx]:
                scores.append(np.nan)
                notes.append("생존성 하드 킬 — 순위 제외")
                continue
            sc, nt = strategy_score({fk: (None if pd.isna(r[f"f_{fk}"]) else float(r[f"f_{fk}"])) for fk in FACTOR_LABELS}, sname)
            scores.append(np.nan if sc is None else sc)
            notes.append(nt)
        s = pd.Series(scores, index=out.index, dtype=float)
        n_ok = int(s.notna().sum())
        out[f"{sname}_score"] = s
        out[f"{sname}_n"] = n_ok
        if n_ok < PEER_CONFIG["MIN_MEMBERS"]:
            out[f"{sname}_rank"] = np.nan
            out[f"{sname}_top"] = pd.Series([None] * len(out), index=out.index, dtype=object)
            notes = [(nt + " · " if nt else "") + f"피어 부족(점수 있는 종목 {n_ok}개 < {PEER_CONFIG['MIN_MEMBERS']})"
                     if not killed.loc[i] else nt for nt, i in zip(notes, out.index)]
        else:
            rk = s.rank(ascending=False, method="min")
            cut = math.ceil(n_ok * PEER_CONFIG["TOP_FRACTION"])
            out[f"{sname}_rank"] = rk
            out[f"{sname}_top"] = pd.Series([None if pd.isna(x) else bool(x <= cut) for x in rk], index=out.index, dtype=object)
        out[f"{sname}_note"] = pd.Series(notes, index=out.index, dtype=object)
    return out


# ------------------------------------------------------------------ 피어 묶음
def peer_dir(base_dir: Optional[str] = None) -> str:
    return base_dir or PEER_CONFIG["DIR"]


def load_peer_map(base_dir: Optional[str] = None) -> Dict[str, Dict[str, str]]:
    """data/peers/peer_map.csv → {티커: {group, group_name, note}}. 없으면 빈 dict. '#'로 시작하는 줄은 설명."""
    p = os.path.join(peer_dir(base_dir), PEER_CONFIG["MAP_FILE"])
    if not os.path.exists(p):
        return {}
    df = pd.read_csv(p, dtype=str, encoding="utf-8-sig", comment="#").fillna("")
    out: Dict[str, Dict[str, str]] = {}
    for _, r in df.iterrows():
        t = str(r.get("ticker", "")).strip().upper()
        g = str(r.get("group", "")).strip()
        if t and g:
            out[t] = {"group": g, "group_name": str(r.get("group_name", "")).strip() or g, "note": str(r.get("note", "")).strip()}
    return out


def _norm_ticker(t: str) -> str:
    t = str(t).strip().upper()
    return t if t.endswith((".KS", ".KQ")) else t.replace(".", "-")


def coverage_universe(holdings_fn: Optional[Callable[[str], Optional[Dict[str, Any]]]] = None,
                      snapshot_dir: Optional[str] = None) -> List[str]:
    """커버리지 = 장부 대상(대표 종목 + watchlist.txt) ∪ 섹터·테마 ETF 상위 구성 종목. 순서 유지·중복 제거."""
    if holdings_fn is None:
        from .radar import fetch_etf_top_holdings as holdings_fn
    out = [_norm_ticker(t) for t in snapshot_universe(snapshot_dir)]
    for b in NARRATIVE_BASKETS_SPEC:
        etf = b.get("ticker")
        if not etf or b.get("asset_class") == "crypto":
            continue
        try:
            h = holdings_fn(etf) or {}
        except Exception:
            h = {}
        for r in (h.get("holdings") or [])[:PEER_CONFIG["ETF_HOLDINGS_TOP"]]:
            if r.get("ticker"):
                out.append(_norm_ticker(r["ticker"]))
    return list(dict.fromkeys(out))


def build_groups(coverage: List[str], info_map: Dict[str, Dict[str, Any]],
                 industry_members: Callable[[str], List[str]], peer_map: Dict[str, Dict[str, str]]) -> Dict[str, Any]:
    """피어 묶음 만들기. 반환 {"groups": {key: {name, source, members}}, "no_group": {티커: 이유}, "group_of": {티커: key}}.
    - peer_map에 있는 티커: 그 묶음(수동). group '-'는 자동 피어에서 제외.
    - 그 밖의 미국 커버리지 종목: 야후 산업(industryKey) 묶음 = 같은 산업 커버리지 종목 + 산업 상위 기업(AUTO_PEERS_CAP).
    - 한국 종목은 peer_map에 없으면 묶음 없음(한국 피어 목록은 수동)."""
    groups: Dict[str, Dict[str, Any]] = {}
    no_group: Dict[str, str] = {}
    mapped = {t: m for t, m in peer_map.items()}
    for t, m in mapped.items():
        if m["group"] == "-":
            continue
        g = groups.setdefault(m["group"], {"name": m["group_name"], "source": "수동", "members": []})
        g["members"].append(t)
    for t in coverage:
        if t in mapped:
            if mapped[t]["group"] == "-":
                no_group[t] = "peer_map에서 제외(-)"
            continue
        if _pb_market(t) == "KR":
            no_group[t] = "한국 종목 — peer_map.csv에 피어 목록 추가 필요"
            continue
        inf = info_map.get(t) or {}
        key = inf.get("industryKey")
        if not key:
            no_group[t] = "야후 산업 분류 없음"
            continue
        gk = f"yf:{key}"
        g = groups.setdefault(gk, {"name": inf.get("industry") or key, "source": "야후 산업", "members": [], "_key": key})
        g["members"].append(t)
    for gk, g in groups.items():
        if g["source"] != "야후 산업":
            continue
        try:
            extra = [_norm_ticker(s) for s in (industry_members(g["_key"]) or [])]
        except Exception:
            extra = []
        extra = [s for s in extra if s not in mapped and s not in g["members"]][:PEER_CONFIG["AUTO_PEERS_CAP"]]
        g["members"].extend(extra)
    for g in groups.values():
        g["members"] = list(dict.fromkeys(g["members"]))
        g.pop("_key", None)
    group_of = {t: gk for gk, g in groups.items() for t in g["members"]}
    return {"groups": groups, "no_group": no_group, "group_of": group_of}


# ------------------------------------------------------------------ 데이터 받기 (yfinance, 주 1회)
def _yf_industry_members(yf: Any) -> Callable[[str], List[str]]:
    def f(key: str) -> List[str]:
        df = yf.Industry(key).top_companies
        return [str(s) for s in list(df.index)] if df is not None and len(df) else []
    return f


def fetch_fund_row(ticker: str, yf: Any, date: str, group_key: str, group: Dict[str, Any], is_coverage: bool,
                   info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """한 종목의 주간 재무 기록. 항목마다 따로 시도하고 실패는 None + notes."""
    row: Dict[str, Any] = {c: None for c in FUND_COLUMNS}
    row.update(fund_date=date, ticker=ticker, market=_pb_market(ticker), group=group_key, group_name=group["name"],
               group_source=group["source"], is_coverage=bool(is_coverage), version=PEER_CONFIG["VERSION"])
    notes: List[str] = []
    tk = yf.Ticker(ticker)
    if info is None:
        try:
            info = tk.info or {}
        except Exception:
            info = {}
            notes.append("기본 정보")
    row.update(industry_key=info.get("industryKey"), industry=info.get("industry"), sector=info.get("sector"),
               currency=info.get("currency"), fwd_pe_info=_f(info.get("forwardPE")), roe=_f(info.get("returnOnEquity")),
               rev_growth_info=_f(info.get("revenueGrowth")), op_margin_info=_f(info.get("operatingMargins")),
               short_name=info.get("shortName") or info.get("longName"),
               price=_f(info.get("currentPrice")) or _f(info.get("regularMarketPrice")))

    def table(attr: str, label: str) -> Any:
        try:
            return getattr(tk, attr)
        except Exception:
            notes.append(label)
            return None
    ee, et, er = table("earnings_estimate", "EPS 추정치"), table("eps_trend", "EPS 추이"), table("eps_revisions", "EPS 수정 건수")
    row["eps_fy1"], row["eps_fy2"], row["n_fy1"] = _cell(ee, "0y", "avg"), _cell(ee, "+1y", "avg"), _cell(ee, "0y", "numberOfAnalysts")
    if row["eps_fy1"] is None:
        row["eps_fy1"] = _cell(et, "0y", "current")
    if row["eps_fy2"] is None:
        row["eps_fy2"] = _cell(et, "+1y", "current")
    row["fy1_30d_ago"] = _cell(et, "0y", "30daysAgo")
    row["up_30d"], row["down_30d"] = _cell(er, "0y", "upLast30days"), _cell(er, "0y", "downLast30days")
    qi = table("quarterly_income_stmt", "분기 손익계산서")
    rev = _stmt_series(qi, ["Total Revenue", "Operating Revenue"])
    opi = _stmt_series(qi, ["Operating Income"])
    if len(rev) >= 8 and all(v is not None for v in rev[:8]):
        row["rev_ttm"], row["rev_ttm_prev"] = float(sum(rev[:4])), float(sum(rev[4:8]))
    elif len(rev) >= 4 and all(v is not None for v in rev[:4]):
        row["rev_ttm"] = float(sum(rev[:4]))
    if len(rev) >= 5 and rev[0] is not None and rev[4] is not None:
        row["rev_q0"], row["rev_q4"] = float(rev[0]), float(rev[4])
    if len(opi) >= 4 and all(v is not None for v in opi[:4]):
        row["op_income_ttm"] = float(sum(opi[:4]))
    try:
        cal = tk.calendar
        ed = (cal or {}).get("Earnings Date") if isinstance(cal, dict) else None
        if ed is not None:
            ds = [pd.Timestamp(x) for x in (ed if isinstance(ed, (list, tuple)) else [ed]) if x is not None]
            if ds:
                row["next_earnings_date"] = min(ds).strftime("%Y-%m-%d")
    except Exception:
        notes.append("실적 발표일")
    if row["market"] == "US":
        b = light_band_z(ticker, yf=yf)
        row["band_z"], row["band_reason"] = b.get("z"), b.get("reason")
        meta: Dict[str, Any] = {"notes": []}
        sv = _us_liquidity_and_survival(tk, meta)
        is_fin = str(info.get("sector") or "").strip().lower() == "financial services"
        s = evaluate_survival(ocf=sv.get("ocf_ttm_krw"), fcf=sv.get("fcf_ttm_krw"),
                              liquid=sv.get("liquid_krw") if sv.get("liquid_krw") is not None else
                              (sv.get("net_cash_krw") if (sv.get("net_cash_krw") or 0) > 0 else None),
                              current_debt=sv.get("current_debt_krw"), dilution=sv.get("share_dilution_yoy"),
                              is_financial=is_fin, capex=sv.get("capex_ttm_krw"), debt_change=sv.get("debt_change_1y_krw"),
                              captive_finance=SURVIVAL_CONFIG["CAPTIVE_FINANCE"].get(ticker.upper()))
        row["surv_kill"], row["surv_warning"], row["surv_applicable"] = bool(s["kill"]), bool(s["warning"]), bool(s["applicable"])
        row["surv_text"] = " / ".join(s["kill_reasons"] or s["warn_reasons"]) or ("해당 없음" if not s["applicable"] else "양호")
    else:
        row["band_reason"] = "한국 종목은 DART 연간 EPS가 필요(딥다이브 V 트랙 값 사용)"
        row["surv_kill"], row["surv_warning"], row["surv_applicable"] = False, False, False
        row["surv_text"] = "한국 종목 생존성은 딥다이브 V 트랙(DART)에서"
    row["notes"] = ", ".join(notes)
    return row


def _download_many(yf: Any, syms: List[str], period: str) -> Dict[str, pd.DataFrame]:
    """여러 종목 일봉을 50개씩 묶어 받는다. 묶음 받기가 실패하면 하나씩."""
    out: Dict[str, pd.DataFrame] = {}
    for i in range(0, len(syms), 50):
        chunk = syms[i:i + 50]
        try:
            df = yf.download(chunk, period=period, auto_adjust=False, actions=False, progress=False, threads=True, group_by="ticker")
        except Exception:
            df = None
        for s in chunk:
            one = None
            try:
                if df is not None and len(df) and isinstance(df.columns, pd.MultiIndex) and s in df.columns.get_level_values(0):
                    one = df[s].copy()
                elif df is not None and len(df) and len(chunk) == 1:
                    one = df.copy()
                    if isinstance(one.columns, pd.MultiIndex):
                        one.columns = one.columns.get_level_values(-1)
            except Exception:
                one = None
            if one is None or "Close" not in one or one["Close"].dropna().empty:
                try:
                    one = yf.download(s, period=period, auto_adjust=False, actions=False, progress=False, threads=False)
                    if one is not None and isinstance(one.columns, pd.MultiIndex):
                        one.columns = one.columns.get_level_values(0)
                except Exception:
                    one = None
            if one is not None and len(one) and "Close" in one:
                one = one.dropna(subset=["Close"])
                if len(one):
                    out[s] = one
    return out


def price_metrics(px: Optional[pd.DataFrame], bench: Optional[pd.DataFrame]) -> Dict[str, Any]:
    """오늘 가격 지표: 마지막 종가·날짜, 벤치마크(섹터 ETF) 대비 63일 상대강도, 틀 A 추세 상태."""
    if px is None or not len(px):
        return {"price": None, "price_as_of": None, "rs63": None, "trend_state": None}
    ind = trend_indicators(px, bench)
    last = ind.iloc[-1]
    st = last.get("state")
    return {"price": _f(last.get("Close")), "price_as_of": str(ind.index[-1])[:10],
            "rs63": _f(last.get("rs63")), "trend_state": st if st in STATE_POINTS else None}


# ------------------------------------------------------------------ 기록 파일
def fund_path(date: str, base_dir: Optional[str] = None) -> str:
    return os.path.join(peer_dir(base_dir), f"peer_fund_{str(date)[:4]}.csv")


def rank_path(date: str, base_dir: Optional[str] = None) -> str:
    return os.path.join(peer_dir(base_dir), f"peer_rank_{str(date)[:4]}.csv")


def _load_many(base_dir: Optional[str], prefix: str, columns: List[str], date_cols: List[str]) -> pd.DataFrame:
    d = peer_dir(base_dir)
    files = sorted(f for f in (os.listdir(d) if os.path.isdir(d) else []) if f.startswith(prefix) and f.endswith(".csv"))
    if not files:
        return pd.DataFrame(columns=columns)
    df = pd.concat([pd.read_csv(os.path.join(d, f), encoding="utf-8-sig", dtype={"ticker": str, "group": str}) for f in files],
                   ignore_index=True)
    df = df.reindex(columns=columns + [c for c in df.columns if c not in columns])
    for c in date_cols:
        df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


REFRESH_LOG_COLUMNS: List[str] = ["refresh_date", "mode", "n_coverage", "n_groups", "n_members", "fund_saved", "n_failed", "version"]


def refresh_log_path(base_dir: Optional[str] = None) -> str:
    return os.path.join(peer_dir(base_dir), "peer_refresh_log.csv")


def last_full_refresh(base_dir: Optional[str] = None) -> Optional[pd.Timestamp]:
    """마지막 '전체' 재무 갱신 날짜. 종목을 지정한 빠른 점검(부분 갱신)은 세지 않는다 —
    부분 갱신만 있으면 다음 매일 실행이 전체 갱신을 한다(커버리지가 일부 종목으로 굳지 않게)."""
    p = refresh_log_path(base_dir)
    if not os.path.exists(p):
        return None
    d = pd.read_csv(p, dtype=str, encoding="utf-8-sig")
    d = d[d["mode"] == "full"]
    return pd.to_datetime(d["refresh_date"], errors="coerce").max() if len(d) else None


def load_peer_fund(base_dir: Optional[str] = None) -> pd.DataFrame:
    return _load_many(base_dir, "peer_fund_", FUND_COLUMNS, ["fund_date"])


def load_peer_ranks(base_dir: Optional[str] = None) -> pd.DataFrame:
    return _load_many(base_dir, "peer_rank_", RANK_COLUMNS, ["rank_date", "fund_date"])


def _append(path: str, rows: List[Dict[str, Any]], columns: List[str], key_cols: List[str]) -> Dict[str, int]:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):   # 열이 늘었으면(새 열은 맨 뒤) 기존 줄은 그대로 두고 빈 새 열만 붙여 다시 쓴다 — 값은 바꾸지 않는다
        head = list(pd.read_csv(path, nrows=0, encoding="utf-8-sig").columns)
        if head != columns:
            if head != columns[:len(head)]:
                raise ValueError(f"{os.path.basename(path)}의 열 순서가 코드와 다릅니다 — 새 열은 맨 뒤에만 추가해야 합니다")
            old = pd.read_csv(path, dtype=str, encoding="utf-8-sig", keep_default_na=False)
            old.reindex(columns=columns).to_csv(path, index=False, encoding="utf-8-sig")
    have = set()
    if os.path.exists(path):
        d = pd.read_csv(path, usecols=key_cols, dtype=str, encoding="utf-8-sig").fillna("")
        have = set(map(tuple, d[key_cols].values.tolist()))
    new = [{c: r.get(c) for c in columns} for r in rows if tuple(str(r.get(k) if r.get(k) is not None else "") for k in key_cols) not in have]
    if new:
        exists = os.path.exists(path)
        pd.DataFrame(new, columns=columns).to_csv(path, mode="a", header=not exists, index=False,
                                                  encoding="utf-8" if exists else "utf-8-sig")
    return {"saved": len(new), "skipped_existing": len(rows) - len(new)}


# ------------------------------------------------------------------ 매일 실행
def run_daily_peers(today: Optional[str] = None, base_dir: Optional[str] = None, yf: Any = None,
                    holdings_fn: Optional[Callable[[str], Optional[Dict[str, Any]]]] = None,
                    snapshot_dir: Optional[str] = None, full: bool = False, tickers: Optional[List[str]] = None,
                    pause: Optional[float] = None, progress: Optional[Callable[[int, int, str], None]] = None) -> Dict[str, Any]:
    """① 재무 기록이 없거나 FUND_REFRESH_DAYS일 이상 지났으면(또는 full) 커버리지·피어 묶음·재무를 다시 받는다.
    ② 오늘 가격 지표를 받아 묶음별 순위를 기록한다(같은 날·같은 종목·같은 묶음은 한 번만).
    tickers를 주면 커버리지를 그 종목들로 줄인다(빠른 점검용)."""
    if yf is None:
        import yfinance as yf
    date = today or pd.Timestamp.now(tz=PEER_CONFIG["TIMEZONE"]).strftime("%Y-%m-%d")
    wait = PEER_CONFIG["PAUSE_SEC"] if pause is None else float(pause)
    res: Dict[str, Any] = {"date": date, "refreshed": False, "fund_saved": 0, "rank_saved": 0, "failed": [], "no_group": {}}
    fund = load_peer_fund(base_dir)
    last_full = last_full_refresh(base_dir)
    stale = last_full is None or pd.isna(last_full) or (pd.Timestamp(date) - pd.Timestamp(last_full)).days >= PEER_CONFIG["FUND_REFRESH_DAYS"]
    if full or stale or tickers:
        cov = [_norm_ticker(t) for t in tickers] if tickers else coverage_universe(holdings_fn, snapshot_dir)
        info_map: Dict[str, Dict[str, Any]] = {}
        for t in cov:
            if _pb_market(t) == "KR":
                continue
            try:
                info_map[t] = yf.Ticker(t).info or {}
            except Exception:
                info_map[t] = {}
        built = build_groups(cov, info_map, _yf_industry_members(yf), load_peer_map(base_dir))
        res["no_group"] = built["no_group"]
        members = list(built["group_of"].keys())
        rows: List[Dict[str, Any]] = []
        for i, t in enumerate(members, 1):
            gk = built["group_of"][t]
            try:
                rows.append(fetch_fund_row(t, yf, date, gk, built["groups"][gk], t in cov, info=info_map.get(t)))
            except Exception as e:
                res["failed"].append(f"{t}({type(e).__name__})")
            if progress:
                progress(i, len(members), t)
            if wait > 0 and i < len(members):
                time.sleep(wait)
        r = _append(fund_path(date, base_dir), rows, FUND_COLUMNS, ["fund_date", "ticker"])
        mode = "partial" if tickers else "full"
        log = {"refresh_date": date, "mode": mode, "n_coverage": len(cov), "n_groups": len(built["groups"]), "n_members": len(members),
               "fund_saved": r["saved"], "n_failed": len(res["failed"]), "version": PEER_CONFIG["VERSION"]}
        lp = refresh_log_path(base_dir)
        pd.DataFrame([log], columns=REFRESH_LOG_COLUMNS).to_csv(lp, mode="a", header=not os.path.exists(lp), index=False,
                                                                 encoding="utf-8" if os.path.exists(lp) else "utf-8-sig")
        res.update(refreshed=True, refresh_mode=mode, fund_saved=r["saved"], n_coverage=len(cov), n_members=len(members),
                   n_groups=len(built["groups"]))
        fund = load_peer_fund(base_dir)
    if not len(fund):
        res["error"] = "재무 기록 없음"
        return res
    fd = fund["fund_date"].max()
    cur = fund[fund["fund_date"] == fd].copy()
    syms = sorted(set(cur["ticker"].astype(str)))
    benches = {PEER_CONFIG["SECTOR_ETF"].get(str(s), PEER_CONFIG["BENCH_FALLBACK"]["US"]) for s in cur.loc[cur["market"] == "US", "sector"]}
    benches |= {PEER_CONFIG["BENCH_FALLBACK"]["KR"]} if (cur["market"] == "KR").any() else set()
    prices = _download_many(yf, syms + sorted(benches), PEER_CONFIG["PRICE_PERIOD"])
    out_rows: List[Dict[str, Any]] = []
    for gk, g in cur.groupby("group", sort=True):
        recs = []
        for _, fr in g.iterrows():
            t = str(fr["ticker"])
            if fr["market"] == "KR":
                bsym = PEER_CONFIG["BENCH_FALLBACK"]["KR"]
            else:
                bsym = PEER_CONFIG["SECTOR_ETF"].get(str(fr.get("sector")), PEER_CONFIG["BENCH_FALLBACK"]["US"])
            pm = price_metrics(prices.get(t), prices.get(bsym))
            rec = {"ticker": t, **derive_metrics(fr.to_dict(), pm), "trend_state": pm["trend_state"], "price_as_of": pm["price_as_of"],
                   "surv_kill": bool(fr.get("surv_kill") is True or str(fr.get("surv_kill")).lower() == "true"),
                   "surv_warning": bool(fr.get("surv_warning") is True or str(fr.get("surv_warning")).lower() == "true"),
                   "is_coverage": fr.get("is_coverage"), "notes": fr.get("notes")}
            recs.append(rec)
        sdf = score_group(pd.DataFrame(recs).set_index("ticker", drop=False))
        for t, r in sdf.iterrows():
            row = {c: r.get(c) for c in RANK_COLUMNS if c in r.index}
            row.update(rank_date=date, group=gk, group_name=g["group_name"].iloc[0], group_source=g["group_source"].iloc[0],
                       n_group=len(g), fund_date=str(pd.Timestamp(fd).date()), version=PEER_CONFIG["VERSION"])
            for c in ("value_rank", "trend_rank"):
                row[c] = None if pd.isna(row.get(c)) else int(row[c])
            out_rows.append(row)
    r = _append(rank_path(date, base_dir), out_rows, RANK_COLUMNS, ["rank_date", "group", "ticker"])
    res.update(rank_saved=r["saved"], rank_skipped=r["skipped_existing"], fund_date=str(pd.Timestamp(fd).date()),
               n_price_missing=len([s for s in syms if s not in prices]))
    return res


def latest_peer_view(ticker: str, base_dir: Optional[str] = None, ranks: Optional[pd.DataFrame] = None,
                     fund: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """딥다이브용: 이 종목이 속한 묶음의 가장 최근 순위표(+ 재무 기록의 회사 이름·생존성 문구·다음 실적 발표일).
    반환 {available, reason, group, group_name, group_source, rank_date, fund_date, table, me}."""
    t = _norm_ticker(ticker)
    df = load_peer_ranks(base_dir) if ranks is None else ranks
    if not len(df):
        return {"available": False, "reason": "피어 기록 없음 — peers_daily.py가 아직 실행되지 않음"}
    mine = df[df["ticker"].astype(str) == t]
    if not len(mine):
        return {"available": False, "reason": "커버리지 밖 종목 — data/snapshots/watchlist.txt에 추가하면 다음 갱신부터 비교"}
    d = mine["rank_date"].max()
    me = mine[mine["rank_date"] == d].iloc[0]
    tbl = df[(df["rank_date"] == d) & (df["group"] == me["group"])].copy()
    fd = me.get("fund_date")
    fu = load_peer_fund(base_dir) if fund is None else fund
    if len(fu) and fd is not None and not pd.isna(fd):
        keep = ["ticker", "short_name", "surv_text", "next_earnings_date", "market", "eps_fy1", "n_fy1", "up_30d", "down_30d"]
        extra = fu[fu["fund_date"] == pd.Timestamp(fd)][[c for c in keep if c in fu]].drop_duplicates("ticker")
        tbl = tbl.merge(extra, on="ticker", how="left")
    for c in ("short_name", "surv_text", "next_earnings_date", "market", "eps_fy1", "n_fy1", "up_30d", "down_30d"):
        if c not in tbl:
            tbl[c] = None
    tbl = tbl.sort_values("value_score", ascending=False, na_position="last").reset_index(drop=True)
    me_full = tbl[tbl["ticker"].astype(str) == t].iloc[0].to_dict()
    return {"available": True, "group": me["group"], "group_name": me["group_name"], "group_source": me.get("group_source"),
            "rank_date": d, "fund_date": fd, "table": tbl, "me": me_full}


# ------------------------------------------------------------------ 딥다이브 요약 (강점·약점) — 표시 규칙도 사전 등록
SUMMARY_RULES: Dict[str, float] = {"STRENGTH_MIN": 60.0, "WEAKNESS_MAX": 40.0, "MAX_ITEMS": 3}
_PCT_KEYS = {"rev_growth_ttm", "eps_growth_fwd", "op_margin_ttm", "roe", "fy1_change_30d"}


def fmt_peer_metric(key: str, v: Any) -> str:
    """지표 값 → 짧은 문구 (빈칸은 '데이터 없음')."""
    x = _f(v)
    if x is None:
        return NO_DATA
    if key in _PCT_KEYS:
        return f"{x * 100:+.1f}%" if key != "op_margin_ttm" and key != "roe" else f"{x * 100:.1f}%"
    if key == "rs63":
        return f"{x * 100:+.1f}%p"
    if key == "rev_net_30d":
        return f"{x * 100:+.0f}%"
    if key in ("fwd_pe", "peg"):
        return f"{x:.1f}배" if key == "fwd_pe" else f"{x:.2f}"
    if key == "band_z":
        return f"z {x:+.2f}"
    if key == "state_pt":
        return {2.0: STATE_UP, 1.0: STATE_SIDE, 0.0: STATE_DOWN}.get(x, NO_DATA)
    return f"{x:.2f}"


def peer_summary(view: Dict[str, Any]) -> Dict[str, Any]:
    """latest_peer_view 결과 → 딥다이브 결론 카드용 요약.
    강점 = 요인 점수 ≥ 60 (높은 순 최대 3개), 약점 = 요인 점수 ≤ 40 (낮은 순 최대 3개), 그 사이는 '보통'.
    요인마다 이 종목의 지표 값과 피어 중앙값(하드 킬 종목 제외)을 함께 적는다."""
    if not view.get("available"):
        return {"available": False, "reason": view.get("reason")}
    tbl, me = view["table"], view["me"]
    killed = tbl["surv_kill"].astype(str).str.lower().eq("true") if "surv_kill" in tbl else pd.Series(False, index=tbl.index)
    base = tbl[~killed]
    rows = []
    for fk, flabel in FACTOR_LABELS.items():
        sc = _f(me.get(f"f_{fk}"))
        parts = []
        for key, f2, _, label in PEER_METRICS:
            if f2 != fk:
                continue
            mv = me.get(key)
            if _f(mv) is None:
                continue
            med = pd.to_numeric(base[key], errors="coerce").median() if key in base else None
            short = label.split("(")[0]
            parts.append(f"{short} {fmt_peer_metric(key, mv)}" + ("" if med is None or pd.isna(med) else f" (피어 중앙 {fmt_peer_metric(key, med)})"))
        rows.append({"factor": fk, "label": flabel, "score": sc, "detail": " · ".join(parts) or NO_DATA})
    scored = [r for r in rows if r["score"] is not None]
    strengths = sorted([r for r in scored if r["score"] >= SUMMARY_RULES["STRENGTH_MIN"]], key=lambda r: -r["score"])[:int(SUMMARY_RULES["MAX_ITEMS"])]
    weaknesses = sorted([r for r in scored if r["score"] <= SUMMARY_RULES["WEAKNESS_MAX"]], key=lambda r: r["score"])[:int(SUMMARY_RULES["MAX_ITEMS"])]

    def strat(name: str) -> Dict[str, Any]:
        rk = _f(me.get(f"{name}_rank"))
        top = me.get(f"{name}_top")
        return {"score": _f(me.get(f"{name}_score")), "rank": None if rk is None else int(rk), "n": int(_f(me.get(f"{name}_n")) or 0),
                "top": (str(top).lower() == "true") if top is not None and not (isinstance(top, float) and math.isnan(top)) else None,
                "note": "" if me.get(f"{name}_note") is None or pd.isna(me.get(f"{name}_note")) else str(me.get(f"{name}_note"))}

    def leader(name: str) -> Optional[str]:
        r = tbl.dropna(subset=[f"{name}_rank"])
        r = r[pd.to_numeric(r[f"{name}_rank"], errors="coerce") == 1]
        return None if not len(r) else str(r["ticker"].iloc[0])

    return {"available": True, "group_name": view["group_name"], "group_source": view.get("group_source"), "n": len(tbl),
            "rank_date": view["rank_date"], "fund_date": view.get("fund_date"), "version": me.get("version"),
            "value": strat("value"), "trend": strat("trend"), "leaders": {"value": leader("value"), "trend": leader("trend")},
            "factors": rows, "strengths": strengths, "weaknesses": weaknesses,
            "surv_kill": str(me.get("surv_kill")).lower() == "true", "surv_warning": str(me.get("surv_warning")).lower() == "true",
            "surv_text": None if me.get("surv_text") is None or pd.isna(me.get("surv_text")) else str(me.get("surv_text")),
            "next_earnings_date": None if me.get("next_earnings_date") is None or pd.isna(me.get("next_earnings_date")) else str(me.get("next_earnings_date"))}


# ------------------------------------------------------------------ 3단계: 앞서려면 (피어 상위 25% 기준까지 남은 차이) — 사전 등록
GAP_RULES: Dict[str, float] = {"TOP_Q": 0.25, "TARGET_FACTOR_SCORE": 75.0}


def _money(v: Optional[float], market: Any) -> str:
    x = _f(v)
    if x is None:
        return NO_DATA
    return f"{x:,.0f}원" if str(market) == "KR" else f"${x:,.2f}"


def metric_thresholds(tbl: pd.DataFrame) -> Dict[str, Optional[float]]:
    """지표마다 피어 상위 25% 기준값(좋은 쪽 25% 경계, 하드 킬 제외). 클수록 좋으면 75% 분위, 작을수록 좋으면 25% 분위."""
    killed = tbl["surv_kill"].astype(str).str.lower().eq("true") if "surv_kill" in tbl else pd.Series(False, index=tbl.index)
    base = tbl[~killed]
    out: Dict[str, Optional[float]] = {}
    for key, _, direction, _ in PEER_METRICS:
        v = pd.to_numeric(base[key], errors="coerce").dropna() if key in base else pd.Series(dtype=float)
        if len(v) < 2:
            out[key] = None
            continue
        q = 1.0 - GAP_RULES["TOP_Q"] if direction > 0 else GAP_RULES["TOP_Q"]
        out[key] = float(v.quantile(q))
    return out


def _metric_condition(key: str, me: Dict[str, Any], target: Optional[float]) -> Optional[Dict[str, Any]]:
    """한 지표가 상위 25% 기준에 못 미치면 무엇이 얼마나 바뀌어야 하는지(산수). 이미 넘었거나 계산 불가면 None."""
    cur = _f(me.get(key))
    if cur is None or target is None:
        return None
    direction = next(m[2] for m in PEER_METRICS if m[0] == key)
    if (direction > 0 and cur >= target) or (direction < 0 and cur <= target):
        return None
    mk, eps = me.get("market"), _f(me.get("eps_fy1"))
    fm = lambda v: fmt_peer_metric(key, v)
    if key == "fwd_pe":
        chg = target / cur - 1.0
        price_now = cur * eps if eps is not None and eps > 0 else None
        txt = (f"가격 경로: 주가 {_money(target * eps, mk)} 이하(지금 {_money(price_now, mk)}, {chg * 100:+.1f}%)" if price_now else
               f"가격 경로: 주가 {chg * 100:+.1f}%")
        if eps is not None and eps > 0:
            txt += f" · 실적 경로: 주가 그대로면 FY1 EPS {_money(price_now / target, mk)} 이상({(cur / target - 1.0) * 100:+.1f}%)"
        return {"metric": key, "kind": "가격·실적", "current": cur, "target": target,
                "text": f"선행 PER {fm(cur)} → {fm(target)} 이하. {txt}"}
    if key == "peg":
        g = _f(me.get("eps_growth_fwd"))
        txt = f"주가 {(target / cur - 1.0) * 100:+.1f}%(성장 그대로)"
        if g is not None and g > 0:
            txt += f" 또는 선행 EPS 성장 {g * 100:.1f}% → {g * cur / target * 100:.1f}% 이상(주가 그대로)"
        return {"metric": key, "kind": "가격·실적", "current": cur, "target": target, "text": f"PEG {fm(cur)} → {fm(target)} 이하: {txt}"}
    if key == "band_z":
        return {"metric": key, "kind": "가격", "current": cur, "target": target,
                "text": f"자기 PER 밴드 {fm(cur)} → {fm(target)} 이하(자기 역사 대비 더 싸져야 — 주가 하락 또는 이익 증가)"}
    if key == "rev_net_30d":
        n = _f(me.get("n_fy1"))
        up, dn = _f(me.get("up_30d")), _f(me.get("down_30d"))
        extra = ""
        if n is not None and n > 0:
            need = math.ceil(target * n - 1e-9)
            now = f"지금 {int(up - dn):+d}건, " if up is not None and dn is not None else ""
            extra = f" — 상향 − 하향 {need:+d}건 이상({now}애널리스트 {int(n)}명)"
        return {"metric": key, "kind": "추정치", "current": cur, "target": target,
                "text": f"30일 추정치 순상향 {fm(cur)} → {fm(target)} 이상{extra}"}
    if key == "fy1_change_30d":
        return {"metric": key, "kind": "추정치", "current": cur, "target": target,
                "text": f"FY1 추정치 30일 변화 {fm(cur)} → {fm(target)} 이상(추정치가 올라가야)"}
    if key == "rs63":
        return {"metric": key, "kind": "가격 흐름", "current": cur, "target": target,
                "text": f"섹터 ETF 대비 63일 상대강도 {fm(cur)} → {fm(target)} 이상 — 섹터 ETF보다 {(target - cur) * 100:.1f}%p 더 올라야"}
    if key == "state_pt":
        want = STATE_UP if target > 1.0 else (STATE_SIDE if target > 0.0 else STATE_DOWN)
        return {"metric": key, "kind": "가격 흐름", "current": cur, "target": target,
                "text": f"틀 A 추세 상태 {fm(cur)} → {want}(피어 상위 25% 기준)"}
    label = next(m[3] for m in PEER_METRICS if m[0] == key).split("(")[0]
    return {"metric": key, "kind": "실적", "current": cur, "target": target,
            "text": f"{label} {fm(cur)} → {fm(target)} 이상({(target - cur) * 100:+.1f}%p)"}


def peer_gaps(view: Dict[str, Any]) -> Dict[str, Any]:
    """앞서려면: ① 요인마다(점수 < 75) 지표별로 피어 상위 25% 기준까지 남은 차이 ② 전략별 '가장 가까운 경로' =
    그 요인 하나만 점수 75(상위 25% 기준)에 닿았을 때 전략 점수가 가장 많이 오르는 요인(다른 종목은 그대로 두고 새 순위도 계산)
    ③ 다음 실적 발표일에 확인할 항목(실적·추정치 조건)."""
    if not view.get("available"):
        return {"available": False, "reason": view.get("reason")}
    tbl, me = view["table"], view["me"]
    th = metric_thresholds(tbl)
    tgt = GAP_RULES["TARGET_FACTOR_SCORE"]
    factors = []
    for fk, flabel in FACTOR_LABELS.items():
        sc = _f(me.get(f"f_{fk}"))
        conds = [c for c in (_metric_condition(k, me, th.get(k)) for k, f2, _, _ in PEER_METRICS if f2 == fk) if c]
        factors.append({"factor": fk, "label": flabel, "score": sc, "below": sc is not None and sc < tgt, "conditions": conds})
    killed_me = str(me.get("surv_kill")).lower() == "true"
    nearest: Dict[str, Any] = {}
    for sname, cfg in PEER_CONFIG["STRATEGIES"].items():
        cur_scores = pd.to_numeric(tbl[f"{sname}_score"], errors="coerce")
        old = _f(me.get(f"{sname}_score"))
        old_rank = _f(me.get(f"{sname}_rank"))
        if killed_me or old is None or old_rank is None:
            nearest[sname] = {"available": False, "reason": "순위 없음(생존성 하드 킬·피어 부족·자료 부족)"}
            continue
        others = cur_scores[tbl["ticker"].astype(str) != str(me.get("ticker"))].dropna()
        if int(old_rank) == 1:
            second = others.max() if len(others) else None
            if second is not None and abs(old - second) < 1e-9:
                tail = " — 공동 1위"
            else:
                tail = "" if second is None else f" — 2위와 {old - second:.1f}점 차"
            nearest[sname] = {"available": True, "already_top": True, "text": f"{cfg['label']}: 이미 1위{tail}"}
            continue
        fs = {fk: _f(me.get(f"f_{fk}")) for fk in FACTOR_LABELS}
        best = None
        for fk in [cfg["core"]] + cfg["aux"]:
            if fs.get(fk) is None or fs[fk] >= tgt:
                continue
            new, _ = strategy_score({**fs, fk: tgt}, sname)
            if new is None:
                continue
            gain = new - old
            new_rank = int((others > new).sum()) + 1
            if best is None or gain > best["gain"] + 1e-12:
                best = {"factor": fk, "label": FACTOR_LABELS[fk], "old_score": old, "new_score": new, "gain": gain,
                        "old_rank": int(old_rank), "new_rank": new_rank}
        if best is None:
            nearest[sname] = {"available": True, "already_top": False, "factor": None,
                              "text": f"{cfg['label']}: 모든 요인이 이미 상위 25% 기준 이상 — 남은 차이는 다른 종목과의 점수 차"}
        else:
            conds = next(f["conditions"] for f in factors if f["factor"] == best["factor"])
            if best["new_rank"] < best["old_rank"]:
                rk_txt = f"{best['old_rank']}위 → {best['new_rank']}위"
            else:
                lead = others.max() if len(others) else None
                rk_txt = f"순위는 {best['old_rank']}위 그대로" + ("" if lead is None else f"(1위와 {lead - best['new_score']:.1f}점 차가 남음)")
            best.update(available=True, already_top=False, conditions=conds,
                        text=f"{cfg['label']}: {best['label']}만 피어 상위 25% 수준이 되면 {best['old_score']:.0f}점 → {best['new_score']:.0f}점, {rk_txt}")
            nearest[sname] = best
    check = [c["text"] for f in factors if f["below"] for c in f["conditions"] if c["kind"] in ("실적", "추정치", "가격·실적")]
    nd = me.get("next_earnings_date")
    return {"available": True, "thresholds": th, "factors": factors, "nearest": nearest, "check_items": check,
            "next_earnings_date": None if nd is None or (isinstance(nd, float) and math.isnan(nd)) else str(nd)}
