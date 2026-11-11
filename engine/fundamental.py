"""가치 트랙(Track 3, V): 입력층 fetch_real_fundamental_inputs, 피어·PER 밴드, analyze_fundamental_track.

engine.py를 기능별로 나눈 모듈. 바깥에서는 `import engine` 후 `engine.이름`으로 쓴다(engine/__init__.py).
"""
import math
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd

from .common import NO_DATA, adjust_krx_price, fmt_price
from .earnings import (
    _DART_ACCOUNTS,
    _dart_fetch,
    _dart_pick,
    _get_dart_client,
    _num,
    _stmt_series,
    analyze_earnings_track,
)
from .macro import fetch_risk_free_10y

# =====================================================================
# [Track 3] 펀더멘탈 분석 트랙 v1.0 (멀티플·가치평가·0~3차 매수기준선 엔진)
# =====================================================================
# ---------------------------------------------------------------------
# [V 트랙] 펀더멘탈 입력 실제 데이터 수집
#   미국/기타: yfinance (info · 가격이력 · 연간/분기 재무제표 · 산업 상위 기업)
#   한국(.KS/.KQ): DART(연간 재무제표) + yfinance(가격·베타·배당·컨센서스 목표가)
#   못 받은 값은 None (가짜 값으로 채우지 않는다). 화면에는 '데이터 없음'으로 표시된다.
# ---------------------------------------------------------------------
FUND_DATA_PARAMS = {
    "peer_candidates": 20,          # 산업 상위 기업 중 조회할 후보 수
    "peer_max_count": 12,           # 유효 피어 상한
    "peer_min_valid": 5,            # 피어 통계(중앙값·상위25%·σ)를 내는 최소 유효 종목 수
    "peer_pe_range": (0.0, 200.0),  # 유효 PER 범위 (초과·음수는 제외)
    "peer_roe_range": (-1.0, 3.0),  # 피어 ROE 유효 범위 (자본잠식 등 극단값 제외)
    "peer_growth_range": (-0.9, 3.0),  # 피어 선행/후행 EPS 성장률 유효 범위
    "band_years": 5,                # 자기 밴드 관측 창
    "band_min_points": 24,          # 밴드 통계에 필요한 최소 월간 관측치 (2년)
    "band_max_eps_change": 3.0,     # 밴드 기간 안 확정 EPS 최대/최소가 이 배수를 넘으면 밴드 신뢰 불가(이익 규모 급변)
    "band_max_log_std": 0.5,        # 로그 PER σ가 이보다 크면 밴드 신뢰 불가
    "band_lag_days_us": 60,         # 사업연도 종료 후 EPS가 시장에 알려지기까지의 지연 (미국)
    "band_lag_days_kr": 90,         # 한국 (사업보고서 제출 기한)
}

# DART 계정 추가 (자본·유동성·차입·지배주주순이익)
_DART_ACCOUNTS.update({
    "equity_parent": {"ids": ["ifrs-full_EquityAttributableToOwnersOfParent"],
                      "names": ["지배기업의소유주에게귀속되는자본", "지배기업소유주지분", "지배기업의소유주에게귀속되는지분"], "sj": ["BS"]},
    "equity": {"ids": ["ifrs-full_Equity"], "names": ["자본총계"], "sj": ["BS"]},
    "ni_parent": {"ids": ["ifrs-full_ProfitLossAttributableToOwnersOfParent"],
                  "names": ["지배기업의소유주에게귀속되는당기순이익", "지배기업소유주지분순이익", "지배기업소유주지분",
                            "지배기업의소유주에게귀속되는당기순이익(손실)"], "sj": ["IS", "CIS"]},
    "cash": {"ids": ["ifrs-full_CashAndCashEquivalents"], "names": ["현금및현금성자산"], "sj": ["BS"]},
    "sti": {"ids": ["ifrs-full_CurrentFinancialAssetsAtFairValueThroughProfitOrLoss"],
            "names": ["단기금융상품", "단기금융자산", "단기투자자산"], "sj": ["BS"]},
    "debt_short": {"ids": ["ifrs-full_ShorttermBorrowings", "ifrs-full_CurrentBorrowings"],
                   "names": ["단기차입금", "유동차입금"], "sj": ["BS"]},
    "debt_cur_ltd": {"ids": ["ifrs-full_CurrentPortionOfLongtermBorrowings"],
                     "names": ["유동성장기부채", "유동성장기차입금", "유동성사채"], "sj": ["BS"]},
    "debt_long": {"ids": ["ifrs-full_LongtermBorrowings", "ifrs-full_NoncurrentPortionOfNoncurrentLoansReceived"],
                  "names": ["장기차입금"], "sj": ["BS"]},
    "bonds": {"ids": ["ifrs-full_BondsIssued", "ifrs-full_NoncurrentPortionOfNoncurrentBondsIssued"],
              "names": ["사채", "회사채"], "sj": ["BS"]},
})


def _tz_naive(ts: Any) -> Any:
    t = pd.Timestamp(ts)
    return t.tz_localize(None) if t.tzinfo is not None else t


def _annual_pairs(df: Any, names: List[str]) -> List[Tuple[Any, float]]:
    """yfinance 연간 재무제표(행=항목, 열=결산일)에서 names 중 처음 찾은 행을 (결산일, 값) 목록으로."""
    try:
        if df is None or len(df) == 0:
            return []
        idx = {str(i).strip().lower(): i for i in df.index}
        for nm in names:
            key = nm.strip().lower()
            if key in idx:
                out = []
                for c, v in df.loc[idx[key]].items():
                    x = _num(v)
                    if x is not None:
                        out.append((_tz_naive(c), x))
                if out:
                    return out
    except Exception:
        pass
    return []


def _month_end_prices(hist: Any) -> List[Tuple[Any, float]]:
    """일봉 종가(분할 보정, 배당 미보정)를 월말 종가 목록 [(날짜, 종가)]로. 오래된 순."""
    out: List[Tuple[Any, float]] = []
    try:
        s = hist["Close"].dropna()
        last_key = None
        for ts, v in s.items():
            t = _tz_naive(ts)
            x = _num(v)
            if x is None or x <= 0:
                continue
            key = (t.year, t.month)
            if out and key == last_key:
                out[-1] = (t, x)
            else:
                out.append((t, x))
            last_key = key
    except Exception:
        return []
    return out


def _asof_fy(pairs: List[Tuple[Any, float]], date: Any, lag_days: int) -> Optional[float]:
    """date 시점에 시장이 알고 있던 가장 최근 사업연도 값 (결산일 + 공시 지연 이후에만 반영)."""
    best = None
    for fy_end, v in sorted(pairs, key=lambda x: x[0]):
        if fy_end + pd.Timedelta(days=lag_days) <= date:
            best = v
    return best


def _lagged_ratio_stats(prices: List[Tuple[Any, float]], pairs: List[Tuple[Any, float]],
                        lag_days: int, years: int, min_points: int) -> Optional[Dict[str, Any]]:
    """월말 주가 ÷ (그 시점에 알려진 연간 주당값) 의 평균·표준편차·최소. 분모가 0 이하인 달은 제외."""
    if not prices or not pairs:
        return None
    cutoff = prices[-1][0] - pd.DateOffset(years=years)
    vals: List[float] = []
    dens: List[float] = []
    series: List[Tuple[str, float]] = []
    for d, p in prices:
        if d < cutoff:
            continue
        den = _asof_fy(pairs, d, lag_days)
        if den is None or den <= 0:
            continue
        vals.append(p / den)
        dens.append(den)
        series.append((str(pd.Timestamp(d).date()), round(p / den, 2)))
    if len(vals) < min_points:
        return None
    # 로그 PER로 계산: 이익이 급변한 종목(예: 확정 EPS가 몇 년 새 수십 배)은 과거 PER이 수백 배로 튀어
    # 산술 평균·σ가 망가진다. 로그 평균의 지수(기하평균)를 중심으로, σ는 로그 단위(배수 비율)로 둔다.
    lv = [math.log(v) for v in vals]
    lm = sum(lv) / len(lv)
    lsd = math.sqrt(sum((x - lm) ** 2 for x in lv) / len(lv))
    m = math.exp(lm)
    return {"mean": m, "log_std": lsd, "std": m * (math.exp(lsd) - 1.0), "min": min(vals), "n": len(vals),
            "span_years": round(len(vals) / 12.0, 1), "series": series,
            "eps_lo": min(dens), "eps_hi": max(dens)}


def _band_unreliable_reasons(band: Dict[str, Any]) -> List[str]:
    """자기 PER 밴드를 믿을 수 없는 이유 목록 (비어 있으면 사용 가능). 이익 규모 급변·PER 분산 과대."""
    why: List[str] = []
    lo, hi = band.get("eps_lo"), band.get("eps_hi")
    chg = (hi / lo) if (lo and hi and lo > 0) else None
    if chg is not None and chg > FUND_DATA_PARAMS["band_max_eps_change"]:
        why.append(f"밴드 기간 확정 EPS {lo:,.2f} → {hi:,.2f} ({chg:.1f}배)")
    if band.get("log_std") is not None and band["log_std"] > FUND_DATA_PARAMS["band_max_log_std"]:
        why.append(f"로그 PER σ {band['log_std']:.2f} > {FUND_DATA_PARAMS['band_max_log_std']}")
    return why


def _percentile(sorted_vals: List[float], q: float) -> float:
    """선형 보간 백분위수 (q: 0~1)."""
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = q * (len(sorted_vals) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def _peer_stats_us(tk: Any, info: Dict[str, Any], ticker: str, meta: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """같은 산업(없으면 같은 섹터)의 상위 기업들로 피어 PER·ROE·성장률 단면 통계를 만든다. 못 받으면 전부 None."""
    out: Dict[str, Optional[float]] = {"peer_median_per": None, "peer_top25_per": None, "peer_std_per": None,
                                       "peer_median_roe": None, "peer_median_fwd_growth": None}
    try:
        import yfinance as yf
    except Exception:
        meta["notes"].append("피어: yfinance 없음")
        return out
    symbols: List[str] = []
    for kind, key_name in (("Industry", "industryKey"), ("Sector", "sectorKey")):
        key = info.get(key_name)
        if not key:
            continue
        try:
            obj = getattr(yf, kind)(key)
            df = obj.top_companies
            if df is not None and len(df) > 0:
                symbols = [str(s) for s in list(df.index)]
                meta["peer_group"] = f"{kind}:{key}"
                break
        except Exception:
            continue
    me = str(ticker).upper()
    symbols = [s for s in symbols if s.upper() != me][:FUND_DATA_PARAMS["peer_candidates"]]
    if not symbols:
        meta["notes"].append("피어 목록 없음")
        return out

    def _one(sym: str) -> Dict[str, Optional[float]]:
        try:
            pi = yf.Ticker(sym).info or {}
        except Exception:
            return {}
        pe = _num(pi.get("forwardPE"))
        lo, hi = FUND_DATA_PARAMS["peer_pe_range"]
        if pe is None or not (lo < pe <= hi):
            pe = None
        te, fe = _num(pi.get("trailingEps")), _num(pi.get("forwardEps"))
        growth = (fe / te - 1.0) if (te is not None and fe is not None and te > 0 and fe > 0) else None
        return {"pe": pe, "roe": _num(pi.get("returnOnEquity")), "g": growth}

    rows: List[Dict[str, Optional[float]]] = []
    try:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=6) as ex:
            rows = list(ex.map(_one, symbols))
    except Exception:
        rows = [_one(s) for s in symbols]
    # 시가총액 비중 순서(원래 순서)대로 유효한 PER만 최대 peer_max_count개 채택 (값 크기로 자르지 않는다)
    _valid_rows = [r for r in rows if r.get("pe") is not None][:FUND_DATA_PARAMS["peer_max_count"]]
    pes = sorted(r["pe"] for r in _valid_rows)
    meta["peer_count"] = len(pes)
    if len(pes) >= FUND_DATA_PARAMS["peer_min_valid"]:
        m = sum(pes) / len(pes)
        out["peer_median_per"] = _percentile(pes, 0.5)
        out["peer_top25_per"] = _percentile(pes, 0.75)
        out["peer_std_per"] = math.sqrt(sum((v - m) ** 2 for v in pes) / len(pes))
    else:
        meta["notes"].append(f"피어 유효 PER {len(pes)}개 (<{FUND_DATA_PARAMS['peer_min_valid']}) — 피어 통계 제외")
    # ROE·성장률도 PER이 유효한 같은 피어 집합에서만, 이상치(ROE −3,092 등)를 빼고 계산한다. 피어 PER 통계가 없으면 같이 '데이터 없음'.
    if len(pes) >= FUND_DATA_PARAMS["peer_min_valid"]:
        r_lo, r_hi = FUND_DATA_PARAMS["peer_roe_range"]
        g_lo, g_hi = FUND_DATA_PARAMS["peer_growth_range"]
        roes = sorted(r["roe"] for r in _valid_rows if r.get("roe") is not None and r_lo <= r["roe"] <= r_hi)
        if len(roes) >= FUND_DATA_PARAMS["peer_min_valid"]:
            out["peer_median_roe"] = _percentile(roes, 0.5)
        gs = sorted(r["g"] for r in _valid_rows if r.get("g") is not None and g_lo <= r["g"] <= g_hi)
        if len(gs) >= FUND_DATA_PARAMS["peer_min_valid"]:
            out["peer_median_fwd_growth"] = _percentile(gs, 0.5)
    return out


def _dart_annual_block(dart: Any, code6: str, cache: Dict[Tuple[int, str], Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    """DART 사업보고서에서 EPS·자본 이력(최대 6년)과 최신 연도의 유동성·차입·FCF·주식수 계산 재료를 모은다."""
    blk: Dict[str, Any] = {"eps_pairs": [], "equity_pairs": [], "latest": {}, "prev": {}}
    this_year = datetime.now().year
    base_year = None
    for y in (this_year - 1, this_year - 2):          # 가장 최근에 공시된 사업연도
        if _dart_fetch(dart, code6, y, "11011", cache) is not None:
            base_year = y
            break
    if base_year is None:
        meta["notes"].append("DART 사업보고서 없음")
        return blk
    meta["dart_base_year"] = base_year
    eps_pairs: Dict[int, float] = {}
    eq_pairs: Dict[int, float] = {}
    for y in (base_year, base_year - 3):               # 사업보고서 1건에 당기·전기·전전기 3개년이 들어 있다
        df = _dart_fetch(dart, code6, y, "11011", cache)
        if df is None:
            continue
        for off, col in ((0, "thstrm_amount"), (1, "frmtrm_amount"), (2, "bfefrmtrm_amount")):
            yy = y - off
            e = _dart_pick(df, "eps", col)
            if e is not None:
                eps_pairs.setdefault(yy, e)
            q = _dart_pick(df, "equity_parent", col)
            if q is None:
                q = _dart_pick(df, "equity", col)
            if q is not None:
                eq_pairs.setdefault(yy, q)
    blk["eps_pairs"] = [(pd.Timestamp(f"{y}-12-31"), v) for y, v in sorted(eps_pairs.items())]
    blk["equity_pairs"] = [(pd.Timestamp(f"{y}-12-31"), v) for y, v in sorted(eq_pairs.items())]
    dfl = _dart_fetch(dart, code6, base_year, "11011", cache)
    for key in ("cash", "sti", "debt_short", "debt_cur_ltd", "debt_long", "bonds", "equity_parent", "equity",
                "ocf", "capex", "ni_parent", "eps"):
        col_now = "thstrm_amount"
        blk["latest"][key] = _dart_pick(dfl, key, col_now)
        blk["prev"][key] = _dart_pick(dfl, key, "frmtrm_amount")
    return blk


def _kr_liquidity_and_survival(blk: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Optional[float]]:
    L, P = blk.get("latest", {}), blk.get("prev", {})
    out: Dict[str, Optional[float]] = {"net_cash_krw": None, "fcf_ttm_krw": None,
                                       "short_term_debt_ratio": None, "share_dilution_yoy": None}
    cash = L.get("cash")
    liquid = None if cash is None else cash + (L.get("sti") or 0.0)
    debt_items = {k: L.get(k) for k in ("debt_short", "debt_cur_ltd", "debt_long", "bonds") if L.get(k) is not None}
    if liquid is not None and debt_items:
        out["net_cash_krw"] = liquid - sum(debt_items.values())
        meta["notes"].append("순현금: DART 계정명 기준 차입 항목(" + ", ".join(sorted(debt_items)) + ")만 합산 — 미확인 차입이 있으면 과대 추정 가능")
    short_items = [L.get(k) for k in ("debt_short", "debt_cur_ltd") if L.get(k) is not None]
    if liquid is not None and liquid > 0 and short_items:
        out["short_term_debt_ratio"] = sum(short_items) / liquid
    ocf, capex = L.get("ocf"), L.get("capex")
    if ocf is not None and capex is not None:
        out["fcf_ttm_krw"] = ocf - abs(capex)
        meta["notes"].append("FCF: 최근 사업연도(연간) OCF − 유형자산 취득")
    n1, e1, n0, e0 = L.get("ni_parent"), L.get("eps"), P.get("ni_parent"), P.get("eps")
    if None not in (n1, e1, n0, e0) and e1 and e0:
        sh1, sh0 = n1 / e1, n0 / e0
        if sh0 > 0 and sh1 > 0:
            out["share_dilution_yoy"] = sh1 / sh0 - 1.0
            meta["notes"].append("희석률: 지배주주순이익÷기본EPS로 역산한 가중평균 주식수의 전년 대비 (완전희석 아님)")
    return out


def _us_liquidity_and_survival(tk: Any, meta: Dict[str, Any]) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {"net_cash_krw": None, "fcf_ttm_krw": None,
                                       "short_term_debt_ratio": None, "share_dilution_yoy": None}
    try:
        bs = tk.quarterly_balance_sheet
        liquid = (_stmt_series(bs, ["Cash Cash Equivalents And Short Term Investments"]) or [None])[0]
        if liquid is None:
            liquid = (_stmt_series(bs, ["Cash And Cash Equivalents"]) or [None])[0]
        total_debt = (_stmt_series(bs, ["Total Debt"]) or [None])[0]
        cur_debt = (_stmt_series(bs, ["Current Debt And Capital Lease Obligation", "Current Debt"]) or [None])[0]
        if liquid is not None and total_debt is not None:
            out["net_cash_krw"] = liquid - total_debt
        if liquid is not None and liquid > 0 and cur_debt is not None:
            out["short_term_debt_ratio"] = cur_debt / liquid
    except Exception:
        meta["notes"].append("분기 재무상태표 없음")
    try:
        cf = tk.quarterly_cashflow
        fcf_q = _stmt_series(cf, ["Free Cash Flow"])
        if len(fcf_q) >= 4 and all(v is not None for v in fcf_q[:4]):
            out["fcf_ttm_krw"] = float(sum(fcf_q[:4]))
        else:
            ocf_q = _stmt_series(cf, ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities"])
            cap_q = _stmt_series(cf, ["Capital Expenditure", "Purchase Of PPE"])
            if len(ocf_q) >= 4 and len(cap_q) >= 4 and all(v is not None for v in ocf_q[:4] + cap_q[:4]):
                out["fcf_ttm_krw"] = float(sum(ocf_q[:4]) - sum(abs(c) for c in cap_q[:4]))
    except Exception:
        meta["notes"].append("분기 현금흐름표 없음")
    try:
        sh_q = _stmt_series(tk.quarterly_income_stmt, ["Diluted Average Shares"])
        if len(sh_q) >= 5 and sh_q[0] is not None and sh_q[4] is not None and sh_q[4] > 0:
            out["share_dilution_yoy"] = sh_q[0] / sh_q[4] - 1.0
        else:
            sh_a = _annual_pairs(tk.income_stmt, ["Diluted Average Shares"])
            sh_a = sorted(sh_a, key=lambda x: x[0])
            if len(sh_a) >= 2 and sh_a[-2][1] > 0:
                out["share_dilution_yoy"] = sh_a[-1][1] / sh_a[-2][1] - 1.0
    except Exception:
        pass
    return out


def fetch_real_fundamental_inputs(ticker: str, current_price: float) -> Dict[str, Any]:
    """V 트랙 입력을 실제 데이터로 만든다. 반환: {"values": {키: 값 또는 None}, "meta": {...}}.
    못 받은 값은 None이며 가짜 값으로 채우지 않는다. 모델 상수(성숙 성장률 등)는 여기서 다루지 않는다."""
    is_krx = str(ticker).upper().endswith((".KS", ".KQ"))
    meta: Dict[str, Any] = {"source": "DART+yfinance" if is_krx else "yfinance", "notes": [],
                            "band_basis": "후행(TTM·연간 확정) EPS 기준", "peer_count": 0, "peer_group": None,
                            "band_points": 0, "band_span_years": None, "pb_points": 0}
    keys = ["industry_beta", "sustainable_roe", "dividend_yield", "is_financial_sector", "listing_years",
            "self_5y_mean", "self_5y_std", "self_5y_log_std", "self_5y_min", "ind_5y_mean", "ind_5y_std",
            "ind_rel_5y_mean", "ind_rel_5y_std", "ind_rel_current",
            "peer_median_per", "peer_top25_per", "peer_std_per", "peer_rel_5y_mean", "peer_rel_5y_std",
            "ind_rerate_evidences", "ind_derate_evidences", "ind_evidence_2q_confirmed",
            "current_pb", "pb_5y_mean", "pb_5y_std",
            "company_roe", "peer_median_roe", "company_fwd_growth", "peer_median_fwd_growth",
            "shareholder_return_advantage", "price_12m_ago", "eps_lag_12m_ago", "eps_lag_now",
            "mkt_per_12m_ago", "mkt_per_now", "ind_per_12m_ago", "ind_per_now", "historical_max_5y_cagr",
            "consensus_target_price", "net_cash_krw", "fcf_ttm_krw", "short_term_debt_ratio", "share_dilution_yoy"]
    vals: Dict[str, Any] = {k: None for k in keys}

    try:
        import yfinance as yf
        tk = yf.Ticker(ticker)
    except Exception:
        meta["notes"].append("yfinance 사용 불가 — 시장 기반 값 전부 없음")
        return {"values": vals, "meta": meta}

    # ---- info: 베타·ROE·배당·섹터·상장연수·컨센서스 목표가 ----
    info: Dict[str, Any] = {}
    try:
        info = tk.info or {}
    except Exception:
        meta["notes"].append("yfinance info 없음")
    px = float(current_price) if current_price else None
    vals["industry_beta"] = _num(info.get("beta"))
    vals["company_roe"] = _num(info.get("returnOnEquity"))
    vals["sustainable_roe"] = vals["company_roe"]
    vals["consensus_target_price"] = _num(info.get("targetMeanPrice"))
    drate = _num(info.get("dividendRate"))
    if drate is None:
        drate = _num(info.get("trailingAnnualDividendRate"))
    if drate is not None and px:
        vals["dividend_yield"] = drate / px
    elif info and not is_krx:
        vals["dividend_yield"] = 0.0     # 미국: info는 받았는데 배당 항목이 없으면 무배당으로 본다
    sector = str(info.get("sector") or "")
    if sector:
        vals["is_financial_sector"] = sector.strip().lower() == "financial services"
    t0 = _num(info.get("firstTradeDateEpochUtc"))
    if t0 is None:
        ms = _num(info.get("firstTradeDateMilliseconds"))
        t0 = None if ms is None else ms / 1000.0
    if t0:
        vals["listing_years"] = round((datetime.now().timestamp() - t0) / (365.25 * 86400), 1)
    te, fe = _num(info.get("trailingEps")), _num(info.get("forwardEps"))
    if te is not None and fe is not None and te > 0 and fe > 0:
        vals["company_fwd_growth"] = fe / te - 1.0

    # ---- 가격 이력: 12개월 전 가격 · 자기 PER 밴드 · P/B 밴드 ----
    prices: List[Tuple[Any, float]] = []
    try:
        hist = tk.history(period=f"{FUND_DATA_PARAMS['band_years'] + 1}y", auto_adjust=False)
        prices = _month_end_prices(hist)
        if len(hist) > 0:
            closes = hist["Close"].dropna()
            last_d = _tz_naive(closes.index[-1])
            target = last_d - pd.DateOffset(years=1)
            older = [(_tz_naive(i), _num(v)) for i, v in closes.items() if _tz_naive(i) <= target]
            if older and older[-1][1]:
                vals["price_12m_ago"] = older[-1][1]
    except Exception:
        meta["notes"].append("가격 이력 없음")

    lag = FUND_DATA_PARAMS["band_lag_days_kr"] if is_krx else FUND_DATA_PARAMS["band_lag_days_us"]
    eps_pairs: List[Tuple[Any, float]] = []
    equity_pairs: List[Tuple[Any, float]] = []
    shares_now = _num(info.get("sharesOutstanding"))
    kr_blk: Dict[str, Any] = {}
    if is_krx:
        dart = _get_dart_client()
        if dart is None:
            meta["notes"].append("DART 사용 불가(API 키 또는 라이브러리 없음)")
        else:
            try:
                kr_blk = _dart_annual_block(dart, str(ticker).split(".")[0], {}, meta)
                eps_pairs, equity_pairs = kr_blk.get("eps_pairs", []), kr_blk.get("equity_pairs", [])
                for k, v in _kr_liquidity_and_survival(kr_blk, meta).items():
                    vals[k] = v
                if vals["company_roe"] is None:
                    L = kr_blk.get("latest", {})
                    eq = L.get("equity_parent") if L.get("equity_parent") is not None else L.get("equity")
                    if L.get("ni_parent") is not None and eq:
                        vals["company_roe"] = L["ni_parent"] / eq
                        vals["sustainable_roe"] = vals["company_roe"]
            except Exception as e:
                meta["notes"].append(f"DART 수집 실패: {type(e).__name__}")
    else:
        try:
            eps_pairs = _annual_pairs(tk.income_stmt, ["Diluted EPS", "Basic EPS"])
            equity_pairs = _annual_pairs(tk.balance_sheet, ["Stockholders Equity", "Common Stock Equity"])
        except Exception:
            meta["notes"].append("연간 재무제표 없음")
        for k, v in _us_liquidity_and_survival(tk, meta).items():
            vals[k] = v
        try:
            for k, v in _peer_stats_us(tk, info, ticker, meta).items():
                if k in vals or k in ("peer_median_per", "peer_top25_per", "peer_std_per",
                                      "peer_median_roe", "peer_median_fwd_growth"):
                    vals[k] = v
        except Exception:
            meta["notes"].append("피어 수집 실패")

    band = _lagged_ratio_stats(prices, eps_pairs, lag, FUND_DATA_PARAMS["band_years"], FUND_DATA_PARAMS["band_min_points"])
    if band:
        meta["band_points"], meta["band_span_years"] = band["n"], band["span_years"]
        meta["band_series"] = band.get("series", [])
        _why = _band_unreliable_reasons(band)
        if _why:
            # 이익 규모가 크게 바뀐 기간의 PER은 지금 회사와 비교할 기준이 못 된다 → 자기 밴드는 '데이터 없음'
            meta["band_excluded"] = "이익 규모 급변으로 자기 PER 밴드 제외: " + " · ".join(_why)
            meta["band_ref"] = {"mean": round(band["mean"], 1), "log_std": round(band["log_std"], 3)}
            meta["notes"].append(meta["band_excluded"])
        else:
            vals["self_5y_mean"], vals["self_5y_std"], vals["self_5y_min"] = band["mean"], band["std"], band["min"]
            vals["self_5y_log_std"] = band["log_std"]
    else:
        meta["notes"].append("자기 PER 밴드: 확정 EPS 이력 또는 가격 이력 부족 (월간 24개 미만)")
    if prices and eps_pairs:
        vals["eps_lag_now"] = _asof_fy(eps_pairs, prices[-1][0], lag)
        vals["eps_lag_12m_ago"] = _asof_fy(eps_pairs, prices[-1][0] - pd.DateOffset(years=1), lag)

    if shares_now and shares_now > 0 and equity_pairs:
        bv_pairs = [(d, v / shares_now) for d, v in equity_pairs]
        pb = _lagged_ratio_stats(prices, bv_pairs, lag, FUND_DATA_PARAMS["band_years"], FUND_DATA_PARAMS["band_min_points"])
        if pb:
            vals["pb_5y_mean"], vals["pb_5y_std"] = pb["mean"], pb["std"]
            meta["pb_points"] = pb["n"]
            meta["notes"].append("P/B 밴드: 연간 자본÷현재 주식수 근사 (자사주·증자 반영 오차 가능)")
        latest_bv = sorted(bv_pairs, key=lambda x: x[0])[-1][1]
        if px and latest_bv > 0:
            vals["current_pb"] = px / latest_bv
    if vals["current_pb"] is None:
        vals["current_pb"] = _num(info.get("priceToBook"))

    meta["missing"] = sorted(k for k, v in vals.items() if v is None)
    return {"values": vals, "meta": meta}


FUND_MODEL_PARAMS = {
    "mature_g": 0.03,        # 성숙기 영구성장률 g_성숙 (모델 상수)
    "position_cap": 0.05,    # 종목당 목표 비중 상한 (모델 상수)
    # 시장 위험프리미엄(ERP): 데이터가 아니라 모델 가정. 출처를 화면에 함께 표시한다.
    "market_erp_us": 0.0446,
    "market_erp_kr": 0.0487,
    "market_erp_source": "Damodaran 국가별 ERP 표 (2026년 1월 갱신: 미국 4.46% · 한국 4.87%)",
}



def _vpx(price: float, direction: str, is_krx: bool) -> float:
    """V 트랙 가격선 보정: 한국은 호가 단위 정수, 미국은 센트(소수 2자리) 유지."""
    p = adjust_krx_price(price, direction, is_krx)
    return int(p) if is_krx else round(float(p), 2)


def analyze_fundamental_track(
    ticker: str,
    current_price: float,
    earnings_output: Optional[Dict[str, Any]] = None,
    macro_input: Optional[Dict[str, Any]] = None,
    fund_override: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    펀더멘탈 분석 트랙 v1.0 메인 파이프라인 (F0 ~ F10 연산, 0~3차 buy_tranches 및 가격 자격 판정).
    * 순환논리 차단 3대 원칙 준수:
      1) 현재가로 목표 배수를 정당화하지 않음 (역사/피어/레짐 평균 중심)[cite: 4]
      2) G4(피크아웃)/G5(쇠퇴) 국면은 디레이팅 Cap 강제 적용 (가치 함정 차단)[cite: 4]
      3) 이론 성숙 PER은 시장 자기자본비용(k_e), 매수기준선은 사용자 요구수익률(r)로 분리 할인[cite: 4]
    """
    is_krx = ticker.upper().endswith((".KS", ".KQ"))

    # 실적 트랙(Track 2) 출력값 수신 (없을 경우 안전한 기본값 생성)
    if earnings_output is None:
        earnings_output = analyze_earnings_track(ticker)

    e_reg_info = earnings_output.get("regime", {})
    e_regime = e_reg_info.get("earnings_regime", "G3")          # "G1" ~ "G5" 또는 "NA"(판정 불가)
    regime_neutral = e_regime not in ("G1", "G2", "G3", "G4", "G5")   # 판정 불가·보류(NA)
    if regime_neutral:
        e_regime = "G3"   # 판정 불가는 화면에는 '판정 불가'로 표시하고, 계산에서만 중립(G3) 처리 (G5 차단으로 흐르지 않게)
    sector_type = e_reg_info.get("sector_type", "구조성장형")     # "사이클형" | "구조성장형" | "방어형" | "정책의존형"[cite: 2, 4]
    cycle_stage = e_reg_info.get("cycle_stage")                 # "C1" ~ "C4"[cite: 2, 4]
    proj = earnings_output.get("projections", {})
    eps_basis = proj.get("eps_basis", "none")
    _e1, _e2 = proj.get("eps_1y") or {}, proj.get("eps_2y") or {}
    _ttm = proj.get("ttm_eps")
    has_eps2 = False   # 1Y·2Y 미래 추정치가 모두 있을 때만 True (2Y 성장률·2Y 목표가 계산 조건)
    if _e1.get("base") is not None:
        _b = float(_e1["base"])
        eps_1y_dict = {
            "base": _b,
            "bull": float(_e1["bull"]) if _e1.get("bull") is not None else _b,
            "bear": float(_e1["bear"]) if _e1.get("bear") is not None else _b,
            "cycle_bear": _e1.get("cycle_bear"),
        }
        eps_2y_dict = {"base": float(_e2["base"]) if _e2.get("base") is not None else _b}
        has_eps2 = _e2.get("base") is not None
    elif _ttm is not None:
        # 참고용: 미래 추정치가 없으면 최근 확정 EPS(TTM) 하나로 계산한다 (시나리오 폭 없음)
        eps_1y_dict = {"base": float(_ttm), "bull": float(_ttm), "bear": float(_ttm), "cycle_bear": None}
        eps_2y_dict = {"base": float(_ttm)}
        eps_basis = "ttm_reference"
    else:
        raise ValueError("EPS 데이터 없음: 미래 추정치와 최근 확정 EPS를 모두 받지 못해 펀더멘탈(V) 트랙을 계산할 수 없습니다.")
    eps_floor = 1.0 if is_krx else 0.01   # 0 나눗셈 방지 하한 (원화 EPS는 1원, 달러 EPS는 1센트)
    certainty_adj = float(proj.get("certainty_adjustment") or 0.0)
    sus = earnings_output.get("sustainability_metrics", {})
    qual = earnings_output.get("quality_checks", {})
    th_status = earnings_output.get("thesis_driver", {}).get("status", "유효")

    # 거시 트랙(Track 4) 입력값 수신 (기본값: 한국/미국 10Y 금리 및 ERP)[cite: 4]
    if macro_input is None:
        macro_input = {}
    if "rf_10y" not in macro_input:   # 거시 트랙을 거치지 않고 호출된 경우 직접 받는다
        _rfi = fetch_risk_free_10y(is_krx)
        macro_input = dict(macro_input, rf_10y=_rfi["value"], rf_source=_rfi["source"], rf_as_of=_rfi["as_of"])

    # 펀더멘탈 입력: 실제 데이터(미국 yfinance · 한국 DART+yfinance). 못 받은 값은 None → '데이터 없음'으로 계산에서 빠진다.
    fdata = fetch_real_fundamental_inputs(ticker, current_price)
    fmeta = dict(fdata.get("meta", {}))
    f = dict(fdata.get("values", {}))
    # 모델 상수와 거시 트랙 입력 (데이터가 아니라 모델 설정값)
    _rf_in = macro_input.get("rf_10y")
    f["rf_10y"] = float(_rf_in) if _rf_in is not None else None      # None → 요구수익률·k_e 계산 불가('데이터 없음')
    _erp_in = macro_input.get("market_erp")
    f["market_erp"] = float(_erp_in) if _erp_in is not None else FUND_MODEL_PARAMS["market_erp_kr" if is_krx else "market_erp_us"]
    if f["rf_10y"] is None:
        fmeta.setdefault("missing", [])
        fmeta["missing"] = sorted(set(fmeta["missing"]) | {"rf_10y"})
    f["mature_g"] = FUND_MODEL_PARAMS["mature_g"]
    f["position_cap"] = FUND_MODEL_PARAMS["position_cap"]
    if fund_override:
        f.update(fund_override)

    eps_1y_base = float(eps_1y_dict["base"])
    eps_1y_bull = float(eps_1y_dict["bull"])
    eps_1y_bear = float(eps_1y_dict["bear"])
    eps_1y_cycle_bear = float(eps_1y_dict.get("cycle_bear") or (eps_1y_bear * 0.80))
    eps_2y_base = float(eps_2y_dict["base"])

    # 2Y EPS CAGR: 미래 추정치(1Y·2Y)가 모두 있을 때만 계산 (없으면 '데이터 없음' — 2% 같은 임의 하한을 만들지 않는다)
    consensus_2y_cagr: Optional[float] = None
    if has_eps2:
        consensus_2y_cagr = max(0.02, (eps_2y_base / max(eps_1y_base, eps_floor)) - 1.0)

    # 자기 PER 밴드는 후행(확정 EPS) 기준이므로 같은 기준의 현재 PER로 z-score를 낸다
    cur_per_band: Optional[float] = None
    if _ttm is not None and float(_ttm) > 0:
        cur_per_band = current_price / float(_ttm)

    # -----------------------------------------------------------------
    # F0. 멀티플 종류 선택 (분모 왜곡 방지 규칙)[cite: 4]
    # -----------------------------------------------------------------
    listing_years = f.get("listing_years")
    if f.get("is_financial_sector"):
        primary_mult, secondary_mult = "P/B", "배당수익률"
    elif e_regime == "G1" and eps_1y_base <= 0:
        primary_mult, secondary_mult = "NTM EV/매출", "목표 OPM 도달 시 내재 PER"
    elif listing_years is not None and listing_years < 3:
        primary_mult, secondary_mult = "피어 그룹 PER (μ, σ)", "PEG"
    elif sector_type == "사이클형":
        primary_mult, secondary_mult = "NTM PER", "P/B 밴드 (필수 교차)"
    else:
        primary_mult, secondary_mult = "NTM PER", "EV/EBITDA"

    current_per = round(current_price / max(eps_1y_base, eps_floor), 2)

    # -----------------------------------------------------------------
    # F1. 산업이 받을 수 있는 멀티플 (이론 성숙 PER, 상대 밴드, 레짐, 사이클 착시)[cite: 4]
    # -----------------------------------------------------------------
    # F1-1. 시장 자기자본비용 k_e = R_f + ERP * beta (요구수익률 r과 엄격 분리)[cite: 4]
    market_ke: Optional[float] = None
    theoretical_mature_per: Optional[float] = None
    if f.get("industry_beta") is not None and f.get("rf_10y") is not None:
        market_ke = f["rf_10y"] + f["market_erp"] * float(f["industry_beta"])
        if f.get("sustainable_roe") is not None:
            g_mat = min(f["mature_g"], market_ke - 0.02)
            roe_sust = max(float(f["sustainable_roe"]), g_mat + 0.02)
            theoretical_mature_per = round((1.0 - (g_mat / roe_sust)) / max(market_ke - g_mat, 0.01), 1)

    # F1-3. 산업 레짐 3단계 판정 (산업 실물 증거는 자동 수집 불가 → 데이터 없음)[cite: 4]
    if f.get("ind_rerate_evidences") is None or f.get("ind_derate_evidences") is None:
        ind_regime = NO_DATA
    elif f["ind_rerate_evidences"] >= 2 and f.get("ind_evidence_2q_confirmed"):
        ind_regime = "전환 확정 (상향)"
    elif f["ind_derate_evidences"] >= 2 and f.get("ind_evidence_2q_confirmed"):
        ind_regime = "전환 확정 (하향)"
    elif f["ind_rerate_evidences"] >= 2 or f["ind_derate_evidences"] >= 2:
        ind_regime = "전환 진행"
    else:
        ind_regime = "기존 밴드 유효"

    # -----------------------------------------------------------------
    # F3. 현재 멀티플 위치 (3중 z-score 입체 진단) — 각 z는 필요한 데이터가 있을 때만[cite: 4]
    # -----------------------------------------------------------------
    # 자기 PER 밴드: 중심 = 로그 평균의 지수(기하평균), 폭 = 로그 σ. k σ 지점은 μ·e^(kσ) (위아래 비대칭).
    self_mean: Optional[float] = None if f.get("self_5y_mean") is None else float(f["self_5y_mean"])
    self_lsd: Optional[float] = None
    if self_mean is not None and self_mean > 0:
        if f.get("self_5y_log_std") is not None:
            self_lsd = max(float(f["self_5y_log_std"]), 0.02)
        elif f.get("self_5y_std") is not None:   # 산술 σ만 주어진 경우(외부 입력) 근사 변환
            self_lsd = max(math.log(1.0 + max(float(f["self_5y_std"]), 0.0) / self_mean), 0.02)
    self_std: Optional[float] = None if self_lsd is None else self_mean * (math.exp(self_lsd) - 1.0)   # 표시용(+1σ − μ)

    def _sb(k: float, base: Optional[float] = None) -> Optional[float]:
        """자기 밴드 k σ 지점의 PER (base 생략 시 자기 기하평균 기준)."""
        b = self_mean if base is None else base
        return None if (b is None or self_lsd is None) else b * math.exp(k * self_lsd)

    # 시나리오 폭(Bull/Bear·2차선)용 로그 σ: 자기 밴드가 없으면(이익 급변 제외 포함) 피어 PER 분산으로 대체
    _pm, _ps = f.get("peer_median_per"), f.get("peer_std_per")
    peer_lsd: Optional[float] = (max(math.log(1.0 + float(_ps) / float(_pm)), 0.02)
                                 if (_pm is not None and _ps is not None and float(_pm) > 0) else None)
    scen_lsd: Optional[float] = self_lsd if self_lsd is not None else peer_lsd
    scen_lsd_src = "자기 밴드 σ" if self_lsd is not None else ("피어 PER σ" if peer_lsd is not None else NO_DATA)

    def _scen(k: float, base: float) -> Optional[float]:
        return None if scen_lsd is None else base * math.exp(k * scen_lsd)

    self_z: Optional[float] = None
    if self_mean is not None and self_lsd is not None and cur_per_band is not None and cur_per_band > 0:
        self_z = round((math.log(cur_per_band) - math.log(self_mean)) / self_lsd, 2)

    ind_rel_z: Optional[float] = None
    if None not in (f.get("ind_rel_current"), f.get("ind_rel_5y_mean"), f.get("ind_rel_5y_std")):
        ind_rel_z = round((f["ind_rel_current"] - f["ind_rel_5y_mean"]) / max(f["ind_rel_5y_std"], 0.05), 2)

    peer_median_per = f.get("peer_median_per")
    peer_top25_per = f.get("peer_top25_per")
    peer_std_per = f.get("peer_std_per")
    peer_rel_now: Optional[float] = None if peer_median_per is None else current_per / max(float(peer_median_per), 1.0)
    peer_rel_z: Optional[float] = None
    if peer_rel_now is not None and f.get("peer_rel_5y_mean") is not None and f.get("peer_rel_5y_std") is not None:
        peer_rel_z = round((peer_rel_now - f["peer_rel_5y_mean"]) / max(f["peer_rel_5y_std"], 0.05), 2)

    if peer_rel_z is not None and peer_rel_z <= -1.0:
        z_interp = "피어 대비 뚜렷한 저평가"
    elif self_z is None:
        z_interp = NO_DATA
    elif self_z <= -1.0:
        z_interp = "역사적 밴드 하단 저평가"
    elif self_z >= 2.0:
        z_interp = "역사적 극단 고점 경고 (+2σ 초과)"
    elif self_z >= 1.0:
        z_interp = "고평가 구간 (+1σ 상회)"
    else:
        z_interp = "정상 거래 밴드 구간"

    def _zs(z: Optional[float]) -> str:
        return NO_DATA if z is None else f"{z:+.2f}σ"

    if peer_rel_z is not None:
        _peer_txt = _zs(peer_rel_z)
    elif peer_rel_now is not None:
        _peer_txt = f"PER {peer_rel_now:.2f}배 (σ {NO_DATA})"
    else:
        _peer_txt = NO_DATA
    triple_z_summary = f"자기 밴드 {_zs(self_z)} | 산업 상대 {_zs(ind_rel_z)} | 피어 상대 {_peer_txt}" + (f" ({z_interp})" if z_interp != NO_DATA else "")

    # F1-4. 사이클형 정점 착시 감지기 (Spread_z = z_PB - z_PER >= 1.5)[cite: 4]
    cycle_type = None
    pb_per_z_spread = None
    is_cycle_peak_trap = False
    if sector_type == "사이클형":
        z_pb: Optional[float] = None
        if None not in (f.get("current_pb"), f.get("pb_5y_mean"), f.get("pb_5y_std")):
            z_pb = (f["current_pb"] - f["pb_5y_mean"]) / max(f["pb_5y_std"], 0.05)
        if z_pb is not None and self_z is not None:
            pb_per_z_spread = round(z_pb - self_z, 2)
        rev_status = earnings_output.get("revision_dynamics", {}).get("status", "보합")
        if (pb_per_z_spread is not None and pb_per_z_spread >= 1.5) or cycle_stage == "C3 정점 과열 (재고 급증 경고)":
            cycle_type = "Type 2 (정점 착시 / 디레이팅 경고)"
            is_cycle_peak_trap = True
        elif rev_status == "하향 지속" or cycle_stage == "C4 하강 (P·Q 동반 둔화)":
            cycle_type = "Type 3 (하강 확정)"
        elif pb_per_z_spread is not None:
            cycle_type = "Type 1 (확장 초중기)"
        else:
            cycle_type = f"{NO_DATA} (P/B·PER 밴드 부족으로 정점 착시 판정 불가)"

    # -----------------------------------------------------------------
    # F9. 재무 생존성 3지표 (Hard Survival Filter — F2-2 7번 요인 연계)[cite: 4]
    # -----------------------------------------------------------------
    fcf_ttm = f.get("fcf_ttm_krw")
    net_cash = f.get("net_cash_krw")
    runway_warn = False
    if fcf_ttm is None:
        runway_str = NO_DATA
    elif fcf_ttm >= 0:
        runway_str = "FCF 흑자 (해당 없음)"
    elif net_cash is None:
        runway_str = f"{NO_DATA} (FCF 적자이나 순현금 미확인)"
    elif net_cash > 0 and (net_cash / abs(fcf_ttm)) >= 2.0:
        runway_str = f"{net_cash / abs(fcf_ttm):.1f}년 (순현금 방어)"
    else:
        rw_yrs = max(0.0, net_cash / abs(fcf_ttm)) if net_cash > 0 else 0.0
        runway_str = f"{rw_yrs:.1f}년 (위험 < 2.0년)"
        runway_warn = True

    debt_mat_ratio = f.get("short_term_debt_ratio")
    debt_warn = bool(debt_mat_ratio is not None and debt_mat_ratio > 1.00)
    dilution_yoy = f.get("share_dilution_yoy")
    dilution_warn = bool(dilution_yoy is not None and dilution_yoy > 0.05)
    survival_warning = bool(runway_warn or debt_warn or dilution_warn)
    survival_checked = sum(1 for v in (fcf_ttm, debt_mat_ratio, dilution_yoy) if v is not None)

    # -----------------------------------------------------------------
    # F2. 피어 대비 프리미엄·디스카운트 정당화 8대 요인 비교표[cite: 4]
    # -----------------------------------------------------------------
    peer_factors = []
    # 1) 성장 격차 (선행/후행 EPS 증가율: 기업·피어 모두 같은 야후 기준)[cite: 4]
    _cg, _pg = f.get("company_fwd_growth"), f.get("peer_median_fwd_growth")
    if _cg is None or _pg is None:
        peer_factors.append({"no": 1, "factor": "성장 격차 (선행/후행 EPS, 피어 대비)", "diff": NO_DATA, "score": 0})
    else:
        g_diff = float(_cg) - float(_pg)
        f1_s = 1 if g_diff >= 0.03 else (-1 if g_diff <= -0.03 else 0)
        peer_factors.append({"no": 1, "factor": "성장 격차 (선행/후행 EPS, 피어 대비)", "diff": f"{g_diff*100:+.1f}%p", "score": f1_s})

    # 2) 수익성·효율 격차 (ROE: 기업·피어 모두 같은 야후 기준)[cite: 4]
    _cr, _pr = f.get("company_roe"), f.get("peer_median_roe")
    if _cr is None or _pr is None:
        peer_factors.append({"no": 2, "factor": "수익성·효율 격차 (ROE, 피어 대비)", "diff": NO_DATA, "score": 0})
    else:
        roic_diff = float(_cr) - float(_pr)
        f2_s = 1 if roic_diff >= 0.03 else (-1 if roic_diff <= -0.03 else 0)
        peer_factors.append({"no": 2, "factor": "수익성·효율 격차 (ROE, 피어 대비)", "diff": f"{roic_diff*100:+.1f}%p", "score": f2_s})

    # 3) 영업레버리지 (ΔOPM)[cite: 4]
    op_lev_spread, rpo_spread = None, None   # 데이터가 없으면 None 유지 (아래 F5 4축에서 '데이터 없음' 처리)
    _inc_opm, _avg_opm = sus.get("incremental_opm"), sus.get("average_opm")
    if _inc_opm is None or _avg_opm is None:
        peer_factors.append({"no": 3, "factor": "영업레버리지 (증분-평균 OPM)", "diff": NO_DATA, "score": 0})
    else:
        op_lev_spread = float(_inc_opm) - float(_avg_opm)
        f3_s = 1 if op_lev_spread > 0.02 else (-1 if op_lev_spread < -0.02 else 0)
        peer_factors.append({"no": 3, "factor": "영업레버리지 (증분-평균 OPM)", "diff": f"{op_lev_spread*100:+.1f}%p", "score": f3_s})

    # 4) 이익 지속성 (RPO·수주)[cite: 4]
    _rpo, _sy = sus.get("rpo_yoy_growth"), sus.get("sales_yoy_growth")
    if _rpo is None or _sy is None:
        peer_factors.append({"no": 4, "factor": "이익 지속성 (RPO - 매출 성장)", "diff": NO_DATA, "score": 0})
    else:
        rpo_spread = float(_rpo) - float(_sy)
        f4_s = 1 if rpo_spread > 0.0 else (-1 if rpo_spread < -0.05 else 0)
        peer_factors.append({"no": 4, "factor": "이익 지속성 (RPO - 매출 성장)", "diff": f"{rpo_spread*100:+.1f}%p", "score": f4_s})

    # 5) 가이던스 신뢰도[cite: 4]
    if str(proj.get("certainty_label", "")).startswith(NO_DATA):
        peer_factors.append({"no": 5, "factor": "경영진 가이던스 신뢰도", "diff": NO_DATA, "score": 0})
    else:
        f5_s = 1 if certainty_adj < 0 else (-1 if certainty_adj > 0 else 0)
        peer_factors.append({"no": 5, "factor": "경영진 가이던스 신뢰도", "diff": f"할인율 {certainty_adj*100:+.0f}%p", "score": f5_s})

    # 6) 이익의 질 (현금 전환)[cite: 4]
    w_cnt = int(qual.get("warning_count", 0))
    if int(qual.get("checked_count", 0)) == 0:
        peer_factors.append({"no": 6, "factor": "이익의 질 (회계·현금전환)", "diff": NO_DATA, "score": 0})
    else:
        f6_s = 1 if w_cnt == 0 else (-1 if w_cnt >= 2 else 0)
        peer_factors.append({"no": 6, "factor": "이익의 질 (회계·현금전환)", "diff": f"경고 {w_cnt}개", "score": f6_s})

    # 7) 재무 생존성[cite: 4]
    if survival_checked == 0 and net_cash is None:
        peer_factors.append({"no": 7, "factor": "재무 생존성 (순현금·런웨이)", "diff": NO_DATA, "score": 0})
    else:
        f7_s = -1 if survival_warning else (1 if (net_cash is not None and net_cash > 0) else 0)
        peer_factors.append({"no": 7, "factor": "재무 생존성 (순현금·런웨이)", "diff": "무경고" if not survival_warning else "경고 발동", "score": f7_s})

    # 8) 주주환원·지배구조 (소각/밸류업 우위는 자동 수집 불가 → 희석 경고만 반영)[cite: 4]
    _sra = f.get("shareholder_return_advantage")
    if dilution_warn or _sra is not None:
        f8_s = -1 if dilution_warn else int(_sra)
        peer_factors.append({"no": 8, "factor": "주주환원·지배구조 (소각/밸류업)", "diff": "우위(+1)" if f8_s > 0 else ("열위(-1)" if f8_s < 0 else "동등(0)"), "score": f8_s})
    else:
        peer_factors.append({"no": 8, "factor": "주주환원·지배구조 (소각/밸류업)", "diff": NO_DATA, "score": 0})

    net_adv_score = sum(item["score"] for item in peer_factors)
    n_factors_scored = sum(1 for item in peer_factors if item["diff"] != NO_DATA)
    if peer_rel_now is None:
        if net_adv_score >= 2:
            peer_just = f"펀더멘탈 순우위 (피어 멀티플 비교 {NO_DATA})"
        elif net_adv_score <= -2:
            peer_just = f"펀더멘탈 순열위 (피어 멀티플 비교 {NO_DATA})"
        else:
            peer_just = f"피어 멀티플 비교 {NO_DATA}"
    elif net_adv_score >= 2 and (peer_rel_now <= 1.0 or (peer_rel_z is not None and peer_rel_z <= -1.0)):
        peer_just = "📈 멀티플 과소평가 후보 (펀더멘탈 순우위이나 피어 대비 할인)"
    elif net_adv_score >= 2 and peer_rel_now > 1.0:
        peer_just = "✅ 정당한 프리미엄 (순우위 뒷받침)"
    elif net_adv_score <= -2 and (peer_rel_now >= 1.0 or (peer_rel_z is not None and peer_rel_z >= 1.0)):
        peer_just = "📉 멀티플 과대평가 후보 (펀더멘탈 열위이나 피어 대비 할증)"
    elif net_adv_score <= -2:
        peer_just = "✅ 정당한 디스카운트 (가치 함정 주의)"
    else:
        peer_just = "피어 평균 부합 구간"

    # -----------------------------------------------------------------
    # F4. 12개월 주가 변화 분해 — 후행 EPS 기준 2층(이익 + 배수). 시장·산업층은 업종 PER 이력이 없어 '데이터 없음'[cite: 4]
    # -----------------------------------------------------------------
    p_0 = f.get("price_12m_ago")
    e_lag_now, e_lag_0 = f.get("eps_lag_now"), f.get("eps_lag_12m_ago")
    log_price_ret = log_eps_contrib = log_mult_contrib = None
    layer_market = layer_industry = layer_company = None
    if p_0 is not None and float(p_0) > 0:
        log_price_ret = math.log(max(current_price, 1.0) / float(p_0))
        if e_lag_now is not None and e_lag_0 is not None and e_lag_now > 0 and e_lag_0 > 0:
            log_eps_contrib = math.log(float(e_lag_now) / float(e_lag_0))
            log_mult_contrib = log_price_ret - log_eps_contrib
    mkt_layers = [f.get("mkt_per_12m_ago"), f.get("mkt_per_now"), f.get("ind_per_12m_ago"), f.get("ind_per_now")]
    if log_mult_contrib is not None and all(v is not None and v > 0 for v in mkt_layers):
        layer_market = math.log(f["mkt_per_now"] / f["mkt_per_12m_ago"])
        layer_industry = math.log((f["ind_per_now"] / f["mkt_per_now"]) / (f["ind_per_12m_ago"] / f["mkt_per_12m_ago"]))
        layer_company = log_mult_contrib - (layer_market + layer_industry)

    if layer_market is not None:
        abs_layers = [("시장층(금리·ERP)", abs(layer_market), layer_market), ("산업층(업종 쏠림/배수)", abs(layer_industry), layer_industry), ("기업층(개별 프리미엄)", abs(layer_company), layer_company)]
        abs_layers.sort(key=lambda x: x[1], reverse=True)
        top_layer_name, _, top_layer_val = abs_layers[0]
        if "산업층" in top_layer_name and top_layer_val < 0 and log_eps_contrib > 0:
            dom_driver = "⚡ 업종 센티먼트 동반 매도 (이익은 성장했으나 산업 배수 압축)"
        elif "시장층" in top_layer_name and top_layer_val < 0:
            dom_driver = "⚡ 시장 금리·ERP 상승에 따른 매크로 배수 압축 (기업 훼손 아님)"
        elif "기업층" in top_layer_name and top_layer_val > 0 and e_regime in ["G1", "G2"]:
            dom_driver = "🏛️ 기업 고유 구조적 리레이팅 (RPO·영업레버리지 개선 반영)"
        elif "기업층" in top_layer_name and top_layer_val < 0 and e_regime in ["G4", "G5"]:
            dom_driver = "🏛️ 피크아웃·성장 둔화에 따른 구조적 디레이팅"
        else:
            dom_driver = f"{top_layer_name} 주도 배수 변화 ({top_layer_val*100:+.1f}%)"
    elif log_mult_contrib is not None:
        if abs(log_eps_contrib) >= abs(log_mult_contrib):
            dom_driver = f"📊 이익 변화 주도 (EPS {log_eps_contrib*100:+.1f}% · 배수 {log_mult_contrib*100:+.1f}%, 후행 EPS 기준)"
        else:
            dom_driver = f"📊 배수 변화 주도 (배수 {log_mult_contrib*100:+.1f}% · EPS {log_eps_contrib*100:+.1f}%, 후행 EPS 기준)"
    else:
        dom_driver = NO_DATA

    # -----------------------------------------------------------------
    # F7-1. 요구수익률 r 산출 (F5-2 역산 및 F7 매수기준선 공통 사용)[cite: 4]
    # -----------------------------------------------------------------
    sector_prem_map = {"방어형": 0.04, "구조성장형": 0.07, "정책의존형": 0.10, "사이클형": 0.12}
    sector_prem = sector_prem_map.get(sector_type, 0.07)
    req_r = None if f.get("rf_10y") is None else round(f["rf_10y"] + sector_prem + certainty_adj, 4)

    # -----------------------------------------------------------------
    # F6. 실적 국면(G1~G5) 연동 Base·Bull·Bear 멀티플 결정 (가치 함정 차단)[cite: 4]
    #     필요한 멀티플 근거(자기 밴드·피어 통계·이론 성숙 PER)가 없으면 해당 값은 None → 가격선 '데이터 없음'
    # -----------------------------------------------------------------
    is_value_trap_blocked = False
    base_mult: Optional[float] = None
    bull_mult: Optional[float] = None
    bear_mult: Optional[float] = None
    _theo11 = None if theoretical_mature_per is None else theoretical_mature_per * 1.1
    if e_regime == "G1":
        mult_rule_desc = "G1 초기 가속: 피어 상위 25% 멀티플 채택 (초기 적자/저수익 자기밴드 왜곡 보정)"
        if peer_top25_per is not None:
            base_mult = round(float(peer_top25_per), 1)
            bull_mult = None if peer_std_per is None else round(base_mult + float(peer_std_per), 1)
            bear_mult = None if peer_median_per is None else round(float(peer_median_per), 1)
    elif e_regime == "G2":
        mult_rule_desc = "G2 우량 지속 성장: 자기 5년 역사적 평균(μ) 중심 표준 밴드 적용"
        if self_mean is not None:
            base_mult = round(self_mean, 1)
            if self_lsd is not None:
                bull_mult = round(_sb(1.0), 1)
                bear_mult = round(max(5.0, _sb(-1.0)), 1)
        elif peer_median_per is not None:
            base_mult = round(float(peer_median_per), 1)
            mult_rule_desc = f"G2 우량 지속 성장: 자기 밴드 {NO_DATA} → 피어 중앙값 PER 대체, 폭은 피어 PER σ"
            if scen_lsd is not None:
                bull_mult = round(_scen(1.0, base_mult), 1)
                bear_mult = round(max(5.0, _scen(-1.0, base_mult)), 1)
    elif e_regime == "G3":
        # 이론 성숙 PER(1단계 고든: 내년부터 영구 g_성숙 성장)은 초과성장 기간을 0으로 보므로 우량 성숙주도 13~16배로
        # 눌러 버린다. 그래서 상한이 아니라 참고값으로만 두고, 실제 관측 배수(자기 평균·피어 중앙값)가 둘 다 없을 때만 쓴다.
        _g3 = [x for x in (self_mean, None if peer_median_per is None else float(peer_median_per)) if x is not None]
        if _g3:
            mult_rule_desc = "G3 성숙/현금회수: min(자기 평균 PER, 피어 중앙값 PER) 적용 (있는 값만 사용 · 이론 성숙 PER은 참고값)"
        elif _theo11 is not None:
            _g3 = [_theo11]
            mult_rule_desc = f"G3 성숙/현금회수: 자기 밴드·피어 PER {NO_DATA} → 이론 성숙 PER×1.1 대체 사용 (보수적 하한 성격)"
        else:
            mult_rule_desc = "G3 성숙/현금회수: min(자기 평균 PER, 피어 중앙값 PER) 적용"
        if _g3:
            base_mult = round(min(_g3), 1)
            if scen_lsd is not None:
                bull_mult = round(_scen(0.5, base_mult), 1)
                bear_mult = round(max(4.0, _scen(-1.0, base_mult)), 1)
                if self_lsd is None:
                    mult_rule_desc += " · Bull/Bear 폭은 피어 PER σ (자기 밴드 없음)"
    elif e_regime == "G4":
        is_value_trap_blocked = True
        mult_rule_desc = "🚨 G4 피크아웃 디레이팅 Cap 발동: 과거 평균 회귀 불인정 (가치 함정 원천 차단)"
        _g4 = []
        if self_mean is not None and self_lsd is not None:
            _g4.append(_sb(-1.0))
        if theoretical_mature_per is not None:
            _g4.append(theoretical_mature_per)
        if _g4:   # 현재 PER만으로는 목표 배수를 만들지 않는다 (현재가로 목표를 정당화하는 순환논리 차단)
            base_mult = round(min(_g4 + [current_per]), 1)
            bull_mult = round(current_per, 1)
            _g4b = []
            if self_mean is not None and self_lsd is not None:
                _g4b.append(_sb(-2.0))
            if f.get("self_5y_min") is not None:
                _g4b.append(float(f["self_5y_min"]))
            if _g4b:
                bear_mult = round(min(_g4b), 1)
    else:  # G5 구조적 쇠퇴[cite: 4]
        is_value_trap_blocked = True
        base_mult = 0.0
        bull_mult = 0.0
        bear_mult = 0.0
        mult_rule_desc = "🚨 G5 구조적 쇠퇴: 목표 멀티플 산출 중단 및 가격 자격 하드 차단"

    if regime_neutral:
        mult_rule_desc = "실적 국면 판정 보류 → 중립 규칙 적용 · " + mult_rule_desc.replace("G3 성숙/현금회수: ", "")
    scenario_width_missing = False
    if base_mult is None:
        mult_rule_desc += f" — 필요한 멀티플 근거 {NO_DATA} (목표 배수 산출 불가)"
    else:
        if bull_mult is None:
            bull_mult, scenario_width_missing = base_mult, True
        if bear_mult is None:
            bear_mult, scenario_width_missing = base_mult, True
        if scenario_width_missing:
            mult_rule_desc += " — Bull/Bear 배수 폭 근거 없음 (Base와 동일 처리)"

    base_independence_passed = bool(e_regime != "G5")

    # -----------------------------------------------------------------
    # F5. 재평가(리레이팅·디레이팅) 4축 진단 및 5년 성숙기 수렴 내재 성장률 역산[cite: 4]
    # -----------------------------------------------------------------
    # 종착 배수: F6에서 채택한 Base 배수(실제 관측 배수 근거)를 쓴다. 없을 때만 이론 성숙 PER·업종 평균으로 대체.
    # (이론 성숙 PER을 종착으로 쓰면 우량 성숙주의 내재 성장률이 30%대로 과장된다)
    _term_c = [x for x in (theoretical_mature_per, f.get("ind_5y_mean")) if x is not None]
    per_terminal: Optional[float] = base_mult if (base_mult is not None and base_mult > 0) else (max(_term_c) if _term_c else None)
    terminal_per_basis = ("채택 Base 배수" if (base_mult is not None and base_mult > 0)
                          else ("이론 성숙 PER·업종 평균 중 큰 값 (Base 배수 없음)" if _term_c else NO_DATA))
    implied_5y_cagr: Optional[float] = None
    if per_terminal is not None and req_r is not None:
        implied_5y_cagr = round(((current_price * ((1.0 + req_r) ** 5)) / max(eps_1y_base * per_terminal, 1.0)) ** (1.0 / 5.0) - 1.0, 4)

    ax_dur = NO_DATA if rpo_spread is None else ("강" if rpo_spread > 0 else ("중" if rpo_spread >= -0.05 else "약"))
    ax_inc = NO_DATA if op_lev_spread is None else ("강" if op_lev_spread > 0 else "약")
    if str(proj.get("certainty_label", "")).startswith(NO_DATA):
        ax_cert = NO_DATA
    else:
        ax_cert = "강" if (certainty_adj < 0 and w_cnt == 0) else ("약" if (certainty_adj > 0 or w_cnt >= 2) else "중")

    _hist_max = f.get("historical_max_5y_cagr")
    _can_cmp = implied_5y_cagr is not None and consensus_2y_cagr is not None
    if implied_5y_cagr is None and self_z is None:
        ax_priced = NO_DATA
    elif _can_cmp and self_z is not None and implied_5y_cagr < consensus_2y_cagr and self_z <= 0.5:
        ax_priced = "낮음"  # 유리 (과소평가)[cite: 4]
    elif (_can_cmp and implied_5y_cagr > max(consensus_2y_cagr, _hist_max if _hist_max is not None else consensus_2y_cagr)) or (self_z is not None and self_z >= 1.0):
        ax_priced = "높음"  # 과열[cite: 4]
    elif not _can_cmp or self_z is None:
        ax_priced = NO_DATA  # 판정에 필요한 근거 일부 없음 (중립으로 단정하지 않는다)
    else:
        ax_priced = "중립"

    strong_cnt = sum(1 for x in [ax_dur, ax_inc, ax_cert] if x == "강")
    if e_regime in ["G4", "G5"] or (ax_priced == "높음" and self_z is not None and self_z >= 1.0):
        rerate_dir = "⬇ 수축 위험"
    elif strong_cnt >= 2 and ax_priced == "낮음":
        rerate_dir = "⬆ 확장 여지"
    elif ax_priced == NO_DATA:
        rerate_dir = NO_DATA
    else:
        rerate_dir = "➡ 유지"

    # -----------------------------------------------------------------
    # F7-2 & F7-3. 사이클 정점 보호 및 0~3차 매수기준선(buy_tranches) 생성[cite: 4]
    # -----------------------------------------------------------------
    if is_cycle_peak_trap:
        # ★ 사이클 정점 착시 보호 규칙: 0.5 * Base + 0.5 * CycleBear 정상화 EPS 강제 사용[cite: 4]
        eps_eff_1y = 0.5 * eps_1y_base + 0.5 * eps_1y_cycle_bear
    else:
        eps_eff_1y = eps_1y_base

    div_d = float(f["dividend_yield"]) if f.get("dividend_yield") is not None else 0.0   # 없으면 0%로 계산(보수적)
    disc_1y = None if req_r is None else max(0.02, 1.0 + req_r - div_d)
    disc_1y_floor = None if req_r is None else max(0.02, 1.0 + (req_r - 0.03) - div_d)

    t0_price = t1_price = t2_price = t3_price = None
    if e_regime == "G5":
        t0_price, t1_price, t2_price, t3_price = 0, 0, 0, 0
    elif base_mult is not None and disc_1y is not None:   # 요구수익률(무위험수익률 필요)이 없으면 매수선을 할인할 수 없다
        raw_t0 = (eps_eff_1y * base_mult) / disc_1y_floor
        raw_t1 = (eps_eff_1y * base_mult) / disc_1y
        raw_t2 = raw_t1 * 0.90
        if scen_lsd is not None:
            raw_t2 = min(raw_t2, (eps_eff_1y * max(1.0, _scen(-0.5, base_mult))) / disc_1y)
        raw_bear_line = (eps_1y_bear * bear_mult) / disc_1y
        raw_t3 = max(raw_bear_line, raw_t1 * 0.80)

        # 상위 차수와 하위 차수 최소 5% 간격 강제 및 KRX 호가 내림 처리[cite: 4]
        t1_price = _vpx(raw_t1, "down", is_krx)
        t0_price = _vpx(max(raw_t0, t1_price * 1.05), "down", is_krx)
        t2_price = _vpx(min(raw_t2, t1_price * 0.92), "down", is_krx)
        t3_price = _vpx(min(raw_t3, t2_price * 0.92), "down", is_krx)
    has_lines = t1_price is not None

    # F7-4. 시나리오 수익률 매트릭스 · 확률가중 기대수익률 E[R] · MDD Ratio[cite: 4]
    r_bull = r_base = r_bear = r_cbear = None
    exp_return = mdd_ratio = downside_convexity = None
    p_bear_1y = None
    if base_mult is not None:
        p_bull_1y = eps_1y_bull * bull_mult
        p_base_1y = eps_eff_1y * base_mult
        p_bear_1y = eps_1y_bear * bear_mult
        p_cbear_1y = eps_1y_cycle_bear * bear_mult

        r_bull = (p_bull_1y / max(current_price, 1.0)) - 1.0 + div_d
        r_base = (p_base_1y / max(current_price, 1.0)) - 1.0 + div_d
        r_bear = (p_bear_1y / max(current_price, 1.0)) - 1.0 + div_d
        r_cbear = (p_cbear_1y / max(current_price, 1.0)) - 1.0 + div_d

        if sector_type == "사이클형":
            # 사이클형: Base 40%, Bull 20%, Bear 20%, CycleBear 20%[cite: 2, 4]
            exp_return = 0.40 * r_base + 0.20 * r_bull + 0.20 * r_bear + 0.20 * r_cbear
            worst_loss = abs(min(r_bear, r_cbear, -0.05))
        else:
            # 일반: Base 50%, Bull 25%, Bear 25%[cite: 2, 4]
            exp_return = 0.50 * r_base + 0.25 * r_bull + 0.25 * r_bear
            worst_loss = abs(min(r_bear, -0.05))

        mdd_ratio = round(exp_return / max(worst_loss, 0.01), 2)
        downside_convexity = round(abs(current_price - p_bear_1y) / max(abs(p_bull_1y - current_price), 1.0), 2)

    # 0차 개방 여부: G1/G2 이고 MDD Ratio >= 0.30 일 때만 0차 개방[cite: 4]
    pos_cap = float(f["position_cap"])
    if survival_warning:
        pos_cap *= 0.5  # F9 런웨이 경고 시 목표 비중 50% 축소[cite: 4]

    open_t0 = bool(e_regime in ["G1", "G2"] and mdd_ratio is not None and mdd_ratio >= 0.30)
    buy_tranches = []
    if e_regime != "G5" and has_lines:
        if open_t0:
            buy_tranches.append({"tranche": 0, "price": t0_price, "weight": round(pos_cap * 0.10, 4), "label": "0차 탐색선 (r_floor 보험)"})
            w_t1 = round(pos_cap * 0.30, 4)
        else:
            w_t1 = round(pos_cap * 0.40, 4)
        buy_tranches.append({"tranche": 1, "price": t1_price, "weight": w_t1, "label": "1차 기본 매수기준선 (Base 1Y)"})
        buy_tranches.append({"tranche": 2, "price": t2_price, "weight": round(pos_cap * 0.30, 4), "label": "2차 안전마진 강화선 (-10%)"})
        buy_tranches.append({"tranche": 3, "price": t3_price, "weight": round(pos_cap * 0.30, 4), "label": "3차 Bear 시나리오 방어선"})

    # -----------------------------------------------------------------
    # F8. 매도목표가(sell_target) 4대 독립 앵커 및 보유기간 상한 검증[cite: 4]
    # -----------------------------------------------------------------
    # 앵커 1: 컨센서스 목표가 (yfinance targetMeanPrice)[cite: 4]
    anchor1_price = anchor1_per = None
    if f.get("consensus_target_price") is not None:
        anchor1_price = _vpx(f["consensus_target_price"], "down", is_krx)
        anchor1_per = round(anchor1_price / max(eps_1y_base, eps_floor), 1)

    # 앵커 2: 역사적 밴드 상단 (단, G3/G4는 자기 평균 μ_self로 캡)[cite: 4]
    anchor2_per = anchor2_price = None
    if self_mean is not None and (e_regime in ["G3", "G4"] or self_lsd is not None):
        anchor2_per = round(self_mean if e_regime in ["G3", "G4"] else _sb(1.0), 1)
        anchor2_price = _vpx(anchor2_per * eps_1y_base, "down", is_krx)

    # 앵커 3: PEG 적정 배수 (한국 0.8 / 미국 1.0 / G1 초기 가속 1.2, 산업 +2σ 및 50배 상한)[cite: 4]
    peg_target = 1.2 if e_regime == "G1" else (0.8 if is_krx else 1.0)
    anchor3_per = anchor3_price = None
    anchor3_peg_raw: Optional[float] = None
    anchor3_floored = False
    if consensus_2y_cagr is not None:
        _caps = [50.0]
        if f.get("ind_5y_mean") is not None and f.get("ind_5y_std") is not None:
            _caps.append(float(f["ind_5y_mean"]) + 2.0 * float(f["ind_5y_std"]))
        anchor3_per = round(min([peg_target * (100.0 * consensus_2y_cagr)] + _caps), 1)
        anchor3_peg_raw = anchor3_per
        if base_mult is not None and base_mult > anchor3_per:
            anchor3_per = base_mult          # Base 배수 하한 적용 → 이 경우 PEG는 독립 근거가 아님(화면에 명시)
            anchor3_floored = True
        anchor3_price = _vpx(anchor3_per * eps_1y_base, "down", is_krx)

    # 앵커 4: 피어 상위 25% 배수 (F2-2 순우위 >= +2일 때만 유효)[cite: 4]
    anchor4_valid = net_adv_score >= 2
    anchor4_per = anchor4_price = None
    if peer_top25_per is not None:
        anchor4_per = round(float(peer_top25_per), 1)
        anchor4_price = _vpx(anchor4_per * eps_1y_base, "down", is_krx)

    # 최종 채택 매도목표가: 독립 앵커 2와 앵커 3의 평균 (있는 앵커만 사용)[cite: 4]
    _indep = [(n, p) for n, p in (("2", anchor2_per), ("3", anchor3_per)) if p is not None]
    adopted_sell_per = round(sum(p for _, p in _indep) / len(_indep), 1) if _indep else None
    ttm_prefix = "[참고용: 최근 확정 EPS 기준 · 미래 시나리오 없음] " if eps_basis == "ttm_reference" else ""
    if adopted_sell_per is None:
        sell_target_1y = None
        sell_target_formula = f"{ttm_prefix}{NO_DATA} (독립 앵커 2·3을 만들 근거가 없음)"
    else:
        sell_target_1y = _vpx(adopted_sell_per * eps_1y_base, "down", is_krx)
        _a2_is_mu = e_regime in ["G3", "G4"]
        _notes = []
        if _a2_is_mu and anchor2_per is not None:
            _notes.append("앵커2=자기 평균 μ(G3·G4 상한)")
        if anchor3_floored:
            _notes.append(f"앵커3=PEG {anchor3_peg_raw:.1f}배가 Base 배수보다 낮아 Base {base_mult:.1f}배로 대체")
        _anch_txt = ("앵커2·3 평균" if len(_indep) == 2 else f"앵커{_indep[0][0]}만 사용 (다른 앵커 {NO_DATA})") + \
                    (" · " + " · ".join(_notes) if _notes else "")
        sell_target_formula = f"{ttm_prefix}{fmt_price(sell_target_1y, is_krx)} = 1Y Base EPS {fmt_price(eps_1y_base, is_krx)} × 목표 PER {adopted_sell_per:.1f}배 ({_anch_txt})"

    # 2Y 매도목표가 및 보유기간 상한 경고 (2Y 목표가 < 1Y 목표가 역전 여부) — 2Y 추정치가 있을 때만[cite: 4]
    _p2 = [x for x in (theoretical_mature_per, base_mult) if x is not None]
    sell_target_2y = None
    if _p2 and has_eps2:
        sell_target_2y = _vpx(max(_p2) * eps_2y_base, "down", is_krx)
    holding_cap_warn = None if (sell_target_2y is None or sell_target_1y is None) else bool(sell_target_2y < sell_target_1y)

    def _a_per(v):
        return NO_DATA if v is None else f"{v:.1f}배"

    def _a_px(v):
        return NO_DATA if v is None else f"{fmt_price(v, is_krx)}"

    anchors_table = [
        {"id": "앵커 1", "name": "증권사 컨센서스 평균", "target_per": _a_per(anchor1_per), "target_price": _a_px(anchor1_price), "indep": "반종속 (참고 교차검증)" if anchor1_price is not None else NO_DATA},
        {"id": "앵커 2",
         "name": ("역사적 밴드 평균 (μ · G3·G4는 μ+1σ 대신 μ로 상한)" if e_regime in ["G3", "G4"] else "역사적 밴드 상단 (μ+1σ)"),
         "target_per": _a_per(anchor2_per), "target_price": _a_px(anchor2_price),
         "indep": NO_DATA if anchor2_per is None else ("✅ 독립 (채택 반영 · Base 배수와 같은 값)" if (base_mult is not None and abs(anchor2_per - base_mult) < 0.05)
                                                       else "✅ 독립 (채택 반영)")},
        {"id": "앵커 3", "name": f"PEG 적정 배수 (PEG={peg_target})", "target_per": _a_per(anchor3_per), "target_price": _a_px(anchor3_price),
         "indep": NO_DATA if anchor3_per is None else (f"⚠️ 독립 아님 — PEG 산출 {anchor3_peg_raw:.1f}배 < Base, Base 배수로 대체 (채택 반영)" if anchor3_floored
                                                       else "✅ 독립 (채택 반영)")},
        {"id": "앵커 4", "name": "피어 상위 25% 배수", "target_per": _a_per(anchor4_per), "target_price": _a_px(anchor4_price),
         "indep": NO_DATA if anchor4_per is None else ("유효 참고" if anchor4_valid else "미충족 (순우위<2)")},
    ]

    # -----------------------------------------------------------------
    # 4장. 트랙 종합 판정: 가격 자격 (QUALIFIED / CONDITIONAL / DISQUALIFIED / NA)[cite: 4]
    # -----------------------------------------------------------------
    if e_regime == "G5" or th_status == "붕괴":
        pq_status = "DISQUALIFIED"
        pq_icon = "🔴"
        pq_reason = "실적 국면 G5(구조적 쇠퇴) 또는 드라이버 붕괴로 가격 자격 하드 차단"
    elif not has_lines:
        pq_status = "NA"
        pq_icon = "⚪"
        pq_reason = (f"무위험수익률(10년 국채) {NO_DATA}로 요구수익률을 계산할 수 없어 매수기준선을 만들 수 없습니다"
                     if req_r is None else
                     f"목표 배수를 만들 근거(자기 PER 밴드·피어 PER·이론 성숙 PER)가 모두 없어({NO_DATA}) 가격 자격을 판정할 수 없습니다")
    elif mdd_ratio < 0.15:
        pq_status = "DISQUALIFIED"
        pq_icon = "🔴"
        pq_reason = f"MDD Ratio({mdd_ratio:.2f}) 최소 기준(0.15) 미달로 위험 대비 기대수익 부적격"
    elif current_price <= t1_price and mdd_ratio >= 0.30 and exp_return >= req_r and not survival_warning and self_z is not None and self_z < 1.0:
        pq_status = "QUALIFIED"
        pq_icon = "🟢"
        gap_pct = (current_price / t1_price - 1.0) * 100.0
        pq_reason = f"현재가({fmt_price(current_price, is_krx)})가 1차 매수기준선({fmt_price(t1_price, is_krx)}, {gap_pct:+.1f}%) 이하 및 MDD Ratio {mdd_ratio:.2f}로 가격 자격 충족"
    elif (open_t0 and current_price <= t0_price) or (current_price <= t1_price and mdd_ratio >= 0.15):
        pq_status = "CONDITIONAL"
        pq_icon = "🟡"
        pq_reason = f"조건부 가격 자격 — 0차 탐색선({fmt_price(t0_price, is_krx)}) 이하 도달 또는 목표 비중 50% 축소 가동 구간 (MDD Ratio {mdd_ratio:.2f})"
        if self_z is None and current_price <= t1_price and mdd_ratio >= 0.30 and exp_return >= req_r and not survival_warning:
            pq_reason = f"조건부 — 1차 기준선 이하이나 자기 PER 밴드 {NO_DATA}로 '정상 밴드' 확인 불가 (QUALIFIED 보류, MDD Ratio {mdd_ratio:.2f})"
    else:
        pq_status = "DISQUALIFIED"
        pq_icon = "🔴"
        gap_to_t1 = (current_price / max(t1_price, 1) - 1.0) * 100.0
        pq_reason = f"현재가({fmt_price(current_price, is_krx)})가 1차 매수기준선({fmt_price(t1_price, is_krx)}) 대비 +{gap_to_t1:.1f}% 높음 — 지정가 도달 전까지 추격 매수 금지(대기)"

    # F10. 멀티플 오판 포인트 자동 문장 생성[cite: 4]
    _dd = "" if dom_driver == NO_DATA else f" ({dom_driver})"
    if implied_5y_cagr is None or consensus_2y_cagr is None:
        underval_pt = f"내재 5년 성장률 대 컨센서스 성장률 비교 {NO_DATA} (성숙 PER 또는 2Y 추정치 부족)."
    elif implied_5y_cagr < consensus_2y_cagr:
        underval_pt = (
            f"현재 주가에 내재된 5년 성장률({implied_5y_cagr*100:.1f}%)이 컨센서스 성장률({consensus_2y_cagr*100:.1f}%)보다 현저히 낮아 "
            f"멀티플 안전마진이 확보됨{_dd}."
        )
    else:
        underval_pt = "현재 배수는 컨센서스 성장률을 상당 부분 반영 중."

    _over_growth = (implied_5y_cagr is not None and consensus_2y_cagr is not None and
                    implied_5y_cagr > max(consensus_2y_cagr, _hist_max if _hist_max is not None else consensus_2y_cagr))
    if _over_growth or (self_z is not None and self_z >= 1.5):
        _ig = f"내재 5년 성장률({implied_5y_cagr*100:.1f}%)" if implied_5y_cagr is not None else "내재 5년 성장률"
        overval_pt = f"{_ig} 또는 자기 밴드({_zs(self_z)})가 과열 구간에 진입하여 멀티플 수축 위험 존재."
    elif is_cycle_peak_trap:
        _sp = NO_DATA if pb_per_z_spread is None else f"{pb_per_z_spread:+.2f}σ"
        overval_pt = f"사이클 정점 착시(P/B-PER Z스프레드 {_sp}) 발동 — 표면적 저PER 함정 주의."
    elif implied_5y_cagr is None and self_z is None:
        overval_pt = f"과열 여부 판정 {NO_DATA} (자기 밴드·내재 성장률 모두 없음)"
    else:
        overval_pt = "해당 없음 (과열 신호 없음)"

    def _r4(v):
        return None if v is None else round(v, 4)

    return {
        "status": "완료",
        "thesis_status": "붕괴" if e_regime == "G5" else th_status,
        "price_qualification": {
            "status": pq_status,
            "grade_icon": pq_icon,
            "headline_reason": pq_reason,
            "is_value_trap_blocked": is_value_trap_blocked,
            "is_cycle_peak_trap": is_cycle_peak_trap,
        },
        "multiple_selection": {
            "primary_metric": primary_mult,
            "secondary_metric": secondary_mult,
            "current_multiple": current_per,
            "rule_description": mult_rule_desc,
        },
        "triple_z_score": {
            "self_5y_z": self_z,
            "self_5y_mean": self_mean,
            "self_5y_std": self_std,
            "self_5y_log_std": self_lsd,
            "self_band_levels": (None if self_lsd is None else {k: round(_sb(float(k)), 2) for k in (-2, -1, 0, 1, 2)}),
            "industry_rel_z": ind_rel_z,
            "peer_rel_z": peer_rel_z,
            "peer_rel_now": None if peer_rel_now is None else round(peer_rel_now, 2),
            "current_per_trailing": None if cur_per_band is None else round(cur_per_band, 2),
            "summary_text": triple_z_summary,
        },
        "industry_and_peer": {
            "theoretical_mature_per": theoretical_mature_per,
            "market_ke": _r4(market_ke),
            "rf_10y": f.get("rf_10y"),
            "rf_source": macro_input.get("rf_source", NO_DATA),
            "rf_as_of": macro_input.get("rf_as_of"),
            "market_erp": f.get("market_erp"),
            "market_erp_source": FUND_MODEL_PARAMS["market_erp_source"],
            "industry_regime": ind_regime,
            "cycle_type": cycle_type,
            "pb_per_z_spread": pb_per_z_spread,
            "peer_net_advantage_score": net_adv_score,
            "peer_factors_scored": n_factors_scored,
            "peer_justification": peer_just,
            "peer_factors_table": peer_factors,
        },
        "multiple_decomposition": {
            "price_return_12m": _r4(log_price_ret),
            "eps_contribution_12m": _r4(log_eps_contrib),
            "multiple_contribution_12m": _r4(log_mult_contrib),
            "layer_market": _r4(layer_market),
            "layer_industry": _r4(layer_industry),
            "layer_company": _r4(layer_company),
            "dominant_driver": dom_driver,
            "basis": "후행(확정) EPS 기준 2층 분해 — 시장·산업층은 업종 PER 이력이 없어 " + NO_DATA,
        },
        "rerating_assessment": {
            "direction": rerate_dir,
            "implied_5y_cagr": implied_5y_cagr,
            "consensus_2y_cagr": _r4(consensus_2y_cagr),
            "terminal_per_used": per_terminal,
            "terminal_per_basis": terminal_per_basis,
            "four_axes": {
                "duration": ax_dur,
                "incremental_margin": ax_inc,
                "certainty": ax_cert,
                "priced_in": ax_priced,
            },
        },
        "valuation_targets": {
            "eps_basis": eps_basis,
            "required_return_r": req_r,
            "effective_eps_1y": int(round(eps_eff_1y, 0)) if is_krx else round(eps_eff_1y, 2),
            "base_multiple_target": base_mult,
            "bull_multiple_target": bull_mult,
            "bear_multiple_target": bear_mult,
            "scenario_width_missing": scenario_width_missing,
            "base_independence_passed": base_independence_passed,
            "buy_tranches": buy_tranches,
            "sell_target": sell_target_1y,
            "sell_target_formula": sell_target_formula,
            "sell_target_2y": sell_target_2y,
            "holding_period_cap_warning": holding_cap_warn,
            "expected_return_weighted": _r4(exp_return),
            "mdd_ratio": mdd_ratio,
            "downside_convexity": downside_convexity,
            "anchors_table": anchors_table,
            "scenario_returns": {
                "bull": _r4(r_bull),
                "base": _r4(r_base),
                "bear": _r4(r_bear),
                "cycle_bear": _r4(r_cbear) if sector_type == "사이클형" else None,
            },
        },
        "financial_survival": {
            "fcf_runway_years": runway_str,
            "debt_maturity_1y_ratio": None if debt_mat_ratio is None else round(float(debt_mat_ratio), 2),
            "share_dilution_yoy": _r4(dilution_yoy),
            "survival_warning": survival_warning,
            "checked_count": survival_checked,
        },
        "mispricing_points": {
            "undervalued_point": underval_pt,
            "overvalued_point": overval_pt,
        },
        "input_data": {
            "source": fmeta.get("source"),
            "band_basis": fmeta.get("band_basis"),
            "band_points": fmeta.get("band_points"),
            "band_series": fmeta.get("band_series", []),
            "band_excluded": fmeta.get("band_excluded"),
            "band_ref": fmeta.get("band_ref"),
            "scenario_width_source": scen_lsd_src,
            "band_span_years": fmeta.get("band_span_years"),
            "peer_count": fmeta.get("peer_count"),
            "peer_group": fmeta.get("peer_group"),
            "notes": list(fmeta.get("notes", [])),
            "missing": list(fmeta.get("missing", [])),
        },
        # 하위 호환용 최상위 키 유지 (기술 트랙 연동)
        "buy_tranches": buy_tranches,
        "sell_target": sell_target_1y,
        "position_cap": round(pos_cap, 4),
    }
