"""딥다이브 화면(Streamlit)을 AppTest로 실제 실행해, 실적·V 트랙 탭이 '데이터 없음'/'참고용' 상태에서 예외 없이 그려지는지 본다."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # 프로젝트 폴더의 engine 패키지를 불러오기 위해
import _offline  # noqa: F401  외부 네트워크 차단 (tests/_offline.py)
import sys, os, types, importlib
import numpy as np
import pandas as pd

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

NO = "데이터 없음"
MODE = sys.argv[1] if len(sys.argv) > 1 else "full"   # full | ttm | none
TICKER = sys.argv[2] if len(sys.argv) > 2 else "NVDA"
is_kr = TICKER.endswith(".KS")


def hist(n=520, start=100.0, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end="2026-11-09", periods=n)
    if os.environ.get("TZAWARE", "on") == "on":   # 실제 야후처럼 거래소 시간대가 붙은 날짜
        idx = idx.tz_localize("Asia/Seoul" if TICKER.upper().endswith((".KS", ".KQ")) else "America/New_York")
    c = start * np.exp(np.cumsum(rng.normal(0.0006, 0.015, n)))
    _df = pd.DataFrame({"Open": c * 0.995, "High": c * 1.01, "Low": c * 0.99, "Close": c,
                        "Volume": rng.integers(1_000_000, 3_000_000, n).astype(float)}, index=idx)
    if os.environ.get("NANROW", "on") == "on":   # 실제 야후처럼 종가가 빈 마지막 행(장 마감 후·장중)을 붙인다
        _nx = idx[-1] + pd.Timedelta(days=1)
        _df.loc[_nx] = [np.nan, np.nan, np.nan, np.nan, np.nan]
    return _df


def qdf(rows):
    cols = pd.to_datetime(["2026-09-30", "2026-06-30", "2026-03-31", "2025-12-31", "2025-09-30"])
    return pd.DataFrame({c: [r[i] for r in rows.values()] for i, c in enumerate(cols)}, index=list(rows.keys()))


class FakeTicker:
    def __init__(self, sym):
        self.sym = sym

    def history(self, period="2y", auto_adjust=False, **kw):
        s = self.sym
        start = 70000.0 if s.endswith(".KS") else (2600 if s in ("^KS11", "^KQ11") else (17000 if s == "^IXIC" else (18 if s == "^VIX" else (1400 if s == "KRW=X" else 120.0))))
        return hist(start=start, seed=abs(hash(s)) % 1000)

    @property
    def info(self):
        base = {"shortName": self.sym, "longName": self.sym, "marketCap": 1e12, "sharesOutstanding": 1.5e9}
        if MODE == "none":
            return base
        base.update({"trailingEps": 3500.0 if is_kr else 8.9, "nextFiscalYearEnd": 1798675200})
        return base

    @property
    def calendar(self):
        return {}

    def _need(self):
        if MODE == "none":
            raise RuntimeError("no data")

    @property
    def quarterly_income_stmt(self):
        self._need()
        k = 1000 if is_kr else 1
        return qdf({"Total Revenue": [100 * k, 95 * k, 90 * k, 88 * k, 80 * k], "Gross Profit": [45 * k, 42 * k, 40 * k, 38 * k, 34 * k],
                    "Operating Income": [30 * k, 27 * k, 24 * k, 22 * k, 16 * k], "Net Income": [25 * k, 22 * k, 20 * k, 18 * k, 14 * k],
                    "Diluted EPS": [2.5, 2.2, 2.0, 1.8, 1.4]})

    @property
    def quarterly_balance_sheet(self):
        self._need()
        return qdf({"Accounts Receivable": [30, 29, 28, 27, 26], "Inventory": [20, 19, 18, 18, 17]})

    @property
    def quarterly_cashflow(self):
        self._need()
        return qdf({"Operating Cash Flow": [30, 28, 25, 24, 20], "Capital Expenditure": [-10, -9, -9, -8, -8]})

    @property
    def earnings_estimate(self):
        self._need()
        if MODE == "ttm":
            return pd.DataFrame()
        return pd.DataFrame({"avg": [9.5, 11.0], "low": [8.5, 9.0], "high": [10.5, 13.0], "numberOfAnalysts": [30, 28]}, index=["0y", "+1y"])

    @property
    def eps_trend(self):
        self._need()
        if MODE == "ttm":
            return pd.DataFrame()
        return pd.DataFrame({"current": [9.5, 11.0], "30daysAgo": [9.3, 10.8], "90daysAgo": [9.0, 10.4]}, index=["0y", "+1y"])

    @property
    def earnings_history(self):
        self._need()
        return pd.DataFrame({"epsActual": [2.7], "epsEstimate": [2.6]}, index=pd.to_datetime(["2026-09-30"]))


fake_yf = types.ModuleType("yfinance")
fake_yf.Ticker = lambda sym: FakeTicker(sym)
fake_yf.download = lambda *a, **k: hist()
sys.modules["yfinance"] = fake_yf

# 네트워크가 필요한 보조 검색은 모두 실패하게 둔다 (페이지가 예외를 삼키는지 함께 확인)
import urllib.request
def _no_net(*a, **k):
    raise OSError("network disabled in test")
urllib.request.urlopen = _no_net

# 한국 종목용 가짜 DART
if is_kr:
    import engine
    def dart_df(rows):
        return pd.DataFrame([{"sj_div": r[0], "account_id": r[1], "account_nm": r[2], "thstrm_amount": r[3],
                              "thstrm_add_amount": r[4], "frmtrm_amount": r[5]} for r in rows])
    class FakeDart:
        def finstate_all(self, corp, year, reprt_code="11011", fs_div="CFS"):
            if fs_div != "CFS" or MODE == "none":
                return None
            if reprt_code == "11011" and year == 2026:
                raise RuntimeError("없음")
            rows = [("IS", "ifrs-full_Revenue", "매출액", "80000", "230000" if reprt_code == "11014" else "", ""),
                    ("IS", "dart_OperatingIncomeLoss", "영업이익", "12000", "", ""),
                    ("IS", "ifrs-full_ProfitLoss", "당기순이익", "10000", "", ""),
                    ("IS", "ifrs-full_BasicEarningsLossPerShare", "기본주당이익", "1400", "", ""),
                    ("BS", "ifrs-full_Inventories", "재고자산", "50,000", "", ""),
                    ("BS", "ifrs-full_TradeAndOtherCurrentReceivables", "매출채권", "40,000", "", ""),
                    ("CF", "ifrs-full_CashFlowsFromUsedInOperatingActivities", "영업활동현금흐름", "30,000", "", "25000"),
                    ("CF", "ifrs-full_PurchaseOfPropertyPlantAndEquipmentClassifiedAsInvestingActivities", "유형자산의 취득", "20,000", "", "")]
            return dart_df(rows)
    engine._get_dart_client = lambda: FakeDart()

import engine as _eng
MACROMODE = os.environ.get("MACROMODE", "on")   # on: 거시 수준 시계열 주입 | partial: 일부만 | off: 전부 못 받음
if MACROMODE != "off":
    _wi = pd.date_range(end=pd.Timestamp.today().normalize(), periods=220, freq="W-FRI")
    _rg = np.random.default_rng(11)   # 테스트 입력용
    def _lv(s0, sd, kind):
        x = np.cumsum(_rg.normal(0, sd, len(_wi)))
        return pd.Series(s0 + x if kind == "bp" else s0 * np.exp(x), index=_wi)
    _LV = {"DFII10": _lv(1.9, .05, "bp"), "T10YIE": _lv(2.3, .04, "bp"), "T10Y2Y": _lv(.35, .05, "bp"), "BAMLC0A0CM": _lv(.85, .02, "bp"),
           "BAMLH0A0HYM2": _lv(3.0, .08, "bp"), "DEXKOUS": _lv(1380, .008, "ln"), "DCOILWTICO": _lv(72, .03, "ln"),
           "ICSA": _lv(225000, .03, "ln"), "NETLIQ": _lv(5.9e6, .01, "ln"), "KR10Y": _lv(3.0, .05, "bp")}
    if MACROMODE == "partial":
        for _k in ("T10YIE", "NETLIQ", "KR10Y", "ICSA"):
            _LV.pop(_k)
    _eng.fetch_macro_levels = lambda ids, start, end=None: {"levels": {k: v for k, v in _LV.items() if k in ids},
                                                            "missing": [k for k in ids if k not in _LV], "sources": {k: f"FRED {k} (테스트 주입)" for k in _LV}}
RFMODE = os.environ.get("RFMODE", "on")   # on: 10년 국채 금리를 받은 상태 | off: 못 받은 상태
if RFMODE == "on":
    import time as _time
    for _k, _v in (("US", 0.0484), ("KR", 0.0305)):
        _eng._RF_CACHE[_k] = {"value": _v, "as_of": "2026-11-05", "source": "FRED (테스트 주입)", "_t": _time.time()}
VMODE = os.environ.get("VMODE", "real")   # real(가짜 yfinance 통과) | full | partial | none
_KEYS = ["industry_beta", "sustainable_roe", "dividend_yield", "is_financial_sector", "listing_years", "self_5y_mean", "self_5y_std", "self_5y_min",
         "ind_5y_mean", "ind_5y_std", "ind_rel_5y_mean", "ind_rel_5y_std", "ind_rel_current", "peer_median_per", "peer_top25_per", "peer_std_per",
         "peer_rel_5y_mean", "peer_rel_5y_std", "ind_rerate_evidences", "ind_derate_evidences", "ind_evidence_2q_confirmed", "current_pb", "pb_5y_mean",
         "pb_5y_std", "company_roe", "peer_median_roe", "company_fwd_growth", "peer_median_fwd_growth", "shareholder_return_advantage", "price_12m_ago",
         "eps_lag_12m_ago", "eps_lag_now", "mkt_per_12m_ago", "mkt_per_now", "ind_per_12m_ago", "ind_per_now", "historical_max_5y_cagr",
         "consensus_target_price", "net_cash_krw", "fcf_ttm_krw", "short_term_debt_ratio", "share_dilution_yoy"]
if VMODE != "real":
    _p = 70000.0 if is_kr else 120.0
    _full = dict(industry_beta=1.1, sustainable_roe=0.20, dividend_yield=0.01, is_financial_sector=False, listing_years=12.0,
                 self_5y_mean=22.0, self_5y_std=4.0, self_5y_min=15.0, peer_median_per=20.0, peer_top25_per=26.0, peer_std_per=5.0,
                 current_pb=8.0, pb_5y_mean=7.0, pb_5y_std=1.5, company_roe=0.25, peer_median_roe=0.15, company_fwd_growth=0.2,
                 peer_median_fwd_growth=0.1, price_12m_ago=_p * 0.9, eps_lag_now=_p / 18, eps_lag_12m_ago=_p / 20,
                 consensus_target_price=_p * 1.3, net_cash_krw=20.0, fcf_ttm_krw=28.0, short_term_debt_ratio=0.2, share_dilution_yoy=0.01)
    _vals = {"full": _full, "partial": {k: _full[k] for k in ("industry_beta", "sustainable_roe", "price_12m_ago", "net_cash_krw")}, "none": {}}[VMODE]
    _d = {k: None for k in _KEYS}
    _d.update(_vals)
    _BS = []
    if VMODE == "full" and os.environ.get("BANDSERIES") == "nvda":   # 이익 급변형 밴드(확정 EPS 0.17→4.9) 재현
        _dts = pd.date_range("2022-11-30", "2026-10-31", freq="ME")
        _prs = [(d, float(15 * (180 / 15) ** (i / (len(_dts) - 1)))) for i, d in enumerate(_dts)]
        _prs_eps = [(pd.Timestamp("2023-01-29"), 0.17), (pd.Timestamp("2024-01-28"), 1.19), (pd.Timestamp("2025-01-26"), 2.94), (pd.Timestamp("2026-01-25"), 4.90)]
        _bb = _eng._lagged_ratio_stats(_prs, _prs_eps, 60, 5, 24)
        _d.update(self_5y_mean=_bb["mean"], self_5y_std=_bb["std"], self_5y_min=_bb["min"])
        _d["self_5y_log_std"] = _bb["log_std"]
        _BS = _bb["series"]
    _eng.fetch_real_fundamental_inputs = lambda t, p: {"values": dict(_d), "meta": {"source": "test", "notes": ["테스트 주입"], "band_series": _BS, "missing": [k for k, x in _d.items() if x is None],
                                                                                      "band_basis": "후행(TTM·연간 확정) EPS 기준", "band_points": 36 if _d["self_5y_mean"] else 0, "peer_count": 12 if _d["peer_median_per"] else 0}}

import streamlit as _st
_st.page_link = lambda *a, **k: None   # 테스트 환경에선 페이지 링크 검증이 불가능해 무력화
from streamlit.testing.v1 import AppTest

at = AppTest.from_file(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pages", "1_🔎_종목_딥다이브.py"), default_timeout=170)
at.session_state["selected_ticker"] = TICKER
at.run()
print("MODE", MODE, "VMODE", VMODE, "RFMODE", RFMODE, "MACROMODE", MACROMODE, "TICKER", TICKER)
print("exceptions:", [str(e.value)[:300] for e in at.exception])
print("warnings:", [w.value[:160] for w in at.warning][:3])
md = " ".join(m.value for m in at.markdown)
cap = " ".join(c.value for c in at.caption)
print("has '데이터 없음':", NO in md or NO in cap, "| tabs:", len(at.tabs))
for kw in ("참고용", "판정 불가", "컨센서스 기준"):
    print(f"  contains {kw!r}:", (kw in md) or (kw in cap))
import re as _re
_allt = md + " " + cap + " " + " ".join(str(d.value)[:2000] for d in at.dataframe) if hasattr(at, "dataframe") else md + cap
_bad = [m for m in _re.findall(r"[^\s<>]{0,12}(?:None|nan|NaN|inf)[^\s<>]{0,8}", md + " " + cap) if "Inf" not in m and "info" not in m.lower() and "yfinance" not in m]
print("leaked None/nan tokens:", _bad[:8])
# ---- HTML 블록 깨짐 방지 검사: '<'로 시작하는 마크다운에 빈 줄/들여쓴 태그가 없어야 함
_hb = []
for m in at.markdown:
    v = m.value
    if v.lstrip().startswith("<") and ("\n\n" in v or any(l.startswith("    <") for l in v.split("\n"))):
        _hb.append(v[:80])
print("broken HTML blocks:", len(_hb), _hb[:2])


# ---- T 실행 카드: 셋업이 없으면 가짜 가격/수량이 없어야 함
_tc = [m.value for m in at.markdown if "Step 3. T 트랙" in m.value]
_tcv = _tc[0] if _tc else ""
import re as _r2; print("T badge:", _r2.findall(r"term-badge-inv\">([^<]*)", _tcv)[:1], _r2.findall(r"상태 코드: ([A-Z_0-9]+)", _tcv)); print("T card has ticket text:", "진입 <b>" in _tcv, "| no-ticket text:", "생성되지 않음" in _tcv, "| '0주':", "<b>0주</b>" in _tcv)
sys.exit(1 if (at.exception or _bad or _hb) else 0)
