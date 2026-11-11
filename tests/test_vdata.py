"""V 트랙 실제 데이터 입력층(미국 yfinance · 한국 DART)과 None 대응 분석을 가짜 데이터로 검증한다. (네트워크 호출 없음)"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # 프로젝트 폴더의 engine.py를 불러오기 위해
import _offline  # noqa: F401  외부 네트워크 차단 (tests/_offline.py)
import sys, types, math, importlib
import os
from datetime import datetime
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
engine = importlib.import_module("engine")
NO = engine.NO_DATA
fails = []


def check(name, cond, extra=""):
    print(("PASS " if cond else "FAIL ") + name + ((" :: " + str(extra)[:170]) if extra else ""))
    if not cond:
        fails.append(name)


NOW = pd.Timestamp(datetime.now().date())
YEAR = NOW.year

# ------------------------------------------------------------------ 가짜 가격 (결정적)
def make_hist(years=6):
    idx = pd.bdate_range(end=NOW, periods=int(252 * years))
    c = 100.0 * (1.0 + 0.0006 * np.arange(len(idx)))
    return pd.DataFrame({"Close": c}, index=idx)

HIST = make_hist()

# ------------------------------------------------------------------ 미국 가짜 yfinance
FY_ENDS = [pd.Timestamp(f"{YEAR-1}-12-31"), pd.Timestamp(f"{YEAR-2}-12-31"), pd.Timestamp(f"{YEAR-3}-12-31"), pd.Timestamp(f"{YEAR-4}-12-31")]
EPS_A = [6.0, 5.0, 4.0, 3.0]
EQ_A = [3000.0, 2700.0, 2400.0, 2100.0]
PEER_PE = [10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]


def ann_df(rows):
    return pd.DataFrame({FY_ENDS[i]: [r[i] for r in rows.values()] for i in range(4)}, index=list(rows.keys()))


def q_df(rows):
    cols = pd.to_datetime(["2026-09-30", "2026-06-30", "2026-03-31", "2025-12-31", "2025-09-30"])
    return pd.DataFrame({c: [r[i] for r in rows.values()] for i, c in enumerate(cols)}, index=list(rows.keys()))


class FakeIndustry:
    def __init__(self, key):
        self.key = key

    @property
    def top_companies(self):
        if MODE["peers"] == "none":
            raise RuntimeError("no industry")
        syms = ["ME"] + [f"P{i:02d}" for i in range(len(PEER_PE))]
        return pd.DataFrame({"name": syms}, index=syms)


MODE = {"us": "full", "peers": "ok"}


class FakeTicker:
    def __init__(self, sym):
        self.sym = sym

    @property
    def info(self):
        if self.sym.startswith("P") and self.sym[1:].isdigit():
            i = int(self.sym[1:])
            return {"forwardPE": PEER_PE[i], "returnOnEquity": 0.10 + 0.01 * i, "trailingEps": 5.0, "forwardEps": 5.5}
        if MODE["us"] == "empty":
            raise RuntimeError("no info")
        first = (NOW - pd.DateOffset(years=12)).timestamp()
        return {"beta": 1.2, "returnOnEquity": 0.25, "dividendRate": 1.2, "sector": "Technology", "industryKey": "semis",
                "sectorKey": "technology", "firstTradeDateEpochUtc": first, "targetMeanPrice": 210.0,
                "trailingEps": 6.0, "forwardEps": 7.2, "sharesOutstanding": 1000.0, "priceToBook": 9.9}

    def history(self, period="6y", auto_adjust=False, **kw):
        if MODE["us"] == "empty":
            raise RuntimeError("no history")
        return HIST.copy()

    @property
    def income_stmt(self):
        if MODE["us"] == "empty":
            raise RuntimeError("x")
        return ann_df({"Diluted EPS": EPS_A, "Diluted Average Shares": [1000.0, 1010.0, 1020.0, 1030.0]})

    @property
    def balance_sheet(self):
        if MODE["us"] == "empty":
            raise RuntimeError("x")
        return ann_df({"Stockholders Equity": EQ_A})

    @property
    def quarterly_balance_sheet(self):
        if MODE["us"] == "empty":
            raise RuntimeError("x")
        return q_df({"Cash Cash Equivalents And Short Term Investments": [50.0] * 5, "Total Debt": [30.0] * 5,
                     "Current Debt And Capital Lease Obligation": [10.0] * 5})

    @property
    def quarterly_cashflow(self):
        if MODE["us"] == "empty":
            raise RuntimeError("x")
        return q_df({"Free Cash Flow": [10.0, 8.0, 6.0, 4.0, 2.0]})

    @property
    def quarterly_income_stmt(self):
        if MODE["us"] == "empty":
            raise RuntimeError("x")
        return q_df({"Diluted Average Shares": [1010.0, 1008.0, 1005.0, 1002.0, 1000.0]})


fake_yf = types.ModuleType("yfinance")
fake_yf.Ticker = lambda s: FakeTicker(s)
fake_yf.Industry = FakeIndustry
fake_yf.Sector = FakeIndustry
fake_yf.download = lambda *a, **k: pd.DataFrame()
sys.modules["yfinance"] = fake_yf

# ================================================================== 1) 미국 입력층
price = float(HIST["Close"].iloc[-1])
out = engine.fetch_real_fundamental_inputs("ME", price)
v, m = out["values"], out["meta"]
check("US: 베타·ROE", v["industry_beta"] == 1.2 and abs(v["company_roe"] - 0.25) < 1e-9 and v["sustainable_roe"] == v["company_roe"])
check("US: 배당수익률 = 주당배당/주가", abs(v["dividend_yield"] - 1.2 / price) < 1e-9, v["dividend_yield"])
check("US: 비금융 · 상장연수", v["is_financial_sector"] is False and 11.5 < v["listing_years"] < 12.5, v["listing_years"])
check("US: 컨센서스 목표가", v["consensus_target_price"] == 210.0)
check("US: 선행/후행 EPS 증가율", abs(v["company_fwd_growth"] - 0.2) < 1e-9)
# 12개월 전 가격: 1년 전 이하 마지막 거래일
target = HIST.index[-1] - pd.DateOffset(years=1)
exp_p0 = float(HIST["Close"][HIST.index <= target].iloc[-1])
check("US: 12개월 전 가격", abs(v["price_12m_ago"] - exp_p0) < 1e-9, (v["price_12m_ago"], exp_p0))
# 자기 PER 밴드: 독립 계산 (월말 종가 ÷ 결산+60일 이후 알려진 연간 EPS)
me = HIST["Close"].groupby([HIST.index.year, HIST.index.month]).last()
me_dates = HIST.groupby([HIST.index.year, HIST.index.month]).apply(lambda d: d.index[-1])
cut = me_dates.iloc[-1] - pd.DateOffset(years=5)
vals = []
for (ym, d), p in zip(me_dates.items(), me.values):
    if d < cut:
        continue
    eps = None
    for fe, e in sorted(zip(FY_ENDS, EPS_A)):
        if fe + pd.Timedelta(days=60) <= d:
            eps = e
    if eps and eps > 0:
        vals.append(p / eps)
check("US: 밴드 관측 개수", m["band_points"] == len(vals), (m["band_points"], len(vals)))
check("US: 밴드 중심 = 로그 PER 평균의 지수(기하평균)", abs(v["self_5y_mean"] - float(np.exp(np.mean(np.log(vals))))) < 1e-6, (v["self_5y_mean"], float(np.exp(np.mean(np.log(vals))))))
check("US: 밴드 로그 σ(모집단)", abs(v["self_5y_log_std"] - float(np.std(np.log(vals)))) < 1e-9)
check("US: 밴드 최소", abs(v["self_5y_min"] - min(vals)) < 1e-6)
check("US: 밴드는 EPS 이력 구간(4개 사업연도, 약 3.8년)만", m["band_span_years"] is not None and m["band_span_years"] <= 4.0, m["band_span_years"])
check("US: 후행 EPS(현재/12개월 전)", v["eps_lag_now"] == 6.0 and v["eps_lag_12m_ago"] == 5.0, (v["eps_lag_now"], v["eps_lag_12m_ago"]))
check("US: 피어 개수·중앙값·상위25%", m["peer_count"] == 12 and abs(v["peer_median_per"] - 21.0) < 1e-9 and abs(v["peer_top25_per"] - 26.5) < 1e-9,
      (m["peer_count"], v["peer_median_per"], v["peer_top25_per"]))
check("US: 피어 σ", abs(v["peer_std_per"] - float(np.std(PEER_PE))) < 1e-9)
check("US: 피어 ROE·성장 중앙값", abs(v["peer_median_roe"] - 0.155) < 1e-9 and abs(v["peer_median_fwd_growth"] - 0.1) < 1e-9)
check("US: 순현금·단기차입 비율", v["net_cash_krw"] == 20.0 and abs(v["short_term_debt_ratio"] - 0.2) < 1e-9)
check("US: FCF 4분기 합", v["fcf_ttm_krw"] == 28.0)
check("US: 희석률", abs(v["share_dilution_yoy"] - 0.01) < 1e-9, v["share_dilution_yoy"])
check("US: 현재 P/B = 주가 ÷ (최근 자본÷주식수)", abs(v["current_pb"] - price / 3.0) < 1e-6, v["current_pb"])
check("US: P/B 밴드 산출", v["pb_5y_mean"] is not None and v["pb_5y_std"] is not None)
check("US: 수집 불가 항목은 None (업종 PER 이력·산업 증거)", all(v[k] is None for k in
      ["ind_5y_mean", "ind_5y_std", "ind_rel_5y_mean", "mkt_per_now", "ind_per_now", "ind_rerate_evidences", "historical_max_5y_cagr",
       "shareholder_return_advantage", "peer_rel_5y_mean"]))
check("US: 못 받은 항목 목록에 반영", "ind_5y_mean" in m["missing"] and "self_5y_mean" not in m["missing"])

# 피어 실패 → 피어 통계만 None, 나머지 유지
MODE["peers"] = "none"
o2 = engine.fetch_real_fundamental_inputs("ME", price)
check("US 피어 실패: 피어 값 None, 자기 밴드 유지", o2["values"]["peer_median_per"] is None and o2["values"]["self_5y_mean"] is not None, o2["meta"]["notes"][:2])
MODE["peers"] = "ok"
# 피어 4개뿐 → 최소 유효 수 미달 → None
_save = PEER_PE[:]
PEER_PE[:] = [10, 12, 14, 16]
o3 = engine.fetch_real_fundamental_inputs("ME", price)
check("US 피어 4개: 통계 제외", o3["values"]["peer_median_per"] is None and o3["meta"]["peer_count"] == 4)
PEER_PE[:] = _save
# 전부 실패
MODE["us"] = "empty"
o4 = engine.fetch_real_fundamental_inputs("ME", price)
check("US 전부 실패: 예외 없이 전부 None", all(x is None for x in o4["values"].values()), [k for k, x in o4["values"].items() if x is not None])
MODE["us"] = "full"

# 무배당 (info는 있는데 배당 항목 없음) → 0%
class NoDivTicker(FakeTicker):
    @property
    def info(self):
        d = dict(FakeTicker.info.fget(self))
        d.pop("dividendRate")
        return d
fake_yf.Ticker = lambda s: NoDivTicker(s) if s == "ND" else FakeTicker(s)
check("US 무배당: 배당수익률 0", engine.fetch_real_fundamental_inputs("ND", price)["values"]["dividend_yield"] == 0.0)
fake_yf.Ticker = lambda s: FakeTicker(s)

# ================================================================== 2) 한국 입력층 (DART)
def dart_rows(y, scale=1.0):
    """사업보고서 한 건: 당기=y, 전기=y-1, 전전기=y-2"""
    def r(sj, aid, nm, a, b, c):
        return {"sj_div": sj, "account_id": aid, "account_nm": nm, "thstrm_amount": a, "frmtrm_amount": b, "bfefrmtrm_amount": c,
                "thstrm_add_amount": ""}
    return pd.DataFrame([
        r("IS", "ifrs-full_BasicEarningsLossPerShare", "기본주당이익", "5,000", "4,000", "3,000"),
        r("IS", "ifrs-full_ProfitLossAttributableToOwnersOfParent", "지배기업의 소유주에게 귀속되는 당기순이익", "50,000", "42,000", "30,000"),
        r("BS", "ifrs-full_EquityAttributableToOwnersOfParent", "지배기업 소유주지분", "400,000", "360,000", "320,000"),
        r("BS", "ifrs-full_CashAndCashEquivalents", "현금및현금성자산", "60,000", "50,000", "40,000"),
        r("BS", "ifrs-full_CurrentFinancialAssetsAtFairValueThroughProfitOrLoss", "단기금융상품", "40,000", "30,000", "20,000"),
        r("BS", "ifrs-full_ShorttermBorrowings", "단기차입금", "20,000", "25,000", "30,000"),
        r("BS", "ifrs-full_CurrentPortionOfLongtermBorrowings", "유동성장기부채", "10,000", "10,000", "10,000"),
        r("BS", "ifrs-full_LongtermBorrowings", "장기차입금", "30,000", "30,000", "30,000"),
        r("CF", "ifrs-full_CashFlowsFromUsedInOperatingActivities", "영업활동현금흐름", "70,000", "60,000", "50,000"),
        r("CF", "ifrs-full_PurchaseOfPropertyPlantAndEquipmentClassifiedAsInvestingActivities", "유형자산의 취득", "30,000", "28,000", "26,000"),
    ])


class FakeDart:
    def finstate_all(self, corp, year, reprt_code="11011", fs_div="CFS"):
        if fs_div != "CFS" or reprt_code != "11011" or year >= YEAR:
            return None
        return dart_rows(year)


class FakeKrTicker(FakeTicker):
    @property
    def info(self):
        return {"beta": 0.9, "sharesOutstanding": 100.0, "trailingEps": 5200.0}   # 배당·섹터·목표가·ROE 없음

    def history(self, period="6y", auto_adjust=False, **kw):
        h = HIST.copy()
        h["Close"] = h["Close"] * 700.0
        return h


fake_yf.Ticker = lambda s: FakeKrTicker(s)
engine._get_dart_client = lambda: FakeDart()
kp = float(HIST["Close"].iloc[-1]) * 700.0
ko = engine.fetch_real_fundamental_inputs("005930.KS", kp)
kv, km = ko["values"], ko["meta"]
check("KR: 베타만 yfinance, 배당·섹터·목표가는 None", kv["industry_beta"] == 0.9 and kv["dividend_yield"] is None and kv["is_financial_sector"] is None
      and kv["consensus_target_price"] is None, (kv["dividend_yield"], kv["is_financial_sector"]))
check("KR: ROE는 DART 지배주주순이익 ÷ 지배자본", abs(kv["company_roe"] - 50000 / 400000) < 1e-9, kv["company_roe"])
check("KR: 순현금 = (현금+단기금융) − 차입합", kv["net_cash_krw"] == (60000 + 40000) - (20000 + 10000 + 30000), kv["net_cash_krw"])
check("KR: 단기차입 비율", abs(kv["short_term_debt_ratio"] - 30000 / 100000) < 1e-9, kv["short_term_debt_ratio"])
check("KR: FCF = OCF − 유형자산 취득", kv["fcf_ttm_krw"] == 40000.0)
check("KR: 희석률 = 가중평균 주식수(순이익÷EPS) 증감", abs(kv["share_dilution_yoy"] - ((50000 / 5000) / (42000 / 4000) - 1)) < 1e-9, kv["share_dilution_yoy"])
check("KR: 피어는 None(미국만 지원)", kv["peer_median_per"] is None and km["peer_count"] == 0)
check("KR: 자기 PER 밴드 산출 (DART 6년 EPS)", kv["self_5y_mean"] is not None and km["band_points"] >= 24, km["band_points"])
check("KR: 12개월 전 EPS(후행)", kv["eps_lag_now"] == 5000.0 and kv["eps_lag_12m_ago"] == 4000.0, (kv["eps_lag_now"], kv["eps_lag_12m_ago"]))
# DART 없음 → 시장 값만, 재무 값은 None
engine._get_dart_client = lambda: None
ko2 = engine.fetch_real_fundamental_inputs("005930.KS", kp)
check("KR DART 없음: 재무 값 None, 베타 유지, 안내 문구", ko2["values"]["net_cash_krw"] is None and ko2["values"]["self_5y_mean"] is None
      and ko2["values"]["industry_beta"] == 0.9 and any("DART" in n for n in ko2["meta"]["notes"]), ko2["meta"]["notes"][:2])
# 차입 항목을 하나도 못 찾으면 순현금을 만들지 않는다 (0으로 단정 금지)
class NoDebtDart(FakeDart):
    def finstate_all(self, corp, year, reprt_code="11011", fs_div="CFS"):
        df = super().finstate_all(corp, year, reprt_code, fs_div)
        return None if df is None else df[~df["account_nm"].str.contains("차입금|유동성장기부채")]
engine._get_dart_client = lambda: NoDebtDart()
ko3 = engine.fetch_real_fundamental_inputs("005930.KS", kp)
check("KR 차입 항목 없음: 순현금·단기차입비율 None", ko3["values"]["net_cash_krw"] is None and ko3["values"]["short_term_debt_ratio"] is None)

# ================================================================== 3) V 트랙 분석 (입력 주입)
def fake_earn(basis, regime="G2", sector="구조성장형", is_krx=False):
    base = 4000.0 if is_krx else 7.4
    e = {"forward": (base, base * 1.2, base * 0.8), "ttm_reference": (None, None, None)}[basis]
    return {
        "regime": {"earnings_regime": regime, "regime_label": "테스트", "sector_type": sector, "cycle_stage": None, "growth_acceleration": NO},
        "verdict": {"score": None, "grade": "⚪", "real_signal": "중립", "headline_reason": "t", "peak_out_warning": False},
        "projections": {"eps_basis": basis, "ttm_eps": base * 0.9, "eps_1y": {"base": e[0], "bull": e[1], "bear": e[2], "cycle_bear": None},
                        "eps_2y": {"base": (base * 1.2) if basis == "forward" else None}, "certainty_adjustment": 0.0, "certainty_label": f"{NO} (미확인)"},
        "sustainability_metrics": {"incremental_opm": None, "average_opm": None, "rpo_yoy_growth": None, "sales_yoy_growth": None},
        "revision_dynamics": {"status": NO}, "quality_checks": {"warning_count": 0, "checked_count": 0},
        "thesis_driver": {"status": "미확인", "invalidation_condition": NO},
    }


FULL = dict(industry_beta=1.1, sustainable_roe=0.20, dividend_yield=0.01, is_financial_sector=False, listing_years=12.0,
            self_5y_mean=22.0, self_5y_std=4.0, self_5y_min=15.0, peer_median_per=20.0, peer_top25_per=26.0, peer_std_per=5.0,
            current_pb=8.0, pb_5y_mean=7.0, pb_5y_std=1.5, company_roe=0.25, peer_median_roe=0.15,
            company_fwd_growth=0.2, peer_median_fwd_growth=0.1, price_12m_ago=100.0, eps_lag_now=6.0, eps_lag_12m_ago=5.0,
            consensus_target_price=210.0, net_cash_krw=20.0, fcf_ttm_krw=28.0, short_term_debt_ratio=0.2, share_dilution_yoy=0.01)
KEYS = ["industry_beta", "sustainable_roe", "dividend_yield", "is_financial_sector", "listing_years", "self_5y_mean", "self_5y_std", "self_5y_min",
        "ind_5y_mean", "ind_5y_std", "ind_rel_5y_mean", "ind_rel_5y_std", "ind_rel_current", "peer_median_per", "peer_top25_per", "peer_std_per",
        "peer_rel_5y_mean", "peer_rel_5y_std", "ind_rerate_evidences", "ind_derate_evidences", "ind_evidence_2q_confirmed", "current_pb", "pb_5y_mean",
        "pb_5y_std", "company_roe", "peer_median_roe", "company_fwd_growth", "peer_median_fwd_growth", "shareholder_return_advantage", "price_12m_ago",
        "eps_lag_12m_ago", "eps_lag_now", "mkt_per_12m_ago", "mkt_per_now", "ind_per_12m_ago", "ind_per_now", "historical_max_5y_cagr",
        "consensus_target_price", "net_cash_krw", "fcf_ttm_krw", "short_term_debt_ratio", "share_dilution_yoy"]


def inject(**vals):
    d = {k: None for k in KEYS}
    d.update(vals)
    engine.fetch_real_fundamental_inputs = lambda t, p: {"values": dict(d), "meta": {"source": "test", "notes": [], "missing": [k for k, x in d.items() if x is None],
                                                                                     "band_basis": "후행", "band_points": 36, "peer_count": 0}}


def run_v(basis="forward", regime="G2", sector="구조성장형", price=120.0, ticker="NVDA", **vals):
    inject(**vals)
    return engine.analyze_fundamental_track(ticker, price, fake_earn(basis, regime, sector, ticker.endswith(".KS")),
                                            {"rf_10y": 0.042, "market_erp": 0.055}, None)


def walk(o, path=""):
    """결과 전체에서 NaN/inf가 없는지 확인"""
    bad = []
    if isinstance(o, dict):
        for k, x in o.items():
            bad += walk(x, f"{path}/{k}")
    elif isinstance(o, (list, tuple)):
        for i, x in enumerate(o):
            bad += walk(x, f"{path}[{i}]")
    elif isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        bad.append(path)
    return bad


fo = run_v(**FULL)
vt = fo["valuation_targets"]
check("V 전체 데이터: 가격선 4개 · 자격 판정", len(vt["buy_tranches"]) in (3, 4) and fo["price_qualification"]["status"] in ("QUALIFIED", "CONDITIONAL", "DISQUALIFIED"),
      (fo["price_qualification"]["status"], [t["price"] for t in vt["buy_tranches"]]))
check("V 전체: 가격선 단조 감소", [t["price"] for t in vt["buy_tranches"]] == sorted([t["price"] for t in vt["buy_tranches"]], reverse=True))
check("V 전체: G2 Base 배수 = 자기 평균 (Bull/Bear = μ·e^(±σ), 로그 기준)", vt["base_multiple_target"] == 22.0 and vt["bull_multiple_target"] == 26.0 and vt["bear_multiple_target"] == 18.6, vt["base_multiple_target"])
check("V 전체: 1차선 = EPS×배수÷(1+r−d)", abs(vt["buy_tranches"][-3]["price"] - round(7.4 * 22.0 / (1 + vt["required_return_r"] - 0.01), 2)) < 0.02,
      (vt["buy_tranches"][-3]["price"], 7.4 * 22.0 / (1 + vt["required_return_r"] - 0.01)))
check("V 전체: 매도목표가 = 앵커2·3 평균 × EPS", vt["sell_target"] is not None and "앵커2·3 평균" in vt["sell_target_formula"], vt["sell_target_formula"][:90])
check("V 전체: 이론 성숙 PER·내재 CAGR 계산", fo["industry_and_peer"]["theoretical_mature_per"] is not None and fo["rerating_assessment"]["implied_5y_cagr"] is not None)
check("V 전체: 3중 z는 자기·피어만 (산업 상대 None)", fo["triple_z_score"]["self_5y_z"] is not None and fo["triple_z_score"]["industry_rel_z"] is None
      and "데이터 없음" in fo["triple_z_score"]["summary_text"], fo["triple_z_score"]["summary_text"])
check("V 전체: 12개월 분해는 2층 (시장·산업 None)", fo["multiple_decomposition"]["eps_contribution_12m"] is not None and fo["multiple_decomposition"]["layer_market"] is None)
check("V 전체: 산업 레짐·4축 일부 데이터 없음", fo["industry_and_peer"]["industry_regime"] == NO and fo["rerating_assessment"]["four_axes"]["duration"] == NO)
check("V 전체: 피어 8요인 중 8번(주주환원)·3~6번 데이터 없음, 1·2·7은 점수", [r["diff"] == NO for r in fo["industry_and_peer"]["peer_factors_table"]] == [False, False, True, True, True, True, False, True],
      [r["diff"] for r in fo["industry_and_peer"]["peer_factors_table"]])
check("V 전체: NaN/inf 없음", walk(fo) == [], walk(fo)[:3])

# 자기 밴드·피어 둘 다 없음 → NA, 가격선 없음
fna = run_v(**{**{k: None for k in FULL}, "industry_beta": 1.1, "sustainable_roe": 0.2, "consensus_target_price": 210.0})
check("V 근거 없음(G2): 가격 자격 NA · 가격선 비움", fna["price_qualification"]["status"] == "NA" and fna["buy_tranches"] == [] and fna["valuation_targets"]["mdd_ratio"] is None,
      fna["price_qualification"]["headline_reason"][:50])
check("V 근거 없음: 기대수익 None · 시나리오 None", fna["valuation_targets"]["expected_return_weighted"] is None and fna["valuation_targets"]["scenario_returns"]["base"] is None)
check("V 근거 없음: 규칙 설명에 데이터 없음 표기", NO in fna["multiple_selection"]["rule_description"])
check("V 근거 없음: 매도 앵커2 없음·앵커3(PEG)만 사용", "앵커3만 사용" in fna["valuation_targets"]["sell_target_formula"], fna["valuation_targets"]["sell_target_formula"][:100])
check("V 근거 없음: NaN/inf 없음", walk(fna) == [])
# 전부 없음 (베타·ROE도 없음) + 참고용 TTM → 매도목표가도 없음
fall = run_v(basis="ttm_reference", **{})
check("V 전부 없음(TTM 참고용): 가격선·매도목표가·2Y 모두 None, 완주", fall["buy_tranches"] == [] and fall["sell_target"] is None
      and fall["valuation_targets"]["sell_target_2y"] is None and fall["price_qualification"]["status"] == "NA", fall["valuation_targets"]["sell_target_formula"][:70])
check("V 전부 없음: 2Y CAGR 하한(2%)을 지어내지 않음", fall["rerating_assessment"]["consensus_2y_cagr"] is None)
check("V 전부 없음: 재평가 방향·선반영도 데이터 없음", fall["rerating_assessment"]["direction"] == NO and fall["rerating_assessment"]["four_axes"]["priced_in"] == NO)
check("V 전부 없음: 생존성 지표 None", fall["financial_survival"]["debt_maturity_1y_ratio"] is None and fall["financial_survival"]["share_dilution_yoy"] is None
      and fall["financial_survival"]["fcf_runway_years"] == NO and fall["financial_survival"]["survival_warning"] is False)
check("V 전부 없음: NaN/inf 없음", walk(fall) == [])

# 국면별 가용 근거 처리
g1 = run_v(regime="G1", **{**FULL, "peer_top25_per": None, "peer_median_per": None, "peer_std_per": None})
check("G1 + 피어 없음: Base 없음 → NA (자기 밴드로 대체하지 않음)", g1["price_qualification"]["status"] == "NA")
g1b = run_v(regime="G1", **FULL)
check("G1 + 피어: Base = 피어 상위25%, Bull = +σ, Bear = 중앙값", (g1b["valuation_targets"]["base_multiple_target"], g1b["valuation_targets"]["bull_multiple_target"], g1b["valuation_targets"]["bear_multiple_target"]) == (26.0, 31.0, 20.0))
g3 = run_v(regime="G3", **{**FULL, "self_5y_mean": None, "self_5y_std": None, "self_5y_min": None})
_g3vt = g3["valuation_targets"]
check("G3 자기 밴드 없음: Base = 피어 중앙값(이론 PER은 참고), Bull/Bear 폭은 피어 PER σ",
      _g3vt["base_multiple_target"] == 20.0 and _g3vt["scenario_width_missing"] is False
      and abs(_g3vt["bull_multiple_target"] - round(20.0 * (1 + 5.0 / 20.0) ** 0.5, 1)) < 0.05
      and abs(_g3vt["bear_multiple_target"] - round(20.0 / (1 + 5.0 / 20.0), 1)) < 0.05,
      (_g3vt["base_multiple_target"], _g3vt["bull_multiple_target"], _g3vt["bear_multiple_target"]))

g4 = run_v(regime="G4", **{**{k: None for k in FULL}, "price_12m_ago": 100.0})
check("G4 근거 전혀 없음: 현재 PER만으로 목표 배수를 만들지 않음 → NA", g4["price_qualification"]["status"] == "NA" and g4["valuation_targets"]["base_multiple_target"] is None)
g4b = run_v(regime="G4", **FULL)
check("G4 + 자기 밴드: Base = min(μ−σ, 이론PER, 현재PER) 이하, Cap 작동", g4b["price_qualification"]["is_value_trap_blocked"] is True and g4b["valuation_targets"]["base_multiple_target"] <= 22.0 - 4.0 + 0.05,
      g4b["valuation_targets"]["base_multiple_target"])
g5 = run_v(regime="G5", **FULL)
check("G5: 가격선 없음 · 자격 DISQUALIFIED(NA 아님)", g5["buy_tranches"] == [] and g5["price_qualification"]["status"] == "DISQUALIFIED")
cyc = run_v(sector="사이클형", **{**FULL, "pb_5y_mean": None})
check("사이클형 + P/B 밴드 없음: 정점 착시 판정 불가 표기", cyc["industry_and_peer"]["cycle_type"] is not None and NO in cyc["industry_and_peer"]["cycle_type"], cyc["industry_and_peer"]["cycle_type"])
cyc2 = run_v(sector="사이클형", **{**FULL, "current_pb": 20.0})
check("사이클형 + P/B 급등: Type 2 (정점 착시) 발동", "Type 2" in cyc2["industry_and_peer"]["cycle_type"] and cyc2["price_qualification"]["is_cycle_peak_trap"] is True)
fin = run_v(**{**FULL, "is_financial_sector": True})
check("금융업: P/B 주 멀티플 선택", fin["multiple_selection"]["primary_metric"] == "P/B")
ipo = run_v(**{**FULL, "listing_years": 1.5})
check("상장 3년 미만: 피어 PER 멀티플 선택", "피어" in ipo["multiple_selection"]["primary_metric"])
unk = run_v(**{**FULL, "listing_years": None})
check("상장연수 모름: IPO 규칙을 적용하지 않음", "피어" not in unk["multiple_selection"]["primary_metric"])
surv = run_v(**{**FULL, "fcf_ttm_krw": -50.0, "net_cash_krw": 20.0})
check("FCF 적자 · 순현금 20 → 런웨이 경고 + 비중 50% 축소", surv["financial_survival"]["survival_warning"] is True and surv["position_cap"] == 0.025, (surv["financial_survival"]["fcf_runway_years"], surv["position_cap"]))
surv2 = run_v(**{**FULL, "fcf_ttm_krw": -50.0, "net_cash_krw": None})
check("FCF 적자 · 순현금 모름: 런웨이 '데이터 없음' (경고 단정 안 함)", NO in surv2["financial_survival"]["fcf_runway_years"] and surv2["financial_survival"]["survival_warning"] is False)
nodiv = run_v(**{**FULL, "dividend_yield": None})
check("배당 모름: 0%로 계산(보수적), 완주", nodiv["valuation_targets"]["buy_tranches"] != [])
krv = run_v(ticker="005930.KS", price=80000.0, **{**FULL, "consensus_target_price": 100000.0})
check("KR: 가격선·매도목표가 정수(원 호가)", all(isinstance(t["price"], int) for t in krv["buy_tranches"]) and isinstance(krv["sell_target"], int), krv["sell_target"])
ov = run_v(**{**FULL, "self_5y_mean": 30.0})
check("fund_override가 수집값보다 우선", True)  # 아래 직접 확인
inject(**FULL)
ovo = engine.analyze_fundamental_track("NVDA", 120.0, fake_earn("forward"), {"rf_10y": 0.042, "market_erp": 0.055}, {"self_5y_mean": 30.0})
check("fund_override: 자기 평균 덮어쓰기", ovo["valuation_targets"]["base_multiple_target"] == 30.0, ovo["valuation_targets"]["base_multiple_target"])

# ================================================================== 4) 종합 판정(Synthesis) · T 트랙 체인
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import test_chain_helpers as H   # noqa: E402

for title, vals, regime, basis in [
    ("전체 데이터", FULL, "G2", "forward"),
    ("근거 없음(NA)", {"industry_beta": 1.1, "sustainable_roe": 0.2}, "G2", "forward"),
    ("전부 없음 · TTM 참고용", {}, "G2", "ttm_reference"),
    ("G4 + 자기 밴드", FULL, "G4", "forward"),
    ("G1 + 피어 없음", {**FULL, "peer_top25_per": None, "peer_median_per": None, "peer_std_per": None}, "G1", "forward"),
]:
    try:
        inject(**vals)
        last, fund, tech, synth = H.run_chain("NVDA", H.fake_earn(basis, False, regime=regime), 120.0)
        card = synth["v_action_card"]
        check(f"체인[{title}]: 완주", True)
        if fund["price_qualification"]["status"] == "NA":
            check(f"체인[{title}]: V_NA · 가격 자격 NA · 기대수익/MDD None", card["status_code"] in ("V_NA", "V_PROFIT_TAKE", "V_BLOCKED") and card["price_qual_code"] in ("NA", "VALUE_TRAP")
                  and card["mdd_ratio"] is None and card["expected_return_weighted"] is None, (card["status_code"], card["price_qual_code"]))
            check(f"체인[{title}]: 비중·집행 0 (가짜 1차선으로 매수 신호 없음)", card["execution_ratio"] == 0.0 and card["recommended_weight"] == 0.0)
            txt = " ".join(synth["three_line_briefing"].values())
            check(f"체인[{title}]: 문구에 '데이터 없음'", NO in txt, txt[:120])
        else:
            check(f"체인[{title}]: 가격선 기반 상태 코드", card["status_code"].startswith("V_") and card["mdd_ratio"] is not None, card["status_code"])
        check(f"체인[{title}]: 합성 결과 NaN/inf 없음", walk(synth) == [], walk(synth)[:3])
    except Exception as ex:
        import traceback; traceback.print_exc()
        check(f"체인[{title}]: 완주", False, repr(ex))

print("\nFAILS:", fails)
sys.exit(1 if fails else 0)
