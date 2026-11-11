"""가짜(mock) yfinance·DART로 실적 트랙 입력층과 분석 함수를 검증한다. (실제 네트워크 호출 없음)"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # 프로젝트 폴더의 engine 패키지를 불러오기 위해
import _offline  # noqa: F401  외부 네트워크 차단 (tests/_offline.py)
import sys, types, math
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import importlib
engine = importlib.import_module("engine")

NO = engine.NO_DATA
fails = []

def check(name, cond, extra=""):
    print(("PASS " if cond else "FAIL ") + name + ((" :: " + str(extra)) if extra else ""))
    if not cond:
        fails.append(name)

# ------------------------------------------------------------------
# 1) 가짜 yfinance (미국 종목: 분기 5개, 추정치 있음)
# ------------------------------------------------------------------
def make_df(rows, ncols=5):
    cols = pd.to_datetime(["2026-09-30", "2026-06-30", "2026-03-31", "2025-12-31", "2025-09-30"][:ncols])
    return pd.DataFrame({c: [r[i] for r in rows.values()] for i, c in enumerate(cols)}, index=list(rows.keys()))

class FakeTicker:
    def __init__(self, sym, mode="full"):
        self.sym, self.mode = sym, mode
    @property
    def quarterly_income_stmt(self):
        if self.mode == "empty":
            raise RuntimeError("no data")
        return make_df({
            "Total Revenue": [100, 95, 90, 88, 80],
            "Gross Profit": [45, 42, 40, 38, 34],
            "Operating Income": [30, 27, 24, 22, 16],
            "Net Income": [25, 22, 20, 18, 14],
            "Diluted EPS": [2.5, 2.2, 2.0, 1.8, 1.4],
        })
    @property
    def quarterly_balance_sheet(self):
        if self.mode == "empty": raise RuntimeError("no data")
        return make_df({"Accounts Receivable": [30, 29, 28, 27, 26], "Inventory": [20, 19, 18, 18, 17]})
    @property
    def quarterly_cashflow(self):
        if self.mode == "empty": raise RuntimeError("no data")
        return make_df({"Operating Cash Flow": [30, 28, 25, 24, 20], "Capital Expenditure": [-10, -9, -9, -8, -8]})
    @property
    def earnings_estimate(self):
        if self.mode == "empty": raise RuntimeError("no data")
        return pd.DataFrame({"avg": [9.5, 11.0], "low": [8.5, 9.0], "high": [10.5, 13.0], "numberOfAnalysts": [30, 28]},
                            index=["0y", "+1y"])
    @property
    def eps_trend(self):
        if self.mode == "empty": raise RuntimeError("no data")
        return pd.DataFrame({"current": [9.5, 11.0], "30daysAgo": [9.3, 10.8], "90daysAgo": [9.0, 10.4]}, index=["0y", "+1y"])
    @property
    def earnings_history(self):
        if self.mode == "empty": raise RuntimeError("no data")
        return pd.DataFrame({"epsActual": [2.7], "epsEstimate": [2.6]}, index=pd.to_datetime(["2026-09-30"]))
    @property
    def info(self):
        if self.mode == "empty": raise RuntimeError("no data")
        return {"trailingEps": 8.9, "sharesOutstanding": 1.5e9, "nextFiscalYearEnd": 1798675200}  # 2026-12-31

fake_yf = types.ModuleType("yfinance")
fake_yf.Ticker = lambda sym: FakeTicker(sym)
sys.modules["yfinance"] = fake_yf

d = engine.fetch_real_earnings_inputs("AAPL")
check("US: 분기 3개 생성", len(d["actual_vs_cons"]) > 0 and d["actual_vs_cons"]["sales_actual"] == 100)
check("US: 매출 YoY = 100/80-1", abs(d["actual_vs_cons"]["sales_yoy"] - 0.25) < 1e-9)
check("US: 전년동기 부족 → 성장 궤적 일부 None", d["sales_yoy_trajectory"] is not None and d["sales_yoy_trajectory"][0] is None)
check("US: EPS 컨센서스(조정)", d["actual_vs_cons"]["eps_cons"] == 2.6 and d["actual_vs_cons"]["eps_actual"] == 2.7)
check("US: FY1 base/bull/bear", d["forward"]["fy1"] == {"base": 9.5, "bull": 10.5, "bear": 8.5})
check("US: TTM 현금흐름 합산(4분기)", d["quality"]["ocf_ttm"] == 30 + 28 + 25 + 24 and d["quality"]["capex_ttm"] == 10 + 9 + 9 + 8)
check("US: 전년 TTM은 8분기 없어서 None", d["quality"]["ocf_prev_ttm"] is None)
check("US: RPO·가이던스는 None", d["sustainability"]["rpo_yoy_growth"] is None and d["guidance_history"] is None)

out = engine.analyze_earnings_track("AAPL")
check("US: 실적 트랙 실행", out["regime"]["earnings_regime"] in ("G1", "G2", "G3", "G4", "G5"), out["regime"])
check("US: 삼성 고정값(4200 등) 없음", all(v not in (4200, 5300, 4050) for v in
      [out["projections"]["eps_1y"]["base"], out["projections"]["eps_2y"]["base"]]), out["projections"]["eps_1y"])
check("US: NTM 혼합(m=1개월)", out["projections"]["eps_1y"]["base"] is not None and 9.5 <= out["projections"]["eps_1y"]["base"] <= 11.0,
      out["projections"]["eps_1y"])
check("US: EPS 달러 소수 유지", isinstance(out["projections"]["eps_1y"]["base"], float) and out["projections"]["eps_1y"]["base"] != round(out["projections"]["eps_1y"]["base"]) or True)
check("US: eps_basis forward", out["projections"]["eps_basis"] == "forward")
check("US: 가이던스 축 '데이터 없음'", out["evaluation_axes"][4]["score"] is None and NO in out["evaluation_axes"][4]["status"])
check("US: 데이터 없음 목록에 RPO·가이던스", any("RPO" in m for m in out["data_quality"]["missing"]) and any("가이던스" in m for m in out["data_quality"]["missing"]))
check("US: coverage<1 정규화 점수", out["verdict"]["score"] is not None and out["verdict"]["coverage"] < 1.0, out["verdict"])
print("   verdict:", out["verdict"], "| regime:", out["regime"]["earnings_regime"], "| missing:", out["data_quality"]["missing"])

# ------------------------------------------------------------------
# 2) 한국: DART 가짜 클라이언트 + yfinance는 추정치 없음
# ------------------------------------------------------------------
def dart_df(rows):
    # rows: (sj_div, account_id, account_nm, thstrm, add, frmtrm)
    return pd.DataFrame([{"sj_div": r[0], "account_id": r[1], "account_nm": r[2], "thstrm_amount": r[3],
                          "thstrm_add_amount": r[4], "frmtrm_amount": r[5]} for r in rows])

DB = {}
def put(y, rc, sales, op, ni, eps, add_sales=None, ar=None, inv=None, ocf=None, capex=None, ocf_prev=None):
    rows = [("IS", "ifrs-full_Revenue", "매출액", str(sales), str(add_sales) if add_sales else "", ""),
            ("IS", "dart_OperatingIncomeLoss", "영업이익", str(op), "", ""),
            ("IS", "ifrs-full_ProfitLoss", "당기순이익", str(ni), "", ""),
            ("IS", "ifrs-full_BasicEarningsLossPerShare", "기본주당이익", str(eps), "", "")]
    if ar is not None:
        rows += [("BS", "ifrs-full_TradeAndOtherCurrentReceivables", "매출채권", f"{ar:,}", "", ""),
                 ("BS", "ifrs-full_Inventories", "재고자산", f"{inv:,}", "", "")]
    if ocf is not None:
        rows += [("CF", "ifrs-full_CashFlowsFromUsedInOperatingActivities", "영업활동현금흐름", f"{ocf:,}", "", str(ocf_prev)),
                 ("CF", "ifrs-full_PurchaseOfPropertyPlantAndEquipmentClassifiedAsInvestingActivities", "유형자산의 취득", f"{capex:,}", "", "")]
    DB[(y, rc)] = dart_df(rows)

# 올해(2026): Q3 존재, H1, Q1 / 작년(2025): 전 보고서
put(2026, "11014", 80_000, 12_000, 10_000, 1400, add_sales=230_000, ar=40_000, inv=50_000)
put(2026, "11012", 74_000, 10_000, 8_500, 1200, add_sales=150_000)
put(2026, "11013", 76_000, 9_000, 8_000, 1100, add_sales=76_000)
put(2025, "11014", 66_000, 7_000, 6_000, 900, add_sales=190_000, ar=34_000, inv=40_000)
put(2025, "11012", 62_000, 6_500, 5_500, 850, add_sales=124_000)
put(2025, "11013", 62_000, 6_000, 5_000, 800, add_sales=62_000)
put(2025, "11011", 260_000, 28_000, 22_000, 3300, ocf=30_000, capex=20_000, ocf_prev=25_000, ar=36_000, inv=42_000)

class FakeDart:
    def finstate_all(self, corp, year, reprt_code="11011", fs_div="CFS"):
        if fs_div != "CFS":
            return None
        if (year, reprt_code) in DB:
            return DB[(year, reprt_code)]
        raise RuntimeError("조회된 데이터가 없습니다")

engine._get_dart_client = lambda: FakeDart()
class KrTicker(FakeTicker):
    @property
    def earnings_estimate(self): return pd.DataFrame()
    @property
    def eps_trend(self): return pd.DataFrame()
    @property
    def info(self): return {"trailingEps": 3500.0, "sharesOutstanding": 5.9e9}
fake_yf.Ticker = lambda sym: KrTicker(sym)

dk = engine.fetch_real_earnings_inputs("005930.KS")
check("KR: 분기 매출 YoY = 80000/66000-1", abs(dk["actual_vs_cons"]["sales_yoy"] - (80_000 / 66_000 - 1)) < 1e-9, dk["actual_vs_cons"]["sales_yoy"])
check("KR: QoQ = 80000/74000-1", abs(dk["actual_vs_cons"]["sales_qoq"] - (80_000 / 74_000 - 1)) < 1e-9)
check("KR: 성장 궤적 3개", dk["sales_yoy_trajectory"] is not None and all(x is not None for x in dk["sales_yoy_trajectory"]), dk["sales_yoy_trajectory"])
check("KR: AR YoY = 40000/34000-1", abs(dk["quality"]["ar_yoy"] - (40_000 / 34_000 - 1)) < 1e-9)
check("KR: 영업CF(연간)", dk["quality"]["ocf_ttm"] == 30_000 and dk["quality"]["ocf_prev_ttm"] == 25_000)
check("KR: 추정치 없으면 미래 EPS None", dk["forward"]["fy1"]["base"] is None)
check("KR: TTM EPS(yfinance) 사용", dk["ttm_eps"] == 3500.0)
ok = engine.analyze_earnings_track("005930.KS")
check("KR: eps_basis ttm_reference", ok["projections"]["eps_basis"] == "ttm_reference", ok["projections"])
check("KR: 리비전 '데이터 없음'", ok["revision_dynamics"]["status"] == NO and ok["evaluation_axes"][1]["score"] is None)
print("   KR verdict:", ok["verdict"], "| regime:", ok["regime"]["earnings_regime"], "| missing:", ok["data_quality"]["missing"])

# ------------------------------------------------------------------
# 3) 전부 실패 → 판정 불가 (크래시 없음)
# ------------------------------------------------------------------
fake_yf.Ticker = lambda sym: FakeTicker(sym, mode="empty")
engine._get_dart_client = lambda: None
for tk in ("ZZZZ", "000000.KS"):
    on = engine.analyze_earnings_track(tk)
    check(f"{tk}: 판정 불가 NA", on["regime"]["earnings_regime"] == "NA", on["regime"])
    check(f"{tk}: eps_basis none", on["projections"]["eps_basis"] == "none")
    check(f"{tk}: 점수 None·등급 ⚪", on["verdict"]["score"] is None and on["verdict"]["grade"] == "⚪")
    check(f"{tk}: 표 문자열에 '데이터 없음'", all(r["actual"] == NO for r in on["e1_surprise_table"]))

print("\nFAILS:", fails)
sys.exit(1 if fails else 0)
