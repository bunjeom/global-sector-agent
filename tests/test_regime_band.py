"""엔비디아형 검사: (1) 이익 급변 종목의 로그 PER 밴드, (2) G3는 저성장 근거가 있을 때만."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # 프로젝트 폴더의 engine 패키지를 불러오기 위해
import _offline  # noqa: F401  외부 네트워크 차단 (tests/_offline.py)
import sys, types, math
import numpy as np, pandas as pd
FAILS = []
def chk(n, c):
    print(("PASS " if c else "FAIL ") + n); (None if c else FAILS.append(n))

# ---- 가짜 yfinance (분기 재무) : 매출 성장률·이익의 질 경고를 시나리오별로 바꾼다
SCEN = {}
def qdf(rows):
    cols = pd.to_datetime(["2026-07-31", "2026-04-30", "2026-01-31", "2025-10-31", "2025-07-31"])
    return pd.DataFrame({c: [r[i] for r in rows.values()] for i, c in enumerate(cols)}, index=list(rows.keys()))
class T:
    def __init__(s, sym): s.sym = sym
    def history(s, period="2y", **k):
        idx = pd.bdate_range(end="2026-11-09", periods=300); c = np.linspace(100, 180, 300)
        return pd.DataFrame({"Open": c, "High": c, "Low": c, "Close": c, "Volume": 1e6}, index=idx)
    info = property(lambda s: {"shortName": "X", "trailingEps": 4.9, "sharesOutstanding": 2.4e10})
    calendar = property(lambda s: {})
    quarterly_income_stmt = property(lambda s: qdf(SCEN["is"]))
    quarterly_balance_sheet = property(lambda s: qdf(SCEN["bs"]))
    quarterly_cashflow = property(lambda s: qdf(SCEN["cf"]))
    earnings_estimate = property(lambda s: pd.DataFrame({"avg": [9.5, 14.0], "low": [8.5, 12.0], "high": [10.5, 16.0], "numberOfAnalysts": [50, 48]}, index=["0y", "+1y"]))
    eps_trend = property(lambda s: pd.DataFrame({"current": [9.5, 14.0], "30daysAgo": [9.4, 13.9], "90daysAgo": [8.9, 12.0]}, index=["0y", "+1y"]))
    earnings_history = property(lambda s: pd.DataFrame({"epsActual": [1.1], "epsEstimate": [1.0]}, index=pd.to_datetime(["2026-07-31"])))
yf = types.ModuleType("yfinance"); yf.Ticker = T; yf.download = lambda *a, **k: T("x").history()
sys.modules["yfinance"] = yf
import importlib, engine
importlib.reload(engine)

# 고성장(+100%) + 이익의 질 경고(매출채권·재고 급증, 현금 전환 낮음)
SCEN.update(
    is_={},
    **{"is": {"Total Revenue": [96, 80, 68, 57, 48], "Gross Profit": [72, 60, 51, 42, 35], "Operating Income": [62, 51, 43, 35, 29],
              "Net Income": [52, 43, 36, 30, 25], "Diluted EPS": [2.1, 1.75, 1.47, 1.22, 1.02]},
       "bs": {"Accounts Receivable": [60, 45, 35, 28, 22], "Inventory": [25, 20, 17, 15, 12]},
       "cf": {"Operating Cash Flow": [30, 28, 25, 22, 20], "Capital Expenditure": [-3, -3, -2, -2, -2]}})
e1 = engine.analyze_earnings_track("NVDA")
r1 = e1["regime"]
print("  고성장+경고:", r1["earnings_regime"], "|", r1["regime_label"][:90])
chk("1 고성장인데 G2 조건 미충족이면 G3가 아님", r1["earnings_regime"] != "G3")
chk("1 '판정 보류'와 미충족 사유 표시", r1["earnings_regime"] == "NA" and "판정 보류" in r1["regime_label"] and "경고" in r1["regime_label"])
chk("1 헤드라인도 보류 사유", "G2 조건 미충족" in e1["verdict"]["headline_reason"])

# 저성장(+4%) → G3
SCEN.update(**{"is": {"Total Revenue": [104, 103, 102, 101, 100], "Gross Profit": [45, 44, 44, 43, 43], "Operating Income": [30, 29, 29, 28, 28],
                      "Net Income": [25, 24, 24, 23, 23], "Diluted EPS": [2.5, 2.4, 2.4, 2.3, 2.3]},
               "bs": {"Accounts Receivable": [30, 30, 30, 30, 30], "Inventory": [20, 20, 20, 20, 20]},
               "cf": {"Operating Cash Flow": [30, 29, 29, 28, 28], "Capital Expenditure": [-5, -5, -5, -5, -5]}})
e2 = engine.analyze_earnings_track("AAPL")
print("  저성장:", e2["regime"]["earnings_regime"], "|", e2["regime"]["regime_label"][:60])
chk("2 저성장(매출 +4%)은 G3 가능", e2["regime"]["earnings_regime"] in ("G3", "G4"))

# ---- (2) 로그 PER 밴드: 확정 EPS가 0.17 → 4.9로 급증, 주가 15 → 180
dates = pd.date_range("2022-11-30", "2026-10-31", freq="ME")
prices = [(d, float(15 * (180 / 15) ** (i / (len(dates) - 1)))) for i, d in enumerate(dates)]
pairs = [(pd.Timestamp("2023-01-29"), 0.17), (pd.Timestamp("2024-01-28"), 1.19), (pd.Timestamp("2025-01-26"), 2.94), (pd.Timestamp("2026-01-25"), 4.90)]
b = engine._lagged_ratio_stats(prices, pairs, 60, 5, 24)
vals = [v for _, v in b["series"]]
arith_m, arith_sd = float(np.mean(vals)), float(np.std(vals))
print(f"  산술 평균 {arith_m:.1f}배 σ {arith_sd:.1f} (예전 방식) → 기하평균 {b['mean']:.1f}배, 로그σ {b['log_std']:.2f} (+1σ {b['mean']*math.exp(b['log_std']):.1f} / −1σ {b['mean']*math.exp(-b['log_std']):.1f})")
chk("3 기하평균이 산술평균보다 극단값 영향이 작음", b["mean"] < arith_m)
chk("3 −1σ가 항상 양수(산술 μ−σ처럼 음수·하한 붙음 없음)", b["mean"] * math.exp(-b["log_std"]) > 0)
chk("3 밴드 계산에 쓴 월별 값을 함께 돌려줌", len(b["series"]) == b["n"] and b["n"] >= 24)

# ---- (3) V 트랙: 로그 σ로 2·3차 매수선이 $13 같은 값으로 무너지지 않음
KEYS = ["industry_beta", "sustainable_roe", "dividend_yield", "self_5y_mean", "self_5y_std", "self_5y_log_std", "self_5y_min",
        "peer_median_per", "peer_top25_per", "peer_std_per", "share_dilution_yoy", "net_cash_krw", "fcf_ttm_krw", "short_term_debt_ratio"]
d = {k: None for k in KEYS}
d.update(industry_beta=2.3, sustainable_roe=1.0, dividend_yield=0.0002, self_5y_mean=b["mean"], self_5y_std=b["std"],
         self_5y_log_std=b["log_std"], self_5y_min=b["min"], peer_median_per=22.5, peer_top25_per=29.1, peer_std_per=8.0)
engine.fetch_real_fundamental_inputs = lambda t, p: {"values": dict(d), "meta": {"source": "t", "missing": [], "notes": []}}
from test_chain_helpers import fake_earn
er = fake_earn("forward", False, regime="NA")
er["projections"]["eps_1y"].update(base=14.80, bull=17.28, bear=12.20)
fo = engine.analyze_fundamental_track("NVDA", 178.67, earnings_output=er, macro_input={"rf_10y": 0.0486})
vt = fo["valuation_targets"]
tr = {t["tranche"]: t["price"] for t in vt["buy_tranches"]}
print("  매수선", tr, "| 목표배수", vt["base_multiple_target"], vt["bull_multiple_target"], vt["bear_multiple_target"], "| 매도", vt["sell_target"])
chk("4 2차선이 1차선의 70% 이상 (예전 $13 붕괴 없음)", tr.get(2) is not None and tr[2] >= 0.7 * tr[1])
chk("4 3차선이 1차선의 50% 이상", tr.get(3) is not None and tr[3] >= 0.5 * tr[1])
chk("4 Bear 배수가 4배 하한에 붙지 않음", vt["bear_multiple_target"] > 4.0)

# ---- (4) 밴드 신뢰 게이트: 이익 급변이면 제외, 안정적이면 사용
chk("5 엔비디아형(EPS 0.17→4.9) 밴드는 신뢰 불가", bool(engine._band_unreliable_reasons(b)))
calm = engine._lagged_ratio_stats([(d, 150.0 + 20 * math.sin(i / 3)) for i, d in enumerate(dates)],
                                  [(pd.Timestamp("2023-01-29"), 5.0), (pd.Timestamp("2024-01-28"), 5.5), (pd.Timestamp("2025-01-26"), 6.0), (pd.Timestamp("2026-01-25"), 6.4)], 60, 5, 24)
print(f"  안정형: 로그σ {calm['log_std']:.2f}, EPS {calm['eps_lo']}→{calm['eps_hi']}")
chk("5 안정형(애플류) 밴드는 사용", engine._band_unreliable_reasons(calm) == [])
# ---- (5) 자기 밴드 없음(제외) → 목표 배수 = 피어, 폭 = 피어 σ, 매도목표가 과대 없음
d2 = dict(d); d2.update(self_5y_mean=None, self_5y_std=None, self_5y_log_std=None, self_5y_min=None)
engine.fetch_real_fundamental_inputs = lambda t, p: {"values": dict(d2), "meta": {"source": "t", "missing": [], "notes": [], "band_excluded": "이익 규모 급변"}}
fo2 = engine.analyze_fundamental_track("NVDA", 178.67, earnings_output=er, macro_input={"rf_10y": 0.0486})
vt2 = fo2["valuation_targets"]
print("  제외 후: 배수", vt2["base_multiple_target"], vt2["bull_multiple_target"], vt2["bear_multiple_target"], "| 매도", vt2["sell_target"], "| 폭", fo2["input_data"]["scenario_width_source"])
chk("6 목표 배수 = 피어 중앙값 22.5", vt2["base_multiple_target"] == 22.5)
chk("6 Bull/Bear 폭이 피어 σ로 생김 (Base와 다름)", vt2["bull_multiple_target"] > 22.5 > vt2["bear_multiple_target"] > 4.0)
chk("6 시나리오 폭 출처 = 피어 PER σ", fo2["input_data"]["scenario_width_source"] == "피어 PER σ")
chk("6 매도목표가가 현재가 3배 이상으로 튀지 않음", vt2["sell_target"] is not None and vt2["sell_target"] < 3 * 178.67)

# ---- (6) 판정 보류(NA) 표시 일관성: 규칙 문구, 종합 가격 자격이 가치 트랙보다 높지 않음
chk("7 NA 국면이면 규칙 문구가 '판정 보류 → 중립 규칙'", "판정 보류" in vt2.get("rule_description", "") or "판정 보류" in str(fo2.get("multiple_selection", {}).get("rule_description", "")))
from test_chain_helpers import prices as _prices
_st = _prices(start=178.0, seed=3); _bm = _prices(start=17000, seed=9)
_mac = engine._macro_hold_result("NVDA", False, "x", "y", "테스트")
_acct = {"total_equity": 1e8, "t_sleeve_ratio": 0.3, "is_krx": False, "fx_to_krw": 1400.0, "hard_block_evt": False, "earnings_blackout": False,
         "mktcap_krw": 1e15, "loss_streak": 0, "existing_v_weight": 0.0}
_vmi = {"thesis_status": fo2["thesis_status"], "kill_flags": [], "buy_tranches": fo2["buy_tranches"], "sell_target": fo2["sell_target"],
        "position_cap": fo2["position_cap"], "real_signal": "중립", "analyzed_at": "2026-11-11", "macro_regime_override": None}
_vix = pd.Series(18.0, index=_st.index)
_tech = engine.analyze_technical_track(df=_st, bench_df=_bm, vix_series=_vix, value_macro_input=_vmi, account_cfg=_acct)
_syn = engine.synthesize_4tracks(ticker="NVDA", current_price=178.67, tech_out=_tech, earn_out=er, fund_out=fo2, macro_out=_mac,
                                 synth_cfg={"stock_name": "NVDA", "days_to_earnings": None})
print("  가치 트랙:", fo2["price_qualification"]["status"], "| 종합:", _syn["v_action_card"]["price_qual_code"])
chk("7 종합 가격 자격이 가치 트랙 '조건부'를 '충족'으로 올리지 않음",
    not (fo2["price_qualification"]["status"] == "CONDITIONAL" and _syn["v_action_card"]["price_qual_code"] == "QUALIFIED"))
print("FAILS:", FAILS); sys.exit(1 if FAILS else 0)
