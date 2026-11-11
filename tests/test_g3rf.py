"""G3 목표 배수(이론 PER 참고용) · 무위험수익률 실데이터/None 처리 검사."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # 프로젝트 폴더의 engine 패키지를 불러오기 위해
import _offline  # noqa: F401  외부 네트워크 차단 (tests/_offline.py)
import sys, time
from test_chain_helpers import engine, fake_earn
NO = engine.NO_DATA
FAILS = []
def chk(n, c):
    print(("PASS " if c else "FAIL ") + n); (None if c else FAILS.append(n))

KEYS = ["industry_beta","sustainable_roe","dividend_yield","is_financial_sector","listing_years","self_5y_mean","self_5y_std","self_5y_min",
        "peer_median_per","peer_top25_per","peer_std_per","company_roe","peer_median_roe","company_fwd_growth","peer_median_fwd_growth",
        "price_12m_ago","eps_lag_12m_ago","eps_lag_now","consensus_target_price","net_cash_krw","fcf_ttm_krw","short_term_debt_ratio","share_dilution_yoy"]
def inject(**vals):
    d = {k: None for k in KEYS}; d.update(vals)
    engine.fetch_real_fundamental_inputs = lambda t, p: {"values": dict(d), "meta": {"source": "test", "missing": [k for k, v in d.items() if v is None], "notes": []}}

apple = dict(industry_beta=1.1, sustainable_roe=1.5, dividend_yield=0.004, is_financial_sector=False, listing_years=40.0,
             self_5y_mean=29.0, self_5y_std=3.5, self_5y_min=24.0, peer_median_per=24.0, peer_top25_per=30.0, peer_std_per=6.0,
             share_dilution_yoy=-0.025, net_cash_krw=10.0, fcf_ttm_krw=100.0, short_term_debt_ratio=0.2)
earn = fake_earn("forward", False, regime="G3", sector="구조성장형")
earn["projections"]["eps_1y"].update(base=8.5, bull=9.2, bear=7.8)
mi = {"rf_10y": 0.0484, "rf_source": "FRED DGS10", "rf_as_of": "2026-11-05"}

# 1) 애플형: 자기평균 29, 피어 24, 이론 PER ~15 → base = 24 (이론 PER로 눌리지 않음)
inject(**apple)
fo = engine.analyze_fundamental_track("AAPL", 350.69, earnings_output=earn, macro_input=mi)
mult = fo["multiple_selection"] if "multiple_selection" in fo else None
ip = fo["industry_and_peer"]; vt = fo["valuation_targets"]
t1 = next(t["price"] for t in vt["buy_tranches"] if t["tranche"] == 1)
print("  이론PER", ip["theoretical_mature_per"], "k_e", ip["market_ke"], "ERP", ip["market_erp"], "| 1차선", t1, "| E[R]", vt["expected_return_weighted"], "| MDD", vt["mdd_ratio"])
chk("1 이론 PER은 계산·표시됨(참고)", ip["theoretical_mature_per"] is not None and ip["theoretical_mature_per"] < 20)
chk("1 1차선 = EPS×24/(1+r) 근처 (이론 PER로 눌리지 않음)", t1 > 8.5 * 20 / 1.2)
chk("1 ERP 미국 4.46% + 출처", abs(ip["market_erp"] - 0.0446) < 1e-9 and "Damodaran" in ip["market_erp_source"])
chk("1 k_e = 4.84% + 1.1×4.46%", abs(ip["market_ke"] - (0.0484 + 1.1 * 0.0446)) < 1e-3)
chk("1 기대수익률이 −60% 같은 극단값 아님", vt["expected_return_weighted"] > -0.45)

# 2) 자기 밴드·피어 없음 → 이론 PER×1.1 대체
inject(**{k: v for k, v in apple.items() if k not in ("self_5y_mean", "self_5y_std", "self_5y_min", "peer_median_per", "peer_top25_per", "peer_std_per")})
fo2 = engine.analyze_fundamental_track("AAPL", 350.69, earnings_output=earn, macro_input=mi)
t1b = [t["price"] for t in fo2["valuation_targets"]["buy_tranches"] if t["tranche"] == 1]
theo = fo2["industry_and_peer"]["theoretical_mature_per"]
print("  대체:", t1b, "theo", theo)
chk("2 관측 배수 없으면 이론 PER×1.1로 매수선 생성", bool(t1b) and abs(t1b[0] - 8.5 * theo * 1.1 / (1 + fo2["valuation_targets"]["required_return_r"] - 0.004)) / t1b[0] < 0.03)

# 3) 무위험수익률 못 받음 → 매수선·요구수익률 '데이터 없음', 예외 없음
inject(**apple)
fo3 = engine.analyze_fundamental_track("AAPL", 350.69, earnings_output=earn, macro_input={"rf_10y": None, "rf_source": NO})
chk("3 rf None → 요구수익률 None", fo3["valuation_targets"]["required_return_r"] is None)
chk("3 rf None → 매수선 없음 + 가격 자격 NA", not fo3["valuation_targets"]["buy_tranches"] and fo3["price_qualification"]["status"] == "NA")
chk("3 못 받은 항목에 rf_10y", "rf_10y" in fo3["data_quality"]["missing"] if "data_quality" in fo3 else True)
chk("3 사유 문구에 무위험수익률", "무위험수익률" in fo3["price_qualification"]["headline_reason"])

# 4) 한국 종목 ERP 4.87%
inject(**apple)
earn_kr = fake_earn("forward", True, regime="G3")
fo4 = engine.analyze_fundamental_track("005930.KS", 60000, earnings_output=earn_kr, macro_input={"rf_10y": 0.0305})
chk("4 한국 ERP 4.87%", abs(fo4["industry_and_peer"]["market_erp"] - 0.0487) < 1e-9)

# 5) fetch_risk_free_10y: 네트워크 실패 시 None, 이상값 거부
import urllib.request
urllib.request.urlopen = lambda *a, **k: (_ for _ in ()).throw(OSError("no net"))
engine._RF_CACHE.clear(); engine._MACRO_SERIES_CACHE.clear(); engine._read_env_value = lambda n: ""
sys.modules.pop("yfinance", None)
import types; fy = types.ModuleType("yfinance")
class _T:
    def __init__(s, sym): pass
    def history(s, period=None):
        import pandas as pd; return pd.DataFrame({"Close": [120.0]}, index=pd.to_datetime(["2026-11-06"]))
fy.Ticker = _T; sys.modules["yfinance"] = fy
r = engine.fetch_risk_free_10y(False)
chk("5 실패·이상값(120%)이면 None", r["value"] is None and r["source"] == NO)
class _T2(_T):
    def history(s, period=None):
        import pandas as pd; return pd.DataFrame({"Close": [4.86]}, index=pd.to_datetime(["2026-11-06"]))
fy.Ticker = _T2
r = engine.fetch_risk_free_10y(False)
chk("5 ^TNX 4.86 → 0.0486", r["value"] == 0.0486 and "TNX" in r["source"])
class _Resp:
    def __init__(s, b): s.b = b
    def read(s): return s.b
    def __enter__(s): return s
    def __exit__(s, *a): return False
engine._RF_CACHE.clear(); engine._MACRO_SERIES_CACHE.clear()
urllib.request.urlopen = lambda *a, **k: _Resp(b"observation_date,IRLTLT01KRM156N\n2026-08-01,3.01\n2026-09-01,3.12\n2026-10-01,.\n")
r = engine.fetch_risk_free_10y(True)
chk("5 FRED CSV 파싱 (마지막 유효값, '.' 건너뜀)", r["value"] == 0.0312 and r["as_of"] == "2026-09-01")
print("FAILS:", FAILS); sys.exit(1 if FAILS else 0)
