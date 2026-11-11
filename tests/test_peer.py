"""피어 통계(같은 유효 집합·이상치 제외·섹터 대체 없음)와 내재 성장률 종착 배수 검사. 애플 실측 점검값으로 재현."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # 프로젝트 폴더의 engine 패키지를 불러오기 위해
import _offline  # noqa: F401  외부 네트워크 차단 (tests/_offline.py)
import sys, types
import pandas as pd
from test_chain_helpers import engine, fake_earn
NO = engine.NO_DATA
FAILS = []
def chk(n, c):
    print(("PASS " if c else "FAIL ") + n); (None if c else FAILS.append(n))

INFO = {  # check_peer_data.py AAPL 결과 (2026-11-10)
    "SONO": (13.03448, 0.141759), "TBCH": (16.49037, -0.0053), "FXHO": (None, -3092.88), "UEIC": (10.72727, -0.04981), "AXIL": (None, 0.23136),
    "NVDA": (11.21731, 1.17211), "MSFT": (22.53211, 0.34221), "AVGO": (16.41299, 0.44245), "SKHY": (3.176902, 0.92542), "AMD": (28.75043, 0.1196),
    "MU": (3.528823, 0.8826),
}
def make_yf(industry_syms, sector_syms):
    yf = types.ModuleType("yfinance")
    class _G:
        def __init__(s, key): s.key = key
        @property
        def top_companies(s):
            syms = industry_syms if s.kind == "I" else sector_syms
            return pd.DataFrame({"name": syms}, index=syms)
    class Industry(_G): kind = "I"
    class Sector(_G): kind = "S"
    class Ticker:
        def __init__(s, sym): s.sym = sym
        @property
        def info(s):
            pe, roe = INFO.get(s.sym, (None, None))
            return {"forwardPE": pe, "returnOnEquity": roe, "trailingEps": 5.0, "forwardEps": 5.5}
    yf.Industry, yf.Sector, yf.Ticker = Industry, Sector, Ticker
    return yf

# 1) 애플 실제 상황: 산업 피어 5개(유효 PER 3개) → 피어 통계 전부 '데이터 없음', 섹터로 넘어가지 않음
sys.modules["yfinance"] = make_yf(["AAPL", "SONO", "TBCH", "FXHO", "UEIC", "AXIL"], ["NVDA", "MSFT", "AVGO", "SKHY", "AMD", "MU"])
meta = {"notes": []}
out = engine._peer_stats_us(None, {"industryKey": "consumer-electronics", "sectorKey": "technology"}, "AAPL", meta)
print("  ", out, meta.get("peer_group"), meta["notes"][-1:])
chk("1 피어 PER 통계 없음 (유효 3개)", out["peer_median_per"] is None and out["peer_top25_per"] is None)
chk("1 피어 ROE 중앙값도 없음 (ROE −3,092 포함 왜곡 제거)", out["peer_median_roe"] is None)
chk("1 섹터(반도체 섞임)로 넘어가지 않음", meta.get("peer_group", "").startswith("Industry"))

# 2) 유효 피어 충분: ROE 이상치 제외, ROE는 PER 유효 기업에서만
INFO.update({"P1": (20.0, 0.30), "P2": (22.0, 0.25), "P3": (18.0, 0.20), "P4": (25.0, -3092.0), "P5": (19.0, 0.18), "P6": (None, 5.0)})
sys.modules["yfinance"] = make_yf(["X", "P1", "P2", "P3", "P4", "P5", "P6"], [])
meta = {"notes": []}
out2 = engine._peer_stats_us(None, {"industryKey": "x"}, "X", meta)
print("  ", out2)
chk("2 피어 PER 중앙값 = 20.0 (18,19,20,22,25)", abs(out2["peer_median_per"] - 20.0) < 1e-9)
chk("2 ROE는 이상치(−3,092)·PER 없는 P6 제외 → 4개라 '데이터 없음'", out2["peer_median_roe"] is None)
INFO["P7"] = (21.0, 0.22)
sys.modules["yfinance"] = make_yf(["X", "P1", "P2", "P3", "P4", "P5", "P6", "P7"], [])
out3 = engine._peer_stats_us(None, {"industryKey": "x"}, "X", {"notes": []})
chk("2 ROE 유효 5개면 중앙값 = 0.22", out3["peer_median_roe"] is not None and abs(out3["peer_median_roe"] - 0.22) < 1e-9)

# 3) 내재 5년 성장률: 종착 배수 = 채택 Base 배수
KEYS = ["industry_beta", "sustainable_roe", "dividend_yield", "self_5y_mean", "self_5y_std", "self_5y_min", "peer_median_per",
        "peer_top25_per", "peer_std_per", "share_dilution_yoy", "net_cash_krw", "fcf_ttm_krw", "short_term_debt_ratio"]
d = {k: None for k in KEYS}
d.update(industry_beta=1.1, sustainable_roe=1.42, dividend_yield=0.004, self_5y_mean=37.1, self_5y_std=5.1, self_5y_min=26.0)
engine.fetch_real_fundamental_inputs = lambda t, p: {"values": dict(d), "meta": {"source": "t", "missing": [], "notes": []}}
earn = fake_earn("forward", False, regime="G3")
earn["projections"]["eps_1y"].update(base=9.59, bull=10.4, bear=8.8)
earn["projections"]["eps_2y"] = {"formula": "x", "base": 9.59 * 1.097 ** 2 / 1.097}
fo = engine.analyze_fundamental_track("AAPL", 350.0, earnings_output=earn, macro_input={"rf_10y": 0.0486})
rr = fo["rerating_diagnosis"] if "rerating_diagnosis" in fo else next(v for v in fo.values() if isinstance(v, dict) and "terminal_per_used" in v)
print("   종착", rr["terminal_per_used"], rr.get("terminal_per_basis"), "내재", rr["implied_5y_cagr"])
chk("3 종착 배수 = Base 37.1배", abs(rr["terminal_per_used"] - 37.1) < 1e-9 and rr["terminal_per_basis"] == "채택 Base 배수")
chk("3 내재 성장률이 30%대로 과장되지 않음 (약 10~13%)", 0.08 < rr["implied_5y_cagr"] < 0.15)
print("FAILS:", FAILS); sys.exit(1 if FAILS else 0)
