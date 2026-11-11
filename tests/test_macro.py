"""거시 트랙 실데이터화 검사: 난수 시계열 제거, 실제 수준값 사용, 못 받은 변수·보류 처리, 종합 연동."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # 프로젝트 폴더의 engine.py를 불러오기 위해
import _offline  # noqa: F401  외부 네트워크 차단 (tests/_offline.py)
import sys, copy, os
import numpy as np, pandas as pd
from test_chain_helpers import engine, prices, run_chain, fake_earn
NO = engine.NO_DATA
FAILS = []
def chk(n, c):
    print(("PASS " if c else "FAIL ") + n); (None if c else FAILS.append(n))

stock = prices(start=120.0, seed=3); bench = prices(start=17000, seed=9)
widx = pd.date_range(end=stock.index[-1], periods=200, freq="W-FRI")
rng = np.random.default_rng(7)   # 테스트 입력용 (앱 코드 아님)
def lv(start, sd, kind):
    x = np.cumsum(rng.normal(0, sd, len(widx)))
    return pd.Series(start + x if kind == "bp" else start * np.exp(x), index=widx)
LEVELS = {"DFII10": lv(1.9, 0.05, "bp"), "T10YIE": lv(2.3, 0.04, "bp"), "T10Y2Y": lv(0.35, 0.05, "bp"),
          "BAMLC0A0CM": lv(0.85, 0.02, "bp"), "BAMLH0A0HYM2": lv(3.0, 0.08, "bp"), "DEXKOUS": lv(1380, 0.008, "ln"),
          "DCOILWTICO": lv(72, 0.03, "ln"), "ICSA": lv(225000, 0.03, "ln"), "NETLIQ": lv(5.9e6, 0.01, "ln"), "KR10Y": lv(3.0, 0.05, "bp")}
def inject(levels):
    engine.fetch_macro_levels = lambda ids, start, end=None: {
        "levels": {k: v for k, v in levels.items() if k in ids}, "missing": [k for k in ids if k not in levels],
        "sources": {k: f"FRED {k}" for k in levels if k in ids}}

IND = "AI 반도체·네트워킹(팹리스)"
# 1) 전체 데이터
inject(LEVELS)
m1 = engine.analyze_macro_track("NVDA", stock, bench, IND)
chk("1 상태 완료", m1["status"] == "완료")
chk("1 데이터 품질 🟢 · 못 받은 변수 없음", m1["meta"]["data_quality"] == "🟢" and not m1["meta"]["missing_vars"])
chk("1 국면 판정됨 (T10YIE·DFII10)", m1["macro_regime"]["current_regime"] != NO and m1["macro_regime"]["inflation_week_ratio_13w"] is not None)
st13 = {r["var_name"]: r for r in m1["macro_state_13w"]}
dfii = next(r for n, r in st13.items() if "실질" in n)
lo, hi = LEVELS["DFII10"].loc[:stock.index[-1]].iloc[-13:].min(), LEVELS["DFII10"].loc[:stock.index[-1]].iloc[-13:].max()
print("  DFII10 13주:", dfii["high_low_13w"], "| 실제", round(lo, 2), round(hi, 2))
chk("1 13주 고저가 실제 수준값", f"{lo:,.2f}" in dfii["high_low_13w"] and f"{hi:,.2f}" in dfii["high_low_13w"])
bad = [c["threshold_rule"] for c in m1["watch_calendar"] if "bp" in c["threshold_rule"] and "%" in c["threshold_rule"]]
chk("1 경계 규칙에 bp·% 혼용(685.8% 버그) 없음", not bad)
chk("1 감시 캘린더에 출처", all(c.get("source", NO) != NO for c in m1["watch_calendar"]))
# 2) 같은 데이터면 종목 이름이 달라도 회귀 결과 동일 (종목명 시드 난수 제거 확인)
m1b = engine.analyze_macro_track("ZZZZ", stock, bench, IND)
chk("2 종목명에 따라 결과가 바뀌지 않음", [s["estimated_beta"] for s in m1["sensitivities"]] == [s["estimated_beta"] for s in m1b["sensitivities"]])
# 데이터가 바뀌면 결과도 바뀜
L2 = dict(LEVELS); L2["NETLIQ"] = LEVELS["NETLIQ"] * np.exp(0.05 * np.sin(np.arange(len(widx)) * 1.7))
inject(L2)
m1c = engine.analyze_macro_track("NVDA", stock, bench, IND)
chk("2 입력 데이터가 바뀌면 결과가 바뀜", [s["estimated_beta"] for s in m1["sensitivities"]] != [s["estimated_beta"] for s in m1c["sensitivities"]])
# 3) 일부 누락
inject({k: v for k, v in LEVELS.items() if k not in ("NETLIQ", "T10YIE")})
m3 = engine.analyze_macro_track("NVDA", stock, bench, IND)
print("  누락:", m3["meta"]["missing_vars"], m3["meta"]["regime_missing"], m3["macro_regime"]["current_regime"])
chk("3 상태 완료 + 🟡", m3["status"] == "완료" and m3["meta"]["data_quality"] == "🟡")
chk("3 못 받은 변수 표시", any("순유동성" in x for x in m3["meta"]["missing_vars"]))
chk("3 회귀 변수에서 제외", not any("순유동성" in s["var_name"] for s in m3["sensitivities"]))
chk("3 국면 데이터 없음 + 인플레 비중 None", m3["macro_regime"]["current_regime"] == NO and m3["macro_regime"]["inflation_week_ratio_13w"] is None)
chk("3 국면별 베타 '데이터 없음'", all(s["regime_beta_str"] == NO for s in m3["sensitivities"]))
# 4) 전부 없음 → 보류
inject({})
m4 = engine.analyze_macro_track("NVDA", stock, bench, IND)
chk("4 보류", m4["status"] == "보류" and m4["classification"]["r2_macro_own"] is None)
chk("4 보류 사유", "받지 못함" in m4["meta"]["hold_reason"])
# 5) 주가 없음 → 보류 (난수 주가 생성 안 함)
inject(LEVELS)
m5 = engine.analyze_macro_track("NVDA", None, bench, IND)
chk("5 주가 없음 → 보류", m5["status"] == "보류" and "주가" in m5["meta"]["hold_reason"])
m5b = engine.analyze_macro_track("NVDA", stock, None, IND)
chk("5 지수 없음 → 보류", m5b["status"] == "보류" and "지수" in m5b["meta"]["hold_reason"])
# 6) 종합: 거시 보류여도 가짜 13주 분해 없음
earn = fake_earn("forward", False)
last, fund_out, tech_out, _ = run_chain("NVDA", earn, 120.0)
syn = engine.synthesize_4tracks(ticker="NVDA", current_price=last, tech_out=tech_out, earn_out=earn, fund_out=fund_out,
                                macro_out=m4, synth_cfg={"stock_name": "NVDA", "days_to_earnings": None})
ov = syn["macro_risk_overlay"]
chk("6 보류 시 13주 분해 None", ov["return_13w_actual"] is None and ov["return_13w_idiosyncratic_share"] is None)
chk("6 3줄 근거에 '-5.0%' 같은 가짜값 없음", "-5.0%" not in syn["three_line_briefing"]["line_2_why"] and NO in syn["three_line_briefing"]["line_2_why"])
syn1 = engine.synthesize_4tracks(ticker="NVDA", current_price=last, tech_out=tech_out, earn_out=earn, fund_out=fund_out,
                                 macro_out=m1, synth_cfg={"stock_name": "NVDA", "days_to_earnings": None})
chk("6 정상 시 13주 분해 = 거시 트랙 값", abs(syn1["macro_risk_overlay"]["return_13w_actual"] - m1["attribution_13w"]["13w"]["actual"]) < 1e-9)
# 7) 변화량 계산
idx = pd.date_range("2026-01-02", periods=4, freq="W-FRI")
lvb = pd.Series([4.00, 4.10, 4.05, 4.20], index=idx)
ch = engine._weekly_changes(lvb, "bp", idx[1:])
chk("7 bp 변화 = Δ%×100", np.allclose(ch, [10.0, -5.0, 15.0]))
lvl = pd.Series([100.0, 110.0, 99.0, 99.0], index=idx)
chk("7 ln 변화", np.allclose(engine._weekly_changes(lvl, "ln", idx[1:]), np.log([1.1, 0.9, 1.0])))
# 8) NETLIQ 조합 (백만$ − 백만$ − 십억$×1000)
_orig = engine._fred_series
days = pd.bdate_range("2025-01-01", periods=120)
fake = {"WALCL": pd.Series(7_000_000.0, index=days), "WTREGEN": pd.Series(800_000.0, index=days), "RRPONTSYD": pd.Series(300.0, index=days),
        "ICSA": pd.Series(220000.0, index=days)}
engine._fred_series = lambda sid, start=None: fake.get(sid)
r = engine.fetch_macro_levels.__wrapped__(["NETLIQ", "ICSA", "DFII10"], "2025-01-01") if hasattr(engine.fetch_macro_levels, "__wrapped__") else None
import importlib; eng2 = importlib.reload(engine)   # inject로 바꾼 fetch_macro_levels 원복
eng2._fred_series = lambda sid, start=None: fake.get(sid)
r = eng2.fetch_macro_levels(["NETLIQ", "ICSA", "DFII10"], "2025-01-01")
chk("8 NETLIQ = 7,000,000−800,000−300×1000 = 5,900,000", abs(float(r["levels"]["NETLIQ"].iloc[-1]) - 5_900_000) < 1e-6)
chk("8 실제 마지막 관측일 기록", r["last_obs"]["ICSA"] == str(days[-1].date()))
chk("8 못 받은 DFII10은 missing", "DFII10" in r["missing"] and "DFII10" not in r["levels"])
# 9) FRED CSV 파싱 (키 없음 → 공개 CSV)
import urllib.request
class _Resp:
    def __init__(s, b): s.b = b
    def read(s): return s.b
    def __enter__(s): return s
    def __exit__(s, *a): return False
eng2 = importlib.reload(engine)
eng2._read_env_value = lambda n: ""
urllib.request.urlopen = lambda *a, **k: _Resp(b"observation_date,DFII10\n2026-11-03,1.95\n2026-11-04,.\n2026-11-05,1.98\n")
s9 = eng2._fred_series("DFII10", "2026-11-01")
chk("9 CSV 파싱 ('.' 제외)", s9 is not None and list(s9.values) == [1.95, 1.98])
# 10) 엔진 소스에 거시 난수 생성 코드가 남아 있지 않음
src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "engine.py"), encoding="utf-8").read()
a = src.index("def analyze_macro_track("); b = src.index("\ndef ", a + 10)
chk("10 analyze_macro_track에 난수 없음", "random" not in src[a:b] and "rng" not in src[a:b])

# 11) 야후 주가처럼 시간대가 붙은 주가(뉴욕·서울)여도 오류 없이 계산
inject_tz = importlib.reload(engine)
inject_tz.fetch_macro_levels = lambda ids, start, end=None: {"levels": {k: v for k, v in LEVELS.items() if k in ids}, "missing": [],
                                                             "sources": {k: "x" for k in ids}, "last_obs": {}}
for tzname in ("America/New_York", "Asia/Seoul"):
    st_tz, bm_tz = stock.copy(), bench.copy()
    st_tz.index = st_tz.index.tz_localize(tzname); bm_tz.index = bm_tz.index.tz_localize(tzname)
    try:
        mt = inject_tz.analyze_macro_track("AAPL", st_tz, bm_tz, IND)
        chk(f"11 시간대 {tzname} 주가 → 완료", mt["status"] == "완료")
    except Exception as e:
        chk(f"11 시간대 {tzname} 주가 → 예외 {type(e).__name__}: {e}", False)
print("FAILS:", FAILS); sys.exit(1 if FAILS else 0)
