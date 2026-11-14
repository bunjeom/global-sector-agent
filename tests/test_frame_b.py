"""틀 B(저평가 매수) 검증 — SEC 공시일 기준 EPS(engine/sec_facts.py)와 판정(engine/frame_backtest.py) 검사.

1 SEC 자료 읽기: 10-K·10-Q만, USD/shares만, 받기 실패 시 기본 EPS로 대체, 캐시
2 공시일 기준 TTM: 4분기 = 연간 − 9개월, 연간이 끝나는 분기는 연간 값, 정정 값은 그 제출일부터, 액면분할 단위 맞춤
3 일별 조건: EPS는 제출일 다음 거래일부터, 미래 가격을 바꿔도 과거 밴드·신호 그대로, 밴드 손 계산
4 거래 손 계산: 이익 감소 손절, 배수 주도 고평가 매도, 이익 주도면 재평가로 보유
5 같은 날 다른 종목 비교 손 계산, 짝지은 표준오차
6 우위 없는 가짜 가격에서는 통과하지 않고, 우위가 있는 가짜 가격은 통과, 보고서·묶음 겹침
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # 프로젝트 폴더의 engine 패키지를 불러오기 위해
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))                     # tests/_fake_sec.py
import _offline  # noqa: F401  외부 네트워크 차단 (tests/_offline.py)
import math
import sys
import tempfile
import numpy as np
import pandas as pd
import engine
from _fake_sec import fake_company

FAILS = []
C = engine.TECH_PARAMS["ROUND_TRIP_COST"]


def chk(n, c):
    print(("PASS " if c else "FAIL ") + n, flush=True)
    (None if c else FAILS.append(n))


def row(s, e, v, f, form="10-Q"):
    return {"start": s, "end": e, "val": v, "accn": "x", "fy": 2020, "fp": "Q", "form": form, "filed": f}


# ---------------------------------------------------------------------------
# 1 SEC 자료 읽기
# ---------------------------------------------------------------------------
facts_json = {"units": {"USD/shares": [
    row("2020-01-01", "2020-03-31", 1.0, "2020-05-01"), row("2019-01-01", "2019-03-31", 0.9, "2020-05-01"),
    row("2020-04-01", "2020-06-30", 1.1, "2020-08-03"), row("2020-01-01", "2020-06-30", 2.1, "2020-08-03"),
    row("2019-04-01", "2019-06-30", 0.95, "2020-08-03"),
    row("2020-07-01", "2020-09-30", 1.2, "2020-11-02"), row("2020-01-01", "2020-09-30", 3.3, "2020-11-02"),
    row("2019-07-01", "2019-09-30", 1.0, "2020-11-02"), row("2019-01-01", "2019-09-30", 2.85, "2020-11-02"),
    row("2020-01-01", "2020-12-31", 4.6, "2021-02-19", "10-K"), row("2019-01-01", "2019-12-31", 3.9, "2021-02-19", "10-K"),
    row("2021-01-01", "2021-03-31", 1.5, "2021-05-10"), row("2020-01-01", "2020-03-31", 1.05, "2021-05-10"),
    row("2020-01-01", "2020-03-31", 9.9, "2020-06-01", "8-K"),                      # 10-K·10-Q가 아니면 무시
]}, "otherunit": {}}
facts_json["units"]["USD"] = [row("2020-01-01", "2020-03-31", 77.0, "2020-05-01")]   # 단위가 다르면 무시
f = engine.parse_eps_facts(facts_json)
chk("1-1 10-K·10-Q·USD/shares만 읽음", len(f) == 13 and 9.9 not in f["val"].values and 77.0 not in f["val"].values)
chk("1-2 기간 일수", int(f.loc[(f["val"] == 4.6), "days"].iloc[0]) == 365)

calls = []


def fake_fetch(url):
    calls.append(url)
    if "company_tickers" in url:
        return {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple"}, "1": {"cik_str": 1067983, "ticker": "BRK.B", "title": "B"}}
    if "EarningsPerShareDiluted" in url:
        raise OSError("404")
    return facts_json


with tempfile.TemporaryDirectory() as td:
    cm = engine.load_cik_map("test ua", fetch=fake_fetch, cache_dir=td)
    chk("1-3 티커 → CIK (점은 하이픈으로)", cm.get("AAPL") == 320193 and cm.get("BRK-B") == 1067983)
    r = engine.fetch_eps_concept("AAPL", 320193, "ua", fetch=fake_fetch, cache_dir=td, pause=0)
    chk("1-4 희석 EPS가 없으면 기본 EPS", r["tag"] == "EarningsPerShareBasic" and "CIK0000320193" in calls[-1])
    n_before = len(calls)
    r2 = engine.fetch_eps_concept("AAPL", 320193, "ua", fetch=fake_fetch, cache_dir=td, pause=0)
    chk("1-5 두 번째는 캐시에서 (기본 EPS는 다시 받지 않음)", r2["tag"] == "EarningsPerShareBasic"
        and not any("EarningsPerShareBasic" in u for u in calls[n_before:]))
    bad = engine.fetch_eps_concept("ZZZ", 1, "ua", fetch=lambda u: (_ for _ in ()).throw(OSError("x")), cache_dir=td, pause=0)
    chk("1-6 둘 다 실패하면 tag None과 이유", bad["tag"] is None and "OSError" in bad["error"])

# ---------------------------------------------------------------------------
# 2 공시일 기준 TTM
# ---------------------------------------------------------------------------
ev = engine.pit_eps_events(f).set_index("filed")
chk("2-1 분기가 모자라면 TTM 없음", ev.loc["2020-11-02", "ttm"] != ev.loc["2020-11-02", "ttm"])   # NaN
chk("2-2 연간이 끝나는 분기는 연간 값, 1년 전도 연간", abs(ev.loc["2021-02-19", "ttm"] - 4.6) < 1e-12
    and abs(ev.loc["2021-02-19", "ttm_prev"] - 3.9) < 1e-12)
# 2021-05-10: Q1'21 1.5 + Q4'20(4.6−3.3) + Q3'20 1.2 + Q2'20 1.1 = 5.1
# 1년 전: Q1'20(정정 1.05) + Q4'19(3.9−2.85) + Q3'19 1.0 + Q2'19 0.95 = 4.05
chk("2-3 4분기 = 연간 − 9개월, 최근 4개 분기 합", abs(ev.loc["2021-05-10", "ttm"] - 5.1) < 1e-12)
chk("2-4 정정 값은 그 제출일부터 사용", abs(ev.loc["2021-05-10", "ttm_prev"] - 4.05) < 1e-12)
ev_s = engine.pit_eps_events(f, pd.Series([2.0], index=[pd.Timestamp("2021-03-01")])).set_index("filed")
# 분할(2:1) 전에 제출된 값은 ÷2: 1.5 + (2.3 − 1.65) + 0.6 + 0.55 = 3.3
chk("2-5 액면분할: 분할 전 제출 값은 지금 단위로", abs(ev_s.loc["2021-02-19", "ttm"] - 2.3) < 1e-12
    and abs(ev_s.loc["2021-05-10", "ttm"] - 3.3) < 1e-12)
F = engine.split_factor_fn(pd.Series([2.0, 3.0], index=pd.to_datetime(["2015-06-01", "2020-06-01"])))
chk("2-6 분할 비율 곱 (그날 당일 제외, 이후만)", F(pd.Timestamp("2014-01-01")) == 6.0 and F(pd.Timestamp("2015-06-01")) == 3.0
    and F(pd.Timestamp("2021-01-01")) == 1.0)

# ---------------------------------------------------------------------------
# 3 일별 조건
# ---------------------------------------------------------------------------
idx = pd.bdate_range("2021-01-04", "2021-06-30")
px = pd.DataFrame({"Open": 100.0, "High": 100.0, "Low": 100.0, "Close": 100.0}, index=idx)
dd = engine.frame_b_daily(px, ev.reset_index())
chk("3-1 EPS는 제출일 다음 거래일부터", abs(dd.loc["2021-05-10", "ttm"] - 4.6) < 1e-12 and abs(dd.loc["2021-05-11", "ttm"] - 5.1) < 1e-12
    and np.isnan(dd.loc["2021-02-19", "ttm"]) and abs(dd.loc["2021-02-22", "ttm"] - 4.6) < 1e-12)
chk("3-2 PER = 종가 / TTM", abs(dd.loc["2021-05-11", "PE"] - 100 / 5.1) < 1e-12)

px2, cc = fake_company(7)
ev2 = engine.pit_eps_events(engine.parse_eps_facts(cc))
d1 = engine.frame_b_daily(px2, ev2)
t = d1.index[3000]
px3 = px2.copy()
px3.loc[px3.index > t, ["Open", "High", "Low", "Close"]] *= 2.5
d3 = engine.frame_b_daily(px3, ev2)
cols = ["ttm", "PE", "band_mu", "band_sd", "z", "cheap", "rich", "grow_ok", "signal"]
chk("3-3 t 이후 가격을 바꿔도 t까지 밴드·신호 동일",
    all(np.allclose(d1[c].loc[:t].astype(float), d3[c].loc[:t].astype(float), equal_nan=True) for c in cols))
# 밴드 손 계산: t가 속한 달 이전 월말들(직전 5년) PER의 로그 평균
me = d1.index.to_series().groupby(d1.index.to_period("M")).max()
prev_me = me[me < t].iloc[-1]
w = d1.loc[me[(me > prev_me - pd.DateOffset(years=5)) & (me <= prev_me)].to_numpy(), "PE"]
w = w[w > 0]
chk("3-4 밴드 = 이전 월말까지 5년 로그 PER 평균·σ", abs(d1.loc[t, "band_mu"] - np.log(w).mean()) < 1e-12
    and abs(d1.loc[t, "band_sd"] - np.log(w).std(ddof=0)) < 1e-12)   # 앱 V 트랙과 같이 n으로 나눈 σ
# 날짜 정밀도가 표마다 달라도(가격 초 단위, 공시일 마이크로초 단위) 같은 결과 — 최신 pandas·Python 3.14에서 생긴 오류 재발 방지
px_s = px2.copy(); px_s.index = px_s.index.astype("datetime64[s]")
ev_us = ev2.assign(filed=ev2["filed"].astype("datetime64[us]"))
d_mix = engine.frame_b_daily(px_s, ev_us)
chk("3-6 날짜 정밀도가 달라도 같은 결과", all(np.allclose(d1[c].astype(float).to_numpy(), d_mix[c].astype(float).to_numpy(), equal_nan=True)
                                     for c in cols))
chk("3-5 신호 = 싸다(z ≤ −1) + 이익 안 줄어듦", bool((d1["signal"] == (d1["cheap"] & d1["grow_ok"])).all())
    and bool((d1.loc[d1["cheap"], "z"] <= -1).all()))


# ---------------------------------------------------------------------------
# 4 거래 손 계산
# ---------------------------------------------------------------------------
def fake(close, ttm, rich=None, grow=None, open_=None):
    n = len(close)
    d = pd.DataFrame({"Open": open_ if open_ is not None else close, "Close": close, "ttm": ttm},
                     index=pd.bdate_range("2020-01-01", periods=n), dtype=float)
    d["PE"] = d["Close"] / d["ttm"]
    d["rich"] = rich if rich is not None else [False] * n
    d["grow_ok"] = grow if grow is not None else [True] * n
    return d, engine._fb_arrays(d)


# 이익 감소: 3일 종가 뒤 확인 → 4일 시가 청산
d, a = fake([10, 10, 11, 9, 9, 9], [1] * 6, grow=[True, True, True, False, False, False], open_=[10, 10.5, 11, 9.5, 8.8, 9])
r = engine.simulate_frame_b_trade(a, 0, cost=C)
chk("4-1 이익 감소 손절 → 다음 날 시가", r["exit_kind"] == "stop_eps" and r["exit_idx"] == 4 and r["exit_px"] == 8.8
    and abs(r["ret"] - (8.8 / 10.5 - 1 - C)) < 1e-12 and r["hold_days"] == 3)
chk("4-2 보유 중 최대 낙폭(종가)", abs(r["mdd"] - (9 / 10.5 - 1)) < 1e-12)
# 이익 그대로, 배수만 올라 고평가 → 매도
d, a = fake([10, 10, 13, 14, 14], [1] * 5, rich=[False, False, True, True, True], open_=[10, 10, 12, 13.5, 14])
r = engine.simulate_frame_b_trade(a, 0, cost=C)
chk("4-3 배수 주도 고평가 → 매도", r["exit_kind"] == "sell_rich" and r["exit_idx"] == 3 and r["exit_px"] == 13.5 and not r["rerated"])
# 이익이 배수보다 더 올라 고평가 → 재평가로 보유, 나중에 이익 감소로 손절
d, a = fake([10, 10, 14, 15, 15, 14, 14], [1, 1, 1.6, 1.6, 1.6, 1.6, 1.6], rich=[False, False, True, True, True, True, True],
            grow=[True] * 5 + [False, False])
r = engine.simulate_frame_b_trade(a, 0, cost=C)
chk("4-4 이익 주도면 재평가로 보유", r["exit_kind"] == "stop_eps" and r["exit_idx"] == 6 and r["rerated"])
d, a = fake([10, 11, 12], [1] * 3)
r = engine.simulate_frame_b_trade(a, 0, cost=C)
chk("4-5 끝까지 조건 없으면 마지막 종가", r["exit_kind"] == "open" and r["exit_px"] == 12 and r["exit_gap"] is None)
d, a = fake([10, 11, 12], [-1, -1, -1])
chk("4-6 이익이 0 이하면 진입하지 않음", engine.simulate_frame_b_trade(a, 0, cost=C) is None)

# ---------------------------------------------------------------------------
# 5 같은 날 다른 종목 비교
# ---------------------------------------------------------------------------
di = pd.bdate_range("2020-01-01", periods=6)
O = pd.DataFrame({"A": [10, 10, 11, 12, 13, 14], "B": [20, 20, 22, 24, 26, 30], "C": [5, 5, 5, 5, 5, 4],
                  "D": [8, 8, 8, 8, 8, 10], "E": [1, 1, 1, 1, 1, 2]}, index=di, dtype=float)
CH = pd.DataFrame(False, index=di, columns=O.columns)
CH.loc[di[0], ["B", "C", "D"]] = True
tr = pd.DataFrame({"date": [di[0]], "entry_date": [di[1]], "exit_date": [di[5]], "exit_kind": ["sell_rich"]})
x = engine._fb_peer_returns(tr, "A", O, O, CH, C)
g = np.array([30 / 20, 4 / 5, 10 / 8, 2 / 1]) - 1 - C
gc = g[:3]
chk("5-1 다른 종목 평균(같은 날 사서 같은 날 팜, 같은 비용)", abs(x["peer_ret"].iloc[0] - g.mean()) < 1e-12
    and abs(x["peer_lret"].iloc[0] - np.log1p(g).mean()) < 1e-12 and x["n_peers"].iloc[0] == 4)
chk("5-2 신호일에 싼 다른 종목만(3개 이상)", abs(x["cheap_peer_lret"].iloc[0] - np.log1p(gc).mean()) < 1e-12 and x["n_cheap_peers"].iloc[0] == 3)
CH2 = CH.copy(); CH2.loc[di[0], "D"] = False
x2 = engine._fb_peer_returns(tr, "A", O, O, CH2, C)
chk("5-3 싼 다른 종목이 3개 미만이면 비움", np.isnan(x2["cheap_peer_lret"].iloc[0]))
pt = pd.DataFrame({"date": pd.to_datetime(["2020-01-02", "2020-01-20", "2020-02-03"]), "a": [0.3, 0.1, 0.2], "b": [0.1, 0.1, 0.0]})
ps = engine.paired_stats(pt, "a", "b")
dif = pd.Series([0.2, 0.0, 0.2])
sums = [(0.2 - dif.mean()) + (0.0 - dif.mean()), 0.2 - dif.mean()]
exp_se = math.sqrt(2 / 1 * sum(v * v for v in sums)) / 3
chk("5-4 짝지은 차이의 월 묶음 표준오차", abs(ps["diff"] - dif.mean()) < 1e-12 and abs(ps["z"] - dif.mean() / exp_se) < 1e-9)

# ---------------------------------------------------------------------------
# 6 우위 없는/있는 가짜 가격, 보고서
# ---------------------------------------------------------------------------
def universe(seed0, mr, n=20):
    pm, em = {}, {}
    for i in range(n):
        p, c = fake_company(seed0 + i, mr)
        pm[f"S{i}"], em[f"S{i}"] = p, engine.pit_eps_events(engine.parse_eps_facts(c))
    return pm, em


pm0, em0 = universe(20000, False)
z0 = engine._fb_block(engine.run_frame_b_universe(pm0, em0)["trades"])["vs_peers"]["z"]
chk(f"6-1 우위 없는 가짜 가격(이익과 무관한 가격)에서 통과하지 않음 (z {z0:+.2f})", z0 is not None and z0 < 2.0)
pm1, em1 = universe(21000, True)
z1 = engine._fb_block(engine.run_frame_b_universe(pm1, em1)["trades"])["vs_peers"]["z"]
chk(f"6-2 우위가 있는 가짜 가격(PER 평균 회귀)은 잡아냄 (z {z1:+.2f})", z1 is not None and z1 >= 2.0)
out = engine.evaluate_frame_b(dict(list(pm0.items())[:10]), dict(list(pm1.items())[:10]), {**em0, **em1})
s = out["summary"]
ho = s["universes"]["HOLDOUT(새 미국 종목)"]["blocks"]["전체 기간"]["vs_peers"]
exp_ho = (ho["actual"].get("n", 0) >= engine.FRAME_B_CONFIG["MIN_TRADES"] and ho["actual"]["mean"] > 0 and ho["diff"] > 0
          and ho["z"] is not None and ho["z"] >= 2)
chk("6-3 판정 = 사전 기준", s["verdict"]["holdout_pass"] == exp_ho and s["verdict"]["passed"] == (exp_ho and s["verdict"]["dev_both_positive"]))
rep = engine.render_frame_b_report(s, "test")
chk("6-4 보고서에 판정·비교 기준·주의", "## 판정" in rep and "같은 묶음의 다른 종목" in rep and "상장 폐지" in rep)
tr = out["trades"]
chk("6-5 거래 표: 신호일 이후 진입, 비용 차감, 다른 종목 수", len(tr) > 0 and (tr["entry_date"] > tr["date"]).all()
    and np.allclose(tr["gross"] - tr["ret"], C) and (tr["n_peers"] > 0).all())
cfg = engine.FRAME_B_CONFIG
used_before = (set(engine.FRAME_A_CONFIG["DEV_UNIVERSE"]) | set(engine.FRAME_A_CONFIG["HOLDOUT_UNIVERSE"])
               | set(engine.TBT_CONFIG["HOLDOUT_UNIVERSE"]) | set(engine.BACKTEST_CONFIG["SETUP_UNIVERSE"]["ETF"]))
chk("6-6 HOLDOUT은 이전 모든 백테스트 종목과 겹치지 않음", not set(cfg["HOLDOUT_UNIVERSE"]) & used_before)
chk("6-7 개발 묶음은 미국 대표 종목에서 TSM·BRK-B 제외", "TSM" not in cfg["DEV_UNIVERSE"] and "BRK-B" not in cfg["DEV_UNIVERSE"]
    and all(not t.endswith(".KS") for t in cfg["DEV_UNIVERSE"]))

print(f"\n{'모두 통과' if not FAILS else f'실패 {len(FAILS)}개: ' + ', '.join(FAILS)}")
sys.exit(1 if FAILS else 0)
