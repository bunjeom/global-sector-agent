"""틀 A(추세반등) 검증(engine/frame_backtest.py) 검사.

1 미래 데이터 미사용: 날짜 t 이후 종목·시장 가격을 바꿔도 t의 지표·신호는 그대로, 한국 종목은 KOSPI와 비교
2 거래 손 계산: 다음 날 시가 진입, 직전 저점·200일선 무효화, 50일선 확인 뒤 이탈 매도, 다음 날 시가 청산,
  진입 전 무효화 건너뜀, 미청산, 보유 중 갭 하락 세기
3 종목당 1포지션(청산일부터 새 신호), 보유 기간 맞춘 무작위(같은 보유일·다음 날 시가 실행), '선정만'은 선정일에서만
4 월 묶음 표준오차 손 계산, 판정 규칙, 보고서, 우위 없는 가짜 가격에서는 통과하지 않음
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # 프로젝트 폴더의 engine 패키지를 불러오기 위해
import _offline  # noqa: F401  외부 네트워크 차단 (tests/_offline.py)
import math
import sys
import numpy as np
import pandas as pd
import engine

FAILS = []
C = engine.TECH_PARAMS["ROUND_TRIP_COST"]


def chk(n, c):
    print(("PASS " if c else "FAIL ") + n, flush=True)
    (None if c else FAILS.append(n))


def mk(idx, s, d=4e-4, v=0.016, seed=0):
    r = np.random.default_rng(seed)   # 테스트 입력용 (앱 코드 아님)
    c = s * np.exp(np.cumsum(r.normal(d, v, len(idx))))
    o = c * (1 + r.normal(0, .004, len(idx)))
    return pd.DataFrame({"Open": o, "High": np.maximum(o, c) * 1.004, "Low": np.minimum(o, c) * 0.996, "Close": c,
                         "Volume": 1e6}, index=idx)


idx = pd.bdate_range("2000-01-03", periods=1600)
stock, spy, ks = mk(idx, 50, seed=1), mk(idx, 100, 2e-4, .01, seed=2), mk(idx, 1000, 1e-4, .012, seed=3)

# ---------------------------------------------------------------------------
# 1 미래 데이터 미사용
# ---------------------------------------------------------------------------
ind = engine.frame_a_indicators(stock, spy)
t = 900
s2, b2 = stock.copy(), spy.copy()
s2.iloc[t + 1:, :4] *= 3.0
b2.iloc[t + 1:, :4] *= 0.2
ind2 = engine.frame_a_indicators(s2, b2)
cols = ["SMA200", "SMA50", "RSI2", "LOWREF", "RS126", "RS252", "selected", "signal"]
chk("1-1 t 이후 가격을 바꿔도 t까지의 지표·신호 동일",
    all(np.allclose(ind[c].iloc[:t + 1].astype(float), ind2[c].iloc[:t + 1].astype(float), equal_nan=True) for c in cols))
c, bc = stock["Close"], spy["Close"]
rs = (c.iloc[t] / c.iloc[t - 252] - 1) - (bc.iloc[t] / bc.iloc[t - 252] - 1)
chk("1-2 12개월 상대강도 손 계산", abs(ind["RS252"].iloc[t] - rs) < 1e-12)
chk("1-3 60일 최저가(신호일 포함)", abs(ind["LOWREF"].iloc[t] - stock["Low"].iloc[t - 59:t + 1].min()) < 1e-12)
chk("1-4 신호 = 선정 + RSI(2) < 10", bool((ind["signal"] == (ind["selected"] & (ind["RSI2"] < 10))).all()))
# 시장 날짜가 하루 빠져도 앞 값으로 채움 (그날 이후 값은 쓰지 않음)
spy_gap = spy.drop(idx[t])
ind3 = engine.frame_a_indicators(stock, spy_gap)
exp = (c.iloc[t] / c.iloc[t - 126] - 1) - (bc.iloc[t - 1] / bc.iloc[t - 126] - 1)
chk("1-5 시장 휴장일은 전날 값", abs(ind3["RS126"].iloc[t] - exp) < 1e-12)
chk("1-6 한국 종목 판정", engine._fa_market("005930.KS") == "KR" and engine._fa_market("AAPL") == "US")


# ---------------------------------------------------------------------------
# 2 거래 손 계산 (지표를 직접 넣은 작은 표)
# ---------------------------------------------------------------------------
def fake(close, open_=None, sma200=None, sma50=None, low=None):
    n = len(close)
    d = pd.DataFrame({"Open": open_ if open_ is not None else close, "Close": close,
                      "SMA200": sma200 if sma200 is not None else [1.0] * n,
                      "SMA50": sma50 if sma50 is not None else [1.0] * n,
                      "LOWREF": low if low is not None else [90.0] * n},
                     index=pd.bdate_range("2020-01-01", periods=n), dtype=float)
    return d, engine._fa_arrays(d)


# 0일 신호 → 1일 시가 100 진입. 1~2일 50일선 아래, 3일 위(반등 확인), 5일 아래(추세 전환) → 6일 시가 108 청산
cl = [100, 99, 101, 106, 110, 104, 107, 109]
op = [100, 100, 99, 102, 107, 109, 108, 108]
s50 = [105, 105, 105, 105, 105, 105, 105, 105]
d, a = fake(cl, op, sma50=s50)
r = engine.simulate_frame_a_trade(a, 0, cost=C)
chk("2-1 50일선 확인 뒤 이탈 → 다음 날 시가 청산",
    r["exit_kind"] == "trend" and r["exit_idx"] == 6 and abs(r["exit_px"] - 108) < 1e-12 and r["hold_days"] == 5)
chk("2-2 순수익률·R 손 계산", abs(r["ret"] - (108 / 100 - 1 - C)) < 1e-12
    and abs(r["r"] - (108 - 100 - 100 * C) / (100 - 90)) < 1e-12 and r["armed"])
chk("2-3 청산 갭 = 청산 시가 / 조건일 종가 − 1", abs(r["exit_gap"] - (108 / 104 - 1)) < 1e-12)

# 반등 확인 전에는 50일선 아래여도 팔지 않음, 직전 저점(90) 종가 이탈 → 다음 날 시가(갭 하락) 청산
cl = [100, 98, 96, 95, 89, 85, 86]
op = [100, 99, 97, 96, 94, 80, 85]
d, a = fake(cl, op, sma50=[105] * 7)
r = engine.simulate_frame_a_trade(a, 0, cost=C)
chk("2-4 직전 저점 이탈 무효화 (반등 전 50일선 아래는 무시)", r["exit_kind"] == "inv_low" and r["exit_idx"] == 5 and r["exit_px"] == 80)
chk("2-5 보유 중 −10% 이상 갭 하락 1회", r["gap_days"] == 1 and abs(r["exit_gap"] - (80 / 89 - 1)) < 1e-12)

# 200일선 이탈 무효화
cl = [100, 101, 99, 103]
d, a = fake(cl, [100, 100, 100, 98], sma200=[95, 95, 100, 95], sma50=[105] * 4)
r = engine.simulate_frame_a_trade(a, 0, cost=C)
chk("2-6 200일선 종가 이탈 무효화", r["exit_kind"] == "inv_ma" and r["exit_idx"] == 3 and r["exit_px"] == 98)

# 같은 날 직전 저점과 50일선을 함께 깨면 무효화로 분류
cl = [100, 106, 89, 92]
d, a = fake(cl, [100, 100, 100, 91], sma50=[105] * 4)
r = engine.simulate_frame_a_trade(a, 0, cost=C)
chk("2-7 같은 날 겹치면 무효화 우선", r["exit_kind"] == "inv_low" and r["exit_idx"] == 3)

# 진입 시가가 이미 무효화 가격 이하 → 건너뜀
d, a = fake([100, 95, 96], [100, 89, 96])
chk("2-8 진입 전 무효화는 건너뜀", engine.simulate_frame_a_trade(a, 0, cost=C) == {"skipped": True})

# 끝까지 조건 없음 → 마지막 종가 평가, 마지막 날 조건 → 다음 날 시가를 모르므로 미청산
d, a = fake([100, 101, 102, 103], [100, 100, 101, 102], sma50=[99] * 4)
r = engine.simulate_frame_a_trade(a, 0, cost=C)
chk("2-9 미청산은 마지막 종가", r["exit_kind"] == "open" and r["exit_px"] == 103 and r["exit_gap"] is None)
d, a = fake([100, 101, 102, 98], [100, 100, 101, 102], sma50=[99] * 4)
r = engine.simulate_frame_a_trade(a, 0, cost=C)
chk("2-10 마지막 날 조건도 미청산", r["exit_kind"] == "open" and r["exit_px"] == 98)
chk("2-11 데이터 끝 신호는 None", engine.simulate_frame_a_trade(a, 3, cost=C) is None)

# ---------------------------------------------------------------------------
# 3 종목당 1포지션, 비교 표본
# ---------------------------------------------------------------------------
cl = [100, 99, 101, 106, 110, 104, 107, 109, 108, 111]
op = [100, 100, 99, 102, 107, 109, 108, 108, 108, 110]
d, a = fake(cl, op, sma50=[105] * 10)
d["signal"] = [True, False, True, False, False, False, True, False, False, False]   # 2일은 보유 중, 6일은 청산일
d["selected"] = [True, True, True, False, False, False, True, True, False, False]
res = engine.frame_a_trades(d, start="2019-01-01", cost=C, arrs=a)
tr = res["trades"]
chk("3-1 보유 중 신호 무시, 청산일 신호는 받음", list(tr["date"]) == [d.index[0], d.index[6]])
tr1 = tr.iloc[:1]   # 보유 5일
bs = engine.frame_a_matched_baseline(d, tr1, start="2019-01-01", samples=6, seed=1, cost=C, arrs=a)
j = [d.index.get_loc(x) for x in bs["date"]]
chk("3-2 보유 기간 맞춘 무작위: 표본 수·같은 보유일",
    len(bs) == 6 and (bs["hold_days"] == 5).all() and max(j) + 1 + 5 <= len(d) - 1)
chk("3-2b 다음 날 시가에 사서 보유일 뒤 시가에 판다",
    all(abs(row.gross - (op[k + 6] / op[k + 1] - 1)) < 1e-12 for row, k in zip(bs.itertuples(), j))
    and np.allclose(bs["gross"] - bs["ret"], C))
sel = engine.frame_a_baseline(d, 2, "selected", start="2019-01-01", samples=30, seed=1, cost=C, arrs=a)
chk("3-3 '선정만' 표본은 선정일에서만", set(sel["date"]) <= set(d.index[d["selected"].to_numpy()]))

# ---------------------------------------------------------------------------
# 4 월 묶음 표준오차, 판정, 보고서
# ---------------------------------------------------------------------------
x = pd.Series([1.0, 3.0, 2.0, 6.0])
dt = pd.Series(pd.to_datetime(["2020-01-02", "2020-01-20", "2020-02-03", "2020-03-04"]))
cs = engine.cluster_stats(x, dt)
m = 3.0
sums = [(1 - m) + (3 - m), 2 - m, 6 - m]
exp_se = math.sqrt(3 / 2 * sum(v * v for v in sums)) / 4
chk("4-1 월 묶음 표준오차 손 계산", abs(cs["se"] - exp_se) < 1e-12 and cs["n_months"] == 3
    and abs(cs["se_iid"] - x.std(ddof=1) / 2) < 1e-12)

bench = {"SPY": spy, "^KS11": ks}
dev = {f"D{i}": mk(idx, 50, seed=10 + i) for i in range(4)}
ho = {"H0": mk(idx, 50, seed=40), "H1.KS": mk(idx, 50, seed=41)}
out = engine.evaluate_frame_a(dev, ho, bench)
s = out["summary"]
u = s["universes"]["HOLDOUT(새 종목)"]
chk("4-2 묶음·기간·시장 구분", set(s["universes"]) == {"개발(대표 종목)", "HOLDOUT(새 종목)"}
    and set(u["blocks"]) == {"전체 기간", "앞 기간", "뒤 기간"} and set(u["by_market"]) <= {"US", "KR"})
ho_full = u["blocks"]["전체 기간"]["vs_random"]
dv = s["universes"]["개발(대표 종목)"]["blocks"]
exp_pass = (ho_full["actual"].get("n", 0) >= 2 and ho_full["actual"]["mean"] > 0 and ho_full["z"] is not None and ho_full["z"] >= 2
            and all(dv[p]["vs_random"]["actual"].get("n", 0) >= 2 and dv[p]["vs_random"]["actual"]["mean"] > 0 for p in ("앞 기간", "뒤 기간")))
chk("4-3 판정 = 사전 기준", s["verdict"]["passed"] == exp_pass)
tr = out["trades"]
chk("4-4 거래 표: 진입일이 시작일 이후, 비용 차감", len(tr) > 0 and (tr["date"] >= pd.Timestamp(engine.FRAME_A_CONFIG["START"])).all()
    and np.allclose(tr["gross"] - tr["ret"], C))
rep = engine.render_frame_a_report(s, "test")
chk("4-5 보고서에 판정·기준·주의", "## 판정" in rep and "월 묶음 표준오차" in rep and "살아남은 종목 편향" in rep)
# 우위가 없도록 만든 가짜 가격(기하 브라운 운동)에서는 무작위 대비 차이가 표준오차 2배를 넘지 않아야 한다
pm = {f"G{i}": mk(pd.bdate_range("1999-01-01", "2016-12-30"), 50, d=3e-4, v=.02, seed=200 + i) for i in range(25)}
uu = engine.run_frame_a_universe(pm, {"SPY": mk(pd.bdate_range("1999-01-01", "2016-12-30"), 100, 2e-4, .01, seed=9)})
zz = engine._fa_block(uu["trades"], uu["random"], uu["selected"])["vs_random"]["z"]
chk(f"4-7 우위 없는 가짜 가격에서 통과하지 않음 (z {zz:+.2f})", zz is not None and zz < 2.0)
chk("4-8 무작위 표본 보유일 = 실제 거래 보유일", abs(uu["random"]["hold_days"].mean() - uu["trades"]["hold_days"].mean()) < 1e-9)
cfg = engine.FRAME_A_CONFIG
chk("4-6 HOLDOUT은 개발 묶음·이전 백테스트 종목과 겹치지 않음",
    not set(cfg["HOLDOUT_UNIVERSE"]) & (set(cfg["DEV_UNIVERSE"]) | set(engine.BACKTEST_CONFIG["SETUP_UNIVERSE"]["ETF"])
                                        | set(engine.TBT_CONFIG["HOLDOUT_UNIVERSE"])))

print(f"\n{'모두 통과' if not FAILS else f'실패 {len(FAILS)}개: ' + ', '.join(FAILS)}")
sys.exit(1 if FAILS else 0)
