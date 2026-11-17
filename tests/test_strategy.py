"""전략 선택 4단계 (engine/strategy.py·short_backtest.py) 손 계산 검사 — 단기 규칙 진입·청산, 무작위 비교, 전략 우선순위."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _offline  # noqa: F401
import math, sys
import numpy as np
import pandas as pd
import engine

FAILS = []


def chk(n, c, extra=None):
    print(("PASS " if c else "FAIL ") + n + (f"  [{extra}]" if (not c and extra is not None) else ""))
    if not c:
        FAILS.append(n)


close = lambda a, b, tol=1e-9: a is not None and b is not None and abs(float(a) - float(b)) < tol
C = engine.SHORT_TERM_CONFIG
UP, SIDE, DOWN = engine.STATE_UP, engine.STATE_SIDE, engine.STATE_DOWN

# 1) RSI(2)는 T-B·틀 A 검증과 같은 식
rng = np.random.default_rng(3)
idx = pd.bdate_range("2020-01-01", periods=400)
px = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0004, 0.015, len(idx)))), index=idx)
ohlc = pd.DataFrame({"Open": px.shift().fillna(px.iloc[0]), "High": px * 1.01, "Low": px * 0.99, "Close": px, "Volume": 1e6})
fa = engine.frame_a_indicators(ohlc, None)
chk("1 RSI(2) = 틀 A 검증(frame_a_indicators)과 같은 값", np.allclose(engine.short_rsi2(px).to_numpy(), fa["RSI2"].to_numpy(), equal_nan=True))


# 2) 청산 규칙 손 계산 (배열 직접)
def arrs(o, h, c, stop, down=None):
    n = len(o)
    return {"Open": np.array(o, float), "High": np.array(h, float), "Low": np.array(c, float) * 0.99, "Close": np.array(c, float),
            "stop": np.array(stop, float), "down": np.zeros(n, bool) if down is None else np.array(down, bool),
            "signal": np.zeros(n, bool), "pool_nopull": np.ones(n, bool)}


S = engine.simulate_short_trade
# 신호일 0, 진입 1일 시가 100, 손절선 98(거리 2%) — 1일 장중 고가 103.5 → 103에 익절
t = S(arrs([99, 100, 101], [99, 103.5, 101], [99, 101, 101], [98, 98, 98]), 0)
chk("2 진입일 장중 +3% → 103 익절, 순수익률 3% − 0.3% = 2.7%, 보유 0일, 종가 대응", t["exit_kind"] == "target" and close(t["exit_px"], 103)
    and close(t["ret"], 0.027) and t["hold_days"] == 0 and t["exit_at"] == "close", t)
chk("2-1 R = (103 − 100 − 0.3) ÷ (100 − 98) = 1.35", close(t["r"], 1.35), t["r"])
t = S(arrs([99, 100, 104, 104], [99, 101, 105, 105], [99, 100.5, 104, 104], [98] * 4), 0)
chk("2-2 둘째 날 시가 104가 이미 목표(103) 위 → 시가 104에 익절", t["exit_kind"] == "target" and close(t["exit_px"], 104)
    and t["exit_at"] == "open" and t["hold_days"] == 1, t)
t = S(arrs([99, 100, 99, 96, 95], [99, 101, 99.5, 97, 96], [99, 100, 97.5, 96, 95], [98] * 5), 0)
chk("2-3 2일 종가 97.5 < 손절선 98 → 3일 시가 96에 청산", t["exit_kind"] == "stop" and close(t["exit_px"], 96) and t["exit_idx"] == 3
    and t["hold_days"] == 2, t)
t = S(arrs([99, 100, 100, 99], [99, 101, 101, 100], [99, 100, 100, 99], [98] * 4, down=[0, 0, 1, 0]), 0)
chk("2-4 2일 추세 '하락' → 3일 시가에 청산", t["exit_kind"] == "trend_down" and t["exit_idx"] == 3, t)
n = 14
t = S(arrs([99] + [100] * (n - 1), [99] + [101] * (n - 1), [99] + [100] * (n - 1), [98] * n), 0)
chk("2-5 10거래일(진입일 포함) 동안 +3% 미도달 → 11번째 날 시가, 보유 10일", t["exit_kind"] == "time" and t["exit_idx"] == 11
    and t["hold_days"] == 10, t)
chk("2-6 손절선까지 5% > 3% → 진입 안 함", S(arrs([99, 100, 100], [99, 101, 101], [99, 100, 100], [95, 95, 95]), 0) == {"skipped": "far"})
chk("2-7 시가가 손절선 이하 → 진입 안 함", S(arrs([99, 97, 100], [99, 101, 101], [99, 100, 100], [98, 98, 98]), 0) == {"skipped": "below_stop"})
t = S(arrs([99, 100, 100], [99, 101, 101], [99, 100, 100.5], [98] * 3), 0)
chk("2-8 자료 끝까지 청산 없음 → 미청산(마지막 종가)", t["exit_kind"] == "open" and close(t["exit_px"], 100.5), t)

# 3) 종목당 1포지션 + 보유 기간 맞춘 무작위
idx3 = pd.bdate_range("2021-01-04", periods=30)
o = [100.0] * 30
si = pd.DataFrame({"Open": o, "High": [101.0] * 30, "Low": [99.0] * 30, "Close": [100.0] * 30, "stop": [98.0] * 30,
                   "state": [UP] * 30, "short_signal": [False] * 30}, index=idx3)
si.loc[idx3[[2, 4, 15]], "short_signal"] = True      # 2일 신호 → 3일 진입 → 10거래일 → 13일 시가 청산 → 4일 신호는 무시, 15일 신호는 받음
r = engine.short_trades(si, start="2021-01-01")
tr = r["trades"]
chk("3 종목당 1포지션: 2일·15일 신호만 거래(4일 신호는 보유 중이라 무시), 둘 다 10거래일 시간 청산", len(tr) == 2
    and list(tr["date"]) == [idx3[2], idx3[15]] and list(tr["exit_kind"]) == ["time", "time"] and list(tr["exit_idx"]) == [13, 26], tr)
rb = engine.short_matched_baseline(si, tr, start="2021-01-01", samples=5)
chk("3-2 무작위: 거래당 5표본, 보유일 같음, 평평한 가격이면 순수익률 −0.3%", len(rb) == 10 and set(rb["hold_days"]) == {10}
    and np.allclose(rb["ret"], -0.003), rb.head())
tr_c = pd.DataFrame({"hold_days": [0], "exit_at": ["close"]})
si2 = si.assign(Close=[101.0] * 30)
rb2 = engine.short_matched_baseline(si2, tr_c, start="2021-01-01", samples=3)
chk("3-3 장중 익절 거래(보유 0일)의 무작위는 그날 종가에 팖: 101 ÷ 100 − 1 = 1%", len(rb2) == 3 and np.allclose(rb2["gross"], 0.01), rb2)

# 4) 지지선 = 그날까지 확정된 가장 최근 스윙 저점 (지그재그 3×ATR)
legs = [(100, 130), (130, 112), (112, 145), (145, 125), (125, 160), (160, 150)]
path = np.concatenate([np.linspace(a, b, 15, endpoint=False) for a, b in legs] + [np.array([150.0, 149.0, 148.0])])
idx4 = pd.bdate_range("2022-01-03", periods=len(path))
ind = pd.DataFrame({"Open": path, "High": path * 1.002, "Low": path * 0.998, "Close": path, "ATR20": 2.0, "state": UP}, index=idx4)
s4 = engine.short_indicators(None, ind=ind)
last = s4.iloc[-1]
lo_conf = [p for p in engine.swing_points(ind["High"].to_numpy(), ind["Low"].to_numpy(), ind["ATR20"].to_numpy()) if p["kind"] == "L"]
chk("4 지지선 = 마지막으로 확정된 스윙 저점(125 근처, 아직 확정 안 된 150 저점은 아님)", lo_conf and close(last["support"], lo_conf[-1]["price"])
    and 120 < last["support"] < 130, (last["support"], lo_conf[-1] if lo_conf else None))
chk("4-2 손절선 = 지지선 − 0.5×ATR(2) = 지지선 − 1", close(last["stop"], last["support"] - 1.0))

# 5) 오늘 판정: 상승 추세 + 3일 연속 하락(RSI 낮음) + 지지선 → 신호, 내일 시가 허용 범위
ev = engine.short_term_evaluate(ind.assign(Volume=1.0), ind=ind)
print("  오늘 판정:", ev["label"], "| RSI", None if ev["rsi2"] is None else round(ev["rsi2"], 1), "| 손절선", round(ev["stop"], 2),
      "| 시가 상한", round(ev["open_max"], 2))
chk("5 신호: 상승 + RSI(2) < 10 + 지지선", ev["signal"] and ev["rsi2"] < 10 and "단기 진입 신호" in ev["label"], ev["label"])
chk("5-2 내일 시가 허용: 손절선 < 시가 ≤ 손절선 ÷ 0.97", close(ev["open_max"], ev["stop"] / 0.97) and ev["plan"]["open_max"] == ev["open_max"])
chk("5-3 오늘 종가 기준 거리 = (148 − 손절선) ÷ 148, 3% 넘으면 미충족 표시", close(ev["dist_at_close"], (148 - ev["stop"]) / 148)
    and ev["dist_at_close_ok"] == (ev["dist_at_close"] <= 0.03))
evd = engine.short_term_evaluate(ind, ind=ind.assign(state=DOWN))
chk("5-4 하락 추세면 해당 없음", not evd["signal"] and "하락" in evd["label"])
chk("5-5 버전·검증 상태 문구(사전 등록 시험 미통과 → 참고만)", ev["version"] == C["VERSION"] and "검증 미통과" in ev["verification"]
    and "참고만" in ev["verification"])

# 6) 전략 우선순위
cs = engine.choose_strategy
a = cs(frame_b_stage="진입 가능", value_top=True, frame_a_entry=True, trend_top=True)
chk("6 둘 다 충족 → 가치(보유 기간이 긴 쪽), 추세는 보조 근거 (단기는 2026-11-17부터 진입 후보 아님)", a["chosen"] == "value"
    and a["met"] == ["value", "trend"] and [c_["role"] for c_ in a["cards"]] == ["진입 전략", "보조 근거"], a["headline"])
b = cs(frame_b_stage="조건부 진입", value_top=None, frame_a_entry=True, trend_top=True)
chk("6-2 가치 관점 순위 없음 = 확인 불가 → 충족 아님 → 추세", b["chosen"] == "trend" and "피어 순위 없음" in b["cards"][0]["conditions"][1]["text"]
    and b["cards"][0]["unknown"], b["headline"])
c2 = cs(frame_b_stage="대기", value_top=True, frame_a_entry=False, trend_top=True)
chk("6-3 가치·추세 아님 → 대기 (단기 신호로 넘어가지 않음)", c2["chosen"] is None and "대기" in c2["headline"])
d = cs(frame_b_stage="대기", value_top=False, frame_a_entry=False, trend_top=False)
chk("6-4 아무것도 없음 → 대기", d["chosen"] is None and "대기" in d["headline"])
e = cs(survival_kill=True, frame_b_stage="진입 가능", value_top=True)
chk("6-5 생존성 하드 킬 → 신규 진입 없음(조건 충족 여부는 보여 줌)", e["block"]["kind"] == "survival" and e["chosen"] == "value"
    and "신규 진입 없음" in e["headline"])
g = cs(gate_level=2, frame_a_entry=True, trend_top=True)
chk("6-6 국면 G2(크기 0) → 의견만", g["block"]["kind"] == "gate" and g["gate_mult"] == 0 and "의견만" in g["headline"])
g1 = cs(gate_level=1, frame_a_entry=True, trend_top=True)
chk("6-7 국면 G1(절반) → 차단 없음", g1["block"] is None and g1["gate_mult"] == 0.5 and g1["chosen"] == "trend")
chk("6-8 '경고' 단계는 가치 진입 아님", cs(frame_b_stage="경고", value_top=True)["cards"][0]["met"] is False)

# 7) 검증 실행 전체 경로(가짜 가격) — 판정 기준·보고서
rng = np.random.default_rng(5)


def walk(n, drift, seed):
    r_ = np.random.default_rng(seed)
    c_ = 50 * np.exp(np.cumsum(r_.normal(drift, 0.02, n)))
    o_ = np.concatenate([[c_[0]], c_[:-1]]) * np.exp(r_.normal(0, 0.004, n))
    hi = np.maximum(o_, c_) * (1 + np.abs(r_.normal(0, 0.01, n)))
    lo = np.minimum(o_, c_) * (1 - np.abs(r_.normal(0, 0.01, n)))
    return pd.DataFrame({"Open": o_, "High": hi, "Low": lo, "Close": c_, "Volume": 1e6}, index=pd.bdate_range("2000-01-03", periods=n))


dev = {f"D{i}": walk(2200, 0.0003, i) for i in range(3)}
hol = {f"H{i}": walk(2200, 0.0003, 100 + i) for i in range(3)} | {"005930.KS": walk(2200, 0.0, 999)}
out = engine.evaluate_short_term(dev, hol)
sm = out["summary"]
hb = sm["universes"]["HOLDOUT(새 종목)"]["blocks"]["전체 기간"]
print("  가짜 가격 검증:", {k: sm["verdict"][k] for k in ("passed", "holdout_pass", "enough_trades")}, "HOLDOUT 거래",
      hb["vs_random"]["actual"].get("n"), "z", hb["vs_random"].get("z"))
chk("7 검증 실행: 두 묶음 모두 거래·무작위·눌림 없이 표본", all(u["blocks"]["전체 기간"]["vs_random"]["actual"].get("n", 0) > 0
    and u["blocks"]["전체 기간"]["vs_nopull"]["baseline"].get("n", 0) > 0 for u in sm["universes"].values()))
chk("7-2 판정 기준 z ≥ 3·거래 30건 이상(사전 고정)", engine.SHORT_TERM_CRITERIA["MIN_Z"] == 3.0 and "× 3" in sm["verdict"]["criteria"]
    and engine.SHORT_TERM_CRITERIA["MIN_TRADES"] == 30)
chk("7-3 청산 이유 합 100%, 비용 시나리오 3개", abs(sum(e_["share"] for e_ in hb["exits"].values()) - 100) < 1e-6 and len(hb["cost"]) == 3)
chk("7-4 한국 종목 시장 구분", "KR" in sm["universes"]["HOLDOUT(새 종목)"]["by_market"])
tt = out["trades"]
chk("7-5 실제 거래: 진입 거리 ≤ 3%, 시가 > 손절선, 보유 ≤ 10일(미청산 제외)", len(tt) > 0 and (tt["stop_dist"] <= 0.03 + 1e-12).all()
    and (tt["fill"] > tt["stop"]).all() and (tt.loc[tt["exit_kind"] != "open", "hold_days"] <= 10).all())
chk("7-6 익절 거래의 총수익률 ≥ 3%(시가 갭이면 그 이상)", (tt.loc[tt["exit_kind"] == "target", "gross"] >= 0.03 - 1e-12).all())
rep = engine.render_short_term_report(sm, generated_at="test")
chk("7-7 보고서: 판정·규칙·HOLDOUT 설명", "## 판정:" in rep and "RSI(2) < 10" in rep and "미국 59·한국 25" in rep and "일곱 번째" in rep)
chk("7-8 HOLDOUT 84종목, 이전 검증 묶음과 겹치지 않음",
    len(engine.SHORT_BT_CONFIG["HOLDOUT_UNIVERSE"]) == 84
    and not (set(engine.SHORT_BT_CONFIG["HOLDOUT_UNIVERSE"]) & (set(engine.FRAME_A_CONFIG["HOLDOUT_UNIVERSE"])
             | set(engine.FRAME_B_CONFIG["HOLDOUT_UNIVERSE"]) | set(engine.FRAME_A_CONFIG["DEV_UNIVERSE"])
             | set(engine.TBT_CONFIG["HOLDOUT_UNIVERSE"]) | {s for g_ in engine.BACKTEST_CONFIG["SETUP_UNIVERSE"].values() for s in g_})))



# 8) 우위 없는 가짜 가격(무작위 걸음)에서는 통과하지 않고, 눌림 뒤 되돌림이 있는 가격에서는 무작위보다 나아야 함
#    (틀 A 검증의 교훈: 비교 방식이 치우치면 우위 없는 가격에서도 통과로 나온다 — 보유 기간 맞춘 무작위가 이를 막는지)
def ar_walk(n, phi, seed, drift=0.0003):
    r_ = np.random.default_rng(seed)
    e = r_.normal(0, 0.018, n)
    ret = np.zeros(n)
    for i in range(1, n):
        ret[i] = drift + phi * (ret[i - 1] - drift) + e[i]
    c_ = 50 * np.exp(np.cumsum(ret))
    o_ = np.concatenate([[c_[0]], c_[:-1]])
    hi = np.maximum(o_, c_) * (1 + np.abs(r_.normal(0, 0.006, n)))
    lo = np.minimum(o_, c_) * (1 - np.abs(r_.normal(0, 0.006, n)))
    return pd.DataFrame({"Open": o_, "High": hi, "Low": lo, "Close": c_, "Volume": 1e6}, index=pd.bdate_range("2000-01-03", periods=n))


null = engine.evaluate_short_term({f"N{i}": ar_walk(3000, 0.0, 300 + i) for i in range(4)},
                                  {f"M{i}": ar_walk(3000, 0.0, 400 + i) for i in range(24)})["summary"]
nz = null["universes"]["HOLDOUT(새 종목)"]["blocks"]["전체 기간"]["vs_random"]
print("  우위 없는 가격: 거래", nz["actual"].get("n"), "차이", round((nz["diff"] or 0) * 100, 3), "%p, z", None if nz["z"] is None else round(nz["z"], 2))
chk("8 우위 없는 가격(무작위 걸음 24종목): 판정 미통과, |z| < 3", not null["verdict"]["holdout_pass"] and nz["z"] is not None and abs(nz["z"]) < 3, nz.get("z"))
mr = engine.evaluate_short_term({f"N{i}": ar_walk(3000, -0.5, 500 + i) for i in range(4)},
                                {f"M{i}": ar_walk(3000, -0.5, 600 + i) for i in range(24)})["summary"]
mz = mr["universes"]["HOLDOUT(새 종목)"]["blocks"]["전체 기간"]["vs_random"]
print("  되돌림 있는 가격: 거래", mz["actual"].get("n"), "차이", round((mz["diff"] or 0) * 100, 3), "%p, z", None if mz["z"] is None else round(mz["z"], 2))
chk("8-2 눌림 뒤 되돌림이 있는 가격(전날 수익률의 −0.5배): 무작위보다 나음(차이 > 0, z > 2)", mz["diff"] is not None and mz["diff"] > 0
    and mz["z"] is not None and mz["z"] > 2, (mz.get("diff"), mz.get("z")))

# 9) run_backtest.py 명령줄 도움말이 만들어지는지(도움말의 %는 %%로 — Python 3.14는 add_argument 때 검사)
import subprocess
_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
hp = subprocess.run([sys.executable, "run_backtest.py", "--help"], cwd=_root, capture_output=True, text=True, encoding="utf-8",
                    timeout=120, env={**_os.environ, "PYTHONIOENCODING": "utf-8"})
chk("9 run_backtest.py --help 정상(--short-term 포함)", hp.returncode == 0 and "--short-term" in hp.stdout, (hp.returncode, hp.stderr[-300:]))

print(f"\n결과: {'통과' if not FAILS else '실패 ' + str(FAILS)}")
sys.exit(1 if FAILS else 0)
