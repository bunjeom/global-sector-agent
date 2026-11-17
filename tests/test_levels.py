"""지지선·손절선 (engine/strategy.py support_stop_levels, engine/level_backtest.py) 손 계산 검사 + 검증 방식 자체의 점검."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _offline  # noqa: F401
import sys
import numpy as np
import pandas as pd
import engine

FAILS = []


def chk(n, c, extra=None):
    print(("PASS " if c else "FAIL ") + n + (f"  [{extra}]" if (not c and extra is not None) else ""))
    if not c:
        FAILS.append(n)


close = lambda a, b, tol=1e-9: a is not None and b is not None and abs(float(a) - float(b)) < tol
UP = engine.STATE_UP
LB = engine.LEVEL_BT_CONFIG


def frame(o, lo, c, sup, atr=2.0):
    n = len(c)
    idx = pd.bdate_range("2021-01-04", periods=n)
    return pd.DataFrame({"Open": o, "High": np.maximum(o, c) + 0.5, "Low": lo, "Close": c, "ATR20": [atr] * n,
                         "support": sup, "stop": np.array(sup, float) - 0.5 * atr, "state": [UP] * n}, index=idx)


# 1) 사건 손 계산. 지지선 100, ATR 2 → 손절선 99, 접근 구간 = 저가 ≤ 101
n = 40
c = np.full(n, 110.0)
c[5:8] = [104.0, 102.0, 100.5]          # 7일: 저가 100 ≤ 101, 종가 100.5 ≥ 99 → 사건 (5거래일 변화 = (100.5 − 110) ÷ 2 = −4.75)
c[8:12] = [101.0, 100.0, 98.0, 97.0]    # 10일 종가 98 < 99 → 깨짐
c[12:] = 96.0
lo = c - 0.5
o = np.concatenate([[110.0], c[:-1]])
si = frame(o, lo, c, [100.0] * n)
a = engine.level_arrays(si)
ev = engine.support_events(si, a, start="2021-01-01")
print("  지지선 사건:", ev[["t", "d_atr", "drop", "broke"]].to_dict("records"))
chk("1 지지선 사건: 6일(저가 101.5 → 아님), 7일(저가 100 ≤ 101, 종가 100.5 ≥ 99) 첫 사건", len(ev) >= 1 and int(ev.iloc[0]["t"]) == 7, ev)
chk("1-2 거리 = (100.5 − 99) ÷ 2 = 0.75 ATR, 5일 변화 −4.75 ATR", close(ev.iloc[0]["d_atr"], 0.75) and close(ev.iloc[0]["drop"], -4.75))
chk("1-3 이후 10거래일 종가 최저 96 < 손절선 99 → 깨짐", ev.iloc[0]["broke"] == 1.0)
chk("1-4 사건 뒤 10거래일은 새 사건 없음(다음 사건은 18일 이후)", len(ev) == 1 or int(ev.iloc[1]["t"]) >= 18, ev["t"].tolist())
te = engine.stop_events(si, a, start="2021-01-01")
print("  손절선 사건:", te[["t", "r10"]].to_dict("records"))
chk("1-5 손절선 사건: 10일(종가 98 < 99, 전날 100 ≥ 99)", len(te) == 1 and int(te.iloc[0]["t"]) == 10, te)
chk("1-6 결과 = 11일 시가(= 10일 종가 98) → 20일 종가 96: 96 ÷ 98 − 1", close(te.iloc[0]["r10"], 96 / 98 - 1), te.iloc[0]["r10"])
c2 = c.copy()
c2[8:] = 103.0
si2 = frame(np.concatenate([[110.0], c2[:-1]]), c2 - 0.5, c2, [100.0] * n)
ev2 = engine.support_events(si2, start="2021-01-01")
chk("1-7 버틴 경우: 깨짐 0", int(ev2.iloc[0]["t"]) == 7 and ev2.iloc[0]["broke"] == 0.0)
chk("1-8 버틴 경우 손절선 사건 없음", len(engine.stop_events(si2, start="2021-01-01")) == 0)

# 2) 가짜 선: 같은 ATR 거리, 미래 10일 종가 최저와 비교 — 손 계산
pl = engine.support_placebo(si, ev.iloc[:1], a, matched=False, start="2021-01-01", samples=200, seed=1)
j_ok = [j for j in range(n) if np.isfinite(a["fmin"][j]) and np.isfinite(a["drop"][j]) and abs(j - 7) > 10]
exp = np.mean([a["fmin"][j] < c[j] - 0.75 * 2.0 for j in j_ok])
print("  가짜 선 깨짐 비율:", round(pl["broke"].mean(), 3), "기대", round(float(exp), 3), "후보 날", j_ok)
chk("2 가짜 선(무작위 날): 후보 = 사건과 10일 넘게 떨어진 날, 깨짐 = 이후 10일 종가 최저 < 종가 − 0.75×ATR",
    set(pl["date"]) <= {si.index[j] for j in j_ok} and abs(pl["broke"].mean() - exp) < 0.12, (pl["broke"].mean(), exp))
plm = engine.support_placebo(si, ev.iloc[:1], a, matched=True, start="2021-01-01", samples=50, seed=1)
dr_ok = {si.index[j] for j in j_ok if abs(a["drop"][j] - (-4.75)) <= LB["DROP_TOL"]}
chk("2-2 가짜 선(하락 폭 맞춤): 5일 변화가 −4.75 ± 0.5 ATR인 날에서만(없으면 표본 없음)", (len(plm) == 0 and not dr_ok) or set(plm["date"]) <= dr_ok,
    (len(plm), len(dr_ok)))

# 3) 화면용 판정: 지지선·손절선·거리·매수 비중 한도
legs = [(100, 130), (130, 112), (112, 145), (145, 125), (125, 160), (160, 150)]
path = np.concatenate([np.linspace(x, y, 15, endpoint=False) for x, y in legs] + [np.array([150.0, 149.0, 148.0])])
idx = pd.bdate_range("2022-01-03", periods=len(path))
ind = pd.DataFrame({"Open": path, "High": path * 1.002, "Low": path * 0.998, "Close": path, "ATR20": 2.0, "state": UP}, index=idx)
lv = engine.support_stop_levels(ind.assign(Volume=1.0), ind=ind)
print("  화면 판정:", lv["status"], "| 지지선", round(lv["support"], 2), "| 손절선", round(lv["stop"], 2), "| 거리", round(lv["dist"] * 100, 2),
      "% | 비중 한도", round(lv["max_weight"] * 100, 1), "%")
chk("3 손절선 = 지지선 − 0.5×ATR, 거리 = (종가 148 − 손절선) ÷ 148", close(lv["stop"], lv["support"] - 1.0) and close(lv["dist"], (148 - lv["stop"]) / 148))
chk("3-2 매수 비중 한도 = 1% ÷ 거리", close(lv["max_weight"], 0.01 / lv["dist"]) and lv["risk_pct"] == 0.01)
chk("3-3 상태 '지지선 위'(저가가 지지선 + 0.5×ATR보다 높음)", lv["status"] == "지지선 위" and not lv["near"] and not lv["broken"])
near = ind.copy()
near.iloc[-1, near.columns.get_loc("Low")] = lv["support"] + 0.5
near.iloc[-1, near.columns.get_loc("Close")] = lv["support"] + 3.0
chk("3-4 저가가 지지선 + 0.5×ATR 이하 → '지지선 접근'", engine.support_stop_levels(near, ind=near)["status"] == "지지선 접근")
brk = ind.copy()
brk.iloc[-1, brk.columns.get_loc("Close")] = lv["stop"] - 1.0
b = engine.support_stop_levels(brk, ind=brk)
chk("3-5 종가 < 손절선 → '손절선 이탈', 비중 한도 없음", b["status"] == "손절선 이탈" and b["broken"] and b["max_weight"] is None)
tight = ind.copy()
tight.iloc[-1, tight.columns.get_loc("Close")] = lv["stop"] * 1.004
chk("3-6 거리가 1%보다 짧으면 한도는 100%까지만", engine.support_stop_levels(tight, ind=tight)["max_weight"] == 1.0)
chk("3-7 검증 상태 문구 3개(지지선·손절선·위험 계산)", set(lv["verification"]) == {"support", "stop", "risk"} and "산수" in lv["verification"]["risk"])
chk("3-8 스윙 저점이 없으면 판정 없음", not engine.support_stop_levels(ind.iloc[:2], ind=ind.iloc[:2])["available"])

# 4) 전략 선택: 단기는 진입 후보가 아님(2026-11-17)
cs = engine.choose_strategy
r4 = cs(frame_b_stage="진입 가능", value_top=True, frame_a_entry=True, trend_top=True)
chk("4 카드는 가치 스윙·추세 2장, 둘 다 충족이면 가치", [x["key"] for x in r4["cards"]] == ["value", "trend"] and r4["chosen"] == "value"
    and r4["cards"][1]["role"] == "보조 근거")
chk("4-2 가치·추세 모두 아니면 대기(단기로 넘어가지 않음)", cs(frame_b_stage="대기", value_top=True, frame_a_entry=False, trend_top=True)["chosen"] is None)
chk("4-3 규칙 문구에 단기는 지지·손절선만", "지지·손절선만" in r4["rule"])


# 5) 검증 방식 점검 — 가짜 가격
def mk(c_, r_, n_):
    o_ = np.concatenate([[c_[0]], c_[:-1]])
    hi = np.maximum(o_, c_) * (1 + np.abs(r_.normal(0, 0.006, n_)))
    lo_ = np.minimum(o_, c_) * (1 - np.abs(r_.normal(0, 0.006, n_)))
    return pd.DataFrame({"Open": o_, "High": hi, "Low": lo_, "Close": c_, "Volume": 1e6}, index=pd.bdate_range("2000-01-03", periods=n_))


def walk(n_, seed, floor_p=0.0, cascade=0.0, look=40):
    """floor_p: 종가가 최근 40일 최저 종가 아래로 내려가려 할 때 튕길 확률(지지 효과).
    cascade: 최근 40일 최저 종가를 깬 뒤 10일 동안 하루 수익률에 더하는 값(이탈 뒤 추가 하락 효과)."""
    r_ = np.random.default_rng(seed)
    c_ = np.empty(n_)
    c_[0] = 50.0
    left = 0
    for i in range(1, n_):
        ret = 0.0003 + r_.normal(0, 0.018) + (cascade if left > 0 else 0.0)
        left = max(0, left - 1)
        nxt = c_[i - 1] * np.exp(ret)
        if i > look:
            fl = c_[i - look:i - 1].min()
            if nxt < fl and c_[i - 1] >= fl:
                if r_.random() < floor_p:
                    nxt = c_[i - 1] * np.exp(abs(ret))
                elif cascade:
                    left = 10
        c_[i] = nxt
    return mk(c_, r_, n_)


def run(**kw):
    out = engine.evaluate_levels({f"D{i}": walk(5200, 100 + i, **kw) for i in range(6)},
                                 {f"H{i}": walk(5200, 200 + i, **kw) for i in range(24)} | {"005930.KS": walk(5200, 999, **kw)})["summary"]
    return out, out["universes"]["HOLDOUT(새 종목)"]["blocks"]["전체 기간"]


f3 = lambda x: (x["actual"].get("n"), round(x["actual"]["mean"] * 100, 2), round(x["baseline"]["mean"] * 100, 2), None if x["z"] is None else round(x["z"], 2))
o0, b0 = run()
print("  효과 없는 가격: 지지선(사건·실제·가짜·z)", f3(b0["support"]["vs_matched"]), "| 손절선", f3(b0["stop"]["vs_random"]))
chk("5 효과 없는 가격(무작위 걸음): 지지선·손절선 모두 미통과, |z| < 3", not o0["verdict"]["support"]["passed"] and not o0["verdict"]["stop"]["passed"]
    and abs(b0["support"]["vs_matched"]["z"]) < 3 and abs(b0["stop"]["vs_random"]["z"]) < 3)
chk("5-2 화면 문구는 '검증 안 됨'", o0["verdict"]["support"]["text"].startswith("검증 안 됨") and o0["verdict"]["stop"]["text"].startswith("검증 안 됨"))
o1, b1 = run(floor_p=0.7)
print("  바닥에서 튕기는 가격: 지지선", f3(b1["support"]["vs_matched"]))
chk("5-3 바닥에서 튕기는 가격: 지지선 통과(가짜 선보다 덜 깨짐, z ≤ −3, 개발 묶음 앞·뒤 모두)", o1["verdict"]["support"]["passed"]
    and o1["verdict"]["support"]["text"].startswith("검증됨"), o1["verdict"]["support"])
o2, b2 = run(cascade=-0.006)
print("  이탈 뒤 더 떨어지는 가격: 손절선", f3(b2["stop"]["vs_random"]))
chk("5-4 이탈 뒤 더 떨어지는 가격: 손절선 통과(이탈 뒤 수익이 더 나쁨, z ≤ −3)", o2["verdict"]["stop"]["passed"], o2["verdict"]["stop"])
chk("5-5 판정 기준 사전 고정(z 3, 사건 100건)", engine.LEVEL_CRITERIA == {"MIN_Z": 3.0, "MIN_EVENTS": 100})
rep = engine.render_levels_report(o1, generated_at="test")
chk("5-6 보고서: 판정 2개·화면 문구·한국 구분", "## 판정 1 — 지지선: 통과" in rep and "## 판정 2 — 손절선:" in rep and "화면 문구:" in rep
    and "전체 기간 · 한국" in rep and "미국 85·한국 25" in rep)
used = set(engine.FRAME_A_CONFIG["DEV_UNIVERSE"]) | set(engine.FRAME_A_CONFIG["HOLDOUT_UNIVERSE"]) | set(engine.FRAME_B_CONFIG["HOLDOUT_UNIVERSE"]) \
    | set(engine.TBT_CONFIG["HOLDOUT_UNIVERSE"]) | set(engine.SHORT_BT_CONFIG["HOLDOUT_UNIVERSE"]) \
    | {s for g in engine.BACKTEST_CONFIG["SETUP_UNIVERSE"].values() for s in g}
ho = LB["HOLDOUT_UNIVERSE"]
chk("5-7 HOLDOUT 110종목(미국 85·한국 25), 이전 검증 묶음과 겹치지 않음", len(ho) == len(set(ho)) == 110
    and sum(1 for t in ho if t.endswith(".KS")) == 25 and not (set(ho) & used), sorted(set(ho) & used))

# 6) run_backtest.py 명령줄
import subprocess
_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
hp = subprocess.run([sys.executable, "run_backtest.py", "--help"], cwd=_root, capture_output=True, text=True, encoding="utf-8",
                    timeout=120, env={**_os.environ, "PYTHONIOENCODING": "utf-8"})
chk("6 run_backtest.py --help 정상(--levels 포함)", hp.returncode == 0 and "--levels" in hp.stdout, (hp.returncode, hp.stderr[-300:]))

print(f"\n결과: {'통과' if not FAILS else '실패 ' + str(FAILS)}")
sys.exit(1 if FAILS else 0)
