"""지지선 이탈 뒤 회복 검증 (engine/reclaim_backtest.py) 손 계산 검사 + 검증 방식 점검."""
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


def frame(c, sup=100.0, atr=2.0):
    c = np.asarray(c, float)
    n = len(c)
    o = np.concatenate([[c[0]], c[:-1]])
    return pd.DataFrame({"Open": o, "High": np.maximum(o, c) + 0.5, "Low": np.minimum(o, c) - 0.5, "Close": c, "ATR20": [atr] * n,
                         "support": [sup] * n, "stop": [sup - 0.5 * atr] * n, "state": [UP] * n},
                        index=pd.bdate_range("2021-01-04", periods=n))


# 1) 이탈 → 회복 손 계산. 지지선 100, ATR 2
c = np.full(50, 105.0)
c[10:14] = [99.0, 97.0, 98.0, 101.0]     # 10일 이탈(전날 105 ≥ 100), 13일 종가 101 > 100 → 회복(3일 걸림), 최저 97 → 깊이 (100 − 97) ÷ 2 = 1.5
c[14:] = np.linspace(101.0, 108.2, 36)
si = frame(c)
a = engine.level_arrays(si)
bk = engine.break_events(si, a, start="2021-01-01")
print("  이탈 사건:", bk[["t", "u", "reclaimed", "days", "depth"]].to_dict("records"))
chk("1 이탈 10일, 회복 13일(3거래일), 깊이 1.5 ATR", len(bk) == 1 and int(bk.iloc[0]["t"]) == 10 and int(bk.iloc[0]["u"]) == 13
    and bk.iloc[0]["reclaimed"] == 1.0 and int(bk.iloc[0]["days"]) == 3 and close(bk.iloc[0]["depth"], 1.5), bk)
rc = engine.reclaim_events(si, bk, a)
chk("1-2 판정 사건 = 회복일, 결과 = 14일 시가(= 13일 종가 101) → 23일 종가", len(rc) == 1 and int(rc.iloc[0]["t"]) == 13
    and close(rc.iloc[0]["r10"], c[23] / 101.0 - 1), rc)
chk("1-3 회복 뒤 10거래일 종가가 지지선 위 유지 → 다시 이탈 0", rc.iloc[0]["rebreak"] == 0.0)
c2 = np.full(50, 105.0)
c2[10:] = 96.0                             # 이탈 뒤 10거래일 동안 못 돌아옴
bk2 = engine.break_events(frame(c2), start="2021-01-01")
chk("1-4 못 돌아온 이탈: reclaimed 0, 깊이 (100 − 96) ÷ 2 = 2.0, 판정 사건 없음", len(bk2) == 1 and bk2.iloc[0]["reclaimed"] == 0.0
    and close(bk2.iloc[0]["depth"], 2.0) and len(engine.reclaim_events(frame(c2), bk2)) == 0, bk2)
c3 = np.full(50, 105.0)
c3[10:22] = 96.0
c3[22:] = 103.0                            # 12거래일 뒤에야 회복 → 10거래일 창 밖이라 '못 돌아옴'
bk3 = engine.break_events(frame(c3), start="2021-01-01")
chk("1-5 11거래일 넘어 돌아오면 회복으로 세지 않음", int(bk3.iloc[0]["t"]) == 10 and bk3.iloc[0]["reclaimed"] == 0.0, bk3)

# 2) 접근 뒤 세 갈래: 버팀 / 깼다가 회복 / 깨고 못 돌아옴
def approach(after):
    x = np.full(40, 110.0)
    x[5] = 100.6                           # 저가 100.1 ≤ 101, 종가 ≥ 99 → 접근 사건
    x[6:6 + len(after)] = after
    x[6 + len(after):] = after[-1]
    return engine.approach_outcomes(frame(x), start="2021-01-01")


k1 = approach([102.0, 103.0])
k2 = approach([99.5, 98.0, 101.0, 102.0])
k3 = approach([99.5, 98.0, 97.0])
print("  세 갈래:", k1["kind"].tolist()[:1], k2["kind"].tolist()[:1], k3["kind"].tolist()[:1])
chk("2 접근 뒤: 버팀 / 깼다가 회복 / 깨고 못 돌아옴", k1["kind"].iloc[0] == "held" and k2["kind"].iloc[0] == "reclaimed" and k3["kind"].iloc[0] == "lost")


# 3) 검증 방식 점검 — 가짜 가격
def mk(c_, r_, n_):
    o_ = np.concatenate([[c_[0]], c_[:-1]])
    hi = np.maximum(o_, c_) * (1 + np.abs(r_.normal(0, 0.006, n_)))
    lo_ = np.minimum(o_, c_) * (1 - np.abs(r_.normal(0, 0.006, n_)))
    return pd.DataFrame({"Open": o_, "High": hi, "Low": lo_, "Close": c_, "Volume": 1e6}, index=pd.bdate_range("2000-01-03", periods=n_))


def walk(n_, seed, spring=0.0, look=40):
    """spring: 최근 40일 최저 종가(바닥)를 깼다가 10일 안에 다시 그 위로 올라오면, 그 뒤 10일 동안 하루 수익률에 더하는 값."""
    r_ = np.random.default_rng(seed)
    c_ = np.empty(n_)
    c_[0] = 50.0
    boost, wait, level = 0, 0, None
    for i in range(1, n_):
        ret = 0.0003 + r_.normal(0, 0.018) + (spring if boost > 0 else 0.0)
        boost = max(0, boost - 1)
        c_[i] = c_[i - 1] * np.exp(ret)
        if wait > 0:
            wait -= 1
            if c_[i] > level:
                boost, wait = (10 if spring else 0), 0
        elif i > look:
            fl = c_[i - look:i - 1].min()
            if c_[i] < fl <= c_[i - 1]:
                wait, level = 10, fl
    return mk(c_, r_, n_)


def run(**kw):
    out = engine.evaluate_reclaim({f"D{i}": walk(5200, 100 + i, **kw) for i in range(6)},
                                  {f"H{i}": walk(5200, 200 + i, **kw) for i in range(24)} | {"005930.KS": walk(5200, 999, **kw)})["summary"]
    return out, out["universes"]["HOLDOUT(새 종목)"]["blocks"]["전체 기간"]


o0, b0 = run()
d0 = b0["desc"]
print("  효과 없는 가격: 접근 뒤", {k: round(v * 100, 1) for k, v in d0["approach"].items()}, "| 이탈 뒤 회복", round(d0["reclaim_rate"] * 100, 1),
      "% | 회복 뒤 vs 무작위 z", round(b0["vs_random"]["z"], 2))
chk("3 효과 없는 가격(무작위 걸음): 미통과, |z| < 3, 화면 문구 '검증 안 됨'", not o0["verdict"]["passed"] and abs(b0["vs_random"]["z"]) < 3
    and o0["verdict"]["text"].startswith("검증 안 됨"))
chk("3-2 집계: 세 갈래 합 100%, 회복 비율 0~100%", abs(sum(d0["approach"].values()) - 1) < 1e-9 and 0 < d0["reclaim_rate"] < 1 and d0["n_break"] > 100)
o1, b1 = run(spring=0.004)
print("  깨고 올라오면 더 오르는 가격: 회복 뒤", round(b1["vs_random"]["actual"]["mean"] * 100, 2), "% vs", round(b1["vs_random"]["baseline"]["mean"] * 100, 2),
      "% z", round(b1["vs_random"]["z"], 2))
chk("3-3 깨고 올라온 뒤 더 오르게 만든 가격: 통과(z ≥ 3, 개발 묶음 앞·뒤 모두)", o1["verdict"]["passed"] and o1["verdict"]["text"].startswith("검증됨"),
    o1["verdict"])
chk("3-4 판정 기준 사전 고정(z 3, 사건 100건)", engine.RECLAIM_CRITERIA == {"MIN_Z": 3.0, "MIN_EVENTS": 100})
rep = engine.render_reclaim_report(o1, generated_at="test")
chk("3-5 보고서: 판정·집계 A·판정 B·한국 구분", "## 판정: 통과" in rep and "**A. 얼마나 자주 깨고 회복하나**" in rep and "**B. 회복 다음 날 시가" in rep
    and "전체 기간 · 한국" in rep and "미국 123·한국 25" in rep)
used = set()
for cfg in (engine.FRAME_A_CONFIG, engine.FRAME_B_CONFIG, engine.TBT_CONFIG, engine.SHORT_BT_CONFIG, engine.LEVEL_BT_CONFIG):
    for k in ("DEV_UNIVERSE", "HOLDOUT_UNIVERSE", "SCANNER_UNIVERSE"):
        used |= set(cfg.get(k, []))
used |= {s for g in engine.BACKTEST_CONFIG["SETUP_UNIVERSE"].values() for s in g}
ho = engine.RECLAIM_BT_CONFIG["HOLDOUT_UNIVERSE"]
chk("3-6 HOLDOUT 148종목(미국 123·한국 25), 이전 검증 묶음(지지·손절선 시험 포함)과 겹치지 않음", len(ho) == len(set(ho)) == 148
    and sum(1 for t in ho if t.endswith(".KS")) == 25 and not (set(ho) & used), sorted(set(ho) & used))

import subprocess
_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
hp = subprocess.run([sys.executable, "run_backtest.py", "--help"], cwd=_root, capture_output=True, text=True, encoding="utf-8",
                    timeout=120, env={**_os.environ, "PYTHONIOENCODING": "utf-8"})
chk("4 run_backtest.py --help 정상(--reclaim 포함)", hp.returncode == 0 and "--reclaim" in hp.stdout, (hp.returncode, hp.stderr[-300:]))

print(f"\n결과: {'통과' if not FAILS else '실패 ' + str(FAILS)}")
sys.exit(1 if FAILS else 0)
