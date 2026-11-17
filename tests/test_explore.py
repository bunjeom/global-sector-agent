"""가설 탐색 도구 (engine/explore.py) 검사 — 사건 뒤 수익률 손 계산, 후보 기준, 기록·가격 캐시, 명령줄."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _offline  # noqa: F401
import sys, tempfile
import numpy as np
import pandas as pd
import engine

FAILS = []


def chk(n, c, extra=None):
    print(("PASS " if c else "FAIL ") + n + (f"  [{extra}]" if (not c and extra is not None) else ""))
    if not c:
        FAILS.append(n)


close = lambda a, b, tol=1e-9: a is not None and b is not None and abs(float(a) - float(b)) < tol
EC = engine.EXPLORE_CONFIG

# 1) 사건 위치 정리: 앞 사건 뒤 10거래일 안의 사건은 뺌
chk("1 dedupe: [3, 5, 13, 14, 30] → [3, 14, 30]", engine.dedupe_events(np.array([3, 5, 13, 14, 30])).tolist() == [3, 14, 30])


# 2) 가짜 가격으로 전체 경로: 사건 뒤 수익률 = 다음 날 시가 → 10거래일 뒤 종가
def walk(n, seed, boost=0.0, every=45):
    """every일마다 '사건일'을 두고, 그 뒤 10일 동안 하루 수익률에 boost를 더한다(효과가 있는 가짜 가격)."""
    r_ = np.random.default_rng(seed)
    ret = 0.0003 + r_.normal(0, 0.018, n)
    marks = np.arange(300, n - 30, every)
    for m in marks:
        ret[m + 1:m + 11] += boost
    c_ = 50 * np.exp(np.cumsum(ret))
    o_ = np.concatenate([[c_[0]], c_[:-1]])
    hi = np.maximum(o_, c_) * (1 + np.abs(r_.normal(0, 0.006, n)))
    lo = np.minimum(o_, c_) * (1 - np.abs(r_.normal(0, 0.006, n)))
    return pd.DataFrame({"Open": o_, "High": hi, "Low": lo, "Close": c_, "Volume": 1e6}, index=pd.bdate_range("2000-01-03", periods=n))


marks_fn = lambda si, a: np.arange(300, len(si) - 30, 45)
one = walk(5200, 1)
r1 = engine.explore_ticker(one, marks_fn)
ev = r1["ev"]
t0 = int(ev.iloc[0]["t"])
si1 = engine.short_indicators(engine._bt_naive(one))
chk("2 사건 뒤 수익률 = 다음 날 시가 → 10거래일 뒤 종가", close(ev.iloc[0]["r10"], si1["Close"].iloc[t0 + 10] / si1["Open"].iloc[t0 + 1] - 1), ev.iloc[0]["r10"])
chk("2-2 20거래일(참고)·무작위 표본 20배·하락 폭 맞춘 표본", close(ev.iloc[0]["r20"], si1["Close"].iloc[t0 + 20] / si1["Open"].iloc[t0 + 1] - 1)
    and len(r1["rnd"]) == 20 * len(ev) and 0 < len(r1["mat"]) <= 20 * len(ev), (len(ev), len(r1["rnd"]), len(r1["mat"])))
chk("2-3 2001년 전 사건은 세지 않음", (ev["date"] >= pd.Timestamp(EC["START"])).all())

pm0 = {f"N{i}": walk(5200, 10 + i) for i in range(12)} | {"005930.KS": walk(5200, 99)}
res0 = engine.run_explore("x", pm0, fn=marks_fn)
z0 = res0["blocks"]["전체 기간"]["vs_random"]["z"]
print("  효과 없는 가격: 사건", res0["n_events"], "z", round(z0, 2), "후보", res0["candidate"], res0["not_candidate_reasons"])
chk("3 효과 없는 가격: 후보 아님(|z| < 2), 이유가 적힘", not res0["candidate"] and abs(z0) < 2 and res0["not_candidate_reasons"])
pm1 = {f"N{i}": walk(5200, 10 + i, boost=0.003) for i in range(12)} | {"005930.KS": walk(5200, 99, boost=0.003)}
res1 = engine.run_explore("x", pm1, fn=marks_fn)
z1 = res1["blocks"]["전체 기간"]["vs_random"]
print("  효과 있는 가격: 차이", round(z1["diff"] * 100, 2), "%p z", round(z1["z"], 2), "후보", res1["candidate"])
chk("3-2 사건 뒤 10일 동안 더 오르게 만든 가격: 후보(|z| ≥ 2, 앞·뒤 기간 같은 방향)", res1["candidate"] and z1["diff"] > 0.02 and res1["same_sign"])
chk("3-3 한국 종목 구분·기간 5칸", list(res1["blocks"]) == ["전체 기간", "앞 기간", "뒤 기간", "전체 기간 · 미국", "전체 기간 · 한국"]
    and res1["blocks"]["전체 기간 · 한국"]["vs_random"]["actual"]["n"] > 0)

# 4) 기대 방향: 기대와 반대로 나오면 z가 커도 후보 아님
engine.EXPLORE_HYPOTHESES["_t_neg"] = {"label": "테스트(나쁠 것으로 기대)", "expect": -1, "fn": marks_fn}
try:
    rn = engine.run_explore("_t_neg", pm1)
    chk("4 기대 방향과 반대면 후보 아님", not rn["candidate"] and "기대한 방향과 반대" in rn["not_candidate_reasons"], rn["not_candidate_reasons"])
finally:
    del engine.EXPLORE_HYPOTHESES["_t_neg"]
try:
    engine.run_explore("없는이름", pm0)
    chk("4-2 등록 안 된 이름은 오류", False)
except KeyError:
    chk("4-2 등록 안 된 이름은 오류", True)

# 5) 등록된 견본 가설이 돌아가는지(이탈 뒤 회복·손절선 이탈)
for nm in ("reclaim", "stop_break"):
    rr = engine.run_explore(nm, pm0)
    chk(f"5 견본 가설 `{nm}`: 사건이 잡히고 효과 없는 가격에서 후보 아님", rr["n_events"] > 100 and not rr["candidate"], (rr["n_events"], rr["candidate"]))

# 6) 보고서·기록
rep = engine.render_explore_report(res1, generated_at="2026-11-17 16:00")
chk("6 보고서: '증거 아님' 경고·후보 여부·읽는 법", "탐색 결과 — 증거 아님" in rep and "## 후보 여부: 후보" in rep and "맞추기" in rep)
tmp = tempfile.mkdtemp(prefix="explore_")
engine.append_explore_log(res0, "2026-11-17 16:00", tmp)
p = engine.append_explore_log(res1, "2026-11-17 16:05", tmp)
lg = open(p, encoding="utf-8").read()
chk("6-2 기록: 머리말 한 번 + 볼 때마다 한 줄(2줄), 판정 표시", lg.count("# 탐색 기록") == 1 and lg.count("| `x` |") == 2
    and "| 후보 아님 |" in lg and "| 후보 |" in lg, lg[-300:])

# 7) 가격 캐시: 저장 → 읽기(값·날짜 그대로), 없는 종목은 빠짐
cd = _os.path.join(tmp, "cache")
engine.save_price_cache({"AAA": one.iloc[:50], "005930.KS": one.iloc[:40]}, cd)
back = engine.load_price_cache(["AAA", "005930.KS", "ZZZ"], cd)
chk("7 가격 캐시 왕복", set(back) == {"AAA", "005930.KS"} and len(back["AAA"]) == 50
    and np.allclose(back["AAA"]["Close"].to_numpy(), one["Close"].iloc[:50].to_numpy()) and back["AAA"].index[0] == one.index[0])

# 8) 탐색용 종목 = 이미 쓴 개별 종목 전부(중복 없음, ETF 없음), 명령줄
uni = EC["UNIVERSE"]
chk("8 탐색용 종목: 중복 없음, 판정 묶음들을 모두 포함, 섹터 ETF 없음", len(uni) == len(set(uni))
    and set(engine.RECLAIM_BT_CONFIG["HOLDOUT_UNIVERSE"]) <= set(uni) and set(engine.FRAME_A_CONFIG["DEV_UNIVERSE"]) <= set(uni)
    and not ({"XLK", "SMH", "EWJ", "SPY"} & set(uni)), len(uni))
import subprocess
_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
env = {**_os.environ, "PYTHONIOENCODING": "utf-8"}
hp = subprocess.run([sys.executable, "run_backtest.py", "--help"], cwd=_root, capture_output=True, text=True, encoding="utf-8", timeout=120, env=env)
ls = subprocess.run([sys.executable, "run_backtest.py", "--explore", "list"], cwd=_root, capture_output=True, text=True, encoding="utf-8", timeout=120, env=env)
chk("8-2 run_backtest.py --help(--explore 포함), --explore list(가설 이름)", hp.returncode == 0 and "--explore" in hp.stdout
    and ls.returncode == 0 and "reclaim" in ls.stdout, (hp.returncode, ls.returncode, ls.stderr[-200:]))

print(f"\n결과: {'통과' if not FAILS else '실패 ' + str(FAILS)}")
sys.exit(1 if FAILS else 0)
