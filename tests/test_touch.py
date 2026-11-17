"""탐색: 수평선 터치 횟수별 뚫림 (engine/touch_explore.py) 손 계산 검사 + 방식 점검."""
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


def seg(a, b):
    return list(np.linspace(a, b, int(abs(b - a)), endpoint=False))


def frame(path, atr=2.0):
    c = np.array(path, float)
    return pd.DataFrame({"Open": c, "High": c + 0.1, "Low": c - 0.1, "Close": c, "ATR20": atr}, index=pd.bdate_range("2021-01-04", periods=len(c)))


# 1) 지지선 손 계산 (ATR 2: 스윙 확정 6, 터치 구간 선 + 1, 뚫림 선 − 1, 다시 세려면 선 + 4 위 종가)
#    110 → 100(저점, 저가 99.9 = 선) → 108 → 100.5(2번째) → 108 → 100.5(3번째) → 108 → 100.5(4번째) → 95(뚫림)
path = (seg(110, 100) + seg(100, 108) + seg(108, 100.5) + seg(100.5, 108) + seg(108, 100.5) + seg(100.5, 108) + seg(108, 100.5)
        + seg(100.5, 95) + [95.0] * 20)
ev = engine.touch_events(None, ind=frame(path))
print("  지지선 터치:", ev[["kind", "n", "level", "broke"]].to_dict("records"))
chk("1 지지선(99.9): 2·3번째 터치는 안 뚫림, 4번째 터치 뒤 10거래일 안에 종가 < 98.9 → 뚫림", len(ev) == 3 and (ev["kind"] == "S").all()
    and ev["n"].tolist() == [2, 3, 4] and ev["broke"].tolist() == [0.0, 0.0, 1.0] and abs(ev["level"].iloc[0] - 99.9) < 1e-9, ev)
chk("1-2 가까운 새 저점(100.4)은 새 선을 만들지 않음(기존 선의 재테스트) — 선은 하나", ev["level"].nunique() == 1)
stay = seg(110, 100) + seg(100, 108) + seg(108, 100.5) + [100.5, 101.5, 100.6, 102.0, 100.7, 101.0] + seg(101, 108) + [108.0] * 15
ev2 = engine.touch_events(None, ind=frame(stay))
chk("1-3 선 근처에 붙어 며칠 머무는 것은 한 번의 터치", ev2["n"].tolist() == [2], ev2["n"].tolist())

# 2) 저항선(거울상): 90 → 100(고점, 고가 100.1 = 선) → 92 → 99.5(2번째) → 92 → 99.5(3번째) → 92 → 99.5(4번째) → 105(위로 뚫림)
up = (seg(90, 100) + seg(100, 92) + seg(92, 99.5) + seg(99.5, 92) + seg(92, 99.5) + seg(99.5, 92) + seg(92, 99.5) + seg(99.5, 105) + [105.0] * 20)
ev3 = engine.touch_events(None, ind=frame(up))
r3 = ev3[ev3["kind"] == "R"]
print("  저항선 터치:", r3[["kind", "n", "level", "broke"]].to_dict("records"))
chk("2 저항선(100.1): 4번째 터치에서 위로 뚫림(종가 > 101.1)", r3["n"].tolist() == [2, 3, 4] and r3["broke"].tolist() == [0.0, 0.0, 1.0], r3)

# 3) 수명: 만든 지 250거래일이 지난 선은 터치를 세지 않음
old = seg(110, 100) + seg(100, 108) + [108.0] * 260 + seg(108, 100.5) + seg(100.5, 108) + [108.0] * 15
chk("3 250거래일 지난 선은 없어짐", len(engine.touch_events(None, ind=frame(old)).query("kind == 'S'")) == 0)


# 4) 방식 점검 — 효과 없는 가격(무작위 걸음)에서는 터치 횟수와 뚫림 비율이 무관해야 함
def rw(n, seed):
    r_ = np.random.default_rng(seed)
    c = 50 * np.exp(np.cumsum(0.0003 + r_.normal(0, 0.018, n)))
    o = np.concatenate([[c[0]], c[:-1]])
    return pd.DataFrame({"Open": o, "High": np.maximum(o, c) * (1 + np.abs(r_.normal(0, 0.006, n))),
                         "Low": np.minimum(o, c) * (1 - np.abs(r_.normal(0, 0.006, n))), "Close": c, "Volume": 1e6},
                        index=pd.bdate_range("2000-01-03", periods=n))


res = engine.run_touch_explore({f"N{i}": rw(6500, i) for i in range(40)} | {"005930.KS": rw(6500, 99)})
b = res["blocks_touch"]["전체 기간"]
print("  효과 없는 가격:", {k: (v["n"], None if v["break_rate"] is None else round(v["break_rate"] * 100, 1)) for k, v in b["by_touch"].items()},
      "| 4번째 − 2·3번째 z", round(b["target_vs_base"]["z"], 2))
chk("4 효과 없는 가격: 후보 아님, |z| < 2", not res["candidate"] and abs(b["target_vs_base"]["z"]) < 2 and res["not_candidate_reasons"])
chk("4-2 구분 7칸(기간·시장·지지선만·저항선만), 기록 한 줄용 값", len(res["blocks_touch"]) == 7 and "전체 기간 · 저항선만" in res["blocks_touch"]
    and res["blocks"]["전체 기간"]["vs_random"] is b["target_vs_base"] and res["n_events"] == b["by_touch"]["4번째"]["n"])


# 5) 4번째 터치에서 더 뚫리게 만든 사건표 → 후보 (계산·판정 부분만)
rng = np.random.default_rng(7)
dates = pd.to_datetime(rng.choice(pd.bdate_range("2002-01-01", "2025-12-31"), 9000))
nn = rng.choice([2, 3, 4, 5], size=9000, p=[0.5, 0.25, 0.15, 0.10])
fake = pd.DataFrame({"date": dates, "t": 0, "kind": rng.choice(["S", "R"], 9000), "n": nn, "level": 1.0,
                     "broke": (rng.random(9000) < np.where(nn >= 4, 0.75, 0.55)).astype(float)})
_orig = engine.touch_events
engine.touch_events = lambda d, ind=None, decisive_days=None: fake
try:
    res2 = engine.run_touch_explore({"ONE": rw(400, 1)})
finally:
    engine.touch_events = _orig
x = res2["blocks_touch"]["전체 기간"]["target_vs_base"]
print("  4번째에서 더 뚫리는 사건표: 차이", round(x["diff"] * 100, 1), "%p z", round(x["z"], 1), "후보", res2["candidate"])
chk("5 4번째 터치 뚫림 75% 대 2·3번째 55% → 후보(차이 약 +20%p, z ≥ 2, 앞·뒤 기간 모두)", res2["candidate"] and 0.15 < x["diff"] < 0.25)
rep = engine.render_touch_report(res2, generated_at="test")
chk("5-2 보고서: 증거 아님 경고·후보 여부·터치별 표", "탐색 결과 — 증거 아님" in rep and "## 후보 여부: 후보" in rep and "| 4번째 |" in rep.replace("| 3번째 | 4번째 |", "| 4번째 |"))
chk("5-3 등록: --explore touch4 (전용 계산)", engine.EXPLORE_HYPOTHESES["touch4"]["custom"] == "touch")


# 6) touch4b — 확정 뚫림(넘어간 뒤 20거래일 동안 돌아오지 않음). ATR 2: 넘어감 = 종가 < 98.9, 돌아옴 = 종가 ≥ 99.9
N = engine.TOUCH_CONFIG["DECISIVE_DAYS"]
#    2번째 터치 뒤 일시 이탈(97로 내려갔다가 사흘 뒤 104) → 선은 살아 있어 3·4번째 터치를 셈, 4번째 터치 뒤 95에서 20일 넘게 머묾
tmp_path = (seg(110, 100) + seg(100, 108) + seg(108, 100.5) + [99.0, 97.0, 97.5, 98.0] + seg(98, 108) + seg(108, 100.5) + seg(100.5, 108)
            + seg(108, 100.5) + seg(100.5, 95) + [95.0] * 30)
evb_all = engine.touch_events(None, ind=frame(tmp_path), decisive_days=N)
evb = evb_all[(evb_all["level"] - 99.9).abs() < 1e-9]   # 99.9 선만 (잠깐 이탈 때 생긴 97 저점은 별도의 새 지지선)
print("  확정 뚫림:", evb[["n", "outcome", "decisive"]].to_dict("records"))
chk("6 2번째 터치: 일시 이탈(넘어간 뒤 20일 안에 선 위로 돌아옴)", evb["outcome"].iloc[0] == "temporary" and evb["decisive"].iloc[0] == 0.0, evb)
chk("6-2 일시 이탈한 선은 살아 있어 3·4번째 터치를 셈, 4번째는 확정 뚫림", evb["n"].tolist() == [2, 3, 4]
    and evb["outcome"].tolist() == ["temporary", "held", "decisive"], evb[["n", "outcome"]].to_dict("records"))
ev_old = engine.touch_events(None, ind=frame(tmp_path))
ev_old = ev_old[(ev_old["level"] - 99.9).abs() < 1e-9]
chk("6-3 같은 가격, 예전 정의(touch4)는 일시 이탈에서 99.9 선을 지워 3·4번째 터치가 없음", ev_old["n"].tolist() == [2], ev_old["n"].tolist())
chk("6-3b 잠깐 이탈 때 생긴 97 저점은 1×ATR 넘게 떨어져 있어 별도의 새 지지선", ((evb_all["level"] - 99.9).abs() > 2).any())
short_tail = (seg(110, 100) + seg(100, 108) + seg(108, 100.5) + seg(100.5, 95) + [95.0] * 10)
chk("6-4 넘어간 뒤 20거래일 자료가 없으면 사건에서 뺌", len(engine.touch_events(None, ind=frame(short_tail), decisive_days=N)) == 0)
resb = engine.run_touch_explore({f"N{i}": rw(6500, i) for i in range(40)} | {"005930.KS": rw(6500, 99)}, decisive=True)
bb = resb["blocks_touch"]["전체 기간"]
fx = bb["by_touch"]["4번째"]
print("  효과 없는 가격(확정 뚫림):", {k: (v["n"], None if v["break_rate"] is None else round(v["break_rate"] * 100, 1)) for k, v in bb["by_touch"].items()},
      "| 4번째 버팀·일시·확정", [round((fx[k] or 0) * 100, 1) for k in ("held", "temporary", "decisive")], "| z", round(bb["target_vs_base"]["z"], 2))
chk("6-5 효과 없는 가격: 후보 아님, 세 갈래 합 100%", not resb["candidate"] and abs(fx["held"] + fx["temporary"] + fx["decisive"] - 1) < 1e-9
    and abs(bb["target_vs_base"]["z"]) < 2 and resb["name"] == "touch4b")
repb = engine.render_touch_report(resb, generated_at="test")
chk("6-6 보고서: 세 갈래 표·확정 뚫림 기준", "| 터치 | 버팀 | 일시 이탈 | 확정 뚫림 | 사건 수 |" in repb and "20거래일" in repb and "`touch4b`" in repb)
chk("6-7 등록: --explore touch4b", engine.EXPLORE_HYPOTHESES["touch4b"]["custom"] == "touch_b" and engine.TOUCH_CONFIG["DECISIVE_DAYS"] == 20)

print(f"\n결과: {'통과' if not FAILS else '실패 ' + str(FAILS)}")
sys.exit(1 if FAILS else 0)
