"""탐색: 모멘텀 (engine/momentum_explore.py) 손 계산 검사 + 방식 점검."""
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

# 1) 한 종목 손 계산: 종가 = 100 + 위치, 시가 = 종가 − 0.5
idx = pd.bdate_range("2000-01-03", periods=600)
c = 100.0 + np.arange(600)
df = pd.DataFrame({"Open": c - 0.5, "High": c + 1, "Low": c - 1, "Close": c}, index=idx)
rows = engine.stock_month_rows(df, "mom12")
per = idx.to_period("M")
m0 = rows.iloc[0]
i = int(np.flatnonzero(per == m0["month"])[-1])
print("  첫 달:", m0["month"], "위치", i)
chk("1 12개월 모멘텀 = 21거래일 전 종가 ÷ 252거래일 전 종가 − 1", close(m0["sig"], c[i - 21] / c[i - 252] - 1) and i >= 252)
chk("1-2 1개월 결과 = 21거래일 뒤 종가 ÷ 다음 날 시가 − 1", close(m0["f1"], c[i + 21] / (c[i + 1] - 0.5) - 1))
chk("1-3 3개월 결과 = 63거래일 뒤 종가 ÷ 다음 날 시가 − 1", close(m0["f3"], c[i + 63] / (c[i + 1] - 0.5) - 1))
r63 = engine.stock_month_rows(df, "ret63").iloc[0]
j = int(np.flatnonzero(per == r63["month"])[-1])
chk("1-4 63일 수익률 = 그날 종가 ÷ 63거래일 전 종가 − 1", close(r63["sig"], c[j] / c[j - 63] - 1))
chk("1-5 마지막 달은 1개월 결과 없음(NaN)", np.isnan(rows.iloc[-1]["f1"]))

# 2) 달마다 상위·하위 20%: 종목 25개, 신호 = 1..25, 결과 = 신호 × 1%
pan = pd.DataFrame({"month": pd.Period("2005-01"), "market": "US", "sig": np.arange(1, 26, dtype=float),
                    "f1": np.arange(1, 26) / 100.0, "f3": np.arange(1, 26) / 100.0})
ms = engine.monthly_spreads(pan).iloc[0]
chk("2 상위 20%(21~25) 평균 23%, 하위 20%(1~5) 평균 3%, 차이 20%p, 전체 평균 13%", close(ms["top_f1"], 0.23) and close(ms["bot_f1"], 0.03)
    and close(ms["ls_f1"], 0.20) and close(ms["all_f1"], 0.13) and close(ms["tm_f1"], 0.10), ms.to_dict())
ms2 = engine.monthly_spreads(pan.iloc[:19]).iloc[0]
chk("2-2 종목 19개 < 20 → 그달 계산 안 함", np.isnan(ms2["ls_f1"]))
chk("2-3 Newey-West: 시차 0 = 보통 표준오차", close(engine.nw_se(np.array([1.0, 2, 3, 4, 5]), 0), np.std([1, 2, 3, 4, 5]) / np.sqrt(5)))


# 3) 방식 점검 — 효과 없는 가격 vs 모멘텀이 있는 가격
def walk(n, seed, persist=False):
    r_ = np.random.default_rng(seed)
    e = r_.normal(0, 0.018, n)
    if persist:   # 종목마다 숨은 추세: 달마다 이어지는 AR(1)(계수 0.9, 하루 표준편차 약 0.1%) → 최근에 오른 종목이 계속 오르는 경향
        k = n // 21 + 1
        mu = np.zeros(k)
        for t in range(1, k):
            mu[t] = 0.9 * mu[t - 1] + r_.normal(0, 0.00045)
        e = e + np.repeat(mu, 21)[:n]
    cl = 50 * np.exp(np.cumsum(0.0003 + e))
    o = np.concatenate([[cl[0]], cl[:-1]])
    return pd.DataFrame({"Open": o, "High": np.maximum(o, cl) * 1.004, "Low": np.minimum(o, cl) * 0.996, "Close": cl},
                        index=pd.bdate_range("1999-01-04", periods=n))


null = {f"N{i}": walk(6800, i) for i in range(60)} | {f"{i:06d}.KS": walk(6800, 500 + i) for i in range(25)}
r0 = engine.run_momentum_explore(null, "mom12")
b0 = r0["blocks_mom"]["전체 기간"]["ls_f1"]
print("  효과 없는 가격: 달", b0["n"], "상위 − 하위", round(b0["mean"] * 100, 2), "%p t", round(b0["t"], 2), "후보", r0["candidate"])
chk("3 효과 없는 가격: 후보 아님, |t| < 2", not r0["candidate"] and abs(b0["t"]) < 2 and b0["n"] >= 200)
chk("3-2 한국 묶음(25종목)도 계산됨", r0["blocks_mom"]["전체 기간 · 한국"]["ls_f1"]["n"] >= 200)
mom = {f"N{i}": walk(6800, i, True) for i in range(60)} | {f"{i:06d}.KS": walk(6800, 500 + i, True) for i in range(25)}
r1 = engine.run_momentum_explore(mom, "mom12")
b1 = r1["blocks_mom"]["전체 기간"]["ls_f1"]
print("  모멘텀이 있는 가격: 상위 − 하위", round(b1["mean"] * 100, 2), "%p t", round(b1["t"], 2), "후보", r1["candidate"])
chk("3-3 모멘텀이 있는 가격: 후보(상위 − 하위 > 0, t ≥ 2, 앞·뒤 기간 모두)", r1["candidate"] and b1["mean"] > 0, (b1, r1["not_candidate_reasons"]))
r2 = engine.run_momentum_explore(mom, "ret63")
chk("3-4 ret63도 같은 경로로 계산", r2["name"] == "ret63" and r2["blocks_mom"]["전체 기간"]["ls_f1"]["n"] >= 200)
rep = engine.render_momentum_report(r1, generated_at="test")
chk("3-5 보고서: 증거 아님·판정표·3개월·생존 편향 주의", "탐색 결과 — 증거 아님" in rep and "**1개월 뒤 (판정)**" in rep and "**3개월 뒤 (참고)**" in rep
    and "상장폐지" in rep)
chk("3-6 등록: --explore mom12 / ret63", engine.EXPLORE_HYPOTHESES["mom12"]["custom"] == "mom" and engine.EXPLORE_HYPOTHESES["ret63"]["custom"] == "mom")

print(f"\n결과: {'통과' if not FAILS else '실패 ' + str(FAILS)}")
sys.exit(1 if FAILS else 0)
