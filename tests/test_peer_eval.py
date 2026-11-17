"""피어 순위 채점 (engine/peer_eval.py) 손 계산 검사 + 방식 점검."""
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
PE = engine.PEER_EVAL

# 1) 줄 세운 날: 시작일(2026-11-17) 이후 매달 첫 기록일
rk0 = pd.DataFrame({"rank_date": pd.to_datetime(["2026-11-15", "2026-11-16", "2026-11-17", "2026-11-18", "2026-12-01", "2026-12-02", "2027-01-04"])})
chk("1 줄 세운 날: 11-17(시작일 전 기록은 뺌), 12-01, 01-04", [d.date().isoformat() for d in engine.formation_dates(rk0)]
    == ["2026-11-17", "2026-12-01", "2027-01-04"])

# 2) 수익률 손 계산: 가격 기준일 다음 거래일 시가 → 21거래일째 종가
idx = pd.bdate_range("2026-11-02", periods=400)
c = 100.0 + np.arange(400)
px = pd.DataFrame({"Open": c - 0.5, "High": c + 1, "Low": c - 1, "Close": c}, index=idx)
fr = engine.peer_eval._fwd_returns(px, "2026-11-13")
i = int(np.searchsorted(idx.values, np.datetime64("2026-11-13"), side="right"))
chk("2 1개월 = 기준일 다음 거래일 시가 → 그날 포함 21거래일째 종가", close(fr["1개월"], c[i + 20] / (c[i] - 0.5) - 1) and idx[i] == pd.Timestamp("2026-11-16"))
chk("2-2 6개월(126거래일)·12개월(252거래일)도 계산, 자료 끝을 넘으면 빠짐", "6개월" in fr and "12개월" in fr
    and "12개월" not in engine.peer_eval._fwd_returns(px, idx[200]))


# 3) 묶음 하나 손 계산: 6종목, 가치 순위 1~5 + 하드 킬 1(순위 없음). 상위 1/3 = 2종목(1·2위), 하위 1/3 = 2종목(4·5위)
def flat_then(r1m, n=400):
    """첫 거래일(11-16) 시가 100, 그 뒤 하루 수익률을 일정하게 해서 21거래일째 종가가 100 × (1 + r1m)."""
    g = (1 + r1m) ** (1 / 20)
    cl = 100.0 * g ** np.arange(n)
    return pd.DataFrame({"Open": np.r_[100.0, cl[:-1]], "High": cl * 1.01, "Low": cl * 0.99, "Close": cl},
                        index=pd.bdate_range("2026-11-16", periods=n))


def rank_rows(date, group, ranks, killed=(), as_of="2026-11-13", trend=None):
    out = []
    for t, r in ranks.items():
        out.append({"rank_date": date, "group": group, "ticker": t, "value_rank": r, "trend_rank": (trend or ranks)[t],
                    "surv_kill": t in killed, "price_as_of": as_of})
    for t in killed:
        out.append({"rank_date": date, "group": group, "ticker": t, "value_rank": np.nan, "trend_rank": np.nan, "surv_kill": True,
                    "price_as_of": as_of})
    return out


rets = {"A": 0.10, "B": 0.06, "C": 0.03, "D": 0.00, "E": -0.04, "K": -0.20}
pm = {t: flat_then(r) for t, r in rets.items()}
rk = pd.DataFrame(rank_rows("2026-11-17", "g1", {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5}, killed=("K",)))
gm = engine.group_month_rows(rk, pm)
row = gm[(gm["strategy"] == "value") & (gm["horizon"] == "1개월")].iloc[0]
print("  묶음 하나:", {k: round(float(row[k]), 4) for k in ("top", "bottom", "group_mean", "killed")})
chk("3 상위 1/3(A·B) 평균 8%, 하위 1/3(D·E) 평균 −2%, 묶음 평균(하드 킬 포함 6종목) −0.83%, 하드 킬 −20%",
    close(row["top"], 0.08, 1e-9) and close(row["bottom"], -0.02, 1e-9) and close(row["group_mean"], sum(rets.values()) / 6, 1e-9)
    and close(row["killed"], -0.20, 1e-9), row.to_dict())
chk("3-2 순위 있는 종목 4개 < 5인 묶음은 채점 안 함", len(engine.group_month_rows(
    pd.DataFrame(rank_rows("2026-11-17", "g2", {"A": 1, "B": 2, "C": 3, "D": 4})), pm)) == 0)


# 4) 방식 점검 — 여러 달·여러 묶음: 순위가 수익과 무관한 가격 vs 상위가 더 오르는 가격
def build(edge, n_months=16, n_groups=8, seed=1):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2026-11-02", periods=420 + 21 * n_months)
    months = [d for d in pd.Series(days).groupby(pd.Series(days).dt.to_period("M")).min() if d >= pd.Timestamp("2026-11-17")][:n_months]
    rows, pmap = [], {}
    for g in range(n_groups):
        tickers = [f"G{g}S{j}" for j in range(6)]
        kr = g % 4 == 0
        tickers = [f"{100000 + g * 10 + j:06d}.KS" for j in range(6)] if kr else tickers
        paths = {t: rng.normal(0.0003, 0.015, len(days)) for t in tickers}
        for m in months:
            order = rng.permutation(6)          # 그달 순위는 무작위
            i0 = int(np.searchsorted(days.values, np.datetime64(m)))
            for pos, j in enumerate(order):
                t = tickers[j]
                paths[t][i0 + 1:i0 + 22] += edge * (2.5 - pos) / 2.5 / 21    # 1위 쪽이 그 뒤 21일 더 오름(edge = 1·6위 차이의 절반)
                as_of = days[i0 - 1].date().isoformat()
                rows.append({"rank_date": m, "group": f"g{g}", "ticker": t, "value_rank": pos + 1, "trend_rank": 6 - pos,
                             "surv_kill": False, "price_as_of": as_of})
        for t in tickers:
            cl = 50 * np.exp(np.cumsum(paths[t]))
            pmap[t] = pd.DataFrame({"Open": np.r_[cl[0], cl[:-1]], "High": cl * 1.005, "Low": cl * 0.995, "Close": cl}, index=days)
    return pd.DataFrame(rows), pmap


rk0, pm0 = build(0.0)
r0 = engine.evaluate_peer_ranks(rk0, pm0)
v0 = r0["strategies"]["value"]
one0 = v0["horizons"]["1개월"]["전체"]["ls"]
print("  순위와 무관한 가격: 1개월 상위 − 하위", round(one0["mean"] * 100, 2), "%p t", round(one0["t"], 2), "| 6개월 달 수", v0["verdict"]["months_done"], v0["verdict"]["status"])
chk("4 순위와 무관한 가격: 1개월 |t| < 2, 6개월 판정은 미통과 또는 보류", abs(one0["t"]) < 2 and not v0["verdict"]["passed"])
rk1, pm1 = build(0.03, n_months=20)
r1 = engine.evaluate_peer_ranks(rk1, pm1)
v1, t1 = r1["strategies"]["value"], r1["strategies"]["trend"]
one1 = v1["horizons"]["1개월"]["전체"]["ls"]
six1 = v1["horizons"]["6개월"]["전체"]["ls"]
print("  상위가 더 오르는 가격: 1개월", round(one1["mean"] * 100, 2), "%p t", round(one1["t"], 1), "| 6개월", round(six1["mean"] * 100, 2),
      "%p t", None if six1["t"] is None else round(six1["t"], 1), "달", six1["n"], "→", v1["verdict"]["status"], "| 추세 관점(순위 반대)", t1["verdict"]["status"])
chk("4-2 상위가 더 오르는 가격: 가치 관점 1개월 t ≥ 2, 6개월 결과 12달 이상이면 통과", one1["t"] >= 2 and v1["verdict"]["status"] == "통과", v1["verdict"])
chk("4-3 순위를 뒤집은 추세 관점은 통과하지 않음(상위 − 하위 < 0)", t1["verdict"]["status"] != "통과" and t1["horizons"]["1개월"]["전체"]["ls"]["mean"] < 0)
chk("4-4 6개월 결과가 12달 안 되면 판정 보류", engine.evaluate_peer_ranks(*build(0.03, n_months=8))["strategies"]["value"]["verdict"]["status"] == "판정 보류")
chk("4-5 한국 묶음 따로 계산", v1["horizons"]["1개월"]["한국"]["ls"]["n"] > 0 and v1["horizons"]["1개월"]["미국"]["ls"]["n"] > 0)
rep = engine.render_peer_eval_report(r1, generated_at="test")
chk("4-6 보고서: 관점별 판정·6개월 판정 줄·보류 중 규칙 변경 금지 안내", "## 가치 관점: 통과" in rep and "6개월 (판정)" in rep and "바꾸지 않는다" in rep)
empty = engine.evaluate_peer_ranks(pd.DataFrame(columns=["rank_date", "group", "ticker", "value_rank", "trend_rank", "surv_kill", "price_as_of"]), {})
chk("4-7 기록 없음: 판정 보류, 보고서에 '아직 결과가 나온 달 없음'", empty["strategies"]["value"]["verdict"]["status"] == "판정 보류"
    and "아직 결과가 나온 달 없음" in engine.render_peer_eval_report(empty))
chk("4-8 방법 사전 고정(시작일·6개월·12달·t 2)", PE["START"] == "2026-11-17" and PE["PRIMARY"] == "6개월" and PE["HORIZONS"]["6개월"] == 126
    and PE["MIN_MONTHS"] == 12 and PE["MIN_T"] == 2.0)

import subprocess
_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
hp = subprocess.run([sys.executable, "run_backtest.py", "--help"], cwd=_root, capture_output=True, text=True, encoding="utf-8",
                    timeout=120, env={**_os.environ, "PYTHONIOENCODING": "utf-8"})
chk("5 run_backtest.py --help 정상(--peer-eval 포함)", hp.returncode == 0 and "--peer-eval" in hp.stdout, (hp.returncode, hp.stderr[-300:]))

print(f"\n결과: {'통과' if not FAILS else '실패 ' + str(FAILS)}")
sys.exit(1 if FAILS else 0)
