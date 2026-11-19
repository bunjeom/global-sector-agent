"""H-TR 추세 투자 흐름 검증(engine/tr_backtest.py) — 박스·물량 소화·실적 쇼크·청산 손 계산, 보유 기간 맞춘 기준, 미래 자료 미사용,
탐색 기간 밖 가격 미사용, 우위를 심은 가짜 거래는 후보 / 우위 없는 가짜 거래는 후보 아님, 판정 기준·T1 결정·보고서."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _offline  # noqa: F401
import sys

import numpy as np
import pandas as pd

import engine

CF = engine.TR_CONFIG
FAILS = []


def check(name, cond, detail=""):
    print(("  ok  " if cond else "  FAIL") + f"  {name}" + (f"  — {detail}" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


def near(a, b, tol=1e-9):
    return a is not None and b is not None and np.isfinite(a) and abs(a - b) <= tol


# ---------------------------------------------------------------- 1. 박스·물량 소화
print("1. 박스 시작·고점 수·물량 소화(T1)")
ho, hp = np.array([10, 30, 50, 70]), np.array([105.0, 100.5, 95.0, 100.0])
lo_o, lo_p = np.array([20, 40, 60, 75]), np.array([90.0, 92.0, 94.0, 96.0])
bx = engine.tr_box(ho, hp, lo_o, lo_p, 3, 80, 99.0, 101.0)
check("1-1 낮은 고점은 지나가고, 구간 안 고점까지 박스·구간 위 고점에서 멈춤", bx["start"] == 30 and bx["tops"] == 2, str(bx))
check("1-2 박스 시작 뒤 확정 저점 마지막 둘이 오르고 둘 다 zl 아래 → 물량 소화", bx["absorb"] is True)
check("1-3 저점이 아직 확정 전(오른쪽 5봉)이면 세지 않음", engine.tr_box(ho, hp, lo_o, lo_p, 3, 79, 99.0, 101.0)["absorb"] is True
      and engine.tr_box(ho, hp, np.array([20, 40, 78]), np.array([90.0, 92.0, 97.0]), 3, 80, 99.0, 101.0)["absorb"] is False)
check("1-4 고점이 하나뿐이면 물량 소화 아님", engine.tr_box(ho, np.array([105.0, 90.0, 95.0, 100.0]), lo_o, lo_p, 3, 80, 99.0, 101.0)["absorb"] is False)
check("1-5 마지막 저점이 zl 위면 아님", engine.tr_box(ho, hp, lo_o, np.array([90.0, 92.0, 94.0, 99.5]), 3, 80, 99.0, 101.0)["absorb"] is False)
check("1-6 250거래일 넘으면 멈춤", engine.tr_box(np.array([10, 300, 320]), np.array([100.2, 95.0, 100.0]), lo_o, lo_p, 2, 330, 99.0, 101.0)["start"] == 320)

# ---------------------------------------------------------------- 2. 실적 쇼크
print("2. 실적 쇼크(발표 전날 → 다음 날 종가 ≤ −2 ATR)")
dts = pd.bdate_range("2010-01-04", periods=40)
cl = np.full(40, 100.0)
atr = np.full(40, 1.0)
cl[11] = 97.0
ed = np.array([dts[10]], dtype="datetime64[ns]")
es = lambda start, s, cov=pd.Timestamp("2005-01-01"), cl_=cl, ed_=ed: engine.tr_earn_status(dts.values, cl_, atr, ed_, start, s, cov)
check("2-1 박스 안 발표 반응 −3 ATR → 쇼크", es(5, 20) == "shock")
check("2-2 발표가 박스 시작 전이면 상관없음", es(12, 20) == "ok")
check("2-3 반응 −1.5 ATR은 쇼크 아님", es(5, 20, cl_=np.where(np.arange(40) == 11, 98.5, 100.0)) == "ok")
check("2-4 발표 다음 날이 신호일을 넘으면 신호일 종가까지", es(5, 10, cl_=np.where(np.arange(40) >= 10, 97.0, 100.0)) == "shock")
check("2-5 실적일 자료가 박스 시작을 덮지 못하면 확인 불가", es(5, 20, cov=pd.Timestamp("2010-01-12")) == "unknown" and es(5, 20, cov=pd.Timestamp("2010-01-11")) == "shock" and es(5, 20, cov=None) == "unknown")

# ---------------------------------------------------------------- 3. 청산
print("3. 청산(사전 등록 6장 순서) 손 계산")
n = 30
o = np.full(n, 100.0)
h = np.full(n, 101.0)
c = np.full(n, 100.0)
nob = np.zeros(n, dtype=bool)
ex = engine.tr_exit
h1 = h.copy(); h1[5] = 111.0
check("3-1 진입일 장중 목표 → 목표가에 전부", ex(o, h1, c, 5, 29, 95.0, 110.0, set(), nob) == [(0, 1.0, 110.0, "close", "목표")])
c2 = c.copy(); c2[6] = 94.0
o2 = o.copy(); o2[7] = 93.5
check("3-2 종가 < 무효선 → 다음 날 시가에 전부", ex(o2, h, c2, 5, 29, 95.0, 110.0, set(), nob) == [(2, 1.0, 93.5, "open", "무효(돌파선 아래)")])
bb = nob.copy(); bb[7] = True
o3 = o.copy(); o3[8] = 99.0
h3 = h.copy(); h3[10] = 112.0
legs = ex(o3, h3, c, 5, 29, 95.0, 110.0, set(), bb)
check("3-3 추세선 장대음봉 이탈 → 다음 날 시가 절반, 남은 절반은 목표", legs == [(3, 0.5, 99.0, "open", "추세선 장대음봉 이탈"), (5, 0.5, 110.0, "close", "목표")], str(legs))
check("3-4 거래 수익 = 0.5×99 + 0.5×110 ÷ 100 − 1", near(engine.tr_leg_return(legs, 100.0), 0.045))
bb2 = nob.copy(); bb2[7] = bb2[9] = True
check("3-5 절반은 한 번만", [x[1] for x in ex(o3, h, c, 5, 29, 95.0, 200.0, set(), bb2)[:2]] == [0.5, 0.5]
      and ex(o3, h, c, 5, 29, 95.0, 200.0, set(), bb2)[1][4] == "기간 끝")
check("3-6 붕괴 완성일 → 다음 날 시가에 전부", ex(o, h, c, 5, 29, 95.0, 110.0, {8}, nob) == [(4, 1.0, 100.0, "open", "붕괴")])
o4, h4 = o.copy(), h.copy(); o4[7] = 113.0; h4[7] = 115.0
check("3-7 시가가 이미 목표 위 → 시가에", ex(o4, h4, c, 5, 29, 95.0, 110.0, set(), nob) == [(2, 1.0, 113.0, "close", "목표")])
bb3 = nob.copy(); bb3[6] = True
check("3-8 무효와 추세선 이탈이 같은 날 → 전부(무효 우선)", ex(o2, h, c2, 5, 29, 95.0, 110.0, set(), bb3) == [(2, 1.0, 93.5, "open", "무효(돌파선 아래)")])
check("3-9 시험 기간 끝 → 그날 종가", ex(o, h, c, 5, 8, 95.0, 110.0, set(), nob) == [(3, 1.0, 100.0, "close", "기간 끝")])
_mh = CF["MAX_HOLD"]
CF["MAX_HOLD"] = 4
check("3-10 진입일 포함 최대 보유일째 종가", ex(o, h, c, 5, 29, 95.0, 110.0, set(), nob) == [(3, 1.0, 100.0, "close", "최대 보유")])
CF["MAX_HOLD"] = _mh
check("3-11 참고 단순형: 추세선 이탈도 전부", ex(o3, h3, c, 5, 29, 95.0, 110.0, set(), bb, simple=True) == [(3, 1.0, 99.0, "open", "추세선 장대음봉 이탈")])

# ---------------------------------------------------------------- 4. 보유 기간 맞춘 기준
print("4. 보유 기간 맞춘 무작위 기준")
ob = np.linspace(100, 130, 40)
cb = ob * 1.02
rng = np.random.default_rng(0)
lg = [(3, 0.5, 0.0, "open", ""), (6, 0.5, 0.0, "close", "")]
r_all = np.arange(1, 39 - 6 + 1)
hand = np.mean(0.5 * ob[r_all + 3] / ob[r_all] + 0.5 * cb[r_all + 6] / ob[r_all]) - 1 - CF["COST"]
got = np.mean([engine.tr_base(ob, cb, lg, 0, 39, np.random.default_rng(s), k=4000) for s in range(3)])
check("4-1 같은 날 간격·비율(시가 매도는 시가, 나머지는 종가)·비용", near(got, hand, 2e-3), f"{got} vs {hand}")
one = np.zeros(40, dtype=bool); one[9] = True
check("4-2 셋업만: 전날 셋업인 날만 뽑음", near(engine.tr_base(ob, cb, lg, 0, 39, rng, pool_mask=one),
                                       0.5 * ob[13] / ob[10] + 0.5 * cb[16] / ob[10] - 1 - CF["COST"]))
check("4-3 뽑을 날이 없으면 빈칸", np.isnan(engine.tr_base(ob, cb, [(39, 1.0, 0, "close", "")], 0, 39, rng)))


# ---------------------------------------------------------------- 가짜 가격
def walk(seed, n_days, drift=0.0005, vol=0.02, start=50.0, first="1999-01-04"):
    g = np.random.default_rng(seed)
    idx = pd.bdate_range(first, periods=n_days)
    cc = start * np.exp(np.cumsum(g.normal(drift, vol, n_days)))
    oo = cc * np.exp(g.normal(0, 0.005, n_days))
    hh = np.maximum(oo, cc) * np.exp(np.abs(g.normal(0, 0.01, n_days)))
    ll = np.minimum(oo, cc) * np.exp(-np.abs(g.normal(0, 0.01, n_days)))
    return pd.DataFrame({"Open": oo, "High": hh, "Low": ll, "Close": cc, "Volume": g.lognormal(13, 0.9, n_days)}, index=idx)


N = 5600
pm = {f"T{i}": walk(i, N) for i in range(10)}
pm["005930.KS"] = walk(77, N, start=30000.0)
bench = {"XLK": walk(100, N, 0.0003, 0.011), "^IXIC": walk(101, N, 0.0003, 0.011), "^KS11": walk(102, N, 0.0003, 0.011),
         "139260.KS": walk(103, N, 0.0003, 0.011)}
sec = {k: "Technology" for k in pm}
earn = pd.DataFrame({"ticker": ["T0"] * 40, "date": [str(d.date()) for d in pd.bdate_range("2005-01-03", periods=40, freq="63B")]})
prep = engine.lec_prepare(pm, sec, bench, earn, {"T0": "2004-08-23"}, need_rs=True)
grades = {s: engine.lec_market_grade(engine.lec_clean(bench[s])) for s in ("^IXIC", "^KS11")}

# ---------------------------------------------------------------- 5. 미래 자료 미사용
print("5. 미래 자료 미사용(신호 표는 신호일 종가까지 + 다음 날 시가만)")
cut = 3200
pm_c = {"T1": pm["T1"].iloc[:cut]}
bench_c = {k: v.iloc[:cut] for k, v in bench.items()}
prep_c = engine.lec_prepare(pm_c, sec, bench_c, None, None, need_rs=True)
gr_c = {s: engine.lec_market_grade(engine.lec_clean(bench_c[s])) for s in ("^IXIC",)}
full = engine.tr_context("T1", prep["T1"], grades["^IXIC"])["sig"]
part = engine.tr_context("T1", prep_c["T1"], gr_c["^IXIC"])["sig"]
cols = ["t", "rule", "entry", "zl", "target", "width", "box_start", "tops", "absorb", "market", "rs", "ma120", "stop_ok", "box_ok"]
f_ = full[full["t"] + 1 <= cut - 1][cols].reset_index(drop=True)
p_ = part[part["t"] + 1 <= cut - 1][cols].reset_index(drop=True)
check("5-1 자른 자료와 전체 자료의 신호 표가 같음", len(f_) > 0 and f_.equals(p_), f"{len(f_)} vs {len(p_)}")

# ---------------------------------------------------------------- 6. 실행 전 건수 · 탐색(가짜 가격)
print("6. 실행 전 건수·탐색 기간 밖 미사용(무작위 걸음)")
ctx = engine.tr_contexts(prep, grades)
cnt = engine.run_tr_count(prep, grades, ctx=ctx)
mono = all(all(list(cnt["funnel"][r][ch].values())[i] >= list(cnt["funnel"][r][ch].values())[i + 1]
               for i in range(len(cnt["funnel"][r][ch]) - 1)) for r in engine.TR_RULES for ch in ("with", "without"))
check("6-1 깔때기는 단계마다 줄거나 같음(두 사슬)", mono, str(cnt["funnel"]))
check("6-2 신호가 있음(가짜 가격에서도 시험이 돎)", cnt["funnel"]["A"]["without"]["signal"] > 10 and cnt["funnel"]["A"]["without"]["box"] > 0,
      str(cnt["funnel"]["A"]["without"]))
nz = cnt["noise"]
check("6-3 보유가 길수록 흩어짐 큼, 최소 차이 계산", nz[20] < nz[40] < nz[60] and cnt["mde"]["A"]["table"][40]["cluster"] > cnt["mde"]["A"]["table"][40]["indep"])
check("6-4 T1: 300건 미만이면 뺌", cnt["absorb"]["use"] is False and engine.tr_absorb_decision({"with_absorb": 300})["use"] is True
      and engine.tr_absorb_decision({"with_absorb": 299})["use"] is False)
tr = engine.tr_simulate(ctx, prep, "A", CF["EXPLORE_START"], CF["EXPLORE_END"], False)
hi_ok = all(int(r.exit_t) <= engine.tr_bounds(prep[r.ticker]["b"].index, CF["EXPLORE_START"], CF["EXPLORE_END"])[1] for r in tr.itertuples())
check("6-5 탐색 거래는 2013-12-31 뒤 가격을 쓰지 않음", len(tr) > 0 and hi_ok and pd.to_datetime(tr["date"]).max() <= pd.Timestamp(CF["EXPLORE_END"]))
ov = all((g["t"].to_numpy()[1:] > g["exit_t"].to_numpy()[:-1]).all() for _, g in tr.groupby("ticker"))
check("6-6 종목당 1포지션(판 날 다음 날부터 새 신호)", ov)
check("6-7 기준·차이가 채워짐", tr["base"].notna().all() and np.allclose(tr["x"], tr["net"] - tr["base"]))
res = engine.run_tr(prep, grades)
check("6-8 무작위 걸음은 후보 아님 → 확인 단계 안 엶", res["candidates"] == [] and res["verify"] == {} and "안 엶" in res["verify_status"])
rep = engine.render_tr_report(res, "test")
check("6-9 보고서", "## 판정: 탐색 후보 없음 — 미통과" in rep and "깔때기" in rep and "셋업만" in rep and "단계 하나씩 빼기" in rep)
check("6-10 실행 전 보고서", "가려낼 수 있는 최소 차이" in engine.render_tr_count_report(cnt, "test"))
_je = engine.tr_judge_explore
engine.tr_judge_explore = lambda st: {"candidate": True, "reasons": []}   # 확인 단계 길만 점검(후보라고 가정)
res2 = engine.run_tr(prep, grades)
engine.tr_judge_explore = _je
trv = engine.tr_simulate(ctx, prep, "A", CF["VERIFY_START"], None, False, seed=CF["SEED"] + 1)
check("6-11 후보면 2014~만으로 확인(신호가 2014 앞이면 안 됨)", res2["candidates"] == ["A", "B"] and set(res2["verify"]) == {"A", "B"}
      and (not len(trv) or pd.to_datetime(trv["date"]).min() >= pd.Timestamp(CF["VERIFY_START"]))
      and res2["verify"]["A"]["stats"]["n"] == len(trv) and "확인: 후보 2개" in res2["verify_status"], res2["verify_status"])


# ---------------------------------------------------------------- 7. 우위를 심은 / 없는 가짜 거래
print("7. 우위를 심은 가짜 거래는 후보, 우위 없는 가짜 거래는 후보 아님(판정 기계 점검)")


def fake_world(edge: bool, n_tk=60, per=8, seed=5):
    g = np.random.default_rng(seed)
    prepared, ctx_ = {}, {}
    for i in range(n_tk):
        df = walk(1000 + i, 3800, drift=0.0003, vol=0.02, first="2000-01-03")
        sym = f"F{i}" if i % 4 else f"{100000 + i}.KS"
        days = np.sort(g.choice(np.arange(300, 3450, 60), size=per, replace=False))
        hh = df["High"].to_numpy().copy()
        rows = []
        for s in days:
            E = float(df["Open"].iloc[s + 1])
            tgt = E * (1.06 if edge else 9.0)
            if edge:
                hh[s + 2] = max(hh[s + 2], E * 1.07)
            rows.append({"t": int(s), "rule": "A", "entry": E, "zl": E * 0.5, "zu": E * 0.6, "level": E, "target": tgt, "width": 0.1,
                         "box_start": int(s - 30), "tops": 1, "absorb": False, "market": True, "grade": "🟢", "rs": True, "ma120": True,
                         "earn": "ok", "stop_ok": True, "stop_pct": 0.05, "box_ok": True})
        df["High"] = hh
        prepared[sym] = {"b": df}
        ctx_[sym] = {"sig": pd.DataFrame(rows), "setup": np.ones(len(df), dtype=bool), "bb_break": np.zeros(len(df), dtype=bool),
                     "coll": np.zeros((0, 2), dtype=int), "madb": np.array([], dtype=int), "market": engine.lec_market_of(sym)}
    return prepared, ctx_


pw, cw = fake_world(True)
s_e = engine.tr_stats(engine.tr_simulate(cw, pw, "A", CF["EXPLORE_START"], CF["EXPLORE_END"], False))
check("7-1 우위 심음: 후보(차이 ≈ +6%p, t 큼)", engine.tr_judge_explore(s_e)["candidate"] and s_e["diff"] > 0.04 and s_e["t"] > 5,
      f"{s_e['diff']} {s_e['t']} {engine.tr_judge_explore(s_e)['reasons']}")
pn, cn = fake_world(False)
s_n = engine.tr_stats(engine.tr_simulate(cn, pn, "A", CF["EXPLORE_START"], CF["EXPLORE_END"], False))
check("7-2 우위 없음(같은 보유 길이로만 들고 있음): 후보 아님, |t| < 2.5", not engine.tr_judge_explore(s_n)["candidate"] and abs(s_n["t"]) < 2.5,
      f"{s_n['diff']} {s_n['t']}")
check("7-3 우위 없음의 차이는 0 근처(보유 기간을 맞췄으므로 오르는 추세만으로는 이기지 못함)", abs(s_n["diff"]) < 0.03 and s_n["net"] > 0, f"{s_n['diff']} net {s_n['net']}")

# ---------------------------------------------------------------- 8. 판정 기준
print("8. 판정 기준")
ok = {"n": 400, "n_tickers": 60, "net": 0.01, "diff": 0.02, "t": 2.5, "by_market": {"US": {"n": 300, "diff": 0.02, "t": 2}, "KR": {"n": 100, "diff": 0.01, "t": 1}},
      "years": {"drop_year": 2008, "drop_diff": 0.015, "drop_t": 2.1, "years_ok": 8, "years_n": 10, "years_share": 0.8}}
check("8-1 모두 맞으면 후보", engine.tr_judge_explore(ok)["candidate"])
for nm, patch, word in [("거래 수", {"n": 299}, "거래"), ("순수익", {"net": -0.001}, "순수익"), ("t", {"t": 1.99}, "t <"),
                        ("시장 방향", {"by_market": {"US": {"n": 300, "diff": 0.02, "t": 2}, "KR": {"n": 40, "diff": -0.01, "t": -1}}}, "방향"),
                        ("해 빼기", {"years": {**ok["years"], "drop_t": 1.5}}, "해"), ("해별 방향", {"years": {**ok["years"], "years_share": 0.6}}, "해별")]:
    j = engine.tr_judge_explore({**ok, **patch})
    check(f"8-{nm}", not j["candidate"] and any(word in x for x in j["reasons"]), str(j["reasons"]))
check("8-확인 t ≥ 2 · 100건", engine.tr_judge_verify({**ok, "n": 100})["passed"] and not engine.tr_judge_verify({**ok, "n": 99})["passed"])
check("8-최종 t ≥ 3", not engine.tr_judge_final(ok)["passed"] and engine.tr_judge_final({**ok, "t": 3.1})["passed"])

print(f"\n{'통과' if not FAILS else '실패'}: {len(FAILS)}개 실패", FAILS)
sys.exit(1 if FAILS else 0)
