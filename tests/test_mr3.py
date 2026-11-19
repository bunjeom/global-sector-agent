"""H-MR3 변동성 국면 이격도·지지 반등 (engine/mr3_backtest.py) — VIX 날짜 맞춤(앞선 날만), 청산 손 계산, 규칙 = MR2 신호 + 국면,
보유 기간 맞춘 비교 기준, 1포지션, 섞기 통과선의 날 공유, 해 빼기 점검, 우위 없는 가짜 가격에서 후보 없음, 심은 반등은 검증까지."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _offline  # noqa: F401
import json
import sys
import numpy as np
import pandas as pd
import engine

FAILS = []


def chk(n, c, extra=None):
    print(("PASS " if c else "FAIL ") + n + (f"  [{extra}]" if (not c and extra is not None) else ""))
    if not c:
        FAILS.append(n)


C3 = engine.MR3_CONFIG


def path_walk(n, seed, sd=0.022, steps=20, start="1999-01-04"):
    """장중 경로까지 무작위 걸음인 가짜 가격."""
    r = np.random.default_rng(seed)
    inc = r.normal(0, sd / np.sqrt(steps), (n, steps))
    gap = r.normal(0, sd * 0.3, n)
    lp = np.log(50.0)
    O, H, L, Cc = (np.empty(n) for _ in range(4))
    for i in range(n):
        s0 = lp + gap[i]
        p = s0 + np.cumsum(inc[i])
        O[i], H[i], L[i], Cc[i] = np.exp(s0), np.exp(max(s0, p.max())), np.exp(min(s0, p.min())), np.exp(p[-1])
        lp = p[-1]
    return pd.DataFrame({"Open": O, "High": H, "Low": L, "Close": Cc, "Volume": r.lognormal(13, 0.5, n)},
                        index=pd.bdate_range(start, periods=n))


def fake_vix(start="1989-01-02", n=10200, seed=5):
    r = np.random.default_rng(seed)
    x = np.zeros(n)
    x[0] = np.log(18)
    for i in range(1, n):
        x[i] = x[i - 1] + 0.02 * (np.log(18) - x[i - 1]) + r.normal(0, 0.06)
    return pd.Series(np.exp(x), index=pd.bdate_range(start, periods=n))


# 1) VIX 날짜 맞춤: 그 날짜보다 앞선 가장 최근 값만, 오래되면 없음
vx = pd.Series([15.0, 25.0, 30.0], index=pd.to_datetime(["2020-03-02", "2020-03-03", "2020-03-04"]))
vp = engine.mr3_vix_prev(pd.to_datetime(["2020-03-02", "2020-03-03", "2020-03-05", "2020-03-20"]), vx)
chk("1 같은 날 VIX는 쓰지 않고 앞선 날 값(한국 날짜도 같은 규칙)", np.isnan(vp[0]) and vp[1] == 15.0 and vp[2] == 30.0, vp)
chk("1-2 가장 최근 VIX가 10일(달력)보다 오래되면 자료 없음", np.isnan(vp[3]))

# 2) 청산 손 계산
def ex(rows, stop=90.0, fill=False):
    a = np.array(rows, dtype=float)              # 줄 = (시가, 고가, 저가, 종가, 25일선)
    pad = np.tile(a[-1], (12, 1))
    a = np.vstack([a, pad])
    g, k, h = engine.mr3_exit(a[:, 0], a[:, 1], a[:, 2], a[:, 3], a[:, 4], 0, 100.0, stop, fill_day=fill)
    return round(g, 6), k, h


e0 = [100, 101, 99, 100, 110]
chk("2 둘째 날 종가 ≥ 25일선 → 그날 종가(회복)", ex([e0, [95, 97, 91, 94, 98], [95, 100, 94, 99, 98]]) == (-0.01, 1, 2))
chk("2-2 시가 ≤ 손절가 → 시가", ex([e0, [88, 89, 85, 86, 98]]) == (-0.12, 5, 1))
chk("2-3 저가 ≤ 손절가 → 손절가", ex([e0, [95, 96, 89, 95, 98]]) == (-0.10, 2, 1))
chk("2-4 같은 날 손절·회복 둘 다 → 손절", ex([e0, [95, 101, 89, 100, 98]]) == (-0.10, 2, 1))
rows10 = [e0] + [[100, 101, 95, 100 + 0.1 * i, 120] for i in range(1, 11)]
chk("2-5 10거래일 동안 아무 일 없음 → 10번째 날 종가", ex(rows10) == (0.01, 3, 10))
chk("2-6 지정가 체결일: 저가 ≤ 손절가면 그날 손절(보유 0일)", ex([[100, 101, 89, 95, 110], e0], fill=True) == (-0.10, 6, 0))
chk("2-7 지정가 체결일: 종가 ≥ 25일선이면 그날 종가", ex([[100, 106, 95, 105, 104], e0], fill=True) == (0.05, 7, 0))

# 3) 종목 하나: 규칙 = MR2 신호 + 국면 + 대상, 진입가·손절·청산 다시 계산, 비교 기준 손 계산
vix = fake_vix()
w = path_walk(5600, 21, sd=0.03)
b = engine.lec_base(w)
sw = engine.lec_swings(b)
n = len(b)
el = np.ones(n, dtype=bool)
el[::7] = False
vv = engine.mr3_vix_prev(b.index, vix)
lo_i, hi_i = 600, n - 20
res = engine.mr3_ticker(b, sw, el, vv, lo_i, hi_i, engine.MR3_RULES)
cache = {}
sig, px = engine.mr2_signal_arrays(b, sw, {"S10", "S12"}, cache=cache)
pb = np.r_[np.nan, cache["base"][:-1]]
disp = b["DISP25"].to_numpy()
o, h, l, c = (b[k].to_numpy() for k in ("Open", "High", "Low", "Close"))
ma = b["SMA25"].to_numpy()
patr = np.r_[np.nan, b["ATR14"].to_numpy()[:-1]]
cr = vv >= 20
pel = np.r_[False, el[:-1]]
ok3 = True
cnt = {}
for r_, d in res["rules"].items():
    vid, mode = r_.split("-")
    cnt[r_] = len(d["t"])
    for t, E, g, k, hd in zip(d["t"], d["entry"], d["gross"], d["kind"], d["hold"]):
        ok3 &= bool(cr[t]) and lo_i <= t <= hi_i
        if mode == "C":
            base_sig = {"V1": sig["S10"][t], "V3": sig["S12"][t], "V5": disp[t] <= pb[t] and sig["S12"][t]}[vid]
            ok3 &= bool(base_sig) and bool(el[t]) and abs(E - c[t]) < 1e-12
            gg = engine.mr3_exit(o, h, l, c, ma, t, E, E - 2 * patr[t])
        else:
            P = px["S10" if vid == "V2" else "S12"][t]
            ok3 &= bool(pel[t]) and l[t] <= P and abs(E - min(o[t], P)) < 1e-12
            gg = engine.mr3_exit(o, h, l, c, ma, t, E, E - 2 * patr[t], fill_day=True)
        ok3 &= abs(gg[0] - g) < 1e-12 and gg[1:] == (k, hd)
    tt, hh = d["t"], d["hold"]
    ok3 &= bool(np.all(tt[1:] > tt[:-1] + hh[:-1]))
chk(f"3 거래 = MR2 신호(S10·S12) + 변동성 국면 + 대상, 진입가·손절·청산 다시 계산, 1포지션 ({cnt})", ok3 and all(v > 0 for v in cnt.values()), cnt)
J = res["pool_t"]
expJ = np.flatnonzero((np.arange(n) >= lo_i) & (np.arange(n) <= hi_i) & el & cr & (np.arange(n) + 10 <= n - 1))
chk("3-2 비교 기준 날 = 기간 안 · 대상 · 변동성 국면 · 10일 뒤 자료 있음", np.array_equal(J, expJ) and len(J) > 100)
R = res["pool_R"]
chk("3-3 비교 기준 = j일 종가 → h거래일 뒤 종가 − 비용(h = 1~10)",
    np.allclose(R[:, 4], c[J + 5] / c[J] - 1 - 0.003) and np.allclose(R[:, 9], c[J + 10] / c[J] - 1 - 0.003))
res0 = engine.mr3_ticker(b, sw, el, np.full(n, 15.0), lo_i, hi_i, engine.MR3_RULES)
chk("3-4 VIX가 늘 20 미만이면 거래도 비교 기준도 없음(단타 부적합)", all(len(d["t"]) == 0 for d in res0["rules"].values()) and len(res0["pool_t"]) == 0)
resN = engine.mr3_ticker(b, sw, el, np.full(n, np.nan), lo_i, hi_i, engine.MR3_RULES)
chk("3-5 VIX 자료가 없으면 국면 아님", all(len(d["t"]) == 0 for d in resN["rules"].values()))

# 4) 단계: x = 순수익 − 그 종목의 같은 보유일 평균(체결일 청산은 1일)
prep = {}
for i in range(24):
    sym = f"U{i}" if i < 13 else f"{i:06d}.KS"
    bi = engine.lec_base(path_walk(5000, 300 + i, sd=0.02 + 0.002 * (i % 6)))
    prep[sym] = {"b": bi, "sw": engine.lec_swings(bi)}
elig = {s: np.ones(len(p["b"]), dtype=bool) for s, p in prep.items()}
stg = engine.mr3_stage(prep, elig, vix, engine.MR3_RULES, "2001-01-01", "2014-12-31")
tr = stg["tr"]
s0 = tr["ticker"].iloc[0]
g0 = tr[tr["ticker"] == s0]
mu0 = stg["mu"][s0]
chk("4 x = 순수익 − 같은 종목 같은 보유일 평균, 보유 0일은 1일로", np.allclose(g0["x"], g0["net"] - mu0[np.maximum(g0["hold"], 1) - 1])
    and (tr["hb"] >= 1).all() and len(tr) > 300, len(tr))
chk("4-2 종목 평균을 뺀 비교 기준의 평균 = 0", all(np.allclose(v["R"].mean(axis=0), 0) for v in stg["pool"].values()))
chk("4-3 거래의 VIX(앞선 날) ≥ 20, 신호일이 기간 안", bool((tr["vix"] >= 20).all()) and bool(((tr["date"] >= "2001-01-01") & (tr["date"] <= "2014-12-31")).all()))
chk("4-4 변동성 국면 비율이 0~1 사이로 셈", stg["crisis_share"] is not None and 0.2 < stg["crisis_share"] < 0.8, stg["crisis_share"])

# 5) 섞기 통과선: 같은 종목·신호일은 모든 규칙에서 같은 무작위 날
fake = {"tr": pd.concat([tr[tr["rule"] == "V3-C"], tr[tr["rule"] == "V3-C"].assign(rule="COPY")], ignore_index=True), "pool": stg["pool"]}
saved = C3["RC_REPS"]
C3["RC_REPS"] = 25
try:
    a1 = engine.mr3_reality(fake, ["V3-C"], 7)
    a2 = engine.mr3_reality(fake, ["V3-C", "COPY"], 7)
finally:
    C3["RC_REPS"] = saved
chk("5 똑같은 규칙을 하나 더 넣어도 통과선이 그대로", np.allclose(a1["max_t"], a2["max_t"]) and a1["threshold"] == a2["threshold"])
chk("5-2 무작위 날로 바꾼 값의 평균 ≈ 0", abs(a1["null_mean"]["V3-C"]) < 0.003, a1["null_mean"])

# 6) 해 빼기 · 해별 방향
dts = pd.to_datetime(["2001-03-01"] * 40 + ["2002-03-01"] * 40 + ["2003-03-01"] * 40)
xx = np.r_[np.full(40, 0.01), np.full(40, -0.005), np.full(40, 0.002)] + np.random.default_rng(1).normal(0, 0.001, 120)
yc = engine.mr3_year_checks(xx, dts)
chk("6 기여가 가장 큰 해(2001)를 빼고 다시 잼 · 해별 방향 2/3", yc["drop_year"] == 2001 and yc["drop_diff"] < 0 and yc["years_ok"] == 2 and yc["years_n"] == 3, yc)

# 7) 판정: 심은 차이 → 후보, 비용 뒤 손실이면 후보 아님, 지정가는 엄격 체결
t3 = tr[tr["rule"] == "V3-C"].copy()
sv = {k: C3[k] for k in ("MIN_TRADES", "MIN_TICKERS", "MIN_MARKET", "YEAR_MIN_TRADES")}
C3.update({"MIN_TRADES": 50, "MIN_TICKERS": 5, "MIN_MARKET": 10, "YEAR_MIN_TRADES": 5})
try:
    s_plus = engine.mr3_stats(t3.assign(x=t3["x"] + 0.03, net=t3["net"] + 0.03))
    j1 = engine.mr3_judge_explore(s_plus, 2.5)
    chk("7 거래마다 +3%p를 심으면 찾기 후보", j1["candidate"], j1["reasons"])
    j2 = engine.mr3_judge_explore(engine.mr3_stats(t3.assign(x=t3["x"] + 0.03, net=t3["net"] * 0 - 0.001)), 2.5)
    chk("7-2 기준보다 나아도 비용 뒤 순수익 ≤ 0이면 후보 아님", not j2["candidate"] and any("순수익" in x for x in j2["reasons"]), j2["reasons"])
    j3 = engine.mr3_judge_explore(s_plus, 2.5, strict=engine.mr3_stats(t3))
    chk("7-3 지정가: 엄격 체결이 기준 미달이면 후보 아님", not j3["candidate"] and any("엄격" in x for x in j3["reasons"]))
    jv = engine.mr3_judge_verify(s_plus, {"1990~2000": {"n": 40, "diff": -0.01}, "2020~": {"n": 40, "diff": 0.02}})
    chk("7-4 검증: 30건 이상인 구간 하나가 반대 방향이면 미통과", not jv["passed"] and any("1990~2000" in x for x in jv["reasons"]))
    chk("7-5 검출 가능한 최소 차이 = (z + 0.84) × 표준오차", abs(engine.mr3_mde(s_plus, 2.0) - 2.84 * s_plus["se"]) < 1e-12)
finally:
    C3.update(sv)

# 8) 전체 실행: 우위 없는 가짜 가격 → 후보 없음, 1990년대 자료는 부르지 않음
called = {"n": 0}


def no_loader():
    called["n"] += 1
    return None


prep8 = {}
for i in range(26):
    sym = f"V{i}" if i < 14 else f"{i + 100:06d}.KS"
    bi = engine.lec_base(path_walk(7200, 700 + i, sd=0.02 + 0.002 * (i % 6)))
    prep8[sym] = {"b": bi, "sw": engine.lec_swings(bi)}
sv = {k: C3[k] for k in ("MIN_TRADES", "MIN_TICKERS", "MIN_MARKET", "RC_REPS", "YEAR_MIN_TRADES")}
C3.update({"MIN_TRADES": 50, "MIN_TICKERS": 6, "MIN_MARKET": 10, "RC_REPS": 30, "YEAR_MIN_TRADES": 5})
try:
    r8 = engine.run_mr3(prep8, vix, old_loader=no_loader)
    chk("8 우위 없는 가짜 가격: 후보 없음", r8["candidates"] == [], [(k, r8["rules"][k]["stats"]["t"]) for k in r8["candidates"]])
    chk("8-2 후보가 없으면 1990년대 자료를 부르지도 않음", called["n"] == 0 and r8["verify"] == {})
    rep = engine.render_mr3_report(r8, "test")
    chk("8-3 보고서: 판정표 5줄 · 통과선 · 시점 효과 · 최소 차이", all(engine.MR3_LABELS[k] in rep for k in engine.MR3_RULES)
        and "통과선 t" in rep and "시점 효과" in rep and "검출 가능한 최소 차이" in rep)
    json.dumps(r8, ensure_ascii=False, default=str)
    chk("8-4 JSON 저장 가능 · 기록 5줄", len(engine.mr3_log_lines(r8, "t")) == 5)
finally:
    C3.update(sv)

# 9) 검정력: 변동성 국면의 급락 다음 날 +4% 갭을 심으면 → 찾기 후보 → 1990~2000 + 2020~ 검증 통과
vix_arr = vix


def rebound(df):
    d = df.copy()
    cl = d["Close"].to_numpy()
    vprev = engine.mr3_vix_prev(d.index, vix_arr)
    mult = np.ones(len(d))
    acc = 1.0
    for i in range(1, len(d)):
        if cl[i] / cl[i - 1] <= 0.96 and vprev[i] >= 20:
            acc *= 1.04
        mult[i] = acc
    mult = np.r_[1.0, mult[:-1]]
    for k in ("Open", "High", "Low", "Close"):
        d[k] = d[k].to_numpy() * mult
    return d


prep9 = {}
for i in range(26):
    sym = f"W{i}" if i < 14 else f"{i + 200:06d}.KS"
    bi = engine.lec_base(rebound(path_walk(7200, 900 + i, sd=0.026 + 0.002 * (i % 5))))
    prep9[sym] = {"b": bi, "sw": engine.lec_swings(bi)}


def old9():
    pm = {f"W{i}": rebound(path_walk(3300, 1900 + i, sd=0.026 + 0.002 * (i % 5), start="1989-01-02")) for i in range(14)}
    return engine.mr3_prepare_old(pm)[0]


sv = {k: C3[k] for k in ("MIN_TRADES", "MIN_TICKERS", "MIN_MARKET", "RC_REPS", "YEAR_MIN_TRADES", "VERIFY_MIN_TRADES", "VERIFY_PART_MIN")}
C3.update({"MIN_TRADES": 50, "MIN_TICKERS": 6, "MIN_MARKET": 10, "RC_REPS": 30, "YEAR_MIN_TRADES": 5, "VERIFY_MIN_TRADES": 20, "VERIFY_PART_MIN": 10})
try:
    r9 = engine.run_mr3(prep9, vix, old_loader=old9)
    cz = r9["candidates"]
    chk("9 심은 반등: 찾기 후보가 나옴(종가 진입 규칙 중)", any(k.endswith("-C") for k in cz),
        [(k, r9["rules"][k]["stats"]["t"], r9["rules"][k]["judge"]["reasons"][:2]) for k in engine.MR3_RULES])
    v1 = r9["verify"].get(cz[0], {}) if cz else {}
    chk("9-2 검증은 후보만, 1990~2000(미국만)과 2020~ 두 구간을 합쳐서", r9["verify_status"] == "검증함" and set(r9["verify"]) == set(cz)
        and v1.get("parts", {}).get("1990~2000", {}).get("n", 0) > 0 and v1["parts"]["2020~"]["n"] > 0
        and v1["stats"]["n"] == v1["parts"]["1990~2000"]["n"] + v1["parts"]["2020~"]["n"])
    chk("9-3 심은 반등이 검증도 통과 → 최종 목록(검증 t 순)", len(r9["verified"]) >= 1 and set(r9["verified"]) <= set(cz),
        {k: v["judge"] for k, v in r9["verify"].items()})
    chk("9-4 보고서에 검증 표", "검증 통과" in engine.render_mr3_report(r9, "t"))
    rf = engine.run_mr3_final(prep9, vix, ["V1-C", "V2-L"])
    chk("9-5 최종 실행(지정가는 엄격 체결 포함) · 보고서", "strict" in rf["rules"]["V2-L"] and rf["rules"]["V2-L"]["strict"] is not None
        and "최종" in engine.render_mr3_final_report(rf, [], "t"))
finally:
    C3.update(sv)
pp, dropped = engine.mr3_prepare_old({"A": path_walk(3300, 1, start="1989-01-02"), "B.KS": path_walk(3300, 2, start="1989-01-02"),
                                      "C": path_walk(3000, 3, start="2003-01-01")})
chk("9-6 1990년대 재료: 미국만, 2001-06-30에서 자르고 1990~2000 자료 없는 종목은 뺌",
    list(pp) == ["A"] and "C" in dropped and pp["A"]["b"].index.max() <= pd.Timestamp("2001-06-30") and "DISP25" in pp["A"]["b"])

print(f"\n{'모두 통과' if not FAILS else str(len(FAILS)) + '개 실패: ' + ', '.join(FAILS)}")
sys.exit(1 if FAILS else 0)
