"""강의 조건 시험 2단계 L2 (engine/lecture_events.py · lecture_family.py) — 손 계산·미래 자료 미사용·통과선 방식 점검."""
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


LC = engine.LECTURE_CONFIG
seg = lambda a, b_, k: np.linspace(a, b_, k)[1:]


def frame(cl, vol=None, start="1999-01-04", o=None):
    cl = np.asarray(cl, dtype=float)
    n = len(cl)
    o = np.r_[cl[0], cl[:-1]] if o is None else np.asarray(o, dtype=float)
    v = np.full(n, 1e6) if vol is None else np.asarray(vol, dtype=float)
    return pd.DataFrame({"Open": o, "High": np.maximum(o, cl) + 0.01, "Low": np.minimum(o, cl) - 0.01, "Close": cl, "Volume": v},
                        index=pd.bdate_range(start, periods=n))


def walk(n, seed, drift=0.0003, sd=0.02, start="1999-01-04"):
    r = np.random.default_rng(seed)
    cl = 50 * np.exp(np.cumsum(drift + r.normal(0, sd, n)))
    o = np.r_[cl[0], cl[:-1]] * np.exp(r.normal(0, 0.004, n))
    hi = np.maximum(o, cl) * np.exp(np.abs(r.normal(0, 0.01, n)))
    lo = np.minimum(o, cl) * np.exp(-np.abs(r.normal(0, 0.01, n)))
    return pd.DataFrame({"Open": o, "High": hi, "Low": lo, "Close": cl, "Volume": r.lognormal(13, 0.4, n)},
                        index=pd.bdate_range(start, periods=n))


# 1) 조합 수와 짝짓기 규칙
cb = engine.lec_l2_combos()
ids = [x["id"] for x in cb]
chk("1 조합 수 = 256 (돌파 117 + 바닥 96 + 눌림 36 + 단독 7)", len(cb) == 256, len(cb))
chk("1-2 조합 이름이 겹치지 않음", len(set(ids)) == len(ids))
chk("1-3 C02는 E02에만", all(x["event"] == "E02" for x in cb if x["conf"] == "C02") and any(x["conf"] == "C02" for x in cb))
chk("1-4 단독 사건은 환경·확인 없이 한 번씩",
    all(sum(1 for x in cb if x["event"] == e) == 1 for e in ("E06", "E10", "E11", "E12", "E13", "E14", "E15")))
chk("1-5 바닥 사건에 F04(상승 상태)는 없음(짝짓기 표)", not any(x["event"] == "E05" and x["env"] == "F04" for x in cb))

# 2) 돌파 손 계산: 저점 90 → 고점 110(20번째 봉) → 저점 100 → 109까지 오른 뒤 40번째 봉에 거래량 3배 양봉 115로 돌파
path = np.r_[[100.0], seg(100, 90, 11), seg(90, 110, 11), seg(110, 100, 11), seg(100, 109, 10), [115.0],
             seg(115, 120, 6), seg(120, 113, 6), seg(113, 125, 11), seg(125, 111, 8), seg(111, 120, 6), np.full(30, 120.0)]
vol = np.full(len(path), 1e6)
vol[40] = 3e6
b2 = engine.lec_base(frame(path, vol))
sw2 = engine.lec_swings(b2)
bo = engine.lec_breakouts(b2, sw2)
first = bo[0] if bo else {}
chk("2 돌파일 = 40번째 봉(종가가 확정 고점 110을 처음 넘음)", first.get("t0") == 40 and abs(first.get("level", 0) - 110.01) < 1e-9, first)
chk("2-2 돌파봉 조건 ①(양봉 + RVOL ≥ 1.5) 충족", first.get("c1") is True)
cf = first.get("confirm")
nl = np.flatnonzero(sw2["l_o"] > 40)
chk("2-3 돌파 확정일은 돌파 뒤 새 확정 저점(51번째 봉, 5봉 뒤 확정) 이후", cf is not None and nl.size and cf >= sw2["l_o"][nl[0]] + 5, (cf, sw2["l_o"]))
if cf is not None:
    g20, g60 = b2["SMA20_SLOPE"].to_numpy(), b2["SMA60_SLOPE"].to_numpy()
    exp_cf = next(d for d in range(int(sw2["l_o"][nl[0]]) + 5, 61) if g20[d] > 0 and g60[d] > 0)
    chk("2-4 확정일 = 새 저점 확정 뒤 20·60일선 기울기가 모두 양수인 첫날", cf == exp_cf, (cf, exp_cf))
chk("2-5 가짜 아님", first.get("fake") is None)
zl, zu = first.get("zl", 0), first.get("zu", 0)
lo2, cl2 = b2["Low"].to_numpy(), b2["Close"].to_numpy()
exp_flip = next((d for d in range(cf + 1, cf + 21) if lo2[d] <= zu and cl2[d] >= zl), None) if cf is not None else None
chk("2-6 SR Flip = 확정 뒤 저가가 돌파선 구간 상단 이하이고 종가 ≥ 구간 하단인 첫날", first.get("flip") == exp_flip and exp_flip is not None,
    (first.get("flip"), exp_flip))
fake_path = np.r_[path[:41], [112.0, 108.0, 104.0], np.full(40, 104.0)]
bf = engine.lec_base(frame(fake_path, vol[:len(fake_path)]))
bof = engine.lec_breakouts(bf, engine.lec_swings(bf))
chk("2-7 돌파 뒤 종가가 구간 하단 아래 = 가짜(그날), 확정 없음",
    bof and bof[0]["fake"] == next(d for d in range(40, 60) if bf["Close"].iloc[d] < bof[0]["zl"]) and bof[0]["confirm"] is None, bof[:1])

# 3) 쌍바닥 손 계산: 하락 구조 → L2 110(50) → 넥 125(60) → L1 112(70) → 넥 돌파
dp = np.r_[[150.0], seg(150, 130, 11), seg(130, 140, 11), seg(140, 120, 11), seg(120, 135, 11), seg(135, 110, 11),
           seg(110, 125, 11), seg(125, 112, 11), seg(112, 135, 13), np.full(30, 135.0)]
b3 = engine.lec_base(frame(dp, o=dp))     # 시가 = 종가: 꼭짓점 고가가 둘로 같아지지 않게
sw3 = engine.lec_swings(b3)
st3 = engine.lec_state(b3, sw3)
chk("3 L2(50번째 봉) 날 상태 = 하락", st3[50] == -1, st3[45:55])
dbs = [x for x in engine.lec_double_bottoms(b3, sw3, st3) if x["l1_origin"] == 70]
d0 = dbs[0] if dbs else {}
chk("3-2 쌍바닥 후보: L2 = 110(50), 넥 = 125(60), L1 = 112(70), 후보일 = 75",
    d0.get("l2_origin") == 50 and d0.get("neck_origin") == 60 and d0.get("tc") == 75 and d0.get("cand") is True, d0)
if d0:
    nzu = engine.lec_zone(d0["neck"], b3["ATR14"].iloc[75])[1]
    exp5 = next(d for d in range(75, len(dp)) if b3["Close"].iloc[d] > nzu)
    chk("3-3 쌍바닥 확정 = 종가가 넥 구간 상단을 처음 넘은 날", d0.get("confirm") == exp5, (d0.get("confirm"), exp5))
    chk("3-4 공격(E07)은 확정일 이후 5일선 기울기 > 0인 첫날", d0.get("e07") is not None and d0["e07"] >= d0["confirm"]
        and b3["SMA5_SLOPE"].iloc[d0["e07"]] > 0)
up_only = np.r_[[100.0], seg(100, 90, 11), seg(90, 110, 11), seg(110, 95, 11), seg(95, 120, 11), seg(120, 100, 11), seg(100, 130, 11)]
bu = engine.lec_base(frame(up_only, o=up_only))
swu = engine.lec_swings(bu)
chk("3-5 앞선 하락 구조가 없으면 쌍바닥 아님(Q55)", not engine.lec_double_bottoms(bu, swu, engine.lec_state(bu, swu)))

# 4) 상태 함수: 새 인자를 줘도 상태 값은 그대로
wk = walk(1500, 7)
bw = engine.lec_base(wk)
sww = engine.lec_swings(bw)
s_a = engine.lec_state(bw, sww)
s_b, upf, dnf = engine.lec_state(bw, sww, return_flags=True)
chk("4 lec_state(return_flags) — 상태 값은 그대로", np.array_equal(s_a, s_b) and len(upf) == len(s_a))
chk("4-2 상승 상태인 날은 모두 '돌파 이력' 있음", bool(np.all(upf[s_a == 1])))

# 5) 전환 증거 손 계산: MACD 히스토그램 음수 + 3일 연속 상승
tr = engine.lec_transition(bw, sww, upf)
mh = bw["MACD_HIST"].to_numpy()
t3 = next(t for t in range(3, len(mh)) if mh[t] < 0 and mh[t] > mh[t - 1] > mh[t - 2] > mh[t - 3])
chk("5 증거 ③ = 히스토그램 < 0 + 3일 연속 상승", tr["ev3"][t3] and not tr["ev3"][:t3].any())
chk("5-2 성격 수: ①·②는 하나로 셈", int(tr["cats"].max()) <= 3 and int(tr["cnt"].max()) <= 4)

# 6) 겹침 지지: 체류 칸 + 확정 스윙이 같은 자리면 2종류 이상
flat = np.r_[np.full(120, 100.0), seg(100, 120, 21), seg(120, 100, 21), np.full(60, 100.0)]
bfl = engine.lec_base(frame(flat))
swf = engine.lec_swings(bfl)
tt = len(flat) - 1
types = engine.lec_support_types(bfl, swf, tt, 100.0)
chk("6 오래 머문 100 자리 = 체류(dwell) 기준 포함", "dwell" in types, types)
chk("6-2 110.5 자리(오르내릴 때 한 번씩 지나감)는 체류 아님", "dwell" not in engine.lec_support_types(bfl, swf, tt, 110.5))

# 7) 미래 자료 미사용: 자른 자료로 계산해도 자른 날까지의 사건·조건 값이 같다
pm = {"AAA": walk(2600, 11), "005930.KS": walk(2600, 12)}
sec = {"AAA": "Technology", "005930.KS": "Technology"}
bench = {"XLK": walk(2600, 21), "139260.KS": walk(2600, 22), "^IXIC": walk(2600, 23), "^KS11": walk(2600, 24)}
K = 2200
cut = lambda d: {k: v.iloc[:K + 1] for k, v in d.items()}
full_ok = True
for pmx, bx, tag in ((pm, bench, "full"), (cut(pm), cut(bench), "cut")):
    prep = engine.lec_prepare(pmx, sec, bx, None, {})
    grades = {s: engine.lec_market_grade(engine.lec_clean(bx[s])) for s in ("^IXIC", "^KS11")}
    etfb = engine.lec_etf_context(bx)
    out = {}
    for sym, p in prep.items():
        etf = engine.lec_sector_etf(sym, "Technology")
        out[sym] = engine.lec_l2_events(p, etfb.get(etf), grades[engine.lec_index_of(sym, "Technology")])
    if tag == "full":
        FULL = out
    else:
        CUT = out
cols = ["t", "event"] + list(engine.L2_ENVS) + list(engine.L2_CONFS)
for sym in pm:
    a = FULL[sym][FULL[sym]["t"] <= K][cols].sort_values(["event", "t"]).reset_index(drop=True)
    b_ = CUT[sym][cols].sort_values(["event", "t"]).reset_index(drop=True)
    same = a.shape == b_.shape and bool(((a == b_) | (a.isna() & b_.isna())).all().all())
    full_ok &= same
    chk(f"7 미래 자료 미사용 — {sym}: 사건·환경·확인 값이 자른 자료와 같음 ({len(a)}건)", same, (a.shape, b_.shape))
chk("7-2 사건 종류가 여럿 나옴(점검이 빈 표가 아님)", FULL["AAA"]["event"].nunique() >= 8, FULL["AAA"]["event"].value_counts().to_dict())

# 8) 월 묶음 t = cluster_stats와 같은 식
r = np.random.default_rng(3)
x = r.normal(0.001, 0.05, 400)
dts = pd.Series(pd.bdate_range("2003-01-01", periods=400))
cs = engine.cluster_stats(pd.Series(x), dts)
ct = engine.lec_cluster_t(x, engine.lec_month_code(dts))
chk("8 lec_cluster_t 평균·표준오차 = cluster_stats", abs(ct["mean"] - cs["mean"]) < 1e-12 and abs(ct["se"] - cs["se"]) < 1e-12)

# 9) 통과선 방식: 효과 없는 가짜 가격 → 후보 없음 / 사건 뒤에 수익을 심으면 → 그 조합이 후보
saved = {k: LC[k] for k in ("MIN_EVENTS", "MIN_TICKERS", "RC_REPS", "MIN_MARKET_EVENTS")}
LC.update({"MIN_EVENTS": 60, "MIN_TICKERS": 8, "RC_REPS": 40, "MIN_MARKET_EVENTS": 20})
try:
    pmn = {(f"T{i}" if i % 3 else f"{i:06d}.KS"): walk(4000, 100 + i) for i in range(18)}
    secn = {k: "Technology" for k in pmn}
    benn = {"XLK": walk(4000, 201), "139260.KS": walk(4000, 202), "^IXIC": walk(4000, 203), "^KS11": walk(4000, 204)}
    prn = engine.lec_prepare(pmn, secn, benn, None, {})
    grn = {s: engine.lec_market_grade(engine.lec_clean(benn[s])) for s in ("^IXIC", "^KS11")}
    tab = engine.lec_l2_table(prn, benn, grn)
    res = engine.run_lecture_l2(tab)
    chk("9 효과 없는 가짜 가격: 탐색 후보 없음", res["candidates"] == [] and res["threshold"] is not None,
        (res["candidates"], res["threshold"]))
    chk("9-2 통과선은 가족이 클수록 2보다 큼(여러 조합 중 최고 t)", res["threshold"] > 2.0, res["threshold"])
    chk("9-3 후보가 없으면 검증 기간을 열지 않음", res["verify"] == {})
    ev = tab["events"].copy()
    m = (ev["event"] == "E01")
    ev.loc[m, "x10"] = ev.loc[m, "x10"] + 0.04
    res2 = engine.run_lecture_l2({"events": ev, "xser": tab["xser"]})
    chk("9-4 E01 사건 뒤에 +4%를 심으면 E01 조합이 후보", "E01" in res2["candidates"], res2["candidates"][:5])
    chk("9-5 심지 않은 사건(E03 단독)은 후보 아님", "E03" not in res2["candidates"])
    chk("9-6 후보는 검증(2014~)을 계산함", set(res2["verify"]) == set(res2["candidates"]))
    rep = engine.render_lecture_l2_report(res2, "test")
    chk("9-7 보고서에 통과선·조합 수·판정", "통과선 t" in rep and "256개" in rep and "## 판정" in rep)
finally:
    LC.update(saved)

print(f"\n{'모두 통과' if not FAILS else str(len(FAILS)) + '개 실패: ' + ', '.join(FAILS)}")
sys.exit(1 if FAILS else 0)
