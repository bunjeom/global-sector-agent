"""강의 조건 시험 2단계 L4 매도·약화 · L5 레벨 (engine/lecture_levels.py) — 손 계산·미래 자료 미사용·섞기 통과선 점검."""
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


def frame(cl, vol=None, start="1999-01-04", o=None, hi=None, lo=None):
    cl = np.asarray(cl, dtype=float)
    n = len(cl)
    o = cl.copy() if o is None else np.asarray(o, dtype=float)
    v = np.full(n, 1e6) if vol is None else np.asarray(vol, dtype=float)
    hi = np.maximum(o, cl) + 0.01 if hi is None else np.asarray(hi, dtype=float)
    lo = np.minimum(o, cl) - 0.01 if lo is None else np.asarray(lo, dtype=float)
    return pd.DataFrame({"Open": o, "High": hi, "Low": lo, "Close": cl, "Volume": v}, index=pd.bdate_range(start, periods=n))


def walk(n, seed, drift=0.0003, sd=0.02, start="1999-01-04"):
    r = np.random.default_rng(seed)
    cl = 50 * np.exp(np.cumsum(drift + r.normal(0, sd, n)))
    o = np.r_[cl[0], cl[:-1]] * np.exp(r.normal(0, 0.004, n))
    hi = np.maximum(o, cl) * np.exp(np.abs(r.normal(0, 0.01, n)))
    lo = np.minimum(o, cl) * np.exp(-np.abs(r.normal(0, 0.01, n)))
    return pd.DataFrame({"Open": o, "High": hi, "Low": lo, "Close": cl, "Volume": r.lognormal(13, 0.4, n)},
                        index=pd.bdate_range(start, periods=n))


# 1) 선 넘음·목표 도달 손 계산
cl = np.r_[np.full(5, 100.0), [99.0, 98.5, 97.9, 99.0], np.full(10, 99.0)]
atr = np.full(len(cl), 2.0)
chk("1 지지 깨짐: 선 = 100 − 1×ATR(2) = 98, 10일 안 종가 97.9 → 1", engine.lec_hit(cl, atr, 4, 1.0, 0.0, -1, 10) == 1.0)
chk("1-2 선 = 100 − 1.5×ATR = 97 이면 깨지지 않음 → 0", engine.lec_hit(cl, atr, 4, 1.5, 0.0, -1, 10) == 0.0)
chk("1-3 기울기 있는 선: 하루 +1%씩 오르는 선 98·1.01^k는 99에서 깨짐", engine.lec_hit(np.r_[np.full(5, 100.0), np.full(12, 99.5)], atr[:17], 4, 1.0, np.log(1.01), -1, 10) == 1.0)
chk("1-4 자료가 모자라면 None", engine.lec_hit(cl, atr, len(cl) - 3, 1.0, 0.0, -1, 10) is None)
hi2 = np.r_[np.full(3, 100.0), [104.0, 106.0], np.full(5, 100.0)]
cl2 = np.r_[np.full(3, 100.0), [103.0, 105.0], np.full(5, 100.0)]
chk("1-5 목표(+5%)가 무효선(−5%)보다 먼저 → 1", engine.lec_target_first(hi2, cl2, 0, 105.0, 95.0, 8) == 1.0)
cl3 = np.r_[np.full(3, 100.0), [94.0], np.full(6, 106.0)]
chk("1-6 무효선(종가 < 95)이 먼저 → 0", engine.lec_target_first(cl3 + 1, cl3, 0, 105.0, 95.0, 8) == 0.0)

# 2) 약화 손 계산: 저점 90 → 고점 110 → 저점 100 → 고점 H2 125(40) → 저점 115(50) → 고점 H1 124(60, 거래량 절반)
path = np.r_[[100.0], seg(100, 90, 11), seg(90, 110, 11), seg(110, 100, 11), seg(100, 125, 11), seg(125, 115, 11), seg(115, 124, 11),
             seg(124, 110, 16), np.full(20, 110.0)]
vol = np.full(len(path), 1e6)
vol[51:61] = 5e5
b2 = engine.lec_base(frame(path, vol))
sw2 = engine.lec_swings(b2)
st2 = engine.lec_state(b2, sw2)
wk = [x for x in engine.lec_weakness(b2, sw2) if x["t"] == 65]
w = wk[0] if wk else {}
chk("2 H1(60) 확정일 65에 비교, 앞 고점 H2 = 40번째 봉(확정 45)", w.get("setup") == 45, w)
chk("2-2 H2 확정일 상태 = 상승", st2[45] == 1, st2[40:50])
chk("2-3 ① 전고점 갱신 실패(124 ≤ 125 구간 상단)", w.get("W01") is True)
chk("2-4 ② 상승 구간 거래량 감소(거래량 절반)", w.get("W02") is True)
chk("2-5 ③·④는 H1 > H2일 때만 — 여기선 아님", w.get("W03") is False and w.get("W04") is False)
chk("2-6 약화 2개 이상(W06)", w.get("W06") is True)
ev2 = engine.lec_l4_events(b2, sw2, st2)
chk("2-7 사건 표에 65일 W01·W02·W06", {"W01", "W02", "W06"} <= set(ev2.loc[ev2["t"] == 65, "event"]), ev2[ev2["t"] == 65])

# 3) 윗꼬리 대량: 상승 상태에서 긴 윗꼬리 + 60일 최대 거래량 → 다음 날들 종가가 그 캔들 저가 아래 = 확정(W15)
p3 = np.r_[[100.0], seg(100, 90, 11), seg(90, 110, 11), seg(110, 100, 11), seg(100, 130, 21), np.full(3, 130.0)]
n3 = len(p3)
o3, h3, l3, v3 = p3.copy(), p3 + 0.01, p3 - 0.01, np.full(n3, 1e6)
w3 = n3 - 1
h3[w3], v3[w3] = 140.0, 5e6                       # 윗꼬리 10, 몸통 0
p3 = np.r_[p3, [128.0, 127.0, 126.0], np.full(15, 126.0)]
o3, h3, l3, v3 = np.r_[o3, [128.0, 127, 126], np.full(15, 126.0)], np.r_[h3, [128.01, 127.01, 126.01], np.full(15, 126.01)], \
    np.r_[l3, [127.99, 126.99, 125.99], np.full(15, 125.99)], np.r_[v3, np.full(18, 1e6)]
b3 = engine.lec_base(frame(p3, v3, o=o3, hi=h3, lo=l3))
sw3 = engine.lec_swings(b3)
st3 = engine.lec_state(b3, sw3)
ev3 = engine.lec_l4_events(b3, sw3, st3)
chk("3 윗꼬리 대량 캔들 = 앱 UPPER_WICK_CLIMAX", bool(b3["UPPER_WICK_CLIMAX"].iloc[w3]) and st3[w3] == 1, (b3["UPPER_WICK_CLIMAX"].iloc[w3], st3[w3]))
chk("3-2 다음 날 종가 128 < 그 캔들 저가 129.99 → 확정(W15) 그날", set(ev3.loc[ev3["event"] == "W15", "t"]) == {w3 + 1}, ev3[ev3["event"].isin(["W15", "W16"])])

# 4) 붕괴: 상승 중 확정 저점 115 이탈 → 10일 안 회복 없음 → 다음 확정 고점 120 < 앞 고점 125 → W10
p4 = np.r_[[100.0], seg(100, 90, 11), seg(90, 110, 11), seg(110, 100, 11), seg(100, 125, 11), seg(125, 115, 11), seg(115, 122, 6),
           seg(122, 105, 11), seg(105, 120, 11), seg(120, 100, 11), np.full(20, 100.0)]
b4 = engine.lec_base(frame(p4))
sw4 = engine.lec_swings(b4)
st4 = engine.lec_state(b4, sw4)
col = engine.lec_collapse(b4, sw4)
zl115 = lambda t: engine.lec_zone(115.01 - 0.02, b4["ATR14"].iloc[t])[0]
tb_exp = next(t for t in range(56, len(p4)) if b4["Close"].iloc[t] < engine.lec_zone(float(sw4["l_p"][np.searchsorted(sw4["l_o"], t - 5, "right") - 1]), b4["ATR14"].iloc[t])[0])
c4 = next((x for x in col if x["tb"] == tb_exp), {})
chk("4 지지 이탈일 = 종가가 최근 확정 저점 구간 하단 아래로 처음 마감한 날", bool(c4), (tb_exp, col))
k_after = int(np.searchsorted(sw4["h_o"], tb_exp, "right"))
chk("4-2 붕괴 3단계 날 = 이탈 뒤 첫 확정 고점(120 < 앞 고점) 확정일과 이탈+10 중 늦은 날",
    c4.get("w10") == max(int(sw4["h_o"][k_after]) + 5, tb_exp + 10), (c4, sw4["h_o"][k_after]))

# 5) 상승 추세선 이탈
ul = engine.lec_upline(b4, sw4)
chk("5 추세선 이탈일 = 선 구간 하단 아래 종가(이탈 뒤 선 값 없음)", len(ul["breaks"]) >= 1 and not np.isfinite(ul["val"][ul["breaks"][0]]))

# 6) 미래 자료 미사용: 자른 자료로 계산해도 자른 날까지의 L4·L5 사건(종류·거리·기울기)이 같다
K = 2200
for seed in (31, 32):
    full = walk(2600, seed)
    cut = full.iloc[:K + 1]
    res = {}
    for tag, d in (("full", full), ("cut", cut)):
        bb = engine.lec_base(d)
        sw_ = engine.lec_swings(bb)
        st_ = engine.lec_state(bb, sw_)
        e4 = engine.lec_l4_events(bb, sw_, st_)
        e5 = pd.DataFrame(engine.lec_l5_events(bb, sw_, st_))
        vpe = pd.DataFrame(engine.lec_volume_profile_events(bb))[["t", "grp"]]
        res[tag] = (e4, e5, vpe)
    f4, f5, fv = res["full"]
    c4_, c5, cv = res["cut"]
    a4 = f4[f4["t"] <= K].sort_values(["event", "t"]).reset_index(drop=True)
    b4_ = c4_.sort_values(["event", "t"]).reset_index(drop=True)
    chk(f"6 미래 자료 미사용 — L4 사건 같음(종목 {seed}, {len(a4)}건)", a4.equals(b4_), (len(a4), len(b4_)))
    cols5 = ["t", "q", "kind", "a", "s"]
    a5 = f5[f5["t"] <= K - 1][cols5].sort_values(["q", "t", "kind"]).reset_index(drop=True)
    b5 = c5[c5["t"] <= K - 1][cols5].sort_values(["q", "t", "kind"]).reset_index(drop=True)
    same5 = a5.shape == b5.shape and bool(((a5 == b5) | (a5.isna() & b5.isna())).all().all())
    chk(f"6-2 미래 자료 미사용 — L5 사건·거리 같음(종목 {seed}, {len(a5)}건)", same5, (a5.shape, b5.shape))
    av = fv[fv["t"] <= K].reset_index(drop=True)
    chk(f"6-3 미래 자료 미사용 — 빈 공간·매물대 진입 같음(종목 {seed})", av.equals(cv.reset_index(drop=True)))
chk("6-4 L4 사건 종류가 여럿", f4["event"].nunique() >= 10, f4["event"].value_counts().to_dict())

# 7) 통과선 방식: 효과 없는 가짜 가격 → 후보 없음 / 심은 효과는 잡음
saved = {k: LC[k] for k in ("MIN_EVENTS", "MIN_TICKERS", "RC_REPS", "MIN_MARKET_EVENTS")}
LC.update({"MIN_EVENTS": 60, "MIN_TICKERS": 8, "RC_REPS": 30, "MIN_MARKET_EVENTS": 20})
try:
    pmn = {(f"T{i}" if i % 3 else f"{i:06d}.KS"): walk(4000, 300 + i) for i in range(16)}
    prn = engine.lec_prepare(pmn, {k: "Technology" for k in pmn}, {}, None, {}, need_rs=False)
    t4 = engine.lec_l4_table(prn)
    r4 = engine.run_lecture_l4(t4)
    chk("7 L4 효과 없는 가짜 가격: 후보 없음", r4["candidates"] == [] and r4["threshold"] is not None, (r4["candidates"], r4["threshold"]))
    chk("7-2 L4 후보가 없으면 검증 기간을 열지 않음", r4["verify"] == {})
    ev = t4["events"].copy()
    ev.loc[ev["event"] == "W17", "xdd"] += 0.05
    r4b = engine.run_lecture_l4({**t4, "events": ev})
    chk("7-3 L4 W17 뒤에 낙폭 5%p를 심으면 W17이 후보", "W17" in r4b["candidates"], r4b["candidates"])
    chk("7-4 L4 보고서", "통과선 t" in engine.render_lecture_l4_report(r4b, "t"))
    t5 = engine.lec_l5_table(prn)
    r5 = engine.run_lecture_l5(t5)
    chk("7-5 L5 효과 없는 가짜 가격: 후보 없음", r5["candidates"] == [] and r5["threshold"] is not None, (r5["candidates"], r5["threshold"]))
    pr = t5["paired"].copy()
    pr.loc[pr["q"] == "V02", "real"] = 0.0           # 반응 3회 이상 지지가 절대 안 깨진다고 심음
    r5b = engine.run_lecture_l5({**t5, "paired": pr})
    chk("7-6 L5 V02를 '안 깨짐'으로 심으면 V02가 후보", "V02" in r5b["candidates"], (r5b["candidates"], r5b["stats"]["V02"].get("t")))
    chk("7-7 L5 심지 않은 V07은 후보 아님", "V07" not in r5b["candidates"])
    chk("7-8 L5 보고서", "가짜 선" in engine.render_lecture_l5_report(r5b, "t"))
finally:
    LC.update(saved)

print(f"\n{'모두 통과' if not FAILS else str(len(FAILS)) + '개 실패: ' + ', '.join(FAILS)}")
sys.exit(1 if FAILS else 0)
