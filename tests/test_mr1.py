"""H-MR1 단기 반등 단타 시험 (engine/mr_backtest.py) — 신호 손 계산·규칙 포함 관계·미래 자료 미사용·판정 방식 점검."""
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


MC = engine.MR1_CONFIG
LC = engine.LECTURE_CONFIG


def walk(n, seed, drift=0.0003, sd=0.022, start="1999-01-04"):
    r = np.random.default_rng(seed)
    cl = 50 * np.exp(np.cumsum(drift + r.normal(0, sd, n)))
    o = np.r_[cl[0], cl[:-1]] * np.exp(r.normal(0, 0.004, n))
    hi = np.maximum(o, cl) * np.exp(np.abs(r.normal(0, 0.01, n)))
    lo = np.minimum(o, cl) * np.exp(-np.abs(r.normal(0, 0.01, n)))
    return pd.DataFrame({"Open": o, "High": hi, "Low": lo, "Close": cl, "Volume": r.lognormal(13, 0.5, n)},
                        index=pd.bdate_range(start, periods=n))


# 1) C 신호 손 계산: 최근 5거래일 변화 ÷ ATR14 ≤ −3 이 처음 된 날
cl = np.r_[np.full(80, 100.0), [98.0, 95.0, 92.0, 89.0, 86.0, 84.0, 83.0], np.full(30, 84.0)]
op = np.r_[cl[0], cl[:-1]]                       # 시가 = 전날 종가(갭 없음)
df = pd.DataFrame({"Open": op, "High": np.maximum(op, cl) + 0.5, "Low": np.minimum(op, cl) - 0.5, "Close": cl,
                   "Volume": np.full(len(cl), 1e6)}, index=pd.bdate_range("2003-01-01", periods=len(cl)))
b1 = engine.lec_base(df)
sw1 = engine.lec_swings(b1)
dr = b1["DROP5"].to_numpy()
exp_t = next(t for t in range(46, len(cl)) if dr[t] <= -3.0 and dr[t - 1] > -3.0)
sg = engine.mr_signals(b1, sw1, "C")
chk("1 C 신호일 = 5일 변화 ÷ ATR이 −3 아래로 처음 내려간 날", sg["t"][:1] == [exp_t], (sg["t"][:3], exp_t))
chk("1-2 계속 −3 아래에 있는 날은 새 신호 아님", all(dr[t - 1] > -3.0 for t in sg["t"]))
tr1 = engine.mr_trades(b1, sg["t"])["trades"]
chk("1-3a 거래가 나옴(진입가 > 손절가, 손절폭 ≤ 12%)", len(tr1) >= 1)
if len(tr1):
    t0 = int(tr1["t"].iloc[0])
    p0 = max(0, t0 - 40) + int(np.argmax(b1["Close"].to_numpy()[max(0, t0 - 40):t0 + 1]))
    chk("1-3 손절 = 급락 시작일(40일 종가 최고)~신호일 최저가, 진입 = 다음 날 시가",
        abs(tr1["stop"].iloc[0] - b1["Low"].to_numpy()[p0:t0 + 1].min()) < 1e-9 and abs(tr1["entry"].iloc[0] - b1["Open"].iloc[t0 + 1]) < 1e-9)
    chk("1-4 비용 0.30% 차감", abs(tr1["gross"].iloc[0] - tr1["ret"].iloc[0] - 0.003) < 1e-12)
    chk("1-5 다음 날 종가가 손절가 아래 → 손절", tr1["exit_kind"].iloc[0] == "손절", tr1.iloc[0].to_dict())

# 2) A ⊆ B, A의 각 조건 다시 계산
ws = [walk(5200, s) for s in (5, 6, 7)]
allA = allB = 0
okA = True
for w in ws:
    b = engine.lec_base(w)
    sw = engine.lec_swings(b)
    base = engine.lec_bnf_base_series(b, sw)
    lines = engine.lec_swing_lines(b, sw)
    blk0 = np.zeros(len(b))
    sB = engine.mr_signals(b, sw, "B", base)["t"]
    sA = engine.mr_signals(b, sw, "A", base, blk0, lines)["t"]
    allA += len(sA)
    allB += len(sB)
    okA &= set(sA) <= set(sB)
    cl_, bbl, lo, atr = (b[k].to_numpy() for k in ("Close", "BB_LOW", "Low", "ATR14"))
    for t in sA:
        seg = lo[t - 3:t]
        m = t - 3 + int(len(seg) - 1 - np.argmin(seg[::-1]))
        bb_ok = bool(np.any(cl_[t - 5:t] < bbl[t - 5:t]))
        alive = (lines["start"] <= m) & ((m - lines["start"] <= 250) | (lines["fr"] <= m))
        hw = np.maximum(0.01 * lines["lv"], 0.5 * atr[m])
        sup_ok = bool(np.any(alive & (np.abs(lines["lv"] - lo[m]) <= hw)))
        okA &= bb_ok and sup_ok
    sA1 = engine.mr_signals(b, sw, "A", base, np.ones(len(b)), lines)
    okA &= sA1["t"] == [] and sA1["skips"]["market"] == len(sA)
    sAn = engine.mr_signals(b, sw, "A", base, np.full(len(b), np.nan), lines)
    okA &= sAn["t"] == [] and sAn["skips"]["market_nodata"] == len(sA)
chk(f"2 A 신호는 B 신호의 일부이고 볼린저·지지 조건을 다시 계산해도 맞음 (A {allA}·B {allB})", okA and allB > allA > 0, (allA, allB))
chk("2-2 시장 필터: 지수가 날마다 '강한 −DI'면 A 신호 0, 지수 자료가 없으면 '자료 없음'으로 셈", okA)

# 3) 미래 자료 미사용: 자른 자료로 계산해도 자른 날까지의 신호가 같다
K = 4200
for rule in ("A", "B", "C"):
    same = True
    for w in ws:
        res = []
        idx_full = walk(len(w), 77).set_index(w.index)
        for d in (w, w.iloc[:K + 1]):
            b = engine.lec_base(d)
            sw = engine.lec_swings(b)
            idxw = idx_full.iloc[:len(d)]
            ib = engine.lec_base(idxw)
            blk = ((ib["ADX14"] >= 25) & (ib["MINUS_DI"] > ib["PLUS_DI"])).astype(float).to_numpy()
            res.append([t for t in engine.mr_signals(b, sw, rule, None, blk)["t"] if t <= K])
        same &= res[0] == res[1]
    chk(f"3 미래 자료 미사용 — 규칙 {rule} 신호가 자른 자료와 같음", same)

# 3-2) 대상: 거래대금 상위 절반 → 그중 변동성 상위 절반 (12종목: 거래량 큰 6개 중 변동성 큰 3개)
n12 = 700
pm12 = {}
for i in range(12):
    r = np.random.default_rng(900 + i)
    sd = 0.005 + 0.002 * i                       # i가 클수록 변동성 큼
    cl = 50 * np.exp(np.cumsum(r.normal(0, sd, n12)))
    vol = np.full(n12, 1e7 if i % 2 == 0 else 1e5)   # 짝수 = 거래대금 큼
    pm12[f"S{i:02d}"] = pd.DataFrame({"Open": cl, "High": cl * 1.001, "Low": cl * 0.999, "Close": cl, "Volume": vol},
                                     index=pd.bdate_range("2001-01-01", periods=n12))
prep12 = engine.lec_prepare(pm12, {k: "Industrials" for k in pm12}, {}, None, {}, need_rs=False)
el = engine.mr_eligibility(prep12)
last_day = {k for k, v in el.items() if v[-1]}
chk("3-2 대상 = 거래대금 큰 6개(짝수) 중 변동성 큰 3개(S06·S08·S10)", last_day == {"S06", "S08", "S10"}, sorted(last_day))
chk("3-3 변동성 250일이 다 차기 전에는 대상 없음", not any(v[:250].any() for v in el.values()))
el_cut = engine.mr_eligibility(engine.lec_prepare({k: v.iloc[:500] for k, v in pm12.items()}, {k: "Industrials" for k in pm12}, {}, None, {}, need_rs=False))
chk("3-4 미래 자료 미사용 — 자른 자료로 계산해도 대상이 같음", all(np.array_equal(el[k][:500], el_cut[k]) for k in pm12))
small = {k: v for k, v in list(pm12.items())[:8]}
el8 = engine.mr_eligibility(engine.lec_prepare(small, {k: "Industrials" for k in small}, {}, None, {}, need_rs=False))
chk("3-5 그날 순위를 낼 종목이 10개 미만이면 대상 없음", not any(v.any() for v in el8.values()))
bS = prep12["S06"]["b"]
fake_tr = pd.DataFrame({"h1": [3, 3], "h2": [6, 6], "frac1": [0.5, 0.5]})
mask = np.zeros(len(bS), dtype=bool)
mask[400:420] = True
hb = engine.mr_hold_baseline(bS, fake_tr, (0, len(bS) - 1), 1, mask)
chk("3-6 무작위 비교 날도 대상인 날에서만 뽑음", len(hb) == 40 and set(hb["date"]) <= set(bS.index[400:420]))

# 4) 판정 방식: 효과 없는 가짜 가격 → 후보 없음 / 심은 차이 → 후보
saved = {k: MC[k] for k in ("MIN_TRADES", "MIN_TICKERS", "MIN_MARKET")}
MC.update({"MIN_TRADES": 40, "MIN_TICKERS": 5, "MIN_MARKET": 10})
try:
    pm = {(f"T{i}" if i % 4 else f"{i:06d}.KS"): walk(6000, 400 + i, sd=0.012 + 0.002 * (i % 7)) for i in range(48)}
    bench = {s: walk(6000, 500 + j) for j, s in enumerate(("^IXIC", "^GSPC", "^KS11", "^KQ11"))}
    prep = engine.lec_prepare(pm, {k: "Technology" for k in pm}, bench, None, {}, need_rs=False)
    idx = engine.mr_index_context(bench)
    r = engine.run_mr1(prep, idx)
    chk("4 효과 없는 가짜 가격: 후보 없음", r["candidates"] == [], [(k, r["rules"][k]["cmp"].get("z")) for k in "ABC"])
    chk("4-2 규칙 셋 모두 거래가 나옴(전체 종목)", all(r["all_stocks"][k]["n_trades"] > 0 for k in "ABC"), [r["all_stocks"][k]["n_trades"] for k in "ABC"])
    chk("4-2b 대상 안 거래 ≤ 전체 거래", all(r["rules"][k]["n_trades"] <= r["all_stocks"][k]["n_trades"] for k in "ABC"))
    u = engine.mr_universe(prep, idx, "C", MC["EXPLORE_START"], MC["EXPLORE_END"], eligible=engine.mr_eligibility(prep))
    tr = u["tr"].copy()
    tr["ret"] += 0.03
    cm = engine.compare_clustered(tr, u["hold"], "ret")
    j = engine.mr_judge(cm, tr, u["hold"], MC["EXPLORE_Z"])
    chk("4-3 거래마다 +3%를 심으면 후보", j["candidate"], j["reasons"])
    rep = engine.render_mr1_report(r, "test")
    chk("4-4 보고서에 판정·단순 반전 비교·빠진 신호", "## 판정" in rep and "A − C" in rep and "빠진 신호" in rep)
    chk("4-5 기록 줄 3개", len(engine.mr1_log_lines(r, "t")) == 3)
finally:
    MC.update(saved)

print(f"\n{'모두 통과' if not FAILS else str(len(FAILS)) + '개 실패: ' + ', '.join(FAILS)}")
sys.exit(1 if FAILS else 0)
