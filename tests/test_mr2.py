"""H-MR2 단기 신호 찾기 (engine/mr2_backtest.py) — 청산·체결 손 계산, 신호 다시 계산, 미래 자료 미사용, 1포지션, 비교 기준,
섞기 통과선의 날 공유, 우위 없는 가짜 가격에서 후보 없음, 엄격 체결 조건."""
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


C2 = engine.MR2_CONFIG
C1 = engine.MR1_CONFIG


def walk(n, seed, drift=0.0003, sd=0.022, start="1999-01-04"):
    r = np.random.default_rng(seed)
    cl = 50 * np.exp(np.cumsum(drift + r.normal(0, sd, n)))
    o = np.r_[cl[0], cl[:-1]] * np.exp(r.normal(0, 0.004, n))
    hi = np.maximum(o, cl) * np.exp(np.abs(r.normal(0, 0.01, n)))
    lo = np.minimum(o, cl) * np.exp(-np.abs(r.normal(0, 0.01, n)))
    return pd.DataFrame({"Open": o, "High": hi, "Low": lo, "Close": cl, "Volume": r.lognormal(13, 0.5, n)},
                        index=pd.bdate_range(start, periods=n))


def path_walk(n, seed, sd=0.022, steps=30, start="1999-01-04"):
    """장중 경로까지 진짜 무작위 걸음(마팅게일)인 가짜 가격 — 시가·고가·저가·종가가 한 경로에서 나온다."""
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


# 1) 청산 손 계산 (+5% · −5% · 3거래일, 같은 날 둘 다면 손절, 시가 갭)
def one(rows):
    a = np.array(rows, dtype=float)              # 줄 = (시가, 고가, 저가, 종가), 첫 줄 종가 = 진입가
    r = engine.mr2_bracket_all(a[:, 0], a[:, 1], a[:, 2], a[:, 3])
    return round(float(r["gross"][0]), 6), int(r["kind"][0]), int(r["hold"][0])


base0 = [100, 100, 100, 100]
chk("1 둘째 날 고가 +5% → 105에 익절", one([base0, [101, 103, 99, 102], [102, 105.5, 101, 104], [104, 104, 103, 103]]) == (0.05, 1, 2))
chk("1-2 같은 날 +5%·−5% 둘 다 → 손절", one([base0, [100, 106, 94, 100], base0, base0]) == (-0.05, 2, 1))
chk("1-3 시가가 +5% 위 → 시가에 익절", one([base0, [107, 108, 106, 107], base0, base0]) == (0.07, 4, 1))
chk("1-4 시가가 −5% 아래 → 시가에 손절", one([base0, [92, 93, 90, 91], base0, base0]) == (-0.08, 5, 1))
chk("1-5 3거래일 동안 아무 일 없음 → 셋째 날 종가", one([base0, [101, 103, 98, 102], [102, 104, 99, 99], [99, 101, 97, 98]]) == (-0.02, 3, 3))
r0 = engine.mr2_bracket_all(*(np.ones(3) * 100 for _ in range(4)))
chk("1-6 3거래일 뒤 자료가 없으면 NaN", bool(np.all(np.isnan(r0["gross"]))))
wb = engine.lec_base(walk(2500, 3, sd=0.03))
o_, h_, l_, c_ = (wb[k].to_numpy() for k in ("Open", "High", "Low", "Close"))
ba = engine.mr2_bracket_all(o_, h_, l_, c_)
same = all(abs(engine.mr2_after(o_, h_, l_, c_, t, c_[t])[0] - ba["gross"][t]) < 1e-12
           and engine.mr2_after(o_, h_, l_, c_, t, c_[t])[1:] == (int(ba["kind"][t]), int(ba["hold"][t])) for t in range(len(c_) - 3))
chk("1-7 한 번에 계산(벡터)과 하나씩 계산이 모든 날 같음", same)

# 2) 지정가 체결 손 계산 (S02a: P = 전날 종가 × 0.95)
n = 320
O, H, L, Cc = np.full(n, 100.0), np.full(n, 100.5), np.full(n, 99.5), np.full(n, 100.0)
ev = {260: (97, 98, 94.5, 96),     # 저가 94.5 ≤ 95 → 95에 체결, 다음 날 고가 100.5 ≥ 99.75 → 익절
      261: (96, 100.5, 95.5, 100),
      270: (93, 96, 92, 95),       # 시가 93 ≤ 95 → 시가 93에 체결, 다음 날 시가 100 ≥ 97.65 → 시가 익절
      280: (97, 97.5, 90, 91),     # 95에 체결, 저가 90 ≤ 90.25 → 체결일 손절
      290: (99, 104, 95, 103)}     # 저가가 95에 딱 닿음 → 기본은 체결(체결일 +5%는 안 셈), 엄격 체결은 안 됨
for t, v in ev.items():
    O[t], H[t], L[t], Cc[t] = v
hand = pd.DataFrame({"Open": O, "High": H, "Low": L, "Close": Cc, "Volume": np.full(n, 1e6)}, index=pd.bdate_range("2005-01-03", periods=n))
hb = engine.lec_base(hand)
hsw = engine.lec_swings(hb)
elA = np.ones(n, dtype=bool)
rt = engine.mr2_ticker(hb, hsw, elA, 250, n - 5, ["S02a-L", "S02a-C"])
lt = rt["rules"]["S02a-L"]
chk("2 지정가 체결일 = 260·270·280·290", list(lt["t"]) == [260, 270, 280, 290], list(lt["t"]))
chk("2-2 체결가 = min(시가, P): 95 · 93 · 95 · 95", np.allclose(lt["entry"], [95, 93, 95, 95]), list(lt["entry"]))
chk("2-3 260: 다음 날 +5% 익절(보유 1일)", abs(lt["gross"][0] - 0.05) < 1e-12 and lt["kind"][0] == 1 and lt["hold"][0] == 1)
chk("2-4 270: 다음 날 시가 100 ≥ 97.65 → 시가 익절", abs(lt["gross"][1] - (100 / 93 - 1)) < 1e-12 and lt["kind"][1] == 4)
chk("2-5 280: 체결일 저가 ≤ 체결가 × 0.95 → 그날 손절(보유 0일)", abs(lt["gross"][2] + 0.05) < 1e-12 and lt["kind"][2] == 6 and lt["hold"][2] == 0)
chk("2-6 290: 체결일 고가 104 ≥ 99.75지만 그날 익절로 세지 않음 → 다음 날 시가 익절", lt["kind"][3] == 4 and lt["hold"][3] == 1 and abs(lt["gross"][3] - (100 / 95 - 1)) < 1e-12)
st_ = rt["strict"]["S02a-L"]
chk("2-7 엄격 체결: 딱 닿은 290은 빠지고 260(더 내려감)·270(시가 ≤ P)·280은 체결", list(st_["t"]) == [260, 270, 280], list(st_["t"]))
lim = rt["limit"]["S02a-L"]
chk("2-8 체결 수·시가 체결·체결일 손절 셈", lim["fills"] == 4 and lim["open_fills"] == 1 and lim["fill_day_stop"] == 1, lim)
ct = rt["rules"]["S02a-C"]
chk("2-9 종가 진입 S02a: 종가 ≤ 전날 × 0.95인 270(95)·280(91)", list(ct["t"]) == [270, 280] and np.allclose(ct["entry"], [95, 91]), list(ct["t"]))
chk("2-10 비교 기준 = 대상인 모든 날(3거래일 뒤 자료가 있는 날)", len(rt["pool_t"]) == (n - 5) - 250 + 1 and
    np.allclose(rt["pool_net"], engine.mr2_bracket_all(O, H, L, Cc)["gross"][250:n - 4] - C2["COST"]))
elB = np.ones(n, dtype=bool)
elB[269] = False
rtB = engine.mr2_ticker(hb, hsw, elB, 250, n - 5, ["S02a-L", "S02a-C"])
chk("2-11 지정가는 전날(t−1) 대상일 때만 주문 — 269일이 대상 아니면 270 체결 없음, 종가 진입 270은 그대로",
    270 not in list(rtB["rules"]["S02a-L"]["t"]) and 270 in list(rtB["rules"]["S02a-C"]["t"]))

# 3) 신호 다시 계산
w = walk(3200, 11, sd=0.028)
b = engine.lec_base(w)
sw = engine.lec_swings(b)
o, h, l, c = (b[k].to_numpy() for k in ("Open", "High", "Low", "Close"))
atr = b["ATR14"].to_numpy()
sig, px = engine.mr2_signal_arrays(b, sw, set(engine.MR2_SIGNALS))
T = range(30, len(c))
chk("3 S01a = 종가 ≤ 전날 종가 − 2×전날 ATR", all(sig["S01a"][t] == (c[t] <= c[t - 1] - 2 * atr[t - 1]) for t in T) and sig["S01a"].sum() > 0)
chk("3-2 S01b 지정가 = 전날 종가 − 3×전날 ATR", np.allclose(px["S01b"][30:], (c[29:-1] - 3 * atr[29:-1]), equal_nan=True))
chk("3-3 S03 = 시가 ≤ 전날 종가 × 0.97", all(sig["S03"][t] == (o[t] <= 0.97 * c[t - 1]) for t in T))
chk("3-4 S04b = 4일 연속 종가 하락", all(sig["S04b"][t] == all(c[t - i] < c[t - i - 1] for i in range(4)) for t in T) and sig["S04b"].sum() > 0)
chk("3-5 S04b ⊆ S04a", bool(np.all(sig["S04a"] | ~sig["S04b"])))
mc = engine.mr_signals(b, sw, "C")["t"]
chk("3-6 S05 종가 = MR1 규칙 C 신호", sorted(np.flatnonzero(sig["S05"][45:]) + 45) == sorted(mc) and len(mc) > 0, (len(mc), int(sig["S05"].sum())))
dr = b["DROP5"].to_numpy()
ok5 = all(abs(px["S05"][t] - (c[t - 5] - 3 * atr[t - 1])) < 1e-9 for t in T if np.isfinite(px["S05"][t]))
chk("3-7 S05 지정가 = 5거래일 전 종가 − 3×전날 ATR, 전날 이미 −3 아래면 주문 없음", ok5 and not np.any(np.isfinite(px["S05"]) & (np.r_[np.nan, dr[:-1]] <= -3)))
r2 = engine.short_rsi2(b["Close"]).to_numpy()
chk("3-8 S06 = RSI(2) < 10 (T-B와 같은 식)", np.array_equal(sig["S06"], r2 < 10))
chk("3-9 S07 = S06 + 200일선(다 찬 것) 위 — 200번째 날 전에는 없음", not sig["S07"][:199].any() and bool(np.all(sig["S06"] | ~sig["S07"])) and sig["S07"].sum() > 0)
r14 = b["RSI14"].to_numpy()
chk("3-10 S08 = RSI14가 30 이하로 처음 내려간 날", all(sig["S08"][t] == (r14[t] <= 30 and r14[t - 1] > 30) for t in T) and sig["S08"].sum() > 0)
base = engine.lec_bnf_base_series(b, sw)
disp = b["DISP25"].to_numpy()
ok10 = True
cnt10 = 0
for t in T:
    pb = base[t - 1]
    exp_sig = bool(np.isfinite(pb) and disp[t - 1] > pb and disp[t] <= pb)
    ok10 &= sig["S10"][t] == exp_sig
    if np.isfinite(px["S10"][t]):
        cnt10 += 1
        P = px["S10"][t]
        s24 = c[t - 24:t].sum()
        ok10 &= abs(P / ((s24 + P) / 25) * 100 - pb) < 1e-8 and disp[t - 1] > pb
chk(f"3-11 S10: 종가 = 이격도가 전날 기준을 처음 밑돎 · 지정가 = 그날 종가가 그 값이면 이격도가 기준과 같아지는 가격 ({cnt10}일)", ok10 and cnt10 > 0)
m20 = np.array([l[t - 20:t].min() if t >= 20 else np.nan for t in range(len(l))])
chk("3-12 S11: 지정가 = 직전 20일 저가 최저, 종가 = 그날 저가가 그 아래", np.allclose(px["S11"][20:], m20[20:]) and
    all(sig["S11"][t] == (l[t] < m20[t]) for t in T))
lines = engine.lec_swing_lines(b, sw)
ok12 = True
k12 = 0
for t in range(300, len(c), 37):
    y = t - 1
    best = None
    for lv_, s_, f_ in zip(lines["lv"], lines["start"], lines["fr"]):
        if s_ <= y and ((y - s_) <= 250 or f_ <= y):
            zl, zu = engine.lec_zone(lv_, atr[y])
            if zu < c[y] and (best is None or lv_ > best[0]):
                best = (lv_, zl, zu)
    if best is None:
        ok12 &= np.isnan(px["S12"][t])
    else:
        k12 += 1
        ok12 &= abs(px["S12"][t] - best[2]) < 1e-9 and sig["S12"][t] == (l[t] <= best[2] and c[t] >= best[1])
chk(f"3-13 S12: 전날 종가 아래 가장 가까운 살아 있는 선의 구간(하나씩 다시 계산, {k12}일)", ok12 and k12 > 10)
chk("3-14 S13 = 음봉 + RVOL ≥ 2", np.array_equal(sig["S13"], (c < o) & (b["RVOL"].to_numpy() >= 2)))
mb = engine.mr_signals(b, sw, "B", base)["t"]
chk("3-15 S14 = MR1 규칙 B 신호", sorted(np.flatnonzero(sig["S14"])) == sorted(mb))
chk("3-16 S15 = 앱 BIG_BEAR", np.array_equal(sig["S15"], b["BIG_BEAR"].to_numpy(dtype=bool)) and sig["S15"].sum() > 0)
chk("3-17 지정가 규칙이 있는 신호 = 8개, 규칙 = 26개", len(px) == 8 and len(engine.MR2_RULES) == 26 and set(px) == {s for s, (_, x) in engine.MR2_SIGNALS.items() if x})

# 4) 미래 자료 미사용: t 뒤(지정가는 t 자체도)를 다른 값으로 바꿔도 t의 신호·가격이 같다
K = 2600
alt = walk(len(w), 99, sd=0.04).set_index(w.index)
ratio = w["Close"].iloc[K] / alt["Close"].iloc[K]
w2 = w.copy()
w2.iloc[K + 1:, :4] = (alt.iloc[K + 1:, :4] * ratio).to_numpy()
w2.iloc[K + 1:, 4] = alt.iloc[K + 1:, 4].to_numpy()
w3 = w.copy()
w3.iloc[K:, :4] = (alt.iloc[K:, :4] * w["Close"].iloc[K - 1] / alt["Close"].iloc[K - 1]).to_numpy()
w3.iloc[K:, 4] = alt.iloc[K:, 4].to_numpy() * 50
nf_c = nf_l = True
for ww, cut, which in ((w2, K, "c"), (w3, K, "l")):
    bb_ = engine.lec_base(ww)
    s2, p2 = engine.mr2_signal_arrays(bb_, engine.lec_swings(bb_), set(engine.MR2_SIGNALS))
    if which == "c":
        nf_c = all(np.array_equal(s2[k][:cut + 1], sig[k][:cut + 1]) for k in sig)
    else:
        nf_l = all(np.allclose(p2[k][:cut + 1], px[k][:cut + 1], equal_nan=True) for k in px)
chk("4 미래 자료 미사용 — 종가 신호는 그날 뒤를 바꿔도 같음", nf_c)
chk("4-2 미래 자료 미사용 — 지정가 P는 그날 자체를 바꿔도 같음(전날까지 자료)", nf_l)

# 5) 단계: 1포지션 · 값 x = 순수익 − 그 종목 기준 평균 · 미국/한국
prep = {}
for i in range(24):
    sym = f"U{i}" if i < 14 else f"{i:06d}.KS"
    bi = engine.lec_base(path_walk(4400, 300 + i, sd=0.02 + 0.002 * (i % 6)))
    prep[sym] = {"b": bi, "sw": engine.lec_swings(bi)}
elig_all = {s: np.ones(len(p["b"]), dtype=bool) for s, p in prep.items()}
stg = engine.mr2_stage(prep, elig_all, ["S04a-C", "S06-C", "S11-L", "S02a-L"], "2001-01-01", "2014-12-31")
tr = stg["tr"]
ok1 = True
for (r, s), g in tr.groupby(["rule", "ticker"]):
    tt, hh = g["t"].to_numpy(), g["hold"].to_numpy()
    ok1 &= bool(np.all(tt[1:] > tt[:-1] + hh[:-1]))
chk("5 규칙마다 종목당 1포지션 — 다음 진입 > 앞 거래 판 날", ok1 and len(tr) > 500, len(tr))
s0 = next(iter(prep))
mu0 = stg["mu"][s0]
g0 = tr[tr["ticker"] == s0]
chk("5-2 x = 순수익 − 그 종목 대상 날 전부의 종가 진입 평균", np.allclose(g0["x"], g0["net"] - mu0) and abs(stg["pool"][s0].mean()) < 1e-12)
chk("5-3 순수익 = 비용 전 − 0.30%", np.allclose(tr["gross"] - tr["net"], 0.003))
d = pd.to_datetime(tr["date"])
chk("5-4 신호일이 기간 안", bool(((d >= "2001-01-01") & (d <= "2014-12-31")).all()))
chk("5-5 미국·한국 둘 다 있음", set(tr["market"]) == {"US", "KR"})
mrf = tr[tr["rule"] == "S04a-C"].iloc[0]
same_day = [stg["pool"][s][stg["pool"][s].index == mrf["date"]] + stg["mu"][s] for s in prep if engine.lec_market_of(s) == mrf["market"]]
exp_m = np.mean([v.iloc[0] for v in same_day if len(v)])
chk("5-6 같은 날 같은 시장 대상 종목 평균(참고) 손 계산", abs(mrf["mref"] - exp_m) < 1e-12)

# 6) 섞기 통과선: 같은 종목·신호일은 모든 규칙에서 같은 무작위 날
fake = {"tr": pd.concat([tr[tr["rule"] == "S04a-C"], tr[tr["rule"] == "S04a-C"].assign(rule="COPY")], ignore_index=True),
        "pool": stg["pool"], "period": stg["period"]}
saved_reps = C2["RC_REPS"]
C2["RC_REPS"] = 25
try:
    a1 = engine.mr2_reality(fake, ["S04a-C"], 7)
    a2 = engine.mr2_reality(fake, ["S04a-C", "COPY"], 7)
finally:
    C2["RC_REPS"] = saved_reps
chk("6 똑같은 규칙을 하나 더 넣어도 통과선이 그대로(겹침을 그대로 둠)", np.allclose(a1["max_t"], a2["max_t"]) and a1["threshold"] == a2["threshold"])
chk("6-2 무작위 날로 바꾼 값의 평균 ≈ 0 (종목 평균을 뺐으므로)", abs(a1["null_mean"]["S04a-C"]) < 0.002, a1["null_mean"])

# 7) 판정: 심은 차이 → 후보 / 지정가는 엄격 체결로도 넘어야
t4 = tr[tr["rule"] == "S04a-C"].copy()
s_plus = engine.mr2_stats(t4.assign(x=t4["x"] + 0.03))
saved = {k: C2[k] for k in ("MIN_TRADES", "MIN_TICKERS", "MIN_MARKET")}
C2.update({"MIN_TRADES": 50, "MIN_TICKERS": 5, "MIN_MARKET": 10})
try:
    j1 = engine.mr2_judge_explore(s_plus, 2.5)
    chk("7 거래마다 +3%p를 심으면 찾기 후보", j1["candidate"], j1["reasons"])
    j2 = engine.mr2_judge_explore(s_plus, 2.5, strict=engine.mr2_stats(t4))
    chk("7-2 지정가: 엄격 체결 결과가 기준 미달이면 후보 아님", not j2["candidate"] and any("엄격 체결" in x for x in j2["reasons"]), j2["reasons"])
    j3 = engine.mr2_judge_explore(engine.mr2_stats(t4), 2.5)
    chk("7-3 그대로(우위 없음)면 후보 아님", not j3["candidate"])
    j4 = engine.mr2_judge_simple(s_plus, 3.0, 100, True, strict=s_plus)
    chk("7-4 최종 판정(t ≥ 3, 거래 ≥ 100) 통과", j4["passed"], j4["reasons"])
finally:
    C2.update(saved)

# 8) 전체 실행: 장중 경로가 진짜 무작위 걸음이면 후보 없음 + 보고서·기록·JSON
prep8 = {}
for i in range(26):
    sym = f"V{i}" if i < 14 else f"{i + 100:06d}.KS"
    bi = engine.lec_base(path_walk(5400, 700 + i, sd=0.018 + 0.002 * (i % 7)))
    prep8[sym] = {"b": bi, "sw": engine.lec_swings(bi)}
saved = {k: C2[k] for k in ("MIN_TRADES", "MIN_TICKERS", "MIN_MARKET", "RC_REPS")}
C2.update({"MIN_TRADES": 60, "MIN_TICKERS": 6, "MIN_MARKET": 10, "RC_REPS": 40})
try:
    r8 = engine.run_mr2(prep8)
    chk("8 우위 없는 가짜 가격(장중 경로 = 무작위 걸음): 후보 없음", r8["candidates"] == [],
        [(k, round(r8["rules"][k]["stats"]["t"] or 0, 2)) for k in r8["candidates"]])
    chk("8-2 경쟁 규칙이 여럿이고 통과선 ≥ 2", len(r8["competing"]) >= 10 and r8["threshold"] >= 2.0, (len(r8["competing"]), r8["threshold"]))
    chk("8-3 지정가 규칙 전부 엄격 체결 결과가 있음", all("strict" in r8["rules"][k] for k in engine.MR2_RULES if k.endswith("-L")))
    chk("8-4 후보가 없으면 검증 기간(2020~)을 열지 않음", r8["verify"] == {} and r8["verified"] == [])
    rep = engine.render_mr2_report(r8, "test")
    chk("8-5 보고서: 판정표 26줄 · 통과선 · 진입 방식 비교 · 검증 안 엶", rep.count(" · 종가 |") + rep.count(" · 지정가 |") >= 26 and "통과선 t" in rep
        and "진입 방식 비교" in rep and "열지 않았다" in rep)
    chk("8-6 기록 26줄", len(engine.mr2_log_lines(r8, "t")) == 26)
    json.dumps(r8, ensure_ascii=False, default=str)
    chk("8-7 JSON으로 저장 가능", True)
    rf = engine.run_mr2_final(prep8, ["S06-C", "S11-L"])
    repf = engine.render_mr2_final_report(rf, ["X"], "test")
    chk("8-8 최종 실행·보고서(규칙 2개, 지정가는 엄격 체결 포함)", set(rf["rules"]) == {"S06-C", "S11-L"} and "strict" in rf["rules"]["S11-L"] and "최종" in repf)
    pp, dropped = engine.mr2_prepare_plain({"A": walk(300, 1), "B": walk(200, 2)})
    chk("8-9 최종 재료: 260봉 미만은 빼고 대체 없음", list(pp) == ["A"] and dropped == ["B"])
finally:
    C2.update(saved)

# 9) 검정력: 하루 −5% 뒤 다음 날 +3% 갭이 있는 가짜 가격 → 후보가 나오고 2020년 이후 검증까지 간다
def rebound(df):
    d = df.copy()
    cl = d["Close"].to_numpy()
    mult = np.ones(len(d))
    acc = 1.0
    for i in range(1, len(d)):
        if cl[i] / cl[i - 1] <= 0.95:
            acc *= 1.03
        mult[i] = acc
    mult = np.r_[1.0, mult[:-1]]                 # 급락 다음 날부터 3% 위로
    for k in ("Open", "High", "Low", "Close"):
        d[k] = d[k].to_numpy() * mult
    return d


prep9 = {}
for i in range(26):
    sym = f"W{i}" if i < 14 else f"{i + 200:06d}.KS"
    bi = engine.lec_base(rebound(path_walk(7000, 900 + i, sd=0.024 + 0.002 * (i % 5))))
    prep9[sym] = {"b": bi, "sw": engine.lec_swings(bi)}
saved = {k: C2[k] for k in ("MIN_TRADES", "MIN_TICKERS", "MIN_MARKET", "RC_REPS", "VERIFY_MIN_TRADES")}
C2.update({"MIN_TRADES": 60, "MIN_TICKERS": 6, "MIN_MARKET": 10, "RC_REPS": 30, "VERIFY_MIN_TRADES": 20})
try:
    r9 = engine.run_mr2(prep9)
    chk("9 급락 뒤 반등을 심은 가짜 가격: S02a 종가 진입이 찾기 후보", "S02a-C" in r9["candidates"],
        (r9["candidates"], r9["rules"]["S02a-C"]["judge"]["reasons"]))
    chk("9-2 후보만 검증하고, 검증 거래는 모두 2020년 이후(2001~2019 기간 묶음에 하나도 없음)", set(r9["verify"]) == set(r9["candidates"]) and
        all(v["stats"]["n"] > 0 and v["stats"].get("periods") == {} for v in r9["verify"].values()))
    vs9 = r9["verify"].get("S02a-C", {})
    chk("9-3 S02a 종가 진입 검증 통과 → 최종 목록(최대 5개)", "S02a-C" in r9["verified"] and len(r9["verified"]) <= 5, (r9["verified"], vs9.get("judge")))
    rep9 = engine.render_mr2_report(r9, "test")
    chk("9-4 보고서에 검증 표", "검증 통과" in rep9)
finally:
    C2.update(saved)

print(f"\n{'모두 통과' if not FAILS else str(len(FAILS)) + '개 실패: ' + ', '.join(FAILS)}")
sys.exit(1 if FAILS else 0)
