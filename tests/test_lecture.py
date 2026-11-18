"""강의 조건 시험 1단계 (engine/lecture_signals.py · lecture_backtest.py) — 손 계산·미래 자료 미사용·방식 점검."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _offline  # noqa: F401
import sys
import tempfile
import numpy as np
import pandas as pd
import engine

FAILS = []


def chk(n, c, extra=None):
    print(("PASS " if c else "FAIL ") + n + (f"  [{extra}]" if (not c and extra is not None) else ""))
    if not c:
        FAILS.append(n)


close = lambda a, b, tol=1e-9: a is not None and b is not None and abs(float(a) - float(b)) < tol
LC = engine.LECTURE_CONFIG


def frame(cl, vol=None, start="1999-01-04", o=None, hi=None, lo=None):
    cl = np.asarray(cl, dtype=float)
    n = len(cl)
    o = np.r_[cl[0], cl[:-1]] if o is None else np.asarray(o, dtype=float)
    hi = np.maximum(o, cl) * 1.004 if hi is None else np.asarray(hi, dtype=float)
    lo = np.minimum(o, cl) * 0.996 if lo is None else np.asarray(lo, dtype=float)
    v = np.full(n, 1e6) if vol is None else np.asarray(vol, dtype=float)
    return pd.DataFrame({"Open": o, "High": hi, "Low": lo, "Close": cl, "Volume": v}, index=pd.bdate_range(start, periods=n))


def walk(n, seed, drift=0.0003, sd=0.02):
    r = np.random.default_rng(seed)
    cl = 50 * np.exp(np.cumsum(drift + r.normal(0, sd, n)))
    o = np.r_[cl[0], cl[:-1]] * np.exp(r.normal(0, 0.004, n))
    hi = np.maximum(o, cl) * np.exp(np.abs(r.normal(0, 0.01, n)))
    lo = np.minimum(o, cl) * np.exp(-np.abs(r.normal(0, 0.01, n)))
    v = r.lognormal(13, 0.4, n)
    return frame(cl, v, o=o, hi=hi, lo=lo)


# 1) RSI14: 계속 오르면 100 (앱 식은 50이 나오던 문제)
up = frame(100 + np.arange(60.0))
b_up = engine.lec_base(up)
chk("1 계속 오르는 가격의 RSI14 = 100", close(b_up["RSI14"].iloc[-1], 100.0), b_up["RSI14"].iloc[-1])
chk("1-2 앱 함수는 그대로(바꾸지 않음): 같은 가격에서 50", close(engine.compute_base_primitives(up)["RSI14"].iloc[-1], 50.0))
dn = engine.lec_base(frame(100 - np.arange(60.0) * 0.5))
chk("1-3 계속 내리면 RSI14 = 0", close(dn["RSI14"].iloc[-1], 0.0), dn["RSI14"].iloc[-1])
chk("1-4 최근 5거래일 변화(ATR 단위) = (종가 − 5일 전 종가) ÷ ATR14",
    close(b_up["DROP5"].iloc[-1], (b_up["Close"].iloc[-1] - b_up["Close"].iloc[-6]) / b_up["ATR14"].iloc[-1]))

# 2) 상태 손 계산: 저점 L2=90 → 고점 H=110 → 저점 L1=95(더 높음) → 종가가 110을 넘는 날부터 '상승'
seg = lambda a, b_, k: np.linspace(a, b_, k)[1:]
path = np.r_[[100.0], seg(100, 90, 11), seg(90, 110, 11), seg(110, 95, 11), seg(95, 109, 8), seg(109, 115, 6), np.full(20, 115.0)]
df2 = frame(path, hi=path + 0.01, lo=path - 0.01)
b2 = engine.lec_base(df2)
sw2 = engine.lec_swings(b2)
st2 = engine.lec_state(b2, sw2)
i_l1 = int(np.argmin(np.abs(path - 95.0)[25:])) + 25          # L1 = 95 자리
brk = int(np.flatnonzero(path > 110.0)[0])                      # 처음으로 110 위 종가
chk("2 스윙: 저점 90·95, 고점 110이 잡힘", np.isclose(sw2["l_p"], 90 - 0.01).any() and np.isclose(sw2["l_p"], 95 - 0.01).any()
    and np.isclose(sw2["h_p"], 110 + 0.01).any(), (sw2["l_p"], sw2["h_p"]))
chk("2-2 L1이 확정되기 전(5봉)과 돌파 전에는 상승이 아님", (st2[:brk] != 1).all(), st2[:brk + 1])
chk("2-3 L1 확정 뒤 첫 110 돌파 종가 날 = 상승", st2[brk] == 1 and brk > i_l1 + LC["SWING_LR"], (brk, i_l1, st2[brk]))
chk("2-4 돌파 뒤 종가가 다시 110 아래로 와도(새 스윙 전) 상승 유지(Q2: '한 번이라도')",
    st2[brk + 3] == 1 or path[brk + 3] > 110)


# 3) 미래 자료 미사용: 자른 자료로 계산한 값 = 전체 자료로 계산한 같은 날의 값
w3 = walk(1600, 3)
b_full = engine.lec_base(w3)
st_full = engine.lec_state(b_full)
g_full = engine.lec_market_grade(w3)
ok_st = ok_g = True
for T in (400, 777, 1111, 1599):
    bt = engine.lec_base(w3.iloc[:T + 1])
    ok_st &= bool((engine.lec_state(bt)[:T + 1] == st_full[:T + 1]).all())
    gt = engine.lec_market_grade(w3.iloc[:T + 1])
    ok_g &= bool((gt == g_full.reindex(gt.index)).all())
chk("3 상태: 자른 자료와 전체 자료의 같은 날 값이 모두 같음", ok_st)
chk("3-2 시장 등급: 자른 자료와 전체 자료의 같은 날 값이 모두 같음(시작점 고정)", ok_g)
chk("3-3 시장 등급은 200일 EMA가 생긴 뒤부터", len(g_full) and g_full.index[0] == b_full.index[LC["GRADE_MIN_BARS"]])

# 4) 시장 등급 ⚫은 2일 확인: 오르다가 200일 EMA 아래로 내려간 첫날은 아직 ⚫가 아님, 이튿날 ⚫
p4 = np.r_[100 * np.exp(0.002 * np.arange(400)), np.full(10, 100 * np.exp(0.002 * 399) * 0.55)]
g4 = engine.lec_market_grade(frame(p4))
b4 = engine.lec_base(frame(p4))
k4 = int(np.flatnonzero((b4["Close"] < b4["EMA200"]).to_numpy())[0])
chk("4 200일 EMA 아래 첫날은 ⚫ 아님, 둘째 날 ⚫ (하향 2일)", g4.iloc[k4 - LC["GRADE_MIN_BARS"]] != "⚫" and g4.iloc[k4 + 1 - LC["GRADE_MIN_BARS"]] == "⚫",
    (g4.iloc[k4 - 200], g4.iloc[k4 + 1 - 200]))

# 5) 주봉 RS 한 번에 계산 = 하루씩 계산 / RS 일봉 손 계산
etf = walk(1600, 5)
rs = engine.lec_rs_line(w3["Close"], etf["Close"])
rsw = engine.lec_rs_weekly_series(rs)
rng = np.random.default_rng(0)
same = True
for d in rs.index[rng.integers(60, len(rs), 60)]:
    a = engine.lec_rs_weekly_at(rs, d)
    bv = rsw.get(d)
    same &= (a is None and not np.isfinite(bv)) or (a is not None and np.isfinite(bv) and int(bv) == a)
chk("5 주봉 RS: 한 번에 계산한 값 = 그날까지 진행 중인 주를 넣어 하루씩 계산한 값(60일 무작위)", same)
rsd = engine.lec_rs_daily(rs)
t5 = 500
ma = rs.iloc[t5 - 49:t5 + 1].mean()
roc = rs.iloc[t5] / rs.iloc[t5 - 20] - 1
exp5 = 2 if (rs.iloc[t5] > ma and roc > 0) else (1 if rs.iloc[t5] > ma else (-1 if roc > 0 else -2))
chk("5-2 RS 일봉 4구간 손 계산(50일 평균 × 20일 변화)", int(rsd.iloc[t5]) == exp5)
chk("5-3 기준이 없으면 RS선은 비어 있음('데이터 없음', 시장 지수로 대체 안 함)", len(engine.lec_rs_line(w3["Close"], None)) == 0)
chk("5-4 섹터 짝: 미국 Technology → XLK, 한국 커뮤니케이션 → 없음, 섹터 모름 → 없음",
    engine.lec_sector_etf("AAPL", "Technology") == "XLK" and engine.lec_sector_etf("035420.KS", "Communication Services") is None
    and engine.lec_sector_etf("AAPL", None) is None)
chk("5-5 시장 등급 기준 지수(Q4)", engine.lec_index_of("AAPL", "Technology") == "^IXIC" and engine.lec_index_of("XOM", "Energy") == "^GSPC"
    and engine.lec_index_of("005930.KS", None) == "^KS11" and engine.lec_index_of("247540.KQ", None) == "^KQ11")

# 6) BNF 기준 이격도: 한 번에 = 하루씩, 손 계산
sw3 = engine.lec_swings(b_full)
bs = engine.lec_bnf_base_series(b_full, sw3)
same6 = all((engine.lec_bnf_base_at(b_full, sw3, t) is None and np.isnan(bs[t])) or close(engine.lec_bnf_base_at(b_full, sw3, t), bs[t])
            for t in rng.integers(0, len(b_full), 80))
chk("6 기준 이격도: 한 번에 계산 = 하루씩 계산(80일)", same6)
t6 = 1200
lo_ = sw3["l_o"]
o6 = lo_[(lo_ >= t6 - 749) & (lo_ <= t6 - 5)]
exp6 = np.median(np.sort(b_full["DISP25"].to_numpy()[o6])[:5])
chk("6-2 기준 = 지난 750일 확정 스윙 저점 이격도 중 낮은 5개의 중앙값", close(bs[t6], exp6), (bs[t6], exp6))

# 7) BNF 청산 손 계산 (_lec_bnf_exits)
o7 = np.array([100, 100, 101, 104, 109, 111, 108, 107, 106, 105.0])
c7 = np.array([100, 100.5, 103, 108, 110, 109, 107, 106, 105, 104.0])
ma7 = np.full(10, 108.5)
r7 = engine._lec_bnf_exits(o7, o7, o7, c7, ma7, 1, 100.0, 95.0)
chk("7 25일선 복귀(3번째 날 종가 108 < 108.5 아님, 4번째 날 110 ≥) → 다음 날 시가 111에 절반, 종가 107 < 25일선 → 다음 날 시가 107에 나머지",
    r7 is not None and r7["exit_kind"] == "익절 + 25일선 이탈" and close(r7["gross"], 0.5 * 111 / 100 + 0.5 * 107 / 100 - 1)
    and r7["h1"] == 4 and r7["h2"] == 6, r7)
c7b = np.array([100, 99, 99.5, 100.5, 101, 101, 101, 101, 101, 101.0])
r7b = engine._lec_bnf_exits(o7, o7, o7, c7b, ma7, 1, 100.0, 95.0)
chk("7-2 5거래일 안에 종가 ≥ 진입가 + 0.5R(102.5)이 없으면 6번째 날 시가에 시간 손절",
    r7b is not None and r7b["exit_kind"] == "시간 손절" and r7b["h2"] == 5 and close(r7b["gross"], o7[6] / 100 - 1), r7b)
c7c = np.array([100, 99, 94, 100, 101, 101, 101, 101, 101, 101.0])
r7c = engine._lec_bnf_exits(o7, o7, o7, c7c, ma7, 1, 100.0, 95.0)
chk("7-3 종가 94 < 투매 저점 95 → 다음 날 시가에 손절", r7c is not None and r7c["exit_kind"] == "손절" and r7c["h2"] == 2
    and close(r7c["gross"], o7[3] / 100 - 1), r7c)
chk("7-4 데이터가 끝날 때까지 안 끝난 거래는 None(판정에서 뺌)", engine._lec_bnf_exits(o7, o7, o7, np.full(10, 103.0), np.full(10, 200.0), 1, 100.0, 95.0) is None)


# 8) BNF 거래 찾기 — 만든 가격: 지그재그 상승(섹터 ETF보다 강함) → 대량 음봉 급락 → 홀드봉 2개 → 확인봉 → 25일선 복귀
def bnf_path():
    n0 = 820
    t = np.arange(n0)
    base = 100 * np.exp(0.0009 * t) * (1 + 0.03 * np.sin(2 * np.pi * t / 40))
    cl = list(base)
    for k in range(6):                               # 급락 6일(하루 −4%)
        cl.append(cl[-1] * 0.96)
    lo_crash = cl[-1]
    cl += [lo_crash * 1.002, lo_crash * 1.004]       # 홀드봉 2개
    cl += [lo_crash * 1.03]                          # 확인봉
    for k in range(25):                              # 회복
        cl.append(cl[-1] * 1.012)
    for k in range(10):
        cl.append(cl[-1] * 0.985)
    cl = np.array(cl)
    n = len(cl)
    o = np.r_[cl[0], cl[:-1]]
    hi = np.maximum(o, cl) * 1.002
    lo = np.minimum(o, cl) * 0.998
    v = np.full(n, 1e6)
    v[n0 + 2] = 5e6                                  # 급락 중 대량 음봉
    i_h1, i_h2, i_cf = n0 + 6, n0 + 7, n0 + 8
    lo[i_h1], lo[i_h2] = cl[n0 + 5] * 0.9995, cl[n0 + 5] * 1.0005   # 최저가(n0+5) 뒤 홀드봉 2개
    lo[n0 + 5] = cl[n0 + 5] * 0.997
    o[i_cf] = cl[i_h2]
    hi[i_h2] = cl[i_h2] * 1.003
    return frame(cl, v, o=o, hi=hi, lo=lo), i_cf


d8, i_cf = bnf_path()
etf8 = frame(100 * np.exp(0.0002 * np.arange(len(d8))))
b8 = engine.lec_base(d8)
sw8 = engine.lec_swings(b8)
rsw8 = engine.lec_rs_weekly_series(engine.lec_rs_line(b8["Close"], etf8["Close"]))
r8 = engine.lec_bnf_trades(b8, sw8, rsw8, None, "full")
tr8 = r8["trades"]
chk("8 확인봉 날 신호 1건", len(tr8) == 1 and int(tr8["t"].iloc[0]) == i_cf, (r8["skips"], tr8[["t"]].to_dict() if len(tr8) else None))
if len(tr8):
    x = tr8.iloc[0]
    p8 = max(0, i_cf - 40) + int(np.argmax(b8["Close"].to_numpy()[max(0, i_cf - 40):i_cf + 1]))
    chk("8-2 진입 = 확인봉 다음 날 시가, 손절 = 급락 시작일~확인봉 최저가",
        close(x["entry"], b8["Open"].iloc[i_cf + 1]) and close(x["stop"], b8["Low"].iloc[p8:i_cf + 1].min()), (x["entry"], x["stop"]))
    chk("8-3 실적일 자료 없음 → 악재 제외 '확인 불가'로 셈", r8["skips"]["earn_unknown"] == 1 and not bool(x["earn_known"]))
r8b = engine.lec_bnf_trades(b8, sw8, rsw8, {"dates": np.array([np.datetime64(b8.index[i_cf - 3])]), "cover_from": b8.index[0]}, "full")
chk("8-4 급락 시작일~확인봉 사이 실적 발표일이 있으면 제외(이어지는 날의 신호도 같은 이유로 제외)",
    len(r8b["trades"]) == 0 and r8b["skips"]["earn_excluded"] >= 1, r8b["skips"])
r8c = engine.lec_bnf_trades(b8, sw8, None, None, "full")
chk("8-5 섹터 ETF 자료가 없으면 조건 1을 판정할 수 없어 뺀다(데이터 없음)", len(r8c["trades"]) == 0 and r8c["skips"]["rs_missing"] >= 1)
weak_etf = frame(100 * np.exp(0.003 * np.arange(len(d8))))      # ETF가 훨씬 강함 → 종목 RS 소외
r8d = engine.lec_bnf_trades(b8, sw8, engine.lec_rs_weekly_series(engine.lec_rs_line(b8["Close"], weak_etf["Close"])), None, "full")
chk("8-6 급락 전 RS가 주도·약화가 아니면 뺀다", len(r8d["trades"]) == 0 and r8d["skips"]["rs_fail"] >= 1, r8d["skips"])
v_nc = d8.copy()
v_nc["Volume"] = 1e6
b8n = engine.lec_base(v_nc)
tr_no3 = engine.lec_bnf_trades(b8n, engine.lec_swings(b8n), rsw8, None, "no3")["trades"]
chk("8-7 거래량 폭증(음봉 RVOL ≥ 2)이 없으면 신호 없음, '조건 3 뺌' 변형은 급락 신호를 포함(평소 눌림도 잡힘)",
    len(engine.lec_bnf_trades(b8n, engine.lec_swings(b8n), rsw8, None, "full")["trades"]) == 0
    and i_cf in set(tr_no3["t"].astype(int)) and len(tr_no3) > 1)
# 미래 자료 미사용: 확인봉 날까지 자른 자료로도 같은 신호·같은 진입 전 값
b8t = engine.lec_base(d8.iloc[:i_cf + 1])
cut = engine.lec_bnf_trades(b8t, engine.lec_swings(b8t), rsw8[rsw8.index <= b8t.index[-1]], None, "full")
chk("8-8 확인봉 날까지 자른 자료: 신호는 찾지만 다음 날 시가가 없어 거래를 만들지 않음(미래 자료 미사용)",
    len(cut["trades"]) == 0)
b8t2 = engine.lec_base(d8.iloc[:i_cf + 30])
cut2 = engine.lec_bnf_trades(b8t2, engine.lec_swings(b8t2), rsw8[rsw8.index <= b8t2.index[-1]], None, "full")
chk("8-9 30일 뒤까지 자른 자료: 같은 신호일·진입가·손절가", len(cut2["trades"]) and len(tr8) and int(cut2["trades"]["t"].iloc[0]) == i_cf
    and close(cut2["trades"]["stop"].iloc[0], tr8["stop"].iloc[0]) or (len(cut2["trades"]) == 0 and cut2["skips"]["open_at_end"] == 1))

# 9) 비교 표본
if len(tr8):
    per = (100, len(b8) - 1)
    hb = engine.lec_bnf_hold_baseline(b8, tr8, (500, 500), 1)
    x = tr8.iloc[0]
    o9 = b8["Open"].to_numpy()
    exp9 = x["frac1"] * o9[501 + int(x["h1"])] / o9[501] + (1 - x["frac1"]) * o9[501 + int(x["h2"])] / o9[501] - 1 - LC["COST"]
    chk("9 보유 기간 맞춘 무작위: 고른 날 j=500 → 501 시가 매수, 같은 날수 뒤 시가에 같은 비율로 매도, 비용 0.30%",
        len(hb) == LC["BASELINE_SAMPLES"] and close(hb["ret"].iloc[0], exp9), (len(hb), hb["ret"].iloc[0] if len(hb) else None, exp9))
    rb = engine.lec_bnf_rebound_baseline(b8, tr8, per, 1)
    dr = b8["DROP5"].to_numpy()
    js = [int(np.flatnonzero(b8.index == d)[0]) for d in rb["date"]]
    chk("9-2 반등 비교 표본: 모두 최근 5일 변화가 ±0.5 ATR 안이고 신호일과 10거래일 넘게 떨어진 날",
        len(js) > 0 and all(abs(dr[j] - x["drop"]) <= LC["DROP_TOL"] + 1e-12 and abs(j - int(x["t"])) > LC["HORIZON"] for j in js))
    j0 = js[0]
    cl9 = b8["Close"].to_numpy()
    exp_reb = float(np.any(cl9[j0 + 1:j0 + 6] >= o9[j0 + 1] * (1 + 0.5 * x["stop_pct"])))
    chk("9-3 반등 비교: j+1 시가 대비 같은 % 거리(0.5 × 손절폭%)를 j+1~j+5 종가가 넘는지", close(rb["reb"].iloc[0], exp_reb))

# 10) 판정 도우미: 효과 없음 → 후보 아님, 효과 있음 → 후보, 한 달에 몰린 효과 → 달 쏠림으로 탈락
r10 = np.random.default_rng(10)


def fake(n, eff, months=150, conc=False):
    dts = pd.to_datetime("2002-01-15") + pd.to_timedelta(r10.integers(0, months * 30, n), unit="D")
    x = r10.normal(0, 0.03, n) + eff
    if conc:
        dts = dts.where(np.arange(n) >= 40, pd.Timestamp("2008-10-15"))
        x = np.where(np.arange(n) < 40, x + 0.25, x - eff)
    return pd.DataFrame({"date": dts, "ret": x, "ticker": [f"T{k % 80}" for k in range(n)], "market": np.where(np.arange(n) % 3 == 0, "KR", "US")})


def judge(a, bb):
    cm = engine.compare_clustered(a, bb, "ret")
    mk = {m: engine.compare_clustered(a[a["market"] == m], bb[bb["market"] == m], "ret") for m in ("US", "KR")}
    return engine.lec_judge(cm, a, bb, "ret", +1, a["ticker"].nunique(), mk, LC["SMALL_FAMILY_Z"])


base10 = fake(8000, 0.0)
j_none = judge(fake(600, 0.0), base10)
j_eff = judge(fake(600, 0.008), base10)
j_conc = judge(fake(600, 0.0, conc=True), base10)
chk("10 효과 없는 사건 → 후보 아님", not j_none["candidate"], j_none["reasons"])
chk("10-2 효과 +0.8%p(사건 600, 종목 80) → 후보", j_eff["candidate"], j_eff["reasons"])
chk("10-3 효과가 한 달(2008-10)에 몰림 → 그달을 빼면 기준 미달로 후보 아님",
    (not j_conc["candidate"]) and j_conc["month_drop"]["month"] == "2008-10", (j_conc["reasons"], j_conc["month_drop"]))
j_few = judge(fake(200, 0.02), base10)
chk("10-4 사건 200건 < 300 → 효과가 커도 후보 아님", not j_few["candidate"] and any("< 300" in s for s in j_few["reasons"]))
chk("10-5 검증: 같은 방향 + z ≥ 2", engine.lec_verify({"diff": 0.01, "z": 2.5, "actual": {"n": 400}}, +1)["pass"]
    and not engine.lec_verify({"diff": -0.01, "z": -2.5, "actual": {"n": 400}}, +1)["pass"])

# 11) L3: 초과수익 = 결과 − 같은 날 같은 시장 종목 평균 (손 계산)
fr = engine.lec_forward_returns(frame(100 + np.arange(30.0)), 10)
chk("11 다음 날 시가 → 10거래일 뒤 종가", close(fr.iloc[0], (100 + 10) / (100 + 0) - 1))
uni = {f"U{k}": walk(3800, 100 + k) for k in range(70)}
prep = engine.lec_prepare(uni, {}, {}, None, None, need_rs=False)
r11 = engine.run_lecture_l3(prep)
q11 = r11["questions"]["L3-a"]
ev11 = engine.lec_rsi_event_table(prep)
e11 = ev11.dropna(subset=["x10"]).iloc[7]
mean11 = np.nanmean([engine.lec_forward_returns(p["b"], 10).get(e11["date"], np.nan) for p in prep.values()])
chk("11-1 초과수익 = 다음 날 시가 → 10거래일 뒤 종가 − 같은 날 같은 시장 70종목의 같은 수익률 평균(손 계산)",
    close(e11["x10"], e11["r10"] - mean11), (e11["x10"], e11["r10"] - mean11))
chk("11-2 RSI 사건은 10거래일 안에 같은 종류를 다시 세지 않음",
    all((np.diff(engine.lec_rsi_events(p["b"], p["st"]).query("kind == 'low'")["t"].to_numpy()) > LC["GAP"]).all() for p in list(prep.values())[:10]))
chk("11-3 효과 없는 가격(무작위 걸음 70종목): L3-a·L3-b 모두 후보 아님",
    not r11["questions"]["L3-a"]["judge"]["candidate"] and not r11["questions"]["L3-b"]["judge"]["candidate"],
    {q: (v["cmp"].get("z"), v["judge"]["reasons"]) for q, v in r11["questions"].items()})
chk("11-4 후보가 아니면 검증(2014~)을 계산하지 않음", all(v["verify"] is None for v in r11["questions"].values() if not v["judge"]["candidate"]))

# 12) L1 전체 실행 (무작위 걸음 + 섹터 ETF): 후보 아님, 보고서·기록 줄
uni12 = {f"U{k}": walk(3800, 300 + k) for k in range(12)}
prep12 = engine.lec_prepare(uni12, {k: "Technology" for k in uni12}, {"XLK": walk(3800, 999)}, None, None)
r12 = engine.run_lecture_l1(prep12)
chk("12 효과 없는 가격의 L1: 두 질문 모두 후보 아님", not r12["judge"]["q1"]["candidate"] and not r12["judge"]["q2"]["candidate"])
chk("12-2 참고 변형 7개를 모두 계산", len(r12["refs"]) == 7)
rep = engine.render_lecture_l1_report(r12, "2026-11-18 12:00")
chk("12-3 보고서에 '증거가 아니다'·두 질문·참고 변형", "증거가 아니다" in rep and "질문" in rep and "참고 변형" in rep)
chk("12-4 기록 줄: L1 두 줄, L3 두 줄", len(engine.lec_log_lines(r12, "x")) == 2 and len(engine.lec_log_lines(r11, "x")) == 2)
with tempfile.TemporaryDirectory() as td:
    engine.lec_append_log(r12, "2026-11-18 12:00", td)
    txt = open(_os.path.join(td, engine.EXPLORE_CONFIG["LOG_FILE"]), encoding="utf-8").read()
    chk("12-5 EXPLORE_LOG.md에 한 줄씩 쌓임", txt.count("lec-L1-") == 2)

# 13) 실적 발표일: SEC 8-K Item 2.02만, 오래된 목록 파일까지 / DART '잠정'+'실적'만
with tempfile.TemporaryDirectory() as td:
    old = engine.SEC_CONFIG["CACHE_DIR"]
    engine.SEC_CONFIG["CACHE_DIR"] = td
    calls = []

    def fake_get(u):
        calls.append(u)
        if u.endswith("CIK0000000042.json"):
            return {"filings": {"recent": {"form": ["8-K", "8-K", "10-Q", "8-K/A"], "filingDate": ["2020-01-30", "2020-03-01", "2020-04-30", "2020-02-01"],
                                           "items": ["2.02,9.01", "5.02", "", "2.02"]},
                                "files": [{"name": "CIK0000000042-submissions-001.json"}]}}
        return {"form": ["8-K"], "filingDate": ["2006-07-20"], "items": ["2.02"]}
    try:
        ds = engine.lec_sec_earnings_dates(42, "test test@example.com", fetch=fake_get)
    finally:
        engine.SEC_CONFIG["CACHE_DIR"] = old
    chk("13 SEC: 8-K 중 Item 2.02만(정정 8-K/A·다른 항목·10-Q 제외), 추가 목록 파일 포함", ds == ["2006-07-20", "2020-01-30"], ds)


class FakeDart:
    def list(self, corp, start=None, end=None, kind=None):
        return pd.DataFrame({"report_nm": ["연결재무제표기준영업(잠정)실적(공정공시)", "주요사항보고서", "영업(잠정)실적(공정공시)", "기업설명회(IR)개최"],
                             "rcept_dt": ["20210107", "20210110", "20200408", "20200101"]})


chk("13-2 DART: '잠정'과 '실적'이 모두 들어간 보고서 접수일만", engine.lec_dart_earnings_dates("005930", FakeDart()) == ["2020-04-08", "2021-01-07"])

# 14) 설정: 사전 등록 값(결과 전 고정) 확인
chk("14 판정 기간 10거래일, 비용 0.30%, 탐색 2001~2013 · 검증 2014~, 최종 z ≥ 3",
    LC["HORIZON"] == 10 and close(LC["COST"], 0.003) and LC["EXPLORE_END"] == "2013-12-31" and LC["VERIFY_START"] == "2014-01-01"
    and close(LC["FINAL_Z"], 3.0) and LC["MIN_EVENTS"] == 300 and LC["MIN_TICKERS"] == 50)

print(f"\n{len(FAILS)}개 실패" if FAILS else "\n모두 통과")
sys.exit(1 if FAILS else 0)
