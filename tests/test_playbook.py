"""매매 틀 v2 판정(engine/playbook.py) 검사.

1 추세 상태(상승·하락·횡보), 지표 손 계산, 그날 이후 가격을 바꿔도 판정 그대로
2 근거 저울 항목: 추정치 건수, 자기 밴드 경계, 돌발 이벤트 −2, 데이터 없음은 중립
3 전환 가능성 표시(상승 추세에서는 뒤집어 하락 전환 가능성)
4 진입 의견(50일선 위 필수), 크기(+3 이상 1배·+2 절반), 게이트, 추격 주의, 청산(손절·매도)
5 틀 B 단계와 우선순위(손절 > 매도 > 경고 > 재평가 > 진입), 진입 뒤 배수·이익 분해, 피어 범위
6 실제 데이터 함수(가짜 yfinance): 자기 밴드 z, 한국은 계산하지 않음, 오늘 판정
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # 프로젝트 폴더의 engine 패키지를 불러오기 위해
import _offline  # noqa: F401  외부 네트워크 차단 (tests/_offline.py)
import math
import sys
import types
import numpy as np
import pandas as pd
import engine

FAILS = []


def chk(n, c):
    print(("PASS " if c else "FAIL ") + n, flush=True)
    (None if c else FAILS.append(n))


def path(lp, start="2023-01-02", seed=0, noise=0.004):
    idx = pd.bdate_range(start, periods=len(lp))
    c = 100 * np.exp(np.asarray(lp) + np.random.default_rng(seed).normal(0, noise, len(lp)))
    return pd.DataFrame({"Open": c, "High": c * 1.01, "Low": c * 0.99, "Close": c}, index=idx)


def zig(points, n=420, start="2024-01-01", seed=0, noise=0.003):
    """꺾이는 점(위치, 가격)을 잇는 가짜 경로."""
    idx = pd.bdate_range(start, periods=n)
    pos, val = zip(*points)
    c = np.interp(np.arange(n), pos, val) * np.exp(np.random.default_rng(seed).normal(0, noise, n))
    return pd.DataFrame({"Open": c, "High": c * 1.01, "Low": c * 0.99, "Close": c}, index=idx)


t = np.arange(400)
up_z = zig([(0, 100), (60, 130), (90, 112), (150, 150), (180, 128), (240, 175), (270, 150), (330, 200), (360, 176), (419, 215)])
dn_z = zig([(0, 200), (60, 160), (90, 185), (150, 140), (180, 165), (240, 120), (270, 140), (330, 100), (360, 118), (419, 92)], seed=1)
brk_z = zig([(0, 100), (60, 130), (90, 112), (150, 150), (180, 128), (240, 175), (270, 152), (330, 196), (419, 150)], seed=2)
smooth = path(0.002 * t)
spy = pd.DataFrame({"Close": 100 * np.exp(0.0003 * np.arange(420))}, index=up_z.index)
up = up_z

# ---------------------------------------------------------------------------
# 1 스윙·추세 상태·지표
# ---------------------------------------------------------------------------
h = np.array([10, 11, 12, 11, 9, 8, 9, 10, 12, 13.0])
l = h - 0.5
sp = engine.swing_points(h, l, np.ones(10), k_atr=3.0)
# 고점 12(2번 날)에서 저가 8.5(4번 날)로 3 이상 되돌림 → 4번 날 고점 확정. 저점 7.5(5번 날)에서 고가 12(8번 날)로 4.5 → 8번 날 저점 확정
chk("1-0 스윙 손 계산(3×ATR 되돌림이 나온 날 확정)",
    [(x["kind"], x["idx"], x["price"], x["confirm"]) for x in sp] == [("H", 2, 12.0, 4), ("L", 5, 7.5, 8)])
chk("1-1 저점이 높아지는 지그재그 → 상승", engine.trend_indicators(up_z)["state"].iloc[-1] == "상승")
chk("1-2 고점·저점이 낮아지는 지그재그 → 하락", engine.trend_indicators(dn_z)["state"].iloc[-1] == "하락")
ib = engine.trend_indicators(brk_z)
chk("1-3 상승 추세선을 깼지만 고점 낮아짐 확인 전 → 횡보(상승 추세선 이탈)", ib["state"].iloc[-1] == "횡보"
    and "상승 추세선 이탈" in ib["state_detail"].iloc[-1])
ism = engine.trend_indicators(smooth)
chk("1-4 스윙이 모자라면 이동평균 배열로 대신 판단하고 표시", ism["state"].iloc[-1] == "상승" and "스윙 부족" in ism["state_detail"].iloc[-1]
    and ism["state"].iloc[100] is None)
ind = engine.trend_indicators(up, spy)
k = 300
c, lo = up["Close"], up["Low"]
chk("1-5 50일선 기울기 = 50일선 − 20거래일 전 50일선",
    abs(ind["slope50"].iloc[k] - (c.iloc[k - 49:k + 1].mean() - c.iloc[k - 69:k - 19].mean())) < 1e-9)
L1, L2 = ind["sw_L1"].iloc[k], ind["sw_L2"].iloc[k]
exp_line = math.exp(math.log(L2["price"]) + (math.log(L2["price"]) - math.log(L1["price"])) / (L2["idx"] - L1["idx"]) * (k - L2["idx"]))
chk("1-6 저점 연결선 = 두 스윙 저점을 잇는 로그 직선의 그날 값, 확정일이 그날 이전",
    abs(ind["line_low"].iloc[k] - exp_line) < 1e-9 and L2["confirm"] <= k)
chk("1-7 3개월 상대강도", abs(ind["rs63"].iloc[k] - ((c.iloc[k] / c.iloc[k - 63] - 1) - (spy["Close"].iloc[k] / spy["Close"].iloc[k - 63] - 1))) < 1e-12)
fut = up.copy()
fut.iloc[k + 1:, :4] *= 0.3
r1 = engine.frame_a_evaluate(up, spy, at=up.index[k])
r2 = engine.frame_a_evaluate(fut, spy, at=up.index[k])
chk("1-8 그날 이후 가격을 바꿔도 추세 상태·근거 그대로(스윙은 확정일 이후에만 씀)", r1["state"] == r2["state"]
    and r1["state_detail"] == r2["state_detail"] and r1["diff"] == r2["diff"]
    and [i["sign"] for i in r1["items"]] == [i["sign"] for i in r2["items"]])
chk("1-9 가격 이력 부족은 판정 불가", engine.frame_a_evaluate(up.iloc[:30])["available"] is False)
rd = engine.frame_a_evaluate(dn_z)
txt = engine.render_frame_a_text({**rd, "ticker": "X"})
chk("1-10 근거 표시: 스윙 저점·고점, 연결선 값, 확정 전 진행 중, 이동평균 참고",
    "스윙 저점" in txt and "고점 연결선 오늘 값" in txt and "확정 전 진행 중" in txt and "참고: 50일선" in txt)
chk("1-11 하락 세부 문구", rd["state_detail"].startswith("하락 ("))

# ---------------------------------------------------------------------------
# 2 근거 저울 항목
# ---------------------------------------------------------------------------
ci = engine.consensus_item
chk("2-1 추정치 상향 우세 +1·하향 우세 −1·같음 0·없음 데이터 없음",
    ci({"eps_fy1_up_30d": 5, "eps_fy1_down_30d": 2})["sign"] == 1 and ci({"eps_fy1_up_30d": "1", "eps_fy1_down_30d": 3})["sign"] == -1
    and ci({"eps_fy1_up_30d": 2, "eps_fy1_down_30d": 2})["sign"] == 0 and ci({"eps_fy1_up_30d": None})["no_data"]
    and ci(None)["points"] == 0)
bi = engine.band_item
chk("2-2 자기 밴드 −1σ 이하 +1, +1σ 이상 −1, 사이 0, 없음 데이터 없음",
    bi(-1.0)["sign"] == 1 and bi(-0.99)["sign"] == 0 and bi(1.0)["sign"] == -1 and bi(None)["no_data"] and bi(float("nan"))["no_data"])
g = up.copy()
d0 = g.index[k - 2]
g.loc[d0:, ["Open", "High", "Low", "Close"]] *= 0.94     # 6% 갭 하락 후 그 수준 유지
g.loc[d0, "Open"] = g.loc[d0, "Close"]
ev = engine.frame_a_evaluate(g, at=g.index[k])
e_item = next(i for i in ev["items"] if i["key"] == "event")
chk("2-3 최근 5거래일 −5% 갭 하락 → 돌발 이벤트 −2", e_item["points"] == -2)
ev7 = engine.frame_a_evaluate(g, at=g.index[k + 6])
chk("2-4 5거래일이 지나면 돌발 이벤트 해제", next(i for i in ev7["items"] if i["key"] == "event")["points"] == 0)
chk("2-5 합계 = 항목 점수 합 (같은 사실 두 번 세지 않음: 항목 7개)",
    len(ev["items"]) == 7 and ev["support"] - ev["against"] == sum(i["points"] for i in ev["items"]) == ev["diff"])

# ---------------------------------------------------------------------------
# 3 전환 가능성 표시
# ---------------------------------------------------------------------------
L = engine._lean
chk("3-1 하락·횡보: +2 우세, −2 실패 우세, 사이 팽팽",
    L("하락", 2)["label"] == "전환 가능성 우세" and L("횡보", -2)["label"] == "실패 가능성 우세" and L("하락", 1)["label"] == "팽팽")
chk("3-2 상승: 저울을 뒤집어 하락 전환 가능성", L("상승", -2)["label"] == "하락 전환 가능성 우세"
    and L("상승", 3)["label"] == "상승 추세 유지 우세" and L("상승", 0)["direction"] == "하락 전환 가능성")


# ---------------------------------------------------------------------------
# 4 진입 의견·크기·청산 (지표 표를 직접 넣음)
# ---------------------------------------------------------------------------
def one(state="횡보", close=110.0, sma50=100.0, slope20=1.0, low_recent=95.0, low_prior=90.0, rs=0.05, atr=2.0, sma20=105.0, event=False):
    """지표 한 줄. low_prior → low_recent = 앞 스윙 저점 → 나중 스윙 저점."""
    return pd.DataFrame({"Open": [close], "High": [close], "Low": [close], "Close": [close], "SMA200": [90.0], "SMA50": [sma50],
                         "SMA20": [sma20], "slope50": [0.1], "slope20": [slope20],
                         "sw_L1": [{"kind": "L", "idx": 0, "price": low_prior, "confirm": 0}],
                         "sw_L2": [{"kind": "L", "idx": 0, "price": low_recent, "confirm": 0}],
                         "sw_H1": [None], "sw_H2": [None], "line_low": [np.nan], "line_high": [np.nan], "ma_state": [state],
                         "ATR20": [atr], "rs63": [rs], "event": [event], "state": [state], "state_detail": [state]},
                        index=[pd.Timestamp("2026-11-13")])


up_cs = {"eps_fy1_up_30d": 4, "eps_fy1_down_30d": 1}
r = engine.frame_a_evaluate(None, ind=one(), consensus=up_cs, gate_level=1)
chk("4-1 횡보 + 차이 +5 + 50일선 위 → 상승 전환 의견(강), 크기 1배, G1 절반",
    r["opinion"]["entry"] and r["diff"] == 5 and r["opinion"]["size_mult"] == 1.0 and r["opinion"]["final_mult"] == 0.5
    and r["opinion"]["label"] == "상승 전환 의견 (강)")
chk("4-2 추격 주의: 종가 110 > 20일선 105 + 2×ATR 4", r["opinion"]["chase_caution"] and any("추격" in n for n in r["opinion"]["notes"]))
chk("4-3 변동성 단위 = 2×ATR", r["opinion"]["unit_risk"] == 4.0 and abs(r["opinion"]["unit_risk_pct"] - 4 / 110) < 1e-12)
r = engine.frame_a_evaluate(None, ind=one(close=99.0, sma20=98.0), consensus=up_cs, band_z=-1.5)
chk("4-4 차이가 커도(+4) 50일선 아래면 진입 의견 없음", r["diff"] == 4 and not r["opinion"]["entry"])
r = engine.frame_a_evaluate(None, ind=one(rs=-0.01, low_recent=89.0), consensus=up_cs)
chk("4-5 차이 +1(상대강도·저점이 실패 쪽) → 진입 의견 없음", r["diff"] == 1 and not r["opinion"]["entry"])
r = engine.frame_a_evaluate(None, ind=one(rs=-0.01), consensus={"eps_fy1_up_30d": 1, "eps_fy1_down_30d": 1})
chk("4-5b 정확히 +2면 진입·절반", r["diff"] == 2 and r["opinion"]["entry"] and r["opinion"]["size_mult"] == 0.5)
r = engine.frame_a_evaluate(None, ind=one(), consensus=up_cs, gate_level=2)
chk("4-6 G2는 크기 0 + 의견만 표시", r["opinion"]["entry"] and r["opinion"]["final_mult"] == 0.0 and any("G2" in n for n in r["opinion"]["notes"]))
r = engine.frame_a_evaluate(None, ind=one(state="상승"), consensus=up_cs)
chk("4-7 상승 추세에서는 진입 의견 없음(이미 추세)", not r["opinion"]["entry"] and r["pos_word"] == "유지 근거")
r = engine.frame_a_evaluate(None, ind=one(close=95.0, sma50=100.0, slope20=-1, low_recent=85.0, rs=-0.02, sma20=97.0),
                            position={"entry_price": 100.0})
chk("4-8 하락 쪽 근거가 더 많고 손실 → 손절(전환 실패 우세)", r["position"]["exit"] and r["position"]["label"].startswith("손절"))
r = engine.frame_a_evaluate(None, ind=one(close=120.0, sma50=125.0, slope20=-1, low_recent=85.0, rs=-0.02, sma20=122.0),
                            position={"entry_price": 100.0})
chk("4-9 하락 쪽 근거가 더 많고 이익 → 매도(하락 전환 우세)", r["position"]["exit"] and r["position"]["label"].startswith("매도"))
r = engine.frame_a_evaluate(None, ind=one(close=99.0, sma20=98.0, rs=-0.01, low_recent=95.0), position={"entry_price": 100.0})
chk("4-10 양쪽이 같으면 보유", r["support"] == r["against"] and not r["position"]["exit"])
r = engine.frame_a_evaluate(None, ind=one(), consensus=None, band_z=None)
chk("4-11 데이터 없음 항목은 중립 + 안내", any("데이터 없음" in n for n in r["opinion"]["notes"]) and "검증" not in r["opinion"]["label"]
    and "미검증" in r["verification"])


# ---------------------------------------------------------------------------
# 5 틀 B 단계
# ---------------------------------------------------------------------------
def fund(pq="QUALIFIED", z=-1.2, peer=None, mult=0.05, eps=0.1, cur_pe=12.0):
    return {"price_qualification": {"status": pq},
            "triple_z_score": {"self_5y_z": z, "peer_rel_z": peer, "current_per_trailing": cur_pe},
            "multiple_decomposition": {"multiple_contribution_12m": mult, "eps_contribution_12m": eps}}


ok_cs = {"eps_fy1_up_30d": 3, "eps_fy1_down_30d": 1, "eps_fy1_avg": 5.0, "eps_fy1_30d_ago": 4.9}
dn_cs = {"eps_fy1_up_30d": 1, "eps_fy1_down_30d": 4, "eps_fy1_avg": 4.8, "eps_fy1_30d_ago": 5.0}
fb = engine.frame_b_evaluate
chk("5-1 매수선 + 실적 유지 + 하향 아님 → 진입 가능", fb(fund(), {}, ok_cs, 5.0, 4.5)["stage"] == "진입 가능")
chk("5-2 조건부 가격 자격 → 조건부 진입", fb(fund("CONDITIONAL"), {}, ok_cs, 5.0, 4.5)["stage"] == "조건부 진입")
chk("5-3 추정치 하향 우세 → 경고(신규 매수 중단)", fb(fund(), {}, dn_cs, 5.0, 4.5)["stage"] == "경고")
chk("5-4 하향 우세 + 가격 자격 미충족 → 손절 조건", fb(fund("DISQUALIFIED", z=0.2), {}, dn_cs, 5.0, 4.5)["stage"] == "손절 조건")
chk("5-5 발표 실적 감소 → 손절 조건", fb(fund(), {}, ok_cs, 4.0, 4.5)["stage"] == "손절 조건")
kill = {"hard_kill_gate": {"is_killed": True, "kill_reason": "G5"}, "v_action_card": {"price_qual_code": "HARD_KILL"}}
chk("5-6 킬 조건은 고평가 매도보다 우선", fb(fund(z=1.5, mult=0.3, eps=0.0), kill, ok_cs, 5.0, 4.5)["stage"] == "손절 조건")
chk("5-7 고평가 + 배수 주도 → 매도 조건", fb(fund("DISQUALIFIED", z=1.2, mult=0.3, eps=0.1), {}, ok_cs, 5.0, 4.5)["stage"] == "매도 조건")
chk("5-8 고평가 + 이익 주도 → 재평가 보유", fb(fund("DISQUALIFIED", z=1.2, mult=0.05, eps=0.3), {}, ok_cs, 5.0, 4.5)["stage"] == "재평가 보유")
r = fb(fund("DISQUALIFIED", z=1.2, mult=-0.5, eps=0.9, cur_pe=15.0), {}, ok_cs, 2.1, 2.0, position={"entry_pe": 10.0, "entry_eps": 2.0})
chk("5-9 보유 중이면 진입 뒤 분해(ln 1.5 배수 > ln 1.05 이익)", r["decomposition_basis"] == "진입 뒤" and r["mult_driven"] is True
    and r["stage"] == "매도 조건")
r = fb(fund(z=None, peer=-0.8), {}, ok_cs, 5.0, 4.5)
chk("5-10 자기 밴드 불가 → 피어 기준 + 미검증 표시", r["scope"].startswith("자기 밴드 판정 불가") and "피어 기준" in r["verification"])
r = fb(fund("NA", z=None), None, None, None, None)
chk("5-11 자료 부족 → 대기 + 빠진 항목 목록", r["stage"] == "대기" and len(r["missing"]) >= 3)
chk("5-12 추정치 하향 우세 정의(건수와 값 둘 다)", engine.estimates_down(dn_cs) is True
    and engine.estimates_down({**dn_cs, "eps_fy1_avg": 5.1}) is False and engine.estimates_down({"eps_fy1_up_30d": 1}) is None)
r = fb(fund(), {}, ok_cs, 5.0, 4.5, trend_row={"Close": 90.0, "SMA200": 100.0, "slope50": -1.0})
chk("5-13 하락 추세 중이면 분할 간격을 넓힘(수익 신호 아님)", r["technical_note"].startswith("하락 추세 중") and r["stage"] == "진입 가능")

# ---------------------------------------------------------------------------
# 6 실제 데이터 함수 (가짜 yfinance)
# ---------------------------------------------------------------------------
hidx = pd.bdate_range("2020-11-02", "2026-11-13")
hc = pd.Series(100 * np.exp(np.random.default_rng(5).normal(0, 0.003, len(hidx)).cumsum()), index=hidx)
inc = pd.DataFrame({pd.Timestamp(f"{y}-12-31"): [5.0 + 0.05 * (y - 2019)] for y in range(2019, 2026)}, index=["Diluted EPS"])


class FakeTicker:
    def __init__(self, sym):
        self.sym = sym
        self.income_stmt = inc
        self.info = {"trailingEps": 5.4}

    def history(self, period=None, auto_adjust=False):
        return pd.DataFrame({"Close": hc})


def fake_download(sym, **kw):
    s = up if sym == "AAA" else spy.assign(Open=spy["Close"], High=spy["Close"], Low=spy["Close"])
    return s


fyf = types.SimpleNamespace(Ticker=FakeTicker, download=fake_download)
bz = engine.light_band_z("AAA", yf=fyf)
prices = engine._month_end_prices(pd.DataFrame({"Close": hc}))
band = engine._lagged_ratio_stats(prices, engine._annual_pairs(inc, ["Diluted EPS"]), engine.FUND_DATA_PARAMS["band_lag_days_us"], 5, 24)
exp_z = round((math.log(prices[-1][1] / 5.4) - math.log(band["mean"])) / band["log_std"], 2)
chk("6-1 자기 밴드 z (앱 V 트랙과 같은 밴드 함수)", bz["z"] == exp_z and bz["reason"] is None)
chk("6-2 한국 종목은 계산하지 않고 이유 표시", engine.light_band_z("005930.KS", yf=fyf)["z"] is None)
bad = types.SimpleNamespace(Ticker=lambda s: (_ for _ in ()).throw(RuntimeError("x")))
chk("6-3 실패하면 이유와 함께 None", engine.light_band_z("AAA", yf=bad)["z"] is None and "실패" in engine.light_band_z("AAA", yf=bad)["reason"])
live = engine.fetch_frame_a_live("AAA", yf=fyf, consensus=up_cs, gate_level=0)
chk("6-4 오늘 판정(가짜 데이터): 상승 추세, 가치 항목 반영", live["available"] and live["state"] == "상승" and live["ticker"] == "AAA"
    and next(i for i in live["items"] if i["key"] == "band")["sign"] is not None)
txt = engine.render_frame_a_text(live)
chk("6-5 요약 문구에 추세·근거·검증 상태", "추세: 상승" in txt and "유지 근거" in txt and "검증 상태: 미검증" in txt)

print(f"\n{'모두 통과' if not FAILS else f'실패 {len(FAILS)}개: ' + ', '.join(FAILS)}")
sys.exit(1 if FAILS else 0)
