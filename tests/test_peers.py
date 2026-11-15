"""피어 비교 v1 (engine/peers.py) 손 계산 검사 — 백분위·요인·전략 점수·걸러내기·묶음 만들기·하루 실행(가짜 yfinance)."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _offline  # noqa: F401
import math, os, sys, tempfile
import numpy as np
import pandas as pd
import engine
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))

FAILS = []


def chk(n, c, extra=""):
    print(("PASS " if c else "FAIL ") + n + (f"  [{extra}]" if extra and not c else ""))
    if not c:
        FAILS.append(n)


close = lambda a, b, tol=1e-9: a is not None and b is not None and abs(float(a) - float(b)) < tol

# 1) 백분위: 같은 값은 평균 순위, 빈칸 제외, 방향
p = engine.percentile_ranks(pd.Series([10, 20, 30, None, 20], index=list("abcde")), 1)
chk("1 백분위(클수록 좋음): 10→0, 20→50, 30→100, 빈칸 NaN", close(p["a"], 0) and close(p["b"], 50) and close(p["e"], 50)
    and close(p["c"], 100) and math.isnan(p["d"]), p.tolist())
q = engine.percentile_ranks(pd.Series([10, 20, 30], index=list("abc")), -1)
chk("1-2 백분위(작을수록 좋음): 10→100", close(q["a"], 100) and close(q["c"], 0))
chk("1-3 값 1개면 순위 없음", engine.percentile_ranks(pd.Series([5.0, None]), 1).isna().all())

# 2) 지표 만들기
fund = {"rev_ttm": 120.0, "rev_q0": 33.0, "rev_q4": 27.5, "op_income_ttm": 30.0, "roe": 0.25, "eps_fy1": 5.0, "eps_fy2": 6.0,
        "n_fy1": 20, "up_30d": 8, "down_30d": 2, "fy1_30d_ago": 4.0, "fwd_pe_info": None, "band_z": -1.2, "price": 90.0}
dm = engine.derive_metrics(fund, {"price": 100.0, "rs63": 0.05, "trend_state": "상승"})
chk("2 매출 성장(최근 분기 33 ÷ 전년 동기 27.5 − 1 = 20%), 영업이익률 30 ÷ 120 = 25%", close(dm["rev_growth_ttm"], 0.2) and close(dm["op_margin_ttm"], 0.25))
fb = engine.derive_metrics({"rev_growth_info": 0.07, "op_margin_info": 0.11, "rev_ttm": None}, {})
chk("2-1 분기 자료가 모자라면 야후 값(매출 성장 7%, 영업이익률 11%)", close(fb["rev_growth_ttm"], 0.07) and close(fb["op_margin_ttm"], 0.11))
chk("2-2 선행 EPS 성장 20%, 순상향 (8−2)/20 = 0.3, FY1 30일 변화 +25%",
    close(dm["eps_growth_fwd"], 0.2) and close(dm["rev_net_30d"], 0.3) and close(dm["fy1_change_30d"], 0.25))
chk("2-3 선행 PER = 오늘 가격 100 ÷ FY1 5 = 20 (야후 값 없을 때), PEG = 20 ÷ 20 = 1.0",
    close(dm["fwd_pe"], 20.0) and close(dm["peg"], 1.0), (dm["fwd_pe"], dm["peg"]))
chk("2-4 상대강도·추세 점수(상승 2)", close(dm["rs63"], 0.05) and close(dm["state_pt"], 2.0))
neg = engine.derive_metrics({**fund, "eps_fy1": -1.0, "eps_fy2": 2.0, "fwd_pe_info": -15.0}, {"price": 100.0})
chk("2-5 적자 EPS: PER·PEG·선행 성장 없음", neg["fwd_pe"] is None and neg["peg"] is None and neg["eps_growth_fwd"] is None)
dec = engine.derive_metrics({**fund, "eps_fy2": 4.0, "fwd_pe_info": 18.0}, {"price": 100.0})
chk("2-6 이익 감소 예상(성장 ≤ 0): PEG 없음, PER은 야후 값 18", dec["peg"] is None and close(dec["fwd_pe"], 18.0))

# 3) 전략 점수
S = engine.strategy_score
chk("3 핵심 0 + 보조 100·100 = 50 (핵심이 최하위면 중간을 못 넘음)",
    close(S({"valuation": 0.0, "profitability": 100.0, "revisions": 100.0}, "value")[0], 50.0))
sc, nt = S({"valuation": 0.0, "profitability": 100.0}, "value")
chk("3-2 보조 하나 없음: 50:25 → 67:33 재정규화 = 33.33, 자료 부족 표시", close(sc, 100.0 / 3.0) and "실적 모멘텀" in nt, (sc, nt))
chk("3-3 핵심 없음 → 순위 없음", S({"profitability": 90.0, "revisions": 90.0}, "value")[0] is None)
chk("3-4 핵심만 있음(요인 1개) → 순위 없음", S({"price": 90.0}, "trend")[0] is None)
chk("3-5 추세: 가격 80·실적 모멘텀 40·성장 60 = 0.5×80 + 0.25×40 + 0.25×60 = 65",
    close(S({"price": 80.0, "revisions": 40.0, "growth": 60.0}, "trend")[0], 65.0))

# 4) 묶음 점수: 6종목, 1종목 생존성 하드 킬 → 제외, 5종목 순위, 상위 1/3 = 2종목
base = []
for i, t in enumerate(["A", "B", "C", "D", "E", "K"]):
    base.append({"ticker": t, "fwd_pe": [10, 15, 20, 25, 30, 5][i], "peg": None, "band_z": None,
                 "op_margin_ttm": [0.30, 0.25, 0.20, 0.15, 0.10, 0.5][i], "roe": None,
                 "rev_net_30d": [0.1, 0.2, 0.3, 0.4, 0.5, 0.9][i], "fy1_change_30d": None,
                 "rev_growth_ttm": None, "eps_growth_fwd": None, "rs63": [0.1, 0.0, -0.1, 0.2, 0.3, 0.5][i], "state_pt": None,
                 "surv_kill": t == "K"})
g = engine.score_group(pd.DataFrame(base).set_index("ticker", drop=False))
# 가치: 밸류(PER 낮을수록) A100 B75 C50 D25 E0, 수익성 A100..E0, 모멘텀 A0 B25 C50 D75 E100
# A = .5×100 + .25×100 + .25×0 = 75, B = 37.5+18.75+6.25 = 62.5, C = 50, D = 37.5, E = 25
chk("4 가치 점수 손 계산(A 75, B 62.5, E 25)", close(g.loc["A", "value_score"], 75) and close(g.loc["B", "value_score"], 62.5)
    and close(g.loc["E", "value_score"], 25), g["value_score"].tolist())
chk("4-2 하드 킬 종목은 순위 제외(백분위 계산에서도 빠짐)", math.isnan(g.loc["K", "value_score"]) and "하드 킬" in g.loc["K", "value_note"])
chk("4-3 순위·상위 1/3(5종목 → 2종목)", g.loc["A", "value_rank"] == 1 and g.loc["B", "value_top"] is True
    and g.loc["C", "value_top"] is False and int(g.loc["A", "value_n"]) == 5)
small = engine.score_group(pd.DataFrame(base[:4]).set_index("ticker", drop=False))
chk("4-4 점수 있는 종목 4개 < 5 → 순위 없음 + '피어 부족'", small["value_rank"].isna().all() and "피어 부족" in small.loc["A", "value_note"])

# 5) 묶음 만들기
info = {"AAA": {"industryKey": "semis", "industry": "Semiconductors"}, "BBB": {"industryKey": "semis", "industry": "Semiconductors"},
        "CCC": {"industryKey": "semis"}, "DDD": {}}
built = engine.build_groups(["AAA", "BBB", "CCC", "DDD", "005930.KS", "EEE"], info,
                            lambda key: ["P1", "P2", "EEE", "AAA", "P3"] if key == "semis" else [],
                            {"CCC": {"group": "my", "group_name": "내 묶음", "note": ""}, "EEE": {"group": "-", "group_name": "-", "note": ""}})
G = built["groups"]
chk("5 야후 산업 묶음 = 커버리지(AAA·BBB) + 산업 상위(P1·P2·P3), 수동 지정(CCC)·제외(EEE)는 빠짐",
    G["yf:semis"]["members"] == ["AAA", "BBB", "P1", "P2", "P3"], G["yf:semis"]["members"])
chk("5-2 수동 묶음(CCC → my)", G["my"]["members"] == ["CCC"] and G["my"]["source"] == "수동")
chk("5-3 묶음 없음 사유(산업 분류 없음·한국·제외)", "산업 분류 없음" in built["no_group"]["DDD"] and "한국" in built["no_group"]["005930.KS"]
    and "제외" in built["no_group"]["EEE"])

# 6) 하루 실행 (가짜 yfinance)
DATES = pd.bdate_range(end="2026-11-13", periods=520)


def ohlc(series):
    c = pd.Series(series, index=DATES, dtype=float)
    return pd.DataFrame({"Open": c, "High": c * 1.01, "Low": c * 0.99, "Close": c, "Adj Close": c, "Volume": 1e6})


GROWTH = {"AAA": 1.001, "BBB": 1.0005, "P1": 1.0, "P2": 0.9995, "P3": 1.0015, "BAD": 0.999, "XLK": 1.0, "SPY": 1.0, "^KS11": 1.0}
PX = {s: ohlc(100.0 * (g ** np.arange(len(DATES)))) for s, g in GROWTH.items()}
Q = pd.date_range(end="2026-09-30", periods=8, freq="QE")[::-1]


def stmt(rows):
    return pd.DataFrame({d: {k: v[i] for k, v in rows.items()} for i, d in enumerate(Q)})


class FakeTicker:
    def __init__(self, s):
        self.s = s
        k = list(GROWTH).index(s) if s in GROWTH else 0
        self.k = k

    @property
    def info(self):
        return {"industryKey": "semis", "industry": "Semiconductors", "sector": "Technology", "currency": "USD",
                "forwardPE": [20, 25, 30, 35, 15, 40][self.k % 6], "returnOnEquity": [0.3, 0.2, 0.1, 0.15, 0.25, -0.2][self.k % 6]}

    @property
    def earnings_estimate(self):
        return pd.DataFrame({"avg": [1.0, 1.1, 5.0 + self.k, 6.0 + self.k], "numberOfAnalysts": [10, 10, 20, 18]},
                            index=["0q", "+1q", "0y", "+1y"])

    @property
    def eps_trend(self):
        return pd.DataFrame({"current": [1, 1, 5.0 + self.k, 6.0 + self.k], "30daysAgo": [1, 1, 4.8 + self.k, 5.9 + self.k]},
                            index=["0q", "+1q", "0y", "+1y"])

    @property
    def eps_revisions(self):
        return pd.DataFrame({"upLast30days": [1, 1, 5 + self.k, 3], "downLast30days": [0, 0, 2, 1]}, index=["0q", "+1q", "0y", "+1y"])

    @property
    def calendar(self):
        return {"Earnings Date": [pd.Timestamp("2027-01-28").date()]}

    @property
    def quarterly_income_stmt(self):
        r = [110 + self.k, 108, 105, 103, 100, 98, 96, 94]
        return stmt({"Total Revenue": r, "Operating Income": [x * (0.1 + 0.02 * self.k) for x in r],
                     "Diluted Average Shares": [1000.0] * 8})

    @property
    def quarterly_balance_sheet(self):
        bad = self.s == "BAD"
        return stmt({"Cash Cash Equivalents And Short Term Investments": [5.0 if bad else 500.0] * 8,
                     "Total Debt": [300.0] * 8, "Current Debt And Capital Lease Obligation": [50.0] * 8})

    @property
    def quarterly_cashflow(self):
        bad = self.s == "BAD"
        return stmt({"Operating Cash Flow": [-20.0 if bad else 40.0] * 8, "Capital Expenditure": [-10.0] * 8,
                     "Free Cash Flow": [-30.0 if bad else 30.0] * 8})

    @property
    def income_stmt(self):
        return pd.DataFrame()

    def history(self, period="5y", auto_adjust=False, **kw):
        return PX.get(self.s, PX["SPY"])


class FakeIndustry:
    def __init__(self, key):
        self.top_companies = pd.DataFrame({"name": ["x"] * 4}, index=["P1", "P2", "P3", "BAD"])


class FakeYF:
    Ticker = FakeTicker
    Industry = FakeIndustry
    calls = {"download": 0}

    @staticmethod
    def download(syms, period="2y", group_by=None, **kw):
        FakeYF.calls["download"] += 1
        if isinstance(syms, str):
            return PX.get(syms, pd.DataFrame())
        frames = {s: PX[s] for s in syms if s in PX}
        return pd.concat(frames, axis=1) if frames else pd.DataFrame()


tmp = tempfile.mkdtemp(prefix="peers_test_")
r1 = engine.run_daily_peers(today="2026-11-16", base_dir=tmp, yf=FakeYF, tickers=["AAA", "BBB"], pause=0.0)
print("  1차 실행:", {k: r1[k] for k in ("refreshed", "fund_saved", "rank_saved", "n_groups", "n_members")})
chk("6 1차: 재무 6종목(커버리지 2 + 자동 피어 4) 저장·묶음 1개·순위 6줄", r1["refreshed"] and r1["fund_saved"] == 6
    and r1["n_groups"] == 1 and r1["rank_saved"] == 6, r1)
rk = engine.load_peer_ranks(tmp)
bad = rk[rk["ticker"] == "BAD"].iloc[0]
chk("6-2 생존성 하드 킬 종목(BAD: 영업현금 적자·현금 소진)은 순위 제외", str(bad["surv_kill"]).lower() == "true" and pd.isna(bad["value_score"])
    and "하드 킬" in str(bad["value_note"]))
aaa = rk[rk["ticker"] == "AAA"].iloc[0]
chk("6-3 상대강도 = 63일 수익률 − 섹터 ETF(XLK, 보합) = 1.001^63 − 1", close(aaa["rs63"], 1.001 ** 63 - 1, 1e-6), aaa["rs63"])
chk("6-4 5종목 순위(점수 있는 종목 5개)", int(aaa["value_n"]) == 5 and not pd.isna(aaa["value_rank"]))
chk("6-5 커버리지 표시(AAA 참, P1 거짓)", str(aaa["is_coverage"]).lower() == "true"
    and str(rk[rk["ticker"] == "P1"].iloc[0]["is_coverage"]).lower() == "false")
r2 = engine.run_daily_peers(today="2026-11-16", base_dir=tmp, yf=FakeYF, tickers=["AAA", "BBB"], pause=0.0)
chk("6-6 같은 날 다시 실행: 중복 저장 없음", r2["fund_saved"] == 0 and r2["rank_saved"] == 0, r2)
_orig_uni = engine.snapshot_universe
engine.snapshot_universe = lambda d=None: ["AAA", "BBB"]
try:
    r2b = engine.run_daily_peers(today="2026-11-17", base_dir=tmp, yf=FakeYF, holdings_fn=lambda e: None, pause=0.0)
    chk("6-6b 종목 지정(부분) 갱신만 있었으면 다음 매일 실행이 전체 갱신", r2b["refreshed"] and r2b.get("refresh_mode") == "full"
        and r2b["fund_date"] == "2026-11-17", r2b)
    r3 = engine.run_daily_peers(today="2026-11-19", base_dir=tmp, yf=FakeYF, holdings_fn=lambda e: None, pause=0.0)
    chk("6-7 2일 뒤: 재무는 그대로(주 1회), 순위만 새로 기록", not r3["refreshed"] and r3["rank_saved"] == 6 and r3["fund_date"] == "2026-11-17", r3)
    r4 = engine.run_daily_peers(today="2026-11-24", base_dir=tmp, yf=FakeYF, holdings_fn=lambda e: None, pause=0.0)
    chk("6-8 마지막 전체 갱신 7일 뒤: 재무 다시 받음", r4["refreshed"] and r4["fund_saved"] == 6 and r4["fund_date"] == "2026-11-24", r4)
finally:
    engine.snapshot_universe = _orig_uni
v = engine.latest_peer_view("AAA", base_dir=tmp)
chk("6-9 딥다이브용 보기: 최근 날짜·같은 묶음 6종목", v["available"] and len(v["table"]) == 6 and str(v["rank_date"].date()) == "2026-11-24")
lg = pd.read_csv(engine.refresh_log_path(tmp), dtype=str)
chk("6-9b 갱신 기록: 부분 2번(같은 날 재실행 포함) → 전체 2번", lg["mode"].tolist() == ["partial", "partial", "full", "full"], lg["mode"].tolist())
chk("6-10 커버리지 밖 종목 안내", not engine.latest_peer_view("ZZZ", base_dir=tmp)["available"]
    and "watchlist" in engine.latest_peer_view("ZZZ", base_dir=tmp)["reason"])
fd = engine.load_peer_fund(tmp)
chk("6-11 다음 실적 발표일 기록", (fd["next_earnings_date"].astype(str) == "2027-01-28").all())

# 6-12) 열이 늘어난 기록 파일: 기존 줄은 그대로, 빈 새 열을 붙여 다시 쓴 뒤 이어 붙임
mig = os.path.join(tmp, "mig", "peer_fund_2026.csv")
os.makedirs(os.path.dirname(mig), exist_ok=True)
oldcols = engine.FUND_COLUMNS[:-4]
pd.DataFrame([{c: ("X" if c == "ticker" else ("2026-11-15" if c == "fund_date" else "")) for c in oldcols}], columns=oldcols).to_csv(
    mig, index=False, encoding="utf-8-sig")
engine.peers._append(mig, [{"fund_date": "2026-11-16", "ticker": "Y", "rev_q0": 5.0}], engine.FUND_COLUMNS, ["fund_date", "ticker"])
mg = pd.read_csv(mig, encoding="utf-8-sig")
chk("6-12 열 추가 이전: 머리글 = 새 열 목록, 기존 줄 보존, 새 줄의 새 열 값", list(mg.columns) == engine.FUND_COLUMNS and len(mg) == 2
    and mg["ticker"].tolist() == ["X", "Y"] and pd.isna(mg.loc[0, "rev_q0"]) and mg.loc[1, "rev_q0"] == 5.0, list(mg.columns)[-5:])

# 7) peer_map 읽기: 설명 줄(#)·제외(-)
os.makedirs(tmp, exist_ok=True)
with open(os.path.join(tmp, "peer_map.csv"), "w", encoding="utf-8-sig") as f:
    f.write("ticker,group,group_name,note\n# 설명 줄\n005930.KS,kr_semi,한국 반도체,\nP2,-,,이상한 피어\n")
pm = engine.load_peer_map(tmp)
chk("7 peer_map: 설명 줄 무시, 그룹·제외 읽기", pm["005930.KS"]["group_name"] == "한국 반도체" and pm["P2"]["group"] == "-" and len(pm) == 2, pm)

# 8) 딥다이브 요약 (강점 ≥ 60, 약점 ≤ 40, 피어 중앙값은 하드 킬 제외)
from _fake_peers import write_fake_peers
ft = tempfile.mkdtemp(prefix="peers_sum_")
write_fake_peers(ft, "NVDA")
sm = engine.peer_summary(engine.latest_peer_view("NVDA", base_dir=ft))
print("  요약:", sm["value"], sm["leaders"], [(r["label"], round(r["score"])) for r in sm["factors"]])
chk("8 요약: 묶음 6종목, 가치 2위/5(하드 킬 제외), 1위 PC", sm["n"] == 6 and sm["value"]["rank"] == 2 and sm["value"]["n"] == 5
    and sm["leaders"]["value"] == "PC" and sm["value"]["top"] is True)
st_ = {r["label"] for r in sm["strengths"]}
wk_ = {r["label"] for r in sm["weaknesses"]}
chk("8-2 강점 = 점수 60 이상(성장·수익성·밸류에이션), 약점 = 40 이하(가격 흐름), 실적 모멘텀 42는 보통",
    st_ == {"성장", "수익성", "밸류에이션"} and wk_ == {"가격 흐름"}, (st_, wk_))
grow = [r for r in sm["factors"] if r["factor"] == "growth"][0]
chk("8-3 피어 중앙값은 하드 킬 종목(PK) 빼고: 매출 성장 중앙 +10.0%", "피어 중앙 +10.0%" in grow["detail"], grow["detail"])
chk("8-4 순서: 강점은 높은 순(성장 100 먼저)", sm["strengths"][0]["score"] >= sm["strengths"][-1]["score"])
chk("8-5 다음 실적 발표일·버전", sm["next_earnings_date"] == "2027-01-28" and sm["version"] == engine.PEER_CONFIG["VERSION"])
chk("8-6 기록 없으면 요약도 없음(이유 전달)", not engine.peer_summary(engine.latest_peer_view("NVDA", base_dir=tempfile.mkdtemp()))["available"])

print(f"\n결과: {'통과' if not FAILS else '실패 ' + str(FAILS)}")
sys.exit(1 if FAILS else 0)
