"""H-CS1 횡단면 모멘텀(engine/cs_backtest.py) — French 자료 읽기·대형 전체 가중, 12-1개월 신호·보유 수익 손 계산, 회전율·비용 손 계산,
미래 자료 미사용, 우위를 심은 가짜 자료는 통과 / 우위 없는 가짜 자료는 미통과, 판정 기준·보고서."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _offline  # noqa: F401
import io
import sys
import zipfile

import numpy as np
import pandas as pd

import engine

CF = engine.CS_CONFIG
FAILS = []


def check(name, cond, detail=""):
    print(("  ok  " if cond else "  FAIL") + f"  {name}" + (f"  — {detail}" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


def near(a, b, tol=1e-9):
    return a is not None and b is not None and np.isfinite(a) and abs(a - b) <= tol


# ---------------------------------------------------------------- 가짜 French 자료
def french_mom(months, bh, bn, bl, cols=("SMALL LoPRIOR", "ME1 PRIOR2", "SMALL HiPRIOR", "BIG LoPRIOR", "ME2 PRIOR2", "BIG HiPRIOR"),
               with_size=True):
    L = ["This file was created using the 202509 CRSP database.", "The portfolios are constructed monthly.", "",
         "  Average Value Weighted Returns -- Monthly", "," + ",".join(cols)]
    for m, a, b, c in zip(months, bh, bn, bl):
        L.append(f"{m},  1.00,  1.00,  1.00, {c:.4f}, {b:.4f}, {a:.4f}")
    L += ["", "  Average Equal Weighted Returns -- Monthly", "," + ",".join(cols)]
    for m in months:
        L.append(f"{m},  9.00,  9.00,  9.00,  9.00,  9.00,  9.00")
    if with_size:
        L += ["", "  Number of Firms in Portfolios", "," + ",".join(cols)]
        for m in months:
            L.append(f"{m},   10,   10,   10,   100,   200,   100")
        L += ["", "  Average Firm Size", "," + ",".join(cols)]
        for m in months:
            L.append(f"{m},   1.0,   1.0,   1.0,   10.0,   10.0,   30.0")
    L += ["", "  Annual Returns", "," + ",".join(cols), "  1927,  5.0,  5.0,  5.0,  5.0,  5.0,  5.0"]
    return "\n".join(L) + "\n"


def french_ff(months, mkt_rf, rf):
    L = ["This file was created by CMPT_ME_BEME_RETS using the 202509 CRSP database.", "", ",Mkt-RF,SMB,HML,RF"]
    for m, a, r in zip(months, mkt_rf, rf):
        L.append(f"{m},{a:8.2f},   0.10,   0.10,{r:8.2f}")
    L += ["", " Annual Factors: January-December ", ",Mkt-RF,SMB,HML,RF", "  1927,   10.0,  1.0,  1.0,  3.0"]
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- 1. French 읽기
print("1. Ken French 자료 읽기")
mo = ["192701", "192702", "192703"]
mt = french_mom(mo, [3.0, -99.99, 1.0], [2.0, 1.0, 0.5], [1.0, 0.0, -0.5])
ft = french_ff(mo, [2.0, 1.0, 0.0], [0.3, 0.3, 0.3])
secs = engine.cs_parse_french(mt)
check("1-1 표 나누기(월간 4개 + 연간 1개)", [s["freq"] for s in secs] == ["M", "M", "M", "M", "Y"], str([(s["title"], s["freq"]) for s in secs]))
fr = engine.cs_french_frame(mt, ft)
check("1-2 −99.99는 빈칸 → 그달 제외, %는 소수로", list(map(str, fr.index)) == ["1927-01", "1927-03"] and near(fr.loc[pd.Period("1927-01", "M"), "bh"], 0.03))
check("1-3 시장 = Mkt−RF + RF", near(fr.loc[pd.Period("1927-01", "M"), "mkt"], 0.023))
w = np.array([100 * 10.0, 200 * 10.0, 100 * 30.0])
check("1-4 대형 전체 = 세 대형을 기업 수 × 평균 크기로 가중", near(fr.loc[pd.Period("1927-01", "M"), "big"], (w @ np.array([0.01, 0.02, 0.03])) / w.sum()))
fr2 = engine.cs_french_frame(french_mom(mo, [3.0, 2.0, 1.0], [2.0, 1.0, 0.5], [1.0, 0.0, -0.5], cols=("SL", "SN", "SW", "BL", "BN", "BW"), with_size=False), ft)
check("1-5 선진국식 열 이름(BW)·크기 표 없으면 대형 전체 없음", near(fr2.iloc[0]["bh"], 0.03) and "big" not in fr2)
buf = io.BytesIO()
with zipfile.ZipFile(buf, "w") as z:
    z.writestr("6_Portfolios_ME_Prior_12_2.csv", mt)
check("1-6 zip 안 CSV 읽기", engine.cs_zip_text(buf.getvalue()).startswith("This file"))

# ---------------------------------------------------------------- 2. ① 판정
print("2. ① 대형 모멘텀 − 시장 − 비용, 판정")
rng = np.random.default_rng(1)
pm = pd.period_range("1927-01", "2025-12", freq="M")
months = [p.strftime("%Y%m") for p in pm]
mkt = rng.normal(0.8, 5.0, len(pm))
edge = np.where(pm.year < 2000, 0.6, 0.15)
bh = mkt + 0.3 + edge + rng.normal(0, 1.5, len(pm))
mt2 = french_mom(months, bh, mkt, mkt - 0.3)
ft2 = french_ff(months, mkt - 0.3, np.full(len(pm), 0.3))
p1 = engine.run_cs_part1(mt2, ft2, turnover=0.25)
x_hand = (np.round(bh, 4) - (np.round(mkt - 0.3, 2) + 0.3)) / 100 - 0.25 * 0.003   # 파일에 적힌 자릿수 그대로
check("2-1 비용 = 회전율 × 0.30%", near(p1["cost_per_month"], 0.00075))
check("2-2 전체 월평균 초과 손 계산", near(p1["full"]["mean"], float(np.mean(x_hand)), 1e-6), f"{p1['full']['mean']} vs {np.mean(x_hand)}")
check("2-3 2000년 이후만 따로", near(p1["recent"]["mean"], float(np.mean(x_hand[pm.year >= 2000])), 1e-6) and p1["recent"]["n"] == int((pm.year >= 2000).sum()))
check("2-4 우위가 있으면 통과", p1["judge"]["passed"] and p1["full"]["t"] > 2)
neg = np.where(pm.year < 2000, 0.6, -0.4)
p1n = engine.run_cs_part1(french_mom(months, mkt + 0.3 + neg + rng.normal(0, 1.5, len(pm)), mkt, mkt), ft2, turnover=0.25)
check("2-5 2000년 이후가 ≤ 0이면 미통과", not p1n["judge"]["passed"] and any("2000년" in x for x in p1n["judge"]["reasons"]))
nz = engine.cs_part1_noise(mt2, ft2, 0.25)
check("2-6 실행 전 단계는 흔들림만(평균 없음)", "mean" not in str(list(nz.keys())) and nz["mde_full"] < nz["mde_recent"] and nz["months_full"] == len(pm))
check("2-7 대형 전체 대비(참고)·기간별", "vs_big" in p1 and set(p1["periods"]) == {"1927~1963", "1964~1999", "2000~"})


# ---------------------------------------------------------------- 3. 종목 하나의 신호·보유 수익
print("3. 12-1개월 신호·보유 수익 손 계산")


def path(n, seed, drift=0.0004, vol=0.018, start="1999-01-04", s0=50.0):
    g = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n)
    c = s0 * np.exp(np.cumsum(g.normal(drift, vol, n)))
    o = c * np.exp(g.normal(0, 0.004, n))
    return pd.DataFrame({"Open": o, "High": np.maximum(o, c) * 1.01, "Low": np.minimum(o, c) * 0.99, "Close": c, "Volume": 1e6}, index=idx)


df = path(800, 3)
sm = engine.cs_stock_months(df)
cl, op, idx = df["Close"].to_numpy(), df["Open"].to_numpy(), df.index
row = sm[sm["month"] == pd.Period("2000-03", "M")].iloc[0]
i = int(np.flatnonzero(idx.to_period("M") == pd.Period("2000-03", "M"))[-1])
j = int(np.flatnonzero(idx.to_period("M") == pd.Period("2000-04", "M"))[-1])
check("3-1 모멘텀 = 21거래일 전 종가 ÷ 252거래일 전 종가 − 1", near(row["mom"], cl[i - 21] / cl[i - 252] - 1))
check("3-2 보유 = 달 끝 다음 날 시가 → 다음 달 끝 다음 날 시가", near(row["r"], op[j + 1] / op[i + 1] - 1))
lr = np.diff(np.log(cl), prepend=np.nan)
check("3-3 변동성 대비 = 모멘텀 ÷ 같은 구간 연 변동성", near(row["vadj"], row["mom"] / (np.std(lr[i - 251:i - 20], ddof=1) * np.sqrt(252)), 1e-9))
check("3-4 252거래일 전이 없으면 신호 없음", sm[sm["month"] == pd.Period("1999-06", "M")]["mom"].isna().all())
cut = df.iloc[:600]
sm_c = engine.cs_stock_months(cut).set_index("month")
sm_f = sm.set_index("month")
common = [m for m in sm_c.index if np.isfinite(sm_c.loc[m, "mom"])]
check("3-5 미래 자료 미사용: 자른 자료와 전체 자료의 신호가 같음", len(common) > 5 and np.allclose(sm_c.loc[common, "mom"], sm_f.loc[common, "mom"])
      and np.allclose(sm_c.loc[common, "vadj"], sm_f.loc[common, "vadj"]))
gap = df.drop(df.index[(df.index >= "2000-06-20") & (df.index <= "2000-06-30")])
check("3-6 달 끝 자료가 5일 넘게 비면 그달 건너뜀", pd.Period("2000-06", "M") not in set(engine.cs_stock_months(gap)["month"]))

# ---------------------------------------------------------------- 4. 회전율·비용
print("4. 포트폴리오·회전율·비용 손 계산")
tick = [f"S{k:02d}" for k in range(20)]
rows = []
for k, t in enumerate(tick):
    rows.append({"month": pd.Period("2001-01", "M"), "mom": k, "vadj": k, "r": 0.10 if k >= 16 else 0.0, "ticker": t, "market": "US"})
    rows.append({"month": pd.Period("2001-02", "M"), "mom": (k + 2) % 20, "vadj": 0.0, "r": 0.0, "ticker": t, "market": "US"})
panel = pd.DataFrame(rows)
pf = engine.cs_portfolio(panel, "US")
m1, m2 = pf.iloc[0], pf.iloc[1]
check("4-1 상위 20% = 백분위 > 80%(20종목 중 4개)", m1["top_n"] == 4 and near(m1["top_gross"], 0.10) and near(m1["ew_gross"], 0.02))
check("4-2 첫 달 회전율 0", m1["top_to"] == 0 and m1["ew_to"] == 0)
check("4-3 둘째 달: 상위 4개 중 2개 교체 → 회전율 0.5", near(m2["top_to"], 0.5), str(m2["top_to"]))
wd = np.array([1.1] * 4 + [1.0] * 16) / 20.4
check("4-4 같은 비중 쪽은 흘러간 비중만큼 되돌림", near(m2["ew_to"], float(np.maximum(1 / 20 - wd, 0).sum())))
check("4-5 비용 뒤 x = (상위 − 회전율 × 0.30%) − (전체 − 회전율 × 0.30%)", near(m2["x"], (0 - 0.5 * 0.003) - (0 - m2["ew_to"] * 0.003)))
small = panel[panel["ticker"].isin(tick[:19])]
check("4-6 대상 20개 미만이면 그달 건너뜀", len(engine.cs_portfolio(small, "US")) == 0)


# ---------------------------------------------------------------- 5. 우위를 심은 / 없는 가짜 자료
print("5. 우위를 심은 가짜 자료는 통과, 없는 자료는 미통과(판정 기계 점검)")


def fake_panel(edge, n_us=98, n_kr=53, seed=7):
    g = np.random.default_rng(seed)
    pm2 = pd.period_range("2001-01", "2025-12", freq="M")
    out = []
    for mk, nn in (("US", n_us), ("KR", n_kr)):
        for k in range(nn):
            e = g.normal(0, 0.3, len(pm2))
            mom = np.zeros(len(pm2))
            for t in range(len(pm2)):   # 모멘텀은 달마다 천천히 바뀜(실제처럼 회전율이 너무 크지 않게)
                mom[t] = (0.9 * mom[t - 1] if t else 0.0) + np.sqrt(1 - 0.81) * e[t]
            r = g.normal(0.008, 0.08, len(pm2)) + edge * mom
            out.append(pd.DataFrame({"month": pm2, "mom": mom, "vadj": mom, "r": r, "ticker": f"{mk}{k}", "market": mk}))
    return pd.concat(out, ignore_index=True)


for nm, e, want in (("심음", 0.08, True), ("없음", 0.0, False)):
    pp = fake_panel(e)
    us_, kr_ = engine.cs_portfolio(pp, "US"), engine.cs_portfolio(pp, "KR")
    s_ = engine.cs_stats(engine.cs_combined(us_, kr_))
    g_ = engine.cs_stats(engine.cs_combined(us_, kr_, "top_gross") - engine.cs_combined(us_, kr_, "ew_gross"))
    j_ = engine.cs_judge_part2(s_, engine.cs_stats(us_["x"]), engine.cs_stats(kr_["x"]))
    check(f"5-{nm}: {'통과' if want else '미통과'}", j_["passed"] == want and (s_["t"] > 3 if want else abs(g_["t"]) < 2.5),
          f"t {s_['t']} 비용 전 t {g_['t']} {j_['reasons']}")
check("5-비용: 우위가 없으면 회전율만큼 손해(비용 뒤 < 비용 전)", s_["mean"] < g_["mean"])

# ---------------------------------------------------------------- 6. 판정 기준
print("6. ② 판정 기준")
ok = {"n": 300, "mean": 0.003, "t": 2.2, "first": 0.002, "second": 0.004}
us_ok, kr_ok = {"n": 300, "mean": 0.003}, {"n": 300, "mean": 0.002}
check("6-1 모두 맞으면 통과", engine.cs_judge_part2(ok, us_ok, kr_ok)["passed"])
for nm, patch, word in (("t", {"t": 1.9}, "t <"), ("절반", {"first": -0.001}, "절반"), ("평균", {"mean": -0.001}, "≤ 0")):
    jj = engine.cs_judge_part2({**ok, **patch}, us_ok, kr_ok)
    check(f"6-{nm}", not jj["passed"] and any(word in x for x in jj["reasons"]), str(jj["reasons"]))
check("6-시장 방향(둘 다 60개월 이상일 때만)", not engine.cs_judge_part2(ok, us_ok, {"n": 300, "mean": -0.001})["passed"]
      and engine.cs_judge_part2(ok, us_ok, {"n": 59, "mean": -0.001})["passed"])

# ---------------------------------------------------------------- 7. 실행 전·전체 흐름·보고서
print("7. 실행 전 단계·전체 흐름·보고서")
pmap = {f"T{k}": path(7000, 100 + k) for k in range(24)}
pmap.update({f"{100000 + k}.KS": path(7000, 300 + k, s0=30000.0) for k in range(22)})
cnt = engine.run_cs_count(pmap, pmap, mt2, ft2)
check("7-1 회전율 추정(0~1)·부분집합 최소 차이·대상 달 수", 0 < cnt["seen"]["turnover"]["US"] < 1 and cnt["seen"]["mde_part2"] > 0
      and cnt["eligible"]["US"]["months_ok"] > 100 and cnt["part1"]["mde_full"] > 0, str(cnt["seen"]))
rep_c = engine.render_cs_count_report(cnt, {"100000.KS": "테스트"}, "test")
check("7-2 실행 전 보고서(평균 없음·이름 확인)", "가려낼 수 있는 최소 차이" in rep_c and "100000.KS  테스트" in rep_c and "월평균 초과" not in rep_c)
idx_map = {"^GSPC": path(7000, 900), "^KS11": path(7000, 901, s0=2000.0)}
res = engine.run_cs(pmap, idx_map, cnt["seen"]["turnover"]["US"], mt2, ft2, mt2, ft2, french_date="2026-11-20")
check("7-3 전체 흐름: ① 통과(심은 우위) · ② 무작위 걸음은 미통과", res["part1"]["judge"]["passed"] and not res["part2"]["judge"]["passed"]
      and res["verdict"].startswith("① 통과"), res["verdict"])
check("7-4 참고: 상위 10%·변동성 대비·지수 대비·세금 예시", {"top10", "vadj", "vs_index_US", "vs_index_KR"} <= set(res["part2"]["refs"])
      and res["part2"]["tax_us"]["years"] > 10)
check("7-5 판정 시작 달 = 2001-01", res["part2"]["stats"]["start"] == "2001-01")
rep = engine.render_cs_report(res, "test")
check("7-6 결과 보고서", "## 판정: ① 통과" in rep and "선진국(미국 제외)" in rep and "세금 예시" in rep and len(engine.cs_log_lines(res, "t")) == 2)

print(f"\n{'통과' if not FAILS else '실패'}: {len(FAILS)}개 실패", FAILS)
sys.exit(1 if FAILS else 0)
