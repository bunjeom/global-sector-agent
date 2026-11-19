"""매매 기록(engine/journal.py) 검사 — 저장 위치(PC만)·입력 검사·손익 손 계산·지수 대비·채점 규칙·손절 준수(갭)·해외주식 세금·불러오기."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _offline  # noqa: F401
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

os.environ["JOURNAL_MODE"] = "local"
import engine
import engine.journal as J

FAILS = []


def check(name, cond, detail=""):
    print(("  ok  " if cond else "  FAIL") + f"  {name}" + (f"  — {detail}" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


def near(a, b, tol=1e-9):
    return a is not None and b is not None and np.isfinite(a) and abs(a - b) <= tol


def raises(fn, *a, **k):
    try:
        fn(*a, **k)
    except (ValueError, PermissionError) as e:
        return str(e)
    return None


TODAY = "2026-11-20"
BUY = dict(ticker="nvda", style="추세", buy_date="2026-11-02", buy_price=100.0, qty=10, buy_fee=1.0, buy_fx=1400.0,
           stop_plan=92.0, reason="돌파 확정")


def fresh():
    return tempfile.mkdtemp(prefix="journal_")


# ---------------------------------------------------------------- 1. 저장 위치
print("1. 저장 위치(PC만)")
os.environ["JOURNAL_MODE"] = ""
_root = J.PROJECT_ROOT
J.PROJECT_ROOT = "/mount/src/global-sector-agent"
check("1-1 인터넷 서버 경로(/mount/src)면 보기 전용", engine.journal_storage_mode() == "view")
check("1-2 보기 전용이면 저장 거부", raises(engine.journal_save, engine.journal_empty(), fresh()) is not None)
J.PROJECT_ROOT = r"C:\Users\hanju\global-sector-agent"
check("1-3 PC 경로면 저장", engine.journal_storage_mode() == "local")
os.environ["JOURNAL_MODE"] = "view"
check("1-4 JOURNAL_MODE=view로 바꿀 수 있음", engine.journal_storage_mode() == "view")
os.environ["JOURNAL_MODE"] = "local"
J.PROJECT_ROOT = _root
check("1-5 기본 저장 폴더 = 저장소/data/journal", os.path.normpath(engine.JOURNAL_CONFIG["DIR"]) ==
      os.path.normpath(os.path.join(_root, "data", "journal")))
try:
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    r = subprocess.run(["git", "check-ignore", "-q", "data/journal/trades.csv"], cwd=repo, capture_output=True)
    check("1-6 data/journal/은 git이 무시함(공개 저장소에 안 올라감)", r.returncode == 0, f"returncode {r.returncode}")
except FileNotFoundError:
    print("  skip  1-6 git 없음")

# ---------------------------------------------------------------- 2. 산 기록 검사
print("2. 산 기록 검사(진입 근거·손절가·수량 필수)")
d = fresh()
for name, patch, word in [("손절 없음", {"stop_plan": None}, "손절"), ("손절 ≥ 산 값", {"stop_plan": 100.0}, "낮아야"),
                          ("근거 없음", {"reason": " "}, "근거"), ("수량 0", {"qty": 0}, "수량"), ("방식 틀림", {"style": "스윙"}, "방식"),
                          ("미래 날짜", {"buy_date": "2026-11-21"}, "오늘보다 뒤"), ("미국 환율 없음", {"buy_fx": None}, "환율")]:
    msg = raises(engine.journal_add_trade, {**BUY, **patch}, base_dir=d, today=TODAY)
    check(f"2-{name}", msg is not None and word in msg, str(msg))
t1 = engine.journal_add_trade(BUY, base_dir=d, today=TODAY, now="x")
t2 = engine.journal_add_trade({**BUY, "ticker": "005930.KS", "style": "가치", "buy_fx": 777.0, "buy_price": 70000, "stop_plan": 65000}, base_dir=d, today=TODAY)
t3 = engine.journal_add_trade(BUY, base_dir=d, today=TODAY)
data = engine.journal_load(d)
tr = data["trades"].set_index("id")
check("2-8 id = T날짜-순번", (t1, t3) == ("T20261102-001", "T20261102-003"), f"{t1} {t3}")
check("2-9 티커 대문자·시장·통화", tr.loc[t1, "ticker"] == "NVDA" and tr.loc[t1, "market"] == "US" and tr.loc[t1, "currency"] == "USD")
check("2-10 한국 종목 환율은 1", tr.loc[t2, "market"] == "KR" and tr.loc[t2, "buy_fx"] == 1.0)
check("2-11 판정 없이 적으면 '당일 판정 없음'", tr.loc[t1, "verdict_source"] == "당일 판정 없음" and tr.loc[t1, "version"] == "jr-1")

# ---------------------------------------------------------------- 3. 그날 판정
print("3. 딥다이브 판정 저장")
fit = {"headline": "지금 이 종목에 맞는 방식: 추세", "closest": None,
       "lines": [{"key": "value", "verdict": "안 맞음"}, {"key": "trend", "verdict": "맞음"}, {"key": "short", "verdict": "부적합"}]}
s0 = engine.journal_verdict_snapshot(fit, "2026-11-13", "2026-11-17")
s5 = engine.journal_verdict_snapshot(fit, "2026-11-13", "2026-11-18")
check("3-1 자료 날짜 + 4일 안이면 저장", s0["verdict_source"] == "딥다이브" and s0["verdict_trend"] == "맞음" and s0["verdict_date"] == "2026-11-13")
check("3-2 5일 이상 떨어지면 당일 판정 없음", s5["verdict_source"] == "당일 판정 없음" and s5["verdict_trend"] == "")
check("3-3 자료보다 앞선 날·판정 없음", engine.journal_verdict_snapshot(fit, "2026-11-13", "2026-11-12")["verdict_source"] == "당일 판정 없음"
      and engine.journal_verdict_snapshot(None, "2026-11-13", "2026-11-13")["verdict_source"] == "당일 판정 없음")
m = lambda st, **k: engine.journal_verdict_match({"verdict_source": "딥다이브", "style": st, **k})
check("3-4 판정 일치: 추세 '맞음' · 단타 '조합 충족'", m("추세", verdict_trend="맞음") is True and m("가치", verdict_value="안 맞음") is False
      and m("단타", verdict_short="주의 · 조합 충족(미검증)") is True and m("단타", verdict_short="주의 · 조합 미충족") is False
      and engine.journal_verdict_match({"verdict_source": "당일 판정 없음", "style": "추세"}) is None)

# ---------------------------------------------------------------- 4. 판 기록
print("4. 판 기록(나눠 팔기·증권거래세)")
check("4-1 남은 수량보다 많이", "많습니다" in (raises(engine.journal_add_sell, t1, {"sell_date": "2026-11-10", "sell_price": 1, "qty": 11, "sell_fx": 1}, base_dir=d, today=TODAY) or ""))
check("4-2 산 날보다 앞", "앞입니다" in (raises(engine.journal_add_sell, t1, {"sell_date": "2026-11-01", "sell_price": 1, "qty": 1, "sell_fx": 1}, base_dir=d, today=TODAY) or ""))
check("4-3 없는 산 기록", raises(engine.journal_add_sell, "T0", {"sell_date": "2026-11-10", "sell_price": 1, "qty": 1}, base_dir=d, today=TODAY) is not None)
sa = engine.journal_add_sell(t1, {"sell_date": "2026-11-05", "sell_price": 90.0, "qty": 4, "sell_fee": 0.5, "sell_fx": 1410.0}, base_dir=d, today=TODAY)
sb = engine.journal_add_sell(t1, {"sell_date": "2026-11-09", "sell_price": 110.0, "qty": 6, "sell_fee": 0.5, "sell_fx": 1420.0}, base_dir=d, today=TODAY)
sk = engine.journal_add_sell(t2, {"sell_date": "2026-11-12", "sell_price": 60000, "qty": 10}, base_dir=d, today=TODAY)
data = engine.journal_load(d)
se = data["sells"].set_index("id")
check("4-4 국내 판 기록 세금 자동 = 판 금액 × 0.20%(2026)", near(se.loc[sk, "sell_tax"], 600000 * 0.0020, 1e-6), str(se.loc[sk, "sell_tax"]))
check("4-5 증권거래세 연도별", [engine.journal_kr_sell_tax_rate(f"{y}-06-01") for y in (2019, 2021, 2022, 2024, 2025, 2026, 2030)]
      == [0.0025, 0.0023, 0.0023, 0.0018, 0.0015, 0.0020, 0.0020])
pos = engine.journal_positions(data).set_index("id")
check("4-6 상태: 청산·보유", pos.loc[t1, "status"] == "청산" and pos.loc[t3, "status"] == "보유" and pos.loc[t1, "remaining"] == 0)

# ---------------------------------------------------------------- 5. 손익 손 계산
print("5. 손익·지수 대비 손 계산")
bench = {"US": pd.Series([500.0, 490.0, 520.0], index=pd.to_datetime(["2026-11-02", "2026-11-05", "2026-11-09"]).tz_localize("America/New_York")),
         "KR": pd.DataFrame({"Close": [2600.0, 2500.0]}, index=pd.to_datetime(["2026-11-02", "2026-11-12"]))}
res = engine.journal_results(data, bench=bench).set_index("id")
cost = 100 * 10 + 1.0
pnl = (90 * 4 + 110 * 6) - 1.0 - cost
idx = (4 * (490 / 500 - 1) + 6 * (520 / 500 - 1)) / 10
check("5-1 비용 포함 수익률", near(res.loc[t1, "ret"], pnl / cost), f"{res.loc[t1, 'ret']} vs {pnl / cost}")
check("5-2 비용 전 = 평균 판 값 ÷ 산 값 − 1", near(res.loc[t1, "gross_ret"], (1020 / 10) / 100 - 1))
check("5-3 지수 = 판 수량 가중", near(res.loc[t1, "idx_ret"], idx) and near(res.loc[t1, "excess"], pnl / cost - idx))
check("5-4 보유일·이익", res.loc[t1, "hold_days"] == 7 and res.loc[t1, "win"] is True and res.loc[t1, "close_date"] == "2026-11-09")
kr_ret = (600000 - 1200 - 700001) / 700001   # 산 수수료 1원
check("5-5 국내: 증권거래세 포함·KOSPI 대비", near(res.loc[t2, "ret"], kr_ret) and near(res.loc[t2, "excess"], kr_ret - (2500 / 2600 - 1)))
check("5-6 보유 중은 비어 있음", not np.isfinite(res.loc[t3, "ret"]) and res.loc[t3, "stop_status"] == "")
res_nb = engine.journal_results(data, bench={}).set_index("id")
check("5-7 지수 자료 없으면 초과 빈칸(수익률은 있음)", np.isfinite(res_nb.loc[t1, "ret"]) and not np.isfinite(res_nb.loc[t1, "excess"]))

# ---------------------------------------------------------------- 6. 손절 준수
print("6. 손절 준수(갭은 따로)")
ss = engine.journal_stop_status
ohlc = pd.DataFrame({"Open": [100, 99, 95, 88, 87], "Low": [98, 97, 94, 86, 85]},
                    index=pd.to_datetime(["2026-11-02", "2026-11-03", "2026-11-04", "2026-11-05", "2026-11-06"]))
check("6-1 이익 거래는 세지 않음", ss(100, 92, 0.05, 0.04, "2026-11-02", ["2026-11-05"]) == "이익")
check("6-2 계획 − 1%p 안이면 준수", ss(100, 92, -0.089, -0.09, "2026-11-02", ["2026-11-05"]) == "준수")
check("6-3 넘었고 가격 없음 → 어김(갭 확인 불가)", ss(100, 92, -0.12, -0.12, "2026-11-02", ["2026-11-05"]) == "어김(갭 확인 불가)")
check("6-4 처음 닿은 날 시가가 손절 아래 + 그날 다 팖 → 갭", ss(100, 92, -0.12, -0.12, "2026-11-02", ["2026-11-05"], ohlc) == "갭")
check("6-5 닿은 날 뒤까지 들고 있었음 → 어김", ss(100, 92, -0.13, -0.13, "2026-11-02", ["2026-11-06"], ohlc) == "어김")
ohlc2 = ohlc.copy()
ohlc2.loc[pd.Timestamp("2026-11-05"), "Open"] = 93.0
check("6-6 시가는 손절 위였는데 넘김 → 어김", ss(100, 92, -0.12, -0.12, "2026-11-02", ["2026-11-05"], ohlc2) == "어김")
check("6-7 손절 계획 없음", ss(100, float("nan"), -0.12, -0.12, "2026-11-02", ["2026-11-05"]) == "계획 없음")
check("6-8 국내 거래(−14.5%, 계획 −7.1%) → 어김(갭 확인 불가)", res.loc[t2, "stop_status"] == "어김(갭 확인 불가)")

# ---------------------------------------------------------------- 7. 채점 규칙
print("7. 채점 규칙(30건 · t ≥ 2 · 월 묶음)")
x = np.array([0.1, -0.1, 0.2])
g = np.array([1, 1, 2])
mm = x.mean()
se_hand = np.sqrt(2 / 1 * (((0.1 - mm) + (-0.1 - mm)) ** 2 + (0.2 - mm) ** 2)) / 3
check("7-1 월 묶음 t 손 계산", near(J.lec_cluster_t(x, g)["t"], mm / se_hand, 1e-12))
rng = np.random.default_rng(3)
months = np.repeat(np.arange(15), 2)
j29 = engine.journal_judge(np.full(29, 0.05), months[:29])
check("7-2 29건이면 기록 중", j29["verdict"] == "기록 중 (29/30)")
jp = engine.journal_judge(0.04 + rng.normal(0, 0.02, 30), months)
jn = engine.journal_judge(-0.04 + rng.normal(0, 0.02, 30), months)
jz = engine.journal_judge(rng.normal(0, 0.08, 30) - 0.0, months)
check("7-3 뚜렷이 위 → 지수보다 나음", jp["verdict"] == "지수보다 나음" and jp["t"] >= 2)
check("7-4 뚜렷이 아래 → 지수 보유가 나음", jn["verdict"] == "지수 보유가 나음")
check("7-5 흔들림 큼 → 아직 구분 안 됨", jz["verdict"] == "아직 구분 안 됨" and abs(jz["t"]) < 2, str(jz["t"]))
check("7-6 빈칸 초과는 세지 않음", engine.journal_judge(np.r_[np.full(30, 0.05), [np.nan]], np.r_[months, [99]])["n"] == 30)
summ = {s["style"]: s for s in engine.journal_style_summary(engine.journal_results(data, bench=bench))}
check("7-7 방식별 요약: 추세 끝난 1·보유 1, 가치 손절 어김 1", summ["추세"]["n_closed"] == 1 and summ["추세"]["n_open"] == 1
      and summ["가치"]["stop_violated"] == 1 and summ["가치"]["stop_rate"] == 0.0 and summ["전체"]["n_closed"] == 2
      and summ["단타"]["verdict"] == "기록 중 (0/30)")

# ---------------------------------------------------------------- 8. 해외주식 세금
print("8. 해외주식 세금(결제일 해 · 손익 합산 · 250만 원 공제 · 22%)")
d8 = fresh()
a = engine.journal_add_trade({**BUY, "buy_price": 100.0, "qty": 100, "buy_fee": 0.0, "buy_fx": 1300.0, "buy_date": "2026-01-05"}, base_dir=d8, today=TODAY)
b = engine.journal_add_trade({**BUY, "ticker": "AAPL", "buy_price": 200.0, "qty": 10, "buy_fee": 2.0, "buy_fx": 1300.0, "buy_date": "2026-01-05"}, base_dir=d8, today=TODAY)
k = engine.journal_add_trade({**BUY, "ticker": "000660.KS", "buy_price": 100000, "stop_plan": 90000, "buy_date": "2026-01-05"}, base_dir=d8, today=TODAY)
engine.journal_add_sell(a, {"sell_date": "2026-06-01", "sell_price": 150.0, "qty": 100, "sell_fee": 5.0, "sell_fx": 1400.0}, base_dir=d8, today=TODAY)
engine.journal_add_sell(b, {"sell_date": "2026-06-02", "sell_price": 150.0, "qty": 10, "sell_fee": 1.0, "sell_fx": 1400.0}, base_dir=d8, today=TODAY)
engine.journal_add_sell(k, {"sell_date": "2026-06-02", "sell_price": 200000, "qty": 10}, base_dir=d8, today=TODAY)
tx = engine.journal_tax_overseas(engine.journal_load(d8))
ga = 100 * 150 * 1400 - 100 * 100 * 1300 - 5 * 1400
gb = 10 * 150 * 1400 - 10 * 200 * 1300 - 2 * 1300 - 1 * 1400
row = tx["rows"][0]
check("8-1 국내 종목은 빼고, 미국 손익 합산", len(tx["rows"]) == 1 and row["n_sells"] == 2 and near(row["gain"], ga + gb, 1e-6), f"{row} vs {ga + gb}")
check("8-2 과세표준·세액", near(row["taxable"], ga + gb - 2_500_000, 1e-6) and near(row["tax"], (ga + gb - 2_500_000) * 0.22, 1e-6))
check("8-3 결제일: 미국 T+1, 한국 T+2(주말 넘김)", engine.journal_settle_date("2026-11-20", "US") == "2026-11-23"
      and engine.journal_settle_date("2026-11-19", "KR") == "2026-11-23")
d9 = fresh()
c = engine.journal_add_trade({**BUY, "buy_date": "2026-11-02"}, base_dir=d9, today="2027-01-10")
engine.journal_add_sell(c, {"sell_date": "2026-12-31", "sell_price": 101.0, "qty": 5, "sell_fx": 1400.0}, base_dir=d9, today="2027-01-10")
data9 = engine.journal_load(d9)
data9["trades"].loc[0, "buy_fx"] = np.nan
engine.journal_save(data9, d9)
engine.journal_update_trade(c, {"buy_fx": 1400.0}, base_dir=d9)
engine.journal_add_sell(c, {"sell_date": "2027-01-05", "sell_price": 99.0, "qty": 5, "sell_fx": 1400.0}, base_dir=d9, today="2027-01-10")
tx9 = engine.journal_tax_overseas(engine.journal_load(d9))
check("8-4 12월 31일 판 것은 결제일 해(다음 해)로", [r["year"] for r in tx9["rows"]] == [2027] and tx9["cross_year"] == 1
      and tx9["rows"][0]["n_sells"] == 2, str(tx9))
check("8-5 손실이면 세금 0", tx9["rows"][0]["tax"] == 0.0 and tx9["rows"][0]["gain"] < 0)

# ---------------------------------------------------------------- 9. 고치기·지우기·저장
print("9. 고치기·지우기·저장 형식")
check("9-1 고칠 수 없는 칸", "고칠 수 없는" in (raises(engine.journal_update_trade, t3, {"qty": 5}, base_dir=d) or ""))
check("9-2 손절 ≥ 산 값으로 고치기 막음", raises(engine.journal_update_trade, t3, {"stop_plan": 120.0}, base_dir=d) is not None)
engine.journal_update_trade(t3, {"style": "가치", "reason": "실적 상향 · 저평가"}, base_dir=d)
check("9-3 고친 값 저장(한글 그대로)", engine.journal_load(d)["trades"].set_index("id").loc[t3, "reason"] == "실적 상향 · 저평가")
n = engine.journal_delete("trade", t1, base_dir=d)
dd = engine.journal_load(d)
check("9-4 산 기록 지우면 판 기록도 함께", n == 3 and t1 not in set(dd["trades"]["id"]) and not (dd["sells"]["trade_id"] == t1).any())
raw = open(os.path.join(d, "trades.csv"), "rb").read()
check("9-5 CSV는 BOM 포함 UTF-8, 열 순서 고정", raw.startswith(b"\xef\xbb\xbf") and
      raw.decode("utf-8-sig").splitlines()[0] == ",".join(engine.JOURNAL_TRADE_COLUMNS))

# ---------------------------------------------------------------- 10. 불러오기
print("10. 불러오기(합치기·바꾸기·검사)")
cur = engine.journal_load(d)
src = engine.journal_load(d8)
bad = src["trades"].copy()
bad.loc[len(bad)] = {**bad.iloc[0].to_dict(), "id": "T20260105-099", "stop_plan": 500.0}
orphan = pd.concat([src["sells"], pd.DataFrame([{**src["sells"].iloc[0].to_dict(), "id": "S-X", "trade_id": "T-없음"}])], ignore_index=True)
rep = engine.journal_import(cur, bad, orphan, mode="merge")
check("10-1 합치기: 새 산 기록 3건 · 판 기록 3건 추가", rep["added_trades"] == 3 and rep["added_sells"] == 3, str(rep["added_trades"]) + " " + str(rep["added_sells"]))
check("10-2 검사에 걸린 줄은 이유와 함께 뺌", any("T20260105-099" in e for e in rep["errors"]) and any("S-X" in e for e in rep["errors"]))
rep2 = engine.journal_import(rep["data"], src["trades"], src["sells"], mode="merge")
check("10-3 같은 ID는 건너뜀", rep2["added_trades"] == 0 and rep2["skipped"] == 6)
rep3 = engine.journal_import(rep["data"], src["trades"], src["sells"], mode="replace")
check("10-4 바꾸기", len(rep3["data"]["trades"]) == 3 and len(rep3["data"]["sells"]) == 3)
check("10-5 열 이름만 있는 빈 CSV도 됨", engine.journal_import(cur, pd.DataFrame(columns=["id"]), None)["added_trades"] == 0)

for p in (d, d8, d9):
    shutil.rmtree(p, ignore_errors=True)
print(f"\n{'통과' if not FAILS else '실패'}: {len(FAILS)}개 실패", FAILS)
sys.exit(1 if FAILS else 0)
