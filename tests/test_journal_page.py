"""매매 기록 화면(pages/3_📒_매매_기록.py)을 AppTest로 실행 — 성과표·세금표·보유 표, 산·판 기록 저장, 빈 기록·인터넷 서버(보기 전용) 안내.
실행: python tests/test_journal_page.py local | empty | view"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _offline  # noqa: F401
import os
import sys
import tempfile
import types

import numpy as np
import pandas as pd

MODE = sys.argv[1] if len(sys.argv) > 1 else "local"
os.environ["JOURNAL_MODE"] = "view" if MODE == "view" else "local"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)


def hist(start=100.0, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2026-01-02", "2026-11-19").tz_localize("America/New_York")
    c = start * np.exp(np.cumsum(rng.normal(0.0003, 0.012, len(idx))))
    return pd.DataFrame({"Open": c * 0.998, "High": c * 1.01, "Low": c * 0.99, "Close": c, "Volume": 1e6}, index=idx)


class FakeTicker:
    def __init__(self, sym):
        self.sym = sym

    def history(self, *a, **k):
        base = {"SPY": 600.0, "^KS11": 2600.0, "KRW=X": 1400.0}.get(self.sym, 70000.0 if self.sym.endswith(".KS") else 150.0)
        return hist(base, seed=abs(hash(self.sym)) % 1000)


fake_yf = types.ModuleType("yfinance")
fake_yf.Ticker = lambda s: FakeTicker(s)
fake_yf.download = lambda *a, **k: hist()
sys.modules["yfinance"] = fake_yf

import engine

tmp = tempfile.mkdtemp(prefix="journal_page_")
engine.JOURNAL_CONFIG["DIR"] = tmp
T = "2026-11-20"
if MODE in ("local", "view"):
    os.environ["JOURNAL_MODE"] = "local"
    a = engine.journal_add_trade({"ticker": "NVDA", "style": "추세", "buy_date": "2026-03-02", "buy_price": 150, "qty": 10, "buy_fee": 1,
                                  "buy_fx": 1400, "stop_plan": 140, "reason": "돌파 확정"}, today=T)
    engine.journal_add_sell(a, {"sell_date": "2026-04-01", "sell_price": 165, "qty": 10, "sell_fee": 1, "sell_fx": 1420}, today=T)
    b = engine.journal_add_trade({"ticker": "005930.KS", "style": "가치", "buy_date": "2026-05-04", "buy_price": 70000, "qty": 5,
                                  "stop_plan": 64000, "reason": "저평가"}, today=T)
    engine.journal_add_sell(b, {"sell_date": "2026-06-01", "sell_price": 62000, "qty": 5}, today=T)
    engine.journal_add_trade({"ticker": "AAPL", "style": "단타", "buy_date": "2026-11-17", "buy_price": 150, "qty": 3, "buy_fx": 1400,
                              "stop_plan": 145, "reason": "변동성 조합"}, today=T)
    os.environ["JOURNAL_MODE"] = "view" if MODE == "view" else "local"
    if MODE == "view":   # 보기 전용: 저장된 파일은 읽지 않는다(불러온 CSV만)
        engine.JOURNAL_CONFIG["DIR"] = tempfile.mkdtemp(prefix="journal_none_")

import streamlit as _st
_st.page_link = lambda *a, **k: None
from streamlit.testing.v1 import AppTest

at = AppTest.from_file(os.path.join(REPO, "pages", "3_📒_매매_기록.py"), default_timeout=120)
at.run()
FAILS = []


def check(name, cond, detail=""):
    print(("  ok  " if cond else "  FAIL") + f"  {name}" + (f"  — {detail}" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


check("예외 없음", not at.exception, str([str(e.value)[:300] for e in at.exception]))
infos = " ".join(i.value for i in at.info)
dfs = [d.value for d in at.dataframe]
if MODE == "local":
    summ = next((d for d in dfs if "판정" in d.columns and "방식" in d.columns), None)
    check("방식별 채점표(가치·추세·단타·전체, 기록 중)", summ is not None and list(summ["방식"]) == ["가치", "추세", "단타", "전체"]
          and summ["판정"].str.startswith("기록 중").all(), str(summ))
    check("끝난 거래 표", any("손절 결과" in d.columns and len(d) == 2 for d in dfs))
    check("해외주식 세금표", any("해(결제일 기준)" in d.columns for d in dfs))
    check("보유 표", any("남은 수량" in d.columns and len(d) == 1 for d in dfs))
    n0 = len(engine.journal_load()["trades"])
    at.text_input(key="jr_buy_tk").input("msft")
    at.number_input(key="jr_buy_px").set_value(400.0)
    at.number_input(key="jr_buy_qty").set_value(2.0)
    at.number_input(key="jr_buy_stop").set_value(380.0)
    at.text_input(key="jr_buy_reason").input("테스트 진입")
    next(b for b in at.button if b.label == "산 기록 저장").click()
    at.run()
    tr = engine.journal_load()["trades"]
    check("산 기록 저장(환율 0 → 결제일 자동)", not at.exception and len(tr) == n0 + 1 and (tr["ticker"] == "MSFT").any()
          and np.isfinite(tr[tr["ticker"] == "MSFT"]["buy_fx"].iloc[0]), str([str(e.value)[:200] for e in at.exception]))
    check("저장 알림", any("산 기록 저장" in s.value for s in at.success))
    at.number_input(key="jr_buy_px").set_value(400.0)
    at.number_input(key="jr_buy_qty").set_value(2.0)
    at.number_input(key="jr_buy_stop").set_value(420.0)
    at.text_input(key="jr_buy_tk").input("msft")
    at.text_input(key="jr_buy_reason").input("손절이 위")
    next(b for b in at.button if b.label == "산 기록 저장").click()
    at.run()
    check("잘못된 입력은 저장 안 하고 이유 표시", len(engine.journal_load()["trades"]) == n0 + 1 and any("낮아야" in e.value for e in at.error))
    ns = len(engine.journal_load()["sells"])
    at.number_input(key="jr_sell_px").set_value(160.0)
    next(b for b in at.button if b.label == "판 기록 저장").click()
    at.run()
    check("판 기록 저장", not at.exception and len(engine.journal_load()["sells"]) == ns + 1)
elif MODE == "empty":
    check("빈 기록 안내", "아직 기록이 없습니다" in infos and "보유 중인 기록이 없습니다" in infos)
    check("저장 위치 안내", any("data/journal/" in c.value for c in at.caption))
else:
    check("보기 전용 안내", "인터넷 서버" in infos)
    check("저장된 파일은 보이지 않음", "아직 기록이 없습니다" in infos)
    try:
        at.text_input(key="jr_buy_tk")
        has_form = True
    except KeyError:
        has_form = False
    check("보기 전용에선 입력 칸 없음", not has_form)

print(f"\n{'통과' if not FAILS else '실패'}: {len(FAILS)}개 실패", FAILS)
sys.exit(1 if FAILS else 0)
