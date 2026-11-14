"""신호 기록 장부(engine/ledger.py) 검사 — 가짜 yfinance, 임시 폴더.

1 하루 기록: 열 순서 고정, 판정 규칙 버전, 스윙 점·원자료 저장, 같은 날 컨센서스 기록의 추정치 건수 사용
2 같은 날 다시 실행해도 중복 없음, 가격을 못 받은 종목만 빠짐, 판정 불가 종목은 이유만 기록
3 채점 손 계산: 전환 가능성 표시별 초과 수익·상승 추세 도달, 진입 의견 거래(하락 쪽 근거가 많아진 다음 날 시가 청산)
4 판정 보류(자료 부족), 판정 규칙 버전별로 따로 채점, 보고서
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # 프로젝트 폴더의 engine 패키지를 불러오기 위해
import _offline  # noqa: F401  외부 네트워크 차단 (tests/_offline.py)
import math
import os
import sys
import tempfile
import types
import numpy as np
import pandas as pd
import engine

FAILS = []


def chk(n, c):
    print(("PASS " if c else "FAIL ") + n, flush=True)
    (None if c else FAILS.append(n))


def zig(points, n=420, seed=0):
    idx = pd.bdate_range("2025-03-03", periods=n)
    pos, val = zip(*points)
    c = np.interp(np.arange(n), pos, val) * np.exp(np.random.default_rng(seed).normal(0, 0.003, n))
    return pd.DataFrame({"Open": c, "High": c * 1.01, "Low": c * 0.99, "Close": c}, index=idx)


PX = {
    "AAA": zig([(0, 100), (60, 130), (90, 112), (150, 150), (180, 128), (240, 175), (270, 150), (330, 200), (360, 176), (419, 215)]),
    "BBB": zig([(0, 200), (60, 160), (90, 185), (150, 140), (180, 165), (240, 120), (270, 140), (330, 100), (360, 118), (419, 92)], seed=1),
    "005930.KS": zig([(0, 100), (419, 140)], seed=2),
    "SHORT": zig([(0, 50), (29, 55)], n=30, seed=3),
    "SPY": zig([(0, 100), (419, 120)], seed=4),
    "^KS11": zig([(0, 100), (419, 110)], seed=5),
}


def fake_download(sym, **kw):
    if sym == "BAD":
        raise ConnectionError("x")
    return PX[sym].copy()


class FakeTicker:
    def __init__(self, sym):
        raise RuntimeError("밴드 자료 없음")   # light_band_z는 이유와 함께 None


fyf = types.SimpleNamespace(download=fake_download, Ticker=FakeTicker)
TODAY = "2026-11-16"

with tempfile.TemporaryDirectory() as td:
    snap_dir, led_dir = os.path.join(td, "snap"), os.path.join(td, "ledger")
    os.makedirs(snap_dir)
    pd.DataFrame([{**{c: None for c in engine.SNAPSHOT_COLUMNS}, "snapshot_date": TODAY, "ticker": "AAA",
                   "eps_fy1_up_30d": 6, "eps_fy1_down_30d": 1}],
                 columns=engine.SNAPSHOT_COLUMNS).to_csv(os.path.join(snap_dir, "consensus_2026.csv"), index=False, encoding="utf-8-sig")
    uni = ["AAA", "BBB", "005930.KS", "SHORT", "BAD"]
    res = engine.run_daily_ledger(uni, base_dir=led_dir, yf=fyf, today=TODAY, snapshot_dir=snap_dir, pause=0)
    path = engine.ledger_path(TODAY, led_dir)
    raw = pd.read_csv(path, encoding="utf-8-sig", dtype=str)
    # -----------------------------------------------------------------------
    # 1 하루 기록
    # -----------------------------------------------------------------------
    chk("1-1 열 순서 고정(LEDGER_COLUMNS)", list(raw.columns) == engine.LEDGER_COLUMNS)
    chk("1-2 저장 4줄(가격 못 받은 BAD만 빠짐)", res["saved"] == 4 and len(raw) == 4 and res["failed"] == ["BAD(ConnectionError)"])
    a = raw[raw["ticker"] == "AAA"].iloc[0]
    chk("1-3 판정 규칙 버전·기록일·시장", a["version"] == engine.PLAYBOOK_VERSION and a["ledger_date"] == TODAY and a["market"] == "US")
    chk("1-4 같은 날 컨센서스 기록의 추정치 건수 → 이익 추정치 +1", a["est_up_30d"] == "6" and a["s_estimates"] == "1")
    ev = engine.frame_a_evaluate(PX["AAA"], PX["SPY"], {"eps_fy1_up_30d": 6, "eps_fy1_down_30d": 1}, None)
    chk("1-5 기록 = 같은 입력의 판정 결과(상태·차이·의견)", a["state"] == ev["state"] and int(a["diff"]) == ev["diff"]
        and a["lean_label"] == ev["lean"]["label"] and (a["entry_opinion"] == "True") == ev["opinion"]["entry"])
    chk("1-6 스윙 점 날짜·가격과 연결선 값 저장", pd.notna(a["sw_l2_date"]) and float(a["sw_l2"]) > 0 and pd.notna(a["line_low"]))
    chk("1-7 컨센서스 기록이 없는 종목은 추정치 빈칸·항목 데이터 없음",
        pd.isna(raw[raw["ticker"] == "BBB"].iloc[0]["est_up_30d"]) and pd.isna(raw[raw["ticker"] == "BBB"].iloc[0]["s_estimates"])
        and res["no_consensus"] == 4)
    chk("1-8 자기 밴드를 못 구하면 이유 저장, 한국은 DART 안내", "실패" in a["band_reason"]
        and "DART" in raw[raw["ticker"] == "005930.KS"].iloc[0]["band_reason"])
    sh = raw[raw["ticker"] == "SHORT"].iloc[0]
    chk("1-9 판정 불가 종목은 이유만 기록", pd.isna(sh["state"]) and "부족" in sh["notes"] and "SHORT" in res["no_judgment"])
    # -----------------------------------------------------------------------
    # 2 다시 실행
    # -----------------------------------------------------------------------
    res2 = engine.run_daily_ledger(uni, base_dir=led_dir, yf=fyf, today=TODAY, snapshot_dir=snap_dir, pause=0)
    chk("2-1 같은 날 다시 실행해도 중복 없음", res2["saved"] == 0 and res2["skipped_existing"] == 4
        and len(pd.read_csv(path, encoding="utf-8-sig")) == 4)
    res3 = engine.run_daily_ledger(["AAA"], base_dir=led_dir, yf=fyf, today="2026-11-17", snapshot_dir=snap_dir, pause=0)
    lg = engine.load_ledger(led_dir)
    chk("2-2 다음 날은 이어 붙임, 읽으면 날짜형", res3["saved"] == 1 and len(lg) == 5 and str(lg["as_of"].dtype).startswith("datetime64"))

# ---------------------------------------------------------------------------
# 3 채점 손 계산
# ---------------------------------------------------------------------------
di = pd.bdate_range("2026-01-05", periods=10)
op_a = [10, 10, 11, 12, 13, 14, 15, 16, 17, 18]
pm = {"A": pd.DataFrame({"Open": op_a, "High": op_a, "Low": op_a, "Close": op_a}, index=di, dtype=float),
      "B": pd.DataFrame({"Open": 10.0, "High": 10.0, "Low": 10.0, "Close": 10.0}, index=di),
      "C": pd.DataFrame({"Open": 20.0, "High": 20.0, "Low": 20.0, "Close": 20.0}, index=di)}


def lrow(d, state, lean, entry, sup, ag, ver="V1", t="A"):
    return {"ticker": t, "as_of": di[d], "state": state, "lean_label": lean, "entry_opinion": entry,
            "support": sup, "against": ag, "version": ver}


L = pd.DataFrame([lrow(0, "횡보", "전환 가능성 우세", True, 4, 0), lrow(1, "횡보", "전환 가능성 우세", False, 4, 0),
                  lrow(2, "상승", "상승 추세 유지 우세", False, 4, 1), lrow(3, "상승", "하락 전환 가능성 우세", False, 1, 5),
                  lrow(4, "하락", "실패 가능성 우세", False, 0, 4)])
r = engine.evaluate_frame_a_ledger(L, pm, horizons=[2])
v = r["versions"]["V1"]
b = v["buckets"]["2"]["전환 가능성 우세"]
# 0일 줄: 1일 시가 10 → 3일 시가 12, 1일 줄: 2일 시가 11 → 4일 시가 13. 다른 종목은 수익 0
exp = [math.log(12 / 10), math.log(13 / 11)]
chk("3-1 전환 가능성 우세: 다음 날 시가 → 2거래일 뒤 시가 초과 수익 평균", b["n"] == 2 and abs(b["mean"] - np.mean(exp)) < 1e-12)
chk("3-2 그 기간 안에 상승 추세가 된 비율(장부의 판정으로 셈)", b["became_up_rate"] == 100.0)
f = v["buckets"]["2"]["실패 가능성 우세"]
chk("3-3 실패 가능성 우세: 4일 줄 5일 시가 14 → 7일 시가 16", f["n"] == 1 and abs(f["mean"] - math.log(16 / 14)) < 1e-12
    and f["became_up_rate"] == 0.0)
tr = v["trades"]
# 0일 의견 → 1일 시가 10에 매수. 3일 줄에서 하락 쪽 근거(5) > 상승 쪽(1) → 4일 시가 13에 매도
chk("3-4 진입 의견 거래: 하락 쪽 근거가 많아진 다음 날 시가에 청산", len(tr) == 1 and tr[0]["kind"] == "exit"
    and abs(tr[0]["lret"] - math.log(13 / 10)) < 1e-12 and tr[0]["peer"] == 0.0)
L2 = pd.DataFrame([lrow(6, "횡보", "전환 가능성 우세", True, 4, 0), lrow(7, "횡보", "팽팽", False, 2, 2)])
t2 = engine.evaluate_frame_a_ledger(L2, pm, horizons=[2])["versions"]["V1"]["trades"]
chk("3-5 청산 조건이 없으면 마지막 종가로 평가(미청산)", len(t2) == 1 and t2[0]["kind"] == "open"
    and abs(t2[0]["lret"] - math.log(18 / 16)) < 1e-12)   # 6일 의견 → 7일 시가 16, 마지막 종가 18

# ---------------------------------------------------------------------------
# 4 판정 보류·버전·보고서
# ---------------------------------------------------------------------------
chk("4-1 기록 12개월·의견 30건 전에는 판정 보류", v["verdict"].startswith("판정 보류"))
Lv = pd.concat([L, pd.DataFrame([lrow(0, "횡보", "팽팽", False, 2, 2, ver="V2")])], ignore_index=True)
rv = engine.evaluate_frame_a_ledger(Lv, pm, horizons=[2])
chk("4-2 판정 규칙 버전별로 따로 채점", set(rv["versions"]) == {"V1", "V2"} and rv["versions"]["V2"]["n_rows"] == 1)
rep = engine.render_ledger_report(r, "test")
chk("4-3 보고서에 판정·고정된 채점 방법", "판정 규칙 V1: 판정 보류" in rep and "LEDGER_EVAL" in rep and "전환 가능성 우세" in rep)
chk("4-4 기록이 없으면 안내", "기록 또는 가격 자료 없음" in engine.render_ledger_report(engine.evaluate_frame_a_ledger(pd.DataFrame(), pm)))

print(f"\n{'모두 통과' if not FAILS else f'실패 {len(FAILS)}개: ' + ', '.join(FAILS)}")
sys.exit(1 if FAILS else 0)
