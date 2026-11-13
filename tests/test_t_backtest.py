"""T 트랙 검증·재설계 후보(engine/t_backtest.py) 검사.

1 실제 스캐너 재현: 날짜 t 이후 가격을 바꿔도 t의 판정은 그대로, 한국 종목은 그날 미국 VIX를 쓰지 않음
2 티켓 청산 시뮬레이션: 손절·추격 금지·시간 손절·1차 목표 후 이동평균·최대 보유·데이터 끝 (손 계산)
3 보유 중 새 티켓 무시, 무작위 비교 표본 수
4 재설계 지표(RSI2·55일 최고·20일 최저)와 후보 거래 손 계산, 판정·보고서
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # 프로젝트 폴더의 engine 패키지를 불러오기 위해
import _offline  # noqa: F401  외부 네트워크 차단 (tests/_offline.py)
import sys
import numpy as np
import pandas as pd
import engine

FAILS = []
C = engine.TECH_PARAMS["ROUND_TRIP_COST"]


def chk(n, c):
    print(("PASS " if c else "FAIL ") + n, flush=True)
    (None if c else FAILS.append(n))


def mk(idx, s, d=4e-4, v=0.016, seed=0):
    r = np.random.default_rng(seed)   # 테스트 입력용 (앱 코드 아님)
    c = s * np.exp(np.cumsum(r.normal(d, v, len(idx))))
    o = c * (1 + r.normal(0, .004, len(idx)))
    return pd.DataFrame({"Open": o, "High": np.maximum(o, c) * (1 + abs(r.normal(0, .006, len(idx)))),
                         "Low": np.minimum(o, c) * (1 - abs(r.normal(0, .006, len(idx)))), "Close": c,
                         "Volume": r.lognormal(16, .6, len(idx))}, index=idx)


def main():
    idx = pd.bdate_range("2020-01-02", "2023-06-30")
    df, bench, vixdf = mk(idx, 80, 6e-4, 0.022, 2), mk(idx, 300, 3e-4, 0.011, 3), mk(idx, 18, 0, 0.05, 4)
    vix = vixdf["Close"]
    t = idx[-40]

    # 1) 미래 데이터 미사용 · 한국 VIX 전날까지
    a = engine.scanner_ticket_on(df, bench, vix, t, "US")
    df2 = df.copy()
    df2.loc[df2.index > t, ["Open", "High", "Low", "Close"]] *= 0.5
    b = engine.scanner_ticket_on(df2, bench, vix, t, "US")
    chk("1 날짜 t 이후 가격을 바꿔도 t의 판정 동일", a == b)
    vx2 = vix.copy()
    vx2.loc[t] = 90.0
    k1 = engine.scanner_ticket_on(df, bench, vix, t, "KR")
    k2 = engine.scanner_ticket_on(df, bench, vx2, t, "KR")
    chk("1 한국 종목: 그날 미국 VIX를 바꿔도 판정 동일", k1 == k2)
    short = engine.scanner_ticket_on(df.iloc[:200], bench, vix, df.index[199], "US")
    chk("1 데이터 260봉 미만이면 '데이터 부족'", short["verdict"] == "데이터 부족" and short["executable"] is False)

    # 2) 티켓 청산 시뮬레이션 (i=0 신호, 1일째 시가 100 진입)
    tk = {"stop_price": 95.0, "target_1": 110.0, "time_stop_days": 5, "time_stop_min_price": 101.0,
          "max_hold_days": 30, "chase_ban_price": 103.0}
    unit = (100 - 95) + 100 * C

    def run(o, h, c, ma=None, tkt=None):
        o, h, c = (np.asarray(x, dtype=float) for x in (o, h, c))
        ma = np.full(len(c), np.nan) if ma is None else np.asarray(ma, dtype=float)
        return engine.simulate_ticket_trade(o, h, c, ma, 0, tkt or tk)

    r = run([99, 100, 99], [100, 101, 100], [99, 94, 99])
    chk("2 손절: 1일째 종가 94 < 95 → 전량", r["exit_reason"] == "손절" and abs(r["r"] - (94 - 100 - 100 * C) / unit) < 1e-12)
    chk("2 추격 금지: 시가 104 > 103 → 진입 안 함", run([99, 104, 104], [100, 105, 105], [99, 104, 104])["skipped"].startswith("추격 금지"))
    flat = [100.0] * 8
    r = run(flat, [100.5] * 8, flat)
    chk("2 시간 손절: 5일째 종가 100 < 101 → 전량", r["exit_reason"] == "시간 손절" and r["hold_days"] == 5)
    o = [99, 100, 105, 108, 106, 104]
    h = [100, 101, 111, 109, 107, 105]
    c = [99, 104, 108, 107, 103, 104]
    ma = [np.nan, 100, 102, 104, 105, 105]
    r = run(o, h, c, ma)
    exp = (0.5 * (110 - 100 - 100 * C) + 0.5 * (103 - 100 - 100 * C)) / unit
    chk("2 1차 목표 110에서 50% → 나머지 4일째 종가 103 < 이동평균 105에서 청산",
        r["exit_reason"] == "1차 목표 후 이동평균 이탈" and abs(r["r"] - exp) < 1e-12 and r["hold_days"] == 4)
    tk3 = {**tk, "max_hold_days": 3, "time_stop_days": 0}
    r = run([99, 100, 100, 100, 100], [100] * 5, [99, 100, 100.5, 102, 99], tkt=tk3)
    chk("2 최대 보유 3일 → 3일째 종가 102", r["exit_reason"] == "최대 보유" and abs(r["r"] - (102 - 100 - 100 * C) / unit) < 1e-12)
    chk("2 데이터가 끝날 때까지 청산 안 되면 제외(None)", run([99, 100, 100], [100, 100.5, 100.5], [99, 100, 100], tkt={**tk, "time_stop_days": 0}) is None)

    # 3) 보유 중 새 티켓 무시 · 무작위 비교 표본 수
    d3 = pd.bdate_range("2021-01-04", periods=40)
    px = pd.DataFrame({"Open": 100.0, "High": 100.5, "Low": 99.5, "Close": 100.0}, index=d3)
    px.iloc[6, px.columns.get_loc("Close")] = 94.0          # 6일째 손절
    tkt_row = {**tk, "executable": True, "setup_id": "ST-3", "grade": "A", "entry_price": 100.0, "time_stop_days": 0}
    sig = pd.DataFrame([{**tkt_row, "date": d3[0]}, {**tkt_row, "date": d3[3]}, {**tkt_row, "date": d3[8]}])
    trs = engine.simulate_scanner_trades(px, sig)
    chk("3 보유 중(0~6일) 티켓은 무시, 손절 뒤 티켓은 진입 → 2건", len(trs) == 2 and list(trs["date"]) == [d3[0], d3[8]])
    big = mk(pd.bdate_range("2014-01-02", periods=900), 100, seed=9)
    sig_b = pd.DataFrame([{**tkt_row, "date": big.index[300], "entry_price": float(big["Close"].iloc[300]),
                           "stop_price": float(big["Close"].iloc[300]) * 0.95, "target_1": float(big["Close"].iloc[300]) * 1.1,
                           "chase_ban_price": float(big["Close"].iloc[300]) * 1.03, "max_hold_days": 20}])
    tr_b = engine.simulate_scanner_trades(big, sig_b)
    bs = engine.scanner_random_baseline(big, tr_b, sig_b, start="2014-01-02", samples=20, seed=1)
    chk("3 무작위 비교: 거래 1건 × 20곳(추격 금지 갭 제외) · 셋업 이름 유지", 10 <= len(bs) <= 20 and set(bs["setup_id"]) == {"ST-3"})
    bs2 = engine.scanner_random_baseline(big, tr_b, sig_b, start="2014-01-02", samples=20, seed=1)
    chk("3 같은 시드면 같은 결과", bs["r"].round(10).tolist() == bs2["r"].round(10).tolist())

    # 4) 재설계 지표·후보 거래
    s = pd.Series([10, 11, 10, 12, 13, 12], dtype=float)
    dd = pd.bdate_range("2022-01-03", periods=6)
    ind = engine.redesign_indicators(pd.DataFrame({"Open": s.values, "High": s.values + .5, "Low": s.values - .5, "Close": s.values}, index=dd))
    # 와일더 RSI(2): 변화 +1,−1,+2,+1,−1 → 평균상승/하락을 α=1/2로 갱신 (첫 값은 1·2번째 변화로 시작)
    au = ad = None
    ups, dns = [0, 1, 0, 2, 1, 0], [0, 0, 1, 0, 0, 1]
    vals = []
    for k in range(1, 6):
        if au is None:
            au, ad = ups[k], dns[k]
        else:
            au, ad = au + 0.5 * (ups[k] - au), ad + 0.5 * (dns[k] - ad)
        vals.append(100 - 100 / (1 + au / ad) if ad > 0 else 100.0)
    chk("4 RSI(2) 와일더 손 계산 (마지막 값)", abs(ind["RSI2"].iloc[-1] - vals[-1]) < 1e-9)
    up = pd.DataFrame({"Open": np.arange(1, 81, dtype=float), "High": np.arange(1, 81) + .5, "Low": np.arange(1, 81) - .5,
                       "Close": np.arange(1, 81, dtype=float)}, index=pd.bdate_range("2022-01-03", periods=80))
    iu = engine.redesign_indicators(up)
    chk("4 직전 55일 최고·20일 최저는 당일 제외", iu["HH55"].iloc[-1] == 79.0 and iu["LL20"].iloc[-1] == 60.0)

    n = 260
    dn = pd.bdate_range("2019-01-02", periods=n)
    cl = np.linspace(50, 100, n)
    cl[-6:] = [100, 97, 94, 96, 99, 101]          # 상승 추세 속 3일 급락 → 반등
    dfb = pd.DataFrame({"Open": cl, "High": cl + .5, "Low": cl - .5, "Close": cl}, index=dn)
    ib = engine.redesign_indicators(dfb)
    i_sig = int(np.flatnonzero(engine.t_backtest._candidate_signal(ib, "T-B 눌림목 반등"))[-1])
    rb = engine.simulate_candidate_trade(ib, "T-B 눌림목 반등", i_sig)
    chk("4 T-B: 200일선 위 RSI(2)<10 신호 → 다음 날 시가 진입, 종가 > 5일선에서 청산",
        dn[i_sig] in (dn[-4], dn[-3]) and rb is not None and rb["exit_reason"] == "5일선 위 마감"
        and abs(rb["ret"] - (cl[rb["exit_idx"]] / cl[i_sig + 1] - 1 - C)) < 1e-12)
    cl2 = np.r_[np.linspace(50, 80, 200), np.full(55, 80.0), [85.0, 86.0, 83.0, 79.0, 78.0]]
    dfa = pd.DataFrame({"Open": cl2, "High": cl2 + .5, "Low": cl2 - .5, "Close": cl2}, index=pd.bdate_range("2019-01-02", periods=len(cl2)))
    ia = engine.redesign_indicators(dfa)
    sa = engine.t_backtest._candidate_signal(ia, "T-A 추세 추종")
    ra = engine.simulate_candidate_trade(ia, "T-A 추세 추종", 255)
    atr = float(ia["ATR20"].iloc[255])
    chk("4 T-A: 55일 최고 돌파(255번째) 신호", bool(sa[255]) and not bool(sa[254]))
    chk("4 T-A: 손절 진입가 − 2×ATR 또는 20일 최저 이탈에서 청산, R 단위 2×ATR",
        ra is not None and ra["exit_reason"] in ("손절", "20일 최저 이탈")
        and abs(ra["r"] - (cl2[ra["exit_idx"]] - 86.0 - 86.0 * C) / (2 * atr + 86.0 * C)) < 1e-9)

    # 5) 판정·보고서
    tr = pd.DataFrame({"r": [1.0, 0.8, 1.2, 0.9, 1.1], "ret": .02, "setup_id": "ST-3", "exit_reason": "최대 보유"})
    bsl = pd.DataFrame({"r": [0.0, 0.1, -0.1, 0.05, -0.05] * 4, "setup_id": "ST-3"})
    sm = engine.summarize_scanner(tr, bsl)
    chk("5 스캐너 판정: 평균 R > 0, z ≥ 2 → 통과", sm["verdict"]["passed"] and abs(sm["overall"]["diff"] - 1.0) < 1e-12)
    sm2 = engine.summarize_scanner(tr.assign(r=-tr["r"]), bsl)
    chk("5 평균 R < 0 → 미통과", sm2["verdict"]["passed"] is False)
    rep = engine.render_scanner_report(sm, generated_at="테스트", meta={"n_tickers": 1, "end": "2023-06-30", "n_days": 100})
    chk("5 스캐너 보고서", "## 판정: 통과" in rep and "| ST-3 |" in rep and "### 청산 이유" in rep)
    long_i = pd.bdate_range("1999-01-04", "2014-12-31")
    rr = engine.evaluate_redesign({"XLK": mk(long_i, 50, seed=11)}, {"EWJ": mk(long_i, 30, seed=12), "EWG": mk(long_i, 20, seed=13)})
    chk("5 재설계: 후보 2개 × 기존(앞·뒤·전체)·HOLDOUT(전체)",
        set(rr["results"]) == set(engine.REDESIGN_CANDIDATES)
        and set(rr["results"]["T-A 추세 추종"]["기존 ETF"]) == {"전체 기간", "앞 기간", "뒤 기간"}
        and rr["results"]["T-B 눌림목 반등"]["HOLDOUT"]["전체 기간"]["actual"].get("n", 0) > 0)
    rr_rep = engine.render_redesign_report(rr, generated_at="테스트")
    chk("5 재설계 보고서", "## T-A 추세 추종:" in rr_rep and "## T-B 눌림목 반등:" in rr_rep and "HOLDOUT" in rr_rep)

    # 6) 캐시 열 고정: 티켓 없는 행·있는 행을 나눠 붙여도 열이 어긋나지 않음
    import tempfile
    pth = _os.path.join(tempfile.gettempdir(), "scanner_cache_test.csv")
    if _os.path.exists(pth):
        _os.remove(pth)
    rows1 = pd.DataFrame([{"ticker": "XLK", "date": d3[0], "verdict": "관찰", "setup_id": None, "executable": False}])
    rows2 = pd.DataFrame([{**tkt_row, "ticker": "XLK", "date": d3[1], "verdict": "진입"}])
    for g in (rows1, rows2):
        g.reindex(columns=engine.SCANNER_CACHE_COLUMNS).to_csv(pth, mode="a", header=not _os.path.exists(pth), index=False)
    back = pd.read_csv(pth)
    chk("6 캐시 열 고정: 나중에 붙은 티켓의 손절가가 제 열에 들어감", list(back.columns) == engine.SCANNER_CACHE_COLUMNS and back["stop_price"].iloc[1] == 95.0)

    print("FAILS:", FAILS)
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
