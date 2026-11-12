"""판정 규칙 백테스트(engine/rule_backtest.py) 검사.

1 미래 데이터 미사용: 날짜 k 이후 가격을 바꿔도 k까지의 게이트는 그대로
2 앱과 같은 판정: 홈 화면 히스테리시스 함수(build_gate_history_state)와 최근 10일 결과가 같음
3 한국 게이트는 미국 당일 데이터(QQQ·VIX)를 쓰지 않음
4 병렬 계산 = 순차 계산
5~8 선행 성과·노출 전략·셋업×게이트 집계 계산을 손으로 만든 예제로 확인
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # 프로젝트 폴더의 engine 패키지를 불러오기 위해
import _offline  # noqa: F401  외부 네트워크 차단 (tests/_offline.py)
import sys
import numpy as np
import pandas as pd
import engine

FAILS = []


def chk(n, c):
    print(("PASS " if c else "FAIL ") + n, flush=True)
    (None if c else FAILS.append(n))


def make_ohlc(idx, drift, vol, start, seed):
    rng = np.random.default_rng(seed)   # 테스트 입력용 (앱 코드 아님)
    c = start * np.exp(np.cumsum(rng.normal(drift, vol, len(idx))))
    o = c * (1 + rng.normal(0, 0.003, len(idx)))
    h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.006, len(idx))))
    lo = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.006, len(idx))))
    v = rng.integers(1_000_000, 3_000_000, len(idx)).astype(float)
    return pd.DataFrame({"Open": o, "High": h, "Low": lo, "Close": c, "Adj Close": c, "Volume": v}, index=idx)


def make_map(idx, seed=0):
    # 지수: 상승 추세 → 급락(−20%) → 회복, VIX: 평균 회귀 + 급락 때 급등. 게이트가 G0~G3를 오가게 만든 합성 데이터
    n = len(idx)
    f = np.ones(n)
    a, b = int(n * 0.80), int(n * 0.86)
    f[a:b] = np.linspace(1.0, 0.80, b - a)
    f[b:] = 0.80 * np.exp(np.r_[0.0, np.cumsum(np.full(n - b - 1, 1.5e-3))])
    pm = {}
    for k, st in (("SPY", 300), ("QQQ", 250), ("^KS11", 2500)):
        pm[k] = make_ohlc(idx, 8e-4, 0.011, st, seed + 30 + len(k))
        for col in ("Open", "High", "Low", "Close", "Adj Close"):
            pm[k][col] = pm[k][col] * f
    rng = np.random.default_rng(seed + 9)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = 0.95 * x[i - 1] + rng.normal(0, 0.08)
    spike = np.ones(n)
    spike[int(n * 0.80):int(n * 0.90)] = 2.0
    vix_c = 17.0 * np.exp(x) * spike
    pm["^VIX"] = pd.DataFrame({"Open": vix_c, "High": vix_c * 1.03, "Low": vix_c * 0.97, "Close": vix_c,
                               "Adj Close": vix_c, "Volume": 0.0}, index=idx)
    return pm


def eq_frames(a, b, cols=("raw_level", "level", "cooling_counter", "regime", "macro_phase")):
    if len(a) != len(b) or not a.index.equals(b.index):
        return False
    return all(a[c].astype(str).tolist() == b[c].astype(str).tolist() for c in cols)


def main():
    idx = pd.bdate_range("2023-01-02", periods=560)
    pm = make_map(idx)
    eval_start = str(idx[-45].date())   # 이 구간에 G0~G3 전환이 모두 있음

    # 1) 미래 데이터 미사용 (미국·한국)
    for m in ("US", "KR"):
        base = engine.compute_gate_series(pm, m, start=eval_start)
        k = base.index[20]
        pm2 = {s: df.copy() for s, df in pm.items()}
        for s in pm2:
            after = pm2[s].index > k
            for col in ("Open", "High", "Low", "Close", "Adj Close"):
                pm2[s].loc[after, col] = pm2[s].loc[after, col] * (0.6 if s != "^VIX" else 2.5)
        alt = engine.compute_gate_series(pm2, m, start=eval_start)
        chk(f"1 {m} 날짜 k 이후 가격을 바꿔도 k까지 게이트 동일", eq_frames(base.loc[:k], alt.loc[:k]))
        chk(f"1 {m} 바꾼 뒤에는 실제로 달라짐(검사가 의미 있음)", not eq_frames(base.loc[k:].iloc[1:], alt.loc[k:].iloc[1:]))
        chk(f"1 {m} 검사 구간에 단계 전환이 있음", base["level"].nunique() >= 3)
        print(f"   {m} 게이트:", base["level"].tolist(), flush=True)

    # 2) 앱 히스테리시스 함수와 같은 결과 (데이터가 3년보다 짧아 앱과 같은 창)
    hist = engine.build_gate_history_state(pm, "US")["history"]
    first = pd.Timestamp(hist[0]["date"])
    ser = engine.compute_gate_series(pm, "US", start=str(first.date()))
    app = pd.DataFrame(hist).set_index(pd.to_datetime([h["date"] for h in hist]))
    same = (ser["raw_level"].tolist() == app["raw_level"].tolist() and ser["level"].tolist() == app["level"].tolist()
            and ser["cooling_counter"].tolist() == app["cooling_counter"].tolist())
    chk("2 최근 10일 게이트(지표·최종·완화 일수)가 홈 화면 함수와 같음", same and len(ser) == len(app) == 10)

    # 3) 한국 게이트: 그날 미국 데이터(QQQ·VIX)를 바꿔도 그날 한국 게이트는 그대로
    t = idx[-5]
    pm3 = {s: df.copy() for s, df in pm.items()}
    pm3["^VIX"].loc[t, ["Open", "High", "Low", "Close", "Adj Close"]] = 80.0
    pm3["QQQ"].loc[t, ["Open", "High", "Low", "Close", "Adj Close"]] *= 0.5
    a = engine.gate_inputs_on(pm, "KR", t)["inputs"]
    b = engine.gate_inputs_on(pm3, "KR", t)["inputs"]
    us_a = engine.gate_inputs_on(pm, "US", t)["inputs"]
    us_b = engine.gate_inputs_on(pm3, "US", t)["inputs"]
    chk("3 한국 게이트는 미국 당일 VIX·QQQ 변화에 영향 없음", a == b)
    chk("3 미국 게이트는 같은 날 변화를 반영(검사가 의미 있음)", us_a["vix_value"] != us_b["vix_value"])

    # 4) 병렬 = 순차
    seq = engine.compute_gate_series(pm, "US", start=str(idx[-90].date()), workers=1)
    par = engine.compute_gate_series(pm, "US", start=str(idx[-90].date()), workers=2)
    chk("4 병렬(2개 프로세스) 결과 = 순차 결과", eq_frames(seq, par) and len(seq) == 90)

    # 5) 선행 성과: 손 계산
    d = pd.bdate_range("2024-01-01", periods=6)
    px = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0, 105.0], index=d)
    g = pd.DataFrame({"level": [0, 0, 0, 0, 3, 3]}, index=d)
    fs = engine.gate_forward_stats(g, px, horizons=[2])
    exp = np.mean([102 / 100 - 1, 103 / 101 - 1, 104 / 102 - 1, 105 / 103 - 1])
    chk("5 G0 2일 뒤 평균 수익률 = 손 계산", abs(fs["2"]["G0"]["mean_ret"] - exp) < 1e-12 and fs["2"]["G0"]["n_days"] == 4)
    chk("5 이후 2일이 없는 마지막 날은 제외 (G3 표본 0)", fs["2"]["G3"]["n_days"] == 0)
    px2 = pd.Series([100.0, 90.0, 95.0], index=d[:3])
    fs2 = engine.gate_forward_stats(pd.DataFrame({"level": [3, 3, 3]}, index=d[:3]), px2, horizons=[2])
    chk("5 최대 낙폭 = −10%, 수익률 = −5%", abs(fs2["2"]["G3"]["mean_mdd"] + 0.10) < 1e-12 and abs(fs2["2"]["G3"]["mean_ret"] + 0.05) < 1e-12)
    chk("5 시간 비중 합 100%", abs(sum(fs["time_share"].values()) - 100.0) < 1e-9)

    # 6) 노출 전략
    d = pd.bdate_range("2024-01-01", periods=40)
    rng = np.random.default_rng(5)
    px = pd.Series(100 * np.cumprod(1 + rng.normal(0.001, 0.01, 40)), index=d)
    ret = px.pct_change()
    all_g0 = pd.DataFrame({"level": 0, "raw_level": 0}, index=d)
    r = engine.gate_exposure_backtest(all_g0, px, cost_one_way=0.0)
    chk("6 항상 G0 = 매수 후 보유", abs(r["strategies"]["게이트(최종)"]["total_return"] - r["strategies"]["매수 후 보유"]["total_return"]) < 1e-12)
    r3 = engine.gate_exposure_backtest(pd.DataFrame({"level": 3, "raw_level": 3}, index=d), px, cost_one_way=0.0)
    chk("6 항상 G3 = 수익 0", abs(r3["strategies"]["게이트(최종)"]["total_return"]) < 1e-12)
    lv = pd.Series([3] * 5 + [0] * 35, index=d)
    sw = engine.gate_exposure_backtest(pd.DataFrame({"level": lv, "raw_level": lv}), px, lag=1, cost_one_way=0.001)
    eqg = sw["equity"]["게이트(최종)"]
    # d[5] 종가에 G0 확인 → d[6] 종가에 매수 → d[7] 수익부터 반영, 그날 비용 0.1% 차감
    chk("6 하루 늦게 진입: d[6] 수익 0", abs(eqg.loc[d[6]] / eqg.loc[d[5]] - 1) < 1e-12)
    chk("6 d[7] = 지수 수익 − 비용 0.1%", abs(eqg.loc[d[7]] / eqg.loc[d[6]] - 1 - (ret.loc[d[7]] - 0.001)) < 1e-12)
    chk("6 고정 비중 비교군의 노출 = 게이트 평균 노출", abs(sw["strategies"]["같은 평균 노출 고정"]["avg_exposure"] - sw["strategies"]["게이트(최종)"]["avg_exposure"]) < 1e-12)

    # 7) 거래에 게이트 붙이기·게이트 적용 비교
    gus = pd.DataFrame({"level": [0, 1, 2]}, index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-05"]))
    gkr = pd.DataFrame({"level": [3]}, index=pd.to_datetime(["2024-01-03"]))
    tr = pd.DataFrame({"ticker": ["AAA", "AAA", "AAA", "005930.KS", "005930.KS"], "group": ["대표 종목"] * 5,
                       "date": ["2024-01-02", "2024-01-04", "2024-01-05", "2024-01-02", "2024-01-08"],
                       "setup": ["ST-1", "ST-2", "ST-2", "ST-1", "ST-2"], "r_multiple": [1.0, -1.0, 2.0, 5.0, -0.5]})
    at = engine.attach_gate_to_trades(tr, {"US": gus, "KR": gkr})
    chk("7 신호일의 게이트 (게이트 없는 날은 직전 날)", at["gate_level"].tolist()[:3] == [0, 1, 2])
    chk("7 한국 종목은 한국 게이트, 게이트 시작 전 거래는 없음", np.isnan(at["gate_level"].iloc[3]) and at["gate_level"].iloc[4] == 3)
    sm = engine.summarize_setup_trades_by_gate(at)
    ga = sm["by_group"]["전체"]["전체"]["게이트 적용"]
    # 가중치 G0 1 · G1 0.5 · G2 0 · G3 0 → 1×1.0 + 0.5×(−1.0) + 0 + 0 = 0.5
    chk("7 게이트 적용 합계 R = 0.5, 미적용 = 1.5, 시작 전 1건 제외",
        ga["sum_r_gated"] == 0.5 and ga["sum_r_ungated"] == 1.5 and sm["excluded_no_gate"] == 1 and ga["trades_taken"] == 2)
    chk("7 게이트 적용 기대값 = 0.5 ÷ 1.5 = 0.33", ga["expectancy_gated"] == 0.33)

    # 8) 셋업 검증 함수의 all_trades가 기존 결과와 일치
    long_idx = pd.bdate_range("2018-01-01", periods=1500)
    stock = make_ohlc(long_idx, 4e-4, 0.02, 50, 11)
    res = engine.run_setup_backtest_validation(engine.compute_base_primitives(stock))
    trs = engine.setup_trades_for_ticker(stock)
    chk("8 all_trades 개수 = sample_n, R 목록 = r_distribution",
        res["sample_n"] > 0 and len(res["all_trades"]) == res["sample_n"]
        and [x["r_multiple"] for x in res["all_trades"]] == res["r_distribution"] and trs == res["all_trades"])

    # 9) 전체 실행 → 요약·보고서 (짧은 데이터, 한국 지수 없음 → '데이터 없음' 처리)
    short = pd.bdate_range("2024-01-01", periods=300)
    pm9 = make_map(short, seed=3)
    pm9.pop("^KS11")
    res9 = engine.run_rule_backtest(pm9, {"XLK": stock, "005930.KS": make_ohlc(long_idx, 2e-4, 0.02, 60000, 12)},
                                    markets=["US", "KR"], generated_at="테스트")
    s9 = res9["summary"]
    chk("9 미국 완료·한국은 지수가 없어 '데이터 없음'", s9["markets"]["US"]["status"] == "완료" and s9["markets"]["KR"]["status"] == "데이터 없음")
    chk("9 받지 못한 입력에 ^KS11 기록", s9["missing_inputs"] == ["^KS11"])
    rep = engine.render_backtest_report(s9)
    chk("9 보고서에 게이트 표·셋업 표·주의 사항", "### 게이트 단계별 이후 20거래일" in rep and "T 셋업" in rep and "주의할 점" in rep)
    import json
    chk("9 요약은 JSON으로 저장 가능", len(json.dumps(s9, ensure_ascii=False, default=str)) > 100)

    chk("9 보고서에 원인 분해·비용 민감도 표", "### 무엇 때문에 그 단계였나" in rep and "### 비용 민감도" in rep)

    # 10) 원인 분해: 손으로 만든 G3 4일
    #   1일: 레짐만 3 · 2일: 거시만 3(VIX 2) · 3일: VIX·거시 3 · 4일: 지표는 2인데 완화 대기로 G3 유지
    da = pd.bdate_range("2024-01-01", periods=4)
    ga = pd.DataFrame({"level": [3, 3, 3, 3], "raw_level": [3, 3, 3, 2],
                       "comp_regime": [3, 1, 0, 2], "comp_vix": [1, 2, 3, 1], "comp_macro": [1, 3, 3, 1],
                       "reg_qqq": ["⚫", "🟡", "🟢", "🔴"], "reg_spy": ["🟡", "🟡", "🟢", "🔴"]}, index=da)
    at10 = engine.gate_component_attribution(ga)["G3"]
    chk("10 만든 요소 비중: 레짐 25% · VIX 25% · 거시 50%", at10["binding"] == {"regime": 25.0, "vix": 25.0, "macro": 50.0})
    chk("10 하나 때문: 레짐만 25% · VIX만 0% · 거시만 25%", at10["only"] == {"regime": 25.0, "vix": 0.0, "macro": 25.0})
    chk("10 완화 대기 25%", at10["held_by_hysteresis"] == 25.0)
    rc = engine.gate_component_attribution(ga)["regime_crisis"]
    chk("10 레짐 위기 1일 중 QQQ만 위기 100%", rc["n_days"] == 1 and rc["qqq_only_pct"] == 100.0)

    # 11) 비용 인수: 기본값 = 지금과 같음, 비용 0이면 기대값이 커짐
    prim = engine.compute_base_primitives(stock)
    base_c = engine.TECH_PARAMS["ROUND_TRIP_COST"]
    r_def = engine.run_setup_backtest_validation(prim)
    r_same = engine.run_setup_backtest_validation(prim, cost=base_c)
    r_zero = engine.run_setup_backtest_validation(prim, cost=0.0)
    chk("11 비용 인수 생략 = 현재 비용 지정 (기존 결과 불변)", r_def["r_distribution"] == r_same["r_distribution"])
    chk("11 비용 0의 평균 R > 현재 비용의 평균 R", np.mean(r_zero["r_distribution"]) > np.mean(r_def["r_distribution"]))
    bc = engine.setup_trades_by_cost(stock, [0.0, base_c])
    chk("11 비용별 거래 = 함수 직접 호출 결과", bc[base_c] == r_def["all_trades"] and bc[0.0] == r_zero["all_trades"])

    # 12) 기간 분할: 각 기간 노출 전략은 그 기간 안에서만
    ds = pd.bdate_range("2012-01-02", periods=1000)
    rng12 = np.random.default_rng(8)
    lv12 = pd.Series(rng12.integers(0, 4, 1000), index=ds).rolling(5).max().fillna(0).astype(int)
    g12 = pd.DataFrame({"level": lv12, "raw_level": lv12}, index=ds)
    px12 = pd.Series(100 * np.cumprod(1 + rng12.normal(3e-4, 0.01, 1000)), index=ds)
    sp = engine.gate_split_stats(g12, px12, split="2014-01-01")
    front, back = sp["앞 기간"], sp["뒤 기간"]
    chk("12 앞 기간은 기준일 전, 뒤 기간은 기준일부터",
        front["period"]["end"] < "2014-01-01" <= back["period"]["start"] and front["n_days"] + back["n_days"] == 1000)
    whole_bh = engine.gate_exposure_backtest(g12[g12.index < "2014-01-01"], px12.loc[:"2013-12-31"])["strategies"]["매수 후 보유"]["total_return"]
    chk("12 앞 기간 매수 후 보유 = 앞 기간 가격만으로 계산", abs(front["exposure"]["매수 후 보유"]["total_return"] - whole_bh) < 1e-12)

    print("FAILS:", FAILS)
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":   # Windows는 병렬 계산 때 이 파일을 다시 읽으므로 반드시 필요
    main()
