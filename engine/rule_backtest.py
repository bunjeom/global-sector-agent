"""판정 규칙 백테스트 (2026-11-11 추가): 국면 게이트와 T 셋업을 과거 가격으로 검증한다.

- 국면 게이트: 날짜마다 앱과 같은 함수(build_regime_vix_header → build_gate_inputs_from_header →
  evaluate_market_gate)로 그날의 게이트를 다시 계산한다. 그날 이전 데이터만 쓴다(미래 데이터 미사용).
- 검증 1: 게이트 단계별로 이후 지수 수익률·최대 낙폭·변동성이 다른가 (gate_forward_stats)
- 검증 2: 게이트대로 노출을 줄이면 매수 후 보유·같은 평균 노출의 고정 비중보다 나은가 (gate_exposure_backtest)
- 검증 3: T 셋업(ST-1·ST-2) 성과가 신호 날짜의 게이트 단계에 따라 다른가 (summarize_setup_trades_by_gate)

가격 데이터는 받지 않는다. PC에서 run_backtest.py가 yfinance로 받아 넘긴다.
"""
import math
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from .common import TECH_PARAMS
from .technical_signals import compute_base_primitives
from .backtest import _bt_summary, _setup_arrays, _simulate_setup_trade, run_setup_backtest_validation
from .home import NARRATIVE_BASKETS_SPEC, build_regime_vix_header
from .gate import GATE_CONFIG, apply_gate_hysteresis, build_gate_inputs_from_header, evaluate_market_gate, gate_components

BACKTEST_CONFIG: Dict[str, Any] = {
    "START": "2000-01-01",           # 다운로드 시작일 (VIX·SPY·QQQ·KOSPI가 모두 있는 시기)
    # 앱(app.py)은 지수를 period="3y"로 받는다. 날짜 t의 게이트도 t 이전 3년치만으로 계산해 앱과 같게 한다.
    "PRICE_WINDOW_YEARS": 3,
    "MIN_HISTORY_DAYS": 252,         # 기준 지수 데이터가 이보다 짧은 날은 계산하지 않음 (VIX 1년 백분위·EMA200)
    "FWD_HORIZONS": [5, 20, 60],     # 게이트 이후 성과를 보는 기간 (거래일)
    "MAIN_HORIZON": 20,              # 보고서 요약에 쓰는 기간
    # 게이트 단계별 주식 노출 비중 — 모델 가정이다(데이터 아님). GATE_CONFIG 설명 문구에서 옮김:
    # G0 100% 실행 · G1 신규 비중 50% · G2 V 0차 25% 탐색만 · G3 신규 매수 동결.
    # 실제 게이트는 '신규 진입'만 제한하므로, 지수 노출 전략은 게이트의 위험 판단을 단순화한 대리 지표다.
    "EXPOSURE_BY_GATE": {0: 1.0, 1: 0.5, 2: 0.25, 3: 0.0},
    # T 셋업 진입 크기 — GATE_CONFIG: G1 신규 진입 50% · G2 T 신규 진입 보류 · G3 동결
    "T_SIZE_BY_GATE": {0: 1.0, 1: 0.5, 2: 0.0, 3: 0.0},
    "EXEC_LAG_DAYS": 1,              # 종가에 게이트 확인 → 다음 거래일 종가에 비중 조정 (보수적)
    "COST_ONE_WAY": TECH_PARAMS["ROUND_TRIP_COST"] / 2,   # 비중 1.0 변경당 편도 비용
    "BENCH": {"US": "SPY", "KR": "^KS11"},                  # 게이트 성과를 재는 지수 (RADAR_CONFIG BENCH와 같음)
    # T 셋업 검증 종목. ETF는 생존 편향이 작고, 대표 종목은 '지금 살아남은 강자'라 결과가 좋게 나올 수 있다(사후 선택 편향).
    "SETUP_UNIVERSE": {
        "ETF": [b["ticker"] for b in NARRATIVE_BASKETS_SPEC] + ["069500.KS", "091160.KS"],   # + KODEX 200·KODEX 반도체
        "대표 종목": sorted({p for b in NARRATIVE_BASKETS_SPEC for p in b["proxies"] if p != b["ticker"]})
                    + ["005930.KS", "000660.KS", "005380.KS", "035420.KS", "051910.KS"],
    },
    "SETUP_MIN_BARS": 150,           # run_setup_backtest_validation 최소 길이
    # 기간 분할: 앞 기간과 뒤 기간 결과를 따로 본다(규칙을 바꿀 때 앞으로 정하고 뒤로 확인하기 위함). 2001~2026의 대략 절반
    "SPLIT_DATE": "2014-01-01",
    # 비용 민감도: T 셋업 왕복 비용 (0.30% = 현재 TP-32). 국내 증권사로 미국 주식을 사면 수수료·환전으로 0.30% 안팎이 될 수 있다
    "SETUP_COST_SCENARIOS": [0.0, 0.001, 0.003],
}


def _bt_naive(df_x: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """시간대 제거·날짜순 정렬·종가 없는 행 제거."""
    if df_x is None or len(df_x) == 0:
        return None
    out = df_x.copy()
    idx = pd.to_datetime(out.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    out.index = idx.normalize()
    out = out[~out.index.duplicated(keep="last")].sort_index()
    if "Close" in out:
        out = out[out["Close"].notna()]
    return out


def _bt_window(df_x: Optional[pd.DataFrame], t: pd.Timestamp, strict: bool = False) -> Optional[pd.DataFrame]:
    """날짜 t 시점에 앱이 갖고 있을 데이터: t 이전 PRICE_WINDOW_YEARS년. strict=True면 t 당일 제외."""
    if df_x is None:
        return None
    lo = t - pd.DateOffset(years=int(BACKTEST_CONFIG["PRICE_WINDOW_YEARS"]))
    hi_mask = (df_x.index < t) if strict else (df_x.index <= t)
    out = df_x.loc[(df_x.index > lo) & hi_mask]
    return out if len(out) else None


def gate_inputs_on(price_map: Dict[str, pd.DataFrame], market: str, t: Any) -> Dict[str, Any]:
    """날짜 t의 게이트 입력(레짐·VIX·거시 국면). 앱과 같은 함수를 그날 이전 데이터로 호출한다.
    한국: 한국 장 마감(15:30) 때 미국 당일 장은 아직 열리지 않았으므로 미국 데이터(QQQ·VIX)는 t 전날까지만 쓴다."""
    m = str(market).upper()
    t = pd.Timestamp(t)
    qqq, spy, ks11, vix = (price_map.get(k) for k in ("QQQ", "SPY", "^KS11", "^VIX"))
    if m == "KR":
        hdr = build_regime_vix_header(_bt_window(qqq, t, strict=True), None,
                                      _bt_window(ks11, t), _bt_window(vix, t, strict=True))
    else:
        hdr = build_regime_vix_header(_bt_window(qqq, t), _bt_window(spy, t), None, _bt_window(vix, t))
    inp = build_gate_inputs_from_header(hdr, m)
    # 지수별 레짐 아이콘 (원인 분해용). 데이터가 없어 기본값으로 채운 레짐은 None
    icons = {k: (r.get("icon") if "기본값" not in str(r.get("label", "")) else None)
             for k, r in (("QQQ", hdr["qqq_regime"]), ("SPY", hdr["spy_regime"]), ("KOSPI", hdr["kospi_regime"]))}
    return {"date": t, "inputs": inp, "icons": icons}


_BT_WORKER: Dict[str, Any] = {}


def _bt_worker_init(price_map: Dict[str, pd.DataFrame], market: str) -> None:
    _BT_WORKER["pm"], _BT_WORKER["m"] = price_map, market


def _bt_worker_run(dates: List[pd.Timestamp]) -> List[Dict[str, Any]]:
    return [gate_inputs_on(_BT_WORKER["pm"], _BT_WORKER["m"], t) for t in dates]


def gate_eval_dates(price_map: Dict[str, pd.DataFrame], market: str,
                    start: Optional[str] = None, end: Optional[str] = None) -> List[pd.Timestamp]:
    """게이트를 계산할 날짜: 기준 지수(미국 SPY, 한국 KOSPI) 거래일 중 이전 데이터가 MIN_HISTORY_DAYS 이상인 날."""
    m = str(market).upper()
    base = price_map.get("^KS11" if m == "KR" else "SPY")
    if base is None or len(base) <= int(BACKTEST_CONFIG["MIN_HISTORY_DAYS"]):
        return []
    idx = base.index[int(BACKTEST_CONFIG["MIN_HISTORY_DAYS"]) - 1:]
    if start:
        idx = idx[idx >= pd.Timestamp(start)]
    if end:
        idx = idx[idx <= pd.Timestamp(end)]
    return list(idx)


def compute_gate_series(price_map: Dict[str, pd.DataFrame], market: str, start: Optional[str] = None,
                        end: Optional[str] = None, workers: int = 1,
                        progress: Optional[Callable[[int, int], None]] = None) -> pd.DataFrame:
    """날짜별 게이트 시계열. 지표 계산(날짜끼리 독립)은 workers개 프로세스로 나눠 돌리고,
    히스테리시스(강화 즉시·완화 2일 확인)는 evaluate_market_gate로 날짜 순서대로 적용한다
    (build_gate_history_state와 같은 방식).
    price_map 키: "QQQ", "SPY", "^KS11", "^VIX" (OHLCV, 시간대 없는 날짜 인덱스 권장)."""
    m = str(market).upper()
    pm = {k: _bt_naive(v) for k, v in (price_map or {}).items() if k in ("QQQ", "SPY", "^KS11", "^VIX")}
    dates = gate_eval_dates(pm, m, start, end)
    if not dates:
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    chunk = 40
    chunks = [dates[i:i + chunk] for i in range(0, len(dates), chunk)]
    if workers and workers > 1 and len(chunks) > 1:
        with ProcessPoolExecutor(max_workers=workers, initializer=_bt_worker_init, initargs=(pm, m)) as ex:
            for res in ex.map(_bt_worker_run, chunks):
                rows.extend(res)
                if progress:
                    progress(len(rows), len(dates))
    else:
        _bt_worker_init(pm, m)
        for ch in chunks:
            rows.extend(_bt_worker_run(ch))
            if progress:
                progress(len(rows), len(dates))
    rows.sort(key=lambda r: r["date"])

    out, state = [], None
    for r in rows:
        inp = r["inputs"]
        g = evaluate_market_gate(m, us_data=inp, kr_data=inp, history_state=state)
        comps = gate_components(inp.get("regime"), inp.get("vix_percentile"), inp.get("vix_value"), inp.get("macro_phase"))
        out.append({
            "date": r["date"], "regime": inp.get("regime"), "vix_value": inp.get("vix_value"),
            "vix_percentile": inp.get("vix_percentile"), "macro_phase": inp.get("macro_phase"),
            "comp_regime": comps["regime"], "comp_vix": comps["vix"], "comp_macro": comps["macro"],
            "raw_level": g["raw_level"], "level": g["level"], "cooling_counter": g["cooling_counter"],
            "reg_qqq": r["icons"]["QQQ"], "reg_spy": r["icons"]["SPY"], "reg_kospi": r["icons"]["KOSPI"],
        })
        state = {"prev_level": g["level"], "cooling_counter": g["cooling_counter"]}
    return pd.DataFrame(out).set_index("date")


def _price_col(df_x: pd.DataFrame, total_return: bool = True) -> pd.Series:
    """성과 계산용 가격: 배당 포함(Adj Close)이 있으면 그것, 없으면 종가."""
    if total_return and "Adj Close" in df_x and df_x["Adj Close"].notna().any():
        return df_x["Adj Close"].astype(float)
    return df_x["Close"].astype(float)


def _spells(levels: pd.Series) -> pd.Series:
    """연속 구간 번호 (같은 단계가 이어지는 동안 같은 번호)."""
    return (levels != levels.shift()).cumsum()


def gate_forward_stats(gate_df: pd.DataFrame, price: pd.Series, horizons: Optional[List[int]] = None,
                       level_col: str = "level") -> Dict[str, Any]:
    """게이트 단계별 이후 h거래일 성과. 날짜 t의 게이트를 t 종가에 알고, t 종가 대비 이후를 잰다.
    표준오차는 겹치는 기간을 고려해 독립 표본 수 ≈ 일수 ÷ h 로 계산한다(대략치)."""
    horizons = horizons or list(BACKTEST_CONFIG["FWD_HORIZONS"])
    px = price.dropna().astype(float)
    px.index = pd.to_datetime(px.index)
    lv = gate_df[level_col].dropna().astype(int)
    lv = lv[lv.index.isin(px.index)]
    pos = px.index.get_indexer(lv.index)
    arr = px.values
    daily = np.diff(np.log(arr))
    spell_id = _spells(lv)
    out: Dict[str, Any] = {}
    for h in horizons:
        recs = []
        for d, p, level, sp in zip(lv.index, pos, lv.values, spell_id.values):
            if p + h >= len(arr):
                continue
            path = arr[p + 1:p + h + 1] / arr[p]
            recs.append({"level": int(level), "spell": int(sp), "ret": float(path[-1] - 1.0),
                         "mdd": float(min(path.min() - 1.0, 0.0)),
                         "vol": float(np.std(daily[p:p + h], ddof=1) * math.sqrt(252)) if h >= 2 else None})
        df_r = pd.DataFrame(recs)
        rows = {}
        groups = [("전체", df_r)] + [(f"G{k}", df_r[df_r["level"] == k]) for k in range(4)] if len(df_r) else []
        for name, g in groups:
            n = len(g)
            if n == 0:
                rows[name] = {"n_days": 0}
                continue
            n_eff = max(1.0, n / h)
            sd = float(g["ret"].std(ddof=1)) if n >= 2 else None
            rows[name] = {
                "n_days": n, "n_spells": int(g["spell"].nunique()),
                "mean_ret": float(g["ret"].mean()), "median_ret": float(g["ret"].median()),
                "se_ret": (sd / math.sqrt(n_eff)) if sd is not None and n_eff >= 2 else None,
                "pct_positive": float((g["ret"] > 0).mean() * 100.0),
                "mean_mdd": float(g["mdd"].mean()), "p10_mdd": float(g["mdd"].quantile(0.10)),
                "mean_vol": float(g["vol"].mean()) if h >= 2 else None,
            }
        out[str(h)] = rows
    share = lv.value_counts(normalize=True).reindex(range(4), fill_value=0.0)
    out["time_share"] = {f"G{k}": float(share[k] * 100.0) for k in range(4)}
    out["period"] = {"start": str(lv.index[0].date()) if len(lv) else None,
                     "end": str(lv.index[-1].date()) if len(lv) else None}
    return out


def _perf(rets: pd.Series) -> Dict[str, Any]:
    """일간 수익률 → 연환산 수익률·변동성·샤프(무위험 수익률 차감 안 함)·최대 낙폭."""
    r = rets.dropna().astype(float)
    if len(r) < 2:
        return {"n_days": len(r)}
    eq = (1.0 + r).cumprod()
    years = len(r) / 252.0
    cagr = float(eq.iloc[-1] ** (1.0 / years) - 1.0) if eq.iloc[-1] > 0 else -1.0
    vol = float(r.std(ddof=1) * math.sqrt(252))
    mdd = float((eq / eq.cummax() - 1.0).min())
    return {"n_days": len(r), "years": round(years, 2), "total_return": float(eq.iloc[-1] - 1.0),
            "cagr": cagr, "vol": vol, "sharpe": (float(r.mean() / r.std(ddof=1) * math.sqrt(252)) if r.std(ddof=1) > 0 else None),
            "mdd": mdd, "calmar": (cagr / abs(mdd)) if mdd < 0 else None}


def gate_exposure_backtest(gate_df: pd.DataFrame, price: pd.Series, exposure_map: Optional[Dict[int, float]] = None,
                           lag: Optional[int] = None, cost_one_way: Optional[float] = None) -> Dict[str, Any]:
    """게이트 단계 → 노출 비중 전략과 비교군.
    - 게이트(최종): 히스테리시스 적용 단계, 게이트(즉시): 그날 지표 단계(raw_level, 완화 확인 없음)
    - 매수 후 보유: 항상 100%
    - 같은 평균 노출 고정 비중: 게이트 전략의 평균 노출로 매일 고정 → '타이밍' 효과만 분리
    날짜 t의 게이트 → t+lag 종가에 비중 조정 → 그 다음 날부터 수익률 반영. 비중 변경 1.0당 편도 비용 차감."""
    exposure_map = exposure_map or BACKTEST_CONFIG["EXPOSURE_BY_GATE"]
    lag = int(BACKTEST_CONFIG["EXEC_LAG_DAYS"] if lag is None else lag)
    cost = float(BACKTEST_CONFIG["COST_ONE_WAY"] if cost_one_way is None else cost_one_way)
    px = price.dropna().astype(float)
    px.index = pd.to_datetime(px.index)
    px = px[px.index >= gate_df.index[0]]
    ret = px.pct_change()

    def _run(level: pd.Series) -> Dict[str, Any]:
        sig = level.reindex(px.index).ffill().map(lambda x: exposure_map.get(int(x)) if pd.notna(x) else np.nan)
        hold = sig.shift(lag)                      # t+lag 종가에 맞춘 비중
        turnover = hold.diff().abs().fillna(0.0)
        strat = hold.shift(1) * ret - cost * turnover.shift(1)
        strat = strat.iloc[lag + 1:]
        perf = _perf(strat)
        perf["avg_exposure"] = float(hold.iloc[lag:-1].mean()) if len(hold) > lag + 1 else None
        perf["switches_per_year"] = float((turnover.iloc[lag + 1:] > 0).sum() / max(perf.get("years") or 1, 1e-9))
        return {"perf": perf, "returns": strat}

    final = _run(gate_df["level"])
    raw = _run(gate_df["raw_level"]) if "raw_level" in gate_df else None
    bh_r = ret.iloc[lag + 1:]
    avg_exp = final["perf"]["avg_exposure"] or 0.0
    const_r = avg_exp * bh_r
    curves = pd.DataFrame({"게이트(최종)": final["returns"], "매수 후 보유": bh_r, "같은 평균 노출 고정": const_r})
    if raw is not None:
        curves["게이트(즉시)"] = raw["returns"]
    return {
        "strategies": {
            "게이트(최종)": final["perf"],
            **({"게이트(즉시)": raw["perf"]} if raw is not None else {}),
            "매수 후 보유": _perf(bh_r),
            "같은 평균 노출 고정": {**_perf(const_r), "avg_exposure": avg_exp},
        },
        "equity": (1.0 + curves.fillna(0.0)).cumprod(),
        "assumptions": {"exposure_by_gate": {f"G{k}": v for k, v in exposure_map.items()},
                        "exec_lag_days": lag, "cost_one_way": cost},
    }


GATE_COMPONENTS = ("regime", "vix", "macro")


def gate_component_attribution(gate_df: pd.DataFrame) -> Dict[str, Any]:
    """단계별로 '무엇 때문에 그 단계였나'를 나눈다.
    - binding: 그 단계와 같은 값을 낸 구성 요소 비중(여럿이 동시에 낼 수 있어 합이 100%를 넘을 수 있음)
    - only: 그 구성 요소 하나만 그 단계를 냈던 날 비중
    - held_by_hysteresis: 그날 지표는 이미 낮아졌지만 완화 확인(2일) 때문에 단계가 유지된 날 비중
    미국은 레짐이 위기(⚫)였던 날 중 QQQ만 위기(SPY는 아님)였던 비중도 계산한다."""
    out: Dict[str, Any] = {}
    total = len(gate_df)
    for k in range(4):
        d = gate_df[gate_df["level"] == k]
        n = len(d)
        if n == 0:
            out[f"G{k}"] = {"n_days": 0}
            continue
        held = d["raw_level"] < k
        live = d[~held]
        binding = {c: float((live[f"comp_{c}"] == k).sum() / n * 100.0) for c in GATE_COMPONENTS}
        only = {}
        for c in GATE_COMPONENTS:
            mask = live[f"comp_{c}"] == k
            for o in GATE_COMPONENTS:
                if o != c:
                    mask &= live[f"comp_{o}"].fillna(-1) < k
            only[c] = float(mask.sum() / n * 100.0)
        out[f"G{k}"] = {"n_days": n, "share_of_time": float(n / total * 100.0) if total else 0.0,
                        "binding": binding, "only": only, "held_by_hysteresis": float(held.mean() * 100.0)}
    reg3 = gate_df[gate_df["comp_regime"] == 3]
    if len(reg3) and "reg_spy" in reg3 and reg3["reg_spy"].notna().any():
        out["regime_crisis"] = {"n_days": int(len(reg3)),
                                "qqq_only_pct": float(((reg3["reg_qqq"] == "⚫") & (reg3["reg_spy"] != "⚫")).mean() * 100.0),
                                "spy_crisis_pct": float((reg3["reg_spy"] == "⚫").mean() * 100.0)}
    return out


def gate_split_stats(gate_df: pd.DataFrame, price: pd.Series, split: Optional[str] = None,
                     horizon: Optional[int] = None) -> Dict[str, Any]:
    """앞 기간(SPLIT_DATE 전)과 뒤 기간(이후)을 따로 계산한다. 노출 전략은 각 기간 안에서만 굴린다."""
    split_t = pd.Timestamp(split or BACKTEST_CONFIG["SPLIT_DATE"])
    h = int(horizon or BACKTEST_CONFIG["MAIN_HORIZON"])
    px = price.dropna().astype(float)
    px.index = pd.to_datetime(px.index)
    out: Dict[str, Any] = {"split_date": str(split_t.date()), "horizon": h}
    for name, sub in (("앞 기간", gate_df[gate_df.index < split_t]), ("뒤 기간", gate_df[gate_df.index >= split_t])):
        if len(sub) < 260:
            out[name] = {"n_days": int(len(sub))}
            continue
        fw = gate_forward_stats(sub, px, horizons=[h])
        ex = gate_exposure_backtest(sub, px.loc[:sub.index[-1]])
        out[name] = {"n_days": int(len(sub)), "period": fw["period"], "time_share": fw["time_share"],
                     "forward": fw[str(h)],
                     "exposure": {k: ex["strategies"][k] for k in ("게이트(최종)", "같은 평균 노출 고정", "매수 후 보유")}}
    return out


def setup_trades_by_cost(df: pd.DataFrame, costs: List[float]) -> Dict[float, List[Dict[str, Any]]]:
    """한 종목의 T 셋업 거래를 왕복 비용별로 다시 계산한다(지표는 한 번만 계산). 비용은 목표가·손절 단위에도 들어가 거래 결과 자체가 달라진다."""
    d = _bt_naive(df)
    if d is None or len(d) < int(BACKTEST_CONFIG["SETUP_MIN_BARS"]):
        return {c: [] for c in costs}
    prim = compute_base_primitives(d)
    return {c: list(run_setup_backtest_validation(prim, cost=c).get("all_trades") or []) for c in costs}


def summarize_setup_cost_sensitivity(trades_by_cost: Dict[float, pd.DataFrame]) -> Dict[str, Any]:
    """비용별 × 종목 묶음(ETF·대표 종목) × 시장(미국·한국) 기대값. 게이트와 무관하게 전체 거래로 계산한다."""
    out: Dict[str, Any] = {}
    for c, t in trades_by_cost.items():
        key = f"{c * 100:.2f}%"
        block: Dict[str, Any] = {}
        if t is None or len(t) == 0:
            out[key] = block
            continue
        mk = t["ticker"].map(_ticker_market)
        for grp in ["전체"] + sorted(t["group"].dropna().unique().tolist()):
            g_mask = (t["group"] == grp) if grp != "전체" else pd.Series(True, index=t.index)
            block[grp] = {m: _bt_summary(t.loc[g_mask & ((mk == m) if m != "전체" else True), "r_multiple"].tolist())
                          for m in ("전체", "US", "KR")}
        out[key] = block
    return out


def setup_trades_for_ticker(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """한 종목의 과거 T 셋업 거래 목록(날짜·셋업·R 배수). 레이더 카드의 과거 검증과 같은 함수를 쓴다."""
    d = _bt_naive(df)
    if d is None or len(d) < int(BACKTEST_CONFIG["SETUP_MIN_BARS"]):
        return []
    res = run_setup_backtest_validation(compute_base_primitives(d))
    return list(res.get("all_trades") or [])


def _ticker_market(sym: str) -> str:
    return "KR" if str(sym).upper().endswith((".KS", ".KQ")) else "US"


def attach_gate_to_trades(trades: pd.DataFrame, gate_by_market: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """각 거래에 신호 날짜(그날 종가에서 신호 확인)의 게이트 단계를 붙인다. 신호일이 게이트 거래일이 아니면 직전 게이트."""
    if trades is None or len(trades) == 0:
        return pd.DataFrame(columns=["ticker", "group", "market", "date", "setup", "r_multiple", "gate_level"])
    t = trades.copy()
    t["date"] = pd.to_datetime(t["date"])
    t["market"] = t["ticker"].map(_ticker_market)
    t["gate_level"] = np.nan
    for m, gdf in (gate_by_market or {}).items():
        if gdf is None or len(gdf) == 0:
            continue
        lv = gdf["level"].sort_index()
        mask = t["market"] == m
        pos = lv.index.searchsorted(t.loc[mask, "date"], side="right") - 1
        vals = np.where(pos >= 0, lv.values[np.clip(pos, 0, None)], np.nan)
        t.loc[mask, "gate_level"] = vals
    return t


def summarize_setup_trades_by_gate(trades: pd.DataFrame, size_map: Optional[Dict[int, float]] = None) -> Dict[str, Any]:
    """셋업 × 게이트 단계별 R 배수 요약과, 게이트 규칙대로 크기를 줄였을 때(T_SIZE_BY_GATE)와 안 줄였을 때 비교.
    게이트가 없는 기간(데이터 시작 전)의 거래는 제외한다."""
    size_map = size_map or BACKTEST_CONFIG["T_SIZE_BY_GATE"]
    t = trades[trades["gate_level"].notna()].copy() if len(trades) else trades
    out: Dict[str, Any] = {"n_trades": int(len(t)), "n_tickers": int(t["ticker"].nunique()) if len(t) else 0,
                           "by_group": {}, "excluded_no_gate": int(len(trades) - len(t))}
    if len(t) == 0:
        return out
    t["gate_level"] = t["gate_level"].astype(int)
    for grp in ["전체"] + sorted(t["group"].dropna().unique().tolist()):
        g = t if grp == "전체" else t[t["group"] == grp]
        block: Dict[str, Any] = {}
        for setup in ["전체", "ST-1", "ST-2"]:
            s = g if setup == "전체" else g[g["setup"] == setup]
            row = {"전체": _bt_summary(s["r_multiple"].tolist())}
            for k in range(4):
                row[f"G{k}"] = _bt_summary(s.loc[s["gate_level"] == k, "r_multiple"].tolist())
            w = s["gate_level"].map(size_map).astype(float)
            r = s["r_multiple"].astype(float)
            taken = w > 0
            row["게이트 적용"] = {
                "sum_r_ungated": round(float(r.sum()), 2),
                "sum_r_gated": round(float((w * r).sum()), 2),
                "trades_taken": int(taken.sum()), "trades_all": int(len(s)),
                # 실제로 들어간 거래의 크기 가중 평균 R (크기 1.0 기준)
                "expectancy_gated": round(float((w * r).sum() / w.sum()), 2) if w.sum() > 0 else None,
                "expectancy_ungated": round(float(r.mean()), 2) if len(r) else None,
            }
            block[setup] = row
        out["by_group"][grp] = block
    return out


# ---------------------------------------------------------------------
# 규칙 후보 비교 (2026-11-12 추가). 후보·판정 기준은 결과를 보기 전에 여기서 고정했다.
# 앞 기간(SPLIT_DATE 전)으로만 고르고, 뒤 기간으로 확인한다. 통과하지 못하면 현재 규칙을 유지한다.
# ---------------------------------------------------------------------
GATE_VARIANTS: Dict[str, str] = {
    "현재": "지수 < 200일 지수이동평균이면 즉시 위기(G3). 최종 = 레짐·VIX·거시 국면 중 가장 나쁜 값",
    "A 200일선 이탈 = G2": "레짐 위기(지수 < EMA200)는 G2로 낮춤. G3는 VIX(90%ile·35 이상)나 거시 국면(VIX 80%ile 이상)에서만",
    "B 추세 붕괴 + 공포 동시 = G3": "모든 요소의 단독 상한을 G2로 두고, 레짐 위기와 공포(VIX 75%ile·25 이상 또는 거시 국면 침체)가 같은 날 겹칠 때만 G3",
    # 2026-11-13 추가 (한국만). 한국 게이트가 미국 데이터(VIX·QQQ 레짐)를 대용으로 쓰는 문제를 원칙으로 고친 안. 기준은 위와 같다.
    "C 한국: KOSPI 자체 변동성": "한국 게이트에서 미국 VIX 자리를 KOSPI 실현 변동성(ATR% 1년 백분위 VOLA_PCTL250)으로, 거시 국면 판단의 QQQ 레짐을 KOSPI 레짐으로 바꿈. "
                                 "백분위 기준(90·75·60%ile)은 그대로, VIX 절대값 기준(35·25·20)은 해당 없음. 지수 레짐 요소는 그대로",
}
VARIANT_MARKETS: Dict[str, List[str]] = {"C 한국: KOSPI 자체 변동성": ["KR"]}   # 없으면 모든 시장
VARIANT_CRITERIA: Dict[str, float] = {
    # 1) 위기는 드물어야 한다: 2001~2013의 큰 하락장(2001~03, 2008~09)이 대략 기간의 20% 안팎
    "G3_MAX_SHARE_FRONT": 20.0,
    # 2) G3는 여전히 고위험을 뜻해야 한다: G3 이후 20일 변동성 ≥ G0·G1 이후 변동성 × 1.5
    "MIN_VOL_RATIO_FRONT": 1.5,
    # 뒤 기간 확인: G3 변동성이 G0·G1보다 커야 하고(>1.0), 타이밍 효과가 현재 규칙보다 나빠지면 안 됨
    "MIN_VOL_RATIO_BACK": 1.0,
}
# T 셋업 진입 크기 후보. 'ST-1 BNF 예외'는 과매도 반전(ST-1)을 앱의 BNF 예외처럼 G2·G3에서 20% 허용
T_SIZE_VARIANTS: Dict[str, Dict[str, Dict[int, float]]] = {
    "현재": {"ST-1": {0: 1.0, 1: 0.5, 2: 0.0, 3: 0.0}, "ST-2": {0: 1.0, 1: 0.5, 2: 0.0, 3: 0.0}},
    "ST-1 BNF 예외 20%": {"ST-1": {0: 1.0, 1: 0.5, 2: 0.2, 3: 0.2}, "ST-2": {0: 1.0, 1: 0.5, 2: 0.0, 3: 0.0}},
}


def kospi_vol_components(gate_df: pd.DataFrame, vola_pctl: pd.Series) -> pd.DataFrame:
    """후보 C: KOSPI 실현 변동성 백분위로 VIX 요소와 거시 국면 요소를 다시 만든다.
    - VIX 요소: GATE_CONFIG VIX_THRESHOLDS의 백분위 기준(90·75·60)만 사용
    - 거시 국면: build_regime_vix_header와 같은 문턱(65·80%ile)에 QQQ 레짐 대신 KOSPI 레짐(reg_kospi)
      골디락스(0) = KOSPI 레짐 🟢·🟡 이고 백분위 < 65, 침체(3) = 백분위 ≥ 80, 그 외 금리 민감(1). 백분위가 없으면 1"""
    th = GATE_CONFIG["VIX_THRESHOLDS"]
    mm = GATE_CONFIG["MACRO_MAP"]
    vp = vola_pctl.reindex(gate_df.index, method="ffill").astype(float)
    vix_c = pd.Series(np.select([vp >= th["G3_PERCENTILE"], vp >= th["G2_PERCENTILE"], vp >= th["G1_PERCENTILE"]], [3.0, 2.0, 1.0], 0.0),
                      index=gate_df.index).where(vp.notna())
    reg_ok = gate_df["reg_kospi"].isin(["🟢", "🟡"])
    macro_c = pd.Series(float(mm["SLOWDOWN"]), index=gate_df.index)
    macro_c = macro_c.where(~(reg_ok & (vp < 65)), float(mm["GOLDILOCKS"]))
    macro_c = macro_c.where(~(vp >= 80), float(mm["CREDIT_CRUNCH"]))
    return pd.DataFrame({"comp_vix": vix_c, "comp_macro": macro_c, "vola_pctl": vp})


def gate_variant_raw(gate_df: pd.DataFrame, variant: str, extra: Optional[Dict[str, Any]] = None) -> pd.Series:
    """저장된 구성 요소 단계(comp_regime·comp_vix·comp_macro)로 후보 규칙의 그날 지표 단계를 다시 계산한다.
    구성 요소가 모두 없으면 2 (calculate_single_gate의 방어 폴백과 같음). 후보 C는 extra["vola_pctl"](KOSPI 변동성 백분위)이 필요."""
    r = gate_df["comp_regime"].astype(float)
    v = gate_df["comp_vix"].astype(float)
    mc = gate_df["comp_macro"].astype(float)
    if variant == "현재":
        raw = pd.concat([r, v, mc], axis=1).max(axis=1)
    elif variant.startswith("A"):
        raw = pd.concat([r.clip(upper=2), v, mc], axis=1).max(axis=1)
    elif variant.startswith("B"):
        base = pd.concat([r.clip(upper=2), v.clip(upper=2), mc.clip(upper=2)], axis=1).max(axis=1)
        crisis = (r == 3) & ((v >= 2) | (mc == 3))
        raw = base.where(~crisis, 3.0)
    elif variant.startswith("C"):
        if not extra or extra.get("vola_pctl") is None:
            raise ValueError("후보 C에는 KOSPI 변동성 백분위(extra['vola_pctl'])가 필요합니다")
        kc = kospi_vol_components(gate_df, extra["vola_pctl"])
        raw = pd.concat([r, kc["comp_vix"], kc["comp_macro"]], axis=1).max(axis=1)
    else:
        raise ValueError(f"알 수 없는 후보: {variant}")
    return raw.fillna(2.0).astype(int)


def gate_variant_series(gate_df: pd.DataFrame, variant: str, extra: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """후보 규칙의 날짜별 지표 단계(raw_level)와 최종 단계(level). 히스테리시스는 앱과 같은 apply_gate_hysteresis."""
    raw = gate_variant_raw(gate_df, variant, extra)
    levels, state = [], None
    for x in raw.values:
        lv, cc = apply_gate_hysteresis(int(x), state)
        levels.append(lv)
        state = {"prev_level": lv, "cooling_counter": cc}
    out = gate_df.copy()
    out["raw_level"] = raw.values
    out["level"] = levels
    return out


def _vol_ratio(fw: Dict[str, Any]) -> Optional[float]:
    """G3 이후 변동성 ÷ G0·G1 이후 변동성(일수 가중)."""
    g3 = fw.get("G3", {})
    lo = [fw.get(k, {}) for k in ("G0", "G1")]
    n = sum(x.get("n_days", 0) for x in lo)
    if not g3.get("n_days") or not g3.get("mean_vol") or n == 0:
        return None
    base = sum(x["mean_vol"] * x["n_days"] for x in lo if x.get("n_days")) / n
    return float(g3["mean_vol"] / base) if base else None


def variant_extras(ohlc_by_market: Optional[Dict[str, pd.DataFrame]]) -> Dict[str, Dict[str, Any]]:
    """후보 C용 시장별 추가 입력: 기준 지수 일봉으로 계산한 실현 변동성 백분위(VOLA_PCTL250, 그날 이전 250일 기준)."""
    out: Dict[str, Dict[str, Any]] = {}
    for m, df in (ohlc_by_market or {}).items():
        d = _bt_naive(df)
        if d is None or len(d) < 260 or not {"High", "Low", "Close"} <= set(d.columns):
            continue
        out[m] = {"vola_pctl": compute_base_primitives(d)["VOLA_PCTL250"]}
    return out


def evaluate_gate_variants(gate_by_market: Dict[str, pd.DataFrame], price_by_market: Dict[str, pd.Series],
                           ohlc_by_market: Optional[Dict[str, pd.DataFrame]] = None) -> Dict[str, Any]:
    """후보 × 시장 × 기간별 지표와, 사전에 정한 기준에 따른 판정. 시장이 정해진 후보(VARIANT_MARKETS)는 그 시장만."""
    extras = variant_extras(ohlc_by_market)
    res: Dict[str, Any] = {}
    for name in GATE_VARIANTS:
        res[name] = {}
        for m, g in gate_by_market.items():
            if g is None or len(g) == 0 or m not in price_by_market:
                continue
            if name in VARIANT_MARKETS and m not in VARIANT_MARKETS[name]:
                continue
            if name.startswith("C") and (m not in extras or "reg_kospi" not in g):
                continue
            gv = gate_variant_series(g, name, extras.get(m))
            sp = gate_split_stats(gv, price_by_market[m])
            row = {}
            for per in ("앞 기간", "뒤 기간"):
                p = sp.get(per, {})
                if not p.get("forward"):
                    row[per] = {"n_days": p.get("n_days", 0)}
                    continue
                ex = p["exposure"]
                gs, cs = ex["게이트(최종)"], ex["같은 평균 노출 고정"]
                row[per] = {"n_days": p["n_days"], "period": p["period"], "g3_share": p["time_share"]["G3"],
                            "time_share": p["time_share"], "vol_ratio": _vol_ratio(p["forward"]),
                            "gate_sharpe": gs.get("sharpe"), "const_sharpe": cs.get("sharpe"),
                            "timing": (gs["sharpe"] - cs["sharpe"]) if gs.get("sharpe") is not None and cs.get("sharpe") is not None else None,
                            "gate_mdd": gs.get("mdd"), "const_mdd": cs.get("mdd"), "gate_cagr": gs.get("cagr"),
                            "avg_exposure": gs.get("avg_exposure"), "switches_per_year": gs.get("switches_per_year")}
            res[name][m] = row
    return {"results": res, "verdict": select_gate_variant(res)}


def select_gate_variant(res: Dict[str, Any]) -> Dict[str, Any]:
    """사전에 정한 기준으로 후보를 고른다. 1단계: 앞 기간 기준(모든 시장) 통과 후보 중 앞 기간 타이밍 효과 평균이 가장 큰 것.
    2단계: 뒤 기간에서 모든 시장 G3 변동성 비율 > 1.0이고 타이밍 효과가 현재 규칙 이상이면 채택 후보, 아니면 현재 유지."""
    c = VARIANT_CRITERIA
    checks: Dict[str, Any] = {}
    passed = []
    cur = res.get("현재", {})
    for name, by_m in res.items():
        if name == "현재" or not by_m:
            continue
        reasons, ok = [], True
        for m, row in by_m.items():
            f = row.get("앞 기간", {})
            if f.get("g3_share") is None:
                ok = False
                reasons.append(f"{m} 앞 기간 데이터 없음")
                continue
            if f["g3_share"] > c["G3_MAX_SHARE_FRONT"]:
                ok = False
                reasons.append(f"{m} G3 비중 {f['g3_share']:.0f}% > {c['G3_MAX_SHARE_FRONT']:.0f}%")
            if f.get("vol_ratio") is None or f["vol_ratio"] < c["MIN_VOL_RATIO_FRONT"]:
                ok = False
                vr = "없음" if f.get("vol_ratio") is None else f"{f['vol_ratio']:.2f}배"
                reasons.append(f"{m} G3 변동성 비율 {vr} < {c['MIN_VOL_RATIO_FRONT']:.1f}배")
        tims = [row.get("앞 기간", {}).get("timing") for row in by_m.values()]
        score = float(np.mean(tims)) if tims and all(t is not None for t in tims) else None
        checks[name] = {"front_pass": ok, "front_reasons": reasons, "front_timing_mean": score}
        if ok and score is not None:
            passed.append((score, name))
    if not passed:
        return {"chosen": None, "decision": "현재 규칙 유지", "reason": "앞 기간 기준을 통과한 후보가 없음", "checks": checks}
    _, best = max(passed)
    back_reasons, back_ok = [], True
    for m, row in res[best].items():
        b, bc = row.get("뒤 기간", {}), cur.get(m, {}).get("뒤 기간", {})
        if b.get("vol_ratio") is None or b["vol_ratio"] <= c["MIN_VOL_RATIO_BACK"]:
            back_ok = False
            back_reasons.append(f"{m} 뒤 기간 G3 변동성 비율 {('없음' if b.get('vol_ratio') is None else format(b['vol_ratio'], '.2f') + '배')} ≤ {c['MIN_VOL_RATIO_BACK']:.1f}배")
        if b.get("timing") is None or bc.get("timing") is None or b["timing"] < bc["timing"]:
            back_ok = False
            back_reasons.append(f"{m} 뒤 기간 타이밍 효과 {b.get('timing') if b.get('timing') is None else format(b['timing'], '+.2f')} < 현재 {bc.get('timing') if bc.get('timing') is None else format(bc['timing'], '+.2f')}")
    checks[best]["back_pass"] = back_ok
    checks[best]["back_reasons"] = back_reasons
    if back_ok:
        return {"chosen": best, "decision": f"채택 후보: {best}", "reason": "앞 기간 기준 통과, 뒤 기간 확인 통과", "checks": checks}
    return {"chosen": None, "decision": "현재 규칙 유지", "reason": f"앞 기간에서 '{best}'를 골랐지만 뒤 기간 확인 실패", "checks": checks}


def evaluate_setup_size_variants(trades: pd.DataFrame, gate_by_market: Dict[str, pd.DataFrame],
                                 split: Optional[str] = None) -> Dict[str, Any]:
    """게이트 후보 × T 진입 크기 후보 × 기간 × 종목 묶음별 결과. 거래의 게이트 단계는 후보 게이트로 다시 붙인다.
    판정: 'ST-1 BNF 예외'는 앞 기간과 뒤 기간 모두 ETF 합계 R이 '현재'보다 클 때만 채택 후보."""
    split_t = pd.Timestamp(split or BACKTEST_CONFIG["SPLIT_DATE"])
    out: Dict[str, Any] = {"split_date": str(split_t.date()), "results": {}, "verdict": {}}
    if trades is None or len(trades) == 0:
        return out
    base = trades[["ticker", "group", "date", "setup", "r_multiple"]].copy()
    base["date"] = pd.to_datetime(base["date"])
    for gname in GATE_VARIANTS:
        if gname in VARIANT_MARKETS:   # 한 시장 전용 후보는 T 진입 크기 비교에서 제외 (거래가 두 시장에 걸쳐 있음)
            continue
        gv = {m: gate_variant_series(g, gname) for m, g in gate_by_market.items() if g is not None and len(g)}
        t = attach_gate_to_trades(base, gv)
        t = t[t["gate_level"].notna()].copy()
        t["gate_level"] = t["gate_level"].astype(int)
        res_g: Dict[str, Any] = {}
        for sname, smap in T_SIZE_VARIANTS.items():
            w = pd.Series([smap.get(s, {}).get(int(lv), 0.0) for s, lv in zip(t["setup"], t["gate_level"])], index=t.index)
            res_s: Dict[str, Any] = {}
            for per, pmask in (("앞 기간", t["date"] < split_t), ("뒤 기간", t["date"] >= split_t), ("전체 기간", pd.Series(True, index=t.index))):
                res_p: Dict[str, Any] = {}
                for grp in ("ETF", "전체"):
                    gm = pmask & ((t["group"] == grp) if grp != "전체" else True)
                    r, ww = t.loc[gm, "r_multiple"].astype(float), w[gm]
                    taken = ww > 0
                    rt = r[taken]
                    res_p[grp] = {"sum_r": round(float((ww * r).sum()), 2), "n_taken": int(taken.sum()), "n_all": int(gm.sum()),
                                  "expectancy": round(float((ww * r).sum() / ww.sum()), 3) if ww.sum() > 0 else None,
                                  "se": round(float(rt.std(ddof=1) / math.sqrt(len(rt))), 3) if len(rt) >= 2 else None}
                res_s[per] = res_p
            res_g[sname] = res_s
        out["results"][gname] = res_g
        cur, alt = res_g["현재"], res_g["ST-1 BNF 예외 20%"]
        ok = all(alt[p]["ETF"]["sum_r"] > cur[p]["ETF"]["sum_r"] for p in ("앞 기간", "뒤 기간"))
        out["verdict"][gname] = {"adopt_st1_exception": ok,
                                 "front_delta_r": round(alt["앞 기간"]["ETF"]["sum_r"] - cur["앞 기간"]["ETF"]["sum_r"], 2),
                                 "back_delta_r": round(alt["뒤 기간"]["ETF"]["sum_r"] - cur["뒤 기간"]["ETF"]["sum_r"], 2)}
    return out


def run_variant_analysis(gate_by_market: Dict[str, pd.DataFrame], price_by_market: Dict[str, pd.Series],
                         trades: Optional[pd.DataFrame] = None,
                         ohlc_by_market: Optional[Dict[str, pd.DataFrame]] = None) -> Dict[str, Any]:
    """게이트 후보 비교와 T 진입 크기 후보 비교를 한 번에. 저장된 게이트 시계열(CSV)만 있으면 몇 초 안에 끝난다."""
    gv = evaluate_gate_variants(gate_by_market, price_by_market, ohlc_by_market)
    sv = evaluate_setup_size_variants(trades, gate_by_market) if trades is not None and len(trades) else None
    return {"gate": gv, "setup": sv, "candidates": GATE_VARIANTS, "criteria": VARIANT_CRITERIA,
            "t_size_variants": {k: {s: {f"G{lv}": w for lv, w in m.items()} for s, m in v.items()} for k, v in T_SIZE_VARIANTS.items()}}


def render_variants_report(v: Dict[str, Any], generated_at: Optional[str] = None) -> str:
    """후보 비교 보고서(마크다운)."""
    L = ["# 판정 규칙 후보 비교", "", f"- 생성: {generated_at or '—'} · 기간 분할 기준일 {BACKTEST_CONFIG['SPLIT_DATE']}",
         "- 후보와 판정 기준은 결과를 보기 전에 engine/rule_backtest.py(GATE_VARIANTS·VARIANT_CRITERIA·T_SIZE_VARIANTS)에 고정했습니다. "
         "앞 기간으로 고르고 뒤 기간으로 확인하며, 통과하지 못하면 현재 규칙을 유지합니다.", "",
         "## 국면 게이트 후보", ""]
    for name, desc in v["candidates"].items():
        L.append(f"- **{name}**: {desc}")
    c = v["criteria"]
    L += ["", f"판정 기준: 앞 기간에서 후보가 적용되는 모든 시장(C는 한국만)의 G3 비중 ≤ {c['G3_MAX_SHARE_FRONT']:.0f}%, G3 이후 변동성 ≥ G0·G1의 {c['MIN_VOL_RATIO_FRONT']:.1f}배 → "
          f"통과 후보 중 앞 기간 타이밍 효과(게이트 샤프 − 같은 평균 노출 고정 샤프)가 가장 큰 것 → 뒤 기간에서 G3 변동성 비율 > {c['MIN_VOL_RATIO_BACK']:.1f}배이고 "
          f"타이밍 효과가 현재 이상이면 채택 후보.", ""]
    res = v["gate"]["results"]
    markets = sorted({m for by_m in res.values() for m in by_m})
    for m in markets:
        L += [f"### {MARKET_NAMES.get(m, m)}", "",
              "| 후보 | 기간 | G3 비중 | G3 변동성 ÷ G0·G1 | 게이트 샤프 | 고정 비중 샤프 | 타이밍 효과 | 게이트 최대 낙폭 | 고정 비중 최대 낙폭 | 평균 노출 | 연간 비중 변경 |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for name in res:
            if name in VARIANT_MARKETS and m not in VARIANT_MARKETS[name]:
                continue   # 다른 시장 전용 후보
            for per in ("앞 기간", "뒤 기간"):
                r = res[name].get(m, {}).get(per, {})
                if r.get("g3_share") is None:
                    L.append(f"| {name} | {per} | 데이터 부족 | | | | | | | | |")
                    continue
                pr = r.get("period") or {}
                L.append(f"| {name} | {per} ({str(pr.get('start'))[:4]}~{str(pr.get('end'))[:4]}) | {r['g3_share']:.0f}% | "
                         f"{'—' if r['vol_ratio'] is None else format(r['vol_ratio'], '.2f') + '배'} | {_num_txt(r['gate_sharpe'])} | {_num_txt(r['const_sharpe'])} | "
                         f"{'—' if r['timing'] is None else format(r['timing'], '+.2f')} | {_pct(r['gate_mdd'])} | {_pct(r['const_mdd'])} | "
                         f"{_pct(r['avg_exposure'], 0, sign=False)} | {_num_txt(r['switches_per_year'], 1)} |")
        L.append("")
    vd = v["gate"]["verdict"]
    L += [f"### 판정: {vd['decision']}", "", f"- {vd['reason']}"]
    for name, ck in vd["checks"].items():
        front = "통과" if ck["front_pass"] else "탈락 — " + "; ".join(ck["front_reasons"])
        L.append(f"- {name}: 앞 기간 {front}" + (f" (타이밍 효과 평균 {ck['front_timing_mean']:+.2f})" if ck.get("front_timing_mean") is not None else ""))
        if "back_pass" in ck:
            L.append(f"  - 뒤 기간 확인 {'통과' if ck['back_pass'] else '실패 — ' + '; '.join(ck['back_reasons'])}")
    L.append("")
    sv = v.get("setup")
    if sv and sv.get("results"):
        L += ["## T 셋업 진입 크기 후보", "",
              "- **현재**: G0 100% · G1 50% · G2·G3 0% (ST-1·ST-2 같음)",
              "- **ST-1 BNF 예외 20%**: 과매도 반전(ST-1)만 G2·G3에서 20% 허용 (앱의 BNF 예외와 같은 크기), ST-2는 현재와 같음",
              "- 판정 기준: 앞 기간과 뒤 기간 모두 ETF 합계 R이 '현재'보다 커야 채택 후보 (대표 종목은 사후 선택 편향이 있어 판정에 쓰지 않음)", "",
              "| 게이트 후보 | 진입 크기 | 기간 | ETF 합계 R | ETF 기대값 ±표준오차 (건수) | 전체 합계 R | 전체 기대값 (건수) |",
              "|---|---|---|---|---|---|---|"]
        for gname, res_g in sv["results"].items():
            for sname, res_s in res_g.items():
                for per in ("앞 기간", "뒤 기간"):
                    e, a = res_s[per]["ETF"], res_s[per]["전체"]
                    ex_e = "—" if e["expectancy"] is None else f"{e['expectancy']:+.2f}R" + (f" ±{e['se']:.2f}" if e.get("se") is not None else "")
                    ex_a = "—" if a["expectancy"] is None else f"{a['expectancy']:+.2f}R"
                    L.append(f"| {gname} | {sname} | {per} | {e['sum_r']:+.1f}R | {ex_e} ({e['n_taken']}/{e['n_all']}) | {a['sum_r']:+.1f}R | {ex_a} ({a['n_taken']}/{a['n_all']}) |")
        L.append("")
        for gname, vv in sv["verdict"].items():
            L.append(f"- 게이트 '{gname}' 기준: ST-1 BNF 예외 {'채택 후보' if vv['adopt_st1_exception'] else '채택 안 함'} "
                     f"(ETF 합계 R 변화 앞 기간 {vv['front_delta_r']:+.1f}R · 뒤 기간 {vv['back_delta_r']:+.1f}R)")
        L.append("")
    L += ["## 주의", "- 전체 기간 결과를 이미 본 뒤에 만든 후보라, 뒤 기간 확인도 완전히 독립적인 검증은 아닙니다. 채택 후보가 나와도 앱에 반영한 뒤 앞으로의 실제 결과로 다시 확인해야 합니다.",
          "- T 셋업은 레이더 카드의 단순화 규칙이고, 노출 비중은 모델 가정입니다(REPORT.md의 주의 사항과 같음).", ""]
    return "\n".join(L)


# ---------------------------------------------------------------------
# T 셋업 진단 (2026-11-13): 실제 신호 vs 같은 매매 방식의 무작위 진입. 판정 기준은 결과를 보기 전에 고정.
# ---------------------------------------------------------------------
SETUP_DIAG_CRITERIA: Dict[str, Any] = {
    # ETF(생존 편향이 작음)에서 '신호 평균 R − 무작위 진입 평균 R'이 표준오차의 2배 이상이어야 '통과'.
    # 미통과 셋업은 레이더 카드에 '과거 검증 미통과'로 표시한다(반영은 결과 확인 후).
    "GROUP": "ETF", "MIN_Z": 2.0,
}


def setup_signals_and_baseline(df: pd.DataFrame, cost: Optional[float] = None) -> Dict[str, List[Dict[str, Any]]]:
    """한 종목의 실제 셋업 거래와, 같은 매매 방식으로 매 보유 기간마다 진입한 무작위(비교용) 거래.
    무작위 진입은 신호 조건 없이 i = 120부터 보유 기간(ST-1 5봉, ST-2 15봉) 간격으로 진입해 거래끼리 겹치지 않는다."""
    d = _bt_naive(df)
    if d is None or len(d) < int(BACKTEST_CONFIG["SETUP_MIN_BARS"]):
        return {"signals": [], "baseline": []}
    c = TECH_PARAMS["ROUND_TRIP_COST"] if cost is None else float(cost)
    prim = compute_base_primitives(d)
    signals = list(run_setup_backtest_validation(prim, cost=c).get("all_trades") or [])
    arr = _setup_arrays(prim)
    base = []
    for setup, st1, step in (("ST-1", True, 5), ("ST-2", False, 15)):
        for i in range(120, len(prim) - 15, step):
            sim = _simulate_setup_trade(arr, i, st1, c)
            if np.isfinite(sim["r"]):
                base.append({"date": str(prim.index[i])[:10], "setup": setup, "r_multiple": round(float(sim["r"]), 2), "exit_type": sim["exit_type"]})
    return {"signals": signals, "baseline": base}


def _r_stats(r: pd.Series) -> Dict[str, Any]:
    r = r.astype(float)
    n = len(r)
    if n == 0:
        return {"n": 0}
    return {"n": n, "mean": float(r.mean()), "se": float(r.std(ddof=1) / math.sqrt(n)) if n >= 2 else None,
            "win_rate": float((r > 0).mean() * 100.0)}


def summarize_setup_diag(signals: pd.DataFrame, baseline: pd.DataFrame) -> Dict[str, Any]:
    """묶음(ETF·대표 종목·전체) × 셋업별: 신호와 무작위 진입의 평균 R, 차이와 z, 청산 이유별 비중·평균 R, 사전 기준 판정."""
    crit = SETUP_DIAG_CRITERIA
    out: Dict[str, Any] = {"criteria": crit, "by_group": {}, "verdict": {}}
    if signals is None or len(signals) == 0:
        return out
    for grp in ["ETF", "대표 종목", "전체"]:
        sg = signals if grp == "전체" else signals[signals["group"] == grp]
        bg = baseline if grp == "전체" else baseline[baseline["group"] == grp]
        block = {}
        for setup in ("ST-1", "ST-2"):
            ss, bb = sg[sg["setup"] == setup], bg[bg["setup"] == setup]
            s_st, b_st = _r_stats(ss["r_multiple"]), _r_stats(bb["r_multiple"])
            diff = z = None
            if s_st.get("n", 0) >= 2 and b_st.get("n", 0) >= 2:
                diff = s_st["mean"] - b_st["mean"]
                se_d = math.sqrt(s_st["se"] ** 2 + b_st["se"] ** 2)
                z = diff / se_d if se_d > 0 else None
            exits = {}
            for et in ("목표", "손절", "기간 만료"):
                e_s, e_b = ss[ss["exit_type"] == et], bb[bb["exit_type"] == et]
                exits[et] = {"signal_share": float(len(e_s) / len(ss) * 100.0) if len(ss) else None,
                             "signal_mean": float(e_s["r_multiple"].mean()) if len(e_s) else None,
                             "baseline_share": float(len(e_b) / len(bb) * 100.0) if len(bb) else None,
                             "baseline_mean": float(e_b["r_multiple"].mean()) if len(e_b) else None}
            block[setup] = {"signal": s_st, "baseline": b_st, "diff": diff, "z": z, "exits": exits}
        out["by_group"][grp] = block
    for setup in ("ST-1", "ST-2"):
        b = out["by_group"].get(crit["GROUP"], {}).get(setup, {})
        z = b.get("z")
        out["verdict"][setup] = {"passed": bool(z is not None and z >= crit["MIN_Z"]), "z": z, "diff": b.get("diff")}
    return out


def run_setup_diag(setup_price_map: Dict[str, pd.DataFrame], cost: Optional[float] = None) -> Dict[str, Any]:
    """T 셋업 진단 전체 실행 (게이트 계산 없음)."""
    groups = {s: grp for grp, syms in BACKTEST_CONFIG["SETUP_UNIVERSE"].items() for s in syms}
    sig_rows, base_rows = [], []
    for sym, df in (setup_price_map or {}).items():
        r = setup_signals_and_baseline(df, cost)
        g = groups.get(sym, "추가 종목")
        sig_rows += [{"ticker": sym, "group": g, **x} for x in r["signals"]]
        base_rows += [{"ticker": sym, "group": g, **x} for x in r["baseline"]]
    sig, base = pd.DataFrame(sig_rows), pd.DataFrame(base_rows)
    summ = summarize_setup_diag(sig, base)
    summ["n_tickers"] = int(sig["ticker"].nunique()) if len(sig) else 0
    summ["cost"] = TECH_PARAMS["ROUND_TRIP_COST"] if cost is None else float(cost)
    return {"summary": summ, "signals": sig, "baseline": base}


def render_setup_diag_report(s: Dict[str, Any], generated_at: Optional[str] = None) -> str:
    """T 셋업 진단 보고서(마크다운)."""
    crit = s.get("criteria", SETUP_DIAG_CRITERIA)
    L = ["# T 셋업 진단: 실제 신호 vs 무작위 진입", "",
         f"- 생성: {generated_at or '—'} · 종목 {s.get('n_tickers', 0)}개 · 왕복 비용 {float(s.get('cost') or 0) * 100:.2f}%",
         "- 무작위 진입: 신호 조건 없이 같은 매매 방식(다음 날 시가 진입, 손절은 종가 기준, 목표는 장중 고가, ST-1 최대 5일·ST-2 최대 15일)으로 보유 기간마다 진입. "
         "신호가 무작위 진입보다 낫지 않으면 신호 조건이 더해 주는 것이 없다는 뜻입니다.",
         f"- 판정 기준(결과 보기 전 고정): {crit['GROUP']}에서 (신호 평균 R − 무작위 평균 R) ≥ 표준오차 × {crit['MIN_Z']:.0f} 이면 통과. "
         "미통과 셋업은 레이더 카드에 '과거 검증 미통과'로 표시합니다.", ""]
    for setup in ("ST-1", "ST-2"):
        v = s.get("verdict", {}).get(setup, {})
        if v:
            z = v.get("z")
            L.append(f"**{setup} 판정: {'통과' if v['passed'] else '미통과'}** (ETF 차이 "
                     f"{'—' if v.get('diff') is None else format(v['diff'], '+.2f') + 'R'}, z {'—' if z is None else format(z, '+.1f')})")
    L.append("")
    L += ["| 묶음 | 셋업 | 신호 평균 R ±표준오차 (승률·건수) | 무작위 평균 R ±표준오차 (승률·건수) | 차이 | z |", "|---|---|---|---|---|---|"]

    def cell(x):
        if not x.get("n"):
            return "표본 없음"
        se = f" ±{x['se']:.2f}" if x.get("se") is not None else ""
        return f"{x['mean']:+.2f}R{se} ({x['win_rate']:.0f}% · {x['n']:,}건)"
    for grp, block in s.get("by_group", {}).items():
        for setup, b in block.items():
            L.append(f"| {grp} | {setup} | {cell(b['signal'])} | {cell(b['baseline'])} | "
                     f"{'—' if b['diff'] is None else format(b['diff'], '+.2f') + 'R'} | {'—' if b['z'] is None else format(b['z'], '+.1f')} |")
    L += ["", "### 청산 이유별 (ETF)", "| 셋업 | 청산 | 신호 비중 · 평균 R | 무작위 비중 · 평균 R |", "|---|---|---|---|"]
    for setup, b in s.get("by_group", {}).get("ETF", {}).items():
        for et, e in b["exits"].items():
            sig = "—" if e["signal_share"] is None else f"{e['signal_share']:.0f}% · " + ("—" if e["signal_mean"] is None else f"{e['signal_mean']:+.2f}R")
            bas = "—" if e["baseline_share"] is None else f"{e['baseline_share']:.0f}% · " + ("—" if e["baseline_mean"] is None else f"{e['baseline_mean']:+.2f}R")
            L.append(f"| {setup} | {et} | {sig} | {bas} |")
    L += ["", "## 주의",
          "- 레이더 카드 과거 검증과 같은 단순화 규칙입니다(실제 셋업 스캐너의 모든 조건을 재현하지 않음).",
          "- 같은 날 여러 종목이 함께 움직이므로 거래끼리 완전히 독립이 아니고, 표준오차는 실제보다 작게 나올 수 있습니다.",
          "- '대표 종목'은 사후 선택 편향이 있어 판정에 쓰지 않습니다.", ""]
    return "\n".join(L)


# ---------------------------------------------------------------------
# 전체 실행·요약·보고서 (run_backtest.py와 이후 화면에서 함께 쓴다)
# ---------------------------------------------------------------------
MARKET_NAMES = {"US": "미국", "KR": "한국"}


def _weekly_equity(eq: pd.DataFrame) -> Dict[str, Any]:
    """누적 수익 곡선을 주간(금요일)으로 줄여 JSON에 담는다."""
    w = eq.resample("W-FRI").last().dropna(how="all")
    return {"dates": [str(d.date()) for d in w.index], "series": {c: [round(float(v), 5) for v in w[c].values] for c in w.columns}}


def run_rule_backtest(index_map: Dict[str, pd.DataFrame], setup_price_map: Optional[Dict[str, pd.DataFrame]] = None,
                      markets: Optional[List[str]] = None, workers: int = 1,
                      progress: Optional[Callable[[str, int, int], None]] = None,
                      generated_at: Optional[str] = None, source: str = "yfinance") -> Dict[str, Any]:
    """게이트 백테스트(시장별)와 T 셋업 × 게이트 집계를 한 번에 실행한다.
    index_map: QQQ·SPY·^KS11·^VIX 일봉, setup_price_map: {티커: 일봉} (없으면 셋업 검증 생략).
    반환: {"summary": JSON으로 저장할 요약, "gate_series": {시장: DataFrame}, "equity": {시장: DataFrame}, "trades": DataFrame}"""
    markets = [m.upper() for m in (markets or ["US", "KR"])]
    pm = {k: _bt_naive(v) for k, v in (index_map or {}).items()}
    summary: Dict[str, Any] = {"generated_at": generated_at, "source": source, "markets": {},
                               "config": {k: v for k, v in BACKTEST_CONFIG.items() if k != "SETUP_UNIVERSE"},
                               "missing_inputs": [k for k in ("QQQ", "SPY", "^KS11", "^VIX") if pm.get(k) is None]}
    summary["config"]["EXPOSURE_BY_GATE"] = {f"G{k}": v for k, v in BACKTEST_CONFIG["EXPOSURE_BY_GATE"].items()}
    summary["config"]["T_SIZE_BY_GATE"] = {f"G{k}": v for k, v in BACKTEST_CONFIG["T_SIZE_BY_GATE"].items()}
    gate_series, equity = {}, {}
    for m in markets:
        bench_sym = BACKTEST_CONFIG["BENCH"][m]
        bench = pm.get(bench_sym)
        if bench is None or pm.get("^VIX") is None:
            summary["markets"][m] = {"status": "데이터 없음", "reason": f"{bench_sym} 또는 ^VIX 가격 없음"}
            continue
        g = compute_gate_series(pm, m, workers=workers, progress=(lambda d, n, _m=m: progress(_m, d, n)) if progress else None)
        if len(g) == 0:
            summary["markets"][m] = {"status": "데이터 없음", "reason": f"{bench_sym} 기간이 {BACKTEST_CONFIG['MIN_HISTORY_DAYS']}거래일보다 짧음"}
            continue
        price = _price_col(bench)
        fwd = gate_forward_stats(g, price)
        expo = gate_exposure_backtest(g, price)
        gate_series[m], equity[m] = g, expo["equity"]
        summary["markets"][m] = {
            "status": "완료", "bench": bench_sym,
            "price_basis": "배당 포함(Adj Close)" if ("Adj Close" in bench and m == "US") else "종가(지수, 배당 미포함)",
            "as_of": str(g.index[-1].date()), "period": fwd["period"], "time_share": fwd["time_share"],
            "forward": {h: fwd[h] for h in fwd if h not in ("time_share", "period")},
            "exposure": {"strategies": expo["strategies"], "assumptions": expo["assumptions"]},
            "equity_weekly": _weekly_equity(expo["equity"]),
            "current": {"level": int(g["level"].iloc[-1]), "raw_level": int(g["raw_level"].iloc[-1])},
            "attribution": gate_component_attribution(g),
            "split": gate_split_stats(g, price),
            "exposure_no_cost": {k: v for k, v in gate_exposure_backtest(g, price, cost_one_way=0.0)["strategies"].items()
                                 if k in ("게이트(최종)", "게이트(즉시)", "같은 평균 노출 고정")},
        }

    trades_df = pd.DataFrame()
    if setup_price_map:
        base_cost = float(TECH_PARAMS["ROUND_TRIP_COST"])
        costs = sorted(set([float(c) for c in BACKTEST_CONFIG["SETUP_COST_SCENARIOS"]] + [base_cost]))
        rows, rows_by_cost = [], {c: [] for c in costs}
        groups = {s: grp for grp, syms in BACKTEST_CONFIG["SETUP_UNIVERSE"].items() for s in syms}
        for sym, df in setup_price_map.items():
            grp = groups.get(sym, "추가 종목")
            for c, trs in setup_trades_by_cost(df, costs).items():
                rows_by_cost[c].extend({"ticker": sym, "group": grp, **tr} for tr in trs)
        rows = rows_by_cost[base_cost]
        trades_df = attach_gate_to_trades(pd.DataFrame(rows), gate_series) if rows else pd.DataFrame()
        summary["setups"] = summarize_setup_trades_by_gate(trades_df) if len(trades_df) else {"n_trades": 0}
        summary["setups"]["tickers_requested"] = len(setup_price_map)
        summary["setups"]["period"] = {"start": str(trades_df["date"].min().date()), "end": str(trades_df["date"].max().date())} if len(trades_df) else None
        summary["setups"]["base_cost"] = base_cost
        summary["setups"]["cost_sensitivity"] = summarize_setup_cost_sensitivity(
            {c: pd.DataFrame(r) for c, r in rows_by_cost.items()})
    prices = {m: _price_col(pm[BACKTEST_CONFIG["BENCH"][m]]) for m in gate_series}
    ohlc = {m: pm[BACKTEST_CONFIG['BENCH'][m]] for m in gate_series}
    variants = run_variant_analysis(gate_series, prices, trades_df, ohlc) if gate_series else None
    return {"summary": summary, "gate_series": gate_series, "equity": equity, "trades": trades_df, "variants": variants}


def _pct(v: Any, nd: int = 1, sign: bool = True) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    return f"{v * 100:+.{nd}f}%" if sign else f"{v * 100:.{nd}f}%"


def _num_txt(v: Any, nd: int = 2) -> str:
    return "—" if v is None else f"{v:.{nd}f}"


def _r_cell(s: Dict[str, Any], min_n: int = 5) -> str:
    n = s.get("sample_n", 0)
    if not n:
        return "표본 없음"
    if n < min_n:
        return f"표본 부족 (n={n})"
    se = f" ±{s['se_r']:.2f}" if s.get("se_r") is not None else ""
    return f"{s['expectancy_r']:+.2f}R{se} · 승률 {s['win_rate']:.0f}% · n={n}"


COMPONENT_LABELS = {"regime": "지수 레짐", "vix": "VIX", "macro": "거시 국면(VIX·레짐으로 만든 값)"}


def _render_attribution(att: Dict[str, Any]) -> List[str]:
    """단계별 원인 분해 표."""
    if not att:
        return []
    L = ["", "### 무엇 때문에 그 단계였나",
         "| 단계 | 일수 | 지수 레짐 | VIX | 거시 국면 | 레짐만 | VIX만 | 거시 국면만 | 완화 대기 |",
         "|---|---|---|---|---|---|---|---|---|"]
    for k in ("G1", "G2", "G3"):
        a = att.get(k, {})
        if not a.get("n_days"):
            continue
        b, o = a["binding"], a["only"]
        L.append(f"| {k} | {a['n_days']:,} ({a['share_of_time']:.0f}%) | {b['regime']:.0f}% | {b['vix']:.0f}% | {b['macro']:.0f}% | "
                 f"{o['regime']:.0f}% | {o['vix']:.0f}% | {o['macro']:.0f}% | {a['held_by_hysteresis']:.0f}% |")
    L.append("- 앞 세 칸은 그 단계를 만든 요소의 비중(동시에 여럿일 수 있어 합이 100%를 넘음), '~만'은 그 요소 하나 때문이었던 날, "
             "'완화 대기'는 지표는 이미 내려왔지만 2일 확인 때문에 단계가 유지된 날입니다.")
    g3 = att.get("G3", {})
    if g3.get("n_days"):
        L.append(f"- G3 중 거시 국면만 G3였던 날 {g3['only']['macro']:.0f}%: 거시 국면은 VIX 1년 백분위 80% 이상이면 '침체'(G3)가 되므로, "
                 f"VIX 자체 기준(90%ile·35)에는 못 미친 날이 여기에 들어갑니다.")
    rc = att.get("regime_crisis")
    if rc:
        L.append(f"- 레짐이 위기(지수가 200일 지수이동평균 아래)였던 {rc['n_days']:,}일 중 QQQ만 위기였던 날 {rc['qqq_only_pct']:.0f}% "
                 f"(SPY도 위기 {rc['spy_crisis_pct']:.0f}%). 미국 게이트는 QQQ·SPY 중 나쁜 쪽을 씁니다.")
    return L


def _render_split(sp: Dict[str, Any], market_name: str) -> List[str]:
    """앞 기간·뒤 기간 비교 표."""
    if not sp:
        return []
    h = sp.get("horizon")
    L = ["", f"### 기간을 나눠 보기 (기준일 {sp.get('split_date')})",
         f"| 기간 | G0 {h}일 수익률 | G3 {h}일 수익률 | G3 / G0 변동성 | G3 시간 비중 | 게이트 샤프 | 고정 비중 샤프 | 게이트 최대 낙폭 | 고정 비중 최대 낙폭 |",
         "|---|---|---|---|---|---|---|---|---|"]
    for name in ("앞 기간", "뒤 기간"):
        r = sp.get(name, {})
        if not r.get("forward"):
            L.append(f"| {name} | 데이터 부족 ({r.get('n_days', 0)}일) | | | | | | | |")
            continue
        fw, ex = r["forward"], r["exposure"]
        g0, g3 = fw.get("G0", {}), fw.get("G3", {})
        vr = (f"{g3['mean_vol'] / g0['mean_vol']:.1f}배" if g0.get("mean_vol") and g3.get("mean_vol") else "—")
        gate, const = ex.get("게이트(최종)", {}), ex.get("같은 평균 노출 고정", {})
        per = r.get("period") or {}
        L.append(f"| {name} ({str(per.get('start'))[:4]}~{str(per.get('end'))[:4]}) | "
                 f"{_pct(g0.get('mean_ret')) if g0.get('n_days') else '—'} | {_pct(g3.get('mean_ret')) if g3.get('n_days') else '—'} | {vr} | "
                 f"{r['time_share']['G3']:.0f}% | {_num_txt(gate.get('sharpe'))} | {_num_txt(const.get('sharpe'))} | "
                 f"{_pct(gate.get('mdd'))} | {_pct(const.get('mdd'))} |")
    L.append(f"- 두 기간에서 같은 방향이면 우연보다는 규칙의 성질에 가깝습니다. 규칙을 바꿀 때는 앞 기간으로 정하고 뒤 기간으로 확인합니다({market_name}).")
    return L


def _render_cost_sensitivity(cs: Dict[str, Any], base_cost: Optional[float]) -> List[str]:
    """T 셋업 비용 민감도 표."""
    if not cs:
        return []
    L = ["", "### 비용 민감도 (왕복 비용별 기대값, 게이트와 무관하게 전체 거래)",
         "| 왕복 비용 | ETF 미국 | ETF 한국 | 대표 종목 미국 | 대표 종목 한국 |", "|---|---|---|---|---|"]
    for key, block in cs.items():
        mark = " (현재)" if base_cost is not None and key == f"{base_cost * 100:.2f}%" else ""
        cell = lambda grp, m: _r_cell((block.get(grp) or {}).get(m) or {})
        L.append(f"| {key}{mark} | {cell('ETF', 'US')} | {cell('ETF', 'KR')} | {cell('대표 종목', 'US')} | {cell('대표 종목', 'KR')} |")
    L.append("- 비용은 목표가·손절 단위 계산에도 들어가므로 비용을 바꾸면 거래 결과 자체가 다시 계산됩니다. 비용 0%에서도 음수면 비용이 아니라 신호 자체에 우위가 없다는 뜻입니다.")
    return L


def render_backtest_report(summary: Dict[str, Any]) -> str:
    """요약(dict) → 마크다운 보고서. 숫자를 사실대로 비교만 하고, 좋고 나쁨의 결론은 한계와 함께 적는다."""
    L: List[str] = ["# 판정 규칙 백테스트 결과", ""]
    L.append(f"- 생성: {summary.get('generated_at') or '—'} · 데이터: {summary.get('source')} 일봉")
    cfg = summary.get("config", {})
    L.append(f"- 게이트는 날짜마다 그날 이전 {cfg.get('PRICE_WINDOW_YEARS')}년 데이터로 앱과 같은 함수로 다시 계산 "
             f"(미래 데이터 미사용, 한국은 미국 당일 데이터 제외). 비중 조정은 {cfg.get('EXEC_LAG_DAYS')}거래일 뒤 종가, "
             f"비용 편도 {cfg.get('COST_ONE_WAY', 0) * 100:.2f}%.")
    if summary.get("missing_inputs"):
        L.append(f"- 받지 못한 입력: {', '.join(summary['missing_inputs'])}")
    L.append("")
    h = str(cfg.get("MAIN_HORIZON", 20))
    for m, r in summary.get("markets", {}).items():
        name = MARKET_NAMES.get(m, m)
        L.append(f"## {name} 국면 게이트 ({r.get('bench', '')} 기준)")
        if r.get("status") != "완료":
            L += [f"데이터 없음 — {r.get('reason')}", ""]
            continue
        ts = r["time_share"]
        L.append(f"기간 {r['period']['start']} ~ {r['period']['end']} · 가격 {r['price_basis']} · "
                 f"시간 비중 G0 {ts['G0']:.0f}% · G1 {ts['G1']:.0f}% · G2 {ts['G2']:.0f}% · G3 {ts['G3']:.0f}%")
        L.append("")
        L.append(f"### 게이트 단계별 이후 {h}거래일")
        L.append("| 단계 | 일수 (구간 수) | 평균 수익률 ±표준오차 | 상승 확률 | 평균 최대 낙폭 | 하위 10% 낙폭 | 연환산 변동성 |")
        L.append("|---|---|---|---|---|---|---|")
        fw = r["forward"].get(h, {})
        for k in ["G0", "G1", "G2", "G3", "전체"]:
            s = fw.get(k, {})
            if not s.get("n_days"):
                L.append(f"| {k} | 0 | — | — | — | — | — |")
                continue
            se = f" ±{s['se_ret'] * 100:.1f}%p" if s.get("se_ret") is not None else ""
            L.append(f"| {k} | {s['n_days']:,} ({s.get('n_spells', '—')}) | {_pct(s['mean_ret'])}{se} | {s['pct_positive']:.0f}% | "
                     f"{_pct(s['mean_mdd'])} | {_pct(s['p10_mdd'])} | {_pct(s.get('mean_vol'), sign=False)} |")
        other = [x for x in r["forward"] if x != h]
        if other:
            L.append("")
            L.append("다른 기간 평균 수익률: " + " · ".join(
                f"{x}일 " + ", ".join(f"{k} {_pct(r['forward'][x].get(k, {}).get('mean_ret'))}" for k in ["G0", "G1", "G2", "G3"]
                                       if r["forward"][x].get(k, {}).get("n_days"))
                for x in other))
        # 표본이 있는 가장 공격적인 단계(보통 G0)와 G3 비교
        lo_k = next((k for k in ("G0", "G1", "G2") if fw.get(k, {}).get("n_days")), None)
        g0, g3 = fw.get(lo_k, {}) if lo_k else {}, fw.get("G3", {})
        L.append("")
        if g0.get("n_days") and g3.get("n_days"):
            L.append(f"- 위험 구분: {h}일 내 평균 최대 낙폭 G3 {_pct(g3['mean_mdd'])} vs {lo_k} {_pct(g0['mean_mdd'])}, "
                     f"변동성 G3 {_pct(g3.get('mean_vol'), sign=False)} vs {lo_k} {_pct(g0.get('mean_vol'), sign=False)}.")
            d = g3["mean_ret"] - g0["mean_ret"]
            se_d = math.sqrt((g3.get("se_ret") or 0) ** 2 + (g0.get("se_ret") or 0) ** 2)
            judge = ("표준오차가 없어 판단 보류" if not se_d else
                     "표준오차 2배 안이라 우연과 구분하기 어려움" if abs(d) < 2 * se_d else "표준오차 2배보다 큼")
            L.append(f"- 수익 차이: G3 − {lo_k} 평균 수익률 {d * 100:+.1f}%p ({judge}).")
        L.append("")
        L.append("### 게이트대로 노출을 줄였을 때")
        L.append("| 전략 | 연수익률 | 변동성 | 샤프 | 최대 낙폭 | 칼마 | 평균 노출 | 연간 비중 변경 |")
        L.append("|---|---|---|---|---|---|---|---|")
        for nm, p in r["exposure"]["strategies"].items():
            L.append(f"| {nm} | {_pct(p.get('cagr'))} | {_pct(p.get('vol'), sign=False)} | {_num_txt(p.get('sharpe'))} | "
                     f"{_pct(p.get('mdd'))} | {_num_txt(p.get('calmar'))} | {_pct(p.get('avg_exposure'), 0, sign=False) if p.get('avg_exposure') is not None else '100%'} | "
                     f"{_num_txt(p.get('switches_per_year'), 1) if p.get('switches_per_year') is not None else '—'} |")
        st = r["exposure"]["strategies"]
        gf, cs = st.get("게이트(최종)", {}), st.get("같은 평균 노출 고정", {})
        if gf.get("sharpe") is not None and cs.get("sharpe") is not None:
            L.append("")
            L.append(f"- 타이밍 효과(같은 평균 노출 고정과 비교): 샤프 {gf['sharpe'] - cs['sharpe']:+.2f}, "
                     f"최대 낙폭 {(gf['mdd'] - cs['mdd']) * 100:+.1f}%p, 연수익률 {(gf['cagr'] - cs['cagr']) * 100:+.1f}%p.")
        gi = st.get("게이트(즉시)")
        if gi and gf.get("sharpe") is not None and gi.get("sharpe") is not None:
            L.append(f"- 완화 2일 확인(히스테리시스) 효과: 샤프 {gf['sharpe'] - gi['sharpe']:+.2f}, "
                     f"연간 비중 변경 {gi.get('switches_per_year', 0):.1f}회 → {gf.get('switches_per_year', 0):.1f}회.")
        a = r["exposure"]["assumptions"]["exposure_by_gate"]
        L.append("- 노출 비중 가정: " + " · ".join(f"{k} {v * 100:.0f}%" for k, v in a.items()) + " (모델 가정, 실제 게이트는 신규 진입만 제한)")
        nc = r.get("exposure_no_cost") or {}
        if nc.get("게이트(최종)", {}).get("sharpe") is not None and nc.get("같은 평균 노출 고정", {}).get("sharpe") is not None:
            g_nc, c_nc = nc["게이트(최종)"], nc["같은 평균 노출 고정"]
            L.append(f"- 비용 0일 때: 게이트 샤프 {g_nc['sharpe']:.2f} · 연수익률 {_pct(g_nc['cagr'])} vs 같은 평균 노출 고정 샤프 {c_nc['sharpe']:.2f} · "
                     f"{_pct(c_nc['cagr'])} → 비용을 빼도 타이밍 효과 샤프 {g_nc['sharpe'] - c_nc['sharpe']:+.2f}.")
        L += _render_attribution(r.get("attribution") or {})
        L += _render_split(r.get("split") or {}, name)
        L.append("")

    s = summary.get("setups")
    if s is not None:
        L.append("## T 셋업(ST-1·ST-2) × 신호일 게이트")
        if not s.get("n_trades"):
            L += ["거래 없음 (가격 데이터를 받지 못했거나 신호가 없음)", ""]
        else:
            per = s.get("period") or {}
            L.append(f"종목 {s['n_tickers']}개 (요청 {s.get('tickers_requested')}) · 거래 {s['n_trades']:,}건 · {per.get('start')} ~ {per.get('end')} · "
                     f"게이트 시작 전 거래 {s.get('excluded_no_gate', 0)}건 제외. R = 손절폭 1단위 대비 손익"
                     f"(왕복 비용 {float(s.get('base_cost') or TECH_PARAMS['ROUND_TRIP_COST']) * 100:.2f}% 차감).")
            sz = cfg.get("T_SIZE_BY_GATE", {})
            L.append("게이트 적용 = 진입 크기 " + " · ".join(f"{k} {v * 100:.0f}%" for k, v in sz.items()) + " (GATE_CONFIG 규칙)")
            for grp, block in s["by_group"].items():
                L.append("")
                L.append(f"### {grp}")
                L.append("| 셋업 | 전체 | G0 | G1 | G2 | G3 | 게이트 적용 (합계 R · 기대값) | 미적용 |")
                L.append("|---|---|---|---|---|---|---|---|")
                for setup, row in block.items():
                    ga = row["게이트 적용"]
                    L.append(f"| {setup} | {_r_cell(row['전체'])} | {_r_cell(row['G0'])} | {_r_cell(row['G1'])} | {_r_cell(row['G2'])} | {_r_cell(row['G3'])} | "
                             f"{ga['sum_r_gated']:+.1f}R · {_num_txt(ga['expectancy_gated'])}R ({ga['trades_taken']}건) | "
                             f"{ga['sum_r_ungated']:+.1f}R · {_num_txt(ga['expectancy_ungated'])}R ({ga['trades_all']}건) |")
        L += _render_cost_sensitivity(s.get("cost_sensitivity") or {}, s.get("base_cost"))   # 게이트와 무관한 전체 거래
        L.append("")

    L += ["## 이 결과를 읽을 때 주의할 점",
          "- 과거 성과는 미래를 보장하지 않습니다. 이 결과를 보고 임계값(VIX 기준선 등)을 다시 맞추면 같은 데이터에 과하게 맞춰질(과최적화) 위험이 큽니다. 바꾸려면 기간을 나눠(앞 기간으로 정하고 뒤 기간으로 확인) 검증하세요.",
          "- 게이트별 노출 비중은 모델 가정입니다. 실제 게이트는 보유분이 아니라 신규 진입만 제한하므로, 노출 전략 결과는 게이트의 위험 판단을 단순화한 대리 지표입니다.",
          "- 이후 기간이 서로 겹쳐 날짜 수만큼 독립 표본이 있는 게 아닙니다. 표준오차는 '일수 ÷ 기간'을 독립 표본 수로 본 대략치이고, 구간 수가 적은 단계(특히 G3)는 몇 번의 위기에 결과가 좌우됩니다.",
          "- '대표 종목'은 지금 시점의 강자라 과거에 미리 알 수 없었던 선택입니다(사후 선택·생존 편향). ETF 결과를 더 신뢰하세요.",
          "- T 셋업 검증은 레이더 카드의 과거 검증과 같은 단순화 규칙(ST-1 이격도 반전·ST-2 돌파)이며, 실제 셋업 스캐너의 모든 조건을 재현하지는 않습니다. 목표가는 장중 고가, 손절은 종가 기준이라 결과가 약간 낙관적일 수 있습니다.",
          "- 게이트의 '거시 국면'은 정의상 VIX·지수 레짐으로 만든 값입니다(FRED 거시 지표 아님). 실적·가치·거시 트랙은 과거 시점의 컨센서스·재무 데이터가 없어 이 검증에 포함하지 않았습니다.",
          "- 야후 데이터 오류·누락이 있을 수 있습니다. 한국은 KOSPI 지수(배당 미포함), 미국은 SPY 배당 포함 가격으로 쟀습니다.",
          ""]
    return "\n".join(L)
