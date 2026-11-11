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
from .backtest import _bt_summary, run_setup_backtest_validation
from .home import NARRATIVE_BASKETS_SPEC, build_regime_vix_header
from .gate import build_gate_inputs_from_header, evaluate_market_gate, gate_components

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
    return {"date": t, "inputs": inp}


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
        }

    trades_df = pd.DataFrame()
    if setup_price_map:
        rows = []
        groups = {s: grp for grp, syms in BACKTEST_CONFIG["SETUP_UNIVERSE"].items() for s in syms}
        for sym, df in setup_price_map.items():
            for tr in setup_trades_for_ticker(df):
                rows.append({"ticker": sym, "group": groups.get(sym, "추가 종목"), **tr})
        trades_df = attach_gate_to_trades(pd.DataFrame(rows), gate_series) if rows else pd.DataFrame()
        summary["setups"] = summarize_setup_trades_by_gate(trades_df) if len(trades_df) else {"n_trades": 0}
        summary["setups"]["tickers_requested"] = len(setup_price_map)
        summary["setups"]["period"] = {"start": str(trades_df["date"].min().date()), "end": str(trades_df["date"].max().date())} if len(trades_df) else None
    return {"summary": summary, "gate_series": gate_series, "equity": equity, "trades": trades_df}


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
        L.append("")

    s = summary.get("setups")
    if s is not None:
        L.append("## T 셋업(ST-1·ST-2) × 신호일 게이트")
        if not s.get("n_trades"):
            L += ["거래 없음 (가격 데이터를 받지 못했거나 신호가 없음)", ""]
        else:
            per = s.get("period") or {}
            L.append(f"종목 {s['n_tickers']}개 (요청 {s.get('tickers_requested')}) · 거래 {s['n_trades']:,}건 · {per.get('start')} ~ {per.get('end')} · "
                     f"게이트 시작 전 거래 {s.get('excluded_no_gate', 0)}건 제외. R = 손절폭 1단위 대비 손익(비용 0.30% 차감).")
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
