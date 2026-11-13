"""T 트랙 검증과 재설계 후보 (2026-11-13 추가).

1) 실제 스캐너 재현: 날짜마다 앱(레이더)과 같은 analyze_technical_track을 그날 이전 2년 데이터로 호출해
   '진입' 티켓이 나온 날만 골라, 티켓의 진입·손절·1차 목표·시간 손절·최대 보유 규칙으로 청산한다.
   같은 티켓 모양(손절·목표 거리)을 무작위 날짜에 적용한 결과와 비교한다.
2) 재설계 후보: 오래 알려진 원칙을 관행 매개변수 그대로(데이터로 숫자를 맞추지 않음) 고정했다.
   - T-A 추세 추종(돌파): 종가 > 200일선 + 직전 55일 최고 종가 돌파 → 다음 날 시가 진입,
     손절 = 진입가 − 2×ATR(20), 종가가 직전 20일 최저 종가 아래면 청산 (터틀 방식)
   - T-B 눌림목 반등: 종가 > 200일선 + RSI(2) < 10 → 다음 날 시가 진입, 종가 > 5일선이면 청산 (코너스 방식)
   판정은 지금까지 한 번도 보지 않은 해외 국가 ETF 묶음(HOLDOUT)에서 한다.
판정 기준은 결과를 보기 전에 TBT_CRITERIA에 고정했다.
"""
import math
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from .common import TECH_PARAMS
from .radar import radar_neutral_value
from .rule_backtest import BACKTEST_CONFIG, _bt_naive

TBT_CONFIG: Dict[str, Any] = {
    "SCANNER_START": "2014-01-01",       # 실제 스캐너 재현 시작일 (사용자 선택: ETF 19개, 2014년~)
    "DATA_START": "2011-06-01",          # 2년 창 + 지표 준비 기간
    "WINDOW_YEARS": 2,                   # 레이더 PRICE_PERIOD "2y"와 같게
    "MIN_BARS": 260,                     # RADAR_CONFIG MIN_BARS와 같음
    "FX_USDKRW": 1350.0,                 # 과거 환율 대신 고정값: 유동성 기준(원화 환산)과 정수 주식 수에만 영향
    "ACCOUNT": {"total_equity": 100_000_000, "t_sleeve_ratio": 0.30},   # 앱 기본값(utils.init_session)
    "PARTIAL_AT_T1": 0.5,                # 1차 목표에서 절반 청산, 나머지는 이동평균 종가 이탈 또는 최대 보유로 청산
    "TRAIL_MA": {"ST-1": "SMA25"},       # 나머지 청산 기준선 (없으면 SMA20). 티켓 target_2 문구 기준
    "BASELINE_SAMPLES": 20,              # 실제 거래 1건당 무작위 비교 표본 수
    "SEED": 7,
    "SCANNER_UNIVERSE": BACKTEST_CONFIG["SETUP_UNIVERSE"]["ETF"],
    # 재설계 후보 검증용: 지금까지 어떤 백테스트에서도 쓰지 않은 해외 국가 ETF (iShares MSCI)
    "HOLDOUT_UNIVERSE": ["EWJ", "EWG", "EWU", "EWC", "EWA", "EWZ", "EWT", "EWH", "EWS", "EWW",
                         "EWL", "EWQ", "EWP", "EWI", "EWN", "EWD", "EZA", "EWM"],
    "REDESIGN_START": "2001-01-01",
}
TBT_CRITERIA: Dict[str, Any] = {
    # 실제 스캐너: ETF 전체 합산 평균 R > 0 이고 (실제 − 무작위) ≥ 표준오차 × 2
    "SCANNER_MIN_Z": 2.0,
    # 재설계 후보: HOLDOUT에서 평균 R > 0 이고 (후보 − 무작위) ≥ 표준오차 × 2, 그리고 기존 ETF 19개(2001~)에서도 평균 R > 0
    "REDESIGN_MIN_Z": 2.0,
}
# 스캐너 판정 캐시(CSV) 열 — 묶음마다 열이 달라 파일이 어긋나지 않도록 고정
SCANNER_CACHE_COLUMNS = ["ticker", "date", "verdict", "setup_id", "executable", "grade", "entry_price", "stop_price",
                         "target_1", "time_stop_days", "time_stop_min_price", "max_hold_days", "chase_ban_price", "expected_r"]
REDESIGN_CANDIDATES: Dict[str, str] = {
    "T-A 추세 추종": "종가 > 200일선 + 직전 55일 최고 종가 돌파 → 다음 날 시가 진입. 손절 진입가 − 2×ATR(20), 종가 < 직전 20일 최저 종가면 청산",
    "T-B 눌림목 반등": "종가 > 200일선 + RSI(2) < 10 → 다음 날 시가 진입. 종가 > 5일선이면 청산 (손절 없음, R 단위는 2×ATR(20))",
}


def _cost() -> float:
    return float(TECH_PARAMS["ROUND_TRIP_COST"])


def _int(v: Any, default: int) -> int:
    try:
        f = float(v)
        return int(f) if np.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def _market(sym: str) -> str:
    return "KR" if str(sym).upper().endswith((".KS", ".KQ")) else "US"


# ---------------------------------------------------------------------
# 1) 실제 스캐너 재현
# ---------------------------------------------------------------------
def _window(df: Optional[pd.DataFrame], t: pd.Timestamp, strict: bool = False) -> Optional[pd.DataFrame]:
    if df is None:
        return None
    lo = t - pd.DateOffset(years=int(TBT_CONFIG["WINDOW_YEARS"]))
    out = df.loc[(df.index > lo) & ((df.index < t) if strict else (df.index <= t))]
    return out if len(out) else None


def scanner_ticket_on(df: pd.DataFrame, bench: Optional[pd.DataFrame], vix: Optional[pd.Series], t: Any, market: str) -> Dict[str, Any]:
    """날짜 t 종가 시점의 레이더 판정. 앱 레이더(build_radar_signal)와 같은 입력으로 analyze_technical_track을 부른다.
    한국 종목은 그날 미국 VIX를 아직 모르므로 VIX는 전날까지만."""
    from .technical import analyze_technical_track
    t = pd.Timestamp(t)
    px = _window(df, t)
    if px is None or len(px) < int(TBT_CONFIG["MIN_BARS"]):
        return {"date": t, "verdict": "데이터 부족", "executable": False}
    px = px.dropna(subset=["Open", "High", "Low", "Close"])
    b = _window(bench, t)
    v = vix.loc[(vix.index < t) if market == "KR" else (vix.index <= t)] if vix is not None else None
    cfg = {**TBT_CONFIG["ACCOUNT"], "is_krx": market == "KR",
           "fx_to_krw": 1.0 if market == "KR" else float(TBT_CONFIG["FX_USDKRW"]), "earnings_blackout": False}
    out = analyze_technical_track(px, b, v, value_macro_input=radar_neutral_value(), account_cfg=cfg)
    tk = out.get("t_ticket") or {}
    row = {"date": t, "verdict": (out.get("summary_banner") or {}).get("t_verdict") or (out.get("summary_banner") or {}).get("track_signal"),
           "setup_id": tk.get("setup_id"), "executable": bool(tk.get("is_executable"))}
    if row["executable"]:
        for k in ("grade", "entry_price", "stop_price", "target_1", "time_stop_days", "time_stop_min_price", "max_hold_days", "chase_ban_price", "expected_r"):
            row[k] = tk.get(k)
    return row


_TW: Dict[str, Any] = {}


def _tw_init(price_map: Dict[str, pd.DataFrame], bench_map: Dict[str, pd.DataFrame], vix: Optional[pd.Series]) -> None:
    _TW["pm"], _TW["bm"], _TW["vix"] = price_map, bench_map, vix
    # 속도: 매 호출마다 도는 셋업 과거 검증(결과 티켓과 무관)을 이 프로세스에서만 끈다
    from . import technical
    technical.run_setup_backtest_validation = lambda df, cost=None: {}


def _tw_run(job: tuple) -> List[Dict[str, Any]]:
    sym, dates = job
    m = _market(sym)
    rows = []
    for t in dates:
        try:
            r = scanner_ticket_on(_TW["pm"][sym], _TW["bm"].get(m), _TW["vix"], t, m)
        except Exception as e:
            r = {"date": pd.Timestamp(t), "verdict": f"오류: {type(e).__name__}", "executable": False}
        rows.append({"ticker": sym, **r})
    return rows


def compute_scanner_signals(price_map: Dict[str, pd.DataFrame], bench_map: Dict[str, pd.DataFrame], vix: Optional[pd.Series],
                            start: Optional[str] = None, end: Optional[str] = None, workers: int = 1,
                            done: Optional[Dict[str, set]] = None,
                            progress: Optional[Callable[[int, int], None]] = None,
                            on_chunk: Optional[Callable[[List[Dict[str, Any]]], None]] = None) -> pd.DataFrame:
    """종목 × 날짜별 레이더 판정. done={티커: 이미 계산한 날짜 집합}이면 건너뛴다(중간에 끊겨도 이어서 계산).
    on_chunk는 묶음마다 결과를 받아 저장할 때 쓴다."""
    pm = {k: _bt_naive(v) for k, v in price_map.items()}
    bm = {k: _bt_naive(v) for k, v in (bench_map or {}).items()}
    vx = _bt_naive(vix.to_frame("Close"))["Close"] if vix is not None else None
    s0 = pd.Timestamp(start or TBT_CONFIG["SCANNER_START"])
    jobs = []
    for sym, df in pm.items():
        if df is None:
            continue
        idx = df.index[df.index >= s0]
        if end:
            idx = idx[idx <= pd.Timestamp(end)]
        skip = (done or {}).get(sym, set())
        todo = [t for t in idx if t not in skip]
        jobs += [(sym, todo[i:i + 30]) for i in range(0, len(todo), 30)]
    total = sum(len(j[1]) for j in jobs)
    rows: List[Dict[str, Any]] = []
    if workers and workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=workers, initializer=_tw_init, initargs=(pm, bm, vx)) as ex:
            for res in ex.map(_tw_run, jobs):
                rows += res
                if on_chunk:
                    on_chunk(res)
                if progress:
                    progress(len(rows), total)
    else:
        from . import technical
        orig = technical.run_setup_backtest_validation
        try:
            _tw_init(pm, bm, vx)
            for j in jobs:
                res = _tw_run(j)
                rows += res
                if on_chunk:
                    on_chunk(res)
                if progress:
                    progress(len(rows), total)
        finally:
            technical.run_setup_backtest_validation = orig
    return pd.DataFrame(rows)


def _trail_ma(df: pd.DataFrame, setup_id: Optional[str]) -> np.ndarray:
    win = 25 if TBT_CONFIG["TRAIL_MA"].get(setup_id or "", "SMA20") == "SMA25" else 20
    return df["Close"].rolling(win, min_periods=win // 2).mean().to_numpy(dtype=float)


def simulate_ticket_trade(o: np.ndarray, h: np.ndarray, c: np.ndarray, ma: np.ndarray, i: int, tk: Dict[str, Any],
                          cost: Optional[float] = None) -> Optional[Dict[str, Any]]:
    """i일 종가에 나온 티켓을 i+1일 시가에 진입해 청산까지 시뮬레이션한다. 추격 금지 가격을 넘는 갭이면 진입하지 않음(None).
    1차 목표 전: 손절가 아래 종가 → 전량, time_stop_days째 종가 < time_stop_min_price → 전량.
    1차 목표(장중 고가 도달) → PARTIAL_AT_T1 청산, 나머지는 손절가 아래 종가·이동평균 아래 종가·최대 보유일 종가에 청산."""
    cst = _cost() if cost is None else float(cost)
    n = len(c)
    if i + 1 >= n:
        return None
    fill = float(o[i + 1])
    chase = tk.get("chase_ban_price")
    if chase is not None and np.isfinite(fill) and fill > float(chase):
        return {"skipped": "추격 금지(갭 상승)"}
    stop, t1 = float(tk["stop_price"]), float(tk["target_1"])
    ts_days, ts_min = _int(tk.get("time_stop_days"), 0), tk.get("time_stop_min_price")
    if ts_min is not None and not np.isfinite(float(ts_min)):
        ts_min = None
    max_hold = _int(tk.get("max_hold_days"), 60)
    unit = max(fill * 0.001, (fill - stop) + fill * cst)
    part = float(TBT_CONFIG["PARTIAL_AT_T1"])
    legs: List[tuple] = []        # (비중, 청산가)
    left, hit_t1, reason, k_end = 1.0, False, "보유 중(데이터 끝)", None
    for k in range(i + 1, min(n, i + 1 + max_hold)):
        held = k - i
        if c[k] < stop:
            legs.append((left, float(c[k]))); left = 0.0
            reason, k_end = ("손절" if not hit_t1 else "1차 목표 후 손절"), k
            break
        if not hit_t1 and h[k] >= t1:
            px_t1 = max(t1, fill) if fill > t1 else t1
            legs.append((part, px_t1)); left -= part; hit_t1 = True
        if hit_t1 and np.isfinite(ma[k]) and c[k] < ma[k]:
            legs.append((left, float(c[k]))); left = 0.0
            reason, k_end = "1차 목표 후 이동평균 이탈", k
            break
        if not hit_t1 and ts_days and held == ts_days and ts_min is not None and c[k] < float(ts_min):
            legs.append((left, float(c[k]))); left = 0.0
            reason, k_end = "시간 손절", k
            break
        if held >= max_hold:
            legs.append((left, float(c[k]))); left = 0.0
            reason, k_end = ("최대 보유" if not hit_t1 else "1차 목표 후 최대 보유"), k
            break
    if left > 0:
        if k_end is None and i + max_hold < n:
            k_end = i + max_hold
        if k_end is None:
            return None            # 데이터 끝까지 청산되지 않음 → 제외
        legs.append((left, float(c[k_end]))); left = 0.0
    r = sum(w * (px - fill - fill * cst) for w, px in legs) / unit
    ret = sum(w * (px / fill - 1.0) for w, px in legs) - cst
    return {"fill": fill, "r": float(r), "ret": float(ret), "exit_reason": reason, "exit_idx": int(k_end), "hold_days": int(k_end - i)}


def simulate_scanner_trades(df: pd.DataFrame, signals: pd.DataFrame, cost: Optional[float] = None) -> pd.DataFrame:
    """한 종목: '진입' 티켓을 날짜순으로 처리. 보유 중에는 새 티켓을 무시한다(종목당 1포지션)."""
    d = _bt_naive(df)
    sig = signals[signals["executable"]].sort_values("date")
    o, h, c = (d[k].to_numpy(dtype=float) for k in ("Open", "High", "Close"))
    pos = {t: p for p, t in enumerate(d.index)}
    out, free_from = [], -1
    for _, s in sig.iterrows():
        i = pos.get(pd.Timestamp(s["date"]))
        if i is None or i <= free_from:
            continue
        ma = _trail_ma(d, s.get("setup_id"))
        res = simulate_ticket_trade(o, h, c, ma, i, s.to_dict(), cost)
        if res is None:
            continue
        if res.get("skipped"):
            out.append({"date": s["date"], "setup_id": s.get("setup_id"), "skipped": res["skipped"]})
            continue
        free_from = res["exit_idx"]
        out.append({"date": s["date"], "setup_id": s.get("setup_id"), "grade": s.get("grade"), **res})
    return pd.DataFrame(out)


def scanner_random_baseline(df: pd.DataFrame, trades: pd.DataFrame, signals: pd.DataFrame, start: Optional[str] = None,
                            samples: Optional[int] = None, seed: Optional[int] = None, cost: Optional[float] = None) -> pd.DataFrame:
    """실제 거래 1건마다 같은 종목의 무작위 날짜 samples개에, 그 티켓과 같은 모양(손절·목표·시간 손절 기준의 진입가 대비 비율,
    시간 손절·최대 보유 일수, 이동평균)을 적용한다. 진입 시점의 가치만 비교하기 위함."""
    d = _bt_naive(df)
    if trades is None or len(trades) == 0 or "r" not in trades:
        return pd.DataFrame()
    rng = np.random.default_rng(TBT_CONFIG["SEED"] if seed is None else seed)
    k = int(samples or TBT_CONFIG["BASELINE_SAMPLES"])
    o, h, c = (d[x].to_numpy(dtype=float) for x in ("Open", "High", "Close"))
    s0 = pd.Timestamp(start or TBT_CONFIG["SCANNER_START"])
    first = int(np.searchsorted(d.index.values, s0.to_datetime64()))
    sig = signals[signals["executable"]].set_index("date")
    out = []
    for _, tr in trades.dropna(subset=["r"]).iterrows():
        tk = sig.loc[pd.Timestamp(tr["date"])]
        if isinstance(tk, pd.DataFrame):
            tk = tk.iloc[0]
        e = float(tk["entry_price"])
        if not e:
            continue
        geo = {"stop": float(tk["stop_price"]) / e, "t1": float(tk["target_1"]) / e,
               "tsmin": (float(tk["time_stop_min_price"]) / e) if pd.notna(tk.get("time_stop_min_price")) else None,
               "chase": (float(tk["chase_ban_price"]) / e) if pd.notna(tk.get("chase_ban_price")) else None}
        ma = _trail_ma(d, tr.get("setup_id"))
        hi = len(c) - _int(tk.get("max_hold_days"), 60) - 2
        if hi <= first:
            continue
        for j in rng.integers(first, hi, size=k):
            ej = float(c[j])
            tkj = {"stop_price": ej * geo["stop"], "target_1": ej * geo["t1"],
                   "time_stop_days": tk.get("time_stop_days"), "max_hold_days": tk.get("max_hold_days"),
                   "time_stop_min_price": ej * geo["tsmin"] if geo["tsmin"] is not None else None,
                   "chase_ban_price": ej * geo["chase"] if geo["chase"] is not None else None}
            res = simulate_ticket_trade(o, h, c, ma, int(j), tkj, cost)
            if res and not res.get("skipped"):
                out.append({"setup_id": tr.get("setup_id"), "r": res["r"], "ret": res["ret"], "exit_reason": res["exit_reason"]})
    return pd.DataFrame(out)


def _stats(x: pd.Series) -> Dict[str, Any]:
    x = pd.Series(x, dtype=float).dropna()
    n = len(x)
    if n == 0:
        return {"n": 0}
    return {"n": n, "mean": float(x.mean()), "se": float(x.std(ddof=1) / math.sqrt(n)) if n >= 2 else None,
            "win_rate": float((x > 0).mean() * 100.0)}


def _compare(a: pd.Series, b: pd.Series) -> Dict[str, Any]:
    sa, sb = _stats(a), _stats(b)
    diff = z = None
    if sa.get("n", 0) >= 2 and sb.get("n", 0) >= 2:
        diff = sa["mean"] - sb["mean"]
        se = math.sqrt(sa["se"] ** 2 + sb["se"] ** 2)
        z = diff / se if se > 0 else None
    return {"actual": sa, "baseline": sb, "diff": diff, "z": z}


def summarize_scanner(trades: pd.DataFrame, baseline: pd.DataFrame) -> Dict[str, Any]:
    """전체·셋업별 실제 vs 무작위, 청산 이유 분포, 추격 금지로 건너뛴 수, 사전 기준 판정."""
    t = trades[trades["r"].notna()] if len(trades) and "r" in trades else pd.DataFrame(columns=["r", "setup_id", "exit_reason", "ret"])
    skipped = int(trades["skipped"].notna().sum()) if len(trades) and "skipped" in trades else 0
    out: Dict[str, Any] = {"n_trades": int(len(t)), "skipped_chase": skipped, "by_setup": {}}
    out["overall"] = _compare(t["r"], baseline["r"] if len(baseline) else pd.Series(dtype=float))
    out["overall_ret"] = _stats(t["ret"]) if len(t) else {"n": 0}
    for sid in sorted(t["setup_id"].dropna().unique().tolist()):
        out["by_setup"][sid] = _compare(t.loc[t["setup_id"] == sid, "r"],
                                        baseline.loc[baseline["setup_id"] == sid, "r"] if len(baseline) else pd.Series(dtype=float))
    out["exit_reasons"] = {k: {"share": float(v / len(t) * 100.0), "mean_r": float(t.loc[t["exit_reason"] == k, "r"].mean())}
                           for k, v in t["exit_reason"].value_counts().items()} if len(t) else {}
    ov = out["overall"]
    passed = bool(ov["actual"].get("n", 0) >= 2 and ov["actual"]["mean"] > 0 and ov["z"] is not None and ov["z"] >= TBT_CRITERIA["SCANNER_MIN_Z"])
    out["verdict"] = {"passed": passed, "criteria": f"평균 R > 0 이고 (실제 − 무작위) ≥ 표준오차 × {TBT_CRITERIA['SCANNER_MIN_Z']:.0f}"}
    return out


# ---------------------------------------------------------------------
# 2) 재설계 후보 (관행 매개변수 고정)
# ---------------------------------------------------------------------
def redesign_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """SMA200·SMA5·RSI(2, 와일더)·ATR(20, 와일더)·직전 55일 최고 종가·직전 20일 최저 종가(당일 제외)."""
    d = _bt_naive(df)
    c, h, lo = d["Close"].astype(float), d["High"].astype(float), d["Low"].astype(float)
    out = pd.DataFrame(index=d.index)
    out["Open"], out["High"], out["Close"] = d["Open"].astype(float), h, c
    out["SMA200"] = c.rolling(200, min_periods=200).mean()
    out["SMA5"] = c.rolling(5, min_periods=5).mean()
    delta = c.diff()
    up, dn = delta.clip(lower=0.0), (-delta).clip(lower=0.0)
    au = up.ewm(alpha=1 / 2, adjust=False, min_periods=2).mean()
    ad = dn.ewm(alpha=1 / 2, adjust=False, min_periods=2).mean()
    rs = au / ad.replace(0.0, np.nan)
    out["RSI2"] = (100.0 - 100.0 / (1.0 + rs)).where(ad > 0, 100.0)
    tr = pd.concat([h - lo, (h - c.shift()).abs(), (lo - c.shift()).abs()], axis=1).max(axis=1)
    out["ATR20"] = tr.ewm(alpha=1 / 20, adjust=False, min_periods=20).mean()
    out["HH55"] = c.shift(1).rolling(55, min_periods=55).max()
    out["LL20"] = c.shift(1).rolling(20, min_periods=20).min()
    return out


def _candidate_signal(ind: pd.DataFrame, name: str) -> np.ndarray:
    c = ind["Close"]
    if name.startswith("T-A"):
        return ((c > ind["SMA200"]) & (c > ind["HH55"])).to_numpy()
    if name.startswith("T-B"):
        return ((c > ind["SMA200"]) & (ind["RSI2"] < 10.0)).to_numpy()
    raise ValueError(name)


def simulate_candidate_trade(ind: pd.DataFrame, name: str, i: int, cost: Optional[float] = None,
                             arrs: Optional[Dict[str, np.ndarray]] = None) -> Optional[Dict[str, Any]]:
    """i일 종가 신호 → i+1일 시가 진입 → 후보 규칙대로 종가 청산. 데이터 끝까지 청산이 안 되면 None."""
    cst = _cost() if cost is None else float(cost)
    a = arrs or {k: ind[k].to_numpy(dtype=float) for k in ("Open", "Close", "SMA5", "ATR20", "LL20")}
    n = len(a["Close"])
    if i + 1 >= n or not np.isfinite(a["ATR20"][i]) or not np.isfinite(a["Open"][i + 1]):
        return None
    fill = float(a["Open"][i + 1])
    unit = 2.0 * float(a["ATR20"][i]) + fill * cst
    stop = fill - 2.0 * float(a["ATR20"][i]) if name.startswith("T-A") else None
    for k in range(i + 1, n):
        ck = a["Close"][k]
        if name.startswith("T-A"):
            if ck < stop:
                reason = "손절"
            elif np.isfinite(a["LL20"][k]) and ck < a["LL20"][k]:
                reason = "20일 최저 이탈"
            else:
                continue
        else:
            if np.isfinite(a["SMA5"][k]) and ck > a["SMA5"][k]:
                reason = "5일선 위 마감"
            else:
                continue
        return {"fill": fill, "r": float((ck - fill - fill * cst) / unit), "ret": float(ck / fill - 1.0 - cst),
                "exit_reason": reason, "exit_idx": k, "hold_days": k - i}
    return None


def candidate_trades(df: pd.DataFrame, name: str, start: Optional[str] = None, cost: Optional[float] = None) -> pd.DataFrame:
    """한 종목의 후보 규칙 거래 (종목당 1포지션, 신호일이 start 이후)."""
    ind = redesign_indicators(df)
    sig = _candidate_signal(ind, name)
    arrs = {k: ind[k].to_numpy(dtype=float) for k in ("Open", "Close", "SMA5", "ATR20", "LL20")}
    s0 = pd.Timestamp(start or TBT_CONFIG["REDESIGN_START"])
    out, free_from = [], -1
    for i in np.flatnonzero(sig):
        if ind.index[i] < s0 or i <= free_from:
            continue
        res = simulate_candidate_trade(ind, name, int(i), cost, arrs)
        if res is None:
            continue
        free_from = res["exit_idx"]
        out.append({"date": ind.index[i], **res})
    return pd.DataFrame(out)


def candidate_random_baseline(df: pd.DataFrame, name: str, n_trades: int, start: Optional[str] = None,
                              samples: Optional[int] = None, seed: Optional[int] = None, cost: Optional[float] = None) -> pd.DataFrame:
    """같은 종목·기간의 무작위 날짜에 같은 청산 규칙을 적용 (실제 거래 수 × samples개)."""
    ind = redesign_indicators(df)
    arrs = {k: ind[k].to_numpy(dtype=float) for k in ("Open", "Close", "SMA5", "ATR20", "LL20")}
    s0 = pd.Timestamp(start or TBT_CONFIG["REDESIGN_START"])
    valid = np.flatnonzero((ind.index >= s0) & np.isfinite(arrs["ATR20"]) & ind["SMA200"].notna().to_numpy())
    valid = valid[valid < len(ind) - 2]
    if len(valid) == 0 or n_trades == 0:
        return pd.DataFrame()
    rng = np.random.default_rng(TBT_CONFIG["SEED"] if seed is None else seed)
    out = []
    for j in rng.choice(valid, size=n_trades * int(samples or TBT_CONFIG["BASELINE_SAMPLES"]), replace=True):
        res = simulate_candidate_trade(ind, name, int(j), cost, arrs)
        if res:
            out.append({"date": ind.index[j], "r": res["r"], "ret": res["ret"], "exit_reason": res["exit_reason"]})
    return pd.DataFrame(out)


def evaluate_redesign(main_map: Dict[str, pd.DataFrame], holdout_map: Dict[str, pd.DataFrame],
                      split: Optional[str] = None) -> Dict[str, Any]:
    """후보별: 기존 ETF(앞·뒤·전체 기간)와 HOLDOUT(전체 기간)에서 실제 vs 무작위, 사전 기준 판정."""
    split_t = pd.Timestamp(split or BACKTEST_CONFIG["SPLIT_DATE"])
    res: Dict[str, Any] = {"candidates": REDESIGN_CANDIDATES, "criteria": TBT_CRITERIA, "results": {}, "verdict": {},
                           "n_tickers": {"기존 ETF": len(main_map or {}), "HOLDOUT": len(holdout_map or {})}}
    for name in REDESIGN_CANDIDATES:
        blocks = {}
        for uname, umap in (("기존 ETF", main_map), ("HOLDOUT", holdout_map)):
            tr_all, bs_all = [], []
            for sym, df in (umap or {}).items():
                d = _bt_naive(df)
                if d is None or len(d) < 260:
                    continue
                tr = candidate_trades(d, name)
                bs = candidate_random_baseline(d, name, len(tr))
                if len(tr):
                    tr_all.append(tr.assign(ticker=sym))
                if len(bs):
                    bs_all.append(bs.assign(ticker=sym))
            tr = pd.concat(tr_all) if tr_all else pd.DataFrame(columns=["date", "r", "ret", "exit_reason"])
            bs = pd.concat(bs_all) if bs_all else pd.DataFrame(columns=["date", "r", "ret", "exit_reason"])
            per = {"전체 기간": (tr, bs)}
            if uname == "기존 ETF":
                per["앞 기간"] = (tr[tr["date"] < split_t], bs[bs["date"] < split_t])
                per["뒤 기간"] = (tr[tr["date"] >= split_t], bs[bs["date"] >= split_t])
            blocks[uname] = {p: {**_compare(a["r"], b["r"]), "ret": _stats(a["ret"]),
                                 "hold_days": float(a["hold_days"].mean()) if len(a) and "hold_days" in a else None}
                             for p, (a, b) in per.items()}
        res["results"][name] = blocks
        ho, mf = blocks["HOLDOUT"]["전체 기간"], blocks["기존 ETF"]["전체 기간"]
        ok_ho = bool(ho["actual"].get("n", 0) >= 2 and ho["actual"]["mean"] > 0 and ho["z"] is not None and ho["z"] >= TBT_CRITERIA["REDESIGN_MIN_Z"])
        ok_main = bool(mf["actual"].get("n", 0) >= 2 and mf["actual"]["mean"] > 0)
        res["verdict"][name] = {"passed": ok_ho and ok_main, "holdout_pass": ok_ho, "main_positive": ok_main}
    return res


# ---------------------------------------------------------------------
# 보고서
# ---------------------------------------------------------------------
def _cmp_cells(x: Dict[str, Any]) -> str:
    def one(s):
        if not s.get("n"):
            return "표본 없음"
        se = f" ±{s['se']:.2f}" if s.get("se") is not None else ""
        return f"{s['mean']:+.2f}R{se} ({s['win_rate']:.0f}% · {s['n']:,}건)"
    d = "—" if x.get("diff") is None else f"{x['diff']:+.2f}R"
    z = "—" if x.get("z") is None else f"{x['z']:+.1f}"
    return f"{one(x['actual'])} | {one(x['baseline'])} | {d} | {z}"


def render_scanner_report(s: Dict[str, Any], generated_at: Optional[str] = None, meta: Optional[Dict[str, Any]] = None) -> str:
    m = meta or {}
    L = ["# 실제 T 셋업 스캐너 검증", "",
         f"- 생성: {generated_at or '—'} · 종목 {m.get('n_tickers', '—')}개(섹터 ETF) · {m.get('start', TBT_CONFIG['SCANNER_START'])}~{m.get('end', '—')} · "
         f"계산한 날 {m.get('n_days', '—')}일 · 왕복 비용 {_cost() * 100:.2f}%",
         "- 날짜마다 앱 레이더와 같은 analyze_technical_track을 그날 이전 2년 데이터로 호출해 '진입' 티켓이 나온 날만 다음 날 시가에 진입했습니다. "
         "추격 금지 가격 위로 갭 상승하면 진입하지 않았고, 보유 중에는 같은 종목의 새 티켓을 무시했습니다.",
         f"- 청산: 손절가 아래 종가 · 시간 손절(N일째 기준가 미달) · 1차 목표(장중 고가)에서 {TBT_CONFIG['PARTIAL_AT_T1'] * 100:.0f}% 청산 후 나머지는 "
         "이동평균(ST-1 25일, 그 외 20일) 아래 종가 또는 최대 보유일.",
         f"- 무작위 비교: 실제 거래마다 같은 종목의 무작위 날짜 {TBT_CONFIG['BASELINE_SAMPLES']}곳에 같은 티켓 모양(손절·목표의 진입가 대비 비율, 시간 규칙)을 적용.",
         f"- 판정 기준(결과 보기 전 고정): {s['verdict']['criteria']}.", "",
         f"## 판정: {'통과' if s['verdict']['passed'] else '미통과'}", "",
         f"실제 거래 {s['n_trades']:,}건 · 추격 금지로 건너뜀 {s.get('skipped_chase', 0)}건 · 거래당 평균 수익률 "
         f"{'—' if not s['overall_ret'].get('n') else format(s['overall_ret']['mean'] * 100, '+.2f') + '%'}", "",
         "| 구분 | 실제 평균 R ±표준오차 (승률·건수) | 무작위 평균 R ±표준오차 (승률·건수) | 차이 | z |", "|---|---|---|---|---|",
         f"| 전체 | {_cmp_cells(s['overall'])} |"]
    for sid, x in s.get("by_setup", {}).items():
        L.append(f"| {sid} | {_cmp_cells(x)} |")
    if s.get("exit_reasons"):
        L += ["", "### 청산 이유 (실제 거래)", "| 청산 | 비중 | 평균 R |", "|---|---|---|"]
        for k, v in sorted(s["exit_reasons"].items(), key=lambda kv: -kv[1]["share"]):
            L.append(f"| {k} | {v['share']:.0f}% | {v['mean_r']:+.2f}R |")
    L += ["", "## 주의",
          "- 과거 환율 대신 고정 환율(유동성 기준·주식 수에만 영향), 실적 발표 일정은 ETF라 없음으로 계산했습니다.",
          "- 1차 목표 이후 청산은 티켓 문구(이동평균 트레일링)를 단순화한 것입니다. 국면 게이트는 적용하지 않은 원신호 기준입니다.",
          "- 같은 날 여러 종목이 함께 움직여 거래끼리 완전히 독립이 아니므로 표준오차는 실제보다 작을 수 있습니다.", ""]
    return "\n".join(L)


def render_redesign_report(r: Dict[str, Any], generated_at: Optional[str] = None) -> str:
    L = ["# T 트랙 재설계 후보 검증", "",
         f"- 생성: {generated_at or '—'} · 기존 ETF {r['n_tickers']['기존 ETF']}개 · HOLDOUT(해외 국가 ETF, 이전 백테스트에서 쓰지 않음) {r['n_tickers']['HOLDOUT']}개 · "
         f"{TBT_CONFIG['REDESIGN_START'][:4]}년~ · 왕복 비용 {_cost() * 100:.2f}%",
         "- 후보와 매개변수는 관행값 그대로이고(데이터로 맞추지 않음), 판정 기준은 결과를 보기 전에 고정했습니다.",
         f"- 판정: HOLDOUT에서 평균 R > 0 이고 (후보 − 무작위) ≥ 표준오차 × {r['criteria']['REDESIGN_MIN_Z']:.0f}, 그리고 기존 ETF 전체 기간 평균 R > 0 이면 통과.",
         f"- 무작위 비교: 같은 종목·기간의 무작위 날짜(실제 거래 수 × {TBT_CONFIG['BASELINE_SAMPLES']})에 같은 청산 규칙 적용.", ""]
    for name, desc in r["candidates"].items():
        v = r["verdict"][name]
        L += [f"## {name}: {'통과' if v['passed'] else '미통과'}", "", f"- 규칙: {desc}",
              f"- HOLDOUT 기준 {'통과' if v['holdout_pass'] else '미통과'} · 기존 ETF 평균 R {'> 0' if v['main_positive'] else '≤ 0'}", "",
              "| 묶음 | 기간 | 후보 평균 R ±표준오차 (승률·건수) | 무작위 평균 R ±표준오차 (승률·건수) | 차이 | z | 거래당 수익률 | 평균 보유일 |",
              "|---|---|---|---|---|---|---|---|"]
        for uname, blk in r["results"][name].items():
            for per, x in blk.items():
                ret = "—" if not x["ret"].get("n") else f"{x['ret']['mean'] * 100:+.2f}%"
                hd = "—" if x.get("hold_days") is None else f"{x['hold_days']:.1f}"
                L.append(f"| {uname} | {per} | {_cmp_cells(x)} | {ret} | {hd} |")
        L.append("")
    L += ["## 주의",
          "- 진입은 신호 다음 날 시가, 청산은 그날 종가로 계산했습니다(원래 규칙의 '당일 종가 진입'보다 보수적).",
          "- 통과한 후보만 레이더 셋업으로 옮깁니다. 통과해도 앞으로의 실제 결과로 다시 확인해야 합니다.", ""]
    return "\n".join(L)
