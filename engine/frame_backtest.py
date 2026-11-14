"""매매 틀 검증 — 틀 A(추세반등)를 개별 종목으로 검증 (2026-11-14 추가).

틀 A의 매수 논리: "상승 추세가 살아 있고, 지금 하락은 일시적 눌림이다."
이 논리 하나를 진입 → 무효화(손절) → 매도까지 한 묶음으로 고정했다(AGENTS.md 4번 '두 가지 틀').
    선정  6개월·12개월 수익률이 모두 시장(미국 SPY, 한국 KOSPI)보다 높고, 종가 > 200일선
    진입  선정 조건 + RSI(2) < 10 (눌림) → 다음 날 시가. 시가가 이미 무효화 가격 아래면 진입하지 않음
    무효화 손절  종가 < 직전 저점(신호일 포함 60일 최저가) 또는 종가 < 200일선. 상승 추세는 저점이 높아지는 구조이므로
                 약 3개월 저점을 깨면 "일시적 눌림"이 아니라 추세 전환으로 본다
    매도  반등 확인(종가 > 50일선) 뒤 다시 종가 < 50일선 (하락 추세 전환). 보유 기간 제한 없음
    청산 실행  조건이 나온 날 종가로 확인하고 다음 날 시가에 판다 (갭 하락 손실이 그대로 반영됨)
매개변수는 모두 관행값이고 데이터로 맞추지 않았다. 판정 기준(FRAME_A_CRITERIA)도 결과를 보기 전에 고정했다.
'이익 추정치가 꺾이지 않음' 조건은 과거 시점 추정치 기록이 없어 여기서 검증하지 못한다(engine/snapshot.py가 앞으로 쌓는다).

비교 대상
    보유 기간 맞춘 무작위  실제 거래마다 같은 종목의 무작위 날짜에 사서 그 거래와 같은 날 수만큼 들고 판다 (판정 기준)
                          주가는 평균적으로 오르므로 오래 들수록 거래당 수익이 커진다. 보유 기간을 맞춰야
                          '언제 사고 언제 파는가'의 실력만 비교된다(같은 청산 규칙을 무작위 날짜에 쓰면 200일선 아래
                          날짜가 곧바로 청산돼 보유가 짧아지고, 우위가 없는 가짜 가격에서도 통과로 나왔다).
    선정만                선정 조건을 만족한 날 중 무작위 + 같은 무효화·매도 규칙 → 눌림 타이밍이 보태는 것이 있는가 (참고)
표준오차는 같은 달에 나온 신호를 한 덩어리로 묶어 계산한다(같은 날 여러 종목이 함께 눌리므로 거래끼리 독립이 아님).
"""
import math
import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .common import TECH_PARAMS
from .rule_backtest import BACKTEST_CONFIG, _bt_naive

FRAME_A_CONFIG: Dict[str, Any] = {
    "START": "2001-01-01",          # 신호 시작일 (200일선·12개월 상대강도 준비 기간 뒤)
    "DATA_START": "1999-01-01",
    "SPLIT_DATE": BACKTEST_CONFIG["SPLIT_DATE"],   # 앞 기간 / 뒤 기간
    "RS_DAYS": [126, 252],          # 6개월·12개월 (거래일)
    "TREND_MA": 200,
    "EXIT_MA": 50,
    "RSI_MAX": 10.0,                # RSI(2, 와일더) < 10 = 눌림
    "LOW_LOOKBACK": 60,             # 직전 저점 = 신호일 포함 60일(약 3개월) 최저가. 20일은 눌림 당일 저가와 겹쳐 손절선이 진입가에 붙음
    "SLIP_WARN": -0.05,             # 무효화 가격보다 5% 넘게 더 밀려 팔린 거래를 따로 센다
    "GAP_DOWN": -0.10,              # 큰 갭 하락 = 시가가 전날 종가보다 10% 이상 낮음
    "BENCH": {"US": "SPY", "KR": "^KS11"},
    "BASELINE_SAMPLES": 20,         # 실제 거래 1건당 무작위 비교 표본 수
    "SEED": 11,
    "COST_SCENARIOS": [0.0, 0.003, 0.006],   # 왕복 비용 민감도 (0.30% = 현재 TP-32)
    # 개발 묶음: 앱이 쓰는 대표 종목(이전 T 셋업 진단에 쓰였지만 이 틀로는 처음)
    "DEV_UNIVERSE": list(BACKTEST_CONFIG["SETUP_UNIVERSE"]["대표 종목"]),
    # 판정 묶음(HOLDOUT): 어떤 백테스트에도 쓰지 않은 개별 종목. 업종을 고르게, 오래 부진했던 종목도 일부러 넣음
    "HOLDOUT_UNIVERSE": [
        "ABT", "ACN", "ADBE", "AIG", "AXP", "BA", "BAC", "BK", "BKNG", "BLK", "BMY", "C", "CL", "CMCSA", "COF",
        "COP", "CSCO", "CVS", "DE", "DHR", "DIS", "EMR", "F", "FDX", "GD", "GM", "IBM", "INTC", "INTU", "ISRG",
        "KO", "LOW", "MA", "MCD", "MDLZ", "MDT", "MET", "MMM", "MO", "MRK", "MS", "NKE", "ORCL", "PEP", "PFE",
        "PM", "QCOM", "SBUX", "SCHW", "SPG", "T", "TGT", "TMO", "TXN", "UPS", "USB", "VZ", "WFC",
        "000270.KS", "005490.KS", "006400.KS", "012330.KS", "015760.KS", "017670.KS", "028260.KS", "033780.KS",
        "035720.KS", "055550.KS", "066570.KS", "068270.KS", "096770.KS", "105560.KS", "009150.KS", "010130.KS",
        "036570.KS", "003550.KS", "034730.KS", "032830.KS", "010950.KS", "000810.KS", "086790.KS", "030200.KS",
        "011200.KS",
    ],
}
FRAME_A_CRITERIA: Dict[str, Any] = {
    # HOLDOUT 전체 기간: 거래당 순수익률 평균 > 0 이고 (실제 − 보유 기간 맞춘 무작위) ≥ 월 묶음 표준오차 × 2
    # 개발 묶음: 앞 기간·뒤 기간 모두 거래당 순수익률 평균 > 0
    "MIN_Z": 2.0,
}
FRAME_A_RULES = ("선정: 6·12개월 수익률 모두 시장보다 높음 + 종가 > 200일선 · 진입: + RSI(2) < 10 → 다음 날 시가 · "
                 "무효화 손절: 종가 < 직전 저점(60일 최저가) 또는 종가 < 200일선 · 매도: 종가 > 50일선 확인 뒤 종가 < 50일선 · "
                 "청산은 다음 날 시가")
EXIT_LABELS = {"trend": "추세 전환(50일선 이탈)", "inv_low": "무효화: 직전 저점 이탈", "inv_ma": "무효화: 200일선 이탈",
               "open": "미청산(마지막 종가로 평가)"}


def _fa_cost() -> float:
    return float(TECH_PARAMS["ROUND_TRIP_COST"])


def _fa_market(sym: str) -> str:
    return "KR" if str(sym).upper().endswith((".KS", ".KQ")) else "US"


def _fa_next_true(mask: np.ndarray) -> np.ndarray:
    """k마다 'k 이상에서 처음 mask가 참인 위치' (없으면 n). n+1 길이로 끝에 n을 붙여 k=n 조회도 안전."""
    n = len(mask)
    idx = np.where(mask, np.arange(n), n)
    out = np.minimum.accumulate(idx[::-1])[::-1] if n else idx
    return np.append(out, n)


def frame_a_indicators(df: pd.DataFrame, bench: Optional[pd.DataFrame]) -> pd.DataFrame:
    """그날 종가까지의 데이터만 쓰는 지표. bench 종가는 종목 날짜에 맞춰 앞 값으로 채운다(그날 이후 값은 쓰지 않음)."""
    cfg = FRAME_A_CONFIG
    d = _bt_naive(df)
    c, lo = d["Close"].astype(float), d["Low"].astype(float)
    out = pd.DataFrame(index=d.index)
    out["Open"], out["Low"], out["Close"] = d["Open"].astype(float), lo, c
    out["SMA200"] = c.rolling(cfg["TREND_MA"], min_periods=cfg["TREND_MA"]).mean()
    out["SMA50"] = c.rolling(cfg["EXIT_MA"], min_periods=cfg["EXIT_MA"]).mean()
    delta = c.diff()
    up, dn = delta.clip(lower=0.0), (-delta).clip(lower=0.0)
    au = up.ewm(alpha=1 / 2, adjust=False, min_periods=2).mean()
    ad = dn.ewm(alpha=1 / 2, adjust=False, min_periods=2).mean()
    out["RSI2"] = (100.0 - 100.0 / (1.0 + au / ad.replace(0.0, np.nan))).where(ad > 0, 100.0)
    out["LOWREF"] = lo.rolling(cfg["LOW_LOOKBACK"], min_periods=cfg["LOW_LOOKBACK"]).min()
    b = _bt_naive(bench) if bench is not None else None
    bc = b["Close"].astype(float).reindex(d.index.union(b.index)).ffill().reindex(d.index) if b is not None and len(b) else None
    rs_ok = pd.Series(True, index=d.index)
    for n in cfg["RS_DAYS"]:
        rs = c / c.shift(n) - 1.0
        if bc is not None:
            rs = rs - (bc / bc.shift(n) - 1.0)
        out[f"RS{n}"] = rs
        rs_ok &= rs > 0
    out["selected"] = rs_ok & (c > out["SMA200"])
    out["signal"] = out["selected"] & (out["RSI2"] < cfg["RSI_MAX"])
    return out


def _fa_arrays(ind: pd.DataFrame) -> Dict[str, np.ndarray]:
    a = {k: ind[k].to_numpy(dtype=float) for k in ("Open", "Close", "SMA200", "SMA50", "LOWREF")}
    c = a["Close"]
    with np.errstate(invalid="ignore"):
        a["nb200"] = _fa_next_true(c < a["SMA200"])
        a["na50"] = _fa_next_true(c > a["SMA50"])
        a["nb50"] = _fa_next_true(c < a["SMA50"])
        prev = np.concatenate([[np.nan], c[:-1]])
        gap = (a["Open"] / prev - 1.0) <= FRAME_A_CONFIG["GAP_DOWN"]
    a["cumgap"] = np.cumsum(gap.astype(int))
    return a


def simulate_frame_a_trade(a: Dict[str, np.ndarray], i: int, cost: Optional[float] = None) -> Optional[Dict[str, Any]]:
    """i일 종가 신호 → i+1일 시가 진입 → 무효화·추세 전환을 종가로 확인 → 다음 날 시가 청산.
    데이터 끝까지 청산 조건이 없으면 마지막 종가로 평가(미청산). 진입 시가가 이미 무효화 가격 이하이면 skipped."""
    cst = _fa_cost() if cost is None else float(cost)
    c, o = a["Close"], a["Open"]
    n = len(c)
    if i + 1 >= n:
        return None
    fill, low_ref = float(o[i + 1]), float(a["LOWREF"][i])
    if not (np.isfinite(fill) and np.isfinite(low_ref) and fill > 0):
        return None
    if fill <= low_ref:
        return {"skipped": True}
    s = i + 1
    inv_ma = int(a["nb200"][s])
    arm = int(a["na50"][s])
    trend = int(a["nb50"][arm + 1]) if arm + 1 <= n else n
    bound = min(inv_ma, trend, n - 1)
    seg = c[s:bound + 1] < low_ref
    inv_low = s + int(np.argmax(seg)) if seg.any() else n
    trig = min(inv_ma, inv_low, trend)
    if trig >= n - 1:                      # 조건 없음, 또는 마지막 날 조건(다음 날 시가를 모름)
        kind, exit_idx, exit_px, exit_gap = "open", n - 1, float(c[n - 1]), None
    else:
        kind = "inv_low" if trig == inv_low else ("inv_ma" if trig == inv_ma else "trend")
        exit_idx, exit_px = trig + 1, float(o[trig + 1])
        exit_gap = float(exit_px / c[trig] - 1.0)
    gross = exit_px / fill - 1.0
    unit = fill - low_ref
    return {"fill": fill, "low_ref": low_ref, "exit_px": exit_px, "exit_kind": kind, "exit_idx": int(exit_idx),
            "hold_days": int(exit_idx - s), "gross": float(gross), "ret": float(gross - cst),
            "r": float((exit_px - fill - fill * cst) / unit), "exit_gap": exit_gap,
            "gap_days": int(a["cumgap"][exit_idx] - a["cumgap"][s]), "armed": bool(arm < trig)}


def frame_a_trades(ind: pd.DataFrame, start: Optional[str] = None, cost: Optional[float] = None,
                   arrs: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, Any]:
    """한 종목의 틀 A 거래 (종목당 1포지션: 청산 실행일 종가부터 새 신호를 받음)."""
    a = arrs or _fa_arrays(ind)
    s0 = pd.Timestamp(start or FRAME_A_CONFIG["START"])
    sig = ind["signal"].to_numpy(dtype=bool)
    out, skipped, free_from = [], 0, -1
    for i in np.flatnonzero(sig):
        if ind.index[i] < s0 or i < free_from:
            continue
        res = simulate_frame_a_trade(a, int(i), cost)
        if res is None:
            continue
        if res.get("skipped"):
            skipped += 1
            continue
        out.append({"date": ind.index[i], **res})
        if res["exit_kind"] == "open":
            break
        free_from = res["exit_idx"]
    return {"trades": pd.DataFrame(out), "skipped": skipped}


def frame_a_baseline(ind: pd.DataFrame, n_trades: int, pool: str = "random", start: Optional[str] = None,
                     samples: Optional[int] = None, seed: Optional[int] = None, cost: Optional[float] = None,
                     arrs: Optional[Dict[str, np.ndarray]] = None) -> pd.DataFrame:
    """비교 표본: pool='random' 지표가 준비된 모든 날, pool='selected' 선정 조건을 만족한 날. 같은 청산 규칙."""
    a = arrs or _fa_arrays(ind)
    s0 = pd.Timestamp(start or FRAME_A_CONFIG["START"])
    ok = (ind.index >= s0) & np.isfinite(a["SMA200"]) & np.isfinite(a["SMA50"]) & np.isfinite(a["LOWREF"])
    if pool == "selected":
        ok &= ind["selected"].to_numpy(dtype=bool)
    valid = np.flatnonzero(ok)
    valid = valid[valid < len(ind) - 2]
    if len(valid) == 0 or n_trades == 0:
        return pd.DataFrame()
    rng = np.random.default_rng((FRAME_A_CONFIG["SEED"] if seed is None else seed) + (1 if pool == "selected" else 0))
    out = []
    for j in rng.choice(valid, size=n_trades * int(samples or FRAME_A_CONFIG["BASELINE_SAMPLES"]), replace=True):
        res = simulate_frame_a_trade(a, int(j), cost)
        if res and not res.get("skipped"):
            out.append({"date": ind.index[j], **res})
    return pd.DataFrame(out)


def frame_a_matched_baseline(ind: pd.DataFrame, trades: pd.DataFrame, start: Optional[str] = None,
                             samples: Optional[int] = None, seed: Optional[int] = None, cost: Optional[float] = None,
                             arrs: Optional[Dict[str, np.ndarray]] = None, pool: Optional[np.ndarray] = None) -> pd.DataFrame:
    """보유 기간 맞춘 무작위: 실제 거래마다 무작위 날짜 j 종가 뒤 j+1일 시가에 사서, 그 거래의 보유일 h만큼 지나
    j+1+h일 시가에 판다(실제 거래와 같은 실행 방식). 같은 종목·같은 시작일 이후에서만 뽑는다.
    pool(참·거짓 배열)을 주면 그 날들 중에서만 뽑는다(예: 틀 B '싼 날만')."""
    a = arrs or _fa_arrays(ind)
    if trades is None or not len(trades):
        return pd.DataFrame()
    o = a["Open"]
    n = len(o)
    s0 = pd.Timestamp(start or FRAME_A_CONFIG["START"])
    first = int(np.searchsorted(ind.index.values, np.datetime64(s0)))
    cand = np.arange(n) if pool is None else np.flatnonzero(np.asarray(pool, dtype=bool))
    cand = cand[cand >= first]
    rng = np.random.default_rng((FRAME_A_CONFIG["SEED"] if seed is None else seed) + 2)
    k = int(samples or FRAME_A_CONFIG["BASELINE_SAMPLES"])
    cst = _fa_cost() if cost is None else float(cost)
    out = []
    for h in trades["hold_days"].astype(int).to_numpy():
        last = n - 2 - h                       # j+1+h ≤ n−1
        if pool is None:                       # 틀 A 결과(FRAME_A.md)와 같은 난수 순서를 유지
            if last < first:
                continue
            draws = rng.integers(first, last + 1, size=k)
        else:
            ok = cand[cand <= last]
            if not len(ok):
                continue
            draws = rng.choice(ok, size=k, replace=True)
        for j in draws:
            fill, ex = o[j + 1], o[j + 1 + h]
            if not (np.isfinite(fill) and np.isfinite(ex) and fill > 0):
                continue
            g = ex / fill - 1.0
            out.append({"date": ind.index[j], "fill": float(fill), "exit_px": float(ex), "gross": float(g),
                        "ret": float(g - cst), "hold_days": int(h), "exit_kind": "hold", "r": np.nan,
                        "gap_days": int(a["cumgap"][j + 1 + h] - a["cumgap"][j + 1])})
    return pd.DataFrame(out)


# ---------------------------------------------------------------------
# 통계
# ---------------------------------------------------------------------
def cluster_stats(x: pd.Series, dates: pd.Series) -> Dict[str, Any]:
    """평균·승률과 두 가지 표준오차: 거래 독립 가정(se_iid), 신호 달 묶음(se, 판정에 사용)."""
    x = pd.Series(x, dtype=float).reset_index(drop=True)
    g = pd.to_datetime(pd.Series(dates)).reset_index(drop=True).dt.to_period("M")
    m = x.notna()
    x, g = x[m], g[m]
    n = len(x)
    if n == 0:
        return {"n": 0}
    mean = float(x.mean())
    se_iid = float(x.std(ddof=1) / math.sqrt(n)) if n >= 2 else None
    sums = (x - mean).groupby(g).sum()
    k = len(sums)
    se = float(math.sqrt(k / (k - 1) * float((sums ** 2).sum())) / n) if k >= 2 else None
    return {"n": n, "mean": mean, "median": float(x.median()), "win_rate": float((x > 0).mean() * 100.0),
            "se": se, "se_iid": se_iid, "n_months": int(k)}


def compare_clustered(a: pd.DataFrame, b: pd.DataFrame, col: str = "ret") -> Dict[str, Any]:
    sa = cluster_stats(a[col], a["date"]) if len(a) else {"n": 0}
    sb = cluster_stats(b[col], b["date"]) if len(b) else {"n": 0}
    diff = z = z_iid = None
    if sa.get("n", 0) >= 2 and sb.get("n", 0) >= 2:
        diff = sa["mean"] - sb["mean"]
        if sa.get("se") is not None and sb.get("se") is not None:
            se = math.sqrt(sa["se"] ** 2 + sb["se"] ** 2)
            z = diff / se if se > 0 else None
        se_i = math.sqrt(sa["se_iid"] ** 2 + sb["se_iid"] ** 2)
        z_iid = diff / se_i if se_i > 0 else None
    return {"actual": sa, "baseline": sb, "diff": diff, "z": z, "z_iid": z_iid}


def tail_stats(t: pd.DataFrame) -> Dict[str, Any]:
    """손실 꼬리와 갭 하락: 무효화로 판 거래가 무효화 가격보다 얼마나 더 밀렸나(slip), 청산 시가 갭, 보유 중 −10% 이상 갭 하락."""
    if not len(t):
        return {"n": 0}
    closed = t[t["exit_kind"] != "open"]
    eg = closed["exit_gap"].dropna()
    inv = t[t["exit_kind"].isin(["inv_low", "inv_ma"])]
    slip = (inv["exit_px"] / inv["low_ref"] - 1.0).where(inv["exit_kind"] == "inv_low") if len(inv) else pd.Series(dtype=float)
    slip = slip.dropna()
    return {"n": int(len(t)),
            "worst_ret": float(t["ret"].min()), "p5_ret": float(t["ret"].quantile(0.05)),
            "n_inv_low": int(len(slip)),
            "share_slip_warn": float((slip <= FRAME_A_CONFIG["SLIP_WARN"]).mean() * 100.0) if len(slip) else 0.0,
            "worst_slip": float(slip.min()) if len(slip) else None,
            "share_exit_gap10": float((eg <= FRAME_A_CONFIG["GAP_DOWN"]).mean() * 100.0) if len(eg) else 0.0,
            "worst_exit_gap": float(eg.min()) if len(eg) else None,
            "share_with_gap_day": float((t["gap_days"] > 0).mean() * 100.0),
            "ret_with_gap_day": float(t.loc[t["gap_days"] > 0, "ret"].mean()) if (t["gap_days"] > 0).any() else None,
            "inv_dist_median": float((t["fill"] / t["low_ref"] - 1.0).median())}


def _fa_block(tr: pd.DataFrame, rnd: pd.DataFrame, sel: pd.DataFrame) -> Dict[str, Any]:
    out = {"vs_random": compare_clustered(tr, rnd), "vs_selected": compare_clustered(tr, sel),
           "r": cluster_stats(tr["r"], tr["date"]) if len(tr) else {"n": 0},
           "hold_days": float(tr["hold_days"].mean()) if len(tr) else None,
           "hold_days_median": float(tr["hold_days"].median()) if len(tr) else None,
           "tail": tail_stats(tr), "tail_selected": tail_stats(sel), "exits": {}, "cost": {},
           "hold_days_random": float(rnd["hold_days"].mean()) if len(rnd) else None}
    if len(tr):
        for k, g in tr.groupby("exit_kind"):
            out["exits"][EXIT_LABELS.get(k, k)] = {"share": float(len(g) / len(tr) * 100.0), "mean_ret": float(g["ret"].mean()),
                                                   "hold_days": float(g["hold_days"].mean())}
        for cst in FRAME_A_CONFIG["COST_SCENARIOS"]:
            a, b = tr.assign(ret=tr["gross"] - cst), rnd.assign(ret=rnd["gross"] - cst) if len(rnd) else rnd
            cmp_ = compare_clustered(a, b)
            out["cost"][f"{cst * 100:.2f}%"] = {"mean": cmp_["actual"].get("mean"), "diff": cmp_["diff"], "z": cmp_["z"]}
    return out


def _fa_vs_hold(ind: pd.DataFrame, tr: pd.DataFrame, start: pd.Timestamp) -> Optional[Dict[str, Any]]:
    """같은 기간 매수 후 보유와 비교(참고): 연 환산 수익률과 보유 비중."""
    d = ind[ind.index >= start]
    if len(d) < 252:
        return None
    years = (d.index[-1] - d.index[0]).days / 365.25
    bh = float((d["Close"].iloc[-1] / d["Close"].iloc[0]) ** (1 / years) - 1.0)
    eq = float(np.prod(1.0 + tr["ret"].to_numpy())) if len(tr) else 1.0
    st = float(eq ** (1 / years) - 1.0) if eq > 0 else -1.0
    expo = float(tr["hold_days"].sum() / len(d)) if len(tr) else 0.0
    return {"years": years, "bh_cagr": bh, "frame_cagr": st, "exposure": expo, "n_trades": int(len(tr))}


def run_frame_a_universe(price_map: Dict[str, pd.DataFrame], bench_map: Dict[str, pd.DataFrame],
                         cost: Optional[float] = None) -> Dict[str, Any]:
    """묶음 하나의 실제 거래·무작위·선정만 표본과 종목별 보유 비교."""
    cfg = FRAME_A_CONFIG
    s0 = pd.Timestamp(cfg["START"])
    trs, rnds, sels, holds, skipped, used = [], [], [], {}, 0, []
    for sym, df in (price_map or {}).items():
        d = _bt_naive(df)
        if d is None or len(d) < cfg["TREND_MA"] + 60:
            continue
        mk = _fa_market(sym)
        ind = frame_a_indicators(d, (bench_map or {}).get(cfg["BENCH"][mk]))
        a = _fa_arrays(ind)
        r = frame_a_trades(ind, cost=cost, arrs=a)
        tr = r["trades"]
        skipped += r["skipped"]
        used.append(sym)
        rnd = frame_a_matched_baseline(ind, tr, cost=cost, arrs=a)
        sel = frame_a_baseline(ind, len(tr), "selected", cost=cost, arrs=a)
        for lst, x in ((trs, tr), (rnds, rnd), (sels, sel)):
            if len(x):
                lst.append(x.assign(ticker=sym, market=mk))
        h = _fa_vs_hold(ind, tr, s0)
        if h:
            holds[sym] = h
    cols = ["date", "ticker", "market", "fill", "low_ref", "exit_px", "ret", "gross", "r", "exit_kind", "hold_days", "exit_gap", "gap_days"]
    cat = lambda lst: pd.concat(lst, ignore_index=True) if lst else pd.DataFrame(columns=cols)
    return {"trades": cat(trs), "random": cat(rnds), "selected": cat(sels), "holds": holds,
            "skipped": skipped, "tickers": used}


def evaluate_frame_a(dev_map: Dict[str, pd.DataFrame], holdout_map: Dict[str, pd.DataFrame],
                     bench_map: Dict[str, pd.DataFrame], cost: Optional[float] = None) -> Dict[str, Any]:
    """개발 묶음(앞·뒤 기간)과 HOLDOUT에서 틀 A를 평가하고 사전 기준으로 판정."""
    split_t = pd.Timestamp(FRAME_A_CONFIG["SPLIT_DATE"])
    res: Dict[str, Any] = {"rules": FRAME_A_RULES, "criteria": FRAME_A_CRITERIA, "config": {
        k: FRAME_A_CONFIG[k] for k in ("START", "SPLIT_DATE", "RS_DAYS", "TREND_MA", "EXIT_MA", "RSI_MAX", "LOW_LOOKBACK",
                                       "GAP_DOWN", "BASELINE_SAMPLES", "COST_SCENARIOS")},
        "cost": _fa_cost() if cost is None else float(cost), "universes": {}, "verdict": {}}
    raw = {"개발(대표 종목)": run_frame_a_universe(dev_map, bench_map, cost),
           "HOLDOUT(새 종목)": run_frame_a_universe(holdout_map, bench_map, cost)}
    for uname, u in raw.items():
        tr, rnd, sel = u["trades"], u["random"], u["selected"]
        per = {"전체 기간": lambda x: x, "앞 기간": lambda x: x[x["date"] < split_t], "뒤 기간": lambda x: x[x["date"] >= split_t]}
        blocks = {p: _fa_block(f(tr), f(rnd), f(sel)) for p, f in per.items()}
        by_mk = {mk: _fa_block(tr[tr["market"] == mk], rnd[rnd["market"] == mk], sel[sel["market"] == mk])
                 for mk in ("US", "KR") if len(tr) and (tr["market"] == mk).any()}
        hv = pd.DataFrame(u["holds"]).T
        hold_summary = ({"n": int(len(hv)), "bh_cagr_median": float(hv["bh_cagr"].median()),
                         "frame_cagr_median": float(hv["frame_cagr"].median()),
                         "exposure_median": float(hv["exposure"].median()),
                         "share_frame_beats_bh": float((hv["frame_cagr"] > hv["bh_cagr"]).mean() * 100.0)} if len(hv) else {"n": 0})
        years = sum(h["years"] for h in u["holds"].values())
        res["universes"][uname] = {"n_tickers": len(u["tickers"]), "skipped_invalid_at_open": u["skipped"],
                                   "trades_per_ticker_year": float(len(tr) / years) if years else None,
                                   "blocks": blocks, "by_market": by_mk, "vs_hold": hold_summary}
    ho = res["universes"]["HOLDOUT(새 종목)"]["blocks"]["전체 기간"]["vs_random"]
    dv = res["universes"]["개발(대표 종목)"]["blocks"]
    ok_ho = bool(ho["actual"].get("n", 0) >= 2 and ho["actual"]["mean"] > 0 and ho["z"] is not None
                 and ho["z"] >= FRAME_A_CRITERIA["MIN_Z"])
    ok_dev = all(dv[p]["vs_random"]["actual"].get("n", 0) >= 2 and dv[p]["vs_random"]["actual"]["mean"] > 0
                 for p in ("앞 기간", "뒤 기간"))
    sel = res["universes"]["HOLDOUT(새 종목)"]["blocks"]["전체 기간"]["vs_selected"]
    res["verdict"] = {"passed": ok_ho and ok_dev, "holdout_pass": ok_ho, "dev_both_positive": ok_dev,
                      "timing_adds": bool(sel.get("z") is not None and sel["z"] >= FRAME_A_CRITERIA["MIN_Z"]),
                      "criteria": (f"HOLDOUT 전체 기간 거래당 순수익률 > 0 이고 (실제 − 보유 기간 맞춘 무작위) ≥ 월 묶음 표준오차 × "
                                   f"{FRAME_A_CRITERIA['MIN_Z']:.0f}, 그리고 개발 묶음 앞·뒤 기간 모두 거래당 순수익률 > 0")}
    trades = pd.concat([u["trades"].assign(universe=k) for k, u in raw.items() if len(u["trades"])], ignore_index=True) \
        if any(len(u["trades"]) for u in raw.values()) else pd.DataFrame()
    return {"summary": res, "trades": trades}


# ---------------------------------------------------------------------
# 보고서
# ---------------------------------------------------------------------
def _fa_pct(v: Optional[float], d: int = 2) -> str:
    return "—" if v is None or (isinstance(v, float) and not math.isfinite(v)) else f"{v * 100:+.{d}f}%"


def _fa_one(s: Dict[str, Any]) -> str:
    if not s.get("n"):
        return "표본 없음"
    se = f" ±{s['se'] * 100:.2f}" if s.get("se") is not None else ""
    return f"{s['mean'] * 100:+.2f}%{se} ({s['win_rate']:.0f}% · {s['n']:,}건)"


def _fa_cmp_row(x: Dict[str, Any]) -> str:
    z = "—" if x.get("z") is None else f"{x['z']:+.1f}"
    zi = "—" if x.get("z_iid") is None else f"{x['z_iid']:+.1f}"
    return f"{_fa_one(x['actual'])} | {_fa_one(x['baseline'])} | {_fa_pct(x.get('diff'))} | {z} ({zi})"


def render_frame_a_report(r: Dict[str, Any], generated_at: Optional[str] = None) -> str:
    v = r["verdict"]
    cfg = r["config"]
    L = ["# 틀 A(추세반등) 개별 종목 검증", "",
         f"- 생성: {generated_at or '—'} · 신호 {cfg['START'][:4]}년~ · 왕복 비용 {r['cost'] * 100:.2f}% 차감",
         f"- 규칙(결과 보기 전 고정, 관행값): {r['rules']}",
         "- 개발 묶음 = 앱 대표 종목, HOLDOUT = 어떤 백테스트에도 쓰지 않은 미국·한국 개별 종목.",
         f"- 무작위 비교(판정 기준): 실제 거래마다 같은 종목의 무작위 날짜 {cfg['BASELINE_SAMPLES']}곳에서 사서 그 거래와 같은 날 수만큼 들고 판 결과. "
         "주가는 평균적으로 오르므로 보유 기간을 맞춰야 사고파는 시점의 실력만 비교됩니다.",
         "- '선정만'(참고): 선정 조건을 만족한 날 중 무작위로 사서 같은 무효화·매도 규칙을 적용 — 눌림을 기다리는 것이 보태는 것이 있는지 봅니다.",
         "- 표준오차는 같은 달 신호를 한 덩어리로 묶어 계산했습니다. 괄호 안 z는 거래를 서로 독립으로 본 값(참고).",
         f"- 판정 기준: {v['criteria']}.", "",
         f"## 판정: {'통과' if v['passed'] else '미통과'}", "",
         f"- HOLDOUT 기준 {'통과' if v['holdout_pass'] else '미통과'} · 개발 묶음 앞·뒤 기간 {'모두 > 0' if v['dev_both_positive'] else '하나 이상 ≤ 0'} · "
         f"눌림 타이밍의 기여(선정만 대비 z ≥ 2) {'확인' if v['timing_adds'] else '확인 안 됨'}", ""]
    for uname, u in r["universes"].items():
        L += [f"## {uname}", "",
              f"종목 {u['n_tickers']}개 · 진입 시가가 이미 무효화 가격 아래라 건너뜀 {u['skipped_invalid_at_open']}건 · "
              f"종목당 연 {u['trades_per_ticker_year'] or 0:.1f}건", "",
              "| 기간 | 틀 A 거래당 순수익률 ±표준오차 (승률·건수) | 보유 기간 맞춘 무작위 | 차이 | z (독립 가정) | 평균·중앙 보유일 |",
              "|---|---|---|---|---|---|"]
        for per, b in u["blocks"].items():
            hd = "—" if b["hold_days"] is None else f"{b['hold_days']:.0f} · {b['hold_days_median']:.0f}"
            L.append(f"| {per} | {_fa_cmp_row(b['vs_random'])} | {hd} |")
        for mk, b in u["by_market"].items():
            L.append(f"| 전체 기간 · {'미국' if mk == 'US' else '한국'} | {_fa_cmp_row(b['vs_random'])} | "
                     f"{'—' if b['hold_days'] is None else format(b['hold_days'], '.0f')} |")
        full = u["blocks"]["전체 기간"]
        L += ["", "| 비교 | 틀 A | 선정만(눌림 없이, 같은 규칙) | 차이 | z (독립 가정) |", "|---|---|---|---|---|",
              f"| 전체 기간 | {_fa_cmp_row(full['vs_selected'])} |", ""]
        if full["exits"]:
            L += ["| 청산 이유 | 비중 | 거래당 순수익률 | 평균 보유일 |", "|---|---|---|---|"]
            for k, e in sorted(full["exits"].items(), key=lambda kv: -kv[1]["share"]):
                L.append(f"| {k} | {e['share']:.0f}% | {_fa_pct(e['mean_ret'])} | {e['hold_days']:.0f} |")
            L.append("")
        t, tr_ = full["tail"], full["tail_selected"]
        if t.get("n"):
            L += ["| 손실 꼬리·갭 하락 | 틀 A | 선정만(같은 규칙) |", "|---|---|---|",
                  f"| 진입가와 무효화 가격 거리(중앙값) | {t['inv_dist_median'] * 100:.1f}% | {tr_.get('inv_dist_median', 0) * 100:.1f}% |",
                  f"| 직전 저점 이탈로 판 거래 중 무효화 가격보다 5% 넘게 더 밀려 팔림 | {t['share_slip_warn']:.1f}% ({t['n_inv_low']:,}건 중) | "
                  f"{tr_.get('share_slip_warn', 0):.1f}% |",
                  f"| 무효화 가격 대비 가장 크게 밀린 청산 | {_fa_pct(t['worst_slip'], 1)} | {_fa_pct(tr_.get('worst_slip'), 1)} |",
                  f"| 하위 5% 거래 수익률 | {_fa_pct(t['p5_ret'])} | {_fa_pct(tr_.get('p5_ret'))} |",
                  f"| 가장 나쁜 거래 | {_fa_pct(t['worst_ret'])} | {_fa_pct(tr_.get('worst_ret'))} |",
                  f"| 청산 시가가 전날 종가보다 10% 이상 낮았던 비율 | {t['share_exit_gap10']:.1f}% | {tr_.get('share_exit_gap10', 0):.1f}% |",
                  f"| 가장 큰 청산 갭 | {_fa_pct(t['worst_exit_gap'], 1)} | {_fa_pct(tr_.get('worst_exit_gap'), 1)} |",
                  f"| 보유 중 −10% 이상 갭 하락을 겪은 거래 | {t['share_with_gap_day']:.1f}% (평균 {_fa_pct(t['ret_with_gap_day'])}) | "
                  f"{tr_.get('share_with_gap_day', 0):.1f}% |", ""]
        if full["cost"]:
            L += ["| 왕복 비용 | 틀 A 거래당 | 보유 기간 맞춘 무작위 대비 차이 | z |", "|---|---|---|---|"]
            for k, c in full["cost"].items():
                L.append(f"| {k} | {_fa_pct(c['mean'])} | {_fa_pct(c['diff'])} | {'—' if c['z'] is None else format(c['z'], '+.1f')} |")
            L.append("")
        h = u["vs_hold"]
        if h.get("n"):
            L += [f"매수 후 보유와 비교(참고, 종목별 연 환산 중앙값): 틀 A {_fa_pct(h['frame_cagr_median'], 1)} · 보유 {_fa_pct(h['bh_cagr_median'], 1)} · "
                  f"틀 A가 보유보다 나은 종목 {h['share_frame_beats_bh']:.0f}% · 주식을 들고 있던 기간 비중 {h['exposure_median'] * 100:.0f}%", ""]
    L += ["## 주의",
          "- 두 묶음 모두 지금 상장된 종목이라(상장 폐지 종목 없음) 살아남은 종목 편향이 있습니다. 무작위 비교도 같은 종목을 쓰므로 "
          "차이에는 영향이 작지만, 수익률 수준 자체는 실제보다 좋게 나올 수 있습니다.",
          "- '이익 추정치가 꺾이지 않음' 진입 조건은 과거 추정치 기록이 없어 빠져 있습니다(컨센서스 기록이 쌓이면 추가 검증).",
          "- 배당은 반영하지 않았습니다(보유와 비교할 때 보유 쪽이 조금 불리하게 나옵니다). 국면 게이트 크기 조절도 적용하지 않은 원신호입니다.",
          "- 진입·청산 모두 신호 다음 날 시가입니다. 미청산 거래는 마지막 종가로 평가했습니다.", ""]
    return "\n".join(L)


# =====================================================================
# 틀 B(저평가 매수) — 미국 개별 종목, SEC 공시일 기준 실제 EPS (2026-11-14 추가)
# =====================================================================
# 매수 논리: "가격이 이익 대비 자기 과거보다 싸고, 이익이 줄고 있지 않다."
#   밴드   그날까지 알 수 있었던 TTM EPS로 만든 월말 PER, 직전 5년(최소 24개월)의 로그 평균·σ
#          = 앱 V 트랙의 자기 PER 밴드와 같은 방식(engine/fundamental.py _lagged_ratio_stats, 신뢰 조건도 같음)
#   진입   PER이 밴드 평균 −1σ 이하(싸다) + TTM EPS ≥ 1년 전 TTM EPS(이익 훼손 없음) → 다음 날 시가
#   손절   새 보고서로 TTM EPS가 1년 전보다 줄면(싼 이유가 생김 = 가치 함정) → 다음 날 시가
#   매도   PER이 밴드 평균 +1σ 이상(고평가)이고, 진입 뒤 주가 상승 중 배수 상승분이 이익 증가분보다 크면
#          → 다음 날 시가. 이익 증가가 더 크면 재평가로 보고 보유
#   보고서 숫자는 제출일 다음 거래일부터 쓴다(장 마감 뒤 제출 대비). 컨센서스 추정치는 과거 기록이 없어 쓰지 않는다.
#   비교 기준 = 같은 날 사서 같은 날 판 '같은 묶음의 다른 종목' 평균(로그 수익률, 같은 비용).
#   틀 A처럼 '같은 종목 무작위 날짜 + 같은 보유일'로 비교하면, 보유가 수백 일이고 보유일 자체가 그 종목 경로(계속 싸게
#   머문 종목일수록 오래 보유)에 달려 있어 비교 기준이 낮게 치우쳤다(우위 없는 가짜 가격에서 무작위가 이론값보다 −6%p).
#   틀 B 자체는 치우침이 없었다(이론값 대비 −0.6%p ± 1.3). 다른 종목의 수익은 이 종목의 보유 결정과 무관하므로 치우치지 않고,
#   '다른 종목 대신 이 싼 종목을 산 것이 나았나'라는 저평가 매수의 질문과 같다 — tests/test_frame_b.py가 이를 확인한다.
from .fundamental import FUND_DATA_PARAMS  # noqa: E402  (틀 A 코드와 분리해 아래에 둠)

FRAME_B_CONFIG: Dict[str, Any] = {
    "START": "2011-01-01",          # SEC XBRL(2009~) + 1년 전 비교 + 밴드 24개월 준비 뒤
    "PRICE_START": "2008-01-01",
    "SPLIT_DATE": "2018-01-01",     # 앞 기간 / 뒤 기간 (자료 기간 2011~2026의 대략 절반)
    "BAND_YEARS": FUND_DATA_PARAMS["band_years"],
    "BAND_MIN_POINTS": FUND_DATA_PARAMS["band_min_points"],
    "BAND_MAX_EPS_CHANGE": FUND_DATA_PARAMS["band_max_eps_change"],
    "BAND_MAX_LOG_STD": FUND_DATA_PARAMS["band_max_log_std"],
    "CHEAP_Z": -1.0,
    "RICH_Z": 1.0,
    "COST_SCENARIOS": [0.0, 0.003, 0.006],
    "DD_WARN": -0.30,               # 보유 중 최대 낙폭 경고선
    "MIN_TRADES": 30,               # HOLDOUT 거래가 이보다 적으면 판정 불가
    "EXCLUDED": {"TSM": "외국 기업(20-F·IFRS)이라 us-gaap EPS 없음",
                 "BRK-B": "EPS가 A주 기준이라 B주 가격과 단위가 다름"},
    # 판정 묶음(HOLDOUT): 어떤 백테스트에도 쓰지 않은 미국 대형·중형주. 업종을 고르게, 오래 부진했던 종목도 넣음
    "HOLDOUT_UNIVERSE": [
        "ADP", "AMAT", "ADI", "APD", "BDX", "BSX", "CB", "CI", "CME", "CSX", "D", "DG", "ECL", "EL", "EW", "GIS",
        "HCA", "HUM", "ICE", "ITW", "KMB", "LRCX", "MAR", "MCK", "MMC", "NSC", "NUE", "ORLY", "PGR", "PH", "PNC",
        "ROP", "SPGI", "SRE", "SYK", "TJX", "TRV", "WM", "ZTS", "AZO", "AFL", "ALL", "APH", "CTAS", "EBAY", "HPQ",
        "KR", "LHX", "MSI", "PAYX", "PPG", "ROST", "SYY", "YUM", "KHC", "GPN", "CAG", "HAS", "MOS", "VFC", "WHR",
        "ADM", "AEP", "KLAC",
    ],
}
FRAME_B_CONFIG["DEV_UNIVERSE"] = [t for t in FRAME_A_CONFIG["DEV_UNIVERSE"]
                                  if _fa_market(t) == "US" and t not in FRAME_B_CONFIG["EXCLUDED"]]
FRAME_B_CRITERIA: Dict[str, Any] = {
    # HOLDOUT 전체 기간: 거래 ≥ MIN_TRADES, 거래당 로그 순수익률 > 0, (실제 − 같은 날 다른 종목 평균) ≥ 월 묶음 표준오차 × 2
    # 개발 묶음(대표 종목 미국): 앞·뒤 기간 모두 (실제 − 같은 날 다른 종목 평균) > 0
    "MIN_Z": 2.0,
}
FRAME_B_RULES = ("진입: PER(그날 알려진 TTM EPS 기준)이 자기 5년 로그 PER 밴드 평균 −1σ 이하 + TTM EPS ≥ 1년 전 → 다음 날 시가 · "
                 "손절: 새 보고서로 TTM EPS가 1년 전보다 줄면 · 매도: PER이 밴드 +1σ 이상이고 진입 뒤 배수 상승분 > 이익 증가분 "
                 "(이익 증가가 더 크면 재평가로 보고 보유) · 청산은 다음 날 시가")
EXIT_LABELS_B = {"stop_eps": "손절: 이익 감소(가치 함정)", "sell_rich": "매도: 고평가(배수 주도)",
                 "open": "미청산(마지막 종가로 평가)"}


def frame_b_daily(price_df: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """일별 PER·밴드·조건. EPS는 제출일 다음 거래일부터, 밴드는 그날 이전 월말 기준(그날 이후 값은 쓰지 않음)."""
    cfg = FRAME_B_CONFIG
    d = _bt_naive(price_df)
    out = pd.DataFrame(index=d.index)
    out["Open"], out["Close"] = d["Open"].astype(float), d["Close"].astype(float)
    ev = (events.dropna(subset=["ttm"]).sort_values("filed") if events is not None and len(events)
          else pd.DataFrame(columns=["filed", "ttm", "ttm_prev"]))
    # 날짜 정밀도(초·마이크로초·나노초)가 표마다 다를 수 있어(pandas 버전·야후 자료) 합치기 전에 나노초로 맞춘다
    left = pd.DataFrame({"date": pd.DatetimeIndex(out.index).astype("datetime64[ns]")})
    if len(ev):
        right = ev[["filed", "ttm", "ttm_prev"]].assign(filed=pd.to_datetime(ev["filed"]).astype("datetime64[ns]"))
        m = pd.merge_asof(left, right, left_on="date", right_on="filed", direction="backward", allow_exact_matches=False)
        out["ttm"], out["ttm_prev"] = m["ttm"].to_numpy(dtype=float), m["ttm_prev"].to_numpy(dtype=float)
    else:
        out["ttm"], out["ttm_prev"] = np.nan, np.nan
    ttm = out["ttm"]
    out["PE"] = (out["Close"] / ttm).where(ttm > 0)
    # 월말 밴드 (직전 BAND_YEARS년, 그 월말까지의 값만)
    me_idx = out.index.to_series().groupby(out.index.to_period("M")).max().to_numpy()
    me = out.loc[me_idx, ["PE", "ttm"]]
    rows = []
    for t in me.index:
        w = me[(me.index > t - pd.DateOffset(years=cfg["BAND_YEARS"])) & (me.index <= t)]
        w = w[w["PE"].notna() & (w["PE"] > 0)]
        if len(w) < cfg["BAND_MIN_POINTS"]:
            rows.append((t, np.nan, np.nan, False))
            continue
        lv = np.log(w["PE"].to_numpy())
        mu, sd = float(lv.mean()), float(lv.std())
        chg = float(w["ttm"].max() / w["ttm"].min())
        ok = bool(sd > 0 and sd <= cfg["BAND_MAX_LOG_STD"] and chg <= cfg["BAND_MAX_EPS_CHANGE"])
        rows.append((t, mu, sd, ok))
    band = pd.DataFrame(rows, columns=["date", "band_mu", "band_sd", "band_ok"])
    if len(band):
        band["date"] = pd.to_datetime(band["date"]).astype("datetime64[ns]")
        mb = pd.merge_asof(left, band, on="date", direction="backward", allow_exact_matches=False)
        out["band_mu"], out["band_sd"] = mb["band_mu"].to_numpy(dtype=float), mb["band_sd"].to_numpy(dtype=float)
        out["band_ok"] = mb["band_ok"].eq(True).to_numpy()
    else:
        out["band_mu"], out["band_sd"], out["band_ok"] = np.nan, np.nan, False
    out["z"] = (np.log(out["PE"]) - out["band_mu"]) / out["band_sd"]
    out["grow_ok"] = (ttm > 0) & out["ttm_prev"].notna() & (ttm >= out["ttm_prev"])
    out["cheap"] = out["band_ok"] & out["z"].notna() & (out["z"] <= cfg["CHEAP_Z"])
    out["rich"] = out["band_ok"] & out["z"].notna() & (out["z"] >= cfg["RICH_Z"])
    out["signal"] = out["cheap"] & out["grow_ok"]
    return out


def _fb_arrays(ind: pd.DataFrame) -> Dict[str, np.ndarray]:
    a = {k: ind[k].to_numpy(dtype=float) for k in ("Open", "Close", "ttm", "PE")}
    a["rich"] = ind["rich"].to_numpy(dtype=bool)
    a["nstop"] = _fa_next_true(~ind["grow_ok"].to_numpy(dtype=bool))
    with np.errstate(invalid="ignore", divide="ignore"):
        prev = np.concatenate([[np.nan], a["Close"][:-1]])
        gap = (a["Open"] / prev - 1.0) <= FRAME_A_CONFIG["GAP_DOWN"]
        a["log_pe"], a["log_eps"] = np.log(a["PE"]), np.log(a["ttm"])
    a["cumgap"] = np.cumsum(gap.astype(int))
    return a


def simulate_frame_b_trade(a: Dict[str, np.ndarray], i: int, cost: Optional[float] = None) -> Optional[Dict[str, Any]]:
    """i일 종가 신호 → i+1일 시가 진입 → 이익 감소 손절 / 배수 주도 고평가 매도를 종가로 확인 → 다음 날 시가."""
    cst = _fa_cost() if cost is None else float(cost)
    c, o = a["Close"], a["Open"]
    n = len(c)
    if i + 1 >= n:
        return None
    fill, pe_e, eps_e = float(o[i + 1]), float(a["PE"][i]), float(a["ttm"][i])
    if not (np.isfinite(fill) and fill > 0 and np.isfinite(pe_e) and pe_e > 0 and np.isfinite(eps_e) and eps_e > 0):
        return None
    s = i + 1
    stop = int(a["nstop"][s])
    hi = min(stop, n - 1)
    with np.errstate(invalid="ignore"):
        mult_up = a["log_pe"][s:hi + 1] - math.log(pe_e)
        eps_up = a["log_eps"][s:hi + 1] - math.log(eps_e)
        rich = a["rich"][s:hi + 1]
        sell_mask = rich & (mult_up > eps_up)
        rerate_mask = rich & (mult_up <= eps_up)
    sell = s + int(np.argmax(sell_mask)) if sell_mask.any() else n
    trig = min(stop, sell)
    if trig >= n - 1:
        kind, exit_idx, exit_px, exit_gap = "open", n - 1, float(c[n - 1]), None
    else:
        kind = "stop_eps" if trig == stop else "sell_rich"
        exit_idx, exit_px = trig + 1, float(o[trig + 1])
        exit_gap = float(exit_px / c[trig] - 1.0)
    gross = exit_px / fill - 1.0
    seen = int(min(trig, n - 1) - s + 1)
    return {"fill": fill, "exit_px": exit_px, "exit_kind": kind, "exit_idx": int(exit_idx), "hold_days": int(exit_idx - s),
            "gross": float(gross), "ret": float(gross - cst), "r": np.nan, "exit_gap": exit_gap,
            "gap_days": int(a["cumgap"][exit_idx] - a["cumgap"][s]),
            "mdd": float(np.nanmin(c[s:exit_idx + 1]) / fill - 1.0),
            "pe_entry": pe_e, "rerated": bool(rerate_mask[:seen].any())}


def frame_b_trades(ind: pd.DataFrame, start: Optional[str] = None, cost: Optional[float] = None,
                   arrs: Optional[Dict[str, np.ndarray]] = None) -> pd.DataFrame:
    """한 종목의 틀 B 거래 (종목당 1포지션: 청산 실행일 종가부터 새 신호를 받음)."""
    a = arrs or _fb_arrays(ind)
    s0 = pd.Timestamp(start or FRAME_B_CONFIG["START"])
    out, free_from = [], -1
    for i in np.flatnonzero(ind["signal"].to_numpy(dtype=bool)):
        if ind.index[i] < s0 or i < free_from:
            continue
        res = simulate_frame_b_trade(a, int(i), cost)
        if res is None:
            continue
        out.append({"date": ind.index[i], "z_entry": float(ind["z"].iloc[i]), **res})
        if res["exit_kind"] == "open":
            break
        free_from = res["exit_idx"]
    return pd.DataFrame(out)


def frame_b_tail(t: pd.DataFrame) -> Dict[str, Any]:
    """손실 꼬리: 이 틀은 가격 손절이 없으므로 보유 중 최대 낙폭과 갭 하락을 따로 본다."""
    if not len(t):
        return {"n": 0}
    eg = t["exit_gap"].dropna() if "exit_gap" in t else pd.Series(dtype=float)
    return {"n": int(len(t)), "worst_ret": float(t["ret"].min()), "p5_ret": float(t["ret"].quantile(0.05)),
            "mdd_median": float(t["mdd"].median()) if "mdd" in t else None,
            "share_dd_warn": float((t["mdd"] <= FRAME_B_CONFIG["DD_WARN"]).mean() * 100.0) if "mdd" in t else None,
            "worst_mdd": float(t["mdd"].min()) if "mdd" in t else None,
            "share_exit_gap10": float((eg <= FRAME_A_CONFIG["GAP_DOWN"]).mean() * 100.0) if len(eg) else 0.0,
            "share_with_gap_day": float((t["gap_days"] > 0).mean() * 100.0)}


def paired_stats(tr: pd.DataFrame, col_a: str, col_b: str) -> Dict[str, Any]:
    """거래마다 짝지은 차이(col_a − col_b)의 평균과 월 묶음 표준오차. compare_clustered와 같은 모양으로 돌려준다."""
    t = tr[tr[col_a].notna() & tr[col_b].notna()] if len(tr) else tr
    if not len(t):
        return {"actual": {"n": 0}, "baseline": {"n": 0}, "diff": None, "z": None, "z_iid": None}
    sa, sb = cluster_stats(t[col_a], t["date"]), cluster_stats(t[col_b], t["date"])
    sd = cluster_stats(t[col_a] - t[col_b], t["date"])
    z = sd["mean"] / sd["se"] if sd.get("se") else None
    zi = sd["mean"] / sd["se_iid"] if sd.get("se_iid") else None
    return {"actual": sa, "baseline": sb, "diff": sd["mean"], "z": z, "z_iid": zi, "diff_win_rate": sd["win_rate"]}


def _fb_block(tr: pd.DataFrame) -> Dict[str, Any]:
    out = {"vs_peers": paired_stats(tr, "lret", "peer_lret"), "vs_cheap_peers": paired_stats(tr, "lret", "cheap_peer_lret"),
           "vs_peers_arith": paired_stats(tr, "ret", "peer_ret"),
           "hold_days": float(tr["hold_days"].mean()) if len(tr) else None,
           "hold_days_median": float(tr["hold_days"].median()) if len(tr) else None,
           "tail": frame_b_tail(tr), "exits": {}, "cost": {}, "rerated": None}
    if len(tr):
        for k, g in tr.groupby("exit_kind"):
            out["exits"][EXIT_LABELS_B.get(k, k)] = {"share": float(len(g) / len(tr) * 100.0), "mean_ret": float(g["ret"].mean()),
                                                     "mean_excess": float((g["lret"] - g["peer_lret"]).mean()),
                                                     "hold_days": float(g["hold_days"].mean())}
        rr = tr[tr["rerated"].astype(bool)]
        out["rerated"] = {"share": float(len(rr) / len(tr) * 100.0), "mean_ret": float(rr["ret"].mean()) if len(rr) else None}
        for cst in FRAME_B_CONFIG["COST_SCENARIOS"]:
            x = tr.assign(lret=np.log1p(tr["gross"] - cst))
            st = cluster_stats(x["lret"], x["date"])
            out["cost"][f"{cst * 100:.2f}%"] = {"mean": st.get("mean"), "se": st.get("se")}
    return out


def _fb_peer_returns(tr: pd.DataFrame, sym: str, O: pd.DataFrame, C: pd.DataFrame, cheap: pd.DataFrame,
                     cost: float) -> pd.DataFrame:
    """거래마다 같은 묶음의 다른 종목을 같은 날(진입일 시가) 사서 같은 날(청산일 시가, 미청산은 마지막 종가) 판 결과의 평균.
    '싼 다른 종목'은 신호일에 그 종목도 밴드 −1σ 이하였던 종목만. 다른 종목도 같은 왕복 비용을 낸다."""
    others = [c for c in O.columns if c != sym]
    pr, pl, cr, cl, npe, nch = [], [], [], [], [], []
    for t in tr.itertuples(index=False):
        d_in, d_out, d_sig = t.entry_date, t.exit_date, t.date
        if d_in not in O.index or d_out not in O.index:
            pr.append(np.nan); pl.append(np.nan); cr.append(np.nan); cl.append(np.nan); npe.append(0); nch.append(0)
            continue
        p_in = O.loc[d_in, others]
        p_out = (C if t.exit_kind == "open" else O).loc[d_out, others]
        g = (p_out / p_in - 1.0 - cost)
        g = g[np.isfinite(g) & (g > -1.0)]
        lg = np.log1p(g)
        ch = cheap.loc[d_sig, g.index] if d_sig in cheap.index else pd.Series(False, index=g.index)
        gc = g[ch.fillna(False).astype(bool)]
        pr.append(float(g.mean()) if len(g) else np.nan); pl.append(float(lg.mean()) if len(g) else np.nan)
        cr.append(float(gc.mean()) if len(gc) >= 3 else np.nan); cl.append(float(np.log1p(gc).mean()) if len(gc) >= 3 else np.nan)
        npe.append(int(len(g))); nch.append(int(len(gc)))
    return tr.assign(peer_ret=pr, peer_lret=pl, cheap_peer_ret=cr, cheap_peer_lret=cl, n_peers=npe, n_cheap_peers=nch)


def run_frame_b_universe(price_map: Dict[str, pd.DataFrame], events_map: Dict[str, pd.DataFrame],
                         cost: Optional[float] = None) -> Dict[str, Any]:
    """묶음 하나: 종목별 일별 조건 → 실제 거래 → 같은 날 다른 종목 평균(전체·싼 종목)과 짝지음, 보유 비교, 자료 범위."""
    cfg = FRAME_B_CONFIG
    cst = _fa_cost() if cost is None else float(cost)
    s0 = pd.Timestamp(cfg["START"])
    inds, holds, cover, trs = {}, {}, {}, []
    for sym, df in (price_map or {}).items():
        d = _bt_naive(df)
        ev = (events_map or {}).get(sym)
        if d is None or len(d) < 300 or ev is None or not len(ev):
            cover[sym] = {"status": "가격 또는 SEC EPS 없음"}
            continue
        ind = frame_b_daily(d, ev)
        inds[sym] = ind
        a = _fb_arrays(ind)
        tr = frame_b_trades(ind, cost=cst, arrs=a)
        after = ind[ind.index >= s0]
        cover[sym] = {"status": "사용", "band_ok_share": float(after["band_ok"].mean() * 100.0) if len(after) else 0.0,
                      "cheap_days": int(after["cheap"].sum()), "signal_days": int(after["signal"].sum()),
                      "n_trades": int(len(tr)),
                      "first_ttm": str(ind["ttm"].first_valid_index().date()) if ind["ttm"].notna().any() else None}
        if len(tr):
            pos = tr["date"].map(lambda x: ind.index.get_loc(x)).to_numpy()
            tr = tr.assign(ticker=sym, entry_date=ind.index[pos + 1], exit_date=ind.index[tr["exit_idx"].to_numpy()])
            trs.append(tr)
        h = _fa_vs_hold(ind, tr, s0)
        if h:
            holds[sym] = h
    cols = ["date", "ticker", "entry_date", "exit_date", "fill", "exit_px", "ret", "lret", "gross", "exit_kind", "hold_days",
            "exit_gap", "gap_days", "mdd", "rerated", "pe_entry", "z_entry",
            "peer_ret", "peer_lret", "cheap_peer_ret", "cheap_peer_lret", "n_peers", "n_cheap_peers"]
    if not trs:
        return {"trades": pd.DataFrame(columns=cols), "holds": holds, "coverage": cover}
    O = pd.DataFrame({k: v["Open"] for k, v in inds.items()})
    C = pd.DataFrame({k: v["Close"] for k, v in inds.items()})
    CH = pd.DataFrame({k: v["cheap"] for k, v in inds.items()})
    done = []
    for tr in trs:
        x = _fb_peer_returns(tr, tr["ticker"].iloc[0], O, C, CH, cst)
        x["lret"] = np.log1p(x["ret"].astype(float))
        done.append(x)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        trades = pd.concat(done, ignore_index=True)
    return {"trades": trades.reindex(columns=cols), "holds": holds, "coverage": cover}


def evaluate_frame_b(dev_map: Dict[str, pd.DataFrame], holdout_map: Dict[str, pd.DataFrame],
                     events_map: Dict[str, pd.DataFrame], cost: Optional[float] = None) -> Dict[str, Any]:
    """개발 묶음(대표 종목 미국, 앞·뒤 기간)과 HOLDOUT에서 틀 B를 평가하고 사전 기준으로 판정."""
    cfg = FRAME_B_CONFIG
    split_t = pd.Timestamp(cfg["SPLIT_DATE"])
    res: Dict[str, Any] = {"rules": FRAME_B_RULES, "criteria": FRAME_B_CRITERIA, "excluded": cfg["EXCLUDED"],
                           "config": {k: cfg[k] for k in ("START", "SPLIT_DATE", "BAND_YEARS", "BAND_MIN_POINTS", "BAND_MAX_EPS_CHANGE",
                                                          "BAND_MAX_LOG_STD", "CHEAP_Z", "RICH_Z", "MIN_TRADES", "COST_SCENARIOS")},
                           "cost": _fa_cost() if cost is None else float(cost), "universes": {}, "verdict": {}}
    raw = {"개발(대표 종목 미국)": run_frame_b_universe(dev_map, events_map, cost),
           "HOLDOUT(새 미국 종목)": run_frame_b_universe(holdout_map, events_map, cost)}
    for uname, u in raw.items():
        tr = u["trades"]
        per = {"전체 기간": tr, "앞 기간": tr[tr["date"] < split_t], "뒤 기간": tr[tr["date"] >= split_t]}
        blocks = {p: _fb_block(x) for p, x in per.items()}
        hv = pd.DataFrame(u["holds"]).T.astype(float) if u["holds"] else pd.DataFrame()
        hold_summary = ({"n": int(len(hv)), "bh_cagr_median": float(hv["bh_cagr"].median()),
                         "frame_cagr_median": float(hv["frame_cagr"].median()),
                         "exposure_median": float(hv["exposure"].median()),
                         "share_frame_beats_bh": float((hv["frame_cagr"] > hv["bh_cagr"]).mean() * 100.0)} if len(hv) else {"n": 0})
        used = [k for k, v in u["coverage"].items() if v["status"] == "사용"]
        res["universes"][uname] = {
            "n_tickers": len(u["coverage"]), "n_used": len(used),
            "no_data": sorted(k for k, v in u["coverage"].items() if v["status"] != "사용"),
            "n_with_trades": int(sum(1 for k in used if u["coverage"][k]["n_trades"] > 0)),
            "band_ok_share_median": float(np.median([u["coverage"][k]["band_ok_share"] for k in used])) if used else None,
            "blocks": blocks, "vs_hold": hold_summary}
    ho = res["universes"]["HOLDOUT(새 미국 종목)"]["blocks"]["전체 기간"]["vs_peers"]
    dv = res["universes"]["개발(대표 종목 미국)"]["blocks"]
    enough = ho["actual"].get("n", 0) >= cfg["MIN_TRADES"]
    ok_ho = bool(enough and ho["actual"]["mean"] > 0 and ho["diff"] is not None and ho["diff"] > 0
                 and ho["z"] is not None and ho["z"] >= FRAME_B_CRITERIA["MIN_Z"])
    ok_dev = all(dv[p]["vs_peers"]["actual"].get("n", 0) >= 2 and (dv[p]["vs_peers"]["diff"] or 0) > 0 for p in ("앞 기간", "뒤 기간"))
    chp = res["universes"]["HOLDOUT(새 미국 종목)"]["blocks"]["전체 기간"]["vs_cheap_peers"]
    res["verdict"] = {"passed": ok_ho and ok_dev, "holdout_pass": ok_ho, "dev_both_positive": ok_dev, "enough_trades": enough,
                      "earnings_filter_adds": bool(chp.get("z") is not None and chp["z"] >= FRAME_B_CRITERIA["MIN_Z"]),
                      "criteria": (f"HOLDOUT 전체 기간 거래 {cfg['MIN_TRADES']}건 이상, 거래당 로그 순수익률 > 0 이고 "
                                   f"(실제 − 같은 날 같은 묶음 다른 종목 평균) ≥ 월 묶음 표준오차 × {FRAME_B_CRITERIA['MIN_Z']:.0f}, "
                                   "그리고 개발 묶음 앞·뒤 기간 모두 (실제 − 다른 종목 평균) > 0")}
    trades = pd.concat([u["trades"].assign(universe=k) for k, u in raw.items() if len(u["trades"])], ignore_index=True) \
        if any(len(u["trades"]) for u in raw.values()) else pd.DataFrame()
    return {"summary": res, "trades": trades, "coverage": {k: u["coverage"] for k, u in raw.items()}}


def render_frame_b_report(r: Dict[str, Any], generated_at: Optional[str] = None) -> str:
    v, cfg = r["verdict"], r["config"]
    head = "판정 불가(HOLDOUT 거래 수 부족)" if not v["enough_trades"] else ("통과" if v["passed"] else "미통과")
    L = ["# 틀 B(저평가 매수) 미국 개별 종목 검증 — SEC 공시일 기준 실제 EPS", "",
         f"- 생성: {generated_at or '—'} · 신호 {cfg['START'][:4]}년~ · 왕복 비용 {r['cost'] * 100:.2f}% 차감 · 앞/뒤 기간 경계 {cfg['SPLIT_DATE']}",
         f"- 규칙(결과 보기 전 고정): {r['rules']}",
         f"- 밴드: 월말 PER의 직전 {cfg['BAND_YEARS']}년 로그 평균·σ(최소 {cfg['BAND_MIN_POINTS']}개월). 밴드 기간 EPS 변화 {cfg['BAND_MAX_EPS_CHANGE']:.0f}배 초과 또는 "
         f"로그 σ {cfg['BAND_MAX_LOG_STD']} 초과면 그 달은 판정하지 않음(앱 V 트랙과 같은 조건).",
         "- EPS: SEC에 제출된 10-Q·10-K의 희석 EPS로 만든 최근 12개월 값. 제출일 다음 거래일부터 사용. 액면분할은 지금 주식 단위로 맞춤.",
         "- 비교(판정 기준): 거래마다 같은 날 사서 같은 날 판 '같은 묶음의 다른 종목' 평균(같은 비용 차감). 보유가 수백 일로 길어 "
         "'같은 종목 무작위 날짜 + 같은 보유일' 비교는 우위가 없는 가짜 가격에서도 한쪽으로 치우쳤기 때문입니다(보유일이 그 종목 경로에 달려 있음).",
         "- '싼 다른 종목'(참고): 같은 날 PER이 밴드 −1σ 이하였던 다른 종목만의 평균(3종목 이상일 때) — 이익 조건과 매도 규칙이 보태는지 봅니다.",
         "- 수익률은 로그 수익률 ln(매도가/매수가)(비용 차감). 단순 수익률 평균은 참고 열에 적었습니다. 표준오차는 같은 달 신호를 한 덩어리로 묶어 계산, 괄호 안 z는 독립 가정(참고).",
         f"- 판정 기준: {v['criteria']}.",
         "- 제외: " + ", ".join(f"{k}({why})" for k, why in r["excluded"].items()), "",
         f"## 판정: {head}", "",
         f"- HOLDOUT 기준 {'통과' if v['holdout_pass'] else '미통과'} · 개발 묶음 앞·뒤 기간 {'모두 다른 종목보다 나음' if v['dev_both_positive'] else '하나 이상 다른 종목 이하'} · "
         f"이익 조건·매도 규칙의 기여('싼 다른 종목' 대비 z ≥ 2) {'확인' if v['earnings_filter_adds'] else '확인 안 됨'}", ""]
    chk = r.get("eps_check") or []
    if chk:
        L += ["### 자료 점검: SEC로 만든 최근 12개월 EPS vs 야후 trailingEps", "",
              "두 값이 비슷하면 SEC 자료를 제대로 읽은 것입니다(야후가 일회성 항목을 다르게 처리하면 조금 다를 수 있음).", "",
              "| 종목 | SEC TTM EPS (기간 끝 · 제출일) | 야후 trailingEps |", "|---|---|---|"]
        for c in chk:
            y = "—" if c.get("yahoo_trailing_eps") is None else f"{float(c['yahoo_trailing_eps']):.2f}"
            L.append(f"| {c['ticker']} | {c['sec_ttm']:.2f} ({c['ttm_end']} · {c['filed']}) | {y} |")
        L.append("")
    notes = r.get("sec_notes") or {}
    if notes:
        L += ["SEC 자료 참고: " + "; ".join(f"{k} {v}" for k, v in notes.items()), ""]
    for uname, u in r["universes"].items():
        L += [f"## {uname}", "",
              f"종목 {u['n_tickers']}개 중 자료 사용 {u['n_used']}개 · 거래가 나온 종목 {u['n_with_trades']}개 · "
              f"밴드를 쓸 수 있던 날 비중(종목 중앙값) {'—' if u['band_ok_share_median'] is None else format(u['band_ok_share_median'], '.0f') + '%'}"
              + (f" · 자료 없음: {', '.join(u['no_data'])}" if u["no_data"] else ""), "",
              "| 기간 | 틀 B 거래당 로그 순수익률 ±표준오차 (승률·건수) | 같은 날 다른 종목 평균 | 차이 (틀 B가 나은 비율) | z (독립 가정) | 평균·중앙 보유일 | 참고: 단순 수익률 (틀 B · 다른 종목) |",
              "|---|---|---|---|---|---|---|"]
        for per, b in u["blocks"].items():
            x = b["vs_peers"]
            hd = "—" if b["hold_days"] is None else f"{b['hold_days']:.0f} · {b['hold_days_median']:.0f}"
            ar = b["vs_peers_arith"]
            arc = f"{_fa_pct(ar['actual'].get('mean'))} · {_fa_pct(ar['baseline'].get('mean'))}" if ar["actual"].get("n") else "—"
            dw = "" if x.get("diff_win_rate") is None else f" ({x['diff_win_rate']:.0f}%)"
            z = "—" if x.get("z") is None else f"{x['z']:+.1f}"
            zi = "—" if x.get("z_iid") is None else f"{x['z_iid']:+.1f}"
            L.append(f"| {per} | {_fa_one(x['actual'])} | {_fa_one(x['baseline'])} | {_fa_pct(x.get('diff'))}{dw} | {z} ({zi}) | {hd} | {arc} |")
        full = u["blocks"]["전체 기간"]
        c = full["vs_cheap_peers"]
        if c["actual"].get("n"):
            L += ["", "| 비교 (로그 수익률) | 틀 B | 같은 날 싼 다른 종목 | 차이 | z (독립 가정) |", "|---|---|---|---|---|",
                  f"| 전체 기간 | {_fa_one(c['actual'])} | {_fa_one(c['baseline'])} | {_fa_pct(c.get('diff'))} | "
                  f"{'—' if c.get('z') is None else format(c['z'], '+.1f')} ({'—' if c.get('z_iid') is None else format(c['z_iid'], '+.1f')}) |"]
        L.append("")
        if full["exits"]:
            L += ["| 청산 이유 | 비중 | 거래당 단순 순수익률 | 다른 종목 대비 로그 초과 | 평균 보유일 |", "|---|---|---|---|---|"]
            for k, e in sorted(full["exits"].items(), key=lambda kv: -kv[1]["share"]):
                L.append(f"| {k} | {e['share']:.0f}% | {_fa_pct(e['mean_ret'])} | {_fa_pct(e['mean_excess'])} | {e['hold_days']:.0f} |")
            rr = full["rerated"]
            L += ["", f"보유 중 '고평가지만 이익 증가가 더 커서 재평가로 보고 보유'한 적이 있는 거래: {rr['share']:.0f}% "
                      f"(거래당 {_fa_pct(rr['mean_ret'])})", ""]
        t = full["tail"]
        if t.get("n"):
            L += ["| 손실 꼬리 (가격 손절이 없는 틀) | 틀 B |", "|---|---|",
                  f"| 보유 중 최대 낙폭 중앙값 | {_fa_pct(t['mdd_median'], 1)} |",
                  f"| 보유 중 30% 넘게 빠진 적 있는 거래 | {t['share_dd_warn']:.1f}% |",
                  f"| 가장 큰 보유 중 낙폭 | {_fa_pct(t['worst_mdd'], 1)} |",
                  f"| 하위 5% 거래 수익률 · 가장 나쁜 거래 | {_fa_pct(t['p5_ret'])} · {_fa_pct(t['worst_ret'])} |",
                  f"| 보유 중 −10% 이상 갭 하락을 겪은 거래 | {t['share_with_gap_day']:.1f}% |", ""]
        if full["cost"]:
            L += ["| 왕복 비용 | 틀 B 거래당 로그 수익률 ±표준오차 |", "|---|---|"]
            for k, cc in full["cost"].items():
                L.append(f"| {k} | {_fa_pct(cc['mean'])}{'' if cc.get('se') is None else ' ±' + format(cc['se'] * 100, '.2f')} |")
            L += ["", "다른 종목도 같은 비용을 낸다고 보므로 '다른 종목 대비 차이'는 비용에 따라 달라지지 않습니다.", ""]
        h = u["vs_hold"]
        if h.get("n"):
            L += [f"매수 후 보유와 비교(참고, 종목별 연 환산 중앙값): 틀 B {_fa_pct(h['frame_cagr_median'], 1)} · 보유 {_fa_pct(h['bh_cagr_median'], 1)} · "
                  f"틀 B가 보유보다 나은 종목 {h['share_frame_beats_bh']:.0f}% · 주식을 들고 있던 기간 비중 {h['exposure_median'] * 100:.0f}%", ""]
    L += ["## 주의",
          "- 지금 상장된 종목만 썼습니다(상장 폐지 종목 없음). 싸 보이다가 결국 망한 종목이 빠져 있어 저평가 매수에 유리하게 나올 수 있습니다.",
          "- 비교 대상인 '다른 종목'도 지금 상장된 종목이라 같은 편향을 함께 갖지만, 싼 종목일수록 상장 폐지 위험이 커서 완전히 상쇄되지는 않습니다.",
          "- 컨센서스 추정치(앞으로의 이익 예상)는 과거 기록이 없어 쓰지 않았습니다. 실제로 발표된 이익 기준입니다.",
          "- 배당은 반영하지 않았습니다(저평가 종목은 배당이 많은 편이라 틀 B에 조금 불리합니다). 국면 게이트·분할 매수·분할 매도는 적용하지 않은 단순화입니다.",
          "- EPS는 희석 EPS(없으면 기본 EPS)이며 일회성 손익을 걸러내지 않았습니다. 미청산 거래는 마지막 종가로 평가했습니다.", ""]
    return "\n".join(L)
