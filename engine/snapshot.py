"""컨센서스·목표가 날짜별 기록 (2026-11-14 추가).

야후는 '지금'의 컨센서스만 준다. 과거 어느 날의 컨센서스 EPS·목표가가 얼마였는지는 나중에 구할 수 없으므로,
매일 한 번 받아 data/snapshots/consensus_<연도>.csv에 쌓는다 (GitHub Actions의 daily_agent.yml이 snapshot_daily.py 실행).
실적·가치 트랙과 '저평가 매수' 틀을 나중에 과거 시점 기준으로 검증하기 위한 원자료다.

- 판정 결과가 아니라 원자료를 저장한다(판정 규칙은 바뀌어도 원자료로 다시 계산할 수 있게).
- 못 받은 값은 빈칸. 추정하거나 전날 값으로 채우지 않는다.
- snapshot_date = 받은 날(한국 날짜). price_as_of = 가격 데이터의 날짜.
- '0y'(올해 회계연도) 추정치는 회계연도가 바뀌면 대상 연도가 바뀌므로 회계연도 끝(fy1_end)을 함께 저장한다.
"""
import os
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from .rule_backtest import BACKTEST_CONFIG

SNAPSHOT_CONFIG: Dict[str, Any] = {
    "DIR": os.path.join("data", "snapshots"),
    "WATCHLIST_FILE": "watchlist.txt",     # 한 줄에 티커 하나. '#'로 시작하면 설명 줄
    "TIMEZONE": "Asia/Seoul",
    "PAUSE_SEC": 0.4,                      # 종목 사이 대기 (야후 요청 제한 완화)
    # 기본 대상: 섹터 대표 종목(미국) + 한국 대표 종목. watchlist.txt로 추가한다
    "UNIVERSE": list(BACKTEST_CONFIG["SETUP_UNIVERSE"]["대표 종목"]),
}

# CSV 열 (순서 고정. 새 열은 맨 뒤에만 추가한다 — 이미 쌓인 파일과 어긋나지 않게)
SNAPSHOT_COLUMNS: List[str] = [
    "snapshot_date", "ticker", "market", "currency", "price", "price_as_of",
    "trailing_eps", "forward_eps", "trailing_pe", "forward_pe", "price_to_book", "market_cap", "shares_outstanding",
    "target_mean", "target_median", "target_high", "target_low", "n_target_analysts", "recommendation_mean",
    "eps_fy1_avg", "eps_fy1_low", "eps_fy1_high", "eps_fy1_n", "eps_fy1_year_ago",
    "eps_fy2_avg", "eps_fy2_low", "eps_fy2_high", "eps_fy2_n",
    "eps_q0_avg", "eps_q1_avg",
    "eps_fy1_7d_ago", "eps_fy1_30d_ago", "eps_fy1_60d_ago", "eps_fy1_90d_ago",
    "eps_fy2_7d_ago", "eps_fy2_30d_ago", "eps_fy2_60d_ago", "eps_fy2_90d_ago",
    "eps_fy1_up_7d", "eps_fy1_up_30d", "eps_fy1_down_7d", "eps_fy1_down_30d",
    "eps_fy2_up_7d", "eps_fy2_up_30d", "eps_fy2_down_7d", "eps_fy2_down_30d",
    "rev_fy1_avg", "rev_fy1_n", "rev_fy2_avg", "rev_fy2_n",
    "fy1_end", "last_fy_end", "next_earnings_date", "source",
]
# 이 중 하나라도 받았으면 저장한다 (전부 없으면 그 종목은 실패로 기록)
_KEY_FIELDS = ["price", "forward_eps", "target_mean", "eps_fy1_avg"]


def _f(v: Any) -> Optional[float]:
    try:
        x = float(v)
        return x if np.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def _cell(df: Any, row: str, col: str) -> Optional[float]:
    """야후 추정치 표에서 (행, 열) 값. 열 이름은 대소문자를 구분하지 않는다(upLast7days · downLast7Days처럼 섞여 있음)."""
    try:
        if df is None or len(df) == 0 or row not in df.index:
            return None
        cols = {str(c).lower(): c for c in df.columns}
        c = cols.get(col.lower())
        return _f(df.loc[row, c]) if c is not None else None
    except Exception:
        return None


def _epoch_date(v: Any) -> Optional[str]:
    x = _f(v)
    if not x:
        return None
    try:
        return pd.Timestamp(x, unit="s").strftime("%Y-%m-%d")
    except Exception:
        return None


def snapshot_market(ticker: str) -> str:
    return "KR" if str(ticker).upper().endswith((".KS", ".KQ")) else "US"


def fetch_consensus_row(ticker: str, yf: Any = None) -> Dict[str, Any]:
    """한 종목의 오늘 컨센서스·목표가·가격을 받아 SNAPSHOT_COLUMNS 모양의 dict로 돌려준다(snapshot_date 제외).
    각 항목은 따로 시도하고, 실패한 항목은 None. 반환 dict의 "_notes"에 실패한 항목 이름."""
    if yf is None:
        import yfinance as yf
    row: Dict[str, Any] = {c: None for c in SNAPSHOT_COLUMNS}
    row.update({"ticker": ticker, "market": snapshot_market(ticker), "source": "yfinance"})
    notes: List[str] = []
    tk = yf.Ticker(ticker)

    try:   # 가격과 그 날짜 (데이터의 날짜)
        h = tk.history(period="5d", auto_adjust=False)
        h = h[h["Close"].notna()] if h is not None and len(h) else h
        if h is not None and len(h):
            row["price"] = _f(h["Close"].iloc[-1])
            row["price_as_of"] = str(h.index[-1])[:10]
    except Exception:
        notes.append("가격")

    info: Dict[str, Any] = {}
    try:
        info = tk.info or {}
    except Exception:
        notes.append("기본 정보")
    for src, dst in (("trailingEps", "trailing_eps"), ("forwardEps", "forward_eps"), ("trailingPE", "trailing_pe"),
                     ("forwardPE", "forward_pe"), ("priceToBook", "price_to_book"), ("marketCap", "market_cap"),
                     ("sharesOutstanding", "shares_outstanding"), ("targetMeanPrice", "target_mean"),
                     ("targetMedianPrice", "target_median"), ("targetHighPrice", "target_high"),
                     ("targetLowPrice", "target_low"), ("numberOfAnalystOpinions", "n_target_analysts"),
                     ("recommendationMean", "recommendation_mean")):
        row[dst] = _f(info.get(src))
    row["currency"] = info.get("currency") or ("KRW" if row["market"] == "KR" else None)
    row["fy1_end"], row["last_fy_end"] = _epoch_date(info.get("nextFiscalYearEnd")), _epoch_date(info.get("lastFiscalYearEnd"))
    if row["price"] is None:   # 일봉을 못 받았을 때만 기본 정보의 현재가(날짜 없음)
        row["price"] = _f(info.get("currentPrice")) or _f(info.get("regularMarketPrice"))

    def table(attr: str, label: str) -> Any:
        try:
            return getattr(tk, attr)
        except Exception:
            notes.append(label)
            return None

    ee = table("earnings_estimate", "EPS 추정치")
    for fy, r in (("fy1", "0y"), ("fy2", "+1y")):
        for col, key in (("avg", "avg"), ("low", "low"), ("high", "high"), ("numberOfAnalysts", "n")):
            row[f"eps_{fy}_{key}"] = _cell(ee, r, col)
    row["eps_fy1_year_ago"] = _cell(ee, "0y", "yearAgoEps")
    row["eps_q0_avg"], row["eps_q1_avg"] = _cell(ee, "0q", "avg"), _cell(ee, "+1q", "avg")

    et = table("eps_trend", "EPS 추이")
    for fy, r in (("fy1", "0y"), ("fy2", "+1y")):
        for col, key in (("7daysAgo", "7d_ago"), ("30daysAgo", "30d_ago"), ("60daysAgo", "60d_ago"), ("90daysAgo", "90d_ago")):
            row[f"eps_{fy}_{key}"] = _cell(et, r, col)
        if row[f"eps_{fy}_avg"] is None:   # 추정치 표가 비었으면 추이 표의 현재값
            row[f"eps_{fy}_avg"] = _cell(et, r, "current")

    er = table("eps_revisions", "EPS 수정 건수")
    for fy, r in (("fy1", "0y"), ("fy2", "+1y")):
        for col, key in (("upLast7days", "up_7d"), ("upLast30days", "up_30d"), ("downLast7days", "down_7d"), ("downLast30days", "down_30d")):
            row[f"eps_{fy}_{key}"] = _cell(er, r, col)

    rv = table("revenue_estimate", "매출 추정치")
    for fy, r in (("fy1", "0y"), ("fy2", "+1y")):
        row[f"rev_{fy}_avg"], row[f"rev_{fy}_n"] = _cell(rv, r, "avg"), _cell(rv, r, "numberOfAnalysts")

    try:   # 다음 실적 발표일 (범위면 가장 이른 날)
        cal = tk.calendar
        ed = (cal or {}).get("Earnings Date") if isinstance(cal, dict) else None
        if ed is not None:
            ds = [pd.Timestamp(x) for x in (ed if isinstance(ed, (list, tuple)) else [ed]) if x is not None]
            if ds:
                row["next_earnings_date"] = min(ds).strftime("%Y-%m-%d")
    except Exception:
        notes.append("실적 발표일")
    row["_notes"] = notes
    return row


def snapshot_universe(base_dir: Optional[str] = None) -> List[str]:
    """기록 대상: 기본 목록 + data/snapshots/watchlist.txt (중복 제거, 순서 유지)."""
    out = list(SNAPSHOT_CONFIG["UNIVERSE"])
    p = os.path.join(base_dir or SNAPSHOT_CONFIG["DIR"], SNAPSHOT_CONFIG["WATCHLIST_FILE"])
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8-sig") as f:
            for line in f:
                t = line.split("#", 1)[0].strip()
                if t:
                    out.append(t.upper())
    return list(dict.fromkeys(out))


def snapshot_path(date: str, base_dir: Optional[str] = None) -> str:
    return os.path.join(base_dir or SNAPSHOT_CONFIG["DIR"], f"consensus_{str(date)[:4]}.csv")


def _saved_keys(path: str) -> set:
    if not os.path.exists(path):
        return set()
    d = pd.read_csv(path, usecols=["snapshot_date", "ticker"], dtype=str, encoding="utf-8-sig")
    return set(zip(d["snapshot_date"], d["ticker"]))


def append_snapshot_rows(rows: List[Dict[str, Any]], date: str, base_dir: Optional[str] = None) -> Dict[str, int]:
    """그날 기록을 연도별 CSV 끝에 붙인다. 같은 (날짜, 티커)가 이미 있으면 건너뛴다(다시 실행해도 중복 없음)."""
    path = snapshot_path(date, base_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    have = _saved_keys(path)
    new = [{**{c: r.get(c) for c in SNAPSHOT_COLUMNS}, "snapshot_date": date}
           for r in rows if (date, r["ticker"]) not in have]
    if new:
        exists = os.path.exists(path)
        pd.DataFrame(new, columns=SNAPSHOT_COLUMNS).to_csv(path, mode="a", header=not exists, index=False,
                                                           encoding="utf-8" if exists else "utf-8-sig")
    return {"saved": len(new), "skipped_existing": len(rows) - len(new)}


def run_daily_snapshot(tickers: Optional[List[str]] = None, base_dir: Optional[str] = None, yf: Any = None,
                       today: Optional[str] = None, pause: Optional[float] = None,
                       progress: Optional[Callable[[int, int, str], None]] = None) -> Dict[str, Any]:
    """오늘 기록을 받아 저장한다. 이미 오늘 저장된 종목은 받지 않는다.
    반환: {date, path, saved, skipped_existing, failed: [티커], partial: {티커: [못 받은 항목]}}"""
    date = today or pd.Timestamp.now(tz=SNAPSHOT_CONFIG["TIMEZONE"]).strftime("%Y-%m-%d")
    uni = list(dict.fromkeys(tickers or snapshot_universe(base_dir)))
    have = _saved_keys(snapshot_path(date, base_dir))
    todo = [t for t in uni if (date, t) not in have]
    wait = SNAPSHOT_CONFIG["PAUSE_SEC"] if pause is None else float(pause)
    rows, failed, partial = [], [], {}
    for i, t in enumerate(todo, 1):
        try:
            r = fetch_consensus_row(t, yf)
        except Exception:
            failed.append(t)
            r = None
        if r is not None:
            notes = r.pop("_notes", [])
            if all(r.get(k) is None for k in _KEY_FIELDS):
                failed.append(t)
            else:
                rows.append(r)
                if notes:
                    partial[t] = notes
        if progress:
            progress(i, len(todo), t)
        if wait > 0 and i < len(todo):
            time.sleep(wait)
    res = append_snapshot_rows(rows, date, base_dir) if rows else {"saved": 0, "skipped_existing": 0}
    return {"date": date, "path": snapshot_path(date, base_dir), "saved": res["saved"],
            "skipped_existing": len(uni) - len(todo), "failed": failed, "partial": partial, "n_universe": len(uni)}


def load_snapshots(base_dir: Optional[str] = None) -> pd.DataFrame:
    """쌓인 기록 전체(연도별 파일을 합침). 없으면 빈 표. 날짜 열은 날짜형으로 바꾼다."""
    d = base_dir or SNAPSHOT_CONFIG["DIR"]
    files = sorted(f for f in (os.listdir(d) if os.path.isdir(d) else []) if f.startswith("consensus_") and f.endswith(".csv"))
    if not files:
        return pd.DataFrame(columns=SNAPSHOT_COLUMNS)
    df = pd.concat([pd.read_csv(os.path.join(d, f), encoding="utf-8-sig", dtype={"ticker": str}) for f in files], ignore_index=True)
    df = df.reindex(columns=SNAPSHOT_COLUMNS + [c for c in df.columns if c not in SNAPSHOT_COLUMNS])
    for c in ("snapshot_date", "price_as_of", "fy1_end", "last_fy_end", "next_earnings_date"):
        df[c] = pd.to_datetime(df[c], errors="coerce")
    return df.sort_values(["snapshot_date", "ticker"]).reset_index(drop=True)
