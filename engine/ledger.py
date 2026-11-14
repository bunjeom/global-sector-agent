"""신호 기록 장부 (2026-11-15 추가) — 틀 A v2 판정을 매일 쌓고, 나중에 미리 정한 방법으로 채점한다.

과거 자료는 여러 번 들여다봤기 때문에, 앞으로 쌓이는 기록이 판정 규칙을 검증할 유일하게 깨끗한 자료다.
    기록   매일 한 번(GitHub Actions daily_agent.yml → ledger_daily.py) 대표 종목 + data/snapshots/watchlist.txt의
           틀 A v2 판정을 data/ledger/frame_a_<연도>.csv에 붙인다. 판정 결과와 함께 판정에 쓴 원자료(스윙 점, 이동평균,
           상대강도, 추정치 건수, 자기 밴드 z)를 저장해 규칙이 바뀌어도 다시 계산할 수 있게 한다.
           줄마다 판정 규칙 버전(PLAYBOOK_VERSION)을 남긴다. 같은 (날짜, 티커)는 다시 실행해도 한 번만 저장.
           쌓인 CSV는 손으로 고치거나 지우지 않는다. 열은 LEDGER_COLUMNS 순서로 고정, 새 열은 맨 뒤에만.
    채점   (기록을 시작하기 전에 LEDGER_EVAL로 고정) evaluate_frame_a_ledger
           1) 전환 가능성 표시별: 하락·횡보 상태의 '우세/팽팽/실패 우세' 줄마다, 다음 날 시가에 사서 20·60거래일 뒤 시가에 판
              로그 수익률 − 같은 날 같은 기간 다른 종목 평균(초과 수익). 그리고 그 기간 안에 상승 추세가 된 비율
           2) 진입 의견: 의견이 나온 다음 날 시가에 사서, 이후 기록에서 하락 쪽 근거가 더 많아진 다음 날 시가에 판 거래의
              초과 수익. 판정 = 기록 12개월 이상, 의견 30건 이상, 초과 수익 평균 > 0, 월 묶음 z ≥ 2
"""
import math
import os
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from .frame_backtest import cluster_stats
from .playbook import PLAYBOOK_CONFIG, PLAYBOOK_VERSION, frame_a_evaluate, light_band_z, trend_indicators, _pb_market
from .rule_backtest import _bt_naive
from .snapshot import load_snapshots, snapshot_universe

LEDGER_CONFIG: Dict[str, Any] = {
    "DIR": os.path.join("data", "ledger"),
    "TIMEZONE": "Asia/Seoul",
    "PAUSE_SEC": 0.3,
    "PRICE_PERIOD": "2y",
}
# 채점 방법 (기록 시작 전에 고정 — 결과를 본 뒤 바꾸지 않는다)
LEDGER_EVAL: Dict[str, Any] = {
    "HORIZONS": [20, 60],           # 거래일
    "MIN_MONTHS": 12,               # 기록이 이만큼 쌓이기 전에는 '판정 보류'
    "MIN_OPINIONS": 30,
    "MIN_Z": 2.0,
    "LEAN_LABELS": ["전환 가능성 우세", "팽팽", "실패 가능성 우세"],
}
LEDGER_COLUMNS: List[str] = [
    "ledger_date", "ticker", "market", "version", "as_of", "close", "atr",
    "state", "state_detail", "direction", "lean_label", "support", "against", "diff",
    "s_ma50", "s_slope20", "s_lows", "s_rs", "s_estimates", "s_band", "s_event",
    "entry_opinion", "size_mult", "chase_caution",
    "sma50", "sma200", "ma_state", "line_low", "line_high",
    "sw_l1_date", "sw_l1", "sw_l2_date", "sw_l2", "sw_h1_date", "sw_h1", "sw_h2_date", "sw_h2",
    "rs63", "est_up_30d", "est_down_30d", "band_z", "band_reason", "notes",
]
_INT_COLS = ["support", "against", "diff", "s_ma50", "s_slope20", "s_lows", "s_rs", "s_estimates", "s_band", "s_event",
             "est_up_30d", "est_down_30d"]
_ITEM_COL = {"ma50": "s_ma50", "slope20": "s_slope20", "lows": "s_lows", "rs": "s_rs", "estimates": "s_estimates",
             "band": "s_band", "event": "s_event"}


def _lv(v: Any) -> Any:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return v
    return None if not math.isfinite(f) else f


def ledger_row(ticker: str, date: str, ev: Dict[str, Any], ind_row: Optional[pd.Series], index: Optional[pd.Index],
               consensus: Optional[Dict[str, Any]], band: Dict[str, Any]) -> Dict[str, Any]:
    """판정 결과(ev)와 원자료를 LEDGER_COLUMNS 모양의 dict로. 판정할 수 없던 날은 notes에 이유만 남긴다."""
    r: Dict[str, Any] = {c: None for c in LEDGER_COLUMNS}
    cs = consensus or {}
    r.update({"ledger_date": date, "ticker": ticker, "market": _pb_market(ticker), "version": PLAYBOOK_VERSION,
              "est_up_30d": _lv(cs.get("eps_fy1_up_30d")), "est_down_30d": _lv(cs.get("eps_fy1_down_30d")),
              "band_z": _lv(band.get("z")), "band_reason": band.get("reason")})
    if not ev.get("available"):
        r["as_of"], r["notes"] = ev.get("as_of"), ev.get("reason")
        return r
    op = ev["opinion"]
    r.update({"as_of": ev["as_of"], "close": _lv(ev["close"]), "atr": _lv(ev["atr"]), "state": ev["state"],
              "state_detail": ev.get("state_detail"), "direction": ev["lean"]["direction"], "lean_label": ev["lean"]["label"],
              "support": ev["support"], "against": ev["against"], "diff": ev["diff"],
              "entry_opinion": bool(op["entry"]), "size_mult": op["size_mult"], "chase_caution": bool(op["chase_caution"]),
              "notes": " / ".join(op["notes"]) or None})
    for it in ev["items"]:
        col = _ITEM_COL.get(it["key"])
        if col:
            r[col] = None if it["no_data"] else int(it["points"])
    if ind_row is not None:
        for k, col in (("SMA50", "sma50"), ("SMA200", "sma200"), ("line_low", "line_low"), ("line_high", "line_high"), ("rs63", "rs63")):
            r[col] = _lv(ind_row.get(k))
        r["ma_state"] = ind_row.get("ma_state")
        for k, col in (("sw_L1", "sw_l1"), ("sw_L2", "sw_l2"), ("sw_H1", "sw_h1"), ("sw_H2", "sw_h2")):
            p = ind_row.get(k)
            if isinstance(p, dict):
                r[col] = p["price"]
                r[col + "_date"] = str(index[p["idx"]].date()) if index is not None else None
    return r


def ledger_path(date: str, base_dir: Optional[str] = None) -> str:
    return os.path.join(base_dir or LEDGER_CONFIG["DIR"], f"frame_a_{str(date)[:4]}.csv")


def _ledger_keys(path: str) -> set:
    if not os.path.exists(path):
        return set()
    d = pd.read_csv(path, usecols=["ledger_date", "ticker"], dtype=str, encoding="utf-8-sig")
    return set(zip(d["ledger_date"], d["ticker"]))


def append_ledger_rows(rows: List[Dict[str, Any]], date: str, base_dir: Optional[str] = None) -> Dict[str, int]:
    path = ledger_path(date, base_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    have = _ledger_keys(path)
    new = [{c: r.get(c) for c in LEDGER_COLUMNS} for r in rows if (date, r["ticker"]) not in have]
    if new:
        exists = os.path.exists(path)
        df = pd.DataFrame(new, columns=LEDGER_COLUMNS)
        for col in _INT_COLS:   # 빈칸이 섞여도 점수는 정수로 저장(1.0이 아니라 1)
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
        df.to_csv(path, mode="a", header=not exists, index=False, encoding="utf-8" if exists else "utf-8-sig")
    return {"saved": len(new), "skipped_existing": len(rows) - len(new)}


def _yf_history(yf: Any, sym: str, period: str) -> Optional[pd.DataFrame]:
    df = yf.download(sym, period=period, auto_adjust=False, actions=False, progress=False, threads=False)
    if df is None or not len(df):
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna(subset=["Close"])


def run_daily_ledger(tickers: Optional[List[str]] = None, base_dir: Optional[str] = None, yf: Any = None,
                     today: Optional[str] = None, snapshot_dir: Optional[str] = None, pause: Optional[float] = None,
                     progress: Optional[Callable[[int, int, str], None]] = None) -> Dict[str, Any]:
    """오늘의 틀 A v2 판정을 기록한다. 추정치 건수는 같은 날 컨센서스 기록(data/snapshots/)에서 가져온다(없으면 빈칸).
    반환: {date, path, saved, skipped_existing, failed: [티커], no_judgment: {티커: 이유}, n_universe}"""
    if yf is None:
        import yfinance as yf
    date = today or pd.Timestamp.now(tz=LEDGER_CONFIG["TIMEZONE"]).strftime("%Y-%m-%d")
    uni = list(dict.fromkeys(tickers or snapshot_universe(snapshot_dir)))
    have = _ledger_keys(ledger_path(date, base_dir))
    todo = [t for t in uni if (date, t) not in have]
    snaps = load_snapshots(snapshot_dir)
    snaps = snaps[snaps["snapshot_date"] == pd.Timestamp(date)] if len(snaps) else snaps
    cs_map = {str(r["ticker"]): r.to_dict() for _, r in snaps.iterrows()} if len(snaps) else {}
    benches: Dict[str, Optional[pd.DataFrame]] = {}
    for mk, sym in PLAYBOOK_CONFIG["BENCH"].items():
        try:
            benches[mk] = _yf_history(yf, sym, LEDGER_CONFIG["PRICE_PERIOD"])
        except Exception:
            benches[mk] = None
    wait = LEDGER_CONFIG["PAUSE_SEC"] if pause is None else float(pause)
    rows, failed, nojudge = [], [], {}
    for i, t in enumerate(todo, 1):
        try:
            px = _yf_history(yf, t, LEDGER_CONFIG["PRICE_PERIOD"])
            if px is None:
                raise ValueError("가격 없음")
            bench = benches.get(_pb_market(t))
            ind = trend_indicators(px, bench)
            cs = cs_map.get(t)
            band = light_band_z(t, yf=yf)
            ev = frame_a_evaluate(px, bench, cs, band.get("z"), ind=ind)
            row = ledger_row(t, date, ev, ind.iloc[-1] if ev.get("available") else None, ind.index, cs, band)
            if not ev.get("available"):
                nojudge[t] = ev.get("reason")
            rows.append(row)
        except Exception as e:   # 네트워크·야후 오류: 그 종목만 빼고 계속
            failed.append(f"{t}({type(e).__name__})")
        if progress:
            progress(i, len(todo), t)
        if wait > 0 and i < len(todo):
            time.sleep(wait)
    res = append_ledger_rows(rows, date, base_dir) if rows else {"saved": 0, "skipped_existing": 0}
    return {"date": date, "path": ledger_path(date, base_dir), "saved": res["saved"],
            "skipped_existing": len(uni) - len(todo), "failed": failed, "no_judgment": nojudge, "n_universe": len(uni),
            "no_consensus": len([t for t in todo if t not in cs_map])}


def load_ledger(base_dir: Optional[str] = None) -> pd.DataFrame:
    d = base_dir or LEDGER_CONFIG["DIR"]
    files = sorted(f for f in (os.listdir(d) if os.path.isdir(d) else []) if f.startswith("frame_a_") and f.endswith(".csv"))
    if not files:
        return pd.DataFrame(columns=LEDGER_COLUMNS)
    df = pd.concat([pd.read_csv(os.path.join(d, f), encoding="utf-8-sig", dtype={"ticker": str}) for f in files], ignore_index=True)
    df = df.reindex(columns=LEDGER_COLUMNS + [c for c in df.columns if c not in LEDGER_COLUMNS])
    for c in ("ledger_date", "as_of"):
        df[c] = pd.to_datetime(df[c], errors="coerce")
    return df.sort_values(["as_of", "ticker"]).reset_index(drop=True)


# ---------------------------------------------------------------------
# 채점 (LEDGER_EVAL로 고정한 방법)
# ---------------------------------------------------------------------
def _open_close_panels(price_map: Dict[str, pd.DataFrame]):
    o, c = {}, {}
    for k, v in (price_map or {}).items():
        d = _bt_naive(v)
        if d is None or not len(d):
            continue
        o[k], c[k] = d["Open"].astype(float), d["Close"].astype(float)
    return pd.DataFrame(o).sort_index(), pd.DataFrame(c).sort_index()


def _window_lret(O: pd.DataFrame, C: pd.DataFrame, sym: str, d_in: pd.Timestamp, d_out: pd.Timestamp, at_close: bool) -> Dict[str, Any]:
    """d_in 시가에 사서 d_out 시가(at_close면 종가)에 판 로그 수익률과, 같은 날 다른 종목 평균."""
    p_out = (C if at_close else O).loc[d_out]
    p_in = O.loc[d_in]
    lr = np.log(p_out / p_in)
    lr = lr[np.isfinite(lr)]
    if sym not in lr.index:
        return {"lret": None, "peer": None, "n_peers": 0}
    peers = lr.drop(sym)
    return {"lret": float(lr[sym]), "peer": float(peers.mean()) if len(peers) else None, "n_peers": int(len(peers))}


def evaluate_frame_a_ledger(ledger: pd.DataFrame, price_map: Dict[str, pd.DataFrame],
                            horizons: Optional[List[int]] = None) -> Dict[str, Any]:
    """기록을 미리 정한 방법(LEDGER_EVAL)으로 채점한다. 판정 규칙 버전이 섞여 있으면 버전별로 따로 채점한다."""
    hz = horizons or LEDGER_EVAL["HORIZONS"]
    O, C = _open_close_panels(price_map)
    out: Dict[str, Any] = {"eval": LEDGER_EVAL, "horizons": list(hz), "versions": {}}
    if ledger is None or not len(ledger) or not len(O):
        return {**out, "note": "기록 또는 가격 자료 없음"}
    lg = ledger[ledger["state"].notna()].copy()
    for ver, g in lg.groupby("version"):
        g = g.sort_values(["ticker", "as_of"])
        months = (g["as_of"].max() - g["as_of"].min()).days / 30.44 if len(g) else 0.0
        # 1) 전환 가능성 표시별
        buckets: Dict[str, Any] = {}
        recs = []
        for r in g[g["state"].isin(["하락", "횡보"])].itertuples(index=False):
            if r.ticker not in O.columns:
                continue
            pos = O.index.searchsorted(r.as_of, side="right")      # as_of 다음 거래일
            for h in hz:
                j = pos + h
                if j >= len(O.index):
                    continue
                w = _window_lret(O, C, r.ticker, O.index[pos], O.index[j], False)
                if w["lret"] is None or w["peer"] is None:
                    continue
                later = g[(g["ticker"] == r.ticker) & (g["as_of"] > r.as_of) & (g["as_of"] <= O.index[j])]
                recs.append({"h": h, "lean": r.lean_label, "date": r.as_of, "excess": w["lret"] - w["peer"],
                             "became_up": bool((later["state"] == "상승").any())})
        rd = pd.DataFrame(recs)
        for h in hz:
            buckets[str(h)] = {}
            for lab in LEDGER_EVAL["LEAN_LABELS"]:
                x = rd[(rd["h"] == h) & (rd["lean"] == lab)] if len(rd) else rd
                buckets[str(h)][lab] = ({**cluster_stats(x["excess"], x["date"]), "became_up_rate": float(x["became_up"].mean() * 100.0)}
                                        if len(x) else {"n": 0})
        # 2) 진입 의견 거래
        trades = []
        for t, gt in g.groupby("ticker"):
            if t not in O.columns:
                continue
            rows = gt.reset_index(drop=True)
            k = 0
            while k < len(rows):
                if not bool(rows.loc[k, "entry_opinion"]):
                    k += 1
                    continue
                pin = O.index.searchsorted(rows.loc[k, "as_of"], side="right")
                if pin >= len(O.index):
                    break
                ex = None
                for m in range(k + 1, len(rows)):
                    if float(rows.loc[m, "against"]) > float(rows.loc[m, "support"]):
                        ex = m
                        break
                if ex is not None:
                    pout = O.index.searchsorted(rows.loc[ex, "as_of"], side="right")
                    if pout >= len(O.index):
                        ex = None
                if ex is None:
                    w = _window_lret(O, C, t, O.index[pin], O.index[-1], True)
                    kind = "open"
                else:
                    w = _window_lret(O, C, t, O.index[pin], O.index[pout], False)
                    kind = "exit"
                if w["lret"] is not None and w["peer"] is not None:
                    trades.append({"ticker": t, "date": rows.loc[k, "as_of"], "kind": kind, "lret": w["lret"],
                                   "peer": w["peer"], "excess": w["lret"] - w["peer"]})
                if ex is None:
                    break
                k = ex + 1
        td = pd.DataFrame(trades)
        st = cluster_stats(td["excess"], td["date"]) if len(td) else {"n": 0}
        z = (st["mean"] / st["se"]) if st.get("se") else None
        enough = months >= LEDGER_EVAL["MIN_MONTHS"] and st.get("n", 0) >= LEDGER_EVAL["MIN_OPINIONS"]
        verdict = ("판정 보류(자료 부족)" if not enough else
                   ("통과" if (st["mean"] > 0 and z is not None and z >= LEDGER_EVAL["MIN_Z"]) else "미통과"))
        out["versions"][ver] = {"months": round(months, 1), "n_rows": int(len(g)), "n_tickers": int(g["ticker"].nunique()),
                                "buckets": buckets, "opinions": {**st, "z": z,
                                                                 "n_open": int((td["kind"] == "open").sum()) if len(td) else 0},
                                "verdict": verdict, "trades": trades}
    return out


def render_ledger_report(r: Dict[str, Any], generated_at: Optional[str] = None) -> str:
    e = r["eval"]
    hz = r.get("horizons") or e["HORIZONS"]
    L = ["# 신호 기록 장부 채점 — 틀 A v2", "",
         f"- 생성: {generated_at or '—'} · 채점 방법은 기록을 시작하기 전에 고정(engine/ledger.py LEDGER_EVAL)",
         f"- 판정: 기록 {e['MIN_MONTHS']}개월 이상 · 진입 의견 {e['MIN_OPINIONS']}건 이상일 때, 의견 거래의 초과 수익(같은 날 다른 종목 평균 대비 로그 수익률) "
         f"평균 > 0 이고 월 묶음 z ≥ {e['MIN_Z']:.0f}", ""]
    if r.get("note"):
        return "\n".join(L + [r["note"], ""])
    for ver, v in r["versions"].items():
        op = v["opinions"]
        L += [f"## 판정 규칙 {ver}: {v['verdict']}", "",
              f"기록 {v['months']}개월 · {v['n_rows']:,}줄 · {v['n_tickers']}종목", "",
              "| 전환 가능성 표시 (하락·횡보) | " + " | ".join(f"{h}거래일 초과 수익 (건수) · 상승 추세 도달" for h in hz) + " |",
              "|---|" + "---|" * len(hz)]
        for lab in e["LEAN_LABELS"]:
            cells = []
            for h in hz:
                b = v["buckets"][str(h)][lab]
                cells.append("—" if not b.get("n") else f"{b['mean'] * 100:+.2f}% ({b['n']:,}) · {b['became_up_rate']:.0f}%")
            L.append(f"| {lab} | " + " | ".join(cells) + " |")
        L += ["", ("진입 의견 거래: 표본 없음" if not op.get("n") else
                   f"진입 의견 거래 {op['n']}건(미청산 {op['n_open']}) · 초과 수익 평균 {op['mean'] * 100:+.2f}%"
                   + (f" ± {op['se'] * 100:.2f} (z {op['z']:+.1f})" if op.get("se") else "")), ""]
    L += ["## 주의",
          "- 같은 종목이 며칠 연속 같은 표시를 받으면 기간이 겹친 표본이 여러 개 생긴다. 표준오차는 같은 달 기록을 한 덩어리로 묶어 계산했다.",
          "- 상승 추세 도달은 장부에 기록된 날의 판정으로만 센다(기록이 빠진 날은 모름).", ""]
    return "\n".join(L)
