"""가설 탐색 도구 (2026-11-17) — run_backtest.py --explore <이름> → data/backtest/EXPLORE_<이름>.md.

탐색용과 확인용을 나눈다(AGENTS.md 4번 '탐색과 확인').
    탐색  지금까지 판정에 이미 쓴 종목 전부(EXPLORE_CONFIG['UNIVERSE'])로 가설을 자유롭게 본다. 결과는 '탐색 — 증거 아님'.
          화면·문구에 검증된 것처럼 쓰지 않는다. 볼 때마다 EXPLORE_LOG.md에 한 줄씩 남긴다(몇 번 봤는지가 남아야 한다).
    확인  탐색에서 '후보' 기준을 넘은 가설만, 정의를 한 글자도 바꾸지 않고, 아직 쓰지 않은 종목(또는 앞으로 쌓이는 기록)에서
          z 3 기준으로 한 번 시험한다. 그때 비로소 '검증됨'.

가설 하나 = 사건을 찾는 함수 하나. fn(si, a) → 사건이 일어난 날의 위치(정수 배열). 그날 종가까지의 자료만 써야 한다.
    si = engine.short_indicators(가격) — 시가·고가·저가·종가, SMA20/50/200, ATR20, rs63, 추세 상태(state), RSI2, 지지선(support), 손절선(stop)
    a  = engine.level_arrays(si)      — 배열 모음(Close, Low, ATR20, support, stop, drop(최근 5거래일 변화÷ATR) 등)
결과는 모두 같은 방식으로 잰다: 사건 다음 날 시가 → 10거래일 뒤 종가 수익률(20거래일은 참고)을 같은 종목 무작위 날,
최근 하락 폭이 비슷한 날과 비교. 표준오차는 같은 달 사건을 한 덩어리로.
"""
import math
import os
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from .frame_backtest import FRAME_A_CONFIG, FRAME_B_CONFIG, compare_clustered
from .level_backtest import LEVEL_BT_CONFIG, level_arrays, stop_baseline, stop_events
from .reclaim_backtest import RECLAIM_BT_CONFIG, break_events
from .rule_backtest import _bt_naive
from .short_backtest import SHORT_BT_CONFIG
from .strategy import short_indicators


def _explore_universe() -> List[str]:
    seen: Dict[str, None] = {}
    for cfg in (FRAME_A_CONFIG, FRAME_B_CONFIG, SHORT_BT_CONFIG, LEVEL_BT_CONFIG, RECLAIM_BT_CONFIG):
        for k in ("DEV_UNIVERSE", "HOLDOUT_UNIVERSE"):
            for t in cfg.get(k, []):
                seen[t] = None
    return list(seen)


EXPLORE_CONFIG: Dict[str, Any] = {
    "START": "2001-01-01",
    "DATA_START": "1999-01-01",
    "SPLIT_DATE": FRAME_A_CONFIG["SPLIT_DATE"],
    "HORIZON": LEVEL_BT_CONFIG["HORIZON"],            # 10거래일(판정에 쓴 것과 같음)
    "HORIZON_LONG": LEVEL_BT_CONFIG["HORIZON_LONG"],  # 20거래일(참고)
    "MIN_BARS": 260,
    "GAP": 10,                    # 기본: 사건 뒤 10거래일은 같은 종목의 새 사건을 세지 않음
    # '후보'(확인 시험으로 올릴 만함) 기준 — 미리 고정. 넘어도 증거가 아니라 '확인해 볼 만하다'는 뜻
    "CANDIDATE_Z": 2.0,           # 무작위 날 대비 |z| ≥ 2
    "MIN_EVENTS": 300,
    # 탐색용 종목 = 지금까지 판정·개발에 쓴 개별 종목 전부(ETF 제외). 여기서 본 결과는 확인이 아니다
    "UNIVERSE": _explore_universe(),
    "CACHE_DIR": os.path.join("data", "backtest", "price_cache"),
    "LOG_FILE": "EXPLORE_LOG.md",
}


def dedupe_events(idx: np.ndarray, gap: Optional[int] = None) -> np.ndarray:
    """사건 위치(오름차순)에서, 앞 사건 뒤 gap 거래일 안에 있는 사건을 뺀다."""
    g = EXPLORE_CONFIG["GAP"] if gap is None else int(gap)
    out, free = [], -1
    for t in np.asarray(idx, dtype=int):
        if t >= free:
            out.append(int(t))
            free = t + g + 1
    return np.array(out, dtype=int)


# ---------------------------------------------------------------------
# 가설 등록부. 새 가설은 함수 하나를 쓰고 여기에 한 줄 추가한다. expect = 기대 방향(+1 평소보다 좋음, −1 나쁨)
# ---------------------------------------------------------------------
def _h_reclaim(si: pd.DataFrame, a: Dict[str, np.ndarray]) -> np.ndarray:
    b = break_events(si, a, start=EXPLORE_CONFIG["START"])
    return b.loc[b["reclaimed"] == 1.0, "u"].astype(int).to_numpy() if len(b) else np.array([], dtype=int)


def _h_stop_break(si: pd.DataFrame, a: Dict[str, np.ndarray]) -> np.ndarray:
    e = stop_events(si, a, start=EXPLORE_CONFIG["START"])
    return e["t"].astype(int).to_numpy() if len(e) else np.array([], dtype=int)


EXPLORE_HYPOTHESES: Dict[str, Dict[str, Any]] = {
    "reclaim": {"label": "지지선을 깼다가 10거래일 안에 다시 올라온 날", "expect": 1, "fn": _h_reclaim,
                "note": "RECLAIM.md에서 확인 시험까지 끝난 가설(미통과) — 도구 점검·견본용"},
    "stop_break": {"label": "종가가 손절선 아래로 처음 내려온 날", "expect": -1, "fn": _h_stop_break,
                   "note": "LEVELS.md 질문 2(미통과) — 도구 점검·견본용"},
}


# ---------------------------------------------------------------------
# 계산
# ---------------------------------------------------------------------
def explore_ticker(df: pd.DataFrame, fn: Callable[[pd.DataFrame, Dict[str, np.ndarray]], np.ndarray]) -> Optional[Dict[str, pd.DataFrame]]:
    c = EXPLORE_CONFIG
    d = _bt_naive(df)
    if d is not None and len(d):
        cols = [k for k in ("Open", "High", "Low", "Close") if k in d]
        d = d[(d[cols].astype(float) > 0).all(axis=1)]
    if d is None or len(d) < c["MIN_BARS"]:
        return None
    si = short_indicators(d)
    a = level_arrays(si)
    idx = np.asarray(fn(si, a), dtype=int)
    n = len(si)
    s0 = pd.Timestamp(c["START"])
    idx = np.array([t for t in np.unique(idx) if 0 <= t < n and si.index[t] >= s0 and np.isfinite(a["r_h"][t]) and np.isfinite(a["drop"][t])],
                   dtype=int)
    if not len(idx):
        empty = pd.DataFrame(columns=["date", "r10", "r20", "tail"])
        return {"ev": empty, "rnd": empty, "mat": empty}
    r20 = a["r_hl"][idx]
    ev = pd.DataFrame({"date": si.index[idx], "t": idx, "drop": a["drop"][idx], "r10": a["r_h"][idx], "r20": r20,
                       "tail": np.where(np.isfinite(r20), (r20 <= LEVEL_BT_CONFIG["TAIL"]).astype(float), np.nan)})
    return {"ev": ev, "rnd": stop_baseline(si, ev, a, matched=False, start=c["START"]),
            "mat": stop_baseline(si, ev, a, matched=True, start=c["START"])}


def _ex_block(ev: pd.DataFrame, rnd: pd.DataFrame, mat: pd.DataFrame) -> Dict[str, Any]:
    out = {"vs_random": compare_clustered(ev, rnd, "r10"), "vs_matched": compare_clustered(ev, mat, "r10"),
           "r20_vs_random": compare_clustered(ev, rnd, "r20"), "tail_vs_random": compare_clustered(ev, rnd, "tail"),
           "win_rate": float((ev["r10"] > 0).mean()) if len(ev) else None,
           "win_rate_random": float((rnd["r10"] > 0).mean()) if len(rnd) else None}
    return out


def run_explore(name: str, price_map: Dict[str, pd.DataFrame],
                fn: Optional[Callable[[pd.DataFrame, Dict[str, np.ndarray]], np.ndarray]] = None) -> Dict[str, Any]:
    """탐색 한 번. name은 EXPLORE_HYPOTHESES의 이름(fn을 직접 주면 그 함수로)."""
    c = EXPLORE_CONFIG
    h = EXPLORE_HYPOTHESES.get(name, {})
    f = fn or h.get("fn")
    if f is None:
        raise KeyError(f"등록되지 않은 가설: {name} (등록된 이름: {', '.join(EXPLORE_HYPOTHESES)})")
    parts: Dict[str, List[pd.DataFrame]] = {"ev": [], "rnd": [], "mat": []}
    used = 0
    for sym, df in (price_map or {}).items():
        r = explore_ticker(df, f)
        if r is None:
            continue
        used += 1
        mk = "KR" if str(sym).upper().endswith((".KS", ".KQ")) else "US"
        for k in parts:
            if len(r[k]):
                parts[k].append(r[k].assign(ticker=sym, market=mk))
    cat = lambda lst: pd.concat(lst, ignore_index=True) if lst else pd.DataFrame(columns=["date", "r10", "r20", "tail", "ticker", "market"])
    ev, rnd, mat = cat(parts["ev"]), cat(parts["rnd"]), cat(parts["mat"])
    split_t = pd.Timestamp(c["SPLIT_DATE"])
    per = {"전체 기간": lambda x: x, "앞 기간": lambda x: x[x["date"] < split_t], "뒤 기간": lambda x: x[x["date"] >= split_t],
           "전체 기간 · 미국": lambda x: x[x["market"] == "US"], "전체 기간 · 한국": lambda x: x[x["market"] == "KR"]}
    blocks = {p: _ex_block(g(ev), g(rnd), g(mat)) for p, g in per.items()}
    full, fr, bk = blocks["전체 기간"]["vs_random"], blocks["앞 기간"]["vs_random"], blocks["뒤 기간"]["vs_random"]
    n = full["actual"].get("n", 0)
    z, diff = full.get("z"), full.get("diff")
    same_sign = bool(fr.get("diff") is not None and bk.get("diff") is not None and diff is not None
                     and np.sign(fr["diff"]) == np.sign(bk["diff"]) == np.sign(diff) and diff != 0)
    expect = int(h.get("expect", 0))
    as_expected = bool(diff is not None and expect != 0 and np.sign(diff) == expect)
    candidate = bool(n >= c["MIN_EVENTS"] and z is not None and abs(z) >= c["CANDIDATE_Z"] and same_sign and (expect == 0 or as_expected))
    reasons = []
    if n < c["MIN_EVENTS"]:
        reasons.append(f"사건 {n:,}건 < {c['MIN_EVENTS']}")
    if z is None or abs(z) < c["CANDIDATE_Z"]:
        reasons.append(f"|z| < {c['CANDIDATE_Z']:.0f}")
    if not same_sign:
        reasons.append("앞·뒤 기간 방향이 다름")
    if expect != 0 and not as_expected and diff is not None:
        reasons.append("기대한 방향과 반대")
    return {"name": name, "label": h.get("label", name), "note": h.get("note"), "expect": expect, "n_tickers": used, "n_events": n,
            "blocks": blocks, "candidate": candidate, "not_candidate_reasons": reasons, "same_sign": same_sign,
            "config": {k: c[k] for k in ("START", "SPLIT_DATE", "HORIZON", "HORIZON_LONG", "CANDIDATE_Z", "MIN_EVENTS")},
            "criteria": (f"후보 = 사건 {c['MIN_EVENTS']}건 이상, 무작위 날 대비 |z| ≥ {c['CANDIDATE_Z']:.0f}, 앞·뒤 기간 같은 방향"
                         + (", 기대한 방향" if expect else ""))}


# ---------------------------------------------------------------------
# 보고서·기록
# ---------------------------------------------------------------------
def _ex_ret(s: Dict[str, Any]) -> str:
    if not s.get("n"):
        return "표본 없음"
    se = f" ±{s['se'] * 100:.2f}" if s.get("se") is not None else ""
    return f"{s['mean'] * 100:+.2f}%{se} ({s['n']:,}건)"


def _ex_dz(x: Dict[str, Any]) -> str:
    if x.get("diff") is None:
        return "—"
    return f"{x['diff'] * 100:+.2f}%p · {'—' if x.get('z') is None else format(x['z'], '+.1f')}"


def render_explore_report(r: Dict[str, Any], generated_at: Optional[str] = None) -> str:
    cfg = r["config"]
    H = cfg["HORIZON"]
    exp = {1: "평소보다 좋음", -1: "평소보다 나쁨", 0: "방향 정하지 않음"}[r["expect"]]
    L = [f"# 탐색: {r['label']}", "",
         "**탐색 결과 — 증거 아님.** 이미 판정에 쓴 종목으로 본 것이라, 좋게 나와도 '확인해 볼 만하다'는 뜻일 뿐이다. "
         "화면·문구에 검증된 것처럼 쓰지 않는다.", "",
         f"- 생성: {generated_at or '—'} · 가설 이름 `{r['name']}` · 기대 방향: {exp}" + (f" · {r['note']}" if r.get("note") else ""),
         f"- 종목 {r['n_tickers']}개(지금까지 쓴 개별 종목 전부) · 사건 {cfg['START'][:4]}년~ · 사건 다음 날 시가 → {H}거래일 뒤 종가 수익률",
         "- 비교: 같은 종목 무작위 날(기준) · 최근 5거래일 하락 폭이 비슷한 날(참고). 표준오차는 월 묶음.",
         f"- {r['criteria']}.", "",
         f"## 후보 여부: {'후보 — 확인 시험으로 올릴 만함' if r['candidate'] else '후보 아님'}"
         + ("" if r["candidate"] else f" ({', '.join(r['not_candidate_reasons'])})"), "",
         f"| 기간 | 사건 뒤 {H}거래일 | 무작위 날 | 차이 · z | 하락 폭 맞춘 날 대비 차이 · z | 오른 비율(사건 · 무작위) |", "|---|---|---|---|---|---|"]
    for per, b in r["blocks"].items():
        x = b["vs_random"]
        wr = "—" if b["win_rate"] is None else f"{b['win_rate'] * 100:.0f}% · {(b['win_rate_random'] or 0) * 100:.0f}%"
        L.append(f"| {per} | {_ex_ret(x['actual'])} | {_ex_ret(x['baseline'])} | {_ex_dz(x)} | {_ex_dz(b['vs_matched'])} | {wr} |")
    full = r["blocks"]["전체 기간"]
    r20, tl = full["r20_vs_random"], full["tail_vs_random"]
    L += ["", f"참고 — {cfg['HORIZON_LONG']}거래일 뒤: {_ex_ret(r20['actual'])} vs 무작위 날 {_ex_ret(r20['baseline'])} ({_ex_dz(r20)}) · "
              f"-10% 이하 비율 차이 · z: {_ex_dz(tl)}", "",
          "## 읽는 법", "",
          "- 후보가 아니면 여기서 끝낸다. 숫자를 바꿔 가며 후보가 될 때까지 다시 보는 것은 탐색이 아니라 맞추기다(본 횟수는 EXPLORE_LOG.md에 남는다).",
          "- 후보면 정의를 그대로 두고, 아직 쓰지 않은 종목이나 앞으로 쌓이는 기록에서 z 3 기준으로 한 번 확인한다.", ""]
    return "\n".join(L)


def explore_log_line(r: Dict[str, Any], generated_at: str) -> str:
    x = r["blocks"]["전체 기간"]["vs_random"]
    d = "—" if x.get("diff") is None else f"{x['diff'] * 100:+.2f}%p"
    z = "—" if x.get("z") is None else f"{x['z']:+.1f}"
    return f"| {generated_at} | `{r['name']}` | {r['label']} | {r['n_events']:,} | {d} | {z} | {'후보' if r['candidate'] else '후보 아님'} |"


def append_explore_log(r: Dict[str, Any], generated_at: str, out_dir: str) -> str:
    """탐색을 한 번 볼 때마다 한 줄. 몇 번 봤는지가 남아야 '여러 번 보다 우연히 걸린 것'을 가려낼 수 있다."""
    p = os.path.join(out_dir, EXPLORE_CONFIG["LOG_FILE"])
    os.makedirs(out_dir, exist_ok=True)
    if not os.path.exists(p):
        with open(p, "w", encoding="utf-8") as f:
            f.write("# 탐색 기록\n\n탐색(run_backtest.py --explore)을 한 번 볼 때마다 한 줄씩 쌓인다. 탐색 결과는 증거가 아니다. "
                    "같은 가설을 숫자만 바꿔 여러 번 본 흔적이 있으면 그 가설의 '후보' 판정은 믿지 않는다.\n\n"
                    "| 본 때 | 이름 | 가설 | 사건 수 | 무작위 날 대비 차이 | z | 판정 |\n|---|---|---|---|---|---|---|\n")
    with open(p, "a", encoding="utf-8") as f:
        f.write(explore_log_line(r, generated_at) + "\n")
    return p


# ---------------------------------------------------------------------
# 가격 캐시 — 탐색은 같은 종목을 여러 번 보므로 한 번 받은 가격을 PC에 저장해 둔다(커밋하지 않음)
# ---------------------------------------------------------------------
def load_price_cache(symbols: List[str], cache_dir: Optional[str] = None) -> Dict[str, pd.DataFrame]:
    d = cache_dir or EXPLORE_CONFIG["CACHE_DIR"]
    out = {}
    for s in symbols:
        p = os.path.join(d, f"{s}.csv")
        if os.path.exists(p):
            try:
                df = pd.read_csv(p, index_col=0, parse_dates=True)
                if len(df) and "Close" in df:
                    out[s] = df
            except Exception:
                pass
    return out


def save_price_cache(price_map: Dict[str, pd.DataFrame], cache_dir: Optional[str] = None) -> int:
    d = cache_dir or EXPLORE_CONFIG["CACHE_DIR"]
    os.makedirs(d, exist_ok=True)
    for s, df in (price_map or {}).items():
        df.to_csv(os.path.join(d, f"{s}.csv"))
    return len(price_map or {})
