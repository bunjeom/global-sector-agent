"""피어 순위 채점 (딥다이브 개편 6단계, 2026-11-17) — run_backtest.py --peer-eval → data/backtest/PEER_EVAL.md.

피어 비교의 가치 관점·추세 관점 순위가 이후 수익을 가르는지 앞으로 쌓이는 기록으로 채점한다. 채점 방법은 결과(첫 채점 가능 시점은
2027-05 무렵)를 보기 전인 2026-11-17에 PEER_EVAL로 고정했다. 기록(data/peers/peer_rank_YYYY.csv)은 손대지 않고 읽기만 한다.

방법
    시작일  2026-11-17 — 첫 전체 갱신 점검 뒤 묶음표를 고친 날(peer_map v1.3·생존성 v2.2가 반영된 첫 기록). 그 전 기록은 채점하지 않는다.
    줄 세운 날  시작일 이후 매달 첫 순위 기록일(rank_date). 그날의 묶음별 순위를 쓴다.
    상위·하위  관점마다 순위가 나온 묶음(점수 있는 종목 5개 이상)에서 상위 1/3(= 기록의 value_top·trend_top, ceil(n/3)개)과
               하위 1/3(순위 > n − ceil(n/3)). 생존성 하드 킬 종목은 순위가 없어 어느 쪽에도 들어가지 않는다.
    수익률  종목마다 순위의 가격 기준일(price_as_of) 다음 거래일 시가에 사서 126거래일(6개월, 판정)·252거래일(12개월, 참고) 뒤 종가.
            21·63거래일(1·3개월)은 판정 전까지 흐름을 보는 중간 참고. 같은 비중, 비용 빼기 전.
    한 달의 값  묶음마다 (상위 평균 − 하위 평균)을 구하고, 그달 묶음들의 평균. (상위 평균 − 묶음 평균)은 참고.
    표준오차  달마다 한 값. 보유 기간이 겹치므로 Newey-West(6개월 시차 5, 12개월 시차 11, 3개월 시차 2, 1개월 0).
    판정  관점마다 따로: 6개월 결과가 다 나온 달이 12개 이상이고, (상위 − 하위) 평균 > 0, t ≥ 2. 그 전에는 '판정 보류'.
    참고  하드 킬 종목의 수익률 − 같은 묶음 평균(생존성 걸러내기가 실제로 나쁜 종목을 걸렀는지).
"""
import math
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .momentum_explore import nw_se
from .rule_backtest import _bt_naive

PEER_EVAL: Dict[str, Any] = {
    "START": "2026-11-17",
    "HORIZONS": {"1개월": 21, "3개월": 63, "6개월": 126, "12개월": 252},
    "PRIMARY": "6개월",
    "NW_LAG": {"1개월": 0, "3개월": 2, "6개월": 5, "12개월": 11},
    "MIN_MONTHS": 12,
    "MIN_T": 2.0,
    "STRATEGIES": {"value": "가치 관점", "trend": "추세 관점"},
}


def formation_dates(ranks: pd.DataFrame) -> List[pd.Timestamp]:
    """시작일 이후 매달 첫 순위 기록일."""
    d = pd.to_datetime(ranks["rank_date"]).dropna()
    d = d[d >= pd.Timestamp(PEER_EVAL["START"])].drop_duplicates().sort_values()
    if not len(d):
        return []
    return list(d.groupby(d.dt.to_period("M")).min())


def _fwd_returns(df: Optional[pd.DataFrame], as_of: Any) -> Dict[str, float]:
    """가격 기준일 다음 거래일 시가 → H거래일 뒤 종가. 자료가 모자라면 그 기간은 빠진다."""
    out: Dict[str, float] = {}
    d = _bt_naive(df) if df is not None else None
    if d is None or not len(d) or as_of is None or pd.isna(as_of):
        return out
    idx = d.index
    i = int(np.searchsorted(idx.values, np.datetime64(pd.Timestamp(as_of)), side="right"))   # 기준일 다음 거래일
    if i >= len(d):
        return out
    op = float(d["Open"].iloc[i]) if "Open" in d else float(d["Close"].iloc[i])
    if not (op > 0):
        return out
    cl = d["Close"].to_numpy(dtype=float)
    for name, h in PEER_EVAL["HORIZONS"].items():
        j = i + h - 1          # 매수한 날 포함 h거래일째 종가
        if j <= len(d) - 1 and cl[j] > 0:
            out[name] = cl[j] / op - 1.0
    return out


def group_month_rows(ranks: pd.DataFrame, price_map: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """묶음·달·관점·기간마다 상위·하위·묶음 평균과 하드 킬 종목 평균. 열: month, date, group, market, strategy, horizon,
    top, bottom, group_mean, killed, n_top, n_bottom."""
    rows = []
    cache: Dict[tuple, Dict[str, float]] = {}
    for d in formation_dates(ranks):
        cur = ranks[pd.to_datetime(ranks["rank_date"]) == d]
        for gk, g in cur.groupby("group"):
            rets = {}
            for _, r in g.iterrows():
                t = str(r["ticker"])
                key = (t, str(r.get("price_as_of")))
                if key not in cache:
                    cache[key] = _fwd_returns(price_map.get(t), r.get("price_as_of"))
                rets[t] = cache[key]
            mk = "한국" if all(str(t).endswith((".KS", ".KQ")) for t in g["ticker"]) else "미국"
            killed = g[g["surv_kill"].map(lambda v: v is True or str(v).lower() == "true")]["ticker"].astype(str).tolist()
            for s in PEER_EVAL["STRATEGIES"]:
                rk = pd.to_numeric(g[f"{s}_rank"], errors="coerce")
                ranked = g[rk.notna()]
                n = len(ranked)
                if n < 5:
                    continue
                k = math.ceil(n / 3)
                top = ranked[pd.to_numeric(ranked[f"{s}_rank"]) <= k]["ticker"].astype(str).tolist()
                bot = ranked[pd.to_numeric(ranked[f"{s}_rank"]) > n - k]["ticker"].astype(str).tolist()
                for h in PEER_EVAL["HORIZONS"]:
                    val = lambda ts: [rets[t][h] for t in ts if h in rets.get(t, {})]
                    tv, bv, allv, kv = val(top), val(bot), val(g["ticker"].astype(str).tolist()), val(killed)
                    if not tv or not bv:
                        continue
                    rows.append({"month": d.to_period("M"), "date": d, "group": gk, "market": mk, "strategy": s, "horizon": h,
                                 "top": float(np.mean(tv)), "bottom": float(np.mean(bv)), "group_mean": float(np.mean(allv)),
                                 "killed": float(np.mean(kv)) if kv else np.nan, "n_top": len(tv), "n_bottom": len(bv)})
    return pd.DataFrame(rows, columns=["month", "date", "group", "market", "strategy", "horizon", "top", "bottom", "group_mean",
                                       "killed", "n_top", "n_bottom"])


def _series_stat(x: pd.Series, lag: int) -> Dict[str, Any]:
    v = x.dropna().to_numpy(dtype=float)
    se = nw_se(v, lag) if len(v) >= 3 else None
    return {"n": int(len(v)), "mean": float(v.mean()) if len(v) else None, "se": se,
            "t": (float(v.mean()) / se if se else None), "hit": float((v > 0).mean()) if len(v) else None}


def evaluate_peer_ranks(ranks: pd.DataFrame, price_map: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    c = PEER_EVAL
    gm = group_month_rows(ranks, price_map)
    res: Dict[str, Any] = {"config": {k: (v if not isinstance(v, dict) else dict(v)) for k, v in c.items()},
                           "n_formation": len(formation_dates(ranks)), "first_formation": None, "strategies": {}}
    fds = formation_dates(ranks)
    res["first_formation"] = str(fds[0].date()) if fds else None
    for s, label in c["STRATEGIES"].items():
        out = {"label": label, "horizons": {}}
        for h in c["HORIZONS"]:
            g = gm[(gm["strategy"] == s) & (gm["horizon"] == h)]
            blk = {}
            for scope, sub in (("전체", g), ("미국", g[g["market"] == "미국"]), ("한국", g[g["market"] == "한국"])):
                if not len(sub):
                    blk[scope] = {"ls": _series_stat(pd.Series(dtype=float), 0), "tm": _series_stat(pd.Series(dtype=float), 0), "n_groups": 0}
                    continue
                m = sub.assign(ls=sub["top"] - sub["bottom"], tm=sub["top"] - sub["group_mean"]).groupby("month")[["ls", "tm"]].mean()
                blk[scope] = {"ls": _series_stat(m["ls"], c["NW_LAG"][h]), "tm": _series_stat(m["tm"], c["NW_LAG"][h]),
                              "n_groups": float(sub.groupby("month")["group"].nunique().mean())}
            out["horizons"][h] = blk
        p = out["horizons"][c["PRIMARY"]]["전체"]["ls"]
        enough = p["n"] >= c["MIN_MONTHS"]
        passed = bool(enough and p["mean"] is not None and p["mean"] > 0 and p["t"] is not None and p["t"] >= c["MIN_T"])
        out["verdict"] = {"status": ("통과" if passed else "미통과") if enough else "판정 보류", "passed": passed,
                          "months_done": p["n"], "months_needed": c["MIN_MONTHS"]}
        res["strategies"][s] = out
    k = gm[(gm["horizon"] == c["PRIMARY"]) & gm["killed"].notna() & (gm["strategy"] == "value")]
    km = (k["killed"] - k["group_mean"]).groupby(k["month"]).mean() if len(k) else pd.Series(dtype=float)
    res["killed_vs_group"] = _series_stat(km, c["NW_LAG"][c["PRIMARY"]])
    res["criteria"] = (f"관점마다 {c['PRIMARY']} 결과가 다 나온 달 {c['MIN_MONTHS']}개 이상, (상위 1/3 − 하위 1/3) 평균 > 0, "
                       f"t ≥ {c['MIN_T']:.0f}(Newey-West). 그 전에는 판정 보류")
    return res


def render_peer_eval_report(r: Dict[str, Any], generated_at: Optional[str] = None) -> str:
    c = r["config"]
    pp = lambda v: "—" if v is None else f"{v * 100:+.2f}%p"
    t = lambda v: "—" if v is None else f"{v:+.1f}"
    L = ["# 피어 순위 채점", "",
         f"- 생성: {generated_at or '—'} · 채점 시작일 {c['START']} · 줄 세운 달 {r['n_formation']}개(첫 달 {r['first_formation'] or '없음'})",
         "- 방법(2026-11-17 결과 보기 전 고정, engine/peer_eval.py): 매달 첫 순위 기록일에 묶음마다 상위 1/3 vs 하위 1/3, 가격 기준일 다음 거래일 "
         f"시가에 사서 {c['PRIMARY']}(판정)·12개월(참고) 뒤 종가. 1·3개월은 판정 전 흐름 참고. 같은 비중, 비용 전. 한 달 값 = 묶음 평균.",
         f"- 판정 기준: {r['criteria']}.", ""]
    for s, o in r["strategies"].items():
        v = o["verdict"]
        L += [f"## {o['label']}: {v['status']}" + (f" ({c['PRIMARY']} 결과가 다 나온 달 {v['months_done']}/{v['months_needed']})" if v["status"] == "판정 보류" else ""), "",
              "| 기간 | 범위 | 달 수 | 묶음 수(달 평균) | 상위 − 하위 | t | 상위가 이긴 달 | 상위 − 묶음 평균 | t |", "|---|---|---|---|---|---|---|---|---|"]
        for h, blk in o["horizons"].items():
            for scope, b in blk.items():
                if not b["ls"]["n"]:
                    continue
                tag = " (판정)" if h == c["PRIMARY"] and scope == "전체" else ""
                L.append(f"| {h}{tag} | {scope} | {b['ls']['n']} | {b['n_groups']:.0f} | {pp(b['ls']['mean'])} | {t(b['ls']['t'])} | "
                         f"{'—' if b['ls']['hit'] is None else format(b['ls']['hit'] * 100, '.0f') + '%'} | {pp(b['tm']['mean'])} | {t(b['tm']['t'])} |")
        if not any(b["ls"]["n"] for blk in o["horizons"].values() for b in blk.values()):
            L.append("| — | — | 0 | — | 아직 결과가 나온 달 없음 | — | — | — | — |")
        L.append("")
    k = r["killed_vs_group"]
    L += [f"참고 — 생존성 하드 킬 종목의 {c['PRIMARY']} 수익률 − 같은 묶음 평균: "
          + ("아직 자료 없음" if not k["n"] else f"{pp(k['mean'])} (t {t(k['t'])}, {k['n']}개월)"), "",
          "## 읽는 법", "",
          "- 판정 보류 동안의 숫자(특히 1·3개월)는 흐름 참고일 뿐이다. 보류 중에 결과를 보고 순위 규칙·가중치를 바꾸지 않는다 — 바꾸면 채점을 처음부터 다시 한다.",
          "- 통과는 '그 기간에 순위가 수익을 갈랐다'는 뜻이지 앞으로의 보장이 아니다.", ""]
    return "\n".join(L)
