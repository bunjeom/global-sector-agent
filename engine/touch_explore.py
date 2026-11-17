"""탐색: 수평선을 몇 번째 건드릴 때 뚫리는가 (2026-11-17) — run_backtest.py --explore touch4.

사용자 가설: "지지선(또는 저항선)을 네 번째 테스트(터치)하는 순간은 뚫린다."
탐색용 종목으로 보는 것이라 결과는 증거가 아니다(engine/explore.py 규칙). 정의는 결과를 보기 전에 아래처럼 고정했다.

수평선  스윙 저점이 확정되면 그 가격이 지지선, 스윙 고점이 확정되면 그 가격이 저항선(틀 A 지그재그 3×ATR — 지금 화면의 선과 같은 재료).
        확정 시점에 같은 종류의 살아 있는 선이 1×ATR 안에 있으면 새 선을 만들지 않는다(그 저점·고점은 기존 선의 재테스트).
        선은 뚫리거나 만든 지 250거래일(약 1년)이 지나면 없어진다.
터치    선을 만든 저점·고점 자체가 1번째 터치. 그 뒤 지지선은 저가 ≤ 선 + 0.5×ATR, 저항선은 고가 ≥ 선 − 0.5×ATR인 날이 터치.
        한 번 터치한 뒤에는 종가가 선에서 2×ATR 넘게 멀어졌다가 돌아와야 다음 터치로 센다(붙어서 며칠 머무는 것은 한 번).
뚫림    터치한 날부터 10거래일 안에 종가가 선을 0.5×ATR(터치한 날 값) 넘게 넘어감(지지선은 아래로, 저항선은 위로).
        그렇게 넘어간 선은 없어진다.
판정    4번째 터치의 뚫림 비율이 2·3번째 터치(합침)보다 높은가. 지지선·저항선 합쳐서 본다(각각도 표시).
        후보 = 4번째 터치 사건 ≥ 300, 차이 > 0, z ≥ 2, 앞·뒤 기간 같은 방향. '반드시 뚫린다'인지는 뚫림 비율 자체로 본다.
"""
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .explore import EXPLORE_CONFIG, EXPLORE_HYPOTHESES
from .frame_backtest import compare_clustered
from .playbook import swing_points, trend_indicators
from .rule_backtest import _bt_naive

TOUCH_CONFIG: Dict[str, Any] = {
    "ZONE_ATR": 0.5,        # 터치 = 선에서 0.5×ATR 이내
    "BREAK_ATR": 0.5,       # 뚫림 = 종가가 선을 0.5×ATR 넘게 넘어감 (화면 손절선과 같은 폭)
    "LEAVE_ATR": 2.0,       # 다음 터치로 세려면 종가가 선에서 2×ATR 넘게 멀어졌다가 돌아와야 함
    "MERGE_ATR": 1.0,       # 새 스윙 점이 기존 선에서 1×ATR 안이면 새 선을 만들지 않음
    "LIFE_DAYS": 250,       # 선의 수명(거래일)
    "HORIZON": 10,          # 뚫림을 보는 거래일
    "TARGET_TOUCH": 4,      # 가설: 4번째 터치
    "BASE_TOUCHES": (2, 3), # 비교: 2·3번째 터치
    "MAX_SHOWN": 6,         # 표: 6번째 이상은 한 칸
    # touch4b(2026-11-17 사용자 정정, 결과 보기 전 고정): '뚫림' = 잠깐 이탈이 아니라 넘어간 뒤 20거래일 동안 돌아오지 않음
    "DECISIVE_DAYS": 20,
}


def _touch_outcome(cl: np.ndarray, t: int, lv: float, s: float, a_t: float, H: int, N: int) -> Optional[str]:
    """터치일 t 뒤: 'held'(H일 안에 종가가 선을 BREAK_ATR 넘게 넘어가지 않음) / 'temporary'(넘어간 날 j 뒤 N일 안에 종가가 선 안쪽으로 돌아옴)
    / 'decisive'(j 뒤 N일 동안 종가가 한 번도 선 안쪽으로 돌아오지 않음). 자료가 모자라면 None."""
    n = len(cl)
    if t + H > n - 1:
        return None
    w = cl[t:t + H + 1]
    cross = np.flatnonzero((lv - w) * s > TOUCH_CONFIG["BREAK_ATR"] * a_t)
    if not len(cross):
        return "held"
    j = t + int(cross[0])
    if j + N > n - 1:
        return None
    after = cl[j + 1:j + 1 + N]
    return "decisive" if bool(((lv - after) * s > 0).all()) else "temporary"


def touch_events(df: pd.DataFrame, ind: Optional[pd.DataFrame] = None, decisive_days: Optional[int] = None) -> pd.DataFrame:
    """한 종목의 터치 사건. 열: date, t, kind('S' 지지선·'R' 저항선), n(몇 번째 터치, 선을 만든 점이 1), level, broke(0/1).
    그날까지의 자료로 선·터치를 정하고, broke만 이후 HORIZON일을 본다.
    decisive_days를 주면(touch4b) 열 outcome('held'·'temporary'·'decisive')·decisive(0/1)를 더하고, 선은 0.5×ATR 넘어간 뒤
    decisive_days 동안 종가가 선 안쪽으로 한 번도 돌아오지 않았을 때(그날 알 수 있음)만 없앤다 — 잠깐 이탈한 선은 살아 있다."""
    c = TOUCH_CONFIG
    base = ind if ind is not None else trend_indicators(df)
    hi, lo, cl = (base[k].to_numpy(dtype=float) for k in ("High", "Low", "Close"))
    atr = base["ATR20"].to_numpy(dtype=float)
    n = len(cl)
    by_confirm: Dict[int, List[Dict[str, Any]]] = {}
    for p in swing_points(hi, lo, atr):
        by_confirm.setdefault(int(p["confirm"]), []).append(p)
    alive: List[Dict[str, Any]] = []
    out: List[Dict[str, Any]] = []
    H = c["HORIZON"]
    for t in range(n):
        a_t = atr[t]
        if not np.isfinite(a_t) or a_t <= 0:
            continue
        keep = []
        for L in alive:
            if t - L["born"] > c["LIFE_DAYS"]:
                continue
            s = 1.0 if L["kind"] == "S" else -1.0      # 지지선: 아래로 뚫림, 저항선: 위로 뚫림
            lv = L["price"]
            near = (lo[t] <= lv + c["ZONE_ATR"] * a_t) if L["kind"] == "S" else (hi[t] >= lv - c["ZONE_ATR"] * a_t)
            if L["armed"] and near:
                L["touches"] += 1
                L["armed"] = False
                if decisive_days is None:
                    if t + H <= n - 1:
                        w = cl[t:t + H + 1]
                        broke = bool(((lv - w) * s > c["BREAK_ATR"] * a_t).any())
                        out.append({"date": base.index[t], "t": t, "kind": L["kind"], "n": L["touches"], "level": float(lv), "broke": float(broke)})
                else:
                    oc = _touch_outcome(cl, t, lv, s, a_t, H, int(decisive_days))
                    if oc is not None:
                        out.append({"date": base.index[t], "t": t, "kind": L["kind"], "n": L["touches"], "level": float(lv),
                                    "broke": float(oc != "held"), "outcome": oc, "decisive": float(oc == "decisive")})
            if decisive_days is None:
                if (lv - cl[t]) * s > c["BREAK_ATR"] * a_t:   # 뚫림 → 선 없어짐
                    continue
            else:   # 0.5×ATR 넘어간 뒤 decisive_days 동안 계속 선 바깥이면 그날 없앰, 선 안쪽으로 돌아오면 다시 살아 있는 선
                if L.get("out", -1) < 0:
                    if (lv - cl[t]) * s > c["BREAK_ATR"] * a_t:
                        L["out"] = 0
                elif (lv - cl[t]) * s > 0:
                    L["out"] += 1
                    if L["out"] >= int(decisive_days):
                        continue
                else:
                    L["out"] = -1
            if L.get("out", -1) >= 0:   # 선 바깥에 있는 동안은 다음 터치를 준비하지 않음
                keep.append(L)
                continue
            if not L["armed"] and (cl[t] - lv) * s > c["LEAVE_ATR"] * a_t:
                L["armed"] = True
            keep.append(L)
        alive = keep
        for p in by_confirm.get(t, []):
            kind = "S" if p["kind"] == "L" else "R"
            if any(L["kind"] == kind and abs(L["price"] - p["price"]) <= c["MERGE_ATR"] * a_t for L in alive):
                continue
            # 확정 = 그 점에서 3×ATR 되돌린 날이라 이미 선에서 멀어져 있다 → 다음 접근이 2번째 터치
            alive.append({"kind": kind, "price": float(p["price"]), "born": t, "touches": 1, "armed": True})
    cols = ["date", "t", "kind", "n", "level", "broke"] + ([] if decisive_days is None else ["outcome", "decisive"])
    return pd.DataFrame(out, columns=cols)


def _touch_block(ev: pd.DataFrame, col: str = "broke") -> Dict[str, Any]:
    c = TOUCH_CONFIG
    rows = {}
    for k in range(2, c["MAX_SHOWN"] + 1):
        g = ev[ev["n"] == k] if k < c["MAX_SHOWN"] else ev[ev["n"] >= k]
        row = {"n": int(len(g)), "break_rate": float(g[col].mean()) if len(g) else None}
        if "outcome" in ev:
            for oc in ("held", "temporary", "decisive"):
                row[oc] = float((g["outcome"] == oc).mean()) if len(g) else None
        rows[f"{k}번째" + (" 이상" if k == c["MAX_SHOWN"] else "")] = row
    tgt = ev[ev["n"] == c["TARGET_TOUCH"]]
    basev = ev[ev["n"].isin(c["BASE_TOUCHES"])]
    later = ev[ev["n"] >= c["TARGET_TOUCH"]]
    return {"by_touch": rows, "target_vs_base": compare_clustered(tgt, basev, col), "later_vs_base": compare_clustered(later, basev, col)}


def run_touch_explore(price_map: Dict[str, pd.DataFrame], decisive: bool = False) -> Dict[str, Any]:
    ec, c = EXPLORE_CONFIG, TOUCH_CONFIG
    parts, used = [], 0
    s0 = pd.Timestamp(ec["START"])
    for sym, df in (price_map or {}).items():
        d = _bt_naive(df)
        if d is not None and len(d):
            cols = [k for k in ("Open", "High", "Low", "Close") if k in d]
            d = d[(d[cols].astype(float) > 0).all(axis=1)]
        if d is None or len(d) < ec["MIN_BARS"]:
            continue
        used += 1
        ev = touch_events(d, decisive_days=c["DECISIVE_DAYS"] if decisive else None)
        ev = ev[ev["date"] >= s0]
        if len(ev):
            parts.append(ev.assign(ticker=sym, market="KR" if str(sym).upper().endswith((".KS", ".KQ")) else "US"))
    ev = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=["date", "t", "kind", "n", "level", "broke", "ticker", "market"] + (["outcome", "decisive"] if decisive else []))
    col = "decisive" if decisive else "broke"
    split_t = pd.Timestamp(ec["SPLIT_DATE"])
    sel = {"전체 기간": lambda x: x, "앞 기간": lambda x: x[x["date"] < split_t], "뒤 기간": lambda x: x[x["date"] >= split_t],
           "전체 기간 · 미국": lambda x: x[x["market"] == "US"], "전체 기간 · 한국": lambda x: x[x["market"] == "KR"],
           "전체 기간 · 지지선만": lambda x: x[x["kind"] == "S"], "전체 기간 · 저항선만": lambda x: x[x["kind"] == "R"]}
    blocks = {p: _touch_block(f(ev), col) for p, f in sel.items()}
    full, fr, bk = (blocks[p]["target_vs_base"] for p in ("전체 기간", "앞 기간", "뒤 기간"))
    n4 = full["actual"].get("n", 0)
    diff, z = full.get("diff"), full.get("z")
    same = bool(diff is not None and fr.get("diff") is not None and bk.get("diff") is not None
                and fr["diff"] > 0 and bk["diff"] > 0 and diff > 0)
    cand = bool(n4 >= ec["MIN_EVENTS"] and z is not None and z >= ec["CANDIDATE_Z"] and same)
    reasons = []
    if n4 < ec["MIN_EVENTS"]:
        reasons.append(f"4번째 터치 사건 {n4:,}건 < {ec['MIN_EVENTS']}")
    if diff is not None and diff <= 0:
        reasons.append("4번째 터치가 2·3번째보다 더 " + ("확정 뚫리지" if decisive else "뚫리지") + " 않음")
    elif z is None or z < ec["CANDIDATE_Z"]:
        reasons.append(f"z < {ec['CANDIDATE_Z']:.0f}")
    if not same and (diff is None or diff > 0):
        reasons.append("앞·뒤 기간 방향이 다름")
    name = "touch4b" if decisive else "touch4"
    label = EXPLORE_HYPOTHESES[name]["label"]
    return {"name": name, "label": label, "decisive": bool(decisive), "n_tickers": used, "n_events": int(n4), "n_touch_events": int(len(ev)),
            "blocks_touch": blocks, "candidate": cand, "not_candidate_reasons": reasons,
            "blocks": {"전체 기간": {"vs_random": full}},   # 탐색 기록(EXPLORE_LOG.md) 한 줄용: 2·3번째 터치 대비 뚫림 비율 차이
            "config": {**{k: c[k] for k in c}, "START": ec["START"], "SPLIT_DATE": ec["SPLIT_DATE"]},
            "criteria": (f"후보 = 4번째 터치 사건 {ec['MIN_EVENTS']}건 이상, (4번째 터치 {'확정 ' if decisive else ''}뚫림 비율 − 2·3번째 터치 "
                         f"{'확정 ' if decisive else ''}뚫림 비율) > 0, z ≥ {ec['CANDIDATE_Z']:.0f}, 앞·뒤 기간 모두 차이 > 0")}


def render_touch_report(r: Dict[str, Any], generated_at: Optional[str] = None) -> str:
    c = r["config"]
    pct = lambda v: "—" if v is None else f"{v * 100:.1f}%"

    def cmp_txt(x: Dict[str, Any]) -> str:
        if x.get("diff") is None:
            return "—"
        return f"{x['diff'] * 100:+.1f}%p · z {'—' if x.get('z') is None else format(x['z'], '+.1f')}"

    dec = r.get("decisive")
    L = [f"# 탐색: {r['label']}", "",
         "**탐색 결과 — 증거 아님.** 이미 판정에 쓴 종목으로 본 것이라, 좋게 나와도 '확인해 볼 만하다'는 뜻일 뿐이다.", "",
         f"- 생성: {generated_at or '—'} · 가설 이름 `{r['name']}` · 종목 {r['n_tickers']}개 · {c['START'][:4]}년~ · 터치 사건 {r['n_touch_events']:,}건",
         "- 수평선: 스윙 저점 = 지지선, 스윙 고점 = 저항선(지그재그 3×ATR). 새 스윙 점이 기존 선에서 "
         f"{c['MERGE_ATR']:g}×ATR 안이면 새 선을 만들지 않음. 수명 {c['LIFE_DAYS']}거래일.",
         f"- 터치: 선을 만든 점이 1번째. 그 뒤 선에서 {c['ZONE_ATR']}×ATR 이내로 온 날. 다음 터치로 세려면 종가가 선에서 "
         f"{c['LEAVE_ATR']:g}×ATR 넘게 멀어졌다가 돌아와야 함.",
         (f"- 세 갈래(touch4 정의 정정, 사용자 확인 2026-11-17): 버팀 = 터치 뒤 {c['HORIZON']}거래일 동안 종가가 선을 {c['BREAK_ATR']}×ATR 넘게 넘어가지 않음 · "
          f"일시 이탈 = 넘어갔지만 넘어간 날부터 {c['DECISIVE_DAYS']}거래일 안에 종가가 선 안쪽으로 돌아옴 · 확정 뚫림 = {c['DECISIVE_DAYS']}거래일 동안 "
          "한 번도 돌아오지 않음. 선은 확정 뚫림일 때만 없어지고, 일시 이탈한 선은 살아 있어 다음 터치를 센다. 판정은 확정 뚫림 비율로."
          if dec else f"- 뚫림: 터치한 날부터 {c['HORIZON']}거래일 안에 종가가 선을 {c['BREAK_ATR']}×ATR 넘게 넘어감(지지선은 아래로, 저항선은 위로)."),
         f"- {r['criteria']}. 표준오차는 월 묶음.", "",
         f"## 후보 여부: {'후보 — 확인 시험으로 올릴 만함' if r['candidate'] else '후보 아님'}"
         + ("" if r["candidate"] else f" ({', '.join(r['not_candidate_reasons'])})"), ""]
    names = list(next(iter(r["blocks_touch"].values()))["by_touch"].keys())
    if dec:
        full = r["blocks_touch"]["전체 기간"]["by_touch"]
        L += ["**몇 번째 터치에서 어떻게 됐나 — 전체 기간 (버팀 · 일시 이탈 · 확정 뚫림)**", "",
              "| 터치 | 버팀 | 일시 이탈 | 확정 뚫림 | 사건 수 |", "|---|---|---|---|---|"]
        for k in names:
            x = full[k]
            L.append(f"| {k} | {pct(x.get('held'))} | {pct(x.get('temporary'))} | {pct(x.get('decisive'))} | {x['n']:,} |")
        L += [""]
    L += [f"**몇 번째 터치에서 {'확정 ' if dec else ''}뚫렸나 — {'확정 뚫림' if dec else str(c['HORIZON']) + '거래일 안에 뚫린'} 비율 (사건 수)**", "",
          "| 구분 | " + " | ".join(names) + " | 4번째 − 2·3번째 | 4번째 이상 − 2·3번째 |", "|---|" + "---|" * (len(names) + 2)]
    for per, b in r["blocks_touch"].items():
        cells = [f"{pct(b['by_touch'][k]['break_rate'])} ({b['by_touch'][k]['n']:,})" for k in names]
        L.append(f"| {per} | " + " | ".join(cells) + f" | {cmp_txt(b['target_vs_base'])} | {cmp_txt(b['later_vs_base'])} |")
    L += ["", "## 읽는 법", "",
          f"- '4번째 터치는 뚫린다'가 맞으려면 4번째 칸의 {'확정 뚫림 ' if dec else ''}비율이 100%에 가깝고, 2·3번째 칸보다 뚜렷이 높아야 한다.",
          "- 4번째까지 온 선은 2·3번째 터치에서 살아남은 선이다. 그래도 표의 비율은 '그 터치가 왔을 때 뚫릴 확률' 그대로다.",
          "- 후보가 아니면 여기서 끝낸다. 숫자(0.5·2×ATR, 250일, 10일)를 바꿔 가며 다시 보지 않는다.", ""]
    return "\n".join(L)


EXPLORE_HYPOTHESES["touch4b"] = {
    "label": "수평선(지지·저항)을 4번째 터치할 때 확정 뚫림(넘어간 뒤 20거래일 동안 돌아오지 않음)이 되는가 — 2·3번째 터치 대비",
    "expect": 1, "custom": "touch_b",
    "note": "touch4의 '뚫림' 정의 정정(사용자 확인 2026-11-17, 같은 가설 두 번째 탐색). 기록의 '차이'는 2·3번째 터치 대비 확정 뚫림 비율 차이",
}
EXPLORE_HYPOTHESES["touch4"] = {
    "label": "수평선(지지·저항)을 4번째 터치할 때 뚫리는가 — 2·3번째 터치 대비 뚫림 비율", "expect": 1, "custom": "touch",
    "note": "사용자 가설(2026-11-17). 기록의 '차이'는 2·3번째 터치 대비 뚫림 비율 차이",
}
