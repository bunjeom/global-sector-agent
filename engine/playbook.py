"""매매 틀 v2 판정 (2026-11-14 추가) — AGENTS.md 4번 '틀 A v2'·'틀 B v2'를 코드로 옮긴 것.

틀 A v2 — 추세 판단과 전환 가능성 (트레이딩, 의견)
    추세 상태   스윙 고점·저점(직전 극점에서 3×ATR 이상 되돌리면 확정)으로 판단. 상승 = 저점이 높아지고 두 저점을 잇는
                상승 추세선 위 / 하락 = 상승이 아니고 고점이 낮아지며 두 고점을 잇는 하락 추세선 아래 / 횡보 = 그 외
    근거 저울   항목마다 전환 쪽 +1 · 실패 쪽 −1 · 중립 0. 같은 사실을 양쪽에 두 번 세지 않는다
                기술 4개(50일선·20일선 기울기·스윙 저점·3개월 상대강도), 실적(30일 추정치 상향·하향 건수),
                가치(자기 PER 밴드 위치), 돌발 이벤트(최근 5거래일 −5% 갭 또는 하루 2.5×ATR 급락 → −2)
                국면 게이트는 저울에 넣지 않고 크기만 줄인다(검증: 수익을 예측하지 못하는 위험 온도계)
    표시        차이(전환 − 실패) +2 이상 '우세', −1~+1 '팽팽', −2 이하 '실패 우세'. 확률(%)로 표시하지 않는다
    진입 의견   하락·횡보 상태 + 차이 +2 이상 + 종가가 50일선 위. 추격 주의: 종가 > 20일선 + 2×ATR
    청산        실패 합 > 전환 합이면 다음 날 시가(손실이면 손절, 이익이면 매도). 가격 폭 손절 없음
    크기        변동성 단위 2×ATR(20). 차이 +3 이상 1배, +2 절반. 게이트 단계로 추가 축소
틀 B v2 — 저평가 매수 (투자 판단): 가치(V)·실적·종합 판정 결과를 받아 단계(진입 가능·조건부·대기·경고·손절·매도·재평가 보유)를 정한다.
모든 판정에 검증 상태를 붙인다. 숫자는 관행값으로 고정했고 과거 자료로 맞추지 않았다(미검증).
"""
import math
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .fundamental import FUND_DATA_PARAMS, _annual_pairs, _band_unreliable_reasons, _lagged_ratio_stats, _month_end_prices
from .rule_backtest import BACKTEST_CONFIG, _bt_naive

# 판정 규칙 버전. 규칙(숫자 포함)을 바꾸면 반드시 올린다 — 신호 기록 장부(engine/ledger.py)가 줄마다 저장해 버전끼리 섞지 않는다
PLAYBOOK_VERSION = "A v2.1 (2026-11-15 스윙 추세)"
PLAYBOOK_CONFIG: Dict[str, Any] = {
    "TREND_MA": 200, "MID_MA": 50, "SHORT_MA": 20,
    "MID_SLOPE_DAYS": 20,           # 50일선 기울기: 20거래일 전과 비교
    "SHORT_SLOPE_DAYS": 5,          # 20일선 기울기: 5거래일 전과 비교
    "SWING_ATR": 3.0,               # 스윙 확정: 직전 극점에서 3×ATR(20) 이상 되돌리면 그 극점을 스윙 고점·저점으로 확정
    "RS_DAYS": 63,                  # 3개월 상대강도
    "ATR_DAYS": 20,
    "EVENT_DAYS": 5, "EVENT_GAP": -0.05, "EVENT_DROP_ATR": 2.5, "EVENT_WEIGHT": 2,
    "BAND_CHEAP_Z": -1.0, "BAND_RICH_Z": 1.0,
    "LEAN_DIFF": 2,                 # 차이 ±2 이상이면 한쪽 우세
    "FULL_SIZE_DIFF": 3,            # 차이 +3 이상이면 크기 1배, +2는 절반
    "CHASE_ATR": 2.0, "SIZE_ATR": 2.0,
    "GATE_SIZE": dict(BACKTEST_CONFIG["T_SIZE_BY_GATE"]),   # 앱 T 크기 규칙과 같음 (G0 1배·G1 절반·G2·G3 0)
    "BENCH": {"US": "SPY", "KR": "^KS11"},
}
VERIFICATION: Dict[str, str] = {
    "frame_a": "미검증 · 의견 — 가격 기반 타이밍 규칙은 과거 검증 6번 모두 우위가 확인되지 않음(AGENTS.md 4번). 신호 기록 장부로 앞으로 검증",
    "frame_b": "미검증 — 발표 실적 기준 v1은 우위 미확인(FRAME_B.md). v2의 추정치 기준 판정은 2026-11-14부터 쌓는 기록으로 검증",
    "frame_b_sell": "미검증 (v1 참고 결과 양호: 다른 종목 대비 +4~+10%p, 표본 작음)",
    "peer_scope": "미검증 — 자기 밴드를 쓸 수 없는 종목의 피어 기준 판정",
}
STATE_UP, STATE_DOWN, STATE_SIDE = "상승", "하락", "횡보"


# ---------------------------------------------------------------------
# 틀 A v2 — 지표·추세 상태
# ---------------------------------------------------------------------
def _pb_market(sym: str) -> str:
    return "KR" if str(sym).upper().endswith((".KS", ".KQ")) else "US"


def trend_indicators(df: pd.DataFrame, bench: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """그날 종가까지의 자료만 쓰는 일별 지표와 추세 상태. 시장 종가는 종목 날짜에 맞춰 앞 값으로 채운다."""
    c = PLAYBOOK_CONFIG
    d = _bt_naive(df)
    close, low = d["Close"].astype(float), d["Low"].astype(float)
    op = d["Open"].astype(float) if "Open" in d else close
    hi = d["High"].astype(float) if "High" in d else close
    out = pd.DataFrame(index=d.index)
    out["Open"], out["High"], out["Low"], out["Close"] = op, hi, low, close
    for k, n in (("SMA200", c["TREND_MA"]), ("SMA50", c["MID_MA"]), ("SMA20", c["SHORT_MA"])):
        out[k] = close.rolling(n, min_periods=n).mean()
    out["slope50"] = out["SMA50"] - out["SMA50"].shift(c["MID_SLOPE_DAYS"])
    out["slope20"] = out["SMA20"] - out["SMA20"].shift(c["SHORT_SLOPE_DAYS"])
    tr = pd.concat([hi - low, (hi - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    out["ATR20"] = tr.ewm(alpha=1 / c["ATR_DAYS"], adjust=False, min_periods=c["ATR_DAYS"]).mean()
    ret = close / close.shift(c["RS_DAYS"]) - 1.0
    b = _bt_naive(bench) if bench is not None and len(bench) else None
    if b is not None:
        bc = b["Close"].astype(float)
        bc = bc.reindex(d.index.union(bc.index)).ffill().reindex(d.index)
        out["rs63"] = ret - (bc / bc.shift(c["RS_DAYS"]) - 1.0)
    else:
        out["rs63"] = np.nan
    prev = close.shift()
    gap = op / prev - 1.0
    drop_atr = (prev - close) / out["ATR20"].shift()
    ev = ((gap <= c["EVENT_GAP"]) | (drop_atr >= c["EVENT_DROP_ATR"])).astype(float)
    out["event"] = ev.rolling(c["EVENT_DAYS"], min_periods=1).max().astype(bool)
    # 참고용 이동평균 배열 상태(스윙이 모자랄 때 대신 씀)
    up = (close > out["SMA50"]) & (out["SMA50"] > out["SMA200"]) & (out["slope50"] > 0)
    dn = (close < out["SMA50"]) & (out["SMA50"] < out["SMA200"]) & (out["slope50"] < 0)
    ready = out[["SMA200", "slope50"]].notna().all(axis=1)
    # object 열로 둔다: pandas 3은 문자열 열의 None을 NaN으로 바꾸므로 '판정 없음'을 None으로 유지하려면 필요
    out["ma_state"] = pd.Series(np.where(~ready, None, np.where(up, STATE_UP, np.where(dn, STATE_DOWN, STATE_SIDE))),
                                index=out.index, dtype=object)
    _swing_state(out, hi.to_numpy(dtype=float), low.to_numpy(dtype=float))
    return out


def swing_points(high: np.ndarray, low: np.ndarray, atr: np.ndarray, k_atr: Optional[float] = None) -> List[Dict[str, Any]]:
    """움직임의 크기로 스윙 고점·저점을 찾는다(지그재그). 상승 구간에서 최고가를 따라가다가 그 최고가보다
    k×ATR(그날 값) 이상 낮은 저가가 나오면 그 최고가를 스윙 고점으로 확정한다(저점도 반대로).
    확정일(confirm)은 되돌림이 나온 날이라, 그 전 날짜의 판정에는 쓰이지 않는다."""
    k = PLAYBOOK_CONFIG["SWING_ATR"] if k_atr is None else float(k_atr)
    n = len(high)
    out: List[Dict[str, Any]] = []
    start = next((i for i in range(n) if np.isfinite(atr[i])), None)
    if start is None:
        return out
    d = None
    hi_i, hi_v, lo_i, lo_v = start, high[start], start, low[start]
    for j in range(start + 1, n):
        thr = k * atr[j]
        if not np.isfinite(thr):
            continue
        if d is None:
            if high[j] > hi_v:
                hi_i, hi_v = j, high[j]
            if low[j] < lo_v:
                lo_i, lo_v = j, low[j]
            if high[j] - lo_v >= thr and lo_i < j:
                out.append({"kind": "L", "idx": lo_i, "price": float(lo_v), "confirm": j})
                d, hi_i, hi_v = "up", j, high[j]
            elif hi_v - low[j] >= thr and hi_i < j:
                out.append({"kind": "H", "idx": hi_i, "price": float(hi_v), "confirm": j})
                d, lo_i, lo_v = "down", j, low[j]
        elif d == "up":
            if high[j] > hi_v:
                hi_i, hi_v = j, high[j]
            elif hi_v - low[j] >= thr:
                out.append({"kind": "H", "idx": hi_i, "price": float(hi_v), "confirm": j})
                d, lo_i, lo_v = "down", j, low[j]
        else:
            if low[j] < lo_v:
                lo_i, lo_v = j, low[j]
            elif high[j] - lo_v >= thr:
                out.append({"kind": "L", "idx": lo_i, "price": float(lo_v), "confirm": j})
                d, hi_i, hi_v = "up", j, high[j]
    return out


def _isna(x: Any) -> bool:
    """None 또는 NaN (pandas 3은 문자열 열의 빈 값을 NaN으로 준다)."""
    return x is None or (isinstance(x, float) and math.isnan(x))


def _line(a: Dict[str, Any], b: Dict[str, Any], t: int) -> float:
    """두 스윙 점을 잇는 추세선(로그 가격)의 t일 값."""
    la, lb = math.log(a["price"]), math.log(b["price"])
    return math.exp(lb + (lb - la) / (b["idx"] - a["idx"]) * (t - b["idx"]))


def _swing_state(out: pd.DataFrame, high: np.ndarray, low: np.ndarray) -> None:
    """날마다 그날까지 확정된 스윙만으로 추세 상태를 정한다(사용자 정의, AGENTS.md 4번).
    상승 = 최근 두 스윙 저점이 높아짐 + 종가가 두 저점을 잇는 상승 추세선 위
    하락 = 상승이 아님(추세선 이탈 또는 저점이 안 높아짐) + 최근 두 스윙 고점이 낮아짐 + 종가가 두 고점을 잇는 하락 추세선 아래
    횡보 = 그 외. 스윙이 2개씩 안 되면 이동평균 배열 상태(ma_state)를 대신 쓰고 표시한다."""
    close = out["Close"].to_numpy(dtype=float)
    pts = swing_points(high, low, out["ATR20"].to_numpy(dtype=float))
    n = len(close)
    states, details, la, lb, ha, hb, lline, hline = ([None] * n for _ in range(8))
    lows: List[Dict[str, Any]] = []
    highs: List[Dict[str, Any]] = []
    p = 0
    ma = out["ma_state"].to_numpy()
    for t in range(n):
        while p < len(pts) and pts[p]["confirm"] <= t:
            (lows if pts[p]["kind"] == "L" else highs).append(pts[p])
            p += 1
        if len(lows) < 2 or len(highs) < 2:
            m = None if _isna(ma[t]) else ma[t]
            states[t] = m
            details[t] = None if m is None else f"{m} (스윙 부족 — 이동평균 배열 기준)"
            continue
        L1, L2, H1, H2 = lows[-2], lows[-1], highs[-2], highs[-1]
        lo_line, hi_line = _line(L1, L2, t), _line(H1, H2, t)
        rising, falling = L2["price"] > L1["price"], H2["price"] < H1["price"]
        up = rising and close[t] >= lo_line
        down = (not up) and falling and close[t] <= hi_line
        if up:
            st, dt = STATE_UP, "상승 (저점이 높아지고 상승 추세선 위)" + (" · 고점은 낮아지는 중(수렴)" if falling else "")
        elif down:
            st, dt = STATE_DOWN, "하락 (" + ("상승 추세선 이탈 뒤 " if rising else "저점이 낮아지고 ") + "고점이 낮아지며 하락 추세선 아래)"
        else:
            st = STATE_SIDE
            if rising and close[t] < lo_line and not falling:
                dt = "횡보 (상승 추세선 이탈 — 고점 낮아짐은 아직 확인 안 됨)"
            elif falling and close[t] > hi_line:
                dt = "횡보 (하락 추세선 돌파 시도 — 저점 높아짐은 아직 확인 안 됨)"
            else:
                dt = "횡보 (저점·고점 방향이 엇갈림)"
        states[t], details[t] = st, dt
        la[t], lb[t], ha[t], hb[t], lline[t], hline[t] = L1, L2, H1, H2, lo_line, hi_line
    out["state"] = pd.Series(states, index=out.index, dtype=object)
    out["state_detail"] = pd.Series(details, index=out.index, dtype=object)
    out["sw_L1"], out["sw_L2"], out["sw_H1"], out["sw_H2"] = la, lb, ha, hb
    out["line_low"], out["line_high"] = lline, hline


def state_basis(row: pd.Series, ind: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """추세 상태를 정한 스윙 고점·저점과 연결선 값, 아직 확정되지 않은 진행 중 극점을 사람이 확인할 수 있게 돌려준다."""
    index = ind.index if ind is not None else None
    def pt(x):
        if not isinstance(x, dict):
            return "—"
        d = str(index[x["idx"]].date()) if index is not None else f"#{x['idx']}"
        return f"{d} {_fmt(x['price'], 0)}"
    L1, L2, H1, H2 = (row.get(k) for k in ("sw_L1", "sw_L2", "sw_H1", "sw_H2"))
    close = float(row["Close"])
    conds = []
    if isinstance(L2, dict):
        conds.append({"name": "스윙 저점", "text": f"{pt(L1)} → {pt(L2)} ({'높아짐' if L2['price'] > L1['price'] else '낮아짐'})"})
        conds.append({"name": "저점 연결선 오늘 값", "text": f"{_fmt(row.get('line_low'), 0)} · 종가 {_fmt(close, 0)} "
                      f"({'위' if close >= float(row.get('line_low')) else '아래'})"})
    if isinstance(H2, dict):
        conds.append({"name": "스윙 고점", "text": f"{pt(H1)} → {pt(H2)} ({'낮아짐' if H2['price'] < H1['price'] else '높아짐'})"})
        conds.append({"name": "고점 연결선 오늘 값", "text": f"{_fmt(row.get('line_high'), 0)} · 종가 {_fmt(close, 0)} "
                      f"({'아래' if close <= float(row.get('line_high')) else '위'})"})
    if ind is not None and isinstance(L2, dict) and isinstance(H2, dict):
        k0 = max(L2["confirm"], H2["confirm"])
        w = ind.iloc[k0:]
        if len(w):
            lo_d, hi_d = w["Low"].idxmin(), w["Close"].idxmax() if "High" not in w else w["High"].idxmax()
            conds.append({"name": "확정 전 진행 중", "text": f"마지막 스윙 확정({index[k0].date()}) 뒤 최저 {_fmt(float(w['Low'].min()), 0)}"
                          f"({lo_d.date()}) · 최고 {_fmt(float(w['High'].max() if 'High' in w else w['Close'].max()), 0)}({hi_d.date()}) "
                          f"— {PLAYBOOK_CONFIG['SWING_ATR']:.0f}×ATR 되돌림이 나와야 스윙으로 확정"})
    s50, s200 = row.get("SMA50"), row.get("SMA200")
    ma_txt = f"참고: 50일선 {_fmt(s50, 0)} · 200일선 {_fmt(s200, 0)} (이동평균 배열 {row.get('ma_state') or '—'})"
    return {"conditions": conds, "why": row.get("state_detail") or "—", "detail": row.get("state_detail"), "ma_note": ma_txt}


def _item(key: str, label: str, track: str, sign: Optional[int], text: str, weight: int = 1) -> Dict[str, Any]:
    """sign: +1 전환 쪽, −1 실패 쪽, 0 중립, None 데이터 없음(중립으로 계산)."""
    return {"key": key, "label": label, "track": track, "sign": sign, "weight": weight,
            "points": 0 if sign is None else int(sign) * weight, "text": text, "no_data": sign is None}


def _sgn(x: Optional[float]) -> Optional[int]:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return None
    return 1 if x > 0 else (-1 if x < 0 else 0)


def _fmt(v: Optional[float], d: int = 2) -> str:
    return "—" if v is None or (isinstance(v, float) and not math.isfinite(v)) else f"{v:,.{d}f}"


def consensus_item(consensus: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """30일간 FY1 EPS 추정치 상향 건수 vs 하향 건수 (snapshot.fetch_consensus_row 모양의 dict)."""
    cs = consensus or {}
    up, dn = cs.get("eps_fy1_up_30d"), cs.get("eps_fy1_down_30d")
    try:
        up, dn = (None if up is None else float(up)), (None if dn is None else float(dn))
    except (TypeError, ValueError):
        up = dn = None
    if up is None or dn is None or not (math.isfinite(up) and math.isfinite(dn)):
        return _item("estimates", "이익 추정치", "실적", None, "30일 추정치 수정 건수 데이터 없음")
    s = 1 if up > dn else (-1 if dn > up else 0)
    word = {1: "상향 우세", -1: "하향 우세", 0: "상향·하향 같음"}[s]
    return _item("estimates", "이익 추정치", "실적", s, f"30일간 FY1 추정치 상향 {up:.0f}건 · 하향 {dn:.0f}건 ({word})")


def band_item(band_z: Optional[float]) -> Dict[str, Any]:
    c = PLAYBOOK_CONFIG
    if band_z is None or (isinstance(band_z, float) and not math.isfinite(band_z)):
        return _item("band", "가격 수준", "가치", None, "자기 PER 밴드 판정 불가(이력 부족 또는 이익 규모 급변)")
    z = float(band_z)
    s = 1 if z <= c["BAND_CHEAP_Z"] else (-1 if z >= c["BAND_RICH_Z"] else 0)
    word = {1: "싼 상태에서 출발", -1: "이미 비쌈", 0: "밴드 안"}[s]
    return _item("band", "가격 수준", "가치", s, f"자기 5년 PER 밴드 z {z:+.2f} ({word})")


def _lows_item(row: pd.Series) -> Dict[str, Any]:
    L1, L2 = row.get("sw_L1"), row.get("sw_L2")
    if not (isinstance(L1, dict) and isinstance(L2, dict)):
        return _item("lows", "스윙 저점", "기술", None, "확정된 스윙 저점이 2개 미만")
    s = _sgn(L2["price"] - L1["price"])
    return _item("lows", "스윙 저점", "기술", s, f"최근 두 스윙 저점 {_fmt(L1['price'])} → {_fmt(L2['price'])} "
                 f"({'높아짐' if s == 1 else ('낮아짐' if s == -1 else '같음')})")


def frame_a_items(row: pd.Series, consensus: Optional[Dict[str, Any]] = None, band_z: Optional[float] = None) -> List[Dict[str, Any]]:
    c = PLAYBOOK_CONFIG
    close, s50 = row.get("Close"), row.get("SMA50")
    it = [
        _item("ma50", "50일선", "기술", None if pd.isna(s50) else _sgn(close - s50),
              f"종가 {_fmt(close)} vs 50일선 {_fmt(s50)}"),
        _item("slope20", "20일선 기울기", "기술", _sgn(row.get("slope20")),
              f"20일선이 {c['SHORT_SLOPE_DAYS']}거래일 전보다 {_fmt(row.get('slope20'))}"),
        _lows_item(row),
        _item("rs", "상대강도", "기술", _sgn(row.get("rs63")),
              "3개월 시장 대비 " + ("데이터 없음" if pd.isna(row.get("rs63")) else f"{float(row.get('rs63')) * 100:+.1f}%p")),
        consensus_item(consensus),
        band_item(band_z),
    ]
    evt = bool(row.get("event")) if not pd.isna(row.get("event")) else False
    it.append(_item("event", "돌발 이벤트", "기술", -1 if evt else 0,
                    f"최근 {c['EVENT_DAYS']}거래일 {c['EVENT_GAP'] * 100:.0f}% 이상 갭 하락 또는 {c['EVENT_DROP_ATR']}×ATR 급락 "
                    + ("있음" if evt else "없음"), weight=c["EVENT_WEIGHT"]))
    return it


def _lean(state: Optional[str], diff: int) -> Dict[str, str]:
    th = PLAYBOOK_CONFIG["LEAN_DIFF"]
    if state == STATE_UP:   # 상승 추세: 같은 저울을 뒤집어 '하락 전환 가능성'
        if diff <= -th:
            return {"direction": "하락 전환 가능성", "label": "하락 전환 가능성 우세"}
        if diff >= th:
            return {"direction": "하락 전환 가능성", "label": "상승 추세 유지 우세"}
        return {"direction": "하락 전환 가능성", "label": "팽팽"}
    if diff >= th:
        return {"direction": "상승 전환 가능성", "label": "전환 가능성 우세"}
    if diff <= -th:
        return {"direction": "상승 전환 가능성", "label": "실패 가능성 우세"}
    return {"direction": "상승 전환 가능성", "label": "팽팽"}


def frame_a_evaluate(df: pd.DataFrame, bench: Optional[pd.DataFrame] = None, consensus: Optional[Dict[str, Any]] = None,
                     band_z: Optional[float] = None, gate_level: Optional[int] = None, at: Any = None,
                     position: Optional[Dict[str, Any]] = None, ind: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """틀 A v2 판정. at을 주면 그날 종가 기준(그날 이후 자료는 쓰지 않음). position={"entry_price": ..}이면 청산 여부도 판정."""
    c = PLAYBOOK_CONFIG
    ind = ind if ind is not None else trend_indicators(df, bench)
    if ind is None or not len(ind):
        return {"frame": "A v2", "available": False, "reason": "가격 데이터 없음", "verification": VERIFICATION["frame_a"]}
    if at is not None:
        ind = ind.loc[:pd.Timestamp(at)]
    row = ind.iloc[-1]
    state = row.get("state")
    if _isna(state):
        return {"frame": "A v2", "available": False, "as_of": str(ind.index[-1].date()),
                "reason": "가격 이력 부족(스윙 고점·저점이 2개씩 확정되지 않았고 200일선도 계산할 수 없음)", "verification": VERIFICATION["frame_a"]}
    items = frame_a_items(row, consensus, band_z)
    support = sum(i["points"] for i in items if i["points"] > 0)
    against = -sum(i["points"] for i in items if i["points"] < 0)
    diff = support - against
    lean = _lean(state, diff)
    close, atr = float(row["Close"]), float(row["ATR20"]) if pd.notna(row["ATR20"]) else None
    ma50_up = next(i for i in items if i["key"] == "ma50")["sign"] == 1
    entry = bool(state in (STATE_DOWN, STATE_SIDE) and diff >= c["LEAN_DIFF"] and ma50_up)
    size_mult = (1.0 if diff >= c["FULL_SIZE_DIFF"] else 0.5) if entry else 0.0
    gate_mult = None if gate_level is None else float(c["GATE_SIZE"].get(int(gate_level), 0.0))
    chase = bool(atr is not None and pd.notna(row["SMA20"]) and close > float(row["SMA20"]) + c["CHASE_ATR"] * atr)
    notes = []
    if entry and gate_mult is not None and gate_mult == 0:
        notes.append(f"국면 G{int(gate_level)}: 신규 진입 크기 0 (의견만 표시)")
    if chase and entry:
        notes.append("추격 주의: 종가가 20일선 + 2×ATR 위")
    if any(i["no_data"] for i in items):
        notes.append("데이터 없음 항목은 중립(0)으로 계산: " + ", ".join(i["label"] for i in items if i["no_data"]))
    opinion = {
        "entry": entry,
        "label": ("상승 전환 의견" + (" (강)" if diff >= c["FULL_SIZE_DIFF"] else "")) if entry else "진입 의견 없음",
        "size_mult": size_mult, "gate_mult": gate_mult,
        "final_mult": None if gate_mult is None else size_mult * gate_mult,
        "unit_risk": None if atr is None else c["SIZE_ATR"] * atr,
        "unit_risk_pct": None if atr is None else c["SIZE_ATR"] * atr / close,
        "chase_caution": chase, "notes": notes,
    }
    pos = None
    if position and position.get("entry_price"):
        ep = float(position["entry_price"])
        exit_now = against > support
        pnl = close / ep - 1.0
        pos = {"entry_price": ep, "pnl": pnl, "exit": exit_now,
               "label": ("손절 (전환 실패 우세)" if pnl < 0 else "매도 (하락 전환 우세)") if exit_now else "보유",
               "rule": "하락 쪽 근거 합(실패·하락 전환)이 상승 쪽 근거 합보다 많으면 다음 날 시가에 청산"}
    words = (("유지 근거", "하락 전환 근거") if state == STATE_UP else ("전환 근거", "실패 근거"))
    basis = state_basis(row, ind)
    return {"frame": "A v2", "version": PLAYBOOK_VERSION, "available": True, "as_of": str(ind.index[-1].date()), "state": state,
            "pos_word": words[0], "neg_word": words[1], "state_basis": basis, "state_detail": basis["detail"],
            "close": close, "atr": atr, "items": items, "support": support, "against": against, "diff": diff,
            "lean": lean, "opinion": opinion, "position": pos, "verification": VERIFICATION["frame_a"]}


# ---------------------------------------------------------------------
# 틀 B v2 — 저평가 매수
# ---------------------------------------------------------------------
def estimates_down(consensus: Optional[Dict[str, Any]]) -> Optional[bool]:
    """추정치 하향 우세 = 30일 하향 건수 > 상향 건수 그리고 FY1 추정치가 30일 전보다 낮음. 자료가 모자라면 None."""
    cs = consensus or {}
    try:
        up, dn = float(cs["eps_fy1_up_30d"]), float(cs["eps_fy1_down_30d"])
        now, ago = float(cs["eps_fy1_avg"]), float(cs["eps_fy1_30d_ago"])
    except (KeyError, TypeError, ValueError):
        return None
    if not all(math.isfinite(x) for x in (up, dn, now, ago)):
        return None
    return bool(dn > up and now < ago)


def frame_b_evaluate(fund_out: Optional[Dict[str, Any]], synth_out: Optional[Dict[str, Any]] = None,
                     consensus: Optional[Dict[str, Any]] = None, ttm_eps: Optional[float] = None,
                     ttm_eps_prev: Optional[float] = None, trend_row: Optional[Dict[str, Any]] = None,
                     position: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """틀 B v2 판정. fund_out = analyze_fundamental_track 결과, synth_out = synthesize_4tracks 결과(킬 조건),
    position={"entry_pe": .., "entry_eps": ..}이면 진입 뒤 배수·이익 분해로 매도를 판정(없으면 12개월 분해)."""
    f = fund_out or {}
    tz = f.get("triple_z_score") or {}
    pq = (f.get("price_qualification") or {}).get("status")
    self_z, peer_z = tz.get("self_5y_z"), tz.get("peer_rel_z")
    scope = ("자기 밴드" if self_z is not None else
             ("자기 밴드 판정 불가 — 피어 기준" if peer_z is not None else "자기 밴드·피어 모두 판정 불가"))
    checks: List[Dict[str, Any]] = []

    def add(key, label, ok, text, status="미검증"):
        checks.append({"key": key, "label": label, "ok": ok, "text": text, "status": status})

    pq_word = {"QUALIFIED": "충족", "CONDITIONAL": "조건부", "DISQUALIFIED": "미충족", "NA": "판정 불가"}.get(pq, "데이터 없음")
    add("price", "V 가격 자격(매수선)", True if pq == "QUALIFIED" else (None if pq in (None, "NA") else (pq == "CONDITIONAL" or False)),
        f"{pq_word}" + (f" · 자기 밴드 z {self_z:+.2f}" if self_z is not None else "") + (f" · 피어 상대 z {peer_z:+.2f}" if self_z is None and peer_z is not None else ""))
    if ttm_eps is None or ttm_eps_prev is None:
        earn_ok = None
        add("ttm", "발표 실적(TTM EPS)", None, "TTM EPS 또는 1년 전 값 데이터 없음")
    else:
        earn_ok = float(ttm_eps) >= float(ttm_eps_prev)
        add("ttm", "발표 실적(TTM EPS)", earn_ok, f"{_fmt(ttm_eps)} vs 1년 전 {_fmt(ttm_eps_prev)} ({'줄지 않음' if earn_ok else '감소'})")
    est_dn = estimates_down(consensus)
    add("est", "추정치 하향 우세 아님", None if est_dn is None else (not est_dn),
        "데이터 없음" if est_dn is None else ("하향 우세(하향 건수 > 상향 건수, FY1 추정치 30일 전보다 낮음)" if est_dn else "하향 우세 아님"))
    sy = synth_out or {}
    killed = bool((sy.get("hard_kill_gate") or {}).get("is_killed"))
    pq_code = (sy.get("v_action_card") or {}).get("price_qual_code")
    trap = pq_code in ("HARD_KILL", "VALUE_TRAP")
    add("kill", "킬 조건 없음", None if not sy else not (killed or trap),
        "종합 판정 없음" if not sy else ("킬 조건: " + ((sy.get("hard_kill_gate") or {}).get("kill_reason") or pq_code or "") if (killed or trap) else "없음"),
        status="운용 규칙")
    # 매도: 고평가 + 배수 주도
    md = f.get("multiple_decomposition") or {}
    if position and position.get("entry_pe") and position.get("entry_eps") and tz.get("current_per_trailing") and ttm_eps:
        mult_up = math.log(float(tz["current_per_trailing"]) / float(position["entry_pe"]))
        eps_up = math.log(float(ttm_eps) / float(position["entry_eps"])) if float(ttm_eps) > 0 else -math.inf
        basis = "진입 뒤"
    else:
        mult_up, eps_up = md.get("multiple_contribution_12m"), md.get("eps_contribution_12m")
        basis = "최근 12개월"
    rich = self_z is not None and self_z >= PLAYBOOK_CONFIG["BAND_RICH_Z"]
    mult_driven = None if (mult_up is None or eps_up is None) else bool(mult_up > eps_up)
    # 단계 (방어가 공격보다 우선: 손절 > 매도 > 경고 > 진입)
    if killed or trap or earn_ok is False or (est_dn is True and pq == "DISQUALIFIED"):
        why = ("킬 조건" if (killed or trap) else ("발표 실적 감소 확정" if earn_ok is False
               else "추정치 하향으로 매수기준선이 현재가 아래로(가격 자격 미충족)"))
        stage, action = "손절 조건", f"보유 중이면 손절 · 신규 매수 금지 ({why})"
    elif rich and mult_driven is True:
        stage, action = "매도 조건", f"고평가(자기 밴드 z {self_z:+.2f})이고 {basis} 상승이 배수 주도 → 분할 매도"
    elif est_dn is True:
        stage, action = "경고", "추정치 하향 우세 — 신규 매수 중단, 보유분은 손절 조건을 지켜봄"
    elif rich and mult_driven is False:
        stage, action = "재평가 보유", f"고평가지만 {basis} 상승이 이익 주도 → 재평가로 보고 보유"
    elif pq == "QUALIFIED" and earn_ok is not False and est_dn is not True:
        stage, action = "진입 가능", "매수선 도달 + 이익 훼손 없음 → 분할 매수"
    elif pq == "CONDITIONAL" and earn_ok is not False and est_dn is not True:
        stage, action = "조건부 진입", "조건부 가격 자격 → 축소 분할 매수"
    else:
        stage, action = "대기", "매수선 미도달 또는 판정 자료 부족"
    tech = None
    if trend_row:
        try:
            down_trend = float(trend_row["Close"]) < float(trend_row["SMA200"]) and float(trend_row["slope50"]) < 0
            tech = "하락 추세 중 — 분할 간격을 넓힘(수익 신호 아님)" if down_trend else "하락 추세 아님"
        except (KeyError, TypeError, ValueError):
            tech = None
    missing = [x["label"] for x in checks if x["ok"] is None]
    return {"frame": "B v2", "scope": scope, "stage": stage, "action": action, "checks": checks,
            "rich": rich, "mult_driven": mult_driven, "decomposition_basis": basis, "technical_note": tech,
            "missing": missing,
            "verification": VERIFICATION["frame_b"] + ("" if self_z is not None else " · " + VERIFICATION["peer_scope"]),
            "sell_verification": VERIFICATION["frame_b_sell"]}


# ---------------------------------------------------------------------
# 실제 데이터로 판정 (PC·매일 기록용). 네트워크가 필요하다
# ---------------------------------------------------------------------
def light_band_z(ticker: str, yf: Any = None) -> Dict[str, Any]:
    """자기 5년 PER 밴드 z를 가볍게 계산(미국: 야후 연간 희석 EPS·월말 가격, 앱 V 트랙과 같은 밴드·신뢰 조건).
    한국은 DART가 필요해 여기서는 계산하지 않는다(딥다이브 V 트랙 값을 쓴다). 반환 {"z", "reason"}."""
    if _pb_market(ticker) == "KR":
        return {"z": None, "reason": "한국 종목은 DART 연간 EPS가 필요(딥다이브 V 트랙 값 사용)"}
    if yf is None:
        import yfinance as yf
    try:
        tk = yf.Ticker(ticker)
        hist = tk.history(period=f"{FUND_DATA_PARAMS['band_years'] + 1}y", auto_adjust=False)
        prices = _month_end_prices(hist)
        pairs = _annual_pairs(tk.income_stmt, ["Diluted EPS", "Basic EPS"])
        band = _lagged_ratio_stats(prices, pairs, FUND_DATA_PARAMS["band_lag_days_us"], FUND_DATA_PARAMS["band_years"],
                                   FUND_DATA_PARAMS["band_min_points"])
        if not band:
            return {"z": None, "reason": "밴드 이력 부족"}
        why = _band_unreliable_reasons(band)
        if why:
            return {"z": None, "reason": "이익 규모 급변: " + " · ".join(why)}
        ttm = (tk.info or {}).get("trailingEps")
        px = prices[-1][1] if prices else None
        if not ttm or ttm <= 0 or not px:
            return {"z": None, "reason": "TTM EPS가 0 이하이거나 없음"}
        return {"z": round((math.log(px / float(ttm)) - math.log(band["mean"])) / band["log_std"], 2), "reason": None}
    except Exception as e:   # 네트워크·야후 형식 변경
        return {"z": None, "reason": f"계산 실패({type(e).__name__})"}


def fetch_frame_a_live(ticker: str, yf: Any = None, consensus: Optional[Dict[str, Any]] = None,
                       gate_level: Optional[int] = None) -> Dict[str, Any]:
    """한 종목의 오늘 틀 A v2 판정을 실제 데이터로(가격 2년·시장 지수·컨센서스·자기 밴드)."""
    if yf is None:
        import yfinance as yf
    from .snapshot import fetch_consensus_row

    def dl(sym):
        df = yf.download(sym, period="2y", auto_adjust=False, actions=False, progress=False, threads=False)
        if df is not None and len(df) and isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    px = dl(ticker)
    bench = dl(PLAYBOOK_CONFIG["BENCH"][_pb_market(ticker)])
    cs = consensus if consensus is not None else fetch_consensus_row(ticker, yf=yf)
    bz = light_band_z(ticker, yf=yf)
    out = frame_a_evaluate(px, bench, cs, bz["z"], gate_level)
    out["ticker"], out["band_reason"] = ticker, bz["reason"]
    return out


def render_frame_a_text(r: Dict[str, Any]) -> str:
    """터미널·로그용 짧은 요약."""
    if not r.get("available"):
        return f"{r.get('ticker', '')}: 판정 불가 — {r.get('reason')}"
    L = [f"{r.get('ticker', '')} ({r['as_of']}) 추세: {r.get('state_detail') or r['state']} · {r['lean']['direction']}: {r['lean']['label']} "
         f"({r['pos_word']} {r['support']} vs {r['neg_word']} {r['against']}, 차이 {r['diff']:+d})"]
    sb = r.get("state_basis")
    if sb:
        for cnd in sb["conditions"]:
            L.append(f"   추세 근거 · {cnd['name']}: {cnd['text']}")
        L.append(f"   {sb['ma_note']}")
    for i in r["items"]:
        mark = ("데이터 없음" if i["no_data"] else
                {1: f"{r['pos_word']} +", -1: f"{r['neg_word']} −", 0: "중립"}[i["sign"]] + (f"{abs(i['points'])}" if i["sign"] else ""))
        L.append(f"   [{i['track']}] {i['label']}: {mark} — {i['text']}")
    op = r["opinion"]
    L.append(f"   의견: {op['label']}" + (f" · 크기 {op['size_mult']:.1f}배" if op["entry"] else "")
             + (f" · 변동성 단위 2×ATR = {op['unit_risk_pct'] * 100:.1f}%" if op.get("unit_risk_pct") else ""))
    for n in op["notes"]:
        L.append(f"   참고: {n}")
    if r.get("band_reason"):
        L.append(f"   가치 항목: {r['band_reason']}")
    L.append(f"   검증 상태: {r['verification']}")
    return "\n".join(L)
