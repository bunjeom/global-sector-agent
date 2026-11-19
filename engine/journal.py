"""매매 기록(내 실제 매매) — PC에만 저장하고, 비용 포함 손익·지수 대비·방식별 채점·손절 준수·해외주식 세금 추정 (jr-1, 2026-11-20).

저장: data/journal/trades.csv(산 기록 한 줄 = 한 번 산 것) · sells.csv(판 기록, 나눠 팔기 가능). 이 폴더는 .gitignore에 있어
공개 저장소에 올라가지 않는다. 인터넷 서버(Streamlit Community Cloud, 코드 위치 /mount/src)에서는 저장하지 않고
내보낸 CSV를 불러와 보기만 한다(journal_storage_mode). 환경 변수 JOURNAL_MODE=local|view로 바꿀 수 있다.

채점 규칙(JOURNAL_SCORING.md, 기록이 쌓이기 전에 고정 — 결과를 보고 바꾸지 않는다):
    방식(가치·추세·단타)마다 끝난 거래 30건부터 판정. 기준 숫자 = 거래마다 (비용 포함 수익률 − 같은 기간 지수 보유 수익률).
    평균 > 0 이고 t ≥ 2 → '지수보다 나음', t ≤ −2 → '지수 보유가 나음', 그 사이 → '아직 구분 안 됨'. t는 산 달로 묶은 월 묶음 표준오차.
    손절 준수 = 손실로 끝난 거래 중 (평균 판 값 ÷ 산 값 − 1) ≥ (계획 손절 ÷ 산 값 − 1) − 1%p. 넘었더라도 손절선에 처음 닿은 날
    시가가 이미 손절선 아래였고 그날까지 다 팔았으면 '갭'(규칙 위반 아님, 따로 셈).
세금(2026-11 확인, 추정): 해외주식은 한 해(결제일 기준) 모든 해외주식 손익을 합쳐 250만 원 공제 뒤 22%. 원화 환산은 결제일 환율.
    국내주식은 대주주가 아니면 매매차익 양도세 없음, 팔 때 증권거래세(2026년부터 코스피·코스닥 0.20%)는 비용에 넣는다.
"""
import json
import math
import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .common import PROJECT_ROOT
from .lecture_family import lec_cluster_t, lec_month_code

JOURNAL_CONFIG = {
    "VERSION": "jr-1",
    "DIR": os.path.join(PROJECT_ROOT, "data", "journal"),
    "TRADES_FILE": "trades.csv",
    "SELLS_FILE": "sells.csv",
    "STYLES": ("가치", "추세", "단타"),
    "STYLE_KEY": {"가치": "value", "추세": "trend", "단타": "short"},
    "BENCH": {"US": "SPY", "KR": "^KS11"},          # 같은 기간 지수 보유 비교(사용자 결정: SPY·KOSPI)
    "MIN_CLOSED": 30,                                 # 방식별 판정 시작 건수
    "T_PASS": 2.0,                                    # 판정 t 기준
    "STOP_TOL": 0.01,                                 # 손절 준수 허용 폭(1%p, 체결 미끄러짐)
    "VERDICT_MAX_GAP_DAYS": 4,                        # 딥다이브 판정 자료 날짜와 산 날 차이(주말·휴일) 이내만 '그날 판정'
    "OVERSEAS_DEDUCTION": 2_500_000,                  # 해외주식 양도소득 기본공제(연, 원)
    "OVERSEAS_RATE": 0.22,                            # 양도소득세 20% + 지방소득세 2%
    "KR_SELL_TAX": {2020: 0.0025, 2021: 0.0023, 2023: 0.0020, 2024: 0.0018, 2025: 0.0015, 2026: 0.0020},   # 증권거래세(농특세 포함, 코스피·코스닥 같음). 표의 해 사이·뒤는 앞 값, 2020 전은 2020 값
    "SETTLE_DAYS": {"US": 1, "KR": 2},                # 결제일 = 체결일 + 영업일(미국 T+1, 한국 T+2, 휴일은 반영 안 함)
    "TAX_AS_OF": "2026-11-20",
}

JOURNAL_TRADE_COLUMNS = ["id", "ticker", "market", "currency", "style", "buy_date", "buy_price", "qty", "buy_fee", "buy_fx",
                         "stop_plan", "reason", "verdict_source", "verdict_date", "verdict_headline", "verdict_value",
                         "verdict_trend", "verdict_short", "verdict_closest", "version", "created_at"]
JOURNAL_SELL_COLUMNS = ["id", "trade_id", "sell_date", "sell_price", "qty", "sell_fee", "sell_tax", "sell_fx", "note", "created_at"]
_JOURNAL_NUM = {"trades": ("buy_price", "qty", "buy_fee", "buy_fx", "stop_plan"),
                "sells": ("sell_price", "qty", "sell_fee", "sell_tax", "sell_fx")}
_JOURNAL_DATE = {"trades": ("buy_date", "verdict_date"), "sells": ("sell_date",)}
_JOURNAL_COLS = {"trades": JOURNAL_TRADE_COLUMNS, "sells": JOURNAL_SELL_COLUMNS}
_JOURNAL_VERDICT_COLS = ("verdict_source", "verdict_date", "verdict_headline", "verdict_value", "verdict_trend",
                         "verdict_short", "verdict_closest")


# ---------------------------------------------------------------------
# 저장 위치·형식
# ---------------------------------------------------------------------
def journal_storage_mode() -> str:
    """'local' = PC(저장함), 'view' = 인터넷 서버(저장 안 함, CSV 불러와 보기만)."""
    m = os.environ.get("JOURNAL_MODE", "").strip().lower()
    if m in ("local", "view"):
        return m
    return "view" if str(PROJECT_ROOT).replace("\\", "/").startswith("/mount/src") else "local"


def journal_market(ticker: str) -> str:
    t = str(ticker).strip().upper()
    return "KR" if (t.endswith((".KS", ".KQ")) or (t.isdigit() and len(t) == 6)) else "US"


def _jr_day(x) -> Optional[str]:
    if x is None or (isinstance(x, float) and math.isnan(x)) or str(x).strip() in ("", "NaT", "nan", "None"):
        return None
    try:
        ts = pd.Timestamp(x)
    except (ValueError, TypeError):
        return None
    if ts is pd.NaT or pd.isna(ts):
        return None
    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)
    return str(ts.date())


def journal_coerce(df: Optional[pd.DataFrame], kind: str) -> pd.DataFrame:
    """열을 정해진 순서로 맞추고(없는 열은 빈칸, 모르는 열은 버림) 숫자·날짜 형식을 정리한다."""
    cols = _JOURNAL_COLS[kind]
    d = pd.DataFrame(columns=cols) if df is None else df.copy()
    for c in cols:
        if c not in d.columns:
            d[c] = np.nan if c in _JOURNAL_NUM[kind] else ""
    d = d[cols].reset_index(drop=True)
    for c in _JOURNAL_NUM[kind]:
        d[c] = pd.to_numeric(d[c], errors="coerce").astype(float)
    for c in _JOURNAL_DATE[kind]:
        d[c] = [(_jr_day(v) or "") for v in d[c]]
    for c in cols:
        if c not in _JOURNAL_NUM[kind] and c not in _JOURNAL_DATE[kind]:
            d[c] = ["" if (v is None or (isinstance(v, float) and math.isnan(v))) else str(v).strip() for v in d[c]]
    return d


def journal_empty() -> Dict[str, pd.DataFrame]:
    return {"trades": journal_coerce(None, "trades"), "sells": journal_coerce(None, "sells")}


def journal_load(base_dir: Optional[str] = None) -> Dict[str, pd.DataFrame]:
    d = base_dir or JOURNAL_CONFIG["DIR"]
    out = {}
    for kind, fn in (("trades", JOURNAL_CONFIG["TRADES_FILE"]), ("sells", JOURNAL_CONFIG["SELLS_FILE"])):
        p = os.path.join(d, fn)
        raw = pd.read_csv(p, dtype=str, keep_default_na=False, encoding="utf-8-sig") if os.path.exists(p) else None
        out[kind] = journal_coerce(raw, kind)
    return out


def journal_csv_bytes(df: pd.DataFrame, kind: str) -> bytes:
    """내보내기·저장용 CSV(엑셀에서 한글이 깨지지 않게 BOM 포함 UTF-8)."""
    return journal_coerce(df, kind).to_csv(index=False).encode("utf-8-sig")


def journal_save(data: Dict[str, pd.DataFrame], base_dir: Optional[str] = None) -> None:
    if journal_storage_mode() != "local":
        raise PermissionError("이 앱은 인터넷 서버에서 실행 중이라 매매 기록을 저장하지 않습니다(PC에서 실행하면 저장).")
    d = base_dir or JOURNAL_CONFIG["DIR"]
    os.makedirs(d, exist_ok=True)
    for kind, fn in (("trades", JOURNAL_CONFIG["TRADES_FILE"]), ("sells", JOURNAL_CONFIG["SELLS_FILE"])):
        p = os.path.join(d, fn)
        tmp = p + ".tmp"
        with open(tmp, "wb") as f:
            f.write(journal_csv_bytes(data[kind], kind))
        os.replace(tmp, p)


# ---------------------------------------------------------------------
# 세금·결제일
# ---------------------------------------------------------------------
def journal_settle_date(day, market: str) -> str:
    n = JOURNAL_CONFIG["SETTLE_DAYS"]["KR" if market == "KR" else "US"]
    return str(np.busday_offset(np.datetime64(_jr_day(day)), n, roll="forward"))


def journal_kr_sell_tax_rate(day) -> float:
    tbl = JOURNAL_CONFIG["KR_SELL_TAX"]
    y = pd.Timestamp(_jr_day(day)).year
    keys = sorted(tbl)
    k = max([x for x in keys if x <= y], default=keys[0])
    return float(tbl[k])


# ---------------------------------------------------------------------
# 기록 추가·고치기·지우기
# ---------------------------------------------------------------------
def journal_verdict_snapshot(fit: Optional[Dict[str, Any]], as_of, trade_date) -> Dict[str, str]:
    """딥다이브 '세 방식 한눈에' 판정을 저장용으로. 판정 자료 날짜와 산 날이 4일 넘게 떨어지면(지난 날짜로 나중에 적은 거래) '당일 판정 없음'."""
    blank = {k: "" for k in _JOURNAL_VERDICT_COLS}
    a, t = _jr_day(as_of), _jr_day(trade_date)
    gap = (pd.Timestamp(t) - pd.Timestamp(a)).days if (a and t) else None
    if not fit or gap is None or gap < 0 or gap > JOURNAL_CONFIG["VERDICT_MAX_GAP_DAYS"]:
        return {**blank, "verdict_source": "당일 판정 없음"}
    lines = {ln.get("key"): str(ln.get("verdict") or "") for ln in fit.get("lines", [])}
    return {"verdict_source": "딥다이브", "verdict_date": a, "verdict_headline": str(fit.get("headline") or ""),
            "verdict_value": lines.get("value", ""), "verdict_trend": lines.get("trend", ""), "verdict_short": lines.get("short", ""),
            "verdict_closest": str((fit.get("closest") or {}).get("label") or "")}


def _jr_new_id(existing, prefix: str, day) -> str:
    base = f"{prefix}{_jr_day(day).replace('-', '')}-"
    used = {str(x) for x in existing}
    n = sum(1 for x in used if x.startswith(base)) + 1
    while f"{base}{n:03d}" in used:
        n += 1
    return f"{base}{n:03d}"


def _jr_num(v, name: str, positive: bool = True, allow_none: bool = False) -> Optional[float]:
    if v is None or (isinstance(v, str) and not v.strip()):
        if allow_none:
            return None
        raise ValueError(f"{name}을(를) 넣어 주세요")
    try:
        x = float(v)
    except (TypeError, ValueError):
        raise ValueError(f"{name}이(가) 숫자가 아닙니다: {v}")
    if not math.isfinite(x) or (x <= 0 if positive else x < 0):
        raise ValueError(f"{name}은(는) {'0보다 커야' if positive else '0 이상이어야'} 합니다: {v}")
    return x


def _jr_today(today=None) -> str:
    return _jr_day(today) if today is not None else str(pd.Timestamp.now(tz="Asia/Seoul").date())


def journal_validate_trade(rec: Dict[str, Any], today=None) -> Dict[str, Any]:
    """산 기록 한 줄 검사. 강의 9-7: 진입 근거·손절가·수량이 없으면 사지 않는다 → 셋 다 필수."""
    tk = str(rec.get("ticker") or "").strip().upper()
    if not tk:
        raise ValueError("종목(티커)을 넣어 주세요")
    style = str(rec.get("style") or "").strip()
    if style not in JOURNAL_CONFIG["STYLES"]:
        raise ValueError(f"방식은 {'·'.join(JOURNAL_CONFIG['STYLES'])} 중 하나여야 합니다: {style or '(빈칸)'}")
    day = _jr_day(rec.get("buy_date"))
    if not day:
        raise ValueError("산 날을 넣어 주세요")
    if day > _jr_today(today):
        raise ValueError(f"산 날이 오늘보다 뒤입니다: {day}")
    mk = journal_market(tk)
    price = _jr_num(rec.get("buy_price"), "산 값")
    qty = _jr_num(rec.get("qty"), "수량")
    fee = _jr_num(rec.get("buy_fee") if rec.get("buy_fee") not in (None, "") else 0.0, "수수료", positive=False)
    fx = 1.0 if mk == "KR" else _jr_num(rec.get("buy_fx"), "환율(원/달러)")
    stop = _jr_num(rec.get("stop_plan"), "손절 계획 가격")
    if stop >= price:
        raise ValueError(f"손절 계획 가격({stop:g})은 산 값({price:g})보다 낮아야 합니다")
    reason = str(rec.get("reason") or "").strip()
    if not reason:
        raise ValueError("진입 근거(한 줄)를 넣어 주세요")
    return {"ticker": tk, "market": mk, "currency": "KRW" if mk == "KR" else "USD", "style": style, "buy_date": day,
            "buy_price": price, "qty": qty, "buy_fee": fee, "buy_fx": fx, "stop_plan": stop, "reason": reason}


def journal_add_trade(rec: Dict[str, Any], verdict: Optional[Dict[str, str]] = None, base_dir: Optional[str] = None,
                      today=None, now=None) -> str:
    v = journal_validate_trade(rec, today=today)
    data = journal_load(base_dir)
    tid = _jr_new_id(data["trades"]["id"], "T", v["buy_date"])
    snap = verdict if verdict else {**{k: "" for k in _JOURNAL_VERDICT_COLS}, "verdict_source": "당일 판정 없음"}
    row = {**v, **{k: snap.get(k, "") for k in _JOURNAL_VERDICT_COLS}, "id": tid, "version": JOURNAL_CONFIG["VERSION"],
           "created_at": str(now or pd.Timestamp.now(tz="Asia/Seoul").strftime("%Y-%m-%d %H:%M"))}
    data["trades"] = journal_coerce(pd.concat([data["trades"], pd.DataFrame([row])], ignore_index=True), "trades")
    journal_save(data, base_dir)
    return tid


def journal_positions(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """산 기록마다 판 수량·남은 수량·상태(보유 / 일부 매도 / 청산)."""
    tr = journal_coerce(data["trades"], "trades")
    se = journal_coerce(data["sells"], "sells")
    sold = se.groupby("trade_id")["qty"].sum() if len(se) else pd.Series(dtype=float)
    out = tr.copy()
    out["sold_qty"] = [float(sold.get(i, 0.0)) for i in out["id"]]
    out["remaining"] = out["qty"] - out["sold_qty"]
    out["status"] = ["청산" if r <= 1e-9 else ("일부 매도" if s > 1e-9 else "보유") for r, s in zip(out["remaining"], out["sold_qty"])]
    return out


def journal_validate_sell(data: Dict[str, pd.DataFrame], trade_id: str, rec: Dict[str, Any], today=None) -> Dict[str, Any]:
    pos = journal_positions(data)
    hit = pos[pos["id"] == str(trade_id)]
    if not len(hit):
        raise ValueError(f"산 기록을 찾을 수 없습니다: {trade_id}")
    t = hit.iloc[0]
    day = _jr_day(rec.get("sell_date"))
    if not day:
        raise ValueError("판 날을 넣어 주세요")
    if day < t["buy_date"]:
        raise ValueError(f"판 날({day})이 산 날({t['buy_date']})보다 앞입니다")
    if day > _jr_today(today):
        raise ValueError(f"판 날이 오늘보다 뒤입니다: {day}")
    price = _jr_num(rec.get("sell_price"), "판 값")
    qty = _jr_num(rec.get("qty"), "판 수량")
    if qty > float(t["remaining"]) + 1e-9:
        raise ValueError(f"판 수량({qty:g})이 남은 수량({float(t['remaining']):g})보다 많습니다")
    fee = _jr_num(rec.get("sell_fee") if rec.get("sell_fee") not in (None, "") else 0.0, "수수료", positive=False)
    if rec.get("sell_tax") in (None, ""):
        tax = price * qty * journal_kr_sell_tax_rate(day) if t["market"] == "KR" else 0.0
    else:
        tax = _jr_num(rec.get("sell_tax"), "세금", positive=False)
    fx = 1.0 if t["market"] == "KR" else _jr_num(rec.get("sell_fx"), "환율(원/달러)")
    return {"trade_id": str(trade_id), "sell_date": day, "sell_price": price, "qty": qty, "sell_fee": fee, "sell_tax": tax,
            "sell_fx": fx, "note": str(rec.get("note") or "").strip()}


def journal_add_sell(trade_id: str, rec: Dict[str, Any], base_dir: Optional[str] = None, today=None, now=None) -> str:
    data = journal_load(base_dir)
    v = journal_validate_sell(data, trade_id, rec, today=today)
    sid = _jr_new_id(data["sells"]["id"], "S", v["sell_date"])
    row = {**v, "id": sid, "created_at": str(now or pd.Timestamp.now(tz="Asia/Seoul").strftime("%Y-%m-%d %H:%M"))}
    data["sells"] = journal_coerce(pd.concat([data["sells"], pd.DataFrame([row])], ignore_index=True), "sells")
    journal_save(data, base_dir)
    return sid


_JOURNAL_EDITABLE = ("style", "stop_plan", "reason", "buy_fee", "buy_fx")


def journal_update_trade(trade_id: str, fields: Dict[str, Any], base_dir: Optional[str] = None) -> None:
    """고칠 수 있는 칸: 방식·손절 계획·진입 근거·수수료·환율. 값·수량·날짜를 잘못 적었으면 지우고 다시 적는다(판 기록과 어긋나지 않게)."""
    data = journal_load(base_dir)
    tr = data["trades"]
    m = tr["id"] == str(trade_id)
    if not m.any():
        raise ValueError(f"산 기록을 찾을 수 없습니다: {trade_id}")
    row = tr[m].iloc[0].to_dict()
    bad = [k for k in fields if k not in _JOURNAL_EDITABLE]
    if bad:
        raise ValueError(f"고칠 수 없는 칸: {', '.join(bad)}")
    row.update(fields)
    v = journal_validate_trade(row, today="2999-12-31")
    for k in _JOURNAL_EDITABLE:
        tr.loc[m, k] = v[k]
    data["trades"] = journal_coerce(tr, "trades")
    journal_save(data, base_dir)


def journal_delete(kind: str, row_id: str, base_dir: Optional[str] = None) -> int:
    """kind='trade'면 그 산 기록과 딸린 판 기록을 함께, 'sell'이면 판 기록 한 줄을 지운다. 지운 줄 수를 돌려준다."""
    data = journal_load(base_dir)
    n = 0
    if kind == "trade":
        m = data["trades"]["id"] == str(row_id)
        ms = data["sells"]["trade_id"] == str(row_id)
        n = int(m.sum() + ms.sum())
        data["trades"], data["sells"] = data["trades"][~m], data["sells"][~ms]
    elif kind == "sell":
        m = data["sells"]["id"] == str(row_id)
        n = int(m.sum())
        data["sells"] = data["sells"][~m]
    else:
        raise ValueError("kind는 'trade' 또는 'sell'")
    if n:
        journal_save({"trades": journal_coerce(data["trades"], "trades"), "sells": journal_coerce(data["sells"], "sells")}, base_dir)
    return n


def journal_import(current: Dict[str, pd.DataFrame], trades: Optional[pd.DataFrame], sells: Optional[pd.DataFrame],
                   mode: str = "merge") -> Dict[str, Any]:
    """불러온 CSV를 검사해 합치거나(같은 id는 건너뜀) 바꾼다. 검사에 걸린 줄은 빼고 이유를 돌려준다(저장은 부르는 쪽)."""
    errs: List[str] = []
    base = journal_empty() if mode == "replace" else {k: journal_coerce(v, k) for k, v in current.items()}
    new_t = journal_coerce(trades, "trades") if trades is not None else journal_coerce(None, "trades")
    new_s = journal_coerce(sells, "sells") if sells is not None else journal_coerce(None, "sells")
    keep_t, skipped = [], 0
    have = set(base["trades"]["id"])
    for _, r in new_t.iterrows():
        rid = r["id"]
        if not rid:
            errs.append("산 기록: id 없는 줄을 뺐습니다")
            continue
        if rid in have:
            skipped += 1
            continue
        try:
            journal_validate_trade(r.to_dict(), today="2999-12-31")
        except ValueError as e:
            errs.append(f"산 기록 {rid}: {e}")
            continue
        keep_t.append(r)
        have.add(rid)
    out = {"trades": journal_coerce(pd.concat([base["trades"], pd.DataFrame(keep_t)], ignore_index=True) if keep_t else base["trades"], "trades"),
           "sells": base["sells"]}
    have_s = set(out["sells"]["id"])
    n_s = 0
    for _, r in new_s.sort_values("sell_date").iterrows():
        rid = r["id"]
        if not rid:
            errs.append("판 기록: id 없는 줄을 뺐습니다")
            continue
        if rid in have_s:
            skipped += 1
            continue
        try:
            v = journal_validate_sell(out, r["trade_id"], r.to_dict(), today="2999-12-31")
        except ValueError as e:
            errs.append(f"판 기록 {rid}: {e}")
            continue
        out["sells"] = journal_coerce(pd.concat([out["sells"], pd.DataFrame([{**r.to_dict(), **v}])], ignore_index=True), "sells")
        have_s.add(rid)
        n_s += 1
    return {"data": out, "errors": errs, "added_trades": len(keep_t), "added_sells": n_s, "skipped": skipped}


# ---------------------------------------------------------------------
# 성과
# ---------------------------------------------------------------------
def _jr_close_series(s) -> Optional[pd.Series]:
    if s is None:
        return None
    if isinstance(s, pd.DataFrame):
        if "Close" not in s:
            return None
        s = s["Close"]
    s = pd.to_numeric(pd.Series(s), errors="coerce").dropna()
    if not len(s):
        return None
    idx = pd.DatetimeIndex(pd.to_datetime(s.index))
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    return pd.Series(s.to_numpy(dtype=float), index=idx.normalize()).sort_index()


def _jr_f0(x) -> float:
    """빈칸(NaN)·None은 0으로(수수료·세금)."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return 0.0
    return v if math.isfinite(v) else 0.0


def _jr_asof(s: Optional[pd.Series], day) -> Optional[float]:
    if s is None:
        return None
    v = s.loc[:pd.Timestamp(day)]
    return float(v.iloc[-1]) if len(v) else None


def _jr_ohlc(df) -> Optional[pd.DataFrame]:
    if df is None or not len(df) or not {"Open", "Low"}.issubset(df.columns):
        return None
    d = df[["Open", "Low"]].apply(pd.to_numeric, errors="coerce").dropna()
    idx = pd.DatetimeIndex(pd.to_datetime(d.index))
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    d.index = idx.normalize()
    return d.sort_index()


def journal_verdict_match(row) -> Optional[bool]:
    """그날 딥다이브 판정이 이 거래의 방식과 같았나(가치·추세 = '맞음', 단타 = '조합 충족'). 판정이 없으면 None."""
    if str(row.get("verdict_source")) != "딥다이브":
        return None
    st = str(row.get("style"))
    if st == "가치":
        return str(row.get("verdict_value")) == "맞음"
    if st == "추세":
        return str(row.get("verdict_trend")) == "맞음"
    v = str(row.get("verdict_short"))
    return ("조합 충족" in v) and ("미충족" not in v)


def journal_stop_status(buy_price: float, stop: float, gross: float, ret: float, buy_date: str, sell_dates: List[str],
                        ohlc=None) -> str:
    """손실로 끝난 거래의 손절 계획 준수(채점 규칙 그대로). 이익 거래는 '이익'(세지 않음)."""
    if ret is None or not math.isfinite(ret) or ret >= 0:
        return "이익"
    if stop is None or not math.isfinite(stop) or stop <= 0:
        return "계획 없음"
    planned = stop / buy_price - 1.0
    if gross >= planned - JOURNAL_CONFIG["STOP_TOL"] - 1e-12:
        return "준수"
    d = _jr_ohlc(ohlc)
    if d is None:
        return "어김(갭 확인 불가)"
    after = d[d.index > pd.Timestamp(buy_date)]
    touch = after[after["Low"] <= stop]
    if not len(touch):
        return "어김(갭 확인 불가)"
    t0 = touch.index[0]
    if float(touch["Open"].iloc[0]) < stop and max(pd.Timestamp(x) for x in sell_dates) <= t0:
        return "갭"
    return "어김"


def journal_results(data: Dict[str, pd.DataFrame], bench: Optional[Dict[str, Any]] = None,
                    prices: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """산 기록마다 결과. 끝난 거래(청산)만 수익률·지수·초과·손절 결과가 채워진다.
    ret = (판 금액 − 판 수수료·세금 − 산 금액 − 산 수수료) ÷ (산 금액 + 산 수수료). 지수 = 판 수량 가중 (판 날 종가 ÷ 산 날 종가 − 1)."""
    bench = {k: _jr_close_series(v) for k, v in (bench or {}).items()}
    prices = prices or {}
    pos = journal_positions(data)
    se = journal_coerce(data["sells"], "sells")
    rows = []
    for _, t in pos.iterrows():
        s = se[se["trade_id"] == t["id"]]
        r = {k: t[k] for k in ("id", "ticker", "market", "style", "buy_date", "buy_price", "qty", "stop_plan", "reason",
                               "verdict_source", "verdict_headline", "verdict_value", "verdict_trend", "verdict_short")}
        r.update({"status": t["status"], "sold_qty": t["sold_qty"], "remaining": t["remaining"], "close_date": "",
                  "hold_days": np.nan, "ret": np.nan, "gross_ret": np.nan, "idx_ret": np.nan, "excess": np.nan,
                  "pnl": np.nan, "win": None, "stop_status": "", "verdict_match": journal_verdict_match(t)})
        cost = float(t["buy_price"]) * float(t["qty"]) + _jr_f0(t["buy_fee"])
        r["cost"] = cost
        if t["status"] == "청산" and len(s):
            proceeds = float((s["sell_price"] * s["qty"]).sum())
            sell_cost = float((s["sell_fee"].fillna(0) + s["sell_tax"].fillna(0)).sum())
            pnl = proceeds - sell_cost - cost
            ret = pnl / cost
            gross = proceeds / float(t["qty"]) / float(t["buy_price"]) - 1.0
            b = bench.get(t["market"])
            i0 = _jr_asof(b, t["buy_date"])
            idx = None
            if i0:
                legs = [(_jr_asof(b, d), q) for d, q in zip(s["sell_date"], s["qty"])]
                if all(x is not None for x, _ in legs):
                    idx = sum(q * (x / i0 - 1.0) for x, q in legs) / float(t["qty"])
            last = max(s["sell_date"])
            r.update({"close_date": last, "hold_days": float((pd.Timestamp(last) - pd.Timestamp(t["buy_date"])).days),
                      "ret": ret, "gross_ret": gross, "pnl": pnl, "win": bool(ret > 0),
                      "idx_ret": np.nan if idx is None else idx, "excess": np.nan if idx is None else ret - idx})
            r["stop_status"] = journal_stop_status(float(t["buy_price"]), float(t["stop_plan"]), gross, ret, t["buy_date"],
                                                   list(s["sell_date"]), prices.get(t["ticker"]))
        rows.append(r)
    return pd.DataFrame(rows)


def journal_judge(excess: np.ndarray, months: np.ndarray) -> Dict[str, Any]:
    """채점 규칙: 끝난 거래 30건 미만이면 '기록 중', 아니면 월 묶음 t로 세 갈래."""
    cfg = JOURNAL_CONFIG
    x = np.asarray(excess, dtype=float)
    m = np.isfinite(x)
    x, g = x[m], np.asarray(months)[m]
    st = lec_cluster_t(x, g) if len(x) else {"n": 0, "mean": None, "se": None, "t": None, "k": 0}
    n = int(st["n"])
    if n < cfg["MIN_CLOSED"]:
        verdict, tone = f"기록 중 ({n}/{cfg['MIN_CLOSED']})", "wait"
    elif st["t"] is None:
        verdict, tone = "아직 구분 안 됨", "wait"
    elif st["mean"] > 0 and st["t"] >= cfg["T_PASS"]:
        verdict, tone = "지수보다 나음", "go"
    elif st["t"] <= -cfg["T_PASS"]:
        verdict, tone = "지수 보유가 나음", "block"
    else:
        verdict, tone = "아직 구분 안 됨", "wait"
    return {**st, "verdict": verdict, "tone": tone}


def journal_style_summary(results: pd.DataFrame) -> List[Dict[str, Any]]:
    """방식별(가치·추세·단타) + 전체: 끝난 거래 수, 승률, 거래당 손익률, 지수 대비 초과, 판정, 손절 준수."""
    out = []
    res = results if results is not None and len(results) else pd.DataFrame(columns=["style", "status"])
    for style in list(JOURNAL_CONFIG["STYLES"]) + ["전체"]:
        r = res if style == "전체" else res[res["style"] == style]
        c = r[r["status"] == "청산"] if len(r) else r
        n = int(len(c))
        ex = c["excess"].to_numpy(dtype=float) if n else np.array([])
        months = lec_month_code(c["buy_date"]) if n else np.array([], dtype=np.int64)
        j = journal_judge(ex, months)
        ss = c["stop_status"].value_counts() if n else pd.Series(dtype=int)
        kept, gap = int(ss.get("준수", 0)), int(ss.get("갭", 0))
        viol = int(ss.get("어김", 0) + ss.get("어김(갭 확인 불가)", 0))
        out.append({"style": style, "n_closed": n, "n_open": int((r["status"] != "청산").sum()) if len(r) else 0,
                    "win_rate": float(c["win"].astype(bool).mean()) if n else None,
                    "mean_ret": float(c["ret"].mean()) if n else None,
                    "mean_excess": j["mean"], "t": j["t"], "n_excess": j["n"], "verdict": j["verdict"], "tone": j["tone"],
                    "stop_kept": kept, "stop_gap": gap, "stop_violated": viol, "stop_noplan": int(ss.get("계획 없음", 0)),
                    "stop_rate": (kept / (kept + viol)) if (kept + viol) else None})
    return out


def journal_verdict_compare(results: pd.DataFrame) -> List[Dict[str, Any]]:
    """참고: 그날 딥다이브 판정이 그 방식과 같았던 거래 vs 아니었던 거래의 지수 대비 초과(판정 없음은 뺌)."""
    out = []
    if results is None or not len(results):
        return out
    c = results[(results["status"] == "청산") & results["verdict_match"].notna()]
    for style in JOURNAL_CONFIG["STYLES"]:
        r = c[c["style"] == style]
        y, n = r[r["verdict_match"] == True], r[r["verdict_match"] == False]   # noqa: E712
        out.append({"style": style, "n_match": int(len(y)), "excess_match": float(y["excess"].mean()) if y["excess"].notna().any() else None,
                    "n_other": int(len(n)), "excess_other": float(n["excess"].mean()) if n["excess"].notna().any() else None})
    return out


def journal_tax_overseas(data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """해외주식(미국) 연도별 세금 추정. 판 기록마다 원화 차익 = 수량 × 판 값 × 판 환율 − 수량 × 산 값 × 산 환율
    − 산 수수료(판 수량 비율만큼) × 산 환율 − (판 수수료 + 세금) × 판 환율. 해는 결제일(T+1) 기준, 손익을 합쳐 250만 원 공제 뒤 22%."""
    cfg = JOURNAL_CONFIG
    tr = journal_coerce(data["trades"], "trades").set_index("id")
    se = journal_coerce(data["sells"], "sells")
    per_year: Dict[int, Dict[str, Any]] = {}
    missing, cross = 0, 0
    for _, s in se.iterrows():
        if s["trade_id"] not in tr.index:
            continue
        t = tr.loc[s["trade_id"]]
        if t["market"] != "US":
            continue
        bfx, sfx = t["buy_fx"], s["sell_fx"]
        if not (np.isfinite(bfx) and np.isfinite(sfx) and bfx > 0 and sfx > 0):
            missing += 1
            continue
        q = float(s["qty"])
        alloc = _jr_f0(t["buy_fee"]) * q / float(t["qty"])
        gain = (q * float(s["sell_price"]) * sfx - q * float(t["buy_price"]) * bfx - alloc * bfx
                - (_jr_f0(s["sell_fee"]) + _jr_f0(s["sell_tax"])) * sfx)
        settle = journal_settle_date(s["sell_date"], "US")
        y = int(settle[:4])
        if y != int(s["sell_date"][:4]):
            cross += 1
        e = per_year.setdefault(y, {"year": y, "n_sells": 0, "gain": 0.0})
        e["n_sells"] += 1
        e["gain"] += float(gain)
    rows = []
    for y in sorted(per_year):
        e = per_year[y]
        taxable = max(0.0, e["gain"] - cfg["OVERSEAS_DEDUCTION"])
        tax = taxable * cfg["OVERSEAS_RATE"]
        rows.append({**e, "deduction": float(cfg["OVERSEAS_DEDUCTION"]), "taxable": taxable, "tax": tax, "after_tax": e["gain"] - tax})
    return {"rows": rows, "missing_fx": missing, "cross_year": cross, "as_of": cfg["TAX_AS_OF"]}
