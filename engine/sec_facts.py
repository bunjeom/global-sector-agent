"""SEC(EDGAR) 공시일 기준 EPS — 틀 B 검증용 과거 시점 데이터 (2026-11-14 추가).

야후는 '지금' 기준 재무만 주므로 과거 날짜에 시장이 알고 있던 이익을 알 수 없다. SEC XBRL API는 숫자마다
제출일(filed)을 함께 주므로, 날짜 t에 이미 제출된 보고서만 써서 '그날 알 수 있었던 최근 12개월(TTM) EPS'를 만든다.
    받기   https://www.sec.gov/files/company_tickers.json (티커 → CIK)
           https://data.sec.gov/api/xbrl/companyconcept/CIK##########/us-gaap/EarningsPerShareDiluted.json
           (희석 EPS가 없으면 EarningsPerShareBasic). 연락처가 담긴 User-Agent 필수(.env의 SEC_USER_AGENT), 초당 10회 이하
    분기   3개월 값은 그대로, 4분기는 연간 값 − 같은 시작일의 9개월 누적 값
    TTM    연간 값이 끝나는 분기면 연간 값, 아니면 3개월 간격으로 이어진 최근 4개 분기의 합
    시점   같은 기간이 여러 보고서에 실리면(정정·비교 표시) 날짜 t까지 제출된 것 중 가장 늦은 값
    단위   액면분할 전 보고서의 EPS는 분할 비율로 나눠 지금 주식 단위로 맞춘다(야후 가격이 분할 반영 값이므로).
           비율은 그 숫자의 제출일 이후에 있었던 분할만 곱한다 — 단위 환산일 뿐 미래 정보를 쓰는 것이 아니다.
원자료는 data/sec_cache/에 저장한다(.gitignore). 다시 받으면 새 보고서가 붙지만 과거 값의 제출일은 바뀌지 않는다.
"""
import gzip
import json
import os
import time
import urllib.request
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from .common import PROJECT_ROOT
from .earnings import _read_env_value

SEC_CONFIG: Dict[str, Any] = {
    "CACHE_DIR": os.path.join(PROJECT_ROOT, "data", "sec_cache"),
    "TICKER_URL": "https://www.sec.gov/files/company_tickers.json",
    "CONCEPT_URL": "https://data.sec.gov/api/xbrl/companyconcept/CIK{cik:010d}/us-gaap/{tag}.json",
    "TAGS": ["EarningsPerShareDiluted", "EarningsPerShareBasic"],
    "UNIT": "USD/shares",
    "FORMS": ("10-Q", "10-K", "10-Q/A", "10-K/A", "10-KT", "10-KT/A"),
    "PAUSE_SEC": 0.15,              # 초당 10회 제한보다 넉넉히
    "CACHE_DAYS": 7,                # 이보다 오래된 캐시는 다시 받는다
    "QUARTER_DAYS": (80, 100),      # 3개월 기간 (52·53주 회계연도 포함)
    "ANNUAL_DAYS": (350, 380),
    "NINE_MONTH_DAYS": (260, 285),
    "PREV_YEAR_TOL_DAYS": 20,       # 1년 전 TTM 끝 날짜 허용 오차
}


def sec_user_agent() -> str:
    """SEC가 요구하는 User-Agent ('이름 이메일'). 환경변수 또는 .env의 SEC_USER_AGENT. 없으면 빈 문자열."""
    return _read_env_value("SEC_USER_AGENT")


def _sec_http_json(url: str, ua: str, timeout: int = 30) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": ua, "Accept-Encoding": "gzip"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
        if r.headers.get("Content-Encoding", "") == "gzip" or raw[:2] == b"\x1f\x8b":
            raw = gzip.decompress(raw)
    return json.loads(raw.decode("utf-8"))


def _sec_cached(path: str, fetch: Callable[[], Any], max_age_days: Optional[float] = None) -> Any:
    age = SEC_CONFIG["CACHE_DAYS"] if max_age_days is None else max_age_days
    if os.path.exists(path) and (time.time() - os.path.getmtime(path)) < age * 86400:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    data = fetch()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f)
    os.replace(tmp, path)
    return data


def load_cik_map(ua: str, fetch: Optional[Callable[[str], Any]] = None, cache_dir: Optional[str] = None) -> Dict[str, int]:
    """티커 → CIK. 야후 표기(BRK-B)와 SEC 표기를 모두 대문자로 맞춘다."""
    get = fetch or (lambda u: _sec_http_json(u, ua))
    d = cache_dir or SEC_CONFIG["CACHE_DIR"]
    data = _sec_cached(os.path.join(d, "company_tickers.json"), lambda: get(SEC_CONFIG["TICKER_URL"]))
    out: Dict[str, int] = {}
    for row in (data.values() if isinstance(data, dict) else data):
        try:
            out[str(row["ticker"]).upper().replace(".", "-")] = int(row["cik_str"])
        except (KeyError, TypeError, ValueError):
            continue
    return out


def fetch_eps_concept(ticker: str, cik: int, ua: str, fetch: Optional[Callable[[str], Any]] = None,
                      cache_dir: Optional[str] = None, pause: Optional[float] = None) -> Dict[str, Any]:
    """희석 EPS(없으면 기본 EPS) 개념 JSON. 반환 {"tag", "data"} 또는 {"tag": None, "error"}."""
    get = fetch or (lambda u: _sec_http_json(u, ua))
    d = cache_dir or SEC_CONFIG["CACHE_DIR"]
    errors = []
    for tag in SEC_CONFIG["TAGS"]:
        path = os.path.join(d, "eps", f"{ticker.upper()}_{tag}.json")
        try:
            data = _sec_cached(path, lambda: get(SEC_CONFIG["CONCEPT_URL"].format(cik=int(cik), tag=tag)))
            if pause is None or pause > 0:
                time.sleep(SEC_CONFIG["PAUSE_SEC"] if pause is None else pause)
            if parse_eps_facts(data).shape[0]:
                return {"tag": tag, "data": data}
            errors.append(f"{tag}: 값 없음")
        except Exception as e:   # 404(그 개념을 쓰지 않는 회사)·네트워크 오류
            errors.append(f"{tag}: {type(e).__name__}")
    return {"tag": None, "error": "; ".join(errors)}


def parse_eps_facts(concept: Any) -> pd.DataFrame:
    """companyconcept JSON → [start, end, val, filed, form, days] (10-K·10-Q 계열, USD/shares만)."""
    cols = ["start", "end", "val", "filed", "form", "days"]
    try:
        rows = (concept or {}).get("units", {}).get(SEC_CONFIG["UNIT"], [])
    except AttributeError:
        return pd.DataFrame(columns=cols)
    out = []
    for r in rows:
        try:
            if r.get("form") not in SEC_CONFIG["FORMS"] or "start" not in r:
                continue
            s, e, f = pd.Timestamp(r["start"]), pd.Timestamp(r["end"]), pd.Timestamp(r["filed"])
            v = float(r["val"])
        except (KeyError, TypeError, ValueError):
            continue
        if not np.isfinite(v):
            continue
        out.append({"start": s, "end": e, "val": v, "filed": f, "form": r.get("form"), "days": int((e - s).days)})
    return pd.DataFrame(out, columns=cols)


def split_factor_fn(splits: Optional[pd.Series]) -> Callable[[pd.Timestamp], float]:
    """F(d) = d 이후(d 당일 제외)에 있었던 분할 비율의 곱. 그날 보고된 주당 값 ÷ F(d) = 지금 주식 단위 값."""
    if splits is None or len(splits) == 0:
        return lambda d: 1.0
    s = pd.Series(splits, dtype=float)
    idx = pd.to_datetime(s.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    s.index = idx.normalize()
    s = s[(s > 0) & (s != 1.0)].sort_index()
    dates, ratios = s.index.values, s.to_numpy()

    def f(d: pd.Timestamp) -> float:
        k = np.searchsorted(dates, np.datetime64(pd.Timestamp(d)), side="right")
        return float(np.prod(ratios[k:])) if k < len(ratios) else 1.0
    return f


def _sec_in(days: int, rng: tuple) -> bool:
    return rng[0] <= days <= rng[1]


def _sec_ttm_tables(known: pd.DataFrame, F: Callable[[pd.Timestamp], float]) -> Dict[str, Dict]:
    """알려진 사실(기간마다 가장 늦게 제출된 값, 분할 단위 맞춤)로 분기·연간 표."""
    cfg = SEC_CONFIG
    k = known.sort_values("filed").drop_duplicates(["start", "end"], keep="last")
    q: Dict[pd.Timestamp, float] = {}
    ann: Dict[pd.Timestamp, tuple] = {}
    nine: Dict[tuple, float] = {}
    for r in k.itertuples(index=False):
        v = r.val / F(r.filed)
        if _sec_in(r.days, cfg["QUARTER_DAYS"]):
            q[r.end] = v
        elif _sec_in(r.days, cfg["ANNUAL_DAYS"]):
            ann[r.end] = (r.start, v)
        elif _sec_in(r.days, cfg["NINE_MONTH_DAYS"]):
            nine[(r.start, r.end)] = v
    for e, (s, v) in ann.items():   # 4분기 = 연간 − 같은 시작일 9개월 누적
        if e in q:
            continue
        for (s9, e9), v9 in nine.items():
            if s9 == s and _sec_in(int((e - e9).days), cfg["QUARTER_DAYS"]):
                q[e] = v - v9
                break
    return {"q": q, "ann": ann}


def _sec_ttm(tab: Dict[str, Dict], end: pd.Timestamp) -> Optional[float]:
    if end in tab["ann"]:
        return float(tab["ann"][end][1])
    q = tab["q"]
    if end not in q:
        return None
    ends = sorted(q)
    total, cur = q[end], end
    lo, hi = SEC_CONFIG["QUARTER_DAYS"]
    for _ in range(3):
        prev = [e for e in ends if lo <= (cur - e).days <= hi]
        if not prev:
            return None
        cur = max(prev)
        total += q[cur]
    return float(total)


def pit_eps_events(facts: pd.DataFrame, splits: Optional[pd.Series] = None) -> pd.DataFrame:
    """제출일마다 그날까지 알 수 있었던 TTM EPS와 1년 전 TTM EPS (지금 주식 단위).
    반환 열: filed, ttm, ttm_end, ttm_prev, n_quarters_known. 값을 만들 수 없는 제출일은 ttm이 NaN."""
    cols = ["filed", "ttm", "ttm_end", "ttm_prev"]
    if facts is None or not len(facts):
        return pd.DataFrame(columns=cols)
    F = split_factor_fn(splits)
    tol = SEC_CONFIG["PREV_YEAR_TOL_DAYS"]
    out = []
    for f in sorted(facts["filed"].unique()):
        f = pd.Timestamp(f)
        tab = _sec_ttm_tables(facts[facts["filed"] <= f], F)
        ends = sorted(set(tab["q"]) | set(tab["ann"]))
        ttm = ttm_end = prev = None
        for e in reversed(ends):          # 가장 최근에 끝난 기간부터, TTM을 만들 수 있는 첫 기간
            ttm = _sec_ttm(tab, e)
            if ttm is not None:
                ttm_end = e
                break
        if ttm is not None:
            target = ttm_end - pd.Timedelta(days=365)
            cands = sorted((abs((e - target).days), e) for e in ends if abs((e - target).days) <= tol)
            for _, e in cands:
                prev = _sec_ttm(tab, e)
                if prev is not None:
                    break
        out.append({"filed": f, "ttm": np.nan if ttm is None else ttm, "ttm_end": ttm_end,
                    "ttm_prev": np.nan if prev is None else prev})
    return pd.DataFrame(out, columns=cols)
