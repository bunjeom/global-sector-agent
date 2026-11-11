"""실적 트랙(Track 2): 실제 데이터 입력층(yfinance·DART), .env 읽기, analyze_earnings_track.

engine.py를 기능별로 나눈 모듈. 바깥에서는 `import engine` 후 `engine.이름`으로 쓴다(engine/__init__.py).
"""
import math
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd

from .common import NO_DATA, PROJECT_ROOT, fmt_price

# =====================================================================
# [Track 2] 실적 분석 트랙 v1.1 — 실제 데이터 입력층 (미국: yfinance / 한국: DART)
#  * 못 받은 값은 None. 화면에는 "데이터 없음"으로 표시한다 (AGENTS.md 데이터 무결성 규칙 3번).
#  * 삼성전자 기준으로 손으로 적어 둔 기본값(rev_base 320 등)은 더 이상 없다.
# =====================================================================

# 사이클형 '정점→저점 EPS 낙폭 기저율'은 종목 데이터가 아니라 모델 정책값이다. (종목 고유값 아님)
CYCLE_PEAK_TO_TROUGH_DROP = 0.45


def _num(v: Any) -> Optional[float]:
    """숫자로 바꿀 수 있고 유한하면 float, 아니면 None."""
    try:
        if v is None or isinstance(v, bool):
            return None
        x = float(v)
        if math.isnan(x) or math.isinf(x):
            return None
        return x
    except (TypeError, ValueError):
        return None


def _ratio_change(cur: Any, base: Any) -> Optional[float]:
    """(cur-base)/|base|. 값이 없거나 base가 0이면 None."""
    c, b = _num(cur), _num(base)
    if c is None or b is None or abs(b) < 1e-12:
        return None
    return (c - b) / abs(b)


def _diff(a: Any, b: Any) -> Optional[float]:
    x, y = _num(a), _num(b)
    return None if (x is None or y is None) else x - y


def _div(a: Any, b: Any) -> Optional[float]:
    x, y = _num(a), _num(b)
    if x is None or y is None or abs(y) < 1e-12:
        return None
    return x / y


def _s_pct(v: Any, digits: int = 1, signed: bool = True) -> str:
    x = _num(v)
    if x is None:
        return NO_DATA
    return f"{x * 100:+.{digits}f}%" if signed else f"{x * 100:.{digits}f}%"


def _s_pp(v: Any, digits: int = 1) -> str:
    x = _num(v)
    return NO_DATA if x is None else f"{x * 100:+.{digits}f}%p"


def _s_price(v: Any, is_krx: bool) -> str:
    x = _num(v)
    return NO_DATA if x is None else fmt_price(x, is_krx)


def _s_big(v: Any, is_krx: bool) -> str:
    """매출 같은 큰 금액: 한국 '조원', 미국 '십억 달러' 단위로 표시."""
    x = _num(v)
    if x is None:
        return NO_DATA
    return f"{x / 1e12:,.2f}조원" if is_krx else f"${x / 1e9:,.2f}B"


def _round_eps(v: Any, is_krx: bool) -> Optional[float]:
    x = _num(v)
    if x is None:
        return None
    return float(round(x, 0)) if is_krx else float(round(x, 2))


def _stmt_series(df: Any, names: List[str]) -> List[Optional[float]]:
    """yfinance 재무제표(행=항목, 열=날짜 최신순)에서 첫 번째로 맞는 행을 최신순 리스트로."""
    try:
        if df is None or len(df) == 0:
            return []
        idx = {str(i).strip().lower(): i for i in df.index}
        for nm in names:
            key = nm.strip().lower()
            if key in idx:
                row = df.loc[idx[key]]
                return [_num(x) for x in list(row.values)]
    except Exception:
        pass
    return []


def _quarters_from_series(sales, gp, op, ni, eps, labels) -> List[Dict[str, Any]]:
    """분기 시계열(최신순)에서 최신 3개 분기를 '분기별 dict(+전년 동기값)'으로 만든다."""
    n = max(len(sales), len(op), len(ni), len(eps), len(gp), 0)
    def g(lst, i):
        return lst[i] if 0 <= i < len(lst) else None
    out = []
    for i in range(min(n, 3)):
        out.append({
            "label": labels[i] if i < len(labels) else f"q{i}",
            "sales": g(sales, i), "gp": g(gp, i), "op": g(op, i), "ni": g(ni, i), "eps": g(eps, i),
            "sales_py": g(sales, i + 4), "gp_py": g(gp, i + 4), "op_py": g(op, i + 4),
            "ni_py": g(ni, i + 4), "eps_py": g(eps, i + 4),
        })
    return out


# ---------------------------------------------------------------------
# 미국(및 yfinance가 주는 모든 종목): yfinance
# ---------------------------------------------------------------------
def _yf_estimates(tk: Any, raw: Dict[str, Any]) -> None:
    """컨센서스: 연간 EPS 평균/최고/최저(earnings_estimate)와 30·90일 전 값(eps_trend)."""
    try:
        ee = tk.earnings_estimate
        if ee is not None and len(ee) > 0:
            for key, row_name in (("fy1", "0y"), ("fy2", "+1y")):
                if row_name in ee.index:
                    r = ee.loc[row_name]
                    raw["estimates"][key] = {
                        "avg": _num(r.get("avg")), "low": _num(r.get("low")),
                        "high": _num(r.get("high")), "n": _num(r.get("numberOfAnalysts")),
                    }
    except Exception:
        pass
    try:
        et = tk.eps_trend
        if et is not None and len(et) > 0:
            for key, row_name in (("fy1", "0y"), ("fy2", "+1y")):
                if row_name in et.index:
                    r = et.loc[row_name]
                    raw["eps_trend"][key] = {
                        "t": _num(r.get("current")), "d30": _num(r.get("30daysAgo")),
                        "d90": _num(r.get("90daysAgo")),
                    }
    except Exception:
        pass


def _fetch_us_raw(ticker: str) -> Dict[str, Any]:
    raw: Dict[str, Any] = {
        "source": "yfinance", "as_of": None, "quarters": [], "ar": [], "inv": [],
        "ocf_q": [], "capex_q": [], "ni_q": [], "estimates": {}, "eps_trend": {},
        "eps_hist": None, "ttm_eps": None, "next_fye": None, "last_fye": None,
        "shares": None, "notes": [],
    }
    try:
        import yfinance as yf
    except Exception:
        raw["notes"].append("yfinance 없음")
        return raw
    try:
        tk = yf.Ticker(ticker)
    except Exception:
        raw["notes"].append("yfinance 접속 실패")
        return raw

    try:
        inc = tk.quarterly_income_stmt
        sales = _stmt_series(inc, ["Total Revenue", "Operating Revenue"])
        gp = _stmt_series(inc, ["Gross Profit"])
        op = _stmt_series(inc, ["Operating Income", "Total Operating Income As Reported"])
        ni = _stmt_series(inc, ["Net Income", "Net Income Common Stockholders"])
        eps = _stmt_series(inc, ["Diluted EPS", "Basic EPS"])
        labels = []
        try:
            labels = [pd.Timestamp(c).strftime("%Y-%m-%d") for c in inc.columns]
            raw["as_of"] = labels[0] if labels else None
        except Exception:
            pass
        raw["quarters"] = _quarters_from_series(sales, gp, op, ni, eps, labels)
        raw["ni_q"] = ni
        if len(eps) >= 4 and all(e is not None for e in eps[:4]):
            raw["ttm_eps"] = float(sum(eps[:4]))
    except Exception:
        raw["notes"].append("분기 손익계산서 없음")
    try:
        bs = tk.quarterly_balance_sheet
        raw["ar"] = _stmt_series(bs, ["Accounts Receivable", "Receivables", "Gross Accounts Receivable"])
        raw["inv"] = _stmt_series(bs, ["Inventory"])
    except Exception:
        raw["notes"].append("분기 재무상태표 없음")
    try:
        cf = tk.quarterly_cashflow
        raw["ocf_q"] = _stmt_series(cf, ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities"])
        raw["capex_q"] = _stmt_series(cf, ["Capital Expenditure", "Purchase Of PPE"])
    except Exception:
        raw["notes"].append("분기 현금흐름표 없음")

    _yf_estimates(tk, raw)
    try:
        eh = tk.earnings_history
        if eh is not None and len(eh) > 0:
            last = eh.sort_index().iloc[-1]
            raw["eps_hist"] = {"actual": _num(last.get("epsActual")), "est": _num(last.get("epsEstimate"))}
    except Exception:
        pass
    try:
        info = tk.info or {}
        t_eps = _num(info.get("trailingEps"))
        if t_eps is not None:
            raw["ttm_eps"] = t_eps
        raw["shares"] = _num(info.get("sharesOutstanding"))
        for k_src, k_dst in (("nextFiscalYearEnd", "next_fye"), ("lastFiscalYearEnd", "last_fye")):
            ts = _num(info.get(k_src))
            if ts:
                raw[k_dst] = pd.Timestamp(ts, unit="s").strftime("%Y-%m-%d")
    except Exception:
        pass
    return raw


# ---------------------------------------------------------------------
# 한국: DART (확정 실적) + yfinance(추정치가 있으면)
# ---------------------------------------------------------------------
_DART_ACCOUNTS = {
    "sales": {"ids": ["ifrs-full_Revenue", "ifrs_Revenue"],
              "names": ["매출액", "수익(매출액)", "영업수익", "매출"], "sj": ["IS", "CIS"]},
    "gp": {"ids": ["ifrs-full_GrossProfit", "ifrs_GrossProfit"],
           "names": ["매출총이익", "매출총이익(손실)"], "sj": ["IS", "CIS"]},
    "op": {"ids": ["dart_OperatingIncomeLoss"],
           "names": ["영업이익", "영업이익(손실)"], "sj": ["IS", "CIS"]},
    "ni": {"ids": ["ifrs-full_ProfitLoss", "ifrs_ProfitLoss"],
           "names": ["당기순이익", "당기순이익(손실)", "분기순이익", "분기순이익(손실)", "반기순이익", "반기순이익(손실)"],
           "sj": ["IS", "CIS"]},
    "eps": {"ids": ["ifrs-full_BasicEarningsLossPerShare", "ifrs_BasicEarningsLossPerShare"],
            "names": ["기본주당이익", "기본주당이익(손실)", "기본주당순이익", "보통주기본주당이익", "보통주기본주당이익(손실)",
                      "기본주당순이익(손실)"], "sj": ["IS", "CIS"]},
    "ar": {"ids": ["ifrs-full_TradeAndOtherCurrentReceivables", "dart_ShortTermTradeReceivable"],
           "names": ["매출채권", "매출채권 및 기타채권", "매출채권및기타채권", "매출채권및기타유동채권"], "sj": ["BS"]},
    "inv": {"ids": ["ifrs-full_Inventories", "ifrs_Inventories"], "names": ["재고자산"], "sj": ["BS"]},
    "ocf": {"ids": ["ifrs-full_CashFlowsFromUsedInOperatingActivities"],
            "names": ["영업활동현금흐름", "영업활동 현금흐름", "영업활동으로인한현금흐름"], "sj": ["CF"]},
    "capex": {"ids": ["ifrs-full_PurchaseOfPropertyPlantAndEquipmentClassifiedAsInvestingActivities",
                      "ifrs-full_PurchaseOfPropertyPlantAndEquipment"],
              "names": ["유형자산의취득", "유형자산의 취득", "유형자산 취득"], "sj": ["CF"]},
}


def _dart_to_num(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        s = str(v).replace(",", "").strip()
        if s in ("", "-", "nan", "None"):
            return None
        return float(s)
    except (TypeError, ValueError):
        return None


def _dart_pick(df: Any, key: str, col: str) -> Optional[float]:
    """DART 재무제표 표(df)에서 계정(key)의 col 열 값을 찾는다. 없으면 None."""
    try:
        if df is None or len(df) == 0 or col not in df.columns:
            return None
        spec = _DART_ACCOUNTS[key]
        sub = df[df["sj_div"].isin(spec["sj"])] if "sj_div" in df.columns else df
        if "account_id" in sub.columns:
            for aid in spec["ids"]:
                hit = sub[sub["account_id"] == aid]
                if len(hit):
                    val = _dart_to_num(hit.iloc[0][col])
                    if val is not None:
                        return val
        if "account_nm" in sub.columns:
            norm = sub["account_nm"].astype(str).str.replace(" ", "", regex=False)
            for nm in spec["names"]:
                hit = sub[norm == nm.replace(" ", "")]
                if len(hit):
                    val = _dart_to_num(hit.iloc[0][col])
                    if val is not None:
                        return val
    except Exception:
        pass
    return None


def _read_env_value(name: str) -> str:
    """환경변수에 없으면 프로젝트 폴더의 .env 파일에서 읽는다 (Streamlit 실행 시 .env가 자동 로드되지 않음)."""
    val = os.getenv(name, "")
    if val:
        return val
    try:
        env_path = os.path.join(PROJECT_ROOT, ".env")  # engine/ 폴더가 아니라 프로젝트 폴더의 .env
        with open(env_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                if k.strip() == name:
                    return v.strip().strip('"').strip("'")
    except Exception:
        pass
    return ""


def _get_dart_client() -> Any:
    key = _read_env_value("DART_API_KEY")
    if not key:
        return None
    try:
        import importlib
        mod = None
        for name in ("OpenDartReader", "opendartreader"):
            try:
                mod = importlib.import_module(name)
                break
            except Exception:
                continue
        if mod is None:
            return None
        cls = getattr(mod, "OpenDartReader", mod)
        return cls(key)
    except Exception:
        return None


_DART_SEQ = [("11011", "FY"), ("11014", "Q3"), ("11012", "H1"), ("11013", "Q1")]  # 최신 → 과거 (한 해 안에서)


def _dart_fetch(dart: Any, code6: str, year: int, reprt: str, cache: Dict[Tuple[int, str], Any]) -> Any:
    k = (year, reprt)
    if k in cache:
        return cache[k]
    df = None
    for fs_div in ("CFS", "OFS"):  # 연결 우선, 없으면 별도
        try:
            cand = dart.finstate_all(code6, year, reprt_code=reprt, fs_div=fs_div)
            if cand is not None and len(cand) > 0:
                df = cand
                break
        except Exception:
            continue
    cache[k] = df
    return df


def _dart_quarter_value(cache, dart, code6, year, reprt, key) -> Optional[float]:
    """한 분기(3개월) 값. 사업보고서(FY)는 '연간 − 3분기 누적'으로 4분기를 계산한다."""
    df = _dart_fetch(dart, code6, year, reprt, cache)
    if df is None:
        return None
    if reprt == "11011":
        fy = _dart_pick(df, key, "thstrm_amount")
        q3 = _dart_fetch(dart, code6, year, "11014", cache)
        q3_cum = _dart_pick(q3, key, "thstrm_add_amount") if q3 is not None else None
        if key == "eps":  # 주당이익은 연간−누적이 성립하지 않으므로 4분기 값은 만들지 않는다
            return None
        return None if (fy is None or q3_cum is None) else fy - q3_cum
    return _dart_pick(df, key, "thstrm_amount")


def _fetch_kr_raw(ticker: str) -> Dict[str, Any]:
    raw: Dict[str, Any] = {
        "source": "DART", "as_of": None, "quarters": [], "ar": [], "inv": [],
        "ocf_q": [], "capex_q": [], "ni_q": [], "estimates": {}, "eps_trend": {},
        "eps_hist": None, "ttm_eps": None, "next_fye": None, "last_fye": None,
        "shares": None, "notes": [], "annual": {},
    }
    code6 = str(ticker).split(".")[0]
    dart = _get_dart_client()
    if dart is None:
        raw["notes"].append("DART 사용 불가(API 키 또는 라이브러리 없음)")
    else:
        cache: Dict[Tuple[int, str], Any] = {}
        this_year = datetime.now().year
        seq: List[Tuple[int, str]] = []
        for y in (this_year, this_year - 1, this_year - 2):
            for rc, _ in _DART_SEQ:
                seq.append((y, rc))
        start = None
        for i, (y, rc) in enumerate(seq[:6]):
            if _dart_fetch(dart, code6, y, rc, cache) is not None:
                start = i
                break
        if start is not None:
            periods = seq[start:start + 3]
            qlist: List[Dict[str, Any]] = []
            for (y, rc) in periods:
                q: Dict[str, Any] = {"label": f"{y}-{dict(_DART_SEQ)[rc]}"}
                for key in ("sales", "gp", "op", "ni", "eps"):
                    q[key] = _dart_quarter_value(cache, dart, code6, y, rc, key)
                    q[key + "_py"] = _dart_quarter_value(cache, dart, code6, y - 1, rc, key)
                qlist.append(q)
            raw["quarters"] = qlist
            raw["as_of"] = qlist[0]["label"] if qlist else None
            # 재무상태표: 최신 보고서 vs 전년 같은 시점 보고서
            y0, rc0 = periods[0]
            d0 = _dart_fetch(dart, code6, y0, rc0, cache)
            d1 = _dart_fetch(dart, code6, y0 - 1, rc0, cache)
            for key in ("ar", "inv"):
                raw[key] = [_dart_pick(d0, key, "thstrm_amount"), None, None, None,
                            _dart_pick(d1, key, "thstrm_amount")]
            # 현금흐름: 가장 최근 사업보고서(연간) 기준 (분기 보고서의 현금흐름은 누적이라 연간만 사용)
            fy_year = None
            for y in (this_year - 1, this_year - 2):
                dfy = _dart_fetch(dart, code6, y, "11011", cache)
                if dfy is not None:
                    fy_year = y
                    break
            if fy_year is not None:
                dfy = _dart_fetch(dart, code6, fy_year, "11011", cache)
                raw["annual"] = {
                    "year": fy_year,
                    "ocf": _dart_pick(dfy, "ocf", "thstrm_amount"),
                    "ocf_prev": _dart_pick(dfy, "ocf", "frmtrm_amount"),
                    "ni": _dart_pick(dfy, "ni", "thstrm_amount"),
                    "capex": _dart_pick(dfy, "capex", "thstrm_amount"),
                    "eps": _dart_pick(dfy, "eps", "thstrm_amount"),
                }
                if raw["annual"].get("eps") is not None:
                    raw["ttm_eps"] = raw["annual"]["eps"]  # 최근 사업연도 기본 EPS (yfinance TTM이 있으면 아래서 대체)
        else:
            raw["notes"].append("DART 재무제표 없음")

    # 추정치(컨센서스)와 TTM EPS: yfinance가 한국 종목에도 주는 경우에만 사용
    try:
        import yfinance as yf
        tk = yf.Ticker(ticker)
        _yf_estimates(tk, raw)
        try:
            info = tk.info or {}
            t_eps = _num(info.get("trailingEps"))
            if t_eps is not None:
                raw["ttm_eps"] = t_eps
            raw["shares"] = _num(info.get("sharesOutstanding"))
        except Exception:
            pass
    except Exception:
        raw["notes"].append("yfinance 추정치 없음")
    return raw


# ---------------------------------------------------------------------
# 입력 조립: raw → 실적 트랙 입력(d). 값이 없으면 None.
# ---------------------------------------------------------------------
def _months_remaining(raw: Dict[str, Any], is_krx: bool) -> Optional[int]:
    """현재 회계연도(FY1) 끝까지 남은 개월 수. 한국은 12월 결산을 기본 가정."""
    now = datetime.now()
    fye = raw.get("next_fye")
    try:
        if fye:
            dt = pd.Timestamp(fye).to_pydatetime()
            m = int(round((dt - now).days / 30.4))
            return int(min(max(m, 0), 12))
    except Exception:
        pass
    if is_krx:  # 한국 상장사 대부분 12월 결산
        m = int(round((datetime(now.year, 12, 31) - now).days / 30.4))
        return int(min(max(m, 0), 12))
    return None


def _fy_labels(raw: Dict[str, Any], is_krx: bool) -> Tuple[str, str, str]:
    try:
        if raw.get("next_fye"):
            y = pd.Timestamp(raw["next_fye"]).year
        elif is_krx:
            y = datetime.now().year
        else:
            return ("FY+1", "FY+2", "FY+3")
        return (f"FY{y % 100:02d}E", f"FY{(y + 1) % 100:02d}E", f"FY{(y + 2) % 100:02d}E")
    except Exception:
        return ("FY+1", "FY+2", "FY+3")


def _assemble_earnings_inputs(raw: Dict[str, Any], is_krx: bool) -> Dict[str, Any]:
    quarters = raw.get("quarters") or []
    q0 = quarters[0] if len(quarters) > 0 else {}
    q1 = quarters[1] if len(quarters) > 1 else {}
    q2 = quarters[2] if len(quarters) > 2 else {}

    def yoy(q, k):
        return _ratio_change(q.get(k), q.get(k + "_py")) if q else None

    def margin(q, num, den="sales", py=False):
        if not q:
            return None
        return _div(q.get(num + ("_py" if py else "")), q.get(den + ("_py" if py else "")))

    def eps_g(a, b):  # EPS 증감률: 기준값이 0 이하이면 의미가 없어 None
        x, y = _num(a), _num(b)
        if x is None or y is None or y <= 0:
            return None
        return (x - y) / y

    g_traj = [yoy(q2, "sales"), yoy(q1, "sales"), yoy(q0, "sales")]
    sales_traj = g_traj if any(g is not None for g in g_traj) else None

    gpm0, gpm1 = margin(q0, "gp"), margin(q1, "gp")
    opm0, opm1 = margin(q0, "op"), margin(q1, "op")
    eh = raw.get("eps_hist") or {}
    eps_actual = _num(q0.get("eps")) if q0 else None
    ac = {
        "sales_actual": _num(q0.get("sales")), "sales_cons": None,
        "sales_qoq": _ratio_change(q0.get("sales"), q1.get("sales")) if q1 else None,
        "sales_yoy": yoy(q0, "sales"), "sales_prev_yoy": yoy(q1, "sales"),
        "gpm_actual": gpm0, "gpm_cons": None,
        "gpm_qoq": _diff(gpm0, gpm1), "gpm_yoy": _diff(gpm0, margin(q0, "gp", py=True)),
        "gpm_prev_yoy": _diff(gpm1, margin(q1, "gp", py=True)) if q1 else None,
        "opm_actual": opm0, "opm_cons": None,
        "opm_qoq": _diff(opm0, opm1), "opm_yoy": _diff(opm0, margin(q0, "op", py=True)),
        "opm_prev_yoy": _diff(opm1, margin(q1, "op", py=True)) if q1 else None,
        "eps_actual": eps_actual, "eps_cons": None,
        "eps_qoq": eps_g(q0.get("eps"), q1.get("eps")) if q1 else None,
        "eps_yoy": eps_g(q0.get("eps"), q0.get("eps_py")) if q0 else None,
        "eps_prev_yoy": eps_g(q1.get("eps"), q1.get("eps_py")) if q1 else None,
    }
    # EPS 서프라이즈: yfinance earnings_history(직전 분기 발표값·추정치)를 같은 기준(조정 EPS)으로 비교
    adj_eps = None
    if eh.get("est") is not None and eh.get("actual") is not None:
        ac["eps_actual"], ac["eps_cons"] = eh["actual"], eh["est"]
        adj_eps = eh["actual"]

    # 컨센서스·리비전·시나리오 EPS
    est, trend = raw.get("estimates") or {}, raw.get("eps_trend") or {}
    m_rem = _months_remaining(raw, is_krx)
    f1, f2 = est.get("fy1") or {}, est.get("fy2") or {}
    t1, t2 = trend.get("fy1") or {}, trend.get("fy2") or {}
    fy1_base = f1.get("avg") if f1.get("avg") is not None else t1.get("t")
    fy2_base = f2.get("avg") if f2.get("avg") is not None else t2.get("t")

    def blend(a, b):
        if a is None or b is None or m_rem is None:
            return None
        return (m_rem / 12.0) * float(a) + ((12 - m_rem) / 12.0) * float(b)

    revision = {
        "ntm_t": blend(t1.get("t"), t2.get("t")), "ntm_30d": blend(t1.get("d30"), t2.get("d30")),
        "ntm_90d": blend(t1.get("d90"), t2.get("d90")),
        "fy1_t": t1.get("t"), "fy1_30d": t1.get("d30"), "fy1_90d": t1.get("d90"),
        "fy2_t": t2.get("t"), "fy2_30d": t2.get("d30"), "fy2_90d": t2.get("d90"),
        "analysts_t": _num(f1.get("n")), "analysts_30d": None, "analysts_90d": None,
    }

    # 분기 데이터가 있는 최신 값들로 '분기→TTM' 계산 (현금흐름·순이익)
    def ttm(lst):
        return float(sum(lst[:4])) if len(lst) >= 4 and all(x is not None for x in lst[:4]) else None

    def prev_ttm(lst):
        return float(sum(lst[4:8])) if len(lst) >= 8 and all(x is not None for x in lst[4:8]) else None

    ann = raw.get("annual") or {}
    if raw.get("source") == "DART":
        quality = {
            "ar_yoy": _ratio_change((raw.get("ar") or [None])[0], (raw.get("ar") or [None] * 5)[4]),
            "inv_yoy": _ratio_change((raw.get("inv") or [None])[0], (raw.get("inv") or [None] * 5)[4]),
            "non_gaap_eps": None, "gaap_eps": _num(q0.get("eps")) if q0 else None,
            "ocf_ttm": ann.get("ocf"), "ocf_prev_ttm": ann.get("ocf_prev"),
            "net_income_ttm": ann.get("ni"),
            "capex_ttm": abs(ann["capex"]) if ann.get("capex") is not None else None,
            "reinvest_per_dollar_rising": None, "basis": f"연간(FY{ann.get('year')}) 기준" if ann else None,
        }
    else:
        ar, inv = raw.get("ar") or [], raw.get("inv") or []
        capex_q = [abs(x) if x is not None else None for x in (raw.get("capex_q") or [])]
        quality = {
            "ar_yoy": _ratio_change(ar[0] if ar else None, ar[4] if len(ar) > 4 else None),
            "inv_yoy": _ratio_change(inv[0] if inv else None, inv[4] if len(inv) > 4 else None),
            "non_gaap_eps": adj_eps, "gaap_eps": _num(q0.get("eps")) if q0 else None,
            "ocf_ttm": ttm(raw.get("ocf_q") or []), "ocf_prev_ttm": prev_ttm(raw.get("ocf_q") or []),
            "net_income_ttm": ttm(raw.get("ni_q") or []), "capex_ttm": ttm(capex_q),
            "reinvest_per_dollar_rising": None, "basis": "최근 4분기 합산(TTM)",
        }

    sus = {
        "rpo_yoy_growth": None,  # 수주잔고(RPO)는 yfinance·DART 표준 항목에 없다
        "op_t": _num(q0.get("op")) if q0 else None, "op_t_minus_4": _num(q0.get("op_py")) if q0 else None,
        "sales_t": _num(q0.get("sales")) if q0 else None, "sales_t_minus_4": _num(q0.get("sales_py")) if q0 else None,
        "design_wins_note": None,
    }

    return {
        "sales_yoy_trajectory": sales_traj,
        "actual_vs_cons": ac,
        "guidance_history": None,  # 가이던스 이행 이력은 표준 데이터 출처 없음
        "revision": revision,
        "forward": {
            "fy1": {"base": fy1_base, "bull": f1.get("high"), "bear": f1.get("low")},
            "fy2": {"base": fy2_base, "bull": f2.get("high"), "bear": f2.get("low")},
            "months_remaining": m_rem, "labels": _fy_labels(raw, is_krx),
        },
        "ttm_eps": _num(raw.get("ttm_eps")),
        "quality": quality,
        "sustainability": sus,
        "thesis": {
            "hard_block_triggered": False,
            "invalidation_condition": None, "current_status": None,   # 종목별 가설 문구는 사용자 입력 영역
            "p_trend": None, "q_trend": None,                          # 가격(P)·수량(Q) 추세는 산업 데이터 필요
            "capital_structure_note": (
                f"발행주식수 {raw['shares']:,.0f}주 (희석 전 · {raw.get('source')})" if _num(raw.get("shares")) else None
            ),
        },
        "meta": {"source": raw.get("source"), "as_of": raw.get("as_of"), "notes": list(raw.get("notes") or [])},
    }


def fetch_real_earnings_inputs(ticker: str) -> Dict[str, Any]:
    """실적 트랙 입력을 실제 데이터로 만든다. 미국/기타: yfinance, 한국(.KS/.KQ): DART(+yfinance 추정치).
    못 받은 값은 None이며 가짜 값으로 채우지 않는다."""
    is_krx = str(ticker).upper().endswith((".KS", ".KQ"))
    try:
        raw = _fetch_kr_raw(ticker) if is_krx else _fetch_us_raw(ticker)
    except Exception as e:  # 어떤 이유로든 실패하면 전부 '데이터 없음'
        raw = {"source": "DART" if is_krx else "yfinance", "as_of": None, "quarters": [], "notes": [f"수집 실패: {type(e).__name__}"]}
    return _assemble_earnings_inputs(raw, is_krx)


def analyze_earnings_track(
    ticker: str,
    sector_type: str = "구조성장형",          # "사이클형" | "구조성장형" | "방어형" | "정책의존형"
    fin_override: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    실적 분석 트랙 v1.1 메인 파이프라인 (E1 ~ E8 연산, G1~G5 생애주기 분류, 5대 축 점수 산출).
    * 순환논리 차단: 주가·PER을 입력으로 받지 않고 실물/재무/리비전 데이터만 사용한다.
    * 입력은 실제 데이터(미국 yfinance · 한국 DART)에서만 만든다. 못 받은 값은 '데이터 없음'(None)이며
      해당 항목은 판정에서 빠진다(점수는 판정 가능한 축의 가중치로만 정규화).
    """
    is_krx = str(ticker).upper().endswith((".KS", ".KQ"))
    d = fetch_real_earnings_inputs(ticker)

    if fin_override:
        for k, v in fin_override.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                d[k].update(v)
            else:
                d[k] = v

    ac = d["actual_vs_cons"]
    missing: List[str] = []

    def need(label: str, *vals: Any) -> bool:
        """모든 값이 있으면 True. 하나라도 없으면 '데이터 없음 목록'에 라벨을 남기고 False."""
        ok = all(_num(v) is not None for v in vals)
        if not ok and label not in missing:
            missing.append(label)
        return ok

    # -----------------------------------------------------------------
    # E1. 실적 액추얼 괴리율 및 성장 가속도 (2차 미분) 판정
    # -----------------------------------------------------------------
    sales_beat_pct = _ratio_change(ac["sales_actual"], ac["sales_cons"])
    gpm_beat_diff = _diff(ac["gpm_actual"], ac["gpm_cons"])
    opm_beat_diff = _diff(ac["opm_actual"], ac["opm_cons"])
    eps_beat_pct = _ratio_change(ac["eps_actual"], ac["eps_cons"])
    if sales_beat_pct is None:
        missing.append("매출 컨센서스(직전 분기)")
    if opm_beat_diff is None and gpm_beat_diff is None:
        missing.append("마진 컨센서스(직전 분기)")

    margin_squeeze_known = bool(sales_beat_pct is not None and (opm_beat_diff is not None or gpm_beat_diff is not None))
    margin_squeeze_detected = bool(
        margin_squeeze_known and sales_beat_pct > 0.0 and
        ((opm_beat_diff is not None and opm_beat_diff < -0.002) or (gpm_beat_diff is not None and gpm_beat_diff < -0.003))
    )

    g_traj = d.get("sales_yoy_trajectory") or [None, None, None]
    g_t2, g_t1, g_t = [_num(x) for x in (list(g_traj) + [None, None, None])[:3]]
    growth_accel: Optional[str] = None
    axis1_score: Optional[int] = None
    if g_t is not None and g_t1 is not None and g_t2 is not None:
        if g_t2 < g_t1 < g_t and (g_t - g_t1) > 0.005:
            growth_accel, axis1_score = "가속", 2
        elif abs(g_t - g_t1) <= 0.02 and g_t >= g_t2:
            growth_accel, axis1_score = "유지", 1
        elif g_t2 > g_t1 > g_t or (g_t - g_t1) < -0.02:
            growth_accel, axis1_score = "감속", 0
        else:
            growth_accel, axis1_score = "유지", 1
        traj_text = f"{g_t2*100:.0f}%→{g_t1*100:.0f}%→{g_t*100:.0f}%"
    elif g_t is not None and g_t1 is not None:      # 2점만 있으면 2점 판정(±2%p)
        delta = g_t - g_t1
        growth_accel, axis1_score = (("가속", 2) if delta > 0.02 else (("감속", 0) if delta < -0.02 else ("유지", 1)))
        traj_text = f"{g_t1*100:.0f}%→{g_t*100:.0f}% (2개 분기만 확인)"
        missing.append("매출 성장률 3개 분기 궤적(2개만 확인)")
    else:
        traj_text = NO_DATA
        missing.append("매출 성장률 궤적")

    e1_table = [
        {"metric": "매출액", "actual": _s_big(ac["sales_actual"], is_krx), "consensus": _s_big(ac["sales_cons"], is_krx),
         "surprise": _s_pct(sales_beat_pct), "qoq": _s_pct(ac["sales_qoq"]), "yoy": _s_pct(ac["sales_yoy"]),
         "prev_yoy": _s_pct(ac["sales_prev_yoy"])},
        {"metric": "매출총이익률(GPM)", "actual": _s_pct(ac["gpm_actual"], signed=False), "consensus": _s_pct(ac["gpm_cons"], signed=False),
         "surprise": _s_pp(gpm_beat_diff), "qoq": _s_pp(ac["gpm_qoq"]), "yoy": _s_pp(ac["gpm_yoy"]),
         "prev_yoy": _s_pp(ac["gpm_prev_yoy"])},
        {"metric": "영업이익률(OPM)", "actual": _s_pct(ac["opm_actual"], signed=False), "consensus": _s_pct(ac["opm_cons"], signed=False),
         "surprise": _s_pp(opm_beat_diff), "qoq": _s_pp(ac["opm_qoq"]), "yoy": _s_pp(ac["opm_yoy"]),
         "prev_yoy": _s_pp(ac["opm_prev_yoy"])},
        {"metric": "EPS (컨센서스 대비는 조정 EPS 기준)", "actual": _s_price(ac["eps_actual"], is_krx),
         "consensus": _s_price(ac["eps_cons"], is_krx), "surprise": _s_pct(eps_beat_pct),
         "qoq": _s_pct(ac["eps_qoq"]), "yoy": _s_pct(ac["eps_yoy"]), "prev_yoy": _s_pct(ac["eps_prev_yoy"])},
    ]

    # -----------------------------------------------------------------
    # E2. 가이던스 이행 이력 -> 확실성 판정 (표준 데이터 출처가 없어 입력이 없으면 '데이터 없음')
    # -----------------------------------------------------------------
    g_hist = d.get("guidance_history")
    axis5_score: Optional[int] = None
    if g_hist == "beat_4q":
        certainty_label, certainty_adj, axis5_score = "확실성 높음 (4분기 연속 Beat & Raise)", -0.02, 2
    elif g_hist == "miss_2q":
        certainty_label, certainty_adj, axis5_score = "확실성 낮음 (2분기 연속 가이던스 미스)", 0.02, 0
    elif g_hist == "neutral":
        certainty_label, certainty_adj, axis5_score = "중립 (부합 또는 가이던스 미제시)", 0.0, 1
    else:
        certainty_label, certainty_adj = f"{NO_DATA} (가이던스 이행 이력 미확인 · 할인율 조정 없음)", 0.0
        missing.append("가이던스 이행 이력")

    # -----------------------------------------------------------------
    # E3. NTM EPS 리비전의 변곡점 추적 (90일 vs 30일 괴리 및 롤오버 검증)
    # -----------------------------------------------------------------
    rev = d["revision"]
    delta_90 = _ratio_change(rev["ntm_t"], rev["ntm_90d"])
    delta_30 = _ratio_change(rev["ntm_t"], rev["ntm_30d"])
    fy1_chg = _ratio_change(rev["fy1_t"], rev["fy1_90d"])
    fy2_chg = _ratio_change(rev["fy2_t"], rev["fy2_90d"])
    true_fy_up = bool(fy1_chg is not None and fy2_chg is not None and fy1_chg > 0.0 and fy2_chg > 0.0)

    inflection_flag = False
    axis2_score: Optional[int] = None
    if delta_90 is None or delta_30 is None:
        rev_status = NO_DATA
        missing.append("EPS 컨센서스 리비전(30·90일 전 추정치)")
    elif delta_90 > 0.01 and delta_30 < -0.005:
        rev_status, inflection_flag, axis2_score = "피크아웃 경고", True, 0
    elif delta_90 < -0.01 and delta_30 > 0.005:
        rev_status, inflection_flag, axis2_score = "턴어라운드 조짐", True, 1
    elif delta_90 > 0.01 and delta_30 > 0.0 and true_fy_up:
        rev_status, axis2_score = "상향 지속", 2
    elif delta_90 < -0.01 and delta_30 < 0.0:
        rev_status, axis2_score = "하향 지속", 0
    else:
        rev_status, axis2_score = "보합", 1

    # -----------------------------------------------------------------
    # E4 & E5. 시나리오 EPS (컨센서스 평균=Base, 최고=Bull, 최저=Bear) 및 1Y/2Y NTM 혼합식
    # -----------------------------------------------------------------
    fw = d["forward"]
    m_rem = fw.get("months_remaining")
    lbl_1, lbl_2, lbl_3 = fw.get("labels", ("FY+1", "FY+2", "FY+3"))
    w_a = round(m_rem / 12.0, 2) if m_rem is not None else None
    w_b = round((12 - m_rem) / 12.0, 2) if m_rem is not None else None

    def _blend(a, b):
        if w_a is None or _num(a) is None or _num(b) is None:
            return None
        return _round_eps(w_a * float(a) + w_b * float(b), is_krx)

    f1, f2 = fw["fy1"], fw["fy2"]
    eps_1y = {k: _blend(f1.get(k), f2.get(k)) for k in ("base", "bull", "bear")}
    # 2Y: FY+3 추정치가 없어 NTM 혼합이 불가능 → FY+2 컨센서스 값을 그대로 사용(표기에 명시)
    eps_2y = {k: _round_eps(f2.get(k), is_krx) for k in ("base", "bull", "bear")}
    cycle_bear_1y = (
        _round_eps(eps_1y["bull"] * (1.0 - CYCLE_PEAK_TO_TROUGH_DROP), is_krx)
        if (sector_type == "사이클형" and eps_1y["bull"] is not None) else None
    )
    ttm_eps = _num(d.get("ttm_eps"))
    if eps_1y["base"] is not None and eps_1y["bull"] is not None and eps_1y["bear"] is not None:
        eps_basis, eps_basis_label = "forward", "컨센서스 기준"
    elif eps_1y["base"] is not None:
        eps_basis, eps_basis_label = "forward_base_only", "컨센서스 평균만 있음(최고·최저 없음)"
    elif ttm_eps is not None:
        eps_basis, eps_basis_label = "ttm_reference", "참고용: 최근 확정 EPS(TTM) · 미래 시나리오 없음"
    else:
        eps_basis, eps_basis_label = "none", NO_DATA
    if eps_1y["base"] is None:
        missing.append("미래 EPS 컨센서스(FY1·FY2)")
    elif eps_1y["bull"] is None or eps_1y["bear"] is None:
        missing.append("EPS 최고·최저 추정(Bull·Bear)")
    if ttm_eps is None:
        missing.append("최근 확정 EPS(TTM)")

    f_1y = f"{w_a:.2f}*{lbl_1} + {w_b:.2f}*{lbl_2}" if w_a is not None else NO_DATA
    f_2y = f"{lbl_2} 컨센서스 (FY+3 추정치 없음 → NTM 혼합 불가)"

    # -----------------------------------------------------------------
    # E6. 이익의 질 5대 체크 — 데이터가 있는 항목만 판정
    # -----------------------------------------------------------------
    q_data = d["quality"]
    sales_yoy = _num(ac["sales_yoy"])
    quality_items: List[Dict[str, Any]] = []
    warn_cnt, n_checked = 0, 0

    def add_item(name, value, warn, desc):
        nonlocal warn_cnt, n_checked
        if warn is not None:
            n_checked += 1
            warn_cnt += 1 if warn else 0
        quality_items.append({"name": name, "value": value, "warn": bool(warn) if warn is not None else False,
                              "checked": warn is not None, "desc": desc})

    # 1) 매출채권 vs 매출
    if need("매출채권 증가율", q_data["ar_yoy"]) and need("매출 증가율(YoY)", sales_yoy):
        ar_diff = q_data["ar_yoy"] - sales_yoy
        add_item("매출채권 vs 매출", f"{ar_diff*100:+.1f}%p", ar_diff > 0.05,
                 "밀어내기 매출·회수 지연 경고" if ar_diff > 0.05 else "정상 회수 범위")
    else:
        add_item("매출채권 vs 매출", NO_DATA, None, NO_DATA)
    # 2) 재고 vs 매출
    if need("재고자산 증가율", q_data["inv_yoy"]) and need("매출 증가율(YoY)", sales_yoy):
        inv_diff = q_data["inv_yoy"] - sales_yoy
        add_item("재고자산 vs 매출", f"{inv_diff*100:+.1f}%p", inv_diff > 0.05,
                 "전방 수요 둔화·재고 누적 경고" if inv_diff > 0.05 else "재고 건전성 양호")
    else:
        inv_diff = None
        add_item("재고자산 vs 매출", NO_DATA, None, NO_DATA)
    # 3) GAAP vs Non-GAAP 격차
    if need("조정(Non-GAAP) EPS", q_data["non_gaap_eps"]) and need("GAAP EPS", q_data["gaap_eps"]) and float(q_data["non_gaap_eps"]) > 0:
        ng_gap = (q_data["non_gaap_eps"] - q_data["gaap_eps"]) / abs(q_data["non_gaap_eps"])
        add_item("GAAP vs Non-GAAP 격차", f"{ng_gap*100:.1f}%", ng_gap > 0.20,
                 "SBC 과다·일회성 비용 만성화" if ng_gap > 0.20 else "회계 이익 괴리 낮음 (<20%)")
    else:
        add_item("GAAP vs Non-GAAP 격차", NO_DATA, None, NO_DATA)
    # 4) 현금 전환율 (OCF / Net Income) 및 투자기 현금흐름 예외
    is_investment_phase_fcf = False
    if need("영업현금흐름", q_data["ocf_ttm"]) and need("순이익(합산)", q_data["net_income_ttm"]):
        ocf_conv = q_data["ocf_ttm"] / max(abs(q_data["net_income_ttm"]), 1e-6)
        capex = _num(q_data["capex_ttm"])
        fcf_ttm = (q_data["ocf_ttm"] - capex) if capex is not None else None
        ocf_prev = _num(q_data["ocf_prev_ttm"])
        ocf_growing = (q_data["ocf_ttm"] > ocf_prev) if ocf_prev is not None else None
        if ocf_growing is not None and fcf_ttm is not None:
            is_investment_phase_fcf = bool(ocf_growing and ocf_conv >= 1.0 and fcf_ttm < 0)
        ocf_warn = (ocf_conv < 0.80) and not is_investment_phase_fcf
        fcf_txt = (f" (FCF {'+' if fcf_ttm >= 0 else '−'}{_s_big(abs(fcf_ttm), is_krx)})" if fcf_ttm is not None
                   else f" (FCF {NO_DATA}: 설비투자 데이터 없음)")
        note = "" if ocf_growing is not None else " · 영업CF 전년 비교 없음(투자기 예외 미적용)"
        add_item("현금 전환율 (영업CF/순이익)", f"{ocf_conv:.2f}배{fcf_txt}", ocf_warn,
                 ("G1 초기 가속 투자의 정상적 과정 (면제)" if is_investment_phase_fcf
                  else ("장부상 이익 위험 (<0.8)" if ocf_warn else "현금 창출력 우수")) + note)
    else:
        add_item("현금 전환율 (영업CF/순이익)", NO_DATA, None, NO_DATA)
    # 5) 매출 1단위당 소요 자본
    reinv = q_data.get("reinvest_per_dollar_rising")
    if reinv is None:
        missing.append("매출 1단위당 소요 자본 추세")
        add_item("매출 1단위당 소요 자본", NO_DATA, None, NO_DATA)
    else:
        add_item("매출 1단위당 소요 자본", "상승 추세" if reinv else "안정적 유지", bool(reinv),
                 "성장의 한계비용 상승 경고" if reinv else "자본 효율성 유지")

    inv_warn = bool(inv_diff is not None and inv_diff > 0.05)
    axis4_score: Optional[int] = None
    if n_checked >= 2:   # 점검 가능한 항목이 2개 미만이면 점수를 내지 않는다
        axis4_score = 2 if warn_cnt == 0 else (1 if warn_cnt == 1 else 0)

    # -----------------------------------------------------------------
    # E7. 미래 먹거리 측정 (RPO vs 매출 증가율 & 증분 영업이익률)
    # -----------------------------------------------------------------
    sus = d["sustainability"]
    inc_opm = None
    if need("분기 영업이익·매출(올해·전년 동기)", sus["op_t"], sus["op_t_minus_4"], sus["sales_t"], sus["sales_t_minus_4"]):
        delta_op = sus["op_t"] - sus["op_t_minus_4"]
        delta_sales = sus["sales_t"] - sus["sales_t_minus_4"]
        inc_opm = (delta_op / delta_sales) if abs(delta_sales) > 1e-6 else _num(ac["opm_actual"])
    avg_opm = _num(ac["opm_actual"])
    rpo_yoy = _num(sus.get("rpo_yoy_growth"))
    if rpo_yoy is None:
        missing.append("수주잔고(RPO) 증가율")

    op_leverage_active: Optional[bool] = (inc_opm > avg_opm) if (inc_opm is not None and avg_opm is not None) else None
    rpo_outpacing_sales: Optional[bool] = (rpo_yoy > sales_yoy) if (rpo_yoy is not None and sales_yoy is not None) else None

    axis3_score: Optional[int] = None
    if op_leverage_active is not None:
        axis3_score = (1 if op_leverage_active else 0) + (1 if rpo_outpacing_sales else 0)

    # -----------------------------------------------------------------
    # 3장. 실적 생애주기 5단계 (G1 ~ G5) 및 사이클 단계 (C1 ~ C4) 분류 — 근거가 있는 조건만 참으로 본다
    # -----------------------------------------------------------------
    th = d["thesis"]
    p_tr, q_tr = th.get("p_trend"), th.get("q_trend")
    pq_both_down = bool(p_tr == "하락" and q_tr == "하락")
    eps_yoy_v = _num(ac["eps_yoy"])
    regime_hold_reason: Optional[str] = None
    n_axes = sum(1 for s in (axis1_score, axis2_score, axis3_score, axis4_score, axis5_score) if s is not None)

    if th.get("hard_block_triggered") or (pq_both_down and g_hist == "miss_2q" and rpo_yoy is not None and rpo_yoy < 0):
        e_regime, e_regime_label, peak_out_warning = "G5", "구조적 쇠퇴 / 어닝 쇼크 (Structural Decline)", True
    elif margin_squeeze_detected or (op_leverage_active is False and growth_accel == "감속") or (delta_30 is not None and delta_30 < -0.005):
        e_regime, e_regime_label, peak_out_warning = "G4", "피크아웃 / 성장 둔화 (Peak-out Risk)", True
    elif growth_accel == "가속" and op_leverage_active and rpo_outpacing_sales and delta_30 is not None and delta_30 > 0:
        e_regime, e_regime_label, peak_out_warning = "G1", "초기 가속 / 턴어라운드 (Emerging Growth)", False
    elif (sales_yoy is not None and sales_yoy >= 0.10 and eps_yoy_v is not None and eps_yoy_v >= 0.10
          and n_checked >= 2 and warn_cnt == 0 and op_leverage_active):
        e_regime, e_regime_label, peak_out_warning = "G2", "우량 지속 성장 (Quality Compounder)", False
    elif n_axes >= 2 and sales_yoy is not None and sales_yoy < 0.10:
        # G3는 저성장 근거(매출 성장 10% 미만)가 있을 때만. 근거 없이 '나머지 전부'를 성숙기로 몰지 않는다.
        e_regime, e_regime_label, peak_out_warning = "G3", "성숙 / 현금 회수 (Mature Cash Cow)", False
    elif sales_yoy is not None and sales_yoy >= 0.10:
        _fail = []
        if eps_yoy_v is None:
            _fail.append(f"EPS 성장률 {NO_DATA}")
        elif eps_yoy_v < 0.10:
            _fail.append(f"EPS 성장 {eps_yoy_v*100:+.0f}% (<10%)")
        if warn_cnt > 0:
            _fail.append(f"이익의 질 경고 {warn_cnt}개")
        if n_checked < 2:
            _fail.append(f"이익의 질 점검 {n_checked}개 (<2)")
        if not op_leverage_active:
            _fail.append("고정비 레버리지 " + (NO_DATA if op_leverage_active is None else "미작동"))
        regime_hold_reason = (f"고성장(매출 {sales_yoy*100:+.0f}%)이나 G2 조건 미충족: " + ", ".join(_fail)) if _fail else None
        e_regime, e_regime_label, peak_out_warning = "NA", f"성장 단계 판정 보류 — {regime_hold_reason or '근거 부족'}", False
    else:
        e_regime, e_regime_label, peak_out_warning = "NA", f"판정 불가 ({NO_DATA}: " + (
            "매출 성장률" if (n_axes >= 2 and sales_yoy is None) else f"판정 가능한 축 {n_axes}개") + ")", False

    cycle_stage = None
    if sector_type == "사이클형":
        if p_tr == "하락" and q_tr == "상승":
            cycle_stage = "C1 바닥 통과 (Q 출하 반등)"
        elif p_tr == "상승" and q_tr == "상승":
            cycle_stage = "C3 정점 과열 (재고 급증 경고)" if inv_warn else "C2 확장 (P·Q 동반 상승)"
        elif p_tr is None or q_tr is None:
            cycle_stage = f"{NO_DATA} (P·Q 추세 미입력)"
        else:
            cycle_stage = "C4 하강 (P·Q 동반 둔화)"

    # -----------------------------------------------------------------
    # 4장. 5대 축 가중합산 — 판정 가능한 축의 가중치로 정규화
    # -----------------------------------------------------------------
    axis_w = [(axis1_score, 0.20), (axis2_score, 0.25), (axis3_score, 0.25), (axis4_score, 0.15), (axis5_score, 0.15)]
    w_sum = sum(w for s, w in axis_w if s is not None)
    weighted_score: Optional[float] = (round(sum(s * w for s, w in axis_w if s is not None) / w_sum, 2) if w_sum > 0 else None)
    coverage = round(w_sum, 2)

    if th.get("hard_block_triggered") or e_regime == "G5":
        final_grade, thesis_status, real_signal = "🔴", "붕괴", "약"
    elif weighted_score is None or coverage < 0.40:
        final_grade, thesis_status, real_signal = "⚪", "미확인", "중립"
    elif weighted_score < 1.0 or e_regime == "G4":
        final_grade, thesis_status, real_signal = "🔴", "경계", "약"
    elif weighted_score >= 1.6 and e_regime in ["G1", "G2"]:
        final_grade, thesis_status, real_signal = "🟢", "유효", "강"
    else:
        final_grade, thesis_status = "🟡", ("유효" if weighted_score >= 1.2 else "경계")
        real_signal = "중립"

    if e_regime in ["G1", "G2"] and inc_opm is not None:
        headline = (f"증분 OPM {inc_opm*100:.1f}%로 영업레버리지 작동"
                    + (f" · RPO +{rpo_yoy*100:.0f}%" if rpo_yoy is not None else " (RPO 데이터 없음)"))
    elif e_regime == "G4":
        headline = f"피크아웃 경고 — {'매출 상회 속 마진 압착 발생' if margin_squeeze_detected else '30일 EPS 리비전 모멘텀 둔화 포착'}"
    elif e_regime == "G5":
        headline = "핵심 이익 드라이버 붕괴 또는 P·Q 동반 역성장 발동 (신규 진입 차단)"
    elif e_regime == "NA" and regime_hold_reason:
        headline = f"{regime_hold_reason} — 성장 단계를 단정하지 않고 V 트랙은 중립 규칙으로 계산"
    elif e_regime == "NA":
        headline = f"실적 판정에 필요한 데이터가 부족합니다 ({NO_DATA} {len(missing)}건)"
    else:
        headline = (f"성숙기 현금 창출 구간 (평균 OPM {_s_pct(avg_opm, signed=False)}, 90일 리비전 {_s_pct(delta_90)})")

    # E8-2. 시장의 이익 오판 포인트 — 근거 데이터가 있을 때만 서술
    if rpo_outpacing_sales and op_leverage_active:
        mispricing_type = "시장 과소평가 포인트 (성장 잠재력)"
        mispricing_detail = (f"RPO 증가율(+{rpo_yoy*100:.0f}%)이 매출 성장률(+{sales_yoy*100:.0f}%)을 상회하고 "
                             f"증분 OPM({inc_opm*100:.1f}%)이 평균 OPM({avg_opm*100:.1f}%)을 웃돕니다.")
    elif peak_out_warning:
        mispricing_type = "시장 과대평가 포인트 (저무는 기업 경고)"
        mispricing_detail = "마진 압착 또는 단기 30일 리비전 하향 변곡점이 확인되어 컨센서스 하향 조정 가능성이 있습니다."
    elif e_regime == "NA" and regime_hold_reason:
        mispricing_type, mispricing_detail = "판정 보류", regime_hold_reason
    elif e_regime == "NA":
        mispricing_type, mispricing_detail = NO_DATA, NO_DATA
    else:
        mispricing_type = "컨센서스 부합 구간"
        mispricing_detail = "확인 가능한 실적 궤적과 시장 기대치에 뚜렷한 괴리가 없습니다."

    acc_note = {"analysts": lambda v: (f"{int(v)}명" if _num(v) is not None else NO_DATA)}
    return {
        "regime": {
            "earnings_regime": e_regime, "regime_label": e_regime_label, "sector_type": sector_type,
            "cycle_stage": cycle_stage, "growth_acceleration": growth_accel if growth_accel else NO_DATA,
        },
        "verdict": {
            "score": weighted_score, "coverage": coverage, "axes_scored": n_axes,
            "grade": final_grade, "real_signal": real_signal,
            "headline_reason": headline, "peak_out_warning": peak_out_warning,
        },
        "projections": {
            "eps_basis": eps_basis, "eps_basis_label": eps_basis_label, "ttm_eps": ttm_eps,
            "eps_1y": {"formula": f_1y, "base": eps_1y["base"], "bull": eps_1y["bull"], "bear": eps_1y["bear"],
                       "cycle_bear": cycle_bear_1y},
            "eps_2y": {"formula": f_2y, "base": eps_2y["base"], "bull": eps_2y["bull"], "bear": eps_2y["bear"]},
            "certainty_adjustment": certainty_adj, "certainty_label": certainty_label,
        },
        "sustainability_metrics": {
            "incremental_opm": round(inc_opm, 4) if inc_opm is not None else None,
            "average_opm": round(avg_opm, 4) if avg_opm is not None else None,
            "rpo_yoy_growth": round(rpo_yoy, 4) if rpo_yoy is not None else None,
            "sales_yoy_growth": round(sales_yoy, 4) if sales_yoy is not None else None,
            "operating_leverage_active": op_leverage_active,
            "design_wins_note": sus.get("design_wins_note") or NO_DATA,
        },
        "revision_dynamics": {
            "status": rev_status,
            "change_90d": round(delta_90, 4) if delta_90 is not None else None,
            "change_30d": round(delta_30, 4) if delta_30 is not None else None,
            "inflection_flag": inflection_flag,
            "table": [
                {"period": "현재 (t)", "ntm_eps": _s_price(rev["ntm_t"], is_krx), "fy1_eps": _s_price(rev["fy1_t"], is_krx),
                 "fy2_eps": _s_price(rev["fy2_t"], is_krx), "analysts": acc_note["analysts"](rev["analysts_t"])},
                {"period": "30일 전 (t-30)", "ntm_eps": _s_price(rev["ntm_30d"], is_krx), "fy1_eps": _s_price(rev["fy1_30d"], is_krx),
                 "fy2_eps": _s_price(rev["fy2_30d"], is_krx), "analysts": acc_note["analysts"](rev["analysts_30d"])},
                {"period": "90일 전 (t-90)", "ntm_eps": _s_price(rev["ntm_90d"], is_krx), "fy1_eps": _s_price(rev["fy1_90d"], is_krx),
                 "fy2_eps": _s_price(rev["fy2_90d"], is_krx), "analysts": acc_note["analysts"](rev["analysts_90d"])},
            ],
        },
        "quality_checks": {
            "warning_count": warn_cnt, "checked_count": n_checked, "basis": q_data.get("basis") or NO_DATA,
            "is_investment_phase_fcf": is_investment_phase_fcf,
            "margin_squeeze_detected": margin_squeeze_detected, "margin_squeeze_known": margin_squeeze_known,
            "items": quality_items,
        },
        "thesis_driver": {
            "status": thesis_status,
            "invalidation_condition": th.get("invalidation_condition") or NO_DATA,
            "current_status": th.get("current_status") or NO_DATA,
            "capital_structure_note": th.get("capital_structure_note") or NO_DATA,
            "mispricing_type": mispricing_type, "mispricing_detail": mispricing_detail,
        },
        "evaluation_axes": [
            {"axis": "축 1: 성장 가속도 (E1)", "weight": "20%", "score": axis1_score,
             "status": (f"2차 미분 [{growth_accel}] ({traj_text})" if growth_accel else NO_DATA)},
            {"axis": "축 2: 리비전 변곡점 (E3)", "weight": "25%", "score": axis2_score,
             "status": (f"{rev_status} (90일 {_s_pct(delta_90)} / 30일 {_s_pct(delta_30)})" if axis2_score is not None else NO_DATA)},
            {"axis": "축 3: 미래 지속성·레버리지 (E7)", "weight": "25%", "score": axis3_score,
             "status": (f"증분OPM {inc_opm*100:.1f}% vs 평균 {avg_opm*100:.1f}% · RPO "
                        + (f"+{rpo_yoy*100:.0f}%" if rpo_yoy is not None else f"{NO_DATA}(RPO 없이 최대 1점)")
                        if axis3_score is not None else NO_DATA)},
            {"axis": "축 4: 질 및 현금 전환 (E6)", "weight": "15%", "score": axis4_score,
             "status": (f"경고 {warn_cnt}개 / 점검 {n_checked}개" + (" (투자기 FCF 감소 면제)" if is_investment_phase_fcf else "")
                        if axis4_score is not None else f"{NO_DATA} (점검 가능 항목 {n_checked}개)")},
            {"axis": "축 5: 경영진 가이드 신뢰 (E2)", "weight": "15%", "score": axis5_score,
             "status": (f"{certainty_label} (할인율 {certainty_adj*100:+.0f}%p)" if axis5_score is not None else certainty_label)},
        ],
        "e1_surprise_table": e1_table,
        "data_quality": {
            "source": d["meta"].get("source"), "as_of": d["meta"].get("as_of"),
            "eps_basis": eps_basis, "eps_basis_label": eps_basis_label,
            "missing": sorted(set(missing)), "notes": d["meta"].get("notes", []),
        },
    }
