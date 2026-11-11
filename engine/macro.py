"""거시 트랙(Track 4): FRED·ECOS 수집(fetch_macro_levels), 10년 국채 금리(fetch_risk_free_10y), analyze_macro_track.

engine.py를 기능별로 나눈 모듈. 바깥에서는 `import engine` 후 `engine.이름`으로 쓴다(engine/__init__.py).
"""
import hashlib
import math
import os
import re
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

from .common import NO_DATA
from .earnings import _read_env_value

# =====================================================================
# [Track 4] 거시적 분석 트랙 v1.0 (가설 등록·잔차화 회귀·OOS 검증 엔진)
# =====================================================================

# 8절. 확정 파라미터 요약표 (12개 항목 Lock)
MACRO_PARAMS: Dict[str, Any] = {
    "USE_EMPIRICAL_REGRESSION": True,    # 1: 가설 등록 + HAC/Huber 실측[cite: 5]
    "REG_WEEKS_MAIN": 104,               # 2a: 주력 회귀 기간 (104주)[cite: 5]
    "REG_WEEKS_SUB": 52,                 # 2b: 비교 회귀 기간 (52주)[cite: 5]
    "REGIME_BETA_NOISE": 0.001,          # 국면별 β 부호 반전 경고 하한 (1σ당 ±0.10%p 미만은 노이즈로 보고 경고 안 함)
    "REG_WEEKS_MIN": 78,                 # 2c: 최소 유효 주간 표본 (78주)[cite: 5]
    "OOS_HORIZON_WEEKS": 13,             # 최근 점검 및 OOS 이동 원점 구간 (13주)[cite: 5]
    "RESIDUALIZE_CORR_THRES": 0.50,      # 3: 시장 동행 잔차화 상관계수 임계값 (|ρ| >= 0.50)[cite: 5]
    "MAX_HYPOTHESIS_VARS": 3,            # 4: 가설 변수 최대 개수 (3개)[cite: 5]
    "HAC_MAXLAGS": 4,                    # 5: Newey-West HAC 최대 시차 (4주 고정)[cite: 5]
    "RECENT_SURGE_SHARE_4W": 0.50,       # 6: 최근 급변 판정 기준 (4주/13주 몫 50% 이상)[cite: 5]
    "R2_DOMINANT": 0.40,                 # 7a: 거시 지배 R^2 경계 (0.40)[cite: 5]
    "R2_MIXED": 0.15,                    # 7b: 거시 혼합 R^2 경계 (0.15)[cite: 5]
    "REGIME_INFL_HIGH": 0.60,            # 8a: 인플레 주도 국면 기준 (60% 이상)[cite: 5]
    "REGIME_INFL_LOW": 0.40,             # 8b: 성장·정책 주도 국면 기준 (40% 이하)[cite: 5]
    "OUTLIER_BETA_DIFF_TOL": 0.50,       # 9: OLS vs Huber 괴리 허용치 (50%)[cite: 5]
    "OOS_RESID_Z_THRES": 2.0,            # 10: 표본 외 잔차 이탈 기준 (|z_oos| >= 2.0, 최소 6주)[cite: 5]
    "OOS_MIN_NORMAL_WEEKS": 6,           # 10b: 평범한 주 최소 표본 (6주)[cite: 5]
    "OOS_MIN_RUNS": 16,                  # 11a: 사후 검증 최소 구간 수[cite: 5]
    "OOS_WIN_RATE": 0.60,                # 11b: ΔR^2_oos > 0 비율 기준 (60%)[cite: 5]
    "ENABLE_SIZE_CAP_LINK": False,       # 12: 포지션 비중 상한 자동 연동 비활성[cite: 5]
}

# 3-1. 산업 기본 노출 매트릭스 (자동 가설 등록 상위 3변수 v0_auto 소스)[cite: 5]
INDUSTRY_MACRO_HYPOTHESIS_MAP: Dict[str, List[Dict[str, str]]] = {
    "메모리 (한국 상장)": [
        {"var_id": "ICSA", "var_name": "미국 신규 실업수당 청구 (경기)", "expected_sign": "-", "strength": "●●●", "unit": "pct", "dup": None},
        {"var_id": "DEXKOUS", "var_name": "원/달러 환율", "expected_sign": "±", "strength": "●●", "unit": "pct", "dup": "실적 트랙 E4 (환율 드라이버)"},
        {"var_id": "KR10Y", "var_name": "국고채 10년물 금리 (817Y002)", "expected_sign": "-", "strength": "●", "unit": "bp", "dup": "펀더멘탈 트랙 F7 (요구수익률 r)"},
    ],
    "AI 반도체·네트워킹(팹리스)": [
        {"var_id": "DFII10", "var_name": "미국 실질 10Y 금리 (TIPS)", "expected_sign": "-", "strength": "●●", "unit": "bp", "dup": "펀더멘탈 트랙 F7 (요구수익률 r)"},
        {"var_id": "BAMLC0A0CM", "var_name": "미국 투자등급(IG) 회사채 스프레드", "expected_sign": "-", "strength": "●●", "unit": "bp", "dup": None},
        {"var_id": "NETLIQ", "var_name": "연준 순유동성 (WALCL-TGA-RRP)", "expected_sign": "+", "strength": "●●", "unit": "pct", "dup": None},
    ],
    "반도체 장비": [
        {"var_id": "ICSA", "var_name": "미국 신규 실업수당 청구 (경기)", "expected_sign": "-", "strength": "●●●", "unit": "pct", "dup": None},
        {"var_id": "DFII10", "var_name": "미국 실질 10Y 금리 (TIPS)", "expected_sign": "-", "strength": "●●", "unit": "bp", "dup": "펀더멘탈 트랙 F7 (요구수익률 r)"},
        {"var_id": "BAMLC0A0CM", "var_name": "미국 투자등급(IG) 회사채 스프레드", "expected_sign": "-", "strength": "●●", "unit": "bp", "dup": None},
    ],
    "고멀티플 SW·플랫폼": [
        {"var_id": "DFII10", "var_name": "미국 실질 10Y 금리 (TIPS)", "expected_sign": "-", "strength": "●●●", "unit": "bp", "dup": "펀더멘탈 트랙 F7 (요구수익률 r)"},
        {"var_id": "BAMLH0A0HYM2", "var_name": "미국 하이일드(HY) 회사채 스프레드", "expected_sign": "-", "strength": "●●", "unit": "bp", "dup": None},
        {"var_id": "NETLIQ", "var_name": "연준 순유동성 (WALCL-TGA-RRP)", "expected_sign": "+", "strength": "●●", "unit": "pct", "dup": None},
    ],
    "은행·금융": [
        {"var_id": "BAMLH0A0HYM2", "var_name": "미국 하이일드(HY) 회사채 스프레드", "expected_sign": "-", "strength": "●●●", "unit": "bp", "dup": None},
        {"var_id": "BAMLC0A0CM", "var_name": "미국 투자등급(IG) 회사채 스프레드", "expected_sign": "-", "strength": "●●", "unit": "bp", "dup": None},
        {"var_id": "T10Y2Y", "var_name": "장단기 금리차 (10Y-2Y)", "expected_sign": "+", "strength": "●", "unit": "bp", "dup": None},
    ],
    "방산 (한국)": [
        {"var_id": "DEXKOUS", "var_name": "원/달러 환율", "expected_sign": "+", "strength": "●●", "unit": "pct", "dup": "실적 트랙 E4 (환율 드라이버)"},
        {"var_id": "KR10Y", "var_name": "국고채 10년물 금리 (817Y002)", "expected_sign": "-", "strength": "●", "unit": "bp", "dup": "펀더멘탈 트랙 F7 (요구수익률 r)"},
        {"var_id": "DCOILWTICO", "var_name": "WTI 원유 현물가", "expected_sign": "+", "strength": "●", "unit": "pct", "dup": None},
    ],
    "에너지·정유": [
        {"var_id": "DCOILWTICO", "var_name": "WTI 원유 현물가", "expected_sign": "+", "strength": "●●●", "unit": "pct", "dup": "실적 트랙 E4 (원자재 드라이버)"},
        {"var_id": "ICSA", "var_name": "미국 신규 실업수당 청구 (경기)", "expected_sign": "-", "strength": "●●", "unit": "pct", "dup": None},
        {"var_id": "BAMLH0A0HYM2", "var_name": "미국 하이일드(HY) 회사채 스프레드", "expected_sign": "-", "strength": "●●", "unit": "bp", "dup": None},
    ],
}


def _ensure_macro_hypothesis_file(ticker: str, industry_category: str) -> Tuple[str, List[Dict[str, str]]]:
    """
    3-3. 가설 파일 불변성(Immutable Lock)과 자동 생성 Fallback (v1.0 확정)[cite: 5]:
    - 수동 가설이 없으면 회귀 연산 직전에 data/macro/hypotheses/<티커>_v0_auto.yaml 자동 생성 후 SHA-256 해시 기록[cite: 5].
    """
    hyp_vars = INDUSTRY_MACRO_HYPOTHESIS_MAP.get(
        industry_category,
        INDUSTRY_MACRO_HYPOTHESIS_MAP["메모리 (한국 상장)" if ticker.upper().endswith((".KS", ".KQ")) else "AI 반도체·네트워킹(팹리스)"]
    )[:MACRO_PARAMS["MAX_HYPOTHESIS_VARS"]]

    hyp_dir = os.path.join("data", "macro", "hypotheses")
    safe_ticker = re.sub(r"[^A-Za-z0-9_.-]", "_", ticker.upper())
    hyp_id = f"{safe_ticker}_v0_auto"
    hyp_path = os.path.join(hyp_dir, f"{hyp_id}.yaml")

    try:
        os.makedirs(hyp_dir, exist_ok=True)
        if not os.path.exists(hyp_path):
            content_lines = [
                f"hypothesis_id: {hyp_id}",
                f"ticker: {ticker.upper()}",
                f"industry_category: {industry_category}",
                f"created_at: {datetime.now().isoformat()}",
                "variables:",
            ]
            for v in hyp_vars:
                content_lines.append(f"  - var_id: {v['var_id']}")
                content_lines.append(f"    expected_sign: '{v['expected_sign']}'")
                content_lines.append(f"    strength: '{v['strength']}'")
            raw_text = "\n".join(content_lines)
            sha_hash = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()[:16]
            with open(hyp_path, "w", encoding="utf-8") as f_out:
                f_out.write(raw_text + f"\nsha256_lock: {sha_hash}\n")
    except Exception:
        pass

    return hyp_id, hyp_vars


def _sanitize_macro_comment(text: str) -> str:
    """
    6-4. 금지 문구 필터 (정규식 차단)[cite: 5]:
    미래 예측·인과 단정·행동 유예 표현("~할 것이다", "순풍", "역풍", "~때문에", "지켜보자/기다린다") 자동 치환[cite: 5].
    """
    replacements = [
        (r"순풍", "우호적 동행 구간"),
        (r"역풍", "비우호적 동행 구간"),
        (r"때문에\s*(빠졌다|올랐다|하락했다|상승했다)", "변화와 같은 주간에 동행했다"),
        (r"할\s*것이다", "해 왔다"),
        (r"하기\s*어렵다", "한 경향이 있었다"),
        (r"거시가\s*불안하니\s*(지켜보자|기다린다)", "거시 발표 일정을 감시 목록에 둔다"),
    ]
    out = text
    for pat, rep in replacements:
        out = re.sub(pat, rep, out)
    return out


def _ols_hac_and_huber(y: np.ndarray, X: np.ndarray, maxlags: int = 4) -> Dict[str, np.ndarray]:
    """
    4-2. OLS 추정 + Newey-West HAC 표준오차(maxlags=4 고정) + Huber M-추정 로버스트 회귀(IRLS)[cite: 5].
    외부 패키지 버전 충돌 없이 순수 NumPy 행렬 연산으로 정확한 HAC 공분산과 Huber 계수를 산출.
    """
    n, k = X.shape
    if n <= k + 2:
        zeros = np.zeros(k)
        return {"beta": zeros, "t_hac": zeros, "beta_huber": zeros, "r2": 0.0, "resid": np.zeros(n)}

    # 1) OLS 계수 및 결정계수 R^2
    xtx = X.T @ X + np.eye(k) * 1e-8
    xtx_inv = np.linalg.inv(xtx)
    beta_ols = xtx_inv @ (X.T @ y)
    y_hat = X @ beta_ols
    resid = y - y_hat
    tss = float(np.sum((y - np.mean(y)) ** 2))
    rss = float(np.sum(resid ** 2))
    r2 = max(0.0, min(1.0, 1.0 - (rss / max(tss, 1e-12))))

    # 2) Newey-West HAC 공분산 행렬 (Bartlett 커널, maxlags=4)[cite: 5]
    u = resid.reshape(-1, 1)
    xu = X * u  # (n, k)
    S = xu.T @ xu
    L = min(maxlags, n - 2)
    for lag in range(1, L + 1):
        weight = 1.0 - (lag / (L + 1.0))
        gamma_l = xu[lag:].T @ xu[:-lag]
        S = S + weight * (gamma_l + gamma_l.T)
    dof_scale = n / max(n - k, 1)
    cov_hac = dof_scale * (xtx_inv @ S @ xtx_inv)
    se_hac = np.sqrt(np.maximum(np.diag(cov_hac), 1e-12))
    t_hac = beta_ols / se_hac

    # 3) Huber M-추정 로버스트 회귀 (IRLS 15회 반복, c = 1.345 * MAD)[cite: 5]
    beta_hub = beta_ols.copy()
    for _ in range(15):
        r_h = y - (X @ beta_hub)
        mad = float(np.median(np.abs(r_h - np.median(r_h)))) / 0.6745
        scale = max(mad, 1e-6)
        c_cut = 1.345 * scale
        abs_r = np.abs(r_h)
        w = np.where(abs_r <= c_cut, 1.0, c_cut / np.maximum(abs_r, 1e-8))
        W_X = X * w.reshape(-1, 1)
        xtwx_inv = np.linalg.inv(X.T @ W_X + np.eye(k) * 1e-8)
        beta_hub = xtwx_inv @ (W_X.T @ y)

    return {
        "beta": beta_ols,
        "t_hac": t_hac,
        "beta_huber": beta_hub,
        "r2": round(float(r2), 4),
        "resid": resid,
    }


def _residualize_against_market(
    var_arr: np.ndarray,
    mkt_arr: np.ndarray,
    train_end_idx: int,
    corr_thres: float = 0.50
) -> Tuple[np.ndarray, bool, float]:
    """
    4-1. 시장 동행 변수의 잔차화 (Frisch-Waugh-Lovell & OOS 미래 정보 누수 차단)[cite: 5]:
    - 잔차화 상관계수 ρ와 회귀계수(a_hat, c_hat)는 반드시 점검 구간을 제외한 과거 추정 구간(:train_end_idx) 안에서만 추정[cite: 5].
    """
    v_train = var_arr[:train_end_idx]
    m_train = mkt_arr[:train_end_idx]
    if len(v_train) < 10 or np.std(v_train) < 1e-10 or np.std(m_train) < 1e-10:
        return var_arr.copy(), False, 0.0

    corr = float(np.corrcoef(v_train, m_train)[0, 1])
    if abs(corr) < corr_thres:
        return var_arr.copy(), False, round(corr, 2)

    X_m_train = np.column_stack([np.ones(len(m_train)), m_train])
    coef = np.linalg.lstsq(X_m_train, v_train, rcond=None)[0]
    a_hat, c_hat = float(coef[0]), float(coef[1])

    # 전체 구간에 과거 추정 계수(a_hat, c_hat) 대입 -> 표본 외(OOS) 미래 정보 누수 완전 차단[cite: 5]
    u_all = var_arr - (a_hat + c_hat * mkt_arr)
    return u_all, True, round(corr, 2)


# =====================================================================
# 거시 트랙 실데이터 입력층 (FRED · 한국은행 ECOS)
# - 수준(Level) 시계열을 받아 주간 금요일(W-FRI)로 정렬한다. 변화량(Δbp, Δln)은 analyze_macro_track에서 계산.
# - FRED: .env FRED_API_KEY가 있으면 공식 API(JSON), 없거나 실패하면 공개 CSV(fredgraph). 한국 국고채 10년물은 ECOS 817Y002.
# - 못 받은 변수는 결과에서 빠지고(missing) 회귀·화면에서 '데이터 없음'. 난수·고정 수준값으로 대체하지 않는다.
# =====================================================================
MACRO_SERIES_SPEC: Dict[str, Dict[str, Any]] = {
    # kind: bp(금리·스프레드, 수준 % → 변화 bp) | ln(가격·환율·건수, 로그 변화)
    "DFII10":       {"src": "fred", "ids": ["DFII10"],       "kind": "bp", "level_unit": "%",     "level_mult": 1.0,  "name": "미국 실질 10Y 금리"},
    "T10YIE":       {"src": "fred", "ids": ["T10YIE"],       "kind": "bp", "level_unit": "%",     "level_mult": 1.0,  "name": "미국 10Y 기대인플레이션"},
    "T10Y2Y":       {"src": "fred", "ids": ["T10Y2Y"],       "kind": "bp", "level_unit": "bp",    "level_mult": 100.0, "name": "장단기 금리차"},
    "BAMLC0A0CM":   {"src": "fred", "ids": ["BAMLC0A0CM"],   "kind": "bp", "level_unit": "bp",    "level_mult": 100.0, "name": "IG 회사채 스프레드"},
    "BAMLH0A0HYM2": {"src": "fred", "ids": ["BAMLH0A0HYM2"], "kind": "bp", "level_unit": "bp",    "level_mult": 100.0, "name": "HY 회사채 스프레드"},
    "DEXKOUS":      {"src": "fred", "ids": ["DEXKOUS"],      "kind": "ln", "level_unit": "원",    "level_mult": 1.0,  "name": "원/달러 환율"},
    "DCOILWTICO":   {"src": "fred", "ids": ["DCOILWTICO"],   "kind": "ln", "level_unit": "달러",  "level_mult": 1.0,  "name": "WTI 유가"},
    "ICSA":         {"src": "fred", "ids": ["ICSA"],         "kind": "ln", "level_unit": "만 건", "level_mult": 1e-4, "name": "신규 실업수당 청구"},
    # 순유동성 = WALCL(백만$) − WTREGEN(백만$) − RRPONTSYD(십억$ ×1000) → 표시 '조 달러'
    "NETLIQ":       {"src": "fred", "ids": ["WALCL", "WTREGEN", "RRPONTSYD"], "kind": "ln", "level_unit": "조 달러", "level_mult": 1e-6, "name": "연준 순유동성"},
    "KR10Y":        {"src": "ecos", "stat": "817Y002", "item": "010210000", "kind": "bp", "level_unit": "%", "level_mult": 1.0, "name": "국고채 10년"},
}
_MACRO_SERIES_CACHE: Dict[str, Tuple[float, Any]] = {}
_MACRO_TTL_SEC = 6 * 3600


def _http_get_text(url: str, timeout: float = 10.0) -> Optional[str]:
    import urllib.request
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read().decode("utf-8", "replace")
    except Exception:
        return None


def _fred_series(series_id: str, start: Optional[str] = None) -> Optional[pd.Series]:
    """FRED 일별/주별 관측치 (원 단위). 실패 시 None."""
    ck = f"fred:{series_id}:{start}"
    hit = _MACRO_SERIES_CACHE.get(ck)
    if hit and time.time() - hit[0] < _MACRO_TTL_SEC:
        return hit[1]
    ser = None
    key = _read_env_value("FRED_API_KEY")
    if key:
        import json as _json
        url = (f"https://api.stlouisfed.org/fred/series/observations?series_id={series_id}&api_key={key}&file_type=json"
               + (f"&observation_start={start}" if start else ""))
        txt = _http_get_text(url)
        try:
            obs = _json.loads(txt)["observations"] if txt else []
            vals = {o["date"]: float(o["value"]) for o in obs if o.get("value") not in (None, "", ".")}
            if vals:
                ser = pd.Series(vals)
        except Exception:
            ser = None
    if ser is None:
        import io as _io
        txt = _http_get_text(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}" + (f"&cosd={start}" if start else ""))
        if txt:
            try:
                df = pd.read_csv(_io.StringIO(txt))
                df = df[df.iloc[:, 1].astype(str).str.strip() != "."]
                ser = pd.Series(pd.to_numeric(df.iloc[:, 1], errors="coerce").values, index=df.iloc[:, 0].astype(str)).dropna()
                if start:
                    ser = ser[ser.index >= start]
                if ser.empty:
                    ser = None
            except Exception:
                ser = None
    if ser is not None:
        ser.index = pd.to_datetime(ser.index)
        ser = ser.sort_index()
        _MACRO_SERIES_CACHE[ck] = (time.time(), ser)
    return ser


def _ecos_series(stat: str, item: str, start: str, end: str) -> Optional[pd.Series]:
    """한국은행 ECOS 일별 통계 (예: 817Y002/010210000 국고채 10년, % 단위). 키가 없거나 실패 시 None."""
    ck = f"ecos:{stat}:{item}:{start}"
    hit = _MACRO_SERIES_CACHE.get(ck)
    if hit and time.time() - hit[0] < _MACRO_TTL_SEC:
        return hit[1]
    key = _read_env_value("ECOS_API_KEY")
    if not key:
        return None
    import json as _json
    url = f"https://ecos.bok.or.kr/api/StatisticSearch/{key}/json/kr/1/100000/{stat}/D/{start.replace('-', '')}/{end.replace('-', '')}/{item}"
    txt = _http_get_text(url, timeout=15.0)
    try:
        rows = _json.loads(txt)["StatisticSearch"]["row"] if txt else []
        vals = {r["TIME"]: float(r["DATA_VALUE"]) for r in rows if r.get("DATA_VALUE") not in (None, "", ".")}
        if not vals:
            return None
        ser = pd.Series(vals)
        ser.index = pd.to_datetime(ser.index, format="%Y%m%d")
        ser = ser.sort_index()
        _MACRO_SERIES_CACHE[ck] = (time.time(), ser)
        return ser
    except Exception:
        return None


def _to_weekly_fri(ser: pd.Series) -> pd.Series:
    """일별·주별 관측을 주간 금요일(W-FRI) 마지막 값으로 정렬 (휴일 등 최대 4영업일 ffill)."""
    s = ser.astype(float).sort_index()
    s = s.asfreq("B").ffill(limit=4) if len(s) > 1 and (s.index.to_series().diff().median() <= pd.Timedelta(days=3)) else s
    return s.resample("W-FRI").last()


def fetch_macro_levels(var_ids: List[str], start: str, end: Optional[str] = None) -> Dict[str, Any]:
    """거시 변수 주간 수준 시계열. 반환 {"levels": {var_id: Series(W-FRI, 원 단위)}, "missing": [...], "sources": {...}}."""
    end = end or datetime.now().strftime("%Y-%m-%d")
    levels: Dict[str, pd.Series] = {}
    sources: Dict[str, str] = {}
    missing: List[str] = []
    last_obs: Dict[str, str] = {}   # 실제 마지막 관측일 (주간 정렬 라벨은 그 주 금요일이라 미래 날짜가 될 수 있음)
    for vid in dict.fromkeys(var_ids):
        spec = MACRO_SERIES_SPEC.get(vid)
        if spec is None:
            missing.append(vid)
            continue
        try:
            if spec["src"] == "ecos":
                raw = _ecos_series(spec["stat"], spec["item"], start, end)
                if raw is None:
                    missing.append(vid)
                    continue
                w = _to_weekly_fri(raw)
                last_obs[vid] = str(raw.dropna().index[-1].date())
                sources[vid] = f"한국은행 ECOS {spec['stat']}/{spec['item']}"
            elif vid == "NETLIQ":
                parts = [_fred_series(i, start) for i in spec["ids"]]
                if any(p is None for p in parts):
                    missing.append(vid)
                    continue
                wa, tga, rrp = (_to_weekly_fri(p) for p in parts)
                w = (wa - tga - rrp * 1000.0).dropna()
                last_obs[vid] = str(min(p.dropna().index[-1] for p in parts).date())
                sources[vid] = "FRED WALCL−WTREGEN−RRPONTSYD×1000"
            else:
                raw = _fred_series(spec["ids"][0], start)
                if raw is None:
                    missing.append(vid)
                    continue
                w = _to_weekly_fri(raw)
                last_obs[vid] = str(raw.dropna().index[-1].date())
                sources[vid] = f"FRED {spec['ids'][0]}"
            w = w.dropna()
            w.index = _tz_naive_index(w.index)
            if spec["kind"] == "ln":
                w = w[w > 0]
            if len(w) < 10:
                missing.append(vid)
                sources.pop(vid, None)
                continue
            levels[vid] = w
        except Exception:
            missing.append(vid)
            sources.pop(vid, None)
    return {"levels": levels, "missing": missing, "sources": sources,
            "last_obs": {k: v for k, v in last_obs.items() if k in levels}}


def _tz_naive_index(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """시간대가 붙은 날짜(야후 주가: 뉴욕·서울 등)를 현지 날짜 그대로 시간대 없는 날짜로 (FRED·ECOS와 비교 가능하게)."""
    idx = pd.DatetimeIndex(idx)
    return idx.tz_localize(None) if idx.tz is not None else idx


def _weekly_changes(level: pd.Series, kind: str, idx: pd.DatetimeIndex) -> Optional[np.ndarray]:
    """주간 수준 → 주가 주간 인덱스(idx)에 맞춘 변화량 (bp: Δ×100, ln: Δlog). 결측이 5% 넘으면 None."""
    level = level.copy()
    level.index = _tz_naive_index(level.index)
    idx = _tz_naive_index(idx)
    lv = level.reindex(level.index.union(idx)).sort_index().ffill(limit=1)
    ch = (lv.diff() * 100.0) if kind == "bp" else np.log(lv).diff()
    ch = ch.reindex(idx)
    if ch.isna().mean() > 0.05:
        return None
    return ch.fillna(0.0).values.astype(float)


def _macro_hold_result(ticker: str, is_krx: bool, hyp_id: str, industry_category: str, reason: str,
                       sample_weeks: int = 0, missing_vars: Optional[List[str]] = None,
                       sources: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """거시 판정 보류 결과: 수치는 모두 None('데이터 없음'). 가짜 0·고정 국면을 넣지 않는다."""
    return {
        "status": "보류",
        "meta": {"ticker": ticker, "as_of": datetime.now().strftime("%Y-%m-%d"), "data_quality": "🔴",
                 "sample_weeks": sample_weeks, "hypothesis_id": hyp_id, "industry_category": industry_category,
                 "is_on_hold": True, "hold_reason": reason, "missing_vars": list(missing_vars or []),
                 "sources": dict(sources or {}), "macro_as_of": None},
        "classification": {"grade_icon": "⚪", "label": f"판정 보류 ({reason})", "r2_macro_own": None, "r2_raw": None,
                           "market_beta": None, "oos_delta_r2_mean": None, "warnings": []},
        "macro_regime": {"current_regime": NO_DATA, "inflation_week_ratio_13w": None, **_rf_fields(is_krx)},
        "sensitivities": [], "attribution_13w": {}, "residual_check": {}, "macro_state_13w": [], "watch_calendar": [],
        "comment": f"거시 회귀 판정을 보류한다: {reason}.",
    }


def analyze_macro_track(
    ticker: str = "005930.KS",
    stock_df: Optional[pd.DataFrame] = None,
    bench_df: Optional[pd.DataFrame] = None,
    industry_category: Optional[str] = None,
    earnings_output: Optional[Dict[str, Any]] = None,
    fundamental_output: Optional[Dict[str, Any]] = None,
    macro_raw_df: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:
    """
    거시적 분석 트랙 v1.0 메인 파이프라인[cite: 5]:
    - 2-2 주간 금요일(W-FRI) 정렬 및 4영업일 ffill 보정[cite: 5]
    - 3-3 가설 사전 등록(v0_auto) 자동 Fallback[cite: 5]
    - 4-1 ~ 4-5 OOS 잔차화, 모형 A/B/B_raw 추정(HAC maxlags=4 + Huber), 설명력 4단계 분류[cite: 5]
    - 5-1 ~ 5-2 13주 경로 및 인플레 vs 성장·정책 주도 국면 판정[cite: 5]
    - 6-1 ~ 6-3 최근 4주·13주 수익률 기여 분해, OOS 잔차 점검(z_oos), 이동 원점 ΔR^2_oos 검증[cite: 5]
    """
    is_krx = ticker.upper().endswith((".KS", ".KQ"))
    if industry_category is None:
        industry_category = "메모리 (한국 상장)" if is_krx else "AI 반도체·네트워킹(팹리스)"

    # 1) 3-3. 가설 사전 등록 확인 및 자동 생성 Fallback[cite: 5]
    hyp_id, hyp_vars = _ensure_macro_hypothesis_file(ticker, industry_category)

    # 2) 2-2. 주간 금요일(W-FRI) 시계열 정렬 (최대 4영업일 ffill 허용)[cite: 5]
    # 주가·지수 데이터가 없으면 판정 보류 (임의 시계열로 대체하지 않는다)
    if stock_df is None or len(stock_df) < 200:
        return _macro_hold_result(ticker, is_krx, hyp_id, industry_category, "주가 일봉 200개 미만 또는 데이터 없음")
    if bench_df is None or len(bench_df) < 200:
        return _macro_hold_result(ticker, is_krx, hyp_id, industry_category, "시장 지수 데이터 없음")
    s_close = pd.to_numeric(stock_df["Close"], errors="coerce").ffill(limit=4)
    s_close.index = _tz_naive_index(s_close.index)   # 야후 주가의 시간대 제거 (거시 데이터는 시간대 없음)
    w_stock = s_close.resample("W-FRI").last().dropna()
    b_close = pd.to_numeric(bench_df["Close"], errors="coerce").ffill(limit=4)
    b_close.index = _tz_naive_index(b_close.index)
    w_bench = b_close.resample("W-FRI").last().reindex(w_stock.index).ffill(limit=1)
    _ok = w_bench.notna()
    w_stock, w_bench = w_stock[_ok], w_bench[_ok]

    # 주간 로그수익률 계산[cite: 5]
    r_i_series = np.log(w_stock / w_stock.shift(1)).dropna()
    r_m_series = np.log(w_bench / w_bench.shift(1)).reindex(r_i_series.index).fillna(0.0)

    n_total = len(r_i_series)
    if n_total < MACRO_PARAMS["REG_WEEKS_MIN"]:
        return _macro_hold_result(ticker, is_krx, hyp_id, industry_category,
                                  f"유효 주간 표본 {n_total}주 < {MACRO_PARAMS['REG_WEEKS_MIN']}주", sample_weeks=n_total)

    # 최근 120주 슬라이스 (주력 104주 + OOS 13주 이동 원점 확보)[cite: 5]
    use_len = min(n_total, 120)
    r_i_sub = r_i_series.iloc[-use_len:]
    r_m_sub = r_m_series.iloc[-use_len:]
    w_idx = r_i_sub.index
    r_i = r_i_sub.values
    r_m = r_m_sub.values

    # 3) 거시 원자료: FRED·ECOS 실제 수준 시계열 → 주간 변화량 (금리·스프레드 Δbp, 가격·환율·건수·순유동성 Δln)[cite: 5]
    #    못 받은 변수는 회귀에서 빠진다. 국면 판정용 T10YIE·DFII10도 함께 받는다.
    _need = [v["var_id"] for v in hyp_vars] + ["T10YIE", "DFII10"]
    _start = (w_idx[0] - pd.Timedelta(days=40)).strftime("%Y-%m-%d")
    if macro_raw_df is not None:   # 외부에서 수준 시계열(열=var_id)을 직접 넣은 경우
        _mlv = {"levels": {c: _to_weekly_fri(macro_raw_df[c].dropna()) for c in macro_raw_df.columns if c in _need},
                "missing": [c for c in _need if c not in macro_raw_df.columns], "sources": {c: "입력 데이터" for c in macro_raw_df.columns}}
    else:
        _mlv = fetch_macro_levels(_need, _start)
    macro_levels: Dict[str, pd.Series] = {}
    raw_changes_map: Dict[str, np.ndarray] = {}
    for _vid, _lv in _mlv["levels"].items():
        _ch = _weekly_changes(_lv, MACRO_SERIES_SPEC[_vid]["kind"], w_idx) if _vid in MACRO_SERIES_SPEC else None
        if _ch is not None:
            raw_changes_map[_vid] = _ch
            macro_levels[_vid] = _lv
    macro_missing = sorted(set(_need) - set(raw_changes_map))
    macro_sources = {k: v for k, v in _mlv["sources"].items() if k in raw_changes_map}
    _lo = _mlv.get("last_obs") or {}
    # 실제 관측일 기준 (주간 라벨=금요일은 미래 날짜일 수 있어 쓰지 않음). 오늘 이후로는 표시하지 않는다.
    _today = datetime.now().strftime("%Y-%m-%d")
    macro_as_of = max((min(_lo.get(k, str(lv.index[-1].date())), _today) for k, lv in macro_levels.items()), default=None)

    avail_vars = [v for v in hyp_vars if v["var_id"] in raw_changes_map]
    missing_hyp = [v["var_name"] for v in hyp_vars if v["var_id"] not in raw_changes_map]
    if not avail_vars:
        return _macro_hold_result(ticker, is_krx, hyp_id, industry_category,
                                  "가설 거시 변수를 하나도 받지 못함 (FRED·ECOS 접속 또는 키 확인)",
                                  sample_weeks=use_len, missing_vars=missing_hyp, sources=macro_sources)

    # 4) 5-2. 거시 국면 자동 판정 (최근 13주 중 |ΔT10YIE| > |ΔDFII10| 인플레 주도 주 비중)[cite: 5]
    regime_known = ("T10YIE" in raw_changes_map) and ("DFII10" in raw_changes_map)
    if regime_known:
        infl_weeks_mask = np.abs(raw_changes_map["T10YIE"]) > np.abs(raw_changes_map["DFII10"])
        infl_ratio_13w: Optional[float] = float(np.mean(infl_weeks_mask[-13:]))
    else:
        infl_weeks_mask = np.zeros(use_len, dtype=bool)
        infl_ratio_13w = None
    if infl_ratio_13w is None:
        current_macro_regime = NO_DATA
    elif infl_ratio_13w >= MACRO_PARAMS["REGIME_INFL_HIGH"]:
        current_macro_regime = "인플레 주도 국면"
    elif infl_ratio_13w <= MACRO_PARAMS["REGIME_INFL_LOW"]:
        current_macro_regime = "성장·정책 주도 국면"
    else:
        current_macro_regime = "혼재 국면"

    _rff = _rf_fields(is_krx)
    rf_10y_current = _rff["rf_10y_current"]

    # 5) 4-1. 시장 동행 변수 잔차화 (최근 13주 OOS 구간을 제외한 과거 추정 구간 기준으로 잔차화 계수 산출)[cite: 5]
    oos_h = MACRO_PARAMS["OOS_HORIZON_WEEKS"]
    train_end = max(52, use_len - oos_h)

    resid_vars_list = []
    raw_vars_list = []
    var_meta_list = []

    for v_info in avail_vars:
        vid = v_info["var_id"]
        raw_arr = raw_changes_map[vid]
        # 표준화 스케일 (1σ 충격 시 주가 반응 직관적 산출을 위해 훈련 구간 표준편차로 정규화)
        train_sd = float(np.std(raw_arr[:train_end]))
        if train_sd < 1e-10:
            train_sd = 1.0

        u_arr, was_resid, corr_m = _residualize_against_market(
            raw_arr, r_m, train_end, MACRO_PARAMS["RESIDUALIZE_CORR_THRES"]
        )
        u_sd = float(np.std(u_arr[:train_end]))
        if u_sd < 1e-10:
            u_sd = 1.0

        resid_vars_list.append(u_arr / u_sd)
        raw_vars_list.append(raw_arr / train_sd)
        var_meta_list.append({
            **v_info,
            "was_residualized": was_resid,
            "corr_with_market": corr_m,
            "raw_1sd": train_sd,
            "resid_1sd": u_sd,
            "norm_resid_series": u_arr / u_sd,
            "raw_series": raw_arr,
        })

    # 6) 4-2. 3종 회귀 모형 추정 (최근 104주 주력 구간 & 최근 52주 비교 구간)[cite: 5]
    main_w = min(use_len, MACRO_PARAMS["REG_WEEKS_MAIN"])
    sub_w = min(use_len, MACRO_PARAMS["REG_WEEKS_SUB"])

    y_104 = r_i[-main_w:]
    m_104 = r_m[-main_w:]
    X_resid_104 = np.column_stack([v["norm_resid_series"][-main_w:] for v in var_meta_list])
    X_raw_104 = np.column_stack([r_arr[-main_w:] for r_arr in raw_vars_list])

    # 모형 A: 시장 + 잔차화 거시[cite: 5]
    XA_104 = np.column_stack([np.ones(main_w), m_104, X_resid_104])
    res_A_104 = _ols_hac_and_huber(y_104, XA_104, MACRO_PARAMS["HAC_MAXLAGS"])

    # 모형 B: 잔차화 거시 고유[cite: 5]
    XB_104 = np.column_stack([np.ones(main_w), X_resid_104])
    res_B_104 = _ols_hac_and_huber(y_104, XB_104, MACRO_PARAMS["HAC_MAXLAGS"])

    # 모형 B_raw: 원시 거시 참고용[cite: 5]
    XB_raw_104 = np.column_stack([np.ones(main_w), X_raw_104])
    res_Braw_104 = _ols_hac_and_huber(y_104, XB_raw_104, MACRO_PARAMS["HAC_MAXLAGS"])

    # 52주 비교 모형 A (안정성 점검용)[cite: 5]
    XA_52 = np.column_stack([np.ones(sub_w), r_m[-sub_w:], X_resid_104[-sub_w:]])
    res_A_52 = _ols_hac_and_huber(r_i[-sub_w:], XA_52, MACRO_PARAMS["HAC_MAXLAGS"])

    # 인플레 주도 주간 vs 성장·정책 주도 주간 분리 추정 (국면 의존 점검용)[cite: 5]
    infl_m_104 = infl_weeks_mask[-main_w:]
    regime_split_ok = bool(regime_known and np.sum(infl_m_104) >= 15 and np.sum(~infl_m_104) >= 15)
    res_A_infl = _ols_hac_and_huber(y_104[infl_m_104], XA_104[infl_m_104], 2) if regime_split_ok else res_A_104
    res_A_grw = _ols_hac_and_huber(y_104[~infl_m_104], XA_104[~infl_m_104], 2) if regime_split_ok else res_A_104

    # 7) 4-3 & 4-4. 변수별 유의성(|t_HAC| >= 2.0) 및 이상치·안정성 3중 점검[cite: 5]
    sensitivities_out = []
    all_warnings: List[str] = []

    for idx_v, v_meta in enumerate(var_meta_list):
        col_idx = idx_v + 2  # 0: alpha, 1: market_beta, 2~: macro vars
        b_ols = float(res_A_104["beta"][col_idx])
        t_val = float(res_A_104["t_hac"][col_idx])
        b_hub = float(res_A_104["beta_huber"][col_idx])
        b_52 = float(res_A_52["beta"][col_idx])
        b_infl = float(res_A_infl["beta"][col_idx])
        b_grw = float(res_A_grw["beta"][col_idx])

        var_warns = []
        # 이상치 점검: |beta_OLS - beta_Huber| / |beta_OLS| >= 0.50[cite: 5]
        if abs(b_ols) > 1e-6 and (abs(b_ols - b_hub) / abs(b_ols)) >= MACRO_PARAMS["OUTLIER_BETA_DIFF_TOL"]:
            var_warns.append("⚠️ 이상치 의존")
        # 104주 vs 52주 부호 반전[cite: 5]
        if (b_ols * b_52) < 0 and abs(t_val) >= 1.5:
            var_warns.append("⚠️ 관계 변화(52주 반전)")
        # 인플레 vs 성장 국면 부호 반전[cite: 5]
        # 국면별 부호 반전: 두 국면 β가 모두 노이즈 하한(1σ당 ±0.10%) 이상일 때만 경고 (0 근처 값의 부호 흔들림 제외)
        if regime_split_ok and (b_infl * b_grw) < 0 and min(abs(b_infl), abs(b_grw)) >= MACRO_PARAMS.get("REGIME_BETA_NOISE", 0.001):
            var_warns.append("⚠️ 국면 의존")

        for w_str in var_warns:
            if w_str not in all_warnings:
                all_warnings.append(w_str)

        is_sig = abs(t_val) >= 2.0
        status_str = "확인" if (is_sig and not var_warns) else ("경고" if is_sig else "미확인")

        raw_sd = v_meta["raw_1sd"]
        if MACRO_SERIES_SPEC[v_meta["var_id"]]["kind"] == "bp":
            shock_str = f"{raw_sd:+.1f}bp"
        else:
            shock_str = f"{raw_sd*100:+.1f}%"

        # 1σ 불리한 방향 충격 시 예상 주가 반응 (%)[cite: 5]
        price_resp_pct = b_ols * 100.0
        v_disp_name = v_meta["var_name"] + (" (시장 잔차화)" if v_meta["was_residualized"] else "")

        sensitivities_out.append({
            "var_id": v_meta["var_id"] + ("_resid" if v_meta["was_residualized"] else ""),
            "var_name": v_disp_name,
            "expected_sign": v_meta["expected_sign"],
            "strength": v_meta["strength"],
            "estimated_beta": round(b_ols, 3),
            "t_stat_hac": round(t_val, 2),
            "ci95_low_pct": round((b_ols - 1.96 * abs(b_ols / max(abs(t_val), 0.5))) * 100.0, 2),
            "ci95_high_pct": round((b_ols + 1.96 * abs(b_ols / max(abs(t_val), 0.5))) * 100.0, 2),
            "shock_1sd_str": shock_str,
            "price_response_1sd_num": round(price_resp_pct, 2),
            "price_response_1sd": f"{price_resp_pct:+.2f}%",
            "regime_beta_str": (f"성장 주도 {b_grw*100:+.2f}% / 인플레 주도 {b_infl*100:+.2f}% (1σ당 주가 반응)" if regime_split_ok else NO_DATA),
            "status": status_str,
            "warnings": var_warns,
            "duplicate_of": v_meta["dup"],
        })

    # 8) 6-3. 사후 검증: 이동 원점 표본 외 설명력 (ΔR^2_oos, 13주 단위 4개 구간)[cite: 5]
    oos_deltas: List[float] = []
    for k_win in range(1, 5):
        te_idx = use_len - k_win * oos_h
        tr_start = max(0, te_idx - 78)
        if te_idx - tr_start < 45 or te_idx + oos_h > use_len:
            continue
        y_tr, y_te = r_i[tr_start:te_idx], r_i[te_idx:te_idx + oos_h]
        m_tr, m_te = r_m[tr_start:te_idx], r_m[te_idx:te_idx + oos_h]
        X_tr_mac = np.column_stack([v["norm_resid_series"][tr_start:te_idx] for v in var_meta_list])
        X_te_mac = np.column_stack([v["norm_resid_series"][te_idx:te_idx + oos_h] for v in var_meta_list])

        # 모형 A (시장 + 거시) OOS 예측
        XA_tr = np.column_stack([np.ones(len(y_tr)), m_tr, X_tr_mac])
        XA_te = np.column_stack([np.ones(len(y_te)), m_te, X_te_mac])
        b_A_tr = np.linalg.lstsq(XA_tr, y_tr, rcond=None)[0]
        pred_A = XA_te @ b_A_tr

        # 시장 단일 모형 OOS 예측
        XM_tr = np.column_stack([np.ones(len(y_tr)), m_tr])
        XM_te = np.column_stack([np.ones(len(y_te)), m_te])
        b_M_tr = np.linalg.lstsq(XM_tr, y_tr, rcond=None)[0]
        pred_M = XM_te @ b_M_tr

        denom_oos = float(np.sum((y_te - np.mean(y_tr)) ** 2))
        if denom_oos > 1e-10:
            r2_oos_A = 1.0 - float(np.sum((y_te - pred_A) ** 2)) / denom_oos
            r2_oos_M = 1.0 - float(np.sum((y_te - pred_M) ** 2)) / denom_oos
            oos_deltas.append(r2_oos_A - r2_oos_M)

    oos_delta_mean = round(float(np.mean(oos_deltas)), 4) if oos_deltas else None
    # 연속 2개 구간 ΔR^2_oos <= 0 자동 강등 여부[cite: 5]
    auto_downgrade_idiosyncratic = bool(len(oos_deltas) >= 2 and oos_deltas[0] <= 0.0 and oos_deltas[1] <= 0.0)

    # 9) 4-5. 거시 설명력 4단계 분류[cite: 5]
    r2_own = res_B_104["r2"]
    r2_raw = res_Braw_104["r2"]
    mkt_beta = round(float(res_A_104["beta"][1]), 2)

    if auto_downgrade_idiosyncratic:
        cls_icon = "🟢"
        cls_label = "기업 고유 (IDIOSYNCRATIC · OOS 과적합 자동 강등)"
        all_warnings.append("⚠️ 연속 2구간 OOS ΔR² ≤ 0 강등")
    elif r2_own >= MACRO_PARAMS["R2_DOMINANT"]:
        cls_icon = "🔴"
        cls_label = "거시 지배 (MACRO_DOMINANT)"
    elif r2_own >= MACRO_PARAMS["R2_MIXED"]:
        cls_icon = "🟡"
        cls_label = "거시 혼합 (MACRO_MIXED)"
    elif r2_raw >= MACRO_PARAMS["R2_DOMINANT"]:
        cls_icon = "🟡"
        cls_label = "시장 경유 노출 (MARKET_MEDIATED)"
    else:
        cls_icon = "🟢"
        cls_label = "기업 고유 (IDIOSYNCRATIC)"

    # 10) 6-1. 최근 4주·13주 수익률 기여 분해 (모형 A 기준)[cite: 5]
    beta_A = res_A_104["beta"]
    X_full_A = np.column_stack([np.ones(use_len), r_m, np.column_stack([v["norm_resid_series"] for v in var_meta_list])])

    def _decompose_window(win_w: int) -> Dict[str, float]:
        act_sum = float(np.sum(r_i[-win_w:]))
        mkt_sum = float(beta_A[1] * np.sum(r_m[-win_w:]))
        mac_sum = float(np.sum(X_full_A[-win_w:, 2:] @ beta_A[2:]))
        idio_sum = act_sum - (mkt_sum + mac_sum)
        return {
            "actual": round(act_sum, 4),
            "market": round(mkt_sum, 4),
            "macro_extra": round(mac_sum, 4),
            "idiosyncratic": round(idio_sum, 4),
        }

    attrib_13w = {
        "4w": _decompose_window(4),
        "13w": _decompose_window(13),
    }

    # 11) 6-2. 최근 13주 표본 외 잔차 점검 (관계 유지 vs 관계 이탈)[cite: 5]
    # 최근 13주(H)를 제외한 이전 구간만으로 모형 A 추정 후 최근 13주 잔차 점검[cite: 5]
    XA_train_only = X_full_A[:train_end]
    y_train_only = r_i[:train_end]
    res_train_only = _ols_hac_and_huber(y_train_only, XA_train_only, MACRO_PARAMS["HAC_MAXLAGS"])
    sigma_train = max(float(np.std(res_train_only["resid"])), 1e-4)

    XA_oos_13 = X_full_A[train_end:]
    y_oos_13 = r_i[train_end:]
    e_oos_13 = y_oos_13 - (XA_oos_13 @ res_train_only["beta"])
    oos_dates = w_idx[train_end:]

    # 기업 고유 이벤트 주간 분리 (|잔차| > 2.2 * sigma_train 주간을 잠정실적/공시 이벤트 주간으로 식별)[cite: 5]
    event_weeks = []
    normal_resids = []
    _last_trade = pd.Timestamp(s_close.dropna().index[-1])
    for d_dt, e_val in zip(oos_dates, e_oos_13):
        r_sig = float(e_val / sigma_train)
        if abs(r_sig) >= 2.2 and len(event_weeks) < 2:
            event_weeks.append({
                "date": str(min(pd.Timestamp(d_dt), _last_trade).date()),   # 주간 라벨(금요일)이 아니라 실제 마지막 거래일 이하
                "type": "기업 고유 충격 주간 (표본 외 잔차 ≥ 2.2σ · 원인 미확인)",
                "residual_sigma": round(r_sig, 2),
            })
        else:
            normal_resids.append(float(e_val))

    n_norm = len(normal_resids)
    if n_norm >= MACRO_PARAMS["OOS_MIN_NORMAL_WEEKS"]:
        z_oos = float(np.sum(normal_resids) / (sigma_train * math.sqrt(n_norm)))
        resid_status = "🟢 관계 유지" if abs(z_oos) < MACRO_PARAMS["OOS_RESID_Z_THRES"] else "⚠️ 관계 이탈"
    else:
        z_oos = None
        resid_status = "판정 불가 (평범한 주 6주 미만)"

    # 12) 5-1. 가설 변수 13주 경로 및 6-4 감시 캘린더 생성[cite: 5]
    macro_state_13w = []
    watch_calendar = []
    for v_meta in var_meta_list:
        vid = v_meta["var_id"]
        raw_s = v_meta["raw_series"]
        _spec = MACRO_SERIES_SPEC[vid]
        chg_kind = _spec["kind"]
        unit_lbl = _spec["level_unit"]
        _lv13 = macro_levels[vid].loc[:w_idx[-1]].iloc[-13:] * _spec["level_mult"]

        chg_13 = float(np.sum(raw_s[-13:]))
        chg_4 = float(np.sum(raw_s[-4:]))
        share_4w = round(chg_4 / chg_13, 2) if abs(chg_13) > 1e-6 else 0.0

        # 13주 내내 같은 부호인지 또는 최근 4주 몫이 50% 이상인지 판정[cite: 5]
        same_dir_13w = bool(np.all(raw_s[-13:] >= 0) or np.all(raw_s[-13:] <= 0))
        if same_dir_13w:
            path_lbl = "13주 추세 지속"
        elif abs(share_4w) >= MACRO_PARAMS["RECENT_SURGE_SHARE_4W"]:
            path_lbl = f"최근 4주 급변 (4주 몫 {int(abs(share_4w)*100)}%)"
        else:
            path_lbl = f"13주 등락 교차 (4주 몫 {int(abs(share_4w)*100)}%)"

        _dec = 2 if unit_lbl in ("%", "조 달러") else (0 if unit_lbl in ("bp", "원") else 1)
        _fmt = (lambda x, d=_dec: f"{x:,.{d}f}")
        chg_13_str = f"{chg_13:+.1f}bp" if chg_kind == "bp" else f"{chg_13*100:+.1f}%"
        hi_lo_str = (f"{_fmt(float(_lv13.min()))} ~ {_fmt(float(_lv13.max()))}{unit_lbl} (최근 {_fmt(float(_lv13.iloc[-1]))}{unit_lbl})"
                     if len(_lv13) else NO_DATA)

        macro_state_13w.append({
            "var_name": v_meta["var_name"],
            "change_13w": chg_13_str,
            "high_low_13w": hi_lo_str,
            "share_4w": share_4w,
            "path_label": path_lbl,
        })

        watch_calendar.append({
            "var_name": v_meta["var_name"],
            "next_release": "매주 목요일 22:30 (주간)" if vid == "ICSA" else "상시 (일별 마켓 데이터)",
            "threshold_rule": (f"주간 1σ({v_meta['raw_1sd']:.1f}bp) 이상 급변 시 민감도 재점검" if MACRO_SERIES_SPEC[vid]["kind"] == "bp"
                               else f"주간 1σ({v_meta['raw_1sd']*100:.1f}%) 이상 급변 시 민감도 재점검"),
            "source": macro_sources.get(vid, NO_DATA),
            "duplicate_tag": f"(중복 — {v_meta['dup']})" if v_meta["dup"] else "거시 고유 감시 지표",
        })

    # 13) 6-4. 금지 문구 필터를 통과한 4단계 표준 코멘트 생성[cite: 5]
    top_s = sensitivities_out[0]
    sec_s = sensitivities_out[1] if len(sensitivities_out) > 1 else top_s
    a13 = attrib_13w["13w"]
    raw_comment = (
        (f"{top_s['var_name']} 및 {sec_s['var_name']} 지표가 1σ 움직일 때 과거 {main_w}주 평균 각각 "
         f"{top_s['price_response_1sd']}, {sec_s['price_response_1sd']}씩 동행해 왔다. "
         if sec_s is not top_s else
         f"{top_s['var_name']} 지표가 1σ 움직일 때 과거 {main_w}주 평균 {top_s['price_response_1sd']}씩 동행해 왔다. ")
        + 
        f"최근 13주 주가 변화({a13['actual']*100:+.1f}%) 중 시장 몫은 {a13['market']*100:+.1f}%p, "
        f"거시 추가 몫은 {a13['macro_extra']*100:+.1f}%p, 기업 고유 몫은 {a13['idiosyncratic']*100:+.1f}%p이며 "
        f"평범한 주 기준 과거 관계는 {resid_status} 상태다(z_oos = {(f'{z_oos:+.2f}' if z_oos is not None else NO_DATA)}). "
        + (f"→ {watch_calendar[0]['var_name']} 발표 일정과 {current_macro_regime} 전환 여부를 감시 목록에 올린다."
           if regime_known else f"→ {watch_calendar[0]['var_name']} 발표 일정을 감시 목록에 올린다 (국면 판정용 금리 데이터 {NO_DATA}).")
    )
    clean_comment = _sanitize_macro_comment(raw_comment)

    return {
        "status": "완료",
        "meta": {
            "ticker": ticker,
            "as_of": datetime.now().strftime("%Y-%m-%d"),
            "data_quality": "🟢" if (not missing_hyp and regime_known) else "🟡",
            "sample_weeks": main_w,
            "hypothesis_id": hyp_id,
            "industry_category": industry_category,
            "is_on_hold": False,
            "missing_vars": missing_hyp,
            "regime_missing": [] if regime_known else [x for x in ("T10YIE", "DFII10") if x not in raw_changes_map],
            "sources": macro_sources,
            "macro_as_of": macro_as_of,
        },
        "classification": {
            "grade_icon": cls_icon,
            "label": cls_label,
            "r2_macro_own": r2_own,
            "r2_raw": r2_raw,
            "market_beta": mkt_beta,
            "oos_delta_r2_mean": oos_delta_mean,
            "warnings": all_warnings,
        },
        "macro_regime": {
            "current_regime": current_macro_regime,
            "inflation_week_ratio_13w": round(infl_ratio_13w, 2) if infl_ratio_13w is not None else None,
            "rf_10y_current": rf_10y_current,
            "rf_source": _rff["rf_source"],
            "rf_as_of": _rff["rf_as_of"],
        },
        "sensitivities": sensitivities_out,
        "attribution_13w": attrib_13w,
        "residual_check": {
            "status": resid_status,
            "z_oos": round(z_oos, 2) if z_oos is not None else None,
            "normal_weeks_count": n_norm,
            "event_weeks": event_weeks,
        },
        "macro_state_13w": macro_state_13w,
        "watch_calendar": watch_calendar,
        "comment": clean_comment,
    }

    # =====================================================================
# [Synthesis] 4대 트랙 종합 결론 프레임워크 v1.1 (Waterfall Gate & MECE 8 아키타입)

# (원래 engine.py 3522~3580줄에서 옮김: 순환 참조 방지)
# ---------------------------------------------------------------------
# 무위험수익률(10년 국채) 실데이터: FRED 공개 CSV (미국 DGS10 일별, 한국 IRLTLT01KRM156N 월별 OECD 장기금리).
# 미국은 실패 시 yfinance ^TNX로 한 번 더 시도. 못 받으면 None → '데이터 없음' (고정값으로 대체하지 않는다).
# ---------------------------------------------------------------------
RF_SERIES = {"US": "DGS10", "KR": "IRLTLT01KRM156N"}
_RF_CACHE: Dict[str, Dict[str, Any]] = {}
_RF_TTL_SEC = 6 * 3600


def _rf_fields(is_krx: bool) -> Dict[str, Any]:
    """거시 트랙 출력용 10년 국채 금리 필드 (못 받으면 None)."""
    r = fetch_risk_free_10y(is_krx)
    return {"rf_10y_current": r["value"], "rf_source": r["source"], "rf_as_of": r["as_of"]}


def fetch_risk_free_10y(is_krx: bool) -> Dict[str, Any]:
    """10년 국채 금리(소수, 예 0.0484). 반환 {value, as_of, source}. 못 받으면 value=None.
    한국: ECOS 817Y002 국고채 10년(일별, ECOS_API_KEY) → FRED OECD 월별 장기금리.
    미국: FRED DGS10(FRED_API_KEY 있으면 API, 없으면 공개 CSV) → yfinance ^TNX."""
    key = "KR" if is_krx else "US"
    hit = _RF_CACHE.get(key)
    if hit and (time.time() - hit["_t"] < _RF_TTL_SEC):
        return {k: v for k, v in hit.items() if k != "_t"}
    val = as_of = None
    src = NO_DATA
    _start = (datetime.now() - pd.Timedelta(days=45)).strftime("%Y-%m-%d")
    if is_krx:
        try:
            ser = _ecos_series("817Y002", "010210000", _start, datetime.now().strftime("%Y-%m-%d"))
            if ser is not None and len(ser):
                val, as_of, src = float(ser.iloc[-1]), str(ser.index[-1].date()), "한국은행 ECOS 국고채 10년"
        except Exception:
            val = None
    if val is None:
        try:
            ser = _fred_series(RF_SERIES[key])
            if ser is not None and len(ser):
                val, as_of = float(ser.iloc[-1]), str(ser.index[-1].date())
                src = f"FRED {RF_SERIES[key]}" + (" (월별 OECD 장기금리)" if is_krx else "")
        except Exception:
            val = None
    if val is None and not is_krx:
        try:
            import yfinance as yf
            h = yf.Ticker("^TNX").history(period="10d")
            if h is not None and len(h) > 0:
                val = float(h["Close"].dropna().iloc[-1])
                as_of = str(h.index[-1].date())
                src = "yfinance ^TNX"
        except Exception:
            val = None
    if val is not None and not (-1.0 < val < 25.0):   # 연 % 단위가 아닌 값(잘못 받은 시세 등)은 버린다
        val = None
    out = {"value": (round(val / 100.0, 5) if val is not None else None), "as_of": as_of if val is not None else None,
           "source": src if val is not None else NO_DATA}
    if val is not None:   # 실패는 캐시하지 않는다 (다음 실행에서 다시 시도)
        _RF_CACHE[key] = dict(out, _t=time.time())
    return out
