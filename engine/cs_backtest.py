"""H-CS1 횡단면 모멘텀 — '언제 살까' 대신 '무엇을 살까' (2026-11-20 사전 등록 — 결과 보기 전 고정, data/backtest/CS1_PREREG.md, cs1-1).

① 외부 자료(Ken French, 상장폐지 포함 CRSP 기반): 미국 대형·모멘텀 상위 30%(시가총액 가중) − 시장 − 추정 비용, 1927~ 전체 t ≥ 2 그리고 2000~ > 0.
② 새 151종목: 매달 말 12-1개월 모멘텀 상위 20% 같은 비중 − 같은 종목 같은 비중, 실제 회전율 × 0.30% 비용, t ≥ 2 · 앞뒤 절반 > 0 · 미국·한국 반대 아님.
t는 Newey-West(시차 3). 숫자는 CS_CONFIG 한 곳. 결과를 보고 바꾸지 않는다.
"""
import io
import math
import zipfile
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .lecture_backtest import lec_market_of
from .momentum_explore import nw_se
from .rule_backtest import _bt_naive

CS_CONFIG: Dict[str, Any] = {
    "VERSION": "cs1-1",
    "REGISTERED": "2026-11-20",
    # --- ② 우리 방식 ---
    "LOOKBACK": 252, "SKIP": 21,           # 12-1개월 모멘텀 (탐색 mom12와 같은 식)
    "MONTH_GAP_DAYS": 5,
    "TOP": 0.8, "TOP_REF": 0.9,            # 상위 20% (참고 10%)
    "MIN_STOCKS": 20,
    "START_MONTH": "2001-01",
    "COST": 0.003,                          # 바뀐 비중(한쪽) × 왕복 0.30%
    "MIN_MARKET_MONTHS": 60,
    # --- ① 외부 자료 ---
    "FRENCH_BASE": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.French/ftp/",
    "FRENCH_FILES": {"us_mom": "6_Portfolios_ME_Prior_12_2_CSV.zip", "us_ff": "F-F_Research_Data_Factors_CSV.zip",
                     "dx_mom": "Developed_ex_US_6_Portfolios_ME_Prior_12_2_CSV.zip", "dx_ff": "Developed_ex_US_3_Factors_CSV.zip"},
    "RECENT_START": "2000-01",
    "SUBPERIODS": (("1927-01", "1963-12"), ("1964-01", "1999-12"), ("2000-01", None)),
    # --- 판정 ---
    "NW_LAG": 3, "PASS_T": 2.0, "POWER_Z": 0.84,
    "MDE_SUBSETS": 20, "SUBSET_US": 98, "SUBSET_KR": 53,
    "TAX_ACCOUNT": 1e8, "TAX_DEDUCTION": 2.5e6, "TAX_RATE": 0.22,
    "SEED": 1123,
}


# ---------------------------------------------------------------------
# ① Ken French 자료 읽기
# ---------------------------------------------------------------------
def cs_zip_text(raw: bytes) -> str:
    """French zip(안에 CSV 하나) → 글자. zip이 아니면 그대로 글자로."""
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            name = [n for n in z.namelist() if n.lower().endswith((".csv", ".txt"))][0]
            return z.read(name).decode("latin-1")
    except zipfile.BadZipFile:
        return raw.decode("latin-1")


def cs_parse_french(text: str) -> List[Dict[str, Any]]:
    """French CSV를 표 묶음으로: [{title, cols, freq('M'|'Y'), df}]. 값은 % 그대로, −99.99·−999는 빈칸."""
    secs: List[Dict[str, Any]] = []
    cur, title = None, ""

    def close():
        if cur and cur["rows"]:
            keys = [k for k, _ in cur["rows"]]
            freq = "M" if len(keys[0]) == 6 else "Y"
            idx = pd.PeriodIndex([f"{k[:4]}-{k[4:]}" for k in keys], freq="M") if freq == "M" else pd.Index([int(k) for k in keys])
            ncol = len(cur["cols"])
            vals = [(v + [np.nan] * ncol)[:ncol] for _, v in cur["rows"]]
            df = pd.DataFrame(vals, index=idx, columns=cur["cols"]).astype(float)
            df = df.mask(df <= -99.99)
            secs.append({"title": cur["title"], "cols": cur["cols"], "freq": freq, "df": df})

    for ln in text.splitlines():
        s = ln.strip()
        if not s:
            if cur is not None:
                close()
                cur = None
            continue
        if s.startswith(","):
            if cur is not None:
                close()
            cur = {"title": title, "cols": [c.strip() for c in s.split(",")[1:]], "rows": []}
            continue
        first = s.split(",")[0].strip()
        if cur is not None and first.isdigit() and len(first) in (4, 6):
            vals = []
            for v in s.split(",")[1:]:
                try:
                    vals.append(float(v))
                except ValueError:
                    vals.append(np.nan)
            cur["rows"].append((first, vals))
            continue
        if cur is not None:
            close()
            cur = None
        title = s
    if cur is not None:
        close()
    return secs


def _cs_pick(secs, want: str) -> Optional[Dict[str, Any]]:
    """want = 'vw'(시가총액 가중 수익) | 'nfirms' | 'size'. 월간 표만."""
    for s in secs:
        if s["freq"] != "M":
            continue
        t = s["title"].lower()
        if want == "vw" and not any(k in t for k in ("equal", "number", "firm", "size", "cap", "book", "sum")):
            return s
        if want == "nfirms" and "number" in t:
            return s
        if want == "size" and ("size" in t or "market cap" in t) and "number" not in t:
            return s
    return None


def _cs_col(cols: List[str], kind: str) -> Optional[str]:
    """6개 포트폴리오에서 대형·높음(BH)·중간(BN)·낮음(BL) 열 찾기."""
    up = [c.upper().replace(" ", "") for c in cols]
    exact = {"BH": "BW", "BN": "BN", "BL": "BL"}[kind]
    for c, u in zip(cols, up):
        if u == exact:
            return c
    key = {"BH": "HI", "BN": "PRIOR2", "BL": "LO"}[kind]
    for c, u in zip(cols, up):
        if ("BIG" in u or "ME2" in u) and key in u:
            return c
    pos = {"BL": 3, "BN": 4, "BH": 5}[kind]
    return cols[pos] if len(cols) == 6 else None


def cs_french_frame(mom_text: str, ff_text: str) -> pd.DataFrame:
    """월간: bh·bn·bl(대형 세 포트폴리오), mkt(시장 = Mkt−RF + RF), big(대형 전체, 표가 있을 때만). 소수(0.01 = 1%)."""
    ms = cs_parse_french(mom_text)
    vw = _cs_pick(ms, "vw")
    if vw is None:
        raise ValueError("모멘텀 파일에서 시가총액 가중 월간 표를 찾지 못했습니다")
    d = vw["df"]
    out = pd.DataFrame({k.lower(): d[_cs_col(vw["cols"], k)] / 100.0 for k in ("BH", "BN", "BL")})
    nf, sz = _cs_pick(ms, "nfirms"), _cs_pick(ms, "size")
    if nf is not None and sz is not None:
        caps = {k: nf["df"][_cs_col(nf["cols"], k)] * sz["df"][_cs_col(sz["cols"], k)] for k in ("BH", "BN", "BL")}
        tot = caps["BH"] + caps["BN"] + caps["BL"]
        out["big"] = sum(out[k.lower()] * caps[k] for k in caps) / tot
    fs = [s for s in cs_parse_french(ff_text) if s["freq"] == "M"]
    if not fs:
        raise ValueError("팩터 파일에서 월간 표를 찾지 못했습니다")
    f = fs[0]["df"]
    mcol = [c for c in f.columns if c.replace(" ", "").upper() in ("MKT-RF", "MKT_RF", "MKTRF")][0]
    out["mkt"] = (f[mcol] + f["RF"]) / 100.0
    return out.dropna(subset=["bh", "mkt"])


# ---------------------------------------------------------------------
# 통계 · 판정
# ---------------------------------------------------------------------
def cs_stats(x: pd.Series) -> Dict[str, Any]:
    v = pd.Series(x).dropna()
    n = len(v)
    if n < 3:
        return {"n": n, "mean": None, "se": None, "t": None, "hit": None, "worst": None, "mdd": None, "first": None, "second": None,
                "start": None, "end": None}
    a = v.to_numpy(dtype=float)
    se = nw_se(a, CS_CONFIG["NW_LAG"])
    cum = np.cumprod(1.0 + a)
    mdd = float(np.min(cum / np.maximum.accumulate(cum)) - 1.0)
    h = n // 2
    return {"n": n, "mean": float(a.mean()), "se": se, "t": (float(a.mean()) / se if se else None), "hit": float((a > 0).mean()),
            "worst": float(a.min()), "worst_month": str(v.index[int(np.argmin(a))]), "mdd": mdd,
            "first": float(a[:h].mean()), "second": float(a[h:].mean()), "start": str(v.index[0]), "end": str(v.index[-1]),
            "ann": float(a.mean() * 12)}


def cs_mde(se: Optional[float], z: Optional[float] = None) -> Optional[float]:
    if se is None or not np.isfinite(se):
        return None
    return float(((CS_CONFIG["PASS_T"] if z is None else z) + CS_CONFIG["POWER_Z"]) * se)


def _cs_since(x: pd.Series, start: Optional[str], end: Optional[str] = None) -> pd.Series:
    m = np.ones(len(x), dtype=bool)
    if start:
        m &= x.index >= pd.Period(start, "M")
    if end:
        m &= x.index <= pd.Period(end, "M")
    return x[m]


def cs_judge_part1(full: Dict[str, Any], recent: Dict[str, Any]) -> Dict[str, Any]:
    rs = []
    if full["mean"] is None or full["mean"] <= 0:
        rs.append("전체 기간 비용 뒤 초과 ≤ 0")
    if full["t"] is None or full["t"] < CS_CONFIG["PASS_T"]:
        rs.append(f"전체 기간 t < {CS_CONFIG['PASS_T']:.0f}")
    if recent["mean"] is None or recent["mean"] <= 0:
        rs.append("2000년 이후 비용 뒤 초과 ≤ 0")
    return {"passed": not rs, "reasons": rs}


def cs_judge_part2(s: Dict[str, Any], us: Dict[str, Any], kr: Dict[str, Any]) -> Dict[str, Any]:
    cf = CS_CONFIG
    rs = []
    if s["mean"] is None or s["mean"] <= 0:
        rs.append("비용 뒤 초과 ≤ 0")
    if s["t"] is None or s["t"] < cf["PASS_T"]:
        rs.append(f"t < {cf['PASS_T']:.0f}")
    if s["first"] is None or s["first"] <= 0 or s["second"] is None or s["second"] <= 0:
        rs.append("앞·뒤 절반 중 하나가 ≤ 0")
    if (us["n"] >= cf["MIN_MARKET_MONTHS"] and kr["n"] >= cf["MIN_MARKET_MONTHS"] and us["mean"] is not None and kr["mean"] is not None
            and np.sign(us["mean"]) != np.sign(kr["mean"])):
        rs.append("미국·한국 방향이 다름")
    return {"passed": not rs, "reasons": rs}


def run_cs_part1(us_mom: str, us_ff: str, turnover: float, dx_mom: Optional[str] = None, dx_ff: Optional[str] = None) -> Dict[str, Any]:
    """① 외부 자료: x = 대형·높음 − 시장 − 회전율 × 0.30%."""
    cf = CS_CONFIG
    fr = cs_french_frame(us_mom, us_ff)
    cost = turnover * cf["COST"]
    x = fr["bh"] - fr["mkt"] - cost
    full, recent = cs_stats(x), cs_stats(_cs_since(x, cf["RECENT_START"]))
    out = {"cost_per_month": cost, "turnover": turnover, "full": full, "recent": recent, "judge": cs_judge_part1(full, recent),
           "mde_full": cs_mde(full["se"]), "mde_recent": cs_mde(recent["se"]),
           "periods": {f"{a[:4]}~{(b or '')[:4]}": cs_stats(_cs_since(x, a, b)) for a, b in cf["SUBPERIODS"]},
           "gross_full": cs_stats(fr["bh"] - fr["mkt"]), "gross_recent": cs_stats(_cs_since(fr["bh"] - fr["mkt"], cf["RECENT_START"]))}
    if "big" in fr:
        xb = (fr["bh"] - fr["big"] - cost).dropna()
        out["vs_big"] = {"full": cs_stats(xb), "recent": cs_stats(_cs_since(xb, cf["RECENT_START"]))}
    if dx_mom and dx_ff:
        try:
            fd = cs_french_frame(dx_mom, dx_ff)
            xd = fd["bh"] - fd["mkt"] - cost
            out["developed_ex_us"] = {"full": cs_stats(xd), "recent": cs_stats(_cs_since(xd, cf["RECENT_START"]))}
        except (ValueError, IndexError, KeyError) as e:
            out["developed_ex_us"] = {"error": str(e)}
    return out


# ---------------------------------------------------------------------
# ② 종목 하나의 달마다 신호·보유 수익
# ---------------------------------------------------------------------
def cs_stock_months(df: pd.DataFrame) -> pd.DataFrame:
    """열: month, mom(12-1개월), vadj(모멘텀 ÷ 같은 구간 연 변동성), r(달 끝 다음 날 시가 → 다음 달 끝 다음 날 시가).
    신호는 달 끝 종가까지, 보유는 그다음 날부터라 미래 자료를 쓰지 않는다."""
    c = CS_CONFIG
    d = _bt_naive(df)
    cols = ["month", "mom", "vadj", "r"]
    if d is None or not len(d) or "Open" not in d:
        return pd.DataFrame(columns=cols)
    d = d[(d[["Open", "Close"]].astype(float) > 0).all(axis=1)]
    cl, op = d["Close"].to_numpy(dtype=float), d["Open"].to_numpy(dtype=float)
    idx = d.index
    n = len(cl)
    lr = np.diff(np.log(cl), prepend=np.nan)
    per = idx.to_period("M")
    last = pd.Series(np.arange(n), index=idx).groupby(per).max()
    ok = {m: int(i) for m, i in last.items() if (m.to_timestamp(how="end") - idx[int(i)]).days <= c["MONTH_GAP_DAYS"]}
    rows = []
    for m, i in ok.items():
        nxt = ok.get(m + 1)
        r = op[nxt + 1] / op[i + 1] - 1.0 if (nxt is not None and nxt + 1 <= n - 1 and i + 1 <= n - 1) else np.nan
        if i - c["LOOKBACK"] < 0:
            mom = vadj = np.nan
        else:
            mom = cl[i - c["SKIP"]] / cl[i - c["LOOKBACK"]] - 1.0
            sd = np.nanstd(lr[i - c["LOOKBACK"] + 1:i - c["SKIP"] + 1], ddof=1) * math.sqrt(252)
            vadj = mom / sd if sd and np.isfinite(sd) and sd > 0 else np.nan
        rows.append({"month": m, "mom": mom, "vadj": vadj, "r": r})
    return pd.DataFrame(rows, columns=cols)


def cs_panel(price_map: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    parts = []
    for sym, df in (price_map or {}).items():
        r = cs_stock_months(df)
        if len(r):
            parts.append(r.assign(ticker=sym, market=lec_market_of(sym)))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["month", "mom", "vadj", "r", "ticker", "market"])


def cs_portfolio(panel: pd.DataFrame, market: str, sig: str = "mom", top: Optional[float] = None,
                 start: Optional[str] = None) -> pd.DataFrame:
    """한 시장의 달마다: 상위 같은 비중 vs 전체 같은 비중의 수익·회전율·비용 뒤 수익. 회전율 = Σ max(새 비중 − 흘러간 비중, 0)."""
    cf = CS_CONFIG
    top = cf["TOP"] if top is None else top
    p = panel[(panel["market"] == market)]
    p = p[np.isfinite(p[sig].to_numpy(dtype=float)) & np.isfinite(p["r"].to_numpy(dtype=float))]
    if start:
        p = p[p["month"] >= pd.Period(start, "M")]
    rows = []
    prev = {"top": None, "ew": None}
    last_m = None
    for m, g in p.groupby("month", sort=True):
        if len(g) < cf["MIN_STOCKS"]:
            prev, last_m = {"top": None, "ew": None}, None
            continue
        pct = g[sig].rank(pct=True, method="average")
        sel = {"top": g[pct > top], "ew": g}
        row = {"month": m, "n": len(g)}
        for k, gg in sel.items():
            w = pd.Series(1.0 / len(gg), index=gg["ticker"].to_numpy())
            rr = pd.Series(gg["r"].to_numpy(dtype=float), index=gg["ticker"].to_numpy())
            gross = float((w * rr).sum())
            if prev[k] is None or last_m is None or m != last_m + 1:
                to = 0.0 if not rows else 1.0
            else:
                wd = prev[k]
                to = float(np.maximum(w.reindex(w.index.union(wd.index), fill_value=0.0) - wd.reindex(w.index.union(wd.index), fill_value=0.0), 0).sum())
            row.update({f"{k}_gross": gross, f"{k}_to": to, f"{k}_net": gross - to * cf["COST"], f"{k}_n": len(gg)})
            drift = w * (1.0 + rr)
            prev[k] = drift / drift.sum()
        last_m = m
        rows.append(row)
    out = pd.DataFrame(rows)
    if len(out):
        out = out.set_index(pd.PeriodIndex(out["month"], freq="M"))
        out["x"] = out["top_net"] - out["ew_net"]
    return out


def cs_combined(us: pd.DataFrame, kr: pd.DataFrame, col: str = "x") -> pd.Series:
    parts = [s[col] for s in (us, kr) if len(s) and col in s]
    return pd.concat(parts, axis=1).mean(axis=1, skipna=True).sort_index() if parts else pd.Series(dtype=float)


def cs_index_monthly(df: Optional[pd.DataFrame]) -> pd.Series:
    """지수의 달 끝 다음 날 시가 → 다음 달 끝 다음 날 시가(참고 비교)."""
    if df is None or not len(df):
        return pd.Series(dtype=float)
    m = cs_stock_months(df)
    return pd.Series(m["r"].to_numpy(dtype=float), index=pd.PeriodIndex(m["month"], freq="M"))


def cs_tax_example(net: pd.Series) -> Dict[str, Any]:
    """세금 예시(참고): 1억 원, 해마다 그해 이익을 모두 실현한다고 단순 가정 → (이익 − 250만 원) × 22%. 지수 ETF 보유는 팔 때까지 0."""
    cf = CS_CONFIG
    if not len(net):
        return {"years": 0}
    yr = net.groupby(net.index.year).apply(lambda v: float(np.prod(1.0 + v.to_numpy()) - 1.0))
    yr = yr[[y for y in yr.index if (net.index.year == y).sum() == 12]]
    if not len(yr):
        return {"years": 0}
    gain = yr * cf["TAX_ACCOUNT"]
    tax = np.maximum(gain - cf["TAX_DEDUCTION"], 0.0) * cf["TAX_RATE"]
    after = (gain - tax) / cf["TAX_ACCOUNT"]
    return {"years": int(len(yr)), "pre_mean": float(yr.mean()), "after_mean": float(after.mean()), "drag": float((yr - after).mean())}


def run_cs_part2(price_map: Dict[str, pd.DataFrame], index_map: Optional[Dict[str, pd.DataFrame]] = None) -> Dict[str, Any]:
    cf = CS_CONFIG
    panel = cs_panel(price_map)
    st = cf["START_MONTH"]
    us, kr = cs_portfolio(panel, "US", start=st), cs_portfolio(panel, "KR", start=st)
    x = cs_combined(us, kr)
    s, su, sk = cs_stats(x), cs_stats(us["x"] if len(us) else pd.Series(dtype=float)), cs_stats(kr["x"] if len(kr) else pd.Series(dtype=float))
    out = {"n_tickers": int(panel["ticker"].nunique()) if len(panel) else 0, "stats": s, "us": su, "kr": sk,
           "judge": cs_judge_part2(s, su, sk), "mde": cs_mde(s["se"]),
           "turnover": {m: (float(p["top_to"].iloc[1:].mean()) if len(p) > 1 else None) for m, p in (("US", us), ("KR", kr))},
           "n_stocks": {m: (float(p["n"].mean()) if len(p) else None) for m, p in (("US", us), ("KR", kr))}}
    refs = {}
    for nm, kw in (("top10", {"top": cf["TOP_REF"]}), ("vadj", {"sig": "vadj"})):
        a, b = cs_portfolio(panel, "US", start=st, **kw), cs_portfolio(panel, "KR", start=st, **kw)
        refs[nm] = cs_stats(cs_combined(a, b))
    ix = index_map or {}
    for m, p, key in (("US", us, "^GSPC"), ("KR", kr, "^KS11")):
        im = cs_index_monthly(ix.get(key))
        if len(p) and len(im):
            refs[f"vs_index_{m}"] = cs_stats((p["top_net"] - im.reindex(p.index)).dropna())
    out["refs"] = refs
    out["tax_us"] = cs_tax_example(us["top_net"]) if len(us) else {"years": 0}
    return out


# ---------------------------------------------------------------------
# 실행 전: 회전율 · 흔들림 · 대상 종목 수 (수익 평균은 보지 않음)
# ---------------------------------------------------------------------
def cs_seen_estimates(price_map545: Dict[str, pd.DataFrame], seed: Optional[int] = None) -> Dict[str, Any]:
    """545종목(이미 쓴 자료): 미국·한국 상위 20% 월평균 한쪽 회전율, 151종목 크기 무작위 부분집합의 월간 x 표준오차 → ② 최소 차이."""
    cf = CS_CONFIG
    panel = cs_panel(price_map545)
    st = cf["START_MONTH"]
    us, kr = cs_portfolio(panel, "US", start=st), cs_portfolio(panel, "KR", start=st)
    to = {m: (float(p["top_to"].iloc[1:].mean()) if len(p) > 1 else None) for m, p in (("US", us), ("KR", kr))}
    rng = np.random.default_rng(cf["SEED"] if seed is None else seed)
    tk = panel.groupby("market")["ticker"].unique().to_dict() if len(panel) else {}
    ses, months = [], []
    for _ in range(cf["MDE_SUBSETS"]):
        pick = []
        for m, k in (("US", cf["SUBSET_US"]), ("KR", cf["SUBSET_KR"])):
            arr = tk.get(m, np.array([]))
            if len(arr):
                pick += list(rng.choice(arr, size=min(k, len(arr)), replace=False))
        sub = panel[panel["ticker"].isin(pick)]
        xx = cs_combined(cs_portfolio(sub, "US", start=st), cs_portfolio(sub, "KR", start=st)).dropna()
        if len(xx) > 24:
            ses.append(nw_se(xx.to_numpy(dtype=float), cf["NW_LAG"]))
            months.append(len(xx))
    se = float(np.median(ses)) if ses else None
    return {"turnover": to, "subset_se": se, "subset_months": int(np.median(months)) if months else 0, "mde_part2": cs_mde(se)}


def cs_part1_noise(us_mom: str, us_ff: str, turnover: float) -> Dict[str, Any]:
    """① 흔들림만: 전체·2000년 이후 표준오차와 최소 차이. 평균은 돌려주지 않는다."""
    cf = CS_CONFIG
    fr = cs_french_frame(us_mom, us_ff)
    x = fr["bh"] - fr["mkt"] - turnover * cf["COST"]
    xr = _cs_since(x, cf["RECENT_START"])
    sf, sr = nw_se(x.dropna().to_numpy(), cf["NW_LAG"]), nw_se(xr.dropna().to_numpy(), cf["NW_LAG"])
    return {"months_full": int(x.notna().sum()), "months_recent": int(xr.notna().sum()), "start": str(x.dropna().index[0]),
            "end": str(x.dropna().index[-1]), "se_full": sf, "se_recent": sr, "mde_full": cs_mde(sf), "mde_recent": cs_mde(sr),
            "has_big": "big" in fr, "cost_per_month": turnover * cf["COST"]}


def cs_eligible_counts(price_map: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """새 151종목: 시장별로 달마다 모멘텀이 있는 종목 수(수익은 안 봄) → 대상 20개 이상인 달 수."""
    cf = CS_CONFIG
    rows = []
    for sym, df in (price_map or {}).items():
        d = _bt_naive(df)
        if d is None or not len(d):
            continue
        idx = d.index
        last = pd.Series(np.arange(len(idx)), index=idx).groupby(idx.to_period("M")).max()
        for m, i in last.items():
            if m >= pd.Period(cf["START_MONTH"], "M") and int(i) - cf["LOOKBACK"] >= 0:
                rows.append({"month": m, "market": lec_market_of(sym)})
    if not rows:
        return {"US": {"months_ok": 0}, "KR": {"months_ok": 0}}
    c = pd.DataFrame(rows).groupby(["market", "month"]).size()
    out = {}
    for mk in ("US", "KR"):
        s = c.loc[mk] if mk in c.index.get_level_values(0) else pd.Series(dtype=int)
        ok = s[s >= cf["MIN_STOCKS"]]
        out[mk] = {"months_ok": int(len(ok)), "first_ok": str(ok.index[0]) if len(ok) else None,
                   "median_n": float(s.median()) if len(s) else 0.0}
    return out


def run_cs_count(seen545: Dict[str, pd.DataFrame], new151: Dict[str, pd.DataFrame], us_mom: Optional[str], us_ff: Optional[str]) -> Dict[str, Any]:
    est = cs_seen_estimates(seen545)
    p1 = cs_part1_noise(us_mom, us_ff, est["turnover"]["US"] or 0.0) if (us_mom and us_ff) else None
    return {"config_version": CS_CONFIG["VERSION"], "seen": est, "part1": p1, "eligible": cs_eligible_counts(new151),
            "n_new": len(new151), "n_seen": len(seen545)}


def run_cs(new151: Dict[str, pd.DataFrame], index_map: Dict[str, pd.DataFrame], turnover_us: float, us_mom: str, us_ff: str,
           dx_mom: Optional[str] = None, dx_ff: Optional[str] = None, french_date: Optional[str] = None) -> Dict[str, Any]:
    p1 = run_cs_part1(us_mom, us_ff, turnover_us, dx_mom, dx_ff)
    p2 = run_cs_part2(new151, index_map)
    both = p1["judge"]["passed"] and p2["judge"]["passed"]
    if both:
        verdict = "통과 — ①·② 모두"
    elif p1["judge"]["passed"]:
        verdict = "① 통과 · ② 미통과 — 대형주 모멘텀은 비용 뒤 남지만 우리 종목·방식으로는 확인 안 됨"
    elif p2["judge"]["passed"]:
        verdict = "② 통과 · ① 미통과 — 생존 편향 없는 자료에서 확인 안 됨"
    else:
        verdict = "미통과 — ①·② 모두"
    return {"config_version": CS_CONFIG["VERSION"], "french_date": french_date, "part1": p1, "part2": p2, "passed": both, "verdict": verdict}


# ---------------------------------------------------------------------
# 보고서
# ---------------------------------------------------------------------
def _cs_pp(v: Optional[float], d: int = 2) -> str:
    return "-" if v is None or not np.isfinite(v) else f"{v * 100:+.{d}f}%"


def _cs_t(v: Optional[float]) -> str:
    return "-" if v is None or not np.isfinite(v) else f"{v:+.2f}"


def _cs_u(v: Optional[float], d: int = 1) -> str:
    return "-" if v is None or not np.isfinite(v) else f"{v * 100:.{d}f}%"


def _cs_mde_txt(v: Optional[float]) -> str:
    return "-" if v is None else f"{v * 100:.2f}%p/월"


def render_cs_count_report(r: Dict[str, Any], names: Optional[Dict[str, str]] = None, generated_at: Optional[str] = None) -> str:
    s, p1, el = r["seen"], r["part1"], r["eligible"]
    L = ["# H-CS1 횡단면 모멘텀 — 실행 전 (수익 평균은 보지 않음)", "",
         f"- 생성: {generated_at or '-'} · 설정 {r['config_version']} · 정의 data/backtest/CS1_PREREG.md. 이 숫자를 보고 규칙을 바꾸지 않는다.", "",
         "## 비용에 쓸 회전율 (545종목 — 이미 쓴 자료)", "",
         f"- 상위 20% 같은 비중의 월평균 한쪽 회전율: 미국 {_cs_u(s['turnover'].get('US'))} · 한국 {_cs_u(s['turnover'].get('KR'))}"
         f" → ① 비용 = 미국 회전율 × 0.30% = 월 {_cs_u((s['turnover'].get('US') or 0) * CS_CONFIG['COST'], 3)}", "",
         "## 가려낼 수 있는 최소 차이 (검정력 80%, t ≥ 2, Newey-West 시차 3)", "",
         "| 질문 | 달 수 | 표준오차 | 최소 차이 |", "|---|---|---|---|"]
    if p1:
        L.append(f"| ① 미국 대형·모멘텀 상위 30% − 시장, 전체 {p1['start']}~{p1['end']} | {p1['months_full']:,} | {_cs_u(p1['se_full'], 3)} | {_cs_mde_txt(p1['mde_full'])} |")
        L.append(f"| ① 같은 것, 2000년 이후 | {p1['months_recent']:,} | {_cs_u(p1['se_recent'], 3)} | {_cs_mde_txt(p1['mde_recent'])} |")
    else:
        L.append("| ① French 자료를 받지 못함 | - | - | - |")
    L.append(f"| ② 151종목 크기 무작위 부분집합(545종목에서 {CS_CONFIG['MDE_SUBSETS']}번) | {s['subset_months']} | {_cs_u(s['subset_se'], 3)} | {_cs_mde_txt(s['mde_part2'])} |")
    L += ["", "## 새 151종목 (대상 종목 수만)", "",
          f"- 받은 종목 {r['n_new']}개 · 대상 20개 이상인 달: 미국 {el['US']['months_ok']}개월(처음 {el['US'].get('first_ok')}, 달마다 중앙 {el['US'].get('median_n', 0):.0f}종목) · "
          f"한국 {el['KR']['months_ok']}개월(처음 {el['KR'].get('first_ok')}, 중앙 {el['KR'].get('median_n', 0):.0f}종목)"]
    if p1:
        L.append(f"- ① 대형 전체 대비(참고) 계산 가능: {'예' if p1['has_big'] else '아니오(기업 수·평균 시가총액 표 없음)'}")
    if names:
        L += ["", "## 한국 종목 이름 확인 (코드가 생각한 회사인지 보고 `--cs --names-ok`)", ""]
        L += [f"- {k}  {v}" for k, v in names.items()]
    L += ["", "다음: 숫자와 이름을 확인한 뒤 `python run_backtest.py --cs --names-ok`."]
    return "\n".join(L) + "\n"


def _cs_row(name: str, s: Dict[str, Any], mde: Optional[float] = None) -> str:
    return (f"| {name} | {s['n']} | {_cs_pp(s['mean'], 3)} | {_cs_pp(s.get('ann'), 1) if s['mean'] is not None else '-'} | {_cs_t(s['t'])} | "
            f"{_cs_mde_txt(mde) if mde is not None else '-'} | {_cs_pp(s.get('worst'), 1)} | {_cs_pp(s.get('mdd'), 1)} |")


def render_cs_report(r: Dict[str, Any], generated_at: Optional[str] = None) -> str:
    p1, p2 = r["part1"], r["part2"]
    hdr = "| 구분 | 달 | 월평균 초과(비용 뒤) | 연 환산 | t | 최소 차이 | 최악의 달 | 최대 낙폭(상대) |"
    sep = "|---|---|---|---|---|---|---|---|"
    L = ["# H-CS1 횡단면 모멘텀 — 결과", "",
         f"- 생성: {generated_at or '-'} · 설정 {r['config_version']} · French 자료 받은 날 {r.get('french_date') or '-'} · 정의 data/backtest/CS1_PREREG.md",
         f"## 판정: {r['verdict']}", "",
         "### ① 미국 대형주 모멘텀 상위 30%(시가총액 가중) − 시장 (Ken French, 상장폐지 포함)", "",
         f"- 비용 = 추정 회전율 {_cs_u(p1['turnover'])} × 0.30% = 월 {_cs_u(p1['cost_per_month'], 3)}. 판정: 전체 t ≥ 2 그리고 2000년 이후 > 0 → "
         + ("통과" if p1["judge"]["passed"] else "미통과(" + " · ".join(p1["judge"]["reasons"]) + ")"), "", hdr, sep,
         _cs_row("전체", p1["full"], p1["mde_full"]), _cs_row("2000년 이후", p1["recent"], p1["mde_recent"])]
    for k, s in p1["periods"].items():
        L.append(_cs_row(f"참고 · {k}", s))
    L.append(_cs_row("참고 · 비용 빼기 전 전체", p1["gross_full"]))
    L.append(_cs_row("참고 · 비용 빼기 전 2000년 이후", p1["gross_recent"]))
    if "vs_big" in p1:
        L.append(_cs_row("참고 · 대형 전체 대비 전체", p1["vs_big"]["full"]))
        L.append(_cs_row("참고 · 대형 전체 대비 2000년 이후", p1["vs_big"]["recent"]))
    dx = p1.get("developed_ex_us") or {}
    if "full" in dx:
        L.append(_cs_row("참고 · 선진국(미국 제외) 전체", dx["full"]))
        L.append(_cs_row("참고 · 선진국(미국 제외) 2000년 이후", dx["recent"]))
    elif "error" in dx:
        L.append(f"- 선진국 자료 읽기 실패: {dx['error']}")
    s, j = p2["stats"], p2["judge"]
    L += ["", "### ② 새 151종목 — 상위 20% 같은 비중 − 같은 종목 같은 비중 (실제 회전율 × 0.30%)", "",
          f"- 종목 {p2['n_tickers']}개 · 달마다 평균 대상 미국 {p2['n_stocks'].get('US') or 0:.0f} · 한국 {p2['n_stocks'].get('KR') or 0:.0f}종목 · "
          f"상위 20% 월평균 회전율 미국 {_cs_u(p2['turnover'].get('US'))} · 한국 {_cs_u(p2['turnover'].get('KR'))}",
          "- 판정: " + ("통과" if j["passed"] else "미통과(" + " · ".join(j["reasons"]) + ")"), "", hdr, sep,
          _cs_row("전체(미국·한국 평균)", s, p2["mde"]), _cs_row("미국", p2["us"]), _cs_row("한국", p2["kr"])]
    if s["mean"] is not None:
        L.append(f"| 앞 절반 / 뒤 절반 | | {_cs_pp(s['first'], 3)} / {_cs_pp(s['second'], 3)} | | | | | |")
    nm = {"top10": "참고 · 상위 10%", "vadj": "참고 · 변동성 대비 모멘텀 상위 20%", "vs_index_US": "참고 · 미국 상위 20% − ^GSPC(지수, 배당 빼고)",
          "vs_index_KR": "참고 · 한국 상위 20% − ^KS11(지수)"}
    for k, v in p2["refs"].items():
        L.append(_cs_row(nm.get(k, k), v))
    tx = p2.get("tax_us") or {}
    if tx.get("years"):
        L += ["", f"- 세금 예시(참고, 미국 상위 20%, 1억 원, 해마다 그해 이익을 모두 실현한다고 단순 가정): 연평균 세전 {_cs_pp(tx['pre_mean'], 1)} → 세후 {_cs_pp(tx['after_mean'], 1)}"
                  f"(연 {_cs_pp(tx['drag'], 1)} 차이, {tx['years']}년). 지수 ETF를 들고만 있으면 팔 때까지 세금이 없다."]
    L += ["", "- 결과를 보고 규칙·숫자·기간을 바꾸지 않는다(CS1_PREREG.md)."]
    return "\n".join(L) + "\n"


def cs_log_lines(r: Dict[str, Any], generated_at: str) -> List[str]:
    p1, p2 = r["part1"], r["part2"]
    return [f"- {generated_at} H-CS1 ① 미국 대형 모멘텀 − 시장(비용 뒤): 전체 {_cs_pp(p1['full']['mean'], 3)}/월 t {_cs_t(p1['full']['t'])} · 2000~ "
            f"{_cs_pp(p1['recent']['mean'], 3)} t {_cs_t(p1['recent']['t'])} → {'통과' if p1['judge']['passed'] else '미통과'}",
            f"- {generated_at} H-CS1 ② 새 151종목 상위 20% − 같은 비중(비용 뒤): {_cs_pp(p2['stats']['mean'], 3)}/월 t {_cs_t(p2['stats']['t'])} → "
            f"{'통과' if p2['judge']['passed'] else '미통과'}"]
