"""SEC companyconcept 형식 가짜 자료 (tests/test_frame_b.py용).

fake_company(seed, mean_revert): 분기 EPS를 만들고 SEC와 같은 형식(10-Q 3개월·누적·전년 비교, 10-K 연간·전년 연간,
제출일 = 분기 말 + 40일·연말 + 60일)의 JSON과 가격을 돌려준다.
    mean_revert=False  가격이 이익과 무관한 무작위 경로 → 우위가 없어야 함
    mean_revert=True   가격 = TTM 이익 × PER, 로그 PER이 평균으로 돌아감 → 싸게 사면 우위가 있어야 함
"""
import numpy as np, pandas as pd

def quarter_ends(y0, y1):
    return [pd.Timestamp(f"{y}-{m:02d}-{d}") for y in range(y0, y1 + 1) for m, d in ((3, 31), (6, 30), (9, 30), (12, 31))]

def fake_concept(q_eps, y0=2007, lag_q=40, lag_k=60, today="2026-11-13"):
    """q_eps: {quarter_end: eps}. 10-Q마다 이번 분기 3개월·누적, 전년 같은 분기 비교; 10-K는 연간·전년 연간."""
    rows, today = [], pd.Timestamp(today)
    ends = sorted(q_eps)
    def add(s, e, v, f, form, fp):
        if f <= today:
            rows.append({"start": str(s.date()), "end": str(e.date()), "val": round(float(v), 4), "accn": "x", "fy": f.year,
                         "fp": fp, "form": form, "filed": str(f.date())})
    for e in ends:
        y = e.year; ys = pd.Timestamp(f"{y}-01-01"); qn = (e.month) // 3
        qs = pd.Timestamp(e.year, e.month - 2, 1)
        ytd = sum(v for k, v in q_eps.items() if k.year == y and k <= e)
        if qn < 4:
            f = e + pd.Timedelta(days=lag_q)
            add(qs, e, q_eps[e], f, "10-Q", f"Q{qn}")
            if qn > 1: add(ys, e, ytd, f, "10-Q", f"Q{qn}")
            prev = pd.Timestamp(y - 1, e.month, e.day)
            if prev in q_eps:
                add(pd.Timestamp(y - 1, e.month - 2, 1), prev, q_eps[prev], f, "10-Q", f"Q{qn}")
        else:
            f = e + pd.Timedelta(days=lag_k)
            add(ys, e, ytd, f, "10-K", "FY")
            ytd9 = sum(v for k, v in q_eps.items() if k.year == y and k.month <= 9)
            pya = sum(v for k, v in q_eps.items() if k.year == y - 1)
            if any(k.year == y - 1 for k in q_eps):
                add(pd.Timestamp(f"{y-1}-01-01"), pd.Timestamp(f"{y-1}-12-31"), pya, f, "10-K", "FY")
    return {"cik": 1, "taxonomy": "us-gaap", "tag": "EarningsPerShareDiluted", "units": {"USD/shares": rows}}

def fake_company(seed, mean_revert=False, idx=None):
    r = np.random.default_rng(seed)
    qe = quarter_ends(2007, 2026)
    qe = [e for e in qe if e <= pd.Timestamp("2026-09-30")]
    g = np.cumsum(r.normal(0.015, 0.06, len(qe)))
    q_eps = {e: float(0.5 * np.exp(v)) for e, v in zip(qe, g)}
    idx = idx if idx is not None else pd.bdate_range("2008-01-01", "2026-11-13")
    if mean_revert:
        # 가격 = TTM 이익 × PER, 로그 PER은 평균으로 돌아감 (우위가 있어야 하는 경우)
        ttm = pd.Series({e: sum(q_eps[k] for k in qe if e - pd.Timedelta(days=300) <= k <= e) for e in qe}).reindex(idx, method="ffill").bfill()
        lp = np.zeros(len(idx)); x = 0.0
        for t in range(len(idx)):
            x = x * 0.995 + r.normal(0, 0.02); lp[t] = x
        c = ttm.to_numpy() * 18 * np.exp(lp)
    else:
        c = 50 * np.exp(np.cumsum(r.normal(3e-4, 0.018, len(idx))))
    o = c * (1 + r.normal(0, 0.004, len(idx)))
    px = pd.DataFrame({"Open": o, "High": np.maximum(o, c) * 1.004, "Low": np.minimum(o, c) * 0.996, "Close": c}, index=idx)
    return px, fake_concept(q_eps)
