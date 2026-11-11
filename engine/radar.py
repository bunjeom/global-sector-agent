"""실행 레이더: 종목 선정(select_radar_universe)과 실제 T 신호(build_radar_signals).

engine.py를 기능별로 나눈 모듈. 바깥에서는 `import engine` 후 `engine.이름`으로 쓴다(engine/__init__.py).
"""
from typing import Dict, List, Optional, Any
import pandas as pd

from .common import TECH_PARAMS, is_earnings_blackout, trading_days_until
from .backtest import _bt_summary
from .technical import analyze_technical_track
from .home import NARRATIVE_BASKETS_SPEC

# ==============================================================================
# 실행 레이더 — 실제 T(추세매매) 신호 (2026-11-09 추가)
# - 대상: '지금 강한 곳'·'다음 후보' 섹터 ETF의 상위 구성 종목 중 추세 통과·3개월 상대강도 상위 (나중에 사용자 관심 종목을 앞에 추가 가능)
# - 신호: analyze_technical_track(딥다이브와 같은 엔진)의 T 판정. V(가치)는 딥다이브에서만 계산한다.
# ==============================================================================
RADAR_CONFIG = {
    "PROXIES_PER_BASKET": 3,       # 섹터당 선정 종목 수
    "HOLDINGS_TOP_N": 10,          # 후보: ETF 상위 구성 종목 수
    "RANK_LOOKBACK_DAYS": 63,      # 순위: 약 3개월 기준 지수 대비 상대 수익률
    "TREND_MA_DAYS": 50,           # 추세 통과: 종가가 50일 평균 위
    "MIN_REL_3M": 0.0,             # 상대강도 통과: 3개월 수익률이 기준 지수보다 이만큼 이상 앞설 것
    "MAX_TICKERS": 18,             # 종목당 약 1.5초 → 최대 약 30초
    "PRICE_PERIOD": "2y",          # EMA200·250일 변동성 계산에 필요한 기간
    "MIN_BARS": 260,               # 이보다 짧으면 신호를 만들지 않음
    "FX_TICKER": "KRW=X",          # 미국 종목을 원화 기준 엔진에 넣기 위한 USD/KRW 환율
    "BENCH": {"US": "SPY", "KR": "^KS11"},
    "MARKET_CLOSE": {"US": ("America/New_York", 16, 15), "KR": ("Asia/Seoul", 15, 45)},  # 장 마감 + 여유 15분
    "DEFENSE_STATES": ["K", "D"],
    "BT_MIN_SAMPLE": 5,            # 과거 검증 표본이 이보다 적으면 숫자 대신 '표본 부족'
    # 엔진 셋업 → 과거 검증(run_setup_backtest_validation) 항목. 없는 셋업은 검증 대상 아님
    "BT_SETUP_FAMILY": {"ST-1": "ST-1", "ST-2a": "ST-2", "ST-2b": "ST-2"},  # 붕괴 경계·하락 진행 → 보유자 방어 신호 (방어가 공격보다 우선)
}


def select_radar_baskets(evaluation: dict) -> List[Dict[str, str]]:
    """섹터 흐름 평가에서 레이더 대상 섹터를 고른다: '지금 강한 곳' 3개 + '다음 후보' 3개."""
    out: List[Dict[str, str]] = []
    for role, key in (("지금 강한 곳", "leading_top3"), ("다음 후보", "igniting_top3")):
        for it in (evaluation or {}).get(key, []):
            if it.get("basket"):
                out.append({"basket": it["basket"], "basket_role": role})
    return out


def _radar_symbol_market(sym: Any) -> Optional[str]:
    """레이더가 다룰 수 있는 종목인지: 미국 상장(접미사 없음) → US, 한국(.KS/.KQ) → KR, 그 외 해외 상장 → None(통화가 달라 제외)"""
    s = str(sym or "").strip().upper()
    if not s or s.startswith("^"):
        return None
    if s.endswith((".KS", ".KQ")):
        return "KR"
    if "." in s:
        return None
    return "US"


def fetch_etf_top_holdings(etf: str) -> Optional[Dict[str, Any]]:
    """yfinance funds_data.top_holdings — ETF 상위 구성 종목(보통 10개)과 비중(소수). 실패하거나 없으면 None."""
    try:
        import yfinance as yf
        th = yf.Ticker(etf).funds_data.top_holdings
        if th is None or len(th) == 0:
            return None
        df_h = th if "Symbol" in th.columns else th.reset_index()
        rows = []
        for _, r in df_h.iterrows():
            sym = r.get("Symbol")
            if sym is None or str(sym).strip() == "":
                continue
            w = pd.to_numeric(r.get("Holding Percent"), errors="coerce")
            rows.append({"ticker": str(sym).strip().upper(), "name": str(r.get("Name", "") or ""),
                         "weight": None if pd.isna(w) else float(w)})
        return {"holdings": rows, "source": f"yfinance 상위 구성 종목 ({etf})"} if rows else None
    except Exception:
        return None


def build_basket_candidates(basket: str, holdings: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    섹터 후보 종목: ETF 실제 상위 구성 종목(미국·한국 상장만, 상위 HOLDINGS_TOP_N개).
    구성 종목이 부족하거나 없으면 NARRATIVE_BASKETS_SPEC의 대표 종목으로 보충하고 출처에 그 사실을 남긴다.
    """
    spec = next((s for s in NARRATIVE_BASKETS_SPEC if s["ticker"] == basket), None) or {}
    cands: List[Dict[str, Any]] = []
    for h in ((holdings or {}).get("holdings") or [])[:RADAR_CONFIG["HOLDINGS_TOP_N"]]:
        mkt = _radar_symbol_market(h.get("ticker"))
        if mkt:
            cands.append({**h, "market": mkt})
    n_hold = len(cands)
    if n_hold >= RADAR_CONFIG["PROXIES_PER_BASKET"]:
        return {"source": "ETF 구성 종목", "candidates": cands}
    have = {c["ticker"] for c in cands}
    for p in spec.get("proxies", []):
        mkt = _radar_symbol_market(p)
        if mkt and p not in have:
            cands.append({"ticker": p, "name": "", "weight": None, "market": mkt})
    src = "구성 종목 + 대표 종목 보충" if n_hold > 0 else "대표 종목 (구성 데이터 없음)"
    return {"source": src, "candidates": cands}


def rank_basket_candidates(cands: List[Dict[str, Any]], price_map: Dict[str, pd.DataFrame],
                           bench_map: Dict[str, Optional[pd.DataFrame]]) -> List[Dict[str, Any]]:
    """
    후보 순위: 추세 유지(종가 > TREND_MA_DAYS일 평균)이고 기준 지수보다 강한(상대 수익률 > MIN_REL_3M) 종목만 '통과'로 보고,
    RANK_LOOKBACK_DAYS(약 3개월) 수익률이 기준 지수보다 얼마나 앞섰는지(상대 수익률) 순으로 정렬한다.
    """
    L, M = RADAR_CONFIG["RANK_LOOKBACK_DAYS"], RADAR_CONFIG["TREND_MA_DAYS"]
    bench_ret: Dict[str, Optional[float]] = {}
    for mkt, bdf in (bench_map or {}).items():
        bc = pd.to_numeric(bdf["Close"], errors="coerce").dropna() if bdf is not None else pd.Series(dtype=float)
        bench_ret[mkt] = float(bc.iloc[-1] / bc.iloc[-1 - L] - 1.0) if len(bc) > L else None
    out: List[Dict[str, Any]] = []
    for c in cands:
        rec = dict(c)
        df_c = (price_map or {}).get(c["ticker"])
        cl = pd.to_numeric(df_c["Close"], errors="coerce").dropna() if df_c is not None else pd.Series(dtype=float)
        if len(cl) <= max(L, M):
            rec.update(eligible=False, rel_3m=None, why="가격 데이터 부족")
            out.append(rec)
            continue
        ret = float(cl.iloc[-1] / cl.iloc[-1 - L] - 1.0)
        above = bool(float(cl.iloc[-1]) > float(cl.iloc[-M:].mean()))
        b_ret = bench_ret.get(c.get("market", "US"))
        rel = None if b_ret is None else ret - b_ret
        strong = rel is not None and rel > RADAR_CONFIG["MIN_REL_3M"]
        if rel is None:
            why = "기준 지수 데이터 없음"
        elif not above:
            why = f"{M}일선 아래"
        elif not strong:
            why = "3개월 기준 지수보다 약함"
        else:
            why = None
        rec.update(ret_3m=round(ret, 4), rel_3m=None if rel is None else round(rel, 4), above_ma=above,
                   eligible=bool(above and strong), why=why)
        out.append(rec)
    return sorted(out, key=lambda r: (bool(r.get("eligible")), r["rel_3m"] if r.get("rel_3m") is not None else -9.0), reverse=True)


def select_radar_universe(baskets: List[Dict[str, str]], candidates_map: Dict[str, Dict[str, Any]],
                          price_map: Dict[str, pd.DataFrame], bench_map: Dict[str, Optional[pd.DataFrame]]) -> Dict[str, Any]:
    """
    강한 섹터마다 후보를 순위 매겨 추세 통과 종목 중 상위 PROXIES_PER_BASKET개를 고른다(전체 최대 MAX_TICKERS개).
    통과 종목이 모자라면 모자란 대로 둔다(약한 종목으로 채우지 않음). 같은 종목은 먼저 나온 섹터에만 넣는다.
    반환: {"universe": [...], "log": 섹터별 선정 과정}
    """
    universe: List[Dict[str, Any]] = []
    log: List[Dict[str, Any]] = []
    seen = set()
    for b in baskets or []:
        cinfo = (candidates_map or {}).get(b["basket"]) or {"source": "후보 없음", "candidates": []}
        ranked = rank_basket_candidates(cinfo["candidates"], price_map, bench_map)
        picked: List[str] = []
        for r in ranked:
            if len(picked) >= RADAR_CONFIG["PROXIES_PER_BASKET"] or not r.get("eligible"):
                break
            if r["ticker"] in seen:
                continue
            seen.add(r["ticker"])
            picked.append(r["ticker"])
            note = [f"{b['basket']} 비중 {r['weight'] * 100:.1f}%"] if r.get("weight") is not None else [f"{b['basket']} 대표 종목"]
            note.append(f"3개월 기준지수 대비 {r['rel_3m'] * 100:+.1f}%p")
            universe.append({"ticker": r["ticker"], "market": r["market"], "basket": b["basket"],
                             "basket_role": b["basket_role"], "origin": cinfo["source"],
                             "selection_note": " · ".join(note)})
        log.append({"섹터": b["basket"], "구분": b["basket_role"], "후보 출처": cinfo["source"],
                    "후보": len(ranked), "추세 통과": sum(1 for r in ranked if r.get("eligible")),
                    "선정": ", ".join(picked) or "없음"})
    return {"universe": universe[:RADAR_CONFIG["MAX_TICKERS"]], "log": log}


def _drop_incomplete_bar(df: Optional[pd.DataFrame], market: str, now=None) -> Optional[pd.DataFrame]:
    """오늘 장이 아직 안 끝났으면 마지막(진행 중) 일봉을 뺀다 — 미완성 봉으로 신호를 만들지 않기 위해."""
    if df is None or len(df) == 0:
        return df
    tz, hh, mm = RADAR_CONFIG["MARKET_CLOSE"].get(market, RADAR_CONFIG["MARKET_CLOSE"]["US"])
    now_l = pd.Timestamp.now(tz=tz) if now is None else pd.Timestamp(now).tz_convert(tz)
    last = pd.Timestamp(df.index[-1])
    last_date = (last.tz_convert(tz) if last.tzinfo is not None else last).date()
    if last_date == now_l.date() and (now_l.hour, now_l.minute) < (hh, mm):
        return df.iloc[:-1]
    return df


def build_radar_signal(item: Dict[str, Any], df: Optional[pd.DataFrame], bench_df: Optional[pd.DataFrame],
                       vix_series: Optional[pd.Series], account_cfg: Optional[Dict[str, Any]],
                       fx_usdkrw: Optional[float] = None) -> Dict[str, Any]:
    """
    한 종목의 실제 T 신호를 레이더 형식(apply_gate_action_matrix 입력)으로 만든다.
    - 가치 트랙(매수 기준선·매도 목표가·Thesis)은 넣지 않는다 → 엔진이 임의 매수선을 만들지 않도록 빈 값 전달.
    - 미국 종목은 fx_to_krw(USD/KRW)를 넘겨, 엔진이 유동성 기준·계좌 금액만 원화로 환산하게 한다(가격은 달러 그대로).
    반환 kind: "signal"(실행·관찰·방어) | "none"(신호 없음) | "error"(계산 불가)
    """
    mkt = item.get("market", "US")
    base = {**item, "source": "엔진 T 트랙 (yfinance 일봉)", "is_sample": False, "tranche_stage": 0}
    try:
        df = _drop_incomplete_bar(df, mkt)
        n_bars = 0 if df is None else len(df)
        as_of = str(df.index[-1])[:10] if n_bars else None
        if n_bars < RADAR_CONFIG["MIN_BARS"]:
            return {**base, "kind": "error", "as_of": as_of,
                    "reason": f"일봉 {n_bars}개 ({RADAR_CONFIG['MIN_BARS']}개 필요)"}
        if mkt == "US" and (not fx_usdkrw or float(fx_usdkrw) <= 0):
            return {**base, "kind": "error", "as_of": as_of, "reason": "USD/KRW 환율 데이터 없음"}
        px = df.dropna(subset=["Open", "High", "Low", "Close"])
        neutral_value = {"thesis_status": "미확인", "kill_flags": [], "buy_tranches": [], "sell_target": None,
                         "position_cap": TECH_PARAMS["MAX_POS_RATIO_TOTAL"], "real_signal": "중립",
                         "macro_regime_override": None}
        days_to_earn = trading_days_until(as_of, item.get("next_earnings"))
        blackout = is_earnings_blackout(days_to_earn)
        cfg = {**(account_cfg or {}), "is_krx": mkt == "KR", "fx_to_krw": 1.0 if mkt == "KR" else float(fx_usdkrw),
               "earnings_blackout": blackout}
        out = analyze_technical_track(px, bench_df, vix_series, value_macro_input=neutral_value, account_cfg=cfg)
    except Exception as e:
        return {**base, "kind": "error", "as_of": None, "reason": f"계산 오류: {type(e).__name__}: {e}"[:200]}

    sb, ls, tkt = out["summary_banner"], out["lifecycle_state"], (out.get("t_ticket") or {})
    to_px = (lambda v: round(float(v), 2)) if mkt == "US" else (lambda v: int(v))
    to_txt = str
    detail = {
        "currency": "USD" if mkt == "US" else "KRW",
        "setup_id": sb.get("setup_id"), "grade": sb.get("grade"),
        "state": ls.get("state"), "state_label": ls.get("state_label"),
        "headline": to_txt(sb.get("headline_reason", "")),
        "next_cond": to_txt(sb.get("next_or_end_condition", "")),
        "as_of": as_of,
        "next_earnings": item.get("next_earnings") if days_to_earn is not None else None,
        "days_to_earnings": days_to_earn,
        "earnings_blackout": False,
    }
    fam = RADAR_CONFIG["BT_SETUP_FAMILY"].get(sb.get("setup_id") or "")
    detail["bt_family"] = fam
    bt_all = out.get("backtest_validation") or {}
    detail["bt"] = (bt_all.get("by_setup") or {}).get(fam) if fam else None
    detail["bt_r_by_setup"] = bt_all.get("r_by_setup") or {}   # 레이더 전체 합산용 (build_radar_signals에서 제거)
    if isinstance(tkt.get("entry_price"), (int, float)) and isinstance(tkt.get("stop_price"), (int, float)):
        detail["entry"] = to_px(tkt["entry_price"])
        detail["stop"] = to_px(tkt["stop_price"])
        if isinstance(tkt.get("target_1"), (int, float)):
            detail["target_1"] = to_px(tkt["target_1"])
        if tkt.get("expected_r") is not None:
            detail["expected_r"] = tkt.get("expected_r")

    if ls.get("state") in RADAR_CONFIG["DEFENSE_STATES"]:
        sig_type, action = "RISK_DEFENSE", f"보유 중이면 비중 축소·손절 검토 ({ls.get('state_label')})"
    elif sb.get("t_verdict") == "진입":
        sig_type, action = "T_ENTRY", f"T {sb.get('setup_id')} {sb.get('grade')}등급 진입"
    elif sb.get("t_verdict") == "대기":
        sig_type, action = "T_WATCH", f"T {sb.get('setup_id')} {sb.get('grade')}등급 대기"
    else:
        why = detail["headline"] + (f" · 실적 발표 D-{days_to_earn} (신규 진입 제한 기간)" if blackout else "")
        return {**base, **detail, "kind": "none", "reason": why}
    if sig_type in ("T_ENTRY", "T_WATCH") and blackout:
        # 종합 결론(synthesize_4tracks)과 같은 규칙: 실적 발표 D-3 ~ 당일 전 셋업 신규 진입 금지. 방어 신호는 막지 않는다.
        return {**base, **detail, "kind": "signal", "signal_type": sig_type, "earnings_blackout": True,
                "action_text": f"T {sb.get('setup_id')} {sb.get('grade')}등급 — 실적 발표 D-{days_to_earn}로 신규 진입 보류",
                "suggested_weight": 0, "is_bnf_oversold": False}
    return {**base, **detail, "kind": "signal", "signal_type": sig_type, "action_text": action,
            "suggested_weight": 100,
            "is_bnf_oversold": bool(sig_type == "T_ENTRY" and sb.get("setup_id") == "ST-1")}


def build_radar_signals(universe: List[Dict[str, Any]], price_map: Dict[str, pd.DataFrame],
                        bench_map: Dict[str, Optional[pd.DataFrame]], vix_series: Optional[pd.Series],
                        account_cfg: Optional[Dict[str, Any]], fx_usdkrw: Optional[float]) -> Dict[str, Any]:
    """레이더 대상 전체를 계산해 신호 / 신호 없음 / 계산 불가로 나눈다."""
    res: Dict[str, Any] = {"signals": [], "no_signal": [], "errors": []}
    for item in universe or []:
        r = build_radar_signal(item, (price_map or {}).get(item["ticker"]),
                               (bench_map or {}).get(item.get("market", "US")), vix_series, account_cfg, fx_usdkrw)
        {"signal": res["signals"], "none": res["no_signal"]}.get(r["kind"], res["errors"]).append(r)
    # 과거 검증 합산: 레이더 대상 종목 전체의 같은 셋업 결과를 모은다(종목당 표본이 작아서).
    evaluated = res["signals"] + res["no_signal"]
    pooled: Dict[str, Any] = {}
    for fam in ("ST-1", "ST-2"):
        rs = [v for x in evaluated for v in (x.get("bt_r_by_setup") or {}).get(fam, [])]
        pooled[fam] = {**_bt_summary(rs), "n_tickers": len(evaluated)}
    for x in evaluated:
        x.pop("bt_r_by_setup", None)
        x["bt_pooled"] = pooled.get(x.get("bt_family")) if x.get("bt_family") else None
    res["bt_pooled"] = pooled
    return res
