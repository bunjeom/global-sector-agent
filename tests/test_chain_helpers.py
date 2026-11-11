import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # 프로젝트 폴더의 engine 패키지를 불러오기 위해
import _offline  # noqa: F401  외부 네트워크 차단 (tests/_offline.py)
import sys, importlib
import os
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
engine = importlib.import_module("engine")
NO = engine.NO_DATA

def prices(n=420, start=100.0, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end="2026-11-09", periods=n)
    c = start * np.exp(np.cumsum(rng.normal(0.0006, 0.015, n)))
    df = pd.DataFrame({"Open": c * 0.995, "High": c * 1.01, "Low": c * 0.99, "Close": c, "Volume": rng.integers(1e6, 3e6, n)}, index=idx)
    return df

def run_chain(ticker, earn_out, price_start):
    is_krx = ticker.endswith((".KS", ".KQ"))
    stock = prices(start=price_start, seed=3)
    bench = prices(start=2600 if is_krx else 17000, seed=9)
    vix = pd.Series(18 + 3 * np.sin(np.arange(len(stock)) / 20.0), index=stock.index)
    last = float(stock["Close"].iloc[-1])
    macro_out = engine.analyze_macro_track(ticker=ticker, stock_df=stock, bench_df=bench,
                                           industry_category="메모리 (한국 상장)" if is_krx else "AI 반도체·네트워킹(팹리스)")
    fund_out = engine.analyze_fundamental_track(ticker=ticker, current_price=last, earnings_output=earn_out,
                                                macro_input={"rf_10y": 0.035 if is_krx else 0.042, "market_erp": 0.055}, fund_override={})
    vmi = {"thesis_status": fund_out["thesis_status"], "kill_flags": [], "buy_tranches": fund_out["buy_tranches"],
           "sell_target": fund_out["sell_target"], "position_cap": fund_out["position_cap"],
           "real_signal": earn_out["verdict"]["real_signal"], "analyzed_at": "2026-11-10", "macro_regime_override": None}
    acct = {"total_equity": 100_000_000, "t_sleeve_ratio": 0.30, "is_krx": is_krx, "fx_to_krw": 1.0 if is_krx else 1400.0,
            "hard_block_evt": False, "earnings_blackout": False, "mktcap_krw": 1_000_000_000_000, "loss_streak": 0, "existing_v_weight": 0.0}
    tech_out = engine.analyze_technical_track(df=stock, bench_df=bench, vix_series=vix, value_macro_input=vmi, account_cfg=acct)
    synth = engine.synthesize_4tracks(ticker=ticker, current_price=last, tech_out=tech_out, earn_out=earn_out, fund_out=fund_out,
                                      macro_out=macro_out, synth_cfg={"stock_name": ticker, "is_halted": False, "bad_evt_flag": False,
                                                                       "pead_lockout_days_left": 0, "days_to_earnings": None})
    return last, fund_out, tech_out, synth

def fake_earn(basis, is_krx, regime="G2", sector="구조성장형"):
    """analyze_earnings_track 출력 형태를 갖는 가짜 결과 (basis별)."""
    base = 4000.0 if is_krx else 7.4
    e = {"forward": (base, base * 1.2, base * 0.8), "forward_base_only": (base, None, None),
         "ttm_reference": (None, None, None), "none": (None, None, None)}[basis]
    return {
        "regime": {"earnings_regime": regime, "regime_label": "테스트", "sector_type": sector, "cycle_stage": None, "growth_acceleration": NO},
        "verdict": {"score": None, "grade": "⚪", "real_signal": "중립", "headline_reason": "t", "peak_out_warning": False},
        "projections": {"eps_basis": basis, "eps_basis_label": basis, "ttm_eps": (base * 0.9) if basis != "none" else None,
                        "eps_1y": {"formula": "x", "base": e[0], "bull": e[1], "bear": e[2], "cycle_bear": None},
                        "eps_2y": {"formula": "x", "base": (base * 1.2) if basis.startswith("forward") else None, "bull": None, "bear": None},
                        "certainty_adjustment": 0.0, "certainty_label": f"{NO} (미확인)"},
        "sustainability_metrics": {"incremental_opm": None, "average_opm": None, "rpo_yoy_growth": None, "sales_yoy_growth": None,
                                   "operating_leverage_active": None, "design_wins_note": NO},
        "revision_dynamics": {"status": NO, "change_90d": None, "change_30d": None, "inflection_flag": False, "table": []},
        "quality_checks": {"warning_count": 0, "checked_count": 0, "is_investment_phase_fcf": False, "margin_squeeze_detected": False, "items": []},
        "thesis_driver": {"status": "미확인", "invalidation_condition": NO, "current_status": NO, "capital_structure_note": NO,
                          "mispricing_type": NO, "mispricing_detail": NO},
        "evaluation_axes": [], "e1_surprise_table": [],
    }

