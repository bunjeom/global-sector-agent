"""종합 탭 T 실행 카드: 셋업이 없을 때 현재가 기반 가짜 진입·손절·목표가가 없어야 한다."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # 프로젝트 폴더의 engine 패키지를 불러오기 위해
import _offline  # noqa: F401  외부 네트워크 차단 (tests/_offline.py)
import copy, sys
from test_chain_helpers import run_chain, fake_earn, engine
FAILS = []
import numpy as np
from test_chain_helpers import prices
MACRO = {}
for _tk in ("NVDA","005930.KS"):
    _k=_tk.endswith(".KS"); MACRO[_tk]=engine.analyze_macro_track(ticker=_tk, stock_df=prices(start=1,seed=3), bench_df=prices(start=2600 if _k else 17000, seed=9), industry_category="메모리 (한국 상장)" if _k else "AI 반도체·네트워킹(팹리스)")
def chk(name, cond):
    print(("PASS " if cond else "FAIL ") + name); (None if cond else FAILS.append(name))

for tk, start in (("NVDA", 120.0), ("005930.KS", 60000.0)):
    is_krx = tk.endswith(".KS")
    last, fund_out, tech_out, synth = run_chain(tk, fake_earn("forward", is_krx), start)
    kw = dict(ticker=tk, current_price=last, earn_out=fake_earn("forward", is_krx), fund_out=fund_out,
              macro_out=engine.analyze_macro_track(ticker=tk, stock_df=tech_out["chart_df"] if False else None, bench_df=None) if False else None)
    # 1) 셋업 없음 강제
    t_no = copy.deepcopy(tech_out)
    t_no["t_ticket"] = {"is_executable": False, "reason": "활성 셋업 없음"}
    t_no["summary_banner"].update({"setup_id": None, "grade": None, "t_verdict": "관찰"})
    s_no = engine.synthesize_4tracks(ticker=tk, current_price=last, tech_out=t_no, earn_out=fake_earn("forward", is_krx),
                                     fund_out=fund_out, macro_out=synth.get("_macro_out_dummy") or engine.analyze_macro_track(
                                         ticker=tk, stock_df=tech_out["chart_df"][["Open","High","Low","Close","Volume"]] if "Volume" in tech_out["chart_df"] else tech_out["chart_df"],
                                         bench_df=None, industry_category="") if False else MACRO[tk],
                                     synth_cfg={"stock_name": tk, "days_to_earnings": None})
    c = s_no["t_action_card"]
    chk(f"{tk} 셋업없음: has_ticket False", c["has_ticket"] is False)
    chk(f"{tk} 셋업없음: 가격·수량 전부 None", all(c[k] is None for k in ("entry_price", "stop_price", "target_price", "shares", "expected_r", "time_stop_rule", "invalidation_rule")))
    chk(f"{tk} 셋업없음: 사유 있음", bool(c["no_ticket_reason"]))
    # 2) 셋업 있음 강제
    t_yes = copy.deepcopy(tech_out)
    ep = 133.57 if not is_krx else 61234
    t_yes["t_ticket"] = {"is_executable": True, "setup_id": "ST-1", "setup_name": "x", "entry_price": ep, "stop_price": ep * 0.95,
                         "shares": 10, "target_1": ep * 1.1, "expected_r": 2.0, "time_stop_days": 5, "time_stop_min_price": ep * 1.01,
                         "invalidation": "inv"}
    t_yes["summary_banner"].update({"setup_id": "ST-1", "grade": "A", "t_verdict": "진입"})
    s_yes = engine.synthesize_4tracks(ticker=tk, current_price=last, tech_out=t_yes, earn_out=fake_earn("forward", is_krx),
                                      fund_out=fund_out, macro_out=MACRO[tk], synth_cfg={"stock_name": tk, "days_to_earnings": None})
    c = s_yes["t_action_card"]
    chk(f"{tk} 셋업있음: has_ticket True", c["has_ticket"] is True)
    chk(f"{tk} 셋업있음: 진입가 보존(USD 센트 절사 없음)", c["entry_price"] == (int(ep) if is_krx else round(ep, 2)))
    chk(f"{tk} 셋업있음: 시간손절 문구", c["time_stop_rule"] and "0.00" not in c["time_stop_rule"])
print("FAILS:", FAILS); sys.exit(1 if FAILS else 0)
