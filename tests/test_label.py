"""종합 라벨: 하드 킬 / 가치 함정 / 고평가(위험 대비 기대수익 부족)를 구분하는지 본다."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # 프로젝트 폴더의 engine.py를 불러오기 위해
import _offline  # noqa: F401  외부 네트워크 차단 (tests/_offline.py)
import copy, sys
from test_chain_helpers import run_chain, fake_earn, engine, prices
FAILS = []
def chk(n, c):
    print(("PASS " if c else "FAIL ") + n); (None if c else FAILS.append(n))

tk = "AAPL"
earn = fake_earn("forward", False, regime="G3", sector="성숙형")
last, fund_out, tech_out, _ = run_chain(tk, earn, 300.0)
macro = engine.analyze_macro_track(ticker=tk, stock_df=prices(start=1, seed=3), bench_df=prices(start=17000, seed=9),
                                   industry_category="AI 반도체·네트워킹(팹리스)")
def synth(fo, cfg=None, eo=earn):
    c = {"stock_name": tk, "days_to_earnings": None}; c.update(cfg or {})
    return engine.synthesize_4tracks(ticker=tk, current_price=last, tech_out=tech_out, earn_out=eo, fund_out=fo, macro_out=macro, synth_cfg=c)

def with_vt(**kw):
    fo = copy.deepcopy(fund_out)
    fo["valuation_targets"].update(kw)
    if not fo["valuation_targets"].get("buy_tranches"):
        fo["valuation_targets"]["buy_tranches"] = [{"tranche": 1, "price": last * 0.4, "weight": 0.015}, {"tranche": 2, "price": last * 0.35, "weight": 0.015}]
    fo["price_qualification"]["status"] = "DISQUALIFIED"
    return fo

# A) 애플형: 실적 정상(G3), 하드 킬 없음, 현재가가 비싸 MDD Ratio 음수
s = synth(with_vt(mdd_ratio=-0.79, expected_return_weighted=-0.604, sell_target=last * 0.71))
v, a = s["v_action_card"], s["archetype"]
print("  A:", v["price_qual_code"], "|", v["badge_text"], "|", a["code"], a["title"])
chk("A 하드킬 아님", not s["hard_kill_gate"]["is_killed"])
chk("A 코드 POOR_RISK_REWARD", v["price_qual_code"] == "POOR_RISK_REWARD")
chk("A ARCH_8 아님", a["code"] != "ARCH_8")
chk("A 제목에 '하드 킬'·'청산' 없음", "하드 킬" not in a["title"] and "청산" not in a["title"])
chk("A V 배지에 '가치 함정' 없음", "가치 함정" not in v["badge_text"])
chk("A 표시명 한글", v["price_qual_label"].startswith("고평가"))
L1 = s["three_line_briefing"]["line_1_action"]; print("  A line1:", L1[:160])
chk("A 3줄 결론이 고평가/목표가 초과를 설명", ("부족" in L1) or ("목표가" in L1 and "높아" in L1))
chk("A 3줄 결론에 청산 원칙 없음", "즉시 청산" not in L1)

# B) 진짜 하드 킬(기업 고유 악재)
s = synth(with_vt(mdd_ratio=0.5, expected_return_weighted=0.2), cfg={"bad_evt_flag": True})
v, a = s["v_action_card"], s["archetype"]
print("  B:", v["price_qual_code"], "|", v["badge_text"], "|", a["code"], a["title"])
chk("B 하드킬", s["hard_kill_gate"]["is_killed"])
chk("B ARCH_8 + 하드 킬 제목", a["code"] == "ARCH_8" and "하드 킬" in a["title"])
chk("B 코드 HARD_KILL", v["price_qual_code"] == "HARD_KILL")

# C) G4 피크아웃 + 위험 대비 기대수익 부족 → 가치 함정
earn_g4 = fake_earn("forward", False, regime="G4", sector="성숙형")
s = synth(with_vt(mdd_ratio=0.05, expected_return_weighted=0.01), eo=earn_g4)
v, a = s["v_action_card"], s["archetype"]
print("  C:", v["price_qual_code"], "|", v["badge_text"], "|", a["code"], a["title"])
chk("C 코드 VALUE_TRAP", v["price_qual_code"] == "VALUE_TRAP")
chk("C 제목 가치 함정, 청산 없음", "가치 함정" in a["title"] and "청산" not in a["title"])

# D) MDD Ratio 계산 불가(None) → 0.0으로 보고 탈락시키지 않음
s = synth(with_vt(mdd_ratio=None, expected_return_weighted=None))
v = s["v_action_card"]
print("  D:", v["price_qual_code"], "|", v["badge_text"])
chk("D None이면 VALUE_TRAP/POOR 아님", v["price_qual_code"] not in ("VALUE_TRAP", "POOR_RISK_REWARD", "HARD_KILL"))
print("FAILS:", FAILS); sys.exit(1 if FAILS else 0)
