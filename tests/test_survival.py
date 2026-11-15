"""재무 생존성 v2 (engine/fundamental.py evaluate_survival) 손 계산 검사 + 종합 판정·틀 B 연결.

v1 정의 오류(AMZN 사례): 순현금 ÷ FCF 적자로 런웨이를 계산해, 영업현금이 크게 플러스인데도 설비투자로
FCF가 적자이고 차입이 있으면 런웨이 0년 → 하드 킬. v2는 기준 숫자는 그대로 두고 재는 대상을 바로잡았다.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _offline  # noqa: F401
import copy, sys
from test_chain_helpers import run_chain, fake_earn, engine, prices

FAILS = []


def chk(n, c, extra=""):
    print(("PASS " if c else "FAIL ") + n + (f"  [{extra}]" if extra and not c else ""))
    if not c:
        FAILS.append(n)


S = engine.evaluate_survival
B = 1e9

# 1) 아마존형: 영업현금 +168B, 설비투자 210B, FCF −42B, 현금 114B, 1년 내 만기 차입 10B, 희석 1%
a = S(ocf=168.3 * B, fcf=-42.4 * B, liquid=114.2 * B, current_debt=10 * B, dilution=0.01, capex=210.7 * B, debt_change=107.5 * B)
print("  아마존형:", a["runway_text"], "|", a["coverage_text"], "|", a["dilution_text"])
chk("1 아마존형: 하드 킬 아님", a["kill"] is False)
chk("1 아마존형: '투자 지출로 FCF 적자' 경고", a["warning"] is True and a["capex_driven_deficit"] is True and "투자 지출로 FCF 적자" in a["runway_text"])
chk("1 아마존형: 경고 문구에 영업현금·설비투자·FCF·차입 변화", all(w in a["runway_text"] for w in ("$168.3B", "−$210.7B", "−$42.4B", "+$107.5B")), a["runway_text"])
# 상환 여력 = (114.2 + 168.3) / 10 = 28.25
chk("1 아마존형: 상환 여력 28.25배", a["coverage_ratio"] == 28.25, a["coverage_ratio"])

# 2) 현금이 말라가는 적자 기업: 영업현금 −30, FCF −40, 현금 30 → 런웨이 0.75년 → 킬
b = S(ocf=-30.0, fcf=-40.0, liquid=30.0, current_debt=5.0, dilution=0.02)
print("  적자 기업:", b["runway_text"], "|", b["coverage_text"])
chk("2 적자 기업: 런웨이 0.75년", b["runway_years"] == 0.75, b["runway_years"])
chk("2 적자 기업: 하드 킬(런웨이)", b["kill"] is True and any("런웨이" in r for r in b["kill_reasons"]))
# 상환 여력 = max(0, 30 − 30) / 5 = 0 → 킬
chk("2 적자 기업: 상환 여력 0배도 킬 사유", b["coverage_ratio"] == 0.0 and any("상환 여력" in r for r in b["kill_reasons"]))
b2 = S(ocf=-10.0, fcf=-20.0, liquid=30.0, current_debt=None, dilution=None)   # 1.5년 → 경고
chk("2-2 런웨이 1.5년: 경고, 킬 아님", b2["runway_years"] == 1.5 and b2["warning"] and not b2["kill"])
b3 = S(ocf=-10.0, fcf=-20.0, liquid=50.0, current_debt=None, dilution=None)   # 2.5년 → 양호
chk("2-3 런웨이 2.5년: 경고 없음", b3["runway_years"] == 2.5 and not b3["warning"])

# 3) 전력회사형: 현금 0.3B, 영업현금 12B, 1년 내 만기 차입 8B, FCF −4B(설비투자) → (0.3+12)/8 = 1.54배 → 양호
c = S(ocf=12 * B, fcf=-4 * B, liquid=0.3 * B, current_debt=8 * B, dilution=0.02, capex=16 * B)
print("  전력회사형:", c["coverage_text"])
chk("3 전력회사형: 상환 여력 1.54배, 킬 아님", c["coverage_ratio"] == 1.54 and not c["kill"], c["coverage_ratio"])
c2 = S(ocf=6 * B, fcf=1 * B, liquid=0.3 * B, current_debt=8 * B, dilution=0.0)   # (0.3+6)/8 = 0.79 → 킬
chk("3-2 상환 여력 0.79배: 하드 킬", c2["coverage_ratio"] == 0.79 and c2["kill"])
c3 = S(ocf=10 * B, fcf=1 * B, liquid=0.3 * B, current_debt=8 * B, dilution=0.0)  # (0.3+10)/8 = 1.29 → 경고
chk("3-3 상환 여력 1.29배: 경고, 킬 아님", c3["coverage_ratio"] == 1.29 and c3["warning"] and not c3["kill"])
c4 = S(ocf=None, fcf=1.0, liquid=3.0, current_debt=8.0, dilution=0.0)          # 영업현금 모름 → 킬 안 함, 경고
chk("3-4 영업현금 모름: 현금만 0.38배 → 경고만", c4["coverage_ratio"] == 0.38 and c4["warning"] and not c4["kill"], c4["coverage_text"])

# 4) 희석 20% → 킬, 8% → 경고, 3% → 양호
chk("4 희석 20%: 하드 킬", S(ocf=10.0, fcf=5.0, liquid=10.0, current_debt=1.0, dilution=0.20)["kill"])
d8 = S(ocf=10.0, fcf=5.0, liquid=10.0, current_debt=1.0, dilution=0.08)
chk("4-2 희석 8%: 경고", d8["warning"] and not d8["kill"])
chk("4-3 희석 3%: 경고 없음", not S(ocf=10.0, fcf=5.0, liquid=10.0, current_debt=1.0, dilution=0.03)["warning"])

# 5) 은행형: 금융업이면 지표가 나빠 보여도 해당 없음
e = S(ocf=-50 * B, fcf=-60 * B, liquid=10 * B, current_debt=300 * B, dilution=0.0, is_financial=True)
chk("5 은행형: 해당 없음(킬·경고 없음)", not e["kill"] and not e["warning"] and e["applicable"] is False and "금융업" in e["runway_text"])

# 6) 자료 부족: FCF 적자인데 현금 모름 → '데이터 없음', 단정 안 함
f = S(ocf=None, fcf=-10.0, liquid=None, current_debt=None, dilution=None)
chk("6 자료 부족: 킬·경고 없음, 데이터 없음 표시", not f["kill"] and not f["warning"] and engine.NO_DATA in f["runway_text"])
chk("6-2 영업현금 모름 + 현금 있음: 런웨이로 판정(보수적)", S(ocf=None, fcf=-20.0, liquid=10.0, current_debt=None, dilution=None)["kill"])

# 7) 한국 금액 문구
chk("7 금액 문구(원)", engine.fmt_amount(1.23e12, True) == "1.2조 원" and engine.fmt_amount(-3.4e10, True) == "−340억 원")

# 8) 종합 판정·틀 B 연결: 아마존형 입력이면 KILL_SURVIVAL 없음 + 경고로 비중 축소, 적자 기업형이면 KILL_SURVIVAL
tk = "AAPL"
earn = fake_earn("forward", False, regime="G3", sector="성숙형")
last, fund_out, tech_out, _ = run_chain(tk, earn, 300.0)
macro = engine.analyze_macro_track(ticker=tk, stock_df=prices(start=1, seed=3), bench_df=prices(start=17000, seed=9),
                                   industry_category="AI 반도체·네트워킹(팹리스)")


def synth(surv):
    fo = copy.deepcopy(fund_out)
    fo["financial_survival"].update(survival_kill=surv["kill"], kill_reasons=surv["kill_reasons"])
    return engine.synthesize_4tracks(ticker=tk, current_price=last, tech_out=tech_out, earn_out=earn, fund_out=fo, macro_out=macro,
                                     synth_cfg={"stock_name": tk, "days_to_earnings": None}), fo


sa, foa = synth(a)
sb, fob = synth(b)
ka = (sa.get("hard_kill_gate") or {})
kb = (sb.get("hard_kill_gate") or {})
print("  종합(아마존형):", ka.get("is_killed"), "| 종합(적자 기업형):", kb.get("is_killed"), kb.get("kill_reason"))
chk("8 아마존형: 종합 판정 하드 킬 없음", "KILL_SURVIVAL" not in str(ka))
chk("8 적자 기업형: 종합 판정 KILL_SURVIVAL", kb.get("is_killed") and "런웨이" in str(kb.get("kill_reason")))
fb_a = engine.frame_b_evaluate(foa, sa, None)
fb_b = engine.frame_b_evaluate(fob, sb, None)
print("  틀 B:", fb_a.get("stage"), "|", fb_b.get("stage"))
chk("8 틀 B: 아마존형은 '손절 조건'이 아님", fb_a.get("stage") != "손절 조건")
chk("8 틀 B: 적자 기업형은 '손절 조건'", fb_b.get("stage") == "손절 조건")

# 9) V 트랙 전체 경로: 입력값만 주면 생존성 v2 결과가 나오는지 (fetch_real_fundamental_inputs 가짜)
FULL = dict(industry_beta=1.1, sustainable_roe=0.20, dividend_yield=0.01, is_financial_sector=False, listing_years=12.0,
            self_5y_mean=22.0, self_5y_std=4.0, self_5y_min=15.0, peer_median_per=20.0, peer_top25_per=26.0, peer_std_per=5.0,
            current_pb=8.0, pb_5y_mean=7.0, pb_5y_std=1.5, company_roe=0.25, peer_median_roe=0.15,
            company_fwd_growth=0.2, peer_median_fwd_growth=0.1, price_12m_ago=100.0, eps_lag_now=6.0, eps_lag_12m_ago=5.0,
            consensus_target_price=210.0, share_dilution_yoy=0.01,
            fcf_ttm_krw=-42.4, net_cash_krw=-60.0, ocf_ttm_krw=168.3, capex_ttm_krw=210.7, liquid_krw=114.2,
            current_debt_krw=10.0, debt_change_1y_krw=107.5)
_orig_fetch = engine.fetch_real_fundamental_inputs
engine.fetch_real_fundamental_inputs = lambda t, p: {"values": dict(FULL), "meta": {"source": "test", "notes": [], "missing": [],
                                                                                    "band_basis": "후행", "band_points": 36, "peer_count": 0}}
try:
    v = engine.analyze_fundamental_track("NVDA", 120.0, fake_earn("forward", False, regime="G2", sector="구조성장형"),
                                         {"rf_10y": 0.042, "market_erp": 0.055}, None)
finally:
    engine.fetch_real_fundamental_inputs = _orig_fetch
fs = v["financial_survival"]
print("  V 트랙:", fs["fcf_runway_years"], "|", fs["coverage_text"], "| 비중 상한", v.get("position_cap"))
chk("9 V 트랙: 하드 킬 없음 + 경고", fs["survival_kill"] is False and fs["survival_warning"] is True, fs)
chk("9 V 트랙: 버전 표시", fs.get("version") == engine.SURVIVAL_CONFIG["VERSION"])

print(f"\n결과: {'통과' if not FAILS else '실패 ' + str(FAILS)}")
sys.exit(1 if FAILS else 0)
