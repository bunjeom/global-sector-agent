"""재무 생존성 하드 킬 점검: 종목마다 v1(예전) 판정과 v2(2026-11-15) 판정을 나란히 보여 준다.

실행: .venv\\Scripts\\python check_kill.py              (기본: 신호 기록 장부 대상 64종목)
      .venv\\Scripts\\python check_kill.py AMZN DUK JPM  (원하는 종목만)
결과는 화면에 출력하고 data/backtest/kill_census_날짜.md 로도 저장한다.

이 점검은 데이터 오류·정의 실수를 찾는 용도다. 결과를 보고 기준 숫자(1·2년, 1.0·1.5배, 15·5%)를 바꾸지 않는다
(AGENTS.md '재무 생존성 v2').
"""
import os
import sys
from datetime import date

import engine
from engine import fundamental as F


def v1_verdict(v: dict) -> str:
    """예전(v1) 규칙 재현: 순현금 ÷ FCF 적자 런웨이 < 1년, 단기차입 ÷ 현금 > 150%, 희석 > 15%."""
    fcf, nc, dr, dil = v.get("fcf_ttm_krw"), v.get("net_cash_krw"), v.get("short_term_debt_ratio"), v.get("share_dilution_yoy")
    why = []
    if fcf is not None and fcf < 0 and nc is not None:
        rw = max(0.0, nc / abs(fcf)) if nc > 0 else 0.0
        if rw < 1.05 and not (nc > 0 and nc / abs(fcf) >= 2.0):
            why.append(f"런웨이 {rw:.1f}년")
    if dr is not None and dr > 1.5:
        why.append(f"단기차입/현금 {dr * 100:.0f}%")
    if dil is not None and dil > 0.15:
        why.append(f"희석 {dil * 100:.0f}%")
    return "킬 (" + ", ".join(why) + ")" if why else "-"


def inputs_for(t: str) -> dict:
    import yfinance as yf
    meta = {"notes": []}
    tk = yf.Ticker(t)
    info = {}
    try:
        info = tk.info or {}
    except Exception:
        pass
    is_krx = t.upper().endswith((".KS", ".KQ"))
    if is_krx:
        dart = F._get_dart_client()
        if dart is None:
            raise RuntimeError("DART 사용 불가(.env의 DART_API_KEY 확인)")
        blk = F._dart_annual_block(dart, t.split(".")[0], {}, meta)
        v = F._kr_liquidity_and_survival(blk, meta)
    else:
        v = F._us_liquidity_and_survival(tk, meta)
    v["is_financial_sector"] = str(info.get("sector") or "").strip().lower() == "financial services"
    v["_sector"] = info.get("sector") or "?"
    v["_is_krx"] = is_krx
    return v


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    tickers = sys.argv[1:] or engine.snapshot_universe()
    rows = []
    for t in tickers:
        try:
            v = inputs_for(t)
        except Exception as e:
            rows.append((t, "?", "받기 실패", f"{type(e).__name__}: {e}"[:80], "", "", ""))
            print(f"{t}: 받기 실패 ({type(e).__name__})")
            continue
        liquid = v.get("liquid_krw")
        if liquid is None and v.get("net_cash_krw") is not None and v["net_cash_krw"] > 0:
            liquid = v["net_cash_krw"]
        s = engine.evaluate_survival(ocf=v.get("ocf_ttm_krw"), fcf=v.get("fcf_ttm_krw"), liquid=liquid,
                                     current_debt=v.get("current_debt_krw"), dilution=v.get("share_dilution_yoy"),
                                     is_financial=v["is_financial_sector"], capex=v.get("capex_ttm_krw"),
                                     debt_change=v.get("debt_change_1y_krw"), is_krx=v["_is_krx"],
                                     captive_finance=engine.SURVIVAL_CONFIG["CAPTIVE_FINANCE"].get(t.upper()))
        new = ("킬 (" + ", ".join(s["kill_reasons"]) + ")") if s["kill"] else ("경고 (" + ", ".join(s["warn_reasons"]) + ")" if s["warning"]
                                                                              else (("판정 보류" if "판정 보류" in s["runway_text"] else "해당 없음") if not s["applicable"] else "-"))
        rows.append((t, v["_sector"], v1_verdict(v), new, s["runway_text"], s["coverage_text"], s["dilution_text"]))
        print(f"{t:10s} | 예전: {rows[-1][2]:28s} | 새: {new}")
    changed = [r for r in rows if r[2].startswith("킬") != r[3].startswith("킬")]
    lines = [f"# 재무 생존성 하드 킬 점검 ({date.today().isoformat()}, {engine.SURVIVAL_CONFIG['VERSION']})", "",
             "check_kill.py 출력. 데이터 오류·정의 실수 확인용이며, 결과를 보고 기준 숫자를 바꾸지 않는다.", "",
             f"- 대상 {len(rows)}종목 · 예전 킬 {sum(r[2].startswith('킬') for r in rows)} · 새 킬 {sum(r[3].startswith('킬') for r in rows)} "
             f"· 판정이 바뀐 종목 {len(changed)}", "",
             "| 종목 | 섹터 | 예전(v1) | 새(v2) | ① 런웨이 | ② 상환 여력 | ③ 희석 |", "|---|---|---|---|---|---|---|"]
    lines += ["| " + " | ".join(str(c).replace("|", "/") for c in r) + " |" for r in rows]
    os.makedirs(os.path.join("data", "backtest"), exist_ok=True)
    path = os.path.join("data", "backtest", f"kill_census_{date.today().isoformat()}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n예전 킬 {sum(r[2].startswith('킬') for r in rows)}종목 → 새 킬 {sum(r[3].startswith('킬') for r in rows)}종목 "
          f"(판정이 바뀐 종목 {len(changed)}: {', '.join(r[0] for r in changed) or '없음'})")
    print(f"저장: {path}")


if __name__ == "__main__":
    main()
