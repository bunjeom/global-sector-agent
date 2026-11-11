"""실적 트랙 실제 데이터 점검 (PC에서 직접 실행)

사용법 (프로젝트 폴더에서):
    .venv\\Scripts\\python check_earnings_data.py AAPL 005930.KS

미국 종목은 yfinance, 한국 종목(.KS/.KQ)은 DART(+yfinance 추정치)에서 값을 받아
무엇이 실제로 들어왔고 무엇이 '데이터 없음'인지 보여준다. 화면(딥다이브)을 열기 전에 확인하는 용도.
"""
import sys

import engine

NO = engine.NO_DATA


def show(title: str, d: dict) -> None:
    print(f"  [{title}]")
    for k, v in d.items():
        print(f"    {k:16s}: {NO if v is None else v}")


def main() -> None:
    tickers = sys.argv[1:] or ["AAPL", "005930.KS"]
    for t in tickers:
        print("=" * 72)
        print(f"{t}")
        raw = engine.fetch_real_earnings_inputs(t)
        print(f"  출처: {raw['meta']['source']} · 최근 분기 기준: {raw['meta']['as_of'] or NO} · 메모: {raw['meta']['notes'] or '-'}")
        show("최근 분기 실적", raw["actual_vs_cons"])
        show("이익의 질 입력", raw["quality"])
        show("EPS 추정치 FY1", raw["forward"]["fy1"])
        show("EPS 추정치 FY2", raw["forward"]["fy2"])
        show("리비전(추정치 변화)", raw["revision"])
        print(f"  최근 확정 EPS(TTM): {NO if raw['ttm_eps'] is None else raw['ttm_eps']}")
        print(f"  매출 성장률 궤적(오래된→최신): {raw['sales_yoy_trajectory'] or NO}")

        out = engine.analyze_earnings_track(t)
        print("  --- 분석 결과 ---")
        print(f"  국면: {out['regime']['earnings_regime']} · {out['regime']['regime_label']}")
        print(f"  종합: {out['verdict']['grade']} 점수 {out['verdict']['score']} (반영 {out['verdict']['axes_scored']}/5축)")
        print(f"  EPS 기준: {out['projections']['eps_basis_label']}")
        for ax in out["evaluation_axes"]:
            print(f"    {ax['axis']}: {ax['score'] if ax['score'] is not None else NO} — {ax['status']}")
        print(f"  받지 못한 데이터 {len(out['data_quality']['missing'])}건:")
        for m in out["data_quality"]["missing"]:
            print(f"    - {m}")


if __name__ == "__main__":
    main()
