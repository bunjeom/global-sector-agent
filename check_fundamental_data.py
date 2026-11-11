"""V(펀더멘탈) 트랙 실제 데이터 점검 (PC에서 직접 실행)

사용법 (프로젝트 폴더에서):
    .venv\\Scripts\\python check_fundamental_data.py AAPL 005930.KS

미국 종목은 yfinance, 한국 종목(.KS/.KQ)은 DART(+yfinance 가격·베타·배당)에서 값을 받아
무엇이 실제로 들어왔고 무엇이 '데이터 없음'인지 보여준다. 현재가는 최근 종가를 쓴다.
"""
import sys

import engine

NO = engine.NO_DATA


def last_close(t: str):
    try:
        import yfinance as yf
        c = yf.Ticker(t).history(period="10d")["Close"].dropna()
        return float(c.iloc[-1]) if len(c) else None
    except Exception:
        return None


def main() -> None:
    tickers = sys.argv[1:] or ["AAPL", "005930.KS"]
    for t in tickers:
        print("=" * 72)
        px = last_close(t)
        print(f"{t}  (최근 종가: {NO if px is None else px})")
        if px is None:
            print("  가격을 받지 못해 점검을 건너뜁니다.")
            continue
        out = engine.fetch_real_fundamental_inputs(t, px)
        v, m = out["values"], out["meta"]
        print(f"  출처: {m['source']} · 자기 밴드: {m['band_basis']} 관측 {m['band_points']}개월(약 {m['band_span_years'] or NO}년)"
              f" · 피어 {m['peer_count']}개({m['peer_group'] or NO})")
        for n in m["notes"]:
            print(f"  메모: {n}")
        for k, x in v.items():
            print(f"    {k:28s}: {NO if x is None else (round(x, 4) if isinstance(x, float) else x)}")
        print(f"  못 받은 항목 {len(m['missing'])}개: {', '.join(m['missing']) or '없음'}")


if __name__ == "__main__":
    main()
