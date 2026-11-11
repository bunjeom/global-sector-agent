"""V 트랙 피어(동종 기업) 수집 점검: 어떤 피어 그룹·종목을 받았고, 각 종목의 선행 PER·ROE가 왜 채택/제외됐는지 출력한다.
실행: .venv\\Scripts\\python check_peer_data.py AAPL
"""
import sys
import yfinance as yf
import engine

t = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
info = yf.Ticker(t).info or {}
P = engine.FUND_DATA_PARAMS
lo, hi = P["peer_pe_range"]
print(f"{t}: industry={info.get('industryKey')} · sector={info.get('sectorKey')} · 자기 ROE={info.get('returnOnEquity')} · 선행 PER={info.get('forwardPE')}")
print(f"규칙: 후보 최대 {P['peer_candidates']}개 · 유효 PER 범위 ({lo}, {hi}] · 유효 {P['peer_min_valid']}개 이상이어야 피어 통계 사용\n")
for kind, key_name in (("Industry", "industryKey"), ("Sector", "sectorKey")):
    key = info.get(key_name)
    if not key:
        print(f"[{kind}] 키 없음"); continue
    try:
        df = getattr(yf, kind)(key).top_companies
    except Exception as e:
        print(f"[{kind}:{key}] 상위 기업 조회 실패: {type(e).__name__} {e}"); continue
    syms = [s for s in (list(df.index) if df is not None else []) if str(s).upper() != t.upper()][:P["peer_candidates"]]
    print(f"[{kind}:{key}] 상위 기업 {0 if df is None else len(df)}개 → 후보 {len(syms)}개")
    ok = 0
    for s in syms:
        try:
            pi = yf.Ticker(s).info or {}
        except Exception as e:
            print(f"   {s:8s} 정보 실패 {type(e).__name__}"); continue
        pe, roe = pi.get("forwardPE"), pi.get("returnOnEquity")
        valid = isinstance(pe, (int, float)) and lo < pe <= hi
        ok += valid
        why = "채택" if valid else ("선행 PER 없음" if pe is None else f"범위 밖({pe:.1f})")
        print(f"   {s:8s} 선행 PER={pe!s:>8.8} · ROE={roe!s:>8.8} · {why}")
    print(f"   → 유효 PER {ok}개 ({'피어 통계 사용' if ok >= P['peer_min_valid'] else '5개 미만이라 제외'})\n")
    if kind == "Industry":
        print("   ※ 현재 엔진은 산업 상위 기업 목록을 받으면 섹터로 넘어가지 않는다 (아래 섹터는 참고용)\n")
