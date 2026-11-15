"""피어 비교 기록: 피어 묶음별 가치 관점·추세 관점 순위를 매일 data/peers/에 쌓는다 (GitHub Actions가 매일 실행, PC에서도 실행 가능).

실행 (프로젝트 폴더에서):
    .venv\\Scripts\\python peers_daily.py                  매일 실행(재무는 7일마다, 가격 흐름은 매일)
    .venv\\Scripts\\python peers_daily.py --full           재무·피어 묶음을 지금 다시 받기(처음 한 번, 수십 분 걸림)
    .venv\\Scripts\\python peers_daily.py AMZN NVDA        이 종목들이 속한 묶음만(빠른 점검)

결과: data/peers/peer_fund_<연도>.csv(재무, 주 1회), data/peers/peer_rank_<연도>.csv(순위, 매일). 같은 날 다시 실행해도 중복되지 않는다.
피어 묶음 수정: data/peers/peer_map.csv. 커버리지 추가: data/snapshots/watchlist.txt 한 줄.
계산 규칙은 AGENTS.md 4번 '투자 방향과 딥다이브 개편 기획', 코드는 engine/peers.py.
"""
import sys
import time


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")   # Windows 콘솔에서 한글 깨짐 방지
    except Exception:
        pass
    import engine
    t0 = time.time()
    args = sys.argv[1:]
    full = "--full" in args
    tickers = [a for a in args if not a.startswith("-")] or None

    def progress(i, n, t):
        if i % 20 == 0 or i == n:
            print(f"   재무 {i}/{n} ({t}) · {time.time() - t0:.0f}초", flush=True)

    res = engine.run_daily_peers(full=full, tickers=tickers, progress=progress)
    print(f"기록일 {res['date']} · {engine.PEER_CONFIG['VERSION']}")
    if res.get("refreshed"):
        print(f"   재무 다시 받음: 커버리지 {res.get('n_coverage')}종목 → 피어 묶음 {res.get('n_groups')}개 · 묶음 구성 {res.get('n_members')}종목 · "
              f"새로 저장 {res['fund_saved']} · 못 받음 {len(res['failed'])}")
        if res["failed"]:
            print("   못 받은 종목: " + ", ".join(res["failed"][:40]) + (" …" if len(res["failed"]) > 40 else ""))
        if res.get("no_group"):
            print(f"   묶음 없는 커버리지 종목 {len(res['no_group'])}개: " +
                  "; ".join(f"{k} {v}" for k, v in list(res["no_group"].items())[:15]))
    if res.get("error"):
        print("   " + res["error"])
        sys.exit(1)
    print(f"   순위 새로 저장 {res['rank_saved']} · 이미 있음 {res.get('rank_skipped', 0)} · 재무 기준일 {res.get('fund_date')} · "
          f"가격 못 받음 {res.get('n_price_missing', 0)} · {time.time() - t0:.0f}초")
    sys.exit(1 if (res["rank_saved"] == 0 and res.get("rank_skipped", 0) == 0) else 0)


if __name__ == "__main__":
    main()
