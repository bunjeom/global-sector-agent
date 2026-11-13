"""컨센서스·목표가 날짜별 기록 실행 (매일 GitHub Actions가 실행, PC에서도 실행 가능).

실행 (프로젝트 폴더에서):
    .venv\\Scripts\\python snapshot_daily.py              기본 대상 + data/snapshots/watchlist.txt
    .venv\\Scripts\\python snapshot_daily.py AAPL 005930.KS  지정한 종목만

결과: data/snapshots/consensus_<연도>.csv 끝에 오늘 날짜 줄이 붙는다. 같은 날 다시 실행해도 중복되지 않는다.
계산은 engine/snapshot.py에 있다.
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
    tickers = [a for a in sys.argv[1:] if not a.startswith("-")] or None

    def progress(i, n, t):
        if i % 10 == 0 or i == n:
            print(f"   {i}/{n} ({t}) · {time.time() - t0:.0f}초", flush=True)

    res = engine.run_daily_snapshot(tickers, progress=progress)
    print(f"기록일 {res['date']} · 대상 {res['n_universe']}종목 · 새로 저장 {res['saved']} · 이미 있음 {res['skipped_existing']} · 못 받음 {len(res['failed'])}")
    if res["failed"]:
        print("   못 받은 종목: " + ", ".join(res["failed"]))
    if res["partial"]:
        print(f"   일부 항목만 받은 종목 {len(res['partial'])}개 (못 받은 항목은 빈칸으로 저장)")
    print(f"파일: {res['path']}")
    # 하나도 저장하지 못했고 이미 있는 것도 없으면 실패로 알린다 (워크플로는 continue-on-error라 리포트 갱신은 계속)
    sys.exit(1 if (res["saved"] == 0 and res["skipped_existing"] == 0) else 0)


if __name__ == "__main__":
    main()
