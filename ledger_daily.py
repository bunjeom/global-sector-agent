"""신호 기록 장부: 오늘의 틀 A v2 판정 기록 (매일 GitHub Actions가 컨센서스 기록 다음에 실행, PC에서도 실행 가능).

실행 (프로젝트 폴더에서):
    .venv\\Scripts\\python ledger_daily.py                 기본 대상 + data/snapshots/watchlist.txt
    .venv\\Scripts\\python ledger_daily.py AAPL 005930.KS  지정한 종목만

결과: data/ledger/frame_a_<연도>.csv 끝에 오늘 날짜 줄이 붙는다. 같은 날 다시 실행해도 중복되지 않는다.
추정치 상향·하향 건수는 같은 날 컨센서스 기록(snapshot_daily.py)에서 가져오므로 그 다음에 실행한다.
계산은 engine/ledger.py·engine/playbook.py에 있다.
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

    res = engine.run_daily_ledger(tickers, progress=progress)
    print(f"기록일 {res['date']} · 판정 규칙 {engine.PLAYBOOK_VERSION} · 대상 {res['n_universe']}종목 · 새로 저장 {res['saved']} · "
          f"이미 있음 {res['skipped_existing']} · 못 받음 {len(res['failed'])} · 오늘 컨센서스 기록 없음 {res['no_consensus']}")
    if res["failed"]:
        print("   못 받은 종목: " + ", ".join(res["failed"]))
    if res["no_judgment"]:
        print("   판정 불가(이유만 기록): " + "; ".join(f"{k} {v}" for k, v in res["no_judgment"].items()))
    print(f"파일: {res['path']}")
    sys.exit(1 if (res["saved"] == 0 and res["skipped_existing"] == 0) else 0)


if __name__ == "__main__":
    main()
