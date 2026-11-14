"""매매 틀 v2 판정 점검: 종목별 틀 A(추세 상태·근거 저울·의견)를 실제 데이터로 계산해 출력한다.
실행: .venv\\Scripts\\python check_playbook.py                  (기본: NVDA AAPL JPM 005930.KS)
      .venv\\Scripts\\python check_playbook.py MSFT KO --gate 1   (국면 게이트 단계를 주면 크기 축소까지 표시)
틀 B(저평가 매수)는 가치·실적·종합 판정 결과가 필요해 딥다이브 화면 개편 때 연결한다(AGENTS.md 4번 다음 작업 순서).
"""
import sys
import engine


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")   # Windows 콘솔에서 한글 깨짐 방지
    except Exception:
        pass
    args = sys.argv[1:]
    gate = None
    if "--gate" in args:
        i = args.index("--gate")
        gate = int(args[i + 1])
        args = args[:i] + args[i + 2:]
    tickers = args or ["NVDA", "AAPL", "JPM", "005930.KS"]
    for t in tickers:
        try:
            r = engine.fetch_frame_a_live(t, gate_level=gate)
        except Exception as e:   # 네트워크·야후 오류
            print(f"{t}: 받기 실패 ({type(e).__name__}: {e})\n")
            continue
        print(engine.render_frame_a_text(r) + "\n")


if __name__ == "__main__":
    main()
