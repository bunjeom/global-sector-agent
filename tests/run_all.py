"""회귀 테스트 일괄 실행기.

엔진(engine/ 폴더)이나 딥다이브 화면을 고친 뒤 한 번 실행해, 예전에 고친 문제가 다시 생기지 않았는지 확인한다.
가짜 가격·가짜 재무 데이터를 쓰므로 인터넷 연결이나 API 키 없이 돌아간다.

실행 (프로젝트 폴더에서):
    Windows : .venv\\Scripts\\python tests\\run_all.py
    macOS/Linux : .venv/bin/python tests/run_all.py

옵션:
    --fast   화면(AppTest) 검사를 빼고 엔진 검사만 빠르게 실행
    -v       각 검사의 출력을 모두 보여줌 (기본: 실패한 검사만 보여줌)
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

# (이름, 스크립트, 인자, 추가 환경변수)
ENGINE_TESTS = [
    ("실적·V 입력층 (가짜 yfinance·DART)", "test_real_data.py", [], {}),
    ("V 트랙 입력층·None 대응", "test_vdata.py", [], {}),
    ("G3 목표 배수·무위험수익률", "test_g3rf.py", [], {}),
    ("거시 트랙 실제 데이터·보류 처리", "test_macro.py", [], {}),
    ("T 실행 카드 (가짜 티켓 금지)", "test_tcard.py", [], {}),
    ("종합 라벨 (하드 킬·가치 함정·고평가)", "test_label.py", [], {}),
    ("피어 통계·내재 성장률", "test_peer.py", [], {}),
    ("이익 급변 종목 로그 PER 밴드·G3 조건", "test_regime_band.py", [], {}),
    ("판정 규칙 백테스트 (미래 데이터 미사용·앱 게이트와 일치)", "test_backtest.py", [], {}),
    ("T 트랙 검증·재설계 후보 (스캐너 재현·청산 손 계산)", "test_t_backtest.py", [], {}),
    ("컨센서스·목표가 날짜별 기록 (가짜 yfinance)", "test_snapshot.py", [], {}),
    ("틀 A 추세반등 검증 (미래 데이터 미사용·거래 손 계산)", "test_frame_backtest.py", [], {}),
    ("틀 B 저평가 매수 검증 (SEC 공시일 기준 EPS·거래 손 계산)", "test_frame_b.py", [], {}),
    ("매매 틀 v2 판정 (추세 상태·근거 저울·틀 B 단계)", "test_playbook.py", [], {}),
]

PAGE_TESTS = [
    ("화면: NVDA 기본", "test_page.py", ["full", "NVDA"], {}),
    ("화면: 삼성전자", "test_page.py", ["full", "005930.KS"], {}),
    ("화면: 실적 TTM만 있음", "test_page.py", ["ttm", "NVDA"], {}),
    ("화면: 실적 데이터 없음", "test_page.py", ["none", "NVDA"], {}),
    ("화면: 이익 급변형 PER 밴드", "test_page.py", ["full", "NVDA"], {"VMODE": "full", "BANDSERIES": "nvda"}),
    ("화면: 가치·금리·거시 일부만 받음", "test_page.py", ["full", "NVDA"], {"VMODE": "partial", "MACROMODE": "partial"}),
    ("화면: 가치·금리·거시 모두 못 받음", "test_page.py", ["full", "NVDA"], {"VMODE": "none", "RFMODE": "off", "MACROMODE": "off"}),
]


def run(name, script, args, extra_env, verbose):
    env = dict(os.environ)
    env.update(extra_env)
    env["PYTHONIOENCODING"] = "utf-8"   # Windows 콘솔(cp949)에서 이모지·한글 출력 오류 방지
    env["PYTHONUTF8"] = "1"
    t0 = time.time()
    proc = subprocess.run([sys.executable, os.path.join(HERE, script)] + args, cwd=REPO, env=env,
                          capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600)
    ok = proc.returncode == 0
    print(f"  {'통과' if ok else '실패'}  {name}  ({time.time() - t0:.0f}초)", flush=True)
    if verbose or not ok:
        out = (proc.stdout or "") + (proc.stderr or "")
        tail = out if verbose else "\n".join(out.strip().splitlines()[-25:])
        print("      " + tail.replace("\n", "\n      "))
    return ok


def main():
    verbose = "-v" in sys.argv
    tests = ENGINE_TESTS + ([] if "--fast" in sys.argv else PAGE_TESTS)
    print(f"회귀 테스트 {len(tests)}개 실행 (프로젝트: {REPO})\n", flush=True)
    results = [run(*t, verbose=verbose) for t in tests]
    failed = results.count(False)
    print(f"\n결과: {len(results) - failed}개 통과 / {failed}개 실패")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
