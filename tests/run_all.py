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
TIMEOUT_SEC = 600   # 검사 하나의 최대 시간
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
    ("신호 기록 장부 (하루 기록·중복 없음·채점 손 계산)", "test_ledger.py", [], {}),
    ("섹터 흐름 그래프 (축 밖 꼬리·축 범위·라벨 위치)", "test_rrg_chart.py", [], {}),
    ("재무 생존성 v2 (투자 주도 FCF 적자·상환 여력·금융업, 하드 킬 연결)", "test_survival.py", [], {}),
    ("피어 비교 (백분위·전략 점수·걸러내기·묶음·하루 실행)", "test_peers.py", [], {}),
    ("전략 선택 (단기 규칙 진입·청산 손 계산·무작위 비교·우선순위)", "test_strategy.py", [], {}),
    ("지지선·손절선 (화면 판정·위험 계산·사건 손 계산·가짜 선 비교)", "test_levels.py", [], {}),
    ("지지선 이탈 뒤 회복 (이탈·회복 손 계산·세 갈래 집계·무작위 비교)", "test_reclaim.py", [], {}),
    ("가설 탐색 도구 (사건 뒤 수익률·후보 기준·기록·가격 캐시)", "test_explore.py", [], {}),
    ("탐색: 수평선 터치 횟수별 뚫림 (터치 세기 손 계산·방식 점검)", "test_touch.py", [], {}),
    ("탐색: 모멘텀 상위·하위 20% (신호·결과 손 계산·방식 점검)", "test_momentum.py", [], {}),
    ("피어 순위 채점 (줄 세운 날·수익률 손 계산·판정 보류·방식 점검)", "test_peer_eval.py", [], {}),
    ("강의 조건 시험 1단계 (상태·등급·RS·BNF 손 계산·미래 자료 미사용·방식 점검)", "test_lecture.py", [], {}),
    ("강의 조건 시험 2단계 L2 (사건 손 계산·미래 자료 미사용·섞기 통과선)", "test_lecture_l2.py", [], {}),
    ("강의 조건 시험 2단계 L4·L5 (약화·붕괴·선 넘음 손 계산·미래 자료 미사용·섞기 통과선)", "test_lecture_l45.py", [], {}),
]

PAGE_TESTS = [
    ("화면: NVDA 기본", "test_page.py", ["full", "NVDA"], {}),
    ("화면: 삼성전자", "test_page.py", ["full", "005930.KS"], {}),
    ("화면: 실적 TTM만 있음", "test_page.py", ["ttm", "NVDA"], {}),
    ("화면: 실적 데이터 없음", "test_page.py", ["none", "NVDA"], {}),
    ("화면: 이익 급변형 PER 밴드", "test_page.py", ["full", "NVDA"], {"VMODE": "full", "BANDSERIES": "nvda"}),
    ("화면: 가치·금리·거시 일부만 받음", "test_page.py", ["full", "NVDA"], {"VMODE": "partial", "MACROMODE": "partial"}),
    ("화면: 가치·금리·거시 모두 못 받음", "test_page.py", ["full", "NVDA"], {"VMODE": "none", "RFMODE": "off", "MACROMODE": "off"}),
    ("화면: 피어 기록 없음 안내", "test_page.py", ["full", "NVDA"], {"PEERMODE": "none"}),
    ("화면: 피어 비교 페이지(가짜 기록 2묶음)", "test_peer_page.py", ["fake"], {}),
    ("화면: 피어 비교 페이지(기록 없음)", "test_peer_page.py", ["none"], {}),
    ("홈: 장부 기반 실행 레이더 (칸 나누기·크기 안내)", "test_home.py", ["ledger"], {}),
    ("홈: 장부 없음 안내", "test_home.py", ["empty"], {}),
    ("홈: 오래된 장부 경고", "test_home.py", ["stale"], {}),
    ("홈: 기존 셋업 레이더 스위치", "test_home.py", ["legacy"], {}),
    ("홈: 국면 크기 0 알림 한 번만", "test_home.py", ["frozen"], {}),
]


def run(name, script, args, extra_env, verbose):
    env = dict(os.environ)
    env.update(extra_env)
    env["PYTHONIOENCODING"] = "utf-8"   # Windows 콘솔(cp949)에서 이모지·한글 출력 오류 방지
    env["PYTHONUTF8"] = "1"
    t0 = time.time()
    try:
        proc = subprocess.run([sys.executable, os.path.join(HERE, script)] + args, cwd=REPO, env=env,
                              capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=TIMEOUT_SEC)
    except subprocess.TimeoutExpired as e:   # 한 검사가 멈춰도 실행기 전체가 죽지 않고 '시간 초과'로 기록하고 다음으로 넘어간다
        print(f"  실패  {name}  (시간 초과 {TIMEOUT_SEC}초 — 멈춘 곳은 아래 마지막 출력)", flush=True)
        part = e.stdout.decode("utf-8", "replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
        tail = "\n".join(part.strip().splitlines()[-15:]) or "(출력 없음)"
        print("      " + tail.replace("\n", "\n      "))
        print(f"      → 따로 실행해 확인: .venv\\Scripts\\python tests\\{script} {' '.join(args)}")
        return False
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
