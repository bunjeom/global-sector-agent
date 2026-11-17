"""투자터미널 계산 엔진 (원래 engine.py 한 파일 → 기능별 모듈로 분리, 2026-11-11).

바깥 코드(app.py, 딥다이브 페이지, tests, check_*.py)는 예전처럼
    import engine
    engine.analyze_technical_track(...)
로 쓴다. 모든 함수·설정 이름을 여기서 다시 내보내므로 바깥 코드는 바꿀 필요가 없다.

모듈 지도 (자세한 설명은 각 파일 맨 위):
    common             공통 파라미터·가격 표기·실적 발표일 유틸, NO_DATA
    technical_signals  T 트랙 기초 지표·환경·생애주기 신호
    technical          T 트랙 상태 기계·셋업·사이징·V 타이밍·analyze_technical_track
    backtest           T 셋업 과거 검증
    earnings           실적 트랙 + DART + .env 읽기
    macro              거시 트랙 + FRED·ECOS + 10년 국채 금리
    fundamental        가치(V) 트랙
    synthesis          4트랙 종합 판정
    home               홈 대시보드·시장 헤더·SAMPLE_DATA
    gate               국면 게이트
    rrg                4분면(RRG)
    radar              실행 레이더
    rule_backtest      판정 규칙 백테스트(국면 게이트·T 셋업), PC 실행은 run_backtest.py
    t_backtest         T 트랙 검증(실제 스캐너 재현)과 재설계 후보(T-A 추세 추종·T-B 눌림목)
    snapshot           컨센서스·목표가 날짜별 기록(data/snapshots/), 매일 snapshot_daily.py가 실행
    sec_facts          SEC 공시일 기준 TTM EPS (틀 B 검증용 과거 시점 데이터, 캐시 data/sec_cache/)
    frame_backtest     매매 틀 검증 — 틀 A(추세반등)·틀 B(저평가 매수) (run_backtest.py --frame-a / --frame-b)
    playbook           매매 틀 v2 판정 — 틀 A 추세 상태·근거 저울·의견, 틀 B 단계 (AGENTS.md 4번)
    ledger             신호 기록 장부 — 틀 A 판정을 매일 data/ledger/에 쌓고 미리 정한 방법으로 채점 (ledger_daily.py)
    peers              피어 비교 — 피어 묶음·요인 지표·가치 관점/추세 관점 순위, data/peers/에 기록 (peers_daily.py)
    strategy           전략 선택 — 단기 규칙 판정과 진입 전략 우선순위(가치 > 추세 > 단기)
    short_backtest     단기 규칙 검증 (run_backtest.py --short-term)
    level_backtest     지지선·손절선 검증 (run_backtest.py --levels)
    reclaim_backtest   지지선 이탈 뒤 회복 검증 (run_backtest.py --reclaim)
    explore            가설 탐색 도구 — 이미 쓴 종목으로 자유롭게 보기, 증거 아님 (run_backtest.py --explore 이름)
    touch_explore      탐색: 수평선을 몇 번째 터치할 때 뚫리는가 (run_backtest.py --explore touch4)
    momentum_explore   탐색: 모멘텀(상대강도) 상위 vs 하위 20% (run_backtest.py --explore mom12 / ret63)
    peer_eval          피어 순위 채점 — 매달 상위·하위 1/3의 이후 6개월 수익, 방법 사전 고정 (run_backtest.py --peer-eval)

테스트에서 `engine.이름 = 가짜함수`로 바꿔 끼우면, 그 이름을 쓰는 모든 하위 모듈에도
같이 반영된다(아래 _EngineModule). 한 파일이던 시절과 똑같이 동작하게 하려는 장치다.
"""
import importlib as _importlib
import sys as _sys
import types as _types

# importlib.reload(engine)을 하면 한 파일이던 시절처럼 모든 하위 모듈도 처음부터 다시 읽는다
# (테스트가 바꿔 끼운 가짜 함수를 원래대로 되돌릴 때 쓴다: tests/test_macro.py 8번).
if "_SUBMODULES" in globals():
    for _m in _SUBMODULES:
        _importlib.reload(_m)

from . import common, technical_signals, backtest, technical, earnings, macro, fundamental
from . import synthesis, home, gate, rrg, radar, rule_backtest, t_backtest, snapshot, sec_facts, frame_backtest, playbook, ledger, peers
from . import strategy, short_backtest, level_backtest, reclaim_backtest, explore, touch_explore, momentum_explore, peer_eval

_SUBMODULES = (common, technical_signals, backtest, technical, earnings, macro, fundamental,
               synthesis, home, gate, rrg, radar, rule_backtest, t_backtest, snapshot, sec_facts, frame_backtest, playbook, ledger, peers,
               strategy, short_backtest, level_backtest, reclaim_backtest, explore, touch_explore, momentum_explore, peer_eval)
_SUBMODULE_NAMES = {m.__name__.rsplit(".", 1)[1] for m in _SUBMODULES}

# 하위 모듈의 모든 이름(밑줄로 시작하는 내부 함수 포함)을 engine.이름으로 내보낸다.
_seen = {}
for _m in _SUBMODULES:
    for _k, _v in vars(_m).items():
        if _k.startswith("__") or _k in _SUBMODULE_NAMES:
            continue
        if _k in _seen and _seen[_k] is not _v and not isinstance(_v, _types.ModuleType):
            raise ImportError(f"engine: 이름 '{_k}'이 여러 모듈에서 다르게 정의되어 있습니다 ({_m.__name__})")
        _seen[_k] = _v
globals().update(_seen)
del _m, _k, _v, _seen


class _EngineModule(_types.ModuleType):
    """engine.이름 = 값 을 그 이름을 가진 모든 하위 모듈에도 적용한다."""

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if name.startswith("__") or name in _SUBMODULE_NAMES:
            return
        for mod in _SUBMODULES:
            if name in mod.__dict__:
                mod.__dict__[name] = value


_sys.modules[__name__].__class__ = _EngineModule
