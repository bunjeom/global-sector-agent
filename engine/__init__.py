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
from . import synthesis, home, gate, rrg, radar

_SUBMODULES = (common, technical_signals, backtest, technical, earnings, macro, fundamental,
               synthesis, home, gate, rrg, radar)
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
