"""홈 화면(app.py)의 실행 레이더를 AppTest로 실제 실행해, 신호 기록 장부 기반 레이더가 예외 없이 그려지는지 본다.

모드 (첫 번째 인자):
    ledger : 가짜 장부 4줄(진입 의견·관찰·보유 점검·판정 불가) → 세 칸에 맞게 나뉘는지
    empty  : 장부 없음 → 안내 문구만, 예외 없음
    stale  : 마지막 기록이 오래됨 → 경고 표시
    legacy : 기존 셋업 레이더 스위치를 켜도 예외 없음
    frozen : 국면이 신규 진입 크기를 0으로 만들 때 → 칸 위에 한 번만 알림, 카드마다 반복하지 않음
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _offline  # noqa: F401  외부 네트워크 차단 (tests/_offline.py)
import os, re, sys, tempfile
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)
MODE = sys.argv[1] if len(sys.argv) > 1 else "ledger"

import engine  # noqa: E402
import engine.ledger as _lg  # noqa: E402

tmp = tempfile.mkdtemp(prefix="ledger_test_")
if MODE == "frozen":   # 모든 국면에서 크기 0 (G3와 같은 상황)
    engine.PLAYBOOK_CONFIG["GATE_SIZE"] = {k: 0.0 for k in engine.PLAYBOOK_CONFIG["GATE_SIZE"]}
_lg.LEDGER_CONFIG["DIR"] = tmp   # 실제 data/ledger/는 건드리지 않는다

today = pd.Timestamp.now(tz="Asia/Seoul").tz_localize(None).normalize()
led_day = today - pd.Timedelta(days=10) if MODE == "stale" else today


def row(t, state, lean, sup, ag, entry, size, detail):
    r = {c: None for c in _lg.LEDGER_COLUMNS}
    r.update(ledger_date=led_day.date().isoformat(), ticker=t, market="US", version=engine.PLAYBOOK_VERSION,
             as_of=(led_day - pd.Timedelta(days=2)).date().isoformat(), close=100.0, atr=2.0, state=state,
             state_detail=detail, lean_label=lean, support=sup, against=ag,
             diff=(None if sup is None else sup - ag), entry_opinion=entry, size_mult=size, chase_caution=False)
    return r


if MODE != "empty":
    rows = [
        row("ENTRYCO", "하락", "전환 가능성 우세", 4, 1, True, 1.0, "하락 (고점이 낮아지는 중)"),
        row("WATCHCO", "횡보", "전환 가능성 우세", 3, 1, False, 0.0, "횡보 (상승 추세선 이탈)"),
        row("TIEDCO", "하락", "팽팽", 2, 2, False, 0.0, "하락 (고점이 낮아지는 중)"),
        row("HOLDCO", "상승", "하락 전환 가능성 우세", 1, 4, False, 0.0, "상승 (저점이 높아지는 중)"),
        row("NODATA", None, None, None, None, False, None, None),
    ]
    pd.DataFrame(rows, columns=_lg.LEDGER_COLUMNS).to_csv(os.path.join(tmp, f"frame_a_{led_day.year}.csv"),
                                                          index=False, encoding="utf-8-sig")

from streamlit.testing.v1 import AppTest  # noqa: E402

at = AppTest.from_file(os.path.join(REPO, "app.py"), default_timeout=300)
at.run()
if MODE == "legacy":
    at.toggle(key="show_legacy_radar").set_value(True).run()

fails = []
exc = [str(e.value)[:300] for e in at.exception]
print("exceptions:", exc)
if exc:
    fails.append("예외 발생")

titles = [m.value for m in at.markdown if m.value.startswith("##### ")]
caps = [c.value for c in at.caption]
md_all = " ".join(m.value for m in at.markdown)
text = md_all + " " + " ".join(caps)
leaks = [m for m in re.findall(r"[^\s<>]{0,12}(?:\bNone\b|\bnan\b|\bNaN\b)[^\s<>]{0,8}", text) if "yfinance" not in m]
print("leaked None/nan tokens:", leaks[:8])
if leaks:
    fails.append("None/nan 노출")

if MODE == "empty":
    ok = any("신호 기록 장부가 아직 없습니다" in i.value for i in at.info)
    print("장부 없음 안내:", ok)
    if not ok:
        fails.append("장부 없음 안내 누락")
else:
    need = ["##### 상승 전환 의견", "##### 관찰", "##### 보유 점검"]
    print("레이더 칸:", [t[:20] for t in titles])
    if not all(any(t.startswith(n + " ") for t in titles) for n in need):
        fails.append("레이더 칸 누락")
    # 각 종목이 맞는 칸에 들어갔는지: 칸 제목 다음에 나오는 첫 종목 이름으로 확인
    seq = [m.value for m in at.markdown]
    def first_after(title):
        i = next(k for k, v in enumerate(seq) if v.startswith(title + " "))
        for v in seq[i + 1:]:
            mm = re.match(r"\*\*([A-Z]+)\*\*", v)
            if mm:
                return mm.group(1)
            if v.startswith("##### "):
                return None
        return None
    got = {n: first_after(n) for n in need}
    print("칸별 첫 종목:", got)
    if got != dict(zip(need, ["ENTRYCO", "WATCHCO", "HOLDCO"])):
        fails.append("종목이 엉뚱한 칸에 들어감")
    if "TIEDCO" not in md_all:
        fails.append("팽팽 종목이 관찰 칸에 없음")
    if "NODATA" in md_all:
        fails.append("판정 불가 종목이 카드로 나옴")
    size_cap = [c for c in caps if "배 × 국면" in c] + [m for m in seq if "신규 진입 크기 0" in m]
    print("크기 안내:", size_cap)
    if not size_cap:
        fails.append("진입 크기 안내 누락")
    if MODE == "frozen":
        n_frz = sum("신규 진입 크기 0" in m for m in seq)
        print("크기 0 알림 개수:", n_frz)
        if n_frz != 1 or any("배 × 국면" in c for c in caps):
            fails.append("크기 0 알림이 한 번이 아님")
    stale = any("마지막 기록이" in w.value for w in at.warning)
    print("오래된 기록 경고:", stale)
    if (MODE == "stale") != stale:
        fails.append("오래된 기록 경고가 맞지 않음")
    if not any(c.startswith("기준일 ") and "미검증" in c for c in caps):
        fails.append("기준일·검증 상태 문구 누락")
    if any("매매 틀 v2 · 틀 A(추세 판단" in c for c in caps):
        fails.append("레이더 아래 긴 설명이 그대로 보임(도움말로 옮겨야 함)")
# 게이트 상자: '레짐' 대신 '지수 추세', 출처 문구 없이 기준일만
gate_md = [m.value for m in at.markdown if "게이트</div>" in m.value]
print("게이트 상자 수:", len(gate_md))
if any("지수 레짐" in g or "yfinance (" in g for g in gate_md):
    fails.append("게이트 상자에 '지수 레짐' 또는 출처 문구가 남음")
if not any("지수 추세(레짐)란?" in e.label for e in at.expander):
    fails.append("레짐 설명 접기 칸 없음")

# 피어 비교 바로가기(5단계): 제목은 늘 있고, 기록이 있으면 표·링크, 없으면 이유 한 줄
peer_head = any("#### 피어 비교" in m.value for m in at.markdown)
peer_cap = [c.value for c in at.caption if c.value.startswith(("같은 사업을 하는 회사끼리", "피어 비교:"))]
print("피어 비교 칸:", peer_head, peer_cap[:1])
if not (peer_head and peer_cap):
    fails.append("홈 피어 비교 칸 없음")

print("결과:", "통과" if not fails else f"실패 — {fails}")
sys.exit(1 if fails else 0)
