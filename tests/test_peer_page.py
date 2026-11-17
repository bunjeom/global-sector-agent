"""피어 비교 페이지(pages/2_🧭_피어_비교.py)를 AppTest로 실제 실행 + engine 요약 함수(peer_groups_overview·peer_group_view) 검사.

모드 (첫 번째 인자): fake = 가짜 기록 2묶음(관심 종목 있는 묶음·없는 묶음) · none = 기록 없음 → 안내만
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _offline  # noqa: F401
import os, re, sys, tempfile
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)
MODE = sys.argv[1] if len(sys.argv) > 1 else "fake"
import engine  # noqa: E402
from _fake_peers import write_fake_peers  # noqa: E402

FAILS = []


def chk(n, c, extra=None):
    print(("PASS " if c else "FAIL ") + n + (f"  [{extra}]" if (not c and extra is not None) else ""))
    if not c:
        FAILS.append(n)


tmp = tempfile.mkdtemp(prefix="peer_page_")
if MODE == "fake":
    write_fake_peers(tmp, "NVDA")                                                      # 관심 종목 NVDA 묶음
    write_fake_peers(tmp, "QA", group="other", group_name="다른 묶음", others=("QB", "QC", "QD", "QE", "QF"))
    pr = engine.load_peer_ranks(tmp)
    pr.loc[pr["group"] == "other", "is_coverage"] = False                               # 관심 종목 없는 묶음
    for f in os.listdir(tmp):
        if f.startswith("peer_rank_"):
            os.remove(os.path.join(tmp, f))
    out = pr.copy()
    out["rank_date"] = out["rank_date"].dt.strftime("%Y-%m-%d")
    out["fund_date"] = pd.to_datetime(out["fund_date"]).dt.strftime("%Y-%m-%d")
    out.to_csv(engine.rank_path("2026-11-16", tmp), index=False, encoding="utf-8-sig")
engine.PEER_CONFIG["DIR"] = tmp

# 1) engine 요약 함수
ov = engine.peer_groups_overview(tmp)
if MODE == "fake":
    g = ov["groups"].set_index("group")
    print("  묶음 요약:", g[["group_name", "n", "n_coverage", "value_leader", "trend_leader"]].to_dict("index"))
    chk("1 묶음 2개, 내 종목(NVDA = 대표 종목) 있는 묶음이 먼저, 종목 수 6", list(ov["groups"]["group"]) == ["test_group", "other"]
        and int(g.loc["test_group", "n"]) == 6 and g.loc["test_group", "mine"] == ["NVDA"] and int(g.loc["other", "n_mine"]) == 0)
    chk("1-1 내 종목 목록을 직접 주면 그것으로", engine.peer_groups_overview(tmp, mine=["QC"])["groups"]["group"].iloc[0] == "other")
    chk("1-2 가치 관점 1위 PC(가짜 기록 손 계산과 같음), 구성 종목 목록", g.loc["test_group", "value_leader"] == "PC"
        and "NVDA" in g.loc["test_group", "members"] and g.loc["test_group", "value_ranked"])
    gv = engine.peer_group_view("test_group", tmp)
    chk("1-3 묶음 보기: 6종목, 회사 이름·생존성 문구 붙음, 가치 점수 높은 순", gv["available"] and len(gv["table"]) == 6
        and gv["table"]["short_name"].notna().all() and gv["table"]["value_score"].iloc[0] >= gv["table"]["value_score"].iloc[1])
    chk("1-4 없는 묶음은 안내", not engine.peer_group_view("zzz", tmp)["available"])
    lv = engine.latest_peer_view("NVDA", tmp)
    chk("1-5 딥다이브 보기(latest_peer_view)는 예전과 같음 — 6종목·이 종목 정보", lv["available"] and len(lv["table"]) == 6 and lv["me"]["ticker"] == "NVDA")
else:
    chk("1 기록 없음 → available False + 이유", not ov["available"] and "피어 기록 없음" in ov["reason"])

# 2) 화면
import streamlit as _st  # noqa: E402
_st.page_link = lambda *a, **k: None   # 하위 페이지를 단독으로 실행하면 페이지 링크 검증이 불가능해 무력화(test_page.py와 같음)
from streamlit.testing.v1 import AppTest  # noqa: E402
at = AppTest.from_file(os.path.join(REPO, "pages", "2_🧭_피어_비교.py"), default_timeout=120)
at.run()
print("exceptions:", [str(e.value)[:300] for e in at.exception])
md = " ".join(m.value for m in at.markdown)
cap = " ".join(c.value for c in at.caption)
chk("2 예외 없음", not at.exception)
bad = [m for m in re.findall(r"[^\s<>]{0,12}(?:None|nan|NaN)[^\s<>]{0,8}", md + " " + cap)]
chk("2-2 None/nan 글자 노출 없음", not bad, bad[:5])
if MODE == "fake":
    chk("2-3 요약 카드: 묶음 2개·검증 상태·ret63 탐색 주의", "묶음 2개" in md and "검증 상태: 미검증" in md and "63일 상대강도" in md)
    chk("2-4 기본 = 내 종목 있는 묶음만 → 테스트 묶음 카드·순위표", "테스트 묶음" in md and "피어 6종목" in md and len(at.dataframe) >= 1)
    tbl = [d.value for d in at.dataframe if "종목" in getattr(d.value, "columns", [])]
    t0 = tbl[-1] if tbl else pd.DataFrame()
    chk("2-5 순위표: 6종목, 가치 관점 점수 순, 1위 PC, 하드 킬 종목 표시, 관심 ★", len(t0) == 6 and t0["종목"].iloc[0] == "PC"
        and (t0["생존성"] == "하드 킬(순위 제외)").any() and t0.loc[t0["내 종목"] == "★", "종목"].tolist() == ["NVDA"], t0.head(3).to_dict() if len(t0) else None)
    at.radio[0].set_value("추세 관점").run()
    t1 = [d.value for d in at.dataframe if "종목" in getattr(d.value, "columns", [])][-1]
    chk("2-6 추세 관점으로 바꾸면 추세 점수 순으로 다시 정렬", "추세 관점 점수" in t1.columns and not at.exception
        and t1["추세 관점 점수"].replace("-", None).dropna().astype(float).is_monotonic_decreasing)
    at.checkbox[0].set_value(False).run()
    chk("2-7 '내 종목 있는 묶음만' 끄면 묶음 2개 모두 고를 수 있음", len(at.selectbox[1].options) == 2, at.selectbox[1].options)
    at.text_input[0].set_value("QC").run()
    chk("2-8 종목으로 찾기(QC) → 그 종목 묶음(다른 묶음)이 선택됨", "다른 묶음" in " ".join(m.value for m in at.markdown) and not at.exception)
    at.text_input[0].set_value("ZZZZ").run()
    chk("2-9 기록에 없는 종목 → 안내", any("피어 기록에 없는 종목" in w.value for w in at.warning))
    chk("2-10 딥다이브 버튼·종목 선택", any(b.label == "딥다이브 →" for b in at.button) and any(s.label == "딥다이브로 볼 종목" for s in at.selectbox))
else:
    chk("2-3 기록 없음 → 안내만", any("피어 기록 없음" in i.value for i in at.info))

print(f"\n결과: {'통과' if not FAILS else '실패 ' + str(FAILS)}")
sys.exit(1 if FAILS else 0)
