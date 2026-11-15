"""홈 섹터 흐름 그래프(build_rrg_figure, app.py)의 그리기 규칙 검사.

- 축 밖에 있는 꼬리 점을 모서리(±3σ)로 끌어오지 않는다(가짜 꺾임 방지)
- 오늘 위치가 ±3σ 밖이면 화면 범위만 넓힌다(최대 AXIS_LIMIT_MAX)
- 지시선 없는 라벨은 xshift·yshift로 점 옆에 둔다(plotly는 showarrow=False일 때 ax·ay를 무시)
- 화살표 시작점은 화면 안
app.py 전체를 실행하지 않고 해당 함수만 꺼내 가짜 좌표로 그린다.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _offline  # noqa: F401
import ast, os, sys
import numpy as np
import plotly.graph_objects as go
import engine

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
tree = ast.parse(open(os.path.join(REPO, "app.py"), encoding="utf-8").read())
g = {"engine": engine, "np": np, "go": go}
for node in tree.body:
    if isinstance(node, ast.Assign) and any(getattr(t, "id", "") in ("QUAD_COLOR", "QUAD_PLAIN") for t in node.targets):
        exec(compile(ast.Module([node], []), "app", "exec"), g)
    if isinstance(node, ast.FunctionDef) and node.name in ("build_rrg_figure", "grade_for"):
        exec(compile(ast.Module([node], []), "app", "exec"), g)
build = g["build_rrg_figure"]


def mk(x, y, tx=(), ty=()):
    return {"x": x, "y": y, "quadrant": engine.classify_rrg_quadrant(x, y), "tail_x": list(tx), "tail_y": list(ty), "obs_count": 300}


fails = []
# 1) 꼬리 시작이 -4.6σ, 오늘 위치 Y 3.3σ → 축 ±4, 꼬리 좌표는 원래 값 그대로
cm = {"XLC": mk(0.05, 3.3, [-4.6, -4.0, -1.5, -0.4, 0.05], [-4.2, -3.5, 0.7, 2.8, 3.3]),
      "XLI": mk(0.35, 0.65), "ITA": mk(0.5, 0.7), "XLK": mk(-1.6, -1.1)}
fig, hid, lim = build(cm, {}, {k: {"name": k} for k in cm}, "일봉", {"XLC"}, False)
print("축 범위:", lim, "| 그래프 x 범위:", fig.layout.xaxis.range)
if lim != 4.0 or list(fig.layout.xaxis.range) != [-4.0, 4.0]:
    fails.append("축 범위가 넓어지지 않음")
tail_x = [v for tr in fig.data if tr.mode == "lines" for v in tr.x]
print("꼬리 x 최소:", min(tail_x))
if min(tail_x) != -4.6:
    fails.append("꼬리 점을 모서리로 끌어옴")
arrows = [a for a in fig.layout.annotations if a.showarrow and a.axref == "x"]
print("화살표 시작:", [(round(a.ax, 2), round(a.ay, 2)) for a in arrows])
if not arrows or any(abs(a.ax) > lim or abs(a.ay) > lim for a in arrows):
    fails.append("화살표 시작점이 화면 밖")
labels = [a for a in fig.layout.annotations if a.text and a.text.startswith("<b>") and a.text not in ("<b>주도</b>", "<b>개선</b>", "<b>소외</b>", "<b>약화</b>")]
no_leader = [a for a in labels if not a.showarrow]
print("라벨:", [(a.text, a.xshift, a.yshift, a.showarrow) for a in labels])
if not no_leader or any((a.xshift or 0) == 0 and (a.yshift or 0) == 0 for a in no_leader):
    fails.append("라벨이 점 위에 겹침(shift 없음)")
# 2) 모두 ±3σ 안 → 기본 범위 그대로, 라벨에 화면 밖 표시 없음
cm2 = {"XLY": mk(2.0, 1.0, [1.0, 1.5, 2.0], [0.2, 0.5, 1.0]), "XLE": mk(-1.0, -0.5)}
fig2, _, lim2 = build(cm2, {}, {k: {"name": k} for k in cm2}, "일봉", {"XLY"}, False)
print("기본 범위:", lim2)
if lim2 != 3.0:
    fails.append("범위 안인데 축이 바뀜")
# 3) 아주 먼 점(Y 9σ) → 최대 범위에서 멈추고 방향 표시 ▲
cm3 = {"XLC": mk(0.5, 9.0)}
fig3, _, lim3 = build(cm3, {}, {"XLC": {"name": "XLC"}}, "일봉", set(), False)
txt3 = [a.text for a in fig3.layout.annotations if a.text and "XLC" in a.text]
print("최대 범위:", lim3, txt3)
if lim3 != engine.RRG_CONFIG["AXIS_LIMIT_MAX"] or not any("▲" in t for t in txt3):
    fails.append("최대 범위·방향 표시 오류")
hov = [t for tr in fig.data if tr.mode == "markers" and tr.text is not None for t in tr.text]
print("꼬리 점 설명:", hov[:2])
if not hov or not all("주 전" in t and "σ" in t for t in hov):
    fails.append("꼬리 점 마우스 설명 없음")
# 4) 꼬리 말로 풀기 (손 계산)
D = engine.describe_rrg_tail
d1 = D([-0.7, 0.6, 1.4, 2.3, 3.6], [2.5, 2.7, 2.2, 2.6, 2.9])     # 계속 주도, 마지막 주 오른쪽 위
d2 = D([2.0, 3.6, 2.8], [2.0, 2.9, 1.25])                         # 주도에서 오른쪽 아래로 크게 → 약화 쪽
d3 = D([-0.2, -3.1, -0.75, 0.1], [0.2, -3.1, 0.75, 3.2])          # 중립 → 소외 → 개선 → 주도
d4 = D([1.0, 1.1], [1.0, 0.9])                                    # 거의 안 움직임
print("말로 풀기:", d1["path_text"], "|", d2["heading"], "|", d3["path_text"], "|", d4["heading"])
if d1["path_text"] != "개선 → 주도" or d1["last_week_text"] != "지수 대비 강도 높아짐 · 강해지는 속도 빨라짐" or d1["heading"] != "주도 칸에 머묾" or d1["heading_change"]:
    fails.append("꼬리 설명 1")
# d2: dx=-0.8, dy=-1.65 → 다음 점 (2.0, -0.4) = 약화
if d2["x_word"] != "낮아짐" or d2["y_word"] != "느려짐" or d2["heading"] != "약화 칸으로 넘어감" or not d2["heading_change"]:
    fails.append("꼬리 설명 2")
if d3["path"] != ["중립", "소외", "개선", "주도"] or d3["weeks"] != 3:
    fails.append("꼬리 설명 3")
if d4["heading"] != "거의 제자리" or d4["path_text"] != "1주 내내 주도":
    fails.append("꼬리 설명 4")
if D([1.0], [1.0])["available"]:
    fails.append("점 1개인데 설명함")
print("결과:", "통과" if not fails else f"실패 — {fails}")
sys.exit(1 if fails else 0)
