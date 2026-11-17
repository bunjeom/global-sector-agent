# C:\Users\hanju\global-sector-agent\pages\2_🧭_피어_비교.py
# 피어 비교 (딥다이브 개편 5단계, 2026-11-17) — 묶음을 고르면 가치 관점·추세 관점 순위표, 종목을 골라 딥다이브로.
# 자료는 매일 GitHub Actions가 쌓는 data/peers/ 기록(engine/peers.py). 화면은 읽기만 한다.
import html as _html

import pandas as pd
import streamlit as st

import engine
import utils

st.set_page_config(page_title="피어 비교 | 투자터미널", layout="wide")
utils.init_session()

DEEPDIVE_PAGE = "pages/1_🔎_종목_딥다이브.py"
PERSPECTIVES = {"가치 관점": "value", "추세 관점": "trend"}


def html_md(s: str) -> None:
    """HTML 블록 렌더러 — 빈 줄·들여쓰기를 지워 마크다운이 코드블록으로 깨지지 않게 한다(딥다이브와 같은 방식)."""
    st.markdown("\n".join(l.strip() for l in str(s).split("\n") if l.strip()), unsafe_allow_html=True)


def _esc(x) -> str:
    return _html.escape("" if x is None else str(x))


def _name(t: str, short=None) -> str:
    nm = utils.TICKER_NAMES.get(str(t))
    if nm:
        return nm
    if short is None or (isinstance(short, float) and pd.isna(short)):
        return ""
    return str(short)[:28]


def _f0(x) -> str:
    return "-" if x is None or pd.isna(x) else f"{float(x):.0f}"


def _norm(t: str) -> str:
    t = str(t).strip().upper()
    if t.isdigit() and len(t) == 6:
        return f"{t}.KS"
    return t if t.endswith((".KS", ".KQ")) else t.replace(".", "-")


@st.cache_data(ttl=600, show_spinner=False)
def load_overview() -> dict:
    return engine.peer_groups_overview()


@st.cache_data(ttl=600, show_spinner=False)
def load_group(group: str) -> dict:
    return engine.peer_group_view(group)


st.markdown(f'## {utils.line_icon("layers", 22)} 피어 비교', unsafe_allow_html=True)
st.caption("같은 사업(주 수익원 기준)을 하는 회사끼리 묶어, 지금 누가 싸고(가치 관점) 누가 강한지(추세 관점) 순위로 봅니다. "
           "종목을 고르면 딥다이브로 넘어갑니다.")

ov = load_overview()
if not ov.get("available"):
    st.info(f"피어 비교: {ov.get('reason')}")
    st.stop()

groups = ov["groups"]
rank_date = pd.Timestamp(ov["rank_date"]).date()
n_val = int(groups["value_ranked"].sum())
n_tr = int(groups["trend_ranked"].sum())
html_md(f"""
<div class="term-card">
    <div>{utils.line_icon("compass", 16)} <b>묶음 {len(groups)}개</b> · 종목 {int(groups["n"].sum())}개 · 순위가 나온 묶음: 가치 관점 {n_val}개 · 추세 관점 {n_tr}개
    <span style="color:#64748b;font-size:0.86rem;">(순위 기준일 {rank_date})</span></div>
    <div style="font-size:0.86rem;margin-top:6px;"><b>가치 관점</b> = 밸류에이션 50 · 수익성 25 · 실적 모멘텀 25 &nbsp;|&nbsp; <b>추세 관점</b> = 가격 흐름 50 · 실적 모멘텀 25 · 성장 25. 점수는 묶음 안 백분위(100이 가장 좋음), 생존성 하드 킬 종목은 순위에서 뺍니다.</div>
    <div style="font-size:0.82rem;color:#64748b;margin-top:6px;">검증 상태: 미검증 — 같은 업종 안의 사실 비교이며, 순위가 이후 수익을 예측한다는 검증은 없습니다(매일 기록해 12개월 뒤 상위·하위 1/3로 채점). 참고: 추세 관점의 핵심인 63일 상대강도는 과거 탐색(EXPLORE_ret63.md, 증거 아님)에서 다음 달 수익을 가르지 못했습니다.</div>
</div>
""")

# ---------------------------------------------------------------- 묶음 고르기
c1, c2, c3 = st.columns([1.3, 1.0, 1.2])
with c1:
    query = st.text_input("종목으로 찾기 (티커, 예: NVDA · 005930)", value="")
with c2:
    market = st.selectbox("시장", ["전체", "미국", "한국"], index=0)
with c3:
    mine_only = st.checkbox("내 종목(대표·관심 종목)이 있는 묶음만", value=True)

view_groups = groups
if market != "전체":
    view_groups = view_groups[view_groups["market"].str.contains(market)]
if mine_only:
    view_groups = view_groups[view_groups["n_mine"] > 0]

found_group, q = None, ""
if query.strip():
    q = _norm(query)
    hit = groups[groups["members"].map(lambda m: q in m)]
    if len(hit):
        found_group = hit["group"].iloc[0]
    if found_group is None:
        st.warning(f"{q}: 피어 기록에 없는 종목입니다 — 커버리지 밖이면 data/snapshots/watchlist.txt에 추가하면 다음 갱신부터 비교합니다.")

options = list(view_groups["group"])
if found_group and found_group not in options:
    options = [found_group] + options
if not options:
    st.info("조건에 맞는 묶음이 없습니다. 위 필터를 바꿔 보세요.")
    st.stop()
label_of = {r["group"]: f'{r["group_name"]} · {r["n"]}종목' + (f' · 내 종목 {", ".join(r["mine"][:4])}' if r["n_mine"] else "")
            for _, r in groups.iterrows()}
sel = st.selectbox("묶음", options, index=(options.index(found_group) if found_group in options else 0),
                   format_func=lambda k: label_of.get(k, k))

with st.expander(f"묶음 한눈에 보기 ({len(view_groups)}개)", expanded=False):
    lead = lambda t: "-" if not t else (t + (f" {_name(t)}" if _name(t) else ""))
    st.dataframe(pd.DataFrame({
        "묶음": view_groups["group_name"], "출처": view_groups["group_source"], "시장": view_groups["market"],
        "종목 수": view_groups["n"], "내 종목": [", ".join(c[:6]) + (" …" if len(c) > 6 else "") for c in view_groups["mine"]],
        "가치 관점 1위": [lead(t) for t in view_groups["value_leader"]], "추세 관점 1위": [lead(t) for t in view_groups["trend_leader"]],
        "순위 없음 사유": view_groups["note"].replace("", "-"),
    }), width="stretch", hide_index=True)

# ---------------------------------------------------------------- 묶음 순위표
gv = load_group(sel)
if not gv.get("available"):
    st.info(f"피어 비교: {gv.get('reason')}")
    st.stop()
tbl = gv["table"]
persp = st.radio("순위 기준", list(PERSPECTIVES), horizontal=True)
key = PERSPECTIVES[persp]
tbl = tbl.sort_values([f"{key}_score"], ascending=False, na_position="last").reset_index(drop=True)
fd = gv.get("fund_date")
fd_txt = pd.Timestamp(fd).date() if fd is not None and not pd.isna(fd) else engine.NO_DATA


def leader_line(k: str) -> str:
    r = tbl[tbl[f"{k}_rank"] == 1]
    if not len(r):
        nt = tbl[f"{k}_note"].dropna().astype(str)
        return f'순위 없음 — {_esc(nt.iloc[0] if len(nt) else "자료 부족")}'
    t = str(r["ticker"].iloc[0])
    nm = _name(t, r["short_name"].iloc[0])
    return f'<b>{_esc(t)}</b> {_esc(nm)} ({_f0(r[f"{k}_score"].iloc[0])}점)'


n_rank = int(tbl[f"{key}_rank"].notna().sum())
html_md(f"""
<div class="term-card">
    <div>{utils.line_icon("layers", 16)} <b>{_esc(gv["group_name"])}</b> 피어 {len(tbl)}종목 <span class="term-badge">{_esc(gv.get("group_source"))}</span></div>
    <div style="margin-top:6px;">1위 — 가치 관점: {leader_line("value")} · 추세 관점: {leader_line("trend")}</div>
    <div style="font-size:0.82rem;color:#64748b;margin-top:6px;">순위 기준일 {pd.Timestamp(gv["rank_date"]).date()} · 재무·추정치 {fd_txt}(주 1회 갱신) · {persp} 순위가 있는 종목 {n_rank}개(상위 1/3 = {max(1, -(-n_rank // 3)) if n_rank else 0}개)</div>
</div>
""")

disp = pd.DataFrame({
    "순위": [_f0(x) for x in tbl[f"{key}_rank"]],
    "종목": tbl["ticker"].astype(str),
    "이름": [_name(a, b) for a, b in zip(tbl["ticker"], tbl["short_name"])],
    f"{persp} 점수": [_f0(x) for x in tbl[f"{key}_score"]],
    "상위 1/3": ["●" if str(x).lower() == "true" else "" for x in tbl[f"{key}_top"]],
    **{lbl: [_f0(x) for x in tbl[f"f_{k}"]] for k, lbl in engine.FACTOR_LABELS.items()},
    "선행 PER": [engine.fmt_peer_metric("fwd_pe", x) for x in tbl["fwd_pe"]],
    "매출 성장": [engine.fmt_peer_metric("rev_growth_ttm", x) for x in tbl["rev_growth_ttm"]],
    "영업이익률": [engine.fmt_peer_metric("op_margin_ttm", x) for x in tbl["op_margin_ttm"]],
    "63일 상대강도": [engine.fmt_peer_metric("rs63", x) for x in tbl["rs63"]],
    "추세": ["-" if x is None or pd.isna(x) else str(x) for x in tbl["trend_state"]],
    "생존성": ["하드 킬(순위 제외)" if str(k).lower() == "true" else ("경고" if str(w).lower() == "true" else "")
             for k, w in zip(tbl["surv_kill"], tbl["surv_warning"])],
    "내 종목": ["★" if t in set(groups.loc[groups["group"] == sel, "mine"].iloc[0]) else "" for t in tbl["ticker"].astype(str)],
})


def _hl(row):
    on = q and row["종목"] == q
    top = row["상위 1/3"] == "●"
    style = "background-color:#e0f2fe;font-weight:700;" if on else ("background-color:#f0fdf4;" if top else "")
    return [style for _ in row]


st.dataframe(disp.style.apply(_hl, axis=1), width="stretch", hide_index=True)
st.caption("요인 점수는 묶음 안 백분위(100이 가장 좋음), '-'는 자료 없음(계산에서 빠짐). 초록 줄 = 이 관점 상위 1/3, ★ = 내 종목(대표·관심 종목). "
           "피어 묶음은 회사의 주 수익원 기준이며 data/peers/peer_map.csv에서 고칩니다.")

# ---------------------------------------------------------------- 딥다이브로
tick_opts = list(tbl["ticker"].astype(str))
_mine_here = list(groups.loc[groups["group"] == sel, "mine"].iloc[0])
default_t = q if q in tick_opts else (next((t for t in tick_opts if t in _mine_here), tick_opts[0]))
d1, d2 = st.columns([2, 1])
with d1:
    pick = st.selectbox("딥다이브로 볼 종목", tick_opts, index=tick_opts.index(default_t),
                        format_func=lambda t: t + (f" {_name(t, tbl.loc[tbl['ticker'].astype(str) == t, 'short_name'].iloc[0])}"))
with d2:
    st.write("")
    if st.button("딥다이브 →", key="peer_to_deepdive"):
        st.session_state["selected_ticker"] = pick
        st.switch_page(DEEPDIVE_PAGE)
