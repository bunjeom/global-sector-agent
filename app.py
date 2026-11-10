# C:\Users\hanju\global-sector-agent\app.py
# 투자터미널 홈 — 결론 → 근거(시장 상태·섹터 흐름·자금 흐름) → 실행 순서
# 2026-11-08 개편: 4분면·게이트를 실제 yfinance 데이터로 연결, 예시 데이터는 기본 숨김
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

import engine
import utils

st.set_page_config(page_title="투자터미널 | 홈", layout="wide")
utils.init_session()

# =====================================================================
# 공통 표시 규칙 (색 의미: 초록=공격 가능 · 노랑=주의 · 주황=방어 · 빨강=위기)
# =====================================================================
REGIME_COLOR = {"🟢": "#16a34a", "🟡": "#ca8a04", "🔴": "#ea580c", "⚫": "#dc2626"}
GATE_COLOR = {0: "#16a34a", 1: "#ca8a04", 2: "#ea580c", 3: "#dc2626"}
QUAD_COLOR = {"주도": "#16a34a", "개선": "#2563eb", "약화": "#ca8a04", "소외": "#dc2626", "중립": "#64748b"}
QUAD_PLAIN = {
    "주도": "지수보다 강하고, 더 강해지는 중",
    "개선": "아직 지수보다 약하지만, 빠르게 좋아지는 중",
    "약화": "지수보다 강하지만, 힘이 빠지는 중",
    "소외": "지수보다 약하고, 더 약해지는 중",
    "중립": "뚜렷한 방향 없음",
}


def fmt_ret(v):
    """수익률(소수) → '+1.2%' 형식, 값이 없으면 '데이터 없음'"""
    return f"{v * 100:+.1f}%" if v is not None else "데이터 없음"


def kr_date(s):
    """'2026-11-06' → '11월 6일'"""
    try:
        _, m, d = str(s)[:10].split("-")
        return f"{int(m)}월 {int(d)}일"
    except Exception:
        return "날짜 없음"


def regime_chip(name: str, reg: dict) -> str:
    """지수 레짐 배지: 이모지 대신 의미가 고정된 색 점 + 글자"""
    icon = reg.get("icon", "")
    label = str(reg.get("label", "데이터 없음"))
    if label[:1] in REGIME_COLOR:
        label = label[1:].strip()
    color = REGIME_COLOR.get(icon, "#64748b")
    return (f'<span class="term-badge"><span style="color:{color};">●</span> '
            f'{name} <b style="color:{color};">{label}</b></span>')


def short_name(full: str) -> str:
    """'우주 · 항공 · 방산' → '우주·항공·방산' (첫 단어만 자르면 '우주'처럼 오해를 부름)"""
    return str(full).split(" (")[0].replace(" · ", "·")


# =====================================================================
# 데이터 로드 (3년치: 주봉 4분면이 2년 표준편차를 쓰기 때문)
# =====================================================================
def download_ohlc_map(symbols: list, period: str, min_rows: int = 25) -> dict:
    """yfinance로 여러 종목 일봉을 한 번에 받는다. 받지 못한 종목은 빠진다(예시 데이터로 채우지 않음)."""
    out = {}
    try:
        raw = yf.download(symbols, period=period, group_by="ticker", auto_adjust=False, progress=False, threads=True)
        if raw is not None and len(raw) > 0:
            for sym in symbols:
                if sym in raw.columns.get_level_values(0):
                    sub_df = raw[sym].dropna(subset=["Close"]).copy()
                    if len(sub_df) >= min_rows:
                        out[sym] = sub_df
    except Exception:
        pass
    return out


@st.cache_data(ttl=1800, show_spinner="시장 데이터를 불러오는 중...")
def load_home_dashboard_data() -> dict:
    core_tickers = ["SPY", "QQQ", "^KS11", "^VIX", "RSP", "IWM", "TLT"]
    basket_tickers = [b["ticker"] for b in engine.NARRATIVE_BASKETS_SPEC]
    all_syms = list(dict.fromkeys(core_tickers + basket_tickers))
    etf_map = download_ohlc_map(all_syms, period="3y")

    report_json = utils.load_latest_report()
    dash = engine.run_home_dashboard_engine(etf_data_map=etf_map, report_json=report_json)
    # 4분면은 실제 종가로만 만든다 (데이터가 없으면 그리지 않음 — 예시 데이터로 대체하지 않음)
    dash["rrg_bundle"] = engine.build_real_rrg_bundle(
        etf_map, dash["narrative_rotation"]["baskets"], write_log=True
    )
    # 국면 게이트 히스테리시스: 최근 거래일을 과거 종가로 다시 계산해 '어제까지의 단계'를 만든다
    dash["gate_history"] = {mk: engine.build_gate_history_state(etf_map, mk) for mk in ("US", "KR")}
    return dash


@st.cache_data(ttl=1800, show_spinner="실행 레이더 신호를 계산하는 중... (종목당 1~2초)")
def load_radar_signals(basket_key: tuple, total_equity: float, t_sleeve_ratio: float) -> dict:
    """
    강한 섹터 ETF의 상위 구성 종목에서 레이더 종목을 고르고(engine.select_radar_universe),
    실제 T 신호를 계산한다(engine.build_radar_signals).
    """
    from concurrent.futures import ThreadPoolExecutor
    baskets = [{"basket": b, "basket_role": r} for b, r in basket_key]
    cfg = engine.RADAR_CONFIG
    # 1) 섹터 ETF 상위 구성 종목 (여러 ETF 동시 조회)
    with ThreadPoolExecutor(max_workers=6) as ex:
        holdings = list(ex.map(engine.fetch_etf_top_holdings, [b["basket"] for b in baskets]))
    cand_map = {b["basket"]: engine.build_basket_candidates(b["basket"], h) for b, h in zip(baskets, holdings)}
    # 2) 후보 전체 일봉 + 환율·기준 지수·VIX
    cand_syms = [c["ticker"] for v in cand_map.values() for c in v["candidates"]]
    syms = list(dict.fromkeys(cand_syms + [cfg["FX_TICKER"], cfg["BENCH"]["US"], cfg["BENCH"]["KR"], "^VIX"]))
    pm = download_ohlc_map(syms, period=cfg["PRICE_PERIOD"])
    bench_map = {m: pm.get(t) for m, t in cfg["BENCH"].items()}
    # 3) 추세 통과 + 3개월 상대강도 순으로 섹터당 상위 종목 선정
    sel = engine.select_radar_universe(baskets, cand_map, pm, bench_map)
    universe = sel["universe"]
    # 4) 다음 실적 발표일 (실적 발표 D-3 ~ 당일 신규 진입 보류용)
    with ThreadPoolExecutor(max_workers=8) as ex:
        earn_dates = list(ex.map(engine.fetch_next_earnings_date, [u["ticker"] for u in universe]))
    for u, d in zip(universe, earn_dates):
        u["next_earnings"] = d
    fx_df = pm.get(cfg["FX_TICKER"])
    fx_close = fx_df["Close"].dropna() if fx_df is not None else None
    fx = float(fx_close.iloc[-1]) if fx_close is not None and len(fx_close) > 0 else None
    vix = pm["^VIX"]["Close"] if "^VIX" in pm else None
    res = engine.build_radar_signals(
        universe, pm, bench_map, vix,
        {"total_equity": total_equity, "t_sleeve_ratio": t_sleeve_ratio}, fx,
    )
    res["fx_usdkrw"] = fx
    res["universe"] = universe
    res["selection_log"] = sel["log"]
    return res


def format_meta_badges(rec: dict) -> str:
    """레코드의 출처·기준일·공시일과 신선도/예시 데이터 회색 배지"""
    src = rec.get("source") or "출처 미상"
    as_of = rec.get("as_of")
    pub = rec.get("published")
    kind = rec.get("kind", "")
    if kind == "13f":
        meta_txt = f"13F · {as_of + ' 보유 기준' if as_of else '보유 기준일 없음'} · {pub + ' 공시' if pub else '공시일 없음'}"
    elif kind == "form4":
        meta_txt = f"{src} · {pub + ' 공시' if pub else (as_of + ' 기준' if as_of else '공시일 없음')}"
    else:
        meta_txt = f"{src} · 기준일: {as_of if as_of else '데이터 없음'}"
    fresh_badges = rec.get("freshness", {}).get("badges", [])
    gray_spans = "".join(
        f'<span style="display:inline-block;background:#e2e8f0;color:#475569;border-radius:4px;'
        f'padding:1px 6px;font-size:0.75rem;font-weight:600;margin-left:4px;">{b}</span>'
        for b in fresh_badges
    )
    return f'<span style="font-size:0.78rem;color:#64748b;">({meta_txt})</span>{gray_spans}'


dash = load_home_dashboard_data()
m_hdr = dash["macro_header"]
s_mon = dash["smart_money_tracker"]
rb = dash.get("rrg_bundle", {}) or {}
ev = rb.get("evaluation", {}) if rb.get("available") else {}
baskets_eval = ev.get("baskets", {})
mb = m_hdr["market_breadth"]
vx = m_hdr["vix_state"]
mkt_as_of = dash["meta"]["as_of"]

# 국면 게이트: 헤더와 같은 실제 데이터로 계산 (하드코딩 값 사용 금지)
# 히스테리시스: 어제까지의 단계(gate_history)를 넘겨 '완화는 2거래일 연속' 규칙을 적용
gate_hist = dash.get("gate_history", {})
us_gate = engine.evaluate_market_gate("US", us_data=engine.build_gate_inputs_from_header(m_hdr, "US"), kr_data={},
                                      history_state=gate_hist.get("US", {}).get("state_before_last"))
kr_gate = engine.evaluate_market_gate("KR", us_data={}, kr_data=engine.build_gate_inputs_from_header(m_hdr, "KR"),
                                      history_state=gate_hist.get("KR", {}).get("state_before_last"))

# =====================================================================
# 1. 오늘의 결론 (페이지 맨 위 큰 글자 자리)
# =====================================================================
top_level = max(us_gate["level"], kr_gate["level"])
if us_gate["level"] == kr_gate["level"]:
    action_txt = us_gate["action_summary"]
    gate_txt = f"미국·한국 모두 {us_gate['code']} {us_gate['name'].split(' (')[0]}"
else:
    action_txt = f"미국: {us_gate['action_summary']} / 한국: {kr_gate['action_summary']}"
    gate_txt = f"미국 {us_gate['code']} · 한국 {kr_gate['code']}"

leaders_txt = ", ".join(
    f"{x['basket']}({short_name(x['name'])})" for x in ev.get("leading_top3", [])
) or "해당 없음"
# 해제 조건은 더 나쁜(높은) 게이트 쪽을 보여준다. 같은 단계면 조건이 다를 때 둘 다 표시.
if top_level == 0:
    release_txt = "해당 없음"
elif us_gate["level"] == kr_gate["level"]:
    release_txt = (us_gate["release_condition"] if us_gate["release_condition"] == kr_gate["release_condition"]
                   else f"미국: {us_gate['release_condition']} / 한국: {kr_gate['release_condition']}")
elif us_gate["level"] > kr_gate["level"]:
    release_txt = f"미국: {us_gate['release_condition']}"
else:
    release_txt = f"한국: {kr_gate['release_condition']}"

st.markdown(f"""
<div style="border-left:6px solid {GATE_COLOR[top_level]};background:#f8fafc;padding:14px 18px;border-radius:6px;margin-bottom:14px;">
  <div style="font-size:0.85rem;color:#64748b;">{kr_date(mkt_as_of)} 마감 기준 · 오늘의 결론</div>
  <div style="font-size:1.55rem;font-weight:800;color:#0f172a;line-height:1.35;margin:2px 0 6px 0;">
    {gate_txt} — <span style="color:{GATE_COLOR[top_level]};">{action_txt}</span>
  </div>
  <div style="font-size:0.92rem;color:#334155;">
    시장 폭: <b>{mb.get('rsp_spy_label', '데이터 없음')}</b> · 지금 강한 곳: <b>{leaders_txt}</b><br>
    <span style="color:#64748b;">게이트 해제 조건: {release_txt}</span>
  </div>
</div>
""", unsafe_allow_html=True)

# =====================================================================
# 2. 시장 상태 (한 줄 배지 + 한 줄 해석, 기준일은 오른쪽에 한 번만)
# =====================================================================
rsp_chg_str = f"{mb['rsp_spy_20d_chg'] * 100:+.1f}%p" if mb["rsp_spy_20d_chg"] is not None else "데이터 없음"
iwm_chg_str = f"{mb['iwm_qqq_20d_chg'] * 100:+.1f}%p" if mb["iwm_qqq_20d_chg"] is not None else "데이터 없음"
vix_val_str = f"{vx['value']} (1년 {vx['percentile_1y']}%ile · {vx['label']})" if vx["value"] is not None else "데이터 없음"
breadth_tip = f"{mb.get('label', '')} | 출처: {mb.get('source', '')} · 기준일: {mb.get('as_of') or '데이터 없음'}"

st.markdown(f"""
<div class="term-banner" style="padding:10px 16px;margin-bottom:18px;">
  <div style="display:flex;flex-wrap:wrap;align-items:center;gap:8px;margin-bottom:6px;">
    {regime_chip("나스닥", m_hdr["qqq_regime"])}
    {regime_chip("S&P 500", m_hdr["spy_regime"])}
    {regime_chip("KOSPI", m_hdr["kospi_regime"])}
    <span class="term-badge">VIX {vix_val_str}</span>
    <span class="term-badge">20일 상대: RSP/SPY {rsp_chg_str} · IWM/QQQ {iwm_chg_str}</span>
    <span style="margin-left:auto;font-size:0.78rem;color:#64748b;">기준일 {mkt_as_of} · 출처 yfinance</span>
  </div>
  <div style="font-size:0.9rem;color:#111;">
    <b>거시 국면:</b> {m_hdr["macro_narrative"]} &nbsp;|&nbsp;
    <b>시장 폭:</b> <span title="{breadth_tip}" style="border-bottom:1px dotted #94a3b8;cursor:help;">
    S&amp;P 500 {mb.get('rsp_spy_label', '데이터 없음')} · 나스닥/소형주 {mb.get('iwm_qqq_label', '데이터 없음')}</span>
  </div>
</div>
""", unsafe_allow_html=True)

# =====================================================================
# 3. 섹터 흐름 (실제 종가 기반 4분면)
# =====================================================================
st.markdown("---")
t_col_title, t_col_clock, t_col_bench = st.columns([2.2, 1.4, 1.4])
with t_col_clock:
    clock_choice = st.radio(
        "시계", options=["daily", "weekly"], horizontal=True, key="rrg_clock_toggle",
        format_func=lambda x: "순환 · 일봉" if x == "daily" else "구조 · 주봉",
    )
with t_col_bench:
    bench_choice = st.radio(
        "기준 지수", options=["SPY", "RSP"], horizontal=True, key="rrg_bench_toggle",
        format_func=lambda x: "SPY (시가총액가중)" if x == "SPY" else "RSP (동일가중)",
    )
with t_col_title:
    st.markdown("#### 섹터 흐름")
    st.caption(f"{bench_choice} 대비 강도와 가속 · 꼬리 = 최근 4주")


def grade_for(quad: str, m: dict) -> str:
    """토글(주봉·RSP)로 바뀐 사분면에 대해 같은 규칙으로 등급 문구를 만든다"""
    if quad in ("주도", "개선", "약화", "소외") and not m.get("revision_known", False):
        return f"{quad} ({'수급 확인' if m.get('flow_positive') else '수급 약함'} · 실적 데이터 없음)"
    return engine.determine_basket_grade(quad, m.get("revision_positive", False), m.get("flow_positive", False))


def build_rrg_figure(coords_map: dict, struct_map: dict, meta_info: dict, clock_name: str,
                     highlight: set, show_all_tails: bool):
    lim = engine.RRG_CONFIG["AXIS_LIMIT"]
    nth = engine.RRG_CONFIG["NEUTRAL_THRESHOLD"]
    chart_h = engine.RRG_CONFIG["CHART_HEIGHT"]
    fig = go.Figure()
    fig.update_layout(shapes=[
        dict(type="rect", x0=0, y0=0, x1=lim, y1=lim, fillcolor="rgba(34,197,94,0.06)", line_width=0, layer="below"),
        dict(type="rect", x0=-lim, y0=0, x1=0, y1=lim, fillcolor="rgba(59,130,246,0.06)", line_width=0, layer="below"),
        dict(type="rect", x0=-lim, y0=-lim, x1=0, y1=0, fillcolor="rgba(239,68,68,0.06)", line_width=0, layer="below"),
        dict(type="rect", x0=0, y0=-lim, x1=lim, y1=0, fillcolor="rgba(234,179,8,0.06)", line_width=0, layer="below"),
        dict(type="rect", x0=-nth, y0=-nth, x1=nth, y1=nth, fillcolor="rgba(148,163,184,0.18)",
             line=dict(color="rgba(148,163,184,0.45)", width=1, dash="dot"), layer="below"),
        dict(type="line", x0=-lim, y0=0, x1=lim, y1=0, line=dict(color="rgba(148,163,184,0.4)", width=1)),
        dict(type="line", x0=0, y0=-lim, x1=0, y1=lim, line=dict(color="rgba(148,163,184,0.4)", width=1)),
    ])
    annotations = [
        dict(x=lim - 0.15, y=lim - 0.18, text="<b>주도</b>", showarrow=False, xanchor="right", yanchor="top", font=dict(size=12, color="#15803d")),
        dict(x=-lim + 0.15, y=lim - 0.18, text="<b>개선</b>", showarrow=False, xanchor="left", yanchor="top", font=dict(size=12, color="#1d4ed8")),
        dict(x=-lim + 0.15, y=-lim + 0.18, text="<b>소외</b>", showarrow=False, xanchor="left", yanchor="bottom", font=dict(size=12, color="#b91c1c")),
        dict(x=lim - 0.15, y=-lim + 0.18, text="<b>약화</b>", showarrow=False, xanchor="right", yanchor="bottom", font=dict(size=12, color="#a16207")),
    ]

    pts = [{"basket": b, "x": d["x"], "y": d["y"]} for b, d in coords_map.items()
           if not (np.isnan(d.get("x", np.nan)) or np.isnan(d.get("y", np.nan)))]
    label_res = engine.place_labels(pts, fig_size=(700, chart_h))
    placements = label_res["placements"]

    for b_code, d in coords_map.items():
        raw_x, raw_y = d.get("x", np.nan), d.get("y", np.nan)
        if np.isnan(raw_x) or np.isnan(raw_y):
            continue
        quad = d["quadrant"]
        m_info = meta_info.get(b_code, {})
        c_hex = QUAD_COLOR.get(quad, "#64748b")
        is_hl = b_code in highlight
        cx, cy = float(np.clip(raw_x, -lim, lim)), float(np.clip(raw_y, -lim, lim))
        clipped = abs(raw_x) > lim or abs(raw_y) > lim

        # 꼬리: 기본은 강조 바스켓(지금 강한 곳·다음 후보)만, 체크하면 전부
        tx = [float(np.clip(v, -lim, lim)) for v in d.get("tail_x", [])]
        ty = [float(np.clip(v, -lim, lim)) for v in d.get("tail_y", [])]
        if len(tx) >= 2 and (show_all_tails or is_hl):
            fig.add_trace(go.Scatter(x=tx, y=ty, mode="lines+markers",
                                     line=dict(color=c_hex, width=1.6), marker=dict(size=4, color=c_hex),
                                     opacity=0.5, hoverinfo="skip", showlegend=False))
            annotations.append(dict(x=tx[-1], y=ty[-1], ax=tx[-2], ay=ty[-2], xref="x", yref="y",
                                    axref="x", ayref="y", showarrow=True, arrowhead=2, arrowsize=1.1,
                                    arrowwidth=1.6, arrowcolor=c_hex, text=""))

        hover = (
            f"<b>{b_code} · {m_info.get('name', b_code)}</b><br>"
            f"{QUAD_PLAIN.get(quad, quad)}<br>"
            f"판정: {grade_for(quad, m_info)}<br>"
            f"구조(주봉): {struct_map.get(b_code, {}).get('quadrant', '데이터 없음')}<br>"
            f"X {raw_x:+.2f}σ · Y {raw_y:+.2f}σ ({clock_name})"
            + (f"<br>표본 짧음({d.get('obs_count')}개)" if d.get("short_sample") else "")
        )
        fig.add_trace(go.Scatter(
            x=[cx], y=[cy], mode="markers",
            marker=dict(size=11 if is_hl else 8, color=c_hex, opacity=1.0 if is_hl else 0.55,
                        line=dict(width=1.5, color="#ffffff")),
            hovertemplate=hover + "<extra></extra>", showlegend=False,
        ))
        pl = placements.get(b_code, {"visible": False})
        if pl.get("visible"):
            annotations.append(dict(
                x=cx, y=cy, text=f"<b>{b_code}{' ▶' if clipped else ''}</b>",
                showarrow=bool(pl.get("use_leader", False)), arrowhead=0, arrowwidth=1,
                arrowcolor="rgba(100,116,139,0.7)", ax=pl["dx"], ay=pl["dy"],
                font=dict(size=11, color="#1e293b" if is_hl else "#64748b"),
                bgcolor="rgba(255,255,255,0.78)", borderpad=2,
            ))

    fig.update_layout(
        annotations=annotations,
        xaxis=dict(title="X: 지수 대비 강도 (σ)", range=[-lim, lim], zeroline=False, dtick=1.0, gridcolor="rgba(148,163,184,0.18)"),
        yaxis=dict(title="Y: 강도가 바뀌는 속도 (σ)", range=[-lim, lim], zeroline=False, dtick=1.0, gridcolor="rgba(148,163,184,0.18)"),
        height=chart_h, margin=dict(l=45, r=20, t=15, b=40), plot_bgcolor="#ffffff", hovermode="closest",
    )
    return fig, label_res["hidden_count"]


def basket_card(item: dict, border: str) -> str:
    tip = f"X {item['x']:+.2f}σ · Y {item['y']:+.2f}σ"
    return (
        f"<div title='{tip}' style='border:1px solid {border};border-radius:8px;padding:9px 12px;margin-bottom:8px;'>"
        f"<b>{item['basket']}</b> <span style='color:#475569;'>{item['name']}</span><br>"
        f"<span style='font-size:0.86rem;color:#0f172a;'>{QUAD_PLAIN.get(item['quadrant'], '')}</span><br>"
        f"<span style='font-size:0.78rem;color:#64748b;'>판정: {item['grade']} · 구조(주봉): {item['structural_quadrant']}</span>"
        f"</div>"
    )


if not rb.get("available"):
    st.warning(
        f"섹터 흐름 데이터를 불러오지 못했습니다 ({rb.get('reason', '원인 미상')}). "
        "예시 데이터로 대신 그리지 않습니다. 잠시 후 새로고침해 주세요."
    )
else:
    prices_df = rb["prices_df"]
    bench_series = rb["bench_spy"] if bench_choice == "SPY" else rb.get("bench_rsp")
    if bench_series is None:
        st.caption("RSP 데이터가 없어 SPY 기준으로 표시합니다.")
        bench_series = rb["bench_spy"]
    active_coords = engine.rrg_coordinates(prices_df, bench_series, clock=clock_choice)
    structural_coords = engine.rrg_coordinates(prices_df, bench_series, clock="weekly")
    highlight = {x["basket"] for x in ev.get("leading_top3", []) + ev.get("igniting_top3", [])}

    rrg_col_chart, rrg_col_cards = st.columns([3.2, 1.8])
    with rrg_col_chart:
        show_all_tails = st.checkbox("모든 섹터의 꼬리 보기", value=False, key="rrg_all_tails")
        rrg_fig, hidden_cnt = build_rrg_figure(
            active_coords, structural_coords, rb["meta"],
            "일봉" if clock_choice == "daily" else "주봉", highlight, show_all_tails,
        )
        st.plotly_chart(rrg_fig, width="stretch", config={"displayModeBar": False})
        hidden_txt = f"라벨 생략 {hidden_cnt}개(마우스를 올리면 표시) · " if hidden_cnt > 0 else ""
        st.caption(f"{hidden_txt}기준일 {rb['as_of']} · 출처 yfinance 종가 · 축은 각 ETF 자체 변동성(σ)으로 정규화")

    with rrg_col_cards:
        st.markdown("##### 지금 강한 곳")
        lead = ev.get("leading_top3", [])
        if lead:
            st.markdown("".join(basket_card(x, "#16a34a") for x in lead), unsafe_allow_html=True)
        else:
            st.caption("지금 주도 사분면에 있는 섹터가 없습니다.")
        st.markdown("##### 다음 후보")
        ign = ev.get("igniting_top3", [])
        if ign:
            st.markdown("".join(basket_card(x, "#2563eb") for x in ign), unsafe_allow_html=True)
        else:
            st.caption("조건을 통과한 개선 사분면 섹터가 없습니다." if ev.get("igniting_excluded")
                       else "지금 개선 사분면에 있는 섹터가 없습니다.")
        ign_ex = ev.get("igniting_excluded", [])
        if ign_ex:
            st.caption("제외: " + " · ".join(f"{x['basket']}({', '.join(x['reasons'])})" for x in ign_ex))
        st.caption("다음 후보 조건: 개선 사분면 + 최근 4주 상대강도 상승 + 주봉 큰 흐름이 '소외'가 아님")
        st.caption(f"판정 기준: 일봉 · SPY 대비 · 기준일 {rb['as_of']}")

# =====================================================================
# 4. 자금 흐름 (예시 데이터는 기본 숨김)
# =====================================================================
st.markdown("---")
f_col_title, f_col_toggle = st.columns([4, 1])
with f_col_title:
    st.markdown("#### 자금 흐름")
    st.caption("자산군 수익률 · 섹터 수급 · 종목 신호")
with f_col_toggle:
    show_sample = st.toggle("예시 데이터 보기", value=False, key="show_sample_flow",
                            help="실제 데이터 출처가 아직 연결되지 않은 항목입니다. 투자 판단에 쓰지 마세요.")


def is_sample_rec(rec: dict) -> bool:
    return bool(rec.get("is_sample")) or "SAMPLE_DATA" in str(rec.get("source", ""))


def visible(recs: list) -> tuple:
    shown = [r for r in recs if show_sample or not is_sample_rec(r)]
    return shown, len(recs) - len(shown)


def hidden_note(n: int) -> str:
    return (f'<div style="font-size:0.8rem;color:#94a3b8;margin-top:4px;">실제 데이터 연결 전 · 예시 {n}건 숨김</div>'
            if n > 0 else "")


col_sm1, col_sm2, col_sm3 = st.columns(3)
with col_sm1:
    ca_li = "".join(
        f'<li><span class="term-badge">{c["flow_status"]}</span> <b>{c["asset"]}</b> — '
        f'4주 <b>{fmt_ret(c["ret_4w"])}</b> / 13주 <b>{fmt_ret(c["ret_13w"])}</b>{format_meta_badges(c)}</li>'
        for c in s_mon["cross_asset_flow"]
    )
    st.markdown(f"""
    <div class="term-card" style="height:100%;">
      <div style="font-weight:700;margin-bottom:8px;">자산군 수익률</div>
      <ul class="term-list">{ca_li}</ul>
      <div style="font-size:0.84rem;color:#111;background:#f8f9fa;padding:8px 10px;border-left:3px solid #111;margin-top:8px;">
        {s_mon["cross_asset_summary"]}
      </div>
    </div>
    """, unsafe_allow_html=True)

with col_sm2:
    us_in_li = "".join(
        f'<li><b>{u["ticker"]} ({u["name"]})</b> CMF(20) <b>{u["cmf_20"]:+.2f}</b> · 상승/하락 거래대금 <b>{u["ud_vol_ratio"]:.2f}배</b> '
        f'<span style="font-size:0.76rem;color:#64748b;">({u["source"]} · 기준일: {u["as_of"] or "데이터 없음"})</span></li>'
        for u in s_mon["us_sector_inflow_top4"]
    )
    kr_shown, kr_hidden = visible(s_mon["kr_inst_foreign_top3"])
    kr_in_li = "".join(
        f'<li><b>{k["sector"]}</b> 20일 외인·기관 <b>{k["net_buy_20d_krw"]}</b> ({k["top_pick"]})<br>{format_meta_badges(k)}</li>'
        for k in kr_shown
    )
    st.markdown(f"""
    <div class="term-card" style="height:100%;">
      <div style="font-weight:700;margin-bottom:8px;">섹터 수급</div>
      <div style="font-size:0.82rem;font-weight:700;color:#475569;margin-bottom:4px;">미국 · 기관 매집(CMF 20일) 상위</div>
      <ul class="term-list" style="margin-bottom:8px;">{us_in_li}</ul>
      <div style="font-size:0.82rem;font-weight:700;color:#475569;margin-bottom:4px;">한국 · 외국인·기관 20일 순매수 상위</div>
      <ul class="term-list">{kr_in_li}</ul>{hidden_note(kr_hidden)}
    </div>
    """, unsafe_allow_html=True)

with col_sm3:
    ms = s_mon["micro_smart_signals"]
    groups = [("매집봉", ms.get("stealth_accumulation", [])), ("내부자", ms.get("insider_buying", [])),
              ("13F", ms.get("hedge_fund_13f_consensus", []))]
    sig_li, sig_hidden = "", 0
    for tag, recs in groups:
        shown, n_hid = visible(recs)
        sig_hidden += n_hid
        sig_li += "".join(
            f'<li><span class="term-badge">{tag}</span> <b>{x["name"]}</b> {x["detail"]}<br>{format_meta_badges(x)}</li>'
            for x in shown
        )
    st.markdown(f"""
    <div class="term-card" style="height:100%;">
      <div style="font-weight:700;margin-bottom:8px;">종목 신호</div>
      <ul class="term-list">{sig_li}</ul>{hidden_note(sig_hidden)}
    </div>
    """, unsafe_allow_html=True)

# =====================================================================
# 5. 실행 레이더 (국면 게이트가 실행 여부와 크기를 정함)
# =====================================================================
st.markdown("---")
st.markdown("#### 실행 레이더")
st.caption("시장 국면 게이트가 종목 신호의 실행 여부와 크기를 정합니다. 손절·비중 축소 같은 방어 행동은 어떤 국면에서도 막지 않습니다.")


def gate_box(title: str, g: dict) -> str:
    color = GATE_COLOR.get(g["level"], "#64748b")
    return (
        f"<div style='border:1px solid {color};border-left:6px solid {color};border-radius:6px;padding:10px 12px;'>"
        f"<div style='font-size:0.8rem;color:#64748b;'>{title}</div>"
        f"<div style='font-size:1.05rem;font-weight:700;color:{color};'>{g['code']} {g['name']}</div>"
        f"<div style='font-size:0.88rem;color:#0f172a;margin:2px 0;'>{g['action_summary']}</div>"
        f"<div style='font-size:0.78rem;color:#475569;'>결정 요인: {g['binding_component']}</div>"
        f"<div style='font-size:0.78rem;color:#475569;'>해제 조건: {g['release_condition']}</div>"
        + ("" if g.get("history_used") else
           "<div style='font-size:0.74rem;color:#ca8a04;'>이전 단계 기록 없음 — 오늘 지표만으로 판정</div>")
        +
        f"<div style='font-size:0.74rem;color:#94a3b8;'>기준일 {g.get('as_of') or '데이터 없음'} · {g.get('source', '')}</div>"
        f"</div>"
    )


g_us, g_kr = st.columns(2)
with g_us:
    st.markdown(gate_box("미국 · 크립토 게이트", us_gate), unsafe_allow_html=True)
with g_kr:
    st.markdown(gate_box("한국 게이트", kr_gate), unsafe_allow_html=True)
st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

# 레이더 대상: '지금 강한 곳'·'다음 후보' 섹터 ETF의 상위 구성 종목 중 추세·상대강도 상위 (나중에 사용자 관심 종목을 앞에 추가)
radar_baskets = engine.select_radar_baskets(ev) if ev else []
radar_universe = []
if radar_baskets:
    radar_res = load_radar_signals(
        tuple((b["basket"], b["basket_role"]) for b in radar_baskets),
        float(st.session_state["total_equity"]), float(st.session_state["t_sleeve_ratio"]),
    )
    radar_universe = radar_res.get("universe", [])
    if not radar_universe:
        radar_res["fatal"] = "강한 섹터의 후보 종목 중 추세(50일선 위)를 통과한 종목이 없습니다"
else:
    no_basket_msg = ("지금 '주도'·'개선' 사분면에 있는 섹터가 없어 레이더 대상 종목이 없습니다" if rb.get("available")
                     else "섹터 흐름 데이터가 없어 레이더 종목을 고르지 못했습니다")
    radar_res = {"signals": [], "no_signal": [], "errors": [], "fatal": no_basket_msg}
all_signals = radar_res["signals"]
if baskets_eval and all_signals:
    all_signals = engine.sync_radar_basket_badges({"all": all_signals}, baskets_eval)["all"]

# 게이트 적용 후 '실행 / 관찰 / 방어 / 동결 / 장기보유 예외'로 다시 분류
buckets = {"실행": [], "관찰": [], "방어": [], "동결": [], "예외": []}
for sig in all_signals:
    g = kr_gate if str(sig.get("market", "US")).upper() == "KR" else us_gate
    if sig.get("earnings_blackout"):
        # 실적 발표 임박: 게이트와 무관하게 신규 진입 보류 (종합 결론과 같은 규칙)
        a = {**sig, "gate_applied": False, "orig_action": sig.get("action_text"), "orig_weight": 100,
             "effective_action": sig.get("action_text"), "effective_weight": 0, "status": "EARNINGS_BLACKOUT",
             "reason": f"실적 발표 D-{engine.TECH_PARAMS['EARNINGS_BLACKOUT_DAYS']} ~ 당일은 전 셋업 신규 진입 제한", "_gate": g}
        buckets["동결"].append(a)
        continue
    a = engine.apply_gate_action_matrix(sig, g["level"])
    a["_gate"] = g
    if a["status"] == "DEFENSE_PRIORITY":
        buckets["방어"].append(a)
    elif a["status"] == "CORE_EXEMPT":
        buckets["예외"].append(a)
    elif a["status"] in ("BLOCKED", "FROZEN") or a.get("effective_weight", 0) == 0:
        buckets["동결"].append(a)
    elif sig.get("signal_type") == "T_WATCH":
        buckets["관찰"].append(a)
    else:
        buckets["실행"].append(a)


def fmt_px(v, currency) -> str:
    """가격 표시: 미국 종목은 달러, 한국 종목은 원"""
    if v is None:
        return "-"
    return f"${float(v):,.2f}" if currency == "USD" else f"{int(v):,}원"


def render_signal(a: dict, key_prefix: str):
    g = a["_gate"]
    color = GATE_COLOR.get(g["level"], "#64748b")
    st.markdown(
        f"**{a.get('ticker', '데이터 없음')}** "
        f"<span style='font-size:0.78rem;color:{color};border:1px solid {color};border-radius:4px;padding:0 5px;'>"
        f"{str(a.get('market', 'US')).upper()} · {g['code']}</span>",
        unsafe_allow_html=True,
    )
    st.caption(f"섹터 맥락: {a.get('basket_sync_badge') or a.get('basket', '')} · {a.get('basket_role', '')}")
    if a.get("selection_note"):
        st.caption(f"선정 이유: {a['selection_note']}")
    if a.get("gate_applied"):
        st.markdown(f"<span style='color:#94a3b8;'>원신호: <s>{a['orig_action']} (계획 비중의 {a['orig_weight']}%)</s></span>",
                    unsafe_allow_html=True)
    is_def = a.get("status") == "DEFENSE_PRIORITY"
    hide_w = is_def or a.get("status") == "EARNINGS_BLACKOUT"
    st.markdown(f"**{a['effective_action']}**" + ("" if hide_w else f" · 계획 비중의 {a.get('effective_weight', 0)}%"))
    if a.get("entry") is not None and not is_def:
        r_txt = f" · 예상 {a['expected_r']}R" if a.get("expected_r") is not None else ""
        st.caption(f"진입 {fmt_px(a['entry'], a.get('currency'))} · 손절 {fmt_px(a.get('stop'), a.get('currency'))}"
                   f" · 1차 목표 {fmt_px(a.get('target_1'), a.get('currency'))}{r_txt}")
    st.caption(f"다음 실적 발표: {a['next_earnings']} (D-{a['days_to_earnings']})" if a.get("days_to_earnings") is not None
               else "다음 실적 발표: 일정 미확인")
    st.caption(f"근거: {a.get('headline', '')} · 상태: {a.get('state_label', '')}")
    if a.get("next_cond"):
        st.caption(a["next_cond"])
    st.caption(f"게이트: {a.get('reason', '')}")
    st.caption(f"출처: {a.get('source', '')} · 기준일: {a.get('as_of') or '없음'}")
    if st.button("딥다이브 →", key=f"{key_prefix}_{a.get('ticker')}_{a.get('market')}"):
        st.session_state["selected_ticker"] = a.get("ticker")
        st.switch_page("pages/1_🔎_종목_딥다이브.py")


def render_radar():
    order = ["방어", "실행", "관찰"] if top_level >= 3 else ["실행", "관찰", "방어"]
    titles = {"실행": "실행", "관찰": "관찰", "방어": "방어 (손절·축소)"}
    cols = st.columns(3)
    for col, name in zip(cols, order):
        with col:
            st.markdown(f"##### {titles[name]}")
            if not buckets[name]:
                st.caption("해당 없음")
            for i, a in enumerate(buckets[name]):
                render_signal(a, f"{name}{i}")
                st.markdown("<hr style='margin:6px 0;'>", unsafe_allow_html=True)
    if buckets["동결"]:
        with st.expander(f"게이트·실적 발표로 보류된 신호 {len(buckets['동결'])}건", expanded=False):
            for i, a in enumerate(buckets["동결"]):
                render_signal(a, f"frz{i}")
                st.markdown("<hr style='margin:6px 0;'>", unsafe_allow_html=True)
    if buckets["예외"]:
        st.caption("장기 보유 예외 (게이트 미적용, 보유 유지): " + ", ".join(a.get("ticker", "") for a in buckets["예외"]))
    if radar_res["no_signal"]:
        with st.expander(f"지금 신호 없는 종목 {len(radar_res['no_signal'])}개", expanded=False):
            st.dataframe(pd.DataFrame([
                {"종목": r["ticker"], "섹터": r.get("basket"), "구분": r.get("basket_role"),
                 "선정 이유": r.get("selection_note", ""),
                 "상태": r.get("state_label", ""),
                 "실적 발표": f"{r['next_earnings']} (D-{r['days_to_earnings']})" if r.get("days_to_earnings") is not None else "미확인",
                 "사유": r.get("reason", "")}
                for r in radar_res["no_signal"]
            ]), width="stretch", hide_index=True)


st.caption(f"대상: '지금 강한 곳'·'다음 후보' 섹터 ETF의 상위 구성 종목 중 추세(50일선 위)·3개월 상대강도 상위 {len(radar_universe)}개 · "
           "T(추세매매) 신호만 표시 — V(가치 분할매수)는 종목 딥다이브에서 확인하세요.")
if radar_res.get("selection_log"):
    with st.expander("종목 선정 과정 보기", expanded=False):
        st.dataframe(pd.DataFrame(radar_res["selection_log"]), width="stretch", hide_index=True)
        st.caption("후보 = 섹터 ETF 상위 구성 종목(미국·한국 상장만). 구성 데이터가 없으면 미리 정한 대표 종목으로 보충합니다.")
if radar_res.get("fatal"):
    st.warning(f"실행 레이더: {radar_res['fatal']}. 예시 신호로 대신 채우지 않습니다.")
else:
    render_radar()

# =====================================================================
# 6. 데이터 점검 (접이식)
# =====================================================================
violations = list(dash.get("consistency_violations", []))
for e_item in radar_res.get("errors", []):
    violations.append({"type": "DATA_MISSING", "detail": f"실행 레이더 {e_item.get('ticker')}: {e_item.get('reason')}"})
if not rb.get("available"):
    violations.append({"type": "DATA_MISSING", "detail": f"섹터 흐름(4분면) 데이터 없음: {rb.get('reason', '')}"})
for mk, g in (("US", us_gate), ("KR", kr_gate)):
    h_info = gate_hist.get(mk, {})
    h_list = h_info.get("history", [])
    if not h_list:
        violations.append({"type": "DATA_MISSING",
                           "detail": f"{mk} 게이트 이전 단계 재계산 불가({h_info.get('reason', '원인 미상')}) — 완화 지연 규칙 미적용"})
    elif h_list[-1]["level"] != g["level"]:
        violations.append({"type": "GATE_HISTORY_MISMATCH",
                           "detail": f"{mk} 게이트: 오늘 판정 G{g['level']} ≠ 과거 재계산 마지막 날 G{h_list[-1]['level']}"})
# 예시 데이터 사용은 '알고 있는 미연결 항목'이라 진짜 문제(데이터 없음·오래됨·불일치)와 따로 센다.
V_TYPE_NAME = {
    "DATA_MISSING": "데이터 없음", "STALE_DATA": "오래된 데이터", "MISSING_AS_OF": "기준일 없음",
    "DIRECTION_MISMATCH": "방향 불일치", "BADGE_STATE_MISMATCH": "배지-판정 불일치",
    "CRYPTO_EPS_BADGE": "크립토 EPS 배지", "GATE_HISTORY_MISMATCH": "게이트 기록 불일치",
    "SAMPLE_DATA_USED": "예시 데이터 사용",
}
problems = [v for v in violations if v.get("type") != "SAMPLE_DATA_USED"]
samples = [v for v in violations if v.get("type") == "SAMPLE_DATA_USED"]
if problems:
    v_title = f"데이터 점검: 확인 필요 {len(problems)}건 · 예시 데이터 {len(samples)}건"
elif samples:
    v_title = f"데이터 점검: 문제 없음 · 예시 데이터 {len(samples)}건"
else:
    v_title = "데이터 점검: 이상 없음"
with st.expander(v_title, expanded=False):
    if not violations:
        st.success("이상 없음 — 방향성, 기준일, 신선도, 배지-상태 정합성이 모두 통과되었습니다.")
    else:
        summary = pd.DataFrame(violations).assign(
            종류=lambda d: d["type"].map(V_TYPE_NAME).fillna(d["type"])
        ).groupby("종류", sort=False).size().reset_index(name="건수")
        st.caption("종류별 건수입니다. '예시 데이터 사용'은 실제 출처가 아직 연결되지 않은 항목이라 오류가 아닙니다.")
        st.dataframe(summary, width="stretch", hide_index=True)
        if problems:
            st.markdown("**확인 필요 항목**")
            st.dataframe(pd.DataFrame(problems), width="stretch", hide_index=True)
        if st.toggle("예시 데이터 항목 전체 보기", value=False, key="show_sample_violations"):
            st.dataframe(pd.DataFrame(samples), width="stretch", hide_index=True)
