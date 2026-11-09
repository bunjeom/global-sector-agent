# C:\Users\hanju\global-sector-agent\utils.py
import os
import json
import streamlit as st

def load_latest_report(filepath: str = "latest_global_report.json") -> dict:
    """최신 글로벌 리포트 JSON 로드 (없을 시 기본 구조 반환)"""
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"macro": {}, "news": [], "stocks": []}

def line_icon(name: str, size: int = 16) -> str:
    """블랙 라인아웃(Black Line-art SVG) 아이콘 반환 함수"""
    icons = {
        "target": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="#111111" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle;margin-right:6px;"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/></svg>',
        "shield": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="#111111" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle;margin-right:6px;"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>',
        "layers": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="#111111" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle;margin-right:6px;"><polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/></svg>',
        "activity": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="#111111" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle;margin-right:6px;"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>',
        "check": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="#111111" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle;margin-right:4px;"><polyline points="20 6 9 17 4 12"/></svg>',
        "alert": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="#111111" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle;margin-right:6px;"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>',
        "sliders": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="#111111" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle;margin-right:6px;"><line x1="4" y1="21" x2="4" y2="14"/><line x1="4" y1="10" x2="4" y2="3"/><line x1="12" y1="21" x2="12" y2="12"/><line x1="12" y1="8" x2="12" y2="3"/><line x1="20" y1="21" x2="20" y2="16"/><line x1="20" y1="12" x2="20" y2="3"/><line x1="1" y1="14" x2="7" y2="14"/><line x1="9" y1="8" x2="15" y2="8"/><line x1="17" y1="16" x2="23" y2="16"/></svg>',
        "tag": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="#111111" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle;margin-right:6px;"><path d="M20.59 13.41l-7.17 7.17a2 2 0 0 1-2.83 0L2 12V2h10l8.59 8.59a2 2 0 0 1 0 2.82z"/><line x1="7" y1="7" x2="7.01" y2="7"/></svg>',
        "compass": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="#111111" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle;margin-right:6px;"><circle cx="12" cy="12" r="10"/><polygon points="16.24 7.76 14.12 14.12 7.76 16.24 9.88 9.88 16.24 7.76"/></svg>',
    }
    return icons.get(name, icons["activity"])

def inject_terminal_css():
    """터미널 전용 CSS 토큰 및 블랙 라인아웃 스타일 주입"""
    st.markdown("""
    <style>
    :root {
        --bg-card: #ffffff;
        --bg-subtle: #f8f9fa;
        --border-color: #e2e8f0;
        --border-strong: #111111;
        --text-main: #111111;
        --text-muted: #555555;
        --font-mono: 'JetBrains Mono', 'Pretendard', monospace;
    }
    .term-banner {
        border: 1.5px solid var(--border-strong);
        border-radius: 6px;
        padding: 16px 20px;
        background: var(--bg-card);
        margin-bottom: 16px;
    }
    .term-banner-line1 {
        font-size: 1.15rem;
        font-weight: 700;
        color: var(--text-main);
        margin-bottom: 6px;
    }
    .term-banner-line2 {
        font-size: 0.95rem;
        font-family: var(--font-mono);
        color: var(--text-main);
        background: var(--bg-subtle);
        padding: 8px 12px;
        border-left: 3px solid var(--border-strong);
        margin-bottom: 8px;
    }
    .term-banner-line3 {
        font-size: 0.92rem;
        color: var(--text-muted);
        border-top: 1px dashed var(--border-color);
        padding-top: 8px;
    }
    .term-card {
        border: 1px solid var(--border-color);
        border-radius: 6px;
        padding: 14px 16px;
        background: var(--bg-card);
        margin-bottom: 12px;
    }
    .term-badge {
        display: inline-block;
        border: 1px solid #111111;
        border-radius: 4px;
        padding: 2px 8px;
        font-size: 0.8rem;
        font-weight: 600;
        color: #111111;
        background: #ffffff;
        margin-right: 6px;
    }
    .term-badge-inv {
        display: inline-block;
        border: 1px solid #111111;
        border-radius: 4px;
        padding: 2px 8px;
        font-size: 0.8rem;
        font-weight: 700;
        color: #ffffff;
        background: #111111;
        margin-right: 6px;
    }
    .state-node {
        display: inline-block;
        padding: 6px 12px;
        border: 1px solid #cbd5e1;
        border-radius: 4px;
        margin-right: 6px;
        font-size: 0.85rem;
        color: #64748b;
    }
    .state-node-active {
        display: inline-block;
        padding: 6px 12px;
        border: 2px solid #111111;
        border-radius: 4px;
        margin-right: 6px;
        font-size: 0.88rem;
        font-weight: 700;
        color: #111111;
        background: #f1f5f9;
    }
    .term-list {
        list-style: none;
        padding-left: 0;
        margin: 0;
    }
    .term-list li {
        padding: 6px 0;
        border-bottom: 1px solid #f1f5f9;
        font-size: 0.9rem;
    }
    </style>
    """, unsafe_allow_html=True)

def init_session():
    """세션 초기화 및 사이드바 네비게이션 렌더링 (실험실/백테스터 제거 완료)"""
    inject_terminal_css()
    if "total_equity" not in st.session_state:
        st.session_state["total_equity"] = 100_000_000  # 기본 총자산 1억 원
    if "t_sleeve_ratio" not in st.session_state:
        st.session_state["t_sleeve_ratio"] = 0.30       # 기본 T 슬리브 30%

    with st.sidebar:
        st.markdown(f'### {line_icon("compass", 20)} 투자터미널', unsafe_allow_html=True)
        st.caption("국면 · 흐름 · 실행")
        st.page_link("app.py", label="홈", icon="📊")
        st.page_link("pages/1_🔎_종목_딥다이브.py", label="종목 딥다이브", icon="📈")
        st.divider()
        st.markdown(f'**{line_icon("sliders", 15)} 계좌 설정**', unsafe_allow_html=True)
        st.session_state["total_equity"] = st.number_input(
            "총자산 (원)", min_value=10_000_000, max_value=10_000_000_000,
            value=int(st.session_state["total_equity"]), step=10_000_000
        )
        st.session_state["t_sleeve_ratio"] = st.slider(
            "추세매매(T) 비중", min_value=0.10, max_value=1.00,
            value=float(st.session_state["t_sleeve_ratio"]), step=0.05
        )