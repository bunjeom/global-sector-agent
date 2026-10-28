import os
import json
import pandas as pd
import streamlit as st

@st.cache_data(ttl=300)
def load_report():
    """latest_report.json 파일을 읽어 리포트 딕셔너리와 전체 종목 DataFrame을 반환합니다."""
    report_path = "latest_report.json"
    if not os.path.exists(report_path):
        return {}, pd.DataFrame()
    try:
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)
    except Exception:
        return {}, pd.DataFrame()

    stocks = report.get("top_picks", report.get("stocks", []))
    df_all = pd.DataFrame(stocks) if stocks else pd.DataFrame()
    return report, df_all


# 블랙 라인아웃(Black Line-art) SVG 아이콘 패스 사전 (Feather/Lucide 표준 24x24 규격)
LINE_ICON_PATHS = {
    "home": '<path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"></path><polyline points="9 22 9 12 15 12 15 22"></polyline>',
    "search": '<circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line>',
    "zap": '<polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon>',
    "news": '<path d="M4 22h16a2 2 0 0 0 2-2V4a2 2 0 0 0-2-2H8a2 2 0 0 0-2 2v16a2 2 0 0 1-2 2Zm0 0a2 2 0 0 1-2-2v-9c0-1.1.9-2 2-2h2"></path><path d="M18 14h-8"></path><path d="M15 18h-5"></path><path d="M10 6h8v4h-8V6Z"></path>',
    "target": '<circle cx="12" cy="12" r="10"></circle><circle cx="12" cy="12" r="6"></circle><circle cx="12" cy="12" r="2"></circle>',
    "globe": '<circle cx="12" cy="12" r="10"></circle><line x1="2" y1="12" x2="22" y2="12"></line><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"></path>',
    "compass": '<circle cx="12" cy="12" r="10"></circle><polygon points="16.24 7.76 14.12 14.12 7.76 16.24 9.88 9.88 16.24 7.76"></polygon>',
    "layers": '<polygon points="12 2 2 7 12 12 22 7 12 2"></polygon><polyline points="2 17 12 22 22 17"></polyline><polyline points="2 12 12 17 22 12"></polyline>',
    "diamond": '<path d="M6 3h12l4 6-10 13L2 9Z"></path><path d="M11 3 8 9l4 13 4-13-3-6"></path><path d="M2 9h20"></path>',
    "sliders": '<line x1="4" y1="21" x2="4" y2="14"></line><line x1="4" y1="10" x2="4" y2="3"></line><line x1="12" y1="21" x2="12" y2="12"></line><line x1="12" y1="8" x2="12" y2="3"></line><line x1="20" y1="21" x2="20" y2="16"></line><line x1="20" y1="12" x2="20" y2="3"></line><line x1="1" y1="14" x2="7" y2="14"></line><line x1="9" y1="8" x2="15" y2="8"></line><line x1="17" y1="16" x2="23" y2="16"></line>',
    "bookmark": '<path d="m19 21-7-4-7 4V5a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2v16z"></path>',
    "flame": '<path d="M8.5 14.5A2.5 2.5 0 0 0 11 12c0-1.38-.5-2-1-3-1.072-2.143-.224-4.054 2-6 .5 2.5 2 4.9 4 6.5 2 1.6 3 3.5 3 5.5a7 7 0 1 1-14 0c0-1.153.433-2.294 1-3a2.5 2.5 0 0 0 2.5 2.5z"></path>'
}


def line_icon(name: str, size: int = 20) -> str:
    """지정한 이름의 블랙 라인아웃 SVG 아이콘 HTML을 반환합니다."""
    inner = LINE_ICON_PATHS.get(name, LINE_ICON_PATHS["target"])
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 24 24" fill="none" stroke="#0F172A" stroke-width="2" '
        f'stroke-linecap="round" stroke-linejoin="round" '
        f'style="display:inline-block; vertical-align:-0.18em; margin-right:0.38rem;">{inner}</svg>'
    )


def init_session():
    # absolute 좌표를 쓰지 않고 span 내부 텍스트만 교체하여 메뉴 겹침을 원천 차단
    st.markdown("""
    <style>
    [data-testid="stSidebarNavItems"] li:nth-child(1) a span,
    [data-testid="stSidebarNavItems"] li:nth-child(2) a span,
    [data-testid="stSidebarNavItems"] li:nth-child(3) a span,
    [data-testid="stSidebarNavItems"] li:nth-child(5) a span {
        font-size: 0 !important;
        line-height: 0 !important;
    }
    [data-testid="stSidebarNavItems"] li:nth-child(1) a span::after,
    [data-testid="stSidebarNavItems"] li:nth-child(2) a span::after,
    [data-testid="stSidebarNavItems"] li:nth-child(3) a span::after,
    [data-testid="stSidebarNavItems"] li:nth-child(5) a span::after {
        font-size: 0.92rem !important;
        line-height: 1.5 !important;
        color: #0F172A !important;
        font-weight: 500 !important;
        visibility: visible !important;
    }
    [data-testid="stSidebarNavItems"] li:nth-child(1) a span::after {
        content: "⌂  홈 · 오늘의 소식";
        font-weight: 700 !important;
    }
    [data-testid="stSidebarNavItems"] li:nth-child(2) a span::after {
        content: "⌕  종목 딥다이브";
    }
    [data-testid="stSidebarNavItems"] li:nth-child(3) a span::after {
        content: "⚞  커스텀 팩터 실험실";
    }
    [data-testid="stSidebarNavItems"] li:nth-child(5) a span::after {
        content: "⊞  섹터 히트맵";
    }
    </style>
    """, unsafe_allow_html=True)

    if "selected_ticker" not in st.session_state:
        st.session_state["selected_ticker"] = "005930"
    if "selected_name" not in st.session_state:
        st.session_state["selected_name"] = "삼성전자"