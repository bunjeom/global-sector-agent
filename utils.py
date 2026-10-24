import json
import os
import pandas as pd
import streamlit as st

def load_report():
    report_path = "latest_global_report.json"
    if not os.path.exists(report_path):
        st.error("latest_global_report.json 파일이 없습니다. global_pipeline.py를 먼저 실행해 주세요.")
        st.stop()
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)
    df_all = pd.DataFrame(report.get("top12_stocks", []))
    return report, df_all

def init_session():
    # 왼쪽 사이드바 첫 번째 메뉴('app')를 '🏠 홈'으로 변경하는 공통 CSS
    st.markdown("""
    <style>
    [data-testid="stSidebarNavItems"] li:first-child a span {
        visibility: hidden;
        position: relative;
    }
    [data-testid="stSidebarNavItems"] li:first-child a span::after {
        content: "🏠 홈";
        visibility: visible;
        position: absolute;
        left: 0;
        top: 0;
        font-weight: 700;
    }
    </style>
    """, unsafe_allow_html=True)

    if "selected_ticker" not in st.session_state:
        st.session_state["selected_ticker"] = "005930"
    if "selected_name" not in st.session_state:
        st.session_state["selected_name"] = "삼성전자"
