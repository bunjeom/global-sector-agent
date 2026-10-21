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
    if "selected_ticker" not in st.session_state:
        st.session_state["selected_ticker"] = "005930"
    if "selected_name" not in st.session_state:
        st.session_state["selected_name"] = "삼성전자"
