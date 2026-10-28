import xml.etree.ElementTree as ET
import urllib.request
import pandas as pd
import streamlit as st
from utils import load_report, init_session, line_icon

st.set_page_config(page_title="오늘의 소식 | 글로벌 퀀트 터미널", page_icon="⌂", layout="wide")
init_session()
report, df_all = load_report()

st.markdown("""
<style>
div[data-testid="stMetricValue"] {
    font-size: 1.5rem !important;
    font-weight: 700 !important;
}
.sec-header {
    font-size: 1.35rem;
    font-weight: 700;
    color: #0F172A;
    margin: 0.4rem 0 0.25rem 0;
    display: flex;
    align-items: center;
}
.main-header {
    font-size: 1.85rem;
    font-weight: 800;
    color: #0F172A;
    margin: 0 0 0.75rem 0;
    display: flex;
    align-items: center;
}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# 1. 상단 타이틀(블랙 라인 홈 아이콘) & 거시 국면 브리핑
# ------------------------------------------------------------------------------
st.markdown(f"<div class='main-header'>{line_icon('home', 26)}오늘의 소식</div>", unsafe_allow_html=True)

macro = report.get("macro", {})
regime_desc = report.get("regime_desc", "데이터 로드 중")
exec_summary = report.get("executive_summary", "")
date_str = report.get("date", "-")
next_reb = report.get("next_rebalance_date", "-")

# 리포트 원문 텍스트에 섞여 있을 수 있는 🔥, ❄️ 등 장식 이모지를 블랙 라인 기호로 정제
clean_regime_desc = str(regime_desc).replace("🔥", "▲").replace("❄️", "▽")

st.info(
    f"**[오늘의 핵심 결론]** {exec_summary}\n\n"
    f"- **거시 국면 진단:** {clean_regime_desc}\n"
    f"- **분석 시계열:** 기준일 `{date_str}` | 목표 보유 `향후 20영업일 (1개월 스윙/포지션, 차기 리밸런싱 예정일: {next_reb})`"
)

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("한국 기준금리", f"{macro.get('kr_base_rate', 2.75)}%", f"{macro.get('kr_rate_source', 'ECOS')}", delta_color="off")
m2.metric("미국채 10년물", f"{macro.get('us_10y_yield', 4.20)}%", f"{macro.get('us_rate_source', 'FRED')}", delta_color="off")
m3.metric("원/달러 환율", f"{macro.get('usd_krw', 1380.0):,.2f}원", f"{macro.get('fx_source', 'FDR')}", delta_color="off")
m4.metric("KOSPI 지수", f"{macro.get('kospi_close', 2600.0):,.1f}", f"{macro.get('kospi_ret20', 0.0):+.2f}% (20D)")
m5.metric("NASDAQ 지수", f"{macro.get('nasdaq_close', 18000.0):,.2f}", f"{macro.get('nasdaq_ret20', 0.0):+.2f}% (20D)")

st.divider()

# ------------------------------------------------------------------------------
# 2. 오늘 가장 영향력 있는 핵심 뉴스 & 공시 촉매 Top 5 (블랙 라인 아이콘 적용)
# ------------------------------------------------------------------------------
@st.cache_data(ttl=1800, show_spinner=False)
def get_top5_impact_news(df_stocks: pd.DataFrame):
    news_items = []

    rss_url = "https://news.google.com/rss/search?q=%EC%A6%9D%EC%8B%9C+OR+%EA%B8%88%EB%A6%AC+OR+%EB%B0%98%EB%8F%84%EC%B2%B4+OR+%EC%8B%A4%EC%A0%81&hl=ko&gl=KR&ceid=KR:ko"
    try:
        req = urllib.request.Request(rss_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=3) as resp:
            xml_data = resp.read()
        root = ET.fromstring(xml_data)
        for item in root.findall("./channel/item")[:3]:
            title = item.findtext("title", "").split(" - ")[0].strip()
            link = item.findtext("link", "")
            pub_date = item.findtext("pubDate", "")[:16]
            if title:
                news_items.append({
                    "icon": "globe",
                    "badge": "거시·시장 헤드라인",
                    "target": "글로벌/국내 마켓",
                    "headline": title,
                    "detail": f"실시간 주요 경제·증시 핵심 이슈 ({pub_date})",
                    "url": link
                })
    except Exception:
        pass

    if df_stocks is not None and not df_stocks.empty:
        df_sorted = df_stocks.copy()
        for sort_col in ["score", "total_score", "final_score", "ret20", "roe"]:
            if sort_col in df_sorted.columns:
                df_sorted = df_sorted.sort_values(by=sort_col, ascending=False)
                break

        for _, row in df_sorted.iterrows():
            if len(news_items) >= 5:
                break
            cat_text = str(row.get("catalyst", row.get("reason", ""))).strip()
            quote_text = str(row.get("quote", "")).strip()
            if (not cat_text or cat_text in ["-", "nan", "None"]) and (not quote_text or quote_text in ["-", "nan", "None"]):
                continue

            headline_str = quote_text if (quote_text and quote_text not in ["-", "nan", "None"]) else cat_text
            ret20_val = float(row.get("ret20", 0.0)) if pd.notna(row.get("ret20")) else 0.0

            news_items.append({
                "icon": "zap",
                "badge": f"핵심 촉매 ({row.get('market', '')} · {row.get('sector', '')})",
                "target": f"{row.get('name', '')} ({row.get('ticker', '')})",
                "headline": headline_str,
                "detail": f"선정 사유: {cat_text} | 최근 20일 수익률 {ret20_val:+.1f}%",
                "url": str(row.get("url", ""))
            })

    return news_items[:5]

st.markdown(f"<div class='sec-header'>{line_icon('news', 22)}오늘 가장 영향력 있는 핵심 뉴스 & 공시 Top 5</div>", unsafe_allow_html=True)
st.caption("글로벌 거시 헤드라인 및 에이전트가 검증한 당일 최고 파급력 공시·실적·수주 재료 상위 5선입니다.")

top5_news = get_top5_impact_news(df_all)
if top5_news:
    with st.container(border=True):
        for idx, n_item in enumerate(top5_news, start=1):
            link_md = f" — [원문 보기 ↗]({n_item['url']})" if n_item.get("url") and n_item["url"].startswith("http") else ""
            item_icon = line_icon(n_item.get("icon", "zap"), 16)
            st.markdown(
                f"**{idx}.** {item_icon}**[{n_item['badge']}] {n_item['target']}** : **{n_item['headline']}**{link_md}  \n"
                f"<span style='color:#475569; font-size:0.88rem;'>└ {n_item['detail']}</span>",
                unsafe_allow_html=True
            )
else:
    st.info("현재 표시할 주요 뉴스 데이터가 없습니다.")

st.divider()

# ------------------------------------------------------------------------------
# 3. 오늘의 선별 12종목 (블랙 라인 타겟 아이콘 적용)
# ------------------------------------------------------------------------------
st.markdown(
    f"<div class='sec-header'>{line_icon('target', 22)}오늘의 선별 12종목 "
    f"<span style='font-size:0.95rem; font-weight:500; color:#475569; margin-left:0.5rem;'>(표의 행을 마우스로 클릭하면 상세 차트 페이지로 이동합니다)</span></div>",
    unsafe_allow_html=True
)

mkt_filter = st.radio("시장 필터 선택", ["전체 (All)", "KOSPI", "NASDAQ"], horizontal=True)
df_view = df_all.copy()
if mkt_filter != "전체 (All)" and not df_view.empty:
    df_view = df_view[df_view["market"] == mkt_filter].reset_index(drop=True)

if not df_view.empty:
    display_cols = ["market", "type", "sector", "ticker", "name", "close", "ret20", "roe", "pbr", "score", "total_score", "catalyst"]
    available_cols = [c for c in display_cols if c in df_view.columns]

    event = st.dataframe(
        df_view[available_cols],
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        column_config={
            "market": "시장",
            "type": "전략구분",
            "sector": "섹터",
            "ticker": "티커",
            "name": "종목명",
            "close": st.column_config.NumberColumn("현재가", format="%.2f"),
            "ret20": st.column_config.NumberColumn("20일 수익률(%)", format="%+.2f%%"),
            "roe": st.column_config.NumberColumn("ROE(%)", format="%.1f%%"),
            "pbr": st.column_config.NumberColumn("PBR(배)", format="%.2f"),
            "score": st.column_config.NumberColumn("종합점수", format="%.1f"),
            "total_score": st.column_config.NumberColumn("종합점수", format="%.1f"),
            "catalyst": "핵심 촉매 요약"
        }
    )

    if len(event.selection.rows) > 0:
        sel_idx = event.selection.rows[0]
        sel_row = df_view.iloc[sel_idx]
        st.session_state["selected_ticker"] = str(sel_row["ticker"])
        st.session_state["selected_name"] = str(sel_row["name"])
        st.switch_page("pages/1_🔎_종목_딥다이브.py")
else:
    st.warning("선별된 종목 데이터가 없습니다.")