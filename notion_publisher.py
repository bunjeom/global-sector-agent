import os
import json
import datetime
import httpx

NOTION_TOKEN = os.getenv("NOTION_TOKEN", "").strip()
NOTION_DATABASE_ID = os.getenv("NOTION_DATABASE_ID", "").strip().replace("-", "")

NOTION_HEADERS = {
    "Authorization": f"Bearer {NOTION_TOKEN}",
    "Content-Type": "application/json",
    "Notion-Version": "2022-06-28"
}

def resolve_or_create_database(target_id: str) -> tuple[str, str]:
    res = httpx.get(f"https://api.notion.com/v1/databases/{target_id}", headers=NOTION_HEADERS, timeout=10.0)
    if res.status_code == 200:
        props = res.json().get("properties", {})
        for prop_name, prop_info in props.items():
            if prop_info.get("type") == "title":
                return target_id, prop_name
        return target_id, "Name"

    if "is a page, not a database" in res.text:
        children_res = httpx.get(f"https://api.notion.com/v1/blocks/{target_id}/children", headers=NOTION_HEADERS, timeout=10.0)
        if children_res.status_code == 200:
            for block in children_res.json().get("results", []):
                if block.get("type") == "child_database":
                    found_db_id = block["id"].replace("-", "")
                    return resolve_or_create_database(found_db_id)

        create_db_payload = {
            "parent": {"type": "page_id", "page_id": target_id},
            "title": [{"type": "text", "text": {"content": "📊 글로벌(KOSPI·NASDAQ) AI 섹터·종목 리포트 DB"}}],
            "properties": {"리포트 제목": {"title": {}}}
        }
        create_res = httpx.post("https://api.notion.com/v1/databases", headers=NOTION_HEADERS, json=create_db_payload, timeout=15.0)
        if create_res.status_code == 200:
            return create_res.json()["id"].replace("-", ""), "리포트 제목"
    raise RuntimeError(f"노션 DB 조회 실패: {res.text}")

def make_table_row(cells: list) -> dict:
    return {
        "object": "block",
        "type": "table_row",
        "table_row": {
            "cells": [[{"type": "text", "text": {"content": str(c)[:180]}}] for c in cells]
        }
    }

def publish_to_notion(report_path: str = "latest_global_report.json") -> str:
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    real_db_id, title_prop = resolve_or_create_database(NOTION_DATABASE_ID)

    th = report.get("time_horizon", {})
    regime = report.get("macro_regime", {})
    exec_conc = report.get("executive_conclusion", {})
    t0 = th.get("anchor_date_t0", datetime.date.today().strftime("%Y-%m-%d"))

    k_hot = report["kospi_summary"]["hot_sector"]
    n_hot = report["nasdaq_summary"]["hot_sector"]
    page_title = f"🎯 [v2.0 | {t0}] Top-Down 글로벌 포트폴리오 결론 (KOSPI: {k_hot} / NASDAQ: {n_hot})"

    macro_str = " | ".join([f"{m['지표']}: {m['표기']}" for m in report.get("macros", [])])
    qa_str = " / ".join([f"{q['market']}: {q['status']} (오차 {q['diff_pct']}%%)" for q in report.get("qa_gates", [])])

    # 12종목 테이블 행 추가
    table_rows = [
        make_table_row(["시장", "구분(섹터)", "종목(티커)", "1M수익률", "ROE / PBR", "최종점수", "투자 전략 및 핵심 결론", "검증 원문(Quote)"])
    ]
    for s in report.get("top12_stocks", []):
        table_rows.append(make_table_row([
            s["market"],
            s["role"],
            f"{s['name']} ({s['ticker']})",
            f"{s['ret_1m']}%%",
            f"ROE {s['roe']}%% / PBR {s['pbr']}",
            f"{s['final_score']}점",
            s.get("catalyst", ""),
            s.get("quote", "")[:90]
        ]))

    children_blocks = [
        {
            "object": "block",
            "type": "callout",
            "callout": {
                "icon": {"type": "emoji", "emoji": "🛡️"},
                "rich_text": [{"type": "text", "text": {"content": f"데이터 품질 게이트(QA): {qa_str}\n거시경제 스냅샷(Point-in-Time): {macro_str}"}}]
            }
        },
        {
            "object": "block",
            "type": "heading_2",
            "heading_2": {"rich_text": [{"type": "text", "text": {"content": "1. Executive Conclusion (두괄식 투자 요약)"}}]}
        },
        {
            "object": "block",
            "type": "paragraph",
            "paragraph": {
                "rich_text": [
                    {
                        "type": "text",
                        "text": {"content": exec_conc.get("headline", "")},
                        "annotations": {"bold": True}  # 🛠️ annotations 위치 교정 완료
                    }
                ]
            }
        },
        {
            "object": "block",
            "type": "bulleted_list_item",
            "bulleted_list_item": {"rich_text": [{"type": "text", "text": {"content": exec_conc.get("kospi_action", "")}}]}
        },
        {
            "object": "block",
            "type": "bulleted_list_item",
            "bulleted_list_item": {"rich_text": [{"type": "text", "text": {"content": exec_conc.get("nasdaq_action", "")}}]}
        },
        {
            "object": "block",
            "type": "heading_2",
            "heading_2": {"rich_text": [{"type": "text", "text": {"content": "2. 최종 선별 12종목 포트폴리오 및 세부 요약"}}]}
        },
        {
            "object": "block",
            "type": "table",
            "table": {
                "table_width": 8,
                "has_column_header": True,
                "has_row_header": False,
                "children": table_rows
            }
        }
    ]

    payload = {
        "parent": {"database_id": real_db_id},
        "properties": {
            title_prop: {
                "title": [{"type": "text", "text": {"content": page_title}}]
            }
        },
        "children": children_blocks
    }

    res = httpx.post("https://api.notion.com/v1/pages", headers=NOTION_HEADERS, json=payload, timeout=15.0)
    if res.status_code != 200:
        raise RuntimeError(f"노션 페이지 생성 실패 (HTTP {res.status_code}): {res.text}")

    page_url = res.json().get("url", "")
    print(f"🎉 [노션 v2.0 발행 완료!] 생성된 리포트 URL: {page_url}")
    return page_url

if __name__ == "__main__":
    publish_to_notion()
