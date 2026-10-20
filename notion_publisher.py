import os
import json
import datetime
import httpx
from google import genai

NOTION_TOKEN = os.getenv("NOTION_TOKEN", "").strip()
NOTION_DATABASE_ID = os.getenv("NOTION_DATABASE_ID", "").strip().replace("-", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

NOTION_HEADERS = {
    "Authorization": f"Bearer {NOTION_TOKEN}",
    "Content-Type": "application/json",
    "Notion-Version": "2022-06-28"
}

def enrich_catalysts_batch(report: dict) -> dict:
    """Gemini 1회 배치 호출로 12종목 핵심 촉매 요약 업데이트"""
    if not GEMINI_API_KEY:
        return report
    stocks = report.get("top12_stocks", [])
    needs_update = any("모멘텀 및 펀더멘털 확인" in s.get("catalyst", "") for s in stocks)
    if not needs_update:
        return report

    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        items_txt = "\n".join([
            f"- {s['ticker']} ({s['name']}): 원문뉴스/공시='{s.get('quote', '')}' | 1개월수익률={s['ret_1m']}% | ROE={s['roe']}% | PBR={s['pbr']}"
            for s in stocks
        ])
        prompt = (
            "아래 12개 글로벌 종목 각각에 대해 제공된 '원문뉴스/공시'와 재무 지표를 바탕으로 핵심 투자 촉매(또는 저평가 매력)를 "
            "한국어 1문장(45자 이내)으로 구체적으로 요약하세요.\n"
            f"{items_txt}\n\n"
            '반드시 티커를 키(Key)로 하는 순수 JSON 객체로만 응답하세요. 예: {"000100": "요약문", "NVDA": "요약문"}'
        )
        resp = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        raw = resp.text.strip()
        if "```" in raw:
            raw = raw.split("```")[1].removeprefix("json").strip()
        cat_map = json.loads(raw)
        for s in stocks:
            tkr = s["ticker"]
            if tkr in cat_map and len(str(cat_map[tkr])) > 5:
                s["catalyst"] = str(cat_map[tkr])
        with open("latest_global_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print("✨ [Gemini 배치 보강] 12종목 맞춤형 촉매 한 줄 요약 업데이트 완료!")
    except Exception as e:
        print(f"ℹ️ 촉매 배치 보강 생략 (기본 검증 문구 유지): {e}")
    return report

def resolve_or_create_database(target_id: str) -> tuple[str, str]:
    """
    입력된 ID가 데이터베이스인지 일반 페이지인지 자동 판별합니다.
    만약 '일반 페이지'라면 페이지 내부의 기존 DB를 찾거나 새 DB를 자동으로 생성해 줍니다!
    반환값: (실제_데이터베이스_ID, 제목_컬럼명)
    """
    # 1) 먼저 데이터베이스로 조회 시도
    res = httpx.get(f"https://api.notion.com/v1/databases/{target_id}", headers=NOTION_HEADERS, timeout=10.0)
    if res.status_code == 200:
        props = res.json().get("properties", {})
        for prop_name, prop_info in props.items():
            if prop_info.get("type") == "title":
                return target_id, prop_name
        return target_id, "Name"

    # 2) 만약 'is a page, not a database' 에러라면 -> 일반 페이지로 처리!
    if "is a page, not a database" in res.text:
        print("💡 [자동 보정] 입력된 ID가 '일반 페이지'임을 감지했습니다. 페이지 내부 데이터베이스를 확인합니다...")
        
        # 2-1) 해당 페이지 안에 이미 만들어진 자식 데이터베이스(child_database)가 있는지 검색
        children_res = httpx.get(f"https://api.notion.com/v1/blocks/{target_id}/children", headers=NOTION_HEADERS, timeout=10.0)
        if children_res.status_code == 200:
            for block in children_res.json().get("results", []):
                if block.get("type") == "child_database":
                    found_db_id = block["id"].replace("-", "")
                    print(f"✅ 페이지 내부에서 기존 데이터베이스를 찾았습니다! (DB ID: {found_db_id})")
                    return resolve_or_create_database(found_db_id)

        # 2-2) 자식 DB가 없다면 해당 페이지 안에 리포트 전용 데이터베이스를 새로 자동 생성!
        print("🛠️ 페이지 내부에 리포트 전용 데이터베이스를 새로 자동 생성합니다...")
        create_db_payload = {
            "parent": {"type": "page_id", "page_id": target_id},
            "title": [{"type": "text", "text": {"content": "📊 글로벌(KOSPI·NASDAQ) AI 섹터·종목 리포트 DB"}}],
            "properties": {
                "리포트 제목": {"title": {}},
                "생성일": {"rich_text": {}},
                "KOSPI 주목/소외": {"rich_text": {}},
                "NASDAQ 주목/소외": {"rich_text": {}}
            }
        }
        create_res = httpx.post("https://api.notion.com/v1/databases", headers=NOTION_HEADERS, json=create_db_payload, timeout=15.0)
        if create_res.status_code == 200:
            new_db_id = create_res.json()["id"].replace("-", "")
            print("=" * 65)
            print(f"🎉 [자동 DB 생성 완료!] 새로운 데이터베이스 ID: {new_db_id}")
            print("💡 (참고: 지금 바로 리포트가 발행되며, 나중에 여유 있을 때 Colab/GitHub의 NOTION_DATABASE_ID를 위 ID로 바꿔두셔도 좋고 그대로 두셔도 자동 인식됩니다!)")
            print("=" * 65)
            return new_db_id, "리포트 제목"
        else:
            raise RuntimeError(f"데이터베이스 자동 생성 실패: {create_res.text}")

    raise RuntimeError(f"노션 DB 조회 실패 (HTTP {res.status_code}): {res.text}")

def make_table_row(cells: list) -> dict:
    return {
        "object": "block",
        "type": "table_row",
        "table_row": {
            "cells": [[{"type": "text", "text": {"content": str(c)[:180]}}] for c in cells]
        }
    }

def publish_to_notion(report_path: str = "latest_global_report.json") -> str:
    if not NOTION_TOKEN or not NOTION_DATABASE_ID:
        raise ValueError("NOTION_TOKEN 또는 NOTION_DATABASE_ID 환경변수가 비어 있습니다.")

    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    report = enrich_catalysts_batch(report)

    # 페이지 ID가 들어와도 알아서 DB ID로 변환 또는 자동 생성!
    real_db_id, title_prop = resolve_or_create_database(NOTION_DATABASE_ID)

    gen_dt = report.get("generated_at", datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
    k_hot = report["kospi_summary"]["hot_sector"]
    k_cold = report["kospi_summary"]["cold_sector"]
    n_hot = report["nasdaq_summary"]["hot_sector"]
    n_cold = report["nasdaq_summary"]["cold_sector"]

    page_title = f"🌍 [{gen_dt[:10]}] 글로벌 섹터·종목 리포트 (KOSPI: {k_hot} / NASDAQ: {n_hot})"

    macro_str = " | ".join([f"{m['지표']}: {m['표기']}" for m in report.get("macros", [])])
    qa_str = " / ".join([f"{q['market']}: {q['status']} (오차 {q['diff_pct']}%)" for q in report.get("qa_gates", [])])

    table_rows = [
        make_table_row(["시장", "구분(섹터)", "종목(티커)", "1M수익률", "ROE / PBR", "최종점수", "핵심 촉매 요약", "검증 원문(Quote)"])
    ]
    for s in report.get("top12_stocks", []):
        table_rows.append(make_table_row([
            s["market"],
            s["role"],
            f"{s['name']} ({s['ticker']})",
            f"{s['ret_1m']}%",
            f"ROE {s['roe']}% / PBR {s['pbr']}",
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
            "heading_2": {"rich_text": [{"type": "text", "text": {"content": "1. 시장별 정량 섹터 선정 결과"}}]}
        },
        {
            "object": "block",
            "type": "bulleted_list_item",
            "bulleted_list_item": {"rich_text": [{"type": "text", "text": {"content": f"🇰🇷 KOSPI -> 🔥 주목 1위: {k_hot} | ❄️ 소외 1위(구조적 필터 통과): {k_cold}"}}]}
        },
        {
            "object": "block",
            "type": "bulleted_list_item",
            "bulleted_list_item": {"rich_text": [{"type": "text", "text": {"content": f"🇺🇸 NASDAQ -> 🔥 주목 1위: {n_hot} | ❄️ 소외 1위(구조적 필터 통과): {n_cold}"}}]}
        },
        {
            "object": "block",
            "type": "heading_2",
            "heading_2": {"rich_text": [{"type": "text", "text": {"content": "2. 최종 선별 상위 12종목 (정량 스코어 + 자율 루프 촉매)"}}]}
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
    print(f"🎉 [노션 자동 발행 성공!] 생성된 리포트 URL: {page_url}")
    return page_url

if __name__ == "__main__":
    publish_to_notion()
