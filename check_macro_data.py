"""거시 트랙 실데이터 점검: FRED·ECOS에서 변수별로 받아지는지, 무위험수익률이 무엇인지 출력한다.
실행: .venv\\Scripts\\python check_macro_data.py   (종목을 주면 그 종목 가설 변수 기준: ... check_macro_data.py 005930.KS)
"""
import sys
from datetime import datetime, timedelta
import engine

start = (datetime.now() - timedelta(days=900)).strftime("%Y-%m-%d")
print("FRED_API_KEY:", "있음" if engine._read_env_value("FRED_API_KEY") else "없음 (공개 CSV로 시도)",
      "| ECOS_API_KEY:", "있음" if engine._read_env_value("ECOS_API_KEY") else "없음 (국고채 10년 '데이터 없음')")
for krx in (False, True):
    r = engine.fetch_risk_free_10y(krx)
    v = f"{r['value']*100:.2f}%" if r["value"] is not None else engine.NO_DATA
    print(f"무위험수익률 {'한국' if krx else '미국'}: {v} · {r['source']} · {r['as_of']}")

ids = list(engine.MACRO_SERIES_SPEC)
if len(sys.argv) > 1:
    t = sys.argv[1]
    _, hv = engine._ensure_macro_hypothesis_file(t, "메모리 (한국 상장)" if t.upper().endswith((".KS", ".KQ")) else "AI 반도체·네트워킹(팹리스)")
    ids = list(dict.fromkeys([v["var_id"] for v in hv] + ["T10YIE", "DFII10"]))
res = engine.fetch_macro_levels(ids, start)
print(f"\n거시 변수 {len(ids)}개 중 받음 {len(res['levels'])} · 못 받음 {len(res['missing'])}")
for vid in ids:
    spec = engine.MACRO_SERIES_SPEC.get(vid, {})
    if vid in res["levels"]:
        s = res["levels"][vid] * spec.get("level_mult", 1.0)
        _d = res.get("last_obs", {}).get(vid, str(s.index[-1].date()))
        print(f"  [O] {vid:13s} {spec.get('name',''):14s} 최근 {s.iloc[-1]:,.2f}{spec.get('level_unit','')} (관측일 {_d}) · {len(s)}주 · {res['sources'][vid]}")
    else:
        print(f"  [X] {vid:13s} {spec.get('name','')} — {engine.NO_DATA}")
