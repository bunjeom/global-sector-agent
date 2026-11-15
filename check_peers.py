"""피어 비교 점검: 가장 최근 순위 기록(data/peers/)을 묶음별로 요약하고 data/peers/peer_census_날짜.md로 저장한다.

실행: .venv\\Scripts\\python check_peers.py          (먼저 peers_daily.py가 한 번 이상 실행되어 있어야 함)
      .venv\\Scripts\\python check_peers.py AMZN     (이 종목의 묶음만 자세히)

용도: 피어 묶음이 이상한지(엉뚱한 종목·너무 작음), 지표가 비어 있는지 같은 데이터·정의 오류 확인.
결과를 보고 가중치·기준을 바꾸지 않는다(AGENTS.md 4번 사전 등록). 이상한 피어는 data/peers/peer_map.csv로 고친다.
"""
import os
import sys

import pandas as pd

import engine


def _pct(s: pd.Series) -> str:
    return f"{int(s.notna().mean() * 100)}%" if len(s) else "-"


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    want = [a.upper() for a in sys.argv[1:] if not a.startswith("-")]
    rk = engine.load_peer_ranks()
    if not len(rk):
        print("피어 기록이 없습니다. 먼저 실행: .venv\\Scripts\\python peers_daily.py --full")
        sys.exit(1)
    d = rk["rank_date"].max()
    cur = rk[rk["rank_date"] == d].copy()
    if want:
        groups = set(cur.loc[cur["ticker"].astype(str).str.upper().isin(want), "group"])
        cur = cur[cur["group"].isin(groups)]
    metrics = [m[0] for m in engine.PEER_METRICS]
    lines = [f"# 피어 비교 점검 ({pd.Timestamp(d).date()}, {engine.PEER_CONFIG['VERSION']})", "",
             "check_peers.py 출력. 데이터·정의 오류 확인용이며 결과를 보고 가중치·기준을 바꾸지 않는다. 이상한 피어는 data/peers/peer_map.csv로 고친다.", "",
             f"- 묶음 {cur['group'].nunique()}개 · 종목 {cur['ticker'].nunique()}개 · 커버리지 종목 "
             f"{int(cur['is_coverage'].astype(str).str.lower().eq('true').sum())}개 · 재무 기준일 {pd.Timestamp(cur['fund_date'].max()).date()}",
             f"- 순위가 매겨진 묶음: 가치 관점 {cur.groupby('group')['value_rank'].apply(lambda s: s.notna().any()).sum()}개 · "
             f"추세 관점 {cur.groupby('group')['trend_rank'].apply(lambda s: s.notna().any()).sum()}개", "",
             "## 지표가 채워진 비율 (전체)", "", "| 지표 | 채워짐 |", "|---|---|"]
    lines += [f"| {lbl} | {_pct(cur[k])} |" for k, _, _, lbl in engine.PEER_METRICS]
    lines += ["", "## 묶음별 요약", "", "| 묶음 | 출처 | 종목 수 | 가치 1위 | 가치 꼴찌 | 추세 1위 | 순위 없음 사유 | 구성 종목 |", "|---|---|---|---|---|---|---|---|"]
    for gk, g in cur.groupby("group"):
        v = g.dropna(subset=["value_rank"]).sort_values("value_rank")
        t = g.dropna(subset=["trend_rank"]).sort_values("trend_rank")
        why = "" if len(v) else str(g["value_note"].dropna().iloc[0] if g["value_note"].notna().any() else "")
        members = ", ".join(f"{x}{'*' if str(c).lower() == 'true' else ''}" for x, c in zip(g["ticker"], g["is_coverage"]))
        lines.append(f"| {g['group_name'].iloc[0]} | {g['group_source'].iloc[0]} | {len(g)} | "
                     f"{v['ticker'].iloc[0] if len(v) else '-'} | {v['ticker'].iloc[-1] if len(v) else '-'} | "
                     f"{t['ticker'].iloc[0] if len(t) else '-'} | {why.replace('|', '/')} | {members} |")
    lines += ["", "(*는 커버리지 종목, 나머지는 야후 산업 상위 기업에서 온 자동 피어)"]
    if want:
        for gk, g in cur.groupby("group"):
            lines += ["", f"## {g['group_name'].iloc[0]} 상세", "",
                      "| 종목 | 가치 점수(순위) | 추세 점수(순위) | " + " | ".join(engine.FACTOR_LABELS.values()) + " | 메모 |",
                      "|" + "---|" * (4 + len(engine.FACTOR_LABELS))]
            for _, r in g.sort_values("value_score", ascending=False, na_position="last").iterrows():
                f = lambda x: "-" if pd.isna(x) else f"{float(x):.0f}"
                lines.append(f"| {r['ticker']} | {f(r['value_score'])} ({f(r['value_rank'])}) | {f(r['trend_score'])} ({f(r['trend_rank'])}) | "
                             + " | ".join(f(r[f'f_{k}']) for k in engine.FACTOR_LABELS) + f" | {'' if pd.isna(r['value_note']) else str(r['value_note']).replace('|', '/')} |")
    text = "\n".join(lines) + "\n"
    print(text)
    path = os.path.join(engine.PEER_CONFIG["DIR"], f"peer_census_{pd.Timestamp(d).date()}.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(f"저장: {path}")


if __name__ == "__main__":
    main()
