"""화면 검사용 가짜 피어 기록: 주어진 종목이 들어 있는 6종목 묶음의 재무·순위 기록을 임시 폴더에 쓴다.
(engine/peers.py의 score_group·_append를 그대로 써서 실제 기록과 같은 모양)"""
import os

import pandas as pd

import engine


def write_fake_peers(base_dir: str, ticker: str, date: str = "2026-11-16", kill_one: bool = True,
                     group: str = "test_group", group_name: str = "테스트 묶음", others: tuple = ("PA", "PB", "PC", "PD", "PK")) -> None:
    os.makedirs(base_dir, exist_ok=True)
    peers = [ticker] + list(others)
    kill_t = peers[-1]
    warn_t = peers[-2]
    recs, funds = [], []
    for i, t in enumerate(peers):
        m = {"ticker": t, "rev_growth_ttm": [0.30, 0.10, 0.05, 0.20, -0.02, 0.5][i], "eps_growth_fwd": [0.40, 0.15, 0.08, 0.20, None, 0.9][i],
             "op_margin_ttm": [0.45, 0.20, 0.15, 0.30, 0.05, 0.6][i], "roe": [0.6, 0.2, 0.1, 0.3, None, 0.9][i],
             "rev_net_30d": [0.05, 0.2, -0.1, 0.3, 0.0, 0.9][i], "fy1_change_30d": [0.0, 0.02, -0.03, 0.04, None, 0.1][i],
             "fwd_pe": [28.0, 18.0, 12.0, 22.0, 35.0, 5.0][i], "peg": [0.7, 1.2, 1.5, 1.1, None, 0.1][i], "band_z": None,
             "rs63": [-0.05, 0.02, 0.06, 0.03, -0.08, 0.2][i], "state_pt": [1.0, 2.0, 2.0, 1.0, 0.0, 2.0][i],
             "surv_kill": kill_one and t == kill_t, "surv_warning": t == warn_t}
        recs.append(m)
        funds.append({"fund_date": date, "ticker": t, "market": "KR" if t.endswith((".KS", ".KQ")) else "US", "group": group,
                      "group_name": group_name, "group_source": "수동", "is_coverage": t == ticker, "short_name": f"{t} Corp",
                      "surv_kill": kill_one and t == kill_t, "surv_warning": t == warn_t, "surv_text": "주식 희석 6.0% > 5%" if t == warn_t else "양호",
                      "next_earnings_date": "2027-01-28", "eps_fy1": 1.0 + i, "n_fy1": 20, "up_30d": 3, "down_30d": 2,
                      "version": engine.PEER_CONFIG["VERSION"]})
    sdf = engine.score_group(pd.DataFrame(recs).set_index("ticker", drop=False))
    rows = []
    for t, r in sdf.iterrows():
        row = {c: r.get(c) for c in engine.RANK_COLUMNS if c in r.index}
        row.update(rank_date=date, group=group, group_name=group_name, group_source="수동", n_group=len(peers),
                   fund_date=date, price_as_of="2026-11-13", trend_state={2.0: "상승", 1.0: "횡보", 0.0: "하락"}[r["state_pt"]],
                   is_coverage=t == ticker, version=engine.PEER_CONFIG["VERSION"])
        for c in ("value_rank", "trend_rank"):
            row[c] = None if pd.isna(row.get(c)) else int(row[c])
        rows.append(row)
    engine.peers._append(engine.fund_path(date, base_dir), funds, engine.FUND_COLUMNS, ["fund_date", "ticker"])
    engine.peers._append(engine.rank_path(date, base_dir), rows, engine.RANK_COLUMNS, ["rank_date", "group", "ticker"])
