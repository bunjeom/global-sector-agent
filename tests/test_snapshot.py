"""컨센서스·목표가 날짜별 기록(engine/snapshot.py) 검사. 가짜 yfinance로 오프라인 실행.

1 야후 표에서 값 읽기(열 이름 대소문자, 회계연도 끝, 실적 발표일)
2 못 받은 항목은 빈칸(추정·대체 없음), 전부 못 받은 종목은 실패로 기록
3 같은 날 다시 실행해도 중복 없음, 이미 받은 종목은 다시 받지 않음, 열 순서 고정
4 관심 종목 파일(watchlist.txt), 연도별 파일, 불러오기
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # 프로젝트 폴더의 engine 패키지를 불러오기 위해
import _offline  # noqa: F401  외부 네트워크 차단 (tests/_offline.py)
import shutil
import sys
import tempfile
import types
import pandas as pd
import engine

FAILS = []
CALLS = []


def chk(n, c):
    print(("PASS " if c else "FAIL ") + n, flush=True)
    (None if c else FAILS.append(n))


def periodic(rows, cols):
    return pd.DataFrame(rows, index=pd.Index(["0q", "+1q", "0y", "+1y"], name="period"), columns=cols)


class FakeTicker:
    """yfinance 1.8의 Ticker 모양을 흉내 낸다 (테스트 입력용)."""

    def __init__(self, sym):
        self.sym = sym
        CALLS.append(sym)

    def history(self, period="5d", auto_adjust=False):
        if self.sym in ("FAIL", "NOPRICE"):
            raise ConnectionError("fake")
        idx = pd.to_datetime(["2026-11-11", "2026-11-12", "2026-11-13"]).tz_localize("America/New_York")
        return pd.DataFrame({"Close": [100.0, 101.5, float("nan")]}, index=idx)   # 마지막 줄은 종가 빈칸(장 마감 직후)

    @property
    def info(self):
        if self.sym == "FAIL":
            raise ConnectionError("fake")
        if self.sym == "KRONLY.KS":
            return {"trailingEps": 5000.0}
        return {"trailingEps": 6.0, "forwardEps": 8.0, "trailingPE": 16.9, "forwardPE": 12.7, "priceToBook": 4.2,
                "marketCap": 3.0e12, "sharesOutstanding": 1.5e10, "targetMeanPrice": 120.0, "targetMedianPrice": 118.0,
                "targetHighPrice": 150.0, "targetLowPrice": 90.0, "numberOfAnalystOpinions": 40, "recommendationMean": 1.9,
                "currency": "USD", "currentPrice": 99.0,
                "nextFiscalYearEnd": 1790726400, "lastFiscalYearEnd": 1759190400}   # 2026-09-30 · 2025-09-30

    @property
    def earnings_estimate(self):
        if self.sym in ("FAIL", "KRONLY.KS"):
            return pd.DataFrame()
        return periodic([[1.9, 1.8, 2.0, 1.7, 30, 0.1], [2.1, 2.0, 2.2, 1.9, 28, 0.1],
                         [7.5, 7.0, 8.0, 6.0, 38, 0.25], [8.4, 7.6, 9.3, 7.5, 37, 0.12]],
                        ["avg", "low", "high", "yearAgoEps", "numberOfAnalysts", "growth"])

    @property
    def eps_trend(self):
        if self.sym in ("FAIL", "KRONLY.KS"):
            return None
        return periodic([[1.9, 1.9, 1.88, 1.85, 1.8], [2.1, 2.1, 2.08, 2.05, 2.0],
                         [7.5, 7.45, 7.3, 7.2, 7.0], [8.4, 8.38, 8.2, 8.1, 7.9]],
                        ["current", "7daysAgo", "30daysAgo", "60daysAgo", "90daysAgo"])

    @property
    def eps_revisions(self):
        if self.sym in ("FAIL", "KRONLY.KS"):
            return None
        return periodic([[1, 3, 0, 0], [1, 2, 1, 0], [4, 12, 2, 1], [3, 10, 3, 0]],
                        ["upLast7days", "upLast30days", "downLast30days", "downLast7Days"])   # 야후 원래 표기(대소문자 섞임)

    @property
    def revenue_estimate(self):
        if self.sym == "NOREV":
            raise RuntimeError("fake")
        if self.sym in ("FAIL", "KRONLY.KS"):
            return pd.DataFrame()
        return periodic([[1e11, 9e10, 1.1e11, 30, 9e10, .1], [1.1e11, 1e11, 1.2e11, 28, 1e11, .1],
                         [4.2e11, 4e11, 4.4e11, 39, 3.9e11, .08], [4.5e11, 4.2e11, 4.8e11, 38, 4.2e11, .07]],
                        ["avg", "low", "high", "numberOfAnalysts", "yearAgoRevenue", "growth"])

    @property
    def calendar(self):
        if self.sym in ("FAIL", "KRONLY.KS"):
            return {}
        return {"Earnings Date": [pd.Timestamp("2027-02-03").date(), pd.Timestamp("2027-01-29").date()]}


def main():
    yf = types.SimpleNamespace(Ticker=FakeTicker)
    tmp = tempfile.mkdtemp(prefix="snap_test_")
    try:
        # 1) 값 읽기
        r = engine.fetch_consensus_row("AAPL", yf)
        chk("1 가격은 종가가 있는 마지막 날(11-12) 값과 그 날짜", r["price"] == 101.5 and r["price_as_of"] == "2026-11-12")
        chk("1 올해·내년 EPS 평균·범위·애널리스트 수", (r["eps_fy1_avg"], r["eps_fy1_low"], r["eps_fy1_high"], r["eps_fy1_n"]) == (7.5, 7.0, 8.0, 38.0)
            and r["eps_fy2_avg"] == 8.4 and r["eps_fy1_year_ago"] == 6.0 and r["eps_q0_avg"] == 1.9 and r["eps_q1_avg"] == 2.1)
        chk("1 EPS 추이 7·30·60·90일 전", (r["eps_fy1_7d_ago"], r["eps_fy1_30d_ago"], r["eps_fy1_60d_ago"], r["eps_fy1_90d_ago"]) == (7.45, 7.3, 7.2, 7.0)
            and r["eps_fy2_90d_ago"] == 7.9)
        chk("1 수정 건수: 열 이름 대소문자가 섞여도 읽음 (downLast7Days)", (r["eps_fy1_up_7d"], r["eps_fy1_up_30d"], r["eps_fy1_down_7d"], r["eps_fy1_down_30d"]) == (4.0, 12.0, 1.0, 2.0))
        chk("1 목표가·매출 추정·PER", r["target_mean"] == 120.0 and r["n_target_analysts"] == 40.0 and r["rev_fy1_avg"] == 4.2e11
            and r["rev_fy2_n"] == 38.0 and r["forward_pe"] == 12.7)
        chk("1 회계연도 끝·다음 실적 발표일(범위면 이른 날)", r["fy1_end"] == "2026-09-30" and r["last_fy_end"] == "2025-09-30"
            and r["next_earnings_date"] == "2027-01-29")
        chk("1 모든 열이 고정 목록에 있음", set(r) - {"_notes"} == set(engine.SNAPSHOT_COLUMNS))

        # 2) 못 받은 항목은 빈칸
        k = engine.fetch_consensus_row("KRONLY.KS", yf)
        chk("2 한국 종목: 컨센서스가 없으면 빈칸, 받은 값(가격·후행 EPS)만 채움",
            k["market"] == "KR" and k["currency"] == "KRW" and k["price"] == 101.5 and k["trailing_eps"] == 5000.0
            and k["eps_fy1_avg"] is None and k["target_mean"] is None and k["next_earnings_date"] is None)
        nr = engine.fetch_consensus_row("NOREV", yf)
        chk("2 한 항목(매출 추정)이 오류여도 나머지는 받고, 못 받은 항목 이름을 남김",
            nr["rev_fy1_avg"] is None and nr["eps_fy1_avg"] == 7.5 and "매출 추정치" in nr["_notes"])
        npx = engine.fetch_consensus_row("NOPRICE", yf)
        chk("2 일봉을 못 받으면 기본 정보의 현재가(날짜는 빈칸)", npx["price"] == 99.0 and npx["price_as_of"] is None)

        # 3) 저장: 중복 없음·다시 받지 않음·열 순서
        CALLS.clear()
        res = engine.run_daily_snapshot(["AAPL", "KRONLY.KS", "FAIL", "NOREV"], base_dir=tmp, yf=yf, today="2026-11-14", pause=0)
        chk("3 4종목 중 3개 저장, 전부 못 받은 1개는 실패 목록", res["saved"] == 3 and res["failed"] == ["FAIL"] and "NOREV" in res["partial"])
        path = _os.path.join(tmp, "consensus_2026.csv")
        raw = pd.read_csv(path, encoding="utf-8-sig", dtype={"ticker": str})
        chk("3 파일 열 순서 = SNAPSHOT_COLUMNS, 3줄", list(raw.columns) == engine.SNAPSHOT_COLUMNS and len(raw) == 3
            and set(raw["snapshot_date"]) == {"2026-11-14"})
        chk("3 빈칸은 빈칸으로 저장 (0이나 다른 값으로 채우지 않음)", pd.isna(raw.loc[raw["ticker"] == "KRONLY.KS", "eps_fy1_avg"].iloc[0]))
        CALLS.clear()
        res2 = engine.run_daily_snapshot(["AAPL", "KRONLY.KS", "FAIL", "NOREV"], base_dir=tmp, yf=yf, today="2026-11-14", pause=0)
        chk("3 같은 날 다시 실행: 저장된 3종목은 받지 않고 실패했던 1종목만 다시 시도", res2["saved"] == 0 and res2["skipped_existing"] == 3 and CALLS == ["FAIL"])
        chk("3 중복 줄 없음", len(pd.read_csv(path, encoding="utf-8-sig")) == 3)
        res3 = engine.run_daily_snapshot(["AAPL"], base_dir=tmp, yf=yf, today="2026-11-15", pause=0)
        raw3 = pd.read_csv(path, encoding="utf-8-sig")
        chk("3 다음 날은 같은 파일 끝에 붙고 머리글은 한 번만", res3["saved"] == 1 and len(raw3) == 4 and (raw3["ticker"] == "ticker").sum() == 0)

        # 4) 관심 종목 파일·연도별 파일·불러오기
        with open(_os.path.join(tmp, "watchlist.txt"), "w", encoding="utf-8") as f:
            f.write("# 내 관심 종목\npltr\n005930.KS   # 삼성전자(기본 목록과 중복)\n\n")
        uni = engine.snapshot_universe(tmp)
        chk("4 watchlist.txt 추가(대문자로, 설명·빈 줄 무시, 중복 제거)", "PLTR" in uni and uni.count("005930.KS") == 1
            and len(uni) == len(engine.SNAPSHOT_CONFIG["UNIVERSE"]) + 1)
        engine.run_daily_snapshot(["AAPL"], base_dir=tmp, yf=yf, today="2027-01-04", pause=0)
        chk("4 해가 바뀌면 새 파일", _os.path.exists(_os.path.join(tmp, "consensus_2027.csv")))
        allr = engine.load_snapshots(tmp)
        chk("4 불러오기: 두 파일 합쳐 5줄, 날짜형, 날짜순", len(allr) == 5 and str(allr["snapshot_date"].dtype).startswith("datetime")
            and allr["snapshot_date"].is_monotonic_increasing and allr["fy1_end"].iloc[0] == pd.Timestamp("2026-09-30"))
        chk("4 기록이 없으면 빈 표", len(engine.load_snapshots(_os.path.join(tmp, "없는폴더"))) == 0)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("FAILS:", FAILS)
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
