"""판정 규칙 백테스트 실행 (PC에서 실제 야후 데이터로).

실행 (프로젝트 폴더에서):
    .venv\\Scripts\\python run_backtest.py                 미국·한국 게이트 + T 셋업 (첫 실행 수 분)
    .venv\\Scripts\\python run_backtest.py --skip-setups   게이트만
    .venv\\Scripts\\python run_backtest.py --markets US    미국만
    .venv\\Scripts\\python run_backtest.py --workers 4     병렬 프로세스 수 (기본: CPU 수 − 1)
    .venv\\Scripts\\python run_backtest.py --variants      규칙 후보 비교만 (저장된 CSV 사용, 1분 안팎)

결과 (data/backtest/):
    REPORT.md          사람이 읽는 보고서 (커밋해 두면 다른 AI도 결과를 읽을 수 있음)
    VARIANTS.md        규칙 후보 비교 (앞 기간으로 고르고 뒤 기간으로 확인) + variants.json
    summary.json       숫자 요약 + 주간 누적 수익 곡선 (이후 화면 표시용)
    *.csv              날짜별 게이트·누적 수익·거래 목록 (용량이 커서 .gitignore로 커밋 제외)
계산은 engine/rule_backtest.py에 있고, 이 파일은 데이터 다운로드와 저장만 한다.
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime

import pandas as pd

OUT_DIR = os.path.join("data", "backtest")


def download(symbols, start):
    """야후 일봉을 종목별로 받는다. 못 받은 종목은 빠진다(예시 데이터로 채우지 않음)."""
    import yfinance as yf
    out, failed = {}, []
    for i, sym in enumerate(symbols, 1):
        try:
            df = yf.download(sym, start=start, auto_adjust=False, actions=False, progress=False, threads=False)
            if df is not None and len(df):
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df = df.dropna(subset=["Close"])
                if len(df):
                    out[sym] = df
                    continue
        except Exception as e:   # 네트워크·야후 오류
            print(f"   {sym}: 받기 실패 ({type(e).__name__})")
        failed.append(sym)
        if i % 10 == 0:
            print(f"   {i}/{len(symbols)}개 받음", flush=True)
    return out, failed


def save_variants(engine, variants):
    """후보 비교 결과 저장 (VARIANTS.md · variants.json)."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    with open(os.path.join(OUT_DIR, "variants.json"), "w", encoding="utf-8") as f:
        json.dump({"generated_at": now, **variants}, f, ensure_ascii=False, indent=1, default=str)
    rep = engine.render_variants_report(variants, generated_at=now)
    with open(os.path.join(OUT_DIR, "VARIANTS.md"), "w", encoding="utf-8") as f:
        f.write(rep)
    return rep


def run_variants_only(engine, args):
    """저장된 게이트 시계열·거래 목록으로 후보만 비교한다 (지표 재계산 없음). 지수 가격만 다시 받는다."""
    gate = {}
    for m in args.markets:
        p = os.path.join(OUT_DIR, f"gate_series_{m}.csv")
        if not os.path.exists(p):
            print(f"{p} 가 없습니다. 먼저 전체 실행(python run_backtest.py)을 한 번 하세요.")
            return
        g = pd.read_csv(p, index_col=0, parse_dates=True)
        if "comp_regime" not in g.columns:
            print(f"{p} 에 구성 요소 열이 없습니다(예전 코드로 만든 파일). 전체 실행을 다시 하세요.")
            return
        gate[m] = g
    tp = os.path.join(OUT_DIR, "setup_trades.csv")
    trades = pd.read_csv(tp, parse_dates=["date"]) if os.path.exists(tp) else None
    benches = [engine.BACKTEST_CONFIG["BENCH"][m] for m in gate]
    print(f"[1/2] 지수 가격 받기: {', '.join(benches)}", flush=True)
    px_map, failed = download(benches, args.start)
    if failed:
        print(f"   받지 못함: {', '.join(failed)} — 해당 시장은 비교에서 빠집니다.")
    prices = {m: engine._price_col(engine._bt_naive(px_map[engine.BACKTEST_CONFIG["BENCH"][m]]))
              for m in gate if engine.BACKTEST_CONFIG["BENCH"][m] in px_map}
    print("[2/2] 후보 비교", flush=True)
    v = engine.run_variant_analysis({m: g for m, g in gate.items() if m in prices}, prices, trades)
    rep = save_variants(engine, v)
    print("\n" + rep)
    print(f"보고서: {os.path.join(OUT_DIR, 'VARIANTS.md')}")


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")   # Windows 콘솔에서 한글 깨짐 방지
    except Exception:
        pass
    import engine
    cfg = engine.BACKTEST_CONFIG
    ap = argparse.ArgumentParser(description="판정 규칙 백테스트")
    ap.add_argument("--markets", nargs="+", default=["US", "KR"], choices=["US", "KR"])
    ap.add_argument("--start", default=cfg["START"])
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--skip-setups", action="store_true")
    ap.add_argument("--tickers", nargs="*", default=[], help="T 셋업 검증에 추가할 종목")
    ap.add_argument("--variants", action="store_true", help="규칙 후보 비교만 (저장된 gate_series·setup_trades CSV 사용)")
    args = ap.parse_args()
    t0 = time.time()
    if args.variants:
        run_variants_only(engine, args)
        return

    print(f"[1/4] 지수 데이터 받기 (시작 {args.start})", flush=True)
    index_map, failed = download(["QQQ", "SPY", "^KS11", "^VIX"], args.start)
    for k, v in index_map.items():
        print(f"   {k}: {v.index[0].date()} ~ {v.index[-1].date()} ({len(v):,}일)")
    if failed:
        print(f"   받지 못함: {', '.join(failed)}")

    setup_map = None
    if not args.skip_setups:
        uni = [s for grp in cfg["SETUP_UNIVERSE"].values() for s in grp] + list(args.tickers)
        uni = list(dict.fromkeys(uni))
        print(f"[2/4] T 셋업 종목 {len(uni)}개 받기", flush=True)
        setup_map, sfail = download(uni, args.start)
        print(f"   받음 {len(setup_map)}개" + (f" · 받지 못함 {len(sfail)}개: {', '.join(sfail)}" if sfail else ""))
    else:
        print("[2/4] T 셋업 생략 (--skip-setups)")

    print(f"[3/4] 계산 (병렬 프로세스 {args.workers}개). 날짜마다 앱과 같은 게이트 계산이라 시간이 걸립니다.", flush=True)
    last = {}

    def progress(m, done, total):
        pct = int(done * 100 / total)
        if pct // 10 != last.get(m, -1):
            last[m] = pct // 10
            print(f"   {engine.MARKET_NAMES.get(m, m)} 게이트 {pct}% ({done:,}/{total:,}일) · {time.time() - t0:.0f}초", flush=True)

    res = engine.run_rule_backtest(index_map, setup_map, markets=args.markets, workers=args.workers,
                                   progress=progress, generated_at=datetime.now().strftime("%Y-%m-%d %H:%M"))

    print("[4/4] 저장", flush=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    for m, g in res["gate_series"].items():
        g.to_csv(os.path.join(OUT_DIR, f"gate_series_{m}.csv"), encoding="utf-8-sig")
        res["equity"][m].to_csv(os.path.join(OUT_DIR, f"equity_{m}.csv"), encoding="utf-8-sig")
    if len(res["trades"]):
        res["trades"].to_csv(os.path.join(OUT_DIR, "setup_trades.csv"), index=False, encoding="utf-8-sig")
    with open(os.path.join(OUT_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(res["summary"], f, ensure_ascii=False, indent=1, default=str)
    if res.get("variants"):
        save_variants(engine, res["variants"])
    report = engine.render_backtest_report(res["summary"])
    with open(os.path.join(OUT_DIR, "REPORT.md"), "w", encoding="utf-8") as f:
        f.write(report)
    print("\n" + report)
    print(f"끝 ({time.time() - t0:.0f}초). 보고서: {os.path.join(OUT_DIR, 'REPORT.md')}")


if __name__ == "__main__":   # Windows 병렬 계산에 꼭 필요 (자식 프로세스가 이 파일을 다시 읽음)
    main()
