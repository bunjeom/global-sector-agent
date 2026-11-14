"""판정 규칙 백테스트 실행 (PC에서 실제 야후 데이터로).

실행 (프로젝트 폴더에서):
    .venv\\Scripts\\python run_backtest.py                 미국·한국 게이트 + T 셋업 (첫 실행 수 분)
    .venv\\Scripts\\python run_backtest.py --skip-setups   게이트만
    .venv\\Scripts\\python run_backtest.py --markets US    미국만
    .venv\\Scripts\\python run_backtest.py --workers 4     병렬 프로세스 수 (기본: CPU 수 − 1)
    .venv\\Scripts\\python run_backtest.py --variants      규칙 후보 비교만 (저장된 CSV 사용, 1분 안팎)
    .venv\\Scripts\\python run_backtest.py --setup-diag    T 셋업 진단: 실제 신호 vs 무작위 진입 (게이트 계산 없음, 수 분)
    .venv\\Scripts\\python run_backtest.py --scanner       실제 T 스캐너 재현 (ETF 19개, 2014년~, 몇 시간, 끊겨도 이어서 계산)
    .venv\\Scripts\\python run_backtest.py --redesign      T 트랙 재설계 후보 검증 (해외 국가 ETF 포함, 수 분)
    .venv\\Scripts\\python run_backtest.py --frame-a       틀 A(추세반등) 개별 종목 검증 (약 150종목 받기 포함 5~10분)
    .venv\\Scripts\\python run_backtest.py --frame-b       틀 B(저평가 매수) 미국 개별 종목 검증, SEC 공시일 기준 EPS (첫 실행 10분 안팎)
                                                     .env에 SEC_USER_AGENT=이름 이메일 필요 (SEC 접속 규칙)
    .venv\\Scripts\\python run_backtest.py --ledger-eval   신호 기록 장부 채점 (data/ledger/ 기록이 쌓인 뒤, 방법은 engine/ledger.py에 고정)

결과 (data/backtest/):
    REPORT.md          사람이 읽는 보고서 (커밋해 두면 다른 AI도 결과를 읽을 수 있음)
    VARIANTS.md        규칙 후보 비교 (앞 기간으로 고르고 뒤 기간으로 확인) + variants.json
    SETUP_DIAG.md      T 셋업 진단 + setup_diag.json
    SCANNER.md         실제 T 스캐너 검증 + scanner.json (계산 캐시: scanner_cache/, .gitignore)
    REDESIGN.md        T 트랙 재설계 후보 검증 + redesign.json
    FRAME_A.md         틀 A(추세반등) 개별 종목 검증 + frame_a.json (거래 목록 frame_a_trades.csv)
    FRAME_B.md         틀 B(저평가 매수) 검증 + frame_b.json (거래 목록 frame_b_trades.csv, SEC 원자료 캐시 data/sec_cache/)
    LEDGER_EVAL.md     신호 기록 장부 채점 + ledger_eval.json
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
    ohlc = {m: px_map[engine.BACKTEST_CONFIG["BENCH"][m]] for m in gate if engine.BACKTEST_CONFIG["BENCH"][m] in px_map}
    v = engine.run_variant_analysis({m: g for m, g in gate.items() if m in prices}, prices, trades, ohlc)
    rep = save_variants(engine, v)
    print("\n" + rep)
    print(f"보고서: {os.path.join(OUT_DIR, 'VARIANTS.md')}")


def run_setup_diag_only(engine, args, t0):
    """T 셋업 진단: 종목 가격을 받아 실제 신호와 무작위 진입을 비교한다 (게이트 계산 없음)."""
    uni = list(dict.fromkeys([s for grp in engine.BACKTEST_CONFIG["SETUP_UNIVERSE"].values() for s in grp] + list(args.tickers)))
    print(f"[1/2] T 셋업 종목 {len(uni)}개 받기", flush=True)
    setup_map, sfail = download(uni, args.start)
    print(f"   받음 {len(setup_map)}개" + (f" · 받지 못함 {len(sfail)}개: {', '.join(sfail)}" if sfail else ""))
    print("[2/2] 실제 신호 · 무작위 진입 계산", flush=True)
    res = engine.run_setup_diag(setup_map)
    os.makedirs(OUT_DIR, exist_ok=True)
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    with open(os.path.join(OUT_DIR, "setup_diag.json"), "w", encoding="utf-8") as f:
        json.dump({"generated_at": now, **res["summary"]}, f, ensure_ascii=False, indent=1, default=str)
    rep = engine.render_setup_diag_report(res["summary"], generated_at=now)
    with open(os.path.join(OUT_DIR, "SETUP_DIAG.md"), "w", encoding="utf-8") as f:
        f.write(rep)
    print("\n" + rep)
    print(f"끝 ({time.time() - t0:.0f}초). 보고서: {os.path.join(OUT_DIR, 'SETUP_DIAG.md')}")


def _save(name_md, name_json, rep, payload):
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, name_json), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1, default=str)
    with open(os.path.join(OUT_DIR, name_md), "w", encoding="utf-8") as f:
        f.write(rep)


def run_scanner_only(engine, args, t0):
    """실제 T 스캐너 재현. 날짜별 판정은 data/backtest/scanner_cache/<티커>.csv에 쌓아 두고, 다시 실행하면 이어서 계산한다."""
    cfg = engine.TBT_CONFIG
    uni = list(cfg["SCANNER_UNIVERSE"])
    start = cfg["DATA_START"]   # 2014년 재현에 필요한 2년 창 + 지표 준비 기간
    print(f"[1/4] 가격 받기: ETF {len(uni)}개 + SPY·^KS11·^VIX (시작 {start})", flush=True)
    pm, fail = download(uni + ["SPY", "^KS11", "^VIX"], start)
    if fail:
        print(f"   받지 못함: {', '.join(fail)}")
    bench = {"US": pm.pop("SPY", None), "KR": pm.pop("^KS11", None)}
    vix = pm.pop("^VIX", None)
    vix = vix["Close"] if vix is not None else None
    cache_dir = os.path.join(OUT_DIR, "scanner_cache")
    os.makedirs(cache_dir, exist_ok=True)
    done, cached = {}, []
    for sym in pm:
        p = os.path.join(cache_dir, f"{sym}.csv")
        if os.path.exists(p):
            c = pd.read_csv(p, parse_dates=["date"])
            cached.append(c)
            done[sym] = set(pd.to_datetime(c["date"]))
    print(f"[2/4] 날짜별 레이더 판정 (병렬 {args.workers}개, 이미 계산한 날 {sum(len(v) for v in done.values()):,}일은 건너뜀)", flush=True)

    def on_chunk(rows):
        df = pd.DataFrame(rows)
        for sym, g in df.groupby("ticker"):
            p = os.path.join(cache_dir, f"{sym}.csv")
            g.reindex(columns=engine.SCANNER_CACHE_COLUMNS).to_csv(p, mode="a", header=not os.path.exists(p), index=False, encoding="utf-8-sig")

    last = {"p": -1}

    def progress(d, n):
        pct = int(d * 100 / max(n, 1))
        if pct // 5 != last["p"]:
            last["p"] = pct // 5
            el = time.time() - t0
            eta = el / max(d, 1) * (n - d)
            print(f"   {pct}% ({d:,}/{n:,}일) · 경과 {el / 60:.0f}분 · 남은 시간 약 {eta / 60:.0f}분", flush=True)

    new = engine.compute_scanner_signals(pm, bench, vix, workers=args.workers, done=done, progress=progress, on_chunk=on_chunk)
    sig = pd.concat(cached + ([new] if len(new) else []), ignore_index=True)
    sig["date"] = pd.to_datetime(sig["date"])
    sig["executable"] = sig["executable"].astype(str).str.lower().isin(["true", "1"])
    sig = sig.drop_duplicates(["ticker", "date"], keep="last")
    print(f"[3/4] 거래 시뮬레이션 · 무작위 비교 (진입 티켓 {int(sig['executable'].sum()):,}건)", flush=True)
    trs, bss = [], []
    for sym, df in pm.items():
        s = sig[sig["ticker"] == sym]
        if not len(s):
            continue
        tr = engine.simulate_scanner_trades(df, s)
        if len(tr):
            trs.append(tr.assign(ticker=sym))
            bs = engine.scanner_random_baseline(df, tr, s)
            if len(bs):
                bss.append(bs.assign(ticker=sym))
    trades = pd.concat(trs, ignore_index=True) if trs else pd.DataFrame()
    base = pd.concat(bss, ignore_index=True) if bss else pd.DataFrame(columns=["r", "setup_id"])
    summ = engine.summarize_scanner(trades, base)
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    meta = {"n_tickers": int(sig["ticker"].nunique()), "start": engine.TBT_CONFIG["SCANNER_START"],
            "end": str(sig["date"].max().date()) if len(sig) else None, "n_days": int(len(sig))}
    rep = engine.render_scanner_report(summ, generated_at=now, meta=meta)
    print("[4/4] 저장", flush=True)
    _save("SCANNER.md", "scanner.json", rep, {"generated_at": now, "meta": meta, **summ})
    if len(trades):
        trades.to_csv(os.path.join(OUT_DIR, "scanner_trades.csv"), index=False, encoding="utf-8-sig")
    print("\n" + rep)
    print(f"끝 ({(time.time() - t0) / 60:.0f}분). 보고서: {os.path.join(OUT_DIR, 'SCANNER.md')}")


def run_redesign_only(engine, args, t0):
    """T 트랙 재설계 후보 검증: 기존 ETF + HOLDOUT(해외 국가 ETF)."""
    cfg = engine.TBT_CONFIG
    main_uni, ho_uni = list(cfg["SCANNER_UNIVERSE"]), list(cfg["HOLDOUT_UNIVERSE"])
    start = "1999-01-01"   # 200일선 준비 기간
    print(f"[1/2] 가격 받기: 기존 ETF {len(main_uni)}개 + HOLDOUT {len(ho_uni)}개", flush=True)
    mm, f1 = download(main_uni, start)
    hm, f2 = download(ho_uni, start)
    if f1 or f2:
        print(f"   받지 못함: {', '.join(f1 + f2)}")
    print("[2/2] 후보 거래 · 무작위 비교", flush=True)
    res = engine.evaluate_redesign(mm, hm)
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    rep = engine.render_redesign_report(res, generated_at=now)
    _save("REDESIGN.md", "redesign.json", rep, {"generated_at": now, **res})
    print("\n" + rep)
    print(f"끝 ({time.time() - t0:.0f}초). 보고서: {os.path.join(OUT_DIR, 'REDESIGN.md')}")


def run_frame_a_only(engine, args, t0):
    """틀 A(추세반등) 검증: 개발 묶음(대표 종목) + HOLDOUT(새 개별 종목), 시장 비교용 SPY·KOSPI."""
    cfg = engine.FRAME_A_CONFIG
    dev_uni, ho_uni = list(cfg["DEV_UNIVERSE"]), list(cfg["HOLDOUT_UNIVERSE"])
    start = cfg["DATA_START"]
    print(f"[1/2] 가격 받기: 시장 2개 + 개발 묶음 {len(dev_uni)}개 + HOLDOUT {len(ho_uni)}개 ({start}~)", flush=True)
    bm, f0 = download(sorted(set(cfg["BENCH"].values())), start)
    if len(bm) < len(cfg["BENCH"]):
        print(f"   시장 데이터를 받지 못했습니다: {', '.join(f0)} — 잠시 뒤 다시 실행해 주세요.")
        return
    dm, f1 = download(dev_uni, start)
    hm, f2 = download(ho_uni, start)
    if f1 or f2:
        print(f"   받지 못함: {', '.join(f1 + f2)}")
    print("[2/2] 거래 계산 · 무작위 비교", flush=True)
    out = engine.evaluate_frame_a(dm, hm, bm)
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    summ = {**out["summary"], "failed_downloads": f1 + f2}
    rep = engine.render_frame_a_report(summ, generated_at=now)
    _save("FRAME_A.md", "frame_a.json", rep, {"generated_at": now, **summ})
    if len(out["trades"]):
        out["trades"].to_csv(os.path.join(OUT_DIR, "frame_a_trades.csv"), index=False, encoding="utf-8-sig")
    print("\n" + rep)
    print(f"끝 ({time.time() - t0:.0f}초). 보고서: {os.path.join(OUT_DIR, 'FRAME_A.md')}")


def download_splits(symbols):
    """야후 액면분할 기록 (틀 B: 분할 전 보고서의 EPS를 지금 주식 단위로 맞추는 데 씀)."""
    import yfinance as yf
    out = {}
    for sym in symbols:
        try:
            s = yf.Ticker(sym).splits
            out[sym] = s if s is not None else pd.Series(dtype=float)
        except Exception:   # 분할 기록을 못 받으면 분할 없음으로 본다(보고서에 종목 표시)
            out[sym] = None
    return out


def run_frame_b_only(engine, args, t0):
    """틀 B(저평가 매수): 미국 대표 종목 + HOLDOUT(새 미국 종목), SEC 공시일 기준 EPS."""
    cfg = engine.FRAME_B_CONFIG
    ua = engine.sec_user_agent()
    if not ua:
        print("SEC는 접속할 때 연락처(User-Agent)를 요구합니다. 프로젝트 폴더의 .env 파일에 아래 한 줄을 추가한 뒤 다시 실행해 주세요.")
        print("   SEC_USER_AGENT=이름 이메일주소      (예: SEC_USER_AGENT=hong research@example.com)")
        return
    dev_uni, ho_uni = list(cfg["DEV_UNIVERSE"]), list(cfg["HOLDOUT_UNIVERSE"])
    uni = dev_uni + ho_uni
    print(f"[1/4] 가격 받기: 개발 묶음 {len(dev_uni)}개 + HOLDOUT {len(ho_uni)}개 ({cfg['PRICE_START']}~)", flush=True)
    pm, fail_px = download(uni, cfg["PRICE_START"])
    print("[2/4] 액면분할 기록 받기", flush=True)
    splits = download_splits(list(pm))
    print("[3/4] SEC EPS 받기 (처음에는 몇 분, 다음부터는 7일 동안 캐시 사용)", flush=True)
    try:
        cik = engine.load_cik_map(ua)
    except Exception as e:
        print(f"   SEC 티커 목록을 받지 못했습니다 ({type(e).__name__}: {e}). 인터넷 연결과 SEC_USER_AGENT를 확인해 주세요.")
        return
    events, sec_note = {}, {}
    for i, sym in enumerate(pm, 1):
        c = cik.get(sym.upper())
        if c is None:
            sec_note[sym] = "SEC 티커 목록에 없음"
            continue
        r = engine.fetch_eps_concept(sym, c, ua)
        if not r.get("tag"):
            sec_note[sym] = r.get("error", "받기 실패")
            continue
        events[sym] = engine.pit_eps_events(engine.parse_eps_facts(r["data"]), splits.get(sym))
        if r["tag"] != "EarningsPerShareDiluted":
            sec_note[sym] = f"{r['tag']} 사용"
        if i % 20 == 0:
            print(f"   {i}/{len(pm)}개", flush=True)
    if fail_px:
        print(f"   가격 못 받음: {', '.join(fail_px)}")
    if sec_note:
        print("   SEC 참고: " + "; ".join(f"{k} {v}" for k, v in sec_note.items()))
    # 자료 점검: 가장 최근 SEC TTM EPS와 야후 trailingEps 비교 (파싱이 맞는지 사람이 확인)
    check = []
    try:
        import yfinance as yf
        for sym in [x for x in ("AAPL", "MSFT", "JPM", "KO", "WMT", "NVDA", "ADP", "TJX") if x in events][:6]:
            ev = events[sym].dropna(subset=["ttm"])
            if not len(ev):
                continue
            last = ev.iloc[-1]
            try:
                yv = yf.Ticker(sym).info.get("trailingEps")
            except Exception:
                yv = None
            check.append({"ticker": sym, "sec_ttm": float(last["ttm"]), "ttm_end": str(pd.Timestamp(last["ttm_end"]).date()),
                          "filed": str(pd.Timestamp(last["filed"]).date()), "yahoo_trailing_eps": yv})
    except Exception:
        pass
    print("[4/4] 거래 계산 · 같은 날 다른 종목과 비교", flush=True)
    dm = {k: v for k, v in pm.items() if k in dev_uni}
    hm = {k: v for k, v in pm.items() if k in ho_uni}
    out = engine.evaluate_frame_b(dm, hm, events)
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    summ = {**out["summary"], "failed_downloads": fail_px, "sec_notes": sec_note, "eps_check": check}
    rep = engine.render_frame_b_report(summ, generated_at=now)
    _save("FRAME_B.md", "frame_b.json", rep, {"generated_at": now, **summ})
    if len(out["trades"]):
        out["trades"].to_csv(os.path.join(OUT_DIR, "frame_b_trades.csv"), index=False, encoding="utf-8-sig")
    print("\n" + rep)
    print(f"끝 ({time.time() - t0:.0f}초). 보고서: {os.path.join(OUT_DIR, 'FRAME_B.md')}")


def run_ledger_eval_only(engine, args, t0):
    """신호 기록 장부 채점: data/ledger/ 기록 + 기록된 종목들의 가격(같은 날 다른 종목 비교용)."""
    lg = engine.load_ledger()
    if not len(lg):
        print("기록이 아직 없습니다(data/ledger/). 매일 자동 기록(ledger_daily.py)이 쌓인 뒤 실행해 주세요.")
        return
    tickers = sorted(lg["ticker"].dropna().unique().tolist())
    start = (lg["as_of"].min() - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    print(f"[1/2] 가격 받기: {len(tickers)}종목 ({start}~) · 기록 {len(lg):,}줄 {lg['as_of'].min().date()}~{lg['as_of'].max().date()}", flush=True)
    pm, fail = download(tickers, start)
    if fail:
        print(f"   받지 못함: {', '.join(fail)}")
    print("[2/2] 채점 (방법은 engine/ledger.py LEDGER_EVAL에 고정)", flush=True)
    res = engine.evaluate_frame_a_ledger(lg, pm)
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    rep = engine.render_ledger_report(res, generated_at=now)
    _save("LEDGER_EVAL.md", "ledger_eval.json", rep, {"generated_at": now, **res})
    print("\n" + rep)
    print(f"끝 ({time.time() - t0:.0f}초). 보고서: {os.path.join(OUT_DIR, 'LEDGER_EVAL.md')}")


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
    ap.add_argument("--setup-diag", action="store_true", help="T 셋업 진단만 (실제 신호 vs 무작위 진입)")
    ap.add_argument("--scanner", action="store_true", help="실제 T 스캐너 재현 (ETF, 2014년~)")
    ap.add_argument("--redesign", action="store_true", help="T 트랙 재설계 후보 검증")
    ap.add_argument("--frame-a", action="store_true", help="틀 A(추세반등) 개별 종목 검증")
    ap.add_argument("--frame-b", action="store_true", help="틀 B(저평가 매수) 미국 개별 종목 검증 (SEC)")
    ap.add_argument("--ledger-eval", action="store_true", help="신호 기록 장부 채점")
    args = ap.parse_args()
    t0 = time.time()
    if args.variants:
        run_variants_only(engine, args)
        return
    if args.setup_diag:
        run_setup_diag_only(engine, args, t0)
        return
    if args.scanner:
        run_scanner_only(engine, args, t0)
        return
    if args.redesign:
        run_redesign_only(engine, args, t0)
        return
    if args.frame_a:
        run_frame_a_only(engine, args, t0)
        return
    if args.frame_b:
        run_frame_b_only(engine, args, t0)
        return
    if args.ledger_eval:
        run_ledger_eval_only(engine, args, t0)
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
