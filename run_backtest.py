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
    .venv\\Scripts\\python run_backtest.py --peer-eval     피어 순위 채점 (data/peers/ 기록, 방법은 engine/peer_eval.py에 고정, 판정은 2028년 무렵)
    .venv\\Scripts\\python run_backtest.py --short-term    단기 규칙(+3% 익절) 검증 — 대표 종목 + 새 HOLDOUT 84종목 (받기 포함 5~10분)
    .venv\\Scripts\\python run_backtest.py --levels        지지선·손절선 검증 — 대표 종목 + 새 HOLDOUT 110종목 (받기 포함 5~10분)
    .venv\\Scripts\\python run_backtest.py --reclaim       지지선 이탈 뒤 회복 검증 — 대표 종목 + 새 HOLDOUT 148종목 (받기 포함 10분 안팎)
    .venv\\Scripts\\python run_backtest.py --explore list  탐색할 수 있는 가설 이름 보기
    .venv\\Scripts\\python run_backtest.py --explore 이름  가설 탐색(이미 쓴 종목 전부, 증거 아님). 처음 한 번만 가격 받기 15분 안팎,
                                                     그 뒤는 저장해 둔 가격(data/backtest/price_cache/)으로 1~2분. --refresh로 다시 받기
    .venv\\Scripts\\python run_backtest.py --lecture data   강의 조건 시험 자료 받기: 섹터 ETF·지수·종목 섹터·실적 발표일 (처음 20~40분, 그 뒤는 캐시)
                                                     .env에 SEC_USER_AGENT(미국 실적일)·DART_API_KEY(한국 실적일) 필요. 끝에 한국 ETF 이름을 확인
    .venv\\Scripts\\python run_backtest.py --lecture L1 --etf-ok   강의 시험 L1(BNF 두 질문) — 한국 ETF 이름을 확인한 뒤에만
    .venv\\Scripts\\python run_backtest.py --lecture L3   강의 시험 L3(상태별 RSI)
    .venv\\Scripts\\python run_backtest.py --lecture L2 --etf-ok   강의 시험 L2(매수 사건 조합 256개, 섞기 200회 통과선) — 노트북 팬이 돌 수 있음
    .venv\\Scripts\\python run_backtest.py --lecture L4   강의 시험 L4(매도·약화 17개, 20일 최대 낙폭)
    .venv\\Scripts\\python run_backtest.py --lecture L5   강의 시험 L5(레벨 11개, 같은 거리 가짜 선 비교) — 30분 넘게 걸릴 수 있음
    .venv\\Scripts\\python run_backtest.py --mr1   H-MR1 단기 반등 단타 탐색(규칙 A·B·C, 기존 종목 2001~2019) — 사전 등록 MR1_PREREG.md
    .venv\\Scripts\\python run_backtest.py --mr2   H-MR2 단기 신호 찾기(26개 규칙, +5%/−5%/3일, 기존 종목 2001~2019 → 후보만 2020~ 검증) — MR2_PREREG.md
    .venv\\Scripts\\python run_backtest.py --mr2-final   H-MR2 최종(검증 통과 규칙만, 새 151종목). 처음엔 받기 + 한국 종목 이름 출력,
                                                     이름을 확인한 뒤 --mr2-final --names-ok
    .venv\\Scripts\\python run_backtest.py --mr3   H-MR3 변동성 국면(VIX ≥ 20) 이격도·지지 반등 찾기 → 후보만 미국 1990~2000 + 2020~ 검증 — MR3_PREREG.md
                                                     (VIX는 처음 한 번 받음, 1990년대 미국 자료는 후보가 있을 때만 받음 10분 안팎)
    .venv\\Scripts\\python run_backtest.py --mr3-final   H-MR3 최종(검증 통과 규칙만, 새 151종목) — 확인 뒤 --mr3-final --names-ok

결과 (data/backtest/):
    REPORT.md          사람이 읽는 보고서 (커밋해 두면 다른 AI도 결과를 읽을 수 있음)
    VARIANTS.md        규칙 후보 비교 (앞 기간으로 고르고 뒤 기간으로 확인) + variants.json
    SETUP_DIAG.md      T 셋업 진단 + setup_diag.json
    SCANNER.md         실제 T 스캐너 검증 + scanner.json (계산 캐시: scanner_cache/, .gitignore)
    REDESIGN.md        T 트랙 재설계 후보 검증 + redesign.json
    FRAME_A.md         틀 A(추세반등) 개별 종목 검증 + frame_a.json (거래 목록 frame_a_trades.csv)
    FRAME_B.md         틀 B(저평가 매수) 검증 + frame_b.json (거래 목록 frame_b_trades.csv, SEC 원자료 캐시 data/sec_cache/)
    LEDGER_EVAL.md     신호 기록 장부 채점 + ledger_eval.json
    PEER_EVAL.md       피어 순위 채점 + peer_eval.json
    SHORT_TERM.md      단기 규칙 검증 + short_term.json (거래 목록 short_term_trades.csv)
    LEVELS.md          지지선·손절선 검증 + levels.json
    RECLAIM.md         지지선 이탈 뒤 회복 검증 + reclaim.json
    EXPLORE_<이름>.md  가설 탐색 결과(증거 아님) + EXPLORE_LOG.md(볼 때마다 한 줄)
    MR1.md · mr1.json  H-MR1 단기 반등 단타 탐색(사전 등록 MR1_PREREG.md)
    MR2.md · mr2.json  H-MR2 단기 신호 찾기·검증(사전 등록 MR2_PREREG.md), 최종은 MR2_FINAL.md · mr2_final.json
    MR3.md · mr3.json  H-MR3 변동성 국면 이격도·지지 반등(사전 등록 MR3_PREREG.md), 최종은 MR3_FINAL.md · mr3_final.json
    LECTURE_L1~L5.md  강의 조건 시험(사전 등록 LECTURE_PREREG.md) + lecture_l*.json, 캐시 lecture_cache/
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


def run_short_term_only(engine, args, t0):
    """단기 규칙 검증: 개발 묶음(대표 종목) + HOLDOUT(어떤 백테스트에도 쓰지 않은 84종목). 규칙·기준은 결과 전에 고정."""
    cfg = engine.SHORT_BT_CONFIG
    dev_uni, ho_uni = list(cfg["DEV_UNIVERSE"]), list(cfg["HOLDOUT_UNIVERSE"])
    start = cfg["DATA_START"]
    print(f"[1/2] 가격 받기: 개발 묶음 {len(dev_uni)}개 + HOLDOUT {len(ho_uni)}개 ({start}~)", flush=True)
    dm, f1 = download(dev_uni, start)
    hm, f2 = download(ho_uni, start)
    if f1 or f2:
        print(f"   받지 못함: {', '.join(f1 + f2)}")
    if len(hm) < len(ho_uni) * 0.8:
        print(f"   HOLDOUT을 {len(hm)}/{len(ho_uni)}개만 받았습니다 — 판정에 쓰기엔 부족해 멈춥니다. 잠시 뒤 다시 실행해 주세요.")
        return
    print("[2/2] 거래 계산 · 무작위 비교 (종목마다 추세 상태를 날짜별로 다시 계산해 몇 분 걸립니다)", flush=True)
    out = engine.evaluate_short_term(dm, hm)
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    summ = {**out["summary"], "failed_downloads": f1 + f2}
    rep = engine.render_short_term_report(summ, generated_at=now)
    if f1 or f2:
        rep += f"\n- 받지 못한 종목(계산에서 빠짐): {', '.join(f1 + f2)}\n"
    _save("SHORT_TERM.md", "short_term.json", rep, {"generated_at": now, **summ})
    if len(out["trades"]):
        out["trades"].to_csv(os.path.join(OUT_DIR, "short_term_trades.csv"), index=False, encoding="utf-8-sig")
    print("\n" + rep)
    print(f"끝 ({time.time() - t0:.0f}초). 보고서: {os.path.join(OUT_DIR, 'SHORT_TERM.md')}")


def run_levels_only(engine, args, t0):
    """지지선·손절선 검증: 개발 묶음(대표 종목) + HOLDOUT(어떤 백테스트에도 쓰지 않은 110종목). 정의·기준은 결과 전에 고정."""
    cfg = engine.LEVEL_BT_CONFIG
    dev_uni, ho_uni = list(cfg["DEV_UNIVERSE"]), list(cfg["HOLDOUT_UNIVERSE"])
    start = cfg["DATA_START"]
    print(f"[1/2] 가격 받기: 개발 묶음 {len(dev_uni)}개 + HOLDOUT {len(ho_uni)}개 ({start}~)", flush=True)
    dm, f1 = download(dev_uni, start)
    hm, f2 = download(ho_uni, start)
    if f1 or f2:
        print(f"   받지 못함: {', '.join(f1 + f2)}")
    if len(hm) < len(ho_uni) * 0.8:
        print(f"   HOLDOUT을 {len(hm)}/{len(ho_uni)}개만 받았습니다 — 판정에 쓰기엔 부족해 멈춥니다. 잠시 뒤 다시 실행해 주세요.")
        return
    print("[2/2] 사건 찾기 · 가짜 선 비교", flush=True)
    out = engine.evaluate_levels(dm, hm)
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    summ = {**out["summary"], "failed_downloads": f1 + f2}
    rep = engine.render_levels_report(summ, generated_at=now)
    if f1 or f2:
        rep += f"\n- 받지 못한 종목(계산에서 빠짐): {', '.join(f1 + f2)}\n"
    _save("LEVELS.md", "levels.json", rep, {"generated_at": now, **summ})
    print("\n" + rep)
    print(f"끝 ({time.time() - t0:.0f}초). 보고서: {os.path.join(OUT_DIR, 'LEVELS.md')}")


def run_reclaim_only(engine, args, t0):
    """지지선 이탈 뒤 회복 검증: 개발 묶음(대표 종목) + HOLDOUT(어떤 백테스트에도 쓰지 않은 148종목). 정의·기준은 결과 전에 고정."""
    cfg = engine.RECLAIM_BT_CONFIG
    dev_uni, ho_uni = list(cfg["DEV_UNIVERSE"]), list(cfg["HOLDOUT_UNIVERSE"])
    start = cfg["DATA_START"]
    print(f"[1/2] 가격 받기: 개발 묶음 {len(dev_uni)}개 + HOLDOUT {len(ho_uni)}개 ({start}~)", flush=True)
    dm, f1 = download(dev_uni, start)
    hm, f2 = download(ho_uni, start)
    if f1 or f2:
        print(f"   받지 못함: {', '.join(f1 + f2)}")
    if len(hm) < len(ho_uni) * 0.8:
        print(f"   HOLDOUT을 {len(hm)}/{len(ho_uni)}개만 받았습니다 — 판정에 쓰기엔 부족해 멈춥니다. 잠시 뒤 다시 실행해 주세요.")
        return
    print("[2/2] 이탈·회복 사건 찾기 · 무작위 비교", flush=True)
    out = engine.evaluate_reclaim(dm, hm)
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    summ = {**out["summary"], "failed_downloads": f1 + f2}
    rep = engine.render_reclaim_report(summ, generated_at=now)
    if f1 or f2:
        rep += f"\n- 받지 못한 종목(계산에서 빠짐): {', '.join(f1 + f2)}\n"
    _save("RECLAIM.md", "reclaim.json", rep, {"generated_at": now, **summ})
    print("\n" + rep)
    print(f"끝 ({time.time() - t0:.0f}초). 보고서: {os.path.join(OUT_DIR, 'RECLAIM.md')}")


def run_explore_only(engine, args, t0):
    """가설 탐색: 지금까지 쓴 개별 종목 전부로 사건 뒤 수익률을 본다. 결과는 증거가 아니다(EXPLORE_LOG.md에 본 횟수가 남는다)."""
    cfg = engine.EXPLORE_CONFIG
    hyp = engine.EXPLORE_HYPOTHESES
    name = args.explore
    if name == "list" or name not in hyp:
        if name != "list":
            print(f"등록되지 않은 가설: {name}")
        print("탐색할 수 있는 가설:")
        for k, h in hyp.items():
            print(f"   {k:<16} {h['label']}" + (f"  ({h['note']})" if h.get("note") else ""))
        return
    uni = list(cfg["UNIVERSE"])
    pm = {} if args.refresh else engine.load_price_cache(uni)
    need = [s for s in uni if s not in pm]
    print(f"[1/2] 가격: 저장해 둔 것 {len(pm)}개" + (f", 새로 받기 {len(need)}개 ({cfg['DATA_START']}~, 종목이 많아 시간이 걸립니다)" if need else ""), flush=True)
    failed = []
    if need:
        got, failed = download(need, cfg["DATA_START"])
        engine.save_price_cache(got)
        pm.update(got)
        if failed:
            print(f"   받지 못함: {', '.join(failed)}")
    if len(pm) < len(uni) * 0.8:
        print(f"   가격을 {len(pm)}/{len(uni)}개만 확보했습니다 — 잠시 뒤 다시 실행해 주세요.")
        return
    print(f"[2/2] 사건 찾기 · 무작위 비교 ({hyp[name]['label']})", flush=True)
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    if hyp[name].get("custom") == "mom":     # 모멘텀: 매달 줄 세워 상위·하위 20% 비교(전용 계산)
        res = engine.run_momentum_explore(pm, signal=name)
        rep = engine.render_momentum_report(res, generated_at=now)
    elif hyp[name].get("custom") in ("touch", "touch_b"):     # 수평선 터치 횟수별 뚫림 비율(전용 계산)
        res = engine.run_touch_explore(pm, decisive=hyp[name]["custom"] == "touch_b")
        rep = engine.render_touch_report(res, generated_at=now)
    else:
        res = engine.run_explore(name, pm)
        rep = engine.render_explore_report(res, generated_at=now)
    if failed:
        rep += f"\n- 받지 못한 종목(계산에서 빠짐): {', '.join(failed)}\n"
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, f"EXPLORE_{name}.md"), "w", encoding="utf-8") as f:
        f.write(rep)
    engine.append_explore_log(res, now, OUT_DIR)
    print("\n" + rep)
    print(f"끝 ({time.time() - t0:.0f}초). 보고서: {os.path.join(OUT_DIR, f'EXPLORE_{name}.md')} · 기록: {os.path.join(OUT_DIR, cfg['LOG_FILE'])}")


def _lecture_prices(engine, args):
    """탐색 종목 가격(--explore와 같은 캐시). 못 받으면 None."""
    cfg = engine.EXPLORE_CONFIG
    uni = list(cfg["UNIVERSE"])
    pm = {} if args.refresh else engine.load_price_cache(uni)
    need = [s for s in uni if s not in pm]
    print(f"   종목 가격: 저장해 둔 것 {len(pm)}개" + (f", 새로 받기 {len(need)}개" if need else ""), flush=True)
    if need:
        got, failed = download(need, cfg["DATA_START"])
        engine.save_price_cache(got)
        pm.update(got)
        if failed:
            print(f"   받지 못함: {', '.join(failed)}")
    if len(pm) < len(uni) * 0.8:
        print(f"   가격을 {len(pm)}/{len(uni)}개만 확보했습니다 — 잠시 뒤 다시 실행해 주세요.")
        return None
    return pm


def _lecture_bench_symbols(engine):
    lc = engine.LECTURE_CONFIG
    syms = list(lc["INDEX"].values()) + list(lc["SECTOR_ETF_US"].values()) + [v for v in lc["SECTOR_ETF_KR"].values() if v]
    return list(dict.fromkeys(syms))


def run_lecture_data(engine, args, t0):
    """강의 조건 시험 자료: 섹터 ETF·지수 가격, 종목 섹터, 실적 발표일(SEC 8-K 2.02 · DART 잠정실적). 모두 lecture_cache/에 저장."""
    import yfinance as yf
    lc = engine.LECTURE_CONFIG
    bdir = engine.lec_cache_path("prices")
    print("[1/4] 종목 가격", flush=True)
    pm = _lecture_prices(engine, args)
    if pm is None:
        return
    print("[2/4] 섹터 ETF·지수 가격", flush=True)
    bsyms = _lecture_bench_symbols(engine)
    got, failed = download(bsyms, lc["DATA_START"])
    engine.save_price_cache(got, cache_dir=bdir)
    for k in bsyms:
        if k in got:
            print(f"   {k}: {got[k].index[0].date()} ~ {got[k].index[-1].date()}")
    if failed:
        print(f"   받지 못함(그 섹터는 '데이터 없음'): {', '.join(failed)}")
    print("[3/4] 종목 섹터(야후 지금 분류) · 한국 ETF 이름", flush=True)
    sec = engine.lec_load_csv_map("sectors.csv", "ticker", "sector")
    todo = [s for s in pm if s not in sec]
    for i, s in enumerate(todo, 1):
        try:
            sec[s] = (yf.Ticker(s).info or {}).get("sector") or None
        except Exception:
            sec[s] = None
        if i % 50 == 0:
            print(f"   {i}/{len(todo)}", flush=True)
            engine.lec_save_csv_map("sectors.csv", "ticker", "sector", sec)
    engine.lec_save_csv_map("sectors.csv", "ticker", "sector", sec)
    print(f"   섹터를 모르는 종목 {sum(1 for s in pm if not sec.get(s))}개 / {len(pm)}개 (그 종목은 섹터 비교가 '데이터 없음')")
    names = {}
    for etf, expect in lc["SECTOR_ETF_KR_EXPECT"].items():
        try:
            inf = yf.Ticker(etf).info or {}
            names[etf] = inf.get("longName") or inf.get("shortName") or ""
        except Exception:
            names[etf] = ""
    engine.lec_save_csv_map("kr_etf_names.csv", "etf", "name", names)
    print("[4/4] 실적 발표일", flush=True)
    rows, cover = [], {}
    us = [s for s in pm if engine.lec_market_of(s) == "US"]
    kr = [s for s in pm if engine.lec_market_of(s) == "KR"]
    ua = engine.sec_user_agent()
    if not ua:
        print("   .env에 SEC_USER_AGENT(이름 이메일)가 없어 미국 실적일을 받지 않았습니다 → 미국은 악재 제외 '확인 불가'")
    else:
        cik = engine.load_cik_map(ua)
        pause = engine.SEC_CONFIG["PAUSE_SEC"]

        def slow_get(u):
            time.sleep(pause)
            return engine._sec_http_json(u, ua)
        for i, s in enumerate(us, 1):
            c = cik.get(s.upper().replace(".", "-"))
            if c is None:
                continue
            try:
                ds = engine.lec_sec_earnings_dates(c, ua, fetch=slow_get)
            except Exception as e:
                print(f"   {s}: SEC 받기 실패 ({type(e).__name__})")
                continue
            rows += [{"ticker": s, "date": d, "source": "SEC 8-K 2.02"} for d in ds]
            cover[s] = lc["EARN_COVER_US"] if ds else None
            if i % 50 == 0:
                print(f"   미국 {i}/{len(us)}", flush=True)
    dart = engine._get_dart_client()
    if dart is None:
        print("   DART_API_KEY 또는 OpenDartReader가 없어 한국 실적일을 받지 않았습니다 → 한국은 악재 제외 '확인 불가'")
    else:
        for i, s in enumerate(kr, 1):
            ds = engine.lec_dart_earnings_dates(s.split(".")[0], dart)
            rows += [{"ticker": s, "date": d, "source": "DART 잠정실적"} for d in ds]
            cover[s] = lc["EARN_COVER_KR"] if ds else None
            if i % 25 == 0:
                print(f"   한국 {i}/{len(kr)}", flush=True)
    os.makedirs(lc["CACHE_DIR"], exist_ok=True)
    pd.DataFrame(rows, columns=["ticker", "date", "source"]).to_csv(engine.lec_cache_path("earnings_dates.csv"), index=False, encoding="utf-8")
    engine.lec_save_csv_map("earnings_cover.csv", "ticker", "cover_from", cover)
    n_cov = sum(1 for v in cover.values() if v)
    print(f"   실적일을 받은 종목 {n_cov}개 / {len(pm)}개 (나머지는 악재 제외 '확인 불가'로 보고서에 비율 표시)")
    print("\n한국 섹터 ETF 이름 확인 — 아래 '받은 이름'이 '기대한 ETF'와 같은 상품인지 확인해 주세요:")
    for etf, expect in lc["SECTOR_ETF_KR_EXPECT"].items():
        print(f"   {etf}  기대: {expect}  /  받은 이름: {names.get(etf) or '(못 받음)'}")
    print(f"\n끝 ({time.time() - t0:.0f}초). 이름이 맞으면: python run_backtest.py --lecture L1 --etf-ok")


def _lecture_load(engine, need_rs):
    lc = engine.LECTURE_CONFIG
    sec = engine.lec_load_csv_map("sectors.csv", "ticker", "sector")
    if not sec:
        print("먼저 자료를 받아 주세요: python run_backtest.py --lecture data")
        return None
    pm = engine.load_price_cache(list(engine.EXPLORE_CONFIG["UNIVERSE"]))
    bench = engine.load_price_cache(_lecture_bench_symbols(engine), cache_dir=engine.lec_cache_path("prices"))
    ep = engine.lec_cache_path("earnings_dates.csv")
    et = pd.read_csv(ep, dtype=str) if os.path.exists(ep) else None
    cover = engine.lec_load_csv_map("earnings_cover.csv", "ticker", "cover_from")
    if et is None or not len(et):
        print("   실적일 자료 없음 → 악재 제외는 모두 '확인 불가'")
    last = {}

    def prog(i, n):
        if i * 10 // n != last.get("p"):
            last["p"] = i * 10 // n
            print(f"   준비 {i}/{n}", flush=True)
    prep = engine.lec_prepare(pm, sec, bench, et, cover, need_rs=need_rs, progress=prog)
    return prep, bench


def run_lecture_only(engine, args, t0):
    lc = engine.LECTURE_CONFIG
    what = args.lecture
    if what == "data":
        run_lecture_data(engine, args, t0)
        return
    if what not in ("L1", "L2", "L3", "L4", "L5"):
        print("--lecture 값: data · L1 · L2 · L3 · L4 · L5")
        return
    if what in ("L1", "L2") and not args.etf_ok:
        names = engine.lec_load_csv_map("kr_etf_names.csv", "etf", "name")
        print("한국 섹터 ETF 이름을 확인한 뒤 --etf-ok를 붙여 실행해 주세요 (잘못된 ETF면 한국 섹터 비교가 틀립니다):")
        for etf, expect in lc["SECTOR_ETF_KR_EXPECT"].items():
            print(f"   {etf}  기대: {expect}  /  받은 이름: {names.get(etf) or '(못 받음 — --lecture data 먼저)'}")
        return
    print(f"[1/2] 재료 준비 (설정 {lc['VERSION']}, 사전 등록 {lc['REGISTERED']})", flush=True)
    got = _lecture_load(engine, need_rs=(what in ("L1", "L2")))
    if got is None:
        return
    prep, bench = got
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    print(f"[2/2] 시험 {what} (종목 {len(prep)}개)", flush=True)
    grades = {}
    if what in ("L2", "L3"):
        for k, sym in lc["INDEX"].items():
            if sym in bench:
                grades[sym] = engine.lec_market_grade(engine.lec_clean(bench[sym]))
    if what == "L1":
        res = engine.run_lecture_l1(prep)
        rep = engine.render_lecture_l1_report(res, generated_at=now)
    elif what == "L2":
        last = {}

        def prog(stage, i, n):
            if i * 10 // n != last.get(stage):
                last[stage] = i * 10 // n
                print(f"   {stage} {i}/{n}", flush=True)
        print("   사건 찾기 (종목마다)", flush=True)
        tab = engine.lec_l2_table(prep, bench, grades, progress=lambda i, n: prog("사건", i, n))
        print(f"   조합 계산 + 섞기 {lc['RC_REPS']}회", flush=True)
        res = engine.run_lecture_l2(tab, progress=prog)
        rep = engine.render_lecture_l2_report(res, generated_at=now)
    elif what in ("L4", "L5"):
        last = {}

        def prog(stage, i, n):
            if i * 10 // n != last.get(stage):
                last[stage] = i * 10 // n
                print(f"   {stage} {i}/{n}", flush=True)
        print("   사건 찾기 + 결과 재기 (종목마다)", flush=True)
        if what == "L4":
            tab = engine.lec_l4_table(prep, progress=lambda i, n: prog("사건", i, n))
            res = engine.run_lecture_l4(tab, progress=prog)
            rep = engine.render_lecture_l4_report(res, generated_at=now)
        else:
            tab = engine.lec_l5_table(prep, progress=lambda i, n: prog("사건", i, n))
            res = engine.run_lecture_l5(tab, progress=prog)
            rep = engine.render_lecture_l5_report(res, generated_at=now)
    else:
        res = engine.run_lecture_l3(prep, grades)
        rep = engine.render_lecture_l3_report(res, generated_at=now)
    name = f"LECTURE_{what}"
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, f"{name}.md"), "w", encoding="utf-8") as f:
        f.write(rep)
    with open(os.path.join(OUT_DIR, f"{name.lower()}.json"), "w", encoding="utf-8") as f:
        json.dump({"generated_at": now, **res}, f, ensure_ascii=False, indent=1, default=str)
    if what == "L2":
        engine.lec_append_lines(engine.lec_l2_log_lines(res, now), OUT_DIR)
    elif what in ("L4", "L5"):
        engine.lec_append_lines(engine.lec_l45_log_lines(res, now), OUT_DIR)
    else:
        engine.lec_append_log(res, now, OUT_DIR)
    print("\n" + rep)
    print(f"끝 ({time.time() - t0:.0f}초). 보고서: {os.path.join(OUT_DIR, name + '.md')}")


def run_mr1_only(engine, args, t0):
    """H-MR1 단기 반등 단타 탐색: 기존 545종목, 신호일 2001~2019, 규칙 A·B·C. 강의 시험 자료(--lecture data)를 그대로 쓴다."""
    mc = engine.MR1_CONFIG
    print(f"[1/2] 재료 준비 (설정 {mc['VERSION']}, 사전 등록 {mc['REGISTERED']})", flush=True)
    got = _lecture_load(engine, need_rs=False)
    if got is None:
        return
    prep, bench = got
    idx = engine.mr_index_context(bench)
    if len(idx) < 3:
        print(f"   지수 자료가 모자랍니다({sorted(idx)}). python run_backtest.py --lecture data 를 먼저 실행해 주세요.")
    last = {}

    def prog(stage, i, n):
        if i * 10 // n != last.get(stage):
            last[stage] = i * 10 // n
            print(f"   {stage} {i}/{n}", flush=True)
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    print(f"[2/2] 규칙 A·B·C (종목 {len(prep)}개)", flush=True)
    res = engine.run_mr1(prep, idx, progress=prog)
    rep = engine.render_mr1_report(res, generated_at=now)
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "MR1.md"), "w", encoding="utf-8") as f:
        f.write(rep)
    with open(os.path.join(OUT_DIR, "mr1.json"), "w", encoding="utf-8") as f:
        json.dump({"generated_at": now, **res}, f, ensure_ascii=False, indent=1, default=str)
    engine.lec_append_lines(engine.mr1_log_lines(res, now), OUT_DIR)
    print("\n" + rep)
    print(f"끝 ({time.time() - t0:.0f}초). 보고서: {os.path.join(OUT_DIR, 'MR1.md')}")


def run_mr2_only(engine, args, t0):
    """H-MR2 단기 신호 찾기: 기존 545종목, 신호일 2001~2019, 26개 규칙 + 섞기 통과선 → 후보만 2020년 이후 검증. 강의 시험 자료를 그대로 쓴다."""
    mc = engine.MR2_CONFIG
    print(f"[1/2] 재료 준비 (설정 {mc['VERSION']}, 사전 등록 {mc['REGISTERED']})", flush=True)
    got = _lecture_load(engine, need_rs=False)
    if got is None:
        return
    prep, _ = got
    last = {}

    def prog(stage, i, n):
        if i * 10 // n != last.get(stage):
            last[stage] = i * 10 // n
            print(f"   {stage} {i}/{n}", flush=True)
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    print(f"[2/2] 규칙 {len(engine.MR2_RULES)}개 찾기 + 섞기 {mc['RC_REPS']}회 (종목 {len(prep)}개)", flush=True)
    res = engine.run_mr2(prep, progress=prog)
    rep = engine.render_mr2_report(res, generated_at=now)
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "MR2.md"), "w", encoding="utf-8") as f:
        f.write(rep)
    with open(os.path.join(OUT_DIR, "mr2.json"), "w", encoding="utf-8") as f:
        json.dump({"generated_at": now, **res}, f, ensure_ascii=False, indent=1, default=str)
    engine.lec_append_lines(engine.mr2_log_lines(res, now), OUT_DIR)
    print("\n" + rep)
    print(f"끝 ({time.time() - t0:.0f}초). 보고서: {os.path.join(OUT_DIR, 'MR2.md')}")


def run_mr2_final_only(engine, args, t0):
    """H-MR2 최종: mr2.json의 검증 통과 규칙만, 새 151종목(LECTURE_CONFIG FINAL_UNIVERSE)에서 한 번.
    검증 통과가 없으면 새 종목 가격을 받지도 열지도 않는다. 처음엔 받기 + 한국 이름 출력, 확인 뒤 --names-ok로 실행."""
    mc, lc = engine.MR2_CONFIG, engine.LECTURE_CONFIG
    jp = os.path.join(OUT_DIR, "mr2.json")
    if not os.path.exists(jp):
        print("먼저 찾기·검증을 실행해 주세요: python run_backtest.py --mr2")
        return
    with open(jp, encoding="utf-8") as f:
        prev = json.load(f)
    if prev.get("config_version") != mc["VERSION"]:
        print(f"mr2.json 설정({prev.get('config_version')})이 지금 설정({mc['VERSION']})과 다릅니다. --mr2를 다시 실행해 주세요.")
        return
    rules = [r for r in prev.get("verified", []) if r in engine.MR2_RULES]
    if not rules:
        print("검증을 통과한 규칙이 없어 최종 단계를 열지 않습니다(새 151종목은 다음 질문을 위해 아껴 둡니다).")
        return
    cdir = engine.lec_cache_path("final_prices")
    uni = list(lc["FINAL_UNIVERSE"])
    pm = engine.load_price_cache(uni, cache_dir=cdir)
    need = [s for s in uni if s not in pm]
    if need:
        print(f"[받기] 새 종목 {len(need)}개 (저장 위치 {cdir})", flush=True)
        got, failed = download(need, lc["DATA_START"])
        engine.save_price_cache(got, cache_dir=cdir)
        pm.update(got)
        if failed:
            print(f"   받지 못함(빼고 채우지 않음): {', '.join(failed)}")
    if not args.names_ok:
        import yfinance as yf
        print("\n한국 종목 이름 확인 — 코드가 생각한 회사인지 확인한 뒤 --mr2-final --names-ok 로 다시 실행해 주세요:")
        for s in [x for x in uni if engine.lec_market_of(x) == "KR"]:
            try:
                inf = yf.Ticker(s).info or {}
                nm = inf.get("longName") or inf.get("shortName") or "(이름 못 받음)"
            except Exception:
                nm = "(이름 못 받음)"
            print(f"   {s}  {nm}  {'(가격 없음)' if s not in pm else ''}")
        return
    print(f"[1/2] 재료 준비 (설정 {mc['VERSION']}) · 규칙: {', '.join(rules)}", flush=True)
    prep, dropped = engine.mr2_prepare_plain(pm)
    dropped += [s for s in uni if s not in pm]
    last = {}

    def prog(stage, i, n):
        if i * 10 // n != last.get(stage):
            last[stage] = i * 10 // n
            print(f"   {stage} {i}/{n}", flush=True)
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    print(f"[2/2] 최종 (종목 {len(prep)}개)", flush=True)
    res = engine.run_mr2_final(prep, rules, progress=prog)
    rep = engine.render_mr2_final_report(res, sorted(set(dropped)), generated_at=now)
    with open(os.path.join(OUT_DIR, "MR2_FINAL.md"), "w", encoding="utf-8") as f:
        f.write(rep)
    with open(os.path.join(OUT_DIR, "mr2_final.json"), "w", encoding="utf-8") as f:
        json.dump({"generated_at": now, "dropped": sorted(set(dropped)), **res}, f, ensure_ascii=False, indent=1, default=str)
    engine.lec_append_lines(engine.mr2_final_log_lines(res, now), OUT_DIR)
    print("\n" + rep)
    print(f"끝 ({time.time() - t0:.0f}초). 보고서: {os.path.join(OUT_DIR, 'MR2_FINAL.md')}")


def _mr3_vix(engine):
    """VIX 종가(1990~). lecture_cache/prices/^VIX.csv에 저장, 7일보다 오래됐으면 다시 받는다."""
    mc = engine.MR3_CONFIG
    sym = mc["VIX_SYMBOL"]
    cdir = engine.lec_cache_path("prices")
    got = engine.load_price_cache([sym], cache_dir=cdir)
    fresh = sym in got and len(got[sym]) and pd.Timestamp(got[sym].index.max()) >= pd.Timestamp(datetime.now()) - pd.Timedelta(days=7)
    if not fresh:
        print(f"   VIX 받기 ({mc['VIX_START']}~)", flush=True)
        dl, _ = download([sym], mc["VIX_START"])
        if sym in dl:
            engine.save_price_cache(dl, cache_dir=cdir)
            got = dl
    if sym not in got:
        return None
    sr = pd.to_numeric(got[sym]["Close"], errors="coerce").dropna()
    idx = pd.DatetimeIndex(sr.index)
    sr.index = idx.tz_localize(None) if idx.tz is not None else idx
    print(f"   VIX: {sr.index[0].date()} ~ {sr.index[-1].date()} ({len(sr):,}일)")
    return sr


def _mr3_old_loader(engine):
    """1990년대 검증 재료(미국 종목, 1989~2001-06). 찾기 후보가 있을 때만 run_mr3이 부른다. lecture_cache/prices_1989/에 저장."""
    def load():
        mc = engine.MR3_CONFIG
        cdir = engine.lec_cache_path("prices_1989")
        uni = [s for s in engine.EXPLORE_CONFIG["UNIVERSE"] if engine.lec_market_of(s) == "US"]
        pm = engine.load_price_cache(uni, cache_dir=cdir)
        need = [s for s in uni if s not in pm]
        if need:
            print(f"   1990년대 자료 받기: 미국 {len(need)}개 ({mc['OLD_DATA_START']}~{mc['OLD_DATA_END']}, 저장 {cdir}) — 10분 안팎", flush=True)
            got, failed = download(need, mc["OLD_DATA_START"])
            end = pd.Timestamp(mc["OLD_DATA_END"])
            cut = {}
            for k, v in got.items():
                ix = pd.DatetimeIndex(v.index)
                ix = ix.tz_localize(None) if ix.tz is not None else ix
                if (ix <= end).any():
                    cut[k] = v[ix <= end]
            got = cut
            engine.save_price_cache(got, cache_dir=cdir)
            pm.update(got)
            if failed:
                print(f"   받지 못함: {len(failed)}개")
        prep, dropped = engine.mr3_prepare_old(pm)
        print(f"   1990년대 재료: 미국 {len(prep)}개 (1990~2000 자료가 없거나 짧은 {len(dropped)}개 제외)", flush=True)
        return prep
    return load


def run_mr3_only(engine, args, t0):
    """H-MR3: 545종목 2001~2019 변동성 국면 찾기 → 후보만 미국 1990~2000 + 2020~ 검증. 강의 시험 자료 + VIX."""
    mc = engine.MR3_CONFIG
    print(f"[1/3] 재료 준비 (설정 {mc['VERSION']}, 사전 등록 {mc['REGISTERED']})", flush=True)
    vix = _mr3_vix(engine)
    if vix is None or not len(vix):
        print("   VIX를 받지 못했습니다. 잠시 뒤 다시 실행해 주세요.")
        return
    got = _lecture_load(engine, need_rs=False)
    if got is None:
        return
    prep, _ = got
    last = {}

    def prog(stage, i, n):
        if i * 10 // n != last.get(stage):
            last[stage] = i * 10 // n
            print(f"   {stage} {i}/{n}", flush=True)
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    print(f"[2/3] 규칙 {len(engine.MR3_RULES)}개 찾기 + 섞기 {mc['RC_REPS']}회 (종목 {len(prep)}개)", flush=True)
    res = engine.run_mr3(prep, vix, old_loader=_mr3_old_loader(engine), progress=prog)
    print(f"[3/3] 저장 ({res.get('verify_status')})", flush=True)
    rep = engine.render_mr3_report(res, generated_at=now)
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "MR3.md"), "w", encoding="utf-8") as f:
        f.write(rep)
    with open(os.path.join(OUT_DIR, "mr3.json"), "w", encoding="utf-8") as f:
        json.dump({"generated_at": now, **res}, f, ensure_ascii=False, indent=1, default=str)
    engine.lec_append_lines(engine.mr3_log_lines(res, now), OUT_DIR)
    print("\n" + rep)
    print(f"끝 ({time.time() - t0:.0f}초). 보고서: {os.path.join(OUT_DIR, 'MR3.md')}")


def run_mr3_final_only(engine, args, t0):
    """H-MR3 최종: mr3.json의 검증 통과 규칙만, 새 151종목에서 한 번. 검증 통과가 없으면 새 종목을 받지도 열지도 않는다."""
    mc, lc = engine.MR3_CONFIG, engine.LECTURE_CONFIG
    jp = os.path.join(OUT_DIR, "mr3.json")
    if not os.path.exists(jp):
        print("먼저 찾기·검증을 실행해 주세요: python run_backtest.py --mr3")
        return
    with open(jp, encoding="utf-8") as f:
        prev = json.load(f)
    if prev.get("config_version") != mc["VERSION"]:
        print(f"mr3.json 설정({prev.get('config_version')})이 지금 설정({mc['VERSION']})과 다릅니다. --mr3를 다시 실행해 주세요.")
        return
    rules = [r for r in prev.get("verified", []) if r in engine.MR3_RULES]
    if not rules:
        print("검증을 통과한 규칙이 없어 최종 단계를 열지 않습니다(새 151종목은 다음 질문을 위해 아껴 둡니다).")
        return
    vix = _mr3_vix(engine)
    if vix is None:
        print("   VIX를 받지 못했습니다.")
        return
    cdir = engine.lec_cache_path("final_prices")
    uni = list(lc["FINAL_UNIVERSE"])
    pm = engine.load_price_cache(uni, cache_dir=cdir)
    need = [s for s in uni if s not in pm]
    if need:
        print(f"[받기] 새 종목 {len(need)}개 (저장 위치 {cdir})", flush=True)
        got, failed = download(need, lc["DATA_START"])
        engine.save_price_cache(got, cache_dir=cdir)
        pm.update(got)
        if failed:
            print(f"   받지 못함(빼고 채우지 않음): {', '.join(failed)}")
    if not args.names_ok:
        import yfinance as yf
        print("\n한국 종목 이름 확인 — 코드가 생각한 회사인지 확인한 뒤 --mr3-final --names-ok 로 다시 실행해 주세요:")
        for s in [x for x in uni if engine.lec_market_of(x) == "KR"]:
            try:
                inf = yf.Ticker(s).info or {}
                nm = inf.get("longName") or inf.get("shortName") or "(이름 못 받음)"
            except Exception:
                nm = "(이름 못 받음)"
            print(f"   {s}  {nm}  {'(가격 없음)' if s not in pm else ''}")
        return
    print(f"[1/2] 재료 준비 (설정 {mc['VERSION']}) · 규칙: {', '.join(rules)}", flush=True)
    prep, dropped = engine.mr2_prepare_plain(pm)
    dropped += [s for s in uni if s not in pm]
    last = {}

    def prog(stage, i, n):
        if i * 10 // n != last.get(stage):
            last[stage] = i * 10 // n
            print(f"   {stage} {i}/{n}", flush=True)
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    print(f"[2/2] 최종 (종목 {len(prep)}개)", flush=True)
    res = engine.run_mr3_final(prep, vix, rules, progress=prog)
    rep = engine.render_mr3_final_report(res, sorted(set(dropped)), generated_at=now)
    with open(os.path.join(OUT_DIR, "MR3_FINAL.md"), "w", encoding="utf-8") as f:
        f.write(rep)
    with open(os.path.join(OUT_DIR, "mr3_final.json"), "w", encoding="utf-8") as f:
        json.dump({"generated_at": now, "dropped": sorted(set(dropped)), **res}, f, ensure_ascii=False, indent=1, default=str)
    engine.lec_append_lines(engine.mr3_final_log_lines(res, now), OUT_DIR)
    print("\n" + rep)
    print(f"끝 ({time.time() - t0:.0f}초). 보고서: {os.path.join(OUT_DIR, 'MR3_FINAL.md')}")


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


def run_peer_eval_only(engine, args, t0):
    """피어 순위 채점: data/peers/ 순위 기록 + 기록된 종목들의 가격. 방법은 engine/peer_eval.py PEER_EVAL에 고정."""
    rk = engine.load_peer_ranks()
    fds = engine.formation_dates(rk) if len(rk) else []
    if not fds:
        print(f"채점할 순위 기록이 없습니다(시작일 {engine.PEER_EVAL['START']} 이후 data/peers/peer_rank_*.csv).")
        return
    use = rk[pd.to_datetime(rk["rank_date"]).isin(fds)]
    tickers = sorted(use["ticker"].dropna().astype(str).unique().tolist())
    start = (pd.to_datetime(use["price_as_of"], errors="coerce").min() - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    print(f"[1/2] 가격 받기: {len(tickers)}종목 ({start}~) · 줄 세운 달 {len(fds)}개 ({fds[0].date()}~{fds[-1].date()})", flush=True)
    pm, fail = download(tickers, start)
    if fail:
        print(f"   받지 못함: {', '.join(fail)}")
    print("[2/2] 채점 (방법은 engine/peer_eval.py PEER_EVAL에 고정)", flush=True)
    res = engine.evaluate_peer_ranks(rk, pm)
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    rep = engine.render_peer_eval_report(res, generated_at=now)
    if fail:
        rep += f"\n- 받지 못한 종목(계산에서 빠짐): {', '.join(fail)}\n"
    _save("PEER_EVAL.md", "peer_eval.json", rep, {"generated_at": now, **res, "failed_downloads": fail})
    print("\n" + rep)
    print(f"끝 ({time.time() - t0:.0f}초). 보고서: {os.path.join(OUT_DIR, 'PEER_EVAL.md')}")


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
    ap.add_argument("--peer-eval", action="store_true", help="피어 순위 채점")
    ap.add_argument("--short-term", action="store_true", help="단기 규칙(+3%% 익절) 검증")
    ap.add_argument("--levels", action="store_true", help="지지선·손절선 검증")
    ap.add_argument("--reclaim", action="store_true", help="지지선 이탈 뒤 회복 검증")
    ap.add_argument("--explore", metavar="NAME", help="가설 탐색(이미 쓴 종목, 증거 아님). list = 가설 이름 보기")
    ap.add_argument("--refresh", action="store_true", help="--explore에서 저장해 둔 가격을 쓰지 않고 다시 받기")
    ap.add_argument("--lecture", metavar="WHAT", help="강의 조건 시험: data(자료 받기) · L1(BNF) · L2(매수 사건 조합) · L3(상태별 RSI) · L4(매도·약화) · L5(레벨)")
    ap.add_argument("--mr1", action="store_true", help="H-MR1 단기 반등 단타 탐색(규칙 A·B·C, 기존 종목 2001~2019)")
    ap.add_argument("--etf-ok", action="store_true", help="--lecture L1·L2: 한국 섹터 ETF 이름을 확인했음")
    ap.add_argument("--mr2", action="store_true", help="H-MR2 단기 신호 찾기(26개 규칙, 기존 종목 2001~2019 → 후보만 2020~ 검증)")
    ap.add_argument("--mr2-final", action="store_true", help="H-MR2 최종(검증 통과 규칙만, 새 151종목)")
    ap.add_argument("--names-ok", action="store_true", help="--mr2-final·--mr3-final: 한국 종목 이름을 확인했음")
    ap.add_argument("--mr3", action="store_true", help="H-MR3 변동성 국면(VIX ≥ 20) 이격도·지지 반등 찾기 → 후보만 검증")
    ap.add_argument("--mr3-final", action="store_true", help="H-MR3 최종(검증 통과 규칙만, 새 151종목)")
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
    if args.peer_eval:
        run_peer_eval_only(engine, args, t0)
        return
    if args.short_term:
        run_short_term_only(engine, args, t0)
        return
    if args.levels:
        run_levels_only(engine, args, t0)
        return
    if args.reclaim:
        run_reclaim_only(engine, args, t0)
        return
    if args.explore:
        run_explore_only(engine, args, t0)
        return
    if args.lecture:
        run_lecture_only(engine, args, t0)
        return
    if args.mr1:
        run_mr1_only(engine, args, t0)
        return
    if args.mr2:
        run_mr2_only(engine, args, t0)
        return
    if args.mr2_final:
        run_mr2_final_only(engine, args, t0)
        return
    if args.mr3:
        run_mr3_only(engine, args, t0)
        return
    if args.mr3_final:
        run_mr3_final_only(engine, args, t0)
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
