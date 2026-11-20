# 투자터미널 — AI 작업 규칙과 인수인계 (Claude · Gemini 공용)

이 파일이 이 프로젝트 규칙의 정본이다. 어떤 AI든 코드를 고치기 전에 이 파일을 먼저 읽는다.
매매 판정의 설계 방향은 4번 "매매 틀과 운용 규칙"에 있다. 작업 이력은 git 커밋 기록으로 남긴다(이 파일에는 이력을 적지 않는다).

## 1. 프로젝트 한눈에
- Streamlit 앱. 실행: `streamlit run app.py`
- `app.py` — 홈 화면(결론 → 시장 상태 → 섹터 흐름 → 자금 흐름 → 실행 레이더 → 데이터 점검)
- `engine/` — 계산 엔진 패키지(원래 engine.py 한 파일, 2026-11-11 기능별 12개 모듈로 분리). 바깥에서는 지금처럼 `import engine` → `engine.이름`으로 쓴다. 모듈 지도는 3번 맨 위
- `utils.py` — 공통 CSS, 사이드바 메뉴(`init_session`)
- `pages/1_🔎_종목_딥다이브.py` — 종목 딥다이브
- `pages/3_📒_매매_기록.py` — 매매 기록(내 실제 매매, PC에만 저장 `data/journal/` — git 제외). 산 기록은 딥다이브 '매매 기록에 추가'에서 적으면 그날 판정이 함께 남는다
- `check_earnings_data.py` — 실적 트랙 실제 데이터 점검(PC에서 `python check_earnings_data.py AAPL 005930.KS`)
- `ledger_daily.py` — 신호 기록 장부: 오늘의 틀 A v2 판정 기록(매일 GitHub Actions가 컨센서스 기록 다음에 실행, 결과 `data/ledger/frame_a_<연도>.csv`). 채점은 `run_backtest.py --ledger-eval`
- `check_playbook.py` — 매매 틀 v2(틀 A) 판정 실제 데이터 점검(PC에서 `python check_playbook.py NVDA 005930.KS --gate 1`)
- `run_backtest.py` — 판정 규칙 백테스트(PC에서 실제 야후 데이터로 실행, 결과 `data/backtest/REPORT.md`·`summary.json`, 모드별 보고서는 파일 맨 위 설명 — 단기 규칙은 `--short-term` → SHORT_TERM.md)
- `peers_daily.py` — 피어 비교 기록(매일 GitHub Actions 4-4, 결과 `data/peers/peer_fund_<연도>.csv`(재무, 7일마다)·`peer_rank_<연도>.csv`(순위, 매일)). 처음 한 번은 `--full`. 점검 `check_peers.py`(묶음별 요약 → `data/peers/peer_census_날짜.md`). 피어 묶음 수정은 `data/peers/peer_map.csv`
- `snapshot_daily.py` — 컨센서스·목표가 날짜별 기록 실행(매일 GitHub Actions `daily_agent.yml`이 실행, 결과 `data/snapshots/consensus_<연도>.csv`). 관심 종목은 `data/snapshots/watchlist.txt`에 한 줄에 하나씩 추가
- `tests/` — 회귀 테스트(가짜 데이터·오프라인). `python tests/run_all.py`로 한 번에 실행
- `.streamlit/config.toml` — Streamlit 기본 페이지 메뉴 숨김(사이드바 메뉴는 utils.py의 st.page_link)

## 2. 데이터 무결성 규칙 (가장 중요)
1. **화면에 보이는 숫자·판정은 실제 데이터에서만 만든다.** 난수(np.random), 사인파, 손으로 적은 숫자를 실제 데이터처럼 화면에 쓰지 않는다.
2. 실제 출처가 아직 없는 항목은 `is_sample: True`, `source: "SAMPLE_DATA ..."`, `as_of: None`으로 표시한다. 홈 화면은 이런 항목을 기본으로 숨긴다("예시 데이터 보기" 토글).
3. 데이터를 못 받으면 "데이터 없음"을 보여준다. 예시 데이터로 대신 채우지 않는다.
4. `as_of`는 **데이터의 날짜**(예: 금요일 종가면 금요일)다. 계산한 날짜나 오늘 날짜를 넣지 않는다.
5. 같은 지표는 한 함수에서만 계산하고 모든 화면이 그 결과를 쓴다(단일 출처).
6. 임계값은 설정 딕셔너리 한 곳에만 둔다: `GATE_CONFIG`(국면 게이트), `RRG_CONFIG`(4분면), `DATA_FRESHNESS`(신선도).

## 3. engine/ 주의사항
### 모듈 지도 (고칠 함수가 어느 파일에 있는지)
| 파일 | 내용 |
|---|---|
| `common.py` | TECH_PARAMS·레짐 허용 매트릭스, 가격 표기(fmt_price·with_paren)·호가, 실적 발표일, NO_DATA, PROJECT_ROOT |
| `technical_signals.py` | T 트랙 기초 지표, 환경·상대강도·추세 신호, 생애주기 신호 |
| `technical.py` | T 트랙 7상태 기계, 셋업·리스크·사이징, V 타이밍, `analyze_technical_track` |
| `backtest.py` | T 셋업 과거 검증(`run_setup_backtest_validation`, `_bt_summary`) |
| `earnings.py` | 실적 트랙, DART, `.env` 읽기(`_read_env_value`) |
| `macro.py` | 거시 트랙, FRED·ECOS(`fetch_macro_levels`), 10년 국채 금리(`fetch_risk_free_10y`) |
| `fundamental.py` | 가치(V) 트랙, 피어·PER 밴드 |
| `synthesis.py` | 4트랙 종합 판정(`synthesize_4tracks`), PRICE_QUAL_LABELS |
| `home.py` | 홈 대시보드, DATA_FRESHNESS·SAMPLE_DATA, 시장 헤더(`build_regime_vix_header`) |
| `gate.py` | 국면 게이트(GATE_CONFIG, 히스테리시스) |
| `rrg.py` | 4분면(RRG_CONFIG, `build_real_rrg_bundle`) |
| `radar.py` | 실행 레이더(RADAR_CONFIG, 종목 선정·T 신호) |
| `rule_backtest.py` | 판정 규칙 백테스트(BACKTEST_CONFIG): 국면 게이트 재현·선행 성과·노출 전략, T 셋업 × 게이트 |
| `t_backtest.py` | T 트랙 검증(TBT_CONFIG): 실제 스캐너 날짜별 재현·티켓 청산, 재설계 후보 T-A 추세 추종·T-B 눌림목(관행값 고정, 해외 국가 ETF로 판정) |
| `snapshot.py` | 컨센서스·목표가 날짜별 기록(SNAPSHOT_CONFIG·SNAPSHOT_COLUMNS): 야후에서 받아 연도별 CSV에 붙임, `load_snapshots`로 읽음 |
| `sec_facts.py` | SEC 공시일 기준 TTM EPS(SEC_CONFIG): 티커→CIK, EPS 개념 받기·캐시(data/sec_cache/), 4분기 = 연간 − 9개월, 정정 값은 제출일부터, 액면분할 단위 맞춤 |
| `playbook.py` | 매매 틀 v2 판정(PLAYBOOK_CONFIG): `frame_a_evaluate`(추세 상태·근거 저울 7항목·전환 가능성·의견·청산), `frame_b_evaluate`(적용 범위·단계), `fetch_frame_a_live`(실제 데이터), 점검은 check_playbook.py |
| `peers.py` | 피어 비교(PEER_CONFIG·PEER_METRICS): `build_groups`(peer_map + 야후 산업, `yf:` 합류), `derive_metrics`·`percentile_ranks`·`strategy_score`·`score_group`(사전 등록 순위 규칙), `run_daily_peers`(재무 주 1회·가격 매일 기록), `latest_peer_view`(딥다이브용), `peer_group_view`·`peer_groups_overview`(피어 비교 페이지·홈용, 내 종목 = 장부 대상) |
| `strategy.py` | 전략 선택(SHORT_TERM_CONFIG·STRATEGY_VERIFICATION): `short_indicators`·`short_term_evaluate`(단기 규칙 — 지지선·손절선·내일 시가 허용 범위), `support_stop_levels`(지지선·손절선·거리·매수 비중 한도 — 진입 신호 아님, LEVEL_CONFIG·LEVEL_VERIFICATION), `choose_strategy`(공통 차단 → 가치 > 추세), 세 방식 한눈에 `strategy_fit`(결론 한 줄 · 가치·추세·단타 한 줄 이유 · 가장 가까운 방식)·`strategy_reason_value`·`strategy_reason_trend` |
| `short_backtest.py` | 단기 규칙 검증(SHORT_BT_CONFIG·SHORT_TERM_CRITERIA): `simulate_short_trade`·`short_trades`, 보유 기간 맞춘 무작위·'눌림 없이' 비교, `evaluate_short_term`·`render_short_term_report` (run_backtest.py --short-term) |
| `level_backtest.py` | 지지선·손절선 검증(LEVEL_BT_CONFIG·LEVEL_CRITERIA·LEVEL_VERDICT_TEXT): `support_events`·`support_placebo`(같은 ATR 거리의 가짜 선), `stop_events`·`stop_baseline`, `evaluate_levels`·`render_levels_report` (run_backtest.py --levels) |
| `reclaim_backtest.py` | 지지선 이탈 뒤 회복 검증(RECLAIM_BT_CONFIG·RECLAIM_CRITERIA·RECLAIM_VERDICT_TEXT): `approach_outcomes`(버팀·깼다가 회복·못 돌아옴 집계), `break_events`·`reclaim_events`, `evaluate_reclaim`·`render_reclaim_report` (run_backtest.py --reclaim) |
| `explore.py` | 가설 탐색 도구(EXPLORE_CONFIG·EXPLORE_HYPOTHESES): `run_explore`(사건 다음 날 시가 → 10거래일 수익률을 무작위 날·하락 폭 맞춘 날과 비교, '후보' 판정), `render_explore_report`, `append_explore_log`, 가격 캐시 `load_price_cache`·`save_price_cache` (run_backtest.py --explore 이름) |
| `touch_explore.py` | 탐색: 수평선 터치 횟수별 뚫림(TOUCH_CONFIG) — `touch_events`(스윙 저점·고점 수평선, 터치 세기, 뚫림), `run_touch_explore`·`render_touch_report` (run_backtest.py --explore touch4) |
| `momentum_explore.py` | 탐색: 모멘텀(MOM_CONFIG) — `stock_month_rows`(매달 신호·1·3개월 결과), `monthly_spreads`(시장별 상위·하위 20%), `run_momentum_explore`·`render_momentum_report` (run_backtest.py --explore mom12 / ret63) |
| `peer_eval.py` | 피어 순위 채점(PEER_EVAL): `formation_dates`(시작일 이후 매달 첫 기록일), `group_month_rows`(묶음·달·관점·기간마다 상위·하위·묶음 평균·하드 킬), `evaluate_peer_ranks`·`render_peer_eval_report` (run_backtest.py --peer-eval) |
| `ledger.py` | 신호 기록 장부(LEDGER_CONFIG·LEDGER_COLUMNS·LEDGER_EVAL): `run_daily_ledger`(판정+원자료 기록), `load_ledger`, `evaluate_frame_a_ledger`(미리 정한 방법으로 채점) |
| `lecture_signals.py` | 강의 조건 시험 사전 등록 설정(LECTURE_CONFIG)과 기초 신호: `lec_base`(앱 기초 지표 + RSI 바로잡음), `lec_swings`·`lec_state`(상승·횡보·하락), `lec_market_grade`(시장 등급), `lec_rs_line`·`lec_rs_daily`·`lec_rs_weekly_series`(종목 ÷ 섹터 ETF 4구간), `lec_bnf_base_series`(BNF 기준 이격도) |
| `lecture_backtest.py` | 강의 조건 시험 1단계: `run_lecture_l1`(BNF 두 질문)·`run_lecture_l3`(상태별 RSI), `lec_prepare`, 판정 `lec_judge`·`lec_month_drop`·`lec_verify`, 자료 받기 `lec_sec_earnings_dates`·`lec_dart_earnings_dates`, 보고서 (run_backtest.py --lecture data / L1 / L3) |
| `lecture_events.py` | 강의 조건 시험 2단계 L2 정의: 사건 E01~E15(`lec_breakouts`·`lec_double_bottoms`·`lec_ma_double_bottoms`·`lec_transition`·`lec_downline_attempts`·`lec_wedge_triangle`·`lec_channel_retests` 등), 겹침 지지 `lec_support_types`, 종목 하나의 사건 표 `lec_l2_events`, 짝짓기 표 `lec_l2_combos` |
| `lecture_family.py` | 강의 조건 시험 2단계 큰 가족 판정: 조합 고르기 `lec_select`, 섞기 통과선 `lec_reality_check`, `lec_pbo`, 후보 `lec_judge_one`, L2 표·실행 `lec_l2_table`·`run_lecture_l2`, 보고서 (run_backtest.py --lecture L2) |
| `lecture_levels.py` | 강의 조건 시험 2단계 L4·L5: 약화 `lec_weakness`, 쌍고점 `lec_double_tops`·`lec_ma_double_tops`, 붕괴 `lec_collapse`, 상승 추세선·채널 `lec_upline`, `lec_l4_events`·`lec_l4_table`·`run_lecture_l4`, 선 넘음 `lec_hit`·목표 `lec_target_first`, 수평선 `lec_swing_lines`, `lec_l5_events`·`lec_volume_profile_events`·`lec_l5_table`·`run_lecture_l5`, 보고서 (run_backtest.py --lecture L4 / L5) |
| `mr_backtest.py` | H-MR1 기술적 단기 반등 단타(평균 회귀) 시험: `MR1_CONFIG`, 지수 맥락 `mr_index_context`, 신호 `mr_signals`(규칙 A·B·C), 거래 `mr_trades`, `mr_universe`·`mr_judge`·`run_mr1`, 보고서 (run_backtest.py --mr1) |
| `mr2_backtest.py` | H-MR2 단기 신호 찾기(신호 즉시 진입 · +5%/−5%/3거래일): `MR2_CONFIG`·`MR2_SIGNALS`·`MR2_RULES`(26개), 청산 `mr2_bracket_all`·`mr2_after`, 신호 `mr2_signal_arrays`·`mr2_support_zone`, 종목 하나 `mr2_ticker`, 단계 `mr2_stage`, 통계·판정 `mr2_stats`·`mr2_judge_explore`·`mr2_judge_simple`, 통과선 `mr2_reality`, `run_mr2`·`run_mr2_final`, 보고서 (run_backtest.py --mr2 / --mr2-final) |
| `mr3_backtest.py` | H-MR3 변동성 국면(VIX ≥ 20) 이격도·지지 반등: `MR3_CONFIG`·`MR3_RULES`(V1~V5), VIX 날짜 맞춤 `mr3_vix_prev`(앞선 미국 거래일), 청산 `mr3_exit`(2 ATR 손절·25일선 회복·10일), 종목 하나 `mr3_ticker`, 단계 `mr3_stage`(보유 기간 맞춘 비교 기준), `mr3_stats`·`mr3_year_checks`·`mr3_mde`, 판정 `mr3_judge_explore`·`mr3_judge_verify`·`mr3_judge_final`, 통과선 `mr3_reality`, 시점 효과 `mr3_timing`, 1990년대 재료 `mr3_prepare_old`, `run_mr3`·`run_mr3_final`, 보고서 (run_backtest.py --mr3 / --mr3-final), 딥다이브 단타 표시 `mr3_short_status`(마지막 봉의 V5 조건) |
| `tr_backtest.py` | H-TR 추세 투자 흐름(TR_CONFIG·TR_RULES A/B, 사전 등록 TR_PREREG.md): 박스·물량 소화 `tr_box`, 실적 쇼크 `tr_earn_status`, 종목 하나 신호 표 `tr_context`(수익 안 봄), 깔때기 `tr_funnel`, 청산 `tr_exit`, 보유 기간 맞춘 기준 `tr_base`, 거래 `tr_simulate`, 통계·판정 `tr_stats`·`tr_judge_explore`·`tr_judge_verify`·`tr_judge_final`, 실행 전 `run_tr_count`(T1 결정 `tr_absorb_decision`·최소 차이 `tr_noise`), `run_tr`·`run_tr_final`, 보고서 (run_backtest.py --tr-count / --tr / --tr-final) |
| `journal.py` | 매매 기록(JOURNAL_CONFIG·JOURNAL_TRADE_COLUMNS·JOURNAL_SELL_COLUMNS, 채점 규칙 data/backtest/JOURNAL_PREREG.md): 저장 `journal_storage_mode`(PC = local, 인터넷 서버 /mount/src = view)·`journal_load`·`journal_save`, 입력 `journal_add_trade`·`journal_add_sell`·`journal_update_trade`·`journal_delete`·`journal_import`, 그날 판정 `journal_verdict_snapshot`, 결과 `journal_results`(비용 포함·지수 대비·손절 `journal_stop_status`), 채점 `journal_judge`·`journal_style_summary`·`journal_verdict_compare`, 세금 `journal_tax_overseas`·`journal_settle_date`·`journal_kr_sell_tax_rate` |
| `frame_backtest.py` | 매매 틀 검증. 틀 A(FRAME_A_CONFIG·FRAME_A_CRITERIA): 개별 종목, 보유 기간 맞춘 무작위와 비교. 틀 B(FRAME_B_CONFIG·FRAME_B_CRITERIA): 미국 개별 종목, SEC EPS로 만든 자기 PER 밴드, 같은 날 다른 종목 평균과 비교. 둘 다 월 묶음 표준오차 |

### 패키지 규칙
- 새 함수는 기능이 맞는 모듈에 넣는다. `engine.py` 파일을 다시 만들지 않는다(폴더 `engine/`이 우선 로드되어 새 파일은 무시됨).
- 다른 모듈 이름을 쓰면 파일 위쪽에 `from .모듈 import 이름`을 추가한다. 의존 방향은 아래 순서의 **앞쪽 모듈만** 가져다 쓴다(거꾸로 가져오면 순환 import 오류): common → technical_signals → backtest → technical → earnings → macro → fundamental → synthesis → home → gate → rrg → radar → rule_backtest → t_backtest → snapshot → sec_facts → frame_backtest → playbook → ledger.
- 강의 시험 모듈은 맨 뒤다: `lecture_signals` → `lecture_backtest` → `lecture_events` → `lecture_family` → `lecture_levels` → `mr_backtest` → `mr2_backtest` → `mr3_backtest` → `tr_backtest` → `journal`(앞의 모든 모듈을 가져다 쓸 수 있고, 다른 모듈은 이 열을 가져오지 않는다). 새 이름은 `lec_`·`LEC_`(H-MR은 `mr_`·`mr2_`·`mr3_`, H-TR은 `tr_`·`TR_`·`_tr_`, 매매 기록은 `journal_`·`JOURNAL_`·`_jr_`)로 시작한다(같은 이름 충돌 방지 — 밑줄로 시작하는 내부 이름도 모든 모듈에서 하나여야 함).
- 바깥(app.py·페이지·테스트)은 `engine.이름`으로만 쓴다. 새 이름도 `engine/__init__.py`가 자동으로 내보낸다(따로 등록할 필요 없음).
- 테스트에서 `engine.이름 = 가짜함수`로 바꿔 끼우면 그 이름을 쓰는 모든 모듈에 같이 적용되고, `importlib.reload(engine)`은 모든 모듈을 처음부터 다시 읽는다(한 파일이던 시절과 같은 동작, `engine/__init__.py`의 _EngineModule).
- 같은 이름을 두 모듈에서 따로 정의하면 import 시 오류로 멈춘다(단일 출처 보호).
- 파일 경로는 실행 폴더 기준 상대경로(`data/`, `logs/`)이고, `.env`만 `PROJECT_ROOT` 기준이다.

### 계산 규칙
- 게이트 관련 함수·설정(GATE_CONFIG 등)은 한 번씩만 정의되어 있다. 새로 같은 이름을 또 정의하지 않는다.
- `_build_default_rrg_sample_dataset()`은 합성 데이터 생성 함수다. **테스트 전용**이며 화면에서 호출하지 않는다.
- 실제 4분면은 `build_real_rrg_bundle(etf_map, baskets)`(engine/rrg.py)로 만든다.
- 4분면 그래프(app.py `build_rrg_figure`)는 축 밖의 점을 ±3σ 모서리로 끌어오지 않는다(가짜 꺾임). 오늘 위치가 ±3σ 밖이면 화면 범위만 넓히고(최대 `AXIS_LIMIT_MAX`), 사분면 판정은 바꾸지 않는다. 라벨은 `place_labels`의 오프셋을 xshift·yshift로 적용한다(plotly는 showarrow=False일 때 ax·ay를 무시). 꼬리는 오래된 구간일수록 흐리게 그리고, 꼬리 점에 마우스를 올리면 몇 주 전·날짜·위치가 나온다. 오른쪽 카드 문구는 `describe_rrg_tail`(engine/rrg.py, 화면 설명용 — 판정에 쓰지 않음)로 꼬리를 풀어 쓴다: 지나온 칸, 최근 1주 X·Y 변화(±0.25σ 미만은 '비슷'), 최근 1주 움직임이 1주 더 이어지면 닿는 칸('이 흐름이 1주 더 이어지면: ○○ 칸으로 넘어감/머묾', 예측이 아님). 좌표 숫자는 카드에 쓰지 않는다. tests/test_rrg_chart.py가 확인한다.
- 백테스트(engine/rule_backtest.py)는 게이트를 따로 계산하지 않고 앱과 같은 함수(build_regime_vix_header → build_gate_inputs_from_header → evaluate_market_gate)를 날짜마다 그날 이전 3년 데이터로 호출한다(앱 period="3y"와 같게). 한국 게이트는 미국 당일 데이터(QQQ·VIX)를 쓰지 않는다. 게이트 함수를 고치면 백테스트도 자동으로 따라간다.
- 백테스트 결과를 보고 같은 데이터로 임계값(GATE_CONFIG 등)을 다시 맞추지 않는다(과최적화). 바꿀 때는 기간을 나눠 앞 기간으로 정하고 뒤 기간으로 확인한다. 노출 비중(EXPOSURE_BY_GATE)은 모델 가정이다.
- 백테스트로 확인된 사실(2026-11-12, data/backtest/REPORT.md): 게이트는 이후 변동성·낙폭을 구분하지만(미국 G3 변동성 G0의 약 2배) 이후 수익률은 구분하지 못한다. G3는 기간의 41~48%이며 주원인은 지수 < EMA200(레짐 ⚫ → 즉시 G3). 화면·문구에서 게이트를 수익 예측처럼 표현하지 않는다. T 셋업(레이더 과거 검증의 단순화 규칙)은 ETF 기준 우위가 확인되지 않았고, ST-1 신호 대부분이 G3에서 나와 'G3 신규 매수 동결'과 충돌한다.
- 규칙 후보 비교(2026-11-13, data/backtest/VARIANTS.md): 게이트 후보 A·B는 사전 기준(앞 기간 G3 비중 ≤20%)에서 탈락해 현재 규칙 유지, ST-1의 G2·G3 20% 예외도 기각. 결과를 본 뒤 기준(20% 등)을 풀거나 뒤 기간 성적으로 후보를 다시 고르지 않는다. 새 후보를 시험하려면 GATE_VARIANTS에 먼저 추가하고 기준을 바꾸지 않은 채 run_backtest.py --variants로 판정한다.
- 한국 후보 C(KOSPI 자체 변동성)도 사전 기준 탈락(2026-11-13). 뒤 기간(2014~2026)을 여러 번 봐서 더는 깨끗한 검증 데이터가 아니므로 게이트 후보를 더 추가해 고르지 않는다. 게이트를 바꾸려면 앞으로 쌓이는 데이터(2026-11 이후)로 검증한다.
- T 셋업 장기 검증(data/backtest/setup_diag.json, run_backtest.py --setup-diag): ST-1·ST-2 모두 '같은 매매 방식의 무작위 진입'보다 낫지 않음(미통과). 레이더 카드는 engine.load_setup_validation으로 이 파일을 읽어 표시하며 숫자를 코드에 적지 않는다. 파일이 없으면 표시하지 않는다.
- T 트랙 재설계(2026-11-13~): 실제 스캐너는 run_backtest.py --scanner(앱 레이더와 같은 입력으로 날짜별 재현, 캐시 data/backtest/scanner_cache/), 재설계 후보는 --redesign. 후보 규칙·매개변수(55·20일, 2×ATR(20), RSI(2)<10, 5일선)와 판정 기준(TBT_CRITERIA)은 결과를 보기 전에 고정했으므로 결과를 보고 바꾸지 않는다. HOLDOUT_UNIVERSE(해외 국가 ETF)는 판정 전용이라 다른 분석에 쓰지 않는다. 결과와 이후 방향은 4번 '검증 상태'·'다음 작업 순서' 참고.
- pandas 2·3 모두에서 돌아가게 쓴다(PC는 Python 3.14 + pandas 3, GitHub Actions는 Python 3.11). pandas 3은 문자열 열의 None을 NaN으로 바꾸므로 '값 없음'을 None으로 다뤄야 하는 열은 `dtype=object`로 만들고, 비교는 `is None` 대신 `pd.isna`(또는 playbook의 `_isna`)로 한다. 날짜 열을 합칠 때(merge_asof 등)는 정밀도(초·마이크로초·나노초)를 먼저 맞춘다.
- 신호 기록 장부(engine/ledger.py, 2026-11-15~): 컨센서스 기록과 같은 규칙(원자료도 함께 저장, 빈칸은 빈칸, 열 순서 고정·새 열은 맨 뒤, 손으로 고치거나 지우지 않음). 줄마다 판정 규칙 버전(`PLAYBOOK_VERSION`)을 저장하고 채점도 버전별로 한다 — **engine/playbook.py의 규칙이나 숫자를 바꾸면 PLAYBOOK_VERSION을 반드시 올린다.** 채점 방법(LEDGER_EVAL)은 결과를 보고 바꾸지 않는다.
- 매매 기록(engine/journal.py, 2026-11-20~): 사용자의 실제 매매라 **`data/journal/`은 절대 커밋하지 않는다**(.gitignore, 공개 저장소 — tests/test_journal.py 1-6이 확인). 저장은 PC에서만(인터넷 서버는 CSV를 불러와 보기만). 기록은 사용자가 화면에서 적고 고치며, AI가 손으로 만들거나 고치지 않는다. 채점 규칙(JOURNAL_PREREG.md: 끝난 거래 30건부터, 지수 대비 초과의 월 묶음 t ≥ 2 / ≤ −2, 손절 준수 = 계획 − 1%p 안, 갭은 따로)은 기록을 보고 바꾸지 않는다 — 바꾸면 JOURNAL_CONFIG['VERSION']을 올린다. 세율·공제(JOURNAL_CONFIG)는 세법이 바뀌면 출처와 함께 고친다.
- 피어 비교 기록(engine/peers.py, 2026-11-15~): 장부와 같은 규칙(열 순서 고정·새 열은 맨 뒤, 손으로 고치거나 지우지 않음, 줄마다 PEER_CONFIG['VERSION']). 순위 규칙(가중치·요인·백분위·걸러내기)을 바꾸면 VERSION을 올린다. 손으로 고치는 파일은 `data/peers/peer_map.csv`(피어 묶음)와 `data/snapshots/watchlist.txt`(커버리지)뿐이다.
- 컨센서스 기록(engine/snapshot.py, 2026-11-14~): 판정 결과가 아니라 원자료를 저장한다. 못 받은 값은 빈칸(전날 값·추정으로 채우지 않음). `snapshot_date`는 받은 날(한국 날짜), `price_as_of`는 가격 데이터의 날짜. 열은 SNAPSHOT_COLUMNS 순서로 고정이며 새 열은 맨 뒤에만 추가한다(이미 쌓인 CSV와 어긋나지 않게). 쌓인 CSV는 손으로 고치거나 지우지 않는다 — 다시 만들 수 없는 자료다. daily_agent.yml은 Python 3.11로 돌므로 engine/ 코드는 3.11 문법을 지킨다.
- 틀 A 검증(engine/frame_backtest.py, run_backtest.py --frame-a): 규칙·매개변수(6·12개월 상대강도, 200·50일선, RSI(2)<10, 60일 최저가)와 판정 기준(FRAME_A_CRITERIA)은 실제 데이터 결과를 보기 전에 고정했다. 결과를 보고 바꾸지 않는다. 무작위 비교는 반드시 **보유 기간을 맞춘다**(주가는 평균적으로 오르므로, 같은 청산 규칙을 무작위 날짜에 쓰면 보유가 짧아져 우위가 없는 가짜 가격에서도 통과로 나온다 — tests/test_frame_backtest.py 4-7이 이를 막는다). HOLDOUT_UNIVERSE(새 개별 종목)는 판정 전용이다.
- 틀 B 검증(run_backtest.py --frame-b, .env에 `SEC_USER_AGENT=이름 이메일` 필요): 규칙(자기 5년 로그 PER 밴드 ±1σ, TTM EPS 전년 대비, 배수·이익 분해)과 판정 기준(FRAME_B_CRITERIA)은 실제 데이터 결과를 보기 전에 고정했다. EPS는 제출일 다음 거래일부터 쓴다. 보유가 수백 일인 틀은 '같은 종목 무작위 날짜 + 같은 보유일' 비교가 치우치므로(보유일이 그 종목 경로에 달림, 가짜 가격에서 −6%p) **같은 날 사서 같은 날 판 같은 묶음 다른 종목 평균**과 로그 수익률로 비교한다 — tests/test_frame_b.py 6-1·6-2가 우위 없는/있는 가짜 가격으로 이를 확인한다. FRAME_B_CONFIG HOLDOUT_UNIVERSE는 판정 전용이다.
- 단기 규칙 검증(engine/short_backtest.py, run_backtest.py --short-term): 규칙(engine/strategy.py SHORT_TERM_CONFIG)과 판정 기준(SHORT_TERM_CRITERIA, z ≥ 3)은 결과 전에 고정했다. 보유 기간 맞춘 무작위와 비교하고, tests/test_strategy.py 8·8-2가 우위 없는 가짜 가격(무작위 걸음)에서는 통과하지 않고 되돌림이 있는 가짜 가격에서는 무작위보다 낫게 나오는지 확인한다. SHORT_BT_CONFIG HOLDOUT_UNIVERSE는 판정 전용이다.
- 지지선·손절선 검증(engine/level_backtest.py, run_backtest.py --levels): 선의 정의·판정 기준(LEVEL_CRITERIA)·결과별 화면 문구(LEVEL_VERDICT_TEXT)는 결과 전에 고정했다. 지지선은 '최근 하락 폭이 비슷한 날에 같은 ATR 거리로 그은 가짜 선'과 비교한다(무작위 날과만 비교하면 하락 뒤 되돌림 때문에 선이 특별하지 않아도 통과할 수 있다). tests/test_levels.py 5~5-4가 효과 없는 가짜 가격에서는 통과하지 않고, 바닥에서 튕기는 가격·이탈 뒤 더 떨어지는 가격에서는 통과하는지 확인한다. LEVEL_BT_CONFIG HOLDOUT_UNIVERSE는 판정 전용이다.
- 국면 게이트 입력은 `build_gate_inputs_from_header(macro_header, "US"|"KR")`로 홈 헤더의 실제 데이터에서 만든다.
- 화면에서는 '지수 레짐'을 '지수 추세'로 쓰고 결정 요인 옆에 지수 이름과 단계 뜻(예: 나스닥 위기 = 200일 EMA 아래)을 붙인다(app.py `REGIME_PLAIN`). 게이트 상자·레이더 아래 작은 글씨는 기준일(+'미검증')만, 긴 설명은 도움말(?)로 옮긴다.
- 실적 트랙(analyze_earnings_track) 입력은 `fetch_real_earnings_inputs` 한 곳에서만 만든다(미국 yfinance, 한국 DART + yfinance 추정치). 못 받은 값은 None이고 화면에는 '데이터 없음'. 삼성전자식 기본값을 다시 넣지 않는다.
- 실적 트랙은 판정 가능한 축만 점수에 반영한다(`verdict.coverage`). 근거가 부족하면 국면은 `NA`(판정 불가)이며, 계산에서만 G3로 중립 처리하고 화면에는 '판정 불가'로 표시한다.
- V 트랙(analyze_fundamental_track) 입력은 `fetch_real_fundamental_inputs` 한 곳에서만 만든다(미국 yfinance, 한국 DART 연간 재무제표 + yfinance 가격·베타·배당). 못 받은 값은 None이고 화면에는 '데이터 없음'. PER 밴드·순현금 같은 기본값을 다시 넣지 않는다. 모델 상수는 `FUND_MODEL_PARAMS`, 수집 기준은 `FUND_DATA_PARAMS`에만 둔다.
- V 트랙은 근거가 있는 멀티플만 쓴다: 자기 PER 밴드·피어 PER·이론 성숙 PER가 모두 없으면 가격선(buy_tranches)을 만들지 않고 가격 자격 `NA`(종합은 `V_NA`). 현재 PER만으로 목표 배수를 만들지 않는다(순환논리 금지). 자기 밴드는 후행(확정 EPS) 기준이며 업종 PER 이력·산업 증거·주주환원 우위·역대 최고 CAGR은 자동 수집하지 않아 항상 '데이터 없음'이다.
- V 트랙 EPS 우선순위: 컨센서스(`eps_basis=forward`) → 최근 확정 EPS(`ttm_reference`, 화면에 '참고용' 표시) → 없음(`none`, 딥다이브는 안내 후 중단).
- V 트랙 할인율: 10년 국채 금리는 engine.fetch_risk_free_10y(FRED/^TNX) 한 곳에서만 받는다. 고정 금리 값을 다시 넣지 않는다. 시장 위험프리미엄(ERP)은 데이터가 아닌 모델 가정이므로 FUND_MODEL_PARAMS에 출처와 함께 둔다.
- G3 목표 배수에서 이론 성숙 PER은 참고값이다(관측 배수가 없을 때만 대체 사용). 다시 상한(min)에 넣지 않는다.
- 자기 PER 밴드는 로그 PER 기준이다: 중심 self_5y_mean = 기하평균, 폭 self_5y_log_std(로그 σ), kσ 지점 = μ·e^(kσ). 밴드를 쓰는 곳(G2·G3 Bull/Bear, G4 상한, 2차선, 앵커2, z-score)은 모두 이 기준. 산술 μ±σ로 되돌리지 않는다(이익 급변 종목에서 σ가 평균보다 커져 매수선·시나리오가 무너짐).
- 자기 PER 밴드 신뢰 게이트(engine._band_unreliable_reasons): 밴드 기간 안 확정 EPS 최대/최소 > FUND_DATA_PARAMS band_max_eps_change(3배) 또는 로그 σ > band_max_log_std(0.5)이면 자기 밴드는 '데이터 없음'(사유 meta.band_excluded, 월별 PER은 참고 표시). 이때 목표 배수는 피어 기준, Bull/Bear·2차선 폭은 피어 PER σ(log(1+σ/중앙값))로 대체.
- 실적 국면 G3는 저성장 근거(매출 성장 10% 미만)가 있을 때만. 고성장인데 G2 조건 미충족이면 `NA`(성장 단계 판정 보류, 미충족 사유 표시)이고 V는 중립(G3) 규칙으로 계산.
- 피어는 같은 산업 상위 기업만 쓴다(섹터로 넘어가지 않는다: 업종이 다른 기업이 섞여 왜곡). 선행 PER 유효 5개 미만이면 PER·ROE·성장률 피어 통계 모두 '데이터 없음'. ROE·성장률은 PER이 유효한 같은 집합에서 이상치(FUND_DATA_PARAMS peer_roe_range·peer_growth_range) 제외 후 계산. 내재 5년 성장률의 종착 배수는 채택 Base 배수(없을 때만 이론 성숙 PER).
- 거시 트랙(analyze_macro_track) 입력은 `fetch_macro_levels` 한 곳(FRED: FRED_API_KEY 있으면 API·없으면 공개 CSV, 한국 국고채 10년은 ECOS 817Y002/010210000)에서만 받는다. 변수 정의·단위는 `MACRO_SERIES_SPEC`. 못 받은 변수는 회귀에서 빼고 '데이터 없음'(meta.missing_vars), 전부 없거나 주가·지수가 없으면 판정 보류(`_macro_hold_result`, 수치 None). 난수·고정 수준값(환율 1,385원 등)으로 거시 시계열을 만들지 않는다. 점검: `check_macro_data.py`.
- 국면 게이트 입력(레짐·VIX·거시 국면)은 build_regime_vix_header 한 곳에서만 계산한다. 이전 단계는 build_gate_history_state로 만들어 evaluate_market_gate(history_state=...)에 넘긴다.

## 4. 매매 틀과 운용 규칙
### 방향 (v2, 2026-11-14 — v1 검증 결과를 반영해 다시 정함)
두 트랙을 성격이 다른 두 도구로 둔다. 매수 논리 하나를 '진입 → 무효화(손절) → 매도'까지 한 묶음(틀)으로 설계하는 원칙은 그대로다.
- **트레이딩 트랙 = 틀 A v2 '추세 판단과 전환 가능성' — 의견.** 기술 트랙이 추세 상태(상승·하락·횡보)를 정하고, 기술·실적·가치 근거를 저울에 올려 전환 가능성이 어느 쪽으로 기울었는지 알려 준다. 실패 근거가 전환 근거보다 무거워지면 손절한다(가격 폭 손절 없음). 확신이 아니므로 크기를 작게, 화면에서도 '의견'으로만 표시한다.
- **저가 매수 트랙 = 틀 B v2 — 투자 판단.** 가치·실적이 중심이고 기술은 타이밍 보조(위험 관리)만 맡는다.
- **모든 판단에 검증 상태(통과 / 미통과 / 미검증)를 붙인다.** 과거 자료로 우위가 확인된 매매 규칙은 아직 없다. v2의 숫자는 관행값으로 고정하고 과거 자료로 맞추지 않는다. 앞으로 신호 기록 장부(다음 작업 순서 2·3)로 검증한다.
- v1에서 v2로 바꾼 근거 (아래 '검증 상태')
  - 가격만으로 사고팔 시점을 정하는 규칙은 여섯 번 모두 우위가 없었다(틀 A v1 포함, 2026-11-16 단기 규칙까지 일곱 번). 다만 손절선은 꼬리 손실을 줄였다 → 트레이딩은 '신호'가 아니라 '추세 판단 + 근거 저울 의견'으로 바꾸고, 저울이 실패 쪽으로 기울면 빠진다.
  - 틀 B v1(발표된 실적 기준)은 우위 미확인. 손해의 주원인은 산 뒤 이익이 꺾인 종목(거래의 약 70%)이었고, 지난 실적 보고서로는 늦게 알아챘다 → 가치 함정 판정을 애널리스트 추정치 하향으로 앞당긴다.
  - 자기 PER 밴드는 이익이 안정적인 회사에서만 쓸 수 있었다(대표 종목은 25%의 날만) → 적용 범위를 분명히 나눈다.
  - 배수 주도 고평가 매도만 결과가 좋았다(다른 종목 대비 +4~+10%p, 표본 작음) → 매도 규칙으로 유지하되 미검증으로 표시한다.

### 트랙의 역할 (v2)
| 트랙 | 트레이딩 — 틀 A v2 | 저가 매수 — 틀 B v2 |
|---|---|---|
| 기술(T) | **중심**: 추세 상태(스윙 고점·저점 구조), 근거 저울의 기술 항목 4개, 돌발 이벤트 감지 | 보조: 하락 추세 중이면 분할 간격을 넓힘(떨어지는 칼 피하기), 보유 중 위험 경고 |
| 실적(E) | 저울 항목: 추정치 상향 우세 +1, 하향 우세 −1 | **중심**: 이익 훼손 판정(추정치 하향으로 먼저 경고, 발표 실적 감소로 확정), 재평가와 일시적 고평가 구별 |
| 가치(V) | 저울 항목: 자기 밴드 −1σ 이하 +1, +1σ 이상 −1 | **중심**: 매수선(자기 밴드 또는 피어), 고평가 매도 |
| 국면 게이트(M) | 저울에 넣지 않고 크기만 줄인다. 위험 온도계이며 수익 예측 신호가 아니다 | 분할 속도 조절 |

### 틀 A v2 — 추세 판단과 전환 가능성 (트레이딩, 의견)
전환 신호를 찾는 틀이 아니라 **지금 추세가 무엇이고, 전환 가능성이 어느 쪽으로 기울어 있는지 알려 주는 틀**이다. 손절은 가격 폭이 아니라 **실패 근거가 전환 근거보다 무거워질 때** 한다. 숫자는 관행값으로 고정했고 미검증이다.

**1) 추세 상태 (기술) — 기간이 아니라 고점·저점 구조로 본다 (2026-11-14 사용자 정의)**
- 스윙 고점·저점: 직전 극점에서 **3×ATR(20) 이상 되돌리면** 그 극점을 스윙으로 확정한다(움직임의 크기로 정함, 날짜 창 없음). 확정은 되돌림이 나온 날에야 되므로 그 전 날짜의 판정에는 쓰지 않는다.
| 상태 | 기준 |
|---|---|
| 상승 추세 | 최근 두 스윙 저점이 높아짐 **그리고** 종가가 두 저점을 잇는 추세선(로그 가격 직선) 위 |
| 하락 추세 | 상승이 아님(저점 추세선 이탈 또는 저점이 안 높아짐) **그리고** 최근 두 스윙 고점이 낮아짐 **그리고** 종가가 두 고점을 잇는 추세선 아래 |
| 횡보 | 그 외. 세부 표시: '상승 추세선 이탈 — 고점 낮아짐 확인 전', '하락 추세선 돌파 시도 — 저점 높아짐 확인 전', '저점·고점 방향 엇갈림' |
- 스윙이 2개씩 확정되지 않은 종목(조정 없이 한 방향으로만 간 경우 등)은 이동평균 배열(종가·50일선·200일선·50일선 기울기)로 대신 판단하고 '스윙 부족'을 표시한다.
- 화면·로그에는 판단에 쓴 스윙 점(날짜·가격), 연결선의 오늘 값, 아직 확정 전인 진행 중 고점·저점, 이동평균 배열(참고)을 함께 보여 준다.
- 이동평균만으로 정의하던 첫 안은 삼성전자(2026-11)처럼 고점·저점이 이미 낮아지는데도 50일선·200일선이 아직 교차하지 않아 '횡보'로 늦게 판단해서 바꿨다.

**2) 근거 저울 (매일 다시 계산)** — 항목마다 전환 쪽(+), 실패 쪽(−), 중립(0) 중 하나. 같은 사실을 양쪽에 두 번 세지 않는다.
| 항목 | 전환 쪽 (+1) | 실패 쪽 (−1) | 트랙 |
|---|---|---|---|
| 50일선 | 종가가 50일선 위 | 종가가 50일선 아래 | 기술 |
| 20일선 기울기 | 20일선이 5거래일 전보다 높음 | 낮음 | 기술 |
| 스윙 저점 | 최근 두 스윙 저점이 높아짐 | 낮아짐 | 기술 |
| 상대강도 | 3개월 수익률이 시장(미국 SPY·한국 KOSPI)보다 높음 | 낮음 | 기술 |
| 이익 추정치 | 30일간 FY1 추정치 상향 건수 > 하향 건수 | 하향 > 상향 | 실적 |
| 가격 수준 | 자기 PER 밴드 −1σ 이하(싼 상태에서 출발) | +1σ 이상(이미 비쌈) | 가치 |
| **돌발 이벤트** | — | 최근 5거래일 안에 시가 갭 −5% 이상, 또는 하루 하락폭이 2.5×ATR(20) 이상 → **−2** | 기술 |
- 국면 게이트는 저울에 넣지 않는다(검증 결과 수익을 예측하지 못하는 위험 온도계) — 크기 조절에만 쓴다.
- 데이터가 없는 항목은 중립(0)으로 두고 '데이터 없음'을 표시한다.

**3) 전환 가능성 표시** — 차이 = 전환 근거 합 − 실패 근거 합
| 차이 | 표시 |
|---|---|
| +2 이상 | 전환 가능성 우세 |
| −1 ~ +1 | 팽팽 |
| −2 이하 | 실패 가능성 우세 |
- 하락·횡보 상태에서는 '상승 전환 가능성', 상승 상태에서는 같은 저울을 뒤집어 '하락 전환 가능성'으로 보여 준다(보유자에게는 매도 경고).
- **확률(%)로 표시하지 않는다.** 근거의 개수를 비교한 것이지 맞힐 확률을 잰 것이 아니다. 신호 기록 장부가 쌓이면 '차이 +3일 때 실제로 몇 %가 상승 추세가 되었나'를 재서 그때 빈도로 표시한다. 항목 가중치를 같게 둔 것도 맞춘 값이 아니라 단순화다.

**4) 의견의 생애 (진입 → 보유 → 청산)**
| 단계 | 기준 |
|---|---|
| 상승 전환 의견 (진입) | 상태가 하락 또는 횡보 **그리고** 차이 +2 이상 **그리고** 종가가 50일선 위(기술 핵심 근거는 반드시 전환 쪽). 다음 날 시가 |
| 추격 주의 | 종가가 20일선 + 2×ATR(20)보다 높으면 의견은 유지하되 '추격 주의'를 붙인다 |
| 보유 | 매일 저울을 다시 잰다. 실패 근거 합 ≤ 전환 근거 합이면 보유 |
| 청산 = 저울이 실패 쪽으로 기움 | 실패 근거 합 > 전환 근거 합이 되면 다음 날 시가에 청산. 손실이면 '손절(전환 실패 우세)', 이익이면 '매도(하락 전환 우세)'로 표시. 돌발 이벤트는 −2라서 저울을 빠르게 기울인다 |
| 크기 | 변동성 기준: T 슬리브 × RISK_PER_TRADE(1%) ÷ 2×ATR(20). 차이 +3 이상은 그대로, +2는 절반. 게이트 단계로 추가 축소, 종목당 최대 비중 상한 유지 |
- 가격 폭 손절이 없으므로 한 번에 잃을 수 있는 최대 손실을 미리 정할 수 없다. 급락은 '돌발 이벤트'가, 천천히 빠지는 경우는 50일선·저점·기울기 근거가 차례로 실패 쪽으로 넘어가며 잡는다. 그래서 크기는 손절 폭이 아니라 변동성(ATR)으로 정하고 작게 유지한다.
- 이 틀은 '의견'이다. 화면에 '매수 확정'·'확신' 같은 표현을 쓰지 않는다.

### 틀 B v2 — 저평가 매수 (투자 판단)
매수 논리: "가격이 가치보다 싸고, 이익이 꺾이고 있지 않다."
| 단계 | 기준 |
|---|---|
| 적용 범위 | 자기 PER 밴드를 믿을 수 있는 종목(밴드 기간 EPS 변화 ≤ 3배, 로그 σ ≤ 0.5)은 자기 밴드로 판정. 아니면 '자기 밴드 판정 불가'를 표시하고 피어 배수·이익 성장 대비 배수로 판정(미검증) |
| 진입 | V 가격 자격(매수선 도달) **그리고** 이익 훼손 없음 = 발표 실적 TTM EPS가 1년 전보다 줄지 않음 **그리고** 추정치 하향 우세가 아님 |
| 추정치 하향 우세 (정의) | 30일간 FY1 EPS 추정치 하향 건수 > 상향 건수 **그리고** FY1 추정치가 30일 전보다 낮음(야후 eps_trend·eps_revisions, 매일 기록은 data/snapshots/) |
| 경고 = 신규 매수 중단 | 보유 중 추정치 하향 우세가 시작됨 |
| 손절 = 논리 무효화 | 추정치 하향으로 적정가가 현재가 아래로 내려감(= 추정치 하향 우세이면서 V 가격 자격이 '미충족' — V 매수기준선은 최신 추정치로 다시 계산되므로) / 발표 실적 TTM EPS 감소 확정 / 킬 조건(HARD_KILL·VALUE_TRAP) |
| 매도 = 고평가 | 자기 밴드 +1σ 이상이고 진입 뒤 상승이 배수 주도(배수 상승분 > 이익 증가분)면 분할 매도. 이익 주도면 재평가로 보고 보유(미검증, v1 참고 결과 양호) |
| 기술의 역할 | 종가 < 200일선이고 50일선이 하락 중이면 분할 간격을 넓힌다. 수익 신호가 아니다 |
- **재평가와 일시적 고평가의 구별**: 주가 상승을 이익 증가분과 배수 상승분으로 나눈다(`multiple_decomposition`). 이익(추정치) 증가가 더 크면 재평가(적정가를 올려 잡고 보유), 이익은 그대로인데 배수만 올랐으면 일시적 고평가(분할 매도).

### v1 (기록용, 2026-11-13~14 검증 후 폐기)
- 틀 A v1 추세반등: 6·12개월 상대강도 + 200일선 위 + RSI(2) < 10 눌림 진입, 60일 최저가·200일선 이탈 손절, 50일선 확인 뒤 이탈 매도 → 미통과(FRAME_A.md).
- 틀 B v1 저평가 매수: 발표 실적 TTM EPS로 만든 자기 밴드 −1σ + TTM 감소 없음 진입, TTM 감소 손절, 배수 주도 고평가 매도 → 우위 미확인(FRAME_B.md).

### 코드 반영 상태 (이 표와 다르면 코드가 현재 동작이다)
| 항목 | 상태 |
|---|---|
| V 매수선·가격 자격, 킬 조건(HARD_KILL·VALUE_TRAP) | 있음 — engine/fundamental.py, engine/synthesis.py. 재무 생존성은 v2(2026-11-15, 아래 '재무 생존성 v2') |
| 주가 변화의 이익·배수 분해 | 있음(후행 EPS 기준 12개월, `multiple_decomposition`) — 매도 판정에는 아직 연결 안 됨 |
| 틀 A v1·틀 B v1 검증 | 검증 코드만 있음(engine/frame_backtest.py, engine/sec_facts.py) — 둘 다 사전 기준 미통과, 화면에 넣지 않음 |
| 틀 A v2·틀 B v2 판정 | 있음(2026-11-14) — engine/playbook.py. 딥다이브 종합 탭 맨 위에 카드로 표시(2026-11-15). 틀 B의 '발표 실적'은 V 트랙 확정 EPS 1년 변화(multiple_decomposition 이익 기여분), 자기 밴드 위치는 V 트랙 self_5y_z |
| 추정치 상향·하향(30일) | 틀 A 저울 항목과 틀 B '추정치 하향 우세'에 사용(snapshot.fetch_consensus_row 모양), 날짜별 기록은 engine/snapshot.py |
| 홈 실행 레이더 | 신호 기록 장부의 최신 기록(틀 A v2)을 읽어 세 칸으로 표시(2026-11-15): 상승 전환 의견(진입 의견, 크기 = 기본 배수 × GATE_SIZE[국면]) · 관찰(하락·횡보 중 전환 우세지만 진입 아님, 또는 팽팽). 칸마다 6개를 보여 주고 나머지는 접음, 국면 때문에 크기가 0이면 칸 위에 한 번만 알림, 티커 옆 이름은 utils.TICKER_NAMES(표시용) · 보유 점검(상승 중 하락 전환 우세). 기록이 3일보다 오래되면 경고. 기존 셋업 스캐너(ST-1~4)는 '기존 셋업 레이더 보기' 스위치를 켤 때만 계산(장기 검증 미통과) |
| 종합 판정 | 딥다이브: 틀 A·B 카드 아래에 기존 결정 트리(ARCH_1~8)를 '기존 종합 판정(개편 전 방식)'으로 그대로 둠. 기술 트랙 탭은 맨 위에 틀 A 요약, 그 아래 7상태 기계·셋업·T 티켓을 '기존 셋업 진단 — 장기 검증 미통과'로 표시, 10장 백테스트 상자에 표본·장기 검증 경고 |
| 컨센서스·목표가 날짜별 저장 | 있음(2026-11-14~) — engine/snapshot.py, 매일 자동 실행. 화면에는 아직 쓰지 않음 |
| 피어 비교 | 1단계(2026-11-15) — engine/peers.py, peers_daily.py(Actions 4-4), check_peers.py, 테스트 tests/test_peers.py. 2단계(2026-11-16) — 딥다이브 탭 위 결론 카드·피어 표(`render_peer_section`). 화면 검사는 가짜 피어 기록(tests/_fake_peers.py, PEERMODE=fake/none) |
| 전략 선택 | 4단계(2026-11-16, 단기는 11-17부터 지지·손절선 카드 `render_levels_card`) — engine/strategy.py(단기 규칙 판정·우선순위), engine/gate.py `market_gate_from_prices`(홈과 같은 국면 계산), 딥다이브 피어 비교 아래 '어느 전략으로 들어갈까'(결론 줄 + 카드 3장, `render_strategy_section`). 단기 규칙 과거 검증은 engine/short_backtest.py(run_backtest.py --short-term, 결과 전) |
| 신호 기록 장부 | 있음(2026-11-15~) — engine/ledger.py, 매일 자동 실행(daily_agent.yml 4-3). 틀 A만 기록(틀 B는 가치·실적 결과가 필요해 화면 개편 때 추가) |
| 매매 기록 | 있음(2026-11-20~) — engine/journal.py, pages/3_📒_매매_기록.py, 딥다이브 '매매 기록에 추가'(그날 세 방식 판정 함께 저장). PC에만 저장(data/journal/, git 제외), 인터넷 서버는 CSV 보기만. 채점 규칙 data/backtest/JOURNAL_PREREG.md(jr-1) |

### 재무 생존성 v2.2 (v2 2026-11-15, v2.2 2026-11-16, engine/fundamental.py `SURVIVAL_CONFIG`·`evaluate_survival`)
- 바꾼 이유: AMZN(2026년 9월 말 기준 영업현금 12개월 +1,683억 달러, 설비투자로 FCF −424억 달러, 차입 > 현금)이 v1에서 'FCF 런웨이 0년'으로 하드 킬됐다. v1은 순현금 ÷ FCF 적자로 런웨이를 계산해 영업현금을 보지 않았고, 차입만기는 보유 현금만, 금융업에도 같은 식을 썼으며, 종합 판정은 화면 문구를 글자로 읽어 킬을 판정했다. **정의 오류 수정이며 기준 숫자는 그대로 뒀다.** 하드 킬은 원래 백테스트한 규칙이 아니다.
- 규칙 (결과를 보기 전에 고정):

| 지표 | 계산 | 하드 킬 | 경고 |
|---|---|---|---|
| ① 런웨이 | FCF 적자이고 영업현금 < 0(또는 모름)일 때만: 보유 현금 ÷ FCF 적자 | < 1년 | < 2년 |
| ①' 투자 주도 FCF 적자 | 영업현금 ≥ 0 인데 설비투자로 FCF < 0 | 없음 | 항상 (영업현금·설비투자·FCF·1년 차입 변화 표시) |
| ② 차입 상환 여력 | (보유 현금 + 12개월 영업현금) ÷ 1년 내 만기 차입. 영업현금 모르면 현금만 | 없음(v2.2) | < 1.5배, < 1.0배는 '차환 의존' 경고 (영업현금 모르면 현금만 < 1.0배) |
| ③ 희석 | 희석 주식수 1년 증가율 | > 15% 이면서 영업현금 또는 FCF 적자(새 주식으로 버팀) | > 5%, 또는 > 15%인데 현금흐름 흑자·미확인 |
| 금융업 | yfinance 섹터 'Financial Services' — 자본비율 미연결 | 해당 없음 | 해당 없음 |
| 금융 자회사 포함 연결 | `CAPTIVE_FINANCE` 원칙 목록(현대차·CAT·DE·F·GM·PCAR·TM·HMC·HOG) — 차입·영업현금에 할부·리스·카드 숫자가 섞임 | 판정 보류 | 판정 보류 |

- 경고의 효과는 그대로: 목표 비중 50% 축소, V 가격 자격 최고 '조건부'. 하드 킬은 종합 판정 KILL_SURVIVAL → 틀 B '손절 조건'으로 이어진다(synthesis는 `financial_survival.survival_kill`만 본다).
- 보유 현금을 못 받으면 순현금(> 0일 때만)을 하한 대용값으로 쓴다. 한국은 DART 연간(영업현금·유형자산 취득·단기차입+유동성장기부채).
- v2.1 (같은 날): 점검(kill_census_2026-11-15.md)에서 남은 킬 4개 중 PANW(사이버아크 인수 대가 주식 발행으로 희석 15.2%, FCF 흑자)와 현대차(현대캐피탈·현대카드 연결로 영업현금 −6.0조 원)가 정의 문제로 확인돼 원칙으로 고쳤다. 종목별 예외를 두지 않았고 기준 숫자도 그대로다. SHW(상환 여력 0.91배)는 단기 기업어음 차환·미사용 약정 한도가 데이터에 없는 한계로 기록만 하고 규칙은 바꾸지 않았다. MSTR은 정당한 킬로 남는다.
- v2.2 (2026-11-16): 피어 564종목 첫 전체 갱신에서 하드 킬 35개 중 18개가 상환 여력 < 1.0배 하나로만 걸렸고, 대부분 신용이 좋은 회사의 기업어음·만기 회사채 차환이었다(KDP·CLX·DPZ·FE·IFF·ZBRA·CCI·SBAC·CTRE·LTC, 커버리지 SHW·SRE). 야후 자료에 쓰지 않은 약정 대출 한도가 없어 이 지표로는 '곧 망할 위험'과 '평범한 차환'을 구분할 수 없다 → 역할을 자료가 허락하는 수준으로 낮췄다: 상환 여력은 '차환 의존' 경고만, 하드 킬은 영업현금 적자 런웨이와 현금흐름 적자 기업의 희석만. 기준 숫자는 그대로. SHW 한 종목일 때는 데이터 한계로 기록만 했지만 같은 유형이 18개라 정의 문제로 판단했다. 효과(11-16 재무 기록의 킬 사유로 추정): 하드 킬 35 → 17.
- 실제 종목 점검: `python check_kill.py` (장부 대상 64종목, v1·v2 판정 비교, data/backtest/kill_census_날짜.md 저장). **데이터 오류·정의 실수 확인용이며 결과를 보고 기준 숫자를 바꾸지 않는다.** 테스트: tests/test_survival.py.

### 투자 방향과 딥다이브 개편 기획 (2026-11-15 확정 — 사전 등록, 결과를 보고 바꾸지 않는다)
**목표.** 코어는 지수 ETF 장기 보유(매매 신호 대상 아님, '장기 보유 예외'와 같음). 나머지는 위성 세 전략으로, 차트·실적·멀티플·거시를 판단 보조로 써서 종목마다 "어느 전략으로 들어갈지(또는 들어가지 말지)"를 알려 준다.

| 전략 | 진입 | 매도 | 판정 코드 | 검증 상태 |
|---|---|---|---|---|
| 단기 | (2026-11-17 변경) 진입 신호 없음 — 지지선·손절선과 손절선까지 거리(위험 계산)만 표시. 진입·청산은 본인 판단 | — | engine/strategy.py `support_stop_levels` | 진입 규칙 **미통과**(SHORT_TERM.md) · 선 자체도 **미통과**(LEVELS.md) — 위험 계산용 |
| 추세 | 틀 A v2 진입 의견 | 추세 종료·전환 실패(실패 근거 > 전환 근거) | engine/playbook.py | 미검증(장부로 채점 중) |
| 가치 스윙 | 틀 B v2 진입 가능·조건부 | 고평가 도달(배수 주도) 또는 손절 조건 | engine/playbook.py | 미검증 |

**화면 역할.** 홈 = 시장 상황(지금 그대로). 딥다이브 = "이 종목에 투자할 이유": ① 결론 카드(피어 N개 중 가치 관점 k위·추세 관점 m위, 강점 3·약점 3, 맞는 진입 전략, 위험), ② 피어 비교 표(요인별 값과 피어 내 백분위, 1위 강조), ③ 앞서려면 조건, ④ 전략 카드 3장, ⑤ 기존 4트랙 탭은 근거 상세. 새 페이지 '피어 비교'(산업을 고르면 순위표 → 딥다이브).

**피어와 커버리지.**
- 미국: yfinance 산업(Industry `top_companies`, 지금 V 트랙과 같음) 자동 + `data/peers/peer_map.csv` 수동 덮어쓰기. 한국: 수동 피어 목록 + DART 재무.
- **묶음 원칙(2026-11-15 확정): 회사의 주 수익원 — 영업이익의 대부분이 나오는 사업 — 이 속한 산업.** 야후 산업 분류(대체로 매출 기준)와 다르면 peer_map.csv로 옮긴다. 첫 적용: AMZN(영업이익 대부분이 AWS)을 Internet Retail에서 '대형 플랫폼(클라우드·광고·소프트웨어)' 묶음(AMZN·MSFT·GOOGL·META·ORCL)으로. AAPL은 제품 매출총이익이 서비스보다 커 이 묶음에 넣지 않는다 — 첫 전체 갱신 점검(peer_census_2026-11-16.md) 뒤 '개인용 기기' 묶음(AAPL·HPQ·LOGI·GRMN·SONO)으로. 그 점검의 제외·옮기기·보태기 목록과, 원칙 확인으로 넣지 않은 회사(혼다: 이익 과반이 이륜차, 소니: 게임·음악·센서, 델: 서버)의 근거는 peer_map.csv에 적었다.
- 커버리지 = 레이더 대상(섹터 ETF 상위 구성 종목) ∪ 장부 대상(대표 종목 + `data/snapshots/watchlist.txt`). watchlist 한 줄 추가로 컨센서스 기록·신호 장부·피어 비교에 함께 들어간다.
- 매일 GitHub Actions가 피어 표를 `data/peers/`에 쌓고 화면은 읽기만 한다(쌓인 기록 = 앞으로 채점 자료). 커버리지 밖 종목은 그 자리에서 계산하거나 '관심 종목 추가' 안내.
- 자료가 있는 피어가 5개 미만이면 '피어 부족' — 순위를 매기지 않는다.
- 구현 세부(1단계, 결과 보기 전): 자동 피어는 야후 산업 상위 기업 8개(AUTO_PEERS_CAP)까지 + 같은 산업의 커버리지 종목 전부, 커버리지의 ETF 상위 구성 종목은 10개. 재무·추정치·자기 밴드·생존성은 마지막 '전체' 갱신 7일 뒤 다시 받고(종목당 야후 호출이 많아서, 갱신 기록 `data/peers/peer_refresh_log.csv`. 종목을 지정한 빠른 점검은 부분 갱신이라 세지 않음), 가격 흐름(섹터 ETF 대비 63일 상대강도·틀 A 추세 상태)은 매일. 섹터 ETF = 야후 섹터 → SPDR(XLK·XLC·XLF·XLV·XLY·XLP·XLI·XLE·XLU·XLB·XLRE), 한국은 KOSPI. 자기 밴드 z와 생존성 걸러내기는 미국만(한국은 DART가 필요해 딥다이브 V 트랙 값). 적자 EPS의 PER, 선행 성장 ≤ 0인 PEG는 정의하지 않음(빈칸). 백분위는 하드 킬 종목을 뺀 묶음 안에서, 같은 값은 평균 순위, 값이 2개 미만인 지표는 빈칸.
- v1.3(2026-11-16, 계산 규칙은 그대로): peer_map에서 group을 `yf:<야후 산업 키>`로 쓰면 그 야후 자동 묶음에 합류한다(다른 산업으로 옮기기·같은 산업 회사 보태기). 재무를 다시 받는 조건에 '설정 서명(피어 비교 버전·생존성 버전·peer_map의 티커·묶음·묶음 이름)이 마지막 전체 갱신 때와 다름'을 더했다 — 같은 날이면 다음 날(같은 날 재무 기록은 종목마다 한 번만 저장). 제외(-)한 종목의 딥다이브는 옛 순위 대신 제외 이유를 보여 준다.

**비교 요인.** 지표마다 피어 안 백분위(0~100, 좋은 쪽이 높게)를 매기고 요인 점수 = 지표 백분위 평균.

| 요인 | 지표 |
|---|---|
| 성장 | 매출 성장률(최근 분기, 전년 동기 대비 — v1.1), 선행 EPS 성장(FY1→FY2) |
| 수익성 | 영업이익률(최근 4분기 합, 분기 자료가 비면 야후 operatingMargins), ROE |
| 실적 모멘텀 | 30일 추정치 순상향 비율 = (상향 − 하향) ÷ FY1 추정 애널리스트 수(종목마다 애널리스트 수가 달라 건수 차이 대신 비율 — 1단계 구현 때 결과 보기 전에 정함), FY1 추정치 30일 변화 |
| 밸류에이션 | 선행 PER(= 오늘 가격 ÷ FY1 컨센서스 EPS, 낮을수록 좋음 — v1.2: 야후 forwardPE는 EPS 기준 연도가 회사·시기마다 달라 FY1 EPS가 없을 때만, FY1 적자면 빈칸), PEG(낮을수록 좋음), 자기 밴드 z(낮을수록 좋음) |
| 재무 건전성 | 생존성 v2.2 (점수 아님 — 걸러내기) |
| 가격 흐름 | 섹터 ETF 대비 63일 상대강도, 틀 A 추세 상태(상승 > 횡보 > 하락) |

**전략별 순위 2개와 가중치.** 핵심 요인 50 · 보조 요인 둘은 25씩(보조끼리 균등 — 최적화하지 않음. 근거: 동일 가중의 견고함 Dawes 1979, 1/N DeMiguel 외 2009). 핵심 요인이 피어 최하위면 종합 점수가 50(중간)을 넘지 못한다 — 50은 이 성질이 성립하는 가장 작은 핵심 비중이다.

| 순위 | 걸러내기 | 핵심 50 | 보조 25 | 보조 25 |
|---|---|---|---|---|
| 가치 관점 | 생존성 하드 킬 → 제외, 경고 → 표시 | 밸류에이션 | 수익성('혜자') | 실적 모멘텀(가치 함정 점검) |
| 추세 관점 | 생존성 하드 킬 → 제외 | 가격 흐름 | 실적 모멘텀 | 성장 |

- 결측: 보조 요인 하나가 없으면 있는 요인끼리 비율 유지 재정규화(50:25 → 67:33, '자료 부족' 표시). 핵심 요인이 없거나 요인이 2개 미만이면 순위 없음. 요인 안 지표가 하나도 없으면 그 요인은 없음.

**앞서려면 조건.** 뒤처진 요인마다 피어 상위 25% 기준까지의 차이를 산수로 보여 준다: 가격 경로(예: 선행 PER 상위 25% 기준 × FY1 EPS = 그 주가 이하), 실적 경로(주가 그대로일 때 필요한 EPS·마진·성장), 실적 모멘텀(상향 > 하향, FY1 추정치 상승 전환), 가격 흐름(섹터 ETF 대비 필요 초과 상승률). '가장 가까운 경로' = 그 요인 하나만 상위 25% 기준에 닿았을 때 전략 점수가 가장 많이 오르는 요인. 다음 실적 발표일과 확인할 항목을 함께 표시.

**전략 선택 우선순위.**
1. 공통 차단: 생존성 하드 킬 → 신규 진입 없음. 국면 게이트 크기 0(G3 등) → 의견만.
2. 가치: 틀 B '진입 가능'·'조건부' 그리고 가치 관점 순위 피어 상위 1/3.
3. 추세: 틀 A 진입 의견 그리고 추세 관점 순위 피어 상위 1/3.
4. ~~단기: 가치·추세가 아니고 단기 규칙 진입 조건 충족.~~ (2026-11-17 삭제 — 단기 규칙이 사전 등록 시험 미통과라 사용자 결정으로 진입 후보에서 뺌. 결과를 보고 규칙 숫자를 고친 것이 아니라 미통과한 규칙을 신호로 쓰지 않기로 한 것)
5. 여럿 충족: 보유 기간이 긴 쪽(가치 > 추세)을 진입 전략으로, 나머지는 보조 근거로 표시.

**단기 규칙.** 진입: 틀 A 상태가 상승 또는 횡보 + RSI(2) < 10 눌림(T-B와 같은 눌림 정의) + 다음 날 시가에서 손절선까지 거리 ≤ 3%(손익비 ≥ 1). 지지선 = 진입 신호일 기준 최근 확정 스윙 저점(틀 A ZigZag 3×ATR), 손절선 = 지지선 − 0.5×ATR(20). 청산(먼저 오는 것): ① 장중 +3% 도달 → +3%(시가가 이미 위면 시가) ② 종가 < 손절선 → 다음 날 시가 ③ 틀 A 상태가 '하락' → 다음 날 시가 ④ 10거래일 안에 +3% 미도달 → 다음 날 시가. 왕복 비용 0.30%. 숫자는 관행값으로 고정, 사전 등록한 새 시험으로 확인(시도 횟수 +1, 홀드아웃 판정).
- 구현 세부(2026-11-16, 결과 보기 전 고정): RSI(2)는 와일더 식(T-B·틀 A 검증과 같음). 지지선 = 신호일까지 확정된 가장 최근 스윙 저점(추세 상태와 무관하게 마지막 L). 진입일 장중에도 +3% 익절을 본다. 10거래일은 진입일 포함, 10번째 날 종가 뒤 다음 날 시가. 같은 날 종가 조건은 손절선 → '하락' → 시간 순으로 이름만 붙인다(모두 다음 날 시가). 종목당 1포지션(청산 실행일 종가부터 새 신호).
- **시험 설계(2026-11-16 결과 보기 전 고정, engine/short_backtest.py):** 개발 묶음 = 대표 종목 64(틀 A와 같음), 판정 묶음(HOLDOUT) = 지금까지 어떤 백테스트에도 쓰지 않은 84종목(미국 11개 섹터 59 · 한국 25, 목록은 SHORT_BT_CONFIG), 2001년~. 판정 기준: HOLDOUT 전체 기간 거래 ≥ 30, 거래당 순수익률(비용 뒤) > 0, (실제 − 보유 기간 맞춘 무작위) ≥ 월 묶음 표준오차 × **3**, 그리고 개발 묶음 앞·뒤 기간(2014 분할) 모두 > 0. 가격 기반 타이밍의 일곱 번째 시험이라 z 기준을 2에서 3으로 높였다(여러 번 시험하면 우연히 z ≥ 2가 나올 확률이 커짐 — Harvey·Liu·Zhu 2016, 새 요인에 t > 3 권고). 무작위 비교는 거래마다 같은 종목 무작위 날짜 20곳, 같은 보유일(실제가 장중 익절이면 그날 종가, 시가 청산이면 시가에 팖). 참고: '눌림 없이'(추세 상승·횡보 + 지지선, RSI 조건만 뺌) 대비, 비용 0·0.30·0.60%, z ≥ 2. 통과·미통과 모두 결과를 보고 숫자를 바꿔 다시 시험하지 않는다 — 미통과면 단기 카드는 '검증 미통과 — 참고만'.
- **지지·손절선 표시와 검증(2026-11-17 사전 등록, 결과 보기 전 고정 — engine/strategy.py LEVEL_CONFIG, engine/level_backtest.py):** 단기는 진입 여부 대신 지지선(확정된 최근 스윙 저점)·손절선(지지선 − 0.5×ATR(20))·손절선까지 거리·매수 비중 한도(= 계좌 위험 1% ÷ 거리, 최대 100% — 산수라 검증 대상 아님)를 보여 준다. 상태: 종가 < 손절선 '손절선 이탈' / 저가 ≤ 지지선 + 0.5×ATR '지지선 접근' / 그 외 '지지선 위'. 선의 정의는 단기 규칙 그대로이며 시험 결과를 보고 바꾸지 않는다. 가격 기반 시험의 여덟·아홉 번째 질문이라 z 기준 3. 개발 묶음 = 대표 종목 64, HOLDOUT = 어떤 백테스트에도 쓰지 않은 110종목(미국 11개 섹터 85 · 한국 25), 2001년~, 사건 뒤 10거래일은 같은 종목의 새 사건을 세지 않음, 표준오차는 월 묶음.
  - 질문 1(지지선): 사건 = 저가 ≤ 지지선 + 0.5×ATR 이고 종가 ≥ 손절선인 날. 결과 = 이후 10거래일 종가가 그날 손절선 아래로 내려간 비율. 판정 비교 = 같은 종목에서 최근 5거래일 변화(ATR 단위)가 사건일과 ±0.5 이내인 다른 날 20곳의 종가에서 같은 ATR 거리만큼 아래 그은 가짜 선. **통과 = HOLDOUT 사건 ≥ 100, (실제 − 가짜) < 0, z ≤ −3, 개발 묶음 앞·뒤 기간 모두 차이 < 0.** 참고: 하락 폭을 맞추지 않은 무작위 날 가짜 선, 10거래일 수익률.
  - 질문 2(손절선): 사건 = 종가가 손절선 아래로 처음 내려온 날. 결과 = 다음 날 시가 → 10거래일 뒤 종가 수익률. 판정 비교 = 같은 종목 무작위 날 20곳. **통과 = HOLDOUT 사건 ≥ 100, (이탈 뒤 − 무작위) < 0, z ≤ −3, 개발 묶음 앞·뒤 모두 차이 < 0.** 참고: 하락 폭 맞춘 날, 20거래일 수익률·−10% 이하 비율.
  - 결과별 화면 문구는 LEVEL_VERDICT_TEXT에 미리 적었다(통과 '검증됨 — …', 미통과 '검증 안 됨 — 위험 계산용/손실 한도로만'). 결과가 나오면 LEVEL_VERIFICATION에 옮겨 적기만 한다.
- **지지선 이탈 뒤 회복(2026-11-17 사전 등록, 결과 보기 전 고정 — engine/reclaim_backtest.py):** 사용자 가설 '지지선을 한 번 깼다가 올라오는 패턴이 많다'. LEVELS.md 결과를 본 뒤 나온 질문이라 새 질문(가격 기반 열 번째, z 기준 3)으로 등록하고 새 HOLDOUT 148종목(미국 123·한국 25, 이전 모든 묶음과 겹치지 않음)을 쓴다. 지지선 정의는 그대로. 이탈 = 종가가 지지선 아래로 처음 내려옴(전날 종가는 지지선 이상), 회복 = 이후 10거래일 안에 종가가 이탈일의 지지선 위로, 이탈 뒤 10거래일은 새 이탈을 세지 않음. A(집계, 판정 없음): 접근 뒤 버팀·깼다가 회복·못 돌아옴 비율, 이탈 뒤 회복 비율·걸린 일수·깊이. B(판정): 회복일 다음 날 시가 → 10거래일 뒤 종가 수익률이 같은 종목 무작위 날 20곳보다 좋은가 — **통과 = HOLDOUT 사건 ≥ 100, 차이 > 0, z ≥ 3, 개발 묶음 앞·뒤 기간 모두 차이 > 0.** 참고: 하락 폭 맞춘 날, 20거래일, 회복 뒤 다시 이탈 비율. 결과별 화면 문구는 RECLAIM_VERDICT_TEXT. 주의: 효과가 전혀 없는 무작위 걸음에서도 '깼다가 회복'은 접근 사건의 약 43%, 이탈 뒤 회복은 약 72%로 흔하다(tests/test_reclaim.py 3) — 흔하다는 것만으로는 신호가 아니며 B가 통과해야 한다.
- **탐색과 확인(2026-11-17 사용자와 합의 — engine/explore.py):** 새 종목 묶음이 줄어 가므로 두 단계로 나눈다. ① 탐색: 지금까지 판정·개발에 쓴 개별 종목 전부(EXPLORE_CONFIG['UNIVERSE'], 약 550개)로 가설을 자유롭게 본다(`run_backtest.py --explore 이름`). 결과는 **'탐색 — 증거 아님'**이며 화면·문구에 검증된 것처럼 쓰지 않는다. 볼 때마다 data/backtest/EXPLORE_LOG.md에 한 줄이 쌓인다(본 횟수를 남기기 위해 — 지우지 않는다). '후보' 기준(미리 고정): 사건 ≥ 300, 무작위 날 대비 |z| ≥ 2, 앞·뒤 기간 같은 방향, 기대한 방향. ② 확인: 후보가 된 가설만, 정의를 한 글자도 바꾸지 않고, 아직 쓰지 않은 종목(또는 앞으로 쌓이는 기록)에서 z 3 기준으로 한 번. 후보가 아니면 거기서 끝낸다 — 같은 가설을 숫자만 바꿔 후보가 될 때까지 다시 보지 않는다. 새 가설 = EXPLORE_HYPOTHESES에 사건 함수 하나(그날 종가까지의 자료만 사용)와 기대 방향을 등록. 같은 시기의 주식은 함께 움직이므로 '새 종목·같은 기간' 확인도 절반쯤만 독립이고, 완전한 확인은 앞으로 쌓이는 기록뿐이다.
- 탐색 가설 `touch4`(2026-11-17 사용자 가설 '지지선·저항선을 4번째 터치하는 순간은 뚫린다', 정의는 결과 보기 전 고정 — engine/touch_explore.py TOUCH_CONFIG): 수평선 = 확정된 스윙 저점(지지)·고점(저항), 새 스윙 점이 기존 선에서 1×ATR 안이면 새 선을 만들지 않음, 수명 250거래일. 터치 = 선에서 0.5×ATR 이내(선을 만든 점이 1번째, 다음 터치는 종가가 2×ATR 넘게 멀어졌다 돌아와야 셈). 뚫림 = 터치일부터 10거래일 안에 종가가 선을 0.5×ATR 넘게 넘어감. 후보 = 4번째 터치 ≥ 300건, 4번째 터치 뚫림 비율 − 2·3번째 터치 뚫림 비율 > 0, z ≥ 2, 앞·뒤 기간 모두 > 0. 효과 없는 무작위 걸음에서는 터치 횟수와 무관하게 약 59%로 평평하다(tests/test_touch.py 4).
- 탐색 가설 `touch4b`(2026-11-17, touch4 결과를 본 뒤 사용자가 '뚫림'의 뜻을 정정 — 같은 가설의 두 번째 탐색으로 기록): touch4 결과(터치 횟수와 무관하게 약 54%, 후보 아님)의 '뚫림'은 잠깐 이탈도 셌다. 사용자 뜻 = 의미 있게 넘어가 일정 기간 돌아오지 않음. 정의(결과 보기 전 고정, 기간은 사용자가 20거래일 선택): 터치마다 버팀(터치 뒤 10거래일 동안 종가가 선을 0.5×ATR 넘게 넘어가지 않음) / 일시 이탈(넘어간 날부터 20거래일 안에 종가가 선 안쪽으로 돌아옴) / 확정 뚫림(20거래일 동안 한 번도 안 돌아옴). 선은 확정 뚫림일 때만 없어지고 일시 이탈한 선은 살아 있다. 판정은 4번째 터치의 확정 뚫림 비율 − 2·3번째, 후보 기준은 touch4와 같다. 이번 결과도 보고 기간·폭을 바꿔 다시 보지 않는다.
- 탐색 가설 `mom12`·`ret63`(2026-11-17, 사용자와 합의 — 차트 패턴 대신 연구로 증거가 있는 모멘텀을 우리 데이터로 확인, 정의는 결과 보기 전 고정 — engine/momentum_explore.py): 2001년부터 매달 마지막 거래일에 시장별(미국·한국 따로) 줄 세우기. mom12 = 252거래일 전 → 21거래일 전 수익률(최근 1개월 제외), ret63 = 63거래일 수익률(앱 rs63과 같은 기간 — 같은 시장 안 순위라 기준 지수를 빼도 순서가 같다). 결과 = 다음 날 시가 → 21거래일(판정)·63거래일(참고) 뒤 종가, 같은 비중, 비용 전. 그달·그 시장 종목 20개 이상일 때 상위 20% − 하위 20%, 전체 = 그달 미국·한국 평균. 후보 = 1개월 상위 − 하위 > 0, t ≥ 2, 앞·뒤 기간 모두 > 0, 달 수 ≥ 100. 3개월은 Newey-West(시차 2). 지금 살아 있는 회사만 있어 하위 20%가 좋게 보이는 쪽으로 치우칠 수 있다(차이가 작게 나오는 방향). 참고: 63일 상대강도는 RSI가 아니다 — 상대강도는 기간 수익률 차이, RSI는 오른 날·내린 날 힘의 비율(0~100).
- 전략 선택 구현 세부(2026-11-16, 결과 보기 전): 조건 자료가 없으면(피어 순위 없음·판정 불가) '확인 불가' = 충족 아님. 국면은 홈과 같은 계산(지수 레짐·VIX·거시 문구 + 최근 거래일 완화 확인, engine/gate.py `market_gate_from_prices`), 기준 지수를 못 받으면 방어 폴백 G2. 생존성 하드 킬이면 조건 충족 여부는 보여 주되 '신규 진입 없음', 국면 크기 0이면 '의견만'.

**검증.** 화면 문구: 피어 비교는 사실 비교이며 순위가 수익을 예측한다는 검증은 없음. 앞으로 채점(미리 고정): 매월 피어 묶음별 전략 순위 상위 1/3 vs 하위 1/3의 이후 6·12개월 상대 수익. **구체적 방법은 2026-11-17 결과 보기 전에 engine/peer_eval.py `PEER_EVAL`로 고정**(run_backtest.py --peer-eval → PEER_EVAL.md): 시작일 2026-11-17(묶음표 정정·생존성 v2.2가 반영된 첫 기록, 그 전 기록은 채점 안 함), 매달 첫 순위 기록일, 관점마다 순위가 나온 묶음(점수 있는 종목 ≥ 5)의 상위 1/3(ceil(n/3)) vs 하위 1/3, 가격 기준일 다음 거래일 시가 → 126거래일(6개월, 판정)·252거래일(12개월, 참고) 뒤 종가, 1·3개월은 중간 참고, 같은 비중·비용 전, 한 달 값 = 묶음 (상위 − 하위) 평균, Newey-West(6개월 시차 5). **판정 = 관점마다 6개월 결과가 다 나온 달 ≥ 12, (상위 − 하위) > 0, t ≥ 2. 그 전에는 '판정 보류'.** 처음 적은 '기록 12개월 뒤 판정'은 6개월 보유 결과가 12달 쌓이려면 기록 약 18개월이 필요해 이렇게 정했다(가장 이른 판정 2028-05 무렵). 참고: 하드 킬 종목 수익률 − 같은 묶음 평균(생존성 걸러내기 점검). 보류 중 결과를 보고 순위 규칙·가중치를 바꾸면 채점을 처음부터 다시 한다. 과거 검증은 미국만 부분 가능(SEC 공시일 기준 매출·EPS·마진 + 가격 흐름, 추정치 변화 제외) — 새 질문이므로 시도 횟수 +1, 홀드아웃 판정.

### 강의 조건 시험 (2026-11-18 사전 등록 — 결과 보기 전 고정)
사용자가 듣는 기술 분석 강의(시장흐름 1~4주차·마스터클래스·BNF) 정리본 1~9번의 조건을 따로, 그리고 조합해서 시험한다. **전체 정의는 `data/backtest/LECTURE_PREREG.md`(질문 Q1~Q69 부록 포함), 숫자는 `engine/lecture_signals.py` `LECTURE_CONFIG` 한 곳.** 강의 정리본 원문은 유료 강의라 저장소에 넣지 않는다(정의 요약만).
- 자료는 강의 **정리본**이다. 정리본의 '가정값'(좌우 5봉, ADX 25/20, 하향 2일·상향 3일, RS 50·20일 등)은 강의에서 나온 값이 아니므로 관행값으로 고정하고 결과를 보고 맞추지 않는다. 기존 T 트랙 코드도 같은 정리본으로 만든 것으로 보이며, 어긋난 곳은 LECTURE_PREREG.md 2장 '코드 현재'에 적었다.
- 시험용 정의는 앱 함수와 **따로** 둔다(lecture_signals.py). 앱 T 트랙·국면 게이트는 바꾸지 않는다 — 게이트는 이미 시험해 고정했고, 앱 쪽 수정은 시험과 별개로 사용자가 결정한다.
- 다섯 가족(한 번에 등록, 순위는 계산 순서일 뿐):

| 가족 | 내용 | 비교 기준 |
|---|---|---|
| L1 BNF 전체 규칙(9번) | 5조건 모두 + 실적일 제외 + 손절·시간 손절·25일선 절반 익절. **질문 1** 5거래일 안 반등(종가 ≥ 진입가 + 0.5R) 비율, **질문 2** 거래당 순수익. 앱 '매매 규칙 통과'는 둘 다 통과일 때만 | 질문 1: 최근 5거래일 변화(ATR 단위) ±0.5인 같은 종목 다른 날의 같은 % 거리 / 질문 2: 보유 기간 맞춘 같은 종목 무작위 진입, 비용 0.30% |
| L2 매수 사건 | 사건 1개 + 환경 0~1개 + 확인 0~1개. 사건 유형(돌파·바닥 전환·눌림)마다 정리본이 짝지은 환경·확인만(약 254조합), 3순위·맨 뒤 사건은 단독만 | 같은 날 같은 시장 종목 평균을 뺀 10거래일 초과수익, 비용 0.30% |
| L3 상태별 RSI(3번) | L3-a RSI ≤ 30 사건에서 상승 상태 > 하락 상태, L3-b RSI ≥ 70 사건에서 상승 > 횡보. 두 집단 각각 사건 ≥ 300 | 상태끼리의 차이(10거래일 초과수익) |
| L4 매도·약화 | 약화 5개, 쌍고점, 이평 쌍봉, 붕괴 3단계, 추세선 장대음봉 이탈, 가짜 돌파, 윗꼬리 대량, 음봉 대량 | 같은 종목·같은 상태 무작위 날 대비 20거래일 최대 낙폭 |
| L5 레벨 | 겹침 수평 지지, 채널 하단·중단, 박스 0.5·0.75, 크로스 기준선, 빈 공간, 1대1 목표 | LEVELS.md 방식(같은 거리 가짜 선) |

- 세 겹 분리: **탐색** = EXPLORE_CONFIG 종목 전부, 신호일 2001~2013 → **검증** = 같은 종목 2014~(다른 질문에서 이미 본 기간이라 절반쯤만 독립), 같은 방향 + z ≥ 2 → **최종** = 새 종목(규칙으로 고정한 목록, 약 150개), z ≥ 3, 후보 전체 최대 5개. 검증 결과는 탐색 후보인 질문만 계산해 보여 준다.
- 탐색 후보 조건: 기대한 방향, 큰 가족(L2·L4·L5)은 같은 종목 안에서 사건 날짜를 섞은 200회의 '가족 안 최고 t' 95% 분위 이상(PBO 참고), 질문 1~2개 가족(L1·L3)은 z ≥ 2. 그리고 사건 ≥ 300, 종목 ≥ 50, 미국·한국 같은 방향(둘 다 30건 이상일 때), 기여가 가장 큰 달 하나를 빼도 같은 방향·z ≥ 2. 표준오차는 월 묶음, 같은 종목·같은 사건은 10거래일 안의 새 사건을 세지 않는다.
- 자료: 섹터 ETF(미국 SPDR, 한국 코스피200 섹터 ETF — 받기 단계에서 이름을 출력해 사람이 확인한 뒤에만 L1을 돌림, 짝 없는 섹터·출시 전 기간은 '데이터 없음'), 지수(S&P 500·나스닥·KOSPI·KOSDAQ), 실적 발표일(SEC 8-K Item 2.02 2004-08~, DART 잠정실적 2002-11~, 그 전은 '데이터 없음'). 캐시 `data/backtest/lecture_cache/`(.gitignore).
- 구현 단계: 1단계 = 기초 정의 + L1·L3(`run_backtest.py --lecture data` → `--lecture L1` / `--lecture L3`, 보고서 LECTURE_L1.md·LECTURE_L3.md, 볼 때마다 EXPLORE_LOG.md에 한 줄). 2단계 = L2·L4·L5 — L2는 사건이 모두 만들어진 뒤 가족 전체를 한 번에 계산한다(일부만 먼저 보면 통과선을 정할 수 없음).
- 결과를 보고 정의·숫자·짝짓기를 바꿔 다시 보지 않는다. 통과하지 못한 조건과 원칙 항목은 앱 '기술적 판독' 체크리스트에 표시만 한다(✅ 최종 통과 / ⚠️ 미통과 / ⏳ 검증 전 / 📋 원칙).
- 테스트: tests/test_lecture.py(1단계), tests/test_lecture_l2.py(2단계 L2), tests/test_lecture_l45.py(2단계 L4·L5)
- **L4·L5 결과(2026-11-19, LECTURE_L4.md·LECTURE_L5.md): 둘 다 탐색 후보 없음 → 강의 다섯 가족 모두 미통과.** L4는 지지 이탈(W11, 추가 낙폭 +1.12%p t 2.2)·추세선 장대음봉 이탈(W12 t 2.1)이 2008-09를 빼면 t 1.94·1.80으로 달 쏠림 점검에서 떨어짐, 약화 W01~W06은 오히려 낙폭이 작음. L5는 11개 모두 가짜 선과 ±2%p 안(모든 스윙 지지 깨짐 44.5% vs 가짜 44.5%). L4 보고서의 '변동성 단위' 참고 열은 ATR%가 거의 0인 날(거래 정지 등) 때문에 값이 깨져 있었음 — 판정과 무관한 참고 열이라 코드만 고치고(ATR% < 0.1%는 나누지 않음) L4는 다시 돌리지 않는다. — 손 계산·자른 자료로 계산해도 같은 값(미래 자료 미사용)·효과 없는 가격에서 후보가 안 되고 심은 효과는 잡는지.
- **1단계 결과(2026-11-18, LECTURE_L1.md·LECTURE_L3.md): 둘 다 미통과.** L1 질문 1(반등)은 탐색 후보(z +2.7)였으나 검증 2014~ z +1.0으로 떨어졌고, 질문 2(수익)는 후보 아님(z −0.2). L3-a(z −1.3)·L3-b(z −2.8, 정리본 주장과 반대 방향) 모두 후보 아님. 반대 방향을 새 발견으로 쓰지 않는다(쓰려면 새 가설로 따로 사전 등록).
- **2단계 L2 구현에서 정한 것(2026-11-18, L2 결과 보기 전).** 사전등록안·Q 결정이 정하지 않은 세부다. 전체는 LECTURE_PREREG.md 부록 B.
  - 판정 t = 10거래일 초과수익 평균 ÷ 월 묶음 표준오차(비용 없이). 비용 0.30%는 '방향'에 넣는다: 후보는 평균 − 비용 > 0, 미국·한국도 비용 뒤 평균의 부호로 비교, 달 쏠림은 그 달을 빼고 비용 뒤 > 0 이고 t ≥ 2. 무작위 날과 실제 사건을 같은 식으로 재야 섞기 통과선과 비교되기 때문.
  - 섞기: 한 번마다 사건(종목·날짜)을 같은 종목의 탐색 기간 무작위 날로 바꾸되, 같은 사건은 모든 조합에서 같은 날로 바꾼다(조합끼리 겹치는 정도 유지). 최고 t 경쟁에는 사건 ≥ 300·종목 ≥ 50인 조합만 넣는다(사건 수는 결과와 무관). 조합은 정확히 256개.
  - 돌파 = 종가가 그날까지의 최근 확정 스윙 고점 위로 처음 마감한 날. 돌파 확정 ①은 돌파봉에서 본다. ④ = 돌파 뒤 생긴 확정 저점이 돌파일의 최근 확정 저점보다 높음. 돌파선 구간의 ATR은 돌파일 값. SR Flip·채널 복사 리테스트 창은 20거래일(Q53 = Q23).
  - 쌍바닥: L2 = L1 바로 앞 확정 저점, 넥 = 그 사이 가장 높은 확정 고점, '하락 구조' = L2 날 상태가 하락, 최소 폭 5%(Q59) 적용. E07(공격) = 쌍바닥 확정일부터 5일선 기울기 > 0인 첫날 — 넥을 넘는 날 5일선은 대개 이미 올라 있어 E05와 거의 같게 나온다(정의가 그렇게 나오는 것, 결과 전 확인). E08 = 5일선 쌍바닥 완성 뒤 20일선 기울기 > 0인 첫날.
  - 전환 증거(E09·C06): ①·②는 그날까지의 구조, ③ MACD·④ 거래량 반등은 그날 하루(앱 S-FALL과 같음). E09는 전날 상태가 하락일 때, 하락 상태 한 번에 한 번.
  - 하락 추세선: 가장 최근 확정 고점 두 개가 낮아지면 긋고, 그 위에 새 확정 고점이 생기거나 0.25 진짜 판정이 나면 지운다. 쐐기·삼각은 가장 최근 고점·저점 두 개씩(250거래일 안), 위 선 구간 상단을 처음 넘은 날.
  - 겹침 지지(C03): 종류 = 확정 스윙 · RVOL ≥ 2 캔들 고가 · 긴 꼬리 대량 캔들 종가 · 250일 체류 상위 20% 칸. 사건이 기댄 자리 = SR Flip은 그날 저가, 쌍바닥·공격은 L1, 이평 쌍바닥은 5일선 두 번째 저점 날 저가, E09는 최근 확정 저점(그 점 자체는 세지 않음).
  - C07(거래량 이유 = 실적)은 실적일이 사건 전날~당일일 때. 사건 다음 날의 발표는 그날 거래량의 이유가 될 수 없고 미래 자료다(Q42 '±1일'을 사건 뒤쪽만 좁힘).
  - 같은 높이 스윙(이웃 봉의 고가가 같음)은 둘 다 스윙이다(앱과 같음). 그래서 그 두 점으로는 '낮아진 고점'이 되지 않는다.
- **L2 결과(2026-11-19, LECTURE_L2.md): 탐색 후보 없음.** 조합 256개 중 가족 199개, 통과선 t 3.49, 최고 t +2.6(E02 + 횡보 + C02, 비용 뒤 +0.07%, 미국·한국 방향 다름), PBO 60%. 사건 단독은 돌파·쌍바닥 확정·분출이 10거래일 시장 대비 음수(E13 t −4.6, E05 t −3.0, E01 t −2.8) — 기대와 반대 방향이라 판정에 쓰지 않는다.
- **2단계 L4·L5 구현에서 정한 것(2026-11-19, L4·L5 결과 보기 전).** 전체는 LECTURE_PREREG.md 부록 C.
  - L4 결과 = 이후 20거래일 최대 낙폭(t+1~t+20 최저가 ÷ t일 종가 − 1). 비교 = 같은 종목·같은 상태(사건 날 상태)·같은 기간 **모든 날 평균**(사전 등록의 '무작위 날 20곳'과 기대값이 같고 잡음만 적음). 추가 낙폭 = 비교 평균 − 사건 낙폭, 후보는 > 0. 섞기는 같은 종목·같은 상태의 무작위 날. 비용 없음.
  - '상승 상태에서 일어난' = 시작점의 상태: 약화 W01~W06은 앞 고점 H2의 확정일(H1 확정일로 보면 ① 갱신 실패는 Q2 정의상 거의 상승 상태가 될 수 없음), 쌍고점·이평 쌍봉은 첫 고점, 붕괴·지지 이탈은 이탈일, 윗꼬리 대량은 그 캔들 날, 나머지는 사건 날.
  - 약화의 H2 = H1 바로 앞 확정 저점보다 앞선 최근 확정 고점(사이에 저점이 있어야 두 고점). 붕괴: 지지 = 최근 확정 저점, 회복 = 10일 안 2일 연속 구간 하단 이상, 전환 = 이탈 뒤 첫 확정 고점(250일 안)이 이탈일 최근 확정 고점보다 낮음, 사건 날 = 그 고점 확정일과 이탈+10 중 늦은 날. 추세선 일반 이탈(W13) = 장대음봉이 아닌 이탈. W08(공격)은 W07과 거의 같게 나온다(E07과 같은 이유).
  - L5 가짜 선 = 사건 날 종가와 선의 거리(ATR 단위)·기울기(로그)를 그대로 두고, 같은 종목·같은 기간에서 최근 5거래일 변화 ±0.5 ATR·사건과 10거래일 넘게 떨어진 날 20곳. 실제와 가짜를 같은 식(종가 ± 거리×ATR, 기울기)으로 잰다. 접근 = 전날 종가가 구간 밖 → 그날 저가(고가)가 구간에 닿고 종가는 안 깨짐.
  - V01·V02 수평선 = 확정 스윙 전부(Q35 삭제 규칙), 날마다 '전날 종가 아래 가장 가까운 살아 있는 선'. V01 = 그 자리 겹침 종류(선 자신 포함) ≥ 3, V02 = 그날까지 그 선의 반응 ≥ 3. V03·V04 = W12와 같은 상승 추세선에 채널 폭 ≥ 4%일 때 하단·중단선(로그 폭 절반). V05·V06 = 쌍바닥 확정 뒤 넥 + 0.5·0.75×폭의 첫 접근. V07·V08·V09 = 20×60 크로스 날 두 선 평균, 다음 크로스까지.
  - V10 결과 = |10거래일 수익| ÷ 그날 ATR%(변동성 차이를 빼려고 Q43 '움직인 폭'을 ATR 단위로), 섞기는 같은 종목 안 이름표 섞기, 두 묶음 각각 ≥ 300. V11 = 쌍바닥 1대1(넥 + 폭, 고가로) vs 무효(L1 구간 하단, 종가로), 250거래일 안 둘 다 없으면 0, 가짜는 같은 기간 무작위 날의 같은 % 거리(Q61).
  - L5 섞기 = 사건마다 실제 1개 + 가짜 20개 중 하나를 무작위로 '실제'로. PBO는 질문마다 잣대가 달라 내지 않는다.
- **최종 확인용 새 종목(N4) 고정(2026-11-18, L2 결과 전):** `LECTURE_CONFIG["FINAL_UNIVERSE"]` — 지금까지 어떤 시험에도 쓰지 않은 오래 상장된 종목, 미국 11개 섹터 98개 + 한국 코스피 53개. 못 받거나 260거래일 미만이면 빼고 채우지 않는다. 한국 코드는 받을 때 이름을 출력해 사람이 확인한다. 검증을 통과한 질문이 생길 때만 연다.

### H-MR1 기술적 단기 반등 단타 (2026-11-19 사전 등록 — 결과 보기 전 고정)
강의 시험이 모두 미통과로 끝난 뒤 사용자가 제안한 평균 회귀 단타(강의 9번 BNF 중심 + 6번 지지·거래량 + 3번 과매도 + 1번 시장 필터, 2번 상대강도·4번 추세 구조 배제)를 시험 가능한 규칙으로 고정했다. **전체 정의는 `data/backtest/MR1_PREREG.md`, 숫자는 `engine/mr_backtest.py` `MR1_CONFIG`.** 사용자 원안(2020~2026, 2020~2023에서 수치 수립, 이름 지정 12~100종목, 150~200건)은 표본(BNF는 종목당 연 0.1건)·위기 구간 쏠림(단기 반전 수익은 VIX와 함께 치솟음, Nagel 2012)·결과를 알고 고른 종목·결과에 맞추기 위험 때문에 아래로 바꿨고, 사용자가 확정했다.
- 규칙 셋: **A** 사용자 규칙 전체(BNF 조건 2~5 + [t−5, t−1] 볼린저(20, 2) 하단 아래 종가 + 바닥 m일 저가가 살아 있는 확정 스윙 수평선 구간 안 + 지수 ADX14 ≥ 25·−DI > +DI인 날 제외), **B** BNF 핵심(조건 2~5, L1 참고 변형 '조건 1 뺌'과 같음 — 2001~2013은 이미 봄), **C** 단순 반전 대조군(최근 5거래일 변화 ÷ ATR14 ≤ −3.0이 처음 된 날).
- 공통: 다음 날 시가 진입, 손절 = 급락 시작일(40일 종가 최고일)~신호일 최저가(손절폭 ≤ 12%), 실적일 제외(Q68), 청산 = L1과 같은 코드(25일선 절반 익절, 5일 +0.5R 시간 손절), 비용 0.30%, 종목당 1포지션.
- **대상(개정 mr1-2, 2026-11-19 실행 전·결과 전, 사용자 결정)**: 신호일마다 같은 시장 시험 종목 중 지난 60거래일 평균 거래대금(그날 제외) 상위 절반 → 그 안에서 지난 250거래일 변동성 상위 절반에 든 종목의 신호만 판정(종목·날의 약 4분의 1). 무작위 비교 날도 같은 종목이 대상이었던 날에서만. 과거로만 재고(미래에 출렁일 종목을 미리 고르지 않게) 날마다 순위로 정한다(고정 숫자면 2008·2020에 몰림). 전체 종목 결과는 참고. A는 거래 300건을 못 채울 수 있다 — 그러면 '거래 부족'으로 끝내고 넓혀 다시 보지 않는다.
- 판정: 거래당 순수익 − 보유 기간 맞춘 같은 종목 무작위 진입 > 0. 탐색 = 기존 545종목 신호일 2001~2019, z ≥ 2 + 거래 ≥ 300 + 종목 ≥ 50 + 미국·한국 같은 방향 + 달 쏠림 점검. 확인 = 후보만 새 151종목(FINAL_UNIVERSE) 2001~2026에서 한 번, z ≥ 3(2020~2023·2024~2026은 참고로 따로). 기존 545종목의 2020년 이후는 열지 않는다. A가 통과해도 C보다 낫지 않으면 '강의 부품의 추가 효과 없음'.
- 참고: A − C·B − C, 5일 +0.5R 반등(하락 폭 맞춘 날 대비), 기간별·시장 등급별, 청산 종류, 필터마다 빠진 신호 수.
- 실행: `run_backtest.py --mr1`(강의 시험 자료 `--lecture data`를 그대로 씀) → MR1.md·mr1.json·EXPLORE_LOG 줄. 테스트 tests/test_mr1.py.
- 레버리지 ETF(SOXL 등)는 범위 밖: 주식 결과를 그대로 옮길 수 없다(업종 반전은 성격이 다르고, 하루마다 배율을 맞추는 L배 ETF는 L × 기초 수익 − (L² − L)/2 × σ² × 일수만큼 깎이며, 레버리지는 t를 바꾸지 않음). 옮기려면 기초 ETF와 가상 3배 시리즈로 따로 사전 등록(MR1_PREREG.md 4장).
- **결과(2026-11-19, MR1.md): 탐색 후보 없음 — 확인 단계(새 151종목)는 열지 않았다.** 대상(종목·날의 25%) 안에서 A −0.24%(146건) vs 무작위 −0.17%(z −0.1, 거래 300 미만), B −0.13%(404건) vs +0.05%(z −0.3), C +0.12%(3,120건) vs +0.03%(z +0.3, 미국 −·한국 +). A − C −0.36%p, B − C −0.25%p — 강의 부품이 단순 반전보다 나쁘거나 같았다. 표본이 작아 C는 거래당 약 0.5%, B는 약 1.2% 크기의 우위만 잡아낼 수 있었다. 규칙·숫자를 바꿔 다시 보지 않는다.
- 함께 고친 것(결과 무관): `lec_swing_lines`의 반응일 fr이 '첫 닿음의 반응'이었는데 '가장 먼저 끝난 반응'으로 바로잡음(뒤 닿음의 반응이 먼저 끝나면 선이 며칠 일찍 삭제된 것처럼 보였음). L5(V01·V02·참고 전체 지지)에 쓰인 계산이며, 영향은 생긴 지 250일이 지난 선이 길어야 10일 남짓 '삭제'로 보인 경우뿐이라 L5 결론을 바꾸지 않는 크기로 본다(L5는 다시 돌리지 않는다).

### H-MR2 단기 신호 찾기 — 신호 즉시 진입 · +5% 익절 (2026-11-19 사전 등록 — 결과 보기 전 고정)
MR1 뒤 사용자 요청: 다음 날 시가가 아니라 신호가 나오면 바로 사고, +5%면 익절하는 규칙을 찾는다. **전체 정의는 `data/backtest/MR2_PREREG.md`, 숫자는 `engine/mr2_backtest.py` `MR2_CONFIG`.** 사용자 결정: 종가·지정가 진입 둘 다 비교, 손절 −5%, 최대 3거래일, 같은 날 둘 다 닿으면 손절, 신호 15개(규칙 26개 = 종가 18 + 지정가 8)는 Claude 초안 그대로.
- 종가 진입 = 신호일 종가. 지정가 = 전날까지 자료로 정한 가격 P, 저가 ≤ P면 min(시가, P)에 체결, 체결일 저가 ≤ 체결가 × 0.95면 그날 손절(가격이 P 아래로 간 것은 체결 뒤가 확실), 체결일 +5%는 안 셈(고가가 체결 전일 수 있음). 그 뒤 t+1~t+3일: 시가 갭은 시가, 저가 −5% 손절 → 고가 +5% 익절 → 셋째 날 종가. 비용 0.30%, 규칙마다 종목당 1포지션(판 날 다음 날부터), 실적일 제외 안 함(비교 기준도 같은 조건).
- 대상 = mr1-2(거래대금 상위 절반 중 변동성 상위 절반, `mr_eligibility` 그대로). 종가 진입은 t일, 지정가는 주문을 내는 t−1일 대상.
- 비교 기준 = 같은 종목이 대상이던 같은 기간 모든 날의 종가 진입(같은 청산). 값 = 거래 순수익 − 그 종목 평균, t = 월 묶음. 승률은 참고만 — 익절·손절 폭이 같으면 아무 날에 사도 승률이 거의 일정하기 때문.
- 판정: 찾기 = 기존 545종목 2001~2019, 거래 ≥ 300·종목 ≥ 50인 규칙끼리 섞기 200회 통과선(같은 종목·신호일은 모든 규칙에서 같은 무작위 날), 차이 > 0, t ≥ max(통과선, 2), 미국·한국 같은 방향, 달 쏠림 점검. 검증 = 후보만 같은 실행에서 같은 종목 2020년~(처음 여는 기간), 차이 > 0 · t ≥ 2 · 거래 ≥ 100. 최종 = 검증 통과(최대 5개, 검증 t 순)만 `--mr2-final`로 새 151종목 2001~ 한 번, t ≥ 3 · 거래 ≥ 100 · 미국·한국 같은 방향. 검증 통과가 없으면 새 종목은 받지도 열지도 않는다.
- **지정가 규칙은 엄격 체결로도 넘어야 한다(결과 전 사용자 결정).** 엄격 체결 = 시가 ≤ P이거나 저가 ≤ P × 0.999. 이유: '저가가 닿으면 체결'은 가격이 P에 딱 닿고 튄 날(실제로는 순서가 안 와 체결이 어려운, 하필 가장 좋은 날)을 넣는다. 코드 점검에서 저가를 종가와 따로 흔든 가짜 가격(무작위 걸음)으로 돌리자 우위가 없는데도 지정가 규칙이 t 4~11로 나왔고, 장중 경로까지 무작위 걸음인 가짜 가격에서는 사라졌다(테스트는 이 두 번째 가짜 가격을 쓴다).
- 한계: 종가 진입은 실제로 마감 동시호가·MOC로 해야 하고 마감 직전 가격과의 차이는 시험에 없다. 같은 날 순서를 몰라 둘 다면 손절(보수적).
- 실행: `run_backtest.py --mr2`(강의 시험 자료 그대로) → MR2.md·mr2.json·EXPLORE_LOG 26줄. 최종 `--mr2-final`(받기 + 한국 이름 출력) → 확인 뒤 `--mr2-final --names-ok` → MR2_FINAL.md·mr2_final.json. 테스트 tests/test_mr2.py.
- **결과(2026-11-19, MR2.md): 미통과 — 새 151종목은 열지 않았다.** 통과선 t 2.12(26개 경쟁, 서로 많이 겹쳐 낮음), PBO 23%. 아무 날 같은 청산은 거래당 −0.21%(비용 전 +0.09% — 3일 단타는 비용 대부분을 떠안는다). 찾기 후보 3개 → 검증(2020~) 모두 t < 2: S04b 4일 연속 하락 종가 +0.20%p(t 2.5) → +0.19%p(t 1.4), S07 RSI(2) < 10 + 200일선 위 종가 +0.15%p(2.2) → +0.11%p(1.0), S05 5일 −3 ATR 지정가 +0.29%p(2.9) → +0.37%p(1.5, 엄격 +0.32%p 1.3). 방향·크기는 비슷했지만 검증 표본으로는 찾기 추정치가 참이어도 t ≥ 2 확률이 약 20~30%였다 — 그래서 다음 사전 등록부터 검출 가능한 최소 차이를 미리 계산한다. 비용 뒤 거래당 순수익은 후보도 0 근처(검증 S04b +0.01%, S07 −0.03%, S05-L +0.19%). 참고(판정 아님): 같은 날 같은 시장 대상 종목 대비로는 연속 하락·RSI(2) 과매도 뒤 +0.1~0.15%p(t 4~6) — 종목 고유의 짧은 반전은 있으나 매수만으로는 시장 흔들림에 묻힌다. 같은 신호에서 지정가가 종가 진입보다 모두 나았고(엄격 체결로 대략 절반), 급락 당일 종가 매수는 아무 날보다 나빴다(S01b 3 ATR −0.57%p t −3.5, S13 대량 음봉 −0.21%p t −3.4 — 찾기 기간만, 2020년 이후는 안 엶).

### H-MR3 변동성 국면 이격도·지지 반등 (2026-11-19 사전 등록 — 결과 보기 전 고정)
MR2 뒤 사용자 제안: '상시 단타'는 버리고, 단기 반전 수익이 VIX와 함께 커진다는 Nagel(2012)을 근거로 변동성 국면에서만 이격도·지지 규칙을 켜고, 그 밖에는 '단타 부적합'으로 표시한다. **전체 정의는 `data/backtest/MR3_PREREG.md`, 숫자는 `engine/mr3_backtest.py` `MR3_CONFIG`.** 사용자 결정: VIX 기준 20(처음 제안한 상위 5~10%가 아니라 1993년 이후 약 43%의 날 — '평균보다 출렁이는 국면'), 나머지는 Claude 추천대로.
- 국면 = 그 종목 날짜보다 앞선 가장 최근 미국 거래일 VIX 종가 ≥ 20(VIX는 미국 장 마감 뒤 확정, 한국 장은 그 전에 끝남 — 당일 값은 미래 자료). 한국 종목도 같은 VIX. 대상은 mr1-2 그대로.
- 규칙 5개(정의는 MR2 S10·S12 그대로): V1 이격도 기준 처음 도달 종가, V2 같은 기준 가격 지정가, V3 지지 구간 종가, V4 지지 구간 상단 지정가, V5 이격도 ≤ 기준 + V3 같은 날 종가. 지정가는 엄격 체결 조건.
- 청산(이격도 회복): 손절 = 진입가 − 2×ATR_{t−1}(장중, 시가가 아래면 시가), 종가 ≥ 25일선이면 그날 종가, 최대 10거래일, 같은 날 둘 다면 손절. 지정가 체결일에도 손절·회복을 본다(둘 다 체결 뒤가 확실).
- 비교 기준 = 같은 종목·같은 단계의 대상이면서 국면인 날에 종가로 사서 그 거래의 보유일(체결일 청산은 1일) 뒤 종가에 판 평균. 25일선 청산은 가격에 따라 보유일이 달라 MR2처럼 '아무 날에 같은 청산'을 쓰면 비교가 안 되기 때문(MR1과 같은 보유 기간 맞춤). 섞기 통과선도 같은 무작위 날·같은 보유일.
- 판정: 찾기 = 545종목 2001~2019(이미 쓴 자료 — MR1 🔴 등급 +2.10% 같은 관찰이 동기라 증거 아님), 차이 > 0, t ≥ 통과선, **비용 뒤 순수익 > 0**, 거래 ≥ 300·종목 ≥ 50, 미국·한국 같은 방향, **기여가 가장 큰 해를 빼도 t ≥ 2, 거래 30건 이상인 해의 3분의 2 이상에서 차이 > 0**(위기는 몇 번의 사건에 몰리므로 '달 빼기' 대신). 검증 = 후보만 VIX로 나눠 본 적 없는 미국 1990~2000(1989년부터 새로 받음, 후보가 있을 때만) + 545종목 2020~을 합쳐 t ≥ 2·거래 ≥ 100·순수익 > 0, 30건 이상인 구간은 각각 같은 방향. 최종 = 검증 통과만 `--mr3-final`로 새 151종목(새 종목이지 새 위기는 아님), t ≥ 3.
- 참고: VIX 20~30 · 30 이상 나눠 본 차이, 시점 효과(대상인 날 중 국면 vs 평상시 5·10일 종가 보유), 해별·기간별, **검출 가능한 최소 차이**((기준 z + 0.84) × 표준오차). 결과 전 추정: 찾기 V1·V2 약 1.0%p, V3·V4 약 0.4%p.
- 앱 표시: 최종 통과 전에는 단타를 '적합'이라 쓰지 않는다(아래 결과 뒤 반영 내용).
- 실행: `run_backtest.py --mr3`(VIX는 lecture_cache/prices/^VIX.csv, 1990년대 미국 자료는 lecture_cache/prices_1989/ — 후보가 있을 때만 받음) → MR3.md·mr3.json·EXPLORE_LOG 5줄. 최종 `--mr3-final` → 확인 뒤 `--mr3-final --names-ok`. 테스트 tests/test_mr3.py.
- **결과(2026-11-19, MR3.md): 찾기 후보 없음 — 1990년대·2020년 이후·새 151종목은 열지 않았다.** 찾기 기간 대상 날의 33%가 국면. V1 이격도 종가 1,997건(91개월) 같은 종목 같은 보유일 대비 +1.55%p(t 1.9, 비용 뒤 거래당 +1.73%), 해별 9/10 양수, 2018년을 빼면 t 1.5. V5 이격도 + 지지 2,274건 +1.44%p(t 1.8), 2018년 빼면 t 1.4. V2 +0.92%p(t 1.2). V3·V4 지지 구간 46,193·66,798건 −0.07%p·+0.04%p(검출 가능한 최소 차이 0.40·0.29%p) — 지지선은 변동성 국면에서도 효과 없음. 시점 효과(국면 vs 평상시 5·10일 보유) z 0.8·0.6. 참고: V1은 VIX 30 이상에서 +2.65%p(t 1.4), 20~30에서 +0.81%p. **결과 전 검정력 추정이 2배 넘게 낙관적이었다**(V1 최소 차이 1.0%p 예상 → 실제 2.34%p): 국면 거래가 같은 달에 몰리고 10일 보유의 흔들림이 커서. V1을 새 자료로 다시 확인해도 진짜 효과가 1.5%p라도 t ≥ 2 확률이 약 3분의 1이라 다시 시험하지 않는다(2026-11-19 Claude 추천).
- **결과 뒤 딥다이브 반영(2026-11-19 사용자 결정)**: 세 방식 한눈에의 단타 줄 — VIX < 20 '부적합'(MR2 근거), VIX ≥ 20 '주의'. 국면이면 MR3에서 가장 나았던 조합 V5(이격도 ≤ 기준 + 지지 구간 지킴)의 충족 여부를 사실로만 보여 주고(충족이면 '주의 · 조합 충족(미검증)' + 손절 = 종가 − 2×ATR(전날)·25일선 회복 매도·최대 10거래일·소액), '맞음'은 내지 않는다. 대상 조건(거래대금·변동성 상위)은 종목 하나로 잴 수 없어 보지 않는다(engine/mr3_backtest.py `mr3_short_status`). V1(이격도만)도 V5와 비슷했고 지지 구간 단독은 0이었다 — 조합의 우위는 대부분 이격도 쪽에서 온다.
- **섞기 통과선 결함(2026-11-19 MR3 결과에서 발견, 아직 안 고침)**: `lec_reality_check`·`mr2_reality`·`mr3_reality`는 거래마다 무작위 날을 따로 뽑는데 t는 같은 달 거래를 묶은 표준오차로 잰다. 실제 거래는 같은 날·같은 달에 몰려 함께 움직이지만 따로 뽑은 날은 그렇지 않아 섞은 t가 덜 흩어지고 통과선이 낮게 나온다(MR3 규칙 5개의 섞은 최고 t 95% 분위 1.08 — 규칙 하나여도 1.64여야 함). 낮은 통과선은 후보를 쉽게 만드는 쪽이라 지금까지 미통과였던 L2·L4·MR2·MR3 결론은 바뀌지 않는다(MR3는 최소 2가 걸림). 다음 큰 가족 시험 전에 날짜 단위 섞기(같은 신호일의 거래는 같은 무작위 날)로 고치고, 독립 규칙 하나일 때 95% 분위 ≈ 1.64인지 보는 테스트를 넣는다.

### H-TR 추세 투자 흐름 검증 (2026-11-20 사전 등록 — 결과 보기 전 고정)
사용자 요청: 1번 시장 → 2번 RS 주도주 → 5번 120일선·이평선 쌍바닥 → 3·4·6번(판독 순서 + 물량 소화 + 돌파 확정 4조건/SR Flip) → 9번(악재 제외 + 리스크 수량) → 7·8번(채널·박스 복사 목표가 + 추세 이탈 청산) 흐름을 시험. **전체 정의는 `data/backtest/TR_PREREG.md`, 숫자는 `engine/tr_backtest.py` `TR_CONFIG`.** 사용자 결정(T1~T5): Claude 추천대로.
- 판정 규칙 두 개(진입만 다름): A 돌파 확정(E02) 다음 날 시가, B SR Flip(E04) 다음 날 시가. 둘뿐이라 섞기 통과선을 쓰지 않는다(그 결함과 무관). 판정 규칙을 늘리려면 결함부터 고친다.
- 조건(신호일 종가까지): 시장 🟢·🟡(F02) · RS 주도(F06) · 120일선 상승(F08) · 실적 쇼크 제외(박스 시작~신호일 발표 반응 ≤ −2 ATR, 좋은 실적 돌파는 남김) · 진입가 > 무효선(돌파선 구간 하단) · 손절폭 ≤ 12% · 박스 폭 ≥ 5% · 진입가 < 1대1 목표. 이평 쌍바닥은 참고 줄(필수 아님).
- 구현에서 정한 것(결과 전): 박스 시작 = 돌파선 고점에서 과거로 같은 구간 안의 가장 이른 확정 고점(구간 위 고점·250거래일에서 멈춤, 낮은 고점은 지나감), 바닥 = 박스 시작~돌파일 최저 저가, 목표 = 돌파선 + (돌파선 − 바닥).
- **T1 물량 소화**: 기존 C04는 구조상 거의 생기지 않는다(돌파선이 '가장 최근 확정 고점'이라 그 사이 확정 저점이 보통 1개 — L2에서 돌파 확정의 2%). 박스 고점 ≥ 2 + 박스 안 확정 저점 마지막 둘이 오름 + 둘 다 구간 하단 아래로 고쳤고, **A + 물량 소화의 탐색 기간 신호 수 ≥ 300이면 판정 사슬에 넣고 아니면 참고** — 코드가 자동으로 정한다.
- 청산(하루 순서): 전날 정한 매도(시가) → 장중 목표(전부) → 250거래일째 종가 → 종가로 무효(전부)·붕괴 W10(진입 뒤 지지 이탈, 전부)·추세선 장대음봉 이탈(절반, 한 번). 채널 목표는 쓰지 않음(채널 돌파 매매용). 비용 0.30%, 종목당 1포지션.
- 비교 = 같은 종목·같은 시험 기간 무작위 날 20곳에서 같은 날 간격·같은 비율로 판 순수익(보유 기간 맞춤, 시가 매도는 시가·나머지는 종가). 탐색은 2013-12-31 뒤 가격을 보지 않는다(기간 끝 = 그날 종가).
- 단계: 탐색 545종목 2001~2013(부품을 L2에서 본 기간이라 참고 성격) → 후보만 확인 2014~ → 확인 통과만 최종 새 151종목(z ≥ 3). 참고: 셋업만·단계 빼기·🟢만·이평 쌍바닥·청산 단순형·1% 위험 수량·기간 내내 보유.
- 실행: `run_backtest.py --tr-count --etf-ok`(실행 전 깔때기·T1 결정·가려낼 수 있는 최소 차이, 수익 안 봄 → TR_COUNT.md) → 사용자 확인 → `--tr --etf-ok`(TR.md·tr.json·EXPLORE_LOG) → 통과 시 `--tr-final` → `--tr-final --names-ok`(새 종목 섹터·실적일 받기 포함). 테스트 tests/test_tr.py(우위를 심은 가짜 거래는 후보, 같은 보유로 들고만 있는 가짜 거래는 후보 아님).
- 결과 전 예상: A 약 1,000건 → 거래당 약 +1.7%p, B 약 350건 → 약 +3%p 이상만 가려냄. 지금까지의 차이(+0.2~0.3%p)보다 커서 미통과는 '이만큼 큰 효과는 없음'으로 읽는다.
- 실행 전 건수(TR_COUNT.md, 2026-11-20): ①~⑧ 통과 신호 A 686 · B 259, 물량 소화 포함 A 177(< 300 → 참고 줄). 박스 폭 ≥ 5% 조건이 절반 가까이 걸렀다(결과 전 추정 1,000건은 이 조건을 빼고 계산했음). 가려낼 수 있는 최소 차이 A 1.5~2.2%p · B 2.5~3.6%p(20~40일 보유, 월 묶음 감안).
- **개정 tr-2(같은 날, 건수만 본 뒤 · 수익 계산 전 · 사용자 결정)**: 탐색 최소 거래 300 → 200(A·B 같게), 해별 방향 점검은 거래 30건 이상인 해가 3개 이상일 때만(적으면 '적용 불가' 표시, 해 빼기 점검은 그대로). 판정 규칙 수는 그대로 2개. 근거와 대가는 TR_PREREG.md 9장.
- **결과(2026-11-20, TR.md · e8c7cad): 탐색 후보 없음 — 확인(2014~)·최종(새 151종목)은 열지 않았다.** A 돌파 확정 680건(313종목, 104개월): 비용 뒤 거래당 +0.73% vs 같은 종목 같은 보유 무작위 +0.87% → 차이 −0.14%p(t −0.4, 표준오차 0.34%p — 95% 범위 약 −0.8~+0.5%p, 사후 최소 차이 1.0%p로 결과 전 추정보다 검정력이 높았음: 평균 보유 17일). 승률 64%(목표 65% +4.4% · 무효 32% −6.5%)지만 같은 날수만 들고 있어도 그만큼 벌었다. 해별 4/10, 2003년 빼면 −0.57%p. 셋업만 대비 −0.08%p(돌파 신호가 보태는 것 없음). B SR Flip 258건: −0.64%p(t −1.7, 무효 60%). 단계 빼기·🟢만·이평 쌍바닥·물량 소화·청산 단순형 모두 0 근처이거나 음수. 참고: 1% 위험 수량 연 합계 +7.5%(겹침·복리 무시) vs 같은 종목 기간 내내 보유 연 +12.1%(현재 상장 종목이라 생존 편향 있음 — 판정 비교는 같은 종목이라 영향 없음).

### 다음 작업 순서
1. ~~틀 v2 정의~~ (2026-11-14, 위 표): 결과를 보고 숫자를 맞추지 않고, v1 검증에서 드러난 원인에 맞춰 정의와 역할을 바꿨다.
2. ~~틀 판정 모듈~~ 완료(2026-11-14, engine/playbook.py): 종목 하나에 대해 틀 A v2(추세 상태·근거 저울 항목별 근거·전환 가능성·의견 단계·청산 조건)와 틀 B v2(적용 범위·진입 조건·경고·손절·매도)를 판정하고, 각 항목의 검증 상태를 함께 돌려준다. 가짜 데이터 테스트 포함.
3. ~~신호 기록 장부~~ 시작(2026-11-15, engine/ledger.py): 매일 대표 종목·관심 종목의 틀 A v2 판정과 원자료를 data/ledger/에 쌓는다. 채점 방법은 기록 전에 LEDGER_EVAL로 고정했다 — ① 하락·횡보 상태의 전환 가능성 표시(우세·팽팽·실패 우세)별 20·60거래일 초과 수익(같은 날 다른 종목 대비)과 상승 추세 도달 비율, ② 진입 의견 거래(다음 날 시가 매수, 하락 쪽 근거가 많아진 다음 날 시가 매도)의 초과 수익. 판정은 기록 12개월·의견 30건 이상에서 평균 > 0, 월 묶음 z ≥ 2. 그 전에는 '판정 보류'.
4. ~~종합 화면 개편~~ 완료(2026-11-15): 딥다이브 종합 탭 틀 A·B 카드, 기술 트랙 탭 틀 A 요약, 홈 실행 레이더를 신호 기록 장부 기반으로 교체.
5. 딥다이브 개편 (위 '투자 방향과 딥다이브 개편 기획'):
   - ~~0단계: 생존성 v2.1~~ 완료(2026-11-15) — 희석 킬 한정, 금융 자회사 판정 보류.
   - 1단계: 피어 데이터층 — 코드 완료(2026-11-15: engine/peers.py, peers_daily.py, check_peers.py, peer_map.csv 한국 4묶음). 첫 점검(peer_census_2026-11-15.md, AMZN·NVDA 부분 갱신 39종목)에서 정의·데이터 문제 2개 → v1.1: ① 매출 성장률이 전 종목 빈칸(야후 분기 손익은 4~5분기뿐이라 '최근 4분기 대 그 전 4분기'를 못 만듦) → 최근 분기 대 전년 동기(모자라면 야후 revenueGrowth, 같은 정의) ② AMZN처럼 분기 손익 한 칸이 비면 영업이익률이 빈칸 → 야후 operatingMargins로 대신. 가중치·기준은 그대로. 기록 파일에 새 열이 생기면 기존 줄은 그대로 두고 빈 새 열을 붙인다(engine/peers.py `_append`). 첫 전체 갱신(2026-11-16: 커버리지 157 → 62묶음·564종목) 점검 반영 완료 — peer_map 제외 7·옮기기 6·보태기, 생존성 v2.2.
   - ~~2단계: 딥다이브 위쪽~~ 완료(2026-11-16): 탭 위에 '이 종목에 투자할 이유 — 피어 비교' — 결론 카드(묶음·피어 수, 가치 관점·추세 관점 순위와 점수, 상위 1/3 표시, 각 1위, 강점·약점, 생존성, 기준일·다음 실적 발표일, 검증 상태)와 피어 비교 표(이 종목 강조). 표시 규칙(결과 보기 전 고정, engine/peers.py `SUMMARY_RULES`): 강점 = 요인 점수 ≥ 60(높은 순 최대 3개), 약점 = ≤ 40(낮은 순 최대 3개), 그 사이는 보통. 요인마다 이 종목 지표 값과 피어 중앙값(하드 킬 제외)을 함께 쓴다. 자료: `latest_peer_view`(순위 기록 + 같은 날 재무 기록의 회사 이름·생존성 문구·다음 실적 발표일) → `peer_summary`. 커버리지 밖 종목은 watchlist 안내.
   - ~~3단계: 앞서려면 조건~~ 완료(2026-11-16, engine/peers.py `peer_gaps`·`GAP_RULES`, 딥다이브 `render_peer_gaps`): 지표마다 피어 상위 25% 경계값(하드 킬 제외, 클수록 좋으면 75% 분위·작을수록 좋으면 25% 분위, 선형 보간)까지 남은 차이를 산수로 — 선행 PER은 가격 경로(목표 PER × FY1 EPS 이하의 주가, %)와 실적 경로(주가 그대로일 때 필요한 FY1 EPS, %), PEG는 주가 변화율 또는 필요한 선행 EPS 성장, 순상향은 필요한 '상향 − 하향' 건수(기준 × 애널리스트 수, 올림), 상대강도는 섹터 ETF 대비 더 올라야 할 %p, 추세 상태는 필요한 상태. 가장 가까운 경로 = 점수 75 미만 요인 하나를 75로 올렸을 때 전략 점수가 가장 많이 오르는 요인(다른 종목은 그대로, 새 순위 함께). 이미 1위면 2위와 점수 차(같으면 공동 1위). 실적·추정치 조건은 '다음 실적 발표 때 확인할 것'으로 모아 보여 준다.
   - 4단계: 전략 선택 — 단기 변경(2026-11-17): 진입 후보에서 빼고 지지·손절선 카드로, 선 검증(--levels) 결과 둘 다 미통과 → 화면 문구 '검증 안 됨'. 코드 완료(2026-11-16): 단기 규칙 판정·우선순위·딥다이브 카드 3장, 단기 규칙 시험 코드(engine/short_backtest.py, 기준 사전 고정). 단기 규칙 시험 결과(2026-11-16) 미통과 → 단기 카드·결론 줄에 '검증 미통과 — 참고만'(규칙은 그대로). **세 방식 한눈에(2026-11-19 사용자 요청)**: 딥다이브 맨 위 결론 줄 + 전략 섹션 첫 카드에 '지금 이 종목에 맞는 방식'(가치 스윙·추세·단타)과 방식별 한 줄 이유(✅ 맞음 · ❌ 안 맞음/부적합 · ⚠️ 주의 · — 판정 불가). 판정은 choose_strategy 그대로(새 기준 없음): 가치 = 틀 B '진입 가능'·'조건부 진입' + 가치 순위 상위 1/3, 추세 = 틀 A 상승 전환 의견 + 추세 순위 상위 1/3, 둘 다면 가치, 생존성 하드 킬·국면 크기 0은 결론 줄에. 이유는 결정적인 사실 최대 2개(충족이면 근거, 아니면 막은 조건 — 틀 B 단계와 가격 자격, 틀 A는 '이미 상승 추세'·'근거 차이 부족'·'50일선'). 맞는 방식이 없으면 충족 조건이 더 많은 쪽(같으면 가치)을 '가장 가까운 방식'과 남은 조건으로. 단타는 MR3 아래 내용. 세 방식 모두 수익 미검증이라고 함께 적는다.
   - ~~5단계: 피어 비교 페이지 + 홈 연결~~ 완료(2026-11-17): pages/2_🧭_피어_비교.py — 요약 카드(묶음·순위 수·가중치·검증 상태, 63일 상대강도 탐색 결과 주의), 종목으로 찾기·시장·'내 종목 있는 묶음만'(내 종목 = 장부 대상: 대표 종목 + watchlist — 커버리지는 ETF 상위 구성 종목까지라 모든 묶음에 있어 거르기에 못 씀), 묶음 한눈에 보기(1위·순위 없음 사유), 가치·추세 관점 순위표(상위 1/3 초록, 내 종목 ★), 딥다이브로 이동. 홈 6번 '피어 비교'(내 종목이 있는 묶음 상위 6개의 1위 + 페이지 링크), 사이드바 링크. 검사 tests/test_peer_page.py(fake·none), test_home 피어 칸.
   - 6단계: ~~피어 순위 채점 방법 고정~~ 완료(2026-11-17, engine/peer_eval.py, tests/test_peer_eval.py). 남은 것: Streamlit Cloud 배포(사용자 계정, 원할 때 — push마다 자동으로 다시 띄워지므로 개발을 멈출 필요 없음).
6. 연구(분기마다 한 질문, 사전 등록·홀드아웃): 국면 게이트를 비중 조절에 쓸 때 지수 보유의 위험 대비 수익 개선 여부, 한국 틀 B 재현 시험(DART 공시일 기준, v1 규칙 그대로), 틀 B 장부 기록. 같은 미국 자료로 새 틀을 만들지 않고 트랙을 늘리지 않는다.
7. 출구(미리 정함): 2027-11 장부 채점에서 틀 A v2가 미통과면 기술 트랙은 위험 관리·차트 표시 역할만, 틀 B v2도 같은 방식.

### 검증 상태 (data/backtest/)
- 국면 게이트: 이후 변동성·낙폭은 구분하지만 수익률은 구분하지 못한다(REPORT.md). 후보 A·B·C는 사전 기준 탈락, 현재 규칙 유지(VARIANTS.md).
- 기존 T 셋업: 단순화 규칙(SETUP_DIAG.md)과 실제 스캐너(SCANNER.md, ETF 19개 2014~2026, 251건 −0.15R vs 무작위 +0.06R) 모두 무작위 진입보다 낫지 않음.
- 재설계 후보(REDESIGN.md): T-A 추세 추종 미통과. T-B 눌림목 반등 통과(처음 보는 해외 국가 ETF에서 +0.01R vs 무작위 −0.06R, z +5.5, 승률 62%, 평균 3.7일 보유). 단 왕복 비용 0.30% 차감 후 거래당 약 +0.10%로 비용에 민감하고 손절이 없는 규칙이다. 단독 매매 신호가 아니라 틀 A의 진입 조건으로 쓴다.
- 틀 A(FRAME_A.md, 2026-11-14): **미통과.** HOLDOUT(새 개별 종목 82개, 2001~) 거래당 +0.78% vs 보유 기간 맞춘 무작위 +0.74%(차이 +0.04%p, z +0.1), 뒤 기간 −0.20%p. 눌림 타이밍도 '선정만'보다 낫지 않음(z +0.6). 주식을 들고 있던 기간 18%에 연 +0.3%로 매수 후 보유(+6.1%)에 크게 못 미침. 손실 꼬리는 줄었음(최악 −42% vs 선정만 −54%, −10% 갭 하락 경험 0.7%).
- 단기 규칙(SHORT_TERM.md, 2026-11-16): **미통과.** HOLDOUT(새 종목 82개, 2001~, 2,543건) 비용 0.30% 뒤 거래당 −0.25% vs 같은 보유 기간 무작위 −0.20%(차이 −0.05%p, z −0.5), 비용 전 +0.05%. 앞 기간 +0.11%p(z +0.8)·뒤 기간 −0.18%p(z −1.4). RSI(2) 눌림의 기여 없음('눌림 없이' 대비 z −0.3). +3% 익절 41% · 손절 38% · 추세 하락 14% · 시간 7%로 3%/3% 대칭 구조가 동전 던지기에 가깝고 비용이 그대로 손실이 된다. 개발 묶음도 앞·뒤 모두 음수. 규칙은 바꾸지 않고 화면 단기 카드는 '검증 미통과 — 참고만'.
- 지지선·손절선(LEVELS.md, 2026-11-17): **둘 다 미통과.** HOLDOUT(새 종목 108개 — HOLX는 못 받음, 2001~). 지지선: 접근 뒤 10거래일 안에 깨진 비율 51.5%(12,131건) vs 최근 하락 폭을 맞춘 같은 거리의 가짜 선 48.6% — 덜 깨지기는커녕 +2.8%p 더 깨짐(z +2.4, 기준은 z ≤ −3), 개발 묶음도 같은 방향(+2.6%p). 손절선: 이탈 뒤 10거래일 수익률 +0.53%(6,083건) vs 무작위 날 +0.52%(차이 0.00%p, z 0.0), 20거래일 −10% 이하 비율 12.0% vs 10.4%(z +1.0). 스윙 저점은 '버티는 선'도 '깨지면 더 떨어지는 선'도 아니었다. 화면 문구는 미리 정한 '검증 안 됨'으로, 손절선까지 거리의 위험 계산(산수)만 남는다. 선의 정의를 바꿔 다시 시험하지 않는다.
- 지지선 이탈 뒤 회복(RECLAIM.md, 2026-11-17): **미통과.** HOLDOUT(새 종목 146개 — CTRA는 못 받음). 집계: 지지선 접근 뒤 10거래일 버팀 36.2% · 깼다가 회복 38.8% · 못 돌아옴 25.0%(14,779건), 이탈 뒤 회복 68.4%(8,474건, 중앙 2일, 회복한 경우 깊이 중앙 0.5 ATR·못 돌아온 경우 2.9 ATR), 회복 뒤 10거래일 안에 다시 이탈 64.5%. 효과 없는 무작위 걸음의 43%·72%와 비슷하거나 낮다 — '자주 보이는 것'은 무작위에서도 나오는 모양. 판정: 회복 다음 날 시가 → 10거래일 +0.55%(5,794건) vs 무작위 날 +0.52%(차이 +0.03%p, z +0.1), 개발 묶음은 −0.24%p. 화면은 미리 정한 '검증 안 됨' 문구. 정의를 바꿔 다시 시험하지 않는다.
- 탐색(증거 아님, data/backtest/EXPLORE_LOG.md, 2026-11-17, 지금까지 쓴 545종목): `touch4`·`touch4b` 후보 아님 — 수평선은 몇 번째 터치든 10거래일 뚫림 약 54%, 20거래일 확정 뚫림 약 19%로 평평(무작위 걸음과 비슷). `mom12` 후보 아님 — 1개월 상위 20% − 하위 20% +0.34%p(t +1.1, 309개월), 앞·뒤 기간 모두 +, 미국 −0.12%p·한국 +0.79%p(t +2.0), 최악의 달 −26%p. 강한 종목만 산 경우(상위 − 전체 평균) 1개월 +0.30%p(t +1.9), 3개월 +1.00%p(t +2.7)는 참고 — 판정 기준이 아니고 결과를 본 뒤 고른 칸이라 후보로 올리지 않는다. `ret63`(앱 rs63과 같은 기간) 후보 아님 — 1개월 상위 − 하위 −0.09%p(t −0.3). **피어 비교 추세 관점의 핵심 요인(가격 흐름의 63일 상대강도)은 이 탐색에서 다음 달 수익을 가르지 못했다.** 순위 규칙은 사전 등록대로 두고 앞으로의 채점(상위·하위 1/3, 12개월 뒤)으로 판정한다.
- 가격 기반 타이밍 전체: 게이트 수익 예측·기존 셋업·실제 스캐너·T-A·T-B·틀 A·단기 규칙 일곱 번, 그 뒤 강의 L1~L5·H-MR1~3·H-TR까지 모두 우위가 없거나(T-B만 거래당 +0.10%) 미미했다. 같은 과거 자료로 기준을 바꿔 가며 더 찾지 않는다.
- 틀 B v1(FRAME_B.md, 2026-11-14): **미통과(우위 미확인).** SEC 공시일 기준 EPS는 야후 값과 거의 같게 읽힘(AAPL 8.98 vs 9.02 등). HOLDOUT(새 미국 종목 63개, 147건) 로그 순수익률 +14.2% vs 같은 날 다른 종목 +12.7%(차이 +1.6%p, z +1.1), 개발 묶음은 −3.4%p(z −1.7). 이익 조건의 기여 없음('싼 다른 종목' 대비 −0.4%p). 청산의 약 70%가 이익 감소 손절이고 그 거래들이 다른 종목보다 뒤짐(−2~−7%p), 배수 주도 고평가 매도는 +4~+10%p. 거래가 147건뿐이라 연 2~3% 크기의 우위는 잡아낼 수 없는 표본이다 — '우위 없음'이 아니라 '확인 안 됨'. 같은 미국 자료로 규칙을 바꿔 다시 시험하지 않는다.
- 강의 조건 시험(LECTURE_PREREG.md): 2026-11-18 사전 등록. 1단계 L1·L3 미통과(LECTURE_L1.md·LECTURE_L3.md). L1~L5 모두 미통과(LECTURE_L1~L5.md). H-MR1 단기 반등 단타 탐색 후보 없음(MR1.md). H-MR2 단기 신호 찾기 미통과(찾기 후보 3개 → 검증 t < 2, MR2.md). H-MR3 변동성 국면 이격도·지지 반등 찾기 후보 없음(MR3.md — 이격도는 +1.5%p 안팎이지만 t 1.8~1.9, 지지 구간은 효과 없음).
- H-TR 추세 투자 흐름(TR.md, 2026-11-20): **미통과** — 탐색 후보 없음. A 돌파 확정 −0.14%p(t −0.4, 680건), B SR Flip −0.64%p(t −1.7, 258건). 강의 흐름 전체(시장·RS·120일선·돌파 확정·실적 쇼크·박스 목표·추세선·붕괴)를 이어 붙여도 같은 종목을 같은 날수만큼 들고 있는 것보다 낫지 않았다.
- 매매 기록(JOURNAL_PREREG.md, 2026-11-20~): 방식별 끝난 거래 30건부터 지수 대비로 채점.
- 틀 A v2·틀 B v2: 미검증(추정치 기록은 2026-11-14부터 쌓는 중). 실적·거시 트랙과 한국 틀 B도 미검증. 화면·문구에서 검증된 것처럼 표현하지 않는다.

### 운용 규칙 (현재 코드에 적용 중)
- 규칙 우선순위: Thesis 붕괴 > 킬 조건 > 안전마진 방어 > 매도 목표가 > 매수 기준선 > 기술적 타이밍. 방어가 공격보다 항상 우선.
- 게이트 G0~G3. 최종 게이트 = 지수 레짐·VIX·거시 국면 중 가장 나쁜 값. 일부 데이터가 없으면 남은 것으로 판정.
- 손절·비중 축소 같은 방어 행동은 어떤 게이트에서도 막지 않는다.
- 장기 보유 예외(QLD, SCHD)는 게이트·매매 신호 대상이 아니다.
- G3에서 BNF 예외는 승인 비중 20%(1차분의 1/3 이하) — 위기 국면 규칙(1차분의 1/3 이하)과 충돌하지 않음.
- 게이트는 "실행 여부와 크기"만 정한다. 매수 기준선·매도 목표가 숫자는 바꾸지 않는다.

## 5. 화면 표시 규칙
- 색 의미 고정: 초록=공격 가능 · 노랑=주의 · 주황=방어 · 빨강=위기. 위기를 초록·검정으로 표시하지 않는다.
- 테마는 밝은 테마로 고정(.streamlit/config.toml `[theme] base = "light"`). 흰 배경 카드·배너를 새로 만들면 글자색도 함께 지정한다(브라우저 다크 모드에서 글자가 흰 배경에 묻히지 않게 — utils.py `.term-card`·`.term-banner`는 `color: var(--text-main)`).
- 제목에는 "무엇을 알려주는지"만 쓰고, 방법론(CMF, 13F, σ)은 작은 설명 줄이나 툴팁으로.
- 매매 틀 카드: 판정 옆에 검증 상태(미검증·의견 등)와 판정 규칙 버전을 항상 함께 쓴다. 틀 A는 '의견', '확률'이 아니라 '근거 개수 비교'로 표현하고 '매수 확정'·'확신' 같은 말을 쓰지 않는다. 데이터 없는 항목은 '데이터 없음'으로 보이고 중립으로 계산된다고 적는다.
- 이름: 앱 이름은 "투자터미널". 섹션: 오늘의 결론 / 섹터 흐름(지금 강한 곳·다음 후보) / 자금 흐름 / 실행 레이더(실행·관찰·방어).

## 6. 작업 방식
- **항상 지금 파일에서 시작한다.** 예전 대화에 남아 있던 옛 버전 코드를 통째로 붙여 넣으면 그 뒤에 고친 내용이 사라진다.
- 파일 전체를 바꾸는 대신 바뀐 함수·구간만 교체한다.
- 작업 하나가 끝나면 git 커밋: `git add <바꾼 파일>` → `git commit -m "무엇을 했는지"`. 작업 이력은 커밋 기록으로 남긴다.
- 규칙이나 설계가 바뀌면 이 파일의 해당 절(3번 계산 규칙, 4번 매매 틀)을 고친다. 작업 이력은 여기에 적지 않는다.

## 7. 확인 방법
- 문법: `python -c "import ast; ast.parse(open('app.py', encoding='utf-8').read())"`
- 실행: `streamlit run app.py` 후 화면 맨 아래 "데이터 점검" 펼쳐서 확인
- GitHub에 push하면 Actions(회귀 테스트 워크플로)가 같은 테스트를 자동 실행한다. 실패(❌)하면 Actions 탭에서 원인을 확인하고 고친다.
- 회귀 테스트: `python tests/run_all.py` (Windows: `.venv\Scripts\python tests\run_all.py`, 약 1~2분, `--fast`는 화면 검사 제외). engine/·딥다이브를 고친 뒤 반드시 실행하고, 실패하면 커밋하지 않는다. 테스트는 tests/_offline.py로 외부 연결을 막고 가짜 데이터만 쓴다. yfinance는 curl_cffi(libcurl)로 접속해 파이썬 소켓 차단을 거치지 않으므로 _offline.py가 curl_cffi 요청도 ConnectionError로 막는다(2026-11-15 — 인터넷이 되는 PC에서 테스트가 실제 야후에 접속해 멈추던 문제). 검사 하나가 10분을 넘으면 run_all.py는 '시간 초과'로 기록하고 다음 검사로 넘어간다.
- 버그를 고치면 그 버그를 재현하는 검사를 tests/에 추가하고 run_all.py 목록에 등록한다. 기대값을 바꿔야 하면 왜 바뀌는지 커밋 메시지에 적는다.
