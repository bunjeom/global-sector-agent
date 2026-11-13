# 투자터미널 — AI 작업 규칙과 인수인계 (Claude · Gemini 공용)

이 파일이 이 프로젝트 규칙의 정본이다. 어떤 AI든 코드를 고치기 전에 이 파일을 먼저 읽는다.
매매 판정의 설계 방향은 4번 "매매 틀과 운용 규칙"에 있다. 작업 이력은 git 커밋 기록으로 남긴다(이 파일에는 이력을 적지 않는다).

## 1. 프로젝트 한눈에
- Streamlit 앱. 실행: `streamlit run app.py`
- `app.py` — 홈 화면(결론 → 시장 상태 → 섹터 흐름 → 자금 흐름 → 실행 레이더 → 데이터 점검)
- `engine/` — 계산 엔진 패키지(원래 engine.py 한 파일, 2026-11-11 기능별 12개 모듈로 분리). 바깥에서는 지금처럼 `import engine` → `engine.이름`으로 쓴다. 모듈 지도는 3번 맨 위
- `utils.py` — 공통 CSS, 사이드바 메뉴(`init_session`)
- `pages/1_🔎_종목_딥다이브.py` — 종목 딥다이브
- `check_earnings_data.py` — 실적 트랙 실제 데이터 점검(PC에서 `python check_earnings_data.py AAPL 005930.KS`)
- `run_backtest.py` — 판정 규칙 백테스트(PC에서 실제 야후 데이터로 실행, 결과 `data/backtest/REPORT.md`·`summary.json`)
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
| `frame_backtest.py` | 매매 틀 검증(FRAME_A_CONFIG·FRAME_A_CRITERIA): 틀 A(추세반등)를 개별 종목으로, 보유 기간 맞춘 무작위와 비교, 월 묶음 표준오차, 갭 하락 통계 |

### 패키지 규칙
- 새 함수는 기능이 맞는 모듈에 넣는다. `engine.py` 파일을 다시 만들지 않는다(폴더 `engine/`이 우선 로드되어 새 파일은 무시됨).
- 다른 모듈 이름을 쓰면 파일 위쪽에 `from .모듈 import 이름`을 추가한다. 의존 방향은 아래 순서의 **앞쪽 모듈만** 가져다 쓴다(거꾸로 가져오면 순환 import 오류): common → technical_signals → backtest → technical → earnings → macro → fundamental → synthesis → home → gate → rrg → radar → rule_backtest → t_backtest → snapshot → frame_backtest.
- 바깥(app.py·페이지·테스트)은 `engine.이름`으로만 쓴다. 새 이름도 `engine/__init__.py`가 자동으로 내보낸다(따로 등록할 필요 없음).
- 테스트에서 `engine.이름 = 가짜함수`로 바꿔 끼우면 그 이름을 쓰는 모든 모듈에 같이 적용되고, `importlib.reload(engine)`은 모든 모듈을 처음부터 다시 읽는다(한 파일이던 시절과 같은 동작, `engine/__init__.py`의 _EngineModule).
- 같은 이름을 두 모듈에서 따로 정의하면 import 시 오류로 멈춘다(단일 출처 보호).
- 파일 경로는 실행 폴더 기준 상대경로(`data/`, `logs/`)이고, `.env`만 `PROJECT_ROOT` 기준이다.

### 계산 규칙
- 게이트 관련 함수·설정(GATE_CONFIG 등)은 한 번씩만 정의되어 있다. 새로 같은 이름을 또 정의하지 않는다.
- `_build_default_rrg_sample_dataset()`은 합성 데이터 생성 함수다. **테스트 전용**이며 화면에서 호출하지 않는다.
- 실제 4분면은 `build_real_rrg_bundle(etf_map, baskets)`(engine/rrg.py)로 만든다.
- 백테스트(engine/rule_backtest.py)는 게이트를 따로 계산하지 않고 앱과 같은 함수(build_regime_vix_header → build_gate_inputs_from_header → evaluate_market_gate)를 날짜마다 그날 이전 3년 데이터로 호출한다(앱 period="3y"와 같게). 한국 게이트는 미국 당일 데이터(QQQ·VIX)를 쓰지 않는다. 게이트 함수를 고치면 백테스트도 자동으로 따라간다.
- 백테스트 결과를 보고 같은 데이터로 임계값(GATE_CONFIG 등)을 다시 맞추지 않는다(과최적화). 바꿀 때는 기간을 나눠 앞 기간으로 정하고 뒤 기간으로 확인한다. 노출 비중(EXPOSURE_BY_GATE)은 모델 가정이다.
- 백테스트로 확인된 사실(2026-11-12, data/backtest/REPORT.md): 게이트는 이후 변동성·낙폭을 구분하지만(미국 G3 변동성 G0의 약 2배) 이후 수익률은 구분하지 못한다. G3는 기간의 41~48%이며 주원인은 지수 < EMA200(레짐 ⚫ → 즉시 G3). 화면·문구에서 게이트를 수익 예측처럼 표현하지 않는다. T 셋업(레이더 과거 검증의 단순화 규칙)은 ETF 기준 우위가 확인되지 않았고, ST-1 신호 대부분이 G3에서 나와 'G3 신규 매수 동결'과 충돌한다.
- 규칙 후보 비교(2026-11-13, data/backtest/VARIANTS.md): 게이트 후보 A·B는 사전 기준(앞 기간 G3 비중 ≤20%)에서 탈락해 현재 규칙 유지, ST-1의 G2·G3 20% 예외도 기각. 결과를 본 뒤 기준(20% 등)을 풀거나 뒤 기간 성적으로 후보를 다시 고르지 않는다. 새 후보를 시험하려면 GATE_VARIANTS에 먼저 추가하고 기준을 바꾸지 않은 채 run_backtest.py --variants로 판정한다.
- 한국 후보 C(KOSPI 자체 변동성)도 사전 기준 탈락(2026-11-13). 뒤 기간(2014~2026)을 여러 번 봐서 더는 깨끗한 검증 데이터가 아니므로 게이트 후보를 더 추가해 고르지 않는다. 게이트를 바꾸려면 앞으로 쌓이는 데이터(2026-11 이후)로 검증한다.
- T 셋업 장기 검증(data/backtest/setup_diag.json, run_backtest.py --setup-diag): ST-1·ST-2 모두 '같은 매매 방식의 무작위 진입'보다 낫지 않음(미통과). 레이더 카드는 engine.load_setup_validation으로 이 파일을 읽어 표시하며 숫자를 코드에 적지 않는다. 파일이 없으면 표시하지 않는다.
- T 트랙 재설계(2026-11-13~): 실제 스캐너는 run_backtest.py --scanner(앱 레이더와 같은 입력으로 날짜별 재현, 캐시 data/backtest/scanner_cache/), 재설계 후보는 --redesign. 후보 규칙·매개변수(55·20일, 2×ATR(20), RSI(2)<10, 5일선)와 판정 기준(TBT_CRITERIA)은 결과를 보기 전에 고정했으므로 결과를 보고 바꾸지 않는다. HOLDOUT_UNIVERSE(해외 국가 ETF)는 판정 전용이라 다른 분석에 쓰지 않는다. 결과와 이후 방향은 4번 '검증 상태'·'다음 작업 순서' 참고.
- 컨센서스 기록(engine/snapshot.py, 2026-11-14~): 판정 결과가 아니라 원자료를 저장한다. 못 받은 값은 빈칸(전날 값·추정으로 채우지 않음). `snapshot_date`는 받은 날(한국 날짜), `price_as_of`는 가격 데이터의 날짜. 열은 SNAPSHOT_COLUMNS 순서로 고정이며 새 열은 맨 뒤에만 추가한다(이미 쌓인 CSV와 어긋나지 않게). 쌓인 CSV는 손으로 고치거나 지우지 않는다 — 다시 만들 수 없는 자료다. daily_agent.yml은 Python 3.11로 돌므로 engine/ 코드는 3.11 문법을 지킨다.
- 틀 A 검증(engine/frame_backtest.py, run_backtest.py --frame-a): 규칙·매개변수(6·12개월 상대강도, 200·50일선, RSI(2)<10, 60일 최저가)와 판정 기준(FRAME_A_CRITERIA)은 실제 데이터 결과를 보기 전에 고정했다. 결과를 보고 바꾸지 않는다. 무작위 비교는 반드시 **보유 기간을 맞춘다**(주가는 평균적으로 오르므로, 같은 청산 규칙을 무작위 날짜에 쓰면 보유가 짧아져 우위가 없는 가짜 가격에서도 통과로 나온다 — tests/test_frame_backtest.py 4-7이 이를 막는다). HOLDOUT_UNIVERSE(새 개별 종목)는 판정 전용이다.
- 국면 게이트 입력은 `build_gate_inputs_from_header(macro_header, "US"|"KR")`로 홈 헤더의 실제 데이터에서 만든다.
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
### 방향 (2026-11-13 확정)
진입 신호를 따로 찾지 않는다. **매수 논리 하나를 '진입 → 논리 무효화(손절) → 매도'까지 한 묶음(틀)으로 설계**하고, 4개 트랙은 그 논리가 성립하는지·깨졌는지를 평가하는 역할을 맡는다.
- 손절은 가격 폭이나 보유 기간이 아니라 **매수 논리가 틀렸다고 판단될 때** 한다.
- 매도는 보유 기간이 아니라 **하락 추세 전환** 또는 **고평가**(재평가와 구별)로 한다.
- 근거: 가격만 보는 단기 진입 신호는 백테스트에서 우위가 없었다(아래 '검증 상태'). 실적·가치·거시 트랙은 과거 시점 데이터가 없어 검증하지 못했을 뿐 틀렸다고 확인된 적은 없다.

### 트랙의 역할
| 트랙 | 맡는 일 |
|---|---|
| 기술(T) | 진입 타이밍 보조, 추세 논리의 무효화 판정, 하락 추세 전환 매도. 수익의 원천으로 쓰지 않는다 |
| 실적(E) | 가치 함정 판별(이익 추정치가 꺾이는가), 재평가와 일시적 고평가의 구별 |
| 가치(V) | 저평가 진입 구간(매수선), 고평가 매도 |
| 거시·국면 게이트(M) | 진입 크기 조절. 위험(변동성) 온도계이며 수익 예측 신호가 아니다 |

### 두 가지 틀
| | A. 추세반등 | B. 저평가 매수 |
|---|---|---|
| 매수 논리 | 상승 추세가 살아 있고 지금 하락은 일시적 눌림이다 | 가격이 가치보다 싸고, 가치가 훼손되고 있지 않다 |
| 종목 선정 | 6~12개월 상대강도 상위, 200일선 위 | V 가격 자격 충족(매수선 도달) |
| 진입 | 눌림 발생(T-B: 200일선 위 + RSI(2) < 10) + 이익 추정치가 꺾이지 않음 | 매수선 도달 + 이익 추정치 안정·상향 |
| 손절 = 논리 무효화 | 눌림이 아니라 추세 전환으로 확인됨(200일선·직전 저점 종가 이탈) | 싼 이유가 생김: 추정치 하향으로 적정가가 현재가 아래로, 또는 킬 조건(HARD_KILL·VALUE_TRAP) |
| 매도 | 하락 추세 전환 | 고평가. 단 재평가면 보유 |
| 크기 | 게이트 단계에 따라 축소 | 게이트 단계에 따라 분할 속도 조절 |

- **재평가와 일시적 고평가의 구별**: 주가 상승을 이익 증가분과 배수 상승분으로 나눈다. 이익 추정치가 올라서 오른 것이면 재평가(적정가를 올려 잡고 보유), 이익은 그대로인데 배수만 올랐으면 일시적 고평가(분할 매도).
- 위 표의 구체 기준(상대강도 기간, 직전 저점 정의, 추정치 '꺾임'의 폭 등)은 아직 정하지 않았다. 정할 때는 후보와 판정 기준을 먼저 코드에 고정하고 검증한다(3번 '계산 규칙'의 백테스트 원칙).

### 코드 반영 상태 (이 표와 다르면 코드가 현재 동작이다)
| 항목 | 상태 |
|---|---|
| V 매수선·가격 자격, 킬 조건(HARD_KILL·VALUE_TRAP) | 있음 — engine/fundamental.py, engine/synthesis.py |
| 주가 변화의 이익·배수 분해 | 있음(후행 EPS 기준 12개월, `multiple_decomposition`) — 매도 판정에는 아직 연결 안 됨 |
| 눌림 진입 조건 T-B | 검증 코드만 있음(engine/t_backtest.py) — 레이더·딥다이브에는 없음 |
| 틀 A 전체(선정·진입·무효화·매도) | 검증 코드만 있음(engine/frame_backtest.py) — 실제 데이터 결과 대기 |
| 레이더 T 신호 | 기존 셋업 스캐너(ST-1~4) 그대로 — 장기 검증 미통과 표시 중 |
| 종합 판정 | 기존 결정 트리(ARCH_1~8) 그대로 — '어느 틀의 어느 단계인지, 무효화 조건이 무엇인지'로 개편 예정 |
| 컨센서스·목표가 날짜별 저장 | 있음(2026-11-14~) — engine/snapshot.py, 매일 자동 실행. 화면에는 아직 쓰지 않음 |
| 신호 기록 장부 | 없음 — 틀 판정 모듈이 생기면 같은 폴더에 추가 |

### 다음 작업 순서
1. 기록 쌓기: 컨센서스 EPS·목표가·PER 날짜별 저장은 시작함(위 표). 남은 일은 틀 판정 모듈이 생긴 뒤 앱이 낸 신호와 무효화 조건을 장부로 남기는 것. 과거 데이터는 여러 번 들여다봤으므로 앞으로 쌓이는 기록이 유일하게 깨끗한 검증 자료다.
2. 틀 A 검증: '상대강도 선정 + 눌림 진입 + 추세 청산 + 무효화 손절'을 후보로 고정하고 개별 종목으로 검증(새 검증용 묶음, 큰 갭 하락 통계 포함). 코드 완료(2026-11-14), `run_backtest.py --frame-a` 실제 데이터 결과 대기.
3. 틀 B 재료: 한국 종목 피어·수급, 공시일 기준 재무 데이터(DART·SEC)로 '저평가 + 이익 훼손 없음' 일부 검증, 이익·배수 분해를 매도 판정에 연결.
4. 종합 화면 개편(틀·단계·무효화 조건·매도 조건) 후 Streamlit Cloud 배포.

### 검증 상태 (data/backtest/)
- 국면 게이트: 이후 변동성·낙폭은 구분하지만 수익률은 구분하지 못한다(REPORT.md). 후보 A·B·C는 사전 기준 탈락, 현재 규칙 유지(VARIANTS.md).
- 기존 T 셋업: 단순화 규칙(SETUP_DIAG.md)과 실제 스캐너(SCANNER.md, ETF 19개 2014~2026, 251건 −0.15R vs 무작위 +0.06R) 모두 무작위 진입보다 낫지 않음.
- 재설계 후보(REDESIGN.md): T-A 추세 추종 미통과. T-B 눌림목 반등 통과(처음 보는 해외 국가 ETF에서 +0.01R vs 무작위 −0.06R, z +5.5, 승률 62%, 평균 3.7일 보유). 단 왕복 비용 0.30% 차감 후 거래당 약 +0.10%로 비용에 민감하고 손절이 없는 규칙이다. 단독 매매 신호가 아니라 틀 A의 진입 조건으로 쓴다.
- 틀 A(FRAME_A.md): 실제 데이터 결과 대기. 판정 = HOLDOUT(새 개별 종목 83개) 거래당 순수익률 > 0 이고 보유 기간 맞춘 무작위 대비 z ≥ 2(월 묶음), 개발 묶음(대표 종목) 앞·뒤 기간 모두 > 0.
- 실적·가치·거시 트랙과 틀 B: 과거 시점 데이터가 없어 미검증. 화면·문구에서 검증된 것처럼 표현하지 않는다.

### 운용 규칙 (현재 코드에 적용 중)
- 규칙 우선순위: Thesis 붕괴 > 킬 조건 > 안전마진 방어 > 매도 목표가 > 매수 기준선 > 기술적 타이밍. 방어가 공격보다 항상 우선.
- 게이트 G0~G3. 최종 게이트 = 지수 레짐·VIX·거시 국면 중 가장 나쁜 값. 일부 데이터가 없으면 남은 것으로 판정.
- 손절·비중 축소 같은 방어 행동은 어떤 게이트에서도 막지 않는다.
- 장기 보유 예외(QLD, SCHD)는 게이트·매매 신호 대상이 아니다.
- G3에서 BNF 예외는 승인 비중 20%(1차분의 1/3 이하) — 위기 국면 규칙(1차분의 1/3 이하)과 충돌하지 않음.
- 게이트는 "실행 여부와 크기"만 정한다. 매수 기준선·매도 목표가 숫자는 바꾸지 않는다.

## 5. 화면 표시 규칙
- 색 의미 고정: 초록=공격 가능 · 노랑=주의 · 주황=방어 · 빨강=위기. 위기를 초록·검정으로 표시하지 않는다.
- 제목에는 "무엇을 알려주는지"만 쓰고, 방법론(CMF, 13F, σ)은 작은 설명 줄이나 툴팁으로.
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
- 회귀 테스트: `python tests/run_all.py` (Windows: `.venv\Scripts\python tests\run_all.py`, 약 1~2분, `--fast`는 화면 검사 제외). engine/·딥다이브를 고친 뒤 반드시 실행하고, 실패하면 커밋하지 않는다. 테스트는 tests/_offline.py로 외부 연결을 막고 가짜 데이터만 쓴다.
- 버그를 고치면 그 버그를 재현하는 검사를 tests/에 추가하고 run_all.py 목록에 등록한다. 기대값을 바꿔야 하면 왜 바뀌는지 커밋 메시지에 적는다.
