# 투자터미널 — AI 작업 규칙과 인수인계 (Claude · Gemini 공용)

이 파일이 이 프로젝트 규칙의 정본이다. 어떤 AI든 코드를 고치기 전에 이 파일을 먼저 읽는다.
작업을 끝내면 맨 아래 "변경 기록"에 한 줄을 추가한다.

## 1. 프로젝트 한눈에
- Streamlit 앱. 실행: `streamlit run app.py`
- `app.py` — 홈 화면(결론 → 시장 상태 → 섹터 흐름 → 자금 흐름 → 실행 레이더 → 데이터 점검)
- `engine.py` — 계산 엔진(약 7,300줄). 홈 화면 계산, 4분면(RRG), 국면 게이트, 종목 분석
- `utils.py` — 공통 CSS, 사이드바 메뉴(`init_session`)
- `pages/1_🔎_종목_딥다이브.py` — 종목 딥다이브
- `check_earnings_data.py` — 실적 트랙 실제 데이터 점검(PC에서 `python check_earnings_data.py AAPL 005930.KS`)
- `tests/` — 회귀 테스트(가짜 데이터·오프라인). `python tests/run_all.py`로 한 번에 실행
- `.streamlit/config.toml` — Streamlit 기본 페이지 메뉴 숨김(사이드바 메뉴는 utils.py의 st.page_link)

## 2. 데이터 무결성 규칙 (가장 중요)
1. **화면에 보이는 숫자·판정은 실제 데이터에서만 만든다.** 난수(np.random), 사인파, 손으로 적은 숫자를 실제 데이터처럼 화면에 쓰지 않는다.
2. 실제 출처가 아직 없는 항목은 `is_sample: True`, `source: "SAMPLE_DATA ..."`, `as_of: None`으로 표시한다. 홈 화면은 이런 항목을 기본으로 숨긴다("예시 데이터 보기" 토글).
3. 데이터를 못 받으면 "데이터 없음"을 보여준다. 예시 데이터로 대신 채우지 않는다.
4. `as_of`는 **데이터의 날짜**(예: 금요일 종가면 금요일)다. 계산한 날짜나 오늘 날짜를 넣지 않는다.
5. 같은 지표는 한 함수에서만 계산하고 모든 화면이 그 결과를 쓴다(단일 출처).
6. 임계값은 설정 딕셔너리 한 곳에만 둔다: `GATE_CONFIG`(국면 게이트), `RRG_CONFIG`(4분면), `DATA_FRESHNESS`(신선도).

## 3. engine.py 주의사항
- 게이트 관련 함수·설정(GATE_CONFIG 등)은 한 번씩만 정의되어 있다. 새로 같은 이름을 또 정의하지 않는다.
- `_build_default_rrg_sample_dataset()`은 합성 데이터 생성 함수다. **테스트 전용**이며 화면에서 호출하지 않는다.
- 실제 4분면은 `build_real_rrg_bundle(etf_map, baskets)`(파일 맨 끝)로 만든다.
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

## 4. 운용 규칙 (게이트·신호)
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

## 6. AI 두 개를 번갈아 쓸 때
- **항상 지금 파일에서 시작한다.** 예전 대화에 남아 있던 옛 버전 코드를 통째로 붙여 넣으면 다른 AI가 고친 내용이 사라진다.
- 파일 전체를 바꾸는 대신 바뀐 함수·구간만 교체한다.
- 작업 하나가 끝나면 git 커밋: `git add -A` → `git commit -m "무엇을 했는지"`.
- 끝나면 아래 변경 기록에 한 줄 추가.

## 7. 확인 방법
- 문법: `python -c "import ast; ast.parse(open('app.py', encoding='utf-8').read())"`
- 실행: `streamlit run app.py` 후 화면 맨 아래 "데이터 점검" 펼쳐서 확인
- 회귀 테스트: `python tests/run_all.py` (Windows: `.venv\Scripts\python tests\run_all.py`, 약 1~2분, `--fast`는 화면 검사 제외). engine.py·딥다이브를 고친 뒤 반드시 실행하고, 실패하면 커밋하지 않는다. 테스트는 tests/_offline.py로 외부 연결을 막고 가짜 데이터만 쓴다.
- 버그를 고치면 그 버그를 재현하는 검사를 tests/에 추가하고 run_all.py 목록에 등록한다. 기대값을 바꿔야 하면 왜 바뀌는지 변경 기록에 적는다.

## 변경 기록
- 2026-11-08 (Claude): 4분면을 합성(난수) 데이터에서 실제 yfinance 3년 종가로 교체(`build_real_rrg_bundle`). 국면 게이트를 하드코딩 값(VIX 36.2 등)에서 홈 헤더 실제 데이터로 교체, 일부 결측 시 G2 고정 버그 수정(삼성전자 G2 표시 문제). 레이더 신호를 예시 데이터로 표시하고 기본 접힘. 장기 보유 예외(QLD·SCHD) 게이트 제외. 실적 리비전이 예시 데이터면 '실적 데이터 없음'으로 표시. 홈 제목을 "오늘의 결론"으로 교체, 섹션 이름 정리, 동결 신호 접기, 기본 페이지 메뉴 숨김, .gitignore 추가(.env 보호). 백업: `_backup_20261108/`.

- 2026-11-09 (Claude): run_home_dashboard_engine의 고정 좌표표(fallback_rrg_coords) 제거. 실제 가격이 75일 미만인 바스켓은 예시 좌표로 채우지 않고 제외하며 missing_baskets에 기록, 데이터 점검에 DATA_MISSING으로 표시. 
- 2026-11-09 (Claude): engine.py의 위쪽 중복 정의 4개(GATE_CONFIG, calculate_single_gate, evaluate_market_gate, apply_gate_action_matrix) 삭제. 이전/이후 결과 992건 동일 확인. 백업: engine.py.bak_before_task6.

- 2026-11-09 (Claude): 국면 게이트 히스테리시스 적용 — build_gate_history_state가 최근 10거래일(GATE_CONFIG["HISTORY_LOOKBACK_DAYS"])을 과거 종가로 재계산해 이전 단계를 만들고(파일 저장 없음), 완화는 2거래일 연속 확인 후 한 단계씩. 레짐·VIX·거시 국면 계산을 build_regime_vix_header로 분리(단일 출처). 백업: *.bak_before_task4.

- 2026-11-09 (Claude): 오늘의 결론 해제 조건을 더 나쁜 게이트 기준으로 표시. use_container_width → width="stretch"(app.py 2곳, 딥다이브 10곳). 데이터 점검을 '확인 필요/예시 데이터'로 분리하고 예시 데이터의 기준일 없음 중복 경고 제거. AGENTS.md 3번 23번째 줄 갱신. 백업: *.bak_before_task7.
- 2026-11-09 (Gemini): `pages/1_🔎_종목_딥다이브.py`에서 종목 검색 시 `resolve_ticker(raw_query)` 호출이 누락되어 한글 종목명·6자리 코드가 변환되지 않던 오류 수정.
- 2026-11-09 (Claude): utils.load_latest_report가 GitHub raw의 최신 리포트를 읽어 PC 파일과 generated_at을 비교해 최신 것을 사용(30분 캐시, 실패 시 PC 파일). PC 파일은 덮어쓰지 않음. 리포트 때문에 git pull할 필요 없음. 백업: utils.py.bak_before_report.
- 2026-11-09 (Gemini): `pages/1_🔎_종목_딥다이브.py`의 `resolve_ticker`에 KRX 상장법인 목록 캐싱 및 네이버 금융 자동완성 보조 검색을 연동하여 한국 코스피·코스닥 전 종목 및 ETF의 한글명·6자리 코드(`.KS`/`.KQ` 자동 판별) 검색 지원.
- 2026-11-09 (Claude): 실행 레이더를 실제 T 신호로 교체 — 섹터 흐름 '지금 강한 곳'·'다음 후보' 대표 종목(RADAR_CONFIG, 최대 18개)을 analyze_technical_track로 계산(build_radar_signals). 미국 종목은 USD/KRW 환율 상수로 원화 환산 후 달러로 표시, 장중 미완성 봉 제외, V 트랙은 딥다이브 전용. 붕괴·하락 상태는 방어 신호 우선. 예시 레이더 신호 사용 중단. 백업: *.bak_before_task2.
- 2026-11-09 (Claude): T 엔진을 통화 인식형으로 수정 — account_cfg["fx_to_krw"]로 해외 종목 유동성·계좌 금액을 원화 환산, 가격 소수 유지, 최소 1R 0.01, 문구는 engine.fmt_price($/원). 딥다이브 가격 표시 28곳·실적/펀더멘털/종합 문구 32곳 통화 표기 수정, 미국 종목은 USD/KRW 환율 전달. 레이더 환산 우회 제거, 옛 예시 기회 레이더 삭제. 한국 종목 결과 불변 확인. 백업: *.bak_before_fx.
- 2026-11-09 (Claude): 실적 발표 블랙아웃을 실제 발표일로 적용 — engine.fetch_next_earnings_date(yfinance calendar, 범위면 가장 이른 날)·trading_days_until·is_earnings_blackout(TECH_PARAMS["EARNINGS_BLACKOUT_DAYS"]=3). 레이더는 D-3~당일 진입·관찰 신호를 보류 목록으로(방어 신호 제외), 딥다이브는 고정 45일 대신 실제 일정(모르면 '일정 미확인'). 백업: *.bak_before_earn.
- 2026-11-09 (Claude): 레이더 종목 선정을 데이터 기반으로 교체 — 강한 섹터 ETF 상위 구성 종목(engine.fetch_etf_top_holdings, 미국·한국 상장만, 부족 시 대표 종목 보충) 중 종가>50일선·3개월 SPY 대비 초과수익>0 통과 종목을 상대강도 순 섹터당 3개(RADAR_CONFIG). 카드에 선정 이유, '종목 선정 과정' 표 추가. 백업: *.bak_before_select.
- 2026-11-10 (Claude): 실적 트랙(analyze_earnings_track)의 삼성전자 고정 재무값(rev_base 등)을 제거하고 실제 데이터로 교체 — 미국 yfinance(분기 실적·컨센서스·리비전), 한국 DART(확정 실적)+yfinance 추정치(있을 때만), 입력층은 engine.fetch_real_earnings_inputs 한 곳. 못 받은 값은 '데이터 없음'(None)이며 해당 축은 판정에서 빼고 점수는 판정 가능한 축 가중치로 정규화(verdict.coverage), 근거 부족 시 국면 `NA`(판정 불가). 미래 EPS가 없으면 V 트랙은 최근 확정 EPS(TTM) '참고용'으로 계산, EPS가 전혀 없으면 딥다이브는 안내 후 중단. V 트랙 가격선을 통화 인식(미국은 센트 유지)으로 수정, DART 키는 .env에서 직접 읽음. 백업: engine.py.bak_before_realdata, deepdive.py.bak_before_realdata, AGENTS.md.bak.
- 2026-11-10 (Claude): V 트랙(analyze_fundamental_track)의 남은 고정값(default_f: 자기 PER 15.2/2.8/9.5, 순현금 120, FCF 18.5, 목표가=현재가×1.28, 12개월 전 주가=현재가×1.12 등)을 제거하고 실제 데이터로 교체 — 입력층 engine.fetch_real_fundamental_inputs 한 곳(미국 yfinance: info·가격이력·연간/분기 재무제표·산업 상위 기업 피어 PER, 한국 DART 사업보고서 6개년 EPS·자본·현금·차입·OCF + yfinance 가격·베타). 자기 PER 밴드는 월말 주가÷공시 지연 반영 연간 EPS(후행 기준, 실제 관측 기간만), 피어는 선행 PER 중앙값·상위25%·σ(유효 5개 이상일 때), 컨센서스 목표가는 yfinance targetMeanPrice. 못 받은 값은 None('데이터 없음')이며 해당 앵커·z-score·요인은 판정에서 빠지고 있는 근거만으로 계산한다. 근거가 전혀 없으면 가격 자격 `NA`·가격선 없음(종합 `V_NA`, 비중 0). 12개월 주가 분해는 후행 EPS 기준 2층(이익+배수)으로 줄였고 시장·산업층은 '데이터 없음'. 2Y 성장률은 미래 추정치 1Y·2Y가 모두 있을 때만 계산(2% 하한 제거). 피어 비교 요인 1·2번은 선행/후행 EPS 성장·ROE(기업·피어 동일 야후 기준)로 교체. 딥다이브 탭3·종합 카드는 None 안전 처리 + '입력 데이터 출처·못 받은 항목' 패널 추가. 점검 스크립트 check_fundamental_data.py. 백업: engine.py.bak_before_vdata, deepdive.py.bak_before_vdata, AGENTS.md.bak_before_vdata.
- 2026-11-10 (Claude): 딥다이브 HTML 깨짐 수정 + 종합 탭 T 실행 카드의 가짜 기본값 제거 — (1) 페이지 HTML 블록 32곳을 html_md()로 렌더(공백만 남은 줄·들여쓰기 제거; 빈 배지 때문에 마크다운이 HTML을 끊고 코드블록으로 보이던 문제). 새 HTML 카드도 st.markdown 대신 html_md 사용. (2) synthesize_4tracks의 t_action_card가 셋업이 없을 때 현재가로 만든 진입가·손절가(×0.95)·목표가(×1.10)·0주·시간손절 0을 넣던 것을 제거 — 티켓이 실제로 계산된 경우(has_ticket)에만 가격·수량을 싣고, 없으면 None + no_ticket_reason·next_condition. USD 가격 int 절사(센트 손실)도 수정. 진입 차단(위험계수 ×0) 상태의 0주는 사유를 함께 표시. 백업: engine.py.bak_before_tcard, deepdive.py.bak_before_tcard, AGENTS.md.bak_before_tcard.
- 2026-11-10 (Claude): 종합 판정 라벨 분리 — 예전엔 하드 킬이거나 MDD Ratio<0.15이면 모두 `VALUE_TRAP` → ARCH_8 "하드 킬 … 청산/동결"이 되어, 실적이 멀쩡하고 비싸기만 한 종목(예: 애플)에도 청산 문구가 붙었다. 이제 가격 자격 코드는 `HARD_KILL`(Step 1 킬), `VALUE_TRAP`(CYC_T2 사이클 고점 또는 G4 피크아웃의 저PER 착시만), `POOR_RISK_REWARD`(실적 훼손 없이 위험 대비 기대수익 부족 = 고평가, V는 가격 대기), `OVERVALUED`/`CONDITIONAL`/`QUALIFIED`/`NA`. ARCH_8은 하드 킬(청산/동결 문구) 또는 진짜 가치 함정(신규 진입 금지, 청산 문구 없음)에만. 고평가 관망은 ARCH_7 "가치 기준 고평가 관망". MDD Ratio를 못 받으면(None) 0.0으로 보고 탈락시키지 않음. 현재가가 매도목표가를 넘으면 V 배지·3줄 결론에 초과 폭과 '신규 매수 없음, 보유분 분할 차익실현'을 명시. 화면에는 코드 대신 engine.PRICE_QUAL_LABELS 한글 이름(v_action_card.price_qual_label). 백업: *.bak_before_label.
- 2026-11-10 (Claude): V 트랙 G3(성숙기) 목표 배수와 할인율 입력 수정 — (1) G3 base 배수를 min(자기 평균 PER, 피어 중앙값 PER)으로 바꾸고, 이론 성숙 PER(1단계 고든·영구 3% 성장 → 우량 성숙주도 13~16배)은 상한이 아닌 참고값으로만 표시, 관측 배수가 둘 다 없을 때만 이론 PER×1.1 대체. (애플이 이론 PER에 눌려 1차선 $129·E[R] −60%로 나오던 원인.) (2) 10년 국채 금리 고정값(미국 4.20%·한국 3.15%/3.5%)을 engine.fetch_risk_free_10y로 교체 — FRED 공개 CSV(미국 DGS10, 한국 IRLTLT01KRM156N 월별), 미국은 실패 시 yfinance ^TNX, 연 −1~25% 밖 값은 버림, 성공만 6시간 캐시. 못 받으면 None → 요구수익률·k_e·매수선 '데이터 없음'(가격 자격 NA, 사유 표시, 못 받은 항목 rf_10y). (3) 시장 위험프리미엄 고정 5.5%를 FUND_MODEL_PARAMS의 모델 상수로 옮기고 출처 명시(Damodaran 국가별 ERP 2026년 1월: 미국 4.46%·한국 4.87%), 딥다이브 탭3에 k_e 구성(금리 출처·날짜, ERP 출처)과 Track 4에 금리 출처 표시. 거시 트랙 보류 경로의 인플레 비중 0.5 가짜값 제거. 백업: *.bak_before_g3rf.
- 2026-11-10 (Claude): 거시 트랙(Track 4) 실데이터화 — 종목명 시드 난수로 만들던 거시 변화량 10종(실질금리·기대인플레·국고채10Y·원달러·실업수당·IG/HY 스프레드·순유동성·WTI·장단기차)과, 주가·지수가 없을 때 만들던 난수 주가·지수를 제거. engine.fetch_macro_levels(FRED API/CSV, ECOS)로 수준 시계열을 받아 W-FRI 정렬 후 Δbp·Δln 계산(_weekly_changes), 순유동성=WALCL−WTREGEN−RRPONTSYD×1000. 못 받은 변수는 회귀 제외·'받지 못한 거시 변수' 패널, 국면 판정은 T10YIE·DFII10이 둘 다 있을 때만(없으면 '데이터 없음', 국면별 베타 생략), 13주 고저는 실제 수준(고정 기준값 제거), 경계 규칙의 bp·% 혼용("685.8%") 수정, OOS ΔR² 기본값 0.015·z_oos 0.0 제거(None). 종합의 13주 수익률 분해 가짜 기본값(−5%/−3%) 제거. 무위험수익률도 같은 층 사용(한국은 ECOS 일별 우선, FRED_API_KEY 사용). Track 4에 출처·기준일(실제 마지막 관측일, 주간 금요일 라벨 아님)·보류 사유 표시. 야후 주가의 시간대(뉴욕·서울)는 거시 데이터와 비교 전 제거(_tz_naive_index). 점검 스크립트 check_macro_data.py. 백업: *.bak_before_macro.
- 2026-11-10 (Claude): 딥다이브 표시 정정 — (1) 매도 앵커2는 G3·G4에서 μ+1σ가 아니라 자기 평균 μ로 상한임을 이름에 명시, Base 배수와 같은 값이면 표시. 앵커3(PEG)이 Base 배수보다 낮아 Base로 대체되면 '독립 아님'으로 표시하고 매도목표가 산식 문구에 사유 기재(계산은 그대로). (2) 국면별 β를 % 단위(1σ당 주가 반응)로 표시, '국면 의존' 경고는 두 국면 β가 모두 1σ당 ±0.10%p(MACRO_PARAMS REGIME_BETA_NOISE) 이상일 때만. (3) 차트: 제목을 plotly 밖(chart_heading)으로, 범례는 아래로, 6-1 실제 수익률은 x축 라벨에·막대 글자는 안 들어가면 숨김, 민감도 막대 글자 잘림 방지(x 여백·cliponaxis), F4 워터폴 y 여백, 13주 표 열 이름·너비. (4) Track 1 로그 축에서 지지·저항·목표 라벨을 가격 그대로 넣어 10^가격 위치로 가던 버그 수정(로그면 log10 변환, 가까운 라벨은 한 줄로 합침, 통화 표기). 피어 점검 스크립트 check_peer_data.py. 백업: *.bak_before_disp.
- 2026-11-10 (Claude): 피어·내재 성장률 정정 — (1) 피어 ROE·성장률 중앙값을 PER 유효 피어 집합에서만, ROE −100%~300%·성장률 −90%~300% 밖 이상치 제외로 계산하고, 피어 PER 통계가 없으면 함께 '데이터 없음'(애플: 산업 피어 소형주 5개·유효 PER 3개, ROE −3,092 포함 중앙값 −0.5%로 ROE 격차 +142.8%p가 나오던 왜곡 제거). 섹터 대체는 하지 않음(기술 섹터는 사이클 정점 반도체가 섞여 중앙값 17.8배로 왜곡). (2) F5 내재 5년 성장률의 종착 배수를 이론 성숙 PER에서 채택 Base 배수로 변경(F5 블록을 F6 뒤로 이동, terminal_per_basis 표시; 애플 34.4% → 약 11.5%). 점검: check_peer_data.py, 테스트 test_peer.py. 백업: *.bak_before_peer.
- 2026-11-10 (Claude): 표기·안내 정리 — FCF 표기를 원 단위 숫자 대신 조원·$B로, '데이터 없음 (데이터 없음)' 중복 제거(engine.with_paren, 페이지 xmul·이익의 질 설명 생략, html_md 안전망), 트랙 신호등 '자기' → '자기 밴드'. 딥다이브 상단에 투자 권유 아님·데이터 출처·미검증 고지와 '용어 설명' 펼침 추가. README.md 추가(개요·데이터 원칙·실행·점검 스크립트·한계). 저장소 점검: .env 커밋 이력 없음, 추적 파일에 키 없음, __pycache__ .pyc 2개가 과거 커밋으로 추적 중(사용자가 git rm --cached로 정리). 백업: 직전 버전 1개(이후 작업에서 교체).
- 2026-11-10 (Claude): 엔비디아 실데이터 점검 후 수정 — (1) 자기 PER 밴드를 로그 PER로(기하평균·로그 σ, kσ=μ·e^(kσ)); 엔비디아는 과거 후행 PER이 200~300배까지 튀어 산술 μ 131배·σ 114배 → 매도목표가 $1,138·2/3차선 $13/$12·Bull +671%가 나오던 문제. 밴드 계산에 쓴 월별 후행 PER(band_series)을 반환해 F3 차트 검은 선을 같은 기준으로 그림, 범위가 크면 로그 축(라벨 log10 좌표). (2) G3를 '나머지 전부' 기본값에서 저성장 근거가 있을 때만으로 변경, 고성장인데 G2 조건 미충족이면 NA '성장 단계 판정 보류 — 미충족 사유'(엔비디아: 매출 +106%·이익의 질 경고 3개). (3) 거시 OOS 기업 고유 충격 주간 날짜를 실제 마지막 거래일 이하로, 근거 없던 '실적·가이던스 공시 주간' 이름을 '기업 고유 충격 주간(원인 미확인)'으로. (4) 기술 트랙 V 판정의 매수선 가격 정수 절사(미국 센트) 수정. 테스트 test_regime_band.py. 백업: *.bak_before_nvda.
- 2026-11-10 (Claude): 자기 PER 밴드 신뢰 게이트 추가 — 실제 엔비디아는 로그 PER로도 기하평균 97배·+1σ 204배(2023~24년 후행 PER 150~500배)라 매도목표가 약 $880·Bull +222%였음. 밴드 기간 확정 EPS가 3배 넘게 변했거나 로그 σ > 0.5면 밴드 제외(사유 표시), 시나리오 폭은 피어 PER σ로 대체(engine scen_lsd, input_data.scenario_width_source), G2도 자기 밴드가 없으면 피어 중앙값 대체. 재현 테스트: 엔비디아형 → 목표 22.5배·Bull/Bear 26.2/16.6배·매도 약 $333, 안정형(로그σ 0.12)은 밴드 유지. 주의: 이익 변동이 큰 사이클주(예: 삼성전자 2023년 EPS 급감)도 밴드가 제외될 수 있음. 백업: *.bak_before_bandgate.
- 2026-11-11 (Claude): 딥다이브 NaN 현재가 오류 수정 — 야후가 장 마감 후 종가가 빈 마지막 행을 붙여 보내 현재가가 NaN이 되어 종합 판정에서 `cannot convert float NaN to integer` 발생. 페이지 fetch_ohlcv_and_benchmark에서 _clean_ohlcv로 종가 없는 행 제거(시가·고가·저가 빈칸은 종가로), synthesize_4tracks는 NaN·0 현재가면 명확한 오류 메시지, 종합 출력 current_price는 미국 센트 유지. 화면 테스트에 빈 마지막 행(NANROW) 재현 추가. 백업: *.bak_before_nanfix.
- 2026-11-11 (Claude): 가치 판정 표시 일관성 — 종합(synthesize_4tracks) 가격 자격이 가치 트랙의 '조건부'(자기 밴드 없음 등)를 '매수 자격 충족'으로 올리던 불일치 수정(가치 트랙 판정을 넘지 않음). 실적 국면 판정 보류(NA)면 목표 배수 규칙 문구를 'G3 성숙/현금회수' 대신 '실적 국면 판정 보류 → 중립 규칙'으로 표시. 백업: *.bak_before_consist.
- 2026-11-11 (Claude): 회귀 테스트를 저장소에 추가 — tests/ (실적·V 입력층, G3·금리, 거시 실데이터, T 실행 카드, 종합 라벨, 피어, 로그 PER 밴드·G3 조건, 딥다이브 화면 7가지 데이터 상태) + tests/run_all.py 일괄 실행기 + tests/_offline.py 네트워크 차단(localhost만 허용). 실제 네트워크·API 키 없이 15개 통과 확인, 이전 엔진(라벨 수정 전)으로는 6개 실패해 회귀를 잡는 것도 확인. 실제 네트워크에 의존하던 옛 체인 테스트는 제외. 변경 기록의 중복 줄 정리.
