# 투자터미널 — AI 작업 규칙과 인수인계 (Claude · Gemini 공용)

이 파일이 이 프로젝트 규칙의 정본이다. 어떤 AI든 코드를 고치기 전에 이 파일을 먼저 읽는다.
작업을 끝내면 맨 아래 "변경 기록"에 한 줄을 추가한다.

## 1. 프로젝트 한눈에
- Streamlit 앱. 실행: `streamlit run app.py`
- `app.py` — 홈 화면(결론 → 시장 상태 → 섹터 흐름 → 자금 흐름 → 실행 레이더 → 데이터 점검)
- `engine.py` — 계산 엔진(약 6,400줄). 홈 화면 계산, 4분면(RRG), 국면 게이트, 종목 분석
- `utils.py` — 공통 CSS, 사이드바 메뉴(`init_session`)
- `pages/1_🔎_종목_딥다이브.py` — 종목 딥다이브
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