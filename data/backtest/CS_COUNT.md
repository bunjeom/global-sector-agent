# H-CS1 횡단면 모멘텀 — 실행 전 (수익 평균은 보지 않음)

- 생성: 2026-11-20 18:09 · 설정 cs1-1 · 정의 data/backtest/CS1_PREREG.md. 이 숫자를 보고 규칙을 바꾸지 않는다.

- Ken French 자료 받은 날: 2026-11-20

## 비용에 쓸 회전율 (545종목 — 이미 쓴 자료)

- 상위 20% 같은 비중의 월평균 한쪽 회전율: 미국 26.1% · 한국 25.7% → ① 비용 = 미국 회전율 × 0.30% = 월 0.078%

## 가려낼 수 있는 최소 차이 (검정력 80%, t ≥ 2, Newey-West 시차 3)

| 질문 | 달 수 | 표준오차 | 최소 차이 |
|---|---|---|---|
| ① 미국 대형·모멘텀 상위 30% − 시장, 전체 1927-01~2026-09 | 1,197 | 0.057% | 0.16%p/월 |
| ① 같은 것, 2000년 이후 | 321 | 0.104% | 0.30%p/월 |
| ② 151종목 크기 무작위 부분집합(545종목에서 20번) | 309 | 0.175% | 0.50%p/월 |

## 새 151종목 (대상 종목 수만)

- 받은 종목 141개 · 대상 20개 이상인 달: 미국 311개월(처음 2001-01, 달마다 중앙 91종목) · 한국 311개월(처음 2001-01, 중앙 48종목)
- ① 대형 전체 대비(참고) 계산 가능: 예

## 한국 종목 이름 확인 (코드가 생각한 회사인지 보고 `--cs --names-ok`)

- 003670.KS  Posco Future M Co., Ltd.
- 000990.KS  DB HiTek CO., LTD.
- 001680.KS  Daesang Corporation
- 004490.KS  Sebang Global Battery Co., Ltd.
- 005180.KS  Binggrae Co., Ltd.
- 005250.KS  GC Corp.
- 005610.KS  Samlip Co., Ltd.
- 006120.KS  SK Discovery Co., Ltd.
- 006650.KS  Korea Petrochemical Ind. Co., Ltd.
- 010620.KS  (이름 못 받음)  (가격 없음)
- 010780.KS  IS DongSeo Co., Ltd.
- 011170.KS  Lotte Chemical Corporation
- 011930.KS  Shinsung E&G Co.,Ltd.
- 014820.KS  Dongwon Systems Corporation
- 016380.KS  KG Dongbu Steel Co.,Ltd.
- 017960.KS  Hankuk Carbon Co., Ltd.
- 019680.KS  Daekyo Co., Ltd.
- 020000.KS  HANDSOME Corporation
- 020150.KS  Lotte Energy Materials Corporation
- 023590.KS  Daou Technology Inc.
- 025540.KS  Korea Electric Terminal Co., Ltd.
- 031430.KS  Shinsegae International Co., Ltd.
- 032640.KS  LG Uplus Corp.
- 039490.KS  Kiwoom Securities Co., Ltd.
- 042670.KS  (이름 못 받음)  (가격 없음)
- 049770.KS  (이름 못 받음)  (가격 없음)
- 051600.KS  KEPCO Plant Service & Engineering Co.,Ltd.
- 052690.KS  KEPCO Engineering & Construction Company, Inc.
- 053690.KS  HanmiGlobal Co., Ltd.
- 064350.KS  Hyundai Rotem Company
- 064960.KS  SNT Motiv Co., Ltd.
- 069260.KS  TKG Huchems Co., Ltd.
- 071320.KS  Korea District Heating Corp.
- 079160.KS  CJ CGV Co., Ltd.
- 081660.KS  Misto Holdings Corp.
- 082740.KS  Hanwha Engine Co., Ltd.
- 093370.KS  Foosung Co., Ltd.
- 095720.KS  Woongjin Thinkbig Co., Ltd.
- 103140.KS  Poongsan Corporation
- 111770.KS  Youngone Corporation
- 114090.KS  Grand Korea Leisure Co., Ltd.
- 120110.KS  Kolon Industries, Inc.
- 128940.KS  Hanmi Pharm. Co., Ltd.
- 161390.KS  Hankook Tire & Technology Co., Ltd.
- 001230.KS  Dongkuk Holdings Co.,Ltd.
- 002240.KS  Kiswire Ltd.
- 003030.KS  SeAH Steel Holdings Corporation
- 007700.KS  F&F Holdings Co., Ltd.
- 008060.KS  Daeduck Co.,Ltd.
- 009970.KS  Youngone Holdings Co., Ltd.
- 012510.KS  (이름 못 받음)  (가격 없음)
- 006040.KS  Dongwon Industries Co., Ltd.
- 004430.KS  Songwon Industrial Co., Ltd.

다음: 숫자와 이름을 확인한 뒤 `python run_backtest.py --cs --names-ok`.
