# 피어 비교 점검 (2026-11-16, 피어 비교 v1.2 (2026-11-16))

check_peers.py 출력. 데이터·정의 오류 확인용이며 결과를 보고 가중치·기준을 바꾸지 않는다. 이상한 피어는 data/peers/peer_map.csv로 고친다.

- 묶음 62개 · 종목 564개 · 커버리지 종목 152개 · 재무 기준일 2026-11-16
- 순위가 매겨진 묶음: 가치 관점 52개 · 추세 관점 54개

## 지표가 채워진 비율 (전체)

| 지표 | 채워짐 |
|---|---|
| 매출 성장률(최근 분기, 전년 동기 대비) | 97% |
| 선행 EPS 성장(FY1→FY2) | 89% |
| 영업이익률(최근 4분기, 없으면 야후 값) | 99% |
| ROE | 92% |
| 30일 추정치 순상향 비율((상향−하향)÷FY1 애널리스트 수) | 96% |
| FY1 추정치 30일 변화 | 90% |
| 선행 PER | 90% |
| PEG(선행 PER ÷ 선행 EPS 성장%) | 77% |
| 자기 PER 밴드 z | 49% |
| 섹터 ETF 대비 63일 상대강도 | 98% |
| 틀 A 추세 상태(상승 2·횡보 1·하락 0) | 97% |

## 묶음별 요약

| 묶음 | 출처 | 종목 수 | 가치 1위 | 가치 꼴찌 | 추세 1위 | 순위 없음 사유 | 구성 종목 |
|---|---|---|---|---|---|---|---|
| 한국 자동차·부품 | 수동 | 5 | 161390.KS | 204320.KS | 161390.KS |  | 005380.KS*, 000270.KS, 012330.KS, 204320.KS, 161390.KS |
| 한국 화학·2차전지 | 수동 | 6 | - | - | 006400.KS | 핵심 요인(밸류에이션) 자료 없음 · 피어 부족(점수 있는 종목 3개 < 5) | 051910.KS*, 006400.KS, 373220.KS, 096770.KS, 011170.KS, 003670.KS |
| 한국 인터넷·게임 | 수동 | 5 | 251270.KS | 035720.KS | 035720.KS |  | 035420.KS*, 035720.KS, 259960.KS, 036570.KS, 251270.KS |
| 한국 반도체 | 수동 | 5 | 000660.KS | 042700.KS | 042700.KS |  | 005930.KS*, 000660.KS*, 042700.KS, 000990.KS, 058470.KQ |
| 대형 플랫폼(클라우드·광고·소프트웨어) | 수동 | 5 | ORCL | META | MSFT |  | AMZN*, MSFT*, GOOGL*, META*, ORCL* |
| Aerospace & Defense | 야후 산업 | 19 | HONA | AXON | HEI-A |  | AXON*, GE*, LMT*, NOC*, RTX*, BA*, GD*, TDG*, HWM*, LHX*, HONA*, SPCX, RKLB, HEI-A, ARXS, WWD, FTAI, CW, TXT |
| Apparel Retail | 야후 산업 | 9 | ANF | BOOT | VSXY |  | TJX*, ROST, BURL, LULU, VSXY, GAP, URBN, ANF, BOOT |
| Auto Manufacturers | 야후 산업 | 6 | - | - | - | 피어 부족(점수 있는 종목 4개 < 5) | TSLA*, GM, F, RIVN, WCHS, CYD |
| Banks - Diversified | 야후 산업 | 10 | NTB | BCSO | FRBT |  | JPM*, BAC*, WFC*, C*, RY, TD, BNY, NTB, FRBT, BCSO |
| Beverages - Non-Alcoholic | 야후 산업 | 10 | COKE | MNST | CELH |  | KO*, PEP*, MNST, KDP, COKE, PRMB, CELH, COCO, FIZZ, SUJA |
| Biotechnology | 야후 산업 | 13 | ALNY | VRTX | REGN |  | REGN*, VRTX*, MRNA*, ARGX*, RVMD*, RPRX, ROIV, INSM, INCY, ALNY, ASND, EXEL, JAZZ |
| Building Materials | 야후 산업 | 10 | USLM | MLM | USLM |  | VMC*, MLM*, CRH, JHX, EXP, KNF, USLM, TTAM, TGLS, MCEM |
| Building Products & Equipment | 야후 산업 | 9 | MAS | MAIR | MAS |  | JCI*, TT, CARR, MAS, CSL, LII, MAIR, WMS, OC |
| Capital Markets | 야후 산업 | 10 | LPLA | CRCL | TW |  | GS*, MS*, SCHW, HOOD, IBKR, TW, LPLA, CRCL, EVR, BMNR |
| Communication Equipment | 야후 산업 | 9 | UI | CIEN | UI |  | CSCO*, MSI, HPE, LITE, UI, CIEN, ZBRA, ASTS, VSAT |
| Confectioners | 야후 산업 | 3 | - | - | - | 피어 부족(점수 있는 종목 3개 < 5) | MDLZ*, HSY, TR |
| Conglomerates | 야후 산업 | 9 | DLX | OTTR | GHC |  | HON*, MMM, VMI, GHC, SEB, OTTR, DLX, CODI, MATW |
| Consumer Electronics | 야후 산업 | 7 | SONO | AAPL | UEIC |  | AAPL*, SONO, TBCH, FXHO, UEIC, AXIL, WTO |
| Copper | 야후 산업 | 4 | - | - | - | 피어 부족(점수 있는 종목 2개 < 5) | FCX*, SCCO, IE, CUAI |
| Credit Services | 야후 산업 | 11 | COF | FCFS | MA |  | V*, MA*, AXP*, COF, PYPL, SYF, AFRM, SOFI, ALLY, FCFS, OMF |
| Diagnostics & Research | 야후 산업 | 11 | ILMN | WAT | TMO |  | NTRA*, ILMN*, TMO*, DHR, WAT, IDXX, A, IQV, MTD, LH, DGX |
| Discount Stores | 야후 산업 | 9 | OLLI | PSMT | OLLI |  | COST*, WMT*, TGT*, DG, DLTR, BJ, TBBB, OLLI, PSMT |
| Drug Manufacturers - General | 야후 산업 | 10 | PFE | JNJ | BIIB |  | ABBV*, AMGN*, GILD*, JNJ*, LLY*, BIIB*, MRK*, PFE*, BMY, OGN |
| Electrical Equipment & Parts | 야후 산업 | 10 | AEIS | AYI | BE |  | NVT*, HUBB*, VRT, BE, AYI, AEIS, FPS, POWL, ENS, HAYW |
| Electronic Gaming & Multimedia | 야후 산업 | 5 | - | - | - | 피어 부족(점수 있는 종목 3개 < 5) | TTWO*, EA, RBLX, FIRY, GDEV |
| Engineering & Construction | 야후 산업 | 9 | EME | J | APG |  | PWR*, FER, FIX, EME, J, APG, MTZ, BLD, TTEK |
| Entertainment | 야후 산업 | 10 | NFLX | ROKU | WMG |  | NFLX*, DIS*, WBD, LYV, FOXA, SKYD, FWONA, ROKU, WMG, NWSA |
| Farm & Heavy Construction Machinery | 야후 산업 | 10 | BLBD | AGCO | CNH |  | CAT*, DE*, PCAR, CNH, OSK, AGCO, FSS, TEX, BLBD, ALG |
| Financial Data & Stock Exchanges | 야후 산업 | 9 | TRU | MSCI | CBOE |  | COIN*, SPGI, CME, ICE, MCO, NDAQ, MSCI, CBOE, TRU |
| Gold | 야후 산업 | 9 | CMCL | CDE | RGLD |  | NEM*, AU, RGLD, CDE, AUGO, HYMC, DC, CMCL, CTGO |
| Healthcare Plans | 야후 산업 | 9 | CI | HUM | HUM |  | UNH*, CVS, ELV, CI, HUM, CNC, MOH, OSCR, PGNY |
| Home Improvement Retail | 야후 산업 | 5 | - | - | - | 피어 부족(점수 있는 종목 4개 < 5) | HD*, LOW*, FND, HVT, LIVE |
| Household & Personal Products | 야후 산업 | 10 | KVUE | ELF | EL |  | PG*, CL*, KMB, EL, KVUE, CHD, CLX, ELF, IPAR, NWL |
| Insurance - Diversified | 야후 산업 | 7 | PLGO | BRK-B | PLGO |  | BRK-B*, AIG, HIG, ACGL, PLGO, XZO, IGIC |
| Internet Content & Information | 야후 산업 | 9 | RDDT | SPOT | SPOT |  | GOOG*, SPOT, NBIS, RDDT, PINS, MTCH, SNAP, ZG, PPLI |
| Lodging | 야후 산업 | 7 | WH | MAR | INTG |  | MAR*, HLT*, H, WH, CHH, CVEO, INTG |
| Medical Devices | 야후 산업 | 9 | GEHC | ABT | EW |  | ABT*, MDT, SYK, BSX, EW, GEHC, DXCM, STE, ZBH |
| Oil & Gas E&P | 야후 산업 | 10 | OXY | EQT | PR |  | EOG*, COP*, OXY, DVN, FANG, EQT, EXE, TPL, PR, OVV |
| Oil & Gas Equipment & Services | 야후 산업 | 9 | WFRD | SLB | HAL |  | SLB*, BKR, HAL, FTI, NOV, WFRD, WHD, AROC, KGS |
| Oil & Gas Integrated | 야후 산업 | 4 | - | - | - | 피어 부족(점수 있는 종목 4개 < 5) | CVX*, XOM*, NFG, DEC |
| Oil & Gas Midstream | 야후 산업 | 10 | VG | LNG | EPD |  | WMB*, TRGP*, EPD, KMI, ET, LNG, MPLX, OKE, VG, CQP |
| Oil & Gas Refining & Marketing | 야후 산업 | 11 | DK | WKC | CVI |  | MPC*, PSX*, VLO*, DINO, SUN, PBF, CVI, DK, PARR, DKL, WKC |
| Railroads | 야후 산업 | 7 | TRN | WAB | CSX |  | UNP*, CSX, NSC, WAB, TRN, GBX, FSTR |
| REIT - Healthcare Facilities | 야후 산업 | 10 | NHI | DOC | AHR |  | WELL*, VTR*, OHI, DOC, AHR, CTRE, HR, SBRA, NHI, LTC |
| REIT - Industrial | 야후 산업 | 10 | FR | PSA | CUBE |  | PLD*, PSA*, EXR, EGP, CUBE, LINE, REXR, TRNO, FR, STAG |
| REIT - Residential | 야후 산업 | 9 | ELS | AMH | VMRK |  | VMRK*, AVB, EQR, ESS, INVH, SUI, MAA, ELS, AMH |
| REIT - Retail | 야후 산업 | 10 | NNN | REG | SPG |  | SPG*, O*, KIM, REG, FRT, BRX, ADC, NNN, MAC, EPRT |
| REIT - Specialty | 야후 산업 | 11 | GLPI | WY | OUT |  | AMT*, EQIX*, DLR*, CCI, IRM, SBAC, LAMR, WY, GLPI, RYN, OUT |
| Restaurants | 야후 산업 | 10 | EAT | CAVA | DRI |  | MCD*, SBUX*, CMG, YUM, DRI, YUMC, TXRH, DPZ, EAT, CAVA |
| Semiconductor Equipment & Materials | 야후 산업 | 11 | ASML | AMKR | ENTG |  | ASML*, KLAC*, AMAT*, LRCX, TER, Q, ENTG, NVMI, ONTO, AMKR, FORM |
| Semiconductors | 야후 산업 | 17 | MU | INTC | AMD |  | AMD*, AVGO*, MU*, NVDA*, TSM*, TXN*, ADI*, SKHY*, INTC*, QCOM, MRVL, MPWR, NXPI, ALAB, MCHP, ON, CBRS |
| Software - Application | 야후 산업 | 15 | ADBE | SNOW | ROP |  | CRM*, MSTR*, NOW*, ADBE*, INTU*, CDNS*, UBER*, SHOP, ADP, SNOW, DDOG, ADSK, ROP, WDAY, PAYX |
| Software - Infrastructure | 야후 산업 | 11 | XYZ | CRWD | NET |  | CRWD*, PANW*, PLTR*, NET, FTNT, SNPS, XYZ, NTAP, TWLO, OKTA, ZS |
| Specialty Chemicals | 야후 산업 | 12 | PPG | WLK | ECL |  | LIN*, SHW*, ECL*, APD*, PPG, LYB, DD, IFF, RPM, ALB, SOLS, WLK |
| Specialty Industrial Machinery | 야후 산업 | 10 | CMI | PH | IR |  | ETN*, GEV*, PH, EMR, ITW, CMI, AME, ROK, IR, OTIS |
| Steel | 야후 산업 | 9 | NUE | MTUS | RS |  | NUE*, STLD*, RS, CLF, WS, NWPX, MTUS, FRD, MSB |
| Telecom Services | 야후 산업 | 12 | T | GSAT | VZ |  | CMCSA*, VZ*, TMUS*, T*, ECHO, CHTR, TIGO, GSAT, LBRDA, IRDM, LBTYA, TDS |
| Tobacco | 야후 산업 | 6 | - | - | - | 피어 부족(점수 있는 종목 3개 < 5) | PM*, MO*, AIIR, TPB, UVV, RYM |
| Travel Services | 야후 산업 | 9 | EXPE | MMYT | ABNB |  | BKNG*, ABNB, RCL, VIK, CCL, EXPE, NCLH, GBTG, MMYT |
| Utilities - Diversified | 야후 산업 | 4 | - | - | - | 생존성 하드 킬 — 순위 제외 | SRE*, AES, AVA, UTL |
| Utilities - Independent Power Producers | 야후 산업 | 8 | - | - | TLN | 피어 부족(점수 있는 종목 4개 < 5) | CEG*, VST*, NRG, TLN, TLNE, OKLO, HNRG, KEN |
| Utilities - Regulated Electric | 야후 산업 | 16 | XEL | ED | XEL |  | DUK*, NEE*, SO*, AEP*, D*, ETR*, XEL*, EXC*, ED, WEC, PEG, AEE, PCG, DTE, CNP, FE |

(*는 커버리지 종목, 나머지는 야후 산업 상위 기업에서 온 자동 피어)
