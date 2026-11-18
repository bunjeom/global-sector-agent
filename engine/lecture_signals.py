"""강의 조건 시험 — 사전 등록 설정과 기초 신호 (2026-11-18, AGENTS.md 4장 '강의 조건 시험').

정의의 근거는 강의 정리본 1~9번과 질문 Q1~Q69 결정(claude.ai Project 문서 '강의점검_결정기록'·'강의시험_사전등록안').
숫자는 LECTURE_CONFIG 한 곳에만 둔다. **결과를 보기 전에 고정했으므로 결과를 보고 바꾸지 않는다.**

앱의 T 트랙 함수(technical_signals.py)는 바꾸지 않는다. 시험용 정의는 여기에 따로 있다(앱 국면 게이트는 이미 시험해 고정했음).
기초 지표는 앱과 같은 함수(compute_base_primitives)를 쓰고, 정리본과 다른 곳만 바로잡는다:
    RSI14   14일 동안 하락이 없으면 100 (앱은 50으로 나옴)
    상태    상승 = 최근 두 확정 저점이 높아짐 + 그 뒤 종가가 한 번이라도 앞선 확정 고점 위(Q2·Q12). 하락은 대칭. 나머지 횡보
    시장 등급  1번 표 + ⚫도 2일 확인, ADX 20~25는 직전 강도 유지, ±DI 미사용(Q2~Q5). 데이터 처음부터 하루씩(시작점에 따라 달라지지 않게)
    RS      종목 ÷ 섹터 ETF(Q6), 주봉은 그날까지의 '진행 중인 주'를 포함
모든 신호는 그날 종가까지의 자료만 쓴다. 스윙은 오른쪽 5봉이 지난 날부터 '확정'으로 쓴다.
"""
import os
from bisect import bisect_left
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .technical_signals import compute_base_primitives

LECTURE_CONFIG: Dict[str, Any] = {
    "VERSION": "lec-1",
    "REGISTERED": "2026-11-18",
    # --- 기초 정의 (정리본 1~8번, 숫자는 정리본 '가정값' 또는 코드 관행값) ---
    "SWING_LR": 5,                 # 좌우 5봉 (Q11)
    "ZONE_PCT_MIN": 0.01,          # 구간 폭 = max(1%, 0.5×ATR14)
    "ZONE_ATR_MULT": 0.5,
    "SLOPE_DAYS": 5,               # 기울기 = 오늘 − 5거래일 전
    "RVOL_UP": 1.5,                # 거래량 실림 (Q15·Q20)
    "RVOL_CLIMAX": 2.0,            # 거래량 폭증·투매 (Q64)
    "ADX_STRONG": 25.0,            # 1번: ≥25 강함, <20 추세 없음, 그 사이 직전 유지
    "ADX_NONE": 20.0,
    "GRADE_DOWN_DAYS": 2,          # 1번: 하향 2일(⚫ 포함), 상향 3일
    "GRADE_UP_DAYS": 3,
    "GRADE_MIN_BARS": 200,         # 200일 EMA가 생긴 뒤부터 등급
    "RS_MA": 50, "RS_MA_MIN": 20, "RS_ROC": 20,      # 2번 4구간(일봉)
    "RS_W_MA": 10, "RS_W_MA_MIN": 5, "RS_W_ROC": 4, "RS_W_MIN_WEEKS": 12,   # 주봉(10주·4주)
    "RS_LOW_DAYS": 20,             # RS선 저점 갱신 (Q9)
    "RSI_LOW": 30.0, "RSI_HIGH": 70.0,               # Q18
    "BB_SQZ_PCTL": 20.0,           # Q17
    # --- 측정 (사전등록안 8장) ---
    "DATA_START": "1999-01-01",
    "EXPLORE_START": "2001-01-01",   # 탐색: 이미 쓴 종목(EXPLORE_CONFIG UNIVERSE), 2001~2013
    "EXPLORE_END": "2013-12-31",
    "VERIFY_START": "2014-01-01",    # 검증: 같은 종목 2014~ (다른 질문에서 이미 본 기간이라 절반쯤만 독립)
    "HORIZON": 10,                   # 판정 기간 (N1). 5·20은 참고
    "HORIZON_REF": (5, 20),
    "COST": 0.003,                   # 매수 가족(L1·L2) 왕복 비용
    "GAP": 10,                       # 같은 종목·같은 사건은 10거래일 안의 새 사건을 세지 않음
    "BASELINE_SAMPLES": 20,
    "DROP_TOL": 0.5,                 # 하락 폭 맞춘 표본: 최근 5거래일 변화(ATR14 단위) ±0.5
    "SEED": 1118,
    "MIN_EVENTS": 300,               # 탐색 후보 조건 (인수인계 E)
    "MIN_TICKERS": 50,
    "MIN_MARKET_EVENTS": 30,         # 미국·한국 같은 방향은 둘 다 이 이상일 때만 본다
    "SMALL_FAMILY_Z": 2.0,           # 질문이 1~2개인 가족(L1·L3)의 탐색 기준. 큰 가족(L2·L4·L5)은 섞기 200회 통과선
    "MONTH_DROP_Z": 2.0,             # 달 쏠림 점검(N3): 기여가 가장 큰 달을 빼도 같은 방향 + z ≥ 2
    "VERIFY_Z": 2.0,                 # 검증: 같은 방향 + z ≥ 2
    "FINAL_Z": 3.0,                  # 최종: 새 종목, z ≥ 3, 후보 전체 최대 5개 (N4)
    "FINAL_MAX": 5,
    "RC_REPS": 200, "RC_Q": 0.95, "PBO_SLICES": 16,    # 큰 가족(다음 단계)
    # --- L1 BNF (9번, Q62~Q69) ---
    "BNF": {
        "BASE_LOOKBACK": 750,        # 기준 이격도: 지난 750거래일 확정 스윙 저점 중
        "BASE_N": 5,                 # 이격도가 가장 낮았던 5곳의 중앙값
        "BASE_MIN": 3,               # 3곳 미만이면 판정 불가
        "PEAK_LOOKBACK": 40,         # 급락 시작일 = 직전 40거래일 최고 종가일 (TP-17)
        "COND2_WINDOW": 5,           # 조건 2: 확인봉 전 5거래일 안에 이격도 ≤ 기준
        "CLIMAX_WINDOW": 5,          # 조건 3: 조건 2 도달일 전후 5거래일 안 음봉 RVOL ≥ 2
        "HOLD_MIN": 1, "HOLD_MAX": 2,  # 조건 4: 최저가 봉 뒤 홀드봉 1~2개
        "MAX_STOP_PCT": 0.12,        # 손절폭 > 12%면 패스 (TP-19)
        "TIME_STOP_DAYS": 5,         # 진입 뒤 5거래일 안에
        "TIME_STOP_R": 0.5,          # 종가 ≥ 진입가 + 0.5R 이 없으면 시간 손절 (= 질문 1의 '반등')
        "TAKE_FRAC": 0.5,            # 25일선 복귀 시 절반
        "CODE_BASE_PCTL": 2.5,       # 참고: 코드식 기준(750일 분포 하위 2.5%)
    },
    # --- L2~L5 (L2는 2단계에서 구현, L4·L5는 다음. 숫자는 사전 등록 때 고정) ---
    "BREAKOUT_CONFIRM_DAYS": 20,     # Q19
    "REACT_DAYS": 10, "REACT_ATR": 1.0,             # Q36
    "TOUCH_AWAY_ATR": 2.0,           # Q48
    "LINE_IDLE_DAYS": 250,           # Q35
    "DWELL_DAYS": 250, "DWELL_TOP": 0.2,            # Q38
    "VP_DAYS": 250, "VP_BINS": 20, "VP_Q": 0.2,     # Q43
    "WICK_BODY": 2.0, "WICK_VOL_DAYS": 60, "WICK_CONFIRM_DAYS": 10,   # Q37·Q40·Q41
    "BREAK_RECOVER_DAYS": 10, "BREAK_RECOVER_STREAK": 2,              # Q26
    "CHANNEL_MIN_WIDTH": 0.04,       # Q51
    "EXT_DAYS": 10,                  # Q50
    "DBL_MAX_GAP": 250,              # Q55
    "TGT_MIN_WIDTH": 0.05, "TGT_MAX_DAYS": 250,     # Q59·Q61
    "SELL_DD_DAYS": 20, "SELL_STATE_DAYS": 60,      # N5
    "RETEST_DAYS": 20,               # Q53
    # --- 자료 ---
    "INDEX": {"US_TECH": "^IXIC", "US": "^GSPC", "KS": "^KS11", "KQ": "^KQ11"},   # Q4
    "SECTOR_ETF_US": {
        "Technology": "XLK", "Communication Services": "XLC", "Financial Services": "XLF", "Healthcare": "XLV",
        "Consumer Cyclical": "XLY", "Consumer Defensive": "XLP", "Industrials": "XLI", "Energy": "XLE",
        "Utilities": "XLU", "Basic Materials": "XLB", "Real Estate": "XLRE",
    },
    # 한국 업종 ETF(코스피200 섹터 시리즈). 받기 단계에서 이름을 출력해 사람이 확인한 뒤에만 시험을 돌린다.
    # 없는 섹터는 None = '데이터 없음'(시장 지수로 대체하지 않음, Q6)
    "SECTOR_ETF_KR": {
        "Technology": "139260.KS", "Financial Services": "139270.KS", "Consumer Cyclical": "139290.KS",
        "Industrials": "227550.KS", "Healthcare": "227540.KS", "Consumer Defensive": "227560.KS",
        "Basic Materials": "139240.KS", "Energy": "139250.KS",
        "Communication Services": None, "Utilities": None, "Real Estate": None,
    },
    "SECTOR_ETF_KR_EXPECT": {
        "139260.KS": "TIGER 200 IT", "139270.KS": "TIGER 200 금융", "139290.KS": "TIGER 200 경기소비재",
        "227550.KS": "TIGER 200 산업재", "227540.KS": "TIGER 200 헬스케어", "227560.KS": "TIGER 200 생활소비재",
        "139240.KS": "TIGER 200 철강소재", "139250.KS": "TIGER 200 에너지화학",
    },
    # 실적 발표일(N7): 미국 SEC 8-K Item 2.02(2004-08-23 시행) 제출일, 한국 DART 영업(잠정)실적 공시일(공정공시 2002-11 시행).
    # 그 전 기간이나 못 받은 종목은 '데이터 없음' — 그 조건은 적용하지 않고 비율을 보고서에 적는다.
    "EARN_COVER_US": "2004-08-23",
    "EARN_COVER_KR": "2002-11-01",
    "CACHE_DIR": os.path.join("data", "backtest", "lecture_cache"),
    # --- 최종 확인용 새 종목 (N4, 2026-11-18 L2 결과 보기 전 고정) ---
    # 규칙: 지금까지 어떤 시험(EXPLORE_CONFIG UNIVERSE·게이트·스캐너 등)에도 쓰지 않은, 오래 상장된 종목.
    # 미국 11개 섹터 98개 + 한국 코스피 53개. 받지 못했거나 가격이 260거래일 미만인 종목은 빼고 채워 넣지 않는다.
    # 한국 코드는 받기 단계에서 이름을 출력해 사람이 확인한다(섹터 ETF와 같은 방식). 검증을 통과한 질문이 생길 때만 연다.
    "FINAL_UNIVERSE": [
        "CIEN", "GEN", "MKSI", "ENTG", "COHR", "LSCC", "CGNX", "MANH", "SSNC", "ARW",              # 기술
        "TMUS", "NYT", "IAC", "TDS", "SIRI", "ZD",                                                 # 통신·미디어
        "KSS", "AAP", "DKS", "HOG", "MAT", "GT", "PVH", "TOL", "LEA", "GNTX",                     # 경기소비재
        "CASY", "UNFI", "FLO", "LANC", "INGR", "CALM", "SAM", "SPB",                               # 필수소비재
        "NOV", "HP", "RRC", "CNX", "MUR", "OII", "PTEN", "DINO",                                   # 에너지
        "ZION", "EWBC", "FHN", "SNV", "CBSH", "UNM", "RGA", "ORI", "MKL", "SEIC",                 # 금융
        "HSIC", "OMI", "CHE", "THC", "UTHR", "BMRN", "EXEL", "TECH", "HAE", "NBIX",               # 헬스케어
        "R", "LECO", "TTC", "GGG", "DCI", "CSL", "FLS", "AOS", "OSK", "HUBB",                     # 산업재
        "RPM", "OLN", "CMC", "RS", "SON", "SEE", "CCK", "ASH", "FMC", "WLK",                      # 소재
        "OGE", "IDA", "POR", "ALE", "BKH", "AVA", "UGI", "NJR",                                    # 유틸리티
        "MAC", "SLG", "HIW", "KRC", "OHI", "NNN", "EGP", "ELS",                                    # 리츠
        "003670.KS", "000990.KS", "001680.KS", "004490.KS", "005180.KS", "005250.KS", "005610.KS", "006120.KS",
        "006650.KS", "010620.KS", "010780.KS", "011170.KS", "011930.KS", "014820.KS", "016380.KS", "017960.KS",
        "019680.KS", "020000.KS", "020150.KS", "023590.KS", "025540.KS", "031430.KS", "032640.KS", "039490.KS",
        "042670.KS", "049770.KS", "051600.KS", "052690.KS", "053690.KS", "064350.KS", "064960.KS", "069260.KS",
        "071320.KS", "079160.KS", "081660.KS", "082740.KS", "093370.KS", "095720.KS", "103140.KS", "111770.KS",
        "114090.KS", "120110.KS", "128940.KS", "161390.KS", "001230.KS", "002240.KS", "003030.KS", "007700.KS",
        "008060.KS", "009970.KS", "012510.KS", "006040.KS", "004430.KS",
    ],
}

LEC_STATE_NAMES = {1: "상승", 0: "횡보", -1: "하락"}
LEC_RS_NAMES = {2: "주도", 1: "약화", -1: "개선", -2: "소외"}
LEC_GRADE_RANK = {"🟢": 0, "🟡": 1, "🔴": 2, "⚫": 3}


# ---------------------------------------------------------------------
# 기초 지표
# ---------------------------------------------------------------------
def lec_base(df: pd.DataFrame) -> pd.DataFrame:
    """앱과 같은 기초 지표(compute_base_primitives) + 정리본에 맞춘 RSI14 + 최근 5거래일 변화(ATR14 단위)."""
    b = compute_base_primitives(df)
    if not len(b):
        return b
    c = b["Close"]
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        rsi = 100.0 - 100.0 / (1.0 + gain / loss)
    rsi = rsi.where(loss > 0, np.where(gain > 0, 100.0, 50.0))   # 하락이 없으면 100, 움직임이 없으면 50
    b["RSI14"] = rsi
    lr = LECTURE_CONFIG["SLOPE_DAYS"]
    with np.errstate(divide="ignore", invalid="ignore"):
        b["DROP5"] = (c - c.shift(lr)) / b["ATR14"].replace(0, np.nan)
    return b


def lec_zone(level: float, atr: float) -> Tuple[float, float]:
    """구간 = 가격 ± max(1%, 0.5×ATR14) (앱 calc_zone_bounds와 같은 식)."""
    a = atr if (atr is not None and np.isfinite(atr) and atr > 0) else level * 0.02
    hw = max(LECTURE_CONFIG["ZONE_PCT_MIN"] * level, LECTURE_CONFIG["ZONE_ATR_MULT"] * a)
    return float(level - hw), float(level + hw)


def lec_swings(b: pd.DataFrame) -> Dict[str, np.ndarray]:
    """확정 스윙 점. origin = 스윙 봉 위치, 확정은 origin + 5 거래일부터. 같은 높이가 둘이면 둘 다(앱과 같음)."""
    sh = np.flatnonzero(np.isfinite(b["RAW_SH"].to_numpy(dtype=float)))
    sl = np.flatnonzero(np.isfinite(b["RAW_SL"].to_numpy(dtype=float)))
    return {"h_o": sh, "h_p": b["High"].to_numpy(dtype=float)[sh],
            "l_o": sl, "l_p": b["Low"].to_numpy(dtype=float)[sl]}


def lec_state(b: pd.DataFrame, sw: Optional[Dict[str, np.ndarray]] = None, return_flags: bool = False):
    """종목(또는 지수) 상태: +1 상승, −1 하락, 0 횡보 (Q2·Q12). 날마다 그날까지 확정된 스윙만 쓴다.

    상승 = 최근 두 확정 저점 L1 > L2 그리고 L1 이후 종가가 한 번이라도 'L1보다 앞선 가장 최근 확정 고점' 위로 마감.
    하락 = 최근 두 확정 고점 H1 < H2 그리고 H1 이후 종가가 한 번이라도 'H1보다 앞선 가장 최근 확정 저점' 아래로 마감.
    둘 다이거나 둘 다 아니면 횡보.
    return_flags=True면 (상태, 상승 쪽 돌파 이력, 하락 쪽 이탈 이력)을 돌려준다(전환 증거 ② '새 고점'에 씀, 2단계).
    """
    sw = sw or lec_swings(b)
    lr = LECTURE_CONFIG["SWING_LR"]
    close = b["Close"].to_numpy(dtype=float)
    n = len(close)
    ho, hp, lo_, lp = list(sw["h_o"]), sw["h_p"], list(sw["l_o"]), sw["l_p"]
    out = np.zeros(n, dtype=int)
    upf = np.zeros(n, dtype=bool)
    dnf = np.zeros(n, dtype=bool)
    nh = nl = 0                     # 확정된 개수
    up_ok = dn_ok = False
    up_ref = dn_ref = np.nan        # 넘어야 할 고점 / 깨야 할 저점
    up_from = dn_from = -1          # 이 위치 뒤의 종가만 본다
    for t in range(n):
        lim = t - lr
        changed_l = changed_h = False
        while nl < len(lo_) and lo_[nl] <= lim:
            nl += 1
            changed_l = True
        while nh < len(ho) and ho[nh] <= lim:
            nh += 1
            changed_h = True
        if changed_l or changed_h:
            # 상승 쪽 기준 다시 잡기: 최근 확정 저점 L1, 그보다 앞선 최근 확정 고점
            if nl >= 1:
                o1 = lo_[nl - 1]
                k = bisect_left(ho, o1, 0, nh)          # origin < o1 인 고점 개수
                new_ref = hp[k - 1] if k >= 1 else np.nan
                if o1 != up_from or not (new_ref == up_ref or (np.isnan(new_ref) and np.isnan(up_ref))):
                    up_from, up_ref = o1, new_ref
                    seg = close[o1 + 1:t + 1]
                    up_ok = bool(np.isfinite(up_ref) and len(seg) and np.nanmax(seg) > up_ref)
            if nh >= 1:
                o1 = ho[nh - 1]
                k = bisect_left(lo_, o1, 0, nl)
                new_ref = lp[k - 1] if k >= 1 else np.nan
                if o1 != dn_from or not (new_ref == dn_ref or (np.isnan(new_ref) and np.isnan(dn_ref))):
                    dn_from, dn_ref = o1, new_ref
                    seg = close[o1 + 1:t + 1]
                    dn_ok = bool(np.isfinite(dn_ref) and len(seg) and np.nanmin(seg) < dn_ref)
        if up_from >= 0 and not up_ok and np.isfinite(up_ref) and t > up_from and close[t] > up_ref:
            up_ok = True
        if dn_from >= 0 and not dn_ok and np.isfinite(dn_ref) and t > dn_from and close[t] < dn_ref:
            dn_ok = True
        up = bool(nl >= 2 and lp[nl - 1] > lp[nl - 2] and up_ok)
        dn = bool(nh >= 2 and hp[nh - 1] < hp[nh - 2] and dn_ok)
        out[t] = 1 if (up and not dn) else (-1 if (dn and not up) else 0)
        upf[t], dnf[t] = up_ok, dn_ok
    return (out, upf, dnf) if return_flags else out


def lec_market_grade(idx_df: pd.DataFrame) -> pd.Series:
    """지수 시장 등급(1번 표, Q2~Q5): ⚫ 종가 < 200일 EMA / 🔴 하락 구조 + 종가 < 20일선 + 20일선 하락 /
    🟢 상승 구조 + 종가 > 20·60일선 + 두 선 상승 + ADX 강함 / 나머지 🟡. 하향 2일·상향 3일 연속이어야 바뀜(⚫ 포함).
    ADX 강함: ≥ 25면 강함, < 20이면 아님, 그 사이는 직전 판정 유지. ±DI는 쓰지 않는다. 데이터 처음부터 하루씩."""
    c = LECTURE_CONFIG
    b = lec_base(idx_df)
    if len(b) < c["GRADE_MIN_BARS"] + 1:
        return pd.Series(dtype=object)
    st = lec_state(b)
    cl = b["Close"].to_numpy(dtype=float)
    s20, s60, e200 = (b[k].to_numpy(dtype=float) for k in ("SMA20", "SMA60", "EMA200"))
    g20, g60 = b["SMA20_SLOPE"].to_numpy(dtype=float), b["SMA60_SLOPE"].to_numpy(dtype=float)
    adx = b["ADX14"].to_numpy(dtype=float)
    out = np.full(len(b), None, dtype=object)
    cur, up_s, dn_s, strong = "🟡", 0, 0, False
    for t in range(len(b)):
        if adx[t] >= c["ADX_STRONG"]:
            strong = True
        elif adx[t] < c["ADX_NONE"]:
            strong = False
        if t < c["GRADE_MIN_BARS"]:
            continue
        if cl[t] < e200[t]:
            cand = "⚫"
        elif st[t] == -1 and cl[t] < s20[t] and g20[t] < 0:
            cand = "🔴"
        elif st[t] == 1 and cl[t] > s20[t] and cl[t] > s60[t] and g20[t] > 0 and g60[t] > 0:
            cand = "🟢" if strong else "🟡"
        else:
            cand = "🟡"
        rc, rr = LEC_GRADE_RANK[cand], LEC_GRADE_RANK[cur]
        if rc > rr:          # 나빠짐
            dn_s, up_s = dn_s + 1, 0
            if dn_s >= c["GRADE_DOWN_DAYS"]:
                cur, dn_s = cand, 0
        elif rc < rr:        # 좋아짐
            up_s, dn_s = up_s + 1, 0
            if up_s >= c["GRADE_UP_DAYS"]:
                cur, up_s = cand, 0
        else:
            up_s = dn_s = 0
        out[t] = cur
    return pd.Series(out, index=b.index).dropna()


# ---------------------------------------------------------------------
# 상대강도 (2번) — 종목 ÷ 섹터 ETF
# ---------------------------------------------------------------------
def _lec_rs_code(above: bool, pos: bool) -> int:
    return 2 if (above and pos) else (1 if above else (-1 if pos else -2))


def lec_rs_line(close: pd.Series, bench: Optional[pd.Series]) -> pd.Series:
    """RS선 = 종목 종가 ÷ 기준 종가 (두 쪽 모두 거래한 날만). 기준이 없으면 빈 시리즈 = '데이터 없음'."""
    if bench is None or not len(bench):
        return pd.Series(dtype=float)
    bb = pd.Series(bench, dtype=float)
    bb = bb[bb > 0]
    common = close.index.intersection(bb.index)
    if not len(common):
        return pd.Series(dtype=float)
    return (close.loc[common].astype(float) / bb.loc[common]).dropna()


def lec_rs_daily(rs: pd.Series) -> pd.Series:
    """일봉 4구간 코드(2 주도·1 약화·−1 개선·−2 소외): RS선 vs 50일 평균(최소 20일) × 20일 변화. 계산 불가는 NaN."""
    c = LECTURE_CONFIG
    if not len(rs):
        return pd.Series(dtype=float)
    ma = rs.rolling(c["RS_MA"], min_periods=c["RS_MA_MIN"]).mean()
    roc = rs / rs.shift(c["RS_ROC"]) - 1.0
    ok = ma.notna() & roc.notna()
    code = np.where(rs > ma, np.where(roc > 0, 2, 1), np.where(roc > 0, -1, -2)).astype(float)
    return pd.Series(np.where(ok, code, np.nan), index=rs.index)


def lec_rs_weekly_at(rs: pd.Series, when: pd.Timestamp) -> Optional[int]:
    """그날의 주봉 4구간 코드. 주봉 = 끝난 주의 마지막 값 + 그날까지 진행 중인 주(그날 값). 10주 평균(최소 5주) × 4주 변화.
    그날까지의 주가 12주 미만이거나 그날 값이 없으면 None(데이터 없음)."""
    c = LECTURE_CONFIG
    if not len(rs):
        return None
    when = pd.Timestamp(when)
    hist = rs[rs.index <= when]
    if not len(hist) or hist.index[-1] != when:
        return None
    wk = hist.resample("W-FRI").last().dropna()     # 마지막 칸 = 진행 중인 주(그날 값)
    if len(wk) < c["RS_W_MIN_WEEKS"]:
        return None
    vals = wk.to_numpy(dtype=float)
    win = vals[-c["RS_W_MA"]:]
    if len(win) < c["RS_W_MA_MIN"] or len(vals) <= c["RS_W_ROC"]:
        return None
    ma = float(np.mean(win))
    roc = vals[-1] / vals[-1 - c["RS_W_ROC"]] - 1.0
    return _lec_rs_code(vals[-1] > ma, roc > 0)


def lec_rs_weekly_series(rs: pd.Series) -> pd.Series:
    """모든 날의 주봉 4구간 코드(lec_rs_weekly_at과 같은 값, 한 번에). 계산 불가는 NaN."""
    c = LECTURE_CONFIG
    if not len(rs):
        return pd.Series(dtype=float)
    v = rs.to_numpy(dtype=float)
    wk = rs.index.to_period("W-FRI")
    codes, uniq = pd.factorize(wk, sort=True)
    W = pd.Series(v).groupby(codes).last().to_numpy(dtype=float)     # 주마다 마지막 값 (주 번호 0..K-1)
    S = np.concatenate([[0.0], np.cumsum(W)])                       # S[k] = W[0..k-1] 합
    k = codes
    m_prev = np.minimum(c["RS_W_MA"] - 1, k)                        # 끝난 주 중 평균에 넣을 개수
    sum_prev = S[k] - S[k - m_prev]
    ma = (sum_prev + v) / (m_prev + 1)
    ok = (k + 1 >= c["RS_W_MIN_WEEKS"]) & (m_prev + 1 >= c["RS_W_MA_MIN"]) & (k >= c["RS_W_ROC"])
    ref = np.where(k >= c["RS_W_ROC"], W[np.maximum(k - c["RS_W_ROC"], 0)], np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        roc = v / ref - 1.0
    code = np.where(v > ma, np.where(roc > 0, 2, 1), np.where(roc > 0, -1, -2)).astype(float)
    return pd.Series(np.where(ok & np.isfinite(v), code, np.nan), index=rs.index)


# ---------------------------------------------------------------------
# BNF 기준 이격도 (Q62)
# ---------------------------------------------------------------------
def lec_bnf_base_at(b: pd.DataFrame, sw: Dict[str, np.ndarray], t: int) -> Optional[float]:
    """날짜 t의 기준 이격도 = 지난 750거래일 안에 있고 t까지 확정된 스윙 저점들의 이격도(종가 ÷ 25일선 × 100) 중
    가장 낮은 5개의 중앙값. 3개 미만이면 None(판정 불가)."""
    c = LECTURE_CONFIG
    bn = c["BNF"]
    lo_ = sw["l_o"]
    lim = t - c["SWING_LR"]
    o = lo_[(lo_ >= t - bn["BASE_LOOKBACK"] + 1) & (lo_ <= lim)]
    if len(o) < bn["BASE_MIN"]:
        return None
    d = b["DISP25"].to_numpy(dtype=float)[o]
    d = d[np.isfinite(d)]
    if len(d) < bn["BASE_MIN"]:
        return None
    return float(np.median(np.sort(d)[:bn["BASE_N"]]))


def lec_bnf_base_series(b: pd.DataFrame, sw: Dict[str, np.ndarray]) -> np.ndarray:
    """모든 날의 기준 이격도(lec_bnf_base_at과 같은 값, 한 번에). 판정 불가인 날은 NaN."""
    c = LECTURE_CONFIG
    bn = c["BNF"]
    n = len(b)
    lo_ = sw["l_o"]
    dl = b["DISP25"].to_numpy(dtype=float)[lo_] if len(lo_) else np.array([])
    out = np.full(n, np.nan)
    for t in range(n):
        a_ = np.searchsorted(lo_, t - bn["BASE_LOOKBACK"] + 1, "left")
        z_ = np.searchsorted(lo_, t - c["SWING_LR"], "right")
        if z_ - a_ < bn["BASE_MIN"]:
            continue
        d = dl[a_:z_]
        d = d[np.isfinite(d)]
        if len(d) >= bn["BASE_MIN"]:
            out[t] = float(np.median(np.sort(d)[:bn["BASE_N"]]))
    return out
