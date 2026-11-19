# C:\Users\hanju\global-sector-agent\pages\3_📒_매매_기록.py
# 매매 기록 (2026-11-20, engine/journal.py) — 내 실제 매매를 PC에만 저장하고, 비용 포함 손익·지수 대비·방식별 채점·손절 준수·해외주식 세금 추정.
# 저장 폴더 data/journal/은 .gitignore에 있다(공개 저장소). 인터넷 서버에서는 저장하지 않고 CSV를 불러와 보기만 한다.
import html as _html

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

import engine
import utils

st.set_page_config(page_title="매매 기록 | 투자터미널", layout="wide")
utils.init_session()

CFG = engine.JOURNAL_CONFIG
MODE = engine.journal_storage_mode()
STYLES = list(CFG["STYLES"])
_TONE = {"go": "#16a34a", "block": "#dc2626", "wait": "#64748b"}


def html_md(s: str) -> None:
    st.markdown("\n".join(l.strip() for l in str(s).split("\n") if l.strip()), unsafe_allow_html=True)


def _esc(x) -> str:
    return _html.escape("" if x is None else str(x))


def _pct(v, signed: bool = True) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "-"
    return f"{v * 100:+.2f}%" if signed else f"{v * 100:.0f}%"


def _won(v) -> str:
    return "-" if v is None or not np.isfinite(v) else f"{v:,.0f}원"


def _px(v, market: str) -> str:
    if v is None or not np.isfinite(v):
        return "-"
    return engine.fmt_price(v, market == "KR")


def _norm(t: str) -> str:
    t = str(t).strip().upper()
    if t.isdigit() and len(t) == 6:
        return f"{t}.KS"
    return t


def _today():
    return pd.Timestamp.now(tz="Asia/Seoul").date()


def _naive(df):
    if df is None or not len(df):
        return pd.DataFrame()
    d = df.copy()
    idx = pd.DatetimeIndex(pd.to_datetime(d.index))
    d.index = (idx.tz_localize(None) if idx.tz is not None else idx).normalize()
    return d


@st.cache_data(ttl=1800, show_spinner=False)
def load_hist(sym: str, start: str) -> pd.DataFrame:
    """일봉(시가·저가·종가). 못 받으면 빈 표 — 지수 비교·갭 확인이 '자료 없음'이 된다."""
    try:
        d = yf.Ticker(sym).history(start=start, auto_adjust=False)
    except Exception:
        return pd.DataFrame()
    d = _naive(d)
    if not len(d) or "Close" not in d:
        return pd.DataFrame()
    d = d[pd.to_numeric(d["Close"], errors="coerce") > 0]
    return d[[c for c in ("Open", "High", "Low", "Close") if c in d]]


@st.cache_data(ttl=1800, show_spinner=False)
def fx_on(day: str):
    """원/달러(야후 KRW=X 종가, 그날 또는 그 전 마지막 값) — 결제일 기준환율의 근사. 증권사 계산 자료 값으로 고치면 정확."""
    d = load_hist("KRW=X", str((pd.Timestamp(day) - pd.Timedelta(days=14)).date()))
    if not len(d):
        return None
    v = d["Close"].loc[:pd.Timestamp(day)]
    return float(v.iloc[-1]) if len(v) else None


def flash(msg: str) -> None:
    st.session_state["journal_flash"] = msg
    st.rerun()


def save_or_error(fn, *a, **k):
    try:
        out = fn(*a, **k)
    except (ValueError, PermissionError) as e:
        st.error(str(e))
        return None
    return out if out is not None else True


st.markdown(f'## {utils.line_icon("check", 22)} 매매 기록', unsafe_allow_html=True)
st.caption("내 실제 매매를 적고, 비용을 뺀 손익을 같은 기간 지수(미국 SPY · 한국 KOSPI)를 그냥 들고 있었을 때와 비교합니다. "
           "방식(가치 · 추세 · 단타)마다 끝난 거래 30건부터 미리 정한 규칙으로 채점합니다.")
if "journal_flash" in st.session_state:
    st.success(st.session_state.pop("journal_flash"))
if MODE == "local":
    st.caption("저장 위치: 이 PC의 data/journal/ — git이 무시하는 폴더라 공개 저장소에 올라가지 않습니다. 백업은 '내보내기 · 불러오기'에서 CSV로.")
    data = engine.journal_load()
else:
    st.info("이 앱은 인터넷 서버에서 실행 중이라 매매 기록을 저장하지 않습니다(공개 저장소). PC에서 앱을 실행하면 저장됩니다. "
            "'내보내기 · 불러오기'에서 PC에서 내보낸 CSV를 불러와 보기만 할 수 있습니다(창을 닫으면 사라짐).")
    data = st.session_state.get("journal_view") or engine.journal_empty()

trades, sells = data["trades"], data["sells"]
pos = engine.journal_positions(data)

tab_perf, tab_pos, tab_buy, tab_edit, tab_io = st.tabs(["성과", "보유 · 매도", "매수 기록", "고치기 · 지우기", "내보내기 · 불러오기"])

# ---------------------------------------------------------------- 성과
with tab_perf:
    if not len(trades):
        st.info("아직 기록이 없습니다. 딥다이브의 '매매 기록에 추가'(그날 판정을 함께 저장) 또는 '매수 기록' 탭에서 적어 주세요.")
    else:
        start = str((pd.Timestamp(min(trades["buy_date"])) - pd.Timedelta(days=10)).date())
        bench = {}
        for mk, sym in CFG["BENCH"].items():
            if (trades["market"] == mk).any():
                h = load_hist(sym, start)
                bench[mk] = h["Close"] if len(h) else None
        res = engine.journal_results(data, bench=bench)
        need = res[res["stop_status"] == "어김(갭 확인 불가)"]
        if len(need):
            prices = {}
            for tk in need["ticker"].unique():
                h = load_hist(tk, str((pd.Timestamp(need[need["ticker"] == tk]["buy_date"].min()) - pd.Timedelta(days=5)).date()))
                prices[tk] = h if len(h) else None
            res = engine.journal_results(data, bench=bench, prices=prices)
        summ = engine.journal_style_summary(res)
        rows = []
        for s in summ:
            stop_txt = ("-" if s["stop_rate"] is None else f"{s['stop_rate'] * 100:.0f}% ({s['stop_kept']}/{s['stop_kept'] + s['stop_violated']})")
            if s["stop_gap"]:
                stop_txt += f" · 갭 {s['stop_gap']}"
            rows.append({"방식": s["style"], "끝난 거래": s["n_closed"], "보유 중": s["n_open"],
                         "승률": _pct(s["win_rate"], signed=False) if s["win_rate"] is not None else "-",
                         "거래당 손익(비용 포함)": _pct(s["mean_ret"]), "지수 대비 초과": _pct(s["mean_excess"]),
                         "t(월 묶음)": "-" if s["t"] is None else f"{s['t']:+.2f}", "판정": s["verdict"], "손절 준수": stop_txt})
        st.markdown("**방식별 채점**")
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption(f"채점 규칙(기록이 쌓이기 전에 고정, {CFG['VERSION']}): 거래마다 (비용 포함 수익률 − 같은 기간 지수 수익률). "
                   f"끝난 거래 {CFG['MIN_CLOSED']}건부터 판정 — 평균 > 0 이고 t ≥ {CFG['T_PASS']:g} → 지수보다 나음, t ≤ −{CFG['T_PASS']:g} → 지수 보유가 나음, "
                   "그 사이 → 아직 구분 안 됨(t는 산 달로 묶음). 한계: 거래당 흔들림이 8%쯤이면 30건으로는 지수와 약 4%p 넘게 차이 나야 구분됩니다. "
                   f"손절 준수 = 손실로 끝난 거래 중 평균 판 값이 계획 손절 − {CFG['STOP_TOL'] * 100:.0f}%p 안. 손절선에 닿은 날 시가부터 손절선 아래였고 그날 다 팔았으면 '갭'(위반 아님).")
        closed = res[res["status"] == "청산"].sort_values("close_date", ascending=False)
        if len(closed):
            st.markdown("**끝난 거래**")
            st.dataframe(pd.DataFrame({
                "종목": closed["ticker"], "방식": closed["style"], "산 날": closed["buy_date"], "판 날": closed["close_date"],
                "보유일": closed["hold_days"].map(lambda v: "-" if not np.isfinite(v) else f"{v:.0f}"),
                "손익(비용 포함)": closed["ret"].map(_pct), "지수": closed["idx_ret"].map(_pct), "초과": closed["excess"].map(_pct),
                "손절 결과": closed["stop_status"], "그날 판정": closed["verdict_headline"].where(closed["verdict_source"] == "딥다이브", "당일 판정 없음"),
                "진입 근거": closed["reason"]}), hide_index=True, width="stretch")
        tax = engine.journal_tax_overseas(data)
        if tax["rows"] or tax["missing_fx"]:
            st.markdown("**해외주식 세금 추정**")
            if tax["rows"]:
                st.dataframe(pd.DataFrame([{"해(결제일 기준)": r["year"], "판 기록": r["n_sells"], "원화 손익(비용 뺌)": _won(r["gain"]),
                                            "기본공제": _won(r["deduction"]), "과세표준": _won(r["taxable"]), "예상 세액(22%)": _won(r["tax"]),
                                            "세후 손익": _won(r["after_tax"])} for r in tax["rows"]]), hide_index=True, width="stretch")
            notes = []
            if tax["missing_fx"]:
                notes.append(f"환율이 없는 판 기록 {tax['missing_fx']}건은 뺐습니다('고치기'에서 환율을 넣어 주세요)")
            if tax["cross_year"]:
                notes.append(f"결제일이 다음 해로 넘어간 판 기록 {tax['cross_year']}건은 다음 해에 넣었습니다")
            st.caption(f"{tax['as_of']} 확인 기준: 한 해(결제일 기준) 모든 해외주식 손익을 합쳐 250만 원 공제 뒤 22%(양도소득세 20% + 지방소득세 2%). "
                       "원화 환산은 결제일 환율 — 자동 값은 야후 원/달러 종가라 근사치. 이 기록 밖의 해외주식 매매·RIA 감면은 반영하지 않으며, "
                       "실제 신고는 증권사 해외주식 양도소득 계산 자료로 하세요." + (" · " + " · ".join(notes) if notes else ""))
        comp = engine.journal_verdict_compare(res)
        if any(c["n_match"] or c["n_other"] for c in comp):
            st.markdown("**참고: 그날 딥다이브 판정과 같은 방식으로 산 거래 vs 아닌 거래**")
            st.dataframe(pd.DataFrame([{"방식": c["style"], "판정과 같음(건)": c["n_match"], "지수 대비 초과(같음)": _pct(c["excess_match"]),
                                        "판정과 다름(건)": c["n_other"], "지수 대비 초과(다름)": _pct(c["excess_other"])} for c in comp]),
                         hide_index=True, width="stretch")
            st.caption("가치 · 추세 = 그날 '맞음', 단타 = 그날 '조합 충족'. 판정 없이 적은 거래는 뺌. 참고용(판정 아님).")

# ---------------------------------------------------------------- 보유 · 매도
with tab_pos:
    op = pos[pos["status"] != "청산"]
    if not len(op):
        st.info("보유 중인 기록이 없습니다.")
    else:
        rows = []
        for _, r in op.iterrows():
            h = load_hist(r["ticker"], str((pd.Timestamp(_today()) - pd.Timedelta(days=10)).date()))
            last = float(h["Close"].iloc[-1]) if len(h) else np.nan
            rows.append({"ID": r["id"], "종목": r["ticker"], "방식": r["style"], "산 날": r["buy_date"],
                         "산 값": _px(r["buy_price"], r["market"]), "남은 수량": f"{r['remaining']:g}",
                         "손절 계획": _px(r["stop_plan"], r["market"]), "지금 값": _px(last, r["market"]),
                         "평가 손익(비용 전)": _pct(last / r["buy_price"] - 1 if np.isfinite(last) else np.nan),
                         "손절까지": _pct(r["stop_plan"] / last - 1 if np.isfinite(last) else np.nan),
                         "진입 근거": r["reason"]})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        if MODE == "local":
            st.markdown("**판 기록 적기**")
            opts = {f"{r['id']} · {r['ticker']} · 남은 {r['remaining']:g}": r for _, r in op.iterrows()}
            pick = st.selectbox("판 거래", list(opts), key="jr_sell_pick")
            t = opts[pick]
            with st.form("jr_sell_form"):
                c1, c2, c3 = st.columns(3)
                s_day = c1.date_input("판 날", value=_today(), key="jr_sell_day")
                s_px = c2.number_input("판 값", min_value=0.0, value=float(t["buy_price"]), format="%.4f", key="jr_sell_px")
                s_qty = c3.number_input("판 수량", min_value=0.0, value=float(t["remaining"]), key="jr_sell_qty")
                c4, c5, c6 = st.columns(3)
                s_fee = c4.number_input("수수료(거래 통화)", min_value=0.0, value=0.0, format="%.4f", key="jr_sell_fee")
                s_tax = c5.number_input("세금(0 = 국내는 증권거래세 자동)", min_value=0.0, value=0.0, format="%.4f", key="jr_sell_tax")
                s_fx = (c6.number_input("환율(원/달러, 0 = 결제일 자동)", min_value=0.0, value=0.0, format="%.2f", key="jr_sell_fx")
                        if t["market"] == "US" else 1.0)
                s_note = st.text_input("메모(선택)", key="jr_sell_note")
                if st.form_submit_button("판 기록 저장"):
                    fx = s_fx if (t["market"] == "KR" or s_fx > 0) else fx_on(engine.journal_settle_date(s_day, "US"))
                    sid = save_or_error(engine.journal_add_sell, t["id"], {
                        "sell_date": s_day, "sell_price": s_px, "qty": s_qty, "sell_fee": s_fee,
                        "sell_tax": (None if s_tax == 0 else s_tax), "sell_fx": fx, "note": s_note})
                    if sid:
                        flash(f"판 기록 저장: {sid}")
        else:
            st.caption("인터넷 서버에서는 판 기록을 적을 수 없습니다(PC에서).")

# ---------------------------------------------------------------- 매수 기록
with tab_buy:
    st.caption("그날 '세 방식 한눈에' 판정을 함께 저장하려면 딥다이브에서 '매매 기록에 추가'로 적으세요. 여기서 적은 거래는 '당일 판정 없음'입니다. "
               "강의 9-7: 진입 근거 · 손절가 · 수량이 적히지 않으면 사지 않는다 — 셋 다 필수입니다.")
    if MODE != "local":
        st.caption("인터넷 서버에서는 저장하지 않습니다(PC에서).")
    else:
        with st.form("jr_buy_form"):
            c1, c2, c3 = st.columns(3)
            b_tk = c1.text_input("종목(티커, 예: NVDA · 005930.KS)", key="jr_buy_tk")
            b_style = c2.selectbox("방식", STYLES, key="jr_buy_style")
            b_day = c3.date_input("산 날", value=_today(), key="jr_buy_day")
            c4, c5, c6 = st.columns(3)
            b_px = c4.number_input("산 값", min_value=0.0, value=0.0, format="%.4f", key="jr_buy_px")
            b_qty = c5.number_input("수량", min_value=0.0, value=0.0, key="jr_buy_qty")
            b_fee = c6.number_input("수수료(거래 통화)", min_value=0.0, value=0.0, format="%.4f", key="jr_buy_fee")
            c7, c8 = st.columns(2)
            b_stop = c7.number_input("손절 계획 가격", min_value=0.0, value=0.0, format="%.4f", key="jr_buy_stop")
            b_fx = c8.number_input("환율(미국 종목, 원/달러, 0 = 결제일 자동)", min_value=0.0, value=0.0, format="%.2f", key="jr_buy_fx")
            b_reason = st.text_input("진입 근거(한 줄)", key="jr_buy_reason")
            if st.form_submit_button("산 기록 저장"):
                tk = _norm(b_tk)
                mk = engine.journal_market(tk) if tk else "US"
                fx = 1.0 if mk == "KR" else (b_fx if b_fx > 0 else fx_on(engine.journal_settle_date(b_day, "US")))
                tid = save_or_error(engine.journal_add_trade, {
                    "ticker": tk, "style": b_style, "buy_date": b_day, "buy_price": b_px or None, "qty": b_qty or None,
                    "buy_fee": b_fee, "buy_fx": fx, "stop_plan": b_stop or None, "reason": b_reason})
                if tid:
                    flash(f"산 기록 저장: {tid} (당일 판정 없음)")

# ---------------------------------------------------------------- 고치기 · 지우기
with tab_edit:
    if MODE != "local":
        st.caption("인터넷 서버에서는 고칠 수 없습니다(PC에서).")
    elif not len(trades):
        st.info("고칠 기록이 없습니다.")
    else:
        st.caption("고칠 수 있는 칸: 방식 · 손절 계획 · 진입 근거 · 수수료 · 환율. 값 · 수량 · 날짜를 잘못 적었으면 지우고 다시 적으세요(판 기록과 어긋나지 않게).")
        tmap = {f"{r['id']} · {r['ticker']} · {r['buy_date']}": r for _, r in trades.iterrows()}
        e_pick = st.selectbox("산 기록", list(tmap), key="jr_edit_pick")
        et = tmap[e_pick]
        with st.form("jr_edit_form"):
            c1, c2, c3, c4 = st.columns(4)
            e_style = c1.selectbox("방식", STYLES, index=STYLES.index(et["style"]) if et["style"] in STYLES else 0, key="jr_edit_style")
            e_stop = c2.number_input("손절 계획 가격", min_value=0.0, value=float(et["stop_plan"]) if np.isfinite(et["stop_plan"]) else 0.0,
                                     format="%.4f", key="jr_edit_stop")
            e_fee = c3.number_input("수수료", min_value=0.0, value=float(et["buy_fee"]) if np.isfinite(et["buy_fee"]) else 0.0,
                                    format="%.4f", key="jr_edit_fee")
            e_fx = c4.number_input("환율", min_value=0.0, value=float(et["buy_fx"]) if np.isfinite(et["buy_fx"]) else 0.0,
                                   format="%.2f", key="jr_edit_fx")
            e_reason = st.text_input("진입 근거", value=et["reason"], key="jr_edit_reason")
            if st.form_submit_button("고친 내용 저장"):
                ok = save_or_error(engine.journal_update_trade, et["id"], {"style": e_style, "stop_plan": e_stop or None,
                                                                           "reason": e_reason, "buy_fee": e_fee, "buy_fx": e_fx or None})
                if ok:
                    flash(f"고침: {et['id']}")
        st.markdown("**지우기**")
        d1, d2 = st.columns(2)
        with d1:
            del_t = st.selectbox("산 기록(딸린 판 기록도 함께)", list(tmap), key="jr_del_trade")
            if st.checkbox("정말 지움", key="jr_del_trade_ok") and st.button("산 기록 지우기", key="jr_del_trade_btn"):
                n = save_or_error(engine.journal_delete, "trade", tmap[del_t]["id"])
                if n:
                    flash(f"지움: {tmap[del_t]['id']} ({n}줄)")
        with d2:
            if len(sells):
                smap = {f"{r['id']} · {r['trade_id']} · {r['sell_date']}": r for _, r in sells.iterrows()}
                del_s = st.selectbox("판 기록", list(smap), key="jr_del_sell")
                if st.checkbox("정말 지움", key="jr_del_sell_ok") and st.button("판 기록 지우기", key="jr_del_sell_btn"):
                    n = save_or_error(engine.journal_delete, "sell", smap[del_s]["id"])
                    if n:
                        flash(f"지움: {smap[del_s]['id']}")

# ---------------------------------------------------------------- 내보내기 · 불러오기
with tab_io:
    if len(trades):
        c1, c2 = st.columns(2)
        c1.download_button("산 기록 내보내기 (trades.csv)", engine.journal_csv_bytes(trades, "trades"), file_name="trades.csv",
                           mime="text/csv", key="jr_dl_trades")
        c2.download_button("판 기록 내보내기 (sells.csv)", engine.journal_csv_bytes(sells, "sells"), file_name="sells.csv",
                           mime="text/csv", key="jr_dl_sells")
    up_t = st.file_uploader("산 기록 CSV (trades.csv)", type=["csv"], key="jr_up_trades")
    up_s = st.file_uploader("판 기록 CSV (sells.csv, 선택)", type=["csv"], key="jr_up_sells")
    how = st.radio("불러오는 방식", ["합치기(같은 ID는 건너뜀)", "바꾸기(지금 기록을 지우고 불러온 것으로)"], key="jr_up_mode",
                   disabled=(MODE != "local"))
    if st.button("불러오기", key="jr_up_btn", disabled=up_t is None):
        rd = lambda f: pd.read_csv(f, dtype=str, keep_default_na=False, encoding="utf-8-sig")
        rep = engine.journal_import(data if MODE == "local" else engine.journal_empty(), rd(up_t), rd(up_s) if up_s is not None else None,
                                    mode=("replace" if how.startswith("바꾸기") else "merge") if MODE == "local" else "replace")
        for e in rep["errors"][:20]:
            st.warning(e)
        msg = f"불러옴: 산 기록 {rep['added_trades']}건 · 판 기록 {rep['added_sells']}건 · 같은 ID라 건너뜀 {rep['skipped']}건"
        if MODE == "local":
            if save_or_error(engine.journal_save, rep["data"]):
                flash(msg)
        else:
            st.session_state["journal_view"] = rep["data"]
            flash(msg + " (보기만 — 저장 안 함)")
