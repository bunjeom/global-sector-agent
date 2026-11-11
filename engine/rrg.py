"""4분면(RRG) 축 정규화 v2.0(RRG_CONFIG), 라벨 배치, 일관성 점검, 실제 종가 번들 build_real_rrg_bundle.

engine.py를 기능별로 나눈 모듈. 바깥에서는 `import engine` 후 `engine.이름`으로 쓴다(engine/__init__.py).
"""
import math
import os
import numpy as np
import pandas as pd

from .home import NARRATIVE_BASKETS_SPEC, SAMPLE_DATA

# ==============================================================================
# [Module 2 & 4 Integrated] RRG 축 정규화 엔진 v2.0 + 국면 게이트(Market Gate) v1.1
# - 단일 출처(RRG_CONFIG, GATE_CONFIG), 데이터 무결성 3원칙, 전환 로깅 및 일관성 검증 포함
# ==============================================================================

# ------------------------------------------------------------------------------
# 1. RRG_CONFIG: 임계값 및 시계(Clock) 파라미터 단일 출처 관리
# ------------------------------------------------------------------------------
RRG_CONFIG = {
    "CLOCKS": {
        "daily": {"freq": "D", "n": 50, "m": 10, "L": 252, "tail_step": 5, "tail_points": 5, "label": "순환(일봉)"},
        "weekly": {"freq": "W-FRI", "n": 26, "m": 4, "L": 104, "tail_step": 1, "tail_points": 5, "label": "구조(주봉)"}
    },
    "EMA_SPAN": 3,
    "NEUTRAL_THRESHOLD": 0.25,  # |X| < 0.25 and |Y| < 0.25 -> 중립
    "AXIS_LIMIT": 3.0,          # 차트 고정 범위 [-3, 3] sigma
    "CHART_HEIGHT": 540,        # 우측 Top 3 카드 열과 맞춘 고정 픽셀 높이
    "MIN_OBS_RATIO": 0.8,       # 표본 짧음 판정 기준
    "COMPARE_LOG_PATH": "logs/rrg_compare.csv",
    "COMPARE_WINDOW_DAYS": 20,
    # '다음 후보'(개선 사분면) 필터 — 개선 사분면에 들어왔다가 다시 꺾이는 섹터를 거른다 (2026-11-10)
    "IGNITE_FILTER": {
        "MIN_MOVE_4W_X": 0.0,             # 최근 4주 X(상대강도) 변화가 이보다 커야 함 = 주도 쪽으로 이동 중
        "EXCLUDE_STRUCTURAL": ["소외"],    # 주봉 구조 사분면이 여기 있으면 제외 (큰 흐름이 약세)
    },
}


# [20거래일 뒤 제거 예정] 전환 기간 비교용 옛 축 계산 함수 (삭제 금지 규칙 준수)
def legacy_rrg_coordinates(prices: pd.DataFrame, bench: pd.Series, n: int = 50, m: int = 20) -> dict:
    """
    옛 축 계산: X = (RS / SMA(RS, 50) - 1) * 100, Y = (RS / RS_{t-20} - 1) * 100
    """
    results = {}
    for col in prices.columns:
        s = prices[col].dropna()
        b = bench.reindex(s.index).dropna()
        common_idx = s.index.intersection(b.index)
        if len(common_idx) < max(n, m) + 5:
            results[col] = {"x_series": pd.Series(dtype=float), "y_series": pd.Series(dtype=float), "quadrant": "데이터 없음"}
            continue
        rs = s.loc[common_idx] / b.loc[common_idx]
        sma = rs.rolling(n, min_periods=max(5, n // 2)).mean()
        x_ser = (rs / sma - 1.0) * 100.0
        y_ser = rs.pct_change(m) * 100.0
        xv = float(x_ser.iloc[-1]) if not np.isnan(x_ser.iloc[-1]) else np.nan
        yv = float(y_ser.iloc[-1]) if not np.isnan(y_ser.iloc[-1]) else np.nan
        if np.isnan(xv) or np.isnan(yv):
            q = "데이터 없음"
        elif xv > 0 and yv > 0:
            q = "주도"
        elif xv > 0 and yv <= 0:
            q = "약화"
        elif xv <= 0 and yv <= 0:
            q = "소외"
        else:
            q = "개선"
        results[col] = {"x_series": x_ser, "y_series": y_ser, "x": xv, "y": yv, "quadrant": q}
    return results


def classify_rrg_quadrant(x: float, y: float, neutral_thresh: float = None) -> str:
    """사분면 분류 (원점 주변 중립 영역 포함)"""
    if x is None or y is None or np.isnan(x) or np.isnan(y):
        return "데이터 없음"
    th = RRG_CONFIG["NEUTRAL_THRESHOLD"] if neutral_thresh is None else neutral_thresh
    if abs(x) < th and abs(y) < th:
        return "중립"
    if x > 0 and y > 0:
        return "주도"
    if x > 0 and y <= 0:
        return "약화"
    if x <= 0 and y <= 0:
        return "소외"
    return "개선"


def rrg_coordinates(prices: pd.DataFrame, bench: pd.Series, clock: str = "daily") -> dict:
    """
    1단계 새 축 정의:
    - RS_t = ETF 종가 / 기준 지수 종가 (SPY 또는 RSP)
    - r_t = RS_t / SMA(RS, n) - 1 -> EMA(3) 평활 -> X_t = r_t / std(r, L)
    - dX_t = X_t - X_{t-m} -> Y_t = dX_t / std(dX, L)
    """
    if clock not in RRG_CONFIG["CLOCKS"]:
        raise ValueError(f"Unsupported clock: {clock}")
    cfg = RRG_CONFIG["CLOCKS"][clock]
    n, m, L = cfg["n"], cfg["m"], cfg["L"]
    ema_span = RRG_CONFIG["EMA_SPAN"]

    if clock == "weekly":
        p_df = prices.resample("W-FRI").last()
        b_ser = bench.resample("W-FRI").last()
    else:
        p_df = prices.copy()
        b_ser = bench.copy()

    out = {}
    for basket in p_df.columns:
        s = p_df[basket].dropna()
        b = b_ser.reindex(s.index).dropna()
        idx = s.index.intersection(b.index)
        total_obs = len(idx)

        if total_obs < n + m + 5:
            out[basket] = {
                "x": np.nan, "y": np.nan, "quadrant": "데이터 없음",
                "x_series": pd.Series(dtype=float), "y_series": pd.Series(dtype=float),
                "tail_x": [], "tail_y": [], "short_sample": True, "obs_count": total_obs,
                "move_4w_x": np.nan, "move_4w_y": np.nan
            }
            continue

        rs = s.loc[idx] / b.loc[idx]
        sma_rs = rs.rolling(window=n, min_periods=max(5, n // 2)).mean()
        r_raw = (rs / sma_rs) - 1.0
        r_smooth = r_raw.ewm(span=ema_span, adjust=False).mean()

        min_std_p = max(10, min(L // 4, total_obs - n))
        std_r = r_smooth.rolling(window=L, min_periods=min_std_p).std(ddof=1).replace(0, np.nan)
        x_ser = r_smooth / std_r

        dx_ser = x_ser - x_ser.shift(m)
        std_dx = dx_ser.rolling(window=L, min_periods=min_std_p).std(ddof=1).replace(0, np.nan)
        y_ser = dx_ser / std_dx

        valid_xy = pd.concat([x_ser.rename("x"), y_ser.rename("y")], axis=1).dropna()
        if valid_xy.empty:
            out[basket] = {
                "x": np.nan, "y": np.nan, "quadrant": "데이터 없음",
                "x_series": x_ser, "y_series": y_ser,
                "tail_x": [], "tail_y": [], "short_sample": True, "obs_count": total_obs,
                "move_4w_x": np.nan, "move_4w_y": np.nan
            }
            continue

        cur_x = float(valid_xy["x"].iloc[-1])
        cur_y = float(valid_xy["y"].iloc[-1])
        quad = classify_rrg_quadrant(cur_x, cur_y)

        # 꼬리(최근 4주 5개 점) 추출
        step = cfg["tail_step"]
        n_pts = cfg["tail_points"]
        tail_indices = [len(valid_xy) - 1 - i * step for i in range(n_pts) if (len(valid_xy) - 1 - i * step) >= 0]
        tail_indices.reverse()
        tail_x = [float(valid_xy["x"].iloc[i]) for i in tail_indices]
        tail_y = [float(valid_xy["y"].iloc[i]) for i in tail_indices]

        move_4w_x = cur_x - tail_x[0] if len(tail_x) >= 2 else 0.0
        move_4w_y = cur_y - tail_y[0] if len(tail_y) >= 2 else 0.0
        short_sample = total_obs < (n + L)

        out[basket] = {
            "x": cur_x, "y": cur_y, "quadrant": quad,
            "x_series": x_ser, "y_series": y_ser,
            "tail_x": tail_x, "tail_y": tail_y,
            "short_sample": short_sample, "obs_count": total_obs,
            "move_4w_x": move_4w_x, "move_4w_y": move_4w_y
        }
    return out


def diagnose_rrg_axes(prices: pd.DataFrame, bench: pd.Series) -> dict:
    """
    2단계 진단: 과거 3년 데이터로 옛 축 vs 새 축(daily) 비교 검증
    """
    old_res = legacy_rrg_coordinates(prices, bench)
    new_res = rrg_coordinates(prices, bench, clock="daily")

    old_corrs, new_corrs = [], []
    old_abs_x, new_abs_x, basket_vols = [], [], []
    old_q_counts = {"주도": 0, "약화": 0, "소외": 0, "개선": 0}
    new_q_counts = {"주도": 0, "약화": 0, "소외": 0, "개선": 0, "중립": 0}
    old_total, new_total = 0, 0
    clockwise_Map = {"소외": "개선", "개선": "주도", "주도": "약화", "약화": "소외"}
    cw_hits, cw_total = 0, 0

    for b in prices.columns:
        vol = float(prices[b].pct_change().std())
        ox, oy = old_res[b]["x_series"], old_res[b]["y_series"]
        nx, ny = new_res[b]["x_series"], new_res[b]["y_series"]

        o_df = pd.concat([ox.rename("x"), oy.rename("y")], axis=1).dropna()
        n_df = pd.concat([nx.rename("x"), ny.rename("y")], axis=1).dropna()

        if len(o_df) > 20:
            old_corrs.append(float(o_df["x"].corr(o_df["y"])))
            old_abs_x.append(float(o_df["x"].abs().mean()))
            basket_vols.append(vol)
            for _, row in o_df.iterrows():
                q = classify_rrg_quadrant(row["x"], row["y"], neutral_thresh=0.0)
                if q in old_q_counts:
                    old_q_counts[q] += 1
                    old_total += 1

        if len(n_df) > 20:
            new_corrs.append(float(n_df["x"].corr(n_df["y"])))
            new_abs_x.append(float(n_df["x"].abs().mean()))
            prev_q = None
            for _, row in n_df.iterrows():
                q = classify_rrg_quadrant(row["x"], row["y"])
                if q in new_q_counts:
                    new_q_counts[q] += 1
                    new_total += 1
                if q != "중립" and prev_q is not None and prev_q != q:
                    cw_total += 1
                    if clockwise_Map.get(prev_q) == q:
                        cw_hits += 1
                if q != "중립":
                    prev_q = q

    old_xy_corr = float(np.median(old_corrs)) if old_corrs else 1.0
    new_xy_corr = float(np.median(new_corrs)) if new_corrs else 1.0
    old_vol_bias = float( pd.Series(old_abs_x).corr(pd.Series(basket_vols)) ) if len(basket_vols) > 2 else 1.0
    new_vol_bias = float( pd.Series(new_abs_x).corr(pd.Series(basket_vols)) ) if len(basket_vols) > 2 else 0.0

    new_ii_ratio = (new_q_counts["개선"] / max(1, new_total)) * 100.0
    new_iv_ratio = (new_q_counts["약화"] / max(1, new_total)) * 100.0

    passed = (
        (new_xy_corr < old_xy_corr) and
        (new_ii_ratio >= 10.0 and new_iv_ratio >= 10.0) and
        (abs(new_vol_bias) < abs(old_vol_bias) and abs(new_vol_bias) < 0.25)
    )

    return {
        "old_xy_corr_median": round(old_xy_corr, 3),
        "new_xy_corr_median": round(new_xy_corr, 3),
        "old_quadrant_pct": {k: round(v / max(1, old_total) * 100, 1) for k, v in old_q_counts.items()},
        "new_quadrant_pct": {k: round(v / max(1, new_total) * 100, 1) for k, v in new_q_counts.items()},
        "old_vol_bias_corr": round(old_vol_bias, 3),
        "new_vol_bias_corr": round(new_vol_bias, 3),
        "clockwise_transition_pct": round(cw_hits / max(1, cw_total) * 100, 1),
        "passed": bool(passed)
    }


def determine_basket_grade(quadrant: str, revision_positive: bool, flow_positive: bool) -> str:
    """
    기존 등급 판정 규칙 (변경 금지 원칙 준수: 사분면 입력만 받아 동일 로직 수행)
    """
    if quadrant == "데이터 없음":
        return "데이터 없음"
    if quadrant == "중립":
        return "중립 관망"
    if quadrant == "주도":
        if revision_positive and flow_positive:
            return "실물 확인 주도"
        elif revision_positive or flow_positive:
            return "주도 (수급·실적 부분확인)"
        return "과열 / 다이버전스 경계"
    if quadrant == "개선":
        if revision_positive or flow_positive:
            return "차기 점화"
        return "기술적 반등"
    if quadrant == "약화":
        if revision_positive and flow_positive:
            return "건전한 눌림목"
        return "모멘텀 둔화 경계"
    # 소외
    if revision_positive and flow_positive:
        return "바닥권 역발상 축적"
    return "관망 / 소외"


def evaluate_basket_grades(
    prices: pd.DataFrame,
    bench_spy: pd.Series,
    fundamentals_meta: dict,
    as_of: str = "2026-11-08",
    write_log: bool = True
) -> dict:
    """
    3단계: daily 새 사분면으로 등급 판정 + weekly 구조 사분면(`structural_quadrant`) 부착
    + 최근 20거래일 `logs/rrg_compare.csv` 기록
    """
    daily_rrg = rrg_coordinates(prices, bench_spy, clock="daily")
    weekly_rrg = rrg_coordinates(prices, bench_spy, clock="weekly")
    old_rrg = legacy_rrg_coordinates(prices, bench_spy)

    basket_objects = {}
    grade_diff_report = []

    for b in prices.columns:
        d_info = daily_rrg[b]
        w_info = weekly_rrg[b]
        o_info = old_rrg[b]
        meta = fundamentals_meta.get(b, {})
        rev_pos = bool(meta.get("revision_positive", False))
        flw_pos = bool(meta.get("flow_positive", False))

        # X·Y가 NaN인 바스켓은 등급 부여 차단
        if np.isnan(d_info["x"]) or np.isnan(d_info["y"]):
            new_grade = "데이터 없음"
        else:
            new_grade = determine_basket_grade(d_info["quadrant"], rev_pos, flw_pos)

        old_grade = determine_basket_grade(o_info["quadrant"], rev_pos, flw_pos) if not np.isnan(o_info.get("x", np.nan)) else "데이터 없음"

        basket_objects[b] = {
            "basket": b,
            "name": meta.get("name", b),
            "x": d_info["x"],
            "y": d_info["y"],
            "quadrant": d_info["quadrant"],
            "grade": new_grade,
            "structural_quadrant": w_info["quadrant"],
            "old_x": o_info.get("x", np.nan),
            "old_y": o_info.get("y", np.nan),
            "old_quadrant": o_info["quadrant"],
            "old_grade": old_grade,
            "tail_x": d_info["tail_x"],
            "tail_y": d_info["tail_y"],
            "move_4w_x": d_info["move_4w_x"],
            "move_4w_y": d_info["move_4w_y"],
            "short_sample": d_info["short_sample"],
            "obs_count": d_info["obs_count"],
            "revision_positive": rev_pos,
            "flow_positive": flw_pos,
            "as_of": as_of,
            "source": meta.get("source", "Market RRG Engine v2.0")
        }

        if old_grade != new_grade:
            grade_diff_report.append({
                "basket": b,
                "name": meta.get("name", b),
                "old_quadrant": o_info["quadrant"],
                "old_grade": old_grade,
                "new_quadrant": d_info["quadrant"],
                "new_grade": new_grade,
                "structural_quadrant": w_info["quadrant"]
            })

    # 전환 기간 20거래일 로그 저장 (logs/rrg_compare.csv)
    if write_log:
        os.makedirs(os.path.dirname(RRG_CONFIG["COMPARE_LOG_PATH"]), exist_ok=True)
        rows = []
        recent_dates = prices.index[-RRG_CONFIG["COMPARE_WINDOW_DAYS"]:]
        for dt in recent_dates:
            dt_str = dt.strftime("%Y-%m-%d") if hasattr(dt, "strftime") else str(dt)
            for b in prices.columns:
                meta = fundamentals_meta.get(b, {})
                rev_pos = bool(meta.get("revision_positive", False))
                flw_pos = bool(meta.get("flow_positive", False))
                ox_s, oy_s = old_rrg[b]["x_series"], old_rrg[b]["y_series"]
                nx_s, ny_s = daily_rrg[b]["x_series"], daily_rrg[b]["y_series"]

                ox_v = float(ox_s.loc[dt]) if dt in ox_s.index and not np.isnan(ox_s.loc[dt]) else np.nan
                oy_v = float(oy_s.loc[dt]) if dt in oy_s.index and not np.isnan(oy_s.loc[dt]) else np.nan
                nx_v = float(nx_s.loc[dt]) if dt in nx_s.index and not np.isnan(nx_s.loc[dt]) else np.nan
                ny_v = float(ny_s.loc[dt]) if dt in ny_s.index and not np.isnan(ny_s.loc[dt]) else np.nan

                oq = classify_rrg_quadrant(ox_v, oy_v, neutral_thresh=0.0)
                nq = classify_rrg_quadrant(nx_v, ny_v)
                og = determine_basket_grade(oq, rev_pos, flw_pos)
                ng = determine_basket_grade(nq, rev_pos, flw_pos)

                rows.append({
                    "date": dt_str, "basket": b,
                    "old_x": round(ox_v, 3) if not np.isnan(ox_v) else np.nan,
                    "old_y": round(oy_v, 3) if not np.isnan(oy_v) else np.nan,
                    "old_quadrant": oq, "old_grade": og,
                    "new_x": round(nx_v, 3) if not np.isnan(nx_v) else np.nan,
                    "new_y": round(ny_v, 3) if not np.isnan(ny_v) else np.nan,
                    "new_quadrant": nq, "new_grade": ng
                })
        pd.DataFrame(rows).to_csv(RRG_CONFIG["COMPARE_LOG_PATH"], index=False, encoding="utf-8-sig")

    # 주도 Top 3 및 차기 점화 Top 3 카드 선정 (새 사분면 기준)
    valid_items = [v for v in basket_objects.values() if v["grade"] != "데이터 없음"]
    leading_candidates = [v for v in valid_items if v["quadrant"] == "주도"]
    leading_top3 = sorted(leading_candidates, key=lambda item: (item["x"] + item["y"]), reverse=True)[:3]

    # 다음 후보: 개선 사분면 중 ① 최근 4주 X가 오르는 중(주도 쪽으로 이동) ② 주봉 구조가 '소외'가 아닌 섹터만
    ign_cfg = RRG_CONFIG["IGNITE_FILTER"]
    igniting_candidates, igniting_excluded = [], []
    for v in valid_items:
        if v["quadrant"] != "개선":
            continue
        reasons = []
        mx = v.get("move_4w_x")
        if mx is None or np.isnan(mx) or mx <= ign_cfg["MIN_MOVE_4W_X"]:
            reasons.append("4주 상대강도 하락")
        if v.get("structural_quadrant") in ign_cfg["EXCLUDE_STRUCTURAL"]:
            reasons.append(f"주봉 {v.get('structural_quadrant')}")
        if reasons:
            igniting_excluded.append({"basket": v["basket"], "name": v.get("name", v["basket"]),
                                      "move_4w_x": mx, "structural_quadrant": v.get("structural_quadrant"),
                                      "reasons": reasons})
        else:
            igniting_candidates.append(v)
    igniting_top3 = sorted(igniting_candidates, key=lambda item: item["y"], reverse=True)[:3]

    return {
        "baskets": basket_objects,
        "leading_top3": leading_top3,
        "igniting_top3": igniting_top3,
        "igniting_excluded": igniting_excluded,
        "grade_diff_report": grade_diff_report
    }


# ------------------------------------------------------------------------------
# 4. 8방향 라벨 배치 및 바운딩 박스 충돌 방지: place_labels(points, fig_size)
# ------------------------------------------------------------------------------
def place_labels(points: list[dict], fig_size: tuple[int, int] = (700, 540)) -> dict:
    """
    점들의 화면 픽셀 좌표를 계산해 8방향 후보 중 겹치지 않는 위치를 배정.
    - 원점에서 먼 순서대로 우선 배치
    - 8방향 모두 겹치면 짧은 지시선(leader line)과 함께 외곽으로 밀어냄
    - 그래도 겹치면 라벨을 숨기고 hidden_count 증가
    """
    width, height = fig_size
    lim = RRG_CONFIG["AXIS_LIMIT"]
    pad_x, pad_y = 55, 45
    plot_w = max(200, width - 2 * pad_x)
    plot_h = max(200, height - 2 * pad_y)

    def to_px(x_val: float, y_val: float) -> tuple[float, float]:
        cx = np.clip(x_val, -lim, lim)
        cy = np.clip(y_val, -lim, lim)
        px = pad_x + ((cx + lim) / (2 * lim)) * plot_w
        py = pad_y + ((lim - cy) / (2 * lim)) * plot_h
        return float(px), float(py)

    def boxes_overlap(b1: tuple[float, float, float, float], b2: tuple[float, float, float, float]) -> bool:
        return not (b1[2] <= b2[0] or b1[0] >= b2[2] or b1[3] <= b2[1] or b1[1] >= b2[3])

    # 원점에서 먼 순서대로 정렬
    sorted_pts = sorted(
        [p for p in points if not (np.isnan(p.get("x", np.nan)) or np.isnan(p.get("y", np.nan)))],
        key=lambda item: math.hypot(item["x"], item["y"]),
        reverse=True
    )

    occupied_boxes = []
    # 모든 점 자체의 마커 영역을 먼저 점유 박스로 등록 (반경 7px)
    pt_px_map = {}
    for p in sorted_pts:
        px, py = to_px(p["x"], p["y"])
        pt_px_map[p["basket"]] = (px, py)
        occupied_boxes.append((px - 7, py - 7, px + 7, py + 7))

    # 8방향 오프셋 (dx, dy) 픽셀 단위
    directions_8 = [
        (0, -18), (0, 18), (28, 0), (-28, 0),
        (24, -15), (-24, -15), (24, 15), (-24, 15)
    ]
    # 지시선(Leader line) 확장 방향
    leader_offsets = [
        (46, -28), (-46, -28), (46, 28), (-46, 28),
        (0, -36), (0, 36), (56, 0), (-56, 0)
    ]

    placements = {}
    hidden_count = 0

    for p in sorted_pts:
        b_code = p["basket"]
        px, py = pt_px_map[b_code]
        lbl_w = max(34, len(str(b_code)) * 7 + 8)
        lbl_h = 16

        placed = False
        # 1차: 8방향 근접 탐색
        for dx, dy in directions_8:
            cx, cy = px + dx, py + dy
            cand_box = (cx - lbl_w / 2, cy - lbl_h / 2, cx + lbl_w / 2, cy + lbl_h / 2)
            if cand_box[0] < 5 or cand_box[2] > width - 5 or cand_box[1] < 5 or cand_box[3] > height - 5:
                continue
            if not any(boxes_overlap(cand_box, ob) for ob in occupied_boxes):
                occupied_boxes.append(cand_box)
                placements[b_code] = {
                    "visible": True, "use_leader": False,
                    "dx": dx, "dy": dy, "bbox": cand_box
                }
                placed = True
                break

        if placed:
            continue

        # 2차: 짧은 지시선(Leader line) 외곽 배치
        for dx, dy in leader_offsets:
            cx, cy = px + dx, py + dy
            cand_box = (cx - lbl_w / 2, cy - lbl_h / 2, cx + lbl_w / 2, cy + lbl_h / 2)
            if cand_box[0] < 5 or cand_box[2] > width - 5 or cand_box[1] < 5 or cand_box[3] > height - 5:
                continue
            if not any(boxes_overlap(cand_box, ob) for ob in occupied_boxes):
                occupied_boxes.append(cand_box)
                placements[b_code] = {
                    "visible": True, "use_leader": True,
                    "dx": dx, "dy": dy, "bbox": cand_box
                }
                placed = True
                break

        if not placed:
            # 3차: 공간 부족 시 숨김 처리 (툴팁 전용)
            hidden_count += 1
            placements[b_code] = {
                "visible": False, "use_leader": False,
                "dx": 0, "dy": 0, "bbox": None
            }

    return {"placements": placements, "hidden_count": hidden_count}


# ------------------------------------------------------------------------------
# 6. 5단계 점검: run_consistency_checks() RRG & 게이트 무결성 검증
# ------------------------------------------------------------------------------
def check_rrg_and_gate_consistency(
    chart_baskets: dict,
    card_baskets: list[dict],
    radar_signals: dict
) -> list[str]:
    """
    위반 점검 3가지:
    1) 카드에 표시된 사분면과 차트의 사분면이 다름
    2) 모듈 4 "[바스켓] … 동조" 배지가 옛 등급(old_grade)을 참조함
    3) X·Y가 NaN인 바스켓이 유효 등급을 받음
    """
    violations = []

    # 점검 1: 카드 사분면 vs 차트 사분면 일치 여부
    for card in card_baskets:
        b = card.get("basket")
        if b in chart_baskets:
            if card.get("quadrant") != chart_baskets[b].get("quadrant"):
                violations.append(
                    f"RRG_QUADRANT_MISMATCH: {b} 카드 사분면({card.get('quadrant')}) != 차트 사분면({chart_baskets[b].get('quadrant')})"
                )

    # 점검 2: 모듈 4 바스켓 동조 배지가 새 등급과 일치하는지 확인
    for col_key, sig_list in radar_signals.items():
        for sig in sig_list:
            b = sig.get("basket")
            badge_str = sig.get("basket_sync_badge", "")
            if b and b in chart_baskets:
                new_g = chart_baskets[b].get("grade", "")
                old_g = chart_baskets[b].get("old_grade", "")
                if new_g not in badge_str or (old_g != new_g and old_g in badge_str):
                    violations.append(
                        f"STALE_BASKET_BADGE: {sig.get('ticker')} 배지('{badge_str}')가 새 등급('{new_g}')과 불일치"
                    )

    # 점검 3: X·Y가 NaN인 바스켓이 등급을 받았는지 확인
    for b, info in chart_baskets.items():
        x_v, y_v = info.get("x", np.nan), info.get("y", np.nan)
        if (x_v is None or y_v is None or np.isnan(x_v) or np.isnan(y_v)):
            if info.get("grade") not in [None, "데이터 없음"]:
                violations.append(
                    f"NAN_COORDINATE_GRADED: {b}의 좌표가 NaN인데 '{info.get('grade')}' 등급이 부여됨"
                )

    return violations


# ------------------------------------------------------------------------------
# 7. 17대 내러티브 바스켓 3년치 표준 시계열 생성 및 SAMPLE_DATA 단일 출처 등록
# ------------------------------------------------------------------------------
def _build_default_rrg_sample_dataset() -> dict:
    np.random.seed(42)
    dates = pd.bdate_range(end="2026-11-06", periods=756)  # 3년 영업일
    n_days = len(dates)
    t = np.linspace(0, 6 * np.pi, n_days)

    # SPY 및 RSP 기준 지수 생성
    spy_ret = 0.0004 + 0.008 * np.random.randn(n_days)
    rsp_ret = 0.00035 + 0.0075 * np.random.randn(n_days)
    spy_prices = pd.Series(400.0 * np.cumprod(1.0 + spy_ret), index=dates, name="SPY")
    rsp_prices = pd.Series(150.0 * np.cumprod(1.0 + rsp_ret), index=dates, name="RSP")

    # 17대 내러티브 바스켓 메타 정의 (고변동~저변동, 위상차 부여로 4분면 고른 분포 보장)
    baskets_spec = [
        ("SMH", "반도체", 0.022, 0.0, True, True),
        ("IBIT", "비트코인 현물", 0.032, 0.6, True, True),
        ("IGV", "소프트웨어", 0.016, 1.1, True, True),
        ("XLI", "산업재·인프라", 0.011, 1.6, True, False),
        ("XLF", "금융", 0.012, 2.1, True, True),
        ("XLU", "유틸리티·전력망", 0.007, 4.9, True, True),
        ("XLP", "필수소비재", 0.006, 3.2, False, False),
        ("XLV", "헬스케어", 0.007, 3.8, False, True),
        ("XLB", "소재", 0.011, 2.7, False, False),
        ("XLRE", "리츠·부동산", 0.012, 1.9, False, False),
        ("XLY", "경기소비재", 0.014, 2.9, False, True),
        ("XLE", "에너지", 0.017, 4.2, True, False),
        ("XLC", "커뮤니케이션", 0.013, 0.4, True, True),
        ("ITA", "방위산업", 0.011, 5.4, True, True),
        ("TAN", "클린에너지", 0.024, 3.5, False, False),
        ("URA", "원자력·우라늄", 0.025, 5.1, True, True),
        ("KWEB", "중국 테크", 0.026, 2.3, False, False),
    ]

    price_dict = {}
    meta_dict = {}
    for code, kor_name, vol, phase, rev_ok, flw_ok in baskets_spec:
        # 사인파 순환 주기 + 고유 변동성 결합 (자연스러운 RRG 시계방향 회전 유도)
        cycle_alpha = 0.0012 * np.sin(t + phase) + 0.0006 * np.cos(2 * t + phase)
        noise = (vol * 0.35) * np.random.randn(n_days)
        rs_path = np.cumprod(1.0 + cycle_alpha + noise)
        price_dict[code] = spy_prices.values * rs_path
        meta_dict[code] = {
            "name": kor_name, "revision_positive": rev_ok, "flow_positive": flw_ok,
            "as_of": "2026-11-08", "source": "Market RRG Engine v2.0"
        }

    prices_df = pd.DataFrame(price_dict, index=dates)
    eval_res = evaluate_basket_grades(prices_df, spy_prices, meta_dict, as_of="2026-11-08", write_log=True)
    diag_res = diagnose_rrg_axes(prices_df, spy_prices)

    return {
        "prices_df": prices_df,
        "bench_spy": spy_prices,
        "bench_rsp": rsp_prices,
        "meta": meta_dict,
        "evaluation": eval_res,
        "diagnostics": diag_res
    }


if "SAMPLE_DATA" not in globals():
    SAMPLE_DATA = {}

# [2026-11-08] 난수·사인파로 만든 합성 시계열은 화면에 쓰지 않는다(테스트 전용 함수로만 남김).
# 실제 4분면은 app.py가 내려받은 실제 종가로 build_real_rrg_bundle()을 호출해 만든다.
SAMPLE_DATA["rrg_bundle"] = {}

SAMPLE_DATA.setdefault("macro_gravity", {
    "US": {
        "regime": "DEEP_BEAR", "vix_percentile": 95.0, "vix_value": 36.2,
        "macro_phase": "CREDIT_CRUNCH", "as_of": "2026-11-08", "source": "CBOE / FRED"
    },
    "KR": {
        "regime": "CORRECTION", "vix_percentile": 78.0, "vix_value": 26.1,
        "macro_phase": "SLOWDOWN", "as_of": "2026-11-08", "source": "KRX / ECOS"
    }
})

_raw_radar = {
    "col1_transitions": [
        {"ticker": "NVDA", "basket": "SMH", "market": "US", "signal_type": "T_ENTRY", "tranche_stage": 0, "action_text": "T ST-3 A등급 진입", "suggested_weight": 100, "is_bnf_oversold": False, "note": "20일선 돌파 거래량 2.1배", "as_of": "2026-11-08", "source": "Track 1/3 Synthesis"},
        {"ticker": "005930", "basket": "SMH", "market": "KR", "signal_type": "V_BUY", "tranche_stage": 0, "action_text": "V 0차 즉시 매수", "suggested_weight": 100, "is_bnf_oversold": False, "note": "PBR 밴드 최하단 지지", "as_of": "2026-11-08", "source": "Track 1/3 Synthesis"},
        {"ticker": "000660", "basket": "SMH", "market": "KR", "signal_type": "V_BUY", "tranche_stage": 1, "action_text": "V 1차 분할 매수", "suggested_weight": 50, "is_bnf_oversold": False, "note": "1차 매수선 도달 (이월 주문)", "as_of": "2026-11-08", "source": "Track 3 Valuation"}
    ],
    "col2_watch": [
        {"ticker": "TSLA", "basket": "XLY", "market": "US", "signal_type": "T_WATCH", "tranche_stage": 0, "action_text": "ST-1 셋업 사정권 (0.8% 내)", "suggested_weight": 50, "is_bnf_oversold": False, "note": "변동성 수축 후 상단 근접", "as_of": "2026-11-08", "source": "Track 1 Engine"},
        {"ticker": "IBIT", "basket": "IBIT", "market": "CRYPTO", "signal_type": "T_WATCH", "tranche_stage": 0, "action_text": "ST-2 눌림목 대기 (1.1% 내)", "suggested_weight": 50, "is_bnf_oversold": False, "note": "ETF 발행좌수 순유입 유지", "as_of": "2026-11-08", "source": "Track 1 Engine"}
    ],
    "col3_risk_and_special": [
        {"ticker": "AAPL", "basket": "XLC", "market": "US", "signal_type": "RISK_DEFENSE", "tranche_stage": 0, "action_text": "손절 및 비중 50% 축소", "suggested_weight": 100, "is_bnf_oversold": False, "note": "50일선 이탈 및 S-REG 악화", "as_of": "2026-11-08", "source": "Track 1 Risk Gate"},
        {"ticker": "INTC", "basket": "SMH", "market": "US", "signal_type": "V_BUY", "tranche_stage": 0, "action_text": "극단 투매 BNF 매수", "suggested_weight": 100, "is_bnf_oversold": True, "note": "F10 적정가 괴리 -38% 과매도", "as_of": "2026-11-08", "source": "Track 3 Valuation"},
        {"ticker": "SCHD", "basket": "XLF", "market": "US", "signal_type": "V_BUY", "tranche_stage": 0, "action_text": "정기 코어 적립 매수", "suggested_weight": 100, "is_bnf_oversold": False, "note": "배당 코어 패시브 적립", "as_of": "2026-11-08", "source": "Core Portfolio Rule"}
    ]
}
# [2026-11-08] 레이더 신호는 아직 실제 신호 엔진과 연결되지 않은 예시 데이터다.
# 화면에서 진짜 엔진 결과처럼 보이지 않도록 기준일을 지우고 출처를 SAMPLE_DATA로 표시한다.
for _col_list in _raw_radar.values():
    for _sig in _col_list:
        _sig["orig_source"] = _sig.get("source", "")
        _sig["as_of"] = None
        _sig["source"] = "SAMPLE_DATA (예시 신호)"
        _sig["is_sample"] = True
SAMPLE_DATA["radar_signals"] = _raw_radar


# ==============================================================================
# [2026-11-08] 실제 종가 기반 4분면(RRG) 번들 — 합성 데이터 대체
# ==============================================================================
def build_real_rrg_bundle(etf_data_map: dict, narrative_baskets: list = None, write_log: bool = True) -> dict:
    """
    app.py가 yfinance로 내려받은 실제 종가로 rrg_coordinates / evaluate_basket_grades를 돌린다.
    - 실적 리비전이 예시 데이터(SAMPLE)이거나 없으면 '실적 데이터 없음'으로 표시하고 등급 판정에 쓰지 않는다.
    - 수급은 실제 CMF(20) > 0 여부를 쓴다.
    """
    spec_map = {s["ticker"]: s for s in NARRATIVE_BASKETS_SPEC}
    spy_df = (etf_data_map or {}).get("SPY")
    if spy_df is None or "Close" not in spy_df or len(spy_df) < 80:
        return {"available": False, "reason": "SPY 종가 데이터 없음"}

    def _close(df_x):
        s = pd.to_numeric(df_x["Close"], errors="coerce")
        s.index = pd.to_datetime(s.index).tz_localize(None) if getattr(s.index, "tz", None) is not None else pd.to_datetime(s.index)
        return s.sort_index()

    closes = {}
    for tk in spec_map:
        df_x = etf_data_map.get(tk)
        if df_x is not None and "Close" in df_x and len(df_x) >= 80:
            closes[tk] = _close(df_x)
    if not closes:
        return {"available": False, "reason": "바스켓 종가 데이터 없음"}

    prices = pd.DataFrame(closes).sort_index()
    spy = _close(spy_df)
    rsp_df = etf_data_map.get("RSP")
    rsp = _close(rsp_df) if (rsp_df is not None and "Close" in rsp_df) else None
    as_of = str(prices.dropna(how="all").index[-1])[:10]

    nb_map = {b.get("ticker"): b for b in (narrative_baskets or [])}
    meta = {}
    for tk in prices.columns:
        nb = nb_map.get(tk, {})
        rev_known = bool(nb) and not nb.get("is_sample_revision", False) and nb.get("fundamental_or_flow_direction", "none") != "none"
        cmf = nb.get("cmf_20")
        meta[tk] = {
            "name": spec_map[tk]["name"],
            "revision_known": rev_known,
            "revision_positive": rev_known and nb.get("fundamental_or_flow_direction") == "up",
            "flow_positive": (cmf is not None) and (not nb.get("is_sample_price", False)) and float(cmf) > 0,
            "as_of": as_of,
            "source": f"yfinance 종가 ({tk})",
        }

    evaluation = evaluate_basket_grades(prices, spy, meta, as_of=as_of, write_log=write_log)

    # 실적 리비전을 모르는 바스켓은 '실적 확인'이 들어간 등급을 받지 않도록 문구를 바꾼다.
    for tk, b in evaluation["baskets"].items():
        if b["grade"] == "데이터 없음" or meta.get(tk, {}).get("revision_known", False):
            continue
        q = b["quadrant"]
        if q in ("주도", "개선", "약화", "소외"):
            flow_txt = "수급 확인" if meta[tk]["flow_positive"] else "수급 약함"
            b["grade"] = f"{q} ({flow_txt} · 실적 데이터 없음)"

    return {
        "available": True,
        "prices_df": prices,
        "bench_spy": spy,
        "bench_rsp": rsp,
        "meta": meta,
        "evaluation": evaluation,
        "as_of": as_of,
    }
