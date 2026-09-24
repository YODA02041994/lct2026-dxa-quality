"""Объяснимость: рисуем на снимке то, по чему принято решение — ориентиры, ось, вердикт по критериям.

Используется API (/api/runs/{id}/overlay/{n}.png) и скриптом экспорта доп. серии. Только OpenCV, без шрифтов
с кириллицей: подписи критериев — латиницей/цифрами, официальный текст нарушения показывается в интерфейсе.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from .labels import REGION_SPINE

SHORT = {"spine_positioning": "positioning", "spine_axis_tilt": "axis", "spine_artifact": "artifact",
         "hip_positioning_rotation": "positioning/rotation", "hip_roi_field": "ROI field"}
SPINE_ORDER = ["th12", "L1", "L2", "L3", "L4", "L5"]


def to_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img
    lo, hi = np.percentile(img, [0.5, 99.5])
    return np.clip((img.astype(np.float32) - lo) / max(1e-6, hi - lo) * 255, 0, 255).astype(np.uint8)


def draw_overlay(img: np.ndarray, region: str, details: dict, verdict: dict, thresholds: dict | None = None,
                 scale: int = 3) -> np.ndarray:
    """img — исходные пиксели; details — как в колонке details (landmarks: {name: [x, y, conf]}, probs, features);
    verdict — {"quality_class", "violation_type", "quality_prob"}. Возвращает BGR-картинку в масштабе scale."""
    g = to_uint8(img)
    h, w = g.shape
    can = cv2.cvtColor(cv2.resize(g, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC), cv2.COLOR_GRAY2BGR)
    lm = details.get("landmarks", {}) or {}
    probs = details.get("probs", {}) or {}
    thresholds = thresholds or {}
    ok_col, bad_col, pt_col = (90, 200, 90), (70, 80, 230), (60, 220, 255)
    P = lambda k: (int(round(lm[k][0] * scale)), int(round(lm[k][1] * scale))) if k in lm else None  # noqa: E731

    # линии: ось позвоночника по центрам тел / ось диафиза бедра
    if region == REGION_SPINE:
        pts = [P(k) for k in SPINE_ORDER[1:] if P(k)]
        if len(pts) >= 2:
            xs, ys = np.array([p[0] for p in pts], float), np.array([p[1] for p in pts], float)
            k, b = np.polyfit(ys, xs, 1)
            y0, y1 = 0, h * scale
            cv2.line(can, (int(k * y0 + b), y0), (int(k * y1 + b), y1), (255, 200, 80), 2, cv2.LINE_AA)
            cv2.line(can, (int(np.mean(xs)), y0), (int(np.mean(xs)), y1), (160, 160, 160), 1, cv2.LINE_AA)
    else:
        if P("shaft_top") and P("shaft_bot"):
            cv2.line(can, P("shaft_top"), P("shaft_bot"), (255, 200, 80), 2, cv2.LINE_AA)
        if P("head") and P("neck_sup") and P("neck_inf"):
            mid = ((P("neck_sup")[0] + P("neck_inf")[0]) // 2, (P("neck_sup")[1] + P("neck_inf")[1]) // 2)
            cv2.line(can, P("head"), mid, (255, 200, 80), 1, cv2.LINE_AA)
    # ориентиры
    for k, v in lm.items():
        p = P(k)
        conf = v[2] if len(v) > 2 else 1.0
        cv2.circle(can, p, 4, pt_col if conf >= 0.5 else (0, 140, 255), -1, cv2.LINE_AA)
        cv2.putText(can, k, (p[0] + 5, p[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
    # вердикт по критериям — сверху
    y = 18
    cls = int(verdict.get("quality_class", 0))
    head = f"{'VIOLATION' if cls else 'OK'}  p={float(verdict.get('quality_prob', 0)):.2f}"
    cv2.rectangle(can, (0, 0), (w * scale, 22 + 16 * len(probs)), (30, 30, 30), -1)
    cv2.putText(can, head, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bad_col if cls else ok_col, 1, cv2.LINE_AA)
    for c, p in probs.items():
        y += 16
        thr = thresholds.get(c, 0.5)
        col = bad_col if p >= thr else (200, 200, 200)
        bar = int(60 * min(1.0, float(p)))
        cv2.rectangle(can, (w * scale - 70, y - 9), (w * scale - 70 + bar, y - 1), col, -1)
        cv2.rectangle(can, (w * scale - 70, y - 9), (w * scale - 10, y - 1), (120, 120, 120), 1)
        cv2.putText(can, f"{SHORT.get(c, c)}: {float(p):.2f} (thr {thr:.2f})", (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1, cv2.LINE_AA)
    # ключевые признаки — снизу
    f = details.get("features", {}) or {}
    keys = ["axis_img_deg", "bottom_width", "th_frac40"] if region == REGION_SPINE else ["shaft_img_deg", "min_cm", "frame_h_cm"]
    txt = "  ".join(f"{k}={f[k]}" for k in keys if k in f)
    if txt:
        cv2.rectangle(can, (0, h * scale - 18), (w * scale, h * scale), (30, 30, 30), -1)
        cv2.putText(can, txt, (6, h * scale - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1, cv2.LINE_AA)
    return can


def encode_png(bgr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError("png encode failed")
    return buf.tobytes()


def angle_deg(p: tuple, q: tuple) -> float:
    return abs(math.degrees(math.atan2(q[0] - p[0], q[1] - p[1])))
