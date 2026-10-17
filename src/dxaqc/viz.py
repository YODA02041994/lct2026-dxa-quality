"""Объяснимость: рисуем на снимке то, по чему принято решение — ориентиры, ось, вердикт по критериям.

Используется API (/api/runs/{id}/overlay/{n}.png) и доп. серией DICOM. Линии и точки — OpenCV, подписи — Pillow со шрифтом
DejaVu Sans (лежит в пакете, src/dxaqc/fonts, свободная лицензия): серия попадает в PACS к русскоязычным врачам, поэтому
все надписи по-русски.
"""
from __future__ import annotations

import math
import os

import cv2
import numpy as np

from .labels import REGION_SPINE

SHORT = {"spine_positioning": "укладка", "spine_axis_tilt": "ось", "spine_artifact": "посторонние предметы",
         "hip_positioning_rotation": "укладка и ротация", "hip_roi_field": "поле сканирования"}
SPINE_ORDER = ["th12", "L1", "L2", "L3", "L4", "L5"]
LM_TITLE = {"th12": "Th12", "crest_l": "гребень", "crest_r": "гребень", "head": "головка", "gt_top": "б. вертел", "gt_lat": "б. вертел",
            "neck_sup": "шейка", "neck_inf": "шейка", "lt_tip": "м. вертел", "lt_up": "", "lt_down": "", "ischium": "седалищный бугор",
            "shaft_top": "диафиз", "shaft_bot": ""}
LEVEL_TITLE = {"repeat": "переснять", "check": "проверить укладку", "ok": "в норме"}
FONT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")
_FONTS: dict = {}


def _font(size: int, bold: bool = False):
    from PIL import ImageFont
    key = (size, bold)
    if key not in _FONTS:
        path = os.path.join(FONT_DIR, "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")
        try:
            _FONTS[key] = ImageFont.truetype(path, size)
        except OSError:                                   # шрифта нет — подписи останутся, но мелким растровым шрифтом
            _FONTS[key] = ImageFont.load_default()
    return _FONTS[key]


def _num(x: float, nd: int = 2) -> str:
    return f"{float(x):.{nd}f}".replace(".", ",")


def _draw_texts(bgr: np.ndarray, items: list) -> np.ndarray:
    """items: (x, y_baseline, текст, BGR-цвет, кегль px, жирный). Рисуем Pillow поверх готовой картинки."""
    if not items:
        return bgr
    from PIL import Image, ImageDraw
    im = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(im)
    for x, y, text, col, size, bold in items:
        if text:
            d.text((int(x), int(y) - size), text, font=_font(size, bold), fill=(int(col[2]), int(col[1]), int(col[0])))
    return cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)


def to_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img
    lo, hi = np.percentile(img, [0.5, 99.5])
    return np.clip((img.astype(np.float32) - lo) / max(1e-6, hi - lo) * 255, 0, 255).astype(np.uint8)


def draw_overlay(img: np.ndarray, region: str, details: dict, verdict: dict, thresholds: dict | None = None,
                 scale: int = 3) -> np.ndarray:
    """img — исходные пиксели; details — как в колонке details (landmarks: {name: [x, y, conf]}, probs, features, level, scoliosis);
    verdict — {"quality_class", "violation_type", "quality_prob"}. Возвращает BGR-картинку в масштабе scale."""
    g = to_uint8(img)
    h, w = g.shape
    can = cv2.cvtColor(cv2.resize(g, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC), cv2.COLOR_GRAY2BGR)
    lm = details.get("landmarks", {}) or {}
    probs = details.get("probs", {}) or {}
    thresholds = thresholds or {}
    ok_col, bad_col, warn_col, pt_col = (90, 200, 90), (70, 80, 230), (40, 180, 240), (60, 220, 255)
    texts: list = []
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
    # предложенная разметка областей измерения: тела L1–L4, шейка бедра
    for name, poly in (details.get("roi") or {}).items():
        pts = np.array([[int(round(x * scale)), int(round(y * scale))] for x, y in poly], np.int32)
        cv2.polylines(can, [pts], True, (110, 220, 110), 1, cv2.LINE_AA)
        texts.append((int(pts[:, 0].max()) + 4, int(pts[:, 1].mean()) + 5, f"ROI {'шейка' if name == 'neck' else name}", (110, 220, 110), 13, False))
    # предметы (карта object-CXR): рамки самых ярких пятен
    for b in details.get("objects", []) or []:
        x0, y0 = int(b["x"] * scale), int(b["y"] * scale)
        cv2.rectangle(can, (x0, y0), (x0 + int(b["w"] * scale), y0 + int(b["h"] * scale)), (0, 0, 255), 2)
        texts.append((x0, max(14, y0 - 4), f"предмет {_num(b['score'])}", (0, 0, 255), 13, False))
    # ориентиры
    for k, v in lm.items():
        p = P(k)
        conf = v[2] if len(v) > 2 else 1.0
        cv2.circle(can, p, 4, pt_col if conf >= 0.5 else (0, 140, 255), -1, cv2.LINE_AA)
        texts.append((p[0] + 6, p[1] - 3, LM_TITLE.get(k, k), (255, 255, 255), 12, False))
    # вердикт — сверху: уровень, критерии с порогами, сколиоз отдельной строкой
    cls = int(verdict.get("quality_class", 0))
    level = str(details.get("level") or verdict.get("level") or ("repeat" if cls else "ok"))
    scol = details.get("scoliosis") or {}
    marked = scol.get("grade") == "marked"
    rows = len(probs) + (1 if marked else 0)
    top_h = 30 + 19 * rows
    cv2.rectangle(can, (0, 0), (w * scale, top_h), (30, 30, 30), -1)
    head_col = bad_col if level == "repeat" else warn_col if level == "check" else ok_col
    texts.append((8, 22, f"{'НАРУШЕНИЕ' if cls else 'НОРМА'} · {LEVEL_TITLE.get(level, level)} · вероятность {_num(verdict.get('quality_prob', 0))}",
                  head_col, 16, True))
    y = 22
    for c, p in probs.items():
        y += 19
        thr = thresholds.get(c, 0.5)
        col = bad_col if p >= thr else (200, 200, 200)
        bar = int(60 * min(1.0, float(p)))
        cv2.rectangle(can, (w * scale - 72, y - 11), (w * scale - 72 + bar, y - 3), col, -1)
        cv2.rectangle(can, (w * scale - 72, y - 11), (w * scale - 12, y - 3), (120, 120, 120), 1)
        tx = int(w * scale - 72 + 60 * min(1.0, float(thr)))
        cv2.line(can, (tx, y - 13), (tx, y - 1), (255, 255, 255), 1)          # отметка порога
        texts.append((8, y, f"{SHORT.get(c, c)}: {_num(p)} (порог {_num(thr)})", col, 13, False))
    if marked:
        y += 19
        texts.append((8, y, f"сколиотическая деформация, {_num(scol.get('bend_deg') or 0, 1)}° — не нарушение укладки"
                      + ("; ось не бракуется" if scol.get("axis_damped") else ""), (255, 210, 120), 13, False))
    # протокол по критериям — снизу
    txt = str(details.get("comment") or "")
    if txt:                                          # протокол по критериям, по словам в строки шириной кадра
        lines, cur, limit = [], "", max(40, int(w * scale / 7.2))
        for word in txt.split(" "):
            if len(cur) + len(word) + 1 > limit and cur:
                lines.append(cur); cur = word
            else:
                cur = (cur + " " + word).strip()
        lines = (lines + [cur])[:3]
        band = 8 + 17 * len(lines)
        cv2.rectangle(can, (0, h * scale - band), (w * scale, h * scale), (30, 30, 30), -1)
        for n, line in enumerate(lines):
            texts.append((8, h * scale - band + 18 + 17 * n, line, (225, 225, 225), 12, False))
    return _draw_texts(can, texts)


def encode_png(bgr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError("png encode failed")
    return buf.tobytes()


def angle_deg(p: tuple, q: tuple) -> float:
    return abs(math.degrees(math.atan2(q[0] - p[0], q[1] - p[1])))
