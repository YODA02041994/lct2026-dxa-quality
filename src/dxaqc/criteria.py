"""Критерии качества по ориентирам: признаки → вероятность нарушения по каждому критерию ТЗ.

Идея (docs/05, эксп. 02): критерии ТЗ — геометрия. Из точек (врача или локализатора) считаем понятные
признаки — угол оси, запасы поля в сантиметрах, видимость структур — и поверх них учим маленькую модель
на метках экспертов. Жёстких порогов нет: сколиоз и перелом ломают «5° в лоб», а границу нормы по
малому вертелу задают эксперты, не мы. Каждый признак объясним врачу и выводится на снимок.

Формат точек: {name: {"x", "y", "present"[, "conf"]}} в пикселях исходника.
"""
from __future__ import annotations

import math

import numpy as np

from .labels import HIP_FLAGS, SPINE_FLAGS, pixel_mm

SPINE_FEATURES = ["axis_deg", "axis_ok", "n_levels", "L5_present", "th12_present", "crests", "crest_l", "crest_r",
                  "top_gap_cm", "bottom_gap_cm", "conf_min", "th12_conf", "crest_l_conf", "crest_r_conf", "crests_conf", "axis_img_deg"]
HIP_FEATURES = ["top_cm", "lat_cm", "bottom_cm", "min_cm", "shaft_deg", "neck_shaft_deg", "neck_mm", "head_gt_dy_mm",
                "lt_mm", "lt_present", "ischium_present", "ischium_cut", "n_missing", "narrow_frame", "conf_min",
                "ischium_conf", "lt_conf", "base_conf_min"]

# Какие признаки видит каждый критерий. При 6–36 положительных примерах лишние признаки только шумят,
# поэтому наборы короткие и клинически осмысленные (docs/05, эксп. 02–03).
CRITERION_FEATURES = {
    "spine_positioning": ["crests_conf", "th12_conf", "bottom_gap_cm", "top_gap_cm"],
    "spine_axis_tilt": ["axis_deg", "axis_img_deg", "n_levels"],
    "spine_artifact": ["axis_deg"],                      # по ориентирам не определяется — критерий уходит к CNN
    "hip_positioning_rotation": ["shaft_deg", "ischium_conf", "ischium_cut", "lt_mm", "lt_conf", "n_missing", "base_conf_min"],
    "hip_roi_field": ["min_cm", "top_cm", "lat_cm", "bottom_cm"],
}


def _pt(lm: dict, k: str):
    p = lm.get(k)
    if not p or not p.get("present", True):
        return None
    return (float(p["x"]), float(p["y"]))


def _conf(lm: dict, k: str) -> float:
    """Уверенность, что ориентир есть: у врача 1/0, у сети — высота пика (непрерывно, без порога)."""
    p = lm.get(k)
    if not p:
        return 0.0
    return float(p.get("conf", 1.0 if p.get("present", True) else 0.0))


def _conf_min(lm: dict) -> float:
    cs = [float(p.get("conf", 1.0)) for p in lm.values() if p.get("present", True)]
    return min(cs) if cs else 0.0


def spine_features(lm: dict, h: int, w: int, img: np.ndarray | None = None) -> dict:
    mm = pixel_mm(w)
    levels = [_pt(lm, k) for k in ("L1", "L2", "L3", "L4", "L5")]
    pres = [p for p in levels if p]
    f = {k: 0.0 for k in SPINE_FEATURES}
    if len(pres) >= 2:
        xs, ys = np.array([p[0] for p in pres]), np.array([p[1] for p in pres])
        k = np.polyfit(ys, xs, 1)[0] if ys.ptp() > 1 else 0.0
        f["axis_deg"] = abs(math.degrees(math.atan(k)))
        f["axis_ok"] = 1.0
    f["n_levels"] = float(len(pres))
    f["L5_present"] = float(levels[4] is not None)
    f["th12_present"] = float(_pt(lm, "th12") is not None)
    f["crest_l"], f["crest_r"] = float(_pt(lm, "crest_l") is not None), float(_pt(lm, "crest_r") is not None)
    f["crests"] = f["crest_l"] + f["crest_r"]
    top = _pt(lm, "th12") or (pres[0] if pres else None)
    bot = pres[-1] if pres else None
    f["top_gap_cm"] = top[1] * mm / 10 if top else 0.0
    f["bottom_gap_cm"] = (h - bot[1]) * mm / 10 if bot else 0.0
    f["conf_min"] = _conf_min(lm)
    f["th12_conf"], f["crest_l_conf"], f["crest_r_conf"] = _conf(lm, "th12"), _conf(lm, "crest_l"), _conf(lm, "crest_r")
    f["crests_conf"] = f["crest_l_conf"] + f["crest_r_conf"]
    f["axis_img_deg"] = spine_axis_from_image(img)
    return f


def spine_axis_from_image(img: np.ndarray | None) -> float:
    """Наклон оси позвоночника прямо по картинке, без ориентиров (эксп. 03: AUC 0,80 против меток экспертов).
    Центр масс яркой кости по строкам в центральной трети кадра (позвоночник по центру), без верхних рёбер и
    нижнего таза; прямая — робастно, с отбросом 20 % худших строк. Устойчиво к путанице уровней у локализатора."""
    if img is None:
        return 0.0
    h, w = img.shape
    band = img[int(h * 0.15):int(h * 0.85)].astype(np.float32)
    nz = band[band > 0]
    if nz.size < 100:
        return 0.0
    m = (band >= np.percentile(nz, 70)).astype(np.float32)
    m[:, :int(w * 0.3)] = 0
    m[:, int(w * 0.7):] = 0
    ys, xs = [], []
    for r in range(m.shape[0]):
        row = m[r]
        if row.sum() >= 3:
            ys.append(r)
            xs.append(float((row * np.arange(w)).sum() / row.sum()))
    if len(ys) < 20:
        return 0.0
    ys, xs = np.array(ys, float), np.array(xs, float)
    k = np.polyfit(ys, xs, 1)[0]
    res = xs - (k * ys + np.median(xs - k * ys))
    keep = np.abs(res) < np.percentile(np.abs(res), 80)
    if keep.sum() >= 10:
        k = np.polyfit(ys[keep], xs[keep], 1)[0]
    return abs(math.degrees(math.atan(k)))


def _near_black(img: np.ndarray | None, x: float, y: float, reach: int = 6) -> float:
    """Точка у границы чёрной маски / края кадра — структура обрезана (используется для седалищной кости)."""
    if img is None:
        return 0.0
    h, w = img.shape
    x0, x1 = max(0, int(x) - 4), min(w, int(x) + 5)
    below = img[min(h - 1, int(y) + 2):min(h, int(y) + reach + 1), x0:x1]
    return float((below.size > 0 and (below == 0).mean() > 0.8) or int(y) + reach >= h)


def hip_features(lm: dict, h: int, w: int, side: str | None, img: np.ndarray | None = None) -> dict:
    mm = pixel_mm(w)
    f = {k: 0.0 for k in HIP_FEATURES}
    head, gt, gl = _pt(lm, "head"), _pt(lm, "gt_top"), _pt(lm, "gt_lat")
    ns, ni, lt, lu, ld = _pt(lm, "neck_sup"), _pt(lm, "neck_inf"), _pt(lm, "lt_tip"), _pt(lm, "lt_up"), _pt(lm, "lt_down")
    isch, st, sb = _pt(lm, "ischium"), _pt(lm, "shaft_top"), _pt(lm, "shaft_bot")
    lateral_x = (lambda p: p[0]) if side != "L" else (lambda p: (w - 1) - p[0])
    f["top_cm"] = gt[1] * mm / 10 if gt else 0.0
    f["lat_cm"] = lateral_x(gl) * mm / 10 if gl else 0.0
    low = ld or lt or st
    f["bottom_cm"] = (h - low[1]) * mm / 10 if low else 0.0
    vals = [v for v, p in ((f["top_cm"], gt), (f["lat_cm"], gl), (f["bottom_cm"], low)) if p]
    f["min_cm"] = min(vals) if vals else 0.0
    if st and sb:
        f["shaft_deg"] = abs(math.degrees(math.atan2(sb[0] - st[0], sb[1] - st[1])))
    if ns and ni and head and st and sb:
        mid = ((ns[0] + ni[0]) / 2, (ns[1] + ni[1]) / 2)
        ax = (head[0] - mid[0], head[1] - mid[1])
        sh = (sb[0] - st[0], sb[1] - st[1])
        den = math.hypot(*ax) * math.hypot(*sh)
        if den > 0:
            f["neck_shaft_deg"] = math.degrees(math.acos(max(-1, min(1, (-ax[0] * sh[0] - ax[1] * sh[1]) / den))))
    if ns and ni:
        f["neck_mm"] = math.hypot(ns[0] - ni[0], ns[1] - ni[1]) * mm
    if head and gt:
        f["head_gt_dy_mm"] = (gt[1] - head[1]) * mm            # верхушка вертела выше/ниже центра головки
    if lt and lu and ld:
        (x1, y1), (x2, y2) = lu, ld
        f["lt_mm"] = abs((y2 - y1) * lt[0] - (x2 - x1) * lt[1] + x2 * y1 - y2 * x1) / max(1e-6, math.hypot(x2 - x1, y2 - y1)) * mm
    f["lt_present"] = float(lt is not None)
    f["ischium_present"] = float(isch is not None)
    f["ischium_cut"] = _near_black(img, *isch) if isch else 0.0
    f["n_missing"] = float(sum(1 for p in (head, gt, gl, ns, ni, st, sb) if p is None))
    f["narrow_frame"] = float(w < 260)
    f["conf_min"] = _conf_min(lm)
    f["ischium_conf"], f["lt_conf"] = _conf(lm, "ischium"), _conf(lm, "lt_tip")
    f["base_conf_min"] = min(_conf(lm, k) for k in ("head", "gt_top", "gt_lat", "neck_sup", "neck_inf", "shaft_top", "shaft_bot"))
    return f


class CriteriaModel:
    """Логистические регрессии по критериям поверх признаков (обучаются в scripts/train_criteria.py)."""

    def __init__(self, models: dict):
        self.models = models                     # criterion → {"features": [...], "coef", "intercept", "mean", "std"}

    @staticmethod
    def _sigmoid(z: float) -> float:
        return 1.0 / (1.0 + math.exp(-z))

    def predict(self, feats: dict, flags: list[str]) -> dict[str, float]:
        out = {}
        for c in flags:
            m = self.models.get(c)
            if not m:
                out[c] = 0.0
                continue
            x = np.array([feats.get(k, 0.0) for k in m["features"]], float)
            z = float(((x - m["mean"]) / m["std"]) @ m["coef"] + m["intercept"])
            out[c] = self._sigmoid(z)
        return out

    def to_json(self) -> dict:
        return {c: {"features": m["features"], "coef": list(map(float, m["coef"])), "intercept": float(m["intercept"]),
                    "mean": list(map(float, m["mean"])), "std": list(map(float, m["std"]))} for c, m in self.models.items()}

    @classmethod
    def from_json(cls, d: dict) -> "CriteriaModel":
        return cls({c: {"features": m["features"], "coef": np.array(m["coef"]), "intercept": m["intercept"],
                        "mean": np.array(m["mean"]), "std": np.array(m["std"])} for c, m in d.items()})


SPINE_CRITERIA = list(SPINE_FLAGS)     # spine_positioning, spine_axis_tilt, spine_artifact
HIP_CRITERIA = list(HIP_FLAGS)         # hip_positioning_rotation, hip_roi_field
