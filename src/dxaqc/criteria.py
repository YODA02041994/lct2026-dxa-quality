"""Критерии качества по ориентирам: признаки → вероятность нарушения по каждому критерию ТЗ.

Идея (docs/05, эксп. 02): критерии ТЗ — геометрия. Из точек (врача или локализатора) считаем понятные
признаки — угол оси, запасы поля в сантиметрах, видимость структур — и поверх них учим маленькую модель
на метках экспертов. Жёстких порогов нет: сколиоз и перелом ломают «5° в лоб», а границу нормы по
малому вертелу задают эксперты, не мы. Каждый признак объясним врачу и выводится на снимок.

Формат точек: {name: {"x", "y", "present"[, "conf"]}} в пикселях исходника.
"""
from __future__ import annotations

import math

import cv2

import numpy as np

from .labels import HIP_FLAGS, REGION_SPINE, SPINE_FLAGS, pixel_mm

SPINE_FEATURES = ["axis_deg", "axis_ok", "n_levels", "L5_present", "th12_present", "crests", "crest_l", "crest_r",
                  "top_gap_cm", "bottom_gap_cm", "conf_min", "th12_conf", "crest_l_conf", "crest_r_conf", "crests_conf", "axis_img_deg", "cnn_artifact",
                  "bottom_width", "bottom_ratio", "th_frac40", "th_frac40_log", "th_p999", "th_lines", "th_linearea", "bg_median", "axis_min_third", "soft_mean"]
HIP_FEATURES = ["top_cm", "lat_cm", "bottom_cm", "min_cm", "shaft_deg", "neck_shaft_deg", "neck_mm", "head_gt_dy_mm",
                "lt_mm", "lt_present", "ischium_present", "ischium_cut", "n_missing", "narrow_frame", "conf_min",
                "ischium_conf", "lt_conf", "base_conf_min", "frame_h_cm", "frame_w_cm", "lowest_gap_cm", "cnn_hip_pos", "shaft_img_deg"]

# Какие признаки видит каждый критерий. При 6–36 положительных примерах лишние признаки только шумят,
# поэтому наборы короткие и клинически осмысленные (docs/05, эксп. 02–03).
CRITERION_FEATURES = {
    "spine_positioning": ["bottom_width", "soft_mean"],   # полоса таза внизу + средняя яркость мягких тканей (эксп. 13: сырой 0,83; крупный пациент → гребни вне кадра)
    "spine_axis_tilt": ["axis_img_deg", "axis_min_third"],   # угол по точкам сети — шум (0,55); прямой наклон ≠ сколиоз (эксп. 05)
    "spine_artifact": ["th_frac40_log", "cnn_artifact"],  # тонкие яркие линии вне столба (эксп. 05: сырой 0,91; лог — хвост тяжёлый) + CNN
    "hip_positioning_rotation": ["shaft_img_deg", "cnn_hip_pos"],   # эксп. 06: с сильным ансамблем CNN остальные точечные признаки только шумят; ось диафиза — для объяснимости
    "hip_roi_field": ["min_cm", "frame_h_cm"],
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


def spine_features(lm: dict, h: int, w: int, img: np.ndarray | None = None, cnn: dict | None = None) -> dict:
    mm = pixel_mm(w)
    levels = [_pt(lm, k) for k in ("L1", "L2", "L3", "L4", "L5")]
    pres = [p for p in levels if p]
    f = {k: 0.0 for k in SPINE_FEATURES}
    if len(pres) >= 2:
        xs, ys = np.array([p[0] for p in pres]), np.array([p[1] for p in pres])
        k = np.polyfit(ys, xs, 1)[0] if np.ptp(ys) > 1 else 0.0
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
    f["axis_min_third"] = spine_axis_min_third(img)
    f["cnn_artifact"] = float((cnn or {}).get("cnn_artifact", 0.0))
    f["bottom_width"], f["bottom_ratio"] = bottom_bone_width(img)
    f["soft_mean"] = float(img[img > 0].mean()) if img is not None and (img > 0).any() else 0.0
    f.update(artifact_features(img))
    return f


def _fit_angle(ys: np.ndarray, xs: np.ndarray) -> float:
    if len(ys) < 8:
        return 0.0
    k = np.polyfit(ys, xs, 1)[0]
    res = xs - (k * ys + np.median(xs - k * ys))
    keep = np.abs(res) < np.percentile(np.abs(res), 80)
    if keep.sum() >= 6:
        k = np.polyfit(ys[keep], xs[keep], 1)[0]
    return math.degrees(math.atan(k))


def spine_axis_min_third(img: np.ndarray | None) -> float:
    """Наименьший |наклон| среди трёх третей позвоночника. Косая укладка наклоняет весь столб (все трети),
    сколиоз — дуга: хотя бы одна треть почти вертикальна (эксп. 05: сколиозы давали ложные тревоги по оси)."""
    c = _spine_centroids(img)
    if c is None:
        return 0.0
    ys, xs = c
    n = ys.max() - ys.min()
    if n < 30:
        return abs(_fit_angle(ys, xs))
    angs = []
    for i in range(3):
        sel = (ys >= ys.min() + n * i / 3) & (ys < ys.min() + n * (i + 1) / 3)
        angs.append(abs(_fit_angle(ys[sel], xs[sel])))
    return float(min(angs))


def _spine_centroids(img: np.ndarray | None):
    """Центры масс кости по строкам в центральной трети кадра (без верхних рёбер и нижнего таза)."""
    if img is None:
        return None
    h, w = img.shape
    band = img[int(h * 0.15):int(h * 0.85)].astype(np.float32)
    nz = band[band > 0]
    if nz.size < 100:
        return None
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
        return None
    return np.array(ys, float), np.array(xs, float)


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


def shaft_axis_from_image(img: np.ndarray | None, x_seed: float, rows_frac: float = 0.30, pct: int = 55) -> float:
    """Наклон диафиза бедра к вертикали — главная ось (PCA) связной области кости в нижних 30 % кадра.
    Затравка — только колонка нижней точки диафиза (у сети ошибка 3,7 мм). Угол по двум точкам сети на коротком
    отрезке рассыпается (эксп. 05: r = 0,24 с врачом, AUC 0,60); по массе кости — r = 0,47, AUC 0,74 ≈ как по точкам врача."""
    if img is None:
        return 0.0
    h, w = img.shape
    nz = img[img > 0]
    if nz.size < 50:
        return 0.0
    y0 = int(h * (1 - rows_frac))
    m = (img[y0:] >= np.percentile(nz, pct)).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    best, bd = None, 1e9
    for r in range(m.shape[0] - 1, max(0, m.shape[0] - 6), -1):
        xs = np.where(lab[r] > 0)[0]
        if xs.size:
            j = xs[np.argmin(np.abs(xs - x_seed))]
            if abs(j - x_seed) < bd:
                bd, best = abs(j - x_seed), lab[r, j]
    if best is None or bd > 25 or stats[best, cv2.CC_STAT_AREA] < 60:
        return 0.0
    ys, xs = np.where(lab == best)
    pts = np.stack([xs, ys], 1).astype(np.float32)
    cov = np.cov((pts - pts.mean(0)).T)
    ev, evec = np.linalg.eigh(cov)
    v = evec[:, int(np.argmax(ev))]
    return abs(math.degrees(math.atan2(v[0], v[1])))


def artifact_features(img: np.ndarray | None) -> dict:
    """Посторонние предметы на снимке позвоночника (эксп. 05, монтаж 17 положительных): это тонкие яркие линии и
    дуги у краёв кадра (фурнитура одежды, цепочки) и «серая вуаль» плотной ткани. Ориентиры их не видят, CNN — слабо
    (0,73–0,76). Здесь: верхний top-hat (ядро 9 px) выделяет тонкие яркие структуры; считаем их вне столба
    позвоночника (±20 % ширины от центра столба) и вне чёрных масок. th_frac40 — доля пикселей с откликом > 40
    (AUC 0,91), th_p999 — 99,9-й перцентиль отклика (0,90), th_lines — число вытянутых компонент (0,78),
    bg_median — яркость фона в средней полосе (вуаль, 0,65)."""
    f = {"th_frac40": 0.0, "th_frac40_log": 0.0, "th_p999": 0.0, "th_lines": 0.0, "th_linearea": 0.0, "bg_median": 0.0}
    if img is None:
        return f
    h, w = img.shape
    nz = img[img > 0]
    if nz.size < 100:
        return f
    thr = np.percentile(nz, 70)
    mid = img[int(h * 0.2):int(h * 0.8)]
    cols = np.convolve((mid >= thr).mean(0), np.ones(15) / 15, "same")
    cx = int(np.argmax(cols))
    outside = np.ones((h, w), bool)
    outside[:, max(0, cx - int(w * 0.2)):min(w, cx + int(w * 0.2))] = False
    outside &= img > 0
    th = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    r = th[outside].astype(np.float32)
    if r.size:
        f["th_frac40"] = float((r > 40).mean())
        f["th_frac40_log"] = float(np.log1p(1000.0 * f["th_frac40"]))      # хвост тяжёлый: 0 у большинства, до 0,05 у предметов
        f["th_p999"] = float(np.percentile(r, 99.9))
    m = ((th > 50) & outside).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    for i in range(1, n):
        a, bw, bh = st[i, cv2.CC_STAT_AREA], st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT]
        if a >= 12 and max(bw, bh) / max(1, min(bw, bh)) >= 2.5:
            f["th_lines"] += 1
            f["th_linearea"] += float(a)
    bg = mid[outside[int(h * 0.2):int(h * 0.8)]]
    if bg.size:
        f["bg_median"] = float(np.median(bg))
    return f


def bottom_bone_width(img: np.ndarray | None, frac: float = 0.10) -> tuple[float, float]:
    """Гребни подвздошных костей в кадре → в нижних 10 % строк широкая яркая полоса (таз), иначе — только узкий
    столб позвоночника/крестца. Доля колонок с костью внизу и её отношение к ширине столба в середине кадра.
    Без ориентиров; эксп. 05: AUC 0,90 против «некорректной укладки» позвоночника (у уверенности сети по гребням — 0,79)."""
    if img is None:
        return 0.0, 0.0
    h, w = img.shape
    nz = img[img > 0]
    if nz.size < 100:
        return 0.0, 0.0
    thr = np.percentile(nz, 70)
    bot = img[int(h * (1 - frac)):]
    mid = img[int(h * 0.4):int(h * 0.6)]
    bw = float((bot.max(0) >= thr).mean())
    mw = float((mid.max(0) >= thr).mean())
    return bw, bw / max(mw, 1e-3)


def _near_black(img: np.ndarray | None, x: float, y: float, reach: int = 6) -> float:
    """Точка у границы чёрной маски / края кадра — структура обрезана (используется для седалищной кости)."""
    if img is None:
        return 0.0
    h, w = img.shape
    x0, x1 = max(0, int(x) - 4), min(w, int(x) + 5)
    below = img[min(h - 1, int(y) + 2):min(h, int(y) + reach + 1), x0:x1]
    return float((below.size > 0 and (below == 0).mean() > 0.8) or int(y) + reach >= h)


def hip_features(lm: dict, h: int, w: int, side: str | None, img: np.ndarray | None = None, cnn: dict | None = None) -> dict:
    mm = pixel_mm(w)
    f = {k: 0.0 for k in HIP_FEATURES}
    head, gt, gl = _pt(lm, "head"), _pt(lm, "gt_top"), _pt(lm, "gt_lat")
    ns, ni, lt, lu, ld = _pt(lm, "neck_sup"), _pt(lm, "neck_inf"), _pt(lm, "lt_tip"), _pt(lm, "lt_up"), _pt(lm, "lt_down")
    isch, st, sb = _pt(lm, "ischium"), _pt(lm, "shaft_top"), _pt(lm, "shaft_bot")
    lateral_x = (lambda p: p[0]) if side != "L" else (lambda p: (w - 1) - p[0])
    f["top_cm"] = gt[1] * mm / 10 if gt else 0.0
    f["lat_cm"] = lateral_x(gl) * mm / 10 if gl else 0.0
    low = ld or lt or st
    # нижних ориентиров нет в кадре → поле снизу срезано: запас 0, и он ОБЯЗАН войти в минимум
    # (5 из 7 нарушений поля ROI в обучающем наборе — короткий кадр с вертелом у нижнего края)
    f["bottom_cm"] = (h - low[1]) * mm / 10 if low else 0.0
    vals = [f["bottom_cm"]] + [v for v, p in ((f["top_cm"], gt), (f["lat_cm"], gl)) if p]
    f["min_cm"] = min(vals)
    f["frame_h_cm"], f["frame_w_cm"] = h * mm / 10, w * mm / 10
    present_y = [p[1] for p in (head, gt, gl, ns, ni, lt, lu, ld, isch, st) if p]
    f["lowest_gap_cm"] = (h - max(present_y)) * mm / 10 if present_y else 0.0
    f["cnn_hip_pos"] = float((cnn or {}).get("cnn_hip_pos", 0.0))
    if st and sb:
        f["shaft_deg"] = abs(math.degrees(math.atan2(sb[0] - st[0], sb[1] - st[1])))
    f["shaft_img_deg"] = shaft_axis_from_image(img, sb[0]) if sb else 0.0
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


# ---------- объяснение для врача и совет лаборанту (текст по признакам, без ML) ----------
ADVICE = {
    "spine_positioning": "Сместить поле сканирования так, чтобы в кадр вошли гребни подвздошных костей снизу и Th12 с рёбрами сверху; проверить, что видны L1–L4 целиком.",
    "spine_axis_tilt": "Выровнять пациента по средней линии стола: позвоночник должен идти вертикально по центру кадра.",
    "spine_artifact": "Убрать металлические предметы и одежду с фурнитурой (пуговицы, молнии, цепочки); переснять.",
    "hip_positioning_rotation": "Довернуть стопу внутрь на 15–25° и зафиксировать (малый вертел должен быть виден минимально), бедро — параллельно оси стола.",
    "hip_roi_field": "Расширить поле: в кадре должны быть вся головка, большой вертел и не менее 2,5 см диафиза ниже малого вертела; латеральный край не срезать.",
}


def explain(region: str, f: dict, probs: dict, thresholds: dict) -> tuple[str, list[str]]:
    """Короткий протокол измерений и советы по нарушенным критериям."""
    parts, advice = [], []
    if region == REGION_SPINE:
        parts.append(f"ось {f.get('axis_img_deg', 0):.1f}°")
        parts.append("гребни в кадре" if f.get("bottom_width", 0) >= 0.45 else "полоса таза внизу узкая — гребни, возможно, вне кадра")
        parts.append(f"Th12 (уверенность {f.get('th12_conf', 0):.2f})")
        if f.get("th_frac40", 0) > 0.003:
            parts.append(f"тонкие яркие линии вне столба ({f.get('th_frac40', 0) * 100:.1f} % площади)")
    else:
        parts.append(f"диафиз {f.get('shaft_img_deg', 0):.1f}° к вертикали")
        parts.append(f"запас до края: снизу {f.get('bottom_cm', 0):.1f} см, сверху {f.get('top_cm', 0):.1f} см, сбоку {f.get('lat_cm', 0):.1f} см")
        parts.append(f"высота кадра {f.get('frame_h_cm', 0):.1f} см")
        if "cnn_hip_pos" in f:
            parts.append(f"ротация по вертелу (сеть) {f['cnn_hip_pos']:.2f}")
    for c, p in probs.items():
        if p >= thresholds.get(c, 0.5):
            advice.append(ADVICE.get(c, ""))
    return "; ".join(parts), [a for a in advice if a]
