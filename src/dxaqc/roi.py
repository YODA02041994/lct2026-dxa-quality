"""Предложение разметки областей измерения по ориентирам сети (доп. функционал ТЗ, п. 2.6: коррекция разметки
с подтверждением специалистом).

Сервис предлагает, где должны стоять области измерения: тела L1–L4 на снимке позвоночника и шейка бедра.
Специалист видит предложение на снимке и подтверждает или отклоняет его; решение сохраняется в отчёте прогона.
Линий, которые наносит сам денситометр, в данных постановщика нет, поэтому сравнить предложение с ними нельзя:
это подсказка для лаборанта, а не измерение качества.
"""
from __future__ import annotations

import numpy as np

SPINE_ORDER = ["th12", "L1", "L2", "L3", "L4", "L5"]
NECK_WIDTH_MM = 15.0                                   # ширина области шейки вдоль её оси (стандарт GE Lunar)


def _pt(lm: dict, name: str, min_conf: float = 0.3) -> np.ndarray | None:
    """Ориентир как точка (x, y); понимает оба вида записи: словарь сети и список [x, y, conf] из details."""
    v = lm.get(name)
    if v is None:
        return None
    if isinstance(v, dict):
        if not v.get("present", True) and float(v.get("conf", 0.0)) < min_conf:
            return None
        return np.array([v["x"], v["y"]], float)
    if len(v) > 2 and float(v[2]) < min_conf:
        return None
    return np.array(v[:2], float)


def _half_width(img: np.ndarray | None, c: np.ndarray, nrm: np.ndarray, step: float) -> float:
    """Половина ширины тела позвонка: по профилю яркости поперёк оси; без снимка — доля шага между телами."""
    default = 0.7 * step
    if img is None:
        return default
    h, w = img.shape[:2]
    reach = int(1.2 * step)
    ts = np.arange(-reach, reach + 1)
    xs = np.clip(np.round(c[0] + ts * nrm[0]).astype(int), 0, w - 1)
    ys = np.clip(np.round(c[1] + ts * nrm[1]).astype(int), 0, h - 1)
    prof = img[ys, xs].astype(np.float32)
    if len(prof) < 9:
        return default
    k = max(2, len(prof) // 10)
    soft = float(np.median(np.r_[prof[:k], prof[-k:]]))
    mid = len(prof) // 2
    bone = float(np.mean(prof[mid - 2: mid + 3]))
    if bone - soft < 8:
        return default
    thr = (bone + soft) / 2.0
    left = right = 0
    while mid - left - 1 >= 0 and prof[mid - left - 1] >= thr:
        left += 1
    while mid + right + 1 < len(prof) and prof[mid + right + 1] >= thr:
        right += 1
    return float(np.clip((left + right) / 2.0 + 2.0, 0.45 * step, 0.95 * step))


def _poly(points: list[np.ndarray], shape: tuple[int, int] | None) -> list[list[float]]:
    out = []
    for p in points:
        x, y = float(p[0]), float(p[1])
        if shape is not None:
            x, y = min(max(x, 0.0), shape[1] - 1.0), min(max(y, 0.0), shape[0] - 1.0)
        out.append([round(x, 1), round(y, 1)])
    return out


def spine_roi(lm: dict, img: np.ndarray | None = None, mm: float = 0.6) -> dict[str, list[list[float]]]:
    """Четырёхугольники тел L1–L4: границы — посередине между центрами соседних тел, ширина — по профилю яркости."""
    c = {k: _pt(lm, k) for k in SPINE_ORDER}
    steps = [float(np.linalg.norm(c[b] - c[a])) for a, b in zip(SPINE_ORDER[:-1], SPINE_ORDER[1:]) if c[a] is not None and c[b] is not None]
    if sum(c[k] is not None for k in SPINE_ORDER[1:5]) < 2 or not steps:
        return {}
    step = float(np.median(steps))
    shape = img.shape[:2] if img is not None else None
    out = {}
    for i in range(1, 5):
        k = SPINE_ORDER[i]
        if c[k] is None:
            continue
        up, dn = c[SPINE_ORDER[i - 1]], c[SPINE_ORDER[i + 1]]
        ax = (dn - up) if (up is not None and dn is not None) else (dn - c[k]) if dn is not None else (c[k] - up) if up is not None else None
        if ax is None or np.linalg.norm(ax) < 1e-6:
            continue
        ax = ax / np.linalg.norm(ax)
        top = (c[k] + up) / 2 if up is not None else c[k] - ax * step / 2
        bot = (c[k] + dn) / 2 if dn is not None else c[k] + ax * step / 2
        nrm = np.array([-ax[1], ax[0]])
        hw = _half_width(img, c[k], nrm, step)
        out[k] = _poly([top - nrm * hw, top + nrm * hw, bot + nrm * hw, bot - nrm * hw], shape)
    return out


def hip_roi(lm: dict, img: np.ndarray | None = None, mm: float = 0.6) -> dict[str, list[list[float]]]:
    """Область шейки бедра: прямоугольник шириной 15 мм поперёк оси шейки, между верхним и нижним её контурами."""
    ns, ni = _pt(lm, "neck_sup"), _pt(lm, "neck_inf")
    if ns is None or ni is None:
        return {}
    across = ni - ns
    length = float(np.linalg.norm(across))
    if length < 5:
        return {}
    across = across / length
    along = np.array([-across[1], across[0]])
    mid = (ns + ni) / 2
    hl, hh = (NECK_WIDTH_MM / 2) / mm, 0.5 * length * 1.25
    shape = img.shape[:2] if img is not None else None
    return {"neck": _poly([mid - along * hl - across * hh, mid + along * hl - across * hh,
                           mid + along * hl + across * hh, mid - along * hl + across * hh], shape)}


def propose(kind: str, lm: dict, img: np.ndarray | None = None, mm: float = 0.6) -> dict[str, list[list[float]]]:
    return spine_roi(lm, img, mm) if kind == "spine" else hip_roi(lm, img, mm)
