"""Сколиотическая деформация по ориентирам поясничного отдела — дополнительная функция, не критерий ТЗ.

Зачем. По словам постановщика (сессия 16.09) и экспертов (консультация 16.10) сколиоз — не нарушение укладки:
денситометрия выполняется лёжа и для оценки статики непригодна. Но отклонение оси у сколиотика объективно
больше 5°, и критерий «ось не выровнена» срабатывал на таких снимках ложно: в обучающем наборе все снимки
с пометкой эксперта «сколиоз» размечены по оси как норма. Поэтому искривление измеряется отдельно, показывается
отдельно, и при выраженной деформации критерий оси не бракует снимок (переключаемо: DXAQC_SCOLIOSIS_DAMP=0).

Мера — угол искривления по центрам тел позвонков (L1–L5): ломаная через центры, разность направлений
соседних отрезков, наибольшая минус наименьшая. Это «центроидный» вариант угла Кобба; по замыкательным
пластинкам на денситометрии в 0,6 мм/px измерять нечем. Пороги клинические, не подобранные по данным:
< 5° — искривления нет; 5–10° — пограничная асимметрия; ≥ 10° — сколиотическая деформация.
На снимках вне обучения правило помечает деформацией 18 из 94 снимков позвоночника (6 из 12 с пометкой
эксперта «сколиоз») и поднимает F1 критерия оси с 0,42 до 0,57 при 10 нарушениях (docs/05, эксп. 22).
"""
from __future__ import annotations

import math
import os

import numpy as np

GRADE_NONE, GRADE_BORDERLINE, GRADE_MARKED = "none", "borderline", "marked"
BORDERLINE_DEG = 5.0
MARKED_DEG = 10.0
MIN_CONF = 0.3                # ориентир с меньшей уверенностью в измерении не участвует
MIN_POINTS = 4
AXIS_DAMP = 0.3               # во сколько раз уменьшается вероятность критерия оси при выраженной деформации
LEVELS = ("L1", "L2", "L3", "L4", "L5")   # Th12 не берём: его ориентир вдвое менее точен (4,8 мм) и ломает изгиб на прямом позвоночнике

GRADE_TITLE = {GRADE_NONE: "", GRADE_BORDERLINE: "пограничная асимметрия", GRADE_MARKED: "сколиотическая деформация"}


def damping_enabled() -> bool:
    return os.environ.get("DXAQC_SCOLIOSIS_DAMP", "1") not in ("0", "false", "no")


def measure(lm: dict, mm_per_px: float = 0.6) -> dict:
    """lm — ориентиры локализатора: {name: {x, y, conf, present}}. → {grade, bend_deg, dev_mm, n_points}."""
    pts = sorted(((float(v["y"]), float(v["x"])) for k, v in (lm or {}).items()
                  if k in LEVELS and float(v.get("conf", 0.0)) >= MIN_CONF), key=lambda t: t[0])
    out = {"grade": GRADE_NONE, "bend_deg": None, "dev_mm": None, "n_points": len(pts)}
    if len(pts) < MIN_POINTS:
        return out
    ys = np.array([p[0] for p in pts]); xs = np.array([p[1] for p in pts])
    segs = [math.degrees(math.atan2(xs[i + 1] - xs[i], ys[i + 1] - ys[i])) for i in range(len(xs) - 1)]
    bend = float(max(segs) - min(segs))
    k, b = np.polyfit(ys, xs, 1)
    dev = float(np.max(np.abs(xs - (k * ys + b)))) * float(mm_per_px)
    out.update(bend_deg=round(bend, 1), dev_mm=round(dev, 1))
    out["grade"] = GRADE_MARKED if bend >= MARKED_DEG else GRADE_BORDERLINE if bend >= BORDERLINE_DEG else GRADE_NONE
    return out


def damp_axis(probs: dict, grade: str) -> tuple[dict, bool]:
    """При выраженной деформации вероятность критерия оси уменьшается: наклон объясняется деформацией, не укладкой."""
    if grade != GRADE_MARKED or not damping_enabled() or "spine_axis_tilt" not in probs:
        return probs, False
    out = dict(probs)
    out["spine_axis_tilt"] = float(out["spine_axis_tilt"]) * AXIS_DAMP
    return out, True


def decision_text(grade: str, other_violations: bool) -> str:
    """Текст для лаборанта и врача: что делать с исследованием. Без цифр и указаний «на сколько повернуть»."""
    if grade == GRADE_MARKED:
        if other_violations:
            return ("Выраженная сколиотическая деформация позвоночника. Кроме неё отмечены нарушения укладки — "
                    "см. рекомендации.")
        return ("Выраженная сколиотическая деформация позвоночника: отклонение оси связано с деформацией, а не с укладкой. "
                "Критерии укладки соблюдены; оценка результатов исследования — на усмотрение врача.")
    if grade == GRADE_BORDERLINE:
        if other_violations:
            return "Небольшая асимметрия оси позвоночника (пограничная); отмечены нарушения укладки — см. рекомендации."
        return ("Небольшая асимметрия оси позвоночника (пограничная). Критерии укладки соблюдены — "
                "исследование можно продолжать.")
    return ""
