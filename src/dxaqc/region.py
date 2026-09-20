"""Определение анатомической области и стороны бедра по изображению.

Область — по ширине кадра (стандарт аппарата GE Lunar Prodigy Advance, подтверждено постановщиком
16–17.09): 300 px — поясничный отдел, 280 px — проксимальный отдел бедра. В закрытом тесте так же.
Сторона бедра для сдачи не нужна (`anatomical_region` без стороны), но нужна нам: разметка
организатора ведётся отдельно по правому и левому бедру.

Сторона — по геометрии: у ПРАВОГО бедра на снимке таз справа, у левого — слева.
Основной признак — положение костной массы таза в верхней половине кадра.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .labels import REGION_HIP, REGION_SPINE

SPINE_WIDTH = 300
HIP_WIDTH = 280
HIP_ORTHO_WIDTH = 248   # ортопедический режим (бедро с эндопротезом), кадр ~248×403 — исследование №70


def region_by_width(cols: int) -> str | None:
    """Официальное значение anatomical_region по ширине кадра; None — нестандартный кадр."""
    if cols == SPINE_WIDTH:
        return REGION_SPINE
    if cols in (HIP_WIDTH, HIP_ORTHO_WIDTH):
        return REGION_HIP
    return None


@dataclass
class SideEstimate:
    side: str | None          # 'R' / 'L' / None
    shaft_x: float            # центр диафиза в нижней части кадра, доля ширины 0..1
    upper_balance: float      # (масса справа − масса слева) / сумма в верхней половине, −1..1
    agree: bool               # оба признака согласны


def hip_side(img: np.ndarray) -> SideEstimate:
    """Сторона бедра по двум признакам.

    1) диафиз: центр масс яркости в нижних 20 % строк. Правое бедро → диафиз левее центра.
    2) таз: в верхней половине кадра костная масса смещена к тазу. Правое бедро → масса справа.
    Чёрные маски лаборанта яркости не добавляют, поэтому на признаки не влияют.
    """
    a = img.astype(np.float32)
    h, w = a.shape
    bottom = a[int(h * 0.8):, :]
    colsum = bottom.sum(axis=0)
    shaft_x = float((colsum * np.arange(w)).sum() / max(colsum.sum(), 1e-6)) / w

    upper = a[: h // 2, :]
    left = float(upper[:, : w // 2].sum())
    right = float(upper[:, w // 2:].sum())
    balance = (right - left) / max(right + left, 1e-6)

    # Проверено на обучающем наборе: признак «таз» один даёт 73/73 пары и 5/5 одиночных бёдер;
    # «диафиз» ошибается, когда диафиз близко к центру кадра, — оставлен только как индикатор согласия.
    side_shaft = "R" if shaft_x < 0.5 else "L"
    side_pelvis = "R" if balance > 0 else "L"
    return SideEstimate(side=side_pelvis, shaft_x=shaft_x, upper_balance=balance,
                        agree=side_shaft == side_pelvis)


def assign_sides(balances: list[float]) -> list[str]:
    """Стороны для бёдер ОДНОГО исследования. Если бёдер два — они обязаны быть разными:
    у кого баланс больше, тот правый. Так пара не получит одинаковую сторону даже при слабом признаке."""
    if len(balances) == 2:
        return ["R", "L"] if balances[0] >= balances[1] else ["L", "R"]
    return ["R" if b > 0 else "L" for b in balances]
