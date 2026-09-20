"""«Розетка» для модели: единый интерфейс предсказателя и сборка итогового ответа.

Любая модель (CNN, геометрическое правило, их смесь) обязана вернуть вероятности по критериям
своей области. Всё остальное — порог, quality_class, официальный текст нарушений, quality_prob —
собирается здесь одинаково для всех моделей, чтобы формат вывода не зависел от способа предсказания.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .labels import HIP_FLAGS, REGION_HIP, REGION_SPINE, SPINE_FLAGS, violation_string

DEFAULT_THRESHOLD = 0.5


def flags_for(region: str) -> list[str]:
    if region == REGION_SPINE:
        return list(SPINE_FLAGS)
    if region == REGION_HIP:
        return list(HIP_FLAGS)
    return []


class Predictor:
    """Базовый класс. Наследник реализует predict_flags()."""

    name = "base"

    def predict_flags(self, img: np.ndarray, region: str, side: str | None) -> dict[str, float]:
        raise NotImplementedError


class NullPredictor(Predictor):
    """Заглушка до появления обученной модели: нарушений нет, вероятность 0."""

    name = "null"

    def predict_flags(self, img, region, side):
        return {f: 0.0 for f in flags_for(region)}


@dataclass
class Verdict:
    quality_class: int
    violation_type: str
    quality_prob: float
    flag_probs: dict[str, float] = field(default_factory=dict)


def aggregate(flag_probs: dict[str, float], thresholds: dict[str, float] | None = None) -> Verdict:
    """Вероятности критериев → итог по снимку.

    quality_class = 1, если хотя бы один критерий выше своего порога (правило организатора: любое
    нарушение → снимок с нарушением). quality_prob = вероятность «есть хоть одно нарушение»,
    берём максимум по критериям: монотонно, не раздувается от числа критериев, пригодно для ROC-AUC.
    """
    thresholds = thresholds or {}
    positive = {f: p >= thresholds.get(f, DEFAULT_THRESHOLD) for f, p in flag_probs.items()}
    prob = float(max(flag_probs.values())) if flag_probs else 0.0
    return Verdict(
        quality_class=int(any(positive.values())),
        violation_type=violation_string(positive),
        quality_prob=round(min(max(prob, 0.0), 1.0), 4),
        flag_probs=dict(flag_probs),
    )
