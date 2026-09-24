"""«Розетка» для модели: единый интерфейс предсказателя и сборка итогового ответа.

Любая модель (CNN, геометрическое правило, их смесь) обязана вернуть вероятности по критериям
своей области. Всё остальное — порог, quality_class, официальный текст нарушений, quality_prob —
собирается здесь одинаково для всех моделей, чтобы формат вывода не зависел от способа предсказания.

Боевая модель — LandmarkPredictor: локализатор ориентиров (landmarks.py) → признаки (criteria.py) →
логистические регрессии по критериям (weights/criteria.json). Каждый ответ объясним: точки, углы,
запасы в сантиметрах — их же можно нарисовать на снимке.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np

from .labels import HIP_FLAGS, REGION_HIP, REGION_SPINE, SPINE_FLAGS, violation_string

DEFAULT_THRESHOLD = 0.5
WEIGHTS_DIR = os.environ.get("DXAQC_WEIGHTS", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "weights"))


def flags_for(region: str) -> list[str]:
    if region == REGION_SPINE:
        return list(SPINE_FLAGS)
    if region == REGION_HIP:
        return list(HIP_FLAGS)
    return []


class Predictor:
    """Базовый класс. Наследник реализует predict_flags(); explain() — необязательные детали для отчёта."""

    name = "base"
    thresholds: dict[str, float] = {}

    def predict_flags(self, img: np.ndarray, region: str, side: str | None) -> dict[str, float]:
        raise NotImplementedError

    def explain(self) -> dict:
        """Подробности последнего предсказания (ориентиры, признаки) — для доп. серии и отладки."""
        return {}


class NullPredictor(Predictor):
    """Заглушка: нарушений нет, вероятность 0."""

    name = "null"

    def predict_flags(self, img, region, side):
        return {f: 0.0 for f in flags_for(region)}


class LandmarkPredictor(Predictor):
    """Ориентиры → признаки → вероятности критериев. Веса: weights/landmarks_{spine,hip}.pt + weights/criteria.json."""

    name = "landmarks"

    def __init__(self, weights_dir: str = WEIGHTS_DIR, device: str | None = None):
        from .criteria import CriteriaModel
        from .landmarks import Localizer
        self.loc = {"spine": Localizer("spine", os.path.join(weights_dir, "landmarks_spine.pt"), device),
                    "hip": Localizer("hip", os.path.join(weights_dir, "landmarks_hip.pt"), device)}
        from .cnn import CNN_SOURCES, CnnEnsemble
        self.cnn = {}
        for feat, names in CNN_SOURCES.items():
            ens = CnnEnsemble(weights_dir, names, device)
            if len(ens):
                self.cnn[feat] = ens
        from .cnn import ObjMapScorer
        om_path = os.path.join(weights_dir, "objmap_objectcxr_r18fpn.pt")
        self.objmap = ObjMapScorer(om_path, device) if os.path.exists(om_path) else None
        cj = json.load(open(os.path.join(weights_dir, "criteria.json"), encoding="utf-8"))
        self.crit = CriteriaModel.from_json(cj["models"])
        self.thresholds = {c: float(m.get("threshold", DEFAULT_THRESHOLD)) for c, m in cj["models"].items()}
        self._last: dict = {}

    def predict_flags(self, img, region, side):
        from .criteria import hip_features, spine_features
        h, w = img.shape
        if region == REGION_SPINE:
            lm = self.loc["spine"].predict(img)
            cnn = {"cnn_artifact": self.cnn["cnn_artifact"].score(img)} if "cnn_artifact" in self.cnn else {}
            boxes = []
            if self.objmap is not None:
                om, boxes = self.objmap.features(img)
                cnn.update(om)
            feats = spine_features(lm, h, w, img, cnn)
        else:
            boxes = []
            lm = self.loc["hip"].predict(img, side)
            cnn = {"cnn_hip_pos": self.cnn["cnn_hip_pos"].score(img, side, lm)} if "cnn_hip_pos" in self.cnn else {}
            feats = hip_features(lm, h, w, side, img, cnn)
        probs = self.crit.predict(feats, flags_for(region))
        from .criteria import explain
        comment, advice = explain(region, feats, probs, self.thresholds)
        self._last = {"landmarks": lm, "features": feats, "probs": probs, "comment": comment, "advice": advice, "objects": boxes}
        return probs

    def explain(self) -> dict:
        return dict(self._last)


def load_default_predictor(device: str | None = None) -> Predictor:
    """Боевая модель, если веса на месте; иначе заглушка (и об этом надо предупредить в логе)."""
    need = [os.path.join(WEIGHTS_DIR, f) for f in ("landmarks_spine.pt", "landmarks_hip.pt", "criteria.json")]
    if all(os.path.exists(p) for p in need):
        return LandmarkPredictor(WEIGHTS_DIR, device)
    return NullPredictor()


REVIEW_BAND = 0.10                                   # |p − порог| < 0,10 → сомнительно, показать врачу


@dataclass
class Verdict:
    quality_class: int
    violation_type: str
    quality_prob: float
    flag_probs: dict[str, float] = field(default_factory=dict)
    needs_review: bool = False                        # хотя бы один критерий у порога: решение стоит проверить глазами
    review_flags: list[str] = field(default_factory=list)


def aggregate(flag_probs: dict[str, float], thresholds: dict[str, float] | None = None) -> Verdict:
    """Вероятности критериев → итог по снимку.

    quality_class = 1, если хотя бы один критерий выше своего порога (правило организатора: любое
    нарушение → снимок с нарушением). quality_prob = вероятность «есть хоть одно нарушение»:
    1 − ∏(1 − p_i) — растёт с каждым подозрительным критерием, не раздувается от их числа.
    """
    thresholds = thresholds or {}
    positive = {f: p >= thresholds.get(f, DEFAULT_THRESHOLD) for f, p in flag_probs.items()}
    prob = 1.0 - float(np.prod([1.0 - min(max(p, 0.0), 1.0) for p in flag_probs.values()])) if flag_probs else 0.0
    if any(positive.values()):                       # порог пройден → вероятность не ниже 0,5, чтобы класс и prob не спорили
        prob = max(prob, 0.5)
    near = [f for f, p in flag_probs.items() if abs(p - thresholds.get(f, DEFAULT_THRESHOLD)) < REVIEW_BAND]
    return Verdict(
        quality_class=int(any(positive.values())),
        violation_type=violation_string(positive),
        quality_prob=round(min(max(prob, 0.0), 1.0), 4),
        flag_probs=dict(flag_probs),
        needs_review=bool(near),
        review_flags=near,
    )
