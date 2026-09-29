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
import math
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
        mode = os.environ.get("DXAQC_MODE", "competition")             # competition (F1-оптимум) | sensitive (recall ≥ 0,9 по критерию)
        if mode != "competition" and mode in cj.get("modes", {}):
            self.thresholds = {c: float(t) for c, t in cj["modes"][mode].items()}
        self.mode = mode
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
        from .labels import pixel_mm
        from .roi import propose
        comment, advice = explain(region, feats, probs, self.thresholds)
        roi = propose("spine" if region == REGION_SPINE else "hip", lm, img, pixel_mm(w))   # предложение разметки областей измерения
        self._last = {"landmarks": lm, "features": feats, "probs": probs, "comment": comment, "advice": advice, "objects": boxes, "roi": roi}
        return probs

    def explain(self) -> dict:
        return dict(self._last)

    def guess_region(self, img: np.ndarray, floor: float = 0.30) -> str | None:
        """Область для кадра нестандартной ширины: у какого локализатора выше средняя уверенность ориентиров.
        На 252 снимках обучающего набора с изменённым масштабом правило верно в 252 случаях, наименьший отрыв 0,36;
        на кадрах без анатомии уверенность обоих локализаторов не выше 0,06 — такой кадр не поддерживается (None)."""
        from .region import hip_side
        cs = float(np.mean([p.get("conf", 0.0) for p in self.loc["spine"].predict(img).values()]))
        ch = float(np.mean([p.get("conf", 0.0) for p in self.loc["hip"].predict(img, hip_side(img).side).values()]))
        if max(cs, ch) < floor:
            return None
        return REGION_SPINE if cs > ch else REGION_HIP


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


def relative_prob(p: float, threshold: float) -> float:
    """Вероятность критерия в шкале, где его порог равен 0,5: logit(p') = logit(p) − logit(порог).
    Пороги критериев разные (0,48–0,89), поэтому без приведения критерий с высоким порогом завышал бы итог по снимку."""
    p = min(max(float(p), 1e-6), 1.0 - 1e-6)
    t = min(max(float(threshold), 1e-6), 1.0 - 1e-6)
    z = math.log(p / (1.0 - p)) - math.log(t / (1.0 - t))
    return 1.0 / (1.0 + math.exp(-z))


def aggregate(flag_probs: dict[str, float], thresholds: dict[str, float] | None = None) -> Verdict:
    """Вероятности критериев → итог по снимку.

    quality_class = 1, если хотя бы один критерий не ниже своего порога (правило организатора: любое
    нарушение → снимок с нарушением). quality_prob — наибольшая из вероятностей критериев, приведённых
    к общей шкале (порог критерия = 0,5). Поэтому quality_prob ≥ 0,5 тогда и только тогда, когда quality_class = 1.
    На снимках вне обучения это правило даёт ROC-AUC 0,857 против 0,849 у 1 − ∏(1 − p_i) (docs/08, эксп. 17).
    """
    thresholds = thresholds or {}
    positive = {f: p >= thresholds.get(f, DEFAULT_THRESHOLD) for f, p in flag_probs.items()}
    prob = max((relative_prob(p, thresholds.get(f, DEFAULT_THRESHOLD)) for f, p in flag_probs.items()), default=0.0)
    if any(positive.values()):                       # защита от округления на самой границе порога
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
