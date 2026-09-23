"""CNN-второе мнение по критерию: ResNet18 → вероятность нарушения по всему снимку (ансамбль фолдов).

Используется там, где геометрия по ориентирам не работает: «посторонние предметы» (мелкие яркие объекты
у краёв) и как дополнительный признак для укладки бедра. Обучение — scripts/train_cnn_criterion.py.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision

from .landmarks import MEAN, STD, letterbox_matrix

SIZE = 320


def make_net(pretrained: bool = True) -> nn.Module:
    r = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    r.fc = nn.Linear(512, 1)
    return r


def to_input(img: np.ndarray, M: np.ndarray, rng: np.random.Generator | None = None, size: int = SIZE) -> np.ndarray:
    x = cv2.warpAffine(img, M, (size, size), flags=cv2.INTER_LINEAR, borderValue=0).astype(np.float32) / 255.0
    if rng is not None:
        x = np.clip(x, 0, 1) ** rng.uniform(0.7, 1.4)
        x = np.clip(x * rng.uniform(0.85, 1.15) + rng.uniform(-0.05, 0.05), 0, 1)
        if rng.random() < 0.5:                                    # чёрные прямоугольники лаборанта по краям
            for _ in range(rng.integers(1, 3)):
                rw, rh = int(rng.uniform(0.08, 0.3) * size), int(rng.uniform(0.15, 0.6) * size)
                x0 = 0 if rng.random() < 0.5 else size - rw
                y0 = int(rng.uniform(0, size - rh))
                x[y0:y0 + rh, x0:x0 + rw] = 0
    x3 = np.stack([x, x, x])
    return (x3 - MEAN[:, None, None]) / STD[:, None, None]


class CnnScorer:
    """Загружает weights/cnn_<criterion>.pt и отдаёт вероятность нарушения для снимка (левое бедро отражается)."""

    def __init__(self, weights: str, device: str | None = None):
        self.device = device or ("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
        ck = torch.load(weights, map_location="cpu")
        self.kind, self.criterion, self.size = ck["kind"], ck["criterion"], ck.get("size", SIZE)
        self.nets = []
        for sd in ck.get("state_dicts") or [ck["state_dict"]]:
            n = make_net(pretrained=False)
            n.load_state_dict({k: v.float() for k, v in sd.items()})
            self.nets.append(n.to(self.device).eval())

    @torch.no_grad()
    def score(self, img: np.ndarray, side: str | None = None) -> float:
        if self.kind == "hip" and side == "L":
            img = np.ascontiguousarray(img[:, ::-1])
        h, w = img.shape
        x = torch.from_numpy(to_input(img, letterbox_matrix(h, w, self.size), size=self.size))[None].to(self.device)
        return float(np.mean([torch.sigmoid(n(x))[0, 0].item() for n in self.nets]))
