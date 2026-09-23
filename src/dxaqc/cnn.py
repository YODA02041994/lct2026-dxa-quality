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

# Признак критерия ← список обученных CNN (усредняются). Обучение критериев берёт data/work/cnn_oof_<имя>.json,
# инференс — weights/cnn_<имя>.pt. Ансамбль разных сетей/меток устойчивее одной (эксп. 05: бедро 0,74 → 0,80).
CNN_SOURCES = {
    "cnn_artifact": ["spine_artifact_r18_512"],                    # 512 px: тонкие линии видны лучше (OOF 0,83 против 0,76 на 320)
    "cnn_hip_pos": ["hip_positioning_rotation_eff_320",            # EfficientNet-B0 по всему кадру (0,76)
                    "hip_any_r18_320",                             # ResNet18 на метке «любое нарушение бедра» (0,79)
                    "hip_positioning_rotation_lt100"],             # ResNet18 на вырезке 100 px вокруг малого вертела и шейки (0,82)
}


ARCHS = ("resnet18", "resnet34", "resnet50", "convnext_tiny", "efficientnet_b0")


def make_net(pretrained: bool = True, arch: str = "resnet18") -> nn.Module:
    """Предобученный на ImageNet классификатор с одним выходом (логит нарушения)."""
    tm = torchvision.models
    if arch == "resnet18":
        r = tm.resnet18(weights=tm.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None); r.fc = nn.Linear(512, 1)
    elif arch == "resnet34":
        r = tm.resnet34(weights=tm.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None); r.fc = nn.Linear(512, 1)
    elif arch == "resnet50":
        r = tm.resnet50(weights=tm.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None); r.fc = nn.Linear(2048, 1)
    elif arch == "convnext_tiny":
        r = tm.convnext_tiny(weights=tm.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None); r.classifier[2] = nn.Linear(768, 1)
    elif arch == "efficientnet_b0":
        r = tm.efficientnet_b0(weights=tm.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None); r.classifier[1] = nn.Linear(1280, 1)
    else:
        raise ValueError(f"неизвестная архитектура {arch}")
    return r


def to_input(img: np.ndarray, M: np.ndarray, rng: np.random.Generator | None = None, size: int = SIZE, masks: bool = True) -> np.ndarray:
    """Кадр → вход сети. rng задан → аугментация (гамма, яркость, и — только если masks — чёрные прямоугольники
    лаборанта по краям, как на бедре; на позвоночнике посторонние предметы лежат у краёв, и маски их прятали бы)."""
    x = cv2.warpAffine(img, M, (size, size), flags=cv2.INTER_LINEAR, borderValue=0).astype(np.float32) / 255.0
    if rng is not None:
        x = np.clip(x, 0, 1) ** rng.uniform(0.7, 1.4)
        x = np.clip(x * rng.uniform(0.85, 1.15) + rng.uniform(-0.05, 0.05), 0, 1)
        if masks and rng.random() < 0.5:                          # чёрные прямоугольники лаборанта по краям
            for _ in range(rng.integers(1, 3)):
                rw, rh = int(rng.uniform(0.08, 0.3) * size), int(rng.uniform(0.15, 0.6) * size)
                x0 = 0 if rng.random() < 0.5 else size - rw
                y0 = int(rng.uniform(0, size - rh))
                x[y0:y0 + rh, x0:x0 + rw] = 0
    x3 = np.stack([x, x, x])
    return (x3 - MEAN[:, None, None]) / STD[:, None, None]


CROP_LM = {"lt": ["lt_tip", "neck_inf", "shaft_top", "lt_up", "lt_down"]}


def crop_around(img: np.ndarray, lm: dict, names: list[str], half: int, mirrored: bool = False) -> np.ndarray:
    """Квадрат 2·half вокруг среднего указанных ориентиров (нули за краем). Если снимок уже отражён (левое бедро),
    x ориентиров зеркалится. Ориентиры вне кадра (present=False) в центр не входят, если есть хоть один в кадре."""
    h, w = img.shape
    pts = [lm[k] for k in names if k in lm]
    pres = [p for p in pts if p.get("present", True)] or pts
    if not pres:
        return np.ascontiguousarray(img)
    cx = float(np.mean([(w - 1 - p["x"]) if mirrored else p["x"] for p in pres]))
    cy = float(np.mean([p["y"] for p in pres]))
    x0, y0 = int(round(cx - half)) + 2 * half, int(round(cy - half)) + 2 * half
    pad = np.pad(img, 2 * half, constant_values=0)
    return np.ascontiguousarray(pad[y0:y0 + 2 * half, x0:x0 + 2 * half])


class CnnEnsemble:
    """Несколько CnnScorer → среднее. Отсутствующие файлы весов пропускаются (с предупреждением в логе)."""

    def __init__(self, weights_dir: str, names: list[str], device: str | None = None):
        import logging
        import os
        self.scorers = []
        for n in names:
            p = os.path.join(weights_dir, f"cnn_{n}.pt")
            if os.path.exists(p):
                self.scorers.append(CnnScorer(p, device))
            else:
                logging.getLogger("dxaqc").warning("нет весов %s — признак считается без него", p)

    def __len__(self):
        return len(self.scorers)

    def score(self, img: np.ndarray, side: str | None = None, lm: dict | None = None) -> float:
        return float(np.mean([s.score(img, side, lm) for s in self.scorers])) if self.scorers else 0.0


class CnnScorer:
    """Загружает weights/cnn_<criterion>.pt и отдаёт вероятность нарушения для снимка (левое бедро отражается)."""

    def __init__(self, weights: str, device: str | None = None):
        self.device = device or ("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
        ck = torch.load(weights, map_location="cpu")
        self.kind, self.criterion, self.size = ck["kind"], ck["criterion"], ck.get("size", SIZE)
        self.arch = ck.get("arch", "resnet18")
        self.crop, self.half = ck.get("crop"), int(ck.get("half", 0))     # режим вырезки вокруг ориентиров
        self.nets = []
        for sd in ck.get("state_dicts") or [ck["state_dict"]]:
            n = make_net(pretrained=False, arch=self.arch)
            n.load_state_dict({k: v.float() for k, v in sd.items()})
            self.nets.append(n.to(self.device).eval())

    @torch.no_grad()
    def score(self, img: np.ndarray, side: str | None = None, lm: dict | None = None) -> float:
        mirrored = self.kind == "hip" and side == "L"
        if mirrored:
            img = np.ascontiguousarray(img[:, ::-1])
        if self.crop:
            if not lm:
                return 0.0
            img = crop_around(img, lm, CROP_LM[self.crop], self.half, mirrored)
        h, w = img.shape
        x = torch.from_numpy(to_input(img, letterbox_matrix(h, w, self.size), size=self.size))[None].to(self.device)
        return float(np.mean([torch.sigmoid(n(x))[0, 0].item() for n in self.nets]))
