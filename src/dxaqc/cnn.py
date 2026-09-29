"""CNN-второе мнение по критерию: ResNet18 → вероятность нарушения по всему снимку (ансамбль фолдов).

Используется там, где геометрия по ориентирам не работает: «посторонние предметы» (мелкие яркие объекты
у краёв) и как дополнительный признак для укладки бедра. Обучение — scripts/train_cnn_criterion.py.
"""
from __future__ import annotations

import os

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
    "cnn_artifact": [                                              # эксп. 12: предобучение на object-CXR (10 тыс. рентгенов с посторонними
        "spine_artifact_r18_320_ocxrF",                            # предметами, CC BY-NC 4.0) → дообучение на DXA; 3 семени: OOF 0,86/0,84/0,87,
        "spine_artifact_r18_320_ocxrF_s1",                         # ансамбль 0,87 (ImageNet-инициализация: 0,77 @320, 0,83 @512)
        "spine_artifact_r18_320_ocxrF_s2",
    ],
    "cnn_hip_pos": [                                               # эксп. 06: ансамбль «разных взглядов» на бедро → OOF 0,88
        "hip_positioning_rotation_eff_320",                        # EfficientNet-B0 по всему кадру (0,76)
        "hip_any_r18_320",                                         # ResNet18 по кадру на метке «любое нарушение бедра» (0,79)
        "hip_positioning_rotation_lt100",                          # ResNet18, вырезка 100 px вокруг малого вертела и шейки (0,82)
        "hip_positioning_rotation_lt100_e40",                      # то же, 40 эпох (0,86)
        "hip_positioning_rotation_isch100",                        # вырезка 100 px вокруг седалищной кости (0,83)
        "hip_positioning_rotation_prox170",                        # вырезка 170 px — весь проксимальный отдел (0,82)
        "hip_any_any_lt100",                                       # вырезка вертела на метке «любое нарушение» (0,85)
        # эксп. 16: DenseNet121 с весами рентгенограмм грудной клетки (TorchXRayVision) → ансамбль OOF 0,88 → 0,89
        "hip_positioning_rotation_lt100_xrv",                      # вырезка малого вертела (0,86)
        "hip_positioning_rotation_isch100_xrv",                    # вырезка седалищной кости (0,84)
        "hip_positioning_rotation_prox170_xrv",                    # проксимальный отдел (0,84)
    ],
}


ARCHS = ("resnet18", "resnet34", "resnet50", "convnext_tiny", "efficientnet_b0")


RADIMAGENET_MAP = {"backbone.0.": "conv1.", "backbone.1.": "bn1.", "backbone.4.": "layer1.", "backbone.5.": "layer2.",
                   "backbone.6.": "layer3.", "backbone.7.": "layer4."}


def load_radimagenet(net: nn.Module, path: str) -> int:
    """RadImageNet (BMEII-AI) ResNet50: ключи backbone.N.* → torchvision. Возвращает число загруженных тензоров."""
    sd = torch.load(path, map_location="cpu", weights_only=False)
    sd = sd.state_dict() if isinstance(sd, nn.Module) else sd.get("state_dict", sd)
    out = {}
    for k, v in sd.items():
        for a, b in RADIMAGENET_MAP.items():
            if k.startswith(a):
                out[b + k[len(a):]] = v
                break
    missing, unexpected = net.load_state_dict(out, strict=False)
    return len(out) - len(unexpected)


class XrvNet(nn.Module):
    """DenseNet121 с одним входным каналом, предобученный на рентгенограммах грудной клетки (веса torchxrayvision, эксп. 16).
    Сеть собирается из torchvision; библиотека torchxrayvision нужна только при обучении — взять исходные веса.
    Яркость на входе — в шкале [-1024; 1024], как при предобучении; голова — один логит."""

    def __init__(self, weights: str | None):
        super().__init__()
        f = torchvision.models.densenet121(weights=None).features
        f.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if weights:
            import torchxrayvision as xrv
            f.load_state_dict(xrv.models.DenseNet(weights=weights).features.state_dict())
        self.features = f
        self.fc = nn.Linear(1024, 1)

    def forward(self, x3: torch.Tensor) -> torch.Tensor:
        x = (x3[:, :1] * float(STD[0]) + float(MEAN[0])) * 2048.0 - 1024.0
        f = torch.relu(self.features(x))
        return self.fc(torch.flatten(nn.functional.adaptive_avg_pool2d(f, 1), 1))


def make_net(pretrained: bool = True, arch: str = "resnet18", init: str | None = None) -> nn.Module:
    """Классификатор с одним выходом (логит нарушения). init — путь к весам RadImageNet (только resnet50):
    инициализация с радиологических изображений вместо ImageNet (эксп. 11)."""
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
    elif arch.startswith("timm:"):                                  # любая сеть из timm: «timm:<имя>[@размер]» (ViT требует размер входа)
        import timm
        name, _, sz = arch[5:].partition("@")
        kw = {"img_size": int(sz)} if sz else {}
        r = timm.create_model(name, pretrained=pretrained, num_classes=1, **kw)
    elif arch.startswith("xrv:"):                                   # torchxrayvision: DenseNet121, предобученный на рентгенограммах грудной клетки
        r = XrvNet(arch[4:] if pretrained else None)
    else:
        raise ValueError(f"неизвестная архитектура {arch}")
    if init:
        sd = torch.load(init, map_location="cpu", weights_only=False)
        sd = sd.state_dict() if isinstance(sd, nn.Module) else sd.get("state_dict", sd)
        if any(k.startswith("backbone.") for k in sd):
            n = load_radimagenet(r, init)
        else:                                                     # наш же формат (torchvision), fc не берём
            sd = {k: v for k, v in sd.items() if not k.startswith(("fc.", "classifier."))}
            missing, unexpected = r.load_state_dict(sd, strict=False)
            n = len(sd) - len(unexpected)
        if n < 100:
            raise ValueError(f"веса {init} не легли на {arch}: загружено {n} тензоров")
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


CROP_LM = {
    "lt": ["lt_tip", "neck_inf", "shaft_top", "lt_up", "lt_down"],          # малый вертел и основание шейки (ротация)
    "neck": ["head", "neck_sup", "neck_inf", "gt_top"],                      # головка–шейка–большой вертел (укорочение шейки при ротации)
    "prox": ["head", "gt_top", "gt_lat", "neck_sup", "neck_inf", "lt_tip", "shaft_top"],   # весь проксимальный отдел
    "isch": ["ischium", "lt_down", "lt_tip"],                                # седалищная кость и низ вертела (поворот таза, приведение)
}


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
        self.tta = int(ck.get("tta", 1))                                   # варианты сдвига/масштаба при предсказании
        self.nets = []
        sds = ck.get("state_dicts") or [ck["state_dict"]]
        limit = int(os.environ.get("DXAQC_NETS_PER_FILE", "0"))    # экономный режим (слабый сервер): взять только первые N сетей из файла
        if limit > 0:
            sds = sds[:limit]
        for sd in sds:
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
        M0 = letterbox_matrix(h, w, self.size)
        mats = [M0]
        for k in range(self.tta - 1):
            s = 1.0 + 0.04 * (1 if k % 2 == 0 else -1) * (1 + k // 2)
            M = M0.copy(); M[0, 0] *= s; M[1, 1] *= s
            M[0, 2] += 0.03 * self.size * ((k % 4) - 1.5); M[1, 2] += 0.03 * self.size * (((k + 1) % 4) - 1.5)
            mats.append(M)
        xs = torch.from_numpy(np.stack([to_input(img, M, size=self.size) for M in mats])).to(self.device)
        return float(np.mean([torch.sigmoid(n(xs))[:, 0].mean().item() for n in self.nets]))


class ObjMapScorer:
    """Карта «где посторонний предмет» (ResNet18+FPN, обучена на масках object-CXR, эксп. 13; на DXA применяется без дообучения).
    features(img) → om_max_all (максимум карты), om_area (лог доли пикселей > 0,5 вне столба позвоночника), om_max (максимум вне столба);
    boxes — до 5 рамок самых ярких пятен для оверлея и details."""

    def __init__(self, weights: str, device: str | None = None):
        from .landmarks import HeatmapNet
        self.device = device or ("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
        ck = torch.load(weights, map_location="cpu", weights_only=False)
        self.size = int(ck.get("size", 320))
        self.net = HeatmapNet(1, pretrained=False)
        self.net.load_state_dict(ck["state_dict"]); self.net.to(self.device).eval()

    @torch.no_grad()
    def heatmap(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape
        M = letterbox_matrix(h, w, self.size)
        x = torch.from_numpy(to_input(img, M, size=self.size, masks=False))[None].to(self.device)
        hm = torch.sigmoid(self.net(x))[0, 0].cpu().numpy()
        hm = cv2.resize(hm, (self.size, self.size))
        return cv2.warpAffine(hm, cv2.invertAffineTransform(M), (w, h))

    def features(self, img: np.ndarray) -> tuple[dict, list]:
        hm = self.heatmap(img); h, w = img.shape
        nz = img[img > 0]
        f = {"om_max_all": float(hm.max()), "om_max": 0.0, "om_area": 0.0}
        if nz.size >= 100:
            thr = np.percentile(nz, 70); mid = img[int(h * 0.2):int(h * 0.8)]
            cx = int(np.argmax(np.convolve((mid >= thr).mean(0), np.ones(15) / 15, "same")))
            outside = np.ones((h, w), bool); outside[:, max(0, cx - int(w * 0.2)):min(w, cx + int(w * 0.2))] = False; outside &= img > 0
            if outside.any():
                f["om_max"] = float(hm[outside].max()); f["om_area"] = float(np.log1p(1000.0 * (hm[outside] > 0.5).mean()))
        boxes = []
        n, lab, st, _ = cv2.connectedComponentsWithStats((hm > 0.5).astype(np.uint8), connectivity=8)
        for i in range(1, n):
            x0, y0, bw, bh, a = st[i]
            if a < 6:
                continue
            boxes.append({"x": int(x0), "y": int(y0), "w": int(bw), "h": int(bh), "score": round(float(hm[y0:y0 + bh, x0:x0 + bw].max()), 2)})
        boxes.sort(key=lambda b: -b["score"])
        return f, boxes[:5]
