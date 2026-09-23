"""Локализатор анатомических ориентиров: тепловые карты поверх ResNet18.

Зачем: в закрытом тесте разметки нет. Сеть находит ориентиры на новом снимке, а критерии ТЗ считаются
по найденным точкам (docs/06). Две модели — позвоночник (8 ориентиров) и бедро (11); левые бёдра
отражаются в «правые», чтобы модель видела одну ориентацию.

Выход сети — K тепловых карт (stride 4). Пик карты → координата; высота пика → уверенность;
низкий пик → «ориентира в кадре нет» (это тоже ответ: так определяются укладка и поле сканирования).
"""
from __future__ import annotations

import glob
import json
import math
import os
from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

SPINE_LM = ["th12", "L1", "L2", "L3", "L4", "L5", "crest_l", "crest_r"]
HIP_LM = ["head", "gt_top", "gt_lat", "neck_sup", "neck_inf", "lt_tip", "lt_up", "lt_down", "ischium", "shaft_top", "shaft_bot"]
LM = {"spine": SPINE_LM, "hip": HIP_LM}
SIZE = 320            # вход сети (снимки ~280–300 px — без потери разрешения)
STRIDE = 4
SIGMA = 1.6           # ширина гауссианы на карте stride 4 (≈ 4 мм на снимке)
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


# ----------------------------------------------------------------------------- данные
@dataclass
class Sample:
    image_id: str
    kind: str
    side: str | None                 # 'L' / 'R' для бедра
    img: np.ndarray                  # uint8 HxW, для бедра L — уже отражён в R
    pts: np.ndarray                  # K×2 в пикселях исходника (после отражения); NaN если нет
    state: np.ndarray                # K: 1 = точка есть, 0 = известно, что нет, -1 = неизвестно
    study: str


def _pick_annotation(files: list[dict]) -> dict | None:
    """Одна разметка на снимок: готовая, свежий протокол (labeler-4), затем разметчик по полному протоколу."""
    done = [a for a in files if a.get("done")]
    if not done:
        return None
    done.sort(key=lambda a: (a.get("tool", "") != "labeler-4", a["annotator"] != "Александр", a.get("updated_at", "")))
    return done[0]


def load_samples(ann_dir: str, png_dir: str, manifest_csv: str, kind: str) -> list[Sample]:
    import csv
    study = {}
    for r in csv.DictReader(open(manifest_csv, encoding="utf-8-sig")):
        iid = f"{int(r['num']):03d}_{'spine' if r['region'].startswith('Пояс') else 'hip' + r['side']}"
        study[iid] = r["study_dir"]
    by_img: dict[str, list[dict]] = {}
    for p in glob.glob(os.path.join(ann_dir, "*.json")):
        try:
            a = json.load(open(p, encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if a.get("kind") == kind:
            by_img.setdefault(a["image_id"], []).append(a)
    names = LM[kind]
    out: list[Sample] = []
    for iid, files in sorted(by_img.items()):
        a = _pick_annotation(files)
        if a is None:
            continue
        img = cv2.imread(os.path.join(png_dir, iid + ".png"), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        h, w = img.shape
        pts = np.full((len(names), 2), np.nan, np.float32)
        state = np.full(len(names), -1, np.int8)
        got = {}
        for p in a["points"]:
            got.setdefault(p["label"], []).append((p["x"], p["y"]))
        absent = set(a.get("absent", []))
        if kind == "spine" and "vb" in got:                 # старый формат без уровней
            vb = sorted(got.pop("vb"), key=lambda t: t[1])
            if len(vb) == 5 and "th12" in got:               # 5 тел под Th12 — почти наверняка L1..L5
                for k, v in zip(["L1", "L2", "L3", "L4", "L5"], vb):
                    got[k] = [v]
            # иначе уровни остаются «неизвестно» — в лоссе маскируются
        for i, k in enumerate(names):
            if k in got:
                pts[i] = got[k][0]
                state[i] = 1
            elif k in absent:
                state[i] = 0
        side = None
        if kind == "hip":
            side = iid[-1]
            if side == "L":                                   # нормализуем в «правое» бедро
                img = np.ascontiguousarray(img[:, ::-1])
                pts[:, 0] = (w - 1) - pts[:, 0]
        out.append(Sample(iid, kind, side, img, pts, state, study.get(iid, iid[:3])))
    return out


# ----------------------------------------------------------------------------- геометрия
def letterbox_matrix(h: int, w: int, size: int = SIZE) -> np.ndarray:
    """Аффинная матрица 2×3: исходник → квадрат size×size с сохранением пропорций (паддинг по центру)."""
    s = size / max(h, w)
    tx, ty = (size - w * s) / 2, (size - h * s) / 2
    return np.array([[s, 0, tx], [0, s, ty]], np.float32)


def apply_affine(pts: np.ndarray, M: np.ndarray) -> np.ndarray:
    out = pts.copy()
    ok = ~np.isnan(pts[:, 0])
    xy = np.concatenate([pts[ok], np.ones((ok.sum(), 1), np.float32)], axis=1)
    out[ok] = xy @ M.T
    return out


def invert_affine(M: np.ndarray) -> np.ndarray:
    return cv2.invertAffineTransform(M.astype(np.float64)).astype(np.float32)


def random_augment(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    """Небольшие масштаб/сдвиг/поворот — точки поворачиваются вместе с картинкой, критерии не портятся."""
    scale = rng.uniform(0.92, 1.08)
    angle = rng.uniform(-4, 4)
    cx, cy = w / 2 + rng.uniform(-0.05, 0.05) * w, h / 2 + rng.uniform(-0.05, 0.05) * h
    R = cv2.getRotationMatrix2D((cx, cy), angle, scale).astype(np.float32)
    return R


def make_input(img: np.ndarray, M: np.ndarray, size: int = SIZE, rng: np.random.Generator | None = None) -> np.ndarray:
    """uint8 HxW → float32 3×size×size (ImageNet-нормализация); с rng — яркостные аугментации и чёрные маски."""
    x = cv2.warpAffine(img, M, (size, size), flags=cv2.INTER_LINEAR, borderValue=0).astype(np.float32) / 255.0
    if rng is not None:
        gamma = rng.uniform(0.7, 1.4)
        x = np.clip(x, 0, 1) ** gamma
        x = np.clip(x * rng.uniform(0.85, 1.15) + rng.uniform(-0.05, 0.05), 0, 1)
        if rng.random() < 0.5:                                   # чёрные прямоугольники лаборанта по краям
            for _ in range(rng.integers(1, 3)):
                rw, rh = int(rng.uniform(0.08, 0.3) * size), int(rng.uniform(0.15, 0.6) * size)
                x0 = 0 if rng.random() < 0.5 else size - rw
                y0 = int(rng.uniform(0, size - rh))
                x[y0:y0 + rh, x0:x0 + rw] = 0
        x = np.clip(x + rng.normal(0, 0.01, x.shape), 0, 1).astype(np.float32)
    x3 = np.stack([x, x, x], axis=0)
    return (x3 - MEAN[:, None, None]) / STD[:, None, None]


def make_heatmaps(pts_in: np.ndarray, state: np.ndarray, size: int = SIZE, stride: int = STRIDE, sigma: float = SIGMA):
    """pts_in — точки уже в системе входа size×size. Возвращает (K×hs×hs карты, K веса лосса)."""
    hs = size // stride
    K = len(pts_in)
    hm = np.zeros((K, hs, hs), np.float32)
    wgt = np.ones(K, np.float32)
    ys, xs = np.mgrid[0:hs, 0:hs]
    for i in range(K):
        if state[i] == -1:
            wgt[i] = 0.0
            continue
        if state[i] == 0 or np.isnan(pts_in[i, 0]):
            continue                                           # известно, что нет → нулевая карта
        cx, cy = pts_in[i] / stride
        if not (0 <= cx < hs and 0 <= cy < hs):
            wgt[i] = 0.0                                       # аугментация вытолкнула точку за кадр
            continue
        hm[i] = np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma ** 2))
    return hm, wgt


# ----------------------------------------------------------------------------- модель
class HeatmapNet(nn.Module):
    """ResNet18 (ImageNet) → FPN-lite → K тепловых карт на stride 4."""

    def __init__(self, k: int, pretrained: bool = True):
        super().__init__()
        r = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool)   # stride 4, 64
        self.l1, self.l2, self.l3, self.l4 = r.layer1, r.layer2, r.layer3, r.layer4   # 64/128/256/512
        c = 96
        self.lat1, self.lat2, self.lat3, self.lat4 = nn.Conv2d(64, c, 1), nn.Conv2d(128, c, 1), nn.Conv2d(256, c, 1), nn.Conv2d(512, c, 1)
        self.smooth = nn.Sequential(nn.Conv2d(c, c, 3, padding=1), nn.BatchNorm2d(c), nn.ReLU(inplace=True),
                                    nn.Conv2d(c, c, 3, padding=1), nn.BatchNorm2d(c), nn.ReLU(inplace=True))
        self.head = nn.Conv2d(c, k, 1)
        nn.init.constant_(self.head.bias, -2.0)

    def forward(self, x):
        x = self.stem(x)
        f1 = self.l1(x)
        f2 = self.l2(f1)
        f3 = self.l3(f2)
        f4 = self.l4(f3)
        p = self.lat4(f4)
        p = self.lat3(f3) + F.interpolate(p, size=f3.shape[-2:], mode="bilinear", align_corners=False)
        p = self.lat2(f2) + F.interpolate(p, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        p = self.lat1(f1) + F.interpolate(p, size=f1.shape[-2:], mode="bilinear", align_corners=False)
        return self.head(self.smooth(p))        # логиты; sigmoid → карта в [0,1]


def heatmap_loss(logits: torch.Tensor, target: torch.Tensor, wgt: torch.Tensor) -> torch.Tensor:
    """MSE по sigmoid-картам с усилением пикселей у пика (иначе фон перевешивает)."""
    prob = torch.sigmoid(logits)
    per_px = (prob - target) ** 2 * (1 + 20 * target)
    per_ch = per_px.mean(dim=(2, 3))                       # B×K
    return (per_ch * wgt).sum() / wgt.sum().clamp(min=1)


def decode(logits: torch.Tensor, stride: int = STRIDE) -> tuple[np.ndarray, np.ndarray]:
    """Логиты B×K×h×w → координаты B×K×2 в системе входа и уверенность B×K (высота пика, 0..1)."""
    prob = torch.sigmoid(logits).detach().cpu().numpy()
    B, K, H, W = prob.shape
    xy = np.zeros((B, K, 2), np.float32)
    conf = np.zeros((B, K), np.float32)
    for b in range(B):
        for k in range(K):
            m = prob[b, k]
            idx = int(m.argmax())
            y, x = divmod(idx, W)
            conf[b, k] = m[y, x]
            dx = dy = 0.0                                  # субпиксельное уточнение по соседям
            if 0 < x < W - 1:
                dx = 0.5 * (m[y, x + 1] - m[y, x - 1]) / max(1e-6, (m[y, x + 1] - 2 * m[y, x] + m[y, x - 1])) * -1
            if 0 < y < H - 1:
                dy = 0.5 * (m[y + 1, x] - m[y - 1, x]) / max(1e-6, (m[y + 1, x] - 2 * m[y, x] + m[y - 1, x])) * -1
            dx, dy = float(np.clip(dx, -1, 1)), float(np.clip(dy, -1, 1))
            xy[b, k] = ((x + dx) * stride, (y + dy) * stride)
    return xy, conf


def _top_peaks(m: np.ndarray, k: int = 4, min_dist: int = 4):
    """k локальных максимумов карты (y, x, value), не ближе min_dist друг к другу."""
    m = m.copy()
    out = []
    for _ in range(k):
        idx = int(m.argmax())
        y, x = divmod(idx, m.shape[1])
        if m[y, x] <= 0.02:
            break
        out.append((y, x, float(m[y, x])))
        y0, y1, x0, x1 = max(0, y - min_dist), y + min_dist + 1, max(0, x - min_dist), x + min_dist + 1
        m[y0:y1, x0:x1] = 0
    return out


def decode_spine_levels(logits: torch.Tensor, names: list[str], stride: int = STRIDE, mm_per_px: float = 0.6 * (300 / SIZE),
                        xy: np.ndarray | None = None, conf: np.ndarray | None = None):
    """Поверх обычного decode: L1..L5 обязаны идти сверху вниз с шагом 20–45 мм (на карте stride 4 — в пикселях карты).
    Перебор по 4 лучшим пикам каждого уровня (4^5 = 1024 комбинаций) — максимум суммы уверенностей при соблюдении порядка.
    Если ни одна комбинация не проходит, остаётся обычный argmax."""
    prob = torch.sigmoid(logits).detach().cpu().numpy()[0]
    idx = [names.index(k) for k in ("L1", "L2", "L3", "L4", "L5")]
    cands = [_top_peaks(prob[i]) for i in idx]
    if any(len(c) == 0 for c in cands):
        return xy, conf
    lo, hi = 20 / mm_per_px / stride, 45 / mm_per_px / stride       # допустимый шаг между соседними уровнями, в пикселях карты
    best, best_score = None, -1.0
    import itertools
    for combo in itertools.product(*cands):
        ys = [c[0] for c in combo]
        if all(lo <= ys[i + 1] - ys[i] <= hi for i in range(4)):
            score = sum(c[2] for c in combo)
            if score > best_score:
                best, best_score = combo, score
    if best is None:
        return xy, conf
    xy, conf = xy.copy(), conf.copy()
    for i, (y, x, v) in zip(idx, best):
        xy[0, i] = (x * stride, y * stride)
        conf[0, i] = v
    return xy, conf


class Localizer:
    """Готовая модель для инференса: снимок → точки в пикселях исходника + уверенность (+ «нет в кадре»)."""

    def __init__(self, kind: str, weights: str, device: str | None = None, absent_thr: float = 0.3):
        self.kind, self.names = kind, LM[kind]
        self.device = device or ("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
        ck = torch.load(weights, map_location="cpu")
        sds = ck.get("state_dicts") or [ck["state_dict"]]
        self.nets = []
        for sd in sds:
            n = HeatmapNet(len(self.names), pretrained=False)
            n.load_state_dict({k: v.float() for k, v in sd.items()})
            self.nets.append(n.to(self.device).eval())
        self.absent_thr = ck.get("absent_thr", absent_thr)
        self.size = ck.get("size", SIZE)

    def _logits(self, x: torch.Tensor) -> torch.Tensor:
        """Ансамбль: среднее sigmoid-карт → обратно в логиты (decode ждёт логиты)."""
        prob = torch.stack([torch.sigmoid(n(x)) for n in self.nets]).mean(0).clamp(1e-4, 1 - 1e-4)
        return torch.log(prob / (1 - prob))

    @torch.no_grad()
    def predict(self, img: np.ndarray, side: str | None = None) -> dict:
        """img uint8 HxW исходника. Для бедра side='L' → отражение туда и обратно."""
        h, w = img.shape
        flip = self.kind == "hip" and side == "L"
        if flip:
            img = np.ascontiguousarray(img[:, ::-1])
        M = letterbox_matrix(h, w, self.size)
        x = torch.from_numpy(make_input(img, M, self.size))[None].to(self.device)
        logits = self._logits(x)
        xy, conf = decode(logits)
        if self.kind == "spine":
            xy, conf = decode_spine_levels(logits, self.names, mm_per_px=0.6 / M[0, 0], xy=xy, conf=conf)
        pts = apply_affine(xy[0], invert_affine(M))
        if flip:
            pts[:, 0] = (w - 1) - pts[:, 0]
        out = {}
        for i, k in enumerate(self.names):
            c = float(conf[0, i])
            out[k] = {"x": float(pts[i, 0]), "y": float(pts[i, 1]), "conf": c, "present": c >= self.absent_thr}
        return out
