"""FleXray (MIT CSAIL, 2026; лицензия весов CC-BY-NC-4.0) — сегментация 60 анатомических структур на рентгенограммах.

Необязательный модуль-бонус. Нужен пакет `flexray` и локальная копия флагманской модели в weights/flexray/
(model.safetensors, config.yml, label_schema.json, preprocessing.json). Без них модуль молча выключен,
и конвейер работает как прежде. Сеть не обращается в интернет: HF_HUB_OFFLINE=1.

Что берём на денситометрии (проверено на 252 снимках обучающего набора, docs/05, эксп. 23):
- голос за анатомическую область: площадь масок позвонков L1–L5 против площади бедренной кости — 252 из 252 верно;
- площадь таза в кадре поясничного отдела — признак для критерия укладки позвоночника (ROC-AUC 0,925 отдельно);
- контуры позвонков и таза для показа на снимке.
Чего НЕ берём: центроиды позвонков (вдвое грубее нашего локализатора: медиана 5–6 мм против 2–3), ориентацию
позвонков для угла Кобба (при выраженном сколиозе маски на денситометрии сливаются), признаки бедра (AUC ≈ 0,5).
"""
from __future__ import annotations

import json
import logging
import os

import numpy as np

from .labels import REGION_HIP, REGION_SPINE

log = logging.getLogger("dxaqc.fxr")
KEEP = ["vertebra_t12", "vertebra_l1", "vertebra_l2", "vertebra_l3", "vertebra_l4", "vertebra_l5", "hips", "sacrum", "femurs"]
VERTEBRAE = ["vertebra_l1", "vertebra_l2", "vertebra_l3", "vertebra_l4", "vertebra_l5"]
MIN_AREA = 0.005              # меньше 0,5 % кадра — структуры нет


class FlexraySeg:
    """Обёртка над fxr.inference.FleXraySegmenter с кэшем последнего снимка (область и признаки считаются по одной сегментации)."""

    def __init__(self, weights_dir: str, device: str = "cpu"):
        self.ok = False
        self._cache: tuple[int, dict] | None = None
        self.path = os.path.join(weights_dir, "flexray")
        if not os.path.exists(os.path.join(self.path, "model.safetensors")):
            return
        try:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            from fxr.inference import FleXraySegmenter
            self.seg = FleXraySegmenter.from_pretrained(self.path, device=device)
            self.names = list(json.load(open(os.path.join(self.path, "label_schema.json")))["label_names"])
            self.idx = [self.names.index(k) for k in KEEP]
            self.ok = True
        except Exception as exc:  # noqa: BLE001 — модуль необязательный
            log.warning("FleXray выключен: %s", exc)

    def probs(self, img: np.ndarray) -> dict[str, np.ndarray]:
        """→ {структура: вероятности 0..1 в размере снимка} для структур KEEP."""
        key = hash(img.tobytes())
        if self._cache and self._cache[0] == key:
            return self._cache[1]
        import torch
        import torch.nn.functional as F
        from PIL import Image
        h, w = img.shape[:2]
        pil = Image.fromarray(img if img.dtype == np.uint8 else np.clip(img, 0, 255).astype(np.uint8)).convert("L")
        with torch.no_grad():
            pred = self.seg.predict(pil, tta_samples=1)
            p = pred.probabilities[0].float().cpu()[self.idx]
            side = max(h, w)                                          # сеть видит кадр, дополненный до квадрата
            p = F.interpolate(p[None], size=(side, side), mode="bilinear", align_corners=False)[0]
            top, left = (side - h) // 2, (side - w) // 2
            p = p[:, top:top + h, left:left + w].numpy()
        out = {k: p[i] for i, k in enumerate(KEEP)}
        self._cache = (key, out)
        return out

    def has_cached(self, img: np.ndarray) -> bool:
        """Сегментация этого снимка уже посчитана (FleXray спрашивали за область) — признаки можно взять бесплатно."""
        return bool(self._cache) and self._cache[0] == hash(img.tobytes())

    @staticmethod
    def _area(p: np.ndarray) -> float:
        return float((p >= 0.5).mean())

    def region_vote(self, img: np.ndarray) -> str | None:
        pr = self.probs(img)
        vert = sum(self._area(pr[k]) for k in VERTEBRAE)
        fem = self._area(pr["femurs"])
        if max(vert, fem) < MIN_AREA:
            return None
        return REGION_SPINE if vert > fem else REGION_HIP

    def spine_features(self, img: np.ndarray) -> dict[str, float]:
        pr = self.probs(img)
        return {"fxr_pelvis_area": round(self._area(pr["hips"]), 4),
                "fxr_vert_area": round(sum(self._area(pr[k]) for k in VERTEBRAE), 4),
                "fxr_th12_area": round(self._area(pr["vertebra_t12"]), 4)}

    def outlines(self, img: np.ndarray, names: list[str] | None = None) -> dict[str, list[list[int]]]:
        """Контуры масок (для показа): {структура: [[x, y], ...]} — внешний контур самой большой компоненты."""
        import cv2
        pr = self.probs(img)
        out = {}
        for k in names or (VERTEBRAE + ["vertebra_t12", "hips"]):
            m = (pr[k] >= 0.5).astype(np.uint8)
            if m.mean() < MIN_AREA:
                continue
            cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cs:
                c = max(cs, key=cv2.contourArea)
                out[k] = [[int(x), int(y)] for x, y in c[:, 0, :].tolist()][::2]
        return out
