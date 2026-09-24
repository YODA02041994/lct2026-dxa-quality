#!/usr/bin/env python3
"""Эксп. 09 (клиницист): индексы поворота таза по запирательному отверстию. На снимке бедра медиальнее седалищной
кости видно запирательное отверстие — тёмный овал в кольце кости. При повороте таза его ширина меняется
(классический «obturator foramen index» на рентгене таза). Считаем: площадь, ширина/высота, положение относительно
головки — и сверяем с меткой «укладка бедра»."""
import csv, json, os, sys
import cv2, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.join(ROOT, "scripts"))
from train_criteria import oof_probs, best_f1
from dxaqc.labels import pixel_mm
from sklearn.metrics import roc_auc_score
WORK = os.path.join(ROOT, "data", "work"); PNG = os.path.join(WORK, "labeler", "png")
net = json.load(open(os.path.join(WORK, "landmarks_oof_hip.json")))

def foramen(img, lm, side):
    """Запирательное отверстие = не-кость, замкнутая костью (дырка в маске кости), медиальнее головки.
    Кость — порог Оцу по ненулевым пикселям; дырки — компоненты не-кости, не касающиеся края кадра."""
    h, w = img.shape
    X = (lambda p: (w - 1 - p["x"])) if side == "L" else (lambda p: p["x"])
    if side == "L": img = np.ascontiguousarray(img[:, ::-1])
    head = lm.get("head")
    if not head: return None
    hx, hy = X(head), head["y"]
    nz = img[img > 0]
    if nz.size < 100: return None
    thr, _ = cv2.threshold(nz.reshape(-1, 1).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bone = (img >= thr).astype(np.uint8)
    bone = cv2.morphologyEx(bone, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))       # замкнуть мелкие разрывы кольца
    nonbone = ((bone == 0) & (img > 0)).astype(np.uint8)
    n, lab, st, cen = cv2.connectedComponentsWithStats(nonbone, connectivity=4)
    best = None
    for i in range(1, n):
        a = st[i, cv2.CC_STAT_AREA]; x, y, bw, bh = st[i, cv2.CC_STAT_LEFT], st[i, cv2.CC_STAT_TOP], st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT]
        if a < 80 or a > 0.15 * h * w: continue
        if x <= 0 or y <= 0 or x + bw >= w or y + bh >= h: continue          # касается края → не замкнуто
        cx, cy = cen[i]
        if cx < hx: continue                                                  # латеральнее головки — не таз
        m = (lab == i).astype(np.uint8); ring = cv2.dilate(m, np.ones((5, 5), np.uint8)) - m
        enc = float(bone[ring > 0].mean()) if ring.sum() else 0.0
        if enc < 0.5: continue
        if best is None or a > best[1]: best = (enc, a, bw, bh, cx, cy)
    if best is None: return None
    mm = pixel_mm(w)
    enc, a, bw, bh, cx, cy = best
    return {"for_area_cm2": a * mm * mm / 100, "for_w_cm": bw * mm / 10, "for_h_cm": bh * mm / 10, "for_ratio": bw / max(1, bh),
            "for_dx_cm": (cx - hx) * mm / 10, "for_dy_cm": (cy - hy) * mm / 10, "for_enc": enc, "for_found": 1.0}

rows = []
for r in csv.DictReader(open(os.path.join(WORK, "manifest.csv"), encoding="utf-8-sig")):
    if not r["region"].startswith("Прокс") or r["labeled"] != "1" or r["hip_positioning_rotation"] == "": continue
    iid = f"{int(r['num']):03d}_hip{r['side']}"; img = cv2.imread(os.path.join(PNG, iid + ".png"), 0)
    f = foramen(img, net[iid], r["side"]) or {"for_area_cm2": 0, "for_w_cm": 0, "for_h_cm": 0, "for_ratio": 0, "for_dx_cm": 0, "for_dy_cm": 0, "for_enc": 0, "for_found": 0.0}
    rows.append({"id": iid, "y": int(r["hip_positioning_rotation"]), "g": r["study_dir"], **f})
y = np.array([r["y"] for r in rows]); g = np.array([r["g"] for r in rows])
print(f"бёдер {len(rows)}, отверстие найдено у {int(sum(r['for_found'] for r in rows))}, нарушений {y.sum()}")
for f in ("for_found", "for_area_cm2", "for_w_cm", "for_h_cm", "for_ratio", "for_dx_cm", "for_dy_cm", "for_enc"):
    x = np.array([r[f] for r in rows]); a = roc_auc_score(y, x); print(f"  {f:13} AUC {a:.3f}{'  ↓' if a < 0.5 else ''}")
found = [r for r in rows if r["for_found"]]
yf = np.array([r["y"] for r in found]); print(f"только найденные ({len(found)}, +{yf.sum()}):")
if len(found) > 10 and 0 < yf.sum() < len(yf):
    for f in ("for_area_cm2", "for_w_cm", "for_h_cm", "for_ratio", "for_dx_cm", "for_dy_cm"):
        print(f"  {f:13} AUC {roc_auc_score(yf, [r[f] for r in found]):.3f}")
json.dump({r["id"]: {k: r[k] for k in r if k.startswith("for_")} for r in rows}, open(os.path.join(WORK, "exp09_foramen.json"), "w"))
