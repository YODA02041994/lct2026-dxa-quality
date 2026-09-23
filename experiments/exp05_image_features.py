#!/usr/bin/env python3
"""Эксп. 05: признаки прямо по картинке там, где точки сети шумят (ось диафиза, гребни, посторонние предметы).

Идея: сеть ставит точки с ошибкой ~5 мм, а угол по двум точкам на коротком отрезке от этого рассыпается
(корреляция с врачом r = 0,05 у диафиза). Робастная оценка по массе кости в окне вокруг точек устойчивее.
Печатает сырые AUC признаков против меток экспертов и корреляцию с признаками по точкам врача.
    PYTHONPATH=src python experiments/exp05_image_features.py
"""
import json, math, os, sys
import cv2, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.join(ROOT, "scripts"))
from train_criteria import manifest, human_landmarks, WORK, PNG  # noqa: E402
from dxaqc.labels import pixel_mm  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402


def robust_axis_deg(m: np.ndarray, min_rows=15):
    """Наклон к вертикали прямой через центры масс по строкам бинарной маски (20 % худших строк отброшены)."""
    ys, xs = [], []
    w = m.shape[1]
    for r in range(m.shape[0]):
        row = m[r]
        if row.sum() >= 3:
            ys.append(r); xs.append(float((row * np.arange(w)).sum() / row.sum()))
    if len(ys) < min_rows:
        return None
    ys, xs = np.array(ys, float), np.array(xs, float)
    k = np.polyfit(ys, xs, 1)[0]
    res = xs - (k * ys + np.median(xs - k * ys))
    keep = np.abs(res) < np.percentile(np.abs(res), 80)
    if keep.sum() >= 10:
        k = np.polyfit(ys[keep], xs[keep], 1)[0]
    return math.degrees(math.atan(k))


def shaft_axis_from_image(img, st, sb, half=22):
    """Ось диафиза бедра: масса кости в полосе ±half px вокруг отрезка shaft_top→shaft_bot (точки — только затравка)."""
    h, w = img.shape
    y0, y1 = int(max(0, min(st[1], sb[1]))), int(min(h, max(st[1], sb[1]) + 1))
    if y1 - y0 < 15:
        return None
    band = img[y0:y1].astype(np.float32)
    nz = band[band > 0]
    if nz.size < 50:
        return None
    thr = np.percentile(nz, 60)
    m = (band >= thr).astype(np.float32)
    for r in range(y1 - y0):                                   # окно скользит вдоль отрезка
        t = r / max(1, y1 - y0 - 1)
        cx = st[0] + (sb[0] - st[0]) * t
        lo, hi = int(max(0, cx - half)), int(min(w, cx + half + 1))
        m[r, :lo] = 0; m[r, hi:] = 0
    return robust_axis_deg(m)


def bottom_bone_width(img, frac=0.10):
    """Доля колонок с костью в нижних 10 % строк (гребни в кадре → широкая яркая полоса) / ширина позвоночника в середине."""
    h, w = img.shape
    nz = img[img > 0]
    if nz.size < 100:
        return 0.0, 0.0
    thr = np.percentile(nz, 70)
    bot = img[int(h * (1 - frac)):]
    mid = img[int(h * 0.4):int(h * 0.6)]
    bw = float((bot.max(0) >= thr).mean())
    mw = float((mid.max(0) >= thr).mean())
    return bw, bw / max(mw, 1e-3)


def bright_blobs(img, thr=250, amin=2, amax=400):
    """Число и площадь мелких насыщенных пятен (металл ярче кости)."""
    m = (img >= thr).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    areas = [s[cv2.CC_STAT_AREA] for s in stats[1:] if amin <= s[cv2.CC_STAT_AREA] <= amax]
    return float(len(areas)), float(sum(areas)), float((img >= thr).mean())


def auc(y, x):
    y, x = np.array(y), np.array(x, float)
    if len(set(y)) < 2 or x.std() < 1e-9:
        return float("nan")
    a = roc_auc_score(y, x)
    return a


man = manifest()
out = {}
for kind in ("hip", "spine"):
    hum = human_landmarks(kind)
    net = json.load(open(os.path.join(WORK, f"landmarks_oof_{kind}.json")))
    rows = []
    for iid, m in man.items():
        if m["kind"] != kind or not m["labeled"] or iid not in hum or iid not in net:
            continue
        img = cv2.imread(os.path.join(PNG, iid + ".png"), cv2.IMREAD_GRAYSCALE)
        r = {"id": iid, **{c: m[c] for c in (("hip_positioning_rotation", "hip_roi_field") if kind == "hip" else ("spine_positioning", "spine_axis_tilt", "spine_artifact"))}}
        if kind == "hip":
            for src, lm in (("h", hum[iid]), ("n", net[iid])):
                st, sb = lm.get("shaft_top"), lm.get("shaft_bot")
                if st and sb and st.get("present", True) and sb.get("present", True):
                    r[f"shaft_pts_{src}"] = abs(math.degrees(math.atan2(sb["x"] - st["x"], sb["y"] - st["y"])))
                    a = shaft_axis_from_image(img, (st["x"], st["y"]), (sb["x"], sb["y"]))
                    r[f"shaft_img_{src}"] = abs(a) if a is not None else None
        else:
            bw, ratio = bottom_bone_width(img)
            r["bottom_width"], r["bottom_ratio"] = bw, ratio
            for thr in (235, 245, 250, 254):
                n, area, frac = bright_blobs(img, thr)
                r[f"blobs{thr}_n"], r[f"blobs{thr}_area"], r[f"sat{thr}"] = n, area, frac
            r["crests_h"] = sum(1 for k in ("crest_l", "crest_r") if hum[iid].get(k, {}).get("present"))
        rows.append(r)
    out[kind] = rows
    print(f"\n=== {kind}: {len(rows)} снимков")
    if kind == "hip":
        y = [r["hip_positioning_rotation"] for r in rows]
        ok = [r for r in rows if r.get("shaft_img_h") is not None and r.get("shaft_img_n") is not None]
        yk = [r["hip_positioning_rotation"] for r in ok]
        print(f"  n с осью по картинке: {len(ok)} (+{sum(yk)})")
        for f in ("shaft_pts_h", "shaft_img_h", "shaft_pts_n", "shaft_img_n"):
            print(f"  {f:14} AUC {auc(yk, [r[f] for r in ok]):.3f}")
        a, b = np.array([r["shaft_pts_h"] for r in ok]), np.array([r["shaft_img_n"] for r in ok])
        c, d = np.array([r["shaft_img_h"] for r in ok]), np.array([r["shaft_pts_n"] for r in ok])
        print(f"  корреляция с углом по точкам врача: точки сети r={np.corrcoef(a, d)[0,1]:.2f} | картинка от точек сети r={np.corrcoef(a, b)[0,1]:.2f} | картинка от точек врача r={np.corrcoef(a, c)[0,1]:.2f}")
    else:
        yp = [r["spine_positioning"] for r in rows]; ya = [r["spine_artifact"] for r in rows if r["spine_artifact"] is not None]
        ra = [r for r in rows if r["spine_artifact"] is not None]
        yc = [int(r["crests_h"] < 2) for r in rows]
        print(f"  гребни (врач: не оба видны) n={sum(yc)}; укладка +{sum(yp)}; предметы +{sum(ya)}")
        for f in ("bottom_width", "bottom_ratio"):
            print(f"  {f:14} AUC vs укладка {auc(yp, [r[f] for r in rows]):.3f} | vs «гребни не оба» {auc(yc, [r[f] for r in rows]):.3f}")
        for thr in (235, 245, 250, 254):
            print(f"  порог {thr}: blobs_n AUC {auc(ya, [r[f'blobs{thr}_n'] for r in ra]):.3f} | area {auc(ya, [r[f'blobs{thr}_area'] for r in ra]):.3f} | доля насыщ. {auc(ya, [r[f'sat{thr}'] for r in ra]):.3f}")
json.dump(out, open(os.path.join(WORK, "exp05_features.json"), "w"), ensure_ascii=False)
