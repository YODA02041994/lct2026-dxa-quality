"""Посторонние предметы по картинке: тонкие яркие линии/дуги вне столба позвоночника (цилиндр верхнего top-hat)
и «серая вуаль» плотной одежды (яркость фона). Монтаж 17 положительных: дуги у верхних углов, «y» у края, вуаль (087)."""
import csv, os, sys
import cv2, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from sklearn.metrics import roc_auc_score
PNG = os.path.join(ROOT, "data", "work", "labeler", "png")

def feats(img):
    h, w = img.shape
    nz = img[img > 0]; thr = np.percentile(nz, 70)
    mid = img[int(h * 0.2):int(h * 0.8)]
    cols = np.convolve((mid >= thr).mean(0), np.ones(15) / 15, "same")
    cx = int(np.argmax(cols))
    outside = np.ones((h, w), bool); outside[:, max(0, cx - int(w * 0.2)):min(w, cx + int(w * 0.2))] = False
    outside &= img > 0
    f = {}
    for k in (5, 9):
        th = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        r = th[outside].astype(np.float32)
        for t in (40, 60, 90):
            f[f"th{k}_frac{t}"] = float((r > t).mean()) if r.size else 0.0
        f[f"th{k}_max"] = float(r.max()) if r.size else 0.0
        f[f"th{k}_p999"] = float(np.percentile(r, 99.9)) if r.size else 0.0
        # линейные компоненты: вытянутые связные области сильного отклика
        m = ((th > 50) & outside).astype(np.uint8)
        n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        lines = 0; larea = 0
        for i in range(1, n):
            a = st[i, cv2.CC_STAT_AREA]; bw, bh = st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT]
            if a >= 12 and max(bw, bh) / max(1, min(bw, bh)) >= 2.5:
                lines += 1; larea += a
        f[f"th{k}_lines"] = float(lines); f[f"th{k}_linearea"] = float(larea)
    bg = img[int(h * 0.2):int(h * 0.8)][outside[int(h * 0.2):int(h * 0.8)]]
    f["bg_median"] = float(np.median(bg)) if bg.size else 0.0
    f["bg_p10"] = float(np.percentile(bg, 10)) if bg.size else 0.0
    f["bg_std"] = float(bg.std()) if bg.size else 0.0
    return f

rows = []
for r in csv.DictReader(open(os.path.join(ROOT, "data", "work", "manifest.csv"), encoding="utf-8-sig")):
    if not r["region"].startswith("Пояс") or r["labeled"] != "1" or r["spine_artifact"] == "": continue
    iid = f"{int(r['num']):03d}_spine"
    img = cv2.imread(os.path.join(PNG, iid + ".png"), 0)
    rows.append({"id": iid, "y": int(r["spine_artifact"]), **feats(img)})
y = [r["y"] for r in rows]
print(f"снимков {len(rows)}, предметов {sum(y)}")
for f in sorted(rows[0].keys() - {"id", "y"}):
    x = [r[f] for r in rows]
    if np.std(x) < 1e-9: continue
    a = roc_auc_score(y, x); print(f"  {f:16} AUC {a:.3f}{'  ↓' if a < 0.5 else ''}")
print("\nпо снимкам (позитивы): th5_frac60, th5_lines, bg_median")
for r in rows:
    if r["y"]: print(f"  {r['id']}  {r['th5_frac60']:.4f}  {r['th5_lines']:3.0f}  {r['bg_median']:5.1f}")
print("негативы с самым большим th5_frac60:")
for r in sorted([r for r in rows if not r["y"]], key=lambda r: -r["th5_frac60"])[:6]:
    print(f"  {r['id']}  {r['th5_frac60']:.4f}  {r['th5_lines']:3.0f}  {r['bg_median']:5.1f}")
