"""Ось диафиза бедра без опоры на две точки: главная ось (PCA) связной области кости в нижней части кадра.
Затравка — только колонка shaft_bot у нижнего края (у сети ошибка 3,7 мм); длина сегмента задаётся долей кадра."""
import json, math, os, sys
import cv2, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.join(ROOT, "scripts"))
from train_criteria import manifest, human_landmarks, WORK, PNG
from sklearn.metrics import roc_auc_score

def pca_shaft(img, x_seed, rows_frac=0.30, pct=65):
    h, w = img.shape
    nz = img[img > 0]
    if nz.size < 50: return None
    y0 = int(h * (1 - rows_frac))
    m = (img[y0:] >= np.percentile(nz, pct)).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    # компонента, ближайшая к затравке в нижних 5 строках
    best, bd = None, 1e9
    for r in range(m.shape[0] - 1, max(0, m.shape[0] - 6), -1):
        xs = np.where(lab[r] > 0)[0]
        if xs.size:
            j = xs[np.argmin(np.abs(xs - x_seed))]
            if abs(j - x_seed) < bd: bd, best = abs(j - x_seed), lab[r, j]
    if best is None or bd > 25 or stats[best, cv2.CC_STAT_AREA] < 60: return None
    ys, xs = np.where(lab == best)
    pts = np.stack([xs, ys], 1).astype(np.float32)
    mean = pts.mean(0); cov = np.cov((pts - mean).T)
    ev, evec = np.linalg.eigh(cov)
    v = evec[:, np.argmax(ev)]
    return abs(math.degrees(math.atan2(v[0], v[1])))   # угол к вертикали

man = manifest(); hum = human_landmarks("hip"); net = json.load(open(os.path.join(WORK, "landmarks_oof_hip.json")))
rows = []
for iid, m in man.items():
    if m["kind"] != "hip" or not m["labeled"] or iid not in hum or iid not in net: continue
    img = cv2.imread(os.path.join(PNG, iid + ".png"), cv2.IMREAD_GRAYSCALE)
    hs, hb, nb = hum[iid].get("shaft_top"), hum[iid].get("shaft_bot"), net[iid]["shaft_bot"]
    r = {"id": iid, "y": m["hip_positioning_rotation"]}
    if hs and hb and hs["present"] and hb["present"]:
        r["h_deg"] = abs(math.degrees(math.atan2(hb["x"] - hs["x"], hb["y"] - hs["y"])))
    for frac in (0.22, 0.30, 0.40):
        for pct in (55, 65, 75):
            a = pca_shaft(img, nb["x"], frac, pct); r[f"pca_{frac}_{pct}"] = a
    rows.append(r)
ok = [r for r in rows if r.get("h_deg") is not None]
print(f"бёдер {len(rows)}, с углом врача {len(ok)}")
print("  вариант        n   AUC   r(врач)")
for frac in (0.22, 0.30, 0.40):
    for pct in (55, 65, 75):
        f = f"pca_{frac}_{pct}"
        sub = [r for r in rows if r.get(f) is not None]; subh = [r for r in ok if r.get(f) is not None]
        if len(sub) < 50: print(f"  {f:14} n={len(sub)} мало"); continue
        a = roc_auc_score([r["y"] for r in sub], [r[f] for r in sub])
        c = np.corrcoef([r["h_deg"] for r in subh], [r[f] for r in subh])[0, 1]
        print(f"  {f:14} {len(sub):3d}  {a:.3f}  {c:.2f}")
