"""Ось позвоночника: наклон ≠ сколиоз. Ложные тревоги по оси — сколиозы (030, 056, 063, 065, 075, 093, 095), у эксперта
это норма. Прямой наклонённый позвоночник (укладка косо) vs дуга: считаем кривизну и углы верхней/нижней половины."""
import csv, math, os, sys
import cv2, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from sklearn.metrics import roc_auc_score
PNG = os.path.join(ROOT, "data", "work", "labeler", "png")

def centroids(img):
    h, w = img.shape
    band = img[int(h * 0.15):int(h * 0.85)].astype(np.float32)
    nz = band[band > 0]
    if nz.size < 100: return None
    m = (band >= np.percentile(nz, 70)).astype(np.float32); m[:, :int(w * 0.3)] = 0; m[:, int(w * 0.7):] = 0
    ys, xs = [], []
    for r in range(m.shape[0]):
        if m[r].sum() >= 3: ys.append(r); xs.append(float((m[r] * np.arange(w)).sum() / m[r].sum()))
    if len(ys) < 20: return None
    return np.array(ys, float), np.array(xs, float)

def fit_angle(ys, xs):
    if len(ys) < 8: return 0.0
    k = np.polyfit(ys, xs, 1)[0]; res = xs - (k * ys + np.median(xs - k * ys))
    keep = np.abs(res) < np.percentile(np.abs(res), 80)
    if keep.sum() >= 6: k = np.polyfit(ys[keep], xs[keep], 1)[0]
    return math.degrees(math.atan(k))

def feats(img):
    c = centroids(img)
    if c is None: return {}
    ys, xs = c
    a = fit_angle(ys, xs)
    mid = np.median(ys)
    a_top, a_bot = fit_angle(ys[ys <= mid], xs[ys <= mid]), fit_angle(ys[ys > mid], xs[ys > mid])
    p2 = np.polyfit(ys, xs, 2); p1 = np.polyfit(ys, xs, 1)
    lin_res = np.std(xs - np.polyval(p1, ys)); quad_res = np.std(xs - np.polyval(p2, ys))
    yy = np.linspace(ys.min(), ys.max(), 50); bow = np.max(np.abs(np.polyval(p2, yy) - np.polyval(p1, yy)))
    n = ys.max() - ys.min()
    thirds = [fit_angle(ys[(ys >= ys.min() + n * i / 3) & (ys < ys.min() + n * (i + 1) / 3)], xs[(ys >= ys.min() + n * i / 3) & (ys < ys.min() + n * (i + 1) / 3)]) for i in range(3)]
    return {"axis": abs(a), "a_top": abs(a_top), "a_bot": abs(a_bot), "min_half": min(abs(a_top), abs(a_bot)), "d_half": abs(a_top - a_bot),
            "bow_px": bow, "lin_res": lin_res, "curv_gain": lin_res - quad_res, "min_third": min(abs(t) for t in thirds),
            "same_sign": float(np.sign(a_top) == np.sign(a_bot)), "tilt_straight": abs(a) if abs(a_top - a_bot) < 4 else 0.0,
            "tilt_minus_bow": abs(a) - 2.0 * bow / max(1, n) * 100}

rows = []
for r in csv.DictReader(open(os.path.join(ROOT, "data", "work", "manifest.csv"), encoding="utf-8-sig")):
    if not r["region"].startswith("Пояс") or r["labeled"] != "1" or r["spine_axis_tilt"] == "": continue
    iid = f"{int(r['num']):03d}_spine"; f = feats(cv2.imread(os.path.join(PNG, iid + ".png"), 0))
    if f: rows.append({"id": iid, "y": int(r["spine_axis_tilt"]), "sc": "сколиоз" in r["comment"].lower(), **f})
y = [r["y"] for r in rows]; print(f"снимков {len(rows)}, ось +{sum(y)}, сколиозов в комментариях {sum(r['sc'] for r in rows)}")
for f in ("axis", "a_top", "a_bot", "min_half", "d_half", "bow_px", "lin_res", "curv_gain", "min_third", "same_sign", "tilt_straight", "tilt_minus_bow"):
    x = [r[f] for r in rows]; print(f"  {f:14} AUC {roc_auc_score(y, x):.3f}   | сколиоз vs норма-без-сколиоза: {roc_auc_score([r['sc'] for r in rows if not r['y']], [r[f] for r in rows if not r['y']]):.3f}")
print("\nпозитивы (ось): axis a_top a_bot bow"); [print(f"  {r['id']} {r['axis']:5.1f} {r['a_top']:5.1f} {r['a_bot']:5.1f} {r['bow_px']:4.1f}") for r in rows if r["y"]]
print("сколиозы (норма): "); [print(f"  {r['id']} {r['axis']:5.1f} {r['a_top']:5.1f} {r['a_bot']:5.1f} {r['bow_px']:4.1f}") for r in rows if r["sc"] and not r["y"]]
