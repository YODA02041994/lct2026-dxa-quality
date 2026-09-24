#!/usr/bin/env python3
"""Эксп. 13: детектор посторонних предметов как карта (ResNet18+FPN → тепловая карта stride 4), обучен на масках object-CXR
(прямоугольники/эллипсы/полигоны, 8000 снимков), затем применён к DXA-позвоночнику без дообучения: признаки = максимум и
площадь карты вне столба позвоночника; карта — оверлей «где предмет» для врача.
    PYTHONPATH=src python experiments/exp13_objectcxr_heatmap.py --epochs 6
"""
import argparse, csv, json, os, sys, time
import cv2, numpy as np, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.join(ROOT, "scripts"))
from dxaqc.cnn import to_input
from dxaqc.landmarks import HeatmapNet, letterbox_matrix
from dxaqc.criteria import artifact_features
from train_criteria import oof_probs, best_f1, manifest, features_for
from sklearn.metrics import roc_auc_score
ap = argparse.ArgumentParser(); ap.add_argument("--epochs", type=int, default=6); ap.add_argument("--size", type=int, default=320)
ap.add_argument("--data", default=os.path.expanduser("~/Downloads/lct_task4/object_cxr")); a = ap.parse_args()
S, ST = a.size, 4
dev = "mps" if torch.backends.mps.is_available() else "cpu"
sizes = {r["image_name"]: (int(r["w"]), int(r["h"])) for r in csv.DictReader(open(os.path.join(a.data, "sizes.csv")))}

def parse(ann, w, h, sc):
    """Маска объектов в масштабе ужатого снимка (sc = 320 / max(w, h))."""
    m = np.zeros((int(round(h * sc)), int(round(w * sc))), np.uint8)
    for o in (ann or "").split(";"):
        t = o.strip().split()
        if not t: continue
        typ, v = t[0], [float(x) * sc for x in t[1:]]
        if typ == "0" and len(v) >= 4: cv2.rectangle(m, (int(v[0]), int(v[1])), (int(v[2]), int(v[3])), 1, -1)
        elif typ == "1" and len(v) >= 4: cv2.ellipse(m, (int((v[0] + v[2]) / 2), int((v[1] + v[3]) / 2)), (max(1, int(abs(v[2] - v[0]) / 2)), max(1, int(abs(v[3] - v[1]) / 2))), 0, 0, 360, 1, -1)
        elif typ == "2" and len(v) >= 6: cv2.fillPoly(m, [np.array(v, np.int32).reshape(-1, 2)], 1)
    return m

def load_split(folder, csvname):
    items = []
    for r in csv.DictReader(open(os.path.join(a.data, folder, csvname))):
        p = os.path.join(a.data, folder, os.path.splitext(r["image_name"])[0] + ".jpg")
        if not os.path.exists(p) or r["image_name"] not in sizes: continue
        w, h = sizes[r["image_name"]]; sc = 320 / max(w, h)
        items.append((p, r["annotation"], w, h, sc))
    return items
train, val = load_split("train320", "train.csv"), load_split("dev320", "dev.csv")
print(f"object-CXR: train {len(train)}, val {len(val)} | {dev} | вход {S}, карта {S // ST}")
net = HeatmapNet(1).to(dev)
opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-3)
steps = a.epochs * int(np.ceil(len(train) / 16)); sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-4, total_steps=steps, pct_start=0.15)
rng = np.random.default_rng(0); cache = {}
def sample(it, aug):
    p, ann, w, h, sc = it
    if p not in cache: cache[p] = cv2.imread(p, 0)
    img = cache[p]; hh, ww = img.shape
    M = letterbox_matrix(hh, ww, S)
    if aug:
        s = rng.uniform(0.9, 1.1); M = M.copy(); M[0, 0] *= s; M[1, 1] *= s
        M[0, 2] += rng.uniform(-0.05, 0.05) * S; M[1, 2] += rng.uniform(-0.05, 0.05) * S
        if rng.random() < 0.5: M[0, 0] *= -1; M[0, 2] = S - M[0, 2]
    x = to_input(img, M, rng if aug else None, size=S, masks=False)
    mk = parse(ann, w, h, sc)
    mk = cv2.resize(mk, (ww, hh), interpolation=cv2.INTER_NEAREST) if mk.shape != (hh, ww) else mk
    t = cv2.warpAffine(mk.astype(np.float32), M, (S, S), flags=cv2.INTER_NEAREST, borderValue=0)
    t = cv2.resize(t, (S // ST, S // ST), interpolation=cv2.INTER_AREA)
    return x, (t > 0.3).astype(np.float32)
def batch(chunk, aug):
    xs, ts = zip(*[sample(it, aug) for it in chunk])
    return torch.from_numpy(np.stack(xs)).to(dev), torch.from_numpy(np.stack(ts))[:, None].to(dev)
pw = torch.tensor([8.0], device=dev)
t0 = time.time()
for ep in range(a.epochs):
    net.train(); order = rng.permutation(len(train)); tot = 0.0
    for i in range(0, len(order), 16):
        chunk = [train[j] for j in order[i:i + 16]]; x, t = batch(chunk, True)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(net(x), t, pos_weight=pw)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step(); tot += float(loss) * len(chunk)
    net.eval(); ys, ps = [], []
    with torch.no_grad():
        for i in range(0, len(val), 32):
            chunk = val[i:i + 32]; x, t = batch(chunk, False); h = torch.sigmoid(net(x))
            ps.extend(h.amax((1, 2, 3)).cpu().numpy().tolist()); ys.extend([int(bool(it[1].strip())) for it in chunk])
    print(f"  эпоха {ep + 1}/{a.epochs}: loss {tot / len(train):.3f} | val AUC (max карты → есть предмет) {roc_auc_score(ys, ps):.3f} | {time.time() - t0:.0f} с", flush=True)
out = os.path.join(ROOT, "weights", "objmap_objectcxr_r18fpn.pt")
torch.save({"state_dict": net.state_dict(), "size": S, "stride": ST, "source": "object-CXR heatmap"}, out); print("сохранено:", out)

# --- перенос на DXA (без дообучения): признаки карты вне столба позвоночника
man = manifest(); PNG = os.path.join(ROOT, "data", "work", "labeler", "png")
ids = sorted(i for i, m in man.items() if m["kind"] == "spine" and m["labeled"] and m["spine_artifact"] is not None)
y = np.array([man[i]["spine_artifact"] for i in ids]); g = np.array([man[i]["study"] for i in ids])
net_feats = features_for("spine", json.load(open(os.path.join(ROOT, "data", "work", "landmarks_oof_spine.json"))), man)
tf = np.array([net_feats[i]["th_frac40_log"] for i in ids])
F = {}; os.makedirs(os.path.join(ROOT, "docs", "img", "objmap"), exist_ok=True)
net.eval()
with torch.no_grad():
    for i in ids:
        img = cv2.imread(os.path.join(PNG, i + ".png"), 0); h, w = img.shape; M = letterbox_matrix(h, w, S)
        x = torch.from_numpy(to_input(img, M, size=S, masks=False))[None].to(dev)
        hm = torch.sigmoid(net(x))[0, 0].cpu().numpy(); hm = cv2.resize(hm, (S, S))
        inv = cv2.invertAffineTransform(M); hm_img = cv2.warpAffine(hm, inv, (w, h))
        nz = img[img > 0]; thr = np.percentile(nz, 70); mid = img[int(h * 0.2):int(h * 0.8)]
        cx = int(np.argmax(np.convolve((mid >= thr).mean(0), np.ones(15) / 15, "same")))
        outside = np.ones((h, w), bool); outside[:, max(0, cx - int(w * 0.2)):min(w, cx + int(w * 0.2))] = False; outside &= img > 0
        F[i] = {"om_max": float(hm_img[outside].max()), "om_area": float((hm_img[outside] > 0.5).mean()), "om_mean": float(hm_img[outside].mean()),
                "om_max_all": float(hm_img.max())}
        if y[ids.index(i)] == 1 or F[i]["om_max"] > 0.5:
            vis = cv2.addWeighted(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), 0.6, cv2.applyColorMap((hm_img * 255).astype(np.uint8), cv2.COLORMAP_JET), 0.4, 0)
            cv2.putText(vis, f"{i} y={y[ids.index(i)]} max={F[i]['om_max']:.2f}", (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imwrite(os.path.join(ROOT, "docs", "img", "objmap", f"{i}.png"), vis)
json.dump(F, open(os.path.join(ROOT, "data", "work", "objmap_spine.json"), "w"))
print("\nперенос на DXA (99 снимков позвоночника, метка «посторонние предметы», +17):")
for k in ("om_max", "om_area", "om_mean", "om_max_all"):
    print(f"  {k:10} сырой AUC {roc_auc_score(y, [F[i][k] for i in ids]):.3f}")
om = np.array([F[i]["om_max"] for i in ids]); oa = np.array([np.log1p(1000 * F[i]["om_area"]) for i in ids])
for name, X in (("top-hat", tf[:, None]), ("top-hat + om_max", np.stack([tf, om], 1)), ("top-hat + om_max + om_area", np.stack([tf, om, oa], 1))):
    p = oof_probs(X, y, g); print(f"  LR {name:28} AUC {roc_auc_score(y, p):.3f} F1 {best_f1(y, p)[0]:.3f}")
