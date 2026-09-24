#!/usr/bin/env python3
"""Grad-CAM для CNN-членов ансамбля по бедру: куда смотрит сеть на вырезке у малого вертела. Картинки для презентации
(docs/img/, вне git). Берём финальную сеть из weights/cnn_hip_positioning_rotation_lt100_e40.pt и точки сети (OOF).
    PYTHONPATH=src python scripts/gradcam_examples.py [n_pos n_neg]
"""
import csv, json, os, sys
import cv2, numpy as np, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from dxaqc.cnn import CROP_LM, crop_around, make_net, to_input
from dxaqc.landmarks import letterbox_matrix
WORK = os.path.join(ROOT, "data", "work"); PNG = os.path.join(WORK, "labeler", "png"); OUT = os.path.join(ROOT, "docs", "img", "gradcam")
os.makedirs(OUT, exist_ok=True)
ck = torch.load(os.path.join(ROOT, "weights", "cnn_hip_positioning_rotation_lt100_e40.pt"), map_location="cpu")
net = make_net(pretrained=False, arch=ck.get("arch", "resnet18")); net.load_state_dict({k: v.float() for k, v in ck["state_dict"].items()}); net.eval()
size, half = ck.get("size", 224), ck.get("half", 50)
acts, grads = {}, {}
net.layer4.register_forward_hook(lambda m, i, o: acts.__setitem__("a", o))
net.layer4.register_full_backward_hook(lambda m, gi, go: grads.__setitem__("g", go[0]))
lms = json.load(open(os.path.join(WORK, "landmarks_oof_hip.json")))
n_pos, n_neg = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (6, 6)
rows = [r for r in csv.DictReader(open(os.path.join(WORK, "manifest.csv"), encoding="utf-8-sig")) if r["region"].startswith("Прокс") and r["labeled"] == "1" and r["hip_positioning_rotation"] != ""]
pos = [r for r in rows if r["hip_positioning_rotation"] == "1"][:n_pos]; neg = [r for r in rows if r["hip_positioning_rotation"] == "0"][:n_neg]
tiles = []
for r in pos + neg:
    iid = f"{int(r['num']):03d}_hip{r['side']}"; img = cv2.imread(os.path.join(PNG, iid + ".png"), 0)
    mirrored = r["side"] == "L"
    if mirrored: img = np.ascontiguousarray(img[:, ::-1])
    crop = crop_around(img, lms[iid], CROP_LM["lt"], half, mirrored)
    h, w = crop.shape
    x = torch.from_numpy(to_input(crop, letterbox_matrix(h, w, size), size=size))[None].requires_grad_(True)
    logit = net(x)[0, 0]; net.zero_grad(); logit.backward()
    a, g = acts["a"][0], grads["g"][0]
    cam = torch.relu((g.mean((1, 2), keepdim=True) * a).sum(0)).detach().numpy()
    cam = cv2.resize(cam / (cam.max() + 1e-6), (size, size))
    base = cv2.warpAffine(crop, letterbox_matrix(h, w, size), (size, size))
    heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    vis = cv2.addWeighted(cv2.cvtColor(base, cv2.COLOR_GRAY2BGR), 0.6, heat, 0.4, 0)
    p = float(torch.sigmoid(logit))
    cv2.putText(vis, f"{iid} y={r['hip_positioning_rotation']} p={p:.2f}", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(os.path.join(OUT, f"{iid}_gradcam.png"), vis); tiles.append(vis)
cols = 6
while len(tiles) % cols: tiles.append(np.zeros_like(tiles[0]))
grid = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
cv2.imwrite(os.path.join(OUT, "gradcam_grid.jpg"), grid, [cv2.IMWRITE_JPEG_QUALITY, 75])
print(f"сохранено {len(pos) + len(neg)} картинок и сетка → {OUT}/gradcam_grid.jpg ({grid.shape})")
