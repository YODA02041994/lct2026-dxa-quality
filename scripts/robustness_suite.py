#!/usr/bin/env python3
"""Эксп. 27: устойчивость к снимкам «другого аппарата» (бонус по словам постановщика).

40 снимков обучающего набора (20 поясничный отдел, 20 бедро) → 10 искажённых копий в формате DICOM → тот же конвейер.
Сравниваем с исходным снимком: область, класс по ТЗ, уровень выдачи, вероятности критериев, измерения в сантиметрах.
  масштаб 0,75 и 1,5 с тегом PixelSpacing (другой размер пикселя, тег честный); масштаб 1,5 без тега (масштаб по анатомии);
  гамма 0,7 и 1,4; контраст 0,8; шум σ = 5; размытие σ = 1; 12 бит; MONOCHROME1 (инвертированная шкала).
Ничего не подбирается: искажения заданы заранее, результат — таблица согласия.
    PYTHONPATH=src python scripts/robustness_suite.py [--n 20]   → data/work/robustness.json, таблица в stdout
"""
import argparse, copy, json, os, random, sys, tempfile
import numpy as np
import cv2
import pydicom
from pydicom.uid import ExplicitVRLittleEndian
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from dxaqc.infer import run                     # noqa: E402
from dxaqc.labels import PIXEL_MM_EXPORT        # noqa: E402
from dxaqc.predict import load_default_predictor  # noqa: E402

VARIANTS = ["исходный", "масштаб 0,75 + тег", "масштаб 1,5 + тег", "масштаб 1,5 без тега", "гамма 0,7", "гамма 1,4",
            "контраст 0,8", "шум σ 5", "размытие σ 1", "12 бит", "MONOCHROME1"]


def write(ds0, arr, path, spacing=None, bits=8, mono1=False):
    ds = copy.deepcopy(ds0)
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.Rows, ds.Columns = arr.shape
    if bits == 8:
        ds.BitsAllocated, ds.BitsStored, ds.HighBit = 8, 8, 7
        data = arr.astype(np.uint8)
    else:
        ds.BitsAllocated, ds.BitsStored, ds.HighBit = 16, 12, 11
        data = arr.astype(np.uint16)
    ds.PixelRepresentation, ds.SamplesPerPixel = 0, 1
    ds.PhotometricInterpretation = "MONOCHROME1" if mono1 else "MONOCHROME2"
    for tag in ("PixelSpacing", "ImagerPixelSpacing"):
        if tag in ds:
            delattr(ds, tag)
    if spacing:
        ds.PixelSpacing = [f"{spacing:.4f}", f"{spacing:.4f}"]
    ds.PixelData = data.tobytes()
    ds.SOPInstanceUID = pydicom.uid.generate_uid()
    ds.save_as(path, write_like_original=False)


def variants(ds, a):
    h, w = a.shape; base = PIXEL_MM_EXPORT.get(w, 0.6); f = a.astype(np.float32)
    rs = lambda k: cv2.resize(a, (int(round(w * k)), int(round(h * k))), interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_CUBIC)  # noqa: E731
    yield "исходный", a, {}
    yield "масштаб 0,75 + тег", rs(0.75), {"spacing": base / 0.75}
    yield "масштаб 1,5 + тег", rs(1.5), {"spacing": base / 1.5}
    yield "масштаб 1,5 без тега", rs(1.5), {}
    yield "гамма 0,7", np.clip(255 * (f / 255) ** 0.7, 0, 255), {}
    yield "гамма 1,4", np.clip(255 * (f / 255) ** 1.4, 0, 255), {}
    yield "контраст 0,8", np.clip(f.mean() + 0.8 * (f - f.mean()), 0, 255), {}
    yield "шум σ 5", np.clip(f + np.random.default_rng(7).normal(0, 5, f.shape), 0, 255), {}
    yield "размытие σ 1", cv2.GaussianBlur(a, (0, 0), 1.0), {}
    yield "12 бит", f * 16 + 30, {"bits": 12}
    yield "MONOCHROME1", 255 - f, {"mono1": True}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--src", default=os.path.join(ROOT, "data", "work", "rob_src")); a = ap.parse_args()
    files = sorted(os.listdir(a.src)); rnd = random.Random(27)
    pick = rnd.sample([f for f in files if "spine" in f], a.n) + rnd.sample([f for f in files if "hip" in f], a.n)
    tmp = tempfile.mkdtemp(prefix="dxaqc_rob_")
    for f in pick:
        ds = pydicom.dcmread(os.path.join(a.src, f)); arr = ds.pixel_array
        for k, (name, v, kw) in enumerate(variants(ds, arr)):
            d = os.path.join(tmp, f"v{k:02d}"); os.makedirs(d, exist_ok=True)
            write(ds, v, os.path.join(d, f), **kw)
    pred = load_default_predictor()
    res = {}
    for k, name in enumerate(VARIANTS):
        rows, _ = run(os.path.join(tmp, f"v{k:02d}"), os.path.join(tmp, f"o{k:02d}"), pred, fmt="csv")
        res[name] = {r["path_to_study"]: r for r in rows}
    base = res["исходный"]; out = {}
    print(f"{'искажение':24s} {'обработано':>10s} {'область':>8s} {'класс ТЗ':>9s} {'уровень':>8s} {'|Δp| мед.':>9s} {'|Δp| макс.':>10s} {'Δ см, мед.':>10s}")
    for name in VARIANTS:
        ok = reg = cls = lvl = 0; dps, dcm = [], []
        for f, r in res[name].items():
            b = base.get(f)
            if not b or r["processing_status"] != "Success":
                continue
            ok += 1; reg += r["anatomical_region"] == b["anatomical_region"]; cls += r["quality_class"] == b["quality_class"]; lvl += r["level"] == b["level"]
            dr, db = json.loads(r["details"] or "{}"), json.loads(b["details"] or "{}")
            for c, p in (db.get("probs") or {}).items():
                if c in (dr.get("probs") or {}):
                    dps.append(abs(dr["probs"][c] - p))
            for key in ("min_cm", "frame_h_cm", "top_gap_cm", "bottom_gap_cm"):
                if key in (db.get("features") or {}) and key in (dr.get("features") or {}):
                    dcm.append(abs(dr["features"][key] - db["features"][key]))
        n = len(pick)
        out[name] = {"processed": ok, "n": n, "region": reg, "class": cls, "level": lvl,
                     "dp_median": round(float(np.median(dps)), 3) if dps else None, "dp_max": round(float(np.max(dps)), 3) if dps else None,
                     "dcm_median": round(float(np.median(dcm)), 2) if dcm else None}
        o = out[name]
        print(f"{name:24s} {ok:>4d}/{n:<5d} {reg:>4d}/{n:<3d} {cls:>4d}/{n:<4d} {lvl:>4d}/{n:<3d} {str(o['dp_median']):>9s} {str(o['dp_max']):>10s} {str(o['dcm_median']):>10s}")
    json.dump(out, open(os.path.join(ROOT, "data", "work", "robustness.json"), "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
