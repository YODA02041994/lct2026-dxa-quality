#!/usr/bin/env python3
"""Манифест уровня изображения: уникальный снимок → область, сторона, метки по критериям.

Разметка организатора дана на исследование (позвоночник / правое бедро / левое бедро), а файлы
названы CR00000N.dcm без указания области. Манифест восстанавливает связь «снимок → метки»:
  область — по ширине кадра (300 позвоночник; 280 и 248 бедро),
  сторона — по положению таза на снимке (src/dxaqc/region.py),
  истина  — из колонок критериев, НЕ из колонки «Итог» (правило организатора, 16.09).

Результат: data/work/manifest.csv (в git не хранится — производное от данных организатора).
    PYTHONPATH=src python scripts/make_manifest.py
"""
from __future__ import annotations

import csv
import os
import sys
import warnings

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from dxaqc.labels import HIP_FLAGS, REGION_HIP, REGION_SPINE, SPINE_FLAGS, read_labels  # noqa: E402
from dxaqc.loader import discover_studies  # noqa: E402
from dxaqc.region import assign_sides, hip_side, region_by_width  # noqa: E402

TRAIN = os.path.join(ROOT, "data", "work", "train", "Исследования")
XLSX = os.path.join(ROOT, "data", "work", "разметка.xlsx")
OUT = os.path.join(ROOT, "data", "work", "manifest.csv")

COLUMNS = ["num", "study_dir", "path", "study_uid", "sop_uid", "pixel_hash", "rows", "cols", "n_copies",
           "region", "side", "side_confidence", "labeled", "truth", "violation_type",
           *SPINE_FLAGS, *HIP_FLAGS, "expert_total", "comment"]


def main() -> int:
    studies = discover_studies(TRAIN)
    labels = {lb.study_dir: lb for lb in read_labels(XLSX)}
    rows = []
    for st in studies:
        lb = labels[st.study_dir]
        uniq = st.unique_images
        copies = {im.pixel_hash: sum(1 for x in st.images if x.pixel_hash == im.pixel_hash) for im in uniq}
        hips = [im for im in uniq if region_by_width(im.cols) == REGION_HIP]
        est = [hip_side(im.load_pixels()) for im in hips]
        sides = dict(zip([im.path for im in hips], assign_sides([e.upper_balance for e in est])))
        conf = {im.path: abs(e.upper_balance) for im, e in zip(hips, est)}
        for im in uniq:
            region = region_by_width(im.cols)
            side = sides.get(im.path, "")
            if region == REGION_SPINE:
                rl = lb.spine
            elif region == REGION_HIP:
                rl = lb.hip_r if side == "R" else lb.hip_l
            else:
                rl = None
            flags = dict(rl.flags) if rl else {}
            rows.append({
                "num": lb.num, "study_dir": st.study_dir, "path": os.path.relpath(im.path, ROOT),
                "study_uid": im.study_uid, "sop_uid": im.sop_uid, "pixel_hash": im.pixel_hash,
                "rows": im.rows, "cols": im.cols, "n_copies": copies[im.pixel_hash],
                "region": region or "", "side": side,
                "side_confidence": f"{conf[im.path]:.2f}" if im.path in conf else "",
                "labeled": int(bool(rl and rl.labeled)),
                "truth": "" if not rl or rl.truth is None else rl.truth,
                "violation_type": rl.violation_type if rl and rl.labeled else "",
                **{f: ("" if flags.get(f) is None else flags.get(f)) for f in (*SPINE_FLAGS, *HIP_FLAGS)},
                "expert_total": "" if not rl or rl.total is None else rl.total,
                "comment": lb.comment or "",
            })
    rows.sort(key=lambda r: (r["num"], r["region"], r["side"]))
    with open(OUT, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)

    # ---- сверка с разметкой организатора
    n = len(rows)
    spine = [r for r in rows if r["region"] == REGION_SPINE]
    hip = [r for r in rows if r["region"] == REGION_HIP]
    print(f"снимков в манифесте: {n} | позвоночник {len(spine)} | бедро {len(hip)} "
          f"(R {sum(r['side'] == 'R' for r in hip)}, L {sum(r['side'] == 'L' for r in hip)}) | "
          f"без области {n - len(spine) - len(hip)}")
    lab = [r for r in rows if r["labeled"]]
    print(f"размечено: {len(lab)} (позвоночник {sum(r['region'] == REGION_SPINE for r in lab)}, "
          f"R {sum(r['side'] == 'R' for r in lab)}, L {sum(r['side'] == 'L' for r in lab)}); "
          f"не размечено: {n - len(lab)}")
    print(f"с нарушением (истина из флагов): {sum(r['truth'] == 1 for r in lab)} из {len(lab)}")
    for f_ in (*SPINE_FLAGS, *HIP_FLAGS):
        print(f"   {f_}: {sum(r[f_] == 1 for r in rows)}")
    weak = [r for r in hip if r["side_confidence"] and float(r["side_confidence"]) < 0.15]
    print(f"бёдер со слабым признаком стороны (<0.15): {len(weak)} → №{sorted({r['num'] for r in weak})}")
    unl = [(r["num"], r["region"], r["side"], r["comment"]) for r in rows if not r["labeled"]]
    print("не размечены:", unl)
    print("→", os.path.relpath(OUT, ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
