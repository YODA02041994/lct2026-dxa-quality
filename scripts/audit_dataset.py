#!/usr/bin/env python3
"""Аудит выданного набора: файлы, дубликаты, теги, разметка, противоречия.

Воспроизводит цифры из docs/02_Данные_аудит.md. Запуск после unpack_data.py:
    PYTHONPATH=src python scripts/audit_dataset.py
"""
from __future__ import annotations

import collections
import os
import sys
import warnings

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import pydicom  # noqa: E402

from dxaqc.labels import HIP_FLAGS, SPINE_FLAGS, read_labels  # noqa: E402
from dxaqc.loader import discover_studies  # noqa: E402

TRAIN = os.path.join(ROOT, "data", "work", "train", "Исследования")
XLSX = os.path.join(ROOT, "data", "work", "разметка.xlsx")


def main() -> int:
    if not os.path.isdir(TRAIN):
        print("нет data/work/train — сначала python scripts/unpack_data.py", file=sys.stderr)
        return 1

    studies = discover_studies(TRAIN)
    labels = {lb.study_dir: lb for lb in read_labels(XLSX)}

    n_files = sum(len(s.images) for s in studies)
    n_uniq = sum(len(s.unique_images) for s in studies)
    print(f"исследований: {len(studies)} | файлов: {n_files} | уникальных изображений: {n_uniq}")
    per_study_files = collections.Counter(len(s.images) for s in studies)
    per_study_uniq = collections.Counter(len(s.unique_images) for s in studies)
    print(f"файлов на исследование: {dict(sorted(per_study_files.items()))}")
    print(f"уникальных на исследование: {dict(sorted(per_study_uniq.items()))}")

    # идентификаторы
    uid_match = sum(1 for s in studies if any(im.study_uid == s.study_dir for im in s.images))
    print(f"папка == StudyInstanceUID: {uid_match} из {len(studies)}")
    in_xlsx = sum(1 for s in studies if s.study_dir in labels)
    print(f"папок, найденных в xlsx: {in_xlsx} из {len(studies)}; строк xlsx: {len(labels)}")

    # дубликаты между исследованиями (утечка)
    h2s: dict[str, set[str]] = collections.defaultdict(set)
    for s in studies:
        for im in s.images:
            h2s[im.pixel_hash].add(s.study_dir)
    cross = sum(1 for v in h2s.values() if len(v) > 1)
    print(f"изображений, встречающихся в разных исследованиях: {cross}")

    # теги
    tag_stats: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    no_exposed = 0
    for s in studies:
        for im in s.images:
            ds = pydicom.dcmread(im.path, stop_before_pixels=True, force=True)
            for t in ("Modality", "ManufacturerModelName", "BitsStored", "PhotometricInterpretation",
                      "BodyPartExamined", "ViewPosition", "Laterality", "PixelSpacing",
                      "ImagerPixelSpacing"):
                tag_stats[t][str(getattr(ds, t, None))[:40]] += 1
            if im.exposed_area_mm is None:
                no_exposed += 1
    for t, c in tag_stats.items():
        print(f"  {t}: {dict(c.most_common(3))}")
    print(f"файлов без ExposedArea (0040,0303): {no_exposed} из {n_files}")
    sizes = collections.Counter((im.rows, im.cols) for s in studies for im in s.unique_images)
    print(f"размеры уникальных изображений (топ-5): {sizes.most_common(5)}")

    # разметка
    lbs = list(labels.values())
    print(f"\nисследований с хотя бы одним нарушением: {sum(1 for lb in lbs if lb.any_violation)} из {len(lbs)}")
    for region_name, getter, flags in (
        ("позвоночник", lambda lb: lb.spine, SPINE_FLAGS),
        ("правое бедро", lambda lb: lb.hip_r, HIP_FLAGS),
        ("левое бедро", lambda lb: lb.hip_l, HIP_FLAGS),
    ):
        regs = [getter(lb) for lb in lbs]
        labeled = [r for r in regs if r.labeled]
        pos = sum(1 for r in labeled if r.total == 1)
        print(f"{region_name}: размечено {len(labeled)}, с нарушением {pos}")
        for f in flags:
            n1 = sum(1 for r in labeled if r.flags.get(f) == 1)
            print(f"    {f}: {n1} положительных")

    print("\nпротиворечия «флаги ↔ Итог»:")
    for lb in lbs:
        for nm, r in (("spine", lb.spine), ("hip_r", lb.hip_r), ("hip_l", lb.hip_l)):
            if r.inconsistent:
                print(f"  №{lb.num} {lb.study_dir} [{nm}] flags={r.flags} total={r.total} // {lb.comment}")

    print("\nизображений больше, чем размеченных областей:")
    for s in studies:
        lb = labels.get(s.study_dir)
        if not lb:
            continue
        n_lab = sum(1 for r in (lb.spine, lb.hip_r, lb.hip_l) if r.labeled)
        if n_lab != len(s.unique_images):
            print(f"  №{lb.num} {s.study_dir}: изображений {len(s.unique_images)}, областей {n_lab} // {lb.comment}")

    comments = collections.Counter((lb.comment or "").lower().strip() for lb in lbs if lb.comment)
    print(f"\nкомментарии эксперта: {dict(comments.most_common())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
