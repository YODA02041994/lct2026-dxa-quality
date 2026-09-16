"""Чтение DICOM-исследований: обход папок, группировка, дедупликация.

Факты про выданные данные, на которые опирается этот модуль (см. docs/02_Данные_аудит.md):
- имя папки исследования ≠ StudyInstanceUID внутри файлов → храним оба идентификатора;
- одно и то же изображение лежит копиями в нескольких сериях → дедуп по хэшу пикселей;
- BodyPartExamined / ViewPosition / Laterality пустые → область определяем по картинке
  (и, если есть, по суффиксу имени файла _ПОП/_ППОБ/_ЛПОБ как подсказке).
"""
from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass, field
from typing import Iterator

import numpy as np
import pydicom

# Суффиксы имён файлов в "Для теста.zip": подсказка, а не источник истины.
_SUFFIX_REGION = {
    "ПОП": ("spine", None),
    "ППОБ": ("hip", "R"),
    "ЛПОБ": ("hip", "L"),
}
_SUFFIX_RE = re.compile(r"_(ПОП|ППОБ|ЛПОБ)\.dcm$", re.IGNORECASE)


@dataclass
class DicomImage:
    path: str
    study_dir: str            # имя папки исследования (ключ разметки в xlsx)
    study_uid: str            # StudyInstanceUID из тегов
    sop_uid: str              # SOPInstanceUID
    series_uid: str
    rows: int
    cols: int
    pixel_hash: str
    hint_region: str | None = None   # из суффикса имени файла
    hint_side: str | None = None
    exposed_area_mm: tuple[float, float] | None = None
    duplicate_of: str | None = None  # path первого файла с тем же pixel_hash

    def load_pixels(self) -> np.ndarray:
        ds = pydicom.dcmread(self.path, force=True)
        arr = ds.pixel_array
        if arr.ndim == 3:
            arr = arr[..., 0]
        return arr


@dataclass
class Study:
    study_dir: str
    root: str
    images: list[DicomImage] = field(default_factory=list)

    @property
    def unique_images(self) -> list[DicomImage]:
        return [im for im in self.images if im.duplicate_of is None]


def _pixel_hash(ds: pydicom.Dataset) -> str:
    arr = ds.pixel_array
    return hashlib.md5(arr.tobytes()).hexdigest()


def iter_dicom_files(root: str) -> Iterator[str]:
    for dirpath, _, files in os.walk(root):
        for f in sorted(files):
            if f.lower().endswith(".dcm"):
                yield os.path.join(dirpath, f)


def read_image(path: str, study_dir: str) -> DicomImage:
    ds = pydicom.dcmread(path, force=True)
    hint_region = hint_side = None
    m = _SUFFIX_RE.search(os.path.basename(path))
    if m:
        hint_region, hint_side = _SUFFIX_REGION[m.group(1).upper()]
    ea = getattr(ds, "ExposedArea", None)
    exposed = None
    if ea is not None:
        try:
            exposed = (float(ea[0]), float(ea[1]))
            if exposed == (0.0, 0.0):
                exposed = None
        except (TypeError, IndexError, ValueError):
            exposed = None
    return DicomImage(
        path=path,
        study_dir=study_dir,
        study_uid=str(getattr(ds, "StudyInstanceUID", "")),
        sop_uid=str(getattr(ds, "SOPInstanceUID", "")),
        series_uid=str(getattr(ds, "SeriesInstanceUID", "")),
        rows=int(getattr(ds, "Rows", 0)),
        cols=int(getattr(ds, "Columns", 0)),
        pixel_hash=_pixel_hash(ds),
        hint_region=hint_region,
        hint_side=hint_side,
        exposed_area_mm=exposed,
    )


def discover_studies(root: str) -> list[Study]:
    """Собрать исследования из корня в любой структуре.

    Дерево `root/<study>/series_*/…/*.dcm` → исследование = папка первого уровня.
    Плоская папка `root/*.dcm` (как в «Для теста») → исследование = сама папка.
    """
    root = os.path.abspath(root)
    studies: dict[str, Study] = {}
    for path in iter_dicom_files(root):
        rel = os.path.relpath(path, root)
        parts = rel.split(os.sep)
        study_dir = parts[0] if len(parts) > 1 else os.path.basename(root)
        st = studies.setdefault(study_dir, Study(study_dir=study_dir, root=root))
        st.images.append(read_image(path, study_dir))
    for st in studies.values():
        seen: dict[str, str] = {}
        for im in st.images:
            if im.pixel_hash in seen:
                im.duplicate_of = seen[im.pixel_hash]
            else:
                seen[im.pixel_hash] = im.path
    return sorted(studies.values(), key=lambda s: s.study_dir)
