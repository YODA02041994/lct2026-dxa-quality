"""Пакетный инференс: папка или zip с DICOM → results.xlsx / results.csv в формате организатора.

Гарантии (ТЗ п. 2.7 и ответы постановщика 16–18.09):
  • строка на КАЖДЫЙ файл, включая дубликаты; дубликат считается один раз (по хэшу пикселей);
  • ни одного необработанного исключения: битый / не-DICOM файл → processing_status = Failure;
  • порядок строк детерминирован (сортировка по пути), результат воспроизводим.
"""
from __future__ import annotations

import csv
import json
import logging
import os
import shutil
import tempfile
import time
import zipfile
from dataclasses import dataclass

import numpy as np
import pydicom
from openpyxl import Workbook

from .labels import PROB_COLUMN, REGION_HIP
from .loader import _pixel_hash
from .predict import NullPredictor, Predictor, aggregate, load_default_predictor
from .region import hip_side, region_by_width

log = logging.getLogger("dxaqc")

# официальные колонки ТЗ (п. 2.5) + разрешённая quality_prob; затем наши служебные — проверке не мешают
OFFICIAL = ["path_to_study", "study_uid", "image_uid", "anatomical_region", "quality_class",
            "violation_type", PROB_COLUMN, "processing_status", "time_of_processing"]
SERVICE = ["study_dir", "side", "duplicate_of", "error_message", "details"]
COLUMNS = OFFICIAL + SERVICE


@dataclass
class RunSummary:
    files: int
    success: int
    failure: int
    unique_images: int
    seconds: float
    xlsx: str | None
    csv: str | None


def _decode_zip_name(name: str) -> str:
    for enc in ("utf-8", "cp866"):
        try:
            return name.encode("cp437").decode(enc)
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
    return name


def _extract_zip(zip_path: str, dst: str) -> None:
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            name = info.filename if info.flag_bits & 0x800 else _decode_zip_name(info.filename)
            target = os.path.normpath(os.path.join(dst, name))
            if not target.startswith(os.path.abspath(dst)):      # защита от ../ в архиве
                continue
            if info.is_dir():
                os.makedirs(target, exist_ok=True)
                continue
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with open(target, "wb") as f:
                f.write(zf.read(info))


def _is_dicom_candidate(path: str) -> bool:
    low = path.lower()
    if low.endswith((".dcm", ".dicom")):
        return True
    if os.path.splitext(low)[1]:          # другое расширение — не наш файл
        return False
    try:                                   # без расширения: смотрим сигнатуру DICM
        with open(path, "rb") as f:
            f.seek(128)
            return f.read(4) == b"DICM"
    except OSError:
        return False


def _list_files(root: str) -> list[str]:
    out = []
    for dirpath, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if not d.startswith((".", "__MACOSX")))
        for f in sorted(files):
            if f.startswith("."):
                continue
            p = os.path.join(dirpath, f)
            if _is_dicom_candidate(p):
                out.append(p)
    return sorted(out)


def _pixels(ds: pydicom.Dataset) -> np.ndarray:
    arr = ds.pixel_array
    if arr.ndim == 3:                      # RGB или многокадровый — берём первый канал/кадр
        arr = arr[..., 0] if arr.shape[-1] in (3, 4) else arr[0]
    if arr.ndim != 2:
        raise ValueError(f"неожиданная размерность пикселей: {arr.shape}")
    return arr


def process_file(path: str, root: str, predictor: Predictor, cache: dict) -> dict:
    t0 = time.perf_counter()
    rel = os.path.relpath(path, root)
    parts = rel.split(os.sep)
    row = {c: "" for c in COLUMNS}
    row["path_to_study"] = rel
    row["study_dir"] = parts[0] if len(parts) > 1 else os.path.basename(os.path.abspath(root))
    try:
        ds = pydicom.dcmread(path, force=False)
        row["study_uid"] = str(getattr(ds, "StudyInstanceUID", "") or "")
        row["image_uid"] = str(getattr(ds, "SOPInstanceUID", "") or "")
        img = _pixels(ds)
        region = region_by_width(int(img.shape[1]))
        if region is None:
            raise ValueError(f"unsupported_image: кадр {img.shape[0]}×{img.shape[1]} не похож на "
                             f"позвоночник (300 px) или бедро (280/248 px)")
        h = _pixel_hash(ds)
        if h in cache:
            verdict, side, first = cache[h]
            row["duplicate_of"] = first
        else:
            side = hip_side(img).side if region == REGION_HIP else ""
            verdict = aggregate(predictor.predict_flags(img, region, side or None), predictor.thresholds)
            cache[h] = (verdict, side, rel)
        ex = predictor.explain() if row["duplicate_of"] == "" else {}
        row["details"] = json.dumps({"probs": {k: round(v, 4) for k, v in verdict.flag_probs.items()},
                                     "needs_review": verdict.needs_review, "review_flags": verdict.review_flags,
                                     "comment": ex.get("comment", ""), "advice": ex.get("advice", []), "objects": ex.get("objects", []),
                                     "landmarks": {k: [round(v["x"], 1), round(v["y"], 1), round(v.get("conf", 1), 2)] for k, v in ex.get("landmarks", {}).items() if v.get("present")},
                                     "features": {k: round(float(v), 2) for k, v in ex.get("features", {}).items()}}, ensure_ascii=False) if ex else ""
        row.update({"anatomical_region": region, "side": side, "quality_class": verdict.quality_class,
                    "violation_type": verdict.violation_type, PROB_COLUMN: verdict.quality_prob,
                    "processing_status": "Success"})
    except Exception as exc:  # noqa: BLE001 — по ТЗ ни одна ошибка не должна уронить пакет
        row["processing_status"] = "Failure"
        row["error_message"] = f"{type(exc).__name__}: {str(exc)[:200]}"
        log.warning("Failure %s — %s", rel, row["error_message"])
    row["time_of_processing"] = round(time.perf_counter() - t0, 4)
    return row


def study_summary(rows: list[dict]) -> list[dict]:
    """Второй лист: одна строка на исследование — что переснять. Официальный лист остаётся первым и не меняется."""
    by: dict[str, list[dict]] = {}
    for r in rows:
        by.setdefault(r["study_dir"], []).append(r)
    out = []
    for study, rs in by.items():
        uniq = [r for r in rs if r["processing_status"] == "Success" and r["duplicate_of"] == ""]
        bad = [r for r in uniq if r["quality_class"] == 1]
        review = [r for r in uniq if r["details"] and json.loads(r["details"]).get("needs_review")]
        advice = []
        for r in bad:
            for a in json.loads(r["details"]).get("advice", []):
                if a not in advice:
                    advice.append(a)
        out.append({"study_dir": study, "files": len(rs), "images": len(uniq), "failures": sum(r["processing_status"] != "Success" for r in rs),
                    "violations": len(bad), "needs_review": len(review),
                    "regions_to_repeat": "; ".join(sorted({f"{r['anatomical_region']}{' ' + r['side'] if r['side'] else ''}: {r['violation_type']}" for r in bad})),
                    "advice": " ".join(advice)})
    return out


SUMMARY_COLUMNS = ["study_dir", "files", "images", "failures", "violations", "needs_review", "regions_to_repeat", "advice"]


def write_xlsx(rows: list[dict], path: str) -> None:
    wb = Workbook()
    ws = wb.active
    ws.title = "results"
    ws.append(COLUMNS)
    for r in rows:
        ws.append([r[c] if r[c] != "" else None for c in COLUMNS])
    ws2 = wb.create_sheet("по исследованиям")
    ws2.append(SUMMARY_COLUMNS)
    for r in study_summary(rows):
        ws2.append([r[c] for c in SUMMARY_COLUMNS])
    wb.save(path)


def write_csv(rows: list[dict], path: str) -> None:
    """CSV: UTF-8 с BOM, разделитель «;», десятичная точка — параметры описаны в README."""
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, delimiter=";")
        w.writeheader()
        w.writerows(rows)


def write_overlays(rows: list[dict], root: str, output_dir: str, thresholds: dict) -> int:
    """Доп. серия: PNG с ориентирами, осью, рамками предметов и вердиктом — по одному на успешный уникальный снимок."""
    from .viz import draw_overlay, encode_png
    out = os.path.join(output_dir, "overlays"); os.makedirs(out, exist_ok=True); n = 0
    by_path = {r["path_to_study"]: r for r in rows}
    for r in rows:
        if r["processing_status"] != "Success":
            continue
        src = by_path.get(r["duplicate_of"], r) if r["duplicate_of"] else r
        details = json.loads(src["details"]) if src["details"] else {}
        try:
            img = _pixels(pydicom.dcmread(os.path.join(root, r["path_to_study"])))
            bgr = draw_overlay(img, r["anatomical_region"], details, {"quality_class": r["quality_class"], "violation_type": r["violation_type"], "quality_prob": r[PROB_COLUMN]}, thresholds)
            name = r["path_to_study"].replace(os.sep, "__").rsplit(".", 1)[0] + ".png"
            with open(os.path.join(out, name), "wb") as f:
                f.write(encode_png(bgr))
            n += 1
        except Exception as exc:  # noqa: BLE001
            log.warning("оверлей не построен %s — %s", r["path_to_study"], exc)
    return n


def run(input_path: str, output_dir: str, predictor: Predictor | None = None,
        fmt: str = "both", overlays: bool = False) -> tuple[list[dict], RunSummary]:
    predictor = predictor or load_default_predictor()
    if isinstance(predictor, NullPredictor):
        log.warning("веса модели не найдены — работает заглушка NullPredictor (нарушений нет)")
    else:
        log.info("модель: %s", predictor.name)
    os.makedirs(output_dir, exist_ok=True)
    t0 = time.perf_counter()
    tmp = None
    try:
        root = input_path
        if os.path.isfile(input_path) and zipfile.is_zipfile(input_path):
            tmp = tempfile.mkdtemp(prefix="dxaqc_")
            _extract_zip(input_path, tmp)
            root = tmp
        if not os.path.isdir(root):
            raise FileNotFoundError(f"вход не найден или не папка/zip: {input_path}")
        cache: dict = {}
        rows = [process_file(p, root, predictor, cache) for p in _list_files(root)]
        if overlays:
            log.info("оверлеев записано: %d → %s", write_overlays(rows, root, output_dir, getattr(predictor, "thresholds", {})), os.path.join(output_dir, "overlays"))
    finally:
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)
    xlsx = os.path.join(output_dir, "results.xlsx") if fmt in ("xlsx", "both") else None
    csv_ = os.path.join(output_dir, "results.csv") if fmt in ("csv", "both") else None
    if xlsx:
        write_xlsx(rows, xlsx)
    if csv_:
        write_csv(rows, csv_)
    ok = sum(r["processing_status"] == "Success" for r in rows)
    return rows, RunSummary(files=len(rows), success=ok, failure=len(rows) - ok,
                            unique_images=len(cache), seconds=round(time.perf_counter() - t0, 2),
                            xlsx=xlsx, csv=csv_)
