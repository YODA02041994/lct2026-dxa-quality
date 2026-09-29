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
    if str(getattr(ds, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":      # кость тёмная → переворот шкалы
        arr = arr.max() - arr
    if arr.dtype != np.uint8:                                                            # 12–16 бит → шкала обучающих снимков (8 бит)
        a = arr.astype(np.float32)
        lo, hi = float(a.min()), float(np.percentile(a, 99.9))
        arr = np.clip((a - lo) / max(hi - lo, 1e-6) * 255.0, 0, 255).astype(np.uint8)
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
        region_by = "width"
        if region is None and hasattr(predictor, "guess_region"):      # нестандартная ширина кадра: область по локализаторам
            region, region_by = predictor.guess_region(img), "localizers"
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
        row["details"] = json.dumps({"probs": {k: round(v, 4) for k, v in verdict.flag_probs.items()}, "region_by": region_by,
                                     "needs_review": verdict.needs_review, "review_flags": verdict.review_flags,
                                     "comment": ex.get("comment", ""), "advice": ex.get("advice", []), "objects": ex.get("objects", []), "roi": ex.get("roi", {}),
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
MARKUP_COLUMNS = ["path_to_study", "anatomical_region", "roi", "polygon_px", "status"]


def markup_rows(rows: list[dict], review: dict | None = None) -> list[dict]:
    """Третий лист: предложенная разметка областей измерения (L1–L4, шейка бедра) и решение специалиста по ней.
    review — {path_to_study: {"status": "confirmed" | "rejected", "comment": str}} из веб-интерфейса; в пакетном режиме пусто."""
    by_path = {r["path_to_study"]: r for r in rows}
    out = []
    for r in rows:
        if r["processing_status"] != "Success":
            continue
        src = by_path.get(r["duplicate_of"], r) if r["duplicate_of"] else r
        roi = (json.loads(src["details"]) if src["details"] else {}).get("roi") or {}
        st = (review or {}).get(r["path_to_study"], {})
        status = {"confirmed": "подтверждено специалистом", "rejected": "отклонено специалистом"}.get(st.get("status"), "предложено, ждёт подтверждения")
        if st.get("comment"):
            status += f": {st['comment']}"
        for name, poly in roi.items():
            out.append({"path_to_study": r["path_to_study"], "anatomical_region": r["anatomical_region"], "roi": name,
                        "polygon_px": "; ".join(f"{x:.1f},{y:.1f}" for x, y in poly), "status": status})
    return out


def write_xlsx(rows: list[dict], path: str, review: dict | None = None) -> None:
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
    ws3 = wb.create_sheet("разметка")
    ws3.append(MARKUP_COLUMNS)
    for r in markup_rows(rows, review):
        ws3.append([r[c] for c in MARKUP_COLUMNS])
    wb.save(path)


def write_csv(rows: list[dict], path: str) -> None:
    """CSV: UTF-8 с BOM, разделитель «;», десятичная точка — параметры описаны в README."""
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, delimiter=";")
        w.writeheader()
        w.writerows(rows)


def write_overlays(rows: list[dict], root: str, output_dir: str, thresholds: dict) -> int:
    """Доп. серия: снимок с ориентирами, осью, рамками предметов и вердиктом — на каждый успешно обработанный файл.
    Два вида: PNG (overlays/) и DICOM Secondary Capture в том же исследовании (series/), одна серия на исследование."""
    from pydicom.uid import generate_uid
    from .sr import build_sc
    from .viz import draw_overlay, encode_png
    out = os.path.join(output_dir, "overlays"); os.makedirs(out, exist_ok=True); n = 0
    out_sc = os.path.join(output_dir, "series"); os.makedirs(out_sc, exist_ok=True)
    series: dict[str, tuple[str, int]] = {}                       # исследование → (SeriesInstanceUID, число снимков)
    by_path = {r["path_to_study"]: r for r in rows}
    for r in rows:
        if r["processing_status"] != "Success":
            continue
        src = by_path.get(r["duplicate_of"], r) if r["duplicate_of"] else r
        details = json.loads(src["details"]) if src["details"] else {}
        try:
            ds = pydicom.dcmread(os.path.join(root, r["path_to_study"]))
            img = _pixels(ds)
            bgr = draw_overlay(img, r["anatomical_region"], details, {"quality_class": r["quality_class"], "violation_type": r["violation_type"], "quality_prob": r[PROB_COLUMN]}, thresholds)
            base = r["path_to_study"].replace(os.sep, "__").rsplit(".", 1)[0]
            with open(os.path.join(out, base + ".png"), "wb") as f:
                f.write(encode_png(bgr))
            key = r["study_uid"] or r["study_dir"]
            uid, k = series.get(key, (generate_uid(), 0))
            series[key] = (uid, k + 1)
            build_sc(ds, bgr, r, uid, k + 1).save_as(os.path.join(out_sc, base + "_SC.dcm"), write_like_original=False)
            n += 1
        except Exception as exc:  # noqa: BLE001
            log.warning("оверлей не построен %s — %s", r["path_to_study"], exc)
    return n


def write_sr(rows: list[dict], root: str, output_dir: str) -> int:
    """DICOM SR с текстом заключения — по одному на успешный снимок (дубликаты получают свой SR со ссылкой на свой файл)."""
    from .sr import build_sr
    out = os.path.join(output_dir, "sr"); os.makedirs(out, exist_ok=True); n = 0
    by_path = {r["path_to_study"]: r for r in rows}
    for r in rows:
        if r["processing_status"] != "Success":
            continue
        src_row = by_path.get(r["duplicate_of"], r) if r["duplicate_of"] else r
        details = json.loads(src_row["details"]) if src_row["details"] else {}
        try:
            src = pydicom.dcmread(os.path.join(root, r["path_to_study"]), stop_before_pixels=True)
            sr = build_sr(src, r, details)
            sr.save_as(os.path.join(out, r["path_to_study"].replace(os.sep, "__").rsplit(".", 1)[0] + "_SR.dcm"), write_like_original=False)
            n += 1
        except Exception as exc:  # noqa: BLE001
            log.warning("SR не записан %s — %s", r["path_to_study"], exc)
    return n


def pack_series(output_dir: str) -> str:
    """Zip-архив с дополнительными сериями (ТЗ п. 2.7): DICOM-серия с разметкой, DICOM SR и PNG."""
    path = os.path.join(output_dir, "additional_series.zip")
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for sub in ("series", "sr", "overlays"):
            d = os.path.join(output_dir, sub)
            for name in sorted(os.listdir(d)) if os.path.isdir(d) else []:
                z.write(os.path.join(d, name), f"{sub}/{name}")
    return path


def run(input_path: str, output_dir: str, predictor: Predictor | None = None,
        fmt: str = "both", overlays: bool = False, sr: bool = False) -> tuple[list[dict], RunSummary]:
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
        if sr:
            log.info("DICOM SR записано: %d → %s", write_sr(rows, root, output_dir), os.path.join(output_dir, "sr"))
        if overlays or sr:
            log.info("архив дополнительных серий: %s", pack_series(output_dir))
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
