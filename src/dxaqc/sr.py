"""DICOM SR (Basic Text SR) с текстом заключения по снимку — доп. функционал ТЗ (п. 6: «DICOM SR с текстом»).

Один SR-объект на каждый успешно обработанный снимок: ссылка на исходное изображение (StudyInstanceUID /
SOPInstanceUID), вердикт, официальные типы нарушений, протокол измерений, совет лаборанту, вероятности по критериям.
Читается любым DICOM-вьюером/PACS как структурированный отчёт; проверено чтением обратно pydicom.
"""
from __future__ import annotations

import datetime as dt
import json

import pydicom
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.sequence import Sequence
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from . import __version__

BASIC_TEXT_SR = "1.2.840.10008.5.1.4.1.1.88.11"


def _code(value: str, scheme: str, meaning: str) -> Dataset:
    d = Dataset(); d.CodeValue, d.CodingSchemeDesignator, d.CodeMeaning = value, scheme, meaning
    return d


def _text(name: str, value: str) -> Dataset:
    d = Dataset(); d.RelationshipType, d.ValueType = "CONTAINS", "TEXT"
    d.ConceptNameCodeSequence = Sequence([_code("DXAQC-" + name.upper()[:12], "99DXAQC", name)])
    d.TextValue = (value or "—")[:1024]
    return d


def build_sr(src: pydicom.Dataset, row: dict, details: dict) -> pydicom.FileDataset:
    """src — исходный DICOM снимка; row — строка результата; details — разобранный JSON колонки details."""
    now = dt.datetime.now()
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID, meta.MediaStorageSOPInstanceUID = BASIC_TEXT_SR, generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = pydicom.FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SpecificCharacterSet = "ISO_IR 192"                        # UTF-8: текст заключения по-русски
    ds.SOPClassUID, ds.SOPInstanceUID = BASIC_TEXT_SR, meta.MediaStorageSOPInstanceUID
    ds.Modality, ds.SeriesInstanceUID, ds.SeriesNumber, ds.InstanceNumber = "SR", generate_uid(), 999, 1
    for tag in ("PatientName", "PatientID", "PatientBirthDate", "PatientSex", "StudyInstanceUID", "StudyID", "StudyDate", "StudyTime", "AccessionNumber", "ReferringPhysicianName"):
        if tag in src:
            setattr(ds, tag, src.data_element(tag).value)
    if "StudyInstanceUID" not in ds:
        ds.StudyInstanceUID = generate_uid()
    ds.ContentDate, ds.ContentTime = now.strftime("%Y%m%d"), now.strftime("%H%M%S")
    ds.SeriesDescription = "DXA quality control (dxaqc)"
    ds.Manufacturer, ds.ManufacturerModelName, ds.SoftwareVersions = "LCT-2026 team", "dxaqc", __version__
    ds.CompletionFlag, ds.VerificationFlag = "COMPLETE", "UNVERIFIED"
    ds.ValueType = "CONTAINER"
    ds.ConceptNameCodeSequence = Sequence([_code("DXAQC-REPORT", "99DXAQC", "DXA acquisition quality report")])
    ds.ContinuityOfContent = "SEPARATE"
    # ссылка на исходный снимок
    ref = Dataset(); ref.ReferencedSOPClassUID = src.SOPClassUID; ref.ReferencedSOPInstanceUID = src.SOPInstanceUID
    series = Dataset(); series.SeriesInstanceUID = getattr(src, "SeriesInstanceUID", generate_uid()); series.ReferencedSOPSequence = Sequence([ref])
    study = Dataset(); study.StudyInstanceUID = ds.StudyInstanceUID; study.ReferencedSeriesSequence = Sequence([series])
    ds.CurrentRequestedProcedureEvidenceSequence = Sequence([study])
    verdict = "НАРУШЕНИЕ" if int(row.get("quality_class") or 0) == 1 else "НОРМА"
    items = [
        _text("Region", str(row.get("anatomical_region", ""))),
        _text("Verdict", f"{verdict} (quality_class={row.get('quality_class')}, quality_prob={row.get('quality_prob')})"),
        _text("Violations", str(row.get("violation_type") or "нет")),
        _text("Protocol", str(details.get("comment", ""))),
        _text("Advice", " ".join(details.get("advice", []) or [])),
        _text("Criteria", "; ".join(f"{k}={v:.2f}" for k, v in (details.get("probs") or {}).items())),
        _text("Review", "требует проверки врача" if details.get("needs_review") else "уверенное решение"),
        _text("Objects", json.dumps(details.get("objects", []), ensure_ascii=False)),
    ]
    ds.ContentSequence = Sequence(items)
    ds.is_little_endian, ds.is_implicit_VR = True, False
    return ds
