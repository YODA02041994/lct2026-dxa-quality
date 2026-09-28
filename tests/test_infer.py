"""Пакетный инференс: формат организатора, строка на файл, устойчивость к мусору, воспроизводимость."""
import os
import shutil
import sys
import zipfile

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from dxaqc.__main__ import main as cli_main  # noqa: E402
from dxaqc.infer import COLUMNS, OFFICIAL, run  # noqa: E402
from dxaqc.labels import REGION_HIP, REGION_SPINE, VIOLATION_TEXT  # noqa: E402
from dxaqc.predict import Predictor, aggregate  # noqa: E402

TEST_DIR = os.path.join(ROOT, "data", "work", "test", "Для теста")
TRAIN_DIR = os.path.join(ROOT, "data", "work", "train", "Исследования")
need_test = pytest.mark.skipif(not os.path.isdir(TEST_DIR), reason="сначала python scripts/unpack_data.py")
need_train = pytest.mark.skipif(not os.path.isdir(TRAIN_DIR), reason="сначала python scripts/unpack_data.py")


def test_official_columns_come_first_in_tz_order():
    assert OFFICIAL == ["path_to_study", "study_uid", "image_uid", "anatomical_region", "quality_class",
                        "violation_type", "quality_prob", "processing_status", "time_of_processing"]
    assert COLUMNS[:len(OFFICIAL)] == OFFICIAL


def test_aggregate_uses_official_wording_and_any_rule():
    v = aggregate({"spine_positioning": 0.1, "spine_axis_tilt": 0.9, "spine_artifact": 0.7})
    assert v.quality_class == 1
    assert v.violation_type == "Не выровнена ось позвоночника;Присутствуют посторонние предметы"
    assert v.quality_prob == 0.9            # наибольшая вероятность критерия в шкале, где порог равен 0,5
    clean = aggregate({"hip_positioning_rotation": 0.2, "hip_roi_field": 0.1})
    assert (clean.quality_class, clean.violation_type) == (0, "")
    assert clean.quality_prob < 0.5
    # разные пороги: вероятность снимка не ниже 0,5 тогда и только тогда, когда сработал хотя бы один критерий
    thr = {"hip_positioning_rotation": 0.53, "hip_roi_field": 0.89}
    assert aggregate({"hip_positioning_rotation": 0.40, "hip_roi_field": 0.80}, thr).quality_prob < 0.5
    hit = aggregate({"hip_positioning_rotation": 0.40, "hip_roi_field": 0.90}, thr)
    assert hit.quality_class == 1 and hit.quality_prob >= 0.5
    assert set(v.violation_type.split(";")) <= set(VIOLATION_TEXT.values())


@need_test
def test_test_folder_gives_three_success_rows(tmp_path):
    rows, s = run(TEST_DIR, str(tmp_path))
    assert (s.files, s.success, s.failure) == (3, 3, 0)
    assert sorted(r["anatomical_region"] for r in rows) == sorted([REGION_HIP, REGION_HIP, REGION_SPINE])
    assert all(r["study_uid"] and r["image_uid"] for r in rows)
    assert os.path.getsize(s.xlsx) > 0 and os.path.getsize(s.csv) > 0


@need_test
def test_zip_input_with_cyrillic_names(tmp_path):
    z = tmp_path / "in.zip"
    with zipfile.ZipFile(z, "w") as zf:
        for f in os.listdir(TEST_DIR):
            zf.write(os.path.join(TEST_DIR, f), arcname=f"Исследование/{f}")
    rows, s = run(str(z), str(tmp_path / "out"))
    assert s.files == 3 and s.failure == 0
    assert all(r["study_dir"] == "Исследование" for r in rows)


@need_test
def test_garbage_never_raises_and_is_reported_as_failure(tmp_path):
    import pydicom
    g = tmp_path / "in" / "study"
    g.mkdir(parents=True)
    good = os.path.join(TEST_DIR, sorted(os.listdir(TEST_DIR))[0])
    shutil.copy(good, g / "good.dcm")
    (g / "random.dcm").write_bytes(os.urandom(4000))
    (g / "text.dcm").write_text("не дайком")
    (g / "empty.dcm").write_bytes(b"")
    (g / "truncated.dcm").write_bytes(open(good, "rb").read()[:30000])
    ds = pydicom.dcmread(good)
    a = np.zeros((200, 512), dtype=np.uint8)
    ds.Rows, ds.Columns, ds.PixelData = 200, 512, a.tobytes()
    ds.save_as(str(g / "wrong_size.dcm"))
    (tmp_path / "in" / "notes.txt").write_text("не DICOM — игнорируется")
    (tmp_path / "in" / "empty_study").mkdir()

    rows, s = run(str(tmp_path / "in"), str(tmp_path / "out"))
    status = {os.path.basename(r["path_to_study"]): r["processing_status"] for r in rows}
    assert status == {"good.dcm": "Success", "random.dcm": "Failure", "text.dcm": "Failure",
                      "empty.dcm": "Failure", "truncated.dcm": "Failure", "wrong_size.dcm": "Failure"}
    for r in rows:
        if r["processing_status"] == "Failure":
            assert r["error_message"] and r["quality_class"] == "" and r["violation_type"] == ""
        assert isinstance(r["time_of_processing"], float)


def test_empty_folder_gives_empty_table(tmp_path):
    (tmp_path / "in").mkdir()
    rows, s = run(str(tmp_path / "in"), str(tmp_path / "out"))
    assert rows == [] and s.files == 0 and os.path.exists(s.xlsx)


def test_cli_missing_input_returns_2(tmp_path):
    assert cli_main(["run", "-i", str(tmp_path / "nope"), "-o", str(tmp_path / "out")]) == 2


@need_train
def test_train_row_per_file_duplicates_share_prediction_and_run_is_reproducible(tmp_path):
    class ByHash(Predictor):
        """Предсказание зависит от картинки — чтобы проверить, что дубликаты получают одинаковый ответ."""
        def predict_flags(self, img, region, side):
            p = float(img.mean() % 1.0)
            return {"spine_artifact": p} if region == REGION_SPINE else {"hip_roi_field": p}

    rows1, s1 = run(TRAIN_DIR, str(tmp_path / "a"), predictor=ByHash(), fmt="csv")
    rows2, _ = run(TRAIN_DIR, str(tmp_path / "b"), predictor=ByHash(), fmt="csv")
    assert (s1.files, s1.unique_images, s1.failure) == (499, 252, 0)
    strip = lambda rs: [{k: v for k, v in r.items() if k != "time_of_processing"} for r in rs]  # noqa: E731
    assert strip(rows1) == strip(rows2)
    by_path = {r["path_to_study"]: r for r in rows1}
    for r in rows1:
        if r["duplicate_of"]:
            assert r["quality_prob"] == by_path[r["duplicate_of"]]["quality_prob"]


WEIGHTS = os.path.join(ROOT, "weights")
need_weights = pytest.mark.skipif(not os.path.exists(os.path.join(WEIGHTS, "landmarks_hip.pt")), reason="сначала scripts/download_weights.sh")


@need_test
@need_weights
def test_nonstandard_frame_width_gets_region_from_localizers(tmp_path):
    """Кадр нестандартной ширины: область определяют локализаторы; кадр без анатомии — Failure, пакет не падает."""
    import json
    import pydicom
    from dxaqc.predict import load_default_predictor
    src = tmp_path / "in"
    src.mkdir()
    expect = {}
    for name in sorted(os.listdir(TEST_DIR)):
        ds = pydicom.dcmread(os.path.join(TEST_DIR, name))
        a = ds.pixel_array[:, 7:-6].copy()                      # ширина 300 → 287, 280 → 267: таких кадров в правиле ширины нет
        expect[name] = REGION_SPINE if ds.Columns == 300 else REGION_HIP
        ds.Rows, ds.Columns = a.shape
        ds.PixelData = a.tobytes()
        ds.save_as(str(src / name))
    ds.Rows, ds.Columns = 200, 190                              # кадр без анатомии
    ds.PixelData = np.zeros((200, 190), a.dtype).tobytes()
    ds.SOPInstanceUID = pydicom.uid.generate_uid()
    ds.save_as(str(src / "blank.dcm"))
    rows, s = run(str(src), str(tmp_path / "out"), predictor=load_default_predictor())
    by = {r["path_to_study"]: r for r in rows}
    assert s.files == 4 and s.failure == 1
    for name, region in expect.items():
        assert by[name]["processing_status"] == "Success"
        assert by[name]["anatomical_region"] == region
        assert json.loads(by[name]["details"])["region_by"] == "localizers"
    assert by["blank.dcm"]["processing_status"] == "Failure" and "unsupported_image" in by["blank.dcm"]["error_message"]


@need_test
@need_weights
def test_16bit_inverted_frame_gives_same_verdict(tmp_path):
    """16 бит и перевёрнутая шкала (MONOCHROME1) приводятся к шкале обучающих снимков: вердикт тот же."""
    import pydicom
    from dxaqc.predict import load_default_predictor
    pred = load_default_predictor()
    base, _ = run(TEST_DIR, str(tmp_path / "a"), predictor=pred)
    src = tmp_path / "in"
    src.mkdir()
    for name in sorted(os.listdir(TEST_DIR)):
        ds = pydicom.dcmread(os.path.join(TEST_DIR, name))
        a = ds.pixel_array.astype(np.uint16)
        a = (int(a.max()) - a) * 16                               # инверсия и 12-битная шкала
        ds.PhotometricInterpretation = "MONOCHROME1"
        ds.BitsAllocated, ds.BitsStored, ds.HighBit = 16, 12, 11
        ds.PixelData = a.tobytes()
        ds.save_as(str(src / name))
    rows, s = run(str(src), str(tmp_path / "b"), predictor=pred)
    assert s.failure == 0
    ref = {r["path_to_study"]: r for r in base}
    for r in rows:
        o = ref[r["path_to_study"]]
        assert (r["anatomical_region"], r["quality_class"], r["violation_type"]) == (o["anatomical_region"], o["quality_class"], o["violation_type"])
        assert abs(float(r["quality_prob"]) - float(ref[r["path_to_study"]]["quality_prob"])) < 0.1
