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
    assert v.quality_prob == 0.973          # 1 − (1−0,1)(1−0,9)(1−0,7): два подозрительных критерия > одного
    clean = aggregate({"hip_positioning_rotation": 0.2, "hip_roi_field": 0.1})
    assert (clean.quality_class, clean.violation_type) == (0, "")
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
