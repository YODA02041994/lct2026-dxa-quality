"""Область по ширине кадра и сторона бедра по положению таза."""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from dxaqc.labels import REGION_HIP, REGION_SPINE  # noqa: E402
from dxaqc.loader import discover_studies  # noqa: E402
from dxaqc.region import assign_sides, hip_side, region_by_width  # noqa: E402

TEST_DIR = os.path.join(ROOT, "data", "work", "test", "Для теста")
TRAIN_DIR = os.path.join(ROOT, "data", "work", "train", "Исследования")


def test_region_by_width():
    assert region_by_width(300) == REGION_SPINE
    assert region_by_width(280) == REGION_HIP
    assert region_by_width(248) == REGION_HIP      # ортопедический режим, эндопротез
    assert region_by_width(512) is None


def test_assign_sides_pair_is_complementary():
    assert assign_sides([0.8, -0.1]) == ["R", "L"]
    assert assign_sides([0.05, 0.6]) == ["L", "R"]   # оба «правые» по знаку — в паре всё равно разные
    assert assign_sides([-0.3]) == ["L"]


@pytest.mark.skipif(not os.path.isdir(TEST_DIR), reason="сначала python scripts/unpack_data.py")
def test_side_matches_filename_hint():
    for st in discover_studies(TEST_DIR):
        for im in st.images:
            assert region_by_width(im.cols) == (REGION_SPINE if im.hint_region == "spine" else REGION_HIP)
            if im.hint_region == "hip":
                assert hip_side(im.load_pixels()).side == im.hint_side


@pytest.mark.skipif(not os.path.isdir(TRAIN_DIR), reason="сначала python scripts/unpack_data.py")
def test_every_hip_pair_gets_different_sides():
    for st in discover_studies(TRAIN_DIR):
        hips = [im for im in st.unique_images if region_by_width(im.cols) == REGION_HIP]
        if len(hips) == 2:
            bal = [hip_side(im.load_pixels()).upper_balance for im in hips]
            assert sorted(assign_sides(bal)) == ["L", "R"], st.study_dir
