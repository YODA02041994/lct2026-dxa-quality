"""Загрузчик на «Для теста»: плоская папка, суффиксы области, уникальные хэши."""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from dxaqc.loader import discover_studies  # noqa: E402

TEST_DIR = os.path.join(ROOT, "data", "work", "test", "Для теста")


@pytest.mark.skipif(not os.path.isdir(TEST_DIR), reason="сначала python scripts/unpack_data.py")
def test_flat_folder_is_one_study():
    studies = discover_studies(TEST_DIR)
    assert len(studies) == 1
    st = studies[0]
    assert len(st.images) == 3
    assert len(st.unique_images) == 3
    hints = sorted((im.hint_region, im.hint_side) for im in st.images)
    assert hints == [("hip", "L"), ("hip", "R"), ("spine", None)]
    for im in st.images:
        assert im.study_uid and im.sop_uid
        assert im.rows > 0 and im.cols > 0
