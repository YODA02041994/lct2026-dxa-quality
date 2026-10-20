"""Доработка после консультации 16.10: область по анатомии, три уровня выдачи, сколиоз отдельно от критериев, советы без цифр."""
import os
import re

from dxaqc.criteria import ADVICE
from dxaqc.labels import REGION_HIP, REGION_SPINE
from dxaqc.predict import aggregate
from dxaqc.region import decide_region
from dxaqc.scoliosis import GRADE_BORDERLINE, GRADE_MARKED, GRADE_NONE, damp_axis, decision_text, measure


def _lm(xs, conf=0.9):
    names = ["th12", "L1", "L2", "L3", "L4", "L5"]
    return {n: {"x": x, "y": 10 + 50 * i, "conf": conf, "present": True} for i, (n, x) in enumerate(zip(names, xs))}


def test_region_anatomy_wins_over_width():
    assert decide_region(REGION_SPINE, REGION_SPINE, None) == (REGION_SPINE, "anatomy", True)
    # кадр шириной «позвоночник», а анатомия — бедро: берём анатомию и просим посмотреть глазами
    region, by, agree = decide_region(REGION_SPINE, REGION_HIP, REGION_HIP)
    assert region == REGION_HIP and not agree and "mismatch" in by
    # анатомия разошлась сама с собой → решает стандартная ширина
    assert decide_region(REGION_HIP, REGION_SPINE, REGION_HIP)[0] == REGION_HIP
    # нет ни анатомии, ни стандартной ширины → не поддерживается
    assert decide_region(None, None, None)[0] is None
    # нестандартная ширина, но анатомия найдена одним голосом
    assert decide_region(None, REGION_SPINE, None) == (REGION_SPINE, "anatomy", True)


def test_levels_three_tier():
    thr = {"a": 0.6, "b": 0.6}
    soft = {"a": 0.3, "b": 0.3}
    assert aggregate({"a": 0.7, "b": 0.1}, thr, soft).level == "repeat"
    v = aggregate({"a": 0.4, "b": 0.1}, thr, soft)
    assert v.level == "check" and v.check_flags == ["a"] and v.quality_class == 0
    assert aggregate({"a": 0.1, "b": 0.1}, thr, soft).level == "ok"
    # без мягких порогов «проверить» даёт только полоса у порога
    assert aggregate({"a": 0.55, "b": 0.1}, thr).level == "check"
    assert aggregate({"a": 0.4, "b": 0.1}, thr).level == "ok"


def test_scoliosis_grades_and_damping(monkeypatch):
    straight = measure(_lm([100, 101, 100, 101, 100, 101]))
    assert straight["grade"] == GRADE_NONE and straight["bend_deg"] < 5
    tilted = measure(_lm([100, 108, 116, 124, 132, 140]))          # прямой наклонённый — не сколиоз
    assert tilted["grade"] == GRADE_NONE
    curved = measure(_lm([100, 112, 120, 120, 112, 100]))          # дуга — деформация
    assert curved["grade"] == GRADE_MARKED and curved["bend_deg"] >= 10
    assert measure(_lm([100, 100, 100, 100], conf=0.1))["grade"] == GRADE_NONE   # мало уверенных точек — не судим
    monkeypatch.delenv("DXAQC_SCOLIOSIS_DAMP", raising=False)
    probs, damped = damp_axis({"spine_axis_tilt": 0.8, "spine_artifact": 0.2}, GRADE_MARKED)
    assert damped and probs["spine_axis_tilt"] < 0.3 and probs["spine_artifact"] == 0.2
    assert damp_axis({"spine_axis_tilt": 0.8}, GRADE_BORDERLINE)[1] is False
    monkeypatch.setenv("DXAQC_SCOLIOSIS_DAMP", "0")
    assert damp_axis({"spine_axis_tilt": 0.8}, GRADE_MARKED)[1] is False


def test_decision_text_without_numbers():
    for other in (False, True):
        t = decision_text(GRADE_MARKED, other)
        assert t and not re.search(r"\d", t), t
    assert decision_text(GRADE_BORDERLINE, False) == ""      # пограничная асимметрия — строка в измерениях, не вердикт
    assert decision_text(GRADE_NONE, False) == ""


def test_advice_is_general():
    for text in ADVICE.values():
        assert not re.search(r"\d", text), text          # без градусов и сантиметров
        assert "повернуть" not in text.lower() and "довернуть" not in text.lower()


def test_decision_text_names_findings():
    t = decision_text(GRADE_MARKED, ["spine_artifact"], [])
    assert "посторонние предметы" in t and "укладк" not in t.split("найдено:")[1], t
    t = decision_text(GRADE_MARKED, [], ["spine_artifact"])
    assert "у порога: посторонние предметы" in t and "соблюдены" not in t, t
    t = decision_text(GRADE_MARKED, [], [])
    assert "соблюдены" in t and not re.search(r"\d", t), t


def test_economy_mode_keeps_full_artifact_ensemble(monkeypatch):
    import glob, os
    from dxaqc.cnn import CnnScorer
    files = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "..", "weights", "cnn_spine_artifact_*.pt")))
    hips = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "..", "weights", "cnn_hip_positioning_rotation_lt100.pt")))
    if not files or not hips:
        import pytest; pytest.skip("нет весов")
    monkeypatch.setenv("DXAQC_NETS_PER_FILE", "1"); monkeypatch.setenv("DXAQC_DEVICE", "cpu")
    monkeypatch.delenv("DXAQC_FULL_CRITERIA", raising=False)
    art, hip = CnnScorer(files[0]), CnnScorer(hips[0])
    monkeypatch.delenv("DXAQC_NETS_PER_FILE")
    assert len(art.nets) == len(CnnScorer(files[0]).nets)      # предметы — все сети файла
    assert len(hip.nets) == 1                                  # бедро — экономно, одна сеть
