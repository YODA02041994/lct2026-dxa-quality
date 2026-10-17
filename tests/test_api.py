"""HTTP-сервис для страницы с двумя режимами: состояния критериев, пороги в сводке, миниатюры, кэш разметки, демо без раздачи файлов."""
import importlib
import os
import shutil

import pytest

TEST_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "work", "test", "Для теста")
need_test = pytest.mark.skipif(not os.path.isdir(TEST_DIR), reason="нет тестовых DICOM")


@pytest.fixture()
def client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    files, demo = tmp_path / "files", tmp_path / "demo"
    files.mkdir(); demo.mkdir()
    for i, name in enumerate(sorted(os.listdir(TEST_DIR))):
        shutil.copy(os.path.join(TEST_DIR, name), demo / f"demo_{i:03d}.dcm")
    monkeypatch.setenv("DXAQC_FILES", str(files)); monkeypatch.setenv("DXAQC_DEMO_DIR", str(demo)); monkeypatch.setenv("DXAQC_RUNS", str(tmp_path / "runs"))
    import dxaqc.api as api
    api = importlib.reload(api)
    with TestClient(api.app) as c:
        yield c


@need_test
def test_analyze_gives_states_thresholds_and_images(client):
    files = [("files", (n, open(os.path.join(TEST_DIR, n), "rb"), "application/dicom")) for n in sorted(os.listdir(TEST_DIR))]
    j = client.post("/api/analyze", files=files).json()
    assert len(j["rows"]) == 3 and set(j["summary"]["thresholds"]) >= {"spine_axis_tilt", "hip_roi_field"} and j["summary"]["soft"]
    for r in j["rows"]:
        d = r["details"]
        assert r["level"] in ("переснять", "проверить укладку", "в норме")
        assert set(d["states"]) == set(d["probs"]) and set(d["states"].values()) <= {"violation", "near", "ok"}
        # состояние «нарушение» ровно у тех критериев, что попали в violation_type
        assert (r["quality_class"] == 1) == any(v == "violation" for v in d["states"].values())
    run = j["run_id"]
    im = client.get(f"/api/runs/{run}/image/0.png")
    assert im.status_code == 200 and im.headers["content-type"] == "image/png" and im.content[:4] == b"\x89PNG"
    ov1 = client.get(f"/api/runs/{run}/overlay/0.png"); ov2 = client.get(f"/api/runs/{run}/overlay/0.png")
    assert ov1.status_code == ov2.status_code == 200 and ov1.content == ov2.content
    assert client.get(f"/api/runs/{run}/image/99.png").status_code == 404


@need_test
def test_demo_runs_on_server_and_files_are_not_served(client):
    assert len(client.get("/api/demo").json()["files"]) == 3
    one = client.post("/api/demo?only=demo_001").json()
    assert len(one["rows"]) == 1
    assert len(client.post("/api/demo").json()["rows"]) == 3
    assert client.get("/files/demo_000.dcm").status_code == 404           # демо-снимки постановщика наружу не отдаются


def test_page_is_not_cached_and_has_two_modes(client):
    r = client.get("/")
    assert r.status_code == 200 and r.headers.get("cache-control") == "no-store"
    assert 'id="tab-lab"' in r.text and 'id="tab-qa"' in r.text and "Переснял — проверить снова" in r.text
