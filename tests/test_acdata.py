import json
import numpy as np
import pytest
from PIL import Image
from yabitsu import acdata, config


def test_blender_to_ac():
    assert acdata.blender_to_ac(1.0, 2.0, 3.0) == (1.0, 3.0, -2.0)


def test_surfaces_ini(tmp_path):
    p = tmp_path / "surfaces.ini"
    acdata.write_surfaces_ini(p)
    t = p.read_text(encoding="utf-8")
    for key in ("KEY=ROAD", "KEY=GRASS", "KEY=WALL", "[SURFACE_0]", "[SURFACE_2]"):
        assert key in t


def test_refuses_protected_folder():
    with pytest.raises(config.ProtectedPathError):
        acdata.write_surfaces_ini(config.AC_TRACKS / "nanohanadai" / "data" / "surfaces.ini")


def test_models_ini(tmp_path):
    p = tmp_path / "models_a.ini"
    acdata.write_models_ini(p, ["a.kn5", "b.kn5"])
    t = p.read_text(encoding="utf-8")
    assert "[MODEL_0]" in t and "FILE=a.kn5" in t and "[MODEL_1]" in t and "FILE=b.kn5" in t


def test_map_params_fit_inside_margin():
    xz = np.array([[-100.0, 50.0], [300.0, -400.0], [0.0, 0.0]])
    p = acdata.map_params(xz, max_px=1024, margin=20)
    px, py = acdata.to_map_px(xz, p)
    assert px.min() == pytest.approx(20) and py.min() == pytest.approx(20)
    assert px.max() <= p.width - 20 + 1 and py.max() <= p.height - 20 + 1
    assert max(p.width, p.height) <= 1024 + 1


def test_write_map(tmp_path):
    xz = np.array([[0.0, 0.0], [100.0, 200.0]])
    p = acdata.write_map(tmp_path / "map.png", tmp_path / "map.ini", xz)
    img = Image.open(tmp_path / "map.png")
    assert img.size == (p.width, p.height) and img.mode == "RGBA"
    t = (tmp_path / "map.ini").read_text(encoding="utf-8")
    for key in ("WIDTH=", "HEIGHT=", "SCALE_FACTOR=", "X_OFFSET=", "Z_OFFSET=", "DRAWING_SIZE="):
        assert key in t


def test_cameras_ini_vectors(tmp_path):
    cam = acdata.Camera("c0", (10.0, 5.0, 0.0), (-1.0, -0.2, 0.0), 0.0, 0.5)
    acdata.write_cameras_ini(tmp_path / "cameras.ini", [cam])
    lines = dict(l.split("=", 1) for l in (tmp_path / "cameras.ini").read_text(encoding="utf-8").splitlines() if "=" in l)
    f = np.array([float(v) for v in lines["FORWARD"].split(",")])
    u = np.array([float(v) for v in lines["UP"].split(",")])
    assert np.linalg.norm(f) == pytest.approx(1, abs=1e-4) and abs(f @ u) < 1e-4
    assert lines["CAMERA_COUNT"] == "1" and lines["OUT_POINT"] == "0.5"


def test_ui_track_json(tmp_path):
    p = tmp_path / "ui_track.json"
    acdata.write_ui_track_json(p, name="T", description="d", length_m=3281, pitboxes=4,
                               geotags=(35.42, 139.21), tags=["touge"], credits="国土地理院 / OpenStreetMap")
    d = json.loads(p.read_text(encoding="utf-8"))
    assert d["pitboxes"] == "4" and d["length"] == "3281 m" and "国土地理院" in d["description"]
