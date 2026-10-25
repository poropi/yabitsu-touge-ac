import configparser
import importlib
import json

import numpy as np
import pytest
from shapely.geometry import Point, Polygon

from yabitsu import config


def rain():
    return importlib.import_module("yabitsu.environment_rain")


def test_current_materials_cover_roads_and_visual_paint_without_wetting_walls(tmp_path):
    path, report = rain().build(tmp_path)
    ini = configparser.ConfigParser(interpolation=None)
    ini.read(path)
    section = ini["RAIN_FX"]
    puddles = set(section["puddles_materials"].split(","))
    paint = set(section["lines_materials"].split(","))
    assert {"asphalt", "concrete_circles", "visual_concrete_circles"} <= puddles
    assert {"visual_line_white", "visual_paint_red", "visual_paint_green", "dress_text_40",
            "dress_green", "dress_red", "mt_mark_diamond"} <= paint
    assert "asphalt" in section["soaking_materials"].split(",")
    assert not {"concrete", "ground", "visual_ground_far", "tunnel_wall"} & puddles
    assert "visual_ground_far" in section["rough_materials"].split(",")
    meshes = set(section["puddles_meshes"].split(","))
    assert "1ROAD_s1_lot" in meshes
    assert all(name.startswith("1ROAD_") for name in meshes)
    assert "relief_materials" not in section
    assert report["tunnel_count"] == 3


def test_occluders_cover_existing_tunnel_segments_at_roof_height_only(tmp_path):
    path, report = rain().build(tmp_path)
    ini = configparser.ConfigParser(interpolation=None)
    ini.read(path)
    source = config.TRACK_BUILD_DIR
    manifest = json.loads((source / "manifest.json").read_text())
    boxes = [s for s in ini.sections() if s.startswith("TRACK_OCCLUDER_BOX_")]
    assert 150 < len(boxes) < 200
    assert boxes == [f"TRACK_OCCLUDER_BOX_{i}" for i in range(len(boxes))]
    for s in boxes:
        item = report["occluders"][s]
        points = np.array([[float(x) for x in ini[s][f"point_{i}"].split(",")] for i in range(4)])
        vertices = np.load(source / manifest["meshes"][item["mesh"]]["file"])["verts"].reshape(-1, 15, 3)
        pair = vertices[item["row"]:item["row"] + 2]
        # AC (east, height, -north) -> source (east, north, height).
        local = points[:, [0, 2, 1]] * [1, -1, 1]
        polygon = Polygon(local[:, :2])
        assert polygon.is_valid and 1 < polygon.area < 40
        assert np.ptp(local[:, 2]) < 1e-6
        # 公式 RainFX wiki の POINT_0..3 は AC 座標の外積 Y が負。
        # 内側（道路）へ向けた上面として、全セグメントを同じ向きにする。
        normal = np.cross(points[1] - points[0], points[2] - points[0])
        assert normal[1] < 0
        assert local[0, 2] >= pair[:, :, 2].max() - .001
        corners = pair[:, [0, -1], :2].reshape(-1, 2)
        assert all(polygon.buffer(.002).covers(Point(p)) for p in corners)
        assert np.linalg.norm(local[:, :2].mean(0) - corners.mean(0)) < .002


def test_invalid_bore_geometry_fails_instead_of_creating_unbounded_rain_shadow():
    with pytest.raises(ValueError, match="断面"):
        rain().bore_occluders(np.zeros((17, 3)))


def test_rain_build_preserves_existing_night_and_kn5(tmp_path):
    night = tmp_path / "extension/parts/night.ini"
    night.parent.mkdir(parents=True)
    night.write_bytes(b"[CONDITION_60]\nNAME=NightLights\n")
    model = tmp_path / "yabitsu_environment.kn5"
    model.write_bytes(b"existing-night-kn5")
    before = {p: p.read_bytes() for p in (night, model)}
    rain().build(tmp_path)
    assert all(p.read_bytes() == content for p, content in before.items())
    assert (tmp_path / "rain_report.json").is_file()
