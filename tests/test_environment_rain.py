import configparser
import importlib
import json

import numpy as np
import pytest
from shapely.geometry import LineString, Polygon

from yabitsu import config


def rain():
    return importlib.import_module("yabitsu.environment_rain")


def test_current_materials_cover_roads_and_visual_paint_without_wetting_walls(tmp_path):
    path, report = rain().build(tmp_path)
    ini = configparser.ConfigParser(interpolation=None)
    ini.read(path)
    section = ini["RAIN_FX"]
    assert section["puddles_materials"] == "" and section["puddles_meshes"] == ""   # 水たまりは作らない（トンネルで鏡になった）
    paint = set(section["lines_materials"].split(","))
    assert {"visual_line_white", "visual_paint_red", "visual_paint_green", "dress_text_40",
            "dress_green", "dress_red", "mt_mark_diamond"} <= paint
    assert "asphalt" in section["soaking_materials"].split(",")
    assert not {"concrete", "ground", "visual_ground_far", "tunnel_wall"} & set(section["soaking_materials"].split(","))
    assert "visual_ground_far" in section["rough_materials"].split(",")
    assert "relief_materials" not in section
    assert report["tunnel_count"] == 3
    assert report.get("occluder_tunnel_count") == 1


def test_small_tunnel_keeps_ten_roof_boxes_in_vertex_and_segment_order(tmp_path):
    from yabitsu.tunnel import N_ARC

    path, report = rain().build(tmp_path)
    ini = configparser.ConfigParser(interpolation=None)
    ini.read(path)
    source = config.TRACK_BUILD_DIR
    manifest = json.loads((source / "manifest.json").read_text())
    bore = "1WALL_tunnel_131561907_bore"
    with np.load(source / manifest["meshes"][bore]["file"]) as mesh:
        vertices = mesh["verts"].reshape(-1, N_ARC + 3, 3)
    boxes = [s for s in ini.sections() if s.startswith("TRACK_OCCLUDER_BOX_")
             and ini[s]["description"].startswith(f"{bore} ")]
    assert boxes == [f"TRACK_OCCLUDER_BOX_{i}" for i in range(10)]
    assert len(vertices) == 11
    for row, s in enumerate(boxes):
        assert ini[s]["description"] == f"{bore} segment {row}"
        assert (report["occluders"][s]["mesh"], report["occluders"][s]["row"]) == (bore, row)
        points = np.array([[float(x) for x in ini[s][f"point_{i}"].split(",")] for i in range(4)])
        pair = vertices[row:row + 2]
        # AC (east, height, -north) -> source (east, north, height).
        local = points[:, [0, 2, 1]] * [1, -1, 1]
        polygon = Polygon(local[:, :2])
        assert polygon.is_valid and 1 < polygon.area < 40
        assert np.ptp(local[:, 2]) < 1e-6
        # 公式 RainFX wiki の POINT_0..3 は AC 座標の外積 Y が負。
        # 内側（道路）へ向けた上面として、全セグメントを同じ向きにする。
        normal = np.cross(points[1] - points[0], points[2] - points[0])
        assert normal[1] < 0
        np.testing.assert_allclose(local[:, 2], pair[:, :, 2].max() + .02, rtol=0, atol=.00005)
        corners = pair[:, [0, -1], :2].reshape(-1, 2)[[0, 1, 3, 2]]
        np.testing.assert_allclose(local[:, :2], corners, rtol=0, atol=.00005)


def test_roof_boxes_do_not_cover_views_from_inside_fubuki_and_odo(tmp_path):
    from yabitsu import track, tunnel
    from yabitsu.crosssection import Section

    path, _ = rain().build(tmp_path)
    ini = configparser.ConfigParser(interpolation=None)
    ini.read(path)
    polygons = {}
    for name in ini.sections():
        if name.startswith("TRACK_OCCLUDER_BOX_"):
            xyz = np.array([[float(x) for x in ini[name][f"point_{i}"].split(",")] for i in range(4)])
            polygons[name] = Polygon(xyz[:, [0, 2]] * [1, -1])

    sec = Section.load_json(config.DATA_PROCESSED / "track_section.json")
    s5_start = track.load_bounds()["s5"][0]
    source = config.TRACK_BUILD_DIR
    manifest = json.loads((source / "manifest.json").read_text())
    covered_bores = {}
    for bore in ("1WALL_tunnel_32745017_bore", "1WALL_tunnel_131561908_bore"):
        with np.load(source / manifest["meshes"][bore]["file"]) as mesh:
            edges = mesh["verts"].reshape(-1, tunnel.N_ARC + 3, 3)[:, [0, -1], :2]
        footprints = [Polygon([a[0], a[1], b[1], b[0]]) for a, b in zip(edges[:-1], edges[1:])]
        hits = [name for name, polygon in polygons.items()
                if any(polygon.intersection(footprint).area > .000001 for footprint in footprints)]
        if hits:
            covered_bores[bore] = hits
    # 依頼20で部品を特定した坑内視点と路面・坑外の視線の、水平投影を検査する。
    # CSP の描画判定そのものはゲームでの A/B 比較で確かめる。
    views = [
        (10848.961, 10857.97),  # 吹風の筒内路面
        (10908.961, 10939.63),  # 宮ヶ瀬側の坑外路面と橋へ続く方向
        (10704.962, 10661.51),  # 峠側の坑外路面
        (2480.491, 2454.40),    # 大洞の筒内路面
        (2471.991, 2448.00),    # 峠側の坑外路面
        (2503.991, 2511.00),    # 宮ヶ瀬側の坑外路面
    ]
    covered = {}
    for camera, target in views:
        samples = s5_start + np.array([camera, target])
        sightline = LineString(np.c_[np.interp(samples, sec.s, sec.x), np.interp(samples, sec.s, sec.y)])
        hits = [name for name, polygon in polygons.items() if polygon.intersection(sightline).length > .001]
        if hits:
            covered[camera, target] = hits
    assert (covered_bores, covered) == ({}, {})


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


def test_dry_road_materials_and_overrides_do_not_enable_reflections(tmp_path):
    from scripts import dress_mountain
    from yabitsu import dress_retex, kn5, track
    from yabitsu.crosssection import Section

    base = config.DATA_PROCESSED
    sec = Section.load_json(base / "track_section.json")
    with np.load(base / "track_centerline.npz") as data:
        cl = dict(data)
    bounds = track.load_bounds()
    assignments = dress_mountain.retex_assign(
        sec, cl, bounds, dress_mountain.track_features(sec, bounds), dress_mountain.table_rows())
    manifest = json.loads((config.TRACK_BUILD_DIR / "manifest.json").read_text())
    dry_roads = {n for n, mesh in manifest["meshes"].items() if mesh["material"] == "asphalt_tunnel"}
    assert {"1ROAD_s1_101_tun", "1ROAD_s1_110_tun", "1ROAD_s1_144_tun"} <= dry_roads
    assert kn5.MATERIAL_PROPS["ksSpecular"] == 0.05
    assert not any(key.lower().startswith("fresnel") for key in kn5.MATERIAL_PROPS)
    covered = set()
    for entry in dress_retex.section_text(assignments).split("[SHADER_REPLACEMENT_...]")[1:]:
        values = dict(row.split("=", 1) for row in entry.splitlines() if "=" in row)
        roads = set(values["MESHES"].split(",")) & dry_roads
        if roads:
            covered |= roads
            assert values["SHADER"] == "ksPerPixelNM"
            assert values["RESOURCE_0"] == "txDiffuse" and values["RESOURCE_1"] == "txNormal"
            assert not any(key.startswith("PROP_") for key in values)
    assert covered == dry_roads

    _, report = rain().build(tmp_path)
    assert not {"asphalt_tunnel"} & set(report["material_lists"]["SOAKING_MATERIALS"])
    assert not report["material_lists"]["PUDDLES_MATERIALS"]
    assert not report["material_lists"]["PUDDLES_MESHES"]
