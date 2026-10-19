import json

import numpy as np
import pytest
from shapely.geometry import Point, Polygon

from yabitsu import asama_area, config
from yabitsu.crosssection import Section


@pytest.fixture(scope="module")
def site():
    sec = Section.load_json(config.DATA_PROCESSED / "track_section.json")
    spec = asama_area.load(config.DATA_MANUAL / "s2_asama.json")
    return asama_area.build(spec, sec), sec


def test_pad_is_beside_road_not_on_it(site):
    s, sec = site
    pad = s.pad_polygon()
    assert pad.area > 80
    road = asama_area.road_formation(sec, s_min=7150, s_max=7350)
    assert pad.intersection(road).area < 1.0


def test_pads_for_terrain(site):
    s, _ = site
    pads = s.pads()
    names = {p["name"] for p in pads}
    assert names == {"asama_pad", "asama_stairs"}
    st = next(p for p in pads if p["name"] == "asama_stairs")
    cx, cy, z0, gx, gy = st["plane"]
    assert gy > 0.4                          # 石段は北へ上る


def test_stairs_climb_to_terrain(site):
    s, _ = site
    meshes = {n: v for n, v, f, uv, mat, r in s.meshes()}
    st = meshes["1ROAD_asama_stairs"]
    assert st[:, 2].max() - st[:, 2].min() > 8.0


def test_meshes_have_collision_and_names(site):
    s, _ = site
    names = [n for n, *_ in s.meshes()]
    for need in ("1ROAD_asama_pad", "1WALL_asama_wall", "1WALL_asama_parapet", "1WALL_asama_monument", "asama_pillar",
                 "asama_handrails"):
        assert any(n.startswith(need) for n in names), need


def test_wall_behind_pad(site):
    s, _ = site
    meshes = {n: v for n, v, f, uv, mat, r in s.meshes()}
    pad = s.pad_polygon()
    w = meshes["1WALL_asama_wall"]
    c = w[:, :2].mean(0)
    assert c[1] > pad.centroid.y                 # 壁は広場の北（奥）
    assert w[:, 2].max() - s.z(*c) > 3.0
