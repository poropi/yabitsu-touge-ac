import numpy as np
import pytest

from yabitsu import houses as hs
from tests.test_minoge_area import SPEC, _ground, _sec, _site

# minoge_area の家を houses.py へ移す前の、試験用の蓑毛の部品（名前・頂点の数・座標の合計）
BEFORE = {"1ROAD_s2_minoge_turnaround": (6, 1159.14), "1WALL_s2_houses": (40, 7937.2), "s2_houses_gables": (12, 2409.0),
          "s2_houses_roof_brown": (8, 1393.314), "s2_houses_roof_grey": (8, 1821.04), "1WALL_s2_shrine": (20, 5026.4),
          "s2_shrine_gables": (6, 1517.6), "s2_shrine_roof": (8, 2023.467), "1WALL_s2_monuments": (20, 4419.2)}


def test_s2_houses_unchanged_by_refactor():
    ms = {n: (len(v), round(float(np.asarray(v).sum()), 3)) for n, v, f, uv, mat, r in _site().meshes()}
    assert ms == BEFORE
    assert [p["name"] for p in _site().pads()] == ["minoge_turnaround", "minoge_house_0", "minoge_house_1", "minoge_shrine"]


def test_roof_colour_from_data_and_prefix():
    b = [dict(SPEC["buildings"][0], roof="blue"), dict(SPEC["buildings"][1], roof="red")]
    h = hs.Houses(b, _sec(), _ground, prefix="s3_houses", pad_name="s3_house_{k}")
    names = {m[0]: m[4] for m in h.meshes()}
    assert names["s3_houses_roof_blue"] == "roof_blue" and names["s3_houses_roof_red"] == "roof_red"
    assert "1WALL_s3_houses" in names
    assert [p["name"] for p in h.pads()] == ["s3_house_0", "s3_house_1"]


def test_houses_avoid_given_polygons():
    from shapely.geometry import Point
    b = SPEC["buildings"][:1]
    h = hs.Houses(b, _sec(), _ground, prefix="x", pad_name="x_{k}", avoid=[Point(40.0, 20.0).buffer(3.0)])
    assert h.meshes() == [] and h.pads() == []
