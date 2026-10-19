import json

import numpy as np
import pytest

from yabitsu import forest_ground as fg

CFG = {**fg.DEFAULT_CLEARANCE, "cell": 0.1, "coarse_cell": 1.0}


def _manifest(tmp_path, meshes):
    """meshes: {name: (verts, faces, material)} の小さな manifest を作る。"""
    (tmp_path / "meshes").mkdir()
    entry = {}
    for name, (v, f, mat) in meshes.items():
        np.savez_compressed(tmp_path / "meshes" / f"{name}.npz", verts=np.asarray(v, float), faces=np.asarray(f, np.int32),
                            uvs=np.zeros((len(v), 2)))
        entry[name] = {"file": f"meshes/{name}.npz", "material": mat, "renderable": True}
    (tmp_path / "manifest.json").write_text(json.dumps({"materials": {}, "meshes": entry, "empties": {}, "groups": {}}),
                                            encoding="utf-8")
    return tmp_path


def _square(x0, y0, x1, y1, z):
    return [(x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z)], [(0, 1, 2), (0, 2, 3)]


def test_ground_prefers_near(tmp_path):
    far = _square(-50, -50, 50, 50, 90.0); near = _square(-10, -10, 10, 10, 100.0)
    g = fg.load_ground(_manifest(tmp_path, {"TERRAIN_FAR_s1_000": (*far, "ground"), "1GRASS_s1_terrain_000": (*near, "ground")}))
    z = g.z(np.array([[0.0, 0.0], [30.0, 30.0]]))
    assert z.tolist() == [100.0, 90.0]


def test_ground_nan_outside(tmp_path):
    g = fg.load_ground(_manifest(tmp_path, {"1GRASS_s1_terrain_000": (*_square(0, 0, 10, 10, 5.0), "ground")}))
    assert np.isnan(g.z(np.array([[20.0, 5.0]]))[0])


def test_ground_interpolates_triangle(tmp_path):
    v = [(0, 0, 0), (10, 0, 10), (0, 10, 20)]
    g = fg.load_ground(_manifest(tmp_path, {"1GRASS_s1_terrain_000": (v, [(0, 1, 2)], "ground")}))
    assert g.z(np.array([[2.0, 3.0]]))[0] == pytest.approx(2.0 + 6.0)


def test_ground_accepts_quads(tmp_path):
    v, _ = _square(0, 0, 10, 10, 7.0)
    g = fg.load_ground(_manifest(tmp_path, {"1GRASS_s1_terrain_000": (v, [(0, 1, 2, 3)], "ground")}))
    assert g.z(np.array([[8.0, 2.0], [2.0, 8.0]])).tolist() == [7.0, 7.0]


def _keep(tmp_path, meshes):
    meshes = {"1GRASS_s1_terrain_000": (*_square(-40, -40, 40, 40, 0.0), "ground"), **meshes}
    return fg.load_keepout(_manifest(tmp_path, meshes), CFG)


def test_keepout_bridge_footprint(tmp_path):
    k = _keep(tmp_path, {"bridge_deck_000": (*_square(-5, -2, 5, 2, 15.0), "concrete")})
    assert k.blocked(np.array([[0.0, 0.0], [0.0, 10.0]])).tolist() == [True, False]


def test_keepout_wall_buffer(tmp_path):
    wall = [(0, -10, 0), (0, 10, 0), (0, 10, 3), (0, -10, 3)]
    k = _keep(tmp_path, {"1WALL_s1_retaining_000": (wall, [(0, 1, 2), (0, 2, 3)], "concrete")})
    assert k.blocked(np.array([[0.4, 0.0], [0.8, 0.0]])).tolist() == [True, False]


def test_road_distance(tmp_path):
    k = _keep(tmp_path, {"1ROAD_s1_000": (*_square(-20, -3, 20, 3, 0.0), "asphalt")})
    d = k.road_distance(np.array([[0.0, 6.0], [0.0, 0.0]]))
    assert d[0] == pytest.approx(3.0, abs=0.15) and d[1] == 0.0


def test_osm_roads_extend_keepout(tmp_path):
    k = _keep(tmp_path, {})
    k.add_lines([np.array([[-30.0, 20.0], [30.0, 20.0]])], [4.0])
    assert k.blocked(np.array([[0.0, 20.0], [0.0, 30.0]])).tolist() == [True, False]
    assert k.road_distance(np.array([[0.0, 30.0]]))[0] == pytest.approx(10.0 - 2.0, abs=1.5)


def test_surface_by_prefix(tmp_path):
    m = _manifest(tmp_path, {"1GRASS_s1_terrain_000": (*_square(-10, -10, 10, 10, 0.0), "ground"),
                             "1ROAD_s1_000": (*_square(-2, -10, 2, 10, 3.0), "asphalt")})
    assert fg.load_ground(m, prefixes=("1ROAD_",)).z(np.array([[0.0, 0.0]]))[0] == 3.0


def test_course_distance_ignores_osm(tmp_path):
    k = _keep(tmp_path, {"1ROAD_s1_000": (*_square(-20, -3, 20, 3, 0.0), "asphalt")})
    k.add_lines([np.array([[-30.0, 20.0], [30.0, 20.0]])], [4.0])
    assert k.course_distance(np.array([[0.0, 25.0]]))[0] == pytest.approx(22.0, abs=1.5)
    assert k.road_distance(np.array([[0.0, 25.0]]))[0] < 5.0
