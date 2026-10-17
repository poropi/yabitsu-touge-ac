import math

import numpy as np
from shapely.geometry import Point, Polygon

from yabitsu import meshgen as mg
from yabitsu import s1_layouts as lay
from yabitsu import start_lot as sl
from tests.test_meshgen import _straight

# 道は x=0 から +x へ。駐車場は道の始まりの南側から、+y（道の方）へ伸びる
SPEC = {
    "origin": [0.0, -20.0], "axis_deg": 90.0,
    "lot": {"a": [-40.0, 25.0], "b": [-2.5, 9.0]},
    "stalls": {"a": [-36.0, -4.0], "b": [-2.5, 2.5], "pitch": 2.5},
    "fence": {"a": [-40.0, 18.0], "b": -2.6, "height": 1.1, "post_every": 2.0},
    "blocks": {"a": [-30.0, -3.0], "b": 9.3, "every": 3.0, "size": [0.9, 0.35, 0.35]},
    "monument": {"a": 10.0, "b": 11.0, "size": [1.4, 0.6, 2.2]},
    "building": {"a": [-45.0, -36.0], "b": [9.5, 15.0], "wall_h": 3.0, "roof_h": 1.6, "eaves": 0.6},
    "flats": [{"a": [-47.0, -34.0], "b": [9.0, 17.0]}],
    "pits": {"a": -15.0, "b": 3.5, "spacing": 2.8},
}


def _pad():
    sec = _straight(201)
    pads = [{"name": "P", "role": "start", "polygon": [[0, -30], [5, -30], [5, -25], [0, -25], [0, -30]], "z": 100.0}]
    return sec, sl.apply(pads, SPEC, sec)[0]


def test_frame_axes():
    f = sl.Frame([10.0, 20.0], 90.0)
    assert np.allclose(f.xy(5.0, 0.0), [10.0, 25.0]) and np.allclose(f.xy(0.0, 2.0), [8.0, 20.0])


def test_lot_replaces_start_pad_and_stays_off_the_road():
    sec, pad = _pad()
    poly = Polygon(pad["polygon"])
    assert pad["kind"] == "lot" and pad["z"] == 100.0
    assert poly.intersection(sl.road_formation(sec)).area < 1e-6
    assert abs(poly.area - 11.5 * 65.0) < 11.5 * 65.0 * 0.1  # 道と重なる角だけ削られる


def test_lot_surface_points_up_and_follows_lot_axis():
    sec, pad = _pad()
    meshes = {name: (v, f, uv, mat) for name, v, f, uv, mat, rend in sl.meshes(SPEC, pad)}
    v, f, uv, mat = meshes["1ROAD_s1_lot"]
    assert mat == "pavers" and (mg.face_normals(v, f)[:, 2] > 0).all() and np.allclose(v[:, 2], 100.0)
    # 長手方向（+y）に進むと u が増え、v は変わらない
    i, j = np.argmin(v[:, 1]), np.argmax(v[:, 1])
    assert uv[j, 0] > uv[i, 0]


def test_meshes_have_expected_parts():
    sec, pad = _pad()
    out = {name: (mat, rend) for name, v, f, uv, mat, rend in sl.meshes(SPEC, pad)}
    assert out["1WALL_s1_lot_fence"] == ("railing", False)
    for name in ("LINES_s1_lot_stalls", "s1_lot_fence", "s1_lot_fence_bars", "1WALL_s1_lot_blocks",
                 "1WALL_s1_monument", "1WALL_s1_building", "s1_building_roof"):
        assert out[name][1] is True, name


def test_boxes_face_outward():
    f = sl.Frame([0.0, 0.0], 30.0)
    v, faces, uv = sl.box(f, (-1.0, 1.0), (-2.0, 2.0), 0.0, 3.0)
    c = v.mean(0)
    n = mg.face_normals(v, faces)
    centres = v[faces].mean(1)
    assert len(faces) == 5 and ((centres - c) * n).sum(1).min() > 0


def test_stall_lines_count():
    sec, pad = _pad()
    v, f, uv, *_ = next(m[1:] for m in sl.meshes(SPEC, pad) if m[0] == "LINES_s1_lot_stalls")
    assert len(f) == int(32 / 2.5) + 1 + 1  # 区切り線と、通路側の長い線


def test_pits_inside_lot_and_face_the_exit():
    sec, pad = _pad()
    pits = [d for d in lay.layout_objects(sec, [pad], 1) if d.name.startswith("AC_PIT_")]
    poly = Polygon(pad["polygon"]).buffer(-1.0)
    assert len(pits) == 4 and all(poly.contains(Point(d.location[0], d.location[1])) for d in pits)
    assert all(abs(math.remainder(d.heading - math.pi / 2, 2 * math.pi)) < 1e-6 for d in pits)


def test_flats_and_clearings_are_at_lot_height():
    sec, pad = _pad()
    flats = sl.flat_pads(SPEC, pad)
    assert len(flats) == 1 and all(abs(p["z"] - pad["z"]) < 0.5 for p in flats)
