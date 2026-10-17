import math

import numpy as np
import pytest

from yabitsu import geo, layouts as lay
from yabitsu import meshgen
from tests.test_meshgen import _sec_from_xy, _straight

NAMES = {f"AC_START_{i}" for i in range(4)} | {f"AC_PIT_{i}" for i in range(4)} | {
    "AC_HOTLAP_START_0", "AC_OPEN_START_L", "AC_OPEN_START_R", "AC_OPEN_FINISH_L", "AC_OPEN_FINISH_R"}
LOT = {"name": "lot", "role": "flat", "polygon": [[10, 10], [40, 10], [40, 40], [10, 40], [10, 10]], "z": 100.0}


def _ll(x, y):
    lat, lon = geo.from_local(x, y)
    return (float(lat), float(lon))


def _layout(direction, pits="lot", grid=40.0, hotlap=5.0, start=50.0, finish=900.0, section="s1"):
    if direction < 0:
        grid, hotlap, start, finish = 1000 - grid, 1000 - hotlap, 1000 - start, 1000 - finish
    return lay.Layout("t", "test", section, direction, _ll(grid, 0), _ll(hotlap, 0), _ll(start, 0), _ll(finish, 0), pits)


def _by(objs):
    return {d.name: d for d in objs}


BOUNDS = {"s1": (0.0, 1000.0)}


@pytest.mark.parametrize("direction", [1, -1])
def test_layout_names_and_order(direction):
    sec = _straight(1001)
    d = _by(lay.layout_objects(sec, [LOT], _layout(direction), BOUNDS))
    assert set(d) == NAMES
    along = lambda n: d[n].location[0] * direction
    assert along("AC_HOTLAP_START_0") < along("AC_START_0") < along("AC_OPEN_START_L") < along("AC_OPEN_FINISH_L")
    expected = 0.0 if direction > 0 else math.pi
    assert abs(math.remainder(d["AC_START_0"].heading - expected, 2 * math.pi)) < 1e-6


def test_positions_are_exact_not_snapped_to_samples():
    sec = _straight(1001)
    d = _by(lay.layout_objects(sec, [LOT], _layout(1, grid=40.4), BOUNDS))
    assert abs(d["AC_START_0"].location[0] - 40.4) < 0.01


def test_pits_use_pad_when_it_fits():
    sec = _straight(1001)
    d = _by(lay.layout_objects(sec, [LOT], _layout(1), BOUNDS))
    assert all(10 < d[f"AC_PIT_{i}"].location[0] < 40 and 10 < d[f"AC_PIT_{i}"].location[1] < 40 for i in range(4))


def test_pits_come_from_named_pad():
    sec = _straight(1001)
    lane = {"name": "pass_lane", "role": "end", "polygon": LOT["polygon"], "z": 100.0,
            "pit_points": [(500.0 + i, 20.0, 100.0, 0.0) for i in range(4)]}
    lot = {**LOT, "name": "nanohanadai_lot", "pit_center": [25.0, 25.0], "heading": 0.5, "pit_spacing": 3.0}
    a = _by(lay.layout_objects(sec, [lot, lane], _layout(1, pits="pass_lane"), BOUNDS))
    b = _by(lay.layout_objects(sec, [lot, lane], _layout(1, pits="nanohanadai_lot"), BOUNDS))
    assert a["AC_PIT_0"].location[0] == 500.0
    assert abs(b["AC_PIT_0"].heading - 0.5) < 1e-9 and 15 < b["AC_PIT_0"].location[0] < 35


def test_missing_pit_pad_is_an_error():
    sec = _straight(1001)
    with pytest.raises(ValueError):
        lay.layout_objects(sec, [LOT], _layout(1, pits="nope"), BOUNDS)


def test_s_of_picks_point_on_requested_section():
    # 下の区間（y=0 を +x）→ ヘアピン → 上の区間（y=13 を −x）。両方の枝に近い点
    x = np.r_[np.arange(0.0, 300.0), 300 + 6.5 * np.sin(np.linspace(0, np.pi, 21))[1:-1], np.arange(300.0, -1.0, -1.0)]
    y = np.r_[np.zeros(300), 6.5 - 6.5 * np.cos(np.linspace(0, np.pi, 21))[1:-1], np.full(301, 13.0)]
    sec = _sec_from_xy(x, y)
    b0 = float(sec.s[np.argmin(np.hypot(sec.x - 300, sec.y - 6.5))])
    bounds = {"s2": (0.0, b0), "s1": (b0, float(sec.s[-1]))}
    p = _ll(200.0, 5.0)
    s_low = lay.s_of(sec, p, bounds, "s2"); s_up = lay.s_of(sec, p, bounds, "s1")
    assert s_low < b0 and abs(s_low - 200.0) < 1.0
    assert s_up > b0 and abs(sec.y[np.argmin(abs(sec.s - s_up))] - 13.0) < 0.5


def test_timed_length():
    sec = _straight(1001)
    assert abs(lay.timed_length(sec, _layout(1, start=50.0, finish=900.0), BOUNDS) - 850.0) < 0.01


def test_cameras_cover_track_in_order():
    sec = _straight(1001)
    for direction in (1, -1):
        cams = lay.cameras(sec, _layout(direction))
        ins = [c.in_point for c in cams]
        assert ins == sorted(ins) and ins[0] == 0.0 and cams[-1].out_point == 1.0


def test_pad_mesh_triangles_point_up():
    v, f, uv = meshgen.pad_mesh([[0, 0], [10, 0], [10, 5], [0, 5], [0, 0]], 50.0)
    assert f.shape[1] == 3 and (meshgen.face_normals(v, f)[:, 2] > 0).all() and np.allclose(v[:, 2], 50.0)


def test_load_layouts_reads_json(tmp_path):
    p = tmp_path / "layouts.json"
    p.write_text('[{"name": "s1_up", "title": "T", "section": "s1", "direction": 1, "grid": [35.4, 139.2],'
                 ' "hotlap": [35.4, 139.2], "timing_start": [35.4, 139.2], "finish": [35.4, 139.2], "pits": "lot"}]',
                 encoding="utf-8")
    (l,) = lay.load_layouts(p)
    assert l.name == "s1_up" and l.direction == 1 and l.grid == (35.4, 139.2)


_REAL = __import__("yabitsu.config", fromlist=["x"]).DATA_PROCESSED / "track_section.json"


@pytest.mark.skipif(not _REAL.exists(), reason="処理済みデータがまだない")
def test_real_layouts_start_on_the_road_and_finish_ahead():
    import json
    from yabitsu import config, track
    from yabitsu.crosssection import Section
    sec = Section.load_json(_REAL); pads = json.loads((config.DATA_PROCESSED / "track_pads.json").read_text(encoding="utf-8"))
    bounds = track.load_bounds()
    ls = lay.load_layouts()
    assert {l.name for l in ls} == {"s1_up", "s1_down", "s2_up", "s2_down"}
    for l in ls:
        objs = _by(lay.layout_objects(sec, pads, l, bounds))
        for i in range(4):
            d = objs[f"AC_START_{i}"]
            k = int(np.argmin(np.hypot(sec.x - d.location[0], sec.y - d.location[1])))
            assert np.hypot(sec.x[k] - d.location[0], sec.y[k] - d.location[1]) <= max(sec.hw_left[k], sec.hw_right[k]), (l.name, i)
        s_start = lay.s_of(sec, l.timing_start, bounds, l.section); s_fin = lay.s_of(sec, l.finish, bounds, l.section)
        s_grid = lay.s_of(sec, l.grid, bounds, l.section)
        assert (s_fin - s_start) * l.direction > 1000, l.name
        assert 0 < (s_start - s_grid) * l.direction < 80, l.name  # s1_down はユーザーが選んだ地点で 60m
