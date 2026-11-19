import numpy as np
import pytest

from tests.test_meshgen import _straight
from yabitsu import features, meshgen, terrain, track


def valley(mode="relative", **kwargs):
    return {
        "id": "test", "section": "test", "center_m": 100,
        "line_xy": [[100, 20], [100, -20]], "source": "合成データ",
        "bed": {"mode": mode, **kwargs}, "width_m": 4,
        "bank_slopes": {"left": 0.5, "right": 2},
        "reach_m": 14, "evidence": ["合成データ"],
    }


class Flat:
    def sample(self, x, y):
        return np.full_like(x, 100.0)


def fixture_grid():
    sec = _straight()
    xs = np.arange(68., 133., 2)
    ys = np.arange(-24., 25., 2)
    x, y = np.meshgrid(xs, ys)
    z, e = terrain.carve(x, y, Flat().sample(x, y), sec)
    return sec, {"xs": xs, "ys": ys, "Z": z, "E": e,
                 "D": abs(y), "H": np.zeros((len(ys)-1, len(xs)-1), dtype=bool)}


def test_empty_table_preserves_all_grid_bytes():
    sec, grid = fixture_grid()
    grid["Z"][0, :2] = (np.nan, -0.0)
    baseline = {key: val.tobytes() for key, val in grid.items()}
    terrain.apply_valleys(sec, grid, [], {"test": (0, 200)}, Flat())
    assert {key: val.tobytes() for key, val in grid.items()} == baseline


@pytest.mark.parametrize("mode, height", [("relative", {"depth_m": 6}),
                                          ("endpoints", {"depth_m": 6}),
                                          ("absolute", {"z_m": 94})])
def test_valley_only_lowers_with_distinct_banks_and_local_support(mode, height):
    sec, grid = fixture_grid()
    start = grid["Z"].copy()
    ix = list(grid["xs"]).index(100)
    iy = list(grid["ys"]).index(14)
    start[iy, ix-1] = grid["Z"][iy, ix-1] = 90
    terrain.apply_valleys(sec, grid, [valley(mode, **height)], {"test": (0, 200)}, Flat())
    z = grid["Z"]
    assert z[iy, ix] == pytest.approx(94)
    assert z[iy, ix-1] == 90
    assert z[iy, ix+2] < z[iy, ix-2]  # 下流を向くと東が左。左の岸は緩い。
    assert np.all(z <= start)
    outside = np.abs(grid["xs"] - 100) >= 14
    assert z[:, outside].tobytes() == start[:, outside].tobytes()
    assert z[np.abs(grid["ys"]) <= 4].tobytes() == start[np.abs(grid["ys"]) <= 4].tobytes()


@pytest.mark.parametrize("tagged", [False, True])
def test_channel_crosses_bridge_and_leaves_approaches_road_and_gutter_intact(tagged):
    sec, grid = fixture_grid()
    start = grid["Z"].copy()
    def meshes():
        return meshgen.road_surface(sec, slice(None)) + meshgen.gutter(sec, slice(None), 1)

    road_before = meshes()
    if tagged:
        sec.bridge[96:105] = True
    spans = [] if tagged else [(96, 104)]
    terrain.apply_valleys(sec, grid, [valley("absolute", z_m=88)],
                          {"test": (0, 200)}, Flat(), bridge_spans=spans)
    x, y = np.meshgrid(grid["xs"], grid["ys"])
    z = grid["Z"]
    assert np.all(z[(x == 100) & (abs(y) <= 6)] == pytest.approx(88))
    protected = (abs(y) <= 4) & (abs(x - 100) >= 4)
    assert z[protected].tobytes() == start[protected].tobytes()
    for old, new in zip(road_before, meshes()):
        np.testing.assert_array_equal(old, new)


def test_s4_valley_side_has_protection_and_no_upward_walls():
    b = track.load_bounds()
    fs = features.load_features_rows(track.load_section_rows("features", b))
    offset = b["s4"][0]
    for s in (6240, 6280, 6330, 6350, 6359, 6380, 6417, 6422):
        active = [f for f in fs if f.s_from <= offset + s <= f.s_to]
        assert not any(f.side == "R" and f.type == "retaining_wall" for f in active), s
        assert any(f.side == "R" and f.type == "guardrail" for f in active), s
    assert any(f.side == "L" and f.type == "retaining_wall"
               and f.s_from <= offset + 6417 <= f.s_to for f in fs)


def test_large_reach_cannot_move_either_site_boundary_or_a_pad():
    sec = _straight(501)
    xs, ys = np.arange(0., 501., 2), np.arange(-170., 171., 2)
    x, y = np.meshgrid(xs, ys)
    z = np.full(x.shape, 100.0)
    pad = {"polygon": [[235, 20], [265, 20], [265, 30], [235, 30]]}
    grid = {"xs": xs, "ys": ys, "Z": z.copy(), "D": np.zeros_like(z),
            "H": np.zeros((len(ys) - 1, len(xs) - 1), bool)}
    row = valley("absolute", z_m=65)
    row.update(center_m=250, line_xy=[[10, 30], [490, 30]], reach_m=180, bank_slopes={"left": 0.1, "right": 0.1})
    terrain.apply_valleys(sec, grid, [row], {"test": (0, 500)}, Flat(), pads=[pad])
    before, after = z, grid["Z"]
    outside = (x < 103) | (x > 397) | (abs(y) >= 148)
    assert after[outside].tobytes() == before[outside].tobytes()
    on_pad = (x >= 234) & (x <= 266) & (y >= 19) & (y <= 31)
    assert after[on_pad].tobytes() == before[on_pad].tobytes()
    assert after[(x == 200) & (y == 30)] == 65


def test_tree_plan_uses_changed_surface_and_removes_bottom_and_water_only():
    from types import SimpleNamespace
    from shapely.geometry import box
    from scripts import align_valley_trees
    from yabitsu.valleys import Valley

    sec, before = fixture_grid()
    after = {**before, "Z": before["Z"].copy()}
    row = valley("absolute", z_m=92)
    terrain.apply_valleys(sec, after, [row], {"test": (0, 200)}, Flat())
    trees = SimpleNamespace(x=np.array([100., 104., 96., 120.]), y=np.full(4, 14.))
    result = align_valley_trees.plan_changes(
        trees, [(before, after)], [Valley.from_row(row)], waters=[(box(103, 13, 105, 15), lambda x, y: x * 0 + 100)])
    assert [(c.index, c.reason) for c in result] == [(0, "谷底"), (1, "水面"), (2, "")]
    assert result[2].new_ground < result[2].old_ground


def test_river_water_follows_excavation_without_moving_other_reaches():
    from shapely.geometry import LineString
    from yabitsu import valleys, water

    sec = _straight(601)
    row = valley("absolute", z_m=92)
    row["bed"]["z_m"] = [99.7, 94.2, 93.7]
    row.update(center_m=300, line_xy=[[200, 15], [300, 15], [400, 15]], water={"osm_id": 55, "depth_m": 0.65})
    stream = water.Stream(LineString([(0, 15), (600, 15)]), 6, "river", [106, 88], 55)
    unrelated = water.Stream(stream.line, 6, "river", stream.z, 56)
    valley_row = valleys.Valley.from_row(row)
    def ground(x, y):
        at = np.clip(x, 200, 400)
        return np.interp(at, [200, 300, 400], [99.7, 94.2, 93.7]) + np.maximum(abs(y - 15) - 0.8, 0)

    levels = water.valley_surface_levels(sec, {"test": (0, 600)}, [valley_row], [stream], ground)
    x, y = np.arange(601.), np.full(601, 15.)
    old, new = stream.z_fn()(x, y), water.surface_z(stream, levels)(x, y)
    assert new[300] < old[300] - 2
    assert np.all(new <= old)
    assert np.all(np.diff(new) <= 0)
    untouched = (x <= 200) | (x >= 400)
    assert new[untouched].tobytes() == old[untouched].tobytes()
    assert water.surface_z(unrelated, levels)(x, y).tobytes() == old.tobytes()


def test_tree_tool_rejects_changes_outside_the_allowed_window(tmp_path):
    from scripts.align_valley_trees import ground_pairs
    from yabitsu.valleys import Valley

    sec = _straight(501)
    xs, ys = np.arange(0., 501., 2), np.arange(-20., 21., 2)
    before = {"xs": xs, "ys": ys, "Z": np.zeros((len(ys), len(xs))),
              "D": np.zeros((len(ys), len(xs))), "H": np.zeros((len(ys)-1, len(xs)-1), bool)}
    after = {**before, "Z": before["Z"].copy()}
    after["Z"][8, 125] = -5
    row = valley("absolute", z_m=-5)
    row["center_m"] = 250
    args = (tmp_path / "before.npz", tmp_path / "after.npz", sec, [Valley.from_row(row)], {"test": (0, 500)})
    np.savez(args[0], **before)
    np.savez(args[1], **after)
    assert len(ground_pairs(*args)) == 1
    after["Z"][8, 2] = -1
    np.savez(args[1], **after)
    with pytest.raises(ValueError, match="範囲|3か所の外"):
        ground_pairs(*args)


def test_diagonal_bridge_opens_every_vertex_between_end_sections():
    from dataclasses import replace
    import shapely

    sec, grid = fixture_grid()
    angle = np.deg2rad(30)
    sec = replace(sec, x=100 + (sec.x - 100) * np.cos(angle),
                  y=(sec.x - 100) * np.sin(angle), heading=sec.heading + angle)
    x, y = np.meshgrid(grid["xs"], grid["ys"])
    grid["Z"][:] = 100
    row = valley("absolute", z_m=90)
    row.update(line_xy=[[96, 22], [104, -22]], width_m=8, reach_m=18)
    spans = [(96, 104)]
    station = (x - 100) * np.cos(angle) + y * np.sin(angle)
    lateral = -(x - 100) * np.sin(angle) + y * np.cos(angle)
    # 谷底に入り、橋端をかすめる面を持つ格子点も開ける。
    floor = shapely.dwithin(shapely.LineString(row["line_xy"]), shapely.points(x, y), 4)
    opened = floor & (abs(station) < 4) & (abs(lateral) < 4)
    terrain.apply_valleys(sec, grid, [row], {"test": (0, 200)}, Flat(), bridge_spans=spans)
    assert np.all(grid["Z"][opened] == 90), grid["Z"][opened]
    assert np.all(grid["Z"][(abs(lateral) < 4) & (abs(station) >= 4)] == 100)


def test_feathered_valley_is_byte_identical_on_second_application():
    sec, grid = fixture_grid()
    args = (sec, grid, [valley("absolute", z_m=91)], {"test": (0, 200)}, Flat())
    terrain.apply_valleys(*args)
    once = grid["Z"].tobytes()
    assert sum(terrain.apply_valleys(*args).values()) == 0
    assert grid["Z"].tobytes() == once


@pytest.mark.parametrize("state", ["stale", "applied", "original"])
def test_before_near_checks_every_array_and_skips_applied(tmp_path, state):
    from scripts import track_06_terrain

    path = tmp_path / "near.npz"
    before = tmp_path / "before.npz"
    np.savez(before, Z=[100.], D=[0.])
    np.savez(path, Z=[94. if state == "applied" else 100.],
             D=[-0. if state == "stale" else 0.])
    before_bytes = path.read_bytes()
    before_inode = path.stat().st_ino

    def excavate(grid):
        grid["Z"][:] = 94.

    if state == "stale":
        with pytest.raises(ValueError, match="一致"):
            track_06_terrain.apply_before_near(before, path, excavate)
    else:
        track_06_terrain.apply_before_near(before, path, excavate)
        with np.load(path) as result:
            assert result["Z"][0] == 94.
    if state != "original":
        assert path.stat().st_ino == before_inode
        assert path.read_bytes() == before_bytes


def test_clear_fubuki_announces_that_valley_tree_alignment_must_follow(monkeypatch, capsys):
    import sys
    from scripts import clear_fubuki_trees

    monkeypatch.setattr(sys, "argv", ["clear_fubuki_trees", "--help"])
    with pytest.raises(SystemExit, match="0"):
        clear_fubuki_trees.main()
    text = capsys.readouterr().out.split("実行順は", 1)[1]
    assert text.index("clear_fubuki_trees") < text.index("plant_fubuki_shrubs") < text.index("align_valley_trees")


def test_diagonal_approach_preserves_terrain_triangles_past_the_end_band():
    from dataclasses import replace

    sec, grid = fixture_grid()
    grid["Z"][:] = 100.
    angle = np.deg2rad(30)
    cs, sn = np.cos(angle), np.sin(angle)
    sec = replace(sec, x=100 + (sec.x - 100) * cs, y=(sec.x - 100) * sn, heading=sec.heading + angle)
    row = valley("absolute", z_m=90)
    row.update(line_xy=[[96, 22], [104, -22]], width_m=8, reach_m=18)
    terrain.apply_valleys(sec, grid, [row], {"test": (0, 200)}, Flat(), bridge_spans=[(96, 104)])
    ground = terrain.terrain_mesh_z(grid, grid)
    s, q = np.meshgrid(np.r_[np.arange(-7., -4., .25), np.arange(4.25, 7.25, .25)],
                       np.array([-3.6, 3.6]))
    x, y = 100 + s * cs - q * sn, s * sn + q * cs
    previous = np.full_like(x, 100.)
    # 斜めの両端では隣の三角形が橋内に残る。帯の外端では元の地面に戻る。
    assert np.allclose(ground(x[:, [0, -1]], y[:, [0, -1]]), previous[:, [0, -1]])
    assert np.all(ground(x, y) <= previous)
