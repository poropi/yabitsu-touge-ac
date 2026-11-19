from dataclasses import replace

import numpy as np
import pytest

from tests.test_meshgen import _straight
from yabitsu import config, features


def _s5_features():
    return features.load_features_csv(config.DATA_MANUAL / "s5_features.csv")


def _right_walls():
    return [f for f in _s5_features() if f.type == "retaining_wall" and f.side == "R"
            and 10300 <= f.s_from < 10507]


def test_right_wall_height_affects_only_wall_mesh():
    sec = _straight()
    wall = replace(_right_walls()[0], s_from=20, s_to=40)
    old = replace(wall, height_m=4)
    v, _, _ = features.retaining_wall_mesh(sec, wall)
    assert v[:, 2].max() - sec.edge_z(-1)[20] == pytest.approx(2.7)
    for fn in (features.cut_slopes, features.cut_drops, features.sidewalk_offsets):
        for before, after in zip(fn(sec, [old]), fn(sec, [wall])):
            np.testing.assert_array_equal(after, before)


def _local_terrain(s_from, s_to, initial=100.0):
    sec = _straight(s_to - s_from + 1)
    sec.s[:] += s_from
    xs = np.arange(sec.n, dtype=float)
    ys = np.arange(-24.0, 9.0)
    return sec, {"xs": xs, "ys": ys, "Z": np.full((len(ys), len(xs)), initial)}


def test_gate_opening_removes_wall_and_has_a_level_apron():
    from scripts import track_06_terrain as build

    assert sorted((f.s_from, f.s_to) for f in _right_walls()) == [(10338, 10405), (10436, 10507)]
    sec, near = _local_terrain(10400, 10440, initial=104.0)
    before = near["Z"].copy()
    walk = features.Feature(10400, 10440, "R", "sidewalk", 0, "", width_m=1.5)
    build.shape_fubuki_gate(sec, near, [walk], {"s5": (0, 12000)})
    assert near["Z"][14, 13] == pytest.approx(sec.edge_z(-1)[13] + features.CURB_HEIGHT - 0.03)
    np.testing.assert_array_equal(near["Z"][:, :6], before[:, :6])
    np.testing.assert_array_equal(near["Z"][:, 35:], before[:, 35:])
    np.testing.assert_array_equal(near["Z"][19:, :], before[19:, :])


def test_tree_filter_keeps_the_gate_far_side_and_outside_section():
    from yabitsu import fubuki

    sec = _straight(12000)
    points = np.array([
        [10337, -10], [10338, -10], [10410, -10], [10480, -10],
        [10490, 10], [10501, -17], [10507, -10], [10508, -10], [10480, -4.8],
    ])
    trees = np.rec.fromarrays(points.T, names="x,y")
    keep = fubuki.keep_slope_trees(sec, trees, _s5_features(), 0)
    np.testing.assert_array_equal(keep, [True, False, True, False, True, True, False, True, True])


def test_fill_reaches_the_crest_and_stops_before_bridge():
    from scripts import track_06_terrain as build

    sec, near = _local_terrain(10480, 10512, initial=98.0)
    before = near["Z"].copy()
    wall = replace(_right_walls()[-1], s_from=10480)
    walk = features.Feature(10480, 10507, "R", "sidewalk", 0, "", width_m=1.5)
    build.raise_fubuki_slope(sec, near, [wall, walk], {"s5": (0, 12000)})
    actual = near["Z"].copy()
    near["Z"][:] = before
    build.raise_fubuki_slope(sec, near, [replace(wall, height_m=4), walk], {"s5": (0, 12000)})
    assert near["Z"][17, 20] > actual[17, 20]  # 右 7 m、壁が高ければ盛り土の法尻も高い
    assert near["Z"][12, 23] - sec.z[23] == pytest.approx(6.5, abs=0.2)
    np.testing.assert_array_equal(near["Z"][:, :6], before[:, :6])
    np.testing.assert_array_equal(near["Z"][:, 27:], before[:, 27:])
    np.testing.assert_array_equal(near["Z"][19:, :], before[19:, :])
    np.testing.assert_array_equal(near["Z"][0, :], before[0, :])


def test_terrain_triangles_stay_behind_the_battered_wall():
    from scripts import track_06_terrain as build
    from yabitsu import terrain

    sec, near = _local_terrain(10476, 10510, initial=108.0)
    near["ys"] = np.arange(-24.0, 10.0, 2)
    near["Z"] = np.full((len(near["ys"]), len(near["xs"])), 108.0)
    near["D"] = np.zeros_like(near["Z"])
    near["Z"][near["ys"] >= -4] = 99.8
    before = near["Z"].copy()
    wall = replace(_right_walls()[-1], s_from=10476)
    walk = features.Feature(10476, 10507, "R", "sidewalk", 0, "", width_m=1.5)
    build.clear_fubuki_wall_face(sec, near, [walk, wall], {"s5": (0, 12000)})
    ground = terrain.terrain_mesh_z(near, near)
    foot = 3.0 + features.GUTTER_WIDTH + 1.5
    d = np.linspace(foot, foot + features.WALL_LEAN * 2.7, 41)
    face_z = sec.edge_z(-1)[14] - 0.3 + (d - foot) * 3.0 / (features.WALL_LEAN * 2.7)
    assert np.all(ground(np.full_like(d, 14.2), -d) < face_z)
    np.testing.assert_array_equal(near["Z"][near["ys"] >= -3.6], before[near["ys"] >= -3.6])
    np.testing.assert_array_equal(near["Z"][:, :5], before[:, :5])
    np.testing.assert_array_equal(near["Z"][:, 32:], before[:, 32:])
    np.testing.assert_array_equal(near["Z"][0], before[0])


def _landmark_fixture(folder, name, length=2.0):
    from PIL import Image

    folder.mkdir(parents=True, exist_ok=True)
    np.savez(folder / f"{name}.npz", v=[[0, 0, 0], [length, 0, 0], [0, 0, 1]],
             f=[[0, 1, 2]], uv=[[0, 0], [1, 0], [0, 1]])
    Image.new("RGB", (2, 2)).save(folder / f"{name}.png")


def test_slope_fence_follows_ground_and_skips_missing_assets(tmp_path, monkeypatch):
    from yabitsu import fubuki, landmarks

    monkeypatch.setattr(landmarks, "DIR", tmp_path)
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    sec = _straight(10700)
    ground = lambda x, y: 10.0 + x / 100.0 - y / 10.0
    assert fubuki.slope_fence_meshes(sec, 0, ground) == []
    folder = tmp_path / "r70_fubuki"
    _landmark_fixture(folder, "slope_fence_wire_end", length=0.05)
    assert fubuki.slope_fence_meshes(sec, 0, ground) == []
    for name in ("slope_fence_wire_2m", "slope_fence_wire_end", "slope_fence_bar_2m"):
        _landmark_fixture(folder, name, length=0.05 if name.endswith("_end") else 2.0)
    meshes = fubuki.slope_fence_meshes(sec, 0, ground)
    assert [part for _, part, *_ in meshes] == [
        "slope_fence_wire_2m", "slope_fence_wire_end", "slope_fence_wire_2m",
        "slope_fence_wire_end", "slope_fence_bar_2m",
    ]
    for _, part, v, f, _ in meshes:
        foot = v[f[:, :2].ravel()]
        np.testing.assert_allclose(foot[:, 2], ground(foot[:, 0], foot[:, 1]))
        assert np.all((-14.01 <= foot[:, 1]) & (foot[:, 1] <= -11.99))
        assert not np.any((foot[:, 0] > 10405.08) & (foot[:, 0] < 10434.92))
        if part == "slope_fence_bar_2m":
            assert (foot[:, 0].min(), foot[:, 0].max()) == (10500, 10504)


def test_kamatagawa_widens_right_girder_rail_and_collision_only():
    from yabitsu import bridge_parts

    bridges = bridge_parts.load(config.DATA_MANUAL / "s5_bridges.json")
    target = next(b for b in bridges if b["osm_id"] == 131561912)
    other = next(b for b in bridges if b["osm_id"] != 131561912)
    sec = _straight()
    ranges = {target["osm_id"]: (20, 60), other["osm_id"]: (80, 120)}
    ground = lambda x, y: np.full_like(x, 90)
    old_target = {k: v for k, v in target.items() if k != "right_extension_m"}
    before = {n: v for n, v, *_ in bridge_parts.meshes(sec, [old_target, other], ranges, ground)}
    after = {n: v for n, v, *_ in bridge_parts.meshes(sec, [target, other], ranges, ground)}
    for name in ("bridge_131561912_rail_R", "1WALL_bridge_131561912_rail_R", "bridge_131561912_post_aR"):
        np.testing.assert_allclose(after[name][:, 1], before[name][:, 1] - 2.9)
    name = "bridge_131561912_girder"
    np.testing.assert_allclose(after[name][::4, 1], before[name][::4, 1] - 2.9)
    np.testing.assert_array_equal(after[name][2::4], before[name][2::4])
    for name in before:
        if str(other["osm_id"]) in name or name == "bridge_131561912_rail_L":
            np.testing.assert_array_equal(after[name], before[name])


def test_kamatagawa_walkway_rows_touch_across_fractional_chainage(tmp_path, monkeypatch):
    from yabitsu import assembly, bridge_parts, fubuki, landmarks
    from yabitsu.manifest import Manifest

    monkeypatch.setattr(landmarks, "DIR", tmp_path / "parts")
    _landmark_fixture(landmarks.DIR / "r70_fubuki", "red_arch_bridge_v2")
    bridges = bridge_parts.load(config.DATA_MANUAL / "s5_bridges.json")
    feats, _ = fubuki.kamatagawa_parts(_s5_features(), bridges, 0)
    walks = [f for f in feats if f.type == "sidewalk" and f.side == "R" and 10480 <= f.s_from < 10684]
    assert sorted((f.s_from, f.s_to, f.width_m) for f in walks) == [
        (10480, 10507, 1.5), (10507, 10607, 2.9), (10607, 10684, 2.7),
    ]
    sec = _straight(10700)
    sec.s[:] += 0.96
    manifest = Manifest(tmp_path)
    assembly.add_features(manifest, sec, sorted(walks, key=lambda f: f.s_from),
                          right_walk_joins=fubuki.walk_joins(0))
    tops = [np.load(tmp_path / manifest.meshes[f"1ROAD_track_sidewalk_{k:03d}"]["file"])["verts"]
            for k in range(3)]
    for a, b in zip(tops, tops[1:]):
        assert a[:, 0].max() == b[:, 0].min()
        # 同じ継ぎ目で内側の点が一致するため、歩ける幅が途切れない。
        np.testing.assert_array_equal(a[a[:, 0] == a[:, 0].max()].max(axis=0),
                                      b[b[:, 0] == b[:, 0].min()].max(axis=0))


def test_kamatagawa_prefers_complete_v2_and_falls_back_to_original(tmp_path, monkeypatch):
    import json
    from scripts import dress_mountain
    from yabitsu import landmarks

    monkeypatch.setattr(landmarks, "DIR", tmp_path)
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    spec = json.loads((config.DATA_MANUAL / "s45_route70_bridges.json").read_text())
    bridge = next(b for b in spec["bridges"] if b.get("whole") == "red_arch_bridge")
    assert dress_mountain.whole_bridge_part(bridge) == ("r70_bridges", "red_arch_bridge")
    _landmark_fixture(tmp_path / "r70_fubuki", "red_arch_bridge_v2")
    assert dress_mountain.whole_bridge_part(bridge) == ("r70_fubuki", "red_arch_bridge_v2")
    (tmp_path / "r70_fubuki" / "red_arch_bridge_v2.png").unlink()
    assert dress_mountain.whole_bridge_part(bridge) == ("r70_bridges", "red_arch_bridge")


@pytest.mark.parametrize("thickness", [0.22, 0.40])
def test_white_wall_starts_on_left_retaining_wall_cap(tmp_path, monkeypatch, thickness):
    import json

    from yabitsu import fubuki, landmarks

    monkeypatch.setattr(landmarks, "DIR", tmp_path)
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    sec = _straight(12000)
    feats = _s5_features()
    assert fubuki.wall_top_meshes(sec, 0, feats) == []
    folder = tmp_path / "r70_fubuki"
    name = "wall_top_concrete_2m"
    _landmark_fixture(folder, name)
    (folder / "asset.json").write_text(json.dumps({"parts": [{"name": name, "dimensions_m": [2, thickness, 1]}]}))
    np.savez(folder / f"{name}.npz", v=[[0, -thickness/2, 0], [2, -thickness/2, 0], [0, thickness/2, 0]],
             f=[[0, 1, 2]], uv=[[0, 0], [1, 0], [0, 1]])
    (mesh,) = fubuki.wall_top_meshes(sec, 0, feats)
    _, name, v, faces, _ = mesh
    assert name == "wall_top_concrete_2m"
    foot = v[faces[:, :2].ravel()]
    assert (foot[:, 0].min(), foot[:, 0].max()) == pytest.approx((10610, 10675))
    height = next(f.height_m for f in feats if f.side == "L" and f.s_from == 10610)
    front = 3 + features.GUTTER_WIDTH + features.WALL_LEAN * height
    assert v[:, 1].min() == pytest.approx(front)
    assert v[:, 1].max() == pytest.approx(front + thickness)
    np.testing.assert_allclose(foot[:, 2], sec.edge_z(1)[10610] + height)


def test_portal_guardrail_uses_existing_wire_mesh_variant(tmp_path, monkeypatch):
    from scripts import dress_mountain
    from yabitsu import landmarks
    from yabitsu.manifest import Manifest

    monkeypatch.setattr(landmarks, "DIR", tmp_path / "landmarks")
    for name in ("w_beam", "w_beam_end", "pipe", "pipe_mesh", "pipe_end"):
        _landmark_fixture(landmarks.DIR / "r70_guardrail", name)
    monkeypatch.setattr(dress_mountain, "gaps", lambda sec: [])
    monkeypatch.setattr(dress_mountain.dm, "leaf_meshes", lambda *args, **kwargs: [])

    sec, _ = _local_terrain(10600, 10690)
    sec.s[:] += 0.4
    bounds = {name: (0, 0) for name in ("s2", "s1", "s4")}
    bounds["s5"] = 0, 10690
    f = next(f for f in _s5_features() if f.type == "guardrail" and f.s_from == 10610)
    root = tmp_path / "parts"
    m = Manifest(root)
    hide = dress_mountain.build_parts(sec, bounds, [f], dress_mountain.table_rows(), m)
    assert "GUARDRAIL_s1_000" in hide
    wire = [e for e in m.meshes.values() if e["material"] == "mt_r70_rail_pipe_mesh"]
    assert len(wire) == 1
    v = np.load(root / wire[0]["file"])["verts"]
    assert (v[:, 0].min(), v[:, 0].max()) == pytest.approx((54.6, 83.6))
    assert np.all(v[:, 1] < -5.5)
    pipe, = [e for e in m.meshes.values() if e["material"] == "mt_r70_rail_pipe"]
    v = np.load(root / pipe["file"])["verts"]
    assert (v[:, 0].min(), v[:, 0].max()) == pytest.approx((9.6, 54.6))
    ends = [e for e in m.meshes.values() if e["material"] == "mt_r70_rail_pipe_end"]
    assert len(ends) == 2
    outer_x = np.concatenate([np.load(root / e["file"])["verts"][:, 0] for e in ends])
    assert (outer_x.min(), outer_x.max()) == pytest.approx((9.6, 83.6))
    assert not any(e["material"] == "mt_r70_rail_span_rust" for e in m.meshes.values())


def test_replaced_bridge_hides_its_reverse_rail_faces_but_keeps_collision(tmp_path, monkeypatch):
    import json

    from scripts import dress_mountain
    from yabitsu import landmarks
    from yabitsu.manifest import Manifest

    real = json.loads((config.DATA_MANUAL / "s45_route70_bridges.json").read_text())
    target = next(b for b in real["bridges"] if b.get("whole") == "red_arch_bridge")
    tables = tmp_path / "manual"
    tables.mkdir()
    (tables / "s45_route70_bridges.json").write_text(json.dumps({"bridges": [target], "tunnels": []}))
    monkeypatch.setattr(config, "DATA_MANUAL", tables)
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(landmarks, "DIR", tmp_path / "landmarks")
    _landmark_fixture(landmarks.DIR / "r70_bridges", "red_arch_bridge")

    sec, _ = _local_terrain(10500, 10620)
    source = Manifest(tmp_path / "base")
    names = ("bridge_131561912_girder", "bridge_131561912_rail_L", "bridge_131561912_rail_R",
             "bridge_131561912_arch", "1WALL_bridge_131561912_rail_R", "bridge_131561917_rail_R")
    for name in names:
        source.add_mesh("s5", name, [[7, -4, 99], [107, -4, 99], [50, 4, 99]],
                        [[0, 1, 2]], [[0, 0], [1, 0], [0, 1]], "test",
                        renderable=not name.startswith("1WALL_"))
    source.save()
    monkeypatch.setattr(dress_mountain, "MANIFEST", source.root)
    output = Manifest(tmp_path / "dress")
    tex = output.root / "textures"
    tex.mkdir()
    hide = dress_mountain.add_r70_structures(sec, {"s4": (0, 0), "s5": (0, 12000)}, output, tex)
    assert set(hide) == set(names[:4]) | {f"{name}_back" for name in names[1:4]}


def test_fubuki_parts_are_ready_from_the_delivered_asset_directory(tmp_path, monkeypatch):
    from scripts import dress_mountain
    from yabitsu import fubuki, landmarks

    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(landmarks, "DIR", tmp_path / "processed")
    delivered = tmp_path / "assets" / "landmarks" / "r70_fubuki"
    _landmark_fixture(delivered, "slope_fence_bar_2m")
    _landmark_fixture(delivered, "red_arch_bridge_v2")
    sec = _straight(10700)
    meshes = fubuki.slope_fence_meshes(sec, 0, lambda x, y: np.zeros_like(x))
    assert [mesh[1] for mesh in meshes] == ["slope_fence_bar_2m"]
    assert landmarks.texture("r70_fubuki", "red_arch_bridge_v2") == delivered / "red_arch_bridge_v2.png"
    bridge = {"whole": "red_arch_bridge", "preferred_whole": {"asset": "r70_fubuki", "name": "red_arch_bridge_v2"}}
    assert dress_mountain.whole_bridge_part(bridge) == ("r70_fubuki", "red_arch_bridge_v2")


def test_gate_split_preserves_indices_of_unrelated_features():
    feats = _s5_features()
    first, second = _right_walls()
    before = [replace(first, s_from=10300, s_to=10507) if f == first else f
              for f in feats if f != second and not (f.type == "sidewalk" and f.s_from == 10507)]
    old_branch_gap = {"R": [(10414.0, 10429.0)]}

    def indexed_outside(items):
        return [(k, f) for k, f in enumerate(features.cut_at(items, old_branch_gap)) if f.s_from > 10684]

    assert indexed_outside(feats) == indexed_outside(before)


def test_gate_does_not_raise_ground_under_the_house():
    from scripts import track_06_terrain as build

    sec, near = _local_terrain(10400, 10445, initial=99.5)
    before = near["Z"].copy()
    build.shape_fubuki_gate(sec, near, [], {"s5": (0, 12000)})
    # 床の手前側の角（S5 10430.8 m、右 8.8 m）を含む格子の四隅。
    np.testing.assert_array_equal(near["Z"][15:17, 30:32], before[15:17, 30:32])


def test_gate_ground_mesh_stays_below_stub_plane():
    from scripts import track_06_terrain as build
    from yabitsu import terrain

    sec, near = _local_terrain(10400, 10445, initial=101.0)
    near["D"] = np.zeros_like(near["Z"])
    pad = {"name": "side_stub_595991910_0", "kind": "stub", "z": 99.9,
           "plane": [15, -4.1, 99.9, 0, 0.03],
           "polygon": [[14.6, -4.1], [15.4, -4.1], [15.4, -14.1], [14.6, -14.1]]}
    build.shape_fubuki_gate(sec, near, [], {"s5": (0, 12000)}, [pad])
    x, y = np.meshgrid(np.linspace(14.6, 15.4, 5), np.linspace(-14.1, -4.1, 41))
    ground = terrain.terrain_mesh_z(near, near)
    excess = (ground(x, y) - (99.9 + 0.03 * (y + 4.1))).max()
    assert excess <= 1e-9, f"分かれ道の路面との高低差 {excess:+.3f} m"


def test_slope_fence_avoids_route70_and_remaining_building_floors(tmp_path, monkeypatch):
    import json

    from shapely.geometry import MultiPoint, Polygon

    from yabitsu import fubuki, geo, landmarks, track
    from yabitsu.crosssection import Section

    sec = Section.load_json(config.DATA_PROCESSED / "track_section.json")
    s5 = track.load_bounds()["s5"][0]
    buildings = json.loads((config.DATA_MANUAL / "s45_route70.json").read_text())["landmarks"]
    floors = [(b["name"], MultiPoint(list(zip(*geo.to_local(*np.asarray(b["z_at"]).T)))).convex_hull)
              for b in buildings if b["asset"] == "r70_buildings"]
    originals = json.loads((config.DATA_MANUAL / "track_buildings.json").read_text())
    floors += [(b["id"], Polygon(list(zip(*geo.to_local(*np.asarray(b["footprint"]).T)))))
               for b in originals if not b.get("gone")]
    monkeypatch.setattr(landmarks, "DIR", tmp_path)
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    for run in fubuki.load()["slope_fences"]:
        for name in (run["part"], run.get("end")):
            if name:
                _landmark_fixture(landmarks.DIR / "r70_fubuki", name, length=0.03 if name.endswith("_end") else 2)
    for tag, _, v, faces, _ in fubuki.slope_fence_meshes(sec, s5, lambda x, y: np.zeros_like(x)):
        for face in faces:
            panel = MultiPoint(v[face, :2]).convex_hull.buffer(0.04)
            assert not (hits := [name for name, floor in floors if panel.intersects(floor)]), (tag, hits)


def test_slope_rises_behind_wall_after_the_gate():
    from scripts import track_06_terrain as build

    sec, near = _local_terrain(10430, 10465)
    wall = replace(_right_walls()[-1], s_to=10465)
    walk = features.Feature(10430, 10465, "R", "sidewalk", 0, "", width_m=1.5)
    build.raise_fubuki_slope(sec, near, [wall, walk], {"s5": (0, 12000)})
    height = near["Z"][17, 17] - sec.edge_z(-1)[17]
    assert height >= wall.height_m, f"S5 10447 m、右 7 m の地面は車道の縁から {height:+.2f} m"
    assert near["Z"][10, 9] == 100.0  # S5 10439 m、右 14 m の家の床


def test_tree_cleanup_grounds_retained_trees_on_rendered_triangles(tmp_path, monkeypatch):
    import sys

    from types import SimpleNamespace

    from scripts import clear_fubuki_trees
    from yabitsu import forest_ac, track
    from yabitsu.crosssection import Section

    sec, near = _local_terrain(10330, 10512, initial=97.0)
    near["D"] = np.zeros_like(near["Z"])
    near["Z"][6, 177] = 109.0  # 補間が双一次だと出ない値が対角線上に乗る
    np.savez(tmp_path / "track_terrain_near.npz", **near)
    np.savez(tmp_path / "track_terrain_mid.npz", **near)
    trees = np.rec.fromarrays(
        [[86.5, 172.5, 176.5, 173, 171], [-10.8, -14.0, -18.5, 14, -11], [101., 90., 96., 100., 100.]],
        names="x,y,z",
    )
    np.save(tmp_path / "trees.npy", trees)
    monkeypatch.setattr(config, "DATA_PROCESSED", tmp_path)
    monkeypatch.setattr(Section, "load_json", lambda *_: sec)
    monkeypatch.setattr(track, "load_bounds", lambda: {"s5": (0, 12000)})
    monkeypatch.setattr(track, "load_section_rows", lambda *_: [])
    monkeypatch.setitem(sys.modules, "scripts.forest_build", SimpleNamespace(old_tree_meshes=lambda: []))
    written = []
    monkeypatch.setattr(forest_ac, "write_lists", lambda trees, *_: written.append(trees.copy()) or [])
    monkeypatch.setattr(forest_ac, "write_ext_config", lambda *_: None)
    monkeypatch.setattr(sys, "argv", ["clear_fubuki_trees", str(tmp_path), "--apply"])

    clear_fubuki_trees.main()
    actual = np.load(tmp_path / "trees.npy")
    retained = np.isin(actual["x"], [86.5, 176.5, 173])
    np.testing.assert_array_equal(actual["z"][retained], [97, 103, 100])
    np.testing.assert_array_equal(actual["x"], [86.5, 176.5, 173])
    np.testing.assert_array_equal(written[0], actual)
    np.testing.assert_array_equal(np.load(tmp_path / "trees_before_fubuki.npy"), trees)


def test_wall_cut_preserves_fill_behind_the_crest_and_starts_gradually():
    from scripts import track_06_terrain as build

    sec, near = _local_terrain(10476, 10510, initial=98.0)
    near["Z"][19:21] = 104.0
    bounds = {"s5": (0, 12000)}
    wall = replace(_right_walls()[-1], s_from=10476)
    walk = features.Feature(10476, 10507, "R", "sidewalk", 0, "", width_m=1.5)
    build.raise_fubuki_slope(sec, near, [wall, walk], bounds)
    before = near["Z"].copy()
    build.clear_fubuki_wall_face(sec, near, [wall, walk], bounds)
    assert near["Z"][17, 27] == before[17, 27]  # S5 10503 m、天端より奥の右 7 m
    cut = before[19] - near["Z"][19]
    assert cut[4] == 0
    assert cut[5] == pytest.approx(cut[6] / 2.0)


def test_right_wall_cap_tapers_at_bridge():
    from yabitsu import fubuki
    from yabitsu.kn5 import _triangles
    from yabitsu.meshgen import face_normals

    sec = _straight(10510)
    wall = _right_walls()[-1]
    walk = features.Feature(10300, 10507, "R", "sidewalk", 0, "", width_m=1.5)
    ground = lambda x, y: np.interp(x, [10503, 10504, 10505, 10506, 10506.5, 10507],
                                    [102.7, 101.7, 99, 97, 96.1, 96])
    (_, v, faces, uv, _), = fubuki.wall_cap_meshes(sec, [wall, walk], 0, ground)
    front, _, _ = features.retaining_wall_mesh(sec, wall, features.sidewalk_offsets(sec, [walk])[1])
    for top in front[1::2]:
        assert np.any(np.all(v == top, axis=1))
    for x in (10504, 10505, 10506, 10507):
        row = v[v[:, 0] == x]
        bottom = row[row[:, 2].argmin()]
        assert bottom[2] <= ground(*bottom[:2]), (x, bottom[2] - ground(*bottom[:2]))
        if x > 10504:
            foot = front[(front[:, 0] == x) & (front[:, 2] < 100)][0]
            assert bottom[1] == pytest.approx(foot[1] - .04)
    stations = np.unique(v[v[:, 0] >= 10504, 0])
    depths = [v[v[:, 0] == x, 2].min() for x in stations]
    samples = np.linspace(10504, 10507, 61)
    assert np.max(np.interp(samples, stations, depths) - ground(samples, None)) <= 0
    end_faces = faces[np.all(v[faces, 0] == 10507, axis=1)]
    end = v[np.unique(end_faces)]
    for z in np.unique(end[:, 2]):
        assert np.ptp(end[end[:, 2] == z, 1]) == pytest.approx(.04, abs=.01)
    assert np.all(face_normals(v, _triangles(end_faces))[:, 0] > 0)
    assert np.ptp(uv[end_faces, 1]) > 0
    rear = _triangles(faces[np.any(v[faces, 2] < 99, axis=1) & ~np.all(v[faces, 0] == 10507, axis=1)])
    assert len(rear) and np.all(face_normals(v, rear)[:, 1] < 0)


def test_wall_cut_follows_height_and_lean_in_feature_table(monkeypatch):
    from scripts import track_06_terrain as build

    sec, near = _local_terrain(10480, 10510, initial=108.0)
    walk = features.Feature(10480, 10507, "R", "sidewalk", 0, "", width_m=1.5)
    wall = replace(_right_walls()[-1], s_from=10480, height_m=4)
    bounds = {"s5": (0, 12000)}
    build.clear_fubuki_wall_face(sec, near, [wall, walk], bounds)
    right_seven = near["Z"][17, 10]
    near["Z"][:] = 108.0
    monkeypatch.setattr(features, "WALL_LEAN", 0.5)
    build.clear_fubuki_wall_face(sec, near, [wall, walk], bounds)
    assert near["Z"][17, 10] < right_seven


def test_route70_fubuki_lamp_keeps_approved_position_and_marks_road_edge():
    import json

    from export.compare.route70.work import gen_route70_landmarks as gen

    spec = json.loads((config.DATA_MANUAL / "s45_route70.json").read_text())
    light = next(item for item in spec["landmarks"] if item["src"] == "S5-0981-01")
    assert light["at_road_edge"] is True
    assert light["ll"] == [35.5105896, 139.2312506]
    assert light["dz"] == -0.11
    item = {"src": "S5-0981-01", "asset": "r70_lights", "name": "arm_arc", "ll": [0, 0], "bearing": 0, "z": 100}
    neighbor = {"src": "S5-0981-02", "ll": [1, 2]}
    gen.place_fubuki_entry_light([item, neighbor])
    assert item == light
    assert neighbor == {"src": "S5-0981-02", "ll": [1, 2]}


def test_kamatagawa_fallback_keeps_old_width_and_v2_shares_one_width(tmp_path, monkeypatch):
    from yabitsu import bridge_parts, fubuki, landmarks

    monkeypatch.setattr(landmarks, "DIR", tmp_path)
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    feats = _s5_features()
    bridges = bridge_parts.load(config.DATA_MANUAL / "s5_bridges.json")
    target = next(b for b in bridges if b["osm_id"] == 131561912)
    width = float(target["right_extension_m"])

    # v2 のメッシュも画像もない場合、両方旧幅に戻る。
    prepare = fubuki.kamatagawa_parts
    fallback_walks, fallback_bridges = prepare(feats, bridges, 0)
    assert not any(f.type == "sidewalk" and f.s_from == 10507 for f in fallback_walks)
    fallback = next(b for b in fallback_bridges if b["osm_id"] == 131561912)
    assert fallback.get("right_extension_m", 0) == 0

    delivered = tmp_path / "r70_fubuki"
    _landmark_fixture(delivered, "red_arch_bridge_v2")
    widths = [width, width + 0.35]
    for expected in widths:
        changed = [dict(b, right_extension_m=expected) if b == target else b for b in bridges]
        ready_walks, ready_bridges = prepare(feats, changed, 0)
        walk = next(f for f in ready_walks if f.type == "sidewalk" and f.s_from == 10507)
        assert walk.width_m == expected
        bridge = next(b for b in ready_bridges if b["osm_id"] == 131561912)
        assert bridge["right_extension_m"] == walk.width_m
        assert [b for b in ready_bridges if b["osm_id"] != 131561912] == [b for b in bridges if b != target]
    (delivered / "red_arch_bridge_v2.npz").unlink()
    missing_mesh_walks, missing_mesh_bridges = prepare(feats, bridges, 0)
    assert (missing_mesh_walks, missing_mesh_bridges) == (fallback_walks, fallback_bridges)


def test_fubuki_stone_retexture_reaches_each_wall_and_cap(tmp_path, monkeypatch):
    from scripts import dress_mountain as dress
    from yabitsu import fubuki
    from yabitsu.manifest import Manifest

    sec = _straight(12000)
    feats = [f for f in _s5_features() if f.type == "retaining_wall"]
    bounds = {s: (0, 0) for s in dress.dm.SECTIONS}
    bounds["s5"] = (0, 11999)
    # s4 の表は同じ通し距離に重ねない。
    bounds["s4"] = (-10000, -1)
    manifest = Manifest(tmp_path)
    names = {}
    bridge_wall = set()
    for k, f in enumerate(feats):
        name = f"1WALL_s1_retaining_{k:03d}"
        manifest.add_mesh("s5", name, *features.retaining_wall_mesh(sec, f), "concrete")
        if 10300 <= f.s_from < 10684:
            names[name] = "mt_wall_fubuki_kenchi"
        if (f.s_from, f.s_to, f.side) == (8961, 8980, "L"):
            bridge_wall.add(name)
    for name, v, faces, uv, mat in fubuki.wall_cap_meshes(sec, feats, 0, lambda x, y: x * 0 + 103):
        manifest.add_mesh("s5", name, v, faces, uv, mat)
        names[name] = "mt_wall_fubuki_kenchi"
    manifest.save()
    monkeypatch.setattr(dress, "MANIFEST", tmp_path)
    monkeypatch.setattr(dress, "S3_DRESS", tmp_path / "s3")
    assign = dress.retex_assign(sec, {"x": sec.x, "y": sec.y, "s": sec.s}, bounds, feats, dress.table_rows())
    assert {name: assign.get(name) for name in names} == names
    assert len(bridge_wall) == 1
    assert {name for name, key in assign.items() if key == "mt_wall_fubuki_kenchi"} == set(names)
    assert {assign[n] for n in bridge_wall} == {"mt_wall_valley_kenchi"}
    assert dress.RETEX["mt_wall_fubuki_kenchi"][0] == "kenchi_moss"


def test_fubuki_grass_splits_only_the_right_slope_between_wall_and_fence(tmp_path, monkeypatch):
    import shapely
    from scripts import dress_mountain as dress
    from yabitsu.manifest import Manifest
    from yabitsu.meshgen import face_normals

    sec = _straight(12000)
    # 広い 2 三角形が両端・門・道・フェンスをすべてまたぐ。
    v = np.array([[10325.0, -20], [10520, -20], [10520, 5], [10325, 5]])
    z = 100 + v[:, 1] / 2
    name = "1GRASS_s1_terrain_286"
    untouched = "1GRASS_s1_terrain_287"
    main = Manifest(tmp_path / "track")
    main.add_mesh("s5", name, np.c_[v, z], [[0, 1, 2], [0, 2, 3]], v / 8, "ground")
    main.add_mesh("s5", untouched, np.c_[v + [0, 30], z], [[0, 1, 2], [0, 2, 3]], v / 8, "ground")
    main.save()
    other_file = main.root / main.meshes[untouched]["file"]
    original = other_file.read_bytes()
    monkeypatch.setattr(dress, "MANIFEST", main.root)
    feats = _s5_features()
    bounds = {s: (-20000, -1) for s in dress.dm.SECTIONS}
    bounds["s5"] = (0, 11999)
    out = Manifest(tmp_path / "dress")
    assign = {name: "mt_terrain", untouched: "mt_terrain"}
    cl = {"x": sec.x, "y": sec.y, "s": sec.s}
    hidden = dress.split_grass_terrain(sec, cl, bounds, dress.table_rows(), assign, out, tmp_path, feats)
    assert hidden == [name]
    assert assign[untouched] == "mt_terrain"
    assert all("terrain_287" not in mesh for mesh in out.meshes)
    assert other_file.read_bytes() == original
    footprints = {}
    for mesh, e in out.meshes.items():
        with np.load(out.root / e["file"]) as data:
            verts, faces = data["verts"], data["faces"]
            assert np.all(face_normals(verts, faces)[:, 2] > 0)
            np.testing.assert_allclose(verts[:, 2], 100 + verts[:, 1] / 2)
            np.testing.assert_allclose(data["uvs"], verts[:, :2] / 8, atol=1e-4)
            footprints[assign[mesh]] = shapely.union_all(shapely.polygons(verts[faces, :2]))
    grass, rest = footprints["mt_grass"], footprints["mt_terrain"]
    for x, y in [(10338.1, -9), (10404.9, -13), (10436.1, -7), (10445.1, -12), (10506.9, -11)]:
        assert grass.covers(shapely.Point(x, y))
    for x, y in [(10337.9, -9), (10405.1, -9), (10435.9, -10), (10507.1, -9),
                 (10400, 6), (10400, -5), (10400, -14.1), (10439, -9), (10504, -12.1)]:
        assert not grass.covers(shapely.Point(x, y))
    assert grass.intersection(rest).area == pytest.approx(0)
    assert grass.union(rest).symmetric_difference(shapely.Polygon(v)).area == pytest.approx(0, abs=1e-7)


def test_fubuki_gate_uses_two_posts_and_keeps_the_driveway_clear(tmp_path, monkeypatch):
    from yabitsu import fubuki, landmarks

    monkeypatch.setattr(landmarks, "DIR", tmp_path)
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    sec = _straight(12000)
    feats = _s5_features()
    ground = lambda x, y: 101 + (x - 10400) * .01 - (y + 7.1) * .30
    folder = tmp_path / "r70_fubuki"
    assert fubuki.gate_meshes(sec, 0, feats, ground) == []
    row = fubuki.load()["gate"]
    _landmark_fixture(folder, row["post"], length=.3)
    (folder / f'{row["post"]}.png').unlink()
    assert fubuki.gate_meshes(sec, 0, feats, ground) == []
    _landmark_fixture(folder, row["post"], length=.3)
    np.savez(folder / f'{row["post"]}.npz', v=[[0, 0, 0], [.3, 0, 0], [0, -.45, 0], [0, 0, 1]],
             f=[[0, 1, 3], [0, 2, 3]], uv=[[0, 0], [1, 0], [0, 0], [0, 1]])
    posts = fubuki.gate_meshes(sec, 0, feats, ground)
    assert len(posts) == 2
    for _, _, v, _, _ in posts:
        foot = v[:3]
        assert np.max(foot[:, 2] - ground(foot[:, 0], foot[:, 1])) <= 0.0
        assert np.ptp(foot[:, 2]) == 0.0
    _landmark_fixture(folder, row["wall"])
    meshes = fubuki.gate_meshes(sec, 0, feats, ground)
    walls = [v for _, part, v, _, _ in meshes if part == row["wall"]]
    assert len(walls) == 2
    assert all(v[:, 1].max() < -(3 + features.GUTTER_WIDTH + 1.5) for _, _, v, _, _ in meshes)
    assert all(10405 <= v[:, 0].min() < v[:, 0].max() <= 10436 for _, _, v, _, _ in meshes)
    assert walls[0][:, 0].max() <= 10424
    assert walls[1][:, 0].min() >= 10428
    v = walls[0]
    assert np.all(v[:, 2] >= ground(v[:, 0], v[:, 1]) - .025)
    assert np.all(v[:, 2] <= ground(v[:, 0], v[:, 1]) + 1.01)


@pytest.fixture
def fubuki_gate_site():
    import shapely

    # 1巡目の中心線、side_stub_595991910_0、s5_10413_white_house の床を門柱を原点とする XY に移す。
    sec, _ = _local_terrain(10404, 10439)
    stations = [10403.9631, 10407.9631, 10417.9631, 10427.9630, 10434.9630, 10438.9630, 10439.9630]
    sec.x[:] = np.interp(sec.s, stations, [-1.3949, -1.0109, -.0513, .8675, 1.4483, 1.7653, 1.844])
    sec.y[:] = np.interp(sec.s, stations, [-14.470, -10.4883, -.5345, 9.4230, 16.3988, 20.3862, 21.383])
    sec.heading[:] = np.interp(sec.s, stations, [1.4746, 1.4746, 1.4751, 1.4845, 1.4915, 1.4924, 1.4924])
    sec.hw_right[:] = 3.5
    sec.z[:] = np.interp(sec.s, stations, [297.8029, 297.7211, 297.5380, 297.3940, 297.3190, 297.2856, 297.278])
    branch = shapely.Polygon([(4.5975, 5.0863), (4.8656, 8.0753), (14.8081, 7.4414), (14.6172, 4.4475)])
    house = shapely.Polygon([(10.5488, 20.2042), (17.2955, 19.7748), (16.7524, 11.0578), (10.0058, 11.4873)])

    def ground(x, y):
        edge = np.interp(y, sec.y, sec.edge_z(-1))
        return edge + np.interp(y, [-10, 0, 9.5, 13, 13.75, 14.5, 17],
                                [.02, .02, -.30, -.32, -.41, -.33, -.35]) - .01 * (x - 7)

    walk = features.Feature(10404, 10439, "R", "sidewalk", 0, "", width_m=1.5)
    return sec, walk, ground, branch, house


@pytest.mark.parametrize("kind", ["post", "wall"])
def test_fubuki_gate_heights_follow_the_walk_and_reach_low_ground(fubuki_gate_site, kind):
    from yabitsu import fubuki, landmarks

    sec, walk, ground, _, _ = fubuki_gate_site
    top = []
    for tag, part, v, _, _ in fubuki.gate_meshes(sec, 0, [walk], ground):
        if not tag.startswith(f"gate_{kind}"):
            continue
        source, _, _ = landmarks.load(fubuki.ASSET, part)
        copies = {"gate_wall_0": 5, "gate_wall_1": 4}.get(tag, 1)
        corners = v.reshape(copies, -1, 3)[:, :len(source)]
        z = source[:, 2]
        base, crown = corners[:, z == z.min()].reshape(-1, 3), corners[:, z == z.max()].reshape(-1, 3)
        if kind == "post":
            assert np.max(base[:, 2] - ground(*base[:, :2].T)) <= 0
            top.append(crown[:, 2].max())
        else:
            # この門前では、右に振った各頂点を中心線に戻すと y は最大約 0.7 m 先になる。
            s = np.interp(crown[:, 1] + .6, sec.y, sec.s)
            edge = np.interp(s, sec.s, sec.edge_z(-1))
            height = crown[:, 2] - edge
            expected = .63 if tag == "gate_wall_1" else .48
            np.testing.assert_allclose(height, expected, atol=.03, err_msg=tag)
            if tag == "gate_wall_1":
                np.testing.assert_allclose(base[:, 2] - ground(*base[:, :2].T), -.02, atol=1e-7)
    if kind == "post":
        assert len(top) == 2
        assert abs(top[0] - top[1]) <= .03
        high_walk = np.interp(10418.5, sec.s, sec.edge_z(-1)) + .15
        assert top[0] - high_walk == pytest.approx(2.50, abs=.03)


def test_fubuki_far_wall_reaches_ground_between_its_corners(fubuki_gate_site):
    from yabitsu import fubuki
    from yabitsu.meshgen import face_normals

    sec, walk, ground, _, _ = fubuki_gate_site
    _, _, v, faces, _ = next(mesh for mesh in fubuki.gate_meshes(sec, 0, [walk], ground) if mesh[0] == "gate_wall_1")
    low = v[:, 2] - ground(*v[:, :2].T) < .10
    floor = faces[low[faces].all(axis=1) & (face_normals(v, faces)[:, 2] < -.5)]
    assert len(floor) > 0
    edges = np.concatenate([floor[:, [0, 1]], floor[:, [1, 2]], floor[:, [2, 0]]])
    t = np.linspace(0, 1, 41)[None, :, None]
    a, b = v[edges[:, 0], None], v[edges[:, 1], None]
    x, y, z = (a + t * (b - a)).transpose(2, 0, 1)
    assert np.max(z - ground(x, y)) <= 0


def test_fubuki_gate_clears_the_branch_and_house_and_faces_the_road(fubuki_gate_site):
    import shapely
    from yabitsu import fubuki, landmarks
    from yabitsu.meshgen import face_normals

    sec, walk, ground, branch, house = fubuki_gate_site
    meshes = fubuki.gate_meshes(sec, 0, [walk], ground)
    assert len(meshes) == 4
    for tag, part, v, faces, _ in meshes:
        triangles = shapely.polygons(v[faces, :2])
        plan = shapely.union_all(triangles[shapely.area(triangles) > 1e-10])
        for obstacle, min_distance in ((branch, .45), (house, 1.55)):
            assert plan.intersection(obstacle).area == 0
            assert plan.distance(obstacle) >= min_distance, tag
        if tag.startswith("gate_post"):
            original, _, _ = landmarks.load(fubuki.ASSET, part)
            plaque = np.all(original[faces, 1] == original[:, 1].min(), axis=1)
            assert plaque.any()
            s = [10418.5, 10428.15][int(tag[-1])]
            road = np.array([np.interp(s, sec.s, sec.x), np.interp(s, sec.s, sec.y)])
            toward_road = road - v[faces[plaque], :2].mean(axis=(0, 1))
            normals = face_normals(v, faces[plaque])[:, :2]
            assert np.all(normals @ (toward_road / np.linalg.norm(toward_road)) > .99)


def test_fubuki_white_wall_matches_half_height_and_has_its_own_tint(tmp_path):
    from PIL import Image
    from scripts import dress_mountain as dress
    from yabitsu import fubuki, landmarks

    sec = _straight(12000)
    feats = _s5_features()
    wall = next(f for f in feats if f.side == "L" and f.s_from == 10610)
    assert wall.height_m == pytest.approx(1.8)
    _, name, v, _, _ = fubuki.wall_top_meshes(sec, 0, feats)[0]
    assert v[:, 2].min() == pytest.approx(sec.edge_z(1)[10610] + wall.height_m)
    assert np.ptp(v[:, 2]) / wall.height_m == pytest.approx(.5, abs=.06)
    src = landmarks.texture(fubuki.ASSET, name)
    dest = tmp_path / "wall.png"
    dress.copy_fubuki_texture(name, src, dest)
    before = np.array(Image.open(src).convert("RGB"))
    after = np.array(Image.open(dest).convert("RGB"))
    assert 0.48 < after.mean() / before.mean() < 0.65
    assert np.std(after) > 5


def test_fubuki_roadside_shrubs_are_beyond_the_walk_and_repeatable():
    from scripts import plant_fubuki_shrubs as plant

    sec = _straight(12000)
    feats = _s5_features()
    dtype = [("x", "f8"), ("y", "f8"), ("z", "f8"), ("pattern", "U16"),
             ("angle_deg", "f8"), ("scale", "f8")]
    trees = np.array([(10630, -8, 100, "sugi_a", 90, 1)], dtype=dtype).view(np.recarray)
    ground = lambda x, y: np.full_like(x, 99.0)
    added = plant.roadside_shrubs(sec, trees, feats, 0, ground)
    assert 100 < len(added) < 250
    assert (added.x.min() > 10607) and (added.x.max() < 10684)
    assert np.all(added.y < -(3 + features.GUTTER_WIDTH + 2.7))
    assert np.all(added.z == 99)
    assert set(added.pattern) == {"shrub_a", "shrub_b"}
    assert np.min(np.hypot(added.x - 10630, added.y + 8)) >= .65
    combined = np.concatenate((trees, added)).view(np.recarray)
    assert len(plant.roadside_shrubs(sec, combined, feats, 0, ground)) == 0
    np.testing.assert_array_equal(plant.roadside_shrubs(sec, trees, feats, 0, ground), added)


def test_fubuki_clear_warns_only_when_it_removes_planted_shrubs(tmp_path, monkeypatch, capsys):
    import sys
    from scripts import clear_fubuki_trees as clear, forest_build, plant_fubuki_shrubs as plant
    from yabitsu.manifest import Manifest

    sec = _straight(12000)
    sec.save_json(tmp_path / "track_section.json")
    monkeypatch.setattr(config, "DATA_PROCESSED", tmp_path)
    monkeypatch.setattr(clear.track, "load_bounds", lambda: {"s5": (0, 11999)})
    monkeypatch.setattr(clear.track, "load_section_rows", lambda *_: [])
    manifest = Manifest(tmp_path / "track")
    manifest.save()
    monkeypatch.setattr(forest_build, "MANIFEST", manifest.root)
    xs, ys = np.array([10300., 10700.]), np.array([-20., 1.])
    for detail in ("near", "mid"):
        np.savez(tmp_path / f"track_terrain_{detail}.npz", xs=xs, ys=ys,
                 Z=np.full((2, 2), 99.0), D=np.zeros((2, 2)))
    forest = tmp_path / "forest with spaces"
    forest.mkdir()
    dtype = [("x", "f8"), ("y", "f8"), ("z", "f8"), ("pattern", "U16"),
             ("angle_deg", "f8"), ("scale", "f8")]
    trees = np.array([(10390, -9, 99, "shrub_a", 90, 1), (10630, -9, 99, "sugi_a", 90, 1)], dtype=dtype)
    current = forest / "trees.npy"
    np.save(current, trees)
    monkeypatch.setattr(sys, "argv", ["clear_fubuki_trees", str(forest), "--apply"])
    clear.main()
    assert "plant_fubuki_shrubs" not in capsys.readouterr().out
    np.testing.assert_array_equal(np.load(current), trees[1:])

    monkeypatch.setattr(sys, "argv", ["plant_fubuki_shrubs", str(forest), "--apply"])
    plant.main()
    capsys.readouterr()
    added = len(np.load(current)) - 1
    assert added > 0
    forest_list = forest / "extension" / "trees" / "forest_10_-1.txt"
    assert "shrub_" in forest_list.read_text()

    monkeypatch.setattr(sys, "argv", ["clear_fubuki_trees", str(forest), "--apply"])
    np.save(forest / "trees_before_valleys.npy", np.load(current))
    clear.main()
    np.testing.assert_array_equal(np.load(current), trees[1:])
    assert "shrub_" not in forest_list.read_text()
    output = capsys.readouterr().out
    assert f"低木 {added} 本" in output
    assert "谷の補正も消えます" in output
    assert "align_valley_trees" in output
    assert f".venv/bin/python -m scripts.plant_fubuki_shrubs '{forest}' --apply" in output

    clear.main()
    assert ".venv/bin/python -m scripts.plant_fubuki_shrubs" not in capsys.readouterr().out
