"""保存済みの横断面と欄干部品で、全体を組み直さず橋の見た目と衝突壁を検査する。"""
import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image
from shapely.geometry import Polygon

from scripts import dress_mountain
from yabitsu import assembly, buildings, config, features, house_kit, kn5, landmarks, small_bridges, terrain, track
from yabitsu.crosssection import GUTTER_WIDTH, Section
from yabitsu.forest_ground import TriSoup, mesh_class
from yabitsu.manifest import Manifest

SECTION = config.DATA_PROCESSED / "track_section.json"
pytestmark = pytest.mark.skipif(not SECTION.exists(), reason="保存済みの道路断面が無い")
CASES = [
    (2, "s4", 1721, 1728, "concrete_two_2m", "square_09", (181,)),
    (16, "s4", 6351, 6358, "wood_clad_2m", None, (281, 394)),
    (25, "s5", 8953, 8960, "steel_white_lattice_2m", None, (366,)),
]


@pytest.fixture(scope="module")
def course():
    sec = Section.load_json(SECTION)
    bounds = track.load_bounds()
    return sec, bounds, dress_mountain.track_features(sec, bounds)


def mesh(m, name):
    entry = m.meshes[name]
    with np.load(m.root / entry["file"]) as data:
        return data["verts"], data["faces"], data["uvs"]


def signed_ends(sec, i, j, vertices):
    xy = vertices[:, :2]
    return np.column_stack([
        (xy - [sec.x[k], sec.y[k]]) @ [np.cos(sec.heading[k]), np.sin(sec.heading[k])]
        for k in (i, j)
    ])


@pytest.fixture(scope="module")
def bridge_spans(course):
    sec, bounds, _ = course
    return small_bridges.spans(sec, bounds, small_bridges.load())


@pytest.mark.parametrize("number, section, start, stop, part, post, rails", CASES)
def test_codex_rails_and_feet(course, bridge_spans, tmp_path, monkeypatch,
                             number, section, start, stop, part, post, rails):
    sec, bounds, _ = course
    source = Manifest(tmp_path / "track")
    source.save()
    monkeypatch.setattr(dress_mountain, "MANIFEST", source.root)
    m = Manifest(tmp_path / "dress")
    tex = m.root / "textures"
    tex.mkdir()
    dress_mountain.add_r70_structures(sec, bounds, m, tex)
    i, j = bridge_spans[number]
    for side, sd in (("left", 1), ("right", -1)):
        name = f"mt_r70_bridge_{number:02d}_{side}"
        assert name in m.meshes
        assert part in m.meshes[name]["material"]
        v, faces, _ = mesh(m, name)
        part_v, part_faces, _ = landmarks.load("r70_bridges", part)
        assert len(faces) % len(part_faces) == 0
        count = len(faces) // len(part_faces)
        outer_side = -1 if number == 16 else 1
        assert count == (4 if sd == outer_side else 3)
        feet = v[np.tile(part_v[:, 2] == 0, len(v) // len(part_v))]
        for p in (0, 1):
            if post:
                post_v, _, _ = mesh(m, f"{name}_post{p}")
                feet = np.vstack([feet, post_v[landmarks.load("r70_bridges", post)[0][:, 2] == 0]])
            else:
                assert f"{name}_post{p}" not in m.meshes
        ends = signed_ends(sec, i, j, v)
        assert ends[:, 0].min() == pytest.approx(0, abs=1e-6)
        assert ends[:, 1].max() == pytest.approx(0, abs=1e-6)

        sill_v, sill_f, _ = mesh(m, f"{name}_sill")
        sill_z = TriSoup(sill_v, kn5._triangles(sill_f))(*feet[:, :2].T).filled(np.nan)
        assert np.isfinite(sill_z).all()
        np.testing.assert_allclose(feet[:, 2], sill_z, atol=0.05, rtol=0)

        # 側溝の外縁。既存の道路メッシュと同じ直線補間で縁の高さを読む。
        h = sec.heading[i:j + 1]
        hw = (sec.hw_left if sd > 0 else sec.hw_right)[i:j + 1]
        normal = sd * np.c_[-np.sin(h), np.cos(h)]
        centers = np.c_[sec.x[i:j + 1], sec.y[i:j + 1]]
        run = centers + normal * (hw + 0.37)[:, None]
        panels = v.reshape(count, len(part_v), 3)
        # 始終端断面にある左右の頂点の平均から、各枚が占める弧長を測る。
        panel_s = []
        segment = np.diff(run, axis=0)
        length = np.linalg.norm(np.c_[segment, np.diff(sec.edge_z(sd)[i:j + 1])], axis=1)
        arc = np.r_[0.0, np.cumsum(length)]
        for panel in panels:
            ends_xy = np.array([panel[part_v[:, 0] == x, :2].mean(0) for x in (-1, 1)])
            delta = ends_xy[:, None, :] - run[:-1]
            t = np.clip((delta * segment).sum(2) / (segment ** 2).sum(1), 0, 1)
            k = np.argmin(np.linalg.norm(delta - t[..., None] * segment, axis=2), axis=1)
            panel_s.append(abs(np.diff(arc[k] + t[np.arange(2), k] * length[k])[0]))
        np.testing.assert_allclose(panel_s, arc[-1] / count, atol=0.025, rtol=0)
        outer = np.c_[sec.x[i:j + 1] - sd * (hw + GUTTER_WIDTH) * np.sin(h),
                      sec.y[i:j + 1] + sd * (hw + GUTTER_WIDTH) * np.cos(h)]
        vec = np.diff(outer, axis=0)
        t = np.clip(((feet[:, None, :2] - outer[:-1]) * vec).sum(2) / (vec ** 2).sum(1), 0, 1)
        dist = np.linalg.norm(feet[:, None, :2] - outer[:-1] - t[..., None] * vec, axis=2)
        k = np.argmin(dist, axis=1)
        ze = sec.edge_z(sd)[i:j + 1]
        expected = ze[k] + t[np.arange(len(k)), k] * np.diff(ze)[k]
        np.testing.assert_allclose(feet[:, 2], expected, atol=0.05, rtol=0)
        skirt = centers + normal * (hw + GUTTER_WIDTH)[:, None]
        outer = sill_v.reshape(-1, 4, 3)[:, -1, :2]
        np.testing.assert_allclose(((outer - skirt) * normal).sum(1), 0.03, atol=1e-7)


def triangles_in(mesh_arrays):
    v, f, uv = mesh_arrays
    return np.column_stack((v, uv))[kn5._triangles(f)]


def assert_rail_cut(sec, i, j, old, new):
    old_triangles = triangles_in(old)
    ends = signed_ends(sec, i, j, old_triangles.reshape(-1, 5)).reshape(-1, 3, 2)
    outside = (ends[:, :, 0] < -1e-6).all(1) | (ends[:, :, 1] > 1e-6).all(1)
    new_triangles = triangles_in(new)
    remaining = {t.tobytes() for t in new_triangles}
    assert all(t.tobytes() in remaining for t in old_triangles[outside])

    ends = signed_ends(sec, i, j, new_triangles.reshape(-1, 5)).reshape(-1, 3, 2)
    assert ((ends[:, :, 0] <= 1e-6).all(1) | (ends[:, :, 1] >= -1e-6).all(1)).all()


@pytest.mark.parametrize("number, section, start, stop, part, post, rails", CASES)
def test_body_walls_and_dressed_w_beam_are_cut_only_at_bridge(
        course, bridge_spans, tmp_path, monkeypatch, number, section, start, stop, part, post, rails):
    sec, bounds, feats = course
    # Keep feature numbers without writing unrelated meshes to disk.
    class LocalManifest(Manifest):
        def add_mesh(self, group, name, *args, **kwargs):
            if name.startswith(("GUARDRAIL_s1_", "1WALL_s1_rail_", "1WALL_s1_retaining_", "mt_rail_")):
                index = int(name.split("_")[2 if name.startswith("mt_rail_") else -1])
                if index not in {181, 183, 278, 281, 366, 367, 368, 394}:
                    return
                super().add_mesh(group, name, *args, **kwargs)
            elif name.startswith("1WALL_track_small"):
                super().add_mesh(group, name, *args, **kwargs)

    monkeypatch.setattr(dress_mountain, "gaps", lambda sec: [])
    monkeypatch.setattr(dress_mountain.dm, "leaf_meshes", lambda *args, **kwargs: [])
    i, j = bridge_spans[number]
    spans = {number: (i, j)}
    for kind in ("base", "dress"):
        old = LocalManifest(tmp_path / f"old_{kind}")
        new = LocalManifest(tmp_path / f"new_{kind}")
        if kind == "base":
            assembly.add_features(old, sec, feats)
            assembly.add_features(new, sec, feats, bridge_spans=spans)
            for side, sd in (("left", 1), ("right", -1)):
                name, = [name for name in new.meshes if name.startswith("1WALL_track_small")
                         and name.endswith(f"_{number:02d}_{side}")]
                assert new.meshes[name]["renderable"] is False
                assert mesh_class(name, new.meshes[name]["material"]) == "wall"
                v, faces, _ = mesh(new, name)
                assert np.isfinite(v).all()
                a = i - 1 if number == 2 and sd > 0 else i
                rows = np.arange(a, j + 1)
                assert len(faces) == j - a
                # どの面も隣り合う断面を結び、端の横向きの面がない。
                np.testing.assert_array_equal(np.ptp(faces // 2, axis=1), 1)
                normal = sd * np.c_[-np.sin(sec.heading[rows]), np.cos(sec.heading[rows])]
                xy = np.c_[sec.x[rows], sec.y[rows]]
                hw = (sec.hw_left if sd > 0 else sec.hw_right)[rows]
                offset = ((v.reshape(-1, 2, 3)[:, :, :2] - xy[:, None]) * normal[:, None]).sum(2) - hw[:, None]
                panel = landmarks.load("r70_bridges", part)
                face_offset = 0.37 + panel[0][:, 1].min()
                np.testing.assert_allclose(offset[2:-2], face_offset, atol=1e-8)
                w_beam = (number in (2, 25) and sd < 0) or number == 16
                end_offset = 0.70 if w_beam else 0.60
                taper = 0.12 if number == 25 and sd < 0 else 0.
                np.testing.assert_allclose(offset[[0, -1], 0], [end_offset, end_offset - taper], atol=1e-8)
                if number == 2 and sd > 0:
                    wall = feats[183]
                    overlap = (wall.height_m + 0.3) / 1.5
                    np.testing.assert_allclose(offset[0, 0] + overlap * (offset[0, 1] - offset[0, 0]),
                                               0.60 + 0.30 * wall.height_m, atol=1e-8)
                z = sec.edge_z(sd)[i:j + 1]
                assert v[:, 2].min() <= z.min() - 0.29
                assert v[:, 2].max() >= z.max() + 1.19
        else:
            rows = dress_mountain.table_rows()
            hide = dress_mountain.build_parts(sec, bounds, feats, rows, old)
            assert dress_mountain.build_parts(sec, bounds, feats, rows, new, bridge_spans=spans) == hide

        for name in old.meshes:
            index = int(name.split("_")[2 if name.startswith("mt_rail_") else -1])
            if number == 25 and kind == "dress" and name in ("mt_rail_366_0_r70end_1", "mt_rail_368_0_r70end_2"):
                assert name not in new.meshes
                continue
            if number == 25 and kind == "base" and index == 368:
                before, after = triangles_in(mesh(old, name)), triangles_in(mesh(new, name))
                assert len(after) == len(before) + 2
                # V is the centreline station divided by 4 m in wall_ribbon.
                # The join extends one row and adjusts the next five metres.
                taper_end = (sec.s[j] + 5.) / 4.
                unchanged = before[before[..., 4].min(axis=1) >= taper_end]
                assert len(unchanged)
                np.testing.assert_allclose(after[after[..., 4].min(axis=1) >= taper_end],
                                           unchanged, atol=1e-8, rtol=0)
                continue
            if number == 25 and name == "mt_rail_368_0_r70span_0":
                v0, f0, uv0 = mesh(old, name)
                v1, f1, uv1 = mesh(new, name)
                np.testing.assert_array_equal(f0, f1)
                np.testing.assert_array_equal(uv0, uv1)
                moved = v0[np.linalg.norm(v1 - v0, axis=1) > 1e-8]
                assert len(moved)
                frame = terrain.road_frame(sec, *moved[:, :2].T)
                distance = sec.s[frame["idx"]] + frame["along"] - sec.s[j]
                assert np.all((distance >= -1e-5) & (distance <= 5. + 1e-5))
                continue
            affected = index in rails
            if affected:
                if name in new.meshes:
                    assert_rail_cut(sec, i, j, mesh(old, name), mesh(new, name))
                else:
                    ends = signed_ends(sec, i, j, mesh(old, name)[0])
                    assert np.all(ends[:, 0] >= -1e-6) and np.all(ends[:, 1] <= 1e-6)
            else:
                assert old.meshes[name] == new.meshes[name]
                for a, b in zip(mesh(old, name), mesh(new, name)):
                    np.testing.assert_array_equal(a, b)


def test_s5_bridge_stays_before_width_jump(course, bridge_spans):
    sec, bounds, _ = course
    i, j = bridge_spans[25]
    assert sec.s[[i, j]] - bounds["s5"][0] == pytest.approx([8952.968, 8959.968], abs=0.001)
    for hw in (sec.hw_left, sec.hw_right):
        np.testing.assert_array_equal(hw[i:j + 1], 2.75)
        assert hw[j + 1] > 2.75


@pytest.fixture(scope="module")
def dressed_r70_bridges(course, tmp_path_factory):
    if not (dress_mountain.MANIFEST / "manifest.json").is_file():
        pytest.skip("保存済みの本体マニフェストが無い")
    sec, bounds, _ = course
    m = Manifest(tmp_path_factory.mktemp("r70-bridges"))
    tex = m.root / "textures"
    tex.mkdir()
    dress_mountain.add_r70_structures(sec, bounds, m, tex)
    return m


@pytest.mark.parametrize("number,part,mesh_suffixes,rgb_ceiling", [
    (2, "concrete_two_2m", ("left", "right"), (50, 50, 40)),
    (2, "square_09", ("left_post0", "left_post1", "right_post0", "right_post1"), (50, 50, 40)),
    (16, "wood_clad_2m", ("left", "right"), (40, 30, 28)),
])
def test_bridge_tint_is_private_and_darkens_the_rendered_surfaces(
        dressed_r70_bridges, number, part, mesh_suffixes, rgb_ceiling):
    m = dressed_r70_bridges
    shared = f"mt_r70_bridges_{part}"
    private = f"{shared}_b{number:02d}"
    if number == 2:
        private += "_valley38"
    targets = {f"mt_r70_bridge_{number:02d}_{suffix}" for suffix in mesh_suffixes}
    assert {n for n, record in m.meshes.items() if record["material"] == private} == targets
    other_users = {n for n, record in m.meshes.items() if record["material"] == shared}
    assert other_users and not (other_users & targets)

    source = np.asarray(Image.open(landmarks.texture("r70_bridges", part)).convert("RGBA"))
    original = np.asarray(Image.open(m.root / m.materials[shared]["texture"]).convert("RGBA"))
    tinted = np.asarray(Image.open(m.root / m.materials[private]["texture"]).convert("RGBA"))
    np.testing.assert_array_equal(original, source)
    np.testing.assert_array_equal(tinted[..., 3], original[..., 3])
    _, faces, uv = landmarks.load("r70_bridges", part)
    u, v = uv[faces].mean(axis=1).T
    h, w = tinted.shape[:2]
    x = (u * w).astype(int).clip(0, w - 1)
    y = ((1 - v) * h).astype(int).clip(0, h - 1)
    rgb = np.median(tinted[y, x, :3], axis=0)
    assert np.all((rgb > 0) & (rgb <= rgb_ceiling)), rgb


def test_s5_small_bridge_rail_uses_its_own_white_lattice_finish(dressed_r70_bridges):
    m = dressed_r70_bridges
    for side in ("left", "right"):
        assert m.meshes[f"mt_r70_bridge_25_{side}"]["material"] == "mt_r70_bridges_steel_white_lattice_2m_b25_valley38"


def test_s4_1720_removes_only_the_white_house_and_keeps_its_tree_clearance(course, tmp_path):
    import json

    path = config.DATA_MANUAL / "track_buildings.json"
    entries = json.loads(path.read_text(encoding="utf-8"))
    removed_id = "vm866_484"
    original = tmp_path / path.name
    original.write_text(json.dumps([
        {**record, "gone": False} if record["id"] == removed_id else record for record in entries
    ]), encoding="utf-8")
    before = buildings.load(original)
    after = buildings.load(path)
    removed, = [b for b in before if b.id == removed_id]
    assert after == [b for b in before if b.id != removed_id]
    assert buildings.load(path, clearance_only=True) == [removed]

    sec, _, _ = course
    nearby = [b for b in before if b.footprint.distance(removed.footprint) < 45]
    remaining = [b for b in after if b.footprint.distance(removed.footprint) < 45]
    def flat_ground(x, y):
        return np.zeros_like(x)

    road = np.c_[sec.x, sec.y]
    before_meshes, _ = house_kit.house_meshes(nearby, flat_ground, road)
    after_meshes, _ = house_kit.house_meshes(remaining, flat_ground, road)
    before_parts = {part[0]: part[1:] for part in before_meshes}
    after_parts = {part[0]: part[1:] for part in after_meshes}
    gone = set(before_parts) - set(after_parts)
    assert len(gone) == 8 and all(removed_id in name for name in gone)
    assert any("vm873_477" in name for name in after_parts)
    for name, (v, f, uv, mat, render) in after_parts.items():
        old_v, old_f, old_uv, old_mat, old_render = before_parts[name]
        np.testing.assert_array_equal(v, old_v)
        np.testing.assert_array_equal(f, old_f)
        np.testing.assert_array_equal(uv, old_uv)
        assert (mat, render) == (old_mat, old_render)


def test_wall_367_is_left_after_bridge_and_uses_left_finish(course, bridge_spans, tmp_path, monkeypatch):
    sec, bounds, feats = course
    wall = feats[367]
    assert (wall.type, wall.side) == ("retaining_wall", "L")
    assert sec.s[bridge_spans[25][1]] < wall.s_from
    assert (wall.s_from - bounds["s5"][0], wall.s_to - bounds["s5"][0]) == (8961, 8980)
    assert not any(f.type == "retaining_wall" and f.side == "R" and
                   f.s_from < wall.s_to and bounds["s5"][0] + 8940 < f.s_to for f in feats)
    assert [(feats[k].s_from - bounds["s5"][0], feats[k].side) for k in (366, 368)] == [
        (8686, "R"), (8960, "R")]
    m = Manifest(tmp_path)
    for k in (367, 370):
        v, f, uv = features.retaining_wall_mesh(sec, feats[k])
        m.add_mesh("s5", f"1WALL_s1_retaining_{k}", v, f, uv, "concrete")
    m.save()
    monkeypatch.setattr(dress_mountain, "MANIFEST", m.root)
    cl = dict(x=sec.x, y=sec.y, s=sec.s)
    rows = dress_mountain.table_rows()
    row, = [r for r in rows if r["section"] == "s5" and r["item"] == "wall" and r["s_from"] == "8961"]
    assert row["side"] == "L"
    assign = dress_mountain.retex_assign(sec, cl, bounds, feats, rows)
    assert assign["1WALL_s1_retaining_367"] == "mt_wall_valley_kenchi"
    assert assign["1WALL_s1_retaining_370"] == "mt_wall_r70_block"


def test_no_tall_wall_behind_rail_and_281_is_replaced_by_right_guardrail(course, bridge_spans):
    sec, _, feats = course
    class Walls:
        def __init__(self):
            self.meshes = {}

        def add_mesh(self, group, name, v, f, uv, material, renderable=True):
            if name.startswith("1WALL_s1_retaining_") and renderable:
                self.meshes[name] = (v, f, uv)

    after = Walls()
    assembly.add_features(after, sec, feats, bridge_spans=bridge_spans)
    name = "1WALL_s1_retaining_281"
    assert feats[281].side == "R" and feats[281].type == "guardrail"
    assert name not in after.meshes
    spec = small_bridges.load()["bridges"]
    for number in bridge_spans:
        a, b = bridge_spans[number]
        for side, sd in (("L", 1), ("R", -1)):
            rail = spec[number]
            pv, _, _ = landmarks.load("r70_bridges", rail["left" if sd > 0 else "right"])
            rows = np.arange(a, b + 1)
            hw = (sec.hw_left if sd > 0 else sec.hw_right)[rows]
            normal = sd * np.c_[-np.sin(sec.heading[rows]), np.cos(sec.heading[rows])]
            xy = np.c_[sec.x[rows], sec.y[rows]]
            outer = xy + normal * (hw + 0.37 + pv[:, 1].max())[:, None]
            far = outer + normal
            # 端そのものは擁壁を残す境界。橋の内側に 1 mm 入って調べる。
            for points in (outer, far):
                points[0] += 0.001 * (points[1] - points[0])
                points[-1] += 0.001 * (points[-2] - points[-1])
            zone = Polygon(np.concatenate([outer, far[::-1]]))
            active = [k for k, f in enumerate(feats) if f.type == "retaining_wall" and f.side == side
                      and f.s_from < sec.s[b] and sec.s[a] < f.s_to]
            for k in active:
                v, f, _ = after.meshes[f"1WALL_s1_retaining_{k:03d}"]
                for tri in v[kn5._triangles(f)]:
                    nearest = np.argmin(np.linalg.norm(tri[:, None, :2] - outer, axis=2), axis=1)
                    if (tri[:, 2] > sec.edge_z(sd)[rows[nearest]] + rail["height_m"]).any():
                        assert not zone.intersects(Polygon(tri[:, :2]))


def test_collision_wall_tapers_to_guardrail_and_matches_visible_face(course, bridge_spans):
    sec, _, feats = course
    i, j = bridge_spans[16]
    # 左：394 番の W ビームと同じ端点。中央は wood_clad_2m の道側の面。
    v, f, _ = small_bridges.collision_mesh(sec, i, j, 1, [feats[394]], 0.24)
    assert len(v) == 2 * (j - i + 1)
    rows = np.arange(i, j + 1)
    normal = np.c_[-np.sin(sec.heading[rows]), np.cos(sec.heading[rows])]
    center = np.c_[sec.x[rows], sec.y[rows]]
    offset = ((v[::2, :2] - center) * normal).sum(1) - sec.hw_left[rows]
    np.testing.assert_allclose(offset[[0, -1]], 0.70, atol=1e-8)
    np.testing.assert_allclose(offset[2:-2], 0.24, atol=1e-8)
    assert 0.24 < offset[1] < 0.70 and 0.24 < offset[-2] < 0.70
    assert all(len(set(face // 2)) == 2 for face in f)


@pytest.mark.parametrize("length", [6.999975, 7.0])
def test_panel_count_at_rounding_boundary(length):
    x = np.linspace(0, length, 8)
    sec = SimpleNamespace(x=x, y=np.zeros(8), s=x, heading=np.zeros(8),
                          hw_left=np.full(8, 2.75), edge_z=lambda side: np.zeros(8))
    part = landmarks.load("r70_bridges", "concrete_two_2m")
    _, faces, _ = small_bridges.rail_mesh(sec, 0, 7, 1, part, 2.0)
    assert len(faces) == 4 * len(part[1])


def test_cut_leaves_remote_faces_between_infinite_planes():
    sec = SimpleNamespace(x=np.r_[np.arange(10), np.arange(10)],
                          y=np.r_[np.zeros(10), np.full(10, 10.0)], heading=np.zeros(20))
    v = np.array([[4, 0, 0], [5, 0, 0], [4, 0, 1],
                  [4, 10, 0], [5, 10, 0], [4, 10, 1]], float)
    mesh0 = (v, np.array([[0, 1, 2], [3, 4, 5]]), v[:, :2])
    v1, f1, uv1 = small_bridges.cut_guardrail(sec, mesh0, [(3, 6)])
    np.testing.assert_array_equal(v1[f1], v[mesh0[1][1:]])
    np.testing.assert_array_equal(uv1[f1], mesh0[2][mesh0[1][1:]])


def test_sill_face_has_clearance_from_gutter_skirt(course, bridge_spans):
    sec, _, _ = course
    i, j = bridge_spans[16]
    for sd, hw in ((1, sec.hw_left), (-1, sec.hw_right)):
        v, _, _ = small_bridges.sill_mesh(sec, i, j, sd)
        distance = np.linalg.norm(v[2::4, :2] - np.c_[sec.x[i:j + 1], sec.y[i:j + 1]], axis=1)
        np.testing.assert_allclose(distance - hw[i:j + 1], GUTTER_WIDTH + 0.03, atol=1e-8)


def test_w_beam_cut_edges_have_support_and_s5_continues(course, bridge_spans, tmp_path, monkeypatch):
    sec, _, feats = course
    monkeypatch.setattr(dress_mountain, "gaps", lambda sec: [])
    monkeypatch.setattr(dress_mountain.dm, "leaf_meshes", lambda *args, **kwargs: [])
    m = Manifest(tmp_path)
    dress_mountain.build_parts(sec, course[1], feats, dress_mountain.table_rows(), m, bridge_spans=bridge_spans)
    for number, side in ((2, "right"), (16, "left"), (25, "right")):
        i, j = bridge_spans[number]
        sd = 1 if side == "left" else -1
        for row, end in ((i, "start"), (j, "end")):
            v, _, _ = mesh(m, f"mt_smallspan_{number:02d}_{side}_{end}_post")
            assert np.ptp(v[:, 2]) > 0.65
            assert np.max(np.abs(signed_ends(sec, row, row, v))) < 0.08
            off = (v[:, :2] - [sec.x[row], sec.y[row]]) @ (sd * np.array(
                [-np.sin(sec.heading[row]), np.cos(sec.heading[row])]))
            hw = (sec.hw_left if sd > 0 else sec.hw_right)[row]
            assert off.min() > hw + 0.6
    _, j = bridge_spans[25]
    v, _, _ = mesh(m, "mt_smallspan_25_right_join")
    ends = signed_ends(sec, j, j + 1, v)
    assert ends[:, 0].min() == pytest.approx(0, abs=1e-6)
    assert ends[:, 1].max() == pytest.approx(0, abs=1e-6)


@pytest.mark.parametrize("filename,callee", [("track_build_manifest.py", "add_features"),
                                             ("dress_mountain.py", "build_parts")])
def test_normal_build_entry_points_supply_production_spans(filename, callee):
    # 全体のビルドは呼ばず、main の引数が本番データに由来することを確かめる。
    tree = ast.parse((Path(__file__).parents[1] / "scripts" / filename).read_text())
    main, = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main"]
    passed, = [kw.value for call in ast.walk(main) if isinstance(call, ast.Call)
               and (getattr(call.func, "id", None) or getattr(call.func, "attr", None)) == callee
               for kw in call.keywords if kw.arg == "bridge_spans"]
    if isinstance(passed, ast.Name):
        passed, = [n.value for n in main.body if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == passed.id for t in n.targets)]
    assert ast.unparse(passed) == "small_bridges.spans(sec, bounds, small_bridges.load())"


def test_shortened_rail_278_preserves_outside_triangles(course, tmp_path, monkeypatch):
    from dataclasses import replace
    from yabitsu import terrain

    sec, bounds, fs = course
    revised = fs[278]
    original = replace(revised, s_to=revised.layout_s_to, layout_s_to=0.0)
    assert revised.s_to == pytest.approx(bounds["s4"][0] + 6237)
    assert revised.layout_s_to == pytest.approx(bounds["s4"][0] + 6420)
    rows = dress_mountain.table_rows()
    monkeypatch.setattr(dress_mountain, "gaps", lambda sec: [])
    monkeypatch.setattr(dress_mountain.dm, "leaf_meshes", lambda *args, **kwargs: [])

    def outside_faces(feat, tag, table):
        m = Manifest(tmp_path / tag)
        dress_mountain.build_parts(sec, bounds, [feat], table, m)
        result = []
        for name in m.meshes:
            v, f, uv = mesh(m, name)
            triangles = np.c_[v, uv][kn5._triangles(f)]
            xy = triangles[..., :2].reshape(-1, 2)
            frame = terrain.road_frame(sec, *xy.T)
            distances = (sec.s[frame["idx"]] + frame["along"] - bounds["s4"][0]).reshape(-1, 3)
            external = (distances < 6204).any(axis=1)
            result.extend(t.tobytes() for t in triangles[external])
        return sorted(result)

    old_rows = [r for r in rows if not (r["section"] == "s4" and r["s_from"] == "6336" and r["item"] == "rail")]
    expected = outside_faces(original, "old", old_rows)
    assert expected
    assert outside_faces(revised, "new", rows) == expected


def test_shortened_rail_only_extends_the_last_piece_for_layout_and_gets_a_terminal(course, tmp_path, monkeypatch):
    sec, bounds, fs = course
    rail = fs[278]
    split = bounds["s4"][0] + 6226
    table = [{"section": "s4", "item": "rail", "side": "L", "s_from": "6226", "s_to": "6237", "kind": "wbeam_aged"}]
    monkeypatch.setattr(dress_mountain, "rail_pieces",
                        lambda f, _: [(f.s_from, split, "wbeam"), (split, f.s_to, "wbeam_aged")])
    monkeypatch.setattr(dress_mountain, "gaps", lambda sec: [])
    monkeypatch.setattr(dress_mountain.dm, "leaf_meshes", lambda *args, **kwargs: [])
    original = dress_mountain.dm.r70_guardrail_meshes
    calls = []
    def record(sec, feature, parts, **kwargs):
        calls.append((feature.s_from, feature.s_to))
        return original(sec, feature, parts, **kwargs)
    monkeypatch.setattr(dress_mountain.dm, "r70_guardrail_meshes", record)
    m = Manifest(tmp_path)
    dress_mountain.build_parts(sec, bounds, [rail], table, m)
    assert calls[:2] == [(rail.s_from, split), (split, rail.layout_s_to)]
    assert "mt_rail_000_1_r70_cut_end" in m.meshes
