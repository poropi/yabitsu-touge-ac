"""保存済みの入力に局所処理だけを適用し、実際の三角形を検査する。"""
import json

import numpy as np
import pytest
import shapely

from scripts import track_06_terrain
from yabitsu import (
    config, features, gsi, kn5, meshgen, nagoki, side_roads, small_bridges, start_lot,
    terrain, track, tunnel, valleys, water,
)
from yabitsu.crosssection import GUTTER_WIDTH, Section
from yabitsu.forest_ground import TriSoup

P = config.DATA_PROCESSED
NEAR = config.PROJECT_ROOT / "export" / "compare" / "route70" / "work" / "before_34" / "track_terrain_near.npz"
pytestmark = pytest.mark.skipif(
    not all(p.exists() for p in (P / "track_section.json", NEAR, P / "track_pads.json")),
    reason="保存済みの道路断面と近景が必要",
)


@pytest.fixture(scope="module")
def course():
    sec = Section.load_json(P / "track_section.json")
    bounds = track.load_bounds()
    # 依頼35の実際の入力 data/manual/valleys.json（依頼38着手前）。
    rows = valleys.load(config.PROJECT_ROOT / "tests/fixtures/valleys_round1.json")
    spans = small_bridges.spans(sec, bounds, small_bridges.load())
    with np.load(NEAR) as saved:
        before = dict(saved)
    after = {**before, "Z": before["Z"].copy()}
    pads = json.loads((P / "track_pads.json").read_text(encoding="utf-8"))
    pads += start_lot.flats_for(config.DATA_MANUAL / "s1_start_lot.json", pads)
    keep = tunnel.carve_mask(sec.s, tunnel.from_osm(side_roads.load_elements(), sec))
    sampler = gsi.default_sampler(gsi.TileCache(offline=True))
    feats = features.load_features_rows(track.load_section_rows("features", bounds))
    _, feats = nagoki.track_stubs(sec, pads, feats, sampler=sampler)
    track_06_terrain.update_valley_ground(sec, after, rows, bounds, sampler, track_06_terrain.terrain_features(feats, bounds), pads=pads, keep_rows=keep)
    return sec, bounds, rows, spans, before, after


def footprint(sec, i, j, shoulder):
    sl = slice(i, j + 1)
    xy = np.c_[sec.x[sl], sec.y[sl]]
    n = np.c_[-np.sin(sec.heading[sl]), np.cos(sec.heading[sl])]
    left = xy + n * (sec.hw_left[sl] + GUTTER_WIDTH + shoulder)[:, None]
    right = xy - n * (sec.hw_right[sl] + GUTTER_WIDTH + shoulder)[:, None]
    return shapely.Polygon(np.r_[left, right[::-1]])


@pytest.mark.parametrize("bridge, row", [(2, 0), (16, 2), (25, 3)])
def test_saved_terrain_has_continuous_passage_under_bridge(course, bridge, row):
    sec, _, rows, spans, before, after = course
    valley = rows[row]
    foot = footprint(sec, *spans[bridge], valleys.SHOULDER_M)
    line = shapely.LineString(valley.line).intersection(foot)
    q = shapely.get_coordinates(shapely.line_interpolate_point(line, np.linspace(0, line.length, 161)))
    ground = terrain.terrain_mesh_z(after, before)
    cap, _ = valley.profile(*q.T, None, 0)
    assert np.max(ground(*q.T) - cap) < 0.3
    x, y, z, _ = terrain.near_edit_window(after, foot.buffer(4))
    r, c = np.nonzero(shapely.contains_xy(foot, x, y))
    cap, _ = valley.profile(x[r, c], y[r, c], None, 0)
    assert np.max(z[r, c] - cap) < 0.5
    # 近接する格子点すべてより2m掘れた孤立点を数える。
    adjacent = np.stack([z[r + dr, c + dc] for dr, dc in
                         ((-1, 0), (0, -1), (0, 1), (1, 0), (-1, -1), (1, 1))])
    assert np.max(adjacent.min(axis=0) - z[r, c]) < 2


def test_saved_terrain_changes_only_inside_sites(course):
    sec, bounds, rows, _, before, after = course
    assert np.all(after["Z"] <= before["Z"])
    r, c = np.nonzero(before["Z"] != after["Z"])
    x, y = before["xs"][c], before["ys"][r]
    sites = shapely.union_all([valleys.site_area(sec, bounds, row.section, row.center_m) for row in rows])
    assert shapely.contains_xy(sites.buffer(-np.hypot(2, 2)), x, y).all()
    for key in before.keys() - {"Z"}:
        assert before[key].tobytes() == after[key].tobytes()


@pytest.mark.parametrize("bridge", [2, 16, 25])
def test_small_bridge_deck_and_abutments_are_solid_and_buried(course, bridge):
    sec, _, _, spans, before, after = course
    ground = terrain.terrain_mesh_z(after, before)
    parts = {name: (v, f) for name, v, f, uv, mat, visible in small_bridges.support_meshes(sec, spans, ground)
             if name.startswith(f"smallspan_{bridge:02d}_") and visible}
    deck, face = parts[f"smallspan_{bridge:02d}_deck"]
    underside = kn5._triangles(face)[meshgen.face_normals(deck, kn5._triangles(face))[:, 2] < -0.7]
    foot = footprint(sec, *spans[bridge], 0)
    x, y = np.meshgrid(np.arange(foot.bounds[0], foot.bounds[2], .25),
                       np.arange(foot.bounds[1], foot.bounds[3], .25))
    inside = shapely.contains_xy(foot, x, y)
    xy = np.c_[x[inside], y[inside]]
    assert np.isfinite(TriSoup(deck, underside)(*xy.T).filled(np.nan)).all()
    for tag in ("start", "end"):
        v, _ = parts[f"smallspan_{bridge:02d}_abutment_{tag}"]
        low = v[v[:, 2] == v[:, 2].min()]
        assert np.min(ground(*low[:, :2].T) - low[:, 2]) >= 0.5
    assert list(small_bridges.support_meshes(sec, {999: spans[bridge]}, ground)) == []


@pytest.fixture(scope="module")
def main_river(course):
    sec, _, rows, _, _, _ = course
    streams, _, _ = water.load(sec, gsi.default_sampler(gsi.TileCache(offline=True)))
    valley = rows[1]
    return next(st for st in streams if st.osm_id == valley.water["osm_id"]), valley


def test_main_river_before_edit_uses_original_water(course, main_river):
    sec, bounds, rows, _, before, _ = course
    stream, _ = main_river
    ground = terrain.terrain_mesh_z(before, before)
    levels = water.valley_surface_levels(sec, bounds, rows, [stream], ground)
    q = shapely.get_coordinates(shapely.line_interpolate_point(stream.line, np.arange(0., stream.line.length, .5)))
    assert water.surface_z(stream, levels, water.LIFT)(*q.T).tobytes() == stream.z_fn(water.LIFT)(*q.T).tobytes()


def test_main_river_connects_downstream_and_meets_ground_at_banks(course, main_river):
    sec, bounds, rows, _, before, after = course
    stream, valley = main_river
    ground = terrain.terrain_mesh_z(after, before)
    levels = water.valley_surface_levels(sec, bounds, rows, [stream], ground)
    reach = levels[stream.osm_id]
    line = shapely.LineString(valley.line)
    q = shapely.get_coordinates(shapely.line_interpolate_point(line, np.linspace(0., line.length, 801)))
    z = ground(*q.T)
    # 2m格子を三角形で補間した底にも、下流の堰になる逆勾配を残さない。
    assert np.max(np.diff(z)) <= 1e-7
    direction = (q[-1] - q[-2]) / np.linalg.norm(q[-1] - q[-2])
    assert np.max(ground(*(q[-1] + np.outer([.5, 1., 2.], direction)).T)) <= z[-1] + .01
    depth = water.surface_z(stream, levels, water.LIFT)(*q.T) - z
    assert depth.min() >= .2 and depth.max() <= .8
    for bank in reach.banks:
        assert np.max(abs(water.surface_z(stream, levels, water.LIFT)(*bank.T) - ground(*bank.T))) <= .3
        assert np.max(shapely.distance(shapely.points(bank), line)) <= valley.width_m / 2 + .01


def test_river_mesh_preserves_triangles_outside_the_edit_site(course, main_river):
    sec, bounds, rows, _, before, after = course
    stream, valley = main_river
    ground = terrain.terrain_mesh_z(after, before)
    levels = water.valley_surface_levels(sec, bounds, rows, [stream], ground)
    road = shapely.Polygon()
    site = valleys.site_area(sec, bounds, valley.section, valley.center_m)

    def outside(meshes):
        result = []
        for _, v, f, uv, _, _ in meshes:
            triangles = np.c_[v, uv][f]
            remote = ~shapely.intersects(site, shapely.polygons(triangles[..., :2]))
            result.extend(t.tobytes() for t in triangles[remote])
        return sorted(result)

    old = water.meshes([stream], road)
    changed = water.meshes([stream], road, levels, ground, near=after)
    assert any("_valley_" in name for name, *_ in changed)
    assert outside(changed) == outside(old)


def test_281_posts_are_supported_and_left_cut_post_is_buried(course, tmp_path, monkeypatch):
    from scripts import dress_mountain as dress
    from yabitsu import assembly
    from yabitsu.manifest import Manifest

    sec, bounds, _, spans, before, after = course
    ground = terrain.terrain_mesh_z(after, before)
    feats = dress.track_features(sec, bounds)
    assert feats[281].side == "R" and feats[281].type == "guardrail"
    monkeypatch.setattr(dress, "gaps", lambda sec: [])
    monkeypatch.setattr(dress.dm, "leaf_meshes", lambda *args, **kwargs: [])
    rail = Manifest(tmp_path / "dress")
    dress.build_parts(sec, bounds, feats, dress.table_rows(), rail, bridge_spans=spans, ground_z=ground)
    body = Manifest(tmp_path / "track")
    assembly.add_features(body, sec, feats, bridge_spans=spans, ground_z=ground)
    tops = []
    for name, entry in body.meshes.items():
        if name.startswith("smallspan_16_rail_support_"):
            with np.load(body.root / entry["file"]) as mesh:
                v = mesh["verts"]
                f = kn5._triangles(mesh["faces"])
                tops.append(TriSoup(v, f[meshgen.face_normals(v, f)[:, 2] > .9]))
    for name, entry in rail.meshes.items():
        if name.startswith("mt_rail_281_"):
            with np.load(rail.root / entry["file"]) as mesh:
                v = mesh["verts"]
            idx = terrain.road_frame(sec, *v[:, :2].T)["idx"]
            feet = v[v[:, 2] < sec.edge_z(-1)[idx] + .04]
            assert len(feet)
            base = np.maximum.reduce([ground(*feet[:, :2].T)] +
                                     [top(*feet[:, :2].T).filled(-np.inf) for top in tops])
            assert np.max(feet[:, 2] - base) <= .1
        if name.startswith("mt_smallspan_16_") and name.endswith("_post"):
            with np.load(rail.root / entry["file"]) as mesh:
                v = mesh["verts"]
            feet = v[v[:, 2] == v[:, 2].min()]
            if "_left_" in name:
                assert np.min(ground(*feet[:, :2].T) - feet[:, 2]) >= .1
            else:
                support = np.maximum.reduce([top(*feet[:, :2].T).filled(-np.inf) for top in tops])
                assert np.min(support - feet[:, 2]) >= .1


def test_three_left_walls_do_not_exceed_the_cut_slope(course, tmp_path):
    from scripts import dress_mountain as dress
    from yabitsu import assembly
    from yabitsu.manifest import Manifest

    sec, bounds, _, spans, before, after = course
    ground = terrain.terrain_mesh_z(after, before)
    feats = dress.track_features(sec, bounds)
    body = Manifest(tmp_path / "track")
    assembly.add_features(body, sec, feats, bridge_spans=spans, ground_z=ground)
    for k in (279, 280, 395):
        sl = features.rows_of(sec, feats[k])
        with np.load(body.root / body.meshes[f"1WALL_s1_retaining_{k:03d}"]["file"]) as mesh:
            top = mesh["verts"][1::2]
        h = sec.heading[sl]
        x = top[:, 0] - .30 * np.sin(h)
        y = top[:, 1] + .30 * np.cos(h)
        expected = np.maximum(ground(x, y), sec.edge_z(1)[sl])
        assert np.max(abs(top[:, 2] - expected)) < .02


@pytest.fixture(scope="module")
def fit_inputs(course, main_river):
    from yabitsu import assembly, config, valley_fit
    from scripts import dress_mountain as dress

    class Meshes:
        def __init__(self):
            self.items = {}
            self.materials = {}

        def add_mesh(self, group, name, verts, faces, uvs, material, renderable=True, *, normals=None):
            self.items[name] = valley_fit.Mesh(name, verts, faces, material, renderable, uvs, group)

    sec, bounds, rows, spans, before, after = course
    old = terrain.terrain_mesh_z(before, before)
    ground = terrain.terrain_mesh_z(after, before)
    area = shapely.union_all([valleys.site_area(sec, bounds, row.section, row.center_m) for row in rows])
    road, openings = valleys.road_areas(sec, area, (None, None), [spans[k] for k in small_bridges.SUPPORT_SPANS],
                                       -valleys.SHOULDER_M)
    feats = dress.track_features(sec, bounds)
    m = Meshes()
    assembly.add_road(m, sec, deck_spans=[spans[k] for k in small_bridges.SUPPORT_SPANS],
                      ground_z=ground, fit_area=area)
    assembly.add_features(m, sec, feats, bridge_spans=spans, ground_z=ground)
    assembly.add_meshes(m, small_bridges.support_meshes(sec, spans, ground))
    dug = valleys.changed_ground(before, after)
    dress.build_parts(sec, bounds, feats, dress.table_rows(), m, bridge_spans=spans, ground_z=ground, dug=dug)
    for k in small_bridges.SUPPORT_SPANS:
        for side, label in ((1, "left"), (-1, "right")):
            m.add_mesh(dress.GROUP, f"mt_r70_bridge_{k:02d}_{label}_sill",
                       *small_bridges.sill_mesh(sec, *spans[k], side, supported=True), "concrete")
    stream, valley = main_river
    levels = water.valley_surface_levels(sec, bounds, rows, [stream], ground)
    for name, v, f, uv, mat, visible in water.meshes([stream], shapely.Polygon(), levels, ground, near=after,
                                                    changed_area=dug.polygon):
        m.add_mesh(assembly.GROUP, name.replace("_000_", "_012_"), v, f, uv, mat, visible)
    checked = [m for m in m.items.values() if m.visible and not m.name.startswith("GUARDRAIL_s1_")]
    return sec, spans, before, after, ground, old, area, openings, checked, levels


@pytest.fixture(scope="module")
def fit_issues(fit_inputs):
    from yabitsu import valley_fit

    sec, spans, before, after, ground, old, area, openings, checked, levels = fit_inputs
    rows = valleys.load()
    channels = [(shapely.LineString(r.line), r.width_m) for r in rows if r.water]
    return (
        valley_fit.mesh_checks(checked, area, ground, old, openings=openings, channels=channels,
                               ground_tri=valley_fit.terrain_triangles(after, area)),
        valley_fit.bridge_checks(checked, sec, spans, ground),
        valley_fit.wall_checks(checked, sec, ground),
        valley_fit.water_join_checks(checked, levels, ground),
    )


def test_fit_no_bridge_end_posts_float(fit_inputs, fit_issues):
    from yabitsu import valley_fit

    sec, spans, _, _, ground, old, _, _, meshes, _ = fit_inputs
    assert not [i for i in fit_issues[0] if i.category == "foot" and i.mesh.startswith("mt_smallspan_")]
    near = shapely.buffer(footprint(sec, *spans[25], 0.), 5.)
    checked = valley_fit.mesh_checks(meshes, near, ground, old, overlaps=False)
    previous = valley_fit.mesh_checks(meshes, near, old, old, overlaps=False)
    _, worse = valley_fit.issue_changes(previous, checked)
    assert not [i for i in worse if i.category == "foot" and "mt_rail_366_" in i.mesh]


def test_fit_no_gap_at_bridge_approaches(fit_inputs):
    from yabitsu import assembly
    from yabitsu import valley_fit

    sec, spans, _, _, ground, old, _, openings, meshes, _ = fit_inputs
    near = shapely.union_all([footprint(sec, *spans[k], 0.).buffer(5.) for k in small_bridges.SUPPORT_SPANS])
    class Capture:
        meshes = []

        def add_mesh(self, group, name, v, faces, uv, material, renderable=True):
            if name.startswith("1ROAD_s1_gutter"):
                self.meshes.append(valley_fit.Mesh(name, v, faces, material, renderable))

    before = Capture()
    assembly.add_road(before, sec)
    previous = valley_fit.mesh_checks(before.meshes, near, old, old, openings=openings, overlaps=False)
    checked = valley_fit.mesh_checks(meshes, near, ground, old, openings=openings, overlaps=False)
    _, worse = valley_fit.issue_changes(previous, checked)
    assert not [i for i in worse if i.category == "skirt"]


def test_fit_reshapes_old_riverbed_and_fixes_new_drape(fit_issues):
    assert not [i for i in fit_issues[0] if i.category == "bed_over_cut"]


def test_fit_walls_follow_immediate_ground(fit_issues):
    assert fit_issues[2] == []


def test_fit_no_coplanar_bridge_parts(fit_issues):
    bad = []
    for issue in fit_issues[0]:
        if issue.category != "overlap":
            continue
        a, b = issue.mesh, issue.other
        if (("abutment" in a + b and "sill" in a + b)
                or ("gutter" in a + b and "deck" in a + b)
                or ("abutment" in a + b and "rail_support" in a + b)
                or ("riverbed_012_valley" in a and b == "terrain")):
            bad.append(issue)
    assert bad == []


def test_fit_bridge_shell_and_water_joins(fit_issues):
    assert fit_issues[1] == []
    assert fit_issues[3] == []
