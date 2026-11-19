"""実際に描く谷の面で水面を切り、川原の急な板と旧水位を残さない。"""
import numpy as np
import shapely
from shapely.geometry import LineString, box

from yabitsu import water
from yabitsu.valleys import Valley


def ground(x, y):
    return .02 * (12 - x) + .5 * np.abs(y)


def test_clipping_uses_fine_ground_and_removes_old_bank():
    xs = np.arange(0., 12.01, .5)
    ys = np.arange(-4., 4.01, .5)
    x, y = np.meshgrid(xs, ys)
    verts = np.c_[x.ravel(), y.ravel(), ground(x, y).ravel()]
    a = np.arange(x.size).reshape(x.shape)[:-1, :-1].ravel()
    faces = np.r_[np.c_[a, a + 1, a + 1 + len(xs)], np.c_[a, a + 1 + len(xs), a + len(xs)]]
    near = dict(xs=xs[::4], ys=ys[::4], Z=np.zeros((5, 7)), D=np.zeros((5, 7)), H=np.zeros((4, 6), bool), valley_vertices=verts, valley_faces=faces)
    area = box(0., -4., 12., 4.)
    v = np.array([[-2., -1., 1.8], [14., -1., 1.8], [14., 1., 1.8], [-2., 1., 1.8]])
    mesh = (v, np.array([[0, 1, 2], [0, 2, 3]]), v[:, :2])
    line = LineString([[0., 0.], [12., 0.]])
    st = water.Stream(line, 2., "river", [1.78, 1.78], 90334744)
    valley = Valley("B", "s4", 6354., np.asarray(line.coords), {"mode": "absolute", "z_m": [.24, 0.]},
                    2., .5, .5, 4., {"osm_id": st.osm_id, "depth_m": .30})
    h = np.array([.54, .3])
    reach = water.ValleyWater(valley, np.array([0., 12.]), h, None, area, area)
    parts = list(water._scoped_stream(12, st, [mesh], [mesh], reach, {st.osm_id: reach},
                                      ground, near, area, {}, shapely.Polygon(), None))
    assert not any(n.startswith("riverbed_012_valley") for n, *_ in parts)
    fresh = np.concatenate([v[f] for n, v, f, *_ in parts if n.startswith("water_012_valley")])
    # 両端の接続部以外では旧水位を拾わず、水際の頂点まで同じ河川水面にある。
    interior = fresh[(fresh[..., 0].min(1) >= 5.) & (fresh[..., 0].max(1) <= 7.)]
    assert len(interior)
    np.testing.assert_allclose(interior[..., 2], .30 + .02 * (12. - interior[..., 0]), atol=1e-7)
    bank = water.shore_samples(fresh)
    bank = bank[(bank[:, 0] > 5.) & (bank[:, 0] < 7.)]
    assert len(bank) > 10
    np.testing.assert_allclose(bank[:, 2], ground(*bank[:, :2].T), atol=1e-7)


def test_a_manual_inflow_exists_when_existing_osm_is_lower():
    from types import SimpleNamespace
    sec = SimpleNamespace(s=np.array([0., 100.]), x=np.array([0., 100.]), y=np.zeros(2))
    stream = water.Stream(LineString([[0., 0.], [4., 0.]]), 2., "stream", [3.0, 2.5], water.A_CREEK)
    row = Valley("A", "s4", 50., np.array([[2., 6.], [2., 3.], [2., 0.]]),
                 {"mode": "absolute", "z_m": [5., 4., 2.5]}, 3., 1., 1., 11.,
                 {"osm_id": water.A_CREEK, "depth_m": .1, "width_m": 1.})
    ground_z = lambda x, y: np.interp(y, [0., 3., 6.], [2.5, 4., 5.]) + .2 * abs(x - 2.)
    levels = water.valley_surface_levels(sec, {"s4": (0., 100.)}, [row], [stream], ground_z)
    assert water.A_CREEK in levels
    assert levels[water.A_CREEK].height[0] > levels[water.A_CREEK].height[-1]


def test_audit_pairs_each_new_water_with_its_own_original():
    from types import SimpleNamespace
    from yabitsu import valley_fit

    faces = np.array([[0, 1, 2], [0, 2, 3]])
    def surface(name, x0, x1):
        v = np.array([[x0, 0, 1], [x1, 0, 1], [x1, 1, 1], [x0, 1, 1.]])
        return valley_fit.Mesh(name, v, faces, "water", True)

    # 旧水面につながる一辺を谷岸として測ると、水深が浮きに誤分類される。
    meshes = [surface("water_022_00", -1, 0), surface("water_022_valley_00", 0, 1)]
    levels = {water.A_CREEK: SimpleNamespace(valley=SimpleNamespace(id="A"))}
    ground_z = lambda x, y: np.where((x == 0) & (0 < y) & (y < 1), 0., 1.)
    assert not valley_fit.water_join_checks(meshes, levels, ground_z)


def test_both_streams_meet_the_drawn_valleys_and_their_original_water():
    """控えの実地形から二本だけを組み立て、水際・接続・元の青い段の位置を測る。"""
    import json
    import pytest
    from yabitsu import config, gsi, kn5, side_roads, small_bridges, start_lot, terrain, track, tunnel
    from yabitsu import valley_surface, valleys
    from yabitsu.crosssection import Section
    from yabitsu.valley_fit import Surfaces

    saved = config.PROJECT_ROOT / "export/compare/route70/work/before_34"
    if not (saved / "track_terrain_near.npz").is_file():
        pytest.skip("保存済みの依頼34前の地形と河川を使用")
    sec = Section.load_json(config.DATA_PROCESSED / "track_section.json")
    bounds = track.load_bounds()
    rows = valleys.load()
    spans = small_bridges.spans(sec, bounds, small_bridges.load())
    pads = json.loads((config.DATA_PROCESSED / "track_pads.json").read_text())
    pads += start_lot.flats_for(config.DATA_MANUAL / "s1_start_lot.json", pads)
    keep = tunnel.carve_mask(sec.s, tunnel.from_osm(side_roads.load_elements(), sec))
    with np.load(saved / "track_terrain_near.npz") as src:
        before = dict(src)
    after = {**before, **valley_surface.build(sec, before, rows, bounds, spans, pads=pads, keep_rows=keep)}
    ground_z = terrain.terrain_mesh_z(after, before)
    ground = valleys.changed_ground(before, after)
    streams, _, _ = water.load(sec, gsi.default_sampler(gsi.TileCache(offline=True)))
    levels = water.valley_surface_levels(sec, bounds, rows, streams, ground_z, near=after)
    road = water.road_area(sec, start_lot.road_formation(sec, 0., sec.s[-1])).buffer(.5)
    for index, stream in enumerate(streams):
        if stream.osm_id not in (water.A_CREEK, 90334744):
            continue
        reach = levels[stream.osm_id]
        meshes = []
        for kind in ("water", "riverbed"):
            with np.load(saved / "track/meshes" / f"{kind}_{index:03d}_00.npz") as src:
                meshes.append([(src["verts"], src["faces"], src["uvs"])])
        parts = list(water._scoped_stream(index, stream, *meshes, reach, levels, ground_z,
                                          after, ground.polygon, {}, road, ground.old.height))
        new = np.concatenate([v[f] for name, v, f, *_ in parts if "_valley_" in name])
        old = np.concatenate([v[kn5._triangles(f)] for name, v, f, *_ in parts
                              if name.startswith("water_") and "_valley_" not in name])
        shore = water.shore_samples(new, old)
        assert len(shore) > 500
        assert np.max(np.abs(shore[:, 2] - ground_z(*shore[:, :2].T))) <= .05
        edge = water.shore_samples(new)
        previous = Surfaces(old).height(edge)
        joins = np.isfinite(previous)
        assert joins.any()
        assert np.max(np.abs(edge[joins, 2] - previous[joins])) <= .01
        if stream.osm_id == water.A_CREEK:
            line = LineString(reach.valley.line)
            xy = shapely.get_coordinates(shapely.line_interpolate_point(line, np.arange(0., line.length, .10)))
            assert np.isfinite(Surfaces(np.r_[new, old]).height(xy)).all()
        else:
            center = sec.s[spans[16][0]]
            # 橋の中央右、標準地図の青線を横切る断面。対岸の砂利州の
            # 裏側にある低い地面まで、二本目の水面を伸ばさない。
            station = bounds["s4"][0] + 6354.
            theta = np.interp(station, sec.s, np.unwrap(sec.heading))
            origin = np.array([np.interp(station, sec.s, axis) for axis in (sec.x, sec.y)])
            right = np.array([np.sin(theta), -np.cos(theta)])
            transect = LineString([origin, origin + 30. * right])
            channels = [p for p in shapely.get_parts(water._wet_footprint(new).intersection(transect))
                        if p.length > .01]
            assert len(channels) == 1 and 1.5 <= channels[0].length <= 5.
            for name, v, f, *_ in parts:
                triangles = v[kn5._triangles(f)]
                xy = triangles.mean(axis=1)[:, :2]
                rf = terrain.road_frame(sec, *xy.T)
                local = sec.s[rf["idx"]] + rf["along"] - center
                nearby = (local >= -19.) & (local <= 24.) & (abs(rf["lat"]) < 35.)
                if "_valley_" in name:
                    vectors = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
                    assert (vectors[nearby, 2] / np.linalg.norm(vectors[nearby], axis=1) >= np.cos(np.pi/6)).all()
                else:
                    assert not nearby.any()


def test_shallow_water_over_ground_distinguishes_wetted_area_from_duplicate_faces():
    from yabitsu.valley_fit import overlap_checks

    soil = np.array([[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]])
    def inspect(depth, other="terrain"):
        wet = soil.copy()
        wet[0, :, 2] = depth
        return overlap_checks([("water_012_valley_00", wet), (other, soil)])

    assert not inspect([0., .012, .012])
    assert not inspect([.006, .016, .025])
    assert inspect([0., 0., 0.])
    assert inspect([.0004, .0005, .0006])
    assert inspect([-.002, .012, .012])
    assert inspect([.005, .010, .010], other="smallspan_16_deck")
