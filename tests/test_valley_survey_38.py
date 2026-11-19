import json
from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import LineString, Point

from yabitsu import config, geo, gsi, kn5, valleys
from yabitsu.valley_fit import Surfaces


def test_tributary_head_uses_local_laser_elevations():
    tributary = next(v for v in valleys.load() if v.id.endswith("crossing"))
    samples = gsi.default_sampler(gsi.TileCache(offline=True)).samplers[0]
    laser = samples.sample(*geo.from_local(*tributary.line[:3].T))
    # bed は削り込みの上限。地形を持ち上げず、既存のレーザー標高を残す。
    assert np.max(laser - tributary.bed["z_m"][:3]) < .15


def test_bridge_b_survey_stays_anchored_to_the_mapped_waterline():
    data = Path("export/compare/terrain_audit/waterlines.geojson")
    if not data.exists():
        pytest.skip("保存済み水系線を使用")
    waterlines = json.loads(data.read_text())["features"]
    river = next(f["properties"] for f in waterlines if f["properties"]["id"] == "G070")
    mapped = LineString(river["local_xy_m"])
    channel = next(v for v in valleys.load() if v.id.endswith("river"))
    from yabitsu import valley_survey
    reference = valley_survey.bank_repair(channel).row
    assert max(mapped.distance(Point(p)) for p in reference.line[20:27]) < .02


def test_a_tributary_flows_into_original_downstream_surface():
    saved = Path("export/compare/route70/work/before_34/track/meshes/water_022_00.npz")
    if not saved.exists():
        pytest.skip("保存済み水面を使用")
    creek = valleys.load()[0]
    assert creek.water and creek.water["width_m"] == 1.
    with np.load(saved) as d:
        surface = Surfaces(d["verts"][kn5._triangles(d["faces"])])
    z = surface.height(creek.line[-3:])
    assert np.isfinite(z).all()
    assert np.diff(z).max() < 0
    np.testing.assert_allclose(z - creek.bed["z_m"][-3:], .32, atol=.02)


def test_b_survey_recovers_missing_laser_bank_only():
    from yabitsu import valley_survey

    row = next(v for v in valleys.load() if v.id.endswith("river"))
    repair = valley_survey.bank_repair(row)
    # 約 S4 6354。橋の直右の10m補間域と対岸の5m実測域・山側を比較する。
    from yabitsu import track
    from yabitsu.crosssection import Section
    section = Section.load_json(config.DATA_PROCESSED / "track_section.json")
    station = track.load_bounds()["s4"][0] + 6354
    heading = np.interp(station, section.s, np.unwrap(section.heading))
    lateral = np.array([-16., -20., -25., -29., 15.])
    x = np.interp(station, section.s, section.x) - np.sin(heading) * lateral
    y = np.interp(station, section.s, section.y) + np.cos(heading) * lateral
    sampler = gsi.default_sampler(gsi.TileCache(offline=True))
    laser = sampler.samplers[0].sample(*geo.from_local(x, y))
    assert np.isnan(laser[:3]).all() and np.isfinite(laser[3:]).all()
    assert repair.contains(x, y)[:3].all()
    z = repair.height(x, y)
    assert np.all((438.8 < z[:3]) & (z[:3] < 440.8))
    assert np.isinf(z[-1]) and not repair.contains(x, y)[-1]
    assert valley_survey.bank_repair(valleys.load()[0]) is None


def test_tributary_head_and_c_outlet_join_a_descending_existing_slope():
    from yabitsu.valley_surface import sample_grid
    saved = Path("export/compare/route70/work/before_34/track_terrain_near.npz")
    if not saved.exists():
        pytest.skip("保存済み地形を使用")
    rows = valleys.load()
    with np.load(saved) as data:
        grid = {k: data[k] for k in ("xs", "ys", "Z")}
    for a, b in (rows[2].line[:2], rows[3].line[-2:]):
        xy = np.linspace(a, b, 201)
        z = sample_grid(grid, *xy.T)
        assert np.max(z - np.minimum.accumulate(z)) < .001


def test_bank_repair_extends_to_both_ends_of_the_same_dem_void():
    from yabitsu import valley_survey

    river = next(v for v in valleys.load() if v.id.endswith("river"))
    repair = valley_survey.bank_repair(river)
    # 旧2 m格子の10 m DEMの山に連なり、5 m標高面よりそれぞれ12.2 m・2.8 m高い2点。
    xy = np.array([[-382., 4078.], [-452., 4148.]])
    valid = np.isfinite(repair.height(*xy.T))
    assert valid.all()


def test_b_low_flow_channel_tracks_map_at_bridge_and_connects_to_tributary():
    from yabitsu import terrain
    from yabitsu.crosssection import Section

    main, tributary = valleys.load()[1:3]
    assert main.width_m == 3.
    assert 0. < main.water["depth_m"] <= .25
    assert LineString(main.line).distance(Point(tributary.line[-1])) < .01
    sec = Section.load_json(config.DATA_PROCESSED / "track_section.json")
    near_bridge = LineString(main.line).intersection(Point(tributary.line[-1]).buffer(3.))
    xy = np.asarray(near_bridge.coords)
    frame = terrain.road_frame(sec, *xy.T)
    assert np.all((8.25 < -frame["lat"]) & (-frame["lat"] < 10.75))


def test_original_missing_laser_nodes_use_the_same_measured_tin_after_moving_channel():
    from yabitsu import valley_survey

    repair = valley_survey.bank_repair(valleys.load()[1])
    p = np.array([[-440., 4114.], [-442., 4116.], [-438., 4112.]])
    reference, missing = repair.reference(*p.T)
    assert missing.all()
    np.testing.assert_allclose(reference, [443.559, 443.210, 443.263], atol=.003)


def test_drawn_centerline_grade_is_enforced_across_triangle_diagonals():
    from yabitsu import valley_drainage
    from types import SimpleNamespace

    x, y = np.meshgrid(np.arange(13.)*.5, np.arange(9.)*.5)
    z = 15. - .12*x + .03*(y-2.)**2
    z += .55*np.exp(-((x-3.)**2+(y-2.)**2)/.3)
    ids = np.arange(x.size).reshape(x.shape)
    a, b, c, d = ids[:-1,:-1], ids[:-1,1:], ids[1:,:-1], ids[1:,1:]
    faces = np.concatenate([np.stack((a,b,d),-1).reshape(-1,3), np.stack((a,d,c),-1).reshape(-1,3)])
    vertices = np.c_[x.ravel(),y.ravel(),z.ravel()]
    fixed = ((x == 0)|(x == 6)|(y == 0)|(y == 4)).ravel()
    row = SimpleNamespace(line=np.array([[0.,1.8],[6.,2.25]]))
    xy = np.linspace(*row.line,801)
    start = Surfaces(vertices[faces]).height(xy)
    assert np.max(start - np.minimum.accumulate(start)) > .1
    def no_survey(x, y):
        return np.zeros_like(x), np.zeros_like(x,dtype=bool)
    revised = valley_drainage.downhill_grades(vertices, faces, fixed, [row], no_survey)
    np.testing.assert_array_equal(revised[fixed],vertices[fixed,2])
    vertices[:,2] = revised
    end = Surfaces(vertices[faces]).height(xy)
    assert np.max(end - np.minimum.accumulate(end)) < .001
