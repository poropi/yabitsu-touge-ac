import numpy as np
import pytest
from shapely.geometry import Point

from yabitsu import geo, gomayashiki_area as ga, site, town_objects as to
from yabitsu.crosssection import GUTTER_WIDTH
from yabitsu.meshgen import face_normals
from tests.test_meshgen import _straight


def _ll(x, y):
    lat, lon = geo.from_local(float(x), float(y))
    return [float(lat), float(lon)]


def _spec(sec):
    e = sec.hw_left[0] + GUTTER_WIDTH       # 道は x 方向（東）へまっすぐ、西＝左（L）は +y
    return {
        "apron": {"points": [_ll(60, e - 0.3), _ll(80, e - 0.3), _ll(80, e + 10), _ll(60, e + 10)], "anchor_ll": _ll(70, e), "slope": 0.0},
        "island": {"points": [_ll(82, e - 0.3), _ll(90, e - 0.3), _ll(90, e + 2), _ll(82, e + 2)], "dz": 0.12},
        "west_curb": {"points": [_ll(60, e + 10), _ll(70, e + 10)]},
        "spring": {"pad": [_ll(60, e + 10), _ll(70, e + 10), _ll(70, e + 18), _ll(60, e + 18)], "steps_top": [_ll(61, e + 10), _ll(69, e + 10)],
                   "steps": 3, "spout": _ll(65, e + 15), "basin_r": 1.3, "spout_h": 0.9, "pipes_deg": [0, 180]},
        "loop": {"points": [_ll(76, e + 8), _ll(76, e + 40), _ll(120, e + 40), _ll(120, 0)], "width_m": 5.0},
        "paint": {"bays": [[_ll(62, e + 2), _ll(62, e + 7)]], "edge": [_ll(60, e + 1), _ll(80, e + 1)], "hatch": [[_ll(64, e + 1), _ll(65, e + 2)]]},
    }


def test_apron_spring_and_loop_heights():
    sec = _straight(201)
    sec.z = 100.0 + 0.02 * sec.s               # 道は 2 % で上る
    s = ga.build(_spec(sec), sec)
    e = sec.hw_left[0] + GUTTER_WIDTH
    for x in (62.0, 78.0):                      # 広場は道の縁にそって段がない
        assert site.plane_z(s.plane, x, e) == pytest.approx(float(sec.edge_z(1)[int(x)]), abs=0.01)
    assert s.pad_z == pytest.approx(float(site.plane_z(s.plane, *s.pad.centroid.coords[0])) - 0.6)   # 湧き水の面は 3 段下
    assert s.loop_z0 == pytest.approx(float(site.plane_z(s.plane, 76, e + 8)))
    assert s.loop_z1 == pytest.approx(float(sec.edge_z(1)[120]), abs=0.05)                         # 戻り口は道の縁の高さ
    zs = [pl[2] for _, pl in s.loop_segs]
    assert zs == sorted(zs, reverse=zs[0] > zs[-1])                                                 # 輪は一方へなめらかに移る
    for (pa, a), (pb, b) in zip(s.loop_segs, s.loop_segs[1:]):                                       # 区切りの線で前後の平面がそろう
        q = pa.buffer(0.01).intersection(pb.buffer(0.01)).representative_point()
        assert site.plane_z(a, q.x, q.y) == pytest.approx(site.plane_z(b, q.x, q.y), abs=0.03)
    mo = s.mouths()
    assert (1, pytest.approx(60.0, abs=1.5), pytest.approx(80.0, abs=1.5)) in mo                    # 広場の口とわき道の戻り口は別の区間
    assert any(a > 110 and b < 125 for side, a, b in mo)
    names = {m[0]: m for m in s.meshes()}
    assert {"1ROAD_goma_apron", "1ROAD_goma_spring_pad", "1ROAD_goma_steps", "1WALL_goma_basin", "1WALL_goma_spout", "goma_spout_water",
            "goma_paint", "1WALL_goma_curb", "1GRASS_goma_island"} <= set(names)
    spout = names["1WALL_goma_spout"][1]
    assert spout[:, 2].max() == pytest.approx(s.pad_z + 0.9)
    assert {p["name"] for p in s.pads()} >= {"goma_apron", "goma_spring", "goma_island", "goma_loop_00"}
    area = s.area()
    assert area.contains(s.apron.representative_point()) and area.contains(Point(*s.spout))


def test_faces_point_the_right_way():
    sec = _straight(201)
    s = ga.build(_spec(sec), sec)
    names = {m[0]: m for m in s.meshes()}
    for n in ("1ROAD_goma_apron", "1ROAD_goma_spring_pad", "goma_paint", "goma_basin_floor"):     # 面は上向き
        v, f = names[n][1], names[n][2]
        assert (face_normals(v, f)[:, 2] > 0.9).all(), n
    v, f = names["1WALL_goma_spout"][1:3]                                                          # 柱の側面は外向き
    c = np.r_[s.spout, 0.0]; nrm = face_normals(v, f); mid = v[np.asarray(f)[:, :3]].mean(1)
    side = np.abs(nrm[:, 2]) < 0.5
    assert (np.einsum("ij,ij->i", nrm[side, :2], (mid - c)[side, :2]) > 0).all()


def test_guard_pipe_rails_sawhorse_legs_and_log_fence_ends():
    flat = lambda x, y: np.full(np.shape(x), 100.0)
    ms = to.object_meshes([{"kind": "guard_pipe", "material": "pole_white", "rails": [0.3, 0.55, 0.8], "points": [_ll(0, 0), _ll(4, 0)]},
                           {"kind": "sawhorse", "beam": "pole_orange", "legs": "pole_orange", "points": [_ll(0, 5), _ll(1.5, 5)]},
                           {"kind": "wood_fence", "ends": "wood", "points": [_ll(0, 9), _ll(4, 9)]}], flat, prefix="o")
    by = {m[0]: m for m in ms}
    w = by["o_pole_white"][1]
    assert w[:, 2].max() == pytest.approx(100.84, abs=0.01)                         # 3 段の横棒の上の段 + 支柱の頭
    assert np.isclose(w[:, 2], 100.3 - 0.03).any() and np.isclose(w[:, 2], 100.55 + 0.03).any()
    assert "o_wood" in by and w[:, 1].max() < 1.0                                  # 丸太の柵（y = 9 m）の両端は白くない
    o = by["o_pole_orange"][1]
    assert o[:, 2].min() < 100.0                                                    # 橙色の脚が地面まで


def test_bar_fence_has_bars_rails_and_hidden_collision():
    flat = lambda x, y: np.full(np.shape(x), 100.0)
    ms = to.object_meshes([{"kind": "bar_fence", "height_m": 1.1, "bar_every_m": 0.15, "points": [_ll(0, 0), _ll(3, 0)]}], flat, prefix="o")
    by = {m[0]: m for m in ms}
    v = by["o_rubber_black"][1]
    assert v[:, 2].max() == pytest.approx(101.15, abs=0.01)
    bars = [x for x in np.unique(np.round(v[:, 0], 3)) if 0.05 < x < 2.95]
    assert len(bars) >= 2 * 18                                                     # 0.15 m おきに 20 本（両側の縁で 2 つの x）
    assert by["1WALL_o_woodfence"][5] is False


def test_mesh_fence_material_and_seamless_green_mesh():
    from yabitsu import textures as tx
    flat = lambda x, y: np.full(np.shape(x), 100.0)
    ms = to.object_meshes([{"kind": "mesh_fence", "material": "mesh_green", "points": [_ll(0, 0), _ll(4, 0)]},
                           {"kind": "mesh_fence", "points": [_ll(0, 5), _ll(4, 5)]}], flat, prefix="o")
    assert {"o_mesh_green", "o_railing_bars"} <= {m[0] for m in ms}                 # 既定は今までどおりの縦格子
    a = tx.mesh_green()[..., 3]
    assert (a[:, :16] == a[:, -16:]).all() and (a[:16] == a[-16:]).all()            # 端と端がつながる（2 m ごとに継ぎ目が出ない）
    assert "mesh_green" in tx.MATERIALS and tx.MATERIALS["mesh_green"]["alpha"]
