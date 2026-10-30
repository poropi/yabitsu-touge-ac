import json

import numpy as np
import pytest
from shapely.geometry import Point, Polygon

from yabitsu import asama_area, config
from yabitsu.crosssection import Section


@pytest.fixture(scope="module")
def site():
    sec = Section.load_json(config.DATA_PROCESSED / "track_section.json")
    spec = asama_area.load(config.DATA_MANUAL / "s2_asama.json")
    return asama_area.build(spec, sec), sec


def test_pad_is_beside_road_not_on_it(site):
    s, sec = site
    pad = s.pad_polygon()
    assert pad.area > 80
    road = asama_area.road_formation(sec, s_min=7150, s_max=7350)
    assert pad.intersection(road).area < 1.0


def test_pads_for_terrain(site):
    s, _ = site
    pads = s.pads()
    names = {p["name"] for p in pads}
    assert names == {"asama_pad", "asama_stairs"}
    st = next(p for p in pads if p["name"] == "asama_stairs")
    cx, cy, z0, gx, gy = st["plane"]
    assert gy > 0.4                          # 石段は北へ上る


def test_stairs_climb_to_terrain(site):
    s, _ = site
    meshes = {n: v for n, v, f, uv, mat, r in s.meshes()}
    st = meshes["1ROAD_asama_stairs"]
    assert st[:, 2].max() - st[:, 2].min() > 8.0


def test_meshes_have_collision_and_names(site):
    s, _ = site
    names = [n for n, *_ in s.meshes()]
    for need in ("1ROAD_asama_pad", "1WALL_asama_wall", "1WALL_asama_parapet", "1WALL_asama_monument", "asama_pillar",
                 "asama_handrails"):
        assert any(n.startswith(need) for n in names), need


def test_wall_behind_pad(site):
    s, _ = site
    meshes = {n: v for n, v, f, uv, mat, r in s.meshes()}
    pad = s.pad_polygon()
    w = meshes["1WALL_asama_wall"]
    c = w[:, :2].mean(0)
    assert c[1] > pad.centroid.y                 # 壁は広場の北（奥）
    assert w[:, 2].max() - s.z(*c) > 3.0


def _normals(v, f):
    f = np.asarray(f)
    return np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]]), v[f[:, :3]].mean(1)


@pytest.mark.parametrize("name", ["1WALL_asama_wall", "asama_board", "asama_notice"])
def test_faces_look_toward_lot_or_sky(site, name):
    """表（頂点が反時計回りに見える側）が駐車場か空を向く（裏向きだと片面の材質で消える）。"""
    s, _ = site
    v, f = next((v, f) for n, v, f, uv, mat, r in s.meshes() if n == name)
    nrm, c = _normals(v, f)
    lot = np.asarray(s.pad_polygon().centroid.coords[0])
    toward = (lot - c[:, :2]) * nrm[:, :2]
    ok = (toward.sum(1) > 0) | (nrm[:, 2] > 0.3 * np.linalg.norm(nrm, axis=1))
    assert ok.mean() > 0.6, ok.mean()


def test_lot_matches_photo_size(site):
    """航空写真の駐車場：奥の壁から道の縁まで約 5.7 m、壁の長さ（斜めの東の端まで）約 30 m。"""
    s, sec = site
    sp = s.spec
    L = np.asarray(sp["wall"]["line"], float)
    assert 28.0 < np.linalg.norm(np.diff(L, axis=0), axis=1).sum() < 34.0
    xy = s.f.xy(*L[len(L) // 3])
    i = int(np.argmin(np.hypot(sec.x - xy[0], sec.y - xy[1])))
    d = np.hypot(sec.x[i] - xy[0], sec.y[i] - xy[1]) - sec.hw_right[i]
    assert 5.0 < d < 6.5, d


def test_road_edge_beside_lot_has_no_gutter(site):
    s, _ = site
    (side, s0, s1), = s.mouths()
    assert side == -1 and s0 < 7235 and s1 > 7262      # 駐車場の前（右＝北）の側溝をやめ、路肩の線を破線にする


def test_monument_faces_lot_and_sits_on_ground(site):
    """石碑の正面（模型の −Y＝碑文面）は駐車場（道）の側、台座は路面から浮かない。"""
    s, sec = site
    d = {n: (v, f) for n, v, f, uv, mat, r in s.meshes()}
    v, _ = d["asama_monument"]
    c = v[:, :2].mean(0)
    i = int(np.argmin(np.hypot(sec.x - c[0], sec.y - c[1])))
    road = np.array([sec.x[i], sec.y[i]])
    top = v[v[:, 2] > v[:, 2].min() + 1.5]               # 碑の部分（台座より上）
    hi = top[np.argsort(np.hypot(*(top[:, :2] - road).T))[:20]].mean(0)
    lo = top[np.argsort(-np.hypot(*(top[:, :2] - road).T))[:20]].mean(0)
    assert np.hypot(*(hi[:2] - road)) < np.hypot(*(lo[:2] - road))
    base = v[v[:, 2] < v[:, 2].min() + 0.05][:, :2].mean(0)
    assert v[:, 2].min() <= s.z(*base) + 0.01
    assert any(n.startswith("1WALL_asama_monument") for n in d)


def test_vines_cover_whole_wall(site):
    """蔓草（Codex の区切りか、無ければ帯）が壁の全長にかぶさり、下の端が地面から 1.2 m より上。"""
    s, _ = site
    d = {n: v for n, v, f, uv, mat, r in s.meshes()}
    vines = np.concatenate([v for n, v in d.items() if n.startswith("asama_vines")])
    wall = d["1WALL_asama_wall"]
    wa, _ = s.f.ab(wall[:, :2]); va, _ = s.f.ab(vines[:, :2])
    assert va.min() <= wa.min() + 0.5 and va.max() >= wa.max() - 0.5
    for a in np.linspace(wa.min() + 1, wa.max() - 1, 8):
        near = vines[np.abs(va - a) < 0.5]
        assert len(near), a
        assert near[:, 2].min() - s.z(*near[:, :2].mean(0)) > 1.0, a


def test_terrain_pad_reaches_behind_wall(site):
    """地形を平らにする範囲は、壁の根元から裏へ 2 m 以上（網目 2 m でも壁の手前に斜面がにじみ出ない）。"""
    s, _ = site
    pad = next(p for p in s.pads() if p["name"] == "asama_pad")
    poly = Polygon(pad["polygon"])
    for a in (0.3, 5.0, 15.0):
        a_, b_ = s.wall_point(a, -2.0)
        assert poly.contains(Point(*s.f.xy(a_, b_))), a


def test_lot_pad_sinks_terrain_cells(site):
    """駐車場の土台は kind=pass（地形の格子の角を舗装より下げる）。石段の側面は駐車場の高さより下まで届く。"""
    s, _ = site
    pad = next(p for p in s.pads() if p["name"] == "asama_pad")
    assert pad["kind"] == "pass"
    st = {n: v for n, v, f, uv, mat, r in s.meshes()}["1ROAD_asama_stairs"]
    a0, a1, b0, b1, z0, z1, n = s._stairs()
    sa, sb = s.f.ab(st[:, :2])
    high = st[sb > b0 + (b1 - b0) * 0.6]
    assert high[:, 2].min() < z0             # 上の方の段の側面も、下の端は石段の下の端より下


def test_stair_wall_rises_beside_stairs(site):
    """石段の左の石積み：石段ぞいの面の上の端は段より上、道ぞいの面は石段の角で高い。中の詰まった壁で、面は外向き。"""
    s, _ = site
    d = {n: (v, f) for n, v, f, uv, mat, r in s.meshes()}
    a0, a1, b0, b1, z0, z1, n = s._stairs()
    v, f = d["1WALL_asama_stairwall_0"]
    assert v[:, 2].max() > z0 + (z1 - z0) * 0.3 + 1.5
    nrm, c = _normals(v, f)
    assert (np.einsum("ij,ij->i", nrm, c - v.mean(0)) > 0).all()
    v2, _ = d["1WALL_asama_stairwall_1"]
    assert v2[:, 2].max() - z0 > 3.5


def test_wall_faces_have_spread_uv(site):
    """石積みの大きな面は、uv が縦横どちらにも広がっている（端の面の u が全部同じだと模様の 1 列が伸びて横縞になった。2026-10-30）。"""
    s, _ = site
    for name in ("1WALL_asama_stairwall_0", "1WALL_asama_stairwall_1", "1WALL_asama_wall"):
        v, f, uv = next((v, f, uv) for n, v, f, uv, mat, r in s.meshes() if n == name)
        for fc in f:
            p = v[fc]; area = np.linalg.norm(np.cross(p[1] - p[0], p[2] - p[0])) + np.linalg.norm(np.cross(p[2] - p[0], p[3] - p[0]))
            if area < 2 * 0.2:
                continue
            span = np.ptp(uv[fc], axis=0)
            assert (span > 0.02).all(), (name, fc, uv[fc])
