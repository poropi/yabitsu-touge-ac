import numpy as np
import pytest
from shapely.geometry import Point, Polygon

from yabitsu import buildings as bd, geo
from yabitsu.meshgen import face_normals


def _ll(pts):
    lat, lon = geo.from_local(*np.asarray(pts, float).T)
    return [[float(a), float(b)] for a, b in zip(lat, lon)]


L_SHAPE = [[0, 0], [12, 0], [12, 5], [5, 5], [5, 10], [0, 10]]


def _b(**kw):
    d = {"id": "t1", "footprint": _ll(L_SHAPE), "storeys": 2, "height_m": 6.0,
         "roof": {"type": "flat", "color": "#808080"}, "wall": "#E0D8C8", "windows": True, "shopfront": [], "signs": []}
    d.update(kw)
    return bd.Building.from_spec(d)


FLAT = lambda x, y: np.full(np.shape(x), 100.0)
SLOPE = lambda x, y: 100.0 + 0.1 * np.asarray(x, float)


def _mesh(meshes, name):
    return next(m for m in meshes if m[0] == name)


def test_footprint_kept_as_is():
    ms = bd.building_meshes([_b()], FLAT)
    v = _mesh(ms, "1WALL_bld_t1_base")[1]
    bottom = v[np.isclose(v[:, 2], 100.0 - 0.3)][:, :2]
    for p in L_SHAPE:
        assert np.hypot(*(bottom - p).T).min() < 0.01        # L字の角がそのまま残る


def test_building_sits_on_lowest_ground():
    ms = bd.building_meshes([_b()], SLOPE)
    base, wall = _mesh(ms, "1WALL_bld_t1_base"), _mesh(ms, "1WALL_bld_t1")
    assert base[1][:, 2].min() == pytest.approx(100.0 - 0.3) and base[4] == bd.BASE_MAT     # x=0 が一番低い。床までは基礎
    floor = 100.0 + 1.2 + bd.BASE_H                                                          # 一番高い地面（x=12）＋基礎
    assert base[1][:, 2].max() == pytest.approx(floor) and wall[1][:, 2].min() == pytest.approx(floor)
    assert wall[1][:, 2].max() == pytest.approx(floor + 6.0 + bd.PARAPET_H)
    assert wall[3][:, 1].min() == pytest.approx(0.0)                                         # 壁の絵（窓）は床から上だけ


def test_storeys_map_to_uv_rows():
    ms = bd.building_meshes([_b(storeys=3, height_m=9.0)], FLAT)
    uv = _mesh(ms, "1WALL_bld_t1")[3]
    assert uv[:, 1].max() == pytest.approx(3.0 + bd.PARAPET_H / 3.0, abs=0.01)


def test_flat_roof_has_parapet():
    ms = bd.building_meshes([_b()], FLAT)
    v, f = _mesh(ms, "bld_t1_roof")[1:3]
    top = 106.0 + bd.BASE_H
    assert np.isclose(v[:, 2], top).any() and np.isclose(v[:, 2], top + bd.PARAPET_H).any()
    up = face_normals(v, f)[:, 2] > 0.99
    assert up.any()


def test_gable_ridge_follows_given_bearing():
    rect = [[0, 0], [10, 0], [10, 6], [0, 6]]
    ms = bd.building_meshes([_b(footprint=_ll(rect), roof={"type": "gable", "rise_m": 1.5, "ridge_deg": 0.0, "color": "#704030"})], FLAT)
    v = _mesh(ms, "bld_t1_roof")[1]
    top = v[np.isclose(v[:, 2], v[:, 2].max())]
    assert np.ptp(top[:, 0]) < 0.01 and np.ptp(top[:, 1]) > 5.0   # 棟（一番高い線）が南北


def test_sign_on_given_edge_facing_out():
    rect = [[0, 0], [10, 0], [10, 6], [0, 6]]
    sign = {"text": "テスト", "bg": "#FFFFFF", "fg": "#000000", "edge": 0, "w_m": 4.0, "h_m": 1.0, "z_m": 3.0}
    ms = bd.building_meshes([_b(footprint=_ll(rect), signs=[sign])], FLAT)
    name, v, f, uv, mat, rend = _mesh(ms, "bld_t1_sign0")
    n = face_normals(v, f)
    assert (n[:, 1] < -0.99).all()                          # 辺 0（y=0、南の辺）の外＝南を向く
    assert v[:, 1].max() < 0.0 and mat == "sign_t1_0"


def test_overlap_with_road_is_clipped_and_reported():
    road = Polygon([(-5, -3), (20, -3), (20, 1.0), (-5, 1.0)])   # 南の辺に 1m かかる
    b = _b()
    ms = bd.building_meshes([b], FLAT, avoid=road)
    v = _mesh(ms, "1WALL_bld_t1")[1]
    assert v[:, 1].min() >= 1.0 - 1e-6
    rep = bd.clipped_report([b], road)
    assert rep and rep[0][0] == "t1" and rep[0][1] == pytest.approx(12.0, abs=0.5)


def test_materials_and_textures_for_each_building(tmp_path):
    b = _b(shopfront=[0], signs=[{"text": "店", "bg": "#FF0000", "fg": "#FFFFFF", "edge": 0, "w_m": 3, "h_m": 1, "z_m": 3}])
    mats = bd.write_textures([b], tmp_path)
    assert {bd.facade_mat(b), bd.roof_mat(b), "sign_t1_0", "shopfront"} <= set(mats)
    for rel in mats.values():
        assert (tmp_path / rel.split("/")[-1]).exists()


def test_canopy_only_building_has_slab_and_columns_but_no_walls():
    rect = [[0, 0], [16, 0], [16, 9], [0, 9]]
    ms = bd.building_meshes([_b(footprint=_ll(rect), canopy={"self": True, "z_m": 4.8})], FLAT)
    names = [m[0] for m in ms]
    assert "1WALL_bld_t1" not in names and "bld_t1_canopy" in names and "1WALL_bld_t1_columns" in names
    v = _mesh(ms, "bld_t1_canopy")[1]
    assert v[:, 2].min() == pytest.approx(104.8) and v[:, 2].max() == pytest.approx(104.8 + bd.CANOPY_T)


def test_edge_facing_picks_the_side_towards_a_bearing():
    rect = [[0, 0], [10, 0], [10, 6], [0, 6]]
    b = _b(footprint=_ll(rect))
    assert bd.edge_facing(b, 180.0) == 0 and bd.edge_facing(b, 90.0) == 1 and bd.edge_facing(b, 0.0) == 2


def test_same_colours_share_materials_and_meshes_merge_by_material():
    a = _b(id="a"); b = _b(id="b", footprint=_ll([[30, 0], [40, 0], [40, 8], [30, 8]]))
    assert bd.facade_mat(a) == bd.facade_mat(b) and bd.roof_mat(a) == bd.roof_mat(b)
    ms = bd.merge_by_material(bd.building_meshes([a, b], FLAT))
    walls = [m for m in ms if m[0].startswith("1WALL_bld_") and m[4] != bd.BASE_MAT]
    assert len(walls) == 1 and walls[0][4] == bd.facade_mat(a)
    assert len({m[0] for m in ms}) == len(ms)


def test_merge_skips_empty_parts():
    v, f, uv = bd._merge([(np.zeros((0, 3)), np.zeros((0,)), np.zeros((0, 2))),
                          (np.zeros((3, 3)), np.array([[0, 1, 2]]), np.zeros((3, 2)))])
    assert f.shape == (1, 4) and len(v) == 3


def test_buildings_marked_gone_are_not_built(tmp_path):
    """地図（地理院）にはあるが、もう無い建物（S5 6232〜6367m の広場）：gone=true は作らない。checked なので track_scenery でも戻らない"""
    import json
    from yabitsu import buildings as bd
    fp = [[35.48, 139.21], [35.48, 139.2101], [35.4801, 139.2101], [35.4801, 139.21], [35.48, 139.21]]
    p = tmp_path / "b.json"
    p.write_text(json.dumps([{"id": "a", "footprint": fp, "checked": True}, {"id": "b", "footprint": fp, "checked": True, "gone": True}]), encoding="utf-8")
    assert [b.id for b in bd.load(p)] == ["a"]


def test_facade_texture_has_cladding_and_varied_windows():
    """壁の絵：写真の素材の下地に壁の色、4 区画で窓の種類が混ざる（全部同じ窓の等間隔はやめた。名古木 2026-10-28）。"""
    from yabitsu import facades
    widths = []
    for seed in range(6):
        a = np.asarray(facades.facade_image("#EDEBE6", True, "siding", seed=seed), float)
        assert a.shape[1] == int(facades.FACADE_BAYS * facades.BAY_W * facades.PX_PER_M)
        win = (a.mean(-1) < 200).mean(0) > 0.1                   # 列ごとに窓（ガラスとサッシ）があるか
        widths += list(np.diff(np.flatnonzero(np.diff(np.r_[0, win.astype(int), 0])))[::2])
    assert len(widths) >= 12 and np.ptp(widths) > 40             # 大きな窓・小窓・2 連が混ざる
    plain = np.asarray(facades.facade_image("#EDEBE6", False, "siding"), float)
    assert abs(plain.mean() - 0xE0) < 25 and plain[..., 0].std() > 3   # 色は壁の色のまま、下地のむらがある


def test_dark_reddish_wall_becomes_tin_and_cladding_can_be_given():
    from yabitsu import facades
    assert facades.cladding_of(_b(wall="#7A4A3A")) == "tin"
    assert facades.cladding_of(_b(cladding="plaster", wall="#7A4A3A")) == "plaster"
    assert bd.facade_mat(_b(cladding="tin")) != bd.facade_mat(_b(cladding="siding"))


def test_fill_gaps_adds_only_buildings_not_already_there():
    from yabitsu import footprints
    sq = lambda x, y, s=10: Polygon([(x, y), (x + s, y), (x + s, y + s), (x, y + s)])
    base = [(sq(0, 0), "gsi", {})]
    out = footprints.fill_gaps(base, [(sq(1, 1), {"_id": "a"}), (sq(30, 0), {"_id": "b"}), (sq(9, 9), {"_id": "c"})])
    assert [(s, t.get("_id")) for _, s, t in out] == [("gsi", None), ("ov", "b"), ("ov", "c")]   # 重なる a は捨て、角が少しかかる c は足す
