"""部品の家（ゲーム用、yabitsu/house_kit.py）：軽さ・面の向き・当たり判定・絵。"""
import numpy as np
from shapely.geometry import Polygon

from yabitsu import house_kit as hk
from yabitsu.meshgen import face_normals

FLAT = lambda x, y: np.full(np.shape(x), 100.0)
ROAD = np.array([[x, -20.0] for x in np.linspace(-50, 50, 21)])     # 家の南 20 m を東西に通る道
FP = Polygon([(-5, -3.5), (5, -3.5), (5, 3.5), (-5, 3.5)])


def _tris(items):
    return sum(len(f) for _, v, f, uv, m, r in items) * 2      # 四角 1 つ＝三角 2 つ（三角は最後の頂点を重ねた四角）


def test_near_house_is_light_enough_and_far_house_lighter():
    near, _ = hk.house_parts("a1", FP, FLAT, ROAD, near=True)
    far, _ = hk.house_parts("a1", FP, FLAT, ROAD, near=False)
    assert _tris(near) < 2500 and _tris(far) < _tris(near) * 0.75     # Blender の試作は 1 軒 9,800（窓の枠・格子が箱の積み重ね）
    assert _tris(far) < 1200


def test_front_faces_the_road_and_walls_face_out():
    items, pal = hk.house_parts("b2", FP, FLAT, ROAD, near=True)
    walls = [it for it in items if it[0].startswith("1WALL_")]
    assert len(walls) == 1 and walls[0][4] == pal["wall"]
    v, f = walls[0][1], walls[0][2]
    n = face_normals(v, f); ctr = v[f].mean(1)
    side = np.abs(n[:, 2]) < 0.5
    out = (n[side, :2] * ctr[side, :2]).sum(1)                       # 家の中心（原点）から外へ
    assert (out > 0).mean() > 0.95
    win = next(it for it in items if it[4] == pal["win"])
    wc = win[1][win[2]].mean(1)
    assert (wc[:, 1] < 0).sum() > 0 and wc[:, 1].min() < -3.4        # 窓は南（道の側）の面にもある


def test_floor_sits_on_highest_ground_with_base_below():
    slope = lambda x, y: 100.0 + 0.2 * np.asarray(x, float)
    items, pal = hk.house_parts("c3", FP, slope, ROAD, near=False)
    base = next(it for it in items if it[4] == "hk_base")[1]
    assert base[:, 2].min() <= 99.0 - 0.3 + 1e-6 and base[:, 2].max() >= 101.0 + hk.FL - 1e-6


def test_window_atlas_has_eight_kinds(tmp_path):
    im = hk.window_atlas((0.7, 0.72, 0.74))
    assert im.size == (hk.TILE * 4, hk.TILE * 2) and len(hk.WIN_KINDS) == 8
    a = np.asarray(im, float)
    door = a[:hk.TILE, hk.TILE: 2 * hk.TILE] if hk.WIN_KINDS.index("door") == 5 else None
    assert a.std() > 20
    _, pal = hk.house_parts("d4", FP, FLAT, ROAD)
    mats = hk.write_textures([pal], tmp_path)
    assert {pal["wall"], pal["roof"], "hk_win_silver", "hk_base"} <= set(mats)


def test_nagoki_town_draws_no_barricades_but_keeps_invisible_walls():
    """名古木の交差点：封鎖は見えない壁だけ（柵とコーンがストリートビューと全く違って見えた）。"""
    import json
    from yabitsu import config
    assert json.loads((config.DATA_MANUAL / "s3_nagoki.json").read_text(encoding="utf-8"))["barricades"] is False


def test_long_footprint_is_split_into_several_houses():
    long_fp = Polygon([(-15, -4), (15, -4), (15, 4), (-15, 4)])
    parts = hk._split_long(long_fp)
    lens = sorted(p.bounds[2] - p.bounds[0] for p in parts)
    assert len(parts) == 3 and np.allclose(lens, 9.0, atol=0.01)        # 30 m を 10 m ずつ、間を 1 m 空ける
    assert hk._split_long(FP) == [FP]


def test_window_glass_is_not_black_in_game_light():
    a = np.asarray(hk._window_tile("koshi", (0.7, 0.72, 0.74), np.random.default_rng(0)), float)
    assert a[40:-40, 40:-40].mean() > 70        # 暗すぎるとゲームで黒い穴に見えた


def test_convenience_store_has_glass_front_band_and_name(tmp_path):
    items, pal = hk.store_parts("s1", FP, FLAT, ROAD, name="セブン-イレブン")
    mats = {it[4] for it in items}
    assert {"hk_store_glass", "hk_store_band", "hk_store_name", "hk_store_pillar"} <= mats
    glass = next(it for it in items if it[4] == "hk_store_glass")
    assert glass[1][:, 1].mean() < -3.0                                  # ガラスは道（南）の側
    n = face_normals(glass[1], glass[2]); assert (n[:, 1] < -0.9).all()  # 外（南）を向く
    for m in ("hk_store_band", "hk_store_name"):
        it = next(x for x in items if x[4] == m); n = face_normals(it[1], it[2]); c = it[1][it[2][:, :3]].mean(1)
        assert ((n[:, :2] * c[:, :2]).sum(1) > 0).all()                   # 帯と店名の板は外を向く
    out = hk.write_textures([pal], tmp_path)
    assert {"hk_store_glass", "hk_store_band", "hk_store_name"} <= set(out)


def test_split_covers_the_whole_course_not_only_near_nagoki():
    """部品の家は名古木〜1.7 km だけにしていて、その先が箱のままだった（2026-10-28）。コースのどこでも、勾配屋根の家は部品の家。"""
    from types import SimpleNamespace
    from yabitsu.buildings import Building
    sec = SimpleNamespace(x=np.linspace(0, 20000, 2001), y=np.zeros(2001), s=np.linspace(0, 20000, 2001))
    far = Building("f1", Polygon([(15000, 30), (15010, 30), (15010, 37), (15000, 37)]), 2, 6.0, {"type": "gable"}, "#DDDDDD")
    flat = Building("f2", Polygon([(9000, 30), (9030, 30), (9030, 50), (9000, 50)]), 3, 9.0, {"type": "flat"}, "#ddd")
    gs = Building("f3", Polygon([(5000, 30), (5020, 30), (5020, 45), (5000, 45)]), 1, 5.0, {"type": "flat"}, "#ddd", canopy={"self": True, "z_m": 4.5})
    kit, rest, line, road = hk.split([far, flat, gs], sec)
    assert kit == [far, flat] and rest == [gs]          # 平屋根も部品で。屋根だけの建物（ガソリンスタンド）は今までどおり


def _bld(bid, fp, storeys, h, roof="flat", **kw):
    from yabitsu.buildings import Building
    return Building(bid, fp, storeys, h, {"type": roof}, "#E6E1D8", **kw)


def test_flat_roof_buildings_are_built_from_parts_with_windows_and_parapet(tmp_path):
    """平屋根のアパート・工場・物置・店も部品で建てる（箱のままだった）。"""
    apt = _bld("apt", Polygon([(-10, -4), (10, -4), (10, 4), (-10, 4)]), 2, 6.0)
    fac = _bld("fac", Polygon([(-20, -15), (20, -15), (20, 15), (-20, 15)]), 1, 6.0)
    shed = _bld("shed", Polygon([(-2, -2), (2, -2), (2, 3), (-2, 3)]), 1, 2.6)
    shop = _bld("shop", Polygon([(-6, -5), (6, -5), (6, 5), (-6, 5)]), 1, 4.5,
                signs=[{"text": "食堂", "bg": "#FFFFFF", "fg": "#000000", "w_m": 3.0, "h_m": 0.8, "z_m": 3.2, "edge": 0}])
    assert [hk.flat_kind(b) for b in (apt, fac, shed)] == ["block", "factory", "shed"]
    items, pals = hk.house_meshes([apt, fac, shed, shop], FLAT, ROAD, near_line=None)
    mats = {it[4] for it in items}
    assert {"hk_shutter", "hk_shop_glass", "sign_shop_0"} <= mats
    assert any(it[4] in ("hk_win_silver", "hk_win_bronze") and it[0].startswith("hk_apt") for it in items)   # アパートに窓
    apt_v = np.concatenate([it[1] for it in items if it[0].startswith(("1WALL_hk_apt", "hk_apt"))])
    assert apt_v[:, 2].max() >= 100.4 + 6.0 + 0.5                                                     # 2 階＋パラペット
    w = next(it for it in items if it[0].startswith("1WALL_hk_apt"))
    n = face_normals(w[1], w[2]); ctr = w[1][w[2]].mean(1); side = (np.abs(n[:, 2]) < 0.5) & (ctr[:, 2] < 100.4 + 6.0)   # パラペットの内側は除く
    assert ((n[side, :2] * ctr[side, :2]).sum(1) > 0).mean() > 0.95                                  # 壁は外を向く
    out = hk.write_textures(pals, tmp_path)
    assert {"hk_shutter", "hk_shop_glass", "hk_showroom", "sign_shop_0"} <= set(out)


def test_merged_meshes_stay_inside_one_cell():
    """まとめたメッシュが CELL_M の区画を越えない（コース全体に広がると視界の外でも描かれる）。"""
    fps = [Polygon([(x, 30), (x + 10, 30), (x + 10, 37), (x, 37)]) for x in np.arange(0, 3000, 25)]
    road = np.array([[x, 0.0] for x in np.linspace(0, 3000, 301)])
    items = []
    for k, fp in enumerate(fps):
        items += hk.house_parts(f"h{k}", fp, FLAT, road, near=False)[0]
    merged = hk.merge_by_cell(items)
    span = max(v[:, 0].max() - v[:, 0].min() for _, v, f, uv, m, r in merged)
    assert span < hk.CELL_M + 30 and len({n for n, *_ in merged}) == len(merged)


def test_house_whose_outline_reaches_the_road_keeps_its_walls_off_the_road():
    """輪郭が道にかかる家（S4 2806 m の vm695_1380）：部品の家の壁が道に出て AI の車が止まった（2026-10-28）。"""
    from shapely.geometry import LineString
    from yabitsu.buildings import Building
    road_line = LineString([(-50, 0), (50, 0)]); road = road_line.buffer(4.0)          # 幅 8 m の道（＋1 m の余裕は keep_out 側）
    from shapely import affinity
    fp = affinity.rotate(Polygon([(-5, 2.5), (5, 2.5), (5, 11), (-5, 11)]), 30, origin=(0, 6.75))   # 道に斜めにかかる
    b = Building("r1", fp, 2, 6.0, {"type": "gable"}, "#DDDDDD")
    items, _ = hk.house_meshes([b], FLAT, np.c_[np.linspace(-50, 50, 51), np.zeros(51)], near_line=road_line, avoid=road.buffer(1.0))
    walls = np.concatenate([it[1] for it in items if it[0].startswith("1WALL_")])
    assert np.abs(walls[:, 1]).min() > 4.0                                               # 当たり判定の壁は道の外


def test_small_outbuildings_are_built_as_sheds_not_dropped():
    """8〜14 m² の物置・小屋が部品の家に替えたとき抜けていた（箱の家は 2 m² 以上を全部描いていた）。"""
    from yabitsu.buildings import Building
    b = Building("sm1", Polygon([(0, 10), (3, 10), (3, 13), (0, 13)]), 1, 2.6, {"type": "gable"}, "#DDDDDD")
    items, pals = hk.house_meshes([b], FLAT, ROAD, avoid=Polygon([(-50, -30), (50, -30), (50, -10), (-50, -10)]))
    assert any(it[0].startswith("1WALL_") for it in items) and pals[0]["clad"] == "tin"


def test_large_multi_storey_flat_building_is_not_a_factory():
    """3 階建て 948 m² の宿泊施設（札掛森の家）が工場の作りになっていた。"""
    lodge = _bld("lodge", Polygon([(-28, -12), (28, -12), (28, 12), (-28, 12)]), 3, 9.0, name="札掛森の家")
    big = _bld("ware", Polygon([(-30, -30), (30, -30), (30, 30), (-30, 30)]), 2, 6.0)
    assert hk.flat_kind(lodge) == "block" and hk.flat_kind(big) == "factory"


def test_hand_made_buildings_and_nanohanadai_tower_are_not_turned_into_houses():
    """菜の花台展望台が 2 階建ての家に、峠の売店・神社・古民家がありふれた家になっていた（2026-10-28）。"""
    import json
    from yabitsu import buildings, config
    from yabitsu.crosssection import Section
    raw = {x["id"]: x for x in json.loads((config.DATA_MANUAL / "track_buildings.json").read_text(encoding="utf-8"))}
    assert raw["osm952587264"]["gone"] is True                                   # 展望台は s1_start_lot の tower だけ
    B = buildings.load(config.DATA_MANUAL / "track_buildings.json")
    assert all(b.id != "osm952587264" for b in B)
    sec = Section.load_json(str(config.DATA_PROCESSED / "track_section.json"))
    kit, rest, _, _ = hk.split(B, sec)
    keep = {"manual_pass_shop_main", "osm184602558", "osm952810353", "osm369868058", "osm371207678"}
    assert keep <= {b.id for b in rest} and not keep & {b.id for b in kit}


def test_kit_house_uses_the_wall_colour_and_cladding_from_the_data():
    from yabitsu.buildings import Building
    b = Building("c9", FP, 2, 6.0, {"type": "gable", "color": "#5A3C2C"}, "#5A4636", cladding="siding")
    _, pals = hk.house_meshes([b], FLAT, ROAD)
    assert pals[0]["wall_hex"] == "#5A4636" and pals[0]["clad"] == "siding" and pals[0]["roof_hex"] == "#5A3C2C"
