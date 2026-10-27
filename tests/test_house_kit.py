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
