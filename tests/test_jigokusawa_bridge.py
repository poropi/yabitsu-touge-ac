"""地獄沢橋（S4 4745〜4753）の部品：コンクリートの欄干・沢の岩・白い流れ・材質の絵。"""
import numpy as np
import pytest

from yabitsu import geo, town_objects as to


def _ll(x, y):
    lat, lon = geo.from_local(float(x), float(y))
    return [float(lat), float(lon)]


FLAT = lambda x, y: np.full(np.shape(x), 100.0)


def test_concrete_rail_has_cap_rails_end_posts_fascia_and_hidden_collision():
    ms = to.object_meshes([{"kind": "concrete_rail", "height_m": 0.8, "post_every_m": 1.7, "fascia_m": 0.7, "outside": 1.0,
                            "points": [_ll(0, 0), _ll(7, 0)]}], FLAT, prefix="o")
    by = {m[0]: m for m in ms}
    v = by["o_concrete_moss"][1]
    assert v[:, 2].max() == pytest.approx(100.92, abs=0.01)              # 親柱は笠木（0.8 m）より 0.12 m 高い
    assert v[:, 2].min() == pytest.approx(99.3, abs=0.01)                # 橋の側面は路面から 0.7 m 下まで
    side = v[np.isclose(v[:, 2], 99.3)]
    assert np.allclose(side[:, 1], 0.12, atol=0.01)                      # 側面は外側（線の進む向きの左＝北）に出る
    tops = np.unique(np.round(v[:, 2], 2))
    assert {100.25, 100.35, 100.5, 100.6, 100.62, 100.8} <= set(tops)    # 横木 0.3・0.55 と笠木 0.62〜0.8
    assert by["1WALL_o_woodfence"][5] is False


def test_steel_rail_stands_on_a_concrete_curb_with_two_rails():
    """札掛橋の鋼の欄干：地覆の上に柱と横木 2 段（ストリートビュー 札掛橋 A045・A225）。"""
    ms = to.object_meshes([{"kind": "steel_rail", "rails": [0.6, 1.05], "curb_h": 0.25, "points": [_ll(0, 0), _ll(8, 0)]}], FLAT, prefix="o")
    by = {m[0]: m for m in ms}
    curb, steel = by["o_concrete"][1], by["o_steel_galv"][1]
    assert curb[:, 2].max() == pytest.approx(100.25) and curb[:, 2].min() == pytest.approx(99.4)
    assert steel[:, 2].max() == pytest.approx(101.1, abs=0.01) and steel[:, 2].min() == pytest.approx(100.25)   # 柱は地覆の上から
    assert len(np.unique(np.round(steel[:, 0], 2))) >= 2 * 5                                                     # 2 m おきの柱 5 本
    assert by["1WALL_o_woodfence"][5] is False


def test_steel_rail_can_draw_round_pipes_and_inner_bars():
    """塩水橋：丸い管 3 本と、内側の縦格子（ストリートビュー 塩水橋 A045・A135・A225）。"""
    ms = to.object_meshes([{"kind": "steel_rail", "rails": [0.55, 0.85, 1.1], "pipe_r": 0.05, "bars": {"every_m": 0.15, "z0": 0.3, "z1": 0.8},
                            "points": [_ll(0, 0), _ll(6, 0)]}], FLAT, prefix="o")
    steel = {m[0]: m for m in ms}["o_steel_galv"][1]
    assert steel[:, 2].max() == pytest.approx(101.15, abs=0.01)                    # 柱の頭（最上段の管 1.1 m ＋ 0.05）
    near_top = steel[np.isclose(steel[:, 2], 101.1, atol=0.051)]
    assert np.ptp(near_top[:, 1]) == pytest.approx(0.1, abs=0.01)                 # 管の太さ（半径 0.05）
    bars = steel[np.isclose(steel[:, 2], 100.8) | np.isclose(steel[:, 2], 100.3)]
    assert len(np.unique(np.round(bars[:, 0], 2))) >= 30                          # 0.15 m おきの縦の棒


def test_bridge_rail_none_keeps_posts_and_girder_but_no_rail_panels():
    from yabitsu import bridge_parts
    from tests.test_meshgen import _straight
    sec = _straight()
    ms = bridge_parts.meshes(sec, [{"osm_id": 7, "name": "塩水橋", "rail": "none", "post_material": "concrete"}], {7: (20.0, 40.0)}, FLAT)
    names = {m[0]: m[4] for m in ms}
    assert not any("_rail_" in n for n in names)
    assert names["bridge_7_post_aL"] == "concrete" and "bridge_7_girder" in names


def test_rock_sits_partly_buried_with_its_size():
    ms = to.object_meshes([{"kind": "rock", "sizes": [[1.2, 0.8], [0.5, 0.3]], "sink": 0.25, "points": [_ll(0, 0), _ll(5, 0)]}], FLAT, prefix="o")
    (name, v, f, uv, mat, rend), = ms
    assert mat == "rock_moss" and rend
    big = v[v[:, 0] < 2.5]
    assert big[:, 2].min() == pytest.approx(100 - 0.2, abs=0.06) and big[:, 2].max() == pytest.approx(100.6, abs=0.06)
    assert np.ptp(big[:, 0]) < 1.2 * 1.2 and np.ptp(big[:, 0]) > 0.5


def test_cascade_lays_white_foam_downstream_and_a_lip():
    ms = to.object_meshes([{"kind": "cascade", "down_deg": 90.0, "foam_m": 1.5, "lip_m": 0.3, "water_m": 0.3,
                            "points": [_ll(0, -1), _ll(0, 1)]}], FLAT, prefix="o")
    (name, v, f, uv, mat, rend), = ms
    assert mat == "water_fall"
    assert v[:, 0].max() == pytest.approx(1.5, abs=0.01)                 # 下流（東）へ 1.5 m
    assert v[:, 2].max() == pytest.approx(100.35, abs=0.01) and v[:, 2].min() == pytest.approx(100.0, abs=0.01)
    (_, v2, *_), = to.object_meshes([{"kind": "cascade", "down_deg": 90.0, "z_abs": 97.5, "lip_m": 0.0, "points": [_ll(0, -1), _ll(0, 1)]}], FLAT, prefix="o")
    assert v2[:, 2].max() == pytest.approx(97.54, abs=0.01)              # 岸の地面ではなく、渡した水面の高さ


def test_moss_textures_are_dark_grey_with_green():
    from yabitsu import textures as tx
    for fn in (tx._concrete_moss, tx._rock_moss):
        a = np.asarray(fn(128, 128), float)
        assert 60 < a.mean() < 150
        assert (a[..., 1] > a[..., 0] + 10).mean() > 0.05               # 苔（緑が赤より強い画素）がある


def test_tunnel_portal_faces_out_once_with_rim_and_style_materials():
    """坑口：表は外向きの 1 枚（表裏を重ねると縞に見えた）、アーチの縁は手前に RIM_D 出る、材質は PORTAL_STYLES。"""
    from yabitsu import tunnel
    from yabitsu.meshgen import face_normals
    from tests.test_meshgen import _straight
    sec = _straight()
    t = tunnel.Tunnel(1, "x", 30.0, 60.0) if len(tunnel.Tunnel.__dataclass_fields__) == 4 else None
    t = t or tunnel.Tunnel(**{k: v for k, v in zip(tunnel.Tunnel.__dataclass_fields__, (1, "x", 30.0, 60.0))})
    for tag, s, sgn in (("a", 30.0, -1.0), ("b", 60.0, 1.0)):
        ms = {m[0]: m for m in tunnel._portal(sec, t, tag, s, sgn, FLAT, "大洞", "stone")}
        name, v, f, uv, mat, rend = ms[f"tunnel_1_portal_{tag}"]
        assert mat == "kenchi_moss"
        nr = face_normals(v, f); vert = abs(nr[:, 2]) < 0.5
        assert (nr[vert, 0] * sgn > 0.99).all()                       # 縦の面は全部、外（s0 の a は -x、s1 の b は +x）を向く
        rim = ms[f"tunnel_1_rim_{tag}"]
        out = (rim[1][:, 0] - s) * sgn                                 # 外へ出た量（a は -x、b は +x が外）
        assert rim[4] == "concrete_moss" and out.min() >= -1e-6 and out.max() == pytest.approx(tunnel.RIM_D)


def test_kenchi_texture_has_dark_joints_and_moss():
    from yabitsu import textures as tx
    a = np.asarray(tx._kenchi_moss(128, 128), float)
    assert 50 < a.mean() < 140 and (a[..., 1] > a[..., 0] + 8).mean() > 0.05 and (a.mean(-1) < 60).mean() > 0.02
