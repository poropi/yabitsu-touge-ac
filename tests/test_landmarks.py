"""Codex の部品を場所の座標に置く（yabitsu/landmarks.py）と、駐車場で替えた簡単な形を描かない（start_lot の replaced）。"""
import numpy as np

from yabitsu import landmarks as lm
from yabitsu import start_lot as sl
from tests.test_start_lot import SPEC, _pad


def _fake(monkeypatch, v, uv=None):
    """npz の代わりに、正面 −Y・+X 右の小さな部品を返す。"""
    f = np.array([[0, 1, 2]])
    uv = np.array([[0, 0], [1, 0], [0, 1]], float) if uv is None else uv
    monkeypatch.setattr(lm, "exists", lambda a, n: True)
    monkeypatch.setattr(lm, "load", lambda a, n, uv_top=True: (np.asarray(v, float), f, uv.copy()))


def test_yaw_zero_puts_front_to_minus_b_and_x_to_plus_a(monkeypatch):
    _fake(monkeypatch, [[1, 0, 0], [0, -1, 0], [0, 0, 2]])
    fr = sl.Frame([0.0, 0.0], 30.0)
    (_, v, *_), = lm.place([{"asset": "nanohanadai_x", "name": "n", "a": 5.0, "b": 2.0}], fr, 100.0, "p")
    a, b = fr.ab(v[:, :2])
    assert np.allclose([a[0], b[0]], [6.0, 2.0])      # +X → +a
    assert np.allclose([a[1], b[1]], [5.0, 1.0])      # −Y（正面）→ −b
    assert np.isclose(v[2, 2], 102.0)


def test_positive_yaw_turns_from_plus_a_to_plus_b(monkeypatch):
    _fake(monkeypatch, [[1, 0, 0], [0, -1, 0], [0, 0, 1]])
    fr = sl.Frame([0.0, 0.0], 0.0)
    (_, v, *_), = lm.place([{"asset": "nanohanadai_x", "name": "n", "a": 0.0, "b": 0.0, "yaw_deg": 90.0}], fr, 0.0, "p")
    assert np.allclose(v[0, :2], [0.0, 1.0], atol=1e-9)   # +X → +b
    assert np.allclose(v[1, :2], [1.0, 0.0], atol=1e-9)   # 正面 → +a


def test_absolute_z_overrides_place_height_and_adds_dz(monkeypatch):
    _fake(monkeypatch, [[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    fr = sl.Frame([0.0, 0.0], 0.0)
    out = lm.place([{"asset": "a", "name": "n", "a": 0.0, "b": 0.0, "z": 562.5, "dz": 0.3},
                    {"asset": "a", "name": "n", "a": 0.0, "b": 0.0}], fr, lambda a, b: 100.0, "p")
    assert np.allclose(out[0][1][:, 2], 562.8)
    assert np.allclose(out[1][1][:, 2], 100.0)


def test_load_flips_codex_uv_to_course_convention(tmp_path, monkeypatch):
    (tmp_path / "a").mkdir()
    np.savez(tmp_path / "a" / "n.npz", v=np.zeros((3, 3)), f=np.array([[0, 1, 2]]), uv=np.array([[0, 0.2], [1, 0.2], [0, 1.0]]))
    monkeypatch.setattr(lm, "DIR", tmp_path)
    assert np.allclose(lm.load("a", "n")[2][:, 1], [0.8, 0.8, 0.0])   # Codex の uv（v=0 が上）→ このコース（v=0 が下）
    assert np.allclose(lm.load("a", "n", uv_top=False)[2][:, 1], [0.2, 0.2, 1.0])


def test_collide_boxes_are_hidden_walls(monkeypatch):
    _fake(monkeypatch, [[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    fr = sl.Frame([0.0, 0.0], 0.0)
    out = lm.place([{"asset": "nanohanadai_toilet", "name": "walls", "a": 0.0, "b": 0.0,
                     "collide": [[[-1, 1], [-2, 2], 3.0]]}], fr, 10.0, "s1_lm")
    (n0, v0, f0, uv0, mat0, ren0), (n1, v1, f1, _, _, ren1) = out
    assert mat0 == "lm_nh_toilet_walls" and ren0
    assert n1.startswith("1WALL_") and not ren1
    assert np.isclose(v1[:, 2].max(), 13.0) and np.isclose(v1[:, 2].min(), 9.7)
    assert f1.max() < len(v1)


def test_missing_asset_is_skipped(monkeypatch):
    monkeypatch.setattr(lm, "exists", lambda a, n: False)
    assert lm.place([{"asset": "x", "name": "n", "a": 0, "b": 0}], sl.Frame([0, 0], 0), 0.0, "p") == []
    assert lm.materials([{"asset": "x", "name": "n"}]) == {}


def test_replaced_parts_are_not_drawn_but_building_keeps_collision():
    sec, pad = _pad()
    base = {n: r for n, *_, r in sl.meshes(SPEC, pad)}
    spec = {**SPEC, "replaced": ["monument", "building", "blocks"]}
    out = {n: r for n, *_, r in sl.meshes(spec, pad)}
    assert base["1WALL_s1_building"] and not out["1WALL_s1_building"]
    assert not out["1WALL_s1_lot_blocks"]
    assert "1WALL_s1_monument" in base and "1WALL_s1_monument" not in out
    assert "s1_building_roof" in base and "s1_building_roof" not in out


def test_paved_rect_is_a_road_surface_at_lot_level():
    sec, pad = _pad()
    spec = {**SPEC, "paved": [{"a": [-50.0, -42.0], "b": [-3.0, 1.0]}]}
    out = {n: (v, mat) for n, v, f, uv, mat, r in sl.meshes(spec, pad)}
    v, mat = out["1ROAD_s1_paved_0"]
    a, b = sl.frame(spec).ab(v[:, :2])
    assert mat == "tiles" and np.allclose(v[:, 2], pad["z"])
    assert np.isclose(a.min(), -50.0) and np.isclose(b.max(), 1.0)


def test_lot_extra_polygons_widen_the_lot():
    sec, pad = _pad()
    base = sl.lot_polygon(SPEC, sec).area
    spec = {**SPEC, "lot_extra": [[(-30.0, 8.0), (-10.0, 8.0), (-10.0, 14.0), (-30.0, 14.0)]]}
    assert np.isclose(sl.lot_polygon(spec, sec).area - base, 20.0 * 5.0, rtol=0.02)   # b 9〜14 の 20 m 分だけ広がる


def test_fence_return_runs_up_the_slope_side_at_the_sw_end():
    sec, pad = _pad()
    spec = {**SPEC, "fence": {**SPEC["fence"], "return_b": 2.15}}
    out = {n: (v, r) for n, v, f, uv, mat, r in sl.meshes(spec, pad)}
    v, _ = out["s1_lot_fence_return"]
    a, b = sl.frame(spec).ab(v[:, :2])
    assert np.allclose(a, SPEC["fence"]["a"][0], atol=0.05)
    assert np.isclose(b.min(), SPEC["fence"]["b"], atol=0.05) and np.isclose(b.max(), 2.15)
    assert not out["1WALL_s1_lot_fence_return"][1]


def test_white_wall_steps_down_from_h0_to_h1():
    sec, pad = _pad()
    spec = {**SPEC, "white_walls": [{"points": [[-20.0, 7.0], [-10.0, 7.0]], "h": [0.7, 0.06]}]}
    v = {n: v for n, v, *_ in sl.meshes(spec, pad)}["1WALL_s1_white_wall_0"]
    a, _ = sl.frame(spec).ab(v[:, :2])
    top_w = v[a < -19.5, 2].max() - pad["z"]; top_e = v[a > -10.1, 2].max() - pad["z"]
    assert 0.6 < top_w <= 0.7 and np.isclose(top_e, 0.06)


def test_flats_keep_half_a_metre_off_the_lot():
    from shapely.geometry import Polygon
    sec, pad = _pad()
    spec = {**SPEC, "path": {"points": [[-40.0, 6.0], [-60.0, 6.0]], "width": 2.0}}
    lot = Polygon(pad["polygon"])
    for fl in sl.flat_pads(spec, pad):
        if fl["name"] == "flat":
            assert Polygon(fl["polygon"]).distance(lot) >= 0.49


def test_replaced_tower_is_not_built():
    sec, pad = _pad()
    spec = {**SPEC, "tower": {"a": -60.0, "b": 20.0, "radius": 2.6, "deck_h": 6.0, "post_top": 8.4, "roof_h": 2.0,
                              "eaves": 0.8, "post": 0.25, "rail_h": 1.1, "plaza_r": 11.0}}
    base = {n for n, *_ in sl.meshes(spec, pad)}
    out = {n for n, *_ in sl.meshes({**spec, "replaced": ["tower"]}, pad)}
    tower = {n for n in base if "tower" in n}
    assert tower and not (tower & out)
    assert any(p["name"] == "flat" for p in sl.flat_pads({**spec, "replaced": ["tower"]}, pad))   # 広場の平らな地面は残る


def test_paved_and_flats_take_height_and_ramp():
    sec, pad = _pad()
    spec = {**SPEC, "replaced": ["tower"],
            "paved": [{"points": [[-70, 0], [-60, 0], [-60, 3], [-70, 3]], "ramp": {"a": [-70.0, -60.0], "dz": [0.7, 0.0]}},
                      {"points": [[-80, 5], [-72, 5], [-72, 9], [-80, 9]], "dz": 0.72, "material": "gravel"}],
            "flats": [{"points": [[-80, 5], [-72, 5], [-72, 9], [-80, 9]], "dz": 0.72, "sink": True},
                      {"points": [[-70, 0], [-60, 0], [-60, 3], [-70, 3]], "ramp": {"a": [-70.0, -60.0], "dz": [0.7, 0.0]}}]}
    out = {n: (v, mat) for n, v, f, uv, mat, r in sl.meshes(spec, pad)}
    f = sl.frame(spec)
    v, _ = out["1ROAD_s1_paved_0"]; a, _ = f.ab(v[:, :2])
    assert np.allclose(v[:, 2] - pad["z"], np.interp(a, [-70, -60], [0.7, 0.0]))
    v, mat = out["1ROAD_s1_paved_1"]
    assert mat == "gravel" and np.allclose(v[:, 2], pad["z"] + 0.72)
    fl = [p for p in sl.flat_pads(spec, pad) if p["name"] == "flat"]
    mound = [p for p in fl if p.get("kind") == "pass"][0]
    assert np.isclose(mound["z"], pad["z"] + 0.25 + 0.72)
    ramp = [p for p in fl if "plane" in p][0]
    cx, cy, z0, gx, gy = ramp["plane"]
    x, y = f.xy(-60.0, 1.0)
    assert np.isclose(z0 + gx * (x - cx) + gy * (y - cy), pad["z"] + 0.25, atol=1e-6)


def test_nanohanadai_area_reads_polygon_flats_and_paved():
    """OSM の小道を重ねない範囲は、多角形の flats・paved（points）も読む（2026-10-30 に長方形しか読めず組み立てが止まった）。"""
    from scripts.track_build_manifest import _nanohanadai_area
    from yabitsu import config
    from yabitsu.crosssection import Section
    p = config.DATA_PROCESSED / "track_section.json"
    if not p.exists():
        import pytest
        pytest.skip("track_section.json がまだない")
    spec = sl.load_spec(config.DATA_MANUAL / "s1_start_lot.json")
    areas = _nanohanadai_area(Section.load_json(p))
    assert len(areas) >= 1 + len(spec.get("paved", [])) + len(spec.get("flats", []))


def test_place_ll_turns_front_to_bearing_and_takes_lowest_z_at(monkeypatch):
    """ヤビツ峠（s1_pass.json の landmarks）：正面（−Y）が bearing（北から時計回り）を向き、高さは z_at の点のいちばん低い所。"""
    from yabitsu import geo
    _fake(monkeypatch, [[1, 0, 0], [0, -1, 0], [0, 0, 1]])
    ll = [35.42795, 139.21925]
    c = np.array(geo.to_local(*ll), float)
    east = geo.from_local(c[0] + 3.0, c[1])
    out = lm.place_ll([{"asset": "pass_x", "name": "n", "ll": ll, "bearing": 90.0, "z_at": [ll, list(east)]}],
                      lambda x, y: 500.0 + np.asarray(x, float) - c[0], "s1_pass_lm")
    (_, v, *_), = out
    assert np.allclose(v[1, :2] - c, [1.0, 0.0], atol=1e-6)    # 正面 → 東（90°）
    assert np.allclose(v[0, :2] - c, [0.0, 1.0], atol=1e-6)    # +X → 北（正面に立って見た人の右）
    assert np.isclose(v[2, 2], 501.0)                          # z_at の低い方（500）＋部品の高さ 1


def test_replaced_pass_buildings_and_fence_are_not_drawn():
    """Codex の部品に替えた峠の棟と島の柵（replaced）は、箱と杭を作らない。"""
    import json
    from yabitsu import config, pass_area
    from yabitsu.crosssection import Section
    sec = Section.load_json(config.DATA_PROCESSED / "track_section.json")
    spec = json.loads((config.DATA_MANUAL / "s1_pass.json").read_text(encoding="utf-8"))
    names = lambda sp: {n for n, *_ in pass_area.build(sp, sec).meshes()}
    for b in spec["buildings"]:
        b.pop("replaced", None)
    spec["island_fence"].pop("replaced", None)
    before = names(spec)
    assert "1WALL_s1_pass_shelter" in before and "s1_pass_island_fence" in before
    for b in spec["buildings"]:
        b["replaced"] = "pass_toilet"
    spec["island_fence"]["replaced"] = "pass_road_items"
    after = names(spec)
    assert not any("shelter" in n for n in after) and "s1_pass_island_fence" not in after
    assert {p["name"] for p in pass_area.build(spec, sec).pads()} >= {"pass_building_0", "pass_building_1"}   # 地形を削る台は残す


def test_place_ll_uses_ground_for_slope_items(monkeypatch):
    """山の斜面の階段（z_from＝ground）は地形の高さ。歩道に寄せる高さ（z_fn）だと斜面に埋まった（2026-10-31）。"""
    _fake(monkeypatch, [[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    ll = [35.42795, 139.21925]
    out = lm.place_ll([{"asset": "pass_x", "name": "n", "ll": ll, "z_from": "ground"}, {"asset": "pass_x", "name": "n", "ll": ll}],
                      lambda x, y: np.full(np.shape(x), 760.0), "s1_pass_lm", ground_fn=lambda x, y: np.full(np.shape(x), 763.0))
    assert np.allclose(out[0][1][:, 2], 763.0) and np.allclose(out[1][1][:, 2], 760.0)


def test_copy_texture_caps_game_size_and_keeps_small(tmp_path):
    """ゲームへ写す部品の画像は 1 辺 1024 まで（kn5 は無圧縮で、2048 が 19 枚で約 0.5 GB 増えた）。小さい画像はそのまま。"""
    from PIL import Image
    Image.new("RGB", (2048, 1024), (200, 10, 10)).save(tmp_path / "big.png")
    Image.new("RGB", (512, 512), (10, 200, 10)).save(tmp_path / "small.png")
    lm.copy_texture(tmp_path / "big.png", tmp_path / "big_out.png")
    lm.copy_texture(tmp_path / "small.png", tmp_path / "small_out.png")
    assert Image.open(tmp_path / "big_out.png").size == (1024, 512)
    assert Image.open(tmp_path / "small_out.png").size == (512, 512)


def test_tex_picks_another_image_for_the_same_shape(monkeypatch, tmp_path):
    """県道 70 号の標識：同じ形（warning_diamond）に、面の絵の画像（warning_diamond_face_left.png）を選んで貼る。材質も画像ごとに分かれる。"""
    _fake(monkeypatch, [[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    (tmp_path / "r70_x").mkdir()
    for n in ("sign", "sign_face_left"):
        (tmp_path / "r70_x" / f"{n}.png").write_bytes(b"x")
    monkeypatch.setattr(lm, "DIR", tmp_path)
    items = [{"asset": "r70_x", "name": "sign", "ll": [35.45, 139.21]},
             {"asset": "r70_x", "name": "sign", "ll": [35.45, 139.21], "tex": "sign_face_left"}]
    out = lm.place_ll(items, lambda x, y: np.zeros(np.shape(x)), "r70")
    assert [o[4] for o in out] == ["lm_r70_x_sign", "lm_r70_x_sign_face_left"]
    mats = lm.materials(items)
    assert mats["lm_r70_x_sign_face_left"][0].name == "sign_face_left.png" and mats["lm_r70_x_sign"][0].name == "sign.png"


def test_place_ll_takes_written_z_over_z_fn(monkeypatch):
    """道の縁の高さで置く物（県道 70 号）は、書いた z をそのまま使う（中心線の高さの z_fn ではない）。"""
    _fake(monkeypatch, [[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    out = lm.place_ll([{"asset": "r70_x", "name": "n", "ll": [35.45, 139.21], "z": 412.3, "dz": 0.1}],
                      lambda x, y: np.full(np.shape(x), 400.0), "r70")
    assert np.allclose(out[0][1][:, 2], 412.4)
