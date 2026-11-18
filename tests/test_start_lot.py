import pytest
import math

import numpy as np
from shapely.geometry import Point, Polygon

from yabitsu import meshgen as mg
from yabitsu import layouts as lay
from yabitsu import geo
from yabitsu import start_lot as sl
from tests.test_meshgen import _straight

# 道は x=0 から +x へ。駐車場は道の始まりの南側から、+y（道の方）へ伸びる
SPEC = {
    "osm_name": "菜の花台休憩所", "origin": [0.0, -20.0], "axis_deg": 90.0,
    "lot": {"a": [-40.0, 25.0], "b": [-2.5, 9.0]},
    "stalls": {"a": [-36.0, -4.0], "b": [-2.5, 2.5], "pitch": 2.5},
    "fence": {"a": [-40.0, 18.0], "b": -2.6, "height": 1.1, "post_every": 2.0},
    "blocks": {"a": [-30.0, -3.0], "b": 9.3, "every": 3.0, "size": [0.9, 0.35, 0.35]},
    "monument": {"a": 10.0, "b": 11.0, "size": [1.4, 0.6, 2.2]},
    "building": {"a": [-45.0, -36.0], "b": [9.5, 15.0], "wall_h": 3.0, "roof_h": 1.6, "eaves": 0.6},
    "flats": [{"a": [-47.0, -34.0], "b": [9.0, 17.0]}],
    "pits": {"a": -15.0, "b": 3.5, "spacing": 2.8},
}


def _pad():
    sec = _straight(201)
    pads = [{"name": "菜の花台休憩所", "role": "start", "polygon": [[0, -30], [5, -30], [5, -25], [0, -25], [0, -30]], "z": 100.0}]
    return sec, sl.apply(pads, SPEC, sec)[0]


def test_frame_axes():
    f = sl.Frame([10.0, 20.0], 90.0)
    assert np.allclose(f.xy(5.0, 0.0), [10.0, 25.0]) and np.allclose(f.xy(0.0, 2.0), [8.0, 20.0])


def test_lot_replaces_start_pad_and_stays_off_the_road():
    sec, pad = _pad()
    poly = Polygon(pad["polygon"])
    assert pad["kind"] == "lot" and pad["z"] == 100.0
    assert poly.intersection(sl.road_formation(sec)).area < 1e-6
    assert abs(poly.area - 11.5 * 65.0) < 11.5 * 65.0 * 0.1  # 道と重なる角だけ削られる


def test_lot_surface_points_up_and_follows_lot_axis():
    sec, pad = _pad()
    meshes = {name: (v, f, uv, mat) for name, v, f, uv, mat, rend in sl.meshes(SPEC, pad)}
    v, f, uv, mat = meshes["1ROAD_s1_lot"]
    assert mat == "pavers" and (mg.face_normals(v, f)[:, 2] > 0).all() and np.allclose(v[:, 2], 100.0)
    # 長手方向（+y）に進むと u が増え、v は変わらない
    i, j = np.argmin(v[:, 1]), np.argmax(v[:, 1])
    assert uv[j, 0] > uv[i, 0]


def test_meshes_have_expected_parts():
    sec, pad = _pad()
    out = {name: (mat, rend) for name, v, f, uv, mat, rend in sl.meshes(SPEC, pad)}
    assert out["1WALL_s1_lot_fence"] == ("railing", False)
    for name in ("LINES_s1_lot_stalls", "s1_lot_fence", "s1_lot_fence_bars", "1WALL_s1_lot_blocks",
                 "1WALL_s1_monument", "1WALL_s1_building", "s1_building_roof"):
        assert out[name][1] is True, name


def test_boxes_face_outward():
    f = sl.Frame([0.0, 0.0], 30.0)
    v, faces, uv = sl.box(f, (-1.0, 1.0), (-2.0, 2.0), 0.0, 3.0)
    c = v.mean(0)
    n = mg.face_normals(v, faces)
    centres = v[faces].mean(1)
    assert len(faces) == 5 and ((centres - c) * n).sum(1).min() > 0


def test_stall_lines_count():
    sec, pad = _pad()
    v, f, uv, *_ = next(m[1:] for m in sl.meshes(SPEC, pad) if m[0] == "LINES_s1_lot_stalls")
    assert len(f) == int(32 / 2.5) + 1 + 1  # 区切り線と、通路側の長い線


def test_pits_inside_lot_and_face_the_exit():
    sec, pad = _pad()
    ll = lambda x, y: tuple(float(v) for v in geo.from_local(x, y))
    layout = lay.Layout("t", "t", "s1", 1, ll(40, 0), ll(5, 0), ll(50, 0), ll(190, 0), "nanohanadai_lot")
    pits = [d for d in lay.layout_objects(sec, [pad], layout, {"s1": (0.0, 200.0)}) if d.name.startswith("AC_PIT_")]
    poly = Polygon(pad["polygon"]).buffer(-1.0)
    assert len(pits) == 4 and all(poly.contains(Point(d.location[0], d.location[1])) for d in pits)
    assert all(abs(math.remainder(d.heading - math.pi / 2, 2 * math.pi)) < 1e-6 for d in pits)


def test_flats_and_clearings_are_at_lot_height():
    sec, pad = _pad()
    flats = sl.flat_pads(SPEC, pad)
    assert len(flats) == 1 and all(abs(p["z"] - pad["z"]) < 0.5 for p in flats)


TOWER_SPEC = {**SPEC,
              "tower": {"a": -80.0, "b": 15.0, "radius": 2.6, "deck_h": 6.0, "post_top": 8.4, "roof_h": 2.0,
                        "eaves": 0.8, "post": 0.25, "rail_h": 1.1, "plaza_r": 11.0},
              "path": {"points": [[-42.0, 6.5], [-60.0, 8.0], [-74.0, 13.0]], "width": 3.0},
              "bollards": {"a": -41.0, "b": [5.2, 6.5, 7.8], "height": 0.8}}


def _tower_meshes():
    sec, pad = _pad()
    return pad, {m[0]: m[1:] for m in sl.meshes(TOWER_SPEC, pad)}


def test_tower_parts_and_height():
    pad, ms = _tower_meshes()
    for name in ("s1_tower_frame", "s1_tower_deck", "s1_tower_rail", "s1_tower_roof", "1WALL_s1_tower", "1ROAD_s1_path",
                 "1WALL_s1_bollards"):
        assert name in ms, name
    roof = ms["s1_tower_roof"][0]
    assert abs(roof[:, 2].max() - (pad["z"] + 8.4 + 2.0)) < 1e-6
    centre = sl.frame(TOWER_SPEC).xy(-80.0, 15.0)
    assert np.hypot(*(roof[:, :2] - centre).T).max() <= 2.6 + 0.8 + 1e-6
    assert ms["1WALL_s1_tower"][4] is False  # 当たり判定だけ（描画しない）


def test_tower_roof_and_deck_face_up():
    pad, ms = _tower_meshes()
    for name in ("s1_tower_roof", "s1_tower_deck"):
        v, f = ms[name][0], ms[name][1]
        n = mg.face_normals(v, f)
        assert (n[:, 2] > 0).sum() >= len(f) * 0.5, name


def test_path_is_on_lot_level_and_points_up():
    pad, ms = _tower_meshes()
    v, f = ms["1ROAD_s1_path"][0], ms["1ROAD_s1_path"][1]
    assert np.allclose(v[:, 2], pad["z"]) and (mg.face_normals(v, f)[:, 2] > 0).all()


def test_bollard_gaps_too_narrow_for_a_car():
    b = TOWER_SPEC["bollards"]["b"]
    assert max(np.diff(b)) - 0.15 < 1.6


def test_flats_cover_plaza_and_path():
    sec, pad = _pad()
    polys = [Polygon(p["polygon"]) for p in sl.flat_pads(TOWER_SPEC, pad)]
    f = sl.frame(TOWER_SPEC)
    for a, b in [(-80.0, 15.0), (-80.0 + 9.0, 15.0), (-60.0, 8.0)]:
        assert any(p.contains(Point(*f.xy(a, b))) for p in polys), (a, b)


def test_octagon_prism_faces_outward():
    v, f, uv = sl._octagon_prism((5.0, 5.0), 2.0, 0.0, 3.0)
    n = mg.face_normals(v, f); out = v[f].mean(1)[:, :2] - [5.0, 5.0]
    assert len(f) == 8 and ((n[:, :2] * out).sum(1) > 0).all()


def test_apply_matches_by_osm_name_and_renames():
    sec = _straight(201)
    pads = [{"name": "other", "role": "start", "polygon": [[0, -30], [5, -30], [5, -25], [0, -25], [0, -30]], "z": 90.0},
            {"name": "菜の花台休憩所", "role": "end", "polygon": [[0, -30], [5, -30], [5, -25], [0, -25], [0, -30]], "z": 100.0}]
    out = sl.apply(pads, SPEC, sec)
    assert out[0]["name"] == "other" and "kind" not in out[0]
    assert out[1]["name"] == "nanohanadai_lot" and out[1]["kind"] == "lot" and out[1]["z"] == 100.0


def test_lot_in_middle_of_road_is_trimmed_by_both_branches():
    # 下の枝（y=-3 を +x、x=0..120）→ ヘアピン → 上の枝（y=10 を −x）。駐車場は x=100..125 の帯で両方の枝にかかる
    import numpy as np
    from tests.test_meshgen import _sec_from_xy
    x = np.r_[np.arange(0.0, 120.0), 120 + 6.5 * np.sin(np.linspace(0, np.pi, 21))[1:-1], np.arange(120.0, -1.0, -1.0)]
    y = np.r_[np.full(120, -3.0), 3.5 - 6.5 * np.cos(np.linspace(0, np.pi, 21))[1:-1], np.full(121, 10.0)]
    sec = _sec_from_xy(x, y)
    spec = {**SPEC, "origin": [125.0, -12.0], "axis_deg": 90.0, "lot": {"a": [0.0, 30.0], "b": [-2.5, 9.0]}}
    poly = sl.lot_polygon(spec, sec)
    road = sl.road_formation(sec, s_min=0.0, s_max=float(sec.s[-1]))
    assert poly.intersection(road).area < 0.5 and poly.area > 50


def test_lot_details_from_street_view():
    """ヘアピンの内側の東屋（屋根と柱）、北の階段、自然石の石碑と銘板、トイレの妻の筋かい。"""
    import json
    from yabitsu import config, start_lot
    spec = start_lot.load_spec(config.DATA_MANUAL / "s1_start_lot.json")
    pad = {"z": 563.0, "polygon": [[0, 0], [1, 0], [1, 1], [0, 1]]}
    names = {n: (v, mat) for n, v, f, uv, mat, r in start_lot.meshes(spec, pad)}
    rep = set(spec.get("replaced", []))   # Codex の部品（landmarks）に替えた石碑・トイレは、簡単な形を作らない
    for need in ("s1_shelter_roof", "1WALL_s1_shelter_posts", "1ROAD_s1_hill_stairs", "s1_hill_handrails",
                 *(() if "monument" in rep else ("s1_monument_plaque",)), *(() if "building" in rep else ("s1_building_truss",))):
        assert need in names, need
    from yabitsu import landmarks
    for asset, part in (("nanohanadai_toilet", "walls"), ("nanohanadai_monument", "monument")):
        if landmarks.exists(asset, part):
            assert any(n.startswith("s1_lm_") and n.endswith(part) for n in names), part
    roof = names["s1_shelter_roof"][0]
    assert roof[:, 2].min() > 563.0 + spec["shelter"]["mound"] + 2.0
    st = names["1ROAD_s1_hill_stairs"][0]
    assert st[:, 2].max() >= spec["hill_stairs"]["z1"] - 0.05
    fl = [p for p in start_lot.flat_pads(spec, pad) if p["name"] in ("shelter_mound", "shelter_hill", "hill_stairs")]   # 丘があれば shelter_hill
    assert len(fl) == 2 and "plane" in [p for p in fl if p["name"] == "hill_stairs"][0]


def test_lot_surface_meets_road_edge_without_step():
    """駐車場の入口：舗装は道路（側溝の外の縁）の高さから LOT_JOIN_M かけて駐車場の高さへ寄せる。すき間も段差も無い（2026-10-30
    ゲームで道路から駐車場へ入ると段差で衝撃が出た：舗装は 562.27 の平らで、道路の縁は 562.31〜562.80、間に 0.2〜0.4 m のすき間）。"""
    import numpy as np
    from shapely.geometry import Point, Polygon
    from yabitsu import start_lot
    from yabitsu.crosssection import GUTTER_WIDTH, Section
    n = 200
    s = np.linspace(0, 199, n)
    sec = Section(s=s, x=s.copy(), y=np.zeros(n), z=np.linspace(10.0, 12.0, n), heading=np.zeros(n), curvature=np.zeros(n),
                  lanes=np.full(n, 2), bridge=np.zeros(n, bool), hw_left=np.full(n, 3.0), hw_right=np.full(n, 3.0), slope=np.zeros(n), crown=np.zeros(n))
    spec = {"origin": [100.0, 3.0 + GUTTER_WIDTH + 0.3], "axis_deg": 0.0}
    lot = Polygon([(80, 3.9), (120, 3.9), (120, 30), (80, 30)])          # 道路（側溝まで y 3.6）との間に 0.3 m のすき間
    v, f, uv = start_lot.lot_surface(spec, {"polygon": list(lot.exterior.coords), "z": 10.5}, sec)
    road_edge = 3.0 + GUTTER_WIDTH
    tri = v[f]
    cen = tri.mean(1)
    near = cen[:, 1] < road_edge + 0.2
    assert near.any()                                                     # すき間を埋めた
    for x0 in (85.0, 100.0, 115.0):                                       # 縁の高さ＝道路の縁、4 m 先で駐車場の高さ
        ze = 10.0 + 2.0 * x0 / 199
        on = (np.abs(v[:, 0] - x0) < 0.3)
        e = on & (np.abs(v[:, 1] - road_edge) < 0.05)
        assert e.any() and np.allclose(v[e, 2], ze, atol=0.02)
    far = v[:, 1] > road_edge + start_lot.LOT_JOIN_M + 0.01
    assert far.any() and np.allclose(v[far, 2], 10.5, atol=1e-6)
    # 隣り合う頂点（同じ位置）の高さがそろう＝割れ目が無い
    key = np.round(v[:, :2], 4)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    for k in np.unique(inv):
        zz = v[inv == k, 2]
        assert np.ptp(zz) < 1e-6
    # 坂はゆるい（縁の差 ≤ 0.5 m を 4 m で）
    for t in tri:
        d = np.hypot(*(t[:, None, :2] - t[None, :, :2]).reshape(-1, 2).T)
        dz = np.abs(t[:, None, 2] - t[None, :, 2]).ravel()
        ok = d > 1e-6
        assert (dz[ok] / d[ok]).max() < 0.2


def test_lot_entrance_has_no_gutter_dip():
    """駐車場の入口では側溝の V 字のくぼみ（深さ 0.12 m）を作らず、路面を道路の縁の高さのまま延ばす（mouths）。
    2026-10-30 ゲームで、舗装を寄せたあとも側溝のくぼみが段差として残っていた。"""
    import numpy as np
    from shapely.geometry import Polygon
    from yabitsu import assembly, start_lot
    from yabitsu.crosssection import GUTTER_WIDTH, Section
    from yabitsu.manifest import Manifest
    n = 200
    s = np.linspace(0, 199, n)
    sec = Section(s=s, x=s.copy(), y=np.zeros(n), z=np.full(n, 10.0), heading=np.zeros(n), curvature=np.zeros(n),
                  lanes=np.full(n, 2), bridge=np.zeros(n, bool), hw_left=np.full(n, 3.0), hw_right=np.full(n, 3.0),
                  slope=np.zeros(n), crown=np.zeros(n))
    lot = {"kind": "lot", "polygon": list(Polygon([(80, 3.9), (120, 3.9), (120, 30), (80, 30)]).exterior.coords), "z": 10.0}
    mo = start_lot.mouths([lot], sec)
    assert len(mo) == 1 and mo[0][0] == 1 and mo[0][1] <= 80.5 and mo[0][2] >= 119.5
    import tempfile
    m = Manifest(tempfile.mkdtemp())
    assembly.add_road(m, sec, mouths=mo)
    for name, e in m.meshes.items():
        if name.startswith("1ROAD_s1_gutterL"):
            d = np.load(m.root / e["file"]); fv = d["verts"][np.unique(d["faces"])]   # 面に使われている頂点
            dip = fv[(fv[:, 0] > 82) & (fv[:, 0] < 118) & (fv[:, 1] > 3.0 + 0.05) & (fv[:, 1] < 3.0 + GUTTER_WIDTH - 0.05)]
            assert not len(dip) or dip[:, 2].min() > 10.0 - 0.01, "入口に側溝のくぼみが残っている"
    assert any(k.startswith("1ROAD_s1_mouthL") for k in m.meshes)


def test_shelter_hill_replaces_the_low_mound():
    """東屋の島の丘（shelter.hill：a・b の多角形と、駐車場からの高さ dz）。あれば小さな丸い盛り土の代わりに、その形の平らな上の面を
    地形に置く（上の面の地形＝駐車場＋dz）。2026-10-30 写真で東屋の床は約 565.6 m、今の島は 563.2〜563.6 m で北の道より低かった。"""
    from yabitsu import config, start_lot, terrain
    spec = start_lot.load_spec(config.DATA_MANUAL / "s1_start_lot.json")
    pad = {"z": 562.27, "polygon": [[0, 0], [1, 0], [1, 1], [0, 1]]}
    spec = {**spec, "shelter": {**spec["shelter"], "mound": 3.33,
                                "hill": {"points": [[20, 0], [28, 0], [28, 6], [20, 6]], "dz": 3.33}}}
    fl = {p["name"]: p for p in start_lot.flat_pads(spec, pad)}
    assert "shelter_mound" not in fl and "shelter_hill" in fl
    assert fl["shelter_hill"]["z"] - terrain.UNDER_DEPTH == pytest.approx(562.27 + 3.33)
    names = {n: v for n, v, f, uv, mat, r in start_lot.meshes(spec, pad)}
    assert names["1WALL_s1_shelter_posts"][:, 2].min() == pytest.approx(562.27 + 3.33 - 0.5)
