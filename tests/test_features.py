import numpy as np
import pytest

from yabitsu import features as ft
from yabitsu.meshgen import face_normals
from tests.test_meshgen import _straight


def test_wall_slope_hides_terrain():
    assert ft.WALL_CUT_SLOPE < 1 / ft.WALL_LEAN


def test_load_csv(tmp_path):
    p = tmp_path / "f.csv"
    p.write_text("s_from,s_to,side,type,height_m,note\n10,50,L,retaining_wall,3,x\n60,90,R,guardrail,,\n", encoding="utf-8")
    fs = ft.load_features_csv(p)
    assert fs[0] == ft.Feature(10.0, 50.0, "L", "retaining_wall", 3.0, "x") and fs[1].height_m == 0.0
    assert ft.load_features_csv(tmp_path / "none.csv") == []


def test_rows_of_validates():
    sec = _straight()
    assert ft.rows_of(sec, ft.Feature(10, 20, "L", "guardrail", 0, "")) == slice(10, 21)
    with pytest.raises(ValueError):
        ft.rows_of(sec, ft.Feature(50, 20, "L", "guardrail", 0, ""))
    with pytest.raises(ValueError):
        ft.rows_of(sec, ft.Feature(10, 20, "X", "guardrail", 0, ""))


def test_cut_slopes_behind_walls():
    sec = _straight()
    cl, cr = ft.cut_slopes(sec, [ft.Feature(10, 20, "L", "retaining_wall", 3, "")])
    assert cl[15] == ft.WALL_CUT_SLOPE and cl[50] == ft.CUT_SLOPE and cr[15] == ft.CUT_SLOPE


def test_meshes_face_the_road():
    sec = _straight()
    vis, col = ft.guardrail_meshes(sec, ft.Feature(10, 60, "R", "guardrail", 0, ""))
    assert (face_normals(vis[0], vis[1])[:, 1] > 0).all()   # 右側の壁は +y（道路側）を向く
    assert col[0][:, 2].max() > vis[0][:, 2].max()
    v, f, _ = ft.retaining_wall_mesh(sec, ft.Feature(10, 60, "L", "retaining_wall", 3, ""))
    n = face_normals(v, f)
    assert (n[:, 1] < 0).all() and v[:, 2].max() == pytest.approx(sec.edge_z(1)[10] + 3.0)


def test_suggest_candidates():
    sec = _straight(401)
    def z_nat(x, y):
        x = np.asarray(x); y = np.asarray(y)
        z = np.full(x.shape, 100.0)
        z[(y > 0) & (x >= 100) & (x <= 200)] = 106.0
        z[(y < 0) & (x >= 300) & (x <= 350)] = 94.0
        return z
    c = ft.suggest_candidates(sec, z_nat)
    kinds = sorted((r["side"], r["type"]) for r in c)
    assert kinds == [("L", "retaining_wall"), ("R", "guardrail")]
    wall = next(r for r in c if r["type"] == "retaining_wall")
    assert 95 <= wall["s_from"] <= 105 and 195 <= wall["s_to"] <= 205 and "google.com/maps" in wall["streetview"]


def test_candidate_range_stays_inside_section_at_the_end():
    from tests.test_meshgen import _sec_from_xy
    x = np.linspace(0.0, 200.6, 202)
    sec = _sec_from_xy(x, np.zeros_like(x))
    c = ft.suggest_candidates(sec, lambda x, y: np.where(np.asarray(y) > 0, 110.0, 100.0))
    assert c and all(r["s_to"] <= sec.s[-1] for r in c)
    for r in c:
        ft.rows_of(sec, ft.Feature(r["s_from"], r["s_to"], r["side"], r["type"], r["height_m"], ""))


def test_features_from_shifted_rows_keep_local_meaning():
    # 区間の始点が通しで 100m の表：区間の距離 10〜20m → 通しの 110〜120m
    sec = _straight(201)
    rows = [{"s_from": 110.0, "s_to": 120.0, "side": "L", "type": "retaining_wall", "height_m": "2.0", "note": "x"},
            {"s_from": 130.0, "s_to": 150.0, "side": "R", "type": "guardrail", "height_m": "", "note": ""}]
    feats = ft.load_features_rows(rows)
    assert (feats[0].s_from, feats[0].s_to, feats[0].height_m) == (110.0, 120.0, 2.0)
    assert feats[1].height_m == 0.0 and feats[1].type == "guardrail"
    sl = ft.rows_of(sec, feats[0])
    assert sec.s[sl.start] >= 110.0 - 1e-9 and sec.s[sl.stop - 1] <= 120.0 + 1e-9


def test_parapet_is_vertical_and_at_gutter_edge():
    from yabitsu.crosssection import GUTTER_WIDTH
    sec = _straight()
    v, f, _ = ft.parapet_mesh(sec, ft.Feature(10, 30, "R", "parapet", 0, ""))
    hw = sec.hw_right[10]
    assert np.allclose(v[:, 1], -(hw + GUTTER_WIDTH))           # 内側の縁＝側溝の外側、まっすぐ立つ
    assert v[:, 2].max() == pytest.approx(sec.edge_z(-1)[10] + 0.9)  # 高さが空なら 0.9m
    assert (face_normals(v, f)[:, 1] > 0).all()                  # 道路側を向く
    v2, _, _ = ft.parapet_mesh(sec, ft.Feature(10, 30, "L", "parapet", 1.2, ""))
    assert v2[:, 2].max() == pytest.approx(sec.edge_z(1)[10] + 1.2)


def test_parapet_type_is_accepted():
    sec = _straight()
    assert ft.rows_of(sec, ft.Feature(10, 30, "L", "parapet", 0, "")) == slice(10, 31)


_REAL = __import__("yabitsu.config", fromlist=["x"]).DATA_PROCESSED / "track_pads.json"


@pytest.mark.skipif(not _REAL.exists(), reason="処理済みデータがまだない")
def test_real_barriers_do_not_cross_parking_lots():
    # ガードレール・擁壁が駐車場の中に入ると、出入りできなくなる
    import json
    from shapely.geometry import LineString, Polygon
    from yabitsu import config, track
    from yabitsu.crosssection import GUTTER_WIDTH, Section
    sec = Section.load_json(config.DATA_PROCESSED / "track_section.json")
    drive_in = lambda p: p.get("kind") in ("lot", "pass") or p["name"] == "minoge_turnaround"  # 家の土台は除く（家の前の塀は本物）
    lots = [Polygon(p["polygon"]) for p in json.loads(_REAL.read_text(encoding="utf-8")) if drive_in(p)]
    for f in ft.load_features_rows(track.load_section_rows("features", track.load_bounds())):
        sl = ft.rows_of(sec, f); side = ft.SIDES[f.side]
        L = (sec.hw_left if side > 0 else sec.hw_right)[sl] + GUTTER_WIDTH + 0.1
        x = sec.x[sl] - side * L * np.sin(sec.heading[sl]); y = sec.y[sl] + side * L * np.cos(sec.heading[sl])
        if len(x) < 2:
            continue
        line = LineString(np.c_[x, y])
        assert all(line.intersection(lot).length < 0.5 for lot in lots), f


def test_width_column_is_optional():
    rows = [{"s_from": "0", "s_to": "10", "side": "L", "type": "sidewalk", "height_m": "", "width_m": "2.5", "note": ""},
            {"s_from": "0", "s_to": "10", "side": "L", "type": "guardrail", "height_m": "", "note": ""}]
    a, b = ft.load_features_rows(rows)
    assert a.width_m == 2.5 and b.width_m == 0.0


def test_sidewalk_top_is_curb_height_above_road_edge():
    from yabitsu.crosssection import GUTTER_WIDTH
    sec = _straight()
    for side in ("L", "R"):
        (v, f, uv), sides = ft.sidewalk_mesh(sec, ft.Feature(10, 60, side, "sidewalk", 0.0, "", width_m=2.0))
        from yabitsu.meshgen import folded_faces
        assert len(folded_faces(v, f)) == 0                     # 走る面（上面）に縦の面を混ぜない
        assert np.isclose(face_normals(*sides[:2])[:, 2], 0.0, atol=1e-6).all()   # 縁石と外の縁は縦の面
        sgn = 1 if side == "L" else -1
        hw = sec.hw_left[0] if side == "L" else sec.hw_right[0]
        ze = sec.edge_z(sgn)[0]
        top = v[np.isclose(v[:, 2], ze + ft.CURB_HEIGHT)]
        assert np.abs(top[:, 1]).min() == pytest.approx(hw + GUTTER_WIDTH)
        assert np.abs(top[:, 1]).max() == pytest.approx(hw + GUTTER_WIDTH + 2.0)
        assert (face_normals(v, f)[:, 2] > 0.99).all()           # 上面は上向き


def test_hedge_is_box_outside_gutter():
    from yabitsu.crosssection import GUTTER_WIDTH
    sec = _straight()
    v, f, uv = ft.hedge_mesh(sec, ft.Feature(10, 60, "L", "hedge", 1.5, ""))
    ze = sec.edge_z(1)[0]
    assert v[:, 1].min() == pytest.approx(sec.hw_left[0] + GUTTER_WIDTH)
    assert v[:, 2].max() == pytest.approx(ze + 1.5)


def test_overpass_deck_clears_road_and_piers_stand_outside():
    from yabitsu.crosssection import GUTTER_WIDTH
    sec = _straight()
    deck, piers = ft.overpass_meshes(sec, ft.Feature(40, 60, "L", "overpass", 6.0, "", width_m=8.0))
    ze = max(sec.edge_z(1)[50], sec.edge_z(-1)[50])
    assert deck[0][:, 2].min() >= ze + 6.0 - 1e-6
    assert deck[0][:, 1].max() == pytest.approx(sec.hw_left[0] + GUTTER_WIDTH + 8.0)
    inner = np.abs(piers[0][:, 1]).min()
    assert inner >= sec.hw_left[0] + GUTTER_WIDTH + 0.5 - 1e-6


def test_walls_stand_behind_the_sidewalk():
    from yabitsu.crosssection import GUTTER_WIDTH
    sec = _straight()
    feats = [ft.Feature(10, 60, "L", "sidewalk", 0.0, "", width_m=2.0), ft.Feature(20, 50, "L", "retaining_wall", 2.0, "")]
    off_l, off_r = ft.sidewalk_offsets(sec, feats)
    assert off_l[30] == 2.0 and off_l[80] == 0.0 and off_r[30] == 0.0
    v, f, uv = ft.retaining_wall_mesh(sec, feats[1], offset=off_l)
    assert v[:, 1].min() == pytest.approx(sec.hw_left[0] + GUTTER_WIDTH + 2.0)
    v, f, uv = ft.hedge_mesh(sec, ft.Feature(20, 50, "L", "hedge", 1.2, ""), offset=off_l)
    assert v[:, 1].min() == pytest.approx(sec.hw_left[0] + GUTTER_WIDTH + 2.0)


def test_fence_is_see_through_railing_on_collision_wall():
    sec = _straight()
    from yabitsu.manifest import Manifest
    from yabitsu import assembly
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        m = Manifest(d)
        assembly.add_features(m, sec, [ft.Feature(10, 60, "L", "fence", 1.1, "")])
        (name, info), = m.meshes.items()
        assert name.startswith("1WALL_track_fence_") and info["material"] == "railing_bars"


def test_features_are_cut_at_branch_mouths():
    feats = [ft.Feature(0, 100, "L", "retaining_wall", 3.0, ""), ft.Feature(0, 100, "R", "guardrail", 0.0, ""),
             ft.Feature(0, 100, "L", "sidewalk", 0.0, "", width_m=2.0)]
    cut = ft.cut_at(feats, {"L": [(40.0, 50.0)], "R": []})
    walls = [(f.s_from, f.s_to) for f in cut if f.type == "retaining_wall"]
    assert walls == [(0.0, 40.0), (50.0, 100.0)]
    assert [(f.s_from, f.s_to) for f in cut if f.type == "guardrail"] == [(0.0, 100.0)]
    assert [(f.s_from, f.s_to) for f in cut if f.type == "sidewalk"] == [(0.0, 100.0)]   # 歩道は切らない


def test_tiny_leftovers_after_a_cut_are_dropped():
    feats = [ft.Feature(10.3, 52.1, "L", "hedge", 1.2, "")]
    cut = ft.cut_at(feats, {"L": [(12.0, 50.0)], "R": []})
    assert cut == []          # 1.7m と 2.1m の切れ端は作らない（面が作れない・意味がない）
