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
