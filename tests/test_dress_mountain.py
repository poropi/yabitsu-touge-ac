import numpy as np
from PIL import Image

from yabitsu import dress_mountain as dm
from yabitsu import features as ft
from yabitsu.crosssection import GUTTER_WIDTH
from yabitsu.crosssection import Section
from yabitsu.meshgen import face_normals


def _sec(n=300, crown=0.02):
    s = np.arange(n, dtype=float)
    return Section(s, s.copy(), np.zeros(n), np.full(n, 10.0), np.zeros(n), np.zeros(n), np.full(n, 2, np.int32),
                   np.zeros(n, bool), np.full(n, 3.0), np.full(n, 3.0), np.zeros(n), np.full(n, crown))


def test_wall_finish_from_notes():
    assert dm.wall_finish("苔むした石積み（ストリートビュー）") == "kenchi_moss"
    assert dm.wall_finish("石積み（s=200・250）") == "kenchi_moss"
    assert dm.wall_finish("高いブロック擁壁（s=3100・3150）") == "block"
    assert dm.wall_finish("苔むしたブロック擁壁（ストリートビュー）") == "kenchi_moss"
    assert dm.wall_finish("岩の切土（s=1050）") == "rock"
    assert dm.wall_finish("岩の切土とネット（ストリートビュー）") == "rock_net"
    assert dm.wall_finish("落石防護網の急斜面（ストリートビュー）・見た目は仕上げで直す") == "rock_net"
    assert dm.wall_finish("大きな法枠とネット（ストリートビュー）") == "norikae"
    assert dm.wall_finish("吹付けの斜面とネット（下にコンクリートの土台）（ストリートビュー）") == "shotcrete"
    assert dm.wall_finish("ネットの下のコンクリートの土台（ストリートビュー）") == "concrete"
    assert dm.wall_finish("低いコンクリート壁（ストリートビュー）") == "concrete"
    assert dm.wall_finish("低い擁壁（s=6350）") == "concrete"
    assert dm.wall_finish("石積み擁壁（ストリートビュー）") == "kenchi_moss"
    assert dm.wall_finish("低い石積み（s=950）") == "kenchi"


def test_guardrail_on_the_collision_line_with_posts():
    sec = _sec()
    f = ft.Feature(20.0, 120.0, "R", "guardrail", 0.0, "")
    parts = dm.guardrail_meshes(sec, f)
    beam = [p for p in parts if p[3] == "beam"]; post = [p for p in parts if p[3] == "post"]
    assert beam and post
    v = beam[0][0]
    mid = (v[:, 0] > 30) & (v[:, 0] < 110)                       # 端の曲げの外
    line = 3.0 + GUTTER_WIDTH + ft.GUARDRAIL_OFFSET              # 当たり判定の壁（右＝ y が負）
    assert np.all(-v[mid, 1] <= line + 1e-6) and np.all(-v[mid, 1] >= line - 0.07)
    z = v[mid, 2] - (10.0 - 0.02 * 3.0)                          # 道の縁（水勾配で中心より 6 cm 低い）から
    assert np.isclose(z.min(), 0.425, atol=0.02) and np.isclose(z.max(), 0.775, atol=0.02)
    ends = v[np.isclose(v[:, 0], 20.0, atol=0.6)]
    assert -ends[:, 1].min() > line + 0.15                       # 端は外へ曲がる
    pv = post[0][0]
    assert len(pv) == 25 * 23                                    # 4 m ごとに 25 本（1 本 23 頂点）
    assert -pv[:, 1].min() > line                                # 支柱は板の後ろ


def test_guardrail_long_run_is_split_under_vertex_limit():
    sec = _sec(8000)
    parts = dm.guardrail_meshes(sec, ft.Feature(0.0, 7999.0, "L", "guardrail", 0.0, ""))
    assert len([p for p in parts if p[3] == "beam"]) >= 3
    assert all(len(p[0]) <= 65535 for p in parts)


def test_leaves_face_up_and_float_just_above_the_road():
    sec = _sec()
    parts = dm.leaf_meshes(sec, 10.0, 200.0)
    assert {p[3] for p in parts} == {"edge", "lane"}
    for v, f, uv, part in parts:
        n = face_normals(v, f)
        assert np.all(n[:, 2] > 0.9), part
        lat = np.abs(v[:, 1])
        road = 10.0 - 0.02 * lat                                 # 路面（横断の水勾配）
        assert np.all(v[:, 2] - road > 0.005) and np.all(v[:, 2] - road < 0.02), part
        assert lat.max() <= 3.0 + 1e-6                           # 道の縁より外へ出ない（側溝の上には置かない）
        if part == "lane":
            assert uv[:, 0].min() >= -1e-6 and uv[:, 0].max() <= 1 + 1e-6


def test_leaf_textures_are_dense_at_the_edge_and_sparse_on_the_lane():
    leaves = Image.new("RGB", (256, 256), (120, 70, 30))
    edge, lane = dm.leaf_textures(leaves)
    a = np.asarray(edge)[..., 3] > 0
    w = a.shape[1]
    um = np.linspace(0, dm.EDGE_W, w)
    assert a[:, um < 0.15].mean() > 0.6                          # 縁（u=0）にたまる
    assert a[:, (um > 0.19) & (um < 0.31)].mean() < 0.1          # 外側線の上はほとんど無い
    assert a[:, um > 0.4].mean() < 0.3
    b = np.asarray(lane)[..., 3] > 0
    assert 0.002 < b.mean() < 0.08
