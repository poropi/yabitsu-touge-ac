import csv
import pytest

import numpy as np

from yabitsu import config, dress_lines
from tests.test_dress_parts import _sec


def test_real_table_centre_rows_do_not_overlap():
    rows = [r for r in csv.DictReader(open(config.DATA_MANUAL / "s3_lines.csv", encoding="utf-8")) if r["item"] == "centre"]
    iv = sorted((float(r["s_from"]), float(r["s_to"])) for r in rows)
    assert all(a < b for a, b in iv) and all(iv[k][1] <= iv[k + 1][0] for k in range(len(iv) - 1))
    assert {r["value"] for r in rows} <= set(dress_lines.CENTRE)


def test_dotted_centre_is_segmented():
    sec = _sec()
    meshes = dress_lines.line_meshes(sec, [{"s_from": "0", "s_to": "20", "item": "centre", "side": "C", "value": "orange_dotted"}])
    (name, v, f, uv, mat), = meshes
    assert mat == "dress_line_orange"
    xs = np.unique(np.round(v[:, 0], 2))
    gaps = np.diff(xs)
    assert gaps.max() > 0.1 and len(f) >= 20          # 20 m に 20 個以上の区切り


def test_green_moves_edge_line_inward():
    sec = _sec()
    rows = [{"s_from": "0", "s_to": "50", "item": "edge", "side": "B", "value": "solid"},
            {"s_from": "0", "s_to": "50", "item": "green", "side": "B", "value": "0.75"}]
    ms = {m[0]: m for m in dress_lines.line_meshes(sec, rows)}
    edge = [m for n, m in ms.items() if m[4] == "dress_line_white"]
    green = [m for n, m in ms.items() if m[4] == "dress_green"]
    assert green and edge
    ymax = max(np.abs(m[1][:, 1]).max() for m in edge)
    assert ymax < 3.0 - 0.6                              # 道の端（3 m）から 0.75 m 内側
    assert np.abs(green[0][1][:, 1]).max() > 2.95


def test_decals_above_road():
    sec = _sec()
    for name, v, f, uv, mat in dress_lines.line_meshes(sec, [{"s_from": "0", "s_to": "30", "item": "edge", "side": "B", "value": "dashed"}]):
        assert (v[:, 2] > 10.0).all()


def test_flanked_only_in_nagoki_town():
    """中央線のすぐ両側の白い破線は、名古木の町なか（s=200〜372、ストリートビュー s=250）だけ。"""
    rows = [r for r in csv.DictReader(open(config.DATA_MANUAL / "s3_lines.csv", encoding="utf-8")) if r["value"] == "orange_flanked"]
    assert [(float(r["s_from"]), float(r["s_to"])) for r in rows] == [(200.0, 372.0)]


def test_gap_rows_cut_edges_and_green():
    sec = _sec()
    rows = [{"s_from": "0", "s_to": "100", "item": "edge", "side": "B", "value": "solid"},
            {"s_from": "0", "s_to": "100", "item": "green", "side": "B", "value": "0.4"},
            {"s_from": "40", "s_to": "60", "item": "gap", "side": "B", "value": ""}]
    for name, v, f, uv, mat in dress_lines.line_meshes(sec, rows):
        assert not ((v[:, 0] > 41) & (v[:, 0] < 59)).any(), name


def test_diamonds_before_unsignalled_crosswalks():
    sec = _sec(400)
    pave = [{"s_from": "200", "s_to": "204", "kind": "crosswalk"},
            {"s_from": "100", "s_to": "104", "kind": "crosswalk"}, {"s_from": "97.5", "s_to": "98", "kind": "stop_line"}]
    d = dress_lines.diamond_spots(sec, pave)
    # 信号のない 200 の横断歩道だけ：上り（左の車線）は 150・170、下り（右の車線）は 234・254
    assert sorted((round(s), side) for s, side in d) == [(150, "L"), (170, "L"), (234, "R"), (254, "R")]


def test_text_and_diamond_meshes_on_lane():
    sec = _sec(400)
    ms = dress_lines.marking_meshes(sec, [(150.0, "L")], [{"s_from": "300", "item": "text", "side": "R", "value": "40"}])
    by = {m[4]: m for m in ms}
    assert set(by) == {"dress_diamond", "dress_text_40"}
    assert (by["dress_diamond"][1][:, 1] > 0).all() and (by["dress_text_40"][1][:, 1] < 0).all()
    assert (by["dress_diamond"][1][:, 2] > 10.0).all()


def test_marking_follows_slope():
    """坂道でも、標示の四隅が路面の高さに乗る（真ん中の高さで平らにすると半分が路面の下に沈む）。"""
    sec = _sec(400)
    sec.z[:] = sec.s * 0.1                                     # 10 % の坂
    (name, v, f, uv, mat), = dress_lines.marking_meshes(sec, [(150.0, "L")], [])
    assert np.allclose(v[:, 2], v[:, 0] * 0.1 + dress_lines.LIFT, atol=0.02)


def test_split_lines_stay_on_carriageway():
    """島で分かれる所の線：島に乗らない・路面の外に出ない・外側は実線、島側は破線、緑は外側線の外。"""
    from types import SimpleNamespace
    from shapely.geometry import LineString, Point, box
    island = box(40, -1.5, 60, 1.5)
    up = LineString([(0, 0), (35, 0), (45, -3.6), (55, -3.6), (65, 0), (100, 0)])
    down = LineString([(0, 0), (35, 0), (45, 3.6), (55, 3.6), (65, 0), (100, 0)])
    lanes = {"u": up.buffer(1.8, cap_style="flat"), "d": down.buffer(1.8, cap_style="flat")}
    from shapely.ops import unary_union
    cw = unary_union(list(lanes.values())).difference(island)
    sp = SimpleNamespace(island=island, carriageway=cw, zebra=box(30, -2, 40, 2).difference(island).intersection(cw),
                         lines={"u": up, "d": down}, singles={"u", "d"}, window=box(0, -10, 100, 10),
                         _ends=lambda: box(-5, -10, 2, 10).union(box(98, -10, 105, 10)))
    out = dress_lines.split_lines(sp, lambda x, y: np.zeros_like(np.asarray(x, float)))
    mats = {m[4] for m in out}
    assert {"dress_line_white", "dress_green"} <= mats
    # 島のまわりは白の実線だけ（緑は外側だけ）：ストリートビュー s=3400 上り・下り
    green = [m for m in out if m[4] == "dress_green"]
    assert all(island.distance(Point(p)) > 0.6 for _, v, *_ in green for p in v[:, :2])
    for name, v, f, uv, mat in out:
        pts = v[:, :2]
        assert not any(island.contains(Point(p)) for p in pts), name
        assert all(cw.buffer(0.05).contains(Point(p)) for p in pts), name


def test_red_lane_on_right_half():
    sec = _sec()
    (name, v, f, uv, mat), = dress_lines.line_meshes(sec, [{"s_from": "100", "s_to": "150", "item": "red", "side": "R", "value": ""}])
    assert mat == "dress_red" and (v[:, 1] < 0.01).all() and (v[:, 1] > -3.0).all()
    assert v[:, 0].min() >= 99.5 and v[:, 0].max() <= 150.5
    assert (v[:, 2] > 10.0).all() and (v[:, 2] < 10.0 + dress_lines.LIFT).all()      # 線より下


def test_red_row_starts_after_gap():
    sec = _sec()
    rows = [{"s_from": "100", "s_to": "200", "item": "red", "side": "R", "value": ""},
            {"s_from": "90", "s_to": "150", "item": "gap", "side": "B", "value": ""}]
    (name, v, f, uv, mat), = [m for m in dress_lines.line_meshes(sec, rows) if m[4] == "dress_red"]
    assert v[:, 0].min() >= 149.5


def test_split_lines_keep_out():
    from shapely.geometry import box
    from types import SimpleNamespace
    from shapely.geometry import LineString
    cw = box(0, -4, 100, 4)
    sp = SimpleNamespace(island=box(45, -0.5, 55, 0.5), carriageway=cw.difference(box(45, -0.5, 55, 0.5)),
                         zebra=box(20, -1, 40, 1), lines={"a": LineString([(0, 0), (100, 0)])}, singles=set(),
                         _ends=lambda: box(-5, -5, 1, 5).union(box(99, -5, 105, 5)))
    out = dress_lines.split_lines(sp, lambda x, y: np.zeros_like(np.asarray(x, float)), keep_out=box(15, -4, 42, 0))
    for name, v, f, uv, mat in out:
        if mat == "dress_line_white":
            cen = v[f].mean(1)
            assert not ((cen[:, 0] > 16) & (cen[:, 0] < 41) & (cen[:, 1] < -0.1)).any()


def test_road_marks_number_faces_its_traffic():
    """数字：dir=b（s が減る向きに走る車）の標示は右の車線（y < 0）。uv の v は走る向き（-x）に増える。"""
    sec = _sec(400)
    (name, v, f, uv, mat), = dress_lines.road_mark_meshes(sec, [{"s": "200", "side": "R", "dir": "b", "kind": "number", "value": "30", "color": "orange"}])
    assert mat == "mt_mark_n30_orange" and (v[:, 1] < 0).all()
    assert v[uv[:, 1].argmax(), 0] < v[uv[:, 1].argmin(), 0]
    assert np.ptp(v[:, 0]) == pytest.approx(dress_lines.MARK_NUMBER_SIZE[1], abs=0.01)


def test_road_marks_text_reads_far_to_near():
    """文字：1 字ずつの四角。走る車から見て最初の字がいちばん奥（2026-10-22 ストリートビュー s2_1150_b「左急カーブ」）。C は道の真ん中。"""
    sec = _sec(400)
    ms = dress_lines.road_mark_meshes(sec, [{"s": "200", "side": "C", "dir": "f", "kind": "text", "value": "左急カーブ", "color": "white"}])
    centre = {}
    for name, v, f, uv, mat in ms:
        for q in f:
            centre.setdefault(mat, []).append(v[q].mean(0))
    x = {mat: np.mean([c[0] for c in cs]) for mat, cs in centre.items()}
    order = [dress_lines.char_material(c, "white") for c in "左急カーブ"]
    assert [x[m] for m in order] == sorted([x[m] for m in order], reverse=True)    # 左 が +x（奥）
    assert all(abs(c[1]) < 0.01 for cs in centre.values() for c in cs)


def test_mark_textures_are_long_with_alpha():
    img = dress_lines.mark_texture("text", "急", "white")
    a = np.asarray(img)[..., 3]
    assert img.height > img.width * 1.5 and (a > 0).mean() > 0.05 and (a == 0).mean() > 0.2


def test_mark_quad_uv_reads_upright_for_its_driver():
    """標示の四角の uv：走る車から見て u は左→右、v は手前→奥。画像は正立のまま貼れば、走る車に正しく読める
    （kn5 と確認描画はどちらも v を 1-v にして画像の下を v=0 に置く）。2026-10-22 に 180 度回って見えた不具合の確かめ。"""
    sec = _sec(400)
    for side, fwd in (("L", 1), ("R", -1)):
        v, f, uv = dress_lines._lane_quad(sec, 200.0, side, (1.0, 2.0), False)
        drv_left = np.array([0.0, 1.0]) * fwd          # 進む向き (+x)*fwd の左
        u_dir = v[uv[:, 0].argmax(), :2] - v[uv[:, 0].argmin(), :2]
        v_dir = v[uv[:, 1].argmax(), :2] - v[uv[:, 1].argmin(), :2]
        assert np.dot(u_dir, -drv_left) > 0             # u が増えると右へ
        assert np.dot(v_dir, [fwd, 0.0]) > 0            # v が増えると奥へ


def test_road_marks_face_up_for_every_kind_and_direction():
    """山道の標示の面は上向き（材質は片面。下向きだとゲームでは裏になって見えない。Blender の描画は裏も描くので気づかなかった。2026-10-27 札掛橋）。"""
    from yabitsu.meshgen import face_normals
    sec = _sec(400)
    rows = [{"s": "100", "side": "L", "dir": "f", "kind": "label", "value": "県道70号", "color": "white"},
            {"s": "150", "side": "R", "dir": "b", "kind": "label", "value": "県道70号", "color": "white"},
            {"s": "200", "side": "R", "dir": "b", "kind": "number", "value": "30", "color": "orange"},
            {"s": "250", "side": "C", "dir": "f", "kind": "text", "value": "止まれ", "color": "white"},
            {"s": "300", "side": "L", "dir": "f", "kind": "diamond", "value": "", "color": "white"}]
    for name, v, f, uv, mat in dress_lines.road_mark_meshes(sec, rows):
        assert (face_normals(v, f)[:, 2] > 0.99).all(), name
