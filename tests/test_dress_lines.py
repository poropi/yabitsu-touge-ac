import csv

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
