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
