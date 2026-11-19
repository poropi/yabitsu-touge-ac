from importlib import import_module

import numpy as np
from numpy.testing import assert_array_equal

from yabitsu import features, forest_ac
from yabitsu.crosssection import Section


def test_selects_only_changed_triangles_with_more_than_one_centimetre():
    align = import_module("scripts.align_s5_8960_trees")
    old = {"xs": np.array([0., 1.]), "ys": np.array([0., 1.]), "Z": np.zeros((2, 2)),
           "D": np.zeros((2, 2)), "H": np.zeros((1, 1), bool)}
    new = {**old, "Z": np.array([[0., 0.04], [0., 0.]])}
    hits, before, after = align.ground_changes(
        old, new, np.array([0.75, 0.25, 0.50, 0.25, 1.5]), np.array([0.25, 0.00, 0.375, 0.75, 0.25])
    )
    assert_array_equal(hits, [0, 1, 2])  # Lower-left triangle does not share the changed vertex.
    assert_array_equal(before, [0., 0., 0.])
    np.testing.assert_allclose(after, [0.020, 0.010, 0.005], rtol=0, atol=1e-12)
    assert_array_equal(hits[np.abs(after - before) > align.HEIGHT_THRESHOLD], [0])


def test_updates_both_lists_in_place_and_is_idempotent(tmp_path):
    align = import_module("scripts.align_s5_8960_trees")
    x = np.arange(41., dtype=float)
    zeros = np.zeros_like(x)
    width = zeros + 3
    sec = Section(x + 8950, x, zeros, zeros, zeros, zeros, zeros, zeros, width, width, zeros, zeros)
    feats = [features.Feature(8961, 8980, "L", "retaining_wall", 4, "", width_m=0.4)]
    ax = np.array([0., 15., 20., 25., 40.])
    ay = np.arange(-8., 9.)
    old = {"xs": ax, "ys": ay, "Z": np.zeros((len(ay), len(ax))),
           "D": np.zeros((len(ay), len(ax))), "H": np.zeros((len(ay) - 1, len(ax) - 1), bool)}
    z = old["Z"].copy()
    z[:, 2] = np.where(ay < 0, 2.236, -2.236)
    new = {**old, "Z": z}
    xy = [(1., 4.), (20., -4.5), (20., -7.), (20., 4.5), (20., 3.8), (35., -7.), (20., 7.)]
    trees = np.array([(tx, ty, 0., "sugi_a", 35 + i, 1.) for i, (tx, ty) in enumerate(xy)],
                     dtype=[("x", "f8"), ("y", "f8"), ("z", "f8"), ("pattern", "U16"),
                            ("angle_deg", "f4"), ("scale", "f4")]).view(np.recarray)
    path = tmp_path / "trees.npy"
    np.save(path, trees)
    ext = tmp_path / "extension"
    forest_ac.write_lists(trees, ext)
    right = ext / "trees/forest_0_-1.txt"
    left = ext / "trees/forest_0_0.txt"
    other = ext / "trees/forest_1_0.txt"
    other.write_bytes(b"unrelated tile\n")
    before = {p: p.read_bytes() for p in (path, right, left, other)}
    clearance = {"road": 2., "wall": 0.5}

    align.update_forest(tmp_path, old, new, sec, feats, 0., clearance, apply=True)

    actual = np.load(path)
    expected = trees[[0, 2, 5, 6]].copy()
    expected.z[[1, 3]] = [2.236, -2.236]
    assert_array_equal(actual, expected)
    for tile, kept, old_height, new_height in ((right, [5, 6], b", 0.00,", b", 2.24,"),
                                               (left, [4, 7], b", 0.00,", b", -2.24,")):
        lines = before[tile].splitlines(keepends=True)
        expected_lines = [lines[i] for i in [0, 1, 2, 3, *kept]]
        changed_line = 4 if tile == right else 5
        expected_lines[changed_line] = expected_lines[changed_line].replace(old_height, new_height, 1)
        assert tile.read_bytes() == b"".join(expected_lines)
        assert tile.with_name(f"{tile.stem}_before_s5_8960{tile.suffix}").read_bytes() == before[tile]
    assert path.with_name("trees_before_s5_8960.npy").read_bytes() == before[path]
    assert other.read_bytes() == before[other]
    after = {p: p.read_bytes() for p in before}

    align.update_forest(tmp_path, old, new, sec, feats, 0., clearance, apply=True)

    assert all(p.read_bytes() == content for p, content in after.items())
