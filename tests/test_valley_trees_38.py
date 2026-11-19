import numpy as np

from scripts.align_valley_trees import plan_forest_changes
from scripts.track_build_manifest import card_adjustments
from yabitsu import valleys, valley_fit


def test_fine_trees_follow_actual_bed_water_and_raised_bank():
    old = {"xs": np.arange(4.0), "ys": np.arange(2.0), "Z": np.zeros((2, 4))}
    cells = np.array([[0, 0], [0, 1], [0, 2]])
    tri = valleys.original_triangles(old, cells)
    tri[:, :, 2] = 1.0
    old_water = valley_fit.Surfaces(tri[tri[:, :, 0].mean(axis=1).astype(int) == 1] + [0.0, 0.0, 0.2])
    after = {
        **old,
        "valley_faces": np.arange(tri.size // 3).reshape(-1, 3),
        "valley_vertices": tri.reshape(-1, 3),
        "valley_cells": cells,
        "valley_bed": tri[:, :, 0].mean(axis=1) < 1.0,
    }
    dug = valleys.changed_ground(old, after)
    roots = np.array([[0.5, 0.3, 0.0], [1.5, 0.3, 0.0], [2.5, 0.3, 0.0], [3.5, 0.3, 0.0]])
    trees = np.rec.fromarrays(roots.T, names=("x", "y", "z"))
    plan = plan_forest_changes(trees, dug, [], (old_water, old_water))
    assert [(c.index, c.reason, c.new_ground) for c in plan] == [
        (0, "谷底", 1.0),
        (1, "水面", 1.0),
        (2, "", 1.0),
    ]
    z, remove = card_adjustments(roots, dug, [], (old_water, old_water))
    assert remove.tolist() == [True, True, False, False]
    assert z[2] == 1.0 and np.isnan(z[3])


def test_cut_rail_post_foot_follows_ground_at_new_intersection():
    from types import SimpleNamespace
    from yabitsu import small_bridges

    sec = SimpleNamespace(x=np.arange(4.0), y=np.zeros(4), heading=np.zeros(4), n=4)
    # x=0が切断面。両端は高さ1の岸、中央は高さ0の折れ目。
    dug = SimpleNamespace(height=lambda xy: np.abs(np.asarray(xy)[..., 0]))
    verts = np.array([[-1.0, 0.0, 0.88], [1.0, 0.0, 0.88], [-1.0, 1.0, 0.88]])
    uv = np.zeros((3, 2))
    clipped, _, _ = small_bridges.cut_guardrail(
        sec, (verts, np.array([[0, 1, 2]]), uv), [(0, 2)], foot_surface=dug
    )
    cut = clipped[np.isclose(clipped[:, 0], 0.0)]
    assert len(cut) == 2
    np.testing.assert_allclose(cut[:, 2], [-0.12, -0.12])
