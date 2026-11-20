"""谷の点検が、測点間や集約メッシュ内の不具合も検出することを確かめる。"""
import numpy as np
import pytest
import shapely
from shapely.geometry import box

from yabitsu import valley_fit


def test_changed_vertex_hole_checks_all_six_neighbors():
    z = np.full((7, 7), 10.)
    after = z.copy()
    after[2, 2] = 8
    after[4, 3] = 11
    after[5, 5] = 7
    before = {"xs": np.arange(7.), "ys": np.arange(7.), "Z": z}
    issues, _ = valley_fit.grid_checks(before, {**before, "Z": after}, {"A": box(0, 0, 4, 5)})
    assert {(i.category, i.xy) for i in issues} == {
        ("terrain_hole", (2., 2.)), ("terrain_hole", (5., 5.)),
        ("terrain_raised", (3., 4.)), ("terrain_outside", (5., 5.)),
    }


def test_mesh_audit_checks_lower_edges_both_faces_and_skirt():
    ground = lambda x, y: np.zeros_like(x)
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    post = np.array([[0, 0, .4], [.1, 0, .4], [.1, .1, .4], [0, .1, .4]])
    wall = np.array([[0, 0, -.5], [0, 1, -.5], [0, 1, 1], [0, 0, 1]])
    skirt = wall + [1, 0, .7]
    layers = [
        valley_fit.Mesh("mt_smallspan_02_left_start_post", post, faces),
        valley_fit.Mesh("smallspan_02_abutment_start", wall, faces),
        valley_fit.Mesh("smallspan_02_sill", wall + [.015, 0, 0], faces),
        valley_fit.Mesh("1ROAD_s1_gutterR_00", skirt, faces),
    ]
    issues = valley_fit.mesh_checks(layers, box(-2, -2, 2, 2), ground, ground)
    assert any(i.category == "foot" and i.mesh == layers[0].name for i in issues)
    assert any(i.category == "skirt" and i.mesh == layers[-1].name for i in issues)
    assert any(i.category == "overlap" and {i.mesh, i.other} ==
               {layers[1].name, layers[2].name} for i in issues)


def test_outside_comparison_sees_uv_group_and_crossing_triangle():
    verts = np.array([[0, 0, 0], [2, 0, 0], [0, 2, 0]], float)
    faces = np.array([[0, 1, 2]])
    outside = box(-1, -1, 1, 1)
    old = valley_fit.Mesh("part", verts, faces, uv=np.zeros((3, 2)), group="s4")
    new = valley_fit.Mesh("part", verts, faces, uv=np.ones((3, 2)), group="s5")
    assert valley_fit.outside_changes([old], [old], outside) == []
    assert [i.mesh for i in valley_fit.outside_changes([old], [new], outside)] == ["part"]


def test_riverbed_audit_finds_float_between_vertices():
    old = lambda x, y: np.ones_like(x)
    ground = lambda x, y: np.minimum(abs(x - 1), 1)
    verts = np.array([[0, 0, 1], [2, 0, 1], [2, 1, 1]], float)
    mesh = valley_fit.Mesh("riverbed_012_00", verts, np.array([[0, 1, 2]]), material="riverbed")
    issues = valley_fit.mesh_checks([mesh], shapely.box(-1, -1, 3, 2), ground, old)
    assert max(i.amount for i in issues if i.category == "water_float") >= .95


def test_tree_plan_uses_changed_triangles_and_does_not_fix_existing_float():
    from scripts.align_valley_trees import plan_forest_changes
    from yabitsu.valleys import ChangedGround

    old = np.array([[[0., 0., 0.], [2., 0., 0.], [0., 2., 0.]]])
    new = old.copy()
    new[..., 2] = -1.
    dug = ChangedGround(old, new)
    trees = np.rec.fromarrays([[.5, 1.5, .75], [.5, 1.5, .75], [0., 2., 0.]],
                              names="x,y,z")
    empty = valley_fit.Surfaces(np.empty((0, 3, 3)))
    wet = old.copy()
    wet[..., 2] = -.5
    plan = plan_forest_changes(trees, dug, [], (empty, valley_fit.Surfaces(wet)))
    assert [(c.index, c.new_ground, c.reason) for c in plan] == [(0, -1., "水面"), (2, -1., "水面")]


def test_material_fragments_share_the_original_foot():
    import shapely

    # Match the two layers in XY so that only the lower slab is the foot.
    top = np.array([[0., 0., 1.], [1., 0., 1.], [0., 1., 1.]])
    face = np.array([[0, 1, 2]])
    wall = "smallspan_16_rail_support_after"
    ground = lambda x, y: np.zeros_like(x)
    parts = [valley_fit.Mesh(wall, top, face),
             valley_fit.Mesh(wall + "_edit_bottom", top - [0, 0, 1], face, parent=wall)]
    area = shapely.box(-1, -1, 2, 2)
    assert not valley_fit.mesh_checks(parts, area, ground, ground, overlaps=False)
    parts[1].verts[:, 2] += .3
    result = valley_fit.mesh_checks(parts, area, ground, ground, overlaps=False)
    assert [(issue.category, issue.mesh) for issue in result] == [("foot", wall)]
    assert result[0].amount == pytest.approx(.3)
