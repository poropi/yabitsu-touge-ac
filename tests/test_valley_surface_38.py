import numpy as np
from shapely.geometry import box

from scripts.check_valley_scope import Scope, triangle_keys
from yabitsu import terrain, valleys, valley_fit


def patch():
    grid = {"xs": np.arange(4.) * 2, "ys": np.arange(4.) * 2,
            "Z": np.ones((4, 4)), "D": np.zeros((4, 4)), "H": np.zeros((3, 3), bool)}
    pts = np.array([[2, 2, 1], [4, 2, 1], [4, 4, 1], [2, 4, 1], [3, 3, -2.]])
    faces = np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]])
    return grid, {**grid, "valley_cells": np.array([[1, 1]]), "valley_vertices": pts,
                  "valley_faces": faces, "valley_bed": np.zeros(len(faces), bool)}


def test_scope_compares_removed_and_added_terrain_against_exact_mesh():
    base, after = patch()
    fine = after["valley_vertices"][after["valley_faces"]]
    scope = Scope.__new__(Scope)
    scope.tapers = []
    scope.edited_parts = {}
    scope.local = lambda xy: (np.zeros(xy.shape[:-1]), np.zeros(xy.shape[:-1]))
    scope.terrain_keys = [triangle_keys(fine), triangle_keys(fine)]
    for name, kind, removed in (("1GRASS_s1_terrain_123", "removed", True),
                                ("1GRASS_valley_000_bed", "added", False)):
        assert scope.permits(name, fine, kind, removed=removed).all()
        shifted = fine.copy()
        shifted[:, 0, 0] -= .001
        assert not scope.permits(name, shifted, kind, removed=removed).any()


def test_ground_collision_and_tree_fit_share_replacement_surface():
    base, after = patch()
    assert terrain.terrain_mesh_z(after, base)(3., 3.) == -2.
    dug = valleys.changed_ground(base, after)
    assert dug.height([3., 3.]) == -2.
    assert dug.old.height(np.array([[3., 3.]]))[0] == 1.
    tri = valley_fit.terrain_triangles(after, box(2.01, 2.01, 3.99, 3.99))
    assert len(tri) == 4
    # An old face left under the new patch would produce a second ray hit at +1.
    assert valley_fit.Surfaces(tri).height(np.array([[3., 3.]]))[0] == -2.


def test_refinement_keeps_single_unsplit_boundary_edges_and_exact_cell_area():
    from yabitsu import valleys
    assert hasattr(valleys, "fine_topology"), "2 mセル外縁を保つ細分関数が必要"
    cells = np.array([[2, 3], [2, 4], [3, 4]])
    rc, faces = valleys.fine_topology(cells)
    edge = np.sort(faces[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1)
    edges, n = np.unique(edge, axis=0, return_counts=True)
    assert (n <= 2).all()
    assert len(edges[n == 1]) == 8
    assert np.all(rc[edges[n == 1]] % 4 == 0)
    xy = rc[faces][:, :, ::-1] / 2.
    d = xy[:, 1:] - xy[:, :1]
    area = (d[:, 0, 0] * d[:, 1, 1] - d[:, 0, 1] * d[:, 1, 0]) / 2
    assert area.min() > 0
    assert area.sum() == 12.


def test_terrain_export_replaces_both_faces_and_freezes_neighbors(tmp_path):
    from yabitsu import assembly, kn5
    from yabitsu.manifest import Manifest
    base, near = patch()
    far = {**base, "Dc": np.zeros((3, 3))}
    original, new = [Manifest(tmp_path / name) for name in ("original", "new")]
    assembly.add_terrain(original, base, far)
    assembly.add_terrain(new, near, far, before_near=base)
    name = "1GRASS_s1_terrain_000"
    from scripts.check_valley_scope import mesh_data
    old = mesh_data(original.root, original.meshes[name])
    keep = mesh_data(new.root, new.meshes[name])
    assert len(keep) == len(old)-2
    assert sorted(triangle_keys(keep).tolist()) == sorted(triangle_keys(old[~np.all(
        (old[:,:,:2]>=2) & (old[:,:,:2]<=4),axis=(1,2))]).tolist())
    parts = [mesh_data(new.root,e)[:,:,:3] for n,e in new.meshes.items() if n.startswith("1GRASS_valley_")]
    assert len(np.concatenate(parts)) == 4


def test_scope_rejects_an_undeclared_patch_or_a_hole_even_inside_a_site():
    base, near = patch()
    sites = {"A": box(0,0,6,6)}
    for change in ({"valley_faces":near["valley_faces"][:-1], "valley_bed":near["valley_bed"][:-1]},
                   {"valley_cells":np.array([[0,0]])},
                   {"valley_vertices":near["valley_vertices"]+np.array([0,0,.01])}):
        issues, _ = valley_fit.grid_checks(base,{**near,**change},sites)
        assert issues


def test_foot_straddling_a_dig_is_lowered_only_on_the_affected_segment():
    base, near = patch()
    dug = valleys.changed_ground(base,near)
    chain = np.array([[2.,3.,.8],[4.,3.,.8],[6.,3.,.8]])
    assert hasattr(valleys,"fit_foot_chain")
    bottom = valleys.fit_foot_chain(chain,dug,.1)
    assert bottom[0] < -2 and bottom[1] < -2
    assert bottom[2] == .8


def test_valley_deck_and_abutment_have_distinct_weathered_finishes(tmp_path):
    from yabitsu import config, small_bridges, track
    from yabitsu.crosssection import Section
    from yabitsu.manifest import Manifest
    sec = Section.load_json(config.DATA_PROCESSED / "track_section.json")
    spans = small_bridges.spans(sec,track.load_bounds(),small_bridges.load())
    parts = {n:mat for n,v,f,uv,mat,show in small_bridges.support_meshes(sec,spans,lambda x,y:x*0+250)}
    for k in (2,16,25):
        assert parts[f"smallspan_{k:02d}_deck"] == "valley_soffit"
        assert parts[f"smallspan_{k:02d}_abutment_end"] == "concrete_moss"
    m = Manifest(tmp_path)
    small_bridges.add_valley_materials(m)
    for mat in ("valley_rusted_steel","valley_soffit","valley_forest_floor"):
        assert (tmp_path / m.materials[mat]["texture"]).is_file()


def test_bridge_specific_finishes_are_measured_separately():
    from yabitsu import small_bridges
    rows = small_bridges.load()["bridges"]
    assert rows[2]["tint"].get("moss")
    assert rows[25].get("tint",{}).get("chalk")


def test_foot_samples_compare_the_same_xy_after_a_skirt_moves():
    edge = np.array([[[0.,0.,5.],[1.,0.,4.]]])
    after = edge.copy()
    after[0,1,2] -= 2
    old = valley_fit._edge_points(edge)
    new = valley_fit._edge_points(after)
    np.testing.assert_array_equal(old[:,:2],new[:,:2])


def test_bank_seam_retains_the_coarse_slope_within_the_fold_limit():
    from types import SimpleNamespace
    from yabitsu import valley_surface

    coords = np.arange(0., 10., 2.)
    grid = dict(xs=coords, ys=coords, Z=np.zeros((5, 5)), D=np.zeros((5, 5)),
                H=np.zeros((4, 4), bool))
    cells = np.array([[1, 1], [1, 2], [2, 1], [2, 2]])
    rc, faces = valleys.fine_topology(cells)
    vertices = np.c_[rc[:, 1]/2, rc[:, 0]/2, np.zeros(len(rc))]
    vertices[np.all(vertices[:, :2] == [3., 2.5], axis=1), 2] = .8
    sec = SimpleNamespace(x=np.array([-30., -10.]), y=np.array([-10., -10.]),
                          s=np.array([0.,20.]), heading=np.zeros(2), z=np.zeros(2),
                          hw_left=np.ones(2)*2., hw_right=np.ones(2)*2.)
    boundary = np.isin(vertices[:, :2], [2., 6.]).any(axis=1)
    result = valley_surface.blend_outer_slopes(sec, grid, vertices, faces, boundary)
    assert np.array_equal(result[boundary], vertices[boundary, 2])
    tip = np.all(vertices[:, :2] == [3., 2.5], axis=1)
    assert .01 < abs(result[tip].item()) < .5 * np.tan(np.deg2rad(35))
