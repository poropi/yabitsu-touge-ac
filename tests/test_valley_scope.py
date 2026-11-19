import numpy as np
import pytest

from yabitsu import valley_fit


def test_fit_detects_worsening_at_a_different_location():
    before = [valley_fit.Issue("foot", "wall", (1., 2.), .4)]
    after = [valley_fit.Issue("foot", "wall", (1., 2.), .4),
             valley_fit.Issue("foot", "wall", (4., 2.), .2)]
    existing, regressions = valley_fit.issue_changes(before, after)
    assert existing == before
    assert regressions == after[1:]


def test_scope_uses_drawn_triangles_and_not_whole_cells():
    from scripts.check_valley_scope import changed_ground

    old = {"xs": np.arange(3.), "ys": np.arange(3.), "Z": np.zeros((3, 3)),
           "D": np.zeros((3, 3)), "H": np.zeros((2, 2), bool)}
    new = {**old, "Z": old["Z"].copy()}
    new["Z"][0, 1] = -1
    dug = changed_ground(old, new)
    assert dug.covers([.75, .25])
    assert not dug.covers([.25, .75])
    assert dug.height([.75, .25]) == pytest.approx(-.5)


def test_scope_triangle_comparison_preserves_uv_winding_normals_and_duplicates():
    from scripts.check_valley_scope import triangle_changes

    a = np.arange(24, dtype=float).reshape(1, 3, 8)
    assert not len(triangle_changes(a, np.roll(a, 1, axis=1)))
    assert len(triangle_changes(a, a[:, ::-1])) == 1
    assert len(triangle_changes(np.repeat(a, 2, axis=0), a)) == 1
    for column in (0, 3, 5):
        b = a.copy()
        b[0, 0, column] += .1
        assert len(triangle_changes(a, b)) == 1


def test_bridge_skirt_keeps_visible_position_and_uv():
    from yabitsu.meshgen import gutter

    s = np.arange(4.)
    from types import SimpleNamespace

    sec = SimpleNamespace(x=s, y=s * .5, z=s * .1, heading=s * .03, hw_right=s * 0 + 3.,
                          slope=s * .02, crown=s * 0 + .02, s=s)
    v, _, uv = gutter(sec, slice(None), -1)
    v1, _, uv1 = gutter(sec, slice(None), -1, skirt_depth=np.array([1., .4, .4, 1.]))
    assert np.array_equal(v[1::4], v1[1::4])
    assert np.array_equal(uv, uv1)


def test_build_recovers_saved_grid_delta_and_rejects_a_stale_delta(tmp_path):
    from yabitsu import valleys

    path = tmp_path / "track_terrain_near.npz"
    old = np.zeros((3, 3))
    new = {"xs": np.arange(3.), "ys": np.arange(3.), "Z": old.copy()}
    new["Z"][1, 1] = -2.
    valleys.save_ground_delta(path, valleys.ground_delta(old, new))
    assert np.array_equal(valleys.before_for_build(None, path, new)["Z"], old)
    new["Z"][0, 0] = -1.
    with pytest.raises(ValueError, match="近景"):
        valleys.before_for_build(None, path, new)


def test_shared_image_overwrite_is_visible_even_when_material_path_is_identical(tmp_path):
    from scripts import check_valley_scope as scope

    before, after = [tmp_path / name for name in ("old", "new")]
    for root in (before, after):
        (root / "extension").mkdir(parents=True)
        (root / "textures").mkdir()
        (root / "textures/forest.png").write_bytes(b"same")
    (before / "extension/kenchi.dds").write_bytes(b"old")
    (after / "extension/kenchi.dds").write_bytes(b"new")
    (after / "extension/valley_kenchi.dds").write_bytes(b"extra")
    assert list(scope.overwritten_images(before, after)) == ["extension/kenchi.dds"]


def test_new_image_cannot_reuse_a_stem_with_a_different_suffix(tmp_path):
    from scripts import check_valley_scope as scope

    before, after = tmp_path / "before", tmp_path / "after"
    for root in (before, after):
        (root / "textures").mkdir(parents=True)
        (root / "textures" / "kenchi.png").write_bytes(b"old")
    (after / "textures" / "KENCHI.dds").write_bytes(b"new")
    assert list(scope.overwritten_images(before, after)) == ["textures/KENCHI.dds"]
