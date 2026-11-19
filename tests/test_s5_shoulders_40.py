import numpy as np
import shapely

from yabitsu import valley_fit, valleys
from yabitsu import valley_surface


def test_s5_shoulder_replaces_one_ground_sheet_and_preserves_the_old_cell_outline():
    # Two metres of ground under the old road edge, with a road-height shoulder
    # required in the middle and exact, unchanged boundaries.
    grid = {
        "xs": np.arange(0., 8., 2.),
        "ys": np.arange(0., 8., 2.),
        "Z": np.full((4, 4), -2.),
        "H": np.zeros((3, 3), bool),
        "D": np.zeros((4, 4)),
    }
    fine = dict(valley_cells=np.empty((0, 2), int), valley_vertices=np.empty((0, 3)),
                valley_faces=np.empty((0, 3), int), valley_bed=np.empty(0, bool))
    xyz = np.array([[[1., 1., -2.], [3., 1., -2.], [5., 1., -2.]],
                    [[1., 2., -2.], [3., 2., -.08], [5., 2., -2.]],
                    [[1., 4., -2.], [3., 4., -.08], [5., 4., -2.]],
                    [[1., 5., -2.], [3., 5., -2.], [5., 5., -2.]]])
    z = valley_fit.Surfaces(valley_fit.terrain_triangles(grid, shapely.box(0, 0, 6, 6)))
    point = np.array([[3.015, 3.031]])
    assert z.height(point)[0] == -2.
    weights = np.zeros(xyz.shape[:2])
    weights[1:3, 1] = 1.
    xyz[:, :, 2] = -.08
    xyz = np.dstack((xyz, weights))
    result = valley_surface.replace_shoulder_ribbons(grid, fine, [xyz])
    tri = valley_fit.terrain_triangles({**grid, **result}, shapely.box(0, 0, 6, 6))
    shapes = shapely.polygons(tri[..., :2])
    assert sum(shapely.contains_xy(shapes, *point[0])) == 1
    np.testing.assert_allclose(valley_fit.Surfaces(tri).height(point), [-.0944])
    assert shapely.union_all(shapes).equals(shapely.box(0, 0, 6, 6))
    assert abs(shapely.area(shapes).sum() - 36.) < 1e-8
    unchanged = np.array([[.9, 3.], [5.1, 3.], [3., .9], [3., 5.1]])
    np.testing.assert_allclose(valley_fit.Surfaces(tri).height(unchanged), -2.)
    changed = valleys.changed_ground(grid, {**grid, **result})
    assert changed.covers(point).all()
    assert not changed.covers(unchanged).any()
