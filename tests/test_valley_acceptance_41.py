import numpy as np

from yabitsu import features, small_bridges
from yabitsu.crosssection import Section, GUTTER_WIDTH


def test_collision_reaches_retaining_wall_first_section():
    stations = np.arange(20.)
    zeros = np.zeros_like(stations)
    widths = np.full_like(stations, 3.)
    sec = Section(stations, stations.copy(), zeros, zeros, zeros, zeros, widths, zeros, widths, widths, zeros, zeros)
    retaining = features.Feature(9.1, 18., "L", "retaining_wall", 4., "")
    wall, _, _ = features.retaining_wall_mesh(sec, retaining)
    vertices, _, _ = small_bridges.collision_mesh(
        sec, 3, 8, 1, [], .37, retaining_after=(10, wall[:2]),
    )
    assert vertices[-2, 0] == vertices[-1, 0] == 10.
    np.testing.assert_allclose(vertices[-2, 1], 3. + GUTTER_WIDTH)


def test_acceptance_limits_and_exceptions():
    from scripts.valley_acceptance import fold_failures, pit_failures, shoulder_outside_height

    deg = np.array([125., 35., 35.001, 60., 60.001, 36., 36., 36.])
    station = np.array([8964., 8940., 8940., 8981., 8981., 9006.1, 8961.9, 8980.])
    right = np.array([False, False, False, True, True, True, True, False])
    hidden = np.arange(len(deg)) == 0
    np.testing.assert_array_equal(fold_failures(deg, hidden, station, right),
                                  [False, False, True, False, True, True, True, True])
    np.testing.assert_array_equal(
        pit_failures(np.array([.002, .002, .001]), np.array([True, False, False])),
        [False, True, False],
    )
    actual = np.array([-.97, -.973, -.20, -.20, .051, -.97, -.17, -.97])
    before = np.array([-.97, -.97, -.18, -.20, -.97, -.97, -.07, np.nan])
    station = np.array([9009., 9010., 9009.25, 9008.99, 9009.25, 9010.01, 9009.25, 9010.])
    failures, exceptions = shoulder_outside_height(actual, before, station)
    np.testing.assert_array_equal(failures, [False, True, True, True, True, True, True, True])
    np.testing.assert_array_equal(exceptions, [True, True, True, False, True, False, False, False])


def test_water_keeps_the_old_cut_height_and_blends_away_from_the_old_bank():
    from yabitsu.water import seam_bank_height

    calculated = seam_bank_height(
        np.zeros(5), np.ones(5),
        np.array([0., .8, .8, .8, .8]),
        np.array([.04, .00001, .10, .40, .64]), 1.,
    )
    assert calculated[0] == 1.
    assert calculated[1] < .001
    assert 0. < calculated[2] < calculated[3] < calculated[4] == 1.


def test_a_shoulder_replacement_does_not_move_b_faces_to_another_part():
    from yabitsu.valley_surface import mesh_parts

    v = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.],
                  [0., 3000., 0.], [1., 3000., 0.], [0., 3001., 0.]])
    f = np.vstack((np.tile([0, 1, 2], (15999, 1)), [[3, 4, 5], [3, 4, 5]]))
    original = dict(valley_vertices=v, valley_faces=f, valley_bed=np.zeros(len(f), bool))
    replacement = {**original, "valley_faces": f[1:], "valley_bed": original["valley_bed"][1:],
                   "valley_source_faces": np.arange(1, len(f))}
    before = [(n, faces) for n, _, faces, *_ in mesh_parts(original) if "_B_" in n]
    after = [(n, faces) for n, _, faces, *_ in mesh_parts(replacement) if "_B_" in n]

    assert [name for name, _ in before] == [name for name, _ in after]
    for (_, old), (_, new) in zip(before, after):
        np.testing.assert_array_equal(old, new)
