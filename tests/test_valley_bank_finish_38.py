import numpy as np

from scripts import dress_mountain as dress
from tests.test_meshgen import _straight
from yabitsu import features, kn5, meshgen, small_bridges
from yabitsu.crosssection import GUTTER_WIDTH


def test_white_wave_beam_only_in_river_table_range():
    rails = dress.ranges({"s1": (0,1), "s2": (0,1), "s3": (0,1), "s4": (0,8000), "s5": (10000,20000)},
                         dress.table_rows(), "rail")
    assert dress.override(6354, rails, "R") == "wbeam"
    assert dress.override(6200, rails, "R") == "wbeam_aged"


def test_rail_revetment_has_battered_faces_and_unrounded_crest():
    sec = _straight()
    feature = features.Feature(30., 150., "R", "guardrail", 0., "")
    ground = lambda x, y: np.zeros_like(x) + 94.
    parts = list(small_bridges.rail_support_meshes(sec, feature, (75,82), ground))
    assert len(parts) == 2
    for name, v, f, uv, mat, draw in parts:
        assert mat == "kenchi_shade" and draw
        n = meshgen.face_normals(v, kn5._triangles(f))
        assert np.any((abs(n[:, 1]) > .7) & (n[:, 2] > .20) & (n[:, 2] < .65))
        tri = kn5._triangles(f)
        normal = kn5._normals(v @ kn5.C.T, tri) @ kn5.C
        assert np.min(normal[tri[n[:, 2] > .97], 2]) > .97


def test_367_gravel_covers_the_wall_toe_outside_the_asphalt():
    sec = _straight()
    feat = features.Feature(30., 49., "L", "retaining_wall", 4., "")
    v, f, uv = small_bridges.wall_367_gravel(sec, feat)
    side = v.reshape(-1, 3, 3)[:, :, 1]
    assert (side[:, 0] > sec.hw_left[30]).all()
    assert (side[:, -1] > sec.hw_left[30] + GUTTER_WIDTH).all()
    assert np.all((side[:, -1] - side[:, 0] > .35) & (side[:, -1] - side[:, 0] < .65))
    assert (meshgen.face_normals(v, f)[:, 2] > .7).all()
