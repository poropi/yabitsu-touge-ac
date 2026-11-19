
import numpy as np

from yabitsu import assembly, crosssection, small_bridges, track
from yabitsu.config import DATA_PROCESSED


def test_s5_width_table_preserves_bridge_and_spreads_the_step():
    sec = crosssection.Section.load_json(DATA_PROCESSED / "track_section.json")
    bounds = track.load_bounds()
    rows = crosssection.parse_width_rows(track.load_section_rows("widths", bounds))
    width, lanes = crosssection.widths(sec.s, sec.lanes, rows)
    old_rows = [{k: v for k, v in row.items() if k != "taper_m"} for row in rows]
    old_width, old_lanes = crosssection.widths(sec.s, sec.lanes, old_rows)
    local = sec.s - bounds["s5"][0]
    i, j = small_bridges.spans(sec, bounds, small_bridges.load())[25]

    np.testing.assert_array_equal(width[i:j + 1], 5.5)
    unchanged = (local <= 8960) | (local >= 9005)
    np.testing.assert_array_equal(width[unchanged], old_width[unchanged])
    np.testing.assert_array_equal(lanes, old_lanes)
    window = (local >= 8955) & (local <= 9010)
    rate = np.diff(width[window]) / (2 * np.diff(sec.s[window]))
    assert 0 < rate.max() <= .15
    assert rate.min() >= 0


class Meshes(dict):
    def add_mesh(self, group, name, verts, faces, uvs, material):
        self[name] = verts, faces, uvs, material


def test_s5_median_begins_past_bridge_and_keeps_full_band():
    sec = crosssection.Section.load_json(DATA_PROCESSED / "track_section.json")
    bounds = track.load_bounds()
    start = bounds["s5"][0]
    meshes = Meshes()
    assembly.add_centrelines(meshes, sec, track.load_section_rows("centrelines", bounds),
                             s_min=0)
    v, faces, _, material = meshes["track_centre_021_median"]
    i = np.searchsorted(sec.s, start + 8990) - 1
    head = np.array([sec.x[i], sec.y[i], sec.z[i]])
    assert np.min(np.linalg.norm(v[:, :2] - head[:2], axis=1)) < .02
    nose = v[np.argmin(np.linalg.norm(v[:, :2] - head[:2], axis=1))]
    assert abs(nose[2] - head[2]) <= .015001
    bridge_i, bridge_j = small_bridges.spans(sec, bounds, small_bridges.load())[25]
    from scipy.spatial import cKDTree
    nearest = cKDTree(np.c_[sec.x, sec.y]).query(v[:, :2])[1]
    assert nearest.min() >= i > bridge_j
    assert material == "concrete"
    assert meshes["LINES_track_centre_021"][3] == "line_orange"
    # 描画面を始端から常設帯までつなぎ、面積がゼロの面を作らない。
    from yabitsu import kn5
    triangles = v[kn5._triangles(faces)]
    double_area = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                         triangles[:, 2] - triangles[:, 0]), axis=1)
    assert double_area.min() > 0


def test_scope_checks_entire_terrain_triangle_and_excludes_unrelated_props():
    from scripts.check_valley_scope import Scope

    sec = crosssection.Section.load_json(DATA_PROCESSED / "track_section.json")
    bounds = track.load_bounds()
    rows = crosssection.parse_width_rows(track.load_section_rows("widths", bounds))
    bays = crosssection.parse_bay_rows(track.load_section_rows("busbays", bounds))
    crosssection.update_width_tapers(sec, rows, bays)
    near = dict(xs=np.arange(2.), ys=np.arange(2.), Z=np.zeros((2, 2)),
                D=np.zeros((2, 2)), H=np.zeros((1, 1), bool))
    scope = Scope(sec, bounds, near, near)
    row = int(np.argmin(abs(sec.s - bounds["s5"][0] - 8985)))
    p = np.array([sec.x[row], sec.y[row], sec.z[row]])
    h = sec.heading[row]
    normal = np.array([-np.sin(h), np.cos(h), 0.])
    along = np.array([np.cos(h), np.sin(h), 0.])
    p += (sec.hw_left[row] + 2.8) * normal
    inside = np.array([p, p + .1 * along, p + .1 * normal])
    crosses_limit = inside.copy()
    crosses_limit[2] += .4 * normal
    tris = np.stack([inside, crosses_limit])
    assert scope.taper_permits("1GRASS_s1_terrain_271", tris, "position").tolist() == [True, False]
    assert not scope.taper_permits("sign_other", tris, "position").any()


def test_median_clearance_distinguishes_close_layers_from_penetration():
    from yabitsu.valley_fit import overlap_checks

    asphalt = np.array([[[0., 0., 0.], [1., 0., 0.], [0., 2., 0.]]])
    median = np.array([[[0., .1, .015], [.7, .1, .015], [.1, 1., .015]]])
    base = ("1ROAD_s1_134", asphalt)
    assert not overlap_checks([base, ("track_centre_021_median", median)])
    intersecting = median.copy()
    intersecting[0, 0, 2] = -.002
    assert overlap_checks([base, ("track_centre_021_median", intersecting)])


def test_c_wave_connector_reaches_the_lattice_at_the_bridge_end():
    from yabitsu import features, landmarks

    sec = crosssection.Section.load_json(DATA_PROCESSED / "track_section.json")
    bounds = track.load_bounds()
    rows = crosssection.parse_width_rows(track.load_section_rows("widths", bounds))
    crosssection.update_width_tapers(sec, rows, ())
    bridge_i, j = small_bridges.spans(sec, bounds, small_bridges.load())[25]
    rail = features.Feature(bounds["s5"][0] + 8960, bounds["s5"][0] + 9480, "R", "guardrail", 0., "")
    beam, _, _ = small_bridges.w_beam_join(sec, j, rail, landmarks.load("r70_guardrail", "w_beam"))
    lattice, _, _ = small_bridges.rail_mesh(
        sec, bridge_i, j, -1, landmarks.load("r70_bridges", "steel_white_lattice_2m"), 2.)
    origin = np.array([sec.x[j], sec.y[j]])
    heading = sec.heading[j]
    normal = np.array([np.sin(heading), -np.cos(heading)])
    advance = np.array([np.cos(heading), np.sin(heading)])

    def joint_vertices(v):
        xy = v[:, :2] - origin
        hit = (abs(xy @ advance) < 1e-8) & (v[:, 2] > sec.edge_z(-1)[j] + .5)
        return xy[hit] @ normal

    assert abs(joint_vertices(beam).min() - joint_vertices(lattice).max()) <= .001
