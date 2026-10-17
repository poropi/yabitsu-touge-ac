import numpy as np
import pytest

from yabitsu import crosssection as cs
from yabitsu.centerline import Centerline


def _straight(n=301, lanes=2, k=0.0):
    s = np.arange(float(n))
    return Centerline(s, s.copy(), np.zeros(n), np.zeros(n), np.full(n, k), np.full(n, lanes, np.int32), np.zeros(n, bool), 1.0)


def test_straight_two_lane_has_crown():
    slope, crown = cs.cross_slope(np.zeros(300), np.full(300, 2), 1.0)
    assert np.allclose(slope, 0) and np.allclose(crown, 0.02)
    assert cs.surface_height(3.0, 0.0, slope[0], crown[0]) == pytest.approx(-0.06)
    assert cs.surface_height(-3.0, 0.0, slope[0], crown[0]) == pytest.approx(-0.06)


def test_left_curve_banks_inward():
    slope, crown = cs.cross_slope(np.full(300, 1 / 20), np.full(300, 2), 1.0)
    i = 150
    assert slope[i] == pytest.approx(-0.06, abs=1e-6) and crown[i] == pytest.approx(0.0, abs=1e-6)
    assert cs.surface_height(3.0, 0.0, slope[i], crown[i]) < cs.surface_height(-3.0, 0.0, slope[i], crown[i])


def test_one_lane_straight_has_min_cross_slope():
    slope, crown = cs.cross_slope(np.zeros(300), np.ones(300), 1.0)
    assert np.allclose(np.abs(slope[50:250]), 0.02, atol=1e-6) and np.allclose(crown, 0)


def test_no_step_in_surface_between_rows():
    k = np.r_[np.zeros(150), np.full(150, 1 / 25)]
    slope, crown = cs.cross_slope(k, np.full(300, 2), 1.0)
    edge = cs.surface_height(3.0, 0.0, slope, crown)
    assert np.abs(np.diff(edge)).max() < 0.01


def test_widths_default_and_override():
    s = np.arange(0.0, 100.0); lanes = np.r_[np.ones(50), np.full(50, 2)].astype(int)
    w, ln = cs.widths(s, lanes, [{"s_from": 60.0, "s_to": 80.0, "width_m": 5.2, "lanes": 2}])
    assert w[10] == 4.0 and w[55] == 6.0 and w[70] == 5.2 and ln[70] == 2


def test_widths_csv_missing(tmp_path):
    assert cs.load_widths_csv(tmp_path / "none.csv") == []


def test_section_round_trip(tmp_path):
    cl = _straight()
    sec = cs.build_section(cl, np.full(cl.s.shape, 100.0), [])
    sec.save_json(tmp_path / "s.json")
    s2 = cs.Section.load_json(tmp_path / "s.json")
    assert np.allclose(s2.hw_left, 3.0) and s2.n == cl.s.size and s2.lanes.dtype.kind == "i"
    assert s2.edge_z(1)[0] == pytest.approx(100 - 0.06)


def test_pads_roles():
    sec = cs.build_section(_straight(), np.full(301, 100.0), [])
    parking = [{"name": "A", "lat": [0, 0, 10, 10, 0], "lon": [-30, -10, -10, -30, -30]},
               {"name": "B", "lat": [0, 0, 10, 10, 0], "lon": [310, 330, 330, 310, 310]},
               {"name": "far", "lat": [900, 900, 910, 910, 900], "lon": [0, 10, 10, 0, 0]}]
    pads = cs.pads_from_parking(parking, sec, project=lambda lat, lon: (np.asarray(lon, float), np.asarray(lat, float)))
    assert [(p["name"], p["role"]) for p in pads] == [("A", "start"), ("B", "end")]
    assert pads[0]["z"] == pytest.approx(100.0)


def test_width_rows_parse_shifted_rows():
    rows = [{"s_from": 300.0, "s_to": 350.0, "width_m": "7.0", "lanes": "2", "note": ""},
            {"s_from": 350.0, "s_to": 400.0, "width_m": "5.0", "lanes": "", "note": ""}]
    got = cs.parse_width_rows(rows)
    assert got == [{"s_from": 300.0, "s_to": 350.0, "width_m": 7.0, "lanes": 2},
                   {"s_from": 350.0, "s_to": 400.0, "width_m": 5.0, "lanes": None}]


def test_parking_near_the_middle_of_the_road_is_kept():
    from tests.test_meshgen import _straight
    from yabitsu import geo
    sec = _straight(1001)
    poly = [[500, 10], [520, 10], [520, 20], [500, 20], [500, 10]]
    far = [[500, 400], [520, 400], [520, 420], [500, 420], [500, 400]]
    def ll(pts):
        lat, lon = geo.from_local(*np.asarray(pts, float).T)
        return list(map(float, lat)), list(map(float, lon))
    parking = [{"name": "mid", "lat": ll(poly)[0], "lon": ll(poly)[1]}, {"name": "far", "lat": ll(far)[0], "lon": ll(far)[1]}]
    pads = cs.pads_from_parking(parking, sec)
    assert [p["name"] for p in pads] == ["mid"] and abs(pads[0]["z"] - sec.z[510]) < 1e-6
