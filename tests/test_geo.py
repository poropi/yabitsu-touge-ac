import numpy as np
from yabitsu import geo


def test_origin_is_zero():
    x, y = geo.to_local(geo.ORIGIN_LAT, geo.ORIGIN_LON)
    assert abs(x) < 1e-6 and abs(y) < 1e-6


def test_axes_point_north_and_east():
    x, y = geo.to_local(geo.ORIGIN_LAT + 0.001, geo.ORIGIN_LON)
    assert abs(x) < 0.5 and 110.0 < y < 112.0
    x, y = geo.to_local(geo.ORIGIN_LAT, geo.ORIGIN_LON + 0.001)
    assert 90.0 < x < 91.5 and abs(y) < 0.5


def test_round_trip_arrays():
    lat = np.array([35.41025, 35.42782])
    lon = np.array([139.20926, 139.21934])
    x, y = geo.to_local(lat, lon)
    lat2, lon2 = geo.from_local(x, y)
    assert np.allclose(lat, lat2, atol=1e-9) and np.allclose(lon, lon2, atol=1e-9)
