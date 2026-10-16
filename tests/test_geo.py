import numpy as np
from yabitsu import geo


def test_origin_is_zero():
    x, y = geo.to_local(geo.ORIGIN_LAT, geo.ORIGIN_LON)
    assert abs(x) < 1e-6 and abs(y) < 1e-6


def test_axes_point_north_and_east():
    # 第IX系の中央経線（139°50′）より西にあるため、座標の北は真北から約0.36°傾く（111m で約0.7m）
    x, y = geo.to_local(geo.ORIGIN_LAT + 0.001, geo.ORIGIN_LON)
    assert 0.5 < x < 0.9 and 110.0 < y < 112.0
    x, y = geo.to_local(geo.ORIGIN_LAT, geo.ORIGIN_LON + 0.001)
    assert 90.0 < x < 91.5 and -0.8 < y < -0.4


def test_round_trip_arrays():
    lat = np.array([35.41025, 35.42782])
    lon = np.array([139.20926, 139.21934])
    x, y = geo.to_local(lat, lon)
    lat2, lon2 = geo.from_local(x, y)
    assert np.allclose(lat, lat2, atol=1e-9) and np.allclose(lon, lon2, atol=1e-9)
