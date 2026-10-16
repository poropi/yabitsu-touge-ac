import numpy as np
from pyproj import Transformer

ORIGIN_LAT, ORIGIN_LON = 35.42782, 139.21934  # ヤビツ峠
_fwd = Transformer.from_crs("EPSG:4326", "EPSG:6677", always_xy=True)
_inv = Transformer.from_crs("EPSG:6677", "EPSG:4326", always_xy=True)
_OX, _OY = _fwd.transform(ORIGIN_LON, ORIGIN_LAT)


def to_local(lat, lon):
    e, n = _fwd.transform(np.asarray(lon, float), np.asarray(lat, float))
    return e - _OX, n - _OY


def from_local(x, y):
    lon, lat = _inv.transform(np.asarray(x, float) + _OX, np.asarray(y, float) + _OY)
    return lat, lon
