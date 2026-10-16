import io
import math
import re

import numpy as np
import pytest
from PIL import Image

from yabitsu import gsi


def _enc(v):
    v = np.asarray(v, np.int64) % (2 ** 24)
    return np.stack([(v >> 16) & 255, (v >> 8) & 255, v & 255], -1).astype(np.uint8)


def test_decode_values():
    rgb = _enc(np.array([[76130, 2 ** 23, 2 ** 24 - 100]]))
    h = gsi.decode_dem_png(rgb)
    assert h[0, 0] == pytest.approx(761.30) and math.isnan(h[0, 1]) and h[0, 2] == pytest.approx(-1.0)


def test_tile_coords_reference_points():
    fx, fy = gsi.tile_coords(0.0, -180.0, 1)
    assert fx == pytest.approx(0) and fy == pytest.approx(1.0)


def _latlon_from_px(gx, gy, z):
    n = 2 ** z
    lon = gx / 256 / n * 360 - 180
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * gy / 256 / n))))
    return lat, lon


def _png(arr):
    b = io.BytesIO(); Image.fromarray(arr).save(b, "PNG"); return b.getvalue()


def _fake_fetch_linear(calls):
    def fetch(url):
        calls.append(url)
        z, x, y = map(int, re.findall(r"/(\d+)/(\d+)/(\d+)\.png", url)[0])
        px = np.tile(np.arange(256), (256, 1))
        return 200, _png(_enc(x * 256 + px))  # 高さ = 全体の x ピクセル番号 × 0.01m
    return fetch


def test_bilinear_is_exact_on_linear_field_across_tile_border(tmp_path):
    calls = []
    s = gsi.DemSampler(gsi.TileCache(tmp_path, fetch=_fake_fetch_linear(calls), delay=0), "dem5a_png", 15)
    gx0 = 29000 * 256
    for gx in (gx0 + 0.2, gx0 + 100.7, gx0 + 255.9):
        lat, lon = _latlon_from_px(gx, 12900 * 256 + 50.5, 15)
        assert s.sample(lat, lon)[0] == pytest.approx((gx - 0.5) * 0.01, abs=1e-6)


def test_cache_and_missing_tile(tmp_path):
    calls = []
    def fetch(url):
        calls.append(url); return 404, b""
    c = gsi.TileCache(tmp_path, fetch=fetch, delay=0)
    assert c.get_bytes("dem5a_png", 15, 1, 2, "png") is None
    assert c.get_bytes("dem5a_png", 15, 1, 2, "png") is None
    assert len(calls) == 1


def test_server_error_raises(tmp_path):
    c = gsi.TileCache(tmp_path, fetch=lambda url: (500, b""), delay=0)
    with pytest.raises(RuntimeError):
        c.get_bytes("dem5a_png", 15, 1, 2, "png")


class _Const:
    def __init__(self, v): self.v = v
    def sample(self, lat, lon): return np.full(np.shape(np.atleast_1d(lat)), self.v, float)


def test_fallback_fills_nan():
    out = gsi.FallbackSampler([_Const(np.nan), _Const(5.0)]).sample(np.array([1.0, 2.0]), np.array([3.0, 4.0]))
    assert np.allclose(out, 5.0)
