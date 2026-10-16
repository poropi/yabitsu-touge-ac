import io
import math
import time
from pathlib import Path

import numpy as np
import requests
from PIL import Image

from .config import DATA_RAW

GSI_URL = "https://cyberjapandata.gsi.go.jp/xyz/{layer}/{z}/{x}/{y}.{ext}"
NA = 2 ** 23


def decode_dem_png(rgb):
    rgb = np.asarray(rgb).astype(np.int64)
    v = rgb[..., 0] * 65536 + rgb[..., 1] * 256 + rgb[..., 2]
    h = np.where(v < NA, v, v - 2 ** 24).astype(np.float64) * 0.01
    h[v == NA] = np.nan
    return h


def tile_coords(lat, lon, z):
    n = 2 ** z
    fx = (np.asarray(lon, float) + 180.0) / 360.0 * n
    fy = (1.0 - np.arcsinh(np.tan(np.radians(np.asarray(lat, float)))) / math.pi) / 2.0 * n
    return fx, fy


def _http_get(url):
    r = requests.get(url, timeout=30, headers={"User-Agent": "yabitsu-track-builder"})
    return r.status_code, r.content


class TileCache:
    def __init__(self, root=DATA_RAW / "gsi", fetch=None, delay=0.05):
        self.root, self.fetch, self.delay = Path(root), fetch or _http_get, delay

    def path(self, layer, z, x, y, ext):
        return self.root / layer / str(z) / str(x) / f"{y}.{ext}"

    def get_bytes(self, layer, z, x, y, ext):
        p = self.path(layer, z, x, y, ext)
        missing = p.with_name(p.name + ".missing")
        if p.exists():
            return p.read_bytes()
        if missing.exists():
            return None
        url = GSI_URL.format(layer=layer, z=z, x=x, y=y, ext=ext)
        status, content = self.fetch(url)
        if self.delay:
            time.sleep(self.delay)
        p.parent.mkdir(parents=True, exist_ok=True)
        if status == 404:
            missing.touch(); return None
        if status != 200:
            raise RuntimeError(f"国土地理院タイルの取得に失敗 ({status}): {url}")
        p.write_bytes(content)
        return content


class DemSampler:
    def __init__(self, cache, layer="dem5a_png", z=15):
        self.cache, self.layer, self.z, self._tiles = cache, layer, z, {}

    def _tile(self, tx, ty):
        k = (tx, ty)
        if k not in self._tiles:
            b = self.cache.get_bytes(self.layer, self.z, tx, ty, "png")
            self._tiles[k] = (np.full((256, 256), np.nan) if b is None
                              else decode_dem_png(np.array(Image.open(io.BytesIO(b)).convert("RGB"))))
        return self._tiles[k]

    def _pixels(self, gx, gy):
        tx, ty = np.floor_divide(gx, 256), np.floor_divide(gy, 256)
        px, py = gx - tx * 256, gy - ty * 256
        out = np.full(gx.shape, np.nan)
        for a, b in np.unique(np.stack([tx.ravel(), ty.ravel()], 1), axis=0):
            m = (tx == a) & (ty == b)
            out[m] = self._tile(int(a), int(b))[py[m], px[m]]
        return out

    def sample(self, lat, lon):
        lat = np.atleast_1d(np.asarray(lat, float)); lon = np.atleast_1d(np.asarray(lon, float))
        fx, fy = tile_coords(lat, lon, self.z)
        gx, gy = fx * 256 - 0.5, fy * 256 - 0.5
        x0, y0 = np.floor(gx).astype(np.int64), np.floor(gy).astype(np.int64)
        wx, wy = gx - x0, gy - y0
        v00, v10 = self._pixels(x0, y0), self._pixels(x0 + 1, y0)
        v01, v11 = self._pixels(x0, y0 + 1), self._pixels(x0 + 1, y0 + 1)
        return v00 * (1 - wx) * (1 - wy) + v10 * wx * (1 - wy) + v01 * (1 - wx) * wy + v11 * wx * wy


class FallbackSampler:
    def __init__(self, samplers):
        self.samplers = samplers

    def sample(self, lat, lon):
        lat = np.atleast_1d(np.asarray(lat, float)); lon = np.atleast_1d(np.asarray(lon, float))
        out = self.samplers[0].sample(lat, lon)
        for s in self.samplers[1:]:
            m = np.isnan(out)
            if m.any():
                out[m] = s.sample(lat[m], lon[m])
        return out


def default_sampler(cache=None):
    cache = cache or TileCache()
    return FallbackSampler([DemSampler(cache, "dem5a_png", 15), DemSampler(cache, "dem10b_png", 14)])
