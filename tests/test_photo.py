import io
import re

import numpy as np
from PIL import Image

from yabitsu import gsi, photo


def _fake_tile_fetch(url):
    z, x, y = map(int, re.findall(r"/(\d+)/(\d+)/(\d+)\.png", url)[0])
    b = io.BytesIO(); Image.new("RGB", (256, 256), ((y >> 16) & 255, (y >> 8) & 255, y & 255)).save(b, "PNG")
    return 200, b.getvalue()  # 色にタイル番号 y をそのまま入れる（PNG なので劣化しない）


def test_north_is_up(tmp_path):
    cache = gsi.TileCache(tmp_path, fetch=_fake_tile_fetch, delay=0)
    unproject = lambda x, y: (35.4 + np.asarray(y) / 111000.0, 139.2 + np.asarray(x) / 90700.0)
    img = photo.render_local_image(cache, unproject, 0.0, 0.0, 400.0, px_size=2.0, ext="png")
    assert img.shape == (200, 200, 3)
    ty = lambda p: (int(p[0]) << 16) | (int(p[1]) << 8) | int(p[2])
    assert ty(img[0, 100]) < ty(img[-1, 100])  # 北（上）のほうがタイル番号 y が小さい


def test_hillshade_flat_and_slopes():
    flat = photo.hillshade(np.zeros((10, 10)), 1.0)
    assert np.allclose(flat, flat[0, 0])
    yy, xx = np.mgrid[0:20, 0:20].astype(float)  # 行0が北、列0が西
    nw_facing = photo.hillshade((xx + yy) * 0.5, 1.0)    # 北と西に向かって下がる斜面
    se_facing = photo.hillshade(-(xx + yy) * 0.5, 1.0)
    assert nw_facing[10, 10] > se_facing[10, 10]


def test_review_chunks_cover_route():
    class C: pass
    c = C(); c.s = np.arange(0.0, 1001.0); c.x = c.s.copy(); c.y = np.zeros_like(c.s)
    ch = photo.review_chunks(c, every_m=150, size_m=250)
    assert ch[0][0] == 0 and ch[-1][0] >= 900 and all(x0 <= c.x[int(sc)] <= x0 + 250 for sc, x0, y0 in ch)


def test_overlay_size():
    class C: pass
    c = C(); c.s = np.arange(0.0, 101.0); c.x = c.s.copy(); c.y = np.zeros_like(c.s); c.heading = np.zeros_like(c.s)
    im = photo.draw_overlay(np.zeros((100, 100, 3), np.uint8), c, -10.0, -50.0, 100.0, 1.0, half_width=np.full(101, 3.0))
    assert im.size == (100, 100)
