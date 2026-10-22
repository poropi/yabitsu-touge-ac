import io

import numpy as np
from PIL import Image, ImageDraw

from .gsi import tile_coords


def render_local_image(cache, unproject, x0, y0, size_m, px_size=0.5, layer="seamlessphoto", z=18, ext="jpg", height_m=None):
    """左下 (x0, y0)、横 size_m・縦 height_m（省くと正方形）の空中写真。1 行目が北。"""
    height_m = size_m if height_m is None else height_m
    n = int(round(size_m / px_size)); nh = int(round(height_m / px_size))
    xs = x0 + (np.arange(n) + 0.5) * px_size
    ys = y0 + height_m - (np.arange(nh) + 0.5) * px_size
    X, Y = np.meshgrid(xs, ys)
    lat, lon = unproject(X.ravel(), Y.ravel())
    fx, fy = tile_coords(lat, lon, z)
    tx, ty = np.floor(fx).astype(np.int64), np.floor(fy).astype(np.int64)
    px = np.clip(((fx - tx) * 256).astype(np.int64), 0, 255); py = np.clip(((fy - ty) * 256).astype(np.int64), 0, 255)
    out = np.full((n * nh, 3), 128, np.uint8)
    for a, b in np.unique(np.stack([tx, ty], 1), axis=0):
        data = cache.get_bytes(layer, z, int(a), int(b), ext)
        if data is None:
            continue
        t = np.array(Image.open(io.BytesIO(data)).convert("RGB"))
        m = (tx == a) & (ty == b)
        out[m] = t[py[m], px[m]]
    return out.reshape(nh, n, 3)


def hillshade(zgrid, cell, azimuth=315.0, altitude=45.0):
    dzdy_rows, dzdx = np.gradient(zgrid, cell)
    dzdy = -dzdy_rows  # 行0が北なので、北向きの傾きは符号が逆
    slope = np.arctan(np.hypot(dzdx, dzdy))
    aspect = np.arctan2(-dzdx, -dzdy)  # 斜面が下がっていく方位（北=0、時計回り）
    az, alt = np.radians(azimuth), np.radians(altitude)
    hs = np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect)
    return np.clip(hs, 0, 1)


def review_chunks(cl, every_m=150.0, size_m=250.0):
    out = []
    for sc in np.arange(0.0, cl.s[-1] + 1e-9, every_m):
        i = int(np.argmin(np.abs(cl.s - sc)))
        out.append((float(sc), float(cl.x[i] - size_m / 2), float(cl.y[i] - size_m / 2)))
    return out


def draw_overlay(img, cl, x0, y0, size_m, px_size, half_width=None):
    im = Image.fromarray(img).convert("RGB")
    d = ImageDraw.Draw(im)
    to_px = lambda x, y: list(zip((np.asarray(x) - x0) / px_size, (y0 + size_m - np.asarray(y)) / px_size))
    d.line(to_px(cl.x, cl.y), fill=(255, 0, 0), width=1)
    if half_width is not None:
        nx, ny = -np.sin(cl.heading), np.cos(cl.heading)
        for sgn in (1, -1):
            d.line(to_px(cl.x + sgn * half_width * nx, cl.y + sgn * half_width * ny), fill=(255, 255, 0), width=1)
    for sv in range(0, int(cl.s[-1]) + 1, 50):
        i = int(np.argmin(np.abs(cl.s - sv)))
        (px, py), = to_px([cl.x[i]], [cl.y[i]])
        if 0 <= px < im.width and 0 <= py < im.height:
            d.ellipse([px - 2, py - 2, px + 2, py + 2], outline=(0, 255, 255)); d.text((px + 4, py - 6), str(sv), fill=(0, 255, 255))
    return im
