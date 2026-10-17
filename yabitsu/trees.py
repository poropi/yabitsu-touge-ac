import numpy as np
import shapely
from scipy.interpolate import RegularGridInterpolator
from shapely.geometry import Polygon


def scatter(near, pads=(), density=1 / 40, clear_m=3.0, band=105.0, seed=7, pad_clear_m=None):
    rng = np.random.default_rng(seed)
    xs, ys = near["xs"], near["ys"]
    n = int((xs[-1] - xs[0]) * (ys[-1] - ys[0]) * density)
    P = np.c_[rng.uniform(xs[0], xs[-1], n), rng.uniform(ys[0], ys[-1], n)]
    q = P[:, ::-1]
    E = RegularGridInterpolator((ys, xs), near["E"])(q); D = RegularGridInterpolator((ys, xs), near["D"])(q)
    keep = (E >= clear_m) & (D <= band)
    if pads:
        pts = shapely.points(P)
        for p in pads:
            keep &= shapely.distance(pts, Polygon(p["polygon"])) > (clear_m if pad_clear_m is None else pad_clear_m)
    P = P[keep]
    z = RegularGridInterpolator((ys, xs), near["Z"])(P[:, ::-1])
    return np.c_[P, z, rng.uniform(8.0, 16.0, len(P))]


def cards(trees, seed=8):
    rng = np.random.default_rng(seed)
    verts, uvs, faces = [], [], []
    for x, y, z, h in trees:
        a = rng.uniform(0, np.pi); w = h * 0.45 / 2
        for ang in (a, a + np.pi / 2):
            dx, dy = w * np.cos(ang), w * np.sin(ang)
            base = len(verts)
            verts += [(x - dx, y - dy, z), (x + dx, y + dy, z), (x + dx, y + dy, z + h), (x - dx, y - dy, z + h)]
            uvs += [(0, 0), (1, 0), (1, 1), (0, 1)]
            faces.append((base, base + 1, base + 2, base + 3))
    return np.array(verts, float).reshape(-1, 3), np.array(faces, np.int32).reshape(-1, 4), np.array(uvs, float).reshape(-1, 2)


def chunk_ids(trees, size=200.0):
    return (np.floor(trees[:, 0] / size).astype(np.int64) * 100000 + np.floor(trees[:, 1] / size).astype(np.int64))
