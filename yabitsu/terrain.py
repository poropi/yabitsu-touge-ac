import numpy as np
import shapely
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree
from shapely.geometry import Polygon

from . import geo
from .crosssection import GUTTER_WIDTH, surface_height
from .meshgen import compact

FILL_SLOPE, CUT_SLOPE, UNDER_DEPTH = 1.0, 2.0, 0.3
NEAR_BAND, NEAR_CELL = 110.0, 2.0
FAR_EXTENT, FAR_CELL, FAR_HOLE, FAR_DROP = 2000.0, 20.0, 95.0, 1.0
EXTRA_NEAR, EXTRA_HOLE = 20.0, 15.0  # 道から離れた場所（展望台の広場など）のまわりに、細かい地形を張る幅と粗い地形の穴の幅


def corridor_grid(sec, band=NEAR_BAND, cell=NEAR_CELL, extra=()):
    x0, x1, y0, y1 = sec.x.min() - band, sec.x.max() + band, sec.y.min() - band, sec.y.max() + band
    for poly in extra:
        bx0, by0, bx1, by1 = Polygon(poly).bounds
        x0, y0 = min(x0, bx0 - EXTRA_NEAR), min(y0, by0 - EXTRA_NEAR)
        x1, y1 = max(x1, bx1 + EXTRA_NEAR), max(y1, by1 + EXTRA_NEAR)
    xs = np.arange(np.floor(x0 / cell) * cell, x1 + cell, cell)
    ys = np.arange(np.floor(y0 / cell) * cell, y1 + cell, cell)
    return xs, ys


def _with_extra(dist, X, Y, extra, reach, limit):
    """道からの距離に、追加の場所（多角形）からの距離を混ぜる。多角形から reach 以内なら limit 以下になる。"""
    if not len(extra):
        return dist
    pts = shapely.points(np.c_[np.ravel(X), np.ravel(Y)])
    d = np.min([shapely.distance(pts, Polygon(p)) for p in extra], axis=0).reshape(np.shape(dist))
    return np.minimum(dist, d + (limit - reach))


def road_frame(sec, X, Y):
    P = np.c_[np.ravel(X), np.ravel(Y)]
    dist, idx = cKDTree(np.c_[sec.x, sec.y]).query(P)
    h = sec.heading[idx]
    dx, dy = P[:, 0] - sec.x[idx], P[:, 1] - sec.y[idx]
    return {"idx": idx, "lat": -dx * np.sin(h) + dy * np.cos(h), "along": dx * np.cos(h) + dy * np.sin(h), "dist": dist}


def carve(X, Y, Z_nat, sec, pads=(), cut_left=None, cut_right=None, side_extra_left=None, side_extra_right=None):
    """side_extra_*（行ごとの歩道の幅）があれば、そのぶん道の範囲を外へ広げる（歩道の下に地形がかぶらない）。"""
    shape = np.shape(X)
    f = road_frame(sec, X, Y)
    idx, lat, along = f["idx"], f["lat"], f["along"]
    left = lat >= 0
    hw = np.where(left, sec.hw_left[idx], sec.hw_right[idx])
    xl = np.zeros(sec.n) if side_extra_left is None else np.asarray(side_extra_left, float)
    xr = np.zeros(sec.n) if side_extra_right is None else np.asarray(side_extra_right, float)
    form = hw + GUTTER_WIDTH + np.where(left, xl[idx], xr[idx])
    zc, sl, cr = sec.z[idx], sec.slope[idx], sec.crown[idx]
    z_edge = surface_height(np.where(left, hw, -hw), zc, sl, cr)
    e = np.hypot(np.maximum(np.abs(lat) - form, 0), np.maximum(np.abs(along) - 0.75, 0))
    cl_ = np.full(sec.n, CUT_SLOPE) if cut_left is None else np.asarray(cut_left)
    cr_ = np.full(sec.n, CUT_SLOPE) if cut_right is None else np.asarray(cut_right)
    cut = np.where(left, cl_[idx], cr_[idx])
    z = np.clip(np.ravel(Z_nat), z_edge - FILL_SLOPE * e, z_edge + cut * e)
    # 道の下は、外側の切り土の斜面を内側へ延長した高さにする。境目をまたぐ三角形が同じ斜面上に乗り、
    # 側溝や路面の上に地形がはみ出さない
    e_in = np.maximum(form - np.abs(lat), 0)
    z_under = np.minimum(surface_height(np.clip(lat, -hw, hw), zc, sl, cr) - UNDER_DEPTH, z_edge - cut * e_in)
    z = np.where(e == 0, z_under, z)
    if pads:
        pts = shapely.points(np.c_[np.ravel(X), np.ravel(Y)])
        for p in pads:
            d = shapely.distance(pts, Polygon(p["polygon"]))
            zp = p["z"]
            if "plane" in p:  # 傾いた平面 [cx, cy, z0, gx, gy]：z = z0 + gx(x−cx) + gy(y−cy)
                cx, cy, z0, gx, gy = p["plane"]
                zp = z0 + gx * (np.ravel(X) - cx) + gy * (np.ravel(Y) - cy)
            z = np.where(d == 0, zp - UNDER_DEPTH, np.clip(z, zp - FILL_SLOPE * d, zp + CUT_SLOPE * d))
            e = np.where(d == 0, 0.0, np.minimum(e, d))
    return z.reshape(shape), e.reshape(shape)


def grid_mesh_chunks(xs, ys, Z, face_keep, block=100, uv_scale=8.0):
    X, Y = np.meshgrid(xs, ys)
    verts = np.c_[X.ravel(), Y.ravel(), Z.ravel()]
    uvs = verts[:, :2] / uv_scale
    i = np.arange(X.size).reshape(X.shape)
    faces = np.stack([i[:-1, :-1], i[:-1, 1:], i[1:, 1:], i[1:, :-1]], -1)
    ry, rx = np.meshgrid(np.arange(Z.shape[0] - 1) // block, np.arange(Z.shape[1] - 1) // block, indexing="ij")
    keep = np.asarray(face_keep, bool)
    out = []
    for by, bx in sorted(set(zip(ry[keep].tolist(), rx[keep].tolist()))):
        m = keep & (ry == by) & (rx == bx)
        out.append(compact(verts, faces[m].reshape(-1, 4), uvs))
    return out


def fill_nan_nearest(Z):
    m = np.isnan(Z)
    if not m.any():
        return Z
    idx = distance_transform_edt(m, return_distances=False, return_indices=True)
    return Z[tuple(idx)]


def _sample(sampler, unproject, X, Y):
    lat, lon = unproject(X.ravel(), Y.ravel())
    return fill_nan_nearest(sampler.sample(lat, lon).reshape(X.shape))


def build_near(sec, sampler, pads, cut_left=None, cut_right=None, unproject=geo.from_local, extra=(), side_extra=(None, None)):
    xs, ys = corridor_grid(sec, extra=extra)
    X, Y = np.meshgrid(xs, ys)
    D = _with_extra(road_frame(sec, X, Y)["dist"].reshape(X.shape), X, Y, extra, EXTRA_NEAR, NEAR_BAND)
    Z, E = carve(X, Y, _sample(sampler, unproject, X, Y), sec, pads, cut_left, cut_right, *side_extra)
    return {"xs": xs, "ys": ys, "Z": Z, "E": E, "D": D}


def build_far(sec, sampler, unproject=geo.from_local, extra=()):
    xs, ys = corridor_grid(sec, FAR_EXTENT + NEAR_BAND, FAR_CELL)
    X, Y = np.meshgrid(xs, ys)
    Z = _sample(sampler, unproject, X, Y) - FAR_DROP
    cx, cy = np.meshgrid((xs[:-1] + xs[1:]) / 2, (ys[:-1] + ys[1:]) / 2)
    Dc = _with_extra(road_frame(sec, cx, cy)["dist"].reshape(cx.shape), cx, cy, extra, EXTRA_HOLE, FAR_HOLE)
    return {"xs": xs, "ys": ys, "Z": Z, "Dc": Dc}
