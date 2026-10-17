"""道の外の場所（駐車場・広場・集落）を作るときの共通の道具。位置は緯度経度で受け取る。"""
import numpy as np
import shapely

from . import geo

PLANE_RADIUS = 40.0  # この半径の中の路面の高さに、平面を合わせる


def ll_to_xy(ll):
    a = np.asarray(ll, float).reshape(-1, 2)
    x, y = geo.to_local(a[:, 0], a[:, 1])
    return np.c_[x, y]


def fit_plane(sec, center_xy, radius=PLANE_RADIUS):
    """center_xy のまわりの路面の高さに合わせた平面 [cx, cy, z0, gx, gy]。道がまっすぐで横の傾きが決まらないときは 0。"""
    c = np.asarray(center_xy, float)
    m = np.hypot(sec.x - c[0], sec.y - c[1]) <= radius
    A = np.c_[np.ones(m.sum()), sec.x[m] - c[0], sec.y[m] - c[1]]
    ridge = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    z0, gx, gy = np.linalg.lstsq(np.r_[A, ridge], np.r_[sec.z[m], 0.0, 0.0], rcond=None)[0]
    return [float(c[0]), float(c[1]), float(z0), float(gx), float(gy)]


def plane_z(plane, x, y):
    cx, cy, z0, gx, gy = plane
    return z0 + gx * (np.asarray(x, float) - cx) + gy * (np.asarray(y, float) - cy)


def draped(poly, z_fn, dz=0.0, uv_scale=4.0):
    """多角形を三角形に分け、高さ z_fn(x, y) + dz に置く（面は上向き）。"""
    verts, faces = [], []
    for t in shapely.constrained_delaunay_triangles(poly).geoms:
        p = np.asarray(t.exterior.coords)[:3]
        if (p[1, 0] - p[0, 0]) * (p[2, 1] - p[0, 1]) - (p[1, 1] - p[0, 1]) * (p[2, 0] - p[0, 0]) < 0:
            p = p[::-1]
        b = len(verts); verts += p.tolist(); faces.append((b, b + 1, b + 2))
    v = np.array(verts, float)
    return np.c_[v, z_fn(v[:, 0], v[:, 1]) + dz], np.array(faces, np.int32), v / uv_scale


def largest(g):
    return g if g.geom_type == "Polygon" else max(g.geoms, key=lambda q: q.area)


def footprint_box(poly):
    """建物の外周線を、向きのついた箱（中心、長辺の向き[度]、[長さ, 幅]）にする。L字などで外接する長方形が
    大きすぎるときは、向きと縦横比を保ったまま面積を外周線の面積に合わせる。"""
    r = np.asarray(poly.minimum_rotated_rectangle.exterior.coords)[:4]
    e1, e2 = r[1] - r[0], r[2] - r[1]
    a, b = float(np.linalg.norm(e1)), float(np.linalg.norm(e2))
    axis = e1 if a >= b else e2
    L, W = max(a, b), min(a, b)
    k = min(1.0, np.sqrt(poly.area / max(L * W, 1e-9)))
    c = poly.centroid
    return (float(c.x), float(c.y)), float(np.degrees(np.arctan2(axis[1], axis[0]))), [L * k, W * k]
