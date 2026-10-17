import numpy as np
import shapely
from shapely.geometry import Polygon

from .crosssection import GUTTER_DEPTH, GUTTER_WIDTH, SKIRT_DEPTH, surface_height


def ribbon(x, y, heading, lat, z, s, u_scale=4.0, v_scale=4.0, face_mask=None, flip=False):
    lat = np.asarray(lat, float); z = np.asarray(z, float)
    N, M = lat.shape
    nx, ny = -np.sin(heading)[:, None], np.cos(heading)[:, None]
    X = np.asarray(x)[:, None] + lat * nx; Y = np.asarray(y)[:, None] + lat * ny
    verts = np.stack([X, Y, z], -1).reshape(-1, 3)
    d = np.sqrt(np.diff(X, axis=1) ** 2 + np.diff(Y, axis=1) ** 2 + np.diff(z, axis=1) ** 2)
    u = np.c_[np.zeros(N), np.cumsum(d, axis=1)] / u_scale
    v = np.broadcast_to(np.asarray(s)[:, None] / v_scale, (N, M))
    uvs = np.stack([u, v], -1).reshape(-1, 2)
    i = np.arange(N * M).reshape(N, M)
    faces = np.stack([i[:-1, :-1], i[1:, :-1], i[1:, 1:], i[:-1, 1:]], -1).reshape(-1, 4)
    if face_mask is not None:
        faces = faces[np.asarray(face_mask, bool).reshape(-1)]
    if flip:
        faces = faces[:, ::-1]
    return verts, faces, uvs


def face_normals(verts, faces):
    p = verts[faces]
    q = np.roll(p, -1, axis=1)
    n = np.stack([((p[..., 1] - q[..., 1]) * (p[..., 2] + q[..., 2])).sum(1),
                  ((p[..., 2] - q[..., 2]) * (p[..., 0] + q[..., 0])).sum(1),
                  ((p[..., 0] - q[..., 0]) * (p[..., 1] + q[..., 1])).sum(1)], -1)
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)


def folded_faces(verts, faces):
    # 四角形は2つの三角形に分けて調べる（折り返した四角形は、全体の向きだけでは上向きに見えることがある）
    p = verts[faces]

    def up(a, b, c):
        return (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])

    bad = up(p[:, 0], p[:, 1], p[:, 2]) <= 0
    if faces.shape[1] == 4:
        bad |= up(p[:, 0], p[:, 2], p[:, 3]) <= 0
    return np.nonzero(bad)[0]


def compact(verts, faces, uvs):
    used, inv = np.unique(faces.ravel(), return_inverse=True)
    return verts[used], inv.reshape(faces.shape), uvs[used]


def split_rows(n, rows_per_chunk):
    out, i = [], 0
    while i < n - 1:
        j = min(i + rows_per_chunk, n - 1); out.append((i, j)); i = j
    return out


def _cols(sec, sl, a):
    return np.asarray(a)[sl][:, None]


def road_surface(sec, sl, n_lat=5):
    fr = np.linspace(-1, 1, n_lat)[None, :]
    lat = np.where(fr < 0, fr * _cols(sec, sl, sec.hw_right), fr * _cols(sec, sl, sec.hw_left))
    z = surface_height(lat, _cols(sec, sl, sec.z), _cols(sec, sl, sec.slope), _cols(sec, sl, sec.crown))
    return ribbon(sec.x[sl], sec.y[sl], sec.heading[sl], lat, z, sec.s[sl])


def gutter(sec, sl, side):
    hw = _cols(sec, sl, sec.hw_left if side > 0 else sec.hw_right)
    z_edge = surface_height(side * hw, _cols(sec, sl, sec.z), _cols(sec, sl, sec.slope), _cols(sec, sl, sec.crown))
    offs = np.array([0.0, GUTTER_WIDTH / 2, GUTTER_WIDTH, GUTTER_WIDTH])[None, :]
    dz = np.array([0.0, -GUTTER_DEPTH, 0.0, -SKIRT_DEPTH])[None, :]
    lat = side * (hw + offs); z = z_edge + dz
    if side < 0:
        lat, z = lat[:, ::-1], z[:, ::-1]
    return ribbon(sec.x[sl], sec.y[sl], sec.heading[sl], lat, z, sec.s[sl], u_scale=1.0)


def line(sec, sl, center_lat, width=0.15, lift=0.005, dash=None, row_mask=None):
    lat = np.asarray(center_lat, float)[:, None] + np.array([-width / 2, width / 2])[None, :]
    z = surface_height(lat, _cols(sec, sl, sec.z), _cols(sec, sl, sec.slope), _cols(sec, sl, sec.crown)) + lift
    s = sec.s[sl]
    mask = np.ones(len(s) - 1, bool)
    if dash is not None:
        on, off = dash
        mask &= (s[:-1] % (on + off)) < on
    if row_mask is not None:
        mask &= np.asarray(row_mask, bool)
    return ribbon(sec.x[sl], sec.y[sl], sec.heading[sl], lat, z, s, u_scale=1.0, face_mask=mask[:, None])


def wall_ribbon(sec, sl, side, lat_bottom, lat_top, z_bottom, z_top):
    lat = np.stack([side * np.asarray(lat_bottom, float), side * np.asarray(lat_top, float)], 1)
    z = np.stack([np.asarray(z_bottom, float), np.asarray(z_top, float)], 1)
    return ribbon(sec.x[sl], sec.y[sl], sec.heading[sl], lat, z, sec.s[sl], u_scale=2.0, flip=side < 0)


def pad_mesh(polygon, z):
    tris = shapely.constrained_delaunay_triangles(Polygon(polygon))
    verts, faces = [], []
    for t in tris.geoms:
        c = np.asarray(t.exterior.coords)[:3]
        if (c[1, 0] - c[0, 0]) * (c[2, 1] - c[0, 1]) - (c[1, 1] - c[0, 1]) * (c[2, 0] - c[0, 0]) < 0:
            c = c[::-1]
        b = len(verts); verts += [(p[0], p[1], z) for p in c]; faces.append((b, b + 1, b + 2))
    v = np.array(verts, float)
    return v, np.array(faces, np.int32), v[:, :2] / 4.0
