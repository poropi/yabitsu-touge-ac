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
FAR_EXTENT, FAR_CELL, FAR_HOLE, FAR_DROP = 2000.0, 20.0, 205.0, 1.0
MID_BAND, MID_CELL, MID_DROP, MID_OVERLAP = 210.0, 4.0, 0.15, 6.0  # 中くらいの地形：道から 200m まで（細かい地形の下へ 6m もぐり込ませる）
EXTRA_NEAR, EXTRA_HOLE = 20.0, 15.0
TOWN_BLEND_M = 10.0  # 町の範囲の円の外で、町の地面から今までの地形へ移す幅
TUNNEL_COVER_M, TUNNEL_SIDE_M = 3.0, 4.0   # トンネルの天井の上の土かぶり、その幅（筒の外へ）
CHANNEL_REACH_M = 6.0   # 川の溝が地形を動かす範囲（溝の外）
ROAD_CAP_M = 3.0     # 土台があっても、道の縁からこの距離までは道の斜面より上に地形を出さない
PAD_REACH_M = 40.0   # 土台（駐車場・家・分かれ道）が地形を動かす範囲  # 道から離れた場所（展望台の広場など）のまわりに、細かい地形を張る幅と粗い地形の穴の幅


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


def carve(X, Y, Z_nat, sec, pads=(), cut_left=None, cut_right=None, side_extra_left=None, side_extra_right=None, channels=(),
          keep_rows=None, no_fill_rows=None):
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
    road_cap = np.where(e == 0, z_under, z_edge + cut * e)   # 道の路面・切り土の斜面より上には地形を出さない（道から ROAD_CAP_M まで）
    e_road = e.copy()
    if pads:
        xr, yr = np.ravel(X), np.ravel(Y)
        for p in pads:
            poly = Polygon(p["polygon"])
            x0, y0, x1, y1 = poly.bounds
            # 土台が地形を動かすのは、まわり PAD_REACH_M まで（家や分かれ道の土台が千個近くあっても速く）
            sel = np.nonzero((xr >= x0 - PAD_REACH_M) & (xr <= x1 + PAD_REACH_M) & (yr >= y0 - PAD_REACH_M) & (yr <= y1 + PAD_REACH_M))[0]
            if not len(sel):
                continue
            d = shapely.distance(shapely.points(np.c_[xr[sel], yr[sel]]), poly)
            near = d <= PAD_REACH_M
            sel, d = sel[near], d[near]
            zp = p["z"]
            if "plane" in p:  # 傾いた平面 [cx, cy, z0, gx, gy]：z = z0 + gx(x−cx) + gy(y−cy)
                cx, cy, z0, gx, gy = p["plane"]
                zp = z0 + gx * (xr[sel] - cx) + gy * (yr[sel] - cy)
            z[sel] = np.where(d == 0, zp - UNDER_DEPTH, np.clip(z[sel], zp - FILL_SLOPE * d, zp + CUT_SLOPE * d))
            e[sel] = np.where(d == 0, 0.0, np.minimum(e[sel], d))
        z = np.where(e_road <= ROAD_CAP_M, np.minimum(z, road_cap), z)   # 道のすぐ横の高い土台（家など）が、道の上に地形を持ち上げないように
    if channels:   # 川の溝（water.Stream.channel）：溝の中は川底、まわりは岸の勾配 BANK で元の地形へ
        from .water import BANK
        xr, yr = np.ravel(X), np.ravel(Y)
        for poly, z_bed in channels:
            x0, y0, x1, y1 = poly.bounds; r = CHANNEL_REACH_M
            sel = np.nonzero((xr >= x0 - r) & (xr <= x1 + r) & (yr >= y0 - r) & (yr <= y1 + r))[0]
            if not len(sel):
                continue
            d = shapely.distance(shapely.points(np.c_[xr[sel], yr[sel]]), poly)
            near = d <= r
            sel, d = sel[near], d[near]
            z[sel] = np.minimum(z[sel], z_bed(xr[sel], yr[sel]) + BANK * d)
    if no_fill_rows is not None and np.any(no_fill_rows):   # 大きな橋（S5）の下：谷を盛らない（元の地形より上げない）
        nf = np.asarray(no_fill_rows, bool)[idx]
        z[nf] = np.minimum(z[nf], np.ravel(Z_nat)[nf])
    if keep_rows is not None and np.any(keep_rows):   # トンネル：一番近い行がトンネルの所は、山を削らず、筒の天井＋土かぶりより下げない
        from .tunnel import bore_height
        k = np.asarray(keep_rows, bool)[idx]
        roof = np.array([bore_height(sec, i) for i in range(sec.n)]) if k.any() else None
        if k.any():
            band = np.abs(lat) <= hw + GUTTER_WIDTH + TUNNEL_SIDE_M
            zk = np.ravel(Z_nat)[k]
            zk = np.where(band[k], np.maximum(zk, sec.z[idx][k] + roof[idx][k] + TUNNEL_COVER_M), zk)
            z[k] = zk; e[k] = np.maximum(e[k], 1.0)
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


def build_near(sec, sampler, pads, cut_left=None, cut_right=None, unproject=geo.from_local, extra=(), side_extra=(None, None),
               town=None, channels=(), keep_rows=None, no_fill_rows=None):
    """town = (町の地面 TownGround, 町の範囲の円, 舗装された所の多角形) なら、円の中は町の地面（舗装の下は UNDER_DEPTH 下げる）、
    円の外 TOWN_BLEND_M の間で今までの地形へなめらかに移す。"""
    towns = [] if town is None else (list(town) if isinstance(town, list) else [town])
    for tw in towns:
        extra = list(extra) + [np.asarray(tw[1].exterior.coords).tolist()]
    xs, ys = corridor_grid(sec, extra=extra)
    X, Y = np.meshgrid(xs, ys)
    D = _with_extra(road_frame(sec, X, Y)["dist"].reshape(X.shape), X, Y, extra, EXTRA_NEAR, NEAR_BAND)
    Z, E = carve(X, Y, _sample(sampler, unproject, X, Y), sec, pads, cut_left, cut_right, *side_extra, channels=channels,
                 keep_rows=keep_rows, no_fill_rows=no_fill_rows)
    if towns:   # 本線の帯（路面＋側溝）の下。町の地面へ移す帯の中でも、ここは削った高さより上にしない
        f = road_frame(sec, X, Y); idx = f["idx"]
        hw = np.where(f["lat"] >= 0, sec.hw_left[idx], sec.hw_right[idx])
        under = ((np.abs(f["lat"]) <= hw + GUTTER_WIDTH) & (np.abs(f["along"]) <= 0.75)).reshape(X.shape)
        Zc = Z.copy()
    for ground, circle, paved in towns:
        x0, y0, x1, y1 = circle.buffer(TOWN_BLEND_M).bounds
        m = (X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1)
        pts = shapely.points(np.c_[X[m], Y[m]])
        d = shapely.distance(pts, circle)
        zt = ground.z(X[m], Y[m]) - np.where(shapely.contains(paved, pts), UNDER_DEPTH, 0.0)
        t = np.clip(d / TOWN_BLEND_M, 0.0, 1.0); w = t * t * (3 - 2 * t)
        Z[m] = (1 - w) * zt + w * Z[m]
        Z[m] = np.where((d > 0) & under[m], np.minimum(Z[m], Zc[m]), Z[m])
        E[m] = np.where(d == 0, 0.0, E[m])
    return {"xs": xs, "ys": ys, "Z": Z, "E": E, "D": D}


def build_far(sec, sampler, unproject=geo.from_local, extra=()):
    xs, ys = corridor_grid(sec, FAR_EXTENT + NEAR_BAND, FAR_CELL)
    X, Y = np.meshgrid(xs, ys)
    Z = _sample(sampler, unproject, X, Y) - FAR_DROP
    cx, cy = np.meshgrid((xs[:-1] + xs[1:]) / 2, (ys[:-1] + ys[1:]) / 2)
    Dc = _with_extra(road_frame(sec, cx, cy)["dist"].reshape(cx.shape), cx, cy, extra, EXTRA_HOLE, FAR_HOLE)
    return {"xs": xs, "ys": ys, "Z": Z, "Dc": Dc}


def build_mid(sec, sampler, pads=(), unproject=geo.from_local, extra=(), channels=(), keep_rows=None, no_fill_rows=None):
    """細かい地形（NEAR_BAND まで）の外、MID_BAND までの 4m 格子の地形。家などの土台（pads）にも合わせ、MID_DROP 下げる。"""
    xs, ys = corridor_grid(sec, MID_BAND, MID_CELL, extra=extra)
    X, Y = np.meshgrid(xs, ys)
    D = road_frame(sec, X, Y)["dist"].reshape(X.shape)
    Z, _ = carve(X, Y, _sample(sampler, unproject, X, Y), sec, pads, channels=channels, keep_rows=keep_rows, no_fill_rows=no_fill_rows)
    return {"xs": xs, "ys": ys, "Z": Z - MID_DROP, "D": D}


def mid_keep(D):
    """中くらいの地形の、描く四角（4つの角がどれも NEAR_BAND−MID_OVERLAP 以上、MID_BAND 以下）。"""
    lo, hi = NEAR_BAND - MID_OVERLAP, MID_BAND
    c = [D[:-1, :-1], D[:-1, 1:], D[1:, 1:], D[1:, :-1]]
    return np.all([(q >= lo) & (q <= hi) for q in c], axis=0)


def terrain_z(near, mid):
    """地形の高さを返す関数（細かい地形の中はそれ、外は中くらいの地形）。景色の道や家を地形の上に置くため。"""
    from scipy.interpolate import RegularGridInterpolator
    fn = RegularGridInterpolator((near["ys"], near["xs"]), near["Z"], bounds_error=False, fill_value=np.nan)
    fd = RegularGridInterpolator((near["ys"], near["xs"]), near["D"], bounds_error=False, fill_value=np.inf)
    fm = RegularGridInterpolator((mid["ys"], mid["xs"]), mid["Z"] + MID_DROP, bounds_error=False, fill_value=np.nan)

    def z(x, y):
        q = np.c_[np.ravel(y), np.ravel(x)]
        zn, dn, zm = fn(q), fd(q), fm(q)
        return np.where((dn <= NEAR_BAND - 2.0) & ~np.isnan(zn), zn, zm).reshape(np.shape(x))
    return z


BRANCH_MARGIN = -0.02   # 分かれ道は、路面より 2cm 以上上に出たときだけ（土台の継ぎ目の丸め）


def road_clear_count(sec, near, stubs=(), margin=0.1, skip_rows=None):
    """本線の路面（1m ごと、横 9 点）と分かれ道の路面（軸の上 1m ごと）で、地形が路面の margin 以内まで来る点の数。"""
    from scipy.interpolate import RegularGridInterpolator
    fn = RegularGridInterpolator((near["ys"], near["xs"]), near["Z"], bounds_error=False, fill_value=np.nan)
    k = np.linspace(-1.0, 1.0, 9)[None, :]
    lat = np.where(k >= 0, k * (sec.hw_left[:, None] - 0.2), k * (sec.hw_right[:, None] - 0.2))
    nx, ny = -np.sin(sec.heading)[:, None], np.cos(sec.heading)[:, None]
    X = sec.x[:, None] + lat * nx; Y = sec.y[:, None] + lat * ny
    zt = fn(np.c_[Y.ravel(), X.ravel()]).reshape(X.shape)
    zr = surface_height(lat, sec.z[:, None], sec.slope[:, None], sec.crown[:, None])
    mg = np.full(k.shape, margin); mg[:, [0, -1]] = 0.0   # 端の点（側溝のきわ）は、路面より上に出たときだけ
    hit = zt > zr - mg
    if skip_rows is not None:   # トンネルの行（道の上は山）
        hit[np.asarray(skip_rows, bool)] = False
    bad = int(np.sum(hit))
    for st in stubs:
        for t in np.arange(0.0, st.axis.length, 1.0):
            p = st.axis.interpolate(t)
            z = fn([[p.y, p.x]])[0]
            if np.isfinite(z) and z > st.z_at(p.x, p.y) - BRANCH_MARGIN:   # 分かれ道は土台の継ぎ目で数cm 寄るので、路面より上に出る所だけ
                bad += 1
    return bad
