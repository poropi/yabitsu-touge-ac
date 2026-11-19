"""川・沢（OSM の waterway=river|stream）を、谷の底の水面と川原（石の帯）として作る。砂防ダム・堰（OSM の waterway=weir|dam）と
湧き水の水くみ場も置く。水は動かさない。

- 水面の高さ：川の線に沿って 2m ごとに標高を取り、下流へ向かって「それまでの最小」でならし（上流へ上がらない）、20m の平均で
  5m 標高のでこぼこを消す。橋（本線の bridge の行）の下では「橋の路面 − BRIDGE_CLEAR_M」を上限にする
- 地形：水面のまわりに溝を彫る（terrain.carve の channels）。本線の路面（橋を除く）の下は彫らない
- 描かない所：road_area（本線の路面＋側溝（橋を除く）、分かれ道の路面）の 0.5m 内。橋の下には水面が見える"""
from dataclasses import dataclass

import numpy as np
import shapely
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import substring, unary_union

from . import geo
from .valleys import Valley, site_area
from .town import grid_drape, _polys

WIDTH = {"river": 6.0, "stream": 2.5}
BED_W = 1.5            # 水面の両側の川原
STEP_M, SMOOTH_M = 2.0, 20.0
BRIDGE_CLEAR_M = 2.0   # 橋の路面から水面までの最小
BANK = 1.0             # 溝の岸の勾配（水平 1m で 1m 上がる）
WEIR_H = 1.5           # 堰・砂防ダムの高さ（決まった値。OSM に高さは無い）
LIFT = 0.02


def _xy(w):
    return np.array([geo.to_local(g["lat"], g["lon"]) for g in w["geometry"]], float).reshape(-1, 2)


class Stream:
    def __init__(self, line, width_m, kind, z, osm_id=0):
        self.line, self.width_m, self.kind, self.z, self.osm_id = line, float(width_m), kind, np.asarray(z, float), osm_id
        self.t = np.linspace(0.0, line.length, len(self.z))

    def z_at(self, t):
        return np.interp(t, self.t, self.z)

    def z_fn(self, dz=0.0):
        def f(x, y):
            x = np.atleast_1d(x); y = np.atleast_1d(y)
            t = shapely.line_locate_point(self.line, shapely.points(np.c_[np.ravel(x), np.ravel(y)]))
            return (self.z_at(t) + dz).reshape(np.shape(x))
        return f

    def water(self):
        return self.line.buffer(self.width_m / 2, cap_style="flat")

    def channel(self):
        """(溝の範囲, 川底の高さの関数)：terrain.carve の channels に渡す。"""
        return self.line.buffer(self.width_m / 2 + BED_W, cap_style="flat"), self.z_fn(-0.3)


def road_area(sec, formation, stubs=()):
    """水を描かない所：本線の路面＋側溝のうち橋でない所と、分かれ道の路面。"""
    return unary_union([formation.difference(_bridges(sec))] + [st.surface for st in stubs])


def _bridges(sec):
    """本線の橋の行の、路面＋側溝（+1m）の範囲。"""
    from .start_lot import road_formation
    out, i = [], 0
    b = np.asarray(sec.bridge, bool)
    while i < sec.n:
        if b[i]:
            j = i
            while j + 1 < sec.n and b[j + 1]:
                j += 1
            out.append(road_formation(sec, float(sec.s[max(i - 1, 0)]), float(sec.s[min(j + 1, sec.n - 1)])).buffer(1.0))
            i = j + 1
        else:
            i += 1
    return unary_union(out) if out else Polygon()


def from_osm(elements, sec, sampler, band_m=200.0, s_from=0.0, unproject=geo.from_local):
    """本線の s_from より先から band_m 以内の川・沢。線の向き（OSM）は下流向き。"""
    m = sec.s >= s_from
    band = LineString(np.c_[sec.x[m], sec.y[m]]).buffer(band_m)
    bridges = _bridges(sec)
    deck = lambda p: float(sec.z[int(np.argmin(np.hypot(sec.x - p[0], sec.y - p[1])))])
    out = []
    for w in elements:
        tg = w.get("tags", {})
        if w.get("type") != "way" or tg.get("waterway") not in WIDTH or len(w.get("geometry", [])) < 2:
            continue
        if tg.get("tunnel", "no") not in ("no",):
            continue
        ln = LineString(_xy(w))
        if not band.intersects(ln):
            continue
        for piece in _polys_lines(ln.intersection(band)):
            if piece.length < 10.0:
                continue
            t = np.r_[np.arange(0.0, piece.length, STEP_M), piece.length]
            p = np.array([piece.interpolate(v).coords[0] for v in t])
            zg = np.asarray(sampler.sample(*unproject(p[:, 0], p[:, 1])), float)
            zg = np.where(np.isnan(zg), np.nanmin(zg) if np.isfinite(np.nanmin(zg)) else 0.0, zg)
            z = np.minimum.accumulate(zg)
            k = max(1, int(SMOOTH_M / STEP_M))
            z = np.convolve(np.pad(z, k, mode="edge"), np.ones(2 * k + 1) / (2 * k + 1), mode="same")[k:-k]
            if not bridges.is_empty:
                under = shapely.contains_xy(bridges, p[:, 0], p[:, 1])
                for i in np.nonzero(under)[0]:
                    z[i] = min(z[i], deck(p[i]) - BRIDGE_CLEAR_M)
            z = np.minimum.accumulate(z)
            width = float(tg["width"]) if str(tg.get("width", "")).replace(".", "", 1).isdigit() else WIDTH[tg["waterway"]]
            out.append(Stream(LineString(p), width, tg["waterway"], z, int(w.get("id", 0))))
    _chain(out)
    return out


def _chain(streams, join_m=1.0):
    """way のつなぎ目で、下流の水面が上流の終わりより高くならないようにする：上流の way の終わりが別の way の上
    （始まりでも途中でも）にあれば、そこから下流をその高さ以下にする。変わらなくなるまでくり返す。"""
    for _ in range(len(streams) + 1):
        changed = False
        for up in streams:
            end = Point(up.line.coords[-1]); z_end = float(up.z[-1])
            for dn in streams:
                if dn is up or dn.line.distance(end) > join_m:
                    continue
                t = dn.line.project(end)
                m = dn.t >= t - 1e-6
                if np.any(dn.z[m] > z_end + 1e-9):
                    dn.z[m] = np.minimum(dn.z[m], z_end); dn.z = np.minimum.accumulate(dn.z); changed = True
        if not changed:
            break


def _polys_lines(g):
    if g.is_empty:
        return []
    if g.geom_type == "LineString":
        return [g]
    return [q for q in getattr(g, "geoms", []) if q.geom_type == "LineString"]


@dataclass(frozen=True)
class ValleyWater:
    valley: Valley
    station: np.ndarray
    height: np.ndarray
    banks: np.ndarray
    cut: Polygon
    bed_cut: Polygon

    def polygon(self):
        return shapely.make_valid(Polygon(np.r_[self.banks[0], self.banks[1, ::-1]]))


def valley_surface_levels(sec, bounds, rows, streams, ground_z, *, near=None):
    """谷筋の掘削を地形で確認し、その実際の谷底から水深と水の縁を求める。"""
    by_id = {st.osm_id: st for st in streams}
    levels = {}
    for valley in rows:
        if not valley.water or valley.water["osm_id"] not in by_id:
            continue
        st = by_id[valley.water["osm_id"]]
        line = LineString(valley.line)
        # 折れ点と端点を含め、地形の2m格子より細かく測る。
        arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(valley.line, axis=0), axis=1))]
        t = np.unique(np.r_[arc, np.linspace(0, arc[-1], int(np.ceil(arc[-1] / 0.25)) + 1)])
        xy = shapely.get_coordinates(shapely.line_interpolate_point(line, t))
        floor = ground_z(*xy.T)
        old = st.z_fn(LIFT)(*xy.T)
        target = np.interp(t, arc, np.broadcast_to(valley.bed["z_m"], arc.shape))
        manual_inflow = "width_m" in valley.water
        refined = near is not None and "valley_faces" in near
        if manual_inflow or refined:
            # 支流の始点は OSM 本流の旧水位より高い。ここは旧水位を上限にしない。
            area = site_area(sec, bounds, valley.section, valley.center_m)
            cut = line.buffer(valley.reach_m).intersection(area)
            depth = valley.water["depth_m"]
            height = np.minimum.accumulate(floor + depth)
            a = shapely.get_coordinates(shapely.line_interpolate_point(line, np.maximum(t - .5, 0)))
            b = shapely.get_coordinates(shapely.line_interpolate_point(line, np.minimum(t + .5, t[-1])))
            tangent = b - a
            normal = np.c_[-tangent[:, 1], tangent[:, 0]] / np.linalg.norm(tangent, axis=1)[:, None]
            side = valley.water.get("width_m", valley.width_m) / 2
            banks = np.stack((xy - side * normal, xy + side * normal))
            levels[st.osm_id] = ValleyWater(valley, t, height, banks, cut, area)
            continue
        witness = old - 0.30 - target > 0.60
        if not witness.any() or np.mean(floor[witness] <= target[witness] + 0.25) < 0.9:
            continue

        a = shapely.get_coordinates(shapely.line_interpolate_point(line, np.maximum(t - .5, 0)))
        b = shapely.get_coordinates(shapely.line_interpolate_point(line, np.minimum(t + .5, t[-1])))
        d = b - a
        normal = np.c_[-d[:, 1], d[:, 0]] / np.linalg.norm(d, axis=1)[:, None]
        # 断面を測り、幅の上限へ達する前に地面と水面が交わる最初の位置を縁にする。
        offsets = np.linspace(0, valley.width_m / 2, 31)
        bank_samples = xy[None, :, None, :] + np.array([-1, 1])[:, None, None, None] * normal[None, :, None, :] * offsets[None, None, :, None]
        elevations = ground_z(bank_samples[..., 0], bank_samples[..., 1])
        limit = elevations.max(axis=-1).min(axis=0) + .25
        depth = valley.water["depth_m"]
        blend = np.clip(np.minimum(t, t[-1] - t) / 5, 0, 1)
        intended = old + blend * (floor + depth - old)
        height = np.minimum.accumulate(np.minimum(np.minimum(intended, limit), old))
        hit = elevations >= height[None, :, None]
        crossing = np.argmax(hit, axis=-1)
        k = np.maximum(crossing, 1)
        e0 = np.take_along_axis(elevations, (k - 1)[..., None], axis=-1)[..., 0]
        e1 = np.take_along_axis(elevations, k[..., None], axis=-1)[..., 0]
        fraction = np.clip((height - e0) / np.maximum(e1 - e0, 1e-9), 0, 1)
        # 3m外端まで地面を横切らない岸は、実際の地形との差が25cm以内の最も遠い点で止める。
        near_level = elevations >= height[None, :, None] - .25 - 1e-9
        outer = len(offsets) - 1 - np.argmax(near_level[..., ::-1], axis=-1)
        widths = np.where(hit.any(axis=-1), offsets[k - 1] + fraction * (offsets[k] - offsets[k - 1]), offsets[outer])
        banks = xy + np.array([-1, 1])[:, None, None] * normal * widths[..., None]
        area = site_area(sec, bounds, valley.section, valley.center_m)
        safe = area.buffer(-2 * np.sqrt(2))
        cut = line.buffer(valley.reach_m, cap_style="flat").intersection(safe)
        levels[st.osm_id] = ValleyWater(valley, t, height, banks, cut, safe)
    return levels


def surface_z(stream, levels, dz=0.0):
    reach = levels.get(stream.osm_id)
    if reach is None:
        return stream.z_fn(dz)
    covered = reach.cut.buffer(1e-7)

    def z(x, y):
        x, y = np.asarray(x), np.asarray(y)
        original = stream.z_fn(dz)(x.ravel(), y.ravel())
        active = shapely.intersects_xy(covered, x.ravel(), y.ravel())
        _, t, _, _ = reach.valley.project(x.ravel()[active], y.ravel()[active])
        original[active] = np.interp(t, reach.station, reach.height) + (dz - LIFT)
        return original.reshape(x.shape)

    return z


def _outside_water_cut(mesh, cut, original_z, *, join=None):
    """切替の端にかかる三角形だけ分割し、外へ続く既存の面を保つ。"""
    from .meshgen import compact
    v, f, uv = mesh
    triangles = shapely.polygons(v[f, :2])
    hit = shapely.intersects(cut, triangles)
    if join is not None:
        v = v.copy()
        v[:, 2] = join(v[:, 0], v[:, 1], v[:, 2])
    if not hit.any():
        return v, f, uv
    vs, fs, uvs = [v], [f[~hit]], [uv]
    base = len(v)
    for piece in shapely.difference(triangles[hit], cut):
        if piece.area < 1e-8:
            continue
        for tri in shapely.get_parts(shapely.constrained_delaunay_triangles(piece)):
            xy = shapely.get_coordinates(shapely.orient_polygons(tri))[:3]
            z = original_z(*xy.T)
            vs.append(np.c_[xy, join(*xy.T, z) if join is not None else z])
            fs.append([[base, base + 1, base + 2]])
            uvs.append(xy / 4.)
            base += 3
    return compact(np.concatenate(vs), np.concatenate(fs), np.concatenate(uvs))


def _drape_bed(poly, ground_z, near):
    """地面と同じ対角線で切り、三角形の途中でも浮き沈みしない川原を作る。"""
    if poly.is_empty:
        return
    if near is None:
        yield from grid_drape(poly, ground_z, dz=.08, cell=.25)
        return
    from . import terrain

    x, y, _, _ = terrain.near_edit_window(near, poly)
    xy = np.stack((x, y), axis=-1)
    pieces = []
    for a, b, c in ((xy[:-1, :-1], xy[:-1, 1:], xy[1:, 1:]),
                    (xy[:-1, :-1], xy[1:, 1:], xy[1:, :-1])):
        triangles = shapely.polygons(np.stack((a, b, c), axis=2).reshape(-1, 3, 2))
        for piece in shapely.intersection(triangles[shapely.intersects(poly, triangles)], poly):
            if piece.area <= 1e-9:
                continue
            parts = shapely.get_parts(shapely.constrained_delaunay_triangles(piece))
            parts = parts[shapely.area(parts) > 1e-9]
            coords = shapely.get_coordinates(shapely.orient_polygons(parts))
            if len(coords):
                pieces.append(coords.reshape(-1, 4, 2)[:, :3])
    for block in np.array_split(np.concatenate(pieces), max(1, (sum(map(len, pieces)) + 14999) // 15000)) if pieces else []:
        xy = block.reshape(-1, 2)
        yield np.c_[xy, ground_z(*xy.T) + .08], np.arange(len(xy)).reshape(-1, 3), xy / 4.


A_CREEK = 90337123


def _triangle_polygons(triangles):
    return shapely.polygons(triangles[..., :2])


def _wet_footprint(triangles):
    merged = shapely.union_all(_triangle_polygons(triangles), grid_size=1e-8)
    # GEOS の 10 nm スナップで、同一直線上の共有辺が数 nm 幅の閉環に
    # なることがある。丸め幅に収まる穴だけを共有辺に戻す。
    polygons = []
    for poly in _polys(merged):
        if poly.area < 1e-12:
            continue
        rings = [ring for ring in poly.interiors if not Polygon(ring).buffer(-1e-8).is_empty]
        polygons.append(Polygon(poly.exterior, rings))
    return shapely.union_all(polygons)


def shore_samples(triangles, connecting=(), step=.25):
    """実水面の外周を採り、別の水面に続く切り口を水際から除く。"""
    from .valley_fit import Surfaces

    boundary = _wet_footprint(triangles).boundary
    lines = [line for line in shapely.get_parts(boundary) if line.length > 1e-6]
    if not lines:
        return np.empty((0, 3))
    xy = np.concatenate([shapely.get_coordinates(shapely.line_interpolate_point(
        line, np.r_[np.arange(0., line.length, step), line.length])) for line in lines])
    if len(connecting):
        xy = xy[~shapely.intersects_xy(_wet_footprint(connecting).buffer(1e-6), *xy.T)]
    return np.c_[xy, Surfaces(triangles).height(xy)]


def _replace_bed_or_water(mesh, cut, patches, name, frozen_normals):
    from . import kn5

    v, f, uv = mesh
    f = kn5._triangles(f)
    old = v[f]
    polys = _triangle_polygons(old)
    shapely.prepare(cut)
    hit = shapely.intersects(cut, polys)
    hit[hit] = shapely.area(shapely.intersection(cut, polys[hit])) > 1e-9
    patches = list(patches)
    if not hit.any() and not patches:
        return mesh
    saved = kn5._normals(v @ kn5.C.T, f) if frozen_normals is not None else None
    explicit = [saved, *[None for _ in patches]]
    # 触っていない面の頂点番号・UVを保つ。切り口の旧高さは面の平面から取る。
    for indices, remainder in zip(f[hit], shapely.difference(polys[hit], cut)):
        if remainder.area <= 1e-9:
            continue
        coords = shapely.get_coordinates(shapely.orient_polygons(shapely.constrained_delaunay_triangles(remainder)))
        if not len(coords):
            continue
        xy = coords.reshape(-1, 4, 2)[:, :3].reshape(-1, 2)
        tri = v[indices]
        weights = np.linalg.solve((tri[1:, :2] - tri[0, :2]).T, (xy - tri[0, :2]).T).T
        z = tri[0, 2] + weights @ (tri[1:, 2] - tri[0, 2])
        tex = uv[indices].astype(np.float32).astype(float)
        tex = tex[0] + weights @ (tex[1:] - tex[0])
        patches.append((np.c_[xy, z], np.arange(len(xy)).reshape(-1, 3), tex))
        if saved is not None:
            original_normals = saved[indices]
            explicit.append(original_normals[0] + weights @ (original_normals[1:] - original_normals[0]))
    verts, faces, uvs = [v], [f[~hit]], [uv]
    offset = len(v)
    for p, t, tex in patches:
        faces.append(t + offset)
        verts.append(p)
        uvs.append(tex)
        offset += len(p)
    out_v, out_f, out_uv = np.concatenate(verts), np.concatenate(faces), np.concatenate(uvs)
    live, index = np.unique(out_f, return_inverse=True)
    out_v, out_f, out_uv = out_v[live], index.reshape(-1, 3), out_uv[live]
    if saved is not None and len(out_f):
        normals = kn5._normals(out_v @ kn5.C.T, out_f)
        explicit = np.concatenate([normal if normal is not None else np.full_like(p, np.nan)
                                   for normal, p in zip(explicit, verts)])[live]
        stable = np.isfinite(explicit[:, 0])
        normals[stable] = explicit[stable]
        frozen_normals[name] = normals
    return out_v, out_f, out_uv


def _clip_water(triangle, depth):
    # 水位と地面は同じ三角形上で一次なので、辺の交点を線形補間できる。
    points = []
    for i in range(3):
        a, b = depth[i - 1], depth[i]
        if (a > 0) != (b > 0):
            points.append(triangle[i - 1] + a / (a - b) * (triangle[i] - triangle[i - 1]))
        if b > 0:
            points.append(triangle[i])
    if len(points) == 4:
        return [points[:3], [points[0], *points[2:]]]
    return [points] if len(points) == 3 else []


def _opening_outline(poly, old_outline):
    """新旧の境で旧水面の左右端を分割点にする。"""
    def split_ring(ring):
        line = shapely.LineString(ring)
        rests = shapely.get_coordinates(line.intersection(old_outline.boundary))
        if not len(rests):
            return ring
        coords = np.asarray(ring.coords)
        station = np.r_[0., np.linalg.norm(np.diff(coords, axis=0), axis=1).cumsum()]
        cross = shapely.line_locate_point(line, shapely.points(rests))
        edge = np.unique(np.clip(np.r_[cross - .002, cross, cross + .002], 0., line.length))
        inserted = shapely.get_coordinates(shapely.line_interpolate_point(line, edge))
        order = np.argsort(np.r_[station, edge], kind="stable")
        result = np.r_[coords, inserted][order]
        return result[np.r_[True, (np.diff(result, axis=0) != 0.).any(axis=1)]]

    return shapely.MultiPolygon([Polygon(split_ring(p.exterior), [split_ring(r) for r in p.interiors])
                                 for p in _polys(poly)])


def _wet_mesh(poly, desired_z, ground_z, near, original):
    from .valley_fit import Surfaces

    old_water = Surfaces(original)
    old_outline = _wet_footprint(original)
    poly = _opening_outline(poly, old_outline)
    boundary = poly.boundary
    # 地面の各三角形を同じ比率で分割して、水位を最大0.71m間隔で採る。
    n = 4
    ij = np.array([(i, j) for i in range(n + 1) for j in range(n + 1 - i)])
    weights = np.c_[n - ij.sum(axis=1), ij] / n
    index = {tuple(p): k for k, p in enumerate(ij)}
    sub = []
    for i, j in ij:
        if i + j >= n:
            continue
        sub.append([index[i, j], index[i + 1, j], index[i, j + 1]])
        if i + j + 1 < n:
            sub.append([index[i + 1, j], index[i + 1, j + 1], index[i, j + 1]])
    tiles = []
    for v, f, _ in _drape_bed(poly, ground_z, near):
        xy = (weights @ v[f, :2]).reshape(-1, 2)
        ground = ground_z(*xy.T)
        intended = desired_z(*xy.T)
        old = old_water.height(xy)
        at_edge = np.minimum(intended, ground)
        points = shapely.points(xy)
        blend = np.clip(shapely.distance(points, boundary), 0, 1)
        # 接合の幅の端だけを閉じ、隣の陸側へ旧水面の高さを持ち越さない。
        closing = (blend <= 1e-8) & shapely.dwithin(points, old_outline.boundary, 1e-6)
        connected = np.isfinite(old) & ~closing
        at_edge[connected] = old[connected]
        z = at_edge + (intended - at_edge) * blend
        depth = np.where(np.abs(z - ground) <= 1e-9, 0., z - ground)
        faces = (np.arange(len(f))[:, None, None] * len(weights) + sub).reshape(-1, 3)
        faces = faces[(depth[faces] > 0).any(axis=1)]
        points = np.c_[xy, z]
        tiles += [tri for t in faces for tri in _clip_water(points[t], depth[t])]
    if tiles:
        triangles = np.asarray(tiles)
        triangles = triangles[shapely.area(_triangle_polygons(triangles)) > 1e-10]
        for start in range(0, len(triangles), 20000):
            verts = triangles[start:start + 20000].reshape(-1, 3)
            yield verts, np.arange(len(verts)).reshape(-1, 3), verts[:, :2] / 4.


def _creek_bed(mesh, area, ground, before_ground, near):
    """掘削面の川原を張り、端の25cmだけ旧面と同じ高さへ戻す。"""
    from . import kn5

    v, f, _ = mesh
    triangles = v[kn5._triangles(f)]
    polygons = _triangle_polygons(triangles)
    touch = shapely.intersects(area, polygons)
    if not touch.any():
        return
    boundary = area.boundary
    transition_width = .25
    transition = boundary.buffer(transition_width)
    for tri, piece in zip(triangles[touch], shapely.intersection(area, polygons[touch])):
        if piece.area <= 1e-9:
            continue
        if before_ground is None:
            raise ValueError("Aの川原には掘削前の地面の高さが必要です")
        normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        for poly in (piece.intersection(transition), piece.difference(transition)):
            for verts, faces, uv in _drape_bed(poly, ground, near):
                xy = verts[:, :2]
                old_z = tri[0, 2] - (xy - tri[0, :2]) @ normal[:2] / normal[2]
                old_gap = old_z - before_ground(xy)
                distance = shapely.distance(shapely.points(xy), boundary)
                blend = np.clip(distance / transition_width, 0, 1)
                # 旧面の川原自身が高い所でも、地面との差を増やさず連続にする。
                gap = old_gap + blend * (np.minimum(old_gap, .08) - old_gap)
                verts[:, 2] += gap - .08
                verts[distance <= 1e-8, 2] = old_z[distance <= 1e-8]
                yield verts, faces, uv


def _ground_pieces(triangles, domain):
    """外周に掛かる面だけ多角形で分割し、標高を元の地形三角形の平面上に戻す。"""
    polygons = _triangle_polygons(triangles)
    shapely.prepare(domain)
    inside = shapely.covers(domain, polygons)
    blocks = [triangles[inside]]
    edge = shapely.intersects(domain, polygons) & ~inside
    for tri, clipped in zip(triangles[edge], shapely.intersection(polygons[edge], domain)):
        if clipped.area <= 1e-10:
            continue
        coords = shapely.get_coordinates(shapely.orient_polygons(shapely.constrained_delaunay_triangles(clipped)))
        if not len(coords):
            continue
        xy = coords.reshape(-1, 4, 2)[:, :3].reshape(-1, 2)
        weights = np.linalg.solve((tri[1:, :2] - tri[0, :2]).T, (xy - tri[0, :2]).T).T
        z = tri[0, 2] + weights @ (tri[1:, 2] - tri[0, 2])
        blocks.append(np.c_[xy, z].reshape(-1, 3, 3))
    return np.concatenate(blocks)


def _low_flow_corridor(center, width, outer_radius):
    """低水路と両端の旧川筋をつなぐ探索域。実際の岸は内部の地形との交線で決まる。"""
    radius = width / 2
    stations = np.arange(0., center.length + 1., 1.)
    stations[-1] = center.length
    # 本流の上下流で既存の水面に戻る42mは、従来の川幅を取り込む。
    end_distance = np.minimum(stations, center.length - stations)
    radii = radius + (outer_radius - radius) * np.clip(1 - end_distance / 42., 0., 1.)
    centers = shapely.line_interpolate_point(center, stations)
    return center.buffer(radius).union(shapely.union_all(shapely.buffer(centers, radii, quad_segs=12)))


def seam_bank_height(ground, old, cut_distance, bank_distance, seam_width):
    """切り口で旧水位を保ち、旧水際の内側で地面から水面へ連続的に上げる。"""
    width = np.minimum(.62, .015 + .78 * cut_distance)
    t = np.clip(bank_distance / width, 0., 1.)
    weight = t * t * (3 - 2 * t)
    weight[(cut_distance <= 1e-7) & (bank_distance > 1e-7)] = 1.
    connects = np.isfinite(old) & (cut_distance <= seam_width)
    return ground + weight * (np.where(connects, old, ground) - ground)


def _refined_stream(index, stream, base_water, base_bed, reach, near, area, normals, keep_out):
    """床の三角形と水位の交線を水際とする。旧水面へ割り戻すのは外周の接続断面だけ。"""
    from . import kn5
    from .valley_fit import Surfaces

    base_water, base_bed = list(base_water), list(base_bed)
    manual = "width_m" in reach.valley.water
    low_flow_width = reach.valley.water.get("low_flow_width_m")
    center = LineString(reach.valley.line)
    local = area.intersection(reach.bed_cut)
    # 支流は手入力小橋の下を横切る。地形の低い沢の部分にだけ流れる。
    original = np.concatenate([v[kn5._triangles(f)] for v, f, _ in base_water])
    original = original[shapely.intersects(local.buffer(1.), _triangle_polygons(original))]
    old_outline = _wet_footprint(original)
    if manual:
        channel = center.buffer(reach.valley.water["width_m"] / 2)
        domain = local.intersection(channel.union(old_outline))
    else:
        domain = local.intersection(reach.cut).difference(keep_out)
        if low_flow_width is not None:
            domain = domain.intersection(_low_flow_corridor(center, low_flow_width, reach.valley.reach_m))

    for name, items in (("water", base_water), ("riverbed", base_bed)):
        for j, mesh in enumerate(items):
            part = f"{name}_{index:03d}_{j:02d}"
            v, f, uv = _replace_bed_or_water(mesh, local, (), part, normals)
            if len(f):
                yield part, v, f, uv, name, True

    if domain.is_empty:
        return
    domain = _opening_outline(domain, old_outline)
    fine = near["valley_vertices"][near["valley_faces"]]
    lo_x, lo_y, hi_x, hi_y = domain.bounds
    xy = fine[..., :2]
    near_domain = ((xy[..., 0].min(axis=1) <= hi_x) & (xy[..., 0].max(axis=1) >= lo_x)
                   & (xy[..., 1].min(axis=1) <= hi_y) & (xy[..., 1].max(axis=1) >= lo_y))
    fine = fine[near_domain]
    if manual or low_flow_width is not None:
        # 三角形を切る前に細分し、共有辺に片側だけ新たな中点を出さない。
        a, b, c = fine.transpose(1, 0, 2)
        ab, bc, ca = (a + b) / 2, (b + c) / 2, (c + a) / 2
        fine = np.concatenate([np.stack(block, axis=1) for block in
                               ((a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca))])
    # 2mの長辺が旧水面の両岸を横切る接続断面では、片方の頂点の旧水位を
    # 反対側の陸まで補間しない。旧水面の岸を明示的な分割線にする。
    seam = local.boundary.buffer(.8).intersection(old_outline)
    rim_band = seam.intersection(old_outline.boundary.buffer(.0001))
    domains = (domain.difference(seam), domain.intersection(seam.difference(rim_band)),
               domain.intersection(rim_band))
    tris = np.concatenate([_ground_pieces(fine, part) for part in domains if not part.is_empty])
    if not len(tris):
        return

    pts, index_map = np.unique(tris.reshape(-1, 3), axis=0, return_inverse=True)
    index_map = index_map.reshape(-1, 3)
    xy = pts[:, :2]
    points = shapely.points(xy)
    ground = pts[:, 2]
    station = shapely.line_locate_point(center, points)
    level = np.interp(station, reach.station, reach.height)
    old = Surfaces(original).height(xy)
    # 旧水面とつなぐ端面の中だけ、その三角形の高さに戻す。
    edge_d = shapely.distance(points, local.boundary)
    border = shapely.distance(points, domain.boundary)
    if manual:
        old_channel = np.isfinite(old)
        level[old_channel] = old[old_channel]
        max_depth = np.where(old_channel, .32, reach.valley.water["depth_m"])
        depth = np.clip(level - ground, .015, max_depth)
        level = ground + depth * np.clip(border / .20, 0, 1)
    seam_width = reach.valley.water.get("shore_blend_m", 1.)
    blend = np.clip((edge_d if manual else border) / seam_width, 0., 1.)
    seam_z = seam_bank_height(np.minimum(level, ground), old, edge_d,
                              shapely.distance(points, old_outline.boundary), seam_width)
    level = seam_z + blend * (level - seam_z)
    depth = level - ground
    if low_flow_width is not None:
        # 厚さ1.5mmまでの水膜は河床のまま描き、砂利と水のほぼ共面のちらつきを避ける。
        depth -= .0015
    depth[np.abs(depth) < 1e-8] = 0.
    water_points = np.c_[xy, level]
    wet = depth[index_map] > 0
    full = wet.all(axis=1)
    pieces = [water_points[index_map[full]]]
    for face in index_map[wet.any(axis=1) & ~full]:
        pieces.extend(np.asarray(t)[None] for t in _clip_water(water_points[face], depth[face]))
    triangles = np.concatenate(pieces)
    triangles = triangles[shapely.area(_triangle_polygons(triangles)) > 1e-10]
    if not manual and len(triangles):
        # 低い隣の斜面だけが水位を下回る飛び地には河川水を置かない。
        channels = [p for p in _polys(_wet_footprint(triangles)) if p.intersects(center)]
        connected_water = shapely.union_all(channels)
        triangles = triangles[shapely.intersects_xy(connected_water, *triangles.mean(axis=1)[:, :2].T)]
    for n, start in enumerate(range(0, len(triangles), 19000)):
        v = triangles[start:start+19000].reshape(-1, 3)
        yield f"water_{index:03d}_valley_{n:02d}", v, np.arange(len(v)).reshape(-1, 3), v[:, :2]/4, "water", True


def _scoped_stream(index, stream, base_water, base_bed, reach, levels, ground, near, area, normals, keep_out,
                   before_ground):
    from . import kn5

    if near is not None and "valley_faces" in near and reach is not None:
        yield from _refined_stream(index, stream, base_water, base_bed, reach, near, area, normals, keep_out)
        return
    if reach is not None and "width_m" in reach.valley.water:
        # 旧格子の保存ファイルを組み立て直す場合。細い手入力の流れは細分面と対で生成する。
        reach = None
    wet = []
    if reach is not None:
        original = np.concatenate([v[kn5._triangles(f)] for v, f, _ in base_water])
        old_outline = _wet_footprint(original)
        # 下流端をまたぐ旧水面も、掘削面に掛かる分だけ新しい水位に接続する。
        domain = reach.cut.union(old_outline).intersection(area).difference(keep_out)
        wet = list(_wet_mesh(domain, surface_z(stream, levels, LIFT), ground, near, original))
        footprint = _wet_footprint(np.concatenate([v[f] for v, f, _ in wet])) if wet else Polygon()
        retained_water = old_outline.difference(area)
        bank = stream.line.buffer(stream.width_m / 2 + BED_W, cap_style="flat")
        # 水位の切り替え端を越えても、掘削面に掛かる旧川原は描画地面に沿わせる。
        bed_cut = bank.intersection(area).difference(keep_out).difference(footprint.union(retained_water))
    for name, items in (("water", base_water), ("riverbed", base_bed)):
        for j, mesh in enumerate(items):
            part = f"{name}_{index:03d}_{j:02d}"
            v, f, uv = mesh
            if reach is not None and name == "water":
                v, f, uv = _replace_bed_or_water(mesh, area, (), part, normals)
            elif name == "riverbed":
                patches = _creek_bed(mesh, area, ground, before_ground, near) if reach is None else ()
                v, f, uv = _replace_bed_or_water(mesh, area, patches, part, normals)
            if len(f):
                yield part, v, f, uv, name, True
    if reach is not None:
        for name, items in (("water", wet), ("riverbed", _drape_bed(bed_cut, ground, near))):
            for j, mesh in enumerate(items):
                yield f"{name}_{index:03d}_valley_{j:02d}", *mesh, name, True


def meshes(streams, road, levels=None, ground_z=None, *, near=None, changed_area=None, frozen_normals=None,
           before_ground=None):
    """水面と川原を生成する。掘った区間は近景メッシュに追従させる。"""
    keep_out = road.buffer(0.5) if not road.is_empty else road
    levels = levels or {}
    out = []
    for k, st in enumerate(streams):
        wtr = st.water().difference(keep_out)
        bed = st.line.buffer(st.width_m / 2 + BED_W, cap_style="flat").difference(st.water()).difference(keep_out)
        reach = levels.get(st.osm_id)
        if changed_area is not None and (reach is not None or st.osm_id == A_CREEK):
            out.extend(_scoped_stream(k, st, grid_drape(wtr, st.z_fn(LIFT), cell=2.),
                                      grid_drape(bed, st.z_fn(.15), cell=2.), reach, levels,
                                      ground_z, near, changed_area, frozen_normals, keep_out, before_ground))
            continue

        def join_water(x, y, z):
            points = shapely.points(x, y)
            distance = shapely.distance(reach.cut, points)
            near_join = (distance < 2.) & shapely.intersects_xy(reach.bed_cut, x, y)
            _, t, _, _ = reach.valley.project(x[near_join], y[near_join])
            shifted = np.copy(z)
            level = np.interp(t, reach.station, reach.height)
            shifted[near_join] += (1 - distance[near_join] / 2.) * (level - z[near_join])
            return shifted

        for name, poly, dz in (("water", wtr, LIFT), ("riverbed", bed, 0.15)):
            original_z = st.z_fn(dz)
            for j, mesh in enumerate(grid_drape(poly, original_z, cell=2.0)):
                if reach is not None:
                    mesh = _outside_water_cut(mesh, reach.cut if name == "water" else reach.bed_cut, original_z,
                                              join=join_water if name == "water" else None)
                if len(mesh[1]):
                    out.append((f"{name}_{k:03d}_{j:02d}", *mesh, name, True))
        if reach is not None:
            wet = reach.polygon().difference(keep_out)
            bank = st.line.buffer(st.width_m / 2 + BED_W, cap_style="flat").intersection(reach.bed_cut)
            bank = bank.difference(wet.union(wtr.difference(reach.cut))).difference(keep_out)
            for j, mesh in enumerate(grid_drape(wet.intersection(reach.cut), surface_z(st, levels, LIFT), cell=.5)):
                out.append((f"water_{k:03d}_valley_{j:02d}", *mesh, "water", True))
            for j, mesh in enumerate(_drape_bed(bank, ground_z, near)):
                out.append((f"riverbed_{k:03d}_valley_{j:02d}", *mesh, "riverbed", True))
    return out


def falls_from_osm(elements, streams, near_m=3.0):
    """OSM の堰・砂防ダム（waterway=weir|dam）のうち、川の線から near_m 以内のもの：[{"line", "z_top", "z_bottom", "down"}]。"""
    out = []
    for w in elements:
        if w.get("type") != "way" or w.get("tags", {}).get("waterway") not in ("weir", "dam") or len(w.get("geometry", [])) < 2:
            continue
        ln = LineString(_xy(w))
        st = min(streams, key=lambda s: s.line.distance(ln), default=None)
        if st is None or st.line.distance(ln) > near_m:
            continue
        hit = st.line.intersection(ln)
        c = hit if hit.geom_type == "Point" else ln.interpolate(0.5, normalized=True)
        t = st.line.project(c)
        a = np.array(st.line.interpolate(max(t - 1.0, 0)).coords[0]); b = np.array(st.line.interpolate(min(t + 1.0, st.line.length)).coords[0])
        z = float(st.z_at(t))
        out.append({"line": ln, "z_top": z + 0.2, "z_bottom": z - WEIR_H, "down": (b - a) / max(np.linalg.norm(b - a), 1e-9)})
    return out


def feature_meshes(falls, springs, prefix="water"):
    """堰・砂防ダム（コンクリートの段＋下流側の白い水の落ちる面）と、湧き水の水くみ場（屋根・水場）。"""
    out = []
    for k, fl in enumerate(falls):
        c = np.asarray(fl["line"].coords); d = fl["down"]
        a, b = c[0], c[-1]; zt, zb = fl["z_top"], fl["z_bottom"]
        th = 0.8
        q = np.array([a, b, b + d * th, a + d * th])
        v = np.r_[np.c_[q, np.full(4, zt)], np.c_[q, np.full(4, zb)]]
        f = np.array([[0, 1, 2, 3], [4, 7, 6, 5], [0, 4, 5, 1], [1, 5, 6, 2], [2, 6, 7, 3], [3, 7, 4, 0]], np.int32)
        out.append((f"{prefix}_weir_{k:02d}", v, f, v[:, :2] / 2.0, "concrete", True))
        fa = np.array([a + d * (th + 0.05), b + d * (th + 0.05)])
        vf = np.array([[*fa[0], zb], [*fa[1], zb], [*fa[1], zt - 0.05], [*fa[0], zt - 0.05]])
        out.append((f"{prefix}_fall_{k:02d}", vf, np.array([[0, 1, 2, 3], [3, 2, 1, 0]], np.int32), vf[:, [0, 2]], "water_fall", True))
    for k, sp in enumerate(springs):
        x, y, z, h = sp["x"], sp["y"], sp["z"], sp.get("facing", 0.0)
        d = np.array([np.sin(h), np.cos(h)]); n = np.array([d[1], -d[0]])
        corners = lambda w, l: np.array([[x, y] + n * w + d * l, [x, y] - n * w + d * l, [x, y] - n * w - d * l, [x, y] + n * w - d * l])
        tr = corners(0.6, 0.3)
        v = np.r_[np.c_[tr, np.full(4, z)], np.c_[tr, np.full(4, z + 0.6)]]
        f = np.array([[4, 5, 6, 7], [0, 4, 5, 1], [1, 5, 6, 2], [2, 6, 7, 3], [3, 7, 4, 0]], np.int32)
        out.append((f"{prefix}_spring_trough_{k:02d}", v, f, v[:, :2], "concrete", True))
        rf = corners(1.2, 1.0)
        vr = np.c_[rf, np.array([z + 2.3, z + 2.3, z + 2.0, z + 2.0])]
        out.append((f"{prefix}_spring_roof_{k:02d}", vr, np.array([[0, 1, 2, 3], [3, 2, 1, 0]], np.int32), vr[:, :2], "roof_dark", True))
        for j, pc in enumerate(corners(1.0, 0.8)):
            pv = np.array([[*(pc + [-0.06, -0.06]), z], [*(pc + [0.06, -0.06]), z], [*(pc + [0.06, -0.06]), z + 2.2], [*(pc + [-0.06, -0.06]), z + 2.2]])
            out.append((f"{prefix}_spring_post_{k:02d}_{j}", pv, np.array([[0, 1, 2, 3], [3, 2, 1, 0]], np.int32), pv[:, [0, 2]], "wood", True))
    return out


def load_elements():
    """S4 の川・沢・堰（data/raw/osm_s4_places.json の waterway）と湧き水（osm_s4_nature.json）。"""
    import json
    from . import config
    el = json.loads((config.DATA_RAW / "osm_s4_places.json").read_text(encoding="utf-8"))["elements"]
    nat = json.loads((config.DATA_RAW / "osm_s4_nature.json").read_text(encoding="utf-8"))["elements"]
    return [e for e in el if "waterway" in e.get("tags", {})], [e for e in nat if e.get("tags", {}).get("natural") == "spring"]


def load(sec, sampler=None):
    """S4（峠の先）の川・沢（Stream のリスト）、堰・砂防ダム、湧き水。"""
    from . import gsi, side_roads
    ways, springs = load_elements()
    streams = from_osm(ways, sec, sampler or gsi.default_sampler(), s_from=side_roads.s4_start(sec))
    lake, level = load_lake()
    if lake is not None:   # 宮ヶ瀬湖に入る川は、湖の手前で湖の水位につなぐ
        streams = attach_to_lake(streams, lake, level)
    return streams, falls_from_osm(ways, streams), springs


def channels(sec, streams, formation):
    """terrain.carve に渡す溝。本線の路面（橋を除く）のまわり 2m は彫らない。"""
    keep = formation.difference(_bridges(sec)).buffer(2.0)
    out = []
    for st in streams:
        poly, zf = st.channel()
        poly = poly.difference(keep)
        if not poly.is_empty:
            out.append((poly, zf))
    return out


def spring_items(springs, sec, ground_z, near_m=200.0):
    """湧き水の水くみ場：OSM の点に、道の方を向けて置く（道から near_m 以内）。"""
    out = []
    for e in springs:
        x, y = (float(v) for v in geo.to_local(e["lat"], e["lon"]))
        i = int(np.argmin(np.hypot(sec.x - x, sec.y - y)))
        if np.hypot(sec.x[i] - x, sec.y[i] - y) > near_m:
            continue
        out.append({"x": x, "y": y, "z": float(np.atleast_1d(ground_z(np.array([x]), np.array([y])))[0]),
                    "facing": float(np.arctan2(sec.x[i] - x, sec.y[i] - y)), "name": e["tags"].get("name", "")})
    return out


def lake_from_osm(elements):
    """湖の relation（outer・inner の way をつなぐ）を多角形（ローカル座標）に。outer・inner は way の向きがばらばらでもよい。"""
    from shapely.ops import polygonize, linemerge
    nodes = {e["id"]: e for e in elements if e.get("type") == "node"}
    ways = {e["id"]: e for e in elements if e.get("type") == "way"}
    rel = next(e for e in elements if e.get("type") == "relation")
    rings = {"outer": [], "inner": []}
    for m in rel.get("members", []):
        w = ways.get(m.get("ref"))
        if m.get("type") != "way" or w is None or m.get("role") not in rings:
            continue
        pts = [geo.to_local(nodes[n]["lat"], nodes[n]["lon"]) for n in w.get("nodes", []) if n in nodes]
        if len(pts) >= 2:
            rings[m["role"]].append(LineString(np.array(pts, float)))
    outer = unary_union(list(polygonize(linemerge(rings["outer"])))) if rings["outer"] else Polygon()
    inner = unary_union(list(polygonize(linemerge(rings["inner"])))) if rings["inner"] else Polygon()
    return outer.difference(inner).buffer(0)


def lake_meshes(poly, level, keep_out, cell=8.0):
    """湖面（平ら、高さ level）。keep_out（本線・橋・分かれ道）の 0.5m 内には描かない。"""
    g = poly.difference(keep_out.buffer(0.5)) if not keep_out.is_empty else poly
    return [(f"lake_{k:02d}", v, f, uv, "water", True)
            for k, (v, f, uv) in enumerate(grid_drape(g, lambda x, y: np.full(np.shape(x), float(level)), LIFT, cell=cell))]


def load_lake():
    """宮ヶ瀬湖（data/raw/osm_s5_lake_rel.json）と水位（data/manual/s5_water.json）。なければ (None, None)。"""
    import json
    from . import config
    p, q = config.DATA_RAW / "osm_s5_lake_rel.json", config.DATA_MANUAL / "s5_water.json"
    if not (p.exists() and q.exists()):
        return None, None
    return lake_from_osm(json.loads(p.read_text(encoding="utf-8"))["elements"]), float(json.loads(q.read_text(encoding="utf-8"))["lake_level_m"])


def attach_to_lake(streams, lake, level):
    """湖の中の川は描かない（線を湖の外だけにする）。水面は湖の水位より下げない（湖に入る所で段差なくつながる）。"""
    out = []
    for st in streams:
        rest = st.line.difference(lake)
        for piece in _polys_lines(rest):
            if piece.length < 5.0:
                continue
            t0, t1 = st.line.project(Point(piece.coords[0])), st.line.project(Point(piece.coords[-1]))
            tt = np.linspace(t0, t1, max(2, int(piece.length / STEP_M) + 1))
            z = np.maximum(st.z_at(tt), level)
            out.append(Stream(LineString([st.line.interpolate(v).coords[0] for v in tt]), st.width_m, st.kind, np.minimum.accumulate(z), st.osm_id))
    return out


def manual_falls(spec, sec, ground_z):
    """data/manual/s5_water.json の falls（ストリートビュー・OSM で見た滝）：feature_meshes の形に。
    facing_deg がなければ一番近い本線の点の方へ落ちる。"""
    out = []
    for f in spec:
        p = np.array([float(v) for v in geo.to_local(*f["at"])])
        if f.get("facing_deg") is None:
            i = int(np.argmin(np.hypot(sec.x - p[0], sec.y - p[1]))); d = np.array([sec.x[i], sec.y[i]]) - p
        else:
            r = np.radians(float(f["facing_deg"])); d = np.array([np.sin(r), np.cos(r)])
        d = d / max(np.linalg.norm(d), 1e-9); n = np.array([-d[1], d[0]]); w = float(f.get("width_m", 2.0)) / 2
        zb = float(np.atleast_1d(ground_z(np.array([p[0]]), np.array([p[1]])))[0])
        out.append({"line": LineString([p - n * w, p + n * w]), "z_top": zb + float(f.get("height_m", 3.0)), "z_bottom": zb, "down": d})
    return out
