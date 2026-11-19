"""valleys.json の谷筋。XY はコースの東・北方向のメートル。

reach_m は谷底線から変更面の外周までの上限。2m格子では対角長2.83mを
引いた範囲の頂点を対象にする。その内側の taper_m の帯で上限標高を
元の斜面へ戻すため、bank_slopes がそのまま効くのはさらに内側。
"""
import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import shapely

from . import config, geo
from .crosssection import GUTTER_WIDTH

SITE_REACH_M = 150.0
SHOULDER_M = 0.75
TAPER_RELIEF_M = 8.0


@dataclass(frozen=True)
class Valley:
    id: str
    section: str
    center_m: float
    line: np.ndarray
    bed: dict
    width_m: float
    bank_left: float
    bank_right: float
    reach_m: float
    water: dict | None = None

    @classmethod
    def from_row(cls, row):
        line = np.asarray(row["line_xy"], float)
        width, reach = float(row["width_m"]), float(row["reach_m"])
        left, right = (float(row["bank_slopes"][side]) for side in ("left", "right"))
        bed = row["bed"]
        mode = bed["mode"]
        taper = bed.get("taper_m", 5.0)
        end_taper = bed.get("end_taper_m", taper)
        values = np.asarray(bed["z_m" if mode == "absolute" else "depth_m"], float)
        water = row.get("water")
        if (line.ndim != 2 or line.shape[1] != 2 or len(line) < 2
                or not np.isfinite(line).all() or np.any(np.all(np.diff(line, axis=0) == 0, axis=1))
                or not np.isfinite([width, reach, left, right, row["center_m"]]).all()
                or not np.isfinite([taper, end_taper]).all() or taper <= 0 or end_taper < 0
                or not 0 < width / 2 < reach or min(left, right) <= 0
                or mode not in {"absolute", "endpoints", "relative"} or not np.isfinite(values).all()
                or (mode == "absolute" and values.shape not in {(), (len(line),)})
                or (mode != "absolute" and (values.ndim != 0 or values < 0))
                or (water is not None and (mode != "absolute" or not np.isfinite(water["depth_m"])
                                           or water["depth_m"] <= 0 or water["osm_id"] <= 0))
                or not row["source"] or not row["evidence"]):
            raise ValueError(f"谷筋の指定が不正です: {row['id']}")
        return cls(row["id"], row["section"], row["center_m"], line, bed, width, left, right, reach, water)

    def project(self, x, y):
        """折れ線の最近傍、線の始点からの距離、右を負とする横距離。"""
        p = np.column_stack((x, y))
        vector = np.diff(self.line, axis=0)
        lengths = np.linalg.norm(vector, axis=1)
        unit = vector / lengths[:, None]
        delta = p[:, None, :] - self.line[None, :-1, :]
        along = np.clip(np.einsum("nki,ki->nk", delta, unit), 0, lengths)
        residual = delta - along[:, :, None] * unit
        dist2 = np.sum(residual**2, axis=2)
        i = np.argmin(dist2, axis=1)
        row = np.arange(len(p))
        distance = np.sqrt(dist2[row, i])
        side = unit[i, 0] * residual[row, i, 1] - unit[i, 1] * residual[row, i, 0]
        arc = np.r_[0.0, np.cumsum(lengths)]
        return p - residual[row, i], arc[i] + along[row, i], np.copysign(distance, side), arc

    def profile(self, x, y, sampler, halo):
        """岸の設計上限とすり付け係数。reach_m の外周は面を守る余白を含む。"""
        xy, along, signed, arc = self.project(x, y)
        mode = self.bed["mode"]
        if mode == "absolute":
            z = np.broadcast_to(self.bed["z_m"], arc.shape)
            bottom = np.interp(along, arc, z)
        else:
            xy = self.line[[0, -1]] if mode == "endpoints" else xy
            sampled = sampler.sample(*geo.from_local(xy[:, 0], xy[:, 1]))
            if not np.isfinite(sampled).all():
                raise ValueError(f"谷底の標高が欠測しています: {self.id}")
            bottom = (np.interp(along, arc[[0, -1]], sampled) if mode == "endpoints" else sampled)
            bottom = bottom - self.bed["depth_m"]
        d = abs(signed)
        side_slope = np.where(signed < 0, self.bank_right, self.bank_left)
        cap = bottom + side_slope * np.maximum(d - self.width_m / 2, 0)
        reach = self.reach_m - halo
        taper = self.bed.get("taper_m", 5.0)
        if reach <= self.width_m / 2:
            raise ValueError(f"近景格子に対して谷筋の範囲が狭すぎます: {self.id}")
        across = np.clip((reach - d) / min(taper, reach - self.width_m / 2), 0, 1)
        end_distance = np.minimum(along, arc[-1] - along)
        end_taper = self.bed.get("end_taper_m", taper)
        ends = np.clip(end_distance / end_taper, 0, 1) if end_taper else end_distance > 0
        weight = across * ends
        return cap, weight * weight * (3 - 2 * weight)

    def ceiling(self, x, y, sampler, halo):
        """端のすり付けを含む絶対標高の上限。適用済みの標高には依存しない。"""
        cap, weight = self.profile(x, y, sampler, halo)
        return np.where(weight > 0, cap + TAPER_RELIEF_M * (1 - weight), np.inf)


def load(path=None):
    path = path or config.DATA_MANUAL / "valleys.json"
    return [Valley.from_row(row) for row in json.loads(path.read_text(encoding="utf-8"))] if path.exists() else []


def site_area(sec, bounds, section, center_m):
    """区間距離の前後150mと、中心線から150m以内。端の横断面で閉じる。"""
    start, end = bounds[section]
    if not 0 <= center_m <= end - start:
        raise ValueError(f"谷筋の基準距離が区間外です: {section} {center_m}")
    a = max(sec.s[0], start + center_m - SITE_REACH_M)
    b = min(sec.s[-1], start + center_m + SITE_REACH_M)
    stations = np.r_[a, sec.s[(a < sec.s) & (sec.s < b)], b]
    line = shapely.LineString(np.c_[np.interp(stations, sec.s, sec.x), np.interp(stations, sec.s, sec.y)])
    return line.buffer(SITE_REACH_M, cap_style="flat")


def bridge_intervals(sec, spans=()):
    """OSM 橋と手入力小橋の両端。連続する OSM 橋行を一つの開口にする。"""
    steps = np.diff(np.r_[False, sec.bridge, False].astype(int))
    return [*zip(np.flatnonzero(steps > 0), np.flatnonzero(steps < 0) - 1), *spans]


def road_areas(sec, area, side_extra, spans, opening_extra, approach_extra=0.):
    """道路の保護帯と橋の端面の間の開口。開口は側溝の外までつなぐ。"""
    a, b, c, d = area.bounds
    nearby = (a - 10 <= sec.x) & (sec.x <= c + 10) & (b - 10 <= sec.y) & (sec.y <= d + 10)
    # 幅の変わる境界も既存の横断面で閉じる。飛び地は別の多角形として扱う。
    n = np.c_[-np.sin(sec.heading), np.cos(sec.heading)]
    xy = np.c_[sec.x, sec.y]
    walk_l, walk_r = (0.0 if v is None else np.asarray(v) for v in side_extra)
    left = xy + n * (sec.hw_left + GUTTER_WIDTH + walk_l + SHOULDER_M)[:, None]
    right = xy - n * (sec.hw_right + GUTTER_WIDTH + walk_r + SHOULDER_M)[:, None]
    segments = nearby[:-1] | nearby[1:]
    openings = []
    approaches = []
    for i, j in spans:
        if segments[i:j].any():
            sl = slice(i, j + 1)
            openings.append(shapely.Polygon(np.r_[left[sl] + opening_extra * n[sl],
                                                  (right[sl] - opening_extra * n[sl])[::-1]]))
        if approach_extra and segments[i:j].any():
            for a, b in ((max(0, np.searchsorted(sec.s, sec.s[i] - approach_extra) - 1), i),
                         (j, min(sec.n - 1, np.searchsorted(sec.s, sec.s[j] + approach_extra)))):
                sl = slice(a, b + 1)
                approaches.append(shapely.Polygon(np.r_[left[sl] + approach_extra * n[sl],
                                                       (right[sl] - approach_extra * n[sl])[::-1]]))
        segments[i:j] = False
    pieces = [shapely.Polygon([left[i], right[i], right[i + 1], left[i + 1]])
              for i in np.flatnonzero(segments)]
    return shapely.union_all(pieces + approaches), shapely.union_all(openings)


def protected_area(sec, area, pads=(), keep_rows=None):
    """橋の開口でも保護する既存土台とトンネル。"""
    pieces = []
    if keep_rows is not None:
        a, b, c, d = area.bounds
        nearby = (a - 10 <= sec.x) & (sec.x <= c + 10) & (b - 10 <= sec.y) & (sec.y <= d + 10)
        roofs = np.flatnonzero(nearby & keep_rows)
        pieces.extend(shapely.Point(sec.x[i], sec.y[i]).buffer(max(sec.hw_left[i], sec.hw_right[i]) + 6)
                      for i in roofs)
    for pad in pads:
        p = shapely.Polygon(pad["polygon"])
        if p.intersects(area):
            pieces.append(p)
    return shapely.union_all(pieces)


class ChangedGround:
    def __init__(self, old_tri, new_tri, *, bed_faces=None, copied=None):
        from .valley_fit import Surfaces

        self.polygon = shapely.union_all(shapely.polygons(old_tri[..., :2]))
        if copied is not None:
            original = shapely.union_all(shapely.polygons(new_tri[copied, :, :2]))
            self.polygon = self.polygon.difference(original)
        self.triangles = (old_tri, new_tri)
        self.old = Surfaces(old_tri)
        self.new = Surfaces(new_tri)
        self.bed = None if bed_faces is None else Surfaces(new_tri[bed_faces])

    def covers(self, xy):
        return shapely.intersects_xy(self.polygon, *np.asarray(xy).T)

    def height(self, xy):
        a = np.asarray(xy)
        z = self.new.height(np.atleast_2d(a))
        return z[0] if a.ndim == 1 else z

    def tree_exclusions(self, points, current_water):
        """細分面の河床材と、そこに実際に載る水を足元のXYで調べる。"""
        points = np.atleast_2d(points)
        z = self.new.height(points)
        bed = np.isfinite(self.bed.height(points))
        wet = np.isfinite(z) & (current_water.height(points) >= z - 1e-6)
        return bed, wet


def changed_ground(before, after):
    from .terrain import NEAR_BAND

    if "valley_faces" in after:
        copied = after.get("valley_coarse_remainder")
        return ChangedGround(original_triangles(before, after["valley_cells"]),
                             after["valley_vertices"][after["valley_faces"]], bed_faces=after["valley_bed"],
                             copied=copied[:, 0] >= 0 if copied is not None else None)
    a, b = before["Z"], after["Z"]
    if a.shape != b.shape or a.dtype != b.dtype:
        raise ValueError("近景の Z の形状または型が異なります")
    changed = a != b
    changed &= ~(np.isnan(a) & np.isnan(b))
    rows, cols = np.nonzero(changed)
    cells = np.unique(np.concatenate([np.c_[rows + i, cols + j] for i in (-1, 0) for j in (-1, 0)]), axis=0)
    cells = cells[((0 <= cells) & (cells < np.array(a.shape) - 1)).all(axis=1)]
    r, c = cells.T
    d = before["D"]
    keep = (~before["H"][r, c] & (d[r, c] <= NEAR_BAND) & (d[r + 1, c] <= NEAR_BAND)
            & (d[r, c + 1] <= NEAR_BAND) & (d[r + 1, c + 1] <= NEAR_BAND))
    r, c = r[keep], c[keep]
    triangles = []
    for dr, dc in (((0, 0, 1), (0, 1, 1)), ((0, 1, 1), (0, 1, 0))):
        rr, cc = r[:, None] + dr, c[:, None] + dc
        use = (changed[rr, cc].any(axis=1) & np.isfinite(a[rr, cc]).all(axis=1) & np.isfinite(b[rr, cc]).all(axis=1))
        rr, cc = rr[use], cc[use]
        xy = np.stack((before["xs"][cc], before["ys"][rr]), axis=-1)
        triangles.append((np.concatenate((xy, a[rr, cc, None]), axis=-1),
                          np.concatenate((xy, b[rr, cc, None]), axis=-1)))
    return ChangedGround(*(np.concatenate(t) for t in zip(*triangles)))


def newly_wet(dug, before_water, after_water, points):
    """水より下に入った根元を、新旧の実際の面で比べる。"""
    points = np.atleast_2d(points)
    old_ground = dug.old.height(points)
    new_ground = dug.new.height(points)
    return ((after_water.height(points) > new_ground + 1e-6)
            & ~(before_water.height(points) > old_ground + 1e-6))


def tree_bottom(rows, x, y, *, forest=False):
    bottom = np.zeros(len(x), bool)
    for row in rows:
        if not forest and row.id[0] not in ("A", "C"):
            continue
        _, at, side, arc = row.project(x, y)
        bottom |= (at > 0) & (at < arc[-1]) & (abs(side) <= row.width_m / 2)
    return bottom


def lower_exposed_feet(verts, bottom, dug):
    """掘削で浮きが新しく出た、または増した下端を掘った面に差し込む。"""
    if dug is None or dug.polygon.is_empty:
        return
    ids = np.flatnonzero(bottom)
    feet = verts[ids]
    old, new = dug.old.height(feet), dug.new.height(feet)
    with np.errstate(invalid="ignore"):
        hit = (old - new > 1e-6) & (feet[:, 2] > new + .10)
    verts[ids[hit], 2] = new[hit] - .12


def read_before_near(path, after):
    """地形の控えが同じ格子・道路・穴マスクであることを確認する。"""
    with np.load(path) as source:
        before = dict(source)
    if before.keys() != set(after.keys()) - SURFACE_KEYS:
        raise ValueError("近景の新旧で配列名が一致しません")
    for k, a in before.items():
        b = after[k]
        if a.shape != b.shape or a.dtype != b.dtype:
            raise ValueError(f"近景の配列 {k} の形状・型が一致しません")
        if (k != "Z" or "valley_faces" in after) and not np.array_equal(a.ravel().view(np.uint8), b.ravel().view(np.uint8)):
            raise ValueError(f"近景の配列 {k} が一致しません")
    return before


def _grid_digest(near):
    digest = sha256()
    for key in ("xs", "ys", "Z"):
        digest.update(np.ascontiguousarray(near[key]))
    return digest.hexdigest()


def ground_delta(old_z, near):
    z = near["Z"]
    ids = np.flatnonzero(old_z.ravel().view(f"V{z.dtype.itemsize}") != z.ravel().view(f"V{z.dtype.itemsize}"))
    return {"indices": ids, "original_z": old_z.ravel()[ids], "digest": _grid_digest(near)}


def _delta_path(near_path):
    return near_path.with_name(near_path.stem + "_valley_delta.npz")


def save_ground_delta(near_path, delta):
    target = _delta_path(near_path)
    with TemporaryDirectory(dir=target.parent) as tmp:
        out = Path(tmp) / target.name
        np.savez_compressed(out, **delta)
        out.replace(target)


def before_for_build(path, near_path, near):
    if path is not None:
        return read_before_near(path, near)
    if "valley_faces" in near:
        return {k: v for k, v in near.items() if k not in SURFACE_KEYS}
    saved = _delta_path(near_path)
    if not saved.is_file():
        raise ValueError("谷を掘る前の近景を --before-near で指定してください")
    with np.load(saved) as delta:
        if delta["digest"].item() != _grid_digest(near):
            raise ValueError("谷の差分と近景が一致しません。--before-near を指定してください")
        z = near["Z"].copy()
        z.ravel()[delta["indices"]] = delta["original_z"]
    return {**near, "Z": z}

# 近景の元の配列はそのまま保存し、描画・当たり判定に用いる置換表を併記する。
SURFACE_KEYS = {"valley_cells", "valley_vertices", "valley_faces", "valley_bed",
                "valley_coarse_remainder", "valley_gravel", "valley_source_faces"}


def fine_topology(cells):
    """2mセルを四分割。外周では辺の途中に頂点を置かず隣の粗い辺を丸ごと共有する。"""
    from scipy.spatial import Delaunay

    cells = np.asarray(cells, int)
    present = set(map(tuple, cells))
    flag = np.array([sum(1 << i for i, (dy, dx) in enumerate(((-1, 0), (0, 1), (1, 0), (0, -1)))
                         if (r + dy, c + dx) not in present) for r, c in cells])
    templates = np.stack(np.indices((5, 5)), -1).reshape(-1, 2)
    r, c = templates.T
    sides = [(r == 0) & (c % 4 != 0), (c == 4) & (r % 4 != 0),
             (r == 4) & (c % 4 != 0), (c == 0) & (r % 4 != 0)]
    triangles = []
    for key in np.unique(flag):
        mask = np.ones(len(r), bool)
        for i, side in enumerate(sides):
            if key & (1 << i):
                mask &= ~side
        local = templates[mask]
        faces = Delaunay(local[:, ::-1]).simplices
        triangles.append(cells[flag == key, None, None, :] * 4 + local[faces])
    points, inv = np.unique(np.concatenate([t.reshape(-1, 2) for t in triangles]), axis=0, return_inverse=True)
    return points, inv.reshape(-1, 3).astype(np.int32)


def original_triangles(grid, cells):
    rr, cc = np.asarray(cells).T
    r = rr[:, None] + [0, 0, 1, 1]
    c = cc[:, None] + [0, 1, 1, 0]
    v = np.stack((grid["xs"][c], grid["ys"][r], grid["Z"][r, c]), axis=-1)
    return v[:, [[0, 1, 2], [0, 2, 3]]].reshape(-1, 3, 3)


def fit_foot_chain(chain, dug, tolerance, *, ground=None, roadside=None):
    """地面が下がり露出した線分だけ、両端を同じ量下げて地中に収める。"""
    z = chain[:,2].copy()
    if len(chain) < 2 or (dug is None and ground is None):
        return z
    segments = np.stack((chain[:-1],chain[1:]),axis=1)
    local = np.zeros(len(segments), bool)
    if dug is not None:
        local |= shapely.intersects(shapely.linestrings(segments[:, :, :2]), dug.polygon)
    if roadside is not None:
        local |= roadside
    for i in np.flatnonzero(local):
        a,b = chain[i:i+2]
        steps = max(2, int(np.ceil(np.linalg.norm(b[:2]-a[:2])/.08))+1)
        pts = a + np.linspace(0,1,steps)[:,None]*(b-a)
        if roadside is not None and roadside[i]:
            new = ground(*pts[:, :2].T)
            ok = np.isfinite(new) & (pts[:, 2] > new)
        else:
            old, new = dug.old.height(pts), dug.new.height(pts)
            ok = np.isfinite(old) & np.isfinite(new) & (old > new + 1e-6) & (pts[:, 2] > new + tolerance)
        if ok.any():
            drop = np.max(pts[ok,2]-new[ok])+.18
            z[i:i+2] = np.minimum(z[i:i+2],chain[i:i+2,2]-drop)
    return z
