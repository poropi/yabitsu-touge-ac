"""谷周辺のメッシュと保存地形を照合する。距離の単位は m。"""
import json
from collections import defaultdict
from hashlib import sha256
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import shapely
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from . import kn5, terrain, valleys


@dataclass
class Mesh:
    name: str
    verts: np.ndarray
    faces: np.ndarray
    material: str = ""
    visible: bool = True
    uv: np.ndarray | None = None
    group: str = ""
    settings: dict | None = None
    parent: str | None = None

    @property
    def triangles(self):
        return self.verts[kn5._triangles(self.faces)]

    @property
    def water_kind(self):
        if self.material in ("water", "riverbed"):
            return self.material
        if self.material.startswith("valley_edit_"):
            prefix = self.name.partition("_")[0]
            if prefix in ("water", "riverbed"):
                return prefix
        return None


@dataclass(frozen=True)
class Issue:
    category: str
    mesh: str
    xy: tuple
    amount: float
    other: str = ""
    count: int = 1
    samples: tuple = ()


def finding(category, name, points, amounts, other=""):
    i = np.argmax(amounts)
    samples = tuple((float(x), float(y), float(z)) for (x, y), z in zip(points[:, :2], amounts))
    return Issue(category, name, tuple(map(float, points[i, :2])), float(amounts[i]), other, len(points), samples)


def issue_changes(before, after, *, old_meshes=(), old_ground=None):
    """同じ場所の既存候補と、追加・悪化した候補を判定する。"""
    def key(issue, xy):
        return issue.category, issue.mesh, issue.other, tuple(round(float(v), 5) for v in xy)

    old_points, old_overlaps = {}, {}
    water_checks = {"water_float", "water_buried", "bed_over_cut"}
    old_water = defaultdict(list)
    for m in old_meshes:
        if m.water_kind:
            old_water[m.parent or m.name].append(m.triangles)
    old_surfaces = {name: Surfaces(np.concatenate(parts)) for name, parts in old_water.items()}
    for issue in before:
        if issue.category == "overlap":
            old_overlaps[issue.mesh, issue.other] = issue
            continue
        samples = issue.samples or ((*issue.xy, issue.amount),)
        for *xy, amount in samples:
            k = key(issue, xy)
            old_points[k] = max(old_points.get(k, -np.inf), amount)
    regressions = []
    for issue in after:
        if issue.category == "overlap":
            old = old_overlaps.get((issue.mesh, issue.other))
            if old is None or issue.count > old.count or issue.amount < old.amount - 1e-4:
                regressions.append(issue)
            continue
        if not issue.samples:
            if issue.amount > old_points.get(key(issue, issue.xy), -np.inf) + 1e-6:
                regressions.append(issue)
            continue
        samples = np.asarray(issue.samples)
        prior = np.array([old_points.get(key(issue, xy), -np.inf) for xy in samples[:, :2]])
        if issue.category in ("card_water", "tree_water"):
            # 旧水面ですでに水中だった個体は、水深の変化で新しい候補にしない。
            prior[np.isfinite(prior)] = np.inf
        if old_ground is not None and issue.category in water_checks and issue.mesh in old_surfaces:
            xy = samples[:, :2]
            water_z = old_surfaces[issue.mesh].height(xy)
            measured = np.isfinite(water_z)
            old_delta = water_z[measured] - old_ground(*xy[measured].T)
            prior[measured] = -old_delta if issue.category == "water_buried" else old_delta
        worse = samples[samples[:, 2] > prior + 1e-6]
        if len(worse):
            regressions.append(finding(issue.category, issue.mesh, worse[:, :2], worse[:, 2], issue.other))
    return list(before), regressions


def grid_checks(before, after, sites):
    issues, stats = [], {}
    for key in sorted(before.keys() | after.keys()):
        if key == "Z" or key in valleys.SURFACE_KEYS:
            continue
        a, b = before.get(key), after.get(key)
        if a is None or b is None or (a.shape, a.dtype, a.tobytes()) != (b.shape, b.dtype, b.tobytes()):
            issues.append(Issue("terrain_array", key, (), 0.))
    if "valley_faces" in after:
        from .valley_surface import sample_grid
        cells = after["valley_cells"]
        v, f = after["valley_vertices"], after["valley_faces"]
        original = valleys.original_triangles(before, cells)
        outline = shapely.union_all(shapely.polygons(original[:,:,:2]))
        allowed = shapely.union_all(list(sites.values()))
        tri = v[f]
        triangles = shapely.polygons(tri[:,:,:2])
        shape = shapely.coverage_union_all(triangles)
        edges = np.sort(f[:,[[0,1],[1,2],[2,0]]].reshape(-1,2), axis=1)
        unique, n = np.unique(edges,axis=0,return_counts=True)
        boundary = v[np.unique(unique[n==1])]
        dz = abs(boundary[:,2]-sample_grid(before,*boundary[:,:2].T))
        if (not shapely.covers(allowed,outline) or not shape.is_valid or not outline.equals(shape)
                or abs(shapely.area(triangles).sum()-outline.area)>1e-6
                or (n>2).any() or dz.max(initial=0)>.001
                or not np.array_equal(before["Z"],after["Z"],equal_nan=True)):
            issues.append(Issue("valley_surface", "valley_faces", (), 0.))
        stats["fine"] = {"cells":len(cells), "triangles":len(f), "boundary_edges":len(boundary),
                         "edge_error_m":float(dz.max(initial=0))}
    a, b = before["Z"], after["Z"]
    if a.shape != b.shape or a.dtype != b.dtype:
        return issues + [Issue("terrain_array", "Z", (), 0.)], stats
    bits = np.dtype(f"V{a.dtype.itemsize}")
    row, col = np.nonzero(a.view(bits) != b.view(bits))
    xs, ys = before["xs"], before["ys"]
    x, y = xs[col], ys[row]
    xy = np.c_[x, y]
    dz = a[row, col] - b[row, col]
    missing = ~np.isfinite(dz)
    issues.extend(Issue("terrain_missing", "Z", tuple(map(float, p)), 0.) for p in xy[missing])
    dz[missing] = 0.
    assigned = np.zeros(len(row), bool)
    # 窓をまたぐ三角形も判定する。メッシュと同じ南西-北東の対角線。
    neighbors = np.array([[-1, -1], [-1, 0], [0, 1], [1, 1], [1, 0], [0, -1]])
    rr, cc = row[:, None] + neighbors[:, 0], col[:, None] + neighbors[:, 1]
    complete = (rr.min(axis=1) >= 0) & (rr.max(axis=1) < len(ys)) & (cc.min(axis=1) >= 0) & (cc.max(axis=1) < len(xs))
    ring = np.c_[rr[complete].ravel(), cc[complete].ravel()]
    surrounding = b[tuple(ring.T)].reshape(-1, 6)
    depth = surrounding.min(axis=1) - b[row[complete], col[complete]]
    feet = np.stack((xs[cc[complete]], ys[rr[complete]]), axis=-1)
    tris = shapely.polygons(np.stack((np.broadcast_to(xy[complete, None], feet.shape), feet,
                                     np.roll(feet, -1, axis=1)), axis=2))
    for label, site in sites.items():
        inside = shapely.intersects_xy(site, x, y)
        enclosed = np.zeros(len(row), bool)
        enclosed[complete] = shapely.covers(site, tris).all(axis=1)
        assigned |= enclosed
        if inside.any():
            stats[label] = {"points": int(inside.sum()), "drop": float(np.max(dz[inside])),
                            "xy_min": xy[inside].min(0).tolist(), "xy_max": xy[inside].max(0).tolist()}
    for category, points, values in (
        ("terrain_outside", xy[~assigned], abs(dz[~assigned])),
        ("terrain_raised", xy[dz < 0], -dz[dz < 0]),
        ("terrain_hole", xy[complete][depth >= 1.], depth[depth >= 1.]),
    ):
        issues.extend(Issue(category, "Z", tuple(map(float, p)), float(z)) for p, z in zip(points, values))
    return issues, stats


def manifest_meshes(root):
    """ゲーム書き出し直前の配列・所属・材質・表示設定を読む。"""
    root = Path(root)
    data = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    def mapping(file, default):
        p = root / file
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else default

    hidden = set(mapping("hidden.json", []))
    retex = mapping("retex.json", {})
    groups = {n: g for g, entry in data["groups"].items() for n in entry["meshes"]}
    for name, entry in data["meshes"].items():
        with np.load(root / entry["file"]) as mesh:
            material = entry["material"]
            yield Mesh(name, mesh["verts"], mesh["faces"], material, entry["renderable"] and name not in hidden,
                       mesh["uvs"], groups[name], {"material": data["materials"].get(material),
                                                  "retexture": retex.get(name)},
                       parent=entry.get("valley_edit_parent"))


def part_triangles(root, entries, name):
    """Load the original named part and its additional material meshes."""
    mesh = []
    for key, entry in entries.items():
        if key == name or entry.get("valley_edit_parent") == name:
            with np.load(root / entry["file"]) as data:
                mesh.append(data["verts"][kn5._triangles(data["faces"])])
    if not mesh:
        raise KeyError(name)
    return np.concatenate(mesh)


def _outside_signature(mesh, area):
    # 面の重複数も保つ。面が窓をまたぐ場合も、その面全体を比較する。
    tri = mesh.triangles
    valid = ~shapely.covers(area, shapely.polygons(tri[..., :2]))
    if mesh.uv is not None:
        tri = np.concatenate((tri, mesh.uv[kn5._triangles(mesh.faces)]), axis=2)
    if not valid.any():
        return None
    rows = np.ascontiguousarray(tri[valid]).reshape(int(valid.sum()), -1)
    packed = rows.view(f"S{rows.dtype.itemsize * rows.shape[1]}").ravel()
    return (mesh.material, mesh.visible, mesh.group, mesh.settings, sha256(np.sort(packed)).digest())


def outside_changes(before, after, area):
    old = {m.name: _outside_signature(m, area) for m in before}
    new = {m.name: _outside_signature(m, area) for m in after}
    return [Issue("mesh_outside", name, (), 0.) for name in sorted(old.keys() | new.keys())
            if old.get(name) != new.get(name)]


def terrain_triangles(near, area):
    if area.geom_type == "MultiPolygon":
        return np.concatenate([terrain_triangles(near, p) for p in area.geoms])
    x, y, z, _ = terrain.near_edit_window(near, area)
    n = x.shape[1]
    ids = np.arange(x.size).reshape(x.shape)[:-1, :-1]
    a = np.searchsorted(near["ys"], y[0, 0])
    b = np.searchsorted(near["xs"], x[0, 0])
    holes = near["H"][a:a + ids.shape[0], b:b + ids.shape[1]]
    distance = near["D"][a:a + ids.shape[0] + 1, b:b + ids.shape[1] + 1]
    draw = ~holes & (distance[:-1, :-1] <= terrain.NEAR_BAND) & (distance[1:, :-1] <= terrain.NEAR_BAND)
    draw &= (distance[:-1, 1:] <= terrain.NEAR_BAND) & (distance[1:, 1:] <= terrain.NEAR_BAND)
    if "valley_cells" in near:
        rr, cc = near["valley_cells"].T - np.array([[a], [b]])
        mask = (rr >= 0) & (cc >= 0) & (rr < draw.shape[0]) & (cc < draw.shape[1])
        draw[rr[mask], cc[mask]] = False
    ids = ids[draw]
    v = np.c_[x.ravel(), y.ravel(), z.ravel()]
    f = np.r_[np.c_[ids, ids + 1, ids + n + 1], np.c_[ids, ids + n + 1, ids + n]]
    tri = v[f[np.isfinite(v[:, 2])[f].all(axis=1)]]
    if "valley_faces" in near:
        tri = np.concatenate([tri, near["valley_vertices"][near["valley_faces"]]])
    return tri[_in_area(tri, area)]


def _in_area(tri, area):
    lo, hi = tri[..., :2].min(axis=1), tri[..., :2].max(axis=1)
    a, b, c, d = area.bounds
    sel = (hi[:, 0] >= a) & (hi[:, 1] >= b) & (lo[:, 0] <= c) & (lo[:, 1] <= d)
    idx = np.flatnonzero(sel)
    return idx[shapely.intersects(area, shapely.polygons(tri[idx, :, :2]))]


def _edge_points(edges, step=.125):
    """新旧で同じ水平位置を比べる。足を深くしたときも標本の位置は動かさない。"""
    if not len(edges):
        return np.empty((0, 3))
    n = np.maximum(1, np.ceil(np.linalg.norm(edges[:, 1, :2] - edges[:, 0, :2], axis=1) / step).astype(int))
    index = np.repeat(np.arange(len(n)), n + 1)
    t = (np.arange(len(index)) - np.repeat(np.r_[0, np.cumsum(n[:-1] + 1)], n + 1)) / n[index]
    a, b = edges[index].transpose(1, 0, 2)
    return a + t[:, None] * (b - a)


def _lower_edges(tri):
    """同じ水平座標の上下がある面から、底の両頂点を取る。"""
    v = tri.reshape(-1, 3)
    _, inv = np.unique(np.round(v[:, :2], 7), axis=0, return_inverse=True)
    bottom = np.full(inv.max() + 1, np.inf)
    np.minimum.at(bottom, inv, v[:, 2])
    low = (v[:, 2] <= bottom[inv] + 1e-7).reshape(-1, 3)
    return np.concatenate([tri[low[:, a] & low[:, b], :][:, [a, b]]
                           for a, b in ((0, 1), (1, 2), (2, 0))])


def _component_feet(tri, name):
    """頂点を共有する立体ごとに下端を抽出する。波板、本体の看板などは支柱から区別する。"""
    v, f = np.unique(np.round(tri.reshape(-1, 3), 7), axis=0, return_inverse=True)
    f = f.reshape(-1, 3)
    a, b = f.ravel(), np.roll(f, 1, axis=1).ravel()
    _, lab = connected_components(coo_matrix((np.ones(len(a)), (a, b)), shape=(len(v), len(v))), directed=False)
    lo, hi = np.full(lab.max() + 1, np.inf), np.full(lab.max() + 1, -np.inf)
    np.minimum.at(lo, lab, v[:, 2])
    np.maximum.at(hi, lab, v[:, 2])
    height = hi[lab] - lo[lab]
    if name.startswith("bld_hk_base_"):
        take = height >= .45
    elif any(k in name for k in ("_r70span_", "_r70end_", "_pipe_span_", "_bars", "poles_")):
        xylo, xyhi = np.full((len(lo), 2), np.inf), np.full((len(lo), 2), -np.inf)
        np.minimum.at(xylo, lab, v[:, :2])
        np.maximum.at(xyhi, lab, v[:, :2])
        width = np.linalg.norm(xyhi - xylo, axis=1)[lab]
        take = ((height >= 2.) if "poles_" in name else
                ((height >= .30) & (width <= .65)))
    else:
        take = np.ones(len(v), bool)
    return v[take & (v[:, 2] <= lo[lab] + np.minimum(.08, height * .2))]


class Surfaces:
    """XYの三角形索引。最も上の面、または指定高の直下にある面を返す。"""
    def __init__(self, triangles):
        normal = _normals(triangles)
        self.tri = triangles[abs(normal[:, 2]) > 1e-8]
        self.normal = normal[abs(normal[:, 2]) > 1e-8]
        self.tree = shapely.STRtree(shapely.polygons(self.tri[..., :2]))

    def height(self, points, ceiling=np.inf):
        out = np.full(len(points), -np.inf)
        a, b = self.tree.query(shapely.points(points[:, :2]), predicate="dwithin", distance=1e-7)
        origin = self.tri[b, 0]
        n = self.normal[b]
        z = origin[:, 2] - np.sum(n[:, :2] * (points[a, :2] - origin[:, :2]), axis=1) / n[:, 2]
        ok = z <= np.broadcast_to(ceiling, (len(points),))[a]
        np.maximum.at(out, a[ok], z[ok])
        return out


def water_surfaces(meshes):
    parts = [m.triangles for m in meshes if m.water_kind == "water" and m.visible]
    return Surfaces(np.concatenate(parts) if parts else np.empty((0, 3, 3)))


def river_surface(root):
    """組み立て済みの藤熊川の面。012 はマニフェストに記録される既存の河川番号。"""
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    meshes = []
    for name, entry in manifest["meshes"].items():
        if name.startswith(("water_012_", "water_022_")):
            with np.load(root / entry["file"]) as mesh:
                meshes.append(Mesh(name, mesh["verts"], mesh["faces"], entry["material"], entry["renderable"]))
    return water_surfaces(meshes)


def _normals(tri):
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-18)


def _sample_triangles(tri):
    # 各三角形の頂点・辺・内部を0.25m以下の間隔で取る。大きい面も重心だけで判定しない。
    n = np.maximum(1, np.ceil(np.linalg.norm(np.roll(tri, 1, axis=1) - tri, axis=2).max(axis=1) / .25).astype(int))
    n = 2 ** np.ceil(np.log2(n)).astype(int)
    for k in np.unique(n):
        ids = np.flatnonzero(n == k)
        u, v = np.tril_indices(k + 1)
        weights = np.c_[k - u, u - v, v] / k
        for block in np.array_split(ids, max(1, int(np.ceil(len(ids) * len(weights) / 150000)))):
            yield block, weights @ tri[block]


def _water_checks(mesh, tri, area, ground, old_ground, channels):
    category = mesh.water_kind
    problems = {"water_float": [], "water_buried": []}
    for ids, samples in _sample_triangles(tri):
        xy = samples[..., :2]
        x, y = xy[..., 0], xy[..., 1]
        z = ground(x, y)
        dz = samples[..., 2] - z
        active = shapely.intersects_xy(area, x, y)
        if category == "water" and "_valley_" not in mesh.name:
            # 岸と中心を分ける。設計線の水深は許容する。
            for line, width in channels:
                active &= shapely.distance(shapely.points(x, y), line) > width / 2 + 1e-6
        changed_ground = old_ground(x, y) - z > 1e-6
        for key, error, limit in (("water_float", dz, .10), ("water_buried", -dz, .05)):
            if key == "water_buried" and category == "riverbed":
                continue
            if key == "water_float" and category == "water" and "_valley_" in mesh.name:
                continue  # 水深は許容し、外周の接地は water_join_checks で調べる。
            bad = np.where(active, error, -np.inf)
            worst = np.argmax(bad, axis=1)
            size = bad[np.arange(len(ids)), worst]
            hit = size >= limit
            at = np.arange(len(ids))
            problems[key].append((samples[at, worst][hit], size[hit]))
            if key == "water_float" and category == "riverbed":
                raised = np.where(active & changed_ground, error, -np.inf)
                worst = np.argmax(raised, axis=1)
                size = raised[at, worst]
                hit = size >= limit
                problems.setdefault("bed_over_cut", []).append((samples[at, worst][hit], size[hit]))
    return [finding(key, mesh.parent or mesh.name, np.concatenate([a for a, _ in p]), np.concatenate([b for _, b in p]))
            for key, p in problems.items() if any(len(a) for a, _ in p)]


def _wall_foot(name):
    return any(tag in name for tag in ("_retaining_", "_abutment_", "_rail_support_", "_approach_", "_sill_"))


def _solid_support(name):
    return (name.startswith("1ROAD_") or name.startswith("smallspan_") or _wall_foot(name)
            or "sidewalk" in name or "pad" in name or "deck" in name or name.endswith("_sill")
            or name.startswith("bld_hk_base_"))


def _standing(name):
    tags = ("post", "pole", "fence", "_bars", "mirror", "sign", "light", "_obj", "_lm_", "_poi",
            "bld_", "_r70span_", "_pipe_span_", "bollard", "bench", "shrub", "block")
    if "_polesign_" in name and not name.startswith("1WALL_"):
        return False  # 板と笠は別メッシュの同名支柱に載る。
    if "bld_hk" in name:
        return name.startswith("bld_hk_base_")  # 集約された窓、軒、屋根は家の基礎で点検する。
    return any(tag in name.lower() for tag in tags) and not any(tag in name for tag in ("wire", "shadow", "leaves"))


def mesh_checks(meshes, area, ground, old_ground, *, openings=None, channels=(), ground_tri=None, overlaps=True):
    local = []
    for mesh in meshes:
        if not mesh.visible:
            continue
        tri = mesh.triangles
        ids = _in_area(tri, area)
        if len(ids):
            local.append((mesh, tri[ids]))

    parts = defaultdict(list)
    for mesh, tri in local:
        parts[mesh.parent or mesh.name].append((mesh, tri))
    supports = {name: np.concatenate([tri for _, tri in fragments])
                for name, fragments in parts.items() if _solid_support(name)}
    if supports:
        all_support = list(supports.values())
    else:
        all_support = [np.empty((0, 3, 3))]
    issues = []
    for n, fragments in parts.items():
        mesh = fragments[0][0]
        if mesh.water_kind:
            for fragment, tri in fragments:
                issues.extend(_water_checks(fragment, tri, area, ground, old_ground, channels))
            continue
        tri = fragments[0][1] if len(fragments) == 1 else np.concatenate([t for _, t in fragments])
        if n.startswith("1ROAD_s1_gutter"):
            vertical = tri[abs(_normals(tri)[:, 2]) < 1e-7]
            points = _edge_points(_lower_edges(vertical)) if len(vertical) else np.empty((0, 3))
            if openings is not None:
                points = points[~shapely.intersects_xy(openings.buffer(1e-6), *points[:, :2].T)]
            category, tolerance = "skirt", 1e-6
        elif _wall_foot(n):
            points = _edge_points(_lower_edges(tri))
            if "_retaining_" in n and mesh.parent is None and len(fragments) == 1:
                # 面の傾きで上下のXYが異なる擁壁。偶数頂点は足元の連続列。
                bottom = mesh.verts[::2]
                points = _edge_points(np.stack((bottom[:-1], bottom[1:]), axis=1))
            category, tolerance = "foot", .10
        elif _standing(n):
            points = _component_feet(tri, n)
            category, tolerance = "foot", .10
        else:
            continue
        points = points[shapely.intersects_xy(area, *points[:, :2].T)]
        if not len(points):
            continue
        gz = ground(*points[:, :2].T)
        if n.startswith("1ROAD_s1_gutter"):
            # 横断方向を見る。スカートの外側に接して立つ壁の天端を支えとして扱う。
            cover = [p for name, p in supports.items() if "_rail_support_" in name or "_approach_" in name]
            if cover:
                gz = np.maximum(gz, Surfaces(np.concatenate(cover)).height(points))
        else:
            others = [p for name, p in supports.items() if name != n] if n in supports else all_support
            if others:
                gz = np.maximum(gz, Surfaces(np.concatenate(others)).height(points, points[:, 2] + .19))
        if n.startswith(("mt_rail_281_", "mt_smallspan_16_right_")):
            river_wall = [p for name, p in supports.items() if "_rail_support_" in name]
            if river_wall:
                gz = np.maximum(gz, Surfaces(np.concatenate(river_wall)).height(points))
        dz = points[:, 2] - gz
        bad = (dz > tolerance) | ~np.isfinite(dz)
        if bad.any():
            missing = ~np.isfinite(dz) & bad
            if missing.any():
                issues.append(finding("ground_missing", n, points[missing], np.zeros(missing.sum())))
            measured = bad & ~missing
            if measured.any():
                issues.append(finding(category, n, points[measured], dz[measured]))
    if overlaps and local:
        layers = [(m.name, t) for m, t in local if m.material not in ("ground", "ground_far", "tree")
                  and not m.name.startswith("1GRASS_valley_")]
        if ground_tri is not None:
            layers.append(("terrain", ground_tri))
        issues.extend(overlap_checks(layers))
    return issues


def overlap_checks(layers):
    """同方向の法線を持ち、投影面が面積を持って重なる近接三角形を集計する。"""
    if not layers:
        return []
    tri = np.concatenate([t for _, t in layers])
    layer = np.repeat(np.arange(len(layers)), [len(t) for _, t in layers])
    terrain_layer = np.array([name == "terrain" for name, _ in layers])
    river_layer = np.array([name.startswith("water_") and "_valley_" in name for name, _ in layers])
    median_layer = np.array([name == "track_centre_021_median" for name, _ in layers])
    median_underlay = np.array([name.startswith(("1ROAD_s1_", "mt_leaves_lane_")) for name, _ in layers])
    normal = _normals(tri)
    lo, hi = tri.min(axis=1), tri.max(axis=1)
    bounds = shapely.box(lo[:, 0] - .03, lo[:, 1] - .03, hi[:, 0] + .03, hi[:, 1] + .03)
    tree = shapely.STRtree(bounds)
    issues = {}
    for start in range(0, len(tri), 1024):
        a, b = tree.query(bounds[start:start + 1024])
        a += start
        keep = ((a < b) & (lo[a, 2] <= hi[b, 2] + .03) & (lo[b, 2] <= hi[a, 2] + .03)
                & (np.einsum("ij,ij->i", normal[a], normal[b]) >= .999))
        a, b = a[keep], b[keep]
        keep = ~((layer[a] == layer[b]) & terrain_layer[layer[a]])
        a, b = a[keep], b[keep]
        axis = np.argmax(abs(normal[a]), axis=1)
        for k in range(3):
            p, q = a[axis == k], b[axis == k]
            axes = [j for j in range(3) if j != k]
            intersection = shapely.intersection(shapely.polygons(tri[p][..., axes]),
                                                 shapely.polygons(tri[q][..., axes]))
            hit = shapely.area(intersection) > 1e-7
            p, q, intersection = p[hit], q[hit], intersection[hit]
            uv, ids = shapely.get_coordinates(intersection, return_index=True)
            if not len(uv):
                continue
            def plane(index):
                v = tri[index[ids], 0]
                n = normal[index[ids]]
                return v[:, k] - np.sum((uv - v[:, axes]) * n[:, axes], axis=1) / n[:, k]

            delta = (plane(p) - plane(q)) * abs(normal[p[ids], k])
            lower, upper = np.full(len(p), np.inf), np.full(len(p), -np.inf)
            np.minimum.at(lower, ids, delta)
            np.maximum.at(upper, ids, delta)
            distance = np.maximum(0., np.maximum(lower, -upper))
            overlap = distance <= .03
            if k == 2:
                # 水位と地面はそれぞれ一次平面。交差多角形の全頂点の水深を
                # 調べれば、一辺だけが地面に接する浅瀬と、面的な重複を区別できる。
                wet_p = river_layer[layer[p]] & terrain_layer[layer[q]]
                wet_q = river_layer[layer[q]] & terrain_layer[layer[p]]
                scale = abs(normal[p, k])
                depth_min = np.where(wet_p, lower, -upper) / scale
                depth_max = np.where(wet_p, upper, -lower) / scale
                shallow = (wet_p | wet_q) & (depth_min >= -1e-6) & (depth_max > .001)
                overlap &= ~shallow
                # 白い帯の鼻と路面・落ち葉は3cm以内でも、交差多角形全体で
                # 帯が1mm以上離れて載るなら同じ面を二重に出していない。
                cap_p = median_layer[layer[p]] & median_underlay[layer[q]]
                cap_q = median_layer[layer[q]] & median_underlay[layer[p]]
                overlap &= ~((cap_p & (lower >= .001 * scale)) | (cap_q & (-upper >= .001 * scale)))
            for i in np.flatnonzero(overlap):
                m, n = sorted((layers[layer[p[i]]][0], layers[layer[q[i]]][0]))
                xy = tuple(map(float, tri[p[i], :, :2].mean(0)))
                old = issues.get((m, n))
                count = old.count + 1 if old else 1
                best = min(old.amount, float(distance[i])) if old else float(distance[i])
                issues[m, n] = Issue("overlap", m, xy if best == distance[i] else old.xy, best, n, count)
    return list(issues.values())


def bridge_checks(meshes, sec, spans, ground):
    """床版の裏を鉛直に、桁と橋台の継ぎ目を横断面の両側から調べる。"""
    from .crosssection import GUTTER_WIDTH
    from .small_bridges import SUPPORT_SPANS

    by_name = defaultdict(list)
    for m in meshes:
        if m.visible:
            by_name[m.parent or m.name].append(m)
    issues = []
    for k in SUPPORT_SPANS:
        if k not in spans:
            continue
        i, j = spans[k]
        name = f"smallspan_{k:02d}_deck"
        if name not in by_name:
            issues.append(Issue("deck_missing", name, (float(sec.x[i]), float(sec.y[i])), 0.))
            continue
        tri = np.concatenate([m.triangles for m in by_name[name]])
        normals = _normals(tri)
        underside = Surfaces(tri[normals[:, 2] < -.5])
        sl = slice(i, j + 1)
        normal = np.c_[-np.sin(sec.heading[sl]), np.cos(sec.heading[sl])]
        origin = np.c_[sec.x[sl], sec.y[sl]]
        left = origin + normal * (sec.hw_left[sl] + GUTTER_WIDTH)[:, None]
        right = origin - normal * (sec.hw_right[sl] + GUTTER_WIDTH)[:, None]
        area = shapely.Polygon(np.r_[left, right[::-1]])
        x0, y0, x1, y1 = area.bounds
        x, y = np.meshgrid(np.arange(x0, x1, .25), np.arange(y0, y1, .25))
        inside = shapely.contains_xy(area, x, y)
        points = np.c_[x[inside], y[inside], np.zeros(inside.sum())]
        bad = ~np.isfinite(underside.height(points))
        if bad.any():
            issues.append(finding("deck_missing", name, points[bad], np.zeros(bad.sum())))
        # 反対側の桁が点検を通してしまわないよう、面を左右に分けて横投影する。
        side_tri = tri[abs(normals[:, 2]) < .5]
        mid = (left + right) / 2
        axis = np.diff(mid, axis=0).mean(0)
        axis /= np.linalg.norm(axis)
        transverse = np.array([-axis[1], axis[0]])
        for label, side_xy, sign in (("L", left, 1), ("R", right, -1)):
            edge_z = sec.edge_z(sign)[sl]
            route = (side_xy - mid[0]) @ axis
            p = np.stack((side_xy[:-1], side_xy[1:]), axis=1)
            points = _edge_points(np.pad(p, ((0, 0), (0, 0), (0, 1))))
            along = (points[:, :2] - mid[0]) @ axis
            order = np.argsort(route)
            pavement = np.interp(along, route[order], edge_z[order])
            near_side = sign * ((side_tri[:, :, :2].mean(1) - mid[0]) @ transverse) > 0
            flat = np.stack(((side_tri[near_side, :, :2] - mid[0]) @ axis, side_tri[near_side, :, 2]), axis=2)
            wall = shapely.polygons(flat)
            cover = shapely.union_all(wall[shapely.area(wall) > 1e-9]).buffer(1e-6)
            for below in (.24, .49, .74):
                hit = shapely.intersects_xy(cover, along, pavement - below)
                if (~hit).any():
                    issues.append(finding("deck_side", name, points[~hit], np.full(np.count_nonzero(~hit), below), label))
        for tag, row, ends in (("start", i, (right[0], left[0])), ("end", j, (right[-1], left[-1]))):
            wall_name = f"smallspan_{k:02d}_abutment_{tag}"
            points = _edge_points(np.array([[np.r_[ends[0], 0], np.r_[ends[1], 0]]]))
            if wall_name not in by_name:
                issues.append(finding("abutment_missing", wall_name, points, points[:, 2]))
                continue
            wall = np.concatenate([m.triangles for m in by_name[wall_name]])
            base = _edge_points(_lower_edges(wall))
            margin = ground(*base[:, :2].T) - base[:, 2]
            bad = margin < .5
            if bad.any():
                issues.append(finding("abutment_burial", wall_name, base[bad], .5 - margin[bad]))
    return issues


def wall_checks(meshes, sec, ground):
    """壁面の天端とそのすぐ山側、道路縁の高さを照合する。"""
    issues = []
    for mesh in meshes:
        if mesh.name not in {"1WALL_s1_retaining_279", "1WALL_s1_retaining_280", "1WALL_s1_retaining_395"}:
            continue
        v = mesh.verts
        top = v[1::2]
        frame = terrain.road_frame(sec, *top[:, :2].T)
        rows = frame["idx"]
        normal = np.c_[-np.sin(sec.heading[rows]), np.cos(sec.heading[rows])]
        behind = top[:, :2] + .30 * normal
        delta = top[:, 2] - ground(*behind.T)
        for name, difference, limit, xy in (
            ("wall_crest_above", delta, .10, behind),
            ("wall_crest_below", -delta, .10, behind),
            ("wall_below_road", sec.edge_z(1)[rows] - top[:, 2], 1e-6, top),
        ):
            bad = difference > limit
            if bad.any():
                issues.append(finding(name, mesh.name, xy[bad], difference[bad]))
    return issues


def water_join_checks(meshes, levels, ground):
    """水際と旧・新水面が接続する端面の高さを実際の三角形から読む。"""
    issues = []
    from .water import A_CREEK, shore_samples

    for osm_id, reach in levels.items():
        prefix = {A_CREEK: "water_022_", 90334744: "water_012_"}.get(osm_id)
        if prefix is None:
            continue
        parts = [m for m in meshes if m.water_kind == "water" and m.name.startswith(prefix)]
        fresh = [m.triangles for m in parts if "_valley_" in m.name]
        if not fresh:
            continue
        fresh = np.concatenate(fresh)
        old_tri = [m.triangles for m in parts if "_valley_" not in m.name]
        old_tri = np.concatenate(old_tri) if old_tri else np.empty((0, 3, 3))
        old_water = Surfaces(old_tri) if len(old_tri) else None
        points = shore_samples(fresh, old_tri)
        delta = abs(points[:, 2] - ground(*points[:, :2].T))
        bad = (delta > .05) | ~np.isfinite(delta)
        if bad.any():
            issues.append(finding("bank_gap", reach.valley.id, points[bad], np.nan_to_num(delta[bad], nan=1e6)))
        if old_water is not None:
            points = shore_samples(fresh)
            new = points[:, 2]
            old = old_water.height(points)
            ok = np.isfinite(new) & np.isfinite(old)
            delta = abs(new[ok] - old[ok])
            bad = delta > .01
            if bad.any():
                issues.append(finding("water_join", reach.valley.id, points[ok][bad], delta[bad]))
    return issues


def tree_checks(trees, area, rows, ground, old_ground, meshes):
    water = [m.triangles for m in meshes if m.water_kind == "water" and m.visible]
    xy = np.c_[trees.x, trees.y]
    ids = np.flatnonzero(shapely.intersects_xy(area, *xy.T))
    points = np.c_[xy[ids], trees.z[ids]]
    z = ground(*points[:, :2].T)
    original = old_ground(*points[:, :2].T)
    lower = original - z > 1e-6
    beds = [m.triangles for m in meshes if m.name.startswith("1GRASS_valley_") and m.name.endswith("_bed")]
    if beds:
        bottom = np.isfinite(Surfaces(np.concatenate(beds)).height(points))
    else:
        bottom = np.zeros(len(ids), bool)
        for valley in rows:
            _, s, d, arc = valley.project(*points[:, :2].T)
            bottom |= lower & (abs(d) <= valley.width_m / 2) & (s > 0) & (s < arc[-1])
    wet = Surfaces(np.concatenate(water)).height(points) > z if water else np.zeros(len(ids), bool)
    delta = abs(points[:, 2] - z)
    issues = []
    for kind, bad, amount in (("tree_root", delta > .02, delta),
                              ("tree_bottom", bottom, abs(original - z)), ("tree_water", wet, delta)):
        issues += [Issue(kind, "trees.npy", tuple(map(float, p[:2])), float(a))
                   for p, a in zip(points[bad], amount[bad])]
    return issues


def card_checks(meshes, area, rows, ground, old_ground):
    from .valleys import tree_bottom

    beds = [m.triangles for m in meshes if m.name.startswith("1GRASS_valley_") and m.name.endswith("_bed")]
    bed = Surfaces(np.concatenate(beds)) if beds else None
    river = water_surfaces(m for m in meshes if m.name.startswith(("water_012_", "water_022_")))
    issues = []
    for m in meshes:
        if not m.name.startswith("TREES_s1_") or not m.visible:
            continue
        points = m.verts.reshape(-1, 8, 3)[:, :2].mean(axis=1)
        points = points[shapely.intersects_xy(area, *points[:, :2].T)]
        if not len(points):
            continue
        z = ground(*points[:, :2].T)
        drop = old_ground(*points[:, :2].T) - z
        root = abs(points[:, 2] - z)
        bottom = np.isfinite(bed.height(points)) if bed is not None else (drop > 1e-6) & tree_bottom(rows, *points[:, :2].T)
        wet = river.height(points) - z
        for category, bad, value in (("card_root", root > .02, root), ("card_bottom", bottom, abs(drop)),
                                     ("card_water", wet > 1e-6, wet)):
            if bad.any():
                issues.append(finding(category, m.name, points[bad], value[bad]))
    return issues
