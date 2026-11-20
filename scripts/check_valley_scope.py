"""小橋3か所の許可範囲を、組み立て前後の三角形と設定から検査する。"""
import argparse
from collections import Counter, defaultdict
import filecmp
import json
from pathlib import Path
import re

import numpy as np
import shapely
from scipy.spatial import cKDTree

from scripts.align_s5_8960_trees import _line, _patch_tile
from scripts.check_valley_fit import npz
from yabitsu import buildings, config, kn5, side_roads, small_bridges, start_lot, track, tunnel, valleys, valley_edits, valley_fit
from yabitsu.crosssection import GUTTER_WIDTH, Section, parse_width_rows, width_taper_areas
from yabitsu.valleys import changed_ground


HOUSE = re.compile(r"(?:1WALL_)?bld_hk.*_cm3_1_hk_00")
B_PARTS = re.compile(r"(?:GUARDRAIL_s1_|1WALL_s1_rail_)(?:278|281|394)|"
                     r"1WALL_s1_retaining_(?:279|280|281|395)|mt_rail_(?:278|281|394)_.+")
BRIDGE = re.compile(r"(?:1WALL_track_smallspan_|mt_smallspan_|smallspan_|mt_r70_bridge_)(02|16|25)_.+")
BANK = re.compile(r"(?:water|riverbed)_(?:012|022)_(?:\d+|valley_\d+)(?:_edit_[A-Za-z0-9_.-]+)?")
NEAR = ("1GRASS_s1_terrain_", "mtsplit_s1_terrain_", "1GRASS_valley_")
MESH_PAYLOAD = ("file", "material", "verts", "faces", "valley_edit_source", "valley_edit_result", "valley_edit_parent")
SKIRT = "1ROAD_s1_gutter"
TAPER_PARTS = re.compile(r"1ROAD_s1_.+|LINES_s1_edge.+|(?:LINES_)?track_centre_021(?:_median)?|"
                         r"mt_leaves_.+|(?:GUARDRAIL_s1_|1WALL_s1_rail_)(?:366|368)|mt_rail_(?:366|368)_.+|"
                         r"1WALL_s1_retaining_(?:367|370)|1GRASS_track_shelf_(?:367|370)|"
                         r"(?:1WALL_track_smallspan_|mt_smallspan_|smallspan_)(?:25)_.+")


def rows_key(rows):
    width = int(np.prod(rows.shape[1:]))
    return np.ascontiguousarray(rows).reshape(len(rows), width).view(f"V{rows.dtype.itemsize * width}").ravel()


def triangle_keys(tri):
    # 頂点の巡回開始位置は比較しない。反転、重複、各頂点の属性は照合する。
    keys = np.stack([rows_key(np.roll(tri, n, axis=1)) for n in range(3)]).T
    return np.sort(keys, axis=1)[:, 0]


def triangle_changes(a, b):
    ka, kb = triangle_keys(a), triangle_keys(b)
    both = np.concatenate((ka, kb))
    _, inverse, counts = np.unique(both, return_inverse=True, return_counts=True)
    allowed = counts - np.bincount(inverse[:len(a)], minlength=len(counts))
    seen = np.zeros(len(counts), int)
    misses = []
    for i, key in enumerate(inverse[:len(a)]):
        seen[key] += 1
        if seen[key] > allowed[key]:
            misses.append(i)
    return np.asarray(misses, int)


def mesh_data(root, entry):
    with np.load(root / entry["file"]) as data:
        v, f, uv = data["verts"], data["faces"], data["uvs"]
        tris = kn5._triangles(f)
        n = kn5.vertex_normals(data, v @ kn5.C.T, tris)
    return np.concatenate((v[tris], uv[tris], n.astype("<f4")[tris]), axis=2)


def overwritten_images(before, after):
    """共有画像の上書きと、新たに同じstemを使う別ファイルを検出する。"""
    def listing(root):
        return {p.relative_to(root) for folder in ("textures", "extension") for p in (root / folder).rglob("*")
                if p.is_file() and p.suffix.lower() in {".dds", ".png", ".jpg", ".jpeg"}}

    old, new = listing(before), listing(after)
    stems = Counter(p.stem.casefold() for p in new)
    for path in sorted(new):
        if path in old:
            dedicated = (path.parent == Path("textures") and path.suffix == ".png"
                         and path.stem.startswith("valley_edit_") and stems[path.stem.casefold()] == 1)
            if not dedicated and not filecmp.cmp(before / path, after / path, shallow=False):
                yield path.as_posix()
        elif stems[path.stem.casefold()] > 1:
            yield path.as_posix()


def json_file(root, name, default):
    p = root / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else default


def reorderings(old, new):
    ranks = {n: i for i, n in enumerate(old)}
    common = [n for n in new if n in ranks]
    if all(ranks[a] < ranks[b] for a, b in zip(common, common[1:])):
        return []
    return [(a, b) for i, a in enumerate(common) for b in common[i + 1:] if ranks[b] < ranks[a]]


class Scope:
    def __init__(self, sec, bounds, before, after, *, edits=(), sites_root=None, round2=False):
        self.sec, self.bounds = sec, bounds
        self.round2 = round2
        self.road = cKDTree(np.c_[sec.x, sec.y])
        width_rows = parse_width_rows(track.load_section_rows("widths", bounds))
        self.tapers = list(width_taper_areas(sec, width_rows))
        self.dug = changed_ground(before, after)
        self.surface_keys = after.keys()
        self.band = self.dug.polygon.buffer(3.)
        self.terrain_keys = [triangle_keys(m) for m in self.dug.triangles]
        if "valley_faces" in before:
            exact_before = before["valley_vertices"][before["valley_faces"]]
            self.terrain_keys[0] = np.union1d(self.terrain_keys[0], triangle_keys(exact_before))
        terrain, self.edited_parts = [[], []], {}
        for edit in edits:
            directory = sites_root / edit.site
            meta = json_file(directory, "site.json", None)
            if edit.metadata["terrain_changed"]:
                for i, path in enumerate((directory / "source.npz", edit.path / "terrain.npz")):
                    with np.load(path) as data:
                        prefix = "terrain_" if i == 0 else ""
                        v, f = data[prefix + "vertices"], data[prefix + "faces"]
                        keep = data[prefix + "coarse_remainder"][:, 0] < 0
                        terrain[i].append(triangle_keys(v[f[keep]]))
            for part in edit.metadata["changed_parts"]:
                self.edited_parts[part] = (np.asarray(meta["origin"][:2]), meta["radius"])
        self.edited_terrain_keys = [np.concatenate(keys) if keys else np.empty(0, dtype="V72")
                                    for keys in terrain]
        self.spans = small_bridges.spans(sec, bounds, small_bridges.load())
        rows = self.rows = valleys.load()
        self.water = None
        self.sites = {r.id[0]: valleys.site_area(sec, bounds, r.section, r.center_m) for r in rows}
        self.river = next((r for r in rows if r.water and r.water["osm_id"] == 90334744), None)
        self.river_area = self.dug.polygon.intersection(self.sites["B"]) if self.river else shapely.Polygon()
        goods = buildings.load(config.DATA_MANUAL / "track_buildings.json", clearance_only=True)
        house = next((b.footprint for b in goods if b.id == "vm866_484"), shapely.Polygon())
        # 既存の家部品は輪郭から矩形に整形し、軒と付属部品が張り出す。
        self.house = house.convex_hull.buffer(2.)
        self.gutters = {}
        self.skirt_floor = {}
        for sign, widths in ((1, sec.hw_left), (-1, sec.hw_right)):
            lat = sign * (widths + GUTTER_WIDTH)
            xy = np.c_[sec.x - lat * np.sin(sec.heading), sec.y + lat * np.cos(sec.heading)]
            self.gutters[sign] = cKDTree(xy)
            if "valley_cells" in after:
                from yabitsu.meshgen import SKIRT_DEPTH
                foot = np.c_[xy,sec.edge_z(sign)-SKIRT_DEPTH]
                self.skirt_floor[sign] = valleys.fit_foot_chain(foot,self.dug,0.)
        self.issues = []
        self.allowed = defaultdict(set)
        self.counts = Counter()

    def local(self, xy):
        sec = self.sec
        k = self.road.query(xy)[1]
        dx, dy = xy[..., 0] - sec.x[k], xy[..., 1] - sec.y[k]
        h = sec.heading[k]
        return sec.s[k] + dx * np.cos(h) + dy * np.sin(h), -dx * np.sin(h) + dy * np.cos(h)

    def location(self, xy):
        s, lateral = self.local(np.asarray(xy))
        for section, (a, b) in self.bounds.items():
            if a <= s <= b:
                return dict(section=section, m=round(float(s - a), 4), lateral=round(float(lateral), 4))
        return dict(m=float(s), lateral=float(lateral))

    def issue(self, source, name, what, xy=None, **detail):
        loc = self.location(xy) if xy is not None else {}
        self.issues.append(dict(source=source, mesh=name, change=what, **loc, **detail))
        self.counts[what] += 1

    def interval(self, s, key, a, b):
        return (s >= self.bounds[key][0] + a) & (s <= self.bounds[key][0] + b)

    def bridge_rows(self, s, *, margin=0.):
        return np.logical_or.reduce([(s >= self.sec.s[a] - margin) & (s <= self.sec.s[b] + margin)
                                     for k, (a, b) in self.spans.items() if k in small_bridges.SUPPORT_SPANS])

    def settings_allowed(self, name):
        name = name.removesuffix("_back")
        return bool(B_PARTS.fullmatch(name) or BRIDGE.fullmatch(name)
                    or name == "1WALL_s1_retaining_367")

    def taper_permits(self, name, tri, what):
        out = np.zeros(len(tri), bool)
        if not len(tri) or what not in ("position", "added", "removed", "normal", "uv"):
            return out
        earth = name.startswith(NEAR)
        if not earth and not TAPER_PARTS.fullmatch(name):
            return out
        xy = tri[..., :2]
        for a, b, area in self.tapers:
            if earth:
                # 地形は重心ではなく三角形全体が道路縁から3 m以内に収まること。
                out |= shapely.covers(area.buffer(1e-8), shapely.polygons(xy))
            else:
                s, _ = self.local(xy)
                out |= ((a <= s) & (s <= b)).all(axis=1)
        return out

    def permits(self, name, tri, what, *, removed=False):
        xy = tri[..., :2]
        attribute = what in ("uv", "normal")
        out = np.zeros(len(tri), bool)
        parent = name.split("_edit_", 1)[0]
        if parent in self.edited_parts and what in ("uv", "normal", "material", "position", "added", "removed"):
            origin, radius = self.edited_parts[parent]
            out |= (np.linalg.norm(xy - origin, axis=2) <= radius + 1e-7).all(axis=1)
        if attribute and name.startswith(NEAR):
            out |= np.isin(triangle_keys(tri[..., :3]), self.edited_terrain_keys[0 if removed else 1])
        # Round 2 already includes the procedural taper, bridge and channel.
        if attribute and self.round2:
            return out
        center = xy.mean(axis=1)
        s, d = self.local(center)
        out |= self.taper_permits(name, tri, what)
        config_only = what in ("material", "renderable", "settings")
        if what == "kn5":
            return out
        if B_PARTS.fullmatch(name):
            # 行端の曲面・張り出した支柱を含む。中心線の入力範囲の外は比較する。
            out |= self.interval(s, "s4", 6236., 6423.)
            if name.startswith(("mt_rail_278_", "GUARDRAIL_s1_278", "1WALL_s1_rail_278")):
                # 切断面をまたぐ旧波板と、同じ切断面に接する残片。
                edge = self.bounds["s4"][0] + 6237.
                ns = self.local(xy)[0]
                out |= (ns.min(axis=1) <= edge) & (ns.max(axis=1) >= edge - .04)
        bridge = BRIDGE.fullmatch(name)
        if bridge and int(bridge[1]) in self.spans:
            a, b = self.spans[int(bridge[1])]
            margin = 5. if any(t in name for t in ("_approach_", "_abutment_", "_rail_support_")) else 2.
            if name == "smallspan_25_wall_367_gravel":
                out |= self.interval(s,"s5",8960,8981) & (d>0) & (d<8)
            if name.startswith("smallspan_16_old_road_"):
                out |= self.interval(s, "s4", 6338., 6369.) & (d >= 3.) & (d <= 15.)
            if "smallspan_16_rail_support_" in name:
                out |= self.interval(s, "s4", 6236., 6423.)
            out |= (s >= self.sec.s[a] - margin) & (s <= self.sec.s[b] + margin)
        if HOUSE.fullmatch(name) and removed:
            # このセルには複数の家が集約される。対象家が建つ側の面だけを許す。
            out |= self.interval(s, "s4", 1709., 1722.) & (d > 0.) & shapely.intersects_xy(self.house, *center.T)
        if name == "1WALL_s1_retaining_367" and config_only:
            out |= self.interval(s, "s5", 8959., 8981.)
        if config_only:
            return out
        if name.startswith(NEAR):
            if what in ("position", "added", "removed"):
                out |= np.isin(triangle_keys(tri[..., :3]), self.terrain_keys[0 if removed else 1])
        elif name.startswith(SKIRT):
            return out  # 頂点の対応、UV、法線を skirt_changes で検査する。
        elif name.startswith("TREES_s1_"):
            pass  # These meshes are compared tree by tree.
        elif BANK.fullmatch(name):
            if what in ("position", "removed", "added"):
                area = self.water_area(name)
                shapes = shapely.polygons(xy)
                if removed:
                    # 境界越しの旧三角形は残片との照合で確認する。
                    out |= shapely.area(shapely.intersection(area, shapes)) > 0
                else:
                    out |= shapely.covers(area.buffer(1e-7), shapes)
        return out

    def water_area(self, name):
        return self.dug.polygon.intersection(self.sites["A"]) if "_022_" in name else self.river_area

    def triangles(self, source, group, name, tri, what, *, removed=False):
        if not len(tri):
            return
        ok = self.permits(name, tri, what, removed=removed)
        for t in tri[~ok]:
            self.issue(source, name, what, t[:, :2].mean(0), kn5=group)
        if ok.any():
            self.allowed[group].add(name)

    def manifests(self, source, before, after):
        for path in overwritten_images(before, after):
            self.issue(source, path, "image_overwritten")
        a, b = [json_file(root, "manifest.json", None) for root in (before, after)]
        for key in ("empties",):
            if a[key] != b[key]:
                self.issue(source, key, "manifest")
        if list(a["groups"]) != list(b["groups"]):
            self.issue(source, "groups", "kn5")
        for group in a["groups"].keys() | b["groups"].keys():
            ga, gb = a["groups"].get(group, {}), b["groups"].get(group, {})
            for n, other in reorderings(ga.get("meshes", []), gb.get("meshes", [])):
                e = b["meshes"][n]
                tri = mesh_data(after, e)
                self.issue(source, n, "order", tri[..., :2].mean(axis=(0, 1)), kn5=group, other=other)
            if {k: v for k, v in ga.items() if k != "meshes"} != {k: v for k, v in gb.items() if k != "meshes"}:
                self.issue(source, group, "empty_order")
        groups = [{n: tuple(g for g, e in m["groups"].items() if n in group_names[g])
                   for n in m["meshes"]}
                  for m in (a, b) for group_names in [{g: set(e["meshes"]) for g, e in m["groups"].items()}]]
        for name in a["meshes"].keys() | b["meshes"].keys():
            ea, eb = a["meshes"].get(name), b["meshes"].get(name)
            g0, g1 = groups[0].get(name, ()), groups[1].get(name, ())
            group = ",".join(g1 or g0)
            if ea is None or eb is None:
                root, entry = (after, eb) if ea is None else (before, ea)
                self.triangles(source, group, name, mesh_data(root, entry),
                               "added" if ea is None else "removed", removed=eb is None)
                continue
            props_a, props_b = a["materials"][ea["material"]], b["materials"][eb["material"]]
            af, bf = before / ea["file"], after / eb["file"]
            categories = []
            if ea["material"] != eb["material"] or props_a != props_b:
                categories.append("material")
            if {k: v for k, v in ea.items() if k not in MESH_PAYLOAD} != {
                    k: v for k, v in eb.items() if k not in MESH_PAYLOAD}:
                categories.append("renderable")
            if g0 != g1:
                categories.append("kn5")
            same_bytes = af.read_bytes() == bf.read_bytes()
            if same_bytes and not categories:
                continue
            ta, tb = mesh_data(before, ea), mesh_data(after, eb)
            for what in categories:
                self.triangles(source, group, name, ta, what)
            if not same_bytes:
                if name.startswith("TREES_s1_"):
                    self.cards(source, group, name, before, ea, after, eb)
                elif name.startswith(SKIRT) and self.skirt_changes(source, group, name, before, ea, after, eb, ta, tb):
                    continue
                elif (name.startswith("1WALL_s1_retaining_") and not B_PARTS.fullmatch(name)
                      and self.wall_base_changes(name,before,ea,after,eb,ta,tb)):
                    self.allowed[group].add(name)
                elif (valley_fit._standing(name)
                      and not re.match(r"(?:1WALL_)?s[145]_obj_", name)
                      and self.support_changes(source, group, name, ta, tb)):
                    continue
                else:
                    self.geometry(source, group, name, ta, tb)
        if source == "dress":
            self.settings(before, after)

    def geometry(self, source, group, name, ta, tb):
        for t, other, removed in ((ta, tb, True), (tb, ta, False)):
            ids = triangle_changes(t, other)
            if not len(ids):
                continue
            tri = t[ids]
            # 再分割を伴う部品でも、同じ位置のUVと法線の差を見逃さない。
            with_pos = np.isin(triangle_keys(tri[..., :3]), triangle_keys(other[..., :3]))
            with_uv = np.isin(triangle_keys(tri[..., :5]), triangle_keys(other[..., :5]))
            tag = "position" if len(t) == len(other) else "removed" if removed else "added"
            kinds = np.select([with_uv, with_pos], ["normal", "uv"], default=tag)
            if not removed and name.startswith("1GRASS_s1_terrain_") and "valley_coarse_remainder" in self.surface_keys:
                polys = shapely.polygons(ta[..., :2])
                candidates = shapely.intersects(self.dug.polygon, polys)
                copied = self.cut_remainders(tri, ta[candidates], polys[candidates], normals=True)
                if copied.any():
                    self.allowed[group].add(name)
                    tri, kinds = tri[~copied], kinds[~copied]
            if not removed and BANK.fullmatch(name) and "_valley_" not in name:
                area = self.water_area(name)
                old_polys = shapely.polygons(ta[..., :2])
                touched = shapely.area(shapely.intersection(area, old_polys)) > 1e-9
                olds = ta[touched]
                polys = old_polys[touched]
                if len(olds):
                    outside = self.cut_remainders(tri, olds, polys)
                    if outside.any():
                        self.allowed[group].add(name)
                        tri, kinds = tri[~outside], kinds[~outside]
            for kind in np.unique(kinds):
                self.triangles(source, group, name, tri[kinds == kind], str(kind), removed=removed)

    @staticmethod
    def cut_remainders(tri, originals, polygons, *, normals=False):
        """境界をまたぐ旧面から切り出した、元の平面とUV上の残片。"""
        tree = shapely.STRtree(shapely.buffer(polygons, 1e-7))
        indices, source = tree.query(shapely.polygons(tri[..., :2]), predicate="covered_by")
        allowed = np.zeros(len(tri), bool)
        if len(indices):
            base = originals[source]
            a = base[:, 0]
            transform = np.stack((base[:, 1, :2] - a[:, :2], base[:, 2, :2] - a[:, :2]), axis=2)
            weights = (tri[indices, :, :2] - a[:, None, :2]) @ np.linalg.inv(transform).transpose(0, 2, 1)
            end = 8 if normals else 5
            expect = a[:, None, 2:end] + weights @ (base[:, 1:, 2:end] - a[:, None, 2:end])
            tolerance = [1e-7, .0002, .0002] + ([2e-6] * 3 if normals else [])
            ok = (abs(tri[indices, :, 2:end] - expect) <= tolerance).all(axis=(1, 2))
            allowed[indices[ok]] = True
        return allowed

    def skirt_changes(self, source, group, name, before, ea, after, eb, ta, tb):
        with (np.load(before / ea["file"]) as a, np.load(after / eb["file"]) as b):
            if not np.array_equal(a["faces"], b["faces"]) or a["verts"].shape != b["verts"].shape:
                return False
            v, w, faces = a["verts"], b["verts"], kn5._triangles(a["faces"])
        side = 1 if name[len(SKIRT)] == "L" else -1
        outer = 3 if side > 0 else 0
        changed = v[:, 2] != w[:, 2]
        indices = np.flatnonzero(changed)
        distance, rows = self.gutters[side].query(v[indices, :2])
        good = np.zeros(len(v), bool)
        good[indices] = ((indices % 4 == outer) & (distance < 1e-6)
                         & self.bridge_rows(self.sec.s[rows]) & (w[indices, 2] > v[indices, 2]))
        if side in self.skirt_floor:
            good[indices] |= ((indices % 4 == outer) & (distance < 1e-6)
                              & (w[indices,2] < v[indices,2]) & ~self.bridge_rows(self.sec.s[rows])
                              & (abs(self.skirt_floor[side][rows]-w[indices,2]) < 1e-8))
        position = (ta[..., :3] != tb[..., :3]).any(axis=(1, 2))
        bad_pos = position & ((changed & ~good)[faces].any(axis=1)
                              | (ta[..., :2] != tb[..., :2]).any(axis=(1, 2)))
        taper = self.taper_permits(name, ta, "position") & self.taper_permits(name, tb, "position")
        for what, bad in [("position", bad_pos),
                          ("uv", (ta[..., 3:5] != tb[..., 3:5]).any(axis=(1, 2))),
                          ("normal", (ta[..., 5:] != tb[..., 5:]).any(axis=(1, 2)))]:
            for t in tb[bad & ~taper]:
                self.issue(source, name, what, t[:, :2].mean(axis=0), kn5=group)
        if (position & (~bad_pos | taper)).any():
            self.allowed[group].add(name)
        return True

    def wall_base_changes(self,name,before,ea,after,eb,ta,tb):
        with np.load(before/ea["file"]) as a, np.load(after/eb["file"]) as b:
            v, w = a["verts"], b["verts"]
            if not np.array_equal(a["faces"],b["faces"]) or not np.array_equal(v[:,:2],w[:,:2]):
                return False
            expected = v.copy()
            expected[::2,2] = valleys.fit_foot_chain(v[::2],self.dug,.10)
            return np.array_equal(expected,w) and np.array_equal(ta[...,3:],tb[...,3:])

    def support_changes(self, source, group, name, old, new):
        if old.shape != new.shape or not np.array_equal(old[..., :2], new[..., :2]):
            return False
        moved = old[..., 2] != new[..., 2]
        if not moved.any() or not np.array_equal(old[..., 3:5], new[..., 3:5]):
            return False
        feet = old[..., :3][moved]
        expected = feet.copy()
        valleys.lower_exposed_feet(expected, np.ones(len(feet), bool), self.dug)
        if (not np.all(expected[:, 2] < feet[:, 2])
                or not np.allclose(expected[:, 2], new[..., 2][moved], atol=1e-9, rtol=0)):
            return False
        changed = (old != new).any(axis=(1, 2))
        # 法線差は、下端の面と頂点を共有する面のうち掘削から水平3m以内に限る。
        affected = old[moved.any(axis=1), :, :3].reshape(-1, 3)
        connected = np.isin(rows_key(old[..., :3].reshape(-1, 3)), rows_key(affected)).reshape(-1, 3).any(axis=1)
        xy = old[..., :2].mean(axis=1)
        authorized = moved.any(axis=1) | (connected & shapely.intersects_xy(self.band, *xy.T))
        for point in xy[changed & ~authorized]:
            self.issue(source, name, "normal", point, kn5=group)
        self.allowed[group].add(name)
        return True

    def cards(self, source, group, name, before, ea, after, eb):
        # 本番の cards() は1本を連続する8頂点、2枚の板として書く。
        def read(root, entry):
            with np.load(root / entry["file"]) as data:
                v, uv = data["verts"], data["uvs"]
                if len(v) % 8 or not np.array_equal(data["faces"], np.arange(len(v)).reshape(-1, 4)):
                    raise ValueError(f"{name}: 木の板の形式が異なります")
            return v.reshape(-1, 8, 3), uv.reshape(-1, 8, 2)

        old, uv0 = read(before, ea)
        new, uv1 = read(after, eb)
        xy0, xy1 = (arr[:, :2, :2].mean(axis=1) for arr in (old, new))
        lookup = {tuple(xy): i for i, xy in enumerate(xy1)}
        allowed = self.dug.covers(xy0)
        height = self.dug.height(xy0)
        for i, xy in enumerate(xy0):
            j = lookup.pop(tuple(xy), None)
            if j is None:
                if allowed[i] and self.removable_tree(xy):
                    self.allowed[group].add(name)
                else:
                    self.issue(source, name, "card_removed", xy, kn5=group)
                continue
            if np.array_equal(old[i], new[j]) and np.array_equal(uv0[i], uv1[j]):
                continue
            if (allowed[i] and abs(new[j, 0, 2] - height[i]) < 1e-6
                    and np.array_equal(old[i, :, :2], new[j, :, :2])
                    and np.array_equal(uv0[i], uv1[j])
                    and np.allclose(new[j, :, 2] - old[i, :, 2], new[j, 0, 2] - old[i, 0, 2], rtol=0, atol=1e-9)):
                self.allowed[group].add(name)
            else:
                self.issue(source, name, "card_geometry", xy, kn5=group)
        for xy in lookup:
            self.issue(source, name, "card_added", xy, kn5=group)

    def removable_tree(self, xy, *, forest=False):
        if self.dug.bed is not None:
            bed, wet = self.dug.tree_exclusions(xy, self.water[1])
            return bool(bed[0] or wet[0])
        x, y = np.atleast_2d(xy).T
        return bool(valleys.tree_bottom(self.rows, x, y, forest=forest)[0] or (
            self.water is not None and valleys.newly_wet(self.dug, *self.water, xy)[0]))

    def settings(self, before, after):
        hidden = [set(json_file(root, "hidden.json", [])) for root in (before, after)]
        for name in hidden[0] ^ hidden[1]:
            if not self.settings_allowed(name):
                self.issue("hidden.json", name, "settings")
        a, b = [json_file(root, "retex.json", {}) for root in (before, after)]
        for name in a.keys() | b.keys():
            if a.get(name) != b.get(name) and not self.settings_allowed(name):
                self.issue("retex.json", name, "settings")
        p = [root / "extension" / "dress_mountain.ini" for root in (before, after)]
        for line in ini_changes(*(x.read_text(encoding="utf-8") if x.exists() else "" for x in p),
                                self.settings_allowed):
            self.issue("dress_mountain.ini", line, "settings")


def ini_changes(before, after, allows):
    def sections(content):
        parts = []
        for line in content.splitlines():
            if line.startswith("[") or not parts:
                parts.append([])
            parts[-1].append(line)
        return parts

    def protected(content):
        keep = []
        for lines in sections(content):
            filtered = []
            targeted = False
            for line in lines:
                key, _, value = line.partition("=")
                if key.strip() in ("MESHES", "MESH"):
                    names = value.split(",")
                    protected_names = [n for n in names if not allows(n.strip())]
                    targeted |= not protected_names
                    filtered.append(key + "=" + ",".join(protected_names))
                else:
                    filtered.append(line)
            if not targeted:
                keep.extend(filtered)
        return keep

    from difflib import ndiff

    return [line for line in ndiff(protected(before), protected(after)) if line.startswith(("- ", "+ "))]


def forest_lists(root):
    direct = root / "extension" / "trees"
    return direct if direct.is_dir() else root / "forest_extension" / "trees"


def forest_changes(scope, before, after, *, extra_trees=None):
    if extra_trees is not None:
        from scripts.add_extra_trees import list_files

        for error in extra_trees.errors:
            scope.issue("forest", "trees.npy", "extra_tree_csv", reason=error)
        old, new = [np.load(p / "trees.npy", mmap_mode="r") for p in (before, after)]
        expected = extra_trees.additions
        if old.dtype != new.dtype or len(new) != len(old) + len(expected):
            scope.issue("forest", "trees.npy", "extra_tree_rows")
        else:
            kind = f"V{old.dtype.itemsize}"
            if (not np.array_equal(old.view(kind), new[:len(old)].view(kind))
                    or not np.array_equal(np.asarray(expected).view(kind), new[len(old):].view(kind))):
                scope.issue("forest", "trees.npy", "extra_tree_rows")
        old_files, new_files = list_files(before), list_files(after)
        for name in sorted(old_files.keys() | new_files.keys()):
            left = (extra_trees.tiles[name] if name in extra_trees.tiles else old_files[name].read_bytes()
                    if name in old_files else None)
            right = new_files[name].read_bytes() if name in new_files else None
            if left != right:
                scope.issue("forest", name, "tree_text")
        return
    old, new = [np.load(p / "trees.npy", mmap_mode="r").view(np.recarray) for p in (before, after)]
    if old.dtype != new.dtype:
        scope.issue("forest", "trees.npy", "dtype")
        return
    files = [{p.name: p for p in forest_lists(root).glob("*.txt") if "_before_" not in p.stem}
             for root in (before, after)]
    if not all(files):
        raise ValueError("森の控えと出力に extension/trees/*.txt が必要です")

    # XY は全個体の識別子。高さや柄などを変えても同じ個体として照合する。
    def keys(arr):
        return np.rec.fromarrays((arr.x, arr.y), names=("x", "y"))

    if len(old) == len(new) and np.array_equal(old, new):
        edit, removed, matches = np.empty(0, int), np.empty(0, int), np.empty(0, int)
    else:
        old_xy, new_xy = keys(old), keys(new)
        order = np.argsort(old_xy, kind="stable")
        rank = np.searchsorted(old_xy, new_xy, sorter=order)
        # XY が同じ個体もある。離れた行にあっても出現順で対応させる。
        by_xy = np.argsort(new_xy, kind="stable")
        sorted_xy = new_xy[by_xy]
        duplicate = np.r_[False, sorted_xy[1:] == sorted_xy[:-1]] if len(new) else np.empty(0, bool)
        pos = np.arange(len(new))
        rank[by_xy] += pos - np.maximum.accumulate(np.where(duplicate, 0, pos))
        matches = order[np.minimum(rank, len(order) - 1)] if len(old) else np.full(len(new), -1)
        valid = (rank < len(old)) & (old_xy[matches] == new_xy) if len(old) else np.zeros(len(new), bool)
        for j in np.flatnonzero(~valid):
            scope.issue("forest", "trees.npy", "tree_added", (new.x[j], new.y[j]))
        exist = np.zeros(len(old), bool)
        exist[matches[valid]] = True
        removed = np.flatnonzero(~exist)
        kept = matches[valid]
        if np.any(np.diff(kept) <= 0):
            scope.issue("forest", "trees.npy", "tree_order")
        edit = np.flatnonzero(valid)
        edit = edit[old[matches[edit]] != new[edit]]
    edits = defaultdict(dict)
    drops = defaultdict(set)

    def tile(x, y):
        return int(np.floor(float(x) / 1000)), int(np.floor(float(y) / 1000))

    for j in edit:
        i = matches[j]
        a, b = old[i], new[j]
        xy = (a.x, a.y)
        name = f"forest_{tile(*xy)[0]}_{tile(*xy)[1]}.txt"
        allowed = (scope.dug.covers(xy) and abs(float(b.z) - scope.dug.height(xy)) < .0001
                   and all(a[c] == b[c] for c in old.dtype.names if c != "z"))
        if not allowed:
            scope.issue("forest", "trees.npy", "tree_height_or_attributes", xy)
        edits[name][i] = _line(b)
    for i in removed:
        xy = (old.x[i], old.y[i])
        if not (scope.dug.covers(xy) and scope.removable_tree(xy, forest=True)):
            scope.issue("forest", "trees.npy", "tree_removed", xy)
        tx, ty = tile(*xy)
        drops[f"forest_{tx}_{ty}.txt"].add(i)

    a, b = files
    for name in sorted(a.keys() | b.keys()):
        left = a[name].read_bytes() if name in a else None
        right = b[name].read_bytes() if name in b else None
        if name not in edits and name not in drops:
            if left != right:
                scope.issue("forest", name, "tree_text")
            continue
        tx, ty = map(int, Path(name).stem.split("_")[1:])
        indices = np.flatnonzero((tx <= old.x / 1000) & (old.x / 1000 < tx + 1)
                                 & (ty <= old.y / 1000) & (old.y / 1000 < ty + 1))
        replacements = {j: edits[name][i] for j, i in enumerate(indices) if i in edits[name]}
        removed_rows = {j for j, i in enumerate(indices) if i in drops[name]}
        try:
            expected = _patch_tile(a[name], old[indices], replacements, removed_rows) if name in a else None
        except ValueError:
            expected = None
        if (expected if expected is not None else left) != right:
            scope.issue("forest", name, "tree_text")
        else:
            for i in sorted(edits[name].keys() | drops[name]):
                xy = (old.x[i], old.y[i])
                if (i in edits[name] and not scope.dug.covers(xy)) or (
                        i in drops[name] and not (scope.dug.covers(xy) and scope.removable_tree(xy, forest=True))):
                    scope.issue("forest", name, "tree_text_scope", xy)


def roadside_forest_changes(before, after, *, road, span, band=(1.5, 12.), exclusion=None, manual=None):
    """試験区間の名前変更・末尾への追記だけを許す。タイルは全バイトを照合する。"""
    from scripts import roadside_forest
    from scripts.add_extra_trees import list_files

    old, all_new = [np.load(p / "trees.npy", mmap_mode="r").view(np.recarray) for p in (before, after)]
    if old.dtype.names != all_new.dtype.names or len(all_new) < len(old):
        return ["trees.npy: 行が不足するか列が違います"]
    new, added = all_new[:len(old)], all_new[len(old):]
    if any(old.dtype[n] != all_new.dtype[n] or not np.array_equal(old[n], new[n])
           for n in old.dtype.names if n != "pattern"):
        return ["trees.npy: 既存の位置・向き・大きさ、または行順が違います"]
    changed = old.pattern != new.pattern
    issues = []
    if not road.interval_mask(np.c_[old.x[changed], old.y[changed]], span).all():
        issues.append("trees.npy: 試験区間外で名前が変更されています")
    from yabitsu.roadside_cover import open_mask

    roots = np.round(np.c_[added.x, added.y], 2)
    if not np.isfinite(roots).all():
        return ["trees.npy: 追植の座標に有限でない値があります"]
    if not road.interval_mask(roots, span).all():
        issues.append("trees.npy: 試験区間外に新しい行があります")
    edge = road.edge_distance(roots)
    if np.any((edge < band[0]) | (edge > band[1])):
        issues.append("trees.npy: 舗装端からの植栽帯を超えた追植があります")
    if exclusion is not None and (not open_mask(roots, exclusion).all() or road.locate(roots).bridge.any()):
        issues.append("trees.npy: 保護範囲・橋・地物に追植があります")
    if manual is not None and np.any(cKDTree(manual).query(roots)[0] < 1):
        issues.append("trees.npy: 手植えした幹から1 m未満の追植があります")
    if len(np.unique(roots, axis=0)) != len(roots):
        issues.append("trees.npy: 追植位置に重複があります")
    xy = np.c_[old.x[changed], old.y[changed]]
    towards = road.locate(xy).towards
    if (np.linalg.norm(towards, axis=1) > roadside_forest.PILOT_MODEL_REACH_M).any():
        issues.append("trees.npy: 試験帯の横幅を超えて名前が変更されています")
    expected = roadside_forest.pilot_patterns(old[changed], towards, rotate_by_name=True, manual_xy=manual)
    if not np.array_equal(expected.pattern, new.pattern[changed]):
        issues.append("trees.npy: 既存行がモデルの使い分けと一致しません")
    try:
        edits = dict(roadside_forest.touched_tiles(before, old, new, added))
    except ValueError as exc:
        return [*issues, str(exc)]
    originals, actual = list_files(before), list_files(after)
    for name in sorted(originals.keys() | actual.keys() | edits.keys()):
        left = edits.get(name)
        if left is None and name in originals:
            left = originals[name].read_bytes()
        right = actual[name].read_bytes() if name in actual else None
        if left != right:
            issues.append(f"{name}: 許可範囲の行以外にも差分があります")
    return issues


def check(args):
    before, after = npz(args.before_near), npz(args.after_near)
    sec = Section.load_json(args.section)
    edits = valley_edits.load(args.edits) if getattr(args, "edits", None) is not None else ()
    scope = Scope(sec, track.load_bounds(), before, after, edits=edits,
                  sites_root=getattr(args, "sites_root", config.PROJECT_ROOT / "export/blender_sites"),
                  round2=getattr(args, "round2", False))
    scope.water = tuple(valley_fit.river_surface(root) for root in (args.before_track, args.track))
    if "valley_cells" in after:
        from yabitsu.valley_surface import allowed_area
        from yabitsu.valley_survey import bank_repair
        pads = json_file(config.DATA_PROCESSED, "track_pads.json", [])
        pads += start_lot.flats_for(config.DATA_MANUAL / "s1_start_lot.json", pads)
        keep = tunnel.carve_mask(sec.s, tunnel.from_osm(side_roads.load_elements(), sec))
        permitted = shapely.union_all([
            allowed_area(before, [r for r in scope.rows if r.id[0] == site], area,
                         valleys.protected_area(sec, area, pads, keep), next((repair for r in scope.rows if r.id[0] == site
                                                  if (repair := bank_repair(r)) is not None), None))
            for site, area in scope.sites.items()] + [area.difference(valleys.protected_area(sec, area, pads, keep))
                                                      for _, _, area in scope.tapers])
        if "valley_coarse_remainder" in after:
            changed = after["valley_coarse_remainder"][:, 0] < 0
            tri = after["valley_vertices"][after["valley_faces"][changed]]
            outside = shapely.union_all(shapely.difference(shapely.polygons(tri[..., :2]), permitted))
        else:
            outside = shapely.difference(scope.dug.polygon, permitted)
        if outside.area > 1e-8:
            pt = outside.representative_point()
            scope.issue("near", "valley_cells", "terrain_outside", [pt.x,pt.y], area=outside.area)
    grid, stats = valley_fit.grid_checks(before, after, scope.sites)
    for item in grid:
        scope.issue("near", item.mesh, item.category, item.xy or None, amount=item.amount)
    scope.manifests("track", args.before_track, args.track)
    scope.manifests("dress", args.before_dress, args.dress)
    extra_trees = None
    if getattr(args, "extra_trees", None) is not None:
        from scripts.add_extra_trees import prepare

        extra_trees = prepare(args.before_forest, args.extra_trees, config.DATA_PROCESSED, args.buildings,
                              section=args.section, near=args.after_near, mid=args.after_mid,
                              mesh_roots=(args.track, args.dress))
    forest_changes(scope, args.before_forest, args.forest_dir, extra_trees=extra_trees)
    return dict(terrain=stats, violations=len(scope.issues), counts=dict(sorted(scope.counts.items())),
                allowed={g: sorted(n) for g, n in sorted(scope.allowed.items())}, issues=scope.issues)


def main():
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "roadside":
        from scripts.roadside_forest import PILOT, barriers, inputs, manual_stems
        from yabitsu.roadside_cover import Road

        p = argparse.ArgumentParser(description=roadside_forest_changes.__doc__)
        p.add_argument("before", type=Path)
        p.add_argument("after", type=Path)
        args = p.parse_args(sys.argv[2:])
        sec, spec, bounds = inputs()
        cfg = spec["roadside_cover"]
        road = Road(sec)
        exclusion, _ = barriers(road, spec, config.TRACK_BUILD_DIR)
        manual = manual_stems(Section.load_json(config.DATA_PROCESSED / "track_section.json"), bounds)
        issues = roadside_forest_changes(args.before, args.after, road=road,
                                          span=tuple(bounds["s4"][0]+x for x in PILOT),
                                          band=(cfg["min_edge_m"], cfg["max_edge_m"]), exclusion=exclusion, manual=manual)
        print(json.dumps({"violations": len(issues), "issues": issues}, ensure_ascii=False, indent=2))
        return bool(issues)
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("track", "dress", "near", "forest"):
        p.add_argument(f"--before-{key}", required=True, type=Path)
    p.add_argument("--track", type=Path, default=config.TRACK_BUILD_DIR)
    p.add_argument("--dress", type=Path, default=config.DATA_PROCESSED / "dress_mountain")
    p.add_argument("--after-near", type=Path, default=config.DATA_PROCESSED / "track_terrain_near.npz")
    p.add_argument("--section", type=Path, default=config.DATA_PROCESSED / "track_section.json")
    p.add_argument("--forest-dir", required=True, type=Path)
    p.add_argument("--extra-trees", type=Path, help="追植 CSV。森は追記分だけを許可する")
    p.add_argument("--after-mid", type=Path, help="追植の計算に使う中景地形")
    p.add_argument("--buildings", type=Path, default=config.DATA_MANUAL / "track_buildings.json")
    p.add_argument("--edits", type=Path, help="検査する Blender 手直しのディレクトリ")
    p.add_argument("--sites-root", type=Path, default=config.PROJECT_ROOT / "export/blender_sites")
    p.add_argument("--round2", action="store_true", help="第2巡を基準に、手直し面だけの UV・法線変更を許可する")
    result = check(p.parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return bool(result["violations"])


if __name__ == "__main__":
    raise SystemExit(main())
