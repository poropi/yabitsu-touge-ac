"""森の木を植える地面の高さと、植えない範囲。AC に入っている kn5 の元データ（manifest）から作る。

地面：材質 ground のメッシュ（近景・中景 1GRASS_* を遠景 TERRAIN_FAR_* より優先）。三角形の面で高さを求める。
植えない範囲：ground・tree 以外の全メッシュを真上から塗り、種類ごとの幅だけ広げる（格子 cell m）。
"""
import json
from pathlib import Path

import numpy as np
from matplotlib.tri import LinearTriInterpolator, Triangulation
from PIL import Image, ImageDraw
from scipy import ndimage

# 種類ごとに広げる幅 m（data/manual/forest.json の "clearance" で上書き）
DEFAULT_CLEARANCE = {"road": 2.0, "wall": 0.5, "building": 2.0, "bridge": 3.0, "water": 1.0, "tunnel": 5.0,
                     "other": 1.0, "osm_road": 1.5, "cell": 1.0, "coarse_cell": 4.0}
CLASSES = ("road", "wall", "building", "bridge", "water", "tunnel", "other")


def mesh_class(name: str, material: str):
    n = name.lower()
    if material in ("ground", "tree"):
        return None
    if "bridge" in n:
        return "bridge"
    if "tunnel" in n:
        return "tunnel"
    if material in ("water", "water_fall", "riverbed") or n.startswith(("lake", "water", "riverbed", "s5_water")):
        return "water"
    if n.startswith(("1road_", "scenery_road", "town_lot", "lines_")):
        return "road"
    if "bld_" in n or any(k in n for k in ("houses", "shrine", "building", "tower", "shelter", "roof", "gables")):
        return "building"
    if n.startswith(("1wall_", "guardrail_")):
        return "wall"
    return "other"


def _load(manifest_dir: Path):
    manifest_dir = Path(manifest_dir)
    m = json.loads((manifest_dir / "manifest.json").read_text(encoding="utf-8"))
    for name, e in m["meshes"].items():
        d = np.load(manifest_dir / e["file"])
        f = d["faces"]
        tris = np.r_[f[:, [0, 1, 2]], f[:, [0, 2, 3]]] if f.shape[1] == 4 else f
        yield name, e["material"], d["verts"], tris


class Ground:
    def __init__(self, parts):
        self.parts = parts      # [(bbox, interpolator)]、優先度の低い順

    def z(self, xy) -> np.ndarray:
        xy = np.asarray(xy, float)
        out = np.full(len(xy), np.nan)
        for (x0, y0, x1, y1), interp in self.parts:
            sel = np.flatnonzero((xy[:, 0] >= x0) & (xy[:, 0] <= x1) & (xy[:, 1] >= y0) & (xy[:, 1] <= y1))
            if len(sel) == 0:
                continue
            z = interp(xy[sel, 0], xy[sel, 1])
            ok = ~np.ma.getmaskarray(z)
            out[sel[ok]] = np.ma.getdata(z)[ok]
        return out


def load_ground(manifest_dir: Path) -> Ground:
    far, near = [], []
    for name, mat, v, tris in _load(manifest_dir):
        if mat != "ground":
            continue
        try:
            interp = LinearTriInterpolator(Triangulation(v[:, 0], v[:, 1], tris), v[:, 2])
        except (ValueError, RuntimeError) as e:
            print(f"地面 {name} を読めません: {e}")
            continue
        bbox = (v[:, 0].min(), v[:, 1].min(), v[:, 0].max(), v[:, 1].max())
        (far if name.startswith("TERRAIN_FAR") else near).append((bbox, interp))
    return Ground(far + near)


class Grid:
    def __init__(self, x0, y0, x1, y1, cell):
        self.x0, self.y1, self.cell = x0, y1, cell
        self.shape = (int(np.ceil((y1 - y0) / cell)) + 1, int(np.ceil((x1 - x0) / cell)) + 1)

    def px(self, xy):
        xy = np.asarray(xy, float)
        return (xy[:, 0] - self.x0) / self.cell, (self.y1 - xy[:, 1]) / self.cell

    def rc(self, xy):
        c, r = self.px(xy)
        r, c = np.floor(r).astype(np.int64), np.floor(c).astype(np.int64)
        ok = (r >= 0) & (c >= 0) & (r < self.shape[0]) & (c < self.shape[1])
        return r, c, ok

    def canvas(self):
        return Image.new("L", (self.shape[1], self.shape[0]), 0)

    def draw_tris(self, img, v, tris, value):
        d = ImageDraw.Draw(img)
        c, r = self.px(v[:, :2])
        for a, b, e in tris:
            d.polygon([(c[a], r[a]), (c[b], r[b]), (c[e], r[e])], fill=value, outline=value)


class KeepOut:
    def __init__(self, fine: Grid, blocked, road_d, coarse: Grid, coarse_road, coarse_blocked, cfg):
        self.fine, self._blocked, self._road_d = fine, blocked, road_d
        self.coarse, self._croad, self._cblocked, self.cfg = coarse, coarse_road, coarse_blocked, cfg
        self._coarse_d = None

    def add_lines(self, lines, widths):
        """OSM の道など（中心線と幅）。植えない範囲は 幅/2＋osm_road、道からの距離にも使う。"""
        img_r = Image.fromarray(self._croad.astype(np.uint8) * 255)
        img_b = Image.fromarray(self._cblocked.astype(np.uint8) * 255)
        dr, db = ImageDraw.Draw(img_r), ImageDraw.Draw(img_b)
        cell = self.coarse.cell
        for line, w in zip(lines, widths):
            c, r = self.coarse.px(line)
            pts = list(zip(c, r))
            dr.line(pts, fill=255, width=max(1, int(round(w / cell))))
            db.line(pts, fill=255, width=max(1, int(round((w + 2 * self.cfg["osm_road"]) / cell))))
        self._croad = np.asarray(img_r) > 0
        self._cblocked = np.asarray(img_b) > 0
        self._coarse_d = None

    def blocked(self, xy) -> np.ndarray:
        r, c, ok = self.fine.rc(xy)
        out = np.zeros(len(r), bool)
        out[ok] = self._blocked[r[ok], c[ok]]
        r2, c2, ok2 = self.coarse.rc(xy)
        out[ok2] |= self._cblocked[r2[ok2], c2[ok2]]
        return out

    def road_distance(self, xy) -> np.ndarray:
        if self._coarse_d is None:
            self._coarse_d = (ndimage.distance_transform_edt(~self._croad) * self.coarse.cell).astype(np.float32)
        r, c, ok = self.fine.rc(xy)
        out = np.full(len(r), np.inf)
        out[ok] = self._road_d[r[ok], c[ok]]
        r2, c2, ok2 = self.coarse.rc(xy)
        out[ok2] = np.minimum(out[ok2], self._coarse_d[r2[ok2], c2[ok2]])
        return out


def load_keepout(manifest_dir: Path, cfg: dict) -> KeepOut:
    cfg = {**DEFAULT_CLEARANCE, **cfg}
    items, ground_bb = [], [np.inf, np.inf, -np.inf, -np.inf]
    for name, mat, v, tris in _load(manifest_dir):
        if mat == "ground":
            ground_bb = [min(ground_bb[0], v[:, 0].min()), min(ground_bb[1], v[:, 1].min()),
                         max(ground_bb[2], v[:, 0].max()), max(ground_bb[3], v[:, 1].max())]
        k = mesh_class(name, mat)
        if k is not None:
            items.append((k, v, tris))
    allv = np.concatenate([v for _, v, _ in items]) if items else np.zeros((1, 3))
    m = max(cfg[c] for c in CLASSES) + 10
    fine = Grid(allv[:, 0].min() - m, allv[:, 1].min() - m, allv[:, 0].max() + m, allv[:, 1].max() + m, cfg["cell"])
    label = fine.canvas()
    road = fine.canvas()
    # 広げる幅の小さい順に塗る（重なったら大きい幅が勝つ）
    for k in sorted(CLASSES, key=lambda c: cfg[c]):
        for kk, v, tris in items:
            if kk == k:
                fine.draw_tris(label, v, tris, CLASSES.index(k) + 1)
                if k == "road":
                    fine.draw_tris(road, v, tris, 255)
    lab = np.asarray(label)
    radius = np.array([0.0] + [cfg[c] for c in CLASSES])
    d, idx = ndimage.distance_transform_edt(lab == 0, return_indices=True)
    blocked = d * fine.cell <= radius[lab[idx[0], idx[1]]] + 1e-6
    blocked &= lab[idx[0], idx[1]] > 0
    del d, idx
    road_mask = np.asarray(road) > 0
    road_d = (ndimage.distance_transform_edt(~road_mask) * fine.cell).astype(np.float32)
    if not np.isfinite(ground_bb[0]):
        ground_bb = [fine.x0, fine.y1 - fine.shape[0] * fine.cell, fine.x0 + fine.shape[1] * fine.cell, fine.y1]
    coarse = Grid(*ground_bb, cfg["coarse_cell"])
    cr = coarse.canvas()
    for kk, v, tris in items:
        if kk == "road":
            coarse.draw_tris(cr, v, tris, 255)
    return KeepOut(fine, blocked, road_d, coarse, np.asarray(cr) > 0, np.zeros(coarse.shape, bool), cfg)


def osm_road_lines(bounds, cache=None):
    """地形の範囲（bounds=(x0, y0, x1, y1)、ローカル座標）の OSM 車道の中心線と幅。林道（track）と歩道は含めない。"""
    from . import config, geo, osm, scenery
    cache = cache or config.DATA_RAW / "osm_forest_roads.json"
    lat0, lon0 = geo.from_local(bounds[0], bounds[1])
    lat1, lon1 = geo.from_local(bounds[2], bounds[3])
    kinds = [k for k in scenery.WIDTH if k not in ("track", "footway", "path", "steps", "cycleway", "pedestrian")]
    q = (f'[out:json][timeout:180];way["highway"~"^({"|".join(kinds)})$"]'
         f"({min(lat0, lat1)},{min(lon0, lon1)},{max(lat0, lat1)},{max(lon0, lon1)});out geom;")
    data = osm.fetch_overpass(q, cache)
    lines, widths = [], []
    for e in data["elements"]:
        g = e.get("geometry")
        if not g:
            continue
        x, y = geo.to_local([p["lat"] for p in g], [p["lon"] for p in g])
        lines.append(np.c_[x, y]); widths.append(scenery.WIDTH.get(e["tags"].get("highway"), 4.0))
    return lines, widths
