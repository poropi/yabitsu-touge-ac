"""地理院の空中写真（シームレス空中写真）から、森／森以外と、針葉樹／広葉樹の地図を作る。

出典：地理院タイル（シームレス空中写真）。
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.ndimage import uniform_filter

from . import geo, gsi, photo

CLASSES = {0: "open", 1: "conifer", 2: "broadleaf", 3: "water"}
COLORS = np.array([(215, 200, 160), (20, 70, 40), (120, 170, 60), (60, 110, 200)], np.uint8)

# 値は data/manual/forest.json の "classify" で上書きする。
# 全域の写真は撮影時期の違う写真のつぎはぎ（夏の緑・夏の青み・冬）。明るさや色の絶対値では森を決めず、
# 「きめの粗さ」（樹冠と影のざらざら：明るさの局所的な変動係数）で森を決める（2026-10-19 夏・冬の写真で実測：
# 森 0.12〜0.27、伐採地・畑・裸地 0.05〜0.10）。
DEFAULT_CLASSIFY = {
    "cell_m": 4.0,
    "tex_window_m": 10.0,       # きめを測る範囲
    "cv_min": 0.105,            # きめの粗さがこれ以上なら森
    "shadow_lum": 0.13,         # これより暗いところ（谷の深い影）は森
    "block_m": 500.0,           # 写真の違いをならす区画
    "brown_rg": 1.0,            # R/G がこれ以上の森は茶色（冬の写真の落葉広葉樹）
    "winter_brown_share": 0.08, # 区画の森のうち茶色がこれ以上なら冬の写真：青緑の森はすべて針葉樹
    "conifer_rel_lum": 1.0,     # 夏の写真：区画の森の明るさの中央値×これ より暗い緑の森は針葉樹
    "open_rel_lum": 1.1,        # なめらかな所は、区画の森の明るさの中央値×これ より明るい時だけ森以外
    "smooth_cells": 2,          # 多数決でならす範囲（マス）
}


@dataclass
class ForestMap:
    x0: float       # 西の端
    y1: float       # 北の端（cls の 0 行目）
    cell: float
    cls: np.ndarray

    def at(self, xy) -> np.ndarray:
        xy = np.asarray(xy, float)
        c = np.floor((xy[:, 0] - self.x0) / self.cell).astype(np.int64)
        r = np.floor((self.y1 - xy[:, 1]) / self.cell).astype(np.int64)
        ok = (c >= 0) & (r >= 0) & (c < self.cls.shape[1]) & (r < self.cls.shape[0])
        out = np.zeros(len(xy), np.uint8)
        out[ok] = self.cls[r[ok], c[ok]]
        return out

    def save(self, path: Path):
        np.savez_compressed(path, x0=self.x0, y1=self.y1, cell=self.cell, cls=self.cls)

    @classmethod
    def load(cls, path: Path):
        d = np.load(path)
        return cls(float(d["x0"]), float(d["y1"]), float(d["cell"]), d["cls"])


def _cells(a, k):
    h, w = a.shape[0] // k * k, a.shape[1] // k * k
    return a[:h, :w].reshape(h // k, k, w // k, k, *a.shape[2:])


def classify(rgb: np.ndarray, px_size: float, cfg: dict) -> np.ndarray:
    k = max(1, int(round(cfg["cell_m"] / px_size)))
    f = rgb.astype(np.float32) / 255.0
    lum = f.mean(-1)
    win = max(3, int(round(cfg["tex_window_m"] / px_size)) | 1)
    m = ndimage.uniform_filter(lum, win)
    sd = np.sqrt(np.maximum(ndimage.uniform_filter(lum * lum, win) - m * m, 0))
    cv = _cells(sd / np.maximum(m, 1e-3), k).mean((1, 3))
    c = _cells(f, k).mean((1, 3))
    lum_c = c.mean(-1)
    rough = (cv >= cfg["cv_min"]) | (lum_c < cfg["shadow_lum"])
    forest = rough   # なめらかでも、周りの森より暗ければ（葉の詰まった森）森にする（下で区画ごとに足す）
    rg = c[..., 0] / np.maximum(c[..., 1], 1e-3)
    brown = rg >= cfg["brown_rg"]
    # 区画ごとに「冬の写真か」と「森の明るさの中央値」を求め、区画の中心の間をなめらかにつないで使う（境目が四角くならない）
    green = forest & ~brown
    nb = max(1, int(round(cfg["block_m"] / cfg["cell_m"])))
    R, C = int(np.ceil(forest.shape[0] / nb)), int(np.ceil(forest.shape[1] / nb))
    share = np.zeros((R, C), np.float32); med = np.full((R, C), np.nan, np.float32)
    for i in range(R):
        for j in range(C):
            sl = (slice(i * nb, (i + 1) * nb), slice(j * nb, (j + 1) * nb))
            fo = forest[sl]
            if fo.any():
                share[i, j] = brown[sl][fo].mean()
            if green[sl].any():
                med[i, j] = np.median(lum_c[sl][green[sl]])
    if np.isnan(med).all():
        med[:] = 0.5
    med = np.where(np.isnan(med), np.nanmedian(med), med)
    zoom = (forest.shape[0] / R, forest.shape[1] / C)
    share_f = ndimage.zoom(share, zoom, order=1, mode="nearest", grid_mode=True)[:forest.shape[0], :forest.shape[1]]
    med_f = ndimage.zoom(med, zoom, order=1, mode="nearest", grid_mode=True)[:forest.shape[0], :forest.shape[1]]
    winter = share_f >= cfg["winter_brown_share"]
    # なめらかな所は、区画の森の明るさの中央値×open_rel_lum より明るければ森以外（伐採地・畑）、暗ければ森
    fmed = np.full((R, C), np.nan, np.float32)
    for i in range(R):
        for j in range(C):
            sl = (slice(i * nb, (i + 1) * nb), slice(j * nb, (j + 1) * nb))
            if rough[sl].any():
                fmed[i, j] = np.median(lum_c[sl][rough[sl]])
    fmed = np.where(np.isnan(fmed), np.nanmedian(fmed) if not np.isnan(fmed).all() else 0.5, fmed)
    fmed_f = ndimage.zoom(fmed, zoom, order=1, mode="nearest", grid_mode=True)[:forest.shape[0], :forest.shape[1]]
    forest = rough | (lum_c < fmed_f * cfg["open_rel_lum"])
    green = forest & ~brown
    conifer = green & (winter | (lum_c < med_f * cfg["conifer_rel_lum"]))
    cls = np.where(forest, np.where(conifer, 1, 2), 0).astype(np.uint8)
    n = int(cfg["smooth_cells"])
    if n > 0:
        votes = np.stack([uniform_filter((cls == i).astype(np.float32), 2 * n + 1, mode="nearest") for i in range(3)])
        cls = votes.argmax(0).astype(np.uint8)
    return cls


def clear_polygons(fm: ForestMap, polygons):
    """多角形（畑・住宅地・ゴルフ場・水など、ローカル座標）の中を森以外にする。"""
    img = Image.fromarray(np.zeros(fm.cls.shape, np.uint8))
    d = ImageDraw.Draw(img)
    for poly in polygons:
        poly = np.asarray(poly, float)
        d.polygon(list(zip((poly[:, 0] - fm.x0) / fm.cell, (fm.y1 - poly[:, 1]) / fm.cell)), fill=1, outline=1)
    fm.cls[np.asarray(img) > 0] = 0


OPEN_TAGS = {"landuse": ("residential", "farmland", "meadow", "grass", "industrial", "commercial", "retail",
                         "orchard", "allotments", "recreation_ground", "cemetery", "construction", "quarry",
                         "farmyard", "greenhouse_horticulture", "railway", "brownfield"),
             "leisure": ("golf_course", "park", "pitch", "sports_centre", "stadium", "playground"),
             "natural": ("water", "grassland", "bare_rock", "scree", "beach", "sand"),
             "amenity": ("parking", "school", "hospital"), "building": None}


def osm_open_areas(bounds, cache=None):
    """森にしない場所（OSM の土地利用・建物）の多角形（ローカル座標）。"""
    from . import config, osm
    cache = cache or config.DATA_RAW / "osm_forest_open.json"
    lat0, lon0 = geo.from_local(bounds[0], bounds[1])
    lat1, lon1 = geo.from_local(bounds[2], bounds[3])
    bb = f"({min(lat0, lat1)},{min(lon0, lon1)},{max(lat0, lat1)},{max(lon0, lon1)})"
    parts = [f'way["{k}"~"^({"|".join(v)})$"]{bb};' if v else f'way["{k}"]{bb};' for k, v in OPEN_TAGS.items()]
    parts += [f'relation["{k}"~"^({"|".join(v)})$"]{bb};' for k, v in OPEN_TAGS.items() if v]
    data = osm.fetch_overpass(f"[out:json][timeout:300];({''.join(parts)});out geom;", cache)
    polys = []
    for e in data["elements"]:
        rings = [e["geometry"]] if e.get("geometry") else [m["geometry"] for m in e.get("members", [])
                                                          if m.get("role") == "outer" and m.get("geometry")]
        for g in rings:
            if len(g) >= 3:
                x, y = geo.to_local([p["lat"] for p in g], [p["lon"] for p in g])
                polys.append(np.c_[x, y])
    return polys


def fetch_mosaic(bounds, px_size: float, z: int, cache=None, chunk_m=1000.0) -> np.ndarray:
    """bounds=(x0, y0, x1, y1)。北が上の画像を、chunk_m 四方ずつ取って貼り合わせる。"""
    cache = cache or gsi.TileCache()
    x0, y0, x1, y1 = bounds
    n = int(round(chunk_m / px_size))
    cols, rows = int(np.ceil((x1 - x0) / chunk_m)), int(np.ceil((y1 - y0) / chunk_m))
    out = np.zeros((rows * n, cols * n, 3), np.uint8)
    for r in range(rows):
        for cidx in range(cols):
            img = photo.render_local_image(cache, geo.from_local, x0 + cidx * chunk_m, y1 - (r + 1) * chunk_m,
                                           chunk_m, px_size, z=z)
            out[r * n:(r + 1) * n, cidx * n:(cidx + 1) * n] = img
    return out


def build(bounds, cfg: dict, px_size=2.0, z=17, cache=None):
    """(ForestMap, 写真) を返す。地図の北西の角は bounds の (x0, y1)。"""
    rgb = fetch_mosaic(bounds, px_size, z, cache)
    return ForestMap(bounds[0], bounds[3], cfg["cell_m"], classify(rgb, px_size, cfg)), rgb


def preview(rgb: np.ndarray, fm: ForestMap, path: Path, max_px=4000):
    """空中写真｜色分け地図 を横に並べた PNG。"""
    a = Image.fromarray(rgb)
    b = Image.fromarray(COLORS[fm.cls]).resize(a.size, Image.NEAREST)
    s = min(1.0, max_px / (2 * a.width))
    size = (max(1, int(a.width * s)), max(1, int(a.height * s)))
    out = Image.new("RGB", (size[0] * 2, size[1]))
    out.paste(a.resize(size), (0, 0)); out.paste(b.resize(size, Image.NEAREST), (size[0], 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    out.save(path)
