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

CLASSES = {0: "open", 1: "conifer", 2: "broadleaf", 3: "water", 4: "garden"}
COLORS = np.array([(215, 200, 160), (20, 70, 40), (120, 170, 60), (60, 110, 200), (230, 120, 200)], np.uint8)

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
    "farm_m": 150.0,            # 家からこの距離までの茶色い所は畑・田（森にしない）
    "settle_m": 40.0,           # 家からこの距離までは集落（暗い緑だけを森にする）
    "settle_rg_max": 0.93,
    "settle_lum_max": 0.48,
    "garden_rg_max": 0.98,      # 集落で森にしなかった、きめの粗い緑（R/G がこれ未満で G>B）は庭木（4）
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
    forest &= ~(brown & ~winter)        # 夏の写真の茶色い所（耕した畑・刈った田）は、ざらざらでも森にしない
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


def restore_above_water(fm: ForestMap, raw: np.ndarray, polygons, ground_z, level_m: float, margin_m: float = 3.0) -> int:
    """湖の多角形（OSM の natural=water）の中でも、地形が水面＋margin_m より高く、空中写真では森（raw > 0）のマスを森に戻す。
    OSM の宮ヶ瀬湖の輪郭は陸の斜面まで含んでいて（北原の南東の丘、湖面より 15〜65 m 上）、clear_polygons で森が消えていた（2026-10-27）。
    戻したマスの数を返す。"""
    img = Image.fromarray(np.zeros(fm.cls.shape, np.uint8))
    d = ImageDraw.Draw(img)
    for poly in polygons:
        poly = np.asarray(poly, float)
        d.polygon(list(zip((poly[:, 0] - fm.x0) / fm.cell, (fm.y1 - poly[:, 1]) / fm.cell)), fill=1, outline=1)
    rr, cc = np.nonzero((np.asarray(img) > 0) & (fm.cls == 0) & (raw > 0))
    if not len(rr):
        return 0
    xy = np.c_[fm.x0 + (cc + 0.5) * fm.cell, fm.y1 - (rr + 0.5) * fm.cell]
    z = np.asarray(ground_z(xy), float)
    ok = np.isfinite(z) & (z > level_m + margin_m)
    fm.cls[rr[ok], cc[ok]] = raw[rr[ok], cc[ok]]
    return int(ok.sum())


def mountain_forest(fm: ForestMap, raw: np.ndarray, rgb: np.ndarray, px_size: float, keep_open: np.ndarray, slope_deg: np.ndarray,
                    min_slope_deg: float = 15.0, green_gr: float = 1.06, bare_lum: float = 0.55) -> int:
    """山の斜面は森にする（2026-10-27、ユーザー「基本的に山は木々で覆われているはず」）。開けた土地のマスのうち、keep_open（OSM の土地利用・
    家のまわり）の外で、傾きが min_slope_deg 以上で、写真では森（raw > 0。冬の茶色い広葉樹を集落・畑として外していた所）か、写真が緑っぽい
    （G/R ≥ green_gr かつ明るさ < bare_lum。夏の写真の森をまだらに開けた土地としていた所）マスを森にする。ベージュ・白のむき出しの土や
    岩（崩れた斜面・採石場）は残す。森にしたマスの数を返す。"""
    k = max(1, int(round(fm.cell / px_size)))
    c = _cells(rgb.astype(np.float32) / 255.0, k).mean(axis=(1, 3))
    h, w = min(c.shape[0], fm.cls.shape[0]), min(c.shape[1], fm.cls.shape[1])
    c = c[:h, :w]; r, g = c[..., 0], c[..., 1]; lum = c.mean(-1)
    green = (g / np.maximum(r, 1e-3) >= green_gr) & (lum < bare_lum)
    sub = fm.cls[:h, :w]
    pick = (sub == 0) & ~keep_open[:h, :w] & (slope_deg[:h, :w] >= min_slope_deg) & ((raw[:h, :w] > 0) | green)
    sub[pick] = np.where(raw[:h, :w][pick] > 0, raw[:h, :w][pick], 2)   # 写真が開けた土地なら広葉樹（夏の写真の森）
    return int(pick.sum())


def slope_map(fm: ForestMap, ground_z, mask=None) -> np.ndarray:
    """マスごとの地形の傾き（度）。mask のマスだけ測る（ほかは 0）。ground_z(xy) は地形の高さ（無ければ nan → 0 度）。"""
    out = np.zeros(fm.cls.shape, np.float32)
    rr, cc = np.nonzero(np.ones(fm.cls.shape, bool) if mask is None else mask)
    if not len(rr):
        return out
    x = fm.x0 + (cc + 0.5) * fm.cell; y = fm.y1 - (rr + 0.5) * fm.cell; e = fm.cell
    z = lambda dx, dy: np.asarray(ground_z(np.c_[x + dx, y + dy]), float)
    gx = (z(e, 0) - z(-e, 0)) / (2 * e); gy = (z(0, e) - z(0, -e)) / (2 * e)
    s = np.degrees(np.arctan(np.hypot(gx, gy)))
    out[rr, cc] = np.nan_to_num(s, nan=0.0)
    return out


def osm_lake_polygons(bounds, cache=None):
    """湖（OSM の natural=water・water=lake|reservoir）の多角形（ローカル座標）。osm_open_areas と同じキャッシュから。"""
    from . import config
    cache = cache or config.DATA_RAW / "osm_forest_open.json"
    import json
    data = json.loads(Path(cache).read_text(encoding="utf-8"))
    polys = []
    for e in data["elements"]:
        tg = e.get("tags", {})
        if tg.get("natural") != "water" or tg.get("water") not in ("lake", "reservoir"):
            continue
        rings = [e["geometry"]] if e.get("geometry") else [m["geometry"] for m in e.get("members", [])
                                                          if m.get("role") == "outer" and m.get("geometry")]
        for g in rings:
            if len(g) >= 3:
                x, y = geo.to_local([p["lat"] for p in g], [p["lon"] for p in g])
                polys.append(np.c_[x, y])
    return polys


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


def settlement_mask(fm: ForestMap, polygons, tri_meshes, radius_m=40.0) -> np.ndarray:
    """家（多角形、または (頂点, 三角形) のメッシュ）から radius_m 以内のマス＝集落。"""
    img = Image.fromarray(np.zeros(fm.cls.shape, np.uint8))
    d = ImageDraw.Draw(img)
    px = lambda p: list(zip((p[:, 0] - fm.x0) / fm.cell, (fm.y1 - p[:, 1]) / fm.cell))
    for poly in polygons:
        d.polygon(px(np.asarray(poly, float)), fill=1, outline=1)
    for v, tris in tri_meshes:
        for t in tris:
            d.polygon(px(np.asarray(v, float)[list(t), :2]), fill=1, outline=1)
    houses = np.asarray(img) > 0
    if not houses.any():
        return houses
    return ndimage.distance_transform_edt(~houses) * fm.cell <= radius_m


def settlement_filter(fm: ForestMap, rgb, px_size, mask, cfg):
    """集落では、暗い緑（R/G が settle_rg_max 未満、明るさ settle_lum_max 未満）だけを森に残す
    （田んぼ・畑・屋根の並びは、ざらざらに写っても森にしない。2026-10-19 名古木〜蓑毛の写真で実測：
    森 明るさ 0.39・R/G 0.85、畑・田んぼ・家 0.55〜0.73・0.99〜1.0）。"""
    k = max(1, int(round(fm.cell / px_size)))
    c = _cells(rgb.astype(np.float32) / 255.0, k).mean((1, 3))
    h, w = min(c.shape[0], fm.cls.shape[0]), min(c.shape[1], fm.cls.shape[1])
    c, m = c[:h, :w], mask[:h, :w]
    lum = c.mean(-1)
    rg = c[..., 0] / np.maximum(c[..., 1], 1e-3)
    green = (rg < cfg["settle_rg_max"]) & (lum < cfg["settle_lum_max"])
    sub = fm.cls[:h, :w]
    # 森にしない所のうち、きめが粗く緑っぽい所は庭木・生け垣（名古木で、道ぞいの庭木まで消えて芝生ばかりに見えた。2026-10-28）
    garden = m & ~green & (sub > 0) & (sub < 3) & (rg < cfg.get("garden_rg_max", 0.98)) & (c[..., 1] > c[..., 2])
    sub[m & ~green] = 0
    sub[garden] = 4
    # 家の影は暗い緑の点に見える：集落の中では 3×3 マス（12 m）より小さい森の点を消す
    forest = (sub > 0) & (sub < 3)
    opened = ndimage.binary_opening(forest, np.ones((3, 3), bool))
    sub[m & forest & ~opened] = 0


def osm_buildings(bounds, cache=None):
    """OSM の建物の多角形（osm_open_areas と同じキャッシュから）。"""
    from . import config
    cache = cache or config.DATA_RAW / "osm_forest_open.json"
    osm_open_areas(bounds, cache)       # まだ取っていなければ取る
    import json
    polys = []
    for e in json.loads(Path(cache).read_text(encoding="utf-8"))["elements"]:
        if "building" in e.get("tags", {}) and e.get("geometry"):
            g = e["geometry"]
            x, y = geo.to_local([p["lat"] for p in g], [p["lon"] for p in g])
            polys.append(np.c_[x, y])
    return polys


def farm_filter(fm: ForestMap, rgb, px_size, polygons, tri_meshes, cfg):
    """家から farm_m 以内の茶色い所（R/G ≥ brown_rg）は畑・田として森にしない。
    冬の写真の落葉樹も茶色に写るが、それは家から離れた山にある（2026-10-19：S3 の神社の西の畑が森になっていた）。"""
    near = settlement_mask(fm, polygons, tri_meshes, cfg["farm_m"])
    k = max(1, int(round(fm.cell / px_size)))
    c = _cells(rgb.astype(np.float32) / 255.0, k).mean((1, 3))
    h, w = min(c.shape[0], fm.cls.shape[0]), min(c.shape[1], fm.cls.shape[1])
    brown = c[:h, :w, 0] / np.maximum(c[:h, :w, 1], 1e-3) >= cfg["brown_rg"]
    sub = fm.cls[:h, :w]
    sub[near[:h, :w] & brown] = 0
