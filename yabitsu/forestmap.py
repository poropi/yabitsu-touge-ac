"""地理院の空中写真（シームレス空中写真）から、森／森以外と、針葉樹／広葉樹の地図を作る。

出典：地理院タイル（シームレス空中写真）。
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import uniform_filter

from . import geo, gsi, photo

CLASSES = {0: "open", 1: "conifer", 2: "broadleaf", 3: "water"}
COLORS = np.array([(215, 200, 160), (20, 70, 40), (120, 170, 60), (60, 110, 200)], np.uint8)

# 明るさは 0〜1。値は data/manual/forest.json の "classify" で上書きする。
# この地域の写真は全体に青みがかる。森は暗い青緑、伐採地・畑・裸地・道は明るい（2026-10-19 S1 付近の実測）。
DEFAULT_CLASSIFY = {
    "cell_m": 4.0,
    "bright_max": 0.46,         # これより明るいものは森にしない（伐採地・畑・裸地・屋根）
    "rg_max": 0.95,             # R/G がこれ以上（茶色・灰色）は森にしない
    # 森のうち、(G−B) + 重み×(明るさ−基準) がこれ未満（青く暗い緑）は杉・檜、以上（黄緑で明るい）は広葉樹
    "conifer_score_max": 0.005,
    "conifer_bright_weight": 0.5,
    "bright_ref": 0.38,
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
    c = _cells(rgb.astype(np.float32) / 255.0, k).mean((1, 3))
    bright = c.mean(-1)
    forest = (bright < cfg["bright_max"]) & (c[..., 0] < cfg["rg_max"] * c[..., 1])
    score = (c[..., 1] - c[..., 2]) + cfg["conifer_bright_weight"] * (bright - cfg["bright_ref"])
    conifer = score < cfg["conifer_score_max"]
    cls = np.where(forest, np.where(conifer, 1, 2), 0).astype(np.uint8)
    n = int(cfg["smooth_cells"])
    if n > 0:
        votes = np.stack([uniform_filter((cls == i).astype(np.float32), 2 * n + 1, mode="nearest") for i in range(3)])
        cls = votes.argmax(0).astype(np.uint8)
    return cls


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
