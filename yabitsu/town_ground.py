"""町の範囲（名古木交差点のまわり）の地面の高さ。町の道・歩道・建物・地形は、すべてこの高さにそろえる。

地理院の標高を 1m 格子で取ってなめらかにし（家や塀の段差を消す）、本線の路面の範囲（側溝の外側の縁まで）では
本線の路面の高さそのもの、そこから blend_m の間でなめらかに地面へ移す。本線の帯と町の路面が境目で段差なくつながる。"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

from . import config, geo
from .crosssection import GUTTER_WIDTH, surface_height
from .terrain import fill_nan_nearest, road_frame

CELL = 1.0


@dataclass
class TownGround:
    xs: np.ndarray
    ys: np.ndarray
    Z: np.ndarray  # [len(ys), len(xs)]

    def z(self, x, y):
        """双一次補間（範囲の外は端の値）。"""
        x = np.asarray(x, float); y = np.asarray(y, float)
        fx = np.clip((x - self.xs[0]) / CELL, 0, len(self.xs) - 1.000001)
        fy = np.clip((y - self.ys[0]) / CELL, 0, len(self.ys) - 1.000001)
        i, j = np.floor(fx).astype(int), np.floor(fy).astype(int)
        tx, ty = fx - i, fy - j
        Z = self.Z
        return (Z[j, i] * (1 - tx) * (1 - ty) + Z[j, i + 1] * tx * (1 - ty)
                + Z[j + 1, i] * (1 - tx) * ty + Z[j + 1, i + 1] * tx * ty)

    def save(self, path):
        np.savez_compressed(path, xs=self.xs, ys=self.ys, Z=self.Z)

    @classmethod
    def load(cls, path):
        d = np.load(path)
        return cls(d["xs"], d["ys"], d["Z"])


def build(sec, sampler, center_xy, radius=config.TOWN_RADIUS_M + 30.0, sigma_m=5.0, blend_m=8.0,
          unproject=geo.from_local):
    cx, cy = center_xy
    xs = np.arange(np.floor((cx - radius) / CELL) * CELL, cx + radius + CELL, CELL)
    ys = np.arange(np.floor((cy - radius) / CELL) * CELL, cy + radius + CELL, CELL)
    X, Y = np.meshgrid(xs, ys)
    dem = np.asarray(sampler.sample(*unproject(X.ravel(), Y.ravel())), float).reshape(X.shape)
    dem = gaussian_filter(fill_nan_nearest(dem), sigma_m / CELL, mode="nearest")
    f = road_frame(sec, X, Y)
    idx, lat, along = f["idx"], f["lat"], f["along"]
    left = lat >= 0
    hw = np.where(left, sec.hw_left[idx], sec.hw_right[idx])
    form = hw + GUTTER_WIDTH
    road = surface_height(np.clip(lat, -hw, hw), sec.z[idx], sec.slope[idx], sec.crown[idx])
    e = np.hypot(np.maximum(np.abs(lat) - form, 0), np.maximum(np.abs(along) - 0.75, 0))
    t = np.clip(e / blend_m, 0.0, 1.0); w = t * t * (3 - 2 * t)
    Z = (1 - w) * road + w * dem.ravel()
    return TownGround(xs, ys, Z.reshape(X.shape))
