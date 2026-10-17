import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

from . import geo


@dataclass
class Centerline:
    s: np.ndarray
    x: np.ndarray
    y: np.ndarray
    heading: np.ndarray
    curvature: np.ndarray
    lanes: np.ndarray
    bridge: np.ndarray
    step: float

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **{k: getattr(self, k) for k in ("s", "x", "y", "heading", "curvature", "lanes", "bridge")},
                 step=self.step)

    @classmethod
    def load(cls, path):
        d = np.load(path)
        return cls(d["s"], d["x"], d["y"], d["heading"], d["curvature"], d["lanes"], d["bridge"], float(d["step"]))


def _cum(xy):
    return np.r_[0.0, np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]


def resample_polyline(xy, step=1.0):
    xy = np.asarray(xy, float)
    keep = np.r_[True, np.hypot(*np.diff(xy, axis=0).T) > 1e-9]
    xy = xy[keep]
    s = _cum(xy)
    n = max(1, int(round(s[-1] / step)))
    s_new = np.linspace(0.0, s[-1], n + 1)
    return s_new, np.c_[np.interp(s_new, s, xy[:, 0]), np.interp(s_new, s, xy[:, 1])], s[-1] / n


def seg_value_at(s_vert, seg_values, s_query):
    j = np.clip(np.searchsorted(s_vert, s_query, side="right") - 1, 0, len(seg_values) - 1)
    return np.asarray(seg_values)[j]


def smooth_xy(xy, sigma_m, step):
    sg = sigma_m / step
    return np.c_[gaussian_filter1d(xy[:, 0], sg, mode="nearest"), gaussian_filter1d(xy[:, 1], sg, mode="nearest")]


def heading_curvature(xy, step):
    dx, dy = np.gradient(xy[:, 0], step), np.gradient(xy[:, 1], step)
    ddx, ddy = np.gradient(dx, step), np.gradient(dy, step)
    return np.unwrap(np.arctan2(dy, dx)), (dx * ddy - dy * ddx) / np.power(dx * dx + dy * dy, 1.5)


def apply_offsets(s, xy, heading, offsets):
    offsets = np.asarray(offsets, float).reshape(-1, 2)
    if len(offsets) == 0:
        return xy.copy()
    o = np.argsort(offsets[:, 0])
    lat = np.interp(s, offsets[o, 0], offsets[o, 1])
    return xy + lat[:, None] * np.c_[-np.sin(heading), np.cos(heading)]


def load_offsets_csv(path):
    p = Path(path)
    if not p.exists():
        return np.zeros((0, 2))
    with p.open(encoding="utf-8") as f:
        rows = [(float(r["s_m"]), float(r["lateral_m"])) for r in csv.DictReader(f) if r.get("s_m")]
    return np.array(rows, float).reshape(-1, 2)


def build_centerline(route, project=geo.to_local, step=1.0, sigma_m=4.0, offsets=None):
    x, y = project(np.asarray(route.lat, float), np.asarray(route.lon, float))
    raw = np.c_[x, y]
    s_vert = _cum(raw)
    s, xy, st = resample_polyline(raw, step)
    lanes = seg_value_at(s_vert, route.seg_lanes, s)
    bridge = seg_value_at(s_vert, route.seg_bridge, s).astype(bool)
    xy = smooth_xy(xy, sigma_m, st)
    if offsets is not None and len(offsets):
        h, _ = heading_curvature(xy, st)
        xy = smooth_xy(apply_offsets(s, xy, h, offsets), 2.0, st)
        s_old = _cum(xy)
        s, xy, st = resample_polyline(xy, step)
        lanes = seg_value_at(s_old, lanes[:-1], s)
        bridge = seg_value_at(s_old, bridge[:-1], s).astype(bool)
    h, k = heading_curvature(xy, st)
    return Centerline(s, xy[:, 0], xy[:, 1], h, k, lanes.astype(np.int32), bridge, st)
