import csv
import json
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from shapely.geometry import Polygon

from . import geo

DEFAULT_WIDTH_BY_LANES = {0: 4.0, 1: 4.0, 2: 6.0}
GUTTER_WIDTH = 0.6
GUTTER_DEPTH = 0.12
SKIRT_DEPTH = 1.0  # 盛り土側で地形と側溝の縁の間にすき間が見えないよう、縁から下へ垂らす深さ
CROWN_SLOPE = 0.02


def surface_height(lat, z_c, slope, crown):
    return z_c + slope * lat - crown * np.abs(lat)


def parse_width_rows(rows):
    return [{"s_from": float(r["s_from"]), "s_to": float(r["s_to"]), "width_m": float(r["width_m"]),
             "lanes": int(r["lanes"]) if r.get("lanes") not in (None, "") else None,
             **({"taper_m": float(r["taper_m"])} if r.get("taper_m") else {})}
            for r in rows if r.get("s_from") not in (None, "")]


def load_widths_csv(path):
    p = Path(path)
    if not p.exists():
        return []
    with p.open(encoding="utf-8") as f:
        return parse_width_rows(list(csv.DictReader(f)))


def parse_bay_rows(rows):
    """バス停の切り込み（片側だけ広がる所）の表：s_from, s_to, side（L/R）, depth_m, taper_m, note。depth_m が負なら細くなる所。"""
    return [{"s_from": float(r["s_from"]), "s_to": float(r["s_to"]), "side": str(r["side"]).strip(),
             "depth_m": float(r["depth_m"]), "taper_m": float(r.get("taper_m") or 10.0)}
            for r in rows if r.get("s_from") not in (None, "")]


def apply_bays(s, hw_left, hw_right, bays):
    """s_from〜s_to の間はその側を depth_m 広げ、前後 taper_m でなめらかに（smoothstep）もとの幅へ戻す。"""
    hl, hr = np.array(hw_left, float), np.array(hw_right, float)
    for b in bays:
        t = np.clip(np.minimum((s - (b["s_from"] - b["taper_m"])) / b["taper_m"], ((b["s_to"] + b["taper_m"]) - s) / b["taper_m"]), 0, 1)
        add = b["depth_m"] * t * t * (3 - 2 * t)
        if b["side"] == "L":
            hl = hl + add
        else:
            hr = hr + add
    return hl, hr


def smoothstep(t):
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)


def width_tapers(rows):
    """幅員表のすりつけ。開始断面は直前の行の幅、終了断面は指定幅。"""
    previous = None
    for row in rows:
        length = row.get("taper_m", 0)
        if length:
            start = row["s_from"]
            end = start + length
            if previous is None or previous["s_to"] != start or not start < end <= row["s_to"]:
                raise ValueError(f"すりつけは前の幅員区間に接し、指定区間内で終わる必要があります：{row}")
            yield start, end, previous["width_m"], row["width_m"]
        previous = row


def widths(s, lanes, rows):
    lanes = np.asarray(lanes).astype(np.int32).copy()
    w = np.array([DEFAULT_WIDTH_BY_LANES.get(int(l), 6.0) for l in lanes])
    for r in rows:
        m = (s >= r["s_from"]) & (s <= r["s_to"])
        w[m] = r["width_m"]
        if r.get("lanes"):
            lanes[m] = r["lanes"]
    for a, b, initial, final in width_tapers(rows):
        m = (s >= a) & (s < b)
        w[m] = initial + (final - initial) * smoothstep((s[m] - a) / (b - a))
    return w, lanes


def update_width_tapers(sec, rows, bays, *, restore_step=False):
    """保存済みの断面へ、幅員表に指定したすりつけだけを反映する。"""
    for a, b, initial, final in width_tapers(rows):
        mask = (sec.s >= a) & (sec.s < b)
        stations = sec.s[mask]
        width = (np.full_like(stations, final) if restore_step else
                 initial + (final - initial) * smoothstep((stations - a) / (b - a)))
        sec.hw_left[mask], sec.hw_right[mask] = apply_bays(stations, width / 2, width / 2, bays)
    return sec


def width_taper_areas(sec, rows, *, along=5., across=3.):
    """すりつけの前後 along m と、左右の舗装端から across m の帯。"""
    for start, stop, _, _ in width_tapers(rows):
        a, b = max(sec.s[0], start - along), min(sec.s[-1], stop + along)
        s = np.r_[a, sec.s[(a < sec.s) & (sec.s < b)], b]
        x, y, heading, left, right = [
            np.interp(s, sec.s, values)
            for values in (sec.x, sec.y, np.unwrap(sec.heading), sec.hw_left, sec.hw_right)]
        normal = np.c_[-np.sin(heading), np.cos(heading)]
        center = np.c_[x, y]
        yield a, b, Polygon(np.r_[center + normal * (left + across)[:, None],
                                  (center - normal * (right + across)[:, None])[::-1]])


def cross_slope(curvature, lanes, step, max_slope=0.06, k_full=1 / 40, min_slope=0.02, smooth_m=15.0):
    k = np.asarray(curvature, float); lanes = np.asarray(lanes)
    se = -np.sign(k) * np.minimum(max_slope, max_slope * np.abs(k) / k_full)
    se = gaussian_filter1d(se, smooth_m / step, mode="nearest")
    two = lanes >= 2
    crown = np.where(two, CROWN_SLOPE * np.clip(1 - np.abs(se) / (2 * min_slope), 0, 1), 0.0)
    small = ~two & (np.abs(se) < min_slope)
    se = np.where(small, np.where(se > 0, 1.0, -1.0) * min_slope, se)
    se = np.where(two, se, gaussian_filter1d(se, 5.0 / step, mode="nearest"))
    return se, gaussian_filter1d(crown, 5.0 / step, mode="nearest")


@dataclass
class Section:
    s: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    heading: np.ndarray
    curvature: np.ndarray
    lanes: np.ndarray
    bridge: np.ndarray
    hw_left: np.ndarray
    hw_right: np.ndarray
    slope: np.ndarray
    crown: np.ndarray

    @property
    def n(self):
        return len(self.s)

    def edge_z(self, side):
        hw = self.hw_left if side > 0 else self.hw_right
        return surface_height(side * hw, self.z, self.slope, self.crown)

    def path(self):
        return {"s": self.s, "x": self.x, "y": self.y, "z": self.z, "heading": self.heading}

    def save_json(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({f.name: np.asarray(getattr(self, f.name)).tolist() for f in fields(self)}),
                              encoding="utf-8")

    @classmethod
    def load_json(cls, path):
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        kinds = {"lanes": np.int32, "bridge": bool}
        return cls(**{k: np.asarray(v, kinds.get(k, np.float64)) for k, v in d.items()})


def build_section(cl, z, width_rows):
    w, lanes = widths(cl.s, cl.lanes, width_rows)
    slope, crown = cross_slope(cl.curvature, lanes, cl.step)
    return Section(cl.s, cl.x, cl.y, np.asarray(z, float), cl.heading, cl.curvature, lanes, cl.bridge,
                   w / 2, w / 2, slope, crown)


def pads_from_parking(parking, sec, project=geo.to_local, max_dist=150.0):
    """道のどこかから max_dist 以内にある駐車場を、道の高さの平らな場所にする（道の途中の駐車場も含む）。"""
    ends = {"start": np.array([sec.x[0], sec.y[0]]), "end": np.array([sec.x[-1], sec.y[-1]])}
    out = []
    for p in parking:
        x, y = project(np.asarray(p["lat"], float), np.asarray(p["lon"], float))
        poly = Polygon(np.c_[x, y]).buffer(0)
        c = np.array(poly.centroid.coords[0])
        d_road = np.hypot(sec.x - c[0], sec.y - c[1])
        if d_road.min() > max_dist:
            continue
        role, d = min(((r, np.linalg.norm(c - e)) for r, e in ends.items()), key=lambda t: t[1])
        role = role if d <= max_dist else "mid"
        i = int(np.argmin(d_road))
        out.append({"name": p["name"], "role": role, "polygon": np.asarray(poly.exterior.coords).tolist(), "z": float(sec.z[i])})
    return out
