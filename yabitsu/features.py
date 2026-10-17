import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import geo
from .crosssection import GUTTER_WIDTH
from .meshgen import wall_ribbon
from .terrain import CUT_SLOPE

WALL_LEAN = 0.3
WALL_CUT_SLOPE = 3.0
GUARDRAIL_OFFSET = 0.1
SIDES = {"L": 1, "R": -1}
TYPES = ("guardrail", "retaining_wall")


@dataclass
class Feature:
    s_from: float
    s_to: float
    side: str
    type: str
    height_m: float
    note: str


def load_features_rows(rows):
    """表の行（区間の表を通しの距離へずらしたもの、または CSV をそのまま読んだもの）を Feature に。"""
    return [Feature(float(r["s_from"]), float(r["s_to"]), str(r["side"]).strip(), str(r["type"]).strip(),
                    float(r["height_m"]) if r.get("height_m") not in (None, "") else 0.0, r.get("note") or "")
            for r in rows if r.get("s_from") not in (None, "")]


def load_features_csv(path):
    p = Path(path)
    if not p.exists():
        return []
    with p.open(encoding="utf-8") as f:
        return load_features_rows(list(csv.DictReader(f)))


def rows_of(sec, f):
    if f.side not in SIDES or f.type not in TYPES:
        raise ValueError(f"side は L/R、type は {TYPES} のどれか: {f}")
    if not (0 <= f.s_from < f.s_to <= sec.s[-1] + 1e-6):
        raise ValueError(f"範囲が区間の外か逆順です（0〜{sec.s[-1]:.0f}m）: {f}")
    i0 = int(np.searchsorted(sec.s, f.s_from)); i1 = int(np.searchsorted(sec.s, f.s_to, side="right")) - 1
    return slice(i0, i1 + 1)


def cut_slopes(sec, feats):
    cut = {1: np.full(sec.n, CUT_SLOPE), -1: np.full(sec.n, CUT_SLOPE)}
    for f in feats:
        if f.type == "retaining_wall":
            cut[SIDES[f.side]][rows_of(sec, f)] = WALL_CUT_SLOPE
    return cut[1], cut[-1]


def _side_arrays(sec, f):
    sl = rows_of(sec, f); side = SIDES[f.side]
    hw = (sec.hw_left if side > 0 else sec.hw_right)[sl]
    return sl, side, hw, sec.edge_z(side)[sl]


def guardrail_meshes(sec, f):
    sl, side, hw, ze = _side_arrays(sec, f)
    L = hw + GUTTER_WIDTH + GUARDRAIL_OFFSET
    visual = wall_ribbon(sec, sl, side, L, L, ze + 0.55, ze + 0.85)
    collision = wall_ribbon(sec, sl, side, L, L, ze - 0.3, ze + 1.2)
    return visual, collision


def retaining_wall_mesh(sec, f):
    sl, side, hw, ze = _side_arrays(sec, f)
    L0 = hw + GUTTER_WIDTH
    return wall_ribbon(sec, sl, side, L0, L0 + WALL_LEAN * f.height_m, ze - 0.3, ze + f.height_m)


def _runs(mask, s, min_len, max_gap):
    out, start, last = [], None, None
    for i, m in enumerate(mask):
        if m:
            if start is None or s[i] - s[last] > max_gap:
                if start is not None and s[last] - s[start] >= min_len:
                    out.append((start, last))
                start = i
            last = i
    if start is not None and s[last] - s[start] >= min_len:
        out.append((start, last))
    return out


def suggest_candidates(sec, z_natural, threshold=2.5, min_len=10.0, max_gap=5.0):
    out = []
    for tag, side in SIDES.items():
        hw = sec.hw_left if side > 0 else sec.hw_right
        L = side * (hw + GUTTER_WIDTH + 4.0)
        x = sec.x - L * np.sin(sec.heading); y = sec.y + L * np.cos(sec.heading)
        diff = np.asarray(z_natural(x, y)) - sec.edge_z(side)
        for kind, mask in (("retaining_wall", diff > threshold), ("guardrail", diff < -threshold)):
            for a, b in _runs(mask, sec.s, min_len, max_gap):
                mid = (a + b) // 2
                lat, lon = geo.from_local(sec.x[mid], sec.y[mid])
                h = float(np.clip(np.median(diff[a:b + 1]) - 1.0, 1.0, 6.0)) if kind == "retaining_wall" else 0.0
                s_to = min(round(float(sec.s[b])), int(np.floor(sec.s[-1])))  # 丸めて区間の外に出ないように
                out.append({"s_from": round(float(sec.s[a])), "s_to": s_to, "side": tag, "type": kind,
                            "height_m": round(h, 1),
                            "streetview": f"https://www.google.com/maps/@?api=1&map_action=pano&viewpoint={float(lat):.6f},{float(lon):.6f}"})
    return sorted(out, key=lambda r: r["s_from"])
