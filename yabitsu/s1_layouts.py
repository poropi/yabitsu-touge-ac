import math

import numpy as np
from shapely.geometry import Point, Polygon

from . import acdata
from .acobjects import frame_at, hotlap_start, pit_boxes, start_grid, timing_gate

LAYOUTS = {"s1_up": 1, "s1_down": -1}


def _s_from_start(sec, direction, d):
    return d if direction > 0 else sec.s[-1] - d


def _pits(sec, pads, direction):
    role = "start" if direction > 0 else "end"
    for p in pads:
        if p["role"] != role:
            continue
        poly = Polygon(p["polygon"]); c = poly.centroid
        i = int(np.argmin(np.hypot(sec.x - c.x, sec.y - c.y)))
        h = sec.heading[i] if direction > 0 else sec.heading[i] + math.pi
        boxes = pit_boxes((c.x, c.y), p["z"], math.atan2(math.sin(h), math.cos(h)))
        if all(poly.contains(Point(b.location[0], b.location[1])) for b in boxes):
            return boxes
    # 駐車場が無い・入らないときは、スタート列と同じ場所をピットにする
    # （ピットは練習走行、スタート列はレースで使うので、同時には使われない）
    out = start_grid(sec.path(), _s_from_start(sec, direction, 40.0), direction)
    for i, d in enumerate(out):
        d.name = f"AC_PIT_{i}"
    return out


def layout_objects(sec, pads, direction):
    path = sec.path()
    hw = float(max(sec.hw_left.max(), sec.hw_right.max()))
    return [*start_grid(path, _s_from_start(sec, direction, 40.0), direction),
            hotlap_start(path, _s_from_start(sec, direction, 5.0), direction),
            *timing_gate(path, _s_from_start(sec, direction, 50.0), direction, hw + 1.0, "START"),
            *timing_gate(path, _s_from_start(sec, direction, sec.s[-1] - 30.0), direction, hw + 1.0, "FINISH"),
            *_pits(sec, pads, direction)]


def cameras(sec, direction, every=150.0):
    L = float(sec.s[-1]); path = sec.path(); cams = []
    marks = np.arange(0.0, L, every)
    for k, d in enumerate(marks):
        s = _s_from_start(sec, direction, d)
        x, y, z, h = frame_at(path, s)
        side = 1 if k % 2 == 0 else -1
        off = float((sec.hw_left if side > 0 else sec.hw_right)[int(np.argmin(np.abs(sec.s - s)))]) + 4.0
        cx, cy, cz = x - side * off * math.sin(h), y + side * off * math.cos(h), z + 3.0
        tx, ty, tz, _ = frame_at(path, float(np.clip(s + direction * 20.0, 0, L)))
        pos = acdata.blender_to_ac(cx, cy, cz)
        fwd = acdata.blender_to_ac(tx - cx, ty - cy, tz - cz)
        cams.append(acdata.Camera(f"cam{k:02d}", pos, fwd, max(0.0, (d - every / 2) / L) if k else 0.0,
                                  1.0 if k == len(marks) - 1 else min(1.0, (d + every / 2) / L)))
    return cams
