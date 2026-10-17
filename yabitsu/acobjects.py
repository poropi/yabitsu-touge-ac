import math
from dataclasses import dataclass

import numpy as np

# Task 2 の試験で確かめる値。結果は docs/ac-conventions.md に記録する。
DUMMY_YAW_OFFSET = math.pi / 2  # 空オブジェクトの -Y（Blender）が車の前方になる、という仮定
DUMMY_Z_OFFSET = 0.3            # 路面からの高さ（m）


@dataclass
class Dummy:
    name: str
    location: tuple
    heading: float  # 進行方向（ラジアン、+X から反時計回り）

    @property
    def rotation_z(self) -> float:
        return self.heading + DUMMY_YAW_OFFSET


def frame_at(path, s):
    S = path["s"]
    x = float(np.interp(s, S, path["x"])); y = float(np.interp(s, S, path["y"])); z = float(np.interp(s, S, path["z"]))
    c = np.interp(s, S, np.cos(path["heading"])); sn = np.interp(s, S, np.sin(path["heading"]))
    return x, y, z, math.atan2(sn, c)


def _place(path, s, direction, lateral):
    x, y, z, h = frame_at(path, s)
    hd = h if direction > 0 else h + math.pi
    nx, ny = -math.sin(hd), math.cos(hd)
    return (x + lateral * nx, y + lateral * ny, z + DUMMY_Z_OFFSET), math.atan2(math.sin(hd), math.cos(hd))


def start_grid(path, s0, direction, n=4, spacing=8.0, stagger=1.2):
    out = []
    for i in range(n):
        loc, hd = _place(path, s0 - direction * i * spacing, direction, stagger if i % 2 == 0 else -stagger)
        out.append(Dummy(f"AC_START_{i}", loc, hd))
    return out


def hotlap_start(path, s, direction):
    loc, hd = _place(path, s, direction, 0.0)
    return Dummy("AC_HOTLAP_START_0", loc, hd)


def timing_gate(path, s, direction, half_width, kind):
    out = []
    for side, lat in (("L", half_width), ("R", -half_width)):
        loc, hd = _place(path, s, direction, lat)
        out.append(Dummy(f"AC_OPEN_{kind}_{side}", loc, hd))
    return out


def pit_boxes(center_xy, z, heading, n=4, spacing=4.0):
    nx, ny = -math.sin(heading), math.cos(heading)
    out = []
    for i in range(n):
        lat = (n - 1) / 2 * spacing - i * spacing
        out.append(Dummy(f"AC_PIT_{i}", (center_xy[0] + lat * nx, center_xy[1] + lat * ny, z + DUMMY_Z_OFFSET), heading))
    return out
