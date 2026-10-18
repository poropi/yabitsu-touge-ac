"""レイアウト（どこからスタートしてどこでゴールするか）。data/manual/layouts.json に緯度経度で書く。
区間を足して通しの距離が変わっても、地点は動かない。"""
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from shapely.geometry import Point, Polygon

from . import acdata, config, geo
from .acobjects import DUMMY_Z_OFFSET, Dummy, frame_at, hotlap_start, pit_boxes, start_grid, timing_gate

SECTION_SLACK_M = 5.0  # 区間の境目の外側でも、これだけは探してよい（つなぎ目ちょうどの地点のため）
OFF_ROAD_M = 4.0       # 道の端からこれ以上離れた地点は、別の枝か打ち間違いとして止める
GATE_HALF_WIDTH_M = 4.5  # 計測ゲートの半分の幅の最小（S1・S2 を作ったときの値）。ゲートの所の道が広ければ 道幅/2＋1m


@dataclass(frozen=True)
class Layout:
    name: str
    title: str
    section: str
    direction: int
    grid: tuple
    hotlap: tuple
    timing_start: tuple
    finish: tuple
    pits: str


def load_layouts(path=None):
    rows = json.loads(Path(path or config.DATA_MANUAL / "layouts.json").read_text(encoding="utf-8"))
    return [Layout(r["name"], r["title"], r["section"], int(r["direction"]), tuple(r["grid"]), tuple(r["hotlap"]),
                   tuple(r["timing_start"]), tuple(r["finish"]), r["pits"]) for r in rows]


def s_of(sec, ll, bounds, section):
    """緯度経度の地点を、指定した区間の中の道の中心線に下ろしたときの s（サンプルの間も直線で補う）。"""
    x, y = geo.to_local(ll[0], ll[1]); p = np.array([float(x), float(y)])
    s0, s1 = bounds[section]
    idx = np.nonzero((sec.s >= s0 - SECTION_SLACK_M) & (sec.s <= s1 + SECTION_SLACK_M))[0]
    best = (math.inf, 0.0)
    i = idx[int(np.argmin(np.hypot(sec.x[idx] - p[0], sec.y[idx] - p[1])))]
    for j in (i - 1, i):
        if j < 0 or j + 1 >= sec.n:
            continue
        a = np.array([sec.x[j], sec.y[j]]); b = np.array([sec.x[j + 1], sec.y[j + 1]])
        t = float(np.clip(np.dot(p - a, b - a) / max(np.dot(b - a, b - a), 1e-12), 0.0, 1.0))
        d = float(np.linalg.norm(a + t * (b - a) - p))
        if d < best[0]:
            best = (d, float(sec.s[j] + t * (sec.s[j + 1] - sec.s[j])))
    hw = float(max(sec.hw_left[i], sec.hw_right[i]))
    if best[0] > hw + OFF_ROAD_M:
        raise ValueError(f"地点 {ll} は区間 {section} の道から {best[0]:.1f}m 離れています（別の枝か、地点の間違い）")
    return best[1]


def point_s(sec, pt, bounds, default_section):
    """レイアウトの地点（[lat, lon] か [lat, lon, 区間]）の s。区間が書いてあればその区間で、なければ default_section で探す。"""
    section = pt[2] if len(pt) > 2 else default_section
    return s_of(sec, (float(pt[0]), float(pt[1])), bounds, section)


def _pits(sec, pads, layout):
    for p in pads:
        if p["name"] != layout.pits:
            continue
        if "pit_points" in p:  # 場所と向きを1台ずつ決めてあるピット（峠のわき道など）
            return [Dummy(f"AC_PIT_{i}", (x, y, z + DUMMY_Z_OFFSET), h) for i, (x, y, z, h) in enumerate(p["pit_points"])]
        if "pit_center" in p:  # 手で測った駐車場：ピットの場所と向き（出口向き）も決めてある
            return pit_boxes(p["pit_center"], p["z"], p["heading"], spacing=p["pit_spacing"])
        poly = Polygon(p["polygon"]); c = poly.centroid
        i = int(np.argmin(np.hypot(sec.x - c.x, sec.y - c.y)))
        h = sec.heading[i] if layout.direction > 0 else sec.heading[i] + math.pi
        return pit_boxes((c.x, c.y), p["z"], math.atan2(math.sin(h), math.cos(h)))
    raise ValueError(f"レイアウト {layout.name} のピット「{layout.pits}」が見つかりません")


def layout_objects(sec, pads, layout, bounds):
    path = sec.path(); d = layout.direction
    at = lambda pt: point_s(sec, pt, bounds, layout.section)

    def half(s):
        i = int(np.argmin(np.abs(sec.s - s)))
        return max(GATE_HALF_WIDTH_M, float(max(sec.hw_left[i], sec.hw_right[i])) + 1.0)

    s_start, s_finish = at(layout.timing_start), at(layout.finish)
    return [*start_grid(path, at(layout.grid), d),
            hotlap_start(path, at(layout.hotlap), d),
            *timing_gate(path, s_start, d, half(s_start), "START"),
            *timing_gate(path, s_finish, d, half(s_finish), "FINISH"),
            *_pits(sec, pads, layout)]


def timed_length(sec, layout, bounds):
    return abs(point_s(sec, layout.finish, bounds, layout.section) - point_s(sec, layout.timing_start, bounds, layout.section))


def cameras(sec, layout, every=150.0):
    """リプレイ用カメラ。AI の線と同じく道全体を、レイアウトの向きにたどって置く。"""
    L = float(sec.s[-1]); path = sec.path(); cams = []; direction = layout.direction
    marks = np.arange(0.0, L, every)
    for k, dist in enumerate(marks):
        s = dist if direction > 0 else L - dist
        x, y, z, h = frame_at(path, s)
        side = 1 if k % 2 == 0 else -1
        off = float((sec.hw_left if side > 0 else sec.hw_right)[int(np.argmin(np.abs(sec.s - s)))]) + 4.0
        cx, cy, cz = x - side * off * math.sin(h), y + side * off * math.cos(h), z + 3.0
        tx, ty, tz, _ = frame_at(path, float(np.clip(s + direction * 20.0, 0, L)))
        pos = acdata.blender_to_ac(cx, cy, cz)
        fwd = acdata.blender_to_ac(tx - cx, ty - cy, tz - cz)
        cams.append(acdata.Camera(f"cam{k:02d}", pos, fwd, max(0.0, (dist - every / 2) / L) if k else 0.0,
                                  1.0 if k == len(marks) - 1 else min(1.0, (dist + every / 2) / L)))
    return cams


def kn5_groups(layout_names):
    """make_kn5 で書くグループ：本体・名古木の町・S4・S5・宮ヶ瀬北原の町（全部のレイアウトが読む）と、レイアウトごとの置き物。"""
    return shared_groups() + [f"{n}_objects" for n in layout_names]


def shared_groups():
    from . import config
    return [config.TRACK_GROUP, config.TOWN_GROUP, config.S4_GROUP, config.S5_GROUP, config.KITAHARA_GROUP]


def kn5_files(layout_name):
    """models_<レイアウト>.ini に書く kn5 の並び。"""
    return [f"{g}.kn5" for g in shared_groups()] + [f"{layout_name}_objects.kn5"]
