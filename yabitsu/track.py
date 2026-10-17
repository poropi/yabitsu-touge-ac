"""区間（config.SECTIONS）をつないだ1本の道。手作業の表は区間ごとのファイルのまま、読むときに通しの距離へずらす。"""
import csv
from pathlib import Path

import numpy as np

from . import config, geo
from .config import SectionDef  # noqa: F401  （ほかのモジュールは track から使う）
from .osm import Route

JOIN_TOL_M = 1.0
S_KEYS = ("s_from", "s_to", "s_m", "s_center")


def join_routes(routes):
    """区間ごとの Route を1本にする。つなぎ目の重複点は1つにし、つなぎ目の緯度経度の並びも返す。"""
    lat, lon = list(routes[0].lat), list(routes[0].lon)
    lanes, bridge = list(routes[0].seg_lanes), list(routes[0].seg_bridge)
    joints = []
    for r in routes[1:]:
        x0, y0 = geo.to_local(lat[-1], lon[-1]); x1, y1 = geo.to_local(r.lat[0], r.lon[0])
        if np.hypot(float(x1 - x0), float(y1 - y0)) > JOIN_TOL_M:
            raise ValueError(f"区間のつなぎ目が離れています（{np.hypot(float(x1 - x0), float(y1 - y0)):.1f}m）")
        joints.append((r.lat[0], r.lon[0]))
        lat += list(r.lat[1:]); lon += list(r.lon[1:])
        lanes += list(r.seg_lanes); bridge += list(r.seg_bridge)
    return Route(lat, lon, lanes, bridge), joints


def section_bounds(cl, defs, joints):
    """各区間の通しの s の範囲。境目は、つなぎ目に一番近い中心線の点。"""
    cuts = [0.0]
    for lat, lon in joints:
        x, y = geo.to_local(lat, lon)
        cuts.append(float(cl.s[int(np.argmin(np.hypot(cl.x - x, cl.y - y)))]))
    cuts.append(float(cl.s[-1]))
    return {d.name: (cuts[k], cuts[k + 1]) for k, d in enumerate(defs)}


def load_section_rows(kind, bounds, manual_dir=None):
    """{区間}_{kind}.csv を区間の順に読み、距離の列に区間の始点を足す。区間の終わりを越える範囲は切る。"""
    manual_dir = Path(manual_dir or config.DATA_MANUAL)
    out = []
    for name, (s0, s1) in bounds.items():
        p = manual_dir / f"{name}_{kind}.csv"
        if not p.exists():
            continue
        with p.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                first = next(iter(r.values()), "")
                if not first or not str(first).strip():
                    continue
                r = dict(r)
                for k in S_KEYS:
                    if r.get(k) not in (None, ""):
                        r[k] = min(float(r[k]) + s0, s1)
                if r.get("s_from", s0) >= s1 and "s_from" in r:
                    continue
                out.append(r)
    return out
