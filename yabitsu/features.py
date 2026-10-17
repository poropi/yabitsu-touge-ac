import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import geo
from .crosssection import GUTTER_WIDTH
from .meshgen import ribbon, wall_ribbon
from .terrain import CUT_SLOPE

WALL_LEAN = 0.3
WALL_CUT_SLOPE = 3.0
GUARDRAIL_OFFSET = 0.1
SIDES = {"L": 1, "R": -1}
TYPES = ("guardrail", "retaining_wall", "parapet", "fence", "sidewalk", "hedge", "overpass")
PARAPET_HEIGHT = 0.9
FENCE_HEIGHT = 1.1  # 歩道の白い柵
CURB_HEIGHT = 0.15
HEDGE_DEPTH = 0.6
DECK_THICKNESS, PIER_GAP, PIER_DEPTH = 1.5, 0.5, 1.2  # 道の上を渡る橋：床の厚さ、側溝の外から柱までのすき間、柱の厚さ


@dataclass
class Feature:
    s_from: float
    s_to: float
    side: str
    type: str
    height_m: float
    note: str
    width_m: float = 0.0  # 歩道の幅、道の上を渡る橋が道の外へ張り出す長さ


def load_features_rows(rows):
    """表の行（区間の表を通しの距離へずらしたもの、または CSV をそのまま読んだもの）を Feature に。"""
    num = lambda v: float(v) if v not in (None, "") else 0.0
    return [Feature(float(r["s_from"]), float(r["s_to"]), str(r["side"]).strip(), str(r["type"]).strip(),
                    num(r.get("height_m")), r.get("note") or "", num(r.get("width_m")))
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


def _side_arrays(sec, f, offset=None):
    """offset（行ごと）があれば、道の端をそれだけ外へずらす（歩道の外側に立つ壁など）。"""
    sl = rows_of(sec, f); side = SIDES[f.side]
    hw = (sec.hw_left if side > 0 else sec.hw_right)[sl]
    if offset is not None:
        hw = hw + np.asarray(offset, float)[sl]
    return sl, side, hw, sec.edge_z(side)[sl]


def sidewalk_offsets(sec, feats):
    """左右それぞれ、行ごとの歩道の幅（歩道のない所は 0）。壁・生け垣・欄干はこの外側に立てる。"""
    off = {1: np.zeros(sec.n), -1: np.zeros(sec.n)}
    for f in feats:
        if f.type == "sidewalk":
            sl = rows_of(sec, f)
            off[SIDES[f.side]][sl] = np.maximum(off[SIDES[f.side]][sl], f.width_m)
    return off[1], off[-1]


def guardrail_meshes(sec, f):
    sl, side, hw, ze = _side_arrays(sec, f)
    L = hw + GUTTER_WIDTH + GUARDRAIL_OFFSET
    visual = wall_ribbon(sec, sl, side, L, L, ze + 0.55, ze + 0.85)
    collision = wall_ribbon(sec, sl, side, L, L, ze - 0.3, ze + 1.2)
    return visual, collision


def parapet_mesh(sec, f, offset=None):
    """橋の欄干：側溝の外側（歩道があればその外側）に、まっすぐ立つ壁（高さが空なら PARAPET_HEIGHT）。"""
    sl, side, hw, ze = _side_arrays(sec, f, offset)
    L = hw + GUTTER_WIDTH
    return wall_ribbon(sec, sl, side, L, L, ze - 0.3, ze + (f.height_m or PARAPET_HEIGHT))


def _side_ribbon(sec, sl, side, offs, dz, ze, offset=None):
    """側溝の外側の縁からの横の位置 offs・縁の高さからの差 dz の列でできた帯（道の側を表にする）。"""
    hw = (sec.hw_left if side > 0 else sec.hw_right)[sl][:, None]
    if offset is not None:
        hw = hw + np.asarray(offset, float)[sl][:, None]
    lat = side * (hw + GUTTER_WIDTH + np.asarray(offs, float)[None, :])
    z = ze[:, None] + np.asarray(dz, float)[None, :]
    if side < 0:
        lat, z = lat[:, ::-1], z[:, ::-1]
    return ribbon(sec.x[sl], sec.y[sl], sec.heading[sl], lat, z, sec.s[sl], u_scale=1.0)


def sidewalk_mesh(sec, f):
    """歩道：側溝の外側から幅 f.width_m、上面は道の縁＋CURB_HEIGHT。内側の縁に縦の縁石の面（乗り上げられる段差）。"""
    sl, side, hw, ze = _side_arrays(sec, f)
    # (上面＝走る面, 縦の面＝縁石と、地形とのすき間を隠すため下へ垂らした外の縁)
    top = _side_ribbon(sec, sl, side, [0.0, f.width_m], [CURB_HEIGHT, CURB_HEIGHT], ze)
    curb = _side_ribbon(sec, sl, side, [0.0, 0.0], [0.0, CURB_HEIGHT], ze)
    skirt = _side_ribbon(sec, sl, side, [f.width_m, f.width_m], [CURB_HEIGHT, -0.5], ze)
    from .meshgen import compact
    v = np.concatenate([curb[0], skirt[0]]); fc = np.concatenate([curb[1], skirt[1] + len(curb[0])])
    return top, compact(v, fc, np.concatenate([curb[2], skirt[2]]))


def hedge_mesh(sec, f, offset=None):
    """生け垣：側溝の外側（歩道があればその外側）に、高さ f.height_m・厚さ HEDGE_DEPTH の箱。"""
    sl, side, hw, ze = _side_arrays(sec, f)
    h = f.height_m
    return _side_ribbon(sec, sl, side, [0.0, 0.0, HEDGE_DEPTH, HEDGE_DEPTH], [-0.3, h, h, -0.3], ze, offset)


def overpass_meshes(sec, f):
    """道の上を渡る橋：s_from〜s_to の幅の床（下面が道の縁の高い方＋f.height_m）を、道の外へ f.width_m 張り出させる。
    柱は両側、側溝の外側から PIER_GAP 離して立てる。(床, 柱) を返す。床は見た目だけ、柱は当たり判定に使う。"""
    sl = rows_of(sec, f)
    s = sec.s[sl]; x, y, h = sec.x[sl], sec.y[sl], sec.heading[sl]
    ze = np.maximum(sec.edge_z(1)[sl], sec.edge_z(-1)[sl])
    z0 = float(ze.max()) + f.height_m
    reach_l = sec.hw_left[sl] + GUTTER_WIDTH + f.width_m; reach_r = sec.hw_right[sl] + GUTTER_WIDTH + f.width_m
    n = len(s)
    # 床：下面・上面・両端の面を1本の帯で（横に -r, -r, l, l, -r と回る）
    lat = np.stack([-reach_r, -reach_r, reach_l, reach_l, -reach_r], 1)
    z = np.tile([z0, z0 + DECK_THICKNESS, z0 + DECK_THICKNESS, z0, z0], (n, 1))
    deck = ribbon(x, y, h, lat, z, s, u_scale=2.0)
    piers = []
    for side in (1, -1):
        hw = (sec.hw_left if side > 0 else sec.hw_right)[sl][:, None] + GUTTER_WIDTH + PIER_GAP
        offs = np.array([0.0, 0.0, PIER_DEPTH, PIER_DEPTH])[None, :]
        lt = side * (hw + offs)
        zz = np.broadcast_to(np.array([[float(ze.min()) - 0.5, z0, z0, float(ze.min()) - 0.5]]), lt.shape)
        if side < 0:
            lt, zz = lt[:, ::-1], zz[:, ::-1]
        piers.append(ribbon(x, y, h, lt, zz, s, u_scale=2.0))
    from .meshgen import compact
    v = np.concatenate([p[0] for p in piers]); fc = np.concatenate([piers[0][1], piers[1][1] + len(piers[0][0])])
    uv = np.concatenate([p[2] for p in piers])
    return deck, compact(v, fc, uv)


def retaining_wall_mesh(sec, f, offset=None):
    sl, side, hw, ze = _side_arrays(sec, f, offset)
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
