"""町の範囲（と道ぞい）の物：工事中の柵・カラーコーン・「工事中」の看板、見えない壁、信号機、バス停などの小物。
どれも (名前, 頂点, 面, UV, 材質, 描画するか) の並びで返す。高さは z_fn(x, y)（町の地面など）から。"""
import math

import numpy as np
from shapely.geometry import LineString

from . import geo

WALL_H = 3.0
BOARD_BANDS = ((0.35, 0.60), (0.70, 0.95))   # 柵の横板（下・上）の高さ
PANEL_W = 1.8
CONE_EVERY, CONE_H, CONE_R, CONE_FRONT = 1.5, 0.7, 0.18, 0.6
SIGN_W, SIGN_H, SIGN_Z = 1.2, 0.9, 1.15
POLE_R = 0.13
HEAD = {"vehicle": (1.25, 0.42, 0.3), "pedestrian": (0.45, 0.8, 0.3)}   # 灯器の幅・高さ・奥行き


def _heading(deg):
    """方位（北=0、東=90）→ 単位ベクトル (x=東, y=北)。"""
    r = math.radians(deg)
    return np.array([math.sin(r), math.cos(r)])


def _merge(parts):
    vs, fs, us, base = [], [], [], 0
    for v, f, uv in parts:
        vs.append(np.asarray(v, float)); fs.append(np.asarray(f) + base); us.append(np.asarray(uv, float)); base += len(v)
    return np.concatenate(vs), np.concatenate(fs).astype(np.int32), np.concatenate(us)


def _quad(p0, p1, z0, z1, u0=0.0, u1=1.0, both=False):
    """p0→p1 の上に立つ縦の四角（表は p0→p1 の右側）。"""
    v = np.array([[*p0, z0], [*p1, z0], [*p1, z1], [*p0, z1]], float)
    uv = np.array([[u0, 0.0], [u1, 0.0], [u1, 1.0], [u0, 1.0]])
    f = [[0, 1, 2, 3]] + ([[3, 2, 1, 0]] if both else [])
    return v, np.array(f), uv


def _box(c, d, w, depth, z0, z1):
    """中心 c、向き d（奥行きの向き）、幅 w、奥行き depth の箱（上面つき）。"""
    d = np.asarray(d, float) / np.linalg.norm(d); n = np.array([-d[1], d[0]])
    p = [c - n * w / 2 - d * depth / 2, c + n * w / 2 - d * depth / 2, c + n * w / 2 + d * depth / 2, c - n * w / 2 + d * depth / 2]
    v = np.array([[*q, z0] for q in p] + [[*q, z1] for q in p], float)
    f = np.array([[0, 3, 7, 4], [3, 2, 6, 7], [2, 1, 5, 6], [1, 0, 4, 5], [4, 7, 6, 5]])
    return v, f, v[:, :2] / 2.0


def _prism(c, r, z0, z1, n=8):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = np.c_[c[0] + r * np.cos(a), c[1] + r * np.sin(a)]
    v = np.array([[*q, z0] for q in ring] + [[*q, z1] for q in ring], float)
    f = np.array([[k, (k + 1) % n, n + (k + 1) % n, n + k] for k in range(n)])
    return v, f, np.c_[np.arange(2 * n) % n / n, np.repeat([0.0, 1.0], n)]


def _cone(c, z):
    a = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    ring = np.c_[c[0] + CONE_R * np.cos(a), c[1] + CONE_R * np.sin(a)]
    apex = [c[0], c[1], z + CONE_H]
    v = np.array([[*q, z] for q in ring] + [apex] * 8, float)
    f = np.array([[k, (k + 1) % 8, 8 + (k + 1) % 8, 8 + k] for k in range(8)])  # 頂点は同じ所に重ねた四角（三角）
    uv = np.c_[np.r_[a / (2 * np.pi), a / (2 * np.pi)], np.r_[np.zeros(8), np.ones(8)]]
    base = _box(c, (1, 0), 0.38, 0.38, z, z + 0.04)
    return _merge([(v, f, uv), base])


def barricade_meshes(a, b, z_fn, front, k=0):
    """封鎖の線 a→b に、工事中の柵（横板2枚・脚）、柵の手前（front の側）にカラーコーン、真ん中に「工事中」の看板。"""
    a, b = np.asarray(a, float), np.asarray(b, float)
    L = float(np.linalg.norm(b - a)); d = (b - a) / max(L, 1e-9)
    front = np.asarray(front, float) / max(np.linalg.norm(front), 1e-9)
    z = lambda p: float(z_fn(np.atleast_1d(p[0]), np.atleast_1d(p[1]))[0])
    boards, legs, cones = [], [], []
    n_panels = max(1, int(round(L / PANEL_W)))
    for i in range(n_panels):
        p0 = a + d * L * i / n_panels; p1 = a + d * L * (i + 1) / n_panels
        zz = min(z(p0), z(p1))
        for lo, hi in BOARD_BANDS:
            boards.append(_quad(p0, p1, zz + lo, zz + hi, 0.0, 1.0, both=True))
        for q in (p0 + d * 0.1, p1 - d * 0.1):
            legs.append(_box(q, front, 0.05, 0.5, zz, zz + 1.0))
    for t in np.arange(CONE_EVERY / 2, L, CONE_EVERY):
        q = a + d * t + front * CONE_FRONT
        cones.append(_cone(q, z(q)))
    mid = a + d * L / 2 + front * 0.05
    zm = z(mid)
    n = np.array([-front[1], front[0]])
    sign = _quad(mid - n * SIGN_W / 2, mid + n * SIGN_W / 2, zm + SIGN_Z, zm + SIGN_Z + SIGN_H)   # 表は front の側
    back = _quad(mid + n * SIGN_W / 2, mid - n * SIGN_W / 2, zm + SIGN_Z, zm + SIGN_Z + SIGN_H)
    stand = [_box(mid + s * n * (SIGN_W / 2 - 0.05) - front * 0.03, front, 0.04, 0.04, zm, zm + SIGN_Z) for s in (1, -1)]
    out = [(f"barricade_{k:03d}", *_merge(boards), "barricade", True),
           (f"barricade_{k:03d}_legs", *_merge(legs + stand + [back]), "pole_grey", True),
           (f"sign_koji_{k:03d}", *sign, "sign_koji", True)]
    if cones:
        out.append((f"cone_{k:03d}", *_merge(cones), "cone", True))
    return out


def wall_meshes(lines, z_fn, height=WALL_H, prefix="1WALL_town_block", step=2.0):
    """見えない壁（両面、当たり判定だけ）。線ごとに地面の高さに沿わせる。"""
    parts = []
    for ln in lines:
        p = np.asarray(LineString(ln).segmentize(step).coords)
        zg = np.asarray(z_fn(p[:, 0], p[:, 1]), float)
        for i in range(len(p) - 1):
            v = np.array([[*p[i], zg[i] - 0.5], [*p[i + 1], zg[i + 1] - 0.5], [*p[i + 1], zg[i + 1] + height], [*p[i], zg[i] + height]])
            parts.append((v, np.array([[0, 1, 2, 3], [3, 2, 1, 0]]), v[:, :2] / 2.0))
    out, cur, nv = [], [], 0
    for q in parts:
        cur.append(q); nv += 4
        if nv >= 60000:
            out.append(cur); cur, nv = [], 0
    if cur:
        out.append(cur)
    return [(f"{prefix}_{k:03d}", *_merge(c), "concrete", False) for k, c in enumerate(out)]


def _xy(ll):
    x, y = geo.to_local(float(ll[0]), float(ll[1]))
    return np.array([float(x), float(y)])


def signal_meshes(signals, z_fn, prefix="town_signal"):
    """signals = [{"at": [lat, lon], "facing_deg": 灯器の表の向き, "kind": vehicle | pedestrian, "arm_m", "arm_deg", "height_m"}]。
    柱は当たり判定あり（1WALL）。灯器の表の面に信号の絵（車用は青、歩行者用は赤が点いた絵）。"""
    poles, bodies, faces = [], [], {"vehicle": [], "pedestrian": []}
    for s in signals:
        p = _xy(s["at"]); z0 = float(z_fn(np.atleast_1d(p[0]), np.atleast_1d(p[1]))[0])
        kind = s.get("kind", "vehicle"); h = float(s.get("height_m", 5.5 if kind == "vehicle" else 2.6))
        face = _heading(float(s["facing_deg"]))
        poles.append(_prism(p, POLE_R, z0 - 0.3, z0 + h + 0.3))
        w, hh, dep = HEAD[kind]
        if kind == "vehicle":
            arm = _heading(float(s.get("arm_deg", s["facing_deg"] + 90.0)))
            L = float(s.get("arm_m", 3.0))
            poles.append(_box(p + arm * L / 2, arm, 0.1, L, z0 + h, z0 + h + 0.12))
            c = p + arm * L
            zc = z0 + h - 0.2
        else:
            c = p + face * (POLE_R + dep / 2)
            zc = z0 + h - hh / 2
        bodies.append(_box(c, face, w, dep, zc - hh / 2, zc + hh / 2))
        n = np.array([-face[1], face[0]])
        fc = c + face * (dep / 2 + 0.005)
        faces[kind].append(_quad(fc - n * w / 2, fc + n * w / 2, zc - hh / 2, zc + hh / 2))   # 表は face の側
    out = []
    if poles:
        out.append((f"1WALL_{prefix}_poles", *_merge(poles), "pole_grey", True))
        out.append((f"{prefix}_bodies", *_merge(bodies), "pole_grey", True))
    if faces["vehicle"]:
        out.append((f"{prefix}_heads", *_merge(faces["vehicle"]), "signal_head", True))
    if faces["pedestrian"]:
        out.append((f"{prefix}_peds", *_merge(faces["pedestrian"]), "signal_ped", True))
    return out


def object_meshes(objects, z_fn, prefix="town_obj"):
    """objects = [{"kind": bus_stop | guard_pipe | stele | pole | torii | lantern, "points": [[lat, lon], ...], ...}]"""
    groups = {}
    add = lambda mat, part: groups.setdefault(mat, []).append(part)
    zf = lambda p: float(z_fn(np.atleast_1d(p[0]), np.atleast_1d(p[1]))[0])
    for ob in objects:
        pts = [_xy(q) for q in ob["points"]]
        kind = ob["kind"]
        if kind == "bus_stop":
            p = pts[0]; z = zf(p); face = _heading(float(ob.get("facing_deg", 0.0)))
            add("pole_grey", _prism(p, 0.04, z, z + 2.2))
            n = np.array([-face[1], face[0]]); c = p + face * 0.05
            add("sign_bus", _quad(c - n * 0.25, c + n * 0.25, z + 1.7, z + 2.2))
            add("sign_bus", _quad(c + n * 0.25 - face * 0.1, c - n * 0.25 - face * 0.1, z + 1.7, z + 2.2))
            add("stone", _box(p, face, 0.4, 0.4, z, z + 0.15))
        elif kind == "guard_pipe":   # 歩道の白いガードパイプ（高さ 0.8m、2段の横棒＋支柱 2m おき）
            ln = LineString(pts).segmentize(2.0); q = np.asarray(ln.coords)
            for i in range(len(q) - 1):
                za, zb = zf(q[i]), zf(q[i + 1])
                for hgt in (0.45, 0.8):
                    add("railing", _quad(q[i], q[i + 1], min(za, zb) + hgt - 0.04, min(za, zb) + hgt + 0.04, both=True))
                add("railing", _prism(q[i], 0.035, za, za + 0.8, 6))
        elif kind == "stele":
            p = pts[0]; z = zf(p); w, dpt, h = ob.get("size", [0.6, 0.3, 1.5])
            add("stone", _box(p, _heading(float(ob.get("facing_deg", 0.0))), w, dpt, z - 0.2, z + h))
        elif kind == "pole":         # 電柱（電線は張らない）
            for p in pts:
                z = zf(p); add("concrete", _prism(p, 0.17, z - 0.5, z + float(ob.get("height_m", 10.0))))
        elif kind == "torii":        # 石の鳥居：柱2本と笠木・貫
            p = pts[0]; z = zf(p); face = _heading(float(ob.get("facing_deg", 0.0))); n = np.array([-face[1], face[0]])
            span, h = float(ob.get("span_m", 2.6)), float(ob.get("height_m", 3.6))
            for s in (1, -1):
                add("stone", _prism(p + n * s * span / 2, 0.16, z - 0.2, z + h))
            add("stone", _box(p, face, span + 1.0, 0.35, z + h, z + h + 0.35))
            add("stone", _box(p, face, span + 0.3, 0.2, z + h - 0.7, z + h - 0.45))
        elif kind == "lantern":      # 石灯籠
            p = pts[0]; z = zf(p)
            add("stone", _box(p, (1, 0), 0.6, 0.6, z, z + 0.6))
            add("stone", _prism(p, 0.14, z + 0.6, z + 1.5))
            add("stone", _box(p, (1, 0), 0.5, 0.5, z + 1.5, z + 1.9))
            add("stone", _box(p, (1, 0), 0.8, 0.8, z + 1.9, z + 2.15))
        else:
            raise ValueError(f"物の kind が分かりません: {ob}")
    return [(f"{prefix}_{mat}", *_merge(parts), mat, True) for mat, parts in groups.items()]
