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


def _wall_uv(box, a, t, n, s0):
    """_box の箱を面ごとの頂点に分け、壁の UV にする：横 u＝高さ（1 = 2 m）、縦 v＝壁に沿った長さ（1 = 4 m、s0 は壁の始まりからの距離）。
    dress_retex.UV_LAYOUT["wall"] と同じ寸法なので、間知ブロックなどを CSP でそのまま貼れる。上面は u＝奥行き。"""
    v, f, _ = box
    vs, fs, us = [], [], []
    for k, face in enumerate(f):
        p = v[face]
        along = s0 + (p[:, :2] - a) @ t
        top = np.ptp(p[:, 2]) < 1e-9
        us.append(np.c_[((p[:, :2] - a) @ n) / 2.0 if top else p[:, 2] / 2.0, along / 4.0])
        vs.append(p); fs.append(np.arange(4) + 4 * k)
    return np.vstack(vs), np.array(fs), np.vstack(us)


def _slanted(p0, z0, p1, z1, r, n=6):
    """p0（高さ z0）から p1（高さ z1）への細い棒（斜めの杭・横木）。"""
    a0 = np.array([*p0, z0], float); a1 = np.array([*p1, z1], float); ax = a1 - a0; ax /= np.linalg.norm(ax)
    e1 = np.cross(ax, [0.0, 0.0, 1.0] if abs(ax[2]) < 0.9 else [1.0, 0.0, 0.0]); e1 /= np.linalg.norm(e1); e2 = np.cross(ax, e1)
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = r * (np.cos(t)[:, None] * e1 + np.sin(t)[:, None] * e2)
    v = np.r_[a0 + ring, a1 + ring]
    f = np.array([[k, (k + 1) % n, n + (k + 1) % n, n + k] for k in range(n)])
    return v, f, np.c_[np.arange(2 * n) % n / n, np.repeat([0.0, 1.0], n)]


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
    """封鎖の線 a→b に、工事中の柵（横板2枚・脚）、真ん中に「工事中」の看板（front＝走ってくる側を向く）。
    カラーコーンは柵の奥（front の反対側）に置き、道へ飛び出さないようにする。"""
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
        q = a + d * t - front * CONE_FRONT
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


OFF_ROAD_M = 0.5   # 路面の中にあったバス停を、道の縁＋側溝からこれだけ外へ


def off_road_xy(sec, p):
    """点 p（ローカル座標）が本線の路面＋側溝の中なら、同じ s のまま道の外（縁＋側溝＋OFF_ROAD_M）へずらした点。外ならそのまま。"""
    from .crosssection import GUTTER_WIDTH
    p = np.asarray(p, float); i = int(np.argmin(np.hypot(sec.x - p[0], sec.y - p[1])))
    h = sec.heading[i]; n = np.array([-np.sin(h), np.cos(h)]); c = np.array([sec.x[i], sec.y[i]])
    lat = float((p - c) @ n); hw = float(sec.hw_left[i] if lat >= 0 else sec.hw_right[i])
    want = hw + GUTTER_WIDTH + OFF_ROAD_M
    return p + n * (np.sign(lat or 1.0) * want - lat) if abs(lat) < want - 1e-6 else p


def off_road(objects, sec):
    """路面（本線の帯＋側溝）の中に点があるバス停を、同じ s のまま道の外へずらす（off_road_xy）。ほかの物はそのまま。"""
    out = []
    for o in objects:
        if o.get("kind") == "bus_stop":
            p = _xy(o["points"][0]); q = off_road_xy(sec, p)
            if np.any(q != p):
                la, lo = geo.from_local(q[0], q[1])
                o = {**o, "points": [[float(la), float(lo)]] + list(o["points"][1:])}
        out.append(o)
    return out


def road_ground(sec):
    """道ぞいの物の地面の高さ：一番近い道の点の高さ（道の外へ水平に延ばす）。"""
    from scipy.spatial import cKDTree
    tree = cKDTree(np.c_[sec.x, sec.y])
    return lambda x, y: sec.z[tree.query(np.c_[np.ravel(x), np.ravel(y)])[1]].reshape(np.shape(x))


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
    """objects = [{"kind": bus_stop | guard_pipe | stele | pole | torii | lantern | hoarding | mirror | road_sign | bench | tree_stake | vending | bollard | retaining_wall | delineator | sawhorse | cushion_drum, "points": [[lat, lon], ...], ...}]"""
    groups, walls, mat_walls, fence_walls = {}, [], {}, []
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
        elif kind == "guard_pipe":   # 歩道の白いガードパイプ（高さ 0.8m、2段の横棒＋支柱 2m おき）。rails：横棒の高さ、material：色、post_r：支柱の半径
            ln = LineString(pts).segmentize(float(ob.get("post_every_m", 2.0))); q = np.asarray(ln.coords)
            rails = ob.get("rails", [0.45, 0.8]); mat = ob.get("material", "railing"); top = max(rails); pr = float(ob.get("post_r", 0.035))
            for i in range(len(q) - 1):
                za, zb = zf(q[i]), zf(q[i + 1])
                for hgt in rails:
                    add(mat, _quad(q[i], q[i + 1], min(za, zb) + hgt - 0.03, min(za, zb) + hgt + 0.03, both=True))
            for p in q:
                z = zf(p); add(mat, _prism(p, pr, z - 0.1, z + top + 0.04, 8))
        elif kind == "stele":
            p = pts[0]; z = zf(p); w, dpt, h = ob.get("size", [0.6, 0.3, 1.5])
            if ob.get("shape") == "boulder":   # 自然石の碑：幅 w・厚さ dpt の楕円の 10 角を上へすぼめ、頭を斜めに（石ごとにばらつかせる）
                fd = _heading(float(ob.get("facing_deg", 0.0))); sd = np.array([fd[1], -fd[0]]); rng = np.random.default_rng(int(ob.get("seed", 5)))
                a = np.linspace(0, 2 * np.pi, 10, endpoint=False); j = 1.0 + rng.uniform(-0.15, 0.15, (3, 10))
                v = []
                for k, (sc, zz) in enumerate(((1.0, z - 0.2), (0.95, z + 0.55 * h), (0.6, z + 0.9 * h))):
                    tilt = 0.25 * h * np.cos(a) if k == 2 else 0.0                       # 頭は横（sd）の片側へ高い
                    xy = p + (np.cos(a) * w / 2)[:, None] * sd * (sc * j[k])[:, None] + (np.sin(a) * dpt / 2)[:, None] * fd * (sc * j[k])[:, None]
                    v.append(np.c_[xy, np.full(10, zz) + tilt])
                v = np.r_[np.concatenate(v), [[*p, z + 0.95 * h]]]
                fs = [[k * 10 + i, k * 10 + (i + 1) % 10, (k + 1) * 10 + (i + 1) % 10, (k + 1) * 10 + i] for k in range(2) for i in range(10)]
                fs += [[20 + i, 20 + (i + 1) % 10, 30, 30] for i in range(10)]                # 頭（中心へ閉じる）
                uv = np.c_[np.r_[np.tile(np.arange(10) / 10 * 2.0, 3), 0.5], (v[:, 2] - z) / max(h, 0.1)]
                add("stone", (v, np.array(fs), uv))
            else:
                add("stone", _box(p, _heading(float(ob.get("facing_deg", 0.0))), w, dpt, z - 0.2, z + h))
        elif kind == "pole":         # 電柱（電線は張らない）。material で色（pole_white・pole_redwhite など）
            for p in pts:
                z = zf(p); add(ob.get("material", "concrete"), _prism(p, float(ob.get("radius_m", 0.17)), z - 0.5, z + float(ob.get("height_m", 10.0))))
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
        elif kind == "hoarding":     # 工事の白い仮囲い（高さ height_m、両面）
            q = np.asarray(LineString(pts).segmentize(4.0).coords); h = float(ob.get("height_m", 3.0))
            for i in range(len(q) - 1):
                za, zb = zf(q[i]), zf(q[i + 1])
                u0 = float(np.linalg.norm(q[i] - q[0])); u1 = u0 + float(np.linalg.norm(q[i + 1] - q[i]))
                add("hoarding", _quad(q[i], q[i + 1], min(za, zb) - 0.2, min(za, zb) + h, u0 / 2.0, u1 / 2.0, both=True))
        elif kind == "mirror":       # カーブミラー：オレンジの柱（2.9m）と丸い鏡（直径 0.8m、face の側を向く）
            p = pts[0]; z = zf(p); face = _heading(float(ob.get("facing_deg", 0.0))); n = np.array([-face[1], face[0]])
            add("pole_orange", _prism(p, 0.04, z - 0.2, z + 2.9))
            c = p + face * 0.12; a = np.linspace(0, 2 * np.pi, 12, endpoint=False); zc = z + 2.6
            ring = [[*(c + n * 0.4 * np.cos(q)), zc + 0.4 * np.sin(q)] for q in a]
            v = np.array([[*c, zc]] + ring, float); f = [[0, 1 + i, 1 + (i + 1) % 12] for i in range(12)]
            f += [[0, 1 + (i + 1) % 12, 1 + i] for i in range(12)]
            add("mirror", (v, np.array(f, np.int32), (v[:, [0, 2]] - v[0, [0, 2]]) / 0.8 + 0.5))
        elif kind == "road_sign":    # 道路標識：灰色の柱と、柱の上の板（speed30＝丸い速度 30、warning＝黄色のひし形、stop＝止まれ、fire_cistern＝防火水そう。pole＝柱の色の材質）
            p = pts[0]; z = zf(p); face = _heading(float(ob.get("facing_deg", 0.0))); n = np.array([-face[1], face[0]])
            s = float(ob.get("size_m", 0.6)) / 2
            add(ob.get("pole", "pole_grey"), _prism(p, 0.035, z - 0.2, z + 1.8 + 2 * s))
            c = p + face * 0.06
            add(f"sign_{ob.get('sign', 'warning')}", _quad(c - n * s, c + n * s, z + 1.8, z + 1.8 + 2 * s, both=True))
        elif kind == "bench":        # 木のベンチ：座面（幅 1.8m・奥行き 0.4m・高さ 0.42m）と脚2本。points ごとに1脚、facing の側に座る
            face = _heading(float(ob.get("facing_deg", 0.0))); n = np.array([-face[1], face[0]]); w = float(ob.get("width_m", 1.8))
            for p in pts:
                z = zf(p)
                add("wood", _box(p, face, w, 0.4, z + 0.36, z + 0.42))
                for s in (1, -1):
                    add("wood", _box(p + n * s * (w / 2 - 0.15), face, 0.08, 0.36, z - 0.1, z + 0.36))
        elif kind == "tree_stake":   # 若木の竹の支柱：幹のまわりに斜めの杭3本（地面で幹から 0.55m、高さ 1.5m で幹に寄せる）と、高さ 1.1m の横木
            for p in pts:
                z = zf(p); mid = []
                for k in range(3):
                    a = 2 * math.pi * k / 3 + 0.4; u = np.array([math.cos(a), math.sin(a)])
                    base, top = p + u * 0.55, p + u * 0.07
                    add("bamboo", _slanted(base, z - 0.2, top, z + 1.5, 0.022))
                    mid.append(base + (top - base) * (1.3 / 1.7))   # 杭の高さ 1.1m の所
                for k in range(3):
                    add("bamboo", _slanted(mid[k], z + 1.1, mid[(k + 1) % 3], z + 1.1, 0.018))
        elif kind == "vending":      # 自動販売機：白い箱（幅 1.0m・奥行き 0.75m・高さ 1.83m）と正面の見本の窓。facing の側が正面
            d = _heading(float(ob.get("facing_deg", 0.0))); n = np.array([-d[1], d[0]])
            for p in pts:
                z = zf(p); add("vending_red" if ob.get("color") == "red" else "vending", _box(p, d, 1.0, 0.75, z, z + 1.83))
                fc = p + d * 0.38; add("vending_front", _quad(fc - n * 0.45, fc + n * 0.45, z + 0.9, z + 1.75))
        elif kind == "bollard":      # 赤い車止め（高さ 0.8m）。material で色（pole_orange など）
            for p in pts:
                z = zf(p); add(ob.get("material", "post_red"), _prism(p, 0.05, z - 0.2, z + float(ob.get("height_m", 0.8)), 8))
        elif kind == "cone":         # 赤いカラーコーン（工事中の柵と同じ形）。当たり判定なし
            for p in pts:
                add("cone", _cone(p, zf(p)))
        elif kind == "delineator":   # オレンジの車線分離標（直径 8 cm・高さ 0.8 m、白い反射帯2本、黒い台座）。当たり判定なし（倒れて戻る物）
            line = ob.get("spacing_m")   # spacing_m があれば points を線として、その間隔で並べる
            if line:
                ln = LineString(pts); n = max(int(round(ln.length / float(line))), 1)
                pts = [np.array(ln.interpolate(k / n, normalized=True).coords[0]) for k in range(n + 1)]
            for p in pts:
                z = zf(p) + float(ob.get("dz", 0.0))
                add("delineator", _prism(p, 0.04, z, z + 0.8, 8)); add("rubber_black", _prism(p, 0.11, z - 0.02, z + 0.05, 8))
        elif kind == "sawhorse":     # 木の A 型の柵：points の各点に A 形の脚、隣り合う点の間に横木（高さ height_m、既定 0.85 m）
            q = pts; hh = float(ob.get("height_m", 0.85)); beam = ob.get("beam", "wood")   # beam：横木の材質（pole_orange など）
            tops = []
            for i in range(len(q) - 1):
                a, b = q[i], q[i + 1]; u = (b - a) / np.linalg.norm(b - a); n = np.array([-u[1], u[0]])
                za, zb = zf(a), zf(b); top = max(za, zb) + hh
                add(beam, _slanted(a, top, b, top, 0.05, 4))
                for p, z in ((a, za), (b, zb)):
                    for s in (1, -1):
                        add(ob.get("legs", "wood"), _slanted(p + n * s * 0.32 * hh / 0.85, z - 0.05, p, top, 0.035, 4))
                tops += [(a, top), (b, top)]
                if i in ob.get("plates", []):   # 横木の真ん中に下げた灰色の三角の板（反射板。一辺 0.45 m）
                    m = (a + b) / 2 + n * 0.06
                    add("pole_grey", (np.array([[*m, top - 0.47], [*m, top - 0.47], [*(m + u * 0.22), top - 0.08], [*(m - u * 0.22), top - 0.08]]),
                                      np.array([[0, 1, 2, 3], [3, 2, 1, 0]]), np.array([[0.5, 1], [0.5, 1], [1, 0], [0, 0]])))
            if ob.get("rope"):   # 柵の頭をつなぐ白いロープ（間で 0.12 m たるむ）
                for (a, za), (b, zb) in zip(tops[0::2], tops[1::2]):
                    m = (a + b) / 2; zm = (za + zb) / 2 - 0.12
                    add("pole_white", _slanted(a, za, m, zm, 0.008, 4)); add("pole_white", _slanted(m, zm, b, zb, 0.008, 4))
        elif kind == "cabinet":      # 電柱の脇の灰白の機器の箱（w_m×d_m×height_m、足元から z_m 浮かせる）。facing の側が正面
            d = _heading(float(ob.get("facing_deg", 0.0)))
            for p in pts:
                z = zf(p) + float(ob.get("z_m", 0.0))
                add(ob.get("material", "pole_white"), _box(p, d, float(ob.get("w_m", 0.8)), float(ob.get("d_m", 0.4)), z, z + float(ob.get("height_m", 1.4))))
        elif kind == "wood_fence":   # 木の柵：2 m おきの角柱（高さ height_m）と横木 2 本、両端の柱は白。見えない当たり判定の壁つき
            q = np.asarray(LineString(pts).segmentize(2.0).coords); h = float(ob.get("height_m", 1.0))
            for k, p in enumerate(q):
                z = zf(p); end = k in (0, len(q) - 1)
                end = end and ob.get("ends", "white") == "white"   # ends: "wood" で両端も木の柱（丸太の柵）
                add("pole_white" if end else "wood", _box(p, (1, 0), 0.12, 0.12, z - 0.2, z + h + (0.1 if end else 0.0)))
            for i in range(len(q) - 1):
                za, zb = zf(q[i]), zf(q[i + 1])
                for r in (0.45, 0.9):
                    add("wood", _slanted(q[i], za + r * h, q[i + 1], zb + r * h, 0.05, 4))
                fence_walls.append(_quad(q[i], q[i + 1], min(za, zb) - 0.3, max(za, zb) + h + 0.2, both=True))
        elif kind == "pipe_u":       # 車止めの U 字のステンレスのパイプ（幅 width_m・高さ height_m）。facing の向きに並ぶ面
            d = _heading(float(ob.get("facing_deg", 0.0))); n = np.array([-d[1], d[0]])
            w, h = float(ob.get("width_m", 0.6)), float(ob.get("height_m", 0.6))
            for p in pts:
                z = zf(p); a, b = p - n * w / 2, p + n * w / 2
                add("pole_grey", _slanted(a, z - 0.1, a, z + h, 0.03, 6)); add("pole_grey", _slanted(b, z - 0.1, b, z + h, 0.03, 6))
                add("pole_grey", _slanted(a, z + h, b, z + h, 0.03, 6))
        elif kind == "bar_fence":    # 縦格子の鉄の柵（橋の欄干）：上下の横棒と bar_every_m おきの縦の棒、material の色。見えない当たり判定の壁つき
            q = np.asarray(LineString(pts).segmentize(2.0).coords); h = float(ob.get("height_m", 1.1)); mat = ob.get("material", "rubber_black")
            step = float(ob.get("bar_every_m", 0.15)); bw = float(ob.get("bar_w_m", 0.024)) / 2   # bar_w_m：縦の棒の太さ（擬木の欄干は太い）
            for i in range(len(q) - 1):
                za, zb = zf(q[i]), zf(q[i + 1]); u = q[i + 1] - q[i]; L = float(np.linalg.norm(u)); u = u / L; n = np.array([-u[1], u[0]])
                for r in (0.12, h):
                    add(mat, _slanted(q[i], za + r, q[i + 1], zb + r, 0.03, 4))
                for t in np.arange(step / 2, L, step):
                    p = q[i] + u * t; z = za + (zb - za) * t / L
                    add(mat, _quad(p - u * bw, p + u * bw, z + 0.12, z + h, both=True))
                fence_walls.append(_quad(q[i], q[i + 1], min(za, zb) - 0.3, max(za, zb) + h + 0.2, both=True))
            for p in q:
                z = zf(p); add(mat, _box(p, (1, 0), 0.07, 0.07, z - 0.1, z + h + 0.05))
        elif kind == "mesh_fence":   # 金網のフェンス（高さ height_m、2.0 m おきに灰色の支柱）。material：網の絵（既定は緑の縦格子 railing_bars）
            q = np.asarray(LineString(pts).segmentize(2.0).coords); h = float(ob.get("height_m", 1.5))
            for i in range(len(q) - 1):
                za, zb = zf(q[i]), zf(q[i + 1]); L = float(np.linalg.norm(q[i + 1] - q[i]))
                add(ob.get("material", "railing_bars"), _quad(q[i], q[i + 1], min(za, zb), min(za, zb) + h, 0.0, L / 2.0, both=True))
            for p in q:
                z = zf(p); add("pole_grey", _prism(p, 0.03, z - 0.2, z + h, 6))
        elif kind == "cushion_drum":   # 紅白の市松の緩衝ドラム（直径 0.6 m・高さ 0.85 m）
            for p in pts:
                z = zf(p); add("drum_checker", _prism(p, 0.3, z, z + 0.85, 12)); add("drum_checker", _box(p, (1, 0), 0.5, 0.5, z + 0.84, z + 0.86))
        elif kind == "retaining_wall":   # 擁壁（厚さ 0.3m、当たり判定あり）と、上の金網のフェンス（fence_m）
            q = np.asarray(LineString(pts).segmentize(2.0).coords); h = float(ob.get("height_m", 2.5)); fh = float(ob.get("fence_m", 0.0))
            zw = ob.get("_z_fn") or zf   # 町では歩道の高さ（裏の山の高さではなく）
            s0 = 0.0
            for i in range(len(q) - 1):
                za, zb = zw(q[i]), zw(q[i + 1]); u = q[i + 1] - q[i]; L = float(np.linalg.norm(u))
                if L < 1e-6:
                    continue
                n = np.array([-u[1], u[0]]) / L; top = max(za, zb) + h
                box = _box((q[i] + q[i + 1]) / 2, n, L + 0.02, 0.3, min(za, zb) - 0.3, top)
                if ob.get("top") == "follow":   # 上端を地面の傾きに沿わせる（2 m ごとの段にしない）
                    v = box[0].copy(); up = v[:, 2] > top - 1e-9
                    t = np.clip((v[up, :2] - q[i]) @ (u / L) / L, 0.0, 1.0); v[up, 2] = za + h + (zb - za) * t
                    box = (v, box[1], box[2])
                part = _wall_uv(box, q[i], u / L, n, s0)
                (mat_walls.setdefault(ob["material"], []) if ob.get("material") else walls).append(part)
                s0 += L
                if fh > 0:
                    add("railing_bars", _quad(q[i], q[i + 1], top, top + fh, 0.0, L / 2.0, both=True))
        else:
            raise ValueError(f"物の kind が分かりません: {ob}")
    out = [(f"{prefix}_{mat}", *_merge(parts), mat, True) for mat, parts in groups.items()]
    if walls:
        out.append((f"1WALL_{prefix}_retaining", *_merge(walls), "concrete", True))
    if fence_walls:   # 木の柵の当たり判定（見えない）
        out.append((f"1WALL_{prefix}_woodfence", *_merge(fence_walls), "wood", False))
    for k, (mat, parts) in enumerate(mat_walls.items()):   # 材質を指定した擁壁（material）
        out.append((f"1WALL_{prefix}_retaining_{mat}", *_merge(parts), mat, True))
    return out


def pole_sign_meshes(signs, z_fn, prefix="town"):
    """立て看板：柱（高さ pole_h）と、柱の上の方の板（両面に文字）。signs = [{"text", "bg", "fg", "at", "facing_deg", "w_m", "h_m", "z_m", "pole_h", "no_pole"}]"""
    out, poles, wood_posts, caps = [], [], [], []
    for k, s in enumerate(signs):
        p = _xy(s["at"]); z = float(z_fn(np.atleast_1d(p[0]), np.atleast_1d(p[1]))[0])
        face = _heading(float(s.get("facing_deg", 0.0))); n = np.array([-face[1], face[0]])
        w, h, z1 = float(s.get("w_m", 1.2)), float(s.get("h_m", 2.0)), z + float(s.get("z_m", 3.0))
        dep = 0.05 if s.get("no_pole") or s.get("posts") == "single" else 0.3   # no_pole：信号の腕などに付ける薄い板（柱なし）
        front = _quad(p - n * w / 2, p + n * w / 2, z1, z1 + h)          # 表（face の側）
        back = _quad(p + n * w / 2 - face * dep, p - n * w / 2 - face * dep, z1, z1 + h)
        mat = "polesign_" + str(k) if prefix == "town" else f"{prefix}_polesign_{k}"
        out.append((f"{prefix}_polesign_{k}", *_merge([front, back]), mat, True))
        if s.get("posts") == "log":    # 丸太の柱2本（板より少し上まで、頭は白く塗る）、板の裏は木
            for side in (1, -1):
                q = p - face * dep / 2 + n * side * (w / 2 + 0.09)
                wood_posts.append(_prism(q, 0.09, z - 0.3, z1 + h + 0.12, 10))
                caps.append(_prism(q, 0.092, z1 + h + 0.08, z1 + h + 0.125, 10))
            wood_posts.append(_box(p - face * dep / 2, face, w, dep - 0.02, z1, z1 + h))
        elif s.get("posts") == "wood":   # 木の看板：板の両端に角柱2本（板より少し上まで）、板の裏は木
            for side in (1, -1):
                wood_posts.append(_box(p - face * dep / 2 + n * side * (w / 2 + 0.06), face, 0.12, 0.12, z - 0.3, z1 + h + 0.12))
            wood_posts.append(_box(p - face * dep / 2, face, w, dep - 0.02, z1, z1 + h))
        elif s.get("posts") == "single":   # 細い柱 1 本に薄い板（監視カメラの札など）
            poles.append(_prism(p - face * 0.06, 0.035, z - 0.3, z1 + h, 8))
        elif not s.get("no_pole"):
            poles.append(_prism(p - face * 0.15, 0.12, z - 0.3, z + float(s.get("pole_h", z1 - z + h)), 8))
            poles.append(_box(p - face * 0.15, face, w, 0.3, z1, z1 + h))
    if poles:
        out.append((f"1WALL_{prefix}_polesign_poles", *_merge(poles), "pole_grey", True))
    if wood_posts:
        out.append((f"1WALL_{prefix}_polesign_posts", *_merge(wood_posts), "wood", True))
    if caps:
        out.append((f"{prefix}_polesign_caps", *_merge(caps), "pole_white", True))
    return out


def write_pole_sign_textures(signs, out_dir, prefix="town"):
    from pathlib import Path
    from .textures import sign_texture
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    paths = {}
    for k, s in enumerate(signs):
        w, h = float(s.get("w_m", 1.2)), float(s.get("h_m", 2.0))
        px = 160
        mat = "polesign_" + str(k) if prefix == "town" else f"{prefix}_polesign_{k}"
        if s.get("style") in ("shop_front", "annex_front", "toilet_front", "trail_map", "terayama_toilet_front", "cafe_front", "cafe_attic", "rail_band", "hall_front", "hall_sign"):   # 建物の正面に貼る絵（峠の売店・トイレ）と案内地図
            from . import textures as tx
            {"shop_front": tx.shop_front_texture, "annex_front": tx.annex_front_texture, "toilet_front": tx.toilet_front_texture,
             "trail_map": tx.trail_map_texture, "terayama_toilet_front": tx.terayama_toilet_front_texture,
             "cafe_front": tx.cafe_front_texture, "cafe_attic": tx.cafe_attic_texture, "rail_band": tx.rail_band_texture,
             "hall_front": tx.hall_front_texture, "hall_sign": tx.hall_sign_texture}[s["style"]](int(w * px * 2), int(h * px * 2)).save(out / f"{mat}.png")
        elif s.get("style") == "wood_carved":   # 木を彫って白く塗った看板（lines に行ごとの大きさ・帯）
            from .textures import wood_sign_texture
            wood_sign_texture(s["lines"], int(w * px * 2), int(h * px * 2)).save(out / f"{mat}.png")
        else:
            sign_texture(s["text"], s.get("bg", "#FFFFFF"), s.get("fg", "#000000"), int(w * px), int(h * px), border=s.get("border")).save(out / f"{mat}.png")
        paths[mat] = f"textures/{mat}.png"
    return paths
