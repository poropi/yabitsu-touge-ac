"""S3 の道路まわりに足す部品：柵（白い縦格子・白い波形ガードレール・紅白・緑の金網）、電柱と電線、
路面の貼り物（マンホール・グレーチング・補修跡）。どれも中心線の Section と表（s3_dress.csv）から機械的に作る。
座標は manifest と同じ（x 東, y 北, z 上）。"""
import numpy as np

from .crosssection import GUTTER_WIDTH, surface_height
from .features import SIDES
from .meshgen import ribbon

DECAL_LIFT = 0.02           # 路面の貼り物は路面から 2 cm 上げる（ちらつき防止）
FENCE_GAP = 0.15            # 側溝の外側から柵までの間
POLE_GAP = 0.8              # 側溝（歩道）の外側から電柱の中心まで


def _rows(sec, s_from, s_to):
    return np.flatnonzero((sec.s >= s_from) & (sec.s <= s_to))


def _lat(sec, rows, side, extra, sidewalk):
    hw = (sec.hw_left if side > 0 else sec.hw_right)[rows]
    return hw + GUTTER_WIDTH + np.asarray(sidewalk, float)[rows] + extra


# ---------- 柵 ----------

# 柵の形：(lat の追加, z の下, z の上) の帯の並び。uv は ribbon の u（m/2）・v（s/4）
FENCES = {
    "wbeam": [(0.0, 0.60, 0.70), (0.06, 0.70, 0.78), (0.0, 0.78, 0.90)],         # 波形の3面（白）
    "wbeam_redwhite": [(0.0, 0.60, 0.70), (0.06, 0.70, 0.78), (0.0, 0.78, 0.90)],
    "white_bar": [(0.0, 0.0, 1.1)],                                                # 縦格子は透けるテクスチャの板
    "green_mesh": [(0.05, 0.0, 1.5)],
}
POST_EVERY = {"wbeam": 4.0, "wbeam_redwhite": 4.0, "white_bar": 2.0, "green_mesh": 2.5}


def fence_mesh(sec, s_from, s_to, side, kind, sidewalk):
    """柵（板と支柱）を1つのメッシュにする。(verts, faces, uvs)"""
    rows = _rows(sec, s_from, s_to)
    sd = SIDES[side]
    base = _lat(sec, rows, sd, FENCE_GAP, sidewalk)
    ze = sec.edge_z(sd)[rows]
    V, F, U = [], [], []
    for dl, z0, z1 in FENCES[kind]:
        L = base + dl
        v, f, uv = ribbon(sec.x[rows], sec.y[rows], sec.heading[rows], np.c_[sd * L, sd * L],
                          np.c_[ze + z0, ze + z1], sec.s[rows], u_scale=2.0, flip=sd < 0)
        V.append(v); F.append(f); U.append(uv)
    # 支柱：細い四角柱
    every = POST_EVERY[kind]
    top = max(z1 for _, _, z1 in FENCES[kind])
    for s in np.arange(sec.s[rows[0]], sec.s[rows[-1]] + 1e-6, every):
        i = int(np.clip(np.searchsorted(sec.s, s), rows[0], rows[-1]))
        h = sec.heading[i]
        n = np.array([-np.sin(h), np.cos(h)]) * sd
        t = np.array([np.cos(h), np.sin(h)])
        c = np.array([sec.x[i], sec.y[i]]) + n * (_lat(sec, np.array([i]), sd, FENCE_GAP + 0.05, sidewalk)[0])
        r = 0.035
        corners = [c + (a * t + b * n) * r for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        zb, zt = sec.edge_z(sd)[i] - 0.1, sec.edge_z(sd)[i] + top + 0.05
        b0 = sum(len(v) for v in V)
        V.append(np.array([(p[0], p[1], z) for z in (zb, zt) for p in corners]))
        F.append(np.array([(b0 + k, b0 + (k + 1) % 4, b0 + 4 + (k + 1) % 4, b0 + 4 + k) for k in range(4)]))
        U.append(np.array([(k / 4, 0) for k in range(4)] + [(k / 4, 1) for k in range(4)], float))
    off = np.cumsum([0] + [len(v) for v in V[:-1]])
    return (np.concatenate(V), np.concatenate([np.asarray(f) + o if k < len(FENCES[kind]) else np.asarray(f)
                                               for k, (f, o) in enumerate(zip(F, off))]), np.concatenate(U))


# ---------- 電柱と電線 ----------

def utility_poles(sec, s_from, s_to, side, spacing, keep, sidewalk, rng=None, tries=(0, 4, -4, 8, -8, 12)):
    """電柱の位置 [(x, y, z)]。道路・建物など植えない所に当たれば、前後へずらして探す。"""
    rng = rng or np.random.default_rng(int(s_from))
    sd = SIDES[side]
    out = []
    for s in np.arange(s_from + rng.uniform(0, spacing * 0.5), s_to, spacing):
        for d in tries:
            i = int(np.clip(np.searchsorted(sec.s, s + d), 0, sec.n - 1))
            h = sec.heading[i]
            n = np.array([-np.sin(h), np.cos(h)]) * sd
            c = np.array([sec.x[i], sec.y[i]]) + n * _lat(sec, np.array([i]), sd, POLE_GAP, sidewalk)[0]
            if not keep.blocked(c[None])[0]:
                out.append((float(c[0]), float(c[1]), float(sec.edge_z(sd)[i])))
                break
    return out


def pole_mesh(poles, height=12.0, r0=0.19, r1=0.12, sides=8):
    V, F, U = [], [], []
    for x, y, z in poles:
        b = len(V)
        for k, (zz, r) in enumerate(((z - 0.5, r0), (z + height, r1))):
            for j in range(sides + 1):
                a = 2 * np.pi * j / sides
                V.append((x + r * np.cos(a), y + r * np.sin(a), zz)); U.append((j / sides, k * height / 2))
        for j in range(sides):
            F.append((b + j, b + j + 1, b + sides + 2 + j, b + sides + 1 + j))
        # 腕木（電線を張る横棒）
        b = len(V)
        for zz in (z + height - 1.0, z + height - 0.85):
            for dx in (-0.9, 0.9):
                V.append((x + dx, y, zz)); U.append((0, 0))
        F.append((b, b + 1, b + 3, b + 2))
    return np.array(V, float), np.array(F, np.int64), np.array(U, float)


def wire_mesh(poles, n_wires=5, top=11.0, sag=0.6, width=0.025, steps=12):
    """隣どうしの電柱の間に、たるんだ電線（細い帯）を張る。"""
    V, F, U = [], [], []
    for (x0, y0, z0), (x1, y1, z1) in zip(poles[:-1], poles[1:]):
        d = np.hypot(x1 - x0, y1 - y0)
        if d > 80 or d < 1:
            continue
        for w in range(n_wires):
            dz = top - 0.35 * w
            off = (w % 3 - 1) * 0.6
            t = np.linspace(0, 1, steps + 1)
            px = x0 + (x1 - x0) * t; py = y0 + (y1 - y0) * t
            nx, ny = -(y1 - y0) / d, (x1 - x0) / d
            px, py = px + nx * off, py + ny * off
            pz = z0 + (z1 - z0) * t + dz - sag * (d / 40) * 4 * t * (1 - t)
            b = len(V)
            for k in range(steps + 1):
                V += [(px[k], py[k], pz[k] - width / 2), (px[k], py[k], pz[k] + width / 2)]
                U += [(0, k), (1, k)]
            F += [(b + 2 * k, b + 2 * k + 2, b + 2 * k + 3, b + 2 * k + 1) for k in range(steps)]
    return np.array(V, float).reshape(-1, 3), np.array(F, np.int64).reshape(-1, 4), np.array(U, float).reshape(-1, 2)


# ---------- 路面の貼り物 ----------

DECAL_SIZE = {"manhole": (0.75, 0.75), "grating": (0.45, 1.0), "patch": (1.6, 2.4)}


def _quad(sec, i, lat, size, rot=0.0):
    h = sec.heading[i] + rot
    t = np.array([np.cos(h), np.sin(h)]); n = np.array([-np.sin(sec.heading[i]), np.cos(sec.heading[i])])
    c = np.array([sec.x[i], sec.y[i]]) + n * lat
    w, l = size
    side = np.array([-t[1], t[0]])
    pts = [c + a * t * l / 2 + b * side * w / 2 for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    z = surface_height(lat, sec.z[i], sec.slope[i], sec.crown[i]) + DECAL_LIFT
    v = np.array([(p[0], p[1], z) for p in pts])
    return v, np.array([(0, 1, 2, 3)]), np.array([(0, 1), (0, 0), (1, 0), (1, 1)], float)


def road_decals(sec, rules, seed=1):
    """rules: [(s_from, s_to, kind, spacing)] → [(verts, faces, uvs, kind)]。
    マンホールは車線の中、グレーチングは左右の側溝のきわ、補修跡は車線のどこか。"""
    rng = np.random.default_rng(seed)
    out = []
    for s_from, s_to, kind, spacing in rules:
        for s in np.arange(s_from + rng.uniform(0, spacing), s_to, spacing):
            i = int(np.clip(np.searchsorted(sec.s, s), 0, sec.n - 1))
            if kind == "grating":
                for sd in (1, -1):
                    hw = sec.hw_left[i] if sd > 0 else sec.hw_right[i]
                    out.append((*_quad(sec, i, sd * (hw - 0.25), DECAL_SIZE[kind]), kind))
            else:
                lat = rng.choice([-1, 1]) * rng.uniform(0.8, 1.8)
                out.append((*_quad(sec, i, lat, DECAL_SIZE[kind], rot=rng.uniform(-0.2, 0.2) if kind == "patch" else 0.0), kind))
    return out
