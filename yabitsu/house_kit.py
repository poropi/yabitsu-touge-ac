"""ゲーム用の日本の家（部品の家）。家の輪郭（向きのついた長方形にまとめる）に、半間 0.91 m 刻みで壁・窓・玄関・ベランダ・屋根を並べる。
Blender の試作（.claude/skills/japanese-house-kit の kit.py）の軽い版：窓の枠・召し合わせ・カーテン・面格子は窓の絵（アトラス）に焼き、
形は壁のへこみ（見込み）・窓台・シャッターの箱・ベランダ・軒・破風・雨樋だけ。道に近い家（near）は凹凸あり、遠い家は窓の絵を壁に貼るだけ。
箱に窓の絵を貼る作り方（buildings.py）は名古木でストリートビューと全く違って見えた（2026-10-28）ので、S3 の家はこちらで作る。
部品は buildings.building_meshes と同じ (名前, 頂点, 面, UV, 材質, 描画するか)。壁は当たり判定あり（1WALL_）。"""
import hashlib
import math

import numpy as np
from PIL import Image, ImageDraw

from . import facades

M = 0.91            # 半間
SH = 2.9            # 階の高さ
FL = 0.5            # 基礎（一番高い地面から 1 階の床）
REVEAL = 0.1        # 窓のへこみ（壁の面からガラスまで）
NEAR_M = 35.0       # 道からこの距離までの家は凹凸あり
WALLS = ("#ECEAE4", "#E6E0D2", "#D9D2C2", "#C9C4BA", "#B8AE9C", "#8C8478", "#5A5650", "#3E4656", "#5A4636", "#7A8A78")
ROOFS = ("#5A3C2C", "#3A3A3C", "#4C4840", "#2E3438", "#6A4A36")
TILE = 256          # 窓の絵 1 枚の画素
WIN_KINDS = ("big", "big_lace", "koshi", "koshi_lace", "small", "door", "big_shutter", "plain_door")


def _seed(*k):
    return int(hashlib.md5("|".join(map(str, k)).encode()).hexdigest()[:8], 16)


def _hx(c):
    return c.lstrip("#").upper()


# ---------------- 形の部品 ----------------
class Parts:
    def __init__(self):
        self.m = {}

    def quad(self, mat, p, uv):
        """p＝4 点（表から見て反時計回り）、uv＝4 点。"""
        self.m.setdefault(mat, []).append((np.asarray(p, float), np.array([[0, 1, 2, 3]]), np.asarray(uv, float)))

    def tri(self, mat, p, uv):
        self.m.setdefault(mat, []).append((np.asarray(p, float), np.array([[0, 1, 2, 2]]), np.asarray(uv, float)))

    def box(self, mat, P, a0, a1, b0, b1, z0, z1, tile=1.0):
        """P(a, b, z) → 世界の点。a・b・z の範囲の箱。UV は面の中の長さ m / tile。"""
        c = [[P(a, b, z) for a, b, z in ((a0, b0, z0), (a1, b0, z0), (a1, b1, z0), (a0, b1, z0))],
             [P(a, b, z) for a, b, z in ((a0, b0, z1), (a1, b0, z1), (a1, b1, z1), (a0, b1, z1))]]
        lo, hi = c
        la, lb, lz = abs(a1 - a0) / tile, abs(b1 - b0) / tile, abs(z1 - z0) / tile
        sides = [((lo[0], lo[1], hi[1], hi[0]), la), ((lo[1], lo[2], hi[2], hi[1]), lb), ((lo[2], lo[3], hi[3], hi[2]), la), ((lo[3], lo[0], hi[0], hi[3]), lb)]
        if (a1 - a0) * (b1 - b0) < 0:
            sides = [(s[::-1], l) for s, l in sides]
        for s, l in sides:
            self.quad(mat, s, [[0, 0], [l, 0], [l, lz], [0, lz]])
        top = (hi[0], hi[1], hi[2], hi[3]) if (a1 - a0) * (b1 - b0) > 0 else (hi[3], hi[2], hi[1], hi[0])
        self.quad(mat, top, [[0, 0], [la, 0], [la, lb], [0, lb]])

    def cyl(self, mat, p0, p1, r, seg=6):
        p0, p1 = np.asarray(p0, float), np.asarray(p1, float); d = p1 - p0; L = np.linalg.norm(d)
        if L < 1e-6:
            return
        d /= L; a = np.cross(d, [0, 0, 1.0] if abs(d[2]) < 0.9 else [1.0, 0, 0]); a /= np.linalg.norm(a); b = np.cross(d, a)
        for k in range(seg):
            t0, t1 = 2 * math.pi * k / seg, 2 * math.pi * (k + 1) / seg
            o0 = r * (math.cos(t0) * a + math.sin(t0) * b); o1 = r * (math.cos(t1) * a + math.sin(t1) * b)
            self.quad(mat, [p0 + o0, p0 + o1, p1 + o1, p1 + o0], [[k / seg, 0], [(k + 1) / seg, 0], [(k + 1) / seg, L], [k / seg, L]])

    def items(self, name, wall_mat):
        out = []
        for mat, parts in self.m.items():
            v = np.concatenate([p[0] for p in parts]); uv = np.concatenate([p[2] for p in parts])
            f = np.concatenate([p[1] + sum(len(q[0]) for q in parts[:i]) for i, p in enumerate(parts)]).astype(np.int32)
            out.append((f"{'1WALL_' if mat == wall_mat else ''}hk_{name}_{mat}", v, f, uv, mat, True))
        return out


# ---------------- 1 軒 ----------------
def rect_of(fp, road_xy):
    """輪郭 → (中心, 長辺 L, 短辺 W, 長辺の向きの角度, 正面の符号)。正面（+y）は道に近い長辺。"""
    rect = np.asarray(fp.minimum_rotated_rectangle.exterior.coords)[:4]
    e = [np.linalg.norm(rect[(i + 1) % 4] - rect[i]) for i in range(4)]
    c = rect.mean(0)
    v = rect[1] - rect[0] if e[0] >= e[1] else rect[2] - rect[1]
    ang = math.atan2(v[1], v[0]); n = np.array([-math.sin(ang), math.cos(ang)])
    near = road_xy[np.argmin(np.hypot(*(road_xy - c).T))]
    if (near - c) @ n < 0:
        ang += math.pi
    return c, max(e[0], e[1]), min(e[0], e[1]), ang


def palette(bid, wall=None, roof=None):
    r = _seed(bid)
    wall = wall or WALLS[r % len(WALLS)]
    clad = "tin" if (facades._hex(wall).mean() < 0.4 and r % 5 == 0) else ("plaster" if r % 3 == 0 else "siding")
    roof = roof or ROOFS[(r >> 4) % len(ROOFS)]
    return dict(wall=f"hkw_{clad}_{_hx(wall)}", wall_hex=wall, clad=clad, roof=f"hkr_{_hx(roof)}", roof_hex=roof,
                trim="hk_trim_white" if (r >> 8) % 3 else "hk_trim_dark", win="hk_win_silver" if (r >> 10) % 5 < 3 else "hk_win_bronze",
                gutter="hk_gutter_light" if (r >> 12) % 2 else "hk_gutter_dark", base="hk_base", metal="hk_metal", ac="hk_ac")


def _layout(nb, storey, face, rnd):
    out, left = [], nb
    while left > 0:
        if face == "front":
            if storey == 0 and not any(k == "door" for k, _ in out) and left <= nb - 1:
                out.append(("door", 2)); left -= 2; continue
            k = rnd.choice(["big", "big", "koshi", "koshi", "plain", "small"] if storey == 0 else ["big", "koshi", "koshi", "koshi", "plain", "small"])
        elif face == "back":
            k = rnd.choice(["koshi", "koshi", "small", "small", "plain"])
        else:
            k = rnd.choice(["small", "small", "koshi", "plain", "plain"])
        wb = min({"big": 2, "koshi": 2, "small": 1, "plain": 1, "door": 2}[k], left)
        if wb == 1 and k in ("big", "koshi", "door"):
            k = "small"
        out.append((k, wb)); left -= wb
    return out


def _win_uv(kind):
    i = WIN_KINDS.index(kind); col, row = i % 4, i // 4
    u0, v0 = col / 4, 1 - (row + 1) / 2
    return [[u0, v0], [u0 + 0.25, v0], [u0 + 0.25, v0 + 0.5], [u0, v0 + 0.5]]


def house_parts(bid, fp, ground_z, road_xy, storeys=2, near=True, wall=None, roof=None, roof_type=None):
    import random
    rnd = random.Random(_seed(bid, "layout"))
    c, L, W, ang = rect_of(fp, road_xy)
    nx, ny = max(4, round(L / M)), max(4, round(W / M)); L, W = nx * M, ny * M
    ring = np.asarray(fp.exterior.coords)
    zg = np.asarray(ground_z(ring[:, 0], ring[:, 1]), float); zg = zg[np.isfinite(zg)]
    zc = float(np.asarray(ground_z(np.atleast_1d(c[0]), np.atleast_1d(c[1])), float)[0])
    z_low = min(zg.min(), zc) if len(zg) else zc; floor = (max(zg.max(), zc) if len(zg) else zc) + FL
    ca, sa = math.cos(ang), math.sin(ang)
    P = lambda x, y, z: (c[0] + ca * x - sa * y, c[1] + sa * x + ca * y, floor + z)    # 家の局所 → 世界
    pal = palette(bid, wall, roof); G = Parts()
    x0, x1, y0, y1 = -L / 2, L / 2, -W / 2, W / 2
    G.box(pal["base"], P, x0, x1, y0, y1, z_low - floor - 0.3, 0.0, tile=2.0)
    has_balc = storeys >= 2 and rnd.random() < 0.8
    if has_balc:
        bw = max(3, int(nx * rnd.uniform(0.5, 0.75))); bs = rnd.randint(0, nx - bw); bal = (x0 + bs * M, bw * M)
    # 面：(角, 面に沿う向き, 外向き, 半間の数)
    faces = {"front": ((x0, y1), (1, 0), (0, 1), nx), "back": ((x1, y0), (-1, 0), (0, -1), nx),
             "right": ((x1, y1), (0, -1), (1, 0), ny), "left": ((x0, y0), (0, 1), (-1, 0), ny)}
    for st in range(storeys):
        zb = st * SH
        for face, (o, ud, od, nb) in faces.items():
            F = lambda u, d, z, o=o, ud=ud, od=od, zb=zb: P(o[0] + ud[0] * u + od[0] * d, o[1] + ud[1] * u + od[1] * d, zb + z)
            FQ = lambda mat, pts, uv: G.quad(mat, pts[::-1], uv[::-1])   # 面に沿う u と上の z で回ると内向きになるので逆に
            u = 0.0
            for kind, wb in _layout(nb, st, face, rnd):
                w = wb * M
                if kind == "plain":
                    FQ(pal["wall"], [F(u, 0, 0), F(u + w, 0, 0), F(u + w, 0, SH), F(u, 0, SH)], [[u / 2, (zb) / 2], [(u + w) / 2, zb / 2], [(u + w) / 2, (zb + SH) / 2], [u / 2, (zb + SH) / 2]])
                    u += w; continue
                if kind == "door":
                    ow, z0, z1, tex = 1.2, 0.0, 2.3, ("door" if rnd.random() < 0.6 else "plain_door")
                elif kind == "big":
                    ow, z1 = w - 0.25, 2.05
                    z0 = 0.05 if (st == 0 or (has_balc and face == "front")) else 0.9
                    tex = rnd.choice(["big", "big_lace", "big_shutter"]) if z0 < 0.5 else rnd.choice(["koshi", "koshi_lace"])
                elif kind == "koshi":
                    ow, z0, z1, tex = min(1.65, w - 0.25), 0.9, 2.0, rnd.choice(["koshi", "koshi_lace"])
                else:
                    ow, z0, z1, tex = 0.6, 1.25, 1.95, "small"
                a, b = u + (w - ow) / 2, u + (w + ow) / 2
                wq = lambda p0, p1, q0, q1: FQ(pal["wall"], [F(p0, 0, q0), F(p1, 0, q0), F(p1, 0, q1), F(p0, 0, q1)],
                                                   [[p0 / 2, (zb + q0) / 2], [p1 / 2, (zb + q0) / 2], [p1 / 2, (zb + q1) / 2], [p0 / 2, (zb + q1) / 2]])
                wq(u, a, 0, SH); wq(b, u + w, 0, SH)
                if z0 > 0: wq(a, b, 0, z0)
                wq(a, b, z1, SH)
                dd = -REVEAL if near else 0.015
                FQ(pal["win"], [F(a, dd, z0), F(b, dd, z0), F(b, dd, z1), F(a, dd, z1)], _win_uv(tex))
                if near:   # 見込み（窓のまわりの壁の厚み）と窓台・シャッターの箱
                    FQ(pal["trim"], [F(a, 0, z0), F(a, dd, z0), F(a, dd, z1), F(a, 0, z1)], [[0, 0], [0.1, 0], [0.1, 1], [0, 1]])
                    FQ(pal["trim"], [F(b, dd, z0), F(b, 0, z0), F(b, 0, z1), F(b, dd, z1)], [[0, 0], [0.1, 0], [0.1, 1], [0, 1]])
                    FQ(pal["trim"], [F(a, dd, z1), F(b, dd, z1), F(b, 0, z1), F(a, 0, z1)], [[0, 0], [1, 0], [1, 0.1], [0, 0.1]])
                    if z0 > 0.2:
                        G.box(pal["metal"], lambda p, q, r: F(p, q, r), a - 0.06, b + 0.06, dd, 0.07, z0 - 0.04, z0)
                    if tex in ("big", "big_lace", "big_shutter") or (tex.startswith("koshi") and rnd.random() < 0.3):
                        G.box(pal["metal"], lambda p, q, r: F(p, q, r), a - 0.06, b + 0.06, 0.0, 0.2, z1, z1 + 0.28)
                    if kind == "door":
                        G.box(pal["trim"], lambda p, q, r: F(p, q, r), a - 0.4, b + 0.4, 0.0, 1.0, 2.45, 2.55)
                        G.box(pal["base"], lambda p, q, r: F(p, q, r), a - 0.4, b + 0.4, 0.0, 1.3, -FL, -0.02)
                u += w
        if st > 0:
            G.box(pal["trim"], P, x0 - 0.02, x1 + 0.02, y0 - 0.02, y1 + 0.02, st * SH - 0.04, st * SH + 0.04)
    if has_balc:
        bx, bw_ = bal; zb = SH; D = 0.91
        G.box(pal["base"], P, bx, bx + bw_, y1, y1 + D, zb - 0.25, zb)
        pm = pal["wall"] if rnd.random() < 0.6 else pal["metal"]
        G.box(pm, P, bx, bx + bw_, y1 + D - 0.1, y1 + D, zb, zb + 1.0)
        G.box(pm, P, bx, bx + 0.1, y1, y1 + D - 0.1, zb, zb + 1.0); G.box(pm, P, bx + bw_ - 0.1, bx + bw_, y1, y1 + D - 0.1, zb, zb + 1.0)
        G.box(pal["metal"], P, bx - 0.02, bx + bw_ + 0.02, y1 + D - 0.13, y1 + D + 0.03, zb + 1.0, zb + 1.05)
    top = storeys * SH
    _roof(G, P, pal, L, W, top, roof_type or ("hip" if rnd.random() < 0.4 else "gable"), near)
    if near:
        ze = top - 0.6 * 0.4 - 0.2
        for s in (1, -1):
            y = s * (W / 2 + 0.66)
            G.cyl(pal["gutter"], P(x0 - 0.5, y, ze), P(x1 + 0.5, y, ze), 0.06)
            xd = (L / 2 - 0.25) * (1 if rnd.random() < 0.5 else -1)
            G.cyl(pal["gutter"], P(xd, s * (W / 2 + 0.07), ze - 0.3), P(xd, s * (W / 2 + 0.07), z_low - floor), 0.035, seg=5)
        ax = rnd.uniform(x0 + 0.6, x1 - 0.6)
        G.box(pal["ac"], P, ax - 0.4, ax + 0.4, y0 - 0.5, y0 - 0.2, z_low - floor, z_low - floor + 0.65)
    return G.items(bid, pal["wall"]), pal


def _roof(G, P, pal, L, W, top, kind, near):
    ev, eg, t, pitch = 0.6, 0.45, 0.18, 0.4
    x0, x1, y0, y1 = -L / 2, L / 2, -W / 2, W / 2
    zr, ze = top + (W / 2) * pitch, top - ev * pitch
    R, S = pal["roof"], pal["trim"]
    sl = math.hypot(W / 2 + ev, zr - ze)
    if kind == "hip" and L > W:
        rx = L / 2 - W / 2; X0, X1, Y0, Y1 = x0 - ev, x1 + ev, y0 - ev, y1 + ev
        A, B = P(-rx, 0, zr), P(rx, 0, zr)
        G.quad(R, [P(X0, Y1, ze), P(X1, Y1, ze), B, A][::-1], [[0, 0], [L / 2 + ev, 0], [rx / 2 + L / 4, sl / 2], [-rx / 2 + L / 4, sl / 2]][::-1])
        G.quad(R, [P(X1, Y0, ze), P(X0, Y0, ze), A, B][::-1], [[0, 0], [L / 2 + ev, 0], [rx / 2 + L / 4, sl / 2], [-rx / 2 + L / 4, sl / 2]][::-1])
        G.tri(R, [P(X1, Y1, ze), P(X1, Y0, ze), B][::-1], [[0, 0], [W / 2 + ev, 0], [W / 4, sl / 2]][::-1])
        G.tri(R, [P(X0, Y0, ze), P(X0, Y1, ze), A][::-1], [[0, 0], [W / 2 + ev, 0], [W / 4, sl / 2]][::-1])
        rim = [(X0, Y1), (X1, Y1), (X1, Y0), (X0, Y0)]
        for i in range(4):
            (a, b), (c_, d) = rim[i], rim[(i + 1) % 4]; ln = math.hypot(c_ - a, d - b)
            G.quad(S, [P(a, b, ze - t), P(c_, d, ze - t), P(c_, d, ze), P(a, b, ze)], [[0, 0], [ln, 0], [ln, 0.2], [0, 0.2]])
        G.box(R, P, -rx - 0.05, rx + 0.05, -0.1, 0.1, zr - 0.03, zr + 0.08, tile=2.0)
        _soffit(G, P, S, X0, X1, Y0, Y1, ze - t, x0, x1, y0, y1, top)
        return
    for s in (1, -1):
        yo = s * (W / 2 + ev)
        q = [P(x0 - eg, 0, zr), P(x1 + eg, 0, zr), P(x1 + eg, yo, ze), P(x0 - eg, yo, ze)]
        uv = [[0, sl / 2], [(L + 2 * eg) / 2, sl / 2], [(L + 2 * eg) / 2, 0], [0, 0]]
        G.quad(R, q if s > 0 else q[::-1], uv if s > 0 else uv[::-1])
        f = [P(x0 - eg, yo, ze - t), P(x1 + eg, yo, ze - t), P(x1 + eg, yo, ze), P(x0 - eg, yo, ze)]
        G.quad(S, f if s > 0 else f[::-1], [[0, 0], [L, 0], [L, 0.2], [0, 0.2]])
        so = [P(x0 - eg, s * W / 2, top), P(x1 + eg, s * W / 2, top), P(x1 + eg, yo, ze - t), P(x0 - eg, yo, ze - t)]
        G.quad(S, so[::-1] if s > 0 else so, [[0, 0], [1, 0], [1, 1], [0, 1]])   # 軒天
    for xx, s in ((x1, 1), (x0, -1)):
        tri = [P(xx, -W / 2, top), P(xx, W / 2, top), P(xx, 0, zr)]
        G.tri(pal["wall"], tri if s > 0 else tri[::-1], [[0, top / 2], [W / 2, top / 2], [W / 4, zr / 2]])
        for side in (1, -1):   # 破風板（両面）
            b = [P(xx + s * eg, 0, zr + 0.02), P(xx + s * eg, side * (W / 2 + ev), ze + 0.02), P(xx + s * eg, side * (W / 2 + ev), ze - t - 0.05), P(xx + s * eg, 0, zr - t - 0.05)]
            G.quad(S, b if s * side > 0 else b[::-1], [[0, 0], [1, 0], [1, 0.1], [0, 0.1]])
            G.quad(S, b[::-1] if s * side > 0 else b, [[0, 0], [1, 0], [1, 0.1], [0, 0.1]])
    G.box(R, P, x0 - eg, x1 + eg, -0.11, 0.11, zr - 0.03, zr + 0.09, tile=2.0)


def _soffit(G, P, S, X0, X1, Y0, Y1, z, x0, x1, y0, y1, top):
    """寄棟の軒天：軒先の内側から壁まで、4 面。"""
    for (a, b), (c_, d), (e, f), (g, h) in (((X0, Y1), (X1, Y1), (x1, y1), (x0, y1)), ((X1, Y0), (X0, Y0), (x0, y0), (x1, y0)),
                                           ((X1, Y1), (X1, Y0), (x1, y0), (x1, y1)), ((X0, Y0), (X0, Y1), (x0, y1), (x0, y0))):
        G.quad(S, [P(c_, d, z), P(a, b, z), P(g, h, top), P(e, f, top)], [[0, 0], [1, 0], [1, 1], [0, 1]])


def store_parts(bid, fp, ground_z, road_xy, name="", wall="#F4F4F2"):
    """コンビニの店（平屋・陸屋根）：正面はガラス張りの店構え（中の棚・明かり・ポスターの絵）と茶色の柱、その上に 3 本線の看板の帯
    （正面と両横）、真ん中に店名の板、白い壁、パラペット、屋上の室外機（ストリートビュー 名古木のセブン-イレブン、2026-10-28）。"""
    c, L, W, ang = rect_of(fp, road_xy)
    ring = np.asarray(fp.exterior.coords)
    zg = np.asarray(ground_z(ring[:, 0], ring[:, 1]), float); zg = zg[np.isfinite(zg)]
    z_low = float(zg.min()) if len(zg) else 0.0; floor = (float(zg.max()) if len(zg) else 0.0) + 0.15
    ca, sa = math.cos(ang), math.sin(ang)
    P = lambda x, y, z: (c[0] + ca * x - sa * y, c[1] + sa * x + ca * y, floor + z)
    G = Parts(); wm = f"hkw_plaster_{_hx(wall)}"; H = 4.6
    x0, x1, y0, y1 = -L / 2, L / 2, -W / 2, W / 2
    G.box("hk_base", P, x0, x1, y0, y1, z_low - floor - 0.3, 0.0, tile=2.0)
    G.box(wm, P, x0, x1, y0, y1 - 0.25, 0.0, H, tile=2.0)                       # 正面より 0.25 m 引っ込めた壁の箱（正面はガラス）
    G.box(wm, P, x0, x0 + 0.6, y1 - 0.25, y1, 0.0, H, tile=2.0); G.box(wm, P, x1 - 0.6, x1, y1 - 0.25, y1, 0.0, H, tile=2.0)
    G.quad("hk_store_glass", [P(x1 - 0.6, y1 - 0.24, 0.05), P(x0 + 0.6, y1 - 0.24, 0.05), P(x0 + 0.6, y1 - 0.24, 2.65), P(x1 - 0.6, y1 - 0.24, 2.65)],
           [[0, 0], [(L - 1.2) / 3.0, 0], [(L - 1.2) / 3.0, 1], [0, 1]])
    n = max(2, int((L - 1.2) / 3.0))
    for k in range(n + 1):                                                         # 茶色の柱
        x = x0 + 0.6 + k * (L - 1.2) / n
        G.box("hk_store_pillar", P, x - 0.12, x + 0.12, y1 - 0.25, y1 + 0.05, 0.0, 2.75)
    G.box(wm, P, x0, x1, y1 - 0.25, y1 + 0.1, 2.65, 2.85)                         # ガラスの上の小さなひさし
    for (a, b, s0) in (((x0 - 0.05, y1 + 0.12), (x1 + 0.05, y1 + 0.12), 0), ((x1 + 0.12, y1 + 0.12), (x1 + 0.12, y0), 1), ((x0 - 0.12, y0), (x0 - 0.12, y1 + 0.12), 2)):
        ln = math.hypot(b[0] - a[0], b[1] - a[1])                                 # 3 本線の看板の帯（正面と両横、壁から 12 cm 出す）
        G.quad("hk_store_band", [P(a[0], a[1], 2.95), P(b[0], b[1], 2.95), P(b[0], b[1], 3.95), P(a[0], a[1], 3.95)][::-1], [[0, 0], [ln / 4, 0], [ln / 4, 1], [0, 1]])
    G.quad("hk_store_name", [P(1.2, y1 + 0.14, 3.05), P(-1.2, y1 + 0.14, 3.05), P(-1.2, y1 + 0.14, 3.85), P(1.2, y1 + 0.14, 3.85)],
           [[0, 0], [1, 0], [1, 1], [0, 1]])
    G.box("hk_trim_white", P, x0 - 0.05, x1 + 0.05, y0 - 0.05, y1 + 0.05, H, H + 0.3)   # パラペット
    for k in range(3):
        ax = x0 + (k + 1) * L / 4
        G.box("hk_ac", P, ax - 0.45, ax + 0.45, y0 + 1.0, y0 + 1.6, H, H + 0.9)
    return G.items(bid, wm), dict(wall=wm, wall_hex=wall, clad="plaster", roof="hk_trim_white", roof_hex="#ECEBE6", store_name=name)


SPLIT_M = 15.0      # これより長い輪郭は、約 11 m ずつ 1 m 空けて別の家に（1 軒にすると窓が延々と並び、工場や長屋に見えた）


def _split_long(fp):
    from shapely.geometry import Polygon
    rect = np.asarray(fp.minimum_rotated_rectangle.exterior.coords)[:4]
    e = [np.linalg.norm(rect[(i + 1) % 4] - rect[i]) for i in range(4)]
    k = 0 if e[0] >= e[1] else 1
    L = e[k]
    if L <= SPLIT_M:
        return [fp]
    n = int(math.ceil(L / 11.0)); a, b = rect[k], rect[k + 1]; w = rect[(k + 3) % 4] - rect[k]
    u = (b - a) / L; seg = L / n
    return [Polygon([a + u * (i * seg + 0.5), a + u * ((i + 1) * seg - 0.5), a + u * ((i + 1) * seg - 0.5) + w, a + u * (i * seg + 0.5) + w])
            for i in range(n)]


def house_meshes(buildings, ground_z, road_xy, near_line=None, avoid=None):
    """buildings（buildings.Building）を部品の家にした部品と、使った材質の一覧。near_line（shapely の線）から NEAR_M 以内は凹凸あり。"""
    out, pals = [], []
    for b in buildings:
        fp = b.footprint
        if getattr(b, "kit", None) == "conv_store":
            items, pal = store_parts(b.id, fp, ground_z, road_xy, b.name, b.wall)
            out += items; pals.append(pal); continue
        if avoid is not None and not avoid.is_empty:
            fp = fp.difference(avoid)
            if fp.is_empty or fp.area < 15:
                continue
            if fp.geom_type != "Polygon":
                fp = max(fp.geoms, key=lambda g: g.area)
        near = near_line is None or fp.distance(near_line) < NEAR_M
        for k, piece in enumerate(_split_long(fp)):
            items, pal = house_parts(f"{b.id}_{k}" if k else b.id, piece, ground_z, road_xy, storeys=2 if piece.area > 45 else 1, near=near,
                                     wall=None, roof=b.roof.get("color"))
            out += items; pals.append(pal)
    return out, pals


# ---------------- 絵 ----------------
def _window_tile(kind, sash, rng):
    """窓の絵 1 枚（TILE 四方、上が y=0）。枠・障子 2 枚・召し合わせ・ガラス（空の映り込み）・カーテン・網戸・面格子・ドア。"""
    T = TILE; im = Image.new("RGB", (T, T), tuple(int(x * 255) for x in sash)); d = ImageDraw.Draw(im)
    fr = int(T * 0.05)
    sky = np.linspace(1, 0, T)[:, None]
    if kind in ("door", "plain_door"):
        wood = (92, 72, 54) if kind == "door" else (52, 52, 56)
        d.rectangle([fr, fr, int(T * 0.7), T - 1], fill=wood)
        for k in range(6):
            y = fr + k * (T - fr) // 6; d.line([fr, y, int(T * 0.7), y], fill=tuple(max(0, c - 18) for c in wood), width=2)
        d.rectangle([int(T * 0.74), fr, T - fr, T - fr], fill=(185, 190, 192))       # 袖のすりガラス
        d.rectangle([int(T * 0.6), int(T * 0.45), int(T * 0.63), int(T * 0.6)], fill=(200, 200, 196))   # 取っ手
        return im
    # AC の ksPerPixel は映り込みが無いので、空を映した明るさを絵に入れる（暗いガラスはゲームで黒い穴に見えた。2026-10-28）
    g = np.zeros((T, T, 3)); dark = np.array([62, 74, 86.0]); lit = np.array([150, 166, 180.0])
    g[:] = dark + (lit - dark) * np.clip(0.75 - sky * 0.9, 0, 1)[..., None] * 1.0
    glass = Image.fromarray(g.astype(np.uint8))
    if kind == "small":
        glass = Image.new("RGB", (T, T), (176, 184, 188))
    im.paste(glass.crop((fr, fr, T - fr, T - fr)), (fr, fr))
    if kind in ("big_lace", "koshi_lace"):
        ov = Image.new("RGB", (T - 2 * fr, T - 2 * fr), (206, 206, 198)); im.paste(Image.blend(im.crop((fr, fr, T - fr, T - fr)), ov, 0.3), (fr, fr))   # 濃いと障子紙のように白く見えた
    if kind in ("big", "koshi", "big_shutter"):
        cw = int(T * rng.uniform(0.15, 0.3)); x = fr if rng.random() < 0.5 else T - fr - cw
        d.rectangle([x, fr, x + cw, T - fr], fill=(196, 188, 170) if rng.random() < 0.5 else (150, 160, 170))
    if kind == "big_shutter":
        h = int(T * rng.uniform(0.3, 0.6)); d.rectangle([fr, fr, T - fr, fr + h], fill=(150, 152, 150))
        for y in range(fr, fr + h, 6):
            d.line([fr, y, T - fr, y], fill=(120, 122, 120), width=1)
    if kind != "small":
        d.rectangle([T // 2 - 4, fr, T // 2 + 4, T - fr], fill=tuple(int(x * 255) for x in sash))           # 召し合わせ
        for x in (fr + 3, T - fr - 6):
            d.rectangle([x, fr, x + 3, T - fr], fill=tuple(int(x * 230) for x in sash))
        sx = fr if rng.random() < 0.5 else T // 2                                                          # 網戸（片側）
        scr = im.crop((sx, fr, sx + T // 2 - fr, T - fr)); im.paste(Image.blend(scr, Image.new("RGB", scr.size, (40, 42, 44)), 0.35), (sx, fr))
    else:
        for k in range(1, 5):                                                                              # 面格子
            x = k * T // 5; d.rectangle([x - 3, 0, x + 3, T], fill=(150, 154, 158))
    d.rectangle([0, 0, T - 1, T - 1], outline=tuple(int(x * 200) for x in sash), width=3)
    return im


def window_atlas(sash):
    rng = np.random.default_rng(_seed(str(sash)))
    im = Image.new("RGB", (TILE * 4, TILE * 2))
    for i, k in enumerate(WIN_KINDS):
        im.paste(_window_tile(k, sash, rng), ((i % 4) * TILE, (i // 4) * TILE))
    return im


def _flat(hexc, size=16):
    return Image.new("RGB", (size, size), tuple(int(x * 255) for x in facades._hex(hexc)))


def write_textures(pals, out_dir):
    """使った材質の絵を書き、{材質: "textures/xxx.png"}。壁・屋根は写真素材の下地（facades）、窓は窓の絵のアトラス。"""
    from pathlib import Path
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True); mats = {}

    def save(name, im):
        if name not in mats:
            im.save(out / f"{name}.png"); mats[name] = f"textures/{name}.png"
    save("hk_win_silver", window_atlas((0.70, 0.72, 0.74))); save("hk_win_bronze", window_atlas((0.24, 0.20, 0.17)))
    save("hk_trim_white", _flat("#ECEBE6")); save("hk_trim_dark", _flat("#3A3430"))
    save("hk_gutter_light", _flat("#E2E0DA")); save("hk_gutter_dark", _flat("#4A4642"))
    save("hk_metal", _flat("#8E9296")); save("hk_ac", _flat("#D6D6D0"))
    base = facades.detail("Concrete034", 2.0, 2.0, 2.0, 0.5, 0.6)[..., None] * facades._hex("#A9A79F")[None, None, :]
    save("hk_base", Image.fromarray(np.clip(base * 255, 0, 255).astype(np.uint8)))
    if any(p.get("store_name") is not None for p in pals):
        save("hk_store_glass", store_glass()); save("hk_store_band", store_band()); save("hk_store_pillar", _flat("#5A4030"))
        from .textures import sign_texture
        nm = next(p["store_name"] for p in pals if p.get("store_name") is not None) or "STORE"
        save("hk_store_name", sign_texture(nm, "#FFFFFF", "#008C45", 480, 160))
    for p in pals:
        if p["wall"] not in mats:
            asset, tile, lk, sk = facades.CLADDING[p["clad"]]
            img = facades.detail(asset, tile, 2.0, 2.0, lk, sk)[..., None] * facades._hex(p["wall_hex"])[None, None, :]
            save(p["wall"], Image.fromarray(np.clip(img * 255, 0, 255).astype(np.uint8)))
        if p["roof"] not in mats:
            save(p["roof"], facades.roof_image(p["roof_hex"]))
    return mats


# ---------------- コースの組み立てへのつなぎ ----------------
S3_RANGE = (0.0, 1700.0)   # S3 の起点（名古木）からの距離：まずこの区間の道ぞいだけ部品の家に置き換える
BAND_M = 80.0


def split(buildings, sec, s0):
    """buildings を (部品の家にするもの, 今までどおり) に分ける。S3_RANGE の道から BAND_M 以内の、勾配屋根・260 m² 以下・
    店の看板／ガラス／屋根だけの建物でないもの。"""
    from shapely.geometry import LineString
    m = (sec.s >= s0 + S3_RANGE[0]) & (sec.s <= s0 + S3_RANGE[1])
    line = LineString(np.c_[sec.x[m], sec.y[m]])
    kit, rest = [], []
    for b in buildings:
        ok = getattr(b, "kit", None) == "conv_store" or (
            b.roof.get("type", "flat") != "flat" and b.footprint.area <= 260 and not b.shopfront and not b.signs and not b.canopy
            and b.footprint.distance(line) < BAND_M)
        (kit if ok else rest).append(b)
    return kit, rest, line, np.c_[sec.x[m], sec.y[m]]


def store_glass(w=768, h=640):
    """店のガラス（横 3 m・縦 2.6 m）：明るい店の中（天井の明かり・色とりどりの棚）、ガラスに貼ったポスター、縦の桟。"""
    rng = np.random.default_rng(7)
    a = np.zeros((h, w, 3))
    a[:] = np.array([215, 218, 214.0])
    im = Image.fromarray(a.astype(np.uint8)); d = ImageDraw.Draw(im)
    for k in range(6):                                                             # 天井の明かり
        x = int((k + 0.5) * w / 6); d.rectangle([x - 40, 10, x + 40, 22], fill=(250, 250, 244))
    for r in range(3):                                                             # 棚
        y0 = int(h * (0.42 + r * 0.17))
        for x in range(0, w, 14):
            d.rectangle([x, y0, x + 11, y0 + int(h * 0.12)], fill=tuple(int(v) for v in rng.integers(60, 230, 3)))
        d.rectangle([0, y0 + int(h * 0.12), w, y0 + int(h * 0.14)], fill=(200, 200, 196))
    for k in range(rng.integers(2, 4)):                                            # ポスター
        x = int(rng.uniform(0.05, 0.7) * w); d.rectangle([x, int(h * 0.12), x + int(w * 0.18), int(h * 0.36)], fill=tuple(int(v) for v in rng.integers(80, 250, 3)))
    a = np.asarray(im, float) * 0.78 + np.array([30, 40, 50.0]) * 0.22            # ガラス越しの色
    im = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)); d = ImageDraw.Draw(im)
    d.rectangle([w // 2 - 5, 0, w // 2 + 5, h], fill=(150, 152, 154)); d.rectangle([0, 0, w, 8], fill=(150, 152, 154)); d.rectangle([0, h - 8, w, h], fill=(150, 152, 154))
    return im


def store_band(w=1024, h=256):
    """看板の帯（横 4 m・縦 1 m）：白地に、下寄りにオレンジ・緑・赤の 3 本の線（ロゴの絵は入れない）。"""
    im = Image.new("RGB", (w, h), (246, 246, 242)); d = ImageDraw.Draw(im)
    for k, col in enumerate(((238, 119, 0), (0, 140, 69), (226, 24, 54))):
        y = int(h * (0.52 + k * 0.12)); d.rectangle([0, y, w, y + int(h * 0.08)], fill=col)
    return im
