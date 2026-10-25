"""町の範囲（名古木交差点のまわり）の建物を1棟ずつ。形は外周線そのまま（四角に丸めない）。
階数・高さ・屋根・壁の色・窓・1階のガラス・看板は data/manual/s3_buildings.json に Street View で確かめて書く。

  {"id", "name", "footprint": [[lat, lon], ...], "storeys", "height_m",
   "roof": {"type": flat | gable | hip | mono, "rise_m", "ridge_deg"（棟の方位）, "color": "#RRGGBB"},
   "wall": "#RRGGBB", "windows": bool, "shopfront": [辺の番号], "signs": [{"text", "bg", "fg", "edge", "w_m", "h_m", "z_m"}],
   "canopy": {"points": [[lat, lon], ...], "z_m"} | null, "lot": {"points": [[lat, lon], ...]} | null}

辺の番号は、外周線を反時計回りにしたときの i 番目の点 → i+1 番目の点。"""
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import shapely
from PIL import Image
from shapely.geometry import Polygon
from shapely.geometry.polygon import orient

from . import geo
from .start_lot import Frame, _gable_ends

PARAPET_H = 0.3
SHOP_H = 3.0          # 1階の店のガラスの高さ
BAY_W = 4.0           # 壁の絵 1 枚の横幅（窓1つぶん）
CANOPY_T = 0.6
EAVES = 0.4


def _xy(ll):
    a = np.asarray(ll, float).reshape(-1, 2)
    x, y = geo.to_local(a[:, 0], a[:, 1])
    return np.c_[x, y]


@dataclass
class Building:
    id: str
    footprint: Polygon
    storeys: int
    height_m: float
    roof: dict
    wall: str
    windows: bool = True
    shopfront: list = field(default_factory=list)
    signs: list = field(default_factory=list)
    canopy: dict = None
    lot: dict = None
    name: str = ""

    @classmethod
    def from_spec(cls, d):
        fp = orient(Polygon(_xy(d["footprint"])).buffer(0), 1.0)
        return cls(str(d["id"]), fp, int(d.get("storeys", 2)), float(d.get("height_m", 6.0)), dict(d.get("roof", {})),
                   d.get("wall", "#E8E4DC"), bool(d.get("windows", True)), list(d.get("shopfront", [])),
                   list(d.get("signs", [])), d.get("canopy"), d.get("lot"), d.get("name", ""))


def load(path):
    """建物の json（buildings.py の形）を読む。gone=true（地図にはあるが、もう無い建物）は作らない。"""
    import json
    from pathlib import Path
    return [Building.from_spec(d) for d in json.loads(Path(path).read_text(encoding="utf-8")) if not d.get("gone")]


def facade_mat(b):
    """壁の材質：色と窓の有無が同じ建物は同じ絵を使う。"""
    return f"facade_{b.wall.lstrip('#').upper()}{'_w' if b.windows else ''}"


def gable_end_mat(b):
    """roof.gable_end の妻の三角の材質（窓の無い壁の色）。"""
    return f"facade_{b.roof['gable_end']['wall'].lstrip('#').upper()}"


def roof_mat(b):
    return f"roofc_{b.roof.get('color', '#707070').lstrip('#').upper()}"


def merge_by_material(meshes, max_verts=60000):
    """同じ材質・同じ種類（当たり判定あり 1WALL／なし）の部品を、頂点 max_verts ごとに1つにまとめる（描く回数を減らす）。"""
    groups = {}
    for name, v, f, uv, mat, rend in meshes:
        key = (name.startswith("1WALL_"), mat, rend)
        groups.setdefault(key, []).append((v, f, uv))
    out = []
    for (wall, mat, rend), parts in groups.items():
        cur, n, k = [], 0, 0
        for q in parts + [None]:
            if q is None or (n + len(q[0]) > max_verts and cur):
                if cur:
                    out.append((f"{'1WALL_' if wall else ''}bld_{mat}_{k:02d}", *_merge(cur), mat, rend)); k += 1
                cur, n = [], 0
            if q is not None:
                cur.append(q); n += len(q[0])
    return out


def _clip(b, avoid):
    fp = b.footprint
    if avoid is not None and fp.intersection(avoid).area > 0.3:
        g = fp.difference(avoid)
        g = g if g.geom_type == "Polygon" else max(getattr(g, "geoms", [Polygon()]), key=lambda q: q.area)
        fp = orient(g, 1.0)
    return fp


def clipped_report(buildings, avoid):
    """道・歩道と重なって削った建物：(id, 削った面積 m²)。"""
    return [(b.id, float(b.footprint.intersection(avoid).area)) for b in buildings
            if b.footprint.intersection(avoid).area > 0.3]


def _merge(parts):
    """三角形と四角形が混じるときは、三角形を「最後の頂点を重ねた四角形」にしてそろえる。"""
    vs, fs, us, base = [], [], [], 0
    for v, f, uv in parts:
        f = np.asarray(f)
        if f.size == 0:   # 長さのない辺など、面のない部品
            continue
        if f.shape[1] == 3:
            f = np.c_[f, f[:, 2]]
        vs.append(np.asarray(v, float)); fs.append(f + base); us.append(np.asarray(uv, float)); base += len(v)
    return np.concatenate(vs), np.concatenate(fs).astype(np.int32), np.concatenate(us)


def _tri_poly(poly, z):
    """多角形を三角形にし、高さ z に（上向き）。"""
    vs, fs = [], []
    for t in shapely.constrained_delaunay_triangles(poly).geoms:
        p = np.asarray(t.exterior.coords)[:3]
        if (p[1, 0] - p[0, 0]) * (p[2, 1] - p[0, 1]) - (p[1, 1] - p[0, 1]) * (p[2, 0] - p[0, 0]) < 0:
            p = p[::-1]
        b = len(vs); vs += p.tolist(); fs.append([b, b + 1, b + 2])
    v = np.array(vs, float)
    return np.c_[v, np.full(len(v), z)], np.array(fs), v / 4.0


def _wall_quads(ring, z0, z1, floor, storey_h, inward=False, u0=0.0):
    """外周（反時計回り）の各辺に立つ壁。表は外（inward なら内）。UV は u＝辺に沿った長さ/BAY_W、v＝(高さ−床)/1階の高さ。"""
    p = np.asarray(ring.coords)
    vs, fs, us, u = [], [], [], u0
    for i in range(len(p) - 1):
        a, b = p[i], p[i + 1]
        L = float(np.linalg.norm(b - a))
        if L < 1e-6:
            continue
        if inward:
            a, b = b, a
        k = len(vs)
        vs += [[*a, z0], [*b, z0], [*b, z1], [*a, z1]]
        us += [[u, (z0 - floor) / storey_h], [u + L / BAY_W, (z0 - floor) / storey_h],
               [u + L / BAY_W, (z1 - floor) / storey_h], [u, (z1 - floor) / storey_h]]
        fs.append([k, k + 1, k + 2, k + 3]); u += L / BAY_W
    return np.array(vs, float), np.array(fs), np.array(us, float)


def _edge_quad(a, b, z0, z1, floor, storey_h):
    return _wall_quads(shapely.LineString([a, b]), z0, z1, floor, storey_h)


def _frame(b, fp):
    rd = b.roof.get("ridge_deg")
    if rd is None:
        r = np.asarray(fp.minimum_rotated_rectangle.exterior.coords)[:4]
        e1, e2 = r[1] - r[0], r[2] - r[1]
        ax = e1 if np.linalg.norm(e1) >= np.linalg.norm(e2) else e2
        deg = math.degrees(math.atan2(ax[1], ax[0]))
    else:
        deg = 90.0 - float(rd)
    c = fp.centroid
    f = Frame(np.array([c.x, c.y]), deg)
    a, b = f.ab(np.asarray(fp.exterior.coords))
    return f, (float(a.min()), float(a.max())), (float(b.min()), float(b.max()))


def _q(f, a, b, z):
    v = np.c_[f.xy(np.asarray(a, float), np.asarray(b, float)), np.asarray(z, float)]
    return v


def _pitched_roof(b, fp, z_eave):
    kind = b.roof.get("type", "gable"); rise = float(b.roof.get("rise_m", 1.5))
    f, (a0, a1), (b0, b1) = _frame(b, fp)
    A0, A1, B0, B1 = a0 - EAVES, a1 + EAVES, b0 - EAVES, b1 + EAVES
    bm = (b0 + b1) / 2; W = b1 - b0; L = a1 - a0
    drop = rise * EAVES / max(W / 2, 1e-6)
    s = z_eave - drop; zr = z_eave + rise
    parts, gables, odd = [], [], None
    if kind == "mono":   # 片流れ：b0 側が低く、b1 側が高い
        drop = rise * EAVES / max(W, 1e-6)
        parts.append((_q(f, [A0, A1, A1, A0], [B0, B0, B1, B1], [z_eave - drop, z_eave - drop, zr + drop, zr + drop]),
                      [[0, 1, 2, 3]], [[0, 0], [L / 2, 0], [L / 2, W / 2], [0, W / 2]]))
        for aa, order in ((a1, (b0, b1)), (a0, (b1, b0))):
            zz = [z_eave if order[0] == b0 else zr, z_eave if order[1] == b0 else zr]
            gables.append((_q(f, [aa, aa, aa, aa], [order[0], order[1], order[1], order[0]], [z_eave, z_eave, zz[1], zz[0]]),
                           [[0, 1, 2, 3]], [[0, 0], [W / 4, 0], [W / 4, 1], [0, 1]]))
        # 高い側の壁（壁の上から屋根の高い縁まで）
        gables.append((_q(f, [a1, a0, a0, a1], [b1, b1, b1, b1], [z_eave, z_eave, zr, zr]), [[0, 1, 2, 3]],
                       [[0, 0], [L / 4, 0], [L / 4, 1], [0, 1]]))
    elif kind == "hip" and L > W:   # 寄棟
        r0, r1 = a0 + W / 2, a1 - W / 2
        parts.append((_q(f, [A0, A1, r1, r0], [B0, B0, bm, bm], [s, s, zr, zr]), [[0, 1, 2, 3]], [[0, 0], [L / 2, 0], [L / 2, 1], [0, 1]]))
        parts.append((_q(f, [A1, A0, r0, r1], [B1, B1, bm, bm], [s, s, zr, zr]), [[0, 1, 2, 3]], [[0, 0], [L / 2, 0], [L / 2, 1], [0, 1]]))
        parts.append((_q(f, [A1, A1, r1, r1], [B0, B1, bm, bm], [s, s, zr, zr]), [[0, 1, 2, 3]], [[0, 0], [W / 2, 0], [W / 2, 1], [0, 1]]))
        parts.append((_q(f, [A0, A0, r0, r0], [B1, B0, bm, bm], [s, s, zr, zr]), [[0, 1, 2, 3]], [[0, 0], [W / 2, 0], [W / 2, 1], [0, 1]]))
    else:                # 切妻
        parts.append((_q(f, [A0, A1, A1, A0], [B0, B0, bm, bm], [s, s, zr, zr]), [[0, 1, 2, 3]], [[0, 0], [L / 2, 0], [L / 2, 1], [0, 1]]))
        parts.append((_q(f, [A1, A0, A0, A1], [B1, B1, bm, bm], [s, s, zr, zr]), [[0, 1, 2, 3]], [[0, 0], [L / 2, 0], [L / 2, 1], [0, 1]]))
        v, fc, uv = _gable_ends(f, (a0, a1), (b0, b1), z_eave, rise)
        ge = b.roof.get("gable_end")   # 外向きの方位が bearing に近い方の妻の三角だけ別の色（正面の妻壁がタイル張りなど）
        if ge:
            out_a1 = f.xy([a1], [bm])[0] - f.xy([a0], [bm])[0]   # a1 の端の外向き
            want = np.array([math.sin(math.radians(ge["bearing"])), math.cos(math.radians(ge["bearing"]))])
            k = 0 if float(out_a1 @ want) > 0 else 1          # _gable_ends は a1 の端、a0 の端の順
            odd = (v[3 * k:3 * k + 3], np.array([[0, 1, 2]]), uv[3 * k:3 * k + 3])
            keep = [i for i in range(2) if i != k][0]
            v, fc, uv = v[3 * keep:3 * keep + 3], np.array([[0, 1, 2]]), uv[3 * keep:3 * keep + 3]
        gables.append((v, fc, uv))
    roof = _merge([(v, np.asarray(fc), np.asarray(uv, float)) for v, fc, uv in parts])
    return roof, (_merge(gables) if gables else None), odd


def edge_facing(b, bearing):
    """外周の辺のうち、外向きの向きが方位 bearing（北=0、東=90）に一番近い辺の番号。"""
    p = np.asarray(b.footprint.exterior.coords)
    want = np.array([math.sin(math.radians(bearing)), math.cos(math.radians(bearing))])
    best, k = -2.0, 0
    for i in range(len(p) - 1):
        d = p[i + 1] - p[i]; L = float(np.linalg.norm(d))
        if L < 0.5:
            continue
        n = np.array([d[1], -d[0]]) / L
        if float(n @ want) > best:
            best, k = float(n @ want), i
    return k


def building_meshes(buildings, ground_z, avoid=None):
    """(名前, 頂点, 面, UV, 材質, 描画するか)。壁は当たり判定あり（1WALL）。
    canopy に "self": true があれば、外周線そのものが屋根だけの建物（ガソリンスタンドの屋根など）。"""
    out = []
    for b in buildings:
        fp = _clip(b, avoid)
        if fp.is_empty or fp.area < 2.0:
            continue
        if b.canopy and b.canopy.get("self"):
            out += _canopy(b, fp, ground_z, fp)
            out += _signs(b, fp, ground_z, base=float(b.canopy.get("z_m", 4.5)) - 3.0)
            continue
        ring = fp.exterior
        p = np.asarray(ring.coords)
        zg = np.asarray(ground_z(p[:, 0], p[:, 1]), float)
        c = fp.centroid
        zc = float(np.asarray(ground_z(np.atleast_1d(c.x), np.atleast_1d(c.y)), float)[0])
        z_low, floor = min(float(zg.min()), zc), max(float(zg.max()), zc)
        z0 = z_low - 0.3
        top = floor + b.height_m
        sh = b.height_m / max(b.storeys, 1)
        flat = b.roof.get("type", "flat") == "flat"
        wall_top = top + (PARAPET_H if flat else 0.0)
        shop = set(int(e) for e in b.shopfront)
        walls, shops = [], []
        for i in range(len(p) - 1):
            if i in shop:
                shops.append(_edge_quad(p[i], p[i + 1], floor, floor + min(SHOP_H, b.height_m), floor, SHOP_H))
                walls.append(_edge_quad(p[i], p[i + 1], z0, floor, floor, sh))
                walls.append(_edge_quad(p[i], p[i + 1], floor + min(SHOP_H, b.height_m), wall_top, floor, sh))
            else:
                walls.append(_edge_quad(p[i], p[i + 1], z0, wall_top, floor, sh))
        out.append((f"1WALL_bld_{b.id}", *_merge(walls), facade_mat(b), True))
        if shops:
            out.append((f"bld_{b.id}_shop", *_merge(shops), "shopfront", True))
        if flat:
            inner = fp.buffer(-PARAPET_H, join_style="mitre")
            parts = [_tri_poly(fp, top)]
            if not inner.is_empty and inner.geom_type == "Polygon":
                parts.append(_wall_quads(orient(inner, 1.0).exterior, top, top + PARAPET_H, top, 1.0, inward=True))
                cap = fp.difference(inner)
                for g in ([cap] if cap.geom_type == "Polygon" else list(getattr(cap, "geoms", []))):
                    parts.append(_tri_poly(g, top + PARAPET_H))
            out.append((f"bld_{b.id}_roof", *_merge(parts), roof_mat(b), True))
        else:
            roof, gables, odd = _pitched_roof(b, fp, top)
            out.append((f"bld_{b.id}_roof", *roof, roof_mat(b), True))
            if gables is not None:
                out.append((f"bld_{b.id}_gables", *gables, facade_mat(b), True))
            if odd is not None:
                out.append((f"bld_{b.id}_gable_end", *odd, gable_end_mat(b), True))
        out += _signs(b, fp, ground_z)
        if b.canopy:
            out += _canopy(b, orient(Polygon(_xy(b.canopy["points"])).buffer(0), 1.0), ground_z, fp)
    return out


def _floor(fp, ground_z):
    p = np.asarray(fp.exterior.coords)
    zg = np.asarray(ground_z(p[:, 0], p[:, 1]), float)
    c = fp.centroid
    zc = float(np.asarray(ground_z(np.atleast_1d(c.x), np.atleast_1d(c.y)), float)[0])
    return min(float(zg.min()), zc), max(float(zg.max()), zc)


def _signs(b, fp, ground_z, base=0.0):
    out = []
    p = np.asarray(fp.exterior.coords)
    floor = _floor(fp, ground_z)[1] + base
    for n, s in enumerate(b.signs):
        e = int(s.get("edge", 0)) % (len(p) - 1)
        a, bb = p[e], p[e + 1]
        d = (bb - a) / max(np.linalg.norm(bb - a), 1e-9); out_n = np.array([d[1], -d[0]])   # 反時計回りの辺の右＝外
        w = min(float(s.get("w_m", 3.0)), float(np.linalg.norm(bb - a)) - 0.2)
        m = (a + bb) / 2 + out_n * 0.05
        z1 = floor + float(s.get("z_m", 3.0)); z2 = z1 + float(s.get("h_m", 1.0))
        v = np.array([[*(m - d * w / 2), z1], [*(m + d * w / 2), z1], [*(m + d * w / 2), z2], [*(m - d * w / 2), z2]])
        out.append((f"bld_{b.id}_sign{n}", v, np.array([[0, 1, 2, 3]], np.int32),
                    np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float), f"sign_{b.id}_{n}", True))
    return out


def _canopy(b, cp, ground_z, fp):
    """屋根だけの構造（ガソリンスタンドの屋根）：厚さ CANOPY_T の板と、内側 1.5m の所に柱。"""
    z_low, floor = _floor(fp, ground_z)
    zc0 = floor + float(b.canopy.get("z_m", 4.5))
    parts = [_tri_poly(cp, zc0 + CANOPY_T), _wall_quads(cp.exterior, zc0, zc0 + CANOPY_T, zc0, CANOPY_T)]
    under = _tri_poly(cp, zc0)
    parts.append((under[0], under[1][:, ::-1], under[2]))
    out = [(f"bld_{b.id}_canopy", *_merge(parts), "wall_white", True)]
    from .town_objects import _prism
    inner = cp.buffer(-1.5)
    if not inner.is_empty and inner.geom_type == "Polygon":
        cols = [_prism(np.asarray(q), 0.25, z_low - 0.3, zc0) for q in np.asarray(inner.exterior.coords)[:-1]]
        out.append((f"1WALL_bld_{b.id}_columns", *_merge(cols), "pole_grey", True))
    return out


def _hex(c):
    h = c.lstrip("#")
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], float) / 255.0


def facade_image(wall, windows=True, w=256, h=192, seed=0):
    """壁の絵 1 枚＝横 BAY_W(4m)・縦 1 階ぶん。窓（濃い色のガラスと白い枠）が真ん中に1つ。"""
    rng = np.random.default_rng(seed)
    base = _hex(wall)[None, None, :] * (0.94 + 0.06 * rng.random((h, w, 1)))
    img = base.copy()
    if windows:
        x0, x1 = int(w * 0.30), int(w * 0.70); y0, y1 = int(h * 0.22), int(h * 0.62)   # 上が y=0（窓の下端は床から 0.9m）
        img[y0 - 4:y1 + 4, x0 - 4:x1 + 4] = [0.92, 0.92, 0.90]
        img[y0:y1, x0:x1] = [0.20, 0.25, 0.30]
        img[y0:y1, (x0 + x1) // 2 - 2:(x0 + x1) // 2 + 2] = [0.92, 0.92, 0.90]
    return Image.fromarray(np.clip(img * 255, 0, 255).astype(np.uint8))


def shopfront_image(w=256, h=192):
    img = np.full((h, w, 3), 0.85)
    img[int(h * 0.1):int(h * 0.95), int(w * 0.05):int(w * 0.95)] = [0.22, 0.28, 0.32]
    for x in (w // 3, 2 * w // 3):
        img[int(h * 0.1):int(h * 0.95), x - 3:x + 3] = [0.75, 0.75, 0.75]
    return Image.fromarray((img * 255).astype(np.uint8))


def write_textures(buildings, out_dir):
    """建物ごとの壁・屋根・看板の絵を書き、{材質の名前: "textures/xxx.png"} を返す。"""
    from .textures import _gray, sign_texture
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    mats = {}

    def save(name, im):
        im.save(out / f"{name}.png"); mats[name] = f"textures/{name}.png"
    save("shopfront", shopfront_image())
    for k, b in enumerate(buildings):
        if facade_mat(b) not in mats:
            save(facade_mat(b), facade_image(b.wall, b.windows, seed=k))
        if b.roof.get("gable_end") and gable_end_mat(b) not in mats:
            save(gable_end_mat(b), facade_image(b.roof["gable_end"]["wall"], False, seed=k))
        if roof_mat(b) not in mats:
            r = _hex(b.roof.get("color", "#707070"))
            g = _gray(128, 100 + k, 1.0, 0.08)
            save(roof_mat(b), Image.fromarray(np.clip(g.astype(float) * r[None, None, :], 0, 255).astype(np.uint8)))
        for n, s in enumerate(b.signs):
            w_px = int(np.clip(float(s.get("w_m", 3.0)) * 96, 96, 1024)); h_px = int(np.clip(float(s.get("h_m", 1.0)) * 96, 48, 512))
            save(f"sign_{b.id}_{n}", sign_texture(s["text"], s.get("bg", "#FFFFFF"), s.get("fg", "#000000"), w_px, h_px))
    return mats
