"""日本の家の部品集（Blender）。半間 M=0.91 m 刻みの部品を作り込み、家の輪郭（長さ×奥行き、正面の向き）に並べて 1 軒にする。
部品：壁（素材の凹凸つき）、腰窓・掃き出し窓・小窓（面格子）・玄関ドア（袖のガラス・庇・ポーチ・照明）、シャッターの箱とレール、
水切り、ベランダ（腰壁の手すり・笠木・床・排水）、屋根（切妻・寄棟、厚さ・破風板・軒天・棟）、軒樋・たて樋（金具）、基礎（換気口）、
室外機（ファンの格子）、電気のメーター、給湯器。
他の人のやり方（Buildify など、部品を作り込んで輪郭に並べる。2026-10-28 調べ）にならう。Blender の中で import して使う（one_house.py・street.py）。"""
import math, os, random
import bmesh, bpy
from mathutils import Matrix, Vector

M = 0.91           # 半間
SH = 2.9           # 階の高さ
FL = 0.5           # 基礎の高さ（地面から 1 階の床）
from pathlib import Path
ROOT = Path(__file__).resolve().parents[4]   # .claude/skills/japanese-house-kit/scripts/kit.py → プロジェクトの根
ACG = str(ROOT / "data" / "raw" / "free_assets" / "ambientcg")

# ---------------- 材質 ----------------
_mats = {}


def _lin(h):
    h = h.lstrip("#"); c = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    return [x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4 for x in c] + [1.0]


def _sock(col, ident):
    return next(s for s in col if s.identifier == ident)


def mat(name, color, rough=0.6, metal=0.0, tex=None, scale=1.0, tint=1.0, normal=1.0, spec=0.3, alpha=1.0, transmission=0.0, grime=0.0):
    """tex（ambientCG の素材）があれば、素材の明暗×color、法線の絵。座標は部品の中の位置（box 投影）、scale は 1 m あたりの素材の枚数。"""
    key = (name, tuple(color) if not isinstance(color, str) else color)
    if key in _mats:
        return _mats[key]
    col = _lin(color) if isinstance(color, str) else color
    m = bpy.data.materials.new(name); nt = m.node_tree; b = nt.nodes["Principled BSDF"]
    b.inputs["Roughness"].default_value = rough; b.inputs["Metallic"].default_value = metal
    b.inputs["Specular IOR Level"].default_value = spec
    if alpha < 1.0:
        b.inputs["Alpha"].default_value = alpha; m.blend_method = "BLEND" if hasattr(m, "blend_method") else None
    if transmission:
        b.inputs["Transmission Weight"].default_value = transmission
    if tex:
        tc = nt.nodes.new("ShaderNodeTexCoord"); mp = nt.nodes.new("ShaderNodeMapping"); mp.inputs["Scale"].default_value = (scale,) * 3
        nt.links.new(tc.outputs["Object"], mp.inputs["Vector"])
        im = nt.nodes.new("ShaderNodeTexImage"); im.image = bpy.data.images.load(f"{ACG}/{tex}/{tex}_1K-JPG_Color.jpg", check_existing=True)
        im.projection = "BOX"; im.projection_blend = 0.15; nt.links.new(mp.outputs["Vector"], im.inputs["Vector"])
        bw = nt.nodes.new("ShaderNodeRGBToBW"); nt.links.new(im.outputs["Color"], bw.inputs["Color"])
        ma = nt.nodes.new("ShaderNodeMath"); ma.operation = "MULTIPLY_ADD"; ma.inputs[1].default_value = 0.7 * tint; ma.inputs[2].default_value = 1.0 - 0.35 * tint
        nt.links.new(bw.outputs["Val"], ma.inputs[0])
        mx = nt.nodes.new("ShaderNodeMix"); mx.data_type = "RGBA"; mx.blend_type = "MULTIPLY"; mx.inputs["Factor"].default_value = 1.0
        _sock(mx.inputs, "A_Color").default_value = col; nt.links.new(ma.outputs["Value"], _sock(mx.inputs, "B_Color"))
        last = _sock(mx.outputs, "Result_Color")
        if grime:   # くすみ：ゆるい斑と、下ほど暗い（地面の跳ね返り・雨だれ）。壁がきれいすぎて作り物に見えた
            nz = nt.nodes.new("ShaderNodeTexNoise"); nz.inputs["Scale"].default_value = 1.5; nz.inputs["Detail"].default_value = 4.0
            nt.links.new(mp.outputs["Vector"], nz.inputs["Vector"])
            r1 = nt.nodes.new("ShaderNodeMapRange"); r1.inputs["To Min"].default_value = 1.0 - 0.18 * grime; r1.inputs["To Max"].default_value = 1.0
            nt.links.new(nz.outputs["Fac"], r1.inputs["Value"])
            sx = nt.nodes.new("ShaderNodeSeparateXYZ"); nt.links.new(tc.outputs["Object"], sx.inputs["Vector"])
            r2 = nt.nodes.new("ShaderNodeMapRange"); r2.inputs["From Min"].default_value = -0.6; r2.inputs["From Max"].default_value = 1.2
            r2.inputs["To Min"].default_value = 1.0 - 0.25 * grime; r2.inputs["To Max"].default_value = 1.0
            nt.links.new(sx.outputs["Z"], r2.inputs["Value"])
            m1 = nt.nodes.new("ShaderNodeMath"); m1.operation = "MULTIPLY"; nt.links.new(r1.outputs["Result"], m1.inputs[0]); nt.links.new(r2.outputs["Result"], m1.inputs[1])
            mx2 = nt.nodes.new("ShaderNodeMix"); mx2.data_type = "RGBA"; mx2.blend_type = "MULTIPLY"; mx2.inputs["Factor"].default_value = 1.0
            nt.links.new(last, _sock(mx2.inputs, "A_Color")); nt.links.new(m1.outputs["Value"], _sock(mx2.inputs, "B_Color")); last = _sock(mx2.outputs, "Result_Color")
        nt.links.new(last, b.inputs["Base Color"])
        nrm = f"{ACG}/{tex}/{tex}_1K-JPG_NormalGL.jpg"
        if os.path.exists(nrm) and normal:
            ni = nt.nodes.new("ShaderNodeTexImage"); ni.image = bpy.data.images.load(nrm, check_existing=True); ni.image.colorspace_settings.name = "Non-Color"
            ni.projection = "BOX"; ni.projection_blend = 0.15; nt.links.new(mp.outputs["Vector"], ni.inputs["Vector"])
            nm = nt.nodes.new("ShaderNodeNormalMap"); nm.inputs["Strength"].default_value = normal
            nt.links.new(ni.outputs["Color"], nm.inputs["Color"]); nt.links.new(nm.outputs["Normal"], b.inputs["Normal"])
    else:
        b.inputs["Base Color"].default_value = col
    _mats[key] = m
    return m


def palette(rnd, wall_hex=None, roof_hex=None):
    """1 軒ぶんの材質。"""
    kind = rnd.choice(["plaster", "plaster", "siding", "siding", "siding"])
    wall_c = wall_hex or rnd.choice(["#ECEAE4", "#E6E0D2", "#D9D2C2", "#C9C4BA", "#B8AE9C", "#8C8478", "#5A5650"])
    roof_c = roof_hex or rnd.choice(["#5A3C2C", "#3A3A3C", "#4C4840", "#2E3438", "#6A4A36"])   # 下から見上げると明るく写るので暗めに
    P = dict(
        wall=mat(f"wall_{kind}", wall_c, 0.85, tex="Plaster003" if kind == "plaster" else "WoodSiding011",
                 scale=0.6 if kind == "plaster" else 0.45, tint=0.25 if kind == "plaster" else 0.18, normal=0.6 if kind == "plaster" else 1.2, grime=1.0),
        base=mat("base", "#A9A79F", 0.9, tex="Concrete034", scale=0.6, tint=0.5, grime=1.5),
        roof=mat("roof", roof_c, 0.85, 0.0, tex="RoofingTiles013A" if rnd.random() < 0.5 else "CorrugatedSteel005", scale=0.8, tint=0.5, normal=0.25, spec=0.05),   # 照り返しと強い凹凸で、低い角度から空を映して白く飛んだ
        trim=mat("trim", "#EEEDEA" if rnd.random() < 0.6 else "#3C3632", 0.5),
        soffit=mat("soffit", "#E4E2DC", 0.7),
        sash=mat("sash_silver", "#A8ACB0", 0.3, 0.8) if rnd.random() < 0.55 else mat("sash_bronze", "#3A322B", 0.35, 0.6),
        glass=mat("glass", "#1C252C", 0.03, spec=1.0),
        curtain=mat("curtain", "#D8D6CE", 0.9),
        screen=mat("screen", "#2A2C2E", 0.8, alpha=0.55),
        frost=mat("frost", "#B8C0C4", 0.35),
        door=mat("door", rnd.choice(["#4A3A2E", "#6A5440", "#2E2E30", "#7A6A58"]), 0.45, tex="WoodSiding005", scale=0.8, tint=0.4, normal=0.4),
        gutter=mat("gutter", rnd.choice(["#E6E4DE", "#5A5A58", "#3A3430"]), 0.4),
        metal=mat("metal", "#8A8E92", 0.35, 0.7),
        ac=mat("ac", "#DCDCD6", 0.5),
        dark=mat("dark", "#151515", 0.8),
        lamp=mat("lamp", "#F2F0E6", 0.3),
    )
    return P


# ---------------- 形を組み立てる道具 ----------------
class Mesh:
    def __init__(self):
        self.parts = {}

    def _bm(self, m):
        return self.parts.setdefault(m, bmesh.new())

    def box(self, m, x0, x1, y0, y1, z0, z1, T=None):
        bm = self._bm(m)
        r = bmesh.ops.create_cube(bm, size=1.0)
        S = Matrix.Translation(((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2)) @ Matrix.Diagonal((abs(x1 - x0), abs(y1 - y0), abs(z1 - z0), 1))
        bmesh.ops.transform(bm, matrix=(T or Matrix()) @ S, verts=r["verts"])

    def cyl(self, m, p0, p1, r, seg=10, T=None):
        bm = self._bm(m); p0, p1 = Vector(p0), Vector(p1); d = p1 - p0
        res = bmesh.ops.create_cone(bm, cap_ends=True, segments=seg, radius1=r, radius2=r, depth=d.length)
        bmesh.ops.transform(bm, matrix=(T or Matrix()) @ Matrix.Translation((p0 + p1) / 2) @ d.to_track_quat("Z", "Y").to_matrix().to_4x4(), verts=res["verts"])

    def poly(self, m, pts, T=None, two=False):
        bm = self._bm(m); T = T or Matrix()
        vs = [bm.verts.new(T @ Vector(p)) for p in pts]; bm.faces.new(vs)
        if two:
            bm.faces.new([bm.verts.new(T @ Vector(p)) for p in pts[::-1]])

    def emit(self, name, M_world=None, bevel=0.012):
        out = []
        for m, bm in self.parts.items():
            bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-5)
            me = bpy.data.meshes.new(f"{name}_{m.name}"); bm.to_mesh(me); bm.free()
            o = bpy.data.objects.new(me.name, me); o.data.materials.append(m)
            if M_world is not None:
                o.matrix_world = M_world
            bpy.context.scene.collection.objects.link(o)
            if bevel and m.name.split(".")[0] not in ("glass", "screen", "curtain", "roof"):
                bv = o.modifiers.new("bevel", "BEVEL"); bv.width = bevel; bv.segments = 1; bv.limit_method = "ANGLE"
            out.append(o)
        return out


# ---------------- 壁の部品（正面の面は y=0、外は +y、u は面に沿った位置、z は床から） ----------------
def wall_with_opening(G, P, T, u0, w, ox0, ox1, oz0, oz1, h=SH):
    """幅 w の壁に、u0+ox0〜u0+ox1・高さ oz0〜oz1 の開口。壁の厚さ 0.12 m（内側は見えない）。"""
    t = 0.12
    if ox0 > 0: G.box(P["wall"], u0, u0 + ox0, -t, 0, 0, h, T)
    if ox1 < w: G.box(P["wall"], u0 + ox1, u0 + w, -t, 0, 0, h, T)
    if oz0 > 0: G.box(P["wall"], u0 + ox0, u0 + ox1, -t, 0, 0, oz0, T)
    if oz1 < h: G.box(P["wall"], u0 + ox0, u0 + ox1, -t, 0, oz1, h, T)


def sash_window(G, P, T, u, w, z0, z1, rnd, shutter=False, screen=True, kind="slide"):
    """アルミの引き違い窓：外枠（壁から 2 cm 出る）、障子 2 枚（召し合わせ）、ガラス（4 cm 奥）、カーテン、網戸（半分）、水切り、シャッターの箱とレール。"""
    f = 0.045; S = P["sash"]
    a, b = u - w / 2, u + w / 2
    for x0, x1, zz0, zz1 in ((a - f, a, z0 - f, z1 + f), (b, b + f, z0 - f, z1 + f), (a, b, z1, z1 + f), (a, b, z0 - f, z0)):
        G.box(S, x0, x1, -0.06, 0.02, zz0, zz1, T)
    G.box(P["glass"], a, b, -0.075, -0.065, z0, z1, T)
    if kind == "slide":
        G.box(S, u - 0.035, u + 0.035, -0.06, 0.0, z0, z1, T)                        # 召し合わせ
        for x0, x1 in ((a, a + 0.035), (b - 0.035, b)):
            G.box(S, x0, x1, -0.06, -0.01, z0, z1, T)
        for zz in (z0, z1 - 0.035):
            G.box(S, a, b, -0.06, -0.01, zz, zz + 0.035, T)
    if rnd.random() < 0.7:
        cw = rnd.uniform(0.25, 0.5) * w                                              # 片側に寄せたカーテン
        x = a if rnd.random() < 0.5 else b - cw
        G.box(P["curtain"], x, x + cw, -0.16, -0.15, z0 + 0.05, z1 - 0.05, T)
        if rnd.random() < 0.5:
            G.box(P["curtain"], a, b, -0.18, -0.17, z0 + 0.05, z1 - 0.05, T)       # レース
    if screen and kind == "slide":
        x = a if rnd.random() < 0.5 else u
        G.box(P["screen"], x + 0.02, x + w / 2 - 0.02, 0.005, 0.015, z0 + 0.02, z1 - 0.02, T)
    G.box(S, a - f - 0.03, b + f + 0.03, 0.0, 0.07, z0 - f - 0.035, z0 - f, T)       # 水切り
    if shutter:
        G.box(S, a - f - 0.02, b + f + 0.02, 0.0, 0.2, z1 + f, z1 + f + 0.28, T)    # シャッターの箱
        for x0 in (a - f - 0.02, b + f - 0.02):
            G.box(S, x0, x0 + 0.04, 0.0, 0.08, z0 - f, z1 + f, T)                    # レール


def small_window(G, P, T, u, w, z0, z1, rnd):
    """浴室・トイレの小窓：すりガラス、面格子。"""
    f = 0.04; S = P["sash"]; a, b = u - w / 2, u + w / 2
    for x0, x1, zz0, zz1 in ((a - f, a, z0 - f, z1 + f), (b, b + f, z0 - f, z1 + f), (a, b, z1, z1 + f), (a, b, z0 - f, z0)):
        G.box(S, x0, x1, -0.05, 0.02, zz0, zz1, T)
    G.box(P["frost"], a, b, -0.06, -0.05, z0, z1, T)
    if rnd.random() < 0.7:
        for k in range(5):
            x = a + (k + 0.5) * w / 5
            G.box(P["metal"], x - 0.012, x + 0.012, 0.06, 0.085, z0 - 0.03, z1 + 0.03, T)
        for zz in (z0 - 0.03, z1 + 0.01):
            G.box(P["metal"], a - 0.05, b + 0.05, 0.02, 0.085, zz, zz + 0.025, T)
    G.box(S, a - f - 0.02, b + f + 0.02, 0.0, 0.06, z0 - f - 0.03, z0 - f, T)


def entrance(G, P, T, u, rnd, ground_drop):
    """玄関：ドア（0.9×2.2）＋袖のすりガラス、庇、ポーチ（2 段）、照明、表札。"""
    dw = 0.9; a = u - 0.65
    G.box(P["sash"], a - 0.05, a + dw + 0.3 + 0.05, -0.08, 0.02, 0, 2.3, T)
    G.box(P["door"], a, a + dw, -0.07, -0.03, 0.02, 2.25, T)
    G.box(P["frost"], a + dw + 0.03, a + dw + 0.27, -0.07, -0.06, 0.3, 2.2, T)
    G.box(P["metal"], a + dw - 0.12, a + dw - 0.09, -0.03, 0.03, 0.9, 1.3, T)       # 取っ手
    G.box(P["trim"], a - 0.4, a + dw + 0.7, 0.0, 1.0, 2.45, 2.55, T)                # 庇
    G.box(P["base"], a - 0.4, a + dw + 0.7, 0.0, 1.3, -0.25, 0.0, T)                # ポーチ
    G.box(P["base"], a - 0.3, a + dw + 0.6, 1.3, 1.65, -FL - ground_drop, -0.25, T) # 段
    G.box(P["lamp"], a + dw + 0.4, a + dw + 0.55, 0.0, 0.1, 1.9, 2.1, T)
    G.box(P["dark"], a - 0.3, a - 0.1, 0.0, 0.02, 1.4, 1.55, T)                     # 表札


def balcony(G, P, T, u0, w, rnd, depth=0.91):
    """ベランダ：床、腰壁の手すり（壁と同じ仕上げか、アルミの縦格子）、笠木、排水のたて樋。z=0 は 2 階の床。"""
    G.box(P["base"], u0, u0 + w, 0.0, depth, -0.25, 0.0, T)
    solid = rnd.random() < 0.6
    if solid:
        G.box(P["wall"], u0, u0 + w, depth - 0.1, depth, -0.25, 1.0, T)
        for x in (u0, u0 + w - 0.1):
            G.box(P["wall"], x, x + 0.1, 0.0, depth, -0.25, 1.0, T)
        G.box(P["sash"], u0 - 0.02, u0 + w + 0.02, depth - 0.13, depth + 0.03, 1.0, 1.05, T)   # 笠木
    else:
        G.box(P["metal"], u0, u0 + w, depth - 0.06, depth, 0.95, 1.05, T)
        G.box(P["metal"], u0, u0 + w, depth - 0.06, depth, 0.05, 0.12, T)
        n = int(w / 0.11)
        for k in range(n + 1):
            x = u0 + k * w / n
            G.box(P["metal"], x - 0.012, x + 0.012, depth - 0.05, depth - 0.02, 0.1, 0.95, T)
        G.box(P["trim"], u0, u0 + w, depth - 0.02, depth, -0.25, 0.05, T)
    G.cyl(P["gutter"], (u0 + 0.15, depth + 0.05, -0.25), (u0 + 0.15, depth + 0.05, -SH - FL), 0.03, T=T)


# ---------------- 1 軒 ----------------
def facade_layout(nb, storey, face, rnd, has_balc):
    """面の割り付け：半間ごとの [(種類, 幅の半間数)]。"""
    out = []; left = nb
    while left > 0:
        if face == "front":
            if storey == 0 and not any(k == "door" for k, _ in out) and left <= nb - 1:
                out.append(("door", 2)); left -= 2; continue
            k = rnd.choices(["big", "koshi", "plain", "small"], [3, 3, 1, 1] if storey == 0 else [2, 4, 1, 1])[0]
        elif face == "back":
            k = rnd.choices(["koshi", "small", "plain"], [3, 3, 2])[0]
        else:
            k = rnd.choices(["small", "koshi", "plain"], [3, 2, 3])[0]
        wb = {"big": 2, "koshi": 2, "small": 1, "plain": 1, "door": 2}[k]
        wb = min(wb, left)
        if wb == 1 and k in ("big", "koshi"):
            k = "small"
        out.append((k, wb)); left -= wb
    return out


def build_house(name, L, W, rnd, roof="gable", storeys=2, ground_drop=0.0, wall_hex=None, roof_hex=None, origin=Matrix()):
    """L×W（m、半間に丸める）の家。正面は +y、原点は 1 階の床の面の中心。ground_drop は床から地面までの余分の高さ。"""
    P = palette(rnd, wall_hex, roof_hex)
    nx, ny = max(4, round(L / M)), max(4, round(W / M)); L, W = nx * M, ny * M
    G = Mesh()
    x0, x1, y0, y1 = -L / 2, L / 2, -W / 2, W / 2
    # 基礎（ところどころ換気口）
    G.box(P["base"], x0, x1, y0, y1, -FL - ground_drop - 0.3, 0.0)
    for k in range(int(L / 3)):
        x = x0 + 1.0 + k * 3
        G.box(P["dark"], x, x + 0.3, y1 - 0.01, y1 + 0.005, -0.3, -0.18)
    G.box(P["metal"], x0, x1, y0 - 0.015, y1 + 0.015, -0.02, 0.0)                      # 水切り
    has_balc = storeys >= 2 and rnd.random() < 0.85
    bal_u, bal_w = None, None
    if has_balc:
        bw = max(3, int(nx * rnd.uniform(0.5, 0.75))); bs = rnd.randint(0, nx - bw)
        bal_u, bal_w = x0 + bs * M, bw * M
    faces = {"front": (Matrix.Translation((x0, y1, 0)), nx), "back": (Matrix.Translation((x1, y0, 0)) @ Matrix.Rotation(math.pi, 4, "Z"), nx),
             "right": (Matrix.Translation((x1, y1, 0)) @ Matrix.Rotation(-math.pi / 2, 4, "Z"), ny),
             "left": (Matrix.Translation((x0, y0, 0)) @ Matrix.Rotation(math.pi / 2, 4, "Z"), ny)}
    for st in range(storeys):
        for face, (Tf, nb) in faces.items():
            T = Matrix.Translation((0, 0, st * SH)) @ Tf
            u = 0.0
            for kind, wb in facade_layout(nb, st, face, rnd, has_balc):
                w = wb * M; c = u + w / 2
                if kind == "plain":
                    G.box(P["wall"], u, u + w, -0.12, 0, 0, SH, T)
                elif kind == "door":
                    wall_with_opening(G, P, T, u, w, w / 2 - 0.75, w / 2 + 0.6, 0, 2.3); entrance(G, P, T, c, rnd, ground_drop)
                elif kind == "big":
                    ww = w - 0.2; z1 = 2.05
                    big_floor = st == 0 or (has_balc and face == "front")
                    z0 = 0.05 if big_floor else 0.9
                    wall_with_opening(G, P, T, u, w, (w - ww) / 2 - 0.05, (w + ww) / 2 + 0.05, z0 - 0.05, z1 + 0.05)
                    sash_window(G, P, T, c, ww, z0, z1, rnd, shutter=rnd.random() < 0.7)
                elif kind == "koshi":
                    ww = min(1.65, w - 0.2); z0, z1 = 0.9, 2.0
                    wall_with_opening(G, P, T, u, w, (w - ww) / 2 - 0.05, (w + ww) / 2 + 0.05, z0 - 0.05, z1 + 0.05)
                    sash_window(G, P, T, c, ww, z0, z1, rnd, shutter=rnd.random() < 0.35)
                else:   # small
                    ww = 0.6; z0, z1 = 1.25, 1.95
                    wall_with_opening(G, P, T, u, w, (w - ww) / 2 - 0.05, (w + ww) / 2 + 0.05, z0 - 0.05, z1 + 0.05)
                    small_window(G, P, T, c, ww, z0, z1, rnd)
                u += w
        # 階の間の帯（胴差しの見切り）
        if st > 0:
            G.box(P["trim"], x0 - 0.02, x1 + 0.02, y0 - 0.02, y1 + 0.02, st * SH - 0.04, st * SH + 0.04)
    if has_balc:
        balcony(G, P, Matrix.Translation((x0, y1, SH)), bal_u - x0, bal_w, rnd)
    top = storeys * SH
    roof_parts(G, P, L, W, top, roof, rnd)
    gutters(G, P, L, W, top, rnd, ground_drop)
    props(G, P, L, W, rnd, ground_drop)
    return G.emit(name, origin)


def roof_parts(G, P, L, W, top, kind, rnd):
    """屋根：勾配 4 寸、軒の出 0.6 m、ケラバ 0.45 m、厚さ 0.18 m、破風板、軒天、棟。"""
    ev, eg, t, pitch = 0.6, 0.45, 0.18, 0.4
    x0, x1, y0, y1 = -L / 2, L / 2, -W / 2, W / 2
    zr = top + (W / 2) * pitch
    ze = top - ev * pitch
    R = P["roof"]
    if kind == "hip":
        rx = L / 2 - W / 2                          # 棟の長さの半分
        X0, X1, Y0, Y1 = x0 - ev, x1 + ev, y0 - ev, y1 + ev
        A, B = (-rx, 0, zr), (rx, 0, zr)
        quads = [[(X0, Y1, ze), (X1, Y1, ze), B, A], [(X1, Y0, ze), (X0, Y0, ze), A, B]]
        tris = [[(X1, Y1, ze), (X1, Y0, ze), B], [(X0, Y0, ze), (X0, Y1, ze), A]]
        for q in quads + tris:   # 並びは下向きの面になるので逆にする（寄棟の屋根が裏向きで、軒天の色が見えて淡く写った）
            G.poly(R, q[::-1]); G.poly(P["soffit"], [(p[0], p[1], p[2] - t) for p in q])
        rim = [(X0, Y1), (X1, Y1), (X1, Y0), (X0, Y0)]
        for i in range(4):
            (a, b), (c, d) = rim[i], rim[(i + 1) % 4]
            G.poly(P["trim"], [(a, b, ze - t), (c, d, ze - t), (c, d, ze), (a, b, ze)])
        G.box(R, -rx - 0.05, rx + 0.05, -0.1, 0.1, zr - 0.03, zr + 0.08)
        for (sx, sy) in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            G.cyl(R, (sx * rx, 0, zr + 0.03), (sx * (L / 2 + ev), sy * (W / 2 + ev), ze + 0.03), 0.06, seg=6)
        return
    for s in (1, -1):
        yo = s * (W / 2 + ev)
        q = [(x0 - eg, 0, zr), (x1 + eg, 0, zr), (x1 + eg, yo, ze), (x0 - eg, yo, ze)]
        G.poly(R, q if s > 0 else q[::-1])
        G.poly(P["soffit"], [(p[0], p[1], p[2] - t) for p in (q[::-1] if s > 0 else q)])
        G.poly(P["trim"], [(x0 - eg, yo, ze - t), (x1 + eg, yo, ze - t), (x1 + eg, yo, ze), (x0 - eg, yo, ze)][:: (1 if s > 0 else -1)])
    for xx, s in ((x1, 1), (x0, -1)):
        G.poly(P["wall"], [(xx, -W / 2, top), (xx, W / 2, top), (xx, 0, zr)][:: (1 if s > 0 else -1)])
        for side in (1, -1):   # 破風板
            G.poly(P["trim"], [(xx + s * eg, 0, zr + 0.02), (xx + s * eg, side * (W / 2 + ev), ze + 0.02), (xx + s * eg, side * (W / 2 + ev), ze - t - 0.05),
                               (xx + s * eg, 0, zr - t - 0.05)][:: (1 if s * side > 0 else -1)], two=True)
        G.box(P["trim"], *sorted((xx, xx + s * 0.05)), -0.3, 0.3, top + (zr - top) * 0.35, top + (zr - top) * 0.6)
    G.box(R, x0 - eg, x1 + eg, -0.11, 0.11, zr - 0.03, zr + 0.09)


def gutters(G, P, L, W, top, rnd, ground_drop):
    ev = 0.6; ze = top - ev * 0.4 - 0.2
    for s in (1, -1):
        y = s * (W / 2 + ev + 0.06)
        G.cyl(P["gutter"], (-L / 2 - 0.5, y, ze), (L / 2 + 0.5, y, ze), 0.06, seg=8)
        x = (L / 2 - 0.25) * (1 if rnd.random() < 0.5 else -1)
        G.cyl(P["gutter"], (x, y, ze), (x, s * (W / 2 + 0.07), ze - 0.35), 0.035)
        G.cyl(P["gutter"], (x, s * (W / 2 + 0.07), ze - 0.35), (x, s * (W / 2 + 0.07), -FL - ground_drop), 0.035)
        for z in (0.5, 2.5, 4.5):
            G.box(P["metal"], x - 0.03, x + 0.03, *sorted((s * W / 2, s * (W / 2 + 0.1))), z, z + 0.03)


def props(G, P, L, W, rnd, ground_drop):
    """室外機（ファンの格子）、メーター、給湯器。裏と横に。"""
    zg = -FL - ground_drop
    for k in range(rnd.randint(1, 2)):
        x = rnd.uniform(-L / 2 + 0.6, L / 2 - 0.6); y = -W / 2 - 0.35
        G.box(P["ac"], x - 0.4, x + 0.4, y - 0.15, y + 0.15, zg + 0.1, zg + 0.65)
        G.cyl(P["dark"], (x - 0.1, y - 0.16, zg + 0.38), (x - 0.1, y - 0.152, zg + 0.38), 0.18, seg=16)
        G.box(P["base"], x - 0.45, x + 0.45, y - 0.2, y + 0.2, zg, zg + 0.1)
    x = L / 2 + 0.01
    G.box(P["ac"], x, x + 0.2, -0.5, 0.0, 0.8, 1.5)                                  # 給湯器
    G.box(P["metal"], x, x + 0.12, 0.6, 0.85, 0.9, 1.2)                              # メーター


# ---------------- 植物（生け垣・庭木） ----------------
def leaf_mat(name="leaf", color="#2F4A24"):
    """葉：濃淡の斑（Voronoi）と細かい凹凸（Noise の Bump）。"""
    key = (name, color)
    if key in _mats:
        return _mats[key]
    m = bpy.data.materials.new(name); nt = m.node_tree; b = nt.nodes["Principled BSDF"]
    b.inputs["Roughness"].default_value = 0.75; b.inputs["Specular IOR Level"].default_value = 0.25
    tc = nt.nodes.new("ShaderNodeTexCoord")
    vo = nt.nodes.new("ShaderNodeTexVoronoi"); vo.inputs["Scale"].default_value = 18.0; nt.links.new(tc.outputs["Object"], vo.inputs["Vector"])
    ramp = nt.nodes.new("ShaderNodeValToRGB"); nt.links.new(vo.outputs["Distance"], ramp.inputs["Fac"])
    c = _lin(color); ramp.color_ramp.elements[0].color = [x * 1.35 for x in c[:3]] + [1]; ramp.color_ramp.elements[1].color = [x * 0.45 for x in c[:3]] + [1]
    nt.links.new(ramp.outputs["Color"], b.inputs["Base Color"])
    nz = nt.nodes.new("ShaderNodeTexNoise"); nz.inputs["Scale"].default_value = 40.0; nt.links.new(tc.outputs["Object"], nz.inputs["Vector"])
    bp = nt.nodes.new("ShaderNodeBump"); bp.inputs["Strength"].default_value = 0.8; nt.links.new(nz.outputs["Fac"], bp.inputs["Height"]); nt.links.new(bp.outputs["Normal"], b.inputs["Normal"])
    _mats[key] = m
    return m


def _blob(bm, c, r, rnd, squash=0.8):
    res = bmesh.ops.create_icosphere(bm, subdivisions=2, radius=r)
    for v in res["verts"]:
        d = v.co.normalized(); k = 1.0 + 0.28 * (rnd.random() - 0.5) + 0.12 * math.sin(7 * d.x + 3 * d.y) * math.cos(5 * d.z)
        v.co = Vector((v.co.x * k, v.co.y * k, v.co.z * k * squash)) + Vector(c)


def plants(name, items, rnd, color="#2F4A24"):
    """items = [(種類, x, y, z, 大きさ)]。種類 hedge（生け垣の 1 m 分）/ shrub（低木）/ tree（庭木）。1 つの部品にまとめる。"""
    me = bpy.data.meshes.new(name); bm = bmesh.new(); trunk = bmesh.new()
    for kind, x, y, z, s in items:
        if kind == "hedge":
            for k in range(6):
                _blob(bm, (x + rnd.uniform(-0.5, 0.5), y + rnd.uniform(-0.15, 0.15), z + rnd.uniform(0.35, 1.05) * s), rnd.uniform(0.3, 0.42) * s, rnd, 0.9)
        elif kind == "shrub":
            for k in range(3):
                _blob(bm, (x + rnd.uniform(-0.3, 0.3) * s, y + rnd.uniform(-0.3, 0.3) * s, z + rnd.uniform(0.3, 0.6) * s), rnd.uniform(0.35, 0.55) * s, rnd)
        else:
            h = rnd.uniform(2.5, 4.5) * s
            r = bmesh.ops.create_cone(trunk, cap_ends=True, segments=6, radius1=0.09 * s, radius2=0.05 * s, depth=h * 0.7)
            bmesh.ops.translate(trunk, vec=(x, y, z + h * 0.35), verts=r["verts"])
            for k in range(rnd.randint(5, 9)):
                _blob(bm, (x + rnd.uniform(-0.9, 0.9) * s, y + rnd.uniform(-0.9, 0.9) * s, z + h * rnd.uniform(0.55, 1.0)), rnd.uniform(0.6, 1.0) * s, rnd, 0.75)
    bm.to_mesh(me); bm.free()
    o = bpy.data.objects.new(name, me); o.data.materials.append(leaf_mat("leaf", color)); bpy.context.scene.collection.objects.link(o)
    for p in me.polygons: p.use_smooth = True
    me2 = bpy.data.meshes.new(name + "_trunk"); trunk.to_mesh(me2); trunk.free()
    o2 = bpy.data.objects.new(name + "_trunk", me2); o2.data.materials.append(mat("bark", "#3A2E24", 0.9)); bpy.context.scene.collection.objects.link(o2)
    return o


# ---------------- Codex で作ったテクスチャ（assets/house_kit、gen_textures.py） ----------------
HK = ROOT / "assets" / "house_kit"


def img_mat(name, size_m=(2.0, 2.0), rough=0.9, normal=1.0, spec=0.1, tint=None):
    """assets/house_kit/<name>.png（色）と <name>_nm.png（法線）。部品の中の位置で box 投影、絵 1 枚が size_m（横, 縦）m。"""
    key = ("img", name, size_m, tint)
    if key in _mats:
        return _mats[key]
    m = bpy.data.materials.new(f"hk_{name}"); nt = m.node_tree; b = nt.nodes["Principled BSDF"]
    b.inputs["Roughness"].default_value = rough; b.inputs["Specular IOR Level"].default_value = spec
    tc = nt.nodes.new("ShaderNodeTexCoord"); mp = nt.nodes.new("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = (1 / size_m[0], 1 / size_m[0], 1 / size_m[1]); nt.links.new(tc.outputs["Object"], mp.inputs["Vector"])
    im = nt.nodes.new("ShaderNodeTexImage"); im.image = bpy.data.images.load(str(HK / f"{name}.png"), check_existing=True)
    im.projection = "BOX"; im.projection_blend = 0.2; nt.links.new(mp.outputs["Vector"], im.inputs["Vector"])
    out = im.outputs["Color"]
    if tint:
        mx = nt.nodes.new("ShaderNodeMix"); mx.data_type = "RGBA"; mx.blend_type = "MULTIPLY"; mx.inputs["Factor"].default_value = 1.0
        nt.links.new(out, _sock(mx.inputs, "A_Color")); _sock(mx.inputs, "B_Color").default_value = _lin(tint); out = _sock(mx.outputs, "Result_Color")
    nt.links.new(out, b.inputs["Base Color"])
    if normal and (HK / f"{name}_nm.png").exists():
        ni = nt.nodes.new("ShaderNodeTexImage"); ni.image = bpy.data.images.load(str(HK / f"{name}_nm.png"), check_existing=True)
        ni.image.colorspace_settings.name = "Non-Color"; ni.projection = "BOX"; ni.projection_blend = 0.2
        nt.links.new(mp.outputs["Vector"], ni.inputs["Vector"])
        nm = nt.nodes.new("ShaderNodeNormalMap"); nm.inputs["Strength"].default_value = normal
        nt.links.new(ni.outputs["Color"], nm.inputs["Color"]); nt.links.new(nm.outputs["Normal"], b.inputs["Normal"])
    _mats[key] = m
    return m
