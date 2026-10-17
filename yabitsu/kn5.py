"""manifest から AC の .kn5 を直接書き出す（ksEditor を使わない）。

形式は実際の kn5（3DSimED / ksEditor 製）を読んで確かめた：
バージョン 5、テクスチャは DDS、ダミーの前方はローカル +Z、行列は行ベクトル形式（4行目が位置）、
UV の V は上下反転、三角形は右手系の巻き順で法線側が表。
"""
import io
import json
import math
import struct
from pathlib import Path

import numpy as np
from PIL import Image

from .acdata import blender_to_ac
from .config import assert_writable

VERSION = 5
NODE_BASE, NODE_MESH = 1, 2
C = np.array([blender_to_ac(*e) for e in np.eye(3)], float).T  # Blender → AC の回転（列ベクトル形式）

# ksPerPixel の値は実在のコースでよく使われる標準値（S1 の見た目確認で調整する）
MATERIAL_PROPS = {"ksAmbient": 0.5, "ksDiffuse": 0.6, "ksSpecular": 0.05, "ksSpecularEXP": 10.0,
                  "ksEmissive": 0.0, "ksAlphaRef": 0.0}
ALPHA_REF = 0.3


# ---------- DDS（無圧縮 32bit BGRA、ミップマップ付き） ----------

def dds_bytes(img: Image.Image) -> bytes:
    img = img.convert("RGBA")
    w, h = img.size
    levels = [img]
    while levels[-1].size != (1, 1):
        lw, lh = levels[-1].size
        levels.append(levels[-1].resize((max(1, lw // 2), max(1, lh // 2)), Image.BOX))
    flags = 0x1 | 0x2 | 0x4 | 0x8 | 0x1000 | 0x20000  # CAPS HEIGHT WIDTH PITCH PIXELFORMAT MIPMAPCOUNT
    pixfmt = struct.pack("<8I", 32, 0x41, 0, 32, 0x00FF0000, 0x0000FF00, 0x000000FF, 0xFF000000)
    caps = 0x1000 | 0x400000 | 0x8  # TEXTURE MIPMAP COMPLEX
    header = struct.pack("<7I", 124, flags, h, w, w * 4, 0, len(levels)) + b"\0" * 44 + pixfmt \
        + struct.pack("<5I", caps, 0, 0, 0, 0)
    return b"DDS " + header + b"".join(l.tobytes("raw", "BGRA") for l in levels)


# ---------- 書き出し ----------

class _Out:
    def __init__(self):
        self.b = io.BytesIO()

    def pack(self, fmt, *v):
        self.b.write(struct.pack("<" + fmt, *v))

    def string(self, s):
        e = s.encode("utf-8"); self.pack("i", len(e)); self.b.write(e)

    def raw(self, data):
        self.b.write(data)


def _triangles(faces):
    faces = np.asarray(faces)
    tris = np.concatenate([faces[:, [0, k, k + 1]] for k in range(1, faces.shape[1] - 1)])
    ok = (tris[:, 0] != tris[:, 1]) & (tris[:, 1] != tris[:, 2]) & (tris[:, 0] != tris[:, 2])
    return tris[ok]


def _normals(pos, tris):
    a, b, c = pos[tris[:, 0]], pos[tris[:, 1]], pos[tris[:, 2]]
    fn = np.cross(b - a, c - a)  # 長さ = 面積の2倍 → 面積で重み付け
    n = np.zeros_like(pos)
    for k in range(3):
        np.add.at(n, tris[:, k], fn)
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    return np.where(ln > 1e-12, n / np.maximum(ln, 1e-12), [0.0, 1.0, 0.0])


def _tangents(pos, uv, nrm, tris):
    a, b, c = (tris[:, k] for k in range(3))
    e1, e2 = pos[b] - pos[a], pos[c] - pos[a]
    d1, d2 = uv[b] - uv[a], uv[c] - uv[a]
    r = d1[:, 0] * d2[:, 1] - d2[:, 0] * d1[:, 1]
    r = np.where(np.abs(r) > 1e-12, 1.0 / np.where(np.abs(r) > 1e-12, r, 1.0), 0.0)
    ft = (e1 * d2[:, 1:2] - e2 * d1[:, 1:2]) * r[:, None]
    t = np.zeros_like(pos)
    for k in range(3):
        np.add.at(t, tris[:, k], ft)
    t -= nrm * np.sum(t * nrm, 1, keepdims=True)
    ln = np.linalg.norm(t, axis=1, keepdims=True)
    fallback = np.cross(nrm, [0.0, 0.0, 1.0]); fallback[np.linalg.norm(fallback, axis=1) < 1e-6] = [1.0, 0.0, 0.0]
    fallback /= np.linalg.norm(fallback, axis=1, keepdims=True)
    return np.where(ln > 1e-9, t / np.maximum(ln, 1e-9), fallback)


def _write_mesh(o, name, pos, nrm, uv, tan, tris, material_id, renderable):
    o.pack("i", NODE_MESH); o.string(name); o.pack("i?", 0, True)
    o.pack("???", renderable, renderable, not renderable)  # 影を落とす / 見える / 透明（隠し壁は実例に合わせる）
    v = np.hstack([pos, nrm, uv, tan]).astype("<f4")
    o.pack("i", len(v)); o.raw(v.tobytes())
    idx = tris.astype("<u2").ravel()
    o.pack("i", len(idx)); o.raw(idx.tobytes())
    center = (pos.min(0) + pos.max(0)) / 2
    radius = float(np.linalg.norm(pos - center, axis=1).max())
    o.pack("ii", material_id, 0)                 # マテリアル番号、レイヤー
    o.pack("ff", 0.0, 0.0)                       # LOD in/out（0 = 常に表示）
    o.pack("4f", *center, radius * 1.0001 + 1e-4)
    o.pack("?", renderable)


def _dummy_matrix(location, rotation_z):
    cz, sz = math.cos(rotation_z), math.sin(rotation_z)
    rb = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    m = np.eye(4)
    m[:3, :3] = (C @ rb @ C.T).T     # 行 i = ローカル軸 i の向き（AC 座標）
    m[3, :3] = C @ np.asarray(location, float)
    return m


def write_from_manifest(manifest_path, group, out_path) -> Path:
    out_path = assert_writable(out_path)
    mpath = Path(manifest_path); root = mpath.parent
    m = json.loads(mpath.read_text(encoding="utf-8"))
    g = m["groups"][group]
    mat_names = list(dict.fromkeys(m["meshes"][n]["material"] for n in g["meshes"]))

    textures = {}
    for name in mat_names:
        spec = m["materials"][name]
        if spec.get("texture"):
            img = Image.open(root / spec["texture"])
            textures[Path(spec["texture"]).stem + ".dds"] = dds_bytes(img)
        else:
            rgb = tuple(int(round(c * 255)) for c in (spec.get("color") or (1.0, 0.0, 1.0)))
            textures[f"{name}.dds"] = dds_bytes(Image.new("RGBA", (4, 4), (*rgb, 255)))
    tex_of = {n: (Path(m["materials"][n]["texture"]).stem if m["materials"][n].get("texture") else n) + ".dds"
              for n in mat_names}

    o = _Out()
    o.raw(b"sc6969"); o.pack("i", VERSION)
    o.pack("i", len(textures))
    for tname, data in textures.items():
        o.pack("i", 1); o.string(tname); o.pack("i", len(data)); o.raw(data)
    o.pack("i", len(mat_names))
    for name in mat_names:
        spec = m["materials"][name]; alpha = bool(spec.get("alpha"))
        o.string(name); o.string("ksPerPixelAT" if alpha else "ksPerPixel")
        o.pack("B?i", 0, alpha, 0)  # ブレンド=不透明、アルファテスト、深度=通常
        props = {**MATERIAL_PROPS, **({"ksAlphaRef": ALPHA_REF} if alpha else {})}
        o.pack("i", len(props))
        for pname, val in props.items():
            o.string(pname); o.pack("10f", val, *([0.0] * 9))
        o.pack("i", 1); o.string("txDiffuse"); o.pack("i", 0); o.string(tex_of[name])

    children = []
    for name in g["meshes"]:
        info = m["meshes"][name]; spec = m["materials"][info["material"]]
        d = np.load(root / info["file"])
        pos = d["verts"] @ C.T
        uv = np.c_[d["uvs"][:, 0], 1.0 - d["uvs"][:, 1]]
        tris = _triangles(d["faces"])
        nrm = _normals(pos, tris); tan = _tangents(pos, uv, nrm, tris)
        mid = mat_names.index(info["material"])
        children.append((name, pos, nrm, uv, tan, tris, mid, info["renderable"]))
        if spec.get("two_sided"):
            children.append((f"{name}_back", pos, -nrm, uv, tan, tris[:, ::-1], mid, info["renderable"]))
    empties = [m["empties"][k] for k in g["empties"]]

    o.pack("i", NODE_BASE); o.string(group); o.pack("i?", len(children) + len(empties), True)
    o.pack("16f", *np.eye(4).ravel())
    for c in children:
        _write_mesh(o, *c)
    for e in empties:
        o.pack("i", NODE_BASE); o.string(e["name"]); o.pack("i?", 0, True)
        o.pack("16f", *_dummy_matrix(e["location"], e["rotation_z"]).ravel())

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(o.b.getvalue())
    return out_path


# ---------- 読み込み（書き出した kn5 の確認用） ----------

class _In:
    def __init__(self, data):
        self.d, self.p = data, 0

    def unpack(self, fmt):
        fmt = "<" + fmt; v = struct.unpack_from(fmt, self.d, self.p); self.p += struct.calcsize(fmt); return v

    def one(self, fmt):
        return self.unpack(fmt)[0]

    def string(self):
        n = self.one("i"); s = self.d[self.p:self.p + n].decode("utf-8", "replace"); self.p += n; return s

    def take(self, n):
        b = self.d[self.p:self.p + n]; self.p += n; return b


def _read_node(r):
    cls = r.one("i"); name = r.string(); nch = r.one("i"); active = r.one("?")
    node = {"cls": cls, "name": name, "active": active}
    if cls == NODE_BASE:
        node["matrix"] = np.array(r.unpack("16f")).reshape(4, 4)
    elif cls == NODE_MESH:
        node["cast_shadows"], node["visible"], node["transparent"] = r.unpack("???")
        nv = r.one("i"); node["vertices"] = np.frombuffer(r.take(nv * 44), "<f4").reshape(nv, 11)
        ni = r.one("i"); node["indices"] = np.frombuffer(r.take(ni * 2), "<u2")
        node["material"], node["layer"] = r.unpack("ii")
        node["lod"] = r.unpack("ff")
        *center, radius = r.unpack("4f"); node["bsphere_center"], node["bsphere_radius"] = tuple(center), radius
        node["renderable"] = r.one("?")
    else:
        raise ValueError(f"未対応のノード種類 {cls}（{name}）")
    node["children"] = [_read_node(r) for _ in range(nch)]
    return node


def read_kn5(path) -> dict:
    r = _In(Path(path).read_bytes())
    if r.take(6) != b"sc6969":
        raise ValueError(f"{path} は kn5 ではありません")
    version = r.one("i")
    if version > 5:
        r.one("i")
    textures = {}
    for _ in range(r.one("i")):
        r.one("i"); name = r.string(); textures[name] = r.take(r.one("i"))
    materials = []
    for _ in range(r.one("i")):
        mat = {"name": r.string(), "shader": r.string()}
        mat["blend"], mat["alpha_tested"], mat["depth"] = r.unpack("B?i")
        mat["props"] = {}
        for _ in range(r.one("i")):
            pname = r.string(); mat["props"][pname] = r.unpack("10f")[0]
        mat["textures"] = [(r.string(), r.one("i"), r.string()) for _ in range(r.one("i"))]
        materials.append(mat)
    root = _read_node(r)
    if r.p != len(r.d):
        raise ValueError(f"{path}: 末尾に {len(r.d) - r.p} バイト余りがあります")
    return {"version": version, "textures": textures, "materials": materials, "root": root}
