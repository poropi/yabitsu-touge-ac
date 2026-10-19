"""CSP（Custom Shaders Patch）の木：木パッケージ（zip）と木の一覧の書き出し。

.mesh の形式はターンパイクの例（treePine1.zip）から読み解いた：
u32 1, u8 1, u32 テクスチャ数, [u32 ハッシュ, u32 長さ, 名前]…,
そのあと部品の繰り返し [u32 diffuse番号, u32 normal番号, u32 index数, u16…, u32 頂点数, 頂点 44B…]。
座標は y が上で、高さ 0〜1・横 ±1 に正規化（実寸は tree.ini の HEIGHT/WIDTH）。
"""
import struct
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

VERT_DTYPE = np.dtype([("p", "<3f4"), ("n", "<3f4"), ("uv", "<2f4"), ("b0", "4u1"), ("b1", "4u1"), ("b2", "4u1")])
assert VERT_DTYPE.itemsize == 44


@dataclass
class Submesh:
    diffuse: int
    normal: int
    indices: np.ndarray
    verts: np.ndarray


def fnv1a(s: str) -> int:
    h = 0x811C9DC5
    for c in s.encode():
        h = ((h ^ c) * 0x01000193) & 0xFFFFFFFF
    return h


def write_mesh(textures, subs) -> bytes:
    """textures は名前か (ハッシュ, 名前)。ハッシュの作り方は不明なので、名前だけなら FNV-1a を入れる。"""
    out = [struct.pack("<IBI", 1, 1, len(textures))]
    for t in textures:
        h, name = t if isinstance(t, tuple) else (fnv1a(t), t)
        b = name.encode()
        out.append(struct.pack("<II", h, len(b)) + b)
    for s in subs:
        if len(s.verts) > 65535:
            raise ValueError(f"頂点数 {len(s.verts)} が上限 65535 を超えています")
        idx = np.asarray(s.indices, "<u2")
        out.append(struct.pack("<III", s.diffuse, s.normal, len(idx)) + idx.tobytes())
        out.append(struct.pack("<I", len(s.verts)) + np.asarray(s.verts, VERT_DTYPE).tobytes())
    return b"".join(out)


def read_mesh(data: bytes):
    """(テクスチャ [(ハッシュ, 名前)…], 部品…) を返す。最後の任意 12 バイト（ff ff ff ff…）は読み飛ばす。"""
    _, _, n = struct.unpack_from("<IBI", data, 0)
    o, tex = 9, []
    for _ in range(n):
        h, L = struct.unpack_from("<II", data, o)
        tex.append((h, data[o + 8:o + 8 + L].decode()))
        o += 8 + L
    subs = []
    while len(data) - o > 12:
        d, nm, ni = struct.unpack_from("<III", data, o); o += 12
        idx = np.frombuffer(data, "<u2", ni, o).copy(); o += 2 * ni
        nv, = struct.unpack_from("<I", data, o); o += 4
        v = np.frombuffer(data, VERT_DTYPE, nv, o).copy(); o += 44 * nv
        subs.append(Submesh(d, nm, idx, v))
    return tex, subs


def ini_text(ini: dict) -> str:
    return "".join(f"[{sec}]\n" + "".join(f"{k}={v}\n" for k, v in kv.items()) + "\n" for sec, kv in ini.items())


def write_package(path: Path, ini: dict, meshes: dict, textures: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("tree.ini", ini_text(ini))
        for name, b in {**meshes, **textures}.items():
            z.writestr(name, b)


def tree_line(pkg, x, y, z, angle, size) -> str:
    return f"tree: {pkg}; pos={x:.2f}, {y:.2f}, {z:.2f}; angle = {angle:.0f}; size = {size:.2f}"


def to_ac(xyz):
    """プロジェクトの座標（x 東, y 北, z 上）→ AC の座標 (x, z, -y)。"""
    xyz = np.asarray(xyz, float)
    return np.c_[xyz[:, 0], xyz[:, 2], -xyz[:, 1]]
