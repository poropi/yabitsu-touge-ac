"""Codex が作った部品（data/processed/landmarks/<asset-id>/<名前>.npz と .png）を、場所の座標（Frame の a・b）に置く。

npz：v（m、+Z 上、正面 −Y、原点＝根元の中央）、f（三角形）、uv。uv_top＝true の部品は v=0 が画像の上で書かれているので、
このコースの決まり（v=0 が画像の下。kn5 で 1−v）へ直す。
置き方（items の 1 つ）：{"asset": "<asset-id>", "name": "<npz の名前>", "a", "b", "yaw_deg", "dz", "scale", "collide", "z", "tex"}。
z を書いた物は、場所の高さの代わりにその高さ（m）に置く。
tex：同じ形に貼る別の画像（<tex>.png。標識の面の絵・柱の色違いなど。県道 70 号の r70_* は「<名前>_face_<内容>.png」が丸ごとの差し替え画像）。
yaw_deg＝0 で、部品の正面（−Y）が場所の −b を向き、+X が +a を向く。yaw_deg は +a から +b へ回す向きが正。
collide：部品の座標の箱 [[x0, x1], [y0, y1], 高さ] の並び（当たり判定。描かない）。
"""
import json
from pathlib import Path

import numpy as np

from . import config

DIR = config.DATA_PROCESSED / "landmarks"


def exists(asset, name):
    return (DIR / asset / f"{name}.npz").exists()


def load(asset, name, uv_top=True):
    d = np.load(DIR / asset / f"{name}.npz")
    uv = np.array(d["uv"], float)
    if uv_top:
        uv[:, 1] = 1.0 - uv[:, 1]
    return np.asarray(d["v"], float), np.asarray(d["f"], np.int64), uv


def material(asset, name, tex=None):
    return f"lm_{asset.replace('nanohanadai_', 'nh_')}_{tex or name}"


def texture(asset, name, tex=None):
    """部品の画像（<tex>.png か <名前>.png。無ければ部品のフォルダの最初の png）。"""
    p = DIR / asset / f"{tex or name}.png"
    if p.exists():
        return p
    pngs = sorted((DIR / asset).glob("*.png"))
    return pngs[0] if pngs else None


def _axes(f, yaw_deg):
    t = np.radians(yaw_deg)
    ux = np.cos(t) * f.u + np.sin(t) * f.n       # 部品の +X
    uy = -np.sin(t) * f.u + np.cos(t) * f.n      # 部品の +Y（正面の反対）
    return ux, uy


def place(items, f, z0, prefix):
    """items を置いた (名前, 頂点, 面, uv, 材質, 描画するか) の並び。z0：場所の高さ（a, b → z の関数か数）。"""
    out = []
    for k, it in enumerate(items):
        if not exists(it["asset"], it["name"]):
            continue
        v, fc, uv = load(it["asset"], it["name"], it.get("uv_top", True))
        s = it.get("scale", 1.0)
        ux, uy = _axes(f, it.get("yaw_deg", 0.0))
        c = f.xy(it["a"], it["b"])
        if "z" in it:                   # 道の縁など、場所の高さと違う所（絶対の高さ m）
            z = float(it["z"]) + it.get("dz", 0.0)
        else:
            z = (z0(it["a"], it["b"]) if callable(z0) else float(z0)) + it.get("dz", 0.0)
        w = lambda q: np.c_[c[0] + s * (q[:, 0] * ux[0] + q[:, 1] * uy[0]), c[1] + s * (q[:, 0] * ux[1] + q[:, 1] * uy[1]), z + s * q[:, 2]]
        tag = f"{prefix}_{k:02d}_{it['name']}"
        out.append((tag, w(v), fc, uv, material(it["asset"], it["name"], it.get("tex")), True))
        for j, ((x0, x1), (y0, y1), h) in enumerate(it.get("collide", [])):
            q = np.array([[x0, y0, -0.3], [x1, y0, -0.3], [x1, y1, -0.3], [x0, y1, -0.3]])
            bv = w(np.r_[q, q + [0, 0, h + 0.3]])
            bf = np.array([[0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7], [4, 5, 6, 7]])
            out.append((f"1WALL_{tag}_c{j}", bv, bf, bv[:, :2], "concrete", False))
    return out


def place_ll(items, z_fn, prefix, ground_fn=None):
    """緯度経度で置く（ヤビツ峠の Codex の部品。data/manual/s1_pass.json の landmarks）。
    items の 1 つ：{"asset", "name", "ll": [緯度, 経度], "bearing": 正面（−Y）の向き（北から時計回りの度）, "dz", "z_at": [[緯度, 経度], ...],
    "collide", "z_from"}。高さは z_at の点（無ければ ll）の z_fn(x, y) のいちばん低い所＋dz。建物は部品ごとに ll がずれても z_at を同じにしてそろえる。
    z_from＝"ground" の物（山の斜面の階段）は ground_fn（地形の高さ）を使う（z_fn は近くの歩道の高さに寄せるので、斜面に埋まる）。"""
    from . import geo
    from .start_lot import Frame
    f = Frame((0.0, 0.0), 0.0)   # a＝東、b＝北
    out = []
    for k, it in enumerate(items):
        x, y = (float(v) for v in geo.to_local(*it["ll"]))
        zp = np.asarray(geo.to_local(*np.asarray(it.get("z_at", [it["ll"]]), float).T), float).T
        fn = ground_fn if it.get("z_from") == "ground" and ground_fn is not None else z_fn
        z = float(it["z"]) if "z" in it else float(np.min(fn(zp[:, 0], zp[:, 1])))   # z を書いた物（道の縁の高さで置く県道 70 号の物）はその高さ
        out += place([{**it, "a": x, "b": y, "yaw_deg": 180.0 - it.get("bearing", 180.0), "z": z}], f, z, f"{prefix}{k:03d}")
    return out


GAME_MAX_PX = 1024   # ゲームへ写す画像の 1 辺の上限。kn5 は無圧縮の DDS なので、2048 が 19 枚で kn5 が約 0.5 GB 増えた（峠、2026-10-31）


def copy_texture(src, dst, max_px=GAME_MAX_PX):
    """部品の画像をゲームの組み立てへ写す。1 辺が max_px より大きければ縮める（元の data/processed/landmarks は変えない）。"""
    from PIL import Image
    im = Image.open(src)
    if max(im.size) > max_px:
        k = max_px / max(im.size)
        im.resize((max(1, round(im.width * k)), max(1, round(im.height * k))), Image.LANCZOS).save(dst)
    else:
        import shutil
        shutil.copy2(src, dst)


def materials(items):
    """{材質名: (画像のパス, 透過か)}（置いた部品の分だけ）。"""
    out = {}
    for it in items:
        if exists(it["asset"], it["name"]):
            out[material(it["asset"], it["name"], it.get("tex"))] = (texture(it["asset"], it["name"], it.get("tex")), it.get("alpha", False))
    return out
