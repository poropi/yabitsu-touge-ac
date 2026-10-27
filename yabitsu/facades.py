"""家の壁と屋根の絵を、写真の素材（ambientCG、CC0。data/raw/free_assets/ambientcg）から作る。
素材の色の明暗と、法線の絵から焼き込んだ凹凸の陰（AC の ksPerPixel は法線の絵を使わないので）を「下地」にし、壁・屋根の色を掛ける。
壁の絵は横 FACADE_BAYS 区画（1 区画 BAY_W m）× 縦 1 階ぶん。区画ごとに窓の種類を変え、同じ窓が等間隔に並ばないようにする。
素材が無いとき（素材を落としていない環境）は、ゆらぎだけの下地にする。"""
import hashlib

import numpy as np
from PIL import Image

from . import config

ACG = config.DATA_RAW / "free_assets" / "ambientcg"
BAY_W = 4.0            # 1 区画の横幅（buildings.BAY_W と同じ）
FACADE_BAYS = 4        # 壁の絵 1 枚の区画の数
STOREY_M = 3.0         # 壁の絵の縦 1 枚＝1 階ぶんの高さ（実際の階の高さに合わせて縦に伸び縮みする）
PX_PER_M = 64
# 壁の仕上げ：(素材, 素材 1 枚の大きさ m, 色むらの強さ, 凹凸の陰の強さ)。羽目板＝横張りの窯業・金属サイディング、吹き付け＝モルタル、トタン＝波板。
# 色むらを強くすると木目・しみに見える（試しの絵で羽目板が板張りの小屋に見えた）ので、塗装の壁は陰を主にする
CLADDING = {"siding": ("WoodSiding011", 2.0, 0.15, 1.0), "plaster": ("Plaster003", 2.0, 0.1, 0.7), "tin": ("CorrugatedSteel005", 1.6, 0.4, 1.2)}
ROOFING = ("RoofingTiles003", 2.0, 0.6, 0.9)   # スレート（化粧スレートの横の段）


def _hex(c):
    h = c.lstrip("#")
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], float) / 255.0


def _seed(*keys):
    return int(hashlib.md5("|".join(map(str, keys)).encode()).hexdigest()[:8], 16)


def detail(asset, tile_m, w_m, h_m, lum_k=1.0, shade_k=1.0, px_per_m=PX_PER_M):
    """素材の明暗（平均 1）を w_m × h_m の大きさに敷き詰めた配列（h, w）。凹凸の陰は法線の絵を左上前からの光で焼き込む。"""
    w, h = int(round(w_m * px_per_m)), int(round(h_m * px_per_m))
    d = ACG / asset
    col, nrm = d / f"{asset}_1K-JPG_Color.jpg", d / f"{asset}_1K-JPG_NormalGL.jpg"
    if not col.exists():
        rng = np.random.default_rng(_seed(asset))
        return 1.0 + 0.04 * (rng.random((h, w)) - 0.5)
    lum = np.asarray(Image.open(col).convert("L"), float)
    lum = lum / max(lum.mean(), 1e-6)
    lum = 1.0 + lum_k * (lum - 1.0)
    if nrm.exists():
        n = np.asarray(Image.open(nrm).convert("RGB"), float) / 127.5 - 1.0
        light = np.array([-0.35, 0.45, 0.82]); light /= np.linalg.norm(light)   # 絵の左上・手前からの光（GL は y が上）
        shade = np.clip(n @ light, 0, 1); shade = shade / max(shade.mean(), 1e-6)
        lum = lum * (1.0 + shade_k * (shade - 1.0))
    tile_px = max(int(round(tile_m * px_per_m)), 8)
    t = np.asarray(Image.fromarray(np.clip(lum * 128, 0, 255).astype(np.uint8)).resize((tile_px, tile_px), Image.LANCZOS), float) / 128
    reps = (h // tile_px + 1, w // tile_px + 1)
    return np.tile(t, reps)[:h, :w]


def cladding_of(b):
    """建物の壁の仕上げ：b.cladding があればそれ、無ければ壁の色から（赤茶・焦げ茶の暗い壁はトタン、残りは色ごとに羽目板か吹き付け）。"""
    c = getattr(b, "cladding", None)
    if c:
        return c
    rgb = _hex(b.wall)
    if rgb.mean() < 0.55 and rgb[0] > rgb[2] + 0.08:
        return "tin"
    return "siding" if _seed(b.wall) % 3 else "plaster"


# 窓の種類：(幅 m, 下端 m, 上端 m)。下端・上端は床から
WINDOWS = {"big": (1.7, 0.85, 2.05), "door": (1.7, 0.05, 2.05), "small": (0.6, 1.4, 2.0), "pair": (0.75, 1.0, 2.0), "none": None}
BAY_PICK = ("big", "small", "pair", "big", "none", "door", "small", "big")


def facade_image(wall, windows=True, cladding="siding", seed=0):
    """壁の絵（横 FACADE_BAYS×BAY_W m、縦 STOREY_M m、上が y=0）。"""
    W_m, H_m = FACADE_BAYS * BAY_W, STOREY_M
    asset, tile, lk, sk = CLADDING[cladding]
    base = detail(asset, tile, W_m, H_m, lk, sk)[..., None] * _hex(wall)[None, None, :]
    h, w = base.shape[:2]; P = PX_PER_M
    yy = np.arange(h)[:, None] / P                           # 上からの m
    img = base * (1.0 - 0.10 * np.clip((yy - (H_m - 0.5)) / 0.5, 0, 1)[..., None])   # 階の下 0.5 m は少し汚れて暗い
    if not windows:
        return Image.fromarray(np.clip(img * 255, 0, 255).astype(np.uint8))
    rng = np.random.default_rng(seed)
    frame = np.array([0.72, 0.74, 0.76]) if rng.random() < 0.6 else np.array([0.28, 0.25, 0.22])   # 銀かブロンズのサッシ
    for i in range(FACADE_BAYS):
        kind = BAY_PICK[(seed + 3 * i + int(rng.integers(0, 3))) % len(BAY_PICK)]
        if WINDOWS[kind] is None:
            continue
        ww, z0, z1 = WINDOWS[kind]
        cx = (i + 0.5) * BAY_W + rng.uniform(-0.5, 0.5)
        for dx in ((-0.45, 0.45) if kind == "pair" else (0.0,)):
            x0, x1 = int((cx + dx - ww / 2) * P), int((cx + dx + ww / 2) * P)
            y0, y1 = int((H_m - z1) * P), int((H_m - z0) * P)
            _window(img, x0, x1, y0, y1, frame, rng, shutter=kind in ("big", "door") and rng.random() < 0.6)
    return Image.fromarray(np.clip(img * 255, 0, 255).astype(np.uint8))


def _window(img, x0, x1, y0, y1, frame, rng, shutter):
    h = img.shape[0]; f = 4
    y0 = max(y0, 6); y1 = min(y1, h - 2)
    img[y0 - f:y1 + f, x0 - f:x1 + f] = frame
    gy = np.linspace(0, 1, y1 - y0)[:, None, None]
    sky = np.array([0.55, 0.62, 0.68]); dark = np.array([0.16, 0.19, 0.22])
    glass = dark + (sky - dark) * np.clip(0.7 - gy * 0.9 + 0.1 * rng.random(), 0, 1)     # 上は空が映って明るい
    if rng.random() < 0.55:                                                            # レースのカーテン
        glass = glass * 0.5 + np.array([0.80, 0.80, 0.78]) * 0.5
    img[y0:y1, x0:x1] = glass
    mid = (x0 + x1) // 2
    img[y0:y1, mid - 2:mid + 2] = frame                                                # 引き違いの召し合わせ
    img[y1 + f:y1 + f + 3, x0 - f - 2:x1 + f + 2] = frame * 0.9                        # 水切り
    if shutter:                                                                        # 雨戸・シャッターの箱
        img[max(y0 - f - 16, 0):y0 - f, x0 - f - 2:x1 + f + 2] = frame * 0.95
    img[y1 + f + 3:min(y1 + f + 40, h), x0:x1] *= 0.95                                 # 窓の下の雨だれ


def roof_image(color, size_m=2.0, seed=0):
    """屋根の絵（size_m 四方。屋根の UV は 1＝2 m なので、2 m で UV 1 に 1 枚）。"""
    asset, tile, lk, sk = ROOFING
    d = detail(asset, tile, size_m, size_m, lk, sk)
    rng = np.random.default_rng(seed)
    img = d[..., None] * _hex(color)[None, None, :] * (0.95 + 0.1 * rng.random())
    return Image.fromarray(np.clip(img * 255, 0, 255).astype(np.uint8))
