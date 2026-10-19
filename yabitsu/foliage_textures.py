"""木の葉・枝・樹皮のテクスチャを、Poly Haven（CC0）の写真素材から合成する。

素材（data/raw/polyhaven/、出典は assets/forest/README.md）：
  fir_tree_01 の針葉の房（twig）と樹皮、tree_small_02 の葉の房、fern_02 のシダ。
針葉や葉の房を切り出し、2D で並べて「枝1本分」「葉の茂み1つ分」の画像を作る（ゲームの木の一般的な作り方）。
"""
import io
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from PIL import Image
from scipy import ndimage

from .config import DATA_RAW

SRC = DATA_RAW / "polyhaven"
ATLASES = {
    "twig": ("fir_tree_01", "fir_tree_01_twig_diff_2k.jpg", "fir_tree_01_twig_alpha_2k.png", "fir_tree_01_twig_nor_gl_2k.jpg"),
    "leaf": ("tree_small_02", "tree_small_02_leaves_diff_2k.jpg", "tree_small_02_leaves_alpha_2k.png", "tree_small_02_leaves_nor_gl_2k.jpg"),
    "fern": ("fern_02", "fern_02_diff_2k.jpg", "fern_02_alpha_2k.png", "fern_02_nor_gl_2k.jpg"),
}
MAX_ASPECT = {"twig": 2.2, "leaf": 2.2, "fern": 8.0}   # 細長い棒（茎だけの部分）を捨てる。シダの葉は細長いので別
BARK = ("fir_tree_01", "fir_tree_01_bark_diff_2k.jpg", "fir_tree_01_bark_nor_gl_2k.jpg")


def sources_available() -> bool:
    return all((SRC / a / "textures" / f).exists() for a, *fs in ATLASES.values() for f in fs) and \
        (SRC / BARK[0] / "textures" / BARK[1]).exists()


@dataclass
class Sprite:
    rgba: np.ndarray    # (h, w, 4) float 0〜1。房の付け根が下端の中央に来る向き
    nrm: np.ndarray     # (h, w, 3) float −1〜1（接空間）


def _load(asset, name, mode):
    im = Image.open(SRC / asset / "textures" / name)
    if im.mode.startswith("I;16") or im.mode == "I":
        a = np.asarray(im, np.float32)
        return a / (65535.0 if a.max() > 255 else 255.0)
    return np.asarray(im.convert(mode), np.float32) / 255.0


@lru_cache(maxsize=None)
def sprites(kind: str, min_frac=0.004) -> tuple:
    """atlas の房を1つずつ切り出す。小さすぎるもの・細長い線（茎だけの部分）は捨てる。"""
    asset, diff, alpha, nor = ATLASES[kind]
    rgb, a, n = _load(asset, diff, "RGB"), _load(asset, alpha, "L"), _load(asset, nor, "RGB")
    if a.ndim == 3:
        a = a[..., 0]
    mask = a > 0.5
    lab, k = ndimage.label(ndimage.binary_dilation(mask, iterations=6))
    out = []
    for i, sl in enumerate(ndimage.find_objects(lab), 1):
        h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        own = (lab[sl] == i) & mask[sl]
        fill = own.mean() if h * w else 0
        green = (rgb[sl][..., 1] - rgb[sl][..., 0])[own].mean() if own.any() else 0
        # 房らしいもの：小さすぎず、atlas 全体を覆うほど大きくなく、透け（隙間）があり、緑が勝つ
        if h * w < min_frac * mask.size or min(h, w) < 40 or max(h, w) > 0.9 * max(mask.shape)                 or not 0.12 < fill < 0.55 or green < 0.02 or max(h, w) > MAX_ASPECT[kind] * min(h, w):
            continue
        rgba = np.dstack([rgb[sl], np.where(lab[sl] == i, a[sl], 0)])
        if w > h:     # 横長の房は、付け根が下に来るよう立てる（茎の向きは付け根側が濃いほう）
            rgba = np.rot90(rgba, 1); nn = np.rot90(n[sl], 1)
        else:
            nn = n[sl]
        out.append(Sprite(rgba.copy(), nn * 2.0 - 1.0))
    return tuple(out)


def _rotate(img, angle):
    """PIL の回転を float 配列に（チャンネルごと）。"""
    chans = [np.asarray(Image.fromarray(img[..., c].astype(np.float32), "F").rotate(angle, Image.BILINEAR, expand=True))
             for c in range(img.shape[2])]
    return np.dstack(chans)


def _place(canvas, nrm, sp: Sprite, cx, cy, angle, scale, tint):
    """付け根 (cx, cy) から angle 度（0＝上、反時計回り）の向きへ房を置く。"""
    h, w = sp.rgba.shape[:2]
    s = scale * canvas.shape[0] / h
    nh, nw = max(4, int(h * s)), max(4, int(w * s))
    rgba = np.dstack([np.asarray(Image.fromarray(sp.rgba[..., c]).resize((nw, nh), Image.BILINEAR)) for c in range(4)])
    nn = np.dstack([np.asarray(Image.fromarray(sp.nrm[..., c].astype(np.float32)).resize((nw, nh), Image.BILINEAR)) for c in range(3)])
    # 付け根（下端中央）を回転の中心にするため、上下に余白を足して中心を付け根に合わせる
    pad = np.zeros((nh, nw, 4), np.float32); padn = np.zeros((nh, nw, 3), np.float32)
    rgba = np.concatenate([rgba, pad]); nn = np.concatenate([nn, padn])
    rgba = _rotate(rgba, angle); nn = _rotate(nn, angle)
    t = np.radians(angle); c, s_ = np.cos(t), np.sin(t)
    nx, ny = nn[..., 0] * c - nn[..., 1] * s_, nn[..., 0] * s_ + nn[..., 1] * c
    nn = np.dstack([nx, ny, nn[..., 2]])
    H, W = rgba.shape[:2]
    x0, y0 = int(cx - W / 2), int(cy - H / 2)
    xa, ya, xb, yb = max(0, x0), max(0, y0), min(canvas.shape[1], x0 + W), min(canvas.shape[0], y0 + H)
    if xa >= xb or ya >= yb:
        return
    src = rgba[ya - y0:yb - y0, xa - x0:xb - x0]; srcn = nn[ya - y0:yb - y0, xa - x0:xb - x0]
    al = (src[..., 3:4] > 0.5).astype(np.float32)
    dst = canvas[ya:yb, xa:xb]
    dst[..., :3] = dst[..., :3] * (1 - al) + src[..., :3] * np.asarray(tint, np.float32) * al
    dst[..., 3:4] = np.maximum(dst[..., 3:4], al)
    nrm[ya:yb, xa:xb] = nrm[ya:yb, xa:xb] * (1 - al) + srcn * al


def _canvas(w, h):
    nrm = np.zeros((h, w, 3), np.float32); nrm[..., 2] = 1.0
    return np.zeros((h, w, 4), np.float32), nrm


def _stem(canvas, x0, y0, x1, y1, width0, width1, color):
    n = int(np.hypot(x1 - x0, y1 - y0)) + 1
    for i in range(n):
        t = i / max(1, n - 1)
        x, y, r = x0 + (x1 - x0) * t, y0 + (y1 - y0) * t, (width0 + (width1 - width0) * t) / 2
        ya, yb, xa, xb = int(y - r), int(y + r) + 1, int(x - r), int(x + r) + 1
        canvas[max(0, ya):yb, max(0, xa):xb, :3] = color
        canvas[max(0, ya):yb, max(0, xa):xb, 3] = 1.0


def conifer_branch(seed, tint, density=1.0, w=1024, h=512):
    """左端が付け根、右端が枝先の枝1本分（針葉の房を左右に交互に並べる）。"""
    rng = np.random.default_rng(seed)
    sp = sprites("twig")
    canvas, nrm = _canvas(w, h)
    cy = h * 0.5
    _stem(canvas, 0, cy, w * 0.97, cy, h * 0.035, h * 0.008, (0.28, 0.20, 0.14))
    n = int(26 * density)
    for i in range(n):
        t = 0.04 + 0.92 * i / n
        side = 1 if i % 2 else -1
        ang = -90 + side * rng.uniform(30, 62)          # 枝先（右）へ向けて左右に開く
        scale = (0.62 - 0.35 * t) * rng.uniform(0.85, 1.15)
        tt = np.r_[tint] * rng.uniform(0.85, 1.12)
        _place(canvas, nrm, sp[rng.integers(len(sp))], t * w, cy + rng.normal(0, h * 0.02), ang, scale, tt)
    _place(canvas, nrm, sp[rng.integers(len(sp))], w * 0.93, cy, -90, 0.35, tint)
    return canvas, nrm


def leaf_cluster(seed, tint, n=34, w=1024, h=1024):
    """下端中央を付け根とする葉の茂み1つ分（葉の房を扇状に重ねる）。"""
    rng = np.random.default_rng(seed)
    sp = sprites("leaf")
    canvas, nrm = _canvas(w, h)
    for _ in range(n):
        ang = rng.uniform(-75, 75)
        r = rng.uniform(0.0, 0.35) * h
        t = np.radians(ang)
        cx, cy = w / 2 - np.sin(t) * r, h * 0.98 - np.cos(t) * r
        tt = np.r_[tint] * rng.uniform(0.8, 1.15)
        _place(canvas, nrm, sp[rng.integers(len(sp))], cx, cy, ang + rng.normal(0, 12), rng.uniform(0.35, 0.6), tt)
    return canvas, nrm


def fern_fronds():
    """シダの葉を縦に並べたもの（1枚ずつの UV は fern_uvs で返す）。"""
    sp = sprites("fern")[:4]
    w, h = 1024, 1024
    canvas, nrm = _canvas(w, h)
    for i, s in enumerate(sp):
        _place(canvas, nrm, s, w * (i + 0.5) / len(sp), h * 0.99, 0, 0.97, (0.95, 1.0, 0.9))
    return canvas, nrm, len(sp)


def bark(kind, seed=3, w=512, h=1024):
    """幹の樹皮（縦に繰り返す）。杉・檜は赤茶の縦すじ、広葉樹は灰褐色。"""
    base = _load(BARK[0], BARK[1], "RGB")
    nb = _load(BARK[0], BARK[2], "RGB") * 2 - 1
    img = np.asarray(Image.fromarray((base * 255).astype(np.uint8)).resize((w, h)), np.float32) / 255
    nrm = np.dstack([np.asarray(Image.fromarray(nb[..., c]).resize((w, h))) for c in range(3)])
    if kind in ("sugi", "hinoki"):
        rng = np.random.default_rng(seed)
        strips = ndimage.gaussian_filter(rng.normal(0, 1, (h // 16, w)), (1, 2.5))
        strips = np.repeat(strips, 16, 0)[:h]
        strips = ndimage.gaussian_filter(strips, (24, 0.8))
        strips = strips / (np.abs(strips).max() + 1e-6)
        fine = ndimage.gaussian_filter(rng.normal(0, 1, (h, w)), (6, 0.6))
        fine = fine / (np.abs(fine).max() + 1e-6)
        col = np.array((0.52, 0.38, 0.30) if kind == "sugi" else (0.50, 0.40, 0.33), np.float32)
        lum = img.mean(-1, keepdims=True); lum = lum / lum.mean()
        img = col * (0.55 + 0.45 * lum) * (1 + 0.35 * strips[..., None] + 0.2 * fine[..., None])
        nrm[..., 0] += 6.0 * np.gradient(strips + 0.5 * fine, axis=1)
    else:
        img = img * np.array((1.05, 1.0, 0.95), np.float32)
    return np.clip(img, 0, 1), nrm


def png_rgba(canvas) -> bytes:
    """アルファの縁の色を周りの葉の色でにじませてから PNG にする（縁が黒く光るのを防ぐ）。"""
    rgb, a = canvas[..., :3].copy(), canvas[..., 3]
    if (a > 0.5).any():
        idx = ndimage.distance_transform_edt(a <= 0.5, return_distances=False, return_indices=True)
        rgb = rgb[idx[0], idx[1]]
    out = np.dstack([rgb, a]) if canvas.shape[2] == 4 else rgb
    b = io.BytesIO()
    Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8)).save(b, "PNG", optimize=True)
    return b.getvalue()


def png_rgb(img) -> bytes:
    b = io.BytesIO()
    Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(b, "PNG", optimize=True)
    return b.getvalue()


def png_normal(nrm) -> bytes:
    n = nrm / np.maximum(np.linalg.norm(nrm, axis=-1, keepdims=True), 1e-6)
    b = io.BytesIO()
    Image.fromarray(((n * 0.5 + 0.5) * 255).astype(np.uint8)).save(b, "PNG", optimize=True)
    return b.getvalue()
