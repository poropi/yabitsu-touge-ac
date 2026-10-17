from pathlib import Path

import numpy as np
from PIL import Image

MATERIALS = {
    "asphalt": {"texture": "textures/asphalt.png", "shader": "ksPerPixel"},
    "concrete": {"texture": "textures/concrete.png", "shader": "ksPerPixel"},
    "ground": {"texture": "textures/ground.png", "shader": "ksPerPixel"},
    "line_white": {"texture": "textures/line_white.png", "shader": "ksPerPixel"},
    "guardrail": {"texture": "textures/guardrail.png", "shader": "ksPerPixel", "two_sided": True},
    "tree": {"texture": "textures/tree.png", "shader": "ksPerPixelAT", "alpha": True, "two_sided": True},
    "pavers": {"texture": "textures/pavers.png", "shader": "ksPerPixel"},
    "railing": {"texture": "textures/railing.png", "shader": "ksPerPixel"},
    "railing_bars": {"texture": "textures/railing_bars.png", "shader": "ksPerPixelAT", "alpha": True, "two_sided": True},
    "wall_white": {"texture": "textures/wall_white.png", "shader": "ksPerPixel"},
    "roof": {"texture": "textures/roof.png", "shader": "ksPerPixel", "two_sided": True},
    "stone": {"texture": "textures/stone.png", "shader": "ksPerPixel"},
    "wood": {"texture": "textures/wood.png", "shader": "ksPerPixel", "two_sided": True},
    "wood_bars": {"texture": "textures/wood_bars.png", "shader": "ksPerPixelAT", "alpha": True, "two_sided": True},
    "roof_metal": {"texture": "textures/roof_metal.png", "shader": "ksPerPixel", "two_sided": True},
    "tiles": {"texture": "textures/tiles.png", "shader": "ksPerPixel"},
    "paint_red": {"texture": "textures/paint_red.png", "shader": "ksPerPixel"},
    "concrete_circles": {"texture": "textures/concrete_circles.png", "shader": "ksPerPixel"},
}


def periodic_noise(size, feature_px, rng):
    f = np.fft.fftfreq(size)
    fx, fy = np.meshgrid(f, f)
    filt = np.exp(-(fx ** 2 + fy ** 2) * (feature_px ** 2) * 2.0)
    n = np.real(np.fft.ifft2(np.fft.fft2(rng.standard_normal((size, size))) * filt))
    return (n - n.min()) / (n.max() - n.min())


def _u8(a):
    return np.clip(np.asarray(a) * 255, 0, 255).astype(np.uint8)


def _gray(size, seed, base, amp, tint=(1.0, 1.0, 1.0), coarse=48, fine=3, specks=0.0):
    rng = np.random.default_rng(seed)
    n = 0.6 * periodic_noise(size, coarse, rng) + 0.4 * periodic_noise(size, fine, rng)
    g = base + amp * (n - 0.5)
    if specks:
        g = np.where(rng.random((size, size)) < specks, g + 0.12, g)
    return _u8(np.stack([g * tint[0], g * tint[1], g * tint[2]], -1))


def ground(size=1024, seed=4):
    rng = np.random.default_rng(seed)
    t = periodic_noise(size, 96, rng)[..., None]; d = periodic_noise(size, 4, rng)[..., None]
    green, brown = np.array([0.20, 0.28, 0.11]), np.array([0.30, 0.25, 0.16])
    return _u8((green * (1 - t) + brown * t) * (0.85 + 0.3 * d))


def tree_card(w=512, h=1024, seed=5):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    edge = periodic_noise(max(w, h), 6, rng)[:h, :w]
    crown_h = 0.85 * h
    half = np.where(yy < crown_h, (yy / crown_h) * 0.46 * w * (0.8 + 0.4 * edge), 0.0)
    trunk = (yy >= crown_h * 0.95) & (np.abs(xx - w / 2) < 0.035 * w)
    mask = (np.abs(xx - w / 2) < half) | trunk
    shade = 0.6 + 0.4 * periodic_noise(max(w, h), 10, rng)[:h, :w]
    rgb = np.where(trunk[..., None], np.array([0.25, 0.18, 0.12]), np.array([0.10, 0.22, 0.09]) * shade[..., None] * 1.6)
    return np.dstack([_u8(rgb), np.where(mask, 255, 0).astype(np.uint8)])


def pavers(size=1024, tile_m=4.0, brick=(0.25, 0.125), seed=9):
    """菜の花台の駐車場のブロック舗装。灰色に赤茶のブロックが混じる。1枚＝tile_m 四方。"""
    rng = np.random.default_rng(seed)
    px = size / tile_m; bw, bh = int(round(brick[0] * px)), int(round(brick[1] * px))
    yy, xx = np.mgrid[0:size, 0:size]
    row = yy // bh; xs = (xx + (row % 2) * (bw // 2)) % size; col = xs // bw
    ncol, nrow = size // bw + 1, size // bh + 1
    pick = rng.random((nrow, ncol)); shade = 0.9 + 0.2 * rng.random((nrow, ncol))
    grey, red = np.array([0.56, 0.55, 0.52]), np.array([0.55, 0.40, 0.33])
    base = np.where((pick[row, col] < 0.3)[..., None], red, grey) * shade[row, col][..., None]
    joint = ((yy % bh) < 2) | ((xs % bw) < 2)
    base = np.where(joint[..., None], base * 0.55, base)
    return _u8(base * (0.9 + 0.2 * periodic_noise(size, 3, rng))[..., None])


def railing_bars(size=256, bars=4, bar_px=10, rgb=(0.22, 0.52, 0.42)):
    """緑の柵の縦格子（すき間は透明）。横 1 枚に bars 本。"""
    xx = np.arange(size)[None, :].repeat(size, 0)
    on = (xx % (size // bars)) < bar_px
    return np.dstack([_u8(np.broadcast_to(np.array(rgb), (size, size, 3))), np.where(on, 255, 0).astype(np.uint8)])


def tiles(size=512, tile_px=64, seed=14):
    """展望台への歩道の四角いタイル。1枚＝2m 四方、タイルは 0.25m 角。"""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    n = size // tile_px; shade = 0.92 + 0.12 * rng.random((n, n))
    g = 0.62 * shade[yy // tile_px, xx // tile_px] * (0.93 + 0.14 * periodic_noise(size, 3, rng))
    g = np.where(((yy % tile_px) < 2) | ((xx % tile_px) < 2), g * 0.6, g)
    return _u8(np.stack([g, g * 0.98, g * 0.94], -1))


def concrete_circles(size=512, ring_px=32, seed=17):
    """急坂の滑り止めコンクリート舗装：明るい灰色に、丸い溝（輪）が規則正しく並ぶ。1枚＝2m 四方。"""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    cy = (yy % ring_px) - ring_px / 2; cx = (xx % ring_px) - ring_px / 2
    r = np.hypot(cx, cy)
    g = 0.66 * (0.93 + 0.14 * periodic_noise(size, 3, rng))
    g = np.where((r > ring_px * 0.28) & (r < ring_px * 0.36), g * 0.62, g)
    return _u8(np.stack([g, g * 0.99, g * 0.96], -1))


def wood(size=256, seed=15):
    rng = np.random.default_rng(seed)
    grain = 0.8 + 0.2 * periodic_noise(size, 2, rng)[:, :1].repeat(size, 1) * periodic_noise(size, 20, rng)
    return _u8(np.array([0.30, 0.19, 0.12]) * grain[..., None] * 1.1)


def roof(size=512, seed=10):
    rng = np.random.default_rng(seed)
    yy = np.arange(size)[:, None].repeat(size, 1)
    plank = 0.85 + 0.15 * ((yy // 32) % 2) - 0.25 * ((yy % 32) < 2)
    g = (0.75 + 0.25 * periodic_noise(size, 8, rng)) * plank
    return _u8(np.array([0.42, 0.27, 0.16]) * g[..., None])


def write_textures(out_dir):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    imgs = {"asphalt": _gray(1024, 1, 0.26, 0.10, (1.0, 1.0, 1.02), specks=0.005),
            "concrete": _gray(1024, 2, 0.58, 0.12, (1.02, 1.0, 0.96)),
            "ground": ground(),
            "line_white": _gray(256, 3, 0.86, 0.06),
            "guardrail": _gray(256, 6, 0.74, 0.08, coarse=24),
            "tree": tree_card(),
            "pavers": pavers(),
            "railing": _gray(256, 11, 0.50, 0.06, (0.44, 1.04, 0.84), coarse=24),
            "railing_bars": railing_bars(),
            "wall_white": _gray(512, 12, 0.86, 0.06, (1.0, 1.0, 0.98)),
            "roof": roof(),
            "stone": _gray(512, 13, 0.50, 0.22, (1.05, 0.98, 0.88), coarse=16, fine=2),
            "wood": wood(),
            "wood_bars": railing_bars(rgb=(0.30, 0.19, 0.12), bars=4, bar_px=14),
            "roof_metal": _gray(256, 16, 0.50, 0.06, (0.84, 1.0, 0.94), coarse=24),
            "tiles": tiles(),
            "paint_red": _gray(256, 18, 0.62, 0.08, (1.0, 0.42, 0.36)),
            "concrete_circles": concrete_circles()}
    paths = {}
    for name, a in imgs.items():
        paths[name] = out / Path(MATERIALS[name]["texture"]).name
        Image.fromarray(a).save(paths[name])
    return paths


def register_materials(m):
    for name, spec in MATERIALS.items():
        m.add_material(name, texture=spec["texture"], alpha=spec.get("alpha", False), two_sided=spec.get("two_sided", False))
