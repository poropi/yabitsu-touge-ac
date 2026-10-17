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
    "line_orange": {"texture": "textures/line_orange.png", "shader": "ksPerPixel"},
    "concrete_circles": {"texture": "textures/concrete_circles.png", "shader": "ksPerPixel"},
    "hedge": {"texture": "textures/hedge.png", "shader": "ksPerPixel"},
    "barricade": {"texture": "textures/barricade.png", "shader": "ksPerPixel", "two_sided": True},
    "cone": {"texture": "textures/cone.png", "shader": "ksPerPixel"},
    "sign_koji": {"texture": "textures/sign_koji.png", "shader": "ksPerPixel"},
    "signal_head": {"texture": "textures/signal_head.png", "shader": "ksPerPixel"},
    "signal_ped": {"texture": "textures/signal_ped.png", "shader": "ksPerPixel"},
    "pole_grey": {"texture": "textures/pole_grey.png", "shader": "ksPerPixel"},
    "sign_bus": {"texture": "textures/sign_bus.png", "shader": "ksPerPixel"},
    "text_tomare": {"texture": "textures/text_tomare.png", "shader": "ksPerPixelAT", "alpha": True},
}

FONT_FILES = (r"C:\Windows\Fonts\BIZ-UDGothicB.ttc", r"C:\Windows\Fonts\meiryob.ttc", r"C:\Windows\Fonts\YuGothB.ttc")


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


def _font(px):
    from PIL import ImageFont
    for f in FONT_FILES:
        if Path(f).exists():
            return ImageFont.truetype(f, px)
    return ImageFont.load_default()


def _rgb(hexs):
    h = hexs.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def sign_texture(text, bg, fg, w_px, h_px, border=None):
    """看板の絵：背景色 bg に、文字 fg を枠いっぱいの大きさで中央に書く（日本語のフォント）。"""
    from PIL import ImageDraw
    im = Image.new("RGB", (w_px, h_px), _rgb(bg)); d = ImageDraw.Draw(im)
    if border:
        b = max(2, h_px // 16); d.rectangle([b // 2, b // 2, w_px - b // 2, h_px - b // 2], outline=_rgb(border), width=b)
    lines = text.split("\n")
    size = int(min(h_px * 0.8 / len(lines), w_px * 0.9 / max(len(s) for s in lines)))
    f = _font(max(8, size))
    y = (h_px - size * len(lines) * 1.05) / 2
    for s in lines:
        tw = d.textlength(s, font=f)
        d.text(((w_px - tw) / 2, y), s, font=f, fill=_rgb(fg)); y += size * 1.05
    return im


def _barricade(w=512, h=128):
    """工事用の柵の横板：黄と黒の斜めのしま（1枚＝1.8m）。"""
    yy, xx = np.mgrid[0:h, 0:w]
    on = ((xx + yy) // 48) % 2 == 0
    return _u8(np.where(on[..., None], np.array([0.98, 0.80, 0.05]), np.array([0.06, 0.06, 0.06])))


def _cone(size=256):
    """カラーコーン：赤橙に、白い帯（下から 45〜70%）。"""
    yy = np.arange(size)[:, None].repeat(size, 1) / size
    white = (yy > 0.30) & (yy < 0.55)
    return _u8(np.where(white[..., None], np.array([0.95, 0.95, 0.95]), np.array([0.92, 0.30, 0.08])))


def _lamps(w, h, n, lit, lit_rgb, vertical=False):
    from PIL import ImageDraw
    im = Image.new("RGB", (w, h), (28, 30, 32)); d = ImageDraw.Draw(im)
    for k in range(n):
        if vertical:
            cx, cy, r = w / 2, h * (k + 0.5) / n, min(w, h / n) * 0.38
        else:
            cx, cy, r = w * (k + 0.5) / n, h / 2, min(w / n, h) * 0.38
        d.ellipse([cx - r - 6, cy - r - 6, cx + r + 6, cy + r + 6], fill=(12, 12, 12))
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=lit_rgb if k == lit else (55, 58, 60))
    return im


def _text_alpha(text, w=160, h=500):
    """路面の文字（白、まわりは透明）。手前の字が下（v=0）。字は縦に細長く（道路の文字と同じ）。"""
    from PIL import ImageDraw
    n = len(text); tile = Image.new("L", (w, w), 0)
    out = Image.new("L", (w, h), 0)
    for k, ch in enumerate(text):
        tile = Image.new("L", (w, w), 0); ImageDraw.Draw(tile).text((w * 0.08, -w * 0.05), ch, font=_font(int(w * 0.85)), fill=255)
        tile = tile.resize((w, h // n))
        out.paste(tile, (0, h - (k + 1) * (h // n)))
    a = np.array(out)
    return np.dstack([np.full((h, w, 3), 235, np.uint8), a])


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
            "line_orange": _gray(256, 19, 0.80, 0.05, (1.0, 0.62, 0.16)),
            "concrete_circles": concrete_circles(),
            "hedge": _gray(512, 20, 0.42, 0.30, (0.55, 1.0, 0.45), coarse=6, fine=2),
            "barricade": _barricade(),
            "cone": _cone(),
            "sign_koji": np.array(sign_texture("工事中\n通行止め", "#F7D417", "#111111", 384, 288, border="#111111")),
            "signal_head": np.array(_lamps(384, 128, 3, 0, (40, 215, 170))),         # 青（左）が点灯
            "signal_ped": np.array(_lamps(128, 224, 2, 0, (230, 40, 40), vertical=True)),   # 赤（上）が点灯
            "pole_grey": _gray(128, 21, 0.62, 0.05),
            "sign_bus": np.array(sign_texture("バス", "#FFFFFF", "#1B4FA0", 128, 128, border="#1B4FA0")),
            "text_tomare": _text_alpha("止まれ")}
    paths = {}
    for name, a in imgs.items():
        paths[name] = out / Path(MATERIALS[name]["texture"]).name
        Image.fromarray(a).save(paths[name])
    return paths


def register_materials(m):
    for name, spec in MATERIALS.items():
        m.add_material(name, texture=spec["texture"], alpha=spec.get("alpha", False), two_sided=spec.get("two_sided", False))
