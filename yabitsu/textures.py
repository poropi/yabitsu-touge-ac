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
    "wall_cream": {"texture": "textures/wall_cream.png", "shader": "ksPerPixel"},
    "wall_beige": {"texture": "textures/wall_beige.png", "shader": "ksPerPixel"},
    "wall_grey": {"texture": "textures/wall_grey.png", "shader": "ksPerPixel"},
    "wall_brown": {"texture": "textures/wall_brown.png", "shader": "ksPerPixel"},
    "roof": {"texture": "textures/roof.png", "shader": "ksPerPixel", "two_sided": True},
    "stone": {"texture": "textures/stone.png", "shader": "ksPerPixel"},
    "wood": {"texture": "textures/wood.png", "shader": "ksPerPixel", "two_sided": True},
    "wood_bars": {"texture": "textures/wood_bars.png", "shader": "ksPerPixelAT", "alpha": True, "two_sided": True},
    "roof_metal": {"texture": "textures/roof_metal.png", "shader": "ksPerPixel", "two_sided": True},
    "roof_blue": {"texture": "textures/roof_blue.png", "shader": "ksPerPixel", "two_sided": True},
    "roof_red": {"texture": "textures/roof_red.png", "shader": "ksPerPixel", "two_sided": True},
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
    "hoarding": {"texture": "textures/hoarding.png", "shader": "ksPerPixel", "two_sided": True},
    "tunnel_dark": {"texture": "textures/tunnel_dark.png", "shader": "ksPerPixel"},
    "gravel": {"texture": "textures/gravel.png", "shader": "ksPerPixel"},
    "vending": {"texture": "textures/vending.png", "shader": "ksPerPixel"},
    "vending_front": {"texture": "textures/vending_front.png", "shader": "ksPerPixel"},
    "post_red": {"texture": "textures/post_red.png", "shader": "ksPerPixel"},
    "glass_grey": {"texture": "textures/glass_grey.png", "shader": "ksPerPixel", "two_sided": True},
    "info_board": {"texture": "textures/info_board.png", "shader": "ksPerPixel"},
    "water": {"texture": "textures/water.png", "shader": "ksPerPixel"},
    "riverbed": {"texture": "textures/riverbed.png", "shader": "ksPerPixel"},
    "water_fall": {"texture": "textures/water_fall.png", "shader": "ksPerPixel", "two_sided": True},
    "roof_dark": {"texture": "textures/roof_dark.png", "shader": "ksPerPixel", "two_sided": True},
    "pole_orange": {"texture": "textures/pole_orange.png", "shader": "ksPerPixel"},
    "mirror": {"texture": "textures/mirror.png", "shader": "ksPerPixel", "two_sided": True},
    "sign_speed30": {"texture": "textures/sign_speed30.png", "shader": "ksPerPixelAT", "alpha": True, "two_sided": True},
    "sign_warning": {"texture": "textures/sign_warning.png", "shader": "ksPerPixelAT", "alpha": True, "two_sided": True},
    "sign_speed40": {"texture": "textures/sign_speed40.png", "shader": "ksPerPixelAT", "alpha": True, "two_sided": True},
    "tunnel_wall": {"texture": "textures/tunnel_wall.png", "shader": "ksPerPixel"},
    "tunnel_lamp": {"texture": "textures/tunnel_lamp.png", "shader": "ksPerPixel", "two_sided": True},
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
    """菜の花台の駐車場のブロック舗装。灰色に、ときどき赤茶のブロックが混じる（ストリートビュー 2025）。1枚＝tile_m 四方。"""
    rng = np.random.default_rng(seed)
    px = size / tile_m; bw, bh = int(round(brick[0] * px)), int(round(brick[1] * px))
    yy, xx = np.mgrid[0:size, 0:size]
    row = yy // bh; xs = (xx + (row % 2) * (bw // 2)) % size; col = xs // bw
    ncol, nrow = size // bw + 1, size // bh + 1
    pick = rng.random((nrow, ncol)); shade = 0.9 + 0.2 * rng.random((nrow, ncol))
    grey, red = np.array([0.56, 0.55, 0.52]), np.array([0.55, 0.40, 0.33])
    base = np.where((pick[row, col] < 0.08)[..., None], red, grey) * shade[row, col][..., None]   # 写真は灰色が主（2026-10-20）
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
    from . import config
    return ImageFont.truetype(config.font_file("gothic_bold"), px)


def _rgb(hexs):
    h = hexs.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def _mirror(px=128):
    """カーブミラー：オレンジの縁の、空色の丸い鏡。"""
    from PIL import ImageDraw
    im = Image.new("RGB", (px, px), (230, 110, 40)); d = ImageDraw.Draw(im)
    d.ellipse([10, 10, px - 10, px - 10], fill=(150, 175, 190))
    return np.array(im)


def _round_sign(text, px=256):
    """速度制限の標識：白地・赤い輪・青い数字（まわりは透明）。"""
    from PIL import ImageDraw
    im = Image.new("RGBA", (px, px), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    d.ellipse([4, 4, px - 4, px - 4], fill=(205, 30, 35, 255)); d.ellipse([32, 32, px - 32, px - 32], fill=(250, 250, 250, 255))
    font = _font(int(px * 0.42)); w = d.textlength(text, font=font)
    d.text(((px - w) / 2, px * 0.27), text, fill=(25, 60, 150, 255), font=font)
    return np.array(im)


def _diamond_sign(px=256):
    """警戒標識：黄色のひし形に黒い縁（まわりは透明）。"""
    from PIL import ImageDraw
    im = Image.new("RGBA", (px, px), (0, 0, 0, 0)); d = ImageDraw.Draw(im); h = px / 2
    d.polygon([(h, 2), (px - 2, h), (h, px - 2), (2, h)], fill=(20, 20, 20, 255))
    d.polygon([(h, 18), (px - 18, h), (h, px - 18), (18, h)], fill=(245, 200, 20, 255))
    d.polygon([(h - 30, h + 40), (h, h - 45), (h + 30, h + 40)], fill=(20, 20, 20, 255))
    return np.array(im)


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


def _hoarding(w=256, h=256):
    """工事の仮囲い：白い波板（縦の溝）、1枚＝横 2m・縦いっぱい。"""
    xx = np.arange(w)[None, :].repeat(h, 0)
    g = 0.90 - 0.08 * (np.sin(xx / w * 2 * np.pi * 12) > 0.6)
    return _u8(np.stack([g, g, g * 0.99], -1))


def _vending_front(w=256, h=256):
    """自動販売機の見本の窓：缶の並び（3段）と、下の取り出し口。"""
    rng = np.random.default_rng(33)
    img = np.full((h, w, 3), 0.92)
    for r in range(3):
        y0 = int(h * (0.06 + r * 0.22))
        img[y0:y0 + int(h * 0.18), int(w * 0.05):int(w * 0.95)] = [0.85, 0.88, 0.9]
        for c in range(8):
            x0 = int(w * (0.07 + c * 0.11))
            img[y0 + 6:y0 + int(h * 0.17), x0:x0 + int(w * 0.07)] = rng.random(3) * 0.7 + 0.2
    img[int(h * 0.8):int(h * 0.92), int(w * 0.2):int(w * 0.8)] = [0.1, 0.1, 0.1]
    return _u8(img)


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
            "wall_cream": _gray(512, 24, 0.86, 0.06, (1.02, 0.98, 0.86)),
            "wall_beige": _gray(512, 25, 0.76, 0.06, (1.04, 0.96, 0.84)),
            "wall_grey": _gray(512, 26, 0.66, 0.05, (0.98, 1.0, 1.02)),
            "wall_brown": _gray(512, 27, 0.48, 0.06, (1.10, 0.90, 0.74)),
            "roof": roof(),
            "stone": _gray(512, 13, 0.50, 0.22, (1.05, 0.98, 0.88), coarse=16, fine=2),
            "wood": wood(),
            "wood_bars": railing_bars(rgb=(0.30, 0.19, 0.12), bars=4, bar_px=14),
            "roof_metal": _gray(256, 16, 0.50, 0.06, (0.84, 1.0, 0.94), coarse=24),
            "roof_blue": _gray(256, 22, 0.42, 0.05, (0.55, 0.80, 1.25), coarse=24),
            "roof_red": _gray(256, 23, 0.45, 0.05, (1.35, 0.62, 0.50), coarse=24),
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
            "text_tomare": _text_alpha("止まれ"),
            "hoarding": _hoarding(),
            "tunnel_dark": _gray(64, 28, 0.04, 0.02),
            "gravel": _gray(512, 29, 0.62, 0.30, (1.02, 1.0, 0.95), coarse=3, fine=1, specks=0.05),
            "vending": _gray(128, 30, 0.90, 0.03),
            "vending_front": _vending_front(),
            "post_red": _gray(128, 31, 0.62, 0.04, (1.45, 0.25, 0.22)),
            "glass_grey": _gray(128, 32, 0.45, 0.05, (0.9, 0.98, 1.05)),
            "info_board": np.array(sign_texture("関東ふれあいのみち\nご案内", "#F4F1E6", "#2F4A2A", 384, 256, border="#5A3E22")),
            "water": _gray(512, 33, 0.36, 0.10, (0.72, 0.98, 1.10), coarse=20, fine=2),           # 山の沢の、緑がかった水
            "riverbed": _gray(512, 34, 0.60, 0.34, (1.03, 1.0, 0.94), coarse=5, fine=1, specks=0.08),   # 白っぽい川原の石
            "water_fall": _gray(256, 35, 0.90, 0.10, (0.96, 1.0, 1.04), coarse=4, fine=1),
            "roof_dark": _gray(256, 36, 0.32, 0.05, (0.95, 1.0, 1.05), coarse=24),
            "pole_orange": _gray(64, 37, 0.70, 0.04, (1.30, 0.62, 0.22)),
            "mirror": _mirror(),
            "sign_speed30": _round_sign("30"),
            "sign_speed40": _round_sign("40"),
            "sign_warning": _diamond_sign(),
            "tunnel_wall": _gray(512, 38, 0.46, 0.10, (1.0, 0.99, 0.96), coarse=10, fine=2),
            "tunnel_lamp": _gray(64, 39, 0.97, 0.02, (1.04, 0.88, 0.55))}
    paths = {}
    for name, a in imgs.items():
        paths[name] = out / Path(MATERIALS[name]["texture"]).name
        Image.fromarray(a).save(paths[name])
    return paths


def register_materials(m):
    for name, spec in MATERIALS.items():
        m.add_material(name, texture=spec["texture"], alpha=spec.get("alpha", False), two_sided=spec.get("two_sided", False))
