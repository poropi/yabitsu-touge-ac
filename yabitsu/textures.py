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
    "wood_cedar": {"texture": "textures/wood_cedar.png", "shader": "ksPerPixel"},   # 杉板の写真（assets/free、CC0）
    "wood_bars": {"texture": "textures/wood_bars.png", "shader": "ksPerPixelAT", "alpha": True, "two_sided": True},
    "roof_metal": {"texture": "textures/roof_metal.png", "shader": "ksPerPixel", "two_sided": True},
    "roof_blue": {"texture": "textures/roof_blue.png", "shader": "ksPerPixel", "two_sided": True},
    "roof_red": {"texture": "textures/roof_red.png", "shader": "ksPerPixel", "two_sided": True},
    "tiles": {"texture": "textures/tiles.png", "shader": "ksPerPixel"},
    "paint_red": {"texture": "textures/paint_red.png", "shader": "ksPerPixel"},
    "paint_green": {"texture": "textures/paint_green.png", "shader": "ksPerPixel"},
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
    "sign_stop": {"texture": "textures/sign_stop.png", "shader": "ksPerPixelAT", "alpha": True, "two_sided": True},
    "sign_fire_cistern": {"texture": "textures/sign_fire_cistern.png", "shader": "ksPerPixelAT", "alpha": True, "two_sided": True},
    "pole_white": {"texture": "textures/pole_white.png", "shader": "ksPerPixel"},
    "bamboo": {"texture": "textures/bamboo.png", "shader": "ksPerPixel"},
    "pole_redwhite": {"texture": "textures/pole_redwhite.png", "shader": "ksPerPixel"},
    "vending_red": {"texture": "textures/vending_red.png", "shader": "ksPerPixel"},
    "stone_nozura": {"texture": "textures/stone_nozura.png", "shader": "ksPerPixel"},
    "sign_speed40": {"texture": "textures/sign_speed40.png", "shader": "ksPerPixelAT", "alpha": True, "two_sided": True},
    "tunnel_wall": {"texture": "textures/tunnel_wall.png", "shader": "ksPerPixel"},
    "tunnel_lamp": {"texture": "textures/tunnel_lamp.png", "shader": "ksPerPixel", "two_sided": True},
    "pavers_brick": {"texture": "textures/pavers_brick.png", "shader": "ksPerPixel"},   # 峠のトイレ前の角地の灰茶のれんが（杉綾）
    "delineator": {"texture": "textures/delineator.png", "shader": "ksPerPixel"},       # オレンジの車線分離標（白い反射帯2本）
    "drum_checker": {"texture": "textures/drum_checker.png", "shader": "ksPerPixel"},   # 紅白の市松の緩衝ドラム
    "siding_white": {"texture": "textures/siding_white.png", "shader": "ksPerPixel"},   # 白い縦の羽目板（峠のトイレの棟）
    "rubber_black": {"texture": "textures/rubber_black.png", "shader": "ksPerPixel"},   # 車線分離標の黒い台座
    "concrete_board": {"texture": "textures/concrete_board.png", "shader": "ksPerPixel"},   # 板の型枠の跡がある打ち放しのコンクリート（寺山のトイレ）
    "roof_shingle": {"texture": "textures/roof_shingle.png", "shader": "ksPerPixel", "two_sided": True},   # 苔の付いた茶色い屋根板（寺山のトイレ）
    "pole_green": {"texture": "textures/pole_green.png", "shader": "ksPerPixel"},   # 緑に塗った柱（山のトイレチップ塔）
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


def cedar_planks():
    """杉板の写真（Poly Haven「Japanese Cedar Planks」CC0、実寸 約 1.13 m 四方）。少し暗くして焦げ茶の板張りに寄せる。"""
    from . import config
    a = np.asarray(Image.open(config.PROJECT_ROOT / "assets" / "free" / "japanese_cedar_planks_1k.jpg").convert("RGB"), float)
    return np.clip(a * [0.55, 0.45, 0.40], 0, 255).astype(np.uint8)


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


def _stop_sign(px=256):
    """一時停止：赤い逆三角に白い縁、白い「止まれ」（まわりは透明）。"""
    from PIL import ImageDraw
    im = Image.new("RGBA", (px, px), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    d.polygon([(4, 20), (px - 4, 20), (px / 2, px - 10)], fill=(245, 245, 245, 255))
    d.polygon([(22, 30), (px - 22, 30), (px / 2, px - 30)], fill=(200, 30, 36, 255))
    d.text((px / 2, 80), "止まれ", fill=(250, 250, 250, 255), font=_font(42), anchor="mm")
    return np.array(im)


def _bands(colors, n, w=32, h=512):
    """縦に n 段の色のしま（赤白の柱など。上から colors を交互に）。"""
    y = np.arange(h)[:, None].repeat(w, 1)
    c = np.array(colors, np.uint8)
    return c[(y * n // h) % len(colors)]


def _delineator(w=64, h=512):
    """オレンジの車線分離標（高さ 0.8 m を縦 1 枚）：上から 8〜20 %・28〜40 % に白い反射帯（ストリートビューの峠）。"""
    y = np.arange(h)[:, None].repeat(w, 1) / h
    white = ((y > 0.08) & (y < 0.20)) | ((y > 0.28) & (y < 0.40))
    return np.where(white[..., None], np.array([236, 236, 228], np.uint8), np.array([238, 92, 30], np.uint8))


def _checker(colors, nu, nv, w=256, h=256):
    """市松模様（円柱の周りに nu、縦に nv）。"""
    yy, xx = np.mgrid[0:h, 0:w]
    c = np.array(colors, np.uint8)
    return c[((xx * nu // w) + (yy * nv // h)) % 2]


def _siding(size=512, seed=43):
    """白い縦の羽目板（2 m 四方を 1 枚。15 cm ごとに細い溝、少し汚れ）。"""
    rng = np.random.default_rng(seed)
    xx = np.arange(size)[None, :].repeat(size, 0)
    groove = (xx % (size * 0.15 / 2.0)) < 2.5
    base = np.array([0.88, 0.87, 0.84]) * (0.94 + 0.06 * periodic_noise(size, 40, rng))[..., None]
    return _u8(np.where(groove[..., None], base * 0.72, base))


def herringbone_bricks(size=1024, tile_m=2.0, brick_w=0.1, seed=44):
    """灰茶のれんがの杉綾（ヘリンボーン）敷き（2:1 のれんが、45° 回し）。1 枚＝tile_m 四方で継ぎ目なくくり返す。"""
    rng = np.random.default_rng(seed)
    n = int(round(tile_m / brick_w / np.sqrt(2)))      # 45° 回した格子の 1 枚あたりのマス数（4 の倍数にそろえる）
    n -= n % 4
    yy, xx = np.mgrid[0:size, 0:size].astype(float) * (n / size)
    gx = np.floor(xx + yy).astype(int) % (2 * n); gy = np.floor(yy - xx).astype(int) % (2 * n)
    d = (gx - gy) % 4
    horiz = d < 2
    bx = np.where(horiz, gx - d, gx); by = np.where(horiz, gy, gy - (3 - d))
    bid = (bx % (2 * n)) * 4096 + (by % (2 * n)) * 2 + horiz
    h = (np.sin(bid * 12.9898) * 43758.5453) % 1.0
    shade = 0.80 + 0.32 * h
    # 灰茶のれんがの混ぜ敷き（ストリートビュー 012° の中央値 RGB (65, 64, 56)、R/G 1.02。赤茶一色ではない）
    tones = np.array([[0.46, 0.44, 0.38], [0.50, 0.45, 0.38], [0.43, 0.42, 0.37], [0.52, 0.44, 0.37]])
    base = tones[(np.floor(((np.sin(bid * 78.233) * 12345.678) % 1.0) * 4)).astype(int)] * shade[..., None]
    joint = (bid != np.roll(bid, 1, 0)) | (bid != np.roll(bid, 1, 1)) | (bid != np.roll(bid, -1, 1))
    base = np.where(joint[..., None], np.array([0.40, 0.38, 0.35]), base)
    return _u8(base * (0.9 + 0.2 * periodic_noise(size, 3, rng))[..., None])


def trail_map_texture(w_px, h_px):
    """峠の横長の案内地図（左右に写真の枠、真ん中に黄緑の地図・山・道。ストリートビューの峠のトイレ前）。"""
    from PIL import ImageDraw
    rng = np.random.default_rng(7)
    im = Image.new("RGB", (w_px, h_px), (238, 234, 220)); d = ImageDraw.Draw(im)
    W, H = w_px, h_px
    d.rectangle([0, 0, W - 1, H - 1], outline=(96, 70, 44), width=max(3, H // 40))
    d.rectangle([int(W * 0.03), int(H * 0.04), int(W * 0.97), int(H * 0.12)], fill=(60, 96, 60))
    d.text((W / 2, H * 0.08), "丹沢大山国定公園 ヤビツ峠 周辺案内図", fill=(245, 245, 235), font=_font(int(H * 0.06)), anchor="mm")
    mx0, mx1, my0, my1 = int(W * 0.22), int(W * 0.78), int(H * 0.16), int(H * 0.94)
    d.rectangle([mx0, my0, mx1, my1], fill=(214, 226, 150))
    for _ in range(9):   # 山
        cx, cy, r = rng.uniform(mx0 + 40, mx1 - 40), rng.uniform(my0 + 40, my1 - 30), rng.uniform(H * 0.05, H * 0.10)
        d.polygon([(cx - r, cy + r * 0.6), (cx, cy - r * 0.7), (cx + r, cy + r * 0.6)], fill=(110, 160, 90))
    for _ in range(3):   # 登山道（赤い線）：峠から山へ曲がりながら延びる
        x, y = (mx0 + mx1) / 2, (my0 + my1) / 2; a = rng.uniform(0, 2 * np.pi); pts = [(x, y)]
        for _ in range(40):
            a += rng.normal(0, 0.25); x = float(np.clip(x + 9 * np.cos(a), mx0 + 5, mx1 - 5)); y = float(np.clip(y + 9 * np.sin(a), my0 + 5, my1 - 5)); pts.append((x, y))
        d.line(pts, fill=(200, 60, 50), width=max(2, H // 110), joint="curve")
    d.line([(mx0, my1 - 30), (W * 0.45, H * 0.6), (mx1, my0 + 40)], fill=(150, 150, 150), width=max(4, H // 45))
    for side in (0, 1):  # 左右の写真と説明の枠
        x0 = int(W * (0.03 if side == 0 else 0.80))
        for j in range(5):
            y0 = int(H * (0.16 + j * 0.16))
            d.rectangle([x0, y0, x0 + int(W * 0.07), y0 + int(H * 0.13)], fill=tuple(int(c) for c in rng.integers(70, 190, 3)))
            for q in range(4):
                d.line([(x0 + int(W * 0.08), y0 + 8 + q * int(H * 0.03)), (x0 + int(W * 0.17), y0 + 8 + q * int(H * 0.03))],
                       fill=(120, 110, 100), width=2)
    return im


def _board_concrete(w=512, h=512, seed=47):
    """板の型枠の跡がある打ち放しのコンクリート（1 枚＝横 2 m・縦 2 m、板の幅 0.15 m ごとに横の継ぎ目と木目）。"""
    rng = np.random.default_rng(seed)
    g = 0.70 + 0.08 * (periodic_noise(w, 40, rng) - 0.5) + 0.05 * (periodic_noise(w, 3, rng) - 0.5)   # 写真では日の当たる壁はほぼ白
    yy = np.arange(h)[:, None]; board = h * 0.15 / 2.0
    g = g - 0.07 * (yy % board < 2) + 0.03 * np.sin(np.arange(w)[None, :] / w * 2 * np.pi * 9 + (yy // board) * 1.7)
    return _u8(np.stack([g * 1.0, g * 0.99, g * 0.95], -1)[:h, :w])


def _shingles(w=512, h=512, seed=48):
    """苔の付いた茶色い屋根板（段の継ぎ目）。"""
    rng = np.random.default_rng(seed)
    n = periodic_noise(w, 24, rng); f = periodic_noise(w, 2, rng)
    g = 0.28 + 0.08 * (n - 0.5) + 0.05 * (f - 0.5)   # 写真では暗い焦げ茶
    g = g * np.where(np.arange(h)[:, None] % 32 < 3, 0.7, 1.0)
    moss = np.clip(n - 0.55, 0, 1) * 0.6
    return _u8(np.stack([g * (1.05 - moss * 0.4), g * (0.92 + moss * 0.3), g * (0.82 - moss * 0.2)], -1))


def terayama_toilet_front_texture(w_px, h_px):
    """寺山富士見橋のトイレの正面（南）：打ち放しのコンクリートの壁、真ん中に V 字に切れ込んだ入口（奥に木の扉）、
    左に掲示板と黒い銘板（ユーザーの写真 2025-06〜09）。"""
    from PIL import ImageDraw
    base = Image.fromarray(_board_concrete(w_px, h_px)) if w_px == h_px else Image.fromarray(_board_concrete(max(w_px, h_px), max(w_px, h_px))).crop((0, 0, w_px, h_px))
    d = ImageDraw.Draw(base); W, H = w_px, h_px; X = lambda f: int(W * f); Y = lambda f: int(H * f)
    d.polygon([(X(0.40), 0), (X(0.60), 0), (X(0.57), H), (X(0.43), H)], fill=(52, 46, 40))          # 入口の暗い奥
    d.rectangle([X(0.455), Y(0.25), X(0.545), H], fill=(196, 140, 78))                                # 奥の木の扉
    d.line([(X(0.5), Y(0.25)), (X(0.5), H)], fill=(120, 84, 50), width=max(2, W // 300))
    d.rectangle([X(0.14), Y(0.22), X(0.30), Y(0.55)], fill=(70, 66, 60))                               # 掲示板の枠
    d.rectangle([X(0.15), Y(0.24), X(0.29), Y(0.53)], fill=(232, 230, 214))
    for k in range(4):
        d.rectangle([X(0.16), Y(0.27 + k * 0.06), X(0.28 - 0.02 * (k % 2)), Y(0.30 + k * 0.06)], fill=(60, 90, 160) if k == 1 else (90, 90, 90))
    d.rectangle([X(0.17), Y(0.63), X(0.27), Y(0.80)], fill=(48, 48, 46))                                # 黒い銘板
    d.rectangle([X(0.66), Y(0.30), X(0.71), Y(0.42)], fill=(232, 190, 40))                              # 入口の右の黄色い貼り紙
    return base


def toilet_front_texture(w_px, h_px):
    """峠のトイレの正面（南）：白い壁と柱、焦げ茶の縦板の面、掲示板 2 枚、男女の入口（暗い）、白い腰（ストリートビュー 012°・072°）。"""
    from PIL import ImageDraw
    im = Image.new("RGB", (w_px, h_px), (232, 230, 222)); d = ImageDraw.Draw(im)
    W, H = w_px, h_px; X = lambda f: int(W * f); Y = lambda f: int(H * f)
    brown = _boards(X(0.62), Y(0.62), (70, 46, 32), board_px=max(8, W // 60))
    im.paste(brown, (X(0.06), Y(0.12)))
    d.rectangle([0, Y(0.74), W, H], fill=(226, 224, 216))                       # 白い腰
    d.line([(0, Y(0.74)), (W, Y(0.74))], fill=(150, 146, 138), width=max(2, H // 120))
    for x0, x1 in ((0.10, 0.30), (0.36, 0.60)):                                 # 掲示板（黒っぽい枠に貼り紙）
        d.rectangle([X(x0), Y(0.22), X(x1), Y(0.52)], fill=(56, 44, 34), outline=(30, 24, 18), width=3)
        for k in range(6):
            px0 = X(x0) + 6 + (k % 3) * (X(x1 - x0) // 3); py0 = Y(0.25) + (k // 3) * Y(0.13)
            d.rectangle([px0, py0, px0 + X(x1 - x0) // 3 - 10, py0 + Y(0.11)], fill=(236, 232, 214))
    for x0, col in ((0.70, (40, 60, 120)), (0.84, (150, 40, 50))):              # 男・女の入口
        d.rectangle([X(x0), Y(0.30), X(x0 + 0.11), Y(1.0)], fill=(34, 32, 30))
        d.rectangle([X(x0 + 0.025), Y(0.20), X(x0 + 0.085), Y(0.27)], fill=col)
    return im


def _fire_cistern_sign(px=256):
    """防火水そうの標識：赤い丸に白い細い縁、上に小さく FIRE CISTERN、真ん中に「防火水そう」、下に容量（まわりは透明）。"""
    from PIL import ImageDraw
    im = Image.new("RGBA", (px, px), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    d.ellipse([3, 3, px - 3, px - 3], fill=(245, 245, 245, 255)); d.ellipse([9, 9, px - 9, px - 9], fill=(200, 32, 40, 255))
    white = (250, 250, 250, 255)
    d.text((px / 2, px * 0.26), "FIRE CISTERN", fill=white, font=_font(int(px * 0.075)), anchor="mm")
    f = _font(int(px * 0.20)); t = "防火水そう"
    w = d.textlength(t, font=f); f = _font(int(px * 0.20 * min(1.0, px * 0.80 / w)))
    d.text((px / 2, px * 0.50), t, fill=white, font=f, anchor="mm")
    d.text((px / 2, px * 0.74), "40m³", fill=white, font=_font(int(px * 0.08)), anchor="mm")
    return np.array(im)


def wood_sign_texture(lines, w_px, h_px):
    """木を彫って白く塗った看板（灰色がかった古い板、横に板目）。lines = [{"text", "size"(高さの割合), "band"(帯の色・任意),
    "font"(gothic_bold | mincho・任意)}]。帯のある行は、帯の中に明るい字。"""
    from PIL import ImageDraw, ImageFont
    from . import config
    a = np.asarray(Image.open(config.PROJECT_ROOT / "assets" / "free" / "japanese_cedar_planks_1k.jpg").convert("RGB"), float)
    g = a.mean(2, keepdims=True)
    a = np.clip((0.30 * a + 0.70 * g) * [0.92, 0.88, 0.80] + 40, 0, 255).astype(np.uint8)   # 日に焼けて灰色がかった板
    im = Image.fromarray(np.rot90(a)).resize((w_px, h_px)).convert("RGB"); d = ImageDraw.Draw(im)   # 板目は横
    for k in range(1, 4):   # 横に4枚の板の継ぎ目
        y = h_px * k // 4; d.line([(0, y), (w_px, y)], fill=(70, 62, 54), width=max(1, h_px // 120))
    gap = 1.45   # 行の送り（字の高さの倍）
    total = sum(float(l["size"]) for l in lines) * gap
    y = (h_px - total * h_px) / 2
    for l in lines:
        sz = float(l["size"]) * h_px; font_kind = l.get("font", "gothic_bold")
        f = ImageFont.truetype(config.font_file(font_kind), int(sz))
        w = d.textlength(l["text"], font=f)
        if w > w_px * 0.88:
            f = ImageFont.truetype(config.font_file(font_kind), int(sz * w_px * 0.88 / w)); w = d.textlength(l["text"], font=f)
        cy = y + sz * 0.56
        if l.get("band"):
            pad = sz * 0.25
            d.rectangle([(w_px - w) / 2 - pad, cy - sz * 0.62, (w_px + w) / 2 + pad, cy + sz * 0.62], fill=_rgb(l["band"]))
        d.text((w_px / 2 + 2, cy + 2), l["text"], fill=(60, 52, 44), font=f, anchor="mm")    # 彫りの影
        d.text((w_px / 2, cy), l["text"], fill=(238, 234, 224), font=f, anchor="mm")
        y += sz * gap
    return im


def nozura_stones():
    """野面石積みの写真（Poly Haven「Japanese Stone Wall」CC0、assets/dress/nozura.png。峠の石積みと同じ素材）。"""
    from . import config
    return np.asarray(Image.open(config.PROJECT_ROOT / "assets" / "dress" / "nozura.png").convert("RGB").resize((512, 512)))


def _boards(w_px, h_px, rgb, board_px=22, seed=5):
    """縦の板張り（古い焦げ茶の板）。"""
    rng = np.random.default_rng(seed)
    x = np.arange(w_px)
    shade = 0.82 + 0.18 * rng.random(w_px // board_px + 2)[x // board_px]
    shade = shade * np.where(x % board_px < 2, 0.55, 1.0)
    g = shade[None, :] * (0.92 + 0.08 * rng.random((h_px, w_px)))
    return Image.fromarray(np.clip(np.array(rgb, float)[None, None, :] * g[..., None], 0, 255).astype(np.uint8))


def shop_front_texture(w_px, h_px):
    """峠の売店の正面（看板建築）：上に大きな看板、下は奥まった軒下に梁・柱・入口・窓・写真のパネル・OPEN の札（ユーザーの写真から）。"""
    from PIL import ImageDraw
    im = _boards(w_px, h_px, (74, 59, 46)); d = ImageDraw.Draw(im)
    W, H = w_px, h_px; Y = lambda f: int(H * f); X = lambda f: int(W * f)
    d.rectangle([X(0.01), Y(0.02), X(0.99), Y(0.24)], fill=(62, 50, 40), outline=(40, 32, 26), width=max(2, H // 150))
    white = (236, 230, 218)
    d.text((X(0.17), Y(0.085)), "国民宿舎", fill=white, font=_font(Y(0.055)), anchor="mm")
    d.text((X(0.17), Y(0.170)), "丹沢ホーム", fill=white, font=_font(Y(0.055)), anchor="mm")
    d.text((X(0.63), Y(0.13)), "ヤビツ峠売店", fill=white, font=_font(Y(0.115)), anchor="mm")
    d.rectangle([X(0.04), Y(0.26), X(0.96), Y(1.0)], fill=(40, 32, 26))                 # 奥まった軒下（暗い）
    d.rectangle([X(0.04), Y(0.26), X(0.96), Y(0.30)], fill=(26, 20, 16))                 # 軒の影
    d.rectangle([X(0.04), Y(0.36), X(0.96), Y(0.40)], fill=(110, 58, 42))                # 赤茶の梁
    d.rectangle([X(0.47), Y(0.36), X(0.50), Y(1.0)], fill=(70, 54, 40))                  # 真ん中の柱
    d.rectangle([X(0.0), Y(0.24), X(0.04), Y(1.0)], fill=(70, 56, 44)); d.rectangle([X(0.96), Y(0.24), X(1.0), Y(1.0)], fill=(70, 56, 44))
    for i in range(6):                                                                    # 写真を貼ったパネル
        for j in range(4):
            x0, y0 = X(0.06 + i * 0.04), Y(0.62 + j * 0.06)
            d.rectangle([x0, y0, x0 + X(0.034), y0 + Y(0.05)], fill=(150 + 15 * ((i + j) % 3), 150, 145))
    d.rectangle([X(0.31), Y(0.50), X(0.45), Y(0.56)], fill=(232, 224, 200))              # 入口の上の白い箱
    d.rectangle([X(0.315), Y(0.56), X(0.445), Y(1.0)], fill=(225, 222, 212))             # 入口の白い枠
    d.rectangle([X(0.33), Y(0.58), X(0.43), Y(1.0)], fill=(48, 52, 54))                  # 入口のガラス
    d.rectangle([X(0.52), Y(0.50), X(0.70), Y(0.56)], fill=(232, 224, 200))
    d.rectangle([X(0.52), Y(0.58), X(0.70), Y(0.86)], fill=(225, 222, 212))              # 窓の白い枠
    d.rectangle([X(0.535), Y(0.60), X(0.685), Y(0.84)], fill=(60, 66, 62))
    d.rectangle([X(0.452), Y(0.63), X(0.468), Y(0.68)], fill=(205, 30, 36))              # OPEN の札
    return im


def annex_front_texture(w_px, h_px):
    """売店の右の小屋の正面：上にクリーム色の幕板、下は暗い中（前に自動販売機が立つ）。"""
    from PIL import ImageDraw
    im = _boards(w_px, h_px, (74, 59, 46), seed=6); d = ImageDraw.Draw(im)
    d.rectangle([0, 0, w_px, int(h_px * 0.2)], fill=(230, 220, 194))
    d.rectangle([int(w_px * 0.06), int(h_px * 0.2), int(w_px * 0.94), h_px], fill=(42, 34, 28))
    return im


def sign_texture(text, bg, fg, w_px, h_px, border=None):
    """看板の絵：背景色 bg に、文字 fg を枠いっぱいの大きさで中央に書く（日本語のフォント）。"""
    from PIL import ImageDraw
    im = Image.new("RGB", (w_px, h_px), _rgb(bg)); d = ImageDraw.Draw(im)
    if border:
        b = max(2, h_px // 16); d.rectangle([b // 2, b // 2, w_px - b // 2, h_px - b // 2], outline=_rgb(border), width=b)
    lines = text.split("\n")
    scale = [0.4 if s.isascii() and any(not t.isascii() for t in lines) else 1.0 for s in lines]   # 和文の下の英文は 0.4 倍の大きさ
    ref = _font(100)
    size = int(min(h_px * 0.8 / sum(scale), *(w_px * 0.9 / max(d.textlength(s, font=ref) / 100 * k, 1e-6) for s, k in zip(lines, scale))))
    y = (h_px - size * sum(scale) * 1.05) / 2
    for s, k in zip(lines, scale):
        f = _font(max(8, int(size * k)))
        tw = d.textlength(s, font=f)
        d.text(((w_px - tw) / 2, y), s, font=f, fill=_rgb(fg)); y += size * k * 1.05
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
            "wood_cedar": cedar_planks(),
            "wood_bars": railing_bars(rgb=(0.30, 0.19, 0.12), bars=4, bar_px=14),
            "roof_metal": _gray(256, 16, 0.50, 0.06, (0.84, 1.0, 0.94), coarse=24),
            "roof_blue": _gray(256, 22, 0.42, 0.05, (0.55, 0.80, 1.25), coarse=24),
            "roof_red": _gray(256, 23, 0.45, 0.05, (1.35, 0.62, 0.50), coarse=24),
            "tiles": tiles(),
            "paint_red": _gray(256, 18, 0.62, 0.08, (1.0, 0.42, 0.36)),
            "paint_green": _gray(256, 24, 0.55, 0.08, (0.64, 0.78, 0.62)),
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
            "sign_stop": _stop_sign(),
            "sign_fire_cistern": _fire_cistern_sign(),
            "pole_white": _gray(128, 40, 0.90, 0.03),
            "bamboo": _gray(128, 41, 0.62, 0.07, (1.18, 1.02, 0.66), coarse=24),
            "pole_redwhite": _bands(((200, 40, 36), (236, 236, 230)), 8),
            "vending_red": _gray(128, 42, 0.62, 0.04, (1.40, 0.22, 0.20)),
            "stone_nozura": nozura_stones(),
            "tunnel_wall": _gray(512, 38, 0.46, 0.10, (1.0, 0.99, 0.96), coarse=10, fine=2),
            "tunnel_lamp": _gray(64, 39, 0.97, 0.02, (1.04, 0.88, 0.55)),
            "pavers_brick": herringbone_bricks(),
            "delineator": _delineator(),
            "drum_checker": _checker(((205, 70, 80), (238, 236, 230)), 4, 2),
            "siding_white": _siding(),
            "rubber_black": _gray(64, 45, 0.12, 0.03),
            "concrete_board": _board_concrete(),
            "roof_shingle": _shingles(),
            "pole_green": _gray(64, 46, 0.42, 0.04, (0.45, 1.05, 0.62))}
    paths = {}
    for name, a in imgs.items():
        paths[name] = out / Path(MATERIALS[name]["texture"]).name
        Image.fromarray(a).save(paths[name])
    return paths


def register_materials(m):
    for name, spec in MATERIALS.items():
        m.add_material(name, texture=spec["texture"], alpha=spec.get("alpha", False), two_sided=spec.get("two_sided", False))
