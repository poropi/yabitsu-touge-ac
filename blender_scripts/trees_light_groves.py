"""林の板に使う六つの透過絵。枝の向きと奥行きに沿って自作の小枝を束ねる。"""
from functools import lru_cache
from math import cos, pi, sin, sqrt
from pathlib import Path
from random import Random

import numpy as np
from PIL import Image, ImageDraw, ImageOps

SOURCE = Path(__file__).resolve().parents[1] / "assets/landmarks/trees_mountain"
WIDTH, HEIGHT = 1056, 1370
ROOT_Y = 1362
FILTER = Image.Resampling.LANCZOS


@lru_cache(maxsize=3)
def leaf_spray(species):
    src = Image.open(SOURCE / f"leaf_{species}.png").convert("RGBA")
    src = src.crop(src.getchannel("A").getbbox())
    src.thumbnail((96, 128), FILTER)
    pixels = np.array(src)
    opaque = pixels[:, :, 3] > 192
    base = {"sugi": (36, 51, 40), "hinoki": (40, 55, 41), "oak": (47, 62, 37)}[species]
    luma = pixels[:, :, 1].astype(float)
    # 元の小枝の葉脈や細い枝先の明暗を残す。
    shading = np.clip(luma / np.mean(luma[opaque]), 0.52, 1.48)
    pixels[:, :, :3] = shading[:, :, None] * np.array(base)
    return Image.fromarray(pixels)


@lru_cache(maxsize=1800)
def sprite(species, width, height, angle, value):
    im = leaf_spray(species).resize((width, height), FILTER)
    if angle:
        im = im.rotate(angle, Image.Resampling.BICUBIC, expand=True)
    px = np.array(im)
    px[:, :, :3] = np.minimum(255, px[:, :, :3] * (value / 10.0)).astype(np.uint8)
    return Image.fromarray(px)


def dab(canvas, species, x, y, width, height, angle, value):
    # 有限個の筆を大きさ・傾き・明暗ごとに用意する。
    brush = sprite(species, max(4, 4 * round(width / 4)), max(6, 6 * round(height / 6)),
                   15 * round(angle / 15), max(4, min(16, round(value * 10))))
    canvas.alpha_composite(brush, (round(x - brush.width / 2), round(y - brush.height / 2)))


def limb(draw, nodes, radii, bark, rng):
    """連続する枝の断面を結び、下の幹から枝先まで太さを連続させる。"""
    left, right = [], []
    for i, (x, y) in enumerate(nodes):
        a = nodes[max(0, i - 1)]
        b = nodes[min(len(nodes) - 1, i + 1)]
        dx, dy = b[0] - a[0], b[1] - a[1]
        r = radii[i] / max(.1, sqrt(dx * dx + dy * dy))
        left.append((x + dy * r, y - dx * r))
        right.append((x - dy * r, y + dx * r))
    draw.polygon(left + right[::-1], fill=bark)
    # 樹皮の縦溝は幹の曲がりに追従し、陰側は実際の幅の内側に収める。
    for fraction in (-.76, -.44, -.23, .12, .45, .74):
        tone = rng.uniform(.62, 1.25) + .12 * fraction
        color = tuple(round(v * tone) for v in bark)
        line = []
        for p, q in zip(left, right):
            f = (1 + fraction + rng.uniform(-.05, .05)) / 2
            line.append((p[0] * f + q[0] * (1 - f), p[1] * f + q[1] * (1 - f)))
        draw.line(line, fill=color, width=max(1, round(radii[0] * .12)))


def conifer(im, species, x, h, radius, rng, gain):
    draw = ImageDraw.Draw(im)
    drift = rng.uniform(-.017, .017) * h
    base_r = h * rng.uniform(.009, .013)
    bark = (52, 40, 34) if species == "sugi" else (56, 45, 38)
    axis = lambda t: x + drift * t + .0025 * h * sin(t * 7.3)
    ts = [0, .035, .15, .36, .56, .75, .91, 1]
    limb(draw, [(axis(t), ROOT_Y - t * h) for t in ts],
         [base_r * s for s in (1.5, 1.08, .95, .76, .58, .38, .18, .025)], bark, rng)
    is_cedar = species == "sugi"
    live = rng.uniform(.39, .53) if is_cedar else rng.uniform(.48, .64)
    # 生枝より下に短い枯れ枝や枝痕がところどころ残る。
    for _ in range(rng.randrange(8, 16)):
        t = rng.uniform(.24, 1 - live)
        z = ROOT_Y - t * h
        side = rng.choice((-1, 1))
        length = h * rng.uniform(.009, .040)
        draw.line([(axis(t), z), (axis(t) + length * side * .66, z - .18 * length),
                   (axis(t) + length * side, z - .38 * length)], fill=bark, width=2)

    boughs = []
    whorls = rng.randrange(16, 21)
    azimuth = rng.random() * 2 * pi
    for level in range(whorls):
        u = (level + .65) / whorls
        azimuth += rng.uniform(1.6, 3.2)
        for j in range(rng.randrange(5, 8)):
            a = azimuth + 2 * pi * j / 6 + rng.uniform(-.28, .28)
            t = 1 - live * (u + rng.uniform(-.014, .014))
            # 下段の老枝は先端だけ疎らになり、一段上の長枝が外郭を作る。
            envelope = (u ** (.79 if is_cedar else .53)) * (1 - .17 * u**6)
            length = radius * envelope * rng.uniform(.70, 1.15)
            boughs.append((sin(a), t, a, length))
    stamps = []
    for depth, t, a, length in boughs:
        cx, cy = axis(t), ROOT_Y - t * h
        dx = length * cos(a)
        droop = length * rng.uniform(.30, .62) if is_cedar else length * rng.uniform(-.02, .27)
        # 筆跡の中心を曲線上に置く。針葉は先で垂れ、鱗片葉は横へ扇を開く。
        def at(s):
            return (cx + dx * s, cy + droop * s ** 1.55 - length * .12 * sin(s * pi))
        if length > 24 and depth > -.25:
            draw.line([at(s) for s in (0, .33, .66, 1)], fill=bark, width=max(1, int(length / 60)))
        count = max(3, round(length / 5))
        for j in range(count):
            s = rng.uniform(.14, 1)
            bx, by = at(s)
            spread = sqrt(s) * (1 - .72 * s)
            bx += rng.gauss(0, .14) * length * spread
            by += rng.gauss(0, .13) * length * spread
            size = (h * .029 + .14 * length) * rng.uniform(.58, 1.12)
            if is_cedar:
                angle = -cos(a) * rng.uniform(38, 94) + rng.uniform(-22, 22)
                width, height = size * .85, size * 1.36
            else:
                angle = -cos(a) * rng.uniform(58, 109) + rng.uniform(-14, 14)
                width, height = size * 1.1, size * 1.22
            tone = gain * (.75 + .24 * s + .14 * cos(a + 1.8) + rng.uniform(-.13, .15))
            stamps.append((depth * s * length + rng.uniform(-4, 4), bx, by, width, height, angle, tone))
    for _, bx, by, w, h_spray, angle, tone in sorted(stamps):
        dab(im, species, bx, by, w, h_spray, angle, tone)


def oak(im, x, h, radius, rng, gain):
    draw = ImageDraw.Draw(im)
    # 芽吹きの位置を、根から連続する一本の曲がった主軸上に選ぶ。
    # 側枝の分岐点をずらすと、下から見上げた太い幹の空隙にも奥行きが出る。
    lean = rng.uniform(-.28, .28) * radius
    def axis(pos):
        return (x + lean * pos + radius * (.067 * sin(9.4 * pos) + .028 * sin(21.0 * pos)),
                ROOT_Y - h * pos)
    bark = (52, 46, 37)
    base_r = h * rng.uniform(.011, .015)
    levels = (0, .028, .13, .27, .39, .48, .59, .69, .77, .84, .90)
    limb(draw, [axis(t) for t in levels],
         [base_r * v for v in (1.6, 1.12, .97, .84, .74, .61, .47, .35, .22, .10, .02)], bark, rng)
    terminals = []
    phase = rng.uniform(0, 2*pi)
    for j in range(11):
        t = .25 + j * .051 + rng.uniform(-.025, .021)
        angle = phase + j * 2.4 + rng.uniform(-.35, .35)
        dx = cos(angle) * radius * (1.0 - .60 * t) * rng.uniform(.88, 1.24)
        rise = h * rng.uniform(.19, .30) * (1.25 - .7 * t)
        sx, sy = axis(t)
        branch = [(sx, sy), (sx + dx * .14, sy - .27 * rise),
                  (sx + dx * .46, sy - .47 * rise),
                  (sx + dx * .72, sy - .83 * rise),
                  (sx + dx, sy - rise)]
        limb(draw, branch, [base_r * f * (1-.44*t) for f in (.64, .53, .42, .25, .09)], bark, rng)
        # 二又を枝ごとに段違いに配置。先端より低い位置にも葉が残る。
        for k in range(4):
            joint = rng.choice((2, 3, 4))
            xx, yy = branch[joint]
            s = rng.choice((-1, 1))
            ex = xx + s * radius * rng.uniform(.16, .46)
            ey = yy - h * rng.uniform(.045, .13)
            elbow = (xx + .65 * (ex - xx), yy + .40 * (ey - yy))
            limb(draw, [(xx, yy), elbow, (ex, ey)], [base_r*.24, base_r*.10, .6], bark, rng)
            for tj in range(3):
                tx = ex + rng.uniform(-.16,.16) * radius
                ty = ey - rng.uniform(.018, .06) * h
                draw.line([(ex, ey), (tx, ty)], fill=bark, width=2)
                terminals.append((sin(angle) + rng.uniform(-.21, .21), tx, ty,
                                  radius * rng.uniform(.13,.25), rng.uniform(.80, 1.24)))
    for j in range(5):
        sx, sy = axis(.87 + j*.005)
        tx, ty = sx + radius * rng.uniform(-.17,.17), sy - h * rng.uniform(.04,.09)
        draw.line([(sx,sy),(tx,ty)], fill=bark, width=2)
        terminals.append((0, tx, ty, radius*.21, 1.07))
    # 葉群の内側から外縁へ、小枝をいくつかのまとまりに分けて描く。
    for depth, cx, cy, extent, local_gain in sorted(terminals):
        satellites = rng.randrange(4, 8)
        for _ in range(satellites):
            a = rng.random() * 2 * pi
            r = sqrt(rng.random()) * extent
            sx, sy = cx + r * cos(a), cy + r * sin(a) * .76
            fac = gain * local_gain * (.72 + .24 * depth + rng.uniform(-.04, .16))
            for _ in range(rng.randrange(4, 8)):
                rx = rng.uniform(-1, 1)
                ry = rng.uniform(-1, 1)
                size = h * rng.uniform(.016, .042)
                dab(im, "oak", sx + rx * extent * .42, sy + ry * extent * .32,
                    size, size * 1.04, rng.uniform(-120, 120), fac * (1 - .12 * ry))


# 根元の横位置、樹高、枝張り（描画用座標）。林ごとに手前・奥の順も変える。
GROVES = (
    (("sugi", 180, .77, 95), ("sugi", 480, .85, 93), ("sugi", 334, 1, 111), ("sugi", 566, .68, 81)),
    (("hinoki", 384, .71, 89), ("hinoki", 208, .90, 109), ("hinoki", 520, 1, 118)),
    (("hinoki", 335, .74, 116), ("sugi", 546, .88, 88), ("sugi", 195, 1, 94), ("hinoki", 435, .62, 102)),
    (("oak", 193, .75, 131), ("oak", 541, .60, 121), ("oak", 353, 1, 180)),
    (("oak", 399, .65, 117), ("oak", 225, 1, 159), ("oak", 501, .90, 147)),
    (("sugi", 470, 1, 96), ("hinoki", 258, .89, 98), ("oak", 154, .62, 110), ("oak", 537, .74, 114)),
)


def paint_groves() -> list[Image.Image]:
    paintings = []
    for index, group in enumerate(GROVES):
        scene = Image.new("RGBA", (WIDTH, HEIGHT))
        for order, (species, x, relative_height, radius) in enumerate(group):
            rng = Random(60211 + index * 177 + order * 31)
            tree = oak if species == "oak" else conifer
            args = () if species == "oak" else (species,)
            tree(scene, *args, x + 144, 1030 * relative_height, radius, rng, rng.uniform(.91, 1.09))
        # 広がった枝先も切らずに収める。幹の根元だけはタイルの下端に達する。
        x0, y0, x1, _ = scene.getchannel("A").getbbox()
        assert x0 > 0 and x1 < WIDTH and y0 > 0, "生成中の林が作画領域から出ています"
        content = scene.crop((x0, y0, x1, ROOT_Y + 1)).resize((504, 1020), FILTER)
        result = ImageOps.expand(content, border=(4, 4, 4, 0), fill=(0, 0, 0, 0))
        paintings.append(result)
    return paintings
