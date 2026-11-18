"""菜の花台入口の石碑・木製園地名看板。写真は観察用、自作画像だけを使用。

自作画像: .venv/bin/python blender_scripts/build_nanohanadai_entrance.py textures
形状生成: build(asset_id) を nanohanadai_common のシーンから呼び出す。
保存・ベイク・ゲーム用UVへの変換は nanohanadai_common 側で行う。
"""
from pathlib import Path
import math
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
STONE = ROOT / "assets/landmarks/nanohanadai_monument"
SIGN = ROOT / "assets/landmarks/nanohanadai_park_sign"


def textures():
    """写真の画素を使わない石肌・銘板・板の印字。Pillowは外部Pythonのみ。"""
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont, ImageFilter

    STONE.mkdir(parents=True, exist_ok=True)
    SIGN.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(103003)
    rr = random.Random(103003)
    w, h = 1024, 1536
    yy, xx = np.mgrid[0:h, 0:w]
    # 写真の大きな石は黄褐色の層理と明るい細い石英脈を持つ。
    coarse = np.asarray(Image.fromarray(np.uint8(rng.uniform(0, 255, (30, 20)))).resize((w, h), Image.Resampling.BICUBIC), dtype=float)
    medium = np.asarray(Image.fromarray(np.uint8(rng.uniform(0, 255, (120, 80)))).resize((w, h), Image.Resampling.BICUBIC), dtype=float)
    streak = np.sin(xx / 37 + np.sin(yy / 171) * 1.3) * 7
    lum = (coarse - 127) * .20 + (medium - 127) * .10 + streak + rng.normal(0, 3.2, (h, w))
    arr = np.clip(np.array([150, 135, 108])[None, None, :] + lum[..., None], 0, 255).astype('uint8')
    im = Image.fromarray(arr)
    veins = Image.new('RGBA', (w, h), (0, 0, 0, 0)); dv = ImageDraw.Draw(veins)
    # 横脈は銘板上の高さにまとまり、縦脈は曲がりながら枝分かれする。
    for i in range(21):
        x = rr.randint(-60, w + 60); points = []
        for y in range(-20, h + 50, 35):
            x += rr.uniform(-14, 14)
            points.append((x, y))
        dv.line(points, fill=(218, 208, 178, rr.randint(60, 140)), width=rr.randint(2, 6))
    points = []
    for x in range(-10, w + 35, 28):
        points.append((x, 880 + 13 * math.sin(x / 53) + rr.uniform(-9, 9)))
    dv.line(points, fill=(235, 231, 212, 175), width=7)
    for start in [(90, 820), (550, 858), (720, 892), (920, 884)]:
        x, y = start
        dv.line([(x, y), (x - 35, y - 60), (x - 21, y - 102), (x - 60, y - 140)], fill=(234, 231, 216, 155), width=5)
    # 雨筋は表面の色のみ。形の代わりにはしない。
    im = Image.alpha_composite(im.convert('RGBA'), veins.filter(ImageFilter.GaussianBlur(.6))).convert('RGB')
    im.save(STONE / 'stone_front.png')

    # 判読できない文字は作らない。離散した短い刻線の密度だけを近似。
    for name, wh, bg, seed in [('small_plaque', (256, 256), (115, 119, 112), 730), ('lower_plaque', (1024, 768), (80, 85, 80), 731)]:
        rw, rh = wh; rr = random.Random(seed)
        a = np.clip(np.array(bg)[None, None, :] + rng.normal(0, 4, (rh, rw, 1)), 0, 255).astype('uint8')
        pl = Image.fromarray(a); dp = ImageDraw.Draw(pl)
        dp.rectangle((3, 3, rw - 4, rh - 4), outline=tuple(int(c * .7) for c in bg), width=3)
        if name == 'small_plaque':
            for j in range(24):
                x = rr.randint(55, rw - 55); y = rr.randint(45, rh - 50)
                dp.line((x, y, x + rr.randint(-7, 15), y + rr.randint(4, 23)), fill=(148, 150, 139), width=rr.choice([2, 3, 4]))
        else:
            for j in range(9):
                x = 32 + j * 107
                for q in range(5):
                    dp.line((x, 37 + q * 7, x + rr.randint(39, 72), 37 + q * 7), fill=(179, 178, 158), width=3)
            for j in range(85):
                x = rr.randint(40, rw - 50); y = rr.randint(130, rh - 30)
                dp.line((x, y, x + rr.randint(4, 20), y), fill=(112, 118, 106), width=1)
            # 低解像度写真で見える中央の淡い摩耗模様。意味のある図柄・文章ではない。
            dp.polygon([(220, 361), (260, 358), (277, 391), (332, 395), (361, 379), (389, 411), (410, 390), (422, 354), (447, 340), (461, 348), (449, 381), (433, 404), (435, 474), (412, 480), (405, 433), (362, 425), (330, 418), (305, 433), (275, 425), (269, 395), (237, 394)], fill=(175, 175, 151))
        pl.save(STONE / (name + '.png'))

    # 園地名の板。3枚の板の継ぎ目は形状にも作り、印字は写真で読める2行のみ。
    w, h = 1536, 768
    yy, xx = np.mgrid[0:h, 0:w]
    grain = 3 * np.sin(yy * .27 + 2 * np.sin(xx / 190)) + 2 * np.sin(yy * .091 + xx / 350)
    broad = np.asarray(Image.fromarray(np.uint8(rng.uniform(0, 255, (54, 4)))).resize((w, h), Image.Resampling.BICUBIC), dtype=float)
    lum = grain + (broad - 127) * .08 + rng.normal(0, 1.8, (h, w))
    a = np.uint8(np.clip(np.array([75, 52, 41])[None, None, :] + lum[..., None], 0, 255))
    board = Image.fromarray(a); d = ImageDraw.Draw(board)
    for y in [250, 507]:
        d.line((0, y, w, y), fill=(40, 29, 22), width=4)
        d.line((0, y + 5, w, y + 5), fill=(101, 74, 54), width=2)
    fonts = list(Path('/System/Library/Fonts').glob('*明朝*'))
    jpfont = ImageFont.truetype(str(fonts[0]), 195, index=2)
    roman = ImageFont.truetype('/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf', 78)
    for text, font, cy in [('菜ノ花台園地', jpfont, 370), ('Nanohanadai Picnic Site', roman, 582)]:
        bb = d.textbbox((0, 0), text, font=font)
        d.text(((w - (bb[2] - bb[0])) / 2 - bb[0], cy - (bb[3] - bb[1]) / 2 - bb[1]), text, font=font, fill=(232, 227, 212))
    board.save(SIGN / 'park_name.png')

    # 写真で視認できるトイレ・Pの案内。細かい説明やマークは判読不能なので省略。
    icons = Image.new('RGB', (768, 192), (78, 54, 42)); di = ImageDraw.Draw(icons)
    sans = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial Bold.ttf', 125)
    for j in range(3):
        di.rectangle((8 + j * 253, 10, 245 + j * 253, 180), fill=(229, 229, 215))
    di.text((586, 7), 'P', font=sans, fill=(48, 51, 45))
    # 男女のピクトグラムは人体の丸と直線を自作する。
    for cx in (315, 440):
        di.ellipse((cx - 14, 35, cx + 14, 63), fill=(45, 48, 42))
        di.rectangle((cx - 15, 70, cx + 15, 118), fill=(45, 48, 42))
        di.line((cx - 8, 116, cx - 10, 154), fill=(45, 48, 42), width=10)
        di.line((cx + 8, 116, cx + 10, 154), fill=(45, 48, 42), width=10)
    di.polygon([(421, 82), (458, 82), (468, 130), (412, 130)], fill=(45, 48, 42))
    di.polygon([(72, 74), (177, 74), (177, 50), (215, 96), (177, 142), (177, 118), (72, 118)], fill=(45, 48, 42))
    icons.save(SIGN / 'amenity_icons.png')


def _front_uv(ob, width, height, x0, z0):
    """正面のXYとは独立したXZ投影。UVMapはゲーム用BakeUVから分離。"""
    uv = ob.data.uv_layers['UVMap'].data
    for p in ob.data.polygons:
        for li in p.loop_indices:
            v = ob.data.vertices[ob.data.loops[li].vertex_index].co
            uv[li].uv = ((v.x - x0) / width, (v.z - z0) / height)


def _slab(C, name, outline, front_y, back_y, mat):
    """実形状の外周を持つ閉じた押し出し。正面−Y、奥に厚みを持たせる。"""
    n = len(outline)
    verts = [(x, front_y, z) for x, z in outline] + [(x, back_y, z) for x, z in outline]
    faces = [tuple(range(n - 1, -1, -1)), tuple(range(n, 2 * n))]
    faces += [(i, (i + 1) % n, (i + 1) % n + n, i + n) for i in range(n)]
    return C.mesh(name, verts, faces, mat)


def _monument(C):
    import bpy
    rr = random.Random(103005)
    objects = []
    body_col = C.collection('01 石碑本体と石台')
    stone = C.proc('黄褐色の自然石', (150, 135, 108), scale=7)
    edge = C.proc('石台の灰褐色', (132, 128, 110), scale=9)
    front_mat = C.image_material('自作_石英脈のある正面', STONE / 'stone_front.png')
    # 写真road225の本体は上部がやや細く、左上の肩が低く中央の頂点が高い。
    outline = [(-1.33, .20), (-1.30, .75), (-1.20, 1.42), (-1.14, 2.19),
               (-1.08, 2.96), (-.86, 3.06), (-.62, 3.04), (-.31, 3.17),
               (-.08, 3.25), (.12, 3.20), (.35, 3.23), (.59, 3.16),
               (.91, 3.10), (1.05, 2.91), (1.10, 2.32), (1.23, 1.49),
               (1.31, .75), (1.32, .20)]
    # 前面・厚い側面・欠けた背面を別リングにして、直方体の外形を避ける。
    n = len(outline)
    verts = [(x * .962, -.385, z) for x, z in outline]
    verts += [(x, -.27 + rr.uniform(-.07, .035), z + rr.uniform(-.025, .025)) for x, z in outline]
    verts += [(x * .88 + .04, .40 + rr.uniform(-.05, .04), z * .987) for x, z in outline]
    faces = [tuple(range(n - 1, -1, -1))]
    for ring in range(2):
        faces += [(ring*n+i, ring*n+(i+1)%n, (ring+1)*n+(i+1)%n, (ring+1)*n+i) for i in range(n)]
    faces.append(tuple(range(2*n, 3*n)))
    body = C.mesh('石碑本体_不規則外周と厚み', verts, faces, stone)
    body.data.materials.append(front_mat)
    # C.meshの再法線計算後も先頭ポリゴンは正面のまま。
    body.data.polygons[0].material_index = 1
    _front_uv(body, 2.66, 3.25, -1.33, 0)
    objects.append(body)
    # ne000/045で石の低い台が確認できる。正面写真では植込みに大半が隠れる。
    base_outline = [(-1.56, 0), (-1.52, .17), (-1.45, .23), (1.39, .23), (1.53, .16), (1.55, 0)]
    base = _slab(C, '石台_低い自然石基壇', base_outline, -.56, .55, edge)
    objects.append(base)

    plaques_col = C.collection('02 銘板と鳥形レリーフ')
    patina = C.proc('褐色の金属レリーフ', (108, 79, 60), scale=18)
    border = C.proc('銘板の鋳物縁', (103, 108, 100), scale=25)
    small_mat = C.image_material('自作_判読不能の小銘板刻線', STONE / 'small_plaque.png')
    low_mat = C.image_material('自作_下の大きい銘板', STONE / 'lower_plaque.png')
    for row, z in enumerate([2.035, 1.605]):
        for j, x in enumerate([-.88, -.41, .06, .53]):
            objects.append(C.box(f'小銘板_{row+1}_{j+1}_厚み', (x, -.403, z), (.29, .04, .245), border, bevel=.005))
            objects.append(C.front_panel(f'小銘板_{row+1}_{j+1}_刻線', (x, -.426, z), .267, .222, small_mat))
    objects.append(C.box('下部銘板_厚み', (-.15, -.405, .81), (1.77, .044, 1.14), border, bevel=.009))
    objects.append(C.front_panel('下部銘板_摩耗と刻線', (-.15, -.430, .81), 1.735, 1.105, low_mat))
    # 3つの鳥形の薄い鋳物。写真で見える翼の折れと浮いた厚みを形にする。
    bird_shapes = [
        [(-.89, 2.77), (-.75, 2.58), (-.79, 2.42), (-.86, 2.53), (-1.00, 2.67)],
        [(-.54, 2.98), (-.61, 2.85), (-.71, 2.78), (-.60, 2.82)],
        [(-.47, 2.79), (-.26, 2.62), (-.11, 2.56), (.03, 2.79), (-.07, 2.43), (-.14, 2.54), (-.30, 2.61)],
        [(.92, 2.77), (.87, 2.46), (.70, 2.30), (.82, 2.57)],
    ]
    for i, poly in enumerate(bird_shapes):
        objects.append(_slab(C, f'鳥形レリーフ_{i+1}', poly, -.47, -.391, patina))

    C.collection('03 足元の小さい植込み')
    leafmats = [C.proc('葉_黄緑', (101, 121, 49), scale=12), C.proc('葉_深緑', (64, 89, 39), scale=12), C.proc('葉_明緑', (126, 139, 60), scale=12)]
    # 植込みは輪郭の低いまとまり。個々の樹種は資料から特定できない。
    lv, lf, mi = [], [], []
    for k in range(1800):
        ang = rr.uniform(0, math.tau)
        x = rr.uniform(-1.64, 1.64); y = rr.uniform(-.82, .69)
        # 本体正面の下部銘板を埋めない。横と前端の草木だけ。
        if abs(x) < 1.15 and y > -.48:
            continue
        z = rr.uniform(.05, .37); length = rr.uniform(.065, .13); width = length * rr.uniform(.27, .46)
        dx, dy = math.cos(ang), math.sin(ang)
        px, py = -dy * width / 2, dx * width / 2
        start = len(lv)
        lv.extend([(x - dx * length / 2, y - dy * length / 2, z),
                   (x + px, y + py, z + .026),
                   (x + dx * length / 2, y + dy * length / 2, z + .045),
                   (x - px, y - py, z + .026),
                   (x, y, z + .048)])
        # 中央の折れを持つ葉。裏面も逆向きの面を持つ。
        for q in range(4):
            f = (start + q, start + (q + 1) % 4, start + 4)
            lf.extend([f, f[::-1]]); mi.extend([k % 3, k % 3])
    leaves = C.mesh('足元の植込み_両面葉', lv, lf, leafmats[0])
    for m in leafmats[1:]: leaves.data.materials.append(m)
    for p, i in zip(leaves.data.polygons, mi): p.material_index = i
    objects.append(leaves)
    return {
        'dimension_basis': 'road225/270/315の本体幅高さ比・隣の看板とガードレール高との比から推定。石本体約2.66×0.85×3.25m、基壇約3.11×1.11×0.23m。現地実測ではない。',
        'approximations': ['本体正面の輪郭・銘板4列×2段・下部大銘板・鳥形レリーフは複数写真から形状を近似。', '小銘板・下部銘板の文字は未判読のため創作せず、短い刻線と摩耗模様で外観だけ近似。', '背面の厚み、石英脈・雨筋、植込みの樹種と分布は推定。大きい周辺樹木は本アセットに含めない。', 'レリーフの鳥形は写真で見える翼形からの近似で、作者・作品名・細部意匠は未確認。'],
        'refs': ['sv_road_225.png', 'sv_road_270.png', 'sv_road_315.png', 'sv_ne_000.png', 'sv_ne_045.png'],
        'origin': '石台の根元中央（本体と植込みは同じ原点）。正面−Yは県道側。',
        'reference_views': {'reference_road225': (1, -2.7, .18), 'reference_ne045_back': (-1, 2.7, .18)},
        'parts': {'monument': objects},
        'generator': 'blender_scripts/build_nanohanadai_entrance.py',
    }


def _park_sign(C):
    objects = []
    C.collection('01 二本の柱と木の板')
    wood = C.proc('古い濃褐色木材', (77, 54, 42), kind='wood', scale=5)
    iron = C.proc('看板留めボルト', (79, 77, 65), scale=20)
    name_mat = C.image_material('自作_菜ノ花台園地_印字', SIGN / 'park_name.png')
    icon_mat = C.image_material('自作_トイレと駐車場案内', SIGN / 'amenity_icons.png')
    for x in [-.96, .96]:
        objects.append(C.box('看板柱_左' if x < 0 else '看板柱_右', (x, .09, .86), (.16, .18, 1.72), wood, bevel=.009))
    for j in range(3):
        z = .38 + j * .31
        objects.append(C.box(f'横板_{j+1}', (0, -.035, z), (2.12, .105, .304), wood, bevel=.004))
        # 印字用の面は板ごとにUVの該当帯だけを使う。隙間に文字を描かない。
        panel = C.front_panel(f'横板_{j+1}_印字', (0, -.090, z), 2.105, .296, name_mat)
        for li in panel.data.polygons[0].loop_indices:
            co = panel.data.uv_layers['UVMap'].data[li].uv
            co.y = j / 3 + co.y / 3
        objects.append(panel)
    top_outline = [(-1.06, 1.153), (-1.06, 1.61), (-.78, 1.64), (-.48, 1.57), (-.24, 1.48), (.13, 1.43), (.46, 1.43), (.79, 1.48), (1.06, 1.50), (1.06, 1.153)]
    objects.append(_slab(C, '看板上端_曲線の飾り板', top_outline, -.087, .02, wood))
    objects.append(C.box('上部案内の木の裏板',(-.30,-.035,1.43),(1.22,.105,.25),wood))
    objects.append(C.front_panel('上端のトイレ_P案内', (-.30, -.091, 1.43), 1.19, .225, icon_mat))
    for x in [-.96, .96]:
        for z in [.38, .69, 1.0]:
            b = C.cylinder('木板の留めボルト', (x, -.094, z), .009, .006, iron, vertices=8)
            b.rotation_euler[0] = math.pi / 2
            objects.append(b)
    for ob in objects:ob.location.y-=.09
    return {
        'dimension_basis': 'road225/270の石碑に対する幅・高さとガードレール比から推定。板幅2.12m、2本柱を含む高さ1.72m。現地実測ではない。',
        'approximations': ['判読できる「菜ノ花台園地」「Nanohanadai Picnic Site」を自作の印字画像で再現。書体・文字送りは近似。', '上部の案内は視認できるトイレとPを自作ピクトグラムで近似。小さく判読不能の文章・意匠は省略。', '背面の横桟・柱の奥行き・木口と細かな留め具位置は推定。', '看板上端の浅い曲線は折線で近似。'],
        'refs': ['sv_road_225.png', 'sv_road_270.png'],
        'origin': '2本柱の根元の中点。正面−Yは県道側。',
        'reference_views': {'reference_road225': (1, -2.7, .18)},
        'parts': {'park_sign': objects},
        'generator': 'blender_scripts/build_nanohanadai_entrance.py',
    }


def build(asset_id):
    import nanohanadai_common as C
    if asset_id == 'nanohanadai_monument':
        return _monument(C)
    if asset_id == 'nanohanadai_park_sign':
        return _park_sign(C)
    raise ValueError(asset_id)


if __name__ == '__main__':
    args = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else sys.argv[1:]
    if args == ['textures']:
        textures()
    else:
        raise SystemExit('外部Pythonから textures、Blenderから build(asset_id) を使用')
