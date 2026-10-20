"""山道（表示の S2〜S5＝内部の s2・s1・s4・s5）の道路まわりの仕上げの部品。

- 壁の仕上げ：表（*_features.csv）の note から、石積み・苔むした石積み・間知ブロック・コンクリート・岩・落石防護網・法枠・吹付けを決める
- ガードレール：波形の断面（W 形の鋼板）、丸い支柱（4 m おき）、端は外へ曲げる。位置は今のガードレール（features.guardrail_meshes）と同じ線
- 落ち葉：路肩の帯（外ほど濃い）と、車線の上のまばらな葉。透ける画像（アルファテスト）を路面から少し浮かせて貼る
座標は manifest と同じ（x 東, y 北, z 上）。
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .crosssection import GUTTER_WIDTH, surface_height
from .features import GUARDRAIL_OFFSET, SIDES, rows_of
from .meshgen import ribbon

SECTIONS = ("s2", "s1", "s4", "s5")      # 内部の区間名（表示の S2・S3・S4・S5）


def wall_finish(note: str) -> str:
    """壁の note → 貼り替えのテクスチャのキー（wall_*）。"""
    n = note
    if "法枠" in n:
        return "norikae"
    if "吹付け" in n:
        return "shotcrete"
    if "土台" in n and "コンクリ" in n:
        return "concrete"
    if "落石防護網" in n or "ネット" in n:
        return "rock_net"
    if "岩" in n or "崖" in n:
        return "rock"
    if "ブロック" in n:
        return "kenchi_moss" if "苔" in n else "block"
    if "コンクリ" in n or ("擁壁" in n and "石" not in n):
        return "concrete"
    if "低い" in n and "苔" not in n:
        return "kenchi"        # 低い石積み（家の前・土手の下など）は苔が少ない
    return "kenchi_moss"       # 山道の石積み・石垣は、ストリートビューではほとんど苔むしている


# ---------- ガードレール ----------

# 断面：(道へ向けた張り出し m（負が道の側）, 道の縁からの高さ m)。下の縁 → 山 → 谷 → 山 → 上の縁（高さ 35 cm の W 形）
BEAM = [(0.0, 0.425), (-0.045, 0.46), (-0.065, 0.515), (-0.045, 0.57), (-0.02, 0.60), (-0.045, 0.63),
        (-0.065, 0.685), (-0.045, 0.74), (0.0, 0.775)]
POST_EVERY = 4.0
POST_R = 0.057                 # φ114 の鋼管
POST_TOP = 0.70
FLARE_M, FLARE_OUT = 1.6, 0.30   # 端の曲げ：最後の 1.6 m で外へ 30 cm
CHUNK_ROWS = 2500              # 1 メッシュの行数の上限（頂点 65535 まで）


def guardrail_line(sec, f):
    """ガードレールの基準線（今の当たり判定と同じ横の位置）。(rows, side, lat, ze)"""
    sl = rows_of(sec, f)
    rows = np.arange(sl.start, sl.stop)
    side = SIDES[f.side]
    hw = (sec.hw_left if side > 0 else sec.hw_right)[rows]
    lat = hw + GUTTER_WIDTH + GUARDRAIL_OFFSET
    s = sec.s[rows]
    d = np.minimum(s - s[0], s[-1] - s)                     # 端からの距離
    t = np.clip(1 - d / FLARE_M, 0, 1)
    lat = lat + FLARE_OUT * t * t                            # 端は外へ曲げる
    return rows, side, lat, sec.edge_z(side)[rows]


def guardrail_meshes(sec, f):
    """波形の板と支柱のメッシュ（行の多い所は分ける）。[(verts, faces, uvs, part)]。part は 'beam' か 'post'。"""
    rows, side, lat, ze = guardrail_line(sec, f)
    out = []
    off = np.array([p[0] for p in BEAM]); hz = np.array([p[1] for p in BEAM])
    for a in range(0, len(rows), CHUNK_ROWS):
        r = slice(a, min(a + CHUNK_ROWS + 1, len(rows)))
        if r.stop - r.start < 2:
            continue
        L = side * (lat[r][:, None] + off[None, :])
        Z = ze[r][:, None] + hz[None, :]
        rr = rows[r]
        v, fc, uv = ribbon(sec.x[rr], sec.y[rr], sec.heading[rr], L, Z, sec.s[rr], u_scale=0.5, v_scale=16.0, flip=side < 0)
        out.append((v, fc, uv, "beam"))
    # 支柱：板の後ろの丸い鋼管
    V, F, U = [], [], []
    s0, s1 = sec.s[rows[0]], sec.s[rows[-1]]
    for s in np.arange(s0 + 0.3, s1 - 0.29, POST_EVERY):
        k = int(np.clip(np.searchsorted(sec.s[rows], s), 0, len(rows) - 1))
        i = rows[k]
        h = sec.heading[i]
        n = np.array([-np.sin(h), np.cos(h)]) * side
        c = np.array([sec.x[i], sec.y[i]]) + n * (lat[k] + POST_R + 0.02)
        v, fc, uv = cylinder((*c, ze[k] - 0.3), POST_R, ze[k] + POST_TOP - (ze[k] - 0.3))
        F.append(fc + sum(len(x) for x in V)); V.append(v); U.append(uv)
    if V:
        out.append((np.concatenate(V), np.concatenate(F), np.concatenate(U), "post"))
    return out


def cylinder(p, r, h, sides=10):
    """縦の円柱（ふた付き）。p は下の中心。"""
    a = np.linspace(0, 2 * np.pi, sides + 1)
    ring = np.c_[np.cos(a) * r, np.sin(a) * r]
    bot = np.c_[ring + p[:2], np.full(len(a), p[2])]
    top = np.c_[ring + p[:2], np.full(len(a), p[2] + h)]
    v = np.r_[bot, top, [[p[0], p[1], p[2] + h]]]
    n = len(a)
    side = [[k, k + 1, n + k + 1, n + k] for k in range(sides)]
    cap = [[n + k, n + k + 1, 2 * n, 2 * n] for k in range(sides)]
    uv = np.r_[np.c_[a / (2 * np.pi), np.zeros(n)], np.c_[a / (2 * np.pi), np.ones(n)], [[0.5, 1.0]]]
    return v, np.array(side + cap), uv


def guardrail_texture(seed=11, w=128, h=2048, aged=False):
    """波形の板の画像：横＝断面（0.5 m）、縦＝長さ 16 m。白い塗装に、泥はね・雨だれ・錆のしみ、4 m ごとの継ぎ目のボルト。
    aged：古く黄ばんだもの（S4・S5 のストリートビュー）。"""
    rng = np.random.default_rng(seed)
    base = np.full((h, w, 3), (160, 155, 137) if aged else (226, 228, 224), float)
    from scipy.ndimage import gaussian_filter
    grime = gaussian_filter(rng.random((h, w)), (40, 6))
    grime = (grime - grime.min()) / np.ptp(grime)
    base *= (0.86 + 0.14 * grime)[..., None]
    low = np.clip(1 - np.linspace(0, 1, w) * 2.2, 0, 1)[None, :]                       # 下の縁ほど泥はね
    mud = gaussian_filter(rng.random((h, w)), (3, 2)) * low
    base = base * (1 - 0.35 * mud[..., None]) + np.array((120, 105, 85)) * 0.35 * mud[..., None]
    img = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(img)
    for k in range(int(rng.integers(5, 9) * (3 if aged else 1))):                           # 錆の縦のしみ（ボルトから下へ）
        y = int(rng.integers(0, h)); x = int(rng.integers(w // 4, w - 10))
        for j in range(18):
            d.line([(x + j % 3, y), (x + j % 3 - 2, y + 30 + j * 2)], fill=(150, 95, 60), width=1)
    for y in range(0, h, h // 4):                                                           # 4 m ごとの継ぎ目
        d.line([(0, y), (w, y)], fill=(170, 170, 165), width=3)
        for x in (w * 0.2, w * 0.4, w * 0.6, w * 0.8):
            d.ellipse((x - 3, y + 6, x + 3, y + 12), fill=(110, 110, 105))
    return img.filter(ImageFilter.GaussianBlur(0.6))


def mossy_ground(img: Image.Image, seed=3) -> Image.Image:
    """土と落ち葉の地面に、苔と草のまだら（数 m の大きさ）を混ぜる。ストリートビューの土手は茶色一色ではない。"""
    rng = np.random.default_rng(seed)
    a = np.asarray(img.convert("RGB"), float)
    h, w = a.shape[:2]
    green = a.mean(-1, keepdims=True) * np.array([0.78, 1.05, 0.55]) + np.array([5, 18, 0])
    m = _noise((h, w), (h / 10, w / 10), rng) + 0.35 * _noise((h, w), (h / 60, w / 60), rng)
    mix = np.clip((m - 0.1) * 1.5, 0, 0.85)[..., None]
    return Image.fromarray(np.clip(a * (1 - mix) + green * mix, 0, 255).astype(np.uint8))


# ---------- 落ち葉 ----------

EDGE_W = 1.3          # 路肩の帯の幅（道の縁から内側へ）
LINE_IN, LINE_OUT = 0.17, 0.33   # 外側線の位置（道の縁から内へ m。LINES_s1_edge を測った値）
LEAF_LIFT = 0.012     # 路面から浮かせる高さ
TILE_M = 8.0          # 画像の縦 1 枚＝8 m


def _noise(shape, sigma, rng):
    from scipy.ndimage import gaussian_filter
    n = gaussian_filter(rng.random(shape), sigma, mode="wrap")
    return (n - n.mean()) / (n.std() + 1e-9)


def leaf_textures(leaves: Image.Image, seed=5):
    """落ち葉の画像（すき間なく葉）から、透ける 2 枚を作る。
    edge：横＝道の縁から内へ EDGE_W（u=0 が外側）、縦＝8 m。外ほど濃く、ふちはぎざぎざ。
    lane：横＝車線の幅（u=0〜1）、縦＝8 m。まばらな小さい葉のかたまり（タイヤの通る所は少なめ）。"""
    rng = np.random.default_rng(seed)
    from PIL import ImageEnhance
    src = ImageEnhance.Color(ImageEnhance.Brightness(leaves.convert("RGB")).enhance(0.62)).enhance(0.7)   # 写真の落ち葉は暗い茶色
    ppm = src.width / 2.0                                       # 元画像は 2 m 四方
    def tiled(wm, hm, w, h):
        W, H = int(wm * ppm), int(hm * ppm)
        canvas = Image.new("RGB", (W, H))
        for y in range(0, H, src.height):
            for x in range(0, W, src.width):
                canvas.paste(src, (x, y))
        return canvas.resize((w, h), Image.LANCZOS)
    # 路肩の帯
    w, h = 128, 1024
    col = tiled(EDGE_W, TILE_M, w, h)
    um = np.linspace(0, EDGE_W, w)[None, :]                     # 道の縁から内へ m
    # 外側線（縁から 0.175〜0.325 m）の外にたまり、線の上と内側はまばら（ストリートビューでは線が見える）
    density = np.where(um < LINE_IN, 1.0, np.where(um < LINE_OUT + 0.03, -0.2, 0.30 - 0.25 * (um - LINE_OUT) / (EDGE_W - LINE_OUT)))
    a = density + 0.30 * _noise((h, w), (10, 4), rng) + 0.15 * _noise((h, w), (2, 1.5), rng)
    along = 0.25 * _noise((h, 1), (60, 0), rng)                 # 長さ方向の濃い所・薄い所
    edge = col.convert("RGBA"); edge.putalpha(Image.fromarray(((a + along) > 0.62).astype(np.uint8) * 255))
    # 車線の上
    w2, h2 = 512, 1024
    col2 = tiled(3.0, TILE_M, w2, h2)
    u2 = np.linspace(0, 1, w2)[None, :]
    tracks = 1 - 0.6 * (np.exp(-((u2 - 0.25) / 0.07) ** 2) + np.exp(-((u2 - 0.75) / 0.07) ** 2))   # 左右のタイヤの通り道
    a2 = _noise((h2, w2), (3, 3), rng) * tracks + 0.4 * _noise((h2, w2), (25, 25), rng)
    lane = col2.convert("RGBA"); lane.putalpha(Image.fromarray((a2 > 2.1).astype(np.uint8) * 255))
    return edge, lane


def leaf_meshes(sec, s_from, s_to, lane=True):
    """路肩の左右の帯と、車線の上の 1 枚。[(verts, faces, uvs, part)]。part は 'edge' か 'lane'。"""
    i0 = int(np.searchsorted(sec.s, s_from)); i1 = int(np.searchsorted(sec.s, s_to, side="right"))
    out = []
    for a in range(i0, i1 - 1, CHUNK_ROWS):
        rows = np.arange(a, min(a + CHUNK_ROWS + 1, i1))
        if len(rows) < 2:
            continue
        x, y, h, s = sec.x[rows], sec.y[rows], sec.heading[rows], sec.s[rows]
        z, sl, cr = sec.z[rows], sec.slope[rows], sec.crown[rows]
        for side in (1, -1):
            hw = (sec.hw_left if side > 0 else sec.hw_right)[rows]
            lat = side * np.c_[hw, hw - EDGE_W * 0.5, hw - EDGE_W]
            Z = surface_height(lat, z[:, None], sl[:, None], cr[:, None]) + LEAF_LIFT
            v, f, uv = ribbon(x, y, h, lat, Z, s, u_scale=EDGE_W, v_scale=TILE_M, flip=side > 0)   # 上向きの面に
            out.append((v, f, uv, "edge"))
        if lane:
            hl, hr = sec.hw_left[rows], sec.hw_right[rows]
            t = np.linspace(0, 1, 7)[None, :]
            lat = hl[:, None] * (1 - t) - hr[:, None] * t          # 左の縁 → 右の縁
            Z = surface_height(lat, z[:, None], sl[:, None], cr[:, None]) + LEAF_LIFT * 0.8
            v, f, uv = ribbon(x, y, h, lat, Z, s, u_scale=1.0, v_scale=TILE_M, flip=True)
            u = uv[:, 0].reshape(len(rows), 7)
            uv = np.c_[(u / np.maximum(u[:, -1:], 1e-6)).ravel(), uv[:, 1]]   # 横は縁から縁で 0〜1
            out.append((v, f, uv, "lane"))
    return out
