"""山道（表示の S2〜S5＝内部の s2・s1・s4・s5）の道路まわりの仕上げの部品。

- 壁の仕上げ：表（*_features.csv）の note から、石積み・苔むした石積み・間知ブロック・コンクリート・岩・落石防護網・法枠・吹付けを決める
- ガードレール：波形の断面（W 形の鋼板）、丸い支柱（4 m おき）、端は外へ曲げる。位置は今のガードレール（features.guardrail_meshes）と同じ線
- 落ち葉：路肩の帯（外ほど濃い）と、車線の上のまばらな葉。透ける画像（アルファテスト）を路面から少し浮かせて貼る
座標は manifest と同じ（x 東, y 北, z 上）。
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .crosssection import GUTTER_WIDTH, surface_height
from .features import CURB_HEIGHT, GUARDRAIL_OFFSET, SIDES, rows_of
from .meshgen import ribbon

SECTIONS = ("s2", "s1", "s4", "s5")      # 内部の区間名（表示の S2・S3・S4・S5）


def town_walls(names, taken=()) -> dict:
    """町の物の擁壁（town_objects の 1WALL_<prefix>_obj_retaining）→ 間知ブロック。宮ヶ瀬北原の擁壁はストリートビューで間知ブロック
    （s5_kitahara.json の note）。taken（S3 の仕上げが貼り替える・隠すもの）は除く。"""
    # ヤビツ峠のわき道の野面石積み（地面の高さに立てる物は objg）
    stone = {"1WALL_s1_obj_retaining": "mt_wall_nozura", "1WALL_s1_objg_retaining": "mt_wall_nozura"}
    # 材質 wall_grey の擁壁（town_objects の material、例：峠の駐車場の向かいの道から離れた所）も間知ブロック
    return {n: stone.get(n, "mt_wall_block") for n in names
            if n.startswith("1WALL_") and n.endswith(("_obj_retaining", "_obj_retaining_wall_grey", "_objg_retaining", "_objg_retaining_wall_grey"))
            and n not in taken}


def chainlink_texture():
    """2 m×1.5 mの銀色の金網。約6cmの網目と上下の横桟。"""
    w, h = 512, 384
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    for k in range(-w, h + w, 16):
        d.line([(0, k), (w, k + w)], fill=(155, 161, 151, 255), width=1)
        d.line([(0, k), (w, k - w)], fill=(155, 161, 151, 255), width=1)
    d.rectangle((0, 0, w, 5), fill=(178, 182, 171, 255))
    d.rectangle((0, h - 6, w, h), fill=(178, 182, 171, 255))
    return img


def chainlink_parts(v, f, uv):
    """既存の擁壁上フェンスの面を利用し、銀色の支柱を加える。元の壁の形と高さは保つ。"""
    from .town_objects import _merge
    posts, last = [], None
    for q in np.asarray(v).reshape(-1, 4, 3):
        p = q[0]
        if last is None or np.linalg.norm(p[:2] - last[:2]) >= 1.7:
            posts.append(cylinder(p, 0.032, q[3, 2] - p[2], sides=8))
            last = p
    q = np.asarray(v).reshape(-1, 4, 3)[-1]
    if np.linalg.norm(q[1, :2] - last[:2]) > 0.1:
        posts.append(cylinder(q[1], 0.032, q[2, 2] - q[1, 2], sides=8))
    return (v, f[::2], uv), _merge(posts)


def pole_cover_texture():
    """黄黒の斜め縞。円周1m×高さ1.6m、縞の間隔約20cm。"""
    y, x = np.mgrid[:512, :320]
    stripe = ((x + y) // 64) % 2
    colors = np.array([(40, 40, 34), (210, 178, 42)], np.uint8)
    return Image.fromarray(colors[stripe])


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
    if "ブロック" in n and "古い" in n:
        return "block_aged"    # 目地が深く苔の付いた古い間知ブロック（Codex の r70_block は白い平らな板に見えた。2026-11-01）
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
    lat = hw + GUTTER_WIDTH + GUARDRAIL_OFFSET + f.width_m   # 行の width_m：路肩の外の縁へ出す幅（features.guardrail_meshes と同じ）
    s = sec.s[rows]
    d = np.minimum(s - s[0], s[-1] - s)                     # 端からの距離
    t = np.clip(1 - d / FLARE_M, 0, 1)
    lat = lat + FLARE_OUT * t * t                            # 端は外へ曲げる
    return rows, side, lat, sec.edge_z(side)[rows] + (CURB_HEIGHT if f.width_m > 0 else 0.0)


R70_SECTIONS = ("s4", "s5")   # Codex の部品（data/processed/landmarks/r70_*）で作る区間（ヤビツ峠→宮ヶ瀬北原、2026-10-31）
R70_RAIL_SPAN = 4.0           # r70_guardrail の w_beam：始端の支柱から +X へ 4 m


def bend_on_rail(v, sec, rows, side, lat, ze, s_at):
    """部品の頂点（+X＝つなぐ向き、−Y＝道の側、+Z＝上）を、ガードレールの基準線に沿って曲げて置く。
    s_at(x)：部品の x を中心線の s へ。横は基準線（lat：中心からの距離）から外へ y、高さは基準線の路面の縁（ze）から z。"""
    S = sec.s[rows]
    s = s_at(v[:, 0])
    k = np.clip(np.searchsorted(S, s) - 1, 0, len(rows) - 2)
    t = (s - S[k]) / np.maximum(S[k + 1] - S[k], 1e-9)   # 両端の外（袖）は、端の 2 行をそのまま延ばす
    lerp = lambda a: a[rows][k] * (1 - t) + a[rows][k + 1] * t
    cx, cy = lerp(sec.x), lerp(sec.y)
    h = np.unwrap(sec.heading[rows])
    hh = h[k] * (1 - t) + h[k + 1] * t
    la = lat[k] * (1 - t) + lat[k + 1] * t
    z = ze[k] * (1 - t) + ze[k + 1] * t
    off = side * (la + v[:, 1])                          # 外へ y（道の側は −Y）
    return np.c_[cx - np.sin(hh) * off, cy + np.cos(hh) * off, z + v[:, 2]]


def r70_rail_layout(length, span=R70_RAIL_SPAN):
    """基準線の長さ → 区切りの数と、1 区切りの実際の長さ（4 m 前後にそろえ、端数は全体に配る）。"""
    n = max(1, int(round(length / span)))
    return n, length / n


def r70_guardrail_meshes(sec, f, parts):
    """Codex の W 形ガードレール（parts＝{"span": (v, f, uv), "end": (v, f, uv)}）を、基準線に沿って並べる。[(verts, faces, uvs, part)]。
    part は 'span'（4 m 前後の区切りを、まとめて 1 つ）と 'end'（両端の袖）。進む向きの左の柵は +s へ、右の柵は −s へつなぐ（−Y が道の側になる）。"""
    rows, side, lat, ze = guardrail_line(sec, f)
    s0, s1 = float(sec.s[rows[0]]), float(sec.s[rows[-1]])
    n, L = r70_rail_layout(s1 - s0)
    sv, sf, su = parts["span"]
    kx = L / R70_RAIL_SPAN
    out, V, F, U = [], [], [], []
    for j in range(n):
        base = s0 + j * L if side > 0 else s1 - j * L
        s_at = (lambda x, b=base: b + x * kx) if side > 0 else (lambda x, b=base: b - x * kx)
        V.append(bend_on_rail(sv, sec, rows, side, lat, ze, s_at)); F.append(sf + sum(len(a) for a in V[:-1])); U.append(su)
        if sum(len(a) for a in V) > 60000:
            out.append((np.concatenate(V), np.concatenate(F), np.concatenate(U), "span")); V, F, U = [], [], []
    if V:
        out.append((np.concatenate(V), np.concatenate(F), np.concatenate(U), "span"))
    ev, ef, eu = parts["end"]
    for at_start in (True, False):          # 終わりの端は +X へ伸ばし、始まりの端は X を裏返して外へ伸ばす（面の向きも裏返す）
        flip = at_start
        v = ev * [(-1 if flip else 1), 1, 1]
        fc = ef[:, ::-1] if flip else ef
        b = (s0 if side > 0 else s1) if at_start else (s1 if side > 0 else s0)
        s_at = (lambda x, b=b: b + x) if side > 0 else (lambda x, b=b: b - x)
        out.append((bend_on_rail(v, sec, rows, side, lat, ze, s_at), fc, eu, "end"))
    return out


def rows_between(sec, s0, s1, pad=2.0):
    """中心線の s0〜s1 を覆う行（両端に pad m の余り）。"""
    i0 = max(0, int(np.searchsorted(sec.s, s0 - pad)) - 1)
    i1 = min(sec.n, int(np.searchsorted(sec.s, s1 + pad)) + 1)
    return np.arange(i0, i1)


def r70_run_mesh(sec, s0, s1, side, lat, part, span, centered=True):
    """Codex の欄干の区切り（part、+X へ span m ごと。centered＝原点が区切りの中央）を、中心線の s0〜s1、
    中心から lat m の線に沿って並べる（side＝+1 左／−1 右、−Y が道の側）。高さは道の縁の路面から。(verts, faces, uvs)"""
    rows = rows_between(sec, s0, s1)
    latv = np.full(len(rows), float(lat))
    ze = sec.edge_z(side)[rows]
    pv, pf, pu = part
    n, L = r70_rail_layout(s1 - s0, span)
    kx, sh = L / span, (span / 2 if centered else 0.0)
    V, F, U = [], [], []
    for j in range(n):
        base = s0 + j * L if side > 0 else s1 - j * L
        s_at = (lambda x, b=base: b + (x + sh) * kx) if side > 0 else (lambda x, b=base: b - (x + sh) * kx)
        V.append(bend_on_rail(pv, sec, rows, side, latv, ze, s_at)); F.append(pf + sum(len(a) for a in V[:-1])); U.append(pu)
    return np.concatenate(V), np.concatenate(F), np.concatenate(U)


def r70_post_mesh(sec, s, side, lat, part):
    """1 本の部品（親柱・橋名板）を、中心線の s・中心から lat m の所に立てる（−Y が道の側）。"""
    rows = rows_between(sec, s - 1.0, s + 1.0)
    s_at = (lambda x: s + x) if side > 0 else (lambda x: s - x)
    v = bend_on_rail(part[0], sec, rows, side, np.full(len(rows), float(lat)), sec.edge_z(side)[rows], s_at)
    return v, part[1], part[2]


def r70_pole_mesh(poles, part, sink=0.3):
    """Codex のコンクリート電柱（r70_utility_pole、腕金は +X）を電柱の位置に立てる。腕金は電線（隣の柱を結ぶ向き）と直角。
    根元は道の縁の高さから sink m 埋める（斜面で浮かないように。今の柱も 0.5 m 埋めている）。"""
    pv, pf, pu = part
    P = np.array([p[:2] for p in poles], float)
    V, F, U = [], [], []
    for i, (x, y, z) in enumerate(poles):
        a, b = P[max(i - 1, 0)], P[min(i + 1, len(P) - 1)]
        t = b - a
        t = t / np.linalg.norm(t) if np.linalg.norm(t) > 1e-6 else np.array([1.0, 0.0])
        ux = np.array([-t[1], t[0]]); uy = np.array([-ux[1], ux[0]])
        V.append(np.c_[x + pv[:, 0] * ux[0] + pv[:, 1] * uy[0], y + pv[:, 0] * ux[1] + pv[:, 1] * uy[1], z - sink + pv[:, 2]])
        F.append(pf + sum(len(q) for q in V[:-1])); U.append(pu)
    return np.concatenate(V), np.concatenate(F), np.concatenate(U)


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


def r70_whole_bridge_verts(sec, i, pts, deck_z=0.7):
    """Codex の橋の全体の部品の頂点 pts（+Y が道の進む向き、路面が Z＝deck_z）を、中心線の行 i の所に置いた世界の座標。
    高さは部品の Y（道に沿った長さ）の所の路面に合わせる（水平のまま置くと、赤いアーチ橋は 100 m で路面が 2.8 m 上がるので、峠側の端で床の
    横桁が路面から 0.9 m 出て、宮ヶ瀬側の端で床が 1.9 m 沈んだ。2026-11-01 にゲームで見つかった）。"""
    fwd = np.array([np.cos(sec.heading[i]), np.sin(sec.heading[i])]); ux = np.array([fwd[1], -fwd[0]])
    zr = np.interp(sec.s[i] + pts[:, 1], sec.s, sec.z)
    return np.c_[sec.x[i] + pts[:, 0] * ux[0] + pts[:, 1] * fwd[0], sec.y[i] + pts[:, 0] * ux[1] + pts[:, 1] * fwd[1], zr - deck_z + pts[:, 2]]


def r70_cover_scale(part, pts_xz, margin=0.03, s_max=2.5):
    """Codex の部品（v, f, uv）を正面（x・z、底の真ん中を原点）から見た外形（口もふさいだ外まわり＝凸包）から、点 pts_xz（今の坑口のアーチの縁など）が
    はみ出さない、いちばん小さい倍率（1 以上、縦横同じ）。覆えなければ s_max。margin：外形の縁からの余裕（m）。
    縁は隠さず口の内側の裏打ちとして残す（Codex の口が今の縁の内側より広い坑口があり、縁を隠すと口の内側にすき間が開く）。"""
    from shapely import contains_xy
    from shapely.geometry import MultiPoint
    v = np.asarray(part[0], float)
    hull = MultiPoint(v[:, [0, 2]]).convex_hull
    p = np.asarray(pts_xz, float).copy()
    p[:, 1] = np.maximum(p[:, 1], 2 * margin)          # 路面の上の足元（z＝0）は外形の底の線の上なので、少し上で調べる
    for k in np.arange(1.0, s_max + 1e-9, 0.01):
        if np.all(contains_xy(hull.buffer(-margin / k), p[:, 0] / k, p[:, 1] / k)):
            return float(round(k, 2))
    return s_max


PLATE_PX = (6, 770)   # Codex の坑口の絵で名板の札の字が入る四角（px、u・v とも。v＝0 が上）


def _plate_faces(f, uv, size_px=1024):
    u = uv[:, 0] * size_px; w = uv[:, 1] * size_px
    inn = (u >= PLATE_PX[0]) & (u <= PLATE_PX[1]) & (w >= PLATE_PX[0]) & (w <= PLATE_PX[1])
    return np.flatnonzero(inn[f].all(1))


def r70_respace_plate(v, f, uv, size_m, pitch_m):
    """Codex の坑口の名板の札（絵の名板の四角に uv が入る面）を、1 枚ずつ size_m 角・中心を pitch_m おきに並べ直した頂点。
    札の中心の高さと、並び全体の真ん中は変えない。ほかの頂点はそのまま（吹風トンネル出口の写真 R70-0992_b では札が約 0.55 m 角・0.81 m おき、
    Codex の部品は 0.46 m・0.64 m おきで、札がほとんどくっついて見えた。2026-11-01）。"""
    v = np.array(v, float)
    fi = _plate_faces(f, uv)
    groups = []                                   # 頂点を共有する面のまとまり＝札 1 枚
    for t in fi:
        s = set(f[t].tolist())
        hit = [g for g in groups if g & s]
        for g in hit:
            groups.remove(g); s |= g
        groups.append(s)
    tiles = sorted((np.array(sorted(g)) for g in groups), key=lambda ix: v[ix, 0].mean())
    if not tiles:
        return v
    cx0 = np.mean([v[ix, 0].mean() for ix in tiles])
    for j, ix in enumerate(tiles):
        c = np.array([v[ix, 0].mean(), v[ix, 2].mean()])
        k = size_m / max(np.ptp(v[ix, 0]), 1e-9)
        cn = cx0 + (j - (len(tiles) - 1) / 2) * pitch_m
        v[ix, 0] = cn + (v[ix, 0] - c[0]) * k
        v[ix, 2] = c[1] + (v[ix, 2] - c[1]) * k
    return v


PLATE_TILE = np.array([0.65, 0.75, 0.91])   # 写真 R70-0992_b：札の地 (93,107,130)・字 (41,52,68)・札の下の壁 (143,143,143) の比
PLATE_INK = np.array([0.29, 0.36, 0.48])


def _brush_font(px):
    """筆文字に近い教科書体（macOS のダウンロード済みのフォント）。無ければヒラギノ明朝。"""
    from glob import glob
    from PIL import ImageFont
    for p in sorted(glob("/System/Library/AssetsV2/com_apple_MobileAsset_Font*/*/AssetData/Kyokasho.ttc")) + \
            ["/System/Library/PrivateFrameworks/FontServices.framework/Versions/A/Resources/Fonts/Subsets/Kyokasho.ttc",
             "/System/Library/Fonts/ヒラギノ明朝 ProN.ttc"]:
        try:
            return ImageFont.truetype(p, px, index=1 if p.endswith("Kyokasho.ttc") else 0)
        except OSError:
            continue
    return ImageFont.load_default()


def r70_plate_texture(img, text, cells, face_rgb):
    """坑口の絵 img の名板の札の四角 cells（[(x0, y0, x1, y1) px]、左の札から）を、地 face_rgb×PLATE_TILE・黒っぽい筆文字 face_rgb×PLATE_INK に描き直す。"""
    out = img.convert("RGB").copy(); d = ImageDraw.Draw(out)
    tile = tuple(int(round(c)) for c in np.asarray(face_rgb, float) * PLATE_TILE)
    ink = tuple(int(round(c)) for c in np.asarray(face_rgb, float) * PLATE_INK)
    for ch, (x0, y0, x1, y1) in zip(text, cells):
        d.rectangle((x0, y0, x1 - 1, y1 - 1), fill=tile)
        font = _brush_font(int((y1 - y0) * 0.78))
        bb = d.textbbox((0, 0), ch, font=font)
        d.text(((x0 + x1 - (bb[2] - bb[0])) / 2 - bb[0], (y0 + y1 - (bb[3] - bb[1])) / 2 - bb[1]), ch, fill=ink, font=font)
    return out


def r70_portal_setback(q, face_y, wall_out, gap=0.03):
    """坑口の部品の頂点 q（外が −Y、縁の真ん中が原点）を Y に沿ってずらし、正面（y＝face_y）を今の坑口の壁の正面（外へ wall_out m）の gap m 手前に置く。
    吹風トンネル出口では Codex の坑口が壁より約 2.7 m 手前に出て、写真の 1 枚の平らな壁と違って張り出して見えた（2026-11-01）。"""
    q = np.array(q, float)
    q[:, 1] += (-face_y) - (wall_out + gap)
    return q


def r70_wall_normal(wall_xy, out_hint):
    """坑口の壁の頂点（x, y）の並びの向き（いちばん長い向き）に直角な、外向きの単位ベクトル（out_hint と同じ側）。
    カーブしたトンネルでは、両端を結ぶ向きが坑口の壁・道の向きと 10° ほどずれ、Codex の坑口が斜めに置かれた（吹風トンネル出口、2026-11-01）。"""
    p = np.asarray(wall_xy, float); p = p - p.mean(0)
    lat = np.linalg.svd(p, full_matrices=False)[2][0]
    o = np.array([lat[1], -lat[0]])
    return o if o @ np.asarray(out_hint, float) >= 0 else -o
