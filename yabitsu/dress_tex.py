"""S3 の道路まわりのテクスチャ：Codex に生成を頼む指示（TEXTURES）と、届いた画像の後処理。

元画像は assets/dress/src/<name>.png（Codex の image_generation で生成）。後処理した結果は
assets/dress/<name>.png（色）と <name>_nm.png（凹凸＝法線マップ、明るさを高さとみなして作る）。
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .config import PROJECT_ROOT

DIR = PROJECT_ROOT / "assets" / "dress"
SRC = DIR / "src"

_COMMON = ("真上から見た写真風のゲーム用テクスチャ。影・物・文字・ロゴは入れない。均一な昼の光。"
           "上下左右が継ぎ目なく繋がる（シームレス）。できるだけ高解像度。")


@dataclass
class TexSpec:
    size_m: tuple          # 実寸（横, 縦）m
    prompt: str
    normal: float = 2.0    # 凹凸の強さ
    tile: bool = True      # 継ぎ目をなめらかにするか
    circle: bool = False   # 丸く切り抜く（マンホール）


TEXTURES = {
    "asphalt_town": TexSpec((4, 2), "日本の郊外の県道の、使い込まれた明るい灰色のアスファルト舗装。細かい骨材の粒、薄いタイヤの跡、色ムラ、ごく小さなひび。白線やマンホールは入れない。横長 2:1。"),
    "asphalt_dark": TexSpec((4, 2), "日本の山すその県道の、やや新しい黒っぽいアスファルト舗装。細かい骨材、うっすらしたタイヤの跡が縦方向に走る、色ムラは少なめ。白線は入れない。横長 2:1。"),
    "paint_white": TexSpec((1, 1), "アスファルトの上に塗られた白い道路標示の塗料を、ごく近くから見たもの。全面が白い塗料で、ところどころかすれて下の灰色のアスファルトの粒が少し透けて見える。正方形。", normal=1.0),
    "paint_orange": TexSpec((1, 1), "アスファルトの上に塗られた日本の道路のオレンジ色（黄色）の中央線の塗料を、ごく近くから見たもの。全面がオレンジ色の塗料で、ところどころかすれて灰色のアスファルトが少し透ける。正方形。", normal=1.0),
    "gutter": TexSpec((1, 2), "日本の道路わきのコンクリートの側溝のふち（L 字の路肩）。くすんだ灰色のコンクリート、雨のしみ、土の汚れ、細いひび、長さ方向に 2 m ごとの目地。縦長 1:2。"),
    "sidewalk": TexSpec((2, 2), "日本の郊外の歩道のアスファルト舗装。車道より少し明るく細かい、色あせた灰色、落ち葉のしみ少し。正方形。"),
    "kenchi": TexSpec((2, 2), "日本の道路わきの擁壁の間知石（けんちいし）の石積みを正面から。暗い灰色の四角っぽい石が斜めの目地で網目状に積まれ、石の表面はざらざら、目地は黒く、ところどころ苔と雨の黒いしみ。正方形。", normal=4.0),
    "block_wall": TexSpec((2, 1), "日本の家のコンクリートブロック塀を正面から。灰色のブロック（横 40 cm×縦 20 cm）が目地で並ぶ、雨だれの黒いしみ、うすい苔。横長 2:1。", normal=3.0),
    "concrete_wall": TexSpec((2, 2), "日本の道路わきのコンクリート擁壁を正面から。打ちっぱなしの灰色、型枠の横線、雨だれの黒い縦のしみ、ところどころ苔、小さな水抜き穴（丸い黒い穴）が1つ。正方形。", normal=2.5),
    "curb": TexSpec((1, 1), "日本の歩道の縁石（コンクリート）を上から。くすんだ明るい灰色、角は少し欠け、汚れ。正方形。"),
    "grating": TexSpec((1, 1), "日本の道路わきの側溝の鉄のふた（グレーチング）を真上から。細い鉄の板が縦に細かく並び、すき間は暗い。少しさびた濃い灰色。正方形。", normal=3.0, tile=False),
    "manhole": TexSpec((1, 1), "日本の道路のマンホールのふたを真上から。丸い鋳鉄のふたで、すべり止めの細かい幾何学模様、少し錆びて黒ずんだ灰色。ふたが正方形の画像いっぱいに丸く入る（ふたの外は灰色のアスファルト）。文字やロゴは入れない。", normal=3.0, tile=False, circle=True),
    "pole": TexSpec((1, 2), "日本の電柱（コンクリート柱）の表面を正面から。明るい灰色のコンクリート、細い縦すじ、雨のしみ。縦長 1:2。", normal=1.0),
    # 山道（S2〜S5）。ストリートビューの山道は、町より明るくあせた灰色で、ひびと補修跡が多い
    "asphalt_mountain": TexSpec((4, 2), "日本の山奥の古い県道のアスファルト舗装。日に焼けて明るめの灰色にあせ、骨材の粒が浮き出て、細い網目状のひび割れ、黒い補修のつぎはぎが少し、土と落ち葉の薄いしみ。白線やマンホールは入れない。横長 2:1。"),
    "gutter_mountain": TexSpec((1, 2), "日本の山道の路肩の、古いコンクリートの L 字の側溝。黒ずんだ灰色、苔の緑のしみ、溜まった土と茶色い落ち葉、細いひび、長さ方向に 2 m ごとの目地。縦長 1:2。"),
    "kenchi_moss": TexSpec((2, 2), "日本の山道の苔むした古い間知石（けんちいし）の石積み擁壁を正面から。暗い灰色の四角っぽい石が斜めの目地で網目状に積まれ、石の表面と目地に緑の苔がびっしり、雨の黒いしみ。正方形。", normal=4.0),
    "kenchi_block": TexSpec((2, 2), "日本の山道のコンクリートの間知ブロック擁壁を正面から。同じ大きさ（約 35 cm）の四角いコンクリートブロックが、斜め 45 度の目地で規則正しい菱形の網目に積まれている。くすんだ灰色、雨だれの黒いしみ、目地にうすい苔。正方形。", normal=3.5),
    "rock_cut": TexSpec((2, 2), "日本の山道わきの岩の切土の斜面を正面から。灰色から茶色の割れた岩肌、ぎざぎざの割れ目、ところどころ苔と小さな草、湿った黒いしみ。正方形。", normal=5.0),
    "rock_net": TexSpec((2, 2), "日本の山道わきの落石防護網の斜面を正面から。土と岩と枯れ草の斜面の上に、錆びた細いワイヤーのひし形の金網（目は約 5 cm）が全面にかかり、太いワイヤーロープが 1 m ごとの格子に走る。正方形。", normal=3.0),
    "norikae": TexSpec((4, 4), "日本の山道わきの斜面のコンクリート法枠（のりわく）を正面から。幅 30 cm の灰色のコンクリートの梁が 2 m 間隔の正方形の格子になり、格子の中は灰色のモルタル吹付けと少しの草。雨のしみ。正方形。", normal=4.0),
    "leaves_road": TexSpec((2, 2), "日本の山道のアスファルトの上に散らばって積もった落ち葉を真上から。茶色・こげ茶・黄土色の広葉樹の枯れ葉と杉の茶色い枯れ葉、小枝が、すき間なく画像全体にびっしり重なっている（アスファルトはほとんど見えない）。正方形。", normal=2.0),
    "forest_floor": TexSpec((2, 2), "日本の杉と広葉樹の林の地面を真上から。こげ茶の湿った土、茶色い落ち葉と杉の枯れ葉、小枝、まばらな小さい草とシダ。緑は少なめ。正方形。", normal=3.0),
    "bank_soil": TexSpec((2, 2), "日本の山道わきの土手の斜面を正面から。茶色の土がむき出しで、細い木の根が露出し、小石、ところどころ苔と枯れ草と落ち葉。正方形。", normal=3.5),
    "shotcrete": TexSpec((2, 2), "日本の山道わきの、モルタル吹付けの斜面を正面から。灰色のごつごつした吹付けモルタル、黒い雨のしみと苔、小さな丸い水抜きパイプの穴が 1 つ。正方形。", normal=3.0),
}


def height_to_normal(h: np.ndarray, strength: float) -> np.ndarray:
    """高さ（0〜1）→ 接空間の法線（-1〜1）。"""
    gy, gx = np.gradient(np.asarray(h, float))
    n = np.dstack([-gx * strength, -gy * strength, np.ones_like(gx)])
    return n / np.linalg.norm(n, axis=-1, keepdims=True)


def make_tileable(img: Image.Image) -> Image.Image:
    """半分ずらした画像と混ぜ、上下左右の端を繋がるようにする（中央は元の画像、端はずらした画像）。"""
    a = np.asarray(img.convert("RGB"), float)
    h, w = a.shape[:2]
    rolled = np.roll(np.roll(a, h // 2, 0), w // 2, 1)
    wy = np.sin(np.linspace(0, np.pi, h))[:, None]
    wx = np.sin(np.linspace(0, np.pi, w))[None, :]
    m = np.clip((wy * wx) * 1.6, 0, 1)[..., None]
    return Image.fromarray(np.clip(a * m + rolled * (1 - m), 0, 255).astype(np.uint8))


def circle_alpha(img: Image.Image) -> Image.Image:
    img = img.convert("RGBA")
    mask = Image.new("L", img.size, 0)
    w, h = img.size
    ImageDraw.Draw(mask).ellipse((w * 0.03, h * 0.03, w * 0.97, h * 0.97), fill=255)
    img.putalpha(mask)
    return img


def _pow2(n):
    return 1 << int(np.floor(np.log2(max(64, n))))


def process(name: str) -> tuple:
    """src/<name>.png → <name>.png・<name>_nm.png。"""
    spec = TEXTURES[name]
    img = Image.open(SRC / f"{name}.png").convert("RGB")
    if spec.tile:
        img = make_tileable(img)
    w, h = img.size
    img = img.resize((min(2048, _pow2(w)), min(2048, _pow2(h))), Image.LANCZOS)
    lum = np.asarray(img.convert("L"), float) / 255.0
    n = height_to_normal(lum, spec.normal)
    nm = Image.fromarray(((n * 0.5 + 0.5) * 255).astype(np.uint8))
    if spec.circle:
        img = circle_alpha(img)
    img.save(DIR / f"{name}.png"); nm.save(DIR / f"{name}_nm.png")
    return img, nm


def load(name: str) -> tuple:
    return Image.open(DIR / f"{name}.png"), Image.open(DIR / f"{name}_nm.png")
