"""試作の地面の塗り分けの絵：空中写真の色から芝（緑）・土（茶）・舗装（灰）。R＝芝、G＝土。使い方: ground_mask.py <フォルダ>"""
import sys
import numpy as np
from PIL import Image, ImageFilter
D = sys.argv[1]
ph = np.asarray(Image.open(f"{D}/photo.png").convert("RGB"), float) / 255
r, g, b = ph[..., 0], ph[..., 1], ph[..., 2]; lum = ph.mean(-1)
# 植物は緑の強さ 2G−R−B（暗い林・庭木の影も拾う。前の「G が R・B より 2〜3% 大きい」では暗い緑が落ちた）
grass = (2 * g - r - b) > 0.015
# 土は赤すぎない茶色（赤い屋根を土にしていた）
rg = r - g
soil = (rg > 0.02) & (rg < 0.12) & (r > b * 1.1) & (lum < 0.55) & ~grass
mk = np.stack([grass, soil, np.zeros_like(grass)], -1).astype(np.uint8) * 255
Image.fromarray(mk).filter(ImageFilter.MedianFilter(5)).filter(ImageFilter.GaussianBlur(3)).save(f"{D}/ground_mask.png")
print(grass.mean(), soil.mean())

# 庭木の位置：濃い緑（木の葉の塊、明るさ 0.36 未満）を 3 m おきに拾う。trees.json（中心からの x, y と大きさ）。家・道の上は street.py が外す
import json
S = json.load(open(f"{D}/scene.json")); R = S["R"]; px = 2 * R / ph.shape[0]
tree = grass & (lum < 0.36)
step = int(3.0 / px); pts = []
rng = np.random.default_rng(1)
for j in range(step // 2, ph.shape[0], step):
    for i in range(step // 2, ph.shape[1], step):
        win = tree[max(0, j - step // 2):j + step // 2, max(0, i - step // 2):i + step // 2]
        if win.mean() > 0.6:
            x = -R + (i + rng.uniform(-0.3, 0.3) * step) * px; y = R - (j + rng.uniform(-0.3, 0.3) * step) * px
            pts.append([round(float(x), 2), round(float(y), 2), round(float(0.6 + 0.6 * win.mean()), 2)])
json.dump(pts, open(f"{D}/trees.json", "w"))
print("trees", len(pts))
