"""神社の島まわり（S3 3410 付近）を上から：空中写真｜今のコース を並べてビューアーで開く。
使い方: uv run python -m scripts.split_top [s] [size_m]"""
import json
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from yabitsu import config, forest_ground, geo, gsi, photo, track
from scripts import dress_compare as dc
from scripts import forest_build as fb


def upscaled_aerial(c, size, px, layer="seamlessphoto"):
    """地理院の空中写真は z=18（約 0.5 m/画素）が最大。元の細かさで取り、なめらかに拡大（Lanczos）してからくっきりさせる。"""
    from PIL import ImageFilter
    native = 0.5
    a = Image.fromarray(photo.render_local_image(gsi.TileCache(), geo.from_local, c[0] - size / 2, c[1] - size / 2, size,
                                                 native, layer=layer, z=18))
    a = a.resize((px, px), Image.LANCZOS)
    return a.filter(ImageFilter.UnsharpMask(radius=2.0, percent=120, threshold=2))


def main():
    s = float(sys.argv[1]) if len(sys.argv) > 1 else 3430.0
    section = sys.argv[3] if len(sys.argv) > 3 else "s3"
    size = float(sys.argv[2]) if len(sys.argv) > 2 else 240.0
    forest = sorted((config.PROJECT_ROOT / "export" / "forest").glob("2026*"))[-1]
    cl = np.load(config.DATA_PROCESSED / "track_centerline.npz"); b = track.load_bounds()
    c, _ = fb._at(section, s, cl, b)
    z = float(forest_ground.load_ground(fb.MANIFEST, prefixes=("1ROAD_",)).z(c[None])[0])
    if not np.isfinite(z):
        z = float(forest_ground.load_ground(fb.MANIFEST).z(c[None])[0])
    v = [{"name": "top_cmp", "type": "top", "center": c.tolist(), "size": size, "z": z,
          "window": [c[0] - size * 0.7, c[1] - size * 0.7, c[0] + size * 0.7, c[1] + size * 0.7], "lod": 2}]
    (forest / "views.json").write_text(json.dumps(v), encoding="utf-8")
    subprocess.run([dc.BLENDER, "-b", "--factory-startup", "--python", str(config.PROJECT_ROOT / "blender_scripts" / "render_forest.py"),
                    "--", str(forest), "s3"], env={**os.environ, "DRESS_DIR": str(dc.DRESS)}, capture_output=True, timeout=1800, check=True)
    ren = Image.open(forest / "views" / "top_cmp.png").convert("RGB").resize((1000, 1000))
    aer = upscaled_aerial(c, size, 1000)
    f = ImageFont.truetype(config.font_file("gothic"), 30)
    out = Image.new("RGB", (2000, 1050), (255, 255, 255)); out.paste(aer, (0, 50)); out.paste(ren, (1000, 50))
    d = ImageDraw.Draw(out)
    d.text((10, 8), f"空中写真（地理院） s={s:.0f} 付近 {size:.0f} m 四方・上が北", fill=(0, 0, 0), font=f)
    d.text((1010, 8), "今のコース（上から）", fill=(0, 0, 0), font=f)
    p = config.COMPARE_DIR / "s3_split_top.jpg"; p.parent.mkdir(parents=True, exist_ok=True)
    out.save(p, quality=90)
    # 開くのは呼び出した側（ユーザーに見せる時だけ、説明と一緒に）
    print(p)


if __name__ == "__main__":
    main()
