"""ストリートビューと同じ位置・向きで S3 を描き、写真｜描画 を並べる（ビューアーで開く）。
使い方: uv run python -m scripts.dress_compare [s=f ...]   例: 500f 1050b"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from yabitsu import config, forest_ground, geo

SV = config.PROJECT_ROOT / ".superpowers" / "sdd" / "2026-10-17-yabitsu-s3"
FOREST = sorted((config.PROJECT_ROOT / "export" / "forest").glob("2026*"))[-1]     # いちばん新しい森（木の配置）
DRESS = config.DATA_PROCESSED / "dress_s3"
BLENDER = r"C:/Program Files/Blender Foundation/Blender 5.2/blender.exe"
DEFAULT = ["500f", "1050b", "1450f", "2150f", "2350b", "3950f"]
EYE = 2.5                     # ストリートビューのカメラの高さ


def main():
    picks = sys.argv[1:] or DEFAULT
    if picks == ["all"]:      # 写真のある全地点（上り・下り）
        picks = [f"{p['s']}{d}" for p in json.loads((SV / "sv_s3_points.json").read_text(encoding="utf-8")) for d in "fb"
                 if (SV / "sv" / f"s3_{p['s']:04d}_{d}.png").exists()]
    pts = {p["s"]: p for p in json.loads((SV / "sv_s3_points.json").read_text(encoding="utf-8"))}
    road = forest_ground.load_ground(config.DATA_PROCESSED / "track", prefixes=("1ROAD_",))
    views = []
    for pk in picks:
        s, fb = int(pk[:-1]), pk[-1]
        p = pts[s]
        x, y = geo.to_local(p["lat"], p["lon"])
        c = np.array([float(x), float(y)])
        brg = np.radians(p["brg"] + (180 if fb == "b" else 0))
        d = np.array([np.sin(brg), np.cos(brg)])            # 方位（北から時計回り）→ x 東・y 北
        z = float(np.nan_to_num(road.z(c[None])[0], nan=0.0))
        views.append({"name": f"sv_{s:04d}_{fb}", "type": "persp", "eye": [*c, z + EYE], "target": [*(c + d * 50), z + EYE - 1.0],
                      "lens": 18, "lod": 0, "res": [900, 600], "window": [c[0] - 250, c[1] - 250, c[0] + 250, c[1] + 250]})
    (FOREST / "views.json").write_text(json.dumps(views), encoding="utf-8")
    env = {**os.environ, "DRESS_DIR": str(DRESS)}
    r = subprocess.run([BLENDER, "-b", "--factory-startup", "--python", str(config.PROJECT_ROOT / "blender_scripts" / "render_forest.py"),
                        "--", str(FOREST), "s3"], env=env, capture_output=True, timeout=3000)
    if r.returncode:
        print(r.stderr.decode("utf-8", "replace")[-2000:]); return
    font = ImageFont.truetype("C:/Windows/Fonts/meiryo.ttc", 26)
    rows = []
    for v in views:
        sv = Image.open(SV / "sv" / f"s3_{v['name'][3:]}.png").convert("RGB").resize((900, 600))
        rd = Image.open(FOREST / "views" / f"{v['name']}.png").convert("RGB").resize((900, 600))
        row = Image.new("RGB", (1800, 600)); row.paste(sv, (0, 0)); row.paste(rd, (900, 0))
        dr = ImageDraw.Draw(row)
        label = f"s={int(v['name'][3:7])} {'上り' if v['name'].endswith('f') else '下り'}"
        for x0, t in ((0, "ストリートビュー"), (900, "今回の仕上げ")):
            dr.rectangle((x0, 0, x0 + 360, 40), fill=(0, 0, 0)); dr.text((x0 + 8, 4), f"{label}  {t}", fill=(255, 255, 0), font=font)
        rows.append(row)
    out_dir = config.PROJECT_ROOT / "export" / "dress" / "s3" / "compare"
    out_dir.mkdir(parents=True, exist_ok=True)
    per = 6
    outs = []
    for k in range(0, len(rows), per):
        chunk = [r.resize((1200, 400)) for r in rows[k:k + per]]
        sheet = Image.new("RGB", (1200, 400 * len(chunk)))
        for j, row in enumerate(chunk):
            sheet.paste(row, (0, 400 * j))
        out = out_dir / f"compare_{k // per:02d}.jpg"
        sheet.save(out, quality=88)
        outs.append(out)
    print("\n".join(map(str, outs)))
    if len(outs) == 1:
        pass   # 開くのは呼び出した側（ユーザーに見せる時だけ、説明と一緒に）


if __name__ == "__main__":
    main()
