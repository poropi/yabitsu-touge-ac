"""ストリートビューの写真と、同じ位置・向きの今のコースの描画を並べる（どの区間でも）。
使い方: uv run python -m scripts.sv_compare <jobs.json> <名前.jpg>   （config.COMPARE_DIR に書く。絶対パスならそこへ）
jobs.json: [{"name", "lat", "lon", "yaw"(度・北から時計回り), "photo"(写真のパス), "pitch"(任意・下向きが正), "fov"(任意),
            "window_m"(任意・コースと仕上げを読む範囲の半分 m。遠くを見る絵で広げる), "clip_end"(任意・見える距離 m),
            "eye_height"(任意・路面または地面からの目の高さ m)}]"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from scripts import dress_compare as dc
from yabitsu import config, forest_ground, geo

EYE = 2.5


def main():
    jobs = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    out = config.COMPARE_DIR / sys.argv[2]; out.parent.mkdir(parents=True, exist_ok=True)
    surf = forest_ground.load_ground(config.DATA_PROCESSED / "track", prefixes=("1ROAD_",))
    ground = forest_ground.load_ground(config.DATA_PROCESSED / "track")
    views = []
    for j in jobs:
        x, y = geo.to_local(j["lat"], j["lon"])
        c = np.array([float(x), float(y)])
        z = surf.z(c[None])[0]
        if not np.isfinite(z):
            z = ground.z(c[None])[0]
        eye_z = float(z) + j.get("eye_height", EYE)
        yaw = np.radians(j["yaw"]); pitch = np.radians(j.get("pitch", 0.0))
        d = np.array([np.sin(yaw), np.cos(yaw)])
        tgt = [*(c + d * 50 * np.cos(pitch)), eye_z - 50 * np.sin(pitch)]
        views.append({"name": j["name"], "type": "persp", "eye": [*c, eye_z], "target": tgt,
                      "lens": 18 / np.tan(np.radians(j.get("fov", 100) / 2)), "lod": 0, "res": [900, 600],
                      "window": [c[0] - j.get("window_m", 250), c[1] - j.get("window_m", 250), c[0] + j.get("window_m", 250), c[1] + j.get("window_m", 250)],
                      "tree_window": [c[0] - 250, c[1] - 250, c[0] + 250, c[1] + 250], "clip_end": j.get("clip_end", 20000)})
    (dc.FOREST / "views.json").write_text(json.dumps(views), encoding="utf-8")
    r = subprocess.run([dc.BLENDER, "-b", "--factory-startup", "--python-exit-code", "1", "--python",
                        str(config.PROJECT_ROOT / "blender_scripts" / "render_forest.py"), "--", str(dc.FOREST), "x"],
                       env={**os.environ, "DRESS_DIR": os.pathsep.join(str(d) for d in (dc.DRESS, config.DATA_PROCESSED / "dress_mountain", config.DATA_PROCESSED / "horizon", config.DATA_PROCESSED / "parking_props", config.DATA_PROCESSED / "visual_updates", config.DATA_PROCESSED / "environment") if d.exists())},
                       capture_output=True, timeout=3000)   # S3 と山道（S2〜S5・宮ヶ瀬北原）の仕上げ
    if r.returncode:
        raise SystemExit(r.stderr.decode("utf-8", "replace")[-1500:])
    font = ImageFont.truetype(config.font_file("gothic"), 22)
    sheet = Image.new("RGB", (1200, 400 * len(jobs)))
    for k, (j, v) in enumerate(zip(jobs, views)):
        a = Image.open(j["photo"]).convert("RGB").resize((600, 400))
        b = Image.open(dc.FOREST / "views" / f"{v['name']}.png").convert("RGB").resize((600, 400))
        for im, t in ((a, "ストリートビュー"), (b, "今のコース")):
            d = ImageDraw.Draw(im); d.rectangle((0, 0, 360, 30), fill=(0, 0, 0)); d.text((5, 2), f"{j['name']} {t}", fill=(255, 255, 0), font=font)
        sheet.paste(a, (0, 400 * k)); sheet.paste(b, (600, 400 * k))
    sheet.save(out, quality=88)
    # 開くのは呼び出した側（ユーザーに見せる時だけ、説明と一緒に）
    print(out)


if __name__ == "__main__":
    main()
