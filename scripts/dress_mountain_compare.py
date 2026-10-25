"""山道の仕上げの確認：ストリートビューと同じ位置・向きで描き、写真｜今（仕上げ前）｜仕上げ後 を並べる。
使い方: uv run python -m scripts.dress_mountain_compare s4:1200f s2:0500b ...   （区間は内部の名前 s2・s1・s4・s5）
       --after-only で仕上げ後だけ描く（写真｜仕上げ後）
写真：S2・S1 は .superpowers/sdd/2026-10-20-mountain-dress/sv（50 m ごと、画角 80°・水平）、
      S4・S5 は以前の調査の sv（画角 90°・3° 下向き）。"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from yabitsu import config, forest_ground, geo

SDD = config.PROJECT_ROOT / ".superpowers" / "sdd"
NEW = SDD / "2026-10-20-mountain-dress"
OLD = {"s4": SDD / "2026-10-18-yabitsu-s4", "s5": SDD / "2026-10-18-yabitsu-s5"}
FOREST = sorted((config.PROJECT_ROOT / "export" / "forest").glob("2026*"))[-1]
DRESS = [config.DATA_PROCESSED / "dress_s3", config.DATA_PROCESSED / "dress_mountain"]
BLENDER = str(config.BLENDER_EXE)
OUT = config.COMPARE_DIR
EYE = 2.5
LABEL = {"s2": "S2", "s1": "S3", "s4": "S4", "s5": "S5"}       # 表示の区間名


def spots():
    """{(区間, s, 'f'|'b'): (写真, lat, lon, 方位, 画角, 下向き°)}"""
    out = {}
    for sec in ("s2", "s1"):
        p = NEW / f"sv_{sec}_points.json"
        if not p.exists():
            continue
        for q in json.loads(p.read_text(encoding="utf-8")):
            for fb in "fb":
                img = NEW / "sv" / f"{sec}_{q['s']:04d}_{fb}.png"
                out[(sec, q["s"], fb)] = (img, q["lat"], q["lon"], (q["brg"] + (180 if fb == "b" else 0)) % 360, 80.0, 0.0)
    for sec, d in OLD.items():
        for js in sorted(d.glob("sv_batch_*.js")):
            jobs = json.loads(re.search(r"const jobs = (\[.*?\]);", js.read_text(encoding="utf-8"), re.S).group(1))
            for n, la, lo, yaw in jobs:
                s = int(n.split("_")[1])
                for fb in "fb":
                    img = d / "sv" / f"{n}_{fb}.png"
                    out[(sec, s, fb)] = (img, la, lo, (yaw + (180 if fb == "b" else 0)) % 360, 90.0, 3.0)
    return out


def render(views, dress_dirs, tag):
    for v in views:
        v["name"] = f"{tag}_{v['key']}"
    (FOREST / "views.json").write_text(json.dumps(views), encoding="utf-8")
    env = {**os.environ, "DRESS_DIR": os.pathsep.join(str(d) for d in dress_dirs if d.exists())}
    r = subprocess.run([BLENDER, "-b", "--factory-startup", "--python-exit-code", "1", "--python", str(config.PROJECT_ROOT / "blender_scripts" / "render_forest.py"),
                        "--", str(FOREST), "mountain"], env=env, capture_output=True, timeout=3600)
    if r.returncode:
        raise SystemExit(r.stderr.decode("utf-8", "replace")[-2000:])


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    after_only = "--after-only" in sys.argv
    sp = spots()
    road = forest_ground.load_ground(config.DATA_PROCESSED / "track", prefixes=("1ROAD_",))
    views, picks = [], []
    for a in args:
        sec, rest = a.split(":")
        s, fb = int(rest[:-1]), rest[-1]
        img, la, lo, brg, fov, pitch = sp[(sec, s, fb)]
        c = np.array([float(v) for v in geo.to_local(la, lo)])
        b = np.radians(brg)
        d = np.array([np.sin(b), np.cos(b)])
        z = float(np.nan_to_num(road.z(c[None])[0], nan=0.0))
        lens = 36.0 / (2 * np.tan(np.radians(fov) / 2)) * (900 / 900)
        drop = 50 * np.tan(np.radians(pitch))
        views.append({"key": f"{sec}_{s:05d}_{fb}", "type": "persp", "eye": [*c, z + EYE], "target": [*(c + d * 50), z + EYE - drop],
                      "lens": float(lens), "lod": 0, "res": [900, 600], "window": [c[0] - 250, c[1] - 250, c[0] + 250, c[1] + 250]})
        picks.append((sec, s, fb, img))
    if not after_only:
        render([dict(v) for v in views], [DRESS[0]], "before")
    render([dict(v) for v in views], DRESS, "after")
    font = ImageFont.truetype(config.font_file("gothic"), 24)
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for v, (sec, s, fb, img) in zip(views, picks):
        cols = [("ストリートビュー", Image.open(img).convert("RGB").resize((900, 600)))]
        if not after_only:
            cols.append(("今", Image.open(FOREST / "views" / f"before_{v['key']}.png").convert("RGB")))
        cols.append(("仕上げ後", Image.open(FOREST / "views" / f"after_{v['key']}.png").convert("RGB")))
        row = Image.new("RGB", (900 * len(cols), 600))
        dr = ImageDraw.Draw(row)
        for j, (t, im) in enumerate(cols):
            row.paste(im.resize((900, 600)), (900 * j, 0))
            up = (fb == "f") == (sec in ("s2", "s1"))             # S2・S3 は s が増える向きが上り、S4・S5 は下り
            label = f"{LABEL[sec]} {s}m {'上り' if up else '下り'}  {t}"
            dr.rectangle((900 * j, 0, 900 * j + 14 * len(label) + 40, 36), fill=(0, 0, 0)); dr.text((900 * j + 8, 2), label, fill=(255, 255, 0), font=font)
        rows.append(row)
    outs = []
    per = 4
    for k in range(0, len(rows), per):
        chunk = rows[k:k + per]
        W = chunk[0].width // 2
        sheet = Image.new("RGB", (W, 300 * len(chunk)))
        for j, r in enumerate(chunk):
            sheet.paste(r.resize((W, 300)), (0, 300 * j))
        out = OUT / f"mountain_{k // per:02d}.jpg"
        sheet.save(out, quality=88)
        outs.append(out)
    print("\n".join(map(str, outs)))


if __name__ == "__main__":
    main()
