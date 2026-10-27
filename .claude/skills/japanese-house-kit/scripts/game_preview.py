"""ゲーム用の家（yabitsu/house_kit.py）を、scene_data.py の地点のまわりで作って OBJ（＋MTL・絵）に書き出す。street.py に HK_OBJ を渡すと、
Blender の部品の家の代わりにこれを置いて撮る（ゲームでの見た目の確かめ。三角形の数も出す）。
使い方（プロジェクトの根で）: uv run python .claude/skills/japanese-house-kit/scripts/game_preview.py --lat --lon --out <scene_data と同じフォルダ>"""
import argparse, json, sys
import numpy as np
sys.path.insert(0, ".")
from shapely.geometry import LineString, Point
from yabitsu import buildings as bl, config, forest_ground, geo, house_kit as hk
from yabitsu.crosssection import Section
from scripts.forest_build import MANIFEST

ap = argparse.ArgumentParser(); ap.add_argument("--lat", type=float, required=True); ap.add_argument("--lon", type=float, required=True)
ap.add_argument("--out", required=True); ap.add_argument("--radius", type=float, default=100.0)
a = ap.parse_args(); cx, cy = map(float, geo.to_local(a.lat, a.lon))
ground = forest_ground.load_ground(MANIFEST); sec = Section.load_json("data/processed/track_section.json")
m = np.hypot(sec.x - cx, sec.y - cy) < a.radius + 60; road = np.c_[sec.x[m], sec.y[m]]
B = [bl.Building.from_spec(d) for d in json.load(open(config.DATA_MANUAL / "track_buildings.json")) + json.load(open(config.DATA_MANUAL / "s3_buildings.json"))
     if not d.get("gone")]
B = [b for b in B if b.footprint.distance(Point(cx, cy)) < a.radius - 5 and b.footprint.area >= 15 and b.roof.get("type") != "flat" and b.footprint.area <= 260]
gz = lambda x, y: ground.z(np.c_[np.atleast_1d(x), np.atleast_1d(y)])
items, pals = hk.house_meshes(B, gz, road, near_line=LineString(road))
mats = hk.write_textures(pals, f"{a.out}/hk_tex")
tris = sum(len(f) for _, v, f, uv, mm, r in items) * 2
with open(f"{a.out}/hk.mtl", "w") as fm:
    for name, rel in mats.items():
        fm.write(f"newmtl {name}\nKd 1 1 1\nmap_Kd hk_tex/{rel.split('/')[-1]}\n\n")
with open(f"{a.out}/hk.obj", "w") as fo:
    fo.write("mtllib hk.mtl\n"); vb = 1
    for name, v, f, uv, mat, r in items:
        fo.write(f"o {name}\nusemtl {mat}\n")
        for p in v: fo.write(f"v {p[0] - cx:.3f} {p[1] - cy:.3f} {p[2]:.3f}\n")
        for t in uv: fo.write(f"vt {t[0]:.4f} {t[1]:.4f}\n")
        for q in f:
            idx = [int(i) + vb for i in (q[:3] if q[2] == q[3] else q)]
            fo.write("f " + " ".join(f"{i}/{i}" for i in idx) + "\n")
        vb += len(v)
print("houses", len(B), "tris", tris, "per house", tris // max(1, len(B)))
