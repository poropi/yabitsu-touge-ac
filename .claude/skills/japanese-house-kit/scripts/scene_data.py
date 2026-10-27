"""町並みの試作の材料：地点（緯度・経度）のまわり R m の家（向きのついた長方形、道に近い長辺が正面）、地面の高さの格子（2 m）、
道の中心線と幅、空中写真（0.2 m/px）、カメラ（道の中心、路面から 2.4 m、指定の向き）。
家は data/manual/track_buildings.json と s3_buildings.json（ゲームと同じ輪郭）。プロジェクトの根で uv run python で流す。
使い方: uv run python .claude/skills/japanese-house-kit/scripts/scene_data.py --lat 35.38531 --lon 139.23738 --heading 334.5 --out export/compare/<場所>/kit/<名前>"""
import argparse, json, os, sys
import numpy as np
from PIL import Image
sys.path.insert(0, ".")
from shapely.geometry import LineString, Point
from yabitsu import buildings as bl, config, forest_ground, geo, gsi, photo
from yabitsu.crosssection import Section
from scripts.forest_build import MANIFEST

ap = argparse.ArgumentParser()
ap.add_argument("--lat", type=float, required=True); ap.add_argument("--lon", type=float, required=True)
ap.add_argument("--heading", type=float, required=True, help="カメラの向き（北から時計回りの度）")
ap.add_argument("--radius", type=float, default=100.0); ap.add_argument("--out", required=True)
a = ap.parse_args(); R = a.radius
cx, cy = map(float, geo.to_local(a.lat, a.lon)); os.makedirs(a.out, exist_ok=True)
ground = forest_ground.load_ground(MANIFEST)
sec = Section.load_json("data/processed/track_section.json")
g = np.arange(-R, R + 0.1, 2.0); X, Y = np.meshgrid(g, g)
Z = ground.z(np.c_[X.ravel() + cx, Y.ravel() + cy]).reshape(X.shape)
z0 = float(np.nanmedian(Z)); Z = np.where(np.isfinite(Z), Z, z0)
Image.fromarray(photo.render_local_image(gsi.TileCache(), geo.from_local, cx - R, cy - R, 2 * R, 0.2)).save(f"{a.out}/photo.png")
m = np.hypot(sec.x - cx, sec.y - cy) < R + 30
road = dict(x=(sec.x[m] - cx).tolist(), y=(sec.y[m] - cy).tolist(), z=sec.z[m].tolist(), hl=sec.hw_left[m].tolist(), hr=sec.hw_right[m].tolist())
rl = LineString(np.c_[sec.x[m], sec.y[m]])
houses = []
for d in json.load(open(config.DATA_MANUAL / "track_buildings.json")) + json.load(open(config.DATA_MANUAL / "s3_buildings.json")):
    if d.get("gone"):
        continue
    b = bl.Building.from_spec(d); fp = b.footprint
    if fp.distance(Point(cx, cy)) > R - 5 or fp.area < 15:
        continue
    rect = np.asarray(fp.minimum_rotated_rectangle.exterior.coords)[:4]
    e = [np.linalg.norm(rect[(i + 1) % 4] - rect[i]) for i in range(4)]
    c = rect.mean(0); L, W = max(e[0], e[1]), min(e[0], e[1])
    v = rect[1] - rect[0] if e[0] >= e[1] else rect[2] - rect[1]; ang = float(np.degrees(np.arctan2(v[1], v[0])))
    nrm = np.array([-np.sin(np.radians(ang)), np.cos(np.radians(ang))])
    p = rl.interpolate(rl.project(Point(*c))); front = 1 if (np.array([p.x, p.y]) - c) @ nrm > 0 else -1
    zg = ground.z(np.asarray(fp.exterior.coords)); zg = zg[np.isfinite(zg)]
    houses.append(dict(id=b.id, c=(c - [cx, cy]).tolist(), L=float(L), W=float(W), ang=ang, front=front, storeys=b.storeys,
                       z_low=float(zg.min()) if len(zg) else z0, z_high=float(zg.max()) if len(zg) else z0,
                       wall=b.wall, roof=b.roof.get("color", "#707070"), flat=b.roof.get("type") == "flat", area=float(fp.area)))
# ゲームの擁壁・塀・ガードレール（manifest の 1WALL_*）。その位置のまま試作に取り込む（石垣はここに石垣の絵を貼る）
walls = []
man = json.load(open(MANIFEST / "manifest.json"))["meshes"]
for nm, info in man.items():
    kind = "ishigaki" if "retaining" in nm else "block" if "block" in nm else "rail" if "_rail_" in nm else None
    if kind is None or not nm.startswith("1WALL"):
        continue
    f = MANIFEST / "meshes" / f"{nm}.npz"
    if not f.exists():
        continue
    dd = np.load(f); v = dd["verts"]
    if np.hypot(v[:, 0] - cx, v[:, 1] - cy).min() > R:
        continue
    walls.append(dict(name=nm, kind=kind, v=(v - [cx, cy, 0]).round(3).tolist(), f=dd["faces"].tolist()))
iz = int(np.argmin(np.hypot(sec.x - cx, sec.y - cy)))
cam = dict(x=float(sec.x[iz] - cx), y=float(sec.y[iz] - cy), z=float(sec.z[iz]) + 2.4, heading=a.heading)
json.dump(dict(R=R, grid=g.tolist(), Z=Z.tolist(), road=road, houses=houses, walls=walls, cam=cam, z0=z0), open(f"{a.out}/scene.json", "w"))
print("houses", len(houses), "walls", len(walls), "->", a.out)
