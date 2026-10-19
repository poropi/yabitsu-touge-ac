"""全域の森を作って AC へ入れる（コマンド1回）。
使い方: uv run python -m scripts.forest_build --area pilot|all [--no-install] [--render] [--refresh-map]
  pilot：S1 の道路から 400 m 以内だけ。all：地形のある範囲すべて。
順番：素材の取得 → 木のモデル → 空中写真の地図 → 地面と植えない範囲 → 配置 → export/forest/<日時>/ へ書き出し
→ AC のコースへコピー → 確認画像（--render）。設定は data/manual/forest.json。
"""
import argparse
import json
import shutil
import subprocess
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
from shapely.geometry import LineString

from yabitsu import config, ext_config, forest_ac, forest_ground, forest_place, forestmap, polyhaven, track, treemodels

CFG = config.PROJECT_ROOT / "data" / "manual" / "forest.json"
MANIFEST = config.DATA_PROCESSED / "track"
MAP = config.DATA_PROCESSED / "forest_map.npz"
PHOTO = config.DATA_PROCESSED / "forest_photo.npy"   # 全域の空中写真（2 m/画素）
BLENDER = Path(r"C:/Program Files/Blender Foundation/Blender 5.2/blender.exe")
NO_FOREST_LAYOUTS = ("s1_landscape_up", "s1_landscape_down")   # Codex の S1 景観確認


def log(t0, msg):
    print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)


def terrain_bounds():
    d = np.load(config.DATA_PROCESSED / "track_terrain_far.npz")
    return float(d["xs"][0]), float(d["ys"][0]), float(d["xs"][-1]), float(d["ys"][-1])


def forest_map(cfg, refresh, out_dir, t0):
    if MAP.exists() and not refresh:
        return forestmap.ForestMap.load(MAP)
    bounds = terrain_bounds()
    if PHOTO.exists():
        rgb = np.load(PHOTO)
    else:
        rgb = forestmap.fetch_mosaic(bounds, cfg["map"]["px_size"], cfg["map"]["z"])
        np.save(PHOTO, rgb)
    c = {**forestmap.DEFAULT_CLASSIFY, **cfg.get("classify", {})}
    fm = forestmap.ForestMap(bounds[0], bounds[3], c["cell_m"], forestmap.classify(rgb, cfg["map"]["px_size"], c))
    forestmap.clear_polygons(fm, forestmap.osm_open_areas(bounds))
    # 集落（家から settle_m 以内）は暗い緑だけを森に：田んぼ・畑・屋根の並びを森にしない
    houses = [(v, t) for n, mat, v, t in forest_ground._load(MANIFEST) if forest_ground.mesh_class(n, mat) == "building"]
    mask = forestmap.settlement_mask(fm, forestmap.osm_buildings(bounds), houses, c["settle_m"])
    forestmap.settlement_filter(fm, rgb, cfg["map"]["px_size"], mask, c)
    fm.save(MAP)
    forestmap.preview(rgb, fm, out_dir / "forest_map.png", max_px=6000)
    log(t0, f"空中写真の地図を作りました（森 {np.mean(fm.cls > 0) * 100:.0f}%、針葉樹 {np.mean(fm.cls == 1) * 100:.0f}%）")
    return fm


def pilot_area():
    cl = np.load(config.DATA_PROCESSED / "track_centerline.npz")
    s0, s1 = track.load_bounds()["s1"]
    sel = (cl["s"] >= s0) & (cl["s"] <= s1)
    return LineString(np.c_[cl["x"][sel], cl["y"][sel]]).buffer(400.0)


def _at(sec_name, s_local, cl, bounds):
    """区間の起点から s_local m の中心線の点と向き（単位ベクトル）。"""
    s = bounds[sec_name][0] + s_local
    i = int(np.clip(np.searchsorted(cl["s"], s), 1, len(cl["s"]) - 2))
    d = np.array([cl["x"][i + 1] - cl["x"][i - 1], cl["y"][i + 1] - cl["y"][i - 1]])
    return np.array([cl["x"][i], cl["y"][i]]), d / np.linalg.norm(d)


def view_specs(area, ground):
    """確認画像のカメラ：上から2か所（空中写真と並べる）、運転席1か所、森の中を歩く視点1か所。"""
    cl = np.load(config.DATA_PROCESSED / "track_centerline.npz")
    b = track.load_bounds()
    picks = ([("s1", 800), ("s1", 2500)], ("s1", 350), ("s1", 1700)) if area == "pilot" else         ([("s4", 3000), ("s2", 1500)], ("s4", 1000), ("s5", 3000))
    views = []
    for sec, sl in picks[0]:
        c, _ = _at(sec, sl, cl, b)
        z = float(ground.z(c[None])[0])
        views.append({"name": f"top_{sec}_{sl}", "type": "top", "center": c.tolist(), "size": 600.0, "z": z,
                      "window": [c[0] - 320, c[1] - 320, c[0] + 320, c[1] + 320], "lod": 2})
    c, h = _at(*picks[1], cl, b)
    z = float(forest_ground.load_ground(MANIFEST, prefixes=("1ROAD_",)).z(c[None])[0])   # 路面の高さ
    views.append({"name": f"drive_{picks[1][0]}_{picks[1][1]}", "type": "persp", "eye": [*c, z + 1.2],
                  "target": [*(c + h * 60), z + 1.2 + 2.0], "lens": 22, "lod": 0,
                  "window": [c[0] - 350, c[1] - 350, c[0] + 350, c[1] + 350]})
    c, h = _at(*picks[2], cl, b)
    side = np.array([-h[1], h[0]])
    for k in (1, -1):        # 道の左右のうち、木のある側
        e = c + side * k * 25
        ze = ground.z(e[None])[0]
        if np.isfinite(ze):
            break
    views.append({"name": f"walk_{picks[2][0]}_{picks[2][1]}", "type": "persp", "eye": [*e, float(ze) + 1.6],
                  "target": [*(e + side * k * 40), float(np.nan_to_num(ground.z((e + side * k * 40)[None])[0], nan=ze)) + 1.6],
                  "lens": 20, "lod": 0,
                  "window": [e[0] - 300, e[1] - 300, e[0] + 300, e[1] + 300]})
    return views


def compare_top(out, views):
    """上からの画像の左に、同じ範囲の空中写真を並べる。"""
    from PIL import Image

    from yabitsu import geo, gsi, photo
    for v in views:
        if v["type"] != "top":
            continue
        r = Image.open(out / "views" / f"{v['name']}.png").convert("RGB")
        x0, y0 = v["center"][0] - v["size"] / 2, v["center"][1] - v["size"] / 2
        a = Image.fromarray(photo.render_local_image(gsi.TileCache(), geo.from_local, x0, y0, v["size"], v["size"] / r.width))
        img = Image.new("RGB", (r.width * 2, r.height)); img.paste(a.resize(r.size), (0, 0)); img.paste(r, (r.width, 0))
        img.save(out / "views" / f"{v['name']}_compare.png")


def old_tree_meshes():
    m = json.loads((MANIFEST / "manifest.json").read_text(encoding="utf-8"))
    names = sorted(n for n, e in m["meshes"].items() if e["material"] == "tree")
    return [x for n in names for x in (n, f"{n}_back")]   # kn5 では両面の材質は「_back」の付いた裏側の部品が別にある


def install(ext_src: Path, track_dir: Path):
    """AC のコースの extension/ を森のものに入れ替える（自分が書いたファイルだけ）。"""
    dst = track_dir / "extension"
    cfg = dst / "ext_config.ini"
    if cfg.exists() and not cfg.read_text(encoding="utf-8").startswith(forest_ac.MARK):
        raise RuntimeError(f"{cfg} はこのスクリプトが書いたものではありません")
    if (dst / "trees").exists():
        shutil.rmtree(dst / "trees")
    shutil.copytree(ext_src / "extension" / "trees", dst / "trees")
    (dst / "parts").mkdir(parents=True, exist_ok=True)
    shutil.copy2(ext_src / "extension" / "parts" / "forest.ini", dst / "parts" / "forest.ini")
    ext_config.assemble(track_dir)        # 森の部品だけを入れ替え、ほかの部品（S3 の仕上げ）は残す
    forest_ac.write_layout_off(track_dir, [l for l in NO_FOREST_LAYOUTS if (track_dir / l).exists()])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--area", choices=("pilot", "all"), default="pilot")
    ap.add_argument("--no-install", action="store_true")
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--refresh-map", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    cfg = json.loads(CFG.read_text(encoding="utf-8"))
    out = config.PROJECT_ROOT / "export" / "forest" / datetime.now().strftime("%Y%m%d_%H%M%S")
    ext = out / "extension"
    out.mkdir(parents=True)

    polyhaven.fetch(log=lambda m: log(t0, m))
    models = treemodels.write_all(ext / "trees")
    log(t0, f"木のモデル {len(models)} 種類")
    fm = forest_map(cfg, a.refresh_map, out, t0)
    ground = forest_ground.load_ground(MANIFEST)
    keep = forest_ground.load_keepout(MANIFEST, cfg.get("clearance", {}))
    keep.add_lines(*forest_ground.osm_road_lines(terrain_bounds()))
    log(t0, "地面と植えない範囲")
    area = pilot_area() if a.area == "pilot" else None
    surface = forest_ground.load_ground(MANIFEST, prefixes=("1ROAD_",))     # 路面
    trees = forest_place.place(area, fm, ground, keep, models, cfg.get("place", {}), road_surface=surface)
    cl = np.load(config.DATA_PROCESSED / "track_centerline.npz")
    cxy = np.c_[cl["x"], cl["y"]]
    cz = surface.z(cxy)
    cz = np.where(np.isfinite(cz), cz, np.nan_to_num(ground.z(cxy)))
    keep_sight = forest_place.sightline_keep(trees, models, cxy, cz, cfg.get("sight", {}))
    np.save(out / "removed_sight.npy", np.asarray(trees[~keep_sight]))
    trees = trees[keep_sight]
    log(t0, f"コースの視界をさえぎる木を {int((~keep_sight).sum()):,} 本取り除きました")
    counts = Counter(p.split("_")[0] for p in trees.pattern)
    log(t0, f"配置 {len(trees):,} 本 {dict(counts)}")
    np.save(out / "trees.npy", np.asarray(trees))
    lists = forest_ac.write_lists(trees, ext)
    forest_ac.write_ext_config(out, lists, old_tree_meshes())
    (out / "summary.json").write_text(json.dumps({"area": a.area, "trees": len(trees), "by_species": counts,
                                                  "lists": len(lists), "seconds": round(time.time() - t0)},
                                                 ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
    if not a.no_install:
        install(out, config.TRACK_DIR)
        log(t0, f"AC へ入れました：{config.TRACK_DIR / 'extension'}")
    if a.render:
        (out / "views.json").write_text(json.dumps(view_specs(a.area, ground), ensure_ascii=False, indent=1),
                                         encoding="utf-8", newline="\n")
        subprocess.run([str(BLENDER), "-b", "--factory-startup", "--python",
                        str(config.PROJECT_ROOT / "blender_scripts" / "render_forest.py"), "--", str(out), a.area],
                       check=True, timeout=1800)
        compare_top(out, json.loads((out / "views.json").read_text(encoding="utf-8")))
        log(t0, f"確認画像：{out / 'views'}")
    log(t0, f"完了：{out}")


if __name__ == "__main__":
    main()
