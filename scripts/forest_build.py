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
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import shapely
from shapely.geometry import LineString, Polygon

from yabitsu import (acdata, config, ext_config, forest_ac, forest_ground, forest_light, forest_place, forestmap,
                     layouts, polyhaven, roadside_cover, roadside_inputs, roadside_vegetation, track, treemodels)
from yabitsu.crosssection import Section

CFG = config.PROJECT_ROOT / "data" / "manual" / "forest.json"
MANIFEST = config.DATA_PROCESSED / "track"
MAP = config.DATA_PROCESSED / "forest_map.npz"
PHOTO = config.DATA_PROCESSED / "forest_photo.npy"   # 全域の空中写真（2 m/画素）
BLENDER = config.BLENDER_EXE
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
    forestmap.farm_filter(fm, rgb, cfg["map"]["px_size"], forestmap.osm_buildings(bounds), houses, c)
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
    light = ext_src / forest_light.KN5_NAME
    installed_light = track_dir / light.name
    use_light = light.is_file()
    if use_light:
        track_dir.mkdir(parents=True, exist_ok=True)
        pending = installed_light.with_suffix(".kn5.tmp")
        try:
            shutil.copy2(light, pending)
            pending.replace(installed_light)
        finally:
            pending.unlink(missing_ok=True)
    if (dst / "trees").exists():
        shutil.rmtree(dst / "trees")
    shutil.copytree(ext_src / "extension" / "trees", dst / "trees")
    (dst / "parts").mkdir(parents=True, exist_ok=True)
    shutil.copy2(ext_src / "extension" / "parts" / "forest.ini", dst / "parts" / "forest.ini")
    ext_config.assemble(track_dir)        # 森の部品だけを入れ替え、ほかの部品（S3 の仕上げ）は残す
    forest_ac.write_layout_off(track_dir, [l for l in NO_FOREST_LAYOUTS if (track_dir / l).exists()])
    if not use_light:
        installed_light.unlink(missing_ok=True)
    for layout in layouts.load_layouts():
        models_ini = track_dir / f"models_{layout.name}.ini"
        if models_ini.is_file() and (use_light or f"FILE={light.name}" in models_ini.read_text(encoding="utf-8")):
            acdata.write_models_ini(models_ini, layouts.kn5_files(layout.name, track_dir=track_dir))


def finish_forest(out, lists, hidden_meshes, *, light=None, manifest=None):
    """追記済みの森を配布用の CSP 一覧と、設定に応じて軽い木の KN5 にする。"""
    if light is not None and light.enabled:
        tree_dir = out / "extension/trees"
        manifest = manifest if manifest is not None else MANIFEST / "manifest.json"
        with TemporaryDirectory(dir=tree_dir.parent) as temp:
            filtered = Path(temp)
            result = forest_light.build(tree_dir, tree_dir, manifest, {light.texture: out / forest_light.KN5_NAME},
                                        light, filtered_lists=filtered)
            for tile in filtered.iterdir():
                tile.replace(tree_dir / tile.name)
        prefixes = tuple(f"{name}_" for name in light.tall_families)
        for package in tree_dir.glob("*.zip"):
            if package.name.startswith(prefixes) and package.name not in result["retained_packages"]:
                package.unlink()
    forest_ac.write_ext_config(out, lists, hidden_meshes)


def publish_forest(out, lists, light, ground, t0, args):
    tree_dir = out / "extension/trees"
    # 描画は元の CSP モデルを使う。軽い木の生成で消える前に描画用の控えを取る。
    preview = TemporaryDirectory(prefix=".render-trees-", dir=out) if args.render and light.enabled else nullcontext()
    with preview as saved:
        model_dir = Path(saved) if saved else tree_dir
        if saved:
            for package in tree_dir.glob("*.zip"):
                shutil.copyfile(package, model_dir / package.name)
        finish_forest(out, lists, old_tree_meshes(), light=light)
        if not args.no_install:
            install(out, config.TRACK_DIR)
            log(t0, f"AC へ入れました：{config.TRACK_DIR / 'extension'}")
        if args.render:
            views = view_specs(args.area, ground)
            (out / "views.json").write_text(json.dumps(views, ensure_ascii=False, indent=1),
                                           encoding="utf-8", newline="\n")
            subprocess.run([str(BLENDER), "-b", "--factory-startup", "--python-exit-code", "1", "--python",
                            str(config.PROJECT_ROOT / "blender_scripts" / "render_forest.py"),
                            "--", str(out), args.area, str(model_dir)], check=True, timeout=1800)
            compare_top(out, views)
            log(t0, f"確認画像：{out / 'views'}")


def pass_island():
    """ヤビツ峠の分かれ道の島の形（data/manual/s1_pass.json から、コースと同じ作り方で）。"""
    from yabitsu import pass_area
    sec = Section.load_json(config.DATA_PROCESSED / "track_section.json")
    return pass_area.build(pass_area.load(config.DATA_MANUAL / "s1_pass.json"), sec).polys["island"]


def roadside_additions(trees, tree_dir, cfg, *, area=None, ground=None):
    """合格時の植え方で、既存の森に追記する行とシダの ZIP を作る。"""
    settings = cfg["roadside_vegetation"]
    section = Section.load_json(MANIFEST.parent / "track_section.json")
    bounds = track.load_bounds(MANIFEST.parent)
    road = roadside_cover.Road(vars(section))
    grid = road.grid(settings["max_edge_m"])
    if area is not None:
        grid = grid[shapely.intersects_xy(area, *grid.T)]
    exclusion, _ = roadside_inputs.barriers(road, cfg, MANIFEST, cover=settings)
    ground = roadside_inputs.ground_from_manifest(MANIFEST, base=ground)
    manual = roadside_inputs.manual_stems(section, bounds)
    reach = max(35.0, settings["max_edge_m"] + settings["forest_neighbor_m"] + 6.0)
    old = trees[roadside_inputs.near_road(trees, road, reach)]

    roadside_vegetation.fern_packages(tree_dir, tree_dir, settings["fern_height_factor"])
    models = roadside_inputs.package_models(tree_dir)
    # 配置時の枝の判定は承認時の寸法を使い、出力は既存のスギの名前と樹高に合わせる。
    sugi = {name: SimpleNamespace(height=geometry["height"], crown_base=geometry["crown_base"],
                                  crown_radius=np.float32(geometry["crown_radius"]))
            for name, geometry in settings["sugi_placement_geometry"].items()}
    variants = {"fern": ("fern_tall_a", "fern_tall_b"), "sugi": tuple(sugi)}
    added = roadside_vegetation.place(grid, old, road, forestmap.ForestMap.load(MAP), ground, exclusion,
                                      models | sugi, manual, cfg, bridge_station=bounds["s4"][0], variants=variants)
    for name, geometry in sugi.items():
        match = added.pattern == name
        added.scale[match] = added.scale[match] * geometry.height / models[name].height
    return added


def extend_snapshot(source, out, package_dir, cfg, *, light=None):
    """手植えなどを反映した森のスナップショットに、道ぎわの行を追記する。"""
    from scripts.add_extra_trees import list_files

    trees = np.load(source / "trees.npy", mmap_mode="r").view(np.recarray)
    base_lists = list_files(source)
    if not base_lists:
        raise ValueError(f"{source}: 森のタイル一覧がありません")
    ext = out / "extension"
    tree_dir = ext / "trees"
    tree_dir.mkdir(parents=True)
    for package in package_dir.glob("*.zip"):
        if not package.name.startswith("mt_"):
            shutil.copyfile(package, tree_dir / package.name)
    add = roadside_additions(trees, tree_dir, cfg)
    lists = forest_ac.write_lists(add, ext, base_lists=base_lists)
    finish_forest(out, lists, old_tree_meshes(), light=light)
    np.save(out / "trees.npy", np.concatenate((trees, add)))
    return len(add)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--area", choices=("pilot", "all"), default="pilot")
    ap.add_argument("--no-install", action="store_true")
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--refresh-map", action="store_true")
    ap.add_argument("--base-forest", type=Path, help="追記前の trees.npy とタイル一覧を含む全域の森")
    ap.add_argument("--tree-packages", type=Path, help="--base-forest 用の通常モデル ZIP。既定はその extension/trees")
    a = ap.parse_args()
    if a.base_forest is not None and (a.area != "all" or a.render or a.refresh_map):
        ap.error("--base-forest を使うときは --area all を指定してください（--render・--refresh-map は対象外）")
    if a.tree_packages is not None and a.base_forest is None:
        ap.error("--tree-packages には --base-forest が必要です")
    t0 = time.time()
    cfg = json.loads(CFG.read_text(encoding="utf-8"))
    light = forest_light.load_settings()
    out = config.PROJECT_ROOT / "export" / "forest" / datetime.now().strftime("%Y%m%d_%H%M%S")
    ext = out / "extension"
    if a.base_forest is not None:
        count = extend_snapshot(a.base_forest, out, a.tree_packages or a.base_forest / "extension/trees", cfg, light=light)
        if not a.no_install:
            install(out, config.TRACK_DIR)
        log(t0, f"山の道ぎわに {count:,} 本・株を追加：{out}")
        return

    out.mkdir(parents=True)
    polyhaven.fetch(log=lambda m: log(t0, m))
    models = treemodels.write_all(ext / "trees")
    log(t0, f"木のモデル {len(models)} 種類")
    fm = forest_map(cfg, a.refresh_map, out, t0)
    ground = forest_ground.load_ground(MANIFEST)
    # 湖（宮ヶ瀬湖）の OSM の輪郭が陸の斜面まで含み、森が消えていた所を戻す（保存済みの地図を作り直さずに、読んだあとで毎回）
    c = {**forestmap.DEFAULT_CLASSIFY, **cfg.get("classify", {})}
    raw = forestmap.classify(np.load(PHOTO), cfg["map"]["px_size"], c)
    level = json.loads((config.DATA_MANUAL / "s5_water.json").read_text(encoding="utf-8"))["lake_level_m"]
    n = forestmap.restore_above_water(fm, raw, forestmap.osm_lake_polygons(terrain_bounds()), ground.z, level)
    log(t0, f"湖の輪郭の中の陸（湖面 {level} m＋3 m より上）を森に戻した：{n:,} マス（{n * fm.cell ** 2 / 1e4:.1f} ha）")
    # 山の斜面は森に：OSM の土地利用（湖を除く）と家から 25 m の外の、傾き 15° 以上の開けた土地で、写真が森か緑っぽい所
    bounds = terrain_bounds()
    lakes = forestmap.osm_lake_polygons(bounds)
    keep_open = np.zeros(fm.cls.shape, bool)
    probe = forestmap.ForestMap(fm.x0, fm.y1, fm.cell, np.ones(fm.cls.shape, np.uint8))
    forestmap.clear_polygons(probe, forestmap.osm_open_areas(bounds)); keep_open |= probe.cls == 0
    probe.cls[:] = 1; forestmap.clear_polygons(probe, lakes); keep_open &= ~(probe.cls == 0)   # 湖の輪郭は高さで決めた（restore_above_water）
    houses = [(v, t_) for nm, mat, v, t_ in forest_ground._load(MANIFEST) if forest_ground.mesh_class(nm, mat) == "building"]
    keep_open |= forestmap.settlement_mask(fm, forestmap.osm_buildings(bounds), houses, 25.0)
    # 写真に合わせて木を決めた範囲（species_zones・fill_bare の円、keep_map の作り込んだ場所のまわり）は今までの判定のまま
    from yabitsu import geo as _geo
    yy, xx = np.mgrid[0:fm.cls.shape[0], 0:fm.cls.shape[1]]
    for zz in cfg.get("species_zones", []) + cfg.get("fill_bare", []) + cfg.get("keep_map", []):   # keep_map：作り込んだ場所のまわり
        zx, zy = _geo.to_local(*zz["center_ll"]); rad = float(zz.get("radius", 0.0))
        r0, r1 = int((fm.y1 - zy - rad) / fm.cell), int((fm.y1 - zy + rad) / fm.cell) + 1
        c0, c1 = int((zx - rad - fm.x0) / fm.cell), int((zx + rad - fm.x0) / fm.cell) + 1
        r0, c0 = max(r0, 0), max(c0, 0)
        sub = (fm.x0 + (xx[r0:r1, c0:c1] + 0.5) * fm.cell - zx) ** 2 + (fm.y1 - (yy[r0:r1, c0:c1] + 0.5) * fm.cell - zy) ** 2 <= rad ** 2
        keep_open[r0:r1, c0:c1] |= sub
    slope = forestmap.slope_map(fm, ground.z, (fm.cls == 0) & ~keep_open)
    n = forestmap.mountain_forest(fm, raw, np.load(PHOTO), cfg["map"]["px_size"], keep_open, slope)
    log(t0, f"山の斜面（15° 以上・家と土地利用の外）を森にした：{n:,} マス（{n * fm.cell ** 2 / 1e4:.1f} ha）")
    keep = forest_ground.load_keepout(MANIFEST, cfg.get("clearance", {}))
    keep.add_lines(*forest_ground.osm_road_lines(terrain_bounds()))
    log(t0, "地面と植えない範囲")
    area = pilot_area() if a.area == "pilot" else None
    surface = forest_ground.load_ground(MANIFEST, prefixes=("1ROAD_",))     # 路面
    from yabitsu import geo
    zones = [{**z, "center": [float(v) for v in geo.to_local(*z["center_ll"])],
              **({"points": [[float(v) for v in geo.to_local(*p)] for p in z["points_ll"]]} if "points_ll" in z else {}),
              **({"inside": pass_island()} if z.get("inside_pass_island") else {}),
              **({"inside": Polygon(np.c_[geo.to_local(*np.asarray(z["inside_ll"], float).T)]).buffer(0)} if "inside_ll" in z else {})}
             for z in cfg.get("species_zones", [])]
    trees = forest_place.place(area, fm, ground, keep, models, cfg.get("place", {}), road_surface=surface, zones=zones)
    # keep_sight：写真に合わせて植えた所（峠のトイレ前の紅葉など）。コースの視界の判定では消さない
    keep_areas = [z["inside"] for z in zones if z.get("keep_sight") and "inside" in z]
    # sight_exempt：盛り土が見通しをもともとさえぎる所（菜の花台のヘアピンの内側の島）。木も草も見通しの判定で消さない
    # （判定は地形を見ないので、島の木と縁の草まで消えて土がむき出しになっていた。2026-10-30 ユーザーが絵で確認した計画）
    exempt_areas = [z["inside"] for z in zones if z.get("sight_exempt") and "inside" in z]
    for fb in cfg.get("fill_bare", []):   # むき出しの地面を埋める（峠のまわりなど）
        fz = {**fb, "center": [float(v) for v in geo.to_local(*fb["center_ll"])],
              "open": [[*map(float, geo.to_local(*o["center_ll"])), o["radius"]] for o in fb.get("open", [])]}
        add = forest_place.fill_bare(trees, ground, keep, treemodels.SPECIES, fz)
        trees = np.concatenate([np.asarray(trees), np.asarray(add)]).view(np.recarray)
        log(t0, f"むき出しの地面を埋めた：{fb['note'][:20]} に {len(add):,} 本")
    cl = np.load(config.DATA_PROCESSED / "track_centerline.npz")
    cxy = np.c_[cl["x"], cl["y"]]
    cz = surface.z(cxy)
    cz = np.where(np.isfinite(cz), cz, np.nan_to_num(ground.z(cxy)))
    zones = []                      # 眺めの開けた場所（data/manual/forest.json の view_clear：緯度経度・半径）
    for zn in cfg.get("view_clear", []):
        from yabitsu import geo
        x, y = geo.to_local(*zn["center_ll"])
        zones.append({"center": [float(x), float(y)], "radius": zn["radius"], "z": zn["z"], "margin": zn.get("margin", 0.0)})
    vk = forest_place.view_clear_keep(trees, models, zones)
    trees = trees[vk]
    log(t0, f"眺めの開けた場所の木を {int((~vk).sum()):,} 本取り除きました")
    sight = cfg.get("sight", {})
    keep_sight = forest_place.sightline_keep(trees, models, cxy, cz, sight, include_shrubs=True)
    # 写真に合わせて植えた所（keep_sight）も、目の高さから 1 m の高さの路面への見通しにかかる木は取り除く（道の上の高い枝葉は残る）。
    # 丸ごと外していたら、塩水橋（S5 4,360〜4,475 m）で道をおおう広葉樹に 7.5〜10 m 先しか見えなかった（2026-10-28）
    if keep_areas:
        low = forest_place.sightline_keep(trees, models, cxy, cz, {**sight, "target_h": [1.0]}, include_shrubs=True)
        for g in keep_areas:
            keep_sight |= shapely.contains_xy(g, np.asarray(trees.x, float), np.asarray(trees.y, float)) & low
    for g in exempt_areas:
        keep_sight |= shapely.contains_xy(g, np.asarray(trees.x, float), np.asarray(trees.y, float))
    np.save(out / "removed_sight.npy", np.asarray(trees[~keep_sight]))
    trees = trees[keep_sight]
    log(t0, f"コースの視界をさえぎる木を {int((~keep_sight).sum()):,} 本取り除きました")
    add = roadside_additions(trees, ext / "trees", cfg, area=area, ground=ground)
    trees = np.concatenate((trees, add)).view(np.recarray)
    log(t0, f"山の道ぎわに {len(add):,} 本・株を追加")
    counts = Counter(p.split("_")[0] for p in trees.pattern)
    log(t0, f"配置 {len(trees):,} 本 {dict(counts)}")
    np.save(out / "trees.npy", np.asarray(trees))
    lists = forest_ac.write_lists(trees, ext)
    (out / "summary.json").write_text(json.dumps({"area": a.area, "trees": len(trees), "by_species": counts,
                                                  "lists": len(lists), "seconds": round(time.time() - t0)},
                                                 ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
    publish_forest(out, lists, light, ground, t0, a)
    log(t0, f"完了：{out}")


if __name__ == "__main__":
    main()
