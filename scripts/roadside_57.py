"""橋Aの覆いを強め、既存データから試作と全線の本数見込みを作る。"""
import argparse
from hashlib import file_digest
from pathlib import Path
import json
import shutil

import numpy as np
from PIL import Image

from scripts import check_valley_scope, dress_mountain, roadside_forest as pilot
from yabitsu import config, ext_config, forestmap, kn5, roadside_cover, roadside_vegetation
from yabitsu.crosssection import Section
from yabitsu.roadside_vegetation import (  # noqa: F401（試作を検証する既存の呼び出し口）
    exclude_grass_pad, grass_meshes, overlaps_open_land, spacing,
)

BASE_EXTENSION = pilot.WORK / "deploy_valleys_41/extension"
OUTPUT = pilot.WORK / "forest_57"
GROUND_COLORS = {"light_060": .6, "middle_050": .5, "dark_040": .4}
GROUND_KEYS = ("mt_terrain", "mt_grass")
VARIANTS = {"fern": ("fern_tall_a", "fern_tall_b"), "sugi": ("mt_sugi_edge_a", "mt_sugi_edge_b")}


def scale_limits(model, height, edge, span=0., *, settings):
    return roadside_vegetation.scale_limits(model, height, edge, span, max_scale=settings["max_scale"])


def fern_packages(source, dest, settings):
    roadside_vegetation.fern_packages(source, dest, settings["fern_height_factor"])


def place(grid, old, road, fm, ground, exclusion, models, manual, spec, *, bridge_station):
    return roadside_vegetation.place(grid, old, road, fm, ground, exclusion, models, manual, spec,
                                     bridge_station=bridge_station, variants=VARIANTS)


def forest_tone(image, brightness):
    pixels = np.asarray(image, dtype=np.float32)
    luma = np.array([.299, .587, .114], dtype=np.float32)
    green = pixels * np.array([.87, 1.06, .85], dtype=np.float32)
    factor = brightness * (pixels @ luma) / np.maximum(green @ luma, 1.e-6)
    return Image.fromarray(np.rint(np.clip(green * factor[..., None], 0, 255)).astype(np.uint8))


def textures(output, processed):
    extension = output / "extension"
    measurements = {}
    for key in GROUND_KEYS:
        image = dress_mountain.retex_diffuse(key)
        before = processed / "dress_mountain/extension" / f"dress_{key}.dds"
        with Image.open(before) as current:
            measurements[key] = {"baseline": average(current)}
        for name, factor in GROUND_COLORS.items():
            dest = extension / "variants" / name / before.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(kn5.dds_bytes(forest_tone(image, factor)))
            with Image.open(dest) as result:
                measurements[key][name] = average(result)
            if name == "middle_050":
                shutil.copyfile(dest, extension / before.name)
    return measurements


def average(image):
    colors = np.asarray(image.convert("RGB")).mean(axis=(0, 1))
    return {"mean_rgb": colors.round(2).tolist(), "mean_luma": round(float(colors @ [.299, .587, .114]), 2),
            "size": list(image.size)}


def sha256(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def write_deployment(output, deployment):
    table = ["案\t検証用コース内パス\t作業用ファイル（forest_57から）\tSHA-256\t戻す操作"]
    fixed = [(rel, rel, "バックアップを復元" if not rel.endswith(".zip") and "grass_fx" not in rel else
              "今回新設なら削除、既存ならバックアップを復元")
             for rel in deployment]
    fixed.extend((f"extension/parts/{p.name}", f"extension/parts/{p.name}", "バックアップを復元")
                 for p in sorted((BASE_EXTENSION / "parts").glob("*.ini")))
    fixed.extend((rel, rel, restore) for rel, restore in (
        ("extension/ext_config.ini", "バックアップを復元"),
        ("extension/parts/grass_fx.ini", "今回新設なら削除、既存ならバックアップを復元"),
    ))
    for dest, src, restore in fixed:
        table.append(f"共通\t{dest}\t{src}\t{sha256(output / src)}\t{restore}")
    for variant in GROUND_COLORS:
        for key in GROUND_KEYS:
            dest = f"extension/dress_{key}.dds"
            src = f"extension/variants/{variant}/dress_{key}.dds"
            table.append(f"{variant}\t{dest}\t{src}\t{sha256(output / src)}\tバックアップを復元")
    (output / "deploy_list.tsv").write_text("\n".join(table) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", "--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true", help="既存の出力先への書き込みを許可")
    args = parser.parse_args()
    output = config.assert_writable(args.out)
    try:
        output.mkdir(parents=True, exist_ok=args.force)
    except FileExistsError:
        parser.error(f"{output} は既にあります。上書きするには --force を指定してください")
    processed = config.DATA_PROCESSED
    section, spec, bounds = pilot.inputs()
    cfg = spec["roadside_vegetation"]
    finish = (BASE_EXTENSION / "parts/dress_mountain.ini").read_text(encoding="utf-8")
    selected, excluded, replacement, meshes = roadside_vegetation.grass_selection(
        processed, finish, pilot.local_polygons("osm_forest_open.json"))
    shutil.copytree(BASE_EXTENSION / "parts", output / "extension/parts", dirs_exist_ok=True)
    ext_config.write_part(output, "grass_fx", roadside_vegetation.grass_ini(selected, meshes, cfg["grass_fx"]))
    ext_config.assemble(output)
    (output / "grass_selection.json").write_text(json.dumps(
        dict(selected=selected, overlaps_open_land=excluded, replacement=replacement), indent=2) + "\n")
    print(f"GrassFX: {len(selected)} メッシュ、開地と重なる {len(excluded)} メッシュを除外", flush=True)
    (output / "texture_colors.json").write_text(json.dumps(textures(output, processed), indent=2) + "\n")
    road = roadside_cover.Road(section)
    manifest = processed / "track"
    fm = forestmap.ForestMap.load(processed / "forest_map.npz")
    grid = road.grid(cfg["max_edge_m"])
    exclusion, _ = pilot.barriers(road, spec, manifest, cover=cfg)
    ground = pilot.ground_from_manifest(manifest)
    manual = pilot.manual_stems(Section.load_json(processed / "track_section.json"), bounds)
    old = np.load(pilot.CURRENT / "trees.npy", mmap_mode="r").view(np.recarray)
    close = pilot.near_road(old, road, max(35., cfg["max_edge_m"] + cfg["forest_neighbor_m"] + 6.))
    tree_dir = output / "extension/trees"
    fern_packages(pilot.PACKAGES, tree_dir, cfg)
    for name in VARIANTS["sugi"]:
        shutil.copyfile(pilot.OUTPUT / "extension/trees" / f"{name}.zip", tree_dir / f"{name}.zip")
    models = {**pilot.package_models(pilot.PACKAGES), **pilot.package_models(tree_dir)}
    print(f"舗装端12 mの格子 {len(grid):,} 点から追加候補を試算", flush=True)
    span = tuple(bounds["s4"][0] + x for x in pilot.PILOT)
    trial = road.interval_mask(grid, span)
    local = place(grid[trial], old[close], road, fm, ground, exclusion, models, manual, spec, bridge_station=bounds["s4"][0])
    use = road.interval_mask(np.c_[local.x, local.y], span)
    local = local[use]
    package_names = sorted(set(local.pattern) & {name for names in VARIANTS.values() for name in names})
    snapshot = pilot.write_snapshot(pilot.CURRENT, output, old, old, local, package_names)
    if snapshot["new_tiles"]:
        raise ValueError(f"登録が必要なタイルがあります: {snapshot['new_tiles']}")
    scope = check_valley_scope.roadside_forest_changes(
        pilot.CURRENT, output, road=road, span=span, exclusion=exclusion, manual=manual)
    if scope:
        raise ValueError(scope)
    write_deployment(output, snapshot["deployment"])
    full = place(grid, old[close], road, fm, ground, exclusion, models, manual, spec, bridge_station=bounds["s4"][0])
    stats = dict(snapshot, base_rows=len(old), pilot_total=len(local), pilot_by_species=pilot.count_species(local),
                 estimate_total=len(full), estimate_by_species=pilot.count_species(full), scope_violations=scope)
    for name in VARIANTS["fern"] + VARIANTS["sugi"] + ("shrub_a", "shrub_b", "sapling_a", "sapling_b"):
        rows = local[local.pattern == name]
        model = models[name]
        if len(rows):
            stats.setdefault("pilot_dimensions", {})[name] = {
                "height_m": [float(rows.scale.min()*model.height), float(rows.scale.max()*model.height)],
                "span_m": [float(rows.scale.min()*model.crown_radius*2), float(rows.scale.max()*model.crown_radius*2)],
                "pavement_edge_m": [float(v) for v in (lambda d: (d.min(), d.max()))(road.edge_distance(np.c_[rows.x, rows.y]))],
            }
    (output / "estimate.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n")
    print(f"試作 {len(local):,} 本・株、全線見込み {len(full):,} 本・株", flush=True)


if __name__ == "__main__":
    main()
