"""既存の森を入力に、道ぎわの追加・S4試作の一覧・範囲検査を組み立てる。"""
import argparse
from collections import Counter, defaultdict
from configparser import ConfigParser
import json
from pathlib import Path
import re
import shutil
from zipfile import ZipFile

import numpy as np
import shapely
from scipy.spatial import cKDTree

from scripts import add_extra_trees
from scripts.align_s5_8960_trees import _line, _patch_tile
from yabitsu import config, csptree, forest_place, forestmap, roadside_cover, track, treemodels
from yabitsu.crosssection import Section
from yabitsu.roadside_inputs import (barriers, ground_from_manifest, local_polygons, manual_stems,
                                    near_road, package_models)

WORK = config.PROJECT_ROOT / "export/compare/route70/work"
CURRENT = WORK / "trees_54/forest"
PACKAGES = WORK / "supervisor_41.fmEVx6/deliver/forest/extension/trees"
OUTPUT = WORK / "forest_56"
VALIDATION_FOREST_INI = config.COMPARE_DIR / "windows-validation/overlay/extension/parts/forest.ini"
PILOT = (1550., 1900.)
PILOT_MODEL_REACH_M = 160.
TILE_HEADER = (
    b"configure: angle variance = 0, 0\nconfigure: size variance = 1, 1\n"
    b"configure: width variance = 1, 1\nconfigure: color variance = 0, 0, 0\n"
)


def inputs():
    sec = json.loads((config.DATA_PROCESSED / "track_section.json").read_text(encoding="utf-8"))
    spec = json.loads((config.DATA_MANUAL / "forest.json").read_text(encoding="utf-8"))
    return sec, spec, track.load_bounds()


def manual_model_mask(rows, manual_xy):
    """手植えの実データの根元を識別する。CSVの位置と同じ幹だけを対象にする。"""
    if manual_xy is None or not len(manual_xy):
        return np.zeros(len(rows), bool)
    return cKDTree(manual_xy).query(np.c_[rows.x, rows.y])[0] < .02


def pilot_patterns(rows, towards, *, rotate_by_name, manual_xy=None):
    """既存の向き指定を温存する場合は、最も道路側を向く派生モデルを参照する。"""
    result = rows.astype([(n, "U32" if n == "pattern" else rows.dtype[n]) for n in rows.dtype.names]).view(np.recarray)
    d = np.linalg.norm(towards, axis=1)
    kind = roadside_cover.genus(rows.pattern)
    names = np.select([kind == "sugi", kind == "hinoki", kind == "broadleaf", kind == "sapling"],
                      [np.where(d <= 10, "sugi_edge", "sugi_forest"), "hinoki",
                       np.where(d <= 10, "roadside", "oak"), np.where(d <= 10, "roadside", "oak")], default="")
    for i in np.flatnonzero((names != "") & ~manual_model_mask(rows, manual_xy)):
        variant = "a" if rows.pattern[i][-1] in "ac" else "b"
        name = f"mt_{names[i]}_{variant}"
        if names[i] in ("sugi_edge", "roadside"):
            angle = np.degrees(np.arctan2(towards[i,1], towards[i,0])) % 360
            if rotate_by_name:
                sector = int(np.rint((angle - np.rint(rows.angle_deg[i])) / 22.5)) % 16
                if sector:
                    name = f"{name}_r{sector:02d}"
            else:
                result.angle_deg[i] = angle
        result.pattern[i] = name
    return result


def write_rotated(source, dest, sector):
    """根元を原点に LOD 全段の位置・法線・接線を一緒に回す。"""
    alpha = np.deg2rad(sector * 22.5)
    c, s = np.cos(alpha), np.sin(alpha)
    matrix = np.array([[c, 0., -s], [0., 1., 0.], [s, 0., c]])
    ini = ConfigParser()
    ini.optionxform = str
    with ZipFile(source) as z:
        ini.read_string(z.read("tree.ini").decode())
        meshes = {}
        for name in (v["MODEL"] for key, v in ini.items() if key.startswith("LOD_")):
            tex, parts = csptree.read_mesh(z.read(name))
            for p in parts:
                for field in ("p", "n"):
                    p.verts[field] = p.verts[field] @ matrix
                tangent = p.verts["b0"][:, :3].astype(float) / 127.5 - 1
                p.verts["b0"][:, :3] = np.rint(np.clip(tangent @ matrix, -1, 1) * 127.5 + 127.5)
            meshes[name] = csptree.write_mesh(tex, parts)
        textures = {n: z.read(n) for n in z.namelist() if n.lower().endswith(".dds")}
    csptree.write_package(dest, {key: dict(ini[key]) for key in ini.sections()}, meshes, textures)


def touched_tiles(before, old, new, added):
    """変更行があるタイルの期待するバイト列。元の改行や行の順番を保つ。"""
    groups = defaultdict(list)
    for i in np.flatnonzero(old.pattern != new.pattern):
        groups[add_extra_trees.tile_name(old.x[i], old.y[i])].append(i)
    additions = defaultdict(list)
    for row in added:
        additions[add_extra_trees.tile_name(row.x, row.y)].append(_line(row))
    baseline = add_extra_trees.list_files(before)
    for name in sorted(groups.keys() | additions.keys()):
        path = baseline.get(name)
        if path:
            tx, ty = map(int, Path(name).stem.split("_")[1:])
            i = np.flatnonzero((tx <= old.x/1000) & (old.x/1000 < tx+1) & (ty <= old.y/1000) & (old.y/1000 < ty+1))
            replacements = {int(j): _line(new[k]) for k,j in zip(groups[name], np.searchsorted(i, groups[name]))}
            result = _patch_tile(path, old[i], replacements, set()) or path.read_bytes()
        else:
            result = TILE_HEADER
        newline = b"\r\n" if b"\r\n" in result else b"\n"
        if not result.endswith(b"\n"):
            if additions[name]:
                raise ValueError(f"{name}: 末尾改行がなく元の行を保って追加できません")
        yield name, result + b"".join(line + newline for line in additions[name])


def count_species(rows):
    return dict(sorted(Counter(map(str, rows.pattern)).items()))


def generate(old, road, ground, fm, exclusion, models, manual, spec, *, grid=None):
    cfg = spec["roadside_cover"]
    if grid is None:
        grid = road.grid(cfg["max_edge_m"])
    return roadside_cover.plant(grid, old, road=road, ground=ground, fm=fm, exclusion=exclusion,
                                models=models, species=treemodels.SPECIES, manual_xy=manual, cfg=cfg,
                                place_cfg=spec.get("place", {}), sight=spec.get("sight", {}))


def close_trial_boundary(cells, old, additions, road, ground, manual, models, cfg):
    """全線から区間だけ切り出した際、境界の外のシダで埋まっていた格子を補う。"""
    visible = np.concatenate((old, additions)).view(np.recarray)
    is_ground = forest_place._under(visible.pattern) | np.char.startswith(visible.pattern, "mt_shrub")
    low = visible[is_ground]
    target = cells[roadside_cover.bare(cells, low, cfg["cover_radius_m"])]
    if not len(target):
        return additions
    possible = cKDTree(target).query(cells)[0] <= cfg["cover_radius_m"]
    possible &= cKDTree(manual).query(cells)[0] >= cfg["manual_clear_m"] + .015
    possible &= roadside_cover.bare(cells, visible, .15)
    p = cells[possible]
    chosen = roadside_cover._cover_plants(target, p, low, cfg)
    xy = p[chosen]
    rng = np.random.default_rng(cfg["seed"] + 1)
    new = roadside_cover._rows(xy, ground.z(xy), ["fern"]*len(xy), treemodels.SPECIES, rng, cfg["ground_size"], cfg["ground_height_m"], models)
    roadside_cover.limit_ground_crown(new, road, models)
    return np.concatenate((additions, new)).view(np.recarray)


def write_snapshot(before, out, old, new, additions, packages=()):
    """元と同じ一覧を揃え、送り出し先を変更のあるタイルだけから組み立てる。"""
    tree_dir = out / "extension/trees"
    tree_dir.mkdir(parents=True, exist_ok=True)
    edits = dict(touched_tiles(before, old, new, additions))
    originals = add_extra_trees.list_files(before)
    new_tiles = sorted(edits.keys() - originals.keys())
    delta = []
    if new_tiles:
        ini_path = before / "extension/parts/forest.ini"
        if not ini_path.is_file():
            ini_path = VALIDATION_FOREST_INI
        text = ini_path.read_text()
        start = max(map(int, re.findall(r"^LIST_(\d+)=", text, re.M)), default=-1)+1
        delta = [f"LIST_{i}=trees\\{name}" for i,name in enumerate(new_tiles, start)]
        deploy_path = out / "trees_config_delta.txt"
        deploy_path.write_text("\n".join(delta)+"\n")
    for name, path in originals.items():
        if name not in edits:
            shutil.copyfile(path, tree_dir / name)
    for name, data in edits.items():
        (tree_dir / name).write_bytes(data)
    # 描画側が全行を読めるよう、完全な配列を作ってから交換する。
    temporary = out / "trees_writing.npy"
    np.save(temporary, np.concatenate((new, additions)))
    temporary.replace(out / "trees.npy")
    deploy = [f"extension/trees/{name}" for name in edits] + [f"extension/trees/{n}.zip" for n in packages]
    return {"changed_tiles": sorted(edits), "new_tiles": new_tiles, "deployment": deploy}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--before", type=Path, default=CURRENT)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--force", action="store_true", help="既存の出力先への書き込みを許可")
    modes = p.add_mutually_exclusive_group()
    modes.add_argument("--estimate-only", action="store_true")
    modes.add_argument("--all-additions", action="store_true",
                       help="既存の型を使って、全線の追加分を作業用の写しに出力する")
    p.add_argument("--packages", type=Path, default=PACKAGES, help="既存の木の ZIP のディレクトリ")
    args = p.parse_args()
    if args.all_additions and args.out.resolve() == OUTPUT.resolve():
        p.error("全線の出力には試作品と別の --out を指定してください")
    protected = (args.before, config.PROJECT_ROOT / "export/forest", WORK / "trees_54",
                 WORK / "supervisor_41.fmEVx6")
    if any(args.out.resolve().is_relative_to(p.resolve()) for p in protected):
        p.error("出力先は入力先と別のディレクトリにしてください")
    args.out = config.assert_writable(args.out)
    try:
        args.out.mkdir(parents=True, exist_ok=args.force)
    except FileExistsError:
        p.error(f"{args.out} は既にあります。上書きするには --force を指定してください")
    section, spec, bounds = inputs()
    cfg = spec["roadside_cover"]
    road = roadside_cover.Road(section)
    manifest = config.DATA_PROCESSED / "track"
    old = np.load(args.before / "trees.npy", mmap_mode="r").view(np.recarray)
    print("1 m 格子", flush=True)
    fm = forestmap.ForestMap.load(config.DATA_PROCESSED / "forest_map.npz")
    grid = road.grid(cfg["max_edge_m"])
    print("地物の除外範囲", flush=True)
    exclusion, rails = barriers(road, spec, manifest)
    print("地面を読込", flush=True)
    ground = ground_from_manifest(manifest)
    manual = manual_stems(Section.load_json(config.DATA_PROCESSED / "track_section.json"), bounds)
    vicinity = max(35., cfg["max_edge_m"] + cfg["forest_neighbor_m"])
    close = near_road(old, road, vicinity)
    print(f"既存 {len(old):,} 行、近傍 {len(close):,} 行、格子 {len(grid):,} 点", flush=True)
    models = package_models(args.packages)
    added = generate(old[close], road, ground, fm, exclusion, models, manual, spec, grid=grid)
    print(f"全線への追加見込み {len(added):,} 行", flush=True)
    valid, d, _ = roadside_cover.eligible(grid, road, fm, ground, exclusion, cfg, minimum=0, maximum=cfg["max_edge_m"])
    station = road.locate(grid).s
    combined = np.concatenate((old[close], added)).view(np.recarray)
    bare0 = roadside_cover.bare(grid, old[close], cfg["cover_radius_m"])
    bare1 = roadside_cover.bare(grid, combined, cfg["cover_radius_m"])
    under = roadside_cover.genus(combined.pattern)
    ground_bare = roadside_cover.bare(grid, combined[np.isin(under, ("fern", "grass", "shrub"))], cfg["cover_radius_m"])
    estimate = {}
    for section_name, (a,b) in bounds.items():
        cell = valid & (d <= 10) & (station >= a) & (station < b)
        estimate[section_name] = {"cells": int(cell.sum()), "bare_before": int((cell & bare0).sum()),
                                  "bare_after_full_estimate": int((cell & bare1).sum())}
    span = tuple(bounds["s4"][0] + x for x in PILOT)
    pilot_grid = valid & (d >= cfg["min_edge_m"]) & (station >= span[0]) & (station <= span[1])
    add_s = road.locate(np.c_[added.x, added.y]).s
    pilot = added[(add_s >= span[0]) & (add_s <= span[1])]
    notes = {"base_rows": len(old), "estimate_total": len(added), "estimate_by_species": count_species(added),
             "sections": estimate, "pilot_total": len(pilot), "pilot_by_legacy_species": count_species(pilot),
             "pilot_cells": int(pilot_grid.sum()), "pilot_bare_before": int((pilot_grid & bare0).sum()),
             "full_eligible_cells": int((valid & (d >= cfg["min_edge_m"])).sum()),
             "full_ground_uncovered": int((valid & (d >= cfg["min_edge_m"]) & ground_bare).sum())}
    if args.estimate_only:
        (args.out / "estimate_only.json").write_text(json.dumps(notes, ensure_ascii=False, indent=2)+"\n")
        return
    if args.all_additions:
        notes.update(write_snapshot(args.before, args.out, old, old, added))
        (args.out / "estimate.json").write_text(json.dumps(notes, ensure_ascii=False, indent=2)+"\n")
        return
    print("試験区間のモデル差し替え", flush=True)
    tree_dir = args.out / "extension/trees"
    trial = package_models(tree_dir)
    if len([n for n in trial if re.fullmatch(r"mt_.*_[ab]", n)]) != 12:
        raise ValueError("先に試作の基本12 ZIPを書き出してください")
    new = old.astype([(n, "U32" if n == "pattern" else old.dtype[n]) for n in old.dtype.names]).view(np.recarray)
    # 山の遠景を改名しないよう、試作で目視できる幅に絞る。
    route = road.xy[(road.s >= span[0]-1) & (road.s <= span[1]+1)]
    lo, hi = route.min(axis=0)-PILOT_MODEL_REACH_M, route.max(axis=0)+PILOT_MODEL_REACH_M
    xy = np.c_[old.x, old.y]
    local = np.flatnonzero(np.all((xy >= lo) & (xy <= hi), axis=1))
    candidates = local[road.interval_mask(xy[local], span)]
    pos = road.locate(np.c_[old.x[candidates], old.y[candidates]])
    selected = candidates[np.linalg.norm(pos.towards, axis=1) <= PILOT_MODEL_REACH_M]
    pos = road.locate(np.c_[old.x[selected], old.y[selected]])
    switched = pilot_patterns(old[selected], pos.towards, rotate_by_name=True, manual_xy=manual)
    projected = switched.copy()
    projected.pattern = [re.sub(r"_r\d\d$", "", n) for n in projected.pattern]
    possible = np.flatnonzero(switched.pattern != old.pattern[selected])
    # 大きさの設定は元の行を維持するので、近い木は試作の実寸でも再点検する。
    nearby = possible[np.linalg.norm(pos.towards[possible], axis=1) < 35]
    clear = roadside_cover.safe_trees(projected[nearby], road, trial, cfg, spec["sight"])
    refused = nearby[~clear]
    switched.pattern[refused] = old.pattern[selected[refused]]
    new.pattern[selected] = switched.pattern
    pos_add = road.locate(np.c_[pilot.x, pilot.y])
    trial_add = pilot_patterns(pilot, pos_add.towards, rotate_by_name=False)
    accepted = roadside_cover.safe_trees(trial_add, road, {**models, **trial}, cfg, spec["sight"])
    pilot = trial_add[accepted]
    by_rail = np.zeros(len(pilot), bool)
    by_rail[rails.query(shapely.points(pilot.x, pilot.y), predicate="dwithin", distance=cfg["rail_clear_m"] + 1.4)[0]] = True
    accents = []
    for i in np.flatnonzero(by_rail & np.char.startswith(pilot.pattern, "shrub_")):
        if all(np.hypot(pilot.x[i]-pilot.x[j], pilot.y[i]-pilot.y[j]) >= 14 for j in accents):
            accents.append(i)
    for i in accents:
        variant = pilot.pattern[i][-1]
        pilot.pattern[i] = f"mt_shrub_{variant}"
        pilot.scale[i] = min(pilot.scale[i], .95 / trial[pilot.pattern[i]].height)

    all_models = {**models, **trial}
    light = forest_place._under(pilot.pattern) | np.char.startswith(pilot.pattern, "mt_shrub")
    low_rows = pilot[light]
    roadside_cover.limit_ground_crown(low_rows, road, all_models)
    pilot[light] = low_rows
    pilot = close_trial_boundary(grid[pilot_grid], old[close], pilot, road, ground, manual, all_models, cfg)

    changed_names = np.unique(new.pattern[selected][new.pattern[selected] != old.pattern[selected]])
    for name in changed_names:
        match = re.fullmatch(r"(.*)_r(\d\d)", name)
        if match:
            base, sector = match.groups()
            write_rotated(tree_dir / f"{base}.zip", tree_dir / f"{name}.zip", int(sector))
    packages = sorted(set(changed_names) | {s for s in pilot.pattern if s.startswith("mt_")})
    notes.update(write_snapshot(args.before, args.out, old, new, pilot, packages))
    pilot_xy = np.c_[pilot.x, pilot.y]
    visible = np.concatenate((old[close], pilot)).view(np.recarray)
    pilot_cells = grid[pilot_grid]
    exposed = roadside_cover.bare(pilot_cells, visible, cfg["cover_radius_m"])
    low = visible[np.char.startswith(visible.pattern, "mt_shrub") | forest_place._under(visible.pattern)]
    ground_gap = roadside_cover.bare(pilot_cells, low, cfg["cover_radius_m"])
    notes.update(pilot_bare_after=int(exposed.sum()), pilot_ground_uncovered=int(ground_gap.sum()),
                 pilot_minimum_edge_m=float(road.edge_distance(pilot_xy).min()),
                 pilot_minimum_manual_distance_m=float(cKDTree(manual).query(np.round(pilot_xy, 2))[0].min()))
    notes.update(pilot_final=len(pilot), pilot_final_by_species=count_species(pilot),
                 pilot_dropped_model_clearance=int((~accepted).sum()),
                 replacements_kept_for_clearance=len(refused),
                 manual_models_preserved=int(manual_model_mask(old[selected], manual).sum()), accents=len(accents),
                 changed_existing=int(np.count_nonzero(new.pattern[selected] != old.pattern[selected])),
                 renamed_by_species=count_species(new[selected][new.pattern[selected] != old.pattern[selected]]))
    temp = args.out / "estimate_writing.json"
    temp.write_text(json.dumps(notes, ensure_ascii=False, indent=2)+"\n")
    temp.replace(args.out / "estimate.json")
    print(f"試験の追加 {len(pilot):,}、既存の名前変更 {notes['changed_existing']:,}、タイル {len(notes['changed_tiles'])}", flush=True)


if __name__ == "__main__":
    main()
