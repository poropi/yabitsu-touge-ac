"""指定した木だけを、CSP の森の作業用コピーへ追記する。"""
import argparse
from collections import Counter
from configparser import ConfigParser, Error as ConfigError
import csv
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import shutil
import tempfile
import warnings
from zipfile import BadZipFile, ZipFile

import numpy as np
import shapely
from scipy.spatial import cKDTree

from scripts.align_s5_8960_trees import _line
from scripts.extra_trees_from_plan import COLUMNS, DEFAULT_CSV
from yabitsu import buildings, config, terrain, track
from yabitsu.crosssection import Section


WORK = config.PROJECT_ROOT / "export/compare/route70/work"
FOREST_41 = WORK / "supervisor_41.fmEVx6/deliver/forest"
OUTPUT = WORK / "forest_53"
BUILDINGS = config.DATA_MANUAL / "track_buildings.json"
CONDITIONAL = "if_building_removed:"


@dataclass
class Patch:
    additions: np.ndarray
    tiles: dict
    counts: Counter
    errors: list
    skipped: int


def list_files(root):
    direct = root / "extension/trees"
    folder = direct if direct.is_dir() else root / "forest_extension/trees"
    return {p.name: p for p in folder.glob("*.txt") if "_before_" not in p.stem}


def tile_name(x, y):
    return f"forest_{math.floor(x / 1000)}_{math.floor(y / 1000)}.txt"


def read_candidates(table, sec, bounds, records):
    candidates, errors, skipped = [], [], 0
    heading = np.unwrap(sec.heading)
    with table.open(encoding="utf-8-sig", newline="") as f:
        rows = csv.reader(f)
        if tuple(next(rows, ())) != COLUMNS:
            raise ValueError(f"{table}: CSV の見出しが一致しません")
        while True:
            number = rows.line_num + 1
            values = next(rows, None)
            if values is None:
                break
            if not values:
                continue
            try:
                if len(values) != len(COLUMNS):
                    raise ValueError("列の数が見出しと一致しません")
                row = dict(zip(COLUMNS, values))
                group = row["group"]
                if group != "always":
                    building_id = group.removeprefix(CONDITIONAL)
                    if not group.startswith(CONDITIONAL) or building_id not in records:
                        raise ValueError(f"group の建物 ID または書式が不明です ({group})")
                    if records[building_id].get("gone") is not True:
                        skipped += 1
                        continue
                if row["section"] not in bounds or row["side"] not in ("L", "R"):
                    raise ValueError("区間または左右（L/R）が不明です")
                station, offset, height = (float(row[key]) for key in ("s_m", "edge_offset_m", "height_m"))
                start, end = bounds[row["section"]]
                if not (math.isfinite(station) and 0 <= station <= end - start):
                    raise ValueError("距離が区間外です")
                if not (math.isfinite(offset) and math.isfinite(height) and height > 0):
                    raise ValueError("離れまたは高さが不正です")
                if offset < 1.5:
                    raise ValueError("舗装・側溝・路肩の範囲です（舗装縁から 1.5 m 未満）")
                s = start + station
                left = row["side"] == "L"
                width = np.interp(s, sec.s, sec.hw_left if left else sec.hw_right)
                direction = np.interp(s, sec.s, heading)
                lateral = (width + offset) * (1 if left else -1)
                x = np.interp(s, sec.s, sec.x) - lateral * math.sin(direction)
                y = np.interp(s, sec.s, sec.y) + lateral * math.cos(direction)
                candidates.append((number, row, x, y, height))
            except ValueError as exc:
                errors.append((number, str(exc)))
    return candidates, errors, skipped


def mesh_kind(name, material):
    name = name.lower()
    if name.startswith("trees_s1_"):
        return "card"
    if material in ("water", "water_fall"):
        return "水面"
    if re.search(r"retaining|abutment|approach_(?:start|end)|revetment|wall_cap|rail_support", name) or material.startswith("mt_wall_"):
        return "擁壁・護岸・橋台・袖壁"
    if "fence" in name or name.startswith("mt_bars") or material in ("railing", "railing_bars", "mt_bars"):
        return "柵"
    if (name.startswith(("guardrail_", "1wall_s1_rail_", "mt_rail_", "1wall_track_smallspan_"))
            or re.search(r"(?:bridge.*_rail_|mt_r70_bridge_\d+_(?:left|right)|old_road_.*rail)", name)):
        return "ガードレール"
    if (re.search(r"(?:^smallspan_\d+_(?:deck|old_road_(?:floor|slab))|^bridge_\d+_girder)", name)
            or name.endswith("_bridge_deck")):
        return "橋"
    if re.search(r"shoulder", name):
        return "路肩"
    if material in ("asphalt", "asphalt_tunnel", "mt_road", "mt_plane", "mt_road_dark"):
        return "舗装"
    if name.startswith(("1road_", "scenery_road_", "town_lot", "mt_road_", "mt_gutter_")):
        return "舗装・側溝"
    return None


def mesh_exclusions(roots, xy):
    """組み立て済みの面を XY へ投影して照合する。鉛直面の投影は線になる。"""
    hits = [{} for _ in xy]
    points = shapely.points(xy)
    for root in roots:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        for name, entry in manifest["meshes"].items():
            kind = mesh_kind(name, entry["material"])
            if kind is None:
                continue
            radius = {"card": 1.0, "ガードレール": 0.5, "柵": 0.5, "舗装": 1.5 - 1e-7}.get(kind, 1e-7)
            with np.load(root / entry["file"]) as part:
                verts = part["verts"][:, :2]
                if not len(verts):
                    continue
                near = np.flatnonzero(((verts.min(axis=0) - radius <= xy) &
                                       (xy <= verts.max(axis=0) + radius)).all(axis=1))
                if not len(near):
                    continue
                if kind == "card":
                    stems = verts.reshape(-1, 8, 2)[:, :2].mean(axis=1)
                    for i in near[cKDTree(stems).query(xy[near])[0] <= 1.0]:
                        hits[i]["既存の木の幹"] = name
                    continue
                faces = part["faces"]
            projected = verts[faces]
            for i in near:
                close = ((projected.min(axis=1) - radius <= xy[i]) &
                         (xy[i] <= projected.max(axis=1) + radius)).all(axis=1)
                shapes = shapely.convex_hull(shapely.polygons(projected[close]))
                if np.any(shapely.dwithin(shapes, points[i], radius)):
                    hits[i][kind] = name
    return hits


def model_scale(species, trees, folder):
    selected = trees.scale[trees.pattern == species]
    package = folder / f"{species}.zip"
    if not len(selected) or not package.is_file():
        raise ValueError(f"既存一覧に木の種類 {species!r} がありません")
    ini = ConfigParser()
    try:
        with ZipFile(package) as zip_file:
            ini.read_string(zip_file.read("tree.ini").decode("utf-8-sig"))
        height = ini.getfloat("BASIC", "HEIGHT")
        if not math.isfinite(height) or height <= 0:
            raise ValueError("HEIGHT は正の数が必要です")
    except (ConfigError, BadZipFile, KeyError, ValueError) as exc:
        raise ValueError(f"{package.name} の基準高を読めません ({exc})") from exc
    return height, float(selected.min()), float(selected.max())


def append_tile(path, trees, additions):
    tx, ty = map(int, path.stem.split("_")[1:])
    old = trees[(tx <= trees.x / 1000) & (trees.x / 1000 < tx + 1)
                & (ty <= trees.y / 1000) & (trees.y / 1000 < ty + 1)]
    original = path.read_bytes()
    lines = original.splitlines(keepends=True)
    actual = [line.rstrip(b"\r\n") for line in lines if line.startswith(b"tree: ")]
    if actual != [_line(t) for t in old]:
        raise ValueError(f"{path.name} の木の行と trees.npy が一致しません")
    newline = b"\r\n" if lines and lines[-1].endswith(b"\r\n") else b"\n"
    separator = newline if original and not original.endswith(b"\n") else b""
    return original + separator + b"".join(_line(t) + newline for t in additions)


def prepare(source, table, processed, building_file, *, section=None, near=None, mid=None, mesh_roots=None):
    """入力と描画データから追植分を計算する。出力はここでは書かない。"""
    trees = np.load(source / "trees.npy", mmap_mode="r").view(np.recarray)
    files = list_files(source)
    if not files:
        raise ValueError(f"CSP の木の一覧がありません ({source})")
    records = {b["id"]: b for b in json.loads(building_file.read_text(encoding="utf-8"))}
    sec = Section.load_json(section or processed / "track_section.json")
    candidates, errors, skipped = read_candidates(table, sec, track.load_bounds(processed), records)
    additions = np.empty(len(candidates), dtype=trees.dtype).view(np.recarray)
    counts = Counter()
    if not candidates:
        return Patch(additions, {}, counts, [f"{n} 行目: {error}" for n, error in sorted(errors)], skipped)

    xy = np.asarray([(x, y) for _, _, x, y, _ in candidates])
    roots = mesh_roots if mesh_roots is not None else (processed / "track", processed / "dress_mountain")
    with (np.load(near or processed / "track_terrain_near.npz") as near_grid,
          np.load(mid or processed / "track_terrain_mid.npz") as mid_grid):
        ground = terrain.terrain_mesh_z(near_grid, mid_grid, drawn_only=True)(*xy.T)
    exclusions = mesh_exclusions(roots, xy)
    close_tree = cKDTree(np.c_[trees.x, trees.y]).query(xy)[0] <= 1.0
    active = [buildings.Building.from_spec(b) for b in records.values() if not b.get("gone")]
    for name in ("s3_buildings.json", "s5_kitahara_buildings.json"):
        path = building_file.parent / name
        if path.is_file():
            active.extend(buildings.load(path))
    footprints = shapely.STRtree([b.footprint for b in active])
    by_building = dict(zip(*footprints.query(shapely.points(xy), predicate="covered_by")))
    species_models, tiles, indices = {}, {}, {}
    for j, (number, row, x, y, height) in enumerate(candidates):
        species = row["species"]
        name = tile_name(x, y)
        try:
            if species not in species_models:
                species_models[species] = model_scale(species, trees, next(iter(files.values())).parent)
            if exclusions[j]:
                raise ValueError("、".join(f"{kind} ({part})" for kind, part in exclusions[j].items()))
            if close_tree[j] or any(math.hypot(x - additions.x[i], y - additions.y[i]) <= 1.0 for i in indices):
                raise ValueError("既存または先行する追加の木の幹から 1 m 以内です")
            if j in by_building:
                raise ValueError(f"建物の輪郭内です ({active[by_building[j]].id})")
            if not math.isfinite(ground[j]):
                raise ValueError("描画地面の高さが取れません")
            if name not in files:
                raise ValueError(f"既存のタイル {name} がありません")
            base, low, high = species_models[species]
            requested = height / base
            scale = min(max(requested, low), high)
            if scale != requested:
                warnings.warn(f"{number} 行目: {species} の倍率 {requested:.3f} を {scale:.3f} に制限します",
                              stacklevel=2)
            seed = sha256(json.dumps(row, sort_keys=True, ensure_ascii=False).encode()).digest()
            angle = int.from_bytes(seed[:8], "big") * 360 / 2**64
            additions[j] = (x, y, ground[j], species, angle, scale)
            indices[j] = (name, number)
            counts[name] += 1
        except ValueError as exc:
            errors.append((number, str(exc)))

    for name in counts.copy():
        ids = [i for i, (tile, _) in indices.items() if tile == name]
        try:
            tiles[name] = append_tile(files[name], trees, additions[ids])
        except ValueError as exc:
            for i in ids:
                errors.append((indices.pop(i)[1], str(exc)))
            del counts[name]
    return Patch(additions[list(indices)], tiles, counts, [f"{n} 行目: {error}" for n, error in sorted(errors)], skipped)


def write_copy(source, output, patch):
    if patch.errors:
        raise ValueError("CSV に植えられない行があります")
    output = output.resolve()
    protected = (source.resolve(), FOREST_41.resolve(), (config.PROJECT_ROOT / "export/forest").resolve())
    if any(output == p or p in output.parents or output in p.parents for p in protected):
        raise ValueError(f"森の入力と重ならない新しい出力先を指定してください ({output})")
    if output.exists():
        raise ValueError(f"出力先が既にあります ({output})")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as tmp:
        stage = Path(tmp) / "forest"
        lists = stage / "extension/trees"
        lists.mkdir(parents=True)
        for name, path in list_files(source).items():
            if name in patch.tiles:
                (lists / name).write_bytes(patch.tiles[name])
            else:
                shutil.copyfile(path, lists / name)
        if len(patch.additions):
            old = np.load(source / "trees.npy", mmap_mode="r")
            merged = np.lib.format.open_memmap(stage / "trees.npy", mode="w+", dtype=old.dtype,
                                               shape=(len(old) + len(patch.additions),))
            merged[:len(old)] = old
            merged[len(old):] = patch.additions
            del merged
        else:
            shutil.copyfile(source / "trees.npy", stage / "trees.npy")
        stage.rename(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--forest-dir", type=Path, default=FOREST_41)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--processed", type=Path, default=config.DATA_PROCESSED)
    parser.add_argument("--buildings", type=Path, default=BUILDINGS)
    args = parser.parse_args()
    try:
        patch = prepare(args.forest_dir, args.csv, args.processed, args.buildings)
        for error in patch.errors:
            print(error)
        print(f"追加候補 {len(patch.additions)} 本、条件待ち {patch.skipped} 本、エラー {len(patch.errors)} 行")
        for name, count in sorted(patch.counts.items()):
            print(f"extension/trees/{name}: +{count} 本")
        if patch.errors:
            return 1
        write_copy(args.forest_dir, args.output, patch)
        print(f"{args.output} に {len(patch.additions)} 本を追加しました")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
