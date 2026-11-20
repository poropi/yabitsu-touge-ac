"""CSP の高木を道ぎわの立体木と遠景の板に置き換える。"""
from collections import Counter, defaultdict
from contextlib import ExitStack
import configparser
from dataclasses import dataclass
import hashlib
import io
import json
import math
import re
from pathlib import Path
import struct
import zipfile

import numpy as np
from PIL import Image
import shapely

from . import config, kn5


KN5_NAME = "yabitsu_trees_light.kn5"
SETTINGS_FILE = config.DATA_MANUAL / "trees_light.json"
CATALOG = config.PROJECT_ROOT / "assets/landmarks/trees_light/export/models.json"
TREE = re.compile(rb"\s*pos=([^;]+);\s*angle\s*=\s*([^;]+);\s*size\s*=\s*([^\s]+)\s*")
DEFAULT_AMBIENT = kn5.MATERIAL_PROPS["ksAmbient"]
DEFAULT_DIFFUSE = kn5.MATERIAL_PROPS["ksDiffuse"]
KN5_UP = (0, 1, 0)
MAX_CSP_TALL = 300


@dataclass(frozen=True)
class Settings:
    enabled: bool
    tall_families: tuple[str, ...]
    near_edge_m: float
    near_yaw_deg: float
    near_scale: tuple[float, float]
    far_edge_m: tuple[float, float]
    far_grid_m: tuple[int, int, int]
    far_height_percentile: float
    far_yaw_deg: float
    far_scale: tuple[float, float]
    far_widest_grid_width_factor: float
    mesh_grid_m: int
    shadows: str
    normals: str
    ambient: float
    diffuse: float
    alpha_ref: float
    texture: str
    mip_alpha: str


def load_settings():
    return Settings(**json.loads(SETTINGS_FILE.read_text(encoding="utf-8")))


def tall_pattern(settings):
    families = b"|".join(re.escape(name.encode()) for name in settings.tall_families)
    return re.compile(rb"tree: ((?:" + families + rb")_[^;]+)\.zip;")


@dataclass
class Trees:
    xyz: np.ndarray
    family: np.ndarray
    height: np.ndarray


@dataclass
class Model:
    height: float
    verts: np.ndarray
    uvs: np.ndarray
    faces: np.ndarray


@dataclass
class Placement:
    model: str
    xyz: np.ndarray
    angle: float  # Z軸周り、ラジアン
    scale: float
    width_factor: float = 1.0


def load_assets(catalog):
    spec = json.loads(catalog.read_text(encoding="utf-8"))
    models = {}
    for info in spec["models"]:
        with np.load(catalog.parent / info["npz"]) as asset:
            verts, uvs, faces = (asset[key] for key in ("verts", "uvs", "faces"))
        if faces.shape != (info["triangles"], 3) or faces.min() < 0 or faces.max() >= len(verts):
            raise ValueError(f"面の仕様が一覧と一致しません: {info['name']}")
        uvs[:, 1] = 1 - uvs[:, 1]
        models[info["name"]] = Model(info["height_m"], verts, uvs, faces)
    material = spec["material"]
    atlases = {variant: catalog.parent / material[key]
               for variant, key in (("middle", "texture"), ("dark", "alternate"))}
    return models, atlases


def road_edges(manifest):
    """舗装の水平投影を結合し、部品の継ぎ目や内部の三角形の辺を除く。"""
    parts = []
    meshes = json.loads(manifest.read_text(encoding="utf-8"))["meshes"]
    for name, info in meshes.items():
        if info["material"] != "asphalt" or not name.startswith("1ROAD") or "side_stub" in name:
            continue
        with np.load(manifest.parent / info["file"]) as mesh:
            triangles = mesh["verts"][:, :2][kn5._triangles(mesh["faces"])]
        polygons = shapely.polygons(triangles)
        parts.append(shapely.union_all(polygons[shapely.area(polygons) > 0]))
    boundary = shapely.boundary(shapely.union_all(parts))
    segments = []
    for line in shapely.get_parts(boundary):
        xy = shapely.get_coordinates(line)
        segments.append(np.stack((xy[:-1], xy[1:]), axis=1))
    if not segments:
        raise ValueError("マニフェストに対象の舗装端がありません")
    return shapely.STRtree(shapely.linestrings(np.concatenate(segments)))


def read_tile(tile, pattern):
    """CSP 一覧の元バイトと、高木の行番号・モデル名・数値を返す。"""
    lines = tile.read_bytes().splitlines(keepends=True)
    indices, models, rows = [], [], []
    for index, line in enumerate(lines):
        tall = pattern.match(line)
        if tall is None:
            continue
        match = TREE.fullmatch(line[tall.end():])
        if match is None:
            raise ValueError(f"{tile}:{index + 1}: 木の行を解釈できません")
        pos, angle, scale = match.groups()
        indices.append(index)
        models.append(tall[1])
        rows.append((*map(float, pos.split(b",")), float(angle), float(scale)))
    return lines, np.asarray(indices, dtype=int), models, np.asarray(rows, dtype=float).reshape(-1, 5)


def near_mask(xy, edges, distance):
    is_near = np.zeros(len(xy), dtype=bool)
    is_near[edges.query(shapely.points(xy), predicate="dwithin", distance=distance)[0]] = True
    return is_near


def keep_other_lines(lines, indices):
    keep = np.ones(len(lines), dtype=bool)
    keep[indices] = False
    return b"".join(line for line, retained in zip(lines, keep) if retained)


def package_height(path):
    ini = configparser.ConfigParser()
    with zipfile.ZipFile(path) as package:
        ini.read_string(package.read("tree.ini").decode("utf-8"))
    return ini["BASIC"].getfloat("HEIGHT")


def read_trees(source, original_models, edges, settings, near_models, *, filtered_lists=None, near_lists=None):
    tiles = sorted(source.glob("forest_*.txt"))
    if not tiles:
        raise ValueError(f"木の一覧がありません: {source}")
    pattern = tall_pattern(settings)
    species = {}
    grouped = [], []
    retained_packages = set()
    retained_heights = Counter()
    filtered_tiles = {}
    for tile in tiles:
        lines, indices, names, rows = read_tile(tile, pattern)
        xyz = rows[:, (0, 2, 1)] * [1, -1, 1]
        if near_lists is None:
            near = near_mask(xyz[:, :2], edges, settings.near_edge_m)
        else:
            remaining = Counter(line for line in (near_lists / tile.name).read_bytes().splitlines()
                                if pattern.match(line))
            near = np.zeros(len(rows), dtype=bool)
            for i, index in enumerate(indices):
                line = lines[index].rstrip(b"\r\n")
                if remaining[line] > 0:
                    remaining[line] -= 1
                    near[i] = True
            if +remaining:
                raise ValueError(f"{tile}: 道ぎわ一覧に元の一覧に無い木があります")
        for name in dict.fromkeys(names):
            if name not in species:
                model = name.decode()
                family = settings.tall_families.index(model.split("_", 1)[0])
                species[name] = (family, package_height(original_models / f"{model}.zip"))
        kind, height = np.asarray([species[name] for name in names], dtype=float).reshape(-1, 2).T
        data = np.column_stack((xyz, kind, height * rows[:, 4]))
        keep_csp = np.zeros(len(rows), dtype=bool)
        for i in np.flatnonzero(near):
            model, _ = near_model(xyz[i], int(kind[i]), settings)
            keep_csp[i] = data[i, 4] < settings.near_scale[0] * near_models[model].height
            if keep_csp[i]:
                retained_packages.add(f"{names[i].decode()}.zip")
                retained_heights[math.floor(data[i, 4])] += 1
        grouped[0].append(data[near & ~keep_csp])
        grouped[1].append(data[~near])
        if filtered_lists is not None:
            filtered_tiles[tile.name] = keep_other_lines(lines, indices[~keep_csp])
    csp_count = sum(retained_heights.values())
    if csp_count > MAX_CSP_TALL:
        distribution = ", ".join(f"{h}〜{h + 1} m未満: {n}本" for h, n in sorted(retained_heights.items()))
        raise ValueError(f"CSP に残る高木が {csp_count:,} 本（上限 {MAX_CSP_TALL} 本）。樹高別: {distribution}")
    for name, contents in filtered_tiles.items():
        (filtered_lists / name).write_bytes(contents)
    near, far = (Trees(rows[:, :3], rows[:, 3].astype(int), rows[:, 4])
                 for rows in map(np.concatenate, grouped))
    return near, far, retained_packages, csp_count


def nearest_edges(xy, edges):
    # 等距離の辺は先頭を選び、STRtreeの列挙順に依存させない。
    (points, segments), distances = edges.query_nearest(shapely.points(xy), return_distance=True)
    nearest = np.full(len(xy), len(edges.geometries), dtype=int)
    np.minimum.at(nearest, points, segments)
    distance = np.full(len(xy), np.inf)
    np.minimum.at(distance, points, distances)
    return nearest, distance


def edge_vectors(xy, edges):
    nearest, _ = nearest_edges(xy, edges)
    lines = edges.geometries[nearest]
    start = shapely.get_coordinates(shapely.get_point(lines, 0))
    end = shapely.get_coordinates(shapely.get_point(lines, -1))
    tangent = end - start
    # 接線の符号を統一して、道路境界の頂点順による向きの反転を防ぐ。
    reverse = (tangent[:, 0] < 0) | ((tangent[:, 0] == 0) & (tangent[:, 1] < 0))
    tangent[reverse] *= -1
    length = np.linalg.norm(tangent, axis=1, keepdims=True)
    tangent = np.divide(tangent, length, out=np.tile([1., 0.], (len(xy), 1)), where=length > 0)
    closest = shapely.get_coordinates(shapely.line_interpolate_point(
        lines, shapely.line_locate_point(lines, shapely.points(xy))
    ))
    return tangent, closest - xy


def location_random(xyz):
    # 座標に固定した乱数。版や入力行の並び順を変えても配置が揃う。
    digest = hashlib.blake2b(struct.pack("<3d", *xyz), digest_size=8).digest()
    a, b = struct.unpack("<2I", digest)
    return ("a" if a < 2**31 else "b"), b / 2**32


def near_model(xyz, family, settings):
    variant, random = location_random(xyz)
    return f"near_{settings.tall_families[family]}_{variant}", random


def place_near(trees, edges, models, settings=None):
    settings = settings or load_settings()
    tangent, toward = edge_vectors(trees.xyz[:, :2], edges)
    on_edge = np.all(toward == 0, axis=1)
    toward[on_edge] = tangent[on_edge][:, [1, 0]] * [-1, 1]
    angles = np.arctan2(toward[:, 1], toward[:, 0])
    placements = []
    for xyz, family, height, angle in zip(trees.xyz, trees.family, trees.height, angles):
        model, random = near_model(xyz, family, settings)
        scale = min(settings.near_scale[1], max(settings.near_scale[0], height / models[model].height))
        placements.append(Placement(model, xyz, angle + math.radians(2 * settings.near_yaw_deg * random - settings.near_yaw_deg), scale))
    return placements


def grid_sizes(xy, edges, settings):
    # 24 m格子の中心距離と点までの距離から上下限を出す。閾値をまたぐ点だけ距離を引き直す。
    inner, middle, outer = settings.far_grid_m
    inner_edge, outer_edge = settings.far_edge_m
    size = outer
    cells, inverse = np.unique(np.floor(xy / size).astype(int), axis=0, return_inverse=True)
    centers = (cells + 0.5) * size
    _, center_distances = nearest_edges(centers, edges)
    distance = center_distances[inverse]
    radius = np.linalg.norm(xy - centers[inverse], axis=1) + 1e-9
    uncertain = ((np.abs(distance - inner_edge) <= radius) | (np.abs(distance - outer_edge) <= radius))
    _, distance[uncertain] = nearest_edges(xy[uncertain], edges)
    return np.where(distance <= inner_edge, inner, np.where(distance <= outer_edge, middle, outer))


def group_percentile(values, inverse, counts, percentile):
    sorted_values = values[np.lexsort((values, inverse))]
    starts = np.cumsum(counts) - counts
    ranks = (counts - 1) * percentile
    lower = starts + np.floor(ranks).astype(int)
    upper = starts + np.ceil(ranks).astype(int)
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * (ranks % 1)


def place_far(trees, edges, models, settings=None):
    settings = settings or load_settings()
    xy = trees.xyz[:, :2]
    sizes = grid_sizes(xy, edges, settings)
    keys = np.column_stack((sizes, np.floor(xy / sizes[:, None]))).astype(int)
    cells, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    centers = np.column_stack([
        np.bincount(inverse, weights=xy[:, axis]) / counts for axis in range(2)
    ] + [group_percentile(trees.xyz[:, 2], inverse, counts, 0.5)])
    height = group_percentile(trees.height, inverse, counts, settings.far_height_percentile)
    # 遠景素材の種類に合わせ、表の順や対象の絞り込みと独立に票を数える。
    families = np.asarray(settings.tall_families)[trees.family]
    votes = np.column_stack([
        np.bincount(inverse, weights=families == family, minlength=len(cells))
        for family in ("sugi", "hinoki", "broadleaf")
    ])
    tangent, _ = edge_vectors(centers[:, :2], edges)
    angle = np.arctan2(tangent[:, 1], tangent[:, 0])
    angle += np.deg2rad(np.where(cells[:, 1:].sum(axis=1) % 2 == 0, settings.far_yaw_deg, -settings.far_yaw_deg))

    placements = []
    for (size, _, _), xyz, (sugi, hinoki, broadleaf), h, yaw in zip(cells, centers, votes, height, angle):
        if broadleaf:
            if sugi or hinoki:
                model = "far_mixed_a"
            else:
                variant, _ = location_random(xyz)
                model = f"far_broadleaf_{variant}"
        elif sugi == hinoki:
            model = "far_conifer_mix"
        else:
            model = "far_sugi_a" if sugi > hinoki else "far_hinoki_a"
        scale = min(settings.far_scale[1], max(settings.far_scale[0], h / models[model].height))
        width = settings.far_widest_grid_width_factor if size == settings.far_grid_m[-1] else 1.
        placements.append(Placement(model, xyz, yaw, scale, width))
    cell_sizes = cells[:, 0]
    counts_by_size = {size: int(np.count_nonzero(cell_sizes == size)) for size in settings.far_grid_m}
    sparse = {n: int(np.count_nonzero((cell_sizes == settings.far_grid_m[0]) & (counts == n))) for n in (1, 2)}
    return placements, counts_by_size, sparse


def mesh_batches(placements, models, settings=None):
    settings = settings or load_settings()
    cells = defaultdict(list)
    for item in placements:
        kind = item.model.split("_", 1)[0]
        x, y = np.floor(item.xyz[:2] / settings.mesh_grid_m).astype(int)
        cells[kind, x, y].append(item)
    batches = []
    for (kind, x, y), items in sorted(cells.items(), key=lambda pair: (pair[0][0] == "far", *pair[0][1:])):
        part, vertices, chunk = 0, 0, []
        for item in items:
            count = len(models[item.model].verts)
            if count > 65535:
                raise ValueError(f"{item.model}: {count:,} 頂点（KN5 の部品の上限は 65,535 頂点）")
            if vertices + count > 65535:
                batches.append((f"treelight_{kind}_{x}_{y}_{part:02d}", chunk))
                part, vertices, chunk = part + 1, 0, []
            chunk.append(item)
            vertices += count
        batches.append((f"treelight_{kind}_{x}_{y}_{part:02d}", chunk))
    return batches


def geometry(items, models):
    positions, uvs, faces = [], [], []
    offset = 0
    for item in items:
        model = models[item.model]
        c, s = math.cos(item.angle), math.sin(item.angle)
        rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        xyz = (model.verts * [item.width_factor, 1, 1]) @ rotation.T * item.scale + item.xyz
        positions.append(xyz)
        uvs.append(model.uvs)
        faces.append(model.faces + offset)
        offset += len(xyz)
    pos = np.concatenate(positions) @ kn5.C.T
    uv = np.concatenate(uvs)
    tris = np.concatenate(faces)
    normals = kn5._normals(pos, tris)
    return pos, normals, uv, kn5._tangents(pos, uv, normals, tris), tris


def casts_shadow(name, shadows):
    return shadows == "all" or (shadows == "near" and name.startswith("treelight_near_"))


def write_kn5s(textures, batches, models, *, shadows="none", normals="mesh",
               ambient=DEFAULT_AMBIENT, diffuse=DEFAULT_DIFFUSE, alpha_ref=kn5.ALPHA_REF):
    with ExitStack() as cleanup, ExitStack() as streams:
        files = []
        for path, (name, atlas) in textures.items():
            temp = path.with_suffix(".kn5.tmp")
            stream = streams.enter_context(temp.open("wb"))
            cleanup.callback(temp.unlink, missing_ok=True)
            writer = kn5._Out(stream)
            writer.raw(b"sc6969")
            writer.pack("iii", kn5.VERSION, 1, 1)
            writer.string(name)
            writer.pack("i", len(atlas))
            writer.raw(atlas)
            writer.pack("i", 1)
            writer.string("yabitsu_treelight_61")
            writer.string("ksPerPixelAT")
            writer.pack("B?i", 0, True, 0)
            props = {**kn5.MATERIAL_PROPS, "ksAmbient": ambient, "ksDiffuse": diffuse,
                     "ksAlphaRef": alpha_ref, "ksSpecular": 0.0}
            writer.pack("i", len(props))
            for key, value in props.items():
                writer.string(key)
                writer.pack("10f", value, *([0.0] * 9))
            writer.pack("i", 1)
            writer.string("txDiffuse")
            writer.pack("i", 0)
            writer.string(name)
            writer.pack("i", kn5.NODE_BASE)
            writer.string("yabitsu_treelight_61")
            writer.pack("i?", len(batches), True)
            writer.pack("16f", *np.eye(4).ravel())
            files.append((path, temp, writer))

        for name, items in batches:
            node = kn5._Out(io.BytesIO())
            pos, nrm, uv, tan, tris = geometry(items, models)
            if normals == "up":
                nrm[:] = KN5_UP
            kn5._write_mesh(node, name, pos, nrm, uv, tan, tris, 0, True,
                            cast_shadows=casts_shadow(name, shadows))
            payload = node.b.getvalue()
            for _, _, writer in files:
                writer.raw(payload)
        streams.close()
        for path, temp, _ in files:
            temp.replace(path)


def verify_kn5(path, texture, batches, models, *, shadows="none", normals="mesh",
               ambient=DEFAULT_AMBIENT, diffuse=DEFAULT_DIFFUSE, alpha_ref=kn5.ALPHA_REF):
    decoded = kn5.read_kn5(path)
    atlas_name, atlas = texture
    mat, = decoded["materials"]
    if (decoded["textures"] != {atlas_name: atlas} or mat["shader"] != "ksPerPixelAT"
            or not mat["alpha_tested"] or mat["blend"] != 0 or mat["depth"] != 0
            or not math.isclose(mat["props"]["ksAlphaRef"], alpha_ref, abs_tol=1e-7)
            or not math.isclose(mat["props"]["ksAmbient"], ambient, rel_tol=1e-7)
            or not math.isclose(mat["props"]["ksDiffuse"], diffuse, rel_tol=1e-7)
            or mat["textures"] != [("txDiffuse", 0, atlas_name)]):
        raise ValueError(f"{path}: 材質・絵が一致しません")
    digest = hashlib.sha256()
    parts, triangles, max_vertices = Counter(), Counter(), Counter()
    for mesh, (name, items) in zip(decoded["root"]["children"], batches, strict=True):
        kind = name.split("_")[1]
        vertices, indices = mesh["vertices"], mesh["indices"]
        if (mesh["name"] != name or mesh["material"] != 0
                or mesh["cast_shadows"] != casts_shadow(name, shadows)
                or not (mesh["active"] and mesh["visible"] and mesh["renderable"])
                or mesh["transparent"] or mesh["lod"] != (0., 0.)
                or len(vertices) != sum(len(models[item.model].verts) for item in items)
                or len(indices) != 3 * sum(len(models[item.model].faces) for item in items)
                or indices.max() >= len(vertices) or len(vertices) > 65535
                or (normals == "up" and not np.all(vertices[:, 3:6] == KN5_UP))):
            raise ValueError(f"{path}: 部品の構成が一致しません: {name}")
        digest.update(vertices)
        digest.update(indices)
        parts[kind] += 1
        triangles[kind] += len(indices) // 3
        max_vertices[kind] = max(max_vertices[kind], len(vertices))
    return {"parts": dict(parts), "triangles": dict(triangles), "max_vertices": dict(max_vertices),
            "geometry_sha256": digest.hexdigest()}


def build(source, original_models, manifest, outputs, settings, *, catalog=CATALOG,
          filtered_lists=None, near_lists=None, verify=False):
    """高木の台帳と舗装から KN5 を書き出す。outputs は版名から出力パスへの対応。"""
    models, atlases = load_assets(catalog)
    edges = road_edges(manifest)
    near, far, retained_packages, csp_count = read_trees(
        source, original_models, edges, settings, models, filtered_lists=filtered_lists, near_lists=near_lists)
    near_items = place_near(near, edges, models, settings)
    far_items, counts, sparse = place_far(far, edges, models, settings)
    batches = mesh_batches(near_items + far_items, models, settings)
    textures = {}
    mip_alpha_ref = settings.alpha_ref if settings.mip_alpha == "coverage" else None
    for variant, path in outputs.items():
        atlas = atlases[variant]
        with Image.open(atlas) as img:
            if img.size != (2048, 2048) or img.mode != "RGBA":
                raise ValueError(f"アトラスの寸法または画像モードが異なります: {atlas}")
            textures[path] = (atlas.stem + ".dds", kn5.dds_bytes(img, alpha_ref=mip_alpha_ref))
    options = dict(shadows=settings.shadows, normals=settings.normals, ambient=settings.ambient,
                   diffuse=settings.diffuse, alpha_ref=settings.alpha_ref)
    write_kn5s(textures, batches, models, **options)

    stats = {"csp_trees": csp_count, "retained_packages": sorted(retained_packages)}
    if verify:
        stats.update(near_trees=len(near_items), far_trees=len(far.xyz), far_plates=counts,
                     sparse_9m_plates=sparse,
                     placements_by_model=dict(sorted(Counter(p.model for p in near_items + far_items).items())),
                     files={})
        for path, texture in textures.items():
            result = verify_kn5(path, texture, batches, models, **options)
            result["bytes"] = path.stat().st_size
            stats["files"][path.name] = result
        if len({result["geometry_sha256"] for result in stats["files"].values()}) != 1:
            raise ValueError("2版のKN5で形状または配置が異なります")
    return stats
