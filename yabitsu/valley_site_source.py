"""橋まわりの Blender 作業用データを、現行の描画部品から切り出す。"""

from dataclasses import dataclass
from hashlib import file_digest
import json
from pathlib import Path

import numpy as np
import shapely
from scipy.spatial import cKDTree

from yabitsu import features, kn5, small_bridges, terrain, track, valleys, valley_surface
from yabitsu.valley_fit import Surfaces
from yabitsu.crosssection import GUTTER_WIDTH, Section


# B の二つの谷筋は、一つの橋の作業範囲を共有する。
from .valley_site_light import SITE_LIGHT

SITES = {"A-s4-1724": 2, "B-s4-6354": 16, "C-s5-8956": 25}
PHOTO_POINTS = {"A": {"0366"}, "B": {"0551", "0552"}, "C": {"0912", "0913"}}
GROUND_PREFIXES = ("1GRASS_valley_", "1GRASS_s1_terrain_", "mtsplit_s1_terrain_")
# 2mの投影格子に入れた袖壁等は、近傍の細分頂点と同じXYに乗るとは限らない。
FOOT_MARGIN = 0.40


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def distance_xy(vertices, origin):
    return np.hypot(vertices[..., 0] - origin[0], vertices[..., 1] - origin[1])


def triangles_and_ids(faces):
    """分割後の三角形と元の manifest の面番号を並べて返す。"""
    tris = kn5._triangles(faces)
    order = np.tile(np.arange(len(faces)), faces.shape[1] - 2)
    if len(tris) != len(order):
        raw = np.concatenate([faces[:, [0, i, i + 1]] for i in range(1, faces.shape[1] - 1)])
        order = order[(raw != np.roll(raw, 1, axis=1)).all(axis=1)]
    return tris, order


@dataclass
class VisibleMesh:
    name: str
    root: str
    path: Path
    material: str
    spec: dict
    vertices: np.ndarray
    faces: np.ndarray
    uv: np.ndarray
    normals: np.ndarray
    triangles: np.ndarray
    triangle_ids: np.ndarray
    render: np.ndarray

    @property
    def world_normals(self):
        if len(self.normals):
            return self.normals @ kn5.C
        return kn5.vertex_normals({}, self.vertices @ kn5.C.T, self.triangles) @ kn5.C


def visible_meshes(processed, origin, radius):
    dress = processed / "dress_mountain"
    hidden = set(read_json(dress / "hidden.json"))
    retex = read_json(dress / "retex.json")
    for root in (processed / "track", dress):
        manifest = read_json(root / "manifest.json")
        for name, entry in manifest["meshes"].items():
            if name in hidden or not entry["renderable"]:
                continue
            path = root / entry["file"]
            with np.load(path, allow_pickle=False) as source:
                vertices = source["verts"]
                lower, upper = vertices[:, :2].min(axis=0), vertices[:, :2].max(axis=0)
                if np.linalg.norm(np.clip(origin[:2], lower, upper) - origin[:2]) > radius:
                    continue
                faces, uv = source["faces"], source["uvs"]
                normals = source["normals"] if "normals" in source else np.empty((0, 3))
            tris, tri_ids = triangles_and_ids(faces)
            keep = distance_xy(vertices[tris].mean(axis=1), origin) <= radius
            if not keep.any():
                continue
            mat = entry["material"]
            spec = dict(manifest["materials"][mat])
            spec["name"] = mat
            if spec["texture"]:
                spec["texture"] = str((root / spec["texture"]).resolve())
            if name in retex:
                key = retex[name]
                spec["texture"] = str((dress / "extension" / f"dress_{key}.dds").resolve())
                spec["name"] = "retex:" + key
            yield VisibleMesh(name, root.name, path, mat, spec, vertices, faces, uv, normals,
                              tris, tri_ids, keep)


def terrain_selection(near, origin, radius):
    vertices, faces = near["valley_vertices"], near["valley_faces"]
    face_ids = np.flatnonzero((distance_xy(vertices[faces], origin) <= radius).all(axis=1))
    if not len(face_ids):
        raise ValueError("半径内に谷の細分面がありません")
    picked = faces[face_ids]
    vertex_ids, inverse = np.unique(picked, return_inverse=True)
    local = inverse.reshape(picked.shape)
    links = np.sort(local[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1)
    edges, count = np.unique(links, axis=0, return_counts=True)
    n = len(faces)
    coarse = near.get("valley_coarse_remainder", np.full((n, 3), -1))[face_ids]
    bed = near["valley_bed"][face_ids]
    gravel = near.get("valley_gravel", np.zeros(n, bool))[face_ids]
    return {
        "terrain_vertices": vertices[vertex_ids],
        "terrain_faces": local,
        "terrain_face_ids": face_ids,
        "terrain_vertex_ids": vertex_ids,
        "terrain_source_faces": near.get("valley_source_faces", np.arange(n))[face_ids],
        "terrain_coarse_remainder": coarse,
        "terrain_face_kind": np.where(gravel, 3, np.where(bed, 1, 0)),
        "lock": np.unique(vertex_ids[np.r_[edges[count == 1].ravel(), local[coarse[:, 0] >= 0].ravel()]]),
    }


def editable_part(mesh, bridge, site):
    if mesh.root != "track":
        return False
    if mesh.name.startswith(f"smallspan_{bridge:02d}_"):
        return True
    stream = {"A": "022", "B": "012"}.get(site[0])
    return stream is not None and mesh.name.startswith(
        (f"water_{stream}_valley_", f"riverbed_{stream}_valley_")
    )


def terrain_attributes(selected, meshes):
    """表示中の三角形に合わせ、元のUV・法線・材質をその角順に並べる。"""
    vertices, faces = selected["terrain_vertices"], selected["terrain_faces"]
    vertex_index = cKDTree(vertices)
    lookup = {tuple(tri): i for i, tri in enumerate(np.sort(faces, axis=1))}
    uv = np.empty((*faces.shape, 2), np.float32)
    normals = np.empty((*faces.shape, 3), np.float64)
    materials = [None] * len(faces)
    visible = [None] * len(faces)
    for mesh in meshes:
        if not mesh.name.startswith(GROUND_PREFIXES):
            continue
        # Bounds are checked per corner so a different vertical layer is never mistaken for this ground.
        dist, index = vertex_index.query(mesh.vertices, distance_upper_bound=1.e-7)
        mtri = mesh.triangles
        matched = np.flatnonzero(mesh.render & np.isfinite(dist[mtri]).all(axis=1))
        local_ids = np.array([lookup.get(tuple(key), -1) for key in
                              np.sort(index[mtri[matched]], axis=1)], dtype=int)
        use = local_ids >= 0
        ii, jj = matched[use], local_ids[use]
        if not len(ii):
            continue
        if any(visible[i] is not None for i in jj):
            raise ValueError(f"谷の表示面が重複しています: {mesh.name}")
        xyz = index[mtri[ii]]
        order = (faces[jj, :, None] == xyz[:, None, :]).argmax(axis=2)
        aligned = np.take_along_axis(mtri[ii], order, axis=1)
        uv[jj] = mesh.uv[aligned]
        normals[jj] = (mesh.normals if len(mesh.normals) else mesh.world_normals @ kn5.C.T)[aligned]
        for j in jj:
            materials[j] = mesh.material
            visible[j] = mesh.spec
        mesh.render[ii] = False
    missing = np.array([material is None for material in materials])
    if missing.any():
        raise ValueError(f"近景と表示部品で谷の三角形が合いません: {missing.sum()} 面"
                         f" (元の面番号 {selected['terrain_face_ids'][missing][:8].tolist()})")
    selected["terrain_uv"] = uv
    selected["terrain_normals"] = normals
    # 山の貼り替えが当たる残片も、元の材質名で組み立てられるようにする。
    selected["terrain_materials"] = np.asarray(materials)
    return visible


def keep_height(sec, meshes, site, bounds, span, selected, near):
    v = selected["terrain_vertices"]
    frame = terrain.road_frame(sec, *v[:, :2].T)
    at = sec.s[frame["idx"]] + frame["along"]
    lateral = frame["lat"]
    fs = features.load_features_rows(track.load_section_rows("features", bounds))
    walks = features.sidewalk_offsets(sec, fs)
    left_edge = np.interp(at, sec.s, sec.hw_left)
    right_edge = np.interp(at, sec.s, sec.hw_right)
    width = np.where(lateral >= 0, left_edge, right_edge)
    walk = np.where(lateral >= 0, np.interp(at, sec.s, walks[0]), np.interp(at, sec.s, walks[1]))
    distance_to_edge = abs(lateral) - width
    surface = Surfaces(near["valley_vertices"][near["valley_faces"]])

    def ground_z(x, y):
        fine = surface.height(np.c_[x, y])
        return np.where(np.isfinite(fine), fine, valley_surface.sample_grid(near, x, y))
    paved_edge = (distance_to_edge >= -.05) & (
        distance_to_edge <= GUTTER_WIDTH + walk + valleys.SHOULDER_M
    )
    i, j = span
    # 床版の橋の開口には道路縁の接地帯がない。
    paved_edge &= (at <= sec.s[i] + .05) | (at >= sec.s[j] - .05)
    road_rows = (np.abs(at - sec.s[i]) <= .45) | (np.abs(at - sec.s[j]) <= .45)
    abutment = road_rows & (distance_to_edge <= GUTTER_WIDTH + .08)
    masks = [(paved_edge, "舗装縁から側溝・歩道・路肩端まで"), (abutment, "橋台の横断面")]
    xy = shapely.points(v[:, :2])
    for prefixes, reason in (
        ((f"smallspan_{SITES[site]:02d}_approach_", f"smallspan_{SITES[site]:02d}_old_road_abutment"),
         "袖壁・旧道橋台の足元"),
        (("smallspan_16_rail_support_", "1WALL_s1_retaining_", "mt_wall_"), "護岸・擁壁の足元"),
        (("GUARDRAIL_", "mt_rail_", "mt_wbeam_"), "ガードレールの足元"),
    ):
        foot = np.zeros(len(v), bool)
        for mesh in meshes:
            if not mesh.name.startswith(prefixes):
                continue
            verts = mesh.vertices[np.unique(mesh.triangles[mesh.render])]
            if not len(verts):
                continue
            ground = ground_z(*verts[:, :2].T)
            anchor = verts[(verts[:, 2] <= ground + .16) & np.isfinite(ground)]
            if len(anchor):
                # 点を結ぶのは、元メッシュの辺のみ。離れた支柱同士の間は土を仕上げられる。
                _, index = np.unique(mesh.triangles[mesh.render], return_inverse=True)
                is_foot = (verts[:, 2] <= ground + .16) & np.isfinite(ground)
                tris = index.reshape(-1, 3)
                edges = np.unique(np.sort(tris[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1), axis=0)
                edges = edges[is_foot[edges].all(axis=1)]
                line = shapely.linestrings(verts[edges, :2])
                near_foot = shapely.union_all(np.r_[shapely.points(anchor[:, :2]), line]).buffer(FOOT_MARGIN)
                foot |= shapely.intersects(near_foot, xy)
        masks.append((foot, reason))
    reasons = {}
    for mask, reason in masks:
        for vertex_id in selected["terrain_vertex_ids"][mask]:
            key = str(vertex_id)
            reasons[key] = reasons[key] + "・" + reason if key in reasons else reason
    # 区分は地形上の役割。見える元材質は別配列に保持する。
    in_shoulder = paved_edge[selected["terrain_faces"]].all(axis=1)
    selected["terrain_face_kind"][in_shoulder & (selected["terrain_face_kind"] == 0)] = 2
    return reasons


def reference_views(route, site):
    views = read_json(route / "work" / "blender_prep_tools" / "views41.json")
    photos = route.parent / "route70_inventory_20261030" / "images" / "sv"
    captures = route / "game_valleys_valleys38_11200741"
    label = site[0]
    result = []
    for name, (position, direction, fov) in views.items():
        photo = None
        if name.startswith("r70_"):
            if name.split("_")[1] not in PHOTO_POINTS[label]:
                continue
            photo = photos / (name.replace("r70_", "R70-", 1) + ".jpg")
            if not photo.is_file():
                raise ValueError(f"視点 {name} の写真がありません: {photo}")
        elif not name.startswith((f"val_{label}_", f"val38_{label}_")):
            continue
        game = captures / f"cap_{name}.png"
        if not game.is_file():
            raise ValueError(f"視点 {name} のゲーム撮影がありません: {game}")
        result.append(dict(name=name, camera=name, position=position, direction=direction, fov=fov,
                           game=str(game.resolve()), photo=str(photo.resolve()) if photo else None))
    if not result:
        raise ValueError(f"視点がありません: {site}")
    return result


def reference_trees(path, origin, radius):
    t = np.load(path, mmap_mode="r", allow_pickle=False)
    keep = np.hypot(t["x"] - origin[0], t["y"] - origin[1]) <= radius
    t = t[keep]
    return [dict(loc=[x - origin[0], y - origin[1], z - origin[2]], pattern=pattern,
                 scale=float(scale), angle=float(angle))
            for x, y, z, pattern, angle, scale in zip(t["x"], t["y"], t["z"], t["pattern"],
                                                      t["angle_deg"], t["scale"])]


def write_bundle(directory, processed, manual, route, forest, site, radius):
    from yabitsu.valley_edits import surface_digest

    valley_file = manual / "valleys.json"
    rows = [row for row in valleys.load(valley_file)
            if row.id == site or row.id.startswith(site + "-")]
    if not rows:
        raise ValueError(f"{valley_file} に {site} の行がありません")
    section_path = processed / "track_section.json"
    sec = Section.load_json(section_path)
    bounds = track.load_bounds(processed)
    bridge_file = manual / "s45_route70_bridges.json"
    spec = read_json(bridge_file)
    bridge = SITES[site]
    definition = spec["bridges"][bridge]
    if definition["section"] != rows[0].section:
        raise ValueError(f"{site} の橋と谷筋の区間が一致しません")
    station = bounds[rows[0].section][0] + (definition["s_from"] + definition["s_to"]) / 2
    origin = np.array([np.interp(station, sec.s, value) for value in (sec.x, sec.y, sec.z)])
    near_path = processed / "track_terrain_near.npz"
    with np.load(near_path, allow_pickle=False) as data:
        near = {k: data[k] for k in data.files if k.startswith("valley_") or k in {"xs", "ys", "Z"}}
    meshes = list(visible_meshes(processed, origin, radius))
    selected = terrain_selection(near, origin, radius)
    display_specs = terrain_attributes(selected, meshes)
    keep_z = keep_height(sec, meshes, site, bounds, small_bridges.spans(sec, bounds, spec)[bridge],
                         selected, near)

    sources = {k: v for k, v in selected.items() if k.startswith("terrain_")}
    geometry = {
        "terrain_vertices": selected["terrain_vertices"] - origin,
        "terrain_faces": selected["terrain_faces"],
        "terrain_uv": selected["terrain_uv"],
        "terrain_normals": selected["terrain_normals"] @ kn5.C,
        "terrain_lock": np.flatnonzero(np.isin(selected["terrain_vertex_ids"], selected["lock"])),
        "terrain_keep_z": np.flatnonzero(np.isin(selected["terrain_vertex_ids"], [int(k) for k in keep_z])),
    }
    materials = {}
    for display in display_specs:
        materials[display["name"]] = display
    geometry["terrain_materials"] = np.array([display["name"] for display in display_specs])
    objects, parts = [], []
    fingerprint = {
        "terrain": surface_digest(near),
        "valleys": sha256(valley_file),
        "parts": {},
        "processed": {str(path.relative_to(processed)): sha256(path) for path in (
            near_path, section_path, processed / "track_profile.npz", processed / "track" / "manifest.json",
            processed / "dress_mountain" / "manifest.json", processed / "dress_mountain" / "retex.json",
            processed / "dress_mountain" / "hidden.json",
        )},
    }
    for mesh in meshes:
        if not mesh.render.any():
            continue
        edit = editable_part(mesh, bridge, site)
        name = mesh.name
        mat = mesh.spec["name"]
        collection = "EDIT_parts" if edit else ("REF_trees" if name.startswith("TREES_") else "LOCK_context")
        key = f"object_{len(objects)}"
        if edit:
            src_key = f"part_{len(parts)}"
            fingerprint["parts"][name] = sha256(mesh.path)
            sources.update({src_key + suffix: values for suffix, values in (
                ("_vertices", mesh.vertices), ("_faces", mesh.faces), ("_uv", mesh.uv),
                ("_normals", mesh.normals),
            )})
            outside = distance_xy(mesh.vertices, origin) > radius
            boundary_faces = outside[mesh.faces].any(axis=1)
            locked = np.unique(mesh.faces[boundary_faces])
            parts.append(dict(name=name, material=mesh.material, key=src_key, lock=locked.tolist(),
                              protected_faces=np.flatnonzero(boundary_faces).tolist()))
            # Keep manifest vertex order even if a given vertex isn't referenced by a triangle.
            used = np.arange(len(mesh.vertices))
            ts = mesh.triangles
            geometry[key + "_ids"] = used + 1
            geometry[key + "_face_ids"] = mesh.triangle_ids + 1
            geometry[key + "_lock"] = locked
        else:
            used, inverse = np.unique(mesh.triangles[mesh.render], return_inverse=True)
            ts = inverse.reshape(-1, 3)
        geometry[key + "_v"] = mesh.vertices[used] - origin
        geometry[key + "_f"] = ts
        geometry[key + "_uv"] = mesh.uv[used][ts]
        geometry[key + "_n"] = mesh.world_normals[used][ts]
        materials[mat] = mesh.spec
        objects.append(dict(name=name, collection=collection, key=key, material=mat))

    v = selected["terrain_vertices"][:, :2]
    trees = reference_trees(forest, origin, radius)
    views = reference_views(route, site)
    data = {
        "version": 1, "site": site, "origin": origin.tolist(), "radius": radius,
        "terrain_name": "valley_terrain", "terrain_bounds": np.r_[v.min(axis=0), v.max(axis=0)].tolist(),
        "lock": selected["lock"].tolist(), "keep_z": keep_z,
        "terrain_vertices": [[int(i), *xyz] for i, xyz in
                             zip(selected["terrain_vertex_ids"], selected["terrain_vertices"].tolist())],
        "terrain_face_ids": selected["terrain_face_ids"].tolist(),
        "parts": parts, "views": views, "fingerprint": fingerprint,
        "terrain_kinds": {"bank": 0, "bed": 1, "shoulder": 2, "gravel": 3},
        "source_forest": str(forest.resolve()),
        "light": SITE_LIGHT,
        "stats": {"terrain_faces": len(selected["terrain_faces"]), "editable_parts": len(parts),
                  "context_and_card_parts": len(objects) - len(parts), "csp_trees": len(trees),
                  "photos": sum(bool(view["photo"]) for view in views), "cameras": len(views)},
    }
    (directory / "site.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    (directory / "render.json").write_text(json.dumps(dict(materials=materials, objects=objects, trees=trees),
                                                      ensure_ascii=False), encoding="utf-8")
    np.savez_compressed(directory / "source.npz", **sources)
    np.savez_compressed(directory / "render.npz", **geometry)
    return data
