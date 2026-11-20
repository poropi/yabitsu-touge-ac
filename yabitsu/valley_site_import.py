"""Validate Blender edits and restore the original coordinates wherever possible."""

from collections import Counter
import json
from pathlib import Path
import re
import shutil

import numpy as np
from PIL import Image
from scipy.sparse import coo_array, csgraph
import shapely

from . import kn5

KINDS = np.array(["bank", "bed", "shoulder", "gravel"])
SOURCE_FIELDS = {
    "vertices": "vertices", "faces": "faces", "uv": "uv", "materials": "materials",
    "face_kind": "face_kind", "vertex_ids": "vertex_ids", "original_face_ids": "face_ids",
    "source_faces": "source_faces", "coarse_remainder": "coarse_remainder",
}


def _indices(ids, original):
    """Convert stable IDs to array offsets. Blender gives new elements ID -1."""
    order = np.argsort(original)
    slots = np.searchsorted(original[order], ids)
    offsets = order[np.minimum(slots, len(order) - 1)]
    valid = (slots < len(order)) & (ids == original[offsets]) & (ids >= 0)
    if np.any((ids != -1) & ~valid):
        raise ValueError("Unknown original element ID")
    return np.where(valid, offsets, -1)


def _points(source, baseline, current, site, locked, keep_z):
    indices = _indices(current["vertex_ids"], baseline["vertex_ids"])
    known = indices >= 0
    if len(np.unique(indices[known])) != known.sum():
        raise ValueError("Duplicate vertex IDs. Reset copied vertex IDs to 0 in Blender")
    if not np.isfinite(current["vertices"]).all():
        raise ValueError("Non-finite vertices")
    original_offsets = _indices(baseline["vertex_ids"], source["vertex_ids"])
    known_offsets = original_offsets[indices[known]]
    delta = current["vertices"][known].astype(float) - baseline["vertices"][indices[known]]
    vertices = current["vertices"].astype(float) + site["origin"]
    vertices[known] = source["vertices"][known_offsets] + delta
    for ids, message, tolerance in ((locked, "LOCK", 0.001), (keep_z, "KEEP_Z", 0.005)):
        mask = np.isin(current["vertex_ids"][known], ids)
        missing = set(ids) - set(current["vertex_ids"][known])
        displacement = delta[mask]
        movement = np.linalg.norm(displacement, axis=1)
        if message == "KEEP_Z":
            movement = np.maximum(np.linalg.norm(displacement[:, :2], axis=1), abs(displacement[:, 2]))
        if missing or (movement > tolerance).any():
            raise ValueError(f"{message}: missing or displaced vertex (limit {tolerance * 1000:g} mm)")
        slots = np.flatnonzero(known)[mask]
        if message == "LOCK":
            vertices[slots] = source["vertices"][known_offsets[mask]]
        else:
            vertices[slots, 2] = source["vertices"][known_offsets[mask], 2]
    moved = ~known
    moved[known] = np.any(vertices[known] != source["vertices"][known_offsets], axis=1)
    if np.any(np.linalg.norm(vertices[moved, :2] - np.asarray(site["origin"])[:2], axis=1) > site["radius"] + 1e-7):
        raise ValueError("Vertex outside edit scope (radius)")
    return vertices, moved, known, known_offsets


def _edge_counts(faces):
    edges = faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)
    undirected = Counter(map(tuple, np.sort(edges, axis=1)))
    directed = Counter(map(tuple, edges))
    boundary = {edge if edge in directed else edge[::-1]
                for edge, count in undirected.items() if count == 1}
    defective = {
        edge for edge, count in undirected.items()
        if count > 2 or (count == 2 and directed[edge] != 1)
    }
    return boundary, defective


def _duplicate_faces(vertices, faces):
    _, positions = np.unique(np.round(vertices, 7), axis=0, return_inverse=True)
    keys = np.sort(positions[faces], axis=1)
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    return counts[inverse] > 1


def _welded_corner_ids(vertices, faces, ids, new_id_offset):
    """Join the coincident corners that separate a part's UVs or normals."""
    positions, inverse = np.unique(np.round(vertices, 7), axis=0, return_inverse=True)
    priority = np.where(ids >= 0, ids, new_id_offset - ids)
    representative = np.full(len(positions), np.iinfo(np.int64).max, dtype=np.int64)
    np.minimum.at(representative, inverse, priority)
    canonical = np.where(representative < new_id_offset, representative, new_id_offset - representative)
    return canonical[inverse[faces]]


def _check_closed_winding(triangles, corner_ids, new_faces, open_edges):
    ids, local = np.unique(corner_ids, return_inverse=True)
    left, right = local.reshape(-1, 3)[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2).T
    graph = coo_array((np.ones(len(left), bool), (left, right)), shape=(len(ids), len(ids))).tocsr()
    _, components = csgraph.connected_components(graph, directed=False)
    labels = components[np.searchsorted(ids, corner_ids[:, 0])]
    open_vertices = [edge[0] for edge in open_edges]
    open_components = components[np.searchsorted(ids, open_vertices)]
    for label in np.setdiff1d(labels[new_faces], open_components):
        shell = triangles[labels == label]
        shell = shell - shell[0, 0]
        volume = np.einsum("ij,ij->i", shell[:, 0], np.cross(shell[:, 1], shell[:, 2])).sum()
        if volume < -1e-10:
            raise ValueError("Reversed closed part winding")


def _geometry(vertices, original, baseline, current, terrain):
    faces = current["faces"]
    if not len(faces) or np.min(faces) < 0 or np.max(faces) >= len(vertices):
        raise ValueError("Invalid mesh faces")
    uv = current.get("uv")
    if uv is None or uv.shape != (len(faces), 3, 2) or not np.isfinite(uv).all():
        raise ValueError("Every face needs finite corner UV")
    old_faces = baseline["faces"]
    old_ids = baseline["vertex_ids"][old_faces]
    used = np.unique(faces)
    if len(used) != len(vertices):
        old_unused = np.setdiff1d(baseline["vertex_ids"], old_ids)
        unused = np.setdiff1d(current["vertex_ids"], current["vertex_ids"][used])
        if not np.isin(unused, old_unused).all():
            raise ValueError("New loose vertices")
    new_ids = current["vertex_ids"].copy()
    new_ids[new_ids < 0] = -2 - np.flatnonzero(new_ids < 0)
    corner_ids = new_ids[faces]

    before = baseline["vertices"][old_faces].astype(float)
    after = vertices[faces]
    before_normal = np.cross(before[:, 1] - before[:, 0], before[:, 2] - before[:, 0])
    after_normal = np.cross(after[:, 1] - after[:, 0], after[:, 2] - after[:, 0])
    known = original >= 0
    same_loop = np.zeros(len(faces), bool)
    if known.any():
        expected = old_ids[original[known]]
        actual = corner_ids[known]
        same_loop[known] = np.logical_or.reduce([
            np.all(actual == np.roll(expected, index, axis=1), axis=1) for index in range(3)
        ])
    zero = np.linalg.norm(after_normal, axis=1) <= 1e-10
    if np.any(zero & ~(same_loop & (np.linalg.norm(before_normal[original], axis=1) <= 1e-10))):
        raise ValueError("Face has zero area")
    normal_dot = np.einsum("ij,ij->i", before_normal[original], after_normal)
    # A legacy vertical or folded triangle can remain exactly as it was.
    reversed_face = known & (normal_dot < -1e-10) & (
        np.all(np.sort(corner_ids, axis=1) == np.sort(old_ids[original], axis=1), axis=1) | terrain
    )
    if terrain:
        reversed_face |= (after_normal[:, 2] < -1e-10) & (~known | (before_normal[original, 2] > 1e-10))
    if reversed_face.any():
        raise ValueError("Reversed face winding")
    old_duplicate = _duplicate_faces(baseline["vertices"], old_faces)
    if np.any(_duplicate_faces(vertices, faces) & ~(same_loop & old_duplicate[original])):
        raise ValueError("Duplicate face")
    if not terrain:
        offset = int(baseline["vertex_ids"].max()) + 1
        old_ids = _welded_corner_ids(baseline["vertices"], old_faces, baseline["vertex_ids"], offset)
        corner_ids = _welded_corner_ids(vertices, faces, new_ids, offset)
    if not np.array_equal(corner_ids, old_ids):
        old_boundary, old_defects = _edge_counts(old_ids)
        new_boundary, new_defects = _edge_counts(corner_ids)
        if new_boundary != old_boundary:
            raise ValueError("Open boundary changed (hole, missing exterior edge or winding)")
        if new_defects - old_defects:
            raise ValueError("Inconsistent winding or non-manifold edges")
        if not terrain and (~known).any():
            _check_closed_winding(after, corner_ids, ~known, new_boundary | new_defects)


def _inside_terrain(source, xy):
    if not len(xy):
        return
    triangles = source["vertices"][source["faces"], :2]
    low, high = triangles.min(axis=1), triangles.max(axis=1)
    for point in xy:
        candidates = np.all((point >= low - 1e-7) & (point <= high + 1e-7), axis=1)
        polygons = shapely.polygons(triangles[candidates])
        if not len(polygons) or np.min(shapely.distance(polygons, shapely.Point(point))) > 1e-7:
            raise ValueError("Vertex outside terrain edit scope (original surface)")


def validate_terrain(source, baseline, current, site):
    """Return world coordinates, original provenance and Blender corner UVs."""
    if "uv" not in current:
        raise ValueError("Every face needs UV")
    if all(key in current and np.array_equal(value, current[key]) for key, value in baseline.items()):
        return source.copy()
    vertices, moved, known, offsets = _points(
        source, baseline, current, site, site["lock"], list(map(int, site["keep_z"]))
    )
    xy_moved = moved.copy()
    xy_moved[known] &= np.any(vertices[known, :2] != source["vertices"][offsets, :2], axis=1)
    _inside_terrain(source, vertices[xy_moved, :2])
    original = _indices(current["face_ids"], baseline["face_ids"])
    if len(np.unique(original[original >= 0])) != (original >= 0).sum():
        raise ValueError("Duplicate face IDs. Reset copied face IDs to 0 in Blender")
    _geometry(vertices, original, baseline, current, terrain=True)
    source_offsets = _indices(current["face_ids"], source["original_face_ids"])
    has_source = source_offsets >= 0
    remainder = np.full((len(original), 3), -1, dtype=np.int32)
    remainder[has_source] = source["coarse_remainder"][source_offsets[has_source]]
    coarse = source["coarse_remainder"][:, 0] >= 0
    coarse_ids = source["original_face_ids"][coarse]
    if not np.isin(coarse_ids, current["face_ids"]).all():
        raise ValueError("LOCK: coarse remainder face is missing")
    coarse_index = _indices(coarse_ids, current["face_ids"])
    before = source["vertex_ids"][source["faces"][coarse]]
    after = current["vertex_ids"][current["faces"][coarse_index]]
    if not np.array_equal(before, after):
        same_winding = np.logical_or.reduce([
            np.all(before == np.roll(after, i, axis=1), axis=1) for i in (0, 1, 2)
        ])
        if not same_winding.all():
            raise ValueError("LOCK: coarse remainder topology changed")
        order = np.argmax(before[:, :, None] == after[:, None, :], axis=2)
        current = current.copy()
        for key in ("faces", "uv", "normals"):
            if key in current:
                current[key] = current[key].copy()
                current[key][coarse_index] = current[key][coarse_index[:, None], order]
    source_faces = np.full(len(original), -1, dtype=source["source_faces"].dtype)
    source_faces[has_source] = source["source_faces"][source_offsets[has_source]]
    kinds = current["kind"]
    if np.any((kinds < 0) | (kinds >= len(KINDS))):
        raise ValueError("Invalid valley face kind")
    materials = np.asarray([
        source["materials"][src] if src >= 0 and material == baseline["materials"][old] else material
        for material, src, old in zip(current["materials"], source_offsets, original)
    ])
    _validate_face_materials(materials, current["face_ids"], np.unique(source["materials"]),
                             site, site.get("terrain_name", "valley_terrain"))
    if any(not np.array_equal(current[key][coarse_index], baseline[key][original[coarse_index]])
           for key in ("uv", "materials", "kind", "normals") if key in baseline):
        raise ValueError("LOCK: coarse remainder UV, material, kind or normals changed")
    result = dict(
        vertices=vertices, faces=current["faces"], uv=current["uv"],
        materials=materials, face_kind=KINDS[kinds],
        vertex_ids=current["vertex_ids"], original_face_ids=current["face_ids"],
        source_faces=source_faces, coarse_remainder=remainder,
    )
    if "normals" in source and "normals" in current:
        normal = current["normals"] @ kn5.C.T
        same_indices = has_source & np.all(
            current["vertex_ids"][current["faces"]] ==
            baseline["vertex_ids"][baseline["faces"][original]], axis=1,
        )
        unchanged = same_indices[:, None] & np.all(current["normals"] == baseline["normals"][original], axis=2)
        normal[unchanged] = source["normals"][source_offsets][unchanged]
        if not np.isfinite(normal).all():
            raise ValueError("Terrain has non-finite normals")
        result["normals"] = normal
    return result


def _dedicated_material(name, site):
    return re.fullmatch(rf"valley_edit_{re.escape(site)}_[A-Za-z0-9_-]+", name) is not None


def _validate_face_materials(materials, face_ids, allowed, site, part):
    problems = []
    for index in np.flatnonzero(~np.isin(materials, allowed)):
        name = materials[index]
        if not _dedicated_material(name, site.get("site", "")):
            face = str(face_ids[index]) if face_ids[index] >= 0 else f"new (index {index})"
            problems.append(f"{part} face {face}: {name}")
    if problems:
        raise ValueError("Material is not allowed on this surface:\n" + "\n".join(problems))


def validate_materials(used, baseline, current, site, staging):
    textures = {}
    for name in sorted(used):
        if name in baseline:
            if current.get(name, {}).get("signature") != baseline[name]["signature"]:
                raise ValueError(f"Shared material was modified: {name}")
            continue
        if not _dedicated_material(name, site):
            raise ValueError(f"Unregistered material: {name}")
        filename = current.get(name, {}).get("texture")
        if not filename or Path(filename).name != filename:
            raise ValueError(f"Material needs a single baked PNG: {name}")
        path = staging / filename
        try:
            with Image.open(path) as image:
                if image.format != "PNG" or max(image.size) > 2048:
                    raise ValueError("Expected PNG up to 2048 × 2048")
                image.verify()
        except (OSError, ValueError) as exc:
            raise ValueError(f"Invalid baked PNG for {name}: {exc}") from exc
        textures[name] = path
    return textures


def _same(before, after):
    return before.keys() == after.keys() and all(
        np.array_equal(before[key], after[key]) for key in before
    )


def _part_reference(baseline, current):
    """Map a Blender triangle back to one baseline triangle, including split quads."""
    candidates = {}
    for index, face_id in enumerate(baseline["face_ids"]):
        candidates.setdefault(int(face_id), []).append(index)
    result = np.full(len(current["faces"]), -1, int)
    for index, face_id in enumerate(current["face_ids"]):
        if face_id == -1:
            continue
        matches = candidates.get(int(face_id))
        if not matches:
            raise ValueError("Unknown original part face ID")
        target = set(current["vertex_ids"][current["faces"][index]])
        result[index] = max(matches, key=lambda j: len(target.intersection(
            baseline["vertex_ids"][baseline["faces"][j]]
        )))
    return result


def _validate_part(source, baseline, current, site, part):
    if _same(baseline, current):
        return source.copy()
    protected = part.get("protected_faces", [])
    vertices, _, _, _ = _points(source, baseline, current, site, part.get("lock", []), [])
    original = _part_reference(baseline, current)
    _geometry(vertices, original, baseline, current, terrain=False)
    if np.isin(current["face_ids"], protected).sum() != np.isin(baseline["face_ids"], protected).sum():
        raise ValueError("Protected part faces removed or duplicated")
    for index in np.flatnonzero(np.isin(current["face_ids"], protected)):
        slot = original[index]
        if (not np.array_equal(current["vertex_ids"][current["faces"][index]],
                               baseline["vertex_ids"][baseline["faces"][slot]])
                or any(not np.array_equal(current[key][index], baseline[key][slot])
                       for key in ("uv", "materials", "normals"))):
            raise ValueError("Protected part faces changed")
    material = source["material"].item()
    assignments = np.where(
        (original >= 0) & (current["materials"] == baseline["materials"][original]),
        material, current["materials"],
    )
    _validate_face_materials(assignments, current["face_ids"], (material,), site, part.get("name", "part"))
    materials = np.unique(assignments)
    normals = current["normals"] @ kn5.C.T
    unchanged = (original >= 0) & np.all(
        current["vertex_ids"][current["faces"]] ==
        baseline["vertex_ids"][baseline["faces"][original]], axis=1
    )
    source_normals = source["normals"]
    if not source_normals.size:
        source_normals = kn5._normals(source["vertices"] @ kn5.C.T, kn5._triangles(source["faces"]))
    keep = unchanged[:, None] & np.all(current["normals"] == baseline["normals"][original], axis=2)
    normals[keep] = source_normals[current["vertex_ids"][current["faces"]][keep]]
    if not np.isfinite(normals).all():
        raise ValueError("Part has non-finite normals")
    result = dict(vertices=vertices, faces=current["faces"], uv=current["uv"], normals=normals,
                  material=np.asarray(materials[0] if len(materials) == 1 else material))
    if len(materials) > 1:
        result["materials"] = assignments
    return result


def _record(data, prefix):
    names = (*SOURCE_FIELDS.values(), "material", "normals", "kind")
    return {name: data[f"{prefix}_{name}"] for name in names if f"{prefix}_{name}" in data}


def source_fingerprint(meta):
    return {**meta["fingerprint"], "part_materials": {p["name"]: p["material"] for p in meta["parts"]}}


def stage_edit(directory, capture_path, output):
    meta = json.loads((directory / "site.json").read_text(encoding="utf-8"))
    with np.load(directory / "source.npz") as data:
        source = dict(data)
    with np.load(directory / "baseline_scene.npz") as data:
        baseline = dict(data)
    with np.load(capture_path) as data:
        current = dict(data)
    frozen = json.loads(baseline["frozen"].item())
    ref = json.loads(current["frozen"].item())
    if frozen != ref:
        changed = sorted(key for key in frozen.keys() | ref.keys() if frozen.get(key) != ref.get(key))
        raise ValueError(f"LOCK_context or REF changed: {', '.join(changed[:8])}")
    edit_objects = json.loads(current["editable"].item())
    expected = {meta.get("terrain_name", "valley_terrain"): "EDIT_terrain"}
    expected.update({part["name"]: "EDIT_parts" for part in meta["parts"]})
    if expected != edit_objects:
        raise ValueError("Editable object list or collection has changed")
    terrain_source = {key: source["terrain_" + original] for key, original in SOURCE_FIELDS.items()}
    if np.issubdtype(terrain_source["face_kind"].dtype, np.integer):
        terrain_source["face_kind"] = KINDS[terrain_source["face_kind"]]
    if "terrain_normals" in source:
        terrain_source["normals"] = source["terrain_normals"]
    terrain_name = meta.get("terrain_name", "valley_terrain")
    terrain_current = _record(current, terrain_name)
    terrain = validate_terrain(
        terrain_source, _record(baseline, terrain_name), terrain_current, meta
    )
    used = set(terrain_current["materials"])
    material_base = json.loads(baseline["materials"].item())
    material_current = json.loads(current["materials"].item())
    changed_parts = []
    (output / "parts").mkdir()
    (output / "textures").mkdir()
    np.savez_compressed(output / "terrain.npz", **terrain)
    for part in meta["parts"]:
        name = part["name"]
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", name) or name in (".", ".."):
            raise ValueError(f"Invalid part name: {name}")
        base = _record(source, part["key"])
        base["material"] = np.asarray(part["material"])
        base["vertex_ids"] = np.arange(len(base["vertices"]))
        current_part = _record(current, name)
        data = _validate_part(base, _record(baseline, name), current_part, meta, part)
        data.pop("vertex_ids", None)
        if not _same({key: value for key, value in base.items() if key != "vertex_ids"}, data):
            changed_parts.append(name)
        used.update(current_part["materials"])
        np.savez_compressed(output / "parts" / f"{name}.npz", **data)
    textures = validate_materials(used, material_base, material_current, meta["site"], capture_path.parent)
    specs = {}
    for key, path in textures.items():
        filename = f"{key}.png"
        shutil.copyfile(path, output / "textures" / filename)
        specs[key] = dict(texture=f"textures/{filename}", alpha=bool(material_current[key].get("alpha")),
                          two_sided=bool(material_current[key].get("two_sided")))
    edit = dict(site=meta["site"], fingerprint=source_fingerprint(meta),
                terrain_faces=terrain_source["original_face_ids"].tolist(),
                terrain_changed=not _same(terrain_source, terrain), changed_parts=changed_parts,
                materials=specs, blender_version=current["blender_version"].item())
    (output / "edit.json").write_text(json.dumps(edit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return edit
