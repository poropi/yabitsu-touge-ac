"""Apply the Blender finishing data to the procedural valley build."""
from dataclasses import dataclass
from hashlib import file_digest as hash_stream, sha256
import json
from pathlib import Path
import re
import shutil
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image

from . import config, kn5
from .manifest import Manifest, MAX_VERTS

SITES = ("A-s4-1724", "B-s4-6354", "C-s5-8956")
SITES_ROOT = config.PROJECT_ROOT / "export" / "blender_sites"
SURFACE_KEYS = ("valley_cells", "valley_vertices", "valley_faces", "valley_bed",
                "valley_gravel", "valley_coarse_remainder", "valley_source_faces")
EDIT_KEYS = ("valley_edit_uv", "valley_edit_materials", "valley_edit_normals", "valley_edit_blocks",
             "valley_edit_corner_normals", "valley_edit_fingerprint", "valley_edit_base", "valley_edit_centers")
NAME = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]*")


@dataclass(frozen=True)
class Edit:
    site: str
    path: Path
    metadata: dict


def file_digest(path):
    with Path(path).open("rb") as stream:
        return hash_stream(stream, "sha256").hexdigest()


def surface_digest(near):
    """Hash the procedural arrays, independently of the NPZ archive encoding."""
    digest = sha256()
    for key in SURFACE_KEYS:
        if key in near:
            value = np.ascontiguousarray(near[key])
            digest.update(f"{key}:{value.dtype.str}:{value.shape}\n".encode())
            digest.update(value)
    return digest.hexdigest()


def load(directory=None):
    directory = Path(directory) if directory is not None else config.DATA_MANUAL / "valley_edits"
    result = []
    for site in SITES:
        if (directory / f".{site}.importing").exists():
            raise ValueError(f"{site}: 書き戻し中です")
        path = directory / site
        if not path.exists():
            continue
        metadata = json.loads((path / "edit.json").read_text(encoding="utf-8"))
        if metadata["terrain_changed"] or metadata["changed_parts"]:
            expected = metadata["fingerprint"].get("valleys")
            if expected and expected != file_digest(config.DATA_MANUAL / "valleys.json"):
                raise ValueError(f"{site}: valleys.json の指紋が変わりました。切り出した骨組みを確認してください")
            result.append(Edit(site, path, metadata))
    return tuple(result)


def _read_terrain(near, edit):
    with np.load(edit.path / "terrain.npz") as data:
        saved = dict(data)
    old = np.asarray(edit.metadata["terrain_faces"], dtype=np.int64)
    f, v = saved["faces"], saved["vertices"]
    face_ids, vertex_ids = saved["original_face_ids"], saved["vertex_ids"]
    coarse = saved["coarse_remainder"]
    if (f.ndim != 2 or f.shape[1] != 3 or v.shape != (len(vertex_ids), 3)
            or len(face_ids) != len(f) or saved["uv"].shape != (len(f), 3, 2)
            or any(saved[key].shape != (len(f),) for key in ("face_kind", "materials", "source_faces"))
            or coarse.shape != (len(f), 3) or np.any((f < 0) | (f >= len(v)))
            or not np.isfinite(v).all() or not np.isfinite(saved["uv"]).all()
            or not np.isin(saved["face_kind"], ("bed", "bank", "shoulder", "gravel")).all()
            or not np.all([NAME.fullmatch(material) is not None for material in saved["materials"]])):
        raise ValueError(f"{edit.site}: terrain.npz の形状または属性が不正です")
    if "normals" in saved:
        normals = saved["normals"]
        if normals.shape != (len(f), 3, 3) or not np.isfinite(normals).all():
            raise ValueError(f"{edit.site}: 地形の角法線が不正です")
    source = near["valley_faces"]
    if (old.ndim != 1 or not len(old) or len(np.unique(old)) != len(old)
            or (old < 0).any() or (old >= len(source)).any()):
        raise ValueError(f"{edit.site}: 切り出した元の面番号が不正です")
    for ids, allowed in ((face_ids, old), (vertex_ids, np.unique(source[old]))):
        existing = ids >= 0
        if ((ids < -1).any() or not np.isin(ids[existing], allowed).all()
                or len(np.unique(ids[existing])) != int(existing.sum())):
            raise ValueError(f"{edit.site}: 元の頂点番号または面番号が不正です")
    return old, saved


def _terrain_fingerprints(edits):
    return [f"{edit.site}:{file_digest(edit.path / 'terrain.npz')}"
            for edit in edits if edit.metadata["terrain_changed"]]


def _check_base(edits, digest):
    for edit in edits:
        if edit.metadata["fingerprint"].get("terrain") != digest:
            raise ValueError(f"{edit.site}: 谷の元配列の指紋が一致しません。新しい骨組みで切り出してください")


def check_applied(near, edits):
    """Avoid building meshes from a near file predating the selected edits."""
    actual = near.get("valley_edit_fingerprint", np.empty(0, str))
    if actual.tolist() != _terrain_fingerprints(edits):
        raise ValueError("近景に反映された地形の手直しが一致しません。先に近景を組み立ててください")
    if edits:
        base = near["valley_edit_base"].item() if "valley_edit_base" in near else surface_digest(near)
        _check_base(edits, base)


def terrain_centers(near):
    return {str(row["name"]): row["xy"] for row in near.get("valley_edit_centers", ())}


def apply_terrain(near, edits):
    """Replace only the recorded faces. Keep their original ordering and block IDs."""
    if not edits:
        return False
    expected = surface_digest(near)
    _check_base(edits, expected)
    patches = [e for e in edits if e.metadata["terrain_changed"]]
    if not patches:
        return False
    from .valley_surface import face_blocks, mesh_parts

    inputs = []
    occupied = []
    for edit in patches:
        old, data = _read_terrain(near, edit)
        inputs.append((edit, old, data))
        occupied.append(old)
    selected = np.concatenate(occupied)
    if len(np.unique(selected)) != len(selected):
        raise ValueError("手直しの対象面が橋の間で重複しています")
    # Material splits may move a mesh centroid across the S4/S5 boundary.
    # Keep the original group of meshes that retain their original names.
    prefixes = tuple(f"1GRASS_valley_{edit.site[0]}_" for edit in patches)
    centers = np.array([(name, v[:, :2].mean(axis=0)) for name, v, *_ in mesh_parts(near)
                        if name.startswith(prefixes)],
                       dtype=[("name", "U40"), ("xy", "f8", (2,))])

    old_v, old_f = near["valley_vertices"], near["valley_faces"]
    bed, gravel = near["valley_bed"], near["valley_gravel"]
    coarse, source = near["valley_coarse_remainder"], near["valley_source_faces"]
    vertices = np.empty((len(old_v) + sum((d["vertex_ids"] < 0).sum() for _, _, d in inputs), 3))
    vertices[:len(old_v)] = old_v
    faces = np.empty((len(old_f) + sum((d["original_face_ids"] < 0).sum() for _, _, d in inputs), 3),
                     dtype=old_f.dtype)
    faces[:len(old_f)] = old_f
    is_bed, is_gravel = np.r_[bed, np.zeros(len(faces) - len(old_f), bool)], np.r_[gravel, np.zeros(len(faces) - len(old_f), bool)]
    remainders = np.full((len(faces), 3), -1, dtype=coarse.dtype)
    remainders[:len(old_f)] = coarse
    source_faces = np.full(len(faces), -1, dtype=source.dtype)
    source_faces[:len(old_f)] = source
    blocks = np.empty(len(faces), dtype=np.int64)
    blocks[:len(old_f)] = face_blocks(near)
    uv = np.zeros((len(faces), 3, 2), dtype=np.float32)
    corner_normals = (np.full((len(faces), 3, 3), np.nan)
                      if any("normals" in d for _, _, d in inputs) else None)
    max_material = max(d["materials"].dtype.itemsize // 4 for _, _, d in inputs)
    materials = np.full(len(faces), "", dtype=f"<U{max_material}")
    use_face = np.ones(len(faces), bool)
    use_face[selected] = False
    untouched = np.ones(len(old_f), bool)
    untouched[selected] = False
    locked = np.unique(old_f[untouched | (coarse[:, 0] >= 0)])
    coarse_edit = selected[coarse[selected, 0] >= 0]
    cursor_v, cursor_f = len(old_v), len(old_f)
    for edit, old, data in inputs:
        j = data["vertex_ids"].copy()
        new = j < 0
        j[new] = np.arange(cursor_v, cursor_v + new.sum())
        cursor_v += new.sum()
        fixed = np.isin(j, locked)
        if not np.array_equal(vertices[j[fixed]], data["vertices"][fixed]):
            raise ValueError(f"{edit.site}: 共用頂点または粗い格子の残片が移動しています")
        vertices[j] = data["vertices"]

        i = data["original_face_ids"].copy()
        new = i < 0
        i[new] = np.arange(cursor_f, cursor_f + new.sum())
        cursor_f += new.sum()
        retained = np.isin(i, coarse_edit)
        rem = data["coarse_remainder"]
        f = j[data["faces"]]
        same_coarse = np.logical_or.reduce([
            np.all(f[retained] == np.roll(old_f[i[retained]], n, axis=1), axis=1)
            for n in range(3)
        ])
        if (not same_coarse.all()
                or not np.array_equal(coarse[i[retained]], rem[retained])
                or not np.array_equal(rem[:, 0] >= 0, retained)
                or not np.isin(old[coarse[old, 0] >= 0], i).all()):
            raise ValueError(f"{edit.site}: 粗い格子の残片または印が一致しません")
        faces[i], uv[i], materials[i] = f, data["uv"], data["materials"]
        if corner_normals is not None and "normals" in data:
            corner_normals[i] = data["normals"]
        is_bed[i] = data["face_kind"] == "bed"
        is_gravel[i] = data["face_kind"] == "gravel"
        remainders[i], source_faces[i] = rem, data["source_faces"]
        blocks[i[new]] = blocks[old[0]]
        use_face[i] = True

    normals = kn5._normals(vertices @ kn5.C.T, faces[use_face])
    normals[locked] = kn5._normals(old_v @ kn5.C.T, old_f)[locked]
    near.update(valley_vertices=vertices, valley_faces=faces[use_face], valley_bed=is_bed[use_face],
                valley_gravel=is_gravel[use_face], valley_coarse_remainder=remainders[use_face],
                valley_source_faces=source_faces[use_face], valley_edit_uv=uv[use_face],
                valley_edit_materials=materials[use_face], valley_edit_normals=normals,
                valley_edit_blocks=blocks[use_face], valley_edit_fingerprint=np.asarray(_terrain_fingerprints(edits)),
                valley_edit_base=np.asarray(expected), valley_edit_centers=centers)
    if corner_normals is not None:
        near["valley_edit_corner_normals"] = corner_normals[use_face]
    return True


def register_materials(manifest, edits):
    """Put each bake in a dedicated txDiffuse texture, leaving shared materials intact."""
    output = manifest.root / "textures"
    for edit in edits:
        for image in sorted((edit.path / "textures").glob("*.png")):
            name = image.stem
            if not NAME.fullmatch(name) or not name.startswith(f"valley_edit_{edit.site}_"):
                raise ValueError(f"{edit.site}: 専用の材質名が必要です　{name}")
            with Image.open(image) as source:
                if source.format != "PNG" or max(source.size) > 2048:
                    raise ValueError(f"{name}: 2048×2048以下のPNGが必要です")
                source.verify()
            relative = f"textures/{image.name}"
            existing = manifest.materials.get(name)
            if existing is not None and existing.get("texture") != relative:
                raise ValueError(f"材質名が重複しています　{name}")
            output.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(image, output / image.name)
            material = edit.metadata.get("materials", {}).get(name, {})
            manifest.add_material(name, texture=relative,
                                  alpha=material.get("alpha", False),
                                  two_sided=material.get("two_sided", False))


def _part_objects(edit, name, data):
    verts, faces, uv, normals = (data[k] for k in ("vertices", "faces", "uv", "normals"))
    base_material = data["material"].item()
    materials = data.get("materials", np.full(len(faces), base_material))
    if (faces.ndim != 2 or faces.shape[1] not in (3, 4) or not len(faces)
            or verts.shape != (len(verts), 3) or (faces < 0).any() or (faces >= len(verts)).any()
            or uv.shape not in ((len(verts), 2), (*faces.shape, 2))
            or (normals.size and normals.shape not in ((len(verts), 3), (*faces.shape, 3)))
            or any(not np.isfinite(a).all() for a in (verts, uv, normals))
            or materials.shape != (len(faces),) or not all(NAME.fullmatch(k) for k in np.unique(materials))):
        raise ValueError(f"{edit.site}: 部品の形状または属性が不正です　{name}")
    choices = np.unique(materials).tolist()
    primary = base_material if base_material in choices else choices[0]
    for material in [primary, *(m for m in choices if m != primary)]:
        select = materials == material
        f = faces[select]
        generated = name if material == primary else f"{name}_edit_{material.removeprefix(f'valley_edit_{edit.site}_')}"
        # Keep the original index/attribute layout of a single-material part
        # if it already has one UV and normal per vertex.
        if len(choices) == 1 and uv.ndim == 2 and normals.ndim == 2:
            v, u, n = verts, uv, normals
        else:
            v = verts[f].reshape(-1, 3)
            u = (uv[select] if uv.ndim == 3 else uv[f]).reshape(-1, 2)
            n = ((normals[select] if normals.ndim == 3 else normals[f]).reshape(-1, 3)
                 if normals.size else normals)
            f = np.arange(len(v)).reshape(f.shape)
        if len(v) > MAX_VERTS:
            raise ValueError(f"{edit.site}: 部品の材質ごとの頂点数が上限を超えています　{generated}")
        yield generated, (v, f, u, material, n if n.size else None)


def part_edits(edits):
    replacements = {}
    for edit in edits:
        for name in edit.metadata["changed_parts"]:
            if not NAME.fullmatch(name) or name.startswith("mt_") or name in replacements:
                raise ValueError(f"{edit.site}: 置き換えられない部品名です　{name}")
            with np.load(edit.path / "parts" / f"{name}.npz") as data:
                pieces = tuple(_part_objects(edit, name, data))
            replacements[name] = dict(pieces)
            if len(pieces) != len(replacements[name]):
                raise ValueError(f"{edit.site}: 部品名が重複しています　{name}")
    return replacements


def _part_source(edit, name, fallback=None):
    fingerprint = edit.metadata["fingerprint"]
    return dict(sha256=fingerprint.get("parts", {}).get(name),
                material=fingerprint.get("part_materials", {}).get(name, fallback))


def check_part_sources(manifest, edits, replacements, *, sites_root=SITES_ROOT):
    """Compare rebuilt original parts with the arrays saved at extraction."""
    if not replacements:
        return
    from .valley_site_import import source_fingerprint

    for edit in edits:
        if not edit.metadata["changed_parts"]:
            continue
        directory = sites_root / edit.site
        source = directory / "source.npz"
        if not source.is_file():
            raise ValueError(f"{edit.site}: 切り出し時の source.npz がありません　{source}")
        meta = json.loads((directory / "site.json").read_text(encoding="utf-8"))
        expected_sha = edit.metadata.get("source_sha256")
        if (meta["site"] != edit.site or source_fingerprint(meta) != edit.metadata["fingerprint"]
                or (expected_sha and file_digest(source) != expected_sha)):
            raise ValueError(f"{edit.site}: 切り出し時の source.npz または site.json の指紋が一致しません")
        originals = {part["name"]: part for part in meta["parts"]}
        with np.load(source) as saved:
            for name in edit.metadata["changed_parts"]:
                part, entry = originals.get(name), manifest.meshes.get(name)
                if part is None or entry is None:
                    raise ValueError(f"{edit.site}: 切り出し元または再生成した部品が見つかりません　{name}")
                if part["material"] != entry["material"]:
                    raise ValueError(f"{edit.site}: 再生成した部品の元の材質が違います　{name}")
                with np.load(manifest.root / entry["file"]) as fresh:
                    for key, suffix in (("verts", "vertices"), ("faces", "faces"),
                                        ("uvs", "uv"), ("normals", "normals")):
                        old = saved[f'{part["key"]}_{suffix}']
                        new = fresh[key] if key in fresh else np.empty((0, 3))
                        if old.dtype != new.dtype or not np.array_equal(old, new):
                            raise ValueError(f"{edit.site}: 再生成した部品の配列が違います　{name}: {key}")


def replace_water(items, replacements):
    """Use the final water and bed meshes for tree exclusion, before the manifest is saved."""
    if not replacements:
        return items
    final = []
    for name, v, f, uv, material, visible in items:
        if name not in replacements:
            final.append((name, v, f, uv, material, visible))
            continue
        final.extend((key, vv, ff, uu, mat, visible)
                     for key, (vv, ff, uu, mat, _) in replacements[name].items())
    return final


def replace_parts(manifest, edits, *, replacements=None, sites_root=SITES_ROOT):
    replacements = part_edits(edits) if replacements is None else replacements
    if not replacements:
        return
    check_part_sources(manifest, edits, replacements, sites_root=sites_root)
    group = {name: g for g, spec in manifest.groups.items() for name in spec["meshes"]}
    owners = {name: edit for edit in edits for name in edit.metadata["changed_parts"]}
    names = set()
    for parent, pieces in replacements.items():
        if parent not in manifest.meshes:
            raise ValueError(f"置換元の部品が見つかりません　{parent}")
        for name, (_, _, _, material, _) in pieces.items():
            present = manifest.meshes.get(name)
            if name in names or (name != parent and present and present.get("valley_edit_parent") != parent):
                raise ValueError(f"部品名が重複しています　{name}")
            if material not in manifest.materials:
                raise ValueError(f"部品の材質が登録されていません　{name}: {material}")
            names.add(name)
    with TemporaryDirectory(dir=manifest.root) as directory:
        stage = Manifest(directory)
        for parent, pieces in replacements.items():
            for name, (v, f, uv, material, normals) in pieces.items():
                stage.add_mesh(group[parent], name, v, f, uv, material,
                               manifest.meshes[parent]["renderable"], normals=normals)
        for parent, pieces in replacements.items():
            original = manifest.meshes[parent]
            saved_source = original.get("valley_edit_source", {})
            source = _part_source(owners[parent], parent, saved_source.get("material", original["material"]))
            for name in pieces:
                entry = stage.meshes[name]
                old = manifest.meshes.get(name)
                if old:
                    relative = old["file"]
                    old.update(verts=entry["verts"], faces=entry["faces"], material=entry["material"])
                else:
                    relative = entry["file"]
                    old = manifest.meshes[name] = {**entry, "valley_edit_parent": parent}
                    manifest.groups[group[parent]]["meshes"].append(name)
                path = manifest.root / relative
                (stage.root / entry["file"]).replace(path)
                if name == parent:
                    old["valley_edit_source"] = source
                    old["valley_edit_result"] = dict(sha256=file_digest(path), material=entry["material"])
        for name, entry in list(manifest.meshes.items()):
            if entry.get("valley_edit_parent") in replacements and name not in names:
                manifest.groups[group[name]]["meshes"].remove(name)
                del manifest.meshes[name]
