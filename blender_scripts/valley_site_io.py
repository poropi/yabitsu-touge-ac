"""Blender-side capture shared by site export and import."""

import hashlib
import json
from pathlib import Path

import bpy
import numpy as np


EDIT_COLLECTIONS = {"EDIT_terrain", "EDIT_parts"}


def _json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def _hash(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _walk_collections(collection):
    yield collection
    for child in collection.children:
        yield from _walk_collections(child)


def _scalar_rna(item):
    values = {}
    for prop in item.bl_rna.properties:
        if prop.type not in {"BOOLEAN", "INT", "FLOAT", "STRING", "ENUM"}:
            continue
        value = getattr(item, prop.identifier)
        if getattr(prop, "is_array", False):
            value = list(value)
        elif isinstance(value, set):
            value = sorted(value)
        values[prop.identifier] = value
    return values


def _image_signature(image, cache):
    if image.name in cache:
        return cache[image.name]
    if image.packed_file:
        content = hashlib.sha256(image.packed_file.data).hexdigest()
    else:
        path = Path(bpy.path.abspath(image.filepath, library=image.library))
        if path.is_file():
            with path.open("rb") as handle:
                content = hashlib.file_digest(handle, "sha256").hexdigest()
        else:
            content = None
    # Saving a blend in a different directory rewrites Blender's relative paths.
    # The packed or on-disk bytes determine the image when they are available.
    result = image.name, image.source, image.colorspace_settings.name, content or image.filepath
    cache[image.name] = result
    return result


def _socket(socket):
    value = getattr(socket, "default_value", None)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bpy.types.ID):
        return value.name
    return list(value)


def _nodes(tree, images):
    if tree is None:
        return None
    nodes = {}
    for node in tree.nodes:
        params = _scalar_rna(node)
        # Node editor interaction does not change scene content.
        for field in ("select", "location", "width", "width_hidden", "height", "dimensions", "show_options",
                      "show_preview", "show_texture", "hide", "location_absolute"):
            params.pop(field, None)
        image = getattr(node, "image", None)
        group = getattr(node, "node_tree", None)
        ramp = getattr(node, "color_ramp", None)
        nodes[node.name] = dict(
            params=params,
            inputs=[(port.identifier, _socket(port)) for port in node.inputs],
            outputs=[(port.identifier, _socket(port)) for port in node.outputs],
            image=_image_signature(image, images) if image else None,
            group=_nodes(group, images) if group else None,
            ramp=([ramp.interpolation, ramp.color_mode, ramp.hue_interpolation]
                  + [(item.position, list(item.color)) for item in ramp.elements]) if ramp else None,
        )
    links = sorted((link.from_node.name, link.from_socket.identifier,
                    link.to_node.name, link.to_socket.identifier) for link in tree.links)
    return nodes, links


def _material(material, images, texture_dir, prefix):
    values = _scalar_rna(material)
    for key in ("users", "session_uid", "use_fake_user", "is_evaluated", "is_editable"):
        values.pop(key, None)
    spec = {
        "signature": _hash([values, _nodes(material.node_tree, images), material.get("valley_alpha", False)]),
        "two_sided": not material.use_backface_culling,
        "alpha": bool(material.get("valley_alpha", False)),
    }
    if texture_dir is None or not material.name.startswith(prefix):
        return spec
    tree = material.node_tree
    output = next((node for node in tree.nodes
                   if node.type == "OUTPUT_MATERIAL" and node.is_active_output), None) if tree else None
    surface = output.inputs["Surface"].links if output else ()
    if len(surface) != 1 or surface[0].from_node.type != "BSDF_PRINCIPLED":
        return spec
    base_color = surface[0].from_node.inputs["Base Color"].links
    if len(base_color) != 1 or base_color[0].from_node.type != "TEX_IMAGE":
        return spec
    node = base_color[0].from_node
    image = node.image
    if not image or base_color[0].from_socket.identifier != "Color" or node.inputs["Vector"].is_linked:
        return spec
    path = Path(bpy.path.abspath(image.filepath, library=image.library))
    if image.is_dirty:
        # A baked but unsaved image would otherwise emit an obsolete PNG from disk.
        return spec
    filename = f"baked_{len(list(texture_dir.glob('baked_*.png')))}.png"
    target = texture_dir / filename
    if image.packed_file:
        target.write_bytes(image.packed_file.data)
    elif path.is_file():
        target.write_bytes(path.read_bytes())
    else:
        return spec
    spec["texture"] = filename
    return spec


def _coords(mesh):
    value = np.empty((len(mesh.vertices), 3), np.float32)
    mesh.vertices.foreach_get("co", value.ravel())
    return value


def _attribute(mesh, key, size, default):
    layer = mesh.attributes.get(key)
    if layer is None:
        return np.full(size, default, np.int32)
    if len(layer.data) != size or layer.data_type != "INT":
        raise ValueError(f"Invalid attribute {key!r} in {mesh.name}")
    value = np.empty(size, np.int32)
    layer.data.foreach_get("value", value)
    return value


def _uv(mesh):
    if not mesh.uv_layers.active:
        return None
    value = np.empty((len(mesh.loops), 2), np.float32)
    mesh.uv_layers.active.data.foreach_get("uv", value.ravel())
    return value


def _mesh_signature(mesh):
    digest = hashlib.sha256()
    for values in (
        _coords(mesh),
        np.array([loop.vertex_index for loop in mesh.loops], np.int32),
        np.array([poly.loop_total for poly in mesh.polygons], np.int32),
        np.array([poly.material_index for poly in mesh.polygons], np.int32),
        np.array([poly.use_smooth for poly in mesh.polygons], bool),
        np.array([n.vector[:] for n in mesh.corner_normals], np.float32),
    ):
        digest.update(values.tobytes())
    for layer in mesh.uv_layers:
        uv = np.empty((len(mesh.loops), 2), np.float32)
        layer.data.foreach_get("uv", uv.ravel())
        digest.update(layer.name.encode())
        digest.update(uv.tobytes())
    return digest.hexdigest()


def _frozen_object(obj, specs, images, mesh_signatures):
    record = dict(
        transform=[list(row) for row in obj.matrix_world], type=obj.type,
        collections=sorted(collection.name for collection in obj.users_collection),
        visibility=[obj.hide_render, obj.hide_viewport, obj.hide_select],
        parent=obj.parent.name if obj.parent else None,
        modifiers=[_scalar_rna(item) for item in obj.modifiers],
        constraints=[_scalar_rna(item) for item in obj.constraints],
        instanced_collection=obj.instance_collection.name if obj.instance_collection else None,
    )
    if obj.type == "MESH":
        if obj.data.name not in mesh_signatures:
            mesh_signatures[obj.data.name] = _mesh_signature(obj.data)
        record["mesh"] = mesh_signatures[obj.data.name]
        record["materials"] = [(slot.material.name, specs[slot.material.name]["signature"])
                               if slot.material else None for slot in obj.material_slots]
    elif obj.type in {"CAMERA", "LIGHT"}:
        params = _scalar_rna(obj.data)
        for key in ("users", "session_uid", "use_fake_user", "is_evaluated", "is_editable"):
            params.pop(key, None)
        record["data"] = params
        if obj.type == "CAMERA":
            record["background"] = [
                [_scalar_rna(item), _image_signature(item.image, images) if item.image else None]
                for item in obj.data.background_images
            ]
    elif obj.type == "EMPTY" and isinstance(obj.data, bpy.types.Image):
        record["image"] = [_image_signature(obj.data, images), obj.get("camera"),
                           obj.empty_display_type, obj.empty_display_size, list(obj.empty_image_offset)]
    return _hash(record)


def _editable_mesh(obj):
    if obj.type != "MESH" or obj.mode != "OBJECT":
        raise ValueError(f"Save in Object Mode with a mesh: {obj.name}")
    if obj.modifiers or obj.constraints or obj.data.shape_keys:
        raise ValueError(f"Apply modifiers, constraints and shape keys before importing: {obj.name}")
    mesh = obj.data
    active_uv = mesh.uv_layers.active
    if active_uv is not None and not active_uv.active_render:
        raise ValueError(f"Use the same active and render UV map: {obj.name}")
    mesh.calc_loop_triangles()
    tris = list(mesh.loop_triangles)
    loops = np.array([tri.loops[:] for tri in tris], dtype=np.int32).reshape(-1, 3)
    polygons = np.array([tri.polygon_index for tri in tris], dtype=np.int32)
    matrix = np.array(obj.matrix_world)
    coords = _coords(mesh)
    normals = np.empty((len(mesh.loops), 3), np.float32)
    mesh.corner_normals.foreach_get("vector", normals.ravel())
    normal_matrix = np.linalg.inv(matrix[:3, :3])
    normals = normals[loops] @ normal_matrix
    norms = np.linalg.norm(normals, axis=2, keepdims=True)
    normals = normals / np.where(norms > 0, norms, 1.0)
    materials = [slot.material.name if slot.material else "" for slot in obj.material_slots]
    result = dict(
        vertices=coords @ matrix[:3, :3].T + matrix[:3, 3],
        faces=np.array([tri.vertices[:] for tri in tris], dtype=np.int32).reshape(-1, 3),
        normals=normals,
        vertex_ids=_attribute(mesh, "valley_vertex_id", len(coords), 0) - 1,
        face_ids=_attribute(mesh, "valley_face_id", len(mesh.polygons), 0)[polygons] - 1,
        materials=np.array([materials[mesh.polygons[index].material_index]
                            if mesh.polygons[index].material_index < len(materials) else ""
                            for index in polygons], dtype=str),
    )
    uv = _uv(mesh)
    if uv is not None:
        result["uv"] = uv[loops]
    return result, polygons


def capture(scene, output, *, texture_dir=None, site=None):
    """Capture the visible Blender state without saving or modifying the blend file."""
    bpy.context.view_layer.update()
    images = {}
    material_specs = {
        material.name: _material(material, images, texture_dir, f"valley_edit_{site}_")
        for material in bpy.data.materials
    }
    frozen = {}
    editable = {}
    mesh_signatures = {}
    arrays = {}
    for collection in _walk_collections(scene.collection):
        if collection.name in EDIT_COLLECTIONS:
            continue
        frozen[f"collection {collection.name}"] = _hash([
            sorted(child.name for child in collection.children),
            collection.hide_select, collection.hide_render, collection.hide_viewport,
        ])
    for layer in scene.view_layers:
        for collection in _walk_collections(layer.layer_collection):
            if collection.exclude or collection.hide_viewport:
                frozen[f"view layer {layer.name}/{collection.name}"] = _hash([
                    collection.exclude, collection.hide_viewport,
                ])
    for obj in scene.objects:
        collections = set(collection.name for collection in obj.users_collection)
        edit = collections & EDIT_COLLECTIONS
        if not edit:
            frozen[f"object {obj.name}"] = _frozen_object(obj, material_specs, images, mesh_signatures)
            continue
        if len(collections) != 1 or obj.hide_render or obj.hide_viewport:
            raise ValueError(f"Editable object has changed collection or visibility: {obj.name}")
        editable[obj.name] = edit.pop()
        values, polygons = _editable_mesh(obj)
        if editable[obj.name] == "EDIT_terrain":
            mesh = obj.data
            values["kind"] = _attribute(mesh, "valley_kind", len(mesh.polygons), -1)[polygons]
            values["source_faces"] = _attribute(mesh, "valley_source_faces", len(mesh.polygons), -1)[polygons]
            values["coarse_remainder"] = np.column_stack([
                _attribute(mesh, f"valley_coarse_remainder_{key}", len(mesh.polygons), -1)[polygons]
                for key in ("r", "c", "face")
            ])
        arrays.update({f"{obj.name}_{key}": value for key, value in values.items()})
    if scene.world:
        frozen["world"] = _hash(_nodes(scene.world.node_tree, images))
    np.savez_compressed(output, **arrays, frozen=_json(frozen), editable=_json(editable),
                        materials=_json(material_specs), blender_version=bpy.app.version_string)
