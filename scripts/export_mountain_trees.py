"""依頼55の Blender 樹木を CSP の木パッケージへ書き出す。"""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np


SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[1]
DEFAULT_ASSET = ROOT / "assets/landmarks/trees_mountain/asset.json"
PARTS = ("leaf", "bark")


def mesh_islands(triangles, vertex_count):
    """共有頂点でつながった三角形を、葉板または幹・枝の単位にまとめる。"""
    roots = list(range(vertex_count))

    def root(vertex):
        while roots[vertex] != vertex:
            roots[vertex] = roots[roots[vertex]]
            vertex = roots[vertex]
        return vertex

    for a, b, c in triangles.tolist():
        a = root(a)
        roots[root(b)] = a
        roots[root(c)] = a
    return np.array([root(int(first)) for first in triangles[:, 0]])


def mesh_arrays(obj):
    mesh = obj.data
    mesh.calc_loop_triangles()

    def read(collection, attribute, width, dtype=np.float32):
        buffer = np.empty(len(collection) * width, dtype=dtype)
        collection.foreach_get(attribute, buffer)
        return buffer.reshape(-1, width)

    corners = read(mesh.loops, "vertex_index", 1, np.int32).ravel()
    triangles = read(mesh.loop_triangles, "loops", 3, np.int32)
    transform = np.array(obj.matrix_world)
    linear = transform[:3, :3]
    if np.linalg.det(linear) < 0:
        triangles = triangles[:, [0, 2, 1]]
    uv = read(mesh.uv_layers.active.uv, "vector", 2)
    uv[:, 1] = 1 - uv[:, 1]  # Blender の下原点から DDS の上原点へ
    return {
        "p": read(mesh.vertices, "co", 3)[corners] @ linear.T + transform[:3, 3],
        "n": read(mesh.corner_normals, "vector", 3) @ np.linalg.inv(linear),
        "uv": uv,
        "faces": triangles,
        "islands": mesh_islands(corners[triangles], len(mesh.vertices)),
    }


def extract(asset_path, staging):
    """Blender 同梱 Python でコレクションとパック済み画像を取り出す。"""
    import bpy

    asset = json.loads(asset_path.read_text(encoding="utf-8"))
    written_images = set()
    for model in asset["models"]:
        identifier = model["id"]
        for lod in range(2):
            name = f"{identifier}_LOD{lod}"
            collection = bpy.data.collections[name]
            for part in PARTS:
                suffix = "leaves" if part == "leaf" else "wood"
                obj = collection.objects[f"{name}_{suffix}"]
                material, = obj.data.materials
                image, = (node.image for node in material.node_tree.nodes if node.type == "TEX_IMAGE")
                if image.name != model[part]:
                    raise ValueError(f"{obj.name}: {model[part]} を使用する材質が必要です（実際は {image.name}）")
                if image.name not in written_images:
                    (staging / image.name).write_bytes(image.packed_file.data)
                    written_images.add(image.name)
                np.savez(staging / f"{name}_{part}.npz", **mesh_arrays(obj))


def distant_faces(mesh, leaf):
    faces = mesh["faces"]
    islands = [faces[mesh["islands"] == label] for label in np.unique(mesh["islands"])]
    points = [mesh["p"][indices.ravel()] for indices in islands]
    minimum = np.array([p.min(axis=0) for p in points])
    maximum = np.array([p.max(axis=0) for p in points])
    if not leaf:
        # いちばん長い幹・茎を残し、細い枝を省く。
        return islands[np.argmax(maximum[:, 2] - minimum[:, 2])]

    # 樹冠の端を出す葉板を残し、残りは大きく離れた板から選んで樹冠の間を埋める。
    keep = set(minimum.argmin(axis=0)) | set(maximum.argmax(axis=0))
    span = np.maximum(maximum.max(axis=0) - minimum.min(axis=0), 0.001)
    centers = (minimum + maximum) / (2 * span)
    areas = np.array([np.linalg.norm(np.cross(p[1::3] - p[::3], p[2::3] - p[::3]), axis=1).sum()
                      for p in points])
    weight = np.sqrt(areas / areas.max())
    while len(keep) < min(24, len(islands)):
        nearest = np.linalg.norm(centers[:, None] - centers[sorted(keep)], axis=2).min(axis=1)
        score = nearest * weight
        score[list(keep)] = -1
        keep.add(int(score.argmax()))
    return np.concatenate([islands[index] for index in sorted(keep)])


def pack_lod(source, lod, height, half_width, textures):
    from yabitsu import csptree, kn5, treemodels

    parts = []
    report = {}
    for name in PARTS:
        mesh = source[name]
        faces = mesh["faces"] if lod < 2 else distant_faces(mesh, name == "leaf")
        used, indices = np.unique(faces.ravel(), return_inverse=True)
        indices = indices.reshape(-1, 3)
        positions = mesh["p"][used]
        ac_positions = csptree.to_ac(positions)
        normals = treemodels._unit(csptree.to_ac(mesh["n"][used]))
        uv = mesh["uv"][used]
        tangents = kn5._tangents(ac_positions, uv, normals, indices)
        part = treemodels.Part()
        ao = np.full(len(used), 0.88 if name == "leaf" else 0.75)
        part.add(ac_positions, normals, uv, tangents, ao, indices)
        parts.append(part)
        report[name] = {
            "triangles": len(faces),
            "bounds_m": [positions.min(axis=0).tolist(), positions.max(axis=0).tolist()],
        }
        if name == "leaf":
            reach = positions[positions[:, 0] >= 1.0]
            if len(reach):
                report[name]["lowest_at_positive_x_ge_1m_m"] = float(reach[:, 2].min())
    return csptree.write_mesh(textures, treemodels._pack(parts, height, half_width, textures)), report


def model_package(model, staging, images):
    from yabitsu import treemodels

    name = f"mt_{model['id'].lower()}"
    source = []
    for lod in range(2):
        materials = {}
        for part in PARTS:
            with np.load(staging / f"{model['id']}_LOD{lod}_{part}.npz") as archive:
                materials[part] = dict(archive)
        source.append(materials)
    positions = np.concatenate([m["p"][m["faces"].ravel()] for lod in source for m in lod.values()])
    height = float(positions[:, 2].max())
    # 根元を中心とする回転で規格化範囲を越えないよう、最大の水平半径を半幅にする。
    half_width = float(np.linalg.norm(positions[:, :2], axis=1).max())
    category = model["id"].split("_")[0]
    species = {"oak": "broadleaf", "roadside": "sapling"}.get(category, category)
    ini = {
        "BASIC": {"HEIGHT": height, "WIDTH": 2 * half_width, "DISTANCE": treemodels.DRAW_DISTANCE[species]},
        "SHADING": {"SPECULAR": 0.03, "SUBSCATTERING": 0.3, "REFLECTIVITY": 0.02, "BRIGHTNESS": 1.0},
        "NORMALS": {"BOOST": 1, "SPHERE": 0.3},
    }
    meshes, textures, lod_report = {}, {}, []
    for part in PARTS:
        png = model[part]
        stem = Path(png).stem
        diffuse, normal = images[png]
        textures[f"{stem}.dds"] = diffuse
        textures[f"{stem}_nm.dds"] = normal
    for lod, distance in enumerate(treemodels.LOD_DISTANCE):
        mesh_name = f"LOD{lod}.mesh"
        meshes[mesh_name], metrics = pack_lod(source[min(lod, 1)], lod, height, half_width, list(textures))
        ini[f"LOD_{lod}"] = {"MODEL": mesh_name, "DISTANCE": distance}
        lod_report.append({"lod": lod, "distance_m": distance, **metrics})
    report = {"package": name, "source_id": model["id"], "height_m": height, "width_m": 2 * half_width,
              "draw_distance_m": ini["BASIC"]["DISTANCE"], "lods": lod_report}
    return name, ini, meshes, textures, report


def export(asset_path, out_dir, audit_path, blender):
    from PIL import Image
    from yabitsu import csptree, kn5

    asset = json.loads(asset_path.read_text(encoding="utf-8"))
    source = asset_path.parent / asset["source"]
    out_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".mountain_trees_", dir=out_dir) as temporary:
        staging = Path(temporary)
        subprocess.run([blender, "-b", "--factory-startup", str(source), "--python-exit-code", "1",
                        "--python", str(SCRIPT), "--", str(asset_path), str(staging)], check=True)
        images = {}
        for png in sorted({m[p] for m in asset["models"] for p in PARTS}):
            with Image.open(staging / png) as image:
                # 制作元に法線画像はないため、元の頂点法線を変えない平坦な接空間法線を使う。
                neutral = Image.new("RGBA", image.size, (128, 128, 255, 255))
                images[png] = (kn5.dds_bytes(image), kn5.dds_bytes(neutral))
        reports = []
        for model in asset["models"]:
            name, ini, meshes, textures, report = model_package(model, staging, images)
            path = staging / f"{name}.zip"
            csptree.write_package(path, ini, meshes, textures)
            with zipfile.ZipFile(path) as package:
                invalid = package.testzip()
                if invalid is not None:
                    raise ValueError(f"{path.name}: {invalid} の CRC が一致しません")
            report["zip_bytes"] = path.stat().st_size
            with path.open("rb") as file:
                report["zip_sha256"] = hashlib.file_digest(file, "sha256").hexdigest()
            reports.append(report)
        for report in reports:
            filename = f"{report['package']}.zip"
            (staging / filename).replace(out_dir / filename)
        if audit_path:
            audit_path.parent.mkdir(parents=True, exist_ok=True)
            audit = {"models": reports, "normal_map": "neutral tangent-space RGBA (128,128,255,255)",
                     "alpha": "original source alpha channel; CSP package uses existing alpha handling"}
            audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{out_dir} に CSP の木 {len(reports)} 個体を書き出しました")


def main():
    sys.path.insert(0, str(ROOT))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", type=Path, default=DEFAULT_ASSET, help="書き出す Blender 資産の asset.json")
    parser.add_argument("--out-dir", type=Path, required=True, help="ZIP の出力先（extension/trees）")
    parser.add_argument("--audit", type=Path, help="寸法・三角形数の監査JSONの出力先")
    parser.add_argument("--blender", default=shutil.which("blender") or "/opt/homebrew/bin/blender")
    args = parser.parse_args()
    export(args.asset.resolve(), args.out_dir.resolve(), args.audit.resolve() if args.audit else None, args.blender)


if __name__ == "__main__":
    if "--" in sys.argv:
        asset_path, staging = map(Path, sys.argv[sys.argv.index("--") + 1:])
        extract(asset_path, staging)
    else:
        main()
