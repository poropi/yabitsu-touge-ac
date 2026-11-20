import configparser
import io
import json
import os
import shutil
import struct
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from yabitsu import config, csptree


ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "assets/landmarks/trees_mountain"
BLENDER = shutil.which("blender") or (str(config.BLENDER_EXE) if config.BLENDER_EXE.is_file() else None)
pytestmark = pytest.mark.skipif(os.environ.get("YABITSU_RUN_BLENDER_TESTS") != "1",
                                reason="Blender を使うため YABITSU_RUN_BLENDER_TESTS=1 が必要")


@pytest.fixture(scope="module")
def asset():
    manifest = ASSET_DIR / "asset.json"
    if not manifest.is_file():
        pytest.skip("依頼55の Blender 樹木資産が見つかりません")
    return json.loads(manifest.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def exported(tmp_path_factory, asset):
    if not BLENDER:
        pytest.skip("Blender が見つかりません")
    root = tmp_path_factory.mktemp("mountain_tree_export")
    packages = root / "trees"
    audit = root / "audit.json"
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/export_mountain_trees.py"),
         "--blender", BLENDER, "--out-dir", str(packages), "--audit", str(audit)],
        cwd=ROOT, capture_output=True, text=True, timeout=180,
    )
    return result, packages, audit


def test_exports_one_package_per_model(exported, asset):
    result, packages, audit = exported
    assert result.returncode == 0, result.stdout + result.stderr
    names = {f"mt_{m['id'].lower()}.zip" for m in asset["models"]}
    assert {p.name for p in packages.iterdir()} == names
    assert {f"{m['package']}.zip" for m in json.loads(audit.read_text())["models"]} == names


def test_mesh_round_trip_preserves_materials_and_dimensions(exported, asset):
    result, packages, _ = exported
    assert result.returncode == 0, result.stdout + result.stderr
    for model in asset["models"]:
        with zipfile.ZipFile(packages / f"mt_{model['id'].lower()}.zip") as package:
            ini = configparser.ConfigParser()
            ini.read_string(package.read("tree.ini").decode())
            height = ini.getfloat("BASIC", "HEIGHT")
            half_width = ini.getfloat("BASIC", "WIDTH") / 2
            assert height == pytest.approx(model["height_m"], abs=0.001)
            expected_distance = {"shrub": 300, "roadside": 1500}.get(model["id"].split("_")[0], 4000)
            assert ini.getint("BASIC", "DISTANCE") == expected_distance
            lod1_triangles = {}
            for lod, distance in enumerate((40, 110, 230)):
                assert ini.getint(f"LOD_{lod}", "DISTANCE") == distance
                assert ini[f"LOD_{lod}"]["MODEL"] == f"LOD{lod}.mesh"
                textures, parts = csptree.read_mesh(package.read(f"LOD{lod}.mesh"))
                names = [name for _, name in textures]
                expected_materials = {(model[part].replace(".png", ".dds"),
                                       model[part].replace(".png", "_nm.dds")) for part in ("leaf", "bark")}
                assert {(names[p.diffuse], names[p.normal]) for p in parts} == expected_materials
                for part in parts:
                    assert part.indices.max() < len(part.verts)
                    assert np.isfinite(part.verts["uv"]).all()
                    np.testing.assert_allclose(np.linalg.norm(part.verts["n"], axis=1), 1, atol=1e-5)
                    if lod >= 1:
                        positions = part.verts["p"][part.indices].reshape(-1, 9)
                        triangles_by_position = {tuple(triangle) for triangle in positions}
                        if lod == 1:
                            lod1_triangles[names[part.diffuse]] = triangles_by_position
                        else:
                            assert triangles_by_position < lod1_triangles[names[part.diffuse]]
                triangles = sum(len(p.indices) // 3 for p in parts)
                if lod < 2:
                    assert triangles == model[f"triangles_lod{lod}"]
                    leaf = next(p for p in parts if names[p.diffuse].startswith("leaf_"))
                    assert len(leaf.indices) // 3 == model[f"leaf_triangles_lod{lod}"]
                else:
                    assert 0 < triangles < 0.45 * model["triangles_lod1"]
                pos = np.concatenate([p.verts["p"][p.indices] for p in parts])
                assert np.linalg.norm(pos[:, [0, 2]], axis=1).max() <= 1.0001
                physical = pos[:, [0, 2, 1]] * [half_width, -half_width, height]
                assert physical[:, 2].max() == pytest.approx(height, abs=0.05)
                if lod == 0:
                    np.testing.assert_allclose([physical.min(0), physical.max(0)], model["bounds_m"], atol=0.001)


def test_dds_mips_preserve_source_rgb_and_alpha(exported, asset):
    result, packages, _ = exported
    assert result.returncode == 0, result.stdout + result.stderr
    for model in asset["models"]:
        with zipfile.ZipFile(packages / f"mt_{model['id'].lower()}.zip") as package:
            for part in ("leaf", "bark"):
                png = model[part]
                with Image.open(ASSET_DIR / png) as image:
                    expected = image.convert("RGBA")
                data = package.read(png.replace(".png", ".dds"))
                normal = package.read(png.replace(".png", "_nm.dds"))
                width, height = expected.size
                for dds in (data, normal):
                    assert dds[:4] == b"DDS "
                    assert struct.unpack_from("<8I", dds, 76) == (
                        32, 0x41, 0, 32, 0x00FF0000, 0x0000FF00, 0x000000FF, 0xFF000000,
                    )
                    assert struct.unpack_from("<I", dds, 28)[0] == max(width, height).bit_length()
                    assert len(dds) == 128 + 4 * sum(max(1, width >> k) * max(1, height >> k)
                                                    for k in range(max(width, height).bit_length()))
                with Image.open(io.BytesIO(data)) as loaded:
                    assert loaded.size == expected.size
                    assert loaded.tobytes() == expected.tobytes()
                with Image.open(io.BytesIO(normal)) as loaded:
                    assert loaded.size == expected.size
                    assert loaded.getextrema() == ((128, 128), (128, 128), (255, 255), (255, 255))
