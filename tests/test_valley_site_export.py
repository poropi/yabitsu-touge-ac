import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def source_module():
    try:
        from yabitsu import valley_site_source
    except ImportError:
        pytest.fail("谷の切り出しで使う元メッシュの読取処理が未実装")
    return valley_site_source


def test_coarse_remainders_and_cut_radius_boundary_lock_original_vertex_ids():
    source = source_module()
    vertices = np.array([[10.001, 25., 400.], [11., 25., 400.], [12., 26., 403.], [10., 26., 400.],
                         [15., 25., 400.], [15., 26., 400.], [11., 26., 402.]])
    faces = np.array([[0, 1, 6], [0, 6, 3], [1, 2, 6], [1, 4, 2], [2, 4, 5]])
    near = {
        "valley_vertices": vertices,
        "valley_faces": faces,
        "valley_bed": np.array([False, True, False, False, False]),
        "valley_gravel": np.array([False, False, True, False, False]),
        "valley_source_faces": np.array([41, 42, -1, 44, 45]),
        "valley_coarse_remainder": np.array([[-1, -1, -1], [-1, -1, -1], [5, 6, 1],
                                             [-1, -1, -1], [-1, -1, -1]]),
    }
    terrain = source.terrain_selection(near, np.array([10., 25., 400.]), 3.)
    np.testing.assert_array_equal(terrain["terrain_vertex_ids"], [0, 1, 2, 3, 6])
    np.testing.assert_array_equal(terrain["terrain_face_ids"], [0, 1, 2])
    np.testing.assert_array_equal(terrain["terrain_source_faces"], [41, 42, -1])
    np.testing.assert_array_equal(terrain["terrain_coarse_remainder"][2], [5, 6, 1])
    assert terrain["terrain_vertices"][0, 0] == 10.001
    assert set(terrain["lock"]) == {0, 1, 2, 3, 6}


def test_visible_manifest_mesh_uses_retex_and_explicit_normal(tmp_path):
    source = source_module()
    track = tmp_path / "track"
    dress = tmp_path / "dress_mountain"
    for root in [track, dress]:
        (root / "meshes").mkdir(parents=True)
        (root / "manifest.json").write_text(json.dumps({
            "materials": {"ground": {"texture": "ground.png", "alpha": False, "two_sided": False}},
            "meshes": {name: {"file": f"meshes/{name}.npz", "material": "ground", "renderable": True}
                       for name in ("slope", "hidden")} if root == track else {},
        }))
    verts = np.array([[100., 200., 10.], [101., 200., 10.], [100., 201., 10.]])
    for name in ("slope", "hidden"):
        np.savez(track / "meshes" / f"{name}.npz", verts=verts, faces=[[0, 1, 2]],
                 uvs=[[0, 0], [1, 0], [0, 1]], normals=np.tile([0., 1., 0.], (3, 1)))
    (dress / "hidden.json").write_text('["hidden"]')
    (dress / "retex.json").write_text('{"slope": "retextured"}')
    meshes = list(source.visible_meshes(tmp_path, [100., 200., 10.], 60.))
    assert [mesh.name for mesh in meshes] == ["slope"]
    assert meshes[0].spec["texture"] == str(dress / "extension" / "dress_retextured.dds")
    np.testing.assert_array_equal(meshes[0].world_normals, [[0., 0., 1.]] * 3)


def test_export_refuses_to_overwrite_existing_scene(tmp_path):
    output = tmp_path / "A-s4-1724"
    output.mkdir()
    (output / "site.blend").write_bytes(b"local edits")
    result = subprocess.run(
        [sys.executable, "scripts/valley_site_export.py", "A-s4-1724", "--output-root", str(tmp_path)],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    assert result.returncode != 0
    assert "site.blend" in result.stderr and "上書き" in result.stderr
    assert (output / "site.blend").read_bytes() == b"local edits"
