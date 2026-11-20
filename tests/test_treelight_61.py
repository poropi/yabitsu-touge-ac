"""依頼61の配置規則とKN5の書き出し。"""
from importlib import import_module
from pathlib import Path
import math

import numpy as np
from PIL import Image
import pytest
import shapely

from yabitsu import kn5


ASSETS = Path("assets/landmarks/trees_light/export/models.json")


def tree_light():
    return import_module("yabitsu.forest_light")


def straight_edge():
    return shapely.STRtree(shapely.linestrings([[[-500, 0], [500, 0]]]))


def test_near_keeps_roots_and_faces_edge_with_bounded_scale(light_settings):
    light = tree_light()
    models, _ = light.load_assets(ASSETS)
    xyz = np.array([[0, 12, 100], [8, -15, 105], [12, 5, 106]])
    trees = light.Trees(xyz, np.array([0, 1, 2]), np.array([1., 100., 20.]))
    placed = light.place_near(trees, straight_edge(), models, light_settings)

    np.testing.assert_array_equal([item.xyz for item in placed], xyz)
    assert [item.model.rsplit("_", 1)[0] for item in placed] == [
        "near_sugi", "near_hinoki", "near_broadleaf"
    ]
    assert [item.scale for item in placed[:2]] == [0.35, 1.3]
    assert placed[2].scale == pytest.approx(20 / models[placed[2].model].height)
    for item, target in zip(placed, [-math.pi / 2, math.pi / 2, -math.pi / 2]):
        assert math.cos(item.angle - target) >= math.cos(math.radians(25))
    reordered = light.place_near(
        light.Trees(xyz[::-1], trees.family[::-1], trees.height[::-1]), straight_edge(), models
    )
    assert [(p.model, p.angle) for p in placed] == [(p.model, p.angle) for p in reversed(reordered)]


def test_far_grid_votes_centroid_root_median_height_quartile_and_angle(light_settings):
    light = tree_light()
    models, _ = light.load_assets(ASSETS)
    xyz = np.array([
        [1, 19, 100], [5, 21, 102], [8, 23, 800], [2, 25, 104],
        [10, 19, 91], [12, 19, 95],
        [19, 19, 100], [20, 19, 100],
        [28, 19, 100], [29, 19, 100], [30, 19, 100],
        [37, 19, 100], [38, 19, 100],
    ])
    family = np.array([0, 0, 0, 1, 0, 1, 0, 2, 0, 1, 1, 2, 2])
    heights = np.array([8, 18, 26, 30] + [20] * 9)
    plates, counts, sparse = light.place_far(light.Trees(xyz, family, heights), straight_edge(), models, light_settings)

    assert counts == {9: 5, 14: 0, 24: 0}
    assert sparse == {1: 0, 2: 3}
    assert [p.model for p in plates[:4]] == [
        "far_sugi_a", "far_conifer_mix", "far_mixed_a", "far_hinoki_a"
    ]
    assert plates[4].model in {"far_broadleaf_a", "far_broadleaf_b"}
    np.testing.assert_allclose(plates[0].xyz, [4, 22, 103])
    assert plates[0].scale == pytest.approx(27 / 22)
    assert [p.angle for p in plates[:2]] == pytest.approx([math.radians(32), math.radians(-32)])


def test_far_grid_distance_boundaries_and_width(light_settings):
    light = tree_light()
    edge = straight_edge()
    models, _ = light.load_assets(ASSETS)
    xyz = np.array([[1, 150, 80], [1, 150.01, 82], [1, 600, 83], [1, 600.01, 84]])
    trees = light.Trees(xyz, np.zeros(4, dtype=int), np.array([1., 22., 22., 100.]))
    plates, counts, sparse = light.place_far(trees, edge, models, light_settings)
    assert counts == {9: 1, 14: 2, 24: 1}
    assert sparse == {1: 1, 2: 0}
    assert [p.scale for p in plates] == [0.7, 1, 1, 1.4]
    assert [p.width_factor for p in plates] == [1, 1, 1, 1.6]


def test_batches_split_by_kind_grid_and_actual_vertex_count(light_settings):
    light = tree_light()
    models, _ = light.load_assets(ASSETS)
    near = light.Placement("near_broadleaf_a", np.array([1, 1, 1]), 0., 1.)
    far = light.Placement("far_sugi_a", np.array([1, 1, 1]), 0., 1.)
    next_cell = light.Placement("near_sugi_a", np.array([-0.01, 1, 1]), 0., 1.)
    batches = light.mesh_batches([near] * 177 + [far, next_cell], models, light_settings)
    assert [name for name, _ in batches] == [
        "treelight_near_-1_0_00", "treelight_near_0_0_00", "treelight_near_0_0_01",
        "treelight_far_0_0_00"
    ]
    assert [len(items) for _, items in batches] == [1, 176, 1, 1]
    for _, items in batches:
        pos, _, _, _, faces = light.geometry(items, models)
        assert faces.max() < len(pos) <= 65535


@pytest.mark.parametrize(("options", "shadow_flags", "lighting"), [
    ({}, [False, False], (0.5, 0.6)),
    ({"shadows": "all"}, [True, True], (0.5, 0.6)),
    ({"shadows": "all", "normals": "up", "ambient": 0.45, "diffuse": 0.35},
     [True, True], (0.45, 0.35)),
    ({"shadows": "near", "normals": "up", "ambient": 0.45, "diffuse": 0.35},
     [True, False], (0.45, 0.35)),
])
def test_kn5_keeps_npz_faces_axes_uv_and_atlas(tmp_path, options, shadow_flags, lighting, light_settings):
    light = tree_light()
    models, _ = light.load_assets(ASSETS)
    near = light.Placement("near_sugi_a", np.array([50, 100, 120]), math.pi / 2, 1.)
    far = light.Placement("far_sugi_a", np.array([50, 105, 120]), 0., 1., 1.6)
    batches = light.mesh_batches([near, far], models, light_settings)
    images = [Image.new("RGBA", (4, 4), (*rgb, 100)) for rgb in [(10, 20, 30), (2, 4, 6)]]
    paths = [tmp_path / f"trees_{color}.kn5" for color in ("middle", "dark")]
    textures = {path: (path.stem + ".dds", kn5.dds_bytes(image)) for path, image in zip(paths, images)}
    light.write_kn5s(textures, batches, models, **options)
    middle, dark = [kn5.read_kn5(path) for path in paths]
    for path, decoded in zip(paths, (middle, dark)):
        name, dds = textures[path]
        assert decoded["textures"] == {name: dds}
        mat, = decoded["materials"]
        assert mat["shader"] == "ksPerPixelAT"
        assert mat["alpha_tested"] and mat["props"]["ksAlphaRef"] == pytest.approx(0.30)
        assert (mat["props"]["ksAmbient"], mat["props"]["ksDiffuse"]) == pytest.approx(lighting)
        assert mat["textures"] == [("txDiffuse", 0, name)]
        for mesh, placed, triangle_count, shadow in zip(
            decoded["root"]["children"], [near, far], [244, 4], shadow_flags, strict=True
        ):
            assert mesh["visible"] and mesh["renderable"] and mesh["cast_shadows"] == shadow
            normals = mesh["vertices"][:, 3:6]
            if options.get("normals") == "up":
                np.testing.assert_array_equal(normals, np.broadcast_to([0, 1, 0], normals.shape))
            else:
                assert np.any(normals[:, 1] != 1)
            assert mesh["lod"] == (0., 0.)
            with np.load(ASSETS.parent / (placed.model + ".npz")) as source:
                np.testing.assert_array_equal(mesh["indices"].reshape(-1, 3), source["faces"])
                assert len(mesh["indices"]) == 3 * triangle_count
                np.testing.assert_allclose(mesh["vertices"][:, 6:8], source["uvs"] * [1, -1] + [0, 1])
                if placed is near:
                    x, y, z = source["verts"].T
                    expected = np.column_stack((50 - y, 120 + z, -100 - x))
                    np.testing.assert_allclose(mesh["vertices"][:, :3], expected, atol=1e-5)
                else:
                    assert np.ptp(mesh["vertices"][:, 0]) == pytest.approx(19.2)


def test_oversized_model_is_rejected_before_creating_an_empty_batch(light_settings):
    light = tree_light()
    item = light.Placement("near_test", np.zeros(3), 0., 1.)
    model = light.Model(1., np.zeros((65536, 3)), np.zeros((65536, 2)), np.array([[0, 1, 2]]))
    with pytest.raises(ValueError, match=r"near_test.*65.?536"):
        light.mesh_batches([item], {item.model: model}, light_settings)


def test_failed_kn5_write_cleans_temporary_files_and_preserves_previous_outputs(tmp_path, monkeypatch):
    light = tree_light()
    paths = [tmp_path / name for name in ("first.kn5", "second.kn5")]
    paths[0].write_bytes(b"previous KN5")

    def invalid_geometry(*args):
        raise RuntimeError("geometry failed")

    monkeypatch.setattr(light, "geometry", invalid_geometry)
    with pytest.raises(RuntimeError, match="geometry failed"):
        light.write_kn5s({path: ("sample.dds", b"sample") for path in paths}, [("mesh", [])], {})
    assert paths[0].read_bytes() == b"previous KN5"
    assert not paths[1].exists()
    assert not list(tmp_path.glob("*.kn5.tmp"))
