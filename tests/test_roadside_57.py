"""橋Aの覆いを強める際の配置・対象面・運転席の回帰検査。"""
from types import SimpleNamespace

import numpy as np
import pytest
import shapely

from scripts import make_views_drive_57, roadside_57


def test_grass_only_uses_ground_meshes_named_by_the_mountain_finish():
    meshes = {
        "1GRASS_s1_terrain_050": {"material": "ground", "renderable": True},
        "1GRASS_s1_terrain_051": {"material": "ground", "renderable": False},
        "1GRASS_s1_terrain_052": {"material": "ground", "renderable": True},
        "mtsplit_s1_terrain_051_grass": {"material": "mt_split_ground", "renderable": True},
        "1GRASS_terayama_plaza": {"material": "ground", "renderable": True},
        "1GRASS_valley_A_000_bank": {"material": "valley_forest_floor", "renderable": True},
        "1GRASS_valley_A_000_bed": {"material": "gravel_dark", "renderable": True},
    }
    mountain = set(meshes) - {"1GRASS_s1_terrain_052"}
    assert roadside_57.grass_meshes(meshes, mountain, {"1GRASS_s1_terrain_051"}) == {
        "1GRASS_s1_terrain_050", "mtsplit_s1_terrain_051_grass", "1GRASS_valley_A_000_bank",
    }


def test_greening_keeps_luminance_ratio_and_texture_variation():
    pixels = np.array([[[125, 100, 80], [70, 65, 45]]], dtype=np.uint8)
    from PIL import Image
    original = Image.fromarray(pixels)
    revised = roadside_57.forest_tone(original, .5)
    assert np.mean(np.asarray(revised.convert("L"))) / np.mean(np.asarray(original.convert("L"))) == pytest.approx(.5, abs=.02)
    rgb = np.asarray(revised, dtype=float)
    assert (rgb[..., 1] / rgb[..., 0] > pixels[..., 1] / pixels[..., 0]).all()
    assert np.ptp(rgb[..., 1]) > 5


def test_fern_scale_obeys_height_span_and_the_road_edge():
    model = SimpleNamespace(height=.45, crown_radius=.40)
    low, high = roadside_57.scale_limits(model, (.6, 1.2), np.array([1.5, 2.0]), span=1.0, settings={"max_scale": 2.0})
    assert list(low) == pytest.approx([1.34, 1.34])
    assert list(high) == pytest.approx([2., 2.])
    assert (low * model.crown_radius * 2 >= 1.).all()


def test_trial_reads_fern_height_factor_from_forest_settings(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(roadside_57.roadside_vegetation, "fern_packages",
                        lambda source, dest, factor: calls.append(factor))
    roadside_57.fern_packages(tmp_path, tmp_path, {"fern_height_factor": 1.25})
    assert calls == [1.25]


def test_spacing_takes_existing_plants_into_account():
    candidates = np.array([[0., 0.], [1.3, 0.], [2.6, 0.], [4., 0.]])
    existing = np.array([[.1, 0.]])
    keep = roadside_57.spacing(candidates, existing, np.array([1.2, 1.6, 1.2, 1.2]))
    assert (np.linalg.norm(candidates[keep] - existing[0], axis=1) >= 1.2).all()
    assert np.array_equal(keep, [False, False, True, True])


def test_open_land_overlap_checks_the_triangles():
    tri = np.array([[[0., 0.], [2., 0.], [0., 2.]]])
    zones = shapely.STRtree([shapely.box(1.5, 1.5, 2., 2.)])
    assert not roadside_57.overlaps_open_land(tri, zones)
    zones = shapely.STRtree([shapely.box(.1, .1, .2, .2)])
    assert roadside_57.overlaps_open_land(tri, zones)


def test_capture_uses_left_lanes_and_level_eye_direction():
    section = SimpleNamespace(s=np.array([0., 2000.]), x=[0., 2000.], y=[0., 0.], z=[0., 100.],
                              heading=[0., 0.], hw_left=[4., 4.], hw_right=[3., 3.])
    views = make_views_drive_57.drive_views(section, 0., lambda xy, z: z + .25)
    assert len(views) == 22
    up, down = views[:2]
    assert up["name"] == "s4_1600_up"
    assert up["pos"] == pytest.approx([1600, -1.5, 81.45])
    assert down["pos"] == pytest.approx([1600, 2., 81.45])
    assert up["dir"] == pytest.approx([-1., 0., 0.])
    assert down["dir"] == pytest.approx([1., 0., 0.])
    assert up["horizontal_fov"] == 56


def test_pavement_can_mask_the_goma_loop_without_suppressing_adjacent_slope():
    pad = {"name": "goma_loop_01", "kind": "pass"}
    assert not roadside_57.exclude_grass_pad(pad, {"1ROAD_goma_loop_01"})
    assert roadside_57.exclude_grass_pad(pad, set())
    assert roadside_57.exclude_grass_pad({"name": "nanohanadai_lot", "kind": "lot"}, {"1ROAD_nanohanadai_lot"})
