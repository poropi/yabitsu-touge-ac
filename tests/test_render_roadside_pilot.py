import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

from scripts.render_roadside_pilot import foreground_roadside_mask, model_pose
from yabitsu import config


def test_roadside_mask_counts_only_dominantly_visible_ground_outside_the_pavement():
    # Straight road towards +X. The pavement is 3 m wide on the left and 2 m on the right.
    road = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]])
    widths = np.array([[3.0, 2.0]] * 3)
    # Channels store coverage-weighted world X and Y, then ground coverage.
    aov = np.array([
        [[5, 3, 1], [5, -14, 1], [5, 15.01, 1], [5, -1.99, 1]],
        [[2.5, 7.5, .5], [2, 2, .4], [4.1, -3.28, .82], [0, 0, 0]],
    ])
    np.testing.assert_array_equal(foreground_roadside_mask(aov, road, widths), [
        [1, -1, 0, 0],
        [0, 0, -1, 0],
    ])


def test_model_pose_adds_the_packaged_orientation_without_changing_the_source_angle():
    assert model_pose("mt_sugi_edge_a_r15", 5) == ("mt_sugi_edge_a", 342.5)
    assert model_pose("mt_roadside_b", 147) == ("mt_roadside_b", 147)
    assert model_pose("grass_a", 36) == ("grass_a", 36)


def test_current_and_prototype_shrubs_with_the_same_base_name_keep_distinct_models():
    assert model_pose("mt_shrub_a", 0) != model_pose("shrub_a", 0)
    assert model_pose("mt_hinoki_b", 0) != model_pose("hinoki_b", 0)


@pytest.mark.skipif(os.environ.get("YABITSU_RUN_BLENDER_TESTS") != "1",
                    reason="Blender を使うため YABITSU_RUN_BLENDER_TESTS=1 が必要")
def test_eevee_pass_retains_the_ground_position_only_where_it_is_visible(tmp_path):
    scripts = Path(__file__).resolve().parents[1] / "blender_scripts"
    expression = f'''
import sys
import os
from pathlib import Path
import bpy
sys.path.insert(0, {str(scripts)!r})
from render_roadside_pilot import ground_aov, render_with_ground_aov
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
bpy.ops.mesh.primitive_plane_add(size=4)
ground = bpy.context.object
ground.data.materials.append(bpy.data.materials.new('Ground'))
bpy.ops.mesh.primitive_plane_add(size=2, location=(-1, 0, 1))
bpy.ops.object.camera_add(location=(0, 0, 3))
scene.camera = bpy.context.object
scene.camera.data.type = 'ORTHO'
scene.camera.data.ortho_scale = 2
scene.render.engine = 'BLENDER_EEVEE'
scene.eevee.taa_render_samples = 1
scene.render.resolution_x = scene.render.resolution_y = 64
ground_aov([ground])
render_with_ground_aov(scene, Path({str(tmp_path / 'test')!r}))
'''
    if not config.BLENDER_EXE.is_file():
        pytest.skip(f"Blender が見つかりません: {config.BLENDER_EXE}")
    result = subprocess.run([str(config.BLENDER_EXE), "-b", "--factory-startup", "--python-exit-code", "1",
                             "--python-expr", expression], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    aov = np.load(tmp_path / "test.npy")
    assert np.all(aov[8:-8, 8:24, 2] == 0)
    assert np.all(aov[8:-8, 40:56, 2] == 1)
    assert np.all((aov[8:-8, 40:56, 0] > .25) & (aov[8:-8, 40:56, 0] < .75))
