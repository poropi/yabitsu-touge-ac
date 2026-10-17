import math

import numpy as np

from yabitsu import acdata, ailine, config
from yabitsu.crosssection import Section

T = config.TEST_TRACK_DIR


def main():
    acdata.write_surfaces_ini(T / "a" / "data" / "surfaces.ini")
    acdata.write_models_ini(T / "models_a.ini", ["yabitsu_test.kn5", "test_objects.kn5"])
    ac = np.array([acdata.blender_to_ac(0.0, y, 0.0) for y in np.linspace(0, 200, 41)])
    acdata.write_map(T / "a" / "map.png", T / "a" / "data" / "map.ini", ac[:, [0, 2]])
    cam = acdata.Camera("side", acdata.blender_to_ac(12.0, 100.0, 3.0), acdata.blender_to_ac(-12.0, 0.0, -3.0), 0.0, 1.0)
    acdata.write_cameras_ini(T / "a" / "data" / "cameras.ini", [cam])
    acdata.write_ui_track_json(T / "ui" / "a" / "ui_track.json", name="Yabitsu TEST",
                               description="パイプライン確認用の試験コース", length_m=200, pitboxes=2,
                               geotags=(35.42782, 139.21934), tags=["test"], credits="")
    y = np.arange(0.0, 201.0); n = len(y); z0 = np.zeros(n)
    road = Section(y, z0, y, z0, np.full(n, math.pi / 2), z0, np.full(n, 2, np.int32), np.zeros(n, bool),
                   np.full(n, 3.0), np.full(n, 3.0), z0, z0)
    ailine.write_fast_lane(T / "a" / "ai" / "fast_lane.ai", road, 1)
    print("written", T)


if __name__ == "__main__":
    main()
