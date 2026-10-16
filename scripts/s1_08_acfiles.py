import shutil

import numpy as np

from yabitsu import acdata, config, geo, s1_layouts
from yabitsu.crosssection import Section

T = config.TRACK_DIR
CREDITS = "出典：国土地理院（標高タイル・シームレス空中写真）／© OpenStreetMap contributors (ODbL)"
TITLES = {"s1_up": "ヤビツ峠 S1 菜の花台→ヤビツ峠（上り）", "s1_down": "ヤビツ峠 S1 ヤビツ峠→菜の花台（下り）"}


def main():
    sec = Section.load_json(config.DATA_PROCESSED / "s1_section.json")
    ac = np.array([acdata.blender_to_ac(x, y, z) for x, y, z in zip(sec.x, sec.y, sec.z)])
    mid = sec.n // 2; lat, lon = geo.from_local(sec.x[mid], sec.y[mid])
    for name, direction in s1_layouts.LAYOUTS.items():
        acdata.write_surfaces_ini(T / name / "data" / "surfaces.ini")
        acdata.write_models_ini(T / f"models_{name}.ini", ["yabitsu_s1.kn5", f"{name}_objects.kn5"])
        acdata.write_map(T / name / "map.png", T / name / "data" / "map.ini", ac[:, [0, 2]])
        acdata.write_cameras_ini(T / name / "data" / "cameras.ini", s1_layouts.cameras(sec, direction))
        acdata.write_ui_track_json(T / "ui" / name / "ui_track.json", name=TITLES[name],
                                   description="公開データから再現したヤビツ峠（県道70号 秦野清川線）", length_m=float(sec.s[-1]),
                                   pitboxes=4, geotags=(float(lat), float(lon)), tags=["touge", "japan", "point to point"],
                                   credits=CREDITS)
        out = config.assert_writable(T / "ui" / name / "outline.png")
        shutil.copyfile(T / name / "map.png", out)
    acdata.write_text(T / "read_me.txt", "Yabitsu Touge (S1) for Assetto Corsa\n\n" + CREDITS + "\n")
    print("written", T)


if __name__ == "__main__":
    main()
