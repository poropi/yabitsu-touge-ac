import shutil

import numpy as np

from yabitsu import acdata, ailine, config, geo, layouts, track
from yabitsu.crosssection import Section

T = config.TRACK_DIR
CREDITS = "出典：国土地理院（標高タイル・シームレス空中写真・基盤地図情報）／© OpenStreetMap contributors (ODbL)"
STALE = ["yabitsu_s1.kn5"]  # 区間をつなぐ前の kn5（もう読まない）


def main():
    sec = Section.load_json(config.DATA_PROCESSED / "track_section.json")
    bounds = track.load_bounds()
    ac = np.array([acdata.blender_to_ac(x, y, z) for x, y, z in zip(sec.x, sec.y, sec.z)])
    mid = sec.n // 2; lat, lon = geo.from_local(sec.x[mid], sec.y[mid])
    for layout in layouts.load_layouts():
        name = layout.name
        acdata.write_surfaces_ini(T / name / "data" / "surfaces.ini")
        acdata.write_models_ini(T / f"models_{name}.ini", layouts.kn5_files(name))   # S4 の kn5 も全部のレイアウトが読む
        acdata.write_map(T / name / "map.png", T / name / "data" / "map.ini", ac[:, [0, 2]])
        acdata.write_cameras_ini(T / name / "data" / "cameras.ini", layouts.cameras(sec, layout))
        ailine.write_fast_lane(T / name / "ai" / "fast_lane.ai", sec, layout.direction)
        acdata.write_ui_track_json(T / "ui" / name / "ui_track.json", name=layout.title,
                                   description="公開データから再現したヤビツ峠（県道70号 秦野清川線）",
                                   length_m=layouts.timed_length(sec, layout, bounds), pitboxes=4,
                                   geotags=(float(lat), float(lon)), tags=["touge", "japan", "point to point"],
                                   credits=CREDITS)
        out = config.assert_writable(T / "ui" / name / "outline.png")
        shutil.copyfile(T / name / "map.png", out)
    for f in STALE:
        p = config.assert_writable(T / f)
        if p.exists():
            p.unlink(); print("removed", p)
    acdata.write_text(T / "read_me.txt", "Yabitsu Touge for Assetto Corsa\n\n" + CREDITS + "\n")
    print("written", T)


if __name__ == "__main__":
    main()
