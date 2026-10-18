import numpy as np

from yabitsu import acdata, ailine, config, geo, layouts, track
from yabitsu.crosssection import Section

T = config.TRACK_DIR
CREDITS = "出典：国土地理院（標高タイル・シームレス空中写真・基盤地図情報）／© OpenStreetMap contributors (ODbL)"
STALE = ["yabitsu_s1.kn5"]  # 区間をつなぐ前の kn5（もう読まない）


def _photo(x0, y0, w_m, h_m, w_px, h_px):
    """空中写真（地理院のシームレス写真）の、x0, y0 から w_m × h_m の範囲を w_px × h_px で。"""
    import math
    from yabitsu import gsi, photo
    px = w_m / w_px
    z = int(min(17, max(12, round(math.log2(156543.0 * math.cos(math.radians(35.43)) / px)))))
    img = photo.render_local_image(gsi.TileCache(), geo.from_local, x0, y0, w_m, px, z=z)   # y0 から北へ w_m 四方（上が北）
    return img[img.shape[0] - h_px:]   # 下（南）の h_m ぶん＝ y0 〜 y0 + h_m


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
        # コース選択画面：そのレイアウトで走る道だけの outline.png と、同じ枠の空中写真 preview.png
        s0, s1 = sorted((layouts.point_s(sec, layout.grid, bounds, layout.section), layouts.point_s(sec, layout.finish, bounds, layout.section)))
        m = (sec.s >= s0) & (sec.s <= s1)
        acdata.write_ui_images(T / "ui" / name, np.c_[sec.x[m], sec.y[m]], _photo)
    for f in STALE:
        p = config.assert_writable(T / f)
        if p.exists():
            p.unlink(); print("removed", p)
    acdata.write_text(T / "read_me.txt", "Yabitsu Touge for Assetto Corsa\n\n" + CREDITS + "\n")
    print("written", T)


if __name__ == "__main__":
    main()
