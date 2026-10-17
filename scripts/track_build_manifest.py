import json

import numpy as np

from yabitsu import assembly, config, features, layouts, minoge_area, pass_area, pavements, start_lot, textures, track, trees
from yabitsu.crosssection import Section
from yabitsu.manifest import Manifest

P = config.DATA_PROCESSED


def main():
    sec = Section.load_json(P / "track_section.json")
    m = Manifest(config.TRACK_BUILD_DIR)
    textures.register_materials(m)
    assembly.add_road(m, sec)
    near = dict(np.load(P / "track_terrain_near.npz")); far = dict(np.load(P / "track_terrain_far.npz"))
    assembly.add_terrain(m, near, far)
    bounds = track.load_bounds()
    assembly.add_features(m, sec, features.load_features_rows(track.load_section_rows("features", bounds)))
    for name, v, f, uv, mat in pavements.meshes(sec, pavements.load_rows(track.load_section_rows("pavements", bounds))):
        m.add_mesh(assembly.GROUP, name, v, f, uv, mat)
    pads = json.loads((P / "track_pads.json").read_text(encoding="utf-8"))
    lot_spec = config.DATA_MANUAL / "s1_start_lot.json"
    # 駐車場や広場には、木の板（幅 最大7m）がかぶらないよう少し広めに空ける
    assembly.add_trees(m, trees.scatter(near, pads + start_lot.flats_for(lot_spec, pads), pad_clear_m=5.0))
    assembly.add_pads(m, pads)
    assembly.add_start_lot(m, start_lot.load_spec(lot_spec), pads)
    site_spec = pass_area.load(config.DATA_MANUAL / "s1_pass.json")
    if site_spec:
        assembly.add_pass_area(m, pass_area.build(site_spec, sec))
    minoge = minoge_area.load(config.DATA_MANUAL / "s2_minoge.json")
    if minoge:
        assembly.add_pass_area(m, minoge_area.build(minoge, sec, minoge_area.dem_ground()))
    assembly.add_layout_objects(m, sec, pads, layouts.load_layouts(), bounds)
    print(m.save()); print(m.summary())


if __name__ == "__main__":
    main()
