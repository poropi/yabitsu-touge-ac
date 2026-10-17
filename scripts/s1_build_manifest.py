import json

import numpy as np

from yabitsu import assembly, config, features, pass_area, start_lot, textures, trees
from yabitsu.crosssection import Section
from yabitsu.manifest import Manifest

P = config.DATA_PROCESSED


def main():
    sec = Section.load_json(P / "s1_section.json")
    m = Manifest(config.S1_DIR)
    textures.register_materials(m)
    assembly.add_road(m, sec)
    near = dict(np.load(P / "s1_terrain_near.npz")); far = dict(np.load(P / "s1_terrain_far.npz"))
    assembly.add_terrain(m, near, far)
    assembly.add_features(m, sec, features.load_features_csv(config.DATA_MANUAL / "s1_features.csv"))
    pads = json.loads((P / "s1_pads.json").read_text(encoding="utf-8"))
    lot_spec = config.DATA_MANUAL / "s1_start_lot.json"
    # 駐車場や広場には、木の板（幅 最大7m）がかぶらないよう少し広めに空ける
    assembly.add_trees(m, trees.scatter(near, pads + start_lot.flats_for(lot_spec, pads), pad_clear_m=5.0))
    assembly.add_pads(m, pads)
    assembly.add_start_lot(m, start_lot.load_spec(lot_spec), pads)
    site_spec = pass_area.load(config.DATA_MANUAL / "s1_pass.json")
    marks = None
    if site_spec:
        assembly.add_pass_area(m, pass_area.build(site_spec, sec))
        marks = pass_area.layout_marks(site_spec, sec, config.S1_PASS)
    assembly.add_layout_objects(m, sec, pads, marks)
    print(m.save()); print(m.summary())


if __name__ == "__main__":
    main()
