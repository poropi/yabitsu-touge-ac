import json

import numpy as np

from yabitsu import assembly, config, features, textures, trees
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
    assembly.add_trees(m, trees.scatter(near, pads))
    print(m.save()); print(m.summary())


if __name__ == "__main__":
    main()
