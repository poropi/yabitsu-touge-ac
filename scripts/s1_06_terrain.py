import json

import numpy as np

from yabitsu import config, features, gsi, terrain
from yabitsu.crosssection import Section

P = config.DATA_PROCESSED


def main():
    sec = Section.load_json(P / "s1_section.json")
    pads = json.loads((P / "s1_pads.json").read_text(encoding="utf-8"))
    cut_left, cut_right = features.cut_slopes(sec, features.load_features_csv(config.DATA_MANUAL / "s1_features.csv"))
    sampler = gsi.default_sampler()
    near = terrain.build_near(sec, sampler, pads, cut_left, cut_right)
    np.savez_compressed(P / "s1_terrain_near.npz", **near)
    far = terrain.build_far(sec, sampler)
    np.savez_compressed(P / "s1_terrain_far.npz", **far)
    print(f"near grid {near['Z'].shape}, far grid {far['Z'].shape}")


if __name__ == "__main__":
    main()
