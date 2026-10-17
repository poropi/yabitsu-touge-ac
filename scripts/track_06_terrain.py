import json

import numpy as np

from yabitsu import config, features, gsi, start_lot, terrain, track
from yabitsu.crosssection import Section

P = config.DATA_PROCESSED


def main():
    sec = Section.load_json(P / "track_section.json")
    pads = json.loads((P / "track_pads.json").read_text(encoding="utf-8"))
    flats = start_lot.flats_for(config.DATA_MANUAL / "s1_start_lot.json", pads)
    pads = pads + flats
    extra = [p["polygon"] for p in flats]  # 展望台の広場など、道から離れた場所にも細かい地形を張る
    feats = features.load_features_rows(track.load_section_rows("features", track.load_bounds()))
    cut_left, cut_right = features.cut_slopes(sec, feats)
    sampler = gsi.default_sampler()
    near = terrain.build_near(sec, sampler, pads, cut_left, cut_right, extra=extra,
                              side_extra=features.sidewalk_offsets(sec, feats))  # 歩道の下に地形がかぶらない
    np.savez_compressed(P / "track_terrain_near.npz", **near)
    far = terrain.build_far(sec, sampler, extra=extra)
    np.savez_compressed(P / "track_terrain_far.npz", **far)
    print(f"near grid {near['Z'].shape}, far grid {far['Z'].shape}")


if __name__ == "__main__":
    main()
