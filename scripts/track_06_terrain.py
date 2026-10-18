import json

import numpy as np

from yabitsu import config, features, gsi, nagoki, start_lot, terrain, track
from yabitsu.crosssection import Section

P = config.DATA_PROCESSED


def main():
    sec = Section.load_json(P / "track_section.json")
    pads = json.loads((P / "track_pads.json").read_text(encoding="utf-8"))
    flats = start_lot.flats_for(config.DATA_MANUAL / "s1_start_lot.json", pads)
    pads = pads + flats
    extra = [p["polygon"] for p in flats]  # 展望台の広場など、道から離れた場所にも細かい地形を張る
    from yabitsu import motorway, overpasses, side_roads
    for x in motorway.from_osm(side_roads.load_elements(), sec, gsi.default_sampler(), overpasses.load_overrides()):
        extra.append(np.asarray(x.footprint().buffer(15.0).exterior.coords).tolist())   # 新東名のまわりも細かい地形に
    feats = features.load_features_rows(track.load_section_rows("features", track.load_bounds()))
    _, feats = nagoki.track_stubs(sec, pads, feats)   # 分かれ道の入口では壁の裏の切り土もしない
    cut_left, cut_right = features.cut_slopes(sec, feats)
    sampler = gsi.default_sampler()
    tw = nagoki.load_town(sec)   # 町の範囲：地形は町の地面に合わせる（舗装の下は少し下げる）
    paved = tw.drivable().union(tw.carriageway).union(tw.sidewalk)
    from yabitsu import water   # S4 の川・沢：谷の底に溝を彫る（本線の路面の下は彫らない。橋の下は彫る）
    streams, _, _ = water.load(sec, sampler)
    chans = water.channels(sec, streams, start_lot.road_formation(sec, 0.0, float(sec.s[-1])))
    from yabitsu import tunnel   # トンネル：道の上の山を削らない
    keep = tunnel.carve_mask(sec.s, tunnel.from_osm(side_roads.load_elements(), sec))
    from yabitsu import bridge_parts   # S5 の橋の下は谷を盛らない（湖・深い谷の上の大きな橋）
    no_fill = bridge_parts.no_fill_mask(sec, s_from=float(track.load_bounds()["s5"][0]))
    lake_poly, lake_level = water.load_lake()   # 宮ヶ瀬湖：湖の中の地形は水位より下
    lake = (lake_poly, lake_level) if lake_poly is not None else None
    near = terrain.build_near(sec, sampler, pads, cut_left, cut_right, extra=extra, channels=chans, keep_rows=keep, no_fill_rows=no_fill, lake=lake,
                              side_extra=features.sidewalk_offsets(sec, feats),  # 歩道の下に地形がかぶらない
                              town=[(tw.ground, nagoki.circle(), paved)] + [z.town_override() for z in nagoki.load_splits(sec)])
    np.savez_compressed(P / "track_terrain_near.npz", **near)
    mid = terrain.build_mid(sec, sampler, pads, channels=chans, keep_rows=keep, no_fill_rows=no_fill, lake=lake)   # 道から 200m まで
    np.savez_compressed(P / "track_terrain_mid.npz", **mid)
    far = terrain.build_far(sec, sampler, extra=extra + [np.asarray(nagoki.circle().exterior.coords).tolist()], lake=lake)
    np.savez_compressed(P / "track_terrain_far.npz", **far)
    print(f"near grid {near['Z'].shape}, mid grid {mid['Z'].shape}, far grid {far['Z'].shape}")


if __name__ == "__main__":
    main()
