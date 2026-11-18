import json

import numpy as np

from yabitsu import config, features, gsi, nagoki, start_lot, terrain, track
from yabitsu.crosssection import Section

P = config.DATA_PROCESSED

# 地形を作るときだけ、表で延ばした擁壁・足した歩道を「無いもの」として扱う範囲（区間, s_from, s_to, 側）。擁壁の足元を掘る・歩道の幅だけ
# 道路の範囲を広げる決まりで、道路と側溝の下や坑口の盛り土の縁まで変わってしまうため（2026-11-01、吹風トンネル入口の A：左の擁壁を
# 坑口まで延ばす、E：右の広い路肩。ユーザー「道路を弄る理由が不明」「予定外を出さないように」）
# 2026-11-02、E の 2：橋の手前（10480〜10507 m）の右の擁壁と歩道を釜田川橋まで延ばした所も同じ扱い
TERRAIN_AS_BEFORE = [("s5", 10675.0, 10684.0, "L"), ("s5", 10607.0, 10684.0, "R"), ("s5", 10480.0, 10507.0, "R")]
# 歩道の下の地形だけを、歩道の面より WALK_BELOW 下まで下げる範囲（前の地形が高い点だけ。道路・側溝の下と歩道の外は前のまま）
WALK_FLOOR = [("s5", 10607.0, 10684.0, "R"), ("s5", 10480.0, 10507.0, "R")]
WALK_BELOW = 0.3


def terrain_features(feats, bounds):
    """TERRAIN_AS_BEFORE の範囲で、その手前から延びる擁壁はその範囲の始めで切り、その範囲で始まる歩道は除いた features。"""
    out = list(feats)
    for name, a, b, side in TERRAIN_AS_BEFORE:
        s0 = float(bounds[name][0]); A, B = s0 + a, s0 + b
        nxt = []
        for f in out:
            if f.side != side or f.s_to <= A or f.s_from >= B:
                nxt.append(f)
            elif f.type == "retaining_wall" and f.s_from < A:
                nxt.append(features.Feature(f.s_from, A, f.side, f.type, f.height_m, f.note, f.width_m, f.material))
            elif f.type == "sidewalk" and f.s_from >= A:
                continue
            else:
                nxt.append(f)
        out = nxt
    return out


def lower_under_walks(sec, near, feats, bounds):
    """WALK_FLOOR の範囲で、歩道の下（側溝の外の縁から歩道の幅まで）の地形を歩道の面より WALK_BELOW 下まで下げる。下げた点の数を返す。"""
    walks = features.sidewalk_offsets(sec, feats)
    Z, xs, ys = near["Z"], near["xs"], near["ys"]
    n = 0
    for name, a, b, side in WALK_FLOOR:
        s0 = float(bounds[name][0])
        rows = np.flatnonzero((sec.s > s0 + a) & (sec.s <= s0 + b))
        ix = np.flatnonzero((xs >= sec.x[rows].min() - 20) & (xs <= sec.x[rows].max() + 20))
        iy = np.flatnonzero((ys >= sec.y[rows].min() - 20) & (ys <= sec.y[rows].max() + 20))
        X, Y = np.meshgrid(xs[ix], ys[iy])
        f = terrain.road_frame(sec, X, Y)
        idx, lat = f["idx"].reshape(X.shape), f["lat"].reshape(X.shape)
        sd = features.SIDES[side]
        hw = (sec.hw_left if sd > 0 else sec.hw_right)[idx] + features.GUTTER_WIDTH
        w = np.asarray(walks[0 if sd > 0 else 1])[idx]
        d = lat * sd
        m = np.isin(idx, rows) & (d > hw) & (d <= hw + w)
        sub = Z[np.ix_(iy, ix)]
        new = np.where(m, np.minimum(sub, sec.edge_z(sd)[idx] - WALK_BELOW), sub)
        n += int(np.sum(np.abs(np.nan_to_num(new - sub)) > 1e-9))
        Z[np.ix_(iy, ix)] = new
    return n


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
    bounds = track.load_bounds()
    feats_all, feats = feats, terrain_features(feats, bounds)   # 地形は延ばした擁壁・足した歩道を無いものとして作る
    cut_left, cut_right = features.cut_slopes(sec, feats)
    cut_drop = features.cut_drops(sec, feats)   # 擁壁の裏は地形を下げてから削る（壁の下で地形が手前に出ない）
    sampler = gsi.default_sampler()
    tw = nagoki.load_town(sec)   # 町の範囲：地形は町の地面に合わせる（舗装の下は少し下げる）
    paved = tw.drivable().union(tw.carriageway).union(tw.sidewalk)
    others = nagoki.other_towns(sec)   # 宮ヶ瀬北原など
    from yabitsu import water   # S4 の川・沢：谷の底に溝を彫る（本線の路面の下は彫らない。橋の下は彫る）
    streams, _, _ = water.load(sec, sampler)
    chans = water.channels(sec, streams, start_lot.road_formation(sec, 0.0, float(sec.s[-1])))
    from yabitsu import tunnel   # トンネル：道の上の山を削らない
    tuns = tunnel.from_osm(side_roads.load_elements(), sec)
    keep = tunnel.carve_mask(sec.s, tuns)
    bore = tunnel.row_mask(sec.s, tuns, margin_m=terrain.BORE_MARGIN_M)   # 筒の中に入る地形の面は描かない
    from yabitsu import bridge_parts   # S5 の橋の下は谷を盛らない（湖・深い谷の上の大きな橋）
    no_fill = bridge_parts.no_fill_mask(sec, s_from=float(track.load_bounds()["s5"][0]))
    lake_poly, lake_level = water.load_lake()   # 宮ヶ瀬湖：湖の中の地形は水位より下
    lake = (lake_poly, lake_level) if lake_poly is not None else None
    near = terrain.build_near(sec, sampler, pads, cut_left, cut_right, extra=extra, channels=chans, keep_rows=keep, no_fill_rows=no_fill, lake=lake, bore_rows=bore, cut_drop=cut_drop,
                              side_extra=features.sidewalk_offsets(sec, feats),  # 歩道の下に地形がかぶらない
                              town=[(tw.ground, nagoki.circle(), paved)] + [z.town_override() for z in nagoki.load_splits(sec)]
                              + [(k.ground, nagoki.circle(j), k.drivable().union(k.carriageway).union(k.sidewalk)) for j, k in others])
    print("歩道の下で下げた地形の点", lower_under_walks(sec, near, feats_all, bounds))
    np.savez_compressed(P / "track_terrain_near.npz", **near)
    mid = terrain.build_mid(sec, sampler, pads, channels=chans, keep_rows=keep, no_fill_rows=no_fill, lake=lake)   # 道から 200m まで
    np.savez_compressed(P / "track_terrain_mid.npz", **mid)
    far = terrain.build_far(sec, sampler, extra=extra + [np.asarray(nagoki.circle(j).exterior.coords).tolist()
                                                        for j in [nagoki.NAGOKI] + [j for j, _ in others]], lake=lake)
    np.savez_compressed(P / "track_terrain_far.npz", **far)
    print(f"near grid {near['Z'].shape}, mid grid {mid['Z'].shape}, far grid {far['Z'].shape}")


if __name__ == "__main__":
    main()
