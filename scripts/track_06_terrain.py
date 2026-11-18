import json

import numpy as np

from yabitsu import config, features, fubuki, gsi, nagoki, start_lot, terrain, track
from yabitsu.crosssection import Section

P = config.DATA_PROCESSED

# 地形を作るときだけ、表で延ばした擁壁・足した歩道を「無いもの」として扱う範囲（区間, s_from, s_to, 側）。擁壁の足元を掘る・歩道の幅だけ
# 道路の範囲を広げる決まりで、道路と側溝の下や坑口の盛り土の縁まで変わってしまうため（2026-11-01、吹風トンネル入口の A：左の擁壁を
# 坑口まで延ばす、E：右の広い路肩。ユーザー「道路を弄る理由が不明」「予定外を出さないように」）
# 2026-11-02、E の 2：橋の手前（10480〜10507 m）の右の擁壁と歩道を釜田川橋まで延ばした所も同じ扱い
TERRAIN_AS_BEFORE = [("s5", 10675.0, 10684.0, "L"), ("s5", 10607.0, 10684.0, "R"), ("s5", 10480.0, 10507.0, "R"),
                     ("s5", 10507.0, 10607.0, "R")]  # 橋の歩道の下は谷のまま
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


def shape_fubuki_gate(sec, near, feats, bounds, pads=()):
    """門前の高い地面だけを下げ、分かれ道にかかる三角形を路面より下に置く。"""
    start, end = float(bounds["s5"][0]) + np.array([10405.0, 10435.0])
    foot = sec.hw_right + features.GUTTER_WIDTH + features.sidewalk_offsets(sec, feats)[1]
    level = sec.edge_z(-1) + features.CURB_HEIGHT - 0.03

    def apron(z, idx, s, lat):
        d = -lat
        along = np.clip(np.minimum(s - start, end - s) / 2.0, 0, 1)
        across = np.clip((14.0 - d) / 3.0, 0, 1)
        w = np.where(d > foot[idx], along * across, 0)
        return np.where(w > 0, z - w * np.maximum(z - level[idx], 0), z)

    terrain.edit_roadside_grid(sec, near, start, end, 16.0, apron)
    stub = [p for p in pads if p.get("name") == "side_stub_595991910_0" and p.get("kind") == "stub"]
    if stub:
        terrain.lower_paved_cells(near["Z"], near["xs"], near["ys"], stub, 0.0)


def raise_fubuki_slope(sec, near, feats, bounds):
    """延ばした右擁壁の裏だけを盛り、橋の手前に法面の上端をつくる。"""
    s5 = float(bounds["s5"][0])
    foot = sec.hw_right + features.GUTTER_WIDTH + features.sidewalk_offsets(sec, feats)[1]
    edge = sec.edge_z(-1)
    for _, wall in fubuki.walls_after_gate(feats, s5):
        slope_base = foot + features.WALL_LEAN * wall.height_m
        base_z = edge + wall.height_m

        def fill(z, idx, s, lat):
            d = -lat
            crest = fubuki.crest_offset(s - s5)
            up = np.clip((d - slope_base[idx]) / (crest - slope_base[idx] - 0.5), 0, 1)
            target = base_z[idx] + up * (sec.z[idx] + 6.5 - base_z[idx])
            along = np.clip(np.minimum((s - (s5 + 10485)) / 8.0, (wall.s_to - s) / 2.5), 0, 1)
            across = np.clip((crest + 8.0 - d) / 6.0, 0, 1)
            w = np.where((s >= wall.s_from) & (d >= slope_base[idx]), along * across, 0)
            return z + w * np.maximum(target - z, 0)

        if wall.s_to > s5 + 10485:
            terrain.edit_roadside_grid(sec, near, max(wall.s_from, s5 + 10485), wall.s_to, 24.0, fill)
        if wall.s_from >= s5 + 10450:
            continue

        def fill_after_gate(z, idx, s, lat):
            d = -lat
            base = slope_base[idx]
            crest = fubuki.crest_offset(s - s5)
            up = np.clip((d - base) / (crest - base), 0, 1)
            target = edge[idx] + wall.height_m + up * (5.5 - wall.height_m)
            along = np.clip(np.minimum((s - wall.s_from) / 2.0, (s5 + 10452 - s) / 5.0), 0, 1)
            # 白い家の横では壁のすぐ裏だけを盛り、床を過ぎてから法面を広げる。
            reach = np.interp(s - s5, [10442, 10445], [7.25, 16.0])
            across = np.clip(reach - d, 0, 1)
            w = np.where(d >= base, along * across, 0)
            return z + w * np.maximum(target - z, 0)

        terrain.edit_roadside_grid(sec, near, wall.s_from, s5 + 10452, 17.0, fill_after_gate)


def clear_fubuki_wall_face(sec, near, feats, bounds):
    """側溝の外だけを削り、地形の粗い三角形が擁壁の手前に出ないようにする。"""
    s5 = float(bounds["s5"][0])
    gutter = sec.hw_right + features.GUTTER_WIDTH
    foot = gutter + features.sidewalk_offsets(sec, feats)[1]
    edge = sec.edge_z(-1)

    for _, wall in fubuki.walls_after_gate(feats, s5):
        start = max(wall.s_from, s5 + 10480)
        if start >= wall.s_to:
            continue
        lean = features.WALL_LEAN * wall.height_m
        rise = wall.height_m + features.WALL_BASE_DROP

        def cut(z, idx, s, lat):
            d = -lat
            from_foot = d - foot[idx]
            # 壁面の延長より上にある格子点は、壁を横切る三角形を手前へ出す。
            # 天端より奥では、その三角形を支える範囲だけを調べ、壁面の延長より低い盛り土を保つ。
            face_z = edge[idx] - features.WALL_BASE_DROP + rise * from_foot / lean
            ceiling = np.where(from_foot <= lean, np.minimum(face_z, edge[idx] - features.WALL_BASE_DROP), face_z) - 0.12
            w = np.clip((s - start) / 2.0, 0, 1)
            active = ((s <= wall.s_to) & (d > gutter[idx])
                      & ((from_foot <= lean) | ((from_foot <= lean + terrain.NEAR_CELL * np.sqrt(2)) & (z > face_z))))
            return np.where(active, z - w * np.maximum(z - ceiling, 0), z)

        terrain.edit_roadside_grid(sec, near, start, wall.s_to, 14.0, cut)


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
    shape_fubuki_gate(sec, near, feats_all, bounds, pads)
    raise_fubuki_slope(sec, near, feats_all, bounds)
    clear_fubuki_wall_face(sec, near, feats_all, bounds)
    np.savez_compressed(P / "track_terrain_near.npz", **near)
    mid = terrain.build_mid(sec, sampler, pads, channels=chans, keep_rows=keep, no_fill_rows=no_fill, lake=lake)   # 道から 200m まで
    np.savez_compressed(P / "track_terrain_mid.npz", **mid)
    far = terrain.build_far(sec, sampler, extra=extra + [np.asarray(nagoki.circle(j).exterior.coords).tolist()
                                                        for j in [nagoki.NAGOKI] + [j for j, _ in others]], lake=lake)
    np.savez_compressed(P / "track_terrain_far.npz", **far)
    print(f"near grid {near['Z'].shape}, mid grid {mid['Z'].shape}, far grid {far['Z'].shape}")


if __name__ == "__main__":
    main()
