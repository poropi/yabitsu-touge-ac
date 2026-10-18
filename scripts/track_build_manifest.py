import json

import numpy as np
import shapely
from shapely.geometry import LineString

from yabitsu import (assembly, buildings, config, features, gsi, layouts, minoge_area, motorway, nagoki, overpasses, pass_area,
                     pavements, scenery, side_roads, start_lot, terrain, textures, town_objects, track, trees)
from yabitsu.crosssection import Section
from yabitsu.manifest import Manifest

P = config.DATA_PROCESSED


def main():
    sec = Section.load_json(P / "track_section.json")
    m = Manifest(config.TRACK_BUILD_DIR)
    textures.register_materials(m)
    tw = nagoki.load_town(sec)
    others = nagoki.other_towns(sec)   # 宮ヶ瀬北原など（町の範囲は多角形で作る）
    sps = nagoki.load_splits(sec)   # 島をはさんで2本に分かれる所（地図の線どおり）
    assembly.add_road(m, sec, skip=[tw.main_skip] + [k.main_skip for _, k in others] + [z.main_skip for z in sps])   # 町の範囲と分かれる所は多角形で作る
    near = dict(np.load(P / "track_terrain_near.npz")); far = dict(np.load(P / "track_terrain_far.npz"))
    mid = dict(np.load(P / "track_terrain_mid.npz"))
    assembly.add_terrain(m, near, far, mid)
    ground = terrain.terrain_z(near, mid)   # 景色の建物・道は、できあがった地形の上に置く
    bounds = track.load_bounds()
    assembly.add_centrelines(m, sec, track.load_section_rows("centrelines", bounds), s_min=tw.main_skip[1],
                             gaps=[z.main_skip for z in sps] + [k.main_skip for _, k in others])
    from yabitsu import crosssection as cs
    assembly.add_bay_lines(m, sec, cs.parse_bay_rows(track.load_section_rows("busbays", bounds)))
    pads0 = json.loads((P / "track_pads.json").read_text(encoding="utf-8"))
    stubs, feats = nagoki.track_stubs(sec, pads0, features.load_features_rows(track.load_section_rows("features", bounds)))
    assembly.add_features(m, sec, feats)   # 分かれ道の入口で切った壁・ガードレール
    for name, v, f, uv, mat in pavements.meshes(sec, pavements.load_rows(
            assembly._cut_rows(track.load_section_rows("pavements", bounds), [z.main_skip for z in sps]))):
        m.add_mesh(assembly.GROUP, name, v, f, uv, mat)
    pads = json.loads((P / "track_pads.json").read_text(encoding="utf-8"))
    lot_spec = config.DATA_MANUAL / "s1_start_lot.json"
    # 駐車場や広場には、木の板（幅 最大7m）がかぶらないよう少し広めに空ける
    blds = nagoki.load_buildings()
    track_blds = buildings.load(config.DATA_MANUAL / "track_buildings.json")   # gone=true（もう無い建物）は作らない
    for j, _ in others:   # ほかの交差点の町の中の建物は、町の建物リスト（1棟ずつ確かめたもの）で作る
        track_blds = [b for b in track_blds if not nagoki.circle(j).contains(b.footprint.centroid)]
    assembly.add_trees(m, trees.scatter(near, pads + start_lot.flats_for(lot_spec, pads), pad_clear_m=5.0,
                                        avoid=nagoki.tree_free_areas(sec, bounds, blds + track_blds)))
    assembly.add_pads(m, pads)
    assembly.add_start_lot(m, start_lot.load_spec(lot_spec), pads)
    site_spec = pass_area.load(config.DATA_MANUAL / "s1_pass.json")
    if site_spec:
        assembly.add_pass_area(m, pass_area.build(site_spec, sec))
    minoge = minoge_area.load(config.DATA_MANUAL / "s2_minoge.json")
    if minoge:
        assembly.add_pass_area(m, minoge_area.build(minoge, sec, minoge_area.dem_ground()))
    sampler = gsi.default_sampler()
    base_pads = [p for p in pads if p.get("kind") != "stub"]
    from shapely.ops import unary_union as _uu
    # stubs は上で作った（入口は本線と歩道から離してある）
    assembly.add_meshes(m, side_roads.stub_meshes(stubs))
    from yabitsu import bridge_parts   # S5 の名前のある橋（親柱・銘板・欄干・桁・橋脚・赤いアーチ）
    brs = bridge_parts.load(config.DATA_MANUAL / "s5_bridges.json")
    rng = bridge_parts.ranges(sec, side_roads.load_elements(), {int(b["osm_id"]) for b in brs})
    for name, rel in bridge_parts.write_plate_textures(brs, config.TRACK_BUILD_DIR / "textures").items():
        m.add_material(name, texture=rel)
    for name, rel in bridge_parts.rail_textures(brs, config.TRACK_BUILD_DIR / "textures").items():
        m.add_material(name, texture=rel, alpha=True, two_sided=True)   # 格子のすき間は透明
    assembly.add_meshes(m, bridge_parts.meshes(sec, brs, rng, ground))
    from yabitsu import tunnel, textures as _tx   # トンネル（筒・照明・坑口・表札）
    tns = tunnel.from_osm(side_roads.load_elements(), sec)
    tspec = json.loads((config.DATA_MANUAL / "s5_tunnels.json").read_text(encoding="utf-8")) if (config.DATA_MANUAL / "s5_tunnels.json").exists() else {}
    for tn in tns:
        txt = tspec.get(str(tn.osm_id), {}).get("plate", tn.name)
        if txt:
            pth = config.TRACK_BUILD_DIR / "textures" / f"tunnel_plate_{tn.osm_id}.png"; pth.parent.mkdir(parents=True, exist_ok=True)
            _tx.sign_texture(txt, "#E9E6DC", "#1C1C1C", 512, 112, border="#555555").save(pth)
            m.add_material(f"tunnel_plate_{tn.osm_id}", texture=f"textures/{pth.name}")
    assembly.add_meshes(m, tunnel.meshes(sec, tns, tspec, ground))
    from yabitsu import water   # S4 の川・沢（水面・川原）、堰・砂防ダム、湧き水の水くみ場
    streams, falls, springs = water.load(sec)
    form_all = start_lot.road_formation(sec, 0.0, float(sec.s[-1]))
    assembly.add_meshes(m, water.meshes(streams, water.road_area(sec, form_all, stubs)))
    wspec = json.loads((config.DATA_MANUAL / "s5_water.json").read_text(encoding="utf-8"))   # S5 の滝（ストリートビュー・OSM）
    assembly.add_meshes(m, water.feature_meshes(water.manual_falls(wspec.get("falls", []), sec, ground), [], prefix="s5_water"))
    lake_poly, lake_level = water.load_lake()   # 宮ヶ瀬湖の湖面（見える湖の全体）
    if lake_poly is not None:
        assembly.add_meshes(m, water.lake_meshes(lake_poly, lake_level, water.road_area(sec, form_all, stubs)))
    assembly.add_meshes(m, water.feature_meshes(falls, water.spring_items(springs, sec, ground, near_m=60.0)))
    from yabitsu import geo
    for k, (la, lo) in enumerate(json.loads((config.DATA_MANUAL / "s3_side_crosswalks.json").read_text(encoding="utf-8"))):
        p = np.array([float(v) for v in geo.to_local(la, lo)])
        from shapely.geometry import Point
        near = [st for st in stubs if st.surface.buffer(1.0).contains(Point(*p))]
        if near:   # 横道の上の横断歩道（OSM）。分かれ道を作っていない所（町の範囲など）は描かない
            m.add_mesh(assembly.GROUP, f"LINES_side_cw_{k:03d}", *near[0].crosswalk_mesh(p), "line_white")
    road_z = town_objects.road_ground(sec)
    ways_by_id = {w["id"]: w for w in side_roads.load_elements() if w.get("type") == "way"}
    for k, st in enumerate(stubs):   # 車で入っていける分かれ道の入口にだけ工事中の柵（手前＝本線の側）
        if not side_roads.needs_barricade(st, ways_by_id):
            continue
        (ax, ay), (bx, by) = st.mouth.coords
        p0, p1 = np.array(st.axis.coords[0]), np.array(st.axis.interpolate(min(2.0, st.axis.length)).coords[0])
        assembly.add_meshes(m, town_objects.barricade_meshes((ax, ay), (bx, by), st.ground_z, p0 - p1, k=100 + k))   # 入口の路面の高さに
    ov = overpasses.load_overrides()
    mws = motorway.from_osm(side_roads.load_elements(), sec, sampler, ov,
                            walks=features.sidewalk_offsets(sec, feats))   # 新東名など：両側のトンネルの入口まで（柱は歩道の外）
    for z in sps:   # 路面（2車線・西の上り・東の下り・導流帯）と島（鳥居・灯籠など）
        assembly.add_meshes(m, z.meshes())
    # コースから 200m の景色：建物（1棟ずつ外周線の形）と、走れない道
    from shapely.ops import unary_union
    from shapely.geometry import Polygon
    keep_out = unary_union([start_lot.road_formation(sec, 0.0, float(sec.s[-1])).buffer(1.0), tw.region.buffer(1.0)]
                           + [k.region.buffer(1.0) for _, k in others]
                           + [st.surface.buffer(0.5) for st in stubs] + [x.footprint() for x in mws]
                           + [z.area().buffer(0.5) for z in sps]
                           + [unary_union(nagoki.sidewalk_polygons(sec, feats)).buffer(0.3)]   # 歩道の上に建物を立てない
                           + [Polygon(p["polygon"]) for p in pads if p.get("kind") in ("lot", "pass", "site", "plaza", "branch")
                              and "house" not in p["name"]])
    from yabitsu import pois
    poi_el = json.loads((config.DATA_RAW / "osm_track_pois.json").read_text(encoding="utf-8"))["elements"]
    pois.add_shop_signs(poi_el, track_blds, sec)   # 店の名前の看板
    for e in poi_el:   # 文化財の古民家（旧芦川家住宅主屋）：かやぶきの寄棟
        if e.get("tags", {}).get("historic") == "house" and "center" in e:
            from shapely.geometry import Point
            c = Point(*[float(v) for v in geo.to_local(e["center"]["lat"], e["center"]["lon"])])
            b = min(track_blds, key=lambda q: q.footprint.distance(c))
            if b.footprint.distance(c) < 10:
                b.roof = {"type": "hip", "rise_m": 4.5, "color": "#6B5B45"}; b.wall = "#8A7356"; b.windows = False
    for name, rel in buildings.write_textures(track_blds, config.TRACK_BUILD_DIR / "textures").items():
        m.add_material(name, texture=rel)
    assembly.add_meshes(m, [(f"{n[:-3]}_trk{n[-3:]}", v, f, uv, mat, r) for n, v, f, uv, mat, r in
                            buildings.merge_by_material(buildings.building_meshes(track_blds, ground, keep_out))])
    band = LineString(np.c_[sec.x, sec.y]).buffer(scenery.BAND_M); shapely.prepare(band)   # コースから 200m だけ
    roads = scenery.road_polygons(side_roads.load_elements(), keep_out, near=band)
    from yabitsu import town as town_mod
    for k, (v, f, uv) in enumerate(town_mod.grid_drape(roads, ground, scenery.LIFT, cell=2.0)):
        m.add_mesh(assembly.GROUP, f"scenery_road_{k:02d}", v, f, uv, "asphalt")
    ro = nagoki.roadside_objects()
    # 道ぞいの小さな目印（OSM）：自動販売機・ポスト・電話・案内板・待合所・石碑。手で置いた物の近くは除く
    skip = [tuple(float(v) for v in geo.to_local(*o["points"][0])) for o in ro["objects"]]
    skip += [tuple(float(v) for v in geo.to_local(*mo["center"])) for mo in (minoge or {}).get("monuments", [])]
    skip += [tuple(float(v) for v in geo.to_local(*ob["at"])) for z in sps for ob in z.spec.get("objects", [])]
    poi_nodes = [e for e in poi_el if e.get("type") == "node" and "lat" in e]
    assembly.add_meshes(m, pois.meshes(pois.place(poi_nodes, sec, skip), ground))
    assembly.add_meshes(m, town_objects.signal_meshes(ro["signals"], road_z, prefix="s3_signal"))
    assembly.add_meshes(m, town_objects.object_meshes(town_objects.off_road(ro["objects"], sec), road_z, prefix="s3_obj"))   # 路面の中のバス停は道の外へ
    for sn in ("s4", "s5"):   # S4・S5 のカーブミラー・標識・看板（data/manual/s4_objects.json・s5_objects.json）
        so = json.loads((config.DATA_MANUAL / f"{sn}_objects.json").read_text(encoding="utf-8"))
        assembly.add_meshes(m, town_objects.object_meshes(town_objects.off_road(so["objects"], sec), road_z, prefix=f"{sn}_obj"))
        assembly.add_meshes(m, town_objects.pole_sign_meshes(so.get("pole_signs", []), ground, prefix=sn))
        for name, rel in town_objects.write_pole_sign_textures(so.get("pole_signs", []), config.TRACK_BUILD_DIR / "textures", prefix=sn).items():
            m.add_material(name, texture=rel)
    # 町の範囲（名古木交差点）：別の kn5（config.TOWN_GROUP）
    assembly.add_meshes(m, tw.meshes(), group=config.TOWN_GROUP)
    assembly.add_meshes(m, tw.object_meshes(), group=config.TOWN_GROUP)
    for name, rel in town_objects.write_pole_sign_textures(tw.pole_signs(), config.TRACK_BUILD_DIR / "textures").items():
        m.add_material(name, texture=rel)
    lots = json.loads((config.DATA_MANUAL / "s3_lots.json").read_text(encoding="utf-8"))   # 店の駐車場・敷地の舗装と白線
    assembly.add_meshes(m, tw.lot_meshes(lots), group=config.TOWN_GROUP)
    for name, rel in buildings.write_textures(blds, config.TRACK_BUILD_DIR / "textures").items():
        m.add_material(name, texture=rel)
    avoid = tw.paved_avoid()
    assembly.add_meshes(m, buildings.merge_by_material(buildings.building_meshes(blds, tw.ground.z, avoid)), group=config.TOWN_GROUP)
    for j, k in others:   # ほかの交差点の町：別の kn5（j.group）。名前は town → 交差点の名前
        rn = lambda items, j=j: nagoki.rename_meshes(items, j.name)
        assembly.add_meshes(m, rn(k.meshes()), group=j.group)
        assembly.add_meshes(m, rn(k.object_meshes()), group=j.group)
        for name, rel in town_objects.write_pole_sign_textures(k.pole_signs(), config.TRACK_BUILD_DIR / "textures", prefix=j.name).items():
            m.add_material(name, texture=rel)
        lp = config.DATA_MANUAL / j.lots
        if lp.exists():
            assembly.add_meshes(m, rn(k.lot_meshes(json.loads(lp.read_text(encoding="utf-8")))), group=j.group)
        kb = nagoki.load_buildings(j)
        for name, rel in buildings.write_textures(kb, config.TRACK_BUILD_DIR / "textures").items():
            m.add_material(name, texture=rel)
        assembly.add_meshes(m, [(f"{n}_{j.name}", *rest) for n, *rest in
                                buildings.merge_by_material(buildings.building_meshes(kb, k.ground.z, k.paved_avoid()))], group=j.group)
    el = side_roads.load_elements()
    assembly.add_meshes(m, motorway.meshes(mws))
    skip = {i for x in mws for i in x.way_ids}
    assembly.add_meshes(m, overpasses.meshes(overpasses.from_osm(el, sec, sampler, ov, skip_ids=skip)))
    assembly.add_layout_objects(m, sec, pads, layouts.load_layouts(), bounds)
    # S4（峠の先）の部品は yabitsu_s4.kn5 へ（全部のレイアウトが読む）。メッシュの頂点の平均に一番近い本線の行で決める
    s4_0 = float(bounds["s4"][0]); tree = __import__("scipy.spatial", fromlist=["cKDTree"]).cKDTree(np.c_[sec.x, sec.y])
    print("moved to", config.S4_GROUP, m.split_group(assembly.GROUP, config.S4_GROUP, lambda c: sec.s[tree.query(c)[1]] >= s4_0))
    s5_0 = float(bounds["s5"][0])   # S5（札掛の先）の部品は yabitsu_s5.kn5 へ
    print("moved to", config.S5_GROUP, m.split_group(config.S4_GROUP, config.S5_GROUP, lambda c: sec.s[tree.query(c)[1]] >= s5_0))
    print(m.save()); print(m.summary())


if __name__ == "__main__":
    main()
