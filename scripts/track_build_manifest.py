import json

import numpy as np

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
    assembly.add_road(m, sec, skip=tw.main_skip)   # 町の範囲の本線は、町の多角形で作る
    near = dict(np.load(P / "track_terrain_near.npz")); far = dict(np.load(P / "track_terrain_far.npz"))
    mid = dict(np.load(P / "track_terrain_mid.npz"))
    assembly.add_terrain(m, near, far, mid)
    ground = terrain.terrain_z(near, mid)   # 景色の建物・道は、できあがった地形の上に置く
    bounds = track.load_bounds()
    assembly.add_centrelines(m, sec, track.load_section_rows("centrelines", bounds), s_min=tw.main_skip[1])
    from yabitsu import crosssection as cs
    assembly.add_bay_lines(m, sec, cs.parse_bay_rows(track.load_section_rows("busbays", bounds)))
    assembly.add_features(m, sec, features.load_features_rows(track.load_section_rows("features", bounds)))
    for name, v, f, uv, mat in pavements.meshes(sec, pavements.load_rows(track.load_section_rows("pavements", bounds))):
        m.add_mesh(assembly.GROUP, name, v, f, uv, mat)
    pads = json.loads((P / "track_pads.json").read_text(encoding="utf-8"))
    lot_spec = config.DATA_MANUAL / "s1_start_lot.json"
    # 駐車場や広場には、木の板（幅 最大7m）がかぶらないよう少し広めに空ける
    blds = nagoki.load_buildings()
    track_blds = [buildings.Building.from_spec(d) for d in json.loads((config.DATA_MANUAL / "track_buildings.json").read_text(encoding="utf-8"))]
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
    stubs = side_roads.for_track(sec, base_pads, sampler, exclude=tw.region)
    assembly.add_meshes(m, side_roads.stub_meshes(stubs))
    from yabitsu import geo
    for k, (la, lo) in enumerate(json.loads((config.DATA_MANUAL / "s3_side_crosswalks.json").read_text(encoding="utf-8"))):
        p = np.array([float(v) for v in geo.to_local(la, lo)])
        from shapely.geometry import Point
        near = [st for st in stubs if st.surface.buffer(1.0).contains(Point(*p))]
        if near:   # 横道の上の横断歩道（OSM）。分かれ道を作っていない所（町の範囲など）は描かない
            m.add_mesh(assembly.GROUP, f"LINES_side_cw_{k:03d}", *near[0].crosswalk_mesh(p), "line_white")
    road_z = town_objects.road_ground(sec)
    for k, st in enumerate(stubs):   # 分かれ道の入口に工事中の柵（手前＝本線の側）
        (ax, ay), (bx, by) = st.mouth.coords
        p0, p1 = np.array(st.axis.coords[0]), np.array(st.axis.interpolate(min(2.0, st.axis.length)).coords[0])
        assembly.add_meshes(m, town_objects.barricade_meshes((ax, ay), (bx, by), road_z, p0 - p1, k=100 + k))
    ov = overpasses.load_overrides()
    mws = motorway.from_osm(side_roads.load_elements(), sec, sampler, ov)   # 新東名など：両側のトンネルの入口まで
    brs = nagoki.load_branches(sec)   # 県道70号が2本に分かれる所：もう1本（走れる）と島（鳥居・灯籠など）
    for b in brs:
        assembly.add_meshes(m, b.meshes())
    # コースから 200m の景色：建物（1棟ずつ外周線の形）と、走れない道
    from shapely.ops import unary_union
    from shapely.geometry import Polygon
    keep_out = unary_union([start_lot.road_formation(sec, 0.0, float(sec.s[-1])).buffer(1.0), tw.region.buffer(1.0)]
                           + [st.surface.buffer(0.5) for st in stubs] + [x.footprint() for x in mws]
                           + [b.surface.buffer(0.5) for b in brs] + [b.island for b in brs if b.island is not None]
                           + [Polygon(p["polygon"]) for p in pads if p.get("kind") in ("lot", "pass", "site", "plaza")
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
    roads = scenery.road_polygons(side_roads.load_elements(), keep_out)
    from yabitsu import town as town_mod
    for k, (v, f, uv) in enumerate(town_mod.grid_drape(roads, ground, scenery.LIFT, cell=2.0)):
        m.add_mesh(assembly.GROUP, f"scenery_road_{k:02d}", v, f, uv, "asphalt")
    ro = nagoki.roadside_objects()
    # 道ぞいの小さな目印（OSM）：自動販売機・ポスト・電話・案内板・待合所・石碑。手で置いた物の近くは除く
    skip = [tuple(float(v) for v in geo.to_local(*o["points"][0])) for o in ro["objects"]]
    skip += [tuple(float(v) for v in geo.to_local(*mo["center"])) for mo in (minoge or {}).get("monuments", [])]
    skip += [tuple(float(v) for v in geo.to_local(*ob["at"])) for b in brs for ob in b.spec.get("objects", [])]
    poi_nodes = [e for e in poi_el if e.get("type") == "node" and "lat" in e]
    assembly.add_meshes(m, pois.meshes(pois.place(poi_nodes, sec, skip), ground))
    assembly.add_meshes(m, town_objects.signal_meshes(ro["signals"], road_z, prefix="s3_signal"))
    assembly.add_meshes(m, town_objects.object_meshes(ro["objects"], road_z, prefix="s3_obj"))
    # 町の範囲（名古木交差点）：別の kn5（config.TOWN_GROUP）
    assembly.add_meshes(m, tw.meshes(), group=config.TOWN_GROUP)
    assembly.add_meshes(m, tw.object_meshes(), group=config.TOWN_GROUP)
    for name, rel in town_objects.write_pole_sign_textures(tw.spec.get("pole_signs", []), config.TRACK_BUILD_DIR / "textures").items():
        m.add_material(name, texture=rel)
    lots = json.loads((config.DATA_MANUAL / "s3_lots.json").read_text(encoding="utf-8"))   # 店の駐車場・敷地の舗装と白線
    assembly.add_meshes(m, tw.lot_meshes(lots), group=config.TOWN_GROUP)
    for name, rel in buildings.write_textures(blds, config.TRACK_BUILD_DIR / "textures").items():
        m.add_material(name, texture=rel)
    avoid = tw.paved_avoid()
    assembly.add_meshes(m, buildings.merge_by_material(buildings.building_meshes(blds, tw.ground.z, avoid)), group=config.TOWN_GROUP)
    el = side_roads.load_elements()
    assembly.add_meshes(m, motorway.meshes(mws))
    skip = {i for x in mws for i in x.way_ids}
    assembly.add_meshes(m, overpasses.meshes(overpasses.from_osm(el, sec, sampler, ov, skip_ids=skip)))
    assembly.add_layout_objects(m, sec, pads, layouts.load_layouts(), bounds)
    print(m.save()); print(m.summary())


if __name__ == "__main__":
    main()
