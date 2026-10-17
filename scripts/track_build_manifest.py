import json

import numpy as np

from yabitsu import (assembly, buildings, config, features, gsi, layouts, minoge_area, motorway, nagoki, overpasses, pass_area,
                     pavements, side_roads, start_lot, textures, town_objects, track, trees)
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
    assembly.add_terrain(m, near, far)
    bounds = track.load_bounds()
    assembly.add_centrelines(m, sec, track.load_section_rows("centrelines", bounds), s_min=tw.main_skip[1])
    assembly.add_features(m, sec, features.load_features_rows(track.load_section_rows("features", bounds)))
    for name, v, f, uv, mat in pavements.meshes(sec, pavements.load_rows(track.load_section_rows("pavements", bounds))):
        m.add_mesh(assembly.GROUP, name, v, f, uv, mat)
    pads = json.loads((P / "track_pads.json").read_text(encoding="utf-8"))
    lot_spec = config.DATA_MANUAL / "s1_start_lot.json"
    # 駐車場や広場には、木の板（幅 最大7m）がかぶらないよう少し広めに空ける
    blds = nagoki.load_buildings()
    assembly.add_trees(m, trees.scatter(near, pads + start_lot.flats_for(lot_spec, pads), pad_clear_m=5.0,
                                        avoid=nagoki.tree_free_areas(sec, bounds, blds)))
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
    # S3 の道ぞい：家、信号、バス停・鳥居など
    assembly.add_pass_area(m, nagoki.roadside_houses(sec, minoge_area.dem_ground()))
    ro = nagoki.roadside_objects()
    assembly.add_meshes(m, town_objects.signal_meshes(ro["signals"], road_z, prefix="s3_signal"))
    assembly.add_meshes(m, town_objects.object_meshes(ro["objects"], road_z, prefix="s3_obj"))
    # 町の範囲（名古木交差点）：別の kn5（config.TOWN_GROUP）
    assembly.add_meshes(m, tw.meshes(), group=config.TOWN_GROUP)
    assembly.add_meshes(m, tw.object_meshes(), group=config.TOWN_GROUP)
    for name, rel in buildings.write_textures(blds, config.TRACK_BUILD_DIR / "textures").items():
        m.add_material(name, texture=rel)
    avoid = tw.carriageway.union(tw.sidewalk).union(tw.drivable())
    assembly.add_meshes(m, buildings.merge_by_material(buildings.building_meshes(blds, tw.ground.z, avoid)), group=config.TOWN_GROUP)
    el = side_roads.load_elements()
    ov = overpasses.load_overrides()
    mws = motorway.from_osm(el, sec, sampler, ov)   # 新東名など：両側のトンネルの入口まで
    assembly.add_meshes(m, motorway.meshes(mws))
    skip = {i for x in mws for i in x.way_ids}
    assembly.add_meshes(m, overpasses.meshes(overpasses.from_osm(el, sec, sampler, ov, skip_ids=skip)))
    assembly.add_layout_objects(m, sec, pads, layouts.load_layouts(), bounds)
    print(m.save()); print(m.summary())


if __name__ == "__main__":
    main()
