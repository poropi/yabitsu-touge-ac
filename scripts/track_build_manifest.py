import argparse
import json
from pathlib import Path

import numpy as np
import shapely
from shapely.geometry import LineString

from yabitsu import (assembly, buildings, config, features, fubuki, gsi, layouts, minoge_area, motorway, nagoki, overpasses, pass_area,
                     pavements, scenery, side_roads, small_bridges, start_lot, terrain, textures, town_objects, track, trees,
                     valleys, valley_fit, water)
from yabitsu.crosssection import Section, parse_width_rows, width_tapers
from yabitsu.manifest import Manifest

P = config.DATA_PROCESSED


def add_far_photo(m, far):
    """遠い地形に貼る空中写真（地理院 seamlessphoto z=15、長方形全体を 1 枚、1024×4096）。"""
    from PIL import Image
    from yabitsu import geo, gsi, photo
    x0, x1, y0, y1 = float(far["xs"][0]), float(far["xs"][-1]), float(far["ys"][0]), float(far["ys"][-1])
    img = photo.render_local_image(gsi.TileCache(), geo.from_local, x0, y0, x1 - x0, px_size=5.0, z=15, height_m=y1 - y0)
    (config.TRACK_BUILD_DIR / "textures").mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).resize((1024, 4096), Image.LANCZOS).save(config.TRACK_BUILD_DIR / "textures" / "far_photo.jpg", quality=90)
    m.add_material("ground_far", texture="textures/far_photo.jpg")


def _nanohanadai_area(sec):
    """菜の花台の駐車場・敷石の遊歩道・展望台への道・展望台の広場（s1_start_lot.json）を合わせた範囲。"""
    from shapely.geometry import LineString as _LS, Point as _Pt, Polygon as _Pg
    spec = start_lot.load_spec(config.DATA_MANUAL / "s1_start_lot.json")
    if not spec:
        return []
    f = start_lot.frame(spec)
    out = [start_lot.lot_polygon(spec, sec).buffer(1.0)]
    out += [start_lot._poly(f, r).buffer(1.5) for r in spec.get("paved", [])]
    if "path" in spec:
        out.append(_LS(f.xy(*np.asarray(spec["path"]["points"], float).T)).buffer(spec["path"]["width"] / 2 + 1.5))
    if "tower" in spec:
        out.append(_Pt(*f.xy(spec["tower"]["a"], spec["tower"]["b"])).buffer(spec["tower"]["plaza_r"]))
    out += [start_lot._poly(f, r) for r in spec.get("flats", [])]   # 芝生など平らにした所（OSM の小道は手で引いた道から 4〜5 m ずれている）
    return out


def valley_water_parts(sec, bounds, near, ground, dug, streams, road):
    """組み立てと部分検証で同じ水面・川原を作る。"""
    rows = valleys.load()
    levels = water.valley_surface_levels(sec, bounds, rows, streams, ground, near=near)
    normals = {}
    items = water.meshes(streams, road, levels, ground, near=near, changed_area=dug.polygon, before_ground=dug.old.height,
                         frozen_normals=normals)
    rivers = [i for i, stream in enumerate(streams) if stream.osm_id in (water.A_CREEK, 90334744)]
    old = water.meshes([streams[i] for i in rivers], road) if not dug.polygon.is_empty else []
    names = tuple(f"water_{i:03d}_" for i in rivers)
    new = [item for item in items if item[0].startswith(names)]
    surfaces = tuple(valley_fit.water_surfaces(
        valley_fit.Mesh(n, v, f, mat, visible) for n, v, f, _, mat, visible in part) for part in (old, new))
    return items, normals, surfaces


def card_adjustments(cards, dug, rows, surfaces):
    """木を配置した順序で、根元の補正と除外を返す。"""
    heights = np.full(len(cards), np.nan)
    remove = np.zeros(len(cards), bool)
    inside = np.flatnonzero(dug.covers(cards[:, :2]))
    points = cards[inside, :2]
    if len(points):
        heights[inside] = dug.height(points)
        if dug.bed is not None:
            bed, wet = dug.tree_exclusions(points, surfaces[1])
            remove[inside] = bed | wet
        else:
            remove[inside] = (valleys.tree_bottom(rows, *points.T)
                              | valleys.newly_wet(dug, *surfaces, points))
    return heights, remove


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--before-near", type=Path, help="谷を掘る前の近景 NPZ")
    args = parser.parse_args()
    near_path = P / "track_terrain_near.npz"
    with np.load(near_path) as data:
        near = dict(data)
    before = valleys.before_for_build(args.before_near, near_path, near)
    dug = valleys.changed_ground(before, near)
    sec = Section.load_json(P / "track_section.json")
    m = Manifest(config.TRACK_BUILD_DIR)
    textures.register_materials(m)
    small_bridges.add_valley_materials(m)
    tw = nagoki.load_town(sec)
    others = nagoki.other_towns(sec)   # 宮ヶ瀬北原など（町の範囲は多角形で作る）
    sps = nagoki.load_splits(sec)   # 島をはさんで2本に分かれる所（地図の線どおり）
    site_spec = pass_area.load(config.DATA_MANUAL / "s1_pass.json")
    pass_site = pass_area.build(site_spec, sec) if site_spec else None
    from yabitsu import terayama_area
    tera_spec = terayama_area.load(config.DATA_MANUAL / "s4_terayama.json")
    tera_site = terayama_area.build(tera_spec, sec) if tera_spec else None
    from yabitsu import gomayashiki_area
    goma_spec = gomayashiki_area.load(config.DATA_MANUAL / "s4_gomayashiki.json")
    goma_site = gomayashiki_area.build(goma_spec, sec) if goma_spec else None
    from yabitsu import asama_area   # 浅間神社の入口の駐車場（航空写真・ストリートビュー 2025-07）
    asama = asama_area.load(config.DATA_MANUAL / "s2_asama.json")
    asama_site = asama_area.build(asama, sec) if asama else None
    sites = [x for x in (pass_site, tera_site, goma_site, asama_site) if x]
    from yabitsu import tunnel
    tns = tunnel.from_osm(side_roads.load_elements(), sec)
    bounds = track.load_bounds()
    tapers = tuple(width_tapers(parse_width_rows(track.load_section_rows("widths", bounds))))
    bridge_spans = small_bridges.spans(sec, bounds, small_bridges.load())
    paving = pavements.load_rows(assembly._cut_rows(
        track.load_section_rows("pavements", bounds), [z.main_skip for z in sps]))
    far = dict(np.load(P / "track_terrain_far.npz"))
    mid = dict(np.load(P / "track_terrain_mid.npz"))
    ground_source = near if "valley_faces" in near else before
    ground = terrain.terrain_z(ground_source, mid)
    ground_mesh = terrain.terrain_mesh_z(near, mid)
    prop_ground = ground_mesh if ground_source is near else terrain.terrain_mesh_z(before, mid)
    assembly.add_road(m, sec, skip=[tw.main_skip] + [k.main_skip for _, k in others] + [z.main_skip for z in sps],   # 町の範囲と分かれる所は多角形で作る
                      mouths=[mo for x in sites for mo in x.mouths()]   # 峠・寺山・護摩屋敷のわき道の入口と浅間神社の駐車場の前は側溝をくぼませない
                      + start_lot.mouths(json.loads((P / "track_pads.json").read_text(encoding="utf-8")), sec),   # 菜の花台の駐車場の入口も
                      no_lines=[r for x in (tera_site, goma_site) if x for r in x.no_lines()],   # 寺山富士見橋のまわりは路肩の白線が無い
                      dry=[(t.s0, t.s1) for t in tns],   # トンネルの中は水たまりを作らない材質
                      paving=paving, dug=dug if "valley_faces" in near else None, deck_spans=[bridge_spans[k] for k in small_bridges.SUPPORT_SPANS if k in bridge_spans],
                      ground_z=ground_mesh, width_tapers=tapers)
    add_far_photo(m, far)
    assembly.add_terrain(m, near, far, mid, far_photo=True, before_near=before if not dug.polygon.is_empty else None)
    assembly.add_meshes(m, small_bridges.support_meshes(sec, bridge_spans, ground_mesh,
                                                      graded_shoulder="valley_gravel" in near))
    assembly.add_centrelines(m, sec, track.load_section_rows("centrelines", bounds), s_min=tw.main_skip[1],
                             gaps=[z.main_skip for z in sps] + [k.main_skip for _, k in others])
    from yabitsu import crosssection as cs
    assembly.add_bay_lines(m, sec, cs.parse_bay_rows(track.load_section_rows("busbays", bounds)))
    pads0 = json.loads((P / "track_pads.json").read_text(encoding="utf-8"))
    stubs, feats = nagoki.track_stubs(sec, pads0, features.load_features_rows(track.load_section_rows("features", bounds)))
    from yabitsu import bridge_parts
    brs = bridge_parts.load(config.DATA_MANUAL / "s5_bridges.json")
    feats, brs = fubuki.kamatagawa_parts(feats, brs, bounds["s5"][0])
    assembly.add_features(m, sec, feats, right_walk_joins=fubuki.walk_joins(bounds["s5"][0]),
                          bridge_spans=bridge_spans, ground_z=ground_mesh, dug=dug if "valley_faces" in near else None,
                          width_tapers=tapers, graded_shoulder="valley_gravel" in near)
    for name, v, faces, uv, mat in fubuki.wall_cap_meshes(sec, feats, bounds["s5"][0], ground_mesh):
        m.add_mesh(assembly.GROUP, name, v, faces, uv, mat)
    for name, v, f, uv, mat in pavements.meshes(sec, paving):
        m.add_mesh(assembly.GROUP, name, v, f, uv, mat)
    pads = json.loads((P / "track_pads.json").read_text(encoding="utf-8"))
    lot_spec = config.DATA_MANUAL / "s1_start_lot.json"
    # 駐車場や広場には、木の板（幅 最大7m）がかぶらないよう少し広めに空ける
    blds = nagoki.load_buildings()
    track_buildings_path = config.DATA_MANUAL / "track_buildings.json"
    track_blds = buildings.load(track_buildings_path)   # gone=true（もう無い建物）は作らない
    tree_clearance = buildings.load(track_buildings_path, clearance_only=True)
    for j, _ in others:   # ほかの交差点の町の中の建物は、町の建物リスト（1棟ずつ確かめたもの）で作る
        track_blds = [b for b in track_blds if not nagoki.circle(j).contains(b.footprint.centroid)]
    streams, falls, springs = water.load(sec)
    form_all = start_lot.road_formation(sec, 0.0, float(sec.s[-1]))
    river_meshes, river_normals, water_surfaces = valley_water_parts(
        sec, bounds, near, ground_mesh, dug, streams, water.road_area(sec, form_all, stubs))
    cards = trees.scatter(before, pads + start_lot.flats_for(lot_spec, pads), pad_clear_m=5.0,
                          avoid=nagoki.tree_free_areas(sec, bounds, blds + track_blds + tree_clearance))
    heights, remove = card_adjustments(cards, dug, valleys.load(), water_surfaces)
    assembly.add_trees(m, cards, heights=heights, remove=remove)
    assembly.add_pads(m, pads)
    _lot = start_lot.load_spec(lot_spec)
    if _lot.get("landmarks"):   # 菜の花台の Codex の部品の画像（data/processed/landmarks → textures）
        import shutil as _sh2
        from yabitsu import landmarks as _lm
        _td = config.TRACK_BUILD_DIR / "textures"; _td.mkdir(parents=True, exist_ok=True)
        for _mat, (_png, _alpha) in _lm.materials(_lot["landmarks"]).items():
            if _png:
                _sh2.copy2(_png, _td / f"{_mat}.png")
                m.add_material(_mat, texture=f"textures/{_mat}.png", alpha=_alpha)
    assembly.add_start_lot(m, _lot, pads, sec)   # 入口で舗装を道路と段差なくつなぐ
    if pass_site:
        assembly.add_pass_area(m, pass_site)
    minoge = minoge_area.load(config.DATA_MANUAL / "s2_minoge.json")
    if minoge:
        assembly.add_pass_area(m, minoge_area.build(minoge, sec, minoge_area.dem_ground()))
    if asama:   # 浅間神社の入口の駐車場：広場・石積み・石段・社号標・石碑・看板
        import shutil as _sh
        tdir = config.TRACK_BUILD_DIR / "textures"; tdir.mkdir(parents=True, exist_ok=True)
        _sh.copy2(config.PROJECT_ROOT / "assets" / "dress" / "kenchi.png", tdir / "kenchi.png")
        asama_area.pillar_texture(asama["pillar"]["text"], tdir / "asama_pillar.png")
        asama_area.board_texture(tdir / "asama_board.png")
        asama_area.notice_texture(tdir / "asama_notice.png")
        m.add_material("kenchi", texture="textures/kenchi.png")
        from PIL import Image as _Im   # 茂みの陰に入った奥の石積み：写真（000・045）で RGB 約 46、日なたの石段の左の壁は約 120 → 間知の絵を 0.85 倍
        _k = np.asarray(_Im.open(tdir / "kenchi.png").convert("RGB"), np.float32) * 0.85   # 0.6 では茂みの描画の陰と重なって真っ黒になった
        _Im.fromarray(_k.clip(0, 255).astype(np.uint8)).save(tdir / "kenchi_shade.png")
        m.add_material("kenchi_shade", texture="textures/kenchi_shade.png")
        m.add_material("asama_notice", texture="textures/asama_notice.png")
        mon = config.PROJECT_ROOT / asama["monument"]["model"] / "monument.png"   # 石碑の焼き付け（blender_scripts/bake_asama_monument.py）
        if mon.exists():
            _sh.copy2(mon, tdir / "asama_monument.png")
            m.add_material("asama_monument", texture="textures/asama_monument.png")
        for key, lm in asama.get("landmarks", {}).items():   # Codex 制作の蔓草・立札・草の画像（葉と草は透過）
            if key == "note" or not asama_area._landmark_exists(lm):
                continue
            src = config.PROJECT_ROOT / lm["dir"]
            pngs = {lm["material"]: lm["texture"]} if lm.get("texture") else {f'{lm["material"]}_{n[-1]}': f"{n}.png" for n in lm["names"]}
            for mat, png in pngs.items():
                _sh.copy2(src / png, tdir / f"{mat}.png")
                m.add_material(mat, texture=f"textures/{mat}.png", alpha=key != "post")
        m.add_material("asama_pillar", texture="textures/asama_pillar.png")
        m.add_material("asama_board", texture="textures/asama_board.png")
        assembly.add_pass_area(m, asama_site)
    if tera_site:   # 寺山富士見橋：広場・わき道・トイレ（正面の絵はトイレの横の長さ・壁の高さの割合で書く）
        from yabitsu import textures as _tx
        _L, _ = tera_spec["toilet"]["size"]; _H = tera_spec["toilet"]["wall_h"]
        _tx.terayama_toilet_front_texture(int(_L * 160), int(_H * 160)).save(config.TRACK_BUILD_DIR / "textures" / "terayama_toilet_front.png")
        m.add_material("terayama_toilet_front", texture="textures/terayama_toilet_front.png")
        assembly.add_pass_area(m, tera_site)
    if goma_site:   # 護摩屋敷の水：分かれ目の広場・湧き水・わき道の輪
        assembly.add_pass_area(m, goma_site)
    sampler = gsi.default_sampler()
    base_pads = [p for p in pads if p.get("kind") != "stub"]
    from shapely.ops import unary_union as _uu
    # stubs は上で作った（入口は本線と歩道から離してある）
    assembly.add_meshes(m, side_roads.stub_meshes(stubs))
    # S5 の名前のある橋（親柱・銘板・欄干・桁・橋脚・赤いアーチ）
    rng = bridge_parts.ranges(sec, side_roads.load_elements(), {int(b["osm_id"]) for b in brs})
    for name, rel in bridge_parts.write_plate_textures(brs, config.TRACK_BUILD_DIR / "textures").items():
        m.add_material(name, texture=rel)
    for name, rel in bridge_parts.rail_textures(brs, config.TRACK_BUILD_DIR / "textures").items():
        m.add_material(name, texture=rel, alpha=True, two_sided=True)   # 格子のすき間は透明
    assembly.add_meshes(m, bridge_parts.meshes(sec, brs, rng, ground))
    from yabitsu import textures as _tx   # トンネル（筒・照明・坑口・表札）
    tspec = json.loads((config.DATA_MANUAL / "s5_tunnels.json").read_text(encoding="utf-8")) if (config.DATA_MANUAL / "s5_tunnels.json").exists() else {}
    for tn in tns:
        txt = tspec.get(str(tn.osm_id), {}).get("plate", tn.name)
        if txt:
            pth = config.TRACK_BUILD_DIR / "textures" / f"tunnel_plate_{tn.osm_id}.png"; pth.parent.mkdir(parents=True, exist_ok=True)
            _tx.sign_texture(txt, "#E9E6DC", "#1C1C1C", 512, 112, border="#555555").save(pth)
            m.add_material(f"tunnel_plate_{tn.osm_id}", texture=f"textures/{pth.name}")
    assembly.add_meshes(m, tunnel.meshes(sec, tns, tspec, ground, near=near))
    for name, v, f, uv, mat, visible in river_meshes:
        m.add_mesh(assembly.GROUP, name, v, f, uv, mat, renderable=visible, normals=river_normals.get(name))
    wspec = json.loads((config.DATA_MANUAL / "s5_water.json").read_text(encoding="utf-8"))   # S5 の滝（ストリートビュー・OSM）
    assembly.add_meshes(m, water.feature_meshes(water.manual_falls(wspec.get("falls", []), sec, ground), [], prefix="s5_water"))
    lake_poly, lake_level = water.load_lake()   # 宮ヶ瀬湖の湖面（見える湖の全体）
    if lake_poly is not None:
        assembly.add_meshes(m, water.lake_meshes(lake_poly, lake_level, water.road_area(sec, form_all, stubs)))
    if goma_site:   # 護摩屋敷の水は gomayashiki_area で作るので、OSM の湧き水の点（node 1950963807）に屋根つきの水場を置かない
        from yabitsu import geo as _geo
        springs = [e for e in springs if np.hypot(*(np.asarray(_geo.to_local(e["lat"], e["lon"]), float) - goma_site.spout)) > 15.0]
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
                           + ([tera_site.area().buffer(6.0)] if tera_site else [])
                           + ([goma_site.area().buffer(3.0)] if goma_site else [])   # 護摩屋敷の水の広場・わき道の輪（OSM のわき道を重ねない）   # 寺山富士見橋の広場・わき道・草地（OSM の小道を重ねない。OSM のわき道はストリートビューより 4 m 北にずれている）
                           + [unary_union(nagoki.sidewalk_polygons(sec, feats)).buffer(0.3)]   # 歩道の上に建物を立てない
                           + _nanohanadai_area(sec)   # 菜の花台の園地（手で作った遊歩道・展望台の道。OSM の小道を重ねない）
                           + [Polygon(p["polygon"]) for p in pads if p.get("kind") in ("lot", "pass", "site", "plaza", "branch")
                              and "house" not in p["name"] and not p.get("buildings_ok")])
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
    # コース全体の家は部品の家（house_kit）。箱に窓の絵の家はストリートビューと全く違って見えた（2026-10-28）
    from yabitsu import house_kit
    hk_blds, track_blds, hk_line, hk_road = house_kit.split(track_blds, sec)
    for name, rel in buildings.write_textures(track_blds, config.TRACK_BUILD_DIR / "textures").items():
        m.add_material(name, texture=rel)
    assembly.add_meshes(m, [(f"{n[:-3]}_trk{n[-3:]}", v, f, uv, mat, r) for n, v, f, uv, mat, r in
                            buildings.merge_by_material(buildings.building_meshes(track_blds, ground, keep_out))])
    hk_items, hk_pals = house_kit.house_meshes(hk_blds, ground, hk_road, near_line=hk_line, avoid=keep_out)
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
    for sn in ("s1", "s4", "s5"):   # 手で置いた 1 点の物（峠の売店の自動販売機など）と看板
        so = json.loads((config.DATA_MANUAL / f"{sn}_objects.json").read_text(encoding="utf-8"))
        skip += [tuple(float(v) for v in geo.to_local(*ob["points"][0])) for ob in so.get("objects", []) if len(ob.get("points", [])) == 1]
        skip += [tuple(float(v) for v in geo.to_local(*ob["at"])) for ob in so.get("pole_signs", [])]
    replaced = {(asama or {}).get("monument", {}).get("osm_id")}   # 手で作り込んだ物と同じ OSM の点（浅間神社の駐車場の石碑「愛林共栄」）は作らない
    poi_nodes = [e for e in poi_el if e.get("type") == "node" and "lat" in e and e.get("id") not in replaced]
    assembly.add_meshes(m, pois.meshes(pois.place(poi_nodes, sec, skip), ground))
    assembly.add_meshes(m, town_objects.signal_meshes(ro["signals"], road_z, prefix="s3_signal"))
    assembly.add_meshes(m, town_objects.object_meshes(town_objects.off_road(ro["objects"], sec), road_z, prefix="s3_obj"))   # 路面の中のバス停は道の外へ
    def flats_or_ground(x, y):   # 峠の面（台・歩道・植え込み・駐車場など）の上はその面の高さ、それ以外は地形の高さ
        z = np.asarray(prop_ground(x, y), float).copy()
        if pass_site:
            pz = pass_site.top_z(x, y).reshape(np.shape(z))
            z = np.where(np.isfinite(pz), pz, z)
        return z

    def pass_or_road_z(x, y):    # 峠の面の上はその面の高さ（本線の中心の高さでは歩道の物が 0.8 m 浮く）、それ以外は今までどおり
        z = np.asarray(road_z(x, y), float).copy()
        if pass_site:
            pz = pass_site.top_z(x, y).reshape(np.shape(z))
            z = np.where(np.isfinite(pz), pz, z)
        return z

    if pass_site and pass_site.spec.get("landmarks"):   # ヤビツ峠の Codex の部品（売店・トイレ・小物。data/processed/landmarks/pass_*）
        from yabitsu import landmarks as _lm
        _td = config.TRACK_BUILD_DIR / "textures"; _td.mkdir(parents=True, exist_ok=True)
        for _mat, (_png, _alpha) in _lm.materials(pass_site.spec["landmarks"]).items():
            if _png:
                _lm.copy_texture(_png, _td / f"{_mat}.png")   # 1 辺 1024 まで（kn5 は無圧縮）
                m.add_material(_mat, texture=f"textures/{_mat}.png", alpha=_alpha)
        assembly.add_meshes(m, _lm.place_ll(pass_site.spec["landmarks"], flats_or_ground, "s1_pass_lm", ground_fn=prop_ground))
    r70_path = config.DATA_MANUAL / "s45_route70.json"   # 県道 70 号（S4・S5）の Codex の部品（data/processed/landmarks/r70_*）。ミラー・標識・看板など
    if r70_path.exists():
        from yabitsu import landmarks as _lm
        r70 = json.loads(r70_path.read_text(encoding="utf-8"))["landmarks"]
        _td = config.TRACK_BUILD_DIR / "textures"; _td.mkdir(parents=True, exist_ok=True)
        for _mat, (_png, _alpha) in _lm.materials(r70).items():
            if _png:
                _lm.copy_texture(_png, _td / f"{_mat}.png")
                m.add_material(_mat, texture=f"textures/{_mat}.png", alpha=_alpha)
        r70_meshes = _lm.place_ll(r70, road_z, "r70_lm", ground_fn=prop_ground)
        for name, verts, _, _, _, _ in r70_meshes:
            if name.endswith("_post") or "_twin_post" in name:
                valleys.lower_exposed_feet(verts, verts[:, 2] <= verts[:, 2].min() + .02, dug)
        assembly.add_meshes(m, r70_meshes)
    for sn in ("s1", "s4", "s5"):   # S1（峠）・S4・S5 の標識・看板・ベンチなど（data/manual/sN_objects.json）
        so = json.loads((config.DATA_MANUAL / f"{sn}_objects.json").read_text(encoding="utf-8"))
        so["objects"] = [o for o in so["objects"] if not o.get("replaced_by")]          # Codex の部品に替えた物（峠の landmarks）は作らない
        so["pole_signs"] = [o for o in so.get("pole_signs", []) if not o.get("replaced_by")]
        on_road = [o for o in so["objects"] if o.get("z_from") != "ground"]
        on_ground = [o for o in so["objects"] if o.get("z_from") == "ground"]   # 台の上など、地形の高さに立てる物（売店の自動販売機）
        assembly.add_meshes(m, town_objects.object_meshes(town_objects.off_road(on_road, sec),
                                                          pass_or_road_z if sn == "s1" else road_z,
                                                          prefix=f"{sn}_obj"))
        if on_ground:
            assembly.add_meshes(m, town_objects.object_meshes(on_ground, flats_or_ground, prefix=f"{sn}_objg"))
        assembly.add_meshes(m, town_objects.pole_sign_meshes(so.get("pole_signs", []), flats_or_ground, prefix=sn))
        for name, rel in town_objects.write_pole_sign_textures(so.get("pole_signs", []), config.TRACK_BUILD_DIR / "textures", prefix=sn).items():
            m.add_material(name, texture=rel)
    # 町の範囲（名古木交差点）：別の kn5（config.TOWN_GROUP）
    assembly.add_meshes(m, tw.meshes(), group=config.TOWN_GROUP)
    assembly.add_meshes(m, tw.object_meshes(), group=config.TOWN_GROUP)
    for name, rel in town_objects.write_pole_sign_textures(tw.pole_signs(), config.TRACK_BUILD_DIR / "textures").items():
        m.add_material(name, texture=rel)
    lots = json.loads((config.DATA_MANUAL / "s3_lots.json").read_text(encoding="utf-8"))   # 店の駐車場・敷地の舗装と白線
    assembly.add_meshes(m, tw.lot_meshes(lots), group=config.TOWN_GROUP)
    hk_town, blds, _, _ = house_kit.split(blds, sec)
    for name, rel in buildings.write_textures(blds, config.TRACK_BUILD_DIR / "textures").items():
        m.add_material(name, texture=rel)
    avoid = tw.paved_avoid()
    assembly.add_meshes(m, buildings.merge_by_material(buildings.building_meshes(blds, tw.ground.z, avoid)), group=config.TOWN_GROUP)
    t_items, t_pals = house_kit.house_meshes(hk_town, tw.ground.z, hk_road, near_line=hk_line, avoid=avoid)
    for name, rel in house_kit.write_textures(hk_pals + t_pals, config.TRACK_BUILD_DIR / "textures").items():
        m.add_material(name, texture=rel)
    assembly.add_meshes(m, [(f"{n[:-3]}_hk{n[-3:]}", v, f, uv, mat, r) for n, v, f, uv, mat, r in house_kit.merge_by_cell(hk_items)])
    assembly.add_meshes(m, [(f"{n[:-3]}_hkt{n[-3:]}", v, f, uv, mat, r) for n, v, f, uv, mat, r in house_kit.merge_by_cell(t_items)],
                        group=config.TOWN_GROUP)
    print(f"部品の家：道ぞい {len(hk_blds)} 軒・町 {len(hk_town)} 軒、三角形 {sum(len(x[2]) for x in hk_items + t_items) * 2}", flush=True)
    form = start_lot.road_formation(sec, 0.0, float(sec.s[-1])); shapely.prepare(form)   # 当たり判定の壁が道の上に無いか（AI の車が止まった）
    on_road = sorted({n for n, v, f, uv, mat, r in hk_items + t_items if n.startswith("1WALL_") and shapely.contains_xy(form, v[:, 0], v[:, 1]).any()})
    print(f"部品の家：道の上に壁がある家 {len(on_road)} {on_road[:10]}", flush=True)
    for j, k in others:   # ほかの交差点の町：別の kn5（j.group）。名前は town → 交差点の名前
        rn = lambda items, j=j: nagoki.rename_meshes(items, j.name)
        assembly.add_meshes(m, rn(k.meshes()), group=j.group)
        assembly.add_meshes(m, rn(k.object_meshes()), group=j.group)
        for name, rel in town_objects.write_pole_sign_textures(k.pole_signs(), config.TRACK_BUILD_DIR / "textures", prefix=j.name).items():
            m.add_material(name, texture=rel)
        lp = config.DATA_MANUAL / j.lots
        if lp.exists():
            assembly.add_meshes(m, rn(k.lot_meshes(json.loads(lp.read_text(encoding="utf-8")))), group=j.group)
        kb_kit, kb, _, _ = house_kit.split(nagoki.load_buildings(j), sec)
        for name, rel in buildings.write_textures(kb, config.TRACK_BUILD_DIR / "textures").items():
            m.add_material(name, texture=rel)
        assembly.add_meshes(m, [(f"{n}_{j.name}", *rest) for n, *rest in
                                buildings.merge_by_material(buildings.building_meshes(kb, k.ground.z, k.paved_avoid()))], group=j.group)
        j_items, j_pals = house_kit.house_meshes(kb_kit, k.ground.z, hk_road, near_line=hk_line, avoid=k.paved_avoid())
        for name, rel in house_kit.write_textures(j_pals, config.TRACK_BUILD_DIR / "textures").items():
            m.add_material(name, texture=rel)
        assembly.add_meshes(m, [(f"{n[:-3]}_hk{j.name}{n[-3:]}", v, f, uv, mat, r) for n, v, f, uv, mat, r in
                                house_kit.merge_by_cell(j_items)], group=j.group)
        print(f"部品の家：{j.name} {len(kb_kit)} 軒、三角形 {sum(len(x[2]) for x in j_items) * 2}", flush=True)
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
