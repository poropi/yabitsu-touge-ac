"""川・沢（OSM の waterway=river|stream）を、谷の底の水面と川原（石の帯）として作る。砂防ダム・堰（OSM の waterway=weir|dam）と
湧き水の水くみ場も置く。水は動かさない。

- 水面の高さ：川の線に沿って 2m ごとに標高を取り、下流へ向かって「それまでの最小」でならし（上流へ上がらない）、20m の平均で
  5m 標高のでこぼこを消す。橋（本線の bridge の行）の下では「橋の路面 − BRIDGE_CLEAR_M」を上限にする
- 地形：水面のまわりに溝を彫る（terrain.carve の channels）。本線の路面（橋を除く）の下は彫らない
- 描かない所：road_area（本線の路面＋側溝（橋を除く）、分かれ道の路面）の 0.5m 内。橋の下には水面が見える"""
import numpy as np
import shapely
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import substring, unary_union

from . import geo
from .town import grid_drape, _polys

WIDTH = {"river": 6.0, "stream": 2.5}
BED_W = 1.5            # 水面の両側の川原
STEP_M, SMOOTH_M = 2.0, 20.0
BRIDGE_CLEAR_M = 2.0   # 橋の路面から水面までの最小
BANK = 1.0             # 溝の岸の勾配（水平 1m で 1m 上がる）
WEIR_H = 1.5           # 堰・砂防ダムの高さ（決まった値。OSM に高さは無い）
LIFT = 0.02


def _xy(w):
    return np.array([geo.to_local(g["lat"], g["lon"]) for g in w["geometry"]], float).reshape(-1, 2)


class Stream:
    def __init__(self, line, width_m, kind, z, osm_id=0):
        self.line, self.width_m, self.kind, self.z, self.osm_id = line, float(width_m), kind, np.asarray(z, float), osm_id
        self.t = np.linspace(0.0, line.length, len(self.z))

    def z_at(self, t):
        return np.interp(t, self.t, self.z)

    def z_fn(self, dz=0.0):
        def f(x, y):
            x = np.atleast_1d(x); y = np.atleast_1d(y)
            t = shapely.line_locate_point(self.line, shapely.points(np.c_[np.ravel(x), np.ravel(y)]))
            return (self.z_at(t) + dz).reshape(np.shape(x))
        return f

    def water(self):
        return self.line.buffer(self.width_m / 2, cap_style="flat")

    def channel(self):
        """(溝の範囲, 川底の高さの関数)：terrain.carve の channels に渡す。"""
        return self.line.buffer(self.width_m / 2 + BED_W, cap_style="flat"), self.z_fn(-0.3)


def road_area(sec, formation, stubs=()):
    """水を描かない所：本線の路面＋側溝のうち橋でない所と、分かれ道の路面。"""
    return unary_union([formation.difference(_bridges(sec))] + [st.surface for st in stubs])


def _bridges(sec):
    """本線の橋の行の、路面＋側溝（+1m）の範囲。"""
    from .start_lot import road_formation
    out, i = [], 0
    b = np.asarray(sec.bridge, bool)
    while i < sec.n:
        if b[i]:
            j = i
            while j + 1 < sec.n and b[j + 1]:
                j += 1
            out.append(road_formation(sec, float(sec.s[max(i - 1, 0)]), float(sec.s[min(j + 1, sec.n - 1)])).buffer(1.0))
            i = j + 1
        else:
            i += 1
    return unary_union(out) if out else Polygon()


def from_osm(elements, sec, sampler, band_m=200.0, s_from=0.0, unproject=geo.from_local):
    """本線の s_from より先から band_m 以内の川・沢。線の向き（OSM）は下流向き。"""
    m = sec.s >= s_from
    band = LineString(np.c_[sec.x[m], sec.y[m]]).buffer(band_m)
    bridges = _bridges(sec)
    deck = lambda p: float(sec.z[int(np.argmin(np.hypot(sec.x - p[0], sec.y - p[1])))])
    out = []
    for w in elements:
        tg = w.get("tags", {})
        if w.get("type") != "way" or tg.get("waterway") not in WIDTH or len(w.get("geometry", [])) < 2:
            continue
        if tg.get("tunnel", "no") not in ("no",):
            continue
        ln = LineString(_xy(w))
        if not band.intersects(ln):
            continue
        for piece in _polys_lines(ln.intersection(band)):
            if piece.length < 10.0:
                continue
            t = np.r_[np.arange(0.0, piece.length, STEP_M), piece.length]
            p = np.array([piece.interpolate(v).coords[0] for v in t])
            zg = np.asarray(sampler.sample(*unproject(p[:, 0], p[:, 1])), float)
            zg = np.where(np.isnan(zg), np.nanmin(zg) if np.isfinite(np.nanmin(zg)) else 0.0, zg)
            z = np.minimum.accumulate(zg)
            k = max(1, int(SMOOTH_M / STEP_M))
            z = np.convolve(np.pad(z, k, mode="edge"), np.ones(2 * k + 1) / (2 * k + 1), mode="same")[k:-k]
            if not bridges.is_empty:
                under = shapely.contains_xy(bridges, p[:, 0], p[:, 1])
                for i in np.nonzero(under)[0]:
                    z[i] = min(z[i], deck(p[i]) - BRIDGE_CLEAR_M)
            z = np.minimum.accumulate(z)
            width = float(tg["width"]) if str(tg.get("width", "")).replace(".", "", 1).isdigit() else WIDTH[tg["waterway"]]
            out.append(Stream(LineString(p), width, tg["waterway"], z, int(w.get("id", 0))))
    _chain(out)
    return out


def _chain(streams, join_m=1.0):
    """way のつなぎ目で、下流の水面が上流の終わりより高くならないようにする：上流の way の終わりが別の way の上
    （始まりでも途中でも）にあれば、そこから下流をその高さ以下にする。変わらなくなるまでくり返す。"""
    for _ in range(len(streams) + 1):
        changed = False
        for up in streams:
            end = Point(up.line.coords[-1]); z_end = float(up.z[-1])
            for dn in streams:
                if dn is up or dn.line.distance(end) > join_m:
                    continue
                t = dn.line.project(end)
                m = dn.t >= t - 1e-6
                if np.any(dn.z[m] > z_end + 1e-9):
                    dn.z[m] = np.minimum(dn.z[m], z_end); dn.z = np.minimum.accumulate(dn.z); changed = True
        if not changed:
            break


def _polys_lines(g):
    if g.is_empty:
        return []
    if g.geom_type == "LineString":
        return [g]
    return [q for q in getattr(g, "geoms", []) if q.geom_type == "LineString"]


def meshes(streams, road):
    """(名前, 頂点, 面, UV, 材質, 描画するか)。水面 water、川原 riverbed。road の 0.5m 内には描かない。"""
    keep_out = road.buffer(0.5) if not road.is_empty else road
    out = []
    for k, st in enumerate(streams):
        wtr = st.water().difference(keep_out)
        bed = st.line.buffer(st.width_m / 2 + BED_W, cap_style="flat").difference(st.water()).difference(keep_out)
        for j, (v, f, uv) in enumerate(grid_drape(wtr, st.z_fn(LIFT), 0.0, cell=2.0)):
            out.append((f"water_{k:03d}_{j:02d}", v, f, uv, "water", True))
        for j, (v, f, uv) in enumerate(grid_drape(bed, st.z_fn(0.15), 0.0, cell=2.0)):
            out.append((f"riverbed_{k:03d}_{j:02d}", v, f, uv, "riverbed", True))
    return out


def falls_from_osm(elements, streams, near_m=3.0):
    """OSM の堰・砂防ダム（waterway=weir|dam）のうち、川の線から near_m 以内のもの：[{"line", "z_top", "z_bottom", "down"}]。"""
    out = []
    for w in elements:
        if w.get("type") != "way" or w.get("tags", {}).get("waterway") not in ("weir", "dam") or len(w.get("geometry", [])) < 2:
            continue
        ln = LineString(_xy(w))
        st = min(streams, key=lambda s: s.line.distance(ln), default=None)
        if st is None or st.line.distance(ln) > near_m:
            continue
        hit = st.line.intersection(ln)
        c = hit if hit.geom_type == "Point" else ln.interpolate(0.5, normalized=True)
        t = st.line.project(c)
        a = np.array(st.line.interpolate(max(t - 1.0, 0)).coords[0]); b = np.array(st.line.interpolate(min(t + 1.0, st.line.length)).coords[0])
        z = float(st.z_at(t))
        out.append({"line": ln, "z_top": z + 0.2, "z_bottom": z - WEIR_H, "down": (b - a) / max(np.linalg.norm(b - a), 1e-9)})
    return out


def feature_meshes(falls, springs, prefix="water"):
    """堰・砂防ダム（コンクリートの段＋下流側の白い水の落ちる面）と、湧き水の水くみ場（屋根・水場）。"""
    out = []
    for k, fl in enumerate(falls):
        c = np.asarray(fl["line"].coords); d = fl["down"]
        a, b = c[0], c[-1]; zt, zb = fl["z_top"], fl["z_bottom"]
        th = 0.8
        q = np.array([a, b, b + d * th, a + d * th])
        v = np.r_[np.c_[q, np.full(4, zt)], np.c_[q, np.full(4, zb)]]
        f = np.array([[0, 1, 2, 3], [4, 7, 6, 5], [0, 4, 5, 1], [1, 5, 6, 2], [2, 6, 7, 3], [3, 7, 4, 0]], np.int32)
        out.append((f"{prefix}_weir_{k:02d}", v, f, v[:, :2] / 2.0, "concrete", True))
        fa = np.array([a + d * (th + 0.05), b + d * (th + 0.05)])
        vf = np.array([[*fa[0], zb], [*fa[1], zb], [*fa[1], zt - 0.05], [*fa[0], zt - 0.05]])
        out.append((f"{prefix}_fall_{k:02d}", vf, np.array([[0, 1, 2, 3], [3, 2, 1, 0]], np.int32), vf[:, [0, 2]], "water_fall", True))
    for k, sp in enumerate(springs):
        x, y, z, h = sp["x"], sp["y"], sp["z"], sp.get("facing", 0.0)
        d = np.array([np.sin(h), np.cos(h)]); n = np.array([d[1], -d[0]])
        corners = lambda w, l: np.array([[x, y] + n * w + d * l, [x, y] - n * w + d * l, [x, y] - n * w - d * l, [x, y] + n * w - d * l])
        tr = corners(0.6, 0.3)
        v = np.r_[np.c_[tr, np.full(4, z)], np.c_[tr, np.full(4, z + 0.6)]]
        f = np.array([[4, 5, 6, 7], [0, 4, 5, 1], [1, 5, 6, 2], [2, 6, 7, 3], [3, 7, 4, 0]], np.int32)
        out.append((f"{prefix}_spring_trough_{k:02d}", v, f, v[:, :2], "concrete", True))
        rf = corners(1.2, 1.0)
        vr = np.c_[rf, np.array([z + 2.3, z + 2.3, z + 2.0, z + 2.0])]
        out.append((f"{prefix}_spring_roof_{k:02d}", vr, np.array([[0, 1, 2, 3], [3, 2, 1, 0]], np.int32), vr[:, :2], "roof_dark", True))
        for j, pc in enumerate(corners(1.0, 0.8)):
            pv = np.array([[*(pc + [-0.06, -0.06]), z], [*(pc + [0.06, -0.06]), z], [*(pc + [0.06, -0.06]), z + 2.2], [*(pc + [-0.06, -0.06]), z + 2.2]])
            out.append((f"{prefix}_spring_post_{k:02d}_{j}", pv, np.array([[0, 1, 2, 3], [3, 2, 1, 0]], np.int32), pv[:, [0, 2]], "wood", True))
    return out


def load_elements():
    """S4 の川・沢・堰（data/raw/osm_s4_places.json の waterway）と湧き水（osm_s4_nature.json）。"""
    import json
    from . import config
    el = json.loads((config.DATA_RAW / "osm_s4_places.json").read_text(encoding="utf-8"))["elements"]
    nat = json.loads((config.DATA_RAW / "osm_s4_nature.json").read_text(encoding="utf-8"))["elements"]
    return [e for e in el if "waterway" in e.get("tags", {})], [e for e in nat if e.get("tags", {}).get("natural") == "spring"]


def load(sec, sampler=None):
    """S4（峠の先）の川・沢（Stream のリスト）、堰・砂防ダム、湧き水。"""
    from . import gsi, side_roads
    ways, springs = load_elements()
    streams = from_osm(ways, sec, sampler or gsi.default_sampler(), s_from=side_roads.s4_start(sec))
    lake, level = load_lake()
    if lake is not None:   # 宮ヶ瀬湖に入る川は、湖の手前で湖の水位につなぐ
        streams = attach_to_lake(streams, lake, level)
    return streams, falls_from_osm(ways, streams), springs


def channels(sec, streams, formation):
    """terrain.carve に渡す溝。本線の路面（橋を除く）のまわり 2m は彫らない。"""
    keep = formation.difference(_bridges(sec)).buffer(2.0)
    out = []
    for st in streams:
        poly, zf = st.channel()
        poly = poly.difference(keep)
        if not poly.is_empty:
            out.append((poly, zf))
    return out


def spring_items(springs, sec, ground_z, near_m=200.0):
    """湧き水の水くみ場：OSM の点に、道の方を向けて置く（道から near_m 以内）。"""
    out = []
    for e in springs:
        x, y = (float(v) for v in geo.to_local(e["lat"], e["lon"]))
        i = int(np.argmin(np.hypot(sec.x - x, sec.y - y)))
        if np.hypot(sec.x[i] - x, sec.y[i] - y) > near_m:
            continue
        out.append({"x": x, "y": y, "z": float(np.atleast_1d(ground_z(np.array([x]), np.array([y])))[0]),
                    "facing": float(np.arctan2(sec.x[i] - x, sec.y[i] - y)), "name": e["tags"].get("name", "")})
    return out


def lake_from_osm(elements):
    """湖の relation（outer・inner の way をつなぐ）を多角形（ローカル座標）に。outer・inner は way の向きがばらばらでもよい。"""
    from shapely.ops import polygonize, linemerge
    nodes = {e["id"]: e for e in elements if e.get("type") == "node"}
    ways = {e["id"]: e for e in elements if e.get("type") == "way"}
    rel = next(e for e in elements if e.get("type") == "relation")
    rings = {"outer": [], "inner": []}
    for m in rel.get("members", []):
        w = ways.get(m.get("ref"))
        if m.get("type") != "way" or w is None or m.get("role") not in rings:
            continue
        pts = [geo.to_local(nodes[n]["lat"], nodes[n]["lon"]) for n in w.get("nodes", []) if n in nodes]
        if len(pts) >= 2:
            rings[m["role"]].append(LineString(np.array(pts, float)))
    outer = unary_union(list(polygonize(linemerge(rings["outer"])))) if rings["outer"] else Polygon()
    inner = unary_union(list(polygonize(linemerge(rings["inner"])))) if rings["inner"] else Polygon()
    return outer.difference(inner).buffer(0)


def lake_meshes(poly, level, keep_out, cell=8.0):
    """湖面（平ら、高さ level）。keep_out（本線・橋・分かれ道）の 0.5m 内には描かない。"""
    g = poly.difference(keep_out.buffer(0.5)) if not keep_out.is_empty else poly
    return [(f"lake_{k:02d}", v, f, uv, "water", True)
            for k, (v, f, uv) in enumerate(grid_drape(g, lambda x, y: np.full(np.shape(x), float(level)), LIFT, cell=cell))]


def load_lake():
    """宮ヶ瀬湖（data/raw/osm_s5_lake_rel.json）と水位（data/manual/s5_water.json）。なければ (None, None)。"""
    import json
    from . import config
    p, q = config.DATA_RAW / "osm_s5_lake_rel.json", config.DATA_MANUAL / "s5_water.json"
    if not (p.exists() and q.exists()):
        return None, None
    return lake_from_osm(json.loads(p.read_text(encoding="utf-8"))["elements"]), float(json.loads(q.read_text(encoding="utf-8"))["lake_level_m"])


def attach_to_lake(streams, lake, level):
    """湖の中の川は描かない（線を湖の外だけにする）。水面は湖の水位より下げない（湖に入る所で段差なくつながる）。"""
    out = []
    for st in streams:
        rest = st.line.difference(lake)
        for piece in _polys_lines(rest):
            if piece.length < 5.0:
                continue
            t0, t1 = st.line.project(Point(piece.coords[0])), st.line.project(Point(piece.coords[-1]))
            tt = np.linspace(t0, t1, max(2, int(piece.length / STEP_M) + 1))
            z = np.maximum(st.z_at(tt), level)
            out.append(Stream(LineString([st.line.interpolate(v).coords[0] for v in tt]), st.width_m, st.kind, np.minimum.accumulate(z), st.osm_id))
    return out


def manual_falls(spec, sec, ground_z):
    """data/manual/s5_water.json の falls（ストリートビュー・OSM で見た滝）：feature_meshes の形に。
    facing_deg がなければ一番近い本線の点の方へ落ちる。"""
    out = []
    for f in spec:
        p = np.array([float(v) for v in geo.to_local(*f["at"])])
        if f.get("facing_deg") is None:
            i = int(np.argmin(np.hypot(sec.x - p[0], sec.y - p[1]))); d = np.array([sec.x[i], sec.y[i]]) - p
        else:
            r = np.radians(float(f["facing_deg"])); d = np.array([np.sin(r), np.cos(r)])
        d = d / max(np.linalg.norm(d), 1e-9); n = np.array([-d[1], d[0]]); w = float(f.get("width_m", 2.0)) / 2
        zb = float(np.atleast_1d(ground_z(np.array([p[0]]), np.array([p[1]])))[0])
        out.append({"line": LineString([p - n * w, p + n * w]), "z_top": zb + float(f.get("height_m", 3.0)), "z_bottom": zb, "down": d})
    return out
