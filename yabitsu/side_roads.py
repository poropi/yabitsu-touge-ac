"""本線の分かれ道。入口から STUB_LENGTH_M までを OSM の道の線どおりに作り、入口を見えない壁でふさぐ
（工事中の柵の見た目は town_objects.barricade_meshes を mouth の線に置く）。

道の高さは、入口で本線の縁の高さ、そこから先は地理院の標高へなめらかに移す。地形を合わせるため、
SEG_M ごとに区切った傾いた平面（terrain.carve の pad の "plane"）にする。区切りの線の上では前後の平面の高さがそろう。"""
import math
from dataclasses import dataclass

import numpy as np
import shapely
from shapely.geometry import LineString, Polygon
from shapely.ops import substring

from . import geo, osm, site
from .crosssection import GUTTER_WIDTH
from .start_lot import road_formation

CAR_TYPES = {"primary", "secondary", "tertiary", "unclassified", "residential", "service", "living_street", "track",
             "primary_link", "secondary_link", "tertiary_link", "trunk", "trunk_link"}
DEFAULT_WIDTH = {"trunk": 7.0, "primary": 6.5, "secondary": 6.5, "tertiary": 5.5, "unclassified": 4.5,
                 "residential": 4.0, "living_street": 4.0, "service": 3.0, "track": 3.0}
STUB_LENGTH_M = 40.0
SEG_M = 10.0
MOUTH_SETBACK_M = 0.3   # 入口の壁は、本線の側溝の外側の縁からこれだけ奥
MOUTH_OVERHANG_M = 2.0  # 入口の壁は、道の両端からこれだけはみ出す（斜めに抜けられないように）
BLEND_M = 15.0          # 入口から、この長さで本線の高さから地面の高さへ移す
CW_LEN, CW_BAR, CW_EVERY = 4.0, 0.45, 0.9   # 分かれ道の横断歩道：長さ（道の向き）、帯の幅、帯の間隔


def _width(tags):
    try:
        return float(str(tags.get("width", "")).split()[0])
    except (ValueError, IndexError):
        hw = tags.get("highway", "")
        return DEFAULT_WIDTH.get(hw.replace("_link", ""), 4.0)


def _line(w):
    p = np.array([[g["lat"], g["lon"]] for g in w["geometry"]])
    x, y = geo.to_local(p[:, 0], p[:, 1])
    return LineString(np.c_[x, y])


def town_exit_s(sec, center_xy, radius):
    """本線が、center_xy を中心とする半径 radius の円を（始まりから）出る所の s。"""
    d = np.hypot(sec.x - center_xy[0], sec.y - center_xy[1])
    out = np.nonzero(d > radius)[0]
    return float(sec.s[out[0]]) if len(out) else float(sec.s[-1])


@dataclass
class Stub:
    osm_id: int
    axis: LineString        # 入口（側溝の外側の縁）から先の道の中心線
    width: float
    surface: Polygon
    mouth: LineString
    z_knots: np.ndarray     # axis の上の SEG_M ごとの高さ
    t_knots: np.ndarray

    def _segments(self):
        """(多角形, 平面) の並び。平面は区切りの両端の高さを通り、道を横切る向きには傾かない。"""
        out = []
        L = self.axis.length
        for a, b, za, zb in zip(self.t_knots[:-1], self.t_knots[1:], self.z_knots[:-1], self.z_knots[1:]):
            pa = np.array(self.axis.interpolate(a).coords[0]); pb = np.array(self.axis.interpolate(b).coords[0])
            d = pb - pa; n = np.linalg.norm(d)
            if n < 1e-6:
                continue
            u = d / n
            g = (zb - za) / n
            # 区切りの線は、区切りの点で道を横切る向き（隣どうしで同じ線を共有する）
            cut_a = self._cut(a); cut_b = self._cut(b)
            box = Polygon([*cut_a.coords, *cut_b.coords[::-1]]).buffer(0)
            piece = self.surface.intersection(box)
            if piece.is_empty or piece.area < 0.01:
                continue
            plane = [float(pa[0]), float(pa[1]), float(za), float(g * u[0]), float(g * u[1])]
            out.append((site.largest(piece) if piece.geom_type != "Polygon" else piece, plane))
            if b >= L:
                break
        return out

    def _cut(self, t):
        """axis の t の所で、道を横切る線（道の幅より十分長い）。向きは前後の区切りで同じになるよう、その点の接線で決める。"""
        L = self.axis.length
        p = np.array(self.axis.interpolate(min(t, L)).coords[0])
        q = np.array(self.axis.interpolate(min(max(t + 0.5, 0.5), L)).coords[0])
        r = np.array(self.axis.interpolate(max(min(t - 0.5, L - 0.5), 0.0)).coords[0])
        d = q - r; d = d / max(np.linalg.norm(d), 1e-9)
        nrm = np.array([-d[1], d[0]]) * (self.width + 10.0)
        if t <= 0:  # 始まりは道の外側へ少し出しておく（側溝の縁の上を切る）
            p = p - d * 0.5
        if t >= L:
            p = p + d * 0.5
        return LineString([p - nrm, p + nrm])

    def mesh(self):
        parts = [site.draped(poly, lambda x, y, pl=pl: site.plane_z(pl, x, y)) for poly, pl in self._segments()]
        vs, fs, us, base = [], [], [], 0
        for v, f, uv in parts:
            vs.append(v); fs.append(f + base); us.append(uv); base += len(v)
        return np.concatenate(vs), np.concatenate(fs), np.concatenate(us)

    def pads(self):
        return [{"name": f"side_stub_{self.osm_id}_{k}", "role": "flat", "kind": "stub",
                 "polygon": np.asarray(poly.exterior.coords).tolist(), "z": float(site.plane_z(pl, *poly.centroid.coords[0])),
                 "plane": pl} for k, (poly, pl) in enumerate(self._segments())]

    def z_at(self, x, y):
        """路面の高さ：axis に下ろした位置の、区切りの高さを直線で補う。"""
        t = self.axis.project(shapely.Point(x, y))
        return float(np.interp(t, self.t_knots, self.z_knots))

    def crosswalk_mesh(self, p):
        """横断歩道：点 p（道の上）を中心に、道の向きに長さ CW_LEN の白い帯を、道幅いっぱいに CW_EVERY おきに並べる。"""
        t = self.axis.project(shapely.Point(*p))
        c = np.array(self.axis.interpolate(t).coords[0])
        a = np.array(self.axis.interpolate(max(t - 1.0, 0)).coords[0]); b = np.array(self.axis.interpolate(min(t + 1.0, self.axis.length)).coords[0])
        d = (b - a) / max(np.linalg.norm(b - a), 1e-9); n = np.array([-d[1], d[0]])
        vs, fs = [], []
        for off in np.arange(-self.width / 2 + 0.4 + CW_BAR / 2, self.width / 2 - 0.4 - CW_BAR / 2 + 1e-9, CW_EVERY):
            q = c + n * off
            corners = [q - d * CW_LEN / 2 - n * CW_BAR / 2, q - d * CW_LEN / 2 + n * CW_BAR / 2,
                       q + d * CW_LEN / 2 + n * CW_BAR / 2, q + d * CW_LEN / 2 - n * CW_BAR / 2]
            k = len(vs)
            vs += [[*e, self.z_at(*e) + 0.01] for e in corners]
            fs.append([k, k + 1, k + 2, k + 3])
        v = np.array(vs, float)
        f = np.array(fs, np.int32)
        if (v[f[0, 1]] - v[f[0, 0]])[0] * (v[f[0, 2]] - v[f[0, 0]])[1] - (v[f[0, 1]] - v[f[0, 0]])[1] * (v[f[0, 2]] - v[f[0, 0]])[0] < 0:
            f = f[:, ::-1]   # 上向きに
        return v, f, v[:, :2]

    def wall(self, height=3.0):
        """入口をふさぐ見えない壁（両面）。"""
        (x0, y0), (x1, y1) = self.mouth.coords
        z0 = float(self.z_knots[0]) - 0.5; z1 = float(self.z_knots[0]) + height
        v = np.array([[x0, y0, z0], [x1, y1, z0], [x1, y1, z1], [x0, y0, z1]], float)
        f = np.array([[0, 1, 2, 3], [3, 2, 1, 0]], np.int32)
        return v, f, v[:, :2] / 2.0


def stubs(elements, sec, sampler, keep_out, s_min=0.0, length_m=STUB_LENGTH_M, unproject=geo.from_local):
    """本線に s_min より先でつながる分かれ道（OSM の車道。県道70号・歩道を除く）を、入口から length_m まで切り出す。
    keep_out（ピット・広場の多角形）にかかる道は作らない。"""
    form = road_formation(sec, s_min=0.0, s_max=float(sec.s[-1]))
    r70 = {id(w) for w in osm.route70_ways(elements)}
    out = []
    for w in elements:
        t = w.get("tags", {})
        if w.get("type") != "way" or id(w) in r70 or t.get("highway") not in CAR_TYPES or len(w.get("geometry", [])) < 2:
            continue
        if t.get("bridge", "no") != "no" or t.get("tunnel", "no") != "no":
            continue
        line = _line(w)
        if not line.intersects(form.buffer(0.5)):
            continue
        rest = line.difference(form)
        pieces = list(rest.geoms) if rest.geom_type.startswith("Multi") else [rest]
        for pc in pieces:
            if pc.is_empty or pc.length < 1.0:
                continue
            a, b = np.array(pc.coords[0]), np.array(pc.coords[-1])
            from shapely.geometry import Point
            da, db = form.distance(Point(a)), form.distance(Point(b))
            if min(da, db) > 0.3:
                continue
            if db < da:
                pc = LineString(list(pc.coords)[::-1])
            start = np.array(pc.coords[0])
            i = int(np.argmin(np.hypot(sec.x - start[0], sec.y - start[1])))
            if sec.s[i] < s_min:
                continue
            axis = substring(pc, 0.0, min(length_m, pc.length))
            width = _width(t)
            if any(axis.buffer(width / 2).intersects(Polygon(k) if not isinstance(k, Polygon) else k) for k in keep_out):
                continue
            # 入口の手前へ少し延ばしてから本線の範囲を引く（斜めに交わるとき、本線との間にすき間を残さない）
            p1 = np.array(axis.interpolate(min(1.0, axis.length)).coords[0])
            d = (p1 - start) / max(np.linalg.norm(p1 - start), 1e-9)
            ext = LineString([start - d * (width + 2.0), *axis.coords])
            surf = ext.buffer(width / 2, cap_style="flat").difference(form)
            surf = site.largest(surf) if surf.geom_type != "Polygon" else surf
            if surf.is_empty or surf.area < 1.0:
                continue
            n = np.array([-d[1], d[0]])
            m0 = start + d * MOUTH_SETBACK_M
            half = width / 2 + MOUTH_OVERHANG_M
            mouth = LineString([m0 - n * half, m0 + n * half])
            # 高さ：入口は本線の縁の高さ、BLEND_M で地面へ
            hw_side = 1 if (-(start[0] - sec.x[i]) * math.sin(sec.heading[i]) + (start[1] - sec.y[i]) * math.cos(sec.heading[i])) > 0 else -1
            z_edge = float(sec.edge_z(hw_side)[i])
            tk = np.r_[np.arange(0.0, axis.length, SEG_M), axis.length]
            pts = np.array([axis.interpolate(v).coords[0] for v in tk])
            zg = np.asarray(sampler.sample(*unproject(pts[:, 0], pts[:, 1])), float)
            zg = np.where(np.isnan(zg), z_edge, zg)
            wgt = np.clip(tk / BLEND_M, 0.0, 1.0); wgt = wgt * wgt * (3 - 2 * wgt)
            zk = (1 - wgt) * z_edge + wgt * zg
            out.append(Stub(int(w.get("id", 0)), axis, width, surf, mouth, zk, tk))
    return out


ROADSIDE_FILES = ("osm_nagoki.json", "osm_s3_roadside.json", "osm_track_roads.json", "osm_shintomei.json")


def load_elements(raw_dir=None):
    """道ぞいの OSM（町・S3・S2/S1）をまとめて読む。同じ way は1つにする。"""
    import json
    from . import config
    seen, out = set(), []
    for f in ROADSIDE_FILES:
        p = (raw_dir or config.DATA_RAW) / f
        if not p.exists():
            raise FileNotFoundError(f"{p} がありません（scripts.track_01_fetch で取る）")
        for e in json.loads(p.read_text(encoding="utf-8"))["elements"]:
            k = (e.get("type"), e.get("id"))
            if k not in seen:
                seen.add(k); out.append(e)
    return out


def stub_meshes(stub_list):
    """(名前, 頂点, 面, UV, 材質, 描画するか)。路面は走れる面（入口の壁の手前で車が落ちないように）。"""
    out = []
    for k, st in enumerate(stub_list):
        out.append((f"1ROAD_side_stub_{k:03d}", *st.mesh(), "asphalt", True))
        out.append((f"1WALL_side_block_{k:03d}", *st.wall(), "concrete", False))
    return out


KEEP_OPEN_KINDS = ("lot", "pass", "site", "plaza")
NOT_OPENINGS = ("_house", "_shrine", "side_stub")   # 家・お堂の土台は「ふさがない場所」ではない（家の前の道もふさぐ）


def for_track(sec, pads, sampler=None, exclude=None):
    """本線全体の分かれ道（町の範囲の中は除く）。ピット・広場・駐車場の入口はふさがない。
    exclude（町の範囲など）と重なる所は路面から削る（町の道と二重にならないように）。"""
    from . import config, gsi
    cx, cy = geo.to_local(*config.NAGOKI)
    s_min = town_exit_s(sec, (float(cx), float(cy)), config.TOWN_RADIUS_M)
    keep = [Polygon(p["polygon"]) for p in pads
            if (p.get("kind") in KEEP_OPEN_KINDS or p.get("role") in ("start", "mid", "end"))
            and not any(k in p.get("name", "") for k in NOT_OPENINGS)]
    out = stubs(load_elements(), sec, sampler or gsi.default_sampler(), keep, s_min=s_min)
    if exclude is not None:
        for st in out:
            g = st.surface.difference(exclude)
            if not g.is_empty:
                st.surface = site.largest(g) if g.geom_type != "Polygon" else g
    return [st for st in out if st.surface.area >= 1.0]
