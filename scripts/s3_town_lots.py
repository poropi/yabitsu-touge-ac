"""町の範囲（名古木交差点 200m）の、道の外の舗装（店の駐車場・敷地・車庫の前）と、駐車場の白線を空中写真から拾い、
data/manual/s3_lots.json に書く（{"lots": [[[lat, lon], ...], ...], "lines": [[[lat, lon], ...], ...]}）。
- 舗装：色の薄い（灰色の）画素で、建物・車道・歩道の外、道（歩道）に 3m 以内で接する 60m² 以上のかたまり
- 白線：舗装の中の、とても明るい細長いかたまり（白い車を拾わないよう、面積 4m² 未満で細長いものだけ）
確認画像：data/processed/review/s3_lots.png"""
import json

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

from yabitsu import config, geo, gsi, nagoki, photo
from yabitsu.crosssection import Section

P = config.DATA_PROCESSED
PX = 0.25
R = config.TOWN_RADIUS_M + 10


def _mask_polys(mask, x0, y0, size, min_area):
    """True の画素のかたまりを多角形（ローカル座標）に。"""
    import shapely
    lab, n = ndimage.label(mask)
    out = []
    for k, sl in enumerate(ndimage.find_objects(lab), start=1):
        if sl is None:
            continue
        sub = lab[sl] == k
        if sub.sum() * PX * PX < min_area:
            continue
        rows, cols = np.nonzero(sub)
        cells = [box(x0 + (c + sl[1].start) * PX, y0 + size - (r + sl[0].start + 1) * PX,
                     x0 + (c + sl[1].start + 1) * PX, y0 + size - (r + sl[0].start) * PX) for r, c in zip(rows, cols)]
        g = shapely.union_all(cells)
        out += [g] if g.geom_type == "Polygon" else list(g.geoms)
    return out


# 交差点に近い店の駐車場：空中写真（10m 目盛り、交差点の中心からの距離 [東, 北] m）で測った形。自動で拾った分と合わせる
MANUAL_LOTS = {
    "セブン-イレブン 名古木": [(-42, 5), (0, 5), (0, 46), (-10, 46), (-10, 29), (-31, 29), (-31, 26), (-42, 26)],
    "クーリクのまわり": [(-130, 4), (-82, 4), (-82, 52), (-95, 62), (-118, 82), (-140, 82), (-140, 62), (-130, 62)],
    "北東の青い屋根の建物の前": [(5, 8), (62, 8), (62, 38), (55, 45), (5, 45)],
    "北東の工場の駐車場": [(55, 100), (135, 100), (135, 128), (55, 128)],
    "ENEOS のまわり": [(-75, -3), (-22, -3), (-22, -28), (-40, -42), (-75, -42)],
    "南西の店の駐車場": [(-140, -12), (-112, -12), (-112, -42), (-140, -42)],
}
# 駐車枠の白線：[始め, 終わり]（枠の並ぶ線）、奥行きの向き（"S"/"N"/"E"/"W"）
STALLS = [((-41, 27), (-12, 27), "S"), ((-41, 6), (-12, 6), "N"), ((-2, 31), (-2, 45), "W")]
STALL_W, STALL_D, LINE_W = 2.5, 5.0, 0.15


def stall_lines(cx, cy):
    out = []
    dirs = {"S": (0, -1), "N": (0, 1), "E": (1, 0), "W": (-1, 0)}
    for (a, b, side) in STALLS:
        a = np.array(a, float); b = np.array(b, float); L = np.linalg.norm(b - a); u = (b - a) / L
        dv = np.array(dirs[side], float)
        for t in np.arange(0.0, L + 1e-6, STALL_W):
            p0 = np.array([cx, cy]) + a + u * t; p1 = p0 + dv * STALL_D
            from shapely.geometry import LineString
            out.append(LineString([p0, p1]).buffer(LINE_W / 2, cap_style="flat"))
    return out


def main():
    sec = Section.load_json(P / "track_section.json")
    tw = nagoki.load_town(sec)
    cx, cy = tw.center
    x0, y0, size = cx - R, cy - R, 2 * R
    img = photo.render_local_image(gsi.TileCache(), geo.from_local, x0, y0, size, PX).astype(float) / 255.0
    mx, mn = img.max(-1), img.min(-1)
    sat = np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0)
    green = (img[..., 1] > img[..., 0] * 1.03) & (img[..., 1] > img[..., 2] * 1.03)
    warm = img[..., 0] > img[..., 2] * 1.10                      # 土・れんが・赤いコート
    lum = img.mean(-1)
    smooth = ndimage.uniform_filter(lum * lum, 9) - ndimage.uniform_filter(lum, 9) ** 2 < 0.004   # 木や草のざらつきがない
    paved = (sat < 0.16) & (mx > 0.38) & (mx < 0.92) & ~green & ~warm & smooth
    bright = (mn > 0.80) & (sat < 0.12)
    n = paved.shape[0]
    # 建物・車道・歩道を除く（建物は 0.8m 広げて）
    blds = unary_union([b.footprint.buffer(0.8) for b in nagoki.load_buildings()])
    taken = blds.union(tw.carriageway).union(tw.sidewalk)
    im = Image.new("L", (n, n), 0); d = ImageDraw.Draw(im)
    to_px = lambda c: [((x - x0) / PX, (y0 + size - y) / PX) for x, y in c]
    for g in ([taken] if taken.geom_type == "Polygon" else list(taken.geoms)):
        d.polygon(to_px(g.exterior.coords), fill=255)
        for h in g.interiors:
            d.polygon(to_px(h.coords), fill=0)
    free = np.array(im) == 0
    lotmask = ndimage.binary_opening(ndimage.binary_closing(paved & free, iterations=4), iterations=3)
    lots = []
    road = tw.carriageway.union(tw.sidewalk).buffer(3.0)
    circle = nagoki.circle()
    for g in _mask_polys(lotmask, x0, y0, size, 60.0):
        g = g.buffer(0.5).buffer(-0.5).simplify(0.4).intersection(circle).difference(taken)
        for q in ([g] if g.geom_type == "Polygon" else [h for h in getattr(g, "geoms", []) if h.geom_type == "Polygon"]):
            if q.area >= 60.0 and q.intersects(road):
                lots.append(q)
    manual = [Polygon([(cx + x, cy + y) for x, y in pts]).buffer(0).difference(taken) for pts in MANUAL_LOTS.values()]
    lots += [q for g in manual for q in ([g] if g.geom_type == "Polygon" else list(getattr(g, "geoms", []))) if q.area > 5.0]
    lot_union = unary_union(lots)
    lots = [q for q in ([lot_union] if lot_union.geom_type == "Polygon" else list(lot_union.geoms)) if q.area >= 5.0]
    lines = []
    for g in _mask_polys(bright & lotmask, x0, y0, size, 0.3):
        r = g.minimum_rotated_rectangle
        e = np.asarray(r.exterior.coords)
        a, b = np.linalg.norm(e[1] - e[0]), np.linalg.norm(e[2] - e[1])
        if g.area < 4.0 and max(a, b) / max(min(a, b), 0.05) > 4.0 and lot_union.contains(g.centroid):
            lines.append(r)
    lines += [g for g in stall_lines(cx, cy)]
    ll = lambda q: [[round(float(v), 8) for v in geo.from_local(x, y)] for x, y in q.exterior.coords]
    out = {"note": "空中写真から自動で拾った、道の外の舗装と駐車場の白線（scripts.s3_town_lots）",
           "lots": [ll(q) for q in lots], "lines": [ll(q) for q in lines]}
    (config.DATA_MANUAL / "s3_lots.json").write_text(json.dumps(out, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    rv = Image.fromarray((img * 255).astype(np.uint8)); dr = ImageDraw.Draw(rv, "RGBA")
    for q in lots:
        dr.polygon(to_px(q.exterior.coords), outline=(255, 0, 255, 255), fill=(255, 0, 255, 50))
    for q in lines:
        dr.polygon(to_px(q.exterior.coords), outline=(0, 255, 255, 255))
    rv.resize((1100, 1100)).save(P / "review" / "s3_lots.png")
    print(f"lots {len(lots)} ({lot_union.area:.0f} m²), lines {len(lines)}")


if __name__ == "__main__":
    main()
