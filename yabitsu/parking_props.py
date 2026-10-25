"""菜の花台の駐車車両と見物客。静的景観用で衝突メッシュは作らない。

車：設定に ac_car（content/cars の車）があれば、kn5 には入れず、Lua（CSP のコースのスクリプト）がゲームの起動時に
その車の LOD を読み込んで置く（車のデータを配布物に含めないため。車が入っていない AC では置かない）。
ac_car が無い車は自作の形。人：Microsoft Rocketbox（MIT、assets/free/rocketbox）を腕を下ろした立ち姿で固めたもの。
モデルのローカル軸はx=右、y=前、z=上。駐車場のa/bへ回転して置く。
"""
import json
import shutil
from pathlib import Path

import numpy as np

from . import config, kn5, start_lot
from .manifest import Manifest

GROUP = "yabitsu_parking_props"
COLORS = {
    "body_silver": (0.48, 0.51, 0.53), "body_blue": (0.08, 0.18, 0.29),
    "body_white": (0.78, 0.77, 0.71), "glass": (0.10, 0.15, 0.19),
    "rubber": (0.035, 0.038, 0.042), "wheel": (0.40, 0.42, 0.44),
    "headlamp": (0.75, 0.79, 0.73), "taillamp": (0.54, 0.055, 0.035),
    "plate": (0.78, 0.78, 0.70),
}


def _part(name, verts, faces, material):
    v = np.asarray(verts, float); f = np.asarray(faces, np.int32)
    # 凸な各部品の巻き順を外向きにする。窓など薄い部品も体積を持つ。
    n = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    wrong = np.sum(n * (v[f].mean(1) - v.mean(0)), axis=1) < 0
    f[wrong] = f[wrong, ::-1]
    return name, v, f, material


def _ellipsoid(name, center, radii, material, segments=12, rings=6):
    verts = [[0, 0, -1]]
    for j in range(1, rings):
        el = -np.pi / 2 + np.pi * j / rings
        for i in range(segments):
            a = 2 * np.pi * i / segments
            verts.append([np.cos(el) * np.cos(a), np.cos(el) * np.sin(a), np.sin(el)])
    verts.append([0, 0, 1]); faces = []
    for i in range(segments):
        ni = (i + 1) % segments
        faces.append([0, 1 + i, 1 + ni])
        for j in range(rings - 2):
            a = 1 + j * segments + i; b = 1 + j * segments + ni
            faces.extend([[a, b, b + segments], [a, b + segments, a + segments]])
        a = 1 + (rings - 2) * segments
        faces.append([len(verts) - 1, a + ni, a + i])
    return _part(name, np.asarray(verts) * radii + center, faces, material)


def _loft(name, sections, material):
    """y位置・半幅・底面z・上面zから面取り八角断面をつなぐ。"""
    vertices = []
    for y, w, lo, hi in sections:
        h = hi - lo
        vertices.extend([[w * x, y, lo + h * z] for x, z in
                         [(-0.8, 0), (0.8, 0), (1, .16), (1, .77), (.78, 1), (-.78, 1), (-1, .77), (-1, .16)]])
    faces = []
    for j in range(len(sections) - 1):
        for i in range(8):
            a = j * 8 + i; b = j * 8 + (i + 1) % 8
            faces.extend([[a, b, b + 8], [a, b + 8, a + 8]])
    for offset in (0, (len(sections) - 1) * 8):
        for i in range(1, 7):
            faces.append([offset, offset + i, offset + i + 1])
    return _part(name, vertices, faces, material)


def _tube(name, start, end, radius, material, segments=10):
    a = np.asarray(start, float); b = np.asarray(end, float); d = b - a; d /= np.linalg.norm(d)
    u = np.cross(d, [0, 0, 1] if abs(d[2]) < .9 else [0, 1, 0]); u /= np.linalg.norm(u)
    w = np.cross(d, u); angles = np.arange(segments) * 2 * np.pi / segments
    ring = radius * (np.cos(angles)[:, None] * u + np.sin(angles)[:, None] * w)
    v = np.r_[a + ring, b + ring, [a, b]]; faces = []
    for i in range(segments):
        j = (i + 1) % segments
        faces.extend([[i, j, j + segments], [i, j + segments, i + segments],
                      [2 * segments, j, i], [2 * segments + 1, i + segments, j + segments]])
    return _part(name, v, faces, material)



def _panel(name, front, thickness, material):
    front = np.asarray(front, float)
    v = np.r_[front, front - thickness]
    f = [[0,1,2],[0,2,3],[4,6,5],[4,7,6]]
    for i in range(4):
        j = (i + 1) % 4
        f.extend([[i,j,j+4],[i,j+4,i+4]])
    return _part(name,v,f,material)


def _clothed_limb(name, sections, material, segments=12):
    """肩/腰から袖口/裾まで、曲がる一枚の衣服表面を張る。"""
    angles = np.arange(segments) * 2 * np.pi / segments
    verts = []
    for center, radius in sections:
        verts.extend(np.asarray(center) + np.c_[np.cos(angles)*radius,np.sin(angles)*radius,np.zeros(segments)])
    faces = []
    for row in range(len(sections)-1):
        for i in range(segments):
            a=row*segments+i; b=row*segments+(i+1)%segments
            faces.extend([[a,b,b+segments],[a,b+segments,a+segments]])
    for row in (0,len(sections)-1):
        center_idx=len(verts); verts.append(sections[row][0])
        for i in range(segments):
            faces.append([center_idx,row*segments+i,row*segments+(i+1)%segments])
    return _part(name,verts,faces,material)

def car_meshes(prefix, material):
    p = lambda name: f"{prefix}_{name}"
    out = [_loft(p("body"), [(-2.0,.69,.34,.77),(-1.7,.84,.29,.92),(-.9,.87,.29,.94),
                              (1.1,.85,.29,.87),(1.8,.77,.34,.76),(2.0,.65,.4,.68)], material),
           _loft(p("cabin"), [(-1.5,.74,.84,1.02),(-1.0,.70,.85,1.48),(.25,.70,.85,1.46),(.95,.70,.83,.94)], material)]
    # ガラスをキャビンの面から少し外に置く。左右の窓はBピラーで分かれる。
    for side in (-1, 1):
        for label, yz in (("rear", [(-.96,.98),(-.23,.98),(-.23,1.29),(-.96,1.29)]),
                          ("front",[(-.12,.98),(.65,.98),(.20,1.29),(-.12,1.29)])):
            points=np.array([[side*.708,y,z] for y,z in yz])
            out.append(_panel(p(f"window_{side}_{label}"),points,np.array([side*.012,0,0]),"glass"))
            for edge in range(4):
                out.append(_tube(p(f"window_frame_{side}_{label}_{edge}"),points[edge],points[(edge+1)%4],.012,"rubber",6))
        # ボディ外板上のドア継ぎ目。上下端をつなぎパネル輪郭を作る。
        for door,label in (((-1.02,-.17),"rear"),((-.12,.91),"front")):
            ya,yb=door
            points=np.array([[side*(.87-(y+.9)*.01+.004),y,z] for y,z in [(ya,.47),(yb,.47),(yb,.72),(ya,.72)]])
            for edge in range(4):
                out.append(_tube(p(f"door_seam_{side}_{label}_{edge}"),points[edge],points[(edge+1)%4],.004,"rubber",6))
        out.append(_ellipsoid(p(f"mirror_{side}"), [side*.93,.59,1.01], [.13,.15,.075], material, 10, 4))
        for axle, y in enumerate((-1.27, 1.24)):
            out.append(_tube(p(f"tyre_{side}_{axle}"), [side*.72,y,.29], [side*.91,y,.29], .29,"rubber",16))
            out.append(_tube(p(f"wheel_{side}_{axle}"), [side*.913,y,.29], [side*.923,y,.29], .19,"wheel",12))
            # 暗い奥面に5本の明るいスポークを重ね、円盤だけの車輪にしない。
            out.append(_tube(p(f"wheel_inset_{side}_{axle}"),[side*.924,y,.29],[side*.926,y,.29],.155,"rubber",16))
            for spoke in range(5):
                ang=2*np.pi*spoke/5
                direction=np.array([0,np.cos(ang),np.sin(ang)])
                center=np.array([side*.934,y,.29])
                out.append(_tube(p(f"spoke_{side}_{axle}_{spoke}"),center+.045*direction,center+.17*direction,.021,"wheel",6))
            out.append(_tube(p(f"hub_{side}_{axle}"), [side*.935,y,.29], [side*.941,y,.29], .05,"wheel",10))
        out.append(_ellipsoid(p(f"headlamp_{side}"), [side*.53,1.88,.64],[.24,.07,.09],"headlamp",10,4))
        out.append(_ellipsoid(p(f"taillamp_{side}"), [side*.64,-1.91,.73],[.12,.065,.14],"taillamp",10,4))
        out.append(_ellipsoid(p(f"handle_{side}"), [side*.864,-.12,.88],[.015,.11,.02],"wheel",8,4))
    # 前後窓はキャビンの屋根勾配に合わせ、埋没しない薄い台形板にする。
    for name, rows in (("windscreen", ((.31,.49,1.425),(.87,.56,1.008))),
                       ("rear_window", ((-1.43,.50,1.094),(-1.07,.49,1.425)))):
        (ya,wa,za),(yb,wb,zb) = rows
        top = np.array([[-wa,ya,za],[wa,ya,za],[wb,yb,zb],[-wb,yb,zb]])
        v = np.r_[top,top - [0,0,.012]]
        f = [[0,1,2],[0,2,3],[4,6,5],[4,7,6]]
        for i in range(4):
            j=(i+1)%4
            f.extend([[i,j,j+4],[i,j+4,i+4]])
        out.append(_part(p(name),v,f,"glass"))
    for label, y, z, halfwidth in (("hood",1.36,.835,.66),("bumper_front",1.997,.54,.58),("bumper_rear",-2.002,.56,.60)):
        out.append(_tube(p(label+"_seam"),[-halfwidth,y,z],[halfwidth,y,z],.006,"rubber",8))
    out.append(_ellipsoid(p("grille"),[0,1.999,.47],[.45,.012,.065],"rubber",12,4))
    out.append(_ellipsoid(p("rear_plate"),[0,-2.01,.53],[.21,.012,.065],"plate",8,4))
    return out


HUMAN_DIR = config.PROJECT_ROOT / "assets" / "free" / "rocketbox"


def person_meshes(prefix, model):
    """Microsoft Rocketbox の人（MIT、assets/free/rocketbox）を腕を下ろした立ち姿で。足の裏が z=0、y が前。
    材質ごとの部品 (名前, 頂点, 面, 材質, UV)。材質は person_<テクスチャ名>（体・頭・髪などの透過）。"""
    d = np.load(HUMAN_DIR / f"{model}.npz")
    v = d["verts"].astype(float)
    v[:, :2] -= (v[:, :2].min(0) + v[:, :2].max(0)) / 2
    v[:, 2] -= v[:, 2].min()
    v[:, :2] *= -1          # 素材は -y を向いているので 180° 回す
    out = []
    for k, mn in enumerate(d["mat_names"]):
        tri = np.nonzero(d["mat"] == k)[0]
        if not len(tri):
            continue
        idx = d["faces"][tri].reshape(-1)
        out.append((f"{prefix}_{str(mn).split('_', 1)[1]}", v[idx], np.arange(len(idx), dtype=np.int32).reshape(-1, 3),
                    f"person_{mn}", d["uv"][idx].astype(float)))
    return out


def person_textures(model):
    """{材質名: (画像のパス, 透過か)}"""
    return {f"person_{mn}": (HUMAN_DIR / f"{mn}.png", str(mn).endswith("_opacity")) for mn in np.load(HUMAN_DIR / f"{model}.npz")["mat_names"]}


def _pose(frame, item, z):
    angle = frame.heading + np.radians(item["yaw_deg"])
    c, s_ = np.cos(angle), np.sin(angle)
    return np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1]]), np.array([*frame.xy(item["a"], item["b"]), z])


def placed_meshes(lot_spec, z, settings):
    """(名前, 頂点, 面, 材質, UV or None)。ac_car の車は Lua で置くのでここでは作らない。"""
    frame = start_lot.frame(lot_spec); out = []
    for idx, item in enumerate(settings["cars"]):
        if item.get("ac_car"):
            continue
        rot, center = _pose(frame, item, z)
        for name, v, f, mat in car_meshes(f"park_car_{idx}", item["material"]):
            out.append((name, v @ rot.T + center, f, "parking_" + mat, None))
    for idx, item in enumerate(settings["people"]):
        rot, center = _pose(frame, item, z)
        for name, v, f, mat, uv in person_meshes(f"park_person_{idx}", item["model"]):
            out.append((name, v * item.get("scale", 1) @ rot.T + center, f, mat, uv))
    return out


def ac_cars(lot_spec, z, settings):
    """Lua で置く AC の車：[{id, lod, main, pos(AC 座標), look(AC 座標)}]。AC の座標は (x, z, -y)。"""
    frame = start_lot.frame(lot_spec); out = []
    for item in settings["cars"]:
        if not (car := item.get("ac_car")):
            continue
        rot, center = _pose(frame, item, z)
        fwd = rot @ np.array([0.0, 1.0, 0.0])
        out.append({**car, "pos": [float(center[0]), float(center[2]) + car.get("y_offset", 0.0), float(-center[1])],
                    "look": [float(fwd[0]), 0.0, float(-fwd[1])]})
    return out


def pass_ac_cars(settings):
    """峠の駐車場の車（parking_props.json の pass_cars：{stall, index, ac_car}）。位置と向きは s1_pass.json の駐車枠から。"""
    if not settings.get("pass_cars"):
        return []
    from . import pass_area
    from .crosssection import Section
    sec = Section.load_json(config.DATA_PROCESSED / "track_section.json")
    site_ = pass_area.build(pass_area.load(config.DATA_MANUAL / "s1_pass.json"), sec)
    where = {(n, i): (x, y, z, h) for n, i, x, y, z, h in site_.parked_cars()}
    out = []
    for item in settings["pass_cars"]:
        x, y, z, h = where[(item["stall"], int(item["index"]))]
        car = item["ac_car"]
        if car.get("wheels"):   # 4 つのタイヤの接地点で路面に合わせる（駐車場は傾いている。原点の高さも車ごとに違う）
            z0, look, up = settle(site_.park_z, x, y, h, car["wheels"])
            out.append({**car, "pos": [x, z0, -y], "look": [look[0], look[2], -look[1]], "up": [up[0], up[2], -up[1]]})
        else:
            out.append({**car, "pos": [x, z + car.get("y_offset", 0.0), -y], "look": [float(np.cos(h)), 0.0, float(-np.sin(h))]})
    return out


def settle(z_fn, x, y, h, wheels):
    """車を路面に置く：(原点の高さ, 前の向き, 上の向き)（どれもローカル座標 x 東・y 北・z 上）。
    wheels = {"front"/"rear": [輪距の半分 m, タイヤの底の高さ（車のモデルの原点から）m, 車軸の前後の位置 m]}。
    4 つのタイヤの底が、その下の路面に乗る平面（最小二乗）に車の向きと高さを合わせる。"""
    d = np.array([np.cos(h), np.sin(h)]); n = np.array([-np.sin(h), np.cos(h)])
    rows, tgt = [], []
    for key in ("front", "rear"):
        half, bottom, axle = (float(v) for v in wheels[key])
        for s in (1.0, -1.0):
            q = np.array([x, y]) + d * axle + n * s * half
            rows.append([1.0, axle, s * half]); tgt.append(float(np.ravel(z_fn(q[0], q[1]))[0]) - bottom)
    (z0, b, c), *_ = np.linalg.lstsq(np.array(rows), np.array(tgt), rcond=None)
    look = np.array([*d, b]); look /= np.linalg.norm(look)
    up = np.array([*(-b * d - c * n), 1.0]); up /= np.linalg.norm(up)
    return float(z0), [float(v) for v in look], [float(v) for v in up]


def cars_lua(cars):
    v = lambda a: "vec3(%.3f,%.3f,%.3f)" % tuple(a)
    rows = ",\n".join(f'{{id="{c["id"]}",lod="{c["lod"]}",main="{c["main"]}",pos={v(c["pos"])},look={v(c["look"])},up={v(c.get("up", (0, 1, 0)))}}}' for c in cars)
    return f"""-- Generated by yabitsu.parking_props. 菜の花台・ヤビツ峠の駐車車両：各自の AC に入っている車の LOD を読み込んで置く（車のデータは配布しない）。
local cars = {{
{rows}
}}
local root = ac.findNodes('trackRoot:yes')
for _, c in ipairs(cars) do
  -- コースのスクリプトは content/cars を読めない（絶対・相対パスとも 2026-10-23 実機で失敗）。
  -- 組み込むときに、その PC の AC の車のファイルを extension/parked_cars/<車>/ へコピーしておき、そこから読む
  local dir = 'parked_cars/' .. c.id .. '/'
  local ok, n = pcall(function() return root:loadKN5LOD(dir .. c.lod, dir .. c.main) end)
  if ok and n then
    n:setPosition(c.pos)
    n:setOrientation(c.look, c.up)   -- 上の向き：駐車場の傾きに合わせる
  end
  ac.log('YABITSU_PARKED ' .. c.id .. ' ok=' .. tostring(ok) .. ' node=' .. tostring(n))
end
"""


def build(out_dir=None):
    out_dir = Path(out_dir or config.DATA_PROCESSED / "parking_props")
    lot_spec = start_lot.load_spec(config.DATA_MANUAL / "s1_start_lot.json")
    settings = json.loads((config.DATA_MANUAL / "parking_props.json").read_text(encoding="utf-8"))
    pads = json.loads((config.DATA_PROCESSED / "track_pads.json").read_text(encoding="utf-8"))
    pad = next(p for p in pads if p["name"] == start_lot.LOT_NAME)
    m = Manifest(out_dir)
    for name,color in COLORS.items():
        m.add_material("parking_" + name,color=color)
    (out_dir / "textures").mkdir(parents=True, exist_ok=True)
    for mat, (src, alpha) in {k: t for p in settings["people"] for k, t in person_textures(p["model"]).items()}.items():
        shutil.copy2(src, out_dir / "textures" / src.name)
        m.add_material(mat, texture=f"textures/{src.name}", alpha=alpha, two_sided=alpha)
    for name,v,f,mat,uv in placed_meshes(lot_spec,pad["z"],settings):
        m.add_mesh(GROUP,name,v,f,v[:,:2] * .1 if uv is None else uv,mat)
    m.save()
    (out_dir / "retex.json").write_text("{}",encoding="utf-8")
    (out_dir / "hidden.json").write_text("[]",encoding="utf-8")
    ext = out_dir / "extension"; (ext / "parts").mkdir(parents=True, exist_ok=True)
    (ext / "parked_cars.lua").write_text(cars_lua(ac_cars(lot_spec, pad["z"], settings) + pass_ac_cars(settings)), encoding="utf-8")
    (ext / "parts" / "parked_cars.ini").write_text("[SCRIPT_...]\nACTIVE=1\nSCRIPT=parked_cars.lua\n", encoding="utf-8")
    return kn5.write_from_manifest(out_dir / "manifest.json",GROUP,out_dir / f"{GROUP}.kn5")
