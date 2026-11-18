"""県道70号のトンネル（OSM の tunnel=yes の way）を、本線の s の範囲にする。高さの合わせ（標高データ＝山の表面を使わない）、
地形（道の上の山を削らない）、トンネルの筒・照明・坑口・表札のメッシュに使う。"""
from dataclasses import dataclass

import numpy as np
from shapely.geometry import LineString, Point

from . import geo, osm


@dataclass
class Tunnel:
    osm_id: int
    name: str
    s0: float
    s1: float


def from_osm(elements, sec, near_m=3.0):
    """sec（s・x・y を持つもの）の上に乗っている、県道70号のトンネルの way の範囲。"""
    main = LineString(np.c_[sec.x, sec.y])
    out = []
    for w in osm.route70_ways(elements):
        t = w.get("tags", {})
        if w.get("type") != "way" or t.get("tunnel", "no") in ("no", "") or len(w.get("geometry", [])) < 2:
            continue
        c = np.array([geo.to_local(g["lat"], g["lon"]) for g in w["geometry"]], float)
        if max(main.distance(Point(p)) for p in c) > near_m:
            continue
        ss = [float(sec.s[int(np.argmin(np.hypot(sec.x - p[0], sec.y - p[1])))]) for p in (c[0], c[-1])]
        sh = _shifts().get(str(w.get("id", 0)), {})
        out.append(Tunnel(int(w.get("id", 0)), t.get("name", "") if "トンネル" in t.get("name", "") else "",
                          min(ss) + float(sh.get("s0_shift_m", 0.0)), max(ss) + float(sh.get("s1_shift_m", 0.0))))
    return out


def _shifts():
    """data/manual/s5_tunnels.json の s0_shift_m・s1_shift_m（OSM の端をストリートビューに合わせてずらす量 m。奥へ＝s が増える向きが正）。"""
    import json
    from . import config
    p = config.DATA_MANUAL / "s5_tunnels.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def row_mask(s, tunnels, margin_m=5.0):
    """s のうち、どれかのトンネルの範囲（前後 margin_m を含む）に入る所。"""
    s = np.asarray(s, float); m = np.zeros(len(s), bool)
    for t in tunnels:
        m |= (s >= t.s0 - margin_m) & (s <= t.s1 + margin_m)
    return m


WALL_H = 3.0          # 縦の壁の高さ（その上が半円のアーチ）
LAMP_EVERY = 10.0
STEP = 2.0
N_ARC = 12


def _frame(sec, i):
    from .crosssection import GUTTER_WIDTH
    h = sec.heading[i]; n = np.array([-np.sin(h), np.cos(h)]); c = np.array([sec.x[i], sec.y[i]])
    l = sec.hw_left[i] + GUTTER_WIDTH; r = sec.hw_right[i] + GUTTER_WIDTH
    return c, n, l, r


def bore_height(sec, i):
    """路面の中心から筒の天井までの高さ：縦の壁 WALL_H ＋ 半円の半径（筒の幅の半分）。"""
    _, _, l, r = _frame(sec, i)
    return WALL_H + (l + r) / 2.0


def _profile(sec, i):
    """行 i の筒の断面（右の下 → 右の壁 → アーチ → 左の壁 → 左の下）。[(lat, dz)]。dz は路面の中心からの高さ。"""
    _, _, l, r = _frame(sec, i)
    rad = (l + r) / 2.0; cl = (l - r) / 2.0
    pts = [(-r, -0.3), (-r, WALL_H)]
    for a in np.linspace(np.pi, 0, N_ARC + 1)[1:-1]:
        pts.append((cl + rad * np.cos(a), WALL_H + rad * np.sin(a)))
    return pts + [(l, WALL_H), (l, -0.3)]


def _rows(sec, t):
    return [int(np.argmin(np.abs(sec.s - s))) for s in np.r_[np.arange(t.s0, t.s1, STEP), t.s1]]


def meshes(sec, tunnels, spec, ground_z):
    """トンネルの筒（当たり判定つき、内向き）・照明・両側の坑口（アーチの穴のあいた面）・表札。
    spec = {"<osm_id>": {"plate": 表札の文字}}。ground_z はできあがった地形の高さ（坑口の面の上の端に使う）。"""
    out = []
    for t in tunnels:
        rows = _rows(sec, t)
        V, F = [], []
        for k, i in enumerate(rows):
            c, n, _, _ = _frame(sec, i)
            for lat, dz in _profile(sec, i):
                p = c + n * lat; V.append([p[0], p[1], sec.z[i] + dz])
        m = len(_profile(sec, rows[0]))
        for k in range(len(rows) - 1):
            for j in range(m - 1):
                a, b = k * m + j, k * m + j + 1
                F.append([a, b, b + m, a + m])
        v = np.array(V, float); f = np.array(F, np.int32)
        from .meshgen import face_normals
        cen = v[f].mean(1); nr = face_normals(v, f)
        ii = np.array([rows[min(q // m, len(rows) - 1)] for q in f[:, 0]])
        axis = np.c_[sec.x[ii], sec.y[ii], sec.z[ii] + WALL_H]
        if np.mean(np.sum((axis - cen) * nr, axis=1) > 0) < 0.5:
            f = f[:, ::-1]
        out.append((f"1WALL_tunnel_{t.osm_id}_bore", v, f, np.c_[np.arange(len(v)) % m, np.repeat(np.arange(len(rows)), m) * STEP / 4.0], "tunnel_wall", True))
        lamps = []
        for s in np.arange(t.s0 + 3.0, t.s1 - 2.0, LAMP_EVERY):
            i = int(np.argmin(np.abs(sec.s - s))); c, n, l, r = _frame(sec, i)
            d = np.array([-n[1], n[0]]); zt = sec.z[i] + bore_height(sec, i) - 0.6
            for lat in (-(l + r) * 0.22, (l + r) * 0.22):
                p = c + n * lat
                q = [p - d * 0.6 - n * 0.12, p + d * 0.6 - n * 0.12, p + d * 0.6 + n * 0.12, p - d * 0.6 + n * 0.12]
                lamps.append([[*e, zt] for e in q])
        if lamps:
            lv = np.array(lamps, float).reshape(-1, 3); lf = np.arange(len(lv), dtype=np.int32).reshape(-1, 4)
            out.append((f"tunnel_{t.osm_id}_lamps", lv, np.r_[lf, lf[:, ::-1]], lv[:, :2], "tunnel_lamp", True))
        sp = (spec or {}).get(str(t.osm_id), {})
        for tag, s, sgn in (("a", t.s0, -1.0), ("b", t.s1, 1.0)):
            out += _portal(sec, t, tag, s, sgn, ground_z, sp.get("plate", t.name), sp.get("portal", "concrete"))
    return out


RIM_W, RIM_D = 0.5, 0.25    # 坑口のアーチの縁：幅・手前への出っ張り
HEAD_T = 0.6                # 坑口の壁の厚さ（天端のふた）
PORTAL_IN_DEPTHS = (3.0, 4.0, 5.0, 6.0, 7.0)   # 盛り土を合わせる山の高さを測る所（坑口からの奥行き）。筒にかかる地形の面は 6 m 奥まで抜けることがある
PORTAL_SIDE = 4.0           # 坑口の壁を筒の外へ広げる幅（筒にかかる地形の面を抜く幅：2 m の格子の真ん中が筒の外 terrain.NEAR_CELL＝2 m まで、
                            # 格子は道に斜めに並ぶので面の角は 2＋1.4 m まで届く。2026-11-01 の光線の確かめで 3 m では端にすき間）
HEAD_ABOVE = 0.3            # 坑口の壁の天端：アーチの縁の上
BANK_RISE = 1.5             # 盛り土：壁の天端のふたの奥から山の高さまで上がる土手の奥行き
BACK_D = 7.5                # 盛り土：山の高さのまま続く奥行き（坑口から）。その 1.5 m 奥で地面の下へもぐる
BACK_LIFT = 0.2             # 盛り土：山の高さより上げる量（抜いた面の縁に切れ目を作らない）


def _backfill(c, n, o, lats, top, ks, gz):
    """坑口の壁の上の盛り土（山肌と同じ地面）：壁のふたの奥（HEAD_T）で天端の高さから、BANK_RISE の土手で山の高さ（ks＋BACK_LIFT、
    列ごとの PORTAL_IN_DEPTHS の一番高い所）まで上がり、BACK_D まで続いて、その 1.5 m 奥で地面の 0.5 m 下へもぐる。左右の端は地面の 1 m 下まで
    下ろす。筒にかかる地形の面（terrain.tunnel_holes）を上からふさぎ、壁の上から山の中が見えない。表は上向き（端は外向き）。"""
    from .meshgen import face_normals
    ds = np.array([HEAD_T, HEAD_T + BANK_RISE, 3.0, 4.5, 6.0, BACK_D, BACK_D + 1.5])
    rows = []
    for j, d in enumerate(ds):
        row = []
        for lat, kz in zip(lats, ks):
            p = c + n * lat - o * d
            z = top if j == 0 else (gz(p) - 0.5 if j == len(ds) - 1 else max(top, kz + BACK_LIFT))
            row.append([*p, z])
        rows.append(row)
    P = np.array(rows, float); nr, nc = P.shape[:2]
    idx = np.arange(nr * nc).reshape(nr, nc)
    f = np.stack([idx[:-1, :-1], idx[:-1, 1:], idx[1:, 1:], idx[1:, :-1]], -1).reshape(-1, 4)
    v = P.reshape(-1, 3)
    if np.mean(face_normals(v, f)[:, 2] > 0) < 0.5:
        f = f[:, ::-1]
    parts = [(v, f, v[:, :2] / 8.0)]
    for col, side in ((0, -1.0), (nc - 1, 1.0)):   # 左右の端：地面の 1 m 下まで
        e = P[:, col]; low = np.array([[*q[:2], min(q[2], gz(q[:2])) - 1.0] for q in e])
        sv = np.r_[e, low]; m_ = len(e)
        sf = np.array([[k, k + 1, m_ + k + 1, m_ + k] for k in range(m_ - 1)], np.int32)
        if np.mean((face_normals(sv, sf)[:, :2] @ (n * side)) > 0) < 0.5:
            sf = sf[:, ::-1]
        along = (sv[:, :2] - c) @ (-o)
        parts.append((sv, sf, np.c_[along, sv[:, 2]] / 8.0))
    from .start_lot import _merge
    return _merge(parts)
PORTAL_STYLES = {           # s5_tunnels.json の portal → (壁の材質, アーチの縁の材質)。坑門は山の仕上げの Codex の部品で、壁は筒のまわりの細い部分だけが見える。
    # 縁は Codex の坑口の広い口の内側から見えるので、トンネルの内壁と同じ材質（苔のコンクリートの絵は白黒の市松模様に見えた。2026-11-01 の撮影）
    "stone": ("rock_moss", "tunnel_wall"),             # 大洞トンネル：苔の付いた岩（間知石の菱形の絵は白い金網に、苔のコンクリート・灰色の石は白く見えた）
    "concrete_old": ("rock_moss", "tunnel_wall"),      # S5 4091 m の小さなトンネル：坑門の左右は岩（ストリートビュー R70-0717_f。苔のコンクリートは白く見えた）
    "concrete": ("concrete_grey", "tunnel_wall"),      # 灰色のコンクリート（吹風トンネル）
}


def _orient(v, f, out_dir):
    """面 f の表を out_dir（水平 2D）へ向ける。"""
    from .meshgen import face_normals
    nr = face_normals(v, f)[:, :2] @ np.asarray(out_dir, float)
    return f if np.mean(nr > 0) >= 0.5 else f[:, ::-1]


def _portal(sec, t, tag, s, sgn, ground_z, plate, style="concrete"):
    """坑口：s の所で道を横切る縦の壁（筒の断面の所は穴、その外は下（路面−1m か外の地形の下）から上（内外の地形の高い方＋0.5m）まで）。
    表は外向きの 1 枚（表と裏を同じ所に重ねると縞に見えた。2026-10-28）で、天端に厚さ HEAD_T のふた。アーチの輪郭に沿って、幅 RIM_W・
    手前へ RIM_D 出た縁（外の面・内側・外周）。材質は PORTAL_STYLES（壁・縁）。表札は縁の上。"""
    wall_mat, rim_mat = PORTAL_STYLES.get(style, (style, style))
    i = int(np.argmin(np.abs(sec.s - s))); c, n, l, r = _frame(sec, i); zc = sec.z[i]
    d = np.array([-n[1], n[0]]); o = -d * sgn         # o：坑口から外（道の手前）の向き（d は道の進む向きの逆）
    prof = _profile(sec, i)
    arch = lambda lat: max([dz for (la, dz) in prof if abs(la - lat) < 1e-6] or [np.interp(lat, [p[0] for p in prof[1:-1]], [p[1] for p in prof[1:-1]])])
    lats = np.r_[np.arange(-r - PORTAL_SIDE, -r, 1.0), [p[0] for p in prof[1:-1]], np.arange(l + PORTAL_SIDE, l, -1.0)]   # 両端は筒の外へちょうど PORTAL_SIDE
    lats = np.unique(np.round(lats, 6))
    gz = lambda q: float(np.atleast_1d(ground_z(np.array([q[0]]), np.array([q[1]])))[0])
    # 壁は筒のまわり（筒にかかる地形の面を抜く幅＋1 m）だけで、天端はアーチの縁のすぐ上（ストリートビューの坑門。前は左右に 12 m 広げ、
    # 天端を山の高さまで上げたので、写真に無い大きな灰色の壁になった。2026-11-01）。その上から奥の山へは盛り土（_backfill）でつなぐ
    top = zc + bore_height(sec, i) + RIM_W + HEAD_ABOVE
    V, F, tops, ks = [], [], [], []
    for k, lat in enumerate(lats):
        p = c + n * lat
        # 山を残す所（坑口より PORTAL_INSET 奥から）の一番高い所。坑口と山のあいだの山肌は筒にかかる面を描かない（terrain.tunnel_holes）
        # ので、盛り土をこの高さまで上げて上からふさぐ（2026-10-31 にゲームで壁の上に鏡のような帯）
        g_in = max(gz(p - o * k) for k in PORTAL_IN_DEPTHS)
        g_out = gz(p + o * 1.0)   # 坑口の 1 m 手前
        bot = zc + arch(lat) if -r - 1e-6 <= lat <= l + 1e-6 else min(zc - 1.0, g_out - 0.5, g_in - 0.5)
        V += [[*p, bot], [*p, top]]; tops.append([*p, top]); ks.append(g_in)
        if k:
            a = 2 * (k - 1); F.append([a, a + 2, a + 3, a + 1])
    v = np.array(V, float); f = _orient(v, np.array(F, np.int32), o)
    uv = np.c_[np.repeat(lats, 2) / 2.0, v[:, 2] / 2.0]
    tp = np.array(tops); tb = tp - np.r_[o * HEAD_T, 0.0]          # 天端のふた（上向き）
    cv = np.r_[tp, tb]; m_ = len(tp)
    cf = np.array([[k, k + 1, m_ + k + 1, m_ + k] for k in range(m_ - 1)], np.int32)
    from .meshgen import face_normals
    if np.mean(face_normals(cv, cf)[:, 2] > 0) < 0.5:
        cf = cf[:, ::-1]
    from .start_lot import _merge
    out = [(f"tunnel_{t.osm_id}_portal_{tag}", *_merge([(v, f, uv), (cv, cf, cv[:, :2] / 2.0)]), wall_mat, True)]
    out.append((f"1GRASS_s1_terrain_tunnel_{t.osm_id}_{tag}", *_backfill(c, n, o, lats, top, np.array(ks), gz), "ground", True))
    # アーチの縁：輪郭（右の下 → アーチ → 左の下）の内側の線と、外へ RIM_W 広げた線
    rad = (l + r) / 2.0; cl = (l - r) / 2.0
    inner = [(-r, 0.0)] + [(la, dz) for la, dz in prof[1:-1]] + [(l, 0.0)]
    outer = []
    for la, dz in inner:
        if dz > WALL_H + 1e-6:
            k_ = 1.0 + RIM_W / rad; outer.append((cl + (la - cl) * k_, WALL_H + (dz - WALL_H) * k_))
        else:
            outer.append((la + (RIM_W if la > cl else -RIM_W), dz))
    P = lambda la, dz, f_: [*(c + n * la + o * f_), zc + dz]
    rv, rf = [], []
    for k in range(len(inner) - 1):
        quads = [(inner[k], inner[k + 1], outer[k + 1], outer[k], RIM_D, RIM_D),      # 手前の面
                 (inner[k], inner[k + 1], inner[k + 1], inner[k], 0.0, RIM_D),        # 内側（穴の縁）
                 (outer[k], outer[k + 1], outer[k + 1], outer[k], 0.0, RIM_D)]        # 外周
        for qa, qb, qc, qd, fa, fb in quads:
            b0 = len(rv)
            if fa == fb:
                rv += [P(*qa, fa), P(*qb, fa), P(*qc, fa), P(*qd, fa)]
            else:
                rv += [P(*qa, fa), P(*qb, fa), P(*qc, fb), P(*qd, fb)]
            rf.append([b0, b0 + 1, b0 + 2, b0 + 3]); rf.append([b0 + 3, b0 + 2, b0 + 1, b0])   # 細い帯なので両面（向きの判定を省く）
    rv = np.array(rv, float)
    out.append((f"tunnel_{t.osm_id}_rim_{tag}", rv, np.array(rf, np.int32), np.c_[np.arange(len(rv)) % 4 / 2.0, rv[:, 2] / 2.0], rim_mat, True))
    if plate:
        zt = zc + bore_height(sec, i) + RIM_W + 0.3; q = c + o * (0.08)
        pv = np.array([[*(q - n * 1.6), zt], [*(q + n * 1.6), zt], [*(q + n * 1.6), zt + 0.7], [*(q - n * 1.6), zt + 0.7]])
        pf = np.array([[0, 1, 2, 3]] if sgn < 0 else [[3, 2, 1, 0]], np.int32)
        out.append((f"tunnel_{t.osm_id}_plate_{tag}", pv, np.r_[pf, pf[:, ::-1]], np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float), f"tunnel_plate_{t.osm_id}", True))
    return out


PORTAL_INSET = 2.5   # 山を残す所（terrain.carve の keep_rows）は、坑口からこれだけ奥から（2m の地形の格子が坑口の前へはみ出さない）


def carve_mask(s, tunnels):
    """terrain.carve の keep_rows：トンネルの範囲を両端 PORTAL_INSET ずつ縮めたもの。"""
    return row_mask(s, tunnels, margin_m=-PORTAL_INSET)
