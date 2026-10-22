"""遠景：今の遠い地形（track_terrain_far.npz の長方形）の外を、粗い格子の地形と海で地平線まで埋める。
菜の花台・峠から南に見える秦野の街・渋沢丘陵・相模湾が、今は何も無い（2026-10-22 に確かめた）。
空中写真（地理院 seamlessphoto）を 1 枚の画像にして真上から貼る。走れない（当たり判定なし）。座標は manifest と同じ。"""
import numpy as np

from .meshgen import compact

HALF = 22000.0     # 遠い地形の長方形の中心から、縦横それぞれ ± m（相模湾の海岸 約 12.8 km の先まで）
CELL = 100.0
DROP = 3.0         # 遠い地形の縁と重なる所で上に出ないよう、少し下げる
OVERLAP = 1        # 遠い地形の長方形の内側へ、何マス重ねるか（すき間を出さない）
BLOCK = 60


def grid(far, half=HALF, cell=CELL):
    """遠い地形の長方形の中心を真ん中にした格子 (xs, ys)。"""
    cx = (far["xs"][0] + far["xs"][-1]) / 2
    cy = (far["ys"][0] + far["ys"][-1]) / 2
    xs = np.arange(np.floor((cx - half) / cell) * cell, cx + half + cell, cell)
    ys = np.arange(np.floor((cy - half) / cell) * cell, cy + half + cell, cell)
    return xs, ys


def heights(xs, ys, sample):
    """sample(X, Y) → 標高（海・データなしは nan）。海と nan は 0 m、全体を DROP 下げる。"""
    X, Y = np.meshgrid(xs, ys)
    Z = np.asarray(sample(X, Y), float).reshape(X.shape)
    return np.where(np.isfinite(Z), np.maximum(Z, 0.0), 0.0) - DROP


def face_keep(xs, ys, far, overlap=OVERLAP):
    """面（格子のマス）を作るか：遠い地形の長方形の外（内側へ overlap マス重ねる）。"""
    cell = xs[1] - xs[0]
    cx = (xs[:-1] + xs[1:]) / 2
    cy = (ys[:-1] + ys[1:]) / 2
    x0, x1 = far["xs"][0] + overlap * cell, far["xs"][-1] - overlap * cell
    y0, y1 = far["ys"][0] + overlap * cell, far["ys"][-1] - overlap * cell
    inside = ((cx[None, :] > x0) & (cx[None, :] < x1)) & ((cy[:, None] > y0) & (cy[:, None] < y1))
    return ~inside


def meshes(xs, ys, Z, keep, block=BLOCK):
    """[(名前, verts, faces, uvs)]。uv は全体の四角（xs・ys の端）を 0〜1 にする（v=0 が南＝画像の下）。"""
    X, Y = np.meshgrid(xs, ys)
    verts = np.c_[X.ravel(), Y.ravel(), Z.ravel()]
    uvs = np.c_[(verts[:, 0] - xs[0]) / (xs[-1] - xs[0]), (verts[:, 1] - ys[0]) / (ys[-1] - ys[0])]
    i = np.arange(X.size).reshape(X.shape)
    faces = np.stack([i[:-1, :-1], i[:-1, 1:], i[1:, 1:], i[1:, :-1]], -1)
    ry, rx = np.meshgrid(np.arange(Z.shape[0] - 1) // block, np.arange(Z.shape[1] - 1) // block, indexing="ij")
    out = []
    for k, (by, bx) in enumerate(sorted(set(zip(ry[keep].tolist(), rx[keep].tolist())))):
        m = keep & (ry == by) & (rx == bx)
        v, f, uv = compact(verts, faces[m].reshape(-1, 4), uvs)
        out.append((f"HORIZON_{k:03d}", v, f, uv))
    return out
