"""細分三角形の上にある谷筋に下り勾配を与え、岸の排水と実測標高を保つ。"""

import numpy as np
import shapely
from scipy.sparse import coo_matrix, diags, eye, vstack
from scipy.sparse.linalg import factorized
from scipy.spatial import cKDTree

from .valley_surface import drain_to_boundary


def channel_derivatives(xy, faces, rows):
    """各線分と交わる三角形ごとに、頂点高から下流方向微分を得る係数を作る。"""
    triangles = xy[faces]
    polygons = shapely.polygons(triangles)
    tree = shapely.STRtree(polygons)
    crossed, coefficients = [], []
    for row in rows:
        segments = shapely.linestrings(np.stack((row.line[:-1], row.line[1:]), axis=1))
        seg, face = tree.query(segments, predicate="intersects")
        intersection = shapely.intersection(segments[seg], polygons[face])
        traverses = shapely.length(intersection) > 1e-8
        seg, face = seg[traverses], face[traverses]
        direction = np.diff(row.line, axis=0)[seg]
        direction /= np.linalg.norm(direction, axis=1)[:, None]
        tri = triangles[face]
        ab, ac = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
        area = ab[:, 0] * ac[:, 1] - ab[:, 1] * ac[:, 0]
        opposite = np.roll(tri, -1, axis=1) - np.roll(tri, -2, axis=1)
        gradient = opposite[:, :, ::-1] * [1.0, -1.0] / area[:, None, None]
        coefficients.append(np.einsum("nki,ni->nk", gradient, direction))
        crossed.append(faces[face])
    crossed = np.concatenate(crossed)
    index = np.repeat(np.arange(len(crossed)), 3)
    slopes = coo_matrix(
        (np.concatenate(coefficients).round(10).ravel(), (index, crossed.ravel())),
        shape=(len(crossed), len(xy)),
    ).tocsr()
    return crossed, slopes


def project_heights(bending, constraints, lower, upper):
    """高さの変更量と曲率を最小にしながら、各不等式の区間へ交互に射影する。"""
    penalty = 0.4
    solve = factorized((bending + penalty * constraints.T @ constraints).tocsc())
    bounded = np.clip(np.zeros(len(lower)), lower, upper)
    multiplier = np.zeros_like(bounded)
    for _ in range(2201):
        shift = solve(penalty * (constraints.T @ (bounded - multiplier)))
        if not np.isfinite(shift).all():
            raise ValueError("谷の排水計算で有限の高さが得られません")
        actual = constraints @ shift
        new_bounded = np.clip(actual + multiplier, lower, upper)
        multiplier += actual - new_bounded
        primal = np.max(abs(actual - new_bounded))
        step = np.max(abs(bounded - new_bounded))
        if primal < 8e-7 and step < 2e-6:
            return shift
        bounded = new_bounded
    violation = np.max(np.maximum(lower - actual, actual - upper))
    if violation < 1e-5 and primal < 1e-5 and step < 2e-6:
        return shift
    raise ValueError(f"谷の排水計算が収束しません: 制約超過={violation:g}, 残差={primal:g}, 更新幅={step:g}")


def drainage_edges(z, edges, outer, free):
    """各頂点から一つ下流の頂点を選ぶ。平坦な河原は外周へ抜ける経路を保つ。"""
    _, parent = drain_to_boundary(z, edges, outer, parents=True)
    directed = np.concatenate([edges, edges[:, ::-1]])
    downhill = directed[z[directed[:, 1]] < z[directed[:, 0]] - 1e-8]
    ranked = downhill[np.lexsort((z[downhill[:, 1]], downhill[:, 0]))]
    node, first = np.unique(ranked[:, 0], return_index=True)
    parent[node] = ranked[first, 1]
    affected = np.flatnonzero((free | free[parent]) & (parent != np.arange(len(z))))
    pair = np.c_[affected, parent[affected]]
    return coo_matrix(
        (np.tile([-1.0, 1.0], len(pair)), (np.repeat(np.arange(len(pair)), 2), pair.ravel())),
        shape=(len(pair), len(z)),
    ).tocsr()


def downhill_grades(vertices, faces, outer, rows, reference):
    """三角形の縦断勾配、5 m標高の範囲、外周へ向かう排水方向を同時に守る。"""
    xy, original = vertices[:, :2], vertices[:, 2]
    crossed, slopes = channel_derivatives(xy, faces, rows)
    grade = slopes @ original
    if grade.max(initial=0.0) <= 0.0:
        return original.copy()
    neighbours = cKDTree(xy[np.unique(crossed[grade > -0.01])]).query(xy)[0] < 4.2
    free = neighbours & ~outer
    if not free.any():
        return original.copy()
    edges = np.unique(np.sort(faces[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1), axis=0)
    a, b = edges.T
    adjacency = coo_matrix(
        (np.ones(2 * len(a)), (np.r_[a, b], np.r_[b, a])), shape=(len(xy), len(xy))
    ).tocsr()
    laplace = eye(len(xy), format="csr") - diags(1.0 / np.asarray(adjacency.sum(axis=1)).ravel()) @ adjacency
    laplace = laplace[:, free]
    identity = eye(free.sum(), format="csr")
    bending = 2.0 * laplace.T @ laplace + 0.015 * identity
    slopes = slopes[:, free]
    affected = slopes.getnnz(axis=1) > 0
    ref, missing = reference(*xy[free].T)
    delta = ref[missing] - original[free][missing]
    lower = np.r_[np.full(affected.sum(), -np.inf), delta - 0.99]
    upper = np.r_[-grade[affected] - 1e-5, delta + 0.99]
    constraints = vstack((slopes[affected], identity[missing]), format="csr")
    # 初回は低水路だけを定め、続いてそこで成立する排水経路を全体の拘束へ加える。
    preliminary = original.copy()
    preliminary[free] += project_heights(bending, constraints, lower, upper)
    drained = drain_to_boundary(preliminary, edges, outer)
    downstream = drainage_edges(drained, edges, outer, free)
    constraints = vstack((constraints, downstream[:, free]), format="csr")
    lower = np.r_[lower, np.full(downstream.shape[0], -np.inf)]
    upper = np.r_[upper, -(downstream @ original)]
    z = original.copy()
    z[free] += project_heights(bending, constraints, lower, upper)
    return drain_to_boundary(z, edges, outer)
