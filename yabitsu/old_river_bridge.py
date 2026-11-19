"""藤熊川支流の旧道橋。現橋と平行に杉林の岸へ渡る、写真の二列の錆びた欄干。"""
import numpy as np
import shapely
from shapely.geometry import Polygon, box


def rail_faces(length, lateral, floor):
    """縦断面を押し出す。上弦と傾いた支材を一体にして継ぎ目の重複面を避ける。"""
    limits = np.linspace(-length / 2 + .10, length / 2 - .32, 7)
    shapes = [box(-length / 2, .50, length / 2, .65)]
    for u in limits:
        shapes.append(Polygon([(u - .10, 0.), (u + .12, 0.), (u + .29, .54), (u + .11, .54)]))
    profile = shapely.orient_polygons(shapely.union_all(shapes))
    centers = [lateral - .075, lateral + .075]

    def xyz(uz, lat):
        u, height = uz.T
        return np.c_[u, np.full_like(u, lat), floor(u, lat) + height]

    tris = shapely.get_parts(shapely.constrained_delaunay_triangles(profile))
    for part in shapely.orient_polygons(tris):
        coords = shapely.get_coordinates(part)[:3]
        yield xyz(coords, centers[0])
        yield xyz(coords[::-1], centers[1])
    outline = np.asarray(profile.exterior.coords)
    for p, q in zip(outline[:-1], outline[1:]):
        edge = np.array([p, q])
        yield np.r_[xyz(edge, centers[1]), xyz(edge[::-1], centers[0])]


def meshes(sec, span, ground):
    i, j = span
    middle = (sec.s[i] + sec.s[j]) / 2
    heading = np.interp(middle, sec.s, np.unwrap(sec.heading))
    tangent = np.array([np.cos(heading), np.sin(heading)])
    normal = np.array([-tangent[1], tangent[0]])
    origin = np.array([np.interp(middle, sec.s, sec.x), np.interp(middle, sec.s, sec.y)])
    road_z = np.interp(middle, sec.s, sec.edge_z(1))
    grade = (sec.edge_z(1)[j] - sec.edge_z(1)[i]) / (sec.s[j] - sec.s[i])
    near, far = 5.75, 11.55
    half = 4.35

    def xy(u, lat):
        return origin + np.asarray(u)[..., None] * tangent + np.asarray(lat)[..., None] * normal

    def floor(u, lat):
        return road_z - .18 + grade * u + .012 * (lat - near)

    def terrain_z(u, lat):
        return ground(*xy(u, lat).T)

    def mesh(name, faces, material):
        local = []
        for face in faces:
            face = np.asarray(face, float)
            local.append(face[[0, 1, 2]])
            if len(face) == 4:
                local.append(face[[0, 2, 3]])
        tri = np.asarray(local)
        # 側面に水平面のUVを貼らない。局所座標なので桁と支材の向きにも揃う。
        directions = np.abs(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]))
        axes = np.array([[1, 2], [0, 2], [0, 1]])[directions.argmax(axis=1)]
        uv = np.take_along_axis(tri, axes[:, None], axis=2).reshape(-1, 2) / 1.8
        v = tri.reshape(-1, 3)
        return (f"smallspan_16_old_road_{name}", np.c_[xy(v[:, 0], v[:, 1]), v[:, 2]],
                np.arange(len(v)).reshape(-1, 3), uv, material, True)

    corners = np.array([[-half, near], [half, near], [half, far], [-half, far]])
    surface = np.c_[corners, floor(*corners.T)]
    underside = surface - [0., 0., .29]
    yield mesh("floor", [surface], "gravel_grey")
    edges = [[surface[b], surface[a], underside[a], underside[b]] for a, b in ((0, 1), (1, 2), (2, 3), (3, 0))]
    yield mesh("slab", [underside[::-1], *edges], "concrete_moss")
    for label, lat in (("inner_rail", near + .16), ("outer_rail", far - .16)):
        yield mesh(label, rail_faces(2 * half - .22, lat, floor), "valley_rusted_steel")

    # 盛土は両岸の取り付きだけに置く。中央には山側から現橋まで沢幅を残す。
    for side, label in ((-1, "start"), (1, "end")):
        u = side * half
        a, b = sorted((u - side * .22, u + side * .32))
        corners = np.array([[a, near + .12], [b, near + .12], [b, far - .12], [a, far - .12]])
        upper = np.c_[corners, floor(*corners.T) - .27]
        z0 = min(terrain_z(*corners.T).min() - .55, upper[:, 2].min() - .25)
        base = np.c_[corners, np.full(4, z0)]
        walls = [[upper[q], upper[p], base[p], base[q]] for p, q in ((0, 1), (1, 2), (2, 3), (3, 0))]
        yield mesh("abutment_" + label, walls, "concrete_moss")

        ends = u + side * np.array([0., 1.4, 3.2])
        lateral = np.array([4.65, near, (near + far) / 2, far, 13.15, 14.65])
        us, ls = np.meshgrid(ends, lateral, indexing="ij")
        native = terrain_z(us.ravel(), ls.ravel()).reshape(us.shape)
        contact = floor(us, ls) + np.interp(ls, [far, 13.15, 14.65], [0., .48, 0.])
        # 外側の草土は既存斜面へ潜らせて上下に重ならないようにする。
        blend = np.maximum(np.array([0., .35, 1.])[:, None], np.array([1., 0., 0., 0., 0., 1.]))
        zz = (1. - blend) * contact + blend * (native - .08)
        grid = np.stack((us, ls, zz), axis=-1)
        faces = []
        for r in range(2):
            for c in range(len(lateral) - 1):
                q = np.array([grid[r, c], grid[r+1, c], grid[r+1, c+1], grid[r, c+1]])
                faces.append(q if side > 0 else q[::-1])
        # 川に面する肩は、桁の側部から谷床へ返す。沢口の中央は空く。
        for indices in ((0, 1), (3, 4, 5)):
            edge = grid[0, list(indices)]
            toe = edge.copy()
            toe[:, 0] -= side * .95
            toe[:, 2] = terrain_z(toe[:, 0], toe[:, 1]) - .10
            for q in range(len(edge) - 1):
                face = np.array([toe[q], edge[q], edge[q+1], toe[q+1]])
                faces.append(face if side > 0 else face[::-1])
        yield mesh("bank_" + label, faces, "ground")
