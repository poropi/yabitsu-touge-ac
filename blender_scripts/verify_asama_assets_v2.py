"""依頼02の植物NPZ/PNGを、今回の寸法・面数制約で読み取り検査する。"""
from pathlib import Path
import hashlib
import json

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'export/compare/asama'


def verify():
    records = []
    for kind, asset, limit in [('vines', 'asama_wall_vines', 12000),
                               ('grass', 'asama_grass_tufts', 3000)]:
        source = ROOT / 'assets/landmarks' / asset
        atlas = source / f'{kind}_atlas_v2.png'
        atlas_rgba = np.array(Image.open(atlas).convert('RGBA'))
        for i, letter in enumerate('abc'):
            stem = f'{kind}_{letter}'
            folder = ROOT / 'data/processed/landmarks' / asset
            with np.load(folder / f'{stem}.npz') as data:
                assert set(data.files) == {'v', 'f', 'uv'}, stem
                v, f, uv = (data[k] for k in ['v', 'f', 'uv'])
            assert v.ndim == 2 and v.shape[1] == 3, stem
            assert uv.shape == (len(v), 2), stem
            assert f.ndim == 2 and f.shape[1] == 3 and len(f) <= limit, stem
            assert f.dtype.kind in 'iu' and f.min() >= 0 and f.max() < len(v), stem
            assert np.isfinite(v).all() and np.isfinite(uv).all(), stem
            assert uv.min() >= 0 and uv.max() <= 1, stem
            tri = v[f]
            normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
            assert (np.linalg.norm(normal, axis=1) > 1e-12).all(), stem
            vertex_normals = np.zeros_like(v)
            for j in range(3):
                np.add.at(vertex_normals, f[:, j], normal)
            assert (np.linalg.norm(vertex_normals, axis=1) > 1e-12).all(), stem
            # カード1枚は表2面＋裏2面。Blenderのquad三角形化は
            # 表裏で対角線が異なるため、四隅・UVと合計面法線で比較する。
            assert len(f) % 4 == 0, stem
            for start in range(0, len(f), 4):
                front_ids = set(f[start:start + 2].ravel())
                back_ids = set(f[start + 2:start + 4].ravel())
                assert not front_ids & back_ids, stem
                signature = lambda ids: {tuple(np.round(np.r_[v[j], uv[j]], 7)) for j in ids}
                assert signature(front_ids) == signature(back_ids), stem
                assert np.allclose(normal[start:start + 2].sum(0),
                                   -normal[start + 2:start + 4].sum(0), atol=1e-10), stem
            im = Image.open(folder / f'{stem}.png')
            rgba = np.array(im)
            assert im.mode == 'RGBA' and np.array_equal(rgba, atlas_rgba), stem
            assert rgba[:, :, 3].min() == 0 and rgba[:, :, 3].max() == 255, stem
            extents = np.ptp(v, axis=0)
            extra = {}
            if kind == 'vines':
                assert abs(extents[0] - 2.5) < 1e-6, stem
                assert 1.3 <= v[:, 2].min() <= 2.1, stem
                crown_height = v[:, 2].max() - 3.0
                assert 1.5 <= crown_height <= 2.5, stem
                front = (v[:, 2] <= 3.0) & (v[:, 1] < .33 * v[:, 2] - .25)
                projection = .33 * v[front, 2] - v[front, 1]
                assert projection.min() >= .3 - 1e-6 and projection.max() <= .6 + 1e-6, stem
                extra = {'crown_height_above_wall_m': float(crown_height),
                         'front_leaf_projection_m': [float(projection.min()), float(projection.max())]}
            else:
                assert abs(v[:, 2].min()) < 1e-6, stem
                assert .5 <= extents[2] <= 1.1, stem
                assert .6 <= max(extents[:2]) <= 1.0, stem
                assert abs(extents[2] - [.58, .81, 1.06][i]) < 1e-6, stem
                assert abs(max(extents[:2]) - [.66, .84, .99][i]) < 1e-6, stem
                opaque = rgba[:, :, 3] > 250
                extra = {'texture_opaque_mean_srgb': rgba[:, :, :3][opaque].mean(0).round(2).tolist()}
            records.append({'name': stem, 'triangles': len(f), 'triangle_limit': limit,
                            'vertices': len(v), 'bounds_m': [v.min(0).tolist(), v.max(0).tolist()],
                            'dimensions_m': extents.tolist(), 'uv': 'v=0画像上端',
                            'npz_sha256': hashlib.sha256((folder / f'{stem}.npz').read_bytes()).hexdigest(),
                            'png_sha256': hashlib.sha256((folder / f'{stem}.png').read_bytes()).hexdigest(),
                            'png_mode': im.mode, 'png_size': list(im.size),
                            'explicit_reverse_faces': True, 'atlas_matches_source': True,
                            'zero_vertex_normals': 0, 'degenerate_triangles': 0, 'ok': True, **extra})
    report = {'ok': True, 'assets': records}
    (OUT / 'plants2_delivery_check.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'ok': True, 'assets': [{k: r[k] for k in ['name', 'triangles', 'dimensions_m']} for r in records]}, ensure_ascii=False))
    return report


if __name__ == '__main__':
    verify()
