"""納品NPZの契約と保存した制作元を検査する。Blenderでも通常Pythonでも実行可能。"""
from pathlib import Path
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/processed/landmarks/nanohanadai_tower'
CMP = ROOT / 'export/compare/nanohanadai'


def verify():
    parts = ['frame', 'stairs', 'deck', 'roof', 'base']
    assert all((OUT / (name + '.npz')).exists() for name in parts), '5部品のNPZが必要'
    records = []
    for name in parts:
        with np.load(OUT / (name + '.npz')) as data:
            assert set(data.files) == {'v', 'f', 'uv'}
            v, f, uv = data['v'], data['f'], data['uv']
        assert v.ndim == 2 and v.shape[1] == 3
        assert f.ndim == 2 and f.shape[1] == 3 and np.issubdtype(f.dtype, np.integer)
        assert uv.shape == (len(v), 2)
        assert np.isfinite(v).all() and np.isfinite(uv).all()
        assert f.min() >= 0 and f.max() < len(v) < 65536
        assert uv.min() >= 0 and uv.max() <= 1
        cross = np.cross(v[f[:, 1]]-v[f[:, 0]], v[f[:, 2]]-v[f[:, 0]])
        area = np.linalg.norm(cross, axis=1)/2
        assert area.min() > 1e-10, (name, area.min())
        vn = np.zeros_like(v)
        for j in range(3):
            np.add.at(vn, f[:, j], cross)
        assert np.linalg.norm(vn, axis=1).min() > 1e-10
        assert (OUT / (name + '.png')).stat().st_size > 1000
        records.append(dict(name=name, triangles=len(f), vertices=len(v),
                            bounds_m=[v.min(0).tolist(), v.max(0).tolist()],
                            dimensions_m=np.ptp(v, axis=0).tolist(),
                            min_triangle_area_m2=float(area.min()),
                            uv_range=[uv.min(0).tolist(), uv.max(0).tolist()]))
    total = sum(x['triangles'] for x in records)
    assert total <= 40000, total
    assert abs(records[3]['bounds_m'][1][2] - 13.0) < .02
    source = ROOT / 'assets/landmarks/nanohanadai_tower'
    meta = json.loads((source/'asset.json').read_text())
    assert meta['polygon_sides'] == 12 and meta['stair_paths'] == 2
    source_info = None
    try:
        import bpy
        import bmesh
    except ImportError:
        pass
    else:
        from mathutils import Vector
        from mathutils.bvhtree import BVHTree
        import math
        assert Path(bpy.data.filepath) == source/'nanohanadai_tower.blend'
        source_info = {}
        for name in parts:
            col = bpy.data.collections[name]
            obs = [o for o in col.objects if o.type == 'MESH']
            assert obs
            nonclosed = []
            source_vertices=[]
            source_triangles=0
            for ob in obs:
                bm = bmesh.new()
                bm.from_mesh(ob.data)
                if not all(e.is_manifold for e in bm.edges):
                    nonclosed.append(ob.name)
                bm.free()
                ob.data.calc_loop_triangles()
                source_triangles+=len(ob.data.loop_triangles)
                source_vertices.extend(tuple(ob.matrix_world@v.co) for v in ob.data.vertices)
            assert not nonclosed, nonclosed
            record=next(r for r in records if r['name']==name)
            points=np.array(source_vertices)
            assert source_triangles==record['triangles']
            assert np.allclose([points.min(0),points.max(0)],record['bounds_m'],atol=1e-6)
            source_info[name] = dict(objects=len(obs), all_components_closed=True,
                                     triangles_and_bounds_match_npz=True)
        names = [o.name for o in bpy.data.objects]
        assert sum(n.startswith('main_post_') for n in names) == 12
        assert sum(n.startswith('tread_A_') for n in names) == 48
        assert sum(n.startswith('tread_B_') for n in names) == 48
        assert sum(n.startswith('landing_A_') for n in names) == 6
        assert sum(n.startswith('landing_B_') for n in names) == 6
        assert sum(n.startswith('skylight_cap_') for n in names) == 6
        for im in bpy.data.images:
            if im.source == 'FILE':
                assert im.packed_file or Path(bpy.path.abspath(im.filepath)).exists()
        # 独立した納品メッシュを使い、踏面中央での頭上とデッキへの出口を検査する。
        all_v, all_f = [], []
        for name in parts:
            with np.load(OUT/(name+'.npz')) as d:
                all_f.extend((d['f']+len(all_v)).tolist())
                all_v.extend(d['v'].tolist())
        bvh = BVHTree.FromPolygons(all_v, all_f, all_triangles=True)
        roof_v,roof_f=None,None
        with np.load(OUT/'roof.npz') as d:
            roof_v,roof_f=d['v'],d['f']
        roof_bvh=BVHTree.FromPolygons(roof_v.tolist(),roof_f.tolist(),all_triangles=True)
        for i in range(24):
            a=math.tau*i/24
            for r in (.1,.2,.3,.4,.45,.48,.52,.57,1.0,2.0,3.0):
                hit,normal,idx,distance=roof_bvh.ray_cast(Vector((r*math.cos(a),r*math.sin(a),15)),Vector((0,0,-1)),4)
                assert hit is not None, ('採光部の頂部に穴',i,r)
        clearances=[]
        for ob in bpy.data.collections['stairs'].objects:
            if not ob.name.startswith(('tread_', 'landing_')):
                continue
            co=np.array([tuple(ob.matrix_world@v.co) for v in ob.data.vertices])
            center=co.mean(0)
            center[2]=co[:,2].max()+.015
            hit, normal, idx, distance=bvh.ray_cast(Vector(center),Vector((0,0,1)),20)
            if hit is not None:
                clearances.append(dict(name=ob.name,clearance_m=distance))
        assert min(c['clearance_m'] for c in clearances)>1.85, sorted(clearances,key=lambda c:c['clearance_m'])[:4]
        exits=[]
        for a in (math.radians(135),math.radians(315)):
            direction=Vector((math.cos(a),math.sin(a),0))
            start=direction*1.70+Vector((0,0,9.80))
            hit, normal, idx, distance=bvh.ray_cast(start,direction,1.35)
            exits.append(dict(angle_from_east_deg=math.degrees(a),clear=hit is None))
        assert all(e['clear'] for e in exits), ('デッキの出口が塞がっている',exits)
        source_info['walking_checks']=dict(tread_and_landing_samples=len(clearances),
            min_head_clearance_m=min(c['clearance_m'] for c in clearances),deck_exits=exits)
        source_info['roof_checks']=dict(vertical_ray_samples=264,all_hit=True)
    result = dict(passed=True, total_triangles=total, parts=records,
                  blend_reopened=source_info is not None, source=source_info)
    (CMP/'nanohanadai_tower_validation.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == '__main__':
    verify()
