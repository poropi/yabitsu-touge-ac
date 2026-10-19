"""全線の表示レールと既知侵入位置の葉面を小規模に生成し、安全補正を確認する。"""
import json
import sys
from datetime import datetime
from pathlib import Path

import bpy
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'blender_scripts'))
from landscape import safety, structures
from landscape.scene import Site, collection, fingerprint
from landscape_trial import check_road_envelope, file_hash


def mesh_object(name, vertices, faces, coll):
    """新規表示物の属性を持つ試験メッシュを作る。"""
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    ob = bpy.data.objects.new(name, mesh)
    ob['yls_generated'] = True
    ob['yls_role'] = 'new_visual'
    ob['ac_collision'] = False
    coll.objects.link(ob)
    return ob


def main():
    """原本の全メッシュ保持・レール補正・葉面除外・再実行の無変化を確認する。"""
    config = json.loads((ROOT / 'configs/landscape_full_route.json').read_text(encoding='utf-8-sig'))
    source = ROOT / config['source']
    source_hash = file_hash(source)
    bpy.ops.wm.open_mainfile(filepath=str(source))
    originals = list(bpy.context.scene.objects)
    before = fingerprint(originals)
    site = Site(ROOT, config, originals)
    coll = collection('Safety_Focused_Test')
    materials = {key: bpy.data.materials.new('YLS_test_' + key) for key in ('rail', 'bolt', 'reflector')}
    structures.guardrails(site, ROOT, coll, materials)
    rails = [o for o in coll.objects if o.get('yls_role') == 'new_visual']
    positions = {o.name: np.array([v.co[:] for v in o.data.vertices]) for o in rails}
    rail_result = safety.protect(site, rails + originals)
    rail_check = check_road_envelope(site, rails, 3.0)
    rail_max_shift = max(float(np.linalg.norm(np.array([v.co[:] for v in o.data.vertices]) - positions[o.name], axis=1).max()) for o in rails)
    shifted = []
    for ob in rails:
        current = np.array([v.co[:] for v in ob.data.vertices])
        delta = np.linalg.norm(current - positions[ob.name], axis=1)
        for i in np.flatnonzero(delta > 0.001):
            shifted.append({'object': ob.name, 'vertex': int(i), 'shift_m': float(delta[i]),
                            'before': positions[ob.name][i].tolist(), 'after': current[i].tolist()})
    stable = fingerprint(rails)
    second = safety.protect(site, rails)
    print('RAIL_SECOND_PROTECTION', json.dumps(second), flush=True)
    assert stable == fingerprint(rails), 'レールへの2回目補正が形状を変えています'
    assert second['rail_vertices_adjusted'] == 0 and second['vegetation_faces_removed'] == 0 and second.get('max_rail_displacement_m', 0) == 0
    # 実際に全線検査で侵入したヒノキの葉の位置。上方の別面はそのまま保持する。
    x, y, z = 326.0235900878906, 7694.85546875, 315.4104309082031
    vertices = [(x + dx, y + dy, z + dz) for dz in (0, 10) for dx, dy in ((-.02, -.02), (.02, -.02), (.02, .02), (-.02, .02))]
    leaf = mesh_object('YLS_vegetation_fixture', vertices, [(0, 1, 2, 3), (4, 5, 6, 7)], coll)
    safe_top = {tuple(v.co) for v in list(leaf.data.vertices)[4:]}
    leaf_result = safety.protect(site, [leaf])
    assert leaf_result['vegetation_faces_removed'] == 1 and len(leaf.data.polygons) == 1
    assert {tuple(v.co) for v in leaf.data.vertices} == safe_top, '安全な頭上葉まで変化しています'
    leaf_check = check_road_envelope(site, [leaf], 3.0)
    trial_leaf = mesh_object('YLS_trial_noop_fixture', vertices[:4], [(0, 1, 2, 3)], coll)
    trial_before = fingerprint([trial_leaf])
    site.full_route = False
    safety.protect(site, [trial_leaf])
    assert trial_before == fingerprint([trial_leaf]), '全域専用補正が100m試作へ作用しています'
    site.full_route = True
    assert before == fingerprint(originals), '原本のメッシュが補正で変化しています'
    assert source_hash == file_hash(source), '原本blendが変化しています'
    result = {'source_sha256': source_hash, 'original_meshes_unchanged': len(before), 'rail_objects': len(rails),
              'rail_adjustments': rail_result, 'rail_max_vertex_shift_m': rail_max_shift,
              'rail_largest_shifts': sorted(shifted, key=lambda row: row['shift_m'], reverse=True)[:8],
              'rail_envelope': rail_check, 'rail_repeat_no_change': True,
              'known_intruding_leaf': leaf_result, 'safe_overhead_leaf_preserved': True, 'leaf_envelope': leaf_check}
    result['trial_mode_unchanged'] = True
    out = ROOT / 'export/landscape_validation' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True)
    (out / 'safety_focused_validation.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print('SAFETY_FOCUSED_PASS', str(out), json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
