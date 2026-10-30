"""Blenderで保存済みアセットを読取り検査する。元のblendは変更しない。

blender -b scene.blend --python-exit-code 1 --python check_asset.py -- \
    --collection '01 ' --closed '石碑本体' --min-volume 0.5 --out /abs/check.json

--collection/--closed は名前の先頭で照合。開いた看板・草は --closed に指定しない。
"""
import argparse
import json
import math
import sys
from pathlib import Path

import bmesh
import bpy


def parse_args():
    argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--collection', help='対象コレクション名の先頭。省略時はアクティブシーン全体')
    p.add_argument('--require', action='append', default=[], help='必須部品の物体名の先頭（薄板を含む、複数指定可）')
    p.add_argument('--closed', action='append', default=[], help='閉じた立体として検査する物体名の先頭（複数指定可）')
    p.add_argument('--min-volume', type=float, default=1e-9, help='--closed各物体の体積下限（m³、予定寸法から指定）')
    p.add_argument('--min-height', type=float, help='--closed各物体のZ寸法下限（m）')
    p.add_argument('--out', type=Path, help='JSONレポートの保存先（絶対パス）')
    a = p.parse_args(argv)
    if a.min_volume < 0 or (a.min_height is not None and a.min_height < 0):
        p.error('寸法・体積の下限には0以上の値を指定してください')
    if a.out and not a.out.is_absolute():
        p.error('--out は絶対パスで指定してください')
    return a


def inspect_object(obj, depsgraph):
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    if mesh is None:
        return {'name': obj.name, 'vertices': 0, 'faces': 0, 'error': '評価後のメッシュがありません'}
    bm = bmesh.new()
    try:
        bm.from_mesh(mesh)
        # 非一様スケール・親・回転を含めたワールド座標で面積・体積を検査する。
        bm.transform(obj.matrix_world)
        bm.normal_update()
        finite = all(math.isfinite(c) for v in bm.verts for c in v.co)
        if finite and bm.verts:
            lo = [min(v.co[i] for v in bm.verts) for i in range(3)]
            hi = [max(v.co[i] for v in bm.verts) for i in range(3)]
            bounds = [lo, hi]
            dimensions = [hi[i] - lo[i] for i in range(3)]
        else:
            bounds = dimensions = None
        bad = sum(not e.is_manifold for e in bm.edges)
        closed = bool(bm.faces) and bad == 0
        area = sum(f.calc_area() for f in bm.faces) if finite else None
        volume = abs(bm.calc_volume()) if finite and closed else None
        return {
            'name': obj.name, 'type': obj.type,
            'vertices': len(bm.verts), 'faces': len(bm.faces),
            'triangles': sum(max(0, len(f.verts) - 2) for f in bm.faces),
            'finite': finite, 'bounds_world': bounds, 'dimensions_m': dimensions,
            'surface_area_m2': area, 'non_manifold_edges': bad,
            'closed': closed, 'volume_m3': volume,
        }
    finally:
        bm.free()
        evaluated.to_mesh_clear()


def main():
    a = parse_args()
    scene = bpy.context.scene
    if a.collection:
        collections = [c for c in scene.collection.children_recursive if c.name.startswith(a.collection)]
        if scene.collection.name.startswith(a.collection):
            collections.append(scene.collection)
        objects = {o.name: o for c in collections for o in c.all_objects}
    else:
        objects = {o.name: o for o in scene.objects}
    geometric = sorted((o for o in objects.values()
                        if o.type in {'MESH', 'FONT', 'CURVE', 'SURFACE', 'META'} and not o.hide_render),
                       key=lambda o: o.name)
    errors = []
    if not geometric:
        errors.append('対象の描画可能な形状がありません。シーン・コレクション・非表示設定を確認してください')
    for prefix in a.require + a.closed:
        if not any(o.name.startswith(prefix) for o in geometric):
            errors.append(f'必須部品の指定に一致する形状がありません: {prefix}')
    depsgraph = bpy.context.evaluated_depsgraph_get()
    reports = []
    for o in geometric:
        r = inspect_object(o, depsgraph)
        reports.append(r)
        if not r['vertices'] or not r['faces']:
            errors.append(f'{o.name}: 形状が空です')
            continue
        if not r['finite']:
            errors.append(f'{o.name}: 非有限の座標があります')
            continue
        if r['surface_area_m2'] <= 1e-12:
            errors.append(f'{o.name}: 面積がありません')
        if any(o.name.startswith(prefix) for prefix in a.closed):
            if not r['closed']:
                errors.append(f'{o.name}: 閉じた立体ではありません（非多様体辺 {r["non_manifold_edges"]}）')
            elif r['volume_m3'] <= a.min_volume:
                errors.append(f'{o.name}: 体積 {r["volume_m3"]:g}m³ が下限 {a.min_volume:g}m³ 以下です')
            if a.min_height is not None and r['dimensions_m'][2] < a.min_height:
                errors.append(f'{o.name}: 高さ {r["dimensions_m"][2]:g}m が下限 {a.min_height:g}m 未満です')
    report = {
        'ok': not errors, 'file': bpy.data.filepath, 'scene': scene.name,
        'unit_system': scene.unit_settings.system, 'scale_length': scene.unit_settings.scale_length,
        'collection_prefix': a.collection, 'closed_prefixes': a.closed,
        'required_matches': {prefix: [o.name for o in geometric if o.name.startswith(prefix)]
                             for prefix in dict.fromkeys(a.require + a.closed)},
        'errors': errors, 'objects': reports,
        'limits': '写真との一致・見た目・材質・ゲーム表示の合格を保証しない。寸法はBlender座標1単位=1mとして検査。',
    }
    if not math.isclose(scene.unit_settings.scale_length, 1.0):
        report['warning'] = 'scale_lengthが1ではありません。表示単位と頂点座標の単位を確認してください'
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
