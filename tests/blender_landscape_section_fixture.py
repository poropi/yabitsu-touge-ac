"""S1境界の外側・跨ぎ・内側面と表示状態の保存を小規模に検証する。"""
import sys
from pathlib import Path
import bpy
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'blender_scripts'))
from landscape.scene import collection, fingerprint, split_visual


class Site:
    """X座標を距離とする境界試験用の道路を提供する。"""
    section_scope = True
    profiled = False
    start = 0
    end = 10
    sec = {'s': np.arange(-20, 21, dtype=float)}

    def __init__(self):
        self.boundary_preservation = []

    def nearest(self, points):
        return np.rint(np.asarray(points)[:, 0]).astype(int) + 20

    def inside(self, points):
        x = np.asarray(points)[:, 0]
        return (x >= self.start) & (x <= self.end)


def source(name, hidden=False, renderable=True, spans=None):
    """面ごとに異なるUVと素材を持つ三つの四角面を作る。"""
    verts, faces = [], []
    for a, b in (spans or [(-4, -2), (-1, 1), (2, 4)]):
        offset = len(verts)
        verts += [(a, 0, 0), (b, 0, 0), (b, 1, 0), (a, 1, 0)]
        faces.append(tuple(range(offset, offset + 4)))
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    mesh.uv_layers.new()
    for i, uv in enumerate(mesh.uv_layers.active.data):
        uv.uv = (i / 17, i / 23)
    for i in range(2):
        mesh.materials.append(bpy.data.materials.new(name + str(i)))
    for i, face in enumerate(mesh.polygons):
        face.material_index = i % 2
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj['ac_renderable'] = renderable
    obj.hide_render = hidden
    obj.hide_set(hidden)
    return obj


def face_signature(obj, index):
    """面の座標・UV・素材参照を独立して比較する。"""
    face = obj.data.polygons[index]
    return (tuple(tuple(obj.matrix_world @ obj.data.vertices[obj.data.loops[i].vertex_index].co)
                  for i in face.loop_indices),
            tuple(tuple(obj.data.uv_layers.active.data[i].uv) for i in face.loop_indices),
            obj.data.materials[face.material_index].name)


for hidden in (False, True):
    ob = source('fixture_' + str(hidden), hidden)
    before = fingerprint([ob])
    signatures = [face_signature(ob, i) for i in range(3)]
    site = Site()
    coll = collection('fixture_' + str(hidden))
    replacement, count = split_visual(ob, site, coll, bpy.data.materials.new('new_ground'))
    assert count == 1 and list(replacement['yls_source_face_indices']) == [2]
    remain = next(o for o in coll.objects if o.get('yls_role') == 'outside_preserved')
    assert list(remain['yls_source_face_indices']) == [0, 1]
    assert [face_signature(remain, i) for i in range(2)] == signatures[:2]
    assert fingerprint([ob]) == before
    assert ob.hide_render and ob.hide_get()
    assert all(o.hide_render == hidden and o.hide_get() == hidden and o['ac_renderable'] is True
               for o in coll.objects)
    assert len(site.boundary_preservation) == 1
    assert site.boundary_preservation[0]['faces'] == 1

ob = source('collision_only', renderable=False)
before = fingerprint([ob])
coll = collection('skip_fixture')
assert split_visual(ob, Site(), coll, bpy.data.materials.new('unused')) == (None, 0)
assert not coll.objects and not ob.hide_render and not ob.hide_get()
assert before == fingerprint([ob])

ob = source('multiple_wall_kinds', spans=[(-4, -2), (-1, 1), (2, 4), (6, 8)])
before = fingerprint([ob])
signatures = [face_signature(ob, i) for i in range(4)]
coll = collection('multiple_kinds_fixture')
pieces, count = split_visual(ob, Site(), coll,
                            {'rock': bpy.data.materials.new('YLS_rock'), 'rock_net': bpy.data.materials.new('YLS_rock_net')},
                            face_filter=lambda p: np.where(p[:, 0] < 5, 'rock', 'rock_net'))
assert count == 2 and len(pieces) == 2
assert {o['yls_surface_kind']: list(o['yls_source_face_indices']) for o in pieces} == {'rock': [2], 'rock_net': [3]}
remain = [o for o in coll.objects if o.get('yls_role') == 'outside_preserved']
assert len(remain) == 1 and list(remain[0]['yls_source_face_indices']) == [0, 1]
assert [face_signature(remain[0], i) for i in range(2)] == signatures[:2]
assert before == fingerprint([ob])
assert sorted(i for o in coll.objects for i in o['yls_source_face_indices']) == list(range(4))
assert all(not o.hide_render and not o.hide_get() for o in coll.objects)
assert ob.hide_render and ob.hide_get()
print('SECTION_FIXTURE_PASS outside/crossing UV+material+geometry; visibility; collider skip')
