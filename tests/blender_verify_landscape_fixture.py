"""保存後検証の面対応が、欠落・重複・世界座標の移動を識別することを確認する。"""
import sys
from pathlib import Path

import bpy

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'blender_scripts'))
from verify_landscape_full_route import polygon_positions


def main():
    """小さな合成面だけを作り、重い原本を読み込まず検証器を試す。"""
    mesh = bpy.data.meshes.new('verification_fixture')
    mesh.from_pydata([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)], [], [(0, 1, 2), (0, 2, 3)])
    ob = bpy.data.objects.new('verification_fixture', mesh)
    bpy.context.scene.collection.objects.link(ob)
    original = polygon_positions(ob)
    assert original == polygon_positions(ob, [0]) + polygon_positions(ob, [1])
    assert original != polygon_positions(ob, [0]), '欠落面を識別できません'
    assert original != polygon_positions(ob, [0, 0]), '面の重複を識別できません'
    ob.location.x = 10
    bpy.context.view_layer.update()
    assert original != polygon_positions(ob), '世界座標の移動を識別できません'
    print('FULL_ROUTE_VALIDATOR_FIXTURE_PASS')


if __name__ == '__main__':
    main()
