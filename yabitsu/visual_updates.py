"""受入済み本体を保持し、壁UV・舗装模様・遠景の更新分だけを別KN5にする。"""
import json
import shutil
from pathlib import Path

import numpy as np

from . import config, kn5
from .manifest import Manifest

GROUP = 'yabitsu_visual_updates'


def build(source=None, out=None, wall_texture=None):
    source=Path(source or config.TRACK_BUILD_DIR)
    out=Path(out or config.DATA_PROCESSED/'visual_updates')
    wall_texture=Path(wall_texture or config.DATA_PROCESSED/'dress_mountain/extension/dress_mt_wall_kitahara.dds')
    original=json.loads((source/'manifest.json').read_text())
    m=Manifest(out); hidden=[]
    (out/'textures').mkdir(exist_ok=True)
    for name,entry in original['meshes'].items():
        if not (name.startswith(('TERRAIN_FAR_s1_','LINES_track_pave_')) or name=='1WALL_kitahara_obj_retaining'):
            continue
        material='visual_'+entry['material']
        spec=dict(original['materials'][entry['material']])
        texture=source/spec['texture'] if spec.get('texture') else None
        if name=='1WALL_kitahara_obj_retaining':
            material='visual_kitahara_wall';texture=wall_texture
        if material not in m.materials:
            if texture:
                rel='textures/'+material+texture.suffix
                shutil.copy2(texture,out/rel);spec['texture']=rel
            m.add_material(material,**spec)
        z=np.load(source/entry['file'])
        m.add_mesh(GROUP,'visual_'+name,z['verts'],z['faces'],z['uvs'],material)
        hidden.extend([name,name+'_back'])
    m.save()
    (out/'hidden.json').write_text(json.dumps(hidden))
    (out/'retex.json').write_text('{}')
    ext=out/'extension';ext.mkdir(exist_ok=True)
    ini='[MESH_ADJUSTMENT_...]\nACTIVE=1\nMESHES='+','.join(hidden)+'\nIS_RENDERABLE=0\n'
    if 'visual_ground_far' in m.materials:
        # 空中写真には撮影時の明暗が入っている。地形の反射で白く飛ばさない。
        ini+='\n[SHADER_REPLACEMENT_...]\nMATERIALS=visual_ground_far\nPROP_0=ksSpecular,0\nPROP_1=ksDiffuse,0.3\nPROP_2=ksAmbient,0.3\n'
    (ext/'visual_updates.ini').write_text(ini)
    return kn5.write_from_manifest(out/'manifest.json',GROUP,out/f'{GROUP}.kn5')
