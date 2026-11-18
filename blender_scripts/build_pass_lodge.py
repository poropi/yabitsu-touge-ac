"""白い長い棟の同定候補。同じトイレを駐車場側正面へ向けた代替モデル。"""
from pathlib import Path
import sys,math
ROOT=Path(__file__).resolve().parents[1]

def build():
    import bpy
    from mathutils import Matrix
    import pass_common as C
    import build_pass_toilet as T
    meta=T.build();parts={};pi={}
    mapping={'lodge_walls':['walls'],'lodge_roof':['metal_roof','wood_trusses','west_clerestory'],'lodge_end_details':['front_boards','built_in_bench','entrance_marks']}
    keep={id(o) for names in mapping.values() for name in names for o in meta['parts'][name]}
    for o in list(bpy.context.scene.objects):
        if id(o) not in keep:bpy.data.objects.remove(o,do_unlink=True)
    # 旧−X面（西長壁）を新−Y正面とする正回転。方位の焼込ではない。
    R=Matrix.Rotation(math.pi/2,4,'Z');bpy.context.view_layer.update()
    meaning={'lodge_walls':'2025年に見える白い縦波板の長壁と低い基礎。','lodge_roof':'灰紫色の金属切妻屋根、三角採光帯、前後トラス。','lodge_end_details':'同棟の南妻面の掲示、腰掛け、入口マーク。'}
    for new,names in mapping.items():
        col=C.collection(new);obs=[]
        for old in names:
            for o in meta['parts'][old]:
                o.matrix_world=R@o.matrix_world
                for c in list(o.users_collection):c.objects.unlink(o)
                col.objects.link(o);obs.append(o)
        parts[new]=obs;pi[new]={'meaning':meaning[new],'front':'-Y（駐車場へ向く西長壁）'}
    return {'parts':parts,'part_info':pi,'generator':'blender_scripts/build_pass_lodge.py','resolution':2048,'front_azimuth_deg':256,'roof_footprint_m':[12.4,9.2],
      'placement_exclusion_group':'pass_toilet_building','alternative_to':'pass_toilet','identity_status':'白い長い棟はpass_toiletの西長壁と同一棟である可能性が高い。指定IDとして同じ本体を別の正面軸で供給。両方を独立2棟として配置しない。',
      'refs':['codex07/sv_shop_045.png','codex07/sv_shop_090.png','codex07/sv_mid_000.png','codex07/sv_mid_045.png','codex07/gsat20.png','codex07/gsat21.png'],
      'dimension_basis':'work07/building_measurements.jsonの共有屋根候補。gsat20 NW(738,258),NE(811,242),SE(836,339),SW(763,355)、10m=81px、12.37×9.23m。独立した4棟目の輪郭は未確認。',
      'approximations':['pass_toiletと同じ形の代替データで、単独建物の存在を確定したものではない。同位置に両方を配置しない。','2014年の三ノ塔・塔ノ岳 Mt.Tounodakeの大文字は2025年の壁に見えないため、現在版へは入れていない。','背面と内部、三角採光面の奥行き/材質は推定。本文や貼紙は自作の線と色面。'],
      'reference_views':{'reference_shop090':[0,-2,.04],'reference_mid045':[1,-2,.1]},
      'assembly_notes':'NPZ各部品にassembly_offset_mを加えると代替モデルを復元。正面は駐車場側の長壁−Y。pass_toiletと同時配置しない。'}

if __name__=='__main__':
    sys.path.insert(0,str(Path(__file__).resolve().parent))
    import pass_common as C
    C.init_scene();C.save_asset('pass_lodge',build())
