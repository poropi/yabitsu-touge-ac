"""菜の花台の黒ずんだ低い輪止め。単体制作、写真画素は使用しない。"""
import sys, math, json, shutil
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import bpy
import nanohanadai_common as C

def build():
    C.init_scene()
    col=C.collection('01 輪止め本体')
    mat=C.proc('黒ずんだ風化コンクリート',(70,70,64),scale=15)
    nodes=mat.node_tree.nodes;links=mat.node_tree.links
    bs=next(n for n in nodes if n.type=='BSDF_PRINCIPLED')
    bs.inputs['Roughness'].default_value=.94
    geo=next(n for n in nodes if n.type=='NEW_GEOMETRY')
    fine=nodes.new('ShaderNodeTexNoise');fine.inputs['Scale'].default_value=290;fine.inputs['Detail'].default_value=2.0
    links.new(geo.outputs['Position'],fine.inputs['Vector'])
    grain=nodes.new('ShaderNodeValToRGB')
    grain.color_ramp.elements[0].position=.31;grain.color_ramp.elements[0].color=(.026,.025,.022,1)
    grain.color_ramp.elements[1].position=.72;grain.color_ramp.elements[1].color=(.092,.088,.076,1)
    links.new(fine.outputs['Fac'],grain.inputs[0])
    mix=nodes.new('ShaderNodeMixRGB');mix.blend_type='MIX';mix.inputs[0].default_value=.34
    old=bs.inputs['Base Color'].links[0].from_socket
    links.new(old,mix.inputs[1]);links.new(grain.outputs['Color'],mix.inputs[2]);links.new(mix.outputs[0],bs.inputs['Base Color'])
    # 14頂点の断面を8断面でつなぐ。両端は丸く下がり、上面には2本の浅溝。
    cross=[(-.12,0),(.12,0),(.112,.066),(.0816,.108),(.055,.12),(.044,.113),(.028,.113),(.017,.12),(-.017,.12),(-.028,.113),(-.044,.113),(-.055,.12),(-.0816,.108),(-.112,.066)]
    stations=[(-.825,.16,.38),(-.81,.57,.70),(-.775,.88,.91),(-.705,1,1),(.705,1,1),(.775,.88,.91),(.81,.57,.70),(.825,.16,.38)]
    verts=[(x,y*sy,z*sz) for x,sy,sz in stations for y,z in cross]
    n=len(cross);faces=[]
    for j in range(len(stations)-1):
        for i in range(n):faces.append((j*n+i,j*n+(i+1)%n,(j+1)*n+(i+1)%n,(j+1)*n+i))
    faces += [tuple(range(n-1,-1,-1)),tuple((len(stations)-1)*n+i for i in range(n))]
    ob=C.mesh('輪止め_丸い端と浅溝',verts,faces,mat)
    meta={'parts':{'wheel_stop':[ob]},'generator':'blender_scripts/build_nanohanadai_wheel_stop.py',
      'origin':'底面の中央 (0,0,0)。+X長辺、正面−Y、+Z上。',
      'dimension_basis':'sv_mid2_135: 中央4本の端点x309–431,488–628,702–861,965–1148px。7本の中心x48,205,370,558,782,1056,1382pxへ1次元射影モデルを当て、等間隔の枠幅2.5mで正規化。長さ/枠ピッチ0.623〜0.677、長さ1.56〜1.69m（平均1.64m）から1.65mを採用。前後幅0.24m・高0.12mは上面と側面の写真比率。透視/陰のため長さ±0.15m、幅±0.05m、高±0.03m程度の推定。',
      'material_interpretation':'写真の不規則な粒状面、灰色の摩耗縁、反射帯・成形リブが見えないことから黒ずんだコンクリートとして制作。写真だけではゴム製を完全には排除できない。',
      'approximations':['裏面・底面は写真では見えず平らなコンクリートとして推定。','浅溝2本は写真の上面の暗い筋を近似したもので正確な断面・本数は未確定。','個体ごとの欠けや落葉は複製配置を妨げるため共通の摩耗粒へ近似。両端を丸く低下させた14角断面で作成。'],
      'refs':['sv_mid2_135.png','sv_mid_135.png','sv_mid_090.png'],
      'reference_metadata':{'date':'2020-11','vertical_fov_deg':90,'pitch_down_deg':4.13,'mid2_panoid':'P7jvxAJ_IJlvRcEGVeXJSA','mid_panoid':'MC9FOu1heTQ47rJo-eqsKA'},
      'reference_urls':['https://www.google.com/maps/@35.4104211,139.2095236,3a,90y,135h,85.87t/data=!3m6!1e1!3m4!1sP7jvxAJ_IJlvRcEGVeXJSA!2e0!7i16384!8i8192','https://www.google.com/maps/@35.4103796,139.2093967,3a,90y,135h,85.87t/data=!3m6!1e1!3m4!1sMC9FOu1heTQ47rJo-eqsKA!2e0!7i16384!8i8192','https://www.google.com/maps/@35.4103796,139.2093967,3a,90y,90h,85.87t/data=!3m6!1e1!3m4!1sMC9FOu1heTQ47rJo-eqsKA!2e0!7i16384!8i8192']}
    info=C.save_asset('nanohanadai_wheel_stop',meta)
    src=C.ROOT/'assets/landmarks/nanohanadai_wheel_stop'
    shutil.copy2(C.ROOT/'data/processed/landmarks/nanohanadai_wheel_stop/wheel_stop.png',src/'wheel_stop_color.png')
    info['source_baked_color']='wheel_stop_color.png'
    (src/'asset.json').write_text(json.dumps(info,ensure_ascii=False,indent=2)+'\n')
    return info

if __name__=='__main__':build()
