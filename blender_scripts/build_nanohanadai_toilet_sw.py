"""依頼04: 既存南西端と同じ原点で接続する延長部。既存7部品は不変更。"""
from pathlib import Path
import bpy,sys,json,math
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'blender_scripts'))
import nanohanadai_common as C
CMP=C.CMP
SRC=ROOT/'assets/landmarks/nanohanadai_toilet'
OUT=ROOT/'data/processed/landmarks/nanohanadai_toilet'

def prism(name,poly,bottom,top,mat):
    n=len(poly)
    b=[bottom(x,y) if callable(bottom) else bottom for x,y in poly]
    t=[top(x,y) if callable(top) else top for x,y in poly]
    vs=[(x,y,z) for zs in [b,t] for (x,y),z in zip(poly,zs)]
    return C.mesh(name,vs,[tuple(range(n-1,-1,-1)),tuple(range(n,2*n))]+[(i,(i+1)%n,(i+1)%n+n,i+n) for i in range(n)],mat)

def build():
    # 既存制作元の材質そのものをリンクせず読み込む。既存blendの保存はしない。
    with bpy.data.libraries.load(str(SRC/'nanohanadai_toilet.blend'),link=False) as (data_from,data_to):
        data_to.materials=['風化した白いコンクリート','笠木の灰色','舗装の灰褐色']
    white,cap,floor=data_to.materials
    col=C.collection('08 南西アプローチ ramp_sw_approach')
    # XY境界を共有する帯状通路。-Xへ下り、山側(+Y)に幅1.58mを確保。
    # 写真の白壁/手すり比から南西端床Z.65mを推定。別名v2の端部に接続。
    stations=[(-8.900,-5.040,.650,.8685),(-9.650,-5.400,.580,.7985),
              (-9.650,-5.400,.580,.640),(-11.400,-6.080,.380,.430),
              (-13.200,-6.360,.180,.225),(-15.400,-6.500,.000,.034)]
    width=1.58
    for i,(a,b) in enumerate(zip(stations[:-1],stations[1:])):
        xa,ya,za,ha=a;xb,yb,zb,hb=b
        if abs(xa-xb)<1e-6:continue
        def interp(v0,v1):return lambda x,y:v0+(v1-v0)*(x-xa)/(xb-xa)
        # xの断面位置で山側まで広がるので、旧床のXY断面に一致する。
        poly=[(xa,ya+.08),(xb,yb+.08),(xb,yb+.08+width),(xa,ya+.08+width)]
        prism('灰色の連続した下り通路_%02d'%i,poly,-.10,interp(za,zb),floor)
        # 白壁は前面厚0.25m、笠木厚0.045m。上端の1段落ち以後は連続勾配。
        poly=[(xa,ya-.125),(xb,yb-.125),(xb,yb+.125),(xa,ya+.125)]
        prism('南西の白い擁壁_%02d'%i,poly,0,interp(max(.008,ha-.045),max(.008,hb-.025)),white)
        poly=[(xa,ya-.145),(xb,yb-.145),(xb,yb+.145),(xa,ya+.145)]
        prism('灰色の笠木_%02d'%i,poly,interp(max(.008,ha-.045),max(.008,hb-.025)),interp(ha,hb),cap)
        # 灰色通路の切り目（石板状の区画）は形を変えず色材で焼く。
        length=math.hypot(xb-xa,yb-ya)
        count=max(1,round(length/.60))
        for j in range(1,count):
            t=j/count;x=xa+(xb-xa)*t;y=ya+(yb-ya)*t;z=za+(zb-za)*t
            C.box('通路の細い横目地',(x,y+.08+width/2,z+.0005),(.008,width,.001),cap)
        # 白壁の規則的すぎない縦目地。薄面を使わず細い刻線を灰色でベイク。
        if max(ha,hb)>.10:
            for j in range(1,max(2,round(length/.80))):
                t=j/max(2,round(length/.80));x=xa+(xb-xa)*t;y=ya+(yb-ya)*t;h=ha+(hb-ha)*t
                C.box('白壁の縦目地',(x,y-.1255,(h-.04)/2),(.004,.001,max(.008,h-.04)),cap)
    # 旧山側縁石の断面（Y -3.41〜-3.23）を新通路の山側へ下げてつなぐ。
    a=stations[0];b=stations[1]
    poly=[(-8.90,-3.41),(-9.65,-3.77),(-9.65,-3.59),(-8.90,-3.23)]
    prism('山側縁石から通路への取り合い',poly,lambda x,y:.641+(x+8.9)/(-.75)*(-.071),lambda x,y:.821+(x+8.9)/(-.75)*(-.241),cap)
    # 手すりの無い南西延長。既存の曲がった手すり端には新しいパイプを追加しない。
    return col,stations

def main():
    C.init_scene()
    # 元の7部品を保存・上書きせず、06/07だけ複製して南西端高を修正。
    with bpy.data.libraries.load(str(SRC/'nanohanadai_toilet.blend'),link=False) as (df,dt):
        dt.collections=['06 正面のスロープと白い段状擁壁','07 丸パイプのスロープ手すり']
    parts={}
    def delta(x):return .57*max(0,min(1,(-x-1.5)/7.4))
    for oldcol,name in zip(dt.collections,['ramp_walls_sw_v2','ramp_rails_sw_v2']):
        bpy.context.scene.collection.children.link(oldcol)
        col=C.collection(name)
        bpy.context.view_layer.update();dg=bpy.context.evaluated_depsgraph_get()
        for old in list(oldcol.objects):
            me=bpy.data.meshes.new_from_object(old.evaluated_get(dg),preserve_all_data_layers=True,depsgraph=dg)
            me.transform(old.matrix_world)
            oldh=max(v.co.z for v in me.vertices)
            for v in me.vertices:
                d=delta(v.co.x)
                if name=='ramp_walls_sw_v2' and old.name.startswith('段状白壁'):
                    v.co.z+=d*(v.co.z/oldh if oldh else 0)
                elif name=='ramp_walls_sw_v2' and old.name.startswith('白壁の水平目地'):
                    v.co.z+=d*.5
                else:v.co.z+=d
            ob=bpy.data.objects.new(old.name+'_sw_v2',me);col.objects.link(ob)
        parts[name]=list(col.objects)
        bpy.data.collections.remove(oldcol)
    col,stations=build();parts['ramp_sw_approach']=list(col.objects)
    bpy.context.preferences.filepaths.save_version=0
    bpy.context.preferences.filepaths.file_preview_type = "NONE"
    bpy.ops.wm.save_as_mainfile(filepath=str(SRC/'nanohanadai_toilet_sw_approach.blend'))
    stats=[C.bake_part(obs,OUT,name,2048) for name,obs in parts.items()]
    assert stats[-1]['triangles']<=3000,stats[-1]
    import shutil
    for p in stats:shutil.copyfile(OUT/(p['name']+'.png'),SRC/(p['name']+'_color.png'))
    meta={'asset_id':'nanohanadai_toilet','part':'ramp_sw_approach','source':'nanohanadai_toilet_sw_approach.blend','generator':'blender_scripts/build_nanohanadai_toilet_sw.py','parts':stats,
          'coordinates':{'unit':'m','origin':'既存トイレの共通原点。建物中心直下。','up':'+Z','long_axis':'+X=北東(59°)','front':'-Y=駐車場側、149°'},
          'npz':{'keys':['v','f','uv'],'uv':'v=0画像上端。Blenderの再読込では1-v。'},
          'installation':'既存ramp_wallsをramp_walls_sw_v2、ramp_railsをramp_rails_sw_v2へ置換し、ramp_sw_approachを同じ変換で追加。旧2部品との重複配置不可。ほか5部品不変更。',
          'join':{'x':-8.9,'floor_y':[-4.96,-3.38],'floor_z':.65,'wall_y':[-5.165,-4.915],'wall_cap_top_z':.8685,'overlap_m':.005},
          'front_stations_x_y_floor_z_cap_z':stations,
          'dimension_basis':'sw315の手すり支柱(既存高0.93m)に対する白壁高の比およそ0.7〜0.9から、南西端床を0.65m、笠木頂部0.8685mと推定。写真比率による誤差約±0.2m。旧床端0.08mへ継ぐと壁が低すぎるため、南西側のみ+0.57m、中央床0.50mへ線形に戻した別名v2を制作。延長のXY・勾配は写真と仮カメラから推定。',
          'steps':'元の南西側10段の段状白壁をv2で保持。追加部分は接続後0.75mで1段(0.1585m)下がり、以後3勾配区間で連続的に歩道へ下る。歩行面は階段ではない。',
          'approximations':['延長端X=-15.4,Y=-6.5と通路幅1.58mは写真からの推定。樹木に隠れる山側境界と折れ曲がる接続部も推定。',
             '壁高・南西端の床高は写真の手すりとの寸法比からの推定。与えられた仮カメラと現モデル建物全体を同時に厳密一致させる測量値は無い。写真と同じ方位の比較だが画素一致を主張しない。',
             'v2は旧7部品のうちramp_walls/ramp_railsだけの別名改訂。北東側・中央接続位置・材質は維持。南西側の通路は端0.65mから中央0.50mへ緩く下る形（配置路面の絶対勾配は未測）。',
             '追加の白壁は1段と連続した下りを近似。個々の目地、欠け・苔、不陸は簡略化。',
             '裏面と基礎は推定。灰色通路の下地はZ=-0.10mまで埋めた閉じた形。追加パイプは写真で見えないため付けない。'],
          'reference_capture':{'date':'2020-11','panoid':'hCGwhUE1XaPosSAeVCOneg','latlon':[35.4102783,139.2092117],'vertical_fov_deg':90,'pitch_down_deg':4.13,'camera_local_m':[-9.4,-9.8,2.5],'camera_status':'依頼04の仮位置。測量による位置合わせではない。','urls':['https://www.google.com/maps/@35.4102783,139.2092117,3a,90y,0h,85.87t/data=!3m6!1e1!3m4!1shCGwhUE1XaPosSAeVCOneg!2e0!7i16384!8i8192','https://www.google.com/maps/@35.4102783,139.2092117,3a,90y,315h,85.87t/data=!3m6!1e1!3m4!1shCGwhUE1XaPosSAeVCOneg!2e0!7i16384!8i8192','https://www.google.com/maps/@35.4101451,139.2091123,3a,90y,0h,90t/data=!3m6!1e1!3m4!1sCIHM0ogKEICAgICEzoqyEg!2e10!7i8704!8i4352']},
          'refs':['sv_sw_000.png','sv_sw_315.png','sv_sw_270.png','sv_user_000.png','sv_user_090.png','gsat21.png'],
          'source_image_usage':'比較専用。写真画素の材質転用なし。元制作元と同じ3手続き材質を色のみベイク。',
          'game_status':'単体NPZ/PNG。配置・コース組立・KN5・ゲーム導入なし。既存7部品の実ファイル不変更。'}
    (SRC/'ramp_sw_approach.asset.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n')
    (CMP/'nanohanadai_toilet_sw_inventory.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n')
    print('SW_RESULT',json.dumps(stats),flush=True)
    return {'parts':stats}
if __name__=='__main__':main()
