"""依頼05: 草目地の敷石帯と、南西フェンスの折返し・コンクリート地覆。既存部品は上書きしない。"""
import sys,json,math
from pathlib import Path
import bpy,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'blender_scripts'))
import nanohanadai_common as C
ASSET='nanohanadai_sw_end_v1';ORIGIN=np.array([-66.0,0.0])
def local(p):return (np.asarray(p)-ORIGIN).tolist()
def build():
    C.init_scene()
    stone=C.proc('灰色の敷石',(118,115,105),scale=35);grass=C.proc('土と草の目地',(73,72,45),scale=18)
    concrete=C.proc('古いコンクリート地覆',(126,126,113),scale=14);rail=C.proc('淡い青緑の欄干',(109,157,136),kind='metal',scale=18)
    flag=C.collection('01 草目地の敷石帯');parts={'flagstones':[]}
    geometry=json.loads((ROOT/'export/compare/nanohanadai/work/sw_fix/flag_geometry.json').read_text())
    for name,v,f,material in geometry:
        parts['flagstones'].append(C.mesh(name,v,f,grass if material=='grass' else stone))
    C.collection('02 フェンス南西端と地覆');parts['fence_return']=[]
    def record():return list(C.CURRENT.objects)
    # 平面位置はSW端のみ。既存の長いフェンス(a>=-58)と連続させる。
    path=[(-58.0,-2.6,0.0),(-63.2,-2.6,.55),(-61.7,-1.7,.15)]
    def w(p):x,y=local(p[:2]);return np.array([x,y,p[2]])
    for k,(a,b) in enumerate(zip(path[:-1],path[1:])):
        A,B=w(a),w(b);d=B[:2]-A[:2];n=np.r_[-d[1],d[0],0.0];n/=np.linalg.norm(n)
        # 上端は斜めのコンクリート地覆。谷側へ裾を広げる保持壁。
        corners=[A+n*.18,B+n*.18,B-n*.18,A-n*.18];bottom=[p+np.array([0,0,-.65]) for p in corners]
        C.mesh(f'コンクリート地覆{k}',[tuple(p) for p in corners+bottom],[(0,1,2,3),(7,6,5,4),(0,4,5,1),(1,5,6,2),(2,6,7,3),(3,7,4,0)],concrete)
        for z in [.12,1.08]:C.pipe(f'欄干横材{k}_{z}',A+[0,0,z],B+[0,0,z],.044,rail,10)
        N=math.ceil(np.linalg.norm(B[:2]-A[:2])/.13)
        for i,t in enumerate(np.linspace(0,1,N+1)):
            p=A*(1-t)+B*t
            C.pipe(f'縦格子{k}_{i}',p+[0,0,.16],p+[0,0,1.04],.015,rail,6)
        for t in np.linspace(0,1,math.ceil(np.linalg.norm(B[:2]-A[:2])/1.8)+1):
            p=A*(1-t)+B*t;C.pipe('欄干の支柱',p+[0,0,-.04],p+[0,0,1.11],.044,rail,10)
    parts['fence_return']=record()
    C.collection('03 谷側の連続縁石');parts['edge_curb']=[]
    line=[(-62,-2.50),(-73,-2.50),(-79,-1.25),(-85,.75)]
    for a,b in zip(line[:-1],line[1:]):
        A=np.array(local(a));B=np.array(local(b));d=B-A;n=np.array([-d[1],d[0]])/np.linalg.norm(d)*.09
        q=[A+n,B+n,B-n,A-n];v=[(*p,zz) for zz in [-.04,.08] for p in q]
        parts['edge_curb'].append(C.mesh('連続した低い縁石',v,[(0,3,2,1),(4,5,6,7),(0,1,5,4),(1,2,6,5),(2,3,7,6),(3,0,4,7)],concrete))
    info=C.save_asset(ASSET,{'parts':parts,'generator':'blender_scripts/build_nanohanadai_sw_end_v1.py','origin':'共通原点は駐車場座標 a=-66,b=0 の路面。+X=+a、+Y=+b、正面-Y=谷側、+Z上。','dimension_basis':'sw135/180の足元を高さ2.5m・縦90度・下向き4.13度で逆投影。航空写真gsat20の奥の帯へa=-85で接続。形状と高さは写真からの推定で測量値ではない。','approximations':['敷石寸法60x45cm、目地4cm、地覆高さ0.15〜0.55mは推定。','駐車場全体は既存の平面のまま。フェンス終端の地覆の傾斜で近景の高低差を近似。','航空写真の折返し先端より約4m谷側へ寄る。航空写真とSVで完全な同時復元には至っていない。'],'refs':['sv_sw_135.png','sv_sw_180.png','sv_sw_000.png','gsat20.png','gsat21.png']})
    info['game_status']='NPZ/PNG出力済み、s1_start_lot.jsonへ配置済み。コース全体の組立、KN5、ゲーム導入は未実施。'
    (ROOT/'assets/landmarks'/ASSET/'asset.json').write_text(json.dumps(info,ensure_ascii=False,indent=2)+'\n')
    (ROOT/'export/compare/nanohanadai/nanohanadai_sw_fix_new_parts.json').write_text(json.dumps(info,ensure_ascii=False,indent=2))
if __name__=='__main__':build()
