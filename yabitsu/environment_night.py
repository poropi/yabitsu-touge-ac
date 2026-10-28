"""既存形状に従う夜間演出。受入済みKN5へは書き込まない。"""
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from . import config, kn5, start_lot
from .manifest import Manifest
from .parking_props import _tube

GROUP = 'yabitsu_environment'


def ac_position(p):
    return [float(p[0]), float(p[2]), float(-p[1])]


def _normal(q):
    n = np.cross(q[1]-q[0], q[3]-q[0]); return n / np.linalg.norm(n)


def signal_lens(q, pedestrian):
    """元の灯器UVと同じ位置: 車用左の青、歩行者上の赤。"""
    q = np.asarray(q,float)
    u,v = (.5,.75) if pedestrian else (1/6,.5)
    ru,rv = (.33,.165) if pedestrian else (.11,.33)
    a=np.arange(16)*2*np.pi/16
    uv=np.c_[u+ru*np.cos(a),v+rv*np.sin(a)]
    verts=q[0]+uv[:,0,None]*(q[1]-q[0])+uv[:,1,None]*(q[3]-q[0])+_normal(q)*.008
    center=q[0]+u*(q[1]-q[0])+v*(q[3]-q[0])+_normal(q)*.008
    verts=np.r_[[center],verts]
    faces=np.array([[0,i+1,(i+1)%16+1] for i in range(16)])
    return verts,faces


def _components(v,f):
    edges=np.concatenate([np.c_[f[:,i],f[:,(i+1)%f.shape[1]]] for i in range(f.shape[1])])
    _,labels=connected_components(coo_matrix((np.ones(len(edges)),edges.T),shape=(len(v),len(v))),directed=False)
    return [v[labels==i] for i in np.unique(labels)]


def _quad_at_uv(q,uv,u0,u1,v0,v1):
    """元の四角壁とUVのアフィン写像。窓ガラス領域だけに面を作る。"""
    basis=np.c_[uv,np.ones(4)]
    transform=np.linalg.lstsq(basis,q,rcond=None)[0]
    wanted=np.array([[u0,v0,1],[u1,v0,1],[u1,v1,1],[u0,v1,1]])
    return wanted@transform+_normal(q)*.009


# CSPの列挙セクションはINIppの自動採番に委ねる。用途名はDESCRIPTIONに置く。
def _series(name, positions, color, radius, condition='NightLights'):
    if not positions: return ''
    rows=['[LIGHT_SERIES_...]', f'DESCRIPTION=ENV_{name}', 'ACTIVE=1',f'CONDITION={condition}',
          'DIRECTION=0,-1,0',f'COLOR={color}',f'RANGE={radius}', 'SPOT=115',
          'SPOT_SHARPNESS=0.35','SHADOWS=0','SPECULAR_MULT=0','FADE_AT=240','FADE_SMOOTH=80']
    rows.extend(f'POSITION_{i}='+','.join(f'{x:.4f}' for x in ac_position(p)) for i,p in enumerate(positions))
    return '\n'.join(rows)+'\n\n'


def _emission(name, material, color, condition='NightLights', transparent=False):
    s=f'[MATERIAL_ADJUSTMENT_...]\nDESCRIPTION=ENV_{name}\nMATERIALS={material}\nCONDITION={condition}\nKEY_0=ksEmissive\nVALUE_0={color}\nVALUE_0_OFF=0,0,0\n'
    if transparent: s+='KEY_1=alpha\nVALUE_1=1\nVALUE_1_OFF=0\n'
    return s+'\n'


def build(out_dir=None):
    root=Path(out_dir or config.DATA_PROCESSED/'environment')
    settings=json.loads((config.DATA_MANUAL/'environment_lights.json').read_text())
    source=config.TRACK_BUILD_DIR; original=json.loads((source/'manifest.json').read_text())
    m=Manifest(root); (root/'textures').mkdir(exist_ok=True)
    # CSP無しでは追加発光面を表示しない。CSP側で白テクスチャに替え、alphaで昼夜を制御。
    Image.new('RGBA',(4,4),(255,255,255,0)).save(root/'textures/environment_overlay.png')
    for name in ('env_city','env_window','env_signal_green','env_signal_red'):
        m.add_material(name,texture='textures/environment_overlay.png',alpha=True)
    m.add_material('env_lamp',color=(.82,.78,.63))
    m.add_material('env_metal',color=(.20,.22,.23))
    vending_spec=dict(original['materials']['vending_front'])
    vending_texture=source/vending_spec['texture']
    vending_spec['texture']='textures/env_vending_front'+vending_texture.suffix
    shutil.copy2(vending_texture,root/vending_spec['texture'])
    m.add_material('env_vending_front',**vending_spec)
    vending_bodies=[]
    for info in original['meshes'].values():
        if info['material'] in ('vending','vending_red'):   # 赤い自動販売機（峠の売店）も箱として数える
            d=np.load(source/info['file'])
            vending_bodies.extend(c.mean(0) for c in _components(d['verts'],d['faces']))
    report={'note':settings['note'],'counts':{},'cameras_ac':{},'street_positions':[],'sources':[]}
    def add(name,v,f,material):
        m.add_mesh(GROUP,name,v,f,np.zeros((len(v),2)),material)
    def quad(name,q,material):
        add(name,q,np.array([[0,1,2],[0,2,3],[2,1,0],[3,2,0]]),material)
    tunnel_positions=[]; street_positions=[]; vending_positions=[]; signal_count=0; windows=0
    # 全体の壁から最大36窓だけ選び、すべての家を発光させない。
    candidates=[]; kit_lit=[]
    for name,info in original['meshes'].items():
        mat=info['material']
        wanted=mat in ('signal_head','signal_ped','tunnel_lamp','vending_front') or ('signal_poles' in name) or (mat.startswith('facade_') and mat.endswith('_w') and name.startswith('1WALL_bld_')) \
            or mat in ('hk_win_silver','hk_win_bronze')   # 部品の家（house_kit）の窓ガラスの板。箱の家の窓の壁が無くなり、夜の窓が 0 になった（2026-10-28）
        if not wanted: continue
        d=np.load(source/info['file']); v,f,uv=d['verts'],d['faces'],d['uvs']
        if mat in ('signal_head','signal_ped'):
            for face in f:
                q=v[face]; vv,ff=signal_lens(q,mat=='signal_ped')
                add(f'env_signal_{signal_count:03d}',vv,ff,'env_signal_red' if mat=='signal_ped' else 'env_signal_green'); signal_count+=1
            q=v[f[0]]; center=q.mean(0)
            report['cameras_ac'].setdefault('signal_'+name,{'position':ac_position(center+_normal(q)*8+[0,0,-.4]),'target':ac_position(center)})
        elif mat=='tunnel_lamp':
            # 元が両面のため同じ4頂点を二重に数えない。
            for i in range(0,len(v),4): tunnel_positions.append(v[i:i+4].mean(0)-[0,0,.04])
            report['cameras_ac']['tunnel_'+name]={'position':ac_position(v[:4].mean(0)+[0,0,-2]),'target':ac_position(v[min(16,len(v)-4):min(20,len(v))].mean(0)+[0,0,-2])}
        elif mat=='vending_front':
            for face in f:
                q=v[face]; tx=uv[face]; center=q.mean(0); normal=_normal(q)
                body=min(vending_bodies,key=lambda b:np.linalg.norm(center-b))
                # 元の正面は箱の内側を向く。UVを保って面と照明を箱の外側へ向ける。
                if np.dot(normal,center-body)<0:
                    q=q[::-1]; tx=tx[::-1]; normal=-normal
                m.add_mesh(GROUP,f'env_vending_{len(vending_positions):02d}',q+normal*.008,
                           np.array([[0,1,2,3]]),tx,'env_vending_front')
                vending_positions.append(center+normal*.25)
                report['cameras_ac']['vending']={'position':ac_position(center+normal*5),'target':ac_position(center)}
        elif 'signal_poles' in name:
            chosen=[]
            for c in _components(v,f):
                if np.ptp(c[:,2])<3 or np.max(np.ptp(c[:,:2],axis=0))>.6: continue
                p=np.r_[c[:,:2].mean(0),c[:,2].max()-.18]
                if any(np.linalg.norm(p[:2]-q[:2])<28 for q in chosen): continue
                chosen.append(p)
                if len(chosen)==3: break
            street_positions.extend(chosen)
        elif mat in ('hk_win_silver','hk_win_bronze'):
            # 部品の家の窓：窓の板 1 枚＝四角 1 つ。kit_window_share の割合を光らせ、300 m 区画ごとに 1 つにまとめる（下で追加）。
            # 箱の家の 36 枚のくじ引きのままだと、1 万枚の窓のうち 36 枚で、町を撮っても 1 枚も光っていなかった（2026-10-28）
            krng=np.random.default_rng(__import__("zlib").crc32(name.encode()) ^ settings["seed"])
            for face in f:
                q=v[face]; tx=uv[face]; u0,v0=tx[:,0].min(),tx[:,1].min()
                col,row=int(round(u0*4)),(0 if v0>=.49 else 1)
                if row==1 and col in (1,2,3): continue        # 扉・シャッターを下ろした窓は光らせない
                if np.ptp(q[:,2])<.6 or krng.random()>=settings.get('kit_window_share',0.08): continue
                c=q.mean(0); kit_lit.append(c+(q-c)*.84+_normal(q)*.012)
        else:
            for face in f:
                if len(face)!=4: continue
                q=v[face]; tx=uv[face]
                if np.ptp(q[:,2])<1.8: continue
                u0=np.ceil(tx[:,0].min())+.31; v0=np.ceil(tx[:,1].min())+.39
                if u0+.38>tx[:,0].max() or v0+.37>tx[:,1].max(): continue
                candidates.append((name,q,tx,u0,v0)); break
    cells={}
    for q in kit_lit:
        cells.setdefault(tuple((q.mean(0)[:2]//300).astype(int)),[]).append(q)
    for (cx,cy),qs in sorted(cells.items()):
        vv=np.concatenate(qs); ff=np.concatenate([np.array([[0,1,2],[0,2,3],[2,1,0],[3,2,0]])+4*k for k in range(len(qs))])
        add(f'env_window_hk_c{cx}_{cy}'.replace('-','m'),vv,ff,'env_window'); windows+=len(qs)
        report['cameras_ac'].setdefault('residential_window',{'position':ac_position(qs[0].mean(0)+_normal(qs[0])*9),'target':ac_position(qs[0].mean(0))})
    # 元manifest順に偏らない固定seedの抽出。
    rng=np.random.default_rng(settings['seed'])
    for i in sorted(rng.choice(len(candidates),min(settings['window_limit'],len(candidates)),replace=False)):
        name,q,uv,u,v=candidates[i]
        for lo,hi in ((u,u+.16),(u+.21,u+.38)):
            quad(f'env_window_{windows:03d}',_quad_at_uv(q,uv,lo,hi,v,v+.37),'env_window'); windows+=1
        report['sources'].append(name)
        center=_quad_at_uv(q,uv,u,u+.38,v,v+.37).mean(0)
        report['cameras_ac'].setdefault('residential_window',{'position':ac_position(center+_normal(q)*9),'target':ac_position(center)})
    lot=start_lot.load_spec(config.DATA_MANUAL/'s1_start_lot.json'); frame=start_lot.frame(lot)
    pads=json.loads((config.DATA_PROCESSED/'track_pads.json').read_text()); z=next(p['z'] for p in pads if p['name']==start_lot.LOT_NAME)
    for i,item in enumerate(settings['parking_posts']):
        p=np.r_[frame.xy(item['a'],item['b']),z]; top=p+[0,0,settings['street_height']]
        _,v,f,_=_tube(f'env_parking_post_{i}',p,top,.055,'env_metal',10); add(f'env_parking_post_{i}',v,f,'env_metal')
        street_positions.append(top)
    for i,p in enumerate(street_positions):
        q=p+np.array([[-.24,-.14,0],[.24,-.14,0],[.24,.14,0],[-.24,.14,0]])
        quad(f'env_street_head_{i:02d}',q,'env_lamp')
    # 秦野市街方向の実地形格子から、遠景用の概略明かりを抽出する。
    hr=config.DATA_PROCESSED/'horizon'; hm=json.loads((hr/'manifest.json').read_text()); pts=[]
    xmin,xmax,ymin,ymax=settings['city_bounds_xy']
    for info in hm['meshes'].values():
        v=np.load(hr/info['file'])['verts']; keep=(v[:,0]>xmin)&(v[:,0]<xmax)&(v[:,1]>ymin)&(v[:,1]<ymax)&(v[:,2]>5)&(v[:,2]<260)
        pts.extend(v[keep])
    pts=np.unique(np.asarray(pts),axis=0)
    if len(pts)<settings['city_count']: raise ValueError('市街方向の地形候補点が不足')
    selected=pts[rng.choice(len(pts),settings['city_count'],replace=False)]
    vv=[]; ff=[]
    for p in selected:
        p=p+[0,0,4.0]; base=len(vv)
        # 幅30cmの小面。昼はalpha=0、遠方はCSPのdistant glowへ。
        vv.extend(p+np.array([[-.15,0,-.15],[.15,0,-.15],[.15,0,.15],[-.15,0,.15]])); ff.extend([[base,base+1,base+2],[base,base+2,base+3],[base+2,base+1,base],[base+3,base+2,base]])
    add('env_city_hadano',np.asarray(vv),np.asarray(ff),'env_city')
    report['counts']={'signals':signal_count,'windows':windows,'street_lights':len(street_positions),'tunnel_lights':len(tunnel_positions),'vending_lights':len(vending_positions),'city_dots':len(selected)}
    report['street_positions']=[ac_position(p) for p in street_positions]
    report['cameras_ac']['nanohanadai_city']={'position':ac_position(np.r_[frame.xy(-45,-1.8),z+1.7]),'target':ac_position(selected.mean(0)+[0,0,4])}
    report['cameras_ac']['nanohanadai_parking']={'position':ac_position(np.r_[frame.xy(-42,6),z+2]),'target':ac_position(street_positions[-2])}
    ini='; 自作の概略夜間演出。既存信号筐体と壁全体は発光させない。\n[INCLUDE: common/conditions.ini]\n\n[INCLUDE: common/materials_track.ini]\n\n'
    # common/conditions.ini は NightLights を定義していないため、この部品で明示する。
    ini+='[CONDITION_60]\nNAME=NightLights\nINPUT=SUN\nLUT=(|0=0|86=0|94=1|180=1|)\nLAG=0\n\n'
    ini+='[Material_DistantEmissive]\nMeshes=env_city_hadano\nUseAlpha=1\nBrightnessMult=0.35\nDistantGlowTextured=0\nDotMaxDistance='+str(settings['city_dot_max_distance'])+'\n\n'
    for label,materials in [('CITY','env_city'),('WINDOWS','env_window'),('SIGNALS','env_signal_green,env_signal_red')]:
        shader='' if label=='CITY' else 'SHADER=ksPerPixelAlpha\n'
        ini+=f'[SHADER_REPLACEMENT_...]\nDESCRIPTION=ENV_{label}\nMATERIALS={materials}\n'+shader+'BLEND_MODE=ALPHA_BLEND\nIS_TRANSPARENT=1\nDEPTH_MODE=NOWRITE\nCAST_SHADOWS=0\nCULL_MODE=OFF\nPROP_0=alpha,0\nPROP_1=ksAmbient,0\nPROP_2=ksDiffuse,0\nRESOURCE_0=txDiffuse\nRESOURCE_COLOR_0=1,1,1,1\n\n'
    ini+=_emission('CITY','env_city','1,0.68,0.34,8',transparent=True)
    ini+=_emission('WINDOW','env_window','1,0.73,0.43,0.8',transparent=True)
    ini+=_emission('GREEN','env_signal_green','0.12,1,0.65,2.5','ALWAYS_ON',True)
    ini+=_emission('RED','env_signal_red','1,0.07,0.03,2.5','ALWAYS_ON',True)
    ini+=_emission('STREET','env_lamp','1,0.86,0.63,3')
    ini+=_emission('TUNNEL','tunnel_lamp','1,0.84,0.55,3','ALWAYS_ON')
    ini+=_emission('VENDING','env_vending_front','1,0.94,0.83,1.2')
    ini+=_series('STREET',street_positions,'1,0.86,0.65,10',13)
    ini+=_series('TUNNEL',tunnel_positions,'1,0.84,0.60,7',10,'ALWAYS_ON')
    ini+=_series('VENDING',vending_positions,'1,0.94,0.83,1.5',3)
    m.save(); (root/'retex.json').write_text('{}'); (root/'hidden.json').write_text('[]')
    ini_path=root/'extension/parts/night.ini'; ini_path.parent.mkdir(parents=True,exist_ok=True); ini_path.write_text(ini,encoding='utf-8')
    (root/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return kn5.write_from_manifest(root/'manifest.json',GROUP,root/f'{GROUP}.kn5'),report
