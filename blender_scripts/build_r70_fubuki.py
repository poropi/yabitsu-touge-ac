"""吹風入口の単体モデル。通常の Python (NumPy、Pillow) で実行する。"""
import json
import math
from itertools import groupby
from pathlib import Path

import numpy as np
from PIL import Image

from route70_common import Mesh, atlas

ROOT = Path(__file__).resolve().parents[1]
ASSET = ROOT / 'assets/landmarks/r70_fubuki'
REPORT = ROOT / 'export/compare/route70'
WORK = REPORT / 'work/fubuki10'
SOURCE = ROOT / 'export/compare/route70_inventory_20261030'
ROAD = 0.7
LEFT, RIGHT = -4.4, 7.3
ARCH_MID, ARCH_HALF = -3.5, 31.5
COLOURS = {'dark': (55, 53, 49), 'black': (40, 40, 37), 'metal': (124, 128, 124),
           'white': (178, 181, 177), 'red': (164, 62, 68), 'rust': (137, 49, 51),
           'concrete': (135, 132, 122)}


def wire_post(m):
    m.tube('鋼管支柱', (0, 0, 0), (0, 0, 1.487), .027, 'dark', 12)
    m.tube('柱頭キャップ', (0, 0, 1.475), (0, 0, 1.50), .030, 'black', 12, r2=.025)
    for z in (.10, .76, 1.40):
        m.tube('網の固定帯', (0, 0, z), (0, 0, z+.018), .030, 'black', 10)


def wire(end):
    m = Mesh()
    wire_post(m)
    if end:
        return m
    # 正方形を45度傾けた網。線を境界で切り、透明板に依存しない。
    low, high, span, pitch = .086, 1.402, 2., .086
    for slope in (-1, 1):
        y = -.020 + slope*.0018
        for k in range(-26, 44):
            intercept = low + (k+.37)*pitch
            limits = sorted(((low-intercept)/slope, (high-intercept)/slope))
            start, stop = max(.0, limits[0]), min(span, limits[1])
            if stop - start < .001:
                continue
            m.tube('菱形の網線', (start, y, intercept+slope*start),
                   (stop, y, intercept+slope*stop), .0023, 'metal', 4, caps=False)
    for z in (low, high):
        m.tube('上下の張線', (0, -.020, z), (2, -.020, z), .005, 'metal', 8)
    return m


def bar(end):
    m = Mesh()
    m.box('角形支柱', (0, 0, .744), (.052, .052, 1.488), 'black')
    m.box('角形柱頭', (0, 0, 1.487), (.056, .056, .026), 'dark')
    if not end:
        for z in (.105, 1.43):
            m.box('上下の角形横桟', (1, 0, z), (2, .043, .043), 'dark')
        for i in range(1, 15):
            m.box('縦格子', (2*i/15, 0, .768), (.022, .025, 1.325), 'black')
    return m


def sections(m, label, rings, colour):
    """対応する断面の頂点を接続する。接合部に内部の蓋を作らない。"""
    n = len(rings[0])
    vs = [v for ring in rings for v in ring]
    faces = [tuple(reversed(range(n)))]
    for j in range(len(rings)-1):
        a = j*n
        faces.extend((a+i, a+(i+1)%n, a+n+(i+1)%n, a+n+i) for i in range(n))
    faces.append(tuple(range(len(vs)-n, len(vs))))
    m.surface(label, vs, faces, colour)


def long_box(m, label, x, z, width, height, ys, colour):
    rings = [[(x+dx*width/2, y, z+dz*height/2)
              for dx, dz in [(-1, -1), (-1, 1), (1, 1), (1, -1)]] for y in ys]
    sections(m, label, rings, colour)


def arch_z(y):
    return ROAD + 9.5*(1-((y-ARCH_MID)/ARCH_HALF)**2)


def rib_ring(x, y, width=.76, depth=.87):
    slope = -19*(y-ARCH_MID)/ARCH_HALF**2
    h = depth/2 / math.sqrt(1+slope**2)
    return [(x+dx*width/2, y-dz*slope*h, arch_z(y)+dz*h)
            for dx, dz in [(-1, -1), (-1, 1), (1, 1), (1, -1)]]


def shoulder_ring(x, y, z, w, d):
    return [(x+a*w/2, y+b*d/2, z) for a, b in [(-1,-1),(1,-1),(1,1),(-1,1)]]


def bridge():
    m = Mesh()
    frames = (-13.2, -5.6, 2., 9.6)
    uprights = (-26, -19.5, *frames, 16.8, 23.)
    stations = np.linspace(-49.5, 49.5, 51)
    girder_stations = stations[::2]
    for side, rail in [(-1, LEFT), (1, RIGHT)]:
        x = rail + side*.83
        long_box(m, '灰色の床側桁', rail, ROAD-.32, .32, .64, girder_stations, 'concrete')
        long_box(m, '欄干の地覆', rail, ROAD+.115, .31, .23, girder_stations, 'concrete')
        for h, w, d in [(1.21,.09,.094), (.68,.062,.067), (.36,.052,.055)]:
            long_box(m, '欄干の横桟', rail-side*.019, ROAD+h, w, d, girder_stations, 'white')
        for y in np.linspace(-49.45, 49.45, 51):
            m.box('欄干の親支柱', (rail, y, ROAD+.72), (.10,.09,.98), 'white')
        for a, b in zip(stations[:-1], stations[1:]):
            for y in np.linspace(a,b,11)[1:-1]:
                # 上・中桟の間で下端が止まる。支柱の間の下段は開口。
                # 上下の小口は横桟の内部に収まる。
                picket_x = rail+side*.016
                vs = [(picket_x+dx*.016,y+dy*.015,ROAD+h) for h in (.68,1.20)
                      for dx,dy in [(-1,-1),(1,-1),(1,1),(-1,1)]]
                m.surface('欄干の上段縦格子',vs,[(i,(i+1)%4,(i+1)%4+4,i+4) for i in range(4)],'metal')
        rings = [rib_ring(x,y) for y in np.linspace(ARCH_MID-35,ARCH_MID+35,49)]
        sections(m,'床下へ降りるアーチリブ',rings,'red')
        for y in np.linspace(ARCH_MID-31.5,ARCH_MID+31.5,11):
            sections(m, 'リブの継目板', [rib_ring(x,y+d,.785,.895) for d in (-.13,.13)], 'red')
        # 写真で箱桁の上に見える小さな突起。取付座と暗い側面を持つ。
        for y in np.linspace(ARCH_MID-28,ARCH_MID+28,15):
            slope = -19*(y-ARCH_MID)/ARCH_HALF**2
            top = arch_z(y)+.435*math.sqrt(1+slope**2)
            for across in (-.29,.29):
                m.box('上フランジの取付座',(x+across,y,top+.016),(.098,.155,.045),'rust')
        for y in uprights:
            top = arch_z(y)-.28
            sections(m,'縦材と上端ハンチ', [
                shoulder_ring(x,y,ROAD-.46,.39,.49), shoulder_ring(x,y,top-.88,.39,.49),
                shoulder_ring(x,y,top-.31,.54,.67), shoulder_ring(x,y,top,.61,.71)],'red')
        for y in (ARCH_MID-ARCH_HALF, ARCH_MID+ARCH_HALF):
            long_box(m,'脚位置の赤い短帯',rail+side*.18,ROAD-.24,.13,.48,[y-2.55,y,y+2.55],'red')
            for d in (-2.1, 0., 2.1):
                m.box('脚部の横つなぎ',(rail+side*.46,y+d,ROAD-.34),(.80,.17,.38),'rust')
    for y in frames:
        z = arch_z(y)
        # 横桁の上縁と下縁は中央をわずかに持ち上げる。
        rings = []
        for x, rise in [(LEFT-.83,0),(LEFT+.55,.03),(1.45,.13),(RIGHT-.55,.03),(RIGHT+.83,0)]:
            rings.append([(x,y+dy*.34,z+rise+dz*.35) for dy,dz in [(-1,-1),(1,-1),(1,1),(-1,1)]])
        sections(m,'上部の横桁',rings,'red')
        for x in (LEFT-.05,RIGHT+.05):
            m.box('横桁の添接板',(x,y,z+.022),(.40,.705,.71),'rust')
    for y in (-48.,-38.,*uprights,35.,45.5):
        width = RIGHT-LEFT+(1.66 if y in uprights else 0.)
        m.box('床下横桁', ((LEFT+RIGHT)/2,y,ROAD-.36), (width,.20,.36), 'metal')
    return m


def wall():
    m = Mesh()
    # 押出断面、道路側の正面は -Y。上端の2辺を面取り。
    profile = [(-.11,0),(.11,0),(.103,.844),(.088,.86),(-.088,.86),(-.103,.844)]
    verts = [(x,y,z) for x in (0.,2.) for y,z in profile]
    n = len(profile)
    m.surface('小口',verts,[tuple(range(n-1,-1,-1)),tuple(range(n,2*n))],'cream')
    for i in range(n):
        j = (i+1)%n
        quad = [verts[i],verts[j],verts[n+j],verts[n+i]]
        if i in (1,5):
            uv = [(x/2,1-z/.86) for x,_,z in quad]
            m.surface('白壁の側面',quad,[(0,1,2,3)],uv=uv)
        else:
            m.surface('天端と面取り' if i else '底面',quad,[(0,1,2,3)],'cream')
    return m


def wall_art():
    rng = np.random.default_rng(1910)
    size = 760
    y,x = np.mgrid[0:size,0:size]/size
    grain = rng.normal(0, 3.7, (size,size))
    # 低周波の色むら・雨筋・草際の染み。端には型枠の目地。
    for scale, amplitude in [(12,4.2),(36,2.),(140,1.5)]:
        noise = Image.fromarray(rng.integers(0,255,(scale,scale),dtype=np.uint8)).resize((size,size),Image.Resampling.BICUBIC)
        grain += (np.asarray(noise).astype(float)-127)*amplitude/70
    top = np.zeros_like(x)
    for pos, length, width in zip(rng.random(27),rng.uniform(.05,.48,27),rng.uniform(.001,.012,27)):
        top += np.exp(-((x-pos)/width)**2-y/length)
    seam = np.exp(-(np.minimum(x,1-x)/.004)**2)
    dirt = (np.maximum(0,y-.77)/.23)**2
    rgb = np.array([204,201,191])+grain[:,:,None]-top[:,:,None]*[13,12,10]-seam[:,:,None]*[41,40,38]-dirt[:,:,None]*[18,15,20]
    return Image.fromarray(np.uint8(np.clip(rgb,0,255)))


def write_json(path, obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')


def references():
    refs = {}
    for line in (SOURCE/'captures.jsonl').read_text().splitlines():
        r = json.loads(line)
        for direction in ('f','b'):
            key = r['id'][4:]+'_'+direction
            target = SOURCE/'images/sv'/('R70-'+key+'.jpg')
            if not target.is_file():
                target = WORK/target.name
            if key not in USED or not target.is_file():
                continue
            tilt = 92 if key in ('0976_f','0970_f') else 88
            h = r['heading_'+direction]
            url = f"https://www.google.com/maps/@{r['lat']},{r['lon']},3a,90y,{h}h,{tilt}t/data=!3m6!1e1!3m4!1s{r['pano']}!2e0!7i16384!8i8192"
            refs[key] = dict(image=str(target.relative_to(ROOT)),panoid=r['pano'],date=r['date'],
                             station_m=r['section_s_m'],heading=h,fov_deg=90,tilt_url_deg=tilt,url=url)
    assert refs.keys()==USED
    return refs


FENCE_REFS = [f'{n:04}_f' for n in range(968,975)]
BAR_REFS = ['0974_f','0975_f']
BRIDGE_REFS = ['0975_f','0976_f','0978_f','0978_b']
WALL_REFS = ['0978_f','0980_f']
USED = set(FENCE_REFS + BAR_REFS + BRIDGE_REFS + WALL_REFS)


def main():
    ASSET.mkdir(exist_ok=True,parents=True)
    refs = references()
    assets = [('slope_fence_wire_2m','斜面の菱形金網',wire(False),FENCE_REFS),
              ('slope_fence_wire_end','金網の終端支柱',wire(True),FENCE_REFS),
              ('slope_fence_bar_2m','橋側の黒い縦格子',bar(False),BAR_REFS),
              ('slope_fence_bar_end','縦格子の終端支柱',bar(True),BAR_REFS),
              ('red_arch_bridge_v2','釜田川橋の上部工・欄干',bridge(),BRIDGE_REFS),
              ('wall_top_concrete_2m','擁壁天端の白いコンクリート壁',wall(),WALL_REFS)]
    parts = []
    for name, meaning, mesh, keys in assets:
        v, f, uv = np.asarray(mesh.v),np.array(mesh.f,np.int32),np.array(mesh.uv)
        assert np.isfinite(v).all() and np.isfinite(uv).all() and uv.min()>=0 and uv.max()<=1
        assert np.linalg.norm(np.cross(v[f[:,1]]-v[f[:,0]],v[f[:,2]]-v[f[:,0]]),axis=1).min()>1e-9
        np.savez_compressed(ASSET/(name+'.npz'),v=v,f=f,uv=uv)
        is_wall, is_bridge = name.startswith('wall'), name=='red_arch_bridge_v2'
        colours = {**COLOURS,'cream':(199,195,185)}
        im = atlas(wall_art() if is_wall else None, changes=colours,seed=name)
        im.save(ASSET/(name+'.png'),optimize=True)
        merged = []
        for label,group in groupby(mesh.components,key=lambda c:c['name']):
            cs = list(group)
            merged.append(dict(name=label,triangle_start=cs[0]['triangle_start'],
                               triangle_count=sum(c['triangle_count'] for c in cs)))
        p = dict(name=name,meaning=meaning,triangles=len(f),vertices=len(v),dimensions_m=np.ptp(v,axis=0).round(6).tolist(),
                 bounds_m=[v.min(0).round(6).tolist(),v.max(0).round(6).tolist()],
                 origin='橋長の中央・道路中心線、路面から0.70 m下' if is_bridge else
                        '始端小口の底面中央' if is_wall else '支柱の根元中央',
                 front='-Y', textures=[name+'.png'],components=merged,
                 references=[refs[k] for k in keys],dimension_basis='dimension_evidence.json')
        if name.endswith('_2m'):
            p.update(repeat_length_m=2,connect_direction='+X',connect_points_m=[[0,0,0],[2,0,0]])
        if is_bridge:
            p.update(length_m=99,deck_z_m=ROAD,long_axis='+Y',entry_y_m=-49.5,rail_center_x_m=[LEFT,RIGHT],
                     arch_center_x_m=[LEFT-.83,RIGHT+.83])
        parts.append(p)
    total = sum(p['triangles'] for p in parts)
    assert total<=15000,total
    write_json(ASSET/'asset.json',dict(asset_id='r70_fubuki',source='r70_fubuki.blend',generator='blender_scripts/build_r70_fubuki.py',
               blender_generator='blender_scripts/r70_fubuki_source.py',coordinates=dict(unit='m',up='+Z',front='-Y',origin='各partのorigin参照'),
               npz=dict(keys=['v','f','uv'],uv='v=0が画像の上端。Blender・landmarks.loadで1-v。',location='制作元と同じフォルダ'),
               texture_method='1024角、自作の手続き色・コンクリートの色むら。網線は立体。全材質不透明。',
               source_image_usage='ストリートビューは照合専用。配布物への画素転用なし。',
               game_status='単体で照合。コース側の配置とゲーム確認は監督担当。',triangle_total=total,
               source_visibility='部品ごとにコレクション。初期表示は赤い橋。',parts=parts))
    for p in parts:
        print(p['name'],p['triangles'],p['dimensions_m'])
    print('total',total)


if __name__=='__main__':
    main()
