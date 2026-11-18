"""依頼15の門柱と塀。通常のPythonでNPZ・画像と部品台帳を更新する。"""
from itertools import groupby
import json
from pathlib import Path

import numpy as np
from PIL import Image

from route70_common import Mesh

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'assets/landmarks/r70_fubuki'
POST = 'gate_brick_post'
WALL = 'gate_brick_wall_2m'
SIDE = (8, 8, 1016, 712)
TOP = (8, 726, 728, 1016)
METAL = (744, 726, 1016, 1016)


def field(rng, res, size):
    coarse = Image.fromarray(rng.integers(0, 256, res, dtype=np.uint8))
    return np.asarray(coarse.resize(size, Image.Resampling.BICUBIC)).astype(float) / 128 - 1


def brick_patch(rng, size, span, height, top, course):
    """メートル単位の馬踏み目地。照明・陰影は焼き込まない。"""
    w, h = size
    py, px = np.mgrid[:h, :w]
    s, z = (px+.5)/w*span, (h-py-.5)/h*height
    course = .125 if top else course
    row = np.floor(z/course).astype(int)
    along = s + .125*(row % 2)
    col = np.floor(along/.25).astype(int)
    jitter = .0007*field(rng, (45, 65), size)
    a, b = (along+jitter) % .25, (z+jitter) % course
    seam = np.minimum.reduce([a, .25-a, b, course-b])
    clay = np.clip((seam-.0024)/.0035, 0, 1)[..., None]
    salts = field(rng, (38, 96), size)
    broad = field(rng, (9, 20), size)
    fines = rng.normal(0, 2.8, (h,w))
    base = np.array((97, 61, 51) if top else (90, 53, 44))
    tones = rng.normal(0, (7, 3, 2), (row.max()+1,col.max()+1,3))
    bricks = base + tones[row,col] + (broad*4 + salts*6)[...,None]
    # 基部の泥と、横目地に残る白華を疎らに重ねる。
    if not top:
        bricks -= (np.exp(-z/.14)*(6+3*broad))[...,None]
        bricks += (np.maximum(salts+.45*broad-.32, 0)*18)[...,None]
    mortar = np.array((62,48,42)) + (3*broad)[...,None]
    rgb = mortar*(1-clay)+bricks*clay+fines[...,None]
    return np.uint8(np.clip(rgb,0,255))


def paste_padded(canvas, rect, pixels):
    x,y,x1,y1 = rect
    assert pixels.shape[:2] == (y1-y, x1-x)
    canvas[y-4:y1+4, x-4:x1+4] = np.pad(pixels, ((4,4),(4,4),(0,0)), mode='edge')


def texture(name, width, depth, height):
    post = name == POST
    rng = np.random.default_rng(971 if post else 972)
    atlas = np.full((1024,1024,3), (73,51,43), dtype=np.uint8)
    for rect, span, rise, top in [(SIDE, 2*(width+depth), height, False),
                                  (TOP, width, depth, True)]:
        size = rect[2]-rect[0], rect[3]-rect[1]
        paste_padded(atlas,rect,brick_patch(rng,size,span,rise,top,height/(30 if post else 6)))
    if post:
        w,h = METAL[2]-METAL[0], METAL[3]-METAL[1]
        py,px = np.mgrid[:h,:w]
        worn = field(rng,(17,12),(w,h))
        metal = (np.array((37,42,39)) + (3*worn+rng.normal(0,1.6,(h,w)))[...,None])
        # 右1/4に縁用の古びた金属。左3/4は無地の表札。
        rim = px >= w*.75
        metal[rim] += (43,29,16)
        paste_padded(atlas,METAL,np.uint8(np.clip(metal,0,255)))
    Image.fromarray(atlas).save(SRC/(name+'.png'))


def masonry(mesh, label, x0, width, depth, height, wall):
    def rim(z, inset):
        # 塀の両小口を X=0/2 に保ち、隣と突き合わせても天端に溝を作らない。
        left = x0+(0 if wall else inset)
        right = x0+width-(0 if wall else inset)
        return [(left,-depth/2+inset,z),(right,-depth/2+inset,z),
                (right,depth/2-inset,z),(left,depth/2-inset,z)]
    rings = [rim(0,0),rim(height-.009,0),rim(height,.009)]
    dist = np.cumsum([0,width,depth,width,depth])
    for j in range(2):
        for i in range(4):
            k = (i+1) % 4
            quad = [rings[j][i],rings[j][k],rings[j+1][k],rings[j+1][i]]
            uv = [(u/dist[-1],1-p[2]/height)
                  for u,p in zip([dist[i],dist[i+1],dist[i+1],dist[i]],quad)]
            mesh.surface(label,quad,[(0,1,2,3)],uv=uv,rect=SIDE)
    for ring,order in [(rings[0],(3,2,1,0)),(rings[-1],(0,1,2,3))]:
        mesh.surface(label,ring,[order],uv=[((x-x0)/width, .5-y/depth) for x,y,z in ring],rect=TOP)


def plaque(mesh, depth):
    label = '額縁付き表札'
    def rim(width, height, offset):
        return [(sx*width/2,-depth/2-offset,1.36+sz*height/2)
                for sx,sz in [(-1,-1),(1,-1),(1,1),(-1,1)]]
    loops = [rim(.35,1.60,0), rim(.35,1.60,.028), rim(.34,1.59,.035),
             rim(.323,1.573,.035), rim(.318,1.568,.030)]
    trim = [(.80,.12),(.95,.12),(.95,.86),(.80,.86)]
    mesh.surface(label,loops[0],[(3,2,1,0)],uv=trim,rect=METAL)
    for outer,inner in zip(loops, loops[1:]):
        for i in range(4):
            k = (i+1)%4
            mesh.surface(label,[outer[i],outer[k],inner[k],inner[i]],[(0,1,2,3)],uv=trim,rect=METAL)
    mesh.surface(label,loops[-1],[(0,1,2,3)],uv=[(.05,.98),(.69,.98),(.69,.02),(.05,.02)],rect=METAL)


def build(name, refs):
    post = name == POST
    width,depth,height = (1.20,1.32,2.50) if post else (2.,.25,.48)
    mesh = Mesh()
    masonry(mesh,'門柱本体' if post else '低塀本体', -width/2 if post else 0,
            width,depth,height,not post)
    if post:
        plaque(mesh,depth)
    v, f, uv = np.array(mesh.v), np.asarray(mesh.f,dtype=np.int32), np.array(mesh.uv)
    np.savez_compressed(SRC/(name+'.npz'),v=v,f=f,uv=uv)
    texture(name,width,depth,height)
    groups = []
    for label,faces in groupby(mesh.components,key=lambda c:c['name']):
        faces = list(faces)
        groups.append(dict(name=label,triangle_start=faces[0]['triangle_start'],
                           triangle_count=sum(c['triangle_count'] for c in faces)))
    p = dict(name=name,meaning='表札付きのれんが門柱' if post else '門外側の低いれんが塀',
             triangles=len(f),vertices=len(v),dimensions_m=np.ptp(v,axis=0).round(6).tolist(),
             bounds_m=np.stack((v.min(0),v.max(0))).round(6).tolist(),
             origin='れんが本体の底面中央' if post else '始端小口の底辺中央',front='-Y',
             textures=[name+'.png'],components=groups,references=refs,dimension_basis='dimension_evidence.json')
    if not post:
        p.update(repeat_length_m=2,connect_direction='+X',connect_points_m=[[0,0,0],[2,0,0]])
    return p


def main():
    path = SRC/'asset.json'
    info = json.loads(path.read_text())
    ref = next(ref for ref in info['parts'][0]['references'] if '0971_f' in ref['image'])
    rear = dict(ref,image=ref['image'].replace('0971_f','0972_b'),station_m=10438, heading=184.5,
                panoid='RDyrZgr849nF9CcY_1Lt1A',url='https://www.google.com/maps/@35.5084421,139.231088,3a,90y,184.5h,88t/data=!3m6!1e1!3m4!1sRDyrZgr849nF9CcY_1Lt1A!2e0!7i16384!8i8192')
    info['parts'] = [p for p in info['parts'] if p['name'] not in (POST,WALL)]
    info['parts'].extend(build(name,[ref,rear]) for name in (POST,WALL))
    info['triangle_total'] = sum(p['triangles'] for p in info['parts'])
    info['round_15_generators'] = ['blender_scripts/r70_fubuki_bridge_color_15.py',
                                   'blender_scripts/build_r70_fubuki_gate_15.py']
    info['texture_method'] = '1024角。塗装・れんが・金属表札・コンクリートを自作。全材質不透明。'
    assert sum(p['triangles'] for p in info['parts'][-2:]) <= 1500
    path.write_text(json.dumps(info,ensure_ascii=False,indent=2)+'\n')


if __name__ == '__main__':
    main()
