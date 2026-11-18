"""依頼09: 写真の棟ごとの外形、板壁、屋根、開口・枠・庇を立体化。"""
from route70_09_common import *

def slab(m,n,quad,thick=.08,c='dark'):
    v=list(quad)+[(x,y,z-thick) for x,y,z in quad]
    m.surface(n,v,[(0,1,2,3),(7,6,5,4),(0,4,5,1),(1,5,6,2),(2,6,7,3),(3,7,4,0)],c)

def window(m,x,y,z,w,h,door=False,c='metal',panes=2):
    # 壁に本当の開口を空け、枠・ガラスを別の奥行きに置く。
    m.box('開口の奥のガラス',(x,y+.055,z),(w-.06,.018,h-.06),'dark')
    for xx in [x-w/2,x+w/2]:m.box('建具の縦枠',(xx,y,z),(.045,.11,h+.05),c)
    for zz in [z-h/2,z+h/2]:m.box('建具の横枠',(x,y,zz),(w+.05,.11,.045),c)
    for k in range(1,panes):m.box('引違い召し合わせ框',(x-w/2+w*k/panes,y-.014,z),(.032,.085,h),c)
    if door:
        m.box('戸の中桟',(x,y-.025,z-.32),(w,.08,.035),c)
        m.tube('引戸の取手',(x+.12,y-.09,z-.15),(x+.12,y-.09,z+.1),.012,'metal',8)
        m.box('沓摺',(x,y-.01,z-h/2-.015),(w+.09,.16,.04),'metal')
    else:
        m.box('窓の下の水切り',(x,y-.055,z-h/2-.03),(w+.1,.18,.04),c)
        m.box('ガラスの反射帯',(x,y+.043,z+.07),(w-.12,.008,.028),'glass')

def wall(m,width,H,openings=(),c='white',siding=None,rotation=0,offset=(0,0,0)):
    q=Mesh();bottom=.25
    xs={-width/2,width/2};zs={bottom,H}
    for x,z,w,h,door in openings:
        xs.update([max(-width/2,x-w/2),min(width/2,x+w/2)]);zs.update([max(bottom,z-h/2),min(H,z+h/2)])
    if siding=='vertical':xs.update(np.arange(-width/2,width/2,.22).tolist())
    if siding=='horizontal':zs.update(np.arange(bottom,H,.22).tolist())
    xx=sorted(xs);zz=sorted(zs)
    for x0,x1 in zip(xx[:-1],xx[1:]):
        for z0,z1 in zip(zz[:-1],zz[1:]):
            x=(x0+x1)/2;z=(z0+z1)/2
            if any(abs(x-a)<w/2-1e-7 and abs(z-b)<h/2-1e-7 for a,b,w,h,_ in openings):continue
            if x1-x0<1e-6 or z1-z0<1e-6:continue
            gx=.006 if siding=='vertical' else 0;gz=.006 if siding=='horizontal' else 0
            q.box('羽目板' if siding else '外壁パネル',(x,0,z),(max(.001,x1-x0-gx),.12,max(.001,z1-z0-gz)),c)
    for x,z,w,h,door in openings:window(q,x,-.075,z,w,h,door,'white' if c=='white' else 'metal')
    merge(m,q,rotation,offset)

def roofs(m,w,d,eave,top,kind='gable',color='dark',ribs=True,front_gable=True,gable_open=False):
    a=w/2+.32;b=d/2+.32
    if kind=='gable':
        slab(m,'厚みのある左屋根',[(-a,-b,eave),(0,-b,top),(0,b,top),(-a,b,eave)],.10,color)
        slab(m,'厚みのある右屋根',[(0,-b,top),(a,-b,eave),(a,b,eave),(0,b,top)],.10,color)
        m.box('棟包み',(0,0,top+.015),(.12,2*b+.05,.09),color)
        for yy in [-b,b]:
            for side in [-1,1]:m.beam('破風板',(side*a,yy,eave-.05),(0,yy,top-.05),.12,.12,'wood')
        for yy in [-d/2,d/2]:
            if gable_open and yy<0:continue
            m.surface('切妻の三角壁',[(-w/2,yy,eave),(w/2,yy,eave),(0,yy,top-.15)],[(0,1,2)] if yy<0 else [(2,1,0)],'white' if front_gable else 'wood')
        if ribs:
            for x in np.arange(-a+.1,a,.25):
                zz=top-(top-eave)*abs(x)/a+.018;m.beam('屋根の波板リブ',(x,-b,zz),(x,b,zz),.014,.018,color)
        for side in [-1,1]:m.tube('軒樋',(side*a,-b,eave-.12),(side*a,b,eave-.12),.045,'dark',8)
    else:
        # 片流れは正面が高く、背面が低い。
        slab(m,'厚みのある片流れ屋根',[(-a,-b,top),(a,-b,top),(a,b,eave),(-a,b,eave)],.10,color)
        for yy,zz in [(-b,top),(b,eave)]:m.box('屋根端の鼻隠し',(0,yy,zz-.06),(2*a,.12,.14),color)
        for side in [-1,1]:
            m.surface('片流れ屋根下の三角壁',[(side*w/2,-d/2,eave),(side*w/2,d/2,eave),(side*w/2,-d/2,top-.12)],[(2,1,0)] if side<0 else [(0,1,2)],'white')
        if ribs:
            for x in np.arange(-a+.1,a,.25):m.beam('片流れ波板リブ',(x,-b,top+.018),(x,b,eave+.018),.014,.018,color)
        m.tube('軒樋',(-a,b,eave-.12),(a,b,eave-.12),.045,'dark',8)
    for x in [-w/2,w/2]:
        m.tube('縦樋',(x,d/2+.1,.2),(x,d/2+.1,eave-.1),.035,'dark',8)

def awning(m,w,d,z,c='dark',wave=False):
    slab(m,'入口の庇',[(-w/2,-d/2-d,z-.12),(w/2,-d/2-d,z-.12),(w/2,-d/2,z),(-w/2,-d/2,z)],.06,c)

def ac(m,x,y,z):
    m.box('室外機本体',(x,y,z),(.78,.30,.55),'white')
    m.disc('室外機ファン格子',(x+.13,y-.16,z),.19,.19,.018,'metal',20)
    for xx in np.arange(x-.04,x+.31,.04):m.box('ファンの縦桟',(xx,y-.172,z),(.009,.008,.30),'white')
    for xx in [x-.28,x+.28]:m.box('室外機脚',(xx,y,z-.30),(.09,.38,.08),'concrete')
    m.tube('エアコン配管',(x-.28,y+.1,z+.1),(x-.28,y+.1,z+1.4),.025,'cream',8)

def stairs(m,w,d,floor):
    x=w/2+.75;y0=-d/2
    count=16;run=.24;rise=floor/count
    for i in range(count):m.box('外階段の踏板',(x,y0+i*run,(i+1)*rise),(1.05,.25,.055),'metal')
    for side in [-1,1]:
        xx=x+side*.52;m.beam('外階段の桁',(xx,y0,.08),(xx,y0+count*run,floor),.12,.15,'dark')
        m.tube('階段手すり',(xx,y0,.95),(xx,y0+count*run,floor+.95),.03,'dark',8)
        for k in [0,4,8,12,16]:m.tube('階段手すり支柱',(xx,y0+k*run,k*rise),(xx,y0+k*run,k*rise+.95),.025,'dark',8)
    landing_front=y0+(count-1)*run+.12
    m.box('階段踊り場',(w/2+.635,landing_front+.5,floor),(1.33,1.0,.08),'metal')
    m.tube('踊り場の外手すり',(w/2+1.27,landing_front,floor+.95),(w/2+1.27,landing_front+1,floor+.95),.03,'dark',8)

def building(b):
    n=b['model_name'];w,d,top=b['estimated_dimensions_xyz_m'];m=Mesh();stories=b['stories'];eave=top-(.85 if stories==1 else 1.0)
    wallc='white';siding=None;roof='gable';rc='dark';front=[];right=[];left=[];back=[]
    if n=='s4_0021_brown':
        wallc='wood';siding='vertical';front=[(0,1.55,2.0,2.2,True),(0,4.1,3.2,1.1,False)]
    elif n=='s4_1550_water_hut':
        wallc='white';right=[(0,1.35,1.3,2.1,True)]
    elif n=='s4_1995_low_hut':
        wallc='dark';siding='horizontal';roof='shed';eave=top-.3
    elif n=='s4_2080_brick_white':
        roof='shed';eave=top-.7;front=[(1.2,1.4,2.25,2.2,True),(3.1,1.8,.8,.6,False)];right=[(-2.1,1.8,1.6,2.25,False),(1,1.8,1.6,2.25,False)]
    elif n=='s4_2120_cafe':
        siding='vertical';front=[(-2.2,1.45,2.5,2,False),(0,1.45,1.25,2.3,True),(2.2,1.45,2.5,2,False),(-1.7,4.3,2.2,1,False),(1.4,4.3,2.2,1,False)];right=[(0,4.2,1.69,1.17,False)]
    elif n=='s4_2744_tin':
        wallc='metal';siding='vertical';roof='shed';eave=top-.25;rc='rust'
    elif n=='s4_2919_notice_hut':
        wallc='concrete';siding='vertical';front=[]
    elif n=='s4_2919_school':
        wallc='wood';siding='horizontal';eave=top-.65
        front=[(float(x),z,1.8,1.2,False) for z in [1.75,4.5] for x in np.linspace(-6,6,6)]
    elif n.startswith('s4_2919_shed_'):
        wallc='metal' if n.endswith('a') else 'wood';siding='horizontal' if n.endswith('a') else 'vertical';front=[(0,1.2,1.55,1.9,True)];rc='rust' if n.endswith('c') else 'dark';eave=top-.55
    elif n=='s4_4044_aframe':
        # A字の屋根が地際へ届く。普通の箱の上の切妻へ置換しない。
        a=w/2;bdepth=d/2
        for side in [-1,1]:slab(m,'地面まで下がるA字屋根',[(side*a,-bdepth,.15),(0,-bdepth,top),(0,bdepth,top),(side*a,bdepth,.15)] if side<0 else [(0,-bdepth,top),(side*a,-bdepth,.15),(side*a,bdepth,.15),(0,bdepth,top)],.08,'metal')
        for y in [-bdepth,bdepth]:m.surface('三角の妻壁',[(-a,y,.15),(a,y,.15),(0,y,top-.09)],[(0,1,2)] if y<0 else [(2,1,0)],'white')
        window(m,0,-bdepth-.03,1.0,.72,1.7,True)
        for side in [-1,1]:m.box('A字の裾の木帯',(side*(a-.05),0,.22),(.12,d,.25),'rust')
        return m
    elif n=='s5_7639_green_hut':
        wallc='wood';siding='vertical';roof='shed';eave=top-.35;rc='green'
    elif n=='s5_7789_frame_hut':
        eave=top-.85;front=[(0,2.05,1.4,.38,False)]
    elif n=='s5_7825_prefab':
        roof='shed';eave=top-.17;front=[(1.2,1.7,2.4,.9,False),(-2,1.35,.9,2.1,True)]
    elif n=='s5_10413_white_house':
        front=[(-1.7,1.65,2.2,1.15,False),(1.65,1.45,1.65,2.0,True)]
    elif n=='s5_11637_green_kiosk':
        wallc='green';roof='shed';eave=top-.25;front=[(0,1.65,2.3,1.25,False)]
    elif n=='s5_11637_mokko_office':
        siding='vertical';roof='shed';top=6.15;eave=5.9;front=[(0,1.65,2.1,1.2,False),(0,4.2,2.1,1.2,False)];right=[(.22,4.1,.9,2.0,True)]
    elif n=='s5_11662_mokko_warehouse':
        wallc='wood';siding='horizontal';roof='shed';eave=top-.55;rc='rust';front=[(-4.2,1.35,2,2.1,True),(0,2.9,2.4,.5,False)]
    m.box('基礎',(0,0,.125),(w+.1,d+.1,.25),'concrete')
    wall(m,w,eave,front,wallc,siding,offset=(0,-d/2,0))
    wall(m,w,eave,back,wallc,siding,math.pi,(0,d/2,0))
    wall(m,d,eave,right,'rust' if n=='s4_2080_brick_white' else wallc,siding,math.pi/2,(w/2,0,0))
    wall(m,d,eave,left,wallc,siding,-math.pi/2,(-w/2,0,0))
    # 片流れの正面側の高い帯を閉じる。開口を作ったeave以下には重ねない。
    if roof=='shed':m.box('片流れの正面上帯',(0,-d/2,(eave+top-.12)/2),(w,.12,max(.01,top-.12-eave)),wallc)
    if n=='s5_7789_frame_hut':
        for y in [-d/2,d/2]:
            m.beam('露出した切妻骨組み',(-w/2,y,eave),(0,y,top),.08,.08,'dark');m.beam('露出した切妻骨組み',(w/2,y,eave),(0,y,top),.08,.08,'dark');m.box('露出した小屋梁',(0,y,eave),(w,.08,.08),'dark')
        # 写真で不確かな屋根材は全面を塞がず、奥の半分だけ残す。
        slab(m,'残存する奥の屋根材',[(-w/2,d*.1,eave),(0,d*.1,top),(0,d/2,top),(-w/2,d/2,eave)],.05,'white')
    else:roofs(m,w,d,eave,top,roof,rc,True,wallc=='white',n=='s4_2919_notice_hut')
    if stories==2:m.box('上下階の胴差',(0,-d/2-.08,2.95),(w,.12,.13),'wood' if wallc=='white' else 'dark')
    # 対象固有の特徴。
    if n=='s4_0021_brown':
        for z in [2.7,5]:
            slab(m,'遠景に見える張出し庇',[(-w/2,-d/2-.9,z),(w/2,-d/2-.9,z),(w/2,-d/2,z+.12),(-w/2,-d/2,z+.12)],.1,'wood')
    if n=='s4_2080_brick_white':
        m.disc('玄関上の楕円窓枠',(1.2,-d/2-.075,3.4),.52,.28,.08,'wood',32)
        m.disc('楕円窓の暗いガラス',(1.2,-d/2-.12,3.4),.46,.22,.015,'dark',32)
        m.box('玄関上の長方形看板',(1.2,-d/2-.1,2.85),(2.2,.08,.36),'cream')
        # 煉瓦側面の額入り案内板。文字は読めないので創作しない。
        q=Mesh();q.box('側面案内板',(0,-.09,1.9),(1.55,.10,.9),'wood');q.box('案内板の無地面',(0,-.151,1.9),(1.4,.014,.76),'cream');merge(m,q,math.pi/2,(w/2,-3.6,0))
        m.box('付属の低い煉瓦縁',(w/2+.55,0,.3),(.24,d,.6),'rust')
    if n=='s4_2120_cafe':
        # 波形の裾は平面画像ではなく立体の帯。
        slab(m,'喫茶店の前庇',[(-w/2,-d/2-1,2.55),(w/2,-d/2-1,2.55),(w/2,-d/2,2.85),(-w/2,-d/2,2.85)],.07,'dark')
        for x in np.arange(-w/2,w/2,.16):m.disc('波形庇の縁',(float(x),-d/2-1.01,2.51),.08,.06,.035,'dark',10)
        for x in [-1.7,1.4]:m.box('窓下の花台',(x,-d/2-.16,3.73),(2.1,.3,.15),'wood')
        m.box('店の無地の横看板',(0,-d/2-.1,3.25),(2.5,.08,.35),'wood')
    if n=='s4_2919_notice_hut':
        for x,ww,hh,c in [(-1.6,1.5,.55,'white'),(-.3,.25,.9,'yellow')]:m.box('読めない貼紙',(x,-d/2-.068,2.3),(ww,.012,hh),c)
        for x in [-w/2,0,w/2]:m.box('露出する木の柱',(x,-d/2-.09,eave/2),(.12,.17,eave),'wood')
        m.box('露出する妻梁',(0,-d/2-.09,eave),(w,.16,.12),'wood')
        m.beam('露出する妻の斜梁',(-w/2,-d/2-.09,eave),(0,-d/2-.09,top-.1),.1,.1,'wood');m.beam('露出する妻の斜梁',(w/2,-d/2-.09,eave),(0,-d/2-.09,top-.1),.1,.1,'wood')
        m.beam('妻の束柱',(0,-d/2-.09,eave),(0,-d/2-.09,top-.1),.08,.08,'wood')
    if n.startswith('s4_2919_shed_'):
        m.box('閉じた引戸',(0,-d/2-.065,1.2),(1.5,.045,1.88),'wood');m.box('引戸の上レール',(0,-d/2-.12,2.2),(1.9,.06,.07),'dark')
        if n.endswith('c'):slab(m,'錆屋根の付け足し庇',[(-w/2,-d/2-.8,1.9),(w/2,-d/2-.8,1.9),(w/2,-d/2,2.25),(-w/2,-d/2,2.25)],.05,'rust')
    if n=='s5_7825_prefab':ac(m,-.5,-d/2-.3,.6)
    if n=='s5_10413_white_house':m.box('煉瓦色の腰帯',(0,-d/2-.07,.55),(w,.12,.6),'rust')
    if n=='s5_11637_mokko_office':
        stairs(m,w,d,3.1);m.box('上部の帯看板',(0,-d/2-.08,6.35),(w,.15,.7),'rust');m.panel('判読した三六の2字',0,-d/2-.16,6.35,2.1,.6)
    if n=='s5_11662_mokko_warehouse':
        m.box('外壁に沿う青シート',(1,-d/2-.13,.65),(w*.7,.04,1.1),'blue')
    return m

def main():
    obs=json.loads((W/'building_observations.json').read_text());a=Delivery('r70_buildings')
    for b in obs['buildings']:
        # 依頼本文で個別行のない1020-04等の参照もasset台帳へ含める。
        for ref in b['references']:
            if not any(r['image']==ref['image'] for r in a.refs):a.refs.append(dict(oid=b['source_oids'][0],image=ref['image'],url=ref['url'],camera=ref.get('camera'),description='観察台帳の追加参照'))
        art=lettering('三六',768,200,(183,176,151),(37,42,38)) if b['model_name']=='s5_11637_mokko_office' else None
        m=building(b)
        p=a.add(b['model_name'],b['title'],m,b['source_oids'],art=art,basis=b['dimension_basis'],notes=b['estimated_or_hidden']+b.get('notes',[])+['背面の開口は資料で見えない所へ創作せず無窓で近似。壁と建具を別の立体部品にし、窓の絵だけでは代替しない。'],extra={'footprint_dimensions_m':b['estimated_dimensions_xyz_m'][:2],'estimated_overall_height_m':b['estimated_dimensions_xyz_m'][2],'observed_features':b['visible_features'],'source_object_ids':b['source_oids'],'photo_roi_xyxy':b['photo_roi_xyxy'],'storeys':b['stories']},changes={'wood':(105,108,88)} if b['model_name']=='s4_2919_school' else None)
        a.locations.append(dict(section=b['section'].upper(),chainage_m=b['section_s_m'],building=b['model_name'],count=1,source_object_ids=b['source_oids'],footprint_dimensions_m=b['estimated_dimensions_xyz_m'][:2],bounds_dimensions_m=p['dimensions_m'],notes=b.get('notes',[]),confidence=b['confidence']))
    a.save();print(a.id,len(a.parts),sum(p['triangles'] for p in a.parts))
if __name__=='__main__':main()
