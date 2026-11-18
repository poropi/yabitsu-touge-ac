"""依頼09: 大洞、塩水、吹風の坑門、反復内壁、照明。"""
from route70_09_common import *

SOURCE='https://www.pref.kanagawa.jp/documents/4656/r6shisetsu_gaiyou.pdf#page=25'

def portal(width,height,kind='concrete',south=True,text=''):
    m=Mesh();r=width/2;spring=height-r;dep=.8
    # コンクリートの坑門は上が山形、石張りは台形。
    edge=height+.75;peak=height+1.55 if kind=='concrete' else height+1.15
    def top(x):return peak-(peak-edge)*abs(x)/(r+.8)
    for sign in [-1,1]:
        x0=-r-.8 if sign<0 else r;x1=-r if sign<0 else r+.8
        m.prism('坑門側柱',[(x0,0),(x1,0),(x1,top(x1)),(x0,top(x0))],dep,'concrete')
    arc=[(r*math.cos(t),spring+r*math.sin(t)) for t in np.linspace(math.pi,0,33)]
    for (x,z),(x1,z1) in zip(arc[:-1],arc[1:]):
        vs=[(x,-dep/2,z),(x1,-dep/2,z1),(x1,-dep/2,top(x1)),(x,-dep/2,top(x)),(x,dep/2,z),(x1,dep/2,z1),(x1,dep/2,top(x1)),(x,dep/2,top(x))]
        m.surface('坑門アーチ上の充填',vs,[(0,1,2,3),(7,6,5,4),(0,4,5,1),(3,2,6,7)],'concrete')
    # 入口奥の覆工を1m持たせ、輪郭の厚みを見せる。
    points=arch_path(width,height)
    for (x,z),(x1,z1) in zip(points[:-1],points[1:]):
        m.surface('坑口の内面',[(x,dep/2,z),(x,dep/2+1,z),(x1,dep/2+1,z1),(x1,dep/2,z1)],[(0,1,2,3)],'concrete')
    # 翼壁は非対称。地形は作らず、側面の袖壁だけ。
    for side in [-1,1]:
        H=2.5 if side==1 else 1.3
        m.beam('坑門の翼壁',(side*(r+.6),0,H/2),(side*(r+1.7),-1.2,H/2),H,.38,'concrete')
    if kind=='stone':
        # 浅い化粧石の目地。坑口を塞がない位置だけに配置。
        for z in np.arange(.28,edge,.3):
            opening=0 if z>=height else r if z<spring else math.sqrt(max(0,r*r-(z-spring)**2))
            for side in [-1,1]:
                x0=side*(opening+.08);x1=side*(r+.78)
                if abs(x1-x0)>.02:m.beam('石張り水平目地',(x0,-.406,z),(x1,-.406,z),.008,.008,'dark')
    art=None
    if text:
        z=height+.55
        if kind=='stone':
            m.box('黒地のトンネル名板',(0,-.425,z),(2.3,.05,.36),'black');m.panel('読めた坑名',0,-.452,z,2.2,.30,rect=(8,8,768,768));art=lettering(text,768,130,(27,29,25),(185,179,143))
        else:
            art=Image.new('RGBA',(760,760),(55,58,54,255));dd=ImageDraw.Draw(art)
            from route70_common import font
            for i,ch in enumerate(text):
                # 6文字を画像内で横に並べ、板ごとの領域へUVを振る。
                dd.text(((i+.5)*760/len(text),380),ch,font=font(110),fill=(163,167,158),anchor='mm')
                xx=(i-(len(text)-1)/2)*.64
                m.box('1字ずつの坑名板',(xx,-.425,z),(.5,.05,.5),'dark')
                cell=760/len(text)
                m.panel('読めた坑名 '+ch,xx,-.452,z,.46,.46,rect=(8+i*cell,8+380-cell/2,8+(i+1)*cell,8+380+cell/2))
    return m,art

def lining(width,height,L=20,mesh=False):
    m=Mesh();r=width/2;spring=height-r
    profile=arch_path(width,height)
    for (y,z),(y1,z1) in zip(profile[:-1],profile[1:]):
        # 法線は開口の中心へ向ける。端面に蓋は張らない。
        xx=np.linspace(-L/2,L/2,int(math.ceil(L/2))+1)
        for x,x1 in zip(xx[:-1],xx[1:]):
            # 内壁: 経路に沿って断面→長手方向で回し、法線を内向きにする。
            m.surface('反復内壁',[(x,y,z),(x,y1,z1),(x1,y1,z1),(x1,y,z)],[(0,1,2,3)],'concrete')
    for side in [-1,1]:
        y=side*(r-.06);m.box('側溝の縁',(0,y,.06),(L,.12,.12),'concrete')
        m.box('腰壁',(0,side*r,.55),(L,.035,1.1),'concrete' if mesh else 'white')
        if mesh:m.box('腰壁上の白い帯',(0,side*(r-.025),1.42),(L,.055,.36),'white')
        if not mesh:
            m.tube('壁際の配管',(-L/2,side*(r-.05),1.9),(L/2,side*(r-.05),1.9),.018,'dark',8)
    # 覆工の継ぎ目は内壁から数mm手前。上側のみ、穴は塞がない。
    for x in np.arange(-L/2+5,L/2,5):
        for (y,z),(y1,z1) in zip(profile[1:-2],profile[2:-1]):
            m.beam('覆工の打継目',(x,y*.998,z*.998),(x,y1*.998,z1*.998),.009,.012,'dark')
    if mesh:
        # 補強金網は立体覆工の上に自作透過グリッドを置く。ベースの岩壁は別メッシュ。
        for (y,z),(y1,z1) in zip(profile[1:-2],profile[2:-1]):
            m.surface('補強鋼製メッシュ', [(-L/2,y*.992,z*.995),(L/2,y*.992,z*.995),(L/2,y1*.992,z1*.995),(-L/2,y1*.992,z1*.995)],[(0,3,2,1)],'white',[(0,1),(1,1),(1,0),(0,0)])
        for x in np.arange(-L/2+.6,L/2,1.2):
            for t in np.linspace(.18,math.pi-.18,9):
                y=(r-.03)*math.cos(t);z=spring+(r-.03)*math.sin(t)
                st=len(m.v);m.box('ロックボルト座金',(0,0,0),(.10,.014,.10),'white')
                for vi in range(st,len(m.v)):
                    xx,yy,zz=m.v[vi];m.v[vi]=(xx+float(x),yy*math.cos(t)-zz*math.sin(t)+y,yy*math.sin(t)+zz*math.cos(t)+z)
    return m

def lamp(long=False):
    m=Mesh();w=.65 if long else .28
    m.box('灯具本体',(0,0,.10),(w,.18,.20),'metal')
    m.box('乳白の発光カバー',(0,-.101,.10),(w*.85,.024,.12),'cream')
    for x in [-w*.4,w*.4]:m.box('取付金具',(x,.08,.1),(.04,.15,.16),'dark')
    return m

def main():
    a=Delivery('r70_tunnels')
    moved=json.loads((W/'references.json').read_text())['r70_bridges'];a.refs=[r for r in moved if r['oid']=='S5-0652-01']+a.refs
    configs=[
      ('s5_2442_portal','大洞トンネルの石張り坑口',6.2,5.0,'stone','大洞トンネル',['S5-0652-01']),
      ('s5_4091_portal','小トンネルのコンクリート坑口',4.0,3.8,'small','',['S5-0717-01']),
      ('s5_10663_portal','吹風トンネル 峠側坑口',7.0,5.3,'concrete','吹風トンネル',['S5-0974-01']),
      ('s5_10937_portal','吹風トンネル 宮ヶ瀬側坑口',7.0,5.3,'concrete','吹風トンネル',['S5-0992-01'])]
    for n,meaning,w,h,k,t,refs in configs:
        m,art=portal(w,h,k,text=t)
        if k=='small':
            m.tube('右上の配管',(2.0,-.45,2.4),(2.0,-.45,4.4),.035,'dark',8)
        p=a.add(n,meaning,m,refs,art=art,basis=f'県R6施設概要PDF25頁（紙面22頁）：大洞は車道5.5m、吹風6.0m。開口{w}×{h}mは路肩を含む写真比率の推定。建築限界は開口高と同一ではない。',notes=['地形は含まない。袖壁の背面と厚みは推定。写真を貼らず形状と目地を自作。','小トンネルの正面は参照写真が内部からなので反対側外観を近似。塩水隧道との同定は位置順による推定。'],extra={'opening_width_m':w,'opening_height_m':h})
        p['portal_plane_y_m']=-.4-p['source_anchor_m'][1]
    for n,w,h,mesh,refs in [('oodo_lining_20m',6.2,5,False,['S5-0652-01']),('small_lining_20m',4,3.8,True,['S5-0717-01']),('fukikaze_lining_20m',7,5.3,False,['S5-0974-01','S5-0992-01'])]:
        art=None
        if mesh:
            art=Image.new('RGBA',(760,760),(0,0,0,0));d=ImageDraw.Draw(art)
            for x in range(0,760,8):d.line((x,0,x,759),fill=(156,159,137,255),width=1)
            for y in [0,759]:d.line((0,y,759,y),fill=(156,159,137,255),width=5)
        a.add(n,'20m区切りの内壁 '+n,lining(w,h,20,mesh),refs,span=20,art=art,notes=['内壁は+Xに20m反復。坑口の奥(+Y)へ接続するにはZ軸+90°回して並べる。','透明な補強メッシュとロックボルト座金を立体の覆工に重ねる。継ぎ目は5mおきの推定。','腰壁・配管・曲率は写真による近似。実際の線形のカーブは配置担当が区切りごとに調整。'],extra={'cross_section_width_m':w,'cross_section_height_m':h,'connection_note':'中心原点。両端X=-10,+10。端面の蓋なし。坑口へはZ+90°回転。','lining_inward_normals':True})
    # 末端を実長に合わせられるよう、余り長の部品も供給。
    for n,w,h,L,mesh,refs in [('oodo_lining_end_10_4m',6.2,5,10.4,False,['S5-0652-01']),('small_lining_end_1_4m',4,3.8,1.4,True,['S5-0717-01']),('fukikaze_lining_end_4m',7,5.3,4,False,['S5-0992-01'])]:
        art=None
        if mesh:
            art=Image.new('RGBA',(760,760),(0,0,0,0));d=ImageDraw.Draw(art)
            for x in range(0,760,8):d.line((x,0,x,759),fill=(156,159,137,255),width=1)
            for y in [0,759]:d.line((0,y,759,y),fill=(156,159,137,255),width=5)
        a.add(n,'実長へ合わせる端数内壁',lining(w,h,L,mesh),refs,span=L,art=art,notes=['20m区切りの長手方向の端数。平面の坑門背面厚と接続長は配置時に重なりを調整。'],extra={'cross_section_width_m':w,'cross_section_height_m':h})
    for n,long in [('lamp_box',False),('lamp_fluorescent',True)]:
        a.add(n,'壁付け照明器具',lamp(long),['S5-0992-01'],notes=['器具寸法と光源色は写真比率の推定。夜間の点灯設定は配置担当。'],extra={'emission_surface_component':'乳白の発光カバー','emission_suggested_rgb':[1,.78,.45]})
    a.locations=[
      dict(section='S5',chainage_m=2442,name='大洞トンネル',name_basis='0652_f坑門の字を拡大して判読。',portal='s5_2442_portal',lining='oodo_lining_20m',length_m=50.4,assembly='20m×2＋10.4m×1',official_source=SOURCE,unseen='逆側坑門は指定参照がないため同じ坑門の180°回転で近似。'),
      dict(section='S5',chainage_m=4091,name='塩水隧道（同定は推定、名板は作らない）',name_basis='県道70号の3本の順序と小断面。写真に名板なし。',portal='s5_4091_portal',lining='small_lining_20m',length_m=21.4,assembly='20m×1＋1.4m×1',official_source=SOURCE,unseen='坑門は内部側写真しかなく外観と逆側は推定。別年度資料には25mの記載もあり、R6県概要の21.4mを採用。'),
      dict(section='S5',chainage_m=10663,other_end_m=10937,name='吹風トンネル',name_basis='0992_bの「吹風トンネル」の6字を実画像で判読。県R6一覧と整合。拾い出しの仮称を無確認では使っていない。',portal='s5_10663_portal',other_portal='s5_10937_portal',lining='fukikaze_lining_20m',length_m=244,assembly='20m×12＋4m×1',official_source=SOURCE,unseen='中間部の線形・灯具間隔は未実測。直線区切りで供給。')]
    a.save();print(a.id,len(a.parts),sum(p['triangles'] for p in a.parts))
if __name__=='__main__':main()
