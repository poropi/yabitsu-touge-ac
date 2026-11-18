"""依頼09 橋の部品と全橋構造。路面と谷の地形を含めない。"""
from route70_09_common import *

def railing(kind,L=2,H=.95,c='white',rows=2,spacing=.13):
    m=Mesh();D=.14
    if kind in ['log','timber','concrete_two']:
        if kind=='log':
            m.tube('擬木支柱',(0,0,0),(0,0,H),.105,'concrete',10,r2=.09)
            for z in [.35,H-.09]:m.tube('丸太風横木',(-L/2,0,z),(L/2,0,z),.065,'concrete',10)
            for z in [.37,H-.08]:m.tube('接合部の木口',(0,-.12,z),(0,.12,z),.045,'concrete',8)
        else:
            m.box('角柱',(0,0,H/2),(.18,.2,H),c)
            for z in [.28,H-.12]:m.box('角形横桟',(0,0,z),(L,.12,.13),c)
            m.box('柱頭',(0,0,H),(.21,.23,.065),c)
    elif kind in ['solid','window','window_metal']:
        # 窓状開口は実際に穴を空けた構成。裏側にも抜ける。
        hh=.85 if kind!='solid' else H
        if kind=='solid':m.box('無開口の腰壁',(0,0,H/2),(L,.28,H),'concrete')
        else:
            m.box('下帯',(0,0,.13),(L,.28,.26),'concrete')
            m.box('上帯',(0,0,hh-.12),(L,.3,.24),'concrete')
            m.box('開口間のコンクリート縦柱',(0,0,(.26+hh-.24)/2),(.7,.28,hh-.50),'concrete')
            for x in [-L/2+.12,L/2-.12]:m.box('開口端',(x,0,(.26+hh-.24)/2),(.24,.28,hh-.50),'concrete')
        if kind=='window_metal':
            for z in [.93,1.48]:m.box('増設手すり横桁',(0,0,z),(L,.07,.065),'wood')
            for x in np.arange(-L/2+.1,L/2,.16):m.box('増設縦格子',(float(x),0,1.2),(.028,.04,.58),'wood')
            m.box('増設支柱',(0,0,1.18),(.065,.075,.68),'wood')
    else:
        m.box('縁石',(0,0,.075),(L,.26,.15),'concrete')
        m.box('鋼製主柱',(0,0,H/2),(.09,.1,H),c)
        zs=np.linspace(.3,H-.045,rows)
        for z in zs:
            if kind=='pipe':m.tube('丸鋼管',(-L/2,0,z),(L/2,0,z),.036,c,10)
            elif kind=='board':m.box('木の化粧板',(0,0,z),(L,.09,.16),'wood')
            elif kind=='wbeam':
                profile=[(-.08,-.14),(.02,-.08),(-.025,0),(.02,.08),(-.08,.14)]
                for (yy,z1),(yb,z2) in zip(profile[:-1],profile[1:]):
                    m.surface('波形防護板',[(-L/2,yy,z+z1),(L/2,yy,z+z1),(L/2,yb,z+z2),(-L/2,yb,z+z2)],[(0,1,2,3)],c)
            else:m.box('角形横桁',(0,0,z),(L,.09,.07),c)
        if kind=='lattice':
            for x in np.arange(-L/2+.1,L/2,spacing):m.box('細い縦格子',(float(x),0,H/2+.06),(.026,.04,H-.3),c)
    return m

def post(kind='square',H=1,c='concrete',text=None):
    m=Mesh();w=.4
    if kind=='large':w=.7
    if kind=='brick':w=.72;c='rust'
    if kind=='stone':w=.8
    if kind=='round':
        m.box('丸頭親柱本体',(0,0,(H-.15)/2),(.38,.38,H-.15),c)
        m.tube('丸みのある柱頭',(-.192,0,H-.19),(.192,0,H-.19),.19,c,12)
    else:
        m.box('親柱本体',(0,0,(H-.07)/2),(w,.4,H-.07),c)
        m.box('笠石',(0,0,H-.035),(w+.06,.45,.07),c)
        if kind=='stone':
            for z in np.arange(.16,H,.18):m.box('石積みの横目地',(0,-.201,float(z)),(w,.006,.012),'dark')
    plate(m,min(w*.82,.62),.19,H*.57,text=text,y=-.212)
    return m

def full_bridge(arch=False):
    m=Mesh();L=99 if arch else 20;W=8.8 if arch else 4.4;road=.7
    for side in [-1,1]:
        x=side*W/2
        m.box('主桁',(x,0,.35),(.32,L,.7),'red' if arch else 'dark')
        count=math.ceil(L/2);span=L/count
        rail=railing('lattice' if arch else 'pipe',span,1.1,'white',3,spacing=.23)
        for y in np.arange(count)*span-L/2+span/2:merge(m,rail,math.pi/2,(x,float(y),road))
        if arch:
            # 箱桁リブの傾斜部、上部の緩い曲線、吊材を立体化。
            ys=np.linspace(-41,41,21);zz=road+9*np.maximum(0,1-(ys/41)**2)
            for y0,y1,z0,z1 in zip(ys[:-1],ys[1:],zz[:-1],zz[1:]):m.beam('赤い箱断面アーチリブ',(x,y0,z0),(x,y1,z1),.55,.8,'red')
            for y in np.linspace(-34,34,9):
                z=road+9*(1-(y/41)**2);m.beam('鉛直吊材',(x,y,road),(x,y,z),.15,.18,'red')
            for y in np.linspace(-30,30,6):
                z=road+9*(1-(y/41)**2);m.beam('リブ接合の補剛板',(x-.07,y,z-.5),(x+.07,y,z+.5),.14,.2,'red')
        elif side==-1:
            # 写真左の片側だけにある平行弦トラス。
            for z in [road+.25,road+2.6]:m.box('トラス弦材',(x-.16,0,z),(.14,L,.16),'dark')
            for j,y in enumerate(np.linspace(-L/2,L/2,9)):
                m.beam('トラス縦材',(x-.16,y,road+.25),(x-.16,y,road+2.6),.09,.1,'dark')
                if j<8:
                    y1=y+L/8;m.beam('トラス斜材',(x-.16,y,road+.25),(x-.16,y1,road+2.6),.08,.08,'dark')
                    m.beam('トラス交差斜材',(x-.16,y,road+2.6),(x-.16,y1,road+.25),.07,.07,'dark')
    for y in np.linspace(-L/2,L/2,13):m.box('床下横桁',(0,y,.37),(W,.18,.3),'red' if arch else 'dark')
    if arch:
        for y in np.linspace(-25,25,5):
            z=road+9*(1-(y/41)**2);m.box('上横構',(0,y,z),(W,.22,.28),'red')
    return m

def main():
    a=Delivery('r70_bridges')
    definitions=[
      ('log_3m','灰色擬木丸太2段','log',3,.9,'concrete',2),
      ('timber_2m','木調角形2段','timber',2,.9,'wood',2),
      ('concrete_two_2m','苔むすコンクリート角柱と2段横木','concrete_two',2,.9,'concrete',2),
      ('steel_black_2m','黒塗装縦格子','lattice',2,1,'dark',2),
      ('timber_lattice_2m','淡い木調の縦格子','lattice',2,.95,'cream',2),
      ('steel_white_lattice_2m','灰白縦格子','lattice',2,1.1,'white',3),
      ('steel_brown_lattice_2m','茶色縦格子','lattice',2,1.1,'wood',3),
      ('steel_rust_2m','赤茶角形横桁3段','horizontal',2,1,'rust',3),
      ('steel_white_two_2m','白灰角形横桁2段','horizontal',2,1,'white',2),
      ('pipe_white_two_2m','白鋼管2段','pipe',2,.95,'white',2),
      ('pipe_moss_two_2m','緑がかった鋼管2段','pipe',2,.9,'cream',2),
      ('pipe_gray_three_2m','灰色鋼管3段','pipe',2,1,'metal',3),
      ('pipe_white_three_2m','白鋼管3段','pipe',2,1.1,'white',3),
      ('wood_clad_2m','木化粧板付き鋼製欄干','board',2,1,'wood',3),
      ('solid_concrete_2m','低いコンクリート壁','solid',2,.8,'concrete',2),
      ('concrete_window_2m','四角い窓が抜けたコンクリート高欄','window',2,.9,'concrete',2),
      ('window_brown_top_2m','窓あき高欄と茶色増設手すり','window_metal',2,1.5,'concrete',2),
      ('wbeam_white_2m','橋端につながる白い波形防護板','wbeam',2,.8,'white',1)]
    # 同型を使う橋を割り当てた後、参照IDを各部品へ逆引きする。
    rows=[
    ('S4',1520,8,'log_3m','log_3m','log_terminal','擬木の柱は写真で約3m間隔。専用の丸太形端柱'),
    ('S4',1645,24,'steel_black_2m','steel_black_2m','square_10','縦格子は両側。長さ20m前後の依頼値から推定24m'),
    ('S4',1720,8,'concrete_two_2m','concrete_two_2m','square_09','手前に白い注意板。橋外のガードレールは別'),
    ('S4',1895,10,'timber_lattice_2m','timber_lattice_2m','timber_terminal','写真右の明るい縦格子と木調端柱を優先'),
    ('S4',2095,10,'wbeam_white_2m','solid_concrete_2m',None,'左は白防護板、右は低壁'),
    ('S4',2694,18,'pipe_moss_two_2m','pipe_moss_two_2m','square_08','檜沢橋の青文字の縦札が読める'),
    ('S4',2969,10,'pipe_moss_two_2m','pipe_moss_two_2m','square_09','右横材は白に近い灰褐色'),
    ('S4',3069,18,'concrete_two_2m','concrete_two_2m','large_stone','親柱の文字は未判読'),
    ('S4',3369,14,'steel_white_lattice_2m','steel_white_two_2m','square_10','BACK写真の右側が進行方向左。左右を区別'),
    ('S4',3919,12,'pipe_white_two_2m','pipe_white_two_2m','square_10','親柱0.3m程度。写真では幅を近似'),
    ('S4',4269,8,'concrete_two_2m','concrete_two_2m','square_09','写真は無開口壁でなく2段横材と支柱。説明を訂正'),
    ('S4',4768,10,'concrete_two_2m','concrete_two_2m','square_10','木調横桟の太い角柱'),
    ('S4',5468,8,'pipe_white_two_2m','pipe_white_two_2m','square_10','白い太い角柱'),
    ('S4',5793,10,'timber_2m','timber_2m','square_09','木調の角形横桟。長さは推定'),
    ('S4',6093,12,'steel_white_two_2m','steel_white_two_2m',None,'遠景。長さ・欄干詳細の不確かさ±50%。札掛橋と同一の可能性、併置保留'),
    ('S4',6168,24,'pipe_white_three_2m','pipe_white_three_2m','square_10','写真は3段の白横桁。見える複数スパンから推定'),
    ('S4',6367,10,'wood_clad_2m','wood_clad_2m','square_09','横材に茶色化粧板'),
    ('S5',43,18,'steel_rust_2m','steel_rust_2m','square_10','写真では横材3段。依頼説明の2段から訂正'),
    ('S5',1168,8,'pipe_moss_two_2m','steel_brown_lattice_2m',None,'写真の右側は板壁ではなく細い縦格子'),
    ('S5',4216,8,'steel_white_lattice_2m','wood_clad_2m',None,'BACK右側の白格子と左側の木調横材を分ける'),
    ('S5',4591,18,'pipe_gray_three_2m','pipe_gray_three_2m','rounded_08','塩水橋。丸頭親柱'),
    ('S5',4841,8,'timber_2m','timber_2m','square_09','木調低欄干。全長は推定'),
    ('S5',5390,20,'aouji_bridge','aouji_bridge',None,'片側トラスを含む全構造。県橋梁点検一覧の橋長20mを採用。路面は含まない'),
    ('S5',8064,6,'concrete_two_2m','concrete_two_2m','square_09','短い木目風コンクリート'),
    ('S5',8689,14,'steel_white_lattice_2m','steel_white_two_2m','square_10','BACK右側は角横桁、左側は縦格子'),
    ('S5',8938,6,'steel_white_two_2m','steel_white_two_2m',None,'遠景の薄い鋼製欄干'),
    ('S5',9813,100,'steel_white_lattice_2m','steel_white_lattice_2m','brick_10','0947は前景、0949は橋内、0952は同じ橋終端の可能性。重複配置しない'),
    ('S5',9938,100,'steel_white_lattice_2m','steel_white_lattice_2m','brick_10','9813の橋と同一候補。レンガ親柱はこの記録で供給'),
    ('S5',10338,30,'pipe_white_three_2m','steel_brown_lattice_2m',None,'BACK左の茶格子は進行方向右。旧橋が並ぶ可能性'),
    ('S5',10513,99,'steel_white_lattice_2m','steel_white_lattice_2m','square_11','赤アーチと同一橋の取付区間。別橋として重ねない'),
    ('S5',10588,99,'red_arch_bridge','red_arch_bridge','square_11','釜田川橋の全構造。県資料99m。10513の欄干と重複注意'),
    ('S5',10937,40,'concrete_window_2m','window_brown_top_2m','stone_10','両側は異なる高さ'),
    ('S5',11037,50,'pipe_white_three_2m','pipe_white_three_2m',None,'右外側は旧橋の茶格子。別部品として併記'),
    ('S5',11162,36,'concrete_window_2m','window_brown_top_2m','roundcap_10','写真右の増設手すりを維持。10937から続く旧橋の可能性'),
    ('S5',11287,50,'concrete_window_2m','concrete_window_2m','stone_arch_10','1006と1008は同一橋の両端候補'),
    ('S5',11437,54,'steel_brown_lattice_2m','steel_brown_lattice_2m','s5_11437_stone','親柱は熊野橋の可能性。先頭字が写真左端で切れ、全文の刻字は再現しない')]
    for sec,s,L,left,right,p,note in rows:
        refs=[r['oid'] for r in a.refs if r['place'].startswith(f'{sec} {s}')]
        a.locations.append(dict(section=sec,chainage_m=s,length_m=L,left_rail=left,right_rail=right,end_post=p,references=refs,confidence='写真比率による推定、通常±30%。遠景±50%。',notes=note,length_basis='依頼文の橋長の目安と写真の支柱数。25mの地点番号差は橋長ではない。',left_right_basis='ヤビツ峠→宮ヶ瀬の進行方向。BACK画像は左右を反転して読み取る。'))
    official=json.loads((W/'bridge_name_evidence.json').read_text())
    for l in a.locations:
        e=next((e for e in official['other_official_lengths'] if e['source_oid'] in l['references']),None)
        if e:
            l['original_photo_estimate_m']=l['length_m'];l['length_m']=e['length_m'];l['official_bridge_length_m']=e['length_m']
            l['length_basis']='正式名に対応する管理者資料の橋長を欄干部の目安に採用。橋外の取付ガードレールを含まない。'
            l['dimension_source']=e
        if l['section']=='S5' and l['chainage_m'] in [10513,10588]:
            l['official_bridge_length_m']=99;l['length_basis']='釜田川橋の有力候補として県PDF10頁・村統計53頁の橋長99mを採用。アーチ支間とは区別。'
        if l['section']=='S5' and l['chainage_m']==1168:
            l['length_m']=6;l['official_bridge_length_m']=6;l['length_basis']='写真の吉田橋の札と県PDF10頁609行の橋長6m。'
        if l['section']=='S5' and l['chainage_m']==11037:l['right_outer_rail']='steel_brown_lattice_2m'
    for n,meaning,kind,L,H,c,rows_ in definitions:
        refs=[oid for l in a.locations if n in [l['left_rail'],l['right_rail']] for oid in l['references']]
        a.add(n,meaning,railing(kind,L,H,c,rows_),refs,800,L,notes=['柱・横材は写真から近似した断面。縁石以外の路面は含まない。'])
    m=Mesh();m.tube('擬木の終端支柱',(0,0,0),(0,0,.9),.11,'concrete',10,r2=.095)
    a.add('log_terminal','丸太形擬木欄干の終端柱',m,['S4-0358-01'],600,notes=['太い丸太風コンクリート支柱。木目と径の変化は近似。'])
    m=Mesh();m.box('木調終端角柱',(0,0,.475),(.16,.18,.95),'cream');m.box('木調柱の笠木',(0,0,.97),(.19,.21,.05),'cream')
    a.add('timber_terminal','淡い木調縦格子の終端柱',m,['S4-0373-01'],600,notes=['写真で見える木調の角柱。専用の石造親柱を足していない。'])
    for n,kind,H,c in [('square_08','square',.8,'concrete'),('square_09','square',.9,'concrete'),('square_10','square',1,'concrete'),('square_11','large',1.1,'concrete'),('large_stone','large',1,'concrete'),('rounded_08','round',.8,'concrete'),('brick_10','brick',1,'rust'),('stone_10','stone',1,'concrete'),('roundcap_10','round',1,'concrete'),('stone_arch_10','stone',1,'concrete'),('s5_11437_stone','stone',1,'concrete')]:
        refs=[oid for l in a.locations if l['end_post']==n for oid in l['references']]
        m=post(kind,H,c)
        if n=='stone_arch_10':
            m.disc('アーチ形名板の上端',(0,-.223,.68),.18,.15,.014,'dark',20)
        a.add(n,'親柱 '+n,m,refs,600,notes=['判読不能な名板は無地。石目・煉瓦は自作の近似。'])
    for n,text,oid in [('hinokisawa_name','檜沢橋','S4-0404-01'),('tokoyo_name','常世橋','S4-0415-01'),('kiyotaki_name','清滝橋','S4-0467-02'),('jigokuzawa_name','地獄沢橋','S4-0488-02'),('yoshida_name','吉田橋','S5-0600-01'),('aouji_name','青宇治橋','S5-0769-02')]:
        m=Mesh();m.box('橋名標の板',(0,0,1.25),(.33,.035,1.05),'cream');m.panel('青文字の縦札',0,-.018,1.25,.31,1.02);m.tube('標柱',(0,.03,0),(0,.03,1.65),.025,'metal',8)
        a.add(n,'読めた橋名の独立縦札',m,[oid],600,art=lettering(text,240,768,(222,217,162),(40,90,133),vertical=True),notes=['字形・劣化は近似。道路のGoogle表示はモデル化しない。'])
    for n,ar in [('aouji_bridge',False),('red_arch_bridge',True)]:
        refs=['S5-0970-01','S5-0979-01','S5-0975-02','S5-0975-03'] if ar else ['S5-0769-02']
        a.add(n,'赤い鋼製アーチ（釜田川橋の有力候補）の全構造' if ar else '青宇治橋の片側トラス全構造',full_bridge(ar),refs,20000,notes=['路面はコース側で制作。床下桁は推定。ボルト・断面内側・支承は簡略化。','全長は県点検資料の釜田川橋99m・青宇治橋20m。赤橋のアーチ支間82m・立上り9mは写真比率による推定。幅8.8mは村統計資料。','赤橋の同定は形状と場所による有力候補。0979の銘板は画素不足で全文判読できず無地。'],basis='神奈川県橋梁定期点検結果等一覧（PDF10頁）に青宇治橋20m・釜田川橋99m。赤橋の幅8.8mは村統計、ほかは写真比率の推定。',extra={'road_datum_z_m':.7,'longitudinal_direction':'+Y','approach':'-Y端から進入','length_m':99 if ar else 20})
    a.save();print(a.id,len(a.parts),sum(p['triangles'] for p in a.parts))
if __name__=='__main__':main()
