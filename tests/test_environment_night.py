import importlib
import importlib.util
import json
import configparser
from pathlib import Path

import numpy as np

from yabitsu import config, kn5



def read_sections(path):
    """各ブロックを独立に読む。INIpp自動採番の実装をテスト側で模倣しない。"""
    import re
    result={}
    for block in re.split(r'(?m)(?=^\[)',path.read_text()):
        if not block.startswith('['): continue
        one=configparser.ConfigParser(interpolation=None,strict=True)
        one.read_string(block)
        section=one.sections()[0]
        key=section
        if section.endswith('_...'):
            key=section[:-3]+one[section]['description']
        assert key not in result
        result[key]=dict(one[section])
    return result

def module():
    assert importlib.util.find_spec('yabitsu.environment_night'), '夜間生成器が必要'
    return importlib.import_module('yabitsu.environment_night')


def test_signal_lenses_stay_inside_existing_head_and_use_correct_colour():
    e = module()
    q = np.array([[0,0,0],[1.2,0,0],[1.2,0,.4],[0,0,.4]])
    v, f = e.signal_lens(q, False)
    assert v[:,0].min() > 0 and v[:,0].max() < .4
    assert v[:,2].min() > 0 and v[:,2].max() < .4
    assert abs(v[:,1]).max() < .02
    v, f = e.signal_lens(q, True)
    assert v[:,2].mean() > .2


def test_build_is_additive_and_bounds_light_cost(tmp_path):
    e = module()
    out, report = e.build(tmp_path)
    m = json.loads((tmp_path/'manifest.json').read_text())
    model = kn5.read_kn5(out)
    assert len(model['root']['children']) == len(m['meshes'])
    assert report['counts']['signals'] > 0 and report['counts']['windows'] > 0
    assert report['counts']['city_dots'] >= 80
    assert report['counts']['street_lights'] <= 15
    assert report['counts']['tunnel_lights'] <= 75
    assert json.loads((tmp_path/'retex.json').read_text()) == {}
    assert json.loads((tmp_path/'hidden.json').read_text()) == []
    assert all(not n[0].isdigit() for n in m['meshes'])
    assert all(np.isfinite(np.load(tmp_path/x['file'])['verts']).all() for x in m['meshes'].values())


def test_ini_hides_daytime_city_and_never_emits_signal_housing(tmp_path):
    e = module(); e.build(tmp_path)
    ini = read_sections(tmp_path/'extension/parts/night.ini')
    sections = list(ini.values())
    city = ini['MATERIAL_ADJUSTMENT_ENV_CITY']
    assert city['condition'] == 'NightLights'
    assert city['value_0_off'] == '0,0,0'
    assert city['key_1'] == 'alpha' and city['value_1_off'] == '0'
    assert float(ini['Material_DistantEmissive']['dotmaxdistance']) >= 15000
    for s in sections:
        if 'materials' in s:
            assert s['materials'] not in ('signal_head','signal_ped','pole_grey')
        if 'range' in s:
            assert float(s['range']) <= 14 and s['shadows'] == '0'
            assert s['direction'] == '0,-1,0'
    assert ini['LIGHT_SERIES_ENV_TUNNEL']['condition'] == 'ALWAYS_ON'


def test_new_parking_posts_are_outside_road_and_pit_lane():
    from yabitsu import start_lot
    settings=json.loads((config.DATA_MANUAL/'environment_lights.json').read_text())
    lot=start_lot.load_spec(config.DATA_MANUAL/'s1_start_lot.json')
    frame=start_lot.frame(lot)
    sec=json.loads((config.DATA_PROCESSED/'track_section.json').read_text())
    centerline=np.c_[sec['x'],sec['y']]
    for item in settings['parking_posts']:
        p=frame.xy(item['a'],item['b'])
        distances=np.linalg.norm(centerline-p,axis=1); i=distances.argmin()
        assert distances[i] > max(sec['hw_left'][i],sec['hw_right'][i])+1
        assert item['b'] > lot['lot']['b'][1]+.5


def test_night_condition_is_defined_locally_and_switches_with_sun(tmp_path):
    e=module(); e.build(tmp_path)
    ini=read_sections(tmp_path/'extension/parts/night.ini')
    conditions=[value for name,value in ini.items() if name.startswith('CONDITION_')]
    night=[c for c in conditions if c.get('name')=='NightLights']
    assert len(night)==1, 'NightLightsを共通ファイルの未定義名に依存させない'
    condition=night[0]
    assert condition['input']=='SUN' and float(condition['lag'])==0
    points=[tuple(map(float,p.split('='))) for p in condition['lut'].strip('()|').split('|')]
    x,y=np.asarray(points).T
    assert np.interp(40,x,y)==0 and np.interp(100,x,y)==1
    assert np.interp(90,x,y)==.5
    for section in ini:
        if section.startswith('SHADER_REPLACEMENT_ENV_'):
            assert all(mat.strip().startswith('env_') for mat in ini[section]['materials'].split(','))
    assert ini['Material_DistantEmissive']['meshes']=='env_city_hadano'


def test_repeated_csp_sections_use_numeric_or_auto_indices(tmp_path):
    import re
    e=module(); e.build(tmp_path)
    text=(tmp_path/'extension/parts/night.ini').read_text()
    names=re.findall(r'^\[(LIGHT_SERIES_[^]]+|MATERIAL_ADJUSTMENT_[^]]+|SHADER_REPLACEMENT_[^]]+)\]$',text,re.M)
    assert len(names)==13
    assert all(re.fullmatch(r'(LIGHT_SERIES|MATERIAL_ADJUSTMENT|SHADER_REPLACEMENT)_(\d+|\.\.\.)',n) for n in names)


def test_vending_front_faces_out_of_body_and_keeps_texture(tmp_path):
    e=module(); e.build(tmp_path)
    m=json.loads((tmp_path/'manifest.json').read_text())
    source=json.loads((config.TRACK_BUILD_DIR/'manifest.json').read_text())
    bodies=[]; n_front=0
    for info in source['meshes'].values():
        if info['material'] in ('vending','vending_red'):   # 峠の売店の前のコカ・コーラは赤い箱（vending_red）
            z=np.load(config.TRACK_BUILD_DIR/info['file'])
            bodies.extend(c.mean(0) for c in e._components(z['verts'],z['faces']))
        if info['material']=='vending_front':
            z=np.load(config.TRACK_BUILD_DIR/info['file']); n_front+=len(e._components(z['verts'],z['faces']))
    fronts=[v for v in m['meshes'].values() if v['material']=='env_vending_front']
    assert len(fronts)==n_front>=2
    for info in fronts:
        z=np.load(tmp_path/info['file']); q=z['verts'][z['faces'][0]]
        c=z['verts'].mean(0); body=min(bodies,key=lambda b:np.linalg.norm(c-b))
        normal=np.cross(q[1]-q[0],q[2]-q[0])
        assert np.dot(normal,c-body)>0
        assert np.ptp(z['uvs'],axis=0).min()>.9
    ini=read_sections(tmp_path/'extension/parts/night.ini')
    assert ini['MATERIAL_ADJUSTMENT_ENV_VENDING']['materials']=='env_vending_front'
