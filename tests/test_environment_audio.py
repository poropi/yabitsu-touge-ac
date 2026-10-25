import configparser
import importlib
import json
import wave

import numpy as np
import pytest
from shapely.geometry import Point, Polygon

from yabitsu import config, kn5


def module():
    return importlib.import_module('yabitsu.environment_audio')


@pytest.fixture(scope='module')
def package(tmp_path_factory):
    return module().build(tmp_path_factory.mktemp('environment-audio'))


def test_sources_are_local_daytime_and_cicadas_are_summer_only(package):
    _, report = package
    e = module()
    sources = report['sources']
    assert {s['kind'] for s in sources} == {'birds_stream', 'cicadas'}
    assert 3 <= sum(s['kind'] == 'birds_stream' for s in sources) <= 8
    assert 5 <= sum(s['kind'] == 'cicadas' for s in sources) <= 20
    for source in sources:
        assert e.audible_gain(source, 12, 210, 0) > 0
        assert e.audible_gain(source, 0, 210, 0) == 0
        assert e.audible_gain(source, 12, 210, source['max_distance'] + 1) == 0
        assert e.audible_gain(source, 12, 25, 0) == (0 if source['kind'] == 'cicadas' else source['volume'])
        assert np.isfinite(source['position_ac']).all()
        assert source['min_distance'] < source['max_distance'] <= 100
        if source['kind'] == 'birds_stream':
            assert source['source_mesh'].startswith('water_')
        else:
            assert source['forest_class'] in (1, 2)


def test_reverb_nodes_match_kn5_and_stay_inside_tunnel_longitudinal_bounds(package):
    path, report = package
    model = kn5.read_kn5(path)
    nodes = {n['name']: n for n in model['root']['children']}
    assert not model['materials']
    assert len(nodes) == len(report['reverb_zones'])
    assert {x['tunnel'] for x in report['reverb_zones']} == {
        '1WALL_tunnel_32745017_bore', '1WALL_tunnel_131561907_bore', '1WALL_tunnel_131561908_bore'}
    for zone in report['reverb_zones']:
        p = np.asarray(zone['position_ac'])
        np.testing.assert_allclose(nodes[zone['node']]['matrix'][3, :3], p, atol=.001)
        assert zone['station'] > zone['max_distance']
        assert zone['length'] - zone['station'] > zone['max_distance']
        assert 0 < zone['min_distance'] < zone['max_distance'] <= 13
        assert all(np.linalg.norm(p[[0, 2]] - np.array(portal)[[0, 2]]) > zone['max_distance']
                   for portal in zone['portals_ac'])


def test_audio_sources_merge_preserves_foreign_sections_and_is_idempotent(package):
    path, _ = package
    generated = (path.parent / 'audio_sources.ini').read_text()
    original = '; user settings\n[SOURCE_0]\nVOLUME=0.4\n[REVERB_5]\nNODE=OLD_NODE\n'
    merged = module().merge_audio_sources(original, generated)
    assert merged.startswith(original)
    assert module().merge_audio_sources(merged, generated) == merged
    ini = configparser.ConfigParser(interpolation=None)
    ini.read_string(merged)
    assert ini['REVERB_5']['node'] == 'OLD_NODE'
    assert ini['REVERB_6']['node'].startswith('AC_AUDIO_YABITSU_')
    without_newline = module().merge_audio_sources(original.rstrip(), generated)
    ini.read_string(without_newline)
    assert ini['REVERB_5']['node'] == 'OLD_NODE'
    assert module().merge_audio_sources(without_newline, generated) == without_newline


def test_packaged_audio_is_decodable_not_clipped_and_all_references_resolve(package):
    path, report = package
    root = path.parent
    text = (root / 'extension/parts/audio.ini').read_text()
    assert '[SCRIPT_...]' in text and 'SCRIPT=environment_audio.lua' in text
    script = (root / 'extension/environment_audio.lua').read_text()
    assert 'isValid' in script and 'isPlaying' in script and 'YABITSU_AUDIO' in script
    assert 'ac.onRelease' in script and ':dispose()' in script
    assert all(f'sfx/{s["file"]}' in script for s in report['sources'])
    for entry in report['assets'].values():
        p = root / 'extension/sfx' / entry['filename']
        with wave.open(str(p)) as w:
            assert w.getnchannels() == 1 and w.getsampwidth() == 2
            assert w.getframerate() == 48000
            x = np.frombuffer(w.readframes(w.getnframes()), '<i2').astype(float) / 32768
        assert len(x) > 48000 * 3
        assert .001 < np.sqrt(np.mean(x*x)) < .15
        assert abs(x).max() < .51
        assert abs(x[0] - x[-1]) < .04
    assert json.loads((root / 'retex.json').read_text()) == {}
    assert json.loads((root / 'hidden.json').read_text()) == []


def test_loop_crossfade_does_not_clip_and_keeps_endpoint_continuity():
    sr = 48000
    x = .3 * np.sin(np.arange(sr * 4) * .01)
    x[-sr//4:] *= .3
    y = module().seamless_loop(x, sr, .25)
    assert len(y) == len(x) - sr//4
    assert np.isfinite(y).all() and abs(y).max() < .5
    assert abs(y[0] - y[-1]) < .005
