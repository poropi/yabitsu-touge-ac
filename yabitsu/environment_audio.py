"""日本録音のCC0環境音と、トンネル内に限定した反響ノードを追加する。"""
import json
import re
import subprocess
import wave
import urllib.request
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.spatial import cKDTree

from . import config, kn5, tunnel
from .manifest import Manifest

GROUP = 'yabitsu_environment_audio'
BEGIN = '; BEGIN YABITSU ENVIRONMENT AUDIO'
END = '; END YABITSU ENVIRONMENT AUDIO'
ASSETS = {
    'cicadas': {'source': 'cicadas_japan_cc0.mp3', 'download': 'https://cdn.freesound.org/previews/643/643877_1778576-hq.mp3',
                'sha256': 'd281f919abb2c6a34d78a7723c5efd1abdbdab90d31666a078e2b4dd2141641a', 'filename': 'cicadas.wav',
                'author': 'cartoonrob', 'url': 'https://freesound.org/people/cartoonrob/sounds/643877/'},
    'birds_stream': {'source': 'birds_stream_japan_cc0.mp3', 'download': 'https://cdn.freesound.org/previews/843/843520_1680683-hq.mp3',
                'sha256': '4b73d8d77a459322794a9d5d00c81b00ea083ca990fd664713f8934c10c3d299', 'filename': 'birds_stream.wav',
                     'author': 'tomasloos', 'url': 'https://freesound.org/people/tomasloos/sounds/843520/'},
}


def seamless_loop(x, sample_rate, fade_seconds=.25):
    n = int(sample_rate * fade_seconds)
    t = np.linspace(0, np.pi / 2, n)
    blend = x[-n:] * np.cos(t) + x[:n] * np.sin(t)
    y = np.r_[x[n:-n], blend]
    # 高域の強い蝉でも切替点にクリックを作らない短い端処理。
    edge = min(int(sample_rate * .002), len(y) // 2)
    y[:edge] *= np.linspace(0, 1, edge)
    y[-edge:] *= np.linspace(1, 0, edge)
    return y


def audible_gain(source, hour, day, distance):
    if not 6 <= hour < 19 or distance >= source['max_distance']:
        return 0
    if source['kind'] == 'cicadas' and not 170 <= day <= 260:
        return 0
    return source['volume']


def merge_audio_sources(original, generated):
    original = re.sub(re.escape(BEGIN) + r'.*?' + re.escape(END) + r'\n?', '', original, flags=re.S)
    start = max([int(x) for x in re.findall(r'\[REVERB_(\d+)\]', original)] + [-1]) + 1
    generated = re.sub(r'\[REVERB_(\d+)\]', lambda m: f'[REVERB_{start + int(m[1])}]', generated)
    prefix = original + ('\n' if original and not original.endswith('\n') else '')
    return prefix + BEGIN + '\n' + generated.rstrip() + '\n' + END + '\n'


def _sources(manifest, root):
    cl = np.load(config.DATA_PROCESSED / 'track_centerline.npz')
    road = cKDTree(np.c_[cl['x'], cl['y']])
    candidates = []
    for name, item in manifest['meshes'].items():
        if not name.startswith('water_'):
            continue
        v = np.load(root / item['file'])['verts']
        d, _ = road.query(v[:, :2])
        i = int(np.argmin(d))
        if d[i] < 65:
            candidates.append((float(d[i]), name, v[i]))
    sources = []
    for _, name, p in sorted(candidates):
        if any(np.linalg.norm(p[:2] - np.array(s['position_local'])[:2]) < 700 for s in sources):
            continue
        p = p + [0, 0, .5]
        sources.append(dict(kind='birds_stream', source_mesh=name, position_local=p.tolist(),
                            min_distance=8, max_distance=90, volume=.5, file='birds_stream.wav'))
        if len(sources) == 6:
            break
    # 生成済みの写真分類を使い、道から10–25mの森へ置く。
    from .forestmap import ForestMap
    fm = ForestMap.load(config.DATA_PROCESSED / 'forest_map.npz')
    from .forest_ground import load_ground
    ground = load_ground(root)
    selected = []
    for station in np.linspace(800, float(cl['s'][-1])-800, 18):
        k = int(np.argmin(abs(cl['s'] - station)))
        angle = cl['heading'][k]
        center = np.array([cl['x'][k], cl['y'][k]])
        for side in (1, -1):
            p = center + side * 18 * np.array([-np.sin(angle), np.cos(angle)])
            cls = int(fm.at(p[None])[0]); z = float(ground.z(p[None])[0])
            if cls in (1, 2) and np.isfinite(z):
                selected.append(dict(kind='cicadas', forest_class=cls, position_local=[*p, z+3],
                                     min_distance=7, max_distance=70, volume=.3, file='cicadas.wav'))
                break
        if len(selected) == 12:
            break
    sources += selected
    for s in sources:
        x, y, z = s['position_local']; s['position_ac'] = [float(x), float(z), float(-y)]
    return sources


def _reverbs(manifest, root, out):
    zones = []
    for name, item in manifest['meshes'].items():
        if not (name.startswith('1WALL_tunnel_') and name.endswith('_bore')):
            continue
        rows = np.load(root / item['file'])['verts'].reshape(-1, tunnel.N_ARC+3, 3)
        center = (rows[:, 0] + rows[:, -1]) / 2
        center[:, 2] += 1.5
        station = np.r_[0, np.cumsum(np.linalg.norm(np.diff(center[:, :2], axis=0), axis=1))]
        length = float(station[-1]); radius = min(12., length/4)
        for d in np.arange(radius+1, length-radius-1, radius*1.4):
            p = np.array([np.interp(d, station, center[:, i]) for i in range(3)])
            node = f'AC_AUDIO_YABITSU_{len(zones):03d}'
            out.add_empty(GROUP, SimpleNamespace(name=node, location=p, rotation_z=0))
            zones.append(dict(node=node, tunnel=name, station=float(d), length=length,
                              position_ac=(p[[0, 2, 1]]*[1, 1, -1]).tolist(),
                              portals_ac=(center[[0, -1]][:, [0, 2, 1]]*[1, 1, -1]).tolist(),
                              min_distance=radius*.65, max_distance=radius))
    return zones


def _lua(sources):
    lines = []
    for s in sources:
        p = ','.join(map(str, s['position_ac']))
        lines.append('{kind="%s",file="sfx/%s",pos=vec3(%s),min=%s,max=%s,volume=%s}' %
                     (s['kind'], s['file'], p, s['min_distance'], s['max_distance'], s['volume']))
    return '''-- Generated by yabitsu.environment_audio. CC0 source credits accompany the WAV files.
local sources = {\n''' + ',\n'.join(lines) + '''\n}
for i,s in ipairs(sources) do
  s.event=ac.AudioEvent.fromFile({filename=__dirname..'/'..s.file,use3D=true,loop=true,minDistance=s.min,maxDistance=s.max},true)
  s.event:setPosition(s.pos)
  s.event.volume=0
  s.event.inAutoLoopMode=true
  s.logged=false
  ac.log('YABITSU_AUDIO loaded '..i..' '..s.kind..' valid='..tostring(s.event:isValid()))
end
local timer=0
local nextReport=10
function script.update(dt)
  timer=timer+dt
  local sim=ac.getSim()
  local day=sim.timeHours>=6 and sim.timeHours<19
  for i,s in ipairs(sources) do
    local season=s.kind~='cicadas' or (sim.dayOfYear>=170 and sim.dayOfYear<=260)
    local active=day and season and ac.getCameraPosition():distance(s.pos)<s.max
    s.event.volume=active and s.volume or 0
    if (active~=s.logged and timer>3) or (active and timer>=nextReport) then
      ac.log('YABITSU_AUDIO state '..i..' active='..tostring(active)..' valid='..tostring(s.event:isValid())..' playing='..tostring(s.event:isPlaying())..' hour='..sim.timeHours..' day='..sim.dayOfYear)
      s.logged=active
    end
  end
  if timer>=nextReport then nextReport=timer+15 end
end
ac.onRelease(function() for _,s in ipairs(sources) do s.event:dispose() end end)
'''


def build(out_dir=None):
    root = Path(out_dir or config.DATA_PROCESSED / 'environment_audio')
    out = Manifest(root)
    original = json.loads((config.TRACK_BUILD_DIR / 'manifest.json').read_text())
    sources = _sources(original, config.TRACK_BUILD_DIR)
    zones = _reverbs(original, config.TRACK_BUILD_DIR, out)
    out.save()
    path = kn5.write_from_manifest(root/'manifest.json', GROUP, root/f'{GROUP}.kn5')
    ext = root/'extension'; (ext/'sfx').mkdir(parents=True, exist_ok=True); (ext/'parts').mkdir(exist_ok=True)
    assets = {}
    for kind, entry in ASSETS.items():
        src = config.DATA_RAW/'environment_audio'/entry['source']
        if not src.exists():
            src.parent.mkdir(parents=True, exist_ok=True)
            data = urllib.request.urlopen(entry['download'], timeout=120).read()
            if hashlib.sha256(data).hexdigest() != entry['sha256']:
                raise ValueError('環境音素材のハッシュが出典確認時と異なります')
            src.write_bytes(data)
        if hashlib.sha256(src.read_bytes()).hexdigest() != entry['sha256']:
            raise ValueError(f'環境音素材が変更されています: {src}')
        result = subprocess.run(['ffmpeg','-v','error','-i',str(src),'-f','f32le','-ac','1','-ar','48000','pipe:1'], check=True, capture_output=True)
        x = np.frombuffer(result.stdout, '<f4').astype(float)
        x -= x.mean(); x = seamless_loop(x, 48000)
        x *= min(.45/max(abs(x).max(), 1e-9), .06/max(np.sqrt(np.mean(x*x)), 1e-9))
        with wave.open(str(ext/'sfx'/entry['filename']), 'wb') as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(48000)
            w.writeframes((x*32767).astype('<i2').tobytes())
        assets[kind] = {**entry, 'license':'CC0-1.0', 'seconds':len(x)/48000,
                        'rms':float(np.sqrt(np.mean(x*x))), 'peak':float(abs(x).max())}
    (ext/'sfx/CREDITS.json').write_text(json.dumps(assets, ensure_ascii=False, indent=2))
    (ext/'environment_audio.lua').write_text(_lua(sources))
    (ext/'parts/audio.ini').write_text('[SCRIPT_...]\nACTIVE=1\nSCRIPT=environment_audio.lua\n')
    text = ''
    for i,z in enumerate(zones):
        text += f'[REVERB_{i}]\nENABLED=1\nNODE={z["node"]}\nMINDISTANCE={z["min_distance"]}\nMAXDISTANCE={z["max_distance"]}\n'
        text += 'PRESET=CUSTOM\nDECAY_TIME=1200\nEARLY_DELAY=5\nLATE_DELAY=10\nHF_REFERENCE=5000\nHF_DECAY_RATIO=20\nDIFFUSION=80\nDENSITY=60\nLOW_SHELF_FREQUENCY=250\nLOW_SHELF_GAIN=0\nHIGH_CUT=750\nEARLY_LATE_MIX=80\nWET_LEVEL=-3\n\n'
    (root/'audio_sources.ini').write_text(text)
    (root/'retex.json').write_text('{}'); (root/'hidden.json').write_text('[]')
    report = dict(sources=sources, reverb_zones=zones, assets=assets,
                  limitations='鳥と小川は同一録音。ヤビツ現地録音ではない。波形検査済み、聴感評価は未実施。')
    (root/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return path, report
