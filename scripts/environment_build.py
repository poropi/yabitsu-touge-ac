"""追加KN5と夜間・雨設定を生成。--no-installではゲーム側を変更しない。"""
import argparse
import shutil

from yabitsu import acdata, config, environment_night, environment_rain, environment_audio, environment_seasons, ext_config, layouts


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-install',action='store_true')
    args=parser.parse_args()
    path,report=environment_night.build()
    print('夜間環境:',path,report['counts'])
    rain_path,rain_report=environment_rain.build(path.parent)
    print('雨設定:',rain_path,'トンネル',rain_report['tunnel_count'],
          '雨除け',len(rain_report['occluders']))
    audio_path,audio_report=environment_audio.build()
    print('環境音:',len(audio_report['sources']),'反響領域:',len(audio_report['reverb_zones']))
    environment_seasons.write_config(path.parent)
    if args.no_install: return
    track=config.assert_writable(config.TRACK_DIR)
    shutil.copy2(path,track/path.name)
    shutil.copy2(audio_path,track/audio_path.name)
    shutil.copytree(audio_path.parent/'extension',track/'extension',dirs_exist_ok=True)
    ext_config.write_part(track,'night',(path.parent/'extension/parts/night.ini').read_text())
    ext_config.write_part(track,'rain',rain_path.read_text())
    environment_seasons.write_config(track)
    ext_config.assemble(track)
    for layout in layouts.load_layouts():
        acdata.write_models_ini(track/f'models_{layout.name}.ini',layouts.kn5_files(layout.name))
        p=track/layout.name/'data/audio_sources.ini'
        p.parent.mkdir(parents=True,exist_ok=True)
        original=p.read_text() if p.exists() else ''
        p.write_text(environment_audio.merge_audio_sources(original,(audio_path.parent/'audio_sources.ini').read_text()))


if __name__=='__main__':
    main()
