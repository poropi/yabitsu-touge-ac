"""既存本体のUVに依存しない表示差分を出力する。--no-install で生成のみ。"""
import argparse
import shutil
from yabitsu import acdata,config,ext_config,layouts,visual_updates


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--no-install',action='store_true');a=ap.parse_args()
    p=visual_updates.build();print(p,f'{p.stat().st_size/1e6:.1f} MB')
    if not a.no_install:
        shutil.copy2(p,config.assert_writable(config.TRACK_DIR/p.name))
        ext_config.write_part(config.TRACK_DIR,'visual_updates',(p.parent/'extension/visual_updates.ini').read_text())
        ext_config.assemble(config.TRACK_DIR)
        for lay in layouts.load_layouts():
            acdata.write_models_ini(config.TRACK_DIR/f'models_{lay.name}.ini',layouts.kn5_files(lay.name))


if __name__=='__main__':main()
