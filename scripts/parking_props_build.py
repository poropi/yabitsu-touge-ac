"""菜の花台の駐車車両（AC の車を Lua で置く）と見物客（Rocketbox）を生成。--no-installでゲームへのコピーを省く。
ゲームへ入れるときは、その PC の AC の content/cars から車の LOD と本体の kn5 を extension/parked_cars/<車>/ へコピーする
（コースのスクリプトは content/cars を読めないため。車のデータは配布物には入れない）。"""
import argparse
import json
import shutil

from yabitsu import acdata, config, ext_config, layouts, parking_props


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-install",action="store_true")
    args = parser.parse_args()
    path = parking_props.build()
    print(f"駐車小物: {path} ({path.stat().st_size:,} bytes)")
    if not args.no_install:
        target = config.assert_writable(config.TRACK_DIR / path.name)
        shutil.copy2(path,target)
        for layout in layouts.load_layouts():
            acdata.write_models_ini(config.TRACK_DIR / f"models_{layout.name}.ini",layouts.kn5_files(layout.name))
        print("ACへ配置:",target)
        ext = config.TRACK_DIR / "extension"
        shutil.copy2(path.parent / "extension" / "parked_cars.lua", config.assert_writable(ext / "parked_cars.lua"))
        ext_config.write_part(config.TRACK_DIR, "parked_cars", (path.parent / "extension" / "parts" / "parked_cars.ini").read_text(encoding="utf-8"))
        ext_config.assemble(config.TRACK_DIR)
        settings = json.loads((config.DATA_MANUAL / "parking_props.json").read_text(encoding="utf-8"))
        cars = {c["ac_car"]["id"]: c["ac_car"] for c in settings["cars"] + settings.get("pass_cars", []) if c.get("ac_car")}
        for car in cars.values():
            src = config.AC_ROOT / "content" / "cars" / car["id"]
            if not all((src / f).exists() for f in (car["lod"], car["main"])):
                print("この AC に車が無いので置かない:", car["id"]); continue
            dst = config.assert_writable(ext / "parked_cars" / car["id"]); dst.mkdir(parents=True, exist_ok=True)
            for f in (car["lod"], car["main"]):
                shutil.copy2(src / f, dst / f)
            print("駐車車両をコピー:", car["id"])


if __name__ == "__main__":
    main()
