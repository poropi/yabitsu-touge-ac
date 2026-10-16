from yabitsu import assembly, config, textures
from yabitsu.crosssection import Section
from yabitsu.manifest import Manifest

P = config.DATA_PROCESSED


def main():
    sec = Section.load_json(P / "s1_section.json")
    m = Manifest(config.S1_DIR)
    textures.register_materials(m)
    assembly.add_road(m, sec)
    print(m.save()); print(m.summary())


if __name__ == "__main__":
    main()
