import argparse

from yabitsu import config, kn5

TARGETS = {
    "test": (config.DATA_PROCESSED / "test" / "manifest.json", config.TEST_TRACK_DIR,
             ["yabitsu_test", "test_objects"]),
}


def _track_target():
    from yabitsu import layouts
    return (config.TRACK_BUILD_DIR / "manifest.json", config.TRACK_DIR,
            layouts.kn5_groups([l.name for l in layouts.load_layouts()]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target", choices=[*TARGETS, "track"])
    target = ap.parse_args().target
    manifest, track_dir, groups = _track_target() if target == "track" else TARGETS[target]
    for g in groups:
        out = kn5.write_from_manifest(manifest, g, track_dir / f"{g}.kn5")
        k = kn5.read_kn5(out)
        nodes = k["root"]["children"]
        meshes = [n for n in nodes if n["cls"] == kn5.NODE_MESH]
        tris = sum(len(n["indices"]) // 3 for n in meshes)
        print(f"{out}  {out.stat().st_size / 1e6:.1f} MB  meshes={len(meshes)} dummies={len(nodes) - len(meshes)} "
              f"tris={tris} materials={len(k['materials'])} textures={len(k['textures'])}")


if __name__ == "__main__":
    main()
