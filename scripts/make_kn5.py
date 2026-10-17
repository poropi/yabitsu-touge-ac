import argparse

from yabitsu import config, kn5

TARGETS = {
    "test": (config.DATA_PROCESSED / "test" / "manifest.json", config.TEST_TRACK_DIR,
             ["yabitsu_test", "test_objects"]),
    "s1": (config.S1_DIR / "manifest.json", config.TRACK_DIR, ["yabitsu_s1", "s1_up_objects", "s1_down_objects"]),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target", choices=TARGETS)
    manifest, track_dir, groups = TARGETS[ap.parse_args().target]
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
