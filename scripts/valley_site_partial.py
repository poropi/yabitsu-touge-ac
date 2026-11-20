"""Run the saved valley partial build against a scratch edit directory."""
import argparse
import os
from pathlib import Path
import sys
from unittest.mock import patch

from yabitsu import assembly, config, valley_edits, valley_surface


def run(out, edits, *, baseline=None, sites_root=valley_edits.SITES_ROOT):
    work = config.PROJECT_ROOT / "export/compare/route70/work"
    sys.path.insert(0, str(work))
    try:
        import reproduce_valleys_39 as reproduce
    finally:
        sys.path.remove(str(work))

    build = valley_surface.build
    local_build = reproduce.previous.LocalBuild
    partial_manifest = reproduce.previous.PartialManifest
    replacements = valley_edits.part_edits(edits)
    rebuilt = set()

    class EditedManifest(partial_manifest):
        def __init__(self, out, source, group_of):
            super().__init__(out, baseline / source.name if baseline is not None else source, group_of)

        def add_mesh(self, group, name, *args, **kwargs):
            if self.root.name == "track" and self.keep(name):
                rebuilt.add(name)
            super().add_mesh(group, name, *args, **kwargs)

    def shape(*args, **kwargs):
        fine = build(*args, **kwargs)
        valley_edits.apply_terrain(fine, edits)
        return fine

    class EditedBuild(local_build):
        def terrain(self):
            valley_edits.register_materials(self.track, edits)
            # The procedural digest is checked before editing, and the coarse
            # cells and remnants are frozen. Keep their exact assembled UVs
            # from the current manifest instead of interpolating a second time.
            parts = list(valley_surface.mesh_parts(self.near))
            if "valley_edit_fingerprint" in self.near:
                names = {n for n, *_ in parts}
                for name in list(self.track.meshes):
                    if name.startswith("1GRASS_valley_") and name not in names:
                        self.track.remove(name)
            for name, v, f, uv, material, normals in parts:
                self.track.add_mesh(assembly.GROUP, name, v, f, uv, material, normals=normals)

        def rivers(self):
            super().rivers()
            if missing := replacements.keys() - rebuilt:
                raise ValueError(f"部分計算で再生成していない部品: {', '.join(sorted(missing))}")
            valley_edits.replace_parts(self.track, edits, replacements=replacements, sites_root=sites_root)
            if replacements:
                self.track.save()

    with (patch.object(valley_surface, "build", shape),
          patch.object(reproduce.previous, "LocalBuild", EditedBuild),
          patch.object(reproduce.previous, "PartialManifest", EditedManifest)):
        reproduce.main(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="リポジトリ外の新規または空の部分計算先")
    parser.add_argument("--edits", type=Path, required=True, help="<site>/edit.json を含む手直し用ディレクトリ")
    parser.add_argument("--baseline", type=Path, help="track と dress_mountain を含む第2巡の控え")
    parser.add_argument("--sites-root", type=Path, default=valley_edits.SITES_ROOT)
    args = parser.parse_args()
    out = args.output.resolve()
    root = config.PROJECT_ROOT.resolve()
    edit_root = args.edits.resolve()
    if (out.is_relative_to(root) or root.is_relative_to(out) or out.is_relative_to(edit_root)
            or edit_root.is_relative_to(out) or (out.exists() and (not out.is_dir() or any(out.iterdir())))):
        parser.error("入力と重ならない、リポジトリ外の新規または空の出力先を指定してください")
    if not edit_root.is_dir():
        parser.error("手直し用ディレクトリが見つかりません")
    out.mkdir(parents=True, exist_ok=True)
    os.environ["YABITSU_GSI_OFFLINE"] = "1"
    os.environ["MPLCONFIGDIR"] = str(out / "mpl")
    run(out, valley_edits.load(edit_root), baseline=args.baseline, sites_root=args.sites_root)


if __name__ == "__main__":
    main()
