"""依頼56と同じカメラで試作を描き、GrassFX対象面を除いた裸地を数える。"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import cKDTree

from scripts import render_roadside_pilot as prior
from scripts.roadside_57 import OUTPUT
from yabitsu import config, kn5


def editable_grass_faces(scene, selected):
    """谷の編集メッシュを元のKN5メッシュの三角形と照合する。"""
    data = json.loads((scene / "site.json").read_text())
    with np.load(scene / "source.npz") as source:
        centers = source["terrain_vertices"][source["terrain_faces"]].mean(axis=1)
    nearby = cKDTree(centers)
    lower, upper = centers.min(axis=0), centers.max(axis=0)
    picked = np.zeros(len(centers), bool)
    for root_name in {mesh["source"] for mesh in selected.values()}:
        root = config.DATA_PROCESSED / root_name
        meshes = json.loads((root / "manifest.json").read_text())["meshes"]
        for name, meta in selected.items():
            if meta["source"] != root_name:
                continue
            with np.load(root / meshes[name]["file"]) as source:
                vertices = source["verts"]
                low, high = vertices.min(axis=0), vertices.max(axis=0)
                if np.any(low > upper) or np.any(high < lower):
                    continue
                triangles = vertices[kn5._triangles(source["faces"])].astype(float)
            d, i = nearby.query(triangles.mean(axis=1), distance_upper_bound=1.e-5)
            picked[i[np.isfinite(d)]] = True
    if not picked.any():
        raise ValueError("谷の編集メッシュでGrassFX対象面を照合できません")
    return np.flatnonzero(picked).tolist()


def compare(out, views, road, widths):
    before = prior.WORK / "views"
    prior_counts = {v["view"]: v for v in json.loads((before / "ground_pixels.json").read_text())}
    cameras = {r["view"]: r for r in json.loads((before / "cameras_after.json").read_text())}
    new_cameras = json.loads((out / "cameras_cover57.json").read_text())
    for row in new_cameras:
        for key in ("location", "direction", "fov_horizontal", "height_from_road", "road_mesh"):
            if row[key] != cameras[row["view"]][key]:
                raise ValueError(f"依頼56からカメラが変わっています: {row['view']} {key}")

    photos, drive = [], []
    results = []
    w, h = prior.SIZE
    font = ImageFont.truetype(str(prior.FONT_PATH), 15)
    for view in views:
        name = view["name"]
        path = out / f"{name}__cover57"
        pixels = np.load(path.with_suffix(".npy"))
        grass_path = path.with_name(path.name + "_grass.npy")
        grass = np.load(grass_path)
        terrain = prior.foreground_roadside_mask(pixels, road, widths)
        visible = terrain != 0
        cover = grass > .5
        bare = visible & ~cover
        Image.fromarray(np.where(bare, 255, 0).astype(np.uint8)).save(path.with_name(path.name + "_ground.png"))
        pixels_total = int(visible.sum())
        counts = prior_counts[name]
        row = dict(view=name, before=counts["before"], request56=counts["after"],
                   now_bare=int(bare.sum()), now_visible_terrain=pixels_total, excluded_as_grass=int((visible & cover).sum()),
                   now_left=int((bare & (terrain > 0)).sum()), now_right=int((bare & (terrain < 0)).sum()))
        results.append(row)
        path.with_suffix(".npy").unlink()
        grass_path.unlink()

        files = [before / f"{name}__{mode}.png" for mode in prior.MODES] + [path.with_suffix(".png")]
        titles = [f"今  {row['before']:,} px", f"依頼56  {row['request56']:,} px",
                  f"依頼57  {row['now_bare']:,} px　緑の地表はGrassFX対象面の代用"]
        if view["photo"]:
            files.insert(0, prior.INVENTORY / "images/sv" / f"{name.replace('r70_', 'R70-', 1)}.jpg")
            titles.insert(0, f"{name}  現地写真")
        sheet = Image.new("RGB", (w * len(files), h + 28), "#24272c")
        panel = ImageDraw.Draw(sheet)
        for i, (file, title) in enumerate(zip(files, titles)):
            with Image.open(file) as img:
                frame = img.convert("RGB").resize(prior.SIZE, Image.Resampling.LANCZOS)
            sheet.paste(frame, (w * i, 28))
            panel.text((w * i + 10, 4), title, font=font, fill="white")
        if view["photo"]:
            photos.append(sheet)
        else:
            panel.text((w - 155, 4), name, font=font, fill="white")
            drive.append(sheet.resize((w * 3 // 2, (h + 28) // 2), Image.Resampling.LANCZOS))

    for panels, name in ((photos, "photo_views.jpg"), (drive, "drive_views.jpg")):
        width, height = panels[0].size
        columns = 1 if name.startswith("photo") else 2
        collage = Image.new("RGB", (width * columns, height * ((len(panels) + columns - 1) // columns)), "#24272c")
        for i, frame in enumerate(panels):
            collage.paste(frame, (width * (i % columns), height * (i // columns)))
        collage.save(out / name, quality=94)
    (out / "ground_pixels.json").write_text(json.dumps(results, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True, type=Path, help="依頼56と同じ半径160 mの切り出し")
    args = parser.parse_args()
    scene = args.scene.resolve()
    site = json.loads((scene / "site.json").read_text())
    if site["site"] != prior.SITE or site["radius"] != 160:
        parser.error("橋Aの半径160 mの切り出しを指定してください")
    out = OUTPUT / "views"
    out.mkdir(exist_ok=True)
    bundle = out / "scene_inputs"
    bundle.mkdir(exist_ok=True)
    views, road, widths = prior.prepare(site, OUTPUT / "trees.npy", bundle, "cover57", None)
    selection = json.loads((OUTPUT / "grass_selection.json").read_text())
    setup_path = bundle / "setup.json"
    setup = json.loads(setup_path.read_text())
    setup["tree_packages"] = str(OUTPUT / "extension/trees")
    setup["grass"] = dict(meshes=sorted(selection["selected"]),
                          valley_faces=editable_grass_faces(scene, selection["selected"]))
    setup["retex"] = {key: str(OUTPUT / "extension" / f"dress_{key}.dds") for key in ("mt_terrain", "mt_grass")}
    setup_path.write_text(json.dumps(setup, indent=2) + "\n")
    with (out / "render_cover57.log").open("w") as log:
        subprocess.run([str(config.BLENDER_EXE), "-b", "--factory-startup", str(scene / "site_original.blend"),
                        "--python-exit-code", "1", "--python", str(prior.ROOT / "blender_scripts/render_roadside_pilot.py"),
                        "--", str(bundle), str(out), "cover57"], stdout=log, stderr=subprocess.STDOUT, check=True)
    compare(out, views, road, widths)
    shutil.rmtree(bundle)


if __name__ == "__main__":
    main()
