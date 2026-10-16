import json

import numpy as np
from PIL import Image

from yabitsu import config, geo, gsi, photo
from yabitsu.centerline import Centerline

P = config.DATA_PROCESSED
SIZE, PX = 250.0, 0.5


def main():
    cl = Centerline.load(P / "s1_centerline.npz")
    cache = gsi.TileCache(); dem = gsi.default_sampler(cache)
    (P / "review").mkdir(parents=True, exist_ok=True); (P / "s1_underlay").mkdir(parents=True, exist_ok=True)
    width_csv = config.DATA_MANUAL / "s1_widths.csv"
    hw = None
    if (P / "s1_section.json").exists():
        from yabitsu.crosssection import Section
        sec = Section.load_json(P / "s1_section.json"); hw = np.interp(cl.s, sec.s, (sec.hw_left + sec.hw_right) / 2)
    under = []
    for k, (sc, x0, y0) in enumerate(photo.review_chunks(cl, 150.0, SIZE)):
        img = photo.render_local_image(cache, geo.from_local, x0, y0, SIZE, PX)
        xs = x0 + np.arange(0.5, SIZE, 1.0); ys = y0 + SIZE - np.arange(0.5, SIZE, 1.0)
        X, Y = np.meshgrid(xs, ys)
        Z = dem.sample(*geo.from_local(X.ravel(), Y.ravel())).reshape(X.shape)
        hs = (photo.hillshade(np.nan_to_num(Z, nan=float(np.nanmean(Z))), 1.0) * 255).astype(np.uint8)
        hs_img = np.array(Image.fromarray(hs).resize(img.shape[1::-1])); hs_img = np.stack([hs_img] * 3, -1)
        left = photo.draw_overlay(img, cl, x0, y0, SIZE, PX, hw); right = photo.draw_overlay(hs_img, cl, x0, y0, SIZE, PX, hw)
        both = Image.new("RGB", (left.width * 2, left.height)); both.paste(left, (0, 0)); both.paste(right, (left.width, 0))
        both.save(P / "review" / f"s1_chunk_{k:02d}_s{int(sc):04d}.png")
        Image.fromarray(img).save(P / "s1_underlay" / f"photo_{k:02d}.jpg", quality=90)
        under.append({"file": f"photo_{k:02d}.jpg", "x0": x0, "y0": y0, "size_m": SIZE})
    (P / "s1_underlay" / "underlay.json").write_text(json.dumps(under), encoding="utf-8")
    print(f"{len(under)} chunks written (width overlay: {'yes' if hw is not None else 'no'}; widths csv: {width_csv.exists()})")


if __name__ == "__main__":
    main()
