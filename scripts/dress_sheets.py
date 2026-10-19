"""S3 のストリートビュー写真（以前の調査の 50 m ごと・上り下り）を、12 枚ずつの見本帳にする。
使い方: uv run python -m scripts.dress_sheets   → export/dress/s3/sheets/sheet_NN.jpg"""
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from yabitsu import config

SV = config.PROJECT_ROOT / ".superpowers" / "sdd" / "2026-10-17-yabitsu-s3" / "sv"
OUT = config.PROJECT_ROOT / "export" / "dress" / "s3" / "sheets"
W, H, COLS, ROWS = 400, 267, 3, 4


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    files = sorted(SV.glob("s3_*_*.png"), key=lambda p: (int(re.findall(r"s3_(\d+)_", p.name)[0]), p.name))
    font = ImageFont.truetype("C:/Windows/Fonts/meiryo.ttc", 22)
    per = COLS * ROWS
    for k in range(0, len(files), per):
        sheet = Image.new("RGB", (W * COLS, H * ROWS), (255, 255, 255))
        d = ImageDraw.Draw(sheet)
        for i, f in enumerate(files[k:k + per]):
            x, y = (i % COLS) * W, (i // COLS) * H
            sheet.paste(Image.open(f).convert("RGB").resize((W, H)), (x, y))
            s, fb = re.findall(r"s3_(\d+)_(\w)", f.name)[0]
            d.rectangle((x, y, x + 150, y + 30), fill=(0, 0, 0))
            d.text((x + 4, y + 2), f"s={int(s)} {'上り' if fb == 'f' else '下り'}", fill=(255, 255, 0), font=font)
        sheet.save(OUT / f"sheet_{k // per:02d}.jpg", quality=85)
    print(len(files), "photos ->", OUT)


if __name__ == "__main__":
    main()
