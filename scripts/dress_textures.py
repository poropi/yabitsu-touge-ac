"""S3 の道路まわりのテクスチャを Codex（image_generation）に作ってもらい、後処理する。
使い方: uv run python -m scripts.dress_textures [name ...]   （src に元画像があれば頼まない。作り直すときは src の画像を消す）
確認画像：export/dress/textures_check.png（各テクスチャを 2×2 に並べたもの）"""
import subprocess
import sys

from PIL import Image, ImageDraw, ImageFont

from yabitsu import config, dress_tex

CODEX = "codex"


def request(name: str):
    spec = dress_tex.TEXTURES[name]
    prompt = (f"画像生成ツールで、ゲーム用のテクスチャ画像を1枚作って、このフォルダに {name}.png という名前で保存してください。"
              f"内容：{spec.prompt} 実寸で横{spec.size_m[0]}m×縦{spec.size_m[1]}mの範囲。{dress_tex._COMMON}"
              "保存したらファイル名と画像の大きさだけ答えてください。")
    dress_tex.SRC.mkdir(parents=True, exist_ok=True)
    r = subprocess.run([CODEX, "exec", "--skip-git-repo-check", "--sandbox", "workspace-write", "-C", str(dress_tex.SRC),
                        "--enable", "image_generation", prompt], capture_output=True, timeout=900,
                       stdin=subprocess.DEVNULL)   # 入力をつなぐと codex はそれを読み終わるまで待つ
    ok = (dress_tex.SRC / f"{name}.png").exists()
    print(f"{name}: {'ok' if ok else 'FAILED'}", flush=True)
    if not ok:
        print(r.stdout.decode("utf-8", "replace")[-1500:])


def check_sheet(names):
    font = ImageFont.truetype(config.font_file("gothic"), 22)
    cell = 320
    cols = 4
    rows = (len(names) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell, rows * cell), (255, 255, 255))
    d = ImageDraw.Draw(sheet)
    for k, n in enumerate(names):
        img, _ = dress_tex.load(n)
        img = img.convert("RGBA")
        w, h = img.size
        t = Image.new("RGBA", (w * 2, h * 2), (40, 40, 40, 255))
        for x in (0, 1):
            for y in (0, 1):
                t.alpha_composite(img, (x * w, y * h))
        t = t.convert("RGB")
        t.thumbnail((cell - 8, cell - 36))
        x0, y0 = (k % cols) * cell, (k // cols) * cell
        sheet.paste(t, (x0 + 4, y0 + 32))
        d.text((x0 + 6, y0 + 2), n, fill=(0, 0, 0), font=font)
    out = config.PROJECT_ROOT / "export" / "dress" / "textures_check.png"
    sheet.save(out)
    return out


def main():
    names = sys.argv[1:] or list(dress_tex.TEXTURES)
    for n in names:
        if not (dress_tex.SRC / f"{n}.png").exists():
            request(n)
    done = [n for n in names if (dress_tex.SRC / f"{n}.png").exists()]
    for n in done:
        dress_tex.process(n)
    print("check:", check_sheet(done))


if __name__ == "__main__":
    main()
