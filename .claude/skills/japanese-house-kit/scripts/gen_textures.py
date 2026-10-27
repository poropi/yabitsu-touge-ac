"""町並みの試作の石垣・道・地面のテクスチャを Codex（image_generation）に作ってもらい、継ぎ目なしにして法線の絵を付ける。
後処理は scripts/dress_textures.py と同じ（yabitsu.dress_tex の make_tileable・height_to_normal）。
元画像 assets/house_kit/src/<name>.png、結果 assets/house_kit/<name>.png・<name>_nm.png。src に元画像があれば頼まない（作り直すときは消す）。
使い方（プロジェクトの根で）: uv run python .claude/skills/japanese-house-kit/scripts/gen_textures.py [name ...]"""
import subprocess, sys
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, ".")
import numpy as np
from PIL import Image
from yabitsu import config, dress_tex

DIR = config.PROJECT_ROOT / "assets" / "house_kit"; SRC = DIR / "src"
COMMON = dress_tex._COMMON
TEX = {   # 名前: (実寸 横×縦 m, 内容, 凹凸の強さ)
    "ishigaki": ((2, 2), "日本の郊外の道ぞいの、古い石積みの擁壁（間知石の谷積み）を正面から。こげ茶から暗い灰色の、角ばった割り石がすき間なく斜めの網目に積まれ、"
                 "石の表面はごつごつして欠けや黒い雨のしみがあり、目地は暗く、ところどころに小さな苔。正方形。", 4.5),
    # 補修のつぎはぎを入れると、敷いたときに 4 m ごとに同じ四角が並んで作り物に見えた（2026-10-28）。目立つ形は入れない
    "asphalt": ((4, 2), "日本の郊外の生活道路（県道）の、使い込まれた中くらいの灰色のアスファルト舗装。細かい骨材の粒が一様に広がり、ごく弱い色むら。"
                "補修のつぎはぎ・ひび・タイヤの跡・白線・マンホールなど、目立つ形や線は一切入れない（何度並べても繰り返しが分からない一様な面）。横長 2:1。", 1.5),
    "lawn": ((2, 2), "日本の家の庭の、少し手入れの甘い芝生を真上から。緑の芝に、ところどころ黄色く枯れた所と小さな雑草（クローバー・タンポポの葉）、土が少し見える。正方形。", 2.5),
    "garden_soil": ((2, 2), "日本の家の庭や空き地の、乾いた茶色の土を真上から。小石、砂利の粒、まばらな雑草の芽、足跡のくぼみ。正方形。", 3.0),
    "driveway": ((2, 2), "日本の家の駐車場の土間コンクリートを真上から。刷毛引きの細い筋が一方向に入った明るい灰色のコンクリート、タイヤの黒い跡、雨のしみ、"
                 "画像の端に細い伸縮目地の線が 1 本。正方形。", 1.5),
    "gravel": ((2, 2), "日本の家の駐車場や空き地の砂利敷きを真上から。灰色と白の砕石（2〜4 cm）がすき間なく敷かれ、ところどころ土と小さな雑草。正方形。", 4.0),
    "sidewalk_town": ((2, 2), "日本の郊外の歩道の、車道より少し明るい色あせた灰色のアスファルト舗装を真上から。細かい骨材、薄いしみ、落ち葉が 1〜2 枚。正方形。", 1.5),
}


def request(name):
    size, prompt, _ = TEX[name]
    p = (f"画像生成ツールで、ゲーム用のテクスチャ画像を1枚作って、このフォルダに {name}.png という名前で保存してください。"
         f"内容：{prompt} 実寸で横{size[0]}m×縦{size[1]}mの範囲。{COMMON}保存したらファイル名と画像の大きさだけ答えてください。")
    SRC.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(["codex", "exec", "--skip-git-repo-check", "--sandbox", "workspace-write", "-C", str(SRC), "--enable", "image_generation", p],
                       capture_output=True, timeout=1200, stdin=subprocess.DEVNULL)   # 入力をつなぐと codex はそれを読み終わるまで待つ
    ok = (SRC / f"{name}.png").exists()
    print(f"{name}: {'ok' if ok else 'FAILED'}", flush=True)
    if not ok:
        print(r.stdout.decode("utf-8", "replace")[-800:], r.stderr.decode("utf-8", "replace")[-400:])


def process(name):
    img = dress_tex.make_tileable(Image.open(SRC / f"{name}.png").convert("RGB"))
    w, h = img.size; img = img.resize((min(2048, dress_tex._pow2(w)), min(2048, dress_tex._pow2(h))), Image.LANCZOS)
    n = dress_tex.height_to_normal(np.asarray(img.convert("L"), float) / 255.0, TEX[name][2])
    img.save(DIR / f"{name}.png"); Image.fromarray(((n * 0.5 + 0.5) * 255).astype(np.uint8)).save(DIR / f"{name}_nm.png")


def main():
    names = sys.argv[1:] or list(TEX)
    todo = [n for n in names if not (SRC / f"{n}.png").exists()]
    with ThreadPoolExecutor(4) as ex:
        list(ex.map(request, todo))
    for n in names:
        if (SRC / f"{n}.png").exists():
            process(n); print("processed", n, Image.open(DIR / f"{n}.png").size)


if __name__ == "__main__":
    main()
