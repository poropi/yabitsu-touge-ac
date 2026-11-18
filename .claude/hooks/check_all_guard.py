"""PreToolUse（Bash）：比較画像を open で見せる前に、一度止めて「全て確認したか？」を問い直す。

2026-11-01、直した所だけを確かめて画像を見せ、ほかの所が悪化していた（木を 4 本外して坑口の上の土が見えた）。
ユーザー「少なくとも、今回の修正後の確認にて、確認終了後に『全て確認したか？』を問い直すことはしてほしい」。

同じ画像の 1 回目の open は止めて（exit 2）確認項目を返す。確認項目への答えを報告に書いたうえで同じ open をもう一度実行したら通す。
"""
import json
import re
import sys
import time
from pathlib import Path

STATE = Path(__file__).with_name(".check_all_state.json")
IMG = re.compile(r"[^\s'\"]+\.(?:jpg|jpeg|png)", re.I)
OPEN = re.compile(r"(?:^|[;&|(\s])open\s")

QUESTION = """【全て確認したか？】比較画像を見せる前に、次の 4 つに答えよ（答えはユーザーへの報告文に書く）：
1. 直す前と直した後の画像を、直した所だけでなく画面全体で見比べ、変わった所を全部挙げたか。
2. 変わった所それぞれが「直した所」か「意図しない変化（悪化を含む）」か、言えるか。意図しない変化があるのに「直った」と書いていないか。
3. ストリートビューの写真と照らし合わせて、写っている物を 1 つずつ確かめたか（部品の正体を思い込みで決めていないか）。
4. 新しく見えるようになった物・見えなくなった物はないか（外した物・足した物が、ほかの物を隠したり見せたりしていないか）。
答えたら、同じ open をもう一度実行すること。"""


def main():
    try:
        data = json.load(sys.stdin)
    except Exception:
        return 0
    cmd = (data.get("tool_input") or {}).get("command") or ""
    if not OPEN.search(cmd):
        return 0
    imgs = sorted({m for m in IMG.findall(cmd) if "compare" in m or "scratchpad" in m})
    if not imgs:
        return 0
    try:
        state = json.loads(STATE.read_text(encoding="utf-8"))
    except Exception:
        state = {}
    now = time.time()
    state = {k: v for k, v in state.items() if now - v < 3600}   # 1 時間で忘れる
    key = (data.get("session_id") or "") + "|" + "|".join(imgs)
    if key in state:                                             # 2 回目：問い直しに答えたあと
        del state[key]
        STATE.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
        return 0
    state[key] = now
    STATE.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    print(QUESTION, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
