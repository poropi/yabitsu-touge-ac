"""重い組み立て・描画・ゲームの撮影（1 回 数分〜20 分）を流す前に、このセッションで何回目かを数えて Claude に突きつける。
2026-10-25、護摩屋敷の水で小さな直しのたびに作り直しを回し、ユーザーに「作り直しを何回もやるような、
そんな効率悪いやり方は止めてね！」「頼むから覚えてくれよ！」と言われたことから。止めはしない（文脈に足すだけ）。"""
import json
import re
import sys
import tempfile
from pathlib import Path

HEAVY = re.compile(r"-m\s+scripts\.(track_06_terrain|track_build_manifest|dress_mountain|forest_build|sv_compare)\b"   # 実際に流すコマンドだけ
                   r"|\b(bash|sh)\s+\S*(build|rebuild_render|all_cycle_\w+|quick_cap\w*)\.sh\b"
                   r"|launch-cap\.ps1")

data = json.load(sys.stdin)
cmd = (data.get("tool_input") or {}).get("command", "")
if not HEAVY.search(cmd) or "--help" in cmd:   # 使い方を見るだけは数えない
    sys.exit(0)
f = Path(tempfile.gettempdir()) / f"yabitsu_heavy_{data.get('session_id', 'x')}.txt"
n = int(f.read_text()) + 1 if f.exists() else 1
f.write_text(str(n))
msg = (f"【重い処理 {n} 回目（このセッション）】流す前に確かめる：(1) 基準と比べた違いを全部書き出し、まとめて直したか。"
       "(2) その確認は、部分だけの軽い確認（export/compare/gomayashiki/work/quick_view.py・単体テスト）で済まないか。"
       "(3) Mac の比較描画とゲームの撮影を二重にしていないか。(4) ユーザーに「これで何回目・あと何回」を言ったか。"
       "（Knowledge/rules.md・mistakes.md 2026-10-25）")
print(json.dumps({"hookSpecificOutput": {"hookEventName": "PreToolUse", "additionalContext": msg}}, ensure_ascii=False))
