import json
import subprocess
import sys
from pathlib import Path

HOOK = Path(__file__).resolve().parents[1] / ".claude" / "hooks" / "japanese_check.py"

EN = ("All three spots in your screenshots are fixed. I've opened a sheet showing your screenshot, "
      "before and after side by side for each spot. The tests all pass and the changes are committed.")
JA = ("スクショの3か所はすべて直りました。直す前と直したあとを並べた画像を画面に開いています。"
      "画像：`C:\\Users\\porop\\Documents\\yabitsu\\.superpowers\\sdd\\seams_before_after.jpg`、"
      "コマンド `uv run python -m scripts.track_06_terrain` と scripts/track_build_manifest.py を直しました。"
      "The tests pass（テストは 334 件すべて通りました）。")


def _run(payload):
    r = subprocess.run([sys.executable, str(HOOK)], input=json.dumps(payload).encode("utf-8"), capture_output=True)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout.decode("utf-8")) if r.stdout.strip() else None


def _transcript(tmp_path, user, *assistant):
    lines = [{"type": "user", "message": {"role": "user", "content": user}}]
    lines += [{"type": "assistant", "message": {"role": "assistant", "content": [{"type": "text", "text": t}]}} for t in assistant]
    p = tmp_path / "t.jsonl"
    p.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in lines), encoding="utf-8")
    return str(p)


def test_english_reply_is_sent_back_to_be_rewritten_in_japanese(tmp_path):
    out = _run({"transcript_path": _transcript(tmp_path, "直して", EN), "stop_hook_active": False})
    assert out["decision"] == "block" and "日本語" in out["reason"]


def test_japanese_reply_with_paths_and_code_passes(tmp_path):
    assert _run({"transcript_path": _transcript(tmp_path, "直して", JA), "stop_hook_active": False}) is None


def test_short_reply_and_second_round_are_left_alone(tmp_path):
    assert _run({"transcript_path": _transcript(tmp_path, "直して", "OK, done."), "stop_hook_active": False}) is None
    assert _run({"transcript_path": _transcript(tmp_path, "直して", EN), "stop_hook_active": True}) is None   # 止まらなくならない


def test_last_assistant_message_field_is_used_when_given():
    assert _run({"last_assistant_message": EN})["decision"] == "block"
    assert _run({"last_assistant_message": JA}) is None
