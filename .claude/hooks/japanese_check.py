"""Stop hook：Claude の最後の返事がほぼ英語なら、終わらせずに「日本語で書き直して」と差し戻す。

判定：コード（``` の中・`...`）、URL、ファイルのパスを除いた本文で、英字の数と日本語の文字（ひらがな・カタカナ・漢字）の
数を比べる。英字が MIN_LETTERS 以上あり、日本語の文字が英字の JA_RATIO 未満なら「英語」とみなす。
差し戻しは1回だけ（stop_hook_active のときは何もしない）なので、止まらなくなることはない。"""
import json
import re
import sys

MIN_LETTERS = 80     # これより短い返事（「OK」など）は見ない
JA_RATIO = 0.25      # 日本語の文字が英字のこの割合より少なければ英語

JA = re.compile(r"[぀-ヿ㐀-鿿ｦ-ﾟ]")
LATIN = re.compile(r"[A-Za-z]")
STRIP = [re.compile(p, re.S) for p in (r"```.*?```", r"`[^`\n]*`", r"https?://\S+", r"[A-Za-z]:[\\/][^\s)」]*", r"(?:\.{0,2}/)?(?:[\w.-]+/)+[\w.-]+")]


def is_english(text):
    for p in STRIP:
        text = p.sub(" ", text)
    latin = len(LATIN.findall(text)); ja = len(JA.findall(text))
    return latin >= MIN_LETTERS and ja < JA_RATIO * latin


def last_assistant_text(transcript_path):
    """会話の記録（jsonl）から、最後のユーザーの発言より後の、Claude の文章をつなげて返す。"""
    texts = []
    with open(transcript_path, encoding="utf-8-sig", errors="replace") as f:
        for line in f:
            try:
                e = json.loads(line)
            except ValueError:
                continue
            msg = e.get("message") or {}
            content = msg.get("content")
            if e.get("type") == "user" and (isinstance(content, str) or any(
                    isinstance(c, dict) and c.get("type") == "text" for c in (content or []))):
                texts = []          # ユーザーの発言（道具の結果ではない）が来たら、そこから数え直す
            elif e.get("type") == "assistant" and isinstance(content, list):
                texts += [c.get("text", "") for c in content if isinstance(c, dict) and c.get("type") == "text"]
    return "\n".join(texts)


def main():
    data = json.loads(sys.stdin.buffer.read().decode("utf-8-sig") or "{}")
    if not isinstance(data, dict) or data.get("stop_hook_active"):
        return
    text = data.get("last_assistant_message")
    if not isinstance(text, str) or not text:
        text = last_assistant_text(data["transcript_path"]) if isinstance(data.get("transcript_path"), str) else ""
    if text and is_english(text):
        out = {"decision": "block",
               "reason": "最後の返事が英語になっています。同じ内容を、やさしい日本語で書き直してください（コード・ファイル名・コマンドはそのままで構いません）。"}
        sys.stdout.buffer.write(json.dumps(out, ensure_ascii=False).encode("utf-8"))


if __name__ == "__main__":
    try:
        main()
    except Exception:   # おかしな入力でも、会話を止めない（何もしない）
        pass
