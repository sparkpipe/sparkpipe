#!/usr/bin/env python3
"""Build the Metal gate prompts for Qwen3.8-27B from files in this tree.

Each prompt is one chat turn, <|im_start|>user ... <|im_end|> followed by the
assistant header with an empty think block, in the same token framing as
qualification/qwen38_27b/bench_prompts_tp1.json. The user text is an
instruction followed by a repository file, tokenized with the checkpoint's
tokenizer.json by build/sparkpipe_tokenize_prompt. The numerics set cuts each
file to a target length; the speculation set holds a whole-file rewrite and a
prose request with no file. The output records every source path, its sha256
and the token counts, so the same prompts can be rebuilt from the same
revision.

  qwen38_27b_gate_prompts.py --tokenizer TOKENIZER_JSON --output PROMPTS_JSON [--set numerics|speculation]
"""

import argparse
import hashlib
import html
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOKENIZER_TOOL = ROOT / "build" / "sparkpipe_tokenize_prompt"
OPEN = [248045, 846, 198]
CLOSE = [248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271]
PROMPTS = (
    ("design", "docs/INFERENCE_OS_DESIGN.md", "Summarize the design described below in five short bullet points.\n\n", 1100),
    ("code", "inference/kernels/linear_attn.cuh", "Review this CUDA header. Explain what it does and name one risk in it.\n\n", 1800),
    ("korean", "site/index-ko.html", "다음 글을 세 문장으로 요약해 주세요.\n\n", 700),
    ("rules", "sparkpipe_invariants.md", "Read these engineering rules, then say which two matter most for a new driver and why.\n\n", 2040),
)
SPECULATION = (
    ("rewrite", "tools/token_prompts.c", "Rewrite this C file in full. Rename SparkTokenPrompt to SparkPromptTokens everywhere and change nothing "
                                         "else. Output only the file.\n\n", 0),
    ("prose", None, "Write a short essay, about four hundred words, on why lighthouses were built where they were and what their keepers' "
                    "days were like.", 0),
)


def plain_text(path):
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".html":
        text = re.sub(r"(?s)<(script|style)[^>]*>.*?</\1>", " ", text)
        text = html.unescape(re.sub(r"<[^>]+>", " ", text))
        text = re.sub(r"[ \t\r\f\v]+", " ", text)
        text = re.sub(r"\n\s*\n+", "\n\n", text).strip()
    return text


def tokenize(tokenizer, text):
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt", delete=False) as handle:
        handle.write(text)
        name = handle.name
    result = subprocess.run([str(TOKENIZER_TOOL), "--tokenizer-json", str(tokenizer), "--text-file", name],
                            capture_output=True, text=True, check=False)
    Path(name).unlink()
    if result.returncode != 0:
        raise SystemExit(f"tokenizer failed: {result.stderr.strip()}")
    return [int(line) for line in result.stdout.split()]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--set", choices=("numerics", "speculation"), default="numerics")
    arguments = parser.parse_args()
    if not TOKENIZER_TOOL.is_file():
        raise SystemExit(f"build {TOKENIZER_TOOL.relative_to(ROOT)} first (make {TOKENIZER_TOOL.relative_to(ROOT)})")
    prompts = []
    for name, source, instruction, target in PROMPTS if arguments.set == "numerics" else SPECULATION:
        path = ROOT / source if source else None
        body = tokenize(arguments.tokenizer, instruction + (plain_text(path) if path else ""))
        keep = target - len(OPEN) - len(CLOSE) if target else len(body)
        if len(body) < keep:
            raise SystemExit(f"{source} yields {len(body)} tokens, fewer than the {keep} the prompt needs")
        tokens = OPEN + body[:keep] + CLOSE
        prompts.append({"name": name, "source": source, "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest() if path else None,
                        "instruction": instruction, "tokens": len(tokens), "prompt_token_ids": tokens})
        print(f"{name}: {len(tokens)} tokens from {source}")
    document = {"tokenizer_sha256": hashlib.sha256(arguments.tokenizer.read_bytes()).hexdigest(),
                "framing": {"open": OPEN, "close": CLOSE}, "prompts": prompts}
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(document, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
