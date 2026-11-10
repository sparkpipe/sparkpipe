#!/usr/bin/env python3
"""Build the Metal gate prompts for Qwen3.8-27B from files in this tree.

Each prompt is one chat turn, <|im_start|>user ... <|im_end|> followed by the
assistant header with an empty think block, in the same token framing as
qualification/qwen38_27b/bench_prompts_tp1.json. The user text is an
instruction followed by a repository file, tokenized with the checkpoint's
tokenizer.json by build/sparkpipe_tokenize_prompt. The numerics set cuts each
file to a target length; the speculation set holds a whole-file rewrite and a
prose request with no file; the agent set holds code-editing requests on whole
files (rename, feature, diff, test), framed the same way. The texts set is the
raw files, no chat framing, for teacher-forced scoring.

The gsm8k set is the first --count questions of a GSM8K test.jsonl as drafter
vendors evaluate them: the chat_template.json turns, a system turn with the
checkpoint template's xhigh reasoning instruction, the question with the
z-lab/dflash benchmark suffix, and the assistant turn opened with <think>, the
checkpoint template's thinking generation prompt (chat_template.json has no
thinking variant). Each prompt keeps the reference answer. The output records
every source path, its sha256 and the token counts, so the same prompts can be
rebuilt from the same revision.

  qwen38_27b_gate_prompts.py --tokenizer TOKENIZER_JSON --output PROMPTS_JSON [--set numerics|speculation|agent|texts]
  qwen38_27b_gate_prompts.py --tokenizer TOKENIZER_JSON --output PROMPTS_JSON --set gsm8k --dataset TEST_JSONL [--count 100]
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
CHAT_TEMPLATE = ROOT / "model-families" / "qwen38" / "chat_template.json"
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
AGENT = (
    ("agent-rename", "tools/spec_offline/tapstream.py", "Rename the function read_hello to parse_hello in this Python file, update every "
                                                        "caller in it, and change nothing else. Output only the complete updated file.\n\n", 0),
    ("agent-feature", "tools/spec_offline/streams.py", "Add a method prefix(self, count) to the Stream dataclass that returns a new Stream "
                                                       "holding the first count tokens, with the prompt length clipped to count. Output only "
                                                       "the complete updated file.\n\n", 0),
    ("agent-diff", "src/spark_tap_stream.c", "Make SparkTapStreamFingerprint also mix tap_rank and tp_degree into the hash, after "
                                             "entry_count. Reply with a unified diff of the file only.\n\n", 0),
    ("agent-test", "tools/context_retention.py", "Write a Python unittest module for this file that covers expiry at the retention "
                                                 "boundary and eviction down to one context. Output only the test file.\n\n", 0),
)
TEXTS = ("docs/SPECULATION_TAPS.md", "tools/qwen38_27b_reference.c", "tools/spec_offline/block_drafter.py", "README.md")
TEXT_TOKENS = 3000
XHIGH = ("Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider "
         "plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.")
GSM8K_SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."
THINK_OPEN = "<think>\n"


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


def gsm8k_prompts(tokenizer, dataset, count):
    template = json.loads(CHAT_TEMPLATE.read_text(encoding="utf-8"))
    with open(dataset, encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()][:count]
    prompts = []
    for index, row in enumerate(rows):
        text = (template["system"] + XHIGH + template["turn_suffix"] + template["user"] + row["question"] + GSM8K_SUFFIX + template["turn_suffix"] +
                template["assistant"] + THINK_OPEN)
        tokens = tokenize(tokenizer, text)
        prompts.append({"name": f"gsm8k-{index:04d}", "source": str(dataset), "answer": row["answer"].split("####")[-1].strip().replace(",", ""),
                        "tokens": len(tokens), "prompt_token_ids": tokens})
    print(f"gsm8k: {len(prompts)} prompts, {sum(p['tokens'] for p in prompts)} tokens from {dataset}")
    return prompts


def text_prompts(tokenizer):
    prompts = []
    for source in TEXTS:
        path = ROOT / source
        tokens = tokenize(tokenizer, path.read_text(encoding="utf-8"))[:TEXT_TOKENS]
        prompts.append({"name": source, "source": source, "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "tokens": len(tokens),
                        "prompt_token_ids": tokens})
        print(f"{source}: {len(tokens)} tokens")
    return prompts


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--set", choices=("numerics", "speculation", "agent", "texts", "gsm8k"), default="numerics")
    parser.add_argument("--dataset", type=Path, help="GSM8K test.jsonl, for --set gsm8k")
    parser.add_argument("--count", type=int, default=100, help="questions, for --set gsm8k")
    arguments = parser.parse_args()
    if not TOKENIZER_TOOL.is_file():
        raise SystemExit(f"build {TOKENIZER_TOOL.relative_to(ROOT)} first (make {TOKENIZER_TOOL.relative_to(ROOT)})")
    if arguments.set == "gsm8k" and arguments.dataset is None:
        parser.error("--set gsm8k needs --dataset")
    prompts = gsm8k_prompts(arguments.tokenizer, arguments.dataset, arguments.count) if arguments.set == "gsm8k" else \
        text_prompts(arguments.tokenizer) if arguments.set == "texts" else []
    for name, source, instruction, target in {"numerics": PROMPTS, "speculation": SPECULATION, "agent": AGENT}.get(arguments.set, ()):
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
