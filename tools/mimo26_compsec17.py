#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "qualification" / "ds4_eval"))
from glm5_next_compsec17 import COMPSEC_IDS, load_decoder
from compare_runs import answer_matches, extract_answer

MIMO_USER = "<|im_start|>user\n"
MIMO_ASSISTANT = "<|im_end|><|im_start|>assistant\n"
MIMO_THINKING = {"off": "<think></think>", "on": ""}


def cases_of(fixture: Path) -> list:
    data = json.loads(fixture.read_text())
    cases = sorted((c for c in data["cases"] if c["id"] in COMPSEC_IDS), key=lambda c: c["id"])
    if len(cases) != 17:
        sys.exit(f"expected 17 COMPSEC cases, found {len(cases)}")
    return cases


def ids_of(case: dict) -> list:
    value = case["prompt_token_ids"]
    return json.loads(value) if isinstance(value, str) else value


def prepare(args) -> int:
    from tokenizers import Tokenizer
    decode = load_decoder(Path(args.fixture_tokenizer))
    tokenizer = Tokenizer.from_file(args.tokenizer)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    lines = []
    manifest = []
    for case in cases_of(Path(args.fixture)):
        text = MIMO_USER + decode(ids_of(case)) + MIMO_ASSISTANT + MIMO_THINKING[args.thinking]
        ids = tokenizer.encode(text, add_special_tokens=False).ids
        name = f"{case['id']}.prompt.i32"
        (out / name).write_bytes(struct.pack(f"<{len(ids)}i", *ids))
        lines.append(f"{args.remote_dir.rstrip('/')}/{name}")
        manifest.append({"id": case["id"], "prompt_tokens": len(ids), "prompt_sha256": hashlib.sha256(text.encode()).hexdigest()})
    (out / "requests.txt").write_text("\n".join(lines) + "\n")
    (out / "prepare.json").write_text(json.dumps({"thinking": args.thinking, "cases": manifest}, indent=1))
    print(f"prepared {len(lines)} prompts, tokens {min(m['prompt_tokens'] for m in manifest)}..{max(m['prompt_tokens'] for m in manifest)}")
    return 0


def grade(args) -> int:
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(args.tokenizer)
    outputs = Path(args.outputs)
    out = Path(args.out)
    (out / "responses").mkdir(parents=True, exist_ok=True)
    results = []
    for index, case in enumerate(cases_of(Path(args.fixture)), 1):
        raw = (outputs / f"{case['id']}.prompt.i32.out.i32").read_bytes()
        ids = list(struct.unpack(f"<{len(raw) // 4}i", raw))
        text = tokenizer.decode(ids, skip_special_tokens=True)
        extracted, _ = extract_answer({"source": "COMPSEC", "choices": [], "answer": case["answer"]}, text, "")
        passed = answer_matches({"source": "COMPSEC", "choices": [], "answer": case["answer"]}, extracted)
        record = {"index": index, "id": case["id"], "expected": case["answer"], "extracted": extracted,
                  "passed": passed, "generated_token_count": len(ids),
                  "output_sha256": hashlib.sha256(text.encode()).hexdigest()}
        results.append(record)
        (out / "responses" / f"{index:03d}-{case['id']}.json").write_text(json.dumps({"id": case["id"], "token_ids": ids, "decoded_text": text, "grade": record}, indent=1))
        print(f"[{index:2d}/17] {case['id']} {'PASS' if passed else 'FAIL'} expected={case['answer']!r} got={extracted[:60]!r} ({len(ids)} tok)")
    passed_count = sum(1 for r in results if r["passed"])
    summary = {"format": "ds4-eval-mimo26-compsec17-summary-v1", "generated_at": time.strftime("%FT%T%z"),
               "fixture": Path(args.fixture).name, "fixture_sha256": hashlib.sha256(Path(args.fixture).read_bytes()).hexdigest(),
               "parameters": {"chat_template": "mimo", "thinking": args.thinking, "max_tokens": args.max_tokens, "temperature": 0.0,
                              "decoding": "greedy, TP4 generate tool, B1, no speculation"},
               "grading_rule": "qualification/ds4_eval/compare_runs.py: last Answer: line after </think>; PASS iff its line set is a non-empty subset of the expected lines",
               "completed": len(results), "passed": passed_count, "results": results}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"COMPSEC-17 {passed_count}/17")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--fixture", required=True)
    p.add_argument("--fixture-tokenizer", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--thinking", choices=sorted(MIMO_THINKING), default="off")
    p.add_argument("--remote-dir", required=True)
    p.add_argument("--out", required=True)
    g = sub.add_parser("grade")
    g.add_argument("--fixture", required=True)
    g.add_argument("--tokenizer", required=True)
    g.add_argument("--outputs", required=True)
    g.add_argument("--thinking", choices=sorted(MIMO_THINKING), default="off")
    g.add_argument("--max-tokens", type=int, required=True)
    g.add_argument("--out", required=True)
    args = parser.parse_args()
    return prepare(args) if args.command == "prepare" else grade(args)


if __name__ == "__main__":
    sys.exit(main())
