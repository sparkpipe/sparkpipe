#!/usr/bin/env python3
"""COMPSEC-17 quality gate for any served model (ds4_eval protocol).

Same cases, decoding, request and grading as tools/glm5_next_compsec17.py;
the chat template is an explicit argument. The fixture's prompt_token_ids
are GLM-tokenized, so --fixture-tokenizer must be the GLM tokenizer.json
that produced them; the endpoint tokenizes the rendered prompt with the
served model's own tokenizer.

Templates (thinking off / on), rendered from each model's reference:
  glm            [gMASK]<sop><|user|>\\n{q}<|assistant|>\\n<think></think>\\n | <think>
  deepseek_v41   encoding/encoding.py at dba1be0a, thinking_mode chat |
                 thinking (reasoning effort 75)

usage:
  compsec17_eval.py --chat-template deepseek_v41 --thinking off \\
      --endpoint http://127.0.0.1:8434 \\
      --fixture qualification/ds4_eval/quality-fixtures-glm5.3-flash.json \\
      --fixture-tokenizer <glm runtime>/tokenizer/tokenizer.json \\
      --out qualification/dsv41_flash/runs/<run-id>/compsec17
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from glm5_next_compsec17 import COMPSEC_IDS, call, grade, load_decoder

DEEPSEEK_V41_EFFORT = ("<｜System｜>Reasoning Effort: 75 (range 1-100, the higher the "
                       "value, the more thorough the reasoning)\n\n")
TEMPLATES = {
    "glm": {
        "off": ("[gMASK]<sop><|user|>\n", "<|assistant|>\n<think></think>\n"),
        "on": ("[gMASK]<sop><|user|>\n", "<|assistant|>\n<think>"),
    },
    "deepseek_v41": {
        "off": ("<｜begin▁of▁sentence｜><｜User｜>", "<｜Assistant｜></think>"),
        "on": ("<｜begin▁of▁sentence｜>" + DEEPSEEK_V41_EFFORT + "<｜User｜>",
               "<｜Assistant｜><think>"),
    },
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--chat-template", required=True, choices=sorted(TEMPLATES))
    ap.add_argument("--thinking", required=True, choices=["off", "on"])
    ap.add_argument("--fixture", required=True)
    ap.add_argument("--fixture-tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, required=True)
    ap.add_argument("--pass-threshold", type=int, required=True)
    ap.add_argument("--concurrency", type=int, required=True)
    ap.add_argument("--temperature", type=float, required=True)
    ap.add_argument("--timeout", type=int, required=True)
    args = ap.parse_args()
    decode = load_decoder(Path(args.fixture_tokenizer))
    cases = sorted((c for c in json.loads(Path(args.fixture).read_text())["cases"] if c["id"] in COMPSEC_IDS),
                   key=lambda c: c["id"])
    if len(cases) != len(COMPSEC_IDS):
        print(f"FATAL: expected {len(COMPSEC_IDS)} COMPSEC cases, found {len(cases)}", file=sys.stderr)
        return 2
    head, tail = TEMPLATES[args.chat_template][args.thinking]
    prompts = [head + decode(c["prompt_token_ids"]) + tail for c in cases]
    out = Path(args.out)
    (out / "responses").mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        replies = list(pool.map(lambda p: call(args.endpoint, p, args.max_tokens, args.temperature, args.timeout), prompts))
    wall_s = time.monotonic() - started
    results = []
    for index, (case, prompt, reply) in enumerate(zip(cases, prompts, replies), 1):
        payload = reply["payload"]
        tokens = payload.get("tokens") or []
        text = (payload.get("choices") or [{}])[0].get("text") or ""
        passed, extracted = grade(text, case["answer"])
        record = {"index": index, "id": case["id"], "domain": case["domain"], "expected": case["answer"],
                  "extracted": extracted, "passed": passed, "elapsed_ms": round(reply["elapsed_s"] * 1000, 1),
                  "generated_token_count": len(tokens), "output_sha256": hashlib.sha256(text.encode()).hexdigest(),
                  "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "status": payload.get("status")}
        results.append(record)
        (out / "responses" / f"{index:03d}-{case['id']}.json").write_text(
            json.dumps({"prompt": prompt, "response": payload, "decoded_text": text, "grade": record}, indent=1, ensure_ascii=False))
        print(f"[{index:2d}/{len(cases)}] {case['id']} {'PASS' if passed else 'FAIL'} expected={case['answer']!r} "
              f"got={extracted[:60]!r} ({reply['elapsed_s']:.1f}s, {len(tokens)} tok)", flush=True)
    passed_n = sum(1 for r in results if r["passed"])
    summary = {"format": "ds4-eval-compsec17-summary-v1", "generated_at": time.strftime("%FT%T%z"),
               "endpoint": args.endpoint, "fixture": Path(args.fixture).name,
               "fixture_sha256": hashlib.sha256(Path(args.fixture).read_bytes()).hexdigest(),
               "chat_template": args.chat_template, "thinking": args.thinking, "max_tokens": args.max_tokens,
               "temperature": args.temperature, "concurrency": args.concurrency, "wall_s": round(wall_s, 2),
               "passed": passed_n, "total": len(results), "pass_threshold": args.pass_threshold,
               "gate": "PASS" if passed_n >= args.pass_threshold else "FAIL"}
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    (out / "cases.json").write_text(json.dumps(results, indent=1) + "\n")
    print(f"COMPSEC-17 {passed_n}/{len(results)} gate {summary['gate']} (threshold {args.pass_threshold})")
    return 0 if passed_n >= args.pass_threshold else 1


if __name__ == "__main__":
    raise SystemExit(main())
