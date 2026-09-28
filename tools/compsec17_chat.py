#!/usr/bin/env python3
"""COMPSEC-17 quality gate for any served model, with that model's chat template.

The 17 COMPSEC cases (compsec-076..092) are stored pre-tokenized with the
fixture's tokenizer (qualification/ds4_eval/tokenizer/). Each question is
decoded with that tokenizer, wrapped in the served model's chat template,
tokenized with the served model's tokenizer.json (HF `tokenizers`) and sent
as prompt_token_ids to /v1/completions, temperature 0. Output tokens are
decoded with the served model's tokenizer and graded two ways:

  ds4_eval           qualification/ds4_eval/compare_runs.py, exactly as
                     tools/glm5_next_compsec17.py does for GLM (falls back to the
                     last integer anywhere in the reply when no Answer line exists)
  final-answer-line  only the reply's last non-empty line counts, and only when it
                     is an "Answer: <lines>" line (markdown emphasis allowed); a
                     truncated or unfinished reply grades "?"

--grading picks the rule that sets passed and the exit status; both are recorded.

Templates (thinking off / on):
  qwen  <|im_start|>user\\n{q}<|im_end|>\\n<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n
        (the Qwen3.8 chat_template.jinja with enable_thinking=false; on ends at <think>\\n)
  glm   [gMASK]<sop><|user|>\\n{q}<|assistant|>\\n<think></think>\\n

usage:
  compsec17_chat.py --endpoint http://127.0.0.1:8435 --template qwen --thinking off \\
      --model-tokenizer /mnt/model-warm/qwen3.8-27b-fp8/tokenizer.json \\
      --out qualification/qwen38_27b/runs/<run-id>/compsec17
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "qualification" / "ds4_eval"))
sys.path.insert(0, str(ROOT / "tools"))
from compare_runs import answer_matches, extract_answer, normalize_line_spec
from glm5_next_compsec17 import COMPSEC_IDS, load_decoder

FINAL_ANSWER_LINE = re.compile(r"^[\s*_#>`-]*answer[\s*_`]*:[\s*_`]*(.*)$", re.IGNORECASE)

TEMPLATES = {
    "qwen": ("<|im_start|>user\n", "<|im_end|>\n<|im_start|>assistant\n",
             {"off": "<think>\n\n</think>\n\n", "on": "<think>\n"}),
    "glm": ("[gMASK]<sop><|user|>\n", "<|assistant|>\n",
            {"off": "<think></think>\n", "on": "<think>"}),
}


def build_prompt(template: str, question: str, thinking: str) -> str:
    user, assistant, think = TEMPLATES[template]
    return user + question + assistant + think[thinking]


def call(endpoint: str, ids: list, max_tokens: int, temperature: float, timeout: int) -> dict:
    body = json.dumps({"prompt_token_ids": ids, "max_tokens": max_tokens,
                       "temperature": temperature}).encode()
    req = urllib.request.Request(endpoint.rstrip("/") + "/v1/completions", data=body,
                                 headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read())
    return {"elapsed_s": time.monotonic() - t0, "payload": payload}


def final_answer_line(text: str) -> str:
    surface = text.split("</think>", 1)[1] if "</think>" in text else text
    lines = [line for line in surface.strip().splitlines() if line.strip()]
    if not lines:
        return "?"
    match = FINAL_ANSWER_LINE.match(lines[-1])
    return normalize_line_spec(match.group(1)) if match else "?"


def grade(text: str, answer: str, grading: str) -> tuple:
    case = {"source": "COMPSEC", "choices": [], "answer": answer}
    if grading == "final-answer-line":
        extracted = final_answer_line(text)
    else:
        extracted, _ = extract_answer(case, text, "")
    return answer_matches(case, extracted), extracted


def write_integrity(out: Path, count: int) -> None:
    hashes, lines = {}, []
    for rel in sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()):
        if rel in ("INTEGRITY.json", "REPORT.md"):
            continue
        digest = hashlib.sha256((out / rel).read_bytes()).hexdigest()
        hashes[rel] = digest
        if rel.startswith("responses/"):
            lines.append(f"{digest}  {rel}\n")
    (out / "INTEGRITY.json").write_text(json.dumps({
        "format": "ds4-eval-run-integrity-v1", "file_sha256": hashes,
        "response_count": count,
        "response_stream_sha256": hashlib.sha256("".join(lines).encode()).hexdigest(),
        "response_stream_paths": "run-relative-posix"}, indent=1))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--template", required=True, choices=sorted(TEMPLATES))
    ap.add_argument("--thinking", required=True, choices=("off", "on"))
    ap.add_argument("--model-tokenizer", required=True, help="served model tokenizer.json")
    ap.add_argument("--fixture", default=str(ROOT / "qualification/ds4_eval/quality-fixtures-glm5.3-flash.json"))
    ap.add_argument("--fixture-tokenizer", default=str(ROOT / "qualification/ds4_eval/tokenizer/glm-5.3-flash-tokenizer.json"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--grading", required=True, choices=("ds4_eval", "final-answer-line"))
    ap.add_argument("--pass-threshold", type=int, default=14)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--timeout", type=int, default=900)
    args = ap.parse_args()
    import tokenizers
    model_tok = tokenizers.Tokenizer.from_file(args.model_tokenizer)
    fixture_decode = load_decoder(Path(args.fixture_tokenizer))
    fixture = json.loads(Path(args.fixture).read_text())
    cases = sorted((c for c in fixture["cases"] if c["id"] in COMPSEC_IDS), key=lambda c: c["id"])
    if len(cases) != 17:
        print(f"FATAL: expected 17 COMPSEC cases, found {len(cases)}", file=sys.stderr)
        return 2
    out = Path(args.out)
    (out / "responses").mkdir(parents=True, exist_ok=True)
    prompts = [build_prompt(args.template, fixture_decode(c["prompt_token_ids"]), args.thinking) for c in cases]
    prompt_ids = [model_tok.encode(p, add_special_tokens=False).ids for p in prompts]
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
        replies = list(pool.map(lambda ids: call(args.endpoint, ids, args.max_tokens, args.temperature, args.timeout), prompt_ids))
    wall_s = time.monotonic() - started
    results = []
    other = "ds4_eval" if args.grading == "final-answer-line" else "final-answer-line"
    for i, (c, prompt, ids, r) in enumerate(zip(cases, prompts, prompt_ids, replies), 1):
        payload = r["payload"]
        tokens = payload.get("tokens") or []
        choices = payload.get("choices") or [{}]
        text = choices[0].get("text") if choices[0].get("text") is not None else model_tok.decode(tokens, skip_special_tokens=False)
        passed, extracted = grade(text, c["answer"], args.grading)
        other_passed, other_extracted = grade(text, c["answer"], other)
        rec = {"index": i, "id": c["id"], "source": "COMPSEC", "domain": c["domain"],
               "expected": c["answer"], "extracted": extracted, "passed": passed,
               "grading": args.grading, other + "_passed": other_passed, other + "_extracted": other_extracted,
               "finish_reason": choices[0].get("finish_reason"),
               "elapsed_ms": round(r["elapsed_s"] * 1000, 1),
               "output_sha256": hashlib.sha256(text.encode()).hexdigest(),
               "prompt_token_count": len(ids), "generated_token_count": len(tokens),
               "status": payload.get("status")}
        results.append(rec)
        (out / "responses" / f"{i:03d}-{c['id']}.json").write_text(json.dumps({
            "id": c["id"],
            "request": {"endpoint": args.endpoint + "/v1/completions", "template": args.template,
                        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                        "prompt_token_ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                        "thinking": args.thinking, "max_tokens": args.max_tokens,
                        "temperature": args.temperature},
            "response": payload, "decoded_text": text, "grade": rec}, indent=1))
        print(f"[{i:2d}/17] {c['id']} {c['domain'][:24]:24s} {'PASS' if passed else 'FAIL'} "
              f"expected={c['answer']!r} got={extracted[:60]!r} ({r['elapsed_s']:.1f}s, {len(tokens)} tok)", flush=True)
    passed_n = sum(1 for r in results if r["passed"])
    (out / "summary.json").write_text(json.dumps({
        "format": "ds4-eval-compsec17-chat-summary-v1", "generated_at": time.strftime("%FT%T%z"),
        "endpoint": args.endpoint, "fixture": Path(args.fixture).name,
        "fixture_sha256": hashlib.sha256(Path(args.fixture).read_bytes()).hexdigest(),
        "model_tokenizer_sha256": hashlib.sha256(Path(args.model_tokenizer).read_bytes()).hexdigest(),
        "wall_s": round(wall_s, 2),
        "parameters": {"max_tokens": args.max_tokens, "temperature": args.temperature,
                       "chat_template": args.template, "thinking": args.thinking,
                       "concurrency": args.concurrency, "pass_threshold": args.pass_threshold,
                       "grading": args.grading},
        "grading_rule": ("last non-empty line must be an Answer line, answer_matches on its line spec"
                         if args.grading == "final-answer-line" else
                         "qualification/ds4_eval/compare_runs.py extract_answer + answer_matches"),
        "completed": len(results), "passed": passed_n,
        "passed_other_rule": sum(1 for r in results if r[other + "_passed"]),
        "truncated": sum(1 for r in results if r["finish_reason"] == "length"),
        "results": results}, indent=1))
    (out / "cases.json").write_text(json.dumps({"format": "ds4-eval-compsec17-chat-cases-v1", "cases": cases}, indent=1))
    write_integrity(out, len(results))
    print(f"\nCOMPSEC-17 RESULT ({args.grading}): {passed_n}/17")
    return 0 if passed_n >= args.pass_threshold else 1


if __name__ == "__main__":
    sys.exit(main())
