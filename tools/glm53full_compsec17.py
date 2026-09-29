#!/usr/bin/env python3
"""GLM-5.3 Full COMPSEC-17 through sparkpipe_model_batch, with the publisher chat template.

render (needs jinja2 and tokenizers; run it where they are installed, e.g. the rtx5090):
  renders every COMPSEC case through the checkpoint's chat_template.jinja
  (messages=[user], add_generation_prompt=True) and encodes the text with the
  checkpoint tokenizer without extra special tokens. --thinking off closes the
  opened <think> with </think>, the template's own rendering of an assistant
  turn without reasoning.
batch: writes sparkpipe_model_batch files, at most --request-capacity requests each.
grade: reads the batch NDJSON outputs and grades them with the ds4_eval rule.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("glm5_next_compsec17", TOOLS / "glm5_next_compsec17.py")
compsec = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(compsec)

FORMAT = "sparkpipe-glm53full-compsec17-prompts-v1"
THINK_CLOSE = "</think>"
EOS_TOKEN_IDS = [154820, 154827, 154829]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compsec_cases(fixture: dict) -> list:
    cases = sorted((c for c in fixture["cases"] if c["id"] in compsec.COMPSEC_IDS), key=lambda c: c["id"])
    if len(cases) != 17:
        raise SystemExit(f"expected 17 COMPSEC cases, found {len(cases)}")
    return cases


def render(args) -> int:
    from jinja2.sandbox import ImmutableSandboxedEnvironment
    from tokenizers import Tokenizer
    template_path, tokenizer_path, fixture_path = Path(args.template), Path(args.tokenizer), Path(args.fixture)
    environment = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"])
    environment.filters["tojson"] = lambda value, ensure_ascii=False, indent=None: json.dumps(value, ensure_ascii=ensure_ascii, indent=indent)
    template = environment.from_string(template_path.read_text())
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    question_decoder = compsec.load_decoder(tokenizer_path)
    fixture = json.loads(fixture_path.read_text())
    close_id = tokenizer.token_to_id(THINK_CLOSE)
    if close_id is None:
        raise SystemExit(f"{THINK_CLOSE} is not a tokenizer token")
    cases = []
    for case in compsec_cases(fixture):
        question = question_decoder(case["prompt_token_ids"])
        text = template.render(messages=[{"role": "user", "content": question}], add_generation_prompt=True)
        ids = tokenizer.encode(text, add_special_tokens=False).ids
        if args.thinking == "off":
            ids = ids + [close_id]
        cases.append({"id": case["id"], "answer": case["answer"], "prompt_token_ids": ids,
                      "rendered_sha256": hashlib.sha256(text.encode()).hexdigest()})
    document = {"format": FORMAT, "thinking": args.thinking, "template_sha256": sha256_file(template_path),
                "tokenizer_sha256": sha256_file(tokenizer_path), "fixture": fixture_path.name,
                "fixture_sha256": sha256_file(fixture_path), "eos_token_ids": EOS_TOKEN_IDS, "cases": cases}
    Path(args.out).write_text(json.dumps(document, indent=1) + "\n")
    print(f"rendered {len(cases)} cases, thinking {args.thinking}, prompt tokens {min(len(c['prompt_token_ids']) for c in cases)}-{max(len(c['prompt_token_ids']) for c in cases)}")
    return 0


def batch_files(prompts: dict, max_tokens: int, capacity: int, max_context: int, ids=None) -> list:
    cases = [c for c in prompts["cases"] if ids is None or c["id"] in ids]
    if ids is not None and len(cases) != len(ids):
        raise SystemExit(f"unknown case ids: {sorted(set(ids) - {c['id'] for c in cases})}")
    for case in cases:
        if len(case["prompt_token_ids"]) + max_tokens > max_context:
            raise SystemExit(f"{case['id']}: {len(case['prompt_token_ids'])} prompt + {max_tokens} new tokens exceed max_context_tokens {max_context}")
    files = []
    for start in range(0, len(cases), capacity):
        chunk = cases[start:start + capacity]
        files.append({"schema_version": 1, "connect_timeout_ms": 20000, "request_capacity": capacity,
                      "max_context_tokens": max_context, "max_prefill_rows_per_submission": capacity,
                      "maximum_messages_per_rank_per_progress": 8, "maximum_new_submissions_per_progress": capacity,
                      "stop_token_ids": [],
                      "requests": [{"request_id": start + index + 1, "sequence_id": start + index + 1, "priority": 0,
                                    "output_token_budget": max_tokens, "prompt_token_ids": case["prompt_token_ids"]}
                                   for index, case in enumerate(chunk)]})
    return files


def batch(args) -> int:
    prompts = json.loads(Path(args.prompts).read_text())
    if prompts.get("format") != FORMAT:
        raise SystemExit(f"{args.prompts} is not {FORMAT}")
    ids = args.ids.split(",") if args.ids else None
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for index, document in enumerate(batch_files(prompts, args.max_tokens, args.request_capacity, args.max_context_tokens, ids)):
        path = out / f"batch-{index:02d}.json"
        path.write_text(json.dumps(document) + "\n")
        print(f"{path} requests={len(document['requests'])}")
    return 0


def output_decoder(tokenizer_path: Path):
    document = json.loads(tokenizer_path.read_text())
    byte_decoder = compsec.load_decoder(tokenizer_path)
    added = {int(token["id"]): token["content"] for token in document.get("added_tokens", [])}

    def decode(ids) -> str:
        pieces, run = [], []
        for token in ids:
            if int(token) in added:
                if run:
                    pieces.append(byte_decoder(run))
                    run = []
                pieces.append(added[int(token)])
            else:
                run.append(int(token))
        if run:
            pieces.append(byte_decoder(run))
        return "".join(pieces)

    return decode


def collect_tokens(event_paths) -> dict:
    tokens = {}
    for path in event_paths:
        for line in Path(path).read_text().splitlines():
            if not line.startswith("{"):
                continue
            event = json.loads(line)
            if event.get("event") == "token":
                tokens.setdefault(int(event["request_id"]), []).append(int(event["token_id"]))
            elif event.get("event") == "error":
                raise SystemExit(f"{path}: error event {event}")
    return tokens


def grade_cases(prompts: dict, tokens: dict, decode, max_tokens: int) -> dict:
    results = []
    for index, case in enumerate(prompts["cases"], 1):
        if index not in tokens:
            continue
        generated = tokens[index]
        stop = [t for t in generated if t in prompts["eos_token_ids"]]
        text = decode([t for t in generated if t not in prompts["eos_token_ids"]])
        passed, extracted = compsec.grade(text, case["answer"])
        results.append({"id": case["id"], "expected": case["answer"], "extracted": extracted, "passed": passed,
                        "generated_token_count": len(generated), "stopped_on_eos": bool(stop),
                        "hit_budget": not stop and len(generated) >= max_tokens,
                        "output_sha256": hashlib.sha256(text.encode()).hexdigest(), "text": text})
    return {"format": "sparkpipe-glm53full-compsec17-grade-v1", "thinking": prompts["thinking"],
            "template_sha256": prompts["template_sha256"], "completed": len(results),
            "passed": sum(1 for r in results if r["passed"]), "results": results}


def grade(args) -> int:
    prompts = json.loads(Path(args.prompts).read_text())
    summary = grade_cases(prompts, collect_tokens(args.events), output_decoder(Path(args.tokenizer)), args.max_tokens)
    Path(args.out).write_text(json.dumps(summary, indent=1) + "\n")
    for r in summary["results"]:
        print(f"{r['id']} {'PASS' if r['passed'] else 'FAIL'} expected={r['expected']!r} got={r['extracted'][:40]!r} tokens={r['generated_token_count']}{' budget' if r['hit_budget'] else ''}")
    print(f"COMPSEC-17 RESULT: {summary['passed']}/{summary['completed']} (thinking {summary['thinking']}, publisher template {summary['template_sha256'][:8]})")
    return 0 if summary["completed"] == 17 and summary["passed"] >= args.pass_threshold else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("render")
    p.add_argument("--template", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--fixture", required=True)
    p.add_argument("--thinking", required=True, choices=("off", "on"))
    p.add_argument("--out", required=True)
    p = commands.add_parser("batch")
    p.add_argument("--prompts", required=True)
    p.add_argument("--max-tokens", type=int, required=True)
    p.add_argument("--request-capacity", type=int, required=True)
    p.add_argument("--max-context-tokens", type=int, required=True)
    p.add_argument("--ids")
    p.add_argument("--out", required=True)
    p = commands.add_parser("grade")
    p.add_argument("--prompts", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--max-tokens", type=int, required=True)
    p.add_argument("--pass-threshold", type=int, default=14)
    p.add_argument("--out", required=True)
    p.add_argument("events", nargs="+")
    args = parser.parse_args()
    return {"render": render, "batch": batch, "grade": grade}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
