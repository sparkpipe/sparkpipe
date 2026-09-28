#!/usr/bin/env python3
"""COMPSEC-17 through sparkpipe_model_batch instead of an HTTP endpoint.

prepare: the 17 COMPSEC prompts in the served model's chat template, tokenized
         with its tokenizer.json, as one sparkpipe_model_batch file (greedy,
         --max-tokens each, the model's stop tokens).
grade:   decode the batch's token events per request and grade them with
         tools/compsec17_chat.py (both rules recorded, --grading sets passed).

The prompt text, template and grading are compsec17_chat.py's; only the
transport differs, so a batch run and an API run of the same engine grade the
same replies.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUEST_BASE = 76000


def load_chat():
    spec = importlib.util.spec_from_file_location("compsec17_chat", ROOT / "tools" / "compsec17_chat.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cases(chat, fixture: Path):
    data = json.loads(fixture.read_text())
    selected = sorted((c for c in data["cases"] if c["id"] in chat.COMPSEC_IDS), key=lambda c: c["id"])
    if len(selected) != 17:
        raise SystemExit(f"expected 17 COMPSEC cases, found {len(selected)}")
    return selected


def prepare(args, chat) -> int:
    import tokenizers
    model_tok = tokenizers.Tokenizer.from_file(args.model_tokenizer)
    decode = chat.load_decoder(Path(args.fixture_tokenizer))
    requests = []
    for index, case in enumerate(cases(chat, Path(args.fixture))):
        prompt = chat.build_prompt(args.template, decode(case["prompt_token_ids"]), args.thinking)
        ids = model_tok.encode(prompt, add_special_tokens=False).ids
        requests.append({"request_id": REQUEST_BASE + index, "sequence_id": REQUEST_BASE + index, "priority": 0,
                         "output_token_budget": args.max_tokens, "prompt_token_ids": ids})
    longest = max(len(r["prompt_token_ids"]) for r in requests) + args.max_tokens
    batch = {"schema_version": 1, "connect_timeout_ms": 30000, "request_capacity": len(requests),
             "max_context_tokens": longest, "max_prefill_rows_per_submission": 128,
             "maximum_messages_per_rank_per_progress": 8, "maximum_new_submissions_per_progress": 1,
             "stop_token_ids": [int(t) for t in args.stop_token_ids.split(",")], "requests": requests}
    Path(args.out).write_text(json.dumps(batch))
    print(json.dumps({"requests": len(requests), "max_context_tokens": longest, "max_tokens": args.max_tokens}))
    return 0


def grade_run(args, chat) -> int:
    import tokenizers
    model_tok = tokenizers.Tokenizer.from_file(args.model_tokenizer)
    selected = cases(chat, Path(args.fixture))
    tokens: dict[int, list[int]] = {}
    stops: dict[int, bool] = {}
    for line in Path(args.batch_stdout).read_text().splitlines():
        if not line.startswith("{"):
            continue
        event = json.loads(line)
        if event.get("event") == "token":
            tokens.setdefault(event["request_id"], []).append(event["token_id"])
            if event.get("stop_token"):
                stops[event["request_id"]] = True
    out = Path(args.out)
    (out / "responses").mkdir(parents=True, exist_ok=True)
    other = "ds4_eval" if args.grading == "final-answer-line" else "final-answer-line"
    results = []
    for index, case in enumerate(selected):
        rid = REQUEST_BASE + index
        ids = tokens.get(rid, [])
        text = model_tok.decode(ids, skip_special_tokens=False)
        passed, extracted = chat.grade(text, case["answer"], args.grading)
        other_passed, other_extracted = chat.grade(text, case["answer"], other)
        finish = "stop" if stops.get(rid) else ("length" if len(ids) >= args.max_tokens else "incomplete")
        rec = {"index": index + 1, "id": case["id"], "domain": case["domain"], "expected": case["answer"],
               "extracted": extracted, "passed": passed, "grading": args.grading,
               other + "_passed": other_passed, other + "_extracted": other_extracted,
               "finish_reason": finish, "generated_token_count": len(ids),
               "output_sha256": hashlib.sha256(text.encode()).hexdigest()}
        results.append(rec)
        (out / "responses" / f"{index + 1:03d}-{case['id']}.json").write_text(json.dumps(
            {"id": case["id"], "token_ids": ids, "decoded_text": text, "grade": rec}, indent=1))
        print(f"[{index + 1:2d}/17] {case['id']} {'PASS' if passed else 'FAIL'} expected={case['answer']!r} "
              f"got={extracted[:40]!r} {finish} {len(ids)} tok", flush=True)
    passed_n = sum(1 for r in results if r["passed"])
    (out / "summary.json").write_text(json.dumps({
        "format": "ds4-eval-compsec17-batch-summary-v1", "label": args.label,
        "parameters": {"max_tokens": args.max_tokens, "grading": args.grading, "temperature": 0.0},
        "completed": sum(1 for r in results if r["generated_token_count"] > 0), "passed": passed_n,
        "passed_other_rule": sum(1 for r in results if r[other + "_passed"]),
        "truncated": sum(1 for r in results if r["finish_reason"] == "length"),
        "results": results}, indent=1))
    chat.write_integrity(out, len(results))
    print(f"\nCOMPSEC-17 RESULT ({args.grading}): {passed_n}/17; {other}: "
          f"{sum(1 for r in results if r[other + '_passed'])}/17; truncated "
          f"{sum(1 for r in results if r['finish_reason'] == 'length')}")
    return 0


def main() -> int:
    chat = load_chat()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="action", required=True)
    for name in ("prepare", "grade"):
        p = sub.add_parser(name)
        p.add_argument("--model-tokenizer", required=True)
        p.add_argument("--fixture", default=str(ROOT / "qualification/ds4_eval/quality-fixtures-glm5.3-flash.json"))
        p.add_argument("--max-tokens", type=int, required=True)
        p.add_argument("--out", required=True)
    prep = sub.choices["prepare"]
    prep.add_argument("--template", required=True, choices=sorted(chat.TEMPLATES))
    prep.add_argument("--thinking", required=True, choices=("off", "on"))
    prep.add_argument("--fixture-tokenizer", default=str(ROOT / "qualification/ds4_eval/tokenizer/glm-5.3-flash-tokenizer.json"))
    prep.add_argument("--stop-token-ids", required=True)
    grade = sub.choices["grade"]
    grade.add_argument("--batch-stdout", required=True)
    grade.add_argument("--grading", required=True, choices=("ds4_eval", "final-answer-line"))
    grade.add_argument("--label", required=True)
    args = ap.parse_args()
    return prepare(args, chat) if args.action == "prepare" else grade_run(args, chat)


if __name__ == "__main__":
    sys.exit(main())
