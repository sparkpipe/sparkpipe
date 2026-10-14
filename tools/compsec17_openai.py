#!/usr/bin/env python3
"""COMPSEC-17 quality gate through an OpenAI-compatible chat endpoint (the UI path).

The 17 COMPSEC cases (compsec-076..092) are stored pre-tokenized with the
fixture's tokenizer (qualification/ds4_eval/tokenizer/). Each question is
decoded with that tokenizer and sent as one user message to
/v1/chat/completions of --endpoint (LiteLLM :4000 or the chat layer :8433)
for --model, temperature 0. The serving stack renders the model's publisher
chat template and splits reasoning from content. The graded text is
"<think>" + reasoning_content + "</think>" + content when the reply carries
reasoning, else content, so the ds4_eval rule (last Answer: line after
</think>, qualification/ds4_eval/compare_runs.py) applies unchanged.

usage:
  compsec17_openai.py --endpoint http://127.0.0.1:4000 --model gemma-4-31b \\
      --reasoning-effort minimal --out qualification/gemma4_31b/compsec17-<run-id>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from compsec17_chat import grade, write_integrity
from glm5_next_compsec17 import COMPSEC_IDS, load_decoder


class RunError(Exception):
    pass


def request(url: str, body: dict | None, timeout: int) -> dict:
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"},
                                 method="GET" if body is None else "POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as error:
        with error:
            detail = error.read().decode("utf-8", "replace")
        raise RunError(f"HTTP {error.code} from {url}: {detail[:400]}") from error
    except (OSError, ValueError) as error:
        raise RunError(f"{url}: {error}") from error


def require_model(endpoint: str, model: str, timeout: int) -> list:
    listing = request(endpoint.rstrip("/") + "/v1/models", None, timeout)
    ids = sorted(entry.get("id") for entry in listing.get("data", []) if isinstance(entry, dict))
    if model not in ids:
        raise RunError(f"{endpoint}/v1/models does not list {model!r} (listed: {ids})")
    return ids


def ask(endpoint: str, body: dict, timeout: int) -> dict:
    started = time.monotonic()
    payload = request(endpoint.rstrip("/") + "/v1/chat/completions", body, timeout)
    return {"elapsed_s": time.monotonic() - started, "payload": payload}


def graded_text(payload: dict) -> tuple:
    choices = payload.get("choices") or []
    if not choices or not isinstance(choices[0].get("message"), dict):
        raise RunError(f"reply has no choices[0].message: {json.dumps(payload)[:400]}")
    message = choices[0]["message"]
    content = message.get("content") or ""
    reasoning = message.get("reasoning_content") or ""
    text = f"<think>{reasoning}</think>{content}" if reasoning else content
    return text, content, reasoning, choices[0].get("finish_reason")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--reasoning-effort")
    ap.add_argument("--fixture", default=str(ROOT / "qualification/ds4_eval/quality-fixtures-glm5.3-flash.json"))
    ap.add_argument("--fixture-tokenizer", default=str(ROOT / "qualification/ds4_eval/tokenizer/glm-5.3-flash-tokenizer.json"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--pass-threshold", type=int, default=14)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--timeout", type=int, default=1800)
    args = ap.parse_args()
    fixture = json.loads(Path(args.fixture).read_text())
    cases = sorted((c for c in fixture["cases"] if c["id"] in COMPSEC_IDS), key=lambda c: c["id"])
    if len(cases) != 17:
        print(f"FATAL: expected 17 COMPSEC cases, found {len(cases)}", file=sys.stderr)
        return 2
    decode = load_decoder(Path(args.fixture_tokenizer))
    questions = [decode(c["prompt_token_ids"]) for c in cases]
    bodies = []
    for question in questions:
        body = {"model": args.model, "messages": [{"role": "user", "content": question}],
                "max_tokens": args.max_tokens, "temperature": args.temperature, "stream": False}
        if args.reasoning_effort:
            body["reasoning_effort"] = args.reasoning_effort
        bodies.append(body)
    try:
        models_before = require_model(args.endpoint, args.model, 60)
        started = time.monotonic()
        with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
            replies = list(pool.map(lambda body: ask(args.endpoint, body, args.timeout), bodies))
        wall_s = time.monotonic() - started
        models_after = require_model(args.endpoint, args.model, 60)
        graded = [graded_text(r["payload"]) for r in replies]
    except RunError as error:
        print(f"FATAL: {error}", file=sys.stderr)
        return 2
    out = Path(args.out)
    (out / "responses").mkdir(parents=True, exist_ok=True)
    results = []
    for i, (c, body, r, (text, content, reasoning, finish)) in enumerate(zip(cases, bodies, replies, graded), 1):
        payload = r["payload"]
        usage = payload.get("usage") or {}
        passed, extracted = grade(text, c["answer"])
        rec = {"index": i, "id": c["id"], "source": "COMPSEC", "domain": c["domain"],
               "expected": c["answer"], "extracted": extracted, "passed": passed,
               "elapsed_ms": round(r["elapsed_s"] * 1000, 1),
               "output_sha256": hashlib.sha256(text.encode()).hexdigest(),
               "prompt_token_count": usage.get("prompt_tokens"),
               "generated_token_count": usage.get("completion_tokens"),
               "reasoning_chars": len(reasoning), "content_chars": len(content),
               "status": finish}
        results.append(rec)
        (out / "responses" / f"{i:03d}-{c['id']}.json").write_text(json.dumps({
            "id": c["id"],
            "request": {"endpoint": args.endpoint.rstrip("/") + "/v1/chat/completions", "body": body,
                        "body_sha256": hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()},
            "response": payload, "graded_text": text, "grade": rec}, indent=1))
        print(f"[{i:2d}/17] {c['id']} {c['domain'][:24]:24s} {'PASS' if passed else 'FAIL'} "
              f"expected={c['answer']!r} got={extracted[:60]!r} ({r['elapsed_s']:.1f}s, "
              f"{usage.get('completion_tokens')} tok, {finish})", flush=True)
    passed_n = sum(1 for r in results if r["passed"])
    (out / "summary.json").write_text(json.dumps({
        "format": "ds4-eval-compsec17-openai-summary-v1", "generated_at": time.strftime("%FT%T%z"),
        "endpoint": args.endpoint, "model": args.model, "fixture": Path(args.fixture).name,
        "fixture_sha256": hashlib.sha256(Path(args.fixture).read_bytes()).hexdigest(),
        "wall_s": round(wall_s, 2),
        "parameters": {"max_tokens": args.max_tokens, "temperature": args.temperature,
                       "reasoning_effort": args.reasoning_effort, "concurrency": args.concurrency,
                       "pass_threshold": args.pass_threshold},
        "grading_rule": "qualification/ds4_eval/compare_runs.py extract_answer + answer_matches over "
                        "<think>reasoning_content</think>content",
        "models_listed_before": models_before, "models_listed_after": models_after,
        "completed": len(results), "passed": passed_n, "results": results}, indent=1))
    (out / "cases.json").write_text(json.dumps({"format": "ds4-eval-compsec17-openai-cases-v1", "cases": cases}, indent=1))
    write_integrity(out, len(results))
    print(f"\nCOMPSEC-17 RESULT: {passed_n}/17")
    return 0 if passed_n >= args.pass_threshold else 1


if __name__ == "__main__":
    sys.exit(main())
