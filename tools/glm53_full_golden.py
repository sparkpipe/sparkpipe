#!/usr/bin/env python3
"""GLM-5.3 Full golden capture and comparison through a /v1/completions endpoint.

render: builds the golden prompt set with the checkpoint chat template
  (messages=[user], add_generation_prompt=True, thinking closed with </think>)
  and the checkpoint tokenizer. Long prompts quote a source file of this
  repository, cut on a line boundary at a target prompt length.
capture: sends every prompt once, greedy, and records tokens, finish reason,
  usage and (with --logprobs N) the served token's logprob plus N alternatives.
batch: sends the named prompts one at a time, then all at once, without
  logprobs, and records both token lists.
compsec: sends the committed COMPSEC-17 prompts (prompt_token_ids) with
  --concurrency requests in flight and grades them with the ds4_eval rule.
compare: compares two capture files: token equality, first divergence,
  bit-identical logprobs, served-token logprob deltas.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import importlib.util
import json
import sys
import time
import urllib.request
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
THINK_CLOSE = "</think>"
EOS_TOKEN_IDS = [154820, 154827, 154829]

SHORT_PROMPTS = [
    ("capital", "What is the capital of France? Answer in one sentence."),
    ("arith", "Compute 37 * 43 and explain the steps briefly."),
    ("code", "Write a Python function that returns the n-th Fibonacci number iteratively."),
    ("story", "Write the opening paragraph of a short story about a lighthouse keeper who finds a message in a bottle."),
    ("reason", "A train leaves at 14:10 and travels 180 km at 72 km/h. When does it arrive? Show the calculation."),
    ("translate", "Translate into German and French: The library opens at nine and closes at six, except on Sundays."),
]
LONG_QUESTION = "Explain what the C code above does: list its main functions and describe each in one sentence."


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def render(args) -> int:
    from jinja2.sandbox import ImmutableSandboxedEnvironment
    from tokenizers import Tokenizer
    environment = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"])
    environment.filters["tojson"] = lambda value, ensure_ascii=False, indent=None: json.dumps(value, ensure_ascii=ensure_ascii, indent=indent)
    template_path, tokenizer_path = Path(args.template), Path(args.tokenizer)
    template = environment.from_string(template_path.read_text())
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    close_id = tokenizer.token_to_id(THINK_CLOSE)
    if close_id is None:
        raise SystemExit(f"{THINK_CLOSE} is not a tokenizer token")

    def encode(content: str) -> tuple:
        text = template.render(messages=[{"role": "user", "content": content}], add_generation_prompt=True)
        return text, tokenizer.encode(text, add_special_tokens=False).ids + [close_id]

    prompts = []
    for name, content in SHORT_PROMPTS:
        text, ids = encode(content)
        prompts.append({"name": name, "kind": "short", "prompt_token_ids": ids, "rendered_sha256": sha256_bytes(text.encode())})
    for spec in args.long:
        name, path, target = spec.split(":")
        lines = Path(path).read_text().splitlines(keepends=True)
        best = None
        low, high = 1, len(lines)
        while low <= high:
            middle = (low + high) // 2
            content = "```c\n" + "".join(lines[:middle]) + "```\n\n" + LONG_QUESTION
            text, ids = encode(content)
            if len(ids) <= int(target):
                best = (middle, text, ids)
                low = middle + 1
            else:
                high = middle - 1
        if best is None:
            raise SystemExit(f"{name}: no prefix of {path} fits {target} tokens")
        prompts.append({"name": name, "kind": "long", "source": path, "source_lines": best[0],
                        "prompt_token_ids": best[2], "rendered_sha256": sha256_bytes(best[1].encode())})
    document = {"format": "sparkpipe-glm53full-golden-prompts-v1", "template_sha256": sha256_bytes(template_path.read_bytes()),
                "tokenizer_sha256": sha256_bytes(tokenizer_path.read_bytes()), "thinking": "off",
                "eos_token_ids": EOS_TOKEN_IDS, "prompts": prompts}
    Path(args.out).write_text(json.dumps(document, indent=1) + "\n")
    for p in prompts:
        print(p["name"], len(p["prompt_token_ids"]))
    return 0


def complete(port: int, ids: list, max_tokens: int, logprobs: int, timeout: int) -> dict:
    body = {"prompt_token_ids": ids, "max_tokens": max_tokens, "temperature": 0}
    if logprobs:
        body["logprobs"] = logprobs
    request = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    started = time.monotonic()
    reply = json.load(urllib.request.urlopen(request, timeout=timeout))
    reply["wall_s"] = round(time.monotonic() - started, 3)
    return reply


def record(reply: dict) -> dict:
    return {"tokens": reply.get("tokens"), "logprobs": reply.get("token_logprobs"), "finish_reason": reply.get("finish_reason"),
            "usage": reply.get("usage"), "status": reply.get("status"), "wall_s": reply.get("wall_s")}


def selected(document: dict, names: str | None) -> list:
    prompts = document["prompts"]
    if names is None:
        return prompts
    wanted = names.split(",")
    chosen = [p for p in prompts if p["name"] in wanted]
    if len(chosen) != len(wanted):
        raise SystemExit(f"unknown prompt names {sorted(set(wanted) - {p['name'] for p in prompts})}")
    return chosen


def capture(args) -> int:
    document = json.loads(Path(args.prompts).read_text())
    out = {"format": "sparkpipe-glm53full-golden-capture-v1", "label": args.label, "port": args.port, "max_tokens": args.max_tokens,
           "logprobs": args.logprobs, "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "results": {}}
    for p in selected(document, args.names):
        reply = complete(args.port, p["prompt_token_ids"], args.max_tokens, args.logprobs, args.timeout)
        out["results"][p["name"]] = record(reply)
        print(p["name"], len(p["prompt_token_ids"]), len(reply.get("tokens") or []), reply.get("finish_reason"), reply.get("usage"), reply["wall_s"], flush=True)
    Path(args.out).write_text(json.dumps(out) + "\n")
    return 0


def batch(args) -> int:
    document = json.loads(Path(args.prompts).read_text())
    chosen = selected(document, args.names)
    out = {"format": "sparkpipe-glm53full-golden-batch-v1", "label": args.label, "port": args.port, "max_tokens": args.max_tokens,
           "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "alone": {}, "together": {}}
    for p in chosen:
        out["alone"][p["name"]] = record(complete(args.port, p["prompt_token_ids"], args.max_tokens, 0, args.timeout))
    with concurrent.futures.ThreadPoolExecutor(len(chosen)) as pool:
        futures = {p["name"]: pool.submit(complete, args.port, p["prompt_token_ids"], args.max_tokens, 0, args.timeout) for p in chosen}
        for name, future in futures.items():
            out["together"][name] = record(future.result())
    same = 0
    for p in chosen:
        a, b = out["alone"][p["name"]]["tokens"], out["together"][p["name"]]["tokens"]
        same += a == b
        first = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
        print(p["name"], "identical" if a == b else f"differs first={first}", len(a), len(b), flush=True)
    out["identical"] = same
    print(f"batch identity {same}/{len(chosen)}")
    Path(args.out).write_text(json.dumps(out) + "\n")
    return 0


def compsec(args) -> int:
    spec = importlib.util.spec_from_file_location("glm53full_compsec17", TOOLS / "glm53full_compsec17.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    prompts = json.loads(Path(args.prompts).read_text())
    cases = prompts["cases"]
    started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(args.concurrency) as pool:
        replies = list(pool.map(lambda c: complete(args.port, c["prompt_token_ids"], args.max_tokens, args.logprobs, args.timeout), cases))
    wall = round(time.monotonic() - started, 1)
    tokens = {index: list(reply["tokens"]) for index, reply in enumerate(replies, 1)}
    summary = module.grade_cases(prompts, tokens, module.output_decoder(Path(args.tokenizer)), args.max_tokens)
    summary.update({"label": args.label, "port": args.port, "concurrency": args.concurrency, "logprobs": args.logprobs, "wall_s": wall,
                    "tokens": {cases[index - 1]["id"]: tokens[index] for index in tokens},
                    "cached_tokens": {cases[index - 1]["id"]: ((replies[index - 1].get("usage") or {}).get("prompt_tokens_details") or {}).get("cached_tokens")
                                      for index in tokens}})
    Path(args.out).write_text(json.dumps(summary, indent=1) + "\n")
    for r in summary["results"]:
        print(f"{r['id']} {'PASS' if r['passed'] else 'FAIL'} expected={r['expected']!r} got={r['extracted'][:40]!r} tokens={r['generated_token_count']}{' budget' if r['hit_budget'] else ''}")
    print(f"COMPSEC-17 {args.label}: {summary['passed']}/{summary['completed']} wall {wall}s")
    return 0


def compare(args) -> int:
    a = json.loads(Path(args.reference).read_text())["results"]
    b = json.loads(Path(args.candidate).read_text())["results"]
    exact = 0
    for name in a:
        if name not in b:
            print(name, "missing in candidate")
            continue
        ta, tb = a[name]["tokens"], b[name]["tokens"]
        first = next((i for i, (x, y) in enumerate(zip(ta, tb)) if x != y), None)
        if first is None and len(ta) != len(tb):
            first = min(len(ta), len(tb))
        line = f"{name} tokens={'equal' if first is None else f'diverge@{first}'} n={len(ta)}/{len(tb)}"
        la, lb = a[name].get("logprobs"), b[name].get("logprobs")
        if la and lb:
            span = len(ta) if first is None else first + 1
            identical = la[:span] == lb[:span]
            deltas = [abs(x[0][1] - y[0][1]) for x, y in zip(la[:span], lb[:span]) if x[0][1] is not None and y[0][1] is not None]
            deltas.sort()
            line += f" logprobs_bit_identical={identical} served_dlp_median={deltas[len(deltas) // 2] if deltas else 0:.3g} max={deltas[-1] if deltas else 0:.3g}"
            exact += first is None and identical
        else:
            exact += first is None
        print(line)
    print(f"exact {exact}/{len(a)}")
    return 0 if exact == len(a) else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("render")
    p.add_argument("--template", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--long", action="append", default=[], help="name:path:target_tokens")
    p.add_argument("--out", required=True)
    for command in ("capture", "batch", "compsec"):
        p = commands.add_parser(command)
        p.add_argument("--prompts", required=True)
        p.add_argument("--port", type=int, required=True)
        p.add_argument("--max-tokens", type=int, required=True)
        p.add_argument("--label", required=True)
        p.add_argument("--timeout", type=int, default=1800)
        p.add_argument("--out", required=True)
        if command != "compsec":
            p.add_argument("--names")
        if command in ("capture", "compsec"):
            p.add_argument("--logprobs", type=int, default=0)
        if command == "compsec":
            p.add_argument("--concurrency", type=int, required=True)
            p.add_argument("--tokenizer", required=True)
    p = commands.add_parser("compare")
    p.add_argument("reference")
    p.add_argument("candidate")
    args = parser.parse_args()
    return {"render": render, "capture": capture, "batch": batch, "compsec": compsec, "compare": compare}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
