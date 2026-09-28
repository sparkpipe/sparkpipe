#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import struct
import sys
import time
import urllib.request
from pathlib import Path

GLM_PREFIX = "[gMASK]<sop><|user|>\n"
GLM_ASSISTANT = "<|assistant|>\n<think></think>\n"

PROMPTS = {
    "prose": [
        "Explain, in three careful paragraphs, why the sky appears blue during the day and red at sunset.",
        "Write a short essay on the history of the printing press and its effect on literacy in Europe.",
        "Summarize the causes and consequences of the 2008 financial crisis for a general reader.",
    ],
    "code": [
        "Write a Python module implementing a thread-safe LRU cache class with get, put, delete and a size limit, with docstrings and a unittest suite.",
        "Write a C function that parses an IPv4 dotted-quad string into a uint32_t, returning an error code for malformed input, plus a test main that checks 20 cases.",
        "Refactor this JavaScript into modern ES modules with async/await: function load(u,cb){var x=new XMLHttpRequest();x.onload=function(){cb(null,JSON.parse(x.responseText))};x.onerror=function(){cb(new Error('fail'))};x.open('GET',u);x.send()} and repeat the refactor for save(u,data,cb) and remove(u,cb).",
    ],
    "repetitive": [
        "Print the multiplication table from 1 x 1 to 12 x 12, one equation per line, in the form 'a x b = c'.",
        "List the numbers from 1 to 200, each followed by a comma and its square, one per line.",
        "Write a CSV with a header row 'id,name,status' and 60 rows where id runs from 1 to 60, name is 'user_<id>' and status alternates between active and inactive.",
    ],
}

FRAME = re.compile(r"VERIFY-FRAME slot=(\d+) position=(\d+) budget=(\d+) produced=(\d+) rounds=(\d+) accepted=(\d+)(?: steps=(\d+))?")


def post(endpoint: str, path: str, body: dict, timeout: int) -> tuple[dict, float]:
    request = urllib.request.Request(endpoint.rstrip("/") + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
    start = time.monotonic()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read())
    return payload, time.monotonic() - start


def output_tokens(payload: dict) -> list[int]:
    tokens = payload.get("tokens")
    if tokens is None:
        raise SystemExit("the endpoint response carries no 'tokens' field; exactness needs token ids")
    return [int(t[1]) if isinstance(t, list) else int(t) for t in tokens]


def output_text(payload: dict) -> str:
    choice = (payload.get("choices") or [{}])[0]
    return choice.get("text") or (choice.get("message") or {}).get("content") or ""


def record(args: argparse.Namespace) -> int:
    prompt = [int(t) for t in json.loads(Path(args.prompt_ids).read_text())]
    payload, _ = post(args.endpoint, "/v1/chat/completions",
                      {"prompt_token_ids": prompt, "max_tokens": args.max_tokens, "temperature": 0.0}, args.timeout)
    sequence = prompt + output_tokens(payload)
    Path(args.out).write_bytes(struct.pack(f"<{len(sequence)}I", *sequence))
    print(json.dumps({"prompt_tokens": len(prompt), "output_tokens": len(sequence) - len(prompt), "out": args.out}))
    return 0


def run(args: argparse.Namespace) -> int:
    results = []
    for content_class in args.classes.split(","):
        if content_class not in PROMPTS:
            raise SystemExit(f"unknown content class {content_class}; choose from {sorted(PROMPTS)}")
        for index, prompt in enumerate(PROMPTS[content_class]):
            payload, wall = post(args.endpoint, "/v1/completions",
                                 {"prompt": GLM_PREFIX + prompt + GLM_ASSISTANT, "max_tokens": args.max_tokens,
                                  "temperature": 0.0}, args.timeout)
            tokens = payload.get("tokens")
            text = output_text(payload)
            count = len(tokens) if tokens is not None else int((payload.get("usage") or {}).get("completion_tokens", 0))
            results.append({"class": content_class, "index": index, "wall_s": wall, "output_tokens": count,
                            "tokens_per_s": count / wall if wall > 0 else 0.0, "text": text,
                            "token_ids": [int(t[1]) if isinstance(t, list) else int(t) for t in tokens] if tokens is not None else None})
            print(f"{content_class}[{index}] {count} tokens in {wall:.2f}s = {count / wall:.1f} tok/s", flush=True)
    Path(args.out).write_text(json.dumps({"label": args.label, "max_tokens": args.max_tokens, "results": results}, indent=1))
    return 0


def summarize(results: list[dict]) -> dict:
    classes: dict[str, dict] = {}
    for entry in results:
        total = classes.setdefault(entry["class"], {"tokens": 0, "wall_s": 0.0})
        total["tokens"] += entry["output_tokens"]
        total["wall_s"] += entry["wall_s"]
    return {name: total["tokens"] / total["wall_s"] if total["wall_s"] > 0 else 0.0 for name, total in classes.items()}


def compare_runs(base: dict, spec: dict) -> dict:
    keyed = {(entry["class"], entry["index"]): entry for entry in base["results"]}
    mismatches = []
    for entry in spec["results"]:
        other = keyed.get((entry["class"], entry["index"]))
        if other is None:
            mismatches.append(f"{entry['class']}[{entry['index']}] missing from the baseline")
            continue
        same_ids = entry["token_ids"] is None or other["token_ids"] is None or entry["token_ids"] == other["token_ids"]
        if entry["text"] != other["text"] or not same_ids:
            mismatches.append(f"{entry['class']}[{entry['index']}] output differs")
    base_rates, spec_rates = summarize(base["results"]), summarize(spec["results"])
    speedups = {name: spec_rates[name] / base_rates[name] for name in spec_rates if base_rates.get(name)}
    return {"exact": not mismatches, "mismatches": mismatches, "base_tok_s": base_rates, "spec_tok_s": spec_rates, "speedup": speedups}


def compare(args: argparse.Namespace) -> int:
    report = compare_runs(json.loads(Path(args.base).read_text()), json.loads(Path(args.spec).read_text()))
    print(json.dumps(report, indent=1))
    return 0 if report["exact"] else 1


def parse_log(lines) -> dict:
    frames = rounds = accepted = produced = budget = steps = 0
    for line in lines:
        match = FRAME.search(line)
        if match is None:
            continue
        frames += 1
        budget += int(match.group(3))
        produced += int(match.group(4))
        rounds += int(match.group(5))
        accepted += int(match.group(6))
        steps += int(match.group(7) or 0)
    return {"verify_frames": frames, "rounds": rounds, "accepted_drafts": accepted, "plain_steps": steps, "produced_tokens": produced,
            "tokens_per_round": (produced - steps) / rounds if rounds else 0.0,
            "tokens_per_frame": produced / frames if frames else 0.0,
            "frame_fill": produced / budget if budget else 0.0}


def log(args: argparse.Namespace) -> int:
    with open(args.log, errors="replace") as handle:
        print(json.dumps(parse_log(handle), indent=1))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="spec_verify_bench")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("record")
    p.add_argument("--endpoint", default="http://127.0.0.1:8433")
    p.add_argument("--prompt-ids", required=True)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--timeout", type=int, default=900)
    p.add_argument("--out", required=True)
    p.set_defaults(function=record)
    p = sub.add_parser("run")
    p.add_argument("--endpoint", default="http://127.0.0.1:8433")
    p.add_argument("--classes", default="prose,code,repetitive")
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--timeout", type=int, default=900)
    p.add_argument("--label", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(function=run)
    p = sub.add_parser("compare")
    p.add_argument("base")
    p.add_argument("spec")
    p.set_defaults(function=compare)
    p = sub.add_parser("log")
    p.add_argument("log")
    p.set_defaults(function=log)
    args = parser.parse_args(argv)
    return args.function(args)


if __name__ == "__main__":
    sys.exit(main())
