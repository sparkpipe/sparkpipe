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
SOURCES = re.compile(r"VERIFY-MTP drafts=(\d+) tokens=(\d+) cold=(\d+) truncated=(\d+) taps=(\d+) draft_us=(\d+) \| "
                     r"lookup rounds=(\d+) proposed=(\d+) accepted=(\d+) declined=(\d+) \| mtp rounds=(\d+) proposed=(\d+) accepted=(\d+)")
POSITIONS = re.compile(r"VERIFY-POSITIONS((?: p\d+=\d+/\d+)+)")
POSITION = re.compile(r"p(\d+)=(\d+)/(\d+)")


def post(endpoint: str, path: str, body: dict, timeout: int) -> tuple[dict, float]:
    request = urllib.request.Request(endpoint.rstrip("/") + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
    start = time.monotonic()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read())
    return payload, time.monotonic() - start


def token_ids(tokens) -> list[int]:
    return [int(t[1]) if isinstance(t, list) else int(t) for t in tokens]


def output_tokens(payload: dict) -> list[int]:
    tokens = payload.get("tokens")
    if tokens is None:
        raise SystemExit("the endpoint response carries no 'tokens' field; exactness needs token ids")
    return token_ids(tokens)


def stream(endpoint: str, path: str, body: dict, timeout: int) -> dict:
    request = urllib.request.Request(endpoint.rstrip("/") + path, data=json.dumps(dict(body, stream=True)).encode(),
                                     headers={"Content-Type": "application/json", "Accept": "text/event-stream"}, method="POST")
    start = time.monotonic()
    ids: list[int] = []
    text: list[str] = []
    events: list[tuple[float, int]] = []
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw in response:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                continue
            event = json.loads(line[5:])
            if "tokens" not in event:
                raise SystemExit(f"a stream event carries no 'tokens' field; exactness needs token ids: {line[:200]}")
            chunk = token_ids(event["tokens"])
            choice = (event.get("choices") or [{}])[0]
            text.append(choice.get("text") or (choice.get("delta") or {}).get("content") or "")
            if chunk:
                events.append((time.monotonic() - start, len(chunk)))
                ids.extend(chunk)
    wall = time.monotonic() - start
    if not ids:
        raise SystemExit("the stream produced no token ids")
    ttft = events[0][0]
    decode_tokens = len(ids) - events[0][1]
    decode_s = events[-1][0] - ttft
    return {"token_ids": ids, "text": "".join(text), "wall_s": wall, "ttft_s": ttft, "decode_tokens": decode_tokens,
            "decode_s": decode_s, "decode_tok_s": decode_tokens / decode_s if decode_s > 0 and decode_tokens > 0 else None}


def read_u32(path: str) -> list[int]:
    data = Path(path).read_bytes()
    if len(data) % 4 != 0:
        raise SystemExit(f"{path} is not a u32 token file")
    return list(struct.unpack(f"<{len(data) // 4}I", data))


def record(args: argparse.Namespace) -> int:
    prompt = [int(t) for t in json.loads(Path(args.prompt_ids).read_text())]
    payload, _ = post(args.endpoint, "/v1/chat/completions",
                      {"prompt_token_ids": prompt, "max_tokens": args.max_tokens, "temperature": 0.0}, args.timeout)
    sequence = prompt + output_tokens(payload)
    Path(args.out).write_bytes(struct.pack(f"<{len(sequence)}I", *sequence))
    print(json.dumps({"prompt_tokens": len(prompt), "output_tokens": len(sequence) - len(prompt), "out": args.out}))
    return 0


def replay(args: argparse.Namespace) -> int:
    prompt = [int(t) for t in json.loads(Path(args.prompt_ids).read_text())]
    expected = read_u32(args.expect)
    if expected[:len(prompt)] != prompt:
        raise SystemExit(f"{args.expect} does not start with the prompt in {args.prompt_ids}")
    result = stream(args.endpoint, "/v1/chat/completions",
                    {"prompt_token_ids": prompt, "max_tokens": len(expected) - len(prompt), "temperature": 0.0}, args.timeout)
    output = expected[len(prompt):]
    first = next((index for index, (a, b) in enumerate(zip(result["token_ids"], output)) if a != b), None)
    exact = result["token_ids"] == output
    report = {"exact": exact, "expected_tokens": len(output), "output_tokens": len(result["token_ids"]),
              "first_mismatch": first if first is not None or exact else min(len(output), len(result["token_ids"])),
              "ttft_s": result["ttft_s"], "decode_tok_s": result["decode_tok_s"]}
    print(json.dumps(report))
    return 0 if exact else 1


def run(args: argparse.Namespace) -> int:
    results = []
    for content_class in args.classes.split(","):
        if content_class not in PROMPTS:
            raise SystemExit(f"unknown content class {content_class}; choose from {sorted(PROMPTS)}")
        for index, prompt in enumerate(PROMPTS[content_class]):
            result = stream(args.endpoint, "/v1/completions",
                            {"prompt": GLM_PREFIX + prompt + GLM_ASSISTANT, "max_tokens": args.max_tokens, "temperature": 0.0},
                            args.timeout)
            results.append(dict(result, **{"class": content_class, "index": index, "output_tokens": len(result["token_ids"])}))
            rate = result["decode_tok_s"]
            print(f"{content_class}[{index}] {len(result['token_ids'])} tokens, ttft {result['ttft_s']:.3f}s, decode "
                  f"{'n/a' if rate is None else f'{rate:.1f} tok/s'}", flush=True)
    Path(args.out).write_text(json.dumps({"label": args.label, "max_tokens": args.max_tokens, "results": results}, indent=1))
    return 0


def summarize(results: list[dict]) -> dict:
    classes: dict[str, dict] = {}
    for entry in results:
        total = classes.setdefault(entry["class"], {"tokens": 0, "decode_s": 0.0})
        total["tokens"] += entry["decode_tokens"]
        total["decode_s"] += entry["decode_s"]
    return {name: total["tokens"] / total["decode_s"] if total["decode_s"] > 0 else None for name, total in classes.items()}


def compare_runs(base: dict, spec: dict) -> dict:
    keyed = {(entry["class"], entry["index"]): entry for entry in base["results"]}
    mismatches = []
    for entry in spec["results"]:
        other = keyed.get((entry["class"], entry["index"]))
        if other is None:
            mismatches.append(f"{entry['class']}[{entry['index']}] missing from the baseline")
            continue
        if not entry.get("token_ids") or not other.get("token_ids"):
            mismatches.append(f"{entry['class']}[{entry['index']}] has no token ids; exactness cannot be judged from text")
            continue
        if entry["token_ids"] != other["token_ids"] or entry["text"] != other["text"]:
            mismatches.append(f"{entry['class']}[{entry['index']}] output differs")
    base_rates, spec_rates = summarize(base["results"]), summarize(spec["results"])
    speedups = {name: spec_rates[name] / base_rates[name] for name in spec_rates if base_rates.get(name) and spec_rates[name] is not None}
    return {"exact": not mismatches, "mismatches": mismatches, "base_decode_tok_s": base_rates, "spec_decode_tok_s": spec_rates,
            "speedup": speedups}


def compare(args: argparse.Namespace) -> int:
    report = compare_runs(json.loads(Path(args.base).read_text()), json.loads(Path(args.spec).read_text()))
    print(json.dumps(report, indent=1))
    return 0 if report["exact"] else 1


def source_report(rounds: int, proposed: int, accepted: int) -> dict:
    return {"rounds": rounds, "proposed": proposed, "accepted": accepted,
            "acceptance": accepted / proposed if proposed else 0.0,
            "accept_length": accepted / rounds if rounds else 0.0,
            "tokens_per_round": (accepted + rounds) / rounds if rounds else 0.0}


def parse_log(lines) -> dict:
    frames = rounds = accepted = produced = budget = steps = 0
    sources = None
    positions = None
    for line in lines:
        found = POSITIONS.search(line)
        if found is not None:
            positions = [{"position": int(index), "accepted": int(taken), "reached": int(reached),
                          "acceptance": int(taken) / int(reached) if int(reached) else None}
                         for index, taken, reached in POSITION.findall(found.group(1))]
            continue
        found = SOURCES.search(line)
        if found is not None:
            values = [int(value) for value in found.groups()]
            sources = {"mtp_drafts": values[0], "mtp_draft_tokens": values[1], "mtp_cold": values[2], "mtp_truncated": values[3],
                       "mtp_taps": values[4], "mtp_draft_us": values[5], "lookup_declined": values[9],
                       "lookup": source_report(values[6], values[7], values[8]), "mtp": source_report(values[10], values[11], values[12])}
            continue
        match = FRAME.search(line)
        if match is None:
            continue
        frames += 1
        budget += int(match.group(3))
        produced += int(match.group(4))
        rounds += int(match.group(5))
        accepted += int(match.group(6))
        steps += int(match.group(7) or 0)
    report = {"verify_frames": frames, "rounds": rounds, "accepted_drafts": accepted, "plain_steps": steps, "produced_tokens": produced,
              "tokens_per_round": (produced - steps) / rounds if rounds else 0.0,
              "tokens_per_frame": produced / frames if frames else 0.0,
              "frame_fill": produced / budget if budget else 0.0}
    if sources is not None:
        report["sources"] = sources
    if positions is not None:
        report["acceptance_per_position"] = positions
    return report


def log(args: argparse.Namespace) -> int:
    with open(args.log, errors="replace") as handle:
        print(json.dumps(parse_log(handle), indent=1))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="spec_verify_bench")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("record")
    p.add_argument("--endpoint", required=True)
    p.add_argument("--prompt-ids", required=True)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--timeout", type=int, default=900)
    p.add_argument("--out", required=True)
    p.set_defaults(function=record)
    p = sub.add_parser("replay")
    p.add_argument("--endpoint", required=True)
    p.add_argument("--prompt-ids", required=True)
    p.add_argument("--expect", required=True)
    p.add_argument("--timeout", type=int, default=900)
    p.set_defaults(function=replay)
    p = sub.add_parser("run")
    p.add_argument("--endpoint", required=True)
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
