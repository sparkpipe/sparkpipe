#!/usr/bin/env python3
import concurrent.futures
import json
import sys
import time
import urllib.request

port, out = int(sys.argv[1]), sys.argv[2]
source = open(sys.argv[3], errors="replace").read()
api_log = sys.argv[4] if len(sys.argv) > 4 else None
excerpt = source[2000:5200]
prompts = {
    "capital": ("The capital of France is", 32),
    "arith": ("Question: What is 17 multiplied by 23? Answer:", 32),
    "code": ("def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n", 48),
    "story": ("Once upon a time, in a village by the sea,", 48),
    "long_2k": ("Summarize what this C source does.\n\n" + source[-7000:] + "\n\nSummary:", 48),
    "long_8k": ("Summarize what this C source does.\n\n" + source[:28000] + "\n\nSummary:", 48),
    "copy": ("Here is a C excerpt.\n\n" + excerpt + "\n\nThe same excerpt again, unchanged:\n\n", 160),
}
SEED = 1234
if len(sys.argv) > 5:
    unknown = [name for name in sys.argv[5].split(",") if name not in prompts]
    if unknown:
        raise SystemExit(f"release_capture: unknown prompts {unknown}; the kit has {list(prompts)}")
    prompts = {name: prompts[name] for name in sys.argv[5].split(",")}


def post(text, tokens, **extra):
    body = {"prompt": text, "max_tokens": tokens, "temperature": 0}
    body.update(extra)
    request = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    started = time.monotonic()
    reply = json.load(urllib.request.urlopen(request, timeout=1800))
    return reply, time.monotonic() - started


def speculation_counters():
    if api_log is None:
        return None
    last = None
    with open(api_log, errors="replace") as log:
        for line in log:
            if '"engine_measurements"' in line:
                last = line
    if last is None:
        return None
    event = json.loads(last[last.index("{"):])
    return {key: event[key] for key in ("verify_lanes", "draft_tokens", "accepted_draft_tokens")}


result = {}
for name, (text, tokens) in prompts.items():
    single, _ = post(text, tokens, logprobs=4, speculation=False)
    chain, chain_seconds = post(text, tokens, speculation=False)
    before = speculation_counters()
    speculate, speculate_seconds = post(text, tokens)
    after = speculation_counters()
    sampled, _ = post(text, tokens, temperature=0.8, top_p=0.95, seed=SEED, speculation=False)
    usage = single.get("usage") or {}
    result[name] = {"tokens": single.get("tokens"), "logprobs": single.get("token_logprobs"), "prompt_tokens": usage.get("prompt_tokens"),
                    "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
                    "chain_tokens": chain.get("tokens"), "chain_seconds": round(chain_seconds, 3),
                    "speculate_tokens": speculate.get("tokens"), "speculate_seconds": round(speculate_seconds, 3),
                    "sampled_tokens": sampled.get("tokens")}
    if before is not None and after is not None:
        result[name]["speculation"] = {key: after[key] - before[key] for key in after}
    print(name, result[name]["prompt_tokens"], result[name]["cached_tokens"], len(result[name]["tokens"] or []),
          "chain_s", result[name]["chain_seconds"], "speculate_s", result[name]["speculate_seconds"], result[name].get("speculation", ""))
with concurrent.futures.ThreadPoolExecutor(len(prompts)) as pool:
    batch = dict(zip(prompts, pool.map(lambda item: post(item[0], item[1], speculation=False)[0], prompts.values())))
    sampled_batch = dict(zip(prompts, pool.map(lambda item: post(item[0], item[1], temperature=0.8, top_p=0.95, seed=SEED, speculation=False)[0],
                                               prompts.values())))
for name in prompts:
    result[name]["batch_tokens"] = batch[name].get("tokens")
    result[name]["sampled_batch_tokens"] = sampled_batch[name].get("tokens")
json.dump(result, open(out, "w"))
