#!/usr/bin/env python3
import json
import subprocess
import sys
import time


LOG = "/home/spark0/sparkdata/glm53flash.fp8.tp16/api.log"
PREFILL_TOKENS = 512
DECODE_TOKENS = 48


def post(prompt, max_tokens, timeout=600):
    payload = json.dumps({
        "prompt_token_ids": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
    })
    start = time.monotonic()
    result = subprocess.run(
        ["curl", "-s", "--max-time", str(timeout),
         "-X", "POST", "http://127.0.0.1:8433/v1/chat/completions",
         "-H", "Content-Type: application/json",
         "-d", payload],
        capture_output=True, text=True)
    wall = time.monotonic() - start
    if result.returncode != 0:
        raise RuntimeError(f"curl failed: {result.returncode}")
    return json.loads(result.stdout), wall


def measurements(since_monotonic):
    lines = subprocess.run(["tail", "-c", "2000000", LOG],
                           capture_output=True, text=True).stdout
    found = []
    for line in lines.splitlines():
        if '"event":"request_measurements"' not in line:
            continue
        start = line.find("{")
        try:
            entry = json.loads(line[start:line.rfind("}") + 1])
        except ValueError:
            continue
        found.append(entry)
    return found


def main():
    prompt = [151644, 872, 198]
    prompt += [1000 + (i % 8000) for i in range(PREFILL_TOKENS)]
    prompt += [151645, 198]

    print("== prefill pass (populates the JIT KV / prefix entry)")
    payload, wall = post(prompt, 1)
    print(f"prefill: wall={wall:.3f}s tokens={len(payload['tokens'])}")

    print("== decode pass (same prompt: prefill must hit the cache)")
    payload, wall = post(prompt, DECODE_TOKENS)
    print(f"decode: wall={wall:.3f}s tokens={len(payload['tokens'])}")

    time.sleep(0.5)
    for entry in measurements(0):
        tokens = entry.get("tokens", [])
        ready = [t[1] for t in tokens]
        tag = ""
        if len(ready) > 1:
            deltas = sorted(b - a for a, b in zip(ready, ready[1:]))
            decode_s = (ready[-1] - ready[0]) / 1e9
            rate = (len(ready) - 1) / decode_s if decode_s > 0 else 0
            tag = (f"decode_s={decode_s:.3f} decode_tok_s={rate:.2f} "
                   f"median_ms={deltas[len(deltas)//2]/1e6:.1f} "
                   f"max_ms={deltas[-1]/1e6:.1f}")
        else:
            tag = "single-token"
        print(f"req={entry['request_id']} prompt={entry['prompt_tokens']} "
              f"cached={entry['cached_prompt_tokens']} out={len(ready)} {tag}")


if __name__ == "__main__":
    sys.exit(main())
