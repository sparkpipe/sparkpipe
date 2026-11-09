#!/usr/bin/env python3
import argparse
import json
import random
import sys
import time
import urllib.request
from pathlib import Path

TOKEN_LOW = 1000
TOKEN_HIGH = 30000
REGRESSION_SHARE = 0.85


def fill(endpoint, tokens, timeout):
    ids = [random.randrange(TOKEN_LOW, TOKEN_HIGH) for _ in range(tokens)]
    body = json.dumps({"prompt_token_ids": ids, "max_tokens": 1, "temperature": 0}).encode()
    request = urllib.request.Request(endpoint + "/v1/completions", data=body, headers={"Content-Type": "application/json"})
    started = time.monotonic()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        reply = json.loads(response.read())
    seconds = time.monotonic() - started
    cached = (reply.get("usage", {}).get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    return seconds, tokens - cached


def main():
    parser = argparse.ArgumentParser(description="Cold fill of a long prompt, compared with the fastest this model has done")
    parser.add_argument("endpoint")
    parser.add_argument("model")
    parser.add_argument("--tokens", type=int, default=33000)
    parser.add_argument("--best", required=True, help="JSON file of each model's fastest fill, tok/s")
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()
    seconds, filled = fill(args.endpoint, args.tokens, args.timeout)
    rate = filled / seconds
    path = Path(args.best)
    best = json.loads(path.read_text()) if path.exists() else {}
    known = best.get(args.model, 0.0)
    print(f"LONG-FILL {args.model} {filled} tokens {seconds:.1f} s {rate:.0f} tok/s, fastest {known:.0f} tok/s", flush=True)
    if rate > known:
        best[args.model] = round(rate, 1)
        path.write_text(json.dumps(best, indent=1, sort_keys=True) + "\n")
    if rate < REGRESSION_SHARE * known:
        print(f"LONG-FILL-REGRESSION {args.model}: {rate:.0f} tok/s is under {REGRESSION_SHARE:.0%} of {known:.0f} tok/s", flush=True)
        sys.exit(3)


if __name__ == "__main__":
    main()
