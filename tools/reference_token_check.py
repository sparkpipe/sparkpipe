#!/usr/bin/env python3
"""Token-exactness check of a served model against reference greedy tokens.

Reads a reference file (kind sparkpipe.*.reference-tokens.v1: prompts with
prompt_token_ids and generated_token_ids), sends each prompt as
prompt_token_ids to /v1/completions at temperature 0 with max_tokens equal to
the reference length, and reports the first divergent position per prompt.
Exit 0 only when every prompt matches exactly.

usage:
  reference_token_check.py --endpoint http://127.0.0.1:8434 \\
      --reference qualification/dsv41_flash/runs/prep-20260928/reference_tokens.json --timeout 600
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request


def complete(endpoint: str, prompt_ids: list, max_tokens: int, timeout: int) -> list:
    body = json.dumps({"prompt_token_ids": prompt_ids, "max_tokens": max_tokens, "temperature": 0.0}).encode()
    request = urllib.request.Request(endpoint.rstrip("/") + "/v1/completions", data=body,
                                     headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read())
    if payload.get("status") not in (0, None) or "tokens" not in payload:
        raise RuntimeError(f"endpoint returned no tokens: {json.dumps(payload)[:300]}")
    return [int(token) for token in payload["tokens"]]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--timeout", type=int, required=True)
    args = parser.parse_args()
    reference = json.load(open(args.reference, encoding="utf-8"))
    exact = 0
    for prompt in reference["prompts"]:
        want = [int(token) for token in prompt["generated_token_ids"]]
        got = complete(args.endpoint, prompt["prompt_token_ids"], len(want), args.timeout)
        diverge = next((index for index, (a, b) in enumerate(zip(want, got)) if a != b), None)
        if diverge is None and len(got) != len(want):
            diverge = min(len(got), len(want))
        exact += diverge is None
        print(f"{prompt['name']}: {'EXACT' if diverge is None else f'DIVERGE at {diverge}'} "
              f"({len(got)}/{len(want)} tokens) want={want[:8]} got={got[:8]}", flush=True)
    print(f"token-exact {exact}/{len(reference['prompts'])}")
    return 0 if exact == len(reference["prompts"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
