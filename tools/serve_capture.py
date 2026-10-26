#!/usr/bin/env python3
import argparse
import json
import sys
import urllib.request
from pathlib import Path


def complete(endpoint, prompt, max_tokens, logprobs, timeout):
    body = json.dumps({"prompt_token_ids": prompt, "max_tokens": max_tokens, "temperature": 0, "logprobs": logprobs,
                       "speculation": False}).encode()
    request = urllib.request.Request(endpoint.rstrip("/") + "/v1/completions", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--logprobs", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=1800.0)
    args = parser.parse_args()
    capture = {}
    for prompt in json.loads(Path(args.prompts).read_text())["prompts"]:
        reply = complete(args.endpoint, prompt["prompt_token_ids"], args.max_tokens, args.logprobs, args.timeout)
        tokens = reply.get("tokens") or []
        logprobs = reply.get("token_logprobs") or []
        if not tokens or len(logprobs) != len(tokens):
            raise SystemExit(f"{prompt['name']}: the reply has {len(tokens)} tokens and {len(logprobs)} logprob lists: {json.dumps(reply)[:300]}")
        capture[prompt["name"]] = {"tokens": tokens, "logprobs": logprobs, "text": reply["choices"][0]["text"]}
        print(f"{prompt['name']}: {len(tokens)} tokens: {reply['choices'][0]['text'][:120]!r}", flush=True)
    Path(args.out).write_text(json.dumps(capture, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
