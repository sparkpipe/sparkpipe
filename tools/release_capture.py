#!/usr/bin/env python3
import json
import sys
import urllib.request

port, out = int(sys.argv[1]), sys.argv[2]
source = open(sys.argv[3], errors="replace").read()
prompts = {
    "capital": ("The capital of France is", 32),
    "arith": ("Question: What is 17 multiplied by 23? Answer:", 32),
    "code": ("def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n", 48),
    "story": ("Once upon a time, in a village by the sea,", 48),
    "long_2k": ("Summarize what this C source does.\n\n" + source[-7000:] + "\n\nSummary:", 48),
    "long_8k": ("Summarize what this C source does.\n\n" + source[:28000] + "\n\nSummary:", 48),
}
result = {}
for name, (text, tokens) in prompts.items():
    body = json.dumps({"prompt": text, "max_tokens": tokens, "temperature": 0, "logprobs": 4}).encode()
    request = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", body, {"Content-Type": "application/json"})
    reply = json.load(urllib.request.urlopen(request, timeout=1800))
    usage = reply.get("usage") or {}
    result[name] = {"tokens": reply.get("tokens"), "logprobs": reply.get("token_logprobs"), "prompt_tokens": usage.get("prompt_tokens"),
                    "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens")}
    print(name, result[name]["prompt_tokens"], result[name]["cached_tokens"], len(result[name]["tokens"] or []))
json.dump(result, open(out, "w"))
