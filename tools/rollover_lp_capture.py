#!/usr/bin/env python3
import json
import sys
import urllib.request

reference = json.load(open(sys.argv[1]))
port = int(sys.argv[2])
out = {}
for prompt in reference["prompts"]:
    want = prompt.get("reference_token_ids") or prompt.get("generated_token_ids")
    body = json.dumps({"prompt_token_ids": prompt["prompt_token_ids"], "max_tokens": len(want), "temperature": 0, "logprobs": 4}).encode()
    request = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", body, {"Content-Type": "application/json"})
    reply = json.load(urllib.request.urlopen(request, timeout=900))
    out[prompt["name"]] = {"tokens": reply["tokens"], "logprobs": reply.get("token_logprobs")}
    print(prompt["name"], len(reply["tokens"]))
json.dump(out, open(sys.argv[3], "w"))
