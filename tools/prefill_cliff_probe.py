#!/usr/bin/env python3
import json
import subprocess


def try_prompt(n):
    prompt = [151644, 872, 198] + [1000 + (i % 8000) for i in range(n)] + [151645, 198]
    body = json.dumps({"prompt_token_ids": prompt, "max_tokens": 1,
                       "temperature": 0.0})
    result = subprocess.run(
        ["curl", "-s", "--max-time", "120", "-X", "POST",
         "http://127.0.0.1:8433/v1/chat/completions",
         "-H", "Content-Type: application/json", "-d", body],
        capture_output=True, text=True)
    return "error" not in result.stdout


lo, hi = 32, 64
while lo + 1 < hi:
    mid = (lo + hi) // 2
    ok = try_prompt(mid)
    print(f"n={mid} -> {'OK' if ok else 'FAIL'}", flush=True)
    if ok:
        lo = mid
    else:
        hi = mid
print(f"cliff: last OK = {lo}, first FAIL = {hi}")
