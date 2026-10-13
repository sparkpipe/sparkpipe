#!/usr/bin/env python3
import json
import sys
import urllib.request

port = int(sys.argv[1])
text = open("/home/spec/ctx_64k.txt").read()
prompts = [
    ("short", "[golden-v1-short] Explain how binary search works, with a short example."),
    ("p600", "[golden-v1-p600]\n" + text[3000:5400] + "\n\nWhat is the main topic of the text above?"),
    ("p3k", "[golden-v1-p3k]\n" + text[8000:20000] + "\n\nList three facts stated in the text above."),
    ("p9k", "[golden-v1-p9k]\n" + text[20000:52000] + "\n\nSummarize the text above in two sentences."),
]
out = {}
for name, content in prompts:
    body = {"messages": [{"role": "user", "content": content}], "max_tokens": 48, "temperature": 0, "logprobs": 3}
    request = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    reply = json.loads(urllib.request.urlopen(request, timeout=1800).read())
    out[name] = {"content": reply["choices"][0]["message"].get("content"), "usage": reply.get("usage"), "token_logprobs": reply.get("token_logprobs")}
    usage = reply.get("usage") or {}
    print(name, "prompt", usage.get("prompt_tokens"), "cached", (usage.get("prompt_tokens_details") or {}).get("cached_tokens"), repr((out[name]["content"] or "")[:60]))
json.dump(out, open(sys.argv[2], "w"))
