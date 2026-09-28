import json, sys, time, http.client, hashlib
sys.path.insert(0, "tools")
import compsec17 as c
from pathlib import Path
decode = c.load_decoder(Path("qualification/ds4_eval/tokenizer/glm-5.3-flash-tokenizer.json"))
fx = json.loads(Path("qualification/ds4_eval/quality-fixtures-glm5.3-flash.json").read_text())
cases = {x["id"]: x for x in fx["cases"]}
def send(prompt, n):
    h = http.client.HTTPConnection("127.0.0.1", 8436, timeout=600)
    h.request("POST", "/v1/completions", json.dumps({"prompt": prompt, "max_tokens": n, "temperature": 0}), {"Content-Type": "application/json"})
    d = json.loads(h.getresponse().read())
    return d["choices"][0]["text"], d.get("tokens") or [], d.get("usage")
def cached(sha):
    best = None
    for l in open("/home/spec/gemma4-api-channel/api.log"):
        if '"request_measurements"' in l and sha in l:
            try: best = json.loads(l).get("cached_prompt_tokens")
            except Exception: pass
    return best
for cid in sys.argv[1].split(","):
    q = decode(cases[cid]["prompt_token_ids"])
    exact = c.build_prompt(q, "off", "gemma4")
    cold = exact.replace("<bos><|turn>user\n", "<bos><|turn>user\n[probe %d] " % time.time_ns(), 1)
    out = []
    for tag, p in (("cold", cold), ("warm1", exact), ("warm2", exact)):
        t, toks, u = send(p, int(sys.argv[2]))
        time.sleep(0.5)
        out.append({"case": cid, "arm": tag, "prompt_tokens": (u or {}).get("prompt_tokens"), "tokens": len(toks),
                    "token_sha": hashlib.sha256(json.dumps(toks).encode()).hexdigest()[:16], "text": t[:160]})
    ref = out[0]
    for r in out:
        print(json.dumps(r), flush=True)
