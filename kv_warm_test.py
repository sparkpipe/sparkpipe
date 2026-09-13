import json, http.client, time
doc = json.load(open("/tmp/quality-fixtures-glm5.3-flash.json"))
cases = [c for c in (doc if isinstance(doc, list) else doc.get("cases", [])) if str(c.get("id","")).startswith("compsec")]
ids = cases[0]["prompt_token_ids"]
def call(n):
    body = json.dumps({"prompt_token_ids": ids, "max_tokens": n, "temperature": 0})
    conn = http.client.HTTPConnection("127.0.0.1", 8433, timeout=600)
    t0 = time.monotonic()
    conn.request("POST", "/v1/completions", body=body, headers={"Content-Type": "application/json"})
    r = json.loads(conn.getresponse().read())
    el = time.monotonic() - t0
    conn.close()
    return el, r
t1, p = call(1)
print(f"prefill #1 (max_tokens=1): {t1*1000:.0f} ms  got={p.get('tokens')}")
t2, p = call(1)
print(f"prefill #2 same question : {t2*1000:.0f} ms")
t33, p = call(33)
got = len(p.get("tokens", []))
print(f"warm 33-token run        : {t33:.2f} s total, {got} tokens")
if got > 1 and t33 > t2:
    print(f"warm decode rate         : {(got-1)/(t33-t2):.2f} tok/s")
