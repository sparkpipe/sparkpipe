#!/usr/bin/env python3
"""model_api_smoke - drive every model API endpoint with a tiny completion
and validate the response: HTTP 200, valid JSON, non-empty text, and
byte-identical output across two temperature-0 calls (determinism
receipt). Endpoints are internal fleet addresses by design; the URL
must still be well-formed http(s) with an explicit host.

  model_api_smoke --endpoint http://spark0:8433 [--endpoint ...] \
      [--prompt text] [--max-tokens 8] [--timeout 120]

Exit 0 only if every endpoint passes.
"""
import argparse
import http.client
import json
import sys
from urllib.parse import urlsplit


def parse_endpoint(text):
    parts = urlsplit(text)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise SystemExit(f"REFUSED endpoint (want http(s)://host:port): {text}")
    if parts.username or parts.password:
        raise SystemExit(f"REFUSED endpoint credentials: {text}")
    return parts


def request(parts, method, path, payload, timeout):
    if parts.scheme == "https":
        conn = http.client.HTTPSConnection(parts.hostname, parts.port or 443,
                                           timeout=timeout)
    else:
        conn = http.client.HTTPConnection(parts.hostname, parts.port or 80,
                                          timeout=timeout)
    body = json.dumps(payload) if payload is not None else None
    headers = {"Content-Type": "application/json"} if body else {}
    conn.request(method, path, body=body, headers=headers)
    response = conn.getresponse()
    data = response.read()
    conn.close()
    return response.status, data


def smoke_endpoint(parts, prompt, max_tokens, timeout):
    status, data = request(parts, "GET", "/v1/models", None, timeout)
    if status != 200:
        return False, f"/v1/models status {status}"
    models = json.loads(data)
    model_id = (models.get("data") or [{}])[0].get("id", "")
    if not model_id:
        return False, "/v1/models has no model id"
    payload = {"model": model_id, "prompt": prompt,
               "max_tokens": max_tokens, "temperature": 0.0}
    texts = []
    for attempt in (1, 2):
        status, data = request(parts, "POST", "/v1/completions", payload,
                               timeout)
        if status != 200:
            return False, f"completion {attempt} status {status}: {data[:120]!r}"
        document = json.loads(data)
        choices = document.get("choices") or []
        if not choices or not choices[0].get("text"):
            return False, f"completion {attempt} empty: {data[:120]!r}"
        texts.append(choices[0]["text"])
    if texts[0] != texts[1]:
        return False, f"nondeterministic: {texts[0][:40]!r} vs {texts[1][:40]!r}"
    return True, f"model={model_id} tokens={max_tokens} text={texts[0][:40]!r}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", action="append", required=True)
    parser.add_argument("--prompt", default="The quick brown fox")
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()

    failures = 0
    for endpoint in args.endpoint:
        parts = parse_endpoint(endpoint)
        try:
            ok, detail = smoke_endpoint(parts, args.prompt, args.max_tokens,
                                        args.timeout)
        except Exception as error:
            ok, detail = False, repr(error)
        verdict = "PASS" if ok else "FAIL"
        print(f"{verdict} {endpoint} {detail}")
        if not ok:
            failures += 1
    print(f"model_api_smoke endpoints={len(args.endpoint)} failures={failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
