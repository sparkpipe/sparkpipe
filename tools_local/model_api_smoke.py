#!/usr/bin/env python3
"""model_api_smoke - drive every model API endpoint with a tiny completion
and validate the response: HTTP 200, valid JSON, non-empty text, and
byte-identical output across two temperature-0 calls (determinism
receipt). Endpoints are internal fleet addresses by design; the URL
must still be well-formed http(s) with an explicit host.

  model_api_smoke --endpoint http://spark0:8433 [--endpoint ...] \
      [--prompt text] [--max-tokens 8] [--timeout 120] [--token bearer]

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


def request(parts, method, path, payload, timeout, token=None):
    if parts.scheme == "https":
        conn = http.client.HTTPSConnection(parts.hostname, parts.port or 443,
                                           timeout=timeout)
    else:
        conn = http.client.HTTPConnection(parts.hostname, parts.port or 80,
                                          timeout=timeout)
    body = json.dumps(payload) if payload is not None else None
    headers = {"Content-Type": "application/json"} if body else {}
    if token:
        headers["Authorization"] = "Bearer " + token
    conn.request(method, path, body=body, headers=headers)
    response = conn.getresponse()
    data = response.read()
    conn.close()
    return response.status, data


def smoke_model(parts, model_id, prompt, max_tokens, timeout, token=None):
    payload = {"model": model_id, "prompt": prompt,
               "max_tokens": max_tokens, "temperature": 0.0}
    texts = []
    for attempt in (1, 2):
        status, data = request(parts, "POST", "/v1/completions", payload,
                               timeout, token)
        if status != 200:
            return False, f"completion {attempt} status {status}: {data[:120]!r}"
        document = json.loads(data)
        choices = document.get("choices") or []
        if not choices or not choices[0].get("text"):
            return False, f"completion {attempt} empty: {data[:120]!r}"
        texts.append(choices[0]["text"])
    if texts[0] != texts[1]:
        return False, f"nondeterministic: {texts[0][:40]!r} vs {texts[1][:40]!r}"
    return True, f"tokens={max_tokens} text={texts[0][:40]!r}"


def smoke_endpoint(parts, prompt, max_tokens, timeout, token=None):
    status, data = request(parts, "GET", "/v1/models", None, timeout, token)
    if status != 200:
        return [], f"/v1/models status {status}"
    models = json.loads(data)
    entries = models.get("data") or []
    model_ids = [entry.get("id") for entry in entries if entry.get("id")]
    if not model_ids:
        return [], "/v1/models has no model ids"
    results = []
    for model_id in model_ids:
        ok, detail = smoke_model(parts, model_id, prompt, max_tokens, timeout,
                                 token)
        results.append((model_id, ok, detail))
    return results, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", action="append", required=True)
    parser.add_argument("--prompt", default="The quick brown fox")
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--token", default=None,
                        help="bearer token when the endpoint requires auth")
    args = parser.parse_args()

    failures = 0
    for endpoint in args.endpoint:
        parts = parse_endpoint(endpoint)
        try:
            results, error = smoke_endpoint(parts, args.prompt, args.max_tokens,
                                            args.timeout, args.token)
        except Exception as error:
            results, error = [], repr(error)
        if error is not None:
            print(f"FAIL {endpoint} {error}")
            failures += 1
            continue
        for model_id, ok, detail in results:
            verdict = "PASS" if ok else "FAIL"
            print(f"{verdict} {endpoint} model={model_id} {detail}")
            if not ok:
                failures += 1
    print(f"model_api_smoke endpoints={len(args.endpoint)} failures={failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
