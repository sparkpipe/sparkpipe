#!/usr/bin/env python3
import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path


def post(base, path, body, key):
    headers = {"Content-Type": "application/json", "anthropic-version": "2023-06-01"}
    if key:
        headers["X-Api-Key"] = key
    request = urllib.request.Request(base.rstrip("/") + path, data=json.dumps(body).encode(), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=3600) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read() or b"{}")


def check(base, body, key):
    counted_status, counted = post(base, "/v1/messages/count_tokens", {name: value for name, value in body.items() if name not in ("stream", "max_tokens")}, key)
    status, reply = post(base, "/v1/messages", dict(body, stream=False, max_tokens=1), key)
    if counted_status != 200 or status != 200:
        return None, None, f"count_tokens {counted_status}, messages {status}: {json.dumps(counted)[:200]} {json.dumps(reply)[:200]}"
    usage = reply.get("usage") or {}
    prompt = usage.get("input_tokens", 0) + usage.get("cache_creation_input_tokens", 0) + usage.get("cache_read_input_tokens", 0)
    return counted.get("input_tokens"), prompt, None


def main(argv=None):
    parser = argparse.ArgumentParser(description="compare count_tokens with the engine's prompt length for recorded Messages requests")
    parser.add_argument("--base", required=True, help="the chat layer or gateway, for example http://127.0.0.1:8433")
    parser.add_argument("--key-file", help="a file holding the API key, for a base behind the public door")
    parser.add_argument("bodies", type=Path, help="a directory of recorded Messages request bodies, one JSON file each")
    arguments = parser.parse_args(argv)
    key = Path(arguments.key_file).read_text().strip() if arguments.key_file else None
    files = sorted(arguments.bodies.glob("*.json"))
    if not files:
        print(f"FAIL no *.json request bodies in {arguments.bodies}")
        return 1
    differ = 0
    for path in files:
        counted, prompt, error = check(arguments.base, json.loads(path.read_text()), key)
        same = error is None and counted == prompt
        differ += not same
        print(json.dumps({"file": path.name, "count_tokens": counted, "engine_prompt_tokens": prompt, "same": same, "error": error}))
    print(f"{'PASS' if differ == 0 else 'FAIL'} count_tokens equals the engine's prompt length for {len(files) - differ} of {len(files)} recorded requests")
    return 1 if differ else 0


if __name__ == "__main__":
    sys.exit(main())
