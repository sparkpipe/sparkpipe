#!/usr/bin/env python3
import argparse
import base64
import json
import sys
import time
import urllib.error
import urllib.request

FIELDS = {"name", "method", "path", "body", "timeout_s", "status", "json_equals", "text_path", "contains", "not_contains", "body_contains"}


class SpecError(Exception):
    pass


def check_spec(spec):
    reqs = spec.get("requests")
    if not isinstance(reqs, list) or not reqs:
        raise SpecError("smoke spec needs a non-empty requests list")
    for i, r in enumerate(reqs):
        unknown = set(r) - FIELDS
        if unknown:
            raise SpecError(f"request {i}: unknown fields {sorted(unknown)}")
        if not r.get("name") or not r.get("path"):
            raise SpecError(f"request {i}: name and path are required")
        if r.get("method", "POST" if "body" in r else "GET") not in ("GET", "POST"):
            raise SpecError(f"request {r['name']}: method must be GET or POST")
        if ("contains" in r or "not_contains" in r) and "text_path" not in r:
            raise SpecError(f"request {r['name']}: contains/not_contains need text_path")
    return reqs


def dig(doc, path):
    cur = doc
    for part in path.split("."):
        if isinstance(cur, list) and part.isdigit() and int(part) < len(cur):
            cur = cur[int(part)]
        elif isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def call(base, req):
    method = req.get("method", "POST" if "body" in req else "GET")
    data = json.dumps(req["body"]).encode() if "body" in req else None
    r = urllib.request.Request(base + req["path"], data=data, method=method, headers={"Content-Type": "application/json"})
    t = time.time()
    try:
        with urllib.request.urlopen(r, timeout=req.get("timeout_s", 900)) as resp:
            return resp.status, resp.read().decode(errors="replace"), time.time() - t
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors="replace"), time.time() - t


def judge(req, status, body):
    problems = []
    want = req.get("status", 200)
    if status != want:
        problems.append(f"HTTP {status} want {want}")
    doc = None
    try:
        doc = json.loads(body)
    except ValueError:
        pass
    for path, value in req.get("json_equals", {}).items():
        got = dig(doc, path) if doc is not None else None
        if got != value:
            problems.append(f"{path}={json.dumps(got)} want {json.dumps(value)}")
    text = None
    if "text_path" in req:
        text = dig(doc, req["text_path"]) if doc is not None else None
        if not isinstance(text, str):
            problems.append(f"{req['text_path']} is not text")
            text = ""
    for s in req.get("contains", []):
        if s not in text:
            problems.append(f"missing {s!r}")
    for s in req.get("not_contains", []):
        if s in text:
            problems.append(f"contains {s!r}")
    for s in req.get("body_contains", []):
        if s not in body:
            problems.append(f"body missing {s!r}")
    return problems, text


def run(base, spec, out=print):
    bad = 0
    for req in check_spec(spec):
        try:
            status, body, secs = call(base, req)
        except (OSError, urllib.error.URLError) as e:
            out(f"FAIL {req['name']}: {e}")
            bad += 1
            continue
        problems, text = judge(req, status, body)
        shown = (text if text is not None else body).strip().replace("\n", " ")[:120]
        out(f"{'OK  ' if not problems else 'FAIL'} {req['name']}: HTTP {status} {secs:.2f}s {shown!r}" + (f"  <- {'; '.join(problems)}" if problems else ""))
        bad += bool(problems)
    out(f"HUB-SMOKE {'PASS' if not bad else f'FAIL {bad}'}")
    return 1 if bad else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="Send the smoke requests of a release checks file to the API and judge the replies.")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--host", default="127.0.0.1")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--spec")
    src.add_argument("--spec-b64")
    args = ap.parse_args(argv)
    try:
        if args.spec:
            with open(args.spec) as f:
                spec = json.load(f)
        else:
            spec = json.loads(base64.b64decode(args.spec_b64))
        return run(f"http://{args.host}:{args.port}", spec)
    except (OSError, ValueError, SpecError) as e:
        print(f"hub_smoke.py: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
