#!/usr/bin/env python3
"""Mock SparkPipe model_api upstream — the EXACT wire contract of
node/model_api.c, for verifying LiteLLM routing without touching a spark.

Contract (mirrors node/model_api.c api_connection/handle_completion):
  GET  /health            -> 200 {"status":"ok","served":N}
  POST /v1/completions    -> body {"prompt_token_ids":[int,...],"max_tokens":N}
  POST /v1/chat/completions  (same handler)
       -> 200 {"object":"text_completion","tokens":[int,...],"status":0}
  missing/empty prompt_token_ids -> 400 {"error":"prompt_token_ids required"}
  anything else           -> 404 {"error":"not found"}

No /v1/models route exists on the real daemon; the mock reproduces that 404.
Every request is logged as one JSON line to the log file so the exact bytes
LiteLLM forwarded can be inspected.

Usage: litellm_mock_upstream.py --port 9101 --log /tmp/mock-upstream.jsonl
"""
import argparse
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

API_MAX_OUTPUT_TOKENS = 8192


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    served = 0

    def log_message(self, fmt, *args):  # keep stdout clean
        pass

    def _reply(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _record(self, method, path, body):
        rec = {
            "ts": time.time(),
            "method": method,
            "path": path,
            "headers": dict(self.headers),
            "body": body,
        }
        with open(self.server.log_path, "a") as fh:
            fh.write(json.dumps(rec) + "\n")

    def do_GET(self):
        self._record("GET", self.path, None)
        if self.path == "/health":
            self._reply(200, {"status": "ok", "served": Handler.served})
        else:
            self._reply(404, {"error": "not found"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        try:
            body = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            body = None
        self._record("POST", self.path, body)
        if self.path not in ("/v1/completions", "/v1/chat/completions"):
            self._reply(404, {"error": "not found"})
            return
        if not isinstance(body, dict):
            self._reply(400, {"error": "invalid json"})
            return
        ids = body.get("prompt_token_ids")
        if not isinstance(ids, list) or len(ids) == 0:
            self._reply(400, {"error": "prompt_token_ids required"})
            return
        max_tokens = body.get("max_tokens") or 32
        max_tokens = min(int(max_tokens), API_MAX_OUTPUT_TOKENS)
        # deterministic pseudo-generation: derived from the prompt ids
        out = [(t * 7 + i + 1) % 151936 for i, t in enumerate(ids[:max_tokens])]
        Handler.served += 1
        self._reply(200, {"object": "text_completion", "tokens": out, "status": 0})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=9101)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--log", required=True)
    args = ap.parse_args()
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    srv.log_path = args.log
    print("mock model_api listening on %s:%d log=%s" %
          (args.host, args.port, args.log), flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    sys.exit(main())
