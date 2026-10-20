#!/usr/bin/env python3
import hashlib
import importlib.util
import io
import json
import threading
import contextlib
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("i27", ROOT / "tools/i27_session.py")
i27 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(i27)


class FakeEngine:
    def __init__(self, capacity, corrupt_on_hit=False, caches=True):
        self.capacity = capacity
        self.corrupt_on_hit = corrupt_on_hit
        self.caches = caches
        self.prompts = []
        self.lock = threading.Lock()

    def complete(self, prompt, max_tokens):
        with self.lock:
            cached = 0
            for stored in self.prompts:
                common = 0
                while common < min(len(stored), len(prompt)) - 0 and stored[common] == prompt[common]:
                    common += 1
                cached = max(cached, min(common, len(prompt) - 1))
            if self.caches:
                if prompt in self.prompts:
                    self.prompts.remove(prompt)
                self.prompts.append(list(prompt))
                del self.prompts[:-self.capacity]
            else:
                cached = 0
        digest = hashlib.sha256(json.dumps(prompt).encode()).digest()
        tokens = [digest[index % 32] + 1000 for index in range(max_tokens)]
        if self.corrupt_on_hit and cached > 0:
            tokens[-1] += 1
        return tokens, cached


def serve(engine):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            self.send_response(200 if self.path == "/health" else 404)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if body.get("stream"):
                return
            tokens, cached = engine.complete(body["prompt_token_ids"], body["max_tokens"])
            reply = json.dumps({"object": "text_completion", "tokens": tokens,
                                "usage": {"prompt_tokens": len(body["prompt_token_ids"]), "completion_tokens": len(tokens),
                                          "prompt_tokens_details": {"cached_tokens": cached}}}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(reply)))
            self.end_headers()
            self.wfile.write(reply)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


class I27SessionTests(unittest.TestCase):
    def run_session(self, engine, *extra):
        server = serve(engine)
        try:
            arguments = ["--api", "http://127.0.0.1:%d" % server.server_address[1], "--vocab-size", "50000",
                         "--block-tokens", "16", "--prompt-tokens", "64", "--long-prompt-tokens", "128",
                         "--max-tokens", "6", "--concurrency", "4", "--evict-prompts", "8", "--evict-tokens", "64",
                         "--abort-seconds", "0.01", "--seed", "fixed", *extra]
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                status = i27.main(arguments)
            receipt = json.loads(output.getvalue().rsplit("verdict=", 1)[0])
            return status, receipt
        finally:
            server.shutdown()
            server.server_close()

    def test_correct_engine_passes_every_case_with_hooks(self):
        status, receipt = self.run_session(FakeEngine(6), "--restart-command", "true", "--writeback-fault-command", "true")
        self.assertEqual(receipt["verdict"], "PASS", receipt)
        self.assertEqual(status, 0)
        self.assertEqual(set(receipt["cases"]), set(i27.CASES))

    def test_missing_hooks_leave_the_session_incomplete(self):
        status, receipt = self.run_session(FakeEngine(6))
        self.assertEqual((status, receipt["verdict"]), (2, "INCOMPLETE"))
        self.assertEqual(receipt["cases"]["restart"]["verdict"], "NOT-RUN")

    def test_tokens_that_change_on_a_hit_fail(self):
        status, receipt = self.run_session(FakeEngine(6, corrupt_on_hit=True), "--cases", "b1,b16,cow_mid_block")
        self.assertEqual((status, receipt["verdict"]), (1, "FAIL"))
        self.assertFalse(receipt["cases"]["b1"]["identical"])
        self.assertTrue(receipt["cases"]["b16"]["mismatched"])

    def test_an_engine_that_never_reuses_fails(self):
        status, receipt = self.run_session(FakeEngine(6, caches=False), "--cases", "b1,evict_recompute")
        self.assertEqual(status, 1)
        self.assertEqual(receipt["cases"]["b1"]["cached_tokens"], 0)

    def test_an_unevictable_cache_fails_instead_of_using_a_warm_control(self):
        status, receipt = self.run_session(FakeEngine(10000), "--cases", "evict_recompute")
        self.assertEqual(status, 1)
        self.assertEqual(receipt["cases"]["evict_recompute"]["verdict"], "FAIL")


if __name__ == "__main__":
    unittest.main()
