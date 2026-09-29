#!/usr/bin/env python3
import argparse
import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("compsec17", ROOT / "tools/compsec17.py")
compsec = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compsec)
REFERENCE = ROOT / "qualification/gemma4/runs/20260928-hf-greedy-reference/reference.json"
FIXTURE = ROOT / "qualification/ds4_eval/quality-fixtures-glm5.3-flash.json"
TOKENIZER_UNAVAILABLE = {"error": {"message": "deployment has no tokenizer sidecar", "type": "invalid_request_error",
                                   "code": "tokenizer_unavailable"}}


class TokenIdOnlyApi(BaseHTTPRequestHandler):
    def reply(self, status, document):
        body = json.dumps(document).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self.reply(200, {"status": "ok", "served": 0, "tokenizer": self.server.tokenizer})
        else:
            self.reply(404, {"error": "not found"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        request = json.loads(self.rfile.read(length))
        self.server.completions.append(request)
        if "prompt" in request:
            self.reply(400, TOKENIZER_UNAVAILABLE)
        else:
            self.reply(200, {"choices": [{"text": "Answer: 1"}], "tokens": [1], "status": 0,
                             "usage": {"completion_tokens": 1}})

    def log_message(self, *arguments):
        pass


class FakeApiServer(ThreadingHTTPServer):
    request_queue_size = 64


@contextlib.contextmanager
def fake_api(tokenizer):
    server = FakeApiServer(("127.0.0.1", 0), TokenIdOnlyApi)
    server.tokenizer = tokenizer
    server.completions = []
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server, f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


def run_arguments(endpoint, out):
    return argparse.Namespace(endpoint=endpoint, fixture=str(FIXTURE), tokenizer="unused", out=str(out),
                              thinking="off", max_tokens=8, pass_threshold=14, concurrency=17,
                              temperature=0.0, timeout=30)


class TextEndpointPreflight(unittest.TestCase):
    def run_gate(self, tokenizer):
        with fake_api(tokenizer) as (server, endpoint), tempfile.TemporaryDirectory() as root:
            out = Path(root) / "run"
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr), contextlib.redirect_stdout(io.StringIO()):
                status = compsec.run(run_arguments(endpoint, out), lambda question, thinking: question, "glm",
                                     lambda ids: "question", lambda ids: "Answer: 1")
            return status, stderr.getvalue(), list(server.completions), out.exists()

    def test_api_without_tokenizer_fails_before_any_text_prompt(self):
        status, stderr, completions, created = self.run_gate(False)
        self.assertEqual(status, 2)
        self.assertEqual(completions, [])
        self.assertFalse(created)
        self.assertIn("tokenizer=False", stderr)
        self.assertIn("tokenizer_unavailable", stderr)

    def test_http_error_reports_the_api_error_body(self):
        status, stderr, completions, _ = self.run_gate(True)
        self.assertEqual(status, 2)
        self.assertTrue(completions)
        self.assertIn("HTTP 400", stderr)
        self.assertIn('"code": "tokenizer_unavailable"', stderr)

    def test_unreachable_endpoint_fails_the_preflight(self):
        with self.assertRaises(compsec.RunError):
            compsec.require_text_endpoint("http://127.0.0.1:9", 5)

    def test_api_with_tokenizer_passes_the_preflight(self):
        with fake_api(True) as (_, endpoint):
            self.assertIs(compsec.require_text_endpoint(endpoint)["tokenizer"], True)

    def test_concurrent_bench_refuses_an_api_without_tokenizer(self):
        bench_spec = importlib.util.spec_from_file_location("glm5_next_api_bench", ROOT / "tools/glm5_next_api_bench.py")
        bench = importlib.util.module_from_spec(bench_spec)
        bench_spec.loader.exec_module(bench)
        with fake_api(False) as (server, endpoint):
            bench.EP = endpoint + "/v1/completions"
            with self.assertRaises(bench.compsec17.RunError) as raised:
                bench.conc(argparse.Namespace(streams=[8], conc_tokens=8, warm=False))
            self.assertEqual(server.completions, [])
        self.assertIn("tokenizer=False", str(raised.exception))


class Gemma4ChatPrompt(unittest.TestCase):
    def test_thinking_off_matches_the_publisher_rendering(self):
        for result in json.loads(REFERENCE.read_text())["results"]:
            self.assertEqual(compsec.build_prompt(result["text"], "off", "gemma4"),
                             result["rendered_prompt"])

    def test_thinking_on_adds_the_think_system_turn(self):
        self.assertEqual(compsec.build_prompt("Q", "on", "gemma4"),
                         "<bos><|turn>system\n<|think|>\n<turn|>\n<|turn>user\nQ<turn|>\n<|turn>model\n")

    def test_unknown_template_is_rejected(self):
        with self.assertRaises(KeyError):
            compsec.build_prompt("Q", "off", "chatml")


if __name__ == "__main__":
    result = unittest.main(exit=False, verbosity=1).result
    sys.exit(0 if result.wasSuccessful() else 1)
