#!/usr/bin/env python3
import importlib.util
import json
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("ling_compsec17", ROOT / "tools/ling_compsec17.py")
ling = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ling)
FIXTURE = ROOT / "qualification/ds4_eval/quality-fixtures-glm5.3-flash.json"
GLM_TOKENIZER = ROOT / "qualification/ds4_eval/tokenizer/glm-5.3-flash-tokenizer.json"


def output_tokenizer(directory):
    path = Path(directory) / "ling_tokenizer.json"
    path.write_text(json.dumps({
        "model": {"type": "BPE", "vocab": {"Answer": 0, ":": 1, "Ġ": 2, "1": 3, "0": 4, "Ċ": 5}, "merges": []},
        "added_tokens": [{"id": 7, "content": "</think>"}, {"id": 8, "content": "<|role_end|>"}],
    }))
    return path


class LingPrompt(unittest.TestCase):
    def test_thinking_off_matches_the_publisher_template(self):
        self.assertEqual(ling.build_prompt("Q", "off"),
                         "<role>SYSTEM</role>detailed thinking off<|role_end|><role>HUMAN</role>Q<|role_end|><role>ASSISTANT</role>\n<think></think>")

    def test_thinking_on_matches_the_publisher_template(self):
        self.assertEqual(ling.build_prompt("Q", "on"),
                         "<role>SYSTEM</role>detailed thinking on<|role_end|><role>HUMAN</role>Q<|role_end|><role>ASSISTANT</role>\n<think>")

    def test_unknown_thinking_mode_is_rejected(self):
        with self.assertRaises(KeyError):
            ling.build_prompt("Q", "auto")

    def test_output_decoder_keeps_added_tokens_verbatim(self):
        with tempfile.TemporaryDirectory() as directory:
            decode = ling.load_output_decoder(output_tokenizer(directory))
            self.assertEqual(decode([7, 5, 0, 1, 2, 3, 4, 8]), "</think>\nAnswer: 10<|role_end|>")


class Endpoint(BaseHTTPRequestHandler):
    prompts = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        Endpoint.prompts.append(body["prompt"])
        payload = {"tokens": [7, 5, 0, 1, 2, 3, 4], "status": "ok"} if len(Endpoint.prompts) % 2 else {"choices": [{"text": "</think>\nAnswer: 10"}], "tokens": [1, 2], "status": "ok"}
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *arguments):
        pass


class LingRun(unittest.TestCase):
    def test_run_renders_every_case_and_grades_decoded_output(self):
        server = HTTPServer(("127.0.0.1", 0), Endpoint)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as directory:
                out = Path(directory) / "run"
                result = subprocess.run([sys.executable, str(ROOT / "tools/ling_compsec17.py"),
                                         "--endpoint", f"http://127.0.0.1:{server.server_port}",
                                         "--fixture", str(FIXTURE), "--tokenizer", str(GLM_TOKENIZER),
                                         "--output-tokenizer", str(output_tokenizer(directory)),
                                         "--out", str(out), "--thinking", "off", "--pass-threshold", "0"],
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                summary = json.loads((out / "summary.json").read_text())
        finally:
            server.shutdown()
        self.assertEqual(len(Endpoint.prompts), 17)
        for prompt in Endpoint.prompts:
            self.assertTrue(prompt.startswith("<role>SYSTEM</role>detailed thinking off<|role_end|><role>HUMAN</role>"))
            self.assertTrue(prompt.endswith("<|role_end|><role>ASSISTANT</role>\n<think></think>"))
            self.assertNotIn("[gMASK]", prompt)
        cases = sorted((c for c in json.loads(FIXTURE.read_text())["cases"] if c["id"].startswith("compsec")), key=lambda c: c["id"])
        expected = sum(1 for c in cases if ling.compsec.grade("Answer: 10", c["answer"])[0])
        self.assertEqual(summary["parameters"]["chat_template"], "ling")
        self.assertEqual(summary["completed"], 17)
        self.assertEqual(summary["passed"], expected)
        self.assertTrue(all(r["extracted"] == "10" for r in summary["results"]))


if __name__ == "__main__":
    result = unittest.main(exit=False, verbosity=1).result
    sys.exit(0 if result.wasSuccessful() else 1)
