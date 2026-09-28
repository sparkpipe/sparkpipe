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


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


laguna = load("laguna_compsec17", "tools/laguna_compsec17.py")
bench = load("laguna_api_bench", "tools/laguna_api_bench.py")
FIXTURE = ROOT / "qualification/ds4_eval/quality-fixtures-glm5.3-flash.json"
GLM_TOKENIZER = ROOT / "qualification/ds4_eval/tokenizer/glm-5.3-flash-tokenizer.json"
REFERENCE = json.loads((ROOT / "model-families/laguna/reference_tokens.json").read_text())
SYSTEM = "You are a helpful, conversationally-fluent assistant made by Poolside. You are here to be helpful to users through natural language conversations."


def output_tokenizer(directory):
    path = Path(directory) / "laguna_tokenizer.json"
    path.write_text(json.dumps({
        "model": {"type": "BPE", "vocab": {"Answer": 40, ":": 41, "Ġ": 42, "1": 43, "0": 44, "Ċ": 45}, "merges": []},
        "added_tokens": [{"id": 19, "content": "</think>"}, {"id": 24, "content": "</assistant>"}],
    }))
    return path


class LagunaPrompt(unittest.TestCase):
    def test_thinking_off_matches_the_publisher_template(self):
        self.assertEqual(laguna.build_prompt("Q", "off"),
                         "〈|EOS|〉<system>" + SYSTEM + "</system>\n<user>Q</user>\n<assistant></think>")

    def test_thinking_on_matches_the_publisher_template(self):
        self.assertEqual(laguna.build_prompt("Q", "on"),
                         "〈|EOS|〉<system>" + SYSTEM + "</system>\n<user>Q</user>\n<assistant><think>")

    def test_unknown_thinking_mode_is_rejected(self):
        with self.assertRaises(KeyError):
            laguna.build_prompt("Q", "auto")

    def test_output_decoder_keeps_added_tokens_verbatim(self):
        with tempfile.TemporaryDirectory() as directory:
            decode = laguna.load_output_decoder(output_tokenizer(directory))
            self.assertEqual(decode([19, 45, 40, 41, 42, 43, 44, 24]), "</think>\nAnswer: 10</assistant>")


class ReferenceAcceptance(unittest.TestCase):
    def test_reference_tokens_are_accepted(self):
        for result in REFERENCE["results"]:
            self.assertEqual(bench.accepts(result, result["generated_token_ids"]), (True, None))

    def test_tie_step_accepts_either_fp32_candidate(self):
        for result in REFERENCE["results"]:
            strict = result["strict_steps"]
            if strict >= len(result["generated_token_ids"]):
                continue
            for candidate in result["steps"][strict]["fp32_top2_tokens"]:
                generated = list(result["generated_token_ids"])
                generated[strict] = candidate
                generated[strict + 1:] = [0] * (len(generated) - strict - 1)
                self.assertEqual(bench.accepts(result, generated), (True, None))

    def test_mismatch_before_the_tie_step_is_refused(self):
        for result in REFERENCE["results"]:
            generated = list(result["generated_token_ids"])
            generated[0] = generated[0] + 1
            self.assertEqual(bench.accepts(result, generated), (False, 0))

    def test_short_output_is_refused(self):
        result = REFERENCE["results"][0]
        self.assertEqual(bench.accepts(result, result["generated_token_ids"][:5]), (False, 5))


class Endpoint(BaseHTTPRequestHandler):
    prompts = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        Endpoint.prompts.append(body["prompt"])
        payload = {"tokens": [19, 45, 40, 41, 42, 43, 44], "status": "ok"}
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *arguments):
        pass


class LagunaRun(unittest.TestCase):
    def test_run_renders_every_case_and_grades_decoded_output(self):
        server = HTTPServer(("127.0.0.1", 0), Endpoint)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as directory:
                out = Path(directory) / "run"
                result = subprocess.run([sys.executable, str(ROOT / "tools/laguna_compsec17.py"),
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
            self.assertTrue(prompt.startswith("〈|EOS|〉<system>" + SYSTEM + "</system>\n<user>"))
            self.assertTrue(prompt.endswith("</user>\n<assistant></think>"))
            self.assertNotIn("[gMASK]", prompt)
        self.assertEqual(summary["parameters"]["chat_template"], "laguna")
        self.assertEqual(summary["completed"], 17)
        self.assertTrue(all(r["extracted"] == "10" for r in summary["results"]))


if __name__ == "__main__":
    result = unittest.main(exit=False, verbosity=1).result
    sys.exit(0 if result.wasSuccessful() else 1)
