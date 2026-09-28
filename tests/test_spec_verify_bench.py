#!/usr/bin/env python3
import json
import struct
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import spec_verify_bench as bench


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/v1/chat/completions":
            reply = {"tokens": [[0.1, token + 1] for token in body["prompt_token_ids"][-3:]]}
        else:
            reply = {"choices": [{"text": body["prompt"][-12:]}], "tokens": [7, 8, 9]}
        data = json.dumps(reply).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


def check_log():
    lines = [
        "noise",
        "VERIFY-FRAME slot=0 position=100 budget=8 produced=8 rounds=1 accepted=7 | frames=1",
        "VERIFY-FRAME slot=1 position=108 budget=8 produced=5 rounds=3 accepted=2 | frames=2",
    ]
    report = bench.parse_log(lines)
    assert report["verify_frames"] == 2 and report["rounds"] == 4 and report["accepted_drafts"] == 9
    assert report["produced_tokens"] == 13 and abs(report["tokens_per_round"] - 13 / 4) < 1e-9
    assert abs(report["frame_fill"] - 13 / 16) < 1e-9


def check_compare():
    base = {"results": [{"class": "code", "index": 0, "wall_s": 2.0, "output_tokens": 100, "text": "a", "token_ids": [1, 2]},
                        {"class": "prose", "index": 0, "wall_s": 4.0, "output_tokens": 100, "text": "b", "token_ids": None}]}
    spec = json.loads(json.dumps(base))
    spec["results"][0]["wall_s"] = 1.0
    report = bench.compare_runs(base, spec)
    assert report["exact"] and abs(report["speedup"]["code"] - 2.0) < 1e-9 and abs(report["speedup"]["prose"] - 1.0) < 1e-9
    spec["results"][1]["text"] = "c"
    assert not bench.compare_runs(base, spec)["exact"]
    spec["results"][1]["text"] = "b"
    spec["results"][0]["token_ids"] = [1, 3]
    assert not bench.compare_runs(base, spec)["exact"]


def check_endpoint():
    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    endpoint = f"http://127.0.0.1:{server.server_port}"
    with tempfile.TemporaryDirectory() as directory:
        prompt = Path(directory) / "prompt.json"
        prompt.write_text(json.dumps([10, 20, 30, 40]))
        out = Path(directory) / "oracle.u32"
        assert bench.main(["record", "--endpoint", endpoint, "--prompt-ids", str(prompt), "--max-tokens", "3", "--out", str(out)]) == 0
        data = out.read_bytes()
        assert struct.unpack(f"<{len(data) // 4}I", data) == (10, 20, 30, 40, 21, 31, 41)
        results = Path(directory) / "run.json"
        assert bench.main(["run", "--endpoint", endpoint, "--classes", "code", "--max-tokens", "3", "--label", "t", "--out", str(results)]) == 0
        run = json.loads(results.read_text())
        assert len(run["results"]) == len(bench.PROMPTS["code"]) and run["results"][0]["token_ids"] == [7, 8, 9]
        assert bench.main(["compare", str(results), str(results)]) == 0
    server.shutdown()


def main():
    check_log()
    check_compare()
    check_endpoint()
    print("PASS spec_verify_bench records oracle sequences, runs content classes, checks exactness and parses VERIFY-FRAME logs")


if __name__ == "__main__":
    main()
