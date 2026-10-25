#!/usr/bin/env python3
import hashlib
import json
import math
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "ab_api_score.py"
SEEN = []


def answer(prompt, shift):
    last = prompt[-1]
    top = [[(last + shift) % 50, math.log(0.6)], [(last + 1) % 50, math.log(0.3)], [(last + 2) % 50, math.log(0.05)]]
    return [top[0]] + top


class Fake(BaseHTTPRequestHandler):
    shift = 1

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        SEEN.append((self.path, body))
        reply = {"tokens": [0], "token_logprobs": [answer(body["prompt_token_ids"], Fake.shift)], "usage": {"prompt_tokens": len(body["prompt_token_ids"])}}
        data = json.dumps(reply).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


def corpus(directory):
    documents, raw = [], b""
    for index in range(3):
        tokens = [(index * 7 + i) % 50 for i in range(40)]
        body = b"".join(t.to_bytes(4, "little") for t in tokens)
        documents.append({"id": f"doc{index}", "stratum": "code" if index else "prose", "offset": index * 40, "length": 40,
                          "tokens_sha256": hashlib.sha256(body).hexdigest()})
        raw += body
    tokens_path = directory / "c.tokens.u32"
    tokens_path.write_bytes(raw)
    index_path = directory / "c.index.json"
    index_path.write_text(json.dumps({"format": "sparkpipe-ab-corpus-v1", "corpus_sha256": hashlib.sha256(raw).hexdigest(), "documents": documents}))
    return index_path, tokens_path


def run(*args, check=True):
    return subprocess.run([sys.executable, str(TOOL), *args], capture_output=True, text=True, check=check)


def main():
    server = HTTPServer(("127.0.0.1", 0), Fake)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    endpoint = f"http://127.0.0.1:{server.server_address[1]}"
    with tempfile.TemporaryDirectory() as name:
        directory = Path(name)
        index_path, tokens_path = corpus(directory)
        common = ["--endpoint", endpoint, "--index", str(index_path), "--tokens", str(tokens_path), "--first", "8", "--stride", "4", "--count", "5"]
        a = directory / "a.jsonl"
        run("score", *common, "--documents", "2", "--out", str(a))
        assert len(a.read_text().splitlines()) == 10
        run("score", *common, "--out", str(a))
        rows = [json.loads(line) for line in a.read_text().splitlines()]
        assert len(rows) == 15 and len(SEEN) == 15, (len(rows), len(SEEN))
        path, body = SEEN[0]
        assert path == "/v1/completions" and body["max_tokens"] == 1 and body["temperature"] == 0 and body["logprobs"] == 20 and body["speculation"] is False
        assert body["prompt_token_ids"] == list(range(8)) and rows[0]["truth"] == 8 and rows[0]["position"] == 8
        Fake.shift = 3
        b = directory / "b.jsonl"
        run("score", *common, "--out", str(b))
        same = json.loads(run("compare", str(a), str(a)).stdout)
        assert same["identical_lists"] == 15 and same["overall"]["top1_agreement"] == 1.0 and same["overall"]["kl_lower_bound"] == 0.0
        report = json.loads(run("compare", str(a), str(b), "--out", str(directory / "r.json")).stdout)
        overall = report["overall"]
        assert report["identical_lists"] == 0 and overall["top1_agreement"] == 0.0 and overall["positions"] == 15
        assert overall["reference_accuracy"] == 1.0 and overall["arm_accuracy"] == 0.0 and overall["accuracy_only_reference"] == 15
        assert overall["accuracy_mcnemar_p"] < 0.001 and overall["kl_lower_bound"] > 0.0 and set(report["strata"]) == {"code", "prose"}
        assert overall["truth_logprob_arm_minus_reference"] < 0.0
        bad = directory / "bad.tokens.u32"
        bad.write_bytes(tokens_path.read_bytes()[:-4] + b"\x01\x00\x00\x00")
        refused = run("score", "--endpoint", endpoint, "--index", str(index_path), "--tokens", str(bad), "--out", str(directory / "x.jsonl"), check=False)
        assert refused.returncode != 0 and "corpus_sha256" in refused.stderr
        short = run("compare", str(a), str(directory / "r.json"), check=False)
        assert short.returncode != 0
    server.shutdown()
    print("PASS ab_api_score: exact one-token greedy requests with 20 log-probs, resumed runs append only missing positions, corpus SHA refusal, agreement, bounded KL, accuracy and McNemar per stratum")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
