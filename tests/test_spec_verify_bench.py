#!/usr/bin/env python3
import json
import struct
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import spec_verify_bench as bench


class Handler(BaseHTTPRequestHandler):
    omit_tokens = False
    wrong_token = False

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/v1/chat/completions":
            ids = [token + 1 for token in body["prompt_token_ids"][-3:]][:body["max_tokens"]]
            if Handler.wrong_token:
                ids[-1] += 1
        else:
            ids = [7, 8, 9]
        if not body.get("stream"):
            data = json.dumps({"tokens": [[0.1, token] for token in ids]}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        time.sleep(0.2)
        for token in ids:
            event = {"choices": [{"text": f"<{token}>"}]}
            if not Handler.omit_tokens:
                event["tokens"] = [token]
            try:
                self.wfile.write(("data: " + json.dumps(event) + "\n\n").encode())
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                return
            time.sleep(0.01)
        self.wfile.write(b"data: [DONE]\n\n")

    def log_message(self, *args):
        pass


def check_log():
    lines = [
        "noise",
        "VERIFY-FRAME slot=0 position=100 budget=8 produced=8 rounds=1 accepted=7 | frames=1",
        "VERIFY-FRAME slot=1 position=108 budget=8 produced=5 rounds=3 accepted=2 | frames=2",
        "VERIFY-FRAME slot=0 position=113 budget=8 produced=8 rounds=2 accepted=3 steps=3 | frames=3",
    ]
    report = bench.parse_log(lines)
    assert report["verify_frames"] == 3 and report["rounds"] == 6 and report["accepted_drafts"] == 12 and report["plain_steps"] == 3
    assert report["produced_tokens"] == 21 and abs(report["tokens_per_round"] - 18 / 6) < 1e-9
    assert abs(report["frame_fill"] - 21 / 24) < 1e-9
    assert "sources" not in report
    lines.append("VERIFY-MTP drafts=3 tokens=12 cold=1 truncated=0 taps=9 draft_us=1500 | lookup rounds=2 proposed=6 accepted=5 declined=1 | mtp rounds=3 proposed=12 accepted=4")
    lines.append("VERIFY-MTP drafts=4 tokens=19 cold=1 truncated=1 taps=12 draft_us=1400 | lookup rounds=2 proposed=6 accepted=5 declined=1 | mtp rounds=4 proposed=19 accepted=9")
    sources = bench.parse_log(lines)["sources"]
    assert sources["mtp_drafts"] == 4 and sources["mtp_truncated"] == 1 and sources["mtp_draft_us"] == 1400 and sources["lookup_declined"] == 1
    assert sources["mtp"]["rounds"] == 4 and abs(sources["mtp"]["acceptance"] - 9 / 19) < 1e-9 and abs(sources["mtp"]["accept_length"] - 9 / 4) < 1e-9
    assert abs(sources["lookup"]["tokens_per_round"] - 7 / 2) < 1e-9
    assert "acceptance_per_position" not in bench.parse_log(lines)
    lines.append("VERIFY-POSITIONS p1=5/8 p2=2/5 p3=0/2 p4=0/0 p5=0/0 p6=0/0 p7=0/0")
    lines.append("VERIFY-POSITIONS p1=9/12 p2=4/9 p3=1/4 p4=0/1 p5=0/0 p6=0/0 p7=0/0")
    positions = bench.parse_log(lines)["acceptance_per_position"]
    assert [item["position"] for item in positions] == list(range(1, 8))
    assert [(item["accepted"], item["reached"]) for item in positions[:4]] == [(9, 12), (4, 9), (1, 4), (0, 1)]
    assert abs(positions[0]["acceptance"] - 0.75) < 1e-9 and positions[3]["acceptance"] == 0.0 and positions[4]["acceptance"] is None


def entry(klass, tokens, text, decode_s, ids):
    return {"class": klass, "index": 0, "decode_tokens": tokens, "decode_s": decode_s, "text": text, "token_ids": ids}


def check_compare():
    base = {"results": [entry("code", 100, "a", 2.0, [1, 2]), entry("prose", 100, "b", 4.0, [3])]}
    spec = json.loads(json.dumps(base))
    spec["results"][0]["decode_s"] = 1.0
    report = bench.compare_runs(base, spec)
    assert report["exact"] and abs(report["speedup"]["code"] - 2.0) < 1e-9 and abs(report["speedup"]["prose"] - 1.0) < 1e-9
    spec["results"][1]["text"] = "c"
    assert not bench.compare_runs(base, spec)["exact"]
    spec["results"][1]["text"] = "b"
    spec["results"][0]["token_ids"] = [1, 3]
    assert not bench.compare_runs(base, spec)["exact"]
    spec["results"][0]["token_ids"] = [1, 2]
    spec["results"][1]["token_ids"] = None
    report = bench.compare_runs(base, spec)
    assert not report["exact"] and "no token ids" in report["mismatches"][0]
    base["results"][1]["token_ids"] = None
    assert not bench.compare_runs(base, spec)["exact"]


def refused(argv):
    try:
        bench.main(argv)
    except SystemExit as error:
        return error.code != 0
    return False


def check_endpoint():
    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    endpoint = f"http://127.0.0.1:{server.server_port}"
    with tempfile.TemporaryDirectory() as directory:
        prompt = Path(directory) / "prompt.json"
        prompt.write_text(json.dumps([10, 20, 30, 40]))
        out = Path(directory) / "oracle.u32"
        assert refused(["record", "--prompt-ids", str(prompt), "--out", str(out)])
        assert refused(["run", "--label", "t", "--out", str(out)])
        assert bench.main(["record", "--endpoint", endpoint, "--prompt-ids", str(prompt), "--max-tokens", "3", "--out", str(out)]) == 0
        data = out.read_bytes()
        assert struct.unpack(f"<{len(data) // 4}I", data) == (10, 20, 30, 40, 21, 31, 41)
        assert bench.main(["replay", "--endpoint", endpoint, "--prompt-ids", str(prompt), "--expect", str(out)]) == 0
        Handler.wrong_token = True
        assert bench.main(["replay", "--endpoint", endpoint, "--prompt-ids", str(prompt), "--expect", str(out)]) == 1
        Handler.wrong_token = False
        results = Path(directory) / "run.json"
        assert bench.main(["run", "--endpoint", endpoint, "--classes", "code", "--max-tokens", "3", "--label", "t", "--out", str(results)]) == 0
        run = json.loads(results.read_text())
        first = run["results"][0]
        assert len(run["results"]) == len(bench.PROMPTS["code"]) and first["token_ids"] == [7, 8, 9] and first["text"] == "<7><8><9>"
        assert first["decode_tokens"] == 2 and first["ttft_s"] >= 0.0 and first["decode_s"] < first["wall_s"] - first["ttft_s"] + 1e-9
        assert first["ttft_s"] >= 0.2 and first["decode_s"] < 0.1 and first["decode_tok_s"] > 2 / 0.1
        assert bench.main(["compare", str(results), str(results)]) == 0
        Handler.omit_tokens = True
        assert refused(["run", "--endpoint", endpoint, "--classes", "code", "--max-tokens", "3", "--label", "t", "--out", str(results)])
        Handler.omit_tokens = False
    server.shutdown()


def main():
    check_log()
    check_compare()
    check_endpoint()
    print("PASS spec_verify_bench needs an explicit endpoint and token ids, replays recorded oracle sequences, measures decode rate after the first token and parses VERIFY-FRAME and VERIFY-MTP logs")


if __name__ == "__main__":
    main()
