#!/usr/bin/env python3
"""ds4_eval92.py --order submits the 92 cases as listed, reversed or shuffled by a seed, and the summary keeps the cases in their listed order."""

import json
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "ds4_eval92.py"
CASES = ROOT / "qualification" / "ds4_eval" / "runs" / "kimi-k3-api-20260728" / "cases.json"
FAILURES = []


def check(condition, message):
    print(("PASS " if condition else "FAIL ") + message)
    if not condition:
        FAILURES.append(message)


class Endpoint:
    def __init__(self):
        self.prompts = []
        seen = self.prompts

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                seen.append(body["messages"][-1]["content"])
                reply = json.dumps({"choices": [{"message": {"content": "Answer: B"}, "finish_reason": "stop"}],
                                    "usage": {"prompt_tokens": 7, "completion_tokens": 3}}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(reply)))
                self.end_headers()
                self.wfile.write(reply)

            def log_message(self, *arguments):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def run(endpoint, out, *arguments):
    endpoint.prompts.clear()
    done = subprocess.run([sys.executable, str(TOOL), "--endpoint", endpoint.url, "--model", "fake", "--max-tokens", "8", "--temperature", "0",
                           "--phases", "batch", "--concurrency", "1", "--out", str(out), *arguments], capture_output=True, text=True)
    return done, list(endpoint.prompts), json.loads((out / "batch" / "summary.json").read_text()) if done.returncode == 0 else None


def main():
    cases = json.loads(CASES.read_text())["cases"]
    listed = [case["rendered_prompt"] for case in cases]
    check(len(set(listed)) == 92, "the suite's 92 prompts are distinct, so a request names its case")
    endpoint = Endpoint()
    try:
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            done, seen, summary = run(endpoint, work / "listed")
            check(done.returncode == 0 and seen == listed and summary["order"] == "listed" and summary["order_seed"] is None,
                  "with no --order the cases are submitted as listed")
            done, seen, summary = run(endpoint, work / "reversed", "--order", "reversed")
            check(done.returncode == 0 and seen == listed[::-1] and summary["order"] == "reversed",
                  "--order reversed submits the last case first")
            check([record["index"] for record in summary["results"]] == [case["index"] for case in cases] and summary["completed"] == 92,
                  "the summary lists the cases in their listed order whatever the order they ran in")
            done, first, summary = run(endpoint, work / "shuffled-a", "--order", "shuffled", "--order-seed", "7")
            done_again, second, _ = run(endpoint, work / "shuffled-b", "--order", "shuffled", "--order-seed", "7")
            _, other, _ = run(endpoint, work / "shuffled-c", "--order", "shuffled", "--order-seed", "8")
            check(done.returncode == 0 and done_again.returncode == 0 and first == second and sorted(first) == sorted(listed)
                  and first not in (listed, listed[::-1]) and other != first and summary["order"] == "shuffled" and summary["order_seed"] == 7,
                  "--order shuffled submits every case once in an order the seed fixes")
            report = (work / "shuffled-a" / "REPORT.md").read_text()
            check("order `shuffled` (seed 7)" in report, "the report names the order")
            refused = subprocess.run([sys.executable, str(TOOL), "--endpoint", endpoint.url, "--model", "fake", "--max-tokens", "8", "--temperature", "0",
                                      "--out", str(work / "bad"), "--order", "sideways"], capture_output=True, text=True)
            check(refused.returncode != 0 and not endpoint.prompts[92:], "an unknown order is refused before any request")
    finally:
        endpoint.close()
    print(f"{'PASS' if not FAILURES else 'FAIL'} ds4_eval92 order: {len(FAILURES)} failures")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
