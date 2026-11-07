#!/usr/bin/env python3
import json
import pathlib
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

ROOT = pathlib.Path(__file__).resolve().parent.parent
TOOL = ROOT / "tools" / "release_long_fill.py"


class Handler(BaseHTTPRequestHandler):
    delay = 0.0

    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        time.sleep(Handler.delay)
        body = json.dumps({"usage": {"prompt_tokens_details": {"cached_tokens": 0}}}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def run(port, best):
    return subprocess.run([sys.executable, str(TOOL), f"http://127.0.0.1:{port}", "m", "--tokens", "1000", "--best", str(best)], capture_output=True, text=True)


def main():
    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_address[1]
    failures = []
    with tempfile.TemporaryDirectory() as scratch:
        best = pathlib.Path(scratch) / "best.json"
        Handler.delay = 0.1
        first = run(port, best)
        recorded = json.loads(best.read_text()).get("m", 0.0) if best.exists() else 0.0
        if first.returncode != 0 or recorded <= 0.0:
            failures.append(f"a first fill must pass and record its rate: {first.stdout} {first.stderr}")
        Handler.delay = 1.0
        slow = run(port, best)
        if slow.returncode != 3 or "LONG-FILL-REGRESSION" not in slow.stdout:
            failures.append(f"a fill at a tenth of the fastest must fail: rc {slow.returncode} {slow.stdout}")
        if json.loads(best.read_text()).get("m") != recorded:
            failures.append("a slower fill must not change the fastest rate")
        Handler.delay = 0.01
        fast = run(port, best)
        if fast.returncode != 0 or json.loads(best.read_text()).get("m", 0.0) <= recorded:
            failures.append(f"a faster fill must raise the fastest rate: {fast.stdout}")
    server.shutdown()
    if failures:
        print("FAIL " + "; ".join(failures))
        return 1
    print("PASS release long fill: a slow fill fails, a fast one raises the record")
    return 0


if __name__ == "__main__":
    sys.exit(main())
