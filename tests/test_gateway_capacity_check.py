#!/usr/bin/env python3
import http.server
import importlib.util
import json
import math
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location("gateway_capacity_check", ROOT / "tools" / "gateway_capacity_check.py")
check_tool = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(check_tool)


class Fake(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length)
        server = self.server
        with server.lock:
            server.count += 1
            server.keys.add(self.headers.get("Authorization"))
            server.bodies.add(body)
            refuse = server.refuse_every and server.count % server.refuse_every == 0
        time.sleep(server.delay)
        status = 429 if refuse else 200
        payload = b'{"ok": true}'
        self.send_response(status)
        if refuse:
            self.send_header("Retry-After", "60")
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def serve(delay, refuse_every=0):
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Fake)
    server.daemon_threads = True
    server.lock = threading.Lock()
    server.count, server.keys, server.bodies = 0, set(), set()
    server.delay, server.refuse_every = delay, refuse_every
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def main():
    failures = []
    with tempfile.TemporaryDirectory() as directory:
        work = Path(directory)
        (work / "key").write_text("sk-hc-test-key\n")
        (work / "body.json").write_text(json.dumps({"model": "kimi-k3", "max_tokens": 1, "messages": [{"role": "user", "content": "hi"}]}))
        server = serve(0.005)
        url = f"http://127.0.0.1:{server.server_address[1]}/v1/chat/completions"
        out = work / "result.json"
        code = check_tool.main([url, "--key-file", str(work / "key"), "--body-file", str(work / "body.json"), "--concurrency", "4", "--seconds", "1.5", "--out", str(out)])
        result = json.loads(out.read_text())
        if code != 0 or result["ok"] != result["requests"] or result["ok"] < 50 or result["throttled"] != 0:
            failures.append(f"a gateway that answers every request yields a clean measurement: exit {code}, {result}")
        if abs(result["requests_per_second"] - result["ok"] / result["seconds"]) > 0.01 * result["requests_per_second"] + 0.01:
            failures.append(f"the rate is the 2xx answers over the elapsed time: {result}")
        rounding = 0.0005 * 600
        window = result["requests_per_second"] * 600
        if not math.floor(window - rounding) <= result["capacity_per_window"] <= math.floor(window + rounding) or result["window_seconds"] != 600:
            failures.append(f"capacity_per_window is the sustained rate over the gateway's 600 s window: {result}")
        if not (result["p99_ms"] >= result["p50_ms"] >= 5.0):
            failures.append(f"p50 and p99 latency are measured per request and ordered: {result}")
        if server.keys != {"Bearer sk-hc-test-key"} or len(server.bodies) != 1 or result["requests"] != server.count:
            failures.append(f"every request carries the key from the key file and the body file, and none is left uncounted: {server.keys} {result['requests']} {server.count}")
        server.shutdown()

        refusing = serve(0.0, refuse_every=5)
        url = f"http://127.0.0.1:{refusing.server_address[1]}/v1/chat/completions"
        code = check_tool.main([url, "--concurrency", "2", "--seconds", "0.5", "--out", str(out)])
        result = json.loads(out.read_text())
        if code != 1 or result["throttled"] == 0 or result["statuses"].get("429") != result["throttled"]:
            failures.append(f"a measurement that met 429s is reported as throttled and fails, so a capped number is never taken as capacity: exit {code}, {result}")
        refusing.shutdown()

        out.unlink()
        for production in ("https://sparkpipe.ai/v1/chat/completions", "https://api.sparkpipe.ai/v1/messages"):
            code = check_tool.main([production, "--seconds", "0.1", "--out", str(out)])
            if code != 2 or out.exists():
                failures.append(f"a production URL is refused before any request: {production} exit {code}")
        try:
            check_tool.target("http://staging.sparkpipe.ai/v1/messages")
            check_tool.target("https://sparkpipe.ai/pay/sandbox/v1/messages")
        except check_tool.Refused as error:
            failures.append(f"staging and sandbox gateways are accepted: {error}")
    for failure in failures:
        print("FAIL test_gateway_capacity_check:", failure)
    if failures:
        raise SystemExit(1)
    print("PASS test_gateway_capacity_check: sustained 2xx requests per second, p50 and p99 latency and capacity_per_window over 600 s, written as JSON; "
          "429s fail the measurement; production URLs are refused before any request")


if __name__ == "__main__":
    main()
