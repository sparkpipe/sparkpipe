#!/usr/bin/env python3
import http.server
import json
import pathlib
import sys
import threading
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import api_serving_perf


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    first = "token"

    def do_POST(self):
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        usage = {"prompt_tokens": len(request["prompt"].split()), "completion_tokens": 0}
        if Handler.first == "token":
            events = [{"tokens": [7], "choices": [{"text": "x", "finish_reason": None}]},
                      {"choices": [{"text": "", "finish_reason": "length"}], "usage": dict(usage, completion_tokens=1)}]
        else:
            events = [{"choices": [{"text": "", "finish_reason": "stop"}], "usage": usage}]
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "close")
        self.end_headers()
        time.sleep(0.05)
        for event in events:
            self.wfile.write(b"data: " + json.dumps(event).encode() + b"\n\n")
            self.wfile.flush()
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()
        self.close_connection = True

    def log_message(self, *arguments):
        pass


def main():
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    endpoint = f"http://127.0.0.1:{server.server_address[1]}"
    failures = []
    try:
        Handler.first = "token"
        token = api_serving_perf.run_ttft(endpoint, 200, 1)
        if not token["prefill_tok_s"] or token["repeats"][0].get("first_token_stop"):
            failures.append(f"a first content token must time the prefill: {token}")
        Handler.first = "stop"
        stop = api_serving_perf.run_ttft(endpoint, 200, 1)
        record = stop["repeats"][0]
        if not stop["prefill_tok_s"] or not record.get("first_token_stop") or record.get("ttft_ms", 0.0) < 40.0:
            failures.append(f"a first token that stops the reply must still time the prefill, at the stop: {stop}")
    finally:
        server.shutdown()
    for failure in failures:
        print("FAIL", failure)
    if failures:
        return 1
    print("PASS api serving perf: the time to first token is the first content token, or the stop when the model's first token ends the reply")
    return 0


if __name__ == "__main__":
    sys.exit(main())
