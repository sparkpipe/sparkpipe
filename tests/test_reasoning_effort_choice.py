#!/usr/bin/env python3
import http.server
import json
import pathlib
import sys
import threading

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import compsec_effort
import ui_smoke

LISTINGS = {
    "/chat": {"data": [{"id": "thinker", "reasoning_efforts": ["low", "medium", "high"]},
                       {"id": "switch", "reasoning_efforts": ["minimal", "low", "medium", "high"]},
                       {"id": "fixed", "reasoning_efforts": ["high"]}]},
    "/litellm": {"data": [{"id": "thinker"}, {"id": "switch"}]},
}


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        prefix, _, rest = self.path.partition("/v1/")
        listing = LISTINGS.get(prefix)
        if listing is None or rest != "models":
            self.send_error(404)
            return
        body = json.dumps(listing).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *arguments):
        pass


def refused(call):
    try:
        call()
    except SystemExit as stop:
        return str(stop)
    return None


def main():
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    failures = []
    try:
        if ui_smoke.advertised_efforts(base + "/chat", "thinker", 10) != ["low", "medium", "high"]:
            failures.append("ui_smoke must read a model's efforts from the models endpoint it is given")
        message = refused(lambda: ui_smoke.advertised_efforts(base + "/litellm", "thinker", 10))
        if not message or "--models-endpoint" not in message:
            failures.append(f"a models listing without reasoning_efforts (LiteLLM's) must stop the smoke and name --models-endpoint, not read as no efforts: {message!r}")
        cases = (("switch", ["--reasoning-effort", "minimal", "--max-tokens", "2048"]),
                 ("thinker", ["--reasoning-effort", "low", "--max-tokens", "4096"]))
        for model, expected in cases:
            chosen = compsec_effort.arguments(base + "/chat", model)
            if chosen != expected:
                failures.append(f"COMPSEC for {model} must run with {expected}, not {chosen}")
        for model, endpoint, why in (("fixed", "/chat", "a model with neither minimal nor low"), ("thinker", "/litellm", "a listing without efforts"),
                                     ("absent", "/chat", "a model the chat layer does not list")):
            if refused(lambda: compsec_effort.arguments(base + endpoint, model)) is None:
                failures.append(f"{why} must stop the COMPSEC effort choice instead of guessing")
    finally:
        server.shutdown()
    for failure in failures:
        print("FAIL", failure)
    if failures:
        return 1
    print("PASS reasoning effort choice: the UI smoke and COMPSEC read each model's efforts from the chat layer; COMPSEC runs minimal with 2,048 tokens "
          "where the model can switch reasoning off and low with 4,096 where it cannot; a listing without efforts stops both")
    return 0


if __name__ == "__main__":
    sys.exit(main())
