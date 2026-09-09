#!/usr/bin/env python3
import http.server
import json
import os
import re
import socketserver
import threading
import time
import urllib.parse

CURRENT = os.environ.get("FLEET_VIEW_DIR", os.path.expanduser("~/current"))
PORT = int(os.environ.get("FLEET_VIEW_PORT", "8801"))
AGG_EVERY = 2.0
STATE = {"agg": {}, "stamp": 0.0, "lock": threading.Lock()}
HOST_RE = re.compile(r"^spark[a-z0-9]{1,4}$")


def host_file(host):
    if not HOST_RE.fullmatch(host or ""):
        return None
    return CURRENT + os.sep + host + ".json"


def aggregate():
    hosts = {}
    ready = 0
    down = 0
    starting = 0
    now = time.time()
    for name in sorted(os.listdir(CURRENT)):
        if not name.endswith(".json"):
            continue
        try:
            with open(os.path.join(CURRENT, name)) as handle:
                data = json.load(handle)
        except (OSError, ValueError):
            continue
        host = data.get("host", name[:-5])
        age = int(now - data.get("epoch", 0))
        roots = data.get("roots", {})
        states = []
        for root_name, root in roots.items():
            state = root.get("state", "?")
            base = state.split(":")[0].strip()
            states.append(base)
            if base == "ready":
                ready += 1
            elif base == "down":
                down += 1
            else:
                starting += 1
        hosts[host] = {
            "age_s": age,
            "load": data.get("load"),
            "mem_avail_gb": data.get("mem_avail_gb"),
            "roots": {
                name_: {
                    "state": root.get("state", "?"),
                    "pid": root.get("pid", 0),
                    "age_s": root.get("age_s", -1),
                    "rss_mb": root.get("rss_mb", 0),
                    "log_age_s": root.get("log_age_s", -1),
                    "residentd": root.get("residentd", ""),
                    "driver": root.get("driver", ""),
                }
                for name_, root in roots.items()
            },
            "states": states,
        }
    with STATE["lock"]:
        STATE["agg"] = {
            "generated": int(now),
            "summary": {
                "hosts": len(hosts),
                "ready": ready,
                "down": down,
                "starting_or_other": starting,
            },
            "hosts": hosts,
        }
        STATE["stamp"] = now


def aggregator_loop():
    while True:
        try:
            aggregate()
        except Exception as error:
            print("aggregate error:", error, flush=True)
        time.sleep(AGG_EVERY)


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def do_GET(self):
        parts = urllib.parse.urlsplit(self.path)
        if parts.path in ("/", "/summary", "/agg"):
            with STATE["lock"]:
                body = json.dumps(STATE["agg"], indent=1).encode()
        elif parts.path == "/health":
            body = json.dumps({"ok": True, "port": PORT}).encode()
        elif parts.path.startswith("/host/"):
            name = parts.path[len("/host/"):].strip("/") + ".json"
            try:
                with open(os.path.join(CURRENT, name), "rb") as handle:
                    body = handle.read()
            except OSError:
                self.send_error(404)
                return
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


if __name__ == "__main__":
    threading.Thread(target=aggregator_loop, daemon=True).start()
    Server(("0.0.0.0", PORT), Handler).serve_forever()
