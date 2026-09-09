#!/usr/bin/env python3
import os
import socketserver
from http.server import SimpleHTTPRequestHandler

ROOT = os.path.expanduser("~/release")
PORT = int(os.environ.get("FLEET_RELEASE_PORT", "8802"))


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=ROOT, **kwargs)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, *args):
        pass


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


if __name__ == "__main__":
    os.chdir(ROOT)
    with Server(("0.0.0.0", PORT), Handler) as server:
        server.serve_forever()
