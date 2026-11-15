#!/usr/bin/env python3
import functools
import http.server
import importlib.util
import tempfile
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location("hf_fetch_resume", ROOT / "tools" / "glm53full_hf_fetch_resume.py")
fetch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fetch)

NAMES = [
    "config.json",
    "gguf/voices/Mandarin Chinese/Haoran_zh-energetic_M_1.codes.npy",
    "voices/take #2 ?50%.npy",
    "voices/한국어 목소리.npy",
]


class Handler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


def main():
    failures = []
    with tempfile.TemporaryDirectory() as served, tempfile.TemporaryDirectory() as stage:
        root = Path(served) / "org" / "model" / "resolve" / "abc123"
        payloads = {}
        for index, name in enumerate(NAMES):
            payloads[name] = f"payload {index} for {name}\n".encode() * (index + 1)
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payloads[name])
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Handler, directory=served))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        fetch.HF_RESOLVE = f"http://127.0.0.1:{server.server_address[1]}/{{repo}}/resolve/{{revision}}/{{path}}"
        try:
            for name in NAMES:
                base = {"repo": "org/model", "revision": "abc123", "path": name, "expected_bytes": len(payloads[name])}
                try:
                    result = fetch.fetch_one(Path(stage), base, attempts=1)
                except RuntimeError as error:
                    failures.append(f"{name!r} was not fetched: {error}")
                    continue
                got = (Path(stage) / name).read_bytes() if (Path(stage) / name).is_file() else b""
                if result.get("action") != "downloaded" or got != payloads[name]:
                    failures.append(f"{name!r} landed as {result} with {len(got)} of {len(payloads[name])} bytes")
        finally:
            server.shutdown()
    for failure in failures:
        print(f"FAIL test_hf_fetch_resume_paths: {failure}")
    if failures:
        raise SystemExit(1)
    print("PASS test_hf_fetch_resume_paths: repository files whose paths hold spaces, '#', '?', '%' and Hangul download to the same "
          "relative path with the served bytes")


if __name__ == "__main__":
    main()
