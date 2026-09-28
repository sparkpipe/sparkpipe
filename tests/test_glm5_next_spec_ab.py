#!/usr/bin/env python3
import json
import os
import subprocess
import sys
import tempfile
import threading
from http.server import HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from test_spec_verify_bench import Handler  # noqa: E402


def arm(label, out, endpoint, baseline=None):
    env = dict(os.environ, SPEC_AB_ENDPOINT=endpoint, SPEC_AB_LOG_HOST="", SPEC_AB_MAX_TOKENS="3")
    command = ["bash", str(ROOT / "tools/glm5_next_spec_ab.sh"), label, str(out)] + ([str(baseline)] if baseline else [])
    return subprocess.run(command, env=env, capture_output=True, text=True)


def main():
    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    endpoint = f"http://127.0.0.1:{server.server_port}"
    with tempfile.TemporaryDirectory() as directory:
        out = Path(directory)
        assert subprocess.run(["bash", str(ROOT / "tools/glm5_next_spec_ab.sh"), "off"], capture_output=True).returncode == 2
        assert arm("bad label", out, endpoint).returncode == 2
        off = arm("off", out, endpoint)
        assert off.returncode == 0, off.stderr
        summary = json.loads((out / "off.summary.json").read_text())
        assert set(summary["decode_tok_s"]) == {"prose", "code", "repetitive"} and "exact_vs_baseline" not in summary
        same = arm("mtp", out, endpoint, out / "off.json")
        assert same.returncode == 0, same.stderr
        summary = json.loads((out / "mtp.summary.json").read_text())
        assert summary["exact_vs_baseline"] is True and set(summary["speedup_vs_baseline"]) == {"prose", "code", "repetitive"}
        Handler.omit_tokens = False
        baseline = json.loads((out / "off.json").read_text())
        baseline["results"][0]["token_ids"] = [1, 2, 3]
        (out / "altered.json").write_text(json.dumps(baseline))
        differs = arm("mtp+lookup", out, endpoint, out / "altered.json")
        assert differs.returncode != 0
        assert json.loads((out / "mtp+lookup.summary.json").read_text())["exact_vs_baseline"] is False
    server.shutdown()
    print("PASS glm5_next_spec_ab.sh runs an arm, summarizes decode rate per class and fails an arm whose tokens differ from the baseline")


if __name__ == "__main__":
    main()
