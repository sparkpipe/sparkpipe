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


OFF_LOG = "GLM verify regime rows=0 drafter=0\nmodel_residentd ready\n"
SPEC_LOG = ("GLM verify regime rows=8 drafter=4\n"
            "GLM verify MTP pack packs/mtp/glm5_next_mtp.tp16.rank15.g5nsp tensors=27 device_bytes=616611584 tp=16 rank=15\n"
            "model_residentd ready\nGRAPH-VERIFY-TABLE slot=0 rows_max=8 status=0 capture_ms=900\n"
            "VERIFY-FRAME slot=0 position=20 budget=8 produced=8 rounds=3 accepted=5 steps=0 | frames=1 plain=0\n")


def arm(label, out, endpoint, baseline=None, log=None):
    env = dict(os.environ, SPEC_AB_ENDPOINT=endpoint, SPEC_AB_LOG_HOST="", SPEC_AB_MAX_TOKENS="3")
    env.pop("SPEC_AB_LOG_FILE", None)
    if log is not None:
        path = Path(out) / f"{label}.input.log"
        path.write_text(log)
        env["SPEC_AB_LOG_FILE"] = str(path)
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
        good_off = arm("off", out, endpoint, log=OFF_LOG)
        assert good_off.returncode == 0, good_off.stderr
        summary = json.loads((out / "off.summary.json").read_text())
        assert summary["spec_state_ok"] is True and summary["verify_rows"] == 0 and summary["stop_markers"] == []
        assert arm("off", out, endpoint, log=SPEC_LOG).returncode != 0
        assert json.loads((out / "off.summary.json").read_text())["spec_state_ok"] is False
        good_spec = arm("mtp8", out, endpoint, out / "off.json", log=SPEC_LOG)
        assert good_spec.returncode == 0, good_spec.stderr
        summary = json.loads((out / "mtp8.summary.json").read_text())
        assert summary["spec_state_ok"] is True and summary["verify_rows"] == 8 and summary["exact_vs_baseline"] is True
        assert arm("mtp8", out, endpoint, out / "off.json", log=SPEC_LOG.replace("VERIFY-FRAME", "VERIFY-PLAIN-FRAME")).returncode != 0
        assert json.loads((out / "mtp8.summary.json").read_text())["spec_state_ok"] is False
        assert arm("mtp8", out, endpoint, out / "off.json", log=OFF_LOG).returncode != 0
        for marker in ("VERIFY-MTP-DRAFT-FAILED lane=0 position=40 depth=7", "VERIFY-RANK-LOCAL-INELIGIBLE request=3", "VERIFY-MTP-UNSUPPORTED tp=16",
                       "GRAPH-VERIFY-REJECTED rows=8 status=-3", "GRAPH-VERIFY-TABLE slot=1 rows_max=8 status=-7 capture_ms=3",
                       "VERIFY-PLAIN-STEP-FAILED position=9 status=-2", "VERIFY-OBSERVE-FAILED lane=0 position=9 tokens=8"):
            stopped = arm("mtp8", out, endpoint, out / "off.json", log=SPEC_LOG + marker + "\n")
            assert stopped.returncode != 0, marker
            assert json.loads((out / "mtp8.summary.json").read_text())["stop_markers"] == [marker]
    server.shutdown()
    print("PASS glm5_next_spec_ab.sh runs an arm, summarizes decode rate per class and fails an arm whose tokens differ from the baseline, "
          "whose engine log carries a stop marker, or whose verify regime does not match the arm (off must run no verify frame, a spec arm at least one)")


if __name__ == "__main__":
    main()
