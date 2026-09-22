#!/usr/bin/env python3
"""The release builder refuses legacy invocation and propagates build failure."""
import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools/glm5_next_build_release.sh"


def main():
    env = dict(os.environ)
    env.pop("SPARK_QUEUE_ID", None)
    result = subprocess.run(["bash", str(SCRIPT)], env=env, capture_output=True)
    assert result.returncode != 0 and b"SPARK_QUEUE_ID" in result.stderr
    result = subprocess.run(["bash", str(SCRIPT), "legacy-branch"], env=env, capture_output=True)
    assert result.returncode == 2 and b"no arguments" in result.stderr
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "tools").mkdir()
        (root / "model_contracts").mkdir()
        fixture_script = root / "tools/glm5_next_build_release.sh"
        fixture_script.write_bytes(SCRIPT.read_bytes())
        (root / "tools/module_build_release.sh").write_bytes((ROOT / "tools/module_build_release.sh").read_bytes())
        (root / "model_contracts/glm53_flash_authoritative.json").write_text("{}")
        for name, body in (("git", 'if [ "$1" = rev-parse ]; then printf "%040d\\n" 1; fi'), ("make", "exit 23"), ("flock", "exit 0")):
            path = root / name
            path.write_text("#!/bin/sh\n" + body + "\n")
            path.chmod(0o755)
        env.update(SPARK_QUEUE_ID="test-failed-build", PATH=str(root) + ":" + env["PATH"])
        result = subprocess.run(["bash", str(fixture_script)], env=env, capture_output=True)
        assert result.returncode == 23 and b"BUILD-PASS" not in result.stdout
        (root / "build/obj").mkdir()
        result = subprocess.run(["bash", str(fixture_script)], env=env, capture_output=True)
        assert result.returncode == 2 and b"different contract" in result.stderr
    print("PASS queue build: required ownership, legacy refusal, first-failure propagation")


if __name__ == "__main__":
    main()
