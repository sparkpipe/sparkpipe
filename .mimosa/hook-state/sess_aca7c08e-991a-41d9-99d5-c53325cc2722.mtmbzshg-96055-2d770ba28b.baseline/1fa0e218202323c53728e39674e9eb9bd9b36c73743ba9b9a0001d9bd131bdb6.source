#!/usr/bin/env python3

from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE_SHA = "a" * 64


def run(command: list[str], expected: int = 0) -> None:
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if result.returncode != expected:
        raise AssertionError(
            f"command returned {result.returncode}, expected {expected}: {' '.join(command)}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )


def main() -> int:
    with tempfile.TemporaryDirectory() as directory_name:
        directory = pathlib.Path(directory_name)
        module = directory / "libfake_topology_probe.so"
        harness = directory / "spark_topology_characterize"
        receipt = directory / "receipt.json"
        run(["cc", "-std=c11", "-Wall", "-Wextra", "-Werror", "-O2", "-fPIC", "-shared",
             "-I", str(ROOT / "include"), str(ROOT / "tests" / "fake_topology_probe_module.c"),
             "-o", str(module)])
        run(["cc", "-std=c11", "-Wall", "-Wextra", "-Werror", "-O2",
             "-I", str(ROOT / "include"), str(ROOT / "tools" / "hardware" / "spark_topology_characterize.c"),
             "-ldl", "-lm", "-o", str(harness)])
        run([str(harness), "--provider", str(module), "--question", "TOPO-PP-001",
             "--source-package-sha256", SOURCE_SHA, "--run-id", "topology-unit",
             "--topology", "ring_13node_bringup", "--node", "spark0",
             "--model", "glm52_fp8_experts_bf16_rest", "--candidate", "pp13",
             "--batch", "8", "--context", "2048", "--pipeline-degree", "13",
             "--window-depth", "16", "--iterations", "32", "--output", str(receipt)])
        run([sys.executable, str(ROOT / "tools" / "spark_hardware_qualify.py"),
             "validate-receipt", str(receipt)])
        document = json.loads(receipt.read_text(encoding="utf-8"))
        answer = document["answers"][0]
        assert answer["summary"]["global_commit_verified"] is True
        assert answer["summary"]["final_event_ack_verified"] is True
        observation = answer["observations"][0]
        assert observation["parameters"]["pipeline_degree"] == 13
        assert observation["metrics"]["tokens_per_second"] == 500.0
    print("PASS exact production topology probe ABI and receipt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
