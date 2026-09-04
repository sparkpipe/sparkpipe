#!/usr/bin/env python3

from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE_SHA = "a" * 64


def run(command: list[str], expected: int = 0) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, text=True, capture_output=True)
    if result.returncode != expected:
        raise AssertionError(
            f"command returned {result.returncode}, expected {expected}: {' '.join(command)}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result


def main() -> int:
    with tempfile.TemporaryDirectory() as directory_name:
        directory = pathlib.Path(directory_name)
        module = directory / "libfake_transport_probe.so"
        harness = directory / "spark_transport_characterize"
        receipt = directory / "receipt.json"
        run([
            "cc", "-std=c11", "-Wall", "-Wextra", "-Werror", "-O2", "-fPIC", "-shared",
            "-I", str(ROOT / "include"),
            str(ROOT / "tests" / "fake_transport_probe_module.c"),
            "-o", str(module),
        ])
        run([
            "cc", "-std=c11", "-Wall", "-Wextra", "-Werror", "-O2",
            "-I", str(ROOT / "include"),
            str(ROOT / "tools" / "hardware" / "spark_transport_characterize.c"),
            "-ldl", "-lm", "-o", str(harness),
        ])
        run([
            str(harness),
            "--question", "NET-RDMA-001",
            "--provider", str(module),
            "--source-package-sha256", SOURCE_SHA,
            "--run-id", "transport-probe-unit",
            "--topology", "unit",
            "--node", "spark0",
            "--peer", "spark1",
            "--candidate", "mapped_host",
            "--progress-mode", "autonomous",
            "--payload-bytes", "14336",
            "--lane-count", "4",
            "--window-depth", "8",
            "--iterations", "32",
            "--output", str(receipt),
        ])
        run([
            sys.executable,
            str(ROOT / "tools" / "spark_hardware_qualify.py"),
            "validate-receipt", str(receipt),
        ])
        document = json.loads(receipt.read_text())
        observation = document["answers"][0]["observations"][0]
        assert observation["parameters"] == {
            "payload_bytes": 14336,
            "lane_count": 4,
            "window_depth": 8,
            "cq_batch": 1,
            "registered_region_count": 128,
            "progress_mode": "autonomous",
            "candidate": "mapped_host",
            "iterations": 32,
        }
        metrics = observation["metrics"]
        assert metrics["numerical_pass"] is True
        assert metrics["latency_p99_ns"] == 2000
        assert document["answers"][0]["summary"]["exact_production_transport"] is True
        assert document["answers"][0]["summary"]["mapped_host_direct_mr"] is True
        assert document["answers"][0]["summary"]["no_cpu_staging_copy"] is True

        run([
            str(harness),
            "--question", "NET-RDMA-001",
            "--provider", str(ROOT / "build" / "does-not-exist.so"),
            "--source-package-sha256", SOURCE_SHA,
            "--run-id", "transport-probe-unit",
            "--topology", "unit",
            "--node", "spark0",
            "--peer", "spark1",
            "--candidate", "mapped_host",
            "--progress-mode", "autonomous",
            "--payload-bytes", "14336",
            "--lane-count", "4",
            "--window-depth", "8",
            "--output", str(receipt),
        ], 1)
    print("PASS exact production transport probe ABI and receipt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
