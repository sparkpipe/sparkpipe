#!/usr/bin/env python3

from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "hardware"))
from run_probe_job import load_config  # noqa: E402
GENERATOR = ROOT / "tools" / "hardware" / "generate_runner_configs.py"
PROVIDER_MAP = ROOT / "qualification" / "spark" / "provider_map.example.json"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def generate(topology: pathlib.Path, output: pathlib.Path) -> dict[str, object]:
    result = subprocess.run([
        sys.executable,
        str(GENERATOR),
        "--topology", str(topology),
        "--provider-map", str(PROVIDER_MAP),
        "--output-directory", str(output),
        "--run-id", "handoff-test",
        "--receipt-root", "qualification/receipts",
        "--nvme-file", "/mnt/nvme/sparkpipe-hardware-probe.bin",
        "--nvme-file-bytes", str(8 * 1024 * 1024 * 1024),
    ], cwd=ROOT, text=True, capture_output=True, check=False)
    require(result.returncode == 0, result.stderr)
    return json.loads((output / "index.json").read_text(encoding="utf-8"))


def main() -> int:
    with tempfile.TemporaryDirectory() as directory_name:
        directory = pathlib.Path(directory_name)
        ring_output = directory / "ring"
        switch_output = directory / "switch"
        ring_index = generate(
            ROOT / "qualification" / "spark" / "topologies" / "ring_13node_bringup.json",
            ring_output,
        )
        switch_index = generate(
            ROOT / "qualification" / "spark" / "topologies" / "single_switch_16node.json",
            switch_output,
        )
        require(ring_index["node_count"] == 13, "ring runner config count mismatch")
        require(switch_index["node_count"] == 16, "switch runner config count mismatch")

        spark0 = json.loads((ring_output / "spark0.json").read_text(encoding="utf-8"))
        require(spark0["peer_addresses"] == {
            "spark1": "10.13.0.1",
            "sparkc": "10.13.12.2",
        }, f"spark0 ring peer mapping is wrong: {spark0['peer_addresses']}")
        require(spark0["production_providers_required"] is True,
                "runner config allows synthetic production providers")
        require(spark0["nvme"]["prepare"] is False,
                "runner config would rewrite the NVMe corpus per cell")
        require(spark0["executables"]["cuda_characterize"].endswith(
            "build/spark_cuda_characterize"), "runner executable path drift")
        load_config(ring_output / "spark0.json")
        tampered_path = ring_output / "spark0-tampered.json"
        tampered = dict(spark0)
        tampered["rank"] = 99
        tampered_path.write_text(json.dumps(tampered), encoding="utf-8")
        try:
            load_config(tampered_path)
        except ValueError:
            pass
        else:
            raise AssertionError("runner accepted a configuration whose content did not match its SHA-256")

        switch0 = json.loads((switch_output / "spark-00.json").read_text(encoding="utf-8"))
        require(len(switch0["peer_addresses"]) == 15,
                "switch runner config does not expose all peers")
        require(switch0["peer_addresses"]["spark-15"] == "10.16.0.25",
                "switch peer address mismatch")

    print("PASS deterministic ring and single-switch runner configurations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
