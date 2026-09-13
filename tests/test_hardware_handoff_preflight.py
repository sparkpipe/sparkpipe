#!/usr/bin/env python3

from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "hardware"))
from hardware_common import canonical_json_bytes, sha256_bytes  # noqa: E402

SOURCE_SHA = "a" * 64


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def write_json(path: pathlib.Path, document: object) -> None:
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def cell_id(question_id: str, probe_id: str, scope: dict[str, object], parameters: dict[str, object]) -> str:
    return sha256_bytes(canonical_json_bytes({
        "question_id": question_id,
        "probe_id": probe_id,
        "scope": scope,
        "parameters": parameters,
    }))


def build_plan(cuda_executable: pathlib.Path, transport_executable: pathlib.Path) -> dict[str, object]:
    cuda_scope = {"topology": "unit", "node": "spark0"}
    cuda_parameters = {"candidate": "identity", "iterations": 1}
    transport_scope = {"topology": "unit", "node": "spark0", "peer": "spark1"}
    transport_parameters = {
        "candidate": "mapped_host",
        "payload_bytes": 14336,
        "lane_count": 1,
        "window_depth": 1,
        "cq_batch": 1,
        "registered_region_count": 16,
        "progress_mode": "event_loop",
        "iterations": 1,
    }
    jobs = [
        {
            "job_index": 0,
            "cell_id": cell_id("GB10-IDENTITY-001", "cuda_characterize", cuda_scope, cuda_parameters),
            "question_id": "GB10-IDENTITY-001",
            "probe_id": "cuda_characterize",
            "executable": str(cuda_executable),
            "scope": cuda_scope,
            "parameters": cuda_parameters,
        },
        {
            "job_index": 1,
            "cell_id": cell_id("NET-RDMA-001", "transport_characterize", transport_scope, transport_parameters),
            "question_id": "NET-RDMA-001",
            "probe_id": "transport_characterize",
            "executable": str(transport_executable),
            "scope": transport_scope,
            "parameters": transport_parameters,
        },
    ]
    plan: dict[str, object] = {
        "schema_version": 1,
        "plan_kind": "spark_hardware_qualification_plan",
        "source_package_sha256": SOURCE_SHA,
        "question_registry_sha256": "b" * 64,
        "probe_registry_sha256": "c" * 64,
        "workload_profiles_sha256": "d" * 64,
        "topology_source_sha256": "e" * 64,
        "topology": {
            "name": "unit",
            "mode": "single_switch",
            "nodes": [
                {"name": "spark0", "rank": 0, "nvme_device": "/dev/null", "addresses": ["127.0.0.1"]},
                {"name": "spark1", "rank": 1, "nvme_device": "/dev/null", "addresses": ["127.0.0.2"]},
            ],
            "fabrics": [{"name": "unit", "kind": "switch", "speed_gbps": 100, "mtu_bytes": 1500}],
        },
        "coverage": {
            "GB10-IDENTITY-001": {
                "probe_id": "cuda_characterize",
                "applicable": True,
                "expected_observation_count": 1,
                "axes": {"candidate": ["identity"], "iterations": [1]},
            },
            "NET-RDMA-001": {
                "probe_id": "transport_characterize",
                "applicable": True,
                "expected_observation_count": 1,
                "axes": {key: [value] for key, value in transport_parameters.items()},
            },
        },
        "jobs": jobs,
    }
    plan["plan_id"] = sha256_bytes(canonical_json_bytes(plan))
    return plan


def build_config(
    directory: pathlib.Path,
    cuda_executable: pathlib.Path,
    transport_executable: pathlib.Path,
    provider: pathlib.Path,
    nvme_file: pathlib.Path,
) -> dict[str, object]:
    config: dict[str, object] = {
        "schema_version": 1,
        "config_kind": "spark_hardware_runner_config",
        "run_id": "preflight-unit",
        "topology": "unit",
        "topology_mode": "single_switch",
        "node": "spark0",
        "rank": 0,
        "receipt_directory": str(directory / "receipts"),
        "probe_timeout_seconds": 60,
        "probe_timeouts_seconds": {},
        "executables": {
            "cuda_characterize": str(cuda_executable),
            "transport_characterize": str(transport_executable),
        },
        "providers": {
            "transport_characterize": {"mapped_host": str(provider)},
        },
        "peer_addresses": {"spark1": "127.0.0.2"},
        "nvme": {
            "file_path": str(nvme_file),
            "file_bytes": nvme_file.stat().st_size,
            "prepare": False,
            "cuda_device": 0,
        },
        "pmtu": {
            "remote_command_template": ["true"],
            "remote_executable": "/bin/true",
        },
        "production_providers_required": True,
    }
    config["config_sha256"] = sha256_bytes(canonical_json_bytes(config))
    return config


def run(command: list[str], expected: int = 0) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if result.returncode != expected:
        raise AssertionError(
            f"command returned {result.returncode}, expected {expected}: {' '.join(command)}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result


def main() -> int:
    with tempfile.TemporaryDirectory() as directory_name:
        directory = pathlib.Path(directory_name)
        cuda_executable = directory / "cuda_probe"
        transport_executable = directory / "transport_probe"
        provider = directory / "libproduction_transport.so"
        nvme_file = directory / "nvme.bin"
        for executable in (cuda_executable, transport_executable):
            executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            executable.chmod(0o755)
        provider.write_bytes(b"production-provider\n")
        nvme_file.write_bytes(b"n" * 4096)

        plan = build_plan(cuda_executable, transport_executable)
        config = build_config(directory, cuda_executable, transport_executable, provider, nvme_file)
        plan_path = directory / "plan.json"
        config_path = directory / "config.json"
        output_path = directory / "preflight.json"
        write_json(plan_path, plan)
        write_json(config_path, config)
        preflight = ROOT / "tools" / "hardware" / "spark_handoff_preflight.py"
        run([
            sys.executable,
            str(preflight),
            "--plan", str(plan_path),
            "--config", str(config_path),
            "--output", str(output_path),
        ])
        report = json.loads(output_path.read_text(encoding="utf-8"))
        require(report["pass"] is True, "preflight did not pass")
        require(report["job_count"] == 2, "preflight job count mismatch")
        require(len(report["providers"]) == 1, "preflight provider inventory mismatch")
        require(len(report["executables"]) == 2, "preflight executable inventory mismatch")

        provider.unlink()
        run([
            sys.executable,
            str(preflight),
            "--plan", str(plan_path),
            "--config", str(config_path),
            "--output", str(output_path),
        ], 2)

    print("PASS hardware handoff preflight and artifact inventory")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
