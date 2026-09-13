#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import tempfile
from typing import Any

SCRIPT = pathlib.Path(__file__).resolve()
ROOT = SCRIPT.parents[2]
sys.path.insert(0, str(SCRIPT.parent))
sys.path.insert(0, str(ROOT / "tools"))
from hardware_common import (  # noqa: E402
    canonical_json_bytes,
    load_json,
    require,
    sha256_bytes,
    sha256_file,
    write_json_atomic,
)
from run_probe_job import (  # noqa: E402
    build_command,
    executable_for_job,
    load_config,
    provider_for_job,
    resolve_path,
)
from spark_hardware_qualify import validate_plan_document  # type: ignore  # noqa: E402

PROVIDER_PROBES = {
    "model_kernel_characterize",
    "transport_characterize",
    "topology_characterize",
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fail-closed preflight for one Spark hardware qualification node")
    parser.add_argument("--plan", required=True, type=pathlib.Path)
    parser.add_argument("--config", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    return parser.parse_args()


def validate_receipt_directory(config: dict[str, Any], config_base: pathlib.Path) -> pathlib.Path:
    receipt_directory = resolve_path(str(config["receipt_directory"]), config_base)
    receipt_directory.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        prefix=".spark-handoff-preflight-",
        dir=receipt_directory,
        delete=False,
    ) as probe_file:
        probe_file.write(b"sparkpipe\n")
        probe_path = pathlib.Path(probe_file.name)
    probe_path.unlink()
    return receipt_directory


def validate_nvme(config: dict[str, Any]) -> dict[str, Any]:
    nvme = config.get("nvme")
    require(isinstance(nvme, dict), "NVMe runner configuration is missing")
    file_path_value = nvme.get("file_path")
    file_bytes = nvme.get("file_bytes")
    prepare = nvme.get("prepare")
    require(isinstance(file_path_value, str) and file_path_value,
            "NVMe probe file path is missing")
    require(isinstance(file_bytes, int) and file_bytes > 0,
            "NVMe probe file size is invalid")
    require(isinstance(prepare, bool), "NVMe prepare flag is invalid")
    file_path = pathlib.Path(file_path_value)
    if prepare:
        parent = file_path.parent
        require(parent.is_dir(), f"NVMe probe directory does not exist: {parent}")
        require(os.access(parent, os.W_OK), f"NVMe probe directory is not writable: {parent}")
        present_bytes = file_path.stat().st_size if file_path.is_file() else 0
    else:
        require(file_path.is_file(), f"prepared NVMe probe file is missing: {file_path}")
        present_bytes = file_path.stat().st_size
        require(present_bytes >= file_bytes,
                f"NVMe probe file is too small: {present_bytes} < {file_bytes}")
    return {
        "path": str(file_path),
        "required_bytes": file_bytes,
        "present_bytes": present_bytes,
        "prepare": prepare,
    }


def validate_topology_membership(plan: dict[str, Any], config: dict[str, Any]) -> None:
    topology = plan["topology"]
    require(config.get("topology") == topology["name"], "runner topology name differs from plan")
    require(config.get("topology_mode") == topology["mode"], "runner topology mode differs from plan")
    node_names = {str(node["name"]) for node in topology["nodes"]}
    require(config["node"] in node_names, "runner node is absent from plan topology")
    plan_node = next(node for node in topology["nodes"] if node["name"] == config["node"])
    require(config.get("rank") == plan_node["rank"], "runner rank differs from plan topology")
    require(config.get("production_providers_required") is True,
            "runner configuration permits non-production providers")


def preflight(plan: dict[str, Any], config: dict[str, Any], config_path: pathlib.Path) -> dict[str, Any]:
    config_base = config_path.resolve().parent
    validate_topology_membership(plan, config)
    receipt_directory = validate_receipt_directory(config, config_base)
    nvme_result = validate_nvme(config)
    jobs = [job for job in plan["jobs"] if job["scope"]["node"] == config["node"]]
    require(jobs, "plan contains no jobs for runner node")

    executable_paths: dict[pathlib.Path, str] = {}
    provider_paths: dict[pathlib.Path, str] = {}
    peer_addresses = config.get("peer_addresses", {})
    require(isinstance(peer_addresses, dict), "runner peer-address table is malformed")

    with tempfile.TemporaryDirectory() as directory_name:
        output = pathlib.Path(directory_name) / "probe.json"
        for job in jobs:
            executable = executable_for_job(config, job, config_base)
            executable_paths.setdefault(executable, sha256_file(executable))
            if job["probe_id"] in PROVIDER_PROBES:
                provider = provider_for_job(config, job, config_base)
                lowered_parts = {part.lower() for part in provider.parts}
                require("tests" not in lowered_parts,
                        f"production provider resolves inside tests: {provider}")
                require("fake" not in provider.name.lower(),
                        f"production provider name is synthetic: {provider}")
                provider_paths.setdefault(provider, sha256_file(provider))
            peer = job.get("scope", {}).get("peer")
            if peer is not None:
                require(isinstance(peer, str) and peer in peer_addresses,
                        f"peer address is missing for {peer}")
            build_command(plan, config, config_base, job, output)

    config_sha256 = config.get("config_sha256")
    report: dict[str, Any] = {
        "schema_version": 1,
        "report_kind": "spark_hardware_handoff_preflight",
        "source_package_sha256": plan["source_package_sha256"],
        "plan_id": plan["plan_id"],
        "config_sha256": config_sha256,
        "topology": plan["topology"]["name"],
        "topology_mode": plan["topology"]["mode"],
        "node": config["node"],
        "rank": config["rank"],
        "job_count": len(jobs),
        "receipt_directory": str(receipt_directory),
        "nvme": nvme_result,
        "executables": [
            {"path": str(path), "sha256": digest}
            for path, digest in sorted(executable_paths.items(), key=lambda item: str(item[0]))
        ],
        "providers": [
            {"path": str(path), "sha256": digest}
            for path, digest in sorted(provider_paths.items(), key=lambda item: str(item[0]))
        ],
        "pass": True,
    }
    report["report_sha256"] = sha256_bytes(canonical_json_bytes(report))
    return report


def main() -> int:
    arguments = parse_arguments()
    plan = validate_plan_document(load_json(arguments.plan))
    config = load_config(arguments.config)
    report = preflight(plan, config, arguments.config)
    write_json_atomic(arguments.output, report)
    print(f"PASS hardware handoff preflight: {report['job_count']} cells for {report['node']}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, KeyError, TypeError, json.JSONDecodeError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
