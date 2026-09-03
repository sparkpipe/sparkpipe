#!/usr/bin/env python3

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pathlib
import select
import subprocess
import sys
import tempfile
from typing import Any

SCRIPT = pathlib.Path(__file__).resolve()
ROOT = SCRIPT.parents[2]
sys.path.insert(0, str(SCRIPT.parent))
sys.path.insert(0, str(ROOT / "tools"))
from hardware_common import (  # noqa: E402
    canonical_json_bytes,
    is_sha256,
    load_json,
    require,
    sha256_bytes,
    write_json_atomic,
)
from spark_hardware_qualify import (  # type: ignore  # noqa: E402
    validate_cell_receipt_document,
    validate_plan_document,
    validate_probe_receipt_for_job,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute exact cells from a Spark hardware plan")
    parser.add_argument("--plan", required=True, type=pathlib.Path)
    parser.add_argument("--config", required=True, type=pathlib.Path)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--job-index", type=int)
    selection.add_argument("--cell-id")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve_path(value: str, base: pathlib.Path) -> pathlib.Path:
    path = pathlib.Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def load_config(path: pathlib.Path) -> dict[str, Any]:
    document = load_json(path)
    require(isinstance(document, dict), "runner configuration is not an object")
    config_sha256 = document.get("config_sha256")
    if config_sha256 is not None:
        require(is_sha256(config_sha256), "runner configuration SHA-256 is invalid")
        unhashed = dict(document)
        unhashed.pop("config_sha256", None)
        require(config_sha256 == sha256_bytes(canonical_json_bytes(unhashed)),
                "runner configuration SHA-256 mismatch")
    require(document.get("schema_version") == 1, "unsupported runner configuration schema")
    require(isinstance(document.get("node"), str) and document["node"], "runner node missing")
    require(isinstance(document.get("run_id"), str) and document["run_id"], "runner run ID missing")
    require(isinstance(document.get("receipt_directory"), str) and document["receipt_directory"],
            "runner receipt directory missing")
    require(isinstance(document.get("executables", {}), dict), "runner executables are malformed")
    require(isinstance(document.get("providers", {}), dict), "runner providers are malformed")
    default_timeout = document.get("probe_timeout_seconds", 7200)
    require(isinstance(default_timeout, int) and 1 <= default_timeout <= 86400,
            "runner probe timeout is invalid")
    timeout_table = document.get("probe_timeouts_seconds", {})
    require(isinstance(timeout_table, dict), "runner probe timeout table is malformed")
    for probe_id, timeout in timeout_table.items():
        require(isinstance(probe_id, str) and probe_id, "runner timeout probe ID is invalid")
        require(isinstance(timeout, int) and 1 <= timeout <= 86400,
                f"runner timeout is invalid for {probe_id}")
    return document


def timeout_for_job(config: dict[str, Any], job: dict[str, Any]) -> int:
    timeout_table = config.get("probe_timeouts_seconds", {})
    timeout = timeout_table.get(job["probe_id"], config.get("probe_timeout_seconds", 7200))
    require(isinstance(timeout, int) and 1 <= timeout <= 86400,
            f"invalid timeout for {job['probe_id']}")
    return timeout


def provider_for_job(config: dict[str, Any], job: dict[str, Any], base: pathlib.Path) -> pathlib.Path:
    provider_table = config.get("providers", {}).get(job["probe_id"])
    require(isinstance(provider_table, dict), f"provider table missing for {job['probe_id']}")
    model_id = job.get("parameters", {}).get("model_id")
    candidate = job.get("parameters", {}).get("candidate")
    provider_value = None
    if isinstance(model_id, str):
        provider_value = provider_table.get(model_id)
    if provider_value is None and isinstance(candidate, str):
        provider_value = provider_table.get(candidate)
    if provider_value is None:
        provider_value = provider_table.get("default")
    require(isinstance(provider_value, str) and provider_value,
            f"no provider configured for {job['probe_id']} cell {job['cell_id']}")
    provider = resolve_path(provider_value, base)
    require(provider.is_file(), f"provider does not exist: {provider}")
    return provider


def executable_for_job(config: dict[str, Any], job: dict[str, Any], base: pathlib.Path) -> pathlib.Path:
    override = config.get("executables", {}).get(job["probe_id"])
    value = override if isinstance(override, str) and override else job["executable"]
    executable = resolve_path(str(value), base)
    require(executable.is_file(), f"probe executable does not exist: {executable}")
    require(os.access(executable, os.X_OK), f"probe executable is not executable: {executable}")
    return executable


def append_common(command: list[str], plan: dict[str, Any], config: dict[str, Any], job: dict[str, Any], output: pathlib.Path) -> None:
    command.extend([
        "--question", str(job["question_id"]),
        "--source-package-sha256", str(plan["source_package_sha256"]),
        "--run-id", str(config["run_id"]),
        "--topology", str(job["scope"]["topology"]),
        "--node", str(job["scope"]["node"]),
        "--output", str(output),
    ])


def add_option(command: list[str], name: str, parameters: dict[str, Any], key: str) -> None:
    if key in parameters:
        command.extend([name, str(parameters[key])])


def build_command(
    plan: dict[str, Any],
    config: dict[str, Any],
    config_base: pathlib.Path,
    job: dict[str, Any],
    output: pathlib.Path,
) -> tuple[list[str], dict[str, Any]]:
    executable = executable_for_job(config, job, config_base)
    parameters = job["parameters"]
    command = [str(executable)]
    context: dict[str, Any] = {}
    probe_id = job["probe_id"]
    if probe_id == "cuda_characterize":
        append_common(command, plan, config, job, output)
        option_map = {
            "candidate": "--candidate",
            "mode": "--mode",
            "load_mode": "--load-mode",
            "sample_phase": "--sample-phase",
            "working_set_bytes": "--working-set-bytes",
            "payload_bytes": "--payload-bytes",
            "dynamic_shared_bytes": "--dynamic-shared-bytes",
            "batch_size": "--batch-size",
            "kernel_count": "--kernel-count",
            "stream_count": "--stream-count",
            "operations": "--operations",
            "iterations": "--iterations",
            "sustained_seconds": "--sustained-seconds",
        }
        for key, option in option_map.items():
            add_option(command, option, parameters, key)
    elif probe_id == "model_kernel_characterize":
        command.extend(["--provider", str(provider_for_job(config, job, config_base))])
        append_common(command, plan, config, job, output)
        option_map = {
            "model_id": "--model",
            "role": "--role",
            "candidate": "--candidate",
            "kernel_class": "--kernel-class",
            "route_distribution": "--route-distribution",
            "batch_size": "--batch",
            "context_tokens": "--context",
            "iterations": "--iterations",
        }
        for key, option in option_map.items():
            add_option(command, option, parameters, key)
    elif probe_id == "nvme_characterize":
        nvme = config.get("nvme")
        require(isinstance(nvme, dict), "NVMe runner configuration is missing")
        file_path = nvme.get("file_path")
        file_bytes = nvme.get("file_bytes")
        require(isinstance(file_path, str) and file_path, "NVMe file path missing")
        require(isinstance(file_bytes, int) and file_bytes > 0, "NVMe file bytes missing")
        command.extend(["--file", file_path, "--file-bytes", str(file_bytes)])
        if nvme.get("prepare") is True:
            command.append("--prepare")
        if isinstance(nvme.get("cuda_device"), int):
            command.extend(["--cuda-device", str(nvme["cuda_device"])])
        append_common(command, plan, config, job, output)
        for key, option in {
            "candidate": "--candidate",
            "block_bytes": "--block-bytes",
            "queue_depth": "--queue-depth",
            "worker_count": "--worker-count",
            "iterations": "--iterations",
        }.items():
            add_option(command, option, parameters, key)
    elif probe_id == "transport_characterize":
        command.extend(["--provider", str(provider_for_job(config, job, config_base))])
        append_common(command, plan, config, job, output)
        command.extend(["--peer", str(job["scope"]["peer"])])
        for key, option in {
            "candidate": "--candidate",
            "payload_bytes": "--payload-bytes",
            "lane_count": "--lane-count",
            "window_depth": "--window-depth",
            "cq_batch": "--cq-batch",
            "registered_region_count": "--registered-region-count",
            "progress_mode": "--progress-mode",
            "iterations": "--iterations",
        }.items():
            add_option(command, option, parameters, key)
    elif probe_id == "pmtu_characterize":
        peer = str(job["scope"]["peer"])
        addresses = config.get("peer_addresses")
        require(isinstance(addresses, dict) and isinstance(addresses.get(peer), str),
                f"peer address missing for {peer}")
        pmtu = config.get("pmtu", {})
        require(isinstance(pmtu, dict), "PMTU configuration is malformed")
        if "port" in pmtu:
            port = int(pmtu["port"])
        else:
            port_base = int(pmtu.get("port_base", 47000))
            port_span = int(pmtu.get("port_span", 1000))
            require(port_span > 0, "PMTU port span must be positive")
            port = port_base + (int(str(job["cell_id"])[:8], 16) % port_span)
        require(1 <= port <= 65535, "PMTU port is outside the valid range")
        context.update({
            "pmtu_peer": peer,
            "pmtu_port": port,
            "pmtu_executable": executable,
            "pmtu_idle_timeout_seconds": int(pmtu.get("server_idle_timeout_seconds", 60)),
        })
        command.extend(["--peer-address", str(addresses[peer]), "--port", str(port)])
        append_common(command, plan, config, job, output)
        command.extend(["--peer", peer])
        for key, option in {
            "candidate": "--candidate",
            "minimum_payload_bytes": "--minimum-payload-bytes",
            "maximum_payload_bytes": "--maximum-payload-bytes",
            "iterations": "--iterations",
        }.items():
            add_option(command, option, parameters, key)
    elif probe_id == "topology_characterize":
        command.extend(["--provider", str(provider_for_job(config, job, config_base))])
        append_common(command, plan, config, job, output)
        for key, option in {
            "model_id": "--model",
            "candidate": "--candidate",
            "batch_size": "--batch",
            "context_tokens": "--context",
            "pipeline_degree": "--pipeline-degree",
            "window_depth": "--window-depth",
            "iterations": "--iterations",
        }.items():
            add_option(command, option, parameters, key)
    else:
        raise ValueError(f"unsupported probe {probe_id}")
    return command, context


def start_pmtu_server(config: dict[str, Any], context: dict[str, Any], config_base: pathlib.Path) -> subprocess.Popen[str]:
    peer = context["pmtu_peer"]
    executable = context["pmtu_executable"]
    port = context["pmtu_port"]
    idle_timeout_seconds = context["pmtu_idle_timeout_seconds"]
    pmtu = config.get("pmtu", {})
    require(isinstance(idle_timeout_seconds, int) and 1 <= idle_timeout_seconds <= 3600,
            "PMTU server idle timeout is invalid")
    if str(config.get("node")) == peer or pmtu.get("local_server") is True:
        bind_address = str(pmtu.get("local_bind_address", "127.0.0.1"))
        command = [str(executable), "--server", "--bind", bind_address,
                   "--port", str(port), "--idle-timeout-seconds", str(idle_timeout_seconds)]
    else:
        template = pmtu.get("remote_command_template")
        require(isinstance(template, list) and template, "remote PMTU command template missing")
        remote_executable = str(pmtu.get("remote_executable", executable))
        command = [str(token).format(peer=peer, node=peer) for token in template]
        command.extend([remote_executable, "--server", "--port", str(port),
                        "--idle-timeout-seconds", str(idle_timeout_seconds)])
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    require(process.stdout is not None, "PMTU server stdout unavailable")
    timeout = float(pmtu.get("startup_seconds", 10.0))
    ready, _, _ = select.select([process.stdout], [], [], timeout)
    if not ready:
        process.terminate()
        try:
            _, error = process.communicate(timeout=5.0)
        except subprocess.TimeoutExpired:
            process.kill()
            _, error = process.communicate()
        raise ValueError(f"PMTU server on {peer} did not become ready: {error.strip()}")
    line = process.stdout.readline().strip()
    if not line.startswith("READY "):
        process.terminate()
        try:
            _, error = process.communicate(timeout=5.0)
        except subprocess.TimeoutExpired:
            process.kill()
            _, error = process.communicate()
        raise ValueError(f"PMTU server failed readiness: {line} {error.strip()}")
    return process


def failure_probe_receipt(plan: dict[str, Any], config: dict[str, Any], job: dict[str, Any], error: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "receipt_kind": "spark_hardware_probe",
        "run_id": config["run_id"],
        "probe_id": job["probe_id"],
        "source_identity": {"source_package_sha256": plan["source_package_sha256"]},
        "scope": dict(job["scope"]),
        "answers": [{
            "question_id": job["question_id"],
            "status": "failed",
            "summary": {"runner_generated_failure": True},
            "observations": [],
            "error": error[:4096],
        }],
    }


def execute_job(plan: dict[str, Any], config: dict[str, Any], config_base: pathlib.Path, job: dict[str, Any], receipt_directory: pathlib.Path, dry_run: bool, resume: bool) -> None:
    receipt_path = receipt_directory / f"{job['cell_id']}.json"
    if receipt_path.exists():
        existing = load_json(receipt_path)
        validate_cell_receipt_document(existing, plan)
        if resume:
            answer = existing["probe_receipt"]["answers"][0]
            require(answer.get("status") == "measured",
                    f"probe receipt status is {answer.get('status')}: {answer.get('error', '')}")
            return
        raise ValueError(f"receipt already exists for {job['cell_id']}")
    lock_path = receipt_path.with_suffix(".lock")
    lock_descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        os.write(lock_descriptor,
                 f"pid={os.getpid()} cell_id={job['cell_id']}\n".encode("ascii"))
    finally:
        os.close(lock_descriptor)
    server: subprocess.Popen[str] | None = None
    try:
        with tempfile.TemporaryDirectory(dir=receipt_directory) as directory_name:
            directory = pathlib.Path(directory_name)
            probe_path = directory / "probe.json"
            command, context = build_command(plan, config, config_base, job, probe_path)
            if dry_run:
                print(json.dumps({"cell_id": job["cell_id"], "command": command}, separators=(",", ":")))
                return
            if job["probe_id"] == "pmtu_characterize":
                server = start_pmtu_server(config, context, config_base)
            try:
                completed = subprocess.run(
                    command,
                    text=True,
                    capture_output=True,
                    check=False,
                    timeout=timeout_for_job(config, job),
                )
            except subprocess.TimeoutExpired as error:
                diagnostic = (
                    f"probe timed out after {error.timeout} seconds: "
                    f"{(error.stderr or error.stdout or '').strip()}"
                )
                probe_receipt = failure_probe_receipt(plan, config, job, diagnostic)
                validate_probe_receipt_for_job(probe_receipt, plan, job)
            else:
                if completed.returncode == 0 and probe_path.is_file():
                    probe_receipt = load_json(probe_path)
                    validate_probe_receipt_for_job(probe_receipt, plan, job)
                else:
                    diagnostic = (
                        completed.stderr.strip() or
                        completed.stdout.strip() or
                        f"probe exited with status {completed.returncode}"
                    )
                    probe_receipt = failure_probe_receipt(plan, config, job, diagnostic)
                    validate_probe_receipt_for_job(probe_receipt, plan, job)
            cell_receipt: dict[str, Any] = {
                "schema_version": 1,
                "receipt_kind": "spark_hardware_cell_receipt",
                "plan_id": plan["plan_id"],
                "cell_id": job["cell_id"],
                "job": job,
                "completed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "probe_receipt": probe_receipt,
            }
            cell_receipt["receipt_sha256"] = sha256_bytes(canonical_json_bytes(cell_receipt))
            validate_cell_receipt_document(cell_receipt, plan)
            write_json_atomic(receipt_path, cell_receipt)
            answer = probe_receipt["answers"][0]
            require(answer.get("status") == "measured",
                    f"probe returned status {answer.get('status')}: {answer.get('error', '')}")
    finally:
        if server is not None:
            server.terminate()
            try:
                server.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
        lock_path.unlink(missing_ok=True)


def selected_jobs(plan: dict[str, Any], config: dict[str, Any], arguments: argparse.Namespace) -> list[dict[str, Any]]:
    jobs = [job for job in plan["jobs"] if job["scope"]["node"] == config["node"]]
    if arguments.job_index is not None:
        jobs = [job for job in jobs if job["job_index"] == arguments.job_index]
    if arguments.cell_id is not None:
        jobs = [job for job in jobs if job["cell_id"] == arguments.cell_id]
    require(arguments.shard_count > 0, "shard count must be positive")
    require(0 <= arguments.shard_index < arguments.shard_count, "invalid shard index")
    jobs = [job for job in jobs if job["job_index"] % arguments.shard_count == arguments.shard_index]
    if arguments.job_index is not None or arguments.cell_id is not None:
        require(len(jobs) == 1, "selected cell is absent from this node/shard")
    return jobs


def main() -> int:
    arguments = parse_arguments()
    plan = validate_plan_document(load_json(arguments.plan))
    config = load_config(arguments.config)
    config_base = arguments.config.resolve().parent
    receipt_directory = resolve_path(config["receipt_directory"], config_base)
    receipt_directory.mkdir(parents=True, exist_ok=True)
    jobs = selected_jobs(plan, config, arguments)
    require(jobs, "no jobs selected")
    for job in jobs:
        execute_job(plan, config, config_base, job, receipt_directory,
                    arguments.dry_run, arguments.resume)
    print(f"PASS executed {len(jobs)} hardware cells for {config['node']}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, KeyError, TypeError, json.JSONDecodeError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
