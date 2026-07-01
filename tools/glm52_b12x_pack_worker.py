#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Dict, Iterable, List, Sequence
from uuid import uuid4


JOB_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]+$")


class WorkerFailure(RuntimeError):
    pass


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def now_ns() -> int:
    return time.time_ns()


def queue_paths(queue_dir: Path) -> Dict[str, Path]:
    return {
        "pending": queue_dir / "pending",
        "running": queue_dir / "running",
        "complete": queue_dir / "complete",
        "failed": queue_dir / "failed",
        "status": queue_dir / "status",
        "logs": queue_dir / "logs",
    }


def ensure_queue(queue_dir: Path) -> Dict[str, Path]:
    paths = queue_paths(queue_dir)
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def atomic_write_json(path: Path, document: Dict[str, Any]) -> None:
    temporary_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary_path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary_path, path)


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_layers(text: str) -> List[int]:
    layers: List[int] = []
    for item in text.replace(";", ",").split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value < 0:
            raise WorkerFailure("layer indices must be non-negative")
        layers.append(value)
    if not layers:
        raise WorkerFailure("at least one layer must be selected")
    return sorted(set(layers))


def validate_job_id(job_id: str) -> str:
    if not JOB_ID_RE.match(job_id):
        raise WorkerFailure(f"invalid job id: {job_id}")
    return job_id


def job_file(directory: Path, job_id: str) -> Path:
    return directory / f"{job_id}.json"


def list_job_files(directory: Path) -> List[Path]:
    return sorted(path for path in directory.glob("*.json") if path.is_file())


def write_status(paths: Dict[str, Path], job: Dict[str, Any], state: str, extra: Dict[str, Any] | None = None) -> None:
    document = dict(job)
    document["state"] = state
    document["updated_ns"] = now_ns()
    if extra is not None:
        document.update(extra)
    atomic_write_json(job_file(paths["status"], str(job["job_id"])), document)


def command_submit(args: argparse.Namespace) -> int:
    queue_dir = args.queue_dir.resolve()
    paths = ensure_queue(queue_dir)
    job_id = validate_job_id(args.job_id or f"pack-{uuid4().hex}")
    pending_path = job_file(paths["pending"], job_id)
    if pending_path.exists() or job_file(paths["running"], job_id).exists():
        raise WorkerFailure(f"job already exists or is running: {job_id}")
    layers = parse_layers(args.layers)
    job = {
        "job_id": job_id,
        "submitted_ns": now_ns(),
        "model_dir": str(Path(args.model_dir).resolve()),
        "aot_manifest": str(Path(args.aot_manifest).resolve()),
        "layers": layers,
        "output_dir": str(Path(args.output_dir).resolve()),
        "local_jobs": int(args.local_jobs),
        "verify_reused_sha256": bool(args.verify_reused_sha256),
        "artifact_transfer": "none_control_plane_only",
    }
    if job["local_jobs"] <= 0:
        raise WorkerFailure("--local-jobs must be positive")
    atomic_write_json(pending_path, job)
    write_status(paths, job, "pending")
    print(json.dumps({"job_id": job_id, "state": "pending", "queue_dir": str(queue_dir)}, sort_keys=True))
    return 0


def queue_depth(queue_dir: Path, max_jobs: int) -> Dict[str, Any]:
    paths = ensure_queue(queue_dir)
    pending = len(list_job_files(paths["pending"]))
    running = len(list_job_files(paths["running"]))
    return {
        "available": max(0, max_jobs - running),
        "capacity": max_jobs,
        "pending": pending,
        "running": running,
    }


def command_queue_depth(args: argparse.Namespace) -> int:
    if args.max_jobs <= 0:
        raise WorkerFailure("--max-jobs must be positive")
    print(json.dumps(queue_depth(args.queue_dir.resolve(), args.max_jobs), sort_keys=True))
    return 0


def command_status(args: argparse.Namespace) -> int:
    paths = ensure_queue(args.queue_dir.resolve())
    if args.job_id:
        job_id = validate_job_id(args.job_id)
        path = job_file(paths["status"], job_id)
        if not path.exists():
            raise WorkerFailure(f"missing job status: {job_id}")
        print(json.dumps(load_json(path), sort_keys=True))
        return 0
    statuses = [load_json(path) for path in list_job_files(paths["status"])]
    print(json.dumps({"jobs": statuses}, sort_keys=True))
    return 0


def pack_command(packer_path: Path, job: Dict[str, Any]) -> List[str]:
    command = [
        sys.executable,
        str(packer_path),
        "--model-dir",
        str(job["model_dir"]),
        "--aot-manifest",
        str(job["aot_manifest"]),
        "--layers",
        ",".join(str(layer) for layer in job["layers"]),
        "--output-dir",
        str(job["output_dir"]),
        "--jobs",
        str(job["local_jobs"]),
        "--reuse-valid",
    ]
    if job.get("verify_reused_sha256", False):
        command.append("--verify-reused-sha256")
    return command


def start_job(paths: Dict[str, Path], pending_path: Path, packer_path: Path) -> Dict[str, Any]:
    job = load_json(pending_path)
    job_id = validate_job_id(str(job["job_id"]))
    running_path = job_file(paths["running"], job_id)
    os.replace(pending_path, running_path)
    log_path = paths["logs"] / f"{job_id}.log"
    log_handle = log_path.open("ab")
    command = pack_command(packer_path, job)
    process = subprocess.Popen(command, stdout=log_handle, stderr=subprocess.STDOUT)
    write_status(paths, job, "running", {
        "command": command,
        "log_path": str(log_path),
        "pid": process.pid,
        "started_ns": now_ns(),
    })
    return {
        "job": job,
        "log_handle": log_handle,
        "process": process,
        "running_path": running_path,
    }


def finish_job(paths: Dict[str, Path], running: Dict[str, Any]) -> bool:
    process = running["process"]
    return_code = process.poll()
    if return_code is None:
        return False
    job = running["job"]
    job_id = str(job["job_id"])
    running["log_handle"].close()
    final_dir = paths["complete"] if return_code == 0 else paths["failed"]
    final_path = job_file(final_dir, job_id)
    os.replace(running["running_path"], final_path)
    state = "complete" if return_code == 0 else "failed"
    extra = {
        "artifact_manifest": str(Path(str(job["output_dir"])) / "resident_moe_pack_manifest.json"),
        "artifact_transfer": "none_control_plane_only",
        "completed_ns": now_ns(),
        "returncode": return_code,
    }
    write_status(paths, job, state, extra)
    return True


def serve_once(paths: Dict[str, Path], packer_path: Path, max_jobs: int, running: List[Dict[str, Any]]) -> None:
    active = []
    for item in running:
        if not finish_job(paths, item):
            active.append(item)
    running[:] = active
    while len(running) < max_jobs:
        pending = list_job_files(paths["pending"])
        if not pending:
            break
        running.append(start_job(paths, pending[0], packer_path))


def command_serve(args: argparse.Namespace) -> int:
    if args.max_jobs <= 0:
        raise WorkerFailure("--max-jobs must be positive")
    paths = ensure_queue(args.queue_dir.resolve())
    packer_path = args.packer.resolve()
    if not packer_path.exists():
        raise WorkerFailure(f"missing packer: {packer_path}")
    running: List[Dict[str, Any]] = []
    while True:
        serve_once(paths, packer_path, args.max_jobs, running)
        if args.once and not running and not list_job_files(paths["pending"]):
            return 0
        time.sleep(args.poll_interval)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GLM52 B12x resident pack worker queue")
    subparsers = parser.add_subparsers(dest="command", required=True)
    common_queue = argparse.ArgumentParser(add_help=False)
    common_queue.add_argument("--queue-dir", required=True, type=Path)
    submit = subparsers.add_parser("submit", parents=[common_queue])
    submit.add_argument("--job-id", default="")
    submit.add_argument("--model-dir", required=True)
    submit.add_argument("--aot-manifest", required=True)
    submit.add_argument("--layers", required=True)
    submit.add_argument("--output-dir", required=True)
    submit.add_argument("--local-jobs", default=2, type=int)
    submit.add_argument("--verify-reused-sha256", action="store_true")
    submit.set_defaults(function=command_submit)
    queue_depth_parser = subparsers.add_parser("queue-depth", parents=[common_queue])
    queue_depth_parser.add_argument("--max-jobs", default=2, type=int)
    queue_depth_parser.set_defaults(function=command_queue_depth)
    status = subparsers.add_parser("status", parents=[common_queue])
    status.add_argument("--job-id", default="")
    status.set_defaults(function=command_status)
    serve = subparsers.add_parser("serve", parents=[common_queue])
    serve.add_argument("--max-jobs", default=2, type=int)
    serve.add_argument("--once", action="store_true")
    serve.add_argument("--packer", default=repo_root() / "tools" / "glm52_b12x_resident_pack.py", type=Path)
    serve.add_argument("--poll-interval", default=0.25, type=float)
    serve.set_defaults(function=command_serve)
    return parser


def main(argv: Sequence[str]) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.function(args))


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except WorkerFailure as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
