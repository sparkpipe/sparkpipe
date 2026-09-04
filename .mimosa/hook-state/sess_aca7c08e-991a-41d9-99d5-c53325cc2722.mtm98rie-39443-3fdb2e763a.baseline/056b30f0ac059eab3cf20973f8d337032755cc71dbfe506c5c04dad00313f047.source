#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import pathlib
import subprocess
import sys
import tempfile
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "hardware"))
from hardware_common import canonical_json_bytes, sha256_bytes  # noqa: E402

SOURCE_SHA = "a" * 64


def write_json(path: pathlib.Path, document: object) -> None:
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def make_plan(executable: pathlib.Path) -> dict[str, object]:
    scope = {"topology": "unit", "node": "spark0"}
    parameters = {"candidate": "identity", "iterations": 1}
    basis = {"question_id": "GB10-IDENTITY-001", "probe_id": "cuda_characterize", "scope": scope, "parameters": parameters}
    cell_id = sha256_bytes(canonical_json_bytes(basis))
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
            "nodes": [{"name": "spark0", "rank": 0, "nvme_device": "/dev/null", "addresses": ["127.0.0.1"]}],
            "fabrics": [{"name": "unit", "kind": "switch", "speed_gbps": 100, "mtu_bytes": 1500}],
        },
        "coverage": {
            "GB10-IDENTITY-001": {
                "probe_id": "cuda_characterize",
                "applicable": True,
                "expected_observation_count": 1,
                "axes": {"candidate": ["identity"], "iterations": [1]},
            }
        },
        "jobs": [{
            "job_index": 0,
            "cell_id": cell_id,
            "question_id": "GB10-IDENTITY-001",
            "probe_id": "cuda_characterize",
            "executable": str(executable),
            "scope": scope,
            "parameters": parameters,
        }],
    }
    plan["plan_id"] = sha256_bytes(canonical_json_bytes(plan))
    return plan


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
        executable = directory / "fake_probe.py"
        executable.write_text(
            """#!/usr/bin/env python3
import argparse, json, time
p=argparse.ArgumentParser()
p.add_argument('--question', required=True); p.add_argument('--source-package-sha256', required=True)
p.add_argument('--run-id', required=True); p.add_argument('--topology', required=True)
p.add_argument('--node', required=True); p.add_argument('--output', required=True)
p.add_argument('--candidate'); p.add_argument('--iterations', type=int, default=1)
p.add_argument('--sleep', type=float, default=0)
a, unknown=p.parse_known_args(); time.sleep(a.sleep)
d={'schema_version':1,'receipt_kind':'spark_hardware_probe','run_id':a.run_id,
'probe_id':'cuda_characterize','source_identity':{'source_package_sha256':a.source_package_sha256},
'scope':{'topology':a.topology,'node':a.node},'answers':[{'question_id':a.question,'status':'measured',
'summary':{'fake':True},'observations':[{'parameters':{'candidate':a.candidate,'iterations':a.iterations},
'metrics':{'device_count':1,'integrity_pass':True}}]}]}
open(a.output,'w').write(json.dumps(d))
""",
            encoding="utf-8",
        )
        executable.chmod(0o755)
        plan = make_plan(executable)
        plan_path = directory / "plan.json"
        write_json(plan_path, plan)
        receipts = directory / "receipts"
        config = {
            "schema_version": 1,
            "node": "spark0",
            "run_id": "runner-unit",
            "receipt_directory": str(receipts),
            "executables": {"cuda_characterize": str(executable)},
            "providers": {},
            "probe_timeout_seconds": 5,
        }
        config_path = directory / "config.json"
        write_json(config_path, config)
        runner = ROOT / "tools" / "hardware" / "run_probe_job.py"
        run([sys.executable, str(runner), "--plan", str(plan_path), "--config", str(config_path)])
        cell_id = str(plan["jobs"][0]["cell_id"])
        receipt_path = receipts / f"{cell_id}.json"
        receipt_bytes = receipt_path.read_bytes()
        first_digest = hashlib.sha256(receipt_bytes).hexdigest()
        run([sys.executable, str(runner), "--plan", str(plan_path), "--config", str(config_path), "--resume"])
        assert hashlib.sha256(receipt_path.read_bytes()).hexdigest() == first_digest

        receipt_path.unlink()
        lock_path = receipt_path.with_suffix(".lock")
        lock_path.write_text("held\n", encoding="utf-8")
        run([sys.executable, str(runner), "--plan", str(plan_path), "--config", str(config_path)], 2)
        lock_path.unlink()

        slow = directory / "slow_probe.py"
        slow.write_text(executable.read_text(encoding="utf-8").replace("p.add_argument('--sleep', type=float, default=0)", "p.add_argument('--sleep', type=float, default=3)"), encoding="utf-8")
        slow.chmod(0o755)
        config["executables"]["cuda_characterize"] = str(slow)
        config["probe_timeout_seconds"] = 1
        write_json(config_path, config)
        started = time.monotonic()
        run([sys.executable, str(runner), "--plan", str(plan_path), "--config", str(config_path)], 2)
        assert time.monotonic() - started < 3.0
        failed = json.loads(receipt_path.read_text(encoding="utf-8"))
        assert failed["probe_receipt"]["answers"][0]["status"] == "failed"
        assert "timed out" in failed["probe_receipt"]["answers"][0]["error"]

    print("PASS exact hardware job runner identity, resume, lock, and timeout behavior")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
