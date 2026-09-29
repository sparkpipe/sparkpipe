#!/usr/bin/env python3
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import glm53full_lane

ADAPTER = ROOT / "modules/glm52_resident_decode_stage/source/spark_glm52_serving_adapter.c"
CONTRACT = ROOT / "model_contracts/glm53_full_authoritative.json"
SOCKET = "/tmp/spark_weightd.sock"


def adapter_members():
    source = ADAPTER.read_text()
    block = re.search(r"SparkGlm52ServingConfigurationMembers\[\] =\s*\{(.*?)\};", source, re.S).group(1)
    return tuple(re.findall(r'"([a-z_0-9]+)"', block))


def rendered(lane, codec="fp8", sequences=8, rows=16, positions=4096, inflight=1):
    arguments = argparse.Namespace(lane=lane, codec=codec, socket=SOCKET, kv_backing_bytes=4 << 30,
                                   max_sequence_positions=positions, execution_row_capacity=rows,
                                   sequences=sequences, inflight=inflight)
    return {name: json.loads(text) for name, text in glm53full_lane.render(arguments).items()}


def lane_problems(lane, codec, files, members, revision):
    failures = []
    ports = glm53full_lane.lane_ports(lane)
    contract = json.loads(CONTRACT.read_text())
    deployment = files["model_resident.json"]
    limits = deployment["runtime_limits"]
    if deployment["eos_token_ids"] != contract["geometry"]["eos_token_ids"]:
        failures.append("eos ids differ from the authoritative contract")
    if limits["kv_physical_page_capacity"] != limits["max_active_sequences"] * (limits["max_sequence_positions"] // 64):
        failures.append("kv pages do not cover every resident sequence")
    if deployment["transport"]["control_port_base"] != ports["transport"]:
        failures.append("transport base")
    for rank, node in enumerate(deployment["nodes"]):
        host = glm53full_lane.HOSTS[rank]
        stage = files[node["adapter_configuration_path"]]
        if tuple(stage) != members:
            failures.append(f"rank {rank}: stage members {tuple(stage)} != adapter {members}")
        if node["runtime_root"] != f"/home/{host}/glmfull-lane{lane}/root" or not node["kv_backing_directory"].startswith(node["runtime_root"] + "/"):
            failures.append(f"rank {rank}: runtime root")
        if node["control_endpoint"] != {"kind": "tcp", "host": host, "port": ports["control"] + rank}:
            failures.append(f"rank {rank}: control endpoint")
        if node["node_target"] != f"cuda.sm121.glm52.resident_decode_stage.bf16.expert_{codec}":
            failures.append(f"rank {rank}: node target")
        if stage["stage_pack_path"] != f"packs/glm53full.{codec}.tp16-rank{rank}.glm52sp":
            failures.append(f"rank {rank}: pack path")
        if stage["tp_rank"] != rank or stage["tp_degree"] != 16:
            failures.append(f"rank {rank}: tp")
        if stage["model_revision"] != revision:
            failures.append(f"rank {rank}: model revision differs from the build identity")
        if stage["max_sequence_positions"] != limits["max_sequence_positions"] or stage["execution_row_capacity"] != limits["max_input_rows"]:
            failures.append(f"rank {rank}: adapter limits differ from the engine's")
        collective = stage["tp_collective"]
        if collective["listen_port"] != ports["collective"] + rank or collective["peer_ports"] != list(range(ports["collective"], ports["collective"] + 16)):
            failures.append(f"rank {rank}: collective ports")
    return failures


def api_host_problems():
    failures = []
    settings = {"GLMFULL_LANE": "6", "GLMFULL_CODEC": "fp8", "GLMFULL_FIRMWARE": "/nonexistent", "GLMFULL_WEIGHTD_SOCKET": SOCKET,
                "GLMFULL_EXPERT_POOL_BYTES": "1", "GLMFULL_SPINE_BUDGET_BYTES": "1", "GLMFULL_MEMORY_MAX": "1G",
                "GLMFULL_SEQUENCES": "8", "GLMFULL_ROWS": "16", "GLMFULL_POSITIONS": "2048", "GLMFULL_INFLIGHT": "1",
                "GLMFULL_API_PORT": "8446", "GLMFULL_API_BUILD": "/opt/api", "GLMFULL_API_TOKENIZER": "/opt/tokenizer.json"}
    with tempfile.TemporaryDirectory() as directory:
        fake = Path(directory)
        log = fake / "remote.log"
        for tool in ("ssh", "scp"):
            (fake / tool).write_text(f'#!/bin/sh\nprintf \'{tool}\' >> "{log}"\nfor a in "$@"; do printf \' [%s]\' "$a" >> "{log}"; done\necho >> "{log}"\n')
            (fake / tool).chmod(0o755)
        path = f"{fake}:{Path(sys.executable).parent}:/usr/bin:/bin"

        def lane(command, extra):
            if log.exists():
                log.unlink()
            result = subprocess.run(["bash", str(ROOT / "tools/glm53full_lane.sh"), *command], capture_output=True, text=True,
                                    env={"PATH": path, "HOME": directory, **settings, **extra})
            return result, (log.read_text().splitlines() if log.exists() else [])

        for command in (["api"], ["api-stop"], ["decode", "1,2", "3"]):
            for host in ("sparkf", "spark0", "sparka"):
                result, calls = lane(command, {"GLMFULL_API_HOST": host})
                if result.returncode != 2 or "refused" not in result.stderr or calls:
                    failures.append(f"{command[0]} with GLMFULL_API_HOST={host}: rc={result.returncode} remote calls={len(calls)}")
            for extra in ({}, {"GLMFULL_API_HOST": "rtx5090"}):
                result, calls = lane(command, extra)
                hosts = set()
                for call in calls:
                    words = [w.strip("[]") for w in call.split(" ")[1:]]
                    if call.startswith("ssh "):
                        hosts.add([w for w in words if not w.startswith("-") and "=" not in w][0])
                    else:
                        hosts.update(w.split(":")[0] for w in words if ":" in w)
                if result.returncode != 0 or not calls or hosts != {"rtx5090"}:
                    failures.append(f"{command[0]} {extra}: rc={result.returncode} hosts={sorted(hosts)} stderr={result.stderr[-200:]}")
    return failures


def main():
    failures = []
    members = adapter_members()
    for codec in ("fp8", "bf16"):
        revision = subprocess.run([sys.executable, str(ROOT / "tools/glm52_model_contract.py"), "--print-build-identity", codec],
                                  capture_output=True, text=True, check=True).stdout.split()[0]
        for lane in range(16):
            failures += [f"lane {lane} {codec}: {problem}" for problem in lane_problems(lane, codec, rendered(lane, codec), members, revision)]
    with tempfile.TemporaryDirectory() as directory:
        command = [sys.executable, str(ROOT / "tools/glm53full_lane.py"), "--lane", "6", "--codec", "fp8", "--socket", SOCKET,
                   "--kv-backing-bytes", str(4 << 30), "--max-sequence-positions", "4096", "--execution-row-capacity", "16",
                   "--sequences", "8", "--inflight", "1", "--output", directory]
        subprocess.run(command, check=True, capture_output=True)
        if subprocess.run(command + ["--check"], capture_output=True).returncode != 0:
            failures.append("--check reports drift on a fresh render")
        stage = Path(directory) / "config/stage_03.json"
        stage.write_text(stage.read_text().replace('"tp_rank": 3', '"tp_rank": 4'))
        if subprocess.run(command + ["--check"], capture_output=True).returncode == 0:
            failures.append("--check misses drift")
    script = subprocess.run(["bash", str(ROOT / "tools/glm53full_lane.sh"), "status"], capture_output=True, text=True,
                            env={"PATH": "/usr/bin:/bin"})
    if script.returncode == 0 or "GLMFULL_LANE" not in script.stderr:
        failures.append("glm53full_lane.sh runs without its settings")
    failures += api_host_problems()
    for failure in failures:
        print(failure)
    print(f"glm53full lane: {'FAIL' if failures else 'PASS'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
