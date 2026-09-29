#!/usr/bin/env python3
import argparse
import json
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
    for failure in failures:
        print(failure)
    print(f"glm53full lane: {'FAIL' if failures else 'PASS'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
