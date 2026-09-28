#!/usr/bin/env python3
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import ling_lane
import ling_multidev_lane

ADAPTER = ROOT / "modules/ling_resident_decode_stage/source/spark_ling_serving_adapter.c"
SOCKET = "/tmp/spark_weightd.sock"


def adapter_members():
    source = ADAPTER.read_text()
    block = re.search(r"SparkLingServingConfigurationMembers\[\] =\s*\{(.*?)\};", source, re.S).group(1)
    return tuple(re.findall(r'"([a-z_0-9]+)"', block))


def rendered(lane, codec="bf16"):
    files = ling_lane.render(lane, codec, SOCKET, 8 << 30, 32768, 128)
    return {name: json.loads(text) for name, text in files.items()}


def check(condition, message, failures):
    if not condition:
        failures.append(message)


def lane_problems(lane, files, members):
    failures = []
    ports = ling_lane.lane_ports(lane)
    deployment = files["model_resident.json"]
    check(deployment["eos_token_ids"] == [156895], f"lane {lane}: eos", failures)
    check(deployment["weightd"]["socket_path"] == SOCKET, f"lane {lane}: socket", failures)
    check(deployment["transport"]["control_port_base"] == ports["transport"], f"lane {lane}: transport base", failures)
    for rank, node in enumerate(deployment["nodes"]):
        host = ling_lane.HOSTS[rank]
        check(node["rank_index"] == rank and node["stage_index"] == rank, f"lane {lane}: rank {rank} identity", failures)
        check(node["runtime_root"] == f"/home/{host}/ling-lane{lane}/root", f"lane {lane}: rank {rank} runtime root", failures)
        check(node["kv_backing_directory"].startswith(node["runtime_root"] + "/"), f"lane {lane}: rank {rank} kv outside root", failures)
        check(0 < node["kv_backing_maximum_bytes"], f"lane {lane}: rank {rank} kv cap", failures)
        check(node["control_endpoint"] == {"kind": "tcp", "host": host, "port": ports["control"] + rank}, f"lane {lane}: rank {rank} control", failures)
        stage = files[node["adapter_configuration_path"]]
        check(tuple(stage) == members, f"lane {lane}: rank {rank} stage members {tuple(stage)} != adapter {members}", failures)
        check(stage["tp_rank"] == rank and stage["tp_degree"] == 16, f"lane {lane}: rank {rank} tp", failures)
        check(stage["stage_pack_path"] == f"packs/ling.bf16.tp16.rank{rank:x}.sp", f"lane {lane}: rank {rank} pack", failures)
        collective = stage["tp_collective"]
        check(collective["listen_port"] == ports["collective"] + rank, f"lane {lane}: rank {rank} listen", failures)
        check(collective["peer_ports"] == list(range(ports["collective"], ports["collective"] + 16)), f"lane {lane}: peer ports", failures)
        check(collective["collective_identifier"] != 0, f"lane {lane}: collective disabled", failures)
        for matrix, base in (("session_ports", ports["session"]), ("session_ports_hc", ports["session"] + 16)):
            for row_index, row in enumerate(collective[matrix]):
                check(row[row_index] == 0 and all(value == base + column for column, value in enumerate(row) if column != row_index), f"lane {lane}: {matrix} row {row_index}", failures)
    return failures


def main():
    failures = []
    members = adapter_members()
    check(members == ling_lane.ADAPTER_MEMBERS, f"tool member list {ling_lane.ADAPTER_MEMBERS} != adapter {members}", failures)
    bound = {}
    identifiers = set()
    for lane in range(16):
        files = rendered(lane)
        failures += lane_problems(lane, files, members)
        stage = files["config/stage_00.json"]
        identifiers.add(stage["tp_collective"]["collective_identifier"])
        for rank in range(16):
            node = files["model_resident.json"]["nodes"][rank]
            for port in (node["control_endpoint"]["port"], files[node["adapter_configuration_path"]]["tp_collective"]["listen_port"]):
                key = (node["transport_host"], port)
                check(key not in bound, f"lane {lane} rebinds {key} of lane {bound.get(key)}", failures)
                bound[key] = lane
    check(len(identifiers) == 16, "collective identifiers repeat across lanes", failures)
    legacy = ling_multidev_lane.stage_config(3, "bf16")
    ours = rendered(9)["config/stage_03.json"]
    for key in members:
        if key != "tp_collective":
            check(legacy[key] == ours[key], f"lane 9 {key} differs from ling_multidev_lane", failures)
    check(set(legacy["tp_collective"]) == set(ours["tp_collective"]), "tp_collective member set differs from ling_multidev_lane", failures)
    check(rendered(10, "fp8")["config/stage_05.json"]["stage_pack_path"] == "packs/ling.fp8.tp16.rank5.sp", "fp8 pack path", failures)
    for arguments in ((16, "bf16", SOCKET), (10, "int8", SOCKET), (10, "bf16", "relative.sock"), (-1, "bf16", SOCKET)):
        try:
            ling_lane.render(*arguments, 8 << 30, 32768, 128)
            failures.append(f"render accepted {arguments}")
        except SystemExit:
            pass
    with tempfile.TemporaryDirectory() as directory:
        command = [sys.executable, str(ROOT / "tools/ling_lane.py"), "--lane", "10", "--codec", "bf16", "--socket", SOCKET,
                   "--kv-backing-bytes", str(8 << 30), "--max-sequence-positions", "32768", "--execution-row-capacity", "128", "--output", directory]
        check(subprocess.run(command, capture_output=True).returncode == 0, "render run failed", failures)
        check(subprocess.run(command + ["--check"], capture_output=True).returncode == 0, "check of fresh render failed", failures)
        path = Path(directory) / "config/stage_07.json"
        path.write_text(path.read_text().replace('"tp_rank": 7', '"tp_rank": 8'))
        check(subprocess.run(command + ["--check"], capture_output=True).returncode != 0, "check missed a drifted stage config", failures)
    check(subprocess.run(["bash", "-n", str(ROOT / "tools/ling_lane.sh")]).returncode == 0, "ling_lane.sh does not parse", failures)
    unset = subprocess.run(["bash", str(ROOT / "tools/ling_lane.sh"), "status"], capture_output=True, text=True, env={"PATH": "/usr/bin:/bin"})
    check(unset.returncode != 0 and "LING_LANE" in unset.stderr, "ling_lane.sh runs without its required settings", failures)
    if failures:
        for failure in failures:
            print("FAIL", failure)
        return 1
    print(f"PASS ling lane deployments: 16 lanes x 16 ranks, {len(bound)} bound ports disjoint, adapter member set {len(members)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
