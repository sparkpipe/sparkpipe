#!/usr/bin/env python3
"""Lane-3 k3 shared-socket deployment gates (stdlib only).

tools/k3_multidev_lane.py builds the private multidev deployment for
developer lane 3 (k3, TP4xPP4, spark0..sparkf). This gate holds the lane
contract:

  1. the deployment keeps the hybrid identity rank_index == stage_index
     == 0..15 and 16 unique per-host control endpoints inside the lane-3
     control block 23048..23063;
  2. the weightd socket is wired through deployment.weightd.socket_path
     and the KV backing stays a finite cap under the private runtime
     root (never the shared sparkdata tree);
  3. every adapter configuration binds its host TCP collective inside
     53048..53051 with STEP-ordered peers, points at the rank's deployed
     pack, and keeps the device-collective session table inside the lane
     blocks (53052..53063) so a future listener cannot leak into another
     lane's range;
  4. no port of any kind lands outside lane 3's three blocks
     (23048-23063, 53048-53063, 64048-64063), and no host listens one
     number twice;
  5. the mesh map is the identity permutation (logical rank = index in
     --nodes), the wrapper scripts parse, and the generator's --check
     mode reproduces its own output byte for byte;
  6. the family wrapper upholds the shared-socket template queue contract
     (tools/devcycle/templates/run-family-job.sh.template): attempt id,
     job namespace, size/rank bounds and queue-reserved port membership
     all fail closed before any filesystem work, and a fully reserved
     lane reaches the shared-socket liveness check.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import k3_multidev_lane as lane  # noqa: E402

LANE_BLOCKS = (
    set(range(lane.CONTROL_BASE, lane.CONTROL_BASE + 16)),
    set(range(lane.COLLECTIVE_BASE, lane.COLLECTIVE_BASE + 16)),
    set(range(lane.TRANSPORT_BASE, lane.TRANSPORT_BASE + 16)),
)
ALL_LANE_PORTS = LANE_BLOCKS[0] | LANE_BLOCKS[1] | LANE_BLOCKS[2]
HEX = lane.HEX


def check(condition, failures, message):
    if not condition:
        failures.append(message)


def deployment_gates(deployment, runtime_root, socket, failures):
    check(deployment["schema_version"] == 2, failures,
          "deployment schema_version must be 2")
    check(deployment["weightd"]["socket_path"] == socket, failures,
          "weightd socket path must be the shared socket")
    limits = deployment["runtime_limits"]
    # residentd's runtime-limits check fails closed on zeros: kv_physical
    # >= max_active_sequences and kv_logical >= resident_sequence_capacity.
    # k3's honest bound: resident capacity x the adapter's kv_pages per
    # sequence (the per-sequence token ceiling the seam commits to).
    expected_pages = limits["resident_sequence_capacity"] * lane.KV_PAGES_PER_SEQUENCE
    for key in ("kv_logical_page_capacity", "kv_physical_page_capacity"):
        check(limits[key] == expected_pages, failures,
              f"{key} must equal resident_capacity x kv_pages "
              f"({expected_pages}), got {limits[key]}")
    check(limits["kv_physical_page_capacity"] >= limits["max_active_sequences"],
          failures, "kv_physical_page_capacity must cover max_active_sequences")
    check(limits["kv_logical_page_capacity"] >= limits["resident_sequence_capacity"],
          failures, "kv_logical_page_capacity must cover resident capacity")
    nodes = deployment["nodes"]
    check(len(nodes) == 16, failures, f"expected 16 nodes, got {len(nodes)}")
    endpoints = set()
    for i, node in enumerate(nodes):
        check(node["rank_index"] == i, failures,
              f"node {i}: rank_index {node['rank_index']}")
        check(node["stage_index"] == i, failures,
              f"node {i}: stage_index {node['stage_index']} != linear rank")
        check(node["runtime_root"] == runtime_root, failures,
              f"node {i}: runtime_root must be the private queue root")
        backing = node["kv_backing_directory"]
        check(backing.startswith(runtime_root + "/"), failures,
              f"node {i}: kv backing {backing} escapes the private root")
        check(0 < node["kv_backing_maximum_bytes"] < 1 << 40, failures,
              f"node {i}: kv backing cap must be finite")
        endpoint = node["control_endpoint"]
        host = f"spark{HEX[i]}"
        check(endpoint["host"] == host, failures,
              f"node {i}: control host {endpoint['host']} != {host}")
        check(endpoint["port"] == lane.CONTROL_BASE + i, failures,
              f"node {i}: control port {endpoint['port']}")
        endpoints.add((endpoint["host"], endpoint["port"]))
    check(len(endpoints) == 16, failures, "control endpoints must be unique")
    transport = deployment["transport"]
    check(lane.TRANSPORT_BASE <= transport["control_port_base"]
          <= lane.TRANSPORT_BASE + 15, failures,
          "transport control base must sit inside the lane transport block")


def adapter_gates(config, rank, failures, per_host_ports):
    host = f"spark{HEX[rank]}"
    stage, tp = rank // 4, rank % 4
    check(config["tp_degree"] == 4, failures, f"{host}: tp_degree")
    check(config["tp_rank"] == tp, failures, f"{host}: tp_rank")
    check(config["world_size"] == 16, failures, f"{host}: world_size")
    expected_pack = (f"/home/{host}/sparkdata/k3.mxfp4.tp4pp4/packs/"
                     f"k3.stage{stage}.rank0{tp}.pack")
    check(config["stage_pack_path"] == expected_pack, failures,
          f"{host}: stage_pack_path {config['stage_pack_path']}")

    collective = config["tp_collective"]
    check(collective["listen_port"] == lane.TP_COLLECTIVE_PORT + tp,
          failures, f"{host}: tp_collective listen_port")
    peers = collective["peers"]
    check(len(peers) == 2, failures, f"{host}: expected 2 STEP peers")
    for peer_index, partner in ((0, tp ^ 1), (1, tp ^ 2)):
        # peers are numeric IPv4 literals: SparkTpCollectiveCreate
        # validates with inet_pton and rejects hostnames outright
        expected = (f"{lane.HOST_ADDRESSES[f'spark{HEX[stage * 4 + partner]}']}:"
                    f"{lane.TP_COLLECTIVE_PORT + partner}")
        check(peers[peer_index] == expected, failures,
              f"{host}: STEP peer {peer_index} {peers[peer_index]} != {expected}")
    per_host_ports.setdefault(host, set()).add(collective["listen_port"])
    per_host_ports[host].add(lane.CONTROL_BASE + rank)

    device = config["device_collective"]
    check(device["local_host"] == host, failures, f"{host}: device local_host")
    check(len(device["peer_hosts"]) == 4, failures, f"{host}: device peers")
    check(device["peer_hosts"] == [f"spark{HEX[stage * 4 + t]}"
                                   for t in range(4)], failures,
          f"{host}: device peer_hosts must be the TP group")
    table = device["session_ports"]
    check(len(table) == 4, failures, f"{host}: session table rows")
    sessions = set()
    for a, row in enumerate(table):
        check(len(row) == 4, failures, f"{host}: session row {a} length")
        for b, value in enumerate(row):
            if a == b:
                check(value == 0, failures,
                      f"{host}: session[{a}][{b}] diagonal must be 0")
            else:
                check(value in range(53052, 53064), failures,
                      f"{host}: session[{a}][{b}]={value} outside 53052..53063")
                check(value not in sessions, failures,
                      f"{host}: session value {value} used twice")
                sessions.add(value)
                if a == tp:
                    per_host_ports[host].add(value)
    per_host_ports[host].add(device["listen_port"])


def wrapper_contract_gates(failures):
    """Exercise the template-aligned queue contract of the family wrapper.

    Every case must fail closed BEFORE any filesystem work (the contract
    and socket checks precede the first mkdir), so driving the real script
    with a synthetic environment is side-effect free.
    """
    script = ROOT / "tools/k3_multidev_run_family.sh"
    attempt = "a" * 32
    base = {
        "SPARK_QUEUE_ATTEMPT": attempt,
        "SPARK_QUEUE_RUNTIME_ROOT": f"/tmp/sparkqueue-{attempt}",
        "SPARK_QUEUE_RANK": "0",
        "SPARK_QUEUE_SIZE": "16",
        "K3_WEIGHTD_SOCKET": "/nonexistent/weightd.sock",
        "K3_EXPERT_POOL_BYTES": "1",
        "K3_SPINE_BUDGET_BYTES": "1",
        "HOME": "/nonexistent",
    }
    lane_ranges = "23048:23063,53048:53063,64048:64063"

    def run(env):
        # hermetic: strip every queue/lane variable the host may carry -
        # under a real queue job SPARK_QUEUE_ATTEMPT/RANK/SIZE/RUNTIME_ROOT
        # leak in through os.environ and flip which fail-closed case fires
        complete = {key: value for key, value in os.environ.items()
                    if not (key.startswith("SPARK_QUEUE_")
                            or key.startswith("K3_")
                            or key == "SPARK_WEIGHTD_SOCKET")}
        complete.update(env)
        return subprocess.run(["bash", str(script)], env=complete,
                              capture_output=True, text=True)

    cases = [
        ("missing attempt id",
         {k: v for k, v in base.items()
          if k != "SPARK_QUEUE_ATTEMPT"},
         "authoritative spark queue"),
        ("malformed attempt id",
         dict(base, SPARK_QUEUE_ATTEMPT="z" * 32),
         "bad attempt id"),
        ("short attempt id",
         dict(base, SPARK_QUEUE_ATTEMPT="abc"),
         "bad attempt id"),
        ("job namespace mismatch",
         dict(base, SPARK_QUEUE_RUNTIME_ROOT="/tmp/elsewhere"),
         "unexpected job namespace"),
        ("queue size mismatch",
         dict(base, SPARK_QUEUE_SIZE="4"),
         "SPARK_QUEUE_SIZE must be 16"),
        ("rank out of range",
         dict(base, SPARK_QUEUE_RANK="16"),
         "SPARK_QUEUE_RANK must be 0..15"),
        ("no reserved ports",
         dict(base),
         "queue reserved no ports"),
        ("ports outside the lane blocks",
         dict(base, SPARK_QUEUE_PORTS="7000:7001"),
         "not inside a queue-reserved range"),
    ]
    for name, env, expected in cases:
        result = run(env)
        check(result.returncode != 0, failures,
              f"wrapper case '{name}' must fail closed")
        check(expected in result.stderr, failures,
              f"wrapper case '{name}': expected '{expected}' in stderr, "
              f"got: {result.stderr.strip()[:200]}")

    accepted = run(dict(base, SPARK_QUEUE_PORTS=lane_ranges))
    check(accepted.returncode != 0, failures,
          "wrapper must still fail without a live shared socket")
    check("not a live socket" in accepted.stderr, failures,
          f"wrapper with reserved lane ranges should reach the socket "
          f"check, got: {accepted.stderr.strip()[:200]}")


def main() -> int:
    failures = []
    with tempfile.TemporaryDirectory() as temporary:
        runtime_root = str(Path(temporary) / "runtime")
        socket_path = str(Path(temporary) / "weightd.shared.sock")
        output = Path(temporary) / "generated"
        subprocess.run(
            [sys.executable, str(ROOT / "tools/k3_multidev_lane.py"),
             "--runtime-root", runtime_root,
             "--weightd-socket", socket_path,
             "--output-dir", str(output)],
            check=True, capture_output=True)

        deployment = json.loads((output / "deployment.json").read_text())
        deployment_gates(deployment, runtime_root, socket_path, failures)

        per_host_ports = {}
        identifiers = set()
        for rank in range(16):
            config = json.loads(
                (output / f"adapter.spark{HEX[rank]}.json").read_text())
            adapter_gates(config, rank, failures, per_host_ports)
            identifiers.add(
                config["device_collective"]["collective_identifier"])
        check(len(identifiers) == 4, failures,
              "device collective identifiers must differ per PP stage")

        for host, ports in per_host_ports.items():
            check(len(ports) == len(set(ports)), failures,
                  f"{host}: duplicate listener {ports - set(ports)}")
            outside = ports - ALL_LANE_PORTS
            check(not outside, failures,
                  f"{host}: ports outside lane 3 blocks: {sorted(outside)}")

        check(lane.MESH_RANKS == ",".join(str(i) for i in range(16)),
              failures, "mesh map must be the identity permutation")

        check_run = subprocess.run(
            [sys.executable, str(ROOT / "tools/k3_multidev_lane.py"),
             "--runtime-root", runtime_root,
             "--weightd-socket", socket_path,
             "--output-dir", str(output), "--check"],
            capture_output=True, text=True)
        check(check_run.returncode == 0, failures,
              f"--check failed: {check_run.stderr.strip()}")

    for script in ("tools/k3_multidev_run_family.sh",
                   "tools/k3_multidev_experts_manifest.sh"):
        if shutil.which("bash") is None:
            continue
        syntax = subprocess.run(
            ["bash", "-n", str(ROOT / script)],
            capture_output=True, text=True)
        check(syntax.returncode == 0, failures,
              f"{script}: bash -n failed: {syntax.stderr.strip()}")

    if shutil.which("bash") is not None:
        wrapper_contract_gates(failures)

    if failures:
        for failure in failures:
            print(f"FAIL {failure}")
        return 1
    print("k3 multidev lane gates PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
