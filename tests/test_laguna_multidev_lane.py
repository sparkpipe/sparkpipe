#!/usr/bin/env python3
"""Lane-8 laguna shared-socket deployment gates (stdlib only).

tools/laguna_multidev_lane.py builds the private multidev deployment for
developer lane 8 (laguna, TP8xPP2, spark0..sparkf). This gate holds the
lane contract:

  1. the deployment keeps the hybrid identity rank_index == stage_index
     == 0..15 (the adapter derives tp_rank = stage_index % 8 and the
     pipeline stage = stage_index // 8 from it) and 16 unique per-host
     control endpoints inside the lane-8 control block 23128..23143;
  2. the deployment carries the contract's EOS tokens, the weightd
     socket is the fleet weightd (/tmp/spark_weightd.sock) wired through
     deployment.weightd.socket_path, any other socket is refused, and the
     KV backing stays a finite cap under the private runtime root (never
     the shared sparkdata tree);
  3. every adapter configuration carries the serving adapter's EXACT
     member set (10 top-level + 15 tp_collective members - a missing OR
     extra member is a load error), binds its host TCP collective inside
     53128..53135 GROUP-LOCALLY (both PP stages reuse the block on
     disjoint hosts), points at the rank's deployed pack under the
     GLOBAL-rank filename convention (stage{rank//8}.rank{rank}.lgsp -
     the M1 inventory finding), and keeps the session matrices inside
     the lane-8 session block 23680..23743;
  4. no port of any kind lands outside lane 8's four blocks
     (23128-23143, 53128-53143, 64128-64143, 23680-23743), no host
     uses one number twice across its control, transport, collective
     and session_ports row, and session_ports_hc mirrors session_ports;
  5. the mesh map is the identity permutation (logical rank = index in
     --nodes), the wrapper scripts parse, and the generator's --check
     mode reproduces its own output byte for byte;
  6. the family wrapper, run with synthetic queue environments, fails
     closed with its stated reason before any filesystem work on a bad
     attempt id, job namespace, size/rank, unreserved port, dead weightd
     socket and node-order mismatch, and defaults to the fleet weightd
     socket;
  7. the .lgsp routed-expert manifest generator emits a loader-verified
     v2 sidecar from a synthetic pack directory with correct per-expert
     spans (uniform bf16 interleave), and --budgets derives the exact
     expert-pool / spine numbers from that sidecar.
"""
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import laguna_multidev_lane as lane  # noqa: E402
from lane_wrapper_contract import (  # noqa: E402
    FLEET_WEIGHTD_SOCKET, STALE_SHARED_UNIT_SOCKET, LaneWrapper, LiveSocket)

LANE_BLOCKS = (
    set(range(lane.CONTROL_BASE, lane.CONTROL_BASE + 16)),
    set(range(lane.COLLECTIVE_BASE, lane.COLLECTIVE_BASE + 16)),
    set(range(lane.TRANSPORT_BASE, lane.TRANSPORT_BASE + 16)),
    set(range(lane.SESSION_BASE, lane.SESSION_BASE + 64)),
)
ALL_LANE_PORTS = LANE_BLOCKS[0] | LANE_BLOCKS[1] | LANE_BLOCKS[2] | LANE_BLOCKS[3]
HEX = lane.HEX
CONTRACT_EOS = json.loads(
    (ROOT / "model_contracts/laguna_authoritative.json").read_text())[
        "tokens"]["eos_token_ids"]

ADAPTER_MEMBERS = {
    "schema_version", "model_revision", "expert_weight_codec",
    "stage_pack_path", "max_sequence_positions", "execution_row_capacity",
    "decode_split_context_threshold", "tp_degree", "tp_rank",
    "tp_collective"}
TP_COLLECTIVE_MEMBERS = {
    "backend", "backend_module_path", "collective_identifier",
    "listen_port", "connect_timeout_milli", "operation_timeout_milli",
    "peer_hosts", "peer_ports", "algorithms",
    "direct_all_to_all_max_payload_bytes", "split_ring_min_payload_bytes",
    "rail_peer_hosts", "step_rail_indices", "session_ports",
    "session_ports_hc"}


def check(condition, failures, message):
    if not condition:
        failures.append(message)


def deployment_gates(deployment, runtime_root, socket, failures,
                     per_host_ports):
    check(deployment["schema_version"] == 2, failures,
          "deployment schema_version must be 2")
    check(deployment["eos_token_ids"] == CONTRACT_EOS, failures,
          f"eos_token_ids {deployment.get('eos_token_ids')} must be the "
          f"contract's {CONTRACT_EOS}")
    check(deployment["weightd"]["socket_path"] == socket, failures,
          "weightd socket path must be the fleet weightd")
    check(deployment["driver"]["program_name"] == "resident_decode",
          failures, "driver program must be resident_decode")
    check(deployment["adapter"]["shared_object_path"]
          == "lib/model_serving_adapter.so", failures,
          "adapter path must be the private lib install")
    limits = deployment["runtime_limits"]
    for key in ("max_inflight_submissions", "max_active_sequences",
                "max_input_rows", "resident_sequence_capacity",
                "kv_logical_page_capacity", "kv_physical_page_capacity"):
        check(limits[key] > 0, failures, f"{key} must stay positive")
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
        per_host_ports.setdefault(host, []).append(endpoint["port"])
        per_host_ports[host].append(
            deployment["transport"]["control_port_base"] + i)
    check(len(endpoints) == 16, failures, "control endpoints must be unique")
    transport = deployment["transport"]
    check(lane.TRANSPORT_BASE <= transport["control_port_base"]
          <= lane.TRANSPORT_BASE + 15, failures,
          "transport control base must sit inside the lane transport block")


def adapter_gates(config, rank, failures, per_host_ports):
    host = f"spark{HEX[rank]}"
    stage, tp = rank // 8, rank % 8
    members = set(config)
    check(members == ADAPTER_MEMBERS, failures,
          f"{host}: adapter member set drift: "
          f"missing {sorted(ADAPTER_MEMBERS - members)} "
          f"extra {sorted(members - ADAPTER_MEMBERS)}")
    check(config["schema_version"] == 3, failures, f"{host}: schema_version")
    check(config["model_revision"] == lane.MODEL_REVISION, failures,
          f"{host}: model_revision must be the placed packs' revision")
    check(config["expert_weight_codec"] == "bf16", failures,
          f"{host}: expert_weight_codec must be the placed bf16 arm")
    expected_pack = (f"packs/laguna_stage.tp8.pp2.stage{stage}.rank{rank}.lgsp")
    check(config["stage_pack_path"] == expected_pack, failures,
          f"{host}: stage_pack_path {config['stage_pack_path']} "
          f"!= {expected_pack} (GLOBAL-rank filename convention)")
    check(config["tp_degree"] == 8, failures, f"{host}: tp_degree")
    check(config["tp_rank"] == tp, failures, f"{host}: tp_rank")

    collective = config["tp_collective"]
    cmembers = set(collective)
    check(cmembers == TP_COLLECTIVE_MEMBERS, failures,
          f"{host}: tp_collective member set drift: "
          f"missing {sorted(TP_COLLECTIVE_MEMBERS - cmembers)} "
          f"extra {sorted(cmembers - TP_COLLECTIVE_MEMBERS)}")
    check(collective["backend"] == "hidden_transport", failures,
          f"{host}: collective backend")
    check(collective["listen_port"] == lane.COLLECTIVE_BASE + tp, failures,
          f"{host}: tp_collective listen_port {collective['listen_port']}")
    check(collective["peer_ports"] == [lane.COLLECTIVE_BASE + t
                                       for t in range(8)], failures,
          f"{host}: peer_ports must be contiguous group-local")
    check(collective["peer_hosts"] == [f"spark{HEX[stage * 8 + t]}"
                                       for t in range(8)], failures,
          f"{host}: peer_hosts must be the rank's PP-stage group")
    check(collective["collective_identifier"] > 0, failures,
          f"{host}: collective_identifier must be positive")
    check(len(collective["rail_peer_hosts"]) == 2
          and all(len(rail) == 8 for rail in collective["rail_peer_hosts"]),
          failures, f"{host}: rail_peer_hosts must be 2 rails x 8 peers")
    check(len(collective["step_rail_indices"]) == 8, failures,
          f"{host}: step_rail_indices must cover the 8 peers")
    per_host_ports.setdefault(host, []).append(collective["listen_port"])

    for member in ("session_ports", "session_ports_hc"):
        table = collective[member]
        check(len(table) == 8, failures, f"{host}: {member} rows")
        sessions = set()
        for a, row in enumerate(table):
            check(len(row) == 8, failures, f"{host}: {member} row {a} length")
            for b, value in enumerate(row):
                if a == b:
                    check(value == 0, failures,
                          f"{host}: {member}[{a}][{b}] diagonal must be 0")
                else:
                    check(lane.SESSION_BASE <= value < lane.SESSION_BASE + 64,
                          failures,
                          f"{host}: {member}[{a}][{b}]={value} outside the "
                          f"session block {lane.SESSION_BASE}.."
                          f"{lane.SESSION_BASE + 63}")
                    check(value not in sessions, failures,
                          f"{host}: {member} value {value} used twice")
                    sessions.add(value)
                    if a == tp and member == "session_ports":
                        per_host_ports[host].append(value)
    check(collective["session_ports_hc"] == collective["session_ports"],
          failures, f"{host}: session_ports_hc must mirror session_ports")


def wrapper_contract_gates(failures):
    wrapper = LaneWrapper(ROOT / "tools/laguna_multidev_run_family.sh",
                          "LAGUNA_")
    blocks = (f"{lane.CONTROL_BASE}:{lane.CONTROL_BASE + 15}",
              f"{lane.COLLECTIVE_BASE}:{lane.COLLECTIVE_BASE + 15}",
              f"{lane.TRANSPORT_BASE}:{lane.TRANSPORT_BASE + 15}",
              f"{lane.SESSION_BASE}:{lane.SESSION_BASE + 63}")
    reserved = ",".join(blocks)
    with tempfile.TemporaryDirectory() as directory:
        base = wrapper.base(Path(directory) / "absent.sock")
        cases = wrapper.queue_cases(base, reserved)
        for index, name in enumerate(("control", "collective", "transport",
                                      "session")):
            cases.append((f"{name} block unreserved",
                          dict(base, SPARK_QUEUE_PORTS=",".join(
                              block for position, block in enumerate(blocks)
                              if position != index)),
                          "not inside a queue-reserved range"))
        cases.append(("zero pool budget",
                      dict(base, SPARK_QUEUE_PORTS=reserved,
                           LAGUNA_EXPERT_POOL_BYTES="0"),
                      "LAGUNA_EXPERT_POOL_BYTES must be positive"))
        with LiveSocket(directory) as live:
            cases.append(("node order mismatch",
                          dict(base, SPARK_QUEUE_PORTS=reserved,
                               LAGUNA_WEIGHTD_SOCKET=str(live)),
                          "node order mismatch: rank "
                          f"{base['SPARK_QUEUE_RANK']} expects spark"))
            wrapper.check_cases(cases, failures)
        wrapper.check_default_socket(base, reserved, "node order mismatch",
                                     failures)


def synthetic_pack(path: Path, group_count: int, experts_per_layer: int):
    """Emit a minimal but structurally faithful .lgsp for the generator.

    Two routed-expert tensors (kinds 14/15) on one layer with the uniform
    bf16 per-expert interleave the real packer emits, plus one spine
    tensor so the sidecar spine arithmetic has a complement to compute.
    """
    HEADER_BYTES, ENTRY_BYTES, ALIGN = 264, 64, 256
    rows = {14: 4, 15: 2}           # w1 fused gate|up rows, w2 rows
    cols = {14: 6, 15: 3}
    entries = []
    # spine tensor first: attention input norm, one f32 scalar payload
    entries.append([3, 0, 2, 0, 0, 1, 1, 1, 0, 4, 0, 0])
    for kind in (14, 15):
        per_expert = rows[kind] * cols[kind] * 2
        entries.append([kind, 5, 1, 1, 0, group_count, rows[kind], cols[kind],
                        0, group_count * per_expert, 0, 0])
    directory = HEADER_BYTES
    cursor = (directory + len(entries) * ENTRY_BYTES + ALIGN - 1) & ~(ALIGN - 1)
    for entry in entries:
        entry[8] = cursor
        cursor = (entry[8] + entry[9] + ALIGN - 1) & ~(ALIGN - 1)
    file_bytes = cursor
    header = struct.pack(
        "<20I2Q65s32s32s32s",
        0x334C4147, 1, HEADER_BYTES, ENTRY_BYTES, 1, 0, len(entries),
        2, 1, 0, 24, 48, 3072, 160000, experts_per_layer, 1, 1, 1, 0, 0,
        directory, file_bytes, b"0f573140834b11cfac0c2af97a101a7a69a13e22",
        b"c" * 32, b"s" * 32, b"r" * 32)
    header += b"\0" * (HEADER_BYTES - len(header))
    with open(path, "wb") as handle:
        handle.write(header)
        handle.seek(directory)
        for entry in entries:
            handle.write(struct.pack("<8I4Q", *entry))
        for entry in entries:
            handle.seek(entry[8])
            payload = bytes((index % 251 for index in range(entry[9])))
            handle.write(payload)
        handle.truncate(file_bytes)
    return entries


def manifest_gates(failures):
    """Drive the .lgsp -> .experts generator on a synthetic pack."""
    if shutil.which("cc") is None:
        failures.append("the .experts manifest generator needs cc on PATH")
        return
    with tempfile.TemporaryDirectory() as temporary:
        pack = Path(temporary) / "laguna_stage.tp8.pp2.stage0.rank0.lgsp"
        group_count = 6
        entries = synthetic_pack(pack, group_count=group_count,
                                 experts_per_layer=256)
        result = subprocess.run(
            ["bash", str(ROOT / "tools/laguna_multidev_experts_manifest.sh"),
             str(pack)], capture_output=True, text=True)
        check(result.returncode == 0, failures,
              f"manifest generator failed: {result.stderr.strip()}")
        sidecar = Path(str(pack) + ".experts")
        check(sidecar.is_file(), failures, "generator wrote no sidecar")
        if not sidecar.is_file():
            return
        blob = sidecar.read_bytes()
        magic, version, count, reserved = struct.unpack_from("<IIII", blob)
        check((magic, version, reserved) == (0x58504557, 2, 0), failures,
              "sidecar header must be WEXP v2")
        check(count == 2 * group_count, failures,
              f"expected {2 * group_count} records, got {count}")
        spans = {}
        expert_total = 0
        for index in range(count):
            layer, expert, kind, pad, offset, span_bytes = \
                struct.unpack_from("<4I2Q", blob, 16 + 48 * index)
            check(pad == 0, failures, "record pad must be 0")
            check(layer == 5, failures,
                  "records must carry the GLOBAL layer index")
            check(kind in (28, 30), failures,
                  f"record kind {kind} outside the laguna convention "
                  "(tensor_kind*2 + payload plane; SparkLagunaManifestCheck "
                  "walks exactly 28/30)")
            entry = entries[1 + (0 if kind == 28 else 1)]
            expected_offset = entry[8] + expert * (entry[9] // group_count)
            check(offset == expected_offset, failures,
                  f"expert {expert} kind {kind}: offset {offset} != "
                  f"{expected_offset} (uniform bf16 interleave)")
            check(span_bytes == entry[9] // group_count, failures,
                  f"expert {expert} kind {kind}: span bytes")
            expert_total += span_bytes
            spans[(offset, span_bytes)] = True
        # spine complement: every non-expert byte of the pack (the f32
        # spine tensor plus inter-tensor padding)
        pack_bytes = pack.stat().st_size
        spine = pack_bytes - expert_total
        budget = subprocess.run(
            [sys.executable, str(ROOT / "tools/laguna_multidev_lane.py"),
             "--budgets", str(pack)], capture_output=True, text=True)
        check(budget.returncode == 0, failures,
              f"--budgets failed: {budget.stderr.strip()[:200]}")
        if budget.returncode == 0:
            pool_text, spine_text = budget.stdout.split()
            chunk = 2 * 1024 * 1024
            pack_chunk_basis = -(-pack_bytes // chunk) * chunk
            check(int(pool_text) == pack_chunk_basis, failures,
                  f"pool {pool_text} != whole-pack chunk basis "
                  f"{pack_chunk_basis} (the daemon's acquire budget law)")
            check(int(spine_text) == spine + 256, failures,
                  f"spine budget {spine_text} != complement {spine} + 256 "
                  "(the aligned spine allocation)")
        # idempotence: a valid sidecar is left untouched
        before = sidecar.read_bytes()
        again = subprocess.run(
            ["bash", str(ROOT / "tools/laguna_multidev_experts_manifest.sh"),
             str(pack)], capture_output=True, text=True)
        check(again.returncode == 0 and sidecar.read_bytes() == before,
              failures, "generator must be idempotent on a valid sidecar")


def main() -> int:
    failures = []
    attempt = "b" * 32
    with tempfile.TemporaryDirectory() as temporary:
        runtime_root = f"/tmp/sparkqueue-{attempt}"
        socket_path = FLEET_WEIGHTD_SOCKET
        output = Path(temporary) / "generated"
        subprocess.run(
            [sys.executable, str(ROOT / "tools/laguna_multidev_lane.py"),
             "--runtime-root", runtime_root,
             "--weightd-socket", socket_path,
             "--output-dir", str(output),
             "--collective-identifier", "12345678901"],
            check=True, capture_output=True)

        deployment = json.loads((output / "deployment.json").read_text())
        per_host_ports = {}
        deployment_gates(deployment, runtime_root, socket_path, failures,
                         per_host_ports)

        for rank in range(16):
            config = json.loads(
                (output / f"adapter.spark{HEX[rank]}.json").read_text())
            adapter_gates(config, rank, failures, per_host_ports)

        check(len(per_host_ports) == 16, failures,
              f"every host listens: {sorted(per_host_ports)}")
        for host, ports in per_host_ports.items():
            duplicates = sorted({port for port in ports
                                 if ports.count(port) > 1})
            check(not duplicates, failures,
                  f"{host}: listens {duplicates} twice")
            outside = set(ports) - ALL_LANE_PORTS
            check(not outside, failures,
                  f"{host}: ports outside lane 8 blocks: {sorted(outside)}")

        check(lane.MESH_RANKS == ",".join(str(i) for i in range(16)),
              failures, "mesh map must be the identity permutation")

        check_run = subprocess.run(
            [sys.executable, str(ROOT / "tools/laguna_multidev_lane.py"),
             "--runtime-root", runtime_root,
             "--weightd-socket", socket_path,
             "--output-dir", str(output),
             "--collective-identifier", "12345678901", "--check"],
            capture_output=True, text=True)
        check(check_run.returncode == 0, failures,
              f"--check failed: {check_run.stderr.strip()}")

        for refused in ("/tmp/other.sock", STALE_SHARED_UNIT_SOCKET):
            refusal = subprocess.run(
                [sys.executable, str(ROOT / "tools/laguna_multidev_lane.py"),
                 "--runtime-root", runtime_root,
                 "--weightd-socket", refused,
                 "--output-dir", str(Path(temporary) / "refused"),
                 "--collective-identifier", "12345678901"],
                capture_output=True, text=True)
            reason = f"weightd socket {refused} is not the fleet weightd"
            check(refusal.returncode != 0 and reason in refusal.stderr,
                  failures, f"generator must refuse {refused}: "
                  f"rc={refusal.returncode} {refusal.stderr.strip()}")
            check(not (Path(temporary) / "refused").exists(), failures,
                  f"refused socket {refused} still wrote output")

        drift = subprocess.run(
            [sys.executable, str(ROOT / "tools/laguna_multidev_lane.py"),
             "--runtime-root", runtime_root,
             "--weightd-socket", socket_path,
             "--output-dir", str(output),
             "--collective-identifier", "999", "--check"],
            capture_output=True, text=True)
        check(drift.returncode != 0, failures,
              "--check must flag a drifted collective identifier")

    for script in ("tools/laguna_multidev_run_family.sh",
                   "tools/laguna_multidev_experts_manifest.sh"):
        syntax = subprocess.run(
            ["bash", "-n", str(ROOT / script)],
            capture_output=True, text=True)
        check(syntax.returncode == 0, failures,
              f"{script}: bash -n failed: {syntax.stderr.strip()}")

    wrapper_contract_gates(failures)
    manifest_gates(failures)

    if failures:
        for failure in failures:
            print(f"FAIL {failure}")
        return 1
    print("laguna multidev lane gates PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
