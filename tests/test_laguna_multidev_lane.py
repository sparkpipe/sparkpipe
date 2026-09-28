#!/usr/bin/env python3
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

LANE = 11
PORTS = lane.lane_ports(LANE)
CONTROL_BASE = PORTS["control"]
COLLECTIVE_BASE = PORTS["collective"]
TRANSPORT_BASE = PORTS["transport"]
SESSION_BASE = PORTS["session"]
ALL_LANE_PORTS = (set(range(CONTROL_BASE, CONTROL_BASE + 16))
                  | set(range(COLLECTIVE_BASE, COLLECTIVE_BASE + 16))
                  | set(range(TRANSPORT_BASE, TRANSPORT_BASE + 16))
                  | set(range(SESSION_BASE, SESSION_BASE + 64)))
HEX = lane.HEX
GENERATOR = str(ROOT / "tools/laguna_multidev_lane.py")

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


def deployment_gates(deployment, runtime_root, socket, failures):
    check(deployment["schema_version"] == 2, failures,
          "deployment schema_version must be 2")
    check(deployment["weightd"]["socket_path"] == socket, failures,
          "weightd socket path must be the requested socket")
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
              f"node {i}: runtime_root must be the requested root")
        backing = node["kv_backing_directory"]
        check(backing.startswith(runtime_root + "/"), failures,
              f"node {i}: kv backing {backing} escapes the private root")
        check(0 < node["kv_backing_maximum_bytes"] < 1 << 40, failures,
              f"node {i}: kv backing cap must be finite")
        endpoint = node["control_endpoint"]
        host = f"spark{HEX[i]}"
        check(endpoint["host"] == host, failures,
              f"node {i}: control host {endpoint['host']} != {host}")
        check(endpoint["port"] == CONTROL_BASE + i, failures,
              f"node {i}: control port {endpoint['port']}")
        endpoints.add((endpoint["host"], endpoint["port"]))
    check(len(endpoints) == 16, failures, "control endpoints must be unique")
    transport = deployment["transport"]
    check(TRANSPORT_BASE <= transport["control_port_base"]
          <= TRANSPORT_BASE + 15, failures,
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
    check(collective["listen_port"] == COLLECTIVE_BASE + tp, failures,
          f"{host}: tp_collective listen_port {collective['listen_port']}")
    check(collective["peer_ports"] == [COLLECTIVE_BASE + t
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
    per_host_ports.setdefault(host, set()).add(collective["listen_port"])
    per_host_ports[host].add(CONTROL_BASE + rank)

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
                    check(SESSION_BASE <= value < SESSION_BASE + 64,
                          failures,
                          f"{host}: {member}[{a}][{b}]={value} outside the "
                          f"session block {SESSION_BASE}.."
                          f"{SESSION_BASE + 63}")
                    check(value not in sessions, failures,
                          f"{host}: {member} value {value} used twice")
                    sessions.add(value)
                    if a == tp:
                        per_host_ports[host].add(value)


def wrapper_contract_gates(failures):
    script = ROOT / "tools/laguna_multidev_run_family.sh"
    attempt = "a" * 32
    common = {
        "LAGUNA_LANE": str(LANE),
        "LAGUNA_KV_BACKING_BYTES": str(2 << 30),
        "LAGUNA_KV_PAGE_CAPACITY": "2048",
        "LAGUNA_WEIGHTD_SOCKET": "/nonexistent/weightd.sock",
        "HOME": "/nonexistent",
    }
    base = dict(common, LAGUNA_MODE="queue", SPARK_QUEUE_ATTEMPT=attempt,
                SPARK_QUEUE_RUNTIME_ROOT=f"/tmp/sparkqueue-{attempt}",
                SPARK_QUEUE_RANK="0", SPARK_QUEUE_SIZE="16")
    direct = dict(common, LAGUNA_MODE="direct",
                  LAGUNA_RUNTIME_ROOT=f"/tmp/sp-laguna-lane{LANE}",
                  LAGUNA_RANK="0", LAGUNA_COLLECTIVE_ID="4242")
    lane_ranges = (f"{CONTROL_BASE}:{CONTROL_BASE + 15},"
                   f"{COLLECTIVE_BASE}:{COLLECTIVE_BASE + 15},"
                   f"{TRANSPORT_BASE}:{TRANSPORT_BASE + 15},"
                   f"{SESSION_BASE}:{SESSION_BASE + 63}")

    def run(env):
        complete = {key: value for key, value in os.environ.items()
                    if not (key.startswith("SPARK_QUEUE_")
                            or key.startswith("LAGUNA_")
                            or key.startswith("SPARK_WEIGHTD_"))}
        complete.update(env)
        return subprocess.run(["bash", str(script)], env=complete,
                              capture_output=True, text=True)

    def without(env, key):
        return {k: v for k, v in env.items() if k != key}

    reserved = dict(base, SPARK_QUEUE_PORTS=lane_ranges)
    cases = [
        ("missing lane", without(reserved, "LAGUNA_LANE"), "LAGUNA_LANE is required"),
        ("lane out of range", dict(reserved, LAGUNA_LANE="16"), "LAGUNA_LANE must be 0..15"),
        ("missing mode", without(reserved, "LAGUNA_MODE"), "LAGUNA_MODE must be queue or direct"),
        ("unknown mode", dict(reserved, LAGUNA_MODE="shared"), "LAGUNA_MODE must be queue or direct"),
        ("missing attempt id", without(base, "SPARK_QUEUE_ATTEMPT"), "authoritative spark queue"),
        ("malformed attempt id", dict(base, SPARK_QUEUE_ATTEMPT="z" * 32), "bad attempt id"),
        ("short attempt id", dict(base, SPARK_QUEUE_ATTEMPT="abc"), "bad attempt id"),
        ("job namespace mismatch", dict(base, SPARK_QUEUE_RUNTIME_ROOT="/tmp/elsewhere"),
         "unexpected job namespace"),
        ("queue size mismatch", dict(base, SPARK_QUEUE_SIZE="8"), "SPARK_QUEUE_SIZE must be 16"),
        ("rank out of range", dict(base, SPARK_QUEUE_RANK="16"), "rank must be 0..15"),
        ("no reserved ports", dict(base), "queue reserved no ports"),
        ("ports outside the lane blocks", dict(base, SPARK_QUEUE_PORTS="7000:7001"),
         "not inside a queue-reserved range"),
        ("lane 8 ports reserved for lane 11",
         dict(base, SPARK_QUEUE_PORTS="23128:23143,53128:53143,64128:64143,23680:23743"),
         "not inside a queue-reserved range"),
        ("session block unreserved",
         dict(base, SPARK_QUEUE_PORTS=(
             f"{CONTROL_BASE}:{CONTROL_BASE + 15},"
             f"{COLLECTIVE_BASE}:{COLLECTIVE_BASE + 15},"
             f"{TRANSPORT_BASE}:{TRANSPORT_BASE + 15}")),
         "not inside a queue-reserved range"),
        ("missing kv backing", without(reserved, "LAGUNA_KV_BACKING_BYTES"),
         "LAGUNA_KV_BACKING_BYTES must be a positive decimal"),
        ("missing kv pages", without(reserved, "LAGUNA_KV_PAGE_CAPACITY"),
         "LAGUNA_KV_PAGE_CAPACITY must be a positive decimal"),
        ("zero kv pages", dict(reserved, LAGUNA_KV_PAGE_CAPACITY="0"),
         "LAGUNA_KV_PAGE_CAPACITY must be a positive decimal"),
        ("bad pool override", dict(reserved, LAGUNA_EXPERT_POOL_BYTES="12k"),
         "LAGUNA_EXPERT_POOL_BYTES must be a positive decimal"),
        ("missing socket", without(reserved, "LAGUNA_WEIGHTD_SOCKET"),
         "LAGUNA_WEIGHTD_SOCKET must name the running weightd socket"),
        ("inherited socket is not used",
         dict(without(reserved, "LAGUNA_WEIGHTD_SOCKET"), SPARK_WEIGHTD_SOCKET="/tmp/spark_weightd.sock"),
         "LAGUNA_WEIGHTD_SOCKET must name the running weightd socket"),
        ("relative socket", dict(reserved, LAGUNA_WEIGHTD_SOCKET="weightd.sock"),
         "LAGUNA_WEIGHTD_SOCKET must name the running weightd socket"),
        ("direct root elsewhere", dict(direct, LAGUNA_RUNTIME_ROOT="/tmp/elsewhere"),
         f"LAGUNA_RUNTIME_ROOT must be /tmp/sp-laguna-lane{LANE}"),
        ("direct missing rank", without(direct, "LAGUNA_RANK"), "LAGUNA_RANK is required"),
        ("direct bad rank", dict(direct, LAGUNA_RANK="x"), "rank must be 0..15"),
        ("direct missing collective id", without(direct, "LAGUNA_COLLECTIVE_ID"),
         "LAGUNA_COLLECTIVE_ID must be a positive decimal"),
        ("direct zero collective id", dict(direct, LAGUNA_COLLECTIVE_ID="0"),
         "LAGUNA_COLLECTIVE_ID must be a positive decimal"),
    ]
    for name, env, expected in cases:
        result = run(env)
        check(result.returncode != 0, failures,
              f"wrapper case '{name}' must fail closed")
        check(expected in result.stderr, failures,
              f"wrapper case '{name}': expected '{expected}' in stderr, "
              f"got: {result.stderr.strip()[:200]}")

    for name, env in (("queue", reserved), ("direct", direct)):
        accepted = run(env)
        check(accepted.returncode != 0, failures,
              f"{name} wrapper must still fail without a live socket")
        check("not a live socket" in accepted.stderr, failures,
              f"{name} wrapper should reach the socket check, got: "
              f"{accepted.stderr.strip()[:200]}")


def warm_receipt_gates(failures):
    script = ROOT / "tools/devcycle/laguna_warm_receipt.sh"
    text = script.read_text()
    check("/run/sparkpipe-weightd-shared" not in text, failures,
          "warm receipt must not name the retired shared weightd socket")

    def run(env):
        complete = {key: value for key, value in os.environ.items()
                    if not (key.startswith("LAGUNA_")
                            or key.startswith("SPARK_WEIGHTD_"))}
        complete.update(env)
        return subprocess.run(["bash", str(script)], env=complete,
                              capture_output=True, text=True)

    with tempfile.TemporaryDirectory() as temporary:
        regular = Path(temporary) / "weightd.sock"
        regular.write_text("")
        cases = [
            ("missing socket", {"LAGUNA_LANE": str(LANE)},
             "LAGUNA_WEIGHTD_SOCKET must name the running weightd socket"),
            ("inherited socket is not used",
             {"LAGUNA_LANE": str(LANE),
              "SPARK_WEIGHTD_SOCKET": "/tmp/spark_weightd.sock"},
             "LAGUNA_WEIGHTD_SOCKET must name the running weightd socket"),
            ("relative socket",
             {"LAGUNA_LANE": str(LANE), "LAGUNA_WEIGHTD_SOCKET": "weightd.sock"},
             "LAGUNA_WEIGHTD_SOCKET must name the running weightd socket"),
            ("socket path is not a socket",
             {"LAGUNA_LANE": str(LANE), "LAGUNA_WEIGHTD_SOCKET": str(regular)},
             "is not a live socket"),
        ]
        for name, env, expected in cases:
            result = run(env)
            check(result.returncode != 0, failures,
                  f"warm receipt case '{name}' must fail closed")
            check(expected in result.stderr, failures,
                  f"warm receipt case '{name}': expected '{expected}' in "
                  f"stderr, got: {result.stderr.strip()[:200]}")


def synthetic_pack(path: Path, group_count: int, experts_per_layer: int):
    HEADER_BYTES, ENTRY_BYTES, ALIGN = 264, 64, 256
    rows = {14: 4, 15: 2}           # w1 fused gate|up rows, w2 rows
    cols = {14: 6, 15: 3}
    entries = []
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
    if shutil.which("cc") is None:
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
              f"manifest generator failed: {result.stderr.strip()[:300]}")
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
        before = sidecar.read_bytes()
        again = subprocess.run(
            ["bash", str(ROOT / "tools/laguna_multidev_experts_manifest.sh"),
             str(pack)], capture_output=True, text=True)
        check(again.returncode == 0 and sidecar.read_bytes() == before,
              failures, "generator must be idempotent on a valid sidecar")


def generate(output, runtime_root, socket_path, identifier, *extra):
    return subprocess.run(
        [sys.executable, GENERATOR, "--lane", str(LANE),
         "--runtime-root", runtime_root, "--weightd-socket", socket_path,
         "--output-dir", str(output), "--collective-identifier", identifier,
         "--kv-backing-bytes", str(2 << 30), "--kv-page-capacity", "2048",
         *extra], capture_output=True, text=True)


def generator_refusal_gates(output, failures):
    good = ["--lane", str(LANE), "--runtime-root", f"/tmp/sp-laguna-lane{LANE}",
            "--weightd-socket", "/tmp/spark_weightd.sock", "--output-dir", str(output),
            "--collective-identifier", "7", "--kv-backing-bytes", "1024",
            "--kv-page-capacity", "2048"]

    def replaced(flag, value):
        arguments = list(good)
        index = arguments.index(flag)
        if value is None:
            del arguments[index:index + 2]
        else:
            arguments[index + 1] = value
        return arguments

    cases = [
        ("missing lane", replaced("--lane", None), "--lane"),
        ("lane 16", replaced("--lane", "16"), "lane must be 0..15"),
        ("missing kv pages", replaced("--kv-page-capacity", None), "--kv-page-capacity"),
        ("zero kv pages", replaced("--kv-page-capacity", "0"), "kv page capacity must be positive"),
        ("missing kv backing", replaced("--kv-backing-bytes", None), "--kv-backing-bytes"),
        ("relative root", replaced("--runtime-root", "tmp/root"), "runtime root must be an absolute"),
        ("unnormalized root", replaced("--runtime-root", "/tmp/a/../b"), "runtime root must be an absolute"),
        ("sparkdata root", replaced("--runtime-root", "/home/spark0/sparkdata/root"), "sparkdata"),
        ("relative socket", replaced("--weightd-socket", "weightd.sock"), "weightd socket must be an absolute"),
        ("zero collective id", replaced("--collective-identifier", "0"), "collective identifier must be positive"),
    ]
    for name, arguments, expected in cases:
        result = subprocess.run([sys.executable, GENERATOR, *arguments],
                                capture_output=True, text=True)
        check(result.returncode != 0 and expected in result.stderr, failures,
              f"generator case '{name}' must fail naming '{expected}': "
              f"rc={result.returncode} {result.stderr.strip()[:200]}")
    accepted = subprocess.run([sys.executable, GENERATOR, *good],
                              capture_output=True, text=True)
    check(accepted.returncode == 0, failures,
          f"generator must accept a direct-mode root and the production socket: "
          f"{accepted.stderr.strip()[:200]}")


def registry_gates(failures):
    registry = json.loads((ROOT / "tools/devcycle/lane_assignments.json").read_text())
    ours = lane.lane_ports(LANE)
    for entry in registry["lanes"]:
        if entry["driver"] == "laguna":
            ports = lane.lane_ports(entry["lane"])
            check(entry["ports"] == {key: ports[key] for key in entry["ports"]}, failures,
                  f"laguna registry lane {entry['lane']}: generator ports {ports} "
                  f"disagree with {entry['ports']}")
            continue
        for key, first in entry["ports"].items():
            check(abs(first - ours[key]) >= 16, failures,
                  f"lane {LANE} {key} block {ours[key]} overlaps registry lane "
                  f"{entry['lane']} ({entry['driver']}) at {first}")
    check(lane.lane_ports(LANE) == {"control": 23176, "collective": 53176,
                                    "transport": 64176, "session": 23872},
          failures, f"lane {LANE} ports: {lane.lane_ports(LANE)}")


def main() -> int:
    failures = []
    with tempfile.TemporaryDirectory() as temporary:
        runtime_root = f"/tmp/sp-laguna-lane{LANE}"
        socket_path = "/tmp/spark_weightd.sock"
        output = Path(temporary) / "generated"
        first = generate(output, runtime_root, socket_path, "12345678901")
        check(first.returncode == 0, failures,
              f"generator failed: {first.stderr.strip()[:300]}")

        deployment = json.loads((output / "deployment.json").read_text())
        deployment_gates(deployment, runtime_root, socket_path, failures)
        check(deployment["runtime_limits"].get("max_sequence_positions")
              == lane.MAX_SEQUENCE_POSITIONS, failures,
              "deployment must bound the API context to the adapter's positions")
        adapter_zero = json.loads((output / "adapter.spark0.json").read_text())
        check(adapter_zero["max_sequence_positions"]
              == deployment["runtime_limits"].get("max_sequence_positions"),
              failures, "adapter and deployment sequence positions must agree")
        for key in ("kv_logical_page_capacity", "kv_physical_page_capacity"):
            check(deployment["runtime_limits"][key] == 2048, failures,
                  f"{key} must be the requested 2048 pages")
        check(all(node["kv_backing_maximum_bytes"] == 2 << 30
                  for node in deployment["nodes"]), failures,
              "kv backing cap must be the requested bytes")

        per_host_ports = {}
        for rank in range(16):
            config = json.loads(
                (output / f"adapter.spark{HEX[rank]}.json").read_text())
            adapter_gates(config, rank, failures, per_host_ports)

        for host, ports in per_host_ports.items():
            outside = ports - ALL_LANE_PORTS
            check(not outside, failures,
                  f"{host}: ports outside lane {LANE} blocks: {sorted(outside)}")

        check(lane.MESH_RANKS == ",".join(str(i) for i in range(16)),
              failures, "mesh map must be the identity permutation")

        check_run = generate(output, runtime_root, socket_path, "12345678901", "--check")
        check(check_run.returncode == 0, failures,
              f"--check failed: {check_run.stderr.strip()}")
        drift = generate(output, runtime_root, socket_path, "999", "--check")
        check(drift.returncode != 0, failures,
              "--check must flag a drifted collective identifier")
        generator_refusal_gates(Path(temporary) / "refusals", failures)
    registry_gates(failures)

    for script in ("tools/laguna_multidev_run_family.sh",
                   "tools/laguna_multidev_experts_manifest.sh",
                   "tools/devcycle/laguna_warm_receipt.sh"):
        syntax = subprocess.run(
            ["bash", "-n", str(ROOT / script)],
            capture_output=True, text=True)
        check(syntax.returncode == 0, failures,
              f"{script}: bash -n failed: {syntax.stderr.strip()}")

    wrapper_contract_gates(failures)
    warm_receipt_gates(failures)
    manifest_gates(failures)

    if failures:
        for failure in failures:
            print(f"FAIL {failure}")
        return 1
    print("laguna multidev lane gates PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
