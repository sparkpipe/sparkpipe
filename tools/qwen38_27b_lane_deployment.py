#!/usr/bin/env python3
"""Prepare the private Qwen 3.8 27B lane deployment under the queue runtime root.

Shared-socket family wrapper for developer lane 1 (qwen38_27b, TP4 on
spark0-spark3), following the multideveloper quickstart contract: a private
deployment under $SPARK_QUEUE_RUNTIME_ROOT whose runtime packs/ directory
holds exactly one valid *.sha256 digest for shared weightd attachment, a
common sparkpipe_model_residentd launch driven by the queue rank, and the
shared-daemon environment (SPARK_WEIGHTD_SOCKET, SPARK_WEIGHTD_LANE,
SPARK_TP_MESH_RANKS) supplied by tools/qwen38_27b_lane_launch.sh.

Default pack source is the node-local Qwen 3.8 27B nvfp4a16 TP4 rank pack
set (nvfp4 FFN + bf16 spine, the quality-law serving arm):

  /home/<host>/sparkdata/qwen38-27b.nvfp4a16.tp4/packs/tp4-rankNN.q38sp

Every listener stays inside the lane-1 port blocks (control 23016-23031,
collective 53016-53031, transport 64016-64031). The collective session grid
uses the twelve ports above the four listen ports because the serving
adapter loads session_ports as a mandatory non-zero off-diagonal matrix
while session_ports_hc stays zero (require_session_ports == 0).

Fail-closed: missing queue identity, unverified firmware, missing pack, a
second *.sha256 in the staged packs/ directory, or any listener outside the
reserved ranges aborts before residentd starts.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path

LANE_FAMILY = "qwen38_27b"
LANE_TOPOLOGY = "TP4"
LANE_HOSTS_DEFAULT = "spark0,spark1,spark2,spark3"
LANE_CONTROL_BASE_DEFAULT = 23016
LANE_COLLECTIVE_BASE_DEFAULT = 53016
LANE_TRANSPORT_BASE_DEFAULT = 64016
LANE_WEIGHTD_LANE_DEFAULT = 1
LANE_SHARED_SOCKET_DEFAULT = "/run/sparkpipe-weightd-shared/weightd.sock"
LANE_MESH_RANKS_DEFAULT = "0,1,2,3"
LANE_PACK_DIR_DEFAULT = "/home/{host}/sparkdata/qwen38-27b.nvfp4a16.tp4/packs"
LANE_PACK_TEMPLATE_DEFAULT = "tp4-rank{rank:02d}.q38sp"
LANE_MODEL_REVISION_DEFAULT = "bf16-h5120-l64-gdn48-full16-v248320-mtp1-v1"
LANE_MAX_SEQUENCE_POSITIONS_DEFAULT = 4096
LANE_KV_BACKING_MAXIMUM_BYTES_DEFAULT = 2 * 1024 * 1024 * 1024
LANE_NODE_TARGET = "cuda.sm121.qwen38_27b.resident_decode_stage.bf16"
LANE_ADAPTER_PATH = "lib/model_serving_adapter.so"
LANE_TRANSPORT_PATH = "lib/hidden_transport.so"
LANE_DRIVER_PATH = "stages/stage_000/model_driver.so"
LANE_RESIDENTD_PATH = "bin/sparkpipe_model_residentd"

# The serving adapter validates the hidden_transport member set exactly
# (runtime/serving_adapter_template.c SparkTpCollectiveValidateMembers).
TP_COLLECTIVE_MEMBERS = (
    "backend",
    "backend_module_path",
    "collective_identifier",
    "listen_port",
    "connect_timeout_milli",
    "operation_timeout_milli",
    "peer_hosts",
    "peer_ports",
    "algorithms",
    "direct_all_to_all_max_payload_bytes",
    "split_ring_min_payload_bytes",
    "rail_peer_hosts",
    "step_rail_indices",
    "session_ports",
    "session_ports_hc",
)
STAGE_MEMBERS = (
    "schema_version",
    "model_revision",
    "stage_pack_path",
    "max_sequence_positions",
    "tp_degree",
    "tp_rank",
    "tp_collective",
)


class LaneError(Exception):
    """A fail-closed deployment preparation error."""


def env_int(env: dict[str, str], name: str, default: int) -> int:
    raw = env.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw, 10)
    except ValueError:
        raise LaneError(f"{name} must be a decimal integer, got {raw!r}")


def env_str(env: dict[str, str], name: str, default: str) -> str:
    return env.get(name) or default


def digest_file(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def parse_hosts(raw: str) -> list[str]:
    hosts = [host.strip() for host in raw.split(",") if host.strip()]
    if not hosts or len(set(hosts)) != len(hosts):
        raise LaneError(f"invalid lane host list {raw!r}")
    for host in hosts:
        if not re.fullmatch(r"spark[0-9a-f]", host):
            raise LaneError(f"invalid lane host name {host!r}")
    return hosts


def parse_port_ranges(raw: str) -> list[tuple[int, int]]:
    ranges = []
    for piece in raw.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if not re.fullmatch(r"\d+:\d+", piece):
            raise LaneError(f"invalid reserved port range {piece!r}")
        first, last = (int(value) for value in piece.split(":"))
        if first > last:
            raise LaneError(f"inverted reserved port range {piece!r}")
        ranges.append((first, last))
    return ranges


def parse_mesh_ranks(raw: str, size: int) -> list[int]:
    values = [piece.strip() for piece in raw.split(",") if piece.strip() != ""]
    if len(values) != size:
        raise LaneError(f"SPARK mesh ranks {raw!r} must list {size} physical ranks")
    physical = []
    for value in values:
        if not re.fullmatch(r"\d+", value) or int(value) >= 16:
            raise LaneError(f"invalid physical mesh rank {value!r}")
        physical.append(int(value))
    if len(set(physical)) != len(physical):
        raise LaneError(f"duplicate physical mesh rank in {raw!r}")
    return physical


def collective_identifier(attempt: str) -> int:
    return int(attempt[:15], 16) + 1


def session_grid(base: int, size: int) -> list[list[int]]:
    """Compact off-diagonal session grid inside the collective block.

    The size*(size-1) ports above the listen ports are assigned in
    row-major order to the off-diagonal cells; the diagonal stays zero
    exactly as the adapter's session-port loader requires. For the lane-1
    TP4 block (53016-53031) that fills 53020-53031 completely.
    """
    cursor = base + size
    grid = []
    for row in range(size):
        cells = []
        for column in range(size):
            if row == column:
                cells.append(0)
            else:
                cells.append(cursor)
                cursor += 1
        grid.append(cells)
    if cursor != base + size + size * (size - 1):
        raise LaneError("collective session grid exhausted the lane port block")
    return grid


def stage_config(hosts: list[str], rank: int, pack_name: str, attempt: str,
                 model_revision: str, max_sequence_positions: int,
                 control_bases: dict[str, int]) -> dict:
    size = len(hosts)
    collective_base = control_bases["collective"]
    config = {
        "schema_version": 3,
        "model_revision": model_revision,
        "stage_pack_path": f"packs/{pack_name}",
        "max_sequence_positions": max_sequence_positions,
        "tp_degree": size,
        "tp_rank": rank,
        "tp_collective": {
            "backend": "hidden_transport",
            "backend_module_path": LANE_TRANSPORT_PATH,
            "collective_identifier": collective_identifier(attempt),
            "listen_port": collective_base + rank,
            "connect_timeout_milli": 30000,
            "operation_timeout_milli": 30000,
            "peer_hosts": list(hosts),
            "peer_ports": [collective_base + index for index in range(size)],
            "algorithms": ["tree"],
            "direct_all_to_all_max_payload_bytes": 0,
            "split_ring_min_payload_bytes": 0,
            "rail_peer_hosts": [list(hosts), list(hosts)],
            "step_rail_indices": [0] + [1] * (size - 1),
            "session_ports": session_grid(collective_base, size),
            "session_ports_hc": [[0] * size for _ in range(size)],
        },
    }
    if tuple(config["tp_collective"]) != TP_COLLECTIVE_MEMBERS:
        raise LaneError("tp_collective member set drifted from the adapter contract")
    if tuple(config) != STAGE_MEMBERS:
        raise LaneError("stage member set drifted from the adapter contract")
    return config


def resident_deployment(hosts: list[str], runtime_root: str, kv_root: str,
                        kv_maximum_bytes: int, control_bases: dict[str, int],
                        shared_socket: str, max_sequence_positions: int) -> dict:
    size = len(hosts)
    page_capacity = size * ((max_sequence_positions + 63) // 64)
    nodes = []
    for rank, host in enumerate(hosts):
        nodes.append({
            "rank_index": rank,
            "stage_index": rank,
            "runtime_root": runtime_root,
            "node_target": LANE_NODE_TARGET,
            "transport_host": host,
            "adapter_configuration_path": "config/stage.json",
            "kv_backing_directory": kv_root,
            "kv_backing_maximum_bytes": kv_maximum_bytes,
            "control_endpoint": {
                "kind": "tcp",
                "host": host,
                "port": control_bases["control"] + rank,
            },
        })
    return {
        "schema_version": 2,
        "coordinator_rank_index": 0,
        "adapter": {"shared_object_path": LANE_ADAPTER_PATH},
        "driver": {
            "shared_object_path": LANE_DRIVER_PATH,
            "program_name": "resident_decode",
        },
        "transport": {
            "shared_object_path": LANE_TRANSPORT_PATH,
            "mode": "host-rdma",
            "control_port_base": control_bases["transport"],
        },
        "weightd": {"socket_path": shared_socket},
        "runtime_limits": {
            "max_inflight_submissions": 4,
            "max_active_sequences": 16,
            "max_input_rows": 16,
            "resident_sequence_capacity": 16,
            "kv_logical_page_capacity": page_capacity,
            "kv_physical_page_capacity": page_capacity,
        },
        "nodes": nodes,
    }


def verify_firmware(firmware_root: Path) -> dict[str, str]:
    if not firmware_root.is_dir():
        raise LaneError(f"firmware root is not a directory: {firmware_root}")
    commit_file = firmware_root / "SOURCE_COMMIT"
    if not commit_file.is_file():
        raise LaneError("firmware root has no SOURCE_COMMIT (not a verified build)")
    commit = commit_file.read_text().strip()
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise LaneError(f"firmware SOURCE_COMMIT is not a commit sha: {commit!r}")
    sums = firmware_root / "SHA256SUMS"
    if not sums.is_file():
        raise LaneError("firmware root has no SHA256SUMS (not a verified build)")
    listed: dict[str, str] = {}
    for line in sums.read_text().splitlines():
        if not line.strip():
            continue
        pieces = line.split(None, 1)
        if len(pieces) != 2 or not re.fullmatch(r"[0-9a-f]{64}", pieces[0]):
            raise LaneError(f"malformed SHA256SUMS line: {line[:80]!r}")
        relative = pieces[1].strip().lstrip("*")
        if relative.startswith("/") or ".." in Path(relative).parts:
            raise LaneError(f"SHA256SUMS entry escapes the firmware root: {relative}")
        if relative in listed:
            raise LaneError(f"SHA256SUMS lists {relative} twice")
        listed[relative] = pieces[0]
    required = (
        LANE_RESIDENTD_PATH,
        LANE_ADAPTER_PATH,
        LANE_TRANSPORT_PATH,
        LANE_DRIVER_PATH,
    )
    for relative in required:
        if relative not in listed:
            raise LaneError(f"firmware SHA256SUMS does not pin {relative}")
        artifact = firmware_root / relative
        if not artifact.is_file():
            raise LaneError(f"firmware build is missing {relative}")
        if digest_file(artifact) != listed[relative]:
            raise LaneError(f"firmware artifact differs from its digest: {relative}")
    return {"source_commit": commit}


def stage_pack(pack_source: Path, packs_dir: Path, trust_source_digest: bool) -> tuple[str, str]:
    if not pack_source.is_file():
        raise LaneError(f"lane pack is missing: {pack_source}")
    experts_source = pack_source.with_name(pack_source.name + ".experts")
    if not experts_source.is_file():
        raise LaneError(
            "pack has no .experts manifest (weightd lazy attach refuses without it): "
            f"{experts_source} - generate it with build/qwen38_27b_experts_manifest")
    packs_dir.mkdir(parents=True, exist_ok=True)
    staged = packs_dir / pack_source.name
    if staged.exists() or staged.is_symlink():
        raise LaneError(f"pack already staged: {staged}")
    staged.symlink_to(pack_source.resolve())
    sidecar = pack_source.with_name(pack_source.name + ".sha256")
    if trust_source_digest and sidecar.is_file():
        digest = sidecar.read_text().split()[0]
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise LaneError(f"source digest sidecar is malformed: {sidecar}")
    else:
        digest = digest_file(pack_source)
    (packs_dir / (pack_source.name + ".sha256")).write_text(digest + "\n")
    (packs_dir / experts_source.name).symlink_to(experts_source.resolve())
    digests = sorted(entry.name for entry in packs_dir.iterdir()
                     if entry.name.endswith(".sha256"))
    if digests != [pack_source.name + ".sha256"]:
        raise LaneError(f"staged packs/ must hold exactly one *.sha256, found {digests}")
    return pack_source.name, digest


def stage_runtime(root: Path, firmware_root: Path, hosts: list[str], rank: int,
                  attempt: str, pack_dir: Path, pack_template: str,
                  model_revision: str, max_sequence_positions: int,
                  control_bases: dict[str, int], shared_socket: str,
                  kv_maximum_bytes: int, trust_source_digest: bool) -> dict:
    runtime = root / "runtime"
    runtime.mkdir(parents=True)
    for relative in (LANE_ADAPTER_PATH, LANE_TRANSPORT_PATH, LANE_DRIVER_PATH):
        target = runtime / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to((firmware_root / relative).resolve())
    config_dir = runtime / "config"
    config_dir.mkdir()
    local_host = hosts[rank]
    resolved_pack_dir = Path(str(pack_dir).format(host=local_host))
    pack_name = pack_template.format(rank=rank)
    pack_digest_pair = stage_pack(resolved_pack_dir / pack_name, runtime / "packs",
                                  trust_source_digest)
    config = stage_config(hosts, rank, pack_name, attempt, model_revision,
                          max_sequence_positions, control_bases)
    (config_dir / "stage.json").write_text(json.dumps(config) + "\n")
    kv_root = root / "kv"
    kv_root.mkdir()
    deployment = resident_deployment(hosts, str(runtime), str(kv_root),
                                     kv_maximum_bytes, control_bases,
                                     shared_socket, max_sequence_positions)
    (root / "deployment.json").write_text(json.dumps(deployment) + "\n")
    return {
        "runtime_root": str(runtime),
        "deployment_path": str(root / "deployment.json"),
        "pack_name": pack_digest_pair[0],
        "pack_sha256": pack_digest_pair[1],
        "stage_config_path": str(config_dir / "stage.json"),
        "kv_root": str(kv_root),
    }


def listener_ports(hosts: list[str], control_bases: dict[str, int]) -> list[int]:
    """Every port the deployment listens on: per-rank control and transport
    listeners, the four collective listen ports and the twelve collective
    session ports above them."""
    size = len(hosts)
    ports = set()
    ports.update(control_bases["control"] + index for index in range(size))
    ports.update(control_bases["collective"] + index for index in range(2 * size))
    ports.update(control_bases["transport"] + index for index in range(size))
    return sorted(ports)


def check_reserved(ports: list[int], reserved: list[tuple[int, int]]) -> None:
    for port in ports:
        if not any(first <= port <= last for first, last in reserved):
            raise LaneError(f"listener {port} is outside the reserved lane port blocks")


def prepare(environment: dict[str, str] | None = None) -> dict:
    env = dict(os.environ if environment is None else environment)
    required = ("SPARK_QUEUE_RUNTIME_ROOT", "SPARK_QUEUE_RANK", "SPARK_QUEUE_SIZE",
                "SPARK_QUEUE_ATTEMPT")
    for name in required:
        if not env.get(name):
            raise LaneError(f"missing required queue variable {name}")
    attempt = env["SPARK_QUEUE_ATTEMPT"]
    if not re.fullmatch(r"[0-9a-f]{32}", attempt):
        raise LaneError("run through the authoritative spark queue (bad attempt id)")
    root = Path(env["SPARK_QUEUE_RUNTIME_ROOT"])
    if str(root) != "/tmp/sparkqueue-" + attempt:
        raise LaneError(f"unexpected job runtime namespace: {root}")
    if root.exists():
        raise LaneError(f"private runtime roots are never reused: {root}")
    try:
        rank = int(env["SPARK_QUEUE_RANK"], 10)
        size = int(env["SPARK_QUEUE_SIZE"], 10)
    except ValueError:
        raise LaneError("queue rank/size must be decimal integers")
    hosts = parse_hosts(env_str(env, "QWEN38_27B_LANE_HOSTS", LANE_HOSTS_DEFAULT))
    if size != len(hosts):
        raise LaneError(f"queue size {size} differs from lane topology {len(hosts)}")
    if not 0 <= rank < size:
        raise LaneError(f"queue rank {rank} outside 0..{size - 1}")
    parse_mesh_ranks(env_str(env, "QWEN38_27B_LANE_MESH_RANKS", LANE_MESH_RANKS_DEFAULT), size)
    if env_int(env, "QWEN38_27B_LANE_WEIGHTD_LANE", LANE_WEIGHTD_LANE_DEFAULT) >= size * 2:
        raise LaneError("weightd mesh lane exceeds the shared-daemon lane capacity")
    control_bases = {
        "control": env_int(env, "QWEN38_27B_LANE_CONTROL_BASE", LANE_CONTROL_BASE_DEFAULT),
        "collective": env_int(env, "QWEN38_27B_LANE_COLLECTIVE_BASE", LANE_COLLECTIVE_BASE_DEFAULT),
        "transport": env_int(env, "QWEN38_27B_LANE_TRANSPORT_BASE", LANE_TRANSPORT_BASE_DEFAULT),
    }
    reserved = parse_port_ranges(env.get("SPARK_QUEUE_PORTS", ""))
    if not reserved:
        reserved = [(base, base + 2 * len(hosts) - 1) for base in control_bases.values()]
    check_reserved(listener_ports(hosts, control_bases), reserved)
    firmware_root = Path(env_str(env, "QWEN38_27B_LANE_FIRMWARE_ROOT", ""))
    if not str(firmware_root):
        raise LaneError("set QWEN38_27B_LANE_FIRMWARE_ROOT to a verified module build")
    firmware = verify_firmware(firmware_root)
    shared_socket = env_str(env, "QWEN38_27B_LANE_SHARED_SOCKET", LANE_SHARED_SOCKET_DEFAULT)
    if not shared_socket.startswith("/"):
        raise LaneError("the shared weightd socket must be an absolute path")
    max_sequence_positions = env_int(env, "QWEN38_27B_LANE_MAX_SEQUENCE_POSITIONS",
                                     LANE_MAX_SEQUENCE_POSITIONS_DEFAULT)
    if not 1 <= max_sequence_positions <= 8192:
        raise LaneError("max_sequence_positions exceeds the adapter cap (8192)")
    kv_maximum_bytes = env_int(env, "QWEN38_27B_LANE_KV_BACKING_MAXIMUM_BYTES",
                               LANE_KV_BACKING_MAXIMUM_BYTES_DEFAULT)
    if kv_maximum_bytes <= 0:
        raise LaneError("kv backing maximum must be finite and positive")
    root.mkdir(mode=0o700)
    staged = stage_runtime(
        root,
        firmware_root,
        hosts,
        rank,
        attempt,
        Path(env_str(env, "QWEN38_27B_LANE_PACK_DIR", LANE_PACK_DIR_DEFAULT)),
        env_str(env, "QWEN38_27B_LANE_PACK_TEMPLATE", LANE_PACK_TEMPLATE_DEFAULT),
        env_str(env, "QWEN38_27B_LANE_MODEL_REVISION", LANE_MODEL_REVISION_DEFAULT),
        max_sequence_positions,
        control_bases,
        shared_socket,
        kv_maximum_bytes,
        env.get("QWEN38_27B_LANE_TRUST_SOURCE_DIGEST") == "1",
    )
    staged.update(
        family=LANE_FAMILY,
        topology=LANE_TOPOLOGY,
        rank=rank,
        hosts=hosts,
        shared_socket=shared_socket,
        weightd_lane=env_int(env, "QWEN38_27B_LANE_WEIGHTD_LANE", LANE_WEIGHTD_LANE_DEFAULT),
        mesh_ranks=env_str(env, "QWEN38_27B_LANE_MESH_RANKS", LANE_MESH_RANKS_DEFAULT),
        source_commit=firmware["source_commit"],
    )
    return staged


def main() -> int:
    try:
        staged = prepare()
    except (LaneError, OSError) as error:
        print(f"qwen38_27b_lane_deployment: FAIL: {error}", file=sys.stderr)
        return 1
    print(json.dumps(staged, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
