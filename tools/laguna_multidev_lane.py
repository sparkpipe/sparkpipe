#!/usr/bin/env python3
"""Laguna TP8xPP2 multidev deployment generator for one weightd lane."""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fleet_weightd import fleet_weightd_socket_error  # noqa: E402

LANE_COUNT = 16
WORLD = 16
TP = 8
PP = 2
LAYER_COUNT = 48
LAYERS_PER_STAGE = LAYER_COUNT // PP
HEX = "0123456789abcdef"
HOSTS = [f"spark{HEX[i]}" for i in range(WORLD)]

SESSION_SPAN = 64

MESH_RANKS = ",".join(str(i) for i in range(WORLD))

MODEL_REVISION = "0f573140834b11cfac0c2af97a101a7a69a13e22"
NODE_TARGET = "cuda.sm121.laguna.resident_decode_stage.bf16.expert_bf16"
EXPERT_CODEC = "bf16"

DEPLOYED_PACK_TEMPLATE = (
    "/home/{host}/sparkdata/laguna-s-2.1.bf16.tp8pp2/packs/"
    "laguna_stage.tp8.pp2.stage{stage}.rank{rank}.lgsp")

EXPERTS_MAGIC = 0x58504557
EXPERTS_VERSION = 2


def lane_ports(lane: int) -> dict:
    if not 0 <= lane < LANE_COUNT:
        raise ValueError(f"lane must be 0..{LANE_COUNT - 1}")
    return {"control": 23000 + 16 * lane, "collective": 53000 + 16 * lane,
            "transport": 64000 + 16 * lane, "session": 23168 + SESSION_SPAN * lane}


def host_of(rank: int) -> str:
    return HOSTS[rank]


def stage_of(rank: int) -> int:
    return rank // TP


def tp_rank_of(rank: int) -> int:
    return rank % TP


def group_hosts(rank: int) -> list[str]:
    first = stage_of(rank) * TP
    return HOSTS[first:first + TP]


def deployed_pack(rank: int) -> str:
    return DEPLOYED_PACK_TEMPLATE.format(
        host=host_of(rank), stage=stage_of(rank), rank=rank)


def session_table(lane: int) -> list[list[int]]:
    base = lane_ports(lane)["session"]
    table = []
    for a in range(TP):
        row = []
        for b in range(TP):
            if a == b:
                row.append(0)
            else:
                row.append(base + 7 * a + (b if b < a else b - 1))
        table.append(row)
    return table


def tp_collective(rank: int, collective_identifier: int, lane: int) -> dict:
    peers = group_hosts(rank)
    collective = lane_ports(lane)["collective"]
    return {
        "backend": "hidden_transport",
        "backend_module_path": "lib/hidden_transport.so",
        "algorithms": ["tree"],
        "collective_identifier": collective_identifier,
        "listen_port": collective + tp_rank_of(rank),
        "connect_timeout_milli": 300000,
        "operation_timeout_milli": 30000,
        "peer_hosts": peers,
        "peer_ports": [collective + peer for peer in range(TP)],
        "split_ring_min_payload_bytes": 0,
        "direct_all_to_all_max_payload_bytes": 0,
        "rail_peer_hosts": [peers, peers],
        "step_rail_indices": [0] + [1] * (TP - 1),
        "session_ports": session_table(lane),
        "session_ports_hc": session_table(lane),
    }


def adapter_config(rank: int, collective_identifier: int, lane: int) -> dict:
    return {
        "schema_version": 3,
        "model_revision": MODEL_REVISION,
        "expert_weight_codec": EXPERT_CODEC,
        "stage_pack_path": "packs/" + os.path.basename(deployed_pack(rank)),
        "max_sequence_positions": 32768,
        "execution_row_capacity": 128,
        "decode_split_context_threshold": 2048,
        "tp_degree": TP,
        "tp_rank": tp_rank_of(rank),
        "tp_collective": tp_collective(rank, collective_identifier, lane),
    }


def resident_deployment(runtime_root: str, weightd_socket: str,
                        kv_backing_bytes: int, kv_page_capacity: int,
                        lane: int) -> dict:
    ports = lane_ports(lane)
    nodes = []
    for rank, host in enumerate(HOSTS):
        nodes.append({
            "rank_index": rank,
            "stage_index": rank,
            "runtime_root": runtime_root,
            "node_target": NODE_TARGET,
            "transport_host": host,
            "adapter_configuration_path": "config/adapter.json",
            "kv_backing_directory": os.path.join(runtime_root, "kvcache"),
            "kv_backing_maximum_bytes": kv_backing_bytes,
            "control_endpoint": {
                "kind": "tcp",
                "host": host,
                "port": ports["control"] + rank,
            },
        })
    return {
        "schema_version": 2,
        "eos_token_ids": [2, 24],
        "coordinator_rank_index": 0,
        "adapter": {"shared_object_path": "lib/model_serving_adapter.so"},
        "driver": {
            "shared_object_path": "lib/model_driver.so",
            "program_name": "resident_decode",
        },
        "transport": {
            "shared_object_path": "lib/hidden_transport.so",
            "mode": "host-rdma",
            "control_port_base": ports["transport"],
        },
        "weightd": {
            "socket_path": weightd_socket,
        },
        "runtime_limits": {
            "max_inflight_submissions": 4,
            "max_active_sequences": 16,
            "max_input_rows": 128,
            "resident_sequence_capacity": 16,
            "kv_logical_page_capacity": kv_page_capacity,
            "kv_physical_page_capacity": kv_page_capacity,
        },
        "nodes": nodes,
    }


def budgets(pack_path: str) -> int:
    pack = os.path.abspath(pack_path)
    sidecar = pack + ".experts"
    if not os.path.isfile(pack):
        raise SystemExit(f"budgets: pack not found: {pack}")
    if not os.path.isfile(sidecar):
        raise SystemExit(f"budgets: .experts sidecar missing: {sidecar} "
                         "(generate it with "
                         "tools/laguna_multidev_experts_manifest.sh)")
    with open(sidecar, "rb") as handle:
        head = handle.read(16)
        if len(head) != 16:
            raise SystemExit("budgets: short sidecar header")
        magic, version, count, reserved = struct.unpack("<IIII", head)
        if magic != EXPERTS_MAGIC or version != EXPERTS_VERSION \
                or reserved != 0 or count == 0:
            raise SystemExit("budgets: not a v2 routed-expert sidecar")
        spans = []
        for _ in range(count):
            record = handle.read(48)
            if len(record) != 48:
                raise SystemExit("budgets: short sidecar record")
            _, _, _, _, offset, span_bytes = struct.unpack_from("<4I2Q", record)
            spans.append((offset, span_bytes))
        trailing = handle.read(1)
        if trailing:
            raise SystemExit("budgets: trailing bytes after last record")
    spans.sort()
    spine_bytes = 0
    cursor = 0
    pack_bytes = os.path.getsize(pack)
    for offset, span_bytes in spans:
        if offset < cursor or offset + span_bytes > pack_bytes:
            raise SystemExit("budgets: sidecar spans disagree with the pack")
        spine_bytes += offset - cursor
        cursor = offset + span_bytes
    spine_bytes += pack_bytes - cursor
    chunk = 2 * 1024 * 1024
    pool_bytes = -(-pack_bytes // chunk) * chunk
    print(f"{pool_bytes} {spine_bytes + 256}")
    return 0


def smoke_budgets(source: str, rank: int) -> int:
    document = json.load(open(source, encoding="utf-8"))
    if document.get("family") != "laguna":
        raise SystemExit("budgets source is not the laguna census manifest")
    bases = document["provenance"]["byte_bases"]
    spans = bases["per_rank_expert_span_bytes"]
    w1, w2 = int(spans["w1"]), int(spans["w2"])
    chunk = 2 * 1024 * 1024
    stage = stage_of(rank)
    raw = 0
    chunked = 0
    pairs = 0
    for entry in document["experts"]:
        if int(entry["layer"]) // LAYERS_PER_STAGE != stage:
            continue
        pairs += 1
        raw += w1 + w2
        chunked += -(-w1 // chunk) * chunk + -(-w2 // chunk) * chunk
    if pairs == 0:
        raise SystemExit(f"rank {rank}: the census head has no pairs for "
                         f"stage {stage}")
    print(f"{raw} {chunked}")
    return 0


def emit_wset(source: str, output: str, rank: int | None = None) -> int:
    document = json.load(open(source, encoding="utf-8"))
    if document.get("family") != "laguna":
        raise SystemExit("wset source is not the laguna census manifest")
    pairs = sorted({(int(e["layer"]), int(e["expert"]))
                    for e in document["experts"]})
    filtered = pairs
    if rank is not None:
        stage = stage_of(rank)
        filtered = [pair for pair in pairs
                    if pair[0] // LAYERS_PER_STAGE == stage]
        if not filtered:
            raise SystemExit(f"rank {rank}: the census head has no pairs "
                             f"for stage {stage}")
    with open(output, "wb") as handle:
        for layer, expert in filtered:
            handle.write(layer.to_bytes(4, "little"))
            handle.write(expert.to_bytes(4, "little"))
    print(json.dumps({"wset": output, "keys": len(filtered),
                      "rank_filter": rank}))
    return 0


def valid_path(value: str, name: str) -> str:
    if not os.path.isabs(value) or os.path.normpath(value) != value or value == "/":
        raise SystemExit(f"{name} must be an absolute normalized path, got {value!r}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lane", type=int)
    parser.add_argument("--runtime-root")
    parser.add_argument("--weightd-socket")
    parser.add_argument("--output-dir")
    parser.add_argument("--rank", type=int)
    parser.add_argument("--collective-identifier", type=int)
    parser.add_argument("--kv-backing-bytes", type=int)
    parser.add_argument("--kv-page-capacity", type=int)
    parser.add_argument("--budgets", metavar="PACK")
    parser.add_argument("--smoke-budgets", nargs=2, metavar=("MANIFEST", "RANK"))
    parser.add_argument("--emit-wset", metavar="OUTPUT")
    parser.add_argument("--wset-source", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "model-families", "laguna", "smoke_experts.json"))
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()

    if arguments.budgets:
        return budgets(arguments.budgets)
    if arguments.smoke_budgets:
        source, rank_text = arguments.smoke_budgets
        if not rank_text.isdigit() or not 0 <= int(rank_text) < WORLD:
            raise SystemExit(f"rank must be 0..{WORLD - 1}")
        return smoke_budgets(source, int(rank_text))
    if arguments.emit_wset:
        return emit_wset(arguments.wset_source, arguments.emit_wset, arguments.rank)

    missing = [name for name, value in (
        ("--lane", arguments.lane),
        ("--runtime-root", arguments.runtime_root),
        ("--weightd-socket", arguments.weightd_socket),
        ("--output-dir", arguments.output_dir),
        ("--collective-identifier", arguments.collective_identifier),
        ("--kv-backing-bytes", arguments.kv_backing_bytes),
        ("--kv-page-capacity", arguments.kv_page_capacity)) if value is None]
    if missing:
        raise SystemExit("the following arguments are required: "
                         + ", ".join(missing))
    if not 0 <= arguments.lane < LANE_COUNT:
        raise SystemExit(f"lane must be 0..{LANE_COUNT - 1}")
    if arguments.rank is not None and not 0 <= arguments.rank < WORLD:
        raise SystemExit(f"rank must be 0..{WORLD - 1}")
    runtime_root = valid_path(arguments.runtime_root, "runtime root")
    if "sparkdata" in runtime_root.split(os.sep):
        raise SystemExit("runtime root must not live inside the shared sparkdata tree")
    socket = valid_path(arguments.weightd_socket, "weightd socket")
    if arguments.kv_backing_bytes <= 0:
        raise SystemExit("kv backing must be a finite positive cap")
    if arguments.kv_page_capacity <= 0:
        raise SystemExit("kv page capacity must be positive")
    socket_error = fleet_weightd_socket_error(socket)
    if socket_error:
        raise SystemExit(socket_error)
    if arguments.collective_identifier <= 0:
        raise SystemExit("collective identifier must be positive")

    lane = arguments.lane
    ranks = range(WORLD) if arguments.rank is None else [arguments.rank]
    files = {
        "deployment.json": json.dumps(resident_deployment(
            runtime_root, socket, arguments.kv_backing_bytes,
            arguments.kv_page_capacity, lane), indent=2) + "\n"}
    for rank in ranks:
        name = f"adapter.spark{HEX[rank]}.json" if arguments.rank is None \
            else "adapter.json"
        files[name] = json.dumps(adapter_config(
            rank, arguments.collective_identifier, lane), indent=2) + "\n"
    output_dir = os.path.abspath(arguments.output_dir)
    if arguments.check:
        for name, text in files.items():
            current = open(os.path.join(output_dir, name), encoding="utf-8").read()
            if current != text:
                print(f"drift in {name}: regenerate with "
                      "tools/laguna_multidev_lane.py", file=sys.stderr)
                return 1
        print("multidev lane output matches")
        return 0
    os.makedirs(output_dir, exist_ok=True)
    for name, text in files.items():
        with open(os.path.join(output_dir, name), "w", encoding="utf-8") as fh:
            fh.write(text)
    ports = lane_ports(lane)
    print(json.dumps({"lane": lane, "rank": arguments.rank, "output": output_dir,
                      "control_port": None if arguments.rank is None
                      else ports["control"] + arguments.rank,
                      "collective_base": ports["collective"],
                      "collective_id": arguments.collective_identifier,
                      "transport_base": ports["transport"],
                      "session_base": ports["session"],
                      "pack": None if arguments.rank is None
                      else deployed_pack(arguments.rank)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
