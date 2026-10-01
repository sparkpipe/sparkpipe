#!/usr/bin/env python3
import argparse
import json
import os
import sys

WORLD = 16
HEX = "0123456789abcdef"
HOSTS = [f"spark{HEX[index]}" for index in range(WORLD)]
MODEL_REVISION = "e0dfe7cd0f6e3b572bbbc0a8a84947469e428cc3"
EOS_TOKEN_ID = 156895
TOKENIZER_SHA256 = "40fb9d7d7795b8bd305aeff39ce9963f3f450915b9553f2938e009be9a1fed60"
TOKENIZER_TOKEN_COUNT = 157153
CODECS = ("bf16", "fp8")
MAX_LANES = 16
ADAPTER_MEMBERS = ("schema_version", "model_revision", "expert_weight_codec", "stage_pack_path",
                   "max_sequence_positions", "execution_row_capacity", "decode_split_context_threshold",
                   "tp_degree", "tp_rank", "tp_collective")


def lane_ports(lane):
    if not 0 <= lane < MAX_LANES:
        raise SystemExit(f"lane must be 0..{MAX_LANES - 1}")
    return {
        "control": 23000 + 16 * lane,
        "collective": 53000 + 16 * lane,
        "transport": 64000 + 16 * lane,
        "session": 24256 + 32 * lane,
    }


def runtime_root(host, lane):
    return f"/home/{host}/ling-lane{lane}/root"


def collective_identifier(lane):
    return (0x4C49 << 32) | (lane + 1)


def session_matrix(base):
    return [[0 if column == row else base + column for column in range(WORLD)] for row in range(WORLD)]


def stage_config(rank, lane, codec, max_sequence_positions, execution_row_capacity):
    ports = lane_ports(lane)
    return {
        "schema_version": 3,
        "model_revision": MODEL_REVISION,
        "expert_weight_codec": codec,
        "stage_pack_path": f"packs/ling.{codec}.tp16.rank{rank:x}.sp",
        "max_sequence_positions": max_sequence_positions,
        "execution_row_capacity": execution_row_capacity,
        "decode_split_context_threshold": 2048,
        "tp_degree": WORLD,
        "tp_rank": rank,
        "tp_collective": {
            "backend": "hidden_transport",
            "backend_module_path": "lib/hidden_transport.so",
            "algorithms": ["tree"],
            "collective_identifier": collective_identifier(lane),
            "listen_port": ports["collective"] + rank,
            "connect_timeout_milli": 30000,
            "operation_timeout_milli": 30000,
            "peer_hosts": list(HOSTS),
            "peer_ports": [ports["collective"] + peer for peer in range(WORLD)],
            "split_ring_min_payload_bytes": 0,
            "direct_all_to_all_max_payload_bytes": 0,
            "rail_peer_hosts": [list(HOSTS), list(HOSTS)],
            "step_rail_indices": [0] + [1] * (WORLD - 1),
            "session_ports": session_matrix(ports["session"]),
            "session_ports_hc": session_matrix(ports["session"] + 16),
        },
    }


def deployment(lane, codec, socket_path, kv_backing_bytes, max_sequence_positions):
    ports = lane_ports(lane)
    nodes = []
    for rank, host in enumerate(HOSTS):
        root = runtime_root(host, lane)
        nodes.append({
            "rank_index": rank,
            "stage_index": rank,
            "runtime_root": root,
            "node_target": f"cuda.sm121.ling.resident_decode_stage.bf16.expert_{codec}",
            "transport_host": host,
            "adapter_configuration_path": f"config/stage_{rank:02d}.json",
            "kv_backing_directory": f"{root}/kvcache",
            "kv_backing_maximum_bytes": kv_backing_bytes,
            "control_endpoint": {"kind": "tcp", "host": host, "port": ports["control"] + rank},
        })
    return {
        "schema_version": 2,
        "eos_token_ids": [EOS_TOKEN_ID],
        "coordinator_rank_index": 0,
        "adapter": {"shared_object_path": "lib/model_serving_adapter.so"},
        "driver": {"shared_object_path": "lib/model_driver.so", "program_name": "resident_decode"},
        "transport": {"shared_object_path": "lib/hidden_transport.so", "mode": "host-rdma", "control_port_base": ports["transport"]},
        "weightd": {"socket_path": socket_path},
        "runtime_limits": {
            "max_inflight_submissions": 1,
            "max_active_sequences": 16,
            "max_input_rows": 1024,
            "resident_sequence_capacity": 16,
            "kv_logical_page_capacity": 8192,
            "kv_physical_page_capacity": 8192,
            "max_sequence_positions": max_sequence_positions,
        },
        "tokenizer": {"path": "tokenizer/tokenizer.json", "sha256": TOKENIZER_SHA256, "vocabulary_size": TOKENIZER_TOKEN_COUNT},
        "nodes": nodes,
    }


def render(lane, codec, socket_path, kv_backing_bytes, max_sequence_positions, execution_row_capacity):
    if codec not in CODECS:
        raise SystemExit(f"codec must be one of {CODECS}")
    if not socket_path.startswith("/") or not socket_path.endswith(".sock"):
        raise SystemExit("weightd socket must be an absolute .sock path")
    if kv_backing_bytes <= 0:
        raise SystemExit("kv backing must be a finite positive byte count")
    files = {"model_resident.json": deployment(lane, codec, socket_path, kv_backing_bytes, max_sequence_positions)}
    for rank in range(WORLD):
        files[f"config/stage_{rank:02d}.json"] = stage_config(rank, lane, codec, max_sequence_positions, execution_row_capacity)
    return {name: json.dumps(document, indent=1) + "\n" for name, document in files.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lane", type=int, required=True)
    parser.add_argument("--codec", choices=CODECS, required=True)
    parser.add_argument("--socket", required=True)
    parser.add_argument("--kv-backing-bytes", type=int, required=True)
    parser.add_argument("--max-sequence-positions", type=int, required=True)
    parser.add_argument("--execution-row-capacity", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    files = render(arguments.lane, arguments.codec, arguments.socket, arguments.kv_backing_bytes,
                   arguments.max_sequence_positions, arguments.execution_row_capacity)
    for name, text in files.items():
        path = os.path.join(arguments.output, name)
        if arguments.check:
            with open(path) as handle:
                if handle.read() != text:
                    print(f"drift in {path}", file=sys.stderr)
                    return 1
            continue
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as handle:
            handle.write(text)
    ports = lane_ports(arguments.lane)
    print(json.dumps({"lane": arguments.lane, "codec": arguments.codec, "files": len(files), **ports,
                      "collective_identifier": collective_identifier(arguments.lane)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
