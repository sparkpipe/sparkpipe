#!/usr/bin/env python3
import argparse
import json
import os
import sys

MODEL_REVISION = "mimo26flash.mxfp4.tp4.v2"
EOS_TOKEN_IDS = [151643, 151645, 151672]
TOKENIZER_SHA256 = "ff15eb925890d6b71b5160de4b846fbd13178438ab463b38ecc953e8cd1dcb3e"
TOKENIZER_VOCABULARY_SIZE = 151675
TP_DEGREE = 4
MAX_LANES = 16


def lane_ports(lane):
    if not 2 <= lane <= 12:
        raise SystemExit("dev lanes are 2..12")
    return {"control": 23000 + 16 * lane, "transport": 64000 + 16 * lane}


def stage_config(rank, max_sequence_positions):
    return {
        "schema_version": 3,
        "model_revision": MODEL_REVISION,
        "stage_pack_path": f"packs/rank{rank}.sp",
        "max_sequence_positions": max_sequence_positions,
        "tp_degree": TP_DEGREE,
    }


def deployment(hosts, lane, socket_path, lanes, max_sequence_positions):
    ports = lane_ports(lane)
    blocks = lanes * (max_sequence_positions // 64)
    nodes = []
    for rank, host in enumerate(hosts):
        root = f"/home/{host}/mimo-lane{lane}/root"
        nodes.append({
            "rank_index": rank,
            "stage_index": rank,
            "runtime_root": root,
            "node_target": "cuda.sm121.mimo26.resident_decode_stage.mxfp4",
            "transport_host": host,
            "adapter_configuration_path": f"config/stage_{rank:02d}.json",
            "kv_backing_directory": f"{root}/kvcache",
            "kv_backing_maximum_bytes": 1073741824,
            "control_endpoint": {"kind": "tcp", "host": host, "port": ports["control"] + rank},
        })
    return {
        "schema_version": 2,
        "eos_token_ids": EOS_TOKEN_IDS,
        "prefix_reuse": False,
        "coordinator_rank_index": 0,
        "adapter": {"shared_object_path": "lib/model_serving_adapter.so"},
        "driver": {"shared_object_path": "lib/model_driver.so", "program_name": "resident_decode"},
        "transport": {"shared_object_path": "lib/hidden_transport.so", "mode": "host-rdma", "control_port_base": ports["transport"]},
        "weightd": {"socket_path": socket_path},
        "runtime_limits": {
            "max_inflight_submissions": 1,
            "max_active_sequences": lanes,
            "max_input_rows": 16,
            "resident_sequence_capacity": lanes,
            "kv_logical_page_capacity": blocks,
            "kv_physical_page_capacity": blocks,
            "max_sequence_positions": max_sequence_positions,
        },
        "tokenizer": {"path": "tokenizer/tokenizer.json", "sha256": TOKENIZER_SHA256, "vocabulary_size": TOKENIZER_VOCABULARY_SIZE},
        "nodes": nodes,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hosts", required=True)
    parser.add_argument("--lane", type=int, required=True)
    parser.add_argument("--socket", required=True)
    parser.add_argument("--lanes", type=int, required=True)
    parser.add_argument("--max-sequence-positions", type=int, required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    hosts = arguments.hosts.split(",")
    if len(hosts) != TP_DEGREE:
        raise SystemExit("mimo26 flash serves TP4: exactly four hosts")
    if arguments.max_sequence_positions % 64 != 0 or not 1 <= arguments.lanes <= 16:
        raise SystemExit("positions must be a multiple of 64 and lanes 1..16")
    files = {"model_resident.json": deployment(hosts, arguments.lane, arguments.socket, arguments.lanes, arguments.max_sequence_positions)}
    for rank in range(TP_DEGREE):
        files[f"config/stage_{rank:02d}.json"] = stage_config(rank, arguments.max_sequence_positions)
    for name, document in files.items():
        path = os.path.join(arguments.output, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as handle:
            handle.write(json.dumps(document, indent=1) + "\n")
    print(json.dumps({"lane": arguments.lane, "files": len(files), **lane_ports(arguments.lane)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
