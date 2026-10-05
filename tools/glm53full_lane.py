#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path

WORLD = 16
HEX = "0123456789abcdef"
HOSTS = [f"spark{HEX[index]}" for index in range(WORLD)]
REVISIONS = {
    "fp8": "935644c05e76fc198714f4cca449fd8b970ff6d7",
    "bf16": "935644c05e76fc198714f4cca449fd8b970ff6d7",
    "fp8_s1": "304b8051cfb2b260b61ce0cbe330e02a98e73639",
    "nvfp4_s1": "304b8051cfb2b260b61ce0cbe330e02a98e73639",
}
ARM_CODECS = {"fp8": "fp8", "bf16": "bf16", "fp8_s1": "fp8", "nvfp4_s1": "nvfp4"}
SCORE_MEMBERS = ("score_dump_directory", "score_probe_path", "score_tier2_rows_path")
CHAT_TEMPLATE_PATH = Path(__file__).resolve().parents[1] / "model-families/glm52/chat_template.json"
EOS_TOKEN_IDS = [154820, 154827, 154829]
TOKENIZER_SHA256 = "19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d"
TOKENIZER_TOKEN_COUNT = 154856
BLOCK_TOKENS = 64
CONTRACT_PATH = Path(__file__).resolve().parents[1] / "model_contracts/glm53_full_authoritative.json"


def contract_kv_page_bytes():
    contract = json.loads(CONTRACT_PATH.read_text())
    geometry = contract["geometry"]
    index_layers = contract["structural_census"]["indexer_full_layers_0_77"]
    latent = geometry["latent_dimension"] + geometry["rope_dimension"]
    return BLOCK_TOKENS * (geometry["layer_count"] * latent + len(index_layers) * geometry["dsa_index_head_dimension"]) * 2


KV_FULL_PAGE_BYTES = contract_kv_page_bytes()
if KV_FULL_PAGE_BYTES % WORLD != 0:
    raise SystemExit(f"a {KV_FULL_PAGE_BYTES}-byte KV page does not split across {WORLD} ranks")
KV_PAGE_BYTES = KV_FULL_PAGE_BYTES // WORLD
MAX_LANES = 16


def lane_ports(lane):
    if not 0 <= lane < MAX_LANES:
        raise SystemExit(f"lane must be 0..{MAX_LANES - 1}")
    return {
        "control": 23000 + 16 * lane,
        "collective": 53000 + 16 * lane,
        "transport": 64000 + 16 * lane,
        "session": 24256 + 32 * lane,
    }


def default_node_root(lane):
    return f"glmfull-lane{lane}/root"


def node_root(value, lane):
    value = default_node_root(lane) if value is None else value
    if value == "" or value.startswith("/") or value.endswith("/") or any(part in ("", ".", "..") for part in value.split("/")):
        raise SystemExit(f"node root must be a normalized path relative to the node's home directory: {value!r}")
    return value


def runtime_root(host, lane, root=None):
    return f"/home/{host}/{node_root(root, lane)}"


def pack_name(arm, rank):
    return f"glm53full.{arm}.tp16-rank{rank}.glm52sp"


def chat_template():
    return json.loads(CHAT_TEMPLATE_PATH.read_text(encoding="utf-8"))


def collective_identifier(lane):
    return (0x474C << 32) | (lane + 1)


def session_matrix(base):
    return [[0 if column == row else base + column for column in range(WORLD)] for row in range(WORLD)]


def stage_config(rank, lane, arm, max_sequence_positions, execution_row_capacity):
    ports = lane_ports(lane)
    return {
        "schema_version": 3,
        "model_revision": REVISIONS[arm],
        "expert_weight_codec": ARM_CODECS[arm],
        "stage_pack_path": f"packs/{pack_name(arm, rank)}",
        "max_sequence_positions": max_sequence_positions,
        "execution_row_capacity": execution_row_capacity,
        "decode_split_context_threshold": 64,
        "tp_degree": WORLD,
        "tp_rank": rank,
        "tp_collective": {
            "backend": "hidden_transport",
            "backend_module_path": "lib/hidden_transport.so",
            "algorithms": ["tree"],
            "collective_identifier": collective_identifier(lane),
            "listen_port": ports["collective"] + rank,
            "connect_timeout_milli": 180000,
            "operation_timeout_milli": 30000,
            "peer_hosts": list(HOSTS),
            "peer_ports": [ports["collective"] + peer for peer in range(WORLD)],
            "split_ring_min_payload_bytes": 0,
            "direct_all_to_all_max_payload_bytes": 0,
            "rail_peer_hosts": [list(HOSTS), list(HOSTS)],
            "step_rail_indices": [0] + [1] * (WORLD - 1),
            "session_ports": session_matrix(ports["session"]),
            "session_ports_hc": session_matrix(ports["session"] + 16),
            "wait_mode": "hardware",
        },
    }


KV_BACKING_IN_FLIGHT_PAGES = 2


def kv_pages(sequences, max_sequence_positions, kv_backing_bytes, kv_physical_bytes):
    lane_pages = (max_sequence_positions + BLOCK_TOKENS - 1) // BLOCK_TOKENS
    covered = sequences * lane_pages
    physical = covered if kv_physical_bytes is None else min(covered, kv_physical_bytes // KV_PAGE_BYTES)
    if physical < lane_pages:
        raise SystemExit(f"kv physical budget holds {physical} pages; one lane needs {lane_pages}")
    spill = kv_backing_bytes // KV_PAGE_BYTES - KV_BACKING_IN_FLIGHT_PAGES
    if spill < covered - physical:
        raise SystemExit(f"kv backing holds {spill} pages; lanes beyond the physical pool need {covered - physical}")
    return physical, physical + spill


def deployment(lane, arm, socket_path, kv_backing_bytes, kv_snapshot_bytes, max_sequence_positions, sequences, row_capacity, inflight, root=None,
               kv_physical_bytes=None):
    ports = lane_ports(lane)
    pages, logical_pages = kv_pages(sequences, max_sequence_positions, kv_backing_bytes, kv_physical_bytes)
    nodes = []
    for rank, host in enumerate(HOSTS):
        root_path = runtime_root(host, lane, root)
        nodes.append({
            "rank_index": rank,
            "stage_index": rank,
            "runtime_root": root_path,
            "node_target": f"cuda.sm121.glm52.resident_decode_stage.bf16.expert_{ARM_CODECS[arm]}",
            "transport_host": host,
            "adapter_configuration_path": f"config/stage_{rank:02d}.json",
            "kv_backing_directory": f"{root_path}/kvcache",
            "kv_partition": "/",
            "kv_backing_maximum_bytes": kv_backing_bytes,
            "kv_snapshot_directory": f"{root_path}/kvsnapshot",
            "kv_snapshot_maximum_bytes": kv_snapshot_bytes,
            "control_endpoint": {"kind": "tcp", "host": host, "port": ports["control"] + rank},
        })
    return {
        "schema_version": 2,
        "eos_token_ids": list(EOS_TOKEN_IDS),
        "coordinator_rank_index": 0,
        "adapter": {"shared_object_path": "lib/model_serving_adapter.so"},
        "driver": {"shared_object_path": "lib/model_driver.so", "program_name": "resident_decode"},
        "transport": {"shared_object_path": "lib/hidden_transport.so", "mode": "host-rdma", "control_port_base": ports["transport"]},
        "weightd": {"socket_path": socket_path},
        "runtime_limits": {
            "max_inflight_submissions": inflight,
            "max_active_sequences": sequences,
            "max_input_rows": row_capacity,
            "resident_sequence_capacity": sequences,
            "kv_logical_page_capacity": logical_pages,
            "kv_physical_page_capacity": pages,
            "max_sequence_positions": max_sequence_positions,
        },
        "tokenizer": {"path": "tokenizer/tokenizer.json", "sha256": TOKENIZER_SHA256, "vocabulary_size": TOKENIZER_TOKEN_COUNT},
        "chat_template": chat_template(),
        "nodes": nodes,
    }


def score_members(arguments):
    members = {name: getattr(arguments, name, None) for name in SCORE_MEMBERS}
    members = {name: value for name, value in members.items() if value is not None}
    if not members:
        return {}
    if "score_dump_directory" not in members:
        raise SystemExit("score_probe_path and score_tier2_rows_path require score_dump_directory")
    for name, value in members.items():
        if value == "" or value.startswith("/") or value.endswith("/") or any(part in ("", ".", "..") for part in value.split("/")):
            raise SystemExit(f"{name} must be a normalized path relative to the arm's runtime root: {value!r}")
    return members


def render(arguments):
    arm = getattr(arguments, "arm", None) or arguments.codec
    if arm not in REVISIONS:
        raise SystemExit(f"arm must be one of {tuple(REVISIONS)}")
    if ARM_CODECS[arm] != arguments.codec:
        raise SystemExit(f"arm {arm} carries {ARM_CODECS[arm]} experts, not {arguments.codec}")
    if not arguments.socket.startswith("/") or not arguments.socket.endswith(".sock"):
        raise SystemExit("weightd socket must be an absolute .sock path")
    if arguments.kv_backing_bytes <= 0:
        raise SystemExit("kv backing must be a finite positive byte count")
    if arguments.kv_snapshot_bytes <= 0:
        raise SystemExit("kv snapshot store must be a finite positive byte count")
    if not 1 <= arguments.sequences <= 1024 or not 1 <= arguments.inflight <= 4:
        raise SystemExit("sequences must be 1..1024 (the serving adapter's active-sequence ceiling) and inflight 1..4")
    files = {"model_resident.json": deployment(arguments.lane, arm, arguments.socket, arguments.kv_backing_bytes,
                                               arguments.kv_snapshot_bytes, arguments.max_sequence_positions, arguments.sequences,
                                               arguments.execution_row_capacity, arguments.inflight,
                                               getattr(arguments, "node_root", None), arguments.kv_physical_bytes)}
    score = score_members(arguments)
    for rank in range(WORLD):
        files[f"config/stage_{rank:02d}.json"] = dict(stage_config(rank, arguments.lane, arm,
                                                                  arguments.max_sequence_positions,
                                                                  arguments.execution_row_capacity), **score)
    return {name: json.dumps(document, indent=1) + "\n" for name, document in files.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lane", type=int, required=True)
    parser.add_argument("--codec", choices=("fp8", "bf16", "nvfp4"), required=True)
    parser.add_argument("--arm", choices=tuple(REVISIONS))
    parser.add_argument("--socket", required=True)
    parser.add_argument("--kv-backing-bytes", type=int, required=True)
    parser.add_argument("--kv-snapshot-bytes", type=int, required=True)
    parser.add_argument("--kv-physical-bytes", type=int)
    parser.add_argument("--max-sequence-positions", type=int, required=True)
    parser.add_argument("--execution-row-capacity", type=int, required=True)
    parser.add_argument("--sequences", type=int, required=True)
    parser.add_argument("--inflight", type=int, required=True)
    parser.add_argument("--score-dump-directory", dest="score_dump_directory")
    parser.add_argument("--score-probe-path", dest="score_probe_path")
    parser.add_argument("--score-tier2-rows-path", dest="score_tier2_rows_path")
    parser.add_argument("--node-root", dest="node_root")
    parser.add_argument("--output", required=True)
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    files = render(arguments)
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
    print(json.dumps({"lane": arguments.lane, "codec": arguments.codec, "arm": arguments.arm or arguments.codec,
                      "node_root": node_root(arguments.node_root, arguments.lane),
                      "files": len(files), **lane_ports(arguments.lane),
                      "collective_identifier": collective_identifier(arguments.lane)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
