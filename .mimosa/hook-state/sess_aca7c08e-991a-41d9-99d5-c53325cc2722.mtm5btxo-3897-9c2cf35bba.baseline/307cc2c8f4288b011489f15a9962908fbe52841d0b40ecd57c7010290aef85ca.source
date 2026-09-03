#!/usr/bin/env python3

from __future__ import annotations

import argparse
import itertools
import json
import pathlib
import sys
from typing import Any, Iterable

SCRIPT = pathlib.Path(__file__).resolve()
ROOT = SCRIPT.parents[2]
sys.path.insert(0, str(SCRIPT.parent))
from hardware_common import (  # noqa: E402
    canonical_json_bytes,
    is_sha256,
    load_json,
    require,
    sha256_bytes,
    sha256_file,
    write_json_atomic,
)

QUESTIONS_PATH = ROOT / "model_contracts" / "spark_hardware_questions.json"
PROBE_PLAN_PATH = ROOT / "qualification" / "spark" / "probe_plan.json"
WORKLOADS_PATH = ROOT / "qualification" / "spark" / "workload_profiles.json"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an exact Spark hardware qualification plan")
    parser.add_argument("--topology", required=True, type=pathlib.Path)
    parser.add_argument("--source-package-sha256", required=True)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    return parser.parse_args()


def cartesian(axis_values: dict[str, Iterable[Any]]) -> list[dict[str, Any]]:
    keys = list(axis_values)
    values = [list(axis_values[key]) for key in keys]
    return [dict(zip(keys, combination, strict=True)) for combination in itertools.product(*values)]


def unique(values: Iterable[Any]) -> list[Any]:
    result: list[Any] = []
    seen: set[str] = set()
    for value in values:
        key = json.dumps(value, sort_keys=True, separators=(",", ":"))
        if key not in seen:
            seen.add(key)
            result.append(value)
    return result


def topology_projection(document: dict[str, Any]) -> dict[str, Any]:
    topology = document.get("topology")
    nodes = document.get("compute_nodes")
    fabrics = document.get("fabrics")
    require(isinstance(topology, dict), "topology descriptor missing")
    require(topology.get("mode") in {"ring", "single_switch"}, "only ring and one-switch plans are currently enabled")
    require(isinstance(nodes, list) and nodes, "topology compute nodes missing")
    require(isinstance(fabrics, list) and fabrics, "topology fabrics missing")
    sorted_nodes = sorted(nodes, key=lambda item: int(item["rank"]))
    require([int(node["rank"]) for node in sorted_nodes] == list(range(len(sorted_nodes))), "topology ranks are not contiguous")
    projected_nodes = []
    for node in sorted_nodes:
        ports = node.get("ports")
        require(isinstance(ports, list) and ports, f"{node.get('name')}: no ports")
        projected_nodes.append({
            "name": str(node["name"]),
            "rank": int(node["rank"]),
            "nvme_device": str(node.get("nvme_device", "/dev/nvme0n1")),
            "addresses": [str(port["ipv4"]) for port in ports],
        })
    return {
        "name": str(topology["name"]),
        "mode": str(topology["mode"]),
        "nodes": projected_nodes,
        "fabrics": [
            {
                "name": str(fabric["name"]),
                "kind": str(fabric["kind"]),
                "speed_gbps": int(fabric["speed_gbps"]),
                "mtu_bytes": int(fabric["mtu_bytes"]),
            }
            for fabric in fabrics
        ],
    }


def directed_pairs(topology: dict[str, Any]) -> list[tuple[str, str]]:
    nodes = topology["nodes"]
    names = [str(node["name"]) for node in nodes]
    if topology["mode"] == "ring":
        pairs: list[tuple[str, str]] = []
        for index, name in enumerate(names):
            pairs.append((name, names[(index + 1) % len(names)]))
            pairs.append((name, names[(index - 1) % len(names)]))
        return unique(pairs)
    return [(source, destination) for source in names for destination in names if source != destination]


def role_cells(model: dict[str, Any], roles: Iterable[str], batches: list[int], contexts: Iterable[int]) -> list[dict[str, Any]]:
    return cartesian({
        "model_id": [model["id"]],
        "role": list(roles),
        "batch_size": batches,
        "context_tokens": list(contexts),
    })


def parameters_for_question(
    question_id: str,
    workloads: dict[str, Any],
    topology: dict[str, Any],
) -> tuple[bool, list[tuple[dict[str, Any], dict[str, Any]]]]:
    nodes = [str(node["name"]) for node in topology["nodes"]]
    pairs = directed_pairs(topology)
    models = workloads["models"]
    batches = workloads["batch_sizes"]
    contexts = workloads["context_buckets"]
    payloads = workloads["transport_payload_bytes"]
    per_node = lambda params: [({"topology": topology["name"], "node": node}, dict(params)) for node in nodes]
    cells: list[tuple[dict[str, Any], dict[str, Any]]] = []

    if question_id == "GB10-IDENTITY-001":
        return True, [item for node in nodes for item in [({"topology": topology["name"], "node": node}, {"candidate": "identity", "iterations": 1})]]
    if question_id in {"GB10-MEM-001", "GB10-MEM-002", "GB10-MEM-003"}:
        candidate = {
            "GB10-MEM-001": "bandwidth",
            "GB10-MEM-002": "reuse",
            "GB10-MEM-003": "pointer_chase",
        }[question_id]
        for node in nodes:
            for params in cartesian({
                "candidate": [candidate],
                "working_set_bytes": [4 << 20, 32 << 20, 128 << 20, 512 << 20],
                "iterations": [256],
            }):
                cells.append(({"topology": topology["name"], "node": node}, params))
        return True, cells
    if question_id == "GB10-UMEM-001":
        for node in nodes:
            for params in cartesian({
                "candidate": ["gpu_only", "cpu_read_contention", "cpu_write_contention"],
                "working_set_bytes": [4 << 20, 64 << 20, 512 << 20],
                "iterations": [128],
            }):
                cells.append(({"topology": topology["name"], "node": node}, params))
        return True, cells
    if question_id in {"GB10-MAPPED-001", "GB10-COPY-001"}:
        candidate = "mapped_host" if question_id == "GB10-MAPPED-001" else "copy"
        real_payloads = unique([int(model["pipeline_payload_bytes_per_row"]) * batch for model in models for batch in [1, 8, 64, 256, 1024]])
        real_payloads = [value for value in real_payloads if value <= 64 << 20]
        for node in nodes:
            for params in cartesian({"candidate": [candidate], "payload_bytes": real_payloads, "iterations": [256]}):
                cells.append(({"topology": topology["name"], "node": node}, params))
        return True, cells
    if question_id == "GB10-LAUNCH-001":
        for node in nodes:
            for params in cartesian({
                "mode": ["enqueue", "launch_sync"],
                "load_mode": ["idle", "memory_loaded"],
                "kernel_count": [1, 8, 32, 128],
                "batch_size": [1, 16, 128, 1024],
                "working_set_bytes": [64 << 20],
                "iterations": [1000],
            }):
                cells.append(({"topology": topology["name"], "node": node}, params))
        return True, cells
    if question_id == "GB10-GRAPH-001":
        for node in nodes:
            for params in cartesian({"mode": ["graph"], "candidate": ["direct", "graph"], "kernel_count": [1, 8, 32, 128], "batch_size": batches, "iterations": [512]}):
                cells.append(({"topology": topology["name"], "node": node}, params))
        return True, cells
    if question_id == "GB10-CALLBACK-001":
        for node in nodes:
            for params in cartesian({
                "candidate": ["stream_sync", "event", "host_callback"],
                "load_mode": ["idle", "memory_loaded"],
                "working_set_bytes": [64 << 20],
                "iterations": [512],
            }):
                cells.append(({"topology": topology["name"], "node": node}, params))
        return True, cells
    if question_id == "GB10-CONCURRENCY-001":
        for node in nodes:
            for params in cartesian({
                "candidate": ["copy_copy", "copy_compute", "compute_compute"],
                "stream_count": [1, 2, 4, 8, 16],
                "working_set_bytes": [16 << 20],
                "iterations": [128],
            }):
                cells.append(({"topology": topology["name"], "node": node}, params))
        return True, cells
    if question_id == "GB10-SMEM-001":
        for node in nodes:
            for params in cartesian({"candidate": ["dynamic_shared"], "dynamic_shared_bytes": [0, 16384, 32768, 49152, 65536, 98304, 131072], "iterations": [128]}):
                cells.append(({"topology": topology["name"], "node": node}, params))
        return True, cells
    if question_id == "GB10-ATOMIC-001":
        for node in nodes:
            for params in cartesian({"mode": ["atomic"], "candidate": ["contended", "distributed"], "operations": [1 << 20, 16 << 20], "iterations": [128]}):
                cells.append(({"topology": topology["name"], "node": node}, params))
        return True, cells
    if question_id == "GB10-THERMAL-001":
        for node in nodes:
            cells.append(({
                "topology": topology["name"],
                "node": node,
            }, {
                "candidate": "sustained_memory_copy",
                "sample_phase": "all",
                "working_set_bytes": 512 << 20,
                "sustained_seconds": 900,
                "iterations": 1,
            }))
        return True, cells

    if question_id == "GB10-REG-001":
        for node in nodes:
            for model in models:
                for params in role_cells(model, model["kernel_roles"], batches, [2048]):
                    params.update({"candidate": "production", "kernel_class": "resource", "route_distribution": "none", "iterations": 128})
                    cells.append(({"topology": topology["name"], "node": node}, params))
        return True, cells
    if question_id == "GB10-NATIVE-MMA-001":
        for node in nodes:
            for model in models:
                for params in cartesian({
                    "model_id": [model["id"]], "role": ["expert" if int(model["moe_expert_count"]) else "dense"],
                    "candidate": ["bf16_dequantized", "native_block_scaled", "token_centric"],
                    "kernel_class": ["gemm"], "route_distribution": ["uniform"],
                    "batch_size": batches, "context_tokens": [2048], "iterations": [128],
                }):
                    cells.append(({"topology": topology["name"], "node": node}, params))
        return True, cells
    if question_id == "MODEL-GQA-001":
        attention_models = [model for model in models if "gqa" in model["kernel_roles"]]
        for node in nodes:
            for model in attention_models:
                for params in cartesian({
                    "model_id": [model["id"]], "role": ["attention"],
                    "candidate": ["query_head", "grouped_kv", "split_key_grouped"],
                    "kernel_class": ["gqa"], "route_distribution": ["none"],
                    "batch_size": batches, "context_tokens": contexts, "iterations": [128],
                }):
                    cells.append(({"topology": topology["name"], "node": node}, params))
        return True, cells
    if question_id == "MODEL-MOE-001":
        moe_models = [model for model in models if int(model["moe_expert_count"]) > 0]
        for node in nodes:
            for model in moe_models:
                for params in cartesian({
                    "model_id": [model["id"]], "role": ["expert"],
                    "candidate": ["token_centric", "grouped_tile", "sealed_weight_stationary"],
                    "kernel_class": ["moe"], "route_distribution": ["uniform", "zipf", "hot_expert"],
                    "batch_size": batches, "context_tokens": [2048], "iterations": [128],
                }):
                    cells.append(({"topology": topology["name"], "node": node}, params))
        return True, cells
    if question_id == "MODEL-KDA-001":
        for node in nodes:
            for params in cartesian({
                "model_id": ["kimi_k3_mxfp4_experts_bf16_rest"], "role": ["kda"],
                "candidate": ["token_step", "chunk_replay", "fused_replay"],
                "kernel_class": ["kda"], "route_distribution": ["none"],
                "batch_size": batches, "context_tokens": contexts, "iterations": [128],
            }):
                cells.append(({"topology": topology["name"], "node": node}, params))
        return True, cells

    if question_id in {"NVME-RAW-001", "NVME-GPU-001"}:
        candidate = "direct_io" if question_id == "NVME-RAW-001" else "nvme_to_gpu"
        for node in nodes:
            for params in cartesian({"candidate": [candidate], "block_bytes": [65536, 262144, 1048576, 4194304], "queue_depth": [1, 4, 16, 64], "worker_count": [1, 2, 4], "iterations": [128]}):
                cells.append(({"topology": topology["name"], "node": node}, params))
        return True, cells

    if question_id.startswith("NET-") and question_id not in {"NET-RING-001", "NET-SWITCH-001"}:
        if question_id == "NET-PMTU-001":
            for source, peer in pairs:
                cells.append(({"topology": topology["name"], "node": source, "peer": peer}, {"candidate": "udp_df_binary_search", "minimum_payload_bytes": 512, "maximum_payload_bytes": 65483, "iterations": 5}))
            return True, cells
        base_pairs = pairs
        if question_id == "NET-TCP-001":
            axes = {"candidate": ["tcp"], "payload_bytes": payloads, "lane_count": [1], "window_depth": [1, 4, 16, 64], "cq_batch": [1], "registered_region_count": [1], "progress_mode": ["event_loop"], "iterations": [256]}
        elif question_id == "NET-RDMA-001":
            axes = {"candidate": ["mapped_host", "gpudirect"], "payload_bytes": [256, 14336, 65536, 1048576, 8388608], "lane_count": [1, 2, 4, 8], "window_depth": [1, 4, 16, 64], "cq_batch": [16], "registered_region_count": [128], "progress_mode": ["autonomous"], "iterations": [256]}
        elif question_id == "NET-MR-001":
            axes = {"candidate": ["mapped_host", "gpudirect"], "payload_bytes": [14336], "lane_count": [1], "window_depth": [1], "cq_batch": [16], "registered_region_count": [16, 64, 128, 256, 512], "progress_mode": ["event_loop"], "iterations": [64]}
        elif question_id == "NET-LANES-001":
            axes = {"candidate": ["mapped_host", "gpudirect"], "payload_bytes": [256, 14336, 1048576], "lane_count": [1, 2, 4, 8], "window_depth": [16], "cq_batch": [16], "registered_region_count": [128], "progress_mode": ["autonomous"], "iterations": [256]}
        elif question_id == "NET-CQ-001":
            axes = {"candidate": ["mapped_host", "gpudirect"], "payload_bytes": [14336, 1048576], "lane_count": [4], "window_depth": [16], "cq_batch": [1, 4, 16, 64], "registered_region_count": [128], "progress_mode": ["autonomous"], "iterations": [256]}
        elif question_id == "NET-PROGRESS-001":
            axes = {"candidate": ["mapped_host", "gpudirect"], "payload_bytes": [14336, 1048576], "lane_count": [4], "window_depth": [1, 4, 16, 64], "cq_batch": [16], "registered_region_count": [128], "progress_mode": ["event_loop", "autonomous"], "iterations": [256]}
        else:
            raise ValueError(question_id)
        combinations = cartesian(axes)
        for source, peer in base_pairs:
            for params in combinations:
                cells.append(({"topology": topology["name"], "node": source, "peer": peer}, dict(params)))
        return True, cells

    if question_id in {"NET-RING-001", "NET-SWITCH-001", "TOPO-PP-001", "TOPO-WINDOW-001", "TOPO-PLACEMENT-001"}:
        ring_question = question_id == "NET-RING-001"
        switch_question = question_id == "NET-SWITCH-001"
        applicable = not ((ring_question and topology["mode"] != "ring") or (switch_question and topology["mode"] != "single_switch"))
        if not applicable:
            return False, []
        coordinator = nodes[0]
        pp_degrees = [value for value in workloads["candidate_pipeline_degrees"] if int(value) <= len(nodes)]
        if question_id in {"NET-RING-001", "NET-SWITCH-001"}:
            axes = {"model_id": [model["id"] for model in models], "batch_size": batches, "context_tokens": [2048, 65536, 1048576], "pipeline_degree": pp_degrees, "window_depth": [1, 16, 64], "candidate": [topology["mode"]], "iterations": [32]}
        elif question_id == "TOPO-PP-001":
            for pipeline_degree in pp_degrees:
                axes = {
                    "model_id": [model["id"] for model in models],
                    "batch_size": batches,
                    "context_tokens": [2048, 65536, 1048576],
                    "window_depth": [16],
                    "iterations": [32],
                }
                for parameters in cartesian(axes):
                    parameters["pipeline_degree"] = pipeline_degree
                    parameters["candidate"] = f"pp{pipeline_degree}"
                    cells.append(({"topology": topology["name"], "node": nodes[0]}, parameters))
            return True, cells
        elif question_id == "TOPO-WINDOW-001":
            axes = {"model_id": [model["id"] for model in models], "batch_size": batches, "context_tokens": [2048, 65536], "pipeline_degree": pp_degrees, "window_depth": [1, 4, 16, 64, 256], "candidate": ["stop_and_wait", "selective_ack", "rdma_slots"], "iterations": [32]}
        else:
            axes = {"model_id": [model["id"] for model in models], "batch_size": batches, "context_tokens": [2048, 65536], "pipeline_degree": pp_degrees, "window_depth": [16], "candidate": ["physical_order", "balanced_layers", "bandwidth_weighted"], "iterations": [32]}
        for params in cartesian(axes):
            cells.append(({"topology": topology["name"], "node": coordinator}, params))
        return True, cells
    raise ValueError(f"no axis definition for {question_id}")


def axis_inventory(parameter_sets: list[dict[str, Any]]) -> dict[str, list[Any]]:
    keys = sorted({key for parameters in parameter_sets for key in parameters})
    return {key: unique(parameters[key] for parameters in parameter_sets if key in parameters) for key in keys}


def build_plan(topology_path: pathlib.Path, source_sha256: str) -> dict[str, Any]:
    require(is_sha256(source_sha256), "source package SHA-256 is invalid")
    questions_document = load_json(QUESTIONS_PATH)
    probe_document = load_json(PROBE_PLAN_PATH)
    workloads = load_json(WORKLOADS_PATH)
    topology = topology_projection(load_json(topology_path))
    questions = questions_document["questions"]
    probe_registry = {str(probe["id"]): probe for probe in probe_document["probes"]}
    jobs: list[dict[str, Any]] = []
    coverage: dict[str, Any] = {}
    for question in questions:
        question_id = str(question["id"])
        probe_id = str(question["probe"])
        require(probe_id in probe_registry, f"{question_id}: unknown probe")
        applicable, cells = parameters_for_question(question_id, workloads, topology)
        parameter_sets = [parameters for _, parameters in cells]
        coverage[question_id] = {
            "probe_id": probe_id,
            "applicable": applicable,
            "expected_observation_count": len(cells),
            "axes": axis_inventory(parameter_sets),
        }
        for scope, parameters in cells:
            cell_basis = {
                "question_id": question_id,
                "probe_id": probe_id,
                "scope": scope,
                "parameters": parameters,
            }
            cell_id = sha256_bytes(canonical_json_bytes(cell_basis))
            jobs.append({
                "job_index": len(jobs),
                "cell_id": cell_id,
                "question_id": question_id,
                "probe_id": probe_id,
                "executable": str(probe_registry[probe_id]["executable"]),
                "scope": scope,
                "parameters": parameters,
            })
    plan: dict[str, Any] = {
        "schema_version": 1,
        "plan_kind": "spark_hardware_qualification_plan",
        "source_package_sha256": source_sha256,
        "question_registry_sha256": sha256_file(QUESTIONS_PATH),
        "probe_registry_sha256": sha256_file(PROBE_PLAN_PATH),
        "workload_profiles_sha256": sha256_file(WORKLOADS_PATH),
        "topology_source_sha256": sha256_file(topology_path),
        "topology": topology,
        "coverage": coverage,
        "jobs": jobs,
    }
    plan["plan_id"] = sha256_bytes(canonical_json_bytes(plan))
    return plan


def main() -> int:
    arguments = parse_arguments()
    plan = build_plan(arguments.topology.resolve(), arguments.source_package_sha256)
    write_json_atomic(arguments.output.resolve(), plan)
    print(f"wrote {len(plan['jobs'])} exact hardware cells to {arguments.output.resolve()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
