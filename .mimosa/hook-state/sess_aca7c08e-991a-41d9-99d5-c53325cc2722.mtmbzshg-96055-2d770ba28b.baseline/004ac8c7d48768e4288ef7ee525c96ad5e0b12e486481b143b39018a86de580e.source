#!/usr/bin/env python3

from __future__ import annotations

import argparse
import ipaddress
import json
import os
import pathlib
import sys
from typing import Any

SCRIPT = pathlib.Path(__file__).resolve()
ROOT = SCRIPT.parents[2]
sys.path.insert(0, str(SCRIPT.parent))
from hardware_common import (  # noqa: E402
    canonical_json_bytes,
    load_json,
    require,
    sha256_bytes,
    write_json_atomic,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate one exact Spark hardware-runner configuration per topology node")
    parser.add_argument("--topology", required=True, type=pathlib.Path)
    parser.add_argument("--provider-map", required=True, type=pathlib.Path)
    parser.add_argument("--output-directory", required=True, type=pathlib.Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--receipt-root", required=True, type=pathlib.Path)
    parser.add_argument("--nvme-file", required=True)
    parser.add_argument("--nvme-file-bytes", type=int, default=8 * 1024 * 1024 * 1024)
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--prepare-nvme", action="store_true")
    return parser.parse_args()


def absolute_or_root_relative(value: str) -> pathlib.Path:
    path = pathlib.Path(value)
    return path if path.is_absolute() else (ROOT / path).resolve()


def relative_for_config(value: str, config_directory: pathlib.Path) -> str:
    path = absolute_or_root_relative(value)
    return os.path.relpath(path, config_directory.resolve())


def validate_provider_map(document: Any) -> dict[str, Any]:
    require(isinstance(document, dict), "provider map is not an object")
    require(document.get("schema_version") == 1, "unsupported provider-map schema")
    require(document.get("provider_map_kind") == "spark_hardware_provider_map",
            "invalid provider-map kind")
    executables = document.get("executables")
    providers = document.get("providers")
    require(isinstance(executables, dict), "provider-map executables missing")
    require(isinstance(providers, dict), "provider-map providers missing")
    required_executables = {
        "cuda_characterize",
        "model_kernel_characterize",
        "nvme_characterize",
        "transport_characterize",
        "pmtu_characterize",
        "topology_characterize",
    }
    require(set(executables) == required_executables,
            f"provider-map executable set differs: {set(executables)}")
    for probe_id, value in executables.items():
        require(isinstance(value, str) and value, f"{probe_id}: executable path missing")
    for probe_id in ("model_kernel_characterize", "transport_characterize", "topology_characterize"):
        table = providers.get(probe_id)
        require(isinstance(table, dict) and table, f"{probe_id}: provider table missing")
        for key, value in table.items():
            require(isinstance(key, str) and key, f"{probe_id}: provider key invalid")
            require(isinstance(value, str) and value, f"{probe_id}/{key}: provider path invalid")
    pmtu = document.get("pmtu")
    require(isinstance(pmtu, dict), "provider-map PMTU settings missing")
    template = pmtu.get("remote_command_template")
    require(isinstance(template, list) and template, "PMTU remote command template missing")
    require(all(isinstance(token, str) and token for token in template),
            "PMTU remote command template is malformed")
    require(isinstance(pmtu.get("remote_executable"), str) and pmtu["remote_executable"],
            "PMTU remote executable missing")
    return document


def validate_topology(document: Any) -> dict[str, Any]:
    require(isinstance(document, dict), "topology is not an object")
    topology = document.get("topology")
    nodes = document.get("compute_nodes")
    require(isinstance(topology, dict), "topology header missing")
    require(topology.get("mode") in {"ring", "single_switch"},
            "only ring and one-switch runner configurations are enabled")
    require(isinstance(nodes, list) and nodes, "topology nodes missing")
    sorted_nodes = sorted(nodes, key=lambda item: int(item["rank"]))
    require([int(node["rank"]) for node in sorted_nodes] == list(range(len(sorted_nodes))),
            "topology ranks are not contiguous")
    names = [str(node["name"]) for node in sorted_nodes]
    require(len(names) == len(set(names)), "topology node names are not unique")
    for node in sorted_nodes:
        ports = node.get("ports")
        require(isinstance(ports, list) and ports, f"{node['name']}: ports missing")
        for port in ports:
            require(isinstance(port, dict), f"{node['name']}: malformed port")
            ipaddress.IPv4Address(str(port["ipv4"]))
    return {"topology": topology, "compute_nodes": sorted_nodes}


def ring_peer_address(source: dict[str, Any], peer: dict[str, Any]) -> str | None:
    source_networks = {
        ipaddress.IPv4Network(f"{port['ipv4']}/30", strict=False): str(port["ipv4"])
        for port in source["ports"]
    }
    for peer_port in peer["ports"]:
        peer_address = ipaddress.IPv4Address(str(peer_port["ipv4"]))
        for network in source_networks:
            if peer_address in network:
                return str(peer_address)
    return None


def peer_addresses_for_node(
    mode: str,
    source: dict[str, Any],
    nodes: list[dict[str, Any]],
) -> dict[str, str]:
    result: dict[str, str] = {}
    for peer in nodes:
        if peer["name"] == source["name"]:
            continue
        if mode == "ring":
            address = ring_peer_address(source, peer)
            if address is not None:
                result[str(peer["name"])] = address
        else:
            result[str(peer["name"])] = str(peer["ports"][0]["ipv4"])
    expected = 2 if mode == "ring" and len(nodes) > 2 else len(nodes) - 1
    require(len(result) == expected,
            f"{source['name']}: expected {expected} reachable peers, found {len(result)}")
    return result


def rewrite_paths(table: dict[str, Any], config_directory: pathlib.Path) -> dict[str, Any]:
    rewritten: dict[str, Any] = {}
    for key, value in table.items():
        if isinstance(value, str):
            rewritten[key] = relative_for_config(value, config_directory)
        elif isinstance(value, dict):
            rewritten[key] = rewrite_paths(value, config_directory)
        else:
            rewritten[key] = value
    return rewritten


def build_config(
    topology: dict[str, Any],
    node: dict[str, Any],
    nodes: list[dict[str, Any]],
    provider_map: dict[str, Any],
    output_path: pathlib.Path,
    arguments: argparse.Namespace,
) -> dict[str, Any]:
    output_directory = output_path.parent
    receipt_directory = arguments.receipt_root / str(topology["name"]) / str(node["name"])
    if not receipt_directory.is_absolute():
        receipt_directory = (ROOT / receipt_directory).resolve()
    pmtu = dict(provider_map["pmtu"])
    pmtu.setdefault("port_base", 47000)
    pmtu.setdefault("port_span", 1000)
    pmtu.setdefault("startup_seconds", 10.0)
    pmtu.setdefault("server_idle_timeout_seconds", 60)
    pmtu.setdefault("local_bind_address", "127.0.0.1")
    config: dict[str, Any] = {
        "schema_version": 1,
        "config_kind": "spark_hardware_runner_config",
        "run_id": arguments.run_id,
        "topology": str(topology["name"]),
        "topology_mode": str(topology["mode"]),
        "node": str(node["name"]),
        "rank": int(node["rank"]),
        "receipt_directory": os.path.relpath(receipt_directory, output_directory.resolve()),
        "probe_timeout_seconds": 7200,
        "probe_timeouts_seconds": {
            "cuda_characterize": 3600,
            "model_kernel_characterize": 3600,
            "nvme_characterize": 7200,
            "transport_characterize": 3600,
            "pmtu_characterize": 120,
            "topology_characterize": 14400,
        },
        "executables": rewrite_paths(provider_map["executables"], output_directory),
        "providers": rewrite_paths(provider_map["providers"], output_directory),
        "peer_addresses": peer_addresses_for_node(str(topology["mode"]), node, nodes),
        "nvme": {
            "file_path": arguments.nvme_file,
            "file_bytes": arguments.nvme_file_bytes,
            "prepare": bool(arguments.prepare_nvme),
            "cuda_device": arguments.cuda_device,
        },
        "pmtu": pmtu,
        "production_providers_required": True,
    }
    config["config_sha256"] = sha256_bytes(canonical_json_bytes(config))
    return config


def main() -> int:
    arguments = parse_arguments()
    require(arguments.run_id and len(arguments.run_id) <= 255, "run ID is invalid")
    require(arguments.nvme_file_bytes > 0, "NVMe file size must be positive")
    require(0 <= arguments.cuda_device <= 1024, "CUDA device is invalid")
    topology_document = validate_topology(load_json(arguments.topology))
    topology = topology_document["topology"]
    nodes = topology_document["compute_nodes"]
    provider_map = validate_provider_map(load_json(arguments.provider_map))
    arguments.output_directory.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    for node in nodes:
        output_path = arguments.output_directory / f"{node['name']}.json"
        config = build_config(topology, node, nodes, provider_map, output_path, arguments)
        write_json_atomic(output_path, config)
        entries.append({
            "node": node["name"],
            "rank": int(node["rank"]),
            "path": output_path.name,
            "config_sha256": config["config_sha256"],
        })
    index: dict[str, Any] = {
        "schema_version": 1,
        "index_kind": "spark_hardware_runner_config_index",
        "topology": topology["name"],
        "topology_mode": topology["mode"],
        "run_id": arguments.run_id,
        "node_count": len(entries),
        "entries": entries,
    }
    index["index_sha256"] = sha256_bytes(canonical_json_bytes(index))
    write_json_atomic(arguments.output_directory / "index.json", index)
    print(f"wrote {len(entries)} runner configurations for {topology['name']}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, KeyError, TypeError, json.JSONDecodeError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
