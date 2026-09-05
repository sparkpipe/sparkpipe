#!/usr/bin/env python3
"""Expand one strict model-neutral resident deployment specification."""

from __future__ import annotations

import argparse
import copy
import json
import os
import posixpath
from pathlib import Path
from typing import Any


NODE_LAYOUT_PATH = Path(__file__).with_name("spark_node_layout.json")
ROOT_KEYS = {
    "schema_version",
    "coordinator_rank_index",
    "adapter",
    "driver",
    "transport",
    "runtime_limits",
    "topology",
    "tokenizer",
}
TOKENIZER_KEYS = {"path"}
ADAPTER_KEYS = {"shared_object_path"}
DRIVER_KEYS = {"shared_object_path", "program_name"}
TRANSPORT_KEYS = {"shared_object_path", "mode", "control_port_base"}
RUNTIME_KEYS = {
    "max_inflight_submissions",
    "max_active_sequences",
    "max_input_rows",
    "resident_sequence_capacity",
    "kv_logical_page_capacity",
    "kv_physical_page_capacity",
}
TOPOLOGY_KEYS = {
    "rank_hosts",
    "transport_hosts",
    "stage_indices",
    "runtime_dataset",
    "node_target",
    "adapter_configuration_path_template",
    "kv_backing_dataset",
    "kv_backing_maximum_bytes",
    "control_endpoint",
}
NODE_LAYOUT_KEYS = {"schema_version", "node_root_template", "roots"}
NODE_LAYOUT_ROOT_KEYS = {"sparkdata", "srcdata", "extnvme", "kvcache"}
TCP_ENDPOINT_KEYS = {"kind", "host_template", "port"}
UNIX_ENDPOINT_KEYS = {"kind", "path_template"}
PLACEHOLDERS = ("{host}", "{rank}", "{rank_02}", "{stage}", "{stage_02}")


class DeploymentError(ValueError):
    pass


def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DeploymentError(f"duplicate JSON member {key!r}")
        result[key] = value
    return result


def exact_object(value: Any, keys: set[str], where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise DeploymentError(f"{where} must be an object")
    missing = keys - set(value)
    extra = set(value) - keys
    if missing or extra:
        raise DeploymentError(
            f"{where} members differ: missing={sorted(missing)!r} "
            f"extra={sorted(extra)!r}")
    return value


def text_value(value: Any, where: str) -> str:
    if not isinstance(value, str) or value == "":
        raise DeploymentError(f"{where} must be a nonempty string")
    return value


def integer_value(value: Any, where: str, minimum: int, maximum: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise DeploymentError(f"{where} must be an integer")
    if value < minimum or value > maximum:
        raise DeploymentError(
            f"{where} must be in [{minimum}, {maximum}], got {value}")
    return value


def normalized_path(value: str, absolute: bool, where: str) -> str:
    if (value.startswith("/") != absolute or posixpath.normpath(value) != value
            or value == "." or value.startswith("../")):
        kind = "absolute" if absolute else "relative"
        raise DeploymentError(f"{where} must be a normalized {kind} path")
    return value


def dataset_name(value: Any, where: str) -> str:
    value = normalized_path(text_value(value, where), False, where)
    if "/" in value:
        raise DeploymentError(f"{where} must be one directory name")
    return value


def cache_dataset_name(value: Any, where: str) -> str:
    value = normalized_path(text_value(value, where), False, where)
    parts = value.split("/")
    if len(parts) != 2 or "." not in parts[1]:
        raise DeploymentError(
            f"{where} must be model/topology.format")
    return value


def load_node_layout() -> dict[str, Any]:
    layout = exact_object(load_specification(NODE_LAYOUT_PATH),
                          NODE_LAYOUT_KEYS, "node layout")
    if layout["schema_version"] != 1:
        raise DeploymentError("node layout schema_version must be 1")
    text_value(layout["node_root_template"],
               "node layout node_root_template")
    roots = exact_object(layout["roots"], NODE_LAYOUT_ROOT_KEYS,
                         "node layout roots")
    for name in NODE_LAYOUT_ROOT_KEYS:
        root = normalized_path(text_value(
            roots[name], f"node layout roots.{name}"), False,
            f"node layout roots.{name}")
        if "/" in root:
            raise DeploymentError(
                f"node layout roots.{name} must be one directory name")
    return layout


def node_roots(layout: dict[str, Any], host: str, rank: int,
               stage: int) -> dict[str, str]:
    node_root = normalized_path(render_template(
        layout["node_root_template"], host, rank, stage,
        "node layout node_root_template"), True, "rendered node root")
    return {name: posixpath.join(node_root, relative)
            for name, relative in layout["roots"].items()}


def render_template(template: str, host: str, rank: int, stage: int,
                    where: str) -> str:
    rendered = template
    values = {
        "{host}": host,
        "{rank}": str(rank),
        "{rank_02}": f"{rank:02d}",
        "{stage}": str(stage),
        "{stage_02}": f"{stage:02d}",
    }
    for placeholder in PLACEHOLDERS:
        rendered = rendered.replace(placeholder, values[placeholder])
    if "{" in rendered or "}" in rendered:
        raise DeploymentError(f"{where} contains an unknown placeholder")
    return rendered


def validate_common(specification: dict[str, Any]) -> None:
    exact_object(specification["adapter"], ADAPTER_KEYS, "adapter")
    exact_object(specification["driver"], DRIVER_KEYS, "driver")
    exact_object(specification["transport"], TRANSPORT_KEYS, "transport")
    limits = exact_object(
        specification["runtime_limits"], RUNTIME_KEYS, "runtime_limits")
    normalized_path(text_value(
        specification["adapter"]["shared_object_path"],
        "adapter.shared_object_path"), False, "adapter.shared_object_path")
    normalized_path(text_value(
        specification["driver"]["shared_object_path"],
        "driver.shared_object_path"), False, "driver.shared_object_path")
    text_value(specification["driver"]["program_name"], "driver.program_name")
    normalized_path(text_value(
        specification["transport"]["shared_object_path"],
        "transport.shared_object_path"), False,
        "transport.shared_object_path")
    mode = text_value(specification["transport"]["mode"], "transport.mode")
    if mode not in ("host-rdma", "gpudirect-rdma"):
        raise DeploymentError("transport.mode is not a production RDMA mode")
    integer_value(specification["transport"]["control_port_base"],
                  "transport.control_port_base", 1, 65535)
    inflight = integer_value(limits["max_inflight_submissions"],
                             "runtime_limits.max_inflight_submissions", 1, 65535)
    active = integer_value(limits["max_active_sequences"],
                           "runtime_limits.max_active_sequences", 1, 65535)
    rows = integer_value(limits["max_input_rows"],
                         "runtime_limits.max_input_rows", 1, 65535)
    resident = integer_value(limits["resident_sequence_capacity"],
                             "runtime_limits.resident_sequence_capacity", 1, 65535)
    logical_pages = integer_value(limits["kv_logical_page_capacity"],
                                  "runtime_limits.kv_logical_page_capacity",
                                  0, 4294967295)
    physical_pages = integer_value(limits["kv_physical_page_capacity"],
                                   "runtime_limits.kv_physical_page_capacity",
                                   0, 4294967295)
    if inflight > resident or active > rows or active > resident:
        raise DeploymentError("runtime_limits capacities are inconsistent")
    if ((logical_pages == 0) != (physical_pages == 0) or
            (physical_pages != 0 and
             (physical_pages < active or logical_pages < resident or
              physical_pages > logical_pages))):
        raise DeploymentError("runtime_limits KV page capacities are inconsistent")


def build_endpoint(template: dict[str, Any], host: str, rank: int,
                   stage: int) -> dict[str, Any]:
    kind = text_value(template.get("kind"), "topology.control_endpoint.kind")
    if kind == "tcp":
        exact_object(template, TCP_ENDPOINT_KEYS, "topology.control_endpoint")
        endpoint_host = render_template(text_value(
            template["host_template"],
            "topology.control_endpoint.host_template"), host, rank, stage,
            "topology.control_endpoint.host_template")
        if endpoint_host in ("0.0.0.0", "::", "*"):
            raise DeploymentError("rendered TCP endpoint host is a wildcard")
        port = integer_value(template["port"], "topology.control_endpoint.port",
                             1, 65535)
        return {"kind": "tcp", "host": endpoint_host, "port": port}
    if kind == "unix":
        exact_object(template, UNIX_ENDPOINT_KEYS, "topology.control_endpoint")
        path = render_template(text_value(
            template["path_template"],
            "topology.control_endpoint.path_template"), host, rank, stage,
            "topology.control_endpoint.path_template")
        return {"kind": "unix", "path": normalized_path(
            path, True, "rendered control endpoint path")}
    raise DeploymentError("topology.control_endpoint.kind must be tcp or unix")


def build_deployment(specification: dict[str, Any]) -> dict[str, Any]:
    exact_object(specification, ROOT_KEYS, "deployment specification")
    if specification["schema_version"] != 2:
        raise DeploymentError("schema_version must be 2")
    validate_common(specification)
    layout = load_node_layout()
    topology = exact_object(specification["topology"], TOPOLOGY_KEYS, "topology")
    hosts = topology["rank_hosts"]
    stages = topology["stage_indices"]
    if not isinstance(hosts, list) or not isinstance(stages, list):
        raise DeploymentError("rank_hosts and stage_indices must be arrays")
    if len(hosts) == 0 or len(hosts) > 64 or len(hosts) != len(stages):
        raise DeploymentError("rank_hosts and stage_indices have invalid lengths")
    hosts = [text_value(host, f"topology.rank_hosts[{index}]")
             for index, host in enumerate(hosts)]
    stages = [integer_value(stage, f"topology.stage_indices[{index}]", 0,
                            len(hosts) - 1)
              for index, stage in enumerate(stages)]
    if any(host in ("0.0.0.0", "::", "*") for host in hosts):
        raise DeploymentError("rank hosts must not be wildcard addresses")
    if len(set(hosts)) != len(hosts) or sorted(stages) != list(range(len(hosts))):
        raise DeploymentError("rank hosts must be unique and stages a permutation")
    if specification["transport"]["control_port_base"] > 65535 - len(hosts) + 1:
        raise DeploymentError("transport control port range exceeds 65535")
    transport_hosts = topology.get("transport_hosts")
    if transport_hosts is None:
        transport_hosts = hosts
    else:
        if not isinstance(transport_hosts, list) or len(transport_hosts) != len(hosts):
            raise DeploymentError("topology.transport_hosts must match rank_hosts length")
        transport_hosts = [text_value(entry, f"topology.transport_hosts[{index}]")
                           for index, entry in enumerate(transport_hosts)]
    coordinator = integer_value(
        specification["coordinator_rank_index"], "coordinator_rank_index", 0,
        len(hosts) - 1)
    runtime_dataset = dataset_name(
        topology["runtime_dataset"], "topology.runtime_dataset")
    adapter_template = text_value(
        topology["adapter_configuration_path_template"],
        "topology.adapter_configuration_path_template")
    backing_dataset = topology["kv_backing_dataset"]
    if backing_dataset is not None:
        backing_dataset = cache_dataset_name(
            backing_dataset,"topology.kv_backing_dataset")
    backing_maximum_bytes = integer_value(
        topology["kv_backing_maximum_bytes"],
        "topology.kv_backing_maximum_bytes",0,9223372036854775807)
    if backing_dataset is None and backing_maximum_bytes != 0:
        raise DeploymentError(
            "topology KV backing bytes require a cache dataset")
    if backing_dataset is not None and backing_maximum_bytes == 0:
        raise DeploymentError(
            "topology KV cache dataset requires a finite byte limit")
    node_target = text_value(topology["node_target"], "topology.node_target")
    endpoint_value = topology["control_endpoint"]
    if not isinstance(endpoint_value, dict):
        raise DeploymentError("topology.control_endpoint must be an object")
    endpoint_template = exact_object(
        endpoint_value,
        TCP_ENDPOINT_KEYS if endpoint_value.get("kind") == "tcp"
        else UNIX_ENDPOINT_KEYS,
        "topology.control_endpoint")
    nodes = []
    for rank, host in enumerate(hosts):
        stage = stages[rank]
        roots = node_roots(layout,host,rank,stage)
        runtime_root = posixpath.join(roots["sparkdata"],runtime_dataset)
        adapter_path = normalized_path(render_template(
            adapter_template, host, rank, stage,
            "topology.adapter_configuration_path_template"), False,
            "rendered adapter configuration path")
        backing_path = None if backing_dataset is None else posixpath.join(
            roots["kvcache"],backing_dataset)
        nodes.append({
            "rank_index": rank,
            "stage_index": stage,
            "runtime_root": runtime_root,
            "node_target": node_target,
            "transport_host": transport_hosts[rank],
            "adapter_configuration_path": adapter_path,
            "kv_backing_directory": backing_path,
            "kv_backing_maximum_bytes": backing_maximum_bytes,
            "control_endpoint": build_endpoint(
                endpoint_template, host, rank, stage),
        })
    endpoints = [json.dumps(node["control_endpoint"], sort_keys=True)
                 for node in nodes]
    if len(set(endpoints)) != len(endpoints):
        raise DeploymentError("rendered control endpoints must be unique")
    deployment = {
        "schema_version": 2,
        "coordinator_rank_index": coordinator,
        "adapter": copy.deepcopy(specification["adapter"]),
        "driver": copy.deepcopy(specification["driver"]),
        "transport": copy.deepcopy(specification["transport"]),
        "runtime_limits": copy.deepcopy(specification["runtime_limits"]),
        "nodes": nodes,
    }
    tokenizer = specification.get("tokenizer")
    if tokenizer is not None:
        exact_object(tokenizer, TOKENIZER_KEYS, "tokenizer")
        deployment["tokenizer"] = {
            "path": normalized_path(text_value(
                tokenizer["path"], "tokenizer.path"), True,
                "tokenizer.path"),
        }
    return deployment


def load_specification(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"),
                           object_pairs_hook=unique_object)
    except (OSError, json.JSONDecodeError) as error:
        raise DeploymentError(f"cannot load {path}: {error}") from error
    if not isinstance(value, dict):
        raise DeploymentError("deployment specification root must be an object")
    return value


def render_deployment(deployment: dict[str, Any]) -> str:
    return json.dumps(deployment, indent=2) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--specification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    expected = render_deployment(build_deployment(
        load_specification(arguments.specification)))
    if arguments.check:
        if not arguments.output.is_file():
            raise SystemExit(f"missing generated deployment: {arguments.output}")
        if arguments.output.read_text(encoding="utf-8") != expected:
            raise SystemExit(f"generated deployment is stale: {arguments.output}")
        print(f"deployment matches {arguments.specification}")
        return 0
    temporary = arguments.output.with_name(
        arguments.output.name + f".tmp.{os.getpid()}")
    temporary.write_text(expected, encoding="utf-8")
    os.replace(temporary, arguments.output)
    print(f"wrote {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
