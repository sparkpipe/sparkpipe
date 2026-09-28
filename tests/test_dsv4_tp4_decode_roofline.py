#!/usr/bin/env python3
"""Arithmetic and provenance gate for the corrected DSV4 TP4 roofline."""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, "tools")
sys.path.insert(0, TOOLS)

import dsv4_tp4_decode_roofline as roofline  # noqa: E402


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def load(*parts: str) -> dict:
    with open(os.path.join(ROOT, *parts), encoding="utf-8") as handle:
        return json.load(handle)


def main() -> int:
    layers = load("model_contracts", "dsv4_flash.json")["model"]["layer_count"]
    stage = load("examples", "deployments", "dsv4_flash_tp4_stage.json")
    row = roofline.estimate()
    check(row["measured"] is False, "roofline must not claim measurement")
    check("estimate" in row["classification"].lower(),
          "roofline classification says estimate")
    check(row["critical_rank_has_singleton_head"] is True,
          "critical-rank singleton head was dropped")
    check(row["kv_replication_factor_within_tp"] == 4,
          "TP4 KV replication was divided away")
    check(row["layer_count"] == layers,
          "layer count must come from the DSV4 Flash contract")
    check(row["graph_island_count"] == stage["cuda_graph_count_by_pp_stage"][0],
          "graph-island count must come from the TP4 stage configuration")
    check(row["tp_phase_count"] == row["tp_phases_per_layer"] * layers,
          "TP phase count must be phases per layer times layers")
    check(0.0 < row["usable_bandwidth_fraction"] <= 1.0,
          "usable bandwidth fraction must be a fraction")
    for name in ("model_traffic_gb", "node_bandwidth_gb_per_second",
                 "tp_phase_latency_us", "graph_launch_latency_us"):
        check(row[name] > 0.0, f"{name} must be positive")
    check(math.isclose(
        row["effective_bandwidth_gb_per_second"],
        row["node_bandwidth_gb_per_second"] * row["usable_bandwidth_fraction"],
        rel_tol=1e-12,
    ), "effective bandwidth does not derive from bandwidth/efficiency")
    check(math.isclose(
        row["bandwidth_time_ms"],
        row["model_traffic_gb"] / row["effective_bandwidth_gb_per_second"] * 1000.0,
        rel_tol=1e-12,
    ), "bandwidth term does not derive from traffic/effective bandwidth")
    check(math.isclose(
        row["collective_time_ms"],
        row["tp_phase_count"] * row["tp_phase_latency_us"] / 1000.0,
        rel_tol=1e-12,
    ), "collective term does not derive from phases/latency")
    check(math.isclose(
        row["graph_launch_time_ms"],
        row["graph_island_count"] * row["graph_launch_latency_us"] / 1000.0,
        rel_tol=1e-12,
    ), "graph-launch term does not derive from islands/latency")
    check(math.isclose(
        row["total_step_time_ms"],
        row["bandwidth_time_ms"] + row["collective_time_ms"]
        + row["graph_launch_time_ms"],
        rel_tol=1e-12,
    ), "total step time is not the sum of its terms")
    check(math.isclose(row["raw_tokens_per_second"],
                       1000.0 / row["total_step_time_ms"], rel_tol=1e-12),
          "raw throughput is not the reciprocal corrected step time")

    output = subprocess.check_output(
        [sys.executable, os.path.join(TOOLS, "dsv4_tp4_decode_roofline.py"),
         "--json"],
        text=True,
    )
    payload = json.loads(output)
    check(payload == json.loads(json.dumps(row)),
          "JSON roofline differs from the module estimate")

    print("DSV4 TP4 corrected decode roofline holds (estimate, not measured)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
