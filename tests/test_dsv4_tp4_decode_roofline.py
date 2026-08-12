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


def main() -> int:
    row = roofline.estimate()
    check(row["measured"] is False, "roofline must not claim measurement")
    check("estimate" in row["classification"].lower(),
          "roofline classification says estimate")
    check(row["critical_rank_has_singleton_head"] is True,
          "critical-rank singleton head was dropped")
    check(row["kv_replication_factor_within_tp"] == 4,
          "TP4 KV replication was divided away")
    check(row["model_traffic_gb"] == 4.312, "traffic must remain 4.312 GB")
    check(row["node_bandwidth_gb_per_second"] == 273.0,
          "per-Spark bandwidth must remain 273 GB/s")
    check(row["usable_bandwidth_fraction"] == 0.65,
          "conservative bandwidth efficiency must remain 65 percent")
    check(math.isclose(
        row["effective_bandwidth_gb_per_second"],
        273.0 * 0.65,
        rel_tol=1e-12,
    ), "effective bandwidth does not derive from bandwidth/efficiency")
    check(round(row["bandwidth_time_ms"], 1) == 24.3,
          "bandwidth term must round to 24.3 ms")
    check(row["tp_phase_count"] == 172,
          "43 layers must retain four TP phases each")
    check(row["tp_phase_latency_us"] == 50.0,
          "conservative TP phase latency must remain 50 us")
    check(row["collective_time_ms"] == 8.600,
          "collective term must remain 8.6 ms")
    check(row["graph_island_count"] == 87,
          "executor graph-island count must remain 87")
    check(row["graph_launch_latency_us"] == 3.0,
          "estimated graph launch latency must remain 3 us")
    check(row["graph_launch_time_ms"] == 0.261,
          "87 estimated graph launches must remain 0.261 ms")
    check(round(row["total_step_time_ms"], 3) == 33.161,
          "corrected total must round to 33.161 ms")
    check(math.isclose(row["raw_tokens_per_second"],
                       1000.0 / row["total_step_time_ms"], rel_tol=1e-12),
          "raw throughput is not the reciprocal corrected step time")
    check(round(row["raw_tokens_per_second"], 2) == 30.16,
          "headline raw throughput must remain 30.16 tok/s")

    output = subprocess.check_output(
        [sys.executable, os.path.join(TOOLS, "dsv4_tp4_decode_roofline.py"),
         "--json"],
        text=True,
    )
    payload = json.loads(output)
    check(payload["measured"] is False, "JSON roofline claims measurement")
    check(round(payload["total_step_time_ms"], 3) == 33.161,
          "JSON corrected rounded total drifted")

    print("DSV4 TP4 corrected decode roofline holds (estimate, not measured)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
