#!/usr/bin/env python3
"""Corrected DeepSeek V4 Flash TP4 single-sequence decode roofline.

All numbers in this module are ANALYTICAL ESTIMATES, not measurements.  The
step time is the sum of three explicit terms:

    bandwidth    model traffic / (node bandwidth * usable fraction)
    collectives  four TP phases per layer * phase latency
    launches     graph islands * graph launch latency

The layer count comes from model_contracts/dsv4_flash.json and the graph
island count from the TP4 stage configuration
(examples/deployments/dsv4_flash_tp4_stage.json), which the serving adapter
validates against the firmware island law.  At one output token per step the
raw rate is the reciprocal of the step time.  The estimate is intentionally
separate from the older recovered TP4 x PP4 throughput table in
``dsv4_tp4_pp4_perf_estimate.py``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "model_contracts" / "dsv4_flash.json"
TP4_STAGE = ROOT / "examples" / "deployments" / "dsv4_flash_tp4_stage.json"

CLASSIFICATION = "analytical estimate"
MEASURED = False

LAYER_COUNT = json.loads(CONTRACT.read_text(encoding="utf-8"))["model"]["layer_count"]
GRAPH_ISLAND_COUNT = json.loads(TP4_STAGE.read_text(encoding="utf-8"))[
    "cuda_graph_count_by_pp_stage"][0]
MODEL_TRAFFIC_GB = 4.312
NODE_BANDWIDTH_GB_PER_SECOND = 273.0
USABLE_BANDWIDTH_FRACTION = 0.65
TP_PHASES_PER_LAYER = 4
TP_PHASE_COUNT = LAYER_COUNT * TP_PHASES_PER_LAYER
TP_PHASE_LATENCY_US = 50.0
GRAPH_LAUNCH_LATENCY_US = 3.0

# Derived values.  Keeping the total as a formula prevents rounded headline
# components from drifting away from the reported throughput.
EFFECTIVE_BANDWIDTH_GB_PER_SECOND = (
    NODE_BANDWIDTH_GB_PER_SECOND * USABLE_BANDWIDTH_FRACTION
)
BANDWIDTH_TIME_MS = (
    MODEL_TRAFFIC_GB / EFFECTIVE_BANDWIDTH_GB_PER_SECOND * 1000.0
)
COLLECTIVE_TIME_MS = TP_PHASE_COUNT * TP_PHASE_LATENCY_US / 1000.0
GRAPH_LAUNCH_TIME_MS = (
    GRAPH_ISLAND_COUNT * GRAPH_LAUNCH_LATENCY_US / 1000.0
)
TOTAL_STEP_TIME_MS = BANDWIDTH_TIME_MS + COLLECTIVE_TIME_MS + GRAPH_LAUNCH_TIME_MS
RAW_TOKENS_PER_SECOND = 1000.0 / TOTAL_STEP_TIME_MS


def estimate() -> dict[str, Any]:
    """Return the corrected roofline and every term used to derive it."""
    return {
        "classification": CLASSIFICATION,
        "measured": MEASURED,
        "model": "DeepSeek V4 Flash",
        "topology": "TP4",
        "batch_size": 1,
        "critical_rank_has_singleton_head": True,
        "kv_replication_factor_within_tp": 4,
        "layer_count": LAYER_COUNT,
        "model_traffic_gb": MODEL_TRAFFIC_GB,
        "node_bandwidth_gb_per_second": NODE_BANDWIDTH_GB_PER_SECOND,
        "usable_bandwidth_fraction": USABLE_BANDWIDTH_FRACTION,
        "effective_bandwidth_gb_per_second": EFFECTIVE_BANDWIDTH_GB_PER_SECOND,
        "bandwidth_time_ms": BANDWIDTH_TIME_MS,
        "tp_phases_per_layer": TP_PHASES_PER_LAYER,
        "tp_phase_count": TP_PHASE_COUNT,
        "tp_phase_latency_us": TP_PHASE_LATENCY_US,
        "collective_time_ms": COLLECTIVE_TIME_MS,
        "graph_island_count": GRAPH_ISLAND_COUNT,
        "graph_launch_latency_us": GRAPH_LAUNCH_LATENCY_US,
        "graph_launch_time_ms": GRAPH_LAUNCH_TIME_MS,
        "total_step_time_ms": TOTAL_STEP_TIME_MS,
        "raw_tokens_per_second": RAW_TOKENS_PER_SECOND,
    }


def _print_human() -> None:
    row = estimate()
    print("ESTIMATE ONLY - NOT A MEASUREMENT")
    print("DeepSeek V4 Flash TP4, B1 corrected decode roofline")
    print(f"model traffic:       {row['model_traffic_gb']:.3f} GB/step")
    print(
        "effective bandwidth: "
        f"{row['effective_bandwidth_gb_per_second']:.3f} GB/s"
    )
    print(f"bandwidth term:      {row['bandwidth_time_ms']:.3f} ms")
    print(f"collective term:     {row['collective_time_ms']:.3f} ms")
    print(f"graph-launch term:   {row['graph_launch_time_ms']:.3f} ms")
    print(f"total step:          {row['total_step_time_ms']:.3f} ms")
    print(f"raw throughput:      {row['raw_tokens_per_second']:.2f} tok/s")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit the estimate as JSON",
    )
    args = parser.parse_args(argv)
    if args.json:
        print(json.dumps(estimate(), indent=2, sort_keys=True))
    else:
        _print_human()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
