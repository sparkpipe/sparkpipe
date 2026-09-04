#!/usr/bin/env python3
"""Recovered DeepSeek V4 Flash TP4 x PP4 throughput estimates.

This file preserves the analytical table recovered with the TP4 x PP4
deployment artifacts.  It is deliberately a small provenance-preserving
calculator, not a hardware benchmark and not a claim that the native SM121
kernel has been qualified.  The values below are estimates; replacement with
measurements requires a receipt from the sixteen-node deployment.

The table contains two different rates:

* ``per_sequence_tokens_per_second`` is the inverse of one sequence's decode
  step latency at the stated microbatch size.
* ``aggregate_tokens_per_second`` is the recovered sixteen-node serving-rate
  estimate after TP4 and four-stage pipeline overlap.

The native CUDA route admits B1, B8, and B1024 exactly.  B64 and B256 are kept
because they are part of the recovered analytical table; they do not extend
the native kernel's launch-shape contract.
"""
from __future__ import annotations

import argparse
import json
from typing import Any


CLASSIFICATION = "analytical estimate"
MEASURED = False
TP_WIDTH = 4
PP_STAGE_COUNT = 4
NODE_COUNT = TP_WIDTH * PP_STAGE_COUNT
IN_FLIGHT_MICROBATCHES = PP_STAGE_COUNT
NATIVE_BATCH_SIZES = (1, 8, 1024)
BATCH_SIZES = (1, 8, 64, 256, 1024)

# Recovered values.  Keep these as source data rather than silently fitting a
# new curve through them: the original calibration receipts were not present
# in the recovered archive.
RECOVERED_THROUGHPUT = {
    # batch: (tokens/s/sequence, aggregate tokens/s)
    1: (30.46, 86.62),
    8: (17.47, 464.0),
    64: (6.00, 1356.0),
    256: (2.38, 1949.0),
    1024: (0.60, 1962.0),
}


def estimate_batch(batch_size: int) -> dict[str, Any]:
    """Return one recovered estimate and its directly derived latencies."""
    try:
        per_sequence, aggregate = RECOVERED_THROUGHPUT[batch_size]
    except KeyError as exc:
        supported = ", ".join(str(batch) for batch in BATCH_SIZES)
        raise ValueError(
            f"unsupported analytical batch {batch_size}; expected {supported}"
        ) from exc

    return {
        "batch_size": batch_size,
        "admitted_requests": IN_FLIGHT_MICROBATCHES * batch_size,
        "native_launch_shape": batch_size in NATIVE_BATCH_SIZES,
        "single_request_tokens_per_second": per_sequence,
        "per_sequence_tokens_per_second": per_sequence,
        "filled_pipeline_tokens_per_second": aggregate,
        "aggregate_tokens_per_second": aggregate,
        "single_request_latency_ms": 1000.0 / per_sequence,
        "per_sequence_step_ms": 1000.0 / per_sequence,
        "critical_stage_time_ms": 1000.0 * batch_size / aggregate,
        "pipeline_microbatch_interval_ms": 1000.0 * batch_size / aggregate,
        "classification": CLASSIFICATION,
        "measured": MEASURED,
    }


# A concise alias is useful to callers while retaining the explicit public
# name in documentation and tests.
estimate = estimate_batch


def estimate_table() -> list[dict[str, Any]]:
    """Return every recovered table row in increasing batch order."""
    return [estimate_batch(batch) for batch in BATCH_SIZES]


def report(batch_size: int | None = None) -> dict[str, Any]:
    """Build the machine-readable report emitted by ``--json``."""
    rows = estimate_table() if batch_size is None else [estimate_batch(batch_size)]
    return {
        "classification": CLASSIFICATION,
        "measured": MEASURED,
        "model": "DeepSeek V4 Flash",
        "topology": {
            "tensor_parallel_width": TP_WIDTH,
            "pipeline_stage_count": PP_STAGE_COUNT,
            "node_count": NODE_COUNT,
            "in_flight_microbatches": IN_FLIGHT_MICROBATCHES,
        },
        "ownership_assumptions": {
            "kv_cache": "replicated_across_tp_ranks_within_each_pp_stage",
            "final_head": "resident_on_one_rank_of_the_final_pipeline_stage",
            "fixed_and_fp8_weights": "replicated_unless_explicitly_tp_sharded",
            "pipeline_transport": "host_rdma_memory_traffic_included",
        },
        "rate_formulas": {
            "single_request": "1000 / end_to_end_latency_ms",
            "filled_pipeline": "1000 * batch_size / critical_stage_time_ms",
        },
        "native_batch_sizes": list(NATIVE_BATCH_SIZES),
        "estimates": rows,
    }


def _print_human(batch_size: int | None) -> None:
    print("ESTIMATE ONLY - NOT A MEASUREMENT")
    print("DeepSeek V4 Flash, TP4 x PP4 (16 nodes, four PP microbatches)")
    print(
        "batch  admitted  seq tok/s  aggregate tok/s  "
        "seq step ms  native shape"
    )
    rows = estimate_table() if batch_size is None else [estimate_batch(batch_size)]
    for row in rows:
        print(
            f"{row['batch_size']:>5}  "
            f"{row['admitted_requests']:>8}  "
            f"{row['per_sequence_tokens_per_second']:>9.2f}  "
            f"{row['aggregate_tokens_per_second']:>15.2f}  "
            f"{row['per_sequence_step_ms']:>11.3f}  "
            f"{'yes' if row['native_launch_shape'] else 'no':>12}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batch",
        type=int,
        choices=BATCH_SIZES,
        help="emit one recovered batch row (default: all rows)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit a machine-readable report",
    )
    args = parser.parse_args(argv)

    if args.json:
        print(json.dumps(report(args.batch), indent=2, sort_keys=True))
    else:
        _print_human(args.batch)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
