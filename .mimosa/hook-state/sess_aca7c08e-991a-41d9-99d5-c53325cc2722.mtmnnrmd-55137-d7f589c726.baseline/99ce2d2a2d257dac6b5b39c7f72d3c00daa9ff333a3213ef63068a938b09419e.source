#!/usr/bin/env python3
"""Invariants for the recovered DSV4 Flash TP4 x PP4 estimate table."""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, "tools")
sys.path.insert(0, TOOLS)

import dsv4_tp4_pp4_perf_estimate as perf  # noqa: E402


EXPECTED = {
    1: (30.46, 86.62),
    8: (17.47, 464.0),
    64: (6.00, 1356.0),
    256: (2.38, 1949.0),
    1024: (0.60, 1962.0),
}


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> int:
    check(perf.MEASURED is False, "the recovered table must not claim measurement")
    check("estimate" in perf.CLASSIFICATION.lower(), "classification says estimate")
    check(tuple(EXPECTED) == perf.BATCH_SIZES, "all recovered batches are present")
    check(perf.NATIVE_BATCH_SIZES == (1, 8, 1024), "native shape gate is exact")
    check(perf.NODE_COUNT == 16 and perf.PP_STAGE_COUNT == 4,
          "topology is TP4 x PP4")

    for batch, (per_sequence, aggregate) in EXPECTED.items():
        row = perf.estimate_batch(batch)
        check(row["per_sequence_tokens_per_second"] == per_sequence,
              f"B{batch} per-sequence rate drifted")
        check(row["aggregate_tokens_per_second"] == aggregate,
              f"B{batch} aggregate rate drifted")
        check(row["single_request_tokens_per_second"] == per_sequence,
              f"B{batch} single-request alias drifted")
        check(row["filled_pipeline_tokens_per_second"] == aggregate,
              f"B{batch} filled-pipeline alias drifted")
        check(row["admitted_requests"] == 4 * batch,
              f"B{batch} four-microbatch admission drifted")
        check(row["native_launch_shape"] == (batch in (1, 8, 1024)),
              f"B{batch} native-shape classification drifted")
        check(math.isclose(row["per_sequence_step_ms"], 1000.0 / per_sequence),
              f"B{batch} sequence latency is not the reciprocal rate")
        check(math.isclose(row["pipeline_microbatch_interval_ms"],
                           1000.0 * batch / aggregate),
              f"B{batch} aggregate interval is not the reciprocal rate")
        check(math.isclose(row["critical_stage_time_ms"],
                           1000.0 * batch / aggregate),
              f"B{batch} filled throughput is not B / critical-stage time")
        check(row["measured"] is False, f"B{batch} claims measurement")

    try:
        perf.estimate_batch(2)
    except ValueError:
        pass
    else:
        raise AssertionError("unsupported analytical batch B2 was accepted")

    output = subprocess.check_output(
        [sys.executable, os.path.join(TOOLS, "dsv4_tp4_pp4_perf_estimate.py"),
         "--batch", "8", "--json"],
        text=True,
    )
    payload = json.loads(output)
    check(payload["measured"] is False, "JSON report claims measurement")
    check(payload["topology"]["node_count"] == 16, "JSON topology drifted")
    check(payload["ownership_assumptions"]["kv_cache"].startswith("replicated"),
          "JSON lost replicated-KV ownership")
    check("one_rank" in payload["ownership_assumptions"]["final_head"],
          "JSON lost singleton final-head ownership")
    check("host_rdma" in payload["ownership_assumptions"]["pipeline_transport"],
          "JSON lost host-RDMA memory traffic")
    check(len(payload["estimates"]) == 1, "--batch did not select one row")
    check(payload["estimates"][0]["aggregate_tokens_per_second"] == 464.0,
          "JSON B8 aggregate rate drifted")

    print("DSV4 TP4 x PP4 recovered performance estimates hold (not measured)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
