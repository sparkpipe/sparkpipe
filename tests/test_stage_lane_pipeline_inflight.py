#!/usr/bin/env python3
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import stage_lane  # noqa: E402


def main():
    header = (ROOT / "include/sparkpipe/spark_kv_page_cache.h").read_text()
    depth = int(re.search(r"#define SPARK_KV_LANE_TRANSACTION_QUEUE_DEPTH (\d+)u", header).group(1))
    failures = []
    if stage_lane.kv_lane_queue_depth() != depth:
        failures.append(f"the lane renderer reads queue depth {stage_lane.kv_lane_queue_depth()}, the KV header says {depth}")
    for name, model in stage_lane.MODELS.items():
        stages = stage_lane.WORLD // stage_lane.tp_of(model)
        for sequences in (1, 2, 8):
            got = stage_lane.inflight_submissions(model, sequences)
            want = min(max(sequences, min(stages, depth)), 16)
            if got != want:
                failures.append(f"{name}: {sequences} sequences render {got} submissions in flight, expected {want}")
            if stages > 1 and got < min(stages, depth):
                failures.append(f"{name}: a {stages}-stage pipeline renders {got} submissions in flight, fewer than the {min(stages, depth)} waves its stages can hold")
            if stages == 1 and got != min(sequences, 16):
                failures.append(f"{name}: a single-stage lane renders {got} submissions in flight for {sequences} sequences")
    for line in failures:
        print("FAILED: " + line)
    if failures:
        return 1
    print(f"PASS pipelined lanes keep min(stages, KV queue depth {depth}) waves in flight; single-stage lanes keep one submission per sequence")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
