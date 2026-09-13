#!/usr/bin/env python3
"""Canonical qwen3.8-27B spark3 benchmark batch files (2026-08-25).

Lands the previously /tmp-ephemeral cell generators in-repo
(docs/QWEN38_PREFILL_ROWS_FINDINGS.md recommendation 2).

Cells written (spark3 deployment root /home/spark3/sparkdata/qwen38.fp8.tp1):
  o512_seq8.json          - CANONICAL generation cell: output 512,
                            max_prefill_rows_per_submission=8. The ONLY
                            cell whose stream must equal the release pin
                            d7f798801a6e43a6 (24.4-24.5 tok/s, E=5.66,
                            77 rounds). NEVER raise rows here: M>=32
                            prefill rides the native-MMA path which is
                            not tap-stable (acceptance collapses
                            E 5.66 -> ~2.8, stream diverges).
  prefillonly_rows8.json  - prefill-only control (budget=1): 47.9 tok/s.
  prefillonly_rows64.json - PREFILL-ONLY measurement cell (budget=1):
                            ~104 tok/s. Legal ONLY because no decode
                            follows (no drafter taps are consumed).
  o512_gen_rows64.json    - DIAGNOSTIC ONLY: generation at rows=64.
                            Expect bit-exactness LOSS; never quote it.

Usage:
  python3 tools/qwen38_27b_spark3_bench_cells.py [--out-dir /tmp]

Run (on the serving host):
  cd /home/spark3/sparkdata/qwen38.fp8.tp1 && export LD_LIBRARY_PATH=$PWD/lib:$LD_LIBRARY_PATH
  /usr/bin/time -f "WALL %e" bin/sparkpipe_model_batch \
    --deployment config/model_resident.json --runtime-root $PWD --batch <cell>.json
"""
import argparse
import json
import os

# Canonical 128-token real-text prompt (do NOT substitute random tokens:
# drafter acceptance collapses and the pinned streams stop matching -
# see docs/QWEN38_DFLASH2_RUNBOOK.md section 6).
PROMPT = [0, 3476, 477, 18068, 260, 3375, 35312, 3417, 16, 38074, 13254,
          16, 455, 4087, 3287, 2231, 1605, 270, 21361, 8786, 9045, 16,
          128803, 79418, 2317, 566, 8130, 345, 14866, 3312, 2019, 16, 983,
          1142, 469, 1142, 554, 6242, 260, 31191, 603, 19905, 418, 270,
          4031, 2455, 2562, 1167, 1479, 270, 6074, 15398, 344, 10097, 16,
          2052, 270, 15398, 344, 1353, 4521, 538, 260, 2395, 2740, 294,
          18885, 6243, 14, 20430, 418, 270, 19904, 50098, 5898, 1789, 638,
          1341, 294, 6319, 2562, 3737, 603, 25529, 223, 18, 855, 270, 2019,
          344, 7681, 1202, 270, 10844, 22283, 339, 671, 2019, 109029, 260,
          716, 15, 10554, 30347, 112566, 1936, 14327, 436, 304, 270, 489,
          5927, 7104, 339, 9945, 1137, 9854, 69, 201, 223, 19, 28, 1823,
          11006, 334, 30557, 32684, 16617]

CELLS = {
    # name: (rows, budget)
    "o512_seq8.json": (8, 512),
    "prefillonly_rows8.json": (8, 1),
    "prefillonly_rows64.json": (64, 1),
    "o512_gen_rows64.json": (64, 512),
}


def write_cell(path, rows, budget):
    json.dump({
        "schema_version": 1,
        "connect_timeout_ms": 30000,
        "request_capacity": 2,
        "max_context_tokens": 4096,
        "max_prefill_rows_per_submission": rows,
        "maximum_messages_per_rank_per_progress": 8,
        "maximum_new_submissions_per_progress": 2,
        "stop_token_ids": [],
        "requests": [{
            "request_id": 760130,
            "sequence_id": 760130,
            "priority": 0,
            "output_token_budget": budget,
            "prompt_token_ids": PROMPT,
        }],
    }, open(path, "w"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="/tmp")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    for name, (rows, budget) in CELLS.items():
        path = os.path.join(args.out_dir, name)
        write_cell(path, rows, budget)
        print("wrote", path, "rows=%d budget=%d" % (rows, budget))


if __name__ == "__main__":
    main()
