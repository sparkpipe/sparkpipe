#!/usr/bin/env python3
"""Generate the glm5_next M5 exact-32K B1 decode batch json.

Cell shape (matches the family's max_sequence_positions=32768 and the
qwen4_flash B1 exact-32K precedent):
  - one request, batch B1
  - prompt = the 17 COMPSEC prompts' token ids concatenated, cycled, and
    truncated to EXACTLY (32768 - output_budget) ids — real in-vocab text,
    deterministic from the committed fixture
  - output_token_budget = 256  -> total context exactly 32768 positions
  - no stop tokens (the receipt's token hash must cover the full budget)

usage:
  glm5_next_m5_batch.py --fixture qualification/ds4_eval/quality-fixtures-glm5.3-flash.json \
      --out /tmp/glm5-m5-exact32k-b1.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

COMPSEC_IDS = [f"compsec-{i:03d}" for i in range(76, 93)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fixture", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--context", type=int, default=32768,
                    help="max_sequence_positions of the deployment")
    ap.add_argument("--budget", type=int, default=256)
    args = ap.parse_args()

    fixture = json.loads(Path(args.fixture).read_text())
    comp = [c for c in fixture["cases"] if c["id"] in COMPSEC_IDS]
    if len(comp) != 17:
        print(f"FATAL: expected 17 COMPSEC cases, got {len(comp)}", file=sys.stderr)
        return 2

    pool = []
    for c in sorted(comp, key=lambda c: c["id"]):
        pool.extend(c["prompt_token_ids"])

    prompt_len = args.context - args.budget
    reps = prompt_len // len(pool) + 1
    prompt = (pool * reps)[:prompt_len]

    batch = {
        "schema_version": 1,
        "connect_timeout_ms": 120000,
        "request_capacity": 1,
        "max_context_tokens": args.context,
        "max_prefill_rows_per_submission": 64,
        "maximum_messages_per_rank_per_progress": 8,
        "maximum_new_submissions_per_progress": 1,
        "stop_token_ids": [],
        "requests": [{
            "request_id": args.context,
            "sequence_id": args.context,
            "priority": 0,
            "output_token_budget": args.budget,
            "prompt_token_ids": prompt,
        }],
    }
    Path(args.out).write_text(json.dumps(batch))
    print(f"batch: {args.out}")
    print(f"  prompt tokens: {len(prompt)} (pool of {len(pool)} ids from 17 "
          f"COMPSEC prompts, cycled)")
    print(f"  output budget: {args.budget} -> exact {args.context}-position context")
    print(f"  prompt sha256: {hashlib.sha256(json.dumps(prompt).encode()).hexdigest()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
