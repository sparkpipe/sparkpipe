#!/usr/bin/env python3
"""Run one Qwen3.8-27B TP1 benchmark case through sparkpipe_model_batch on the serving node.

Cases: o128 and o512 (a chat-templated prose essay prompt), legacy512 (the
08-28 canonical 128-token id list, which is not text under the Qwen tokenizer),
code512 and rep512 (chat-templated code and repetitive prompts), ttft (a 427-token chat prompt,
8 new tokens) and streams8 (8 concurrent chat requests, 256 new tokens each).
The residentd must be fresh: it serves one client connection.

Prints one JSON line: tokens, time to first token from process launch, decode
tok/s from the first to the last token event, end-to-end tok/s, and a stream hash.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time

CANONICAL = [0, 3476, 477, 18068, 260, 3375, 35312, 3417, 16, 38074, 13254, 16, 455, 4087, 3287, 2231, 1605, 270, 21361, 8786, 9045, 16, 128803, 79418, 2317, 566, 8130, 345, 14866, 3312, 2019, 16, 983, 1142, 469, 1142, 554, 6242, 260, 31191, 603, 19905, 418, 270, 4031, 2455, 2562, 1167, 1479, 270, 6074, 15398, 344, 10097, 16, 2052, 270, 15398, 344, 1353, 4521, 538, 260, 2395, 2740, 294, 18885, 6243, 14, 20430, 418, 270, 19904, 50098, 5898, 1789, 638, 1341, 294, 6319, 2562, 3737, 603, 25529, 223, 18, 855, 270, 2019, 344, 7681, 1202, 270, 10844, 22283, 339, 671, 2019, 109029, 260, 716, 15, 10554, 30347, 112566, 1936, 14327, 436, 304, 270, 489, 5927, 7104, 339, 9945, 1137, 9854, 69, 201, 223, 19, 28, 1823, 11006, 334, 30557, 32684, 16617]


def requests_for(case: str, prompts: dict) -> list[tuple[list[int], int]]:
    if case == "o128":
        return [(prompts["prose"], 128)]
    if case == "o512":
        return [(prompts["prose"], 512)]
    if case == "legacy512":
        return [(CANONICAL, 512)]
    if case == "code512":
        return [(prompts["code"], 512)]
    if case == "rep512":
        return [(prompts["repetitive"], 512)]
    if case == "ttft":
        return [(prompts["ttft350"], 8)]
    if case == "streams8":
        return [(p, 256) for p in prompts["streams"]]
    raise SystemExit(f"unknown case {case}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("case")
    ap.add_argument("--root", required=True)
    ap.add_argument("--deployment", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--prefill-rows", type=int, required=True)
    args = ap.parse_args()
    reqs = requests_for(args.case, json.load(open(args.prompts)))
    batch = {"schema_version": 1, "connect_timeout_ms": 30000, "request_capacity": max(2, len(reqs)),
             "max_context_tokens": 4096, "max_prefill_rows_per_submission": args.prefill_rows,
             "maximum_messages_per_rank_per_progress": 8, "maximum_new_submissions_per_progress": max(2, len(reqs)),
             "stop_token_ids": [],
             "requests": [{"request_id": 7000 + i, "sequence_id": 7000 + i, "priority": 0,
                           "output_token_budget": budget, "prompt_token_ids": prompt}
                          for i, (prompt, budget) in enumerate(reqs)]}
    path = os.path.join(args.root, "logs", f"bench-{args.case}.json")
    json.dump(batch, open(path, "w"))
    env = dict(os.environ, LD_LIBRARY_PATH=os.path.join(args.root, "lib"))
    launch_ns = time.monotonic_ns()
    run = subprocess.run([os.path.join(args.root, "bin", "sparkpipe_model_batch"), "--deployment", args.deployment,
                          "--runtime-root", args.root, "--batch", path], capture_output=True, text=True, env=env, cwd=args.root)
    end_ns = time.monotonic_ns()
    events = [json.loads(line) for line in run.stdout.splitlines() if line.startswith("{")]
    tokens = [e for e in events if e.get("event") == "token"]
    by_request: dict[int, list[dict]] = {}
    for e in tokens:
        by_request.setdefault(e["request_id"], []).append(e)
    per_request = []
    for rid in sorted(by_request):
        ev = by_request[rid]
        first, last = ev[0]["monotonic_ns"], ev[-1]["monotonic_ns"]
        per_request.append({"request_id": rid, "tokens": len(ev), "ttft_ms": round((first - launch_ns) / 1e6, 1),
                            "decode_tok_s": round((len(ev) - 1) / ((last - first) / 1e9), 2) if len(ev) > 1 and last > first else None,
                            "stream16": hashlib.sha256(str([x["token_id"] for x in ev]).encode()).hexdigest()[:16]})
    json.dump({str(rid): [x["token_id"] for x in by_request[rid]] for rid in sorted(by_request)},
              open(os.path.join(args.root, "logs", f"bench-{args.case}-tokens.json"), "w"))
    total = sum(r["tokens"] for r in per_request)
    first_all = min((e["monotonic_ns"] for e in tokens), default=0)
    last_all = max((e["monotonic_ns"] for e in tokens), default=0)
    out = {"case": args.case, "deployment": os.path.basename(args.deployment), "batch_rc": run.returncode,
           "requests": len(reqs), "tokens": total, "wall_s": round((end_ns - launch_ns) / 1e9, 3),
           "e2e_tok_s": round(total / ((end_ns - launch_ns) / 1e9), 2),
           "aggregate_decode_tok_s": round((total - len(per_request)) / ((last_all - first_all) / 1e9), 2) if last_all > first_all else None,
           "per_request": per_request, "stderr_tail": run.stderr.strip().splitlines()[-1:]}
    print(json.dumps(out))
    return 0 if run.returncode == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
