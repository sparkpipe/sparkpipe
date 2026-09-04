#!/usr/bin/env python3
"""Wrap sparkpipe_model_batch: stamp each stdout JSON event with arrival time,
emit a receipt like the qualification ones (tok/s from first->last token)."""
import json, subprocess, sys, time, hashlib, statistics
cmd = sys.argv[1:]
t0 = time.monotonic()
proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
events = []
for line in proc.stdout:
    line = line.strip()
    if not line: continue
    events.append((time.monotonic() - t0, line))
stderr = proc.stderr.read()
rc = proc.wait()
toks = []
for ts, line in events:
    try: o = json.loads(line)
    except Exception: continue
    if o.get("event") == "token":
        toks.append((ts, o))
receipt = {"command": cmd, "process_status": rc, "stderr_tail": stderr[-500:]}
if toks:
    order = [o for _, o in sorted(toks, key=lambda x: x[1]["token_index"])]
    ids = [o["token_id"] for o in order]
    csv = ",".join(str(i) for i in ids) + "\n"
    stamps = [ts for ts, _ in sorted(toks, key=lambda x: x[1]["token_index"])]
    intervals = [b - a for a, b in zip(stamps, stamps[1:])]
    receipt.update({
        "token_count": len(ids),
        "ttft_seconds": stamps[0],
        "total_seconds": stamps[-1],
        "decode_seconds_after_first": stamps[-1] - stamps[0],
        "timed_intervals": len(intervals),
        "decode_tokens_per_second": len(intervals) / (stamps[-1] - stamps[0]) if len(intervals) else None,
        "inter_token_median_seconds": statistics.median(intervals) if intervals else None,
        "inter_token_p95_seconds": sorted(intervals)[int(0.95 * len(intervals))] if intervals else None,
        "token_csv_newline_sha256": hashlib.sha256(csv.encode()).hexdigest(),
        "token_ids": ids,
    })
print(json.dumps(receipt, indent=1))
