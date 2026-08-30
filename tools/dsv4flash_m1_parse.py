#!/usr/bin/env python3
"""Parse one timestamped M1 bench run: exact token hash first (the gate),
then the decode rate over the 127 intervals from first to last emitted
token. Usage: m1_parse.py <run.jsonl>"""
import hashlib, json, sys

EXPECTED = "211462f2525f73b76137ee1ce9bd4e015ad8a3fd825a7c45d38fff0488598083"

events = []
for raw in open(sys.argv[1]):
    ts_str, _, payload = raw.partition(" ")
    try:
        events.append((float(ts_str), json.loads(payload)))
    except (ValueError, json.JSONDecodeError):
        continue

tokens = [(ts, e) for ts, e in events if e.get("event") == "TOKEN" and e.get("status") == 0]
tokens.sort(key=lambda p: p[1].get("token_index", 0))
ids = [e["token_id"] for _, e in tokens]

h = hashlib.sha256((",".join(str(t) for t in ids) + "\n").encode()).hexdigest()
verdict = "HASH-OK" if h == EXPECTED else f"HASH-MISMATCH got={h}"
if len(ids) not in (128,):
    verdict = f"HASH-INVALID count={len(ids)} got={h[:16]}"

rate = float("nan")
span = 0.0
if len(tokens) >= 2:
    intervals = len(tokens) - 1
    span = tokens[-1][0] - tokens[0][0]
    if span > 0:
        rate = intervals / span
print(f"tokens={len(ids)} {verdict} decode_tok_s={rate:.4f} span_s={span:.3f}")
