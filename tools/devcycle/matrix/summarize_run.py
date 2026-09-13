#!/usr/bin/env python3
"""Summarize one model_stream_decode_benchmark receipt.
Emits JSON: decode tok/s (wrapper), prefill estimate from events:
first_token_time - accepted_time (excludes process start/connect), plus raw ttft.
"""
import json, sys

def main():
    path = sys.argv[1]
    r = json.load(open(path))
    t_accept = None
    t_first = None
    n_tok = 0
    for ev in r.get("events", []):
        e = ev["event"]
        kind = e.get("event")
        if kind == "accepted" and t_accept is None:
            t_accept = ev["elapsed_seconds"]
        if kind == "token":
            n_tok += 1
            if t_first is None:
                t_first = ev["elapsed_seconds"]
    prefill_s = (t_first - t_accept) if (t_accept is not None and t_first is not None) else None
    out = {
        "receipt": path,
        "token_count": r["token_count"],
        "ttft_seconds": round(r["ttft_seconds"], 4),
        "accepted_to_first_token_s": round(prefill_s, 4) if prefill_s else None,
        "decode_tokens_per_second": round(r["decode_tokens_per_second"], 3),
        "decode_seconds_after_first": round(r["decode_seconds_after_first"], 4),
    }
    print(json.dumps(out))

if __name__ == "__main__":
    main()
