#!/usr/bin/env python3
"""Summarize the DSV4 TP4 benchmark-matrix receipts into one table.

For every cell tag (e.g. live-d512b1, shadow-p2048b16) reads warm-discarded
measured runs r1..rN and reports mean over measured runs of:
  decode tok/s        - wrapper aggregate decode throughput (127 intervals, O128 shape;
                        sum across concurrent sequences for B>1)
  ttft_s              - process-start -> first token (client-inclusive upper bound)
  ttfb_s              - accepted event -> first token (transport-pure prefill+first step)
  e2e_s               - process-start -> last terminal event (total completion time)
Also prints per-request tok/s for B>1 (aggregate / B).
"""
import glob
import json
import os
import statistics
import sys

RECEIPT_DIR = "/tmp/dsv4-matrix-receipts"

CTX_BY_TAG = {}
B_BY_TAG = {}


def cell_params(tag):
    # tags look like <stack>-p512b4 / <stack>-d2048b16
    part = tag.split("-")[-1]
    kind = "prefill" if part.startswith("p") else "decode"
    ctx = int(part[1:].split("b")[0])
    b = int(part.split("b")[1])
    return kind, ctx, b


def load_runs(tag):
    runs = []
    for path in sorted(glob.glob(os.path.join(RECEIPT_DIR, tag + "-r*.json"))):
        if path.endswith(".stderr"):
            continue
        try:
            r = json.load(open(path))
        except json.JSONDecodeError:
            continue
        events = r.get("events", [])
        t_acc = next((e["elapsed_seconds"] for e in events if e["event"]["event"] == "accepted"), None)
        t_first = next((e["elapsed_seconds"] for e in events if e["event"]["event"] == "token"), None)
        terminal = [e["elapsed_seconds"] for e in events if e["event"]["event"] in ("completed", "cancelled", "error")]
        runs.append({
            "file": os.path.basename(path),
            "tokens": r["token_count"],
            "decode_tps": r["decode_tokens_per_second"],
            "ttft": r["ttft_seconds"],
            "total": r.get("total_seconds"),
            "ttfb": (t_first - t_acc) if (t_acc is not None and t_first is not None) else None,
            "last_terminal": max(terminal) if terminal else None,
            "n_errors": sum(1 for e in events if e["event"]["event"] == "error"),
        })
    return runs


def main():
    tags = sorted({os.path.basename(p).rsplit("-r", 1)[0]
                   for p in glob.glob(os.path.join(RECEIPT_DIR, "*-r*.json"))
                   if not p.endswith(".stderr")})
    print(f"{'cell':>18} {'B':>3} {'ctx':>5} {'kind':>8} {'dec tok/s':>10} {'/req':>7} "
          f"{'ttft_s':>8} {'ttfb_s':>8} {'e2e_s':>8} {'runs':>4}")
    rows = []
    for tag in tags:
        kind, ctx, b = cell_params(tag)
        runs = load_runs(tag)
        if not runs:
            continue
        dec = statistics.mean(r["decode_tps"] for r in runs)
        ttft = statistics.mean(r["ttft"] for r in runs)
        ttfb = statistics.mean(r["ttfb"] for r in runs if r["ttfb"] is not None)
        e2e = statistics.mean((r["last_terminal"] if r["last_terminal"] else r["total"]) for r in runs)
        print(f"{tag:>18} {b:>3} {ctx:>5} {kind:>8} {dec:>10.2f} {dec/b:>7.2f} "
              f"{ttft:>8.3f} {ttfb:>8.3f} {e2e:>8.3f} {len(runs):>4}")
        rows.append({"cell": tag, "kind": kind, "ctx": ctx, "batch": b,
                     "decode_tok_per_s_mean": round(dec, 3),
                     "per_request_tok_per_s": round(dec / b, 3),
                     "prefill_eff_tok_per_s": round(ctx * b / ttfb, 3) if kind == "prefill" else None,
                     "ttft_s_mean": round(ttft, 4),
                     "accepted_to_first_token_s_mean": round(ttfb, 4),
                     "e2e_total_completion_s_mean": round(e2e, 4),
                     "runs": len(runs)})
    with open(os.path.join(RECEIPT_DIR, "matrix-summary.json"), "w") as f:
        json.dump(rows, f, indent=2)
    print("wrote " + os.path.join(RECEIPT_DIR, "matrix-summary.json"))


if __name__ == "__main__":
    main()
