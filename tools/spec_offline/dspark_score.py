#!/usr/bin/env python3
"""Offline acceptance of DSV4.1 1017's DSpark drafter on tap recordings.

Recordings come from a DSV4.1 stage adapter with taps "o36,o37,o38" (the four-stream means entering
target layers 37-39). For every anchor position a of a reply, the drafter holds the taps of positions
a-128 .. a-1, drafts five tokens from the committed token at a, and is scored against the tokens the
target committed at a+1 .. a+5 with SparkPipe's exact-match verify: greedy drafts for greedy streams,
the verifier's keyed Gumbel noise for sampled ones. Tokens per round = accepted drafts + 1.

--select names directories of <task>.u32 token files (prompt count, total count, prompt and reply tokens)
written by the corpus recorders; only the recorded request whose every input token inside a file matches
it, the one covering most positions per file, is scored, so one recording can hold greedy and sampled
corpora. Verify rows past the committed reply (drafts the target rejected at the end) are dropped.
--fast scores the CUDA-graph drafter (dspark_fast.DsparkFast) instead of the eager reference; both
scores on the same recording must agree before the fast drafter serves.
"""
import argparse
import json
import struct
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from block_drafter import load_rows, requests  # noqa: E402
from dspark_drafter import DsparkDrafter  # noqa: E402


def selected(streams, directories):
    chosen = {}
    for directory in directories:
        for path in sorted(Path(directory).glob("*.u32")):
            data = path.read_bytes()
            prompt, total = struct.unpack_from("<II", data)
            tokens = struct.unpack_from(f"<{total}I", data, 8)
            best, best_inside = None, 0
            for key, item in streams.items():
                positions = item["positions"]
                inside = sum(1 for p in positions if p < total)
                if not any(prompt <= p < total for p in positions) or any(p < total and tokens[p] != record[2] for p, record in positions.items()):
                    continue
                if inside > best_inside:
                    best, best_inside = key, inside
            if best is None:
                print(f"DSPARK-SCORE no recorded request matches {path}", flush=True)
                continue
            chosen[best] = path.stem
            streams[best]["positions"] = {p: record for p, record in streams[best]["positions"].items() if p < total}
    return chosen


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--taps", nargs="+", required=True)
    parser.add_argument("--max-anchors", type=int, default=400)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--select", nargs="+")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--out", required=True)
    parser.add_argument("--anchors-out")
    args = parser.parse_args()
    if args.fast:
        from dspark_fast import DsparkFast
        drafter = DsparkFast(args.checkpoint)
    else:
        drafter = DsparkDrafter(args.checkpoint)
    device = drafter.device
    hello, streams = requests(args.taps, None)
    missing = [name for name in drafter.tap_names if name not in hello["entries"]]
    if missing or hello["hidden"] != drafter.dim:
        raise SystemExit(f"recording carries {hello['entries']} at hidden {hello['hidden']}; DSpark needs {list(drafter.tap_names)} at {drafter.dim}")
    columns = [hello["entries"].index(name) for name in drafter.tap_names]
    names = selected(streams, args.select) if args.select else {key: None for key in streams}
    rule = None
    if args.temperature > 0.0:
        seed = args.seed - (1 << 64) if args.seed >= (1 << 63) else args.seed
        rule = (torch.tensor(seed, device=device), torch.tensor(1.0 / args.temperature, device=device),
                torch.tensor(args.top_p if args.top_p < 1.0 else 0.0, device=device), torch.tensor(args.top_k, device=device))
    slot_hits = [0] * drafter.block
    slot_tries = [0] * drafter.block
    calibration = defaultdict(lambda: [0, 0])
    report = {"arm": {"drafter": "fast" if args.fast else "eager", "taps": list(drafter.tap_names), "temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k, "seed": args.seed},
              "requests": []}
    anchor_log = open(args.anchors_out, "w") if args.anchors_out else None
    total_anchors = total_accepted = 0
    for key, item in sorted(streams.items()):
        if key not in names:
            continue
        end = max(item["positions"]) + 1
        start = end - 1
        while start - 1 in item["positions"]:
            start -= 1
        records = [item["positions"][p] for p in range(start, end)]
        tokens = [record[2] for record in records]
        reply = [index for index, record in enumerate(records) if record[3] & 0x6]
        if not reply:
            continue
        taps = load_rows(hello, records, columns).reshape(len(records), -1)
        anchors = [index for index in range(reply[0], len(records) - 1) if index >= drafter.window or start == 0]
        if len(anchors) > args.max_anchors:
            step = len(anchors) / args.max_anchors
            anchors = [anchors[int(i * step)] for i in range(args.max_anchors)]
        context = drafter.context()
        ingested = 0
        lengths = []
        for anchor in anchors:
            first = max(ingested, anchor - drafter.window)
            if first < anchor:
                drafter.ingest(context, list(range(start + first, start + anchor)), taps[first:anchor])
            ingested = anchor
            drafts, confidence = drafter.draft(context, tokens[anchor], start + anchor, rule)
            truth = tokens[anchor + 1: anchor + 1 + len(drafts)]
            accepted = 0
            for index, token in enumerate(drafts):
                if index >= len(truth):
                    break
                slot_tries[index] += 1
                hit = token == truth[index]
                calibration[min(9, int(confidence[index] * 10))][0] += 1
                calibration[min(9, int(confidence[index] * 10))][1] += int(hit)
                if not hit:
                    break
                slot_hits[index] += 1
                accepted += 1
            lengths.append(accepted)
            if anchor_log:
                anchor_log.write(json.dumps({"request": key[1], "position": start + anchor, "accepted": accepted, "drafts": drafts,
                                             "truth": truth, "confidence": [round(c, 4) for c in confidence]}) + "\n")
        total_anchors += len(lengths)
        total_accepted += sum(lengths)
        report["requests"].append({"request": key[1], "task": names[key], "sequence": key[2], "positions": [start, end], "reply_rows": len(records) - reply[0],
                                   "anchors": len(lengths), "tokens_per_round": round(1.0 + sum(lengths) / max(len(lengths), 1), 4)})
        print("DSPARK-SCORE request %d %s positions %d-%d reply %d anchors %d tokens/round %.3f"
              % (key[1], names[key] or "-", start, end, len(records) - reply[0], len(lengths), 1.0 + sum(lengths) / max(len(lengths), 1)), flush=True)
    report["anchors"] = total_anchors
    report["tokens_per_round"] = round(1.0 + total_accepted / max(total_anchors, 1), 4)
    report["slot_acceptance"] = [round(h / t, 4) if t else None for h, t in zip(slot_hits, slot_tries)]
    report["confidence_calibration"] = {f"{bucket / 10:.1f}": [count, round(hit / count, 4)] for bucket, (count, hit) in sorted(calibration.items())}
    Path(args.out).write_text(json.dumps(report, indent=1))
    print("DSPARK-SCORE total anchors %d tokens/round %.3f slot acceptance %s" % (total_anchors, report["tokens_per_round"], report["slot_acceptance"]))


if __name__ == "__main__":
    main()
