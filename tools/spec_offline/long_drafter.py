#!/usr/bin/env python3
"""Long-context block-drafter acceptance by context window.

Reads a tap stream (tapstream.py), builds the drafter's context keys and values
in chunks from host-resident taps, then for anchors spread over each reply
drafts a block with the context limited to the last W positions before the
anchor (W = 0 means all) and scores it against the committed tokens.
For sampled streams the committed token at each position is the one sampled
verify would keep, so the score is the sampled-verify acceptance.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from block_drafter import BlockDrafter, load_rows, requests  # noqa: E402


def context_chunks(drafter, hello, records, columns, start, device, chunk):
    keys = [[] for _ in drafter.layers]
    values = [[] for _ in drafter.layers]
    for first in range(0, len(records), chunk):
        part = records[first:first + chunk]
        taps = load_rows(hello, part, columns).to(device)
        pos = torch.arange(start + first, start + first + len(part), device=device)
        with torch.no_grad():
            k, v = drafter.context(taps.reshape(len(part), -1), pos)
        for index in range(len(drafter.layers)):
            keys[index].append(k[index])
            values[index].append(v[index])
        del taps
    return [torch.cat(k, 0) for k in keys], [torch.cat(v, 0) for v in values]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--drafter", required=True)
    parser.add_argument("--target-shard", required=True)
    parser.add_argument("--embed", default="language_model.model.embed_tokens.weight")
    parser.add_argument("--head", default="language_model.lm_head.weight")
    parser.add_argument("--norm", default="language_model.model.norm.weight")
    parser.add_argument("--taps", nargs="+", required=True)
    parser.add_argument("--windows", default="0,32768,8192,2048")
    parser.add_argument("--max-anchors", type=int, default=600)
    parser.add_argument("--min-positions", type=int, default=1)
    parser.add_argument("--chunk", type=int, default=4096)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    device = "cuda"
    windows = [int(w) for w in args.windows.split(",")]
    drafter = BlockDrafter(args.drafter, args.target_shard, {"embed": args.embed, "head": args.head, "norm": args.norm}, device)
    hello, streams = requests(args.taps, None)
    wanted = [f"o{layer}" for layer in drafter.layer_ids]
    missing = [name for name in wanted if name not in hello["entries"]]
    if missing:
        raise SystemExit(f"tap stream lacks {missing}; it has {hello['entries']}")
    columns = [hello["entries"].index(name) for name in wanted]
    slots = list(range(drafter.block))
    report = {"windows": windows, "requests": []}
    totals = {w: [0, 0] for w in windows}
    slot_hits = {w: [0] * len(slots) for w in windows}
    slot_tries = {w: [0] * len(slots) for w in windows}
    calibration = {w: defaultdict(lambda: [0, 0]) for w in windows}
    for key, item in sorted(streams.items()):
        end = max(item["positions"]) + 1
        start = end - 1
        while start - 1 in item["positions"]:
            start -= 1
        if start != 0 or end < args.min_positions:
            continue
        records = [item["positions"][p] for p in range(start, end)]
        tokens = torch.tensor([record[2] for record in records], device=device)
        flags = [record[3] for record in records]
        reply = [index for index, flag in enumerate(flags) if flag & 0x6]
        if not reply:
            continue
        keys, values = context_chunks(drafter, hello, records, columns, start, device, args.chunk)
        first = reply[0]
        anchors = list(range(first, len(records) - 1))
        if len(anchors) > args.max_anchors:
            step = len(anchors) / args.max_anchors
            anchors = [anchors[int(i * step)] for i in range(args.max_anchors)]
        row = {"request_id": key[1], "positions": len(records), "reply": len(records) - first, "anchors": len(anchors), "windows": {}}
        for w in windows:
            lengths = []
            for anchor in anchors:
                begin = 0 if w == 0 else max(0, anchor - w)
                with torch.no_grad():
                    y, base = drafter.draft_range(int(tokens[anchor]), start + anchor, keys, values, begin, anchor)
                    drafts, confidence = drafter.sequential(y, base, int(tokens[anchor]), slots)
                truth = tokens[anchor + 1: anchor + 1 + len(drafts)].tolist()
                accepted = 0
                for index, token in enumerate(drafts):
                    if index >= len(truth):
                        break
                    slot_tries[w][index] += 1
                    hit = token == truth[index]
                    if confidence:
                        bucket = calibration[w][min(9, int(confidence[index] * 10))]
                        bucket[0] += 1
                        bucket[1] += int(hit)
                    if not hit:
                        break
                    slot_hits[w][index] += 1
                    accepted += 1
                lengths.append(accepted)
            totals[w][0] += len(lengths)
            totals[w][1] += sum(lengths)
            row["windows"][str(w)] = {"tokens_per_round": 1 + sum(lengths) / max(1, len(lengths)),
                                      "histogram": np.bincount(lengths, minlength=len(slots) + 1).tolist()}
            print(f"request {key[1]} positions {len(records)} reply {len(records) - first} window {w} anchors {len(lengths)} "
                  f"tokens/round {1 + sum(lengths) / max(1, len(lengths)):.3f}", flush=True)
        report["requests"].append(row)
        del keys, values
        torch.cuda.empty_cache()
    report["tokens_per_round"] = {str(w): 1 + t[1] / max(1, t[0]) for w, t in totals.items()}
    report["slot_conditional"] = {str(w): [h / t if t else None for h, t in zip(slot_hits[w], slot_tries[w])] for w in windows}
    report["confidence_bins"] = {str(w): {str(b): {"n": v[0], "accepted": v[1] / v[0] if v[0] else None} for b, v in sorted(c.items())}
                                 for w, c in calibration.items()}
    Path(args.out).write_text(json.dumps(report, indent=1))
    print(json.dumps({"tokens_per_round": report["tokens_per_round"], "slot_conditional": report["slot_conditional"]}))


if __name__ == "__main__":
    main()
