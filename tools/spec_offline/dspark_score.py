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
--fast scores the CUDA-graph drafter (dspark_fast.DsparkFast) instead of the eager reference.
--gate scores both on the same anchors in one process and is the gate the fast drafter must pass before it
serves: it exits 1 unless at least --gate-anchors anchors were scored and the two tokens per round differ by
at most --gate-rate-gap (relative); a greedy run also needs the fast drafter's first draft to equal the
reference's on at least --gate-agreement of them. A sampled run reports that agreement without gating on it:
the keyed noise turns rounding-level logit differences into different draws at near-ties (10-21 recording:
97.6% first-draft agreement at T=1, top-p 0.95, with tokens per round 0.8% apart, and the side that wins
varies by subsample). Random taps cannot gate it: the drafter is chaotic on them, so dspark_fast.compare
only reports times.
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
    parser.add_argument("--gate", action="store_true")
    parser.add_argument("--gate-anchors", type=int, default=1000)
    parser.add_argument("--gate-agreement", type=float, default=0.99)
    parser.add_argument("--gate-rate-gap", type=float, default=0.02)
    parser.add_argument("--out", required=True)
    parser.add_argument("--anchors-out")
    args = parser.parse_args()
    if args.fast and args.gate:
        raise SystemExit("--gate scores both drafters; drop --fast")
    drafter, arms = drafters(args)
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
    scores = {name: Score(drafter.block) for name, _, _ in arms}
    report = {"arm": {"drafter": "+".join(scores), "taps": list(drafter.tap_names), "temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k, "seed": args.seed},
              "requests": []}
    anchor_log = open(args.anchors_out, "w") if args.anchors_out else None
    labels = {name: name + " " if args.gate else "" for name in scores}
    agreed = compared = 0
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
        contexts = {name: drafter.context() for name, _, _ in arms}
        ingested = 0
        lengths = {name: [] for name, _, _ in arms}
        for anchor in anchors:
            first = max(ingested, anchor - drafter.window)
            ingested = anchor
            truth = tokens[anchor + 1: anchor + 1 + drafter.block]
            firsts = []
            for name, ingest, draft in arms:
                if first < anchor:
                    ingest(contexts[name], list(range(start + first, start + anchor)), taps[first:anchor])
                drafts, confidence = draft(contexts[name], tokens[anchor], start + anchor, rule)
                accepted = scores[name].add(drafts, confidence, truth)
                lengths[name].append(accepted)
                firsts.append(drafts[0] if drafts else None)
                if anchor_log:
                    anchor_log.write(json.dumps({"drafter": name, "request": key[1], "position": start + anchor, "accepted": accepted, "drafts": drafts,
                                                 "truth": truth, "confidence": [round(c, 4) for c in confidence]}) + "\n")
            if len(firsts) == 2:
                compared += 1
                agreed += int(firsts[0] == firsts[1])
        for name in scores:
            done = lengths[name]
            report["requests"].append({"request": key[1], "task": names[key], "sequence": key[2], "positions": [start, end], "reply_rows": len(records) - reply[0],
                                       "drafter": name, "anchors": len(done), "tokens_per_round": round(1.0 + sum(done) / max(len(done), 1), 4)})
            print("DSPARK-SCORE %srequest %d %s positions %d-%d reply %d anchors %d tokens/round %.3f"
                  % (labels[name], key[1], names[key] or "-", start, end, len(records) - reply[0], len(done), 1.0 + sum(done) / max(len(done), 1)), flush=True)
    for name, score in scores.items():
        report[name] = score.summary()
        print("DSPARK-SCORE %stotal anchors %d tokens/round %.3f slot acceptance %s" % (labels[name], score.anchors, report[name]["tokens_per_round"], report[name]["slot_acceptance"]))
    status = 0
    if args.gate:
        eager, fast = report["eager"]["tokens_per_round"], report["fast"]["tokens_per_round"]
        gap = abs(fast - eager) / eager
        agreement = agreed / max(compared, 1)
        failed = [reason for reason, bad in (
            (f"{compared} anchors, below {args.gate_anchors}", compared < args.gate_anchors),
            (f"first-draft agreement {agreement:.4f}, below {args.gate_agreement}", rule is None and agreement < args.gate_agreement),
            (f"tokens/round gap {gap:.4f}, above {args.gate_rate_gap}", gap > args.gate_rate_gap)) if bad]
        report["gate"] = {"anchors": compared, "first_draft_agreement": round(agreement, 4), "tokens_per_round_gap": round(gap, 4), "failed": failed}
        print("DSPARK-GATE %s anchors %d first-draft agreement %.4f tokens/round eager %.3f fast %.3f gap %.4f%s"
              % ("FAIL" if failed else "PASS", compared, agreement, eager, fast, gap, "".join("; " + reason for reason in failed)))
        status = 1 if failed else 0
    else:
        report.update(report.pop(next(iter(scores))))
    Path(args.out).write_text(json.dumps(report, indent=1))
    return status


class Score:
    def __init__(self, block):
        self.hits = [0] * block
        self.tries = [0] * block
        self.calibration = defaultdict(lambda: [0, 0])
        self.anchors = self.accepted = 0

    def add(self, drafts, confidence, truth):
        accepted = 0
        for index, token in enumerate(drafts):
            if index >= len(truth):
                break
            self.tries[index] += 1
            hit = token == truth[index]
            bucket = min(9, int(confidence[index] * 10))
            self.calibration[bucket][0] += 1
            self.calibration[bucket][1] += int(hit)
            if not hit:
                break
            self.hits[index] += 1
            accepted += 1
        self.anchors += 1
        self.accepted += accepted
        return accepted

    def summary(self):
        return {"anchors": self.anchors, "tokens_per_round": round(1.0 + self.accepted / max(self.anchors, 1), 4),
                "slot_acceptance": [round(h / t, 4) if t else None for h, t in zip(self.hits, self.tries)],
                "confidence_calibration": {f"{bucket / 10:.1f}": [count, round(hit / count, 4)] for bucket, (count, hit) in sorted(self.calibration.items())}}


def drafters(args):
    if args.fast or args.gate:
        from dspark_fast import DsparkFast
        drafter = DsparkFast(args.checkpoint, keep_reference=args.gate)
        fast = [("fast", drafter.ingest, drafter.draft)]
        if not args.gate:
            return drafter, fast
        return drafter, [("eager", lambda context, positions, taps: DsparkDrafter.ingest(drafter, context, positions, taps),
                          lambda context, token, position, rule: DsparkDrafter.draft(drafter, context, token, position, rule))] + fast
    drafter = DsparkDrafter(args.checkpoint)
    return drafter, [("eager", drafter.ingest, drafter.draft)]

if __name__ == "__main__":
    sys.exit(main())
