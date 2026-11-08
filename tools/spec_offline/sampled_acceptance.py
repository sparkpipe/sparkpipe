#!/usr/bin/env python3
"""Tokens per round of the three ways to verify a sampled draft, on a tap recording.

For every anchor of a reply the drafter's distribution q at each slot follows the recorded tokens, and the target's
distribution p comes from the recorded final tap and the target head at the same row. Reported per slot and as
1 + a1 + a1 a2 + ...:

keyed: SparkPipe's verify. Both sides draw with the same Gumbel noise; the draft stays if the two draws are equal.
rejection: speculative sampling. The draft is drawn from q and kept with probability min(1, p/q), so a slot keeps
1 - TV(p, q) and a refused slot resamples from the leftover of p. The output is distributed exactly as p, but it is not
the stream a seed gives without speculation.
greedy: the draft is q's top token and the verify draws from p.

--taus scales q's temperature, which a rejection verify may do freely. --top-k keeps the k most likely tokens of p and
of q before the nucleus (0: no limit).

--replay N also replays N rejection-verify trajectories along each recorded reply, as a speculative decode takes them:
the first round starts at the reply's first token, a slot is kept with probability min(1, q(x)/p(x)) of the recorded
token x (the chance that a verify which committed x kept the draft rather than resampled it), the first refusal ends the
round, and the next round starts at the last committed token. Since a speculative-sampling decode commits a stream
distributed as p, of which the recording is one draw, tokens per round over the trajectories is the decode's; the
spread anchors above weight every position alike instead.
"""
import argparse
import random
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from block_drafter import load_rows, open_drafter, requests, rms  # noqa: E402

GUMBEL_CANDIDATES = 512
GUMBEL_DRAWS = 64


def nucleus(logits, inverse_temperature, top_p, top_k=0):
    scaled = logits * inverse_temperature
    if top_k > 0:
        cut = torch.topk(scaled, min(top_k, scaled.shape[-1])).values[-1]
        scaled = torch.where(scaled >= cut, scaled, torch.full_like(scaled, float("-inf")))
    probabilities = torch.softmax(scaled, -1)
    if top_p >= 1.0:
        return probabilities
    values, order = torch.sort(probabilities, descending=True)
    before = torch.cumsum(values, 0) - values
    keep = before < top_p
    keep[0] = True
    masked = torch.zeros_like(probabilities)
    masked[order[keep]] = values[keep]
    return masked / masked.sum()


def keyed_match(p, q):
    candidates = torch.unique(torch.cat((torch.topk(p, GUMBEL_CANDIDATES).indices, torch.topk(q, GUMBEL_CANDIDATES).indices)))
    uniform = torch.rand(GUMBEL_DRAWS, candidates.numel(), device=p.device).clamp(1e-9, 1.0 - 1e-9)
    noise = -torch.log(-torch.log(uniform))
    target = torch.argmax(torch.log(p[candidates].clamp_min(1e-30)) + noise, -1)
    draft = torch.argmax(torch.log(q[candidates].clamp_min(1e-30)) + noise, -1)
    return float((target == draft).float().mean())


class Chain:
    def __init__(self, block):
        self.block = block
        self.slot_sum = [0.0] * block
        self.slot_count = [0] * block
        self.tokens = 0.0
        self.anchors = 0

    def add(self, rates):
        reach, total = 1.0, 1.0
        for slot, rate in enumerate(rates):
            self.slot_sum[slot] += rate
            self.slot_count[slot] += 1
            reach *= rate
            total += reach
        self.tokens += total
        self.anchors += 1

    def summary(self):
        return {"tokens_per_round": round(self.tokens / max(self.anchors, 1), 3),
                "slot": [round(a / b, 3) if b else None for a, b in zip(self.slot_sum, self.slot_count)]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--drafter", required=True)
    parser.add_argument("--target-shard", required=True)
    parser.add_argument("--embed", default="language_model.model.embed_tokens.weight")
    parser.add_argument("--head", default="language_model.lm_head.weight")
    parser.add_argument("--norm", default="language_model.model.norm.weight")
    parser.add_argument("--taps", nargs="+", required=True)
    parser.add_argument("--delta", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--replay", type=int, default=0, help="rejection-verify trajectories per request (module docstring)")
    parser.add_argument("--replay-seed", type=int, default=1)
    parser.add_argument("--taus", type=float, nargs="+", default=[1.0])
    parser.add_argument("--max-anchors", type=int, default=200)
    parser.add_argument("--min-reply", type=int, default=64)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    device = "cuda"
    torch.manual_seed(1)
    drafter = open_drafter(args.drafter, args.target_shard, {"embed": args.embed, "head": args.head, "norm": args.norm}, device)
    if not hasattr(drafter, "slot_logits"):
        raise SystemExit(f"{type(drafter).__name__} has no slot_logits")
    hello, streams = requests(args.taps, None)
    wanted = [f"o{layer + args.delta}" for layer in drafter.layer_ids]
    missing = [name for name in wanted + ["final"] if name not in hello["entries"]]
    if missing:
        raise SystemExit(f"tap stream lacks {missing}; it has {hello['entries']}")
    columns = [hello["entries"].index(name) for name in wanted]
    final_column = hello["entries"].index("final")
    inverse = 1.0 / args.temperature
    depth = getattr(drafter, "max_drafts", drafter.block)
    arms = {"keyed": Chain(depth), "greedy": Chain(depth)}
    for tau in args.taus:
        arms[f"rejection tau {tau}"] = Chain(depth)
    replayed = {"rounds": 0, "committed": 0, "kept": [0] * depth, "offered": [0] * depth}
    report = {"arm": vars(args), "requests": []}
    for key, item in sorted(streams.items()):
        end = max(item["positions"]) + 1
        start = end - 1
        while start - 1 in item["positions"]:
            start -= 1
        records = [item["positions"][p] for p in range(start, end)]
        reply = [index for index, record in enumerate(records) if record[3] & 0x6]
        if len(reply) < args.min_reply:
            continue
        tokens = [record[2] for record in records]
        taps = load_rows(hello, records, columns)
        final = load_rows(hello, records, [final_column])[:, 0]
        positions = torch.arange(start, end, device=device)
        with torch.no_grad():
            keys, values = drafter.context(taps.reshape(len(records), -1), positions)
        del taps
        anchors = list(range(reply[0], len(records) - 1))
        if len(anchors) > args.max_anchors:
            step = len(anchors) / max(1, args.max_anchors)
            anchors = [anchors[int(i * step)] for i in range(args.max_anchors)]
        for anchor in anchors:
            with torch.no_grad():
                y, _ = drafter.draft(tokens[anchor], start + anchor, keys, values, anchor)
                previous = tokens[anchor]
                rates = {name: [] for name in arms}
                for slot in range(depth):
                    row = anchor + slot
                    if row >= len(records) - 1:
                        break
                    target = rms(final[row:row + 1].to(device), drafter.target_norm, drafter.eps).to(torch.bfloat16)
                    p = nucleus((target @ drafter.head.t()).float()[0], inverse, args.top_p, args.top_k)
                    draft_logits = drafter.slot_logits(y, slot, previous)
                    q = nucleus(draft_logits, inverse, args.top_p, args.top_k)
                    rates["keyed"].append(keyed_match(p, q))
                    rates["greedy"].append(float(p[torch.argmax(draft_logits)]))
                    for tau in args.taus:
                        shaped = nucleus(draft_logits, inverse / tau, args.top_p, args.top_k)
                        rates[f"rejection tau {tau}"].append(float(torch.minimum(p, shaped).sum()))
                    previous = tokens[row + 1]
            for name, chain in arms.items():
                chain.add(rates[name])
        kept = {}

        def keep_rates(anchor):
            if anchor not in kept:
                with torch.no_grad():
                    y, _ = drafter.draft(tokens[anchor], start + anchor, keys, values, anchor)
                    rows = list(range(anchor, min(anchor + depth, len(records) - 1)))
                    target = rms(final[rows[0]:rows[-1] + 1].to(device), drafter.target_norm, drafter.eps).to(torch.bfloat16)
                    logits = (target @ drafter.head.t()).float()
                    previous, rates = tokens[anchor], []
                    for slot, row in enumerate(rows):
                        truth = tokens[row + 1]
                        p = nucleus(logits[slot], inverse, args.top_p, args.top_k)[truth]
                        q = nucleus(drafter.slot_logits(y, slot, previous), inverse, args.top_p, args.top_k)[truth]
                        rates.append(min(1.0, float(q / p)) if p > 0 else 0.0)
                        previous = truth
                kept[anchor] = rates
            return kept[anchor]
        for trial in range(args.replay):
            chance = random.Random(f"{args.replay_seed}:{key[1]}:{trial}")
            anchor = reply[0]
            while anchor < len(records) - 1:
                rates = keep_rates(anchor)
                accepted = 0
                for slot, rate in enumerate(rates):
                    replayed["offered"][slot] += 1
                    if chance.random() >= rate:
                        break
                    replayed["kept"][slot] += 1
                    accepted += 1
                replayed["rounds"] += 1
                replayed["committed"] += min(accepted + 1, len(records) - 1 - anchor)
                anchor += accepted + 1
        report["requests"].append({"request": key[1], "positions": len(records), "reply": len(reply), "anchors": len(anchors), "replay_drafts": len(kept)})
        print(f"request {key[1]} positions {len(records)} reply {len(reply)} " + " ".join(f"{name}={chain.tokens / max(chain.anchors, 1):.2f}" for name, chain in arms.items()) +
              (f" replay={replayed['committed'] / max(1, replayed['rounds']):.2f}" if args.replay else ""), flush=True)
    for name, chain in arms.items():
        report[name] = chain.summary()
        print(name, json.dumps(report[name]), flush=True)
    if args.replay:
        report["rejection replay"] = {"tokens_per_round": round(replayed["committed"] / max(1, replayed["rounds"]), 3), "rounds": replayed["rounds"],
                                      "slot_kept": [round(k / o, 3) if o else None for k, o in zip(replayed["kept"], replayed["offered"])]}
        print("rejection replay", json.dumps(report["rejection replay"]), flush=True)
    Path(args.out).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
