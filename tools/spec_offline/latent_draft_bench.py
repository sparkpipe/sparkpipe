#!/usr/bin/env python3
"""Draft latency of a latent block drafter through its CUDA graph at several context lengths.

The context latents are filled with random bf16 rows (the cost of a draft does not depend on their values), so the
bench needs only the drafter and the target shard. It reports p50/p90 milliseconds per draft for greedy and sampled
drafts at each length, and with --compare-reference the same drafts through the f32 reference attention with the share
of drafts whose tokens agree."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from latent_block_drafter import LatentBlockDrafter, LatentContext, LatentDraftGraph  # noqa: E402


def fill(context, positions, generator):
    d = context.drafter
    context.grow(positions)
    if context.window is not None:
        keep = np.arange(max(0, positions - context.window), positions)
        for index in context.sliding:
            context.rings[index][torch.from_numpy(keep % context.window).to(d.device)] = torch.randn((len(keep), context.width), generator=generator, device=d.device).to(torch.bfloat16)
        context.ring_positions[keep % context.window] = keep
    for index in context.full:
        for page in range(-(-positions // context.page_positions)):
            rows = min(context.page_positions, positions - page * context.page_positions)
            context.pages[index][page][:rows] = torch.randn((rows, context.width), generator=generator, device=d.device).to(torch.bfloat16)
    context.held[:positions] = True


def run(graph, drafter, positions, rows, sampling, drafts):
    times, tokens = [], []
    for draft in range(drafts + 4):
        torch.cuda.synchronize()
        started = time.perf_counter()
        drafted, _ = graph.draft(17 + draft, positions, rows, drafter.max_drafts, sampling)
        elapsed = (time.perf_counter() - started) * 1e3
        if draft >= 4:
            times.append(elapsed)
            tokens.append(drafted)
    return times, tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--drafter", required=True)
    parser.add_argument("--target-shard", required=True)
    parser.add_argument("--embed", default="language_model.model.embed_tokens.weight")
    parser.add_argument("--head", default="language_model.lm_head.weight")
    parser.add_argument("--norm", default="language_model.model.norm.weight")
    parser.add_argument("--contexts", default="4096,137728,270000,737280")
    parser.add_argument("--drafts", type=int, default=64)
    parser.add_argument("--compare-reference", action="store_true")
    parser.add_argument("--out")
    args = parser.parse_args()
    torch.set_num_threads(1)
    drafter = LatentBlockDrafter(args.drafter, args.target_shard, {"embed": args.embed, "head": args.head, "norm": args.norm}, "cuda")
    graph = LatentDraftGraph(drafter)
    generator = torch.Generator(device="cuda").manual_seed(5)
    results = []
    for positions in (int(x) for x in args.contexts.split(",")):
        context = LatentContext(drafter)
        fill(context, positions, generator)
        rows, complete = context.contexts(positions)
        assert complete
        for label, sampling in (("greedy", None), ("sampled", (4242, 1.0, 0.95, 0))):
            times, tokens = run(graph, drafter, positions, rows, sampling, args.drafts)
            result = {"context": positions, "mode": label, "attention": drafter.attention, "p50_ms": round(float(np.percentile(times, 50)), 2),
                      "p90_ms": round(float(np.percentile(times, 90)), 2), "context_bytes": context.bytes(),
                      "gpu_allocated_gb": round(torch.cuda.memory_allocated() / 1e9, 2)}
            if args.compare_reference and drafter.fused is not None:
                fused, drafter.fused = drafter.fused, None
                try:
                    reference_times, reference_tokens = run(graph, drafter, positions, rows, sampling, args.drafts)
                finally:
                    drafter.fused = fused
                result["reference_p50_ms"] = round(float(np.percentile(reference_times, 50)), 2)
                result["token_agreement"] = round(sum(a == b for a, b in zip(tokens, reference_tokens)) / len(tokens), 3)
            print(json.dumps(result), flush=True)
            results.append(result)
        del context, rows
        torch.cuda.empty_cache()
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
