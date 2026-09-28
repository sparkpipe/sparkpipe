#!/usr/bin/env python3
import argparse
import json
import os
import sys

import numpy as np

HIDDEN = 6144
STREAMS = 4
LAYERS = 78
TOP_K = 8


def load_run(directory, prompt):
    name = prompt["name"]
    text = open(os.path.join(directory, f"{name}.txt"), encoding="utf-8").read().splitlines()
    fields = {line.split()[0]: line.split()[1:] for line in text if line.split()[0] in ("prompt", "generated")}
    generated = [int(value) for value in fields.get("generated", [])]
    heads = {}
    for line in text:
        parts = line.split()
        if parts[0] == "head":
            heads[int(parts[1])] = [(int(pair.split(":")[0]), float(pair.split(":")[1])) for pair in parts[2:]]
    total = len(prompt["prompt_token_ids"]) + len(generated)
    capture = prompt["capture_layers"]
    streams = np.fromfile(os.path.join(directory, f"{name}.streams.bin"), dtype=np.float32)
    if streams.size != len(capture) * total * STREAMS * HIDDEN:
        raise SystemExit(f"{directory}/{name}: streams size {streams.size} disagrees with {total} positions")
    routes = np.fromfile(os.path.join(directory, f"{name}.routes.bin"), dtype=np.uint8)
    route_bytes = total * (LAYERS - 1) * TOP_K * 4
    if routes.size != 2 * route_bytes:
        raise SystemExit(f"{directory}/{name}: routes size {routes.size} disagrees with {total} positions")
    return {
        "generated": generated,
        "heads": heads,
        "streams": streams.reshape(len(capture), total, STREAMS * HIDDEN),
        "route_ids": routes[:route_bytes].view(np.int32).reshape(total, LAYERS - 1, TOP_K),
        "route_weights": routes[route_bytes:].view(np.float32).reshape(total, LAYERS - 1, TOP_K),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--layers", type=int, default=LAYERS)
    parser.add_argument("--positions", choices=("prompt", "all"), default="all")
    parser.add_argument("--stream-band", type=float, default=0.02)
    parser.add_argument("--teacher-forced", action="store_true")
    parser.add_argument("--tie-margin", type=float, default=1.0)
    args = parser.parse_args()
    prompts = json.load(open(args.prompts))["prompts"]
    failures = 0
    for prompt in prompts:
        name = prompt["name"]
        reference = load_run(args.reference, prompt)
        candidate = load_run(args.candidate, prompt)
        prompt_count = len(prompt["prompt_token_ids"])
        if args.positions == "prompt":
            positions = prompt_count
        else:
            positions = min(prompt_count + len(reference["generated"]), prompt_count + len(candidate["generated"])) - 1
            positions = max(positions, prompt_count)
        for slot, layer in enumerate(prompt["capture_layers"]):
            if layer >= args.layers:
                continue
            worst = 0.0
            worst_cos = 1.0
            for position in range(positions):
                ref = reference["streams"][slot, position].astype(np.float64)
                cand = candidate["streams"][slot, position].astype(np.float64)
                rel = float(np.max(np.abs(ref - cand)) / max(np.max(np.abs(ref)), 1e-30))
                cos = float(np.dot(ref, cand) / max(np.linalg.norm(ref) * np.linalg.norm(cand), 1e-30))
                worst = max(worst, rel)
                worst_cos = min(worst_cos, cos)
            verdict = "PASS" if worst <= args.stream_band else "FAIL"
            failures += verdict == "FAIL"
            print(f"{verdict} {name} layer {layer} streams over {positions} positions: max rel {worst:.4g}, min cosine {worst_cos:.6f}")
        route_layers = range(1, min(args.layers, LAYERS))
        mismatched = 0
        compared = 0
        weight_rel = 0.0
        for position in range(positions):
            for layer in route_layers:
                ref_ids = reference["route_ids"][position, layer - 1]
                cand_ids = candidate["route_ids"][position, layer - 1]
                compared += 1
                if sorted(ref_ids.tolist()) != sorted(cand_ids.tolist()):
                    mismatched += 1
                    continue
                order = np.argsort(ref_ids)
                corder = np.argsort(cand_ids)
                ref_w = reference["route_weights"][position, layer - 1][order]
                cand_w = candidate["route_weights"][position, layer - 1][corder]
                weight_rel = max(weight_rel, float(np.max(np.abs(ref_w - cand_w) / np.maximum(np.abs(ref_w), 1e-30))))
        if compared:
            print(f"{'PASS' if mismatched == 0 else 'NOTE'} {name} routes: {compared - mismatched}/{compared} (position, layer) expert sets equal, max weight rel {weight_rel:.4g}")
        steps = min(len(reference["generated"]), len(candidate["generated"]))
        if args.teacher_forced:
            if reference["generated"][:steps] != candidate["generated"][:steps]:
                raise SystemExit(f"{name}: candidate was not forced onto the reference tokens")
            agree = 0
            ties = 0
            for position in sorted(set(reference["heads"]) & set(candidate["heads"])):
                ref_top = reference["heads"][position]
                cand_top = candidate["heads"][position]
                if ref_top[0][0] == cand_top[0][0]:
                    agree += 1
                    continue
                margin = ref_top[0][1] - ref_top[1][1]
                tie = margin < args.tie_margin and cand_top[0][0] == ref_top[1][0]
                ties += tie
                failures += not tie
                print(f"  {'TIE ' if tie else 'MISS'} {name} head {position}: reference {ref_top[0][0]} margin {margin:.4f} over {ref_top[1][0]}, candidate {cand_top[0][0]} {cand_top[0][1]:.4f} vs {cand_top[1][0]} {cand_top[1][1]:.4f}")
            total = len(set(reference["heads"]) & set(candidate["heads"]))
            print(f"{'PASS' if agree + ties == total else 'FAIL'} {name} teacher-forced head top1: {agree}/{total} equal, {ties} reference near-ties (margin < {args.tie_margin})")
        elif steps:
            equal = reference["generated"][:steps] == candidate["generated"][:steps]
            failures += not equal
            print(f"{'PASS' if equal else 'FAIL'} {name} tokens: reference {reference['generated'][:steps]} candidate {candidate['generated'][:steps]}")
            for position in sorted(set(reference["heads"]) & set(candidate["heads"])):
                ref_top = reference["heads"][position][:2]
                cand_top = candidate["heads"][position][:2]
                margin = ref_top[0][1] - ref_top[1][1]
                print(f"  head {position}: reference {ref_top[0][0]} {ref_top[0][1]:.4f} (margin {margin:.4f}) candidate {cand_top[0][0]} {cand_top[0][1]:.4f}")
    print("RESULT:", "PASS" if failures == 0 else f"FAIL ({failures})")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
