#!/usr/bin/env python3
"""Assemble a candidate T1R1 fixture from per-stage driver logs carrying
G4-T1 dump lines (SPARK_GEMMA4_T1=1 module build, tp_rank 0 of each stage).

Line grammar (one line per row):
  G4-T1 frame stage<s> rows<n> prefill<f> seq<id> pos <p>...
  G4-T1 stream L<layer> pos<p> <2816 hex u16 words>
  G4-T1 route L<layer> pos<p> ids <8 ints> weights <8 hex f32 bits>
  G4-T1 head pos<p> token <t> score_bits <hex f32 bits>

Streams/routes are keyed (position, layer); the stage that owns the layer
supplies the row. Array names follow the T1R1 contract. prompt_token_ids
and generated_token_ids come from the prompts file (the serving arm decodes
the same canonical prompts); generated tokens are the head tokens strictly
after the last prompt position.
"""
import argparse
import json
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from t1_reference_common import write_fixture


def assemble(log_paths, prompts_path):
    prompts = json.load(open(prompts_path))["prompts"]
    streams = {}
    routes = {}
    heads = {}
    frames = []
    for log_path in log_paths:
        with open(log_path, "r", errors="replace") as handle:
            for line in handle:
                if line.startswith("G4-T1 frame "):
                    fields = line.split()
                    frames.append({
                        "stage": int(fields[3][5:]),
                        "rows": int(fields[4][4:]),
                        "prefill": int(fields[5][7:]),
                        "positions": [int(v) for v in fields[7:]],
                    })
                elif line.startswith("G4-T1 stream "):
                    fields = line.split()
                    layer = int(fields[3][1:])
                    position = int(fields[4][3:])
                    words = [int(w, 16) for w in fields[5:]]
                    streams[(position, layer)] = np.array(words, np.uint16)
                elif line.startswith("G4-T1 route "):
                    fields = line.split()
                    layer = int(fields[3][1:])
                    position = int(fields[4][3:])
                    ids_at = fields.index("ids")
                    weights_at = fields.index("weights")
                    ids = [int(v) for v in fields[ids_at + 1:weights_at]]
                    weights = [int(v, 16) for v in fields[weights_at + 1:]]
                    routes[(position, layer)] = (
                        np.array(ids, np.int32),
                        np.array(weights, np.float32))
                elif line.startswith("G4-T1 head "):
                    fields = line.split()
                    position = int(fields[3][3:])
                    heads[position] = (int(fields[5]), int(fields[7], 16))
    arrays = {}
    capture_layers = set()
    for spec in prompts:
        prompt_len = len(spec["prompt_token_ids"])
        budget = int(spec["new_tokens"])
        for position in range(prompt_len + budget):
            token, score_bits = heads[position]
            arrays[f"pos{position:04d}_head_top1_token"] = \
                np.array([token], np.int32)
            arrays[f"pos{position:04d}_head_top1_score"] = \
                np.array([np.float32(np.uint32(score_bits))], np.float32)
        for layer in spec["capture_layers"]:
            capture_layers.add(int(layer))
        arrays.setdefault("prompt_token_ids",
                          np.array(spec["prompt_token_ids"], np.int32))
        generated = [heads[p][0] for p in
                     range(prompt_len, prompt_len + budget)]
        arrays.setdefault("generated_token_ids",
                          np.array(generated, np.int32))
    for (position, layer), words in streams.items():
        if layer in capture_layers:
            arrays[f"pos{position:04d}_layer{layer:04d}_streams"] = words
    for (position, layer), (ids, weights) in routes.items():
        if layer in capture_layers:
            arrays[f"pos{position:04d}_layer{layer:04d}_route_ids"] = ids
            arrays[f"pos{position:04d}_layer{layer:04d}_route_weights"] = \
                weights
    return arrays, frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", action="append", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    arrays, frames = assemble(arguments.log, arguments.prompts)
    write_fixture(arguments.output, arrays)
    print(json.dumps({"output": arguments.output, "arrays": len(arrays),
                      "frames": len(frames)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
