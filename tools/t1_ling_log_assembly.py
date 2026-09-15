#!/usr/bin/env python3
"""Assemble a candidate T1R1 fixture from a rank-0 driver log carrying
LNG-T1 dump lines (SPARK_LING_T1=1 module build).

Line grammar (one line per row):
  LNG-T1 wave seq<r> rows=<n> pos<p>=<token>...
  LNG-T1 stream L<layer> pos<p> <2560 hex u16 words>
  LNG-T1 route L<layer> pos<p> ids <8 ints> weights <8 hex f32 bits>
  LNG-T1 head pos<p> token <t> score_bits <hex f32 bits>

Array names follow qualification/t1_reference/ling (T1R1 contract):
streams (already the post-MLP residual, reconstructed in the module dump
from hidden + the reduced MLP delta with the kernel's own bf16 rounding)
at every dumped layer/position, route ids/weights at every routed
layer/position, head top-1 token/score, prompt_token_ids (row 0 of the
first wave of a sequence) and generated_token_ids (head tokens from the
last prompt position onward).
"""
import argparse
import json
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from t1_reference_common import write_fixture

HIDDEN = 2560
TOP_K = 8


def parse_number(text):
    return int(text, 16) if text.startswith("0x") else int(text)


def assemble(log_path, prompt_length, new_tokens):
    sequences = {}
    streams = {}
    routes = {}
    heads = {}
    with open(log_path, "r", errors="replace") as handle:
        for line in handle:
            if line.startswith("LNG-T1 wave "):
                fields = line.split()
                seq = int(fields[3][3:])
                rows = int(fields[4][5:])
                pairs = {}
                for field in fields[5:]:
                    pos, token = field[1:].split("=")
                    pairs[int(pos)] = int(token)
                record = sequences.setdefault(seq, {"rows": rows, "waves": []})
                record["waves"].append(pairs)
            elif line.startswith("LNG-T1 stream "):
                fields = line.split()
                layer = int(fields[3][1:])
                position = int(fields[4][3:])
                words = [int(w, 16) for w in fields[5:]]
                if len(words) != HIDDEN:
                    raise ValueError(f"stream row has {len(words)} words")
                streams[(position, layer)] = np.array(words, np.uint16)
            elif line.startswith("LNG-T1 route "):
                fields = line.split()
                layer = int(fields[3][1:])
                position = int(fields[4][3:])
                ids_at = fields.index("ids")
                weights_at = fields.index("weights")
                ids = [int(v) for v in fields[ids_at + 1:weights_at]]
                weights = [int(v, 16) for v in fields[weights_at + 1:]]
                if len(ids) != TOP_K or len(weights) != TOP_K:
                    raise ValueError(f"route row has {len(ids)} ids")
                routes[(position, layer)] = (
                    np.array(ids, np.int32),
                    np.frombuffer(np.array(weights, np.uint32).tobytes(),
                                  np.float32).copy())
            elif line.startswith("LNG-T1 head "):
                fields = line.split()
                position = int(fields[3][3:])
                token = int(fields[5])
                score_bits = int(fields[7], 16)
                heads[position] = (
                    np.array([token], np.int32),
                    np.frombuffer(np.array([score_bits], np.uint32).tobytes(),
                                  np.float32).copy())
    return sequences, streams, routes, heads


def build_arrays(sequences, streams, routes, heads, prompt_length, new_tokens):
    arrays = {}
    for (position, layer), words in sorted(streams.items()):
        arrays[f"pos{position:04d}_layer{layer:04d}_streams"] = words
    for (position, layer), (ids, weights) in sorted(routes.items()):
        arrays[f"pos{position:04d}_layer{layer:04d}_route_ids"] = ids
        arrays[f"pos{position:04d}_layer{layer:04d}_route_weights"] = weights
    for position, (token, score) in sorted(heads.items()):
        arrays[f"pos{position:04d}_head_top1_token"] = token
        arrays[f"pos{position:04d}_head_top1_score"] = score
    for seq, record in sorted(sequences.items()):
        prompt = dict(record["waves"][0])
        prompt_positions = sorted(prompt)[:prompt_length]
        arrays[f"seq{seq}_prompt_token_ids"] = np.array(
            [prompt[p] for p in prompt_positions], np.int32)
        generated = [heads[p][0][0] for p in sorted(heads)
                     if p >= prompt_length - 1][:new_tokens]
        arrays[f"seq{seq}_generated_token_ids"] = np.array(generated, np.int32)
    return arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True)
    parser.add_argument("--prompt-length", type=int, default=5)
    parser.add_argument("--new-tokens", type=int, default=4)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    sequences, streams, routes, heads = assemble(
        arguments.log, arguments.prompt_length, arguments.new_tokens)
    arrays = build_arrays(sequences, streams, routes, heads,
                          arguments.prompt_length, arguments.new_tokens)
    write_fixture(arguments.output, arrays)
    print(json.dumps({"arrays": len(arrays), "streams": len(streams),
                      "routes": len(routes), "heads": len(heads),
                      "sequences": {s: len(r["waves"])
                                    for s, r in sorted(sequences.items())},
                      "output": arguments.output}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
