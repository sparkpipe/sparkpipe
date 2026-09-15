#!/usr/bin/env python3
"""Assemble a candidate T1R1 fixture from rank-0 driver logs carrying
LAG-T1 dump lines (SPARK_LAGUNA_T1=1 module build, tp8.pp2 stage0 rank0
and stage1 rank8 both dump; layer numbering is global in the lines).

Line grammar (one line per row):
  LAG-T1 wave seq<r> rows=<n> pos<p>=<token>...
  LAG-T1 stream L<layer> pos<p> <3072 hex u16 words>
  LAG-T1 route L<layer> pos<p> ids <10 ints> weights <10 hex f32 bits>
  LAG-T1 head pos<p> token <t> score_bits <hex f32 bits>

Array names follow qualification/t1_reference/laguna (T1R1 contract):
streams at every dumped layer/position, route ids/weights at every routed
layer/position, head top-1 token/score, prompt_token_ids (row 0 of the
first wave of the single served sequence) and generated_token_ids (head
tokens strictly after the last prompt position).
"""
import argparse
import json
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from t1_reference_common import write_fixture

FLAT = 3072
TOP_K = 10


def parse_number(text):
    return int(text, 16) if text.startswith("0x") else int(text)


def assemble(log_path, prompt_length, new_tokens):
    sequences = {}
    streams = {}
    routes = {}
    heads = {}
    with open(log_path, "r", errors="replace") as handle:
        for line in handle:
            if line.startswith("LAG-T1 wave "):
                fields = line.split()
                seq = int(fields[3][3:])
                rows = int(fields[4][5:])
                pairs = {}
                for field in fields[5:]:
                    pos, token = field[1:].split("=")
                    pairs[int(pos)] = int(token)
                record = sequences.setdefault(seq, {"rows": rows, "waves": []})
                record["waves"].append(pairs)
            elif line.startswith("LAG-T1 stream "):
                fields = line.split()
                layer = int(fields[3][1:])
                position = int(fields[4][3:])
                words = [int(w, 16) for w in fields[5:]]
                if len(words) != FLAT:
                    raise ValueError(f"stream row has {len(words)} words")
                streams[(position, layer)] = np.array(words, np.uint16)
            elif line.startswith("LAG-T1 route "):
                fields = line.split()
                layer = int(fields[3][1:])
                position = int(fields[4][3:])
                ids_at = fields.index("ids")
                weights_at = fields.index("weights")
                ids = [int(v) for v in fields[ids_at + 1:weights_at]]
                weights = [int(v, 16) for v in fields[weights_at + 1:]]
                if len(ids) != TOP_K or len(weights) != TOP_K:
                    raise ValueError(
                        f"route row has {len(ids)} ids / {len(weights)} weights")
                routes[(position, layer)] = (
                    np.array(ids, np.int32),
                    np.frombuffer(np.array(weights, np.uint32).tobytes(),
                                  np.float32).copy())
            elif line.startswith("LAG-T1 head "):
                fields = line.split()
                position = int(fields[3][3:])
                token = int(fields[5])
                score_bits = int(fields[7], 16)
                heads[position] = (
                    np.array([token], np.int32),
                    np.frombuffer(np.array([score_bits], np.uint32).tobytes(),
                                  np.float32).copy())
    if not sequences:
        raise ValueError("no LAG-T1 wave lines in log")
    seq = sorted(sequences)[0]
    record = sequences[seq]
    prompt = dict(record["waves"][0])
    prompt_positions = sorted(prompt)[:prompt_length]
    if len(prompt_positions) != prompt_length:
        raise ValueError(f"first wave covers {len(prompt_positions)} "
                         f"positions, prompt length is {prompt_length}")
    expected = list(range(prompt_length - 1, prompt_length + new_tokens))
    missing_heads = [p for p in expected if p not in heads]
    if missing_heads:
        raise ValueError(f"head lines missing for positions {missing_heads}")
    arrays = {}
    for (position, layer), words in sorted(streams.items()):
        arrays[f"pos{position:04d}_layer{layer:04d}_streams"] = words
    for (position, layer), (ids, weights) in sorted(routes.items()):
        arrays[f"pos{position:04d}_layer{layer:04d}_route_ids"] = ids
        arrays[f"pos{position:04d}_layer{layer:04d}_route_weights"] = weights
    for position in expected:
        token, score = heads[position]
        arrays[f"pos{position:04d}_head_top1_token"] = token
        arrays[f"pos{position:04d}_head_top1_score"] = score
    arrays["prompt_token_ids"] = np.array(
        [prompt[p] for p in prompt_positions], np.int32)
    arrays["generated_token_ids"] = np.array(
        [heads[p][0][0] for p in expected[1:]], np.int32)
    return arrays, sequences


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True)
    parser.add_argument("--prompt-length", type=int, required=True)
    parser.add_argument("--new-tokens", type=int, required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    arrays, sequences = assemble(arguments.log, arguments.prompt_length,
                                 arguments.new_tokens)
    write_fixture(arguments.output, arrays)
    print(json.dumps({"arrays": len(arrays),
                      "sequences": {s: len(r["waves"])
                                    for s, r in sorted(sequences.items())},
                      "output": arguments.output}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
