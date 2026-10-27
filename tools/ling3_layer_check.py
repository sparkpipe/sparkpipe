#!/usr/bin/env python3
"""Sublayer-isolated comparison of a Ling 3.0 DEBUG dump against tools/ling3_reference.py.

A DEBUG build of the ling stage appends, on rank 0 and for every uncaptured
step, the residual and the all-reduced partial after each attention and
each MLP round to ling3-debug-rank0.bin (records: u32 magic 0x4c474443,
u32 layer (0xffffffff embedding, 0x10000 + i after attention i, i after
MLP i), u32 rows, u32 hidden, u32 positions[rows], u32 sequence_of_row[rows],
bf16 residual[rows, hidden], bf16 partial[rows, hidden]).

For a one-step prefill dump of --tokens, every sublayer is recomputed by
the reference from the driver's own input to that sublayer, so errors do not
accumulate: attention i from the residual entering layer i, MLP i from the
residual after attention i. Reported per sublayer: relative L2 error of the
driver partial against the reference partial (mean, median, max over
rows), and the error of the residual hand-off between the two rounds.
"""

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ling3_reference  # noqa: E402

MAGIC = 0x4c474443


def load(path):
    data = Path(path).read_bytes()
    records, at = {}, 0
    while at < len(data):
        magic, layer, rows, hidden = struct.unpack_from("<4I", data, at)
        if magic != MAGIC:
            raise ValueError(f"{path}: bad record magic at byte {at}")
        at += 16
        positions = np.frombuffer(data, np.uint32, rows, at).copy()
        sequences = np.frombuffer(data, np.uint32, rows, at + 4 * rows).copy()
        at += 8 * rows
        values = (np.frombuffer(data, np.uint16, 2 * rows * hidden, at).astype(np.uint32) << 16).view(np.float32).reshape(2, rows, hidden)
        at += 4 * rows * hidden
        records.setdefault(layer, []).append((positions, values, sequences))
    return records


def error(driver, reference):
    rows = np.linalg.norm(driver - reference, axis=-1) / np.maximum(np.linalg.norm(reference, axis=-1), np.float32(1e-30))
    return {"mean": float(rows.mean()), "median": float(np.median(rows)), "max": float(rows.max()), "argmax": int(rows.argmax())}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dump", required=True)
    parser.add_argument("--tokens", required=True, help="JSON list of the token ids the dumped step prefilled")
    parser.add_argument("--layers", default="all")
    parser.add_argument("--swiglu-limits", choices=("serving", "modeling"), default="serving")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    reference = ling3_reference.Reference(args.checkpoint, args.swiglu_limits)
    tokens = np.asarray(json.loads(Path(args.tokens).read_text()), dtype=np.int64)
    records = load(args.dump)
    steps = [index for index, (positions, _, _) in enumerate(records[0xffffffff]) if len(positions) == len(tokens)]
    if not steps:
        raise ValueError(f"the dump has no step of {len(tokens)} rows")
    step = steps[-1]
    pick = lambda layer: records[layer][step][:2]
    positions, embedded = pick(0xffffffff)
    if not np.array_equal(positions, np.arange(len(tokens), dtype=np.uint32)):
        raise ValueError("the dumped step is not a prefill of positions 0..n-1")
    embed = reference.checkpoint.raw("model.word_embeddings.weight", (reference.vocab, reference.hidden), "BF16")
    expected = (np.asarray(embed[tokens]).astype(np.uint32) << 16).view(np.float32)
    report = {"embed": error(embedded[0], expected)}
    print(f"embed rel_err {report['embed']}", flush=True)
    layers = range(reference.layers) if args.layers == "all" else [int(v) for v in args.layers.split(",")]
    positions = positions.astype(np.int64)
    for layer in layers:
        p = f"model.layers.{layer}."
        _, attention = pick(0x10000 + layer)
        normed = reference.norm(attention[0], p + "input_layernorm.weight", reference.hidden)
        kind = "mla" if reference.is_attention(layer) else "kda"
        computed = reference.mla(layer, normed, positions) if kind == "mla" else reference.kda(layer, normed)
        entry = {"attention": kind, "attention_err": error(attention[1], computed)}
        _, mlp = pick(layer)
        normed = reference.norm(mlp[0], p + "post_attention_layernorm.weight", reference.hidden)
        if layer < reference.config["first_k_dense_replace"]:
            computed = reference.swiglu(normed, p + "mlp.", reference.config["intermediate_size"], 0.0)
        else:
            computed, _ = reference.moe(layer, normed)
        entry["mlp_err"] = error(mlp[1], computed)
        residual = ling3_reference.bf16(attention[0] + attention[1])
        entry["residual_handoff_err"] = error(mlp[0], residual)
        report[f"layer{layer:02d}"] = entry
        print(f"layer {layer:2d} {kind} attention {entry['attention_err']['median']:.5f}/{entry['attention_err']['max']:.5f} "
              f"mlp {entry['mlp_err']['median']:.5f}/{entry['mlp_err']['max']:.5f} handoff {entry['residual_handoff_err']['max']:.5f}", flush=True)
    Path(args.out).write_text(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
