#!/usr/bin/env python3
"""Pack the driver T1 raw dumps into a T1R1 candidate fixture.

Every array name the reference fixture carries must be producible from the
dump files; nothing else is emitted. Stream width and top-k come from the
reference fixture itself.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from t1_reference_common import read_fixture, write_fixture


def read_raw(path, dtype, shape):
    data = np.fromfile(path, dtype=dtype)
    expected = int(np.prod(shape))
    if data.size != expected:
        raise ValueError(f"{path}: {data.size} elements, expected {expected}")
    return data.reshape(shape)


def head_score(dump_dir, position):
    for name in (f"pos{position:04d}_head_top1_score.f32",
                 f"pos{position:04d}_head_top1_score.rank1.f32",
                 f"pos{position:04d}_head_top1_score.rank2.f32",
                 f"pos{position:04d}_head_top1_score.rank3.f32",
                 f"pos{position:04d}_head_top1_score.rank4.f32",
                 f"pos{position:04d}_head_top1_score.rank5.f32",
                 f"pos{position:04d}_head_top1_score.rank6.f32",
                 f"pos{position:04d}_head_top1_score.rank7.f32"):
        path = os.path.join(dump_dir, name)
        if os.path.exists(path):
            return read_raw(path, np.float32, [1])
    raise FileNotFoundError(f"no rank owns the winning score for position {position}")


def build_arrays(reference, dump_dir, prompt_ids, new_tokens):
    meta, _ = read_fixture(reference)
    arrays = {}
    for entry in meta["arrays"]:
        name = entry["name"]
        parts = name.split("_")
        if name == "prompt_token_ids":
            arrays[name] = np.array(prompt_ids, dtype=np.int32)
        elif name == "generated_token_ids":
            tokens = [int(read_raw(os.path.join(dump_dir,
                      f"pos{position:04d}_head_top1_token.u32"), np.uint32, [1])[0])
                for position in range(len(prompt_ids), len(prompt_ids) + new_tokens)]
            arrays[name] = np.array(tokens, dtype=np.int32)
        elif name.endswith("_head_top1_token"):
            position = int(parts[0][3:])
            raw = read_raw(os.path.join(dump_dir,
                           f"pos{position:04d}_head_top1_token.u32"), np.uint32, [1])
            arrays[name] = raw.astype(np.int32)
        elif name.endswith("_head_top1_score"):
            position = int(parts[0][3:])
            arrays[name] = head_score(dump_dir, position)
        elif name.endswith("_route_ids"):
            position, layer = int(parts[0][3:]), int(parts[1][5:])
            topk = int(np.prod(entry["shape"]))
            raw = read_raw(os.path.join(dump_dir,
                           f"pos{position:04d}_layer{layer:04d}_route_ids.u32"),
                np.uint32, [topk])
            arrays[name] = raw.astype(np.int32)
        elif name.endswith("_route_weights"):
            position, layer = int(parts[0][3:]), int(parts[1][5:])
            topk = int(np.prod(entry["shape"]))
            arrays[name] = read_raw(os.path.join(dump_dir,
                                    f"pos{position:04d}_layer{layer:04d}_route_weights.f32"),
                np.float32, [topk])
        elif name.endswith("_streams"):
            position, layer = int(parts[0][3:]), int(parts[1][5:])
            width = int(np.prod(entry["shape"]))
            arrays[name] = read_raw(os.path.join(dump_dir,
                                    f"pos{position:04d}_layer{layer:04d}_streams.u16"),
                np.uint16, [width])
        else:
            raise ValueError(f"unhandled reference array {name}")
    return arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--dump-dir", required=True)
    parser.add_argument("--prompt-ids", required=True)
    parser.add_argument("--new-tokens", type=int, required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    prompt_ids = [int(value) for value in arguments.prompt_ids.split(",")
                  if value != ""]
    arrays = build_arrays(arguments.reference, arguments.dump_dir, prompt_ids,
                          arguments.new_tokens)
    write_fixture(arguments.output, arrays)
    print(json.dumps({"output": arguments.output, "arrays": len(arrays),
                      "prompt_count": len(prompt_ids),
                      "new_tokens": arguments.new_tokens}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
