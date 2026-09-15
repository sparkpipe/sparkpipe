#!/usr/bin/env python3
"""Pack the driver T1 raw dumps into a T1R1 candidate fixture.

The reference fixture is the checklist: every array name it carries must be
producible from the dump files; nothing else is emitted.
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


def head_token(dump_dir, position):
    return int(read_raw(os.path.join(dump_dir,
              f"pos{position:04d}_head_top1_token.u32"), np.uint32, [1])[0])


def build_arrays(reference, dump_dir, prompt_ids, new_tokens):
    meta, _ = read_fixture(reference)
    arrays = {}
    for entry in meta["arrays"]:
        name = entry["name"]
        parts = name.split("_")
        if name == "prompt_token_ids":
            arrays[name] = np.array(prompt_ids, dtype=np.int32)
        elif name == "generated_token_ids":
            tokens = [head_token(dump_dir, position)
                      for position in range(len(prompt_ids), len(prompt_ids) + new_tokens)]
            arrays[name] = np.array(tokens, dtype=np.int32)
        elif name.endswith("_head_top1_token"):
            position = int(parts[0][3:])
            raw = read_raw(os.path.join(dump_dir,
                           f"pos{position:04d}_head_top1_token.u32"), np.uint32, [1])
            arrays[name] = raw.astype(np.int32)
        elif name.endswith("_head_top1_score"):
            position = int(parts[0][3:])
            arrays[name] = read_raw(os.path.join(dump_dir,
                                    f"pos{position:04d}_head_top1_score.f32"), np.float32, [1])
        elif name.endswith("_route_ids"):
            position, layer = int(parts[0][3:]), int(parts[1][5:])
            raw = read_raw(os.path.join(dump_dir,
                           f"pos{position:04d}_layer{layer:04d}_route_ids.u32"), np.uint32, [10])
            arrays[name] = raw.astype(np.int32)
        elif name.endswith("_route_weights"):
            position, layer = int(parts[0][3:]), int(parts[1][5:])
            arrays[name] = read_raw(os.path.join(dump_dir,
                                    f"pos{position:04d}_layer{layer:04d}_route_weights.f32"),
                                    np.float32, [10])
        elif name.endswith("_streams"):
            position, layer = int(parts[0][3:]), int(parts[1][5:])
            arrays[name] = read_raw(os.path.join(dump_dir,
                                    f"pos{position:04d}_layer{layer:04d}_streams.u16"),
                                    np.uint16, [8192])
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
    args = parser.parse_args()
    prompt_ids = [int(value) for value in args.prompt_ids.split(",") if value != ""]
    arrays = build_arrays(args.reference, args.dump_dir, prompt_ids, args.new_tokens)
    write_fixture(args.output, arrays)
    print(json.dumps({"output": args.output, "arrays": len(arrays),
                      "prompt_count": len(prompt_ids),
                      "new_tokens": args.new_tokens}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
