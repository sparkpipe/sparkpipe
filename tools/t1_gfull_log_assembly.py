import argparse
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from t1_reference_common import read_fixture, write_fixture


STREAM_RE = re.compile(
    r"^G52-T1 stream L(\d+) pos(\d+) ((?:[0-9a-f]{4})(?: [0-9a-f]{4})*)$")
ROUTE_RE = re.compile(
    r"^G52-T1 route L(\d+) pos(\d+) ids((?: \d+){8}) weights"
    r"((?: [0-9a-f]{8}){8})$")
HEAD_RE = re.compile(
    r"^G52-T1 head pos(\d+) token (\d+) score_bits ([0-9a-f]{8})$")


def parse_log(path):
    streams = {}
    routes = {}
    heads = {}
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            match = STREAM_RE.match(line)
            if match:
                layer = int(match.group(1))
                position = int(match.group(2))
                values = np.frombuffer(
                    bytes.fromhex(match.group(3)), dtype=np.uint16)
                streams[(position, layer)] = values
                continue
            match = ROUTE_RE.match(line)
            if match:
                layer = int(match.group(1))
                position = int(match.group(2))
                ids = np.array([int(v) for v in match.group(3).split()],
                               dtype=np.int32)
                weights = np.frombuffer(
                    bytes.fromhex(match.group(4).replace(" ", "")),
                    dtype=np.float32)
                routes[(position, layer)] = (ids, weights)
                continue
            match = HEAD_RE.match(line)
            if match:
                heads[int(match.group(1))] = (
                    int(match.group(2)),
                    np.frombuffer(bytes.fromhex(match.group(3)),
                                  dtype=np.float32)[0])
    return streams, routes, heads


def expected_names(reference_dir, prompt_name):
    _meta, arrays = read_fixture(
        os.path.join(reference_dir, f"{prompt_name}.t1r"))
    return sorted(arrays)


def assemble(reference_dir, prompt_name, log_path, output_path):
    prompt_ids = None
    streams, routes, heads = parse_log(log_path)
    arrays = {}
    for name in expected_names(reference_dir, prompt_name):
        if name == "prompt_token_ids":
            continue
        if name == "generated_token_ids":
            continue
        if name.startswith("pos") and name.endswith("_streams"):
            head_part, layer_part = name[3:-8].split("_layer")
            key = (int(head_part), int(layer_part))
            if key not in streams:
                raise SystemExit(f"missing stream tap {key} for {prompt_name}")
            arrays[name] = streams[key]
        elif name.endswith("_route_ids"):
            head_part, layer_part = name[3:-10].split("_layer")
            key = (int(head_part), int(layer_part))
            if key not in routes:
                raise SystemExit(f"missing route tap {key} for {prompt_name}")
            arrays[name] = routes[key][0]
        elif name.endswith("_route_weights"):
            head_part, layer_part = name[3:-14].split("_layer")
            key = (int(head_part), int(layer_part))
            if key not in routes:
                raise SystemExit(f"missing route tap {key} for {prompt_name}")
            arrays[name] = routes[key][1]
        elif name.endswith("_head_top1_token"):
            position = int(name[3:-16])
            if position not in heads:
                raise SystemExit(f"missing head tap pos {position}")
            arrays[name] = np.array([heads[position][0]], dtype=np.int32)
        elif name.endswith("_head_top1_score"):
            position = int(name[3:-16])
            if position not in heads:
                raise SystemExit(f"missing head tap pos {position}")
            arrays[name] = np.array([heads[position][1]], dtype=np.float32)
        else:
            raise SystemExit(f"unrecognized fixture array {name}")
    write_fixture(output_path, arrays)
    print(json.dumps({"prompt": prompt_name, "output": output_path,
                      "arrays": len(arrays)}))
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-dir", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    return assemble(arguments.reference_dir, arguments.prompt,
                    arguments.log, arguments.output)


if __name__ == "__main__":
    raise SystemExit(main())
