#!/usr/bin/env python3
"""Compare a DSV4.1 stage-runner layer dump with a t1r reference fixture.

The runner model writes, on rank 0 when SPARK_DSV41_FLASH_LAYER_DUMP names a
file, one record per layer per step: u32 magic 'DL4D', layer, rows, top_k,
then the four hyper-connection streams of every row (bf16 [rows][4][5120]),
the routed expert ids (u32 [rows][top_k]) and route weights (f32). The
fixture (tools/t1_reference_decoder.py --family dsv41) holds, per prompt
position and layer, the streams of the sampled layers and the route ids and
weights of every layer. A prompt prefilled as one wave is the first 40
records, row r = position r.

usage: dsv41_flash_layer_compare.py --dump FILE --fixture X.t1r [--rel 0.02]
"""

import argparse
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from t1_reference_common import read_fixture  # noqa: E402

MAGIC = 0x44344C44
STREAMS = 4 * 5120


def records(path: Path):
    data = path.read_bytes()
    offset = 0
    while offset + 16 <= len(data):
        magic, layer, rows, top_k = struct.unpack_from("<4I", data, offset)
        if magic != MAGIC:
            raise SystemExit(f"{path}: bad record magic at byte {offset}")
        offset += 16
        streams = np.frombuffer(data, dtype=np.uint16, count=rows * STREAMS, offset=offset).reshape(rows, STREAMS)
        offset += rows * STREAMS * 2
        routes = np.frombuffer(data, dtype=np.uint32, count=rows * top_k, offset=offset).reshape(rows, top_k)
        offset += rows * top_k * 4
        weights = np.frombuffer(data, dtype=np.float32, count=rows * top_k, offset=offset).reshape(rows, top_k)
        offset += rows * top_k * 4
        yield layer, streams, routes, weights


def bf16(values):
    return (values.astype(np.uint32) << 16).view(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dump", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--rel", type=float, default=0.02)
    args = parser.parse_args()
    _, arrays = read_fixture(str(args.fixture))
    first_bad = None
    for layer, streams, routes, weights in records(args.dump):
        worst, route_miss, compared = 0.0, 0, 0
        for position in range(streams.shape[0]):
            key = f"pos{position:04d}_layer{layer:04d}"
            if key + "_streams" in arrays:
                want = bf16(arrays[key + "_streams"])
                got = bf16(streams[position])
                rel = float(np.linalg.norm(got - want) / max(np.linalg.norm(want), 1e-30))
                worst = max(worst, rel)
                compared += 1
            if key + "_route_ids" in arrays:
                route_miss += int(set(arrays[key + "_route_ids"].tolist()) != set(routes[position].tolist()))
        flag = "ok" if worst <= args.rel and route_miss == 0 else "DIFF"
        if flag == "DIFF" and first_bad is None:
            first_bad = layer
        print(f"layer {layer:2d} rows {streams.shape[0]}: streams compared {compared}, worst rel {worst:.4g}, route-set misses {route_miss} {flag}")
        if layer == 39:
            break
    print("first diverging layer:", first_bad)
    return 0 if first_bad is None else 1


if __name__ == "__main__":
    sys.exit(main())
