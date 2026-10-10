#!/usr/bin/env python3
"""DeepSeek-V4.1-Flash engram hash tables for the stage runner: one small file per deployment.

The engram layers (1 and 14) hash the last 4 compressed token ids into 3 orders x 8
heads = 24 table rows per token. Everything the hash needs is derived here exactly
as tools/t1_reference_dsv41.py derives it from the checkpoint config:
  - the compressed token map (vocab -> compressed id, -1 for blocked tokens), read
    from qualification/t1_reference/dsv41/engram_token_map.npy;
  - the per-layer odd multipliers from numpy's default_rng(10007 * layer);
  - the 2 x 24 distinct primes above the bucket size, and the per-layer row
    offsets (exclusive prefix sums of each layer's primes);
  - the per-layer table row counts (the sum of the layer's primes).

Output (little endian): u32 magic 'D41E', version 1, vocab, columns, pad id;
u64 rows[2]; i64 multipliers[2][4]; i64 primes[2][24]; i64 offsets[2][24];
i32 token_map[vocab].

usage: dsv41_flash_engram_tables.py --config CHECKPOINT/config.json --out engram_tables.bin
"""

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MAGIC = 0x45313444
VERSION = 1


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    factor = 3
    while factor * factor <= n:
        if n % factor == 0:
            return False
        factor += 2
    return True


def next_prime(start: int, seen: set) -> int:
    candidate = start + 1
    while not is_prime(candidate) or candidate in seen:
        candidate += 1
    return candidate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--token-map", type=Path, default=ROOT / "qualification/t1_reference/dsv41/engram_token_map.npy")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    text = config.get("text_config", config)
    layers = list(text["engram_layer_ids"])
    orders = int(text["engram_max_ngram_size"]) - 1
    heads = int(text["engram_n_heads"])
    bucket = int(text["engram_vocab_size"])
    compressed = int(text["engram_compressed_vocab_size"])
    token_map = np.load(args.token_map).astype(np.int64)
    if len(layers) != 2 or orders != 3 or heads != 8 or token_map.shape[0] != int(text["vocab_size"]):
        raise SystemExit("engram geometry differs from the stage runner's (2 layers, 3 orders, 8 heads, full vocab map)")
    if int(token_map.max()) + 1 > compressed:
        raise SystemExit("token map exceeds the compressed vocabulary")
    seen, primes = set(), []
    for _ in layers:
        per_layer = []
        for _ in range(orders):
            current = bucket - 1
            for _ in range(heads):
                current = next_prime(current, seen)
                seen.add(current)
                per_layer.append(current)
        primes.append(per_layer)
    offsets = [[0] + list(np.cumsum(np.array(p, dtype=np.int64))[:-1]) for p in primes]
    rows = [int(sum(p)) for p in primes]
    bound = max(1, (np.iinfo(np.int64).max // compressed) // 2)
    multipliers = []
    for layer in layers:
        values = np.random.default_rng(10007 * layer).integers(low=0, high=bound, size=(orders + 1,), dtype=np.int64)
        multipliers.append([int(v) * 2 + 1 for v in values])
    pad = int(token_map[int(text["engram_pad_token_id"])])
    blob = struct.pack("<5I", MAGIC, VERSION, token_map.shape[0], orders * heads, pad & 0xffffffff)
    blob += struct.pack("<2Q", *rows)
    blob += struct.pack("<8q", *[v for m in multipliers for v in m])
    blob += struct.pack("<48q", *[int(v) for p in primes for v in p])
    blob += struct.pack("<48q", *[int(v) for o in offsets for v in o])
    blob += token_map.astype("<i4").tobytes()
    args.out.write_bytes(blob)
    print(f"{args.out}: rows {rows}, pad {pad}, {len(blob)} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
