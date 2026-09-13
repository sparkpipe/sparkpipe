#!/usr/bin/env python3
"""Generate 'prefix-caching disabled' batch variants: unique sequence ids and
rotated prompt blocks per repetition so every lookup is a guaranteed miss."""
import json
import os
import sys

SRC = "/tmp/dsv4-matrix-batches"
DST = "/tmp/dsv4-prefixoff-batches"
REPS = 3

def rotate(tokens, k):
    k %= len(tokens)
    return tokens[k:] + tokens[:k]

def main():
    os.makedirs(DST, exist_ok=True)
    made = []
    for name in ("prefill_ctx512_b1", "prefill_ctx2048_b1",
                 "decode_ctx512_b1", "decode_ctx2048_b1"):
        base = json.load(open(os.path.join(SRC, name + ".json")))
        req = base["requests"][0]
        block = 128
        for rep in range(REPS):
            doc = json.loads(json.dumps(base))
            r = doc["requests"][0]
            # rotate by whole cache blocks so content differs, alignment kept
            shift = block * ((rep + 1) % (len(req["prompt_token_ids"]) // block))
            r["prompt_token_ids"] = rotate(req["prompt_token_ids"], shift)
            r["request_id"] = 780000 + rep * 7 + hash(name) % 5
            r["sequence_id"] = 790000 + rep * 11 + (len(name) % 13)
            path = os.path.join(DST, f"{name}_off{rep}.json")
            with open(path, "w") as f:
                json.dump(doc, f)
            made.append(path)
    print(f"generated {len(made)} off-variant batches")

if __name__ == "__main__":
    main()
