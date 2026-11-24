#!/usr/bin/env python3
"""DeepSeek V4.1 Flash: which rank holds which whole routed expert, per layer.

subcommands:
  place COUNTS_JSON OUT_JSON [RANKS]
      COUNTS_JSON is {"layers": {"<layer>": [routes per expert, 384 numbers]}}. Every layer
      with counts is placed busiest expert first onto the rank with the fewest routes that
      still has a free slot (ties go to the lower expert and the lower rank), 384 / RANKS
      experts a rank; a layer without counts keeps rank r on experts [r*E/RANKS, (r+1)*E/RANKS).
      Each rank's experts are listed in ascending order.
  check PLACEMENT_JSON
      every layer gives every rank 384 / RANKS distinct experts and covers all 384.
  bin PLACEMENT_JSON OUT_BIN
      the stage's sidecar: magic, version, layers, ranks, experts a rank, then each layer's
      ranks' expert ids as little-endian uint16.
Exit 0 = done; nonzero = the failure is named on stderr.
"""

import json
import struct
import sys

LAYER_COUNT = 40
ROUTED_EXPERTS = 384
FORMAT = "dsv41-flash-expert-placement-v1"
MAGIC = 0x50313444
VERSION = 1


def fail(message):
    print(f"dsv41_flash_expert_placement: {message}", file=sys.stderr)
    sys.exit(1)


def contiguous(ranks):
    local = ROUTED_EXPERTS // ranks
    return [list(range(rank * local, (rank + 1) * local)) for rank in range(ranks)]


def balanced(counts, ranks):
    local = ROUTED_EXPERTS // ranks
    load, members = [0] * ranks, [[] for _ in range(ranks)]
    for expert in sorted(range(ROUTED_EXPERTS), key=lambda e: (-counts[e], e)):
        rank = min((r for r in range(ranks) if len(members[r]) < local), key=lambda r: (load[r], r))
        load[rank] += counts[expert]
        members[rank].append(expert)
    return [sorted(group) for group in members]


def check(placement):
    if placement.get("format") != FORMAT:
        fail("not a placement file")
    ranks, local, layers = placement["ranks"], placement["experts_per_rank"], placement["layers"]
    if ranks * local != ROUTED_EXPERTS or len(layers) != LAYER_COUNT:
        fail(f"{ranks} ranks of {local} experts over {len(layers)} layers is not {ROUTED_EXPERTS} experts over {LAYER_COUNT} layers")
    for index, layer in enumerate(layers):
        if len(layer) != ranks or any(len(group) != local for group in layer):
            fail(f"layer {index} does not give every rank {local} experts")
        seen = sorted(expert for group in layer for expert in group)
        if seen != list(range(ROUTED_EXPERTS)):
            fail(f"layer {index} does not hold every expert exactly once")
    return ranks, local


def cmd_place(counts_path, out_path, ranks):
    if ROUTED_EXPERTS % ranks != 0:
        fail(f"{ROUTED_EXPERTS} experts do not split over {ranks} ranks")
    counts = json.load(open(counts_path))["layers"]
    layers = []
    for layer in range(LAYER_COUNT):
        values = counts.get(str(layer))
        if values is not None and len(values) != ROUTED_EXPERTS:
            fail(f"layer {layer} has {len(values)} counts, not {ROUTED_EXPERTS}")
        layers.append(balanced(values, ranks) if values is not None else contiguous(ranks))
    placement = {"format": FORMAT, "ranks": ranks, "experts_per_rank": ROUTED_EXPERTS // ranks, "layers": layers,
                 "placed_layers": sorted(int(layer) for layer in counts)}
    check(placement)
    with open(out_path, "w") as out:
        json.dump(placement, out)
        out.write("\n")


def cmd_bin(placement_path, out_path):
    placement = json.load(open(placement_path))
    ranks, local = check(placement)
    with open(out_path, "wb") as out:
        out.write(struct.pack("<IIIII", MAGIC, VERSION, LAYER_COUNT, ranks, local))
        for layer in placement["layers"]:
            for group in layer:
                out.write(struct.pack(f"<{local}H", *group))


def main(argv):
    if len(argv) >= 3 and argv[0] == "place":
        cmd_place(argv[1], argv[2], int(argv[3]) if len(argv) > 3 else 16)
    elif len(argv) == 2 and argv[0] == "check":
        check(json.load(open(argv[1])))
    elif len(argv) == 3 and argv[0] == "bin":
        cmd_bin(argv[1], argv[2])
    else:
        print(__doc__, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
