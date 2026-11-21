#!/usr/bin/env python3
"""DeepSeek V4.1 Flash expert placement (tools/dsv41_flash_expert_placement.py).

A placement built from routed counts must hold every expert exactly once per layer, 24 a
rank over 16 ranks, list each rank's experts in ascending order, come out the same from the
same counts, and put no more routes on its busiest rank than the contiguous layout does;
on skewed counts it must cut the busiest rank's routes. A layer without counts keeps the
contiguous layout. The sidecar binary carries the header and every id in layer, rank, slot
order, and check refuses a placement that holds an expert twice.
"""

import json
import random
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

TOOLS = Path(__file__).resolve().parent.parent / "tools"
sys.path.insert(0, str(TOOLS))
import dsv41_flash_expert_placement as place


def busiest(layer_counts, groups):
    return max(sum(layer_counts[e] for e in group) for group in groups)


def main() -> int:
    rng = random.Random(20261116)
    counts = {}
    for layer in range(0, 20):
        values = [rng.randint(50, 300) for _ in range(place.ROUTED_EXPERTS)]
        for hot in rng.sample(range(48), 8):
            values[hot] += 20000
        counts[str(layer)] = values
    with tempfile.TemporaryDirectory() as root:
        counts_path, out_a, out_b, binary = (Path(root) / name for name in ("counts.json", "a.json", "b.json", "a.bin"))
        counts_path.write_text(json.dumps({"layers": counts}))
        for out in (out_a, out_b):
            done = subprocess.run([sys.executable, str(TOOLS / "dsv41_flash_expert_placement.py"), "place", str(counts_path), str(out), "16"],
                                  capture_output=True, text=True)
            assert done.returncode == 0, done.stderr
        assert out_a.read_bytes() == out_b.read_bytes(), "the same counts gave two placements"
        placement = json.loads(out_a.read_text())
        place.check(placement)
        contiguous = place.contiguous(16)
        for layer in range(place.LAYER_COUNT):
            groups = placement["layers"][layer]
            assert all(group == sorted(group) for group in groups)
            if str(layer) not in counts:
                assert groups == contiguous, layer
                continue
            assert busiest(counts[str(layer)], groups) < busiest(counts[str(layer)], contiguous) * 0.6, layer
        place.cmd_bin(str(out_a), str(binary))
        data = binary.read_bytes()
        assert struct.unpack_from("<IIIII", data, 0) == (place.MAGIC, place.VERSION, place.LAYER_COUNT, 16, 24)
        assert list(struct.unpack_from(f"<{place.LAYER_COUNT * place.ROUTED_EXPERTS}H", data, 20)) == [
            expert for layer in placement["layers"] for group in layer for expert in group]
        broken = json.loads(out_a.read_text())
        broken["layers"][3][0][0] = broken["layers"][3][1][0]
        broken_path = Path(root) / "broken.json"
        broken_path.write_text(json.dumps(broken))
        refused = subprocess.run([sys.executable, str(TOOLS / "dsv41_flash_expert_placement.py"), "check", str(broken_path)],
                                 capture_output=True, text=True)
        assert refused.returncode != 0 and "exactly once" in refused.stderr, refused.stderr
    print("PASS dsv41 expert placement: every expert once a layer, 24 a rank, ascending, deterministic, hot experts spread "
          "(busiest rank under 60% of contiguous), contiguous without counts, sidecar ids in order, duplicates refused")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
