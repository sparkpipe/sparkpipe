#!/usr/bin/env python3
import json
import math
import random
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import metal_gate_summary as summary

COMPARE = ROOT / "build" / "logit_compare"
VOCAB = 32
POSITIONS = 512


def write_logits(path, rows):
    with open(path, "wb") as handle:
        for row in rows:
            handle.write(struct.pack(f"<{VOCAB}f", *row))


def noisy(base, scale, seed):
    generator = random.Random(seed)
    return [[value + generator.gauss(0.0, scale) for value in row] for row in base]


def compare(work, target, candidate, label, floor):
    result = subprocess.run([str(COMPARE), str(work / "prompts.json"), "synthetic", str(VOCAB), str(work / f"{target}.logits"),
                             str(work / f"{candidate}.logits"), label, "0", str(work / f"{floor}.logits")],
                            capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def main():
    if not COMPARE.exists():
        print("FAIL test_metal_gate_summary: build/logit_compare is missing (make build/logit_compare)")
        return 1
    failures = 0

    def check(what, ok):
        nonlocal failures
        print(("PASS " if ok else "FAIL ") + what)
        failures += 0 if ok else 1

    generator = random.Random(1)
    base = [[generator.gauss(0.0, 2.0) for _ in range(VOCAB)] for _ in range(POSITIONS)]
    for row in base:
        row[generator.randrange(VOCAB)] += 8.0
    with tempfile.TemporaryDirectory() as directory:
        work = Path(directory)
        tokens = [generator.randrange(VOCAB) for _ in range(POSITIONS + 1)]
        (work / "prompts.json").write_text(json.dumps({"prompts": [{"name": "synthetic", "prompt_token_ids": tokens}]}))
        reference_a = noisy(base, 0.01, 11)
        write_logits(work / "a.logits", reference_a)
        write_logits(work / "h.logits", noisy(base, 0.01, 2))
        write_logits(work / "exact.logits", reference_a)
        write_logits(work / "floorlike.logits", noisy(base, 0.01, 3))
        write_logits(work / "control.logits", noisy(reference_a, 0.5, 4))
        rows = [compare(work, "a", "h", "floor", "a")]
        for name in ("exact", "floorlike", "control"):
            rows.append(compare(work, "a", name, name, "h"))
            rows.append(compare(work, "h", name, f"{name}_vs_h", "a"))
        floor = summary.aggregate(rows, "floor")
        check("the synthetic floor agrees on every top-1, so the top-1 ratio divides by a zero floor", floor["top1_agreement"] == 1.0)
        verdicts = {name: summary.verdict(summary.aggregate(rows, name), summary.aggregate(rows, f"{name}_vs_h"), floor)
                    for name in ("exact", "floorlike", "control")}
        exact = verdicts["exact"]["reference_a"]
        check("a candidate identical to reference A passes both legs (paired -inf, ratios 0)",
              verdicts["exact"]["pass"] and exact["paired_sigma"] == -math.inf and exact["kl_ratio_to_floor"] == 0.0 and
              exact["top1_disagreement_ratio_to_floor"] == 0.0)
        check("a candidate with floor-like noise passes both legs", verdicts["floorlike"]["pass"])
        control = verdicts["control"]["reference_a"]
        check("a corrupted control fails both legs (KL, top-1 and paired)",
              not verdicts["control"]["pass"] and not verdicts["control"]["reference_h"]["pass"] and not control["kl_within_factor"] and
              not control["top1_within_factor"] and not control["paired_within_sigma"])
        worse = dict(summary.aggregate(rows, "control"))
        worse["above_floor_blocks"] = [[128, 128] for _ in range(4)]
        worse["above_floor_positions"] = 512
        check("a candidate above the floor at every position has paired +inf and fails", summary.paired(worse)[2] == math.inf)
    for what, candidate in (("one block of positions", {"above_floor_blocks": [[3, 100]], "positions": 100, "above_floor_positions": 3}),):
        try:
            summary.paired(candidate)
            check(f"{what} is refused", False)
        except SystemExit:
            check(f"{what} is refused", True)
    try:
        summary.leg({"kl_mean": 0.1, "floor_kl_mean": None, "top1_agreement": 0.9, "above_floor_blocks": [[1, 128], [1, 128]],
                     "positions": 256, "above_floor_positions": 2}, 0.9)
        check("a compare row without the floor KL is refused", False)
    except SystemExit:
        check("a compare row without the floor KL is refused", True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
