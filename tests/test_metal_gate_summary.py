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
MULTI_POSITIONS = 2048


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
    long_base = [[generator.gauss(0.0, 2.0) for _ in range(VOCAB)] for _ in range(MULTI_POSITIONS)]
    for row in long_base:
        row[generator.randrange(VOCAB)] += 8.0
    with tempfile.TemporaryDirectory() as directory:
        out = Path(directory)
        (out / "compare" / "kl").mkdir(parents=True)
        tokens = [generator.randrange(VOCAB) for _ in range(MULTI_POSITIONS + 1)]
        (out / "prompts.json").write_text(json.dumps({"prompts": [{"name": "synthetic", "prompt_token_ids": tokens}]}))
        logits = {"a": noisy(long_base, 0.2, 21), "b": noisy(long_base, 0.2, 22), "h": noisy(long_base, 0.2, 23),
                  "metal": noisy(long_base, 0.2, 24), "control_layers": noisy(long_base, 1.0, 25), "control_rotate": noisy(long_base, 1.0, 26)}
        for name, rows in logits.items():
            write_logits(out / f"{name}.logits", rows)

        def mf(reference, other):
            result = subprocess.run([str(COMPARE), str(out / "prompts.json"), "synthetic", str(VOCAB), str(out / f"{reference}.logits"),
                                     str(out / f"{other}.logits"), f"mf:{reference}:{other}", "0", "-",
                                     str(out / "compare" / "kl" / f"synthetic.{reference}.{other}.f64")], capture_output=True, text=True, check=True)
            return json.loads(result.stdout)

        rows = [mf(reference, other) for reference in "abh" for other in ("a", "b", "h", "metal", "control_layers", "control_rotate")
                if other != reference]
        check("the per-position KL file holds one double per position",
              (out / "compare" / "kl" / "synthetic.a.metal.f64").stat().st_size == MULTI_POSITIONS * 8)
        floors = summary.multi_floor(rows, out, ["synthetic"])
        check("three reference variants give the multi-floor test", floors is not None and floors["variants"] == ["a", "b", "h"])
        check("a candidate with the variants' own noise passes the multi-floor test",
              floors["metal"]["pass"] and abs(floors["metal"]["paired"]["sigma"]) < 3.0)
        check("corrupted controls fail the multi-floor test on KL and on the paired excess",
              all(not floors[name]["pass"] and floors[name]["paired"]["sigma"] > 3.0 for name in ("control_layers", "control_rotate")))
        check("two reference variants give no multi-floor test",
              summary.multi_floor([row for row in rows if not row["label"].startswith("mf:b:") and ":b" not in row["label"]], out, ["synthetic"]) is None)
        (out / "compare" / "kl-pinned").mkdir()
        for name, seed in (("bp", 27), ("hp", 28)):
            write_logits(out / f"{name}.logits", noisy(long_base, 0.2, seed))

        def pmf(reference, other):
            result = subprocess.run([str(COMPARE), str(out / "prompts.json"), "synthetic", str(VOCAB), str(out / f"{reference}.logits"),
                                     str(out / f"{other}.logits"), f"pmf:{reference}:{other}", "0", "-",
                                     str(out / "compare" / "kl-pinned" / f"synthetic.{reference}.{other}.f64")], capture_output=True, text=True,
                                    check=True)
            return json.loads(result.stdout)

        pinned_rows = [pmf(reference, other) for reference in ("a", "bp", "hp")
                       for other in ("a", "bp", "hp", "metal", "control_layers", "control_rotate") if other != reference]
        pinned = summary.multi_floor(rows + pinned_rows, out, ["synthetic"], "pmf", summary.PINNED_MULTI_FLOOR_VARIANTS, "kl-pinned")
        check("references a, bp and hp give the pinned multi-floor test", pinned is not None and pinned["variants"] == ["a", "bp", "hp"])
        check("the pinned multi-floor test passes a candidate with the variants' noise and fails the corrupted controls",
              pinned["metal"]["pass"] and all(not pinned[name]["pass"] and pinned[name]["paired"]["sigma"] > 3.0
                                              for name in ("control_layers", "control_rotate")))
        check("the free multi-floor test reads only its own rows when pinned rows are present",
              summary.multi_floor(rows + pinned_rows, out, ["synthetic"])["variants"] == ["a", "b", "h"])
        check("without bp there is no pinned multi-floor test",
              summary.multi_floor(rows + [row for row in pinned_rows if ":bp" not in row["label"]], out, ["synthetic"], "pmf",
                                  summary.PINNED_MULTI_FLOOR_VARIANTS, "kl-pinned") is None)
        short = out / "compare" / "kl" / "synthetic.a.metal.f64"
        short.write_bytes(short.read_bytes()[:-8])
        try:
            summary.multi_floor(rows, out, ["synthetic"])
            check("per-position KL files of different lengths are refused", False)
        except SystemExit:
            check("per-position KL files of different lengths are refused", True)
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
