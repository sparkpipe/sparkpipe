#!/usr/bin/env python3
"""Summarize a Qwen3.8-27B Metal gate run into one receipt with verdicts.

Reads the outputs tools/qwen38_27b_metal_gates.sh writes and applies the
gate rules: the Metal run's position-weighted mean KL against reference A must
be at most ten times the floor (reference B against reference A, the same
reference with a different reduction order), and its top-1 agreement with
reference A at least 99%; the corrupted-weight control must fail the same
rules. Speed is reported against the measured streaming-read bandwidth.

  qwen38_27b_metal_summary.py --out RUN_DIR --revision REV --prompts PROMPTS_JSON
"""

import argparse
import json
import re
import sys
from pathlib import Path

KL_FACTOR = 10.0
TOP1_MINIMUM = 0.99


def aggregate(rows, label):
    picked = [row for row in rows if row["label"] == label]
    positions = sum(row["positions"] for row in picked)
    return {
        "positions": positions,
        "kl_mean": sum(row["kl_mean"] * row["positions"] for row in picked) / positions,
        "kl_max": max(row["kl_max"] for row in picked),
        "top1_agreement": sum(row["top1_agree"] for row in picked) / positions,
        "target_logprob_delta_mean": sum(row["target_logprob_delta_mean"] * row["positions"] for row in picked) / positions,
        "target_logprob_delta_max": max(row["target_logprob_delta_max"] for row in picked),
        "reference_top1_logprob_delta_max": max(row["reference_top1_logprob_delta_max"] for row in picked),
        "kl_mean_prefill_positions": sum(row["kl_mean_prefill"] * row["prefill_positions"] for row in picked) /
            max(1, sum(row["prefill_positions"] for row in picked)),
        "kl_mean_decode_positions": sum(row["kl_mean_decode"] * (row["positions"] - row["prefill_positions"]) for row in picked) /
            max(1, sum(row["positions"] - row["prefill_positions"] for row in picked)),
        "top1_agreement_where_references_agree": (sum(row["stable_top1_agree"] for row in picked) /
                                                  sum(row["stable_positions"] for row in picked)) if sum(row["stable_positions"] for row in picked) else None,
        "per_prompt": {row["prompt"]: {key: row[key] for key in ("positions", "prefill_positions", "kl_mean", "kl_mean_prefill", "kl_mean_decode",
                                                                  "kl_max", "kl_max_position", "top1_agreement", "target_logprob_delta_max",
                                                                  "stable_positions", "stable_top1_agree")} for row in picked},
    }


def verdict(candidate, floor):
    ratio = candidate["kl_mean"] / floor["kl_mean"]
    return {"kl_ratio_to_floor": ratio, "kl_within_10x_floor": ratio <= KL_FACTOR,
            "top1_at_least_99_percent": candidate["top1_agreement"] >= TOP1_MINIMUM,
            "pass": ratio <= KL_FACTOR and candidate["top1_agreement"] >= TOP1_MINIMUM}


def common_prefix(first, second):
    count = 0
    for a, b in zip(first, second):
        if a != b:
            break
        count += 1
    return count


def greedy_table(out):
    reference = json.loads((out / "ref" / "greedy.ref.json").read_text())["prompts"]
    metal = {item["name"]: item["generated"] for item in json.loads((out / "metal" / "greedy.metal.json").read_text())["prompts"]}
    table = {}
    for item in reference:
        entry = table.setdefault(item["name"], {"metal": metal[item["name"]]})
        entry[f"reference_{item['variant']}"] = item["generated"]
    for entry in table.values():
        entry["metal_matches_reference_a_tokens"] = common_prefix(entry["metal"], entry["reference_a"])
        entry["metal_matches_reference_b_tokens"] = common_prefix(entry["metal"], entry["reference_b"])
        entry["reference_a_matches_b_tokens"] = common_prefix(entry["reference_a"], entry["reference_b"])
    return table


def pairs_table(out):
    single = {item["name"]: item["generated"] for item in json.loads((out / "metal" / "greedy.metal.json").read_text())["prompts"]}
    batched = json.loads((out / "metal" / "pairs.metal.json").read_text())["prompts"]
    return {item["name"]: {"batched_with": item["batched_with"], "identical_to_single_sequence": item["generated"] == single[item["name"]],
                           "rows_per_second": item["rows_per_second"]} for item in batched}


def speed(out):
    bench = json.loads((out / "metal" / "bench.metal.json").read_text())
    bandwidth = json.loads((out / "bandwidth.json").read_text())
    measured = {run["buffer"]: run for run in bandwidth["runs"]}
    peak = measured["device_private"]["median_gb_per_s"] * 1e9
    streamed = bench["streamed_bytes_per_token"]
    decode_ceiling = peak / streamed
    decode = bench["decode"]["median_tokens_per_second"]
    result = {"bandwidth": measured, "streamed_bytes_per_token": streamed,
              "decode": {"prompt_tokens": bench["decode"]["prompt_tokens"], "output_tokens": bench["decode"]["output_tokens"],
                         "median_tokens_per_second": decode, "runs": bench["decode"]["runs"],
                         "roofline_tokens_per_second": decode_ceiling, "fraction_of_roofline": decode / decode_ceiling,
                         "roofline": f"{streamed / 1e9:.2f} GB read per token / {peak / 1e9:.1f} GB/s measured = "
                                     f"{decode_ceiling:.2f} tok/s; measured {decode:.2f} tok/s = {100.0 * decode / decode_ceiling:.1f}%"}}
    for key in ("prefill_1024", "prefill_8192"):
        item = bench[key]
        rows = min(item["chunk_rows"], int(key.split("_")[1]))
        per_token = streamed / rows
        ceiling = peak / per_token
        rate = item["median_tokens_per_second"]
        flops = 2.0 * streamed / 2.0
        result[key] = {"median_tokens_per_second": rate, "runs_seconds": item["runs_seconds"], "chunk_rows": item["chunk_rows"],
                       "weight_bytes_per_token": per_token, "memory_roofline_tokens_per_second": ceiling,
                       "projection_tflops": rate * flops / 1e12,
                       "roofline": f"weights {streamed / 1e9:.2f} GB per {rows}-row chunk = {per_token / 1e6:.1f} MB per token / "
                                   f"{peak / 1e9:.1f} GB/s = {ceiling:.0f} tok/s memory ceiling; measured {rate:.1f} tok/s, "
                                   f"{rate * flops / 1e12:.1f} TFLOPS in the projections, so prefill is compute-bound"}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    arguments = parser.parse_args()
    out = arguments.out
    rows = [json.loads(line) for line in (out / "compare" / "compare.jsonl").read_text().splitlines() if line.strip()]
    floor = aggregate(rows, "floor")
    metal = aggregate(rows, "metal")
    metal_b = aggregate(rows, "metal_vs_b")
    control = aggregate(rows, "control")
    rotate = aggregate(rows, "control_rotate") if any(row["label"] == "control_rotate" for row in rows) else None
    diagnostics = {label: aggregate(rows, label) for label in ("fp32_activations_vs_a", "fp32_activations_vs_metal")
                   if any(row["label"] == label for row in rows)}
    kernels = json.loads((out / "kernels.json").read_text())
    teacher_log = (out / "metal_teacher.log").read_text()
    sha = re.search(r"sha256 ([0-9a-f]{64})", teacher_log)
    prompts = json.loads(arguments.prompts.read_text())
    receipt = {
        "revision": arguments.revision,
        "pack_sha256": sha.group(1) if sha else None,
        "prompts": {item["name"]: {"tokens": item["tokens"], "source": item["source"], "source_sha256": item["source_sha256"]}
                    for item in prompts["prompts"]},
        "device_contract": (out / "device_contract.txt").read_text().strip().splitlines()[-1],
        "kernels": {"cases": len(kernels), "pass": all(case["pass"] for case in kernels),
                    "worst_ratio": max(case["worst_ratio"] for case in kernels if case["worst_ratio"] is not None),
                    "cases_list": [{key: case[key] for key in ("op", "shape", "checked", "worst_ratio", "control_outside", "pass")}
                                   for case in kernels]},
        "numerics": {"rule": f"KL(ref_a || metal) position-weighted mean <= {KL_FACTOR:g} x KL(ref_a || ref_b) and top-1 >= {TOP1_MINIMUM:.0%}",
                     "floor": floor, "metal": metal, "metal_against_reference_b": metal_b,
                     "metal_verdict": verdict(metal, floor), "control": control,
                     "control_corruption": json.loads((out / "control" / "control.json").read_text()),
                     "control_verdict": verdict(control, floor),
                     "control_rotate": rotate,
                     "control_rotate_corruption": json.loads((out / "control_rotate" / "control.json").read_text()) if rotate else None,
                     "control_rotate_verdict": verdict(rotate, floor) if rotate else None,
                     "reference_b_against_a_verdict": verdict(floor, floor),
                     "diagnostics": diagnostics},
        "greedy": greedy_table(out),
        "two_sequence_decode": pairs_table(out),
        "speed": speed(out),
    }
    receipt["numerics"]["control_fails_as_required"] = not receipt["numerics"]["control_verdict"]["pass"]
    receipt["numerics"]["control_swap_detected_by_kl"] = not receipt["numerics"]["control_verdict"]["kl_within_10x_floor"]
    if rotate:
        receipt["numerics"]["control_rotate_detected_by_kl"] = not receipt["numerics"]["control_rotate_verdict"]["kl_within_10x_floor"]
    (out / "RECEIPT.json").write_text(json.dumps(receipt, indent=1) + "\n")
    print(json.dumps({"metal": receipt["numerics"]["metal_verdict"], "control": receipt["numerics"]["control_verdict"],
                      "control_rotate": receipt["numerics"]["control_rotate_verdict"],
                      "reference_b_against_a": receipt["numerics"]["reference_b_against_a_verdict"],
                      "floor_kl": floor["kl_mean"], "metal_kl": metal["kl_mean"], "metal_top1": metal["top1_agreement"],
                      "decode": receipt["speed"]["decode"]["roofline"]}, indent=1))
    return 0 if receipt["numerics"]["metal_verdict"]["pass"] and receipt["numerics"]["control_fails_as_required"] else 1


if __name__ == "__main__":
    sys.exit(main())
