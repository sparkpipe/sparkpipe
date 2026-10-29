#!/usr/bin/env python3
"""Summarize a Metal stage gate run into one receipt with verdicts.

Reads the outputs a model's gate script (tools/qwen38_27b_metal_gates.sh,
tools/metal_gates.sh) writes and applies the
gate rule against both reference variants (A and B, the same reference with
different reduction orders), each leg measured against the other variant as
its floor: the position-weighted mean KL at most KL_FACTOR times the floor's,
the top-1 disagreement rate at most TOP1_FACTOR times the floor's, and the
paired test (the share of positions where the candidate is further from the
reference than the other variant is, which is one half for a candidate whose
error is distributed like the floor's) not above one half by more than
PAIRED_SIGMA standard errors, the errors taken over contiguous blocks of
positions. The Metal run must pass both legs; the all-layer and whole-tensor
corrupted-weight controls must fail (and, where the gate runs it, the
control that swaps two experts in every layer), and greedy decoding with lookup
speculation must reproduce greedy decoding without it token for token. Speed
is reported against the measured streaming-read bandwidth.

A mixture-of-experts gate adds a pinned-route set (labels prefixed pinned_):
the Metal run and the controls with every layer's experts taken from
reference A's choice, against A and against B pinned the same way (bp), with
A against bp as the floor. The same rule applies to it; the Metal runs
(metal, and metal_rows64 when present: the same pass prefilled in 64-row
chunks) must pass every set present, and a required control must fail in
every set it appears in. A batch ladder run with --exact-rows (exact/) must
give every batched sequence the tokens of its solo run, and no ladder (plain
or exact rows) may give one batch different tokens across its repeats.

  metal_gate_summary.py --out RUN_DIR --revision REV --prompts PROMPTS_JSON
"""

import argparse
import json
import re
import sys
from pathlib import Path

import math

KL_FACTOR = 2.0
TOP1_FACTOR = 1.5
PAIRED_SIGMA = 3.0
REQUIRED_CONTROLS = ("control_layers", "control_rotate")
REQUIRED_WHERE_RUN = ("control_experts",)
PROBE_CONTROLS = ("control",)
CANDIDATES = ("metal", "metal_rows64")


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
        "floor_kl_mean": (sum(row["floor_kl_mean"] * row["positions"] for row in picked) / positions) if all("floor_kl_mean" in row for row in picked) else None,
        "above_floor_positions": sum(row.get("above_floor_positions", 0) for row in picked),
        "above_floor_blocks": [block for row in picked for block in row.get("above_floor_blocks", [])],
        "per_prompt": {row["prompt"]: {key: row[key] for key in ("positions", "prefill_positions", "kl_mean", "kl_mean_prefill", "kl_mean_decode",
                                                                  "kl_max", "kl_max_position", "top1_agreement", "target_logprob_delta_max",
                                                                  "stable_positions", "stable_top1_agree")} for row in picked},
    }


def ratio(value, floor):
    if floor > 0.0:
        return value / floor
    return 0.0 if value <= 0.0 else math.inf


def paired(candidate):
    blocks = candidate["above_floor_blocks"]
    positions = candidate["positions"]
    if len(blocks) < 2:
        raise SystemExit(f"paired test: {len(blocks)} block(s) of positions; the block standard error needs at least two")
    share = candidate["above_floor_positions"] / positions
    spread = sum((above - share * count) ** 2 for above, count in blocks)
    error = math.sqrt(len(blocks) / (len(blocks) - 1) * spread) / positions
    if error > 0.0:
        return share, error, (share - 0.5) / error
    return share, error, -math.inf if share <= 0.5 else math.inf


def leg(candidate, floor_top1):
    if candidate["floor_kl_mean"] is None:
        raise SystemExit("gate legs: a candidate compare row has no floor KL (the second reference variant was not passed)")
    kl_ratio = ratio(candidate["kl_mean"], candidate["floor_kl_mean"])
    top1_ratio = ratio(1.0 - candidate["top1_agreement"], 1.0 - floor_top1)
    share, error, sigma = paired(candidate)
    checks = {"kl_within_factor": kl_ratio <= KL_FACTOR, "top1_within_factor": top1_ratio <= TOP1_FACTOR,
              "paired_within_sigma": sigma <= PAIRED_SIGMA}
    return {"kl_ratio_to_floor": kl_ratio, "top1_disagreement_ratio_to_floor": top1_ratio, "paired_share_above_floor": share,
            "paired_standard_error": error, "paired_sigma": sigma, "blocks": len(candidate["above_floor_blocks"]), **checks,
            "pass": all(checks.values())}


def verdict(against_a, against_b, floor):
    legs = {"reference_a": leg(against_a, floor["top1_agreement"]), "reference_b": leg(against_b, floor["top1_agreement"])}
    return {**legs, "pass": all(item["pass"] for item in legs.values())}


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


def ladder_table(out, peak, run="metal"):
    ladder = json.loads((out / run / "ladder.metal.json").read_text())
    ceiling = peak / ladder["streamed_bytes_per_token"]
    return {"prompt_tokens": ladder["prompt_tokens"], "decode_steps": ladder["decode_steps"], "repeats": ladder["repeats"],
            "single_sequence_roofline_tokens_per_second": ceiling,
            "rows": [{"batch": item["batch"], "step_ms": item["step_ms"], "per_sequence_tokens_per_second": item["per_sequence_tokens_per_second"],
                      "aggregate_tokens_per_second": item["aggregate_tokens_per_second"],
                      "fraction_of_weight_stream_roofline": item["per_sequence_tokens_per_second"] / ceiling,
                      "sequences_identical_to_solo": item["sequences_identical_to_solo"], "divergences": item["divergences"],
                      "nondeterministic": item["nondeterministic"], "runs_seconds": item["runs_seconds"]} for item in ladder["batches"]]}


def speed(out):
    bench = json.loads((out / "metal" / "bench.metal.json").read_text())
    bandwidth = json.loads((out / "bandwidth.json").read_text())
    measured = {run["buffer"]: run for run in bandwidth["runs"]}
    peak = measured["device_private"]["median_gb_per_s"] * 1e9
    weights = bench["streamed_bytes_per_token"]
    streamed = bench["decode"].get("bytes_per_token_mean", weights)
    decode_ceiling = peak / streamed
    decode = bench["decode"]["median_tokens_per_second"]
    result = {"bandwidth": measured, "streamed_bytes_per_token": streamed, "weight_bytes_per_token": weights,
              "decode": {"prompt_tokens": bench["decode"]["prompt_tokens"], "output_tokens": bench["decode"]["output_tokens"],
                         "median_tokens_per_second": decode, "runs": bench["decode"]["runs"],
                         "roofline_tokens_per_second": decode_ceiling, "fraction_of_roofline": decode / decode_ceiling,
                         "roofline": f"{streamed / 1e9:.2f} GB read per token ({weights / 1e9:.2f} GB of weights, the rest KV and recurrent state) / "
                                     f"{peak / 1e9:.1f} GB/s measured = "
                                     f"{decode_ceiling:.2f} tok/s; measured {decode:.2f} tok/s = {100.0 * decode / decode_ceiling:.1f}%"}}
    for key in ("prefill_1024", "prefill_8192"):
        item = bench[key]
        rows = min(item["chunk_rows"], int(key.split("_")[1]))
        chunk = bench.get("streamed_bytes_per_chunk", weights)
        per_token = chunk / rows
        ceiling = peak / per_token
        rate = item["median_tokens_per_second"]
        flops = 2.0 * weights / 2.0
        result[key] = {"median_tokens_per_second": rate, "runs_seconds": item["runs_seconds"], "chunk_rows": item["chunk_rows"],
                       "weight_bytes_per_token": per_token, "memory_roofline_tokens_per_second": ceiling,
                       "projection_tflops": rate * flops / 1e12,
                       "roofline": f"weights {chunk / 1e9:.2f} GB per {rows}-row chunk = {per_token / 1e6:.1f} MB per token / "
                                   f"{peak / 1e9:.1f} GB/s = {ceiling:.0f} tok/s memory ceiling; measured {rate:.1f} tok/s, "
                                   f"{rate * flops / 1e12:.1f} TFLOPS in the projections"}
    return result


def numerics_set(rows, prefix, out):
    labels = {row["label"] for row in rows}
    floor = aggregate(rows, prefix + "floor")
    candidates = {name: (aggregate(rows, prefix + name), aggregate(rows, f"{prefix}{name}_vs_b")) for name in CANDIDATES + REQUIRED_CONTROLS + REQUIRED_WHERE_RUN + PROBE_CONTROLS
                  if prefix + name in labels and f"{prefix}{name}_vs_b" in labels}
    result = {"floor": floor,
              **{name: {"against_reference_a": pair[0], "against_reference_b": pair[1], "verdict": verdict(pair[0], pair[1], floor),
                        "corruption": json.loads((out / name / "control.json").read_text()) if name not in CANDIDATES else None}
                 for name, pair in candidates.items()}}
    return candidates, result


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--tp-degree", type=int, default=1)
    arguments = parser.parse_args()
    out = arguments.out
    rows = [json.loads(line) for line in (out / "compare" / "compare.jsonl").read_text().splitlines() if line.strip()]
    labels = {row["label"] for row in rows}
    candidates, free = numerics_set(rows, "", out)
    pinned_candidates, pinned = numerics_set(rows, "pinned_", out) if "pinned_floor" in labels else ({}, None)
    diagnostics = {label: aggregate(rows, label) for label in ("fp32_activations_vs_a", "fp32_activations_vs_metal")
                   if any(row["label"] == label for row in rows)}
    kernels = json.loads((out / "kernels.json").read_text())
    teacher_log = (out / "metal_teacher.log").read_text()
    sha = re.search(r"sha256 ([0-9a-f]{64})", teacher_log)
    prompts = json.loads(arguments.prompts.read_text())
    receipt = {
        "revision": arguments.revision,
        "tp_degree": arguments.tp_degree,
        "pack_sha256": sha.group(1) if sha else None,
        "prompts": {item["name"]: {"tokens": item["tokens"], "source": item["source"], "source_sha256": item["source_sha256"]}
                    for item in prompts["prompts"]},
        "device_contract": (out / "device_contract.txt").read_text().strip().splitlines()[-1],
        "kernels": {"cases": len(kernels), "pass": all(case["pass"] for case in kernels),
                    "worst_ratio": max(case["worst_ratio"] for case in kernels if case["worst_ratio"] is not None),
                    "cases_list": [{key: case[key] for key in ("op", "shape", "checked", "worst_ratio", "control_outside", "pass")}
                                   for case in kernels]},
        "numerics": {"rule": (f"against each reference variant, with the other variant as the floor: mean KL <= {KL_FACTOR:g} x the floor's, "
                              f"top-1 disagreement <= {TOP1_FACTOR:g} x the floor's, and the share of positions further from the reference than "
                              f"the floor at most 0.5 + {PAIRED_SIGMA:g} block standard errors; Metal passes both legs, "
                              f"{' and '.join(REQUIRED_CONTROLS)} fail"),
                     "diagnostics": diagnostics, **free},
        "greedy": greedy_table(out) if (out / "ref" / "greedy.ref.json").exists() else None,
        "speed": speed(out) if (out / "metal" / "bench.metal.json").exists() else None,
    }
    ladder_deterministic = None
    if (out / "metal" / "ladder.metal.json").exists() and receipt["speed"] is not None:
        receipt["batch_ladder"] = ladder_table(out, receipt["speed"]["bandwidth"]["device_private"]["median_gb_per_s"] * 1e9)
        ladder_deterministic = all(row["nondeterministic"] == 0 for row in receipt["batch_ladder"]["rows"])
        receipt["batch_ladder"]["deterministic"] = ladder_deterministic
    exact_identical = None
    if (out / "exact" / "ladder.metal.json").exists() and receipt["speed"] is not None:
        receipt["batch_ladder_exact_rows"] = ladder_table(out, receipt["speed"]["bandwidth"]["device_private"]["median_gb_per_s"] * 1e9, "exact")
        exact_identical = all(row["sequences_identical_to_solo"] == row["batch"] and row["nondeterministic"] == 0
                              for row in receipt["batch_ladder_exact_rows"]["rows"])
        receipt["batch_ladder_exact_rows"]["identical_to_solo"] = exact_identical
    if (out / "metal" / "spec.metal.json").exists():
        speculation = json.loads((out / "metal" / "spec.metal.json").read_text())
        receipt["speculation"] = {key: speculation[key] for key in ("new_tokens", "draft_max", "repeats", "lookup", "verify_frames")}
        receipt["speculation"]["prompts"] = [{key: item[key] for key in ("name", "prompt_tokens", "tokens_identical", "plain_tokens_per_second",
                                                                          "spec_tokens_per_second", "frames", "drafted", "accepted")}
                                             for item in speculation["prompts"]]
        receipt["speculation"]["exact"] = speculation["prompts_identical"] == speculation["prompt_count"]
    numerics = receipt["numerics"]
    sets = [numerics] + ([pinned] if pinned is not None else [])
    if pinned is not None:
        numerics["pinned_routes"] = pinned
    numerics["controls_fail_as_required"] = all(any(name in item for item in sets) and
                                                all(not item[name]["verdict"]["pass"] for item in sets if name in item) for name in REQUIRED_CONTROLS) and \
        all(not item[name]["verdict"]["pass"] for name in REQUIRED_WHERE_RUN for item in sets if name in item)
    numerics["pass"] = all("metal" in item and all(item[name]["verdict"]["pass"] for name in CANDIDATES if name in item) for item in sets) and \
        numerics["controls_fail_as_required"]
    (out / "RECEIPT.json").write_text(json.dumps(receipt, indent=1) + "\n")
    speculation_exact = receipt["speculation"]["exact"] if "speculation" in receipt else None
    print(json.dumps({"numerics_pass": numerics["pass"], "speculation_exact": speculation_exact, "ladder_deterministic": ladder_deterministic,
                      "ladder_exact_rows_identical": exact_identical,
                      **{name: {leg_name: {key: item[key] for key in ("kl_ratio_to_floor", "top1_disagreement_ratio_to_floor", "paired_sigma", "pass")}
                                for leg_name, item in numerics[name]["verdict"].items() if leg_name != "pass"}
                         for name in candidates},
                      **{"pinned_" + name: {leg_name: {key: item[key] for key in ("kl_ratio_to_floor", "top1_disagreement_ratio_to_floor", "paired_sigma",
                                                                                     "pass")}
                                            for leg_name, item in pinned[name]["verdict"].items() if leg_name != "pass"}
                         for name in pinned_candidates},
                      "decode": receipt["speed"]["decode"]["roofline"] if receipt["speed"] is not None else None}, indent=1))
    return 0 if numerics["pass"] and speculation_exact is not False and ladder_deterministic is not False and exact_identical is not False else 1

if __name__ == "__main__":
    sys.exit(main())
