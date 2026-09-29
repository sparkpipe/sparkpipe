#!/usr/bin/env python3
"""End-to-end dry run of the A/B analysis on synthetic dumps (design §7 L2 "done").

Builds a synthetic corpus, a reference arm and its A/A twin, an anchor, one
close 4-bit arm, one bad arm, a spine bridge and two KV arms (one that holds,
one that degrades past 4k), writes merged dumps, exact-KL files, arm
descriptors, receipts, suite archives and a frozen plan, then runs
ab_score_compare, ab_suite_compare, ab_receipt, ab_verdict and prints the §5
report for every comparison. Nothing touches a GPU or a node. All numbers are
synthetic and the report header says so.

usage:
  ab_dry_run.py OUT_DIR [--replicates N]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ab_arm  # noqa: E402
import ab_dump  # noqa: E402
import ab_plan  # noqa: E402
import ab_receipt  # noqa: E402
import ab_report  # noqa: E402
import ab_score_compare  # noqa: E402
import ab_stats  # noqa: E402
import ab_suite_compare  # noqa: E402
import ab_verdict  # noqa: E402

VOCAB = 512
PROBES = 64
RANKS = 4
COMMIT = "1" * 40
CAMPAIGN = "dryrun-synthetic"


def hexid(*parts) -> str:
    return hashlib.sha256("/".join(str(part) for part in parts).encode()).hexdigest()


def make_corpus(rng) -> tuple:
    documents, rows = [], []
    strata = ["prose-pre"] * 16 + ["prose-post"] * 16 + ["code"] * 12 + ["on-policy"] * 8 + ["long"] * 32
    for index, stratum in enumerate(strata):
        if stratum == "long":
            positions = np.sort(rng.choice(np.arange(1, 20000), size=96, replace=False))
        else:
            positions = np.arange(1, 161)
        documents.append({"id": f"synthetic-{index:03d}", "stratum": stratum, "offset": 0, "length": int(positions[-1]) + 1})
        for position in positions:
            rows.append((index, int(position)))
    tokens = hexid("corpus-tokens")
    return {"format": "sparkpipe-ab-corpus-v1", "name": "SYNTHETIC", "corpus_sha256": tokens, "tokenizer_sha256": hexid("tok"),
            "documents": documents}, np.array(rows, dtype=np.int64)


def arm_descriptor(model, frame, expert, label, producer, latent="bf16", index="bf16", state="fp32", group=0, mode="store", pack_tag="") -> dict:
    arm = {
        "format": ab_arm.FORMAT, "arm_id": "", "model": model, "revision": f"org/Synthetic-{frame}@1",
        "topology": {"tp": RANKS, "pp": 1, "kv_shard": 1},
        "spine": {"frame": frame, "source": f"org/Synthetic-{frame}@1", "spine_digest": [hexid("spine", frame, r) for r in range(RANKS)]},
        "expert": {"codec": expert, "label": label, "producer": producer, "source": f"org/Synthetic-{label}@1", "recipe_sha256": None},
        "kv": {"latent": latent, "index": index, "state": state, "group": group, "mode": mode},
        "drafter": {"kind": "none", "label": "none", "codec": "none", "head": "none", "sidecar_sha256": []},
        "pack_sha256": [hexid("pack", frame, label, pack_tag, r) for r in range(RANKS)],
        "artifacts": {"module_archive_sha256": hexid("archive", label, pack_tag), "driver_sha256": hexid("driver", label, pack_tag), "adapter_sha256": hexid("adapter", label)},
    }
    arm["arm_id"] = ab_arm.expected_id(arm)
    return ab_arm.validate(arm)


def receipt(arm: dict, run_label: str, merged_sha: str, tokens_sha: str, probe_sha, corpus: dict, plan_sha: str, flags: dict, order="corpus") -> dict:
    root = f"/home/spark/ab/{run_label}"
    return {
        "format": "sparkpipe-ab-receipt-v1", "campaign": CAMPAIGN, "plan_sha256": plan_sha, "arm": arm,
        "arm_digest": ab_arm.arm_digest(arm), "run_label": run_label, "source_commit": COMMIT,
        "build": {"flags": flags, "model_revision": arm["revision"], "contract_sha256": hexid("contract", arm["spine"]["frame"])},
        "packs": {"pack_sha256": arm["pack_sha256"], "sha256sums_sha256": [hexid("sums", run_label, r) for r in range(RANKS)],
                  "experts_sha256": [hexid("experts", run_label, r) for r in range(RANKS)], "graft_receipt_sha256": hexid("graft", run_label),
                  "verify_receipt_sha256": hexid("verify", run_label)},
        "topology": {"tp": RANKS, "pp": 1, "kv_shard": 1, "rank_nodes": [f"node{r}" for r in range(RANKS)]},
        "weightd": {"lane": 2, "expert_pool_bytes": 1 << 34, "residency": "pinned"},
        "execution": {"mode": "graph", "dropin_sha256": hexid("dropin"), "expert_paths": ["skinny"]},
        "wave": {"sequential": True, "max_prefill_rows_per_submission": 256, "output_token_budget": 1},
        "ready_event": {"event": "ready", "arm_digest": ab_arm.arm_digest(arm), "pack_set_sha256": ab_arm.pack_set_sha256(arm),
                        "linear_weight_codec": 1, "expert_weight_codec": 5, "kv_cache_codec": 1, "model_revision": arm["revision"]},
        "requests": {"count": len(corpus["documents"]), "cached_prompt_tokens": [0] * len(corpus["documents"]), "generated_token_ids_sha256": tokens_sha},
        "inputs": {"corpus": "SYNTHETIC", "corpus_tokens_sha256": corpus["corpus_sha256"], "corpus_index_sha256": hexid("index"),
                   "tokenizer_sha256": corpus["tokenizer_sha256"], "probe_sha256": probe_sha, "document_order": order},
        "dumps": {"per_rank_sha256": [hexid("dump", run_label, r) for r in range(RANKS)], "merged_sha256": merged_sha, "score_dump_on": True},
        "cache": {"arm_root": root, "kv_snapshot_directory": f"{root}/kv-snapshots", "fresh_engine": True, "snapshot_directory_empty_at_start": True},
        "memory": {"nodes": [{"node": f"node{r}", "mem_available_before_gib": 80.0, "mem_available_after_gib": 57.5} for r in range(RANKS)],
                   "pool_bulk_span_line": None},
        "window": {"id": "dryrun", "utc_start": "2026-09-29T00:00:00Z", "utc_end": "2026-09-29T00:10:00Z", "cotenants": []},
    }


def write_suite(directory: Path, answers: list) -> None:
    (directory / "responses").mkdir(parents=True, exist_ok=True)
    integrity = {}
    for index, (answer, expected, tokens) in enumerate(answers):
        record = {"id": f"case-{index:03d}", "request": {}, "response": {"tokens": tokens, "status": 0},
                  "decoded_text": f"Answer: {answer}", "grade": {"source": "COMPSEC", "expected": expected, "passed": answer == expected}}
        relative = f"responses/{index:03d}.json"
        (directory / relative).write_text(json.dumps(record, sort_keys=True))
        integrity[relative] = hashlib.sha256((directory / relative).read_bytes()).hexdigest()
    (directory / "INTEGRITY.json").write_text(json.dumps({"file_sha256": integrity}, sort_keys=True))


def run(out: Path, replicates: int) -> str:
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260929)
    corpus, rows = make_corpus(rng)
    (out / "corpus.json").write_text(json.dumps(corpus, sort_keys=True))
    doc, pos = rows[:, 0].astype(np.uint32), rows[:, 1].astype(np.uint32)
    reference_logits = rng.normal(0.0, 1.0, size=(len(rows), VOCAB)) * 2.5
    reference_logits[:, :8] += 4.0
    winners = rng.integers(0, VOCAB, size=len(rows))
    reference_logits[np.arange(len(rows)), winners] += np.where(rng.random(len(rows)) < 0.97, 9.0, 3.0)
    target = np.array([rng.choice(VOCAB, p=np.exp(ab_stats.log_softmax_rows(row[None, :])[0])) for row in reference_logits], dtype=np.uint32)
    long_rows = np.array([corpus["documents"][d]["stratum"] == "long" for d in doc])
    noise = {
        "anchor": 0.05, "close": 0.0502, "edge": 0.0524, "bad": 0.35, "bridge": 0.03, "kv-good": 0.02,
    }
    arms = {
        "reference": arm_descriptor("demo", "S1", "bf16", "bf16", "publisher"),
        "anchor": arm_descriptor("demo", "S1", "fp8", "fp8", "publisher"),
        "close": arm_descriptor("demo", "S1", "nvfp4", "nvfp4nv", "community"),
        "edge": arm_descriptor("demo", "S1", "int8", "int8x", "community"),
        "bad": arm_descriptor("demo", "S1", "mxfp4", "mxfp4x", "community"),
        "bridge": arm_descriptor("demo", "S0", "fp8", "fp8", "publisher"),
    }
    arms["kv-good"] = arm_descriptor("demo", "S1", "fp8", "fp8", "publisher", latent="fp8", group=128, mode="sim")
    arms["kv-bad"] = arm_descriptor("demo", "S1", "fp8", "fp8", "publisher", latent="mxfp4", group=32, mode="sim")
    arms["kv-good"]["pack_sha256"] = arms["anchor"]["pack_sha256"]
    arms["kv-bad"]["pack_sha256"] = arms["anchor"]["pack_sha256"]
    for name in ("kv-good", "kv-bad"):
        arms[name]["artifacts"] = dict(arms["anchor"]["artifacts"], module_archive_sha256=hexid("archive-kvsim"), driver_sha256=hexid("driver-kvsim"))
        ab_arm.validate(arms[name])
    anchor_logits = reference_logits + rng.normal(0, noise["anchor"], reference_logits.shape)
    logits = {
        "reference": reference_logits,
        "anchor": anchor_logits,
        "close": reference_logits + rng.normal(0, noise["close"], reference_logits.shape),
        "edge": reference_logits + rng.normal(0, noise["edge"], reference_logits.shape),
        "bad": reference_logits + rng.normal(0, noise["bad"], reference_logits.shape),
        "bridge": anchor_logits + rng.normal(0, noise["bridge"], reference_logits.shape),
        "kv-good": anchor_logits + rng.normal(0, noise["kv-good"], reference_logits.shape),
    }
    scale = np.where(pos[:, None] >= 4096, 0.45, 0.02)
    logits["kv-bad"] = anchor_logits + rng.normal(0, 1.0, reference_logits.shape) * scale
    header = {"corpus_sha256": corpus["corpus_sha256"], "tokenizer_sha256": corpus["tokenizer_sha256"]}
    ref_arrays = ab_dump.from_full_logits(logits["reference"], doc, pos, target, None, PROBES)
    probe_sha = ab_dump.probe_sha256(ref_arrays["top_ids"])
    dumps, shas = {}, {}
    for name in logits:
        arrays = ref_arrays if name == "reference" else ab_dump.from_full_logits(logits[name], doc, pos, target, ref_arrays["top_ids"], PROBES)
        path = out / f"{name}.npz"
        shas[name] = ab_dump.write(path, {**header, "arm_digest": ab_arm.arm_digest(arms[name]),
                                          "probe_sha256": None if name == "reference" else probe_sha}, arrays)
        dumps[name] = path
    aa_path = out / "reference-aa.npz"
    aa_sha = ab_dump.write(aa_path, {**header, "arm_digest": ab_arm.arm_digest(arms["reference"]), "probe_sha256": None}, ref_arrays)
    tier2 = np.sort(rng.choice(len(rows), size=len(rows) // 5, replace=False))
    exact_paths = {}
    for name, base in (("anchor", "reference"), ("close", "reference"), ("edge", "reference"), ("bad", "reference"), ("bridge", "anchor"), ("kv-good", "anchor"), ("kv-bad", "anchor")):
        path = out / f"exact-{name}.npz"
        ab_dump.write_exact(path, {"reference_arm_digest": ab_arm.arm_digest(arms[base]), "arm_digest": ab_arm.arm_digest(arms[name]),
                                   "corpus_sha256": corpus["corpus_sha256"]},
                            tier2, ab_stats.exact_kl_rows(logits[base][tier2], logits[name][tier2]))
        exact_paths[name] = path
    anchor_probe = ab_dump.from_full_logits(logits["anchor"], doc, pos, target, None, PROBES)
    anchor_ref_path = out / "anchor-as-reference.npz"
    ab_dump.write(anchor_ref_path, {**header, "arm_digest": ab_arm.arm_digest(arms["anchor"]), "probe_sha256": None}, anchor_probe)
    anchor_probe_sha = ab_dump.probe_sha256(anchor_probe["top_ids"])
    for name in ("bridge", "kv-good", "kv-bad"):
        arrays = ab_dump.from_full_logits(logits[name], doc, pos, target, anchor_probe["top_ids"], PROBES)
        shas[name] = ab_dump.write(out / f"{name}.npz", {**header, "arm_digest": ab_arm.arm_digest(arms[name]), "probe_sha256": anchor_probe_sha}, arrays)
    draft = json.loads((Path(__file__).resolve().parent.parent / "qualification" / "ab" / "PLAN.template.json").read_text())
    draft["campaign"] = CAMPAIGN
    draft["firmware_commit"] = COMMIT
    draft["notes"] = "SYNTHETIC dry run; not a measurement."
    draft["tokenizer_sha256"] = corpus["tokenizer_sha256"]
    for entry in draft["corpora"].values():
        entry["tokens_sha256"] = corpus["corpus_sha256"]
        entry["index_sha256"] = hexid("index")
    draft["corpora"]["CT-short"]["docs"] = len(corpus["documents"])
    draft["corpora"]["CT-long"]["docs"] = 32
    ids = {name: arm["arm_id"] for name, arm in arms.items()}
    draft["arms"] = [
        {"arm_id": ids["reference"], "role": "reference", "label": "R"},
        {"arm_id": ids["anchor"], "role": "anchor", "label": "A", "axis": "E", "compare_to": ids["reference"], "corpus": "CT-short"},
        {"arm_id": ids["close"], "role": "arm", "label": "E-close", "axis": "E", "compare_to": ids["reference"], "anchor": ids["anchor"], "corpus": "CT-short"},
        {"arm_id": ids["edge"], "role": "arm", "label": "E-edge", "axis": "E", "compare_to": ids["reference"], "anchor": ids["anchor"], "corpus": "CT-short"},
        {"arm_id": ids["bad"], "role": "arm", "label": "E-bad", "axis": "E", "compare_to": ids["reference"], "anchor": ids["anchor"], "corpus": "CT-short"},
        {"arm_id": ids["bridge"], "role": "bridge", "label": "S0", "axis": "spine", "compare_to": ids["anchor"], "corpus": "CT-short"},
        {"arm_id": ids["kv-good"], "role": "arm", "label": "K-good", "axis": "K", "compare_to": ids["anchor"], "corpus": "CT-long"},
        {"arm_id": ids["kv-bad"], "role": "arm", "label": "K-bad", "axis": "K", "compare_to": ids["anchor"], "corpus": "CT-long"},
    ]
    draft["backstops"]["calibration_pair"] = [ids["anchor"], ids["reference"]]
    draft["statistics"]["bootstrap"]["replicates"] = replicates
    (out / "PLAN.draft.json").write_text(json.dumps(draft, indent=1, sort_keys=True))
    plan_path = out / "PLAN.json"
    if plan_path.exists():
        plan_path.unlink()
    plan_sha = ab_plan.freeze(out / "PLAN.draft.json", plan_path)
    plan = ab_plan.load(plan_path)
    token_sha = hexid("generated")
    flags = {"EXPERT_CODEC": "bf16"}
    receipts = {}
    for name, arm in arms.items():
        codec_flags = dict(flags, EXPERT_CODEC=arm["expert"]["codec"])
        if name.startswith("kv-"):
            codec_flags["KV_QUANT_SIM"] = "1"
        if name == "bridge":
            codec_flags["MODEL_REVISION"] = arm["revision"]
        receipts[name] = receipt(arm, name, shas[name], token_sha, None if name == "reference" else (anchor_probe_sha if name in ("bridge", "kv-good", "kv-bad") else probe_sha),
                                 corpus, plan_sha, codec_flags)
    receipts["anchor-reference"] = receipt(arms["anchor"], "anchor-reference", ab_dump.sha256_file(anchor_ref_path), token_sha, None, corpus, plan_sha,
                                           dict(flags, EXPERT_CODEC="fp8"))
    receipts["reference-aa"] = receipt(arms["reference"], "reference-aa", aa_sha, token_sha, None, corpus, plan_sha, flags, order="permuted")
    for name, value in receipts.items():
        (out / f"receipt-{name}.json").write_text(json.dumps(value, indent=1, sort_keys=True))
    aa = ab_receipt.aa(receipts["reference"], receipts["reference-aa"])
    anchor_aa = ab_receipt.aa(receipts["anchor-reference"], dict(receipts["anchor-reference"], run_label="anchor-reference-aa",
                                                                cache={**receipts["anchor-reference"]["cache"], "kv_snapshot_directory": "/home/spark/ab/anchor-reference/kv-aa"}))
    aa["merged_dump_sha256"] = sorted(set(aa["merged_dump_sha256"]) | set(anchor_aa["merged_dump_sha256"]))
    (out / "AA.json").write_text(json.dumps(aa, indent=1, sort_keys=True))
    comparisons = {}
    pairs = {
        "anchor": ("reference", None, "E", dumps["reference"]),
        "close": ("reference", "anchor", "E", dumps["reference"]),
        "edge": ("reference", "anchor", "E", dumps["reference"]),
        "bad": ("reference", "anchor", "E", dumps["reference"]),
        "bridge": ("anchor-reference", None, "spine", anchor_ref_path),
        "kv-good": ("anchor-reference", None, "K", anchor_ref_path),
        "kv-bad": ("anchor-reference", None, "K", anchor_ref_path),
    }
    bins = plan["statistics"]["position_bins"]
    for name, (reference_name, anchor_name, axis, reference_path) in pairs.items():
        result = ab_score_compare.compare(corpus, ab_dump.read(reference_path), ab_dump.read(out / f"{name}.npz"),
                                          ab_dump.read(dumps[anchor_name]) if anchor_name else None,
                                          ab_dump.read_exact(exact_paths[name]), bins,
                                          plan["statistics"]["near_tie_nats"], plan["statistics"]["decisive_nats"])
        comparisons[ids[name]] = result
        (out / f"comparison-{name}.json").write_text(json.dumps(result, indent=1, sort_keys=True))
    suites = {}
    expected = [str(10 + index) for index in range(17)]
    base_answers = [(value if index not in (3, 4, 14) else "0", value, [1, 2, 3, index]) for index, value in enumerate(expected)]
    write_suite(out / "suite-reference", base_answers)
    write_suite(out / "suite-close", [(answer, value, tokens if index != 7 else [1, 2, 9]) for index, (answer, value, tokens) in enumerate(base_answers)])
    bad_answers = [("0" if index in (0, 1, 2, 5, 6, 8, 9) else answer, value, tokens[:2] + [7]) for index, (answer, value, tokens) in enumerate(base_answers)]
    write_suite(out / "suite-bad", bad_answers)
    reference_suite = ab_suite_compare.load(out / "suite-reference")
    for name in ("close", "bad"):
        result = ab_suite_compare.compare(reference_suite, ab_suite_compare.load(out / f"suite-{name}"), "COMPSEC-17")
        (out / f"suite-{name}.json").write_text(json.dumps(result, indent=1, sort_keys=True))
        suites[ids[name]] = [result]
    verdicts = ab_verdict.decide(plan, comparisons, suites, {})
    (out / "VERDICTS.json").write_text(json.dumps({"format": ab_verdict.FORMAT, "plan_sha256": plan_sha, "verdicts": verdicts}, indent=1, sort_keys=True))
    roofline = ("roofline @B=1: memory 41.0% (ceiling 106 tok/s) | compute 3.0% | transport bw 1.0% + latency 55.0% "
                "(inputs: SYNTHETIC spine 2.0 GB + experts 0.3 GB/token/rank + KV/index/state reads 1.1 MB at context 1280 with kv_shard=1, 243 GB/s, 400 Gbps, 92 collective rounds/token)")
    report_input = {"campaign": CAMPAIGN, "plan": "PLAN.json", "verdicts": "VERDICTS.json", "aa": "AA.json", "comparisons": []}
    for name, (reference_name, anchor_name, axis, _) in pairs.items():
        spec = {"arm_id": ids[name], "reference_id": arms[reference_name.replace("anchor-reference", "anchor")]["arm_id"], "axis": axis,
                "comparison": f"comparison-{name}.json", "reference_receipt": f"receipt-{reference_name}.json", "arm_receipt": f"receipt-{name}.json",
                "suites": [f"suite-{name}.json"] if name in ("close", "bad") else [], "routed_parity": "PASS (synthetic)",
                "side": {"pack_gb_per_rank": 12.6 if name == "close" else 21.7, "mem_available_delta_gib": -22.5,
                         "projected_kv_bytes_per_token": 545 if axis == "K" else None},
                "speed": [{"label": "B1 non-spec SYNTHETIC", "tok_s": 38.9, "roofline": roofline}] if name == "close" else []}
        report_input["comparisons"].append(spec)
    (out / "REPORT_INPUT.json").write_text(json.dumps(report_input, indent=1, sort_keys=True))
    return ab_report.render(out / "REPORT_INPUT.json")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out")
    parser.add_argument("--replicates", type=int, default=10000)
    args = parser.parse_args()
    sys.stdout.write(run(Path(args.out), args.replicates))
    return 0


if __name__ == "__main__":
    sys.exit(main())
