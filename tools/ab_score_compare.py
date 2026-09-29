#!/usr/bin/env python3
"""Teacher-forced comparison of one arm against its reference (design §3.2).

Reads merged score dumps (tools/ab_dump.py), checks they score the same corpus
rows with the same probe set, and writes per-document metric tables:

  KL_b(P||Q)   bucketed KL over the reference's top-K probe ids plus one tail bucket
  top-1        agreement with the reference argmax, near-ties (gap < --near-tie) excluded
  decisive     flips where the reference margin exceeds --decisive nats
  dNLL         NLL_arm - NLL_ref of the corpus next token (positive = arm worse)
  exact KL     full-vocabulary KL on the Tier-2 rows, when --exact is given

With --anchor the anchor arm (scored against the same reference and probes) is
tabulated beside the arm for the ratio and difference metrics. Statistics and
verdicts are computed by tools/ab_verdict.py from these tables.

usage:
  ab_score_compare.py --corpus CORPUS.json --reference REF.npz --arm ARM.npz
      [--anchor ANCHOR.npz] [--exact EXACT.npz] [--bins 0,1024,4096,16384,32768]
      --out COMPARISON.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ab_dump
import ab_stats

FORMAT = "sparkpipe-ab-comparison-v1"


class CompareError(ValueError):
    pass


def row_metrics(reference: dict, arm: dict, near_tie: float, decisive: float) -> dict:
    kl = ab_stats.bucket_kl_rows(reference["top_logits"], reference["lse"], arm["probe_logits"], arm["lse"])
    gap = reference["top_logits"][:, 0].astype(np.float64) - reference["top_logits"][:, 1].astype(np.float64)
    eligible = gap >= near_tie
    agree = arm["top_ids"][:, 0] == reference["top_ids"][:, 0]
    nll_ref = reference["lse"] - reference["target_logit"].astype(np.float64)
    nll_arm = arm["lse"] - arm["target_logit"].astype(np.float64)
    return {"kl": kl["kl"], "p_tail": kl["p_tail"], "tail_zero_rows": kl["tail_zero_rows"],
            "infinite_rows": kl["infinite_rows"], "eligible": eligible, "agree": agree,
            "decisive_flip": (~agree) & (gap > decisive), "nll_ref": nll_ref, "dnll": nll_arm - nll_ref}


def per_doc(values: np.ndarray, doc: np.ndarray, docs: int, mask: np.ndarray | None = None) -> np.ndarray:
    if mask is None:
        mask = np.ones(len(values), dtype=bool)
    counts = np.bincount(doc[mask], minlength=docs).astype(np.float64)
    sums = np.bincount(doc[mask], weights=np.asarray(values, dtype=np.float64)[mask], minlength=docs)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(counts > 0, sums / counts, np.nan)


def as_list(values: np.ndarray) -> list:
    return [None if not math.isfinite(float(value)) else float(value) for value in values]


def check_alignment(reference: dict, other: dict, name: str, probe_sha: str) -> None:
    for key in ("corpus_sha256", "tokenizer_sha256"):
        if reference["header"].get(key) != other["header"].get(key):
            raise CompareError(f"{name}: {key} {other['header'].get(key)} differs from the reference {reference['header'].get(key)}")
    for key in ("doc", "pos", "target"):
        if reference[key].shape != other[key].shape or not np.array_equal(reference[key], other[key]):
            raise CompareError(f"{name}: rows differ from the reference in {key}; both runs must score the same corpus rows in the same order")
    if other["header"].get("probe_sha256") != probe_sha:
        raise CompareError(f"{name}: probe_sha256 {other['header'].get('probe_sha256')} is not the reference probe set {probe_sha}")


def compare(corpus: dict, reference: dict, arm: dict, anchor: dict | None, exact: dict | None,
            bins: list, near_tie: float, decisive: float) -> dict:
    probe_sha = ab_dump.probe_sha256(reference["top_ids"])
    own = reference["header"].get("probe_sha256")
    if own is not None and own != probe_sha:
        raise CompareError(f"reference: header probe_sha256 {own} is not the sha of its own top ids {probe_sha}")
    if not np.array_equal(reference["probe_logits"], reference["top_logits"]):
        raise CompareError("reference: probe_logits must equal top_logits for the run that defines the probes")
    if reference["header"].get("corpus_sha256") != corpus["corpus_sha256"]:
        raise CompareError("reference: corpus_sha256 does not match the corpus index")
    check_alignment(reference, arm, "arm", probe_sha)
    if anchor is not None:
        check_alignment(reference, anchor, "anchor", probe_sha)
    docs = len(corpus["documents"])
    doc = reference["doc"].astype(np.int64)
    if doc.size and doc.max() >= docs:
        raise CompareError("dump names a document index outside the corpus")
    metrics = row_metrics(reference, arm, near_tie, decisive)
    table = {
        "doc_ids": [entry["id"] for entry in corpus["documents"]],
        "stratum": [entry["stratum"] for entry in corpus["documents"]],
        "kl": as_list(per_doc(metrics["kl"], doc, docs)),
        "top1_agree": as_list(per_doc(metrics["agree"], doc, docs, metrics["eligible"])),
        "decisive_flip": as_list(per_doc(metrics["decisive_flip"], doc, docs, metrics["eligible"])),
        "nll_ref": as_list(per_doc(metrics["nll_ref"], doc, docs)),
        "dnll": as_list(per_doc(metrics["dnll"], doc, docs)),
        "rows": [int(value) for value in np.bincount(doc, minlength=docs)],
    }
    anchor_rows = None
    if anchor is not None:
        anchor_rows = row_metrics(reference, anchor, near_tie, decisive)
        table["kl_anchor"] = as_list(per_doc(anchor_rows["kl"], doc, docs))
        table["top1_agree_anchor"] = as_list(per_doc(anchor_rows["agree"], doc, docs, anchor_rows["eligible"]))
    calibration = None
    if exact is not None:
        header = exact["header"]
        if header.get("reference_arm_digest") != reference["header"].get("arm_digest") or header.get("arm_digest") != arm["header"].get("arm_digest"):
            raise CompareError("exact KL file is not for this reference and arm")
        rows = exact["rows"].astype(np.int64)
        if rows.size and rows.max() >= len(doc):
            raise CompareError("exact KL file names a row outside the dump")
        bucket = metrics["kl"][rows]
        violation = bucket - exact["kl"]
        tolerance = 1e-9 * np.maximum(1.0, np.abs(exact["kl"]))
        exact_doc = np.full(docs, np.nan)
        counts = np.bincount(doc[rows], minlength=docs)
        sums = np.bincount(doc[rows], weights=exact["kl"], minlength=docs)
        exact_doc[counts > 0] = sums[counts > 0] / counts[counts > 0]
        table["exact_kl"] = as_list(exact_doc)
        calibration = {"rows": int(rows.size), "violations": int(np.sum(violation > tolerance)),
                       "max_violation": float(violation.max()) if rows.size else 0.0,
                       "mean_bucket_kl": float(bucket.mean()) if rows.size else None,
                       "mean_exact_kl": float(exact["kl"].mean()) if rows.size else None}
    positions = reference["pos"].astype(np.int64)
    bin_tables = {}
    for low, high in zip(bins[:-1], bins[1:]):
        inside = (positions >= low) & (positions < high)
        if not np.any(inside):
            continue
        entry = {
            "kl": as_list(per_doc(metrics["kl"], doc, docs, inside)),
            "top1_agree": as_list(per_doc(metrics["agree"], doc, docs, inside & metrics["eligible"])),
            "nll_ref": as_list(per_doc(metrics["nll_ref"], doc, docs, inside)),
            "dnll": as_list(per_doc(metrics["dnll"], doc, docs, inside)),
            "rows": int(np.sum(inside)),
        }
        bin_tables[f"{low}-{high}"] = entry
    selection = None
    if "select_hash" in reference and "select_hash" in arm:
        selection = float(np.mean(reference["select_hash"] != arm["select_hash"]))
    kl_doc = np.array([np.nan if value is None else value for value in table["kl"]])
    order = [int(i) for i in np.argsort(-np.nan_to_num(kl_doc, nan=-np.inf), kind="stable")]
    worst = order[:max(1, math.ceil(0.05 * docs))]
    return {
        "format": FORMAT,
        "reference_arm_digest": reference["header"].get("arm_digest"),
        "arm_digest": arm["header"].get("arm_digest"),
        "anchor_arm_digest": anchor["header"].get("arm_digest") if anchor is not None else None,
        "corpus_sha256": corpus["corpus_sha256"],
        "probe_sha256": probe_sha,
        "reference_dump_sha256": reference["sha256"],
        "arm_dump_sha256": arm["sha256"],
        "anchor_dump_sha256": anchor["sha256"] if anchor is not None else None,
        "rows": int(len(doc)),
        "docs": docs,
        "near_tie_threshold": near_tie,
        "decisive_threshold": decisive,
        "near_tie_rows": int(np.sum(~metrics["eligible"])),
        "tail_mass_p99": float(np.quantile(metrics["p_tail"], 0.99)) if len(doc) else 0.0,
        "tail_zero_rows": metrics["tail_zero_rows"],
        "infinite_rows": metrics["infinite_rows"],
        "row_means": {"kl": float(metrics["kl"].mean()), "top1_agree": float(metrics["agree"][metrics["eligible"]].mean()),
                      "dnll": float(metrics["dnll"].mean()), "nll_ref": float(metrics["nll_ref"].mean())},
        "bucket_vs_exact": calibration,
        "selection_flip_rate": selection,
        "bins": bins,
        "per_doc": table,
        "per_bin": bin_tables,
        "worst_docs": [{"doc": index, "id": table["doc_ids"][index], "kl": table["kl"][index]} for index in worst],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--anchor")
    parser.add_argument("--exact")
    parser.add_argument("--bins", default="0,1024,4096,16384,32768")
    parser.add_argument("--near-tie", type=float, default=1e-3)
    parser.add_argument("--decisive", type=float, default=1.0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    corpus = json.loads(Path(args.corpus).read_text())
    try:
        result = compare(corpus, ab_dump.read(args.reference), ab_dump.read(args.arm),
                         ab_dump.read(args.anchor) if args.anchor else None,
                         ab_dump.read_exact(args.exact) if args.exact else None,
                         [int(value) for value in args.bins.split(",")], args.near_tie, args.decisive)
    except (CompareError, ab_dump.DumpError, ValueError) as error:
        print(f"ab_score_compare: REFUSED: {error}", file=sys.stderr)
        return 1
    Path(args.out).write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    print(f"ab_score_compare: {result['rows']} rows, {result['docs']} docs, KL_b row mean {result['row_means']['kl']:.6g}, "
          f"top1 {100 * result['row_means']['top1_agree']:.3f}%, tail p99 {result['tail_mass_p99']:.3g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
