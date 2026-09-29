#!/usr/bin/env python3
"""Arm run receipts and the comparability contract (design §2.6, §3.1 C1-C6).

validate  one receipt against qualification/ab/receipt.schema.json and its own
          internal consistency (arm digest, ready event, pack SHAs, topology,
          cached_prompt_tokens == 0, snapshot directory inside the arm root)
compare   refuses a comparison unless C1-C6 and the spine rule hold and the two
          arms differ on exactly the declared axis
aa        the A/A gate: two fresh-engine runs of one arm must give bit-identical
          merged dumps and generated token ids
dumpcheck served tokens are identical with the score dump on and off

usage:
  ab_receipt.py validate RECEIPT.json
  ab_receipt.py compare --axis E|K|D|spine REFERENCE_RECEIPT ARM_RECEIPT [--plan PLAN.json]
  ab_receipt.py aa RECEIPT_A RECEIPT_B --out AA.json
  ab_receipt.py dumpcheck RECEIPT_ON RECEIPT_OFF
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import ab_arm  # noqa: E402

SCHEMA_PATH = ROOT / "qualification" / "ab" / "receipt.schema.json"
AXIS_BUILD_FLAGS = {
    "E": {"EXPERT_CODEC"},
    "K": {"KV_QUANT_SIM"},
    "D": {"DRAFT_EXPERT_CODECS"},
    "spine": {"MODEL_REVISION", "CONTRACT_SHA256"},
}
TYPES = {"object": dict, "array": list, "string": str, "integer": int, "boolean": bool, "null": type(None)}


class ReceiptError(ValueError):
    pass


def _type_ok(value, names) -> bool:
    for name in names if isinstance(names, list) else [names]:
        expected = TYPES[name]
        if name == "integer" and isinstance(value, bool):
            continue
        if isinstance(value, expected):
            return True
    return False


def check_schema(value, schema: dict, where: str = "receipt") -> None:
    if "const" in schema and value != schema["const"]:
        raise ReceiptError(f"{where} must be {schema['const']!r}")
    if "enum" in schema and value not in schema["enum"]:
        raise ReceiptError(f"{where} must be one of {schema['enum']}")
    if "type" in schema and not _type_ok(value, schema["type"]):
        raise ReceiptError(f"{where} must be of type {schema['type']}")
    if isinstance(value, str):
        if "pattern" in schema and not re.search(schema["pattern"], value):
            raise ReceiptError(f"{where} {value!r} does not match {schema['pattern']}")
        if len(value) < schema.get("minLength", 0):
            raise ReceiptError(f"{where} is too short")
    if isinstance(value, int) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            raise ReceiptError(f"{where} is below {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            raise ReceiptError(f"{where} is above {schema['maximum']}")
    if isinstance(value, dict):
        for key in schema.get("required", []):
            if key not in value:
                raise ReceiptError(f"{where}.{key} is required")
        properties = schema.get("properties", {})
        extra = schema.get("additionalProperties", True)
        for key, item in value.items():
            if key in properties:
                check_schema(item, properties[key], f"{where}.{key}")
            elif extra is False:
                raise ReceiptError(f"{where}.{key} is not a receipt member")
            elif isinstance(extra, dict):
                check_schema(item, extra, f"{where}.{key}")
    if isinstance(value, list) and "items" in schema:
        for index, item in enumerate(value):
            check_schema(item, schema["items"], f"{where}[{index}]")


def inside(path: str, root: str) -> bool:
    child, parent = PurePosixPath(os.path.normpath(path)), PurePosixPath(os.path.normpath(root))
    return ".." not in PurePosixPath(path).parts and child != parent and parent in child.parents


def validate(receipt: dict) -> dict:
    check_schema(receipt, json.loads(SCHEMA_PATH.read_text()))
    try:
        arm = ab_arm.validate(receipt["arm"])
    except ab_arm.ArmError as error:
        raise ReceiptError(f"receipt.arm: {error}") from error
    digest = ab_arm.arm_digest(arm)
    ranks = arm["topology"]["tp"] * arm["topology"]["pp"]
    if receipt["arm_digest"] != digest:
        raise ReceiptError(f"receipt arm_digest {receipt['arm_digest']} is not the digest of its arm ({digest})")
    ready = receipt["ready_event"]
    if ready["arm_digest"] != digest:
        raise ReceiptError("ready event arm_digest differs from the receipt arm: the engine ran another arm")
    if ready["pack_set_sha256"] != ab_arm.pack_set_sha256(arm):
        raise ReceiptError("ready event pack_set_sha256 differs from the arm pack SHAs")
    if receipt["packs"]["pack_sha256"] != arm["pack_sha256"]:
        raise ReceiptError("receipt pack_sha256 differs from the arm pack SHAs")
    topology = receipt["topology"]
    if (topology["tp"], topology["pp"], topology["kv_shard"]) != (arm["topology"]["tp"], arm["topology"]["pp"], arm["topology"]["kv_shard"]):
        raise ReceiptError("receipt topology differs from the arm topology")
    for name, values in (("topology.rank_nodes", topology["rank_nodes"]), ("packs.sha256sums_sha256", receipt["packs"]["sha256sums_sha256"]),
                         ("packs.experts_sha256", receipt["packs"]["experts_sha256"]), ("dumps.per_rank_sha256", receipt["dumps"]["per_rank_sha256"])):
        if len(values) != ranks:
            raise ReceiptError(f"{name} has {len(values)} entries for {ranks} ranks")
    if receipt["build"]["model_revision"] != arm["revision"]:
        raise ReceiptError("build model_revision differs from the arm revision")
    requests = receipt["requests"]
    if len(requests["cached_prompt_tokens"]) != requests["count"]:
        raise ReceiptError("requests.cached_prompt_tokens must hold one value per request")
    cached = [value for value in requests["cached_prompt_tokens"] if value != 0]
    if cached:
        raise ReceiptError(f"cached_prompt_tokens > 0 on {len(cached)} requests: prefix reuse contaminated the run (C4)")
    cache = receipt["cache"]
    if not inside(cache["kv_snapshot_directory"], cache["arm_root"]):
        raise ReceiptError("kv_snapshot_directory is not under the arm's own root: SparkKvSnapshotPrune would delete another root's snapshots (C4)")
    return receipt


def compare(reference: dict, arm: dict, axis: str, plan: dict | None = None) -> list:
    validate(reference)
    validate(arm)
    problems = []
    if axis not in ab_arm.AXES:
        raise ReceiptError(f"axis must be one of {ab_arm.AXES}")
    differing = ab_arm.differing_axes(reference["arm"], arm["arm"])
    if differing != [axis]:
        problems.append(f"arms differ on {differing or 'no axis'}, the comparison declares exactly [{axis}]")
    if axis != "spine":
        if reference["arm"]["spine"]["spine_digest"] != arm["arm"]["spine"]["spine_digest"]:
            unequal = sum(1 for a, b in zip(reference["arm"]["spine"]["spine_digest"], arm["arm"]["spine"]["spine_digest"]) if a != b)
            problems.append(f"spine digests differ on {unequal} ranks: an {axis} comparison needs a byte-identical spine")
    if reference["source_commit"] != arm["source_commit"]:
        problems.append(f"C1 source commit {arm['source_commit'][:12]} differs from the reference {reference['source_commit'][:12]}")
    if plan is not None and reference["source_commit"] != plan["firmware_commit"]:
        problems.append("C1 the reference is not on the plan's pinned firmware commit")
    allowed = AXIS_BUILD_FLAGS[axis]
    flags_a, flags_b = reference["build"]["flags"], arm["build"]["flags"]
    changed = sorted(key for key in set(flags_a) | set(flags_b) if flags_a.get(key) != flags_b.get(key))
    extra = [key for key in changed if key not in allowed]
    if extra:
        problems.append(f"C1 build flags {extra} differ; an {axis} comparison may change only {sorted(allowed)}")
    if axis != "spine":
        for key in ("model_revision", "contract_sha256"):
            if reference["build"][key] != arm["build"][key]:
                problems.append(f"C1 build {key} differs outside a declared spine comparison")
    if axis in ("K", "D") and reference["packs"]["pack_sha256"] != arm["packs"]["pack_sha256"]:
        problems.append(f"a {axis} arm runs on its base arm's pack bytes; pack SHAs differ")
    for key in ("tp", "pp", "kv_shard", "rank_nodes"):
        if reference["topology"][key] != arm["topology"][key]:
            problems.append(f"C2 topology {key} differs")
    if reference["wave"] != arm["wave"]:
        problems.append("C3 wave shape differs (sequential flag, prefill rows per submission or output budget)")
    if reference["cache"]["kv_snapshot_directory"] == arm["cache"]["kv_snapshot_directory"]:
        problems.append("C4 both runs share one kv_snapshot_directory; every arm needs its own")
    for key in ("mode", "dropin_sha256"):
        if reference["execution"][key] != arm["execution"][key]:
            problems.append(f"C5 execution {key} differs")
    if reference["weightd"]["residency"] != arm["weightd"]["residency"]:
        problems.append("C5 weightd residency differs")
    for key in ("corpus", "corpus_tokens_sha256", "corpus_index_sha256", "tokenizer_sha256"):
        if reference["inputs"][key] != arm["inputs"][key]:
            problems.append(f"C6 input {key} differs")
    if arm["inputs"]["probe_sha256"] is None:
        problems.append("C6 the arm run did not read the reference probe file")
    if reference["plan_sha256"] != arm["plan_sha256"] or (plan is not None and arm["plan_sha256"] != plan["plan_sha256"]):
        problems.append("C6 plan sha differs")
    return problems


def notes(reference: dict, arm: dict) -> list:
    out = []
    for receipt, label in ((reference, "reference"), (arm, "arm")):
        if "gemm" in receipt["execution"]["expert_paths"]:
            out.append(f"{label} scored some waves through the expert GEMM path (bf16 rounding of dequantized weights): report E deltas as prefill-path")
    return out


def aa(first: dict, second: dict) -> dict:
    validate(first)
    validate(second)
    if first["arm_digest"] != second["arm_digest"]:
        raise ReceiptError("an A/A pair runs one arm twice")
    if first["run_label"] == second["run_label"]:
        raise ReceiptError("an A/A pair needs two distinct runs")
    if first["cache"]["kv_snapshot_directory"] == second["cache"]["kv_snapshot_directory"]:
        raise ReceiptError("an A/A pair needs two fresh snapshot directories")
    identical = (first["dumps"]["merged_sha256"] == second["dumps"]["merged_sha256"]
                 and first["requests"]["generated_token_ids_sha256"] == second["requests"]["generated_token_ids_sha256"])
    return {"format": "sparkpipe-ab-aa-v1", "status": "BIT-IDENTICAL" if identical else "DIVERGED",
            "arm_digest": first["arm_digest"], "plan_sha256": first["plan_sha256"],
            "runs": [first["run_label"], second["run_label"]],
            "document_order": [first["inputs"]["document_order"], second["inputs"]["document_order"]],
            "merged_dump_sha256": sorted({first["dumps"]["merged_sha256"], second["dumps"]["merged_sha256"]}),
            "generated_token_ids_sha256": [first["requests"]["generated_token_ids_sha256"], second["requests"]["generated_token_ids_sha256"]]}


def dumpcheck(on: dict, off: dict) -> bool:
    validate(on)
    validate(off)
    if on["arm_digest"] != off["arm_digest"] or not on["dumps"]["score_dump_on"] or off["dumps"]["score_dump_on"]:
        raise ReceiptError("dumpcheck needs one arm run with the score dump on and one with it off")
    return on["requests"]["generated_token_ids_sha256"] == off["requests"]["generated_token_ids_sha256"]


def load(path) -> dict:
    return json.loads(Path(path).read_text())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("validate")
    p.add_argument("receipt")
    p = sub.add_parser("compare")
    p.add_argument("--axis", required=True, choices=ab_arm.AXES)
    p.add_argument("--plan")
    p.add_argument("reference")
    p.add_argument("arm")
    p = sub.add_parser("aa")
    p.add_argument("first")
    p.add_argument("second")
    p.add_argument("--out", required=True)
    p = sub.add_parser("dumpcheck")
    p.add_argument("on")
    p.add_argument("off")
    args = parser.parse_args()
    try:
        if args.command == "validate":
            validate(load(args.receipt))
            print("RECEIPT-OK")
            return 0
        if args.command == "compare":
            plan = None
            if args.plan:
                import ab_plan
                plan = ab_plan.load(args.plan)
            reference, arm = load(args.reference), load(args.arm)
            problems = compare(reference, arm, args.axis, plan)
            for problem in problems:
                print(f"REFUSED {problem}")
            for note in notes(reference, arm):
                print(f"NOTE {note}")
            print("COMPARABLE" if not problems else "NOT-COMPARABLE")
            return 0 if not problems else 1
        if args.command == "aa":
            result = aa(load(args.first), load(args.second))
            Path(args.out).write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
            print(f"A/A {result['status']}")
            return 0 if result["status"] == "BIT-IDENTICAL" else 1
        if args.command == "dumpcheck":
            same = dumpcheck(load(args.on), load(args.off))
            print("SERVED-TOKENS-IDENTICAL" if same else "SERVED-TOKENS-DIFFER")
            return 0 if same else 1
    except ReceiptError as error:
        print(f"ab_receipt: REFUSED: {error}", file=sys.stderr)
        return 1
    return 2


if __name__ == "__main__":
    sys.exit(main())
