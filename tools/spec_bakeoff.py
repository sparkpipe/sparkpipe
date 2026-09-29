#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import statistics
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import spec_roofline  # noqa: E402
import spec_verify_bench  # noqa: E402
from spec_offline.streams import read_u32, sha256_ids  # noqa: E402
from spec_offline.tree import parse_shape  # noqa: E402

METHODS = ("off", "oracle", "adversary", "lookup", "mtp", "mtp+lookup", "suffix", "ngram3", "dflash2", "dflash", "dspark", "eagle3", "tree")
DRAFTER_FREE = {"off", "oracle", "adversary", "lookup", "suffix", "ngram3"}
LICENSED = {"mtp", "mtp+lookup", "dflash2", "dflash", "dspark", "eagle3"}
PLACEMENTS = ("fleet", "rtx5090", "offline")
BATCHES = (1, 2, 4, 8)
SCORED_CLASSES = {"prose": 1.0, "code": 1.0, "chat": 1.0, "long": 1.0, "repetitive": 0.5, "thinking": 1.0, "tool_json": 1.0, "chinese": 1.0}
REGRESSION_FLOOR = 0.98
MEMORY_FLOOR_GIB = 20.0
ARM = re.compile(r"^(?P<model>[a-z0-9]+)/(?P<method>[a-z0-9+]+)(?::(?P<drafter>[A-Za-z0-9._-]+))?@(?P<placement>[a-z0-9]+)/(?P<shape>[a-z0-9+-]+)/B(?P<batch>\d+)$")


@dataclass(frozen=True)
class Arm:
    model: str
    method: str
    drafter: str | None
    placement: str
    shape: str
    batch: int

    @property
    def id(self) -> str:
        drafter = f":{self.drafter}" if self.drafter else ""
        return f"{self.model}/{self.method}{drafter}@{self.placement}/{self.shape}/B{self.batch}"

    @property
    def slug(self) -> str:
        return self.id.replace("/", "_").replace(":", "-").replace("@", "-at-").replace("+", "-plus-")


def parse_arm(text: str) -> Arm:
    match = ARM.match(text)
    if not match:
        raise ValueError(f"{text}: an arm is <model>/<method>[:<drafter>]@<placement>/<shape>/B<b>")
    method, drafter, placement, batch = match.group("method"), match.group("drafter"), match.group("placement"), int(match.group("batch"))
    if method not in METHODS:
        raise ValueError(f"{text}: method {method} is not one of {METHODS}")
    if placement not in PLACEMENTS:
        raise ValueError(f"{text}: placement {placement} is not one of {PLACEMENTS}")
    if batch not in BATCHES:
        raise ValueError(f"{text}: B must be one of {BATCHES}")
    if method in LICENSED and not drafter:
        raise ValueError(f"{text}: method {method} needs a drafter id")
    if method in DRAFTER_FREE and drafter:
        raise ValueError(f"{text}: method {method} takes no drafter id")
    shape = match.group("shape")
    if method == "off":
        if shape != "chain-k0":
            raise ValueError(f"{text}: the off arm has shape chain-k0")
    else:
        parsed = parse_shape(shape)
        if parsed.kind == "chain" and parsed.depth == 0:
            raise ValueError(f"{text}: only the off arm has depth 0")
    return Arm(match.group("model"), method, drafter, placement, shape, batch)


def baseline_arms(model: str, batch: int) -> list[str]:
    return [f"{model}/off@fleet/chain-k0/B{batch}", f"{model}/off@fleet/chain-k0/B{batch}",
            f"{model}/oracle@fleet/chain-k7/B{batch}", f"{model}/adversary@fleet/chain-k1/B{batch}"]


def plan(args: argparse.Namespace) -> int:
    arms: list[str] = []
    for batch in (int(b) for b in args.batches.split(",")):
        arms.extend(baseline_arms(args.model, batch))
        for spec in args.arms or []:
            method, _, drafter = spec.partition(":")
            for placement in args.placements.split(","):
                for shape in args.shapes.split(","):
                    drafter_text = f":{drafter}" if drafter else ""
                    arms.append(f"{args.model}/{method}{drafter_text}@{placement}/{shape}/B{batch}")
    parsed = [parse_arm(arm) for arm in arms]
    seen = set()
    ordered = []
    for arm, item in zip(arms, parsed):
        ordered.append({"arm": arm, "slug": item.slug, "repeat": 2 if arm in seen else 1, **asdict(item)})
        seen.add(arm)
    Path(args.out).write_text(json.dumps({"model": args.model, "classes": args.classes.split(","), "arms": ordered}, indent=1))
    for row in ordered:
        print(f"{row['arm']}{' (repeat)' if row['repeat'] == 2 else ''}")
    return 0


def stream_hashes(results: list[dict]) -> dict[str, str]:
    hashes = {}
    for entry in results:
        key = f"{entry['class']}[{entry['index']}]"
        ids = entry.get("token_ids")
        if not ids:
            raise ValueError(f"{key} has no token ids; exactness cannot be judged from text")
        hashes[key] = sha256_ids([int(t) for t in ids])
    return hashes


def ingest(args: argparse.Namespace) -> int:
    arm = parse_arm(args.arm)
    if args.run:
        run = json.loads(Path(args.run).read_text())
        results = run["results"]
    elif args.replay_json and args.expect:
        replay = json.loads(Path(args.replay_json).read_text())
        stream = expected_stream(Path(args.expect), args.expect_prompt_tokens)
        if not replay.get("exact"):
            raise SystemExit(f"{args.replay_json}: the replay was not exact; ingest it only as evidence with --allow-inexact")
        results = [{"class": args.anchor_class, "index": 0, "token_ids": list(stream.generated), "decode_tokens": stream.length - stream.prompt - 1,
                    "decode_s": (stream.length - stream.prompt - 1) / replay["decode_tok_s"] if replay.get("decode_tok_s") else None,
                    "decode_tok_s": replay.get("decode_tok_s"), "ttft_s": replay.get("ttft_s"), "text": None}]
    elif args.expect:
        stream = expected_stream(Path(args.expect), args.expect_prompt_tokens)
        results = [{"class": args.anchor_class, "index": 0, "token_ids": list(stream.generated), "decode_tokens": None, "decode_s": None,
                    "decode_tok_s": None, "ttft_s": None, "text": None}]
    else:
        raise SystemExit("give --run <spec_verify_bench run json>, or --replay-json with --expect, or --expect alone for a recorded anchor")
    hashes = stream_hashes(results)
    if args.roofline and not spec_roofline.is_roofline_line(args.roofline):
        raise SystemExit("the roofline line does not follow lanes/ROOFLINE_REPORTING.md: 'roofline @B=<b>: memory X% (ceiling T tok/s) | compute Y% | transport bw Z% + latency L%   (inputs: ...)'")
    out = Path(args.out_root) / arm.model / args.window / (arm.slug + (f".r{args.repeat}" if args.repeat > 1 else ""))
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps({"label": arm.id, "results": results}, indent=1))
    acceptance = None
    if args.log:
        shutil.copyfile(args.log, out / "residentd.log")
        with open(args.log, errors="replace") as handle:
            acceptance = spec_verify_bench.parse_log(handle)
    record = {"arm": arm.id, **asdict(arm), "window": args.window, "repeat": args.repeat, "firmware": args.firmware,
              "roofline": args.roofline, "stream_sha256": hashes, "acceptance": acceptance,
              "memavailable_min_gib": args.memavailable_min_gib, "admission_id": args.admission_id, "drafter_sha256": args.drafter_sha256,
              "offline_agreement": args.offline_agreement, "drafter_top1_agreement": args.drafter_top1_agreement,
              "perf_holder": args.perf_holder, "sources": dict(item.split("=", 1) for item in (args.source or []))}
    (out / "arm.json").write_text(json.dumps(record, indent=1))
    print(json.dumps({"receipt": str(out), "streams": len(hashes)}))
    return 0


def expected_stream(path: Path, prompt_tokens: int | None):
    if prompt_tokens is None:
        return read_u32(path)
    data = path.read_bytes()
    ids = list(__import__("struct").unpack(f"<{len(data) // 4}I", data))
    if prompt_tokens <= 0 or prompt_tokens >= len(ids):
        raise SystemExit(f"{path}: --expect-prompt-tokens {prompt_tokens} does not fit {len(ids)} ids")
    from spec_offline.streams import Stream
    return Stream(path.stem, tuple(ids), prompt_tokens)


def load_receipts(root: Path) -> list[dict]:
    receipts = []
    for arm_path in sorted(root.rglob("arm.json")):
        record = json.loads(arm_path.read_text())
        results_path = arm_path.parent / "results.json"
        record["_results"] = json.loads(results_path.read_text())["results"] if results_path.exists() else []
        record["_dir"] = str(arm_path.parent)
        receipts.append(record)
    return receipts


def class_rates(results: list[dict]) -> dict[str, float | None]:
    return spec_verify_bench.summarize([entry for entry in results if entry.get("decode_s")])


def ledger_rows(receipts: list[dict]) -> dict:
    ledger = {"rows": [], "no_spec": [], "refused": [], "windows": {}}
    by_window: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for record in receipts:
        by_window[(record["model"], record["window"])].append(record)
    for (model, window), records in sorted(by_window.items()):
        offs = [record for record in records if record["method"] == "off"]
        verdicts = {"window": window, "model": model, "off_runs": len(offs)}
        if not offs:
            for record in records:
                ledger["refused"].append({"arm": record["arm"], "window": window, "reason": "no matched off (no-spec) arm in the same window"})
            ledger["windows"][f"{model}/{window}"] = verdicts
            continue
        reference_hashes: dict[str, str] = {}
        off_conflicts = []
        shared = 0
        for record in offs:
            for key, digest in record["stream_sha256"].items():
                if key in reference_hashes:
                    shared += 1
                    if reference_hashes[key] != digest:
                        off_conflicts.append(key)
                reference_hashes.setdefault(key, digest)
        verdicts["G2_off_determinism"] = ("FAIL: " + ",".join(off_conflicts)) if off_conflicts else ("PASS" if shared else "not evaluated (one off run)")
        reference = dict(offs[0], stream_sha256=reference_hashes)
        reference_rates = class_rates(reference["_results"])
        for record in offs:
            for content_class, rate in class_rates(record["_results"]).items():
                if rate is None:
                    continue
                if not record.get("roofline"):
                    ledger["refused"].append({"arm": record["arm"], "window": window, "class": content_class, "reason": "no roofline line"})
                    continue
                ledger["no_spec"].append({"arm": record["arm"], "window": window, "repeat": record.get("repeat", 1), "class": content_class,
                                          "no_spec_tok_s": round(rate, 2), "roofline": record["roofline"]})
        spec_records = [record for record in records if record["method"] != "off"]
        repeats: dict[str, list[dict]] = defaultdict(list)
        for record in spec_records:
            repeats[record["arm"]].append(record)
        for arm_id, items in sorted(repeats.items()):
            hashes = [item["stream_sha256"] for item in items]
            determinism = all(h == hashes[0] for h in hashes[1:]) if len(items) > 1 else None
            record = items[0]
            mismatches = [key for key, digest in record["stream_sha256"].items() if reference["stream_sha256"].get(key) not in (None, digest)]
            unmatched = [key for key in record["stream_sha256"] if key not in reference["stream_sha256"]]
            exact = not mismatches and not unmatched
            gates = {"G1_exactness": "PASS" if exact else "FAIL", "G1_mismatches": mismatches, "G1_unmatched": unmatched,
                     "G2_determinism": "PASS" if determinism else ("FAIL" if determinism is False else "not evaluated (one run)"),
                     "G3_offline_online": gate_threshold(record.get("offline_agreement"), 0.98),
                     "G5_memory": "not evaluated" if record.get("memavailable_min_gib") is None else ("PASS" if record["memavailable_min_gib"] >= MEMORY_FLOOR_GIB else "FAIL"),
                     "G7_licence": ("PASS" if record.get("admission_id") and record.get("drafter_sha256") else "FAIL") if record["method"] in LICENSED else "not applicable",
                     "G8_drafter_correctness": gate_threshold(record.get("drafter_top1_agreement"), 0.99) if record["method"] in LICENSED else "not applicable"}
            if not record.get("roofline"):
                ledger["refused"].append({"arm": arm_id, "window": window, "reason": "no roofline line", "gates": gates})
                continue
            acceptance = record.get("acceptance") or {}
            positions = acceptance.get("acceptance_per_position")
            rates = class_rates(record["_results"])
            ratios = {}
            for content_class, rate in rates.items():
                base = reference_rates.get(content_class)
                ratios[content_class] = round(rate / base, 4) if rate and base else None
            scored = [ratio for content_class, ratio in ratios.items() if ratio is not None and content_class in SCORED_CLASSES]
            below = [content_class for content_class, ratio in ratios.items() if ratio is not None and ratio < REGRESSION_FLOOR]
            gates["G4_regression_floor"] = "not evaluated" if not scored else ("PASS" if not below else ("adaptive: controller must reach k=0" if record["shape"] == "adaptive" else f"not eligible: {','.join(below)}"))
            for content_class, rate in rates.items():
                base = reference_rates.get(content_class)
                ledger["rows"].append({"arm": arm_id, "window": window, "class": content_class, "runs": len(items),
                                       "spec_tok_s": round(rate, 2) if rate else None, "no_spec_tok_s": round(base, 2) if base else None,
                                       "ratio": ratios[content_class], "tokens_per_round": acceptance.get("tokens_per_round"),
                                       "acceptance_per_position": [item["acceptance"] for item in positions] if positions else None,
                                       "gates": gates, "roofline": record["roofline"], "evidence_only": not exact})
        ledger["windows"][f"{model}/{window}"] = verdicts
    return ledger


def gate_threshold(value, threshold: float) -> str:
    if value is None:
        return "not evaluated"
    return "PASS" if value >= threshold else f"FAIL ({value:.4f} < {threshold})"


def geometric_mean(values: list[tuple[float, float]]) -> float | None:
    if not values:
        return None
    weight_sum = sum(weight for _, weight in values)
    import math
    return math.exp(sum(weight * math.log(value) for value, weight in values) / weight_sum)


def verdicts(ledger: dict) -> dict:
    per_arm: dict[str, list[dict]] = defaultdict(list)
    for row in ledger["rows"]:
        per_arm[row["arm"]].append(row)
    out = {}
    for arm_id, rows in per_arm.items():
        gates = rows[0]["gates"]
        eligible = all(gates[name] == "PASS" for name in ("G1_exactness",)) and gates["G2_determinism"] != "FAIL" and gates["G5_memory"] != "FAIL" \
            and gates["G7_licence"] != "FAIL" and gates["G8_drafter_correctness"] != "FAIL" and not gates["G4_regression_floor"].startswith("not eligible")
        weighted = [(row["ratio"], SCORED_CLASSES[row["class"]]) for row in rows if row["ratio"] and row["class"] in SCORED_CLASSES]
        out[arm_id] = {"eligible": eligible, "geometric_mean_ratio": round(geometric_mean(weighted), 4) if weighted else None,
                       "classes_scored": sorted(row["class"] for row in rows if row["class"] in SCORED_CLASSES), "gates": gates}
    return out


def markdown(ledger: dict, decided: dict) -> str:
    lines = ["| arm | window | class | spec tok/s | no-spec tok/s | ratio | tau | p1..pk | G1 | G2 | G4 |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for row in ledger["rows"]:
        positions = "" if not row["acceptance_per_position"] else " ".join("-" if p is None else f"{p:.2f}" for p in row["acceptance_per_position"])
        lines.append(f"| {row['arm']} | {row['window']} | {row['class']} | {row['spec_tok_s']} | {row['no_spec_tok_s']} | {row['ratio']} | {row['tokens_per_round']} | {positions} | "
                     f"{row['gates']['G1_exactness']} | {row['gates']['G2_determinism']} | {row['gates']['G4_regression_floor']} |")
    lines.append("")
    lines.append("| no-spec arm | window | class | tok/s |")
    lines.append("|---|---|---|---|")
    for row in ledger["no_spec"]:
        lines.append(f"| {row['arm']} r{row['repeat']} | {row['window']} | {row['class']} | {row['no_spec_tok_s']} |")
    lines.append("")
    seen = set()
    for row in ledger["rows"] + ledger["no_spec"]:
        if row["roofline"] not in seen:
            seen.add(row["roofline"])
            lines.append(f"- {row['arm']}: {row['roofline']}")
    lines.append("")
    for arm_id, verdict in decided.items():
        lines.append(f"- {arm_id}: eligible={verdict['eligible']} geometric mean ratio={verdict['geometric_mean_ratio']} over {verdict['classes_scored']}")
    for refused in ledger["refused"]:
        lines.append(f"- REFUSED {refused['arm']} ({refused['window']}): {refused['reason']}")
    return "\n".join(lines) + "\n"


def report(args: argparse.Namespace) -> int:
    receipts = load_receipts(Path(args.receipts))
    if not receipts:
        raise SystemExit(f"no arm.json receipts under {args.receipts}")
    ledger = ledger_rows(receipts)
    decided = verdicts(ledger)
    output = {"ledger": ledger, "verdicts": decided}
    Path(args.out).write_text(json.dumps(output, indent=1))
    text = markdown(ledger, decided)
    if args.markdown:
        Path(args.markdown).write_text(text)
    print(text)
    return 0 if not any(row["gates"]["G1_exactness"] == "FAIL" for row in ledger["rows"]) else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="spec_bakeoff", description="Speculation bake-off harness: arm ids, plans, receipts and the generated ledger with gates G1-G8.")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("arm", help="validate arm ids")
    p.add_argument("arms", nargs="+")
    p.set_defaults(function=lambda a: print("\n".join(json.dumps(asdict(parse_arm(x))) for x in a.arms)) or 0)
    p = sub.add_parser("plan")
    p.add_argument("--model", required=True)
    p.add_argument("--arms", nargs="*", help="method[:drafter] entries, e.g. mtp:l45 lookup suffix")
    p.add_argument("--placements", default="fleet")
    p.add_argument("--shapes", default="chain-k3")
    p.add_argument("--batches", default="1")
    p.add_argument("--classes", default="prose,code,repetitive,chat,long,thinking,tool_json,chinese")
    p.add_argument("--out", required=True)
    p.set_defaults(function=plan)
    p = sub.add_parser("ingest")
    p.add_argument("--arm", required=True)
    p.add_argument("--window", required=True)
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--run", help="spec_verify_bench run JSON (token ids required)")
    p.add_argument("--replay-json", help="spec_verify_bench replay JSON")
    p.add_argument("--expect", help="the recorded .u32 stream the replay was checked against")
    p.add_argument("--expect-prompt-tokens", type=int, help="prompt token count when --expect is a headerless spec_verify_bench record file")
    p.add_argument("--anchor-class", default="anchor")
    p.add_argument("--log", help="residentd log with VERIFY-FRAME / VERIFY-POSITIONS")
    p.add_argument("--roofline", help="the roofline line for this arm's tok/s")
    p.add_argument("--firmware")
    p.add_argument("--memavailable-min-gib", type=float)
    p.add_argument("--admission-id")
    p.add_argument("--drafter-sha256")
    p.add_argument("--offline-agreement", type=float)
    p.add_argument("--drafter-top1-agreement", type=float)
    p.add_argument("--perf-holder")
    p.add_argument("--source", action="append", help="name=sha256 identity lines")
    p.add_argument("--out-root", required=True)
    p.set_defaults(function=ingest)
    p = sub.add_parser("report")
    p.add_argument("--receipts", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--markdown")
    p.set_defaults(function=report)
    args = parser.parse_args(argv)
    return args.function(args)


if __name__ == "__main__":
    sys.exit(main())
