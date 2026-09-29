#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spec_offline.acceptance import DEPTH_MAX, agreement, position_table, round_replay  # noqa: E402
from spec_offline.cost import fit_verify_cost, host_overhead_ms, verify_ms  # noqa: E402
from spec_offline.drafters import make_drafter  # noqa: E402
from spec_offline.streams import class_of, read_u32  # noqa: E402
from spec_offline.tapdump import read_dump  # noqa: E402
from spec_offline.tree import parse_shape  # noqa: E402
import spec_roofline  # noqa: E402


def model_vocab(args: argparse.Namespace) -> int:
    if args.vocab:
        return args.vocab
    vocab = spec_roofline.load_registry()["models"].get(args.model, {}).get("vocab")
    if not vocab:
        raise SystemExit(f"model {args.model} has no vocab in the registry; give --vocab")
    return vocab


def load_streams(args: argparse.Namespace):
    if args.dump:
        manifest, dump_streams = read_dump(Path(args.dump), want_taps=True)
        return manifest, [(entry.stream, entry.content_class, entry.taps) for entry in dump_streams], model_vocab(args)
    if args.streams:
        files = sorted(Path(args.streams).glob("*.u32"))
        if not files:
            raise SystemExit(f"no .u32 streams in {args.streams}")
        return {"model": args.model, "format": "u32-directory"}, [(read_u32(path), class_of(path.stem), None) for path in files], model_vocab(args)
    raise SystemExit("give --dump <tapdump dir> or --streams <dir of .u32>")


def accumulate(tables: dict[str, list], per_class: dict) -> dict:
    out = {}
    for name, items in tables.items():
        reached = [0] * DEPTH_MAX
        accepted = [0] * DEPTH_MAX
        positions = agree = 0
        for table in items:
            positions += table.positions
            agree += table.top1_agree
            for index in range(DEPTH_MAX):
                reached[index] += table.reached[index]
                accepted[index] += table.accepted[index]
        rates = [accepted[i] / reached[i] if reached[i] else None for i in range(DEPTH_MAX)]
        tau = {}
        total = reach = 1.0
        for k in range(1, DEPTH_MAX + 1):
            if rates[k - 1] is None:
                break
            reach *= rates[k - 1]
            total += reach
            tau[str(k)] = round(total, 4)
        out[name] = {"positions": positions, "reached": reached, "accepted": accepted, "acceptance_per_position": rates,
                     "top1_agreement": round(agree / positions, 4) if positions else None, "tau": tau}
    return out


def acceptance_command(args: argparse.Namespace) -> int:
    manifest, streams, vocab = load_streams(args)
    report = {"model": args.model, "source": manifest, "depth": DEPTH_MAX, "drafters": {}}
    for spec in args.drafters.split(","):
        drafter = make_drafter(spec, vocab=vocab)
        per_class: dict[str, list] = defaultdict(list)
        per_stream = {}
        for stream, content_class, taps in streams:
            table = position_table(stream, drafter, taps, DEPTH_MAX)
            per_class[content_class].append(table)
            per_class["all"].append(table)
            per_stream[stream.name] = table.as_dict()
        report["drafters"][spec] = {"classes": accumulate(per_class, {}), "streams": per_stream}
    Path(args.out).write_text(json.dumps(report, indent=1))
    for spec, entry in report["drafters"].items():
        for name, cell in sorted(entry["classes"].items()):
            rates = ", ".join("n/a" if rate is None else f"{rate:.3f}" for rate in cell["acceptance_per_position"])
            print(f"{spec:24s} {name:12s} positions {cell['positions']:6d} p1..p7 {rates} tau7 {cell['tau'].get('7', 'n/a')}")
    return 0


def rounds_command(args: argparse.Namespace) -> int:
    manifest, streams, vocab = load_streams(args)
    shapes = [parse_shape(text) for text in args.shapes.split(",")]
    report = {"model": args.model, "source": manifest, "shapes": {}}
    for shape in shapes:
        cells: dict[str, dict] = {}
        members = None
        drafter_spec = args.drafter
        if shape.kind == "trie":
            members = [make_drafter(member, vocab=vocab) for member in shape.members]
            drafter_spec = "+".join(shape.members)
        drafter = make_drafter(args.drafter if shape.kind != "trie" else shape.members[0], vocab=vocab)
        per_class: dict[str, list] = defaultdict(list)
        for stream, content_class, taps in streams:
            result = round_replay(stream, drafter, taps, shape, members)
            if not result.stream_exact:
                raise SystemExit(f"{stream.name}: the committed stream differs from the no-spec stream under {shape.text}; the replayer is broken")
            per_class[content_class].append(result)
            per_class["all"].append(result)
        for name, items in per_class.items():
            rounds = sum(item.rounds for item in items)
            committed = sum(item.committed for item in items)
            rows = sum(item.proposed_rows for item in items)
            cells[name] = {"rounds": rounds, "committed": committed, "rows_per_round": round(rows / rounds, 3) if rounds else None,
                           "tokens_per_round": round(committed / rounds, 4) if rounds else None, "stream_exact": all(item.stream_exact for item in items)}
        report["shapes"][shape.text] = {"drafter": drafter_spec, "classes": cells}
        for name, cell in sorted(cells.items()):
            print(f"{shape.text:22s} {drafter_spec:20s} {name:12s} rounds {cell['rounds']:6d} tau {cell['tokens_per_round']} rows/round {cell['rows_per_round']} exact {cell['stream_exact']}")
    Path(args.out).write_text(json.dumps(report, indent=1))
    return 0


def agreement_command(args: argparse.Namespace) -> int:
    manifest, streams, vocab = load_streams(args)
    candidate = make_drafter(args.candidate, vocab=vocab)
    reference = make_drafter(args.reference, vocab=vocab)
    agree = total = 0
    for stream, _, taps in streams:
        result = agreement(stream, candidate, reference, taps)
        agree += result["agree"]
        total += result["positions"]
    rate = agree / total if total else None
    verdict = "PASS" if total >= args.min_positions and rate is not None and rate >= args.threshold else "FAIL"
    report = {"candidate": args.candidate, "reference": args.reference, "positions": total, "agree": agree, "agreement": rate,
              "threshold": args.threshold, "min_positions": args.min_positions, "gate": "G8", "verdict": verdict}
    Path(args.out).write_text(json.dumps(report, indent=1))
    print(f"G8 {verdict}: {args.candidate} agrees with {args.reference} on {agree}/{total} positions ({'n/a' if rate is None else f'{rate:.4f}'}; need >= {args.threshold} on >= {args.min_positions})")
    return 0 if verdict == "PASS" else 1


def cost_fit_command(args: argparse.Namespace) -> int:
    arms = {}
    for item in args.arm_logs:
        name, _, path = item.partition("=")
        if not path:
            raise SystemExit("--arm-logs takes name=<dir of residentd.rank*.log>")
        arms[name] = Path(path)
    extra = []
    for item in args.extra_point or []:
        rows, ms, label = item.split(":", 2)
        extra.append({"rows": int(rows), "ms": float(ms), "label": label})
    model = fit_verify_cost(arms, args.min_samples, extra)
    Path(args.out).write_text(json.dumps(model, indent=1))
    print(f"T(rows) = {model['fit']['intercept_ms']} + {model['fit']['per_row_ms']} x (rows - 1) ms; {model['unit']}")
    for rows, summary in model["samples"].items():
        print(f"rows {rows}: n {summary['n']} engine median {summary['engine_median_ms']} ms (pooled {summary['pooled_median_ms']}, p10 {summary['p10_ms']}, p90 {summary['p90_ms']}) {'measured' if summary['measured'] else 'too few samples'} residual {model['residual_ms'][rows]}")
    return 0


def predict_command(args: argparse.Namespace) -> int:
    acceptance = json.loads(Path(args.acceptance).read_text())
    cost = json.loads(Path(args.cost).read_text())
    registry = spec_roofline.load_registry()
    if args.model not in registry["models"]:
        raise SystemExit(f"unknown model {args.model}; known: {sorted(registry['models'])}")
    model = registry["models"][args.model]
    base_ms, base_source = verify_ms(cost, 0, args.batch)
    no_spec_ms = base_ms + host_overhead_ms(0, args.overhead_ms, args.overhead_row_ms)
    no_spec_tok_s = 1000.0 / no_spec_ms
    report = {"model": args.model, "batch": args.batch, "kind": "[M] predicted from T0 acceptance and T1 verify cost; no fleet run",
              "no_spec": {"step_ms": round(no_spec_ms, 3), "tok_s": round(no_spec_tok_s, 2), "verify_source": base_source,
                          "roofline": spec_roofline.roofline(registry, args.model, args.batch, 1, no_spec_ms, 1.0, model.get("b1_collective_ms"))["line"]},
              "drafters": {}}
    print(f"no-spec [M]: {no_spec_tok_s:.1f} tok/s at {no_spec_ms:.1f} ms/token ({base_source})")
    print(report["no_spec"]["roofline"])
    for spec, entry in acceptance["drafters"].items():
        rows_out = {}
        for content_class, cell in entry["classes"].items():
            best = None
            depths = {}
            for k_text, tau in cell["tau"].items():
                k = int(k_text)
                wave_ms, source = verify_ms(cost, k, args.batch)
                draft_ms = args.draft_ms + args.draft_token_ms * k
                round_ms = wave_ms + host_overhead_ms(k, args.overhead_ms, args.overhead_row_ms) + draft_ms
                tok_s = 1000.0 * tau / round_ms
                line = spec_roofline.roofline(registry, args.model, args.batch, k + 1, round_ms, tau, None)["line"]
                depths[k_text] = {"tau": tau, "verify_ms": round(wave_ms, 3), "verify_source": source, "draft_ms": round(draft_ms, 3),
                                  "round_ms": round(round_ms, 3), "spec_tok_s": round(tok_s, 2), "ratio_to_no_spec": round(tok_s / no_spec_tok_s, 3), "roofline": line}
                if best is None or tok_s > best[1]:
                    best = (k, tok_s)
            rows_out[content_class] = {"depths": depths, "best_k": best[0] if best else None,
                                       "best_spec_tok_s": round(best[1], 2) if best else None, "no_spec_tok_s": round(no_spec_tok_s, 2)}
            if best:
                print(f"{spec:24s} {content_class:12s} best k={best[0]} spec {best[1]:.1f} tok/s vs no-spec {no_spec_tok_s:.1f} ({best[1] / no_spec_tok_s:.2f}x) [M]")
                print("  " + depths[str(best[0])]["roofline"])
        report["drafters"][spec] = rows_out
    Path(args.out).write_text(json.dumps(report, indent=1))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="spec_offline.t0", description="T0 offline speculation replay: acceptance per position, round replays for chains, trees and tries, G8 drafter agreement, the T(rows) verify-cost fit and predicted tok/s.")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, function in (("acceptance", acceptance_command), ("rounds", rounds_command), ("agreement", agreement_command)):
        p = sub.add_parser(name)
        p.add_argument("--dump", help="tap dump directory (manifest.json + SHA256SUMS)")
        p.add_argument("--streams", help="directory of .u32 no-spec streams (token-only drafters)")
        p.add_argument("--model", default="glmflash")
        p.add_argument("--vocab", type=int, help="override the model registry vocabulary (synthetic dumps)")
        p.add_argument("--out", required=True)
        p.set_defaults(function=function)
        if name == "acceptance":
            p.add_argument("--drafters", default="oracle,adversary,lookup,suffix,ngram3")
        if name == "rounds":
            p.add_argument("--drafter", default="lookup")
            p.add_argument("--shapes", default="chain-k1,chain-k3,chain-k7")
        if name == "agreement":
            p.add_argument("--candidate", required=True)
            p.add_argument("--reference", required=True)
            p.add_argument("--threshold", type=float, default=0.99)
            p.add_argument("--min-positions", type=int, default=1000)
    p = sub.add_parser("cost-fit")
    p.add_argument("--arm-logs", nargs="+", required=True, help="name=<dir> with residentd.rank*.log of oracle/adversary/mtp arms")
    p.add_argument("--extra-point", action="append", help="rows:ms:label, e.g. a B8 no-spec step time")
    p.add_argument("--min-samples", type=int, default=16)
    p.add_argument("--out", required=True)
    p.set_defaults(function=cost_fit_command)
    p = sub.add_parser("predict")
    p.add_argument("--acceptance", required=True)
    p.add_argument("--cost", required=True)
    p.add_argument("--model", default="glmflash")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--overhead-ms", type=float, default=2.0, help="host overhead per round at k=0 [M, design S0]")
    p.add_argument("--overhead-row-ms", type=float, default=0.575, help="host overhead per verify row [M, design S0]")
    p.add_argument("--draft-ms", type=float, default=0.0, help="draft cost per round on the critical path")
    p.add_argument("--draft-token-ms", type=float, default=0.0, help="draft cost per drafted token on the critical path")
    p.add_argument("--out", required=True)
    p.set_defaults(function=predict_command)
    args = parser.parse_args(argv)
    return args.function(args)


if __name__ == "__main__":
    sys.exit(main())
