#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import struct
import subprocess
import sys
from pathlib import Path

DEFAULT_CONFIGS = [
    ("lookup", "lookup", 8),
    ("mtp p=0.50", "synthetic:500", 4),
    ("mtp p=0.50", "synthetic:500", 8),
    ("mtp p=0.65", "synthetic:650", 4),
    ("mtp p=0.65", "synthetic:650", 8),
    ("mtp p=0.80", "synthetic:800", 4),
    ("mtp p=0.80", "synthetic:800", 8),
    ("mtp+lookup p=0.65", "lookup+synthetic:650", 8),
    ("mtp+lookup p=0.80", "lookup+synthetic:800", 8),
]


def write_stream(path: Path, prompt: list[int], output: list[int]) -> None:
    tokens = prompt + output
    path.write_bytes(struct.pack(f"<II{len(tokens)}I", len(prompt), len(tokens), *tokens))


def streams(args: argparse.Namespace) -> int:
    run = json.loads(Path(args.run).read_text())
    prompts = json.loads(Path(args.prompt_ids).read_text())
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for entry in run["results"]:
        key = f"{entry['class']}[{entry['index']}]"
        if key not in prompts:
            raise SystemExit(f"{args.prompt_ids} has no prompt ids for {key}")
        if not entry.get("token_ids"):
            raise SystemExit(f"{key} has no token ids")
        path = out / f"{entry['class']}_{entry['index']}.u32"
        write_stream(path, [int(t) for t in prompts[key]], [int(t) for t in entry["token_ids"]])
        written.append(str(path))
    print(json.dumps({"streams": written}))
    return 0


def replay(binary: str, drafter: str, rows: int, frame: int, fixed_depth: bool, files: list[str]) -> list[dict]:
    command = [binary, "--rows", str(rows), "--frame", str(frame), *(["--fixed-depth"] if fixed_depth else []), "--drafter", drafter, "--", *files]
    output = subprocess.run(command, check=True, capture_output=True, text=True).stdout
    return [json.loads(line) for line in output.splitlines() if line.strip()]


def price(result: dict, model: dict) -> dict:
    decode_tokens = result["generated"] - 1
    verify_ms = 0.0
    for rows, count in enumerate(result["rows"]):
        if count:
            verify_ms += count * (model["b1_ms"] + (rows - 1) * model["row_ms"])
    plain_ms = (result["plain_steps"] + result["plain_frame_tokens"]) * model["b1_ms"]
    round_ms = result["rounds"] * model["round_ms"]
    draft_ms = result["synthetic_calls"] * model["mtp_call_ms"] + result["synthetic"][1] * model["mtp_token_ms"]
    total_ms = verify_ms + plain_ms + round_ms + draft_ms
    return {"decode_tokens": decode_tokens, "ms": total_ms}


def estimate(args: argparse.Namespace) -> int:
    files = sorted(str(p) for p in Path(args.streams).glob("*.u32"))
    if not files:
        raise SystemExit(f"no .u32 streams in {args.streams}")
    model = {"b1_ms": args.b1_ms, "row_ms": args.row_ms, "round_ms": args.round_ms,
             "mtp_token_ms": args.mtp_token_ms, "mtp_call_ms": args.mtp_call_ms}
    classes = sorted({Path(f).name.rsplit("_", 1)[0] for f in files})
    report = {"model": model, "frame": args.frame, "fixed_depth": args.fixed_depth, "spec_off_tok_s": 1000.0 / args.b1_ms, "configs": []}
    for label, drafter, rows in DEFAULT_CONFIGS:
        results = replay(args.replay, drafter, rows, args.frame, args.fixed_depth, files)
        per_class = {}
        for name in classes:
            chosen = [r for r in results if Path(r["file"]).name.startswith(name + "_")]
            tokens = sum(price(r, model)["decode_tokens"] for r in chosen)
            ms = sum(price(r, model)["ms"] for r in chosen)
            rounds = sum(r["rounds"] for r in chosen)
            accepted = sum(r["accepted"] for r in chosen)
            proposed = sum(r["proposed"] for r in chosen)
            plain = sum(r["plain_steps"] + r["plain_frame_tokens"] for r in chosen)
            per_class[name] = {
                "spec_tok_s": round(tokens / ms * 1000.0, 1) if ms else None,
                "speedup": round(tokens / ms * args.b1_ms, 2) if ms else None,
                "tokens_per_round": round((tokens - plain) / rounds, 2) if rounds else 0.0,
                "acceptance": round(accepted / proposed, 3) if proposed else 0.0,
                "rounds": rounds, "plain_tokens": plain, "decode_tokens": tokens,
            }
        report["configs"].append({"label": label, "drafter": drafter, "rows": rows, "classes": per_class})
    Path(args.out).write_text(json.dumps(report, indent=1))
    print(f"spec off: {report['spec_off_tok_s']:.1f} tok/s (B1 step {args.b1_ms} ms); frame {args.frame}; "
          f"depth {'fixed' if args.fixed_depth else 'acceptance cap'}; "
          f"row {args.row_ms} ms, round {args.round_ms} ms, MTP {args.mtp_token_ms} ms/token + {args.mtp_call_ms} ms/call")
    header = "config".ljust(22) + "rows " + "".join(name.rjust(26) for name in classes)
    print(header)
    for config in report["configs"]:
        cells = []
        for name in classes:
            cell = config["classes"][name]
            cells.append(f"{cell['spec_tok_s']} ({cell['speedup']}x, tau {cell['tokens_per_round']})".rjust(26))
        print(config["label"].ljust(22) + str(config["rows"]).rjust(4) + " " + "".join(cells))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="glm5_next_spec_estimate", description=(
        "Replay real greedy GLM-5.3 Flash streams through the module's drafters and verify-depth rules "
        "and price the rounds with a per-row cost model. Lookup acceptance is measured on the streams; "
        "MTP acceptance is the synthetic per-token parameter p until the fleet logs VERIFY-MTP counters."))
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("streams")
    p.add_argument("--run", required=True)
    p.add_argument("--prompt-ids", required=True, help="JSON object: 'class[index]' -> prompt token ids")
    p.add_argument("--out-dir", required=True)
    p.set_defaults(function=streams)
    p = sub.add_parser("estimate")
    p.add_argument("--streams", required=True)
    p.add_argument("--replay", required=True, help="the built tools/glm5_next_spec_replay binary")
    p.add_argument("--frame", type=int, default=8)
    p.add_argument("--fixed-depth", action="store_true", help="draft rows - 1 every round instead of the acceptance depth cap")
    p.add_argument("--b1-ms", type=float, default=25.2)
    p.add_argument("--row-ms", type=float, default=2.7)
    p.add_argument("--round-ms", type=float, default=1.0)
    p.add_argument("--mtp-token-ms", type=float, default=1.3)
    p.add_argument("--mtp-call-ms", type=float, default=0.3)
    p.add_argument("--out", required=True)
    p.set_defaults(function=estimate)
    args = parser.parse_args(argv)
    return args.function(args)


if __name__ == "__main__":
    sys.exit(main())
