#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REGISTRY = Path(__file__).resolve().parent / "spec_offline" / "models.json"
LINE = re.compile(r"^roofline @B=(\d+)(?:,rows=\d+)?: memory (\d+(?:\.\d+)?)% \(ceiling (\d+(?:\.\d+)?) tok/s\) \| compute (\d+(?:\.\d+)?)% \| transport (?:bw (\d+(?:\.\d+)?)% \+ latency (\d+(?:\.\d+)?)%|not separated)\s+\(inputs: .+\)$")


def load_registry(path: Path = REGISTRY) -> dict:
    return json.loads(path.read_text())


def expert_union(rows: int, expert_count: int, experts_per_token: int) -> float:
    return expert_count * (1.0 - (1.0 - experts_per_token / expert_count) ** rows)


def ideal_bytes_gb(model: dict, rows: int, batch: int = 1) -> float:
    union = expert_union(rows, model["expert_count"], model["experts_per_token"])
    return model["spine_gb"] + model["experts_gb"] * union / model["expert_count"] + model.get("sequence_bytes_gb", 0.0) * batch


def roofline(registry: dict, model_name: str, batch: int, rows: int, step_ms: float, tokens_per_step: float,
             collective_ms: float | None = None, collective_rounds: int | None = None, step_label: str = "end-to-end fleet per token") -> dict:
    model = registry["models"][model_name]
    peak = registry["peak_read_gb_s"]
    total_rows = rows * batch
    bytes_gb = ideal_bytes_gb(model, total_rows, batch)
    achieved_gb_s = bytes_gb / (step_ms / 1000.0)
    memory_pct = 100.0 * achieved_gb_s / peak
    ceiling_tok_s = peak / bytes_gb * tokens_per_step
    flops = bytes_gb * 1e9 * model["flop_per_active_byte"] * total_rows
    compute_pct = 100.0 * (flops / (step_ms / 1000.0)) / (model["compute_peak_tflops"] * 1e12)
    rounds = collective_rounds if collective_rounds is not None else model["collective_rounds_per_step"]
    inputs = [f"{bytes_gb:.2f} GB ideal per step at {total_rows} rows ({model['spine_gb']} GB spine + expert union of {expert_union(total_rows, model['expert_count'], model['experts_per_token']):.1f}/{model['expert_count']} + {model.get('sequence_bytes_gb', 0.0)} GB KV/state/head per sequence x {batch})",
              f"{step_ms:.2f} ms per step", f"{tokens_per_step:.3f} tokens per step", f"{peak:.0f} GB/s read peak",
              f"compute peak {model['compute_peak_tflops']:.0f} TFLOPS ({model['compute_peak_source']})", step_label]
    result = {"batch": batch, "rows": total_rows, "memory_pct": round(memory_pct, 1), "ceiling_tok_s": round(ceiling_tok_s, 1),
              "compute_pct": round(compute_pct, 2), "ideal_bytes_gb": round(bytes_gb, 3), "step_ms": step_ms, "tokens_per_step": tokens_per_step}
    if collective_ms is not None and collective_ms > 0.0:
        topology_gb = model["topology_mb_per_row"] * total_rows / 1000.0
        bw_pct = 100.0 * (topology_gb / registry["link_gb_s"] * 1000.0) / collective_ms
        latency_pct = 100.0 * (rounds * registry["hop_us"] / 1000.0) / collective_ms
        result.update({"transport_bw_pct": round(bw_pct, 1), "transport_latency_pct": round(latency_pct, 1), "collective_ms": collective_ms, "collective_rounds": rounds})
        transport = f"transport bw {bw_pct:.1f}% + latency {latency_pct:.1f}%"
        inputs.append(f"{collective_ms:.2f} ms in collectives over {rounds} rounds x {registry['hop_us']:.0f} us, {model['topology_mb_per_row'] * total_rows:.1f} MB topology bytes at {registry['link_gb_s']:.0f} GB/s")
    else:
        transport = "transport not separated"
        inputs.append("collective time not measured for this step")
    rows_text = f",rows={total_rows}" if total_rows != batch else ""
    result["line"] = (f"roofline @B={batch}{rows_text}: memory {memory_pct:.0f}% (ceiling {ceiling_tok_s:.0f} tok/s) | compute {compute_pct:.1f}% | "
                      f"{transport}   (inputs: {'; '.join(inputs)})")
    return result


def is_roofline_line(text: str) -> bool:
    return bool(LINE.match(text.strip()))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="spec_roofline", description="Emit the roofline line for a tok/s or step-time number (lanes/ROOFLINE_REPORTING.md).")
    parser.add_argument("--registry", default=str(REGISTRY))
    parser.add_argument("--model", required=True)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--rows", type=int, default=1, help="verify rows per sequence (1 = plain decode)")
    parser.add_argument("--step-ms", type=float, help="measured step time; give this or --tok-s")
    parser.add_argument("--tok-s", type=float, help="measured tok/s per sequence; the step time is tokens_per_step / tok/s")
    parser.add_argument("--tokens-per-step", type=float, default=1.0, help="committed tokens per step per sequence (tau for a verify wave)")
    parser.add_argument("--collective-ms", type=float)
    parser.add_argument("--collective-rounds", type=int)
    parser.add_argument("--step-label", default="end-to-end fleet per token")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if (args.step_ms is None) == (args.tok_s is None):
        raise SystemExit("give exactly one of --step-ms or --tok-s")
    registry = load_registry(Path(args.registry))
    if args.model not in registry["models"]:
        raise SystemExit(f"unknown model {args.model}; known: {sorted(registry['models'])}")
    step_ms = args.step_ms if args.step_ms is not None else 1000.0 * args.tokens_per_step / args.tok_s
    result = roofline(registry, args.model, args.batch, args.rows, step_ms, args.tokens_per_step, args.collective_ms, args.collective_rounds, args.step_label)
    print(json.dumps(result, indent=1) if args.json else result["line"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
